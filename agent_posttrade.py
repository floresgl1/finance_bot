"""
agent_posttrade.py — Weekly post-trade analysis agent.

Reads evaluated ENTRY rows from signal_log.csv, computes four summary
tables (per-ticker, per-feature, toxic combos, confidence calibration),
sends them to the LLM for pattern detection, and posts findings to Discord.

This is a read-only analytical agent — it never modifies signal_log.csv,
adjusts thresholds, or blocks trades.  It surfaces patterns for a human
to investigate.

Usage:
    python agent_posttrade.py                          # console report
    python agent_posttrade.py --discord                # also post to Discord
    python agent_posttrade.py --log path/to/signal_log.csv
    python agent_posttrade.py --window 30              # lookback days
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from groq import Groq

from agent_posttrade_prompts import (
    ALLOWED_CATEGORIES,
    ALLOWED_SEVERITIES,
    RETRY_MESSAGE_TEMPLATE,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
)
from signal_logger import SIGNAL_LOG_PATH

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_WINDOW_DAYS = 30
PARSE_RETRY_BUDGET = 1  # one retry, two attempts max

# Confidence buckets (0–1 scale) for Table D
CONFIDENCE_BINS = [0.0, 0.35, 0.40, 0.50, 0.65, 1.01]
CONFIDENCE_LABELS = ["<0.35", "0.35–0.40", "0.40–0.50", "0.50–0.65", "0.65+"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_evaluated_entries(
    path: str = SIGNAL_LOG_PATH,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> pd.DataFrame:
    """Load ENTRY rows with evaluated results within the lookback window.

    Returns a DataFrame with columns: date, ticker, model_signal, confidence,
    result, price, outcome_price, shap_driver_1, shap_driver_2, shap_driver_3,
    actual_action.  Only rows where result is WIN, LOSS, or NEUTRAL are
    included (SKIPPED and blank rows are excluded).
    """
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, dtype=str)

    if "row_type" not in df.columns or "result" not in df.columns:
        return pd.DataFrame()

    # Filter to evaluated ENTRY rows with BUY/SELL signals
    mask = (
        (df["row_type"].fillna("ENTRY") == "ENTRY")
        & (df["model_signal"].isin(["BUY", "SELL"]))
        & (df["result"].isin(["WIN", "LOSS", "NEUTRAL"]))
    )
    entries = df[mask].copy()

    if entries.empty:
        return entries

    # Parse dates and apply window
    entries["date"] = pd.to_datetime(entries["date"], errors="coerce")
    entries = entries.dropna(subset=["date"])

    # CSV dates are tz-naive; compare with a tz-naive cutoff to avoid
    # "Cannot compare tz-naive and tz-aware" errors from pandas.
    cutoff = pd.Timestamp.now("UTC").tz_localize(None) - timedelta(days=window_days)
    entries = entries[entries["date"] >= cutoff]

    # Parse numeric columns
    for col in ("confidence", "price", "outcome_price"):
        entries[col] = pd.to_numeric(entries.get(col), errors="coerce")

    # Fill missing SHAP drivers
    for col in ("shap_driver_1", "shap_driver_2", "shap_driver_3"):
        entries[col] = entries[col].fillna("")

    return entries.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------
def _return_pct(row: pd.Series) -> float | None:
    """Compute return % from price and outcome_price."""
    price = row.get("price")
    outcome = row.get("outcome_price")
    if pd.isna(price) or pd.isna(outcome) or price == 0:
        return None
    return (outcome - price) / price * 100


def build_table_a(df: pd.DataFrame) -> str:
    """Per-ticker scorecard."""
    if df.empty:
        return "(no data)"

    rows = []
    for (ticker, signal), group in df.groupby(["ticker", "model_signal"]):
        decided = group[group["result"].isin(["WIN", "LOSS"])]
        wins = int((decided["result"] == "WIN").sum())
        losses = int((decided["result"] == "LOSS").sum())
        neutrals = int((group["result"] == "NEUTRAL").sum())
        total_decided = wins + losses
        win_rate = (wins / total_decided * 100) if total_decided > 0 else 0.0

        returns = group.apply(_return_pct, axis=1).dropna()
        avg_return = float(returns.mean()) if len(returns) > 0 else 0.0

        # Average confidence on wins vs losses
        win_conf = group[group["result"] == "WIN"]["confidence"].mean()
        loss_conf = group[group["result"] == "LOSS"]["confidence"].mean()

        rows.append({
            "ticker": ticker,
            "signal": signal,
            "trades": len(group),
            "wins": wins,
            "losses": losses,
            "neutrals": neutrals,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "avg_conf_wins": f"{win_conf:.2f}" if pd.notna(win_conf) else "n/a",
            "avg_conf_losses": f"{loss_conf:.2f}" if pd.notna(loss_conf) else "n/a",
        })

    if not rows:
        return "(no data)"

    result = pd.DataFrame(rows)
    return result.to_string(index=False)


def build_table_b(df: pd.DataFrame) -> str:
    """Feature-outcome cross — aggregated across all tickers."""
    if df.empty:
        return "(no data)"

    # Melt SHAP drivers into one row per (signal, feature)
    records = []
    for _, row in df.iterrows():
        for col in ("shap_driver_1", "shap_driver_2", "shap_driver_3"):
            feature = row.get(col, "")
            if feature and isinstance(feature, str) and feature.strip():
                records.append({
                    "feature": feature.strip(),
                    "signal": row["model_signal"],
                    "result": row["result"],
                })

    if not records:
        return "(no data)"

    melted = pd.DataFrame(records)
    rows = []
    for (feature, signal), group in melted.groupby(["feature", "signal"]):
        decided = group[group["result"].isin(["WIN", "LOSS"])]
        wins = int((decided["result"] == "WIN").sum())
        losses = int((decided["result"] == "LOSS").sum())
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0.0
        rows.append({
            "feature": feature,
            "as_driver_of": signal,
            "trades": len(group),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
        })

    result = pd.DataFrame(rows).sort_values("win_rate")
    return result.to_string(index=False)


def build_table_c(df: pd.DataFrame) -> str:
    """Toxic combos — (ticker × feature × direction) with win_rate ≤ 33% and ≥ 2 decided trades."""
    if df.empty:
        return "(none found)"

    records = []
    for _, row in df.iterrows():
        for col in ("shap_driver_1", "shap_driver_2", "shap_driver_3"):
            feature = row.get(col, "")
            if feature and isinstance(feature, str) and feature.strip():
                records.append({
                    "ticker": row["ticker"],
                    "feature": feature.strip(),
                    "signal": row["model_signal"],
                    "result": row["result"],
                })

    if not records:
        return "(none found)"

    melted = pd.DataFrame(records)
    rows = []
    for (ticker, feature, signal), group in melted.groupby(
        ["ticker", "feature", "signal"]
    ):
        decided = group[group["result"].isin(["WIN", "LOSS"])]
        wins = int((decided["result"] == "WIN").sum())
        losses = int((decided["result"] == "LOSS").sum())
        total = wins + losses
        if total < 2:
            continue
        win_rate = (wins / total * 100) if total > 0 else 0.0
        if win_rate > 33.3:
            continue
        rows.append({
            "ticker": ticker,
            "feature": feature,
            "signal": signal,
            "trades": len(group),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
        })

    if not rows:
        return "(none found)"

    result = pd.DataFrame(rows).sort_values("win_rate")
    return result.to_string(index=False)


def build_table_d(df: pd.DataFrame) -> str:
    """Confidence calibration — bucketed confidence vs actual win rate."""
    if df.empty:
        return "(no data)"

    decided = df[df["result"].isin(["WIN", "LOSS"])].copy()
    if decided.empty:
        return "(no decided trades)"

    decided["conf_bucket"] = pd.cut(
        decided["confidence"],
        bins=CONFIDENCE_BINS,
        labels=CONFIDENCE_LABELS,
        right=False,
    )

    rows = []
    for bucket, group in decided.groupby("conf_bucket", observed=True):
        wins = int((group["result"] == "WIN").sum())
        losses = int((group["result"] == "LOSS").sum())
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0.0
        rows.append({
            "conf_bucket": str(bucket),
            "trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
        })

    if not rows:
        return "(no decided trades)"

    result = pd.DataFrame(rows)
    return result.to_string(index=False)


def compute_overall_win_rate(df: pd.DataFrame) -> float:
    """Portfolio-wide win rate excluding NEUTRALs."""
    if df.empty or "result" not in df.columns:
        return 0.0
    decided = df[df["result"].isin(["WIN", "LOSS"])]
    if decided.empty:
        return 0.0
    wins = (decided["result"] == "WIN").sum()
    return float(wins / len(decided) * 100)


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------
def _parse_findings(content: str) -> dict[str, Any]:
    """Parse and validate the LLM's JSON response."""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {e}")

    if not isinstance(obj, dict):
        raise ValueError("response must be a JSON object")

    if "findings" not in obj:
        raise ValueError("response must have a 'findings' key")

    if not isinstance(obj["findings"], list):
        raise ValueError("'findings' must be a list")

    for i, finding in enumerate(obj["findings"]):
        if not isinstance(finding, dict):
            raise ValueError(f"findings[{i}] must be a dict")

        required = {"category", "severity", "pattern", "evidence", "suggested_action"}
        missing = required - set(finding.keys())
        if missing:
            raise ValueError(f"findings[{i}] missing keys: {missing}")

        if finding["category"] not in ALLOWED_CATEGORIES:
            raise ValueError(
                f"findings[{i}].category must be one of {sorted(ALLOWED_CATEGORIES)}; "
                f"got {finding['category']!r}"
            )
        if finding["severity"] not in ALLOWED_SEVERITIES:
            raise ValueError(
                f"findings[{i}].severity must be one of {sorted(ALLOWED_SEVERITIES)}; "
                f"got {finding['severity']!r}"
            )

    if "summary" not in obj or not isinstance(obj["summary"], str):
        raise ValueError("response must have a 'summary' string")

    return obj


def run_analysis(tables_prompt: str) -> dict[str, Any]:
    """Send the tables to the LLM and parse the structured response.

    Returns the validated findings dict.  Raises on catastrophic failure.
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": tables_prompt},
    ]

    parse_retry_budget = PARSE_RETRY_BUDGET

    for _attempt in range(2 + PARSE_RETRY_BUDGET):
        try:
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
            )
        except Exception as e:
            raise RuntimeError(f"Groq API failed: {type(e).__name__}: {e}") from e

        content = completion.choices[0].message.content
        messages.append({"role": "assistant", "content": content})

        try:
            return _parse_findings(content)
        except ValueError as parse_err:
            if parse_retry_budget > 0:
                parse_retry_budget -= 1
                messages.append({
                    "role": "user",
                    "content": RETRY_MESSAGE_TEMPLATE.format(parse_error=str(parse_err)),
                })
                continue
            raise RuntimeError(
                f"LLM output parse failed after retries: {parse_err}"
            ) from parse_err

    raise RuntimeError("analysis loop exited without result (should not happen)")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
_SEVERITY_ICONS = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}


def format_console_report(result: dict[str, Any], meta: dict[str, Any]) -> str:
    """Format findings for console output."""
    lines = [
        "=" * 68,
        "  POST-TRADE ANALYSIS REPORT",
        f"  Window: {meta['start_date']} to {meta['end_date']}",
        f"  Evaluated signals: {meta['total_signals']}",
        f"  Overall win rate: {meta['overall_win_rate']:.1f}%",
        "=" * 68,
    ]

    findings = result.get("findings", [])
    if not findings:
        lines.append("")
        lines.append("  No actionable patterns found this week.")
    else:
        for i, f in enumerate(findings, 1):
            icon = _SEVERITY_ICONS.get(f["severity"], "⚪")
            lines.append("")
            lines.append(f"  {icon} Finding {i} [{f['severity']}] — {f['category']}")
            lines.append(f"    Pattern: {f['pattern']}")
            lines.append(f"    Evidence: {json.dumps(f['evidence'])}")
            lines.append(f"    Action:  {f['suggested_action']}")

    lines.append("")
    lines.append(f"  Summary: {result.get('summary', '(none)')}")
    lines.append("")
    return "\n".join(lines)


def format_discord(result: dict[str, Any], meta: dict[str, Any]) -> str:
    """Format findings for Discord."""
    findings = result.get("findings", [])

    header = (
        f"📊 **Post-Trade Analysis** "
        f"({meta['start_date']} → {meta['end_date']}, "
        f"{meta['total_signals']} signals, "
        f"{meta['overall_win_rate']:.1f}% win rate)"
    )

    if not findings:
        return f"{header}\n\n✅ No actionable patterns found this week."

    lines = [header, ""]

    for f in findings:
        icon = _SEVERITY_ICONS.get(f["severity"], "⚪")
        lines.append(
            f"{icon} **{f['category']}** [{f['severity']}]: {f['pattern']}"
        )
        lines.append(f"  → _{f['suggested_action']}_")

    lines.append("")
    lines.append(f"**Summary:** {result.get('summary', '')}")
    return "\n".join(lines)


def send_discord(message: str) -> None:
    """POST to Discord webhook.  Silently skips if URL not set."""
    import requests

    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        print("  [DISCORD] DISCORD_WEBHOOK_URL not set — skipping notification.")
        return
    try:
        resp = requests.post(url, json={"content": message}, timeout=10)
        if resp.status_code not in (200, 204):
            print(f"  [DISCORD] HTTP {resp.status_code}: {resp.text.strip()}")
    except Exception as exc:
        print(f"  [DISCORD] Notification failed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(
    log_path: str = SIGNAL_LOG_PATH,
    window_days: int = DEFAULT_WINDOW_DAYS,
    discord: bool = False,
) -> int:
    """Run the post-trade analysis agent end-to-end.

    Returns 0 on success, 1 on failure.
    """
    print("=" * 60)
    print("  Post-Trade Analysis Agent")
    print(f"  Window: {window_days} days")
    print("=" * 60)

    # Load data
    df = load_evaluated_entries(log_path, window_days)
    if df.empty:
        print("\n[INFO] No evaluated signals in the lookback window — nothing to analyze.")
        return 0

    # Compute metadata
    start_date = str(df["date"].min().date())
    end_date = str(df["date"].max().date())
    total_signals = len(df)
    overall_win_rate = compute_overall_win_rate(df)

    meta = {
        "start_date": start_date,
        "end_date": end_date,
        "total_signals": total_signals,
        "overall_win_rate": overall_win_rate,
    }

    print(f"\n  {total_signals} evaluated signals ({start_date} to {end_date})")
    print(f"  Overall win rate: {overall_win_rate:.1f}%\n")

    # Build tables
    print("  Building summary tables...")
    table_a = build_table_a(df)
    table_b = build_table_b(df)
    table_c = build_table_c(df)
    table_d = build_table_d(df)

    # Assemble prompt
    prompt = USER_TEMPLATE.format(
        start_date=start_date,
        end_date=end_date,
        total_signals=total_signals,
        overall_win_rate=overall_win_rate,
        table_a=table_a,
        table_b=table_b,
        table_c=table_c,
        table_d=table_d,
    )

    # Run LLM analysis
    print("  Calling LLM for pattern analysis...")
    try:
        result = run_analysis(prompt)
    except RuntimeError as exc:
        print(f"\n[ERROR] Analysis failed: {exc}", file=sys.stderr)
        return 1

    # Output
    print(format_console_report(result, meta))

    if discord:
        send_discord(format_discord(result, meta))
        print("  [OK] Discord summary posted.")

    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Weekly post-trade analysis agent — surfaces patterns in signal outcomes.",
    )
    parser.add_argument(
        "--log", default=SIGNAL_LOG_PATH, metavar="PATH",
        help=f"Signal log to read (default: {SIGNAL_LOG_PATH}).",
    )
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW_DAYS, metavar="DAYS",
        help=f"Lookback window in days (default: {DEFAULT_WINDOW_DAYS}).",
    )
    parser.add_argument(
        "--discord", action="store_true",
        help="Post findings summary to Discord webhook.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(run(log_path=args.log, window_days=args.window, discord=args.discord))
