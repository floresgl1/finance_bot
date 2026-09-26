"""
pnl_report.py — Realized P&L attribution over signal_log.csv EXIT rows.

signal_logger.py has been writing realized_pnl, exit_reason, exit_price and
shares on every EXIT row for months, and nothing read them. Performance
reporting was win-rate plus portfolio-vs-SPY, which answers "did the account go
up" but not "which behaviour made or lost the money".

This module answers the attribution questions:
    - Total realized P&L, profit factor, and expectancy per closed trade.
    - Which exit reason bleeds: is REBALANCE_TRIM giving back what TAKE_PROFIT
      earns?
    - Which tickers carry the book.
    - Does model confidence actually predict realized dollars?

**DESIGN DECISION:**
Win rate is reported by trade count *and* the dollar figures are shown beside
it, because they routinely disagree. A 60% win rate with an average loss twice
the average win loses money; outcome_tracker.py's win rate cannot see that.

**DESIGN DECISION:**
Entry price is derived from the EXIT row itself
(`exit_price - realized_pnl / shares`) rather than joined from the ENTRY row.
log_exit() records Alpaca's `avg_entry_price`, while the ENTRY row records the
signal-time price; they differ by slippage. Deriving keeps the return
percentage consistent with the dollar figure sitting next to it.

STOP-LOSS COVERAGE — depends on reconcile_stops.py having run.
Standing OTO stops fire on Alpaca's side between bot sessions and never pass
through log_exit(), so closed-by-stop positions do not reach this log on their
own. reconcile_stops.py backfills them as STOP_LOSS_FILL rows. The report
inspects the data for those rows and says which case it is looking at, rather
than asserting coverage it cannot verify: without them, realized P&L is biased
*upward* by however much the stopped-out positions lost.

Usage:
    python pnl_report.py                      # console report
    python pnl_report.py --discord            # also post to Discord
    python pnl_report.py --since 2026-06-01   # window the report
    python pnl_report.py --log path/to.csv
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import requests

from config import (
    ADD_TO_POSITION_CONFIDENCE_NORMAL,
    ADD_TO_POSITION_CONFIDENCE_LARGE,
)
from signal_logger import SIGNAL_LOG_PATH

# Exit reasons that reach log_exit(). Anything outside this set is surfaced
# rather than silently bucketed, so a new exit path shows up in the report the
# first time it fires.
KNOWN_EXIT_REASONS = (
    "TAKE_PROFIT",
    "MODEL_SELL",
    "REBALANCE_TRIM",
    "STOP_LOSS_FILL",   # backfilled by reconcile_stops.py
)

# Written by reconcile_stops.py for stops Alpaca filled outside a bot session.
# Their presence is what tells the report whether the stop-loss tail is covered.
STOP_LOSS_FILL = "STOP_LOSS_FILL"

# Exits that sell the whole position. A position whose last exit is one of
# these is closed; one ending in a trim is still open.
FULL_CLOSE_EXIT_REASONS = ("TAKE_PROFIT", "MODEL_SELL")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_exits(path: str = SIGNAL_LOG_PATH, since: str | None = None) -> pd.DataFrame:
    """Read EXIT rows from the signal log as a numeric-typed frame.

    Rows with an unparseable realized_pnl or shares are dropped — a partial
    row would silently skew every aggregate below it.
    """
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, dtype=str)

    if "row_type" not in df.columns:
        return pd.DataFrame()

    exits = df[df["row_type"].fillna("") == "EXIT"].copy()
    if exits.empty:
        return exits

    for col in ("realized_pnl", "exit_price", "shares"):
        exits[col] = pd.to_numeric(exits.get(col), errors="coerce")

    exits = exits.dropna(subset=["realized_pnl", "shares", "exit_price"])
    exits = exits[exits["shares"] != 0]
    if exits.empty:
        return exits

    exits["date"] = pd.to_datetime(exits["date"], errors="coerce")
    exits = exits.dropna(subset=["date"])

    if since:
        exits = exits[exits["date"] >= pd.Timestamp(since)]

    # Derived per-trade economics.
    exits["entry_price"] = exits["exit_price"] - (exits["realized_pnl"] / exits["shares"])
    exits["return_pct"] = (
        (exits["exit_price"] - exits["entry_price"]) / exits["entry_price"] * 100
    ).where(exits["entry_price"] > 0)
    exits["exit_reason"] = exits["exit_reason"].fillna("(blank)")

    return exits.sort_values("date").reset_index(drop=True)


def load_entries(path: str = SIGNAL_LOG_PATH) -> pd.DataFrame:
    """Read ENTRY rows, used to attach model confidence to closed trades."""
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, dtype=str)
    if "row_type" not in df.columns:
        return pd.DataFrame()

    entries = df[df["row_type"].fillna("ENTRY") == "ENTRY"].copy()
    if entries.empty:
        return entries

    entries["confidence"] = pd.to_numeric(entries.get("confidence"), errors="coerce")
    return entries


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def compute_summary(exits: pd.DataFrame) -> dict:
    """Headline realized-P&L statistics for a set of closed trades."""
    if exits.empty:
        return {
            "trades": 0, "total_pnl": 0.0, "wins": 0, "losses": 0, "flat": 0,
            "win_rate": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
            "profit_factor": None, "avg_win": 0.0, "avg_loss": 0.0,
            "expectancy": 0.0, "largest_win": 0.0, "largest_loss": 0.0,
            "total_shares": 0,
        }

    pnl = exits["realized_pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    decided = len(wins) + len(losses)

    gross_profit = float(wins.sum())
    gross_loss = float(losses.sum())   # negative

    # Profit factor is undefined with no losses; None reads as "n/a" rather
    # than a fabricated infinity.
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else None

    return {
        "trades": int(len(pnl)),
        "total_pnl": float(pnl.sum()),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "flat": int((pnl == 0).sum()),
        "win_rate": (len(wins) / decided * 100) if decided else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "expectancy": float(pnl.mean()),
        "largest_win": float(pnl.max()),
        "largest_loss": float(pnl.min()),
        "total_shares": int(exits["shares"].sum()),
    }


def group_pnl(exits: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per-group P&L attribution, ordered worst total first.

    Worst-first because the reason to read this table is to find the leak.
    """
    if exits.empty or column not in exits.columns:
        return pd.DataFrame()

    rows = []
    for key, group in exits.groupby(column, dropna=False):
        pnl = group["realized_pnl"]
        wins = int((pnl > 0).sum())
        decided = wins + int((pnl < 0).sum())
        rows.append({
            column: key,
            "trades": int(len(pnl)),
            "total_pnl": float(pnl.sum()),
            "avg_pnl": float(pnl.mean()),
            "win_rate": (wins / decided * 100) if decided else 0.0,
            "best": float(pnl.max()),
            "worst": float(pnl.min()),
        })

    return pd.DataFrame(rows).sort_values("total_pnl").reset_index(drop=True)


def confidence_tier(confidence: float) -> str:
    """Bucket a 0-100 confidence score using the capital_allocator tiers."""
    if pd.isna(confidence):
        return "unknown"
    # Entry rows store confidence as a percentage; allocator thresholds are 0-1.
    normalized = confidence / 100.0 if confidence > 1.0 else confidence
    if normalized >= ADD_TO_POSITION_CONFIDENCE_LARGE:
        return "large (>=0.65)"
    if normalized >= ADD_TO_POSITION_CONFIDENCE_NORMAL:
        return "normal (0.50-0.65)"
    return "small (<0.50)"


def _present(series: pd.Series) -> pd.Series:
    return series.notna() & (series.astype(str).str.strip() != "")


def attach_confidence(exits: pd.DataFrame, entries: pd.DataFrame) -> pd.DataFrame:
    """Give each EXIT the confidence of the BUY that opened its position.

    **DESIGN DECISION:**
    Joined through position_id, not entry_order_id. entry_order_id links one
    exit to one BUY, but a position spans several BUYs and several partial
    exits, so that join orphaned every exit after the first trim and linked
    others to BUYs of an earlier, closed position. The opening BUY's
    confidence is used because that is the decision that created the trade
    and set its size; adds carry their own confidence, and averaging them
    would describe no decision the model actually made.

    A position with no opening BUY in the log (held before the log began)
    stays unknown rather than borrowing an add's confidence. Exits with no
    position_id — logs not yet run through backfill_positions.py — fall back
    to the entry_order_id join. Unlinked exits land in the 'unknown' tier
    rather than being dropped — they are still real money.
    """
    if exits.empty:
        return exits

    joined = exits.copy()
    joined["confidence"] = float("nan")

    has_pid = "position_id" in joined.columns and _present(joined["position_id"])
    if not entries.empty and "position_id" in entries.columns and "position_id" in joined.columns:
        openers = entries[
            _present(entries["position_id"])
            & (entries.get("actual_action", pd.Series(index=entries.index, dtype=str)) == "BUY")
        ]
        by_position = (
            openers.drop_duplicates("position_id", keep="first")
            .set_index("position_id")["confidence"]
        )
        joined.loc[has_pid, "confidence"] = joined.loc[has_pid, "position_id"].map(by_position)

    fallback = ~has_pid if isinstance(has_pid, pd.Series) else pd.Series(True, index=joined.index)
    if not entries.empty and "entry_order_id" in entries.columns and fallback.any():
        by_order = (
            entries.dropna(subset=["entry_order_id"])
            .drop_duplicates("entry_order_id", keep="last")
            .set_index("entry_order_id")["confidence"]
        )
        joined.loc[fallback, "confidence"] = joined.loc[fallback, "entry_order_id"].map(by_order)

    joined["confidence_tier"] = joined["confidence"].apply(confidence_tier)
    return joined


def position_pnl(exits: pd.DataFrame) -> pd.DataFrame:
    """Realized P&L per position: every trim and the final sale, together.

    Counting exits counts trims as trades. The rebalancer trims an over-weight
    position several times before it closes, so per-exit win rate and
    expectancy mostly describe trim sizes. This is the per-trade view.
    """
    if exits.empty or "position_id" not in exits.columns:
        return pd.DataFrame()
    linked = exits[_present(exits["position_id"])].sort_values("date", kind="stable")
    if linked.empty:
        return pd.DataFrame()

    rows = []
    for position_id, group in linked.groupby("position_id", sort=False):
        rows.append({
            "position_id": position_id,
            "ticker": group["ticker"].iloc[0],
            "first_exit": group["date"].min(),
            "last_exit": group["date"].max(),
            "exits": int(len(group)),
            "total_pnl": float(group["realized_pnl"].sum()),
            "closed": group["exit_reason"].iloc[-1] in FULL_CLOSE_EXIT_REASONS,
        })
    return pd.DataFrame(rows)


def summarize_positions(exits: pd.DataFrame) -> dict | None:
    """Headline per-position figures over closed positions, or None."""
    positions = position_pnl(exits)
    unlinked = int((~_present(exits["position_id"])).sum()) if (
        not exits.empty and "position_id" in exits.columns
    ) else int(len(exits))
    if positions.empty:
        return None

    closed = positions[positions["closed"]]
    pnl = closed["total_pnl"]
    wins, losses = int((pnl > 0).sum()), int((pnl < 0).sum())
    decided = wins + losses
    return {
        "closed": int(len(closed)),
        "open": int((~positions["closed"]).sum()),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / decided * 100) if decided else 0.0,
        "avg_pnl": float(pnl.mean()) if len(pnl) else 0.0,
        "avg_exits": float(closed["exits"].mean()) if len(closed) else 0.0,
        "unlinked_exits": unlinked,
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def _money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def _table(df: pd.DataFrame, key: str, key_width: int = 20) -> list[str]:
    lines = [
        f"    {key:<{key_width}}{'trades':>8}{'total P&L':>14}"
        f"{'avg P&L':>12}{'win rate':>11}"
    ]
    lines.append(f"    {'-' * (key_width + 45)}")
    for _, row in df.iterrows():
        lines.append(
            f"    {str(row[key]):<{key_width}}{row['trades']:>8}"
            f"{_money(row['total_pnl']):>14}{_money(row['avg_pnl']):>12}"
            f"{row['win_rate']:>10.1f}%"
        )
    return lines


def format_report(exits: pd.DataFrame, entries: pd.DataFrame, since: str | None = None) -> str:
    """Build the full console report."""
    out: list[str] = []
    out.append("=" * 72)
    out.append("  REALIZED P&L REPORT")
    if exits.empty:
        out.append("=" * 72)
        out.append("")
        out.append("  No closed trades found in the signal log.")
        out.append("")
        out.append(_blind_spot_note(None))
        return "\n".join(out)

    window = f"{exits['date'].min().date()} to {exits['date'].max().date()}"
    out.append(f"  {window}" + (f"  (since {since})" if since else ""))
    out.append("=" * 72)

    s = compute_summary(exits)
    pf = f"{s['profit_factor']:.2f}" if s["profit_factor"] is not None else "n/a (no losses)"

    out.append("")
    out.append("  OVERALL")
    out.append(f"    Closed trades       : {s['trades']}")
    out.append(f"    Shares transacted   : {s['total_shares']:,}")
    out.append(f"    Total realized P&L  : {_money(s['total_pnl'])}")
    out.append(
        f"    Win rate (by count) : {s['win_rate']:.1f}%  "
        f"({s['wins']}W / {s['losses']}L"
        + (f" / {s['flat']} flat" if s["flat"] else "")
        + ")"
    )
    out.append(f"    Gross profit        : {_money(s['gross_profit'])}")
    out.append(f"    Gross loss          : {_money(s['gross_loss'])}")
    out.append(f"    Profit factor       : {pf}")
    out.append(f"    Average win         : {_money(s['avg_win'])}")
    out.append(f"    Average loss        : {_money(s['avg_loss'])}")
    out.append(f"    Expectancy / trade  : {_money(s['expectancy'])}")
    out.append(f"    Largest win / loss  : {_money(s['largest_win'])} / {_money(s['largest_loss'])}")

    by_reason = group_pnl(exits, "exit_reason")
    if not by_reason.empty:
        out.append("")
        out.append("  BY EXIT REASON  (worst first - this is where the leak shows)")
        out.extend(_table(by_reason, "exit_reason"))

        unknown = set(by_reason["exit_reason"]) - set(KNOWN_EXIT_REASONS)
        if unknown:
            out.append(f"    NOTE: unrecognised exit reason(s): {', '.join(sorted(unknown))}")

    by_ticker = group_pnl(exits, "ticker")
    if not by_ticker.empty:
        out.append("")
        out.append("  BY TICKER  (worst first)")
        out.extend(_table(by_ticker, "ticker", key_width=10))

    joined = attach_confidence(exits, entries)
    by_tier = group_pnl(joined, "confidence_tier")
    if not by_tier.empty:
        out.append("")
        out.append("  BY CONFIDENCE TIER  (does confidence predict dollars?)")
        out.extend(_table(by_tier, "confidence_tier"))

    ps = summarize_positions(exits)
    if ps is not None:
        out.append("")
        out.append("  BY POSITION  (a trim is part of a trade, not a trade)")
        out.append(f"    Closed positions    : {ps['closed']}  ({ps['open']} still open)")
        out.append(
            f"    Win rate            : {ps['win_rate']:.1f}%  "
            f"({ps['wins']}W / {ps['losses']}L)"
        )
        out.append(f"    Avg P&L / position  : {_money(ps['avg_pnl'])}")
        out.append(f"    Avg exits / position: {ps['avg_exits']:.1f}")
        if ps["unlinked_exits"]:
            out.append(
                f"    NOTE: {ps['unlinked_exits']} exit(s) have no position_id and are "
                f"excluded here. Run backfill_positions.py."
            )

    out.append("")
    out.append(_blind_spot_note(exits))
    return "\n".join(out)


def _blind_spot_note(exits: pd.DataFrame | None = None) -> str:
    """State whether the stop-loss tail is represented in these figures.

    Standing stops fill on Alpaca's side between bot sessions and never pass
    through log_exit(); reconcile_stops.py backfills them as STOP_LOSS_FILL
    rows. Whether that has run is the difference between an honest total and
    one biased upward, so the report says which it is looking at rather than
    asserting either unconditionally.

    ASCII only: printed to the console, and a piped Windows stdout defaults to
    cp1252, which cannot encode a warning sign. Emoji stay in the Discord
    payload, which travels as JSON.
    """
    recovered = 0
    if exits is not None and not exits.empty and "exit_reason" in exits.columns:
        recovered = int((exits["exit_reason"] == STOP_LOSS_FILL).sum())

    if recovered:
        return (
            f"  [i] Stop-loss coverage: {recovered} STOP_LOSS_FILL row(s) reconciled\n"
            f"      from Alpaca, so the losing tail is represented above. Any stop\n"
            f"      filled since the last reconcile_stops.py run is still missing."
        )

    return (
        "  [!] BLIND SPOT: no STOP_LOSS_FILL rows in this window. Standing stops\n"
        "      fill on Alpaca's side between bot sessions and never pass through\n"
        "      log_exit(), so stop-loss closes may be absent and realized P&L\n"
        "      biased UPWARD. Run reconcile_stops.py to backfill them."
    )


def format_discord(exits: pd.DataFrame, entries: pd.DataFrame) -> str:
    """Compact Discord version — headline plus the exit-reason breakdown."""
    if exits.empty:
        return "**Realized P&L** — no closed trades in the signal log yet."

    s = compute_summary(exits)
    pf = f"{s['profit_factor']:.2f}" if s["profit_factor"] is not None else "n/a"
    icon = "🟢" if s["total_pnl"] > 0 else "🔴" if s["total_pnl"] < 0 else "⚪"

    lines = [
        f"{icon} **Realized P&L** "
        f"({exits['date'].min().date()} → {exits['date'].max().date()})",
        "",
        f"Total: **{_money(s['total_pnl'])}** over {s['trades']} closed trades",
        f"Win rate {s['win_rate']:.1f}% ({s['wins']}W/{s['losses']}L) · "
        f"profit factor {pf} · expectancy {_money(s['expectancy'])}/trade",
        f"Avg win {_money(s['avg_win'])} · avg loss {_money(s['avg_loss'])}",
    ]

    by_reason = group_pnl(exits, "exit_reason")
    if not by_reason.empty:
        lines.append("")
        lines.append("**By exit reason:**")
        for _, row in by_reason.iterrows():
            lines.append(
                f"· `{row['exit_reason']}` — {_money(row['total_pnl'])} "
                f"over {row['trades']} ({row['win_rate']:.0f}% win)"
            )

    ps = summarize_positions(exits)
    if ps is not None:
        lines.append("")
        lines.append(
            f"**By position:** {ps['closed']} closed ({ps['open']} open) · "
            f"win rate {ps['win_rate']:.0f}% · {_money(ps['avg_pnl'])}/position · "
            f"{ps['avg_exits']:.1f} exits each"
        )

    lines.append("")
    recovered = int((exits["exit_reason"] == STOP_LOSS_FILL).sum())
    if recovered:
        lines.append(
            f"_Includes {recovered} reconciled stop-loss exit(s)._"
        )
    else:
        lines.append(
            "_Excludes stop-loss closes: standing Alpaca stops bypass log_exit(). "
            "Run reconcile_stops.py._"
        )
    return "\n".join(lines)


def send_discord(message: str) -> None:
    """POST a message to the Discord webhook. Silently skips if URL not set."""
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        print("  [DISCORD] DISCORD_WEBHOOK_URL not set - skipping notification.")
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
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realized P&L attribution over signal_log.csv EXIT rows.",
    )
    parser.add_argument(
        "--log", default=SIGNAL_LOG_PATH, metavar="PATH",
        help=f"Signal log to read (default: {SIGNAL_LOG_PATH}).",
    )
    parser.add_argument(
        "--since", default=None, metavar="YYYY-MM-DD",
        help="Only include trades closed on or after this date.",
    )
    parser.add_argument(
        "--discord", action="store_true",
        help="Post a compact summary to the Discord webhook.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not os.path.exists(args.log):
        print(f"[FATAL] Signal log not found: {args.log}")
        return 1

    try:
        exits = load_exits(args.log, since=args.since)
        entries = load_entries(args.log)
    except Exception as exc:
        print(f"[FATAL] Could not read {args.log}: {exc}")
        return 1

    print(format_report(exits, entries, since=args.since))

    if args.discord:
        send_discord(format_discord(exits, entries))

    return 0


if __name__ == "__main__":
    sys.exit(main())
