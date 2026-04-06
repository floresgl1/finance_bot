"""
outcome_tracker.py — Daily evaluator for bot signal accuracy.

Runs on the GitHub Actions VM via evaluate_signals.yml (cron: 10 15 * * 1-5).
signal_log.csv is downloaded from PythonAnywhere before this script runs and
uploaded back afterwards — this script only reads/writes the local copy at
data/signal_log.csv.

For each pending row in data/signal_log.csv (result is blank):
  - evaluation_date > today         → skip (not reached yet)
  - evaluation_date 8+ days ago     → mark SKIPPED (stale, price no longer reliable)
  - evaluation_date <= today        → fetch outcome price, compute result

Evaluation rules (±2 % threshold):
  BUY  → WIN  if outcome ≥ entry × 1.02
          LOSS if outcome ≤ entry × 0.98
          NEUTRAL otherwise

  SELL → WIN  if outcome ≤ entry × 0.98   (price fell — sell was correct)
          LOSS if outcome ≥ entry × 1.02   (price rose — sell was wrong)
          NEUTRAL otherwise

  HOLD → MISSED GAIN        if outcome ≥ entry × 1.02
          MISSED OPPORTUNITY if outcome ≤ entry × 0.98
          GOOD HOLD          otherwise

Weekend / holiday: uses the last available close price on or before evaluation_date.
Stale rows (8+ days past evaluation_date with no outcome): marked SKIPPED.
All rows are preserved — no deletions (append-only semantics).
"""

import os
import sys
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from config import DATA_DIR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SIGNAL_LOG_PATH    = os.path.join(DATA_DIR, "signal_log.csv")
SPY_CSV_PATH       = os.path.join(DATA_DIR, "market", "SPY.csv")
STALE_DAYS         = 8           # days past evaluation_date before a row is considered stale
WIN_THRESHOLD      = 0.02        # +2 %
LOSS_THRESHOLD     = 0.02        # −2 %
STARTING_PORTFOLIO = 100_000.0   # bot's initial capital
SPY_START_DATE     = date(2026, 3, 1)   # baseline date for SPY comparison
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------
def fetch_close_on_or_before(ticker: str, target_date: date) -> float | None:
    """
    Return the last available closing price on or before target_date.

    Downloads a ±7-day window to handle weekends and market holidays.
    Returns None if no data can be retrieved.
    """
    start = (target_date - timedelta(days=7)).isoformat()
    end   = (target_date + timedelta(days=1)).isoformat()   # yfinance end is exclusive

    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as exc:
        print(f"  [ERROR] {ticker} — yfinance download failed: {exc}")
        return None

    if df.empty:
        print(f"  [WARN]  {ticker} — no price data for window ending {target_date}")
        return None

    # Flatten MultiIndex columns produced by recent yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only trading days on or before the target date
    df.index = pd.to_datetime(df.index).date
    df = df[df.index <= target_date]

    if df.empty:
        print(f"  [WARN]  {ticker} — no trading day found on or before {target_date}")
        return None

    return float(df["Close"].iloc[-1])


# ---------------------------------------------------------------------------
# Result calculation
# ---------------------------------------------------------------------------
def compute_result(model_signal: str, entry_price: float, outcome_price: float) -> str:
    """Classify the outcome of a signal given entry and outcome prices."""
    change = (outcome_price - entry_price) / entry_price

    if model_signal == "BUY":
        if change >= WIN_THRESHOLD:
            return "WIN"
        if change <= -LOSS_THRESHOLD:
            return "LOSS"
        return "NEUTRAL"

    if model_signal == "SELL":
        if change <= -WIN_THRESHOLD:        # price fell — sell was right
            return "WIN"
        if change >= LOSS_THRESHOLD:        # price rose — sell was wrong
            return "LOSS"
        return "NEUTRAL"

    if model_signal == "HOLD":
        if change >= WIN_THRESHOLD:
            return "MISSED GAIN"
        if change <= -LOSS_THRESHOLD:
            return "MISSED OPPORTUNITY"
        return "GOOD HOLD"

    return "NEUTRAL"


# ---------------------------------------------------------------------------
# Discord helper
# ---------------------------------------------------------------------------
def send_discord(message: str) -> None:
    """POST a message to the Discord webhook. Silently skips if URL not set."""
    if not DISCORD_WEBHOOK_URL:
        print("[WARN] DISCORD_WEBHOOK_URL not set — skipping Discord notification")
        return
    try:
        import requests
        resp = requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[WARN] Discord notification failed: {exc}")


# ---------------------------------------------------------------------------
# Weekly summary helpers
# ---------------------------------------------------------------------------
def get_spy_start_price() -> float | None:
    """
    Return the SPY closing price on or after SPY_START_DATE.
    Reads data/market/SPY.csv first; falls back to yfinance if the file is absent.
    """
    if os.path.exists(SPY_CSV_PATH):
        try:
            spy_df = pd.read_csv(SPY_CSV_PATH)
            spy_df["Date"] = pd.to_datetime(spy_df["Date"]).dt.date
            spy_df = spy_df.sort_values("Date")
            eligible = spy_df[spy_df["Date"] >= SPY_START_DATE]
            if not eligible.empty:
                row = eligible.iloc[0]
                print(f"  SPY start (CSV): {row['Date']} → ${float(row['Close']):.2f}")
                return float(row["Close"])
        except Exception as exc:
            print(f"[WARN] Failed to read SPY CSV: {exc}")

    # Fall back to yfinance
    print(f"  [INFO] Fetching SPY start price via yfinance (on or after {SPY_START_DATE})")
    try:
        df = yf.download(
            "SPY",
            start=SPY_START_DATE.isoformat(),
            end=(SPY_START_DATE + timedelta(days=7)).isoformat(),
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            print("[WARN] yfinance returned no SPY data around start date")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).date
        df = df.sort_index()
        print(f"  SPY start (yfinance): {df.index[0]} → ${float(df['Close'].iloc[0]):.2f}")
        return float(df["Close"].iloc[0])
    except Exception as exc:
        print(f"[ERROR] yfinance SPY start fetch failed: {exc}")
        return None


def get_current_portfolio_value() -> float | None:
    """Fetch current portfolio equity from the Alpaca paper trading API."""
    try:
        try:
            import alpaca_trade_api as tradeapi
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "alpaca-trade-api"])
            import alpaca_trade_api as tradeapi

        api_key    = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            print("[WARN] ALPACA_API_KEY / ALPACA_SECRET_KEY not set — portfolio value unavailable")
            return None

        api = tradeapi.REST(
            api_key, secret_key,
            "https://paper-api.alpaca.markets",
            api_version="v2",
        )
        return float(api.get_account().equity)
    except Exception as exc:
        print(f"[ERROR] Failed to fetch portfolio value from Alpaca: {exc}")
        return None


def weekly_summary() -> None:
    """
    Compute win rate and rate of return vs SPY, then post a summary to Discord.

    Win rate = (WIN + GOOD HOLD) / (WIN + GOOD HOLD + LOSS + MISSED GAIN + MISSED OPPORTUNITY)
    Bot RoR  = (current_portfolio − STARTING_PORTFOLIO) / STARTING_PORTFOLIO × 100
    SPY RoR  = (spy_current − spy_at_SPY_START_DATE) / spy_at_SPY_START_DATE × 100
    """
    today = date.today()

    print("=" * 60)
    print("  Finance Bot — Weekly Summary")
    print(f"  As of {today}")
    print("=" * 60)

    if not os.path.exists(SIGNAL_LOG_PATH):
        print(f"[INFO] Signal log not found at '{SIGNAL_LOG_PATH}' — skipping weekly summary.")
        return

    try:
        df = pd.read_csv(SIGNAL_LOG_PATH, dtype=str)
    except Exception as exc:
        print(f"[FATAL] Could not read signal log: {exc}")
        return

    # Only count rows that have been evaluated (not blank, not SKIPPED)
    evaluated = df[
        df["result"].notna()
        & (df["result"].str.strip() != "")
        & (df["result"].str.strip() != "SKIPPED")
    ]

    counts         = evaluated["result"].str.strip().value_counts().to_dict()
    wins           = counts.get("WIN", 0)
    good_holds     = counts.get("GOOD HOLD", 0)
    losses         = counts.get("LOSS", 0)
    missed_gains   = counts.get("MISSED GAIN", 0)
    missed_opps    = counts.get("MISSED OPPORTUNITY", 0)
    neutrals       = counts.get("NEUTRAL", 0)

    denominator = wins + good_holds + losses + missed_gains + missed_opps
    win_rate    = (wins + good_holds) / denominator * 100 if denominator > 0 else 0.0

    print(
        f"\n  WIN: {wins}  GOOD HOLD: {good_holds}  LOSS: {losses}  "
        f"MISSED GAIN: {missed_gains}  MISSED OPPORTUNITY: {missed_opps}  NEUTRAL: {neutrals}"
    )
    print(f"  Win rate: {win_rate:.0f}%  (denominator: {denominator})\n")

    # SPY return
    spy_start   = get_spy_start_price()
    spy_current = fetch_close_on_or_before("SPY", today)

    if spy_start is not None and spy_current is not None:
        spy_return = (spy_current - spy_start) / spy_start * 100
        print(f"  SPY return: {spy_return:+.2f}%  (${spy_start:.2f} → ${spy_current:.2f})")
        spy_str = f"{spy_return:+.2f}%"
    else:
        print("  [WARN] Could not compute SPY return")
        spy_str = "N/A"

    # Bot rate of return
    portfolio_value = get_current_portfolio_value()
    if portfolio_value is not None:
        bot_return = (portfolio_value - STARTING_PORTFOLIO) / STARTING_PORTFOLIO * 100
        print(f"  Bot return: {bot_return:+.2f}%  (${STARTING_PORTFOLIO:,.0f} → ${portfolio_value:,.2f})")
        bot_str = f"{bot_return:+.2f}%"
    else:
        print("  [WARN] Could not compute bot rate of return")
        bot_str = "N/A"

    # Format and send Discord message
    message = (
        f"**Weekly Performance Update** ({today})\n"
        f"Win Rate: {win_rate:.0f}% "
        f"(WIN: {wins}, GOOD HOLD: {good_holds}, LOSS: {losses}, "
        f"MISSED GAIN: {missed_gains}, MISSED OPPORTUNITY: {missed_opps}, NEUTRAL: {neutrals})\n"
        f"Rate of Return: {bot_str} vs SPY: {spy_str}"
    )

    print(f"\n  Discord message preview:\n  {message}\n")
    send_discord(message)
    print("  Weekly summary complete.")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run() -> None:
    today = date.today()

    print("=" * 60)
    print("  Finance Bot — Outcome Tracker")
    print(f"  Evaluating as of {today}")
    print("=" * 60)

    if not os.path.exists(SIGNAL_LOG_PATH):
        print(f"\n[INFO] Signal log not found at '{SIGNAL_LOG_PATH}' — nothing to evaluate.")
        return

    try:
        df = pd.read_csv(SIGNAL_LOG_PATH, dtype=str)
        df["outcome_price"] = pd.to_numeric(df["outcome_price"], errors="coerce")
    except Exception as exc:
        print(f"[FATAL] Could not read signal log: {exc}")
        sys.exit(1)

    if df.empty:
        print("[INFO] Signal log is empty — nothing to evaluate.")
        return

    # Pending rows are those with a blank result
    pending_mask = df["result"].isna() | (df["result"].str.strip() == "")
    pending_idx  = df[pending_mask].index

    if len(pending_idx) == 0:
        print("\n[INFO] No pending rows — all signals already evaluated.")
        return

    print(f"\n  {len(pending_idx)} pending row(s) found.\n")

    updated = 0
    stale   = 0
    errors  = 0

    for idx in pending_idx:
        row = df.loc[idx]
        ticker = str(row.get("ticker", "")).strip()

        # --- Parse row fields ---
        try:
            eval_date    = date.fromisoformat(str(row["evaluation_date"]).strip())
            entry_price  = float(row["price"])
            model_signal = str(row["model_signal"]).strip().upper()
        except (ValueError, KeyError) as exc:
            print(f"  [ERROR]    row {idx} ({ticker}) — malformed data: {exc}")
            errors += 1
            continue

        # Not ready yet
        if eval_date > today:
            print(f"  [PENDING]  {ticker:<6}  eval {eval_date} — not reached yet, skipping")
            continue

        # Stale: evaluation_date passed more than STALE_DAYS ago with no outcome
        days_past = (today - eval_date).days
        if days_past >= STALE_DAYS:
            print(
                f"  [STALE]    {ticker:<6}  eval {eval_date} — "
                f"{days_past}d overdue, marking SKIPPED"
            )
            df.at[idx, "result"] = "SKIPPED"
            stale += 1
            continue

        # Fetch outcome price (handles weekends / holidays)
        outcome_price = fetch_close_on_or_before(ticker, eval_date)
        if outcome_price is None:
            print(f"  [ERROR]    {ticker:<6}  eval {eval_date} — could not fetch price, skipping")
            errors += 1
            continue

        result     = compute_result(model_signal, entry_price, outcome_price)
        change_pct = (outcome_price - entry_price) / entry_price * 100

        print(
            f"  [EVAL]     {ticker:<6}  {model_signal:<4}  "
            f"entry ${entry_price:.2f} → outcome ${outcome_price:.2f} "
            f"({change_pct:+.2f}%)  →  {result}"
        )

        df.at[idx, "outcome_price"] = float(round(outcome_price, 4))
        df.at[idx, "result"]        = result
        updated += 1

    # Write back — all rows preserved, only outcome_price / result fields updated
    try:
        df.to_csv(SIGNAL_LOG_PATH, index=False)
    except OSError as exc:
        print(f"\n[FATAL] Could not save signal log: {exc}")
        sys.exit(1)

    print(
        f"\n  Updated: {updated}  |  Stale/skipped: {stale}  |  Errors: {errors}\n"
        f"  Log saved to {SIGNAL_LOG_PATH}"
    )


if __name__ == "__main__":
    if "--weekly" in sys.argv:
        weekly_summary()
    else:
        run()
