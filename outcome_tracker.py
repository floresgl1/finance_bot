"""
outcome_tracker.py — Daily evaluator for bot signal accuracy.

Runs on the GitHub Actions VM via evaluate_signals.yml (cron: 10 15 * * 1-5).
signal_log.csv is downloaded from PythonAnywhere before this script runs and
uploaded back afterwards — this script only reads/writes the local copy at
data/signal_log.csv.

For each pending row in data/signal_log.csv (result is blank):
  - evaluation_date > today         → skip (not reached yet)
  - evaluation_date 3+ days ago     → mark SKIPPED (stale, price no longer reliable)
  - evaluation_date <= today        → fetch outcome price, compute result

Evaluation rules (±3 % threshold):
  BUY  → WIN  if outcome ≥ entry × 1.03
          LOSS if outcome ≤ entry × 0.97
          NEUTRAL otherwise

  SELL → WIN  if outcome ≤ entry × 0.97   (price fell — sell was correct)
          LOSS if outcome ≥ entry × 1.03   (price rose — sell was wrong)
          NEUTRAL otherwise

  HOLD → MISSED GAIN        if outcome ≥ entry × 1.03
          MISSED OPPORTUNITY if outcome ≤ entry × 0.97
          GOOD HOLD          otherwise

Weekend / holiday: uses the last available close price on or before evaluation_date.
Stale rows (3+ days past evaluation_date with no outcome): marked SKIPPED.
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
SIGNAL_LOG_PATH = os.path.join(DATA_DIR, "signal_log.csv")
STALE_DAYS      = 3      # days past evaluation_date before a row is considered stale
WIN_THRESHOLD   = 0.03   # +3 %
LOSS_THRESHOLD  = 0.03   # −3 %


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

        df.at[idx, "outcome_price"] = round(outcome_price, 4)
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
    run()
