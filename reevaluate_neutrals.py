"""
reevaluate_neutrals.py — One-off script to re-evaluate NEUTRAL signals
using the updated ±2% threshold logic.

Run on PythonAnywhere:
    cd /home/floresgl907/finance_bot
    python reevaluate_neutrals.py

What it does:
  1. Backs up signal_log.csv to the same directory with today's date in the name.
  2. Finds all rows where result == 'NEUTRAL' and outcome_price is not null.
  3. Re-evaluates each row by calling compute_result() from outcome_tracker.py.
  4. Writes updated results back to signal_log.csv (all other rows untouched).
  5. Prints a summary of changes.
"""

import os
import sys
import shutil
from datetime import date

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (PythonAnywhere)
# ---------------------------------------------------------------------------
BASE_DIR        = "/home/floresgl907/finance_bot"
SIGNAL_LOG_PATH = os.path.join(BASE_DIR, "data", "signal_log.csv")

# Add project root to path so outcome_tracker imports cleanly
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from outcome_tracker import compute_result  # noqa: E402  (import after sys.path patch)

# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------
def backup_signal_log() -> str:
    today_str   = date.today().isoformat()           # e.g. 2026-04-06
    backup_path = os.path.join(
        os.path.dirname(SIGNAL_LOG_PATH),
        f"signal_log_backup_{today_str}.csv",
    )
    shutil.copy2(SIGNAL_LOG_PATH, backup_path)
    print(f"  Backup saved → {backup_path}")
    return backup_path


# ---------------------------------------------------------------------------
# Re-evaluation
# ---------------------------------------------------------------------------
def reevaluate_neutrals() -> None:
    print("=" * 60)
    print("  Re-evaluate NEUTRAL signals (±2% threshold)")
    print(f"  {date.today()}")
    print("=" * 60)

    if not os.path.exists(SIGNAL_LOG_PATH):
        print(f"[FATAL] signal_log.csv not found at {SIGNAL_LOG_PATH}")
        sys.exit(1)

    # --- Backup first ---
    backup_signal_log()

    # --- Load ---
    df = pd.read_csv(SIGNAL_LOG_PATH, dtype=str)
    df["outcome_price"] = pd.to_numeric(df["outcome_price"], errors="coerce")
    df["price"]         = pd.to_numeric(df["price"],         errors="coerce")

    # --- Identify target rows: result == NEUTRAL, outcome_price is not null,
    #     and model_signal is not REBALANCER ---
    neutral_mask = (
        df["result"].str.strip().str.upper() == "NEUTRAL"
    ) & df["outcome_price"].notna() \
      & df["price"].notna() \
      & (df["model_signal"].str.strip().str.upper() != "REBALANCER")

    neutral_idx = df[neutral_mask].index
    neutral_count = len(neutral_idx)

    if neutral_count == 0:
        print("\n  No NEUTRAL rows with a known outcome_price found — nothing to do.")
        return

    print(f"\n  Found {neutral_count} NEUTRAL row(s) with outcome_price set.\n")

    # --- Re-evaluate ---
    result_counts: dict[str, int] = {}

    for idx in neutral_idx:
        row          = df.loc[idx]
        ticker       = str(row.get("ticker", "")).strip()
        model_signal = str(row.get("model_signal", "")).strip().upper()
        entry_price  = float(row["price"])
        outcome_price = float(row["outcome_price"])

        new_result  = compute_result(model_signal, entry_price, outcome_price)
        old_result  = str(row["result"]).strip()
        change_pct  = (outcome_price - entry_price) / entry_price * 100

        print(
            f"  {ticker:<6}  {model_signal:<4}  "
            f"entry ${entry_price:.2f} → outcome ${outcome_price:.2f} "
            f"({change_pct:+.2f}%)  |  {old_result} → {new_result}"
        )

        df.at[idx, "result"] = new_result
        result_counts[new_result] = result_counts.get(new_result, 0) + 1

    # --- Save ---
    df.to_csv(SIGNAL_LOG_PATH, index=False)
    print(f"\n  signal_log.csv updated → {SIGNAL_LOG_PATH}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  NEUTRAL signals found  : {neutral_count}")
    updated_total = sum(result_counts.values())
    print(f"  Rows updated           : {updated_total}")
    print()
    for result_label, count in sorted(result_counts.items()):
        print(f"    NEUTRAL → {result_label:<20}  {count}")
    print()


if __name__ == "__main__":
    reevaluate_neutrals()
