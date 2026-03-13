"""
signal_logger.py — Append-only logger for bot signal decisions.

Called by live_trader.py just before order execution.
Creates data/signal_log.csv with a header row on first run.
outcome_price and result are intentionally left blank; they are
filled later by outcome_tracker.py.

Schema:
    date, ticker, model_signal, price, qty, confidence,
    evaluation_date, actual_action, outcome_price, result
"""

import csv
import os
from datetime import date, timedelta

from config import DATA_DIR, PREDICTION_DAYS

SIGNAL_LOG_PATH = os.path.join(DATA_DIR, "signal_log.csv")

FIELDNAMES = [
    "date",
    "ticker",
    "model_signal",
    "price",
    "qty",
    "confidence",
    "evaluation_date",
    "actual_action",
    "outcome_price",
    "result",
]


def _ensure_file() -> None:
    """Create the CSV with a header row if it does not yet exist."""
    os.makedirs(os.path.dirname(os.path.abspath(SIGNAL_LOG_PATH)), exist_ok=True)
    if not os.path.exists(SIGNAL_LOG_PATH):
        with open(SIGNAL_LOG_PATH, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FIELDNAMES).writeheader()
        print(f"  [LOG] Created signal log at {SIGNAL_LOG_PATH}")


def log_signal(
    ticker: str,
    model_signal: str,
    price: float,
    qty: int | float,
    confidence: float,
    actual_action: str,
    today: date | None = None,
) -> None:
    """
    Append one row to signal_log.csv.

    Parameters
    ----------
    ticker        : stock symbol
    model_signal  : final vetoed signal driving the decision (BUY / SELL / HOLD)
    price         : current price at time of decision
    qty           : shares to be traded (0 for HOLD or skipped)
    confidence    : model confidence score (0–100)
    actual_action : what will actually execute (BUY / SELL / HOLD / SKIPPED)
    today         : override today's date; defaults to date.today()
    """
    _ensure_file()

    if today is None:
        today = date.today()

    evaluation_date = today + timedelta(days=PREDICTION_DAYS)

    row = {
        "date":            today.isoformat(),
        "ticker":          ticker,
        "model_signal":    model_signal,
        "price":           round(float(price), 4),
        "qty":             int(qty),
        "confidence":      round(float(confidence), 4),
        "evaluation_date": evaluation_date.isoformat(),
        "actual_action":   actual_action,
        "outcome_price":   "",
        "result":          "",
    }

    try:
        with open(SIGNAL_LOG_PATH, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FIELDNAMES).writerow(row)
        print(
            f"  [LOG] {ticker:<6} {model_signal:<4} → {actual_action:<8} "
            f"@ ${float(price):>8.2f}  eval: {evaluation_date}"
        )
    except OSError as exc:
        print(f"  [LOG ERROR] {ticker} — could not write to signal log: {exc}")
