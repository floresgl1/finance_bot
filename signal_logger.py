"""
signal_logger.py — Append-only logger for bot signal decisions.

Called by live_trader.py once per signal, *after* each trade attempt
completes in the stateful execution loop (not pre-execution batch).
Creates data/signal_log.csv with a header row on first run.
outcome_price and result are intentionally left blank; they are
filled later by outcome_tracker.py.

Schema:
    date, ticker, model_signal, price, qty, confidence,
    evaluation_date, actual_action, outcome_price, result,
    shap_driver_1, shap_driver_2, shap_driver_3

Valid actual_action values:
    BUY             — new position placed successfully
    ADD_TO_POSITION — added shares to an already-owned position
    SELL            — full position sold successfully
    HOLD            — HOLD signal, no action taken
    BUY_ERROR       — BUY attempted but Alpaca returned an error
    SELL_ERROR      — SELL attempted but Alpaca returned an error
    CONFIDENCE_SKIP — confidence below threshold, trade skipped
    INVALID_HEADROOM— add-to-position blocked by MAX_POSITION_PCT cap
    INSUFFICIENT_EQ — not enough equity to buy even 1 share
    NOT_OWNED       — SELL signal but no position held
    EARNINGS_VETO   — signal overridden by earnings surprise check
    SENTIMENT_VETO  — signal overridden by sentiment score check
    EXIT_SKIP       — BUY skipped because ticker was exited via stop-loss/take-profit this session
    STOP_LOSS_SELL  — position sold by the stop-loss guard (loss > threshold)
    COOLDOWN_SKIP   — BUY skipped because a STOP_LOSS_SELL was logged within the past 7 calendar days
    STALE_SKIP      — ticker skipped because its CSV data is older than STALE_DAYS calendar days
    CSV_INVALID_SKIP— ticker skipped because its CSV is missing or the Close price could not be read
    INVALID_PRICE_SKIP— add-to-position blocked because price is zero or negative
    INVALID_PORTFOLIO_SKIP— add-to-position blocked because portfolio_value is zero or negative
    INVALID_SHARES_SKIP — add-to-position blocked because shares_owned is zero or negative
    REBALANCER_TICKERS_SKIP — BUY skipped because the rebalancer trimmed this ticker in the current session
    STALE_MARKET_DATA — pipeline halted: market data CSVs failed freshness check
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
    "shap_driver_1",
    "shap_driver_2",
    "shap_driver_3",
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
    shap_values: dict | None = None,
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
    actual_action : execution outcome — see module docstring for valid values
    shap_values   : dict of {feature_name: shap_value} from the model prediction;
                    top 3 features by value (descending) are written to
                    shap_driver_1/2/3. Pass None or omit for skip/error rows.
    today         : override today's date; defaults to date.today()
    """
    _ensure_file()

    if today is None:
        today = date.today()

    evaluation_date = today + timedelta(days=PREDICTION_DAYS)

    if shap_values:
        top_drivers = [item[0] for item in sorted(shap_values.items(), key=lambda x: x[1], reverse=True)[:3]]
        shap_driver_1 = top_drivers[0] if len(top_drivers) > 0 else ""
        shap_driver_2 = top_drivers[1] if len(top_drivers) > 1 else ""
        shap_driver_3 = top_drivers[2] if len(top_drivers) > 2 else ""
    else:
        shap_driver_1 = shap_driver_2 = shap_driver_3 = ""

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
        "shap_driver_1":   shap_driver_1,
        "shap_driver_2":   shap_driver_2,
        "shap_driver_3":   shap_driver_3,
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
