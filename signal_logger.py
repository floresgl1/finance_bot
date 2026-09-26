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
    CANCEL_STOP_FAILED — SELL skipped because an existing standing stop-loss could not be cancelled
    STOP_BACKFILL     — standing GTC stop-loss attached to a pre-existing position at startup
    PORTFOLIO_HALT_SINGLE_DAY — pipeline halted: portfolio single-day drawdown exceeded MAX_SINGLE_DAY_LOSS_PCT
    PORTFOLIO_HALT_ROLLING    — pipeline halted: portfolio rolling-window drawdown exceeded MAX_ROLLING_LOSS_PCT
    HALT_FLAG_PRESENT         — pipeline exited at startup because the HALT_FLAG file was present
    BUY_SKIPPED_HALT          — BUY skipped because a portfolio halt is active this session
    REBALANCER_SKIPPED_HALT   — rebalancer pass skipped because a portfolio halt is active this session
"""

import csv
import os
import shutil
from datetime import date, datetime, timedelta, timezone

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
    "row_type",
    "entry_order_id",
    "exit_timestamp",
    "exit_price",
    "exit_reason",
    "shares",
    "realized_pnl",
    "position_id",
]

# Header written before position_id existed. _ensure_file() migrates a log
# with exactly this header in place; any other mismatch still raises.
LEGACY_FIELDNAMES = FIELDNAMES[:-1]


def _ensure_file() -> None:
    """Create the CSV with a header row if it does not yet exist.

    If the file exists, verify that its header matches FIELDNAMES.
    Raises ValueError on schema drift to prevent silent data corruption.
    """
    os.makedirs(os.path.dirname(os.path.abspath(SIGNAL_LOG_PATH)), exist_ok=True)

    if not os.path.exists(SIGNAL_LOG_PATH):
        with open(SIGNAL_LOG_PATH, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FIELDNAMES).writeheader()
        print(f"  [LOG] Created signal log at {SIGNAL_LOG_PATH}")
        return

    # File exists — verify schema matches FIELDNAMES
    with open(SIGNAL_LOG_PATH) as fh:
        csv_reader = csv.reader(fh)
        try:
            header = next(csv_reader)
        except StopIteration:
            raise ValueError(
                f"signal_log.csv exists but is empty (no header row).\n"
                f"To fix: delete data/signal_log.csv on PA and let the bot "
                f"recreate it, OR investigate why the file was truncated "
                f"before proceeding."
            )

    if header == LEGACY_FIELDNAMES:
        _migrate_add_position_id()
        return

    if header != FIELDNAMES:
        raise ValueError(
            f"signal_log.csv header does not match FIELDNAMES.\n"
            f"Expected ({len(FIELDNAMES)}): {FIELDNAMES}\n"
            f"Found ({len(header)}): {header}\n"
            f"To fix: update the header of data/signal_log.csv on PA to "
            f"match FIELDNAMES."
        )


def _migrate_add_position_id() -> None:
    """Rewrite a pre-position_id log with a blank position_id column.

    The live log on PA predates the column, and the strict header check would
    otherwise halt the first run after deploy. Blank is correct for history:
    existing rows are assigned position ids by the one-off backfill, not here.

    A copy of the original is kept beside it, and the rewrite goes through a
    temp file + os.replace so a crash mid-write cannot truncate the log.
    """
    backup = SIGNAL_LOG_PATH + ".pre_position_id.bak"
    tmp = SIGNAL_LOG_PATH + ".tmp"

    with open(SIGNAL_LOG_PATH, newline="") as fh:
        rows = list(csv.DictReader(fh))

    # DictReader collects surplus fields under the None key. Writing those
    # rows would silently drop data, so refuse instead.
    if any(None in row for row in rows):
        raise ValueError(
            "signal_log.csv has rows with more fields than its header; "
            "refusing to migrate it. Inspect data/signal_log.csv on PA."
        )

    shutil.copy2(SIGNAL_LOG_PATH, backup)
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, restval="")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, SIGNAL_LOG_PATH)
    print(f"  [LOG] Added position_id column to {SIGNAL_LOG_PATH} (backup: {backup})")


def log_signal(
    ticker: str,
    model_signal: str,
    price: float,
    qty: int | float,
    confidence: float,
    actual_action: str,
    shap_values: dict | None = None,
    today: date | None = None,
    entry_order_id: str | None = None,
    position_id: str | None = None,
) -> None:
    """
    Append one ENTRY row to signal_log.csv.

    Parameters
    ----------
    ticker         : stock symbol
    model_signal   : final vetoed signal driving the decision (BUY / SELL / HOLD / etc)
    price          : current price at time of decision
    qty            : shares to be traded (0 for HOLD or skipped)
    confidence     : model confidence score (0–100)
    actual_action  : execution outcome — see module docstring for valid values
    shap_values    : dict of {feature_name: shap_value} from the model prediction;
                     top 3 features by value (descending) are written to
                     shap_driver_1/2/3. Pass None or omit for skip/error rows.
    today          : override today's date; defaults to datetime.now(timezone.utc).date()
    entry_order_id : Alpaca order UUID from a successful BUY placement, used to
                     link ENTRY rows to future EXIT rows. Pass None for non-BUY
                     rows, skips, errors, or HOLDs.
    position_id    : id of the position this BUY opened or added to — the
                     order id of the BUY that opened it. Pass None for rows
                     that moved no shares.

    All exit-related columns (exit_timestamp, exit_price, exit_reason, shares,
    realized_pnl) are left blank on ENTRY rows.
    """
    _ensure_file()

    if today is None:
        today = datetime.now(timezone.utc).date()

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
        "row_type":        "ENTRY",
        "entry_order_id":  entry_order_id if entry_order_id else "",
        "exit_timestamp":  "",
        "exit_price":      "",
        "exit_reason":     "",
        "shares":          "",
        "realized_pnl":    "",
        "position_id":     position_id if position_id else "",
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


def log_exit(
    ticker: str,
    entry_order_id: str,
    entry_price: float,
    exit_price: float,
    exit_reason: str,
    shares: int | float,
    today: date | None = None,
    exit_timestamp: str | None = None,
    position_id: str | None = None,
) -> None:
    """
    Append one EXIT row to signal_log.csv.

    Called when a position closes (take-profit, model-driven SELL, or
    rebalancer trim). EXIT rows capture realized dollar P&L and link
    back to the original ENTRY row via entry_order_id.

    Parameters
    ----------
    ticker         : stock symbol
    entry_order_id : Alpaca UUID from the original BUY order; links this
                     EXIT row to its ENTRY row for analysis joins
    entry_price    : avg entry price of the closed position
    exit_price     : price at which the position closed
    exit_reason    : "TAKE_PROFIT" | "MODEL_SELL" | "REBALANCE_TRIM"
                     | "STOP_LOSS_FILL"
    shares         : number of shares closed
    today          : override today's date; defaults to datetime.now(timezone.utc).date()
    exit_timestamp : override the exit timestamp; defaults to now. Set by
                     reconcile_stops.py to Alpaca's `filled_at`, so a
                     reconciled row carries the moment the stop actually
                     fired rather than the moment it was discovered — which
                     is also what makes re-running the reconciler idempotent.
    position_id    : id of the position these shares came out of. This, not
                     entry_order_id, is the reliable link: one position spans
                     several BUYs and several partial exits.

    ENTRY-only columns (model_signal, price, qty, confidence,
    evaluation_date, actual_action, outcome_price, result,
    shap_driver_1/2/3) are left blank on EXIT rows.
    """
    _ensure_file()

    if today is None:
        today = datetime.now(timezone.utc).date()

    if exit_timestamp is None:
        exit_timestamp = datetime.now().isoformat(timespec="seconds")
    realized_pnl = round((float(exit_price) - float(entry_price)) * float(shares), 4)

    row = {
        "date":            today.isoformat(),
        "ticker":          ticker,
        "model_signal":    "",
        "price":           "",
        "qty":             "",
        "confidence":      "",
        "evaluation_date": "",
        "actual_action":   "",
        "outcome_price":   "",
        "result":          "",
        "shap_driver_1":   "",
        "shap_driver_2":   "",
        "shap_driver_3":   "",
        "row_type":        "EXIT",
        "entry_order_id":  entry_order_id,
        "exit_timestamp":  exit_timestamp,
        "exit_price":      round(float(exit_price), 4),
        "exit_reason":     exit_reason,
        "shares":          int(shares),
        "realized_pnl":    realized_pnl,
        "position_id":     position_id if position_id else "",
    }

    try:
        with open(SIGNAL_LOG_PATH, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FIELDNAMES).writerow(row)
        print(
            f"  [EXIT] {ticker:<6} {exit_reason:<14} "
            f"entry ${float(entry_price):>8.2f} → exit ${float(exit_price):>8.2f}  "
            f"P&L: ${realized_pnl:>+10.2f}"
        )
    except OSError as exc:
        print(f"  [LOG ERROR] {ticker} — could not write exit to signal log: {exc}")


def find_open_entry_order_id(ticker: str) -> str | None:
    """
    Return the entry_order_id of the most recent open ENTRY row for `ticker`.

    An ENTRY row is "open" if no EXIT row exists with the same entry_order_id.

    Used by exit-path code (take-profit, model-SELL, rebalancer) to link an
    EXIT row back to the original BUY order.

    Returns None if:
      - signal_log.csv does not exist
      - no ENTRY rows exist for ticker
      - all ENTRY rows for ticker have matching EXIT rows
      - ticker's most recent ENTRY row has a blank entry_order_id
        (e.g., pre-migration rows or BUY errors where no order was placed)
    """
    if not os.path.exists(SIGNAL_LOG_PATH):
        return None

    try:
        with open(SIGNAL_LOG_PATH, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    except OSError:
        return None

    closed_order_ids = {
        r["entry_order_id"] for r in rows
        if r.get("row_type") == "EXIT" and r.get("entry_order_id")
    }

    entry_rows = [
        r for r in rows
        if r.get("row_type") == "ENTRY"
        and r.get("ticker") == ticker
        and r.get("entry_order_id")
        and r["entry_order_id"] not in closed_order_ids
    ]

    if not entry_rows:
        return None

    entry_rows.sort(key=lambda r: r.get("date", ""), reverse=True)
    return entry_rows[0]["entry_order_id"]


def find_position_id(ticker: str) -> str | None:
    """
    Return the position_id most recently recorded for `ticker`.

    A position runs flat-to-flat: the BUY that opens it mints the id and every
    later BUY and EXIT reuses it until the ticker goes flat again. So for an
    exit placed *while the bot holds the ticker*, the most recent id is the id
    of the position being sold.

    Not valid for exits written after the fact: a new position may have been
    opened since. reconcile_stops.py resolves those by fill time instead.

    Returns None if the log does not exist or no row for the ticker carries an
    id (positions opened before position_id existed).
    """
    if not os.path.exists(SIGNAL_LOG_PATH):
        return None

    try:
        with open(SIGNAL_LOG_PATH, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return None

    candidates = [
        (r.get("date", ""), idx, r["position_id"])
        for idx, r in enumerate(rows)
        if r.get("ticker") == ticker and r.get("position_id")
    ]
    if not candidates:
        return None

    # Latest date wins; file order breaks same-day ties. Rows are not purely
    # date-ordered because reconciled stop exits are appended backdated.
    return max(candidates)[2]
