"""
live_trader.py — Alpaca paper-trading execution layer.

Connects to Alpaca's paper-trading endpoint, fetches today's signals from
predictor.py, then executes trades one at a time in a stateful loop:

  Execution order:
    1. SELLs  — sorted by confidence descending
    2. BUYs   — sorted by confidence descending
    3. HOLDs  — no action

  Before each trade:
    - Fresh portfolio state is fetched from Alpaca (positions + equity).
    - For BUYs on already-owned tickers, capital_allocator.py receives the
      fresh state and returns the sizing decision.

  Error handling:
    - SELL errors are caught, logged, and the loop continues.
    - INVALID_HEADROOM from capital_allocator skips to the next BUY signal.

  Logging:
    - Each signal is logged to signal_log.csv after its trade attempt,
      with the actual execution outcome as actual_action.

  Discord:
    - A single summary notification is sent after all trades are complete.
    - Stop-loss / take-profit and sentiment-veto alerts still fire immediately.

Veto layers applied before execution:
  1. Sentiment veto  (FinBERT)          — inside get_signals()
  2. Earnings veto   (recent Surprise%) — inside get_signals()
  3. Agent veto      (LLM news check)   — after get_signals(), before execution

Stop-loss check runs before model signals:
  Loss > 10 % from avg entry → market sell + Discord alert

All orders are submitted as market orders.  Because the base URL points to
paper-api.alpaca.markets this script CANNOT place live trades.
"""

import os
import sys
import json
import subprocess
import math
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from capital_allocator import check_add_to_position, get_allocation_tier
from rebalancer import run_rebalancer
from config import (
    CONFIDENCE_THRESHOLD,
    MAX_POSITION_PCT,
    INSUFFICIENT_EQUITY,
    STALE_DAYS,
    REBALANCER_TICKERS_SKIP,
    MAX_MARKET_DATA_AGE_HOURS,
    STALE_MARKET_DATA,
    STALE_MARKET_DATA_SKIP,
    STOP_LOSS_PCT,
    STOP_LOSS_CANCEL_TIMEOUT_S,
    STOP_LOSS_POLL_INTERVAL_S,
    CANCEL_STOP_FAILED,
    STOP_BACKFILL,
    STOP_BACKFILL_FAILED,
    TAKE_PROFIT_FAILED,
    PORTFOLIO_SNAPSHOT_PATH,
    HALT_FLAG_PATH,
    MAX_SINGLE_DAY_LOSS_PCT,
    MAX_ROLLING_LOSS_PCT,
    MAX_PEAK_DRAWDOWN_PCT,
    ROLLING_LOSS_WINDOW_DAYS,
    PORTFOLIO_SNAPSHOT_RETAIN_DAYS,
    PORTFOLIO_HALT_SINGLE_DAY,
    PORTFOLIO_HALT_ROLLING,
    PORTFOLIO_HALT_PEAK_DRAWDOWN,
    HALT_FLAG_PRESENT,
    BUY_SKIPPED_HALT,
    REBALANCER_SKIPPED_HALT,
    PEAK_EQUITY_PATH,
    LAST_RUN_GUARD_PATH,
    AGENT_VETO,
    AGENT_DECISIONS_PATH,
    FILL_TIMEOUT_S,
    FILL_POLL_INTERVAL_S,
    BUY_UNFILLED,
    SELL_UNFILLED,
    ALPACA_INFRA_HALT,
    SIGNAL_ERROR,
    today_utc,
)

# ---------------------------------------------------------------------------
# Dependency check — install alpaca-trade-api if not present
# ---------------------------------------------------------------------------
try:
    import alpaca_trade_api as tradeapi
except ImportError:
    print("[setup] alpaca-trade-api not found — installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "alpaca-trade-api"])
    import alpaca_trade_api as tradeapi


# ---------------------------------------------------------------------------
# Infrastructure-error classification (Audit Finding #4)
# ---------------------------------------------------------------------------
import requests.exceptions as _req_exc


class AlpacaInfraError(Exception):
    """Raised when an Alpaca API call fails with a non-recoverable
    infrastructure error (5xx, 429, auth, network).

    Propagates up to the top-level handler in ``run()``, which logs a
    structured action code, sends a Discord alert, and halts the session.
    Per-ticker logic errors (422 bad params, 404 unknown symbol,
    insufficient buying power) are *not* wrapped in this class — they are
    handled at the call site and the execution loop continues.
    """

    def __init__(self, operation: str, ticker: str, cause: Exception):
        self.operation = operation
        self.ticker    = ticker
        self.cause     = cause
        super().__init__(
            f"[INFRA] {operation} on {ticker}: {cause}"
        )


# Status codes that indicate a per-ticker logic error, not an outage.
_TICKER_ERROR_CODES = frozenset({
    404,   # symbol / resource not found
    422,   # unprocessable entity (bad order params)
})

# Substrings in the Alpaca error message that indicate per-ticker issues
# even when the status code alone is ambiguous (e.g. 403 can be auth OR
# insufficient buying power).
_TICKER_ERROR_MESSAGES = (
    "insufficient",
    "buying power",
    "not found",
    "invalid symbol",
    "account is not allowed to short",
    "position does not exist",
)


def _is_infra_error(exc: Exception) -> bool:
    """Classify an exception as infrastructure (True) or per-ticker (False).

    Defaults to True (fail-safe) — unknown errors halt the session rather
    than silently continuing through the entire signal queue.
    """
    # Network-level failures are always infra.
    if isinstance(exc, (
        _req_exc.ConnectionError,
        _req_exc.Timeout,
        _req_exc.ProxyError,
        _req_exc.SSLError,
        _req_exc.RetryError,
    )):
        return True

    # Alpaca API errors carry a status code.
    status = getattr(exc, "status_code", None)
    if status is not None:
        if status in _TICKER_ERROR_CODES:
            return False

        # 403 is ambiguous: auth failure (infra) vs. buying-power (ticker).
        # Check the message to disambiguate.
        msg = str(exc).lower()
        if status == 403 and any(s in msg for s in _TICKER_ERROR_MESSAGES):
            return False

        # 4xx we didn't whitelist, or 5xx / 429 → infra
        if status >= 500 or status == 429 or status == 401:
            return True

    # Check the error message as a fallback (some errors lack a status code).
    msg = str(exc).lower()
    if any(s in msg for s in _TICKER_ERROR_MESSAGES):
        return False

    # Unknown → fail-safe: treat as infra.
    return True


def _raise_if_infra(exc: Exception, operation: str, ticker: str = "SYSTEM") -> None:
    """If *exc* is an infrastructure error, wrap and raise ``AlpacaInfraError``.

    Call this inside an ``except`` block after the per-ticker fallback logic.
    If the exception is a per-ticker error, this function returns normally
    and the caller continues.
    """
    if _is_infra_error(exc):
        raise AlpacaInfraError(operation, ticker, exc) from exc


# ---------------------------------------------------------------------------
# Portfolio snapshot + halt-flag helpers (Finding #3)
# ---------------------------------------------------------------------------
def _read_portfolio_snapshot() -> dict[str, float]:
    """
    Read portfolio_snapshot.json from PORTFOLIO_SNAPSHOT_PATH.

    Returns an empty dict if the file does not exist or is unreadable.
    This is intentional: first-run bootstrap must succeed without a baseline.
    """
    if not os.path.exists(PORTFOLIO_SNAPSHOT_PATH):
        return {}
    try:
        with open(PORTFOLIO_SNAPSHOT_PATH, "r") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return {}
        return {str(k): float(v) for k, v in data.items()}
    except (OSError, ValueError, TypeError) as exc:
        print(f"  [SNAPSHOT] Warning: could not read {PORTFOLIO_SNAPSHOT_PATH}: {exc}")
        return {}


def _write_portfolio_snapshot(snapshot: dict[str, float]) -> None:
    """
    Write snapshot to PORTFOLIO_SNAPSHOT_PATH, pruned to the most recent
    PORTFOLIO_SNAPSHOT_RETAIN_DAYS entries by date key (ISO format).

    On write error: print a warning but do NOT raise. Snapshot loss is
    degraded observability, not a reason to crash.
    """
    keys_sorted = sorted(snapshot.keys())
    if len(keys_sorted) > PORTFOLIO_SNAPSHOT_RETAIN_DAYS:
        keys_sorted = keys_sorted[-PORTFOLIO_SNAPSHOT_RETAIN_DAYS:]
    pruned = {k: float(snapshot[k]) for k in keys_sorted}
    try:
        with open(PORTFOLIO_SNAPSHOT_PATH, "w") as fh:
            json.dump(pruned, fh, indent=2, sort_keys=True)
    except OSError as exc:
        print(f"  [SNAPSHOT] Warning: could not write {PORTFOLIO_SNAPSHOT_PATH}: {exc}")


def _previous_session_date(snapshot: dict[str, float], before_date: str) -> str | None:
    """
    Return the most recent date key in snapshot that is strictly less than
    before_date (ISO 'YYYY-MM-DD'). Returns None if no such entry exists.
    """
    earlier = [d for d in snapshot.keys() if d < before_date]
    if not earlier:
        return None
    return max(earlier)


def _session_n_back(snapshot: dict[str, float], from_date: str, n: int) -> str | None:
    """
    Return the date key for the entry that is n trading sessions before
    from_date. Sessions are the dates already present in snapshot, sorted
    ascending. Returns None if snapshot has fewer than n entries before
    from_date.

    Note: this counts SNAPSHOT entries as sessions, not calendar days.
    A weekend or market holiday that produced no entry is naturally skipped.
    """
    earlier = sorted(d for d in snapshot.keys() if d < from_date)
    if len(earlier) < n:
        return None
    return earlier[-n]


def check_portfolio_loss_limits(api) -> tuple[bool, str]:
    """
    Evaluate both tiers of the portfolio loss limit at session start.

    Returns (should_halt, reason). When should_halt is True the caller must
    skip BUYs and the rebalancer; SELLs and take-profit still run.
    """
    snapshot = _read_portfolio_snapshot()
    current_equity = float(api.get_account().equity)
    today_str = today_utc()

    # Single-day check
    prev_date = _previous_session_date(snapshot, today_str)
    if prev_date is not None:
        prev_equity = snapshot[prev_date]
        if prev_equity > 0:
            daily_loss = (prev_equity - current_equity) / prev_equity
            if daily_loss > MAX_SINGLE_DAY_LOSS_PCT:
                return (
                    True,
                    f"Single-day loss {daily_loss * 100:.2f}% exceeds "
                    f"{MAX_SINGLE_DAY_LOSS_PCT * 100:.1f}% threshold "
                    f"(prev {prev_date}: ${prev_equity:,.2f} -> today: ${current_equity:,.2f})"
                )

    # Rolling check
    past_date = _session_n_back(snapshot, today_str, ROLLING_LOSS_WINDOW_DAYS)
    if past_date is not None:
        past_equity = snapshot[past_date]
        if past_equity > 0:
            rolling_loss = (past_equity - current_equity) / past_equity
            if rolling_loss > MAX_ROLLING_LOSS_PCT:
                return (
                    True,
                    f"Rolling {ROLLING_LOSS_WINDOW_DAYS}-session loss {rolling_loss * 100:.2f}% "
                    f"exceeds {MAX_ROLLING_LOSS_PCT * 100:.1f}% threshold "
                    f"(session {past_date}: ${past_equity:,.2f} -> today: ${current_equity:,.2f})"
                )

    return (False, "")


# ---------------------------------------------------------------------------
# Peak-to-current drawdown circuit breaker (third tier)
# ---------------------------------------------------------------------------
def _read_peak_equity() -> dict:
    """Read peak_equity.json. Returns {} on first run or if unreadable."""
    if not os.path.exists(PEAK_EQUITY_PATH):
        return {}
    try:
        with open(PEAK_EQUITY_PATH, "r") as fh:
            return json.load(fh)
    except (OSError, ValueError, TypeError) as exc:
        print(f"  [PEAK] Warning: could not read {PEAK_EQUITY_PATH}: {exc}")
        return {}


def _write_peak_equity(state: dict) -> None:
    """Write peak_equity.json. Non-fatal on error."""
    try:
        os.makedirs(os.path.dirname(PEAK_EQUITY_PATH) or ".", exist_ok=True)
        with open(PEAK_EQUITY_PATH, "w") as fh:
            json.dump(state, fh, indent=2)
    except OSError as exc:
        print(f"  [PEAK] Warning: could not write {PEAK_EQUITY_PATH}: {exc}")


def check_peak_drawdown(current_equity: float) -> tuple[bool, str]:
    """
    Compare current equity against the all-time high-water mark.

    Returns (should_halt, reason).  Updates the high-water mark when
    equity exceeds the stored peak.  On first run (no stored peak),
    seeds the peak with current equity and returns (False, "").
    """
    state = _read_peak_equity()
    peak = state.get("peak_equity")

    # First run or invalid stored peak — seed the high-water mark
    if peak is None or peak <= 0:
        state["peak_equity"] = current_equity
        state["updated_at"] = today_utc()
        _write_peak_equity(state)
        print(f"  [PEAK] First run — seeding peak equity at ${current_equity:,.2f}")
        return (False, "")

    # New high-water mark
    if current_equity > peak:
        state["peak_equity"] = current_equity
        state["updated_at"] = today_utc()
        _write_peak_equity(state)
        print(f"  [PEAK] New high-water mark: ${current_equity:,.2f} (was ${peak:,.2f})")
        return (False, "")

    drawdown = (peak - current_equity) / peak
    print(f"  [PEAK] Equity ${current_equity:,.2f} vs peak ${peak:,.2f} "
          f"— drawdown {drawdown * 100:.2f}%")

    if drawdown >= MAX_PEAK_DRAWDOWN_PCT:
        return (
            True,
            f"Peak drawdown {drawdown * 100:.2f}% exceeds "
            f"{MAX_PEAK_DRAWDOWN_PCT * 100:.1f}% threshold "
            f"(peak ${peak:,.2f} on {state.get('updated_at', '?')} "
            f"-> today ${current_equity:,.2f})"
        )

    # Drawdown within tolerance — persist unchanged state
    _write_peak_equity(state)
    return (False, "")


def _write_halt_flag(reason: str) -> None:
    """
    Write HALT_FLAG_PATH with the halt reason and a UTC timestamp.
    Overwrites existing flag file. On write error, print a warning —
    a missing flag is a separate failure mode the operator will notice
    via Discord alerts repeating the halt.
    """
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with open(HALT_FLAG_PATH, "w") as fh:
            fh.write(f"timestamp_utc: {ts}\n")
            fh.write(f"reason: {reason}\n")
    except OSError as exc:
        print(f"  [HALT_FLAG] Warning: could not write {HALT_FLAG_PATH}: {exc}")


def _check_halt_flag() -> tuple[bool, str]:
    """
    Returns (True, contents) if HALT_FLAG_PATH exists and is readable.
    Returns (False, "") otherwise. Missing file is the normal case.
    """
    if not os.path.exists(HALT_FLAG_PATH):
        return (False, "")
    try:
        with open(HALT_FLAG_PATH, "r") as fh:
            return (True, fh.read())
    except OSError as exc:
        print(f"  [HALT_FLAG] Warning: could not read {HALT_FLAG_PATH}: {exc}")
        return (True, "")


# ---------------------------------------------------------------------------
# Daily run guard
# ---------------------------------------------------------------------------
def _write_run_guard() -> None:
    """
    Stamp today's UTC date into LAST_RUN_GUARD_PATH.

    The guard means "the execution block ran to completion for this date, it is
    safe for a later run to skip" — it does NOT mean "the Discord summary was
    posted". It is therefore written after the trade/rebalance block and before
    the summary post: a Discord outage must never be able to cause a re-trade.

    live_trader.py owns this file because it is the only module that knows
    whether the market was open and whether execution actually happened. A
    market-closed no-op exits long before this point and leaves the guard
    untouched, so the later safety-net run still fires. run_bot.py only reads
    it — it cannot distinguish "traded" from "closed" through an exit code.

    A failed write is alerted, not fatal: the trades are already placed, so
    aborting here would suppress the summary while changing nothing. The alert
    matters because an unwritten guard lets the safety-net run trade again.
    """
    today = today_utc()
    try:
        os.makedirs(os.path.dirname(LAST_RUN_GUARD_PATH), exist_ok=True)
        with open(LAST_RUN_GUARD_PATH, "w") as fh:
            fh.write(today)
        print(f"  Run guard stamped: {today}")
    except OSError as exc:
        print(f"  [RUN_GUARD] Warning: could not write {LAST_RUN_GUARD_PATH}: {exc}")
        send_discord(
            f"⚠️ **Run guard write failed** — trades for {today} are already placed, "
            f"but the guard was not stamped (`{exc}`).\n"
            f"The safety-net run may execute again today. Manual review required."
        )


# ---------------------------------------------------------------------------
# Alpaca connection
# ---------------------------------------------------------------------------
BASE_URL = "https://paper-api.alpaca.markets"

def get_api() -> tradeapi.REST:
    """Build an Alpaca REST client from environment variables."""
    api_key    = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise EnvironmentError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set as environment variables."
        )

    return tradeapi.REST(api_key, secret_key, BASE_URL, api_version="v2")


# ---------------------------------------------------------------------------
# Market status
# ---------------------------------------------------------------------------
def market_is_open(api: tradeapi.REST) -> bool:
    clock = api.get_clock()
    return clock.is_open


# ---------------------------------------------------------------------------
# Market data freshness gate (Layer 1)
# ---------------------------------------------------------------------------
def check_market_data_freshness(pipeline_start_time: datetime) -> tuple[bool, str, list[str]]:
    """
    Layer 1 freshness gate: verify all market data CSVs are recent.

    Runs three checks per file:
      1. Lower bound: mtime must be >= today 00:00:00 UTC (refresh ran today)
      2. Upper bound: mtime must be < pipeline_start_time (refresh completed
         before pipeline started; catches race condition where refresh writes
         files mid-pipeline-execution)
      3. Absolute age: now - mtime must be < MAX_MARKET_DATA_AGE_HOURS (24h)

    Scans:
      - data/*.csv (per-ticker OHLCV for WATCHLIST)
      - data/market/*.csv (SPY, VIX, sector ETFs)

    Returns:
        (is_fresh, failure_reason, failed_files)
        is_fresh=True  -> all files passed, safe to proceed
        is_fresh=False -> at least one file failed; see failure_reason
    """
    import os
    from datetime import timedelta
    from config import DATA_DIR, WATCHLIST

    now_utc = datetime.now(timezone.utc)
    today_start_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    max_age = timedelta(hours=MAX_MARKET_DATA_AGE_HOURS)

    # Build list of expected CSV paths
    expected_files = []
    for ticker in WATCHLIST:
        expected_files.append(os.path.join(DATA_DIR, f"{ticker}.csv"))
    market_dir = os.path.join(DATA_DIR, "market")
    if os.path.isdir(market_dir):
        for fname in os.listdir(market_dir):
            if fname.endswith(".csv"):
                expected_files.append(os.path.join(market_dir, fname))

    failed_files = []
    failure_reasons = []

    for path in expected_files:
        if not os.path.exists(path):
            failed_files.append(os.path.basename(path))
            failure_reasons.append(f"{os.path.basename(path)}: missing")
            continue

        mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)

        # Check 1: lower bound
        if mtime < today_start_utc:
            failed_files.append(os.path.basename(path))
            failure_reasons.append(
                f"{os.path.basename(path)}: mtime {mtime.isoformat()} before today 00:00 UTC"
            )
            continue

        # Check 2: upper bound
        if mtime >= pipeline_start_time:
            failed_files.append(os.path.basename(path))
            failure_reasons.append(
                f"{os.path.basename(path)}: mtime {mtime.isoformat()} AFTER pipeline start "
                f"{pipeline_start_time.isoformat()} (refresh raced with pipeline)"
            )
            continue

        # Check 3: absolute age
        if (now_utc - mtime) > max_age:
            failed_files.append(os.path.basename(path))
            failure_reasons.append(
                f"{os.path.basename(path)}: mtime {mtime.isoformat()} older than "
                f"{MAX_MARKET_DATA_AGE_HOURS}h"
            )
            continue

    if failed_files:
        reason = "; ".join(failure_reasons[:3])
        if len(failure_reasons) > 3:
            reason += f" (+ {len(failure_reasons) - 3} more)"
        return (False, reason, failed_files)

    return (True, "", [])


def read_last_close(ticker: str) -> float | None:
    """
    Read the last `Close` value from data/{ticker}.csv.

    Returns the float price on success, None on any failure. Used by the
    STALE_SKIP and STALE_MARKET_DATA_SKIP branches in get_signals() to
    record a last-known price in signal_log.csv even when the ticker is
    being skipped due to stale data.
    """
    import pandas as pd
    from config import DATA_DIR
    csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    try:
        df = pd.read_csv(csv_path)
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pre-trade agent decisions (Phase 3 of the three-phase pipeline)
# ---------------------------------------------------------------------------
def _load_agent_decisions() -> dict[str, str]:
    """Load agent_decisions.json and return a {ticker: verdict} map.

    Permissive: returns an empty dict (= ABSTAIN for all tickers) on any
    failure — missing file, parse error, wrong date, or unexpected schema.
    live_trader.py must never fail to trade because the agent layer broke.
    """
    today = today_utc()

    if not os.path.exists(AGENT_DECISIONS_PATH):
        msg = "No agent_decisions.json found"
        print(f"  [AGENT] {msg} — defaulting to ABSTAIN for all.")
        send_discord(f"⚠️ **Agent pipeline gap** — {msg}. Trading with ABSTAIN defaults (no vetoes).")
        return {}

    try:
        with open(AGENT_DECISIONS_PATH) as fh:
            envelope = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        msg = f"Cannot parse agent_decisions.json ({exc})"
        print(f"  [AGENT] {msg} — defaulting to ABSTAIN.")
        send_discord(f"⚠️ **Agent pipeline gap** — {msg}. Trading with ABSTAIN defaults (no vetoes).")
        return {}

    # Date guard: reject yesterday's decisions
    file_date = envelope.get("date")
    if file_date != today:
        msg = f"agent_decisions.json is for {file_date}, not {today}"
        print(f"  [AGENT] {msg} — defaulting to ABSTAIN.")
        send_discord(f"⚠️ **Agent pipeline gap** — {msg}. Trading with ABSTAIN defaults (no vetoes).")
        return {}

    decisions = envelope.get("decisions", {})
    if not isinstance(decisions, dict):
        msg = "'decisions' is not a dict"
        print(f"  [AGENT] {msg} — defaulting to ABSTAIN.")
        send_discord(f"⚠️ **Agent pipeline gap** — {msg}. Trading with ABSTAIN defaults (no vetoes).")
        return {}

    # Log what the agent decided
    vetoes = [t for t, v in decisions.items() if v == "VETO"]
    confirms = [t for t, v in decisions.items() if v == "CONFIRM"]
    if vetoes:
        print(f"  [AGENT] Will VETO: {', '.join(sorted(vetoes))}")
    if confirms:
        print(f"  [AGENT] CONFIRM: {', '.join(sorted(confirms))}")
    if not vetoes and not confirms:
        print("  [AGENT] All signals ABSTAIN (no vetoes or confirms).")

    return decisions


# ---------------------------------------------------------------------------
# Signal generation (reuses predictor.py logic, returns data instead of printing)
# ---------------------------------------------------------------------------
def get_signals(sentiment_df=None) -> list[dict]:
    """
    Return a list of signal dicts for every ticker in WATCHLIST.

    Applies two veto layers in order:
      1. Earnings veto   (recent Surprise% within 90 days)
      2. Sentiment veto  (CSV-based, thresholds ±0.5)

    sentiment_df must be pre-loaded by the caller (load_sentiment_df()) so
    the CSV is not re-read on every call.

    Each dict contains:
        ticker, signal, final_signal, confidence, current_price, sentiment, note
    """
    import pandas as pd
    from datetime import date as date_cls
    from config import WATCHLIST
    from predictor import (load_model, predict_ticker,
                           apply_earnings_veto, get_recent_earnings_surprise,
                           apply_sentiment_veto, is_ticker_stale)
    from features import StaleMarketDataError

    model = load_model()
    today = date_cls.today()

    results = []
    for ticker in WATCHLIST:
        try:
            stale, days_old = is_ticker_stale(ticker)
            if stale:
                # Translate sentinel values to human-readable age labels
                if days_old == 9999:
                    age_str = "parse error"
                elif days_old == -1:
                    age_str = "csv file missing"
                else:
                    age_str = f"{days_old} days old"

                # Read the last known close price from the CSV for logging.
                # If this fails the CSV is unreadable — log as CSV_INVALID_SKIP
                # and skip entirely rather than appending a stale entry.
                _last_price = read_last_close(ticker)

                if _last_price is None:
                    from signal_logger import log_signal as _log_invalid
                    print(f"  [CSV_INVALID_SKIP] {ticker} — could not read Close price, skipping")
                    _log_invalid(ticker, "CSV_INVALID_SKIP", 0.0, 0, 0.0, "CSV_INVALID_SKIP")
                    continue

                print(f"  [STALE_SKIP] {ticker} — {age_str} (limit: {STALE_DAYS}d)")
                results.append({
                    "ticker":        ticker,
                    "signal":        "STALE_SKIP",
                    "final_signal":  "STALE_SKIP",
                    "confidence":    0.0,
                    "current_price": _last_price,
                    "sentiment":     0.0,
                    "note":          "STALE_SKIP",
                    "veto_reason":   None,
                    "shap_values":   {},
                    "days_old":      days_old,
                    "age_str":       age_str,
                })
                continue

            r            = predict_ticker(ticker, model)
            model_signal = r["signal"]

            veto_reason = None

            # 1. Earnings veto
            surprise = get_recent_earnings_surprise(ticker)
            post_earnings, earn_note = apply_earnings_veto(model_signal, surprise)
            if post_earnings != model_signal:
                veto_reason = "EARNINGS_VETO"

            # 2. Sentiment veto (only if not already HOLD from earnings)
            final_signal = post_earnings
            sent_note    = ""
            sent_score   = 0.0

            if post_earnings != "HOLD" and sentiment_df is not None:
                rows = sentiment_df[
                    (sentiment_df["Ticker"] == ticker) &
                    (sentiment_df["Date"] == today)
                ]
                sent_score = float(rows["sentiment_score"].iloc[-1]) if not rows.empty else 0.0
                final_signal, sent_note = apply_sentiment_veto(post_earnings, sent_score)
                if final_signal == "HOLD" and sent_note:
                    veto_reason = "SENTIMENT_VETO"
                    send_discord(
                        f"\N{NO ENTRY} **Sentiment Veto:** {ticker} {today}  "
                        f"Model said: {post_earnings}  "
                        f"Sentiment score: {sent_score:.3f}  "
                        f"Action: HOLD"
                    )

            r["final_signal"] = final_signal
            r["sentiment"]    = sent_score
            r["note"]         = earn_note or sent_note
            r["veto_reason"]  = veto_reason
            results.append(r)
        except StaleMarketDataError as exc:
            last_price = read_last_close(ticker)
            if last_price is None:
                last_price = 0.0
            print(f"  [STALE_MARKET_DATA_SKIP] {ticker} — {exc}")
            results.append({
                "ticker":        ticker,
                "signal":        STALE_MARKET_DATA_SKIP,
                "final_signal":  STALE_MARKET_DATA_SKIP,
                "confidence":    0.0,
                "current_price": last_price,
                "sentiment":     0.0,
                "note":          STALE_MARKET_DATA_SKIP,
                "veto_reason":   None,
                "shap_values":   {},
                "days_old":      0,
                "age_str":       "stale market data",
            })
        except Exception as exc:
            _raise_if_infra(exc, "get_signals", ticker)
            print(f"  [SIGNAL_ERROR] {ticker} — {exc}")
            last_price = read_last_close(ticker)
            results.append({
                "ticker":        ticker,
                "signal":        SIGNAL_ERROR,
                "final_signal":  SIGNAL_ERROR,
                "confidence":    0.0,
                "current_price": last_price if last_price is not None else 0.0,
                "sentiment":     0.0,
                "note":          SIGNAL_ERROR,
                "veto_reason":   None,
                "shap_values":   {},
                "error":         str(exc),
            })

    return results


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------
def get_owned_tickers(api: tradeapi.REST) -> dict[str, float]:
    """Return {ticker: qty_held} for all current positions."""
    try:
        positions = api.list_positions()
    except Exception as exc:
        _raise_if_infra(exc, "list_positions")
        raise  # per-ticker errors shouldn't happen here, but don't swallow
    return {p.symbol: float(p.qty) for p in positions}


def get_equity(api: tradeapi.REST) -> float:
    try:
        return float(api.get_account().equity)
    except Exception as exc:
        _raise_if_infra(exc, "get_account")
        raise


# ---------------------------------------------------------------------------
# Order helpers
# ---------------------------------------------------------------------------
def get_position_size(confidence: float) -> float | None:
    """
    Map a model confidence score (0–100) to a portfolio fraction.

    Tiers (from capital_allocator, so an open and a top-up share one rule):
      below CONFIDENCE_THRESHOLD × 100 → None  (skip trade entirely, treat as HOLD)
      35 – 50   → SMALL_POSITION_PCT   (3 % of equity)
      50 – 65   → NORMAL_POSITION_PCT  (5 % of equity)
      65+       → LARGE_POSITION_PCT   (7 % of equity)

    **DESIGN DECISION:**
    This used to open positions at 10/15/20% of equity while MAX_POSITION_PCT
    capped the same position at 8%. Every entry therefore arrived over-weight:
    the rebalancer trimmed it back to 7.5% over roughly three sessions, paying
    slippage and a commission on stock the bot had chosen to buy days earlier,
    and check_add_to_position refused every top-up in between with
    INVALID_HEADROOM. The signal log recorded 47 REBALANCER_SELL against 17 BUY
    and 31 INVALID_HEADROOM.

    Sizing from the same tier table the add path uses fixes all three: nothing
    opens above its own cap, the rebalancer returns to being a safety net rather
    than a routine step, and adding to a winner becomes possible for the first
    time. It also keeps the diversified, near-fully-invested book the account
    was already converging on by way of the trims — which is the configuration
    that actually beat every simulated alternative
    (docs/EDGE_INVESTIGATION_2026-09-08.md, finding I).

    Concentration was the alternative resolution — raising MAX_POSITION_PCT to
    0.20 instead. It was rejected because concentration only pays when there is
    selection edge to concentrate into, and six independent measurements say
    there is none.
    """
    if confidence < CONFIDENCE_THRESHOLD * 100:
        return None
    _tier, pct = get_allocation_tier(confidence / 100.0)
    return min(pct, MAX_POSITION_PCT)


def compute_buy_qty(equity: float, price: float, fraction: float = 0.20) -> int:
    """Whole shares that fit within `fraction` of account equity."""
    return max(math.floor(equity * fraction / price), 0)


def _submit_buy(api: tradeapi.REST, ticker: str, qty: int) -> dict:
    """Submit an OTO buy order without verifying the fill. Returns a result dict.

    Separated from :func:`place_buy` so the retry loop can re-submit cleanly.
    """
    live_price_for_stop = float(api.get_latest_trade(ticker).price)
    stop_price = round(live_price_for_stop * (1.0 - STOP_LOSS_PCT), 2)

    order = api.submit_order(
        symbol        = ticker,
        qty           = qty,
        side          = "buy",
        type          = "market",
        time_in_force = "gtc",
        order_class   = "oto",
        stop_loss     = {"stop_price": stop_price},
    )
    print(
        f"  [BUY]  {ticker} x{qty} — OTO entry est. ${live_price_for_stop:.2f} "
        f"stop @ ${stop_price:.2f} — order id {order.id}"
    )
    return {"status": "placed", "order_id": order.id}


def place_buy(api: tradeapi.REST, ticker: str, qty: int) -> dict:
    """Submit an OTO buy, poll for fill, retry once if unfilled.

    The stop_price is computed from the most recent trade price at submission time, so
    the logged entry estimate and the submitted stop are visible in the same line for
    drift inspection.

    Returns ``{"status": "filled", "order_id": ...}`` on confirmed fill,
    ``{"status": "unfilled", "order_id": ...}`` after both attempts fail (a Discord
    alert is sent), or the usual ``"skipped"`` / ``"error"`` dicts.
    """
    if qty <= 0:
        print(f"  [SKIP] {ticker} — insufficient equity for even 1 share")
        return {"status": "skipped", "reason": "insufficient equity"}

    for attempt in (1, 2):
        try:
            result = _submit_buy(api, ticker, qty)
        except Exception as exc:
            _raise_if_infra(exc, "submit_buy", ticker)
            print(f"  [ERROR] {ticker} BUY submit failed: {exc}")
            return {"status": "error", "reason": str(exc)}

        order_id = result["order_id"]
        filled_order = _wait_for_fill(api, order_id)
        if filled_order is not None:
            print(f"  [FILL]  {ticker} order {order_id} — {filled_order.status}")
            return {"status": "filled", "order_id": order_id,
                    "filled_qty": getattr(filled_order, "filled_qty", None)}

        # Not filled — cancel the pending order before a possible retry.
        _cancel_order_best_effort(api, order_id)

        # Guard against the race where the order filled between our last
        # poll and the cancel attempt. Without this check, the retry would
        # double the position.
        late_fill = _check_fill_after_cancel(api, order_id)
        if late_fill is not None:
            return {"status": "filled", "order_id": order_id,
                    "filled_qty": getattr(late_fill, "filled_qty", None)}

        if attempt == 1:
            print(f"  [RETRY] {ticker} order {order_id} not filled — retrying once")

    # Both attempts failed.
    send_discord(
        f"🚨 **BUY UNFILLED** — {ticker} x{qty}\n"
        f"Two submission attempts timed out without a fill.\n"
        f"Last order id: {order_id}\n"
        f"Manual review required."
    )
    return {"status": "unfilled", "order_id": order_id}


def _submit_sell(api: tradeapi.REST, ticker: str, qty: float) -> dict:
    """Submit a market sell order without verifying the fill. Returns a result dict."""
    order = api.submit_order(
        symbol        = ticker,
        qty           = qty,
        side          = "sell",
        type          = "market",
        time_in_force = "day",
    )
    print(f"  [SELL] {ticker} x{qty} — order id {order.id}")
    return {"status": "placed", "order_id": order.id}


def place_sell(api: tradeapi.REST, ticker: str, qty: float) -> dict:
    """Submit a market sell, poll for fill, retry once if unfilled.

    Returns ``{"status": "filled", "order_id": ...}`` on confirmed fill,
    ``{"status": "unfilled", "order_id": ...}`` after both attempts fail (a Discord
    alert is sent), or the usual ``"error"`` dict.
    """
    for attempt in (1, 2):
        try:
            result = _submit_sell(api, ticker, qty)
        except Exception as exc:
            _raise_if_infra(exc, "submit_sell", ticker)
            print(f"  [ERROR] {ticker} SELL submit failed: {exc}")
            return {"status": "error", "reason": str(exc)}

        order_id = result["order_id"]
        filled_order = _wait_for_fill(api, order_id)
        if filled_order is not None:
            print(f"  [FILL]  {ticker} order {order_id} — {filled_order.status}")
            return {"status": "filled", "order_id": order_id,
                    "filled_qty": getattr(filled_order, "filled_qty", None)}

        _cancel_order_best_effort(api, order_id)

        late_fill = _check_fill_after_cancel(api, order_id)
        if late_fill is not None:
            return {"status": "filled", "order_id": order_id,
                    "filled_qty": getattr(late_fill, "filled_qty", None)}

        if attempt == 1:
            print(f"  [RETRY] {ticker} order {order_id} not filled — retrying once")

    send_discord(
        f"🚨 **SELL UNFILLED** — {ticker} x{qty}\n"
        f"Two submission attempts timed out without a fill.\n"
        f"Last order id: {order_id}\n"
        f"Manual review required."
    )
    return {"status": "unfilled", "order_id": order_id}


# ---------------------------------------------------------------------------
# Stop-loss cooldown check
# ---------------------------------------------------------------------------
def get_cooldown_tickers() -> set[str]:
    """
    Read signal_log.csv once and return the set of tickers that had a
    STOP_LOSS_SELL logged within the past STOP_LOSS_COOLDOWN_DAYS calendar days.
    """
    import csv as _csv
    from datetime import date as _date
    from signal_logger import SIGNAL_LOG_PATH
    from config import STOP_LOSS_COOLDOWN_DAYS

    if not os.path.exists(SIGNAL_LOG_PATH):
        return set()

    today   = _date.today()
    tickers = set()
    try:
        with open(SIGNAL_LOG_PATH, newline="") as fh:
            for row in _csv.DictReader(fh):
                if row["actual_action"] != "STOP_LOSS_SELL":
                    continue
                try:
                    log_date = _date.fromisoformat(row["date"])
                    if (today - log_date).days <= STOP_LOSS_COOLDOWN_DAYS:
                        tickers.add(row["ticker"])
                except ValueError:
                    continue
    except OSError:
        pass
    return tickers


# ---------------------------------------------------------------------------
# Standing stop-loss cancellation (required before any SELL)
# ---------------------------------------------------------------------------
def cancel_standing_stops(api, ticker: str) -> tuple[bool, list[str]]:
    """
    Cancel any open standing stop orders for `ticker`.

    Queries Alpaca for open orders on `ticker` where side='sell' and type in
    ('stop', 'stop_limit'). Cancels each one and polls until Alpaca reports a
    terminal status.

    Returns:
        (success, cancelled_order_ids)
        success=True  -> safe to proceed with a subsequent SELL on this ticker.
                         May return (True, []) if no standing stop existed.
        success=False -> at least one cancel errored or timed out. Caller MUST NOT
                         attempt a subsequent SELL on this ticker this session.
    """
    try:
        open_orders = api.list_orders(status="open", symbols=[ticker])
    except Exception as exc:
        _raise_if_infra(exc, "list_orders", ticker)
        print(f"  [CANCEL_STOPS] {ticker} — could not list open orders: {exc}")
        return (False, [])

    stops = [
        o for o in open_orders
        if o.side == "sell" and o.type in ("stop", "stop_limit")
    ]

    if not stops:
        return (True, [])

    if len(stops) > 1:
        print(f"  [CANCEL_STOPS] {ticker} — WARNING: found {len(stops)} standing stops (expected 1)")

    succeeded: list[str] = []
    failed:    list[str] = []

    for stop in stops:
        try:
            api.cancel_order(stop.id)
        except Exception as exc:
            _raise_if_infra(exc, "cancel_order", ticker)
            print(f"  [CANCEL_STOPS] {ticker} — cancel request failed for order {stop.id}: {exc}")
            failed.append(stop.id)
            continue

        deadline = time.time() + STOP_LOSS_CANCEL_TIMEOUT_S
        terminal = False
        while time.time() < deadline:
            try:
                current = api.get_order(stop.id)
            except Exception as exc:
                _raise_if_infra(exc, "get_order_cancel_poll", ticker)
                print(f"  [CANCEL_STOPS] {ticker} — status poll failed for order {stop.id}: {exc}")
                break
            if current.status in ("canceled", "filled"):
                succeeded.append(stop.id)
                terminal = True
                break
            time.sleep(STOP_LOSS_POLL_INTERVAL_S)

        if not terminal:
            print(f"  [CANCEL_STOPS] {ticker} — timeout awaiting cancel of order {stop.id}")
            if stop.id not in failed:
                failed.append(stop.id)

    if failed:
        return (False, succeeded)
    return (True, succeeded)


# ---------------------------------------------------------------------------
# Order fill verification
# ---------------------------------------------------------------------------
_FILL_TERMINAL = frozenset({"filled", "partially_filled"})
_FILL_FAILED   = frozenset({"rejected", "canceled", "expired", "suspended"})


def _wait_for_fill(api, order_id: str) -> object | None:
    """Poll ``api.get_order(order_id)`` until the order reaches a fill state.

    Returns the Alpaca order object on fill/partial fill, or ``None`` on
    timeout or terminal failure (rejected / canceled / expired).
    """
    deadline = time.time() + FILL_TIMEOUT_S
    while time.time() < deadline:
        try:
            order = api.get_order(order_id)
        except Exception as exc:
            _raise_if_infra(exc, "get_order_fill_poll")
            print(f"  [FILL_POLL] order {order_id} — status poll error: {exc}")
            time.sleep(FILL_POLL_INTERVAL_S)
            continue
        if order.status in _FILL_TERMINAL:
            return order
        if order.status in _FILL_FAILED:
            print(f"  [FILL_POLL] order {order_id} — terminal failure: {order.status}")
            return None
        time.sleep(FILL_POLL_INTERVAL_S)
    print(f"  [FILL_POLL] order {order_id} — timed out after {FILL_TIMEOUT_S}s")
    return None


def _cancel_order_best_effort(api, order_id: str) -> None:
    """Try to cancel an unfilled order. Swallows all errors."""
    try:
        api.cancel_order(order_id)
    except Exception as exc:
        print(f"  [FILL_CANCEL] order {order_id} — cancel failed (swallowed): {exc}")


def _check_fill_after_cancel(api, order_id: str) -> object | None:
    """One final status check after cancelling an order.

    If the order filled between our last poll and the cancel attempt, the
    cancel failed silently but the position is open. Retrying without
    checking would double the position. Returns the order object if it
    filled, ``None`` otherwise.
    """
    try:
        order = api.get_order(order_id)
        if order.status in _FILL_TERMINAL:
            print(f"  [FILL_RACE] order {order_id} — filled after cancel ({order.status})")
            return order
    except Exception as exc:
        _raise_if_infra(exc, "get_order_post_cancel")
        print(f"  [FILL_RACE] order {order_id} — post-cancel check failed: {exc}")
    return None


# ---------------------------------------------------------------------------
# Take-profit (stop-loss is now enforced via standing OTO orders)
# ---------------------------------------------------------------------------
def check_position_limits(
    api: tradeapi.REST,
    owned: dict[str, float],
    take_pct: float = 15.0,
) -> tuple[list[str], dict[str, float]]:
    """
    Check every open position for a take-profit trigger.

    Gain formula: (current_price - entry_price) / entry_price * 100

    Take-profit — gain_pct > take_pct (default 15 %): sells and sends 🎯 [TAKE PROFIT] alert.

    Stop-loss is now enforced via standing OTO orders placed at entry time; it is no
    longer evaluated here. Before issuing a take-profit SELL the function cancels any
    standing stop on the ticker; if the cancel fails the SELL is skipped so the two
    orders cannot collide on the same position.

    Returns:
        exited — list of tickers that were sold
        owned  — updated positions dict (exited tickers removed)
    """
    try:
        positions = api.list_positions()
    except Exception as exc:
        _raise_if_infra(exc, "list_positions", "TAKE_PROFIT")
        raise  # per-ticker errors shouldn't happen here, but don't swallow
    exited    = []

    for position in positions:
        ticker        = position.symbol
        current_price = float(position.current_price)
        entry_price   = float(position.avg_entry_price)
        qty           = float(position.qty)

        gain_pct = (current_price - entry_price) / entry_price * 100

        if gain_pct > take_pct:
            cancel_ok, _ = cancel_standing_stops(api, ticker)
            if not cancel_ok:
                send_discord(
                    f"🚨 **[CANCEL_STOP_FAILED]** {ticker} — could not cancel standing stop; "
                    f"SELL skipped to avoid conflict. Manual review required."
                )
                from signal_logger import log_signal
                log_signal(ticker, "SELL", current_price, 0, 0.0, CANCEL_STOP_FAILED)
                continue

            print(f"  [TAKE PROFIT] {ticker} — up {gain_pct:.1f}% — selling {qty} shares")
            result = place_sell(api, ticker, qty)
            if result["status"] == "filled":
                send_discord(
                    f"🎯 **[TAKE PROFIT]** {ticker} sold — up {gain_pct:.1f}%  "
                    f"(entry ${entry_price:.2f} → current ${current_price:.2f})"
                )
                from signal_logger import log_exit, find_open_entry_order_id
                entry_order_id = find_open_entry_order_id(ticker) or "UNLINKED"
                log_exit(
                    ticker         = ticker,
                    entry_order_id = entry_order_id,
                    entry_price    = entry_price,
                    exit_price     = current_price,
                    exit_reason    = "TAKE_PROFIT",
                    shares         = qty,
                )
                exited.append(ticker)
                owned.pop(ticker, None)
            elif result["status"] == "unfilled":
                # place_sell already sent a Discord alert — just log the action code.
                from signal_logger import log_signal
                log_signal(ticker, "SELL", current_price, qty, 0.0, SELL_UNFILLED)
            else:
                from signal_logger import log_signal
                log_signal(ticker, "SELL", current_price, qty, 0.0, TAKE_PROFIT_FAILED)
                send_discord(
                    f"🚨 **[TAKE_PROFIT_FAILED]** {ticker} — sell order failed: "
                    f"{result.get('reason', 'unknown')}. Position still open at {gain_pct:.1f}% gain. "
                    f"Manual review required."
                )

    return exited, owned


# ---------------------------------------------------------------------------
# Discord alerting
# ---------------------------------------------------------------------------
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


def send_discord(message: str) -> None:
    """POST a message to the Discord webhook. Silently skips if URL not set."""
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        import requests
        resp = requests.post(
            DISCORD_WEBHOOK_URL,
            json    = {"content": message},
            timeout = 10,
        )
        resp.raise_for_status()
    except Exception as exc:
        print(f"  [Discord] Alert failed: {exc}")


def build_post_order_alert(
    outcomes: list[dict],
    portfolio_value: float,
    timestamp: str,
    rebalancer_outcomes: list[dict] | None = None,
    shap_by_signal: dict | None = None,
    stale_skips: list[dict] | None = None,
) -> str:
    """
    Build the post-order Discord summary message.

    Each item in `outcomes` / `rebalancer_outcomes` has:
        ticker, action, qty, price, status, order_id, reason
    """
    filled   = [o for o in outcomes if o["status"] == "filled"]
    unfilled = [o for o in outcomes if o["status"] == "unfilled"]
    skipped  = [o for o in outcomes if o["status"] == "skipped"]
    errors   = [o for o in outcomes if o["status"] == "error"]

    lines = [
        f"**Finance Bot — Trade Results** | {timestamp}",
        f"Portfolio value: **${portfolio_value:,.2f}**",
        "```",
    ]

    if filled:
        lines.append("Filled:")
        for o in filled:
            lines.append(f"  {o['action']:<4} {o['ticker']:<6} x{o['qty']:<4} @ ${o['price']:>8.2f}  order {o['order_id']}")
    else:
        lines.append("Filled:  none")

    if unfilled:
        lines.append("Unfilled:")
        for o in unfilled:
            lines.append(f"  {o['action']:<4} {o['ticker']:<6} x{o['qty']:<4}  order {o['order_id']}")

    if skipped:
        lines.append("Skipped:")
        for o in skipped:
            lines.append(f"  {o['action']:<4} {o['ticker']:<6}  — {o['reason']}")
    else:
        lines.append("Skipped: none")

    if errors:
        lines.append("Errors:")
        for o in errors:
            lines.append(f"  {o['action']:<4} {o['ticker']:<6}  — {o['reason']}")
    else:
        lines.append("Errors:  none")

    # Rebalanced section — only shown when the rebalancer acted
    rb = rebalancer_outcomes or []
    rb_placed = [o for o in rb if o["status"] == "placed"]
    rb_failed = [o for o in rb if o["status"] == "error"]

    if rb_placed or rb_failed:
        lines.append("Rebalanced:")
        for o in rb_placed:
            lines.append(f"  SELL {o['ticker']:<6}  {o['qty']} share(s) @ ${o['price']:>8.2f}")
        for o in rb_failed:
            lines.append(f"  SELL {o['ticker']:<6}  {o['qty']} share(s) @ ${o['price']:>8.2f}  [ORDER FAILED: {o['reason']}]")
    else:
        lines.append("Rebalanced: none")

    stale = stale_skips or []
    if stale:
        lines.append(f"Stale data skipped (>{STALE_DAYS}d):")
        for entry in stale:
            lines.append(f"  SKIP {entry['ticker']:<6}  — {entry['age_str']}")
    else:
        lines.append("Stale data: none")

    lines.append("```")

    # SHAP section — average feature contributions per signal group, top 3 by |value|
    def _avg_top3_shap(shap_list: list[dict]) -> str:
        if not shap_list:
            return "—"
        keys = shap_list[0].keys()
        avg  = {k: sum(d.get(k, 0.0) for d in shap_list) / len(shap_list) for k in keys}
        top3 = sorted(avg.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        return ", ".join(f"{k.lower()} {v:+.2f}" for k, v in top3)

    shap_data  = shap_by_signal or {}
    shap_lines = ["**SHAP — Top drivers:**"]
    for group in ("BUY", "HOLD", "SELL"):
        shap_list = shap_data.get(group, [])
        drivers   = _avg_top3_shap(shap_list)
        shap_lines.append(f"{group} ({len(shap_list)} tickers): {drivers}")
    lines.append("\n".join(shap_lines))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
def run() -> None:
    print("=" * 60)
    print("  Finance Bot — Alpaca Paper Trader")
    print("=" * 60)

    # 1. Connect
    api = get_api()
    print(f"  Connected to {BASE_URL}")

    # 2. Market status
    if not market_is_open(api):
        print("\n  Market is currently CLOSED — nothing to do. Exiting.\n")
        sys.exit(0)

    print("  Market is OPEN — proceeding with signal generation.\n")

    # 2a. Halt flag: manual-reset circuit breaker from a prior portfolio halt.
    halt_present, halt_contents = _check_halt_flag()
    if halt_present:
        from signal_logger import log_signal
        log_signal("PIPELINE", "HALT_FLAG_PRESENT", 0, 0, 0, HALT_FLAG_PRESENT)
        send_discord(
            f"🛑 **Halt flag present — bot will not trade.**\n"
            f"Contents:\n```{halt_contents}```\n"
            f"Delete `{HALT_FLAG_PATH}` to resume."
        )
        print(f"\n  [HALT] Flag file present — exiting.\n{halt_contents}\n")
        sys.exit(1)

    # Layer 1: market data freshness gate
    pipeline_start_time = datetime.now(timezone.utc)
    print("  Checking market data freshness...")
    is_fresh, reason, failed = check_market_data_freshness(pipeline_start_time)
    if not is_fresh:
        from signal_logger import log_signal
        log_signal(
            ticker="PIPELINE",
            model_signal=STALE_MARKET_DATA,
            price=0,
            qty=0,
            confidence=0,
            actual_action=STALE_MARKET_DATA,
        )
        files_preview = ", ".join(failed[:5])
        if len(failed) > 5:
            files_preview += f" and {len(failed) - 5} more"
        send_discord(
            f"🚨 **PIPELINE HALTED — STALE MARKET DATA**\n"
            f"Reason: {reason}\n"
            f"Affected files: {files_preview}\n"
            f"Pipeline start: {pipeline_start_time.isoformat()}\n"
            f"No trades placed."
        )
        print(f"\n  [HALT] Market data freshness check failed: {reason}")
        sys.exit(1)
    print("  Market data freshness check passed.\n")

    # --- Top-level infra-error handler (Audit Finding #4) ---
    # Every Alpaca API call below classifies exceptions as infra vs. per-ticker.
    # Infra errors (5xx, 429, auth, network) raise AlpacaInfraError, which
    # propagates here for a structured halt + Discord alert.
    try:
        _run_execution(api)
    except AlpacaInfraError as exc:
        from signal_logger import log_signal as _log_infra
        _log_infra("PIPELINE", "ALPACA_INFRA", 0, 0, 0, ALPACA_INFRA_HALT)
        send_discord(
            f"🚨 **PIPELINE HALTED — ALPACA INFRASTRUCTURE ERROR**\n"
            f"Operation: {exc.operation}\n"
            f"Ticker: {exc.ticker}\n"
            f"Error: {exc.cause}\n"
            f"Session halted to avoid blind trading. Manual review required."
        )
        print(f"\n  [INFRA HALT] {exc}")
        sys.exit(1)
    except Exception as exc:
        # Safety net: any unclassified exception still gets a Discord alert
        # rather than dying with only a stderr traceback.
        send_discord(
            f"🚨 **PIPELINE CRASHED — UNHANDLED ERROR**\n"
            f"Error: {type(exc).__name__}: {exc}\n"
            f"Session terminated. Manual review required."
        )
        print(f"\n  [CRASH] Unhandled error: {exc}")
        raise


def _run_execution(api: tradeapi.REST) -> None:
    """Inner execution body of run(), wrapped by AlpacaInfraError handler."""
    # 3a. Portfolio-level loss limit — checked after freshness so we trust the
    # equity read. On trip: write halt flag, alert, continue (SELLs still run).
    halt_active = False
    current_equity = float(api.get_account().equity)

    should_halt, halt_reason = check_portfolio_loss_limits(api)
    if should_halt:
        halt_active = True
        _write_halt_flag(halt_reason)
        action_code = (
            PORTFOLIO_HALT_SINGLE_DAY
            if "Single-day" in halt_reason
            else PORTFOLIO_HALT_ROLLING
        )
        from signal_logger import log_signal as _log_halt
        _log_halt("PIPELINE", action_code, 0, 0, 0, action_code)
        send_discord(
            f"🚨 **PORTFOLIO HALT TRIGGERED**\n{halt_reason}\n"
            f"BUYs and rebalancer DISABLED for this session. SELLs and take-profit still active.\n"
            f"Flag written to `{HALT_FLAG_PATH}` — delete manually to resume."
        )
        print(f"\n  [HALT] Portfolio loss limit tripped: {halt_reason}\n")
        # continue — do NOT sys.exit. SELLs still run.

    # 3b. Peak-to-current drawdown — catches slow bleeds the rolling window misses.
    if not halt_active:
        peak_halt, peak_reason = check_peak_drawdown(current_equity)
        if peak_halt:
            halt_active = True
            _write_halt_flag(peak_reason)
            from signal_logger import log_signal as _log_peak
            _log_peak("PIPELINE", PORTFOLIO_HALT_PEAK_DRAWDOWN, 0, 0, 0,
                      PORTFOLIO_HALT_PEAK_DRAWDOWN)
            send_discord(
                f"🚨 **PEAK DRAWDOWN HALT TRIGGERED**\n{peak_reason}\n"
                f"BUYs and rebalancer DISABLED for this session. SELLs and take-profit still active.\n"
                f"Flag written to `{HALT_FLAG_PATH}` — delete manually to resume.\n"
                f"To reset the high-water mark after review, delete `{PEAK_EQUITY_PATH}`."
            )
            print(f"\n  [HALT] Peak drawdown tripped: {peak_reason}\n")
    else:
        # Still update the peak tracker even when halted by another tier,
        # so the high-water mark stays current.
        check_peak_drawdown(current_equity)

    # 3. Backfill + take-profit pass — standing stop-loss is enforced via OTO orders
    #    attached at entry time; take-profit is still evaluated in-script.
    owned  = get_owned_tickers(api)
    equity = get_equity(api)

    # Backfill: attach a standing GTC stop-loss to any existing position that lacks one.
    # Runs once per session at startup.
    print("  Backfilling standing stops for existing positions...")
    from signal_logger import log_signal as _log_backfill
    positions = api.list_positions()
    for position in positions:
        ticker      = position.symbol
        entry_price = float(position.avg_entry_price)
        qty         = float(position.qty)

        try:
            existing = api.list_orders(status="open", symbols=[ticker])
        except Exception as exc:
            print(f"    [BACKFILL ERROR] {ticker} — could not list open orders: {exc}")
            _log_backfill(ticker, "STOP_BACKFILL_FAILED", entry_price, qty, 0.0, STOP_BACKFILL_FAILED)
            send_discord(f"⚠️ **Stop-loss backfill failed** for {ticker}: {exc}")
            continue

        has_stop = any(
            o.side == "sell" and o.type in ("stop", "stop_limit")
            for o in existing
        )
        if has_stop:
            continue

        stop_price = round(entry_price * (1.0 - STOP_LOSS_PCT), 2)
        try:
            api.submit_order(
                symbol        = ticker,
                qty           = qty,
                side          = "sell",
                type          = "stop",
                stop_price    = stop_price,
                time_in_force = "gtc",
            )
            print(f"    [BACKFILL] {ticker} standing stop @ ${stop_price:.2f} (entry ${entry_price:.2f})")
            _log_backfill(ticker, "STOP_BACKFILL", entry_price, qty, 0.0, STOP_BACKFILL)
        except Exception as exc:
            print(f"    [BACKFILL ERROR] {ticker} — {exc}")
            _log_backfill(ticker, "STOP_BACKFILL_FAILED", entry_price, qty, 0.0, STOP_BACKFILL_FAILED)
            send_discord(f"⚠️ **Stop-loss backfill failed** for {ticker}: {exc}")

    print(f"  Checking take-profits on {len(owned)} open position(s)...")
    guard_stamped = False                       # stamp on first confirmed fill
    exited, owned = check_position_limits(api, owned)
    if exited:
        print(f"  Exited (take-profit): {exited}")
        _write_run_guard()
        guard_stamped = True
        owned  = get_owned_tickers(api)
        equity = get_equity(api)
    else:
        print("  No take-profits triggered.")

    # 4. Get today's signals
    signals = get_signals()
    if not signals:
        print("  No signals generated. Exiting.")
        sys.exit(0)

    # Separate and log stale-data skips before the main execution pass
    from signal_logger import log_signal as _log_signal_early
    _STALE_FINAL_SIGNALS = ("STALE_SKIP", STALE_MARKET_DATA_SKIP)
    stale_sigs = [s for s in signals if s["final_signal"] in _STALE_FINAL_SIGNALS]
    signals    = [s for s in signals if s["final_signal"] not in _STALE_FINAL_SIGNALS]
    for r in stale_sigs:
        print(f"  [{r['final_signal']}] {r['ticker']} ({r['age_str']}) logged")
        _log_signal_early(r["ticker"], r["final_signal"], r["current_price"], 0, 0.0, r["final_signal"])

    # Separate and log signal errors (per-ticker prediction failures)
    error_sigs = [s for s in signals if s["final_signal"] == SIGNAL_ERROR]
    signals    = [s for s in signals if s["final_signal"] != SIGNAL_ERROR]
    for r in error_sigs:
        print(f"  [SIGNAL_ERROR] {r['ticker']} — {r['error']} — logged")
        _log_signal_early(r["ticker"], SIGNAL_ERROR, r["current_price"], 0, 0.0, SIGNAL_ERROR)

    if not signals and error_sigs:
        failed_tickers = ", ".join(r["ticker"] for r in error_sigs)
        send_discord(
            f"🚨 **[SIGNAL_ERROR]** All {len(error_sigs)} ticker(s) failed during "
            f"signal generation — no trades possible this session.\n"
            f"Failed: {failed_tickers}\n"
            f"Check prediction pipeline (CSVs, features, model file)."
        )
        print(f"  [FATAL] All tickers failed signal generation: {failed_tickers}")
        sys.exit(1)

    # 4b. Agent veto layer (permissive — missing file = ABSTAIN all)
    agent_decisions = _load_agent_decisions()
    for r in signals:
        if r["final_signal"] in ("BUY", "SELL") and agent_decisions.get(r["ticker"]) == "VETO":
            print(f"  [{AGENT_VETO}] {r['ticker']} {r['final_signal']} → HOLD (agent news veto)")
            send_discord(
                f"\N{NO ENTRY} **Agent Veto:** {r['ticker']} {today_utc()}  "
                f"Model said: {r['final_signal']}  "
                f"Agent decision: VETO  "
                f"Action: HOLD"
            )
            r["final_signal"] = "HOLD"
            r["veto_reason"] = AGENT_VETO

    # 5. Sort: SELLs first (confidence desc), then BUYs (confidence desc), then HOLDs
    sell_sigs = sorted(
        [s for s in signals if s["final_signal"] == "SELL"],
        key=lambda s: s.get("confidence", 0.0),
        reverse=True,
    )
    buy_sigs = sorted(
        [s for s in signals if s["final_signal"] == "BUY"],
        key=lambda s: s.get("confidence", 0.0),
        reverse=True,
    )
    hold_sigs = [s for s in signals if s["final_signal"] == "HOLD"]

    print(
        f"\n  Signal queue: {len(sell_sigs)} SELL | "
        f"{len(buy_sigs)} BUY | {len(hold_sigs)} HOLD"
    )
    print("-" * 60)

    from signal_logger import log_signal

    shap_by_signal      = {"BUY": [], "HOLD": [], "SELL": []}
    outcomes            = []
    rebalancer_outcomes = []

    # 6a. SELL pass — process all SELL signals first
    for r in sell_sigs:
        ticker     = r["ticker"]
        price      = r["current_price"]
        confidence = r.get("confidence", 0.0)
        note_str   = f" ({r['note']})" if r.get("note") else ""

        if r.get("shap_values"):
            shap_by_signal["SELL"].append(r["shap_values"])

        owned = get_owned_tickers(api)

        if ticker not in owned:
            print(f"  Signal: SELL {ticker}{note_str} — skipped (not owned)")
            log_signal(ticker, "SELL", price, 0, confidence, "NOT_OWNED")
            outcomes.append({
                "ticker": ticker, "action": "SELL", "qty": 0, "price": price,
                "status": "skipped", "order_id": "", "reason": "not owned",
            })
            continue

        qty = owned[ticker]

        cancel_ok, _ = cancel_standing_stops(api, ticker)
        if not cancel_ok:
            send_discord(
                f"🚨 **[CANCEL_STOP_FAILED]** {ticker} — could not cancel standing stop; "
                f"SELL skipped to avoid conflict. Manual review required."
            )
            log_signal(ticker, "SELL", price, 0, confidence, CANCEL_STOP_FAILED)
            outcomes.append({
                "ticker": ticker, "action": "SELL", "qty": 0, "price": price,
                "status": "skipped", "order_id": "", "reason": "CANCEL_STOP_FAILED",
            })
            continue

        # Look up avg entry price from Alpaca BEFORE submitting the sell,
        # because the position vanishes from api.get_position() after a full
        # liquidation. If the lookup fails, default to price (the signal
        # price) so realized_pnl degrades to zero rather than crashing.
        try:
            position_before_sell = api.get_position(ticker)
            entry_price_for_exit = float(position_before_sell.avg_entry_price)
        except Exception as exc:
            _raise_if_infra(exc, "get_position", ticker)
            print(f"  [WARN] Could not fetch avg_entry_price for {ticker}: {exc} — using signal price as fallback")
            entry_price_for_exit = float(price)

        result = place_sell(api, ticker, qty)
        actual_qty = float(result.get("filled_qty") or qty)
        if result["status"] == "filled":
            actual_action = "SELL"
        elif result["status"] == "unfilled":
            actual_action = SELL_UNFILLED
        else:
            actual_action = "SELL_ERROR"
        log_signal(ticker, "SELL", price, actual_qty, confidence, actual_action, shap_values=r.get("shap_values"))

        if result["status"] == "filled":
            if not guard_stamped:
                _write_run_guard()
                guard_stamped = True
            from signal_logger import log_exit, find_open_entry_order_id
            entry_order_id = find_open_entry_order_id(ticker) or "UNLINKED"
            log_exit(
                ticker         = ticker,
                entry_order_id = entry_order_id,
                entry_price    = entry_price_for_exit,
                exit_price     = price,
                exit_reason    = "MODEL_SELL",
                shares         = actual_qty,
            )

        outcomes.append({
            "ticker":   ticker,
            "action":   "SELL",
            "qty":      actual_qty,
            "price":    price,
            "status":   result["status"],
            "order_id": result.get("order_id", ""),
            "reason":   result.get("reason", ""),
        })
        if result["status"] == "filled":
            equity = get_equity(api)
        # Error → logged and recorded; loop continues to next signal

    # 6b. Refresh after SELLs, then run rebalancer, then refresh again

    # Build skip list for the rebalancer: tickers where a model SELL filled
    sell_executed_tickers = [
        o["ticker"] for o in outcomes
        if o["status"] == "filled" and o["action"] == "SELL"
    ]

    owned  = get_owned_tickers(api)
    equity = get_equity(api)

    if halt_active:
        print("\n  [HALT] Rebalancer skipped — portfolio halt active.")
        log_signal("PIPELINE", "REBALANCER", 0, 0, 0, REBALANCER_SKIPPED_HALT)
        rebalancer_outcomes = []
    else:
        print("\n" + "-" * 60)
        print("  Running rebalancer...")
        rebalancer_outcomes = run_rebalancer(api, sell_executed_tickers)

    # Tickers the rebalancer successfully trimmed — skip in the BUY pass to
    # avoid immediately re-buying into a position that was just reduced
    rebalancer_tickers = [
        o["ticker"] for o in rebalancer_outcomes
        if o["status"] == "placed"
    ]

    if not guard_stamped and rebalancer_tickers:
        _write_run_guard()
        guard_stamped = True

    owned  = get_owned_tickers(api)
    equity = get_equity(api)
    print("-" * 60 + "\n")

    # 6c. BUY + HOLD pass
    cooldown_tickers = get_cooldown_tickers()

    for r in buy_sigs + hold_sigs:
        ticker     = r["ticker"]
        final_sig  = r["final_signal"]
        price      = r["current_price"]
        confidence = r.get("confidence", 0.0)
        note_str   = f" ({r['note']})" if r.get("note") else ""

        # ------------------------------------------------------------------
        # BUY
        # ------------------------------------------------------------------
        if final_sig == "BUY":
            if halt_active:
                print(f"  Signal: BUY {ticker}{note_str} — skipped (halt active)")
                log_signal(ticker, "BUY", price, 0, confidence, BUY_SKIPPED_HALT)
                outcomes.append({
                    "ticker": ticker, "action": "BUY", "qty": 0, "price": price,
                    "status": "skipped", "order_id": "", "reason": "BUY_SKIPPED_HALT",
                })
                continue

            # Skip tickers that were exited via stop-loss/take-profit this session
            if ticker in exited:
                print(f"  Signal: BUY {ticker}{note_str} — skipped (EXIT_SKIP)")
                log_signal(ticker, "BUY", price, 0, confidence, "EXIT_SKIP")
                outcomes.append({
                    "ticker": ticker, "action": "BUY", "qty": 0, "price": price,
                    "status": "skipped", "order_id": "", "reason": "EXIT_SKIP",
                })
                continue

            # Skip if a STOP_LOSS_SELL was logged for this ticker within the past 7 calendar days
            if ticker in cooldown_tickers:
                print(f"  Signal: BUY {ticker}{note_str} — skipped (COOLDOWN_SKIP)")
                log_signal(ticker, "BUY", price, 0, confidence, "COOLDOWN_SKIP")
                outcomes.append({
                    "ticker": ticker, "action": "BUY", "qty": 0, "price": price,
                    "status": "skipped", "order_id": "", "reason": "COOLDOWN_SKIP",
                })
                continue

            # Skip if the rebalancer trimmed this ticker in the current session
            if ticker in rebalancer_tickers:
                print(f"  Signal: BUY {ticker}{note_str} — skipped (REBALANCER_TICKERS_SKIP)")
                log_signal(ticker, "BUY", price, 0, confidence, REBALANCER_TICKERS_SKIP)
                outcomes.append({
                    "ticker": ticker, "action": "BUY", "qty": 0, "price": price,
                    "status": "skipped", "order_id": "", "reason": REBALANCER_TICKERS_SKIP,
                })
                continue

            if r.get("shap_values"):
                shap_by_signal["BUY"].append(r["shap_values"])

            owned  = get_owned_tickers(api)
            equity = get_equity(api)

            try:
                bar        = api.get_latest_trade(ticker)
                live_price = float(bar.price)
            except Exception as exc:
                _raise_if_infra(exc, "get_latest_trade", ticker)
                print(f"  [SKIP] {ticker} — could not fetch live price: {exc}")
                log_signal(ticker, "BUY", price, 0, confidence, "PRICE_FETCH_ERROR")
                outcomes.append({
                    "ticker": ticker, "action": "BUY", "qty": 0, "price": price,
                    "status": "skipped", "order_id": "", "reason": "PRICE_FETCH_ERROR",
                })
                continue

            if ticker in owned:
                # Already owned — ask capital_allocator with fresh state
                alloc = check_add_to_position(
                    ticker                = ticker,
                    confidence_normalized = confidence / 100.0,
                    price                 = live_price,
                    shares_owned          = owned[ticker],
                    portfolio_value       = equity,
                )
                skip_reason = alloc["skip_reason"]

                if skip_reason == "INVALID_HEADROOM":
                    print(f"  Signal: BUY {ticker}{note_str} — skipped (INVALID_HEADROOM)")
                    log_signal(ticker, "BUY", price, 0, confidence, "INVALID_HEADROOM")
                    outcomes.append({
                        "ticker": ticker, "action": "BUY", "qty": 0, "price": price,
                        "status": "skipped", "order_id": "", "reason": "INVALID_HEADROOM",
                    })
                    continue

                if skip_reason:
                    actual_skip = (
                        "CONFIDENCE_SKIP" if skip_reason == "CONFIDENCE_SKIP"
                        else INSUFFICIENT_EQUITY
                    )
                    print(f"  Signal: BUY {ticker}{note_str} — skipped ({skip_reason})")
                    log_signal(ticker, "BUY", price, 0, confidence, actual_skip)
                    outcomes.append({
                        "ticker": ticker, "action": "BUY", "qty": 0, "price": price,
                        "status": "skipped", "order_id": "", "reason": skip_reason,
                    })
                    continue

                qty    = alloc["shares_to_buy"]
                is_add = True

            else:
                # New position
                fraction = get_position_size(confidence)
                if fraction is None:
                    print(f"  Signal: BUY {ticker}{note_str} — skipped (below confidence threshold)")
                    log_signal(ticker, "BUY", price, 0, confidence, "CONFIDENCE_SKIP")
                    outcomes.append({
                        "ticker": ticker, "action": "BUY", "qty": 0, "price": price,
                        "status": "skipped", "order_id": "", "reason": "below confidence threshold",
                    })
                    continue

                qty = compute_buy_qty(equity, live_price, fraction)
                if qty <= 0:
                    print(f"  Signal: BUY {ticker}{note_str} — skipped (insufficient equity)")
                    log_signal(ticker, "BUY", price, 0, confidence, INSUFFICIENT_EQUITY)
                    outcomes.append({
                        "ticker": ticker, "action": "BUY", "qty": 0, "price": price,
                        "status": "skipped", "order_id": "", "reason": "insufficient equity",
                    })
                    continue

                is_add = False

            label = "add to position" if is_add else "new position"
            print(f"  Signal: BUY {ticker}{note_str} @ ${price:.2f} ({label})")
            result = place_buy(api, ticker, qty)
            actual_qty = int(result.get("filled_qty") or qty)
            if result["status"] == "filled":
                if not guard_stamped:
                    _write_run_guard()
                    guard_stamped = True
                actual_action = "ADD_TO_POSITION" if is_add else "BUY"
            elif result["status"] == "unfilled":
                actual_action = BUY_UNFILLED
            else:
                actual_action = "BUY_ERROR"
            log_signal(
                ticker, "BUY", price, actual_qty, confidence, actual_action,
                shap_values=r.get("shap_values"),
                entry_order_id=result.get("order_id") if result["status"] == "filled" else None,
            )
            outcomes.append({
                "ticker":   ticker,
                "action":   "BUY",
                "qty":      actual_qty,
                "price":    price,
                "status":   result["status"],
                "order_id": result.get("order_id", ""),
                "reason":   result.get("reason", ""),
            })

        # ------------------------------------------------------------------
        # HOLD
        # ------------------------------------------------------------------
        else:
            if r.get("shap_values"):
                shap_by_signal["HOLD"].append(r["shap_values"])

            print(f"  Signal: HOLD {ticker}{note_str} — no action")
            log_signal(ticker, "HOLD", price, 0, confidence, r.get("veto_reason") or "HOLD", shap_values=r.get("shap_values"))
            outcomes.append({
                "ticker":   ticker,
                "action":   "HOLD",
                "qty":      0,
                "price":    price,
                "status":   "skipped",
                "order_id": "",
                "reason":   "HOLD signal",
            })

    print("-" * 60)

    # 6d. Stamp the daily run guard — if no fills occurred above (all signals
    #     were skips/holds/errors), stamp now so a safety-net run still won't
    #     re-enter the execution pass. When at least one fill happened the guard
    #     was already stamped inline, making this a harmless no-op (idempotent).
    if not guard_stamped:
        _write_run_guard()

    # 7. Single post-execution Discord summary (after all trades complete)
    portfolio_value = get_equity(api)
    timestamp_end   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    send_discord(build_post_order_alert(outcomes, portfolio_value, timestamp_end, rebalancer_outcomes, shap_by_signal, stale_skips=[{"ticker": r["ticker"], "age_str": r["age_str"]} for r in stale_sigs]))

    # 8. Persist today's equity to the portfolio snapshot (always, even on halt).
    final_equity = get_equity(api)
    today_str    = today_utc()
    snapshot     = _read_portfolio_snapshot()
    snapshot[today_str] = final_equity
    _write_portfolio_snapshot(snapshot)
    print(f"  Snapshot updated: {today_str} = ${final_equity:,.2f}")

    print(f"\n  Portfolio value: ${portfolio_value:,.2f}")
    print("  Done.\n")


if __name__ == "__main__":
    run()
