"""Tests for live_trader.py — the execution layer.

Covers the risk controls that stand between a model signal and a real order:
the portfolio circuit breaker, the halt flag, the daily run guard, the market
data freshness gate, standing-stop cancellation, position sizing, and the
stop-loss child attached to every BUY.

Every test here works against mocked Alpaca clients and tmp_path-scoped state
files. Nothing in this module reaches the network.
"""

import csv
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, call, patch

import pytest

import config
import live_trader
import signal_logger
from live_trader import (
    _read_portfolio_snapshot,
    _write_portfolio_snapshot,
    _previous_session_date,
    _session_n_back,
    _write_halt_flag,
    _check_halt_flag,
    _write_run_guard,
    check_portfolio_loss_limits,
    check_market_data_freshness,
    cancel_standing_stops,
    compute_buy_qty,
    get_position_size,
    get_cooldown_tickers,
    get_owned_tickers,
    get_equity,
    place_buy,
    place_sell,
    read_last_close,
)


# ---------------------------------------------------------------------------
# Portfolio snapshot helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def snapshot_path(monkeypatch, tmp_path):
    path = tmp_path / "portfolio_snapshot.json"
    monkeypatch.setattr(live_trader, "PORTFOLIO_SNAPSHOT_PATH", str(path))
    return path


def test_missing_snapshot_reads_as_empty(snapshot_path):
    """First-run bootstrap must succeed with no baseline on disk."""
    assert _read_portfolio_snapshot() == {}


def test_unreadable_snapshot_reads_as_empty(snapshot_path):
    snapshot_path.write_text("{ not valid json")
    assert _read_portfolio_snapshot() == {}


def test_non_dict_snapshot_reads_as_empty(snapshot_path):
    snapshot_path.write_text('["a", "b"]')
    assert _read_portfolio_snapshot() == {}


def test_snapshot_round_trip(snapshot_path):
    _write_portfolio_snapshot({"2026-09-01": 10_000.0, "2026-09-02": 10_500.0})
    assert _read_portfolio_snapshot() == {
        "2026-09-01": 10_000.0,
        "2026-09-02": 10_500.0,
    }


def test_snapshot_pruned_to_retention_limit(snapshot_path):
    retain = config.PORTFOLIO_SNAPSHOT_RETAIN_DAYS
    entries = {
        (date(2026, 1, 1) + timedelta(days=i)).isoformat(): float(1000 + i)
        for i in range(retain + 15)
    }

    _write_portfolio_snapshot(entries)
    result = _read_portfolio_snapshot()

    assert len(result) == retain
    # The most recent entries survive, the oldest are dropped.
    assert max(result) == max(entries)
    assert min(result) == sorted(entries)[15]


def test_snapshot_write_failure_is_not_fatal(monkeypatch, tmp_path):
    monkeypatch.setattr(
        live_trader,
        "PORTFOLIO_SNAPSHOT_PATH",
        str(tmp_path / "no_such_dir" / "snap.json"),
    )
    _write_portfolio_snapshot({"2026-09-01": 100.0})   # must not raise


# ---------------------------------------------------------------------------
# Session lookups
# ---------------------------------------------------------------------------


SESSIONS = {
    "2026-09-01": 10_000.0,
    "2026-09-02": 10_100.0,
    "2026-09-03": 10_200.0,
    "2026-09-04": 10_300.0,
    "2026-09-08": 10_400.0,   # Mon after a weekend gap
}


def test_previous_session_skips_the_weekend_gap():
    assert _previous_session_date(SESSIONS, "2026-09-08") == "2026-09-04"


def test_previous_session_is_strictly_earlier():
    """An entry already written for today must not be its own baseline."""
    assert _previous_session_date(SESSIONS, "2026-09-04") == "2026-09-03"


def test_previous_session_none_when_no_history():
    assert _previous_session_date({}, "2026-09-08") is None
    assert _previous_session_date(SESSIONS, "2026-08-01") is None


def test_session_n_back_counts_entries_not_calendar_days():
    # Five sessions exist before 2026-09-09; 4 back from 09-08 is 09-01.
    assert _session_n_back(SESSIONS, "2026-09-08", 4) == "2026-09-01"
    assert _session_n_back(SESSIONS, "2026-09-08", 1) == "2026-09-04"


def test_session_n_back_none_when_history_too_short():
    assert _session_n_back(SESSIONS, "2026-09-08", 10) is None
    assert _session_n_back({}, "2026-09-08", 1) is None


# ---------------------------------------------------------------------------
# Portfolio circuit breaker
# ---------------------------------------------------------------------------


def _api_with_equity(equity: float):
    api = MagicMock()
    api.get_account.return_value.equity = str(equity)
    return api


@pytest.fixture
def breaker(monkeypatch, snapshot_path):
    """Freeze 'today' so snapshot dates are deterministic."""
    monkeypatch.setattr(live_trader, "today_utc", lambda: "2026-09-08")
    return snapshot_path


def test_no_halt_on_first_run_with_no_baseline(breaker):
    should_halt, reason = check_portfolio_loss_limits(_api_with_equity(10_000.0))
    assert should_halt is False
    assert reason == ""


def test_single_day_loss_beyond_threshold_halts(breaker):
    _write_portfolio_snapshot({"2026-09-04": 10_000.0})
    over = config.MAX_SINGLE_DAY_LOSS_PCT + 0.01
    equity = 10_000.0 * (1 - over)

    should_halt, reason = check_portfolio_loss_limits(_api_with_equity(equity))

    assert should_halt is True
    assert "Single-day loss" in reason


def test_single_day_loss_exactly_at_threshold_does_not_halt(breaker):
    """The comparison is `>`, so an exact-threshold loss must pass."""
    _write_portfolio_snapshot({"2026-09-04": 10_000.0})
    equity = 10_000.0 * (1 - config.MAX_SINGLE_DAY_LOSS_PCT)

    should_halt, _ = check_portfolio_loss_limits(_api_with_equity(equity))

    assert should_halt is False


def test_gain_never_halts(breaker):
    _write_portfolio_snapshot({"2026-09-04": 10_000.0})
    should_halt, _ = check_portfolio_loss_limits(_api_with_equity(12_000.0))
    assert should_halt is False


def test_rolling_loss_beyond_threshold_halts(breaker):
    """Slow bleed: every single day stays under the daily limit, but the
    cumulative drop over the window exceeds the rolling limit.

    The ramp matters — an abrupt drop would trip the single-day tier first,
    which is checked before the rolling tier.
    """
    _write_portfolio_snapshot({
        "2026-09-01": 10_000.0,   # 5 sessions back
        "2026-09-02": 9_800.0,
        "2026-09-03": 9_600.0,
        "2026-09-04": 9_400.0,
        "2026-09-05": 9_200.0,    # previous session
    })
    equity = 9_100.0   # 1.1% vs yesterday (under 5%), 9.0% vs 5 sessions (over 8%)

    should_halt, reason = check_portfolio_loss_limits(_api_with_equity(equity))

    assert should_halt is True
    assert "Rolling" in reason


def test_rolling_check_skipped_when_history_shorter_than_window(breaker):
    """With too little history the rolling tier must not fire off a partial
    window — comparing against the oldest available entry here would halt.
    """
    _write_portfolio_snapshot({
        "2026-09-04": 10_000.0,
        "2026-09-05": 9_150.0,
    })
    equity = 9_100.0   # 0.5% vs yesterday, but 9.0% vs the oldest entry

    should_halt, _ = check_portfolio_loss_limits(_api_with_equity(equity))

    assert should_halt is False


def test_zero_baseline_equity_does_not_divide_by_zero(breaker):
    _write_portfolio_snapshot({"2026-09-04": 0.0})
    should_halt, _ = check_portfolio_loss_limits(_api_with_equity(5_000.0))
    assert should_halt is False


# ---------------------------------------------------------------------------
# Halt flag
# ---------------------------------------------------------------------------


@pytest.fixture
def halt_path(monkeypatch, tmp_path):
    path = tmp_path / "HALT_FLAG.txt"
    monkeypatch.setattr(live_trader, "HALT_FLAG_PATH", str(path))
    return path


def test_absent_halt_flag_reports_clear(halt_path):
    assert _check_halt_flag() == (False, "")


def test_halt_flag_round_trip_records_reason(halt_path):
    _write_halt_flag("Single-day loss 7.00% exceeds 5.0% threshold")

    present, contents = _check_halt_flag()

    assert present is True
    assert "Single-day loss 7.00%" in contents
    assert "timestamp_utc:" in contents


def test_halt_flag_write_failure_is_not_fatal(monkeypatch, tmp_path):
    monkeypatch.setattr(
        live_trader, "HALT_FLAG_PATH", str(tmp_path / "missing_dir" / "HALT.txt")
    )
    _write_halt_flag("reason")   # must not raise
    assert _check_halt_flag() == (False, "")


# ---------------------------------------------------------------------------
# Daily run guard
# ---------------------------------------------------------------------------


def test_run_guard_stamps_today_utc(monkeypatch, tmp_path):
    guard = tmp_path / "data" / "last_run_date.txt"
    monkeypatch.setattr(live_trader, "LAST_RUN_GUARD_PATH", str(guard))
    monkeypatch.setattr(live_trader, "today_utc", lambda: "2026-09-08")

    _write_run_guard()

    assert guard.read_text() == "2026-09-08"


def test_run_guard_write_failure_alerts_but_does_not_raise(monkeypatch, tmp_path):
    """Trades are already placed by this point — aborting would help nobody."""
    guard = tmp_path / "guard.txt"
    guard.write_text("placeholder")
    monkeypatch.setattr(live_trader, "LAST_RUN_GUARD_PATH", str(guard))
    monkeypatch.setattr(live_trader, "today_utc", lambda: "2026-09-08")
    monkeypatch.setattr(
        live_trader.os, "makedirs", MagicMock(side_effect=OSError("read-only fs"))
    )

    with patch.object(live_trader, "send_discord") as discord:
        _write_run_guard()

    discord.assert_called_once()
    assert "Run guard write failed" in discord.call_args.args[0]


# ---------------------------------------------------------------------------
# Market data freshness gate (Layer 1)
# ---------------------------------------------------------------------------


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    """A DATA_DIR containing one ticker CSV and one market CSV."""
    root = tmp_path / "data"
    (root / "market").mkdir(parents=True)
    (root / "AAPL.csv").write_text("Date,Close\n2026-09-08,100.0\n")
    (root / "market" / "SPY.csv").write_text("Date,Close\n2026-09-08,500.0\n")
    monkeypatch.setattr(config, "DATA_DIR", str(root))
    monkeypatch.setattr(config, "WATCHLIST", ["AAPL"])
    return root


def _touch(path, when: datetime):
    ts = when.timestamp()
    os.utime(path, (ts, ts))


def test_fresh_files_pass_the_gate(data_dir):
    now = datetime.now(timezone.utc)
    written = now - timedelta(minutes=30)
    _touch(data_dir / "AAPL.csv", written)
    _touch(data_dir / "market" / "SPY.csv", written)

    is_fresh, reason, failed = check_market_data_freshness(now)

    assert is_fresh is True
    assert reason == ""
    assert failed == []


def test_missing_file_fails_the_gate(data_dir):
    (data_dir / "AAPL.csv").unlink()

    is_fresh, reason, failed = check_market_data_freshness(datetime.now(timezone.utc))

    assert is_fresh is False
    assert "AAPL.csv" in failed
    assert "missing" in reason


def test_file_written_before_today_fails_the_gate(data_dir):
    """Catches the silent-stale case: yesterday's refresh, today's run."""
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)
    _touch(data_dir / "AAPL.csv", yesterday)
    _touch(data_dir / "market" / "SPY.csv", now - timedelta(minutes=30))

    is_fresh, reason, failed = check_market_data_freshness(now)

    assert is_fresh is False
    assert "AAPL.csv" in failed
    assert "before today" in reason


def test_file_written_after_pipeline_start_fails_the_gate(data_dir):
    """The race condition this gate was written for: refresh still running."""
    now = datetime.now(timezone.utc)
    pipeline_start = now - timedelta(minutes=10)
    _touch(data_dir / "AAPL.csv", now)   # written after the pipeline began
    _touch(data_dir / "market" / "SPY.csv", pipeline_start - timedelta(minutes=5))

    is_fresh, reason, failed = check_market_data_freshness(pipeline_start)

    assert is_fresh is False
    assert "AAPL.csv" in failed
    assert "AFTER pipeline start" in reason


def test_failure_reason_is_truncated_with_a_count(data_dir):
    """Many failures must not produce an unbounded Discord message."""
    for i in range(6):
        (data_dir / "market" / f"ETF{i}.csv").write_text("Date,Close\n")
        _touch(data_dir / "market" / f"ETF{i}.csv", datetime.now(timezone.utc) - timedelta(days=2))
    _touch(data_dir / "AAPL.csv", datetime.now(timezone.utc) - timedelta(days=2))
    _touch(data_dir / "market" / "SPY.csv", datetime.now(timezone.utc) - timedelta(days=2))

    is_fresh, reason, failed = check_market_data_freshness(datetime.now(timezone.utc))

    assert is_fresh is False
    assert len(failed) == 8
    assert "more)" in reason


# ---------------------------------------------------------------------------
# Standing-stop cancellation
# ---------------------------------------------------------------------------


def _order(order_id: str, side: str = "sell", order_type: str = "stop", status: str = "new"):
    o = MagicMock()
    o.id = order_id
    o.side = side
    o.type = order_type
    o.status = status
    return o


def test_no_standing_stop_is_a_success(monkeypatch):
    api = MagicMock()
    api.list_orders.return_value = []

    assert cancel_standing_stops(api, "AAPL") == (True, [])
    api.cancel_order.assert_not_called()


def test_non_stop_orders_are_ignored(monkeypatch):
    api = MagicMock()
    api.list_orders.return_value = [
        _order("o1", side="buy", order_type="market"),
        _order("o2", side="sell", order_type="limit"),
    ]

    assert cancel_standing_stops(api, "AAPL") == (True, [])
    api.cancel_order.assert_not_called()


def test_standing_stop_cancelled_successfully(monkeypatch):
    monkeypatch.setattr(live_trader.time, "sleep", lambda _s: None)
    api = MagicMock()
    api.list_orders.return_value = [_order("stop-1")]
    api.get_order.return_value = _order("stop-1", status="canceled")

    success, cancelled = cancel_standing_stops(api, "AAPL")

    assert success is True
    assert cancelled == ["stop-1"]
    api.cancel_order.assert_called_once_with("stop-1")


def test_stop_that_fills_during_cancel_counts_as_success(monkeypatch):
    """The stop firing reaches the same end state: no standing order left."""
    monkeypatch.setattr(live_trader.time, "sleep", lambda _s: None)
    api = MagicMock()
    api.list_orders.return_value = [_order("stop-1")]
    api.get_order.return_value = _order("stop-1", status="filled")

    success, cancelled = cancel_standing_stops(api, "AAPL")

    assert success is True
    assert cancelled == ["stop-1"]


def test_list_orders_failure_blocks_the_sell(monkeypatch):
    api = MagicMock()
    api.list_orders.side_effect = Exception("alpaca timeout")

    assert cancel_standing_stops(api, "AAPL") == (False, [])


def test_cancel_request_failure_blocks_the_sell(monkeypatch):
    api = MagicMock()
    api.list_orders.return_value = [_order("stop-1")]
    api.cancel_order.side_effect = Exception("rejected")

    success, cancelled = cancel_standing_stops(api, "AAPL")

    assert success is False
    assert cancelled == []


def test_cancel_timeout_blocks_the_sell(monkeypatch):
    """A stop stuck in 'new' must never let a market SELL through."""
    monkeypatch.setattr(live_trader.time, "sleep", lambda _s: None)
    ticks = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0, 200.0])
    monkeypatch.setattr(live_trader.time, "time", lambda: next(ticks))

    api = MagicMock()
    api.list_orders.return_value = [_order("stop-1")]
    api.get_order.return_value = _order("stop-1", status="new")

    success, _ = cancel_standing_stops(api, "AAPL")

    assert success is False


def test_partial_failure_across_multiple_stops_blocks_the_sell(monkeypatch):
    monkeypatch.setattr(live_trader.time, "sleep", lambda _s: None)
    api = MagicMock()
    api.list_orders.return_value = [_order("stop-1"), _order("stop-2")]
    api.cancel_order.side_effect = [None, Exception("rejected")]
    api.get_order.return_value = _order("stop-1", status="canceled")

    success, cancelled = cancel_standing_stops(api, "AAPL")

    assert success is False
    assert cancelled == ["stop-1"]


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "confidence, expected",
    [
        (0.0, None),
        (34.99, None),
        (35.0, 0.10),
        (39.99, 0.10),
        (40.0, 0.15),
        (44.99, 0.15),
        (45.0, 0.20),
        (99.9, 0.20),
    ],
)
def test_position_size_tiers(confidence, expected):
    assert get_position_size(confidence) == expected


def test_compute_buy_qty_floors_to_whole_shares():
    assert compute_buy_qty(10_000.0, 100.0, 0.20) == 20
    assert compute_buy_qty(10_000.0, 300.0, 0.20) == 6   # 6.67 -> 6


def test_compute_buy_qty_never_negative():
    assert compute_buy_qty(100.0, 10_000.0, 0.20) == 0
    assert compute_buy_qty(0.0, 100.0, 0.20) == 0


# ---------------------------------------------------------------------------
# Order placement
# ---------------------------------------------------------------------------


def test_buy_attaches_a_stop_loss_child_at_the_configured_pct():
    """Every BUY must ship with standing downside protection."""
    api = MagicMock()
    api.get_latest_trade.return_value.price = 200.0
    api.submit_order.return_value.id = "order-1"

    result = place_buy(api, "AAPL", 10)

    assert result["status"] == "placed"
    kwargs = api.submit_order.call_args.kwargs
    assert kwargs["order_class"] == "oto"
    assert kwargs["time_in_force"] == "gtc"   # required for a standing child
    assert kwargs["side"] == "buy"
    expected_stop = round(200.0 * (1.0 - config.STOP_LOSS_PCT), 2)
    assert kwargs["stop_loss"] == {"stop_price": expected_stop}


@pytest.mark.parametrize("qty", [0, -5])
def test_buy_with_non_positive_qty_places_no_order(qty):
    api = MagicMock()

    result = place_buy(api, "AAPL", qty)

    assert result["status"] == "skipped"
    api.submit_order.assert_not_called()


def test_buy_error_is_captured_not_raised():
    api = MagicMock()
    api.get_latest_trade.return_value.price = 200.0
    api.submit_order.side_effect = Exception("insufficient buying power")

    result = place_buy(api, "AAPL", 10)

    assert result["status"] == "error"
    assert "insufficient buying power" in result["reason"]


def test_sell_submits_a_day_market_order():
    api = MagicMock()
    api.submit_order.return_value.id = "order-2"

    result = place_sell(api, "AAPL", 10)

    assert result == {"status": "placed", "order_id": "order-2"}
    kwargs = api.submit_order.call_args.kwargs
    assert kwargs["side"] == "sell"
    assert kwargs["type"] == "market"
    assert kwargs["time_in_force"] == "day"


def test_sell_error_is_captured_not_raised():
    api = MagicMock()
    api.submit_order.side_effect = Exception("position not found")

    result = place_sell(api, "AAPL", 10)

    assert result["status"] == "error"
    assert "position not found" in result["reason"]


# ---------------------------------------------------------------------------
# Portfolio reads
# ---------------------------------------------------------------------------


def test_owned_tickers_maps_symbol_to_qty():
    api = MagicMock()
    p1, p2 = MagicMock(), MagicMock()
    p1.symbol, p1.qty = "AAPL", "10"
    p2.symbol, p2.qty = "MSFT", "5.5"
    api.list_positions.return_value = [p1, p2]

    assert get_owned_tickers(api) == {"AAPL": 10.0, "MSFT": 5.5}


def test_equity_is_returned_as_float():
    api = MagicMock()
    api.get_account.return_value.equity = "12345.67"
    assert get_equity(api) == 12345.67


# ---------------------------------------------------------------------------
# Stop-loss cooldown
# ---------------------------------------------------------------------------


@pytest.fixture
def signal_log(monkeypatch, tmp_path):
    path = tmp_path / "signal_log.csv"
    monkeypatch.setattr(signal_logger, "SIGNAL_LOG_PATH", str(path))
    return path


def _write_log(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=signal_logger.FIELDNAMES)
        writer.writeheader()
        for row in rows:
            full = {name: "" for name in signal_logger.FIELDNAMES}
            full.update(row)
            writer.writerow(full)


def test_no_log_file_means_no_cooldowns(signal_log):
    assert get_cooldown_tickers() == set()


def test_recent_stop_loss_puts_ticker_in_cooldown(signal_log):
    recent = (date.today() - timedelta(days=1)).isoformat()
    _write_log(signal_log, [
        {"date": recent, "ticker": "AAPL", "actual_action": "STOP_LOSS_SELL"},
    ])

    assert get_cooldown_tickers() == {"AAPL"}


def test_stop_loss_older_than_the_window_expires(signal_log):
    old = (date.today() - timedelta(days=config.STOP_LOSS_COOLDOWN_DAYS + 1)).isoformat()
    _write_log(signal_log, [
        {"date": old, "ticker": "AAPL", "actual_action": "STOP_LOSS_SELL"},
    ])

    assert get_cooldown_tickers() == set()


def test_cooldown_boundary_day_is_inclusive(signal_log):
    """The check is `<= STOP_LOSS_COOLDOWN_DAYS`."""
    edge = (date.today() - timedelta(days=config.STOP_LOSS_COOLDOWN_DAYS)).isoformat()
    _write_log(signal_log, [
        {"date": edge, "ticker": "AAPL", "actual_action": "STOP_LOSS_SELL"},
    ])

    assert get_cooldown_tickers() == {"AAPL"}


def test_other_actions_do_not_trigger_cooldown(signal_log):
    recent = (date.today() - timedelta(days=1)).isoformat()
    _write_log(signal_log, [
        {"date": recent, "ticker": "AAPL", "actual_action": "SELL"},
        {"date": recent, "ticker": "MSFT", "actual_action": "HOLD"},
        {"date": recent, "ticker": "NVDA", "actual_action": "STOP_LOSS_SELL"},
    ])

    assert get_cooldown_tickers() == {"NVDA"}


def test_malformed_date_row_is_skipped_not_fatal(signal_log):
    recent = (date.today() - timedelta(days=1)).isoformat()
    _write_log(signal_log, [
        {"date": "not-a-date", "ticker": "AAPL", "actual_action": "STOP_LOSS_SELL"},
        {"date": recent, "ticker": "MSFT", "actual_action": "STOP_LOSS_SELL"},
    ])

    assert get_cooldown_tickers() == {"MSFT"}


# ---------------------------------------------------------------------------
# read_last_close
# ---------------------------------------------------------------------------


def test_read_last_close_returns_final_row(data_dir):
    (data_dir / "AAPL.csv").write_text(
        "Date,Close\n2026-09-05,100.0\n2026-09-08,123.45\n"
    )
    assert read_last_close("AAPL") == 123.45


def test_read_last_close_returns_none_when_missing(data_dir):
    assert read_last_close("NOSUCH") is None


def test_read_last_close_returns_none_on_malformed_csv(data_dir):
    (data_dir / "AAPL.csv").write_text("Date,NotClose\n2026-09-08,1\n")
    assert read_last_close("AAPL") is None
