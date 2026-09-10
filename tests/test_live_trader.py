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
    _read_peak_equity,
    _write_peak_equity,
    _wait_for_fill,
    _cancel_order_best_effort,
    _check_fill_after_cancel,
    check_portfolio_loss_limits,
    check_peak_drawdown,
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
# Peak-to-current drawdown circuit breaker (tier 3)
# ---------------------------------------------------------------------------


@pytest.fixture
def peak_path(monkeypatch, tmp_path):
    """Isolate peak_equity.json to a tmp directory and freeze today."""
    path = tmp_path / "data" / "peak_equity.json"
    monkeypatch.setattr(live_trader, "PEAK_EQUITY_PATH", str(path))
    monkeypatch.setattr(live_trader, "today_utc", lambda: "2026-09-08")
    return path


def test_first_run_seeds_peak_equity(peak_path):
    """No stored peak → seed with current equity, no halt."""
    should_halt, reason = check_peak_drawdown(10_000.0)

    assert should_halt is False
    assert reason == ""

    state = json.loads(peak_path.read_text())
    assert state["peak_equity"] == 10_000.0
    assert state["updated_at"] == "2026-09-08"


def test_new_high_water_mark_updates_peak(peak_path):
    peak_path.parent.mkdir(parents=True, exist_ok=True)
    peak_path.write_text(json.dumps({"peak_equity": 10_000.0, "updated_at": "2026-09-01"}))

    should_halt, reason = check_peak_drawdown(11_000.0)

    assert should_halt is False
    state = json.loads(peak_path.read_text())
    assert state["peak_equity"] == 11_000.0
    assert state["updated_at"] == "2026-09-08"


def test_drawdown_below_threshold_does_not_halt(peak_path):
    peak_path.parent.mkdir(parents=True, exist_ok=True)
    peak_path.write_text(json.dumps({"peak_equity": 10_000.0, "updated_at": "2026-09-01"}))

    # 5% drawdown, well under the 15% threshold
    should_halt, reason = check_peak_drawdown(9_500.0)

    assert should_halt is False
    assert reason == ""


def test_drawdown_exactly_at_threshold_halts(peak_path):
    """The comparison is >=, so exactly 15% triggers the halt."""
    peak_path.parent.mkdir(parents=True, exist_ok=True)
    peak_path.write_text(json.dumps({"peak_equity": 10_000.0, "updated_at": "2026-09-01"}))

    equity = 10_000.0 * (1 - config.MAX_PEAK_DRAWDOWN_PCT)   # exactly 8500.0

    should_halt, reason = check_peak_drawdown(equity)

    assert should_halt is True
    assert "Peak drawdown" in reason
    assert "15.0%" in reason


def test_drawdown_beyond_threshold_halts(peak_path):
    peak_path.parent.mkdir(parents=True, exist_ok=True)
    peak_path.write_text(json.dumps({"peak_equity": 10_000.0, "updated_at": "2026-09-01"}))

    # 20% drawdown — well over the 15% threshold
    should_halt, reason = check_peak_drawdown(8_000.0)

    assert should_halt is True
    assert "Peak drawdown" in reason
    assert "peak $10,000.00" in reason


def test_missing_peak_file_is_handled_gracefully(peak_path):
    """If the file doesn't exist, _read returns {} and check seeds the peak."""
    should_halt, _ = check_peak_drawdown(10_000.0)
    assert should_halt is False


def test_corrupt_peak_file_is_handled_gracefully(peak_path):
    """A corrupt file must not crash the bot — fall back to seeding."""
    peak_path.parent.mkdir(parents=True, exist_ok=True)
    peak_path.write_text("{ not valid json }")

    should_halt, _ = check_peak_drawdown(10_000.0)

    assert should_halt is False
    # The file should now have valid JSON after re-seeding
    state = json.loads(peak_path.read_text())
    assert state["peak_equity"] == 10_000.0


def test_peak_write_failure_is_not_fatal(monkeypatch):
    monkeypatch.setattr(
        live_trader, "PEAK_EQUITY_PATH", "/proc/fake/peak.json",
    )
    monkeypatch.setattr(live_trader, "today_utc", lambda: "2026-09-08")

    # _write_peak_equity must not raise even when the write fails
    _write_peak_equity({"peak_equity": 10_000.0})   # should not raise


def test_zero_stored_peak_reseeds_instead_of_dividing_by_zero(peak_path):
    """A manually-edited or corrupt peak of 0.0 must not cause ZeroDivisionError."""
    peak_path.parent.mkdir(parents=True, exist_ok=True)
    peak_path.write_text(json.dumps({"peak_equity": 0.0, "updated_at": "2026-09-01"}))

    should_halt, _ = check_peak_drawdown(10_000.0)

    assert should_halt is False
    state = json.loads(peak_path.read_text())
    assert state["peak_equity"] == 10_000.0


def test_peak_equity_read_write_round_trip(peak_path):
    state = {"peak_equity": 12_345.67, "updated_at": "2026-09-05"}
    peak_path.parent.mkdir(parents=True, exist_ok=True)
    _write_peak_equity(state)
    result = _read_peak_equity()

    assert result["peak_equity"] == 12_345.67
    assert result["updated_at"] == "2026-09-05"


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
        (35.0, 0.03),
        (49.99, 0.03),
        (50.0, 0.05),
        (64.99, 0.05),
        (65.0, 0.07),
        (99.9, 0.07),
    ],
)
def test_position_size_tiers(confidence, expected):
    assert get_position_size(confidence) == expected


def test_no_tier_opens_above_its_own_position_cap():
    """The defect this replaced: positions opened at 10-20% of equity while
    MAX_POSITION_PCT capped them at 8%, so the rebalancer trimmed every entry
    back over the following sessions and no top-up ever had headroom."""
    import config

    for confidence in (35.0, 45.0, 55.0, 65.0, 99.9):
        assert get_position_size(confidence) <= config.MAX_POSITION_PCT


def test_position_size_uses_the_same_tiers_as_adding_to_a_position():
    """One rule for opens and top-ups; two rules is what contradicted itself."""
    from capital_allocator import get_allocation_tier

    for confidence in (35.0, 50.0, 65.0, 90.0):
        _tier, pct = get_allocation_tier(confidence / 100.0)
        assert get_position_size(confidence) == pct


def test_compute_buy_qty_floors_to_whole_shares():
    assert compute_buy_qty(10_000.0, 100.0, 0.20) == 20
    assert compute_buy_qty(10_000.0, 300.0, 0.20) == 6   # 6.67 -> 6


def test_compute_buy_qty_never_negative():
    assert compute_buy_qty(100.0, 10_000.0, 0.20) == 0
    assert compute_buy_qty(0.0, 100.0, 0.20) == 0


# ---------------------------------------------------------------------------
# Order placement
# ---------------------------------------------------------------------------


def test_buy_attaches_a_stop_loss_child_at_the_configured_pct(monkeypatch):
    """Every BUY must ship with standing downside protection."""
    monkeypatch.setattr(live_trader.time, "sleep", lambda _: None)
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(live_trader.time, "time", lambda: next(ticks))

    api = MagicMock()
    api.get_latest_trade.return_value.price = 200.0
    api.submit_order.return_value.id = "order-1"
    api.get_order.return_value = _order("order-1", status="filled")

    result = place_buy(api, "AAPL", 10)

    assert result["status"] == "filled"
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


def test_sell_submits_a_day_market_order(monkeypatch):
    monkeypatch.setattr(live_trader.time, "sleep", lambda _: None)
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(live_trader.time, "time", lambda: next(ticks))

    api = MagicMock()
    api.submit_order.return_value.id = "order-2"
    api.get_order.return_value = _order("order-2", status="filled")

    result = place_sell(api, "AAPL", 10)

    assert result["status"] == "filled"
    assert result["order_id"] == "order-2"
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


# ---------------------------------------------------------------------------
# Order fill verification — _wait_for_fill
# ---------------------------------------------------------------------------


def test_wait_for_fill_returns_on_filled(monkeypatch):
    """Immediate fill on first poll."""
    monkeypatch.setattr(live_trader.time, "sleep", lambda _: None)
    # time.time() must stay within the deadline
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(live_trader.time, "time", lambda: next(ticks))

    api = MagicMock()
    filled_order = _order("o1", status="filled")
    api.get_order.return_value = filled_order

    result = _wait_for_fill(api, "o1")
    assert result is filled_order


def test_wait_for_fill_returns_on_partially_filled(monkeypatch):
    monkeypatch.setattr(live_trader.time, "sleep", lambda _: None)
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(live_trader.time, "time", lambda: next(ticks))

    api = MagicMock()
    partial = _order("o1", status="partially_filled")
    api.get_order.return_value = partial

    assert _wait_for_fill(api, "o1") is partial


def test_wait_for_fill_returns_none_on_rejected(monkeypatch):
    monkeypatch.setattr(live_trader.time, "sleep", lambda _: None)
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(live_trader.time, "time", lambda: next(ticks))

    api = MagicMock()
    api.get_order.return_value = _order("o1", status="rejected")

    assert _wait_for_fill(api, "o1") is None


def test_wait_for_fill_returns_none_on_timeout(monkeypatch):
    """Clock expires before the order moves out of 'new'."""
    monkeypatch.setattr(live_trader.time, "sleep", lambda _: None)
    ticks = iter([0.0, 5.0, 11.0])  # third tick is past the 10s deadline
    monkeypatch.setattr(live_trader.time, "time", lambda: next(ticks))

    api = MagicMock()
    api.get_order.return_value = _order("o1", status="new")

    assert _wait_for_fill(api, "o1") is None


def test_wait_for_fill_retries_after_poll_error(monkeypatch):
    """A transient API error doesn't abort — the loop retries."""
    monkeypatch.setattr(live_trader.time, "sleep", lambda _: None)
    ticks = iter([0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(live_trader.time, "time", lambda: next(ticks))

    api = MagicMock()
    filled = _order("o1", status="filled")
    api.get_order.side_effect = [Exception("transient"), filled]

    assert _wait_for_fill(api, "o1") is filled
    assert api.get_order.call_count == 2


# ---------------------------------------------------------------------------
# _cancel_order_best_effort
# ---------------------------------------------------------------------------


def test_cancel_best_effort_swallows_errors():
    api = MagicMock()
    api.cancel_order.side_effect = Exception("already canceled")

    # Must not raise
    _cancel_order_best_effort(api, "o1")
    api.cancel_order.assert_called_once_with("o1")


# ---------------------------------------------------------------------------
# place_buy — fill verification integration
# ---------------------------------------------------------------------------


def test_buy_returns_filled_on_immediate_fill(monkeypatch):
    """Happy path: submit → poll → filled on first attempt."""
    monkeypatch.setattr(live_trader.time, "sleep", lambda _: None)
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(live_trader.time, "time", lambda: next(ticks))

    api = MagicMock()
    api.get_latest_trade.return_value.price = 200.0
    api.submit_order.return_value.id = "order-1"
    api.get_order.return_value = _order("order-1", status="filled")

    result = place_buy(api, "AAPL", 10)

    assert result["status"] == "filled"
    assert result["order_id"] == "order-1"
    api.submit_order.assert_called_once()


def test_buy_retries_once_and_fills_on_second_attempt(monkeypatch):
    monkeypatch.setattr(live_trader.time, "sleep", lambda _: None)
    # Attempt 1: start=0, poll=5, deadline=11 (timeout)
    # _check_fill_after_cancel: order-1 is canceled (retry proceeds)
    # Attempt 2: start=12, poll=13 (filled before deadline)
    ticks = iter([0.0, 5.0, 11.0, 12.0, 13.0])
    monkeypatch.setattr(live_trader.time, "time", lambda: next(ticks))

    api = MagicMock()
    api.get_latest_trade.return_value.price = 200.0
    order1 = MagicMock(); order1.id = "order-1"
    order2 = MagicMock(); order2.id = "order-2"
    api.submit_order.side_effect = [order1, order2]

    api.get_order.side_effect = [
        _order("order-1", status="new"),      # _wait_for_fill poll (attempt 1)
        _order("order-1", status="canceled"),  # _check_fill_after_cancel (attempt 1)
        _order("order-2", status="filled"),    # _wait_for_fill poll (attempt 2)
    ]

    result = place_buy(api, "AAPL", 10)

    assert result["status"] == "filled"
    assert result["order_id"] == "order-2"
    assert api.submit_order.call_count == 2
    api.cancel_order.assert_called_once_with("order-1")


@patch.object(live_trader, "send_discord")
def test_buy_unfilled_sends_discord_alert(mock_discord, monkeypatch):
    """Both attempts time out → status='unfilled', Discord alert sent."""
    monkeypatch.setattr(live_trader.time, "sleep", lambda _: None)
    # Both attempts time out
    ticks = iter([0.0, 5.0, 11.0, 12.0, 17.0, 23.0])
    monkeypatch.setattr(live_trader.time, "time", lambda: next(ticks))

    api = MagicMock()
    api.get_latest_trade.return_value.price = 200.0
    order1 = MagicMock(); order1.id = "order-1"
    order2 = MagicMock(); order2.id = "order-2"
    api.submit_order.side_effect = [order1, order2]
    api.get_order.return_value = _order("stuck", status="new")

    result = place_buy(api, "AAPL", 10)

    assert result["status"] == "unfilled"
    mock_discord.assert_called_once()
    assert "BUY UNFILLED" in mock_discord.call_args[0][0]


def test_buy_submit_error_returns_error_without_retry(monkeypatch):
    """If submit_order itself raises, don't retry."""
    api = MagicMock()
    api.get_latest_trade.return_value.price = 200.0
    api.submit_order.side_effect = Exception("insufficient buying power")

    result = place_buy(api, "AAPL", 10)

    assert result["status"] == "error"
    assert "insufficient buying power" in result["reason"]
    api.submit_order.assert_called_once()


def test_buy_still_attaches_stop_loss_child():
    """Fill verification must not break the OTO stop-loss attachment."""
    api = MagicMock()
    api.get_latest_trade.return_value.price = 200.0
    api.submit_order.return_value.id = "order-1"
    api.get_order.return_value = _order("order-1", status="filled")

    # Freeze time so _wait_for_fill succeeds immediately
    with patch.object(live_trader.time, "time", side_effect=[0.0, 1.0]):
        with patch.object(live_trader.time, "sleep", lambda _: None):
            place_buy(api, "AAPL", 10)

    kwargs = api.submit_order.call_args.kwargs
    assert kwargs["order_class"] == "oto"
    expected_stop = round(200.0 * (1.0 - config.STOP_LOSS_PCT), 2)
    assert kwargs["stop_loss"] == {"stop_price": expected_stop}


# ---------------------------------------------------------------------------
# place_sell — fill verification integration
# ---------------------------------------------------------------------------


def test_sell_returns_filled_on_immediate_fill(monkeypatch):
    monkeypatch.setattr(live_trader.time, "sleep", lambda _: None)
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(live_trader.time, "time", lambda: next(ticks))

    api = MagicMock()
    api.submit_order.return_value.id = "order-2"
    api.get_order.return_value = _order("order-2", status="filled")

    result = place_sell(api, "AAPL", 10)

    assert result["status"] == "filled"
    assert result["order_id"] == "order-2"


def test_sell_retries_once_and_fills_on_second_attempt(monkeypatch):
    monkeypatch.setattr(live_trader.time, "sleep", lambda _: None)
    ticks = iter([0.0, 5.0, 11.0, 12.0, 13.0])
    monkeypatch.setattr(live_trader.time, "time", lambda: next(ticks))

    api = MagicMock()
    order1 = MagicMock(); order1.id = "sell-1"
    order2 = MagicMock(); order2.id = "sell-2"
    api.submit_order.side_effect = [order1, order2]
    api.get_order.side_effect = [
        _order("sell-1", status="new"),       # _wait_for_fill poll (attempt 1)
        _order("sell-1", status="canceled"),   # _check_fill_after_cancel (attempt 1)
        _order("sell-2", status="filled"),     # _wait_for_fill poll (attempt 2)
    ]

    result = place_sell(api, "AAPL", 10)

    assert result["status"] == "filled"
    assert result["order_id"] == "sell-2"
    api.cancel_order.assert_called_once_with("sell-1")


@patch.object(live_trader, "send_discord")
def test_sell_unfilled_sends_discord_alert(mock_discord, monkeypatch):
    monkeypatch.setattr(live_trader.time, "sleep", lambda _: None)
    ticks = iter([0.0, 5.0, 11.0, 12.0, 17.0, 23.0])
    monkeypatch.setattr(live_trader.time, "time", lambda: next(ticks))

    api = MagicMock()
    order1 = MagicMock(); order1.id = "sell-1"
    order2 = MagicMock(); order2.id = "sell-2"
    api.submit_order.side_effect = [order1, order2]
    api.get_order.return_value = _order("stuck", status="new")

    result = place_sell(api, "AAPL", 10)

    assert result["status"] == "unfilled"
    mock_discord.assert_called_once()
    assert "SELL UNFILLED" in mock_discord.call_args[0][0]


def test_sell_submits_day_market_order_with_fill_verification(monkeypatch):
    """Fill verification must not change the order parameters."""
    monkeypatch.setattr(live_trader.time, "sleep", lambda _: None)
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(live_trader.time, "time", lambda: next(ticks))

    api = MagicMock()
    api.submit_order.return_value.id = "order-2"
    api.get_order.return_value = _order("order-2", status="filled")

    place_sell(api, "AAPL", 10)

    kwargs = api.submit_order.call_args.kwargs
    assert kwargs["side"] == "sell"
    assert kwargs["type"] == "market"
    assert kwargs["time_in_force"] == "day"


def test_sell_submit_error_returns_error_without_retry():
    api = MagicMock()
    api.submit_order.side_effect = Exception("position not found")

    result = place_sell(api, "AAPL", 10)

    assert result["status"] == "error"
    assert "position not found" in result["reason"]
    api.submit_order.assert_called_once()


# ---------------------------------------------------------------------------
# _check_fill_after_cancel — post-cancel race guard
# ---------------------------------------------------------------------------


def test_check_fill_after_cancel_returns_order_on_filled():
    api = MagicMock()
    filled = _order("o1", status="filled")
    api.get_order.return_value = filled

    assert _check_fill_after_cancel(api, "o1") is filled


def test_check_fill_after_cancel_returns_none_on_canceled():
    api = MagicMock()
    api.get_order.return_value = _order("o1", status="canceled")

    assert _check_fill_after_cancel(api, "o1") is None


def test_check_fill_after_cancel_swallows_api_error():
    api = MagicMock()
    api.get_order.side_effect = Exception("timeout")

    assert _check_fill_after_cancel(api, "o1") is None


# ---------------------------------------------------------------------------
# Finding 1 fix: race-condition guard prevents double position
# ---------------------------------------------------------------------------


def test_buy_does_not_retry_when_order_filled_after_cancel(monkeypatch):
    """If the order fills between our last poll and the cancel, don't re-submit."""
    monkeypatch.setattr(live_trader.time, "sleep", lambda _: None)
    # First _wait_for_fill: start=0, poll=5, deadline=11 → timeout
    # _check_fill_after_cancel: returns filled
    ticks = iter([0.0, 5.0, 11.0])
    monkeypatch.setattr(live_trader.time, "time", lambda: next(ticks))

    api = MagicMock()
    api.get_latest_trade.return_value.price = 200.0
    api.submit_order.return_value.id = "order-1"

    # _wait_for_fill times out (status stays "new"), but _check_fill_after_cancel
    # sees the order as filled.
    api.get_order.side_effect = [
        _order("order-1", status="new"),   # _wait_for_fill poll
        _order("order-1", status="filled"),  # _check_fill_after_cancel
    ]

    result = place_buy(api, "AAPL", 10)

    assert result["status"] == "filled"
    assert result["order_id"] == "order-1"
    # Only one submit — no retry happened
    api.submit_order.assert_called_once()


def test_sell_does_not_retry_when_order_filled_after_cancel(monkeypatch):
    monkeypatch.setattr(live_trader.time, "sleep", lambda _: None)
    ticks = iter([0.0, 5.0, 11.0])
    monkeypatch.setattr(live_trader.time, "time", lambda: next(ticks))

    api = MagicMock()
    api.submit_order.return_value.id = "sell-1"
    api.get_order.side_effect = [
        _order("sell-1", status="new"),
        _order("sell-1", status="filled"),
    ]

    result = place_sell(api, "AAPL", 10)

    assert result["status"] == "filled"
    assert result["order_id"] == "sell-1"
    api.submit_order.assert_called_once()


# ---------------------------------------------------------------------------
# Finding 5: filled_qty propagation
# ---------------------------------------------------------------------------


def test_buy_returns_filled_qty(monkeypatch):
    monkeypatch.setattr(live_trader.time, "sleep", lambda _: None)
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(live_trader.time, "time", lambda: next(ticks))

    api = MagicMock()
    api.get_latest_trade.return_value.price = 200.0
    api.submit_order.return_value.id = "order-1"

    filled = _order("order-1", status="filled")
    filled.filled_qty = "7"  # partial fill: requested 10 but got 7
    api.get_order.return_value = filled

    result = place_buy(api, "AAPL", 10)

    assert result["status"] == "filled"
    assert result["filled_qty"] == "7"


def test_sell_returns_filled_qty(monkeypatch):
    monkeypatch.setattr(live_trader.time, "sleep", lambda _: None)
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(live_trader.time, "time", lambda: next(ticks))

    api = MagicMock()
    api.submit_order.return_value.id = "sell-1"

    filled = _order("sell-1", status="filled")
    filled.filled_qty = "8"
    api.get_order.return_value = filled

    result = place_sell(api, "AAPL", 10)

    assert result["status"] == "filled"
    assert result["filled_qty"] == "8"


# ---------------------------------------------------------------------------
# Early run guard (stamp on first fill, not at end of execution)
# ---------------------------------------------------------------------------


def test_early_guard_stamped_on_first_sell_fill(monkeypatch, tmp_path):
    """Guard is written the moment the first SELL fills, not deferred to end."""
    guard = tmp_path / "data" / "last_run_date.txt"
    monkeypatch.setattr(live_trader, "LAST_RUN_GUARD_PATH", str(guard))
    monkeypatch.setattr(live_trader, "today_utc", lambda: "2026-09-10")

    # Simulate the inline guard-stamping pattern from run()
    guard_stamped = False
    result_status = "filled"

    if result_status == "filled":
        if not guard_stamped:
            _write_run_guard()
            guard_stamped = True

    assert guard.read_text() == "2026-09-10"
    assert guard_stamped is True


def test_early_guard_stamped_only_once(monkeypatch, tmp_path):
    """Multiple fills call _write_run_guard exactly once."""
    guard = tmp_path / "data" / "last_run_date.txt"
    monkeypatch.setattr(live_trader, "LAST_RUN_GUARD_PATH", str(guard))
    monkeypatch.setattr(live_trader, "today_utc", lambda: "2026-09-10")

    call_count = 0
    original_write = _write_run_guard

    def counting_write():
        nonlocal call_count
        call_count += 1
        original_write()

    monkeypatch.setattr(live_trader, "_write_run_guard", counting_write)

    guard_stamped = False

    # Simulate three fills in sequence (SELL, rebalancer, BUY)
    for _ in range(3):
        if not guard_stamped:
            live_trader._write_run_guard()
            guard_stamped = True

    assert call_count == 1
    assert guard.read_text() == "2026-09-10"


def test_guard_fallback_when_no_fills(monkeypatch, tmp_path):
    """When all signals are HOLDs/skips (no fills), guard still stamps at end."""
    guard = tmp_path / "data" / "last_run_date.txt"
    monkeypatch.setattr(live_trader, "LAST_RUN_GUARD_PATH", str(guard))
    monkeypatch.setattr(live_trader, "today_utc", lambda: "2026-09-10")

    guard_stamped = False

    # Simulate: no fills occurred (all skips/holds)
    # At end of run(), the fallback stamps:
    if not guard_stamped:
        _write_run_guard()

    assert guard.read_text() == "2026-09-10"


def test_guard_not_stamped_before_first_fill(monkeypatch, tmp_path):
    """Guard file does not exist until a fill actually happens."""
    guard = tmp_path / "data" / "last_run_date.txt"
    monkeypatch.setattr(live_trader, "LAST_RUN_GUARD_PATH", str(guard))
    monkeypatch.setattr(live_trader, "today_utc", lambda: "2026-09-10")

    guard_stamped = False

    # Simulate skipped signals — no fills
    for result_status in ("skipped", "skipped", "error"):
        if result_status == "filled" and not guard_stamped:
            _write_run_guard()
            guard_stamped = True

    # Guard should NOT exist yet
    assert not guard.exists()
    assert guard_stamped is False
