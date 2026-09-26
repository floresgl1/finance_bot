"""Tests for reconcile_stops.py — recovering stop-loss exits from Alpaca.

The whole point of this module is to stop under-reporting losses, so the tests
concentrate on the ways it could quietly go back to doing that: writing a row
twice, skipping a real fill, or fabricating an entry price.
"""

import csv
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

import reconcile_stops
import signal_logger
from reconcile_stops import (
    STOP_LOSS_FILL,
    _normalize_timestamp,
    entry_price_from_log,
    existing_exit_keys,
    fetch_filled_stops,
    format_discord,
    format_summary,
    reconcile,
    resolve_entry,
    select_unreconciled,
)


# --- helpers ---------------------------------------------------------------


def _order(
    *,
    order_id="stop-1",
    symbol="AAPL",
    side="sell",
    order_type="stop",
    status="filled",
    filled_qty="10",
    filled_avg_price="90.0",
    filled_at="2026-07-01T14:30:00+00:00",
):
    o = MagicMock()
    o.id = order_id
    o.symbol = symbol
    o.side = side
    o.type = order_type
    o.status = status
    o.filled_qty = filled_qty
    o.filled_avg_price = filled_avg_price
    o.filled_at = filled_at
    return o


def _write_log(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=signal_logger.FIELDNAMES)
        writer.writeheader()
        for row in rows:
            full = {name: "" for name in signal_logger.FIELDNAMES}
            full.update(row)
            writer.writerow(full)


def _entry(ticker="AAPL", order_id="entry-1", price=100.0, date_="2026-06-01"):
    return {
        "date": date_, "ticker": ticker, "row_type": "ENTRY",
        "model_signal": "BUY", "actual_action": "BUY",
        "price": price, "qty": 10, "confidence": 55.0,
        "entry_order_id": order_id,
    }


def _exit(ticker="AAPL", order_id="entry-1", ts="2026-07-01T14:30:00+00:00"):
    return {
        "date": "2026-07-01", "ticker": ticker, "row_type": "EXIT",
        "entry_order_id": order_id, "exit_timestamp": ts,
        "exit_price": 90.0, "exit_reason": STOP_LOSS_FILL,
        "shares": 10, "realized_pnl": -100.0,
    }


@pytest.fixture
def log(tmp_path, monkeypatch):
    path = tmp_path / "signal_log.csv"
    monkeypatch.setattr(signal_logger, "SIGNAL_LOG_PATH", str(path))
    return path


# --- fetching --------------------------------------------------------------


def test_only_filled_sell_stops_are_collected():
    api = MagicMock()
    api.list_orders.return_value = [
        _order(order_id="keep-1"),
        _order(order_id="keep-2", order_type="stop_limit"),
        _order(order_id="drop-buy", side="buy"),
        _order(order_id="drop-market", order_type="market"),
        _order(order_id="drop-limit", order_type="limit"),
    ]

    fills = fetch_filled_stops(api, date(2026, 6, 1))

    assert [f["order_id"] for f in fills] == ["keep-1", "keep-2"]


def test_cancelled_stop_is_not_a_fill():
    """A stop cancelled because the bot sold first closed no position."""
    api = MagicMock()
    api.list_orders.return_value = [
        _order(order_id="cancelled", status="canceled"),
        _order(order_id="filled"),
    ]

    fills = fetch_filled_stops(api, date(2026, 6, 1))

    assert [f["order_id"] for f in fills] == ["filled"]


@pytest.mark.parametrize(
    "qty, price",
    [("0", "90.0"), ("-5", "90.0"), ("10", "0"), ("10", "-1")],
)
def test_non_positive_fills_are_skipped(qty, price):
    api = MagicMock()
    api.list_orders.return_value = [_order(filled_qty=qty, filled_avg_price=price)]

    assert fetch_filled_stops(api, date(2026, 6, 1)) == []


def test_unparseable_fill_is_skipped_not_fatal():
    api = MagicMock()
    api.list_orders.return_value = [
        _order(order_id="bad", filled_avg_price="not-a-number"),
        _order(order_id="good"),
    ]

    fills = fetch_filled_stops(api, date(2026, 6, 1))

    assert [f["order_id"] for f in fills] == ["good"]


def test_list_orders_failure_propagates():
    """A silent empty result would look like 'no stops fired'."""
    api = MagicMock()
    api.list_orders.side_effect = Exception("alpaca down")

    with pytest.raises(Exception, match="alpaca down"):
        fetch_filled_stops(api, date(2026, 6, 1))


def test_page_limit_warning(capsys):
    api = MagicMock()
    api.list_orders.return_value = [
        _order(order_id=f"s{i}") for i in range(reconcile_stops.ORDER_PAGE_LIMIT)
    ]

    fetch_filled_stops(api, date(2026, 6, 1))

    assert "page limit" in capsys.readouterr().out


# --- timestamp normalisation ----------------------------------------------


def test_timestamp_normalises_to_utc_seconds():
    assert _normalize_timestamp("2026-07-01T14:30:00Z") == "2026-07-01T14:30:00+00:00"


def test_timestamp_from_datetime():
    dt = datetime(2026, 7, 1, 14, 30, 0, tzinfo=timezone.utc)
    assert _normalize_timestamp(dt) == "2026-07-01T14:30:00+00:00"


def test_timestamp_drops_sub_second_precision():
    """Sub-second jitter would break the idempotency key."""
    a = _normalize_timestamp("2026-07-01T14:30:00.123456Z")
    b = _normalize_timestamp("2026-07-01T14:30:00.987654Z")
    assert a == b


def test_timestamp_handles_none_and_garbage():
    assert _normalize_timestamp(None) == ""
    assert _normalize_timestamp("garbage") == "garbage"


# --- deduplication ---------------------------------------------------------


def test_existing_keys_read_from_exit_rows(log):
    _write_log(log, [_entry(), _exit()])

    keys = existing_exit_keys(str(log))

    assert ("AAPL", "2026-07-01T14:30:00+00:00") in keys


def test_existing_keys_ignore_entry_rows(log):
    _write_log(log, [_entry()])
    assert existing_exit_keys(str(log)) == set()


def test_existing_keys_on_missing_file(tmp_path):
    assert existing_exit_keys(str(tmp_path / "nope.csv")) == set()


def test_already_reconciled_fill_is_skipped():
    fills = [{"ticker": "AAPL", "filled_at": "2026-07-01T14:30:00+00:00"}]
    seen = {("AAPL", "2026-07-01T14:30:00+00:00")}

    assert select_unreconciled(fills, seen) == []


def test_same_ticker_different_time_is_a_distinct_fill():
    fills = [
        {"ticker": "AAPL", "filled_at": "2026-07-01T14:30:00+00:00"},
        {"ticker": "AAPL", "filled_at": "2026-08-15T18:00:00+00:00"},
    ]
    seen = {("AAPL", "2026-07-01T14:30:00+00:00")}

    remaining = select_unreconciled(fills, seen)

    assert len(remaining) == 1
    assert remaining[0]["filled_at"] == "2026-08-15T18:00:00+00:00"


def test_duplicates_within_one_batch_are_collapsed():
    fills = [
        {"ticker": "AAPL", "filled_at": "2026-07-01T14:30:00+00:00"},
        {"ticker": "AAPL", "filled_at": "2026-07-01T14:30:00+00:00"},
    ]

    assert len(select_unreconciled(fills, set())) == 1


# --- entry price resolution ------------------------------------------------


def test_entry_price_comes_from_the_actual_buy_fill(log):
    _write_log(log, [_entry(order_id="entry-1", price=100.0)])
    api = MagicMock()
    api.get_order.return_value.filled_avg_price = "98.75"

    order_id, price, note = resolve_entry(api, "AAPL")

    assert order_id == "entry-1"
    assert price == 98.75   # the fill, not the signal-time 100.0
    assert note == ""


def test_unfetchable_entry_order_reports_a_note(log):
    _write_log(log, [_entry(order_id="entry-1")])
    api = MagicMock()
    api.get_order.side_effect = Exception("order not found")

    order_id, price, note = resolve_entry(api, "AAPL")

    assert order_id == "entry-1"
    assert price is None
    assert "could not fetch" in note


def test_no_open_entry_returns_unlinked(log):
    _write_log(log, [])
    api = MagicMock()

    order_id, price, note = resolve_entry(api, "AAPL")

    assert order_id == "UNLINKED"
    assert price is None


def test_log_fallback_price(log):
    _write_log(log, [_entry(order_id="entry-1", price=100.0)])

    assert entry_price_from_log("AAPL", "entry-1", str(log)) == 100.0


def test_log_fallback_returns_none_for_unknown_ticker(log):
    _write_log(log, [_entry(ticker="AAPL")])

    assert entry_price_from_log("MSFT", "entry-9", str(log)) is None


# --- reconciliation --------------------------------------------------------


def test_writes_an_exit_row_for_an_unreconciled_fill(log):
    _write_log(log, [_entry(order_id="entry-1", price=100.0)])
    api = MagicMock()
    api.list_orders.return_value = [_order(filled_qty="10", filled_avg_price="90.0")]
    api.get_order.return_value.filled_avg_price = "100.0"

    result = reconcile(api, date(2026, 6, 1), log_path=str(log))

    assert len(result["written"]) == 1
    row = result["written"][0]
    assert row["ticker"] == "AAPL"
    assert row["realized_pnl"] == -100.0   # (90 - 100) * 10

    written = [r for r in csv.DictReader(open(log)) if r["row_type"] == "EXIT"]
    assert len(written) == 1
    assert written[0]["exit_reason"] == STOP_LOSS_FILL
    assert written[0]["exit_timestamp"] == "2026-07-01T14:30:00+00:00"
    assert float(written[0]["realized_pnl"]) == -100.0


def test_reconciling_twice_writes_nothing_the_second_time(log):
    """Idempotency — the property that makes a daily cron safe."""
    _write_log(log, [_entry(order_id="entry-1", price=100.0)])
    api = MagicMock()
    api.list_orders.return_value = [_order()]
    api.get_order.return_value.filled_avg_price = "100.0"

    first = reconcile(api, date(2026, 6, 1), log_path=str(log))
    second = reconcile(api, date(2026, 6, 1), log_path=str(log))

    assert len(first["written"]) == 1
    assert len(second["written"]) == 0
    assert second["already_reconciled"] == 1

    exits = [r for r in csv.DictReader(open(log)) if r["row_type"] == "EXIT"]
    assert len(exits) == 1


def test_dry_run_writes_nothing(log):
    _write_log(log, [_entry(order_id="entry-1", price=100.0)])
    api = MagicMock()
    api.list_orders.return_value = [_order()]
    api.get_order.return_value.filled_avg_price = "100.0"

    result = reconcile(api, date(2026, 6, 1), dry_run=True, log_path=str(log))

    assert len(result["written"]) == 1   # would have written
    exits = [r for r in csv.DictReader(open(log)) if r["row_type"] == "EXIT"]
    assert exits == []


def test_falls_back_to_the_logged_entry_price_and_counts_it(log):
    _write_log(log, [_entry(order_id="entry-1", price=105.0)])
    api = MagicMock()
    api.list_orders.return_value = [_order(filled_avg_price="90.0", filled_qty="10")]
    api.get_order.side_effect = Exception("order aged out")

    result = reconcile(api, date(2026, 6, 1), log_path=str(log))

    assert len(result["written"]) == 1
    assert result["degraded_entry_price"] == 1
    assert result["written"][0]["realized_pnl"] == -150.0   # (90 - 105) * 10


def test_fill_with_no_recoverable_entry_price_is_skipped_not_fabricated(log):
    """A fabricated P&L is worse than a reported gap."""
    _write_log(log, [])
    api = MagicMock()
    api.list_orders.return_value = [_order()]

    result = reconcile(api, date(2026, 6, 1), log_path=str(log))

    assert result["written"] == []
    assert result["skipped"] == 1


def test_exit_row_is_dated_when_the_stop_fired(log):
    """Not when the reconciler happened to run — windowed reports depend on it."""
    _write_log(log, [_entry(order_id="entry-1", price=100.0)])
    api = MagicMock()
    api.list_orders.return_value = [_order(filled_at="2026-07-01T14:30:00+00:00")]
    api.get_order.return_value.filled_avg_price = "100.0"

    reconcile(api, date(2026, 6, 1), log_path=str(log))

    exits = [r for r in csv.DictReader(open(log)) if r["row_type"] == "EXIT"]
    assert exits[0]["date"] == "2026-07-01"


def test_multiple_tickers_reconciled_independently(log):
    _write_log(log, [
        _entry(ticker="AAPL", order_id="e-aapl", price=100.0),
        _entry(ticker="MSFT", order_id="e-msft", price=200.0),
    ])
    api = MagicMock()
    api.list_orders.return_value = [
        _order(order_id="s1", symbol="AAPL", filled_avg_price="90.0", filled_qty="10"),
        _order(order_id="s2", symbol="MSFT", filled_avg_price="180.0", filled_qty="5",
               filled_at="2026-07-02T15:00:00+00:00"),
    ]
    api.get_order.side_effect = lambda oid: MagicMock(
        filled_avg_price={"e-aapl": "100.0", "e-msft": "200.0"}[oid]
    )

    result = reconcile(api, date(2026, 6, 1), log_path=str(log))

    by_ticker = {r["ticker"]: r for r in result["written"]}
    assert by_ticker["AAPL"]["realized_pnl"] == -100.0
    assert by_ticker["MSFT"]["realized_pnl"] == -100.0


# --- reporting -------------------------------------------------------------


def test_summary_reports_recovered_pnl():
    result = {
        "fills_seen": 3, "already_reconciled": 1, "skipped": 0,
        "degraded_entry_price": 0, "dry_run": False,
        "written": [
            {"ticker": "AAPL", "filled_at": "2026-07-01T14:30:00+00:00",
             "entry_price": 100.0, "exit_price": 90.0, "shares": 10.0,
             "realized_pnl": -100.0},
        ],
    }

    text = format_summary(result)

    assert "EXIT rows written       : 1" in text
    assert "-100.00" in text


def test_summary_flags_skips_and_degraded_prices():
    result = {
        "fills_seen": 2, "already_reconciled": 0, "skipped": 1,
        "degraded_entry_price": 1, "dry_run": True, "written": [],
    }

    text = format_summary(result)

    assert "dry run" in text
    assert "Skipped (no entry price): 1" in text
    assert "Degraded entry price" in text


def test_discord_summary_lists_recovered_exits():
    result = {
        "fills_seen": 1, "already_reconciled": 0, "skipped": 0,
        "degraded_entry_price": 0, "dry_run": False,
        "written": [
            {"ticker": "AAPL", "shares": 10.0, "realized_pnl": -100.0},
        ],
    }

    message = format_discord(result)

    assert "AAPL" in message
    assert "-100" in message
    assert len(message) < 2000


# --- position_id resolution --------------------------------------------------
#
# This job runs after the trading session. By then the bot may have bought the
# ticker again, so "most recent position_id" would hand the old position's stop
# loss to the new position. These tests pin the fill-time resolution.

from reconcile_stops import resolve_position_id


def _pos_entry(position_id, date_):
    return {
        "date": date_, "ticker": "AAPL", "row_type": "ENTRY", "actual_action": "BUY",
        "entry_order_id": position_id, "position_id": position_id,
    }


def test_position_id_is_the_one_open_before_the_fill(log):
    _write_log(log, [_pos_entry("posA", "2026-06-01")])
    assert resolve_position_id(MagicMock(), "AAPL", "2026-07-01T14:30:00+00:00", str(log)) == ("posA", "")


def test_rebuy_after_the_stop_does_not_capture_its_loss(log):
    """Stop fires on 07-01, bot re-buys on 07-02, reconciler runs after that."""
    _write_log(log, [_pos_entry("posA", "2026-06-01"), _pos_entry("posB", "2026-07-02")])
    pid, _ = resolve_position_id(MagicMock(), "AAPL", "2026-07-01T14:30:00+00:00", str(log))
    assert pid == "posA"


def test_same_day_rebuy_after_the_stop_is_resolved_by_fill_time(log):
    _write_log(log, [_pos_entry("posA", "2026-06-01"), _pos_entry("posB", "2026-07-01")])
    api = MagicMock()
    api.get_order.return_value.filled_at = "2026-07-01T15:02:00Z"   # after the 14:30 stop

    pid, _ = resolve_position_id(api, "AAPL", "2026-07-01T14:30:00+00:00", str(log))

    assert pid == "posA"
    api.get_order.assert_called_once_with("posB")


def test_same_day_open_before_the_stop_owns_it(log):
    _write_log(log, [_pos_entry("posA", "2026-06-01"), _pos_entry("posB", "2026-07-01")])
    api = MagicMock()
    api.get_order.return_value.filled_at = "2026-07-01T13:45:00Z"

    assert resolve_position_id(api, "AAPL", "2026-07-01T14:30:00+00:00", str(log)) == ("posB", "")


def test_same_day_unfetchable_open_leaves_the_id_blank(log):
    _write_log(log, [_pos_entry("posA", "2026-06-01"), _pos_entry("posB", "2026-07-01")])
    api = MagicMock()
    api.get_order.side_effect = RuntimeError("boom")

    pid, note = resolve_position_id(api, "AAPL", "2026-07-01T14:30:00+00:00", str(log))

    assert pid is None
    assert "could not fetch" in note


def test_no_position_ids_yet_returns_blank(log):
    _write_log(log, [_entry()])
    assert resolve_position_id(MagicMock(), "AAPL", "2026-07-01T14:30:00+00:00", str(log)) == (None, "")


def test_reconcile_writes_the_resolved_position_id(log):
    _write_log(log, [_pos_entry("posA", "2026-06-01"), _pos_entry("posB", "2026-07-02")])
    api = MagicMock()
    api.list_orders.return_value = [_order()]
    api.get_order.return_value.filled_avg_price = "100.0"

    reconcile(api, date(2026, 6, 1), log_path=str(log))

    written = [r for r in csv.DictReader(open(log)) if r["row_type"] == "EXIT"]
    assert written[0]["position_id"] == "posA"
