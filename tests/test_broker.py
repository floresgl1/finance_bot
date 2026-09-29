"""Tests for broker.py, the alpaca-py adapter behind the old REST interface.

What is load-bearing here is not that calls reach alpaca-py -- it is that
what comes back has the old SDK's plain shapes. The bot compares ids to lists
of strings, checks `order.status in ("canceled", "filled")`, and prints both;
a UUID or an enum leaking through would change behaviour without an error.
The fixtures are real alpaca-py model objects, so the types are the real ones.
"""

from datetime import date, datetime, timezone
from types import SimpleNamespace
from uuid import UUID

import pytest

pytest.importorskip("alpaca")

from alpaca.common.exceptions import APIError
from alpaca.common.enums import Sort
from alpaca.data.models import Trade
from alpaca.trading.enums import OrderClass, OrderSide, OrderType, QueryOrderStatus, TimeInForce
from alpaca.trading.models import Clock, Order, PortfolioHistory, Position, TradeAccount
from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest

import broker
import live_trader

NOW = datetime(2026, 9, 25, 15, 0, 5, tzinfo=timezone.utc)
ORDER_ID = "61e69015-8549-4bfd-b9c3-01e75843f47d"


def _order(**over):
    base = dict(
        id=UUID(ORDER_ID), client_order_id="c1", created_at=NOW, updated_at=NOW,
        submitted_at=NOW, filled_at=NOW, order_class="simple", time_in_force="gtc",
        status="filled", extended_hours=False, symbol="AAPL", side="sell",
        type="stop", order_type="stop", qty="3", filled_qty="3",
        filled_avg_price="187.25",
    )
    base.update(over)
    return Order(**base)


class FakeTrading:
    """Records every request; returns canned alpaca-py models."""

    def __init__(self, orders=None):
        self.calls = []
        self.orders = orders or [_order()]

    def get_account(self):
        return TradeAccount(id=UUID(int=1), account_number="PA1", status="ACTIVE",
                            equity="10523.41")

    def get_clock(self):
        return Clock(timestamp=NOW, is_open=True, next_open=NOW, next_close=NOW)

    def get_all_positions(self):
        return [Position(asset_id=UUID(int=2), symbol="AAPL", exchange="NASDAQ",
                         asset_class="us_equity", avg_entry_price="180.00", qty="3",
                         side="long", cost_basis="540.00", current_price="187.25")]

    def get_open_position(self, symbol):
        return self.get_all_positions()[0]

    def get_orders(self, filter):
        self.calls.append(("get_orders", filter))
        return self.orders

    def get_order_by_id(self, order_id):
        self.calls.append(("get_order_by_id", order_id))
        return self.orders[0]

    def cancel_order_by_id(self, order_id):
        self.calls.append(("cancel_order_by_id", order_id))

    def submit_order(self, order_data):
        self.calls.append(("submit_order", order_data))
        return _order(status="accepted", type="market", order_type="market",
                      filled_at=None, filled_qty="0", filled_avg_price=None)

    def get_portfolio_history(self, history_filter):
        self.calls.append(("get_portfolio_history", history_filter))
        return PortfolioHistory(timestamp=[1758801600, 1758888000], equity=[10000.0, 10100.0],
                                profit_loss=[0.0, 100.0], profit_loss_pct=[0.0, 0.01],
                                timeframe="1D")


class FakeData:
    def get_stock_latest_trade(self, request_params):
        sym = request_params.symbol_or_symbols
        return {sym: Trade(sym, {"t": "2026-09-25T15:00:05Z", "p": 187.3, "s": 100,
                                 "x": "V", "i": 1, "z": "C", "c": ["@"]})}


@pytest.fixture
def fakes():
    trading = FakeTrading()
    return trading, broker.AlpacaREST("k", "s", trading=trading, data=FakeData())


# --- return shapes ------------------------------------------------------------
def test_order_ids_come_back_as_strings_so_list_membership_works(fakes):
    _, api = fakes
    order = api.get_order(ORDER_ID)
    assert order.id == ORDER_ID and isinstance(order.id, str)
    assert order.id in [ORDER_ID]


def test_statuses_types_and_sides_are_plain_strings(fakes):
    _, api = fakes
    o = api.list_orders(status="open", symbols=["AAPL"])[0]
    for value, expected in ((o.status, "filled"), (o.type, "stop"), (o.side, "sell")):
        assert value == expected and type(value) is str
    assert f"{o.status}" == "filled"             # an enum would print OrderStatus.FILLED


def test_the_stop_filter_the_trader_uses_still_matches(fakes):
    """live_trader filters `o.side == "sell" and o.type in ("stop", "stop_limit")`."""
    _, api = fakes
    o = api.list_orders(status="open", symbols=["AAPL"])[0]
    assert o.side == "sell" and o.type in ("stop", "stop_limit")


def test_numeric_fields_still_parse_with_float(fakes):
    _, api = fakes
    assert float(api.get_account().equity) == pytest.approx(10523.41)
    pos = api.list_positions()[0]
    assert (float(pos.qty), float(pos.avg_entry_price), float(pos.current_price)) == (3.0, 180.0, 187.25)
    assert float(api.get_position("AAPL").avg_entry_price) == 180.0
    assert float(api.get_order(ORDER_ID).filled_avg_price) == 187.25


def test_clock_and_latest_trade(fakes):
    _, api = fakes
    assert api.get_clock().is_open is True
    assert api.get_latest_trade("AAPL").price == pytest.approx(187.3)


def test_filled_at_normalises_to_the_same_idempotency_key():
    """reconcile_stops keys fills on filled_at; the key must not change with the SDK."""
    import reconcile_stops

    old_sdk_string = "2026-09-25T15:00:05.123456Z"
    new = broker.AlpacaREST("k", "s", trading=FakeTrading(), data=FakeData()).get_order(ORDER_ID)
    assert (reconcile_stops._normalize_timestamp(new.filled_at)
            == reconcile_stops._normalize_timestamp(old_sdk_string))


def test_portfolio_history_keeps_the_df_the_benchmark_reads(fakes):
    import live_benchmark

    _, api = fakes
    df = api.get_portfolio_history(period="30D", timeframe="1D").df
    assert list(df["equity"]) == [10000.0, 10100.0]
    curve = live_benchmark.fetch_equity_curve(api, days=30)
    assert len(curve) == 2 and curve.index.tz is None


# --- requests sent --------------------------------------------------------------
def _last(trading, name):
    return [c[1] for c in trading.calls if c[0] == name][-1]


def test_oto_buy_sends_a_market_order_with_an_attached_stop(fakes):
    trading, api = fakes
    result = api.submit_order(symbol="AAPL", qty=3, side="buy", type="market",
                              time_in_force="gtc", order_class="oto",
                              stop_loss={"stop_price": 171.0})
    req = _last(trading, "submit_order")
    assert isinstance(req, MarketOrderRequest)
    assert (req.side, req.time_in_force, req.order_class) == (OrderSide.BUY, TimeInForce.GTC, OrderClass.OTO)
    assert req.stop_loss.stop_price == 171.0
    assert isinstance(result.id, str)


def test_plain_market_sell(fakes):
    trading, api = fakes
    api.submit_order(symbol="AAPL", qty=3, side="sell", type="market", time_in_force="day")
    req = _last(trading, "submit_order")
    assert isinstance(req, MarketOrderRequest)
    assert (req.side, req.time_in_force, req.order_class) == (OrderSide.SELL, TimeInForce.DAY, None)
    assert req.stop_loss is None


def test_standing_stop_backfill(fakes):
    trading, api = fakes
    api.submit_order(symbol="AAPL", qty=3, side="sell", type="stop",
                     stop_price=171.0, time_in_force="gtc")
    req = _last(trading, "submit_order")
    assert isinstance(req, StopOrderRequest)
    assert (req.stop_price, req.type, req.time_in_force) == (171.0, OrderType.STOP, TimeInForce.GTC)


def test_an_order_shape_the_bot_never_places_is_refused(fakes):
    _, api = fakes
    with pytest.raises(ValueError, match="unsupported order shape"):
        api.submit_order(symbol="AAPL", qty=3, side="buy", type="limit", time_in_force="day")


def test_list_orders_translates_the_reconciler_query(fakes):
    trading, api = fakes
    api.list_orders(status="closed", after=date(2026, 9, 1).isoformat(), limit=500,
                    direction="asc", nested=False)
    f = _last(trading, "get_orders")
    assert (f.status, f.limit, f.direction, f.nested) == (QueryOrderStatus.CLOSED, 500, Sort.ASC, False)
    assert f.after == datetime(2026, 9, 1)


def test_unset_query_fields_stay_unset(fakes):
    """The old SDK sent nothing it was not given; server defaults must still apply."""
    trading, api = fakes
    api.list_orders(status="open", symbols=["AAPL"])
    f = _last(trading, "get_orders")
    assert (f.limit, f.after, f.direction, f.nested) == (None, None, None, None)
    assert f.symbols == ["AAPL"]


def test_cancel_passes_the_string_id(fakes):
    trading, api = fakes
    api.cancel_order(ORDER_ID)
    assert _last(trading, "cancel_order_by_id") == ORDER_ID


# --- errors: the halt-or-continue decision must not change ---------------------------
def _api_error(status: int, message: str) -> APIError:
    http = SimpleNamespace(response=SimpleNamespace(status_code=status), request=None)
    return APIError(f'{{"code": {status}00, "message": "{message}"}}', http_error=http)


@pytest.mark.parametrize("status, message, infra", [
    (503, "service unavailable", True),
    (500, "internal error", True),
    (429, "rate limit exceeded", True),
    (401, "unauthorized", True),
    (422, "qty must be > 0", False),
    (404, "asset not found", False),
    (403, "insufficient buying power", False),
])
def test_alpaca_py_errors_keep_their_halt_classification(status, message, infra):
    assert live_trader._is_infra_error(_api_error(status, message)) is infra


def test_connect_requires_keys(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(EnvironmentError):
        broker.connect()
