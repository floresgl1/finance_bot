"""
Alpaca adapter: the old `alpaca_trade_api.REST` interface, backed by alpaca-py.

`alpaca-trade-api` is deprecated. Rather than rewrite ~30 call sites at once,
this module exposes the exact methods and return shapes the bot already uses,
translated to alpaca-py underneath. Call sites keep working unchanged, and all
SDK-specific translation lives here where it can be tested on its own.

Return values mirror the old SDK: plain objects whose ids, statuses, order
types and sides are strings, not the UUIDs and enums alpaca-py returns. That
matters for more than printing: `stop.id not in failed` compares a UUID to a
list of strings and is always True, and `str(OrderStatus.FILLED)` renders as
"OrderStatus.FILLED" rather than "filled".

Errors pass through untouched. alpaca-py's APIError exposes `status_code` the
same way the old SDK did, which live_trader._is_infra_error depends on, and
network failures still surface as `requests` exceptions.
"""

import os
from datetime import date, datetime
from enum import Enum
from types import SimpleNamespace
from uuid import UUID

import pandas as pd

PAPER_URL = "https://paper-api.alpaca.markets"


def _plain(value):
    """Convert alpaca-py field values to the old SDK's plain types."""
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if hasattr(value, "model_fields"):
        return _entity(value)
    return value


def _entity(model) -> SimpleNamespace:
    """A read-only-by-convention namespace of the model's fields, plain-typed."""
    return SimpleNamespace(**{k: _plain(getattr(model, k)) for k in type(model).model_fields})


def _as_datetime(value):
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    return pd.Timestamp(value).to_pydatetime()


class AlpacaREST:
    """Drop-in replacement for `alpaca_trade_api.REST` over alpaca-py.

    `trading` and `data` can be injected for tests; otherwise real clients
    are built from the keys.
    """

    def __init__(self, api_key: str, secret_key: str, base_url: str = PAPER_URL,
                 trading=None, data=None):
        if trading is None or data is None:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.trading.client import TradingClient

            paper = "paper" in base_url
            trading = trading or TradingClient(api_key, secret_key, paper=paper)
            data = data or StockHistoricalDataClient(api_key, secret_key)
        self._trading = trading
        self._data = data

    # -- account / market ----------------------------------------------------
    def get_account(self):
        return _entity(self._trading.get_account())

    def get_clock(self):
        return _entity(self._trading.get_clock())

    def get_portfolio_history(self, period: str | None = None, timeframe: str | None = None):
        """Returns an object with `.df`, like the old SDK: one row per bar,
        indexed by New York time, with an `equity` column."""
        from alpaca.trading.requests import GetPortfolioHistoryRequest

        h = self._trading.get_portfolio_history(
            GetPortfolioHistoryRequest(period=period, timeframe=timeframe))
        index = pd.to_datetime(h.timestamp, unit="s", utc=True).tz_convert("America/New_York")
        df = pd.DataFrame({"equity": h.equity, "profit_loss": h.profit_loss,
                           "profit_loss_pct": h.profit_loss_pct}, index=index)
        return SimpleNamespace(df=df, timestamp=h.timestamp, equity=h.equity)

    def get_latest_trade(self, symbol: str):
        from alpaca.data.requests import StockLatestTradeRequest

        trades = self._data.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=symbol))
        return _entity(trades[symbol])

    # -- positions -----------------------------------------------------------
    def list_positions(self):
        return [_entity(p) for p in self._trading.get_all_positions()]

    def get_position(self, symbol: str):
        return _entity(self._trading.get_open_position(symbol))

    # -- orders --------------------------------------------------------------
    def list_orders(self, status: str | None = None, limit: int | None = None,
                    after=None, until=None, direction: str | None = None,
                    nested: bool | None = None, side: str | None = None,
                    symbols: list[str] | None = None):
        from alpaca.common.enums import Sort
        from alpaca.trading.enums import OrderSide, QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        req = GetOrdersRequest(
            status=QueryOrderStatus(status) if status else None,
            limit=limit,
            after=_as_datetime(after),
            until=_as_datetime(until),
            direction=Sort(direction) if direction else None,
            nested=nested,
            side=OrderSide(side) if side else None,
            symbols=symbols,
        )
        return [_entity(o) for o in self._trading.get_orders(filter=req)]

    def get_order(self, order_id: str):
        return _entity(self._trading.get_order_by_id(order_id))

    def cancel_order(self, order_id: str) -> None:
        self._trading.cancel_order_by_id(order_id)

    def submit_order(self, symbol: str, qty, side: str, type: str, time_in_force: str,
                     order_class: str | None = None, stop_loss: dict | None = None,
                     stop_price: float | None = None, limit_price: float | None = None):
        """Only the order shapes the bot places are supported; anything else
        raises rather than being silently approximated."""
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import (
            MarketOrderRequest,
            StopLossRequest,
            StopOrderRequest,
        )

        common = dict(symbol=symbol, qty=qty, side=OrderSide(side),
                      time_in_force=TimeInForce(time_in_force))
        if type == "market" and order_class in (None, "simple"):
            req = MarketOrderRequest(**common)
        elif type == "market" and order_class == "oto" and stop_loss:
            req = MarketOrderRequest(
                **common, order_class=OrderClass.OTO,
                stop_loss=StopLossRequest(stop_price=stop_loss["stop_price"]))
        elif type == "stop" and order_class is None and stop_price is not None:
            req = StopOrderRequest(**common, stop_price=stop_price)
        else:
            raise ValueError(
                f"broker.submit_order: unsupported order shape type={type!r} "
                f"order_class={order_class!r} stop_loss={stop_loss!r}")
        return _entity(self._trading.submit_order(order_data=req))


def connect(base_url: str = PAPER_URL) -> AlpacaREST:
    """Build the adapter from ALPACA_API_KEY / ALPACA_SECRET_KEY."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise EnvironmentError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set as environment variables."
        )
    return AlpacaREST(api_key, secret_key, base_url)
