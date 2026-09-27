"""smoke_alpaca.py run against the fake clients from test_broker.

It cannot reach Alpaca from CI, so these pin its own logic: the guards, the
read-only path, and that both order paths clean up after themselves.
"""

import pytest

pytest.importorskip("alpaca")

import broker
import smoke_alpaca
from tests.test_broker import FakeData, FakeTrading, _order


def _api(trading):
    return broker.AlpacaREST("k", "s", trading=trading, data=FakeData())


def test_refuses_a_watchlist_symbol():
    assert smoke_alpaca.main(["--symbol", "AAPL"]) == 2


def test_read_only_checks_pass_against_the_adapter():
    c = smoke_alpaca.Checks()
    smoke_alpaca.read_only_checks(_api(FakeTrading()), "F", c)
    assert c.failed == 0


class UnfilledTrading(FakeTrading):
    def __init__(self):
        super().__init__(orders=[_order(status="accepted", symbol="F", side="buy",
                                        type="market", order_type="market",
                                        filled_at=None)])

    def get_all_positions(self):
        return []

    def cancel_order_by_id(self, order_id):
        super().cancel_order_by_id(order_id)
        self.orders = [_order(status="canceled", symbol="F", filled_at=None)]


def test_an_unfilled_oto_is_cancelled(monkeypatch):
    monkeypatch.setattr(smoke_alpaca, "FILL_WAIT_SECONDS", 0)
    trading = UnfilledTrading()
    c = smoke_alpaca.Checks()
    smoke_alpaca.order_checks(_api(trading), "F", c)
    assert c.failed == 0
    assert [n for n, _ in trading.calls].count("submit_order") == 1
    assert any(n == "cancel_order_by_id" for n, _ in trading.calls)


class FilledTrading(FakeTrading):
    """Entry fills at once; its stop leg and the standing stop cancel cleanly."""

    def __init__(self):
        super().__init__(orders=[_order(symbol="F", side="sell", type="stop",
                                        order_type="stop", status="new")])

    def get_all_positions(self):
        return []

    def get_order_by_id(self, order_id):
        self.calls.append(("get_order_by_id", order_id))
        cancelled = any(n == "cancel_order_by_id" and i == order_id for n, i in self.calls)
        return _order(symbol="F", status="canceled" if cancelled else "filled")


def test_a_filled_oto_exercises_every_shape_and_sells_the_share():
    trading = FilledTrading()
    c = smoke_alpaca.Checks()
    smoke_alpaca.order_checks(_api(trading), "F", c)
    assert c.failed == 0
    sent = [r for n, r in trading.calls if n == "submit_order"]
    shapes = [(r.side.value, r.type.value, r.order_class.value if r.order_class else None)
              for r in sent]
    assert shapes == [("buy", "market", "oto"), ("sell", "stop", None), ("sell", "market", None)]


def test_order_checks_skip_a_symbol_already_held():
    c = smoke_alpaca.Checks()
    trading = FakeTrading()                 # holds AAPL
    smoke_alpaca.order_checks(_api(trading), "AAPL", c)
    assert c.failed == 1
    assert not any(n == "submit_order" for n, _ in trading.calls)


# --- stop coverage diagnostic ---------------------------------------------------
from types import SimpleNamespace as NS


def _pos(sym, qty):
    return NS(symbol=sym, qty=str(qty))


def _ord(sym, side="sell", type="stop", qty=10, status="new"):
    return NS(symbol=sym, side=side, type=type, qty=str(qty), status=status)


def test_one_stop_per_position_is_clean():
    rows, warnings = smoke_alpaca.stop_coverage([_pos("AAPL", 10)], [_ord("AAPL")])
    assert warnings == []
    assert rows[0]["stops"] == 1 and rows[0]["stop_qty"] == 10


def test_one_stop_per_lot_is_clean():
    """A position bought in lots carries one OTO stop per lot; that is fine."""
    rows, warnings = smoke_alpaca.stop_coverage(
        [_pos("AMZN", 38)], [_ord("AMZN", qty=q) for q in (10, 8, 7, 7, 6)])
    assert warnings == []
    assert rows[0]["stops"] == 5 and rows[0]["stop_qty"] == 38


def test_stops_covering_more_than_held_are_flagged():
    _, warnings = smoke_alpaca.stop_coverage(
        [_pos("AAPL", 10)], [_ord("AAPL"), _ord("AAPL"), _ord("AAPL")])
    assert warnings == ["AAPL: stops cover 30 shares but only 10 are held "
                        "(over-covered: could sell shares that are not there)"]


def test_an_unprotected_position_is_flagged():
    _, warnings = smoke_alpaca.stop_coverage([_pos("MSFT", 5)], [])
    assert warnings == ["MSFT: 5 shares held with NO live stop (unprotected)"]


def test_a_partial_stop_is_flagged():
    _, warnings = smoke_alpaca.stop_coverage([_pos("MSFT", 5)], [_ord("MSFT", qty=3)])
    assert warnings == ["MSFT: stops cover 3 shares but 5 are held (partly unprotected)"]


def test_an_orphan_stop_is_flagged():
    _, warnings = smoke_alpaca.stop_coverage([], [_ord("NVDA")])
    assert warnings == ["NVDA: 1 live sell stop(s) but no position (orphan)"]


def test_a_stop_leg_waiting_on_an_unfilled_buy_is_not_an_orphan():
    rows, warnings = smoke_alpaca.stop_coverage(
        [], [_ord("XOM", side="buy", type="market"), _ord("XOM", status="held")])
    assert warnings == []
    assert rows[0]["waiting_stops"] == 1 and rows[0]["other"] == "buy market"


def test_read_only_run_prints_coverage_without_failing(capsys):
    c = smoke_alpaca.Checks()
    smoke_alpaca.read_only_checks(_api(FakeTrading()), "F", c)
    out = capsys.readouterr().out
    assert c.failed == 0
    assert "Stop coverage" in out
