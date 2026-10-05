"""Tests for passive_allocator.py — fixed allocation, deposits first, 5pp band.

The planning functions are pure, so most tests are hand-computed portfolios:
what should a given mix of holdings and cash produce? Three properties matter
most and each has its own tests:

  1. Deposits first.  Cash goes to the underweight fund, in proportion to its
     shortfall, and a deposit that can fix the drift prevents a sale.
  2. The band.  Nothing is sold inside 55–65% VTI, including exactly on the
     edge; outside it, the overweight fund is sold back to target.
  3. Safety in run().  Closed market, second run in a day, and an account
     holding other symbols all place no orders.

Constants are asserted up front so a config change surfaces here as an
explicit failure, not as silently rewritten expectations.
"""

import csv
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import config
import passive_allocator as pa
from passive_allocator import Order, plan_buys, plan_sells

T = {"VTI": 0.60, "VXUS": 0.40}
PRICES = {"VTI": 300.0, "VXUS": 60.0}


def test_constants_match_the_values_these_tests_were_derived_from():
    assert config.PASSIVE_TARGETS == T
    assert config.PASSIVE_BAND == 0.05
    assert config.PASSIVE_MIN_ORDER_USD == 1.00
    assert config.PASSIVE_CASH_BUFFER_USD == 1.00


def _amounts(orders):
    return {(o.symbol, o.side): (o.notional if o.notional is not None else o.qty)
            for o in orders}


# --- plan_buys: deposits first ---------------------------------------------


def test_first_deposit_into_an_empty_account_splits_60_40():
    # $1,000 cash, $1 buffer → $999 investable.
    orders = plan_buys({}, 1000.0, PRICES, T)
    assert _amounts(orders) == {("VTI", "buy"): 599.40, ("VXUS", "buy"): 399.60}


def test_a_deposit_goes_to_the_underweight_fund_by_shortfall():
    # VTI 700 / VXUS 300 + $200 → total 1200; targets 720 / 480;
    # shortfalls 20 / 180 sum to exactly the $200, so it lands on target.
    orders = plan_buys({"VTI": 700.0, "VXUS": 300.0}, 201.0, PRICES, T)
    assert _amounts(orders) == {("VTI", "buy"): 20.0, ("VXUS", "buy"): 180.0}


def test_a_deposit_smaller_than_the_shortfall_all_goes_to_the_underweight_fund():
    # VTI 7000 / VXUS 3000 + $100 → total 10100; targets 6060 / 4040.
    # VTI is over target, so its shortfall is 0 and VXUS gets everything.
    orders = plan_buys({"VTI": 7000.0, "VXUS": 3000.0}, 101.0, PRICES, T)
    assert _amounts(orders) == {("VXUS", "buy"): 100.0}


def test_cash_below_the_minimum_after_the_buffer_buys_nothing():
    assert plan_buys({"VTI": 600.0, "VXUS": 400.0}, 1.50, PRICES, T) == []


def test_a_leg_below_the_minimum_is_dropped():
    # $3 investable on target: 1.80 / 1.20 both clear $1.
    assert len(plan_buys({"VTI": 600.0, "VXUS": 400.0}, 4.0, PRICES, T)) == 2
    # $1.50 investable: 0.90 / 0.60 — both under $1, nothing placed.
    assert plan_buys({"VTI": 600.0, "VXUS": 400.0}, 2.50, PRICES, T) == []


def test_buys_never_spend_more_than_the_investable_cash():
    orders = plan_buys({"VTI": 123.45, "VXUS": 67.89}, 333.33, PRICES, T)
    assert sum(o.notional for o in orders) <= 333.33 - 1.00


def test_whole_share_mode_rounds_down():
    # $999 investable on an empty account: 599.40 / 300 = 1.998 → 1 VTI;
    # 399.60 / 60 = 6.66 → 6 VXUS.
    orders = plan_buys({}, 1000.0, PRICES, T, fractional=False)
    assert _amounts(orders) == {("VTI", "buy"): 1, ("VXUS", "buy"): 6}
    assert all(o.notional is None for o in orders)


def test_whole_share_mode_skips_a_fund_too_expensive_for_the_cash():
    # $199 investable on an empty account:
    # 119.40 / 300 < 1 VTI share → skipped; 79.60 / 60 → 1 VXUS.
    orders = plan_buys({}, 200.0, PRICES, T, fractional=False)
    assert _amounts(orders) == {("VXUS", "buy"): 1}


# --- plan_sells: the band ---------------------------------------------------


def test_inside_the_band_nothing_is_sold():
    assert plan_sells({"VTI": 6300.0, "VXUS": 3700.0}, 0.0, PRICES, T) == []


def test_exactly_on_the_band_edge_nothing_is_sold():
    # 65.0% VTI is |0.65 - 0.60| = 0.05000000000000004 in floats — still inside.
    assert plan_sells({"VTI": 6500.0, "VXUS": 3500.0}, 0.0, PRICES, T) == []


def test_outside_the_band_the_overweight_fund_is_sold_back_to_target():
    # 66% VTI on 10,000 → sell 600 back to 6,000.
    orders = plan_sells({"VTI": 6600.0, "VXUS": 3400.0}, 0.0, PRICES, T)
    assert _amounts(orders) == {("VTI", "sell"): 600.0}


def test_underweight_side_breaching_the_band_sells_the_other_fund():
    # 54% VTI means VXUS is 46% — VXUS is sold back to 40%.
    orders = plan_sells({"VTI": 5400.0, "VXUS": 4600.0}, 0.0, PRICES, T)
    assert _amounts(orders) == {("VXUS", "sell"): 600.0}


def test_a_deposit_that_fixes_the_drift_prevents_the_sale():
    # Same 66% as above, but $1,000 of cash: total 11,000, VTI target 6,600,
    # VXUS shortfall 1,000 — the deposit alone restores 60/40.
    assert plan_sells({"VTI": 6600.0, "VXUS": 3400.0}, 1001.0, PRICES, T) == []


def test_a_deposit_that_only_partly_fixes_the_drift_sells_the_remainder():
    # VTI 8000 / VXUS 2000 + $500 → total 10,500; VXUS gets all 500 → 2,500.
    # VTI is 8000 / 10500 = 76.2%, outside the band: sell 8000 - 6300 = 1700.
    orders = plan_sells({"VTI": 8000.0, "VXUS": 2000.0}, 501.0, PRICES, T)
    assert _amounts(orders) == {("VTI", "sell"): 1700.0}


def test_an_empty_account_sells_nothing():
    assert plan_sells({}, 0.0, PRICES, T) == []


def test_a_one_fund_account_is_rebalanced():
    # All in VTI (100%): sell 40% so the proceeds can buy VXUS.
    orders = plan_sells({"VTI": 1000.0}, 0.0, PRICES, T)
    assert _amounts(orders) == {("VTI", "sell"): 400.0}


@pytest.mark.parametrize("bad", [{"VTI": 0.6, "VXUS": 0.3}, {}, {"VTI": 1.2, "VXUS": -0.2}])
def test_targets_that_do_not_sum_to_one_are_refused(bad):
    with pytest.raises(ValueError, match="sum to 1"):
        plan_buys({}, 100.0, PRICES, bad)


# --- run(): I/O around the plan ---------------------------------------------


class FakeAPI:
    """Holds positions and cash; a notional order fills instantly at PRICES.
    Orders are stamped with `now`, which list_orders filters on like Alpaca."""

    def __init__(self, holdings=None, cash=0.0, is_open=True, foreign=(),
                 now=datetime(2026, 10, 5, 16, 0, tzinfo=timezone.utc)):
        self.holdings = dict(holdings or {})
        self.cash = cash
        self.is_open = is_open
        self.foreign = list(foreign)
        self.now = now
        self.submitted = []
        self.submitted_at = []
        self.fill_status = "filled"

    def get_clock(self):
        return SimpleNamespace(is_open=self.is_open)

    def list_positions(self):
        out = [SimpleNamespace(symbol=s, market_value=str(v), current_price=str(PRICES[s]))
               for s, v in self.holdings.items() if v > 0]
        out += [SimpleNamespace(symbol=s, market_value="100", current_price="10")
                for s in self.foreign]
        return out

    def get_latest_trade(self, symbol):
        return SimpleNamespace(price=PRICES[symbol])

    def get_account(self):
        equity = self.cash + sum(self.holdings.values())
        return SimpleNamespace(cash=str(self.cash), buying_power=str(self.cash * 2),
                               equity=str(equity))

    def submit_order(self, symbol, qty, side, type, time_in_force, notional=None):
        assert (type, time_in_force) == ("market", "day")
        amount = notional if notional is not None else qty * PRICES[symbol]
        sign = 1 if side == "buy" else -1
        self.holdings[symbol] = self.holdings.get(symbol, 0.0) + sign * amount
        self.cash -= sign * amount
        self.submitted.append((symbol, side, notional, qty))
        self.submitted_at.append(self.now)
        return SimpleNamespace(id=f"o{len(self.submitted)}", status="accepted")

    def get_order(self, order_id):
        symbol, _, notional, qty = self.submitted[int(order_id[1:]) - 1]
        if self.fill_status != "filled":
            return SimpleNamespace(status=self.fill_status, filled_qty="0",
                                   filled_avg_price=None)
        shares = notional / PRICES[symbol] if notional is not None else qty
        return SimpleNamespace(status="filled", filled_qty=str(shares),
                               filled_avg_price=str(PRICES[symbol]))

    def list_orders(self, status=None, after=None, limit=None):
        assert status == "all"
        hits = [SimpleNamespace(submitted_at=t) for t in self.submitted_at
                if after is None or t > after]
        return hits[:limit]


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "send_discord", lambda msg: None)
    return dict(log_path=str(tmp_path / "log.csv"), fill_timeout_s=0)


def test_market_closed_places_nothing(paths):
    api = FakeAPI(cash=1000.0, is_open=False)
    assert pa.run(api, today="2026-10-05", **paths) == 0
    assert api.submitted == []


def test_first_run_invests_the_deposit_and_logs_the_fills(paths):
    api = FakeAPI(cash=1000.0)
    assert pa.run(api, today="2026-10-05", **paths) == 0
    assert api.submitted == [("VTI", "buy", 599.40, None), ("VXUS", "buy", 399.60, None)]
    rows = list(csv.DictReader(open(paths["log_path"])))
    assert [r["status"] for r in rows] == ["filled", "filled"]


def test_a_second_run_the_same_day_does_nothing(paths):
    api = FakeAPI(cash=1000.0)
    pa.run(api, today="2026-10-05", **paths)
    api.cash += 500.0                         # a deposit lands later that day
    n = len(api.submitted)
    assert pa.run(api, today="2026-10-05", **paths) == 0
    assert len(api.submitted) == n


def test_a_second_run_while_the_first_runs_buys_are_open_does_not_rebuy(paths):
    # An open buy shows in neither positions nor cash: the account looks
    # exactly as it did before the first run. Only the guard stops a re-buy.
    api = FakeAPI(cash=1000.0)
    api.fill_status = "new"
    pa.run(api, today="2026-10-05", **paths)
    api.holdings, api.cash = {}, 1000.0
    n = len(api.submitted)
    assert pa.run(api, today="2026-10-05", **paths) == 0
    assert len(api.submitted) == n


def test_yesterdays_orders_do_not_block_today(paths):
    api = FakeAPI(cash=1000.0, now=datetime(2026, 10, 2, 16, 0, tzinfo=timezone.utc))
    pa.run(api, today="2026-10-02", **paths)
    api.cash += 500.0
    api.now = datetime(2026, 10, 5, 16, 0, tzinfo=timezone.utc)
    n = len(api.submitted)
    pa.run(api, today="2026-10-05", **paths)
    assert len(api.submitted) > n


def test_a_buy_still_open_at_the_timeout_is_logged_and_fails_the_run(paths):
    api = FakeAPI(cash=1000.0)
    api.fill_status = "new"
    assert pa.run(api, today="2026-10-05", **paths) == 1
    rows = list(csv.DictReader(open(paths["log_path"])))
    assert [r["status"] for r in rows] == ["new", "new"]


def test_the_log_records_what_each_order_filled_at(paths):
    api = FakeAPI(cash=1000.0)
    pa.run(api, today="2026-10-05", **paths)
    vti = next(csv.DictReader(open(paths["log_path"])))
    assert float(vti["filled_avg_price"]) == PRICES["VTI"]
    assert float(vti["filled_qty"]) == pytest.approx(599.40 / PRICES["VTI"])


def test_an_old_log_is_rewritten_under_the_new_header(paths):
    old = ["date", "symbol", "side", "notional", "qty", "status", "order_id", "reason"]
    with open(paths["log_path"], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(old)
        w.writerow(["2026-10-02", "VTI", "buy", "10.0", "", "filled", "x1", "DEPOSIT"])
    pa.run(FakeAPI(cash=1000.0), today="2026-10-05", **paths)
    rows = list(csv.DictReader(open(paths["log_path"])))
    assert [r["date"] for r in rows] == ["2026-10-02", "2026-10-05", "2026-10-05"]
    assert rows[0]["filled_qty"] == "" and rows[1]["filled_qty"] != ""


def test_discord_gets_one_summary_when_orders_are_placed(paths, monkeypatch):
    sent = []
    monkeypatch.setattr(pa, "send_discord", sent.append)
    pa.run(FakeAPI(cash=1000.0), today="2026-10-05", **paths)
    assert len(sent) == 1
    assert sent[0].startswith("**Passive Allocator — Trade Results** | ")
    assert "Portfolio value: **$1,000.00**" in sent[0]


def test_discord_stays_quiet_when_there_is_nothing_to_do(paths, monkeypatch):
    sent = []
    monkeypatch.setattr(pa, "send_discord", sent.append)
    api = FakeAPI(holdings={"VTI": 600.0, "VXUS": 400.0}, cash=0.0)
    assert pa.run(api, today="2026-10-05", **paths) == 0
    assert api.submitted == [] and sent == []


def _row(symbol, status, filled_qty=None, price=None, notional=100.0):
    return {"symbol": symbol, "side": "buy", "notional": notional, "qty": None,
            "status": status, "order_id": f"id-{symbol}", "reason": "DEPOSIT",
            "filled_qty": filled_qty, "filled_avg_price": price}


def test_build_summary_sorts_orders_into_filled_unfilled_and_errors():
    msg = pa.build_summary([_row("VTI", "filled", 1.0412, 287.55),
                            _row("VXUS", "new"),
                            _row("VEA", "error: 422 insufficient buying power")],
                           portfolio_value=500.0, timestamp="2026-10-05 16:00:00 UTC")
    assert msg.splitlines() == [
        "**Passive Allocator — Trade Results** | 2026-10-05 16:00:00 UTC",
        "Portfolio value: **$500.00**",
        "```",
        "Filled:",
        "  BUY  VTI    x1.0412    @ $  287.55  (DEPOSIT)  order id-VTI",
        "Unfilled:",
        "  BUY  VXUS   $100.00    — new  order id-VXUS",
        "Errors:",
        "  BUY  VEA    $100.00    — error: 422 insufficient buying power",
        "```",
    ]


def test_build_summary_says_none_for_empty_sections():
    msg = pa.build_summary([_row("VTI", "filled", 1.0, 100.0)], 100.0, "t")
    assert "Unfilled: none" in msg and "Errors:  none" in msg


def test_an_account_holding_other_symbols_is_refused(paths):
    api = FakeAPI(holdings={"VTI": 600.0, "VXUS": 400.0}, cash=1000.0, foreign=["AAPL"])
    assert pa.run(api, today="2026-10-05", **paths) == 1
    assert api.submitted == []


def test_band_breach_sells_first_then_invests_the_proceeds(paths):
    api = FakeAPI(holdings={"VTI": 6600.0, "VXUS": 3400.0}, cash=0.0)
    assert pa.run(api, today="2026-10-05", **paths) == 0
    assert api.submitted[0] == ("VTI", "sell", 600.0, None)
    # $600 of proceeds, $1 buffer: $599 to VXUS, the only fund short of target.
    assert api.submitted[1:] == [("VXUS", "buy", 599.0, None)]
    assert api.holdings["VXUS"] == pytest.approx(3999.0)


def test_cash_is_capped_at_the_deposited_cash_not_margin_buying_power(paths):
    api = FakeAPI(cash=100.0)                 # buying_power reports 200
    pa.run(api, today="2026-10-05", **paths)
    assert sum(n for _, side, n, _ in api.submitted if side == "buy") <= 99.0


def test_a_failed_order_is_logged_and_the_run_reports_failure(paths, monkeypatch):
    api = FakeAPI(cash=1000.0)
    real = api.submit_order

    def flaky(symbol, **kw):
        if symbol == "VTI":
            raise RuntimeError("422 insufficient buying power")
        return real(symbol, **kw)

    api.submit_order = flaky
    assert pa.run(api, today="2026-10-05", **paths) == 1
    assert api.submitted == [("VXUS", "buy", 399.60, None)]   # the other leg still went
    log = open(paths["log_path"]).read()
    assert "error: 422 insufficient buying power" in log


def test_connect_requires_the_allocators_own_keys(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "model-bot-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "model-bot-secret")
    monkeypatch.delenv("PASSIVE_ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("PASSIVE_ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="PASSIVE_ALPACA_API_KEY"):
        pa.connect()
