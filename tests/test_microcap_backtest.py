"""microcap_backtest.py on synthetic data only.

Each registered rule in docs/MICROCAP_VALUE_PREREG.md is pinned on a case
whose answer is known in advance. No real data is read here.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import microcap_backtest as mb


def _weekdays(start, end):
    d, out = start, []
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


CAL = _weekdays(date(2008, 1, 1), date(2011, 12, 30))


def _prices(spec):
    """spec: ticker -> dict(days=list, close=, high=, low=, volume=, closeadj=, closeunadj=)."""
    rows = []
    for t, s in spec.items():
        days = s.get("days", CAL)
        n = len(days)

        def col(name, default):
            v = s.get(name, default)
            return list(v) if np.ndim(v) else [v] * n

        close = col("close", 10.0)
        rows.append(pd.DataFrame({
            "ticker": t, "date": days, "close": close,
            "high": col("high", None) if "high" in s else [c * 1.01 for c in close],
            "low": col("low", None) if "low" in s else [c * 0.99 for c in close],
            "volume": col("volume", 50_000.0),
            "closeadj": col("closeadj", None) if "closeadj" in s else close,
            "closeunadj": col("closeunadj", None) if "closeunadj" in s else close,
        }))
    df = pd.concat(rows, ignore_index=True)
    return mb.PriceStore(df["ticker"].to_numpy(), mb._days(df["date"]), df["high"], df["low"],
                         df["close"], df["volume"], df["closeadj"], df["closeunadj"])


def _d(s):
    return mb._day(date.fromisoformat(s))


def _first_on_or_after(d):
    return next(x for x in CAL if x >= d)


# --- PriceStore -----------------------------------------------------------

def test_adv_uses_the_252_days_before_d_and_skips_zero_volume_days():
    d = _first_on_or_after(date(2010, 7, 1))
    k = CAL.index(d)
    vol = [0.0 if i % 2 else 20_000.0 for i in range(len(CAL))]
    vol[k] = 1e12                                   # the day itself is excluded
    store = _prices({"A": {"volume": vol}})
    adv = store.adv(store.ids(["A"]), mb._day(d))[0]
    assert adv == pytest.approx(10.0 * 20_000)      # mean over volume days only


def test_adv_needs_126_volume_days():
    d = _first_on_or_after(date(2010, 7, 1))
    days = CAL[CAL.index(d) - 125:]
    store = _prices({"A": {"days": days}})
    assert np.isnan(store.adv(store.ids(["A"]), mb._day(d))[0])


def test_abdi_ranaldo_recovers_a_known_spread():
    # Mid fixed at 10; the close alternates at bid and ask, high = ask, low = bid.
    s = 0.02
    ask, bid = 10 * np.exp(s / 2), 10 * np.exp(-s / 2)
    close = [ask if i % 2 else bid for i in range(len(CAL))]
    store = _prices({"A": {"close": close, "high": ask, "low": bid}})
    d = mb._day(_first_on_or_after(date(2010, 7, 1)))
    assert store.spread(store.ids(["A"]), d)[0] == pytest.approx(s, rel=1e-9)


def test_close_on_needs_a_trade_that_day_and_ratio_looks_back():
    days = [x for x in CAL if x != date(2010, 7, 1)]
    store = _prices({"A": {"days": days, "close": 5.0, "closeunadj": 10.0}, "B": {}})
    ids = store.ids(["A"])
    assert np.isnan(store.close_on(ids, _d("2010-07-01"))[0])
    assert store.close_on(ids, _d("2010-07-02"))[0] == 5.0
    assert store.unadj_ratio_on_or_before(ids, _d("2010-07-01"))[0] == 2.0
    assert np.isnan(store.unadj_ratio_on_or_before(ids, _d("2007-01-01"))[0])


def test_adj_panel_carries_prices_over_gaps_and_stops_after_the_last_day():
    days = [x for x in CAL if x <= date(2010, 7, 9) and x != date(2010, 7, 6)]
    # B trades every day, so the calendar has A's gap day, as in the real data.
    store = _prices({"A": {"days": days, "close": range(1, len(days) + 1)}, "B": {}})
    panel, pdays = store.adj_panel(store.ids(["A"]), _d("2010-07-02"), _d("2010-07-13"))
    col = dict(zip(pdays.tolist(), panel[:, 0]))
    assert col[_d("2010-07-06")] == col[_d("2010-07-05")]      # gap: carried
    assert np.isnan(col[_d("2010-07-12")])                      # after the last day


# --- universe -------------------------------------------------------------

D = mb._day(_first_on_or_after(date(2010, 7, 1)))


def _fund(rows):
    df = pd.DataFrame(rows, columns=["ticker", "date", "reportperiod", "ebit", "debt",
                                     "cashneq", "sharesbas"])
    return df.assign(day=mb._days(df["date"]), reportday=mb._days(df["reportperiod"]))


def _ref(tickers, spac=None, moves=None, reasons=None):
    sec = pd.DataFrame({"ticker": tickers, "permaticker": range(1, len(tickers) + 1)})
    return mb.Reference(sec, spac or {}, moves or {}, reasons or {})


def _universe(fund_rows, price_spec, **ref_kwargs):
    store = _prices(price_spec)
    ref = _ref(list(price_spec), **ref_kwargs)
    return mb.build_universe(D, _fund(fund_rows), ref, store)


GOOD = ["2010-03-01", "2009-12-31", 10e6, 5e6, 2e6, 10e6]   # mcap $100M at $10


def test_a_company_meeting_every_rule_is_in_with_its_ebit_ev():
    u = _universe([["A", *GOOD]], {"A": {}})
    assert list(u["ticker"]) == ["A"]
    assert u["mcap"].iloc[0] == pytest.approx(100e6)
    assert u["ebit_ev"].iloc[0] == pytest.approx(10e6 / (100e6 + 5e6 - 2e6))


def test_filing_on_d_itself_is_not_yet_usable():
    d_str = str(mb._date(D))
    u = _universe([["A", *GOOD],
                   ["A", d_str, "2010-03-31", -99e6, 5e6, 2e6, 10e6]], {"A": {}})
    assert list(u["ebit"]) == [10e6]


def test_stale_filing_is_out():
    # Fiscal year ended 2008-12-31: more than 18 months before July 2010.
    u = _universe([["A", "2009-03-01", "2008-12-31", 10e6, 5e6, 2e6, 10e6]], {"A": {}})
    assert u.empty


@pytest.mark.parametrize("shares,expected", [(4e6, False), (10e6, True), (40e6, False)])
def test_market_cap_band(shares, expected):
    u = _universe([["A", "2010-03-01", "2009-12-31", 10e6, 5e6, 2e6, shares]], {"A": {}})
    assert (len(u) == 1) is expected


def test_market_cap_puts_reported_shares_on_the_split_adjusted_basis():
    # A 2-for-1 split after the filing: adjusted close 10 = unadjusted 20 at the filing.
    split = date(2010, 5, 3)
    unadj = [20.0 if x < split else 10.0 for x in CAL]
    u = _universe([["A", "2010-03-01", "2009-12-31", 10e6, 5e6, 2e6, 5e6]],
                  {"A": {"closeunadj": unadj}})
    assert u["mcap"].iloc[0] == pytest.approx(100e6)    # 10 x 5M x (20/10)


def test_illiquid_stock_is_out():
    u = _universe([["A", *GOOD]], {"A": {"volume": 5_000.0}})   # $50k a day
    assert u.empty


def test_no_trade_on_d_is_out():
    days = [x for x in CAL if mb._day(x) != D]
    assert _universe([["A", *GOOD]], {"A": {"days": days}}).empty


def test_missing_inputs_are_out_not_zero():
    u = _universe([["A", "2010-03-01", "2009-12-31", 10e6, None, 2e6, 10e6]], {"A": {}})
    assert u.empty


def test_negative_ev_is_out():
    u = _universe([["A", "2010-03-01", "2009-12-31", 10e6, 0.0, 150e6, 10e6]], {"A": {}})
    assert u.empty


@pytest.mark.parametrize("ebit,debt,cash,inside", [
    (10e6, 35e6, 2e6, False),     # net debt 33M > 3 x 10M
    (10e6, 32e6, 2e6, True),      # net debt 30M = 3 x 10M
    (-5e6, 10e6, 2e6, False),     # losing money with net debt
    (-5e6, 1e6, 2e6, True),       # losing money with net cash
])
def test_debt_filter_is_part_of_the_universe(ebit, debt, cash, inside):
    u = _universe([["A", "2010-03-01", "2009-12-31", ebit, debt, cash, 10e6]], {"A": {}})
    assert (len(u) == 1) is inside


def test_spac_shell_before_its_merger_and_otc_listing_are_out():
    spac_later = {"A": D + 30}
    assert _universe([["A", *GOOD]], {"A": {}}, spac=spac_later).empty
    assert len(_universe([["A", *GOOD]], {"A": {}}, spac={"A": D - 30})) == 1
    otc = {"A": (np.array([D - 400]), np.array(["OTC"]))}
    back = {"A": (np.array([D - 400, D - 100]), np.array(["OTC", "NASDAQ"]))}
    assert _universe([["A", *GOOD]], {"A": {}}, moves=otc).empty
    assert len(_universe([["A", *GOOD]], {"A": {}}, moves=back)) == 1


def test_ranking_cheapest_first_with_ties_by_permaticker():
    rows = [[t, "2010-03-01", "2009-12-31", e, 5e6, 2e6, 10e6]
            for t, e in [("A", 5e6), ("B", 9e6), ("C", 9e6), ("D", 1e6)]]
    u = _universe(rows, {t: {} for t in "ABCD"})
    assert list(u["ticker"]) == ["B", "C", "A", "D"]


def test_quiet_stock_gets_the_spread_floor():
    u = _universe([["A", *GOOD]], {"A": {"high": 10.0, "low": 10.0}})
    assert u["half_spread"].iloc[0] == pytest.approx(mb.SPREAD_FLOOR / 2)


def test_portfolios_top_30_control_all_and_five_quintiles():
    universe = pd.DataFrame({"sid": np.arange(100)})
    books = mb.portfolios(universe)
    assert list(books["strategy"]) == list(range(30))
    assert len(books["control"]) == 100
    assert list(books["Q1"]) == list(range(20)) and list(books["Q5"]) == list(range(80, 100))


# --- one holding year -----------------------------------------------------

Y0 = _first_on_or_after(date(2010, 7, 1))
Y1 = _first_on_or_after(date(2011, 7, 1))


def _grow(rate):
    """closeadj compounding at rate per day from 1.0."""
    return [(1 + rate) ** i for i in range(len(CAL))]


def test_year_return_is_price_change_minus_opening_cost():
    store = _prices({"A": {"closeadj": _grow(0.001), "high": 10.0, "low": 10.0}})
    ret, end = mb.simulate_year(store.ids(["A"]), {}, mb._day(Y0), mb._day(Y1), store,
                                _ref(["A"]), 0.01, -0.3)
    growth = (1.001) ** (CAL.index(Y1) - CAL.index(Y0))
    assert ret == pytest.approx((1 - mb.SPREAD_FLOOR / 2) * growth - 1)
    assert end == {int(store.ids(["A"])[0]): pytest.approx(1.0)}


def test_kept_names_pay_only_on_the_weight_traded():
    store = _prices({"A": {"high": 10.0, "low": 10.0}, "B": {"high": 10.0, "low": 10.0}})
    a, b = store.ids(["A", "B"])
    ret, _ = mb.simulate_year(np.array([a, b]), {int(a): 0.6, int(b): 0.4}, mb._day(Y0),
                              mb._day(Y1), store, _ref(["A", "B"]), 0.01, -0.3)
    assert ret == pytest.approx(-(mb.SPREAD_FLOOR / 2) * 0.2)   # |0.5-0.6| + |0.5-0.4|


def _delisting_case(reason_action, unknown_return):
    last = date(2011, 1, 3)
    store = _prices({"DEAD": {"days": [x for x in CAL if x <= last], "high": 10.0, "low": 10.0},
                     "LIVE": {"high": 10.0, "low": 10.0}})
    reasons = {}
    if reason_action:
        reasons["DEAD"] = (np.array([mb._day(last) + 5]), np.array([reason_action]))
    ref = _ref(["DEAD", "LIVE"], reasons=reasons)
    ret, end = mb.simulate_year(store.ids(["DEAD", "LIVE"]), {}, mb._day(Y0), mb._day(Y1),
                                store, ref, 0.01, unknown_return)
    return ret, end, store


@pytest.mark.parametrize("action,unknown,dead_return", [
    ("bankruptcyliquidation", 0.0, -0.30),
    ("acquisitionby", -0.30, 0.0),
    (None, -0.30, -0.30),
    (None, 0.0, 0.0),
])
def test_delisting_return_by_reason_and_reinvestment(action, unknown, dead_return):
    ret, end, store = _delisting_case(action, unknown)
    hs = mb.SPREAD_FLOOR / 2
    half = (1 - hs) / 2
    proceeds = half * (1 + dead_return)
    expected = half + proceeds * (1 - hs) - 1
    assert ret == pytest.approx(expected)
    assert list(end) == [int(store.ids(["LIVE"])[0])]


def test_failure_wins_over_acquisition():
    last = date(2011, 1, 3)
    store = _prices({"DEAD": {"days": [x for x in CAL if x <= last]}})
    reasons = {"DEAD": (np.array([mb._day(last), mb._day(last) + 3]),
                        np.array(["acquisitionby", "regulatorydelisting"]))}
    ref = _ref(["DEAD"], reasons=reasons)
    assert ref.delisting_return("DEAD", mb._day(last), 0.0) == -0.30


def test_all_names_delisted_leaves_cash():
    last = date(2011, 1, 3)
    store = _prices({"DEAD": {"days": [x for x in CAL if x <= last], "high": 10.0, "low": 10.0}})
    ret, end = mb.simulate_year(store.ids(["DEAD"]), {}, mb._day(Y0), mb._day(Y1), store,
                                _ref(["DEAD"]), 0.01, 0.0)
    assert ret == pytest.approx(-mb.SPREAD_FLOOR / 2)
    assert end == {mb.CASH: 1.0}


def test_a_trading_gap_is_not_a_delisting():
    gap = [x for x in CAL if not (date(2011, 1, 3) <= x <= date(2011, 2, 28))]
    store = _prices({"A": {"days": gap, "high": 10.0, "low": 10.0}})
    ret, end = mb.simulate_year(store.ids(["A"]), {}, mb._day(Y0), mb._day(Y1), store,
                                _ref(["A"]), 0.01, -0.3)
    assert ret == pytest.approx(-mb.SPREAD_FLOOR / 2)
    assert mb.CASH not in end


# --- evaluation -----------------------------------------------------------

def _years(strategy, control, quintiles):
    return [{"strategy": s, "control": c, **{f"Q{i + 1}": q[i] for i in range(5)}}
            for s, c, q in zip(strategy, control, quintiles)]


STAIRS = [0.10, 0.08, 0.06, 0.04, 0.02]


def test_pass_needs_22_wins_a_strict_staircase_and_3pct_excess():
    good = _years([0.12] * 22 + [0.0] * 5, [0.05] * 27, [STAIRS] * 27)
    r = mb.evaluate(good)
    assert r["wins"] == 22 and r["staircase"] and r["pass"]

    one_short = _years([0.12] * 21 + [0.0] * 6, [0.05] * 27, [STAIRS] * 27)
    assert not mb.evaluate(one_short)["pass"]

    swapped = _years([0.12] * 27, [0.05] * 27, [[0.10, 0.08, 0.04, 0.06, 0.02]] * 27)
    assert not mb.evaluate(swapped)["pass"]

    thin = _years([0.06] * 27, [0.05] * 27, [STAIRS] * 27)
    assert not mb.evaluate(thin)["pass"]


def test_verdict_passing_only_at_zero_is_inconclusive():
    ok, bad = {"pass": True}, {"pass": False}
    assert mb.verdict(ok, ok) == "PASS"
    assert mb.verdict(bad, ok) == "INCONCLUSIVE"
    assert mb.verdict(bad, bad) == "FAIL"


def test_rebalance_days_first_trading_day_on_or_after_july_1():
    cal = mb._days(_weekdays(date(1998, 1, 1), date(2026, 9, 30)))
    days = mb.rebalance_days(np.asarray(cal))
    assert len(days) == 28                       # 27 holding years
    assert mb._date(days[0]) == date(1999, 7, 1)
    assert mb._date(days[-1]) == date(2026, 7, 1)


# --- the run guard --------------------------------------------------------

def test_refuses_to_run_from_a_dirty_tree(monkeypatch, capsys):
    monkeypatch.setattr(mb, "git_state", lambda: ("abc", True))
    assert mb.main() == 2
    assert "uncommitted changes" in capsys.readouterr().out


def test_refuses_to_run_twice(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mb, "git_state", lambda: ("abc", False))
    existing = tmp_path / "result.md"
    existing.write_text("already ran")
    monkeypatch.setattr(mb, "RESULT_MD", existing)
    assert mb.main() == 2
    assert "second draw" in capsys.readouterr().out
