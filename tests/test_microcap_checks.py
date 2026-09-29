"""microcap_checks.py on small hand-built frames.

Each check's rule from docs/MICROCAP_VALUE_PREREG.md is pinned on a case
where the answer is known.
"""

from datetime import date, timedelta

import pandas as pd

import microcap_checks as mc


def _d(s):
    return date.fromisoformat(s)


def _actions(rows):
    return pd.DataFrame(rows, columns=["date", "action", "ticker", "contraname"]).assign(
        date=lambda f: f["date"].map(_d))


# --- guard and coverage ---------------------------------------------------

def test_guard_fails_on_a_free_tier_sized_universe():
    small = pd.DataFrame({"ticker": [f"T{i}" for i in range(30)]})
    big = pd.DataFrame({"ticker": [f"T{i}" for i in range(150)]})
    assert mc.check_guard(small).status == "FAIL"
    assert mc.check_guard(big).status == "PASS"


def _full_calendar():
    start, end = date(1997, 12, 31), date(2026, 9, 29)
    return [start + timedelta(days=i) for i in range((end - start).days + 1)
            if (start + timedelta(days=i)).weekday() < 5]


def test_coverage_passes_when_every_year_has_prices_and_filings():
    filings = pd.Series([date(y, 3, 1) for y in range(1998, 2026)])
    assert mc.check_coverage(_full_calendar(), filings).status == "PASS"


def test_coverage_fails_on_a_missing_filing_year():
    filings = pd.Series([date(y, 3, 1) for y in range(1998, 2026) if y != 2008])
    r = mc.check_coverage(_full_calendar(), filings)
    assert r.status == "FAIL" and "[2008]" in str(r)


def test_coverage_fails_when_prices_stop_before_the_last_holding_year_ends():
    cal = [d for d in _full_calendar() if d < date(2026, 6, 1)]
    filings = pd.Series([date(y, 3, 1) for y in range(1998, 2026)])
    assert mc.check_coverage(cal, filings).status == "FAIL"


# --- check 5 ----------------------------------------------------------------

TICKERS = pd.DataFrame([
    # table, permaticker, ticker, name, category, siccode, isdelisted, relatedtickers
    ["SF1", 1, "AAA", "A INC", "Domestic Common Stock", 3570, "N", None],
    ["SEP", 1, "AAA", "A INC", "Domestic Common Stock", 3570, "N", None],
    ["SF1", 2, "BRK", "B HOLDINGS", "Domestic Common Stock Primary Class", 3711, "N", "XYZU"],
    ["SEP", 2, "BRK", "B HOLDINGS", "Domestic Common Stock Primary Class", 3711, "N", "XYZU"],
    ["SEP", 3, "BRKB", "B HOLDINGS", "Domestic Common Stock Secondary Class", 3711, "N", None],
    # The old SPAC's units, renamed after the company they merged into.
    ["SEP", 8, "XYZU", "B HOLDINGS", "Domestic Common Stock Secondary Class", 6770, "Y", None],
    ["SEP", 9, "XYZ.U", "B HOLDINGS", "Domestic Common Stock Secondary Class", 6770, "Y", None],
    ["SEP", 10, "XYZU1", "B HOLDINGS", "Domestic Common Stock Secondary Class", 6770, "Y", None],
    ["SF1", 4, "CCC", "C CORP", "Domestic Common Stock Primary Class", 7372, "N", None],
    ["SEP", 5, "CCCB", "C CORP", "Domestic Common Stock Secondary Class", 7372, "N", None],
    ["SF1", 6, "SPCA", "SHELL ACQ", "Domestic Common Stock Primary Class", 6770, "N", None],
    ["SEP", 7, "SPCAB", "SHELL ACQ", "Domestic Common Stock Secondary Class", 6770, "N", None],
    ["SF1", 11, "BANK", "D BANCORP", "Domestic Common Stock Primary Class", 6022, "N", None],
    ["SEP", 12, "BANKB", "D BANCORP", "Domestic Common Stock Secondary Class", 6022, "N", None],
], columns=["table", "permaticker", "ticker", "name", "category", "siccode", "isdelisted",
            "relatedtickers"])


def test_ticker_join_passes_and_reports_unmapped_rows():
    r = mc.check_ticker_join(TICKERS, pd.Series({"AAA": 3, "ZZZ": 1}),
                             pd.Series({"AAA": 100}))
    assert r.status == "PASS"
    assert "fundamentals: 1 of 2 tickers (1 of 4 rows) have no permaticker" in str(r)


def test_ticker_join_fails_when_one_ticker_has_two_permatickers():
    bad = pd.concat([TICKERS, TICKERS.iloc[[1]].assign(permaticker=99)])
    r = mc.check_ticker_join(bad, pd.Series({"AAA": 1}), pd.Series({"AAA": 1}))
    assert r.status == "FAIL" and "AAA" in str(r)


# --- check 4 ----------------------------------------------------------------

def test_delisting_outcomes_follow_the_registered_rules():
    actions = _actions([
        ["2010-05-03", "delisted", "ACQ", None],
        ["2010-05-03", "acquisitionby", "ACQ", None],
        ["2011-01-10", "delisted", "BKR", None],
        ["2010-12-20", "bankruptcyliquidation", "BKR", None],   # 21 days before
        ["2012-06-01", "delisted", "BOTH", None],
        ["2012-06-01", "acquisitionby", "BOTH", None],
        ["2012-06-05", "regulatorydelisting", "BOTH", None],    # failure wins
        ["2013-03-01", "delisted", "FAR", None],
        ["2013-01-01", "acquisitionby", "FAR", None],           # 59 days: too far
        ["2014-01-01", "delisted", "ETF", None],                # not eligible
    ])
    d = mc.classify_delistings(actions, {"ACQ", "BKR", "BOTH", "FAR"})
    assert dict(zip(d["ticker"], d["outcome"])) == {
        "ACQ": "acquired", "BKR": "failed", "BOTH": "failed", "FAR": "unknown"}


def test_a_ticker_delisted_twice_gets_each_event_its_own_reason():
    actions = _actions([
        ["2005-01-03", "delisted", "RE", None],
        ["2005-01-03", "bankruptcyliquidation", "RE", None],
        ["2015-01-05", "delisted", "RE", None],
        ["2015-01-05", "acquisitionby", "RE", None],
    ])
    d = mc.classify_delistings(actions, {"RE"})
    assert list(d["outcome"]) == ["failed", "acquired"]


def test_delisting_report_shows_shares_and_rules():
    actions = _actions([["2010-05-03", "delisted", "X", None],
                        ["2011-05-03", "delisted", "Y", None],
                        ["2011-05-03", "acquisitionby", "Y", None]])
    out = str(mc.check_delistings(actions, {"X", "Y"}))
    assert "unknown         1  (50%)  -> -30% primary, 0% sensitivity" in out
    assert "2010s 50%" in out


# --- check 3 ----------------------------------------------------------------

def test_spac_shell_years_count_filings_before_the_merger():
    actions = _actions([["2021-06-01", "spacmerger", "SPCA", None]])
    fundamentals = pd.DataFrame({
        "ticker": ["SPCA", "SPCA", "SPCA"],
        "date": [_d("2020-03-01"), _d("2021-03-01"), _d("2022-03-01")],
        "revenue": [0.0, None, 5e6]})
    out = str(mc.check_spacs(actions, TICKERS, fundamentals))
    assert "current SIC 6770: 1" in out
    assert "ARY filings before the merger: 2, with zero or missing revenue: 2 (100%)" in out


# --- check 2 ----------------------------------------------------------------

def test_otc_periods_end_when_the_stock_moves_back():
    actions = _actions([
        ["2010-01-04", "exchangeto", "DMT", "OTC"],
        ["2012-01-03", "exchangeto", "DMT", "NASDAQ"],
        ["2015-01-05", "exchangeto", "DMT", "OTC"],
        ["2011-01-03", "exchangeto", "UP", "NYSE"],
    ])
    assert mc.otc_periods(actions, {"DMT", "UP"}) == [
        ("DMT", _d("2010-01-04"), _d("2012-01-03")),
        ("DMT", _d("2015-01-05"), None)]


def test_adv_uses_the_prior_252_trading_days_and_needs_126_of_them():
    cal = [d for d in _full_calendar() if date(2009, 1, 1) <= d <= date(2011, 12, 30)]
    d = next(x for x in cal if x >= date(2010, 7, 1))
    window = cal[cal.index(d) - 252:cal.index(d)]
    liquid = pd.DataFrame({"date": window, "close": 10.0, "volume": 20_000.0})
    assert mc.adv(liquid, cal, d) == 200_000
    sparse = liquid.iloc[:125]
    assert mc.adv(sparse, cal, d) is None
    # A huge day on d itself or earlier than the window is ignored.
    extra = pd.DataFrame({"date": [d, cal[cal.index(d) - 253]], "close": 1e6, "volume": 1e6})
    assert mc.adv(pd.concat([liquid, extra]), cal, d) == 200_000


def test_demoted_stock_liquid_on_otc_is_reported():
    cal = [d for d in _full_calendar() if date(2008, 1, 1) <= d <= date(2012, 12, 31)]
    prices = pd.DataFrame({"ticker": "DMT", "date": cal, "close": 5.0, "volume": 50_000.0})
    periods = [("DMT", date(2009, 1, 2), None)]
    out = str(mc.check_demoted(periods, prices, cal))
    assert "with price rows while on OTC: 1" in out
    # Rebalances 2009..2012 while on OTC, each with $250k ADV.
    assert "ADV >= $100,000: 4" in out


def test_demoted_stock_without_prices_after_the_move():
    cal = [d for d in _full_calendar() if date(2008, 1, 1) <= d <= date(2012, 12, 31)]
    prices = pd.DataFrame({"ticker": "GONE", "date": [d for d in cal if d < date(2009, 1, 2)],
                           "close": 5.0, "volume": 50_000.0})
    out = str(mc.check_demoted([("GONE", date(2009, 1, 2), None)], prices, cal))
    assert "with price rows while on OTC: 0" in out and "ADV >= $100,000: 0" in out


# --- check 1 ----------------------------------------------------------------

def test_dual_class_pairs_skip_spac_units_shells_and_financials():
    pairs = mc.dual_class_pairs(TICKERS)
    assert sorted(zip(pairs["ticker"], pairs["ticker_secondary"])) == [
        ("BRK", "BRKB"), ("CCC", "CCCB")]


def test_dual_class_report_lists_latest_share_count():
    fundamentals = pd.DataFrame({
        "ticker": ["BRK", "BRK", "CCC"],
        "date": [_d("2024-02-20"), _d("2025-02-21"), _d("2025-03-01")],
        "sharesbas": [1_000.0, 1_100.0, 500.0]})
    r = mc.check_dual_class(TICKERS, fundamentals)
    out = str(r)
    assert r.status == "MANUAL"
    assert "dual-class companies: 2 (2 still listed)" in out
    assert "BRK" in out and "sharesbas 1,100 as filed 2025-02-21" in out
