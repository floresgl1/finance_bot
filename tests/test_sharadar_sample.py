"""sharadar_sample.py against a fake api.sharadar.com session.

CI has no Sharadar key, so these pin the script's own logic: limit/skip
paging, stopping on rate limits, errors that never leak the key, the
free-tier warning, and a report that shows structure but never data values.
"""

import pytest
import requests

import sharadar_sample as ss

KEY = "secret-key-123"


class FakeResponse:
    def __init__(self, status_code=200, body=None, remaining="4000"):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.headers = {} if remaining is None else {"X-RateLimit-Remaining": remaining}

    def json(self):
        return self._body


def _rows(fields, values):
    return [dict(zip(fields, v)) for v in values]


def _ok(rows, remaining="4000"):
    return FakeResponse(body={"count": len(rows), "data": rows}, remaining=remaining)


class FakeSession:
    """Serves each table's rows, honouring limit/skip; records every query."""

    def __init__(self, tables):
        self.tables = tables  # table -> list of row dicts, a FakeResponse, or an Exception
        self.calls = []

    def get(self, url, params=None, timeout=None):
        table = url.rsplit("/", 1)[-1]
        self.calls.append((table, dict(params)))
        served = self.tables[table]
        if isinstance(served, Exception):
            raise served
        if isinstance(served, FakeResponse):
            return served
        skip, limit = params.get("skip", 0), params["limit"]
        return _ok(served[skip:skip + limit])


TICKER_FIELDS = ["table", "permaticker", "ticker", "category", "exchange", "isdelisted", "siccode"]
TICKER_ROWS = _rows(TICKER_FIELDS, [
    ["fundamentals", "1", "AAA", "Domestic Common Stock", "NASDAQ", "N", "3570"],
    ["fundamentals", "2", "BBB", "Domestic Common Stock Primary Class", "NYSE", "N", "4841"],
    ["fundamentals", "3", "BBBB", "Domestic Common Stock Secondary Class", "NYSE", "N", "4841"],
    ["fundamentals", "4", "SPAC", "Domestic Common Stock", "NASDAQ", "Y", "6770"],
    ["fundamentals", "5", "DEAD", "Domestic Common Stock", "OTC", "Y", "2834"],
])


def _full_session(**overrides):
    tables = {
        "tickers": TICKER_ROWS,
        "actions": [{"action": a} for a in ("acquisitionby", "delisted", "delisted")],
        "fundamentals": _rows(["ticker", "dimension", "date", "ebit"],
                              [["AAA", "ARY", "2020-03-01", "987654321"]]),
        "stocks": _rows(["ticker", "date", "closeadj"], [["AAA", "2020-03-02", "123.45"]]),
        "daily": FakeResponse(403, {"message": "Not available on your plan"}),
    }
    tables.update(overrides)
    return FakeSession(tables)


def test_fetch_pages_with_skip_until_a_short_page():
    rows = [{"action": str(i)} for i in range(5)]
    session = FakeSession({"actions": rows})
    got = ss.fetch(session, KEY, "actions", {"limit": 2, "from": "1990-01-01"})
    assert got == rows
    assert [p["skip"] for _, p in session.calls] == [0, 2, 4]
    assert all(p["from"] == "1990-01-01" and p["format"] == "json" for _, p in session.calls)


def test_fetch_gives_up_after_max_pages():
    session = FakeSession({"actions": [{"action": "x"}] * 10})
    with pytest.raises(ss.SharadarError, match="more than 3 pages"):
        ss.fetch(session, KEY, "actions", {"limit": 1}, max_pages=3)


def test_http_error_reports_the_api_message_without_the_key():
    session = FakeSession({"stocks": FakeResponse(403, {"message": "Not on your plan"})})
    with pytest.raises(ss.SharadarError) as err:
        ss.fetch(session, KEY, "stocks")
    assert "HTTP 403 Not on your plan" in str(err.value)
    assert KEY not in str(err.value)


def test_network_error_does_not_leak_the_url():
    exc = requests.ConnectionError(f"failed: {ss.BASE_URL}/stocks?api_key={KEY}")
    session = FakeSession({"stocks": exc})
    with pytest.raises(ss.SharadarError) as err:
        ss.fetch(session, KEY, "stocks")
    assert KEY not in str(err.value)
    assert err.value.__suppress_context__


def test_low_rate_budget_counts_as_rate_limited():
    session = FakeSession({"stocks": _ok([{"ticker": "A"}], remaining="10")})
    with pytest.raises(ss.RateLimited, match="only 10 requests left"):
        ss.fetch(session, KEY, "stocks")


def test_fields_of_asks_for_one_row():
    session = _full_session()
    assert ss.fields_of(session, KEY, "stocks") == ["ticker", "date", "closeadj"]
    assert session.calls == [("stocks", {"api_key": KEY, "format": "json", "limit": 1})]


def test_summarise_tickers():
    s = ss.summarise_tickers(TICKER_ROWS)
    assert s["companies"] == 5
    assert s["delisted"] == 2
    assert s["by_category"]["Domestic Common Stock"] == 3
    assert s["by_exchange"]["OTC"] == 1
    assert s["dual_class"] == [("BBB", "Domestic Common Stock Primary Class"),
                               ("BBBB", "Domestic Common Stock Secondary Class")]
    assert s["blank_check"] == [("SPAC", "Y")]
    assert s["shared_tickers"] == []


def test_summarise_tickers_flags_a_ticker_with_two_permatickers():
    rows = TICKER_ROWS + [dict(TICKER_ROWS[0], permaticker="99")]
    assert ss.summarise_tickers(rows)["shared_tickers"] == [("AAA", "1, 99")]


def test_main_needs_the_key(monkeypatch):
    monkeypatch.delenv("SHARADAR_API_KEY", raising=False)
    assert ss.main(["--discover"], session=_full_session()) == 2


def test_report_shows_structure_and_never_data_values(monkeypatch, capsys):
    monkeypatch.setenv("SHARADAR_API_KEY", KEY)
    assert ss.main(["--discover"], session=_full_session()) == 0
    out = capsys.readouterr().out
    assert "OK    stocks: ticker, date, closeadj" in out
    assert "NO    daily: HTTP 403 Not available on your plan" in out
    assert "SPAC  (Y)" in out
    assert "      2  delisted" in out
    for value in ("123.45", "987654321", "2020-03-0"):
        assert value not in out
    assert KEY not in out


def test_queries_cover_full_history_and_fundamentals_companies(monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", KEY)
    session = _full_session()
    ss.main(["--discover"], session=session)
    full = [(t, p) for t, p in session.calls if p["limit"] != 1]
    assert all(p["table"] == "fundamentals" for t, p in full if t == "tickers")
    assert all(p["from"] == ss.HISTORY_START for t, p in full if t == "actions")
    assert [p["dimension"] for t, p in full if t == "fundamentals"] == ["ARY"]


def test_small_data_coverage_warns_about_the_free_tier(monkeypatch, capsys):
    monkeypatch.setenv("SHARADAR_API_KEY", KEY)
    ss.main(["--discover"], session=_full_session())
    out = capsys.readouterr().out
    assert "Data coverage: 1 distinct tickers" in out
    assert "WARNING: under 100" in out


def test_a_large_public_tickers_table_does_not_hide_a_small_data_tier(monkeypatch, capsys):
    # The tickers table is public, so thousands of companies there prove nothing.
    monkeypatch.setenv("SHARADAR_API_KEY", KEY)
    many = [dict(TICKER_ROWS[0], ticker=f"T{i}", permaticker=str(i)) for i in range(150)]
    ss.main(["--discover"], session=_full_session(tickers=many))
    assert "WARNING: under 100" in capsys.readouterr().out


def test_full_data_coverage_does_not_warn(monkeypatch, capsys):
    monkeypatch.setenv("SHARADAR_API_KEY", KEY)
    rows = [{"ticker": f"T{i}", "dimension": "ARY"} for i in range(150)]
    ss.main(["--discover"], session=_full_session(fundamentals=rows))
    out = capsys.readouterr().out
    assert "Data coverage: 150 distinct tickers" in out
    assert "WARNING" not in out


def test_a_429_stops_every_further_request(monkeypatch, capsys):
    monkeypatch.setenv("SHARADAR_API_KEY", KEY)
    session = _full_session(tickers=FakeResponse(429, {"message": "slow down"}))
    assert ss.main(["--discover"], session=session) == 1
    assert len(session.calls) == 1
    out = capsys.readouterr().out
    assert "STOPPED EARLY: tickers: HTTP 429" in out
    assert "NO    actions: not tried" in out


def test_denied_tickers_is_a_failure(monkeypatch, capsys):
    monkeypatch.setenv("SHARADAR_API_KEY", KEY)
    denied = FakeResponse(403, {"message": "no"})
    assert ss.main(["--discover"], session=_full_session(tickers=denied)) == 1
    assert "NO    tickers" in capsys.readouterr().out


def test_an_oversized_actions_table_does_not_hide_tickers(monkeypatch, capsys):
    monkeypatch.setenv("SHARADAR_API_KEY", KEY)
    monkeypatch.setattr(ss, "MAX_PAGES", 2)
    monkeypatch.setattr(ss, "PAGE_SIZE", 1)
    session = _full_session(tickers=TICKER_ROWS[:1])
    assert ss.main(["--discover"], session=session) == 0
    out = capsys.readouterr().out
    assert "NO    actions" in out and "more than 2 pages" in out
    assert "tickers (companies with fundamentals): 1" in out
