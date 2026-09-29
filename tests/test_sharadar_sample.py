"""sharadar_sample.py against a fake Nasdaq Data Link session.

CI has no Sharadar key, so these pin the script's own logic: cursor paging,
errors that never leak the key, and a report that shows structure but never
data values.
"""

import pytest
import requests

import sharadar_sample as ss

KEY = "secret-key-123"


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body or {}

    def json(self):
        return self._body


def _page(columns, rows, cursor=None):
    return FakeResponse(body={
        "datatable": {"columns": [{"name": c, "type": "String"} for c in columns],
                      "data": rows},
        "meta": {"next_cursor_id": cursor},
    })


def _denied(code="QEPx04", message="You do not have permission to view this dataset."):
    return FakeResponse(403, {"quandl_error": {"code": code, "message": message}})


class FakeSession:
    """Serves pages by table; records every query it was sent."""

    def __init__(self, tables):
        self.tables = tables  # table -> list of responses, or a single response
        self.calls = []

    def get(self, url, params=None, timeout=None):
        table = url.rsplit("/", 1)[-1].removesuffix(".json")
        self.calls.append((table, dict(params)))
        served = self.tables[table]
        if isinstance(served, Exception):
            raise served
        if not isinstance(served, list):
            return served
        cursor = params.get("qopts.cursor_id")
        index = 0 if cursor is None else int(cursor)
        return served[index]


TICKER_COLS = ["table", "permaticker", "ticker", "category", "exchange", "isdelisted", "siccode"]
TICKER_ROWS = [
    ["SF1", 1, "AAA", "Domestic Common Stock", "NASDAQ", "N", 3570],
    ["SF1", 2, "BBB", "Domestic Common Stock Primary Class", "NYSE", "N", 4841],
    ["SF1", 3, "BBBB", "Domestic Common Stock Secondary Class", "NYSE", "N", 4841],
    ["SF1", 4, "SPAC", "Domestic Common Stock", "NASDAQ", "Y", 6770],
    ["SF1", 5, "DEAD", "Domestic Common Stock", "OTC", "Y", 2834],
]


def _full_session(**overrides):
    tables = {
        "TICKERS": _page(TICKER_COLS, TICKER_ROWS),
        "ACTIONS": _page(["action"], [["acquisitionby"], ["delisted"], ["delisted"]]),
        "SF1": _page(["ticker", "datekey", "ebit"], [["AAA", "2020-03-01", 987654321]]),
        "SEP": _page(["ticker", "date", "closeadj"], [["AAA", "2020-03-02", 123.45]]),
        "DAILY": _denied(),
    }
    tables.update(overrides)
    return FakeSession(tables)


def test_fetch_follows_cursor_pages_and_keeps_the_query():
    session = FakeSession({"ACTIONS": [_page(["action"], [["a"]], cursor="1"),
                                       _page(["action"], [["b"]], cursor=None)]})
    columns, rows = ss.fetch(session, KEY, "ACTIONS", {"qopts.columns": "action"})
    assert columns == ["action"]
    assert rows == [["a"], ["b"]]
    assert session.calls[1][1]["qopts.cursor_id"] == "1"
    assert session.calls[1][1]["qopts.columns"] == "action"


def test_fetch_gives_up_after_max_pages():
    session = FakeSession({"ACTIONS": [_page(["action"], [["a"]], cursor="0")]})
    with pytest.raises(ss.SharadarError, match="more than 3 pages"):
        ss.fetch(session, KEY, "ACTIONS", max_pages=3)


def test_http_error_reports_the_api_message_without_the_key():
    session = FakeSession({"SEP": _denied()})
    with pytest.raises(ss.SharadarError) as err:
        ss.fetch(session, KEY, "SEP")
    assert "HTTP 403" in str(err.value) and "QEPx04" in str(err.value)
    assert KEY not in str(err.value)


def test_network_error_does_not_leak_the_url():
    exc = requests.ConnectionError(f"failed: {ss.BASE_URL}/SEP.json?api_key={KEY}")
    session = FakeSession({"SEP": exc})
    with pytest.raises(ss.SharadarError) as err:
        ss.fetch(session, KEY, "SEP")
    assert KEY not in str(err.value)
    assert err.value.__cause__ is None and err.value.__suppress_context__


def test_columns_of_asks_for_a_single_row_even_when_more_pages_exist():
    session = FakeSession({"SEP": _page(["ticker", "closeadj"], [["AAA", 1.0]], cursor="9")})
    assert ss.columns_of(session, KEY, "SEP") == ["ticker", "closeadj"]
    assert session.calls == [("SEP", {"api_key": KEY, "qopts.per_page": 1})]


def test_summarise_tickers():
    s = ss.summarise_tickers(ss.as_dicts(TICKER_COLS, TICKER_ROWS))
    assert s["companies"] == 5
    assert s["delisted"] == 2
    assert s["by_category"]["Domestic Common Stock"] == 3
    assert s["by_exchange"]["OTC"] == 1
    assert s["dual_class"] == [("BBB", "Domestic Common Stock Primary Class"),
                               ("BBBB", "Domestic Common Stock Secondary Class")]
    assert s["blank_check"] == [("SPAC", "Y")]


def test_main_needs_the_key(monkeypatch):
    monkeypatch.delenv("SHARADAR_API_KEY", raising=False)
    assert ss.main(["--discover"], session=_full_session()) == 2


def test_report_shows_structure_and_never_data_values(monkeypatch, capsys):
    monkeypatch.setenv("SHARADAR_API_KEY", KEY)
    assert ss.main(["--discover"], session=_full_session()) == 0
    out = capsys.readouterr().out
    assert "OK    SEP: ticker, date, closeadj" in out
    assert "NO    DAILY: HTTP 403" in out
    assert "SPAC  (Y)" in out
    assert "      2  delisted" in out
    for value in ("123.45", "987654321", "2020-03-0"):
        assert value not in out
    assert KEY not in out


def test_tickers_query_is_limited_to_sf1_companies(monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", KEY)
    session = _full_session()
    ss.main(["--discover"], session=session)
    full_calls = [p for t, p in session.calls if t == "TICKERS" and p["qopts.per_page"] != 1]
    assert full_calls and all(p["table"] == "SF1" for p in full_calls)


def test_denied_tickers_is_a_failure(monkeypatch, capsys):
    monkeypatch.setenv("SHARADAR_API_KEY", KEY)
    assert ss.main(["--discover"], session=_full_session(TICKERS=_denied())) == 1
    assert "NO    TICKERS" in capsys.readouterr().out


def test_an_oversized_actions_table_does_not_hide_tickers(monkeypatch, capsys):
    monkeypatch.setenv("SHARADAR_API_KEY", KEY)
    endless = [_page(["action"], [["dividend"]], cursor="0")]
    assert ss.main(["--discover"], session=_full_session(ACTIONS=endless)) == 0
    out = capsys.readouterr().out
    assert "NO    ACTIONS" in out and "more than" in out
    assert "TICKERS (SF1 companies): 5" in out
