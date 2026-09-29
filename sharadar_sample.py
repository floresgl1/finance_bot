"""
Discovery run against Sharadar on Nasdaq Data Link, before the full download.

docs/MICROCAP_VALUE_PREREG.md lists four checks on the free sample. They need
particular kinds of company (dual-class, merged SPAC, demoted to OTC,
delisted), and the free tier may not cover them. This script asks what the key
can see and prints STRUCTURE only:

  - which tables respond, and their columns
  - TICKERS: companies by category, delisted count, exchanges,
    dual-class and SIC 6770 (blank-check) companies
  - ACTIONS: event types and how often each appears

It never prints prices, returns or rankings, and writes nothing to disk: the
pre-registration allows the sample for structure only, and the data licence
likely forbids republishing it.

    python sharadar_sample.py --discover

Needs SHARADAR_API_KEY in the environment. Exit 0 if TICKERS responded.
"""

import argparse
import os
import sys
from collections import Counter

import requests

BASE_URL = "https://data.nasdaq.com/api/v3/datatables/SHARADAR"
TABLES = ["TICKERS", "ACTIONS", "SF1", "SEP", "DAILY"]
PER_PAGE = 10000
MAX_PAGES = 50
LIST_LIMIT = 25


class SharadarError(Exception):
    pass


def _get_page(session, table, query):
    """One request: (column names, rows, next cursor or None).

    Errors never include the request URL, which carries the key.
    """
    try:
        resp = session.get(f"{BASE_URL}/{table}.json", params=query, timeout=60)
    except requests.RequestException as exc:
        raise SharadarError(f"{table}: network error ({type(exc).__name__})") from None
    if resp.status_code != 200:
        raise SharadarError(f"{table}: HTTP {resp.status_code} {_api_message(resp)}")
    body = resp.json()
    table_data = body["datatable"]
    columns = [c["name"] for c in table_data["columns"]]
    cursor = (body.get("meta") or {}).get("next_cursor_id")
    return columns, table_data["data"], cursor


def columns_of(session, api_key, table):
    """Column names, from a one-row request whose row is discarded."""
    columns, _, _ = _get_page(session, table, {"api_key": api_key, "qopts.per_page": 1})
    return columns


def fetch(session, api_key, table, params=None, max_pages=MAX_PAGES):
    """Return (column names, rows) for one datatable, following cursor pages."""
    query = {"api_key": api_key, "qopts.per_page": PER_PAGE, **(params or {})}
    rows = []
    for _ in range(max_pages):
        columns, page, cursor = _get_page(session, table, query)
        rows.extend(page)
        if not cursor:
            return columns, rows
        query["qopts.cursor_id"] = cursor
    raise SharadarError(f"{table}: more than {max_pages} pages; narrow the query")


def _api_message(resp):
    try:
        err = resp.json().get("quandl_error") or {}
        return f"{err.get('code', '')} {err.get('message', '')}".strip()
    except ValueError:
        return ""


def as_dicts(columns, rows):
    return [dict(zip(columns, r)) for r in rows]


def summarise_tickers(tickers):
    """Structural counts from TICKERS rows (as dicts)."""
    return {
        "companies": len(tickers),
        "delisted": sum(1 for t in tickers if t.get("isdelisted") == "Y"),
        "by_category": Counter(t.get("category") or "(blank)" for t in tickers),
        "by_exchange": Counter(t.get("exchange") or "(blank)" for t in tickers),
        "dual_class": sorted(
            (t.get("ticker"), t.get("category"))
            for t in tickers if "Class" in (t.get("category") or "")),
        "blank_check": sorted(
            (t.get("ticker"), t.get("isdelisted"))
            for t in tickers if str(t.get("siccode")) == "6770"),
    }


def summarise_actions(actions):
    return Counter(a.get("action") or "(blank)" for a in actions)


def _counter_lines(counter):
    return [f"    {n:>7,}  {k}" for k, n in counter.most_common()]


def _pair_lines(pairs):
    lines = [f"    {a}  ({b})" for a, b in pairs[:LIST_LIMIT]]
    if len(pairs) > LIST_LIMIT:
        lines.append(f"    ... and {len(pairs) - LIST_LIMIT} more")
    return lines or ["    (none)"]


def format_report(probes, ticker_summary, action_counts):
    lines = ["Tables:"]
    for table, result in probes.items():
        if isinstance(result, list):
            lines.append(f"  OK    {table}: {', '.join(result)}")
        else:
            lines.append(f"  NO    {table}: {result}")
    if ticker_summary:
        s = ticker_summary
        lines += ["", f"TICKERS (SF1 companies): {s['companies']:,}, delisted {s['delisted']:,}",
                  "  by category:", *_counter_lines(s["by_category"]),
                  "  by last known exchange:", *_counter_lines(s["by_exchange"]),
                  f"  dual-class ({len(s['dual_class'])}):", *_pair_lines(s["dual_class"]),
                  f"  SIC 6770 blank-check, (isdelisted) ({len(s['blank_check'])}):",
                  *_pair_lines(s["blank_check"])]
    if action_counts is not None:
        lines += ["", "ACTIONS event types:", *(_counter_lines(action_counts) or ["    (none)"])]
    return "\n".join(lines)


def discover(session, api_key):
    probes = {}
    for table in TABLES:
        try:
            probes[table] = columns_of(session, api_key, table)
        except SharadarError as exc:
            probes[table] = str(exc).split(": ", 1)[-1]

    # Each section fails on its own, so one oversized table cannot hide the rest.
    ticker_summary = action_counts = None
    if isinstance(probes["TICKERS"], list):
        try:
            ticker_summary = summarise_tickers(
                as_dicts(*fetch(session, api_key, "TICKERS", {"table": "SF1"})))
        except SharadarError as exc:
            probes["TICKERS"] = str(exc)
    if isinstance(probes["ACTIONS"], list):
        try:
            action_counts = summarise_actions(
                as_dicts(*fetch(session, api_key, "ACTIONS", {"qopts.columns": "action"})))
        except SharadarError as exc:
            probes["ACTIONS"] = str(exc)
    return probes, ticker_summary, action_counts


def main(argv=None, session=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--discover", action="store_true", required=True,
                        help="report what the key can see (structure only)")
    parser.parse_args(argv)

    api_key = os.environ.get("SHARADAR_API_KEY", "").strip()
    if not api_key:
        print("SHARADAR_API_KEY is not set.")
        return 2

    probes, ticker_summary, action_counts = discover(session or requests.Session(), api_key)
    print(format_report(probes, ticker_summary, action_counts))
    return 0 if ticker_summary is not None else 1


if __name__ == "__main__":
    sys.exit(main())
