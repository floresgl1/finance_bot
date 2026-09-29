"""
Discovery run against Sharadar's own API (api.sharadar.com), before any
results exist.

docs/MICROCAP_VALUE_PREREG.md lists checks on Sharadar's data that must pass
before any return or ranking is computed. This script asks what the key can
see and prints STRUCTURE only:

  - which tables respond, and their field names
  - tickers (companies with fundamentals): count by category and last known
    exchange, delisted count, dual-class and SIC 6770 (blank-check) tickers,
    and any ticker mapped to more than one permaticker (check 5)
  - actions: event types and how often each appears

It never prints prices, returns or rankings, and writes nothing to disk.

A wrong key is NOT rejected: the API answers HTTP 200 with limited data. The
tickers table is public, so its size proves nothing; coverage is measured by
counting distinct tickers in one page of fundamentals, with a warning when it
looks like the free tier (the 30 Dow stocks) instead of the full universe.

    python sharadar_sample.py --discover

Needs SHARADAR_API_KEY (a sharadar.com key) in the environment.
Exit 0 if the tickers table was read in full.
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

import requests

BASE_URL = "https://api.sharadar.com/v1.0/data"
TABLES = ["tickers", "actions", "fundamentals", "stocks", "daily"]
PAGE_SIZE = 10000
MAX_PAGES = 100
MIN_RATE_REMAINING = 50     # stop before the per-key request budget runs out
FREE_TIER_MAX_COMPANIES = 100
HISTORY_START = "1990-01-01"  # date-based queries default to the last year only
LIST_LIMIT = 25


class SharadarError(Exception):
    pass


class RateLimited(SharadarError):
    """Stop every further request: more calls can only make it worse."""


def _get_page(session, table, query):
    """One request: list of row dicts.

    Errors never include the request URL, which carries the key.
    """
    try:
        resp = session.get(f"{BASE_URL}/{table}", params=query, timeout=60)
    except requests.RequestException as exc:
        raise SharadarError(f"{table}: network error ({type(exc).__name__})") from None
    if resp.status_code == 429:
        raise RateLimited(f"{table}: HTTP 429 rate limited")
    if resp.status_code != 200:
        raise SharadarError(f"{table}: HTTP {resp.status_code} {_api_message(resp)}")
    remaining = resp.headers.get("X-RateLimit-Remaining")
    if remaining is not None and int(remaining) < MIN_RATE_REMAINING:
        raise RateLimited(f"{table}: only {remaining} requests left in the rate-limit window")
    return resp.json()["data"]


def _api_message(resp):
    try:
        body = resp.json()
    except ValueError:
        return ""
    if isinstance(body, dict):
        return str(body.get("message") or body.get("error") or "")[:200]
    return ""


def fields_of(session, api_key, table):
    """Field names, from a one-row request whose row is discarded."""
    rows = _get_page(session, table, {"api_key": api_key, "format": "json", "limit": 1})
    if not rows:
        raise SharadarError(f"{table}: no rows visible to this key")
    return list(rows[0])


def fetch(session, api_key, table, params=None, max_pages=None):
    """All rows (as dicts) for one query, paging with limit/skip."""
    max_pages = max_pages or MAX_PAGES
    query = {"api_key": api_key, "format": "json", "limit": PAGE_SIZE, **(params or {})}
    rows = []
    for page_no in range(max_pages):
        query["skip"] = page_no * query["limit"]
        page = _get_page(session, table, query)
        rows.extend(page)
        if len(page) < query["limit"]:
            return rows
    raise SharadarError(f"{table}: more than {max_pages} pages; narrow the query")


def fundamentals_coverage(session, api_key):
    """Distinct tickers in one page of as-reported fundamentals.

    The tickers table is public (every key sees all companies), so coverage has
    to be measured on a data table. One request, no date filter.
    """
    rows = _get_page(session, "fundamentals", {
        "api_key": api_key, "format": "json", "limit": PAGE_SIZE,
        "dimension": "ARY", "fields": "ticker"})
    return len({r.get("ticker") for r in rows})


def summarise_tickers(tickers):
    """Structural counts from tickers rows."""
    permatickers = defaultdict(set)
    for t in tickers:
        permatickers[t.get("ticker")].add(t.get("permaticker"))
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
        "shared_tickers": sorted(
            (tk, ", ".join(sorted(map(str, ids))))
            for tk, ids in permatickers.items() if len(ids) > 1),
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


def format_report(probes, ticker_summary, action_counts, stopped=None, coverage=None):
    lines = []
    if stopped:
        lines += [f"STOPPED EARLY: {stopped}", ""]
    if coverage is not None:
        lines.append(f"Data coverage: {coverage:,} distinct tickers in one page of "
                     "ARY fundamentals")
        if coverage < FREE_TIER_MAX_COMPANIES:
            lines.append(f"  WARNING: under {FREE_TIER_MAX_COMPANIES}. This key sees the "
                         "free tier, or was not recognised (a wrong key still returns "
                         "data with HTTP 200). Not the full universe.")
        lines.append("")
    lines.append("Tables:")
    for table in TABLES:
        result = probes.get(table, "not tried")
        if isinstance(result, list):
            lines.append(f"  OK    {table}: {', '.join(result)}")
        else:
            lines.append(f"  NO    {table}: {result}")
    if ticker_summary:
        s = ticker_summary
        lines += ["", f"tickers (companies with fundamentals): {s['companies']:,}, "
                      f"delisted {s['delisted']:,}",
                  "  by category:", *_counter_lines(s["by_category"]),
                  "  by last known exchange:", *_counter_lines(s["by_exchange"]),
                  f"  dual-class ({len(s['dual_class'])}):", *_pair_lines(s["dual_class"]),
                  f"  SIC 6770 blank-check, (isdelisted) ({len(s['blank_check'])}):",
                  *_pair_lines(s["blank_check"]),
                  f"  tickers with more than one permaticker ({len(s['shared_tickers'])}):",
                  *_pair_lines(s["shared_tickers"])]
    if action_counts is not None:
        lines += ["", f"actions event types (since {HISTORY_START}):",
                  *(_counter_lines(action_counts) or ["    (none)"])]
    return "\n".join(lines)


def discover(session, api_key):
    """Returns (probes, ticker summary, action counts, coverage, reason stopped or None)."""
    probes, ticker_summary, action_counts, coverage = {}, None, None, None
    try:
        for table in TABLES:
            try:
                probes[table] = fields_of(session, api_key, table)
            except RateLimited:
                raise
            except SharadarError as exc:
                probes[table] = str(exc).split(": ", 1)[-1]

        if isinstance(probes["fundamentals"], list):
            try:
                coverage = fundamentals_coverage(session, api_key)
            except RateLimited:
                raise
            except SharadarError as exc:
                probes["fundamentals"] = str(exc)

        # Each section fails on its own, so one oversized table cannot hide the rest.
        if isinstance(probes["tickers"], list):
            try:
                ticker_summary = summarise_tickers(
                    fetch(session, api_key, "tickers", {"table": "fundamentals"}))
            except RateLimited:
                raise
            except SharadarError as exc:
                probes["tickers"] = str(exc)
        if isinstance(probes["actions"], list):
            try:
                action_counts = summarise_actions(fetch(
                    session, api_key, "actions",
                    {"fields": "action", "from": HISTORY_START}))
            except RateLimited:
                raise
            except SharadarError as exc:
                probes["actions"] = str(exc)
    except RateLimited as exc:
        return probes, ticker_summary, action_counts, coverage, str(exc)
    return probes, ticker_summary, action_counts, coverage, None


def main(argv=None, session=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--discover", action="store_true", required=True,
                        help="report what the key can see (structure only)")
    parser.parse_args(argv)

    api_key = os.environ.get("SHARADAR_API_KEY", "").strip()
    if not api_key:
        print("SHARADAR_API_KEY is not set.")
        return 2

    probes, ticker_summary, action_counts, coverage, stopped = discover(
        session or requests.Session(), api_key)
    print(format_report(probes, ticker_summary, action_counts, stopped, coverage))
    return 0 if ticker_summary is not None and not stopped else 1


if __name__ == "__main__":
    sys.exit(main())
