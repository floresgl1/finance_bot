"""
Structural checks on the Sharadar data, before any return or ranking exists.

Implements "Structural checks" and "Rules fixed before the structural checks"
in docs/MICROCAP_VALUE_PREREG.md, on the Parquet files from microcap_data.py.
Every check looks at identifiers, categories, event names, dates, counts and
trading volume. None computes a return, a ranking or an EBIT/EV ratio.

  guard     the data is the paid universe, not the free tier
  coverage  price rows and ARY filings in every year 1998-2025
  check 1   dual-class companies to verify by hand against 10-K cover pages
  check 2   stocks demoted to OTC: prices afterwards, and ADV >= $100k?
  check 3   merged SPACs: SIC and revenue in their shell years
  check 4   delisting reasons: acquired / failed / unknown
  check 5   every ticker maps to exactly one permaticker

    python microcap_checks.py

Exit 1 if the guard, coverage or check 5 fails; checks 1-4 report.
"""

import sys
from bisect import bisect_left
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

PARQUET = Path("data/sharadar/parquet")

ELIGIBLE_CATEGORIES = {"Domestic Common Stock", "Domestic Common Stock Primary Class"}
SECONDARY_CATEGORY = "Domestic Common Stock Secondary Class"
MIN_DISTINCT_TICKERS = 100
FIRST_YEAR, LAST_YEAR = 1998, 2025
PRICES_THROUGH = date(2026, 7, 1)
FIRST_REBALANCE, LAST_REBALANCE = 1999, 2025

REASON_WINDOW_DAYS = 30
ACQUIRED = {"acquisitionby", "mergerto", "spacmerger"}
FAILED = {"bankruptcyliquidation", "regulatorydelisting", "voluntarydelisting"}

ADV_WINDOW, ADV_MIN_DAYS, ADV_FLOOR = 252, 126, 100_000
LIST_LIMIT = 15


# --- results --------------------------------------------------------------

class Result:
    def __init__(self, name, status, lines):
        self.name, self.status, self.lines = name, status, lines

    def __str__(self):
        return "\n".join([f"{self.status:<6} {self.name}", *[f"         {l}" for l in self.lines]])


# --- guard and coverage ---------------------------------------------------

def check_guard(fundamentals):
    n = fundamentals["ticker"].nunique()
    status = "PASS" if n >= MIN_DISTINCT_TICKERS else "FAIL"
    return Result("guard: paid universe", status,
                  [f"{n:,} distinct tickers in ARY fundamentals (need >= {MIN_DISTINCT_TICKERS})"])


def check_coverage(price_dates, filing_dates):
    """price_dates: the trading days with prices; filing_dates: ARY filing dates."""
    price_years = pd.Series([d.year for d in price_dates]).value_counts()
    filing_years = pd.Series([d.year for d in filing_dates if pd.notna(d)]).value_counts()
    years = range(FIRST_YEAR, LAST_YEAR + 1)
    no_prices = [y for y in years if price_years.get(y, 0) == 0]
    no_filings = [y for y in years if filing_years.get(y, 0) == 0]
    last_price = max(price_dates) if len(price_dates) else None
    ok = not no_prices and not no_filings and last_price is not None \
        and last_price >= PRICES_THROUGH
    lines = [
        f"trading days with prices per year {FIRST_YEAR}-{LAST_YEAR}: min "
        f"{min(price_years.get(y, 0) for y in years):,}; years with none: {no_prices or 'none'}",
        f"ARY filings per year {FIRST_YEAR}-{LAST_YEAR}: min "
        f"{min(filing_years.get(y, 0) for y in years):,}; years with none: {no_filings or 'none'}",
        f"last price date {last_price} (need >= {PRICES_THROUGH})",
    ]
    return Result("coverage: every test year has data", "PASS" if ok else "FAIL", lines)


# --- check 5: ticker join -------------------------------------------------

def check_ticker_join(tickers, fundamentals_counts, stock_counts):
    """tickers: the tickers table; *_counts: rows per ticker in the data (Series)."""
    rows = tickers[tickers["table"].isin(["SF1", "SEP"])]
    ids = rows.groupby("ticker")["permaticker"].nunique()
    shared = sorted(ids[ids > 1].index)
    known = set(ids.index)
    lines = [f"tickers mapped to more than one permaticker: {len(shared)}"
             + (f" ({', '.join(shared[:LIST_LIMIT])})" if shared else "")]
    for label, counts in [("fundamentals", fundamentals_counts), ("stocks", stock_counts)]:
        missing = counts[~counts.index.isin(known)]
        lines.append(f"{label}: {len(missing):,} of {len(counts):,} tickers "
                     f"({missing.sum():,} of {counts.sum():,} rows) have no permaticker")
    return Result("check 5: ticker join", "FAIL" if shared else "PASS", lines)


# --- check 4: delisting reasons -------------------------------------------

def classify_delistings(actions, eligible_tickers):
    """One row per `delisted` event of an eligible security, with its outcome."""
    delisted = actions[(actions["action"] == "delisted")
                       & actions["ticker"].isin(eligible_tickers)][["ticker", "date"]]
    delisted = delisted.reset_index(drop=True).rename_axis("event").reset_index()
    reasons = actions[actions["action"].isin(ACQUIRED | FAILED)][["ticker", "date", "action"]]
    pairs = delisted.merge(reasons, on="ticker", how="left", suffixes=("", "_reason"))
    gap = (pd.to_datetime(pairs["date_reason"]) - pd.to_datetime(pairs["date"])).dt.days.abs()
    pairs = pairs[gap <= REASON_WINDOW_DAYS]

    failed = set(pairs.loc[pairs["action"].isin(FAILED), "event"])
    acquired = set(pairs.loc[pairs["action"].isin(ACQUIRED), "event"])
    delisted["outcome"] = [
        "failed" if e in failed else "acquired" if e in acquired else "unknown"
        for e in delisted["event"]]
    return delisted


def check_delistings(actions, eligible_tickers):
    d = classify_delistings(actions, eligible_tickers)
    n = len(d)
    counts = d["outcome"].value_counts()
    lines = [f"delisted events for eligible common stocks: {n:,}"]
    for outcome, rule in [("acquired", "0%"), ("failed", "-30%"), ("unknown", "-30% primary, 0% sensitivity")]:
        k = int(counts.get(outcome, 0))
        lines.append(f"{outcome:<9} {k:>7,}  ({k / n:.0%})  -> {rule}" if n else f"{outcome}: 0")
    if n:
        by_decade = (d.assign(decade=pd.to_datetime(d["date"]).dt.year // 10 * 10)
                     .groupby("decade")["outcome"].apply(lambda s: (s == "unknown").mean()))
        lines.append("unknown share by decade: "
                     + ", ".join(f"{int(k)}s {v:.0%}" for k, v in by_decade.items()))
    return Result("check 4: delisting reasons", "INFO", lines)


# --- check 3: SPAC shell years --------------------------------------------

def check_spacs(actions, tickers, fundamentals):
    merges = (actions[actions["action"] == "spacmerger"]
              .groupby("ticker")["date"].min())
    sf1 = tickers[tickers["table"] == "SF1"].drop_duplicates("ticker").set_index("ticker")
    sic = sf1["siccode"].reindex(merges.index)
    shell_rows = fundamentals.merge(merges.rename("merged").reset_index(), on="ticker")
    shell_rows = shell_rows[shell_rows["date"] < shell_rows["merged"]]
    zero_rev = (shell_rows["revenue"].fillna(0) == 0)
    n = len(merges)
    lines = [
        f"tickers with a spacmerger event: {n:,}",
        f"  current SIC 6770: {int((sic == 6770).sum()):,}; other SIC: {int(sic.notna().sum() - (sic == 6770).sum()):,}; "
        f"no SF1 row: {int(sic.isna().sum()):,}",
        f"  ARY filings before the merger: {len(shell_rows):,}, "
        f"with zero or missing revenue: {int(zero_rev.sum()):,}"
        + (f" ({zero_rev.mean():.0%})" if len(shell_rows) else ""),
    ]
    return Result("check 3: SPAC shell years", "INFO", lines)


# --- check 2: demoted stocks ----------------------------------------------

def otc_periods(actions, eligible_tickers):
    """(ticker, start, end) for each spell on OTC; end is None if open-ended."""
    moves = actions[(actions["action"] == "exchangeto")
                    & actions["ticker"].isin(eligible_tickers)][["ticker", "date", "contraname"]]
    moves = moves.sort_values(["ticker", "date"])
    periods = []
    for ticker, g in moves.groupby("ticker"):
        rows = list(g.itertuples(index=False))
        for i, m in enumerate(rows):
            if m.contraname == "OTC":
                nxt = next((r.date for r in rows[i + 1:] if r.contraname != "OTC"), None)
                periods.append((ticker, m.date, nxt))
    return periods


def rebalance_dates(calendar):
    """First trading day on or after July 1, for each rebalance year.

    calendar: sorted list of trading days.
    """
    cal = pd.Series(calendar)
    out = []
    for year in range(FIRST_REBALANCE, LAST_REBALANCE + 1):
        on_or_after = cal[cal >= date(year, 7, 1)]
        if len(on_or_after):
            out.append(on_or_after.iloc[0])
    return out


def adv(prices, calendar, d):
    """Mean close*volume over the ADV_WINDOW trading days before d, or None.

    prices: one security's rows (date, close, volume); calendar: sorted list of
    trading days. Days without volume do not count; fewer than ADV_MIN_DAYS of
    them means liquidity is unknown.
    """
    idx = bisect_left(calendar, d)
    start = calendar[max(0, idx - ADV_WINDOW)]
    w = prices[(prices["date"] >= start) & (prices["date"] < d)].dropna(subset=["volume", "close"])
    if len(w) < ADV_MIN_DAYS:
        return None
    return float((w["close"] * w["volume"]).mean())


def check_demoted(periods, prices, calendar):
    """prices: rows (ticker, date, close, volume) for the demoted tickers only;
    calendar: sorted list of trading days."""
    rebalances = rebalance_dates(calendar)
    with_prices = 0
    liquid = []
    for ticker, start, end in periods:
        p = prices[prices["ticker"] == ticker]
        after = p[(p["date"] >= start) & ((p["date"] < end) if end else True)]
        if len(after):
            with_prices += 1
        for d in rebalances:
            if d <= start or (end and d >= end) or not len(after) or d > after["date"].max():
                continue
            value = adv(p, calendar, d)
            if value is not None and value >= ADV_FLOOR:
                liquid.append((ticker, d.year))
    lines = [f"moves to OTC by eligible common stocks: {len(periods):,}",
             f"  with price rows while on OTC: {with_prices:,}",
             f"  (ticker, rebalance) pairs on OTC with ADV >= ${ADV_FLOOR:,}: {len(liquid):,}"
             + (f" e.g. {', '.join(f'{t} {y}' for t, y in liquid[:LIST_LIMIT])}" if liquid else "")]
    return Result("check 2: demoted stocks", "INFO", lines)


# --- check 1: dual-class companies ----------------------------------------

SPAC_UNIT = r"(?:\.U|U\d*)$"


def dual_class_pairs(tickers):
    """(SF1 issuer row, secondary ticker) pairs for companies in the universe.

    Sharadar files old SPAC units under Secondary Class and renames them after
    the company the SPAC merged into, and relatedtickers links a company to its
    SPAC. So pairs are matched on issuer name only, unit tickers are dropped,
    and financials (not in the universe) are skipped.
    """
    sep = tickers[tickers["table"] == "SEP"]
    sic = tickers["siccode"]
    sf1 = tickers[(tickers["table"] == "SF1")
                  & tickers["category"].isin(ELIGIBLE_CATEGORIES)
                  & (sic != 6770) & ~sic.between(6000, 6999)]
    secondary = sep[(sep["category"] == SECONDARY_CATEGORY)
                    & ~sep["ticker"].str.contains(SPAC_UNIT, regex=True)]
    pairs = sf1.merge(secondary[["name", "ticker"]], on="name", suffixes=("", "_secondary"))
    return pairs.drop_duplicates(["ticker", "ticker_secondary"])


def check_dual_class(tickers, fundamentals):
    pairs = dual_class_pairs(tickers)
    latest = (fundamentals.sort_values("date").groupby("ticker")
              .last()[["date", "sharesbas"]])
    listed = pairs[pairs["isdelisted"] == "N"].merge(latest, left_on="ticker",
                                                      right_index=True, how="left")
    lines = [f"dual-class companies: {pairs['ticker'].nunique():,} "
             f"({listed['ticker'].nunique():,} still listed)",
             "verify 3 by hand: does sharesbas equal ALL classes on the 10-K cover?"]
    for r in listed.sort_values("name").head(LIST_LIMIT).itertuples(index=False):
        shares = f"{r.sharesbas:,.0f}" if pd.notna(r.sharesbas) else "n/a"
        lines.append(f"  {r.ticker:<7} (+{r.ticker_secondary:<7}) {r.name[:34]:<34} "
                     f"sharesbas {shares} as filed {r.date}")
    return Result("check 1: dual-class share counts", "MANUAL", lines)


# --- run ------------------------------------------------------------------

def run(parquet_dir=PARQUET):
    parquet_dir = Path(parquet_dir)
    tickers = pd.read_parquet(parquet_dir / "tickers.parquet")
    actions = pd.read_parquet(parquet_dir / "actions.parquet")
    fundamentals = pd.read_parquet(parquet_dir / "fundamentals.parquet")
    stocks_path = parquet_dir / "stocks.parquet"
    # 45M price rows: count in pyarrow, never as 45M Python strings.
    stock_cols = pq.read_table(stocks_path, columns=["ticker", "date"])
    calendar = sorted(pc.unique(stock_cols.column("date")).to_pylist())
    vc = pc.value_counts(stock_cols.column("ticker"))
    stock_counts = pd.Series(vc.field("counts").to_numpy(),
                             index=vc.field("values").to_pylist())
    del stock_cols

    sep = tickers[tickers["table"] == "SEP"]
    eligible = set(sep.loc[sep["category"].isin(ELIGIBLE_CATEGORIES), "ticker"])
    periods = otc_periods(actions, eligible)
    demoted = sorted({t for t, _, _ in periods})
    demoted_prices = pq.read_table(stocks_path, columns=["ticker", "date", "close", "volume"],
                                   filters=[("ticker", "in", demoted)] if demoted else None
                                   ).to_pandas() if demoted else pd.DataFrame(
        columns=["ticker", "date", "close", "volume"])

    return [
        check_guard(fundamentals),
        check_coverage(calendar, fundamentals["date"]),
        check_ticker_join(tickers, fundamentals["ticker"].value_counts(), stock_counts),
        check_delistings(actions, eligible),
        check_spacs(actions, tickers, fundamentals),
        check_demoted(periods, demoted_prices, calendar),
        check_dual_class(tickers, fundamentals),
    ]


def main(argv=None):
    results = run()
    for r in results:
        print(r)
        print()
    failed = [r.name for r in results if r.status == "FAIL"]
    print("FAILED: " + ", ".join(failed) if failed else "No blocking failures.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
