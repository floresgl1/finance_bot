"""
Backtest for the pre-registered micro-cap value test.

Implements docs/MICROCAP_VALUE_PREREG.md on the Parquet files from
microcap_data.py. Every rule is the registered one; section names in the
comments point to where each is written down.

Protocol: this code is tested on synthetic data only, frozen in one commit,
then run ONCE on the real data. main() refuses to run from a working tree
with uncommitted changes, stamps the commit into the result, and refuses to
overwrite an existing result.

    python microcap_backtest.py --dry-run   # real data, universe sizes only
    python microcap_backtest.py             # THE run: once, from the frozen commit

The dry run builds each year's universe on the real data and prints only its
size and timing, so a crash or a slow step shows up before the real run
without anyone seeing a return. Writes docs/MICROCAP_BACKTEST_RESULT.md and data/microcap_backtest.json.
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PARQUET = Path("data/sharadar/parquet")
RESULT_MD = Path("docs/MICROCAP_BACKTEST_RESULT.md")
RESULT_JSON = Path("data/microcap_backtest.json")

# --- registered parameters ------------------------------------------------
ELIGIBLE_CATEGORIES = {"Domestic Common Stock", "Domestic Common Stock Primary Class"}
FIRST_REBALANCE, LAST_REBALANCE = 1999, 2025
MCAP_MIN, MCAP_MAX = 50e6, 300e6
ADV_FLOOR = 100_000
WINDOW, MIN_WINDOW_DAYS = 252, 126
STALE_MONTHS = 18
MAX_NET_DEBT_TO_EBIT = 3.0
TOP_N = 30
QUINTILES = 5
SPREAD_FLOOR = 0.005
FALLBACK_MEDIAN_MULTIPLE = 2.0
REASON_WINDOW_DAYS = 30
ACQUIRED = {"acquisitionby", "mergerto", "spacmerger"}
FAILED = {"bankruptcyliquidation", "regulatorydelisting", "voluntarydelisting"}
FAILED_RETURN = -0.30
MIN_WINS = 22            # 80% of 27 years, rounded up
MIN_MEAN_EXCESS = 0.03
CASH = -1                # position key for uninvested cash


def _days(values):
    """Dates (date objects, datetime64 or strings) -> int days since epoch."""
    return pd.to_datetime(pd.Series(values)).values.astype("datetime64[D]").astype(np.int64)


def _day(d):
    return int(np.datetime64(d, "D").astype(np.int64))


def _date(day):
    return np.datetime64(int(day), "D").astype(object)


# --- prices ---------------------------------------------------------------

class PriceStore:
    """Daily prices per security on the trading calendar.

    Each security's life (first to last trading day) is laid out densely on the
    calendar, so "the 252 trading days before D" is a slice, and window sums
    are differences of cumulative sums. Securities are addressed by integer id.
    """

    def __init__(self, tickers, days, high, low, close, volume, closeadj, closeunadj):
        codes, self.names = pd.factorize(pd.Series(tickers), sort=True)
        days = np.asarray(days, dtype=np.int64)
        order = np.lexsort((days, codes))
        codes, days = codes[order], days[order]
        self.calendar = np.unique(days)
        pos = np.searchsorted(self.calendar, days)

        g = len(self.names)
        counts = np.bincount(codes, minlength=g)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self.first = pos[starts]
        self.last = pos[starts + counts - 1]
        self.length = self.last - self.first + 1
        self.offset = np.concatenate([[0], np.cumsum(self.length)[:-1]])
        self.id = {name: i for i, name in enumerate(self.names)}

        n = int(self.length.sum())
        dense_idx = self.offset[codes] + pos - self.first[codes]

        def dense(values):
            out = np.full(n, np.nan)
            out[dense_idx] = np.asarray(values, dtype=float)[order]
            return out

        self.close, self.closeadj, self.closeunadj = dense(close), dense(closeadj), dense(closeunadj)
        h, l, vol = dense(high), dense(low), dense(volume)

        seg_end = np.zeros(n, dtype=bool)
        seg_end[self.offset + self.length - 1] = True
        self._last_close = self._last_valid(self.close)
        self._last_adj = self._last_valid(self.closeadj)

        # ADV: a day counts only with volume > 0 (clarification to the rules).
        vol_ok = np.nan_to_num(vol) > 0
        dv_ok = vol_ok & np.isfinite(self.close)
        dv = np.where(dv_ok, self.close * np.nan_to_num(vol), 0.0)

        # Abdi-Ranaldo two-day estimate, split-adjusted high/low/close:
        # s^2 = 4 (c_t - eta_t)(c_t - eta_t+1), eta = mean of log high and log low.
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.log(self.close)
            eta = (np.log(h) + np.log(l)) / 2
        eta_next = np.append(eta[1:], np.nan)
        eta_next[seg_end] = np.nan
        s2 = 4 * (c - eta) * (c - eta_next)
        s_ok = vol_ok & np.isfinite(s2)
        s = np.where(s_ok, np.sqrt(np.clip(np.nan_to_num(s2), 0, None)), 0.0)

        self._cs = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in
                    {"dv": dv, "dv_n": dv_ok.astype(float), "s": s, "s_n": s_ok.astype(float)}.items()}

    def _last_valid(self, values):
        """Index of the last finite value at or before each position, within its segment."""
        idx = np.where(np.isfinite(values), np.arange(len(values)), -1)
        return np.maximum.accumulate(idx)

    def ids(self, tickers):
        return np.array([self.id.get(t, -1) for t in tickers], dtype=np.int64)

    def _pos(self, day, side="left"):
        return int(np.searchsorted(self.calendar, day, side=side))

    def _window(self, ids, day, key):
        """(sum, count) over the WINDOW trading days strictly before day."""
        p = self._pos(day)
        off, ln = self.offset[ids], self.length[ids]
        i = off + (p - self.first[ids])
        hi = np.clip(i, off, off + ln)
        lo = np.clip(i - WINDOW, off, hi)
        cs, cn = self._cs[key], self._cs[key + "_n"]
        return cs[hi] - cs[lo], cn[hi] - cn[lo]

    def adv(self, ids, day):
        total, n = self._window(ids, day, "dv")
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n >= MIN_WINDOW_DAYS, total / n, np.nan)

    def spread(self, ids, day):
        """Estimated full spread, or NaN if fewer than MIN_WINDOW_DAYS valid days."""
        total, n = self._window(ids, day, "s")
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n >= MIN_WINDOW_DAYS, total / n, np.nan)

    def close_on(self, ids, day):
        """Split-adjusted close on exactly this trading day, else NaN."""
        p = self._pos(day)
        if p >= len(self.calendar) or self.calendar[p] != day:
            return np.full(len(ids), np.nan)
        inside = (p >= self.first[ids]) & (p <= self.last[ids])
        idx = self.offset[ids] + np.clip(p - self.first[ids], 0, self.length[ids] - 1)
        return np.where(inside, self.close[idx], np.nan)

    def unadj_ratio_on_or_before(self, ids, day):
        """closeunadj / close on the last trading day at or before day, else NaN."""
        p = self._pos(day, side="right") - 1
        rel = p - self.first[ids]
        ok = rel >= 0
        idx = self.offset[ids] + np.clip(rel, 0, self.length[ids] - 1)
        j = self._last_close[idx]
        ok &= j >= self.offset[ids]
        j = np.where(ok, j, 0)
        return np.where(ok, self.closeunadj[j] / self.close[j], np.nan)

    def last_day(self, ids):
        return self.calendar[self.last[ids]]

    def adj_panel(self, ids, d0, d1):
        """closeadj for trading days d0..d1 (rows) x ids (cols), carried over gaps."""
        p0, p1 = self._pos(d0), self._pos(d1)
        ps = np.arange(p0, p1 + 1)[:, None]
        rel = ps - self.first[ids][None, :]
        inside = (rel >= 0) & (ps <= self.last[ids][None, :])
        idx = self.offset[ids][None, :] + np.clip(rel, 0, self.length[ids][None, :] - 1)
        j = self._last_adj[idx]
        ok = inside & (j >= self.offset[ids][None, :])
        return np.where(ok, self.closeadj[np.where(ok, j, 0)], np.nan), self.calendar[p0:p1 + 1]


# --- reference data -------------------------------------------------------

@dataclass
class Reference:
    """Point-in-time rules that come from tickers and actions."""
    securities: pd.DataFrame        # ticker, permaticker: eligible by category and SIC
    spac_merged: dict               # ticker -> day of first spacmerger
    exchange_moves: dict            # ticker -> (sorted days, destinations)
    reasons: dict                   # ticker -> (sorted days, actions)

    def is_shell(self, ticker, day):
        merged = self.spac_merged.get(ticker)
        return merged is not None and day < merged

    def on_otc(self, ticker, day):
        moves = self.exchange_moves.get(ticker)
        if moves is None:
            return False
        days, dest = moves
        k = int(np.searchsorted(days, day, side="left")) - 1
        return k >= 0 and dest[k] == "OTC"

    def delisting_return(self, ticker, day, unknown_return):
        found = self.reasons.get(ticker)
        if found is not None:
            days, actions = found
            near = actions[np.abs(days - day) <= REASON_WINDOW_DAYS]
            if any(a in FAILED for a in near):
                return FAILED_RETURN
            if any(a in ACQUIRED for a in near):
                return 0.0
        return unknown_return


def build_reference(tickers, actions):
    sf1 = tickers[tickers["table"] == "SF1"].drop_duplicates("ticker").set_index("ticker")
    sep = tickers[(tickers["table"] == "SEP") & tickers["category"].isin(ELIGIBLE_CATEGORIES)]
    sic = sep["ticker"].map(sf1["siccode"])
    # Financials (SIC 6000-6999) and blank-check shells (6770) are out; a
    # missing SIC cannot be shown to be non-financial, so it is out too.
    keep = sic.notna() & ~sic.between(6000, 6999)
    securities = sep.loc[keep, ["ticker", "permaticker"]].drop_duplicates("ticker")

    act = actions.assign(day=_days(actions["date"]))
    spac = act[act["action"] == "spacmerger"].groupby("ticker")["day"].min().to_dict()
    moves = {}
    for t, g in act[act["action"] == "exchangeto"].sort_values("day").groupby("ticker"):
        moves[t] = (g["day"].to_numpy(), g["contraname"].to_numpy())
    reasons = {}
    for t, g in act[act["action"].isin(ACQUIRED | FAILED)].groupby("ticker"):
        reasons[t] = (g["day"].to_numpy(), g["action"].to_numpy())
    return Reference(securities, spac, moves, reasons)


# --- universe -------------------------------------------------------------

def latest_filings(fundamentals, day):
    """Each ticker's latest ARY row filed strictly before day (whole row)."""
    f = fundamentals[fundamentals["day"] < day]
    return f.sort_values(["ticker", "day"]).drop_duplicates("ticker", keep="last")


def build_universe(day, fundamentals, ref, store):
    """The eligible universe on rebalance day, with EBIT/EV and half spreads."""
    f = latest_filings(fundamentals, day)
    f = f.merge(ref.securities, on="ticker")
    stale_cutoff = _day(pd.Timestamp(_date(day)) - pd.DateOffset(months=STALE_MONTHS))
    f = f[f["reportday"] >= stale_cutoff]
    f = f.dropna(subset=["ebit", "debt", "cashneq", "sharesbas"])
    f = f[f["sharesbas"] > 0]
    listed = np.array([not ref.is_shell(t, day) and not ref.on_otc(t, day)
                       for t in f["ticker"]], dtype=bool)
    f = f[listed]
    f = f.assign(sid=store.ids(f["ticker"]))
    f = f[f["sid"] >= 0]
    if f.empty:
        return f.assign(mcap=[], ev=[], ebit_ev=[], adv=[], half_spread=[])

    ids = f["sid"].to_numpy()
    close = store.close_on(ids, day)
    ratio = np.array([store.unadj_ratio_on_or_before(np.array([i]), fd)[0]
                      for i, fd in zip(ids, f["day"])])
    f = f.assign(close=close, mcap=close * f["sharesbas"].to_numpy() * ratio,
                 adv=store.adv(ids, day), est_spread=store.spread(ids, day))
    f = f[np.isfinite(f["close"]) & np.isfinite(f["mcap"])]
    f = f[f["adv"] >= ADV_FLOOR]
    f = f[(f["mcap"] >= MCAP_MIN) & (f["mcap"] <= MCAP_MAX)]
    f = f.assign(ev=f["mcap"] + f["debt"] - f["cashneq"])
    f = f[f["ev"] > 0]
    net_debt = f["debt"] - f["cashneq"]
    passes = (net_debt <= 0) | ((f["ebit"] > 0) & (net_debt <= MAX_NET_DEBT_TO_EBIT * f["ebit"]))
    f = f[passes].assign(ebit_ev=lambda x: x["ebit"] / x["ev"])

    floored = np.maximum(f["est_spread"], SPREAD_FLOOR)
    fallback = FALLBACK_MEDIAN_MULTIPLE * np.nanmedian(floored) if floored.notna().any() else 0.02
    spread = np.where(f["est_spread"].notna(), floored, max(fallback, SPREAD_FLOOR))
    return f.assign(half_spread=spread / 2, fallback_spread=max(fallback, SPREAD_FLOOR)) \
            .sort_values(["ebit_ev", "permaticker"], ascending=[False, True]) \
            .reset_index(drop=True)


def portfolios(universe):
    """Security ids per portfolio: strategy (top 30), control (all), Q1..Q5."""
    ids = universe["sid"].to_numpy()        # already sorted: cheapest first
    out = {"strategy": ids[:TOP_N], "control": ids}
    for q, part in enumerate(np.array_split(ids, QUINTILES), start=1):
        out[f"Q{q}"] = part
    return out


# --- one holding year -----------------------------------------------------

def half_spreads(store, ids, day, fallback_spread):
    s = store.spread(ids, day)
    s = np.where(np.isfinite(s), np.maximum(s, SPREAD_FLOOR), fallback_spread)
    return s / 2


def simulate_year(target, prev, d0, d1, store, ref, fallback_spread, unknown_return):
    """Equal weight in target from the close on d0 to the close on d1.

    prev: {id: weight} drifted from last year (CASH for uninvested cash).
    Returns (year return net of all costs, {id: end weight}).
    """
    target = np.asarray(target, dtype=np.int64)
    n = len(target)
    if n == 0:
        return 0.0, {CASH: 1.0}
    w = dict.fromkeys(target.tolist(), 1.0 / n)
    traded = np.array([i for i in set(w) | set(prev) if i != CASH], dtype=np.int64)
    delta = np.array([abs(w.get(i, 0.0) - prev.get(i, 0.0)) for i in traded])
    cost = float((half_spreads(store, traded, d0, fallback_spread) * delta).sum()) if len(traded) else 0.0

    panel, days = store.adj_panel(target, d0, d1)
    last = store.last_day(target)
    delist_k = np.where(last < d1, np.searchsorted(days, last), -1)
    names = store.names[target]

    v = np.full(n, (1.0 - cost) / n)
    alive = np.ones(n, dtype=bool)
    cash = 0.0
    for k in range(len(days)):
        if k:
            with np.errstate(invalid="ignore", divide="ignore"):
                r = panel[k] / panel[k - 1]
            v[alive] *= np.where(np.isfinite(r[alive]), r[alive], 1.0)
        gone = alive & (delist_k == k)
        if not gone.any():
            continue
        proceeds = sum(v[i] * (1 + ref.delisting_return(names[i], days[k], unknown_return))
                       for i in np.flatnonzero(gone))
        v[gone], alive[gone] = 0.0, False
        if alive.any():
            buys = proceeds * v[alive] / v[alive].sum()
            hs = half_spreads(store, target[alive], days[k], fallback_spread)
            v[alive] += buys * (1 - hs)
        else:
            cash += proceeds

    end_value = float(v.sum() + cash)
    end = {int(target[i]): float(v[i] / end_value) for i in np.flatnonzero(alive)}
    if cash:
        end[CASH] = cash / end_value
    return end_value - 1.0, end


# --- the whole test -------------------------------------------------------

def rebalance_days(calendar):
    out = []
    for year in range(FIRST_REBALANCE, LAST_REBALANCE + 2):   # +1: the last year's end
        k = int(np.searchsorted(calendar, _day(date(year, 7, 1))))
        if k < len(calendar):
            out.append(int(calendar[k]))
    return out


def run_backtest(fundamentals, ref, store, unknown_returns=(FAILED_RETURN, 0.0)):
    """Yearly net returns per portfolio, for each unknown-delisting return."""
    days = rebalance_days(store.calendar)
    runs = {u: {"years": []} for u in unknown_returns}
    prev = {u: {} for u in unknown_returns}
    for d0, d1 in zip(days[:-1], days[1:]):
        universe = build_universe(d0, fundamentals, ref, store)
        books = portfolios(universe)
        fallback = float(universe["fallback_spread"].iloc[0]) if len(universe) else 0.02
        for u in unknown_returns:
            row = {"rebalance": str(_date(d0)), "universe": len(universe)}
            for name, ids in books.items():
                ret, end = simulate_year(ids, prev[u].get(name, {}), d0, d1, store, ref,
                                         fallback, u)
                row[name] = ret
                prev[u][name] = end
            runs[u]["years"].append(row)
    return runs


def evaluate(years):
    """The three registered pass criteria on a list of yearly rows."""
    strat = np.array([y["strategy"] for y in years])
    ctrl = np.array([y["control"] for y in years])
    q = [np.mean([y[f"Q{i}"] for y in years]) for i in range(1, QUINTILES + 1)]
    wins = int((strat > ctrl).sum())
    mean_excess = float((strat - ctrl).mean())
    staircase = all(q[i] > q[i + 1] for i in range(QUINTILES - 1))
    return {"years": len(years), "wins": wins, "wins_needed": MIN_WINS,
            "mean_excess": mean_excess, "quintile_means": q, "staircase": staircase,
            "pass": wins >= MIN_WINS and staircase and mean_excess >= MIN_MEAN_EXCESS}


def verdict(primary, sensitivity):
    if primary["pass"]:
        return "PASS"
    return "INCONCLUSIVE" if sensitivity["pass"] else "FAIL"


# --- loading and running --------------------------------------------------

def load(parquet_dir=PARQUET):
    parquet_dir = Path(parquet_dir)
    tickers = pd.read_parquet(parquet_dir / "tickers.parquet")
    actions = pd.read_parquet(parquet_dir / "actions.parquet")
    ref = build_reference(tickers, actions)
    wanted = sorted(ref.securities["ticker"])

    f = pd.read_parquet(parquet_dir / "fundamentals.parquet")
    f = f[f["ticker"].isin(set(wanted))]
    f = f.assign(day=_days(f["date"]), reportday=_days(f["reportperiod"]))

    t = pq.read_table(parquet_dir / "stocks.parquet",
                      columns=["ticker", "date", "high", "low", "close", "volume",
                               "closeadj", "closeunadj"],
                      filters=[("ticker", "in", wanted)])
    cols = {c: t.column(c).to_numpy() for c in
            ["high", "low", "close", "volume", "closeadj", "closeunadj"]}
    store = PriceStore(t.column("ticker").to_pandas().to_numpy(),
                       _days(t.column("date").to_pandas()), **cols)
    return f, ref, store


def git_state():
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return head, bool(dirty)


def _pct(x):
    return f"{x:+.1%}"


def format_result(runs, commit, ran_utc):
    primary, sensitivity = runs[FAILED_RETURN], runs[0.0]
    pe, se = evaluate(primary["years"]), evaluate(sensitivity["years"])
    v = verdict(pe, se)
    lines = [
        "# Micro-cap value — backtest result",
        "",
        f"Run once, {ran_utc}, from frozen commit `{commit}`, on the files in",
        "`docs/sharadar_manifest.json`, under the rules in `MICROCAP_VALUE_PREREG.md`.",
        "",
        f"## Verdict: **{v}**",
        "",
        "| Criterion | Bar | Primary (unknown delistings −30%) | Sensitivity (0%) |",
        "|---|---|---|---|",
        f"| Years strategy beat control | ≥ {MIN_WINS} of {pe['years']} | {pe['wins']} | {se['wins']} |",
        f"| Quintile staircase, no swaps | strict | {'yes' if pe['staircase'] else 'no'} "
        f"| {'yes' if se['staircase'] else 'no'} |",
        f"| Mean excess return a year | ≥ {MIN_MEAN_EXCESS:.0%} | {_pct(pe['mean_excess'])} "
        f"| {_pct(se['mean_excess'])} |",
        f"| **Pass** | all three | **{'yes' if pe['pass'] else 'no'}** | {'yes' if se['pass'] else 'no'} |",
        "",
        "Quintile mean yearly returns, cheapest (Q1) to most expensive (Q5), primary: "
        + ", ".join(_pct(x) for x in pe["quintile_means"]),
        "",
        "## Year by year (primary run, net of costs)",
        "",
        "| Rebalance | Universe | Strategy | Control | Excess | Q1 | Q2 | Q3 | Q4 | Q5 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for y in primary["years"]:
        lines.append(
            f"| {y['rebalance']} | {y['universe']:,} | {_pct(y['strategy'])} | {_pct(y['control'])} "
            f"| {_pct(y['strategy'] - y['control'])} | "
            + " | ".join(_pct(y[f'Q{i}']) for i in range(1, QUINTILES + 1)) + " |")
    return "\n".join(lines) + "\n", v, pe, se


def dry_run():
    """Load the real data and build each year's universe: sizes only, no returns."""
    import time
    t0 = time.time()
    fundamentals, ref, store = load()
    print(f"  loaded {len(store.names):,} securities, {len(store.calendar):,} trading days "
          f"in {time.time() - t0:.0f}s", flush=True)
    for d0 in rebalance_days(store.calendar)[:-1]:
        t1 = time.time()
        u = build_universe(d0, fundamentals, ref, store)
        print(f"  {mb_date(d0)}  universe {len(u):>5,}  ({time.time() - t1:.1f}s)", flush=True)
    return 0


def mb_date(day):
    return str(_date(day))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if "--dry-run" in argv:
        return dry_run()
    commit, dirty = git_state()
    if dirty:
        print("Refusing to run: uncommitted changes. The run must come from the frozen commit.")
        return 2
    if RESULT_MD.exists():
        print(f"Refusing to run: {RESULT_MD} exists. This test runs once; a rerun is a second draw.")
        return 2
    print(f"Loading data (commit {commit[:12]}) ...", flush=True)
    fundamentals, ref, store = load()
    print(f"  {len(store.names):,} securities, {len(store.calendar):,} trading days", flush=True)
    runs = run_backtest(fundamentals, ref, store)
    ran = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    text, v, pe, se = format_result(runs, commit, ran)
    RESULT_MD.write_text(text, encoding="utf-8")
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps({"commit": commit, "ran_utc": ran, "verdict": v,
                                       "primary": pe, "sensitivity": se,
                                       "runs": {str(k): r for k, r in runs.items()}},
                                      indent=2, default=float), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
