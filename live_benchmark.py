"""
live_benchmark.py — Did the real account actually beat holding the basket?

Every return figure the project produces is simulated. backtest.py simulates the
strategy, edge_probe.py simulates it across regimes, and promote_model.py gates
on a simulation. All of them make the same assumptions about fills, and all of
them could be wrong in the same direction.

This asks the same question of money that actually moved: take the paper
account's own equity curve from Alpaca, and put an equal-weight buy-and-hold of
the same watchlist — and SPY — beside it, over the same dates and from the same
starting capital.

**DESIGN DECISION:**
The strategy arm is Alpaca's portfolio history, NOT a reconstruction from
signal_log.csv. The log records what the bot decided and what it managed to
write down; the account records what actually filled. A reconstruction would
silently absorb missed fills, partial fills, stop-loss fills that never passed
through log_exit(), and cash drag — the exact errors this is meant to detect.
pnl_report.py attributes P&L across the log; this measures the account.

**DESIGN DECISION:**
The hold arms start on the account curve's first day, at the account's own
starting equity, and pay the same slippage and commission backtest.py applies.
A benchmark on different dates or different capital is not a benchmark. The
constants are imported from backtest.py rather than redeclared so the live
comparison and the simulated one cannot drift apart.

**DESIGN DECISION:**
A cash deposit or withdrawal makes the comparison meaningless — equity would
jump for a reason the benchmark cannot mirror. Alpaca's portfolio history does
not flag transfers, so `detect_transfers()` looks for single-day equity moves
too large to be market action and the report refuses to state a verdict when it
finds one, rather than reporting a number it cannot stand behind.

**DESIGN DECISION:**
The price source is stated in the report rather than inferred. Nothing under
data/ is tracked in git, so a CI runner has no per-ticker CSVs and would
silently produce no basket arm — the one number this module exists for. `--fetch`
downloads the window from yfinance instead, and the report always names which
source it used.

**DESIGN DECISION:**
Alpaca reports equity but not how much of it was at risk, and that split is the
whole question: trailing the basket because you hold half as much is a different
problem from picking badly. `reconstruct_exposure()` rebuilds the daily invested
fraction from ENTRY/EXIT rows in signal_log.csv, and refuses to report a number
when the ledger does not reconcile — a missing EXIT row leaves shares on the
book forever and inflates exposure with no obvious symptom.

Usage:
    python live_benchmark.py                  # console report, 90 days
    python live_benchmark.py --days 180
    python live_benchmark.py --fetch          # download prices (CI has no CSVs)
    python live_benchmark.py --discord        # also post to Discord
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from backtest import COMMISSION, SLIPPAGE
from config import WATCHLIST
from features import DATA_DIR, MARKET_DATA_DIR
from signal_logger import SIGNAL_LOG_PATH

load_dotenv()

DEFAULT_LOOKBACK_DAYS = 90

# A single-day equity move beyond this is treated as a possible cash transfer
# rather than market action. An equal-weight basket of large caps does not move
# 25% in a session; a deposit does.
TRANSFER_MOVE_PCT = 25.0

# The account cannot be more than fully invested -- there is no margin here. A
# reconstructed exposure above this means the log is missing EXIT rows, which
# leaves shares on the books forever, so the series is discarded rather than
# reported. Some headroom above 1.0 is allowed for same-day marks and rounding.
MAX_PLAUSIBLE_EXPOSURE = 1.25


# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------
def get_api():
    """Build an Alpaca REST client. Imported lazily so --help works without
    credentials and so tests never need the dependency."""
    import alpaca_trade_api as tradeapi

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise EnvironmentError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set as environment variables."
        )
    return tradeapi.REST(
        api_key, secret_key, "https://paper-api.alpaca.markets", api_version="v2"
    )


def fetch_equity_curve(api, days: int = DEFAULT_LOOKBACK_DAYS) -> pd.Series:
    """Daily account equity, date-indexed and tz-naive.

    Rows with zero or missing equity are dropped: Alpaca pads the series with
    zeroes for sessions before the account existed, and a leading zero would
    make the total return infinite.
    """
    history = api.get_portfolio_history(period=f"{days}D", timeframe="1D")
    frame = history.df
    if frame is None or frame.empty or "equity" not in frame:
        return pd.Series(dtype=float)

    equity = pd.to_numeric(frame["equity"], errors="coerce")
    equity.index = pd.DatetimeIndex(equity.index).tz_localize(None).normalize()
    equity = equity[equity > 0].dropna()
    return equity[~equity.index.duplicated(keep="last")].sort_index()


# ---------------------------------------------------------------------------
# Benchmark arms
# ---------------------------------------------------------------------------
def load_closes(tickers: list[str], data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Close prices per ticker, date-indexed. Tickers with no CSV are skipped.

    Missing files are skipped rather than fatal: the benchmark over eleven of
    twelve names is still informative, and the report says how many it used.
    """
    frames = {}
    for ticker in tickers:
        path = os.path.join(data_dir, f"{ticker}.csv")
        if not os.path.exists(path):
            continue
        try:
            frame = pd.read_csv(path)
            frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", utc=True)
            frame = frame.dropna(subset=["Date"]).set_index("Date")
            frame.index = frame.index.tz_localize(None).normalize()
            close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
            if not close.empty:
                frames[ticker] = close[~close.index.duplicated(keep="last")]
        except Exception:
            continue
    return pd.DataFrame(frames).sort_index() if frames else pd.DataFrame()


def fetch_closes(tickers: list[str], start, end) -> pd.DataFrame:
    """Close prices per ticker straight from yfinance, for runners with no CSVs.

    `end` is extended by a day because yfinance treats it as exclusive, and the
    final session is the one the comparison ends on.
    """
    import yfinance as yf

    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)

    frames = {}
    for ticker in tickers:
        try:
            raw = yf.download(ticker, start=start, end=end,
                              auto_adjust=True, progress=False)
            if raw is None or raw.empty or "Close" not in raw:
                continue
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):       # yfinance multi-index
                close = close.iloc[:, 0]
            close = pd.to_numeric(close, errors="coerce").dropna()
            close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
            if not close.empty:
                frames[ticker] = close[~close.index.duplicated(keep="last")]
        except Exception:
            continue
    return pd.DataFrame(frames).sort_index() if frames else pd.DataFrame()


def hold_curve(closes: pd.DataFrame, dates: pd.DatetimeIndex,
               initial_capital: float) -> pd.Series:
    """Equity curve for buying the basket equal-weight on day one and holding.

    Capital is split evenly across whichever names have a price on the first
    day. Each entry pays slippage and one commission, matching
    backtest._simulate_buy_and_hold, so this arm is comparable both to the live
    account and to the simulated hold arm in backtest.py.
    """
    if closes.empty or len(dates) == 0:
        return pd.Series(dtype=float)

    prices = closes.reindex(dates).ffill()
    first = prices.iloc[0].dropna()
    tradable = [t for t in first.index if first[t] > 0]
    if not tradable:
        return pd.Series(dtype=float)

    per_name = initial_capital / len(tradable)
    shares, spent = {}, 0.0
    for ticker in tradable:
        entry_price = float(first[ticker]) * (1 + SLIPPAGE)
        qty = (per_name - COMMISSION) / entry_price
        if qty <= 0:
            continue
        shares[ticker] = qty
        spent += qty * entry_price + COMMISSION

    if not shares:
        return pd.Series(dtype=float)

    cash = initial_capital - spent
    held = prices[list(shares)].ffill()
    value = sum(held[t] * qty for t, qty in shares.items())
    return (value + cash).dropna()


# ---------------------------------------------------------------------------
# Exposure reconstruction
# ---------------------------------------------------------------------------
def load_position_events(log_path: str = SIGNAL_LOG_PATH) -> pd.DataFrame:
    """Share deltas per ticker per day: +qty on an ENTRY, -shares on an EXIT.

    Exits are dated from `exit_timestamp` where present and `date` otherwise.
    reconcile_stops.py backfills stop fills with the Alpaca `filled_at` time in
    exit_timestamp, and those can land on a different day from the row date.
    """
    if not os.path.exists(log_path):
        return pd.DataFrame()

    frame = pd.read_csv(log_path, dtype=str)
    if "row_type" not in frame.columns:
        return pd.DataFrame()

    events = []

    entries = frame[frame["row_type"].fillna("") == "ENTRY"].copy()
    if not entries.empty:
        entries["when"] = pd.to_datetime(entries["date"], errors="coerce")
        entries["delta"] = pd.to_numeric(entries.get("qty"), errors="coerce")
        events.append(entries[["when", "ticker", "delta"]])

    exits = frame[frame["row_type"].fillna("") == "EXIT"].copy()
    if not exits.empty:
        when = pd.to_datetime(exits.get("exit_timestamp"), errors="coerce",
                              format="mixed", utc=True)
        when = when.dt.tz_localize(None) if hasattr(when, "dt") else when
        fallback = pd.to_datetime(exits["date"], errors="coerce")
        exits["when"] = when.fillna(fallback)
        exits["delta"] = -pd.to_numeric(exits.get("shares"), errors="coerce")
        events.append(exits[["when", "ticker", "delta"]])

    if not events:
        return pd.DataFrame()

    out = pd.concat(events).dropna(subset=["when", "ticker", "delta"])
    out["when"] = pd.DatetimeIndex(out["when"]).normalize()
    return out.sort_values("when").reset_index(drop=True)


def reconstruct_exposure(equity: pd.Series, closes: pd.DataFrame,
                         log_path: str = SIGNAL_LOG_PATH) -> tuple[pd.Series, str]:
    """Daily invested fraction of the account, rebuilt from the signal log.

    Returns (series, reason). The series is empty whenever the reconstruction
    cannot be trusted, and `reason` says why — reporting a decomposition built
    on an incomplete ledger would be worse than reporting none, because the
    error is invisible in the output.
    """
    if equity.empty:
        return pd.Series(dtype=float), "no account equity"
    if closes.empty:
        return pd.Series(dtype=float), "no price history"

    events = load_position_events(log_path)
    if events.empty:
        return pd.Series(dtype=float), (
            f"no ENTRY/EXIT rows in {os.path.basename(log_path)}"
        )

    ledger = (events.pivot_table(index="when", columns="ticker", values="delta",
                                 aggfunc="sum")
              .reindex(equity.index.union(events["when"].unique()))
              .fillna(0.0)
              .cumsum()
              .reindex(equity.index)
              .ffill()
              .fillna(0.0))

    held = [t for t in ledger.columns if t in closes.columns]
    if not held:
        return pd.Series(dtype=float), "no traded ticker has price history"

    prices = closes[held].reindex(equity.index).ffill()
    invested = (ledger[held] * prices).sum(axis=1)
    exposure = (invested / equity).replace([np.inf, -np.inf], np.nan).dropna()

    if exposure.empty:
        return pd.Series(dtype=float), "no overlapping sessions"
    if (exposure < -0.01).any():
        return pd.Series(dtype=float), "negative share balance - log has extra EXIT rows"
    if exposure.max() > MAX_PLAUSIBLE_EXPOSURE:
        return pd.Series(dtype=float), (
            f"reconstructed exposure peaked at {exposure.max() * 100:.0f}% - "
            f"the log is missing EXIT rows"
        )
    return exposure, ""


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def summarise_curve(curve: pd.Series) -> dict:
    """Total return, max drawdown and endpoints for one equity curve.

    Drawdown uses the same peak-to-trough formula as backtest._summarise, so
    the live number and the simulated one mean the same thing.
    """
    if curve is None or curve.empty:
        return {"start_value": 0.0, "final_value": 0.0, "total_return": 0.0,
                "max_drawdown": 0.0, "n_days": 0,
                "start_date": None, "end_date": None}

    start_value = float(curve.iloc[0])
    final_value = float(curve.iloc[-1])
    total_return = ((final_value - start_value) / start_value * 100
                    if start_value else 0.0)
    max_drawdown = (float(((curve - curve.cummax()) / curve.cummax() * 100).min())
                    if len(curve) > 1 else 0.0)
    return {
        "start_value":  start_value,
        "final_value":  final_value,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "n_days":       int(len(curve)),
        "start_date":   curve.index[0].date().isoformat(),
        "end_date":     curve.index[-1].date().isoformat(),
    }


def detect_transfers(curve: pd.Series,
                     threshold_pct: float = TRANSFER_MOVE_PCT) -> list[dict]:
    """Days whose equity move is too large to be market action.

    A deposit, a withdrawal or a paper-account reset breaks the comparison, and
    Alpaca's portfolio history does not distinguish those from P&L. Flagging is
    all that is possible; the report then declines to give a verdict.
    """
    if curve is None or len(curve) < 2:
        return []

    pct = curve.pct_change() * 100
    flagged = pct[pct.abs() >= threshold_pct]
    return [
        {"date": idx.date().isoformat(),
         "change_pct": float(value),
         "from_value": float(curve.shift(1)[idx]),
         "to_value": float(curve[idx])}
        for idx, value in flagged.items()
        if pd.notna(value)
    ]


def compare(equity: pd.Series, closes: pd.DataFrame,
            spy_closes: pd.DataFrame | None = None,
            price_source: str = "local CSVs",
            exposure: pd.Series | None = None,
            exposure_note: str = "") -> dict:
    """Put the account beside an equal-weight hold and an SPY hold.

    The hold arms are built on the account curve's own dates and starting
    equity — anything else compares two different experiments.
    """
    strategy = summarise_curve(equity)
    result = {
        "strategy": strategy,
        "arms": {},
        "transfers": detect_transfers(equity),
        "n_basket_names": 0,
        "price_source": price_source,
        "avg_exposure": None,
        "exposure_note": exposure_note,
    }
    if exposure is not None and not exposure.empty:
        result["avg_exposure"] = float(exposure.mean() * 100)
    if equity.empty:
        return result

    dates = equity.index
    initial = float(equity.iloc[0])

    basket = hold_curve(closes, dates, initial)
    if not basket.empty:
        result["arms"]["equal-weight hold"] = summarise_curve(basket)
        result["n_basket_names"] = int(closes.reindex(dates).ffill().iloc[0].notna().sum())

    if spy_closes is not None and not spy_closes.empty:
        spy = hold_curve(spy_closes, dates, initial)
        if not spy.empty:
            result["arms"]["SPY hold"] = summarise_curve(spy)

    for name, stats in result["arms"].items():
        stats["delta_pp"] = strategy["total_return"] - stats["total_return"]

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _money(value: float) -> str:
    return f"${value:,.2f}"


def format_report(result: dict) -> str:
    """Console report. ASCII only — Windows piped stdout is cp1252."""
    strategy = result["strategy"]
    lines = ["", "=" * 78,
             "  LIVE PERFORMANCE vs BUY-AND-HOLD", "=" * 78]

    if strategy["n_days"] == 0:
        lines += ["  No account equity history available.",
                  "  Nothing to compare - the account may be new, or the",
                  "  lookback may predate its first funded session.", "=" * 78]
        return "\n".join(lines)

    lines += [
        f"  Window   {strategy['start_date']} to {strategy['end_date']} "
        f"({strategy['n_days']} sessions)",
        f"  Capital  {_money(strategy['start_value'])} -> "
        f"{_money(strategy['final_value'])}",
    ]
    if result["n_basket_names"]:
        lines.append(f"  Basket   {result['n_basket_names']} of "
                     f"{len(WATCHLIST)} watchlist names had price history "
                     f"({result.get('price_source', 'unknown source')})")
    else:
        lines.append(f"  Basket   no prices available "
                     f"({result.get('price_source', 'unknown source')}) - "
                     f"pass --fetch to download them")
    lines += ["", f"  {'arm':<22}{'return':>10}{'vs account':>13}{'max DD':>10}", "  " + "-" * 55,
              f"  {'ACCOUNT (live)':<22}{strategy['total_return']:>9.2f}%"
              f"{'--':>13}{strategy['max_drawdown']:>9.2f}%"]

    for name, stats in result["arms"].items():
        lines.append(f"  {name:<22}{stats['total_return']:>9.2f}%"
                     f"{stats['delta_pp']:>+12.2f}pp{stats['max_drawdown']:>9.2f}%")

    lines += ["  " + "-" * 55, ""]

    if result["transfers"]:
        lines.append("  [WARNING] Possible cash transfer or account reset detected:")
        for event in result["transfers"]:
            lines.append(f"    {event['date']}  {event['change_pct']:+.1f}%  "
                         f"{_money(event['from_value'])} -> {_money(event['to_value'])}")
        lines += ["", "  Equity moved for a reason the benchmark cannot mirror, so no",
                  "  verdict is given. Re-run over a window that excludes that date.",
                  "=" * 78]
        return "\n".join(lines)

    lines.append(_verdict(result))
    lines.append("=" * 78)
    return "\n".join(lines)


def _verdict(result: dict) -> str:
    """State plainly whether running the bot beat not running it."""
    basket = result["arms"].get("equal-weight hold")
    if not basket:
        return ("  No basket arm could be built, so the question this report\n"
                "  exists to answer is unanswered.")

    delta = basket["delta_pp"]
    strategy = result["strategy"]
    decomposition = _decompose(result, basket)
    if delta > 0:
        verdict = (f"  The account BEAT holding the basket by {delta:+.2f}pp "
                   f"over this window.")
    elif delta < 0:
        verdict = (f"  The account LOST to holding the basket by {delta:+.2f}pp "
                   f"over this window.")
    else:
        verdict = "  The account exactly matched holding the basket."

    risk = ""
    if strategy["max_drawdown"] > basket["max_drawdown"]:
        risk = (f"\n  It did carry a shallower drawdown "
                f"({strategy['max_drawdown']:.2f}% vs {basket['max_drawdown']:.2f}%), "
                f"which is\n  worth something even when the return is behind.")

    caveat = ("\n  One window is one sample. See "
              "docs/EDGE_INVESTIGATION_2026-09-08.md for\n  what this looked "
              "like across simulated regimes.")
    return verdict + risk + decomposition + caveat


def _decompose(result: dict, basket: dict) -> str:
    """Split the gap to the basket into exposure and selection.

    Crude by construction -- it compares against holding the basket at the
    account's *average* exposure, which ignores when that exposure was carried.
    It is still the difference between "we hold less" and "we pick badly", and
    those call for opposite fixes.
    """
    avg = result.get("avg_exposure")
    if avg is None:
        note = result.get("exposure_note") or "not reconstructed"
        return f"\n  Exposure could not be measured ({note}), so the gap is\n  not split into exposure vs selection."

    strategy_return = result["strategy"]["total_return"]
    expected = basket["total_return"] * avg / 100.0
    selection = strategy_return - expected
    return (
        f"\n  Average exposure {avg:.0f}%. Holding the basket at that exposure\n"
        f"  would have returned about {expected:+.2f}%; the account made "
        f"{strategy_return:+.2f}%,\n"
        f"  so selection and timing contributed {selection:+.2f}pp."
    )


def format_discord(result: dict) -> str:
    strategy = result["strategy"]
    if strategy["n_days"] == 0:
        return "📊 **LIVE vs BUY-AND-HOLD** — no account equity history available."

    basket = result["arms"].get("equal-weight hold")
    if result["transfers"]:
        icon, headline = "⚠️", "possible cash transfer detected — no verdict"
    elif basket and basket["delta_pp"] > 0:
        icon, headline = "✅", f"beating the basket by {basket['delta_pp']:+.2f}pp"
    elif basket:
        icon, headline = "🛑", f"behind the basket by {basket['delta_pp']:+.2f}pp"
    else:
        icon, headline = "📊", "no basket arm available"

    lines = [
        f"{icon} **LIVE vs BUY-AND-HOLD** — {headline}",
        f"_{strategy['start_date']} to {strategy['end_date']}_",
        "",
        f"Account — **{strategy['total_return']:+.2f}%** "
        f"({_money(strategy['start_value'])} → {_money(strategy['final_value'])}), "
        f"drawdown {strategy['max_drawdown']:.2f}%",
    ]
    for name, stats in result["arms"].items():
        lines.append(f"{name} — {stats['total_return']:+.2f}% "
                     f"(account {stats['delta_pp']:+.2f}pp), "
                     f"drawdown {stats['max_drawdown']:.2f}%")

    if result["transfers"]:
        dates = ", ".join(e["date"] for e in result["transfers"])
        lines += ["", f"⚠️ Equity jumped on {dates} — likely a deposit or reset, "
                      f"so the comparison is not meaningful over this window."]
    return "\n".join(lines)


def send_discord(message: str) -> None:
    """POST a message to the Discord webhook. Silently skips if URL not set."""
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        print("  [DISCORD] DISCORD_WEBHOOK_URL not set - skipping notification.")
        return
    try:
        resp = requests.post(url, json={"content": message}, timeout=10)
        if resp.status_code not in (200, 204):
            print(f"  [DISCORD] HTTP {resp.status_code}: {resp.text.strip()}")
    except Exception as exc:
        print(f"  [DISCORD] Notification failed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the live paper account against buy-and-hold.",
    )
    parser.add_argument(
        "--days", type=int, default=DEFAULT_LOOKBACK_DAYS, metavar="N",
        help=f"Lookback in calendar days (default: {DEFAULT_LOOKBACK_DAYS}).",
    )
    parser.add_argument(
        "--data-dir", default=DATA_DIR, metavar="PATH",
        help="Directory of per-ticker price CSVs (default: data/).",
    )
    parser.add_argument(
        "--fetch", action="store_true",
        help="Download benchmark prices from yfinance instead of reading "
             "data/*.csv. Required on a CI runner, which has no price CSVs.",
    )
    parser.add_argument(
        "--log", default=SIGNAL_LOG_PATH, metavar="PATH",
        help="Signal log used to reconstruct daily exposure "
             f"(default: {SIGNAL_LOG_PATH}).",
    )
    parser.add_argument(
        "--discord", action="store_true",
        help="Post a compact summary to the Discord webhook.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        api = get_api()
    except Exception as exc:
        print(f"[FATAL] Could not build an Alpaca client: {exc}")
        return 1

    try:
        equity = fetch_equity_curve(api, days=args.days)
    except Exception as exc:
        print(f"[FATAL] Could not fetch portfolio history: {exc}")
        return 1

    if args.fetch and not equity.empty:
        source = "yfinance"
        closes = fetch_closes(WATCHLIST, equity.index[0], equity.index[-1])
        spy = fetch_closes(["SPY"], equity.index[0], equity.index[-1])
    else:
        source = f"local CSVs in {args.data_dir}"
        closes = load_closes(WATCHLIST, data_dir=args.data_dir)
        spy_path = os.path.join(MARKET_DATA_DIR, "SPY.csv")
        spy = (load_closes(["SPY"], data_dir=MARKET_DATA_DIR)
               if os.path.exists(spy_path) else pd.DataFrame())

    exposure, note = reconstruct_exposure(equity, closes, log_path=args.log)
    result = compare(equity, closes, spy, price_source=source,
                     exposure=exposure, exposure_note=note)
    print(format_report(result))

    if args.discord:
        send_discord(format_discord(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
