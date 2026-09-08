"""
Simulates trading using historical model signals to evaluate strategy performance.

Supports walk-forward split evaluation: the same simulation can be run on the
training, validation, test, or full date ranges so performance on each period
can be compared side-by-side.
"""

import os

import joblib
import numpy as np
import pandas as pd

from config import (
    WATCHLIST,
    FEATURE_COLUMNS,
    CONFIDENCE_THRESHOLD,
    MODEL_DIR,
    MODEL_FILENAME,
    MAX_POSITION_PCT,
    MAX_TOTAL_EXPOSURE,
)
from features import load_and_process

# --- Backtest settings ---
#
# MAX_POSITION_PCT and MAX_TOTAL_EXPOSURE come from config.py so this simulates
# the book the live bot actually builds. They were previously redeclared here as
# 0.20 / 0.80 -- a concentrated four-position strategy, where the live bot runs
# up to twelve names at 8% with no portfolio-level cap. Every simulated return
# figure produced before 2026-09-08 measured that other strategy, and the live
# account beat its own simulation by roughly 20pp because of it.
INITIAL_CAPITAL    = 10_000   # starting cash in USD
SLIPPAGE           = 0.001    # 0.1% price penalty applied on entry and exit
COMMISSION         = 1.00     # flat fee per trade (entry or exit), in USD

# Walk-forward split boundaries (must match trainer.py)
TRAIN_PCT = 0.70
VAL_PCT   = 0.20
# TEST_PCT  = 0.10  (remainder)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_model():
    """Load the XGBoost model bundle from MODEL_DIR and unpack it.

    Returns:
        model   — trained XGBoost classifier
        encoder — fitted LabelEncoder (use encoder.classes_ for class labels)
    """
    path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No model found at {path} — run trainer.py first."
        )
    bundle = joblib.load(path)
    return bundle["model"], bundle["encoder"]


def load_all_tickers(model, encoder) -> dict[str, pd.DataFrame]:
    """
    Load, process, and generate model signals for every ticker in WATCHLIST.

    Returns a dict mapping ticker to DataFrame with columns:
        Close, Signal (prediction after confidence threshold), Confidence
    """
    ticker_data   = {}
    model_classes = np.array(encoder.classes_)

    for ticker in WATCHLIST:
        try:
            df = load_and_process(ticker)
            X  = df[FEATURE_COLUMNS].values

            proba    = model.predict_proba(X)
            top_idx  = np.argmax(proba, axis=1)
            top_prob = proba[np.arange(len(proba)), top_idx]
            signals  = model_classes[top_idx].copy()
            signals[top_prob < CONFIDENCE_THRESHOLD] = "HOLD"

            df = df[["Close"]].copy()
            df["Signal"]     = signals
            df["Confidence"] = top_prob
            ticker_data[ticker] = df
            print(f"[OK]    {ticker} — {len(df)} rows")
        except FileNotFoundError:
            print(f"[SKIP]  {ticker} — CSV not found")

    return ticker_data


def _split_dates(all_dates: list) -> dict[str, list]:
    """
    Partition a sorted list of trading dates into train / validation / test
    slices using the same 70/20/10 proportions as trainer.py.
    """
    n         = len(all_dates)
    train_end = int(n * TRAIN_PCT)
    val_end   = int(n * (TRAIN_PCT + VAL_PCT))
    return {
        "train":      all_dates[:train_end],
        "validation": all_dates[train_end:val_end],
        "test":       all_dates[val_end:],
        "full":       all_dates,
    }


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def _simulate(
    dates: list,
    ticker_data: dict[str, pd.DataFrame],
) -> tuple[list, list]:
    """
    Run the day-by-day portfolio simulation over the given dates.

    Returns:
        trade_log    — list of dicts, one per closed trade
        equity_curve — list of dicts, one per day
    """
    cash         = float(INITIAL_CAPITAL)
    positions    = {}   # ticker to{shares, entry_price, cost}
    trade_log    = []
    equity_curve = []

    for date in dates:

        # Step 1: close positions where today's signal is no longer BUY
        to_close = [
            ticker for ticker, pos in positions.items()
            if ticker in ticker_data
            and date in ticker_data[ticker].index
            and ticker_data[ticker].loc[date, "Signal"] != "BUY"
        ]
        for ticker in to_close:
            pos        = positions.pop(ticker)
            close_px   = ticker_data[ticker].loc[date, "Close"]
            exit_price = close_px * (1 - SLIPPAGE)
            proceeds   = pos["shares"] * exit_price - COMMISSION
            cash      += proceeds
            trade_log.append({
                "ticker":       ticker,
                "entry_price":  pos["entry_price"],
                "exit_price":   exit_price,
                "shares":       pos["shares"],
                "cost":         pos["cost"],
                "proceeds":     proceeds,
                "trade_return": (proceeds - pos["cost"]) / pos["cost"],
            })

        # Step 2: compute current exposure
        invested_value  = sum(
            pos["shares"] * ticker_data[t].loc[date, "Close"]
            if t in ticker_data and date in ticker_data[t].index
            else pos["cost"]
            for t, pos in positions.items()
        )
        portfolio_value = cash + invested_value
        exposure_pct    = invested_value / portfolio_value if portfolio_value > 0 else 0

        # Step 3: open new positions on BUY signals
        for ticker, df in ticker_data.items():
            if ticker in positions or date not in df.index:
                continue
            row = df.loc[date]
            if row["Signal"] != "BUY" or exposure_pct >= MAX_TOTAL_EXPOSURE:
                continue

            alloc_pct  = min(row["Confidence"] * MAX_POSITION_PCT, MAX_POSITION_PCT)
            alloc_pct  = min(alloc_pct, MAX_TOTAL_EXPOSURE - exposure_pct)
            alloc_cash = portfolio_value * alloc_pct
            if alloc_cash < 10:
                continue

            entry_price = row["Close"] * (1 + SLIPPAGE)
            shares      = (alloc_cash - COMMISSION) / entry_price
            if shares <= 0:
                continue

            cost = shares * entry_price + COMMISSION
            if cost > cash:
                shares = (cash - COMMISSION) / entry_price
                if shares <= 0:
                    continue
                cost = shares * entry_price + COMMISSION

            cash          -= cost
            invested_value += shares * entry_price
            exposure_pct   = invested_value / portfolio_value if portfolio_value > 0 else 0
            positions[ticker] = {"shares": shares, "entry_price": entry_price, "cost": cost}

        # Step 4: record daily equity (recalculate after new positions)
        invested_value = sum(
            pos["shares"] * ticker_data[t].loc[date, "Close"]
            if t in ticker_data and date in ticker_data[t].index
            else pos["cost"]
            for t, pos in positions.items()
        )
        equity_curve.append({
            "date":            date,
            "portfolio_value": cash + invested_value,
            "cash":            cash,
            "invested_value":  invested_value,
            "open_positions":  len(positions),
        })

    # Close any positions still open at the end of the period
    last_date = dates[-1]
    for ticker, pos in list(positions.items()):
        close_px   = (ticker_data[ticker].loc[last_date, "Close"]
                      if ticker in ticker_data and last_date in ticker_data[ticker].index
                      else pos["entry_price"])
        exit_price = close_px * (1 - SLIPPAGE)
        proceeds   = pos["shares"] * exit_price - COMMISSION
        cash      += proceeds
        trade_log.append({
            "ticker":       ticker,
            "entry_price":  pos["entry_price"],
            "exit_price":   exit_price,
            "shares":       pos["shares"],
            "cost":         pos["cost"],
            "proceeds":     proceeds,
            "trade_return": (proceeds - pos["cost"]) / pos["cost"],
        })

    return trade_log, equity_curve


# ---------------------------------------------------------------------------
# Buy-and-hold benchmark
# ---------------------------------------------------------------------------
#
# The question the classification metrics cannot answer: does running this bot
# beat simply holding? Without this arm, a positive backtest return says nothing
# -- a rising market makes almost any long-biased strategy look profitable.
#
# Both arms share _summarise() below, so slippage, commission, drawdown and
# return are computed identically. The only difference is which trades happen.

def _simulate_buy_and_hold(
    dates: list,
    ticker_data: dict[str, pd.DataFrame],
) -> tuple[list, list]:
    """Equal-weight the watchlist on day one and hold to the end.

    Uses the same SLIPPAGE and COMMISSION as the strategy: an unpriced
    benchmark would flatter itself against a strategy that pays costs.
    """
    if not dates or not ticker_data:
        return [], []

    first_date = dates[0]
    tradable = [
        t for t, df in ticker_data.items()
        if first_date in df.index and df.loc[first_date, "Close"] > 0
    ]
    if not tradable:
        return [], []

    cash = float(INITIAL_CAPITAL)
    alloc_each = cash / len(tradable)
    positions = {}

    for ticker in tradable:
        entry_price = ticker_data[ticker].loc[first_date, "Close"] * (1 + SLIPPAGE)
        shares = (alloc_each - COMMISSION) / entry_price
        if shares <= 0:
            continue
        cost = shares * entry_price + COMMISSION
        cash -= cost
        positions[ticker] = {"shares": shares, "entry_price": entry_price, "cost": cost}

    equity_curve = []
    for date in dates:
        invested = sum(
            pos["shares"] * ticker_data[t].loc[date, "Close"]
            if date in ticker_data[t].index else pos["cost"]
            for t, pos in positions.items()
        )
        equity_curve.append({
            "date":            date,
            "portfolio_value": cash + invested,
            "cash":            cash,
            "invested_value":  invested,
            "open_positions":  len(positions),
        })

    # Liquidate on the final date, matching how the strategy closes out.
    last_date = dates[-1]
    trade_log = []
    for ticker, pos in positions.items():
        close_px = (ticker_data[ticker].loc[last_date, "Close"]
                    if last_date in ticker_data[ticker].index else pos["entry_price"])
        exit_price = close_px * (1 - SLIPPAGE)
        proceeds = pos["shares"] * exit_price - COMMISSION
        trade_log.append({
            "ticker":       ticker,
            "entry_price":  pos["entry_price"],
            "exit_price":   exit_price,
            "shares":       pos["shares"],
            "cost":         pos["cost"],
            "proceeds":     proceeds,
            "trade_return": (proceeds - pos["cost"]) / pos["cost"],
        })

    return trade_log, equity_curve


def _load_spy_close() -> pd.Series | None:
    """SPY closes for the market-hold arm. None if unavailable."""
    try:
        from features import _load_market_close
        return _load_market_close("SPY")
    except Exception as exc:
        print(f"  [BENCHMARK] Could not load SPY: {exc}")
        return None


def _simulate_market_hold(dates: list) -> tuple[list, list]:
    """Hold SPY for the period — the actual do-nothing alternative."""
    close = _load_spy_close()
    if close is None or not dates:
        return [], []

    available = close.reindex(pd.DatetimeIndex(dates)).ffill().dropna()
    if available.empty:
        print("  [BENCHMARK] SPY has no overlap with the backtest dates.")
        return [], []

    entry_price = float(available.iloc[0]) * (1 + SLIPPAGE)
    shares = (INITIAL_CAPITAL - COMMISSION) / entry_price
    cost = shares * entry_price + COMMISSION
    cash = INITIAL_CAPITAL - cost

    equity_curve = [
        {
            "date":            date,
            "portfolio_value": cash + shares * float(available.loc[date]),
            "cash":            cash,
            "invested_value":  shares * float(available.loc[date]),
            "open_positions":  1,
        }
        for date in available.index
    ]

    exit_price = float(available.iloc[-1]) * (1 - SLIPPAGE)
    proceeds = shares * exit_price - COMMISSION
    trade_log = [{
        "ticker":       "SPY",
        "entry_price":  entry_price,
        "exit_price":   exit_price,
        "shares":       shares,
        "cost":         cost,
        "proceeds":     proceeds,
        "trade_return": (proceeds - cost) / cost,
    }]

    return trade_log, equity_curve


# ---------------------------------------------------------------------------
# Shared statistics
# ---------------------------------------------------------------------------
def _summarise(trade_log: list, equity_curve: list, dates: list, split: str) -> dict:
    """Turn a trade log and equity curve into summary statistics.

    Shared by the strategy and both benchmark arms so the comparison cannot
    drift: any change to how return or drawdown is computed applies to all of
    them at once.
    """
    if not equity_curve:
        return {
            "split": split, "date_range": "N/A", "n_days": 0,
            "final_value": float(INITIAL_CAPITAL), "total_return": 0.0,
            "n_trades": 0, "win_rate": 0.0, "avg_return": 0.0,
            "best_trade": 0.0, "worst_trade": 0.0, "max_drawdown": 0.0,
            "avg_exposure": 0.0, "equity_df": pd.DataFrame(),
        }

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    final_value = float(equity_df["portfolio_value"].iloc[-1])
    returns = [t["trade_return"] for t in trade_log]
    n_trades = len(trade_log)

    if n_trades > 0:
        win_rate = len([r for r in returns if r > 0]) / n_trades * 100
        avg_return = float(np.mean(returns)) * 100
        best_trade = max(returns) * 100
        worst_trade = min(returns) * 100
    else:
        win_rate = avg_return = best_trade = worst_trade = 0.0

    pv = equity_df["portfolio_value"]
    max_drawdown = float(((pv - pv.cummax()) / pv.cummax() * 100).min()) if len(pv) > 1 else 0.0

    # Average capital actually at risk. Reported because a strategy capped at
    # MAX_TOTAL_EXPOSURE is not comparable to a fully-invested benchmark
    # without knowing how much of any gap is exposure rather than selection.
    with np.errstate(divide="ignore", invalid="ignore"):
        exposure = (equity_df["invested_value"] / equity_df["portfolio_value"]).replace(
            [np.inf, -np.inf], np.nan
        )
    avg_exposure = float(exposure.mean() * 100) if exposure.notna().any() else 0.0

    date_range = (
        f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
        if dates else "N/A"
    )

    return {
        "split":        split,
        "date_range":   date_range,
        "n_days":       len(dates),
        "final_value":  final_value,
        "total_return": (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        "n_trades":     n_trades,
        "win_rate":     win_rate,
        "avg_return":   avg_return,
        "best_trade":   best_trade,
        "worst_trade":  worst_trade,
        "max_drawdown": max_drawdown,
        "avg_exposure": avg_exposure,
        "equity_df":    equity_df,
    }


def run_benchmarks(
    split: str = "test",
    *,
    model=None,
    encoder=None,
    ticker_data: dict | None = None,
) -> dict:
    """Run the strategy against buy-and-hold arms over the same dates.

    Returns {"strategy": stats, "buy_and_hold": stats, "market_hold": stats}.
    """
    if ticker_data is None:
        if model is None or encoder is None:
            model, encoder = load_model()
        ticker_data = load_all_tickers(model, encoder)
    if not ticker_data:
        return {}

    all_dates = sorted(set().union(*[set(df.index) for df in ticker_data.values()]))
    dates = _split_dates(all_dates)[split]
    if not dates:
        return {}

    strat_trades, strat_equity = _simulate(dates, ticker_data)
    bh_trades, bh_equity = _simulate_buy_and_hold(dates, ticker_data)
    mkt_trades, mkt_equity = _simulate_market_hold(dates)

    return {
        "strategy":     _summarise(strat_trades, strat_equity, dates, split),
        "buy_and_hold": _summarise(bh_trades, bh_equity, dates, split),
        "market_hold":  _summarise(mkt_trades, mkt_equity, dates, split),
    }


def print_benchmark_comparison(results: dict) -> None:
    """Print the strategy beside its benchmarks, and state the verdict."""
    if not results:
        print("  [BENCHMARK] No results to compare.")
        return

    strategy = results["strategy"]
    rows = [
        ("Model strategy", strategy),
        ("Equal-weight hold", results["buy_and_hold"]),
        ("SPY hold", results["market_hold"]),
    ]

    print("\n" + "=" * 82)
    print(f"  BUY-AND-HOLD BENCHMARK - {strategy['split'].upper()}  ({strategy['date_range']})")
    print("=" * 82)

    # The train split is what the model was fitted on, and `full` is dominated
    # by it. Both produce spectacular returns that measure memorisation, not
    # edge -- reading them as performance is the single easiest way to conclude
    # this strategy works when it does not.
    if strategy["split"] in ("train", "full"):
        print("  *** IN-SAMPLE - NOT EVIDENCE OF EDGE ***")
        print("  The model was fitted on these dates, so the return below reflects")
        print("  memorisation. Only the `test` split is fully out-of-sample.")
        print("=" * 82)
    print(f"  {'':<20}{'return':>10}{'final $':>13}{'max DD':>10}"
          f"{'trades':>9}{'win %':>8}{'avg exp %':>11}")
    print("  " + "-" * 78)
    for label, s in rows:
        if not s or not s["n_days"]:
            print(f"  {label:<20}{'unavailable':>10}")
            continue
        print(f"  {label:<20}{s['total_return']:>9.2f}%{s['final_value']:>13,.2f}"
              f"{s['max_drawdown']:>9.2f}%{s['n_trades']:>9}"
              f"{s['win_rate']:>7.1f}%{s['avg_exposure']:>10.1f}%")
    print("  " + "-" * 78)

    # --- Verdict -----------------------------------------------------------
    beat_any = False
    for label, s in rows[1:]:
        if not s or not s["n_days"]:
            continue
        delta = strategy["total_return"] - s["total_return"]
        verdict = "BEATS" if delta > 0 else "LOSES TO"
        print(f"  Strategy {verdict} {label.lower()} by {abs(delta):.2f}pp")
        beat_any = beat_any or delta > 0

    if not beat_any:
        print()
        print("  The strategy did not beat a passive alternative over this period.")
        print("  Model tuning cannot fix that -- the edge has to come from")
        print("  somewhere other than the current features and labels.")

    if strategy.get("avg_exposure", 0) < 60:
        print()
        print(f"  NOTE: the strategy averaged {strategy['avg_exposure']:.1f}% invested "
              f"against ~100% for the")
        print("  hold arms, so part of any gap is lower market exposure rather than")
        print("  worse selection.")
    print("=" * 82)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtest(
    split: str = "full",
    *,
    model=None,
    encoder=None,
    ticker_data: dict | None = None,
) -> dict:
    """
    Run the backtest simulation for the requested date split.

    Args:
        split:       One of 'full', 'train', 'validation', 'test'.
        model:       Pre-loaded model (loads from disk if None).
        ticker_data: Pre-loaded ticker data (loads if None).

    Returns a dict of summary statistics for the split.
    """
    if split not in ("full", "train", "validation", "test"):
        raise ValueError(f"split must be 'full', 'train', 'validation', or 'test'; got '{split}'")

    if model is None or encoder is None:
        model, encoder = load_model()
    if ticker_data is None:
        print("=== Loading data and generating signals ===")
        ticker_data = load_all_tickers(model, encoder)
        if not ticker_data:
            print("No data loaded — run data_collector.py first.")
            return {}

    all_dates = sorted(set().union(*[set(df.index) for df in ticker_data.values()]))
    dates     = _split_dates(all_dates)[split]

    trade_log, equity_curve = _simulate(dates, ticker_data)
    return _summarise(trade_log, equity_curve, dates, split)


def _print_summary(stats: dict, label: str = "") -> None:
    """Print a detailed performance summary for one split."""
    tag = f"  {label}" if label else ""
    div = "-" * 42
    print(f"\n{'=' * 42}")
    print(f"  BACKTEST RESULTS — {stats['split'].upper()}{tag}")
    print(f"{'=' * 42}")
    print(f"  Period           : {stats['date_range']}")
    print(f"  Initial capital  : ${INITIAL_CAPITAL:>10,.2f}")
    print(f"  Final value      : ${stats['final_value']:>10,.2f}")
    print(f"  Total return     : {stats['total_return']:>+10.2f}%")
    print(div)
    print(f"  Trades executed  : {stats['n_trades']:>10}")
    print(f"  Win rate         : {stats['win_rate']:>10.1f}%")
    print(f"  Avg trade return : {stats['avg_return']:>+10.2f}%")
    print(f"  Best trade       : {stats['best_trade']:>+10.2f}%")
    print(f"  Worst trade      : {stats['worst_trade']:>+10.2f}%")
    print(div)
    print(f"  Max drawdown     : {stats['max_drawdown']:>+10.2f}%")
    print(f"{'=' * 42}")


if __name__ == "__main__":
    import sys

    print("=== Loading data and generating signals ===")
    _model, _encoder = load_model()
    _ticker_data     = load_all_tickers(_model, _encoder)

    # `python backtest.py --benchmark [split]` answers the only question that
    # gates the rest: does running this beat simply holding?
    if "--benchmark" in sys.argv:
        _args = [a for a in sys.argv[1:] if a != "--benchmark"]
        _split = _args[0] if _args else "test"
        _results = run_benchmarks(_split, ticker_data=_ticker_data)
        print_benchmark_comparison(_results)
        sys.exit(0)

    splits   = ["train", "validation", "test", "full"]
    all_stats = {}

    for sp in splits:
        print(f"\n--- Running split: {sp} ---")
        stats = run_backtest(sp, model=_model, encoder=_encoder, ticker_data=_ticker_data)
        all_stats[sp] = stats

    # Save the full-period equity curve for later visualisation
    all_stats["full"]["equity_df"].to_csv("backtest_results.csv")
    print("\n  Full equity curve saved to backtest_results.csv")

    # --- Comparison table ---
    print(f"\n{'=' * 78}")
    print(f"  WALK-FORWARD SPLIT COMPARISON")
    print(f"{'=' * 78}")
    hdr = (f"  {'Split':<12}  {'Date Range':<25}  {'Return':>8}  "
           f"{'Win Rate':>9}  {'Trades':>7}  {'Max DD':>8}")
    print(hdr)
    print(f"  {'-' * 74}")

    for sp in splits:
        s      = all_stats[sp]
        note   = "  *** OUT-OF-SAMPLE (honest result)" if sp == "test" else ""
        label  = sp.capitalize()
        print(
            f"  {label:<12}  {s['date_range']:<25}  {s['total_return']:>+7.1f}%  "
            f"{s['win_rate']:>8.1f}%  {s['n_trades']:>7}  {s['max_drawdown']:>+7.1f}%"
            f"{note}"
        )

    print(f"{'=' * 78}")
