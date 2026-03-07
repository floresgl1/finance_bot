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

from config import WATCHLIST, FEATURE_COLUMNS, CONFIDENCE_THRESHOLD, MODEL_DIR, MODEL_FILENAME
from features import load_and_process

# --- Backtest settings ---
INITIAL_CAPITAL    = 10_000   # starting cash in USD
MAX_POSITION_PCT   = 0.20     # max 20% of portfolio in any single stock
MAX_TOTAL_EXPOSURE = 0.80     # max 80% of portfolio invested at once
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

    equity_df   = pd.DataFrame(equity_curve).set_index("date")
    final_value = float(equity_df["portfolio_value"].iloc[-1]) if equity_curve else INITIAL_CAPITAL
    n_trades    = len(trade_log)
    returns     = [t["trade_return"] for t in trade_log]

    if n_trades > 0:
        win_rate    = len([r for r in returns if r > 0]) / n_trades * 100
        avg_return  = float(np.mean(returns)) * 100
        best_trade  = max(returns) * 100
        worst_trade = min(returns) * 100
    else:
        win_rate = avg_return = best_trade = worst_trade = 0.0

    pv           = equity_df["portfolio_value"]
    max_drawdown = float(((pv - pv.cummax()) / pv.cummax() * 100).min()) if len(pv) > 1 else 0.0
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    date_range = (
        f"{dates[0].strftime('%Y-%m-%d')} to{dates[-1].strftime('%Y-%m-%d')}"
        if dates else "N/A"
    )

    return {
        "split":        split,
        "date_range":   date_range,
        "n_days":       len(dates),
        "final_value":  final_value,
        "total_return": total_return,
        "n_trades":     n_trades,
        "win_rate":     win_rate,
        "avg_return":   avg_return,
        "best_trade":   best_trade,
        "worst_trade":  worst_trade,
        "max_drawdown": max_drawdown,
        "equity_df":    equity_df,
    }


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
    print("=== Loading data and generating signals ===")
    _model, _encoder = load_model()
    _ticker_data     = load_all_tickers(_model, _encoder)

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
