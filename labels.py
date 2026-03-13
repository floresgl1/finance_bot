"""
Creates Buy / Sell / Hold target labels from future price returns relative to SPY,
using a VIX-based dynamic threshold: <15 → 1.0%, 15-25 → 1.5%, >25 → 2.0%.
"""

import os

import numpy as np
import pandas as pd
from config import WATCHLIST
from features import load_and_process

_SPY_PATH = os.path.join(os.path.dirname(__file__), "data", "market", "SPY.csv")
_VIX_PATH = os.path.join(os.path.dirname(__file__), "data", "market", "^VIX.csv")
_WINDOW = 7


def _load_market_data() -> pd.DataFrame:
    """Load SPY and VIX CSVs and return a merged DataFrame with Date, spy_return_7d, vix_close."""
    spy = pd.read_csv(_SPY_PATH, parse_dates=["Date"])
    spy = spy[["Date", "Close"]].rename(columns={"Close": "spy_close"}).drop_duplicates(subset="Date")
    spy["spy_return_7d"] = spy["spy_close"].pct_change(_WINDOW).shift(-_WINDOW)

    vix = pd.read_csv(_VIX_PATH, parse_dates=["Date"])
    vix = vix[["Date", "Close"]].rename(columns={"Close": "vix_close"}).drop_duplicates(subset="Date")

    return spy[["Date", "spy_return_7d"]].merge(vix[["Date", "vix_close"]], on="Date", how="left")


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'Signal' column (BUY / SELL / HOLD) to a featured DataFrame.

    Labels are based on the stock's 7-day forward return relative to SPY,
    with a VIX-adjusted threshold applied row by row.
    The final 7 rows are dropped because no future close price exists.
    """
    df = df.copy()

    # Ensure Date is a column (not index) for merging
    if "Date" not in df.columns:
        df = df.reset_index()

    # Merge SPY returns and VIX levels on Date, then restore Date as index
    market = _load_market_data()
    df = df.merge(market, on="Date", how="left")
    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    df["spy_return_7d"] = df["spy_return_7d"].ffill()
    df["vix_close"] = df["vix_close"].ffill()

    # 7-day forward return for this stock (computed after merge to stay aligned)
    stock_return_7d = df["Close"].pct_change(_WINDOW).shift(-_WINDOW)

    # Relative outperformance vs SPY
    relative_return = stock_return_7d - df["spy_return_7d"]

    # VIX-based dynamic threshold per row
    df["threshold"] = df["vix_close"].apply(lambda vix: 0.010 if vix < 15 else (0.015 if vix < 25 else 0.020))

    conditions = [
        relative_return >= df["threshold"],
        relative_return <= -df["threshold"],
    ]
    choices = ["BUY", "SELL"]
    df["Signal"] = np.select(conditions, choices, default="HOLD")

    # Drop last _WINDOW rows — no future price data to label them
    df = df.iloc[:-_WINDOW]

    return df


if __name__ == "__main__":
    for ticker in WATCHLIST:
        try:
            df = load_and_process(ticker)
            df = add_labels(df)
            counts = df["Signal"].value_counts()
            total  = len(df)
            print(f"\n{ticker} ({total} rows)")
            for signal in ["BUY", "SELL", "HOLD"]:
                n = counts.get(signal, 0)
                print(f"  {signal:<4} {n:>4}  ({n / total * 100:.1f}%)")
        except FileNotFoundError:
            print(f"\n[SKIP] {ticker} — CSV not found, run data_collector.py first")
