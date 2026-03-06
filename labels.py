"""
Creates Buy / Sell / Hold target labels from future price returns.

For each row, calculates the forward N-day return and assigns a label:
Buy if return exceeds the buy threshold, Sell if it falls below the sell
threshold, and Hold otherwise.
"""

import numpy as np
import pandas as pd
from config import WATCHLIST, PREDICTION_DAYS
from features import load_and_process


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'Signal' column (BUY / SELL / HOLD) to a featured DataFrame.

    The label for each row is determined by the forward return over
    PREDICTION_DAYS trading days.  The final PREDICTION_DAYS rows are
    dropped because no future close price exists to label them.
    """
    df = df.copy()

    # Forward return: how much the close price changes over the next N days
    forward_return = df["Close"].shift(-PREDICTION_DAYS) / df["Close"] - 1

    # Volatility-adjusted dynamic thresholds (temporary test)
    dynamic_threshold = df["Volatility"] * 1.5

    # Map each return to a signal using vectorised np.select
    conditions = [
        forward_return >= dynamic_threshold,
        forward_return <= -dynamic_threshold,
    ]
    choices = ["BUY", "SELL"]
    df["Signal"] = np.select(conditions, choices, default="HOLD")

    # The last PREDICTION_DAYS rows have no future price, so their labels are
    # meaningless — drop them rather than keeping NaN-derived values.
    df = df.iloc[:-PREDICTION_DAYS]

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
