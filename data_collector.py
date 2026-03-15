"""
Fetches historical OHLCV price data and earnings surprise data for tickers in the watchlist.

Uses yfinance to download daily price history and quarterly earnings dates/surprises,
handles missing values, and saves cleaned DataFrames for downstream feature engineering.

Earnings note: run once per quarter. Files are overwritten on each run so stale data
is never silently kept. The earnings date index is shifted +1 day to prevent look-ahead
bias (e.g. an earnings release on Oct 14 is only visible to the model from Oct 15).
"""

import os
import pandas as pd
import yfinance as yf
from config import WATCHLIST, HISTORY_PERIOD, DATA_DIR

EARNINGS_DIR = os.path.join(os.path.dirname(__file__), "data", "earnings")


def download_ticker(ticker: str) -> None:
    """Download historical OHLCV data for one ticker and save it as a CSV.

    Always re-downloads and overwrites any existing file so stale data
    is never silently kept.
    """
    path = os.path.join(DATA_DIR, f"{ticker}.csv")

    df = yf.download(ticker, period=HISTORY_PERIOD, auto_adjust=True, progress=False)

    if df.empty:
        print(f"[WARNING] {ticker} — no data returned, skipping")
        return

    # Flatten MultiIndex columns produced by recent yfinance versions
    # e.g. ("Close", "AAPL") → "Close"
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close", "High", "Low", "Open", "Volume"]]
    df.index.name = "Date"
    df.to_csv(path)
    print(f"[OK]      {ticker} — {len(df)} rows saved to {path}")


def download_earnings(ticker: str) -> None:
    """
    Download earnings surprise data for one ticker and save to data/earnings/TICKER.csv.

    The earnings date index is shifted forward by 1 day to prevent look-ahead bias —
    a surprise reported on Oct 14 is only available to the model from Oct 15 onward.
    Files are always overwritten so stale quarterly data is never silently kept.
    """
    path = os.path.join(EARNINGS_DIR, f"{ticker}.csv")

    try:
        t = yf.Ticker(ticker)
        earnings_df = t.get_earnings_dates(limit=20)

        if earnings_df is None or earnings_df.empty:
            print(f"[WARNING] {ticker} earnings — no data returned, skipping")
            return

        # Keep only the Surprise(%) column
        if "Surprise(%)" not in earnings_df.columns:
            print(f"[WARNING] {ticker} earnings — 'Surprise(%)' column not found, skipping")
            return

        earnings_df = earnings_df[["Surprise(%)"]].copy()

        # Drop rows where Surprise(%) is NaN (future earnings dates with no result yet)
        earnings_df = earnings_df.dropna(subset=["Surprise(%)"])

        if earnings_df.empty:
            print(f"[WARNING] {ticker} earnings — no reported surprises available, skipping")
            return

        # Normalise index to date-only (strip any timezone / time component)
        earnings_df.index = pd.to_datetime(earnings_df.index).normalize().tz_localize(None)
        earnings_df.index.name = "Date"

        # Shift +1 day to prevent look-ahead bias
        earnings_df.index = earnings_df.index + pd.Timedelta(days=1)

        earnings_df.to_csv(path)
        print(f"[OK]      {ticker} earnings — {len(earnings_df)} rows saved to {path}")

    except Exception as e:
        print(f"[ERROR]   {ticker} earnings — failed to download: {e}")


def collect_all() -> None:
    """Create the data directory and download data for every ticker in WATCHLIST."""
    os.makedirs(DATA_DIR, exist_ok=True)

    for ticker in WATCHLIST:
        download_ticker(ticker)


def collect_earnings() -> None:
    """Create the earnings directory and download earnings surprises for every ticker."""
    os.makedirs(EARNINGS_DIR, exist_ok=True)

    print("\n=== Downloading earnings data ===")
    for ticker in WATCHLIST:
        download_earnings(ticker)


if __name__ == "__main__":
    collect_all()
    collect_earnings()
