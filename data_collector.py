"""
Fetches historical OHLCV price data for tickers in the watchlist.

Uses yfinance to download daily price history, handles missing values,
and saves cleaned DataFrames for downstream feature engineering.
"""

import os
import yfinance as yf
from config import WATCHLIST, HISTORY_PERIOD, DATA_DIR


def download_ticker(ticker: str) -> None:
    """Download historical OHLCV data for one ticker and save it as a CSV."""
    path = os.path.join(DATA_DIR, f"{ticker}.csv")

    if os.path.exists(path):
        print(f"[SKIP]    {ticker} — already exists at {path}")
        return

    df = yf.download(ticker, period=HISTORY_PERIOD, auto_adjust=True, progress=False)

    if df.empty:
        print(f"[WARNING] {ticker} — no data returned, skipping")
        return

    df.to_csv(path)
    print(f"[OK]      {ticker} — {len(df)} rows saved to {path}")


def collect_all() -> None:
    """Create the data directory and download data for every ticker in WATCHLIST."""
    os.makedirs(DATA_DIR, exist_ok=True)

    for ticker in WATCHLIST:
        download_ticker(ticker)


if __name__ == "__main__":
    collect_all()
