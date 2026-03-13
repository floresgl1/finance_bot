"""
Downloads market reference data (index ETFs and VIX) and saves Close prices
to data/market/.

Run this before training or backtesting:
    python market_data_collector.py

Files produced:
    data/market/SPY.csv
    data/market/XLK.csv
    data/market/XLF.csv
    data/market/XLE.csv
    data/market/XLV.csv
    data/market/XLY.csv
    data/market/^VIX.csv

Always re-downloads (existing files are overwritten).
"""

import os
import yfinance as yf

from config import DATA_DIR, HISTORY_PERIOD, WATCHLIST
from data_collector import download_ticker

MARKET_SYMBOLS = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLY", "^VIX"]
MARKET_DATA_DIR = os.path.join(DATA_DIR, "market")


def download_market_symbol(symbol: str) -> None:
    """Download Close prices for one market symbol and save to data/market/."""
    path = os.path.join(MARKET_DATA_DIR, f"{symbol}.csv")

    df = yf.download(symbol, period=HISTORY_PERIOD, auto_adjust=True, progress=False)

    if df.empty:
        print(f"[WARNING] {symbol} — no data returned, skipping")
        return

    # Flatten MultiIndex columns produced by recent yfinance versions
    if isinstance(df.columns, __import__("pandas").MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df[["Close"]].copy()
    close.index.name = "Date"
    close.to_csv(path)
    print(f"[OK]      {symbol} — {len(close)} rows saved to {path}")


def collect_all() -> None:
    """Create data/market/, (re-)download every market symbol, and download ticker OHLCV data."""
    os.makedirs(MARKET_DATA_DIR, exist_ok=True)

    for symbol in MARKET_SYMBOLS:
        download_market_symbol(symbol)

    print("\n=== Downloading ticker OHLCV data ===")
    for ticker in WATCHLIST:
        download_ticker(ticker)


if __name__ == "__main__":
    collect_all()
