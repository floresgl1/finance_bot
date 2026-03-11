"""
Engineers technical indicators as ML features from raw price data.

Computes momentum, trend, volatility, volume, market-context, sector-momentum,
and fear-index indicators using the `ta` library and yfinance, then drops
warm-up NaN rows before returning the final feature matrix.
"""

import os
from functools import lru_cache

import numpy as np
import pandas as pd
import ta

from config import DATA_DIR, WATCHLIST

MARKET_DATA_DIR = os.path.join(DATA_DIR, "market")


# Sector ETF for each ticker in the watchlist.
# Used to compute sector-relative momentum features.
SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK",
    "META": "XLK", "NFLX": "XLK", "INTC": "XLK", "PYPL": "XLK",
    "JPM":  "XLF",
    "XOM":  "XLE",
    "JNJ":  "XLV",
    "TSLA": "XLY", "AMZN": "XLY",
}


@lru_cache(maxsize=None)
def _load_market_close(symbol: str) -> pd.Series:
    """
    Load and cache the full Close price series for a market symbol from
    data/market/<symbol>.csv.

    Raises FileNotFoundError immediately with a clear message if the CSV is
    missing — run market_data_collector.py to generate it.

    Results are memoised so tickers sharing a sector ETF (e.g. all XLK stocks)
    only trigger one file read per session.
    """
    filename = f"{symbol}.csv"
    path = os.path.join(MARKET_DATA_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] {filename} not found in '{MARKET_DATA_DIR}'\n"
            f"        Run: python market_data_collector.py\n"
        )

    df = pd.read_csv(path, dtype=str)

    # Same defensive Date-index pattern used in load_and_process
    if "Date" in df.columns:
        df = df.set_index("Date")
    else:
        df = df.set_index(df.columns[0])

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    close = pd.to_numeric(df["Close"], errors="coerce")
    close.index = close.index.tz_localize(None)
    return close


def _download_close(symbol: str, start: str, end: str) -> pd.Series:
    """
    Return cached Close prices for a market symbol, read from data/market/.

    The start/end arguments are kept for call-site compatibility but are not
    used for filtering — callers already reindex to the ticker's date range.
    """
    return _load_market_close(symbol)


def add_features(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """
    Add technical indicator, market-context, sector-momentum, and VIX columns
    to a raw OHLCV DataFrame.

    Args:
        df:     Raw OHLCV DataFrame with a DatetimeIndex.
        ticker: Ticker symbol used to look up the correct sector ETF.
                Falls back to SPY for unknown tickers.

    Returns a new DataFrame with all original columns plus indicators,
    with warm-up NaN rows dropped.
    """
    df = df.copy()

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # --- Trend: Simple Moving Averages ---
    df["SMA_20"] = close.rolling(window=20).mean()
    df["SMA_50"] = close.rolling(window=50).mean()

    # --- Momentum: Relative Strength Index (14-period) ---
    df["RSI_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    # --- Trend: MACD and signal line ---
    macd_indicator = ta.trend.MACD(close=close)
    df["MACD"]        = macd_indicator.macd()
    df["MACD_signal"] = macd_indicator.macd_signal()

    # --- Volatility: Bollinger Bands (20-period, 2 std) ---
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df["BB_upper"]  = bb.bollinger_hband()
    df["BB_middle"] = bb.bollinger_mavg()
    df["BB_lower"]  = bb.bollinger_lband()

    # --- Volume: ratio of today's volume to its 20-day average ---
    df["Volume_Ratio"] = vol / vol.rolling(window=20).mean()

    # --- Returns and Volatility ---
    df["Daily_Return"] = close.pct_change()
    df["Volatility"]   = df["Daily_Return"].rolling(window=20).std()

    # --- Longer-horizon momentum ---
    df["Return_20d"] = close.pct_change(periods=20)
    df["Return_60d"] = close.pct_change(periods=60)

    # Date bounds used for all external data downloads
    start = df.index.min().strftime("%Y-%m-%d")
    end   = df.index.max().strftime("%Y-%m-%d")

    # --- Market context: SPY ---
    spy_close = _download_close("SPY", start, end).reindex(df.index, method="ffill")
    df["SPY_Return"]   = spy_close.pct_change(periods=7)
    spy_return_20d     = spy_close.pct_change(periods=20)
    df["Rel_Strength"] = df["Return_20d"] - spy_return_20d

    # --- Sector momentum ---
    # Use the ticker's mapped sector ETF; fall back to SPY for unknown symbols.
    sector_etf   = SECTOR_MAP.get(ticker, "SPY")
    sector_close = _download_close(sector_etf, start, end).reindex(df.index, method="ffill")

    df["Sector_Return_5d"]  = sector_close.pct_change(periods=5)
    df["Sector_Return_20d"] = sector_close.pct_change(periods=20)
    # Positive = stock outperforming its sector over 20 days
    df["Stock_vs_Sector"]   = df["Return_20d"] - df["Sector_Return_20d"]

    # --- VIX fear index ---
    vix_close = _download_close("^VIX", start, end).reindex(df.index, method="ffill")
    df["VIX_Level"]  = vix_close
    # Rising VIX = increasing market fear; 5-day window captures short-term spikes
    df["VIX_Change"] = vix_close.pct_change(periods=5)

    # Drop warm-up rows where any indicator is NaN (longest window: SMA_50 / Return_60d)
    df.dropna(inplace=True)

    return df


EARNINGS_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "earnings")


def _add_earnings_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Merge the most recent quarterly earnings surprise into each row of df.

    Uses merge_asof with direction='backward' so each trading day inherits the
    last reported Surprise(%) value.  The earnings CSVs are already shifted +1 day
    (look-ahead bias prevention handled in data_collector.py).

    Falls back to 0.0 (neutral signal) when the CSV is absent or has no data.
    """
    earnings_path = os.path.join(EARNINGS_DATA_DIR, f"{ticker}.csv")

    if os.path.exists(earnings_path):
        earnings_df = pd.read_csv(earnings_path, index_col="Date", parse_dates=True)
        earnings_df.index = pd.to_datetime(earnings_df.index).tz_localize(None)
        earnings_df = earnings_df.sort_index()

        # merge_asof requires both sides to be sorted; reset index to use as key
        stock_reset    = df.reset_index().sort_values("Date")
        earnings_reset = earnings_df.reset_index().sort_values("Date")

        merged = pd.merge_asof(
            stock_reset,
            earnings_reset[["Date", "Surprise(%)"]],
            on="Date",
            direction="backward",
        )
        merged = merged.set_index("Date")
        merged.index = pd.to_datetime(merged.index)
        df = merged
    else:
        df["Surprise(%)"] = float("nan")

    df["earnings_surprise"] = df["Surprise(%)"].fillna(0.0)
    df = df.drop(columns=["Surprise(%)"], errors="ignore")

    return df


_SENTIMENT_CSV = os.path.join(DATA_DIR, "sentiment", "sentiment_scores.csv")


def _add_sentiment_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Merge daily sentiment scores for `ticker` into `df` and compute derived
    rolling features.

    Reads data/sentiment/sentiment_scores.csv (produced by sentiment_collector.py),
    filters to the given ticker, merges on Date, then computes:

        sent_rolling_7d   — 7-day rolling mean of the daily sentiment score
        sent_momentum_7d  — 7-day change in sentiment (score − score[t-7])

    Forward-fills gaps and falls back to 0.0 when the CSV is absent.
    """
    if os.path.exists(_SENTIMENT_CSV):
        sent_all = pd.read_csv(_SENTIMENT_CSV, parse_dates=["Date"])
        sent = (
            sent_all[sent_all["Ticker"] == ticker]
            .set_index("Date")[["sentiment_score"]]
        )
        sent.index = pd.to_datetime(sent.index).tz_localize(None)
        df = df.join(sent, how="left")
    else:
        df["sentiment_score"] = float("nan")

    # Forward-fill gaps (weekends / holidays) then zero-fill any remainder
    df["sentiment_score"] = df["sentiment_score"].ffill().fillna(0.0)

    df["sent_rolling_7d"]  = df["sentiment_score"].rolling(window=7).mean().fillna(0.0)
    df["sent_momentum_7d"] = (
        df["sentiment_score"].diff(periods=7).fillna(0.0)
    )

    df = df.drop(columns=["sentiment_score"])

    return df


def load_and_process(ticker: str) -> pd.DataFrame:
    """Load a ticker's CSV from DATA_DIR, add features, and return the result."""
    path = os.path.join(DATA_DIR, f"{ticker}.csv")

    # Read without parsing dates so non-date values don't raise before we clean.
    df = pd.read_csv(path, dtype=str)

    # Normalise index: if a "Date" column exists (CSV saved with a leading
    # RangeIndex column), promote it; otherwise use whatever is in column 0.
    if "Date" in df.columns:
        df = df.set_index("Date")
    else:
        df = df.set_index(df.columns[0])

    # Drop any rows whose index cannot be interpreted as a date (e.g. stray
    # "Price" / "Ticker" header rows from older multi-level yfinance CSVs).
    valid_mask = pd.to_datetime(df.index, errors="coerce").notna()
    df = df[valid_mask]

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # Convert all columns to numeric; stray strings become NaN.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)

    # Pass ticker so add_features can look up the correct sector ETF
    df = add_features(df, ticker=ticker)

    # Merge sentiment features (gracefully no-ops if CSV is absent)
    df = _add_sentiment_features(df, ticker=ticker)

    # Merge earnings surprise features (gracefully no-ops if CSV is absent)
    df = _add_earnings_features(df, ticker=ticker)

    return df


if __name__ == "__main__":
    for ticker in WATCHLIST:
        try:
            df = load_and_process(ticker)
            print(f"[OK]    {ticker} — {df.shape[0]} rows x {df.shape[1]} cols")
            print(f"        columns: {', '.join(df.columns.tolist())}")
        except FileNotFoundError:
            print(f"[SKIP]  {ticker} — CSV not found, run data_collector.py first")
