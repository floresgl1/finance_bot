"""
Fetches yesterday's news headlines from the Finnhub API for all tickers
in WATCHLIST and saves them to data/headlines/raw_headlines.csv.

Requires FINNHUB_API_KEY in the environment or a .env file (via python-dotenv).

    python headlines_data_collector.py
"""

import os
import time
from datetime import date, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

from config import DATA_DIR, WATCHLIST

load_dotenv()

FINNHUB_API_KEY  = os.environ.get("FINNHUB_API_KEY")
HEADLINES_DIR    = os.path.join(DATA_DIR, "headlines")
OUTPUT_PATH      = os.path.join(HEADLINES_DIR, "raw_headlines.csv")
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"


def fetch_headlines(ticker: str, from_date: str, to_date: str) -> list[dict]:
    """Return a list of raw article dicts from Finnhub for the given date range."""
    params = {
        "symbol": ticker,
        "from":   from_date,
        "to":     to_date,
        "token":  FINNHUB_API_KEY,
    }
    resp = requests.get(FINNHUB_NEWS_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json() or []


def collect():
    if not FINNHUB_API_KEY:
        raise RuntimeError(
            "FINNHUB_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )

    os.makedirs(HEADLINES_DIR, exist_ok=True)

    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Fetching Finnhub headlines for {yesterday} ...")

    rows = []
    for i, ticker in enumerate(WATCHLIST, start=1):
        print(f"  [{i}/{len(WATCHLIST)}] {ticker} ...", end="", flush=True)
        try:
            articles = fetch_headlines(ticker, yesterday, yesterday)
            count = 0
            for article in articles:
                headline = (article.get("headline") or "").strip()
                summary  = (article.get("summary")  or "").strip()
                dt       = article.get("datetime")   # Unix timestamp int
                if headline:
                    rows.append({
                        "ticker":   ticker,
                        "headline": headline,
                        "summary":  summary,
                        "datetime": dt,
                    })
                    count += 1
            print(f" {count} article(s)")
        except Exception as exc:
            print(f" ERROR: {exc}")

        # Finnhub free tier allows ~30 req/s; be conservative to avoid 429s
        time.sleep(0.5)

    if rows:
        df = pd.DataFrame(rows, columns=["ticker", "headline", "summary", "datetime"])
    else:
        print("WARNING: No headlines fetched — writing empty CSV.")
        df = pd.DataFrame(columns=["ticker", "headline", "summary", "datetime"])

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df):,} rows → {OUTPUT_PATH}")
    print("=== Headlines collection complete ===")


if __name__ == "__main__":
    collect()
