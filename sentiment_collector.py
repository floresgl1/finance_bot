"""
Fetches historical news headlines from Finnhub, scores each with FinBERT,
aggregates daily sentiment per ticker, and saves one CSV per ticker to
data/sentiment_TICKER.csv.

Run this once (or periodically) before training to populate sentiment features.

Requires a Finnhub API key — set the FINNHUB_API_KEY environment variable
or replace the os.getenv() call below with your key string.
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from config import DATA_DIR, WATCHLIST
from sentiment import _get_pipeline, _score_headline

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

def collect():
    if not FINNHUB_API_KEY:
        raise RuntimeError(
            "Finnhub API key not set. Export FINNHUB_API_KEY=<your_key> "
            "or set FINNHUB_API_KEY at the top of sentiment_collector.py."
        )

    try:
        import finnhub
    except ImportError:
        import subprocess, sys
        print("Installing finnhub-python ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "finnhub-python"])
        import finnhub

    client = finnhub.Client(api_key=FINNHUB_API_KEY)

    # Warm up FinBERT once before the loop
    _get_pipeline()

    os.makedirs(DATA_DIR, exist_ok=True)

    for i, ticker in enumerate(WATCHLIST):
        DATE_TO = datetime.now().strftime("%Y-%m-%d")

        out_path = os.path.join(DATA_DIR, f"sentiment_{ticker}.csv")
        if os.path.exists(out_path):
            try:
                existing = pd.read_csv(out_path)
                if not existing.empty:
                    max_date = pd.to_datetime(existing["Date"]).max()
                    DATE_FROM = (max_date + timedelta(days=1)).strftime("%Y-%m-%d")
                else:
                    DATE_FROM = "2022-01-01"
            except Exception:
                DATE_FROM = "2022-01-01"
        else:
            DATE_FROM = "2022-01-01"

        print(f"[{i+1}/{len(WATCHLIST)}] {ticker} — fetching news {DATE_FROM} to {DATE_TO} ...", end="", flush=True)

        try:
            articles = client.company_news(ticker, _from=DATE_FROM, to=DATE_TO)
        except Exception as e:
            print(f" ERROR: {e}")
            time.sleep(1)
            continue

        if not articles:
            print(" no articles found, skipping.")
            time.sleep(1)
            continue

        print(f" {len(articles)} articles found. Scoring ...", flush=True)

        records = []
        for article in articles:
            headline = (article.get("headline") or "").strip()
            if not headline:
                continue
            # Finnhub returns datetime as a Unix timestamp
            ts = article.get("datetime")
            if not ts:
                continue
            date = pd.to_datetime(ts, unit="s", utc=True).tz_localize(None).normalize()
            score = _score_headline(headline)
            records.append({"date": date, "score": score})

        if not records:
            print(f"  No scoreable headlines for {ticker}, skipping.")
            time.sleep(1)
            continue

        df = pd.DataFrame(records)
        daily = (
            df.groupby("date")["score"]
            .mean()
            .reset_index()
            .rename(columns={"date": "Date", "score": "sent_score_daily"})
        )
        daily["Date"] = daily["Date"].dt.strftime("%Y-%m-%d")

        out_path = os.path.join(DATA_DIR, f"sentiment_{ticker}.csv")
        daily.to_csv(out_path, index=False)
        print(f"  Saved {len(daily)} daily scores → {out_path}")

        if i < len(WATCHLIST) - 1:
            time.sleep(1)

    print("\n=== Sentiment collection complete ===")


if __name__ == "__main__":
    collect()
