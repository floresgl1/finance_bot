"""
Scores headlines in data/headlines/raw_headlines.csv with FinBERT and
appends the daily average sentiment per ticker to
data/sentiment/sentiment_scores.csv.

Run after headlines_data_collector.py:

    python sentiment_collector.py
"""

import os

import pandas as pd

from config import DATA_DIR, WATCHLIST
from sentiment import _get_pipeline, _score_headline

HEADLINES_PATH = os.path.join(DATA_DIR, "headlines", "raw_headlines.csv")
SENTIMENT_DIR  = os.path.join(DATA_DIR, "sentiment")
OUTPUT_PATH    = os.path.join(SENTIMENT_DIR, "sentiment_scores.csv")


def collect():
    os.makedirs(SENTIMENT_DIR, exist_ok=True)

    # 1. Load headlines CSV produced by headlines_data_collector.py
    if not os.path.exists(HEADLINES_PATH):
        raise FileNotFoundError(
            f"Headlines file not found: {HEADLINES_PATH}\n"
            "Run headlines_data_collector.py first."
        )

    raw = pd.read_csv(HEADLINES_PATH)
    print(f"Loaded {len(raw):,} headlines from {HEADLINES_PATH}")

    if raw.empty:
        print("WARNING: No headlines to score — nothing written.")
        return

    # 2. Parse date from Unix timestamp
    raw["datetime"] = pd.to_numeric(raw["datetime"], errors="coerce")
    raw["Date"] = (
        pd.to_datetime(raw["datetime"], unit="s", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    raw = raw.dropna(subset=["Date"])

    # 3. Build scoring text: headline + summary (FinBERT sees both)
    raw["text"] = (
        raw["headline"].fillna("").str.strip()
        + " "
        + raw["summary"].fillna("").str.strip()
    ).str.strip()

    # 4. Normalise ticker to uppercase
    raw["Ticker"] = raw["ticker"].astype(str).str.strip().str.upper()

    # 5. Warm up FinBERT once before the scoring loop
    print("Warming up FinBERT ...")
    _get_pipeline()

    # 6. Score each row
    total = len(raw)
    print(f"Scoring {total:,} headlines with FinBERT ...")
    report_every = max(1, total // 20)
    scores = []

    for i, text in enumerate(raw["text"].tolist(), start=1):
        score = 0.0 if not text else _score_headline(text)
        scores.append(score)

        if i % report_every == 0 or i == total:
            pct        = i / total * 100
            bar_filled = int(pct // 5)
            bar        = "#" * bar_filled + "-" * (20 - bar_filled)
            print(f"\r  [{bar}] {i:,}/{total:,} ({pct:.0f}%)", end="", flush=True)

    print()
    raw["_score"] = scores

    # 7. Average score per ticker per calendar day
    daily = (
        raw.groupby(["Ticker", "Date"])["_score"]
        .mean()
        .reset_index()
        .rename(columns={"_score": "sentiment_score"})
    )
    daily["Date"] = daily["Date"].dt.strftime("%Y-%m-%d")

    # 8. Append to existing sentiment_scores.csv, replacing any rows whose
    #    dates are being rewritten (avoids duplicates on re-runs)
    if os.path.exists(OUTPUT_PATH):
        existing  = pd.read_csv(OUTPUT_PATH)
        new_dates = set(daily["Date"].unique())
        existing  = existing[~existing["Date"].isin(new_dates)]
        result    = pd.concat([existing, daily], ignore_index=True)
    else:
        result = daily

    result = result[["Date", "Ticker", "sentiment_score"]]
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(result):,} rows → {OUTPUT_PATH}")
    print("=== Sentiment collection complete ===")


if __name__ == "__main__":
    collect()
