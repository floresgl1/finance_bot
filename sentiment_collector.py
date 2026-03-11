"""
Loads the sabareesh88/FNSPID_nasdaq dataset from HuggingFace by fetching
parquet shard URLs via the HuggingFace datasets-server API, reading each shard
with pyarrow (pq.read_table) using column selection and a push-down filter on
Stock_symbol, scoring the Lsa_summary text column with FinBERT, averaging
scores per ticker per day, filling missing days with ffill/bfill, and saving
a consolidated CSV to data/sentiment/sentiment_scores.csv with columns:
Date, Ticker, sentiment_score.

Run once before training to populate sentiment features used by features.py.

    python sentiment_collector.py
"""

import io
import os
import time

import pandas as pd
import pyarrow.parquet as pq
import requests

from config import DATA_DIR, WATCHLIST
from sentiment import _get_pipeline, _score_headline

HF_REPO    = "sabareesh88/FNSPID_nasdaq"
TICKER_COL = "Stock_symbol"
DATE_COL   = "Date"
TEXT_COL   = "Lsa_summary"

SENTIMENT_DIR     = os.path.join(DATA_DIR, "sentiment")
OUTPUT_PATH       = os.path.join(SENTIMENT_DIR, "sentiment_scores.csv")
CHECKPOINT_PATH   = os.path.join(SENTIMENT_DIR, "sentiment_checkpoint.csv")
CHECKPOINT_BATCH  = 1000


def _fetch_shard_urls(repo: str = HF_REPO) -> list[str]:
    """
    Query the HuggingFace datasets-server API to get all parquet shard URLs
    for `repo`.  Returns a flat list of URLs (all splits combined).
    """
    api_url = f"https://datasets-server.huggingface.co/parquet?dataset={repo}"
    print(f"Fetching shard URLs from HuggingFace API: {api_url}")
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()

    parquet_files = resp.json().get("parquet_files", [])
    if not parquet_files:
        raise RuntimeError(
            f"No parquet files returned for {repo}.  "
            "Check the dataset name or try again later."
        )

    urls = [f["url"] for f in parquet_files]
    print(f"  Found {len(urls)} shard(s).")
    return urls


def _read_shard(url: str, watchlist: list[str], max_retries: int = 5) -> pd.DataFrame:
    """
    Download one parquet shard and return a DataFrame containing only rows
    where Stock_symbol is in `watchlist`, with only the three needed columns.

    Uses pq.read_table with:
      - columns  → load only Date, Stock_symbol, Lsa_summary
      - filters  → push-down predicate to skip non-watchlist row groups

    Retries up to max_retries times with exponential backoff (2 ** attempt
    seconds) to allow the server time to recover between attempts.
    """
    shard_name = url.split("/")[-1]

    for attempt in range(max_retries):
        is_final = attempt == max_retries - 1
        try:
            print(f"  Downloading shard: {shard_name} ...", end="", flush=True)
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()

            buf = io.BytesIO(resp.content)
            table = pq.read_table(
                buf,
                columns=[DATE_COL, TICKER_COL, TEXT_COL],
                filters=[(TICKER_COL, "in", set(watchlist))],
            )
            df = table.to_pandas()
            print(f" {len(df):,} rows after filter.")
            return df

        except Exception as exc:
            if is_final:
                print(f"\n  ERROR: All {max_retries} attempts failed for {shard_name}.")
                raise
            wait = 2 ** attempt
            print(f"\n  Attempt {attempt + 1}/{max_retries} failed: {exc}")
            print(f"  Retrying in {wait}s ...", flush=True)
            time.sleep(wait)


def _score_texts(df: pd.DataFrame, checkpoint_path: str) -> pd.Series:
    """
    Score each row's Lsa_summary text with FinBERT, flushing batches of
    ~CHECKPOINT_BATCH rows to checkpoint_path as scoring progresses.
    Returns a float Series aligned to df.index.
    Empty / null texts are scored as 0.0 (neutral).
    """
    dates   = df["_date"].tolist()
    tickers = df[TICKER_COL].tolist()
    texts   = df[TEXT_COL].tolist()

    scores       = []
    batch_buf    = []
    total        = len(texts)
    report_every = max(1, total // 20)  # progress update ~every 5%
    wrote_header = os.path.exists(checkpoint_path)  # header already present on resume

    for i, (date, ticker, text) in enumerate(zip(dates, tickers, texts), start=1):
        score = (
            0.0
            if (not text or not isinstance(text, str) or not text.strip())
            else _score_headline(text.strip())
        )
        scores.append(score)
        batch_buf.append({"Date": date, "Ticker": ticker, TEXT_COL: text, "_score": score})

        # Flush checkpoint every CHECKPOINT_BATCH rows and on the final row
        if len(batch_buf) >= CHECKPOINT_BATCH or i == total:
            chunk = pd.DataFrame(batch_buf)
            chunk.to_csv(checkpoint_path, mode="a", header=not wrote_header, index=False)
            wrote_header = True
            batch_buf = []

        if i % report_every == 0 or i == total:
            pct = i / total * 100
            bar_filled = int(pct // 5)
            bar = "#" * bar_filled + "-" * (20 - bar_filled)
            print(f"\r  [{bar}] {i:,}/{total:,} ({pct:.0f}%)", end="", flush=True)

    print()  # newline after progress bar completes
    return pd.Series(scores, index=df.index, dtype=float)


def collect():
    os.makedirs(SENTIMENT_DIR, exist_ok=True)

    # 1. Fetch shard URLs via HuggingFace API
    shard_urls = _fetch_shard_urls()

    # 2. Warm up FinBERT once before scoring loop
    print("Warming up FinBERT ...")
    _get_pipeline()

    # 3. Read each shard, filter to WATCHLIST, collect rows
    shard_frames = []
    for i, url in enumerate(shard_urls, start=1):
        print(f"[{i}/{len(shard_urls)}]", end=" ")
        shard_df = _read_shard(url, WATCHLIST)
        if not shard_df.empty:
            shard_frames.append(shard_df)

    if not shard_frames:
        print("WARNING: No rows matched the WATCHLIST across all shards.")
        return

    raw = pd.concat(shard_frames, ignore_index=True)
    print(f"\nTotal rows loaded: {len(raw):,} for {raw[TICKER_COL].nunique()} tickers")

    # 4. Normalise ticker
    raw[TICKER_COL] = raw[TICKER_COL].astype(str).str.strip().str.upper()

    # 5. Normalise date to calendar day
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL], errors="coerce", utc=True)
    raw = raw.dropna(subset=[DATE_COL])
    raw["_date"] = raw[DATE_COL].dt.tz_localize(None).dt.normalize()

    # 6. Score Lsa_summary with FinBERT (checkpoint-resume)
    # Build a composite key so already-scored rows can be identified on resume.
    raw["_key"] = (
        raw["_date"].astype(str) + "|"
        + raw[TICKER_COL] + "|"
        + raw[TEXT_COL].fillna("")
    )

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = pd.read_csv(CHECKPOINT_PATH)
        ckpt["_key"] = (
            pd.to_datetime(ckpt["Date"]).dt.normalize().astype(str) + "|"
            + ckpt["Ticker"].astype(str).str.strip().str.upper() + "|"
            + ckpt[TEXT_COL].fillna("")
        )
        key_to_score = dict(zip(ckpt["_key"], ckpt["_score"]))
        remaining = raw[~raw["_key"].isin(key_to_score)].copy()
        print(
            f"  Checkpoint found: {len(ckpt):,} already scored, "
            f"{len(remaining):,} remaining."
        )
    else:
        key_to_score = {}
        remaining = raw.copy()

    if not remaining.empty:
        print(f"Scoring {len(remaining):,} headlines with FinBERT ...")
        remaining["_score"] = _score_texts(remaining, CHECKPOINT_PATH)
        key_to_score.update(dict(zip(remaining["_key"], remaining["_score"])))
    else:
        print("All rows already scored from checkpoint — skipping FinBERT.")

    raw["_score"] = raw["_key"].map(key_to_score)
    raw = raw.drop(columns=["_key"])

    # Checkpoint no longer needed — delete it
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("Checkpoint deleted.")

    raw = raw.dropna(subset=["_score"])

    # 7. Average score per ticker per calendar day
    daily = (
        raw.groupby([TICKER_COL, "_date"])["_score"]
        .mean()
        .reset_index()
        .rename(columns={TICKER_COL: "Ticker", "_date": "Date", "_score": "sentiment_score"})
    )

    # 8. Build full date spine and ffill/bfill per ticker
    date_min   = daily["Date"].min()
    date_max   = daily["Date"].max()
    full_dates = pd.date_range(start=date_min, end=date_max, freq="D")

    filled_frames = []
    for ticker in WATCHLIST:
        ticker_df = daily[daily["Ticker"] == ticker].set_index("Date")

        if ticker_df.empty:
            print(f"  WARNING: No data for {ticker} — filling with 0.0")
            ticker_df = pd.DataFrame({"sentiment_score": 0.0}, index=full_dates)
        else:
            ticker_df = ticker_df[["sentiment_score"]].reindex(full_dates)
            ticker_df["sentiment_score"] = (
                ticker_df["sentiment_score"]
                .ffill()
                .bfill()
                .fillna(0.0)
            )

        ticker_df["Ticker"]     = ticker
        ticker_df.index.name    = "Date"
        filled_frames.append(ticker_df.reset_index())

    result = pd.concat(filled_frames, ignore_index=True)
    result["Date"] = result["Date"].dt.strftime("%Y-%m-%d")
    result = result[["Date", "Ticker", "sentiment_score"]]

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(result):,} rows → {OUTPUT_PATH}")
    print("=== Sentiment collection complete ===")


if __name__ == "__main__":
    collect()
