"""
FinBERT-based sentiment analysis for stock news headlines.

Loads ProsusAI/finbert once (lazily on first call) and scores recent
yfinance headlines for each ticker. Returns a single sentiment score
in [-1, +1] suitable for use as a signal-confirmation filter in predictor.py.
"""

import numpy as np
import yfinance as yf

from config import WATCHLIST

# FinBERT pipeline — loaded once on first call, reused for every subsequent call
_pipeline = None


def _get_pipeline():
    """Lazily load the FinBERT pipeline so import time stays fast."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        print("  [FinBERT] Loading model (first run only, may take a moment)...")
        _pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None,
        )
        print("  [FinBERT] Model ready.")
    return _pipeline


def _score_headline(text: str) -> float:
    """
    Run one headline through FinBERT and return a score in [-1, +1].

    Score = positive_probability - negative_probability.
    Fully positive text scores near +1; fully negative near -1;
    neutral text scores near 0.
    """
    pipe    = _get_pipeline()
    results = pipe(text, truncation=True, max_length=512)[0]
    scores  = {item["label"]: item["score"] for item in results}
    return scores.get("positive", 0.0) - scores.get("negative", 0.0)


def _extract_title(article: dict) -> str:
    """
    Extract the headline title from a yfinance news dict.

    yfinance 0.2+ nests content in article["content"]["title"];
    older versions expose it directly at article["title"].
    """
    if isinstance(article.get("content"), dict):
        title = article["content"].get("title", "")
    else:
        title = article.get("title", "")
    return (title or "").strip()


def get_sentiment(ticker: str) -> dict:
    """
    Fetch recent news for a ticker, score each headline with FinBERT,
    and return an aggregated sentiment summary.

    Up to 5 most recent headlines are used.  The final sentiment_score
    is the simple average of individual headline scores.

    Returns a dict with keys:
        sentiment_score  — float in [-1, +1]; positive means bullish
        sentiment_label  — "Positive", "Negative", or "Neutral"
        article_count    — number of articles successfully scored
        headlines        — list of headline strings that were analyzed
    """
    try:
        news = yf.Ticker(ticker).news or []
    except Exception:
        news = []

    headlines = [
        _extract_title(a)
        for a in news[:5]
        if _extract_title(a)
    ]

    if not headlines:
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "Neutral",
            "article_count":   0,
            "headlines":       [],
        }

    scores    = [_score_headline(h) for h in headlines]
    avg_score = float(np.mean(scores))

    if avg_score > 0.1:
        label = "Positive"
    elif avg_score < -0.1:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "sentiment_score": round(avg_score, 3),
        "sentiment_label": label,
        "article_count":   len(headlines),
        "headlines":       headlines,
    }


def get_sentiment_all(tickers: list) -> dict:
    """
    Run sentiment analysis for every ticker in the list.

    Errors (no news, network failure, etc.) return a neutral placeholder
    with an "error" key so callers never see an exception.

    Returns a dict keyed by ticker symbol.
    """
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = get_sentiment(ticker)
        except Exception as e:
            results[ticker] = {
                "sentiment_score": 0.0,
                "sentiment_label": "Neutral",
                "article_count":   0,
                "headlines":       [],
                "error":           str(e),
            }
    return results


if __name__ == "__main__":
    print("Fetching and scoring news headlines...\n")
    sentiments = get_sentiment_all(WATCHLIST)

    div = "+" + "-" * 8 + "+" + "-" * 9 + "+" + "-" * 12 + "+" + "-" * 10 + "+"
    hdr = f"| {'Ticker':<6} | {'Score':>7} | {'Label':<10} | {'Articles':>8} |"
    print(div)
    print(hdr)
    print(div)

    for ticker in WATCHLIST:
        s        = sentiments.get(ticker, {})
        score    = s.get("sentiment_score", 0.0)
        label    = s.get("sentiment_label", "N/A")
        articles = s.get("article_count", 0)
        bar      = "+" * round((score + 1) * 5)   # visual bar: 0-10 chars
        print(f"| {ticker:<6} | {score:>+6.3f}  | {label:<10} | {articles:>8} |  {bar}")

    print(div)
    print("\nSample headlines:")
    for ticker in WATCHLIST[:3]:
        s = sentiments.get(ticker, {})
        print(f"\n  {ticker}:")
        for h in s.get("headlines", []):
            print(f"    - {h}")
