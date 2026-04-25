# Uses `requests` against Tavily's REST API rather than the tavily-python SDK,
# to avoid adding a dependency for a single endpoint call.

import os
import requests

TAVILY_ENDPOINT = "https://api.tavily.com/search"

TICKER_TO_QUERY = {
    "AAPL": "Apple stock",
    "MSFT": "Microsoft stock",
    "NVDA": "Nvidia stock",
    "TSLA": "Tesla stock",
    "AMZN": "Amazon stock",
    "JPM":  "JPMorgan stock",
    "XOM":  "ExxonMobil stock",
    "JNJ":  "Johnson & Johnson stock",
    "META": "Meta Platforms stock",
    "NFLX": "Netflix stock",
    "PYPL": "PayPal stock",
    "INTC": "Intel Corporation stock",
}


def search_news(ticker: str, max_results: int = 5) -> list[dict]:
    if ticker not in TICKER_TO_QUERY:
        raise ValueError(
            f"Ticker {ticker!r} is not in TICKER_TO_QUERY. "
            f"Known tickers: {sorted(TICKER_TO_QUERY)}"
        )

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "TAVILY_API_KEY is not set in the environment."
        )

    payload = {
        "api_key": api_key,
        "query": TICKER_TO_QUERY[ticker],
        "max_results": max_results,
        "days": 3,
        "topic": "news",
    }

    try:
        response = requests.post(TAVILY_ENDPOINT, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        return [
            {
                "title": r.get("title", ""),
                "snippet": r.get("content", ""),
                "url": r.get("url", ""),
                "published": r.get("published_date"),
            }
            for r in results
        ]
    except Exception as e:
        print(f"  [search_news] Tavily request failed for {ticker}: {e}")
        return []
