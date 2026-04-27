import os

from tavily import TavilyClient, InvalidAPIKeyError, UsageLimitExceededError
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)


NEWS_WINDOW_DAYS = 3  # framing A: current sentiment, anchored to now


class SearchNewsError(Exception):
    """Raised when Tavily fails after retries are exhausted, or when a
    non-retryable Tavily exception (InvalidAPIKeyError, UsageLimitExceededError)
    is encountered."""
    pass


_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


@retry(
    retry=retry_if_not_exception_type((InvalidAPIKeyError, UsageLimitExceededError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _tavily_search(query: str) -> dict:
    return _client.search(query=query, topic="news", days=NEWS_WINDOW_DAYS)


def search_news(query: str) -> list[dict]:
    """Search Tavily for recent news matching the query.

    Args:
        query: LLM-generated natural-language query string.

    Returns:
        List of dicts. Each dict has exactly the keys {"url", "title", "snippet"}.
        Returns an empty list if Tavily returned zero results.

    Raises:
        SearchNewsError: when Tavily fails after tenacity retries are exhausted,
            or when a non-retryable Tavily exception is encountered.
    """
    try:
        response = _tavily_search(query)
    except Exception as e:
        raise SearchNewsError(str(e)) from e

    return [
        {"url": r["url"], "title": r["title"], "snippet": r["content"]}
        for r in response.get("results", [])
    ]
