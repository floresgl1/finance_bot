import os
from unittest.mock import patch

import pytest

os.environ.setdefault("TAVILY_API_KEY", "test")

from agent_tools import search_news, SearchNewsError
from tavily import InvalidAPIKeyError


def test_search_news_happy_path_returns_three_dicts_with_correct_keys():
    fake_response = {
        "query": "test",
        "results": [
            {"title": "T1", "url": "https://a.com", "content": "snippet 1", "score": 0.9},
            {"title": "T2", "url": "https://b.com", "content": "snippet 2", "score": 0.8},
            {"title": "T3", "url": "https://c.com", "content": "snippet 3", "score": 0.7},
        ],
    }
    with patch("agent_tools._client.search", return_value=fake_response):
        result = search_news("anything")

    assert isinstance(result, list)
    assert len(result) == 3
    for item in result:
        assert set(item.keys()) == {"url", "title", "snippet"}
    assert result[0]["snippet"] == "snippet 1"


def test_search_news_empty_results_returns_empty_list():
    with patch("agent_tools._client.search", return_value={"query": "test", "results": []}):
        result = search_news("anything")

    assert result == []


def test_search_news_strips_score_and_other_fields():
    fake_response = {
        "results": [
            {
                "title": "T",
                "url": "https://a.com",
                "content": "s",
                "score": 0.99,
                "raw_content": "...",
                "published_date": "2026-04-25",
            }
        ]
    }
    with patch("agent_tools._client.search", return_value=fake_response):
        result = search_news("anything")

    assert len(result) == 1
    assert set(result[0].keys()) == {"url", "title", "snippet"}


def test_search_news_raises_search_news_error_on_retryable_failure():
    with patch(
        "agent_tools._client.search",
        side_effect=Exception("network down"),
    ) as mock_search:
        with pytest.raises(SearchNewsError):
            search_news("anything")

    assert mock_search.call_count == 3


def test_search_news_raises_immediately_on_invalid_api_key():
    with patch(
        "agent_tools._client.search",
        side_effect=InvalidAPIKeyError("bad key"),
    ) as mock_search:
        with pytest.raises(SearchNewsError):
            search_news("anything")

    assert mock_search.call_count == 1
