import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("TAVILY_API_KEY", "test")

import agent_runner
from agent_runner import (
    MAX_AGENT_STEPS,
    build_work_list,
    process_ticker,
    run_agent,
)
from agent_tools import SearchNewsError


def _make_completion(content: str):
    completion = MagicMock()
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    completion.choices = [choice]
    return completion


def _work_item(ticker: str = "AAPL", signal: str = "BUY") -> dict:
    return {
        "ticker": ticker,
        "model_signal": signal,
        "confidence": 0.71,
        "shap_values": ["RSI_14", "MACD", "sent_rolling_7d"],
    }


def _search_action(query: str = "Apple earnings") -> str:
    return json.dumps({"action": "search_news", "action_input": {"query": query}})


def _decide_action(decision: str = "CONFIRM", reasoning: str = "Evidence for: x. Evidence against: none found. Verdict: CONFIRM.") -> str:
    return json.dumps({
        "action": "decide",
        "action_input": {"agent_decision": decision, "agent_reasoning": reasoning},
    })


def test_build_work_list_filters_to_buy_sell_entry_rows():
    df = pd.DataFrame({
        "date": [
            "2026-04-26",
            "2026-04-26",
            "2026-04-26",
            "2026-04-26",
            "2026-04-26",
            "2026-04-25",
        ],
        "ticker": ["AAPL", "TSLA", "NFLX", "JPM", "GOOG", "MSFT"],
        "row_type": ["ENTRY", "ENTRY", "ENTRY", "EXIT", "ENTRY", "ENTRY"],
        "model_signal": ["BUY", "SELL", "HOLD", "BUY", "BUY", "BUY"],
        "confidence": [0.71, 0.65, 0.40, 0.50, 0.55, 0.60],
        "shap_driver_1": ["a1", "b1", "c1", "d1", "e1", "f1"],
        "shap_driver_2": ["a2", "b2", "c2", "d2", "e2", "f2"],
        "shap_driver_3": ["a3", "b3", "c3", "d3", "e3", "f3"],
    })

    result = build_work_list(df, run_date="2026-04-26")

    assert [d["ticker"] for d in result] == ["AAPL", "TSLA", "GOOG"]
    assert [d["model_signal"] for d in result] == ["BUY", "SELL", "BUY"]
    for d in result:
        assert set(d.keys()) == {"ticker", "model_signal", "confidence", "shap_values"}
        assert isinstance(d["shap_values"], list)
        assert len(d["shap_values"]) == 3


def test_process_ticker_happy_path_search_then_decide():
    completions = [
        _make_completion(_search_action("Apple iPhone services revenue guidance")),
        _make_completion(_decide_action("CONFIRM", "Evidence for: beat. Evidence against: none found. Verdict: CONFIRM.")),
    ]
    fake_results = [
        {"url": "https://a.com", "title": "T1", "snippet": "S1"},
        {"url": "https://b.com", "title": "T2", "snippet": "S2"},
        {"url": "https://c.com", "title": "T3", "snippet": "S3"},
    ]

    with patch("agent_runner._client.chat.completions.create", side_effect=completions), \
         patch("agent_runner.search_news", return_value=fake_results):
        result = process_ticker(_work_item())

    assert result["agent_decision"] == "CONFIRM"
    assert result["error_category"] is None
    assert result["error"] is None
    assert result["news_query"] == "Apple iPhone services revenue guidance"
    assert result["sources_checked"] == fake_results
    assert result["agent_reasoning"] == "Evidence for: beat. Evidence against: none found. Verdict: CONFIRM."


def test_process_ticker_decide_without_search_returns_null_news_fields():
    with patch(
        "agent_runner._client.chat.completions.create",
        return_value=_make_completion(_decide_action("ABSTAIN", "Evidence for: none. Evidence against: none. Verdict: ABSTAIN.")),
    ), patch("agent_runner.search_news") as mock_search:
        result = process_ticker(_work_item())

    assert mock_search.call_count == 0
    assert result["agent_decision"] == "ABSTAIN"
    assert result["news_query"] is None
    assert result["sources_checked"] == []


def test_process_ticker_tool_failure_returns_tool_failure_category():
    with patch(
        "agent_runner._client.chat.completions.create",
        return_value=_make_completion(_search_action("Apple earnings")),
    ), patch("agent_runner.search_news", side_effect=SearchNewsError("tavily down")):
        result = process_ticker(_work_item())

    assert result["error_category"] == "TOOL_FAILURE"
    assert result["agent_decision"] is None
    assert result["error"].startswith("search_news failed: ")
    assert "tavily down" in result["error"]


def test_process_ticker_parse_failure_then_recovery_succeeds():
    completions = [
        _make_completion("not json"),
        _make_completion(_decide_action("ABSTAIN", "Evidence for: none. Evidence against: none. Verdict: ABSTAIN.")),
    ]

    with patch(
        "agent_runner._client.chat.completions.create",
        side_effect=completions,
    ) as mock_create:
        result = process_ticker(_work_item())

    assert result["agent_decision"] == "ABSTAIN"
    assert result["error_category"] is None
    assert mock_create.call_count == 2


def test_process_ticker_parse_exhausted_returns_parse_exhausted_category():
    completions = [
        _make_completion("not json attempt 1"),
        _make_completion("not json attempt 2 either"),
    ]

    with patch("agent_runner._client.chat.completions.create", side_effect=completions):
        result = process_ticker(_work_item())

    assert result["error_category"] == "PARSE_EXHAUSTED"
    assert result["agent_reasoning"] is not None
    assert "not json attempt 2" in result["agent_reasoning"]
    assert result["error"].startswith("parse retry exhausted after 2 attempts: ")


def test_process_ticker_step_cap_returns_step_cap_category():
    with patch(
        "agent_runner._client.chat.completions.create",
        return_value=_make_completion(_search_action("repeat query")),
    ) as mock_create, patch("agent_runner.search_news", return_value=[]):
        result = process_ticker(_work_item())

    assert result["error_category"] == "STEP_CAP"
    assert result["agent_reasoning"] is not None
    assert result["error"] == f"step cap reached after {MAX_AGENT_STEPS} steps without verdict"
    assert mock_create.call_count == MAX_AGENT_STEPS


def test_process_ticker_groq_api_failure_returns_api_failure_category():
    with patch(
        "agent_runner._client.chat.completions.create",
        side_effect=Exception("rate limited"),
    ):
        result = process_ticker(_work_item())

    assert result["error_category"] == "API_FAILURE"
    assert result["agent_decision"] is None
    assert result["error"].startswith("groq API failed: ")
    assert "rate limited" in result["error"]


def test_run_agent_returns_full_artifact_with_one_decision(tmp_path):
    csv_path = tmp_path / "signal_log.csv"
    df = pd.DataFrame({
        "date": ["2026-04-26"],
        "ticker": ["AAPL"],
        "row_type": ["ENTRY"],
        "model_signal": ["BUY"],
        "confidence": [0.71],
        "shap_driver_1": ["RSI_14"],
        "shap_driver_2": ["MACD"],
        "shap_driver_3": ["sent_rolling_7d"],
    })
    df.to_csv(str(csv_path), index=False)

    completions = [
        _make_completion(_search_action("Apple earnings")),
        _make_completion(_decide_action("CONFIRM", "Evidence for: x. Evidence against: none found. Verdict: CONFIRM.")),
    ]
    fake_results = [{"url": "https://a.com", "title": "T", "snippet": "S"}]

    fake_git = MagicMock()
    fake_git.stdout = "abc1234\n"
    fake_git.returncode = 0

    with patch("agent_runner._client.chat.completions.create", side_effect=completions), \
         patch("agent_runner.search_news", return_value=fake_results), \
         patch("agent_runner.subprocess.run", return_value=fake_git):
        result = run_agent(signal_log_path=str(csv_path), run_date="2026-04-26")

    assert result["run_date"] == "2026-04-26"
    assert result["agent_version"] == "abc1234"
    assert result["llm_model"] == agent_runner.GROQ_MODEL
    assert "run_started_at" in result and isinstance(result["run_started_at"], str)
    assert isinstance(result["run_duration_seconds"], float)
    assert result["run_duration_seconds"] >= 0
    assert len(result["decisions"]) == 1
    assert result["run_error"] is None
