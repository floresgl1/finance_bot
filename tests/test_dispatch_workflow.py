"""Tests for dispatch_workflow.py, which PythonAnywhere runs to start the
pre-trade GitHub workflows on time. It has to fail loudly, never silently:
a workflow that is not started means the 15:00 trade runs on stale data."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import requests

import dispatch_workflow as dw

MONDAY = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)
SATURDAY = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def posts(monkeypatch):
    """Capture requests.post calls; responses are queued per test."""
    calls, queue = [], []

    def fake_post(url, **kw):
        calls.append((url, kw))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(status_code=item, text=f"body {item}",
                               raise_for_status=lambda: None)

    monkeypatch.setattr(dw.requests, "post", fake_post)
    monkeypatch.setenv("GITHUB_DISPATCH_TOKEN", "t0k")
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    return calls, queue


def test_dispatch_posts_the_right_request(posts):
    calls, queue = posts
    queue.append(204)
    assert dw.main(["update_market_data.yml"], now=MONDAY) == 0
    url, kw = calls[0]
    assert url == ("https://api.github.com/repos/floresgl1/finance_bot/"
                   "actions/workflows/update_market_data.yml/dispatches")
    assert kw["json"] == {"ref": "main"}
    assert kw["headers"]["Authorization"] == "Bearer t0k"


def test_weekends_are_skipped_without_calling_github(posts):
    calls, _ = posts
    assert dw.main(["agent_pretrade.yml"], now=SATURDAY) == 0
    assert calls == []


def test_missing_token_fails_loudly(posts, monkeypatch):
    monkeypatch.delenv("GITHUB_DISPATCH_TOKEN")
    calls, _ = posts
    assert dw.main(["agent_pretrade.yml"], now=MONDAY) == 1
    assert calls == []


def test_a_client_error_is_final_and_not_retried(posts):
    calls, queue = posts
    queue.append(403)
    ok, detail = dw.dispatch("agent_pretrade.yml", "t0k", sleep=lambda s: None)
    assert not ok and "HTTP 403" in detail
    assert len(calls) == 1


def test_server_errors_and_network_errors_are_retried(posts):
    calls, queue = posts
    queue.extend([502, requests.ConnectionError("down"), 204])
    ok, _ = dw.dispatch("agent_pretrade.yml", "t0k", sleep=lambda s: None)
    assert ok and len(calls) == 3


def test_giving_up_after_retries_exits_1_and_alerts(posts, monkeypatch):
    calls, queue = posts
    monkeypatch.setattr(dw, "BACKOFF_SECONDS", 0)
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/hook")
    queue.extend([503, 503, 503, 204])       # last one answers the Discord post
    assert dw.main(["agent_pretrade.yml"], now=MONDAY) == 1
    assert calls[-1][0] == "https://discord.example/hook"
    assert "agent_pretrade.yml" in calls[-1][1]["json"]["content"]


def test_usage_error_without_a_workflow_argument():
    assert dw.main([], now=MONDAY) == 1


def test_setup_script_targets_the_two_pre_trade_workflows_at_the_right_times():
    import create_pa_dispatch_tasks as setup
    assert setup.TASKS == [("update_market_data.yml", 12, 0), ("agent_pretrade.yml", 14, 15)]
    assert setup._command("bob", "agent_pretrade.yml") == \
        "cd /home/bob/finance_bot && python dispatch_workflow.py agent_pretrade.yml"
    assert setup._find([{"id": 7, "command": setup._command("bob", "agent_pretrade.yml")}],
                       "agent_pretrade.yml")["id"] == 7
    assert setup._find([{"id": 7, "command": "python sentiment_collector.py"}],
                       "agent_pretrade.yml") is None


@pytest.mark.parametrize("wf", ["update_market_data.yml", "agent_pretrade.yml"])
def test_dispatched_workflows_have_no_github_schedule_left(wf):
    yaml = pytest.importorskip("yaml")
    with open(f".github/workflows/{wf}") as f:
        on = yaml.safe_load(f)[True]
    assert "schedule" not in on and "workflow_dispatch" in on
