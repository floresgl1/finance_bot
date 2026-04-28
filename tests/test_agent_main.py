import json
import os
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("TAVILY_API_KEY", "test")

import agent_main


# --- path helpers ----------------------------------------------------------


def test_run_path_for():
    assert agent_main.run_path_for("2026-04-26") == Path("data/agent_runs/2026-04-26.json")


def test_compact_utc_timestamp_format():
    ts = agent_main.compact_utc_timestamp()
    assert re.match(r"^\d{8}T\d{6}Z$", ts), f"unexpected timestamp format: {ts!r}"


# --- handle_existing_run_file ----------------------------------------------


def test_handle_existing_run_file_no_existing(tmp_path):
    target = tmp_path / "2026-04-26.json"
    agent_main.handle_existing_run_file(target)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_handle_existing_run_file_renames_valid_json(tmp_path):
    target = tmp_path / "2026-04-26.json"
    original = {"run_date": "2026-04-26", "decisions": []}
    target.write_text(json.dumps(original))

    agent_main.handle_existing_run_file(target)

    assert not target.exists()
    backups = list(tmp_path.glob("2026-04-26.*.bak.json"))
    assert len(backups) == 1
    assert re.match(r"^2026-04-26\.\d{8}T\d{6}Z\.bak\.json$", backups[0].name)
    assert json.loads(backups[0].read_text()) == original


def test_handle_existing_run_file_halts_on_unparseable(tmp_path):
    target = tmp_path / "2026-04-26.json"
    garbage = "not json {{{"
    target.write_text(garbage)

    with pytest.raises(RuntimeError) as excinfo:
        agent_main.handle_existing_run_file(target)

    assert "unparseable" in str(excinfo.value)
    assert target.exists()
    assert target.read_text() == garbage
    assert list(tmp_path.glob("*.bak.json")) == []


# --- write_run_file --------------------------------------------------------


def test_write_run_file_atomic_writes_correct_content(tmp_path):
    target = tmp_path / "out.json"
    envelope = {"run_date": "2026-04-26", "decisions": [{"ticker": "AAPL"}]}

    agent_main.write_run_file(target, envelope)

    assert target.exists()
    assert json.loads(target.read_text()) == envelope
    assert list(tmp_path.glob("*.tmp")) == []


# --- count_decisions -------------------------------------------------------


def test_count_decisions_all_categories():
    decisions = [
        {"agent_decision": "CONFIRM", "error": None},
        {"agent_decision": "VETO", "error": None},
        {"agent_decision": "ABSTAIN", "error": None},
        {"agent_decision": None, "error": "TOOL_FAILURE: tavily down"},
        {"agent_decision": None, "error": None},
    ]

    counts = agent_main._count_decisions_by_outcome(decisions)

    assert counts == {"CONFIRM": 1, "VETO": 1, "ABSTAIN": 2, "ERROR": 1}
    assert sum(counts.values()) == len(decisions)


def test_count_decisions_empty_list():
    counts = agent_main._count_decisions_by_outcome([])
    assert counts == {"CONFIRM": 0, "VETO": 0, "ABSTAIN": 0, "ERROR": 0}


# --- send_discord_summary --------------------------------------------------


def test_send_discord_summary_skips_when_env_unset(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    mock_post = MagicMock()
    monkeypatch.setattr(agent_main.requests, "post", mock_post)

    agent_main.send_discord_summary({"run_date": "2026-04-26", "decisions": [], "run_error": None})

    mock_post.assert_not_called()


def test_send_discord_summary_swallows_post_failure(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.invalid/webhook")
    monkeypatch.setattr(
        agent_main.requests, "post",
        MagicMock(side_effect=RuntimeError("network down")),
    )

    agent_main.send_discord_summary({"run_date": "2026-04-26", "decisions": [], "run_error": None})


def test_send_discord_summary_success_message_format(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.invalid/webhook")
    mock_post = MagicMock()
    monkeypatch.setattr(agent_main.requests, "post", mock_post)

    envelope = {
        "run_date": "2026-04-26",
        "decisions": [
            {"agent_decision": "CONFIRM", "error": None},
            {"agent_decision": "VETO", "error": None},
            {"agent_decision": "ABSTAIN", "error": None},
        ],
        "run_error": None,
    }

    agent_main.send_discord_summary(envelope)

    assert mock_post.call_count == 1
    posted_json = mock_post.call_args.kwargs["json"]
    content = posted_json["content"]
    assert "\U0001F916 Agent run complete" in content
    assert "CONFIRM 1, VETO 1, ABSTAIN 1, ERROR 0" in content
    assert "2026-04-26" in content


def test_send_discord_summary_failure_message_format(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.invalid/webhook")
    mock_post = MagicMock()
    monkeypatch.setattr(agent_main.requests, "post", mock_post)

    envelope = {
        "run_date": "2026-04-26",
        "decisions": [],
        "run_error": "workspace setup failed: FileNotFoundError: data/signal_log.csv",
    }

    agent_main.send_discord_summary(envelope)

    assert mock_post.call_count == 1
    content = mock_post.call_args.kwargs["json"]["content"]
    assert "\U0001F6A8 Agent run FAILED" in content
    assert "workspace setup failed" in content


# --- main() ----------------------------------------------------------------


def test_main_happy_path(monkeypatch, tmp_path):
    target = tmp_path / "2026-04-26.json"
    fake_envelope = {
        "run_date": "2026-04-26",
        "agent_version": "abc1234",
        "llm_model": "test-model",
        "run_started_at": "2026-04-26T00:00:00Z",
        "run_duration_seconds": 1.5,
        "decisions": [],
        "run_error": None,
    }

    monkeypatch.setattr(agent_main, "run_agent", lambda **kwargs: fake_envelope)
    monkeypatch.setattr(agent_main, "run_path_for", lambda run_date: target)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

    rc = agent_main.main()

    assert rc == 0
    assert target.exists()
    assert json.loads(target.read_text()) == fake_envelope


def test_main_returns_one_on_run_error(monkeypatch, tmp_path):
    target = tmp_path / "2026-04-26.json"
    fake_envelope = {
        "run_date": "2026-04-26",
        "agent_version": "unknown",
        "llm_model": "test-model",
        "run_started_at": "2026-04-26T00:00:00Z",
        "run_duration_seconds": 0.1,
        "decisions": [],
        "run_error": "workspace setup failed: FileNotFoundError: data/signal_log.csv",
    }

    monkeypatch.setattr(agent_main, "run_agent", lambda **kwargs: fake_envelope)
    monkeypatch.setattr(agent_main, "run_path_for", lambda run_date: target)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

    rc = agent_main.main()

    assert rc == 1
    assert target.exists()
    assert json.loads(target.read_text()) == fake_envelope


def test_main_handles_unparseable_existing_file(monkeypatch, tmp_path):
    target = tmp_path / "2026-04-26.json"
    target.write_text("not json {{{")

    sentinel_run_agent = MagicMock()
    monkeypatch.setattr(agent_main, "run_agent", sentinel_run_agent)
    monkeypatch.setattr(agent_main, "run_path_for", lambda run_date: target)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

    rc = agent_main.main()

    assert rc == 1
    sentinel_run_agent.assert_not_called()
    loaded = json.loads(target.read_text())
    assert loaded["run_error"] is not None
    assert "unparseable" in loaded["run_error"]
    assert list(tmp_path.glob("*.bak.json")) == []
