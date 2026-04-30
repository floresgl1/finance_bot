"""Tests for edge_monitor.py — rolling-edge degradation alarm."""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import config
import edge_monitor


# --- helpers ---------------------------------------------------------------


SIGNAL_LOG_COLUMNS = [
    "date",
    "ticker",
    "model_signal",
    "price",
    "qty",
    "confidence",
    "evaluation_date",
    "actual_action",
    "outcome_price",
    "result",
    "shap_driver_1",
    "shap_driver_2",
    "shap_driver_3",
    "row_type",
    "entry_order_id",
    "exit_timestamp",
    "exit_price",
    "exit_reason",
    "shares",
    "realized_pnl",
]


def _row(
    *,
    date: str,
    model_signal: str = "BUY",
    result: str = "WIN",
    actual_action: str = "BUY",
    row_type: str = "ENTRY",
    ticker: str = "AAPL",
) -> dict:
    return {
        "date": date,
        "ticker": ticker,
        "model_signal": model_signal,
        "price": 100.0,
        "qty": 1,
        "confidence": 0.7,
        "evaluation_date": "",
        "actual_action": actual_action,
        "outcome_price": 102.0,
        "result": result,
        "shap_driver_1": "",
        "shap_driver_2": "",
        "shap_driver_3": "",
        "row_type": row_type,
        "entry_order_id": "",
        "exit_timestamp": "",
        "exit_price": "",
        "exit_reason": "",
        "shares": "",
        "realized_pnl": "",
    }


def _write_signal_log(path: Path, rows: list[dict]) -> str:
    df = pd.DataFrame(rows, columns=SIGNAL_LOG_COLUMNS)
    df.to_csv(str(path), index=False)
    return str(path)


def _make_buys(
    n: int,
    *,
    wins: int,
    end_date: str,
    actual_action: str = "BUY",
    spacing_days: int = 1,
) -> list[dict]:
    """Generate n evaluated BUY rows ending at end_date.

    `wins` rows have result=WIN, the remaining n-wins have result=LOSS.
    Dates step backward from end_date by `spacing_days`.
    """
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    rows = []
    for i in range(n):
        row_date = end - timedelta(days=(n - 1 - i) * spacing_days)
        result = "WIN" if i < wins else "LOSS"
        rows.append(
            _row(
                date=row_date.isoformat(),
                model_signal="BUY",
                result=result,
                actual_action=actual_action,
                row_type="ENTRY",
            )
        )
    return rows


@pytest.fixture
def patched_paths(tmp_path, monkeypatch):
    """Redirect EDGE_MONITOR_STATE_PATH into tmp_path and freeze 'today'."""
    state_path = tmp_path / "edge_monitor_state.json"
    monkeypatch.setattr(config, "EDGE_MONITOR_STATE_PATH", str(state_path))

    # Default frozen "today" — individual tests can override via the fixture's tools
    fake_today = datetime(2026, 4, 28, 15, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(edge_monitor, "_today_utc", lambda: fake_today)

    return {
        "tmp": tmp_path,
        "state_path": state_path,
        "today": fake_today,
        "monkeypatch": monkeypatch,
    }


@pytest.fixture
def mock_post(monkeypatch):
    posted = []

    def fake_post(url, json=None, timeout=None, **kwargs):
        posted.append({"url": url, "json": json, "timeout": timeout})
        resp = MagicMock()
        resp.status_code = 204
        resp.raise_for_status = MagicMock()
        return resp

    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.invalid/webhook")
    monkeypatch.setattr(edge_monitor.requests, "post", fake_post)
    return posted


# --- tests -----------------------------------------------------------------


def test_insufficient_data_first_run(patched_paths, mock_post):
    rows = _make_buys(10, wins=4, end_date="2026-04-26")
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    edge_monitor.run_edge_monitor(signal_log_path=csv)

    state = json.loads(patched_paths["state_path"].read_text())
    assert state["last_state"] == "INSUFFICIENT_DATA"
    assert state["last_meaningful_state"] is None
    assert state["last_hit_rate"] is None
    assert state["last_window_size"] == 10
    assert mock_post == []


def test_first_run_above_is_silent(patched_paths, mock_post):
    rows = _make_buys(30, wins=12, end_date="2026-04-26")  # 40%
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    edge_monitor.run_edge_monitor(signal_log_path=csv)

    state = json.loads(patched_paths["state_path"].read_text())
    assert state["last_state"] == "ABOVE_THRESHOLD"
    assert state["last_meaningful_state"] == "ABOVE_THRESHOLD"
    assert state["last_hit_rate"] == pytest.approx(0.4)
    assert state["last_window_size"] == 30
    assert mock_post == []


def test_first_run_below_alerts(patched_paths, mock_post):
    rows = _make_buys(30, wins=7, end_date="2026-04-26")  # 23.3% < 31%
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    edge_monitor.run_edge_monitor(signal_log_path=csv)

    state = json.loads(patched_paths["state_path"].read_text())
    assert state["last_state"] == "BELOW_THRESHOLD"
    assert state["last_meaningful_state"] == "BELOW_THRESHOLD"
    assert state["last_hit_rate"] == pytest.approx(7 / 30)
    assert len(mock_post) == 1
    content = mock_post[0]["json"]["content"]
    assert "EDGE DEGRADATION ALERT" in content
    assert "23.3%" in content
    assert "below 31% break-even" in content
    assert "Baseline (2026-04-19): 36.6% hit rate, +1.86% mean return" in content


def test_first_run_stale_alerts(patched_paths, mock_post):
    # most recent eval is 12 days before frozen today (2026-04-28) → stale
    rows = _make_buys(30, wins=15, end_date="2026-04-16")
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    edge_monitor.run_edge_monitor(signal_log_path=csv)

    state = json.loads(patched_paths["state_path"].read_text())
    assert state["last_state"] == "STALE"
    assert state["last_meaningful_state"] is None
    assert state["last_hit_rate"] is None
    assert len(mock_post) == 1
    content = mock_post[0]["json"]["content"]
    assert "STALE EVALUATIONS" in content
    assert "2026-04-16" in content
    assert "12 days ago" in content


def test_transition_above_to_below(patched_paths, mock_post):
    patched_paths["state_path"].write_text(json.dumps({
        "last_run_utc": "2026-04-21T15:00:00Z",
        "last_state": "ABOVE_THRESHOLD",
        "last_meaningful_state": "ABOVE_THRESHOLD",
        "last_hit_rate": 0.4,
        "last_window_size": 30,
    }))
    rows = _make_buys(30, wins=8, end_date="2026-04-26")  # 26.7%
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    edge_monitor.run_edge_monitor(signal_log_path=csv)

    state = json.loads(patched_paths["state_path"].read_text())
    assert state["last_state"] == "BELOW_THRESHOLD"
    assert state["last_meaningful_state"] == "BELOW_THRESHOLD"
    assert len(mock_post) == 1
    assert "EDGE DEGRADATION ALERT" in mock_post[0]["json"]["content"]


def test_transition_below_to_above(patched_paths, mock_post):
    patched_paths["state_path"].write_text(json.dumps({
        "last_run_utc": "2026-04-21T15:00:00Z",
        "last_state": "BELOW_THRESHOLD",
        "last_meaningful_state": "BELOW_THRESHOLD",
        "last_hit_rate": 0.2,
        "last_window_size": 30,
    }))
    rows = _make_buys(30, wins=12, end_date="2026-04-26")  # 40%
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    edge_monitor.run_edge_monitor(signal_log_path=csv)

    state = json.loads(patched_paths["state_path"].read_text())
    assert state["last_state"] == "ABOVE_THRESHOLD"
    assert state["last_meaningful_state"] == "ABOVE_THRESHOLD"
    assert len(mock_post) == 1
    assert "EDGE RECOVERY" in mock_post[0]["json"]["content"]


def test_stay_below_no_alert(patched_paths, mock_post):
    patched_paths["state_path"].write_text(json.dumps({
        "last_run_utc": "2026-04-21T15:00:00Z",
        "last_state": "BELOW_THRESHOLD",
        "last_meaningful_state": "BELOW_THRESHOLD",
        "last_hit_rate": 0.2,
        "last_window_size": 30,
    }))
    rows = _make_buys(30, wins=8, end_date="2026-04-26")  # 26.7%
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    edge_monitor.run_edge_monitor(signal_log_path=csv)

    state = json.loads(patched_paths["state_path"].read_text())
    assert state["last_state"] == "BELOW_THRESHOLD"
    assert state["last_hit_rate"] == pytest.approx(8 / 30)
    assert state["last_run_utc"] == "2026-04-28T15:00:00Z"
    assert mock_post == []


def test_stay_above_no_alert(patched_paths, mock_post):
    patched_paths["state_path"].write_text(json.dumps({
        "last_run_utc": "2026-04-21T15:00:00Z",
        "last_state": "ABOVE_THRESHOLD",
        "last_meaningful_state": "ABOVE_THRESHOLD",
        "last_hit_rate": 0.4,
        "last_window_size": 30,
    }))
    rows = _make_buys(30, wins=15, end_date="2026-04-26")  # 50%
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    edge_monitor.run_edge_monitor(signal_log_path=csv)

    state = json.loads(patched_paths["state_path"].read_text())
    assert state["last_state"] == "ABOVE_THRESHOLD"
    assert mock_post == []


def test_below_to_insufficient_carries_meaningful(patched_paths, mock_post):
    patched_paths["state_path"].write_text(json.dumps({
        "last_run_utc": "2026-04-21T15:00:00Z",
        "last_state": "BELOW_THRESHOLD",
        "last_meaningful_state": "BELOW_THRESHOLD",
        "last_hit_rate": 0.2,
        "last_window_size": 30,
    }))
    rows = _make_buys(10, wins=4, end_date="2026-04-26")  # < 30
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    edge_monitor.run_edge_monitor(signal_log_path=csv)

    state = json.loads(patched_paths["state_path"].read_text())
    assert state["last_state"] == "INSUFFICIENT_DATA"
    assert state["last_meaningful_state"] == "BELOW_THRESHOLD"
    assert state["last_hit_rate"] is None
    assert mock_post == []


def test_insufficient_to_above_with_prior_below_alerts(patched_paths, mock_post):
    patched_paths["state_path"].write_text(json.dumps({
        "last_run_utc": "2026-04-14T15:00:00Z",
        "last_state": "INSUFFICIENT_DATA",
        "last_meaningful_state": "BELOW_THRESHOLD",
        "last_hit_rate": None,
        "last_window_size": 12,
    }))
    rows = _make_buys(30, wins=15, end_date="2026-04-26")  # 50%
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    edge_monitor.run_edge_monitor(signal_log_path=csv)

    state = json.loads(patched_paths["state_path"].read_text())
    assert state["last_state"] == "ABOVE_THRESHOLD"
    assert len(mock_post) == 1
    assert "EDGE RECOVERY" in mock_post[0]["json"]["content"]


def test_insufficient_to_below_with_prior_below_no_alert(patched_paths, mock_post):
    patched_paths["state_path"].write_text(json.dumps({
        "last_run_utc": "2026-04-14T15:00:00Z",
        "last_state": "INSUFFICIENT_DATA",
        "last_meaningful_state": "BELOW_THRESHOLD",
        "last_hit_rate": None,
        "last_window_size": 12,
    }))
    rows = _make_buys(30, wins=8, end_date="2026-04-26")  # 26.7%
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    edge_monitor.run_edge_monitor(signal_log_path=csv)

    state = json.loads(patched_paths["state_path"].read_text())
    assert state["last_state"] == "BELOW_THRESHOLD"
    assert mock_post == []


def test_above_to_stale_alerts(patched_paths, mock_post):
    patched_paths["state_path"].write_text(json.dumps({
        "last_run_utc": "2026-04-14T15:00:00Z",
        "last_state": "ABOVE_THRESHOLD",
        "last_meaningful_state": "ABOVE_THRESHOLD",
        "last_hit_rate": 0.4,
        "last_window_size": 30,
    }))
    rows = _make_buys(30, wins=15, end_date="2026-04-15")  # 13 days old
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    edge_monitor.run_edge_monitor(signal_log_path=csv)

    state = json.loads(patched_paths["state_path"].read_text())
    assert state["last_state"] == "STALE"
    assert state["last_meaningful_state"] == "ABOVE_THRESHOLD"
    assert len(mock_post) == 1
    assert "STALE EVALUATIONS" in mock_post[0]["json"]["content"]


def test_stay_stale_no_alert(patched_paths, mock_post):
    patched_paths["state_path"].write_text(json.dumps({
        "last_run_utc": "2026-04-21T15:00:00Z",
        "last_state": "STALE",
        "last_meaningful_state": "ABOVE_THRESHOLD",
        "last_hit_rate": None,
        "last_window_size": 30,
    }))
    rows = _make_buys(30, wins=15, end_date="2026-04-15")
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    edge_monitor.run_edge_monitor(signal_log_path=csv)

    state = json.loads(patched_paths["state_path"].read_text())
    assert state["last_state"] == "STALE"
    assert mock_post == []


def test_stale_to_above_recovery_when_prior_meaningful_below(patched_paths, mock_post):
    patched_paths["state_path"].write_text(json.dumps({
        "last_run_utc": "2026-04-21T15:00:00Z",
        "last_state": "STALE",
        "last_meaningful_state": "BELOW_THRESHOLD",
        "last_hit_rate": None,
        "last_window_size": 30,
    }))
    rows = _make_buys(30, wins=15, end_date="2026-04-26")
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    edge_monitor.run_edge_monitor(signal_log_path=csv)

    state = json.loads(patched_paths["state_path"].read_text())
    assert state["last_state"] == "ABOVE_THRESHOLD"
    assert state["last_meaningful_state"] == "ABOVE_THRESHOLD"
    assert len(mock_post) == 1
    assert "EDGE RECOVERY" in mock_post[0]["json"]["content"]


def test_vetoed_buys_count_when_evaluated(patched_paths, mock_post):
    rows = _make_buys(29, wins=10, end_date="2026-04-25", actual_action="BUY")
    rows.append(
        _row(
            date="2026-04-26",
            model_signal="BUY",
            result="WIN",
            actual_action="EARNINGS_VETO",
            row_type="ENTRY",
        )
    )
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    edge_monitor.run_edge_monitor(signal_log_path=csv)

    state = json.loads(patched_paths["state_path"].read_text())
    assert state["last_window_size"] == 30
    assert state["last_hit_rate"] == pytest.approx(11 / 30)


def test_skipped_buys_with_blank_outcome_excluded(patched_paths, mock_post):
    rows = _make_buys(30, wins=12, end_date="2026-04-26")
    rows.extend([
        _row(
            date="2026-04-26",
            model_signal="BUY",
            result="",
            actual_action="CONFIDENCE_SKIP",
        ),
        _row(
            date="2026-04-26",
            model_signal="BUY",
            result="",
            actual_action="INSUFFICIENT_EQUITY",
        ),
        _row(
            date="2026-04-26",
            model_signal="BUY",
            result="NEUTRAL",
            actual_action="BUY",
        ),
    ])
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    edge_monitor.run_edge_monitor(signal_log_path=csv)

    state = json.loads(patched_paths["state_path"].read_text())
    assert state["last_window_size"] == 30
    assert state["last_hit_rate"] == pytest.approx(12 / 30)


def test_window_uses_only_most_recent_30(patched_paths, mock_post):
    # 50 evaluated BUYs spanning Feb–Apr; first 20 dates are stale wins,
    # most-recent 30 are 8 wins → 26.7% (BELOW)
    older = _make_buys(20, wins=20, end_date="2026-03-15")  # all wins, far past
    recent = _make_buys(30, wins=8, end_date="2026-04-26")
    rows = older + recent
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    edge_monitor.run_edge_monitor(signal_log_path=csv)

    state = json.loads(patched_paths["state_path"].read_text())
    assert state["last_state"] == "BELOW_THRESHOLD"
    assert state["last_hit_rate"] == pytest.approx(8 / 30)


def test_atomic_write_uses_tmp_and_replace(patched_paths, mock_post, monkeypatch):
    rows = _make_buys(30, wins=12, end_date="2026-04-26")
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    real_replace = os.replace
    seen = {"tmp_was_present": False, "replace_called_with": None}

    def spy_replace(src, dst):
        # tmp file should exist on disk before replace is invoked
        seen["tmp_was_present"] = os.path.exists(src) and src.endswith(".tmp")
        seen["replace_called_with"] = (src, dst)
        return real_replace(src, dst)

    monkeypatch.setattr(edge_monitor.os, "replace", spy_replace)

    edge_monitor.run_edge_monitor(signal_log_path=csv)

    assert seen["tmp_was_present"] is True
    src, dst = seen["replace_called_with"]
    assert src == str(patched_paths["state_path"]) + ".tmp"
    assert dst == str(patched_paths["state_path"])


def test_malformed_state_file_raises(patched_paths, mock_post):
    patched_paths["state_path"].write_text("not json {{{")
    rows = _make_buys(30, wins=12, end_date="2026-04-26")
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    with pytest.raises(Exception):
        edge_monitor.run_edge_monitor(signal_log_path=csv)

    # State file unchanged
    assert patched_paths["state_path"].read_text() == "not json {{{"


def test_missing_webhook_url_logs_warning_and_updates_state(patched_paths, monkeypatch, caplog):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    rows = _make_buys(30, wins=7, end_date="2026-04-26")  # would alert
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    with caplog.at_level("WARNING", logger="edge_monitor"):
        edge_monitor.run_edge_monitor(signal_log_path=csv)

    assert any("DISCORD_WEBHOOK_URL not set" in r.message for r in caplog.records)
    assert patched_paths["state_path"].exists()
    state = json.loads(patched_paths["state_path"].read_text())
    assert state["last_state"] == "BELOW_THRESHOLD"


def test_webhook_http_error_logged_and_state_updated(patched_paths, monkeypatch, caplog):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.invalid/webhook")

    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(edge_monitor.requests, "post", boom)

    rows = _make_buys(30, wins=7, end_date="2026-04-26")
    csv = _write_signal_log(patched_paths["tmp"] / "signal_log.csv", rows)

    with caplog.at_level("WARNING", logger="edge_monitor"):
        edge_monitor.run_edge_monitor(signal_log_path=csv)

    assert any("Discord post failed" in r.message for r in caplog.records)
    assert patched_paths["state_path"].exists()
    state = json.loads(patched_paths["state_path"].read_text())
    assert state["last_state"] == "BELOW_THRESHOLD"
