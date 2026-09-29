"""Tests for pipeline_check.py, the daily alarm. It must catch each of the
September 2026 silent failures by the output it left behind, not its exit code."""

import json
import os
from datetime import date, datetime, timezone

import pytest

import pipeline_check as pc

TODAY = date(2026, 9, 30)
TODAY_TS = datetime(2026, 9, 30, 12, 5, tzinfo=timezone.utc).timestamp()
YESTERDAY_TS = datetime(2026, 9, 29, 12, 5, tzinfo=timezone.utc).timestamp()


def _csv(path, ts):
    path.write_text("Date,Close\n")
    os.utime(path, (ts, ts))


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "market").mkdir()
    for t in ("AAPL", "MSFT"):
        _csv(tmp_path / f"{t}.csv", TODAY_TS)
    _csv(tmp_path / "market" / "SPY.csv", TODAY_TS)
    return tmp_path


# --- market data ------------------------------------------------------------------
def test_fresh_csvs_pass(data_dir):
    ok, detail = pc.check_market_data(TODAY, str(data_dir), ["AAPL", "MSFT"])
    assert ok and detail == "3 CSVs refreshed today"


def test_a_missed_refresh_fails_like_the_expired_token_did(data_dir):
    _csv(data_dir / "MSFT.csv", YESTERDAY_TS)
    _csv(data_dir / "market" / "SPY.csv", YESTERDAY_TS)
    ok, detail = pc.check_market_data(TODAY, str(data_dir), ["AAPL", "MSFT"])
    assert not ok and "2 of 3" in detail and "MSFT.csv" in detail


def test_a_missing_watchlist_csv_fails(data_dir):
    ok, _ = pc.check_market_data(TODAY, str(data_dir), ["AAPL", "MSFT", "NVDA"])
    assert not ok


# --- dated JSON outputs -------------------------------------------------------------
def test_todays_json_passes(tmp_path):
    p = tmp_path / "pending_signals.json"
    p.write_text(json.dumps({"date": "2026-09-30", "signals": []}))
    assert pc.check_dated_json(str(p), TODAY) == (True, "pending_signals.json is for today")


def test_a_stale_json_fails_like_the_broken_signals_task_did(tmp_path):
    p = tmp_path / "agent_decisions.json"
    p.write_text(json.dumps({"date": "2026-09-25", "decisions": {}}))
    ok, detail = pc.check_dated_json(str(p), TODAY)
    assert not ok and detail == "agent_decisions.json is for 2026-09-25, not 2026-09-30"


@pytest.mark.parametrize("content", [None, "not json", "[1, 2]"])
def test_missing_or_malformed_json_fails(tmp_path, content):
    p = tmp_path / "pending_signals.json"
    if content is not None:
        p.write_text(content)
    ok, _ = pc.check_dated_json(str(p), TODAY)
    assert not ok


# --- trade -----------------------------------------------------------------------------
def test_guard_stamped_today_passes(tmp_path):
    g = tmp_path / "last_run_date.txt"
    g.write_text("2026-09-30\n")
    assert pc.check_trade(TODAY, str(g), str(tmp_path / "HALT_FLAG.txt"))[0]


def test_an_old_guard_fails(tmp_path):
    g = tmp_path / "last_run_date.txt"
    g.write_text("2026-09-29")
    ok, detail = pc.check_trade(TODAY, str(g), str(tmp_path / "HALT_FLAG.txt"))
    assert not ok and "2026-09-29" in detail


def test_a_halt_flag_fails_even_with_todays_guard(tmp_path):
    g = tmp_path / "last_run_date.txt"
    g.write_text("2026-09-30")
    h = tmp_path / "HALT_FLAG.txt"
    h.write_text("PORTFOLIO_HALT_SINGLE_DAY 2026-09-30T15:00Z\n")
    ok, detail = pc.check_trade(TODAY, str(g), str(h))
    assert not ok and "PORTFOLIO_HALT_SINGLE_DAY" in detail


# --- report and exit code -------------------------------------------------------------
def test_all_green_report_is_one_line():
    msg = pc.format_report(TODAY, [("a", True, "x"), ("b", True, "y")])
    assert msg.startswith("✅") and "\n" not in msg


def test_failure_report_lists_every_step():
    msg = pc.format_report(TODAY, [("signals", False, "stale"), ("trade", True, "ok")])
    assert msg.startswith("🚨") and "❌ signals: stale" in msg and "✅ trade: ok" in msg


def test_main_exits_1_and_alerts_on_any_failure(monkeypatch):
    monkeypatch.setattr(pc, "is_trading_day", lambda d: True)
    monkeypatch.setattr(pc, "run_checks", lambda d: [("x", True, "ok"), ("y", False, "bad")])
    sent = []
    monkeypatch.setattr(pc, "send_discord", sent.append)
    assert pc.main(TODAY) == 1
    assert sent and sent[0].startswith("🚨")


def test_main_posts_green_daily_so_silence_means_the_check_did_not_run(monkeypatch):
    monkeypatch.setattr(pc, "is_trading_day", lambda d: True)
    monkeypatch.setattr(pc, "run_checks", lambda d: [("x", True, "ok")])
    sent = []
    monkeypatch.setattr(pc, "send_discord", sent.append)
    assert pc.main(TODAY) == 0
    assert sent and sent[0].startswith("✅")


def test_non_trading_days_are_skipped_silently(monkeypatch):
    monkeypatch.setattr(pc, "run_checks", lambda d: pytest.fail("must not run"))
    sent = []
    monkeypatch.setattr(pc, "send_discord", sent.append)
    assert pc.main(date(2026, 10, 3)) == 0        # a Saturday
    assert sent == []
