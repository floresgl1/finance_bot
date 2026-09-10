"""Tests for agent_posttrade.py — the weekly post-trade analysis agent."""

import csv
import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

os.environ.setdefault("GROQ_API_KEY", "test")

import agent_posttrade
from agent_posttrade import (
    CONFIDENCE_BINS,
    CONFIDENCE_LABELS,
    build_table_a,
    build_table_b,
    build_table_c,
    build_table_d,
    compute_overall_win_rate,
    format_console_report,
    format_discord,
    load_evaluated_entries,
    run_analysis,
    _parse_findings,
)
from agent_posttrade_prompts import ALLOWED_CATEGORIES, ALLOWED_SEVERITIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).date().isoformat()


def _build_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame that matches what load_evaluated_entries returns."""
    defaults = {
        "date": _days_ago(5),
        "ticker": "AAPL",
        "model_signal": "BUY",
        "confidence": 0.50,
        "result": "WIN",
        "price": 180.0,
        "outcome_price": 185.0,
        "shap_driver_1": "RSI_14",
        "shap_driver_2": "MACD",
        "shap_driver_3": "Volatility",
        "actual_action": "BUY",
        "row_type": "ENTRY",
    }
    filled = [{**defaults, **r} for r in rows]
    df = pd.DataFrame(filled)
    df["date"] = pd.to_datetime(df["date"])
    for col in ("confidence", "price", "outcome_price"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _write_signal_log(path: str, rows: list[dict]) -> None:
    """Write a minimal signal_log.csv for load_evaluated_entries."""
    fieldnames = [
        "date", "ticker", "model_signal", "price", "qty", "confidence",
        "evaluation_date", "actual_action", "outcome_price", "result",
        "shap_driver_1", "shap_driver_2", "shap_driver_3", "row_type",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            full = {f: "" for f in fieldnames}
            full.update(row)
            writer.writerow(full)


def _valid_findings_json(**overrides) -> str:
    """Return valid LLM output JSON."""
    obj = {
        "findings": [
            {
                "category": "TOXIC_COMBO",
                "severity": "HIGH",
                "pattern": "INTC BUYs driven by Volume_Ratio: 0 wins in 4 trades",
                "evidence": {"ticker": "INTC", "wins": 0, "losses": 4},
                "suggested_action": "Investigate Volume_Ratio for INTC",
            },
        ],
        "summary": "1 high-severity finding.",
    }
    obj.update(overrides)
    return json.dumps(obj)


def _make_completion(content: str):
    """Build a mock Groq completion."""
    completion = MagicMock()
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    completion.choices = [choice]
    return completion


# ===========================================================================
# 1. load_evaluated_entries
# ===========================================================================

class TestLoadEvaluatedEntries:

    def test_loads_win_loss_neutral(self, tmp_path):
        csv_path = str(tmp_path / "signal_log.csv")
        _write_signal_log(csv_path, [
            {"date": _days_ago(3), "ticker": "AAPL", "model_signal": "BUY",
             "result": "WIN", "row_type": "ENTRY", "confidence": "0.55",
             "price": "180", "outcome_price": "190"},
            {"date": _days_ago(3), "ticker": "TSLA", "model_signal": "SELL",
             "result": "LOSS", "row_type": "ENTRY", "confidence": "0.60",
             "price": "250", "outcome_price": "260"},
            {"date": _days_ago(3), "ticker": "NVDA", "model_signal": "BUY",
             "result": "NEUTRAL", "row_type": "ENTRY", "confidence": "0.45",
             "price": "140", "outcome_price": "141"},
        ])

        df = load_evaluated_entries(csv_path, window_days=30)

        assert len(df) == 3
        assert set(df["result"]) == {"WIN", "LOSS", "NEUTRAL"}

    def test_excludes_skipped_and_blank_results(self, tmp_path):
        csv_path = str(tmp_path / "signal_log.csv")
        _write_signal_log(csv_path, [
            {"date": _days_ago(3), "ticker": "AAPL", "model_signal": "BUY",
             "result": "WIN", "row_type": "ENTRY", "confidence": "0.50",
             "price": "180", "outcome_price": "190"},
            {"date": _days_ago(3), "ticker": "TSLA", "model_signal": "BUY",
             "result": "SKIPPED", "row_type": "ENTRY", "confidence": "0.50",
             "price": "250", "outcome_price": ""},
            {"date": _days_ago(3), "ticker": "INTC", "model_signal": "BUY",
             "result": "", "row_type": "ENTRY", "confidence": "0.50",
             "price": "30", "outcome_price": ""},
        ])

        df = load_evaluated_entries(csv_path, window_days=30)

        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "AAPL"

    def test_excludes_exit_rows(self, tmp_path):
        csv_path = str(tmp_path / "signal_log.csv")
        _write_signal_log(csv_path, [
            {"date": _days_ago(3), "ticker": "AAPL", "model_signal": "BUY",
             "result": "WIN", "row_type": "ENTRY", "confidence": "0.50",
             "price": "180", "outcome_price": "190"},
            {"date": _days_ago(3), "ticker": "AAPL", "model_signal": "",
             "result": "WIN", "row_type": "EXIT", "confidence": "",
             "price": "", "outcome_price": ""},
        ])

        df = load_evaluated_entries(csv_path, window_days=30)

        assert len(df) == 1

    def test_excludes_hold_signals(self, tmp_path):
        csv_path = str(tmp_path / "signal_log.csv")
        _write_signal_log(csv_path, [
            {"date": _days_ago(3), "ticker": "AAPL", "model_signal": "HOLD",
             "result": "GOOD HOLD", "row_type": "ENTRY", "confidence": "0.50",
             "price": "180", "outcome_price": "181"},
        ])

        df = load_evaluated_entries(csv_path, window_days=30)

        assert len(df) == 0

    def test_window_filter_excludes_old_rows(self, tmp_path):
        csv_path = str(tmp_path / "signal_log.csv")
        _write_signal_log(csv_path, [
            {"date": _days_ago(5), "ticker": "AAPL", "model_signal": "BUY",
             "result": "WIN", "row_type": "ENTRY", "confidence": "0.50",
             "price": "180", "outcome_price": "190"},
            {"date": _days_ago(40), "ticker": "TSLA", "model_signal": "BUY",
             "result": "LOSS", "row_type": "ENTRY", "confidence": "0.50",
             "price": "250", "outcome_price": "240"},
        ])

        df = load_evaluated_entries(csv_path, window_days=30)

        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "AAPL"

    def test_missing_file_returns_empty(self):
        df = load_evaluated_entries("/no/such/file.csv", window_days=30)
        assert df.empty

    def test_empty_file_returns_empty(self, tmp_path):
        csv_path = str(tmp_path / "signal_log.csv")
        _write_signal_log(csv_path, [])
        df = load_evaluated_entries(csv_path, window_days=30)
        assert df.empty


# ===========================================================================
# 2. Table builders
# ===========================================================================

class TestBuildTableA:

    def test_basic_per_ticker_stats(self):
        df = _build_df([
            {"ticker": "AAPL", "model_signal": "BUY", "result": "WIN", "confidence": 0.55},
            {"ticker": "AAPL", "model_signal": "BUY", "result": "LOSS", "confidence": 0.40},
            {"ticker": "AAPL", "model_signal": "BUY", "result": "NEUTRAL", "confidence": 0.45},
        ])

        table = build_table_a(df)

        assert "AAPL" in table
        assert "BUY" in table
        # 1 win, 1 loss out of 2 decided = 50% win rate
        assert "50.0" in table

    def test_separates_buy_and_sell(self):
        df = _build_df([
            {"ticker": "TSLA", "model_signal": "BUY", "result": "WIN"},
            {"ticker": "TSLA", "model_signal": "SELL", "result": "WIN"},
        ])

        table = build_table_a(df)

        assert "BUY" in table
        assert "SELL" in table

    def test_empty_returns_no_data(self):
        assert build_table_a(pd.DataFrame()) == "(no data)"


class TestBuildTableB:

    def test_aggregates_across_tickers(self):
        df = _build_df([
            {"ticker": "AAPL", "shap_driver_1": "RSI_14", "result": "WIN"},
            {"ticker": "INTC", "shap_driver_1": "RSI_14", "result": "LOSS"},
            {"ticker": "TSLA", "shap_driver_2": "RSI_14", "result": "WIN"},
        ])

        table = build_table_b(df)

        # RSI_14 appears as driver across all three: 2W/1L = 66.7%
        assert "RSI_14" in table

    def test_counts_feature_in_any_shap_position(self):
        df = _build_df([
            {"shap_driver_1": "A", "shap_driver_2": "B", "shap_driver_3": "C", "result": "WIN"},
            {"shap_driver_1": "B", "shap_driver_2": "C", "shap_driver_3": "A", "result": "LOSS"},
        ])

        table = build_table_b(df)

        # Each of A, B, C should appear in the table
        for feat in ("A", "B", "C"):
            assert feat in table

    def test_empty_returns_no_data(self):
        assert build_table_b(pd.DataFrame()) == "(no data)"


class TestBuildTableC:

    def test_surfaces_low_win_rate_combos(self):
        df = _build_df([
            {"ticker": "INTC", "shap_driver_1": "Volume_Ratio", "result": "LOSS"},
            {"ticker": "INTC", "shap_driver_1": "Volume_Ratio", "result": "LOSS"},
            {"ticker": "INTC", "shap_driver_1": "Volume_Ratio", "result": "LOSS"},
        ])

        table = build_table_c(df)

        assert "INTC" in table
        assert "Volume_Ratio" in table
        assert "0.0" in table

    def test_excludes_high_win_rate_combos(self):
        df = _build_df([
            {"ticker": "AAPL", "shap_driver_1": "RSI_14", "result": "WIN"},
            {"ticker": "AAPL", "shap_driver_1": "RSI_14", "result": "WIN"},
            {"ticker": "AAPL", "shap_driver_1": "RSI_14", "result": "LOSS"},
        ])

        table = build_table_c(df)

        # 66.7% win rate > 33% threshold — should not appear
        assert table == "(none found)"

    def test_excludes_single_trade_combos(self):
        df = _build_df([
            {"ticker": "INTC", "shap_driver_1": "Volume_Ratio", "result": "LOSS"},
        ])

        table = build_table_c(df)

        # Only 1 trade — below the 2-trade minimum
        assert table == "(none found)"

    def test_empty_returns_none_found(self):
        assert build_table_c(pd.DataFrame()) == "(none found)"

    def test_neutrals_excluded_from_decided_count(self):
        df = _build_df([
            {"ticker": "INTC", "shap_driver_1": "Vol", "result": "NEUTRAL"},
            {"ticker": "INTC", "shap_driver_1": "Vol", "result": "NEUTRAL"},
            {"ticker": "INTC", "shap_driver_1": "Vol", "result": "NEUTRAL"},
        ])

        table = build_table_c(df)

        # 0 decided trades (all NEUTRAL) — below 2-trade min
        assert table == "(none found)"


class TestBuildTableD:

    def test_buckets_confidence_correctly(self):
        df = _build_df([
            {"confidence": 0.36, "result": "LOSS"},
            {"confidence": 0.37, "result": "LOSS"},
            {"confidence": 0.52, "result": "WIN"},
            {"confidence": 0.53, "result": "WIN"},
            {"confidence": 0.70, "result": "WIN"},
        ])

        table = build_table_d(df)

        assert "0.35" in table   # the 0.35-0.40 bucket
        assert "0.50" in table   # the 0.50-0.65 bucket
        assert "0.65+" in table  # the 0.65+ bucket

    def test_excludes_neutrals(self):
        df = _build_df([
            {"confidence": 0.50, "result": "NEUTRAL"},
            {"confidence": 0.50, "result": "NEUTRAL"},
        ])

        table = build_table_d(df)

        assert table == "(no decided trades)"

    def test_empty_returns_no_data(self):
        assert build_table_d(pd.DataFrame()) == "(no data)"


class TestComputeOverallWinRate:

    def test_basic_win_rate(self):
        df = _build_df([
            {"result": "WIN"},
            {"result": "LOSS"},
            {"result": "WIN"},
            {"result": "NEUTRAL"},
        ])

        rate = compute_overall_win_rate(df)

        # 2 wins out of 3 decided (excluding NEUTRAL) = 66.7%
        assert abs(rate - 66.7) < 0.1

    def test_all_neutrals_returns_zero(self):
        df = _build_df([
            {"result": "NEUTRAL"},
            {"result": "NEUTRAL"},
        ])
        assert compute_overall_win_rate(df) == 0.0

    def test_empty_returns_zero(self):
        assert compute_overall_win_rate(pd.DataFrame()) == 0.0


# ===========================================================================
# 3. LLM output parsing (_parse_findings)
# ===========================================================================

class TestParseFindings:

    def test_valid_findings(self):
        result = _parse_findings(_valid_findings_json())

        assert len(result["findings"]) == 1
        assert result["findings"][0]["category"] == "TOXIC_COMBO"
        assert result["findings"][0]["severity"] == "HIGH"
        assert isinstance(result["summary"], str)

    def test_empty_findings_list_is_valid(self):
        result = _parse_findings(json.dumps({
            "findings": [],
            "summary": "No patterns found.",
        }))

        assert result["findings"] == []

    def test_all_categories_accepted(self):
        for cat in ALLOWED_CATEGORIES:
            obj = json.dumps({
                "findings": [{
                    "category": cat,
                    "severity": "MEDIUM",
                    "pattern": "test",
                    "evidence": {},
                    "suggested_action": "test",
                }],
                "summary": "test",
            })
            result = _parse_findings(obj)
            assert result["findings"][0]["category"] == cat

    def test_all_severities_accepted(self):
        for sev in ALLOWED_SEVERITIES:
            obj = json.dumps({
                "findings": [{
                    "category": "TOXIC_COMBO",
                    "severity": sev,
                    "pattern": "test",
                    "evidence": {},
                    "suggested_action": "test",
                }],
                "summary": "test",
            })
            result = _parse_findings(obj)
            assert result["findings"][0]["severity"] == sev

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            _parse_findings("not json at all")

    def test_non_dict_raises(self):
        with pytest.raises(ValueError, match="must be a JSON object"):
            _parse_findings(json.dumps([1, 2, 3]))

    def test_missing_findings_key_raises(self):
        with pytest.raises(ValueError, match="must have a 'findings' key"):
            _parse_findings(json.dumps({"summary": "hi"}))

    def test_findings_not_list_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            _parse_findings(json.dumps({"findings": "nope", "summary": "hi"}))

    def test_invalid_category_raises(self):
        obj = json.dumps({
            "findings": [{
                "category": "MADE_UP_CATEGORY",
                "severity": "HIGH",
                "pattern": "test",
                "evidence": {},
                "suggested_action": "test",
            }],
            "summary": "test",
        })
        with pytest.raises(ValueError, match="category must be one of"):
            _parse_findings(obj)

    def test_invalid_severity_raises(self):
        obj = json.dumps({
            "findings": [{
                "category": "TOXIC_COMBO",
                "severity": "CRITICAL",
                "pattern": "test",
                "evidence": {},
                "suggested_action": "test",
            }],
            "summary": "test",
        })
        with pytest.raises(ValueError, match="severity must be one of"):
            _parse_findings(obj)

    def test_missing_finding_keys_raises(self):
        obj = json.dumps({
            "findings": [{"category": "TOXIC_COMBO"}],
            "summary": "test",
        })
        with pytest.raises(ValueError, match="missing keys"):
            _parse_findings(obj)

    def test_missing_summary_raises(self):
        obj = json.dumps({
            "findings": [{
                "category": "TOXIC_COMBO",
                "severity": "HIGH",
                "pattern": "test",
                "evidence": {},
                "suggested_action": "test",
            }],
        })
        with pytest.raises(ValueError, match="must have a 'summary' string"):
            _parse_findings(obj)


# ===========================================================================
# 4. LLM interaction (run_analysis)
# ===========================================================================

class TestRunAnalysis:

    def test_happy_path_returns_parsed_findings(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _make_completion(_valid_findings_json())
        )

        with patch("agent_posttrade.Groq", return_value=mock_client):
            result = run_analysis("some tables here")

        assert len(result["findings"]) == 1
        assert result["findings"][0]["category"] == "TOXIC_COMBO"

    def test_retry_on_bad_json_then_recover(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_completion("not json"),
            _make_completion(_valid_findings_json()),
        ]

        with patch("agent_posttrade.Groq", return_value=mock_client):
            result = run_analysis("some tables here")

        assert len(result["findings"]) == 1
        assert mock_client.chat.completions.create.call_count == 2

    def test_parse_exhausted_raises_runtime_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_completion("bad json 1"),
            _make_completion("bad json 2"),
        ]

        with patch("agent_posttrade.Groq", return_value=mock_client):
            with pytest.raises(RuntimeError, match="parse failed after retries"):
                run_analysis("some tables here")

    def test_groq_api_failure_raises_runtime_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("rate limited")

        with patch("agent_posttrade.Groq", return_value=mock_client):
            with pytest.raises(RuntimeError, match="Groq API failed"):
                run_analysis("some tables here")


# ===========================================================================
# 5. Formatting
# ===========================================================================

class TestFormatConsoleReport:

    def _meta(self):
        return {
            "start_date": "2026-08-15",
            "end_date": "2026-09-10",
            "total_signals": 24,
            "overall_win_rate": 45.5,
        }

    def test_includes_header_and_findings(self):
        result = json.loads(_valid_findings_json())
        report = format_console_report(result, self._meta())

        assert "POST-TRADE ANALYSIS REPORT" in report
        assert "2026-08-15" in report
        assert "TOXIC_COMBO" in report
        assert "Volume_Ratio" in report

    def test_empty_findings_says_no_patterns(self):
        result = {"findings": [], "summary": "All clear."}
        report = format_console_report(result, self._meta())

        assert "No actionable patterns found" in report

    def test_severity_icons(self):
        findings = []
        for sev in ("HIGH", "MEDIUM", "LOW"):
            findings.append({
                "category": "TOXIC_COMBO",
                "severity": sev,
                "pattern": f"test {sev}",
                "evidence": {},
                "suggested_action": "test",
            })
        result = {"findings": findings, "summary": "test"}
        report = format_console_report(result, self._meta())

        assert "🔴" in report
        assert "🟡" in report
        assert "🔵" in report


class TestFormatDiscord:

    def _meta(self):
        return {
            "start_date": "2026-08-15",
            "end_date": "2026-09-10",
            "total_signals": 24,
            "overall_win_rate": 45.5,
        }

    def test_discord_with_findings(self):
        result = json.loads(_valid_findings_json())
        msg = format_discord(result, self._meta())

        assert "Post-Trade Analysis" in msg
        assert "TOXIC_COMBO" in msg
        assert "Volume_Ratio" in msg

    def test_discord_empty_findings(self):
        result = {"findings": [], "summary": "All clear."}
        msg = format_discord(result, self._meta())

        assert "No actionable patterns found" in msg

    def test_discord_includes_date_range(self):
        result = {"findings": [], "summary": "test"}
        msg = format_discord(result, self._meta())

        assert "2026-08-15" in msg
        assert "2026-09-10" in msg


# ===========================================================================
# 6. Discord send
# ===========================================================================

class TestSendDiscord:

    def test_skips_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        # Should not raise
        agent_posttrade.send_discord("test message")

    def test_posts_when_env_set(self, monkeypatch):
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.invalid/webhook")
        import requests
        mock_post = MagicMock()
        mock_post.return_value.status_code = 200
        monkeypatch.setattr(requests, "post", mock_post)

        agent_posttrade.send_discord("test message")

        mock_post.assert_called_once()
        assert mock_post.call_args.kwargs["json"]["content"] == "test message"

    def test_swallows_post_failure(self, monkeypatch):
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.invalid/webhook")
        import requests
        monkeypatch.setattr(
            requests, "post",
            MagicMock(side_effect=RuntimeError("network down")),
        )
        # Should not raise
        agent_posttrade.send_discord("test message")


# ===========================================================================
# 7. run() integration
# ===========================================================================

class TestRun:

    def test_returns_zero_on_no_data(self, tmp_path):
        csv_path = str(tmp_path / "signal_log.csv")
        _write_signal_log(csv_path, [])

        rc = agent_posttrade.run(log_path=csv_path, window_days=30, discord=False)

        assert rc == 0

    def test_returns_zero_on_missing_file(self, tmp_path):
        rc = agent_posttrade.run(
            log_path=str(tmp_path / "nope.csv"), window_days=30, discord=False
        )
        assert rc == 0

    def test_returns_one_on_llm_failure(self, tmp_path):
        csv_path = str(tmp_path / "signal_log.csv")
        _write_signal_log(csv_path, [
            {"date": _days_ago(3), "ticker": "AAPL", "model_signal": "BUY",
             "result": "WIN", "row_type": "ENTRY", "confidence": "0.50",
             "price": "180", "outcome_price": "190"},
        ])

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API down")

        with patch("agent_posttrade.Groq", return_value=mock_client):
            rc = agent_posttrade.run(log_path=csv_path, window_days=30, discord=False)

        assert rc == 1

    def test_happy_path_with_discord(self, tmp_path, monkeypatch):
        csv_path = str(tmp_path / "signal_log.csv")
        _write_signal_log(csv_path, [
            {"date": _days_ago(3), "ticker": "AAPL", "model_signal": "BUY",
             "result": "WIN", "row_type": "ENTRY", "confidence": "0.50",
             "price": "180", "outcome_price": "190",
             "shap_driver_1": "RSI_14", "shap_driver_2": "MACD",
             "shap_driver_3": "Volatility"},
        ])

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _make_completion(_valid_findings_json())
        )

        mock_discord = MagicMock()
        monkeypatch.setattr(agent_posttrade, "send_discord", mock_discord)

        with patch("agent_posttrade.Groq", return_value=mock_client):
            rc = agent_posttrade.run(
                log_path=csv_path, window_days=30, discord=True
            )

        assert rc == 0
        mock_discord.assert_called_once()
        posted = mock_discord.call_args[0][0]
        assert "Post-Trade Analysis" in posted
