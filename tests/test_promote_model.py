"""Tests for promote_model.py — the champion/challenger promotion gate.

This is the function that decides whether a newly trained model starts trading
real signals. It is failure-biased by design: every ambiguous case must keep the
incumbent. The tests below assert that bias explicitly, because a gate that
silently drifts toward "promote" would reintroduce the exact problem it exists
to prevent — a worse model going live and nobody noticing.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

import config
import promote_model
from promote_model import (
    archive_champion,
    build_report,
    decide_promotion,
    extract_gate_metrics,
    install_candidate,
    write_decision_file,
)


# --- helpers ---------------------------------------------------------------


def _metrics(
    *,
    buy_f1: float = 0.50,
    buy_precision: float = 0.50,
    buy_recall: float = 0.50,
    buy_support: int = 100,
    accuracy: float = 0.60,
    total_return: float = 5.0,
    n_trades: int = 40,
    bt_win_rate: float = 50.0,
    avg_return: float = 1.0,
    max_drawdown: float = -10.0,
    final_value: float = 10_500.0,
) -> dict:
    return {
        "buy_f1": buy_f1,
        "buy_precision": buy_precision,
        "buy_recall": buy_recall,
        "buy_support": buy_support,
        "accuracy": accuracy,
        "total_return": total_return,
        "n_trades": n_trades,
        "bt_win_rate": bt_win_rate,
        "avg_return": avg_return,
        "max_drawdown": max_drawdown,
        "final_value": final_value,
    }


def _winner(champion: dict) -> dict:
    """A challenger that clears every gate against `champion`."""
    return _metrics(
        buy_f1=champion["buy_f1"] + 0.05,
        buy_precision=config.PROMOTION_MIN_BUY_PRECISION + 0.10,
        buy_recall=config.PROMOTION_MIN_BUY_RECALL + 0.10,
        buy_support=config.PROMOTION_MIN_TEST_BUY_SUPPORT + 50,
        total_return=(
            champion["total_return"]
            + config.PROMOTION_MIN_RETURN_IMPROVEMENT_PCT
            + 5.0
        ),
        n_trades=config.PROMOTION_MIN_BACKTEST_TRADES + 25,
        max_drawdown=config.PROMOTION_MAX_DRAWDOWN_PCT + 15.0,
    )


def _failed_check_names(decision: dict) -> set[str]:
    return {c["name"] for c in decision["checks"] if not c["passed"]}


# --- metric extraction -----------------------------------------------------


def test_extract_gate_metrics_reads_the_buy_class():
    report = {
        "BUY": {"precision": 0.55, "recall": 0.40, "f1-score": 0.46, "support": 120},
        "HOLD": {"precision": 0.9, "recall": 0.95, "f1-score": 0.92, "support": 800},
        "accuracy": 0.83,
    }

    metrics = extract_gate_metrics(report)

    assert metrics["buy_precision"] == 0.55
    assert metrics["buy_recall"] == 0.40
    assert metrics["buy_f1"] == 0.46
    assert metrics["buy_support"] == 120
    assert metrics["accuracy"] == 0.83


def test_extract_gate_metrics_handles_a_model_that_never_predicts_buy():
    """sklearn omits or zeroes the BUY block when nothing was predicted BUY."""
    metrics = extract_gate_metrics({"HOLD": {"f1-score": 0.9}, "accuracy": 0.9})

    assert metrics["buy_f1"] == 0.0
    assert metrics["buy_precision"] == 0.0
    assert metrics["buy_recall"] == 0.0
    assert metrics["buy_support"] == 0


def test_extract_gate_metrics_handles_null_buy_block():
    assert extract_gate_metrics({"BUY": None})["buy_f1"] == 0.0


# --- the happy path --------------------------------------------------------


def test_clearly_better_challenger_is_promoted():
    champion = _metrics(buy_f1=0.40)
    decision = decide_promotion(champion, _winner(champion))

    assert decision["promote"] is True
    assert all(c["passed"] for c in decision["checks"])
    assert "PROMOTE" in decision["summary"]


# --- the improvement margin ------------------------------------------------


def test_marginally_better_challenger_is_rejected():
    """Beating the champion by less than the margin is not enough."""
    champion = _metrics(total_return=5.0)
    challenger = _winner(champion)
    challenger["total_return"] = (
        5.0 + config.PROMOTION_MIN_RETURN_IMPROVEMENT_PCT - 0.01
    )

    decision = decide_promotion(champion, challenger)

    assert decision["promote"] is False
    assert "beats_champion_return" in _failed_check_names(decision)


def test_challenger_exactly_at_the_margin_is_promoted():
    """The comparison is `>=`, so hitting the margin exactly must pass."""
    champion = _metrics(total_return=5.0)
    challenger = _winner(champion)
    challenger["total_return"] = 5.0 + config.PROMOTION_MIN_RETURN_IMPROVEMENT_PCT

    decision = decide_promotion(champion, challenger)

    assert decision["promote"] is True


def test_identical_model_is_rejected():
    """Retraining that changes nothing must not churn the live model."""
    champion = _metrics(buy_f1=0.45, buy_precision=0.50, buy_recall=0.45)
    decision = decide_promotion(champion, dict(champion))

    assert decision["promote"] is False
    assert "beats_champion_return" in _failed_check_names(decision)


def test_worse_challenger_is_rejected():
    champion = _metrics(total_return=12.0)
    challenger = _winner(champion)
    challenger["total_return"] = -4.0

    decision = decide_promotion(champion, challenger)

    assert decision["promote"] is False
    assert "beats_champion_return" in _failed_check_names(decision)


def test_better_f1_does_not_promote_a_worse_earner():
    """The whole point of moving the gate off F1: the two can disagree, and
    dollars win."""
    champion = _metrics(buy_f1=0.30, total_return=15.0)
    challenger = _winner(champion)
    challenger["buy_f1"] = 0.95          # far better classifier
    challenger["total_return"] = 2.0     # far worse trader

    decision = decide_promotion(champion, challenger)

    assert decision["promote"] is False
    assert "beats_champion_return" in _failed_check_names(decision)


def test_worse_f1_can_still_promote_a_better_earner():
    """The converse — F1 no longer has a veto."""
    champion = _metrics(buy_f1=0.60, total_return=2.0)
    challenger = _winner(champion)
    challenger["buy_f1"] = 0.35

    decision = decide_promotion(champion, challenger)

    assert decision["promote"] is True


# --- backtest sample size --------------------------------------------------


def test_too_few_simulated_trades_blocks_promotion():
    """A return built on three positions is luck, not a strategy."""
    champion = _metrics(total_return=1.0)
    challenger = _winner(champion)
    challenger["n_trades"] = config.PROMOTION_MIN_BACKTEST_TRADES - 1

    decision = decide_promotion(champion, challenger)

    assert decision["promote"] is False
    assert "backtest_trade_count" in _failed_check_names(decision)


def test_trade_count_exactly_at_minimum_passes():
    champion = _metrics(total_return=1.0)
    challenger = _winner(champion)
    challenger["n_trades"] = config.PROMOTION_MIN_BACKTEST_TRADES

    assert decide_promotion(champion, challenger)["promote"] is True


def test_a_backtest_that_did_not_run_cannot_promote():
    """extract_backtest_metrics collapses empty stats to failing values."""
    from promote_model import extract_backtest_metrics

    empty = extract_backtest_metrics({})
    challenger = {**_winner(_metrics()), **empty}

    decision = decide_promotion(_metrics(total_return=1.0), challenger)

    assert decision["promote"] is False


# --- drawdown floor --------------------------------------------------------


def test_catastrophic_drawdown_blocks_a_higher_return():
    """Earning more by risking ruin is not an improvement."""
    champion = _metrics(total_return=5.0, max_drawdown=-8.0)
    challenger = _winner(champion)
    challenger["total_return"] = 50.0
    challenger["max_drawdown"] = config.PROMOTION_MAX_DRAWDOWN_PCT - 0.1

    decision = decide_promotion(champion, challenger)

    assert decision["promote"] is False
    assert "challenger_max_drawdown" in _failed_check_names(decision)
    # It genuinely did earn more — risk is what stopped it.
    assert "beats_champion_return" not in _failed_check_names(decision)


def test_drawdown_exactly_at_floor_passes():
    champion = _metrics(total_return=1.0)
    challenger = _winner(champion)
    challenger["max_drawdown"] = config.PROMOTION_MAX_DRAWDOWN_PCT

    assert decide_promotion(champion, challenger)["promote"] is True


# --- absolute floors -------------------------------------------------------


def test_low_precision_challenger_rejected_even_when_it_beats_a_worse_champion():
    """A decayed champion must not drag the bar below the absolute floor."""
    champion = _metrics(buy_f1=0.05, buy_precision=0.05)
    challenger = _winner(champion)
    challenger["buy_precision"] = config.PROMOTION_MIN_BUY_PRECISION - 0.01

    decision = decide_promotion(champion, challenger)

    assert decision["promote"] is False
    assert "challenger_buy_precision" in _failed_check_names(decision)
    # It genuinely did beat the champion — the floor is what stopped it.
    assert "beats_champion_return" not in _failed_check_names(decision)


def test_precision_exactly_at_floor_passes():
    champion = _metrics(buy_f1=0.20)
    challenger = _winner(champion)
    challenger["buy_precision"] = config.PROMOTION_MIN_BUY_PRECISION

    assert decide_promotion(champion, challenger)["promote"] is True


def test_low_recall_challenger_is_rejected():
    """Guards the degenerate model that gets high precision by never buying."""
    champion = _metrics(buy_f1=0.20)
    challenger = _winner(champion)
    challenger["buy_recall"] = config.PROMOTION_MIN_BUY_RECALL - 0.01

    decision = decide_promotion(champion, challenger)

    assert decision["promote"] is False
    assert "challenger_buy_recall" in _failed_check_names(decision)


def test_recall_exactly_at_floor_passes():
    champion = _metrics(buy_f1=0.20)
    challenger = _winner(champion)
    challenger["buy_recall"] = config.PROMOTION_MIN_BUY_RECALL

    assert decide_promotion(champion, challenger)["promote"] is True


def test_model_that_never_buys_is_rejected():
    """All-HOLD scores well on an imbalanced set but is worthless to trade."""
    champion = _metrics(buy_f1=0.30)
    challenger = _metrics(
        buy_f1=0.0, buy_precision=0.0, buy_recall=0.0,
        buy_support=config.PROMOTION_MIN_TEST_BUY_SUPPORT + 50,
        accuracy=0.95,   # high accuracy, no edge
    )

    decision = decide_promotion(champion, challenger)

    assert decision["promote"] is False


# --- test-set size ---------------------------------------------------------


def test_thin_test_set_blocks_promotion():
    """Too few BUY rows to distinguish signal from noise."""
    champion = _metrics(buy_f1=0.20)
    challenger = _winner(champion)
    challenger["buy_support"] = config.PROMOTION_MIN_TEST_BUY_SUPPORT - 1

    decision = decide_promotion(champion, challenger)

    assert decision["promote"] is False
    assert "test_buy_support" in _failed_check_names(decision)


def test_support_exactly_at_minimum_passes():
    champion = _metrics(buy_f1=0.20)
    challenger = _winner(champion)
    challenger["buy_support"] = config.PROMOTION_MIN_TEST_BUY_SUPPORT

    assert decide_promotion(champion, challenger)["promote"] is True


def test_thin_test_set_blocks_even_a_dramatically_better_challenger():
    champion = _metrics(buy_f1=0.10)
    challenger = _metrics(
        buy_f1=0.99, buy_precision=0.99, buy_recall=0.99, buy_support=3
    )

    assert decide_promotion(champion, challenger)["promote"] is False


# --- first promotion (no champion) -----------------------------------------


def test_first_promotion_with_no_champion_passes_on_floors_alone():
    challenger = _metrics(
        buy_f1=0.40,
        buy_precision=config.PROMOTION_MIN_BUY_PRECISION + 0.05,
        buy_recall=config.PROMOTION_MIN_BUY_RECALL + 0.05,
        buy_support=config.PROMOTION_MIN_TEST_BUY_SUPPORT + 10,
    )

    decision = decide_promotion(None, challenger)

    assert decision["promote"] is True
    assert "no champion" in dict(
        (c["name"], c["detail"]) for c in decision["checks"]
    )["beats_champion_return"]


def test_first_promotion_still_enforces_the_floors():
    """No incumbent is not a reason to accept anything."""
    challenger = _metrics(
        buy_precision=config.PROMOTION_MIN_BUY_PRECISION - 0.01,
        buy_support=config.PROMOTION_MIN_TEST_BUY_SUPPORT + 10,
    )

    decision = decide_promotion(None, challenger)

    assert decision["promote"] is False
    assert "challenger_buy_precision" in _failed_check_names(decision)


# --- check reporting -------------------------------------------------------


def test_every_check_runs_even_after_one_fails():
    """The Discord report shows the full picture, not just the first failure."""
    champion = _metrics(buy_f1=0.90, total_return=40.0)
    challenger = _metrics(
        buy_f1=0.01, buy_precision=0.01, buy_recall=0.01, buy_support=1,
        total_return=-20.0, n_trades=2, max_drawdown=-90.0,
    )

    decision = decide_promotion(champion, challenger)

    names = [c["name"] for c in decision["checks"]]
    assert names == [
        "test_buy_support",
        "challenger_buy_precision",
        "challenger_buy_recall",
        "backtest_trade_count",
        "challenger_max_drawdown",
        "beats_champion_return",
    ]
    assert len(_failed_check_names(decision)) == 6


def test_summary_names_the_failed_checks():
    champion = _metrics(buy_f1=0.20)
    challenger = _winner(champion)
    challenger["buy_recall"] = 0.0

    decision = decide_promotion(champion, challenger)

    assert "REJECT" in decision["summary"]
    assert "challenger_buy_recall" in decision["summary"]


# --- report formatting -----------------------------------------------------


def test_report_marks_a_promotion():
    champion = _metrics(buy_f1=0.40)
    challenger = _winner(champion)
    decision = decide_promotion(champion, challenger)

    report = build_report(
        decision, champion, challenger, dry_run=False, archived_to="models/archive/x.joblib"
    )

    assert "PROMOTED" in report
    assert "models/archive/x.joblib" in report
    assert "dry run" not in report


def test_report_marks_a_rejection_and_a_dry_run():
    champion = _metrics(buy_f1=0.90)
    challenger = _metrics(buy_f1=0.10, buy_support=200)
    decision = decide_promotion(champion, challenger)

    report = build_report(decision, champion, challenger, dry_run=True, archived_to=None)

    assert "REJECTED" in report
    assert "dry run" in report


def test_report_handles_a_missing_champion():
    challenger = _metrics(buy_support=200)
    decision = decide_promotion(None, challenger)

    report = build_report(decision, None, challenger, dry_run=False, archived_to=None)

    assert "none on disk" in report


# --- archive / install file operations -------------------------------------


@pytest.fixture
def model_dir(monkeypatch, tmp_path):
    archive = tmp_path / "archive"
    champion = tmp_path / "XG_Boost.joblib"
    monkeypatch.setattr(promote_model, "MODEL_ARCHIVE_DIR", str(archive))
    monkeypatch.setattr(promote_model, "CHAMPION_PATH", str(champion))
    return {"root": tmp_path, "archive": archive, "champion": champion}


def test_archive_returns_none_when_no_champion_exists(model_dir):
    assert archive_champion(str(model_dir["champion"])) is None


def test_archive_copies_and_leaves_the_original_in_place(model_dir):
    model_dir["champion"].write_bytes(b"champion-bytes")

    archived = archive_champion(str(model_dir["champion"]))

    assert archived is not None
    assert os.path.exists(archived)
    assert open(archived, "rb").read() == b"champion-bytes"
    # The champion must survive until the candidate actually replaces it.
    assert model_dir["champion"].exists()


def test_archive_prunes_to_the_retention_limit(model_dir, monkeypatch):
    monkeypatch.setattr(promote_model, "MODEL_ARCHIVE_RETAIN", 3)
    model_dir["archive"].mkdir()
    for i in range(5):
        (model_dir["archive"] / f"XG_Boost_2026010{i}T000000Z.joblib").write_bytes(b"x")
    model_dir["champion"].write_bytes(b"champion")

    archive_champion(str(model_dir["champion"]))

    remaining = sorted(p.name for p in model_dir["archive"].glob("*.joblib"))
    assert len(remaining) == 3
    # Newest survive; the two oldest are gone.
    assert "XG_Boost_20260100T000000Z.joblib" not in remaining
    assert "XG_Boost_20260101T000000Z.joblib" not in remaining


def test_install_candidate_replaces_the_champion(model_dir):
    candidate = model_dir["root"] / "candidate.joblib"
    candidate.write_bytes(b"challenger-bytes")
    model_dir["champion"].write_bytes(b"champion-bytes")

    install_candidate(str(candidate), str(model_dir["champion"]))

    assert model_dir["champion"].read_bytes() == b"challenger-bytes"
    # Moved, not copied — no stale candidate left to be re-evaluated later.
    assert not candidate.exists()


# --- decision file ---------------------------------------------------------
#
# CI reads this file to learn what happened. It cannot infer the outcome from
# the filesystem: the candidate is moved on promotion and deleted on
# rejection, so it is absent either way.


def test_decision_file_records_a_promotion(tmp_path):
    path = tmp_path / "promotion_decision.json"
    champion = _metrics(buy_f1=0.40)
    challenger = _winner(champion)
    decision = decide_promotion(champion, challenger)

    write_decision_file(
        decision, champion, challenger,
        dry_run=False, archived_to="models/archive/x.joblib", path=str(path),
    )

    payload = json.loads(path.read_text())
    assert payload["promoted"] is True
    assert payload["dry_run"] is False
    assert payload["archived_to"] == "models/archive/x.joblib"
    assert payload["champion"]["buy_f1"] == 0.40
    assert payload["challenger"] == challenger
    assert len(payload["checks"]) == 6
    assert "timestamp_utc" in payload


def test_decision_file_records_a_rejection(tmp_path):
    path = tmp_path / "promotion_decision.json"
    champion = _metrics(buy_f1=0.90)
    challenger = _metrics(buy_f1=0.10, buy_support=200)
    decision = decide_promotion(champion, challenger)

    write_decision_file(
        decision, champion, challenger,
        dry_run=False, archived_to=None, path=str(path),
    )

    payload = json.loads(path.read_text())
    assert payload["promoted"] is False
    assert payload["archived_to"] is None
    assert "REJECT" in payload["summary"]


def test_decision_file_records_a_missing_champion(tmp_path):
    path = tmp_path / "promotion_decision.json"
    challenger = _metrics(buy_support=200)
    decision = decide_promotion(None, challenger)

    write_decision_file(
        decision, None, challenger,
        dry_run=True, archived_to=None, path=str(path),
    )

    payload = json.loads(path.read_text())
    assert payload["champion"] is None
    assert payload["dry_run"] is True


def test_decision_file_creates_missing_parent_directory(tmp_path):
    path = tmp_path / "nested" / "dir" / "promotion_decision.json"
    challenger = _metrics(buy_support=200)
    decision = decide_promotion(None, challenger)

    write_decision_file(
        decision, None, challenger,
        dry_run=False, archived_to=None, path=str(path),
    )

    assert path.exists()


def test_decision_file_write_failure_is_not_fatal(tmp_path):
    """A promotion already happened by this point — losing the receipt must
    not crash the run and mask it."""
    champion = _metrics(buy_f1=0.40)
    challenger = _winner(champion)
    decision = decide_promotion(champion, challenger)

    with patch("promote_model.open", side_effect=OSError("read-only fs")):
        write_decision_file(
            decision, champion, challenger,
            dry_run=False, archived_to=None, path=str(tmp_path / "d.json"),
        )   # must not raise
