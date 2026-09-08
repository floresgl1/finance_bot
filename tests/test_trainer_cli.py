"""Tests for trainer.py's CLI and headless contract.

The scheduled retrain runs trainer.py unattended. Two invariants make that
safe, and both are asserted here:

  1. Headless mode never reads stdin — a prompt in CI hangs until timeout.
  2. Headless mode never rewrites config.py, which is the source of truth the
     live bot reads, unless told to explicitly.
  3. --output leaves the live model completely untouched.
"""

import os

import numpy as np
import pandas as pd
import pytest

import trainer
from config import FEATURE_COLUMNS
from trainer import _parse_args


# --- argument parsing ------------------------------------------------------


def test_defaults_are_interactive_and_live():
    args = _parse_args([])

    assert args.headless is False
    assert args.output is None            # writes the live model
    assert args.update_config is None     # None means "ask"
    assert args.threshold is None


def test_headless_flag():
    assert _parse_args(["--headless"]).headless is True


def test_output_path():
    args = _parse_args(["--output", "models/candidate.joblib"])
    assert args.output == "models/candidate.joblib"


def test_explicit_config_flags_are_tri_state():
    assert _parse_args(["--update-config"]).update_config is True
    assert _parse_args(["--no-update-config"]).update_config is False
    assert _parse_args([]).update_config is None


def test_config_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        _parse_args(["--update-config", "--no-update-config"])


def test_threshold_is_parsed_as_float():
    assert _parse_args(["--threshold", "0.42"]).threshold == 0.42


@pytest.mark.parametrize("bad", ["0", "1", "1.5", "-0.1"])
def test_out_of_range_threshold_is_rejected_by_main(bad):
    assert trainer.main(["--threshold", bad]) == 1


# --- headless training contract --------------------------------------------


def _synthetic_dataset(n_rows: int = 400):
    """A dataset shaped like build_dataset()'s output.

    Signals are assigned so every chronological split slice contains all three
    classes — LabelEncoder is fit on the training slice only.
    """
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=n_rows, freq="D")
    X = pd.DataFrame(
        rng.normal(size=(n_rows, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
        index=index,
    )
    y = pd.Series(
        ["BUY", "HOLD", "SELL"] * (n_rows // 3) + ["BUY"] * (n_rows % 3),
        index=index,
        name="Signal",
    )
    return X, y


@pytest.fixture
def headless_env(monkeypatch, tmp_path):
    """Stub the dataset, and make any stdin read an immediate failure."""
    monkeypatch.setattr(trainer, "build_dataset", lambda: _synthetic_dataset())

    def _explode(*_args, **_kwargs):
        raise AssertionError("train() attempted to read stdin in headless mode")

    monkeypatch.setattr("builtins.input", _explode)
    return tmp_path


def test_headless_training_never_prompts(headless_env):
    out = headless_env / "candidate.joblib"

    result = trainer.train(output_path=str(out), headless=True)

    assert out.exists()
    assert result["model_path"] == str(out)


def test_headless_defaults_to_leaving_config_alone(headless_env, monkeypatch):
    called = []
    monkeypatch.setattr(
        trainer, "update_config_threshold", lambda t: called.append(t)
    )

    result = trainer.train(
        output_path=str(headless_env / "candidate.joblib"), headless=True
    )

    assert called == []
    assert result["config_updated"] is False


def test_headless_writes_config_only_when_asked(headless_env, monkeypatch):
    called = []
    monkeypatch.setattr(
        trainer, "update_config_threshold", lambda t: called.append(t)
    )

    result = trainer.train(
        output_path=str(headless_env / "candidate.joblib"),
        headless=True,
        update_config=True,
    )

    assert len(called) == 1
    assert result["config_updated"] is True


def test_output_path_leaves_the_live_model_untouched(headless_env, monkeypatch):
    """The core safety property the promotion gate depends on."""
    model_dir = headless_env / "models"
    model_dir.mkdir()
    live = model_dir / "XG_Boost.joblib"
    live.write_bytes(b"champion-bytes")
    monkeypatch.setattr(trainer, "MODEL_DIR", str(model_dir))
    monkeypatch.setattr(trainer, "MODEL_FILENAME", "XG_Boost.joblib")

    trainer.train(output_path=str(model_dir / "candidate.joblib"), headless=True)

    assert live.read_bytes() == b"champion-bytes"
    # No backup either — nothing was displaced, so there is nothing to back up.
    assert not (model_dir / "XG_Boost_backup.joblib").exists()


def test_supplied_threshold_skips_tuning(headless_env, monkeypatch):
    def _fail(*_args, **_kwargs):
        raise AssertionError("tune_threshold ran despite an explicit --threshold")

    monkeypatch.setattr(trainer, "tune_threshold", _fail)

    result = trainer.train(
        output_path=str(headless_env / "candidate.joblib"),
        headless=True,
        threshold=0.42,
    )

    assert result["threshold"] == 0.42


def test_train_returns_the_metrics_the_gate_needs(headless_env):
    result = trainer.train(
        output_path=str(headless_env / "candidate.joblib"), headless=True
    )

    assert set(result) >= {
        "threshold", "model_path", "n_train", "n_val", "n_test",
        "n_forced", "test_report", "config_updated",
    }
    assert result["n_train"] + result["n_val"] + result["n_test"] == 400
    assert "BUY" in result["test_report"]
