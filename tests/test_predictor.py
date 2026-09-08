"""Tests for predictor.py — staleness detection and the veto layers.

The vetoes are the last thing standing between a model signal and an order, and
`is_ticker_stale` decides whether a ticker is allowed to produce a signal at
all. Both are pure enough to pin down precisely.
"""

import os
from datetime import date, timedelta

import pandas as pd
import pytest

import config
import predictor
from predictor import (
    apply_earnings_veto,
    apply_sentiment_veto,
    get_recent_earnings_surprise,
    is_ticker_stale,
)


# --- staleness -------------------------------------------------------------


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(predictor, "DATA_DIR", str(tmp_path))
    return tmp_path


def _write_prices(path, last_date: date, rows: int = 5):
    dates = [(last_date - timedelta(days=i)).isoformat() for i in reversed(range(rows))]
    pd.DataFrame({
        "Date": dates,
        "Open": [100.0] * rows,
        "High": [101.0] * rows,
        "Low": [99.0] * rows,
        "Close": [100.5] * rows,
        "Volume": [1_000_000] * rows,
    }).to_csv(path, index=False)


def test_fresh_csv_is_not_stale(data_dir):
    _write_prices(data_dir / "AAPL.csv", date.today())

    stale, days_old = is_ticker_stale("AAPL")

    assert stale is False
    assert days_old == 0


def test_csv_older_than_the_limit_is_stale(data_dir):
    _write_prices(data_dir / "AAPL.csv", date.today() - timedelta(days=config.STALE_DAYS + 1))

    stale, days_old = is_ticker_stale("AAPL")

    assert stale is True
    assert days_old == config.STALE_DAYS + 1


def test_csv_exactly_at_the_limit_is_not_stale(data_dir):
    """The comparison is `>`, so the boundary day still trades."""
    _write_prices(data_dir / "AAPL.csv", date.today() - timedelta(days=config.STALE_DAYS))

    stale, _ = is_ticker_stale("AAPL")

    assert stale is False


def test_missing_csv_is_stale(data_dir):
    """Regression guard.

    This returned (False, -1) -- "fresh" -- which let a missing CSV fall
    through to predict_ticker(), where the FileNotFoundError landed in the
    bare `except Exception` in live_trader.get_signals() and produced NO
    signal_log.csv row at all. The ticker silently disappeared from the record
    while the run reported success, and the "csv file missing" branch in
    get_signals() was dead code.
    """
    stale, days_old = is_ticker_stale("NOSUCH")

    assert stale is True
    assert days_old == -1


def test_unreadable_csv_is_stale(data_dir):
    """Never trade on data that could not be parsed."""
    (data_dir / "AAPL.csv").write_text("this is not a csv\x00\x01")

    stale, days_old = is_ticker_stale("AAPL")

    assert stale is True
    assert days_old == 9999


def test_csv_without_a_date_column_is_stale(data_dir):
    pd.DataFrame({"Close": [100.0]}).to_csv(data_dir / "AAPL.csv", index=False)

    stale, days_old = is_ticker_stale("AAPL")

    assert stale is True
    assert days_old == 9999


def test_sentinels_are_distinct(data_dir):
    """get_signals() renders -1 and 9999 as different operator-facing labels."""
    (data_dir / "BAD.csv").write_text("garbage")

    assert is_ticker_stale("MISSING")[1] == -1
    assert is_ticker_stale("BAD")[1] == 9999


# --- sentiment veto --------------------------------------------------------


def test_buy_vetoed_by_negative_sentiment():
    assert apply_sentiment_veto("BUY", -0.5) == ("HOLD", "SENTIMENT_VETO")


def test_sell_vetoed_by_positive_sentiment():
    assert apply_sentiment_veto("SELL", 0.5) == ("HOLD", "SENTIMENT_VETO")


def test_buy_survives_mildly_negative_sentiment():
    signal, note = apply_sentiment_veto("BUY", predictor.VETO_BUY_THRESHOLD)
    assert signal == "BUY"
    assert note == ""


def test_sell_survives_mildly_positive_sentiment():
    signal, note = apply_sentiment_veto("SELL", predictor.VETO_SELL_THRESHOLD)
    assert signal == "SELL"
    assert note == ""


def test_sentiment_veto_only_downgrades_never_upgrades():
    """Good news must never turn a HOLD into a BUY."""
    for score in (-1.0, -0.5, 0.0, 0.5, 1.0):
        assert apply_sentiment_veto("HOLD", score) == ("HOLD", "")


def test_positive_sentiment_does_not_touch_a_buy():
    assert apply_sentiment_veto("BUY", 0.9) == ("BUY", "")


def test_negative_sentiment_does_not_touch_a_sell():
    assert apply_sentiment_veto("SELL", -0.9) == ("SELL", "")


def test_neutral_sentiment_passes_everything_through():
    for signal in ("BUY", "SELL", "HOLD"):
        assert apply_sentiment_veto(signal, 0.0)[0] == signal


# --- earnings veto ---------------------------------------------------------


def test_buy_vetoed_by_large_negative_surprise():
    assert apply_earnings_veto("BUY", -20.0) == ("HOLD", "EARNINGS_VETO")


def test_sell_vetoed_by_large_positive_surprise():
    assert apply_earnings_veto("SELL", 20.0) == ("HOLD", "EARNINGS_VETO")


@pytest.mark.parametrize("surprise", [-15.0, -14.9, 0.0, 15.0])
def test_buy_survives_surprises_within_the_band(surprise):
    assert apply_earnings_veto("BUY", surprise) == ("BUY", "")


@pytest.mark.parametrize("surprise", [15.0, 14.9, 0.0, -15.0])
def test_sell_survives_surprises_within_the_band(surprise):
    assert apply_earnings_veto("SELL", surprise) == ("SELL", "")


def test_no_earnings_data_leaves_the_signal_untouched():
    """Absent data is not evidence against the signal."""
    for signal in ("BUY", "SELL", "HOLD"):
        assert apply_earnings_veto(signal, None) == (signal, "")


def test_earnings_veto_only_downgrades():
    for surprise in (-50.0, 0.0, 50.0):
        assert apply_earnings_veto("HOLD", surprise) == ("HOLD", "")


# --- earnings surprise lookup ----------------------------------------------


@pytest.fixture
def earnings_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(predictor, "EARNINGS_DIR", str(tmp_path))
    return tmp_path


def _write_earnings(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_missing_earnings_file_returns_none(earnings_dir):
    assert get_recent_earnings_surprise("AAPL") is None


def test_recent_surprise_is_returned(earnings_dir):
    _write_earnings(earnings_dir / "AAPL.csv", [
        {"Date": (date.today() - timedelta(days=10)).isoformat(), "Surprise(%)": 12.5},
    ])

    assert get_recent_earnings_surprise("AAPL") == 12.5


def test_surprise_outside_the_window_is_ignored(earnings_dir):
    """A year-old earnings beat says nothing about today's signal."""
    _write_earnings(earnings_dir / "AAPL.csv", [
        {"Date": (date.today() - timedelta(days=365)).isoformat(), "Surprise(%)": 12.5},
    ])

    assert get_recent_earnings_surprise("AAPL", window_days=90) is None


def test_most_recent_report_within_the_window_wins(earnings_dir):
    _write_earnings(earnings_dir / "AAPL.csv", [
        {"Date": (date.today() - timedelta(days=80)).isoformat(), "Surprise(%)": 5.0},
        {"Date": (date.today() - timedelta(days=10)).isoformat(), "Surprise(%)": -8.0},
    ])

    assert get_recent_earnings_surprise("AAPL") == -8.0


def test_malformed_earnings_file_returns_none(earnings_dir):
    (earnings_dir / "AAPL.csv").write_text("not,a,valid\nearnings,file,here")

    assert get_recent_earnings_surprise("AAPL") is None
