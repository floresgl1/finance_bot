"""Tests for labels.py — the BUY/SELL/HOLD targets the model learns from.

A bug here is invisible downstream: every accuracy, F1 and backtest number is
computed against these labels, so wrong labels produce a confidently wrong
model that looks fine on every metric. The VIX threshold tiers, the
SPY-relative comparison, and the forward-looking window are pinned exactly.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import labels
from labels import _WINDOW, add_labels


# --- helpers ---------------------------------------------------------------


def _frame(closes: list[float], start="2026-01-01") -> pd.DataFrame:
    """A minimal featured frame: a Close series on a business-day index."""
    index = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"Close": closes}, index=index).rename_axis("Date")


def _market(index, *, spy_return: float, vix: float) -> pd.DataFrame:
    """The merged market frame _load_market_data() would return."""
    return pd.DataFrame({
        "Date": index,
        "spy_return_7d": [spy_return] * len(index),
        "vix_close": [vix] * len(index),
    })


@pytest.fixture
def market(monkeypatch):
    """Patch market data with a controllable SPY return and VIX level."""
    def _install(index, *, spy_return=0.0, vix=20.0):
        monkeypatch.setattr(
            labels, "_load_market_data",
            lambda: _market(index, spy_return=spy_return, vix=vix),
        )
    return _install


def _flat_then_move(tail: int, move_pct: float) -> list[float]:
    """Prices where row 0's 7-day forward return is exactly move_pct.

    The jump must land at index _WINDOW: row i's label is derived from
    close[i + _WINDOW] / close[i] - 1, so a move placed any later leaves row 0
    looking at a flat future.
    """
    base = 100.0
    return [base] * _WINDOW + [base * (1 + move_pct)] * tail


# --- VIX-based threshold tiers ---------------------------------------------


@pytest.mark.parametrize(
    "vix, expected_threshold",
    [
        (10.0, 0.010),
        (14.99, 0.010),
        (15.0, 0.015),   # boundary: 15 is NOT "< 15"
        (24.99, 0.015),
        (25.0, 0.020),   # boundary: 25 is NOT "< 25"
        (40.0, 0.020),
    ],
)
def test_vix_threshold_tiers(market, vix, expected_threshold):
    closes = _flat_then_move(10, 0.05)
    df = _frame(closes)
    market(df.index, spy_return=0.0, vix=vix)

    result = add_labels(df)

    assert result["threshold"].iloc[0] == pytest.approx(expected_threshold)


def test_missing_vix_falls_back_to_the_strictest_threshold(market):
    """NaN fails both `< 15` and `< 25`, landing on 2.0% — the conservative
    end, which produces fewer signals rather than more."""
    closes = _flat_then_move(10, 0.05)
    df = _frame(closes)
    market(df.index, spy_return=0.0, vix=float("nan"))

    result = add_labels(df)

    assert result["threshold"].iloc[0] == pytest.approx(0.020)


# --- signal assignment -----------------------------------------------------


def test_outperformance_beyond_threshold_is_buy(market):
    df = _frame(_flat_then_move(10, 0.05))       # +5% forward
    market(df.index, spy_return=0.0, vix=20.0)   # threshold 1.5%

    result = add_labels(df)

    assert result["Signal"].iloc[0] == "BUY"


def test_underperformance_beyond_threshold_is_sell(market):
    df = _frame(_flat_then_move(10, -0.05))      # -5% forward
    market(df.index, spy_return=0.0, vix=20.0)

    result = add_labels(df)

    assert result["Signal"].iloc[0] == "SELL"


def test_small_move_inside_the_band_is_hold(market):
    df = _frame(_flat_then_move(10, 0.005))      # +0.5% vs a 1.5% threshold
    market(df.index, spy_return=0.0, vix=20.0)

    result = add_labels(df)

    assert result["Signal"].iloc[0] == "HOLD"


def test_just_above_threshold_is_buy(market):
    """Threshold behaviour is asserted just either side of the boundary
    rather than exactly on it.

    The comparison is `>=`, but the boundary is not reachable through
    constructed prices: 100 * 1.015 round-trips through pct_change as
    0.014999999999999999, landing under a 0.015 threshold. The mirror case
    (-1.5%) happens to round the other way. Testing the exact boundary would
    therefore assert a float artifact rather than the rule.
    """
    df = _frame(_flat_then_move(10, 0.0151))
    market(df.index, spy_return=0.0, vix=20.0)   # threshold 1.5%

    result = add_labels(df)

    assert result["Signal"].iloc[0] == "BUY"


def test_just_below_threshold_is_hold(market):
    df = _frame(_flat_then_move(10, 0.0149))
    market(df.index, spy_return=0.0, vix=20.0)

    result = add_labels(df)

    assert result["Signal"].iloc[0] == "HOLD"


def test_just_beyond_negative_threshold_is_sell(market):
    df = _frame(_flat_then_move(10, -0.0151))
    market(df.index, spy_return=0.0, vix=20.0)

    result = add_labels(df)

    assert result["Signal"].iloc[0] == "SELL"


def test_just_inside_negative_threshold_is_hold(market):
    df = _frame(_flat_then_move(10, -0.0149))
    market(df.index, spy_return=0.0, vix=20.0)

    result = add_labels(df)

    assert result["Signal"].iloc[0] == "HOLD"


# --- SPY-relative, not absolute --------------------------------------------


def test_a_stock_matching_spy_is_a_hold_despite_a_large_gain(market):
    """The label is relative outperformance. Rising 5% while the market rises
    5% is not an edge, and labelling it BUY would teach the model to chase
    beta."""
    df = _frame(_flat_then_move(10, 0.05))
    market(df.index, spy_return=0.05, vix=20.0)

    result = add_labels(df)

    assert result["Signal"].iloc[0] == "HOLD"


def test_a_falling_stock_can_be_a_buy_if_spy_fell_further(market):
    df = _frame(_flat_then_move(10, -0.02))      # stock -2%
    market(df.index, spy_return=-0.06, vix=20.0)  # SPY -6% -> +4% relative

    result = add_labels(df)

    assert result["Signal"].iloc[0] == "BUY"


def test_a_rising_stock_can_be_a_sell_if_spy_rose_further(market):
    df = _frame(_flat_then_move(10, 0.02))       # stock +2%
    market(df.index, spy_return=0.08, vix=20.0)   # SPY +8% -> -6% relative

    result = add_labels(df)

    assert result["Signal"].iloc[0] == "SELL"


@pytest.mark.parametrize(
    "stock_move, spy_move, expected",
    [
        (0.05, 0.00, "BUY"),
        (0.00, 0.05, "SELL"),
        (0.05, 0.05, "HOLD"),
        (-0.05, -0.05, "HOLD"),
        (0.02, 0.00, "BUY"),     # +2% vs 1.5% threshold
        (0.01, 0.00, "HOLD"),    # +1% inside the band
        (-0.02, 0.00, "SELL"),
        (-0.01, 0.00, "HOLD"),
    ],
)
def test_relative_return_sweep(market, stock_move, spy_move, expected):
    df = _frame(_flat_then_move(10, stock_move))
    market(df.index, spy_return=spy_move, vix=20.0)

    result = add_labels(df)

    assert result["Signal"].iloc[0] == expected


# --- forward window --------------------------------------------------------


def test_final_window_rows_are_dropped(market):
    """The last 7 rows have no future price and cannot be labelled."""
    closes = [100.0] * 30
    df = _frame(closes)
    market(df.index, spy_return=0.0, vix=20.0)

    result = add_labels(df)

    assert len(result) == 30 - _WINDOW


def test_labels_look_forward_not_backward(market):
    """A price that already moved must not label the rows after it.

    Rows 0-2 sit 7 days before the jump, so they see it; rows after the jump
    see a flat future and must be HOLD.
    """
    closes = [100.0] * 10 + [130.0] * 15
    df = _frame(closes)
    market(df.index, spy_return=0.0, vix=20.0)

    result = add_labels(df)

    # Row 0 looks ahead 7 days -- still flat at 100 -- so it is a HOLD.
    assert result["Signal"].iloc[0] == "HOLD"
    # Row 3 looks ahead to index 10, the jump, so it is a BUY.
    assert result["Signal"].iloc[3] == "BUY"
    # A row well after the jump sees a flat future again.
    assert result["Signal"].iloc[14] == "HOLD"


def test_index_is_preserved_as_datetime(market):
    df = _frame([100.0] * 20)
    market(df.index, spy_return=0.0, vix=20.0)

    result = add_labels(df)

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.name == "Date"


def test_original_columns_survive(market):
    df = _frame([100.0] * 20)
    df["SMA_20"] = 99.0
    market(df.index, spy_return=0.0, vix=20.0)

    result = add_labels(df)

    assert "SMA_20" in result.columns
    assert "Close" in result.columns
    assert "Signal" in result.columns


def test_accepts_a_frame_with_date_as_a_column(market):
    """add_labels resets the index when Date is not already a column."""
    df = _frame([100.0] * 20).reset_index()
    market(pd.to_datetime(df["Date"]), spy_return=0.0, vix=20.0)

    result = add_labels(df)

    assert len(result) == 20 - _WINDOW
    assert "Signal" in result.columns


def test_every_row_gets_one_of_the_three_labels(market):
    rng = np.random.default_rng(0)
    closes = list(100 + rng.normal(0, 5, 60).cumsum())
    df = _frame(closes)
    market(df.index, spy_return=0.0, vix=20.0)

    result = add_labels(df)

    assert set(result["Signal"]).issubset({"BUY", "SELL", "HOLD"})
    assert result["Signal"].notna().all()
