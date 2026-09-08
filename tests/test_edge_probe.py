"""Tests for edge_probe.py's exposure and regime-filter machinery.

The probe's conclusions are only worth anything if the arms it compares are
actually the policies they claim to be. Two things are load-bearing here:

  1. `_apply_policy` must express each policy purely as a rewrite of the signal
     frames, without mutating the caller's data. Every arm then runs through the
     same `backtest._simulate`, so a difference between arms cannot be an
     artifact of the simulator.
  2. `_spy_regime` must be lagged. A market-timing rule that reads the close it
     trades on is the easiest way to manufacture an edge that does not exist,
     and the resulting numbers look plausible.
"""

import numpy as np
import pandas as pd
import pytest

import edge_probe
from edge_probe import (
    EXPOSURE_ARMS,
    REGIME_SMA_WINDOW,
    _apply_policy,
    _label_window,
    _relabel,
    _sizing,
    _spy_regime,
    _window_signals,
    summarise_exposure,
    summarise_horizons,
)


# --- fixtures --------------------------------------------------------------


def _signal_frame(signals: list[str], confidences: list[float],
                  start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=len(signals))
    return pd.DataFrame(
        {"Close": np.linspace(100.0, 110.0, len(signals)),
         "Signal": signals,
         "Confidence": confidences},
        index=idx,
    )


@pytest.fixture
def base_data():
    """Three sessions: a BUY, a HOLD, and a SELL, at distinguishable sizes."""
    return {
        "AAA": _signal_frame(["BUY", "HOLD", "SELL"], [0.42, 0.51, 0.88]),
        "BBB": _signal_frame(["HOLD", "BUY", "BUY"], [0.60, 0.37, 0.95]),
    }


@pytest.fixture
def risk_on(base_data):
    idx = base_data["AAA"].index
    return pd.Series([True, False, True], index=idx)


# --- policy: shipped -------------------------------------------------------


def test_shipped_policy_changes_nothing(base_data, risk_on):
    """The control arm has to be the untouched model output, or the whole
    comparison is against a strawman."""
    out = _apply_policy(base_data, "shipped", risk_on)

    for ticker, frame in out.items():
        pd.testing.assert_frame_equal(frame, base_data[ticker])


def test_policies_do_not_mutate_the_caller(base_data, risk_on):
    """Every arm is applied to the same base frames in turn; in-place edits
    would leak the previous arm's policy into the next one."""
    before = {t: df.copy() for t, df in base_data.items()}

    for policy, *_ in EXPOSURE_ARMS.values():
        _apply_policy(base_data, policy, risk_on)

    for ticker, frame in base_data.items():
        pd.testing.assert_frame_equal(frame, before[ticker])


# --- policy: full_size -----------------------------------------------------


def test_full_size_raises_only_buy_confidence(base_data, risk_on):
    """Position size is `Confidence * MAX_POSITION_PCT`, so conf 1.0 is a
    full-size buy. Non-BUY rows never size anything and must be left alone."""
    out = _apply_policy(base_data, "full_size", risk_on)["AAA"]

    assert out.loc[out["Signal"] == "BUY", "Confidence"].eq(1.0).all()
    assert out.loc[out["Signal"] == "HOLD", "Confidence"].iloc[0] == 0.51
    assert out.loc[out["Signal"] == "SELL", "Confidence"].iloc[0] == 0.88


def test_full_size_leaves_signals_untouched(base_data, risk_on):
    out = _apply_policy(base_data, "full_size", risk_on)["BBB"]

    assert list(out["Signal"]) == ["HOLD", "BUY", "BUY"]


# --- policy: regime_only ---------------------------------------------------


def test_regime_only_ignores_the_model_entirely(base_data, risk_on):
    """This arm is the control: a 200-day SMA rule with no model in it. If it
    kept any model signal, it could not answer whether the model adds anything."""
    out = _apply_policy(base_data, "regime_only", risk_on)["AAA"]

    assert list(out["Signal"]) == ["BUY", "HOLD", "BUY"]
    assert out["Confidence"].eq(1.0).all()


def test_regime_only_is_identical_across_tickers(base_data, risk_on):
    """Risk-on is a market-wide state, so every name must move together —
    that is what makes the arm equal-weight buy-and-hold while risk-on."""
    out = _apply_policy(base_data, "regime_only", risk_on)

    assert list(out["AAA"]["Signal"]) == list(out["BBB"]["Signal"])


# --- policy: regime_plus_model ---------------------------------------------


def test_regime_plus_model_holds_the_basket_when_risk_on(base_data, risk_on):
    out = _apply_policy(base_data, "regime_plus_model", risk_on)["AAA"]

    assert out["Signal"].iloc[0] == "BUY"
    assert out["Confidence"].iloc[0] == 1.0
    assert out["Signal"].iloc[2] == "BUY"


def test_regime_plus_model_defers_to_the_model_when_risk_off(base_data, risk_on):
    """Risk-off is the only condition where the model gets a vote, so its
    signal and its confidence must both survive untouched."""
    out = _apply_policy(base_data, "regime_plus_model", risk_on)["AAA"]

    assert out["Signal"].iloc[1] == "HOLD"
    assert out["Confidence"].iloc[1] == 0.51


def test_regime_plus_model_upgrades_a_sell_to_buy_when_risk_on(base_data, risk_on):
    """A model SELL on a risk-on day is overridden. That is the arm's whole
    premise and it should be visible, not incidental."""
    out = _apply_policy(base_data, "regime_plus_model", risk_on)["AAA"]

    assert base_data["AAA"]["Signal"].iloc[2] == "SELL"
    assert out["Signal"].iloc[2] == "BUY"


# --- regime series alignment ------------------------------------------------


def test_dates_missing_from_the_regime_series_count_as_risk_off(base_data):
    """A gap in SPY's history must not silently become 'invested'. Failing
    safe here means holding cash, not holding the basket."""
    partial = pd.Series([True], index=base_data["AAA"].index[:1])

    out = _apply_policy(base_data, "regime_only", partial)["AAA"]

    assert list(out["Signal"]) == ["BUY", "HOLD", "HOLD"]


def test_unknown_policy_is_rejected(base_data, risk_on):
    with pytest.raises(ValueError, match="unknown policy"):
        _apply_policy(base_data, "no_such_policy", risk_on)


def test_every_declared_arm_uses_a_supported_policy(base_data, risk_on):
    """EXPOSURE_ARMS and _apply_policy are edited separately; a typo in the
    table would otherwise surface only halfway through a long probe run."""
    for label, (policy, _pos, _total) in EXPOSURE_ARMS.items():
        _apply_policy(base_data, policy, risk_on)   # must not raise


def test_arm_caps_are_sane():
    for label, (_policy, max_pos, max_total) in EXPOSURE_ARMS.items():
        assert 0 < max_pos <= 1.0, label
        assert 0 < max_total <= 1.0, label
        assert max_pos <= max_total, f"{label}: position cap exceeds total cap"


# --- _spy_regime ------------------------------------------------------------


def _write_spy(tmp_path, closes: list[float]) -> None:
    idx = pd.bdate_range(start="2020-01-01", periods=len(closes))
    pd.DataFrame({"Date": idx.strftime("%Y-%m-%d"), "Close": closes}).to_csv(
        tmp_path / "SPY.csv", index=False
    )


def test_regime_is_risk_off_through_the_sma_warmup(monkeypatch, tmp_path):
    """Before 200 sessions there is no SMA. NaN comparisons are False, which is
    the right default: no history means no reason to be invested."""
    monkeypatch.setattr(edge_probe, "PROBE_MARKET_DIR", str(tmp_path))
    _write_spy(tmp_path, list(np.linspace(100.0, 200.0, 260)))

    regime = _spy_regime()

    assert not regime.iloc[:REGIME_SMA_WINDOW].any()
    assert regime.iloc[-1]


def test_regime_crossing_is_lagged_one_session(monkeypatch, tmp_path):
    """The day SPY first closes above its SMA, the strategy is still out. It
    can only act on the following session — otherwise the rule is reading a
    price it has not seen yet."""
    monkeypatch.setattr(edge_probe, "PROBE_MARKET_DIR", str(tmp_path))
    closes = [100.0] * 250 + [200.0] * 20
    _write_spy(tmp_path, closes)

    regime = _spy_regime()

    assert not regime.iloc[250]    # the crossing day itself
    assert regime.iloc[251]        # acted on the next session


def test_flat_market_is_never_risk_on(monkeypatch, tmp_path):
    """Close equal to the SMA is not above it; a dead-flat tape must not read
    as an uptrend."""
    monkeypatch.setattr(edge_probe, "PROBE_MARKET_DIR", str(tmp_path))
    _write_spy(tmp_path, [100.0] * 260)

    assert not _spy_regime().any()


def test_regime_index_is_tz_naive(monkeypatch, tmp_path):
    """Signal frames are tz-naive; a tz-aware regime index reindexes to all-NaN
    and silently turns every day risk-off."""
    monkeypatch.setattr(edge_probe, "PROBE_MARKET_DIR", str(tmp_path))
    idx = pd.bdate_range(start="2020-01-01", periods=260, tz="UTC")
    pd.DataFrame({"Date": idx, "Close": np.linspace(100.0, 200.0, 260)}).to_csv(
        tmp_path / "SPY.csv", index=False
    )

    regime = _spy_regime()

    assert regime.index.tz is None
    assert regime.iloc[-1]


# --- _sizing ----------------------------------------------------------------


def test_sizing_overrides_and_restores_the_caps():
    import backtest as bt

    before = (bt.MAX_POSITION_PCT, bt.MAX_TOTAL_EXPOSURE)

    with _sizing(0.05, 0.99):
        assert bt.MAX_POSITION_PCT == 0.05
        assert bt.MAX_TOTAL_EXPOSURE == 0.99

    assert (bt.MAX_POSITION_PCT, bt.MAX_TOTAL_EXPOSURE) == before


def test_sizing_restores_the_caps_after_an_exception():
    """A failed arm must not leave its caps behind for the next one — that
    would silently reattribute one policy's results to another."""
    import backtest as bt

    before = (bt.MAX_POSITION_PCT, bt.MAX_TOTAL_EXPOSURE)

    with pytest.raises(RuntimeError):
        with _sizing(0.05, 0.99):
            raise RuntimeError("arm blew up")

    assert (bt.MAX_POSITION_PCT, bt.MAX_TOTAL_EXPOSURE) == before


# --- _window_signals --------------------------------------------------------


def test_window_signals_declines_a_thin_training_set(capsys):
    """Fewer than 500 rows is not a model, and silently training on them would
    put a meaningless arm in the results table."""
    idx = pd.bdate_range(start="2024-01-01", periods=50)
    combined = pd.DataFrame(
        {"Volatility": np.linspace(0.1, 0.2, 50), "Signal": ["BUY"] * 50},
        index=idx,
    )

    result = _window_signals({}, combined, ["Volatility"],
                             pd.Timestamp("2024-06-01"), pd.Timestamp("2024-07-01"),
                             lookback_years=3, label="thin window")

    assert result is None
    assert "skipping" in capsys.readouterr().out


# --- summarise_exposure -----------------------------------------------------


def _exposure_results(shipped: list[float], full: list[float],
                      only: list[float], plus: list[float]) -> dict:
    windows = {}
    for i, deltas in enumerate(zip(shipped, full, only, plus)):
        windows[f"w{i}"] = {"arms": {
            "shipped":             {"delta_pp": deltas[0]},
            "full-size buys":      {"delta_pp": deltas[1]},
            "full-size, 100% cap": {"delta_pp": deltas[1]},
            "regime filter only":  {"delta_pp": deltas[2]},
            "regime + model":      {"delta_pp": deltas[3]},
        }}
    return windows


def test_summary_reports_no_windows():
    assert "No windows" in summarise_exposure({})


def test_summary_counts_wins_and_means():
    results = _exposure_results([1.0, -3.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0])

    out = summarise_exposure(results)

    assert "1/2" in out          # shipped won one of two
    assert "-1.00pp" in out      # mean of +1.0 and -3.0


def test_summary_calls_out_that_sizing_up_hurts():
    """The finding that matters: if more exposure loses more, the problem is
    which names it buys, not how much of them."""
    results = _exposure_results([-3.7, -3.7], [-7.2, -7.2], [0.0, 0.0], [0.0, 0.0])

    out = summarise_exposure(results)

    assert "WORSE" in out
    assert "selection, not exposure" in out


def test_summary_credits_exposure_when_sizing_up_helps():
    results = _exposure_results([-3.7, -3.7], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0])

    out = summarise_exposure(results)

    assert "Sizing up helps" in out


def test_summary_calls_out_a_model_that_adds_nothing():
    results = _exposure_results([0.0, 0.0], [0.0, 0.0], [-3.0, -3.0], [-5.0, -5.0])

    out = summarise_exposure(results)

    assert "adds nothing" in out


def test_summary_credits_a_model_that_beats_the_bare_filter():
    results = _exposure_results([0.0, 0.0], [0.0, 0.0], [-5.0, -5.0], [-1.0, -1.0])

    out = summarise_exposure(results)

    assert "Worth pursuing as an overlay" in out


def test_summary_is_ascii_only():
    """These probes are run from a Windows terminal, where piped stdout is
    cp1252 and a stray em dash aborts the run after the work is done."""
    results = _exposure_results([-3.7, 1.0], [-7.2, -7.2], [-3.6, -3.6], [-5.0, -5.0])

    summarise_exposure(results).encode("ascii")   # must not raise


# --- label horizon ---------------------------------------------------------
#
# labels._WINDOW drives three things at once: the SPY forward return, the stock
# forward return, and how many unlabelable tail rows get dropped. They have to
# move together or the label compares returns over mismatched windows.


def test_label_window_sets_and_restores_the_module_constant():
    import labels

    before = labels._WINDOW

    with _label_window(21) as active:
        assert active == 21
        assert labels._WINDOW == 21

    assert labels._WINDOW == before


def test_label_window_restores_after_an_exception():
    """A horizon that blows up must not leave its window behind for the next
    one -- that would silently mislabel every later horizon."""
    import labels

    before = labels._WINDOW

    with pytest.raises(RuntimeError):
        with _label_window(14):
            raise RuntimeError("build failed")

    assert labels._WINDOW == before


def test_label_window_of_none_is_a_no_op():
    import labels

    with _label_window(None) as active:
        assert active == labels._WINDOW == 7


def _labelled_frame(n: int = 40, end: float = 140.0) -> pd.DataFrame:
    """A frame shaped like add_labels() output, with a steady climb.

    `end` sets how steep the climb is, which is what decides whether a given
    threshold is crossed. The default clears even a widened band; the scale
    test below uses a gentler slope so widening can actually bite.
    """
    idx = pd.bdate_range(start="2024-01-01", periods=n)
    return pd.DataFrame(
        {"Close": np.linspace(100.0, end, n),
         "spy_return_7d": [0.0] * n,
         "threshold": [0.01] * n,
         "Signal": ["HOLD"] * n},
        index=idx,
    )


def test_relabel_defaults_to_the_active_label_window():
    """The spy_return column holds a forward return over whatever window
    produced it; a hardcoded 7 would compare mismatched horizons."""
    frame = _labelled_frame()

    with _label_window(21):
        long_window = _relabel(frame, 1.0)
    short_window = _relabel(frame, 1.0, window=3)

    assert not long_window["Signal"].equals(short_window["Signal"])


def test_relabel_honours_an_explicit_window():
    frame = _labelled_frame()

    explicit = _relabel(frame, 1.0, window=21)
    with _label_window(21):
        implicit = _relabel(frame, 1.0)

    pd.testing.assert_series_equal(explicit["Signal"], implicit["Signal"])


def test_a_wider_threshold_scale_produces_more_holds():
    # A ~2.5% forward return over the window: above the 1% band, below 5x it.
    frame = _labelled_frame(end=114.0)

    narrow = _relabel(frame, 1.0, window=7)
    wide = _relabel(frame, 5.0, window=7)

    assert (wide["Signal"] == "HOLD").sum() > (narrow["Signal"] == "HOLD").sum()


# --- summarise_horizons ----------------------------------------------------


def _horizon_results(rows: dict) -> dict:
    """rows: {horizon_days: (mean_delta_pp, windows_beating_hold)}"""
    return {
        f"horizon_{days}": {
            "horizon_days": days,
            "mean_delta_pp": mean,
            "windows_beating_hold": wins,
            "n_windows": 4,
        }
        for days, (mean, wins) in rows.items()
    }


def test_horizon_summary_reports_nothing_to_summarise():
    assert "No horizon" in summarise_horizons({})


def test_horizon_summary_says_when_no_window_beats_holding():
    """The expected outcome given findings A, E and F, and the one most likely
    to be glossed over."""
    out = summarise_horizons(_horizon_results({3: (-5.0, 1), 7: (-3.7, 1)}))

    assert "No horizon beats holding" in out


def test_horizon_summary_flags_a_better_window_than_the_shipped_one():
    out = summarise_horizons(_horizon_results({7: (-3.7, 1), 21: (2.5, 3)}))

    assert "21-day labels beat holding" in out
    assert "walk-forward confirmation" in out


def test_horizon_summary_credits_the_shipped_window_when_it_wins():
    out = summarise_horizons(_horizon_results({7: (1.5, 3), 21: (-2.0, 1)}))

    assert "shipped 7-day horizon is the best" in out


def test_horizon_summary_orders_best_first():
    out = summarise_horizons(_horizon_results({3: (-8.0, 0), 7: (-1.0, 2), 14: (-4.0, 1)}))

    body = out.split("-" * 68)[1]
    assert body.index("7 days") < body.index("14 days") < body.index("3 days")


def test_horizon_summary_is_ascii_only():
    out = summarise_horizons(_horizon_results({3: (-8.0, 0), 7: (-1.0, 2)}))

    out.encode("ascii")


# --- window sets -----------------------------------------------------------
#
# REGIME_WINDOWS is half drawdowns by construction, which flatters any strategy
# that holds less stock. BROAD_WINDOWS is the control, so its shape matters:
# a gap or an overlap would quietly reintroduce the same weighting problem.


def test_broad_windows_are_chronological():
    from edge_probe import BROAD_WINDOWS

    for name, (start, end) in BROAD_WINDOWS.items():
        assert pd.Timestamp(start) < pd.Timestamp(end), name


def test_broad_windows_do_not_overlap():
    from edge_probe import BROAD_WINDOWS

    spans = sorted((pd.Timestamp(s), pd.Timestamp(e))
                   for s, e in BROAD_WINDOWS.values())
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        assert prev_end < next_start


def test_broad_windows_leave_room_for_the_training_lookback():
    """Each window trains on the three years before it, and the probe history
    is ten years. A window starting too early silently trains on less."""
    from edge_probe import BROAD_WINDOWS

    earliest = min(pd.Timestamp(s) for s, _ in BROAD_WINDOWS.values())
    assert earliest >= pd.Timestamp("2019-09-01")


def test_broad_windows_are_mostly_not_drawdowns():
    """The point of the set: bear periods in roughly the proportion they
    occurred, rather than two out of four."""
    from edge_probe import BROAD_WINDOWS

    drawdowns = {"covid crash 2020", "bear 2022"}
    assert len(BROAD_WINDOWS) >= 8
    assert len(drawdowns) / len(BROAD_WINDOWS) < 0.3
