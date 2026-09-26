"""Tests for pnl_report.py — realized P&L attribution.

The arithmetic here decides whether the strategy looks profitable, so the
edge cases that would quietly distort a total (no losses, unlinked exits,
malformed rows, partial trims) are covered explicitly.
"""

import csv

import pandas as pd
import pytest

import signal_logger
from pnl_report import (
    attach_confidence,
    compute_summary,
    confidence_tier,
    format_discord,
    format_report,
    group_pnl,
    load_entries,
    load_exits,
)


# --- helpers ---------------------------------------------------------------


def _write_log(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=signal_logger.FIELDNAMES)
        writer.writeheader()
        for row in rows:
            full = {name: "" for name in signal_logger.FIELDNAMES}
            full.update(row)
            writer.writerow(full)


def _exit_row(
    *,
    date="2026-06-01",
    ticker="AAPL",
    entry_order_id="entry-1",
    exit_price=110.0,
    exit_reason="MODEL_SELL",
    shares=10,
    realized_pnl=100.0,
):
    return {
        "date": date,
        "ticker": ticker,
        "row_type": "EXIT",
        "entry_order_id": entry_order_id,
        "exit_timestamp": f"{date}T16:00:00",
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "shares": shares,
        "realized_pnl": realized_pnl,
    }


def _entry_row(
    *,
    date="2026-05-01",
    ticker="AAPL",
    entry_order_id="entry-1",
    confidence=55.0,
    price=100.0,
):
    return {
        "date": date,
        "ticker": ticker,
        "row_type": "ENTRY",
        "model_signal": "BUY",
        "actual_action": "BUY",
        "price": price,
        "qty": 10,
        "confidence": confidence,
        "entry_order_id": entry_order_id,
    }


@pytest.fixture
def log(tmp_path):
    return tmp_path / "signal_log.csv"


# --- loading ---------------------------------------------------------------


def test_missing_log_returns_empty(tmp_path):
    assert load_exits(str(tmp_path / "nope.csv")).empty
    assert load_entries(str(tmp_path / "nope.csv")).empty


def test_log_with_only_entries_yields_no_exits(log):
    _write_log(log, [_entry_row()])
    assert load_exits(str(log)).empty


def test_exit_rows_are_selected_and_typed(log):
    _write_log(log, [_entry_row(), _exit_row()])

    exits = load_exits(str(log))

    assert len(exits) == 1
    assert exits.iloc[0]["realized_pnl"] == 100.0
    assert exits.iloc[0]["shares"] == 10
    assert exits.iloc[0]["ticker"] == "AAPL"


def test_entry_price_is_derived_from_the_exit_row(log):
    """entry = exit - pnl/shares. $110 exit, +$100 over 10 shares -> $100."""
    _write_log(log, [_exit_row(exit_price=110.0, shares=10, realized_pnl=100.0)])

    exits = load_exits(str(log))

    assert exits.iloc[0]["entry_price"] == pytest.approx(100.0)
    assert exits.iloc[0]["return_pct"] == pytest.approx(10.0)


def test_losing_trade_derives_a_negative_return(log):
    _write_log(log, [_exit_row(exit_price=90.0, shares=10, realized_pnl=-100.0)])

    exits = load_exits(str(log))

    assert exits.iloc[0]["entry_price"] == pytest.approx(100.0)
    assert exits.iloc[0]["return_pct"] == pytest.approx(-10.0)


def test_malformed_rows_are_dropped_not_fatal(log):
    """A single unparseable row must not skew every aggregate below it."""
    _write_log(log, [
        _exit_row(realized_pnl=100.0),
        _exit_row(entry_order_id="entry-2", realized_pnl="not-a-number"),
        _exit_row(entry_order_id="entry-3", shares="", realized_pnl=50.0),
        _exit_row(entry_order_id="entry-4", realized_pnl=25.0),
    ])

    exits = load_exits(str(log))

    assert len(exits) == 2
    assert exits["realized_pnl"].sum() == 125.0


def test_zero_share_rows_are_dropped(log):
    """Would divide by zero deriving the entry price."""
    _write_log(log, [_exit_row(shares=0, realized_pnl=0.0), _exit_row(entry_order_id="e2")])

    exits = load_exits(str(log))

    assert len(exits) == 1


def test_since_filter_windows_the_report(log):
    _write_log(log, [
        _exit_row(date="2026-05-01", realized_pnl=100.0),
        _exit_row(date="2026-07-01", entry_order_id="e2", realized_pnl=50.0),
    ])

    exits = load_exits(str(log), since="2026-06-01")

    assert len(exits) == 1
    assert exits.iloc[0]["realized_pnl"] == 50.0


def test_legacy_log_without_row_type_yields_no_exits(tmp_path):
    path = tmp_path / "old.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["date", "ticker", "price"])
        writer.writeheader()
        writer.writerow({"date": "2026-01-01", "ticker": "AAPL", "price": "100"})

    assert load_exits(str(path)).empty


# --- summary arithmetic ----------------------------------------------------


def _frame(pnls: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "realized_pnl": pnls,
        "shares": [10] * len(pnls),
        "ticker": ["AAPL"] * len(pnls),
        "exit_reason": ["MODEL_SELL"] * len(pnls),
        "date": pd.to_datetime(["2026-06-01"] * len(pnls)),
    })


def test_empty_summary_is_all_zero():
    s = compute_summary(pd.DataFrame())

    assert s["trades"] == 0
    assert s["total_pnl"] == 0.0
    assert s["profit_factor"] is None


def test_summary_arithmetic():
    s = compute_summary(_frame([100.0, -50.0, 200.0, -150.0]))

    assert s["trades"] == 4
    assert s["total_pnl"] == 100.0
    assert s["wins"] == 2
    assert s["losses"] == 2
    assert s["win_rate"] == 50.0
    assert s["gross_profit"] == 300.0
    assert s["gross_loss"] == -200.0
    assert s["profit_factor"] == pytest.approx(1.5)
    assert s["avg_win"] == 150.0
    assert s["avg_loss"] == -100.0
    assert s["expectancy"] == 25.0
    assert s["largest_win"] == 200.0
    assert s["largest_loss"] == -150.0


def test_high_win_rate_can_still_lose_money():
    """The exact case win-rate reporting alone cannot see."""
    s = compute_summary(_frame([10.0, 10.0, 10.0, 10.0, -100.0]))

    assert s["win_rate"] == 80.0
    assert s["total_pnl"] == -60.0
    assert s["profit_factor"] < 1.0


def test_profit_factor_is_none_when_there_are_no_losses():
    """Reported as n/a rather than a fabricated infinity."""
    s = compute_summary(_frame([100.0, 50.0]))

    assert s["profit_factor"] is None
    assert s["win_rate"] == 100.0


def test_flat_trades_excluded_from_win_rate_denominator():
    s = compute_summary(_frame([100.0, -100.0, 0.0]))

    assert s["flat"] == 1
    assert s["win_rate"] == 50.0   # 1 win of 2 decided, not of 3
    assert s["trades"] == 3


# --- grouping --------------------------------------------------------------


def test_group_pnl_orders_worst_total_first():
    df = pd.DataFrame({
        "realized_pnl": [100.0, -300.0, 50.0, -20.0],
        "exit_reason": ["TAKE_PROFIT", "REBALANCE_TRIM", "TAKE_PROFIT", "MODEL_SELL"],
        "shares": [10] * 4,
        "ticker": ["AAPL"] * 4,
        "date": pd.to_datetime(["2026-06-01"] * 4),
    })

    grouped = group_pnl(df, "exit_reason")

    assert list(grouped["exit_reason"]) == ["REBALANCE_TRIM", "MODEL_SELL", "TAKE_PROFIT"]
    take_profit = grouped[grouped["exit_reason"] == "TAKE_PROFIT"].iloc[0]
    assert take_profit["total_pnl"] == 150.0
    assert take_profit["trades"] == 2
    assert take_profit["win_rate"] == 100.0


def test_group_pnl_on_empty_frame():
    assert group_pnl(pd.DataFrame(), "exit_reason").empty


def test_group_pnl_on_missing_column():
    assert group_pnl(_frame([1.0]), "nonexistent").empty


# --- confidence tiers ------------------------------------------------------


@pytest.mark.parametrize(
    "confidence, expected",
    [
        (70.0, "large (>=0.65)"),
        (0.70, "large (>=0.65)"),
        (55.0, "normal (0.50-0.65)"),
        (0.55, "normal (0.50-0.65)"),
        (40.0, "small (<0.50)"),
        (0.40, "small (<0.50)"),
        (float("nan"), "unknown"),
    ],
)
def test_confidence_tier_handles_both_scales(confidence, expected):
    """ENTRY rows store confidence as a percentage; config uses 0-1."""
    assert confidence_tier(confidence) == expected


def test_attach_confidence_joins_on_entry_order_id(log):
    _write_log(log, [
        _entry_row(entry_order_id="e1", confidence=70.0),
        _entry_row(entry_order_id="e2", confidence=42.0, ticker="MSFT"),
        _exit_row(entry_order_id="e1", realized_pnl=100.0),
        _exit_row(entry_order_id="e2", ticker="MSFT", realized_pnl=-50.0),
    ])

    joined = attach_confidence(load_exits(str(log)), load_entries(str(log)))

    tiers = dict(zip(joined["ticker"], joined["confidence_tier"]))
    assert tiers["AAPL"] == "large (>=0.65)"
    assert tiers["MSFT"] == "small (<0.50)"


def test_unlinked_exits_are_kept_as_unknown(log):
    """UNLINKED trims are still real money and must not vanish from totals."""
    _write_log(log, [
        _entry_row(entry_order_id="e1", confidence=70.0),
        _exit_row(entry_order_id="e1", realized_pnl=100.0),
        _exit_row(entry_order_id="UNLINKED", ticker="MSFT", realized_pnl=-500.0),
    ])

    joined = attach_confidence(load_exits(str(log)), load_entries(str(log)))

    assert len(joined) == 2
    assert joined["realized_pnl"].sum() == -400.0
    assert "unknown" in set(joined["confidence_tier"])


def test_attach_confidence_with_no_entries(log):
    _write_log(log, [_exit_row()])

    joined = attach_confidence(load_exits(str(log)), pd.DataFrame())

    assert set(joined["confidence_tier"]) == {"unknown"}


# --- report rendering ------------------------------------------------------


def test_report_warns_when_no_stop_losses_are_reconciled(log):
    """The omission biases P&L upward — it must never be silent."""
    _write_log(log, [_entry_row(), _exit_row()])

    report = format_report(load_exits(str(log)), load_entries(str(log)))

    assert "BLIND SPOT" in report
    assert "reconcile_stops.py" in report


def test_report_confirms_coverage_once_stops_are_reconciled(log):
    """With STOP_LOSS_FILL rows present the losing tail IS represented, and
    the report must stop claiming otherwise."""
    _write_log(log, [
        _entry_row(),
        _exit_row(exit_reason="MODEL_SELL", realized_pnl=100.0),
        _exit_row(entry_order_id="e2", exit_reason="STOP_LOSS_FILL",
                  realized_pnl=-250.0),
    ])

    report = format_report(load_exits(str(log)), load_entries(str(log)))

    assert "BLIND SPOT" not in report
    assert "1 STOP_LOSS_FILL row(s) reconciled" in report


def test_stop_loss_fill_is_a_recognised_exit_reason(log):
    _write_log(log, [_exit_row(exit_reason="STOP_LOSS_FILL", realized_pnl=-250.0)])

    report = format_report(load_exits(str(log)), load_entries(str(log)))

    assert "unrecognised exit reason" not in report


def test_reconciled_stops_are_included_in_the_totals(log):
    """The recovered losses must actually move the headline number."""
    _write_log(log, [
        _exit_row(exit_reason="MODEL_SELL", realized_pnl=100.0),
        _exit_row(entry_order_id="e2", exit_reason="STOP_LOSS_FILL",
                  realized_pnl=-250.0),
    ])

    exits = load_exits(str(log))

    assert compute_summary(exits)["total_pnl"] == -150.0


def test_discord_summary_states_stop_coverage(log):
    _write_log(log, [
        _exit_row(exit_reason="STOP_LOSS_FILL", realized_pnl=-250.0),
    ])

    message = format_discord(load_exits(str(log)), load_entries(str(log)))

    assert "Includes 1 reconciled stop-loss exit(s)" in message


def test_empty_report_still_states_the_blind_spot(tmp_path):
    report = format_report(pd.DataFrame(), pd.DataFrame())

    assert "No closed trades" in report
    assert "BLIND SPOT" in report


def test_report_contains_each_section(log):
    _write_log(log, [
        _entry_row(entry_order_id="e1", confidence=70.0),
        _exit_row(entry_order_id="e1", exit_reason="TAKE_PROFIT", realized_pnl=100.0),
        _exit_row(entry_order_id="e2", ticker="MSFT",
                  exit_reason="REBALANCE_TRIM", realized_pnl=-300.0),
    ])

    report = format_report(load_exits(str(log)), load_entries(str(log)))

    assert "OVERALL" in report
    assert "BY EXIT REASON" in report
    assert "BY TICKER" in report
    assert "BY CONFIDENCE TIER" in report
    assert "REBALANCE_TRIM" in report


def test_report_flags_an_unrecognised_exit_reason(log):
    """A new exit path should surface the first time it fires."""
    _write_log(log, [_exit_row(exit_reason="MYSTERY_EXIT")])

    report = format_report(load_exits(str(log)), load_entries(str(log)))

    assert "unrecognised exit reason" in report
    assert "MYSTERY_EXIT" in report


def test_discord_summary_is_compact_and_carries_the_caveat(log):
    _write_log(log, [_entry_row(), _exit_row(realized_pnl=-100.0)])

    message = format_discord(load_exits(str(log)), load_entries(str(log)))

    assert "Realized P&L" in message
    assert "🔴" in message
    assert "stop-loss" in message.lower()
    assert len(message) < 2000   # Discord message limit


def test_discord_summary_on_empty_log():
    message = format_discord(pd.DataFrame(), pd.DataFrame())

    assert "no closed trades" in message.lower()


# --- position_id: confidence and per-position P&L --------------------------
#
# Shapes taken from the live log: AAPL bought once then exited five times
# (four trims + a take-profit); under entry_order_id only the first trim
# linked, so four exits fell into the 'unknown' tier.

import pnl_report
from pnl_report import position_pnl, summarize_positions


def _pos_exit(pid, date, pnl, reason="REBALANCE_TRIM", ticker="AAPL"):
    row = _exit_row(date=date, ticker=ticker, entry_order_id="UNLINKED",
                    exit_reason=reason, realized_pnl=pnl)
    row["position_id"] = pid
    return row


def _opener(pid, confidence, action="BUY", ticker="AAPL", date="2026-06-25"):
    row = _entry_row(date=date, ticker=ticker, entry_order_id=pid, confidence=confidence)
    row["actual_action"] = action
    row["position_id"] = pid
    return row


def _aapl_position():
    return [
        _opener("p1", 70.0),
        _pos_exit("p1", "2026-06-26", -42.25),
        _pos_exit("p1", "2026-06-30", 93.58),
        _pos_exit("p1", "2026-07-01", 143.58),
        _pos_exit("p1", "2026-07-02", 67.85),
        _pos_exit("p1", "2026-07-16", 1423.83, reason="TAKE_PROFIT"),
    ]


def test_every_exit_of_a_position_gets_the_openers_confidence(log):
    _write_log(log, _aapl_position())

    joined = attach_confidence(load_exits(str(log)), load_entries(str(log)))

    assert set(joined["confidence_tier"]) == {"large (>=0.65)"}


def test_adds_do_not_override_the_opening_confidence(log):
    _write_log(log, [
        _opener("p1", 70.0),
        _opener("p1", 40.0, action="ADD_TO_POSITION", date="2026-06-27"),
        _pos_exit("p1", "2026-07-16", 500.0, reason="TAKE_PROFIT"),
    ])

    joined = attach_confidence(load_exits(str(log)), load_entries(str(log)))

    assert joined["confidence"].iloc[0] == 70.0


def test_position_held_before_the_log_stays_unknown(log):
    """Only adds in the log: no opening decision to attribute to."""
    _write_log(log, [
        _opener("bf-NVDA-2026-03-23", 43.4, action="ADD_TO_POSITION", ticker="NVDA"),
        _pos_exit("bf-NVDA-2026-03-23", "2026-04-27", 1285.58, reason="TAKE_PROFIT", ticker="NVDA"),
    ])

    joined = attach_confidence(load_exits(str(log)), load_entries(str(log)))

    assert list(joined["confidence_tier"]) == ["unknown"]


def test_exits_without_position_id_fall_back_to_entry_order_id(log):
    _write_log(log, [
        _opener("p1", 70.0),
        _pos_exit("p1", "2026-07-01", 10.0, reason="MODEL_SELL"),
        _entry_row(entry_order_id="e9", confidence=42.0, ticker="MSFT"),
        _exit_row(entry_order_id="e9", ticker="MSFT", realized_pnl=-5.0),
    ])

    joined = attach_confidence(load_exits(str(log)), load_entries(str(log)))

    tiers = dict(zip(joined["ticker"], joined["confidence_tier"]))
    assert tiers == {"AAPL": "large (>=0.65)", "MSFT": "small (<0.50)"}


def test_position_pnl_sums_trims_and_the_final_sale(log):
    _write_log(log, _aapl_position())

    positions = position_pnl(load_exits(str(log)))

    assert len(positions) == 1
    row = positions.iloc[0]
    assert row["exits"] == 5
    assert row["total_pnl"] == pytest.approx(1686.59)
    assert bool(row["closed"]) is True


def test_position_ending_in_a_trim_is_open(log):
    _write_log(log, [_pos_exit("p1", "2026-07-01", 10.0), _pos_exit("p1", "2026-07-02", 5.0)])

    assert bool(position_pnl(load_exits(str(log))).iloc[0]["closed"]) is False


def test_summary_counts_positions_not_exits(log):
    _write_log(log, _aapl_position() + [
        _pos_exit("p2", "2026-07-03", -30.0, ticker="MSFT"),
        _pos_exit("p2", "2026-07-05", -70.0, reason="MODEL_SELL", ticker="MSFT"),
        _pos_exit("p3", "2026-07-06", 12.0, ticker="JPM"),                 # still open
        _exit_row(date="2026-07-07", ticker="XOM", realized_pnl=5.0),      # no position_id
    ])

    ps = summarize_positions(load_exits(str(log)))

    assert (ps["closed"], ps["open"]) == (2, 1)
    assert (ps["wins"], ps["losses"]) == (1, 1)
    assert ps["avg_pnl"] == pytest.approx((1686.59 - 100.0) / 2)
    assert ps["avg_exits"] == pytest.approx(3.5)
    assert ps["unlinked_exits"] == 1


def test_summary_is_none_before_any_position_ids(log):
    _write_log(log, [_exit_row()])

    assert summarize_positions(load_exits(str(log))) is None


def test_reports_show_the_position_view(log):
    _write_log(log, _aapl_position())
    exits, entries = load_exits(str(log)), load_entries(str(log))

    console = format_report(exits, entries)
    discord = format_discord(exits, entries)

    assert "BY POSITION" in console
    assert "Closed positions    : 1  (0 still open)" in console
    assert "**By position:** 1 closed (0 open)" in discord


def test_reports_omit_the_position_view_without_ids(log):
    _write_log(log, [_exit_row()])
    exits, entries = load_exits(str(log)), load_entries(str(log))

    assert "BY POSITION" not in format_report(exits, entries)
    assert "By position" not in format_discord(exits, entries)


def test_negative_money_puts_the_sign_before_the_dollar():
    """Was rendering 'avg loss $-141.48' in the weekly Discord post."""
    assert pnl_report._money(-141.48) == "-$141.48"
    assert pnl_report._money(1234.5) == "$1,234.50"
