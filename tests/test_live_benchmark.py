"""Tests for live_benchmark.py — the real account against buy-and-hold.

This is the only module that measures money that actually moved, so the ways it
can lie matter more than usual:

  - A benchmark built on different dates or different starting capital is not a
    benchmark, so `compare()` must take both from the account curve itself.
  - Alpaca pads portfolio history with zero-equity rows for sessions before the
    account existed. One leading zero makes the total return infinite.
  - A deposit looks exactly like a spectacular day. The report must refuse to
    give a verdict rather than claim one it cannot support.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from live_benchmark import (
    TRANSFER_MOVE_PCT,
    compare,
    detect_transfers,
    fetch_closes,
    fetch_equity_curve,
    format_discord,
    format_report,
    hold_curve,
    load_closes,
    summarise_curve,
)


# --- helpers ---------------------------------------------------------------


def _dates(n: int, start: str = "2026-06-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _curve(values: list[float], start: str = "2026-06-01") -> pd.Series:
    return pd.Series(values, index=_dates(len(values), start), dtype=float)


def _history(equity: list[float], start: str = "2026-06-01"):
    """A stand-in for alpaca's PortfolioHistory, which exposes `.df`."""
    idx = pd.DatetimeIndex(_dates(len(equity), start)).tz_localize("America/New_York")
    api = MagicMock()
    api.get_portfolio_history.return_value = MagicMock(
        df=pd.DataFrame({"equity": equity}, index=idx)
    )
    return api


def _write_closes(tmp_path, ticker: str, closes: list[float],
                  start: str = "2026-06-01") -> None:
    idx = _dates(len(closes), start)
    pd.DataFrame({"Date": idx.strftime("%Y-%m-%d"), "Close": closes}).to_csv(
        tmp_path / f"{ticker}.csv", index=False
    )


# --- fetch_equity_curve ----------------------------------------------------


def test_equity_curve_is_date_indexed_and_tz_naive():
    """Price CSVs are tz-naive; a tz-aware equity index would align with
    nothing and every benchmark arm would come back empty."""
    curve = fetch_equity_curve(_history([10_000.0, 10_100.0, 10_050.0]))

    assert curve.index.tz is None
    assert len(curve) == 3


def test_leading_zero_equity_rows_are_dropped():
    """Alpaca pads the series with zeroes for sessions before the account was
    funded. Starting from 0 makes total return infinite."""
    curve = fetch_equity_curve(_history([0.0, 0.0, 10_000.0, 10_500.0]))

    assert len(curve) == 2
    assert curve.iloc[0] == 10_000.0


def test_empty_portfolio_history_gives_an_empty_curve():
    api = MagicMock()
    api.get_portfolio_history.return_value = MagicMock(df=pd.DataFrame())

    assert fetch_equity_curve(api).empty


def test_portfolio_history_without_an_equity_column_gives_an_empty_curve():
    api = MagicMock()
    api.get_portfolio_history.return_value = MagicMock(
        df=pd.DataFrame({"profit_loss": [1.0]}, index=_dates(1))
    )

    assert fetch_equity_curve(api).empty


def test_lookback_days_are_passed_through_to_alpaca():
    api = _history([10_000.0, 10_100.0])

    fetch_equity_curve(api, days=180)

    assert api.get_portfolio_history.call_args.kwargs["period"] == "180D"


# --- load_closes -----------------------------------------------------------


def test_missing_ticker_csvs_are_skipped_not_fatal(tmp_path):
    """A benchmark over eleven of twelve names is still informative."""
    _write_closes(tmp_path, "AAPL", [100.0, 101.0])

    closes = load_closes(["AAPL", "NOSUCH"], data_dir=str(tmp_path))

    assert list(closes.columns) == ["AAPL"]


def test_unreadable_csvs_are_skipped(tmp_path):
    _write_closes(tmp_path, "AAPL", [100.0, 101.0])
    (tmp_path / "JUNK.csv").write_text("not,a,price\nfile,at,all\n")

    closes = load_closes(["AAPL", "JUNK"], data_dir=str(tmp_path))

    assert list(closes.columns) == ["AAPL"]


def test_no_readable_tickers_gives_an_empty_frame(tmp_path):
    assert load_closes(["NOSUCH"], data_dir=str(tmp_path)).empty


# --- hold_curve ------------------------------------------------------------


def test_flat_prices_lose_only_the_entry_costs():
    """Slippage and commission are real and must show up. Holding a basket
    that never moves should end slightly below where it started."""
    dates = _dates(5)
    closes = pd.DataFrame({"AAA": [100.0] * 5, "BBB": [50.0] * 5}, index=dates)

    curve = hold_curve(closes, dates, 10_000.0)

    assert curve.iloc[-1] < 10_000.0
    assert curve.iloc[-1] > 9_900.0     # costs are small, not catastrophic


def test_doubling_every_price_roughly_doubles_the_curve():
    dates = _dates(3)
    closes = pd.DataFrame({"AAA": [100.0, 150.0, 200.0],
                           "BBB": [50.0, 75.0, 100.0]}, index=dates)

    curve = hold_curve(closes, dates, 10_000.0)

    assert 19_500.0 < curve.iloc[-1] < 20_000.0


def test_capital_is_split_evenly_across_names():
    """One name doubling while the other is flat should move the basket about
    half as much as the mover did."""
    dates = _dates(2)
    closes = pd.DataFrame({"AAA": [100.0, 200.0], "BBB": [100.0, 100.0]},
                          index=dates)

    curve = hold_curve(closes, dates, 10_000.0)

    assert 14_900.0 < curve.iloc[-1] < 15_000.0


def test_names_with_no_price_on_day_one_are_excluded():
    """Including them would allocate capital to a position never opened, which
    silently understates the benchmark."""
    dates = _dates(3)
    closes = pd.DataFrame({"AAA": [100.0, 100.0, 100.0],
                           "BBB": [None, 50.0, 50.0]}, index=dates)

    curve = hold_curve(closes, dates, 10_000.0)

    # All capital went into AAA, which is flat: only the costs are lost.
    assert 9_900.0 < curve.iloc[-1] < 10_000.0


def test_empty_inputs_give_an_empty_curve():
    assert hold_curve(pd.DataFrame(), _dates(3), 10_000.0).empty
    assert hold_curve(pd.DataFrame({"AAA": [1.0]}, index=_dates(1)),
                      pd.DatetimeIndex([]), 10_000.0).empty


def test_prices_are_forward_filled_over_missing_sessions():
    """A ticker that did not trade on a session must not blank the whole
    basket for that day."""
    dates = _dates(4)
    closes = pd.DataFrame({"AAA": [100.0, None, None, 120.0]}, index=dates)

    curve = hold_curve(closes, dates, 10_000.0)

    assert len(curve) == 4
    assert curve.notna().all()


# --- summarise_curve -------------------------------------------------------


def test_summary_computes_return_and_drawdown():
    stats = summarise_curve(_curve([100.0, 120.0, 90.0, 110.0]))

    assert stats["total_return"] == pytest.approx(10.0)
    assert stats["max_drawdown"] == pytest.approx(-25.0)   # 120 -> 90
    assert stats["n_days"] == 4


def test_summary_of_an_empty_curve_is_zeroed_not_an_error():
    stats = summarise_curve(pd.Series(dtype=float))

    assert stats["n_days"] == 0
    assert stats["total_return"] == 0.0
    assert stats["start_date"] is None


def test_summary_of_a_single_day_has_no_drawdown():
    stats = summarise_curve(_curve([10_000.0]))

    assert stats["max_drawdown"] == 0.0
    assert stats["total_return"] == 0.0


# --- detect_transfers ------------------------------------------------------


def test_a_deposit_sized_jump_is_flagged():
    """A deposit looks exactly like a spectacular day, and the account history
    does not distinguish them."""
    events = detect_transfers(_curve([10_000.0, 10_100.0, 20_000.0]))

    assert len(events) == 1
    assert events[0]["change_pct"] > TRANSFER_MOVE_PCT


def test_ordinary_volatility_is_not_flagged():
    events = detect_transfers(_curve([10_000.0, 10_800.0, 9_900.0, 10_400.0]))

    assert events == []


def test_a_withdrawal_is_flagged_too():
    events = detect_transfers(_curve([10_000.0, 5_000.0]))

    assert len(events) == 1
    assert events[0]["change_pct"] < 0


def test_short_curves_cannot_be_judged():
    assert detect_transfers(_curve([10_000.0])) == []
    assert detect_transfers(pd.Series(dtype=float)) == []


# --- compare ---------------------------------------------------------------


@pytest.fixture
def flat_closes():
    dates = _dates(5)
    return pd.DataFrame({"AAA": [100.0] * 5, "BBB": [100.0] * 5}, index=dates)


def test_hold_arm_starts_from_the_accounts_own_capital(flat_closes):
    """The benchmark has to begin where the account began, or the two curves
    are answering different questions.

    Day one is marked *net of entry costs*, exactly as
    backtest._simulate_buy_and_hold marks it, so the opening value sits a
    fraction below the capital deployed and never above it.
    """
    equity = _curve([25_000.0, 25_100.0, 25_050.0, 25_200.0, 25_300.0])

    result = compare(equity, flat_closes)

    basket = result["arms"]["equal-weight hold"]
    assert basket["start_value"] == pytest.approx(25_000.0, rel=2e-3)
    assert basket["start_value"] < 25_000.0


def test_hold_arm_uses_the_accounts_own_dates(flat_closes):
    equity = _curve([10_000.0] * 3)

    result = compare(equity, flat_closes)

    assert result["arms"]["equal-weight hold"]["n_days"] == 3


def test_delta_is_the_account_minus_the_arm(flat_closes):
    """Positive delta must mean the account won, since the verdict text keys
    off the sign."""
    equity = _curve([10_000.0, 10_500.0, 11_000.0, 11_500.0, 12_000.0])

    result = compare(equity, flat_closes)

    basket = result["arms"]["equal-weight hold"]
    assert basket["delta_pp"] == pytest.approx(
        result["strategy"]["total_return"] - basket["total_return"]
    )
    assert basket["delta_pp"] > 0


def test_spy_arm_is_included_when_available(flat_closes):
    equity = _curve([10_000.0] * 5)
    spy = pd.DataFrame({"SPY": [400.0] * 5}, index=_dates(5))

    result = compare(equity, flat_closes, spy)

    assert "SPY hold" in result["arms"]


def test_compare_on_an_empty_account_curve_returns_no_arms(flat_closes):
    result = compare(pd.Series(dtype=float), flat_closes)

    assert result["arms"] == {}
    assert result["strategy"]["n_days"] == 0


def test_compare_counts_the_names_actually_priced(flat_closes):
    result = compare(_curve([10_000.0] * 5), flat_closes)

    assert result["n_basket_names"] == 2


# --- reporting -------------------------------------------------------------


def test_report_states_a_loss_plainly(flat_closes):
    equity = _curve([10_000.0, 9_800.0, 9_600.0, 9_500.0, 9_400.0])

    report = format_report(compare(equity, flat_closes))

    assert "LOST to holding" in report


def test_report_states_a_win_plainly(flat_closes):
    equity = _curve([10_000.0, 10_500.0, 11_000.0, 11_500.0, 12_000.0])

    report = format_report(compare(equity, flat_closes))

    assert "BEAT holding" in report


def test_report_withholds_a_verdict_when_a_transfer_is_detected(flat_closes):
    """Reporting a number here would be reporting a number about a deposit."""
    equity = _curve([10_000.0, 10_100.0, 25_000.0, 25_100.0, 25_200.0])

    report = format_report(compare(equity, flat_closes))

    assert "WARNING" in report
    assert "no" in report and "verdict" in report
    assert "BEAT holding" not in report
    assert "LOST to holding" not in report


def test_report_handles_no_account_history(flat_closes):
    report = format_report(compare(pd.Series(dtype=float), flat_closes))

    assert "No account equity history" in report


def test_report_is_ascii_only(flat_closes):
    """Run from a Windows terminal, where piped stdout is cp1252 and a stray
    em dash aborts the run after the work is done."""
    equity = _curve([10_000.0, 9_500.0, 12_000.0, 11_000.0, 11_500.0])
    spy = pd.DataFrame({"SPY": [400.0, 380.0, 420.0, 410.0, 415.0]}, index=_dates(5))

    format_report(compare(equity, flat_closes, spy)).encode("ascii")


def test_discord_summary_names_the_winner(flat_closes):
    equity = _curve([10_000.0, 10_500.0, 11_000.0, 11_500.0, 12_000.0])

    message = format_discord(compare(equity, flat_closes))

    assert "beating the basket" in message


def test_discord_summary_names_the_loser(flat_closes):
    equity = _curve([10_000.0, 9_800.0, 9_600.0, 9_500.0, 9_400.0])

    message = format_discord(compare(equity, flat_closes))

    assert "behind the basket" in message


def test_discord_summary_flags_a_transfer_instead_of_a_verdict(flat_closes):
    equity = _curve([10_000.0, 10_100.0, 25_000.0, 25_100.0, 25_200.0])

    message = format_discord(compare(equity, flat_closes))

    assert "transfer" in message
    assert "beating the basket" not in message


def test_discord_summary_handles_no_history(flat_closes):
    message = format_discord(compare(pd.Series(dtype=float), flat_closes))

    assert "no account equity history" in message.lower()


# --- fetch_closes ----------------------------------------------------------
#
# Nothing under data/ is tracked in git, so a CI runner has no price CSVs at
# all. Without a download path the basket arm would come back empty and the
# report would silently omit the one number the module exists to produce.


def _download(frame):
    return patch("yfinance.download", return_value=frame)


def test_downloaded_prices_are_tz_naive():
    """The equity curve is tz-naive; a tz-aware price index aligns with nothing
    and every arm comes back empty."""
    idx = pd.DatetimeIndex(_dates(3)).tz_localize("UTC")
    frame = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=idx)

    with _download(frame):
        closes = fetch_closes(["AAPL"], "2026-06-01", "2026-06-03")

    assert closes.index.tz is None
    assert list(closes.columns) == ["AAPL"]


def test_multiindex_close_columns_are_flattened():
    """yfinance returns a DataFrame under 'Close' when it decides to use
    multi-level columns; indexing it would put a frame in a cell."""
    idx = _dates(2)
    frame = pd.DataFrame({("Close", "AAPL"): [100.0, 110.0]}, index=idx)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)

    with _download(frame):
        closes = fetch_closes(["AAPL"], "2026-06-01", "2026-06-02")

    assert closes["AAPL"].tolist() == [100.0, 110.0]


def test_the_end_date_is_extended_because_yfinance_excludes_it():
    """Without the extra day the final session — the one the comparison ends
    on — is missing."""
    frame = pd.DataFrame({"Close": [100.0]}, index=_dates(1))

    with patch("yfinance.download", return_value=frame) as download:
        fetch_closes(["AAPL"], "2026-06-01", "2026-06-10")

    assert download.call_args.kwargs["end"] == pd.Timestamp("2026-06-11")


def test_a_failed_download_is_skipped_not_fatal():
    with patch("yfinance.download", side_effect=RuntimeError("network")):
        assert fetch_closes(["AAPL"], "2026-06-01", "2026-06-03").empty


def test_an_empty_download_is_skipped():
    with _download(pd.DataFrame()):
        assert fetch_closes(["AAPL"], "2026-06-01", "2026-06-03").empty


# --- price source reporting ------------------------------------------------


def test_report_names_the_price_source(flat_closes):
    """Whether the basket came from tracked CSVs or a live download changes how
    much the number is worth, so it is stated rather than assumed."""
    result = compare(_curve([10_000.0] * 5), flat_closes, price_source="yfinance")

    assert "yfinance" in format_report(result)


def test_report_says_how_to_get_prices_when_there_are_none():
    """The silent-empty-basket case: a CI runner with no CSVs and no --fetch."""
    result = compare(_curve([10_000.0] * 5), pd.DataFrame())

    report = format_report(result)
    assert "no prices available" in report
    assert "--fetch" in report
