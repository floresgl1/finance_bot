"""Tests for rebalancer.py — overweight-position trimming.

The rebalancer sits between the model SELL pass and the model BUY pass and is
the only component that sells shares the model did not ask to sell. Two things
matter most and are covered here:

  1. The trim arithmetic (trigger, target, and the 25% per-session cap).
  2. Skip-list coordination — a ticker the model already sold this session must
     never be trimmed again, or the session double-sells the same position.

Constants are asserted up front so a config change surfaces as an explicit
failure here rather than as silently-rewritten expectations.
"""

import math
from unittest.mock import MagicMock, patch

import pytest

import config
import rebalancer
from rebalancer import run_rebalancer, _TRIGGER_BUFFER, _TARGET_OFFSET


# The hand-computed expectations below assume this cap. If it is retuned the
# arithmetic in these tests must be re-derived deliberately, not auto-adjusted.
def test_constants_match_the_values_these_tests_were_derived_from():
    assert config.MAX_POSITION_PCT == 0.08
    assert _TRIGGER_BUFFER == 0.001
    assert _TARGET_OFFSET == 0.005


TRIGGER = config.MAX_POSITION_PCT + _TRIGGER_BUFFER   # 0.081
TARGET = config.MAX_POSITION_PCT - _TARGET_OFFSET     # 0.075


# --- fixtures / helpers ----------------------------------------------------


def _position(symbol: str, qty: float, current_price: float, avg_entry_price: float = 90.0):
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = str(qty)
    pos.current_price = str(current_price)
    pos.avg_entry_price = str(avg_entry_price)
    return pos


def _api(equity: float, positions: list, order_id: str = "order-1"):
    api = MagicMock()
    api.get_account.return_value.equity = str(equity)
    api.list_positions.return_value = positions
    api.submit_order.return_value.id = order_id
    api.get_position.side_effect = lambda sym: next(
        p for p in positions if p.symbol == sym
    )
    return api


@pytest.fixture
def stub_side_effects():
    """Neutralise every cross-module side effect the rebalancer triggers.

    Yields the patched mocks so individual tests can assert on them.
    """
    with patch("live_trader.cancel_standing_stops", return_value=(True, [])) as cancel, \
         patch("live_trader.send_discord") as discord, \
         patch("signal_logger.log_exit") as log_exit, \
         patch("signal_logger.find_open_entry_order_id", return_value="entry-abc") as find_entry, \
         patch.object(rebalancer, "log_signal") as log_signal:
        yield {
            "cancel": cancel,
            "discord": discord,
            "log_exit": log_exit,
            "find_entry": find_entry,
            "log_signal": log_signal,
        }


# --- trim arithmetic -------------------------------------------------------


def test_trims_overweight_position_to_target(stub_side_effects):
    """100 shares @ $100 on $100k equity = 10% weight; trim back toward 7.5%."""
    equity, price, shares = 100_000.0, 100.0, 100
    api = _api(equity, [_position("AAPL", shares, price)])

    outcomes = run_rebalancer(api)

    # target_value = 7,500 -> target_shares = 75 -> raw sell = 25
    # cap = ceil(100 * 0.25) = 25 -> qty = min(25, 25) = 25
    assert len(outcomes) == 1
    assert outcomes[0]["ticker"] == "AAPL"
    assert outcomes[0]["qty"] == 25
    assert outcomes[0]["status"] == "placed"
    assert outcomes[0]["action"] == "SELL"

    api.submit_order.assert_called_once()
    kwargs = api.submit_order.call_args.kwargs
    assert kwargs["symbol"] == "AAPL"
    assert kwargs["qty"] == 25
    assert kwargs["side"] == "sell"
    assert kwargs["type"] == "market"


def test_quarter_position_cap_binds_on_severely_overweight_position(stub_side_effects):
    """A 20% weight would need a 125-share trim; the 25% cap allows only 50."""
    equity, price, shares = 100_000.0, 100.0, 200
    api = _api(equity, [_position("AAPL", shares, price)])

    outcomes = run_rebalancer(api)

    raw_sell = shares - math.ceil((TARGET * equity) / price)   # 200 - 75 = 125
    cap = math.ceil(shares * 0.25)                             # 50
    assert raw_sell == 125 and cap == 50
    assert outcomes[0]["qty"] == cap


def test_weight_exactly_at_trigger_is_not_trimmed(stub_side_effects):
    """The gate is `<= trigger`, so equality must be left alone."""
    equity, price = 100_000.0, 100.0
    shares = (TRIGGER * equity) / price   # 81 shares = exactly 8.1%
    api = _api(equity, [_position("AAPL", shares, price)])

    outcomes = run_rebalancer(api)

    assert outcomes == []
    api.submit_order.assert_not_called()


def test_weight_one_share_above_trigger_is_trimmed(stub_side_effects):
    equity, price, shares = 100_000.0, 100.0, 82
    api = _api(equity, [_position("AAPL", shares, price)])

    outcomes = run_rebalancer(api)

    assert len(outcomes) == 1
    assert outcomes[0]["qty"] == 7   # 82 - ceil(75) = 7, under the cap of 21


def test_underweight_positions_are_untouched(stub_side_effects):
    api = _api(100_000.0, [_position("AAPL", 10, 100.0)])   # 1% weight

    outcomes = run_rebalancer(api)

    assert outcomes == []
    api.submit_order.assert_not_called()


def test_never_sells_more_shares_than_owned(stub_side_effects):
    """Fractional holdings must not produce a qty above the real position."""
    equity, price, shares = 1_000.0, 100.0, 0.5   # $50 = 5% weight
    api = _api(equity, [_position("AAPL", shares, price)])

    outcomes = run_rebalancer(api)

    for outcome in outcomes:
        assert outcome["qty"] <= shares


# --- skip-list coordination ------------------------------------------------


def test_ticker_already_sold_by_model_is_skipped(stub_side_effects):
    """The core double-sell guard."""
    api = _api(100_000.0, [_position("AAPL", 100, 100.0)])

    outcomes = run_rebalancer(api, sell_executed_tickers=["AAPL"])

    assert outcomes == []
    api.submit_order.assert_not_called()


def test_skip_list_only_suppresses_the_named_ticker(stub_side_effects):
    positions = [
        _position("AAPL", 100, 100.0),   # overweight, on the skip list
        _position("MSFT", 100, 100.0),   # overweight, not on the skip list
    ]
    api = _api(100_000.0, positions)

    outcomes = run_rebalancer(api, sell_executed_tickers=["AAPL"])

    assert [o["ticker"] for o in outcomes] == ["MSFT"]


@pytest.mark.parametrize("skip_arg", [None, []])
def test_empty_skip_list_trims_normally(stub_side_effects, skip_arg):
    api = _api(100_000.0, [_position("AAPL", 100, 100.0)])

    outcomes = run_rebalancer(api, sell_executed_tickers=skip_arg)

    assert len(outcomes) == 1


# --- standing-stop cancellation --------------------------------------------


def test_cancel_stop_is_attempted_before_every_trim(stub_side_effects):
    api = _api(100_000.0, [_position("AAPL", 100, 100.0)])

    run_rebalancer(api)

    stub_side_effects["cancel"].assert_called_once()
    assert stub_side_effects["cancel"].call_args.args[1] == "AAPL"


def test_failed_stop_cancel_skips_the_trim_and_alerts(stub_side_effects):
    """A standing stop that will not cancel must block the trim entirely."""
    stub_side_effects["cancel"].return_value = (False, [])
    api = _api(100_000.0, [_position("AAPL", 100, 100.0)])

    outcomes = run_rebalancer(api)

    api.submit_order.assert_not_called()
    assert len(outcomes) == 1
    assert outcomes[0]["status"] == "skipped"
    assert outcomes[0]["reason"] == config.CANCEL_STOP_FAILED
    assert outcomes[0]["qty"] == 0

    stub_side_effects["discord"].assert_called_once()
    assert config.CANCEL_STOP_FAILED in stub_side_effects["discord"].call_args.args[0]
    stub_side_effects["log_signal"].assert_called_once()


def test_failed_cancel_on_one_ticker_does_not_block_another(stub_side_effects):
    stub_side_effects["cancel"].side_effect = [(False, []), (True, [])]
    positions = [_position("AAPL", 100, 100.0), _position("MSFT", 100, 100.0)]
    api = _api(100_000.0, positions)

    outcomes = run_rebalancer(api)

    by_ticker = {o["ticker"]: o for o in outcomes}
    assert by_ticker["AAPL"]["status"] == "skipped"
    assert by_ticker["MSFT"]["status"] == "placed"


# --- exit logging ----------------------------------------------------------


def test_successful_trim_logs_an_exit_row_with_rebalance_reason(stub_side_effects):
    api = _api(100_000.0, [_position("AAPL", 100, 100.0, avg_entry_price=80.0)])

    run_rebalancer(api)

    stub_side_effects["log_exit"].assert_called_once()
    kwargs = stub_side_effects["log_exit"].call_args.kwargs
    assert kwargs["ticker"] == "AAPL"
    assert kwargs["exit_reason"] == "REBALANCE_TRIM"
    assert kwargs["entry_price"] == 80.0     # avg entry, not current price
    assert kwargs["exit_price"] == 100.0
    assert kwargs["shares"] == 25
    assert kwargs["entry_order_id"] == "entry-abc"


def test_unlinked_entry_order_id_falls_back(stub_side_effects):
    stub_side_effects["find_entry"].return_value = None
    api = _api(100_000.0, [_position("AAPL", 100, 100.0)])

    run_rebalancer(api)

    assert stub_side_effects["log_exit"].call_args.kwargs["entry_order_id"] == "UNLINKED"


def test_avg_entry_price_fetch_failure_falls_back_to_current_price(stub_side_effects):
    api = _api(100_000.0, [_position("AAPL", 100, 100.0)])
    api.get_position.side_effect = Exception("alpaca down")

    run_rebalancer(api)

    assert stub_side_effects["log_exit"].call_args.kwargs["entry_price"] == 100.0


def test_order_failure_records_error_and_logs_no_exit(stub_side_effects):
    """No position closed means no realized P&L row."""
    api = _api(100_000.0, [_position("AAPL", 100, 100.0)])
    api.submit_order.side_effect = Exception("insufficient buying power")

    outcomes = run_rebalancer(api)

    assert len(outcomes) == 1
    assert outcomes[0]["status"] == "error"
    assert "insufficient buying power" in outcomes[0]["reason"]
    stub_side_effects["log_exit"].assert_not_called()


def test_order_failure_on_one_ticker_does_not_abort_the_loop(stub_side_effects):
    positions = [_position("AAPL", 100, 100.0), _position("MSFT", 100, 100.0)]
    api = _api(100_000.0, positions)
    api.submit_order.side_effect = [Exception("rejected"), MagicMock(id="order-2")]

    outcomes = run_rebalancer(api)

    by_ticker = {o["ticker"]: o for o in outcomes}
    assert by_ticker["AAPL"]["status"] == "error"
    assert by_ticker["MSFT"]["status"] == "placed"


# --- infrastructure failure ------------------------------------------------


def test_equity_fetch_failure_returns_empty_without_trading(stub_side_effects):
    api = MagicMock()
    api.get_account.side_effect = Exception("alpaca timeout")

    assert run_rebalancer(api) == []
    api.submit_order.assert_not_called()


@pytest.mark.parametrize("equity", [0.0, -50.0])
def test_non_positive_equity_returns_empty_without_trading(stub_side_effects, equity):
    api = _api(equity, [_position("AAPL", 100, 100.0)])

    assert run_rebalancer(api) == []
    api.submit_order.assert_not_called()


def test_positions_fetch_failure_returns_empty_without_trading(stub_side_effects):
    api = _api(100_000.0, [])
    api.list_positions.side_effect = Exception("alpaca timeout")

    assert run_rebalancer(api) == []
    api.submit_order.assert_not_called()


def test_unparseable_position_is_skipped_not_fatal(stub_side_effects):
    bad = _position("AAPL", 100, 100.0)
    bad.qty = "not-a-number"
    good = _position("MSFT", 100, 100.0)
    api = _api(100_000.0, [bad, good])

    outcomes = run_rebalancer(api)

    assert [o["ticker"] for o in outcomes] == ["MSFT"]


@pytest.mark.parametrize(
    "qty, price",
    [(0, 100.0), (-5, 100.0), (100, 0.0), (100, -10.0)],
)
def test_invalid_shares_or_price_skipped(stub_side_effects, qty, price):
    api = _api(100_000.0, [_position("AAPL", qty, price)])

    assert run_rebalancer(api) == []
    api.submit_order.assert_not_called()


def test_equity_is_refreshed_after_each_successful_trim(stub_side_effects):
    """Later weight checks must use post-trim equity, not the stale snapshot."""
    positions = [_position("AAPL", 100, 100.0), _position("MSFT", 100, 100.0)]
    api = _api(100_000.0, positions)

    run_rebalancer(api)

    # One initial fetch plus one refresh per successful trim.
    assert api.get_account.call_count == 3


def test_equity_refresh_failure_after_trim_is_not_fatal(stub_side_effects):
    positions = [_position("AAPL", 100, 100.0), _position("MSFT", 100, 100.0)]
    api = _api(100_000.0, positions)
    account = MagicMock()
    account.equity = "100000.0"
    api.get_account.side_effect = [account, Exception("timeout"), account]

    outcomes = run_rebalancer(api)

    assert [o["status"] for o in outcomes] == ["placed", "placed"]
