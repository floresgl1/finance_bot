"""Tests for backfill_positions.py — replaying history into positions.

The NVDA fixture is real signal_log.csv rows (2026-03-23 -> 2026-09-23). It
contains every case the replay has to get right: a position held before the
log began, trims that orphaned exits under entry_order_id, a model SELL
logged as both an ENTRY and an EXIT row, and a sale from one position that
entry_order_id linked to a BUY of an earlier one.
"""

import csv
import io

import pytest

import backfill_positions
from backfill_positions import assign_positions, compare_with_broker, main
from signal_logger import FIELDNAMES, LEGACY_FIELDNAMES


NVDA_ROWS = """\
2026-03-23,NVDA,REBALANCER,178.0403,28,0.0,2026-03-30,REBALANCER_SELL,167.035,NEUTRAL,,,,,,,,,,
2026-03-24,NVDA,REBALANCER,175.08,21,0.0,2026-03-31,REBALANCER_SELL,170.99,NEUTRAL,,,,,,,,,,
2026-03-25,NVDA,REBALANCER,180.365,16,0.0,2026-04-01,REBALANCER_SELL,177.1353,NEUTRAL,,,,,,,,,,
2026-03-26,NVDA,REBALANCER,173.795,4,0.0,2026-04-02,REBALANCER_SELL,176.0501,NEUTRAL,,,,,,,,,,
2026-03-27,NVDA,HOLD,168.86,0,69.4,2026-04-03,HOLD,177.39,MISSED GAIN,,,,,,,,,,
2026-04-08,NVDA,BUY,180.87,2,43.4,2026-04-15,ADD_TO_POSITION,199.2899,WIN,,,,,,,,,,
2026-04-14,NVDA,REBALANCER,193.105,3,0.0,2026-04-21,REBALANCER_SELL,199.88,NEUTRAL,,,,,,,,,,
2026-04-20,NVDA,STOP_BACKFILL,179.061,42,0.0,2026-04-27,STOP_BACKFILL,212.66,NEUTRAL,,,,,,,,,,
2026-04-27,NVDA,,,,,,,,,,,,EXIT,UNLINKED,2026-04-27T15:01:38,209.67,TAKE_PROFIT,42,1285.58
2026-05-15,NVDA,BUY,225.7,49,37.2,2026-05-22,BUY,217.665,LOSS,Volume_Ratio,Return_60d,BB_lower,ENTRY,e9e32026,,,,,
2026-05-19,NVDA,,,,,,,,,,,,EXIT,e9e32026,2026-05-19T15:03:39,221.23,REBALANCE_TRIM,11,-52.36
2026-05-22,NVDA,STOP_BACKFILL,225.99,38,0.0,2026-05-29,STOP_BACKFILL,215.84,NEUTRAL,,,,ENTRY,,,,,,
2026-05-22,NVDA,SELL,216.74,38,39.8,2026-05-29,SELL,215.85,NEUTRAL,MACD_signal,Stock_vs_Sector,Sector_Return_20d,ENTRY,,,,,,
2026-05-22,NVDA,,,,,,,,,,,,EXIT,UNLINKED,2026-05-22T15:03:32,216.74,MODEL_SELL,38,-351.5
2026-06-12,NVDA,BUY,205.57,107,60.8,2026-06-19,BUY,210.69,WIN,Rel_Strength,Volume_Ratio,Return_60d,ENTRY,a42ce8f5,,,,,
2026-06-15,NVDA,,,,,,,,,,,,EXIT,a42ce8f5,2026-06-15T17:16:15,212.28,REBALANCE_TRIM,27,176.1233
2026-06-16,NVDA,STOP_BACKFILL,205.7569,80,0.0,2026-06-23,STOP_BACKFILL,202.635,NEUTRAL,,,,ENTRY,,,,,,
2026-06-16,NVDA,,,,,,,,,,,,EXIT,UNLINKED,2026-06-16T16:51:07,209.0709,REBALANCE_TRIM,20,66.2797
2026-06-16,NVDA,BUY,209.05,0,49.9,2026-06-23,REBALANCER_TICKERS_SKIP,202.63,LOSS,,,,ENTRY,,,,,,
2026-06-17,NVDA,STOP_BACKFILL,205.7569,60,0.0,2026-06-24,STOP_BACKFILL,198.43,NEUTRAL,,,,ENTRY,,,,,,
2026-06-17,NVDA,,,,,,,,,,,,EXIT,UNLINKED,2026-06-17T15:27:11,206.261,REBALANCE_TRIM,15,7.5613
2026-06-18,NVDA,STOP_BACKFILL,205.7569,45,0.0,2026-06-25,STOP_BACKFILL,195.14,NEUTRAL,,,,ENTRY,,,,,,
2026-06-18,NVDA,,,,,,,,,,,,EXIT,UNLINKED,2026-06-18T15:23:05,209.1,REBALANCE_TRIM,5,16.7154
2026-06-22,NVDA,STOP_BACKFILL,205.7569,40,0.0,2026-06-29,STOP_BACKFILL,194.31,NEUTRAL,,,,ENTRY,,,,,,
2026-06-24,NVDA,BUY,200.24,4,50.5,2026-07-01,ADD_TO_POSITION,198.09,NEUTRAL,Return_60d,Rel_Strength,Volume_Ratio,ENTRY,e6407ac5,,,,,
2026-06-25,NVDA,BUY,194.75,0,74.6,2026-07-02,INSUFFICIENT_EQUITY,193.18,NEUTRAL,,,,ENTRY,,,,,,
2026-06-26,NVDA,BUY,193.88,1,60.2,2026-07-03,ADD_TO_POSITION,194.83,NEUTRAL,RSI_14,Volume_Ratio,MACD_signal,ENTRY,934d3086,,,,,
2026-07-01,NVDA,BUY,195.8,1,64.0,2026-07-08,ADD_TO_POSITION,200.2,WIN,Rel_Strength,RSI_14,BB_upper,ENTRY,880de4de,,,,,
2026-07-10,NVDA,,,,,,,,,,,,EXIT,880de4de,2026-07-10T14:24:48,206.615,REBALANCE_TRIM,4,7.1838
2026-07-13,NVDA,STOP_BACKFILL,204.7297,42,0.0,2026-07-20,STOP_BACKFILL,203.28,NEUTRAL,,,,ENTRY,,,,,,
2026-07-27,NVDA,BUY,198.75,3,41.6,2026-08-03,ADD_TO_POSITION,207.77,WIN,MACD_signal,Daily_Return,Volume_Ratio,ENTRY,646374ed,,,,,
2026-08-03,NVDA,SELL,205.64,45,35.2,2026-08-10,SELL,219.775,LOSS,Daily_Return,Return_60d,Return_20d,ENTRY,,,,,,
2026-08-03,NVDA,,,,,,,,,,,,EXIT,646374ed,2026-08-03T14:37:31,205.64,MODEL_SELL,45,58.821
2026-08-04,NVDA,BUY,210.82,80,40.1,2026-08-11,BUY,218.39,WIN,Volume_Ratio,Volatility,Sector_Return_5d,ENTRY,82594a23,,,,,
2026-08-05,NVDA,,,,,,,,,,,,EXIT,82594a23,2026-08-05T14:06:40,221.375,REBALANCE_TRIM,20,216.5
2026-08-06,NVDA,STOP_BACKFILL,210.55,60,0.0,2026-08-13,STOP_BACKFILL,225.045,NEUTRAL,,,,ENTRY,,,,,,
2026-08-06,NVDA,,,,,,,,,,,,EXIT,934d3086,2026-08-06T14:05:51,221.73,REBALANCE_TRIM,15,167.7
2026-08-13,NVDA,STOP_BACKFILL,210.55,45,0.0,2026-08-20,STOP_BACKFILL,216.04,NEUTRAL,,,,ENTRY,,,,,,
2026-08-13,NVDA,SELL,224.09,45,39.3,2026-08-20,SELL,216.01,WIN,RSI_14,Daily_Return,MACD,ENTRY,,,,,,
2026-08-13,NVDA,,,,,,,,,,,,EXIT,e6407ac5,2026-08-13T15:00:47,224.09,MODEL_SELL,45,609.3
2026-09-18,NVDA,BUY,219.63,15,37.1,2026-09-25,BUY,224.11,WIN,Volume_Ratio,Return_60d,Volatility,ENTRY,6dd13f71,,,,,
2026-09-21,NVDA,BUY,227.45,26,51.2,2026-09-28,ADD_TO_POSITION,,,Volume_Ratio,Volatility,Return_60d,ENTRY,8e052289,,,,,
"""


def _parse(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text), fieldnames=LEGACY_FIELDNAMES))


def _row(date_, ticker="AAPL", **fields) -> dict:
    row = {name: "" for name in FIELDNAMES}
    row.update(date=date_, ticker=ticker, **fields)
    return row


def _buy(date_, qty, order_id, action="BUY", **kw):
    return _row(date_, row_type="ENTRY", actual_action=action, qty=str(qty),
                entry_order_id=order_id, **kw)


def _exit(date_, shares, reason="REBALANCE_TRIM", ts=None, **kw):
    return _row(date_, row_type="EXIT", exit_reason=reason, shares=str(shares),
                exit_timestamp=ts or f"{date_}T15:30:00", **kw)


# --- the real NVDA history -------------------------------------------------


@pytest.fixture
def nvda():
    rows = _parse(NVDA_ROWS)
    ids, positions, flags = assign_positions(rows)
    return rows, ids, positions, flags


def test_nvda_replays_into_its_real_positions(nvda):
    rows, ids, positions, flags = nvda
    assert [p["opened"] for p in positions] == [
        "2026-03-23", "2026-05-15", "2026-06-12", "2026-08-04", "2026-09-18",
    ]
    assert [p["closed"] for p in positions] == [
        "2026-04-27", "2026-05-22", "2026-08-03", "2026-08-13", None,
    ]


def test_nvda_only_flag_is_the_position_held_before_the_log(nvda):
    _, _, _, flags = nvda
    assert [(f["date"], f["kind"]) for f in flags] == [("2026-03-23", "EXIT_WITHOUT_POSITION")]


def test_every_stop_backfill_checkpoint_agrees_with_the_replay(nvda):
    _, _, _, flags = nvda
    assert not any(f["kind"] == "CHECKPOINT_MISMATCH" for f in flags)


def test_orphaned_trims_join_their_position(nvda):
    """06-16..06-18 trims were UNLINKED under entry_order_id."""
    rows, ids, _, _ = nvda
    june = {r["date"]: pid for r, pid in zip(rows, ids)
            if r["row_type"] == "EXIT" and "2026-06-12" <= r["date"] <= "2026-06-30"}
    assert set(june.values()) == {"a42ce8f5"}
    assert len(june) == 4


def test_cross_position_link_is_corrected(nvda):
    """entry_order_id credited the 08-13 +$609 sale to a June add (e6407ac5)."""
    rows, ids, _, _ = nvda
    sale = next(pid for r, pid in zip(rows, ids)
                if r["date"] == "2026-08-13" and r["exit_reason"] == "MODEL_SELL")
    assert sale == "82594a23"


def test_pre_log_position_gets_a_synthetic_id(nvda):
    rows, ids, _, _ = nvda
    tp = next(pid for r, pid in zip(rows, ids) if r["exit_reason"] == "TAKE_PROFIT")
    assert tp == "bf-NVDA-2026-03-23"


def test_sell_entry_row_is_not_counted_twice(nvda):
    """A model SELL writes ENTRY + EXIT; only the EXIT moves shares."""
    rows, ids, _, _ = nvda
    sell_entries = [pid for r, pid in zip(rows, ids)
                    if r["actual_action"] == "SELL" and r["row_type"] == "ENTRY"]
    assert sell_entries and all(pid == "" for pid in sell_entries)


def test_rows_that_moved_no_shares_get_no_id(nvda):
    rows, ids, _, _ = nvda
    for r, pid in zip(rows, ids):
        if r["actual_action"] in ("HOLD", "STOP_BACKFILL", "INSUFFICIENT_EQUITY",
                                  "REBALANCER_TICKERS_SKIP"):
            assert pid == ""


# --- AAPL: one BUY, four trims, one take-profit ----------------------------


def test_aapl_single_buy_many_exits():
    rows = [
        _buy("2026-06-25", 78, "o1"),
        _exit("2026-06-26", 20), _exit("2026-06-30", 15),
        _exit("2026-07-01", 11), _exit("2026-07-02", 3),
        _exit("2026-07-16", 29, reason="TAKE_PROFIT"),
    ]
    ids, positions, flags = assign_positions(rows)
    assert ids == ["o1"] * 6
    assert flags == []
    assert positions[0]["closed"] == "2026-07-16"


# --- corrections from the broker's view ------------------------------------


def test_unlogged_take_profit_is_caught_by_the_next_buy():
    """Pre-April take-profits wrote no row; the next BUY says we were flat."""
    rows = [_buy("2026-03-10", 10, "", action="BUY"), _buy("2026-04-01", 5, "", action="BUY")]
    ids, positions, flags = assign_positions(rows)
    assert len(positions) == 2
    assert ids == ["bf-AAPL-2026-03-10", "bf-AAPL-2026-04-01"]
    assert [f["kind"] for f in flags] == ["UNLOGGED_EXIT"]


def test_checkpoint_resyncs_a_wrong_balance():
    rows = [
        _buy("2026-06-01", 10, "o1"),
        _row("2026-06-05", actual_action="STOP_BACKFILL", qty="7"),
        _exit("2026-06-06", 7),
    ]
    ids, positions, flags = assign_positions(rows)
    assert [f["kind"] for f in flags] == ["CHECKPOINT_MISMATCH"]
    assert positions[0]["closed"] == "2026-06-06"


def test_oversold_marks_the_balance_unknown_until_a_full_close():
    rows = [
        _buy("2026-06-01", 5, "o1"),
        _exit("2026-06-02", 8),        # more than the log says was bought
        _exit("2026-06-03", 2),        # must not open a new position
        _exit("2026-06-04", 4, reason="MODEL_SELL"),
    ]
    ids, positions, flags = assign_positions(rows)
    assert ids == ["o1"] * 4
    assert [f["kind"] for f in flags] == ["OVERSOLD"]
    assert positions[0]["closed"] == "2026-06-04"


def test_reconciled_stop_before_the_session_precedes_a_same_day_rebuy():
    """The stop is appended to the file after the BUY, but filled before it."""
    rows = [
        _buy("2026-07-01", 10, "old"),
        _buy("2026-07-02", 10, "new"),
        _exit("2026-07-02", 10, reason="STOP_LOSS_FILL", ts="2026-07-02T14:00:00+00:00"),
    ]
    ids, positions, flags = assign_positions(rows)
    assert ids == ["old", "new", "old"]
    assert flags == []


def test_reconciled_stop_after_the_session_belongs_to_the_new_position():
    rows = [
        _buy("2026-07-01", 10, "old"),
        _exit("2026-07-01", 10, reason="MODEL_SELL"),
        _buy("2026-07-02", 10, "new"),
        _exit("2026-07-02", 10, reason="STOP_LOSS_FILL", ts="2026-07-02T19:00:00+00:00"),
    ]
    ids, _, flags = assign_positions(rows)
    assert ids == ["old", "old", "new", "new"]
    assert flags == []


# --- live ids ---------------------------------------------------------------


def test_position_open_at_deploy_adopts_its_live_id():
    """Opened before deploy; the first post-deploy add minted its id live."""
    rows = [
        _buy("2026-09-01", 10, "o1"),
        _exit("2026-09-20", 2),                                  # blank: live trim before any add
        _buy("2026-09-26", 3, "o2", action="ADD_TO_POSITION", position_id="o2"),
        _exit("2026-09-29", 4, position_id="o2"),
    ]
    ids, _, flags = assign_positions(rows)
    assert ids == ["o2"] * 4
    assert flags == []


def test_existing_ids_are_never_overwritten():
    rows = [
        _buy("2026-09-01", 10, "o1", position_id="keep-me"),
        _exit("2026-09-02", 10, reason="MODEL_SELL", position_id="and-me"),
    ]
    ids, _, flags = assign_positions(rows)
    assert ids == ["keep-me", "and-me"]
    assert [f["kind"] for f in flags] == ["LIVE_ID_CONFLICT"]


def test_synthetic_ids_do_not_collide():
    rows = [
        _exit("2026-03-01", 5, reason="MODEL_SELL"),
        _row("2026-03-01", actual_action="STOP_BACKFILL", qty="3"),
        _exit("2026-03-02", 3, reason="MODEL_SELL"),
    ]
    ids, positions, _ = assign_positions(rows)
    assert len({p["position_id"] for p in positions}) == len(positions)


# --- Alpaca cross-check ----------------------------------------------------


def test_broker_comparison_flags_each_disagreement():
    positions = [
        {"ticker": "AAPL", "opened": "2026-09-01", "closed": None, "known": True, "balance": 10.0},
        {"ticker": "MSFT", "opened": "2026-09-01", "closed": None, "known": True, "balance": 5.0},
        {"ticker": "JPM", "opened": "2026-09-01", "closed": None, "known": True, "balance": 3.0},
        {"ticker": "XOM", "opened": "2026-08-01", "closed": "2026-08-09", "known": True, "balance": 0.0},
    ]
    broker = {"AAPL": 10.0, "MSFT": 4.0, "XOM": 7.0}
    flagged = {f["ticker"] for f in compare_with_broker(positions, broker)}
    assert flagged == {"MSFT", "JPM", "XOM"}


# --- CLI -------------------------------------------------------------------


def _write(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def test_main_writes_a_new_file_and_leaves_the_log_alone(tmp_path, capsys):
    log, out = tmp_path / "signal_log.csv", tmp_path / "out.csv"
    _write(log, LEGACY_FIELDNAMES, [_buy("2026-06-25", 5, "o1"), _exit("2026-06-26", 5, reason="MODEL_SELL")])
    before = log.read_bytes()

    assert main(["--log", str(log), "--out", str(out)]) == 0

    assert log.read_bytes() == before
    with open(out, newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == FIELDNAMES
        assert [r["position_id"] for r in reader] == ["o1", "o1"]
    assert "POSITION BACKFILL" in capsys.readouterr().out


def test_main_refuses_to_overwrite_the_log(tmp_path):
    log = tmp_path / "signal_log.csv"
    _write(log, FIELDNAMES, [])
    assert main(["--log", str(log), "--out", str(log)]) == 1


def test_main_rejects_an_unknown_header(tmp_path):
    log = tmp_path / "signal_log.csv"
    _write(log, ["date", "ticker"], [])
    assert main(["--log", str(log), "--out", str(tmp_path / "o.csv")]) == 1
