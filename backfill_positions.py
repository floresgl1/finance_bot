"""
backfill_positions.py — One-off: assign position_id to signal_log.csv rows
written before the column existed (added 2026-09-25).

Replays the log per ticker and groups trades into positions, flat to flat.
Writes a NEW csv and never touches the live log; swapping it in is a manual,
reviewed step.

**DESIGN DECISION — the broker's view beats the share count.**
The log cannot be replayed on share counts alone. Before 2026-04-23 a
take-profit wrote no row at all, and stops that fired between sessions only
appear once reconcile_stops.py has backfilled them, so a replayed balance can
stay positive long after the real position closed. Two kinds of row record
what Alpaca reported at the time, and the replay defers to them:

    actual_action "BUY"   — the bot saw the ticker as not held (an add is
                            logged as ADD_TO_POSITION), so the ticker was flat.
                            Opens a new position whatever the replay thinks.
    actual_action "STOP_BACKFILL" — written at session start with the held
                            quantity. Resyncs the running balance.

Every correction is reported as a flag. A flag is not an error in the replay:
it marks a place where the log is missing a trade, and the report is the
list of those places.

**DESIGN DECISION — which rows move shares.**
    + BUY / ADD_TO_POSITION ENTRY rows (qty is the filled quantity)
    - EXIT rows (shares). TAKE_PROFIT and MODEL_SELL sell the full position.
    - Pre-EXIT-era ENTRY rows: SELL and STOP_LOSS_SELL (full), REBALANCER_SELL
      (partial) — but only when no EXIT row for the same ticker, date and
      share count exists. After 2026-04-23 a model SELL writes both rows, and
      counting both would sell the position twice.
Skips, errors and unfilled orders never moved shares and are ignored.

**DESIGN DECISION — ids.**
A position takes the id of any row the live code already stamped (so history
joins up with the position that was open at deploy), else the order id of the
BUY that opened it (the live convention), else a synthetic
`bf-<TICKER>-<first date>` for positions opened before order ids were logged.
Existing ids are never overwritten.

Usage:
    python backfill_positions.py                    # writes data/signal_log.backfilled.csv
    python backfill_positions.py --check-alpaca     # also compare open positions with Alpaca
    python backfill_positions.py --log PATH --out PATH
"""

import argparse
import csv
import os
import sys
from collections import Counter
from datetime import datetime

from signal_logger import FIELDNAMES, LEGACY_FIELDNAMES, SIGNAL_LOG_PATH

# The daily trader runs at 15:00 UTC. A reconciled stop that filled before then
# happened before that day's session; one that filled after, after it.
SESSION_START_UTC_HOUR = 15

FULL_CLOSE_EXIT_REASONS = {"TAKE_PROFIT", "MODEL_SELL"}

# Pre-EXIT-era sell actions -> whether they closed the whole position.
LEGACY_DECREASE_ACTIONS = {
    "SELL": True,
    "STOP_LOSS_SELL": True,
    "REBALANCER_SELL": False,
}

EPS = 1e-6

DEFAULT_OUT = os.path.join(os.path.dirname(SIGNAL_LOG_PATH), "signal_log.backfilled.csv")


# ---------------------------------------------------------------------------
# Row classification
# ---------------------------------------------------------------------------
def _num(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _exit_keys(rows: list[dict]) -> set[tuple[str, str, float]]:
    keys = set()
    for r in rows:
        if r.get("row_type") == "EXIT":
            shares = _num(r.get("shares"))
            if shares:
                keys.add((r.get("ticker", ""), r.get("date", ""), round(shares, 4)))
    return keys


def classify(row: dict, exit_keys: set) -> tuple[str, float] | None:
    """Return (kind, shares) for a row that moves or reports shares, else None.

    kind is one of CHECKPOINT, OPEN, ADD, CLOSE (full), REDUCE (partial).
    """
    if row.get("row_type") == "EXIT":
        shares = _num(row.get("shares"))
        if not shares or shares <= 0:
            return None
        kind = "CLOSE" if row.get("exit_reason") in FULL_CLOSE_EXIT_REASONS else "REDUCE"
        return (kind, shares)

    action = row.get("actual_action", "")
    qty = _num(row.get("qty"))
    if not qty or qty <= 0:
        return None

    if action == "STOP_BACKFILL":
        return ("CHECKPOINT", qty)
    if action == "BUY":
        return ("OPEN", qty)
    if action == "ADD_TO_POSITION":
        return ("ADD", qty)
    if action in LEGACY_DECREASE_ACTIONS:
        if (row.get("ticker", ""), row.get("date", ""), round(qty, 4)) in exit_keys:
            return None   # the EXIT row already carries this sale
        return ("CLOSE" if LEGACY_DECREASE_ACTIONS[action] else "REDUCE", qty)
    return None


def _sort_key(index: int, row: dict) -> tuple[str, int, int]:
    """Date order, file order within a day — except reconciled stops.

    Those are appended days after the fact, so file order puts them after
    trades they preceded. Place them by fill time against the session start.
    """
    phase = 1
    if row.get("exit_reason") == "STOP_LOSS_FILL":
        try:
            ts = datetime.fromisoformat(row.get("exit_timestamp", ""))
            phase = 0 if ts.hour < SESSION_START_UTC_HOUR else 2
        except ValueError:
            pass
    return (row.get("date", ""), phase, index)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def assign_positions(rows: list[dict]) -> tuple[list[str], list[dict], list[dict]]:
    """Replay the log and assign a position_id to every share-moving row.

    Returns (position_ids, positions, flags). position_ids is parallel to
    rows; rows that moved no shares keep whatever id they had ("" if none).
    """
    exit_keys = _exit_keys(rows)
    positions: list[dict] = []
    flags: list[dict] = []
    current: dict[str, dict] = {}   # ticker -> open position

    def flag(ticker, date_, kind, detail):
        flags.append({"ticker": ticker, "date": date_, "kind": kind, "detail": detail})

    def open_position(ticker, date_, *, order_id="", known, balance):
        pos = {
            "ticker": ticker, "opened": date_, "closed": None,
            "open_order_id": order_id, "known": known, "balance": balance,
            "rows": [], "live_ids": set(), "position_id": None,
        }
        positions.append(pos)
        current[ticker] = pos
        return pos

    def close(pos, date_):
        pos["closed"] = date_
        current.pop(pos["ticker"], None)

    order = sorted(range(len(rows)), key=lambda i: _sort_key(i, rows[i]))
    for i in order:
        row = rows[i]
        c = classify(row, exit_keys)
        if c is None:
            continue
        kind, shares = c
        ticker, date_ = row.get("ticker", ""), row.get("date", "")
        pos = current.get(ticker)

        if kind == "CHECKPOINT":
            if pos is None:
                flag(ticker, date_, "HELD_WITHOUT_OPEN",
                     f"session start held {shares:g} sh with no open position in the log")
                open_position(ticker, date_, known=True, balance=shares)
            else:
                if pos["known"] and abs(pos["balance"] - shares) > EPS:
                    flag(ticker, date_, "CHECKPOINT_MISMATCH",
                         f"replay held {pos['balance']:g} sh, session start held {shares:g}")
                pos["balance"], pos["known"] = shares, True
            continue   # a checkpoint reports shares; it is not a trade

        if kind == "OPEN":
            if pos is not None:
                held = f"{pos['balance']:g} sh" if pos["known"] else "an unknown amount"
                flag(ticker, date_, "UNLOGGED_EXIT",
                     f"broker reported flat but replay still held {held} "
                     f"(position opened {pos['opened']})")
                close(pos, date_)
            pos = open_position(ticker, date_, order_id=row.get("entry_order_id", ""),
                                known=True, balance=shares)

        elif kind == "ADD":
            if pos is None:
                flag(ticker, date_, "ADD_WITHOUT_POSITION",
                     f"added {shares:g} sh to a position the log never opened")
                pos = open_position(ticker, date_, known=False, balance=shares)
            else:
                pos["balance"] += shares

        else:   # CLOSE / REDUCE
            if pos is None:
                flag(ticker, date_, "EXIT_WITHOUT_POSITION",
                     f"sold {shares:g} sh of a position the log never opened")
                pos = open_position(ticker, date_, known=False, balance=0.0)
            pos["balance"] -= shares

        pos["rows"].append(i)
        if row.get("position_id"):
            pos["live_ids"].add(row["position_id"])

        if kind in ("CLOSE", "REDUCE"):
            if kind == "CLOSE":
                if pos["known"] and abs(pos["balance"]) > EPS:
                    flag(ticker, date_, "CLOSE_MISMATCH",
                         f"full close left the replay at {pos['balance']:g} sh")
                close(pos, date_)
            elif pos["known"] and pos["balance"] < -EPS:
                flag(ticker, date_, "OVERSOLD",
                     f"sold {-pos['balance']:g} sh more than the replay held")
                pos["known"] = False
            elif pos["known"] and abs(pos["balance"]) <= EPS:
                close(pos, date_)

    # Resolve ids.
    used: set[str] = {r["position_id"] for r in rows if r.get("position_id")}
    seen_live: Counter = Counter()
    for pos in positions:
        live = sorted(pos["live_ids"])
        seen_live.update(live)
        if len(live) > 1:
            flag(pos["ticker"], pos["opened"], "LIVE_ID_CONFLICT",
                 f"one replayed position holds live ids {', '.join(live)}")
        if live:
            pid = live[0]
        elif pos["open_order_id"] and pos["open_order_id"] not in used:
            pid = pos["open_order_id"]
        else:
            base = f"bf-{pos['ticker']}-{pos['opened']}"
            pid, n = base, 2
            while pid in used:
                pid, n = f"{base}-{n}", n + 1
        used.add(pid)
        pos["position_id"] = pid

    for live_id, count in seen_live.items():
        if count > 1:
            flag("", "", "LIVE_ID_SPLIT",
                 f"live id {live_id} spans {count} replayed positions")

    out = [r.get("position_id", "") or "" for r in rows]
    for pos in positions:
        for i in pos["rows"]:
            if not out[i]:
                out[i] = pos["position_id"]

    return out, positions, flags


def compare_with_broker(positions: list[dict], broker: dict[str, float]) -> list[dict]:
    """Flag every ticker where the replay's end state disagrees with Alpaca."""
    open_now = {p["ticker"]: p for p in positions if p["closed"] is None}
    flags = []
    for ticker in sorted(set(open_now) | set(broker)):
        pos, held = open_now.get(ticker), broker.get(ticker, 0.0)
        if pos is None and held > EPS:
            detail = f"Alpaca holds {held:g} sh; replay has no open position"
        elif pos is not None and held <= EPS:
            detail = f"replay has a position open since {pos['opened']}; Alpaca holds none"
        elif pos is not None and pos["known"] and abs(pos["balance"] - held) > EPS:
            detail = f"replay holds {pos['balance']:g} sh, Alpaca holds {held:g}"
        else:
            continue
        flags.append({"ticker": ticker, "date": "", "kind": "ALPACA_MISMATCH", "detail": detail})
    return flags


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def read_log(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        if header not in (FIELDNAMES, LEGACY_FIELDNAMES):
            raise ValueError(f"unexpected header in {path}: {header}")
        return list(reader)


def write_log(path: str, rows: list[dict], position_ids: list[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, restval="")
        writer.writeheader()
        for row, pid in zip(rows, position_ids):
            writer.writerow({**row, "position_id": pid})
    os.replace(tmp, path)


def broker_positions() -> dict[str, float]:
    from reconcile_stops import get_api
    return {p.symbol: float(p.qty) for p in get_api().list_positions()}


def format_report(rows, position_ids, positions, flags) -> str:
    """ASCII only: this prints to a console that may be cp1252."""
    stamped_before = sum(1 for r in rows if r.get("position_id"))
    stamped_after = sum(1 for p in position_ids if p)
    per_ticker = Counter(p["ticker"] for p in positions)

    out = [
        "=" * 72,
        "  POSITION BACKFILL",
        "=" * 72,
        f"  Rows in log            : {len(rows)}",
        f"  Rows with position_id  : {stamped_before} before -> {stamped_after} after",
        f"  Positions replayed     : {len(positions)} "
        f"({sum(1 for p in positions if p['closed'] is None)} still open)",
        "",
        "  Positions per ticker   : "
        + ", ".join(f"{t} {n}" for t, n in sorted(per_ticker.items())),
        "",
    ]
    if not flags:
        out.append("  No flags: every position replayed cleanly.")
        return "\n".join(out)

    out.append("  FLAGS BY KIND")
    for kind, n in Counter(f["kind"] for f in flags).most_common():
        out.append(f"    {kind:<22}{n:>5}")
    out.append("")
    out.append("  FLAGS")
    for f in sorted(flags, key=lambda f: (f["ticker"], f["date"])):
        out.append(f"    {f['date'] or '-':<11}{f['ticker'] or '-':<7}{f['kind']:<22}{f['detail']}")
    return "\n".join(out)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill position_id on historical signal_log.csv rows.")
    parser.add_argument("--log", default=SIGNAL_LOG_PATH, metavar="PATH")
    parser.add_argument("--out", default=DEFAULT_OUT, metavar="PATH",
                        help=f"Where to write the backfilled log (default: {DEFAULT_OUT}).")
    parser.add_argument("--check-alpaca", action="store_true",
                        help="Compare the replay's open positions with Alpaca's.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    if os.path.abspath(args.out) == os.path.abspath(args.log):
        print("[FATAL] --out must differ from --log; this script never overwrites the live log.")
        return 1
    if not os.path.exists(args.log):
        print(f"[FATAL] Signal log not found: {args.log}")
        return 1

    try:
        rows = read_log(args.log)
    except (OSError, ValueError) as exc:
        print(f"[FATAL] {exc}")
        return 1

    position_ids, positions, flags = assign_positions(rows)

    if args.check_alpaca:
        try:
            flags += compare_with_broker(positions, broker_positions())
        except Exception as exc:
            print(f"[WARN] Alpaca check skipped: {exc}")

    print(format_report(rows, position_ids, positions, flags))
    write_log(args.out, rows, position_ids)
    print(f"\n  Wrote {args.out}  (the live log was not modified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
