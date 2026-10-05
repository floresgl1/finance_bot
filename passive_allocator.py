"""
passive_allocator.py — Hold a fixed stock allocation; invest deposits first.

A separate track from the model bot (live_trader.py). It has no signal: it
keeps the account at config.PASSIVE_TARGETS (VTI 60% / VXUS 40%) by two rules.

    1. Deposits first.  Uninvested cash buys whichever fund is underweight,
       split in proportion to each fund's shortfall. A deposit that covers the
       whole shortfall lands the account exactly on target.
    2. Band.  Shares are sold only when a fund is more than PASSIVE_BAND
       (5pp) from its target *after* the cash has been counted in — so a
       deposit that will fix the drift never triggers a sale. The overweight
       fund is sold back to target, and the proceeds are invested by rule 1
       in the same run.

Selling is the only taxable event in a regular account, so the rules sell as
rarely as the band allows. Inside a Roth IRA that saving disappears; the
rules still mean fewer trades.

Execution sequence (one run per trading day):

    market open? → run guard → read account → refuse foreign positions
      → plan_sells → submit, wait for fills → re-read account
      → plan_buys → submit, wait for fills → log, Discord summary

The run guard asks the broker, not a file: any order the account placed
since midnight UTC today means today's run already happened. On GitHub
Actions every runner starts fresh, so a file would never be there; the
account is the only state that outlives a run. It also covers an earlier
run whose buys are still open — those show in neither positions nor cash,
so without the guard a second run would buy the same deposit again.

The account must hold nothing but the target funds and cash. A position in
any other symbol means the account is shared — with the model bot, say — and
the run stops rather than trade around it.

Paper only: connect() is pinned to broker.PAPER_URL, with its own keys
(PASSIVE_ALPACA_API_KEY / PASSIVE_ALPACA_SECRET_KEY), never the model bot's. Moving to a live or IRA
account is a deliberate later change, not a setting.

plan_sells() and plan_buys() are pure: they take values and return orders,
and are where the allocation rules live. run() is the I/O around them.
"""

import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import broker
from config import (
    PASSIVE_BAND,
    PASSIVE_CASH_BUFFER_USD,
    PASSIVE_FILL_TIMEOUT_S,
    PASSIVE_FRACTIONAL,
    PASSIVE_LOG_PATH,
    PASSIVE_MIN_ORDER_USD,
    PASSIVE_TARGETS,
    today_utc,
)

_EPS = 1e-9                     # float slack: a weight exactly on the band edge is inside it
_TERMINAL = {"filled", "canceled", "expired", "rejected", "done_for_day"}
_LOG_FIELDS = ["date", "symbol", "side", "notional", "qty", "status",
               "order_id", "reason"]


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str                   # "buy" | "sell"
    notional: float | None      # dollars, when trading fractionally
    qty: int | None             # whole shares, when not
    reason: str


# --- planning (pure) --------------------------------------------------------


def _check_targets(targets: dict[str, float]) -> None:
    if not targets or abs(sum(targets.values()) - 1.0) > 1e-6 or min(targets.values()) < 0:
        raise ValueError(f"targets must be non-negative and sum to 1, got {targets}")


def _cents(x: float) -> float:
    """Round down to the cent, so planned orders never exceed the cash."""
    return math.floor(x * 100 + 1e-6) / 100


def _order(symbol, side, amount, prices, fractional, min_order, reason) -> Order | None:
    if fractional:
        amount = _cents(amount)
        return Order(symbol, side, amount, None, reason) if amount >= min_order else None
    qty = math.floor(amount / prices[symbol] + 1e-9)
    return Order(symbol, side, None, qty, reason) if qty >= 1 else None


def _cash_split(values, investable, targets):
    """How rule 1 divides `investable` across the funds: in proportion to
    each fund's shortfall from target, measured on the post-deposit total."""
    total = sum(values.get(s, 0.0) for s in targets) + investable
    shortfall = {s: max(0.0, targets[s] * total - values.get(s, 0.0)) for s in targets}
    owed = sum(shortfall.values())
    if owed <= 0:
        return {s: investable * targets[s] for s in targets}, total
    return {s: investable * shortfall[s] / owed for s in targets}, total


def plan_buys(values: dict[str, float], cash: float, prices: dict[str, float],
              targets: dict[str, float] = PASSIVE_TARGETS,
              min_order: float = PASSIVE_MIN_ORDER_USD,
              buffer: float = PASSIVE_CASH_BUFFER_USD,
              fractional: bool = PASSIVE_FRACTIONAL) -> list[Order]:
    """Rule 1: invest uninvested cash into the underweight funds."""
    _check_targets(targets)
    investable = cash - buffer
    if investable < min_order:
        return []
    split, _ = _cash_split(values, investable, targets)
    orders = [_order(s, "buy", amt, prices, fractional, min_order, "DEPOSIT")
              for s, amt in split.items()]
    return [o for o in orders if o]


def plan_sells(values: dict[str, float], cash: float, prices: dict[str, float],
               targets: dict[str, float] = PASSIVE_TARGETS,
               band: float = PASSIVE_BAND,
               min_order: float = PASSIVE_MIN_ORDER_USD,
               buffer: float = PASSIVE_CASH_BUFFER_USD,
               fractional: bool = PASSIVE_FRACTIONAL) -> list[Order]:
    """Rule 2: sell an overweight fund back to target, but only if it is
    still outside the band once the uninvested cash has been placed."""
    _check_targets(targets)
    investable = max(0.0, cash - buffer)
    split, total = _cash_split(values, investable, targets)
    if total <= 0:
        return []
    projected = {s: values.get(s, 0.0) + split[s] for s in targets}
    if all(abs(projected[s] / total - targets[s]) <= band + _EPS for s in targets):
        return []
    orders = []
    for s in targets:
        excess = projected[s] - targets[s] * total
        if excess > 0:
            # Never more than is held — the excess can only sit in the holding.
            amount = min(excess, values.get(s, 0.0))
            orders.append(_order(s, "sell", amount, prices, fractional, min_order, "BAND"))
    return [o for o in orders if o]


# --- execution (I/O) --------------------------------------------------------


def send_discord(message: str) -> None:
    from run_bot import send_discord as _send
    _send(message)


def _read_state(api, targets):
    """Values, cash and prices for the target funds, plus any other symbols held."""
    positions = api.list_positions()
    foreign = sorted(p.symbol for p in positions if p.symbol not in targets)
    values = {p.symbol: float(p.market_value) for p in positions if p.symbol in targets}
    prices = {p.symbol: float(p.current_price) for p in positions if p.symbol in targets}
    for s in targets:
        if s not in prices:
            prices[s] = float(api.get_latest_trade(s).price)
    account = api.get_account()
    # A margin account can report buying power above cash; never spend more
    # than the cash actually deposited.
    cash = min(float(account.cash), float(account.buying_power))
    return values, cash, prices, foreign


def _submit(api, order: Order):
    if order.notional is not None:
        return api.submit_order(symbol=order.symbol, qty=None, side=order.side,
                                type="market", time_in_force="day",
                                notional=order.notional)
    return api.submit_order(symbol=order.symbol, qty=order.qty, side=order.side,
                            type="market", time_in_force="day")


def _wait_for_fill(api, order_id: str, timeout_s: float, poll_s: float = 1.0) -> str:
    deadline = time.monotonic() + timeout_s
    while True:
        status = str(api.get_order(order_id).status).lower().split(".")[-1]
        if status in _TERMINAL or time.monotonic() >= deadline:
            return status
        time.sleep(poll_s)


def _log(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def _already_ran(api, today: str) -> bool:
    """True if the account has placed any order since midnight UTC `today`
    — the same UTC date today_utc() gives. The allocator only trades while
    the market is open (13:30–21:00 UTC), so yesterday's run can never land
    after midnight UTC."""
    midnight = datetime.fromisoformat(today).replace(tzinfo=timezone.utc)
    return bool(api.list_orders(status="all", after=midnight, limit=1))


def _place(api, orders: list[Order], today: str, rows: list[dict],
           wait_s: float | None = None) -> list[str]:
    """Submit each order; with `wait_s`, block until it reaches a terminal
    status. Returns a one-line summary per order."""
    lines = []
    for o in orders:
        try:
            placed = _submit(api, o)
            status = str(placed.status).lower().split(".")[-1]
            if wait_s is not None:
                status = _wait_for_fill(api, placed.id, wait_s)
            order_id = placed.id
        except Exception as exc:             # one bad order must not stop the rest
            status, order_id = f"error: {exc}", ""
        rows.append({"date": today, "symbol": o.symbol, "side": o.side,
                     "notional": o.notional, "qty": o.qty, "status": status,
                     "order_id": order_id, "reason": o.reason})
        size = f"${o.notional:,.2f}" if o.notional is not None else f"{o.qty} sh"
        lines.append(f"{o.side.upper()} {o.symbol} {size} ({o.reason}) → {status}")
    return lines


def run(api, today: str | None = None,
        targets: dict[str, float] = PASSIVE_TARGETS,
        log_path: str = PASSIVE_LOG_PATH,
        fill_timeout_s: float = PASSIVE_FILL_TIMEOUT_S) -> int:
    """One allocation pass. Returns a process exit code: 0 for done or
    nothing to do, 1 for a refusal or failure that needs a person."""
    today = today or today_utc()
    if not api.get_clock().is_open:
        print("[passive] Market closed — nothing to do.")
        return 0
    if _already_ran(api, today):
        print(f"[passive] Already ran {today}.")
        return 0

    values, cash, prices, foreign = _read_state(api, targets)
    if foreign:
        msg = (f"[passive] REFUSED: account holds {', '.join(foreign)}, which are not "
               f"in the target allocation. Use a dedicated account.")
        print(msg)
        send_discord(msg)
        return 1

    rows: list[dict] = []
    lines = _place(api, plan_sells(values, cash, prices, targets), today, rows,
                   wait_s=fill_timeout_s)
    if lines:                                 # sale proceeds are now cash
        values, cash, prices, _ = _read_state(api, targets)
    lines += _place(api, plan_buys(values, cash, prices, targets), today, rows,
                    wait_s=fill_timeout_s)

    if rows:
        _log(rows, log_path)

    held = sum(values.values())
    weights = ", ".join(f"{s} {values.get(s, 0.0) / held:.1%}" for s in targets) if held else "empty"
    summary = f"[passive] {today}: before buys {weights}; cash ${cash:,.2f}"
    if lines:
        summary += "\n" + "\n".join(lines)
    print(summary)
    if lines:
        send_discord(summary)
    # Every order was waited on, so anything short of filled — an error, a
    # rejection, or an order still open at the timeout — needs a person.
    return 1 if any(r["status"] != "filled" for r in rows) else 0


def connect() -> broker.AlpacaREST:
    """The allocator's own account. Its keys are separate from the model
    bot's on purpose, with no fallback: run against the model bot's account
    it would only find foreign positions and refuse."""
    key = os.environ.get("PASSIVE_ALPACA_API_KEY")
    secret = os.environ.get("PASSIVE_ALPACA_SECRET_KEY")
    if not key or not secret:
        raise EnvironmentError(
            "PASSIVE_ALPACA_API_KEY and PASSIVE_ALPACA_SECRET_KEY must be set — "
            "the keys of a paper account used only by the passive allocator.")
    return broker.AlpacaREST(key, secret, broker.PAPER_URL)


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv()
    return run(connect())


if __name__ == "__main__":
    sys.exit(main())
