"""
Smoke test for broker.AlpacaREST against a real Alpaca PAPER account.

Read-only by default: account, clock, positions, latest trade, open orders,
portfolio history. Nothing is submitted.

With --place-order it also exercises every order shape the bot places, on one
share of a symbol OUTSIDE the watchlist, and cleans up:

  market closed / order not filled:
      OTO buy with stop leg -> confirm it and its leg -> cancel it
  market open / order filled:
      OTO buy -> cancel the stop leg -> place a standing stop (the backfill
      shape) -> cancel it -> market sell the share

Run it outside the bot's 15:00 UTC window so the two never overlap.

    python smoke_alpaca.py                     # read-only
    python smoke_alpaca.py --place-order       # paper orders on 1 share of F
    python smoke_alpaca.py --place-order --symbol T

Exit code 0 only if every check passed.
"""

import argparse
import sys
import time

import broker
from config import WATCHLIST

FILL_WAIT_SECONDS = 20


class Checks:
    def __init__(self):
        self.failed = 0

    def run(self, name, fn):
        try:
            detail = fn()
            print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
            return True
        except Exception as exc:  # noqa: BLE001 - a smoke test reports everything
            self.failed += 1
            print(f"  FAIL  {name} — {type(exc).__name__}: {exc}")
            return False


def _require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def read_only_checks(api, symbol: str, c: Checks) -> None:
    def account():
        eq = float(api.get_account().equity)
        _require(eq > 0, f"equity {eq}")
        return f"equity ${eq:,.2f}"

    def clock():
        is_open = api.get_clock().is_open
        _require(isinstance(is_open, bool), f"is_open is {type(is_open).__name__}")
        return f"market {'OPEN' if is_open else 'CLOSED'}"

    def positions():
        pos = api.list_positions()
        for p in pos:
            float(p.qty), float(p.avg_entry_price)
            _require(isinstance(p.symbol, str), "symbol not a string")
        return f"{len(pos)} held"

    def latest_trade():
        price = float(api.get_latest_trade(symbol).price)
        _require(price > 0, f"price {price}")
        return f"{symbol} ${price:.2f}"

    def open_orders():
        orders = api.list_orders(status="open")
        for o in orders:
            _require(type(o.id) is str and type(o.status) is str, "id/status not plain strings")
        return f"{len(orders)} open"

    def history():
        df = api.get_portfolio_history(period="7D", timeframe="1D").df
        _require("equity" in df, "no equity column")
        return f"{len(df)} rows"

    for name, fn in (("account", account), ("clock", clock), ("positions", positions),
                     ("latest trade", latest_trade), ("open orders", open_orders),
                     ("portfolio history", history)):
        c.run(name, fn)

    try:
        rows, warnings = stop_coverage(api.list_positions(), api.list_orders(status="open"))
        print_stop_coverage(rows, warnings)
    except Exception as exc:  # noqa: BLE001 - a diagnostic must not mask the checks above
        print(f"  WARN  stop coverage could not be computed — {type(exc).__name__}: {exc}")


def _is_sell_stop(o) -> bool:
    return o.side == "sell" and o.type in ("stop", "stop_limit")


def stop_coverage(positions, orders) -> tuple[list[dict], list[str]]:
    """Per-symbol view of positions against open orders, plus risk warnings.

    Each held position should be covered by live sell stops whose quantities
    add up to exactly the shares held. Several stops per symbol are normal:
    every BUY is an OTO order with its own stop, so a position built in N
    lots carries N stops. What matters is the total. Warns on: no stop
    (unprotected), stop total above the position (the stops could sell
    shares that are not there), stop total below it (partly unprotected),
    and a live stop on a symbol not held (orphan). A stop leg still waiting
    on its unfilled OTO parent (status "held") protects nothing yet and is
    reported separately, not as an orphan.
    """
    held = {p.symbol: float(p.qty) for p in positions}
    symbols = sorted(set(held) | {o.symbol for o in orders})
    rows, warnings = [], []
    for sym in symbols:
        mine = [o for o in orders if o.symbol == sym]
        live_stops = [o for o in mine if _is_sell_stop(o) and o.status != "held"]
        waiting_stops = [o for o in mine if _is_sell_stop(o) and o.status == "held"]
        other = [o for o in mine if not _is_sell_stop(o)]
        stop_qty = sum(float(o.qty) for o in live_stops)
        qty = held.get(sym, 0.0)
        rows.append({
            "symbol": sym, "held": qty, "stops": len(live_stops), "stop_qty": stop_qty,
            "waiting_stops": len(waiting_stops),
            "other": ", ".join(sorted(f"{o.side} {o.type}" for o in other)),
        })
        if qty > 0 and not live_stops:
            warnings.append(f"{sym}: {qty:g} shares held with NO live stop (unprotected)")
        if live_stops and qty > 0 and stop_qty > qty + 1e-9:
            warnings.append(f"{sym}: stops cover {stop_qty:g} shares but only {qty:g} are "
                            f"held (over-covered: could sell shares that are not there)")
        elif live_stops and qty > 0 and stop_qty < qty - 1e-9:
            warnings.append(f"{sym}: stops cover {stop_qty:g} shares but {qty:g} are held "
                            f"(partly unprotected)")
        if live_stops and qty == 0:
            warnings.append(f"{sym}: {len(live_stops)} live sell stop(s) but no position (orphan)")
    return rows, warnings


def print_stop_coverage(rows: list[dict], warnings: list[str]) -> None:
    print("\nStop coverage (read-only)")
    print(f"  {'symbol':<8}{'held':>8}{'stops':>7}{'stop qty':>10}{'waiting':>9}  other open orders")
    for r in rows:
        print(f"  {r['symbol']:<8}{r['held']:>8g}{r['stops']:>7}{r['stop_qty']:>10g}"
              f"{r['waiting_stops']:>9}  {r['other'] or '-'}")
    for w in warnings:
        print(f"  WARN  {w}")
    if not warnings:
        print("  every position's stops add up to exactly the shares held")


def _stops(api, symbol):
    return [o for o in api.list_orders(status="open", symbols=[symbol])
            if o.side == "sell" and o.type in ("stop", "stop_limit")]


def _cancel_and_confirm(api, order_id):
    api.cancel_order(order_id)
    for _ in range(10):
        status = api.get_order(order_id).status
        if status in ("canceled", "filled"):
            return status
        time.sleep(1)
    raise AssertionError(f"order {order_id} still {status} after cancel")


def order_checks(api, symbol: str, c: Checks) -> None:
    held = {p.symbol for p in api.list_positions()}
    if symbol in held:
        print(f"  SKIP  order checks — account already holds {symbol}; pick another --symbol")
        c.failed += 1
        return

    price = float(api.get_latest_trade(symbol).price)
    stop_price = round(price * 0.90, 2)
    state = {}

    def oto_buy():
        o = api.submit_order(symbol=symbol, qty=1, side="buy", type="market",
                             time_in_force="gtc", order_class="oto",
                             stop_loss={"stop_price": stop_price})
        _require(type(o.id) is str, "order id not a string")
        state["id"] = o.id
        return f"order {o.id}, stop @ ${stop_price:.2f}"

    if not c.run("submit OTO buy (1 share)", oto_buy):
        return

    deadline = time.time() + FILL_WAIT_SECONDS
    status = api.get_order(state["id"]).status
    while status not in ("filled", "canceled", "rejected") and time.time() < deadline:
        time.sleep(1)
        status = api.get_order(state["id"]).status
    print(f"        entry status: {status}")

    if status != "filled":
        c.run("cancel unfilled OTO", lambda: _cancel_and_confirm(api, state["id"]))
        return

    def stop_leg_present():
        stops = _stops(api, symbol)
        _require(stops, "no open stop leg found for the filled OTO")
        state["stops"] = [s.id for s in stops]
        return f"{len(stops)} stop leg(s)"

    c.run("OTO stop leg is live", stop_leg_present)
    c.run("cancel OTO stop leg",
          lambda: [_cancel_and_confirm(api, s) for s in state.get("stops", [])] and "done")

    def standing_stop():
        o = api.submit_order(symbol=symbol, qty=1, side="sell", type="stop",
                             stop_price=stop_price, time_in_force="gtc")
        state["standing"] = o.id
        return f"order {o.id}"

    if c.run("submit standing stop (backfill shape)", standing_stop):
        c.run("cancel standing stop", lambda: _cancel_and_confirm(api, state["standing"]))

    def market_sell():
        o = api.submit_order(symbol=symbol, qty=1, side="sell", type="market",
                             time_in_force="day")
        return f"order {o.id}"

    c.run("market sell the share (cleanup)", market_sell)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--place-order", action="store_true",
                        help="Also place and clean up paper orders on 1 share.")
    parser.add_argument("--symbol", default="F",
                        help="Symbol for the order checks; must not be in the watchlist.")
    args = parser.parse_args(argv)

    if args.symbol in WATCHLIST:
        print(f"[FATAL] {args.symbol} is in the bot's watchlist; use a symbol it never trades.")
        return 2

    api = broker.connect(broker.PAPER_URL)
    c = Checks()
    print("Read-only checks")
    read_only_checks(api, args.symbol, c)
    if args.place_order:
        print(f"\nOrder checks on 1 share of {args.symbol} (paper)")
        order_checks(api, args.symbol, c)

    print(f"\n{'ALL PASSED' if c.failed == 0 else f'{c.failed} FAILED'}")
    return 0 if c.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
