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
