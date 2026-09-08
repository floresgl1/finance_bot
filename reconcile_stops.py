"""
reconcile_stops.py — Recover stop-loss exits that never reached signal_log.csv.

Since stop-losses became standing Alpaca OTO child orders, they fire on
Alpaca's side between bot sessions. The bot is not running when they fill, so
`log_exit()` is never called and the position closes with no EXIT row. Every
figure in pnl_report.py was therefore missing its losing tail, biasing realized
P&L upward by an unknown amount.

This module queries Alpaca for filled stop orders and writes the missing EXIT
rows with `exit_reason = "STOP_LOSS_FILL"`.

**DESIGN DECISION:**
Idempotency comes from the data, not from a side-car state file. A reconciled
row stores Alpaca's `filled_at` as its `exit_timestamp`, which is stable across
runs, so `(ticker, exit_timestamp)` identifies a stop fill exactly. Re-running
the reconciler re-derives the same key and skips. A separate state file (the
edge_monitor pattern) would have to round-trip through PythonAnywhere and could
drift out of sync with the log it describes; the log is already the thing that
round-trips.

**DESIGN DECISION:**
Entry price comes from `api.get_order(entry_order_id).filled_avg_price` — the
actual fill of the original BUY — rather than the ENTRY row's `price`, which is
the signal-time estimate. The whole point of this module is to stop
under-reporting losses; using the optimistic number would undercut it. When the
order cannot be fetched the row is still written, using the ENTRY row's price
and flagged in the run summary, because a slightly imprecise loss is far closer
to the truth than a missing one.

**DESIGN DECISION:**
Linkage reuses `signal_logger.find_open_entry_order_id(ticker)` rather than
trying to walk the OTO parent/child relationship. Alpaca does not expose a
parent id on the child leg, and the open-ENTRY convention is what every other
exit path in this codebase already uses.

Usage:
    python reconcile_stops.py                  # reconcile, write EXIT rows
    python reconcile_stops.py --dry-run        # report what would be written
    python reconcile_stops.py --since 2026-06-01
    python reconcile_stops.py --discord        # post a summary when rows land
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from dotenv import load_dotenv
load_dotenv()

from signal_logger import (
    SIGNAL_LOG_PATH,
    find_open_entry_order_id,
    log_exit,
)

STOP_LOSS_FILL = "STOP_LOSS_FILL"

# Alpaca caps a single list_orders page; the bot closes far fewer positions
# than this per window, but the limit is explicit so a silent truncation
# cannot under-report.
ORDER_PAGE_LIMIT = 500

# How far back to look when --since is not given. Stops are reconciled daily,
# so this only matters after an outage.
DEFAULT_LOOKBACK_DAYS = 30


# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------
def get_api():
    """Build an Alpaca REST client. Imported lazily so --help works without
    credentials and so tests never need the dependency."""
    import alpaca_trade_api as tradeapi

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise EnvironmentError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set as environment variables."
        )
    return tradeapi.REST(
        api_key, secret_key, "https://paper-api.alpaca.markets", api_version="v2"
    )


def fetch_filled_stops(api, since: date) -> list[dict]:
    """Return every filled stop / stop_limit sell order since `since`.

    A stop that was cancelled (because the bot sold the position first) is
    not a fill and is excluded — only `status == "filled"` closes a position.
    """
    try:
        orders = api.list_orders(
            status="closed",
            after=since.isoformat(),
            limit=ORDER_PAGE_LIMIT,
            direction="asc",
            nested=False,
        )
    except Exception as exc:
        print(f"[FATAL] Could not list orders: {exc}")
        raise

    if len(orders) >= ORDER_PAGE_LIMIT:
        print(
            f"  [WARN] Alpaca returned {len(orders)} orders, the page limit. "
            f"Some fills may be missing — narrow the window with --since."
        )

    fills = []
    for order in orders:
        if getattr(order, "side", None) != "sell":
            continue
        if getattr(order, "type", None) not in ("stop", "stop_limit"):
            continue
        if getattr(order, "status", None) != "filled":
            continue

        try:
            qty = float(order.filled_qty)
            price = float(order.filled_avg_price)
        except (AttributeError, TypeError, ValueError) as exc:
            print(f"  [SKIP] {getattr(order, 'id', '?')} — unparseable fill: {exc}")
            continue

        if qty <= 0 or price <= 0:
            print(f"  [SKIP] {order.id} — non-positive fill qty/price")
            continue

        fills.append({
            "order_id": str(order.id),
            "ticker": order.symbol,
            "shares": qty,
            "exit_price": price,
            "filled_at": _normalize_timestamp(order.filled_at),
        })

    return fills


def _normalize_timestamp(value) -> str:
    """Render Alpaca's filled_at as a stable second-resolution ISO string.

    Stability matters: this string is half the idempotency key, so it must
    render identically on every run regardless of whether the SDK hands back
    a datetime, a pandas Timestamp, or a string.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    try:
        return value.astimezone(timezone.utc).isoformat(timespec="seconds")
    except (AttributeError, TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def existing_exit_keys(path: str = SIGNAL_LOG_PATH) -> set[tuple[str, str]]:
    """Return {(ticker, exit_timestamp)} for EXIT rows already in the log."""
    if not os.path.exists(path):
        return set()

    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as exc:
        print(f"[FATAL] Could not read {path}: {exc}")
        raise

    if "row_type" not in df.columns:
        return set()

    exits = df[df["row_type"].fillna("") == "EXIT"]
    return {
        (str(row["ticker"]), str(row["exit_timestamp"]))
        for _, row in exits.iterrows()
        if pd.notna(row.get("ticker")) and pd.notna(row.get("exit_timestamp"))
    }


def select_unreconciled(fills: list[dict], seen: set[tuple[str, str]]) -> list[dict]:
    """Filter fills down to those with no matching EXIT row.

    Also de-duplicates within the batch itself, so a single run cannot write
    two rows for one stop fill.
    """
    out = []
    batch: set[tuple[str, str]] = set()
    for fill in fills:
        key = (fill["ticker"], fill["filled_at"])
        if key in seen or key in batch:
            continue
        batch.add(key)
        out.append(fill)
    return out


# ---------------------------------------------------------------------------
# Entry price resolution
# ---------------------------------------------------------------------------
def resolve_entry(api, ticker: str) -> tuple[str, float | None, str]:
    """Find the open ENTRY for `ticker` and its true fill price.

    Returns (entry_order_id, entry_price, note). entry_price is None when it
    could not be established at all; the caller decides what to do.
    """
    entry_order_id = find_open_entry_order_id(ticker)
    if not entry_order_id:
        return ("UNLINKED", None, "no open ENTRY row")

    try:
        order = api.get_order(entry_order_id)
        price = float(order.filled_avg_price)
        if price > 0:
            return (entry_order_id, price, "")
        return (entry_order_id, None, "entry order has non-positive fill price")
    except Exception as exc:
        return (entry_order_id, None, f"could not fetch entry order ({exc})")


def entry_price_from_log(ticker: str, entry_order_id: str, path: str = SIGNAL_LOG_PATH) -> float | None:
    """Fall back to the signal-time price on the matching ENTRY row."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        return None
    if "row_type" not in df.columns:
        return None

    rows = df[
        (df["row_type"].fillna("ENTRY") == "ENTRY")
        & (df["ticker"] == ticker)
    ]
    if entry_order_id and entry_order_id != "UNLINKED":
        matched = rows[rows["entry_order_id"] == entry_order_id]
        if not matched.empty:
            rows = matched

    if rows.empty:
        return None

    price = pd.to_numeric(rows.iloc[-1].get("price"), errors="coerce")
    return float(price) if pd.notna(price) and price > 0 else None


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------
def reconcile(api, since: date, *, dry_run: bool = False,
              log_path: str = SIGNAL_LOG_PATH) -> dict:
    """Write EXIT rows for every unreconciled stop fill.

    Returns a summary dict: counts plus the rows written (or that would be).
    """
    fills = fetch_filled_stops(api, since)
    print(f"  Alpaca returned {len(fills)} filled stop order(s) since {since}")

    seen = existing_exit_keys(log_path)
    pending = select_unreconciled(fills, seen)
    print(f"  {len(fills) - len(pending)} already reconciled, {len(pending)} to write")

    written: list[dict] = []
    degraded = 0

    for fill in pending:
        entry_order_id, entry_price, note = resolve_entry(api, fill["ticker"])

        if entry_price is None:
            entry_price = entry_price_from_log(fill["ticker"], entry_order_id, log_path)
            if entry_price is not None:
                degraded += 1
                note = f"{note}; used ENTRY row price {entry_price}"

        if entry_price is None:
            # Without an entry price the P&L would be fabricated. Skipping is
            # the honest outcome, and it is reported rather than swallowed.
            print(
                f"  [SKIP] {fill['ticker']} @ {fill['filled_at']} — "
                f"no entry price available ({note})"
            )
            continue

        realized = round((fill["exit_price"] - entry_price) * fill["shares"], 4)
        row = {
            **fill,
            "entry_order_id": entry_order_id,
            "entry_price": entry_price,
            "realized_pnl": realized,
            "note": note,
        }
        written.append(row)

        if dry_run:
            print(
                f"  [DRY RUN] would write {fill['ticker']:<6} "
                f"entry ${entry_price:>8.2f} -> exit ${fill['exit_price']:>8.2f}  "
                f"P&L ${realized:>+10.2f}"
            )
            continue

        log_exit(
            ticker=fill["ticker"],
            entry_order_id=entry_order_id,
            entry_price=entry_price,
            exit_price=fill["exit_price"],
            exit_reason=STOP_LOSS_FILL,
            shares=fill["shares"],
            today=_exit_date(fill["filled_at"]),
            exit_timestamp=fill["filled_at"],
        )

    return {
        "fills_seen": len(fills),
        "already_reconciled": len(fills) - len(pending),
        "written": written,
        "skipped": len(pending) - len(written),
        "degraded_entry_price": degraded,
        "dry_run": dry_run,
    }


def _exit_date(filled_at: str) -> date | None:
    """Date the stop actually fired, so the EXIT row lands in the right window."""
    try:
        return datetime.fromisoformat(filled_at).date()
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_summary(result: dict) -> str:
    """Console summary of a reconciliation run."""
    lines = ["=" * 68, "  STOP-LOSS RECONCILIATION", "=" * 68]
    if result["dry_run"]:
        lines.append("  (dry run - no rows written)")

    lines.append(f"  Filled stops found      : {result['fills_seen']}")
    lines.append(f"  Already reconciled      : {result['already_reconciled']}")
    lines.append(f"  EXIT rows written       : {len(result['written'])}")

    if result["skipped"]:
        lines.append(f"  Skipped (no entry price): {result['skipped']}")
    if result["degraded_entry_price"]:
        lines.append(
            f"  Degraded entry price    : {result['degraded_entry_price']} "
            f"(used ENTRY row estimate)"
        )

    if result["written"]:
        total = sum(r["realized_pnl"] for r in result["written"])
        lines.append(f"  Recovered realized P&L  : ${total:,.2f}")
        lines.append("")
        for row in result["written"]:
            lines.append(
                f"    {row['ticker']:<6} {row['filled_at']}  "
                f"${row['entry_price']:>8.2f} -> ${row['exit_price']:>8.2f}  "
                f"{row['shares']:>6.0f} sh  ${row['realized_pnl']:>+10.2f}"
            )
    lines.append("=" * 68)
    return "\n".join(lines)


def format_discord(result: dict) -> str:
    total = sum(r["realized_pnl"] for r in result["written"])
    lines = [
        f"🧾 **Stop-loss reconciliation** — "
        f"{len(result['written'])} previously unlogged exit(s) recovered",
        f"Recovered realized P&L: **${total:,.2f}**",
    ]
    for row in result["written"][:10]:
        lines.append(
            f"· `{row['ticker']}` {row['shares']:.0f} sh — ${row['realized_pnl']:,.2f}"
        )
    if len(result["written"]) > 10:
        lines.append(f"· …and {len(result['written']) - 10} more")
    if result["skipped"]:
        lines.append(f"⚠️ {result['skipped']} fill(s) skipped — no entry price available")
    return "\n".join(lines)


def send_discord(message: str) -> None:
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        print("  [DISCORD] DISCORD_WEBHOOK_URL not set - skipping notification.")
        return
    try:
        resp = requests.post(url, json={"content": message}, timeout=10)
        if resp.status_code not in (200, 204):
            print(f"  [DISCORD] HTTP {resp.status_code}: {resp.text.strip()}")
    except Exception as exc:
        print(f"  [DISCORD] Notification failed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover stop-loss exits that Alpaca filled outside a bot session.",
    )
    parser.add_argument(
        "--since", default=None, metavar="YYYY-MM-DD",
        help=f"Look back from this date (default: {DEFAULT_LOOKBACK_DAYS} days ago).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be written without touching the log.",
    )
    parser.add_argument(
        "--discord", action="store_true",
        help="Post a summary to Discord when rows are written.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.since:
        try:
            since = date.fromisoformat(args.since)
        except ValueError:
            print(f"[FATAL] --since must be YYYY-MM-DD, got {args.since!r}")
            return 1
    else:
        since = date.today() - timedelta(days=DEFAULT_LOOKBACK_DAYS)

    try:
        api = get_api()
    except Exception as exc:
        print(f"[FATAL] Could not connect to Alpaca: {exc}")
        return 1

    try:
        result = reconcile(api, since, dry_run=args.dry_run)
    except Exception as exc:
        print(f"[FATAL] Reconciliation failed: {exc}")
        return 1

    print(format_summary(result))

    if args.discord and result["written"] and not args.dry_run:
        send_discord(format_discord(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
