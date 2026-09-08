"""
Pre-flight market data freshness check.

Runs as a PythonAnywhere scheduled task (13:00 UTC) to catch a stale
update_market_data.yml workflow well before live_trader.py kicks off at
15:00 UTC.

Freshness is judged by file MTIME, not by the Date column. The task runs
30 minutes before the 13:30 UTC open and the refresh workflow runs at
12:00 UTC, so no CSV can contain a bar dated today — comparing the Date
column against today would fail on every file, every day. What we actually
want to know is "did the refresh land today?", which is exactly what mtime
answers. This mirrors check_market_data_freshness() in live_trader.py.

Per expected CSV (data/{TICKER}.csv for WATCHLIST plus data/market/*.csv):
    missing     — file absent entirely
    stale       — mtime before today 00:00 UTC, or older than
                  MAX_MARKET_DATA_AGE_HOURS
    unreadable  — present and fresh, but the Date column will not parse
                  (live_trader would hit CSV_INVALID_SKIP on it)

The latest Date is reported alongside each file as context for a human
reading the alert, but never decides pass/fail.

Exit codes:
    0 — all CSVs fresh and readable (or non-trading day)
    1 — at least one file missing, stale, or unreadable; Discord notified
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from glob import glob

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from config import DATA_DIR, MAX_MARKET_DATA_AGE_HOURS, WATCHLIST


WORKFLOW_URL = (
    "https://github.com/floresgl1/finance_bot/actions/workflows/update_market_data.yml"
)


def _expected_csv_paths() -> list[str]:
    """Same construction as check_market_data_freshness in live_trader.py."""
    paths = [os.path.join(DATA_DIR, f"{t}.csv") for t in WATCHLIST]
    market_dir = os.path.join(DATA_DIR, "market")
    paths.extend(sorted(glob(os.path.join(market_dir, "*.csv"))))
    return paths


def _mtime_utc(path: str) -> datetime:
    """File modification time as a timezone-aware UTC datetime."""
    return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)


def _latest_date(path: str) -> str:
    """Latest value in the Date column, ISO formatted. Context only."""
    df = pd.read_csv(path, usecols=["Date"])
    if df.empty:
        raise ValueError("file has no rows")
    return pd.to_datetime(df["Date"]).max().date().isoformat()


def _send_discord(message: str) -> None:
    webhook = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook:
        print("[Discord] DISCORD_WEBHOOK_URL not set — skipping alert.")
        return
    try:
        import requests
        resp = requests.post(webhook, json={"content": message}, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[Discord] Alert failed: {exc}")


def main() -> int:
    now_utc = datetime.now(timezone.utc)

    if now_utc.weekday() >= 5:
        print(f"[SKIP] Weekend ({now_utc:%A}) — no market data expected.")
        return 0

    today_start_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    max_age = timedelta(hours=MAX_MARKET_DATA_AGE_HOURS)

    missing: list[str] = []
    stale: list[tuple[str, str]] = []
    unreadable: list[tuple[str, str]] = []
    fresh = 0

    for path in _expected_csv_paths():
        fname = os.path.basename(path)

        if not os.path.exists(path):
            missing.append(fname)
            continue

        mtime = _mtime_utc(path)

        if mtime < today_start_utc:
            stale.append((fname, f"refreshed {mtime:%Y-%m-%d %H:%M} UTC, before today 00:00"))
            continue

        if (now_utc - mtime) > max_age:
            age_h = (now_utc - mtime).total_seconds() / 3600
            stale.append((fname, f"{age_h:.1f}h old (limit {MAX_MARKET_DATA_AGE_HOURS}h)"))
            continue

        # Fresh by mtime — confirm the contents are actually parseable.
        try:
            latest = _latest_date(path)
        except Exception as exc:
            unreadable.append((fname, str(exc)))
            continue

        fresh += 1
        print(f"  [OK] {fname:<16} refreshed {mtime:%H:%M} UTC, last bar {latest}")

    if not missing and not stale and not unreadable:
        print(f"[OK] All {fresh} market data CSVs refreshed today ({now_utc:%Y-%m-%d} UTC).")
        return 0

    lines = ["\N{POLICE CARS REVOLVING LIGHT} **PRE-RUN VALIDATION — MARKET DATA NOT READY**"]
    if missing:
        lines.append("Missing:")
        lines.append("```")
        lines.extend(f"  {f}" for f in missing)
        lines.append("```")
    if stale:
        lines.append("Stale (refresh did not land today):")
        lines.append("```")
        lines.extend(f"  {f:<16} {why}" for f, why in stale)
        lines.append("```")
    if unreadable:
        lines.append("Unreadable:")
        lines.append("```")
        lines.extend(f"  {f:<16} {why}" for f, why in unreadable)
        lines.append("```")
    lines.append(f"Re-trigger workflow: {WORKFLOW_URL}")

    _send_discord("\n".join(lines))
    print(
        f"[ALERT] {len(missing)} missing, {len(stale)} stale, "
        f"{len(unreadable)} unreadable ({fresh} OK). Discord notified."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
