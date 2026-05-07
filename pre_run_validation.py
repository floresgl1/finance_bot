"""
Pre-flight market data freshness check.

Runs as a PythonAnywhere scheduled task (13:00 UTC) to catch a stale
update_market_data.yml workflow well before live_trader.py kicks off at
15:00 UTC. For each expected CSV (data/{TICKER}.csv for WATCHLIST plus
data/market/*.csv), reads the Date column, takes max(Date), and flags
anything whose latest row is not today's date. If anything is stale or
unreadable, posts a single Discord alert with the workflow re-trigger
link; otherwise exits silently.
"""

import os
import sys
from datetime import date
from glob import glob

import pandas as pd

from config import DATA_DIR, WATCHLIST


WORKFLOW_URL = (
    "https://github.com/floresgl907/finance_bot/actions/workflows/update_market_data.yml"
)


def _expected_csv_paths() -> list[str]:
    """Same construction as check_market_data_freshness in live_trader.py."""
    paths = [os.path.join(DATA_DIR, f"{t}.csv") for t in WATCHLIST]
    market_dir = os.path.join(DATA_DIR, "market")
    paths.extend(sorted(glob(os.path.join(market_dir, "*.csv"))))
    return paths


def _latest_date(path: str) -> date:
    df = pd.read_csv(path, usecols=["Date"])
    if df.empty:
        raise ValueError("file has no rows")
    return pd.to_datetime(df["Date"]).max().date()


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
    today = date.today()
    if today.weekday() >= 5:
        print("[SKIP] Weekend — no market data expected.")
        sys.exit(0)

    stale: list[tuple[str, date]] = []
    failed: list[tuple[str, str]] = []

    for path in _expected_csv_paths():
        fname = os.path.basename(path)
        try:
            latest = _latest_date(path)
        except Exception as exc:
            failed.append((fname, str(exc)))
            continue
        if latest != today:
            stale.append((fname, latest))

    if not stale and not failed:
        print(f"[OK] All market data CSVs current as of {today.isoformat()}.")
        return 0

    lines = ["🚨 **PRE-RUN VALIDATION — STALE MARKET DATA**"]
    if stale:
        lines.append("```")
        for fname, latest in stale:
            lines.append(f"  {fname:<16} last date: {latest.isoformat()}")
        lines.append("```")
    if failed:
        lines.append("Failed to read:")
        lines.append("```")
        for fname, reason in failed:
            lines.append(f"  {fname:<16} {reason}")
        lines.append("```")
    lines.append(f"Re-trigger workflow: {WORKFLOW_URL}")

    _send_discord("\n".join(lines))
    print(f"[ALERT] {len(stale)} stale, {len(failed)} unreadable. Discord notified.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
