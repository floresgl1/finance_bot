"""
pipeline_check.py — daily alarm: did every pipeline step produce today's output?

Run by a PythonAnywhere task at 15:45 UTC on weekdays, after the trade.

Exit codes are not enough to know a step worked. In September 2026 three
failures ran for weeks unnoticed: the 12:00 trigger printed [ERROR] but exited
0 (expired token), generate_signals.py exited 127 every day with nobody reading
the task log, and the pre-trade agent kept re-scoring stale signals. Every one
of them left the same trace: a step's output was not from today. So this
checks outputs, not exit codes.

    step              output checked                      written by
    ----------------  ----------------------------------  ----------------------
    market data       watchlist + market CSVs modified    update_market_data.yml
                      today (UTC)                         (12:00, PA-dispatched)
    signals           pending_signals.json "date" = today generate_signals.py (14:00)
    agent decisions   agent_decisions.json "date" = today agent_pretrade.yml (14:15)
    trade             last_run_date.txt = today, and no   live_trader.py (15:00)
                      HALT_FLAG.txt

Posts one Discord line every trading day: green when all pass, a list of
failures otherwise. A green line every day is deliberate: if the check itself
stops running, the missing line is the alarm. For *why* a step failed, run
pipeline_status.py or the daily-run-triage agent.

Exit codes: 0 all passed or not a trading day, 1 at least one step failed.
"""

import glob
import json
import os
import sys
from datetime import date, datetime, timezone

import requests
from dotenv import load_dotenv

import config

load_dotenv()


def _file_date_utc(path: str) -> date | None:
    try:
        return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc).date()
    except OSError:
        return None


def check_market_data(today: date, data_dir: str = config.DATA_DIR,
                      watchlist: list[str] = config.WATCHLIST) -> tuple[bool, str]:
    paths = [os.path.join(data_dir, f"{t}.csv") for t in watchlist]
    paths += sorted(glob.glob(os.path.join(data_dir, "market", "*.csv")))
    stale = [os.path.basename(p) for p in paths if _file_date_utc(p) != today]
    if stale:
        shown = ", ".join(stale[:4]) + (f" +{len(stale) - 4} more" if len(stale) > 4 else "")
        return False, f"{len(stale)} of {len(paths)} CSVs not refreshed today ({shown})"
    return True, f"{len(paths)} CSVs refreshed today"


def check_dated_json(path: str, today: date) -> tuple[bool, str]:
    name = os.path.basename(path)
    try:
        with open(path) as f:
            stamped = json.load(f).get("date")
    except FileNotFoundError:
        return False, f"{name} missing"
    except (OSError, ValueError, AttributeError) as exc:
        return False, f"{name} unreadable ({type(exc).__name__})"
    if stamped != today.isoformat():
        return False, f"{name} is for {stamped}, not {today.isoformat()}"
    return True, f"{name} is for today"


def check_trade(today: date, guard_path: str = config.LAST_RUN_GUARD_PATH,
                halt_path: str = config.HALT_FLAG_PATH) -> tuple[bool, str]:
    if os.path.exists(halt_path):
        try:
            with open(halt_path) as f:
                reason = f.read().strip().splitlines()[0][:120]
        except (OSError, IndexError):
            reason = "unreadable"
        return False, f"HALT_FLAG.txt present — trading halted ({reason})"
    try:
        with open(guard_path) as f:
            stamped = f.read().strip()
    except FileNotFoundError:
        return False, "run guard missing — no run has ever completed"
    if stamped != today.isoformat():
        return False, f"run guard is {stamped or 'empty'} — today's trade did not complete"
    return True, "trade completed today"


def run_checks(today: date) -> list[tuple[str, bool, str]]:
    return [
        ("market data", *check_market_data(today)),
        ("signals", *check_dated_json(config.PENDING_SIGNALS_PATH, today)),
        ("agent decisions", *check_dated_json(config.AGENT_DECISIONS_PATH, today)),
        ("trade", *check_trade(today)),
    ]


def is_trading_day(today: date) -> bool:
    """Weekend → False. Weekday → ask Alpaca's calendar; assume open if unavailable."""
    if today.weekday() >= 5:
        return False
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not (key and secret):
        return True
    try:
        r = requests.get(
            "https://paper-api.alpaca.markets/v2/calendar",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            params={"start": today.isoformat(), "end": today.isoformat()},
            timeout=20,
        )
        r.raise_for_status()
        return bool(r.json())
    except (requests.RequestException, ValueError):
        return True     # a false alarm on a holiday beats a missed failure


def format_report(today: date, results: list[tuple[str, bool, str]]) -> str:
    failed = [r for r in results if not r[1]]
    if not failed:
        return f"✅ **Pipeline check {today}** — all {len(results)} steps produced today's output."
    lines = [f"🚨 **Pipeline check {today}** — {len(failed)} of {len(results)} steps FAILED"]
    for name, ok, detail in results:
        lines.append(f"{'✅' if ok else '❌'} {name}: {detail}")
    lines.append("Diagnose with `python pipeline_status.py` or the daily-run-triage agent.")
    return "\n".join(lines)


def send_discord(message: str) -> None:
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return
    try:
        requests.post(url, json={"content": message}, timeout=10).raise_for_status()
    except requests.RequestException as exc:
        print(f"  [Discord] post failed: {exc}")


def main(today: date | None = None) -> int:
    today = today or datetime.now(timezone.utc).date()
    if not is_trading_day(today):
        print(f"[SKIP] {today} is not a trading day.")
        return 0
    results = run_checks(today)
    report = format_report(today, results)
    print(report)
    send_discord(report)
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
