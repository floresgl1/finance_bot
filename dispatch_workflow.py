"""
Start a GitHub Actions workflow on time, from a PythonAnywhere scheduled task.

GitHub's `schedule:` trigger is best-effort: on 2026-09-28 the 12:00 UTC
market-data run started at 19:14 and the 14:15 UTC pre-trade agent at 20:32,
so the 15:00 UTC trade halted on stale data and a late webhook run traded at
19:17 with the previous session's agent decisions. PythonAnywhere tasks run on
time, so PA is the clock and GitHub stays the worker: this script asks GitHub
to start a workflow via `workflow_dispatch`, which begins within minutes.

    python dispatch_workflow.py update_market_data.yml
    python dispatch_workflow.py agent_pretrade.yml

Requires in environment or .env:
    GITHUB_DISPATCH_TOKEN — fine-grained token, this repo only, Actions: read and write
    DISCORD_WEBHOOK_URL   — optional; failures are posted there

Skips Saturdays and Sundays (UTC), matching the old `1-5` cron day field.
Exit codes: 0 dispatched or skipped, 1 GitHub refused or unreachable.
"""

import os
import sys
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

REPO = "floresgl1/finance_bot"
REF = "main"
API = "https://api.github.com"
ATTEMPTS = 3
BACKOFF_SECONDS = 10


def send_discord(message: str) -> None:
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return
    try:
        requests.post(url, json={"content": message}, timeout=10).raise_for_status()
    except Exception as exc:  # noqa: BLE001 - an alert must never mask the real failure
        print(f"  [Discord] Alert failed: {exc}")


def is_weekend(now: datetime) -> bool:
    return now.weekday() >= 5


def dispatch(workflow: str, token: str, sleep=time.sleep) -> tuple[bool, str]:
    """POST a workflow_dispatch. Retries network errors and 5xx; a 4xx is final."""
    url = f"{API}/repos/{REPO}/actions/workflows/{workflow}/dispatches"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    detail = ""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            r = requests.post(url, headers=headers, json={"ref": REF}, timeout=30)
        except requests.RequestException as exc:
            detail = f"{type(exc).__name__}: {exc}"
        else:
            if r.status_code == 204:
                return True, "dispatched"
            detail = f"HTTP {r.status_code}: {r.text.strip()[:300]}"
            if r.status_code < 500:
                return False, detail
        if attempt < ATTEMPTS:
            sleep(BACKOFF_SECONDS * attempt)
    return False, detail


def main(argv: list[str] | None = None, now: datetime | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python dispatch_workflow.py <workflow file, e.g. update_market_data.yml>")
        return 1
    workflow = argv[0]
    now = now or datetime.now(timezone.utc)

    if is_weekend(now):
        print(f"[SKIP] {now:%A} UTC — {workflow} runs Monday to Friday only.")
        return 0

    token = os.getenv("GITHUB_DISPATCH_TOKEN")
    if not token:
        msg = f"GITHUB_DISPATCH_TOKEN is not set; {workflow} was NOT started."
        print(f"[FATAL] {msg}")
        send_discord(f"🚨 **Workflow dispatch failed** — {msg}")
        return 1

    ok, detail = dispatch(workflow, token)
    if ok:
        print(f"[OK] {workflow} dispatched on {REF} at {now:%Y-%m-%d %H:%M} UTC")
        return 0
    print(f"[FATAL] {workflow} dispatch failed — {detail}")
    send_discord(f"🚨 **Workflow dispatch failed** — `{workflow}` was NOT started "
                 f"at {now:%H:%M} UTC.\n{detail}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
