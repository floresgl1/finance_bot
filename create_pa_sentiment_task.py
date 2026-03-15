"""
One-time setup script: creates (or updates) a PythonAnywhere daily scheduled
task that runs sentiment_collector.py at 13:30 UTC.

Run this locally once after deploying to PythonAnywhere:

    python create_pa_sentiment_task.py

Requires in environment or .env:
    PYTHONANYWHERE_TOKEN    — API token from your PA account settings
    PYTHONANYWHERE_USERNAME — your PA username

The script is idempotent: if a task whose command ends with
'sentiment_collector.py' already exists it updates that task in-place
rather than creating a duplicate.
"""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN    = os.environ.get("PYTHONANYWHERE_TOKEN")
USERNAME = os.environ.get("PYTHONANYWHERE_USERNAME")

if not TOKEN or not USERNAME:
    print(
        "[FATAL] PYTHONANYWHERE_TOKEN and PYTHONANYWHERE_USERNAME "
        "must be set in your environment or .env file."
    )
    sys.exit(1)

API_BASE  = f"https://www.pythonanywhere.com/api/v0/user/{USERNAME}"
HEADERS   = {"Authorization": f"Token {TOKEN}"}
COMMAND   = f"cd /home/{USERNAME}/finance_bot && python sentiment_collector.py"
TASK_HOUR = 13
TASK_MIN  = 30


def _get_existing_task() -> dict | None:
    """Return the first existing task whose command contains sentiment_collector.py, or None."""
    r = requests.get(f"{API_BASE}/schedule/", headers=HEADERS, timeout=30)
    r.raise_for_status()
    for task in r.json():
        if "sentiment_collector.py" in task.get("command", ""):
            return task
    return None


def main():
    print(f"Checking for existing sentiment_collector task on PA ({USERNAME}) ...")
    existing = _get_existing_task()

    payload = {
        "command":  COMMAND,
        "enabled":  True,
        "interval": "daily",
        "hour":     TASK_HOUR,
        "minute":   TASK_MIN,
    }

    if existing:
        task_id = existing["id"]
        print(f"  Found existing task #{task_id} — updating ...")
        r = requests.patch(
            f"{API_BASE}/schedule/{task_id}/",
            headers=HEADERS,
            json=payload,
            timeout=30,
        )
        action = "updated"
    else:
        print("  No existing task found — creating ...")
        r = requests.post(
            f"{API_BASE}/schedule/",
            headers=HEADERS,
            json=payload,
            timeout=30,
        )
        action = "created"

    if r.status_code in (200, 201):
        task = r.json()
        print(
            f"[OK] Task {action}: #{task['id']} — "
            f"daily at {TASK_HOUR:02d}:{TASK_MIN:02d} UTC\n"
            f"     command: {task['command']}"
        )
    else:
        print(f"[ERROR] HTTP {r.status_code}: {r.text.strip()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
