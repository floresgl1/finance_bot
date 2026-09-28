"""
One-time setup script: creates (or updates) the PythonAnywhere daily tasks
that start GitHub workflows on time via dispatch_workflow.py.

    12:00 UTC  update_market_data.yml   (fresh CSVs before the 15:00 trade)
    14:15 UTC  agent_pretrade.yml       (agent decisions before the 15:00 trade)

These replace the workflows' own `schedule:` crons, which GitHub ran up to
seven hours late (see dispatch_workflow.py). PA daily tasks also fire at
weekends; dispatch_workflow.py skips Saturdays and Sundays itself.

Run this locally once after deploying to PythonAnywhere:

    python create_pa_dispatch_tasks.py

Requires in environment or .env:
    PYTHONANYWHERE_TOKEN    — API token from your PA account settings
    PYTHONANYWHERE_USERNAME — your PA username

Idempotent: a task whose command already runs dispatch_workflow.py for the
same workflow is updated in place rather than duplicated.
"""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

TASKS = [
    ("update_market_data.yml", 12, 0),
    ("agent_pretrade.yml", 14, 15),
]


def _command(username: str, workflow: str) -> str:
    return f"cd /home/{username}/finance_bot && python dispatch_workflow.py {workflow}"


def _find(tasks: list[dict], workflow: str) -> dict | None:
    for task in tasks:
        if f"dispatch_workflow.py {workflow}" in task.get("command", ""):
            return task
    return None


def main() -> int:
    token = os.environ.get("PYTHONANYWHERE_TOKEN")
    username = os.environ.get("PYTHONANYWHERE_USERNAME")
    if not token or not username:
        print("[FATAL] PYTHONANYWHERE_TOKEN and PYTHONANYWHERE_USERNAME "
              "must be set in your environment or .env file.")
        return 1

    api = f"https://www.pythonanywhere.com/api/v0/user/{username}/schedule/"
    headers = {"Authorization": f"Token {token}"}

    r = requests.get(api, headers=headers, timeout=30)
    r.raise_for_status()
    existing = r.json()

    failed = 0
    for workflow, hour, minute in TASKS:
        payload = {
            "command": _command(username, workflow),
            "enabled": True,
            "interval": "daily",
            "hour": hour,
            "minute": minute,
        }
        task = _find(existing, workflow)
        if task:
            r = requests.patch(f"{api}{task['id']}/", headers=headers, json=payload, timeout=30)
            action = "updated"
        else:
            r = requests.post(api, headers=headers, json=payload, timeout=30)
            action = "created"
        if r.status_code in (200, 201):
            t = r.json()
            print(f"[OK] {action} #{t['id']} — daily {hour:02d}:{minute:02d} UTC — {t['command']}")
        else:
            failed += 1
            print(f"[ERROR] {workflow}: HTTP {r.status_code}: {r.text.strip()}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
