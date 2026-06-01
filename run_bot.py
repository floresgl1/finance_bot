"""
run_bot.py — Launcher wrapper for live_trader.py.

Runs live_trader.py as a subprocess and sends a Discord alert if it exits
with a non-zero return code.
"""

import os
import sys
import subprocess
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
LAST_RUN_FILE = os.path.join(os.path.dirname(__file__), "data", "last_run_date.txt")


def send_discord(message: str) -> None:
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        import requests
        resp = requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": message},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as exc:
        print(f"  [Discord] Alert failed: {exc}")


def main() -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        with open(LAST_RUN_FILE, "r") as f:
            last_run = f.read().strip()
        if last_run == today:
            print(f"[INFO] Bot already ran today ({today}), skipping.")
            sys.exit(0)
    except (FileNotFoundError, OSError):
        pass

    script = os.path.join(os.path.dirname(__file__), "live_trader.py")
    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        send_discord(
            f"\N{WARNING SIGN} **Finance Bot — live_trader.py failed** | {timestamp}\n"
            f"Exit code: `{result.returncode}`"
        )
        sys.exit(result.returncode)

    os.makedirs(os.path.dirname(LAST_RUN_FILE), exist_ok=True)
    with open(LAST_RUN_FILE, "w") as f:
        f.write(today)


if __name__ == "__main__":
    main()
