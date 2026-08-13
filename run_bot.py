"""
run_bot.py — Launcher wrapper for live_trader.py.

Runs live_trader.py as a subprocess and sends a Discord alert if it exits
with a non-zero return code.

Daily run guard: this module READS LAST_RUN_GUARD_PATH to skip a duplicate
run, but never writes it. live_trader.py owns the write, because only it
knows whether the market was open and trades actually executed — a zero exit
code means both "traded" and "market closed", so the write cannot live here.
"""

import os
import sys
import subprocess
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from config import LAST_RUN_GUARD_PATH, today_utc

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


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
    today = today_utc()

    try:
        with open(LAST_RUN_GUARD_PATH, "r") as f:
            last_run = f.read().strip()
        if last_run == today:
            print(f"[INFO] Bot already traded today ({today}), skipping.")
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


if __name__ == "__main__":
    main()
