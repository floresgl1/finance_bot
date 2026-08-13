"""
pipeline_status.py — Read-only health snapshot of the live trading pipeline.

Collects, in one pass, the facts needed to answer "did today's run do what it
was supposed to do?":

    1. PythonAnywhere scheduled tasks — last exit code, success history
    2. Daily run guard   (data/last_run_date.txt)
    3. Halt flag         (data/halt_flag.txt)
    4. Trading activity  (data/signal_log.csv — today's ENTRY rows)
    5. News agent output (data/agent_runs/{date}.json)
    6. Deploy drift      (PA file contents vs. local git HEAD)
    7. Market calendar   (was today actually a trading day?)

This script is STRICTLY READ-ONLY. It issues only HTTP GETs and never writes,
deletes, or modifies anything locally or on PythonAnywhere. It is safe to run
at any time, including while the bot is trading.

Exit codes:
    0 — snapshot collected (regardless of what it found)
    2 — could not collect (missing credentials, network failure)

Credentials are read from .env / the environment and are never printed:
    PYTHONANYWHERE_TOKEN, PYTHONANYWHERE_USERNAME
    ALPACA_API_KEY, ALPACA_SECRET_KEY   (optional — calendar check)
"""

import hashlib
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(REPO_ROOT, ".env"))

TOKEN = os.environ.get("PYTHONANYWHERE_TOKEN")
USER = os.environ.get("PYTHONANYWHERE_USERNAME")

if not TOKEN or not USER:
    print("[FATAL] PYTHONANYWHERE_TOKEN and PYTHONANYWHERE_USERNAME must be set.")
    sys.exit(2)

H = {"Authorization": f"Token {TOKEN}"}
API = f"https://www.pythonanywhere.com/api/v0/user/{USER}"
FILES = f"{API}/files/path"
REMOTE = f"/home/{USER}/finance_bot"

TODAY = datetime.now(timezone.utc).date()
TODAY_STR = TODAY.isoformat()

# Files whose deployed copy must match the committed blob.
TRACKED = ["config.py", "run_bot.py", "live_trader.py", "predictor.py", "rebalancer.py"]


def head(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def get(path, timeout=30):
    """GET a PA file. Returns (status_code, text)."""
    try:
        r = requests.get(f"{FILES}{path}", headers=H, timeout=timeout)
        return r.status_code, r.text
    except requests.RequestException as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# 0. Header
# ---------------------------------------------------------------------------
print(f"Pipeline status — {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
print(f"UTC date: {TODAY_STR} ({TODAY:%A})")

# ---------------------------------------------------------------------------
# 1. Market calendar — was today a trading day?
# ---------------------------------------------------------------------------
head("1. MARKET CALENDAR")
is_trading_day = None
if TODAY.weekday() >= 5:
    is_trading_day = False
    print(f"  Weekend ({TODAY:%A}) — NOT a trading day.")
else:
    ak, sk = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not (ak and sk):
        print("  Weekday. Alpaca creds absent — holiday check SKIPPED.")
        print("  CAVEAT: treat as a trading day unless it is a market holiday.")
    else:
        try:
            r = requests.get(
                "https://paper-api.alpaca.markets/v2/calendar",
                headers={"APCA-API-KEY-ID": ak, "APCA-API-SECRET-KEY": sk},
                params={"start": TODAY_STR, "end": TODAY_STR},
                timeout=20,
            )
            if r.status_code == 200:
                days = r.json()
                is_trading_day = bool(days)
                if days:
                    print(f"  TRADING DAY — session {days[0].get('open')}–{days[0].get('close')} ET")
                else:
                    print("  Weekday but NOT a trading day (market holiday).")
            else:
                print(f"  Calendar lookup failed: HTTP {r.status_code}. Assume trading day.")
        except requests.RequestException as exc:
            print(f"  Calendar lookup error: {exc}. Assume trading day.")

# ---------------------------------------------------------------------------
# 2. Scheduled tasks
# ---------------------------------------------------------------------------
head("2. PYTHONANYWHERE SCHEDULED TASKS")
try:
    tasks = requests.get(f"{API}/schedule/", headers=H, timeout=30).json()
except (requests.RequestException, ValueError) as exc:
    print(f"  [FATAL] Could not list tasks: {exc}")
    sys.exit(2)

for t in sorted(tasks, key=lambda x: (x.get("hour", 0), x.get("minute", 0))):
    tid = t["id"]
    when = t.get("printable_time", "?")
    cmd = t.get("command", "")
    enabled = t.get("enabled")
    print(f"\n  [{when} UTC] task {tid}  enabled={enabled}")
    print(f"    command: {cmd}")

    # A bare "&& script.py" will not resolve on PATH — classic exit 127.
    tail_cmd = cmd.split("&&")[-1].strip()
    if tail_cmd.endswith(".py") and not tail_cmd.startswith(("python", "./", "/")):
        print("    [FAIL] command invokes a .py with no interpreter and no path —"
              " bash cannot resolve it (exit 127).")

    status, text = get(f"/var/log/schedule-log-{tid}.log")
    if status != 200:
        print(f"    log: unreadable (HTTP {status})")
        continue

    codes = [ln for ln in text.splitlines() if "return code was" in ln]
    if not codes:
        print("    log: no completed runs recorded")
        continue
    zeros = sum(" was 0." in c for c in codes)
    last = codes[-1].strip()
    print(f"    runs: {len(codes)}   exit-0: {zeros}   exit-nonzero: {len(codes) - zeros}")
    print(f"    last: {last[:88]}")
    if zeros == 0:
        print(f"    [FAIL] This task has NEVER succeeded in {len(codes)} recorded runs.")
    elif " was 0." not in last:
        print("    [WARN] Most recent run exited non-zero.")
    if TODAY_STR not in text and is_trading_day:
        print(f"    [WARN] No entry for {TODAY_STR} — task may not have fired yet today.")

# ---------------------------------------------------------------------------
# 3. Run guard + halt flag
# ---------------------------------------------------------------------------
head("3. RUN GUARD & HALT FLAG")
status, text = get(f"{REMOTE}/data/last_run_date.txt")
guard = text.strip() if status == 200 else None
if status == 200:
    print(f"  last_run_date.txt = {guard!r}")
    if guard == TODAY_STR:
        print("  Guard is stamped for today — live_trader reports it TRADED today.")
    else:
        print(f"  Guard is NOT today ({TODAY_STR}) — no trading recorded yet today.")
elif status == 404:
    print("  last_run_date.txt absent — no run has ever stamped the guard.")
else:
    print(f"  last_run_date.txt unreadable (HTTP {status})")

status, text = get(f"{REMOTE}/data/halt_flag.txt")
if status == 200:
    print(f"  [WARN] HALT FLAG PRESENT: {text.strip()[:200]}")
    print("         Trading is halted until this file is cleared.")
elif status == 404:
    print("  halt_flag.txt absent — not halted.")

# ---------------------------------------------------------------------------
# 4. Trading activity
# ---------------------------------------------------------------------------
head("4. SIGNAL LOG")
status, text = get(f"{REMOTE}/data/signal_log.csv")
todays_entries = None
if status != 200:
    print(f"  signal_log.csv unreadable (HTTP {status})")
else:
    lines = text.strip().splitlines()
    rows = lines[1:]
    dates = sorted({r.split(",")[0] for r in rows if r.strip()})
    todays = [r for r in rows if r.startswith(TODAY_STR + ",")]
    todays_entries = [r for r in todays if ",ENTRY," in r]
    print(f"  {len(rows)} rows;  most recent date logged: {dates[-1] if dates else 'none'}")
    print(f"  rows dated today ({TODAY_STR}): {len(todays)}  (ENTRY: {len(todays_entries)})")
    if dates:
        gap = (TODAY - datetime.fromisoformat(dates[-1]).date()).days
        if gap > 4:
            print(f"  [WARN] Nothing logged for {gap} days — pipeline may be silently idle.")
    for r in todays_entries[:8]:
        f = r.split(",")
        print(f"      {f[1]:6} {f[2]:5} qty={f[4]:>4}  conf={f[5]}")

# ---------------------------------------------------------------------------
# 5. News agent output
# ---------------------------------------------------------------------------
head("5. NEWS AGENT")
status, text = get(f"{REMOTE}/data/agent_runs/{TODAY_STR}.json")
if status == 200:
    print(f"  agent_runs/{TODAY_STR}.json present ({len(text)} bytes)")
elif status == 404:
    print(f"  agent_runs/{TODAY_STR}.json ABSENT")
    print("  (Runs on GitHub Actions at 20:30 UTC — expected absent before then.)")

# ---------------------------------------------------------------------------
# 6. Deploy drift
# ---------------------------------------------------------------------------
head("6. DEPLOY DRIFT (PA vs local git HEAD)")


def norm(b):
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


try:
    local_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  local HEAD: {local_sha}")
except (subprocess.CalledProcessError, FileNotFoundError) as exc:
    print(f"  [WARN] git unavailable: {exc}")
    local_sha = None

drift = 0
for rel in TRACKED:
    try:
        blob = subprocess.run(
            ["git", "show", f"HEAD:{rel}"],
            cwd=REPO_ROOT, capture_output=True, check=True,
        ).stdout
    except subprocess.CalledProcessError:
        continue
    status, _ = get(f"{REMOTE}/{rel}")
    if status != 200:
        print(f"  {rel:20} PA unreadable (HTTP {status})")
        drift += 1
        continue
    r = requests.get(f"{FILES}{REMOTE}/{rel}", headers=H, timeout=30)
    match = norm(blob) == norm(r.content)
    print(f"  {rel:20} {'match' if match else 'DRIFT — PA differs from HEAD'}")
    if not match:
        drift += 1
print(f"  {'No drift detected.' if drift == 0 else f'[WARN] {drift} file(s) out of sync — run git pull on PA.'}")

# ---------------------------------------------------------------------------
# 7. Deterministic invariant checks
# ---------------------------------------------------------------------------
head("7. INVARIANTS")
if is_trading_day is False:
    print("  Non-trading day — no trading expected. Guard should NOT be stamped.")
    if guard == TODAY_STR:
        print("  [FAIL] Guard stamped on a non-trading day — live_trader believes it traded.")
    else:
        print("  [OK] Guard not stamped, as expected.")
else:
    now_utc = datetime.now(timezone.utc)
    after_open = now_utc.hour >= 14  # 13:30 UTC open, allow settle time
    if not after_open:
        print(f"  Before market open ({now_utc:%H:%M} UTC) — nothing expected yet.")
    else:
        if guard == TODAY_STR and todays_entries:
            print("  [OK] Guard stamped AND signals logged today — run completed normally.")
        elif guard == TODAY_STR and not todays_entries:
            print("  [WARN] Guard stamped but NO ENTRY rows today.")
            print("         Either every ticker was skipped, or the guard was stamped")
            print("         by a run that did not trade. Inspect the 15:00 task log.")
        elif guard != TODAY_STR and todays_entries:
            print("  [WARN] Signals logged but guard NOT stamped — a crash may have")
            print("         occurred between the trade block and the guard write.")
            print("         A later run could re-trade and over-weight positions.")
        else:
            print("  [FAIL] Trading day, market open, but NO guard and NO signals.")
            print("         The bot did not trade today. Check the 15:00 task log.")

print(f"\n{'=' * 68}\nSnapshot complete. This script changed nothing.\n{'=' * 68}")
