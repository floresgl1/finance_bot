"""
Uploads market reference CSVs and ticker OHLCV CSVs to PythonAnywhere via the Files API.

Credentials are read exclusively from environment variables:
    PYTHONANYWHERE_TOKEN    — API token (set as a GitHub Actions secret)
    PYTHONANYWHERE_USERNAME — PythonAnywhere account username

Exit codes:
    0 — all files uploaded successfully
    1 — one or more files failed, or no CSV files were found
"""

import os
import sys

import requests

from config import WATCHLIST
from market_data_collector import MARKET_SYMBOLS

# ---------------------------------------------------------------------------
# Credentials — injected from GitHub Actions secrets; never hard-coded.
# ---------------------------------------------------------------------------
_token = os.environ.get("PYTHONANYWHERE_TOKEN")
_username = os.environ.get("PYTHONANYWHERE_USERNAME")

if not _token or not _username:
    print(
        "[FATAL] PYTHONANYWHERE_TOKEN and PYTHONANYWHERE_USERNAME "
        "must be set as environment variables."
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths and file lists
# ---------------------------------------------------------------------------
API_BASE = f"https://www.pythonanywhere.com/api/v0/user/{_username}/files/path"

DIR_PAIRS = [
    (
        "data/market",
        f"/home/{_username}/finance_bot/data/market",
        [f"{symbol}.csv" for symbol in MARKET_SYMBOLS],
    ),
    (
        "data",
        f"/home/{_username}/finance_bot/data",
        [f"{ticker}.csv" for ticker in WATCHLIST],
    ),
]

# ---------------------------------------------------------------------------
# Upload each file
# ---------------------------------------------------------------------------
errors = 0
total = 0

for local_dir, remote_dir, files in DIR_PAIRS:
    for filename in files:
        local_path = os.path.join(local_dir, filename)
        remote_path = f"{remote_dir}/{filename}"
        url = f"{API_BASE}{remote_path}"
        total += 1

        try:
            with open(local_path, "rb") as fh:
                response = requests.post(
                    url,
                    headers={"Authorization": f"Token {_token}"},
                    files={"content": fh},
                    timeout=60,
                )

            if response.status_code in (200, 201):
                print(f"[OK]      {filename}  →  {remote_path}")
            else:
                print(
                    f"[ERROR]   {filename}  —  "
                    f"HTTP {response.status_code}: {response.text.strip()}"
                )
                errors += 1

        except requests.exceptions.Timeout:
            print(f"[ERROR]   {filename}  —  Request timed out after 60 s.")
            errors += 1
        except requests.exceptions.RequestException as exc:
            print(f"[ERROR]   {filename}  —  Network error: {exc}")
            errors += 1
        except OSError as exc:
            print(f"[ERROR]   {filename}  —  Could not read local file: {exc}")
            errors += 1

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
succeeded = total - errors
print(f"\nUpload complete: {succeeded}/{total} succeeded, {errors}/{total} failed.")

if errors:
    sys.exit(1)
