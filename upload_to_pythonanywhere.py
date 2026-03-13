"""
Uploads all CSVs from data/market/ to PythonAnywhere via the Files API.

Credentials are read exclusively from environment variables:
    PYTHONANYWHERE_TOKEN    — API token (set as a GitHub Actions secret)
    PYTHONANYWHERE_USERNAME — PythonAnywhere account username

Exit codes:
    0 — all files uploaded successfully
    1 — one or more files failed, or no CSV files were found
"""

import glob
import os
import sys

import requests

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
# Paths
# ---------------------------------------------------------------------------
LOCAL_MARKET_DIR = os.path.join("data", "market")
REMOTE_MARKET_DIR = f"/home/{_username}/finance_bot/data/market"
API_BASE = f"https://www.pythonanywhere.com/api/v0/user/{_username}/files/path"

# ---------------------------------------------------------------------------
# Discover local CSVs
# ---------------------------------------------------------------------------
csv_files = sorted(glob.glob(os.path.join(LOCAL_MARKET_DIR, "*.csv")))

if not csv_files:
    print(f"[WARNING] No CSV files found in '{LOCAL_MARKET_DIR}'. Nothing to upload.")
    sys.exit(1)

print(f"Found {len(csv_files)} CSV file(s) in '{LOCAL_MARKET_DIR}'. Uploading…\n")

# ---------------------------------------------------------------------------
# Upload each CSV
# ---------------------------------------------------------------------------
errors = 0

for local_path in csv_files:
    filename = os.path.basename(local_path)
    remote_path = f"{REMOTE_MARKET_DIR}/{filename}"
    url = f"{API_BASE}{remote_path}"

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
total = len(csv_files)
succeeded = total - errors
print(f"\nUpload complete: {succeeded}/{total} succeeded, {errors}/{total} failed.")

if errors:
    sys.exit(1)
