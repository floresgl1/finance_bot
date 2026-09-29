"""
Download Sharadar's full-history bulk files for the micro-cap value test.

docs/MICROCAP_VALUE_PREREG.md needs four tables: tickers, actions,
fundamentals, stocks. Each bulk file is a zipped CSV behind

    GET https://api.sharadar.com/v1.0/data/{table}?api_key=...&years=full

which redirects to a time-limited download URL.

Files go to data/sharadar/ (git-ignored: the licence forbids republishing).
A manifest of file names, sizes and SHA-256 hashes goes to
docs/sharadar_manifest.json, which IS committed: it records exactly which
files the test used, without containing any data.

    python sharadar_download.py                  # all four, skip ones on disk
    python sharadar_download.py --tables tickers # just one
    python sharadar_download.py --force          # re-download existing files

Needs SHARADAR_API_KEY in the environment. Never prints the key or the
signed redirect URL. Exit 0 only if every requested file is on disk, is a
valid zip, and is in the manifest.
"""

import argparse
import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "https://api.sharadar.com/v1.0/data"
TABLES = ["tickers", "actions", "fundamentals", "stocks"]
OUT_DIR = Path("data/sharadar")
MANIFEST = Path("docs/sharadar_manifest.json")
CHUNK = 1 << 20


class DownloadError(Exception):
    pass


def zip_path(out_dir, table):
    return Path(out_dir) / f"{table}.csv.zip"


def download(session, api_key, table, dest):
    """Stream one bulk file to dest via a .part file; returns response headers.

    Errors never include a URL: both the request and the redirect carry secrets.
    """
    part = dest.with_name(dest.name + ".part")
    try:
        with session.get(f"{BASE_URL}/{table}", params={"api_key": api_key, "years": "full"},
                         stream=True, allow_redirects=True, timeout=(30, 300)) as resp:
            if resp.status_code != 200:
                raise DownloadError(f"{table}: HTTP {resp.status_code}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(part, "wb") as f:
                for chunk in resp.iter_content(CHUNK):
                    f.write(chunk)
            headers = dict(resp.headers)
    except requests.RequestException as exc:
        part.unlink(missing_ok=True)
        raise DownloadError(f"{table}: network error ({type(exc).__name__})") from None
    except DownloadError:
        part.unlink(missing_ok=True)
        raise
    try:
        check_zip(part, table)
    except DownloadError:
        part.unlink(missing_ok=True)
        raise
    part.replace(dest)
    return headers


def check_zip(path, table):
    """The file must be a zip whose members all read back intact."""
    if not zipfile.is_zipfile(path):
        raise DownloadError(f"{table}: not a zip file (an error page or a wrong key?)")
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad is not None:
            raise DownloadError(f"{table}: corrupt member {bad}")
        if not z.namelist():
            raise DownloadError(f"{table}: empty zip")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(path, headers=None):
    """Manifest entry: metadata only, never contents."""
    with zipfile.ZipFile(path) as z:
        members = [{"name": i.filename, "bytes": i.file_size} for i in z.infolist()]
    entry = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "members": members,
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if headers and headers.get("Last-Modified"):
        entry["last_modified"] = headers["Last-Modified"]
    return entry


def load_manifest(path):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"source": "api.sharadar.com bulk download, years=full", "tables": {}}


def save_manifest(manifest, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None, session=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tables", nargs="+", choices=TABLES, default=TABLES)
    parser.add_argument("--force", action="store_true", help="re-download files already on disk")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)

    api_key = os.environ.get("SHARADAR_API_KEY", "").strip()
    session = session or requests.Session()
    manifest = load_manifest(args.manifest)
    failed = 0

    for table in args.tables:
        dest = zip_path(args.out_dir, table)
        headers = None
        try:
            if dest.exists() and not args.force:
                check_zip(dest, table)
                print(f"  KEEP  {table}: already on disk")
            else:
                if not api_key:
                    raise DownloadError(f"{table}: SHARADAR_API_KEY is not set")
                print(f"  GET   {table} ...", flush=True)
                headers = download(session, api_key, table, dest)
            entry = describe(dest, headers)
            previous = manifest["tables"].get(table)
            if previous and previous["sha256"] == entry["sha256"]:
                entry["recorded_utc"] = previous["recorded_utc"]
                entry.setdefault("last_modified", previous.get("last_modified"))
            manifest["tables"][table] = {k: v for k, v in entry.items() if v is not None}
            print(f"  OK    {table}: {entry['bytes'] / 1e6:,.1f} MB, sha256 {entry['sha256'][:12]}")
        except DownloadError as exc:
            failed += 1
            print(f"  FAIL  {exc}")

    save_manifest(manifest, args.manifest)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
