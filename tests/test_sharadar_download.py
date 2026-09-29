"""sharadar_download.py against a fake session serving small zips.

Pins: the request shape, streaming to disk only after the zip checks out,
skipping files already on disk, the manifest, and that neither the key nor
the signed redirect URL is ever printed.
"""

import io
import json
import zipfile

import pytest
import requests

import sharadar_download as sd

KEY = "secret-key-123"
SIGNED = "https://storage.example/stocks.zip?signature=SIGNED-SECRET"


def _zip_bytes(name="SHARADAR_TICKERS.csv", text="ticker,permaticker\nAAA,1\n"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(name, text)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, body=b"", status_code=200, headers=None):
        self._body = body
        self.status_code = status_code
        self.headers = headers or {"Last-Modified": "Tue, 29 Sep 2026 05:33:00 GMT"}
        self.url = SIGNED

    def iter_content(self, size):
        for i in range(0, len(self._body), 7):
            yield self._body[i:i + 7]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, responses):
        self.responses = responses  # table -> FakeResponse or Exception
        self.calls = []

    def get(self, url, params=None, **kwargs):
        table = url.rsplit("/", 1)[-1]
        self.calls.append((table, dict(params), kwargs))
        served = self.responses[table]
        if isinstance(served, Exception):
            raise served
        return served


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", KEY)
    out, manifest = tmp_path / "sharadar", tmp_path / "manifest.json"
    return out, manifest, ["--out-dir", str(out), "--manifest", str(manifest)]


def test_downloads_full_history_and_records_the_manifest(paths, capsys):
    out, manifest, args = paths
    session = FakeSession({"tickers": FakeResponse(_zip_bytes())})
    assert sd.main(["--tables", "tickers", *args], session=session) == 0

    table, params, kwargs = session.calls[0]
    assert params == {"api_key": KEY, "years": "full"}
    assert kwargs["stream"] and kwargs["allow_redirects"]

    assert (out / "tickers.csv.zip").exists()
    assert not (out / "tickers.csv.zip.part").exists()
    entry = json.loads(manifest.read_text())["tables"]["tickers"]
    assert entry["file"] == "tickers.csv.zip"
    assert entry["sha256"] == sd.sha256(out / "tickers.csv.zip")
    assert entry["members"] == [{"name": "SHARADAR_TICKERS.csv", "bytes": len("ticker,permaticker\nAAA,1\n")}]
    assert entry["last_modified"] == "Tue, 29 Sep 2026 05:33:00 GMT"

    printed = capsys.readouterr().out
    assert KEY not in printed and "SIGNED-SECRET" not in printed


def test_an_error_page_is_not_saved(paths, capsys):
    out, manifest, args = paths
    session = FakeSession({"tickers": FakeResponse(b'{"error": "bad key"}')})
    assert sd.main(["--tables", "tickers", *args], session=session) == 1
    assert not (out / "tickers.csv.zip").exists()
    assert not (out / "tickers.csv.zip.part").exists()
    assert "not a zip file" in capsys.readouterr().out
    assert "tickers" not in json.loads(manifest.read_text())["tables"]


def test_http_error_and_network_error_leak_nothing(paths, capsys):
    out, manifest, args = paths
    session = FakeSession({
        "tickers": FakeResponse(status_code=403),
        "actions": requests.ConnectionError(f"failed {SIGNED}&api_key={KEY}"),
    })
    assert sd.main(["--tables", "tickers", "actions", *args], session=session) == 1
    printed = capsys.readouterr().out
    assert "tickers: HTTP 403" in printed
    assert "actions: network error (ConnectionError)" in printed
    assert KEY not in printed and "SIGNED-SECRET" not in printed


def test_a_failed_redownload_keeps_the_good_file(paths):
    out, manifest, args = paths
    good = _zip_bytes()
    sd.main(["--tables", "tickers", *args], session=FakeSession({"tickers": FakeResponse(good)}))
    bad = FakeSession({"tickers": FakeResponse(b"not a zip")})
    assert sd.main(["--tables", "tickers", "--force", *args], session=bad) == 1
    assert (out / "tickers.csv.zip").read_bytes() == good


def test_existing_file_is_kept_without_a_request_or_a_key(paths, monkeypatch, capsys):
    out, manifest, args = paths
    sd.main(["--tables", "tickers", *args],
            session=FakeSession({"tickers": FakeResponse(_zip_bytes())}))
    first = json.loads(manifest.read_text())["tables"]["tickers"]

    monkeypatch.delenv("SHARADAR_API_KEY")
    session = FakeSession({})
    assert sd.main(["--tables", "tickers", *args], session=session) == 0
    assert session.calls == []
    assert "KEEP  tickers" in capsys.readouterr().out
    assert json.loads(manifest.read_text())["tables"]["tickers"] == first


def test_corrupt_file_on_disk_fails_instead_of_being_kept(paths, capsys):
    out, manifest, args = paths
    out.mkdir(parents=True)
    (out / "tickers.csv.zip").write_bytes(b"truncated")
    assert sd.main(["--tables", "tickers", *args], session=FakeSession({})) == 1
    assert "not a zip file" in capsys.readouterr().out


def test_missing_key_fails_only_the_files_that_need_downloading(paths, monkeypatch, capsys):
    out, manifest, args = paths
    monkeypatch.delenv("SHARADAR_API_KEY")
    assert sd.main(["--tables", "tickers", *args], session=FakeSession({})) == 1
    assert "SHARADAR_API_KEY is not set" in capsys.readouterr().out


def test_manifest_has_no_data_values(paths):
    out, manifest, args = paths
    body = _zip_bytes(text="ticker,closeadj\nAAA,123.45\n")
    sd.main(["--tables", "stocks", *args], session=FakeSession({"stocks": FakeResponse(body)}))
    assert "123.45" not in manifest.read_text()
