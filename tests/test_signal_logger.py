import csv
import os
import tempfile
import pytest
from signal_logger import _ensure_file, FIELDNAMES
import signal_logger


def _write_header(path, columns):
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerow(columns)


def test_happy_path_header_matches(monkeypatch, tmp_path):
    fake_csv = tmp_path / "signal_log.csv"
    _write_header(str(fake_csv), FIELDNAMES)
    monkeypatch.setattr(signal_logger, "SIGNAL_LOG_PATH", str(fake_csv))

    _ensure_file()


def test_file_does_not_exist(monkeypatch, tmp_path):
    fake_csv = tmp_path / "signal_log.csv"
    monkeypatch.setattr(signal_logger, "SIGNAL_LOG_PATH", str(fake_csv))

    _ensure_file()

    assert os.path.exists(str(fake_csv))
    with open(str(fake_csv)) as fh:
        header = next(csv.reader(fh))
    assert header == FIELDNAMES


def test_missing_columns(monkeypatch, tmp_path):
    fake_csv = tmp_path / "signal_log.csv"
    truncated = FIELDNAMES[:10]
    _write_header(str(fake_csv), truncated)
    monkeypatch.setattr(signal_logger, "SIGNAL_LOG_PATH", str(fake_csv))

    with pytest.raises(ValueError) as excinfo:
        _ensure_file()

    msg = str(excinfo.value)
    assert "does not match FIELDNAMES" in msg
    assert f"Expected ({len(FIELDNAMES)})" in msg
    assert "Found (10)" in msg
    assert "To fix:" in msg


def test_extra_columns(monkeypatch, tmp_path):
    fake_csv = tmp_path / "signal_log.csv"
    extended = FIELDNAMES + ["bogus_extra_col"]
    _write_header(str(fake_csv), extended)
    monkeypatch.setattr(signal_logger, "SIGNAL_LOG_PATH", str(fake_csv))

    with pytest.raises(ValueError) as excinfo:
        _ensure_file()

    msg = str(excinfo.value)
    assert f"Expected ({len(FIELDNAMES)})" in msg
    assert f"Found ({len(FIELDNAMES) + 1})" in msg
    assert "bogus_extra_col" in msg


def test_columns_wrong_order(monkeypatch, tmp_path):
    fake_csv = tmp_path / "signal_log.csv"
    swapped = [FIELDNAMES[1], FIELDNAMES[0]] + list(FIELDNAMES[2:])
    _write_header(str(fake_csv), swapped)
    monkeypatch.setattr(signal_logger, "SIGNAL_LOG_PATH", str(fake_csv))

    with pytest.raises(ValueError) as excinfo:
        _ensure_file()

    assert "does not match FIELDNAMES" in str(excinfo.value)


def test_column_renamed_typo(monkeypatch, tmp_path):
    fake_csv = tmp_path / "signal_log.csv"
    typo_header = [
        "shap_drvier_1" if name == "shap_driver_1" else name
        for name in FIELDNAMES
    ]
    _write_header(str(fake_csv), typo_header)
    monkeypatch.setattr(signal_logger, "SIGNAL_LOG_PATH", str(fake_csv))

    with pytest.raises(ValueError) as excinfo:
        _ensure_file()

    assert "shap_drvier_1" in str(excinfo.value)


def test_empty_file(monkeypatch, tmp_path):
    fake_csv = tmp_path / "signal_log.csv"
    fake_csv.write_bytes(b"")
    monkeypatch.setattr(signal_logger, "SIGNAL_LOG_PATH", str(fake_csv))

    with pytest.raises(ValueError) as excinfo:
        _ensure_file()

    msg = str(excinfo.value)
    assert "exists but is empty" in msg
    assert "To fix:" in msg
    assert "delete" in msg
    assert "investigate" in msg


def test_header_only_no_data(monkeypatch, tmp_path):
    fake_csv = tmp_path / "signal_log.csv"
    _write_header(str(fake_csv), FIELDNAMES)
    monkeypatch.setattr(signal_logger, "SIGNAL_LOG_PATH", str(fake_csv))

    _ensure_file()
