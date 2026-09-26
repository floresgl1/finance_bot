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


# --- position_id ------------------------------------------------------------


def _write_rows(path, fieldnames, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_legacy_header_is_migrated_not_rejected(monkeypatch, tmp_path):
    """The live log predates position_id; the first run after deploy must not halt."""
    fake_csv = tmp_path / "signal_log.csv"
    _write_rows(str(fake_csv), signal_logger.LEGACY_FIELDNAMES, [
        {"date": "2026-06-25", "ticker": "AAPL", "row_type": "ENTRY", "entry_order_id": "o1"},
        {"date": "2026-06-26", "ticker": "AAPL", "row_type": "EXIT", "realized_pnl": "-42.25"},
    ])
    monkeypatch.setattr(signal_logger, "SIGNAL_LOG_PATH", str(fake_csv))

    _ensure_file()

    with open(str(fake_csv)) as fh:
        assert next(csv.reader(fh)) == FIELDNAMES
    rows = _read_rows(str(fake_csv))
    assert [r["entry_order_id"] for r in rows] == ["o1", ""]
    assert rows[1]["realized_pnl"] == "-42.25"
    assert all(r["position_id"] == "" for r in rows)
    assert os.path.exists(str(fake_csv) + ".pre_position_id.bak")


def test_migration_refuses_rows_wider_than_the_header(monkeypatch, tmp_path):
    fake_csv = tmp_path / "signal_log.csv"
    with open(str(fake_csv), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(signal_logger.LEGACY_FIELDNAMES)
        w.writerow(["x"] * (len(signal_logger.LEGACY_FIELDNAMES) + 1))
    monkeypatch.setattr(signal_logger, "SIGNAL_LOG_PATH", str(fake_csv))

    with pytest.raises(ValueError, match="refusing to migrate"):
        _ensure_file()
    with open(str(fake_csv)) as fh:
        assert next(csv.reader(fh)) == signal_logger.LEGACY_FIELDNAMES


def test_log_signal_and_log_exit_write_position_id(monkeypatch, tmp_path):
    fake_csv = tmp_path / "signal_log.csv"
    monkeypatch.setattr(signal_logger, "SIGNAL_LOG_PATH", str(fake_csv))

    signal_logger.log_signal("AAPL", "BUY", 100.0, 10, 60.0, "BUY",
                             entry_order_id="o1", position_id="o1")
    signal_logger.log_signal("AAPL", "HOLD", 100.0, 0, 40.0, "HOLD")
    signal_logger.log_exit("AAPL", "o1", 100.0, 110.0, "REBALANCE_TRIM", 3, position_id="o1")

    assert [r["position_id"] for r in _read_rows(str(fake_csv))] == ["o1", "", "o1"]


def test_find_position_id_missing_log(monkeypatch, tmp_path):
    monkeypatch.setattr(signal_logger, "SIGNAL_LOG_PATH", str(tmp_path / "nope.csv"))
    assert signal_logger.find_position_id("AAPL") is None


def test_find_position_id_none_for_pre_position_id_history(monkeypatch, tmp_path):
    fake_csv = tmp_path / "signal_log.csv"
    _write_rows(str(fake_csv), FIELDNAMES, [
        {"date": "2026-04-08", "ticker": "NVDA", "row_type": "ENTRY", "entry_order_id": ""},
    ])
    monkeypatch.setattr(signal_logger, "SIGNAL_LOG_PATH", str(fake_csv))
    assert signal_logger.find_position_id("NVDA") is None


def test_find_position_id_survives_repeated_trims(monkeypatch, tmp_path):
    """The AAPL case: one BUY, four trims, then a take-profit. Every exit links."""
    fake_csv = tmp_path / "signal_log.csv"
    rows = [{"date": "2026-06-25", "ticker": "AAPL", "row_type": "ENTRY",
             "entry_order_id": "o1", "position_id": "o1"}]
    for d in ("2026-06-26", "2026-06-30", "2026-07-01", "2026-07-02"):
        rows.append({"date": d, "ticker": "AAPL", "row_type": "EXIT", "position_id": "o1"})
    _write_rows(str(fake_csv), FIELDNAMES, rows)
    monkeypatch.setattr(signal_logger, "SIGNAL_LOG_PATH", str(fake_csv))

    assert signal_logger.find_position_id("AAPL") == "o1"


def test_find_position_id_is_per_ticker_and_newest_wins(monkeypatch, tmp_path):
    fake_csv = tmp_path / "signal_log.csv"
    _write_rows(str(fake_csv), FIELDNAMES, [
        {"date": "2026-06-24", "ticker": "NVDA", "position_id": "posA"},
        {"date": "2026-08-04", "ticker": "NVDA", "position_id": "posB"},
        {"date": "2026-08-05", "ticker": "MSFT", "position_id": "posM"},
        # A reconciled stop for posA appended later but dated earlier must
        # not make posA look current again.
        {"date": "2026-08-03", "ticker": "NVDA", "row_type": "EXIT", "position_id": "posA"},
    ])
    monkeypatch.setattr(signal_logger, "SIGNAL_LOG_PATH", str(fake_csv))

    assert signal_logger.find_position_id("NVDA") == "posB"
    assert signal_logger.find_position_id("MSFT") == "posM"
