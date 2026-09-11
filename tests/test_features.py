"""Tests for features.py — CSV loading, cleaning, and the staleness tripwire.

Two things matter most here. The loader has to survive the malformed CSVs
yfinance has historically produced (multi-level headers leaking in as data
rows), because a stray row silently becomes a NaN feature the model trains on.
And the Layer 3 tripwire is the last defence against generating signals from
stale prices if the Layer 1 gate is ever bypassed.
"""

from datetime import date, timedelta

from pandas.tseries.offsets import BDay

import pandas as pd
import pytest

import config
import features
from features import (
    SECTOR_MAP,
    StaleMarketDataError,
    _load_market_close,
    load_and_process,
)


# --- fixtures --------------------------------------------------------------


def _price_rows(last_date: date, rows: int = 120) -> pd.DataFrame:
    """Enough rows to clear the indicator warm-up (SMA_50 etc.)."""
    dates = pd.bdate_range(end=pd.Timestamp(last_date), periods=rows)
    return pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Open": [100.0 + i * 0.1 for i in range(rows)],
        "High": [101.0 + i * 0.1 for i in range(rows)],
        "Low": [99.0 + i * 0.1 for i in range(rows)],
        "Close": [100.5 + i * 0.1 for i in range(rows)],
        "Volume": [1_000_000 + i for i in range(rows)],
    })


@pytest.fixture
def data_env(monkeypatch, tmp_path):
    """A DATA_DIR plus a market/ directory with every symbol features needs."""
    market_dir = tmp_path / "market"
    market_dir.mkdir()

    monkeypatch.setattr(features, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(features, "MARKET_DATA_DIR", str(market_dir))
    # _load_market_close is lru_cached; a stale cache would leak between tests.
    _load_market_close.cache_clear()

    today = date.today()
    for symbol in ("SPY", "XLK", "XLF", "XLE", "XLV", "XLY", "^VIX"):
        _price_rows(today).to_csv(market_dir / f"{symbol}.csv", index=False)

    yield tmp_path
    _load_market_close.cache_clear()


# --- sector map ------------------------------------------------------------


def test_every_watchlist_ticker_has_a_sector():
    """A missing entry silently falls back to SPY, which quietly turns the
    sector-relative features into duplicates of the market features."""
    missing = [t for t in config.WATCHLIST if t not in SECTOR_MAP]
    assert missing == [], f"tickers with no sector ETF: {missing}"


def test_sector_map_has_no_tickers_outside_the_watchlist():
    extra = [t for t in SECTOR_MAP if t not in config.WATCHLIST]
    assert extra == [], f"sector entries for untraded tickers: {extra}"


def test_every_sector_etf_is_collected_by_the_market_collector():
    """A sector ETF that market_data_collector.py does not download would
    raise FileNotFoundError on the first ticker that maps to it."""
    from market_data_collector import MARKET_SYMBOLS

    missing = sorted(set(SECTOR_MAP.values()) - set(MARKET_SYMBOLS))
    assert missing == [], f"sector ETFs never downloaded: {missing}"


# --- loading and cleaning --------------------------------------------------


def test_loads_a_clean_csv(data_env):
    _price_rows(date.today()).to_csv(data_env / "AAPL.csv", index=False)

    df = load_and_process("AAPL")

    assert len(df) > 0
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "Date"


def test_all_feature_columns_are_produced(data_env):
    """FEATURE_COLUMNS is what trainer.py indexes; a gap is a KeyError at
    training time, long after the data was collected."""
    _price_rows(date.today()).to_csv(data_env / "AAPL.csv", index=False)

    df = load_and_process("AAPL")

    missing = [c for c in config.FEATURE_COLUMNS if c not in df.columns]
    assert missing == [], f"features.py did not produce: {missing}"


def test_stray_header_rows_are_dropped(data_env):
    """Older multi-level yfinance CSVs leak 'Price'/'Ticker' rows into the
    data. Left in place they become NaN feature rows."""
    rows = _price_rows(date.today())
    polluted = pd.concat([
        pd.DataFrame([{
            "Date": "Ticker", "Open": "AAPL", "High": "AAPL",
            "Low": "AAPL", "Close": "AAPL", "Volume": "AAPL",
        }]),
        rows,
    ], ignore_index=True)
    polluted.to_csv(data_env / "AAPL.csv", index=False)

    df = load_and_process("AAPL")

    assert df.index.notna().all()
    assert not (df.index.astype(str) == "Ticker").any()


def test_rows_with_unparseable_prices_are_dropped(data_env):
    """The bad value is injected into the CSV text rather than via
    `rows.loc[...] = "n/a"`: newer pandas raises LossySetitemError on assigning
    a string into a float64 column, and a corrupt cell arrives from a file in
    production anyway.
    """
    path = data_env / "AAPL.csv"
    _price_rows(date.today()).to_csv(path, index=False)

    lines = path.read_text().splitlines()
    header = lines[0].split(",")
    close_idx = header.index("Close")
    fields = lines[6].split(",")
    fields[close_idx] = "n/a"
    lines[6] = ",".join(fields)
    path.write_text("\n".join(lines) + "\n")

    df = load_and_process("AAPL")

    assert df["Close"].notna().all()


def test_missing_ticker_csv_raises(data_env):
    with pytest.raises(FileNotFoundError):
        load_and_process("NOSUCH")


def test_missing_market_csv_raises_with_a_fix_hint(data_env):
    """The message has to say how to fix it — this fires on fresh checkouts."""
    _price_rows(date.today()).to_csv(data_env / "AAPL.csv", index=False)
    (data_env / "market" / "SPY.csv").unlink()
    _load_market_close.cache_clear()

    with pytest.raises(FileNotFoundError, match="market_data_collector.py"):
        load_and_process("AAPL")


# --- Layer 3 staleness tripwire --------------------------------------------


def test_fresh_data_passes_the_tripwire(data_env):
    _price_rows(date.today()).to_csv(data_env / "AAPL.csv", index=False)

    load_and_process("AAPL")   # must not raise


def test_stale_ticker_data_trips_the_wire(data_env):
    """Defence in depth: if the Layer 1 gate is bypassed, this must still stop
    a signal being generated from old prices."""
    _price_rows(date.today() - timedelta(days=30)).to_csv(data_env / "AAPL.csv", index=False)

    with pytest.raises(StaleMarketDataError, match="business days behind today"):
        load_and_process("AAPL")


def test_tolerance_accommodates_a_long_weekend(data_env):
    """3 business-day tolerance covers long weekends and holidays.
    2 bdays back (e.g. Thursday data checked on Monday) must pass."""
    last_bday = (pd.Timestamp.today() - BDay(2)).date()
    _price_rows(last_bday).to_csv(data_env / "AAPL.csv", index=False)

    load_and_process("AAPL")   # must not raise


def test_beyond_bday_tolerance_trips_the_wire(data_env):
    """Data 5 business days old clearly exceeds the 3 bday tolerance."""
    stale_bday = (pd.Timestamp.today() - BDay(5)).date()
    _price_rows(stale_bday).to_csv(data_env / "AAPL.csv", index=False)

    with pytest.raises(StaleMarketDataError):
        load_and_process("AAPL")


def test_stale_market_reference_data_trips_the_wire(data_env):
    """A fresh ticker against a stale SPY still produces wrong features."""
    _price_rows(date.today()).to_csv(data_env / "AAPL.csv", index=False)
    _price_rows(date.today() - timedelta(days=30)).to_csv(
        data_env / "market" / "SPY.csv", index=False
    )
    _load_market_close.cache_clear()

    with pytest.raises(StaleMarketDataError, match="SPY"):
        load_and_process("AAPL")


# --- market close loading --------------------------------------------------


def test_market_close_is_numeric_and_tz_naive(data_env):
    close = _load_market_close("SPY")

    assert pd.api.types.is_numeric_dtype(close)
    assert close.index.tz is None


def test_market_close_is_memoised(data_env):
    """Tickers sharing a sector ETF must not re-read the file each time."""
    first = _load_market_close("XLK")
    second = _load_market_close("XLK")

    assert first is second


def test_ttl_cache_re_reads_after_expiry(data_env, monkeypatch):
    """After MARKET_CACHE_TTL_SECONDS, the cache must re-read the CSV so
    a long-lived process (Streamlit) picks up refreshed data (Finding 3)."""
    first = _load_market_close("SPY")

    # Overwrite the CSV with different data
    today = date.today()
    new_rows = _price_rows(today, rows=120)
    new_rows["Close"] = 999.99
    new_rows.to_csv(data_env / "market" / "SPY.csv", index=False)

    # Within TTL: should still return the cached (old) series
    second = _load_market_close("SPY")
    assert second is first

    # Advance time past the TTL by patching time.monotonic
    import time
    real_monotonic = time.monotonic
    monkeypatch.setattr(
        time, "monotonic",
        lambda: real_monotonic() + features.MARKET_CACHE_TTL_SECONDS + 1,
    )

    # After TTL: should re-read and return the new data
    third = _load_market_close("SPY")
    assert third is not first
    assert third.iloc[-1] == pytest.approx(999.99)


def test_tz_aware_market_csv_is_normalised(data_env):
    """CSVs written with a UTC index must not break the merge."""
    rows = _price_rows(date.today())
    rows["Date"] = pd.to_datetime(rows["Date"]).dt.tz_localize("UTC")
    rows.to_csv(data_env / "market" / "SPY.csv", index=False)
    _load_market_close.cache_clear()

    close = _load_market_close("SPY")

    assert close.index.tz is None


# --- overlap trim (Finding 1) --------------------------------------------


def test_overlap_trim_removes_leading_rows_when_market_starts_later(data_env, capsys):
    """When a market series starts later than the ticker, the ticker's
    early rows are trimmed to the overlap window rather than silently
    dropped by dropna after reindex introduces leading NaN."""
    today = date.today()

    # Ticker has 200 rows of history
    ticker_rows = _price_rows(today, rows=200)
    ticker_rows.to_csv(data_env / "AAPL.csv", index=False)

    # SPY starts much later — only 80 rows (overlaps the last ~80 trading days)
    short_spy = _price_rows(today, rows=80)
    short_spy.to_csv(data_env / "market" / "SPY.csv", index=False)
    _load_market_close.cache_clear()

    df = load_and_process("AAPL")

    # The result should be shorter because early rows were trimmed
    # (not because dropna silently ate them).  The exact count depends
    # on indicator warm-up, but it must be < 200 minus warm-up.
    assert len(df) > 0

    # The trim message should appear in stdout
    captured = capsys.readouterr()
    assert "OVERLAP_TRIM" in captured.out
    assert "AAPL" in captured.out


def test_no_overlap_trim_when_market_starts_earlier(data_env, capsys):
    """When all market series cover the full ticker range, no trim fires."""
    today = date.today()
    _price_rows(today, rows=120).to_csv(data_env / "AAPL.csv", index=False)

    # Market CSVs (from data_env fixture) already have 120 rows — same range
    df = load_and_process("AAPL")

    assert len(df) > 0
    captured = capsys.readouterr()
    assert "OVERLAP_TRIM" not in captured.out


def test_overlap_trim_no_remaining_rows_raises(data_env):
    """If the ticker's data ends before the market series begin, the trim
    leaves zero rows and add_features raises rather than returning empty."""
    today = date.today()

    # Ticker ends 2 years ago — no overlap with market CSVs (which end today)
    old_ticker = _price_rows(today - timedelta(days=730), rows=60)
    old_ticker.to_csv(data_env / "AAPL.csv", index=False)

    with pytest.raises(ValueError, match="no rows remain after overlap trim"):
        load_and_process("AAPL")
