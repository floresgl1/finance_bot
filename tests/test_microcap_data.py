"""microcap_data.py on tiny fake Sharadar zips.

Pins: only pre-registered columns survive, fundamentals keep ARY rows only,
types are fixed (so chunks never disagree), chunked reads equal one read,
missing values become null, and a failed conversion leaves no file behind.
"""

import zipfile

import pyarrow.parquet as pq
import pytest

import microcap_data as md


def _write_zip(zip_dir, table, text):
    zip_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_dir / f"{table}.csv.zip", "w") as z:
        z.writestr(f"{table}.csv", text)


FUNDAMENTALS = (
    "ticker,dimension,calendardate,date,reportperiod,ebit,debt,cashneq,revenue,sharesbas,pe,roe\n"
    "AAA,ARY,2019-12-31,2020-03-02,2019-12-31,10,5,1,100,1000,12.5,0.2\n"
    "AAA,MRY,2019-12-31,2020-03-02,2019-12-31,11,5,1,100,1000,12.5,0.2\n"
    "BBB,ARY,2019-12-31,2020-03-20,2019-12-31,,7,2,0,500,,\n"
    "BBB,ART,2019-12-31,2020-03-20,2019-12-31,3,7,2,0,500,,\n"
    "CCC,ARY,2020-12-31,2021-02-15,2020-12-31,-4,0,9,50,200,,\n"
)

STOCKS = (
    "ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated\n"
    "AAA,2020-01-02,1,2,0.5,1.5,1000,1.4,1.5,2026-09-29\n"
    "AAA,2020-01-03,1,2,0.5,1.6,,1.5,1.6,2026-09-29\n"
    "BBB,2020-01-02,5,6,4,5.5,200,5.5,5.5,2026-09-29\n"
)

TICKERS = (
    "table,permaticker,ticker,name,exchange,isdelisted,category,siccode,firstpricedate,lastpricedate\n"
    "fundamentals,101,AAA,A Inc,NYSE,N,Domestic Common Stock,3570,1998-01-02,2026-09-28\n"
    "stocks,101,AAA,A Inc,NYSE,N,Domestic Common Stock,3570,1998-01-02,2026-09-28\n"
    "fundamentals,102,BBB,B Corp,NASDAQ,Y,Domestic Common Stock Primary Class,,2001-05-01,\n"
)

ACTIONS = (
    "date,action,ticker,name,value,contraticker,contraname\n"
    "2015-06-01,bankruptcyliquidation,BBB,B Corp,,,\n"
    "2016-01-04,acquisitionby,CCC,C Co,1500,XYZ,Buyer Inc\n"
)


@pytest.fixture
def dirs(tmp_path):
    zip_dir, out_dir = tmp_path / "zips", tmp_path / "parquet"
    for table, text in [("fundamentals", FUNDAMENTALS), ("stocks", STOCKS),
                        ("tickers", TICKERS), ("actions", ACTIONS)]:
        _write_zip(zip_dir, table, text)
    return zip_dir, out_dir


def _read(out_dir, table):
    return pq.read_table(out_dir / f"{table}.parquet")


def test_fundamentals_keep_ary_rows_and_only_named_columns(dirs):
    zip_dir, out_dir = dirs
    summary = md.convert("fundamentals", zip_dir, out_dir)
    t = _read(out_dir, "fundamentals")
    assert t.column_names == ["ticker", "date", "calendardate", "reportperiod",
                              "ebit", "debt", "cashneq", "revenue", "sharesbas"]
    assert t.column("ticker").to_pylist() == ["AAA", "BBB", "CCC"]
    assert t.column("ebit").to_pylist() == [10.0, None, -4.0]
    assert summary["rows"] == 3 and summary["tickers"] == 3
    assert str(summary["first"]) == "2020-03-02" and str(summary["last"]) == "2021-02-15"


def test_stocks_drop_open_and_unadjusted_close(dirs):
    zip_dir, out_dir = dirs
    md.convert("stocks", zip_dir, out_dir)
    t = _read(out_dir, "stocks")
    assert t.column_names == ["ticker", "date", "high", "low", "close", "volume", "closeadj"]
    assert t.column("volume").to_pylist() == [1000.0, None, 200.0]
    assert str(t.schema.field("date").type) == "date32[day]"


def test_tickers_types_and_missing_values(dirs):
    zip_dir, out_dir = dirs
    md.convert("tickers", zip_dir, out_dir)
    t = _read(out_dir, "tickers")
    assert t.column("permaticker").to_pylist() == [101, 101, 102]
    assert t.column("siccode").to_pylist() == [3570, 3570, None]
    assert t.column("lastpricedate").to_pylist()[2] is None


def test_chunked_read_equals_single_read(dirs, tmp_path):
    zip_dir, out_dir = dirs
    md.convert("fundamentals", zip_dir, out_dir, chunksize=1)
    chunked = _read(out_dir, "fundamentals")
    single_dir = tmp_path / "single"
    md.convert("fundamentals", zip_dir, single_dir, chunksize=1000)
    assert chunked.equals(pq.read_table(single_dir / "fundamentals.parquet"))


def test_a_chunk_with_no_kept_rows_is_skipped(dirs):
    zip_dir, out_dir = dirs
    # With chunksize 1, the MRY and ART rows arrive as chunks that filter to nothing.
    assert md.convert("fundamentals", zip_dir, out_dir, chunksize=1)["rows"] == 3


def test_missing_zip_is_a_clear_failure(tmp_path, capsys):
    rc = md.main(["--tables", "tickers", "--zip-dir", str(tmp_path / "none"),
                  "--out-dir", str(tmp_path / "out")])
    assert rc == 1
    assert "run sharadar_download.py" in capsys.readouterr().out


def test_missing_column_fails_without_leaving_a_file(tmp_path):
    zip_dir, out_dir = tmp_path / "zips", tmp_path / "out"
    _write_zip(zip_dir, "stocks", "ticker,date,high\nAAA,2020-01-02,2\n")
    with pytest.raises(md.ConvertError):
        md.convert("stocks", zip_dir, out_dir)
    assert not (out_dir / "stocks.parquet").exists()
    assert not (out_dir / "stocks.parquet.part").exists()


def test_main_prints_counts_and_date_ranges(dirs, capsys):
    zip_dir, out_dir = dirs
    assert md.main(["--zip-dir", str(zip_dir), "--out-dir", str(out_dir)]) == 0
    out = capsys.readouterr().out
    assert "OK    actions: 2 rows, 2 tickers, dates 2015-06-01 to 2016-01-04" in out
    assert "OK    tickers: 3 rows, 2 tickers" in out
