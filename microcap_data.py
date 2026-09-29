"""
Convert the Sharadar bulk zips into Parquet for the micro-cap value test.

Reads data/sharadar/{table}.csv.zip (see sharadar_download.py) and writes
data/sharadar/parquet/{table}.parquet with ONLY the columns
docs/MICROCAP_VALUE_PREREG.md names. The other ~100 fundamentals columns
(P/E, ROE, ...) are dropped on purpose: a column that is not there cannot
tempt anyone into adding a signal after seeing results.

  fundamentals  ARY rows only (as first reported)
  stocks        split-adjusted high/low/close, volume, closeadj, closeunadj
  tickers       identity, name, category, SIC, related tickers, delisting flag
  actions       all columns (delisting reasons, SIC and exchange changes)

Files are read in chunks, so memory stays flat on the 3 GB price file.
This step computes no returns, rankings or checks; it prints row counts and
date ranges only.

    python microcap_data.py
    python microcap_data.py --tables tickers actions

Exit 0 if every requested table was written.
"""

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ZIP_DIR = Path("data/sharadar")
OUT_DIR = Path("data/sharadar/parquet")
CHUNKSIZE = 1_000_000

STR, DATE, FLOAT, INT = pa.string(), pa.date32(), pa.float64(), pa.int64()

# table -> (kept columns with types, rows to keep)
SPECS = {
    "fundamentals": {
        "columns": {"ticker": STR, "date": DATE, "calendardate": DATE, "reportperiod": DATE,
                    "ebit": FLOAT, "debt": FLOAT, "cashneq": FLOAT, "revenue": FLOAT,
                    "sharesbas": FLOAT},
        "where": {"dimension": "ARY"},
    },
    "stocks": {
        # closeunadj puts reported share counts on the split-adjusted basis
        # (market-cap rule in the pre-registration).
        "columns": {"ticker": STR, "date": DATE, "high": FLOAT, "low": FLOAT,
                    "close": FLOAT, "volume": FLOAT, "closeadj": FLOAT,
                    "closeunadj": FLOAT},
    },
    "tickers": {
        # name and relatedtickers pair a Secondary Class security with its issuer.
        "columns": {"permaticker": INT, "ticker": STR, "table": STR, "name": STR,
                    "category": STR, "siccode": INT, "isdelisted": STR,
                    "relatedtickers": STR, "firstpricedate": DATE, "lastpricedate": DATE},
    },
    "actions": {
        "columns": {"date": DATE, "action": STR, "ticker": STR, "name": STR,
                    "value": FLOAT, "contraticker": STR, "contraname": STR},
    },
}


class ConvertError(Exception):
    pass


def schema_for(table):
    return pa.schema([(name, typ) for name, typ in SPECS[table]["columns"].items()])


def _typed(chunk, schema):
    """Coerce a raw string chunk to the schema; bad values become null."""
    out = {}
    for field in schema:
        col = chunk[field.name]
        if field.type == DATE:
            out[field.name] = pd.to_datetime(col, format="%Y-%m-%d", errors="coerce").dt.date
        elif field.type in (FLOAT, INT):
            num = pd.to_numeric(col, errors="coerce")
            out[field.name] = num.astype("Int64") if field.type == INT else num
        else:
            out[field.name] = col.where(col.notna(), None)
    return pa.Table.from_pandas(pd.DataFrame(out), schema=schema, preserve_index=False)


def convert(table, zip_dir=ZIP_DIR, out_dir=OUT_DIR, chunksize=CHUNKSIZE):
    """Write one table to Parquet via a .part file; return a summary dict."""
    spec = SPECS[table]
    schema = schema_for(table)
    where = spec.get("where", {})
    usecols = list(spec["columns"]) + [c for c in where if c not in spec["columns"]]

    src = Path(zip_dir) / f"{table}.csv.zip"
    if not src.exists():
        raise ConvertError(f"{table}: {src} not found; run sharadar_download.py")
    dest = Path(out_dir) / f"{table}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    rows = 0
    tickers = set()
    dates = []
    try:
        with zipfile.ZipFile(src) as z, z.open(z.namelist()[0]) as f, \
                pq.ParquetWriter(part, schema) as writer:
            reader = pd.read_csv(f, usecols=usecols, dtype=str, keep_default_na=False,
                                 na_values=[""], chunksize=chunksize)
            for chunk in reader:
                for col, value in where.items():
                    chunk = chunk[chunk[col] == value]
                if chunk.empty:
                    continue
                typed = _typed(chunk, schema)
                writer.write_table(typed)
                rows += typed.num_rows
                tickers.update(chunk["ticker"].dropna().unique())
                if "date" in chunk:
                    d = typed.column("date").drop_null()
                    if len(d):
                        dates += [pc.min(d).as_py(), pc.max(d).as_py()]
    except ValueError as exc:
        part.unlink(missing_ok=True)
        raise ConvertError(f"{table}: {exc}") from None
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(dest)
    return {"table": table, "rows": rows, "tickers": len(tickers),
            "first": min(dates) if dates else None, "last": max(dates) if dates else None}


def format_summary(s):
    span = f", dates {s['first']} to {s['last']}" if s["first"] else ""
    return f"  OK    {s['table']}: {s['rows']:,} rows, {s['tickers']:,} tickers{span}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tables", nargs="+", choices=list(SPECS), default=list(SPECS))
    parser.add_argument("--zip-dir", type=Path, default=ZIP_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    args = parser.parse_args(argv)

    failed = 0
    for table in args.tables:
        print(f"  READ  {table} ...", flush=True)
        try:
            print(format_summary(convert(table, args.zip_dir, args.out_dir, args.chunksize)))
        except ConvertError as exc:
            failed += 1
            print(f"  FAIL  {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
