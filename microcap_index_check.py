"""
Index check for the micro-cap backtest; see docs/MICROCAP_INDEX_CHECK.md.

Compares the backtest control's yearly returns (primary run, from
data/microcap_backtest.json) with the iShares Micro-Cap ETF (IWC) over the
same July-to-July holding years, 2006-2025. The bar was committed before this
script fetched anything: correlation >= 0.8.

    python microcap_index_check.py      # appends the result to the doc
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

RESULT_JSON = Path("data/microcap_backtest.json")
DOC = Path("docs/MICROCAP_INDEX_CHECK.md")
TICKER = "IWC"
FIRST_YEAR = 2006
MIN_CORRELATION = 0.8
LAST_END = date(2026, 7, 1)


def holding_years(years):
    """(start, end, control return) per year from FIRST_YEAR, ends = next rebalance."""
    starts = [date.fromisoformat(y["rebalance"]) for y in years]
    ends = starts[1:] + [LAST_END]
    return [(s, e, y["control"]) for s, e, y in zip(starts, ends, years) if s.year >= FIRST_YEAR]


def period_return(closes, start, end):
    """Adjusted-close return from the first trading day on or after start to the
    first on or after end."""
    idx = closes.index
    a = closes[idx >= pd.Timestamp(start)].iloc[0]
    b = closes[idx >= pd.Timestamp(end)].iloc[0]
    return float(b / a - 1)


def compare(periods, closes):
    rows = [(s, e, c, period_return(closes, s, e)) for s, e, c in periods]
    control = np.array([r[2] for r in rows])
    index = np.array([r[3] for r in rows])
    corr = float(np.corrcoef(control, index)[0, 1])
    return rows, corr, float((control - index).mean())


def format_result(rows, corr, mean_diff):
    ok = corr >= MIN_CORRELATION
    lines = [
        f"Run {date.today()}. {TICKER} adjusted close from Yahoo Finance.",
        "",
        f"**Correlation of yearly returns: {corr:.2f}** (bar: ≥ {MIN_CORRELATION}) — "
        + ("**PASS: no gross error detected.**" if ok else "**FAIL: see above for what follows.**"),
        "",
        f"Mean yearly difference, control minus {TICKER}: {mean_diff:+.1%} (reported, not judged).",
        "",
        f"| Holding year | Control | {TICKER} | Difference |",
        "|---|---|---|---|",
    ]
    for s, e, c, i in rows:
        lines.append(f"| {s} → {e} | {c:+.1%} | {i:+.1%} | {c - i:+.1%} |")
    return "\n".join(lines) + "\n", ok


def main():
    import yfinance as yf

    runs = json.loads(RESULT_JSON.read_text(encoding="utf-8"))["runs"]
    years = runs[str(-0.3)]["years"]
    data = yf.download(TICKER, start="2006-06-01", end="2026-09-30", auto_adjust=True,
                       progress=False)
    closes = data["Close"].squeeze().dropna()
    if closes.empty:
        print(f"No {TICKER} prices returned.")
        return 1
    rows, corr, mean_diff = compare(holding_years(years), closes)
    text, ok = format_result(rows, corr, mean_diff)
    doc = DOC.read_text(encoding="utf-8")
    DOC.write_text(doc.replace("*To be appended by `microcap_index_check.py`.*\n", text),
                   encoding="utf-8")
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
