# Micro-cap backtest — index check (2026-09-30)

**Written and committed before any index data was fetched.** The backtest
result (`MICROCAP_BACKTEST_RESULT.md`, FAIL) already exists; this check asks
only whether the backtest's machinery has a gross error, and is fixed in
advance so it cannot be bent toward the answer.

## Why

Looking for bugs only after a disappointing result pushes every fix one way.
A single check, chosen before looking, that could catch a gross error — mixed
units, a flipped sign, broken return arithmetic — whichever way it would
move the result, avoids that.

## The check

- **Benchmark:** iShares Micro-Cap ETF (IWC), adjusted close from Yahoo
  Finance. It tracks a published micro-cap index and has traded since
  August 2005.
- **Periods:** the backtest's own holding years, first trading day on or
  after July 1 to the next, 2006–2025: 20 years.
- **Compared:** the backtest's **control** (equal-weight eligible universe,
  primary run) against IWC, year by year.
- **Bar:** correlation of the 20 yearly returns **≥ 0.8**. Both are broad
  baskets of small US stocks, so they should move together; a gross error
  breaks that link.
- **Reported, not judged:** the mean yearly difference. A gap is expected by
  design: the control is equal-weight, $50–300M, profitable-or-cash-rich and
  low-debt; IWC is cap-weighted and holds loss-makers.

## If it fails

The failure is reported here as found. Any fix to the backtest and any
rerun is recorded as a **second draw**, and the first result stays on record.

## Result

*To be appended by `microcap_index_check.py`.*
