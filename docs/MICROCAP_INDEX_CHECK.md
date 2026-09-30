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

Run 2026-09-29. IWC adjusted close from Yahoo Finance.

**Correlation of yearly returns: 0.97** (bar: ≥ 0.8) — **PASS: no gross error detected.**

Mean yearly difference, control minus IWC: +4.3% (reported, not judged).

| Holding year | Control | IWC | Difference |
|---|---|---|---|
| 2006-07-03 → 2007-07-02 | +19.1% | +13.5% | +5.5% |
| 2007-07-02 → 2008-07-01 | -24.8% | -27.2% | +2.3% |
| 2008-07-01 → 2009-07-01 | -17.8% | -23.5% | +5.7% |
| 2009-07-01 → 2010-07-01 | +19.5% | +16.4% | +3.1% |
| 2010-07-01 → 2011-07-01 | +33.6% | +34.9% | -1.3% |
| 2011-07-01 → 2012-07-02 | -0.6% | -0.1% | -0.5% |
| 2012-07-02 → 2013-07-01 | +29.8% | +24.9% | +4.9% |
| 2013-07-01 → 2014-07-01 | +26.4% | +24.5% | +1.9% |
| 2014-07-01 → 2015-07-01 | +5.6% | +7.2% | -1.5% |
| 2015-07-01 → 2016-07-01 | -5.4% | -11.3% | +5.9% |
| 2016-07-01 → 2017-07-03 | +31.8% | +27.4% | +4.3% |
| 2017-07-03 → 2018-07-02 | +48.7% | +20.2% | +28.5% |
| 2018-07-02 → 2019-07-01 | -3.7% | -11.2% | +7.5% |
| 2019-07-01 → 2020-07-01 | -0.5% | -6.5% | +6.0% |
| 2020-07-01 → 2021-07-01 | +87.7% | +79.8% | +7.9% |
| 2021-07-01 → 2022-07-01 | -34.8% | -30.9% | -3.8% |
| 2022-07-01 → 2023-07-03 | +20.0% | +6.1% | +13.9% |
| 2023-07-03 → 2024-07-01 | +1.6% | +3.8% | -2.2% |
| 2024-07-01 → 2025-07-01 | +21.1% | +15.1% | +6.0% |
| 2025-07-01 → 2026-07-01 | +50.2% | +57.4% | -7.2% |
