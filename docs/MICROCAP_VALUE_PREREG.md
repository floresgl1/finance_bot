# Micro-cap value — pre-registration draft (2026-09-27)

**Status: design only. Nothing has been run, and no data has been acquired.**
Every choice below was fixed on paper before any backtest exists. Changing one
after seeing results turns this into a second draw; see "Confirming a future
pass" in `EDGE_INVESTIGATION_2026-09-08.md`.

*Amended 2026-09-27, still before any data exists:* the universe now names
which securities qualify (CRSP share and exchange codes), and "Identifiers and
joins" fixes how prices and fundamentals are matched.

*Amended 2026-09-29, still before any data exists:* every open item is now
decided, and "Implementing on Sharadar" states how each rule is met with
Sharadar data. At the time of writing only Sharadar's free sample had been
accessed, and only to plan this amendment; no returns, rankings or results
had been computed from any source.

*Amended again 2026-09-29, still before any data exists:* Sharadar is used
through its own API (`api.sharadar.com`), not Nasdaq Data Link, and its field
names differ from those first written here. The free tier covers only the 30
Dow stocks, so the structural checks cannot run on it; they now run on the
downloaded data, before any return or ranking is computed. A fifth check
covers the ticker join. The only data seen was public ticker metadata and
Sharadar's AAPL-only demo key.

## Why this, and not more of the current bot

Nine pre-registered probes found no edge in daily technical signals on twelve
mega-caps (`EDGE_INVESTIGATION_2026-09-08.md`). That is the most crowded corner
of the market: every large fund trades the same information. The rule this
design follows instead is **go where size is a disadvantage.** Micro-caps are
too small for institutional money to trade, so an account of ~$10k faces less
competition there.

## The strategy

| # | Component | Decision |
|---|---|---|
| 1 | Universe | Ordinary shares of US companies — CRSP **share code 10 or 11**, **exchange code 1, 2 or 3** (NYSE, AMEX, Nasdaq) — with $50M–$300M market cap, built **point-in-time** (including companies later delisted); average daily dollar volume ≥ $100k; **excluding financials** (banks, insurers, REITs); must have EBIT and EV available as of the rebalance date |
| 2 | Signal | Rank by **EBIT / enterprise value**, highest first, using only filings public before the rebalance date |
| 2a | Filter | **Net debt / EBIT ≤ 3×** |
| 3 | Portfolio | Top **30** names, equal weight |
| 4 | Rebalance | **Yearly**; cost = half the spread per side, per stock |
| 5 | Control | Equal-weight basket of the **whole eligible universe** — same filters, same dates, same cost model |

### Why each choice

1. **Universe.** Point-in-time because a universe built from today's listings
   has already removed every company that went bankrupt, and the cheapest
   stocks — exactly the ones this strategy buys — delist most often. A backtest
   on survivors can show 20%+ a year with no edge at all. The liquidity floor
   is about spreads, not capacity: the thinnest names trade at 5%+ spreads,
   which would consume a year's edge in one round trip. Financials are
   excluded *from the universe*, not just filtered by the debt rule, because
   debt is their raw material and both metrics are meaningless for them; and
   because if the strategy dropped them while the control kept them, the two
   would differ in sector mix as well as signal. Share codes 10/11 keep out
   ETFs, closed-end funds and ADRs, which a market-cap filter lets through:
   ETFs have no EBIT, and ADRs report under foreign rules that North American
   fundamentals data mostly does not cover. Exchange codes 1–3 keep out stray
   over-the-counter listings. A security without EBIT and EV on the rebalance
   date is excluded from the universe, so it is missing from strategy and
   control alike — never sorted to one end of the ranking by a missing value,
   and never held by the control alone.
2. **Signal.** Cheapness has two independent explanations — compensation for
   the risk of holding troubled companies, and investors over-reacting to bad
   news in names no analyst covers — so the effect has a reason to exist.
   EBIT/EV instead of P/E because many micro-caps lose money, which makes P/E
   meaningless at both ends; EBIT/EV puts every company on one scale and
   counts debt in the price.
2a. **Debt filter.** A $100M company cannot borrow its way through a bad year;
   default leads to delisting. A positive-EBIT filter was considered and
   rejected: the top of an EBIT/EV ranking already has positive EBIT, so it
   would filter out nothing the strategy holds.
3. **30 names.** Roughly the top few percent of the eligible universe, so still
   selective, while one blowup cannot decide the result. About $330 a name at
   $10k: far too small to move a price.
4. **Yearly.** Annual cost ≈ round-trip spread × round trips per year. At a
   1.5% round trip, weekly rebalancing costs ~78% a year and monthly ~18%; only
   a slow strategy leaves room for an edge.
5. **Control.** Against SPY, the strategy could win by being in micro-caps
   during a good decade for micro-caps. The control differs from the strategy
   in exactly one respect — which names are picked — so any gap is the signal.

## Identifiers and joins

Tickers are never used as identifiers: they are reused across unrelated
companies and change when a company renames. Joining on them attaches one
company's data to another's years, or drops a delisted company entirely.

- **Securities** are identified by CRSP **PERMNO**; **companies** by Compustat
  **GVKEY**.
- Fundamentals attach to prices only through the **CRSP–Compustat (CCM) link
  table**, and only where all three hold:
  - `linktype` is `LU` or `LC` (the reliable link types);
  - `linkprim` is `P` or `C` (the primary security, so a company with two
    share classes enters once, not twice);
  - the rebalance date lies within `linkdt`–`linkenddt` (a link is valid only
    for its dates; mergers and share-class changes move it).
- A fundamentals row is usable on a rebalance date only if it was public by
  then (see "Filing lag" under the decisions below).

## Pass criteria

All three must hold, net of costs:

1. **Beats the control in ≥ 80% of years** (8 of 10 with ten years of data).
2. **Quintile staircase, no swaps.** Split the eligible universe into five
   quintiles by EBIT/EV each year. Mean yearly return must be strictly ordered:
   cheapest highest, most expensive lowest. A useless signal produces this
   ordering by chance about 1 time in 120.
3. **Mean excess return over the control ≥ 3% a year.**

Why the bar is stricter than the earlier probes': yearly rebalancing gives one
observation per year, not hundreds of days. With no edge, each year is a coin
flip:

| Bar | Chance a useless signal passes |
|---|---|
| wins ≥ 7 of 10 years | 17% |
| wins ≥ 8 of 10 | 5.5% |
| wins ≥ 9 of 10 | 1.1% |

The staircase adds evidence the win count cannot: it tests whether cheapness
itself orders the returns, not just whether one portfolio got lucky.

A pass is a lead, not a result. It goes through the confirmation protocol on
live data before any real money.

## Decisions on the former open items (2026-09-29)

These rules apply whichever data source is used.

- **Data source.** Sharadar (sharadar.com's own API): active and delisted US
  companies, point-in-time fundamentals with filing dates, from the late 1990s.
  WRDS (CRSP/Compustat) access is being pursued; if granted, the test is also
  run on CRSP/Compustat under the rules above, and a pass must hold on both.
  Alternatives rejected: SEC EDGAR XBRL has no prices, covers small companies
  only from 2011, and would need a hand-built EBIT from raw tags — errors there
  would reorder the ranking and look like signal. Norgate has only current
  fundamentals, not what was known on past dates.
- **Rebalance date: July 1** each year (the next trading day if closed). By
  then even late December-year-end filers are past their deadline, and it is
  the standard academic timing, so results stay comparable to published ones.
- **Filing lag.** A fundamentals row is usable on rebalance date D only if its
  filing date is strictly before D; filings often arrive after the close.
- **Staleness.** A company whose latest usable filing covers a fiscal year that
  ended more than **18 months** before D is excluded from the universe, for
  strategy and control alike. Late filing is a warning sign; without this rule
  the ranking would value a troubled company on pre-trouble EBIT against a
  post-trouble price, making it look cheap just before it fails.
- **Enterprise value** is computed on D from that day's market cap plus the
  latest usable filing's debt minus cash. A vendor's precomputed EV is not
  used: it is dated at the filing, so companies filing in different months
  would be ranked on prices from different days.
- **Market cap** is the whole company's, summed across all share classes, for
  both EV and the $50M–$300M filter. Using one class's shares would understate
  EV and make dual-class companies look cheaper than they are.
- **Delisting return**, applied on the delisting day, by reason:

  | Reason | Return beyond the last close |
  |---|---|
  | Acquired or merged away | 0% (a buyout's last trade is near the deal price) |
  | Bankruptcy, regulatory delisting, voluntary, or unknown | **−30%** (Shumway) |

  Unknown reasons take the loss, so unclear data makes the backtest look worse,
  not better.
- **Cash from a delisting** is reinvested the same day into the remaining
  holdings in proportion to their current weights, paying half the spread on
  those buys. The control follows the same rule. Holding the cash instead
  would leave the strategy — whose cheap, troubled stocks delist more often —
  holding more cash than the control, and the gap would partly measure cash
  held rather than the signal.
- **Spread**, when the source has no bid-ask data: the Abdi–Ranaldo (2017)
  estimator on split-adjusted daily high, low and close.
  - Two-day estimates, negatives set to 0, averaged over the **252 trading
    days before** the trade date. Only data known on the trade date is used,
    and the trailing year exists even for a stock that delists soon after.
  - Floor of **0.5%**: estimators undershoot for quiet stocks, and too little
    cost flatters the strategy.
  - Fewer than 126 valid days: **2 × the universe median** on that date.
    Unknown liquidity is treated as poor liquidity.
  - The cost of each trade is half the estimated spread, as in the strategy
    table.

## Implementing on Sharadar

CRSP codes named above have Sharadar equivalents as follows. Table names are
those of `api.sharadar.com/v1.0/data/`: `tickers`, `fundamentals`, `stocks`,
`actions`.

| Rule | Sharadar implementation |
|---|---|
| Identity (PERMNO, GVKEY, CCM link) | `permaticker` is the identity. Only the `tickers` table carries it; `fundamentals` and `stocks` carry only `ticker`, so each row is joined to its `permaticker` through `tickers`. This is safe only because Sharadar renames an old ticker when it is reused (e.g. `AAC2`), keeping each ticker unique in its database — which check 5 verifies. Tickers from any other source are never used. |
| Share code 10/11 | `category` is `Domestic Common Stock` or `Domestic Common Stock Primary Class`. `Domestic Common Stock Secondary Class` is excluded, so a dual-class company enters once — the job CRSP's `linkprim` does. ADR and Canadian categories are excluded. |
| Exchange code 1–3 | **Not implemented from Sharadar's `exchange` field.** It is a snapshot of the last known exchange: a company demoted to OTC before failing would be dropped from every earlier year too, deleting future losers from past rankings. Eligibility rests instead on the point-in-time size and liquidity filters on each rebalance date. |
| Financials | `siccode` 6000–6999 excluded. |
| Blank-check shells | Company-years whose latest usable filing shows **zero revenue** are excluded. A SPAC that later merges takes the operating company's SIC, so the snapshot `siccode` misses its shell years; shells trade flat near $10 at $100M–$300M and would dilute the control. This also drops pre-revenue companies, from strategy and control alike. |
| Fundamentals as known at the time | `fundamentals` rows with dimension **`ARY`** (as first reported) only, never `MRY`: a restatement published later must not appear in an earlier ranking. The filing date is the field `date` (Sharadar's "Date Key": for `ARY`, the date the filing reached the SEC). EBIT, debt and cash are the raw fields `ebit`, `debt`, `cashneq`; the vendor's `ev`, `evebit` and `marketcap` are not used (see "Enterprise value"). |
| Total return | Daily change in `stocks.closeadj`, which is adjusted for splits and dividends. |
| Delisting reason | The `actions` event that ends the listing, mapped to the table above. |
| Date coverage | Date-filtered queries default to the last year only, and the `from`/`to` filter did not behave as documented in a test with the demo key. The download therefore states its date range explicitly, and the loader verifies that every year of the test has rows before anything else runs. |

## Structural checks — before any result

The free tier covers only the 30 Dow stocks, which contain no micro-caps,
SPACs or failures, so these checks run on the **downloaded** data. They
examine structure only: fields, identifiers, categories, event names. They run
before any return, ranking or result is computed. If one fails, the affected
rule is amended here, dated, **before** any result exists.

The download must also prove it is the paid data: a wrong API key returns
HTTP 200 with limited data, not an error. The loader stops if one page of
`ARY` fundamentals holds fewer than 100 distinct tickers.

1. **Dual-class share counts.** For a known dual-class company, does the share
   count in the fundamentals cover all classes? If not, market cap must be
   summed across the classes' price rows.
2. **Demoted stocks.** Does `stocks` carry prices after a stock moves to OTC,
   and do any such names pass the $100k ADV floor? If so, the dropped exchange
   filter needs a point-in-time replacement.
3. **SPAC shell years.** What `siccode` and revenue does a merged SPAC show
   for its shell years? If Sharadar flags shells directly, that flag replaces
   the zero-revenue rule.
4. **Delisting event names.** Which `actions` values mark an acquisition and
   which a failure? The mapping to 0% / −30% is fixed from these names.
5. **Ticker join.** Every ticker in the downloaded `fundamentals` and `stocks`
   rows maps to exactly one `permaticker` in `tickers`. A ticker with none is
   reported; a ticker with two stops the run.

Seen in the public `tickers` table on 2026-09-29 (metadata only, visible to
any key): 17,854 companies with fundamentals, 12,325 of them delisted; no
ticker mapped to two permatickers; no `Secondary Class` category among them;
many SPACs listed as `Primary Class` (their Class A/Class B structure), which
the SIC and zero-revenue rules remove. These observations do not replace the
checks, which run on the downloaded rows.
