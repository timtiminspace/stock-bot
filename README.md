# MarketSignalLab

MarketSignalLab is a deterministic, long-only swing-trading research project. It
builds an approximate technology universe, resolves that universe through dated
listing snapshots, creates daily signals, selects competing entries randomly, and
simulates next-open execution. It never sends broker orders.

## Architecture

```text
tech_seed_universe.csv + optional ETF holding CSVs
                         ↓
                  company_evidence.csv
                         ↓
Alpha Vantage LISTING_STATUS → monthly universe snapshots
                         ↓
             yfinance daily OHLCV history
                         ↓
          features → signals → random selection
                         ↓
          trades, equity curve, metrics, diagnostics
```

Provider responsibilities are deliberately separate:

- Alpha Vantage supplies historical US listing rosters only.
- Local evidence determines whether a listed company is technology-eligible.
- yfinance supplies price history and optional cached company descriptions.
- Saved rolling snapshots, not current membership, control backtest eligibility.

## Installation

The project requires Python 3.11 or newer and uses
[uv](https://docs.astral.sh/uv/) for dependency management. The checked-in
`.python-version` selects Python 3.13 when uv creates the environment:

```bash
uv sync
```

Copy the environment template if you need to build new listing snapshots:

```bash
cp .env.example marketsignallab/.env
```

Then add your Alpha Vantage key to `marketsignallab/.env`, or export it in the
shell:

```bash
export ALPHAVANTAGE_API_KEY="your-key"
```

The key is not needed when running a backtest from existing snapshots and cached
prices.

## Technology evidence

The versioned manual seed list is
`marketsignallab/config/tech_seed_universe.csv`. Additional reproducible holdings
exports can be placed under `marketsignallab/data/seed_sources/`. Supported ticker
column names are `ticker`, `symbol`, `holding`, and `name`.

Build the deduplicated evidence file locally:

```bash
cd marketsignallab
uv run python src/tech_universe_builder.py --valid-from 2023-01-03
```

Optional yfinance metadata enrichment is cached and explicitly capped:

```bash
uv run python src/tech_universe_builder.py \
  --valid-from 2023-01-03 \
  --enrich-yfinance \
  --enrichment-limit 100
```

The output is `config/company_evidence.csv`. The evidence source is marked
`approximate_seed_universe`: present-day ETF membership and descriptions are useful
for broad discovery, but they are not true historical classification evidence.
Every manual seed retains its `compounder` or `recovery_candidate` position label.

## Rolling universe snapshots

`rolling_universe.py` uses Alpha Vantage `LISTING_STATUS` to identify securities
that were active on each snapshot date. It filters non-common securities,
benchmarks, rejected tickers, and unsupported exchanges before joining technology
evidence. `config/company_evidence.csv` is used automatically.

Build one snapshot:

```bash
cd marketsignallab
uv run --env-file .env python src/rolling_universe.py --snapshot-date 2023-01-31
```

Build or resume a monthly series:

```bash
uv run --env-file .env python src/rolling_universe.py \
  --start 2023-01-01 \
  --end 2025-01-31
```

Raw provider responses are cached under `data/raw/alpha_vantage/`. Existing raw
responses and snapshots are reused. Use `--overwrite` to rebuild normalized
snapshots from cached raw data. Use `--refresh-listings --overwrite` only when you
intend to make new provider requests.

A snapshot must contain explicit `eligible` or `primary_category` data. The
backtester refuses unclassified full-market rosters rather than accidentally
trading every listed security.

## Run the strategy

From the repository root:

```bash
uv run python main.py
```

Or from `marketsignallab/`:

```bash
uv run python src/main.py
```

The default run:

- starts on January 3, 2023;
- reads `data/universe_snapshots/`;
- uses deterministic random entry selection with seed 42;
- derives the exclusive end date from the latest snapshot;
- loads cached `data/prices.csv`, or downloads prices when the cache is absent;
- writes all generated results under `marketsignallab/data/`.

Refresh price data through an explicitly covered end date:

```bash
uv run python main.py --refresh --end 2025-03-01 --random-seed 42
```

`--end` is exclusive. A supplied end date beyond rolling-universe coverage is
rejected before a price download begins.

## Strategy rules

### Entry

A ticker can enter the daily candidate pool when it:

- is eligible in the latest non-future rolling snapshot;
- is not rejected, held, pending, or cooling down;
- has average 30-day volume of at least 1,000,000 shares;
- closes above its 50-day moving average;
- has positive 30- and 90-day momentum;
- has 30-day volatility no greater than 8%.

If more candidates qualify than there are open slots, the entire valid pool is
shuffled using `random_seed + YYYYMMDD`. This makes a seed reproducible while
allowing different seeds to choose different stocks. The portfolio holds at most
five active positions and targets 20% of portfolio value per active position.

### Execution

Signals use information available at the daily close. Orders execute at the next
available open. The trade log records the signal and execution dates, both prices,
and the overnight gap return. The simulation uses fractional shares, no leverage,
and a 5% minimum cash buffer.

### Exits and recovery

- A profitable high-volume spike fully exits a `recovery_candidate`.
- The same spike sells 90% of a `compounder` and keeps a 10% runner.
- Weekly underperformance exits a normal position when it falls at least 5% and
  trails QQQ by at least 3% over five trading days.
- At a 10% loss, normal exits are suspended and the position enters recovery mode.
- Recovery exits the full position after a close at least 5% above entry.
- If the position reaches a 30% loss first, 50% is sold and the remainder becomes
  `HELD_SIDELINED` through the end of the backtest.

Sidelined positions remain marked to market and continue consuming capital, but do
not count toward the five active-position slots. Their value and count are reported
separately in the equity curve and metrics.

All exit signals still execute at the following open, so a gap can produce a fill
outside the signal threshold.

## Monte Carlo random-selection runs

Run the same prepared signal dataset across many deterministic seeds:

```bash
cd marketsignallab
uv run python src/run_random_backtests.py \
  --start 2023-01-01 \
  --end 2025-03-01 \
  --runs 200 \
  --max-positions 5
```

The runner reports the return distribution, drawdowns, QQQ comparisons, recovery
episodes, catastrophic half-sales, seed rankings, and best/worst realized stocks.

## Important outputs

- `prices.csv`, `features.csv`, `signals.csv`
- `trades.csv`, `equity_curve.csv`, `entry_candidate_log.csv`
- `loss_recovery_log.csv`, `reentry_watchlist.csv`
- `metrics_summary.csv` and annual/ticker/drawdown metric tables
- `random_backtest_runs.csv`, `random_backtest_summary.csv`
- `random_backtest_seed_rankings.csv`
- `random_backtest_seed_stock_performance.csv`
- `random_backtest_seed_trades.csv`
- `random_backtest_loss_recovery_log.csv`
- `random_backtest_catastrophic_stop_summary.csv`

The entire `marketsignallab/data/` directory is ignored by Git. Configuration,
source code, tests, and experiment notes remain versioned.

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

## Known limitations

- Technology evidence is approximate and can introduce survivorship or lookahead
  bias when it comes from current ETF holdings or current descriptions.
- yfinance is convenient research data, not an exchange-grade market-data feed.
- Commissions, slippage, taxes, spreads, liquidity impact, dividends, and corporate
  actions are not fully modeled.
- A held ticker with missing future prices remains valued at its last available
  close. Delisting and bankruptcy settlement logic is still required before the
  model can be treated as production-quality.
- Random-seed dispersion is part of the result. A single favorable seed is not
  evidence that the strategy is robust.

This repository is research software, not investment advice.
