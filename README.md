# quant-lab

A sandbox for **quantitative trading experiments**. Currently running **Prophet-based price forecasting** + **Markowitz portfolio optimisation**. Will grow as I learn more techniques.

> ⚠️ **PROJECT STATUS: NOT FINISHED.** This project is currently incomplete and under active development. It is **not recommended to run this system** as it has not been properly built, tested, or validated. Many components are placeholders, the Markowitz implementation contains a known bug, and the daily workflow has been disabled because it was failing. Use only for learning purposes.

> ⚠️ **NOT FINANCIAL ADVICE.** This is a learning project. Don't trade real money based on its output.

## Current Pipeline

```
yfinance (price data)
        ↓
   Prophet (forecast next-day prices)
        ↓
   Markowitz (optimal portfolio weights)
        ↓
   Supabase (store results)
        ↓
   Streamlit (visualize)
```

## What's Here

| Component | What it does |
|---|---|
| `src/extractor.py` | Pulls historical prices from `yfinance` |
| `src/processor.py` | Cleans data, computes returns |
| `src/model.py` | Fits Prophet time series model |
| `src/optimiser.py` | Mean-variance (Markowitz) portfolio weights |
| `src/database.py` | Saves results to Supabase |
| `src/main.py` | Runs the pipeline end-to-end |
| `src/streamlit_app.py` | Web dashboard for visualization |

## Quick Start

### 1. Install dependencies

```bash
poetry install
```

### 2. Set environment variables

Create a `.env` file (not committed):

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
```

If you skip this step, the optimisation runs but doesn't save results.

### 3. Run the pipeline

```bash
# Run optimisation for one day
make run

# Open the dashboard
make dashboard
```

## Roadmap

Things I want to add as I learn:

- [ ] **Pairs trading** — statistical arbitrage between correlated stocks
- [ ] **Momentum strategies** — trend-following signals
- [ ] **Mean reversion** — buy dips, sell rips
- [ ] **Factor models** — value, momentum, size, quality factors
- [ ] **Volatility modeling** — GARCH for risk estimation
- [ ] **Sentiment analysis** — news/Twitter NLP for signal generation
- [ ] **Reinforcement learning** — for dynamic allocation
- [ ] **Backtesting framework** — proper out-of-sample testing
- [ ] **Live broker integration** — Alpaca, Interactive Brokers (paper trading first)

## Configuration

### Tickers
Edit `src/settings.py` → `PORTFOLIO_TICKERS`. Currently:
AMD, MSFT, AAPL, TSLA, AMZN, NVDA, META, GOOG, TSM, JPM, NFLX, PLTR

### Risk parameters
- `MINIMUM_ALLOCATION = 0.05` (5% minimum per asset)
- `MAXIMUM_ALLOCATION = 1.0`
- `RISK_AVERSION = 5`

## Testing

```bash
make test          # pytest
make lint          # ruff + black
make type-check    # mypy
make check         # format + lint + type-check + test
```

## Known Issues

- The previous "daily optimisation" GitHub Action was disabled because it was failing daily (missing Supabase secrets). Manual runs via `workflow_dispatch` still work.
- The Markowitz implementation uses historical actual returns instead of Prophet's predictions in some code paths. See `src/optimiser.py:calculate_mean_variance` — this is a known bug, will fix when adding the backtesting framework.

## Project Structure

```
quant-lab/
├── pyproject.toml       Poetry config + dependencies
├── Makefile             Common commands (run, test, lint)
├── README.md            This file
├── src/
│   ├── extractor.py     Data fetching
│   ├── processor.py     Data cleaning + features
│   ├── model.py         Prophet wrapper
│   ├── optimiser.py     Markowitz portfolio math
│   ├── database.py      Supabase storage
│   ├── main.py          CLI entry point
│   ├── streamlit_app.py Web dashboard
│   └── settings.py      Constants + config
├── tests/
└── scripts/             Helper scripts (deploy.sh etc.)
```

## Why "quant-lab"?

Because:
- **"quant"** = quantitative trading (the field)
- **"lab"** = a place to experiment, not a polished product
- Honest about scope — this is learning code, not production trading

## License

MIT (or whatever — not yet chosen)