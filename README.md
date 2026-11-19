# Portfolio Intelligence

Portfolio optimisation, risk analytics, volatility forecasting (historical, GARCH, LSTM) and
market-implied volatility analysis, with a Streamlit interface and PDF risk reports.

> Work in progress — see [docs/PLAN.md](docs/PLAN.md) for phase status and
> [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for all models, assumptions and limitations.

## Setup

Requires Python 3.11+ (developed on 3.12).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` is a pinned lock generated from `requirements.in`.

## Tests

```bash
.venv/bin/python -m pytest            # unit tests (synthetic data, offline)
.venv/bin/python -m pytest -m network # live market-data integration test
```

## Data

Free data from Yahoo Finance via `yfinance`; no API keys required. Downloaded series are
cached in `.cache/` (12-hour freshness). Missing data is never filled or fabricated — data
problems are reported in a data-quality report and affected assets are excluded explicitly.

*Not investment advice.*
