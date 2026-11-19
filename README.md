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

## Run the app

```bash
.venv/bin/streamlit run app.py
```

Then open http://localhost:8501. Volatility forecasts (GARCH + one LSTM per asset) are computed on
demand from the Volatility Forecasting page and take about 2 minutes for 7 assets; option chains
take ~25 seconds on first load. Both are cached for the rest of the day.
