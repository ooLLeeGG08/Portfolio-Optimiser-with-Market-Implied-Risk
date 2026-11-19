# Portfolio Intelligence — Implementation Plan

Status of each phase is tracked at the bottom of this file.

## Design principles

1. **Quant code never imports Streamlit.** Every model is a plain Python function or class
   that can be unit-tested on synthetic data. The UI (`app.py`, `pages/`) only calls the backend.
2. **One source of truth for conventions** (`core/conventions.py`): annualisation, return
   types, loss sign. No module hard-codes `252` or `sqrt(252)`.
3. **Every risk number carries its basis** (`core/types.RiskBasis`):

   | Basis | Meaning | Example |
   |---|---|---|
   | `OBSERVED` | Computed from realised historical data | Historical VaR, 60-day realised vol |
   | `FORECAST` | Output of a statistical / ML model | GARCH or LSTM vol forecast |
   | `MARKET_IMPLIED` | Derived from traded option prices | VIX, ATM implied vol from a listed chain |
   | `IMPLIED_PROXY` | An *estimate* of implied vol where no usable option data exists | hist vol × (VIX / realised S&P vol) |
   | `HYPOTHETICAL` | Result of a user-specified stress scenario | "−20 % market" loss |

   The UI and the PDF render the basis next to every figure. `IMPLIED_PROXY` is never
   displayed under a "market-implied" label.
4. **No fabricated data.** Missing prices are never forward-filled or interpolated; assets
   with insufficient data are excluded with an explicit message. Unavailable option data
   is reported as unavailable.
5. **Chronological discipline.** Every estimator receives only data available at its
   decision date. A single, final out-of-sample test window is reserved and never used
   for any choice of model, hyper-parameter, or default (see METHODOLOGY §8).

## Architecture

```
app.py                      Streamlit entry point + navigation
config.py                   Defaults (tickers, windows, seeds, costs, test-window length)
core/
  conventions.py            Annualisation, simple/log returns, loss sign convention
  types.py                  RiskBasis, IVSource, dataclasses for results
data/
  market_data.py            MarketDataProvider interface + YFinanceProvider
  cache.py                  Parquet cache with fetch metadata (rate-limit protection, provenance)
  data_validation.py        Ticker/price-panel validation -> DataQualityReport
  preprocessing.py          Alignment (no filling), returns, monthly returns, correlation
portfolio/
  portfolio.py              Weights/amounts, value path with rebalancing, performance summary
  optimization.py           Equal-weight, min-variance, max-Sharpe, custom constraints
  efficient_frontier.py
forecasting/
  targets.py                Single definition of the realised-volatility target
  historical_volatility.py
  garch_model.py            GARCH(1,1), walk-forward, simulation-based forecast intervals
  lstm_model.py             LSTM with quantile outputs + split-conformal calibration
  evaluation.py             MAE, RMSE, bias, directional accuracy, interval coverage, DM test
implied_volatility/
  vix.py
  options_data.py           Chain retrieval, liquidity filters, ATM IV, smile, term structure
  proxy.py                  Clearly-labelled IV proxy for assets without usable options
  comparison.py             Horizon-aligned comparison + Volatility Disagreement Indicator
risk/
  covariance.py             Sample / Ledoit-Wolf / forward-looking D·R·D, correlation stress
  risk_metrics.py           Vol, drawdown, Sharpe, Sortino, beta
  var.py, cvar.py           Historical, parametric (normal, Cornish-Fisher), Monte Carlo
  risk_contribution.py      Euler decomposition
  stress_testing.py         Hypothetical scenarios
backtest/
  splits.py                 Development / locked test window definition
  walk_forward.py           Out-of-sample strategy backtest with turnover and costs
reporting/
  report_charts.py          Static (matplotlib) charts for the PDF
  pdf_generator.py          ReportLab research-style report
visualization/
  theme.py, charts.py       Plotly theme + interactive charts
pages/                      One thin Streamlit page per section
tests/                      pytest; synthetic data; network tests marked `network`
docs/                       PLAN.md, METHODOLOGY.md
```

## Phases

| # | Phase | Exit criteria |
|---|---|---|
| 1 | Environment, data layer, cache, validation, portfolio construction | Validation catches bad tickers/gaps/short history; returns and value paths reconcile with hand calculations |
| 2 | Optimisation + efficient frontier | Constraints hold; min-var ≤ frontier; max-Sharpe is frontier-optimal |
| 3 | Risk metrics, VaR/CVaR, risk contribution, D·R·D covariance, stress tests | CVaR ≥ VaR; closed-form checks; Euler contributions sum to σ_p |
| 4 | Historical vol, GARCH, LSTM, forecast intervals, evaluation | No-leakage assertions; GARCH parameter recovery on simulated data; interval coverage reported |
| 5 | VIX, options IV, proxy IV, comparison, Volatility Disagreement Indicator | Actual vs proxy IV labelled; unavailable data flagged |
| 6 | Streamlit dashboard (9 sections) | All pages render on live data; basis labels visible |
| 7 | PDF report (incl. model limitations & data-quality section) | Built from live results with observation dates |
| 8 | Test pass, README, cleanup | Fresh install + smoke run |

## Refinements incorporated (2026-11-19)

1. Actual option-derived IV vs IV proxy distinguished everywhere (`IVSource`, `RiskBasis.IMPLIED_PROXY`).
2. Σ = D·R·D assumptions documented (METHODOLOGY §6) with correlation stress testing.
3. Uncertainty for forecasts: GARCH simulation intervals; LSTM quantile outputs with
   split-conformal calibration; empirical coverage reported out of sample.
4. Locked, chronological out-of-sample test window; hyper-parameters chosen on development data only.
5. Volatility Disagreement Indicator (analytical, not a trading signal).
6. PDF section "Model limitations and data quality".
7. Observed / forecast / market-implied / proxy / hypothetical distinction maintained throughout.

## Status

- [x] Phase 1 — data layer and portfolio construction
- [x] Phase 2 — optimisation and efficient frontier
- [x] Phase 3 — risk metrics, VaR/CVaR, risk contributions, D·R·D, stress tests
- [ ] Phase 4
- [ ] Phase 5
- [ ] Phase 6
- [ ] Phase 7
- [ ] Phase 8
