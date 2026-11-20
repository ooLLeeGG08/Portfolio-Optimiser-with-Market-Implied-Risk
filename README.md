# Portfolio Intelligence

**Portfolio optimisation, risk analytics, volatility forecasting and market-implied volatility — in
one Streamlit app with a generated PDF risk report.**

Portfolio Intelligence builds a portfolio from free market data, optimises it, measures its risk
(VaR, CVaR, drawdowns, Euler risk contributions), forecasts volatility with historical, GARCH(1,1)
and LSTM models, compares those forecasts with option-implied volatility, stress-tests the
portfolio, evaluates everything out of sample, and writes a research-style PDF report.

It is built to be **financially defensible rather than just visual**:

- Every number is labelled by basis — **observed**, **forecast**, **market-implied**,
  **implied proxy (estimate)** or **hypothetical** — in the app and in the PDF.
- Missing data is never filled or invented; assets with data problems are excluded with a message.
- Forecasts use only information available at the time; a final ~2-year window is locked away and
  never used for fitting, tuning or calibration.
- Results are reported as they come out. In the current data the LSTM does **not** reliably beat
  GARCH, and the app says so.

> Not investment advice. Free data may be delayed or revised.

---

## Quick start

Requires **Python 3.11+** (developed and tested on 3.12, macOS).

```bash
git clone https://github.com/ooLLeeGG08/Portfolio-Optimiser-with-Market-Implied-Risk.git
cd Portfolio-Optimiser-with-Market-Implied-Risk
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Open http://localhost:8501. No API keys or accounts are needed.

The default portfolio is AAPL, MSFT, JPM, XOM, JNJ, TLT and GLD, equally weighted, against SPY. Change it
on the **Portfolio Builder** page (weights or dollar amounts, benchmark, horizon, risk-free rate,
rebalancing, history length).

**Timing.** Market data loads in seconds and is cached on disk for 12 hours (`.cache/`). Volatility
forecasts (GARCH plus one LSTM per asset and one for the portfolio) run on demand from the
Volatility Forecasting page and take about 2 minutes for 7 assets on a laptop CPU. Option chains take
about 25 seconds. Both are then reused for the rest of the day.

---

## What the app does

| Section | Contents |
|---|---|
| **Dashboard** | Portfolio value, return, volatility, Sharpe, drawdown; VaR/CVaR; risk contributions; historical vs GARCH vs LSTM vs implied volatility |
| **Portfolio Builder** | Inputs and validation, data-quality report, composition (target vs drifted weights), performance vs benchmark, daily/monthly return statistics, correlation matrix, data provenance |
| **Portfolio Optimisation** | Equal weight, minimum variance, maximum Sharpe, custom constraints (min/max weight, target return or volatility), your own weights; efficient frontier; sensitivity of max-Sharpe weights to estimation error |
| **Volatility Forecasting** | Rolling 20/60/252-day volatility; GARCH(1,1) and LSTM forecasts with 10–90% intervals; out-of-sample forecast chart; GARCH diagnostics; LSTM training details |
| **Market-Implied Volatility** | VIX level, history, percentile and variance risk premium; option-chain implied volatility per asset (term structure, smile, filtered quotes); implied vs forecast comparison; Volatility Disagreement Indicator |
| **Risk Analytics** | Volatility, Sharpe, Sortino, beta, max drawdown; historical, parametric, Cornish–Fisher and Monte Carlo VaR/CVaR; Euler risk contributions; forward-looking risk from historical, GARCH, LSTM or implied volatility (Σ = D·R·D) with correlation stress |
| **Scenario Analysis** | Market −10%/−20%, +100 bp rates, volatility ×1.5, correlation convergence, single-asset shock, and a custom scenario builder — all labelled hypothetical |
| **Model Evaluation** | Walk-forward strategy backtest with costs and turnover (development vs locked test window); out-of-sample accuracy of every volatility model with Diebold–Mariano tests |
| **PDF Report** | 15-section research report generated from the current results |

### Volatility Disagreement Indicator

For each asset, the indicator measures how far historical, GARCH, LSTM and implied (or proxy)
volatility disagree at the same horizon: the standard deviation of their logarithms, with bands fixed
in advance (Low < 0.10 ≤ Moderate < 0.25 ≤ High). It is an analytical comparison, **not a trading
signal** — disagreement reflects different information and assumptions, not mispricing.

### PDF risk report

Generated from the PDF Report page. Sections: executive summary; portfolio composition; performance;
optimisation results; efficient frontier; historical risk metrics; VaR and CVaR; asset risk
contributions; volatility forecast comparison; market-implied volatility; stress tests; model
evaluation; methodology and assumptions; **model limitations and data quality**; data limitations and
disclaimers. The cover lists the generation time and every data observation date.

---

## Methods in brief

Full detail, formulas and the tests behind each: [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

| Area | Approach |
|---|---|
| Conventions | 252 trading days; volatility ×√252; arithmetic mean ×252; losses reported as positive numbers |
| Data | Split- and dividend-adjusted daily closes from Yahoo Finance; foreign listings converted to USD with daily exchange rates (unhedged, pence handled); assets aligned on shared dates, never filled; an unfinished trading day is excluded |
| Covariance | Ledoit–Wolf shrinkage |
| Optimisation | Long-only, fully invested, per-asset bounds; PyPortfolioOpt (CLARABEL); max-Sharpe sensitivity by resampling expected returns from their sampling distribution |
| VaR / CVaR | Historical, normal, Cornish–Fisher (withheld when the expansion is invalid), Student-t Monte Carlo; √h scaling labelled |
| Risk contribution | Euler decomposition; contributions sum exactly to portfolio volatility |
| Forward-looking risk | Σ = D·R·D with per-asset forecast or implied volatilities and historical correlations; correlation stress R → (1−λ)R + λ11ᵀ |
| Forecast target | Annualised realised volatility over the next h trading days (default 21) |
| GARCH(1,1) | Normal or Student-t by BIC on development data; walk-forward refits; simulation intervals |
| LSTM | Small network on backward-looking features; purged chronological splits; training-only scaling; quantile outputs with conformal calibration; early stopping and checkpointing |
| Evaluation | Locked test window; MAE, RMSE, bias, QLIKE, direction, coverage; non-overlapping subsample; Diebold–Mariano with HAC variance |
| Implied volatility | VIX; own Black-76 IV from option mid quotes after liquidity filters, interpolated in total variance; labelled proxy where no usable options exist |
| Backtest | Monthly walk-forward rebalancing on the previous 252 days; 10 bps on traded notional |

## Key limitations

- **Long-only.** No short positions or leverage; weights must sum to 100%.
- **Free data.** Yahoo Finance data can be delayed, revised or rate-limited; adjusted histories change
  retroactively. Delisted assets are largely missing (survivorship bias).
- **Expected returns are imprecise.** Standard errors of several percentage points a year make
  mean–variance weights unstable; all optimisation figures are in-sample.
- **Short test windows.** A 2-year test window contains only ~24 independent 21-day observations (8 at a
  63-day horizon), so differences between good models are rarely statistically significant.
- **Correlations are historical** and tend to rise in stress; use the correlation stress tools.
- **Option data is a current snapshot** with no history; Black-76 approximates American options.
- **Conformal intervals** assume exchangeable data; when the volatility regime shifts, LSTM intervals can
  under-cover (this happens in the current data and is reported).
- **Currencies.** Everything is valued in USD. Foreign listings (e.g. `7203.T`, `VOD.L`, `SAP.DE`) are
  converted unhedged, so their risk includes currency moves; exchange-rate closes are not synchronised
  with each market's close, which adds some noise to daily figures.
- **A short-lived asset shortens everything.** The analysis uses the assets' common history; the app warns
  when one asset truncates it.

---

## Project structure

```
app.py                 Streamlit entry point and navigation
config.py              Defaults (tickers, thresholds, test-window length, seeds)
core/                  Conventions (annualisation, loss sign) and shared types (risk basis labels)
data/                  Provider interface + yfinance implementation, cache, validation, preprocessing
portfolio/             Portfolio construction, optimisation, efficient frontier
risk/                  Covariance (incl. D·R·D), risk metrics, VaR, CVaR, risk contribution, stress tests
forecasting/           Target definition, historical volatility, GARCH, LSTM, evaluation, study runner
implied_volatility/    VIX, option chains, proxy, comparison and disagreement indicator
backtest/              Locked out-of-sample split, walk-forward strategy backtest
app_core/              Analysis pipeline shared by the app and the PDF report (no Streamlit)
reporting/             PDF generator and print charts
visualization/, ui/    Plotly theme and charts; Streamlit components and caching
views/                 One file per app page
tests/                 pytest suite (synthetic data; one live-data test)
docs/                  PLAN.md (phases) and METHODOLOGY.md (models, assumptions, limitations)
```

The quantitative code never imports Streamlit, so every model can be used and tested on its own.

## Tests

```bash
.venv/bin/python -m pytest              # 140+ unit tests on synthetic data (offline, ~15 s)
.venv/bin/python -m pytest -m network   # live Yahoo Finance integration test
```

The suite checks, among other things: closed-form optimisation solutions; that no random portfolio beats
the optimisers; VaR/CVaR against closed forms and each other; that risk contributions reconcile to
portfolio volatility; that changing future data never changes an earlier forecast, feature, scaler fit
or backtest weight; GARCH parameter recovery; reproducibility of GARCH intervals and LSTM training;
implied-volatility recovery from synthetic option chains; and the generated PDF's structure, figures and
labels.

## Data and dependencies

- Market data: Yahoo Finance through [yfinance](https://github.com/ranaroussi/yfinance) — prices, VIX
  (`^VIX`), 13-week T-bill yield (`^IRX`, risk-free rate), 10-year yield (`^TNX`), option chains.
  The `MarketDataProvider` interface allows another vendor to be plugged in.
- `requirements.txt` is a pinned lock generated from `requirements.in`
  (`uv pip compile requirements.in -o requirements.txt --python-version 3.12`).
- Main libraries: NumPy, pandas, SciPy, scikit-learn, statsmodels, PyPortfolioOpt, arch, PyTorch,
  Streamlit, Plotly, matplotlib, ReportLab.

---

*Educational and analytical software. Not investment advice, an offer or a recommendation to buy or sell
any security.*
