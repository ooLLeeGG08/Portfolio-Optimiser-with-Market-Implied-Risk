# Methodology, Assumptions and Limitations

This document is the reference for every quantitative choice in Portfolio Intelligence.
The PDF report's "Methodology" and "Model limitations and data quality" sections are
derived from it. Sections for components not yet implemented are marked
*(specification — implemented in Phase N)*; this file is updated as each phase lands.

---

## 1. Conventions

| Item | Convention |
|---|---|
| Trading days per year | 252 (`core.conventions.TRADING_DAYS`) |
| Volatility annualisation | σ_annual = σ_daily · √252. Assumes i.i.d. daily returns; understates/overstates if returns are autocorrelated |
| Mean-return annualisation | Arithmetic: μ_annual = 252 · μ_daily. Geometric (CAGR) reported separately for realised performance |
| Return types | **Simple** returns R_t = P_t / P_{t−1} − 1 for portfolio aggregation (portfolio simple return = Σ w_i R_i,t). **Log** returns r_t = ln(P_t / P_{t−1}) for volatility modelling |
| Loss convention | VaR and CVaR are reported as **positive numbers representing losses**: VaR_α = −q_{1−α}(R). A VaR of 0.021 at 95 % means "on 5 % of days the loss is expected to exceed 2.1 %" |
| Horizon scaling | Multi-day VaR uses √h scaling of the 1-day figure where stated; this assumes i.i.d. returns and is labelled whenever used |
| Prices | Split- and dividend-adjusted closing prices (total-return basis) |
| Risk-free rate | User-entered, or latest 13-week T-bill yield (^IRX) as an annualised decimal; treated as constant over the analysis window |

Every risk figure carries a **basis** label:
`OBSERVED`, `FORECAST`, `MARKET_IMPLIED`, `IMPLIED_PROXY`, `HYPOTHETICAL` (see PLAN.md).

---

## 2. Data acquisition and data quality

**Source.** yfinance (Yahoo Finance) behind a provider interface
(`data.market_data.MarketDataProvider`) so another vendor can be substituted.

**Caching.** Each ticker's full adjusted history is cached as Parquet with metadata
(source, fetch timestamp, first/last observation). Adjusted histories change retroactively
when dividends/splits occur, so a refresh replaces the whole series rather than appending.
The fetch timestamp is shown in the app and the PDF as the data observation date.

**Validation** (`data.data_validation`), producing a `DataQualityReport` of errors and warnings:

| Check | Treatment |
|---|---|
| Malformed ticker symbol | Error — rejected before any request |
| No data returned | Error — reported as "invalid or delisted"; asset excluded |
| Fewer than the minimum observations (default 252) | Error — asset excluded |
| Last observation materially older than the panel's (default > 10 trading days) | Warning — possible delisting/suspension |
| Non-positive prices | Error — asset excluded |
| Internal gaps (missing days inside an asset's own history relative to the panel calendar) | Warning with count; the rows are dropped at alignment, never filled |
| Extreme daily moves (|R| > 40 %) | Warning only — flagged for review, never removed (they may be real) |

**Alignment.** Assets are aligned on the **intersection** of their trading dates. Rows with
any missing price are dropped, not forward-filled — forward-filling creates artificial
zero returns that bias volatility and correlation downward. The number of dropped rows is
reported. Mixing assets from exchanges with different holiday calendars therefore loses
some dates; this is documented in the data-quality report.

**Monthly returns** are computed from month-end aligned prices.

**Known data limitations**
- Survivorship bias: Yahoo removes most delisted tickers, so historical universes contain
  only survivors.
- Adjusted prices are revised retroactively; re-runs on a later date can differ slightly.
- Yahoo applies rate limits; the cache mitigates but cannot eliminate this.
- Single-stock option data is a current snapshot only (no history) — see §9.

---

## 3. Portfolio construction

**Inputs.** Tickers with either investment amounts (weights = amount / total) or weights
(must be non-negative for long-only portfolios and sum to 1 within 1e-6).

**Value path.** The portfolio holds units of each asset. At each rebalancing date the
holdings are reset to target weights at that day's close; between rebalancing dates the
holdings are fixed and weights drift with prices. Supported frequencies: none
(buy-and-hold), monthly, quarterly, annual. The rebalancing date is the first trading day
of each new period. Phase 1 ignores transaction costs in the value path; the backtest (§8)
charges them explicitly.

Daily portfolio simple return: R_p,t = Σ_i w_i,t−1 · R_i,t, where w_i,t−1 are the drifted
(or freshly rebalanced) weights at the previous close. This is identical to the ratio of
consecutive portfolio values.

**Performance summary (`OBSERVED`).** Total return, CAGR = (V_T / V_0)^(252/N) − 1,
annualised volatility of daily simple returns.

---

## 4. Portfolio optimisation *(specification — implemented in Phase 2)*

- **Covariance:** Ledoit–Wolf shrinkage of the sample covariance of daily simple returns,
  annualised ×252. Raw sample covariance is noisy and drives optimisers to extreme weights.
- **Expected returns:** annualised arithmetic mean of historical returns. This is the
  weakest input of mean–variance optimisation; max-Sharpe weights are highly sensitive to
  it and the app shows a warning and a sensitivity check.
- **Strategies:** equal-weight (baseline), minimum variance, maximum Sharpe
  ((μ_p − r_f) / σ_p), custom (long-only, per-asset min/max, fully invested).
- **Efficient frontier:** minimum variance for a grid of target returns between the
  min-variance return and the maximum feasible return under the constraints.
- **No look-ahead:** estimates use only data up to the decision date.

---

## 5. Risk metrics *(specification — implemented in Phase 3)*

All `OBSERVED` unless stated.

- Max drawdown: min_t (V_t / max_{s≤t} V_s − 1).
- Sharpe: (μ_annual − r_f) / σ_annual using arithmetic annualised mean.
- Sortino: (μ_annual − r_f) / downside deviation, downside deviation computed from
  min(R_t − r_f/252, 0) and annualised ×√252.
- Beta: Cov(R_p, R_b) / Var(R_b) on aligned daily returns.
- **VaR / CVaR** at 95 % and 99 %, 1-day (and √h-scaled where labelled):
  - Historical: empirical quantile of portfolio returns; CVaR = mean loss beyond VaR.
  - Parametric normal: VaR = −(μ + z_{1−α} σ); CVaR = −μ + σ φ(z_{1−α}) / (1−α).
  - Cornish–Fisher: skew/kurtosis-adjusted quantile (fat-tail sensitivity).
  - Monte Carlo: multivariate Student-t simulation of asset returns with fixed seed,
    covariance matched to the estimate; VaR/CVaR from simulated portfolio returns.
  - Interpretation: VaR is a threshold, not a worst case; CVaR is the average loss on the
    days that breach the threshold.
- **Risk contribution (Euler):** σ_p = √(wᵀΣw); MCR_i = (Σw)_i / σ_p;
  CCR_i = w_i · MCR_i; Σ_i CCR_i = σ_p exactly; %CR_i = CCR_i / σ_p.

---

## 6. Forward-looking covariance: Σ = D · R · D *(specification — implemented in Phase 3)*

To translate per-asset volatility estimates (historical, GARCH, LSTM, market-implied or
proxy) into portfolio risk, the covariance matrix is rebuilt as

  Σ_fwd = D · R · D,  D = diag(σ_1, …, σ_n),  R = correlation matrix.

**Assumptions (stated in the app and the PDF):**
1. **Correlations are taken from history.** R is estimated from historical returns
   (sample or shrinkage). Historical correlations may not represent future market
   conditions; in particular, correlations tend to rise in market stress, which reduces
   diversification exactly when it is needed.
2. **Volatility and correlation are treated as separable.** Changing σ_i does not change R.
   In reality volatility shocks and correlation shocks are linked.
3. **Horizon consistency.** All σ_i in D must refer to the same horizon and annualisation
   (e.g. annualised vol over the next 21 trading days). Mixing horizons is rejected.
4. **Basis is inherited.** Σ_fwd built from `FORECAST` vols is a forecast; from
   `MARKET_IMPLIED` vols it mixes risk-neutral volatility with historical correlation and is
   labelled as such; if any σ_i is a proxy the result is labelled `IMPLIED_PROXY`.
5. **Positive semi-definiteness** holds automatically if R is a valid correlation matrix;
   R is checked (and projected to the nearest valid correlation matrix if needed, with a
   warning).

**Correlation stress testing.** Because assumption 1 is the weakest link, every
forward-looking risk figure can be recomputed under stressed correlations:

  R_λ = (1 − λ) · R + λ · 1·1ᵀ, λ ∈ [0, 1]   (diagonal stays 1; PSD for all λ).

The app shows portfolio volatility, VaR and risk contributions as a function of λ, labelled
`HYPOTHETICAL`. λ = 1 is the no-diversification bound σ_p = Σ w_i σ_i (long-only).

---

## 7. Volatility forecasting *(specification — implemented in Phase 4)*

**Target (single definition, `forecasting.targets`).** Realised volatility over the next h
trading days, annualised:

  RV_{t,h} = √( (252 / h) · Σ_{k=1..h} r_{t+k}² )

using daily log returns and a zero-mean assumption (standard for short horizons; the mean
is negligible relative to volatility at daily frequency). Default h = 21. A forecast made at
the close of day t uses only data up to and including day t.

**Historical volatility.** Rolling 20/60/252-day standard deviation of log returns,
annualised. Used as a naive forecast of RV_{t,h} (random-walk-in-volatility benchmark).

**GARCH(1,1)** via `arch`: σ²_t = ω + α ε²_{t−1} + β σ²_{t−1}, with normal or Student-t
innovations, fitted to percentage log returns. Walk-forward: parameters refitted on an
expanding window at a fixed interval (default every 21 days) using data up to t only.
The h-day forecast sums the 1…h step variance forecasts:
RV̂_{t,h} = √( (252/h) · Σ_k σ̂²_{t+k|t} ).
Diagnostics: persistence α+β, half-life ln(0.5)/ln(α+β), Ljung–Box on standardised and
squared standardised residuals, parameter standard errors.

**GARCH uncertainty.** Simulation-based forecast distribution: paths of h daily returns
are simulated from the fitted model (innovations drawn from the fitted distribution); the
realised vol of each path gives a predictive distribution of RV_{t,h}, from which 10 %,
50 % and 90 % quantiles are reported. This captures innovation uncertainty but not
parameter-estimation uncertainty (stated as a limitation).

**LSTM.** Small network (1–2 layers, ~32 hidden units, dropout). Inputs: sequences
(default length 60) of log return, |r|, r², and 5/21/63-day realised vol, all computed with
data ≤ t. Target: log RV_{t,h}. Chronological 70/15/15 train/validation/test split with an
**h-day purge gap** between consecutive segments (otherwise the last training target
window overlaps the first validation window). Feature scaler fitted on training data only.
Fixed seeds, Adam, early stopping on validation loss, best-epoch checkpoint.

**LSTM uncertainty.** The network outputs the 10 %, 50 % and 90 % quantiles of log RV
(pinball loss). Intervals are then **split-conformally calibrated** on the validation
segment (conformalised quantile regression), which gives finite-sample coverage
guarantees under exchangeability. Exchangeability is only approximate for time series,
so **empirical coverage on the test segment is always reported**.

**Evaluation (out-of-sample only).** MAE, RMSE, bias (mean of forecast − realised),
directional accuracy (sign of forecast change vs sign of realised change), QLIKE loss
(robust to noise in the realised-vol proxy), and interval coverage / width where intervals
exist.

**Overlapping horizons.** Daily forecasts of 21-day-ahead RV have overlapping target
windows, so consecutive errors are strongly autocorrelated. Consequences: the effective
sample size is ~N/h, and naive standard errors overstate significance. Therefore metrics
are reported on (a) all forecasts and (b) a non-overlapping subsample (every h-th day), and
pairwise comparisons use a Diebold–Mariano test with Newey–West (HAC) variance with
h−1 lags. No model is assumed to win; results are reported as they come out.

---

## 8. Out-of-sample protocol and backtesting *(specification — implemented in Phases 4 and 8)*

**Locked test window.** The most recent `TEST_WINDOW_DAYS` (default 504 trading days ≈ 2
years) of the dataset are reserved as the out-of-sample test window, defined once in
`backtest.splits`. Everything before it is the development window.

- All hyper-parameters (LSTM architecture/learning rate/sequence length, GARCH
  distribution choice, optimiser lookback, rebalancing defaults) are fixed a priori or
  selected using development data only (with the train/validation split inside the
  development window).
- The test window is evaluated once per configuration; its results are never fed back into
  a choice. Code enforces this: model-selection functions refuse inputs that overlap the
  test window.
- Walk-forward strategy backtest: at each rebalancing date, weights are estimated from the
  preceding lookback window only (default 252 days), applied from the next day, and drift
  until the next rebalance. Turnover = ½ Σ |w_new − w_drifted| (one-way); transaction cost
  = turnover × cost (default 10 bps). Reported: annualised return, volatility, Sharpe,
  max drawdown, turnover, cost drag — versus equal-weight.
- Survivorship bias cannot be removed with this data source and is stated.
- No claim of market-beating performance is made.

---

## 9. Market-implied volatility *(specification — implemented in Phase 5)*

**VIX (`MARKET_IMPLIED`).** The CBOE VIX is the risk-neutral expected volatility of the
S&P 500 over the next ~30 calendar days, derived from SPX option prices. It is **not** a
volatility forecast for individual stocks and is never used as one. Displayed: latest
observation and date, history, rolling averages, percentile rank in its own history,
changes.

**Option-chain implied vol (`MARKET_IMPLIED`, `IVSource.OPTIONS_CHAIN`).** Per-asset implied
vols from listed option chains (current snapshot only). Quotes are filtered: zero bid,
bid/ask spread above threshold, low open interest/volume, stale last trade, and
implausible IV values are excluded and counted. ATM IV per expiry = interpolation of call
and put IVs nearest the forward/spot; smile/skew shown only when enough strikes survive the
filters. Term structure interpolated to the comparison horizon (variance-linear in time).

**Implied-vol proxy (`IMPLIED_PROXY`, `IVSource.PROXY`).** Where no usable option data
exists: σ_proxy,i = σ_hist,i × (VIX / realised S&P 500 vol). This assumes each asset's
implied/realised ratio equals the index's. It is an **estimate**, shown with a distinct
label and style, never under "market-implied", and excluded from any statistic described as
market-implied.

**Interpretation.** Implied volatility is risk-neutral and typically exceeds subsequently
realised volatility because of the volatility risk premium. A gap between implied and
forecast volatility is **not** evidence that options are mispriced.

**Volatility Disagreement Indicator (analytical, not a trading signal).** For each asset,
at a common horizon (default 21 trading days, annualised), compare historical, GARCH,
LSTM, and implied (or proxy) vol:
- Dispersion D = standard deviation of the log of the available estimates (scale-free;
  0 = full agreement). Reported with which estimates were available and their bases.
- Implied-to-forecast ratio IV / mean(model forecasts), only when actual market-implied IV
  is available.
- Context: percentile of the model-only dispersion in its own history (implied vol has no
  single-stock history in this data source, so the percentile is computed only for model
  estimates).
Displayed as "Low / Moderate / High disagreement" bands with the explicit note that
disagreement reflects model and information differences and is not a buy/sell signal.

**Horizon alignment.** VIX is a 30-calendar-day (~21 trading-day) measure; option-chain IV
is interpolated to h; model forecasts are produced for h. All are annualised the same way.
Residual mismatch (calendar vs trading-day variance accrual) is documented.

---

## 10. Stress testing *(specification — implemented in Phase 3)* — all `HYPOTHETICAL`

- Market decline of x %: asset shock_i = β_i · x (β from historical regression on the
  benchmark); idiosyncratic component ignored (stated).
- Volatility increase: D scaled by a factor k; risk recomputed.
- Correlation convergence: R_λ as in §6.
- Interest-rate shock: per-asset sensitivity to daily changes in the 10-year yield (^TNX)
  estimated by regression; equity rate sensitivities are weak and unstable (stated).
- Single-asset shock and fully custom shock vectors.
Results: portfolio loss, asset contributions, post-scenario value, baseline vs stressed risk.
Stress tests are not predictions and are never labelled as such.

---

## 11. General limitations

- Mean–variance optimisation depends on expected-return estimates that are very noisy.
- i.i.d. assumptions behind √252 and √h scaling are violated by volatility clustering.
- Normal and Student-t VaR models are approximations; historical VaR is limited by the
  sample (99 % 1-day VaR from 2 years of data rests on ~5 observations).
- Correlations are unstable and rise in crises.
- Realised volatility is itself a noisy estimate of true volatility.
- Free data sources have gaps, revisions and survivorship bias.
