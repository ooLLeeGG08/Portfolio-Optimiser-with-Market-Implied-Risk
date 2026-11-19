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

## 4. Portfolio optimisation

Implemented in `portfolio/optimization.py`, `portfolio/efficient_frontier.py`, `risk/covariance.py`.

**Inputs (estimated from daily simple returns dated ≤ the decision date — no look-ahead).**
- **Expected returns:** annualised *arithmetic* mean, μ = 252 · mean(R). The arithmetic mean
  is the quantity that aggregates linearly across assets and enters the Sharpe ratio
  consistently. It is by far the noisiest input: its standard error is σ_annual / √years
  (≈ 9 % a year for a 29 %-vol stock over 10 years), which the app displays.
- **Covariance:** Ledoit–Wolf (2004) shrinkage towards a scaled identity, rescaled from the
  1/T to the unbiased 1/(T−1) normalisation and annualised ×252. The shrinkage intensity
  is reported. Sample covariance is available as an alternative.
- **Lookback:** full selected window by default; a rolling lookback is used in the
  walk-forward backtest.

**Constraints.** Long-only, fully invested (Σw = 1), per-asset minimum and maximum weights
(scalar or per ticker). Infeasible bound sets (Σ min > 1, Σ max < 1, min > max, negative
min) are rejected with an explanation before any solve.

**Strategies.**
| Strategy | Definition |
|---|---|
| Equal weight | w_i = 1/N; baseline, ignores constraints |
| Minimum variance | min wᵀΣw |
| Maximum Sharpe | max (wᵀμ − r_f) / √(wᵀΣw); undefined (reported, not forced) if no asset has μ_i > r_f |
| Custom | user bounds with objective min-variance, max-Sharpe, target return (min variance s.t. wᵀμ ≥ target) or target volatility (max return s.t. σ ≤ target) |
| User-defined | the user's own weights, evaluated with the same inputs |

Solved with PyPortfolioOpt (cvxpy, CLARABEL solver). Solver output is cleaned (clipped to
bounds, dust < 1e-8 removed, renormalised) and re-verified against the bounds. All reported
statistics are recomputed by one function (`portfolio_stats`) so every strategy is measured
identically. Each portfolio shows its total expected return, its **excess return**
(expected return − r_f) and Sharpe = excess return / volatility.

**Efficient frontier.** Minimum variance for a grid of target returns from the
minimum-variance portfolio's return to the maximum feasible return under the bounds (an
LP solved exactly by greedy allocation from the lower bounds). Only the efficient branch
is drawn. Failed grid points are dropped and counted, never interpolated.

**Max-Sharpe sensitivity.** Expected returns are resampled from their sampling
distribution, μ* ~ N(μ̂, Σ / years), and the max-Sharpe portfolio is re-solved for each draw
(default 200, fixed seed). The app reports the point weights alongside the mean and 5th–95th
percentile of the resampled weights. On real data the ranges are typically very wide —
i.e. the "optimal" weights are not statistically well determined.

**Interpretation.** Expected return, volatility and Sharpe shown for optimised portfolios are
**in-sample estimates** (basis: `OBSERVED` inputs, not forecasts). They overstate what the
portfolio would have achieved out of sample, because the weights were chosen using the same
data. Out-of-sample performance comes from the walk-forward backtest (§8).

**Verification (tests).** Closed-form global-minimum-variance and tangency portfolios are
recovered when bounds are not binding; no random feasible portfolio has lower volatility than
min-variance or higher Sharpe than max-Sharpe; frontier volatility is non-decreasing in return
and no random portfolio lies below exact frontier solves; the max-return LP matches scipy's
`linprog`; future data cannot change estimates made at an earlier decision date.

---

## 5. Risk metrics

Implemented in `risk/risk_metrics.py`, `risk/var.py`, `risk/cvar.py`, `risk/tail_risk.py`,
`risk/risk_contribution.py`. All `OBSERVED` unless stated.

**Standard measures** (daily simple returns of the portfolio value path):
- Annualised return 252·mean(R); annualised volatility √252·std(R) (ddof = 1).
- Sharpe = (annualised return − r_f) / annualised volatility.
- Sortino = (annualised return − r_f) / downside deviation, where downside deviation =
  √252 · √(mean over all days of min(R_t − r_f/252, 0)²).
- Maximum drawdown: min_t (V_t / max_{s≤t} V_s − 1), with peak, trough and recovery dates
  and durations in trading days (recovery = first close back at the prior peak).
- Beta = Cov(R_p, R_b) / Var(R_b) on shared dates; correlation with the benchmark.
- Factor sensitivities (used for rate shocks): OLS slope of daily asset returns on a daily
  factor change, with Newey–West (HAC, 5 lags) t-statistics and R² so weak or unstable
  sensitivities are visible.

**VaR and CVaR.** Loss L = −R; VaR_a = a-quantile of L; CVaR_a = E[L | L ≥ VaR_a]; both
positive when they are losses. All methods are computed at 1 day; an h-day figure is the
1-day figure × √h and is labelled as an i.i.d. approximation.

| Method | VaR | CVaR |
|---|---|---|
| Historical | Empirical quantile of portfolio losses (linear interpolation) | Mean of losses ≥ VaR |
| Parametric (normal) | −μ + σ z_a | −μ + σ φ(z_a)/(1−a) |
| Cornish–Fisher | −(μ + σ z_CF), z_CF = z + (z²−1)S/6 + (z³−3z)K/24 − (2z³−5z)S²/36 at z = Φ⁻¹(1−a) | (1/(1−a)) ∫_a^1 VaR_CF(u) du (quadrature) |
| Monte Carlo | Quantile of 100,000 simulated portfolio returns (fixed seed) | Mean of simulated losses ≥ VaR |

- μ, σ, S (skew), K (excess kurtosis) are sample moments of the portfolio's daily returns.
- **Cornish–Fisher validity.** The expansion is a valid quantile function only where it is
  increasing in z. For high kurtosis (roughly K > 8, e.g. any sample containing March 2020)
  it is not, and its VaR is badly distorted. In that case the method is **withheld** (shown
  as "not reported" with the reason) rather than displayed.
- **Monte Carlo.** Asset returns are drawn from a multivariate Student-t scaled so that its
  covariance equals the sample covariance of daily asset returns, with the sample mean
  vector; degrees of freedom by method of moments from the portfolio's excess kurtosis,
  ν = 4 + 6/K, clamped to [4.1, 50] (normal if K ≈ 0). Portfolio return = w · R (simple
  returns aggregate linearly). A single ν for all assets is a simplification.
- **Forward-looking VaR** (from a D·R·D covariance, §6) uses parametric-normal and Monte
  Carlo methods with zero expected return; no historical method exists for a forecast
  covariance. It inherits the covariance's basis label.
- **Interpretation.** VaR is a threshold, not a worst case; CVaR is the average loss on the
  days beyond that threshold. A 99 % 1-day historical VaR from 10 years rests on about 25
  observations; from 2 years, about 5 — the app states the count.

**Risk contribution (Euler).** σ_p = √(wᵀΣw); MCR_i = (Σw)_i / σ_p; CCR_i = w_i·MCR_i;
Σ_i CCR_i = σ_p exactly (σ_p is homogeneous of degree one in w); %CR_i = CCR_i / σ_p.
Negative contributions mean the asset hedges the rest of the portfolio. Under a zero-mean
normal model VaR_a = z_a σ_p, so each asset's share of VaR equals %CR_i. Works with any
covariance (historical, forecast, implied, stressed) and inherits its basis.

**Verification (tests).** Hand-calculated drawdown, Sharpe and Sortino; beta and rate
sensitivity recovered from synthetic data; parametric formulas against closed forms;
Cornish–Fisher reduces to normal when S = K = 0; historical and Monte Carlo VaR/CVaR converge
to the normal closed forms on large normal samples; Student-t simulation reproduces the
target covariance; CVaR ≥ VaR and VaR₉₉ > VaR₉₅ for every method; Euler contributions sum
to σ_p and marginal contributions match numerical derivatives.

---

## 6. Forward-looking covariance: Σ = D · R · D

Implemented in `risk/covariance.py` (`VolatilityVector`, `build_forward_covariance`,
`stress_correlation`, `nearest_correlation`, `correlation_stress_profile`).

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
   R is checked (unit diagonal, |ρ| ≤ 1, PSD) and, if invalid, projected to the nearest
   valid correlation matrix by Higham's (2002) alternating-projections algorithm, with a
   recorded note.

Basis precedence when inputs are mixed: hypothetical > implied proxy > market-implied >
forecast > observed — the least-observed component labels the result.

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

## 10. Stress testing — all `HYPOTHETICAL`

Implemented in `risk/stress_testing.py`. Every result is labelled "Hypothetical scenario —
not a prediction".

**Price-shock scenarios** — instantaneous per-asset simple-return shocks, floored at −100 %;
portfolio return = Σ w_i shock_i; asset P&L = w_i · shock_i · value; value after the shock.
| Scenario | Asset shock |
|---|---|
| Market decline x % (default −10 %, −20 %) | β_i · x, with β from historical daily returns against the benchmark |
| Interest-rate shock Δ bp | s_i · Δ/100, s_i = sensitivity to a 1-percentage-point change in the 10-year yield (^TNX) |
| Single-asset shock | user-specified for chosen assets; others unchanged |
| Custom | any per-asset shock vector |

Assumptions shown with the results: idiosyncratic moves are ignored; betas and rate
sensitivities are estimated in normal markets and can change abruptly in a crisis (e.g.
bonds and equities fell together in 2022); equity rate sensitivities are small and
statistically weak (their t-statistics are shown); the response is linear (no convexity).
Sanity check on real data: TLT's estimated sensitivity is about −0.15 per percentage point,
consistent with its ~15–17-year duration.

**Risk-regime scenarios** — the covariance is stressed and compared with the baseline
(annualised volatility and 1-day zero-mean parametric VaR at 95 % / 99 %, plus Euler risk
contributions under stress):
- Volatility increase ×k: Σ → k²Σ (correlations unchanged).
- Correlation convergence: R → (1−λ)R + λ11ᵀ, optionally combined with ×k volatility.

---

## 11. General limitations

- Mean–variance optimisation depends on expected-return estimates that are very noisy.
- i.i.d. assumptions behind √252 and √h scaling are violated by volatility clustering.
- Normal and Student-t VaR models are approximations; historical VaR is limited by the
  sample (99 % 1-day VaR from 2 years of data rests on ~5 observations).
- Correlations are unstable and rise in crises.
- Realised volatility is itself a noisy estimate of true volatility.
- Free data sources have gaps, revisions and survivorship bias.
