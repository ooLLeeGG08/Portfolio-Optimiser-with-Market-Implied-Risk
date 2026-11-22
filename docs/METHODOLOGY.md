# Methodology, Assumptions and Limitations

This document is the reference for every quantitative choice in Portfolio Optimizer.
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

**Currency conversion** (`data/currency.py`, applied inside `MarketDataService.get_price_data`, so
the app, forecasts, backtests and report all receive base-currency prices). All prices are
expressed in the base currency, USD (`config.BASE_CURRENCY`):
- The quote currency of each ticker comes from the provider and is cached. Minor units are
  normalised first: London prices quoted in pence (`GBp`/`GBX`) are divided by 100, likewise
  `ZAc` and `ILA`.
- Conversion is **unhedged**: P_USD,t = P_local,t × X_t, with X_t = USD per unit of local
  currency. The USD-quoted pair (e.g. `USDJPY=X`) is preferred and inverted because it carries
  more significant digits; the direct pair (`JPYUSD=X`) is the fallback. USD returns therefore
  equal (1 + local return)(1 + currency return) − 1, i.e. they include exchange-rate risk, as
  for an investor who holds the foreign asset and measures wealth in USD.
- Prices are cut to the analysis window first; dates inside it with a price but no exchange
  rate are dropped, never filled, and counted in the data-quality report.
- Unknown currency: a ticker with an exchange suffix (e.g. `.T`, `.L`) is excluded rather than
  guessed; a US-style symbol is assumed to be in USD, with a warning.
- Every conversion (currency, FX series, dropped dates) is listed in the data-quality report,
  the provenance table and the PDF.
- Limitation: FX daily closes are not taken at each exchange's closing time, which adds noise to
  daily returns and slightly understates same-day correlations between markets in different
  time zones.

**Known data limitations**
- Survivorship bias: Yahoo removes most delisted tickers, so historical universes contain
  only survivors.
- Adjusted prices are revised retroactively; re-runs on a later date can differ slightly.
- Yahoo applies rate limits; the cache mitigates but cannot eliminate this.
- Single-stock option data is a current snapshot only (no history), often delayed, and
  thin for many strikes/expiries — see §9.

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

## 7. Volatility forecasting

Implemented in `forecasting/` (`targets.py`, `historical_volatility.py`, `garch_model.py`,
`lstm_model.py`, `evaluation.py`, `study.py`) and `backtest/splits.py`. Basis: `FORECAST`
(rolling historical vol displayed as history is `OBSERVED`).

**Target (single definition, `forecasting.targets`).** Realised volatility over the next h
trading days, annualised:

  RV_{t,h} = √( (252 / h) · Σ_{k=1..h} r_{t+k}² )

from daily log returns, zero-mean (the daily mean is negligible relative to volatility).
Default h = 21. The same formula looking backwards, RV^back_{t,w}, is the historical
volatility; by construction RV_{t,h} = RV^back_{t+h,h} (tested). A forecast made at the close
of day t uses only data up to and including day t.

**Historical volatility.** Trailing 20/60/252-day RV^back, used unchanged as the forecast of
RV_{t,h} (a random walk in volatility).

**GARCH(1,1)** (`arch`, constant mean, percent log returns):
σ²_t = ω + α ε²_{t−1} + β σ²_{t−1}. Innovations normal or Student-t, chosen by BIC on
development data only.
- *Timing* (verified against a hand-written recursion): estimation with `last_obs = k` uses
  observations 0…k−1; a forecast row dated t is σ²_{t+1…t+h | t}. Walk-forward therefore
  refits with `last_obs = position(t) + 1`.
- *Walk-forward:* parameters re-estimated every 21 trading days on an expanding window;
  between refits the variance filter is updated with each new return using fixed parameters.
  A test perturbs all returns after a date and checks every earlier forecast is unchanged.
- *h-day forecast:* RV̂_{t,h} = √((252/h) Σ_k σ̂²_{t+k|t}) / 100.
- *Diagnostics:* α, β with standard errors, persistence α+β, half-life ln 0.5 / ln(α+β),
  unconditional volatility √(ω/(1−α−β))·√252, Ljung–Box (lag 10) on standardised and squared
  standardised residuals, AIC/BIC.
- *Intervals:* 500 simulated h-day return paths per forecast date from the fitted model, with
  a seeded innovation generator (arch's own `random_state` is not honoured in simulation, so
  a custom generator is supplied — otherwise intervals would not be reproducible). The
  realised vol of each path gives the predictive distribution; 10/50/90 % quantiles are
  reported. Parameter-estimation uncertainty is not included.

**LSTM** (PyTorch). Hyper-parameters fixed a priori (`LSTMConfig`): sequence length 60, one
LSTM layer of 32 units, dropout 0.1, Adam (lr 1e-3, weight decay 1e-5), batch 64, at most 150
epochs, early-stopping patience 15, gradient clipping 1.0, seed 42 with deterministic
algorithms.
- *Features at t* (data ≤ t only, tested by perturbing future returns): log return, |r|,
  log RV^back over 5, 21 and 63 days.
- *Target:* log RV_{t,h}.
- *Splits:* the development window (before the locked test window, §8) is divided
  chronologically into train 70 % / validation 15 % / calibration 15 %, with an h-day purge
  gap between consecutive segments so no target window spans two segments. Validation drives
  early stopping; calibration is used only for the conformal step.
- *Scaling:* `StandardScaler` fitted on feature rows dated up to the last training date only
  (tested).
- *Output:* 10/50/90 % quantiles of log RV with non-crossing guaranteed by construction
  (q10 = m − softplus(a), q90 = m + softplus(b)); pinball loss. Point forecast = median.
- *Checkpointing:* the best-validation-epoch weights are saved to `.cache/checkpoints/` and
  restored before use.
- *Calibration (CQR):* scores E_i = max(q10_i − y_i, y_i − q90_i) on the calibration segment;
  the interval is widened by the ⌈(n+1)·0.8⌉/n empirical quantile of E (negative values
  narrow it). This guarantees ≥ 80 % coverage only under exchangeability, which time series
  violate when the volatility regime shifts; test-window coverage is therefore always
  reported and can fall short.
- The LSTM is trained once on development data and not retrained in the test window (GARCH
  is refitted); this favours GARCH slightly and is stated.

**Evaluation (test window only).** MAE, RMSE, bias (forecast − realised), QLIKE
(y²/f² − ln(y²/f²) − 1; robust to noise in realised volatility and the preferred loss for
volatility forecasts), directional accuracy, and 10–90 % interval coverage and width.
- *Directional accuracy:* share of dates where sign(f − c) = sign(y − c), with c the trailing
  21-day realised vol at t. For the 20-day historical forecast f ≈ c, so its directional
  accuracy is essentially noise and should not be read as skill.
- *Overlapping horizons:* daily forecasts of 21-day-ahead RV share 20 days of target, so
  errors are strongly autocorrelated and the effective sample is ≈ N/h (≈ 24 independent
  observations in a 2-year test window). Metrics are reported on all dates and on a
  non-overlapping subsample (every h-th date). Pairwise comparisons use the Diebold–Mariano
  test against GARCH under QLIKE and squared error, with a Bartlett HAC variance (h−1 lags)
  and the Harvey–Leybourne–Newbold small-sample correction (Student-t, n−1 df).
- No model is assumed to win. With ~24 independent observations, differences between good
  models are usually not statistically significant, and the app says so.

---

## 8. Out-of-sample protocol and backtesting

**Locked test window** (`backtest/splits.py`). The last `TEST_WINDOW_DAYS` (504) dates on
which the target is known form the test window; the development window ends `purge_days`
(= h for forecasting) earlier so that no development target overlaps test-period returns.
`assert_development_only` raises `TestDataLeakageError` if any model-selection step (GARCH
distribution choice, LSTM scaling/training/validation/calibration) receives a date after the
development window.

- All hyper-parameters are fixed a priori (LSTM) or chosen on development data (GARCH
  distribution). Nothing observed in the test window is used to change any choice; results
  are reported as obtained, even when unfavourable.
- Strategy backtest (`backtest/walk_forward.py`): at each rebalancing date,
  weights are estimated from the preceding lookback window only (default 252 days), applied
  from the next day, and drift until the next rebalance. Turnover = ½ Σ |w_new − w_drifted|
  (one-way); transaction cost = cost rate × Σ |w_new − w_drifted| (total traded notional, i.e.
  2 × one-way turnover; default 10 bps), deducted at the rebalance; the initial allocation is
  not charged. Maximum Sharpe falls back to minimum variance (counted and reported) when no
  asset's estimated return exceeds the risk-free rate. Reported: annualised return,
  volatility, Sharpe, max drawdown, turnover, cost drag — versus equal-weight, separately for
  the development and the locked test window.
- Survivorship bias cannot be removed with this data source and is stated.
- No claim of market-beating performance is made.

---

## 9. Market-implied volatility

Implemented in `implied_volatility/` (`vix.py`, `options_data.py`, `proxy.py`,
`comparison.py`) and `MarketDataProvider.fetch_option_chain`.

Every implied figure carries a source (`IVSource`): `VIX index` and `listed option chain` are
market data (basis `MARKET_IMPLIED`); `proxy estimate` is not (basis `IMPLIED_PROXY`, shown
as "Implied-vol proxy (estimate, not market data)"); `unavailable` is reported as such.

**VIX** (`MARKET_IMPLIED`, `IVSource.VIX_INDEX`). The CBOE VIX is the risk-neutral expected
volatility of the S&P 500 over the next ~30 calendar days, derived from SPX option prices. It
is **not** a volatility forecast for individual stocks and is never used as one. Shown: latest
observation and date; changes over 1, 5 and 21 days (volatility points); 20/60/252-day
rolling means; percentile rank of the latest value in the full history (since 1990) and the
last 5 years.
- *Variance risk premium (ex post):* VIX at t versus S&P 500 (SPY) realised volatility over
  the next 21 trading days (the §7 target definition). Historically VIX exceeds subsequently
  realised volatility on most days (≈ 84 % of days since 2005, by ≈ 3.6 vol points on average
  in our data) — the premium option buyers pay for volatility protection.
- *VIX vs ATM implied vol:* VIX is a variance-swap-style index that integrates the whole
  strike range, including expensive out-of-the-money puts, so it usually sits above the
  at-the-money implied volatility of SPY options. Both are shown; they are not expected to match.

**Option-chain implied volatility** (`MARKET_IMPLIED`, `IVSource.OPTIONS_CHAIN`; current
snapshot only — no history is available from this provider).
- *Our own IV, not the provider's:* implied vols are computed from bid/ask **mid** prices with
  the Black-76 model on the forward, inverted by Brent's method. The provider's
  `impliedVolatility` field (undocumented model) is kept only as a cross-check; on live data
  the median difference is within ±1 vol point.
- *Forward:* from put–call parity F = K + e^{rT}(C − P), median over the three strikes nearest
  spot where both call and put pass the filters; this avoids assuming a dividend yield. If no
  pair exists, F = S·e^{rT} (dividends ignored) and the expiry is flagged.
- *Time to expiry:* calendar time from the quote timestamp to 16:00 New York on the expiry
  date, /365. Rate r = the risk-free rate (§1).
- *Quote filters* (each exclusion counted by reason; missing data fails a filter): bid ≥ 0.01;
  ask > bid; (ask − bid)/mid ≤ 50 %; open interest ≥ 10; last trade within 5 calendar days;
  expiry ≥ 7 calendar days (ATM/term structure); |ln(K/F)| ≤ 0.5; price within no-arbitrage
  bounds; 1 % ≤ IV ≤ 300 %. Typically only 30–50 % of listed quotes survive.
- *Smile outlier rule* (applied last): within an expiry, a quote is excluded when its IV differs by
  more than 25 % (relative) from the median IV of the out-of-the-money quotes at the nearest 2
  strikes below and 2 above (the quote itself excluded; at least 3 reference quotes with one on
  each side, otherwise the quote is never flagged). The OTM quotes are the reference because they
  are the ones used for ATM IV; in-the-money quotes are less liquid and carry an early-exercise
  premium that Black-76 ignores, which shows up as isolated spikes (e.g. an AAPL put at strike 395
  with IV ≈ 35 % against ≈ 24 % at the neighbouring strikes, 20 Nov 2026). A median follows a steep
  but smooth skew, so only isolated departures are flagged. The 25 % threshold is a fixed round
  number, not fitted; excluded quotes are counted under their own reason. On the 20 Nov 2026
  snapshot of the default assets it removed only in-the-money quotes, so ATM and horizon IVs were
  unchanged.
- *ATM IV per expiry:* from out-of-the-money options only (puts K < F, calls K ≥ F — smallest
  early-exercise premium; Black-76 is a European model and listed equity options are
  American, an approximation that is stated), linear interpolation in log-moneyness
  k = ln(K/F) to k = 0 between the nearest valid quotes on each side. Requires ≥ 4 valid OTM
  quotes in the expiry.
- *Horizon IV:* the forecast horizon h trading days is converted to h·365/252 calendar days
  (21 → ≈ 30.4, matching VIX's 30 days). Total implied variance w(T) = σ²T is interpolated
  linearly between the bracketing expiries; σ = √(w/T). If no bracketing pair exists, the
  nearest expiry is used flat and the estimate is marked not reliable.
- *Smile/skew:* valid quotes for a chosen expiry, IV against strike and log-moneyness, calls
  and puts distinguished.

**Implied-volatility proxy** (`IMPLIED_PROXY`, `IVSource.PROXY` — an estimate, not market
data). Used only when no usable option-chain IV exists, with the reason recorded:
σ_proxy,i = σ_hist,i × (VIX / σ_hist,S&P), with 60-day trailing realised volatilities of the
asset and SPY on the same date as VIX. Assumes the asset's implied/realised ratio equals the
index's (transfers the index's current risk premium and fear level; ignores asset-specific
events). Any covariance built with a proxy is labelled `IMPLIED_PROXY` (§6). The proxy is
excluded from every statistic described as market-implied, and no implied-to-forecast ratio
is computed from it.

**Implied vs forecast comparison.** For each asset at the common horizon h (annualised):
historical (trailing 60-day realised, `OBSERVED`), GARCH and LSTM (`FORECAST`, with 10–90 %
intervals), and implied (`MARKET_IMPLIED`) or proxy (`IMPLIED_PROXY`). For the S&P 500,
VIX is additionally aligned with the out-of-sample GARCH/LSTM forecasts and the subsequently
realised volatility over the test window. Implied volatility is risk-neutral and contains a
volatility risk premium; a gap between implied and forecast volatility is **not** evidence
that options are over- or under-priced.

**Volatility Disagreement Indicator** (analytical comparison — not a trading signal).
- *Dispersion* D = population standard deviation of the logs of the available estimates
  (scale-free; D = 0.10 means the estimates sit roughly ±10 % around their geometric mean).
  Reported for all estimates and for the models only (historical, GARCH, LSTM).
- *Bands* (fixed a priori): Low D < 0.10; Moderate 0.10 ≤ D < 0.25; High D ≥ 0.25.
- *Implied-to-forecast ratio* IV / mean(GARCH, LSTM), computed only when actual option-chain
  IV is available.
- *Context:* percentile of today's model-only dispersion within its own history of
  out-of-sample forecasts (the test window). Implied volatility has no single-stock history
  in this data source, so it has no percentile.
- *Interpretation:* disagreement reflects different information sets (backward-looking data
  vs forward-looking, risk-neutral prices) and model assumptions — e.g. GARCH reacting faster
  than a 60-day window to a recent volatility jump. It is not a buy/sell signal.

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

---

## 12. PDF risk report

Implemented in `reporting/` (`pdf_generator.py`, `report_charts.py`) from a `ReportData` object
assembled by `app_core.pipeline.assemble_report` — the same results the app displays; no static
content. Fifteen sections: the fourteen requested plus a dedicated **Model limitations and
data quality** section built from this run (data-quality issues, series provenance with
download times and first/last observations, option-quote exclusions by reason, and model
limitations actually observed, e.g. withheld Cornish–Fisher estimates, LSTM interval
under-coverage, early stopping, max-Sharpe fallbacks, proxy usage).

The cover lists the generation time and every observation date (prices, risk-free rate, VIX,
option quotes, forecasts). Every figure carries its basis label; pages are numbered "Page x of
y". If forecasts or option data are not included, the affected sections say so explicitly
rather than being omitted silently. Fonts: DejaVu Sans (bundled with matplotlib) for full
Greek and mathematical symbol coverage. Tests (`tests/test_report.py`) generate a report from
synthetic data and read it back to check pagination, section order, actual figures and dates,
basis labels and the proxy distinction.
