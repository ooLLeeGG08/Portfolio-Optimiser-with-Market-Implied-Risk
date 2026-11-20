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

## Deploying to Vercel

The app ships as a container image (`Dockerfile.vercel`), which Vercel detects automatically. It
runs the Streamlit server on Linux with CPU-only PyTorch (`requirements-vercel.txt`). No
environment variables or API keys are needed.

**Deploy from GitHub (recommended)**
1. Push the repository to GitHub.
2. On vercel.com, choose **Add New → Project**, import the repository, leave the framework preset
   as **Other**, and click **Deploy**. Every later push to `main` redeploys automatically.

**Or with the CLI:** `vercel deploy --prod` from the project folder.

**Test the exact image locally first (optional):**
```bash
docker build --platform linux/amd64 -f Dockerfile.vercel -t portfolio-intelligence .
docker run --rm -p 8080:80 portfolio-intelligence    # open http://localhost:8080
```

**What to expect on Vercel (especially the free Hobby plan)**

| Vercel behaviour | Effect | How the app handles it |
|---|---|---|
| A connection lasts at most 5 minutes (Hobby) or up to 30 minutes (Pro) | The browser reconnects, sometimes to a fresh server | The portfolio is stored in the page URL and restored automatically. The URL is also a shareable link to that portfolio |
| Idle servers shut down after about 5 minutes | The first visit after a quiet spell takes longer while the server starts | Market data is re-downloaded and cached again |
| Results are not kept between servers | Volatility forecasts (2–4 minutes on a cloud CPU) may need to run again | Each finished asset is cached, so if a run is cut off, clicking **Run** again resumes where it stopped |
| 2 GB memory | The full analysis peaks at about 850 MB | Fits comfortably |

Yahoo Finance sometimes rate-limits requests from cloud servers. If data fails to load, the app
says so and you can retry after a short wait.

---

## What the app does

| Section | Contents |
|---|---|
| **Dashboard** | Portfolio value, return, volatility, Sharpe, drawdown; VaR/CVaR; risk contributions; historical vs GARCH vs LSTM vs implied volatility |
| **Portfolio Builder** | Search by company, fund or ticker (e.g. "Toyota" → `7203.T`); weights or dollar amounts; ticker check; inputs and validation, data-quality report, composition (target vs drifted weights), performance vs benchmark, daily/monthly return statistics, correlation matrix, data provenance |
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

## The ideas behind it, in plain English

No finance or maths background needed. Each part below explains one idea the app uses, with an
everyday comparison. Example numbers come from the default portfolio (Apple, Microsoft, JPMorgan,
Exxon, Johnson & Johnson, a US government bond fund and a gold fund, equal amounts in each) at the
time of writing, November 2026. Your numbers will differ.

### 1. How has the portfolio done, and how bumpy was the ride?

- **Return** is how much the value changed. $100 that becomes $110 is a 10% return.
- **Volatility** measures how much the value jumps around from day to day. A savings account has almost
  none; a single tech stock has a lot. Two investments can end up with the same return, but the one
  that swung wildly on the way was riskier to hold. The app turns daily swings into a yearly figure.
  It multiplies by √252, not 252 (there are about 252 trading days in a year), because random ups and
  downs partly cancel out over time.
- **Sharpe ratio** is reward per unit of bumpiness: the return above what a risk-free government
  bill pays, divided by volatility. Higher is better. The default portfolio's is about 0.9.
- **Maximum drawdown** is the worst fall from a high point to the low that followed. For the default
  portfolio it was about −24%, in the COVID crash of early 2020.
- **Beta** says how strongly the portfolio follows the stock market. A beta of 0.64 means that when
  the market falls 10%, this portfolio has typically fallen about 6.4%.
- **Foreign shares** (Toyota in yen, for example) are converted into dollars every day, so currency
  swings count as part of the risk, just as they would for a US investor who owns them.

### 2. Choosing the mix (portfolio optimisation)

- **Diversification** is "don't put all your eggs in one basket", with a twist: what matters is how
  the baskets move *together*. Government bonds often rise when stocks fall, so holding both makes
  the whole portfolio calmer than either part.
- **Correlation** measures that togetherness, from +1 (always move together) to −1 (always opposite).
- The **optimiser** tries many mixes and finds:
  - the **minimum-variance** mix, the calmest one possible;
  - the **maximum-Sharpe** mix, the most return for each unit of bumpiness.
- The **efficient frontier** is the curve of best possible mixes: for each level of risk, the highest
  return. Any mix below the curve could be improved.
- **The big caveat:** the optimiser trusts past average returns, and those are very unreliable. It's
  like judging a basketball player from ten games. The app reshuffles its return estimates within
  their margin of error 200 times, and the "best" weights swing wildly, so treat optimal weights as
  a rough guide, not an answer. It also applies **shrinkage**, which tones down extreme estimates so
  the optimiser doesn't overreact to noise.

### 3. How bad could a bad day be? (VaR, CVaR and risk contribution)

- **Value at Risk (VaR)** answers: "On a bad day, how much could I lose?" A 1-day 99% VaR of 2.27%
  means: on roughly 1 day in 100, expect to lose more than 2.27% of the portfolio (about $10,600 on
  $468,000). It is a threshold, not the worst case.
- **CVaR (Expected Shortfall)** answers the follow-up question: "When those 1-in-100 days happen, how
  much do I lose on average?" Here about 3.37%. Many risk managers prefer it because it looks at the
  whole tail, not just its edge.
- The app estimates both in four ways:
  - replaying actual history;
  - assuming a smooth bell curve;
  - adjusting the bell curve for the fact that markets have more extreme days than a bell curve
    predicts ("fat tails");
  - simulating 100,000 random days.

  Seeing them side by side shows how much the answer depends on the assumptions. The adjusted bell
  curve is switched off, with an explanation, when the data has too many extreme days for it to work.
- **Risk contribution** shows who is responsible for the bumpiness. Apple is about 15% of the money
  but about 25% of the risk. The bond fund is about 14% of the money but adds almost no risk,
  because it tends to rise when stocks fall. The pieces always add up exactly to the total.

### 4. Looking ahead and stress-testing

- **History may not repeat.** The app can swap each asset's historical bumpiness for a forecast (see
  part 5) or for what the options market expects (see part 7), while keeping how the assets move
  together. Every result says which kind of number it is.
- **In a crisis, everything tends to fall together.** A slider pushes all the assets' "togetherness"
  towards 100% to show what happens when diversification stops working. For the default portfolio,
  yearly volatility rises from about 13% to about 23%.
- **Scenarios** are "what if" questions: what if the market drops 20%, interest rates jump 1 percentage
  point, or one stock falls 30%? A 20% market fall translates into about −13% for the default
  portfolio. These are labelled **hypothetical**: they are calculations, not predictions.

### 5. Forecasting how bumpy the next month will be

Calm and stormy periods in markets come in streaks, like the weather. The app forecasts the coming
month's bumpiness in three ways:

- **Historical:** "next month will be like the last 20, 60 or 252 days." Simple and often hard to beat.
- **GARCH:** a weather-forecast-style rule. Tomorrow's storminess is a mix of the long-run normal
  level, yesterday's surprise and today's storminess. After a shock, storminess fades back towards
  normal gradually. For the default portfolio, half of a shock wears off in about 75 trading days.
- **LSTM:** a small neural network that reads the pattern of the last 60 days and learns from many
  years of examples.

Both models give a **range** (a likely low, middle and high), not just one number, because a forecast
without its uncertainty can be misleading. The neural network's range is "calibrated": it is widened or
narrowed so that, on data it has not trained on, about 80% of outcomes land inside it.

### 6. Checking the forecasts honestly

- **The locked test.** The last two years of data are hidden from every model during building and
  tuning, like exam questions a student has never seen. Results on that hidden period are the honest
  ones, and nothing is changed after seeing them.
- **Scoring:** the app measures the average size of the errors, and also uses a score (QLIKE) that
  punishes *under*estimating a storm more than overestimating one, because being caught unprepared is
  worse.
- **Real difference or luck?** Forecasting next month every day means neighbouring forecasts share
  most of the same days, so two years contain only about 24 truly separate months. A statistical
  test (Diebold–Mariano) checks whether one model's lead could just be luck. The honest result here:
  neither model is clearly better. GARCH usually wins on the storm-weighted score and the neural network
  on average error, but the gap is almost always within what luck could explain, and the app says so
  instead of crowning a winner.

### 7. What the options market expects (implied volatility)

- An **option** works like an insurance policy on a stock. Insurance costs more when people expect
  storms. So from an option's price you can work backwards to how bumpy the market *expects* the
  stock to be. That figure is called **implied volatility**. It's like working out how risky a house
  is from what its flood insurance costs. The app does this calculation itself, from live option
  prices, using a standard pricing formula (Black-76), after throwing out stale or illiquid quotes.
- **The VIX** is the stock market's "fear gauge": the expected bumpiness of the S&P 500 over the next
  30 days, taken from option prices. At the time of writing it was about 28%, higher than on 90% of
  days since 1990. It describes the whole US market, not any one stock.
- **Expected bumpiness is usually higher than what actually happens** (on about 84% of days in our
  data), because people pay extra for insurance. That gap does not by itself mean options are
  overpriced.
- **Assets without options** get a clearly labelled **estimate** instead, never presented as market data.
- The **Volatility Disagreement Indicator** shows how much the four views (history, GARCH, neural
  network, options) disagree for each asset. Big disagreement means "look closer". It is **not** a
  buy or sell signal.

### 8. Testing strategies on the past (backtesting)

- A **backtest** replays history month by month, as if you only knew what was known at the time.
  Each month the strategy picks its weights using only the previous year of data, holds them, and
  repeats. It pays a small trading cost every time it buys or sells.
- **Turnover** measures how much trading that takes. A strategy that looks great but trades
  constantly can lose its edge to costs.
- As with the forecasts, the final two years are a hidden test period. The app reports results for
  both periods, warns when the test period is too short to mean much, and never claims a strategy
  will keep winning. The past does not guarantee the future, and today's asset list only includes
  companies that survived.

### What the coloured labels mean

| Label | Meaning |
|---|---|
| **Observed** | Measured from what actually happened |
| **Forecast** | A model's estimate of the future, with uncertainty |
| **Market-implied** | Read from option prices, i.e. what traders are paying for |
| **Proxy estimate** | A stand-in used when no option data exists. Not market data |
| **Hypothetical** | A "what if" stress test. Not a prediction |

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
