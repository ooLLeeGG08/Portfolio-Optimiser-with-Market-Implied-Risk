"""Analysis pipeline: turns user settings into the results shown in the app and the PDF.

No Streamlit here. Each stage is a plain function that takes the outputs of earlier stages,
so the UI can cache them independently and the PDF generator can reuse them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from backtest.splits import make_oos_split
from backtest.walk_forward import (BacktestResult, backtest_metrics, equal_weight_rule,
                                   fixed_weights_rule, max_sharpe_rule, min_variance_rule,
                                   walk_forward_backtest)
from core import conventions as cv
from core.types import DataQualityReport, PriceData, RebalanceFrequency, RiskBasis
from data.market_data import DataSourceError, MarketDataService, RiskFreeRate
from data.preprocessing import correlation_matrix, monthly_returns
from forecasting.study import VolStudyConfig, VolStudyResult, run_volatility_study
from forecasting.targets import realized_vol_backward
from implied_volatility import vix as vx
from implied_volatility.comparison import (DisagreementIndicator, ImpliedEstimate,
                                           implied_estimate, implied_vol_vector,
                                           volatility_disagreement)
from implied_volatility.options_data import ChainAnalysis, analyze_chain
from implied_volatility.proxy import implied_vol_proxy
from portfolio.efficient_frontier import Frontier, efficient_frontier, random_feasible_portfolios
from portfolio.optimization import (Constraints, OptimizationInputs, OptimizedPortfolio,
                                    SensitivityResult, compare_strategies, estimate_inputs,
                                    max_sharpe_sensitivity)
from portfolio.portfolio import (PerformanceSummary, PortfolioPath, PortfolioSpec,
                                 performance_summary, simulate_portfolio)
from risk import stress_testing as st
from risk.covariance import (ForwardCovariance, VolatilityVector, build_forward_covariance,
                             cov_to_corr, correlation_stress_profile, estimate_covariance,
                             stress_correlation)
from risk.risk_contribution import RiskContribution, euler_contributions
from risk.risk_metrics import RiskSummary, Sensitivity, asset_betas, factor_sensitivities, risk_summary
from risk.tail_risk import TailRiskEstimate, covariance_tail_risk, historical_tail_risk

Progress = Callable[[float, str], None]


def _noop(_p: float, _m: str) -> None:
    pass


# =========================================================================================
# Settings
# =========================================================================================

@dataclass(frozen=True)
class AnalysisSettings:
    tickers: tuple[str, ...] = config.DEFAULT_TICKERS
    weights: tuple[float, ...] = tuple([1 / len(config.DEFAULT_TICKERS)] * len(config.DEFAULT_TICKERS))
    benchmark: str = config.DEFAULT_BENCHMARK
    horizon_days: int = 21
    risk_free_rate: float | None = None  # None -> latest 13-week T-bill
    rebalance: RebalanceFrequency = RebalanceFrequency.MONTHLY
    initial_value: float = config.DEFAULT_INITIAL_VALUE
    history_years: int = config.DEFAULT_HISTORY_YEARS
    forecast_history_years: int = 20
    cov_method: str = "ledoit_wolf"
    cost_bps: float = 10.0
    backtest_lookback: int = 252

    @property
    def weight_series(self) -> pd.Series:
        return pd.Series(self.weights, index=list(self.tickers), dtype=float)

    def start_date(self, years: int | None = None) -> pd.Timestamp:
        return pd.Timestamp.today().normalize() - pd.DateOffset(years=years or self.history_years)


# =========================================================================================
# Stage 1: market data
# =========================================================================================

@dataclass
class MarketBundle:
    settings: AnalysisSettings
    asset_prices: pd.DataFrame  # aligned, assets that passed validation
    benchmark_prices: pd.Series  # aligned with asset_prices
    price_data: PriceData  # full result incl. quality report and provenance
    risk_free: RiskFreeRate | None
    risk_free_rate: float  # value actually used
    risk_free_source: str
    vix: pd.Series | None  # decimal
    ten_year_yield: pd.Series | None  # percent
    spec: PortfolioSpec  # after removing excluded assets
    exclusion_message: str | None
    errors: list[str] = field(default_factory=list)

    @property
    def quality(self) -> DataQualityReport:
        return self.price_data.quality

    @property
    def as_of(self) -> pd.Timestamp:
        return self.asset_prices.index[-1]

    @property
    def asset_returns(self) -> pd.DataFrame:
        return cv.simple_returns(self.asset_prices)

    @property
    def benchmark_returns(self) -> pd.Series:
        return cv.simple_returns(self.benchmark_prices)


class AnalysisError(RuntimeError):
    """The analysis cannot proceed; message is user-readable."""


def load_market_data(settings: AnalysisSettings, service: MarketDataService,
                     refresh: bool = False) -> MarketBundle:
    tickers = list(dict.fromkeys(list(settings.tickers) + [settings.benchmark]))
    pdata = service.get_price_data(tickers, start=settings.start_date(), refresh=refresh)
    q = pdata.quality
    if settings.benchmark in q.excluded_tickers() or settings.benchmark not in pdata.prices:
        raise AnalysisError(f"Benchmark {settings.benchmark} has no usable price data. "
                            f"Choose another benchmark.")
    if any(i.code in ("INSUFFICIENT_COMMON_HISTORY", "NO_USABLE_ASSETS", "NO_COMMON_DATES")
           for i in q.errors):
        raise AnalysisError(" ".join(i.message for i in q.errors if i.ticker is None))

    spec = PortfolioSpec.from_weights(dict(zip(settings.tickers, settings.weights)),
                                      benchmark=settings.benchmark,
                                      horizon_days=settings.horizon_days,
                                      risk_free_rate=0.0, rebalance=settings.rebalance,
                                      initial_value=settings.initial_value)
    spec, message = spec.without(q.excluded_tickers())
    assets = [t for t in spec.tickers if t in pdata.prices]
    if not assets:
        raise AnalysisError("None of the portfolio's assets has usable price data.")

    errors: list[str] = []
    rf = None
    if settings.risk_free_rate is None:
        try:
            rf = service.latest_risk_free_rate(refresh=refresh)
        except DataSourceError as exc:
            errors.append(f"Risk-free rate unavailable: {exc}")
        if rf is None:
            rf_value, rf_source = 0.0, "unavailable — 0% used (enter a rate manually)"
            errors.append("Latest T-bill yield unavailable; a risk-free rate of 0% is used.")
        else:
            rf_value = rf.value
            rf_source = f"{rf.source}, {rf.observation_date.date()} (bond-equivalent)"
    else:
        rf_value, rf_source = settings.risk_free_rate, "entered by user"

    def series(ticker, transform=lambda s: s, full_history: bool = False):
        start = None if full_history else settings.start_date(max(settings.forecast_history_years,
                                                                   settings.history_years))
        try:
            s, _, rep = service.get_series(ticker, start=start, refresh=refresh)
            if s.empty:
                errors.append(f"{ticker} unavailable: {'; '.join(i.message for i in rep.errors)}")
                return None
            return transform(s)
        except DataSourceError as exc:
            errors.append(f"{ticker} unavailable: {exc}")
            return None

    spec = PortfolioSpec(weights=spec.weights, benchmark=spec.benchmark,
                         horizon_days=spec.horizon_days, risk_free_rate=rf_value,
                         rebalance=spec.rebalance, initial_value=spec.initial_value)
    return MarketBundle(settings, pdata.prices[assets], pdata.prices[settings.benchmark], pdata,
                        rf, rf_value, rf_source, series(config.VIX_TICKER, vx.to_decimal, full_history=True),
                        series(config.TEN_YEAR_YIELD_TICKER), spec, message, errors)


# =========================================================================================
# Stage 2: portfolio and observed risk
# =========================================================================================

@dataclass
class PortfolioAnalysis:
    path: PortfolioPath
    benchmark_values: pd.Series  # benchmark scaled to the same starting value
    performance: PerformanceSummary
    benchmark_performance: PerformanceSummary
    risk: RiskSummary
    tail_risk: list[TailRiskEstimate]
    covariance: pd.DataFrame  # annualised, historical (Ledoit-Wolf by default)
    contributions: RiskContribution
    correlation: pd.DataFrame
    monthly_returns: pd.DataFrame
    rolling_vol: pd.DataFrame  # portfolio and benchmark, 20/60/252-day realised
    betas: pd.Series
    rate_sensitivity: Sensitivity | None
    current_value: float
    current_weights: pd.Series


def analyze_portfolio(mb: MarketBundle) -> PortfolioAnalysis:
    w = mb.spec.weights
    prices = mb.asset_prices[w.index]
    path = simulate_portfolio(prices, w, mb.spec.rebalance, mb.spec.initial_value)
    bench_vals = mb.benchmark_prices / mb.benchmark_prices.iloc[0] * mb.spec.initial_value
    rets = mb.asset_returns[w.index]
    cov = estimate_covariance(rets, mb.settings.cov_method).matrix
    current_w = path.weights.iloc[-1]
    value = float(path.values.iloc[-1])

    port_log = cv.log_returns(path.values)
    bench_log = cv.log_returns(mb.benchmark_prices)
    rolling = pd.DataFrame({f"Portfolio {n}d": realized_vol_backward(port_log, n) for n in (20, 60, 252)})
    rolling[f"{mb.settings.benchmark} 60d"] = realized_vol_backward(bench_log, 60)

    rate = None
    if mb.ten_year_yield is not None:
        try:
            rate = factor_sensitivities(rets, mb.ten_year_yield.diff().dropna())
        except ValueError:
            rate = None

    return PortfolioAnalysis(
        path=path, benchmark_values=bench_vals,
        performance=performance_summary(path.values),
        benchmark_performance=performance_summary(bench_vals),
        risk=risk_summary(path.returns, path.values, mb.risk_free_rate, mb.benchmark_returns),
        tail_risk=historical_tail_risk(path.returns, rets, current_w, value,
                                       horizon_days=1),
        covariance=cov,
        contributions=euler_contributions(current_w, cov),
        correlation=correlation_matrix(rets),
        monthly_returns=monthly_returns(pd.concat({"Portfolio": path.values,
                                                   mb.settings.benchmark: bench_vals}, axis=1),
                                        as_of=pd.Timestamp.today()),
        rolling_vol=rolling,
        betas=asset_betas(rets, mb.benchmark_returns),
        rate_sensitivity=rate, current_value=value, current_weights=current_w,
    )


def tail_risk_at_horizon(mb: MarketBundle, pa: PortfolioAnalysis, horizon_days: int) -> list[TailRiskEstimate]:
    return historical_tail_risk(pa.path.returns, mb.asset_returns[pa.current_weights.index],
                                pa.current_weights, pa.current_value, horizon_days=horizon_days)


# =========================================================================================
# Stage 3: optimisation
# =========================================================================================

@dataclass
class OptimizationAnalysis:
    inputs: OptimizationInputs
    constraints: Constraints
    results: list[OptimizedPortfolio]
    errors: list[str]
    frontier: Frontier
    random_portfolios: pd.DataFrame
    sensitivity: SensitivityResult | None
    sensitivity_error: str | None


def run_optimization(mb: MarketBundle, constraints: Constraints = Constraints(),
                     custom_objective: str | None = None, custom_target: float | None = None,
                     sensitivity_draws: int = 200) -> OptimizationAnalysis:
    inp = estimate_inputs(mb.asset_returns, mb.risk_free_rate, cov_method=mb.settings.cov_method)
    results, errors = compare_strategies(inp, constraints, mb.spec.weights, custom_objective, custom_target)
    frontier = efficient_frontier(inp, Constraints(), n_points=40)
    sens, sens_err = None, None
    try:
        sens = max_sharpe_sensitivity(inp, Constraints(), sensitivity_draws, seed=config.RANDOM_SEED)
    except Exception as exc:  # e.g. max-Sharpe undefined
        sens_err = str(exc)
    return OptimizationAnalysis(inp, constraints, results, errors, frontier,
                                random_feasible_portfolios(inp, 1500, config.RANDOM_SEED),
                                sens, sens_err)


# =========================================================================================
# Stage 4: volatility forecasting
# =========================================================================================

@dataclass
class ForecastBundle:
    horizon_days: int
    studies: dict[str, VolStudyResult]  # per asset + "Portfolio"
    failures: dict[str, str]
    latest: pd.DataFrame  # rows = series, columns = model point forecasts (+ intervals)

    def vol_vector(self, model: str, tickers: list[str]) -> VolatilityVector:
        """Per-asset latest forecasts for D.R.D. model in {'Historical 60d','GARCH(1,1)','LSTM'}."""
        missing = [t for t in tickers if t not in self.studies]
        if missing:
            raise ValueError(f"No forecasts for: {', '.join(missing)}.")
        vals = pd.Series({t: float(self.studies[t].latest.loc[model, "point"]) for t in tickers})
        basis = RiskBasis.OBSERVED if model.startswith("Historical") else RiskBasis.FORECAST
        return VolatilityVector(vals, self.horizon_days, {t: basis for t in tickers},
                                f"{model} volatility, {self.horizon_days}-day horizon")


def run_forecasts(mb: MarketBundle, pa: PortfolioAnalysis, service: MarketDataService,
                  progress: Progress = _noop) -> ForecastBundle:
    h = mb.settings.horizon_days
    cfg = VolStudyConfig(horizon=h)
    studies, failures = {}, {}
    tickers = list(mb.spec.tickers)
    long_start = mb.settings.start_date(mb.settings.forecast_history_years)
    for i, t in enumerate(tickers):
        progress(i / (len(tickers) + 1), f"Forecasting {t} ({i + 1}/{len(tickers) + 1})")
        try:
            pdata = service.get_price_data([t], start=long_start)
            if pdata.prices.empty:
                raise ValueError("; ".join(x.message for x in pdata.quality.errors))
            studies[t] = run_volatility_study(cv.log_returns(pdata.prices[t]), t, cfg)
        except Exception as exc:
            failures[t] = str(exc)
    progress(len(tickers) / (len(tickers) + 1), "Forecasting the portfolio")
    try:
        studies["Portfolio"] = run_volatility_study(cv.log_returns(pa.path.values), "Portfolio", cfg)
    except Exception as exc:
        failures["Portfolio"] = str(exc)
    progress(1.0, "Done")

    return forecast_bundle(h, studies, failures)


def forecast_bundle(h: int, studies: dict[str, VolStudyResult],
                    failures: dict[str, str] | None = None) -> ForecastBundle:
    rows = {}
    for name, s in studies.items():
        L = s.latest
        rows[name] = {
            "Historical 20d": L.loc["Historical 20d", "point"],
            "Historical 60d": L.loc["Historical 60d", "point"],
            "GARCH(1,1)": L.loc["GARCH(1,1)", "point"],
            "GARCH 10%": L.loc["GARCH(1,1)", "lower"], "GARCH 90%": L.loc["GARCH(1,1)", "upper"],
            "LSTM": L.loc["LSTM", "point"],
            "LSTM 10%": L.loc["LSTM", "lower"], "LSTM 90%": L.loc["LSTM", "upper"],
            "As of": L["as_of"].iloc[0],
        }
    return ForecastBundle(h, studies, failures or {}, pd.DataFrame(rows).T)


# =========================================================================================
# Stage 5: market-implied volatility
# =========================================================================================

@dataclass
class ImpliedBundle:
    vix_summary: vx.VixSummary | None
    vix_frame: pd.DataFrame | None
    vrp: vx.VarianceRiskPremium | None
    chains: dict[str, ChainAnalysis]
    chain_errors: dict[str, str]
    estimates: dict[str, ImpliedEstimate]
    disagreement: list[DisagreementIndicator]
    notes: list[str] = field(default_factory=list)


def run_implied(mb: MarketBundle, service: MarketDataService,
                forecasts: ForecastBundle | None = None,
                progress: Progress = _noop) -> ImpliedBundle:
    notes = []
    vix_sum = vix_df = vrp = None
    spy_log = None
    if mb.vix is not None:
        vix_sum, vix_df = vx.vix_summary(mb.vix), vx.vix_frame(mb.vix)
        try:
            spx = service.get_price_data(["SPY"], start=mb.vix.index[0])
            if not spx.prices.empty:
                spy_log = cv.log_returns(spx.prices["SPY"])
                vrp = vx.variance_risk_premium(mb.vix, spy_log, mb.settings.horizon_days)
        except (DataSourceError, ValueError) as exc:
            notes.append(f"Variance risk premium unavailable: {exc}")
    else:
        notes.append("VIX data unavailable; VIX analysis and implied-vol proxies are disabled.")

    chains, chain_errors, estimates = {}, {}, {}
    asset_log = cv.log_returns(mb.asset_prices)
    tickers = list(mb.spec.tickers)
    for i, t in enumerate(tickers):
        progress(i / len(tickers), f"Option chain {t}")
        err = None
        try:
            raw = service.get_option_chain(t)
            if raw is not None:
                chains[t] = analyze_chain(raw, mb.risk_free_rate)
        except DataSourceError as exc:
            err = str(exc)
            chain_errors[t] = err
        proxy = None
        if mb.vix is not None and spy_log is not None:
            try:
                proxy = implied_vol_proxy(t, asset_log[t], spy_log, mb.vix)
            except ValueError:
                proxy = None
        estimates[t] = implied_estimate(t, mb.settings.horizon_days, chains.get(t), proxy, err)
    progress(1.0, "Done")

    vdis = compute_disagreement(mb, estimates, forecasts)
    return ImpliedBundle(vix_sum, vix_df, vrp, chains, chain_errors, estimates, vdis, notes)


def compute_disagreement(mb: MarketBundle, estimates: dict[str, ImpliedEstimate],
                         forecasts: ForecastBundle | None) -> list[DisagreementIndicator]:
    """Volatility Disagreement Indicator per asset. Without forecasts only the historical and
    implied estimates are compared."""
    asset_log = cv.log_returns(mb.asset_prices)
    out = []
    for t in mb.spec.tickers:
        hist = float(realized_vol_backward(asset_log[t], 60).iloc[-1])
        g = l = history = None
        if forecasts is not None and t in forecasts.studies:
            L = forecasts.studies[t].latest
            hist = float(L.loc["Historical 60d", "point"])
            g, l = float(L.loc["GARCH(1,1)", "point"]), float(L.loc["LSTM", "point"])
            history = forecasts.studies[t].forecasts.rename(columns={"hist_60d": "historical"})
        out.append(volatility_disagreement(t, mb.settings.horizon_days, hist, g, l,
                                           estimates.get(t), history))
    return out


# =========================================================================================
# Stage 6: forward-looking risk (D.R.D) and stress tests
# =========================================================================================

VOL_SOURCES = ("Historical 60d", "GARCH(1,1)", "LSTM", "Implied")


@dataclass
class ForwardRisk:
    source: str
    covariance: ForwardCovariance
    tail_risk: list[TailRiskEstimate]
    contributions: RiskContribution
    stress_profile: pd.DataFrame


def forward_risk(mb: MarketBundle, pa: PortfolioAnalysis, source: str,
                 forecasts: ForecastBundle | None, implied: ImpliedBundle | None,
                 correlation_lambda: float = 0.0) -> ForwardRisk:
    tickers = list(pa.current_weights.index)
    if source == "Implied":
        if implied is None:
            raise ValueError("Run the implied-volatility analysis first.")
        vv = implied_vol_vector({t: implied.estimates[t] for t in tickers}, mb.settings.horizon_days)
    elif source == "Historical 60d" and forecasts is None:
        lr = cv.log_returns(mb.asset_prices[tickers])
        vals = pd.Series({t: float(realized_vol_backward(lr[t], 60).iloc[-1]) for t in tickers})
        vv = VolatilityVector(vals, mb.settings.horizon_days,
                              {t: RiskBasis.OBSERVED for t in tickers}, "Historical 60-day volatility")
    else:
        if forecasts is None:
            raise ValueError("Run the volatility forecasts first.")
        vv = forecasts.vol_vector(source, tickers)
    _, R = cov_to_corr(pa.covariance.loc[tickers, tickers])
    R_used = stress_correlation(R, correlation_lambda) if correlation_lambda > 0 else R
    fc = build_forward_covariance(vv, R_used, "historical (Ledoit-Wolf)" if correlation_lambda == 0
                                  else f"historical, stressed towards 1 (lambda={correlation_lambda:.2f})")
    basis = fc.basis if correlation_lambda == 0 else RiskBasis.HYPOTHETICAL
    tr = covariance_tail_risk(pa.current_weights, fc.matrix, pa.current_value, basis,
                              horizon_days=1)
    return ForwardRisk(source, fc, tr, euler_contributions(pa.current_weights, fc.matrix, basis),
                       correlation_stress_profile(pa.current_weights, vv.values, R))


def default_scenarios(mb: MarketBundle, pa: PortfolioAnalysis) -> list[st.ScenarioResult]:
    w, v, cov = pa.current_weights, pa.current_value, pa.covariance
    out = [st.market_decline(w, pa.betas, -0.10, v, mb.settings.benchmark),
           st.market_decline(w, pa.betas, -0.20, v, mb.settings.benchmark)]
    if pa.rate_sensitivity is not None:
        out.append(st.rate_shock(w, pa.rate_sensitivity.coefficient, 100, v))
    out += [st.volatility_increase(w, cov, 1.5, v), st.correlation_convergence(w, cov, 0.5, v),
            st.correlation_convergence(w, cov, 0.5, v, vol_multiplier=1.5)]
    top = w.idxmax()
    out.append(st.asset_shock(w, {top: -0.30}, v))
    return out


# =========================================================================================
# Stage 7: strategy backtest
# =========================================================================================

@dataclass
class BacktestBundle:
    results: dict[str, BacktestResult]
    metrics_development: pd.DataFrame
    metrics_test: pd.DataFrame
    test_start: pd.Timestamp
    notes: list[str]


def run_backtests(mb: MarketBundle, constraints: Constraints = Constraints()) -> BacktestBundle:
    prices = mb.asset_prices[mb.spec.tickers]
    rf, lb, cost = mb.risk_free_rate, mb.settings.backtest_lookback, mb.settings.cost_bps
    rules = {"Equal weight": equal_weight_rule, "Minimum variance": min_variance_rule(constraints),
             "Maximum Sharpe": max_sharpe_rule(constraints)}
    user_is_ew = bool(np.allclose(mb.spec.weights.to_numpy(), 1 / len(mb.spec.weights)))
    if not user_is_ew:
        rules["User-defined (fixed weights)"] = fixed_weights_rule(mb.spec.weights)
    res = {n: walk_forward_backtest(prices, r, n, rf, lb, RebalanceFrequency.MONTHLY, cost)
           for n, r in rules.items()}
    first = max(r.values.index[0] for r in res.values())
    idx = res["Equal weight"].values.loc[first:].index
    split = make_oos_split(idx, min(config.TEST_WINDOW_DAYS, len(idx) // 3), purge_days=0,
                           min_dev_days=min(252, len(idx) // 3))
    dev = pd.DataFrame({n: backtest_metrics(r, rf, first, split.dev_end) for n, r in res.items()}).T
    test = pd.DataFrame({n: backtest_metrics(r, rf, split.test_start) for n, r in res.items()}).T
    notes = [f"Monthly rebalancing; weights estimated from the previous {lb} trading days only; "
             f"transaction cost {cost:g} bps of traded notional; initial allocation not charged.",
             f"Development window {first.date()} – {split.dev_end.date()}; locked test window "
             f"{split.test_start.date()} – {split.test_end.date()}.",
             "Survivorship bias: the asset list was chosen today, from assets that still exist.",
             "Past results do not indicate future performance; no strategy is claimed to beat the market."]
    if user_is_ew:
        notes.append("The user-defined portfolio is equal-weighted, so it is identical to the equal-weight strategy "
                     "and is not shown separately.")
    for r in res.values():
        notes += [f"{r.name}: {n}" for n in r.notes]
    return BacktestBundle(res, dev, test, split.test_start, notes)


# =========================================================================================
# Stage 8: report data
# =========================================================================================

@dataclass
class ReportData:
    mb: MarketBundle
    pa: PortfolioAnalysis
    oa: OptimizationAnalysis
    bb: BacktestBundle
    scenarios: list[st.ScenarioResult]
    forward: dict[str, ForwardRisk]  # available volatility sources
    forward_errors: dict[str, str]
    fb: ForecastBundle | None
    ib: ImpliedBundle | None
    generated_at: pd.Timestamp  # UTC


def assemble_report(mb: MarketBundle, pa: PortfolioAnalysis, oa: OptimizationAnalysis,
                    bb: BacktestBundle, fb: ForecastBundle | None,
                    ib: ImpliedBundle | None) -> ReportData:
    forward, errors = {}, {}
    for src in VOL_SOURCES:
        if src in ("GARCH(1,1)", "LSTM") and fb is None:
            errors[src] = "volatility forecasts were not computed"
            continue
        if src == "Implied" and ib is None:
            errors[src] = "option data were not loaded"
            continue
        try:
            forward[src] = forward_risk(mb, pa, src, fb, ib)
        except ValueError as exc:
            errors[src] = str(exc)
    return ReportData(mb, pa, oa, bb, default_scenarios(mb, pa), forward, errors, fb, ib,
                      pd.Timestamp.now(tz="UTC"))
