"""Horizon-aligned comparison of historical, forecast and implied volatility, and the
Volatility Disagreement Indicator. See METHODOLOGY §9.

All estimates are annualised and refer to the same horizon h (trading days):
* historical: trailing realised vol (OBSERVED) — what has happened recently;
* GARCH / LSTM: forecasts of realised vol over the next h days (FORECAST);
* implied: option-chain ATM IV interpolated to h trading days ~ h x 365/252 calendar days
  (MARKET_IMPLIED, risk-neutral), or a PROXY estimate when no usable options exist
  (IMPLIED_PROXY) — never presented as market data.

The Volatility Disagreement Indicator is an ANALYTICAL comparison, not a trading signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.types import IVSource, RiskBasis
from implied_volatility.options_data import ChainAnalysis, horizon_implied_vol
from implied_volatility.proxy import ProxyIV
from risk.covariance import VolatilityVector

VDI_DISCLAIMER = ("Analytical comparison only — not a trading signal. Disagreement reflects "
                  "differences in model assumptions and information sets (backward-looking "
                  "data vs risk-neutral option prices), not mispricing.")
VDI_BANDS = ((0.10, "Low"), (0.25, "Moderate"), (np.inf, "High"))


@dataclass(frozen=True)
class ImpliedEstimate:
    ticker: str
    value: float | None
    source: IVSource
    basis: RiskBasis | None
    horizon_days: int
    detail: str
    reliable: bool


def implied_estimate(ticker: str, horizon_days: int, chain: ChainAnalysis | None,
                     proxy: ProxyIV | None, chain_error: str | None = None) -> ImpliedEstimate:
    """Prefer actual option-chain IV; fall back to the clearly labelled proxy; otherwise
    report unavailable. The reason for any fallback is recorded."""
    why = chain_error or "no listed options returned by the data provider"
    if chain is not None:
        hiv = horizon_implied_vol(chain, horizon_days)
        if hiv.value is not None:
            detail = (f"ATM implied vol {hiv.method} to {hiv.target_days:.0f} calendar days "
                      f"from expiries {', '.join(str(pd.Timestamp(e).date()) for e in hiv.expiries_used)}; "
                      f"quotes as of {pd.Timestamp(chain.quote_time):%Y-%m-%d %H:%M} UTC.")
            if not hiv.reliable:
                detail += " " + hiv.reason
            return ImpliedEstimate(ticker, hiv.value, IVSource.OPTIONS_CHAIN,
                                   RiskBasis.MARKET_IMPLIED, horizon_days, detail, hiv.reliable)
        why = hiv.reason
    if proxy is not None:
        return ImpliedEstimate(ticker, proxy.value, IVSource.PROXY, RiskBasis.IMPLIED_PROXY,
                               horizon_days, f"Option-implied vol unavailable ({why}). "
                               f"{proxy.assumption} Ratio VIX/S&P realised = {proxy.ratio:.2f} "
                               f"as of {proxy.as_of.date()}.", False)
    return ImpliedEstimate(ticker, None, IVSource.UNAVAILABLE, None, horizon_days,
                           f"Implied volatility unavailable ({why}); no proxy could be formed.", False)


def implied_vol_vector(estimates: dict[str, ImpliedEstimate], horizon_days: int) -> VolatilityVector:
    """Per-asset implied (or proxy) vols for Sigma = D R D. Raises if any asset has none."""
    missing = [t for t, e in estimates.items() if e.value is None]
    if missing:
        raise ValueError(f"No implied or proxy volatility for: {', '.join(missing)}.")
    if any(e.horizon_days != horizon_days for e in estimates.values()):
        raise ValueError("Implied estimates refer to a different horizon.")
    n_proxy = sum(e.source is IVSource.PROXY for e in estimates.values())
    desc = (f"Implied volatility ({len(estimates) - n_proxy} from option chains, "
            f"{n_proxy} proxy estimates)")
    return VolatilityVector(pd.Series({t: e.value for t, e in estimates.items()}), horizon_days,
                            {t: e.basis for t, e in estimates.items()}, desc)


# --- Volatility Disagreement Indicator ---------------------------------------------------------

def log_dispersion(values) -> float:
    """Population standard deviation of log volatilities (scale-free; 0 = full agreement).
    0.10 means the estimates are spread roughly +/-10 % around their geometric mean."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x) and x > 0], dtype=float)
    return float(np.std(np.log(v))) if len(v) >= 2 else float("nan")


def disagreement_band(d: float) -> str:
    if not np.isfinite(d):
        return "Insufficient estimates"
    for limit, name in VDI_BANDS:
        if d < limit:
            return name
    return "High"


@dataclass(frozen=True)
class DisagreementIndicator:
    ticker: str
    horizon_days: int
    estimates: dict[str, float]  # label -> annualised vol
    bases: dict[str, RiskBasis]
    dispersion: float  # all available estimates
    dispersion_models: float  # historical, GARCH, LSTM only
    band: str
    implied_to_forecast_ratio: float | None  # only with ACTUAL market-implied vol
    models_percentile: float | None  # current model dispersion vs its own recent history
    history_window: tuple[pd.Timestamp, pd.Timestamp] | None
    notes: tuple[str, ...] = field(default_factory=tuple)
    disclaimer: str = VDI_DISCLAIMER


def volatility_disagreement(ticker: str, horizon_days: int, historical: float, garch: float | None,
                            lstm: float | None, implied: ImpliedEstimate | None,
                            model_history: pd.DataFrame | None = None) -> DisagreementIndicator:
    """``model_history``: past forecasts with columns historical/garch/lstm (e.g. the
    out-of-sample test-window forecasts) used to place today's model disagreement in context.
    Implied vol has no single-stock history in this data source, so only model-only
    dispersion gets a percentile."""
    est = {"Historical": historical}
    bases = {"Historical": RiskBasis.OBSERVED}
    for name, v in (("GARCH", garch), ("LSTM", lstm)):
        if v is not None:
            est[name], bases[name] = v, RiskBasis.FORECAST
    notes = []
    ratio = None
    if implied is not None and implied.value is not None:
        label = "Implied (options)" if implied.source is IVSource.OPTIONS_CHAIN else "Implied proxy (estimate)"
        est[label], bases[label] = implied.value, implied.basis
        fc = [v for v in (garch, lstm) if v is not None]
        if implied.source is IVSource.OPTIONS_CHAIN and fc:
            ratio = implied.value / float(np.mean(fc))
            notes.append("Implied volatility is risk-neutral and usually exceeds subsequently "
                         "realised volatility (variance risk premium); a ratio above 1 is normal "
                         "and is not evidence that options are overpriced.")
        if implied.source is IVSource.PROXY:
            notes.append("The implied figure is a proxy estimate built from historical volatility "
                         "and VIX, so it is not independent of the historical estimate.")
    d_all = log_dispersion(est.values())
    d_models = log_dispersion([historical, garch, lstm])

    pct, window = None, None
    if model_history is not None and len(model_history) >= 20:
        hist_d = model_history[["historical", "garch", "lstm"]].apply(
            lambda r: log_dispersion(r.to_numpy()), axis=1).dropna()
        if len(hist_d) and np.isfinite(d_models):
            pct = float((hist_d <= d_models).mean())
            window = (hist_d.index[0], hist_d.index[-1])
    return DisagreementIndicator(ticker, horizon_days, est, bases, d_all, d_models,
                                 disagreement_band(d_all), ratio, pct, window, tuple(notes))


def disagreement_table(indicators: list[DisagreementIndicator]) -> pd.DataFrame:
    rows = []
    for d in indicators:
        row = {"Ticker": d.ticker, **{k: v for k, v in d.estimates.items()},
               "Dispersion (all)": d.dispersion, "Dispersion (models)": d.dispersion_models,
               "Disagreement": d.band, "IV / mean forecast": d.implied_to_forecast_ratio,
               "Model dispersion percentile": d.models_percentile}
        rows.append(row)
    return pd.DataFrame(rows).set_index("Ticker")


def index_implied_vs_forecast(vix: pd.Series, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Align VIX (market-implied, S&P 500) with S&P-tracker forecasts and the subsequently
    realised volatility over the forecast window (e.g. the out-of-sample test window)."""
    cols = {"realized": "Realised (ex post)", "hist_60d": "Historical 60d", "garch": "GARCH",
            "lstm": "LSTM"}
    df = forecasts[[c for c in cols if c in forecasts]].rename(columns=cols)
    return df.join(vix.rename("VIX (implied)"), how="inner")
