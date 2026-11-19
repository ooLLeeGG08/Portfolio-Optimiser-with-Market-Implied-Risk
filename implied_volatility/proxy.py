"""Implied-volatility PROXY for assets without usable option data.
Basis IMPLIED_PROXY, source IVSource.PROXY — an ESTIMATE, never market data.

    sigma_proxy,i = sigma_hist,i x (VIX / sigma_hist,S&P500)

Assumption: each asset's ratio of implied to realised volatility equals the S&P 500's
current ratio. This transfers the index's volatility risk premium and current fear level to
the asset; it ignores asset-specific events (earnings, idiosyncratic risk premia) and is
only a rough stand-in. See METHODOLOGY §9.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.types import IVSource, RiskBasis
from forecasting.targets import realized_vol_backward

PROXY_WINDOW = 60
PROXY_ASSUMPTION = ("Estimate, not market data: the asset's trailing realised volatility scaled "
                    "by the S&P 500's current ratio of implied (VIX) to realised volatility. "
                    "Assumes the asset's implied/realised ratio equals the index's.")


@dataclass(frozen=True)
class ProxyIV:
    ticker: str
    value: float
    asset_realized: float
    index_realized: float
    vix: float
    ratio: float
    window: int
    as_of: pd.Timestamp
    source: IVSource = IVSource.PROXY
    basis: RiskBasis = RiskBasis.IMPLIED_PROXY
    assumption: str = PROXY_ASSUMPTION


def implied_vol_proxy(ticker: str, asset_log_returns: pd.Series, index_log_returns: pd.Series,
                      vix: pd.Series, window: int = PROXY_WINDOW) -> ProxyIV:
    """All inputs are taken at the latest date the three series share."""
    a = realized_vol_backward(asset_log_returns, window)
    i = realized_vol_backward(index_log_returns, window)
    df = pd.concat({"a": a, "i": i, "v": vix}, axis=1, join="inner").dropna()
    if df.empty:
        raise ValueError("No common date for asset, index and VIX data.")
    last = df.iloc[-1]
    ratio = float(last.v / last.i)
    return ProxyIV(ticker, float(last.a * ratio), float(last.a), float(last.i), float(last.v),
                   ratio, window, df.index[-1])
