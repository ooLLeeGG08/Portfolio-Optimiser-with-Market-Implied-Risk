"""Currency conversion of price series into the base currency. See METHODOLOGY §2.

Prices are converted *unhedged*: P_base,t = P_local,t × X_t, where X_t is the number of
base-currency units per unit of local currency on date t. The portfolio's returns, volatility
and VaR therefore include exchange-rate risk, as they would for an investor who holds the
foreign asset and measures wealth in the base currency.

* Minor units are normalised first (London stocks quoted in pence "GBp"/"GBX" are divided by
  100; likewise South African cents "ZAc" and Israeli agorot "ILA").
* X_t comes from the provider's daily FX series. Dates where the asset has a price but no FX
  rate are dropped (never filled) and counted.
* FX "closes" are not taken at the same moment as each exchange's close; this timing mismatch
  adds a little noise to daily returns and is documented as a limitation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Provider currency code -> (ISO currency, divisor to convert the quote to that currency)
MINOR_UNITS: dict[str, tuple[str, float]] = {
    "GBp": ("GBP", 100.0), "GBX": ("GBP", 100.0), "GBx": ("GBP", 100.0),
    "ZAc": ("ZAR", 100.0), "ZAC": ("ZAR", 100.0),
    "ILA": ("ILS", 100.0),
}


def normalize_currency(code: str) -> tuple[str, float]:
    """('GBp') -> ('GBP', 100.0); ('jpy') -> ('JPY', 1.0)."""
    if code in MINOR_UNITS:
        return MINOR_UNITS[code]
    return code.upper(), 1.0


def fx_candidates(currency: str, base: str) -> list[tuple[str, bool]]:
    """Provider tickers that give base-per-local, as (ticker, invert). The base-quoted pair
    (e.g. USDJPY=X, local units per base unit) is preferred and inverted, because it carries
    more significant digits than the tiny reciprocal quote (e.g. JPYUSD=X ≈ 0.0063)."""
    return [(f"{base}{currency}=X", True), (f"{currency}{base}=X", False)]


@dataclass(frozen=True)
class Conversion:
    series: pd.Series  # converted prices in the base currency
    currency: str  # provider's code, e.g. "GBp"
    iso: str  # normalised ISO code, e.g. "GBP"
    divisor: float  # minor-unit divisor applied
    fx_ticker: str | None  # FX series used (None if only a minor-unit adjustment)
    dates_dropped: int  # asset dates without an FX rate


def convert_series(prices: pd.Series, currency: str, base: str,
                   fx_base_per_local: pd.Series | None = None,
                   fx_ticker: str | None = None) -> Conversion:
    """Convert ``prices`` quoted in ``currency`` into ``base``. ``fx_base_per_local`` is
    required when the (normalised) currency differs from the base."""
    iso, divisor = normalize_currency(currency)
    local = prices / divisor
    if iso == base:
        return Conversion(local.rename(prices.name), currency, iso, divisor, None, 0)
    if fx_base_per_local is None or fx_base_per_local.dropna().empty:
        raise ValueError(f"No {iso}->{base} exchange rate available.")
    fx = fx_base_per_local.dropna()
    if (fx <= 0).any():
        raise ValueError(f"Exchange-rate series {fx_ticker} contains non-positive values.")
    joined = pd.concat({"p": local, "x": fx}, axis=1, join="inner").dropna()
    dropped = int(local.dropna().index.difference(joined.index).size)
    converted = (joined["p"] * joined["x"]).rename(prices.name)
    return Conversion(converted, currency, iso, divisor, fx_ticker, dropped)
