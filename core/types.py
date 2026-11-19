"""Shared types. Every risk/volatility figure carries a RiskBasis so the UI and PDF can
label observed, forecast, market-implied, proxy and hypothetical numbers distinctly."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pandas as pd


class RiskBasis(str, Enum):
    OBSERVED = "observed"
    FORECAST = "forecast"
    MARKET_IMPLIED = "market-implied"
    IMPLIED_PROXY = "implied proxy (estimate)"
    HYPOTHETICAL = "hypothetical"

    @classmethod
    def coerce(cls, value) -> "RiskBasis | None":
        """Recover a RiskBasis from an enum, its value or its name (pandas may store enums as text)."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            for r in cls:
                if value in (r.value, r.name, f"RiskBasis.{r.name}", r.label):
                    return r
        return None

    @property
    def label(self) -> str:
        return {
            RiskBasis.OBSERVED: "Observed (historical data)",
            RiskBasis.FORECAST: "Forecast (model output)",
            RiskBasis.MARKET_IMPLIED: "Market-implied (option prices)",
            RiskBasis.IMPLIED_PROXY: "Implied-vol proxy (estimate, not market data)",
            RiskBasis.HYPOTHETICAL: "Hypothetical (stress scenario, not a prediction)",
        }[self]


class IVSource(str, Enum):
    """Where an implied-volatility number came from."""

    VIX_INDEX = "VIX index"  # actual, S&P 500 only
    OPTIONS_CHAIN = "listed option chain"  # actual, asset-specific
    PROXY = "proxy estimate"  # NOT market data
    UNAVAILABLE = "unavailable"

    @property
    def is_market_data(self) -> bool:
        return self in (IVSource.VIX_INDEX, IVSource.OPTIONS_CHAIN)

    @property
    def basis(self) -> RiskBasis | None:
        if self.is_market_data:
            return RiskBasis.MARKET_IMPLIED
        if self is IVSource.PROXY:
            return RiskBasis.IMPLIED_PROXY
        return None


class Severity(str, Enum):
    ERROR = "error"  # asset excluded / input rejected
    WARNING = "warning"  # kept, but the user should know
    INFO = "info"


class RebalanceFrequency(str, Enum):
    NONE = "none"  # buy-and-hold
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


@dataclass(frozen=True)
class DataIssue:
    ticker: str | None
    severity: Severity
    code: str
    message: str


@dataclass
class DataQualityReport:
    issues: list[DataIssue] = field(default_factory=list)
    rows_dropped_in_alignment: int = 0
    aligned_start: pd.Timestamp | None = None
    aligned_end: pd.Timestamp | None = None
    n_aligned_observations: int = 0

    def add(self, ticker: str | None, severity: Severity, code: str, message: str) -> None:
        self.issues.append(DataIssue(ticker, severity, code, message))

    @property
    def errors(self) -> list[DataIssue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[DataIssue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    def excluded_tickers(self) -> list[str]:
        return sorted({i.ticker for i in self.errors if i.ticker is not None})

    def extend(self, other: "DataQualityReport") -> None:
        self.issues.extend(other.issues)


@dataclass(frozen=True)
class SeriesProvenance:
    """Where and when a series was obtained; shown in the app and the PDF."""

    ticker: str
    source: str
    fetched_at: datetime
    first_date: pd.Timestamp | None
    last_date: pd.Timestamp | None
    from_cache: bool


@dataclass
class PriceData:
    """Adjusted close prices (columns = tickers) plus provenance and quality report."""

    prices: pd.DataFrame
    provenance: dict[str, SeriesProvenance]
    quality: DataQualityReport

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)
