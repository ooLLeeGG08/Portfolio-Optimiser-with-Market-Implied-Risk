"""Implied volatility from listed option chains (MARKET_IMPLIED, IVSource.OPTIONS_CHAIN).
See METHODOLOGY §9.

Method
------
* Implied vols are computed here from bid/ask MID prices with the Black-76 model on the
  forward F; the provider's own IV field is kept only as a cross-check column.
* Forward per expiry from put-call parity on strikes where both call and put pass the
  quote filters: F = K + e^{rT}(C - P), median over the strikes nearest spot. This avoids
  assuming a dividend yield. If no pair exists, F = S e^{rT} (dividends ignored; flagged).
* Only out-of-the-money options (puts K < F, calls K >= F) are used for ATM / term-structure
  IV: they carry the smallest early-exercise premium (listed equity options are American;
  Black-76 is a European model, an approximation stated in the notes).
* ATM IV per expiry: linear interpolation in log-moneyness k = ln(K/F) to k = 0 between the
  nearest valid OTM quotes on each side.
* Horizon IV: linear interpolation of total implied variance w(T) = sigma^2 T between the
  expiries bracketing the target maturity; sigma = sqrt(w / T). Flat extrapolation from the
  nearest expiry is flagged as such.

Nothing is filled in: quotes failing a filter are excluded and counted by reason, and an
asset without enough valid quotes is reported as unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import optimize, stats

from core import conventions as cv
from core.types import IVSource, RiskBasis
from data.market_data import OptionChainRaw

EXPIRY_HOUR_UTC = 21  # 16:00 New York ~ 21:00 UTC (20:00 in summer; error < 1 hour)


@dataclass(frozen=True)
class QuoteFilters:
    min_bid: float = 0.01
    max_relative_spread: float = 0.50  # (ask - bid) / mid
    min_open_interest: int = 10
    max_days_since_trade: int = 5  # calendar days
    min_iv: float = 0.01
    max_iv: float = 3.0
    min_days_to_expiry: int = 7  # very short expiries are excluded from ATM/term structure
    max_abs_log_moneyness: float = 0.5


# --- Black-76 ----------------------------------------------------------------------------------

def black76_price(F, K, T, r, sigma, kind: str):
    F, K, T, sigma = map(np.asarray, (F, K, T, sigma))
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd ** 2) / sd
    d2 = d1 - sd
    df = np.exp(-r * T)
    if kind == "call":
        return df * (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
    return df * (K * stats.norm.cdf(-d2) - F * stats.norm.cdf(-d1))


def implied_vol(price: float, F: float, K: float, T: float, r: float, kind: str) -> float:
    """Black-76 implied vol by Brent's method; NaN if the price is outside no-arbitrage bounds."""
    df = np.exp(-r * T)
    intrinsic = df * max(F - K, 0.0) if kind == "call" else df * max(K - F, 0.0)
    upper = df * F if kind == "call" else df * K
    if not (intrinsic + 1e-10 < price < upper) or T <= 0:
        return float("nan")
    f = lambda s: float(black76_price(F, K, T, r, s, kind)) - price
    try:
        return float(optimize.brentq(f, 1e-4, 6.0, xtol=1e-10, maxiter=200))
    except ValueError:
        return float("nan")


# --- analysis ---------------------------------------------------------------------------------

def year_fraction(expiry: pd.Timestamp, quote_time: datetime) -> float:
    exp = pd.Timestamp(expiry).tz_localize("UTC") + pd.Timedelta(hours=EXPIRY_HOUR_UTC)
    return max((exp - pd.Timestamp(quote_time)).total_seconds(), 0.0) / (365.0 * 86400)


@dataclass
class ChainAnalysis:
    ticker: str
    spot: float
    quote_time: datetime
    source: str
    risk_free_rate: float
    quotes: pd.DataFrame  # every quote with mid, T, forward, k, iv, valid, exclusion reason
    expiries: pd.DataFrame  # per expiry: T_days, forward, forward_method, atm_iv, n_valid
    exclusions: pd.Series  # count of excluded quotes by reason
    notes: list[str] = field(default_factory=list)
    basis: RiskBasis = RiskBasis.MARKET_IMPLIED
    iv_source: IVSource = IVSource.OPTIONS_CHAIN

    def smile(self, expiry: pd.Timestamp) -> pd.DataFrame:
        q = self.quotes
        return q[(q.expiry == expiry) & q.valid][["type", "strike", "k", "iv", "provider_iv",
                                                   "otm"]].sort_values("strike")

    @property
    def term_structure(self) -> pd.DataFrame:
        return self.expiries.dropna(subset=["atm_iv"])[["T_days", "atm_iv", "n_valid"]]


def _forward(q: pd.DataFrame, spot: float, T: float, r: float) -> tuple[float, str]:
    calls = q[(q.type == "call") & q.quote_ok].set_index("strike")["mid"]
    puts = q[(q.type == "put") & q.quote_ok].set_index("strike")["mid"]
    common = calls.index.intersection(puts.index)
    if len(common) == 0:
        return spot * np.exp(r * T), "spot x e^{rT} (no call/put pair; dividends ignored)"
    near = sorted(common, key=lambda k: abs(k - spot))[:3]
    fwd = [k + np.exp(r * T) * (calls[k] - puts[k]) for k in near]
    return float(np.median(fwd)), f"put-call parity ({len(near)} strike pair(s))"


def analyze_chain(raw: OptionChainRaw, risk_free_rate: float,
                  filters: QuoteFilters = QuoteFilters()) -> ChainAnalysis:
    q = raw.quotes.copy()
    qt = pd.Timestamp(raw.quote_time)
    q["mid"] = (q.bid + q.ask) / 2
    q["rel_spread"] = (q.ask - q.bid) / q["mid"].where(q["mid"] > 0)
    q["T"] = q.expiry.map(lambda e: year_fraction(e, raw.quote_time))
    q["days_since_trade"] = (qt - q.last_trade).dt.total_seconds() / 86400

    reason = pd.Series("", index=q.index, dtype=object)

    def require(ok, text):
        """Exclude quotes that do NOT satisfy ``ok``; missing data (NaN) fails the check."""
        reason[(reason == "") & ~ok.fillna(False).astype(bool)] = text

    require(q.bid >= filters.min_bid, "zero or negligible bid")
    require(q.ask > q.bid, "crossed or locked quote")
    require(q.rel_spread <= filters.max_relative_spread, "bid-ask spread too wide")
    require(q.open_interest >= filters.min_open_interest, "low open interest")
    require(q.days_since_trade <= filters.max_days_since_trade, "stale (no recent trade)")
    q["quote_ok"] = reason == ""

    rows = []
    q["forward"] = np.nan
    for exp, g in q.groupby("expiry"):
        T = float(g["T"].iloc[0])
        F, method = _forward(g, raw.spot, T, risk_free_rate)
        q.loc[g.index, "forward"] = F
        rows.append({"expiry": exp, "T": T, "T_days": T * 365, "forward": F, "forward_method": method})
    exp_df = pd.DataFrame(rows).set_index("expiry")

    q["k"] = np.log(q.strike / q.forward)
    q["otm"] = ((q.type == "put") & (q.strike < q.forward)) | ((q.type == "call") & (q.strike >= q.forward))
    require(q["T"] * 365 >= filters.min_days_to_expiry, "expiry too close")
    require(q.k.abs() <= filters.max_abs_log_moneyness, "too far from the money")

    ivs = []
    for i, row in q.iterrows():
        if reason[i]:
            ivs.append(np.nan)
            continue
        ivs.append(implied_vol(row.mid, row.forward, row.strike, row["T"], risk_free_rate, row.type))
    q["iv"] = ivs
    require(q.iv.notna(), "price outside no-arbitrage bounds")
    require((q.iv >= filters.min_iv) & (q.iv <= filters.max_iv), "implausible implied volatility")
    q["exclusion_reason"] = reason
    q["valid"] = reason == ""
    q.loc[~q.valid, "iv"] = np.nan

    atm, nval = [], []
    for exp in exp_df.index:
        g = q[(q.expiry == exp) & q.valid & q.otm]
        nval.append(int(len(g)))
        below, above = g[g.k < 0].sort_values("k"), g[g.k >= 0].sort_values("k")
        if below.empty or above.empty:
            atm.append(np.nan)
            continue
        k0, v0 = below.k.iloc[-1], below.iv.iloc[-1]
        k1, v1 = above.k.iloc[0], above.iv.iloc[0]
        atm.append(float(v1) if k1 == k0 else float(v0 + (v1 - v0) * (0 - k0) / (k1 - k0)))
    exp_df["atm_iv"] = atm
    exp_df["n_valid"] = nval

    notes = ["Implied vols computed from bid/ask mid prices with Black-76 on the put-call-parity "
             "forward; listed equity options are American, so this is an approximation "
             "(smallest for the out-of-the-money options used).",
             f"Quotes as of {pd.Timestamp(raw.quote_time):%Y-%m-%d %H:%M} UTC from {raw.source} "
             "(may be delayed); a snapshot, not a history."]
    if exp_df.forward_method.str.startswith("spot").any():
        notes.append("For some expiries no call/put pair passed the filters; their forward "
                     "assumes no dividends.")
    excl = q.loc[~q.valid, "exclusion_reason"].value_counts()
    return ChainAnalysis(raw.ticker, raw.spot, raw.quote_time, raw.source, risk_free_rate,
                         q, exp_df, excl, notes)


@dataclass(frozen=True)
class HorizonIV:
    value: float | None  # annualised implied vol for the target maturity
    target_days: float  # calendar days
    method: str  # "interpolated", "extrapolated (flat)", "unavailable"
    expiries_used: tuple
    source: IVSource
    basis: RiskBasis | None
    reliable: bool
    reason: str


def trading_to_calendar_days(trading_days: int) -> float:
    return trading_days * 365.0 / cv.TRADING_DAYS


def horizon_implied_vol(chain: ChainAnalysis, horizon_trading_days: int,
                        min_valid_per_expiry: int = 4) -> HorizonIV:
    """ATM implied vol at the horizon by linear interpolation in total variance."""
    target = trading_to_calendar_days(horizon_trading_days)
    ts = chain.expiries.dropna(subset=["atm_iv"])
    ts = ts[ts.n_valid >= min_valid_per_expiry]
    if ts.empty:
        return HorizonIV(None, target, "unavailable", (), IVSource.UNAVAILABLE, None, False,
                         "No expiry has enough liquid out-of-the-money quotes on both sides of "
                         "the forward to measure at-the-money implied volatility.")
    ts = ts.sort_values("T_days")
    T = ts.T_days.to_numpy()
    w = ts.atm_iv.to_numpy() ** 2 * T
    if T[0] <= target <= T[-1]:
        i = int(np.searchsorted(T, target))
        if T[i] == target:
            return HorizonIV(float(ts.atm_iv.iloc[i]), target, "interpolated", (ts.index[i],),
                             IVSource.OPTIONS_CHAIN, RiskBasis.MARKET_IMPLIED, True, "")
        wt = w[i - 1] + (w[i] - w[i - 1]) * (target - T[i - 1]) / (T[i] - T[i - 1])
        return HorizonIV(float(np.sqrt(wt / target)), target, "interpolated",
                         (ts.index[i - 1], ts.index[i]), IVSource.OPTIONS_CHAIN,
                         RiskBasis.MARKET_IMPLIED, True, "")
    j = 0 if target < T[0] else len(T) - 1
    return HorizonIV(float(ts.atm_iv.iloc[j]), target, "extrapolated (flat)", (ts.index[j],),
                     IVSource.OPTIONS_CHAIN, RiskBasis.MARKET_IMPLIED, False,
                     f"No usable expiries on both sides of {target:.0f} days; the nearest expiry "
                     f"({T[j]:.0f} days) is used flat.")
