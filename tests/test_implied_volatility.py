from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from core.types import IVSource, RiskBasis
from data.market_data import OptionChainRaw
from implied_volatility import vix as vx
from implied_volatility.comparison import (disagreement_band, implied_estimate,
                                           implied_vol_vector, log_dispersion,
                                           volatility_disagreement)
from implied_volatility.options_data import (analyze_chain, black76_price, horizon_implied_vol,
                                             implied_vol, trading_to_calendar_days)
from implied_volatility.proxy import implied_vol_proxy
from risk.covariance import build_forward_covariance

QT = datetime(2026, 1, 5, 21, 0, tzinfo=timezone.utc)  # expiry hour => T = days / 365 exactly
R, Q, SPOT = 0.04, 0.02, 100.0


def make_chain(vol_by_days: dict[int, float], strikes=np.arange(80, 121, 2.5),
               extra_rows: list[dict] | None = None) -> OptionChainRaw:
    rows = []
    for days, vol in vol_by_days.items():
        T = days / 365
        F = SPOT * np.exp((R - Q) * T)  # dividend-paying underlying
        exp = pd.Timestamp(QT.date()) + pd.Timedelta(days=days)
        for K in strikes:
            for kind in ("call", "put"):
                p = float(black76_price(F, K, T, R, vol, kind))
                rows.append({"expiry": exp, "type": kind, "strike": float(K),
                             "bid": p * 0.99, "ask": p * 1.01, "last": p, "volume": 50,
                             "open_interest": 500, "last_trade": pd.Timestamp(QT) - pd.Timedelta(hours=2),
                             "provider_iv": np.nan})
    for r in extra_rows or []:
        rows.append(r)
    return OptionChainRaw("XYZ", SPOT, QT, pd.DataFrame(rows), "synthetic")


# --- pricing / inversion ---------------------------------------------------------------------

def test_black76_parity_and_iv_inversion():
    F, K, T = 102.0, 95.0, 0.5
    c = black76_price(F, K, T, R, 0.3, "call")
    p = black76_price(F, K, T, R, 0.3, "put")
    assert c - p == pytest.approx(np.exp(-R * T) * (F - K))
    assert implied_vol(float(c), F, K, T, R, "call") == pytest.approx(0.3, abs=1e-8)
    assert implied_vol(float(p), F, K, T, R, "put") == pytest.approx(0.3, abs=1e-8)
    assert np.isnan(implied_vol(0.0001, F, K, T, R, "call"))  # below intrinsic


# --- chain analysis ---------------------------------------------------------------------------

def test_flat_vol_chain_recovers_forward_and_atm_iv():
    ch = analyze_chain(make_chain({14: 0.25, 45: 0.25, 90: 0.25}), R)
    for exp, row in ch.expiries.iterrows():
        assert row.forward == pytest.approx(SPOT * np.exp((R - Q) * row["T"]), rel=1e-6)
        assert row.forward_method.startswith("put-call parity")
        assert row.atm_iv == pytest.approx(0.25, abs=0.002)
    hiv = horizon_implied_vol(ch, 21)
    assert hiv.method == "interpolated" and hiv.reliable
    assert hiv.value == pytest.approx(0.25, abs=0.002)
    assert hiv.source is IVSource.OPTIONS_CHAIN and hiv.basis is RiskBasis.MARKET_IMPLIED


def test_term_structure_interpolates_total_variance():
    ch = analyze_chain(make_chain({14: 0.30, 60: 0.20}), R)
    target = trading_to_calendar_days(21)
    w = 0.30 ** 2 * 14 + (0.20 ** 2 * 60 - 0.30 ** 2 * 14) * (target - 14) / (60 - 14)
    assert horizon_implied_vol(ch, 21).value == pytest.approx(np.sqrt(w / target), abs=0.002)


def test_filters_exclude_bad_quotes_with_reasons():
    exp = pd.Timestamp(QT.date()) + pd.Timedelta(days=45)
    base = {"expiry": exp, "type": "call", "volume": 1, "last": 1.0, "provider_iv": np.nan,
            "last_trade": pd.Timestamp(QT)}
    bad = [
        {**base, "strike": 131.0, "bid": 0.0, "ask": 0.05, "open_interest": 100},
        {**base, "strike": 132.0, "bid": 0.5, "ask": 2.0, "open_interest": 100},
        {**base, "strike": 133.0, "bid": 1.0, "ask": 1.1, "open_interest": np.nan},
        {**base, "strike": 134.0, "bid": 1.0, "ask": 1.1, "open_interest": 100,
         "last_trade": pd.Timestamp(QT) - pd.Timedelta(days=20)},
        {**base, "strike": 135.0, "bid": 1.0, "ask": 1.1, "open_interest": 100, "last_trade": pd.NaT},
    ]
    ch = analyze_chain(make_chain({45: 0.25}, extra_rows=bad), R)
    q = ch.quotes.set_index("strike")
    assert q.loc[131.0, "exclusion_reason"] == "zero or negligible bid"
    assert q.loc[132.0, "exclusion_reason"] == "bid-ask spread too wide"
    assert q.loc[133.0, "exclusion_reason"] == "low open interest"  # missing OI fails
    assert q.loc[134.0, "exclusion_reason"] == "stale (no recent trade)"
    assert q.loc[135.0, "exclusion_reason"] == "stale (no recent trade)"  # never traded
    assert not q.loc[[131.0, 132.0, 133.0, 134.0, 135.0], "valid"].any()
    assert q.loc[[131.0, 132.0, 133.0, 134.0, 135.0], "iv"].isna().all()
    assert ch.exclusions.sum() >= 5


def test_short_expiry_excluded_and_flat_extrapolation_flagged():
    ch = analyze_chain(make_chain({3: 0.4, 90: 0.22, 120: 0.22}), R)
    assert np.isnan(ch.expiries.iloc[0].atm_iv)  # 3-day expiry filtered out
    hiv = horizon_implied_vol(ch, 21)
    assert hiv.method == "extrapolated (flat)" and not hiv.reliable
    assert hiv.value == pytest.approx(0.22, abs=0.002)


def test_unusable_chain_reports_unavailable():
    raw = make_chain({45: 0.25})
    raw.quotes["bid"] = 0.0
    hiv = horizon_implied_vol(analyze_chain(raw, R), 21)
    assert hiv.value is None and hiv.source is IVSource.UNAVAILABLE and hiv.basis is None


# --- VIX ---------------------------------------------------------------------------------

def test_vix_summary_and_percentile():
    idx = pd.bdate_range("2015-01-01", periods=600)
    v = pd.Series(np.linspace(0.10, 0.40, 600), index=idx)
    s = vx.vix_summary(v)
    assert s.latest == pytest.approx(0.40) and s.percentile_full == pytest.approx(1.0)
    assert s.change_1d == pytest.approx(v.iloc[-1] - v.iloc[-2])
    assert s.mean_20d == pytest.approx(v.iloc[-20:].mean())
    assert s.basis is RiskBasis.MARKET_IMPLIED and s.source is IVSource.VIX_INDEX
    assert "not a forecast of any individual stock" in s.description
    assert vx.percentile_rank(pd.Series([1, 2, 3, 4]), 2) == pytest.approx(0.5)
    assert vx.to_decimal(pd.Series([18.5])).iloc[0] == pytest.approx(0.185)


def test_variance_risk_premium_hand_calculation():
    idx = pd.bdate_range("2020-01-01", periods=200)
    r = pd.Series(0.01 * np.where(np.arange(200) % 2, 1, -1), index=idx)  # |r| = 1% daily
    vix = pd.Series(0.20, index=idx)
    vrp = vx.variance_risk_premium(vix, r, 21)
    assert vrp.mean_spread == pytest.approx(0.20 - 0.01 * np.sqrt(252))
    assert vrp.share_vix_above == pytest.approx(1.0)


# --- proxy and labelling --------------------------------------------------------------------

def test_proxy_formula_and_labels():
    idx = pd.bdate_range("2020-01-01", periods=100)
    asset = pd.Series(0.02 * np.where(np.arange(100) % 2, 1, -1), index=idx)
    index = pd.Series(0.01 * np.where(np.arange(100) % 2, 1, -1), index=idx)
    vix = pd.Series(0.24, index=idx)
    p = implied_vol_proxy("XYZ", asset, index, vix, window=60)
    assert p.ratio == pytest.approx(0.24 / (0.01 * np.sqrt(252)))
    assert p.value == pytest.approx(0.02 * np.sqrt(252) * p.ratio)
    assert p.basis is RiskBasis.IMPLIED_PROXY and p.source is IVSource.PROXY
    assert not p.source.is_market_data


def test_implied_estimate_prefers_options_then_proxy_then_unavailable():
    idx = pd.bdate_range("2020-01-01", periods=100)
    r = pd.Series(0.01 * np.where(np.arange(100) % 2, 1, -1), index=idx)
    proxy = implied_vol_proxy("XYZ", r, r, pd.Series(0.2, index=idx))
    good = analyze_chain(make_chain({14: 0.25, 60: 0.25}), R)
    e1 = implied_estimate("XYZ", 21, good, proxy)
    assert e1.source is IVSource.OPTIONS_CHAIN and e1.basis is RiskBasis.MARKET_IMPLIED
    raw = make_chain({45: 0.25}); raw.quotes["bid"] = 0.0
    e2 = implied_estimate("XYZ", 21, analyze_chain(raw, R), proxy)
    assert e2.source is IVSource.PROXY and e2.basis is RiskBasis.IMPLIED_PROXY
    assert "Estimate, not market data" in e2.detail and not e2.reliable
    e3 = implied_estimate("XYZ", 21, None, None)
    assert e3.value is None and e3.source is IVSource.UNAVAILABLE


def test_implied_vector_with_any_proxy_is_labelled_proxy_in_drd():
    idx = pd.bdate_range("2020-01-01", periods=100)
    r = pd.Series(0.01 * np.where(np.arange(100) % 2, 1, -1), index=idx)
    proxy = implied_vol_proxy("B", r, r, pd.Series(0.2, index=idx))
    est = {"A": implied_estimate("A", 21, analyze_chain(make_chain({14: .25, 60: .25}), R), None),
           "B": implied_estimate("B", 21, None, proxy)}
    vv = implied_vol_vector(est, 21)
    assert "1 from option chains, 1 proxy" in vv.description
    corr = pd.DataFrame([[1, 0.3], [0.3, 1]], index=["A", "B"], columns=["A", "B"])
    assert build_forward_covariance(vv, corr).basis is RiskBasis.IMPLIED_PROXY
    with pytest.raises(ValueError, match="No implied"):
        implied_vol_vector({"C": implied_estimate("C", 21, None, None)}, 21)


# --- Volatility Disagreement Indicator -----------------------------------------------------

def test_log_dispersion_and_bands():
    assert log_dispersion([0.2, 0.2, 0.2]) == pytest.approx(0.0)
    assert log_dispersion([0.2, 0.2 * np.e]) == pytest.approx(0.5)
    assert np.isnan(log_dispersion([0.2]))
    assert disagreement_band(0.05) == "Low"
    assert disagreement_band(0.2) == "Moderate"
    assert disagreement_band(0.4) == "High"


def test_disagreement_indicator_rules():
    chain_est = implied_estimate("A", 21, analyze_chain(make_chain({14: .30, 60: .30}), R), None)
    hist = pd.DataFrame({"historical": [0.2] * 30, "garch": [0.2] * 15 + [0.3] * 15,
                         "lstm": [0.2] * 30}, index=pd.bdate_range("2024-01-01", periods=30))
    d = volatility_disagreement("A", 21, 0.20, 0.22, 0.21, chain_est, hist)
    assert set(d.estimates) == {"Historical", "GARCH", "LSTM", "Implied (options)"}
    assert d.implied_to_forecast_ratio == pytest.approx(chain_est.value / 0.215)
    assert d.models_percentile is not None and 0 <= d.models_percentile <= 1
    assert "not a trading signal" in d.disclaimer
    assert any("not evidence that options are overpriced" in n for n in d.notes)
    # With a proxy, no implied/forecast ratio is reported (it is not market data).
    idx = pd.bdate_range("2020-01-01", periods=100)
    r = pd.Series(0.01 * np.where(np.arange(100) % 2, 1, -1), index=idx)
    proxy_est = implied_estimate("B", 21, None, implied_vol_proxy("B", r, r, pd.Series(0.2, index=idx)))
    d2 = volatility_disagreement("B", 21, 0.20, 0.22, 0.21, proxy_est)
    assert "Implied proxy (estimate)" in d2.estimates and d2.implied_to_forecast_ratio is None
    assert d2.bases["Implied proxy (estimate)"] is RiskBasis.IMPLIED_PROXY
