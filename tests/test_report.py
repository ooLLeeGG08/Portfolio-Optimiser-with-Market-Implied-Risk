"""End-to-end report test on synthetic data: the full pipeline runs against a fake data
provider, and the generated PDF is read back to check structure and labelling."""

import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader

from app_core import pipeline as pl
from core import conventions as cv
from data.cache import PriceCache
from data.market_data import MarketDataProvider, MarketDataService, OptionChainRaw
from forecasting.lstm_model import LSTMConfig
from forecasting.study import VolStudyConfig, run_volatility_study
from implied_volatility.options_data import black76_price
from reporting.pdf_generator import SECTIONS, generate_pdf

TICKERS = ("AAA", "BBB", "CCC")


def _series(n, seed, mu=0.0004, sigma=0.012, level=100.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.offsets.BDay(1), periods=n)
    r = rng.normal(mu, sigma, n)
    r[0] = 0
    return pd.Series(level * np.exp(np.cumsum(r)), index=idx)


def _chain(spot: float) -> OptionChainRaw:
    qt = datetime.now(timezone.utc).replace(hour=21, minute=0, second=0, microsecond=0)
    rows = []
    for days, vol in ((14, 0.24), (45, 0.25), (90, 0.26)):
        T = days / 365
        F = spot * np.exp(0.02 * T)
        exp = pd.Timestamp(qt.date()) + pd.Timedelta(days=days)
        for K in np.arange(spot * 0.8, spot * 1.2, spot * 0.025):
            for kind in ("call", "put"):
                p = float(black76_price(F, K, T, 0.04, vol, kind))
                rows.append({"expiry": exp, "type": kind, "strike": round(float(K), 2), "bid": p * 0.99,
                             "ask": p * 1.01, "last": p, "volume": 10, "open_interest": 500,
                             "last_trade": pd.Timestamp(qt) - pd.Timedelta(hours=1), "provider_iv": np.nan})
    return OptionChainRaw("AAA", spot, qt, pd.DataFrame(rows), "synthetic")


class FakeProvider(MarketDataProvider):
    name = "synthetic test data"
    supports_options = True

    def __init__(self):
        n = 1400
        self.data = {t: _series(n, i, sigma=0.010 + 0.004 * i) for i, t in enumerate(TICKERS)}
        self.data["SPY"] = _series(n, 9)
        self.data["^IRX"] = pd.Series(4.1, index=self.data["SPY"].index)
        self.data["^TNX"] = _series(n, 11, mu=0, sigma=0.01, level=4.0)
        self.data["^VIX"] = _series(n, 12, mu=0, sigma=0.03, level=18.0)

    def fetch_history(self, tickers):
        return {t: self.data.get(t, pd.Series(dtype=float, name=t)) for t in tickers}

    def fetch_currency(self, ticker):
        return "USD"

    def fetch_option_chain(self, ticker, max_days=400):
        return _chain(float(self.data[ticker].iloc[-1])) if ticker == "AAA" else None


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    svc = MarketDataService(FakeProvider(), PriceCache(tmp_path_factory.mktemp("cache")))
    s = pl.AnalysisSettings(tickers=TICKERS, weights=(0.5, 0.3, 0.2), benchmark="SPY", history_years=5,
                            horizon_days=10)
    mb = pl.load_market_data(s, svc)
    pa = pl.analyze_portfolio(mb)
    oa = pl.run_optimization(mb, sensitivity_draws=20)
    bb = pl.run_backtests(mb)
    tiny = VolStudyConfig(horizon=10, test_days=200, garch_dist="normal", garch_sims=50,
                          lstm=LSTMConfig(horizon=10, seq_len=20, hidden_size=8, max_epochs=3, patience=2,
                                          batch_size=128))
    studies = {t: run_volatility_study(cv.log_returns(mb.asset_prices[t]), t, tiny) for t in TICKERS}
    studies["Portfolio"] = run_volatility_study(cv.log_returns(pa.path.values), "Portfolio", tiny)
    fb = pl.forecast_bundle(10, studies)
    ib = pl.run_implied(mb, svc, fb)
    rd = pl.assemble_report(mb, pa, oa, bb, fb, ib)
    pdf = generate_pdf(rd)
    reader = PdfReader(io.BytesIO(pdf))
    text = " ".join("\n".join(p.extract_text() for p in reader.pages).replace("\xa0", " ").split())
    return rd, pdf, reader, text


def test_pdf_is_valid_and_paginated(report):
    rd, pdf, reader, text = report
    assert pdf.startswith(b"%PDF")
    n = len(reader.pages)
    assert n >= 10
    assert f"Page 1 of {n}" in text and f"Page {n} of {n}" in text
    assert reader.metadata.title == "Portfolio Risk Report"


def test_all_sections_present_in_order(report):
    _, _, _, text = report
    positions = [text.find(f"{i:02d} {t}") for i, t in enumerate(SECTIONS, 1)]
    assert all(p >= 0 for p in positions), [t for t, p in zip(SECTIONS, positions) if p < 0]
    body_start = text.find("Page 1 of")  # the cover (with the contents list) must fit on page 1
    body = [text.find(f"{i:02d} {t}", body_start) for i, t in enumerate(SECTIONS, 1)]
    assert all(p > 0 for p in body)
    assert text.find("01 Executive summary", body_start) < text.find("Page 2 of")  # section 1 starts on page 2
    assert body == sorted(body)


def test_report_uses_actual_results_and_dates(report):
    rd, _, _, text = report
    assert f"{rd.generated_at:%d %b %Y}" in text
    assert f"{rd.mb.as_of:%d %b %Y}" in text
    assert f"${rd.pa.current_value:,.0f}" in text
    h99 = next(e for e in rd.pa.tail_risk if e.method == "Historical" and e.confidence == 0.99)
    assert f"{h99.var:.2%}" in text
    for t in TICKERS:
        assert t in text
    assert "synthetic test data" in text


def test_basis_labels_and_proxy_distinction(report):
    rd, _, _, text = report
    flat = " ".join(text.split())
    for label in ("Observed (historical data)", "Forecast (model output)", "Market-implied (option prices)",
                  "Implied-vol proxy (estimate, not market data)", "Hypothetical (stress scenario, not a prediction)"):
        assert label in flat, label
    assert rd.ib.estimates["AAA"].source.value == "listed option chain"
    assert rd.ib.estimates["BBB"].source.value == "proxy estimate"
    assert "not a trading signal" in flat
    assert "S&P 500" in flat and "S&P;" not in flat  # ampersands escaped correctly in markup
    assert "not investment advice" in flat.lower()


def test_report_without_forecasts_states_omission(tmp_path):
    svc = MarketDataService(FakeProvider(), PriceCache(tmp_path))
    s = pl.AnalysisSettings(tickers=TICKERS, weights=(0.4, 0.4, 0.2), benchmark="SPY", history_years=5)
    mb = pl.load_market_data(s, svc)
    pa = pl.analyze_portfolio(mb)
    rd = pl.assemble_report(mb, pa, pl.run_optimization(mb, sensitivity_draws=10), pl.run_backtests(mb), None, None)
    text = " ".join(" ".join(p.extract_text() for p in PdfReader(io.BytesIO(generate_pdf(rd))).pages).replace("\xa0", " ").split())
    assert "Volatility forecasts were not computed for this report" in text
    assert "Market-implied data were not loaded" in text
    assert set(rd.forward) == {"Historical 60d"}
