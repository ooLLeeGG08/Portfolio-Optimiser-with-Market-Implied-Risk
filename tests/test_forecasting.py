import numpy as np
import pandas as pd
import pytest
import torch
from arch import arch_model

from backtest.splits import TestDataLeakageError, assert_development_only, make_oos_split
from forecasting import evaluation as ev
from forecasting import garch_model as gm
from forecasting.historical_volatility import historical_forecast, rolling_volatility
from forecasting.lstm_model import (LSTMConfig, build_features, conformal_adjustment,
                                    pinball_loss, prepare_data, train_lstm)
from forecasting.study import VolStudyConfig, run_volatility_study
from forecasting.targets import realized_vol_backward, realized_vol_forward


def garch_series(n=3000, params=(0.03, 0.02, 0.08, 0.90), dist="normal", seed=0) -> pd.Series:
    """Daily LOG returns (decimal) simulated from a GARCH(1,1) in percent units."""
    np.random.seed(seed)  # arch's simulate() draws from numpy's global RNG
    m = arch_model(None, mean="Constant", vol="GARCH", dist=dist)
    p = list(params) + ([6.0] if dist == "t" else [])
    sim = m.simulate(p, n, burn=1000)
    return pd.Series(sim["data"].to_numpy() / 100.0, index=pd.bdate_range("2010-01-01", periods=n))


@pytest.fixture(scope="module")
def series():
    return garch_series()


# --- target definition ------------------------------------------------------------------------

def test_realized_vol_hand_calculation_and_forward_backward_identity():
    r = pd.Series([0.01, -0.02, 0.03, 0.0, -0.01], index=pd.bdate_range("2024-01-01", periods=5))
    fwd = realized_vol_forward(r, 2)
    assert fwd.iloc[0] == pytest.approx(np.sqrt(252 / 2 * (0.02 ** 2 + 0.03 ** 2)))
    assert fwd.iloc[-2:].isna().all()  # future unknown
    back = realized_vol_backward(r, 2)
    assert back.iloc[1] == pytest.approx(np.sqrt(252 / 2 * (0.01 ** 2 + 0.02 ** 2)))
    # The target at t is exactly the trailing realised vol at t+h.
    np.testing.assert_allclose(fwd.iloc[:-2].to_numpy(), back.iloc[2:].to_numpy())


def test_historical_forecast_uses_same_definition(series):
    pd.testing.assert_series_equal(historical_forecast(series, 60),
                                   realized_vol_backward(series, 60).rename("hist_60d"))
    assert list(rolling_volatility(series).columns) == ["hist_20d", "hist_60d", "hist_252d"]


# --- locked split --------------------------------------------------------------------------

def test_oos_split_and_leakage_guard(series):
    s = make_oos_split(series.index, test_days=500, purge_days=21)
    pos = series.index.get_loc
    assert pos(s.test_start) - pos(s.dev_end) == 22  # 21 purged dates in between
    assert len(s.test(series)) == 500
    assert_development_only(s.development(series), s)
    with pytest.raises(TestDataLeakageError, match="locked"):
        assert_development_only(series, s)
    with pytest.raises(ValueError, match="development observations"):
        make_oos_split(series.index[:600], test_days=500)


# --- GARCH ---------------------------------------------------------------------------------

def test_garch_recovers_parameters(series):
    fit = gm.fit_garch(series)
    assert fit.params["alpha[1]"] == pytest.approx(0.08, abs=0.03)
    assert fit.params["beta[1]"] == pytest.approx(0.90, abs=0.04)
    assert fit.persistence == pytest.approx(fit.params["alpha[1]"] + fit.params["beta[1]"])
    assert fit.half_life_days == pytest.approx(np.log(0.5) / np.log(fit.persistence))
    true_uncond = np.sqrt(0.02 / (1 - 0.98)) / 100 * np.sqrt(252)
    assert fit.unconditional_vol == pytest.approx(true_uncond, rel=0.25)
    # A correctly specified model leaves little autocorrelation in squared std. residuals.
    assert fit.ljung_box.loc["squared standardised residuals (lag 10)", "lb_pvalue"] > 0.01


def test_garch_horizon_vol_constant_variance():
    v = np.full((1, 21), 1.0)  # 1%^2 daily variance each step
    assert gm._horizon_vol(v, 21)[0] == pytest.approx(0.01 * np.sqrt(252))


def test_garch_walk_forward_has_no_look_ahead(series):
    s = series.iloc[:1400]
    start, cut = s.index[1300], s.index[1340]
    a = gm.walk_forward_garch(s, 10, start, refit_every=20, n_sims=50)
    tampered = s.copy()
    tampered.loc[tampered.index > cut] *= 4
    b = gm.walk_forward_garch(tampered, 10, start, refit_every=20, n_sims=50)
    pd.testing.assert_frame_equal(a.loc[:cut], b.loc[:cut])
    assert not np.allclose(a.loc[cut:].iloc[1:]["point"], b.loc[cut:].iloc[1:]["point"])
    assert (a["q10"] <= a["q50"]).all() and (a["q50"] <= a["q90"]).all()


def test_garch_simulation_intervals_are_reproducible(series):
    s = series.iloc[:1400]
    a = gm.walk_forward_garch(s, 10, s.index[1350], refit_every=25, n_sims=50, seed=3)
    b = gm.walk_forward_garch(s, 10, s.index[1350], refit_every=25, n_sims=50, seed=3)
    pd.testing.assert_frame_equal(a, b)
    t_fit = gm.fit_garch(s, "t")
    c1 = gm.current_forecast(t_fit, 10, n_sims=200, seed=5)
    c2 = gm.current_forecast(t_fit, 10, n_sims=200, seed=5)
    pd.testing.assert_series_equal(c1, c2)


def test_garch_distribution_selection_uses_development_only():
    t_series = garch_series(n=2500, dist="t", seed=1)
    split = make_oos_split(t_series.index, 400, 21)
    dist, table = gm.select_distribution(t_series, split)
    assert dist == "t"
    assert set(table.index) == {"normal", "t"}


# --- LSTM data preparation -------------------------------------------------------------------

SMALL = LSTMConfig(horizon=10, seq_len=20, hidden_size=8, max_epochs=4, patience=2, batch_size=128)


def test_features_are_backward_looking(series):
    f = build_features(series)
    tampered = series.copy()
    cut = series.index[1000]
    tampered.loc[tampered.index > cut] *= 3
    pd.testing.assert_frame_equal(f.loc[:cut], build_features(tampered).loc[:cut])


def test_lstm_segments_are_chronological_purged_and_development_only(series):
    s = series.iloc[:1500]
    known = realized_vol_forward(s, SMALL.horizon).dropna().index
    split = make_oos_split(known, 300, purge_days=SMALL.horizon)
    prep = prepare_data(s, split, SMALL)
    seg, pos = prep.segments, s.index.get_loc
    assert seg.train[-1] < seg.val[0] < seg.val[-1] < seg.cal[0] < seg.cal[-1] < seg.test[0]
    # Purge: no training target window (t+1..t+h) reaches a later segment's target window.
    assert pos(seg.val[0]) - pos(seg.train[-1]) > SMALL.horizon
    assert pos(seg.cal[0]) - pos(seg.val[-1]) > SMALL.horizon
    # Development targets end before the test window begins.
    assert pos(seg.cal[-1]) + SMALL.horizon < pos(split.test_start)
    # Scaler fitted on training-period feature rows only.
    rows = prep.features.loc[prep.features.index <= seg.train[-1]].dropna()
    np.testing.assert_allclose(prep.scaler.mean_, rows.mean().to_numpy())


def test_pinball_loss_and_conformal_adjustment():
    pred = torch.tensor([[0.0, 1.0, 2.0]])
    y = torch.tensor([3.0])
    # errors y - q = 3, 2, 1 -> q * err = 0.3, 1.0, 0.9 ; mean = 0.7333
    assert float(pinball_loss(pred, y, (0.1, 0.5, 0.9))) == pytest.approx(2.2 / 3)
    rng = np.random.default_rng(0)
    y = rng.normal(size=2000)
    lo, hi = np.full(2000, -0.5), np.full(2000, 0.5)  # too narrow for 80 %
    c = conformal_adjustment(lo, hi, y, 0.8)
    assert c > 0
    assert np.mean((y >= lo - c) & (y <= hi + c)) >= 0.8


def test_lstm_training_is_reproducible_and_checkpointed(series, tmp_path):
    s = series.iloc[:1500]
    known = realized_vol_forward(s, SMALL.horizon).dropna().index
    split = make_oos_split(known, 300, purge_days=SMALL.horizon)
    a = train_lstm(s, split, SMALL, checkpoint_dir=tmp_path, name="x")
    b = train_lstm(s, split, SMALL, checkpoint_dir=tmp_path, name="x")
    pa, pb = a.predict(a.prep.segments.test), b.predict(b.prep.segments.test)
    pd.testing.assert_frame_equal(pa, pb)
    assert (pa["q10_raw"] <= pa["point"]).all() and (pa["point"] <= pa["q90_raw"]).all()
    assert (pa > 0).all().all()
    assert a.checkpoint_path.exists()
    ckpt = torch.load(a.checkpoint_path, weights_only=False)
    assert ckpt["best_epoch"] == a.best_epoch
    assert np.isfinite(a.conformal_adjustment)


# --- evaluation ------------------------------------------------------------------------------

def test_metrics_hand_calculation():
    idx = pd.bdate_range("2024-01-01", periods=4)
    y = pd.Series([0.2, 0.2, 0.3, 0.1], index=idx)
    f = pd.Series([0.25, 0.15, 0.3, 0.2], index=idx)
    c = pd.Series([0.18, 0.25, 0.2, 0.2], index=idx)
    m = ev.forecast_metrics(f, y, c, f - 0.06, f + 0.06)
    assert m["MAE"] == pytest.approx(0.05)
    assert m["Bias"] == pytest.approx(0.025)
    assert m["RMSE"] == pytest.approx(np.sqrt((0.0025 + 0.0025 + 0 + 0.01) / 4))
    # direction vs current: (+,+), (-,-), (+,+), (0 -> excluded? y-c=-0.1, f-c=0)
    assert m["Directional accuracy"] == pytest.approx(3 / 4)
    assert m["Interval coverage"] == pytest.approx(0.75)
    assert ev.qlike(np.array([0.2]), np.array([0.2]))[0] == pytest.approx(0.0)


def test_diebold_mariano():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=600)
    base = pd.Series(rng.normal(1, 0.3, 600), index=idx)
    better = base - 0.3
    res = ev.diebold_mariano(better, base + rng.normal(0, 0.1, 600), horizon=5)
    assert res.mean_loss_difference < 0 and res.p_value < 0.01
    same = ev.diebold_mariano(base + rng.normal(0, 0.1, 600), base + rng.normal(0, 0.1, 600), 5)
    assert same.p_value > 0.01
    assert list(ev.non_overlapping(idx[:10], 3)) == list(idx[[0, 3, 6, 9]])


# --- study smoke test -----------------------------------------------------------------------

def test_volatility_study_end_to_end(series, tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path)
    s = series.iloc[:1600]
    cfg = VolStudyConfig(horizon=10, test_days=250, garch_dist="normal", garch_sims=50,
                         lstm=SMALL)
    res = run_volatility_study(s, "synthetic", cfg)
    f = res.forecasts
    assert len(f) == 250 and not f.isna().any().any()
    assert set(res.metrics_all.index) == {"Historical 20d", "Historical 60d", "Historical 252d",
                                          "GARCH(1,1)", "LSTM"}
    assert res.metrics_non_overlapping["n"].iloc[0] == 25
    assert "LSTM vs GARCH(1,1)" in res.dm_tests.index
    assert set(res.latest.index) >= {"GARCH(1,1)", "LSTM"}
    assert res.latest.loc["GARCH(1,1)", "lower"] <= res.latest.loc["GARCH(1,1)", "upper"]
    assert f.index[0] > res.split.dev_end
