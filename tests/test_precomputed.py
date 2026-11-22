"""Precomputed forecast files: what is saved must load back unchanged."""

import json

import numpy as np
import pandas as pd
import pytest

from app_core import pipeline as pl
from app_core import precomputed as pc
from core.types import RebalanceFrequency
from forecasting.study import VolStudyConfig, run_volatility_study
from tests.test_forecasting import SMALL, garch_series


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    import config
    old = config.CHECKPOINT_DIR
    config.CHECKPOINT_DIR = tmp_path_factory.mktemp("ckpt")
    try:
        cfg = VolStudyConfig(horizon=10, test_days=250, garch_dist="auto", garch_sims=50, lstm=SMALL)
        studies = {n: run_volatility_study(garch_series(1600, seed=i), n, cfg) for i, n in enumerate(["AAA", "Portfolio"])}
    finally:
        config.CHECKPOINT_DIR = old
    return pl.forecast_bundle(10, studies, {"BBB": "no data"})


def test_round_trip_keeps_every_table(bundle, tmp_path):
    s = pl.AnalysisSettings(tickers=("AAA",), weights=(1.0,), horizon_days=10)
    out = pc.save_snapshot(bundle, s, tmp_path / "default")
    fb = pc.load_snapshot(out)
    assert fb is not None and fb.precomputed["horizon_days"] == 10
    assert fb.failures == {"BBB": "no data"}
    pd.testing.assert_frame_equal(fb.latest.drop(columns="As of").astype(float),
                                  bundle.latest.drop(columns="As of").astype(float))
    for name, a in bundle.studies.items():
        b = fb.studies[name]
        pd.testing.assert_frame_equal(b.forecasts, a.forecasts, check_freq=False)
        for attr in ("metrics_all", "metrics_non_overlapping", "dm_tests"):
            pd.testing.assert_frame_equal(getattr(b, attr), getattr(a, attr), check_dtype=False)
        assert b.split == a.split
        assert b.garch_dev_fit.persistence == pytest.approx(a.garch_dev_fit.persistence)
        assert b.garch_dev_fit.unconditional_vol == pytest.approx(a.garch_dev_fit.unconditional_vol)
        assert b.lstm.segment_spans() == a.lstm.segment_spans()
        assert b.lstm.best_epoch == a.lstm.best_epoch
        np.testing.assert_allclose(b.lstm.history.to_numpy(float), a.lstm.history.to_numpy(float))
        assert b.latest.loc["GARCH(1,1)", "point"] == pytest.approx(a.latest.loc["GARCH(1,1)", "point"])
        assert b.notes == a.notes


def test_snapshot_used_only_for_matching_settings():
    snap = pc.settings_to_dict(pl.AnalysisSettings())
    assert pc.settings_match(snap, pl.AnalysisSettings())
    assert pc.settings_match(snap, pl.AnalysisSettings(initial_value=5_000.0))  # scale does not change vols
    assert not pc.settings_match(snap, pl.AnalysisSettings(horizon_days=10))
    assert not pc.settings_match(snap, pl.AnalysisSettings(rebalance=RebalanceFrequency.QUARTERLY))
    other = pl.AnalysisSettings(weights=(0.4,) + (0.1,) * 6)
    assert not pc.settings_match(snap, other)


def test_missing_or_old_files_are_ignored(bundle, tmp_path):
    assert pc.load_snapshot(tmp_path / "nothing") is None
    out = pc.save_snapshot(bundle, pl.AnalysisSettings(), tmp_path / "default")
    meta = json.loads((out / "meta.json").read_text())
    meta["format_version"] = 0
    (out / "meta.json").write_text(json.dumps(meta))
    assert pc.load_snapshot(out) is None
