"""End-to-end volatility forecasting study for one return series.

1. Build the target RV_{t,h} and the locked OOS split (test window = last ``test_days`` dates
   with a known target; purge gap = h).
2. Development only: choose GARCH innovations by BIC; train the LSTM (train/val/cal).
3. Test window: historical (20/60/252 d), walk-forward GARCH and the trained LSTM all forecast
   RV_{t,h} using information up to t only; metrics on all and non-overlapping dates;
   Diebold–Mariano tests against GARCH.
4. Latest forecasts as of the last available date (for the dashboard / risk engine).

The test window is evaluated once; nothing computed on it feeds back into any choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import config
from backtest.splits import OOSSplit, make_oos_split
from core.types import RiskBasis
from forecasting import evaluation as ev
from forecasting import garch_model as gm
from forecasting.historical_volatility import historical_forecast
from forecasting.lstm_model import LSTMConfig, TrainedLSTM, train_lstm
from forecasting.targets import realized_vol_backward, realized_vol_forward


@dataclass(frozen=True)
class VolStudyConfig:
    horizon: int = 21
    test_days: int = config.TEST_WINDOW_DAYS
    hist_windows: tuple[int, ...] = (20, 60, 252)
    garch_dist: str = "auto"  # "auto" (BIC on development data), "normal" or "t"
    garch_refit_every: int = 21
    garch_sims: int = 500
    lstm: LSTMConfig | None = None  # None -> LSTMConfig(horizon=horizon)
    seed: int = config.RANDOM_SEED


@dataclass
class VolStudyResult:
    name: str
    horizon: int
    split: OOSSplit
    forecasts: pd.DataFrame  # test window
    metrics_all: pd.DataFrame
    metrics_non_overlapping: pd.DataFrame
    dm_tests: pd.DataFrame
    garch_dist: str
    garch_dist_table: pd.DataFrame | None
    garch_dev_fit: gm.GarchFit
    lstm: TrainedLSTM
    latest: pd.DataFrame  # one row per model: point, lower, upper, as_of, basis
    notes: list[str] = field(default_factory=list)
    basis: RiskBasis = RiskBasis.FORECAST


def run_volatility_study(log_returns: pd.Series, name: str = "series",
                         cfg: VolStudyConfig = VolStudyConfig()) -> VolStudyResult:
    h = cfg.horizon
    lstm_cfg = cfg.lstm or LSTMConfig(horizon=h, seed=cfg.seed)
    if lstm_cfg.horizon != h:
        raise ValueError("LSTM horizon must equal the study horizon.")

    target = realized_vol_forward(log_returns, h)
    known = target.dropna().index
    split = make_oos_split(known, cfg.test_days, purge_days=h)

    # --- development-only choices ---------------------------------------------------
    if cfg.garch_dist == "auto":
        dist, dist_table = gm.select_distribution(log_returns, split)
    else:
        dist, dist_table = cfg.garch_dist, None
    garch_dev_fit = gm.fit_garch(log_returns, dist, last_date=split.dev_end)
    lstm = train_lstm(log_returns, split, lstm_cfg, name=name)

    # --- test window: out-of-sample forecasts ------------------------------------------
    test_dates = known[(known >= split.test_start) & (known <= split.test_end)]
    df = pd.DataFrame(index=test_dates)
    df["realized"] = target.loc[test_dates]
    df["current_rv21"] = realized_vol_backward(log_returns, 21).loc[test_dates]
    for w in cfg.hist_windows:
        df[f"hist_{w}d"] = historical_forecast(log_returns, w).loc[test_dates]
    g = gm.walk_forward_garch(log_returns, h, test_dates[0], test_dates[-1], dist,
                              cfg.garch_refit_every, cfg.garch_sims, cfg.seed)
    df["garch"], df["garch_q10"], df["garch_q50"], df["garch_q90"] = g["point"], g["q10"], g["q50"], g["q90"]
    lp = lstm.predict(test_dates)
    df["lstm"], df["lstm_q10"], df["lstm_q90"] = lp["point"], lp["q10"], lp["q90"]
    df["lstm_q10_raw"], df["lstm_q90_raw"] = lp["q10_raw"], lp["q90_raw"]

    models = {f"Historical {w}d": f"hist_{w}d" for w in cfg.hist_windows}
    models.update({"GARCH(1,1)": "garch", "LSTM": "lstm"})
    intervals = {"GARCH(1,1)": ("garch_q10", "garch_q90"), "LSTM": ("lstm_q10", "lstm_q90")}
    metrics_all = ev.compare_models(df, "realized", models, intervals, "current_rv21", h)
    metrics_no = ev.compare_models(df, "realized", models, intervals, "current_rv21", h,
                                   subset=ev.non_overlapping(df.index, h))
    dm = ev.dm_table(df, "realized", models, "GARCH(1,1)", h)

    # --- latest forecasts (information up to the last available date) ---------------
    last = log_returns.index[-1]
    rows = []
    for w in cfg.hist_windows:
        v = float(historical_forecast(log_returns, w).iloc[-1])
        rows.append({"model": f"Historical {w}d", "point": v, "lower": None, "upper": None})
    full_fit = gm.fit_garch(log_returns, dist)
    gcur = gm.current_forecast(full_fit, h, seed=cfg.seed)
    rows.append({"model": "GARCH(1,1)", "point": gcur["point"], "lower": gcur["q10"], "upper": gcur["q90"]})
    lcur = lstm.predict(pd.DatetimeIndex([last])).iloc[0]
    rows.append({"model": "LSTM", "point": lcur["point"], "lower": lcur["q10"], "upper": lcur["q90"]})
    latest = pd.DataFrame(rows).set_index("model")
    latest["as_of"] = last
    latest["horizon_days"] = h
    latest["basis"] = RiskBasis.FORECAST.label

    notes = [
        f"Target: annualised realised volatility over the next {h} trading days (zero-mean, log returns).",
        f"Locked test window {split.test_start.date()} – {split.test_end.date()} "
        f"({len(test_dates)} forecast dates; {len(ev.non_overlapping(df.index, h))} non-overlapping).",
        f"GARCH innovations: {dist} (chosen on development data only)." if dist_table is not None
        else f"GARCH innovations: {dist} (fixed).",
        "GARCH is re-estimated every "
        f"{cfg.garch_refit_every} days on an expanding window; the LSTM is trained once on "
        "development data and not re-trained during the test window.",
        "Intervals are 10–90 % (nominal 80 %). GARCH intervals omit parameter uncertainty; "
        "LSTM intervals are conformally calibrated on development data.",
        "Latest GARCH forecast uses parameters re-estimated on all data; the latest LSTM "
        "forecast uses the model validated above.",
    ] + lstm.notes
    return VolStudyResult(name, h, split, df, metrics_all, metrics_no, dm, dist, dist_table,
                          garch_dev_fit, lstm, latest, notes)
