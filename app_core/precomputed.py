"""Precomputed volatility forecasts for the default portfolio.

Fitting GARCH and training an LSTM for every asset takes a few minutes, which most visitors of the
hosted app will not wait for. ``scripts/precompute_forecasts.py`` runs the full study offline (daily,
from a GitHub Actions workflow) and saves the results here; the app loads them on a first visit, labelled
with the date they were computed.

Files in ``precomputed/<name>/``:

* ``meta.json``             when and for which settings the run was made, failures, library versions;
* ``test_forecasts.parquet`` every series' out-of-sample forecast table (long format, ``series`` column);
* ``studies.json``          per series: split dates, accuracy metrics, Diebold-Mariano tests, latest
                            forecasts with intervals, GARCH diagnostics, LSTM training summary, notes.

Only the results are stored, not the fitted model objects (an arch result or a PyTorch network):
nothing in the app needs to re-run a model from a snapshot, and plain tables are safe to load.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from io import StringIO
from pathlib import Path

import pandas as pd

import config
from backtest.splits import OOSSplit
from core.types import RebalanceFrequency
from forecasting.garch_model import GarchFit
from forecasting.lstm_model import LSTMConfig
from forecasting.study import VolStudyResult

FORMAT_VERSION = 1
PRECOMPUTED_DIR = Path(os.environ.get("PI_PRECOMPUTED_DIR", config.PROJECT_ROOT / "precomputed"))
DEFAULT_NAME = "default"

# Settings that change the forecasts. A snapshot is used only when all of them match.
_FORECAST_FIELDS = ("tickers", "weights", "horizon_days", "rebalance", "history_years", "forecast_history_years")


# --- light stand-ins for the fitted LSTM ----------------------------------------------------------

@dataclass
class LSTMSummary:
    """What the pages and the PDF read from a trained LSTM, without the network itself."""
    cfg: LSTMConfig
    history: pd.DataFrame
    best_epoch: int
    conformal_adjustment: float
    spans: dict[str, tuple[pd.Timestamp, pd.Timestamp, int]]
    notes: list[str] = field(default_factory=list)

    def segment_spans(self) -> dict[str, tuple[pd.Timestamp, pd.Timestamp, int]]:
        return self.spans


# --- settings ---------------------------------------------------------------------------------------

def settings_to_dict(s) -> dict:
    d = {k: getattr(s, k) for k in s.__dataclass_fields__}
    d["tickers"], d["weights"] = list(s.tickers), [float(w) for w in s.weights]
    d["rebalance"] = s.rebalance.value
    return d


def settings_match(snapshot_settings: dict, s) -> bool:
    cur = settings_to_dict(s)
    for k in _FORECAST_FIELDS:
        a, b = snapshot_settings.get(k), cur[k]
        if k == "weights":
            if len(a or []) != len(b) or any(abs(x - y) > 1e-6 for x, y in zip(a, b)):
                return False
        elif a != b:
            return False
    return True


def default_settings():
    from app_core.pipeline import AnalysisSettings
    return AnalysisSettings()


# --- (de)serialisation helpers ------------------------------------------------------------------

def _df(df: pd.DataFrame | None) -> str | None:
    return None if df is None else df.to_json(orient="split", date_format="iso", double_precision=15)


def _undf(text: str | None) -> pd.DataFrame | None:
    return None if text is None else pd.read_json(StringIO(text), orient="split", dtype=False)


def _ts(x) -> str:
    return pd.Timestamp(x).isoformat()


def _study_to_dict(s: VolStudyResult) -> dict:
    g = s.garch_dev_fit
    latest = s.latest.copy()
    latest["as_of"] = latest["as_of"].map(_ts)
    return {
        "name": s.name, "horizon": s.horizon,
        "split": {k: (_ts(v) if isinstance(v, pd.Timestamp) else v) for k, v in asdict(s.split).items()},
        "metrics_all": _df(s.metrics_all), "metrics_non_overlapping": _df(s.metrics_non_overlapping),
        "dm_tests": _df(s.dm_tests), "latest": _df(latest),
        "garch_dist": s.garch_dist, "garch_dist_table": _df(s.garch_dist_table),
        "garch_dev_fit": {
            "dist": g.dist, "params": g.params.to_dict(), "std_err": g.std_err.to_dict(), "n_obs": int(g.n_obs),
            "start": _ts(g.start), "end": _ts(g.end), "log_likelihood": float(g.log_likelihood),
            "aic": float(g.aic), "bic": float(g.bic), "ljung_box": _df(g.ljung_box),
        },
        "lstm": {
            "cfg": asdict(s.lstm.cfg), "history": _df(s.lstm.history), "best_epoch": int(s.lstm.best_epoch),
            "conformal_adjustment": float(s.lstm.conformal_adjustment), "notes": list(s.lstm.notes),
            "spans": {k: [_ts(a), _ts(b), int(n)] for k, (a, b, n) in s.lstm.segment_spans().items()},
        },
        "notes": list(s.notes),
    }


def _study_from_dict(d: dict, forecasts: pd.DataFrame) -> VolStudyResult:
    sp = d["split"]
    split = OOSSplit(**{k: (pd.Timestamp(v) if k != "purge_days" else v) for k, v in sp.items()})
    gd = d["garch_dev_fit"]
    garch = GarchFit(result=None, dist=gd["dist"], params=pd.Series(gd["params"], dtype=float),
                     std_err=pd.Series(gd["std_err"], dtype=float), n_obs=gd["n_obs"],
                     start=pd.Timestamp(gd["start"]), end=pd.Timestamp(gd["end"]),
                     log_likelihood=gd["log_likelihood"], aic=gd["aic"], bic=gd["bic"],
                     ljung_box=_undf(gd["ljung_box"]))
    ld = d["lstm"]
    cfg = dict(ld["cfg"])
    cfg["quantiles"] = tuple(cfg["quantiles"])
    lstm = LSTMSummary(cfg=LSTMConfig(**cfg), history=_undf(ld["history"]), best_epoch=ld["best_epoch"],
                       conformal_adjustment=ld["conformal_adjustment"], notes=ld["notes"],
                       spans={k: (pd.Timestamp(a), pd.Timestamp(b), n) for k, (a, b, n) in ld["spans"].items()})
    latest = _undf(d["latest"])
    latest["as_of"] = pd.to_datetime(latest["as_of"])
    for c in ("lower", "upper"):
        latest[c] = pd.to_numeric(latest[c], errors="coerce")
    return VolStudyResult(d["name"], d["horizon"], split, forecasts, _undf(d["metrics_all"]),
                          _undf(d["metrics_non_overlapping"]), _undf(d["dm_tests"]), d["garch_dist"],
                          _undf(d["garch_dist_table"]), garch, lstm, latest, d["notes"])


# --- public API ---------------------------------------------------------------------------------

def save_snapshot(fb, settings, directory: Path, extra: dict | None = None) -> Path:
    """Write a ForecastBundle to ``directory`` (replaced atomically, so a failed run never leaves
    half-written files behind)."""
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    studies = fb.studies
    data_as_of = max(pd.Timestamp(s.latest["as_of"].iloc[0]) for s in studies.values())
    meta = {
        "format_version": FORMAT_VERSION,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "data_as_of": _ts(data_as_of),
        "horizon_days": fb.horizon_days,
        "settings": settings_to_dict(settings),
        "series": list(studies),
        "failures": dict(fb.failures),
        **(extra or {}),
    }
    frames = []
    for name, s in studies.items():
        f = s.forecasts.copy()
        f.index.name = "date"
        frames.append(f.reset_index().assign(series=name))
    tmp = Path(tempfile.mkdtemp(prefix=directory.name + ".", dir=directory.parent))
    tmp.chmod(0o755)
    try:
        pd.concat(frames, ignore_index=True).to_parquet(tmp / "test_forecasts.parquet", index=False)
        (tmp / "studies.json").write_text(json.dumps({n: _study_to_dict(s) for n, s in studies.items()}))
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
        if directory.exists():
            shutil.rmtree(directory)
        tmp.rename(directory)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return directory


def load_snapshot(directory: Path):
    """Return a ForecastBundle (with ``precomputed`` set to the metadata), or None if the files are
    missing, from an older format, or unreadable."""
    from app_core.pipeline import forecast_bundle
    directory = Path(directory)
    try:
        meta = json.loads((directory / "meta.json").read_text())
        if meta.get("format_version") != FORMAT_VERSION:
            return None
        raw = json.loads((directory / "studies.json").read_text())
        long = pd.read_parquet(directory / "test_forecasts.parquet")
    except (OSError, ValueError):
        return None
    studies = {}
    for name, d in raw.items():
        f = long[long["series"] == name].drop(columns="series").set_index("date")
        f.index = pd.DatetimeIndex(f.index)
        f.index.name = None
        studies[name] = _study_from_dict(d, f)
    fb = forecast_bundle(meta["horizon_days"], studies, meta.get("failures"))
    fb.precomputed = meta
    return fb


def settings_from_meta(meta: dict):
    from app_core.pipeline import AnalysisSettings
    d = dict(meta["settings"])
    d["tickers"], d["weights"] = tuple(d["tickers"]), tuple(d["weights"])
    d["rebalance"] = RebalanceFrequency(d["rebalance"])
    return AnalysisSettings(**d)
