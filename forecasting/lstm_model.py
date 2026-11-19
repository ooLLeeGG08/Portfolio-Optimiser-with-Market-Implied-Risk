"""LSTM volatility forecaster with calibrated quantile intervals (FORECAST basis).
See METHODOLOGY §7.

Pipeline
--------
1. Features at t (backward-looking only, data <= t): log return, |r|, and log of 5/21/63-day
   trailing realised vol.
2. Target at t: log RV_{t,h} (forecasting.targets) — uses t+1..t+h.
3. Chronological split of the DEVELOPMENT window (dates before the locked test window) into
   train / validation / calibration with an h-day purge gap between consecutive segments,
   so no training target window overlaps a later segment's returns.
4. StandardScaler fitted on feature rows dated up to the last training date only.
5. Network: LSTM -> linear head producing the 10/50/90 % quantiles of log RV, with
   non-crossing enforced by construction (q10 = m - softplus(a), q90 = m + softplus(b)).
   Loss: pinball (quantile) loss.
6. Training: Adam, fixed seeds, deterministic algorithms, early stopping on validation loss,
   best-epoch checkpoint saved to disk and restored.
7. Split-conformal calibration (conformalised quantile regression) on the calibration
   segment: intervals are widened (or narrowed) by the finite-sample-corrected quantile of
   the conformity scores. Exchangeability is only approximate for time series, so coverage
   is always re-measured on the test window.

Hyper-parameters are fixed a priori in ``LSTMConfig``; none is tuned on test data.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

import config
from backtest.splits import OOSSplit, assert_development_only
from forecasting.targets import realized_vol_backward, realized_vol_forward

FEATURE_WINDOWS = (5, 21, 63)


@dataclass(frozen=True)
class LSTMConfig:
    horizon: int = 21
    seq_len: int = 60
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 150
    patience: int = 15
    quantiles: tuple[float, float, float] = (0.10, 0.50, 0.90)
    train_frac: float = 0.70
    val_frac: float = 0.15  # remainder = calibration
    seed: int = config.RANDOM_SEED

    @property
    def interval_coverage(self) -> float:
        return self.quantiles[2] - self.quantiles[0]

    def key(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]


# --- data preparation ------------------------------------------------------------------------

def build_features(log_returns: pd.Series) -> pd.DataFrame:
    """Backward-looking features at t; rows with incomplete windows are NaN."""
    f = pd.DataFrame(index=log_returns.index)
    f["ret"] = log_returns
    f["abs_ret"] = log_returns.abs()
    for w in FEATURE_WINDOWS:
        f[f"log_rv_{w}"] = np.log(realized_vol_backward(log_returns, w).clip(lower=1e-8))
    return f


def build_target(log_returns: pd.Series, horizon: int) -> pd.Series:
    return np.log(realized_vol_forward(log_returns, horizon).clip(lower=1e-8)).rename("log_rv_target")


@dataclass
class SegmentIndex:
    train: pd.DatetimeIndex
    val: pd.DatetimeIndex
    cal: pd.DatetimeIndex
    test: pd.DatetimeIndex


def split_dates(sample_dates: pd.DatetimeIndex, oos: OOSSplit, cfg: LSTMConfig) -> SegmentIndex:
    """Chronological train/val/cal within development, h-day purge gaps, plus test dates.

    ``sample_dates`` are dates with complete features, a full sequence history and a known
    target. Dev samples must have their whole target window before the test window
    (guaranteed by ``oos.purge_days >= horizon``)."""
    if oos.purge_days < cfg.horizon:
        raise ValueError("The out-of-sample split must purge at least `horizon` days.")
    dev = sample_dates[sample_dates <= oos.dev_end]
    test = sample_dates[sample_dates >= oos.test_start]
    h = cfg.horizon
    n = len(dev)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    train = dev[:n_train]
    val = dev[n_train + h:n_train + n_val]
    cal = dev[n_train + n_val + h:]
    if min(len(train), len(val), len(cal)) < 50:
        raise ValueError(f"Too little development data for LSTM training "
                         f"(train {len(train)}, val {len(val)}, cal {len(cal)} samples).")
    return SegmentIndex(train, val, cal, test)


@dataclass
class PreparedData:
    features: pd.DataFrame  # unscaled, all dates
    target: pd.Series  # log RV target, NaN where unknown
    scaler: StandardScaler
    segments: SegmentIndex
    sample_dates: pd.DatetimeIndex  # all dates with a full input sequence and known target


def prepare_data(log_returns: pd.Series, oos: OOSSplit, cfg: LSTMConfig) -> PreparedData:
    feats = build_features(log_returns)
    target = build_target(log_returns, cfg.horizon)
    valid_feat = feats.notna().all(axis=1)
    # A date can be a sample if the previous seq_len feature rows are complete.
    full_seq = valid_feat.rolling(cfg.seq_len, min_periods=cfg.seq_len).sum() == cfg.seq_len
    sample_dates = feats.index[full_seq & target.notna()]
    seg = split_dates(sample_dates, oos, cfg)
    # Scaler: feature rows available at or before the last training date only.
    train_rows = feats.loc[(feats.index <= seg.train[-1]) & valid_feat]
    assert_development_only(train_rows, oos, "LSTM feature scaling")
    scaler = StandardScaler().fit(train_rows.to_numpy())
    return PreparedData(feats, target, scaler, seg, sample_dates)


def make_sequences(prep: PreparedData, dates: pd.DatetimeIndex, seq_len: int) -> np.ndarray:
    """X[i] = scaled features for the seq_len dates ending at dates[i] (inclusive)."""
    scaled = prep.scaler.transform(prep.features.to_numpy())
    pos = prep.features.index.get_indexer(dates)
    if (pos < seq_len - 1).any():
        raise ValueError("Not enough history for a full input sequence.")
    X = np.stack([scaled[p - seq_len + 1:p + 1] for p in pos]).astype(np.float32)
    if np.isnan(X).any():
        raise ValueError("Input sequence contains missing features.")
    return X


# --- model ---------------------------------------------------------------------------------

class QuantileLSTM(nn.Module):
    def __init__(self, n_features: int, cfg: LSTMConfig):
        super().__init__()
        self.lstm = nn.LSTM(n_features, cfg.hidden_size, cfg.num_layers, batch_first=True,
                            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0)
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_size, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        z = self.head(self.drop(out[:, -1, :]))
        mid = z[:, 0]
        lo = mid - nn.functional.softplus(z[:, 1])
        hi = mid + nn.functional.softplus(z[:, 2])
        return torch.stack([lo, mid, hi], dim=1)  # non-crossing quantiles


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, quantiles) -> torch.Tensor:
    q = torch.tensor(quantiles, dtype=pred.dtype)
    err = y.unsqueeze(1) - pred
    return torch.maximum(q * err, (q - 1) * err).mean()


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


@dataclass
class TrainedLSTM:
    model: QuantileLSTM
    cfg: LSTMConfig
    prep: PreparedData
    history: pd.DataFrame  # epoch, train_loss, val_loss
    best_epoch: int
    conformal_adjustment: float  # added to the upper / subtracted from the lower log quantile
    checkpoint_path: Path | None
    notes: list[str] = field(default_factory=list)

    def predict(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Annualised vol forecasts for RV_{t,h}: point (median), raw and calibrated 10/90 %."""
        X = torch.from_numpy(make_sequences(self.prep, dates, self.cfg.seq_len))
        self.model.eval()
        with torch.no_grad():
            q = self.model(X).numpy()
        c = self.conformal_adjustment
        return pd.DataFrame({
            "point": np.exp(q[:, 1]),
            "q10_raw": np.exp(q[:, 0]), "q90_raw": np.exp(q[:, 2]),
            "q10": np.exp(q[:, 0] - c), "q90": np.exp(q[:, 2] + c),
        }, index=dates)


def conformal_adjustment(q_lo: np.ndarray, q_hi: np.ndarray, y: np.ndarray, coverage: float) -> float:
    """CQR: scores E_i = max(q_lo - y, y - q_hi); return the ceil((n+1)(1-a))/n empirical
    quantile, which gives >= `coverage` marginal coverage under exchangeability."""
    scores = np.maximum(q_lo - y, y - q_hi)
    n = len(scores)
    level = min(np.ceil((n + 1) * coverage) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def train_lstm(log_returns: pd.Series, oos: OOSSplit, cfg: LSTMConfig = LSTMConfig(),
               checkpoint_dir: Path | None = config.CHECKPOINT_DIR,
               name: str = "series") -> TrainedLSTM:
    _set_seeds(cfg.seed)
    prep = prepare_data(log_returns, oos, cfg)
    seg = prep.segments
    for part in (seg.train, seg.val, seg.cal):
        assert_development_only(part, oos, "LSTM training/validation/calibration")

    def tensors(dates):
        return (torch.from_numpy(make_sequences(prep, dates, cfg.seq_len)),
                torch.from_numpy(prep.target.loc[dates].to_numpy(np.float32)))

    Xtr, ytr = tensors(seg.train)
    Xva, yva = tensors(seg.val)
    model = QuantileLSTM(Xtr.shape[2], cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)

    best_val, best_state, best_epoch, wait, rows = np.inf, None, -1, 0, []
    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        total = 0.0
        for i in range(0, len(Xtr), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = pinball_loss(model(Xtr[b]), ytr[b], cfg.quantiles)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(b)
        model.eval()
        with torch.no_grad():
            val_loss = float(pinball_loss(model(Xva), yva, cfg.quantiles))
        rows.append({"epoch": epoch, "train_loss": total / len(Xtr), "val_loss": val_loss})
        if val_loss < best_val - 1e-6:
            best_val, best_state, best_epoch, wait = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    model.load_state_dict(best_state)
    ckpt = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() else "_" for ch in name)
        ckpt = checkpoint_dir / f"lstm_{safe}_{cfg.key()}_{oos.dev_end:%Y%m%d}.pt"
        torch.save({"state_dict": best_state, "config": asdict(cfg), "best_epoch": best_epoch,
                    "val_loss": best_val}, ckpt)

    trained = TrainedLSTM(model, cfg, prep, pd.DataFrame(rows), best_epoch, 0.0, ckpt)
    cal_pred = trained.predict(seg.cal)
    y_cal = prep.target.loc[seg.cal].to_numpy()
    trained.conformal_adjustment = conformal_adjustment(
        np.log(cal_pred["q10_raw"].to_numpy()), np.log(cal_pred["q90_raw"].to_numpy()),
        y_cal, cfg.interval_coverage)
    if best_epoch == cfg.max_epochs - 1:
        trained.notes.append("Validation loss was still improving at the epoch limit.")
    return trained
