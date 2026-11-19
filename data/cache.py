"""On-disk cache for price series.

Each ticker's full adjusted history is stored as Parquet next to a JSON metadata file.
Adjusted histories are revised retroactively (dividends, splits), so a refresh REPLACES the
cached series rather than appending to it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CacheEntry:
    series: pd.Series
    source: str
    fetched_at: datetime  # timezone-aware UTC

    def age_hours(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        return (now - self.fetched_at).total_seconds() / 3600.0


def _safe_name(ticker: str) -> str:
    # ^VIX -> _caret_VIX ; keeps names unique and filesystem-safe
    name = ticker.replace("^", "_caret_")
    return re.sub(r"[^A-Za-z0-9_.=-]", "_", name)


class PriceCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _paths(self, ticker: str) -> tuple[Path, Path]:
        base = self.directory / _safe_name(ticker)
        return base.with_suffix(".parquet"), base.with_suffix(".json")

    def get(self, ticker: str) -> CacheEntry | None:
        data_path, meta_path = self._paths(ticker)
        if not (data_path.exists() and meta_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text())
            frame = pd.read_parquet(data_path)
        except (OSError, ValueError):
            return None  # corrupt entry -> treat as a miss, it will be refetched
        series = frame.iloc[:, 0].rename(ticker)
        series.index = pd.DatetimeIndex(series.index).astype("datetime64[ns]")
        return CacheEntry(
            series=series,
            source=meta["source"],
            fetched_at=datetime.fromisoformat(meta["fetched_at"]),
        )

    def get_fresh(self, ticker: str, max_age_hours: float,
                  now: datetime | None = None) -> CacheEntry | None:
        entry = self.get(ticker)
        if entry is None or entry.age_hours(now) > max_age_hours:
            return None
        return entry

    def put(self, ticker: str, series: pd.Series, source: str,
            fetched_at: datetime | None = None) -> CacheEntry:
        fetched_at = fetched_at or datetime.now(timezone.utc)
        data_path, meta_path = self._paths(ticker)
        series = series.astype(float).rename("adj_close")
        series.to_frame().to_parquet(data_path)
        meta = {
            "ticker": ticker,
            "source": source,
            "fetched_at": fetched_at.isoformat(),
            "first_date": str(series.index.min()) if len(series) else None,
            "last_date": str(series.index.max()) if len(series) else None,
            "n_obs": int(series.notna().sum()),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        return CacheEntry(series=series.rename(ticker), source=source, fetched_at=fetched_at)
