"""Compute the default portfolio's volatility forecasts offline and save them for the app.

    python scripts/precompute_forecasts.py

Runs the same study as the app's "Run volatility forecasts" button (historical, GARCH(1,1) and LSTM
forecasts, the walk-forward out-of-sample evaluation and the Diebold-Mariano tests) for every asset
of the default portfolio and for the portfolio itself, then writes the results to
``precomputed/default/`` (see app_core/precomputed.py for the file layout).

Environment variables (all optional; no API keys are needed, yfinance is keyless):
    PI_PRECOMPUTED_DIR  output folder (default: <repo>/precomputed)
    PI_CACHE_DIR        price cache folder (default: <repo>/.cache)

Exit code 1, and the previous files are kept, when the portfolio itself or more than half of the
assets could not be forecast.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core import pipeline as pl
from app_core import precomputed as pc
from data.market_data import MarketDataService


def _commit() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"][:7]
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    t0 = time.time()
    settings = pc.default_settings()
    service = MarketDataService()
    print(f"Default portfolio: {', '.join(settings.tickers)} (horizon {settings.horizon_days} days)")
    mb = pl.load_market_data(settings, service, refresh=True)
    pa = pl.analyze_portfolio(mb)
    fb = pl.run_forecasts(mb, pa, service, progress=lambda p, m: print(f"  [{p:4.0%}] {m}", flush=True))

    n_assets = len(mb.spec.tickers)
    failed_assets = [t for t in fb.failures if t != "Portfolio"]
    for name, err in fb.failures.items():
        print(f"  FAILED {name}: {err}")
    if "Portfolio" not in fb.studies or len(failed_assets) > n_assets / 2:
        print("Too many failures; keeping the previous files.")
        return 1

    import arch
    import torch
    out = pc.save_snapshot(fb, settings, pc.PRECOMPUTED_DIR / pc.DEFAULT_NAME, extra={
        "runtime_seconds": round(time.time() - t0),
        "code_commit": _commit(),
        "python": platform.python_version(),
        "torch": torch.__version__, "arch": arch.__version__,
    })
    print(f"Saved {len(fb.studies)} series to {out} in {time.time() - t0:.0f} s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
