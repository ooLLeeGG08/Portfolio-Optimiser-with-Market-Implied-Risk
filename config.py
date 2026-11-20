"""Application-wide defaults.

Financial conventions (trading days, annualisation, loss sign) live in
``core.conventions`` — not here — so there is exactly one source of truth for them.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / ".cache" / "market_data"
CHECKPOINT_DIR = PROJECT_ROOT / ".cache" / "checkpoints"

# --- Default portfolio (example only; fully editable in the app) -------------------
DEFAULT_TICKERS: tuple[str, ...] = ("AAPL", "MSFT", "JPM", "XOM", "JNJ", "TLT", "GLD")
DEFAULT_BENCHMARK = "SPY"
DEFAULT_HISTORY_YEARS = 10
DEFAULT_INITIAL_VALUE = 100_000.0

# --- Currency -------------------------------------------------------------------
# All prices are converted to this currency (unhedged), see METHODOLOGY §2.
BASE_CURRENCY = "USD"

# --- Market reference series -----------------------------------------------------
VIX_TICKER = "^VIX"
RISK_FREE_TICKER = "^IRX"  # 13-week T-bill yield, quoted in percent
TEN_YEAR_YIELD_TICKER = "^TNX"  # 10-year Treasury yield, quoted in percent

# --- Data validation thresholds --------------------------------------------------
MIN_OBSERVATIONS = 252  # minimum aligned daily observations per asset
STALE_DAYS_THRESHOLD = 10  # trading days behind the panel before flagging as stale
EXTREME_RETURN_THRESHOLD = 0.40  # |daily simple return| flagged for review (not removed)

# --- Cache -----------------------------------------------------------------------
CACHE_TTL_HOURS = 12.0

# --- Out-of-sample protocol (see docs/METHODOLOGY.md §8) ---------------------------
TEST_WINDOW_DAYS = 504  # locked final test window, ~2 years of trading days

# --- Reproducibility ---------------------------------------------------------------
RANDOM_SEED = 42
