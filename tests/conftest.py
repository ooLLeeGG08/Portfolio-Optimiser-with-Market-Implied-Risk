import numpy as np
import pandas as pd
import pytest


def make_gbm_prices(tickers=("A", "B", "C"), n=600, seed=0, start="2020-01-01",
                    mu=0.0004, sigma=0.012) -> pd.DataFrame:
    """Deterministic synthetic geometric-Brownian-motion prices on business days."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    rets = rng.normal(mu, sigma, size=(n, len(tickers)))
    rets[0] = 0.0
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=list(tickers))


@pytest.fixture
def gbm_prices() -> pd.DataFrame:
    return make_gbm_prices()
