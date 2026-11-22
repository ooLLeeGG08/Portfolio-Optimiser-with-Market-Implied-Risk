"""Key takeaways are written from the numbers they are given."""

import numpy as np
import pandas as pd

from app_core.takeaways import key_takeaways, risk_concentration, vix_level, worst_drawdown
from core.types import RiskBasis
from implied_volatility.vix import vix_summary
from risk.risk_contribution import euler_contributions
from risk.risk_metrics import MaxDrawdown


def _rc(weights, vols, corr):
    v = np.asarray(vols)
    cov = pd.DataFrame(np.asarray(corr) * np.outer(v, v), index=list("ABC"), columns=list("ABC"))
    return euler_contributions(pd.Series(weights, index=list("ABC")), cov)


def test_concentration_names_the_largest_gap():
    rc = _rc([0.4, 0.3, 0.3], [0.40, 0.10, 0.10], np.eye(3))  # A dominates risk
    t = risk_concentration(rc)
    share = rc.table.loc["A", "pct_contribution"]
    assert t.text.startswith(f"A is 40.0% of the portfolio but {share:.1%} of its volatility")


def test_concentration_explains_a_hedge():
    corr = [[1, 0.8, -0.6], [0.8, 1, -0.6], [-0.6, -0.6, 1]]
    rc = _rc([0.35, 0.35, 0.30], [0.25, 0.25, 0.15], corr)
    assert rc.table.loc["C", "pct_contribution"] < 0
    t = risk_concentration(rc)
    assert t.text.startswith("C is 30.0% of the portfolio but −") and "offsets" in t.text


def test_vix_sentence_uses_level_and_percentile():
    idx = pd.bdate_range("2000-01-03", periods=600)
    v = pd.Series(np.linspace(0.10, 0.40, 600), index=idx)
    t = vix_level(vix_summary(v))
    assert "The VIX is 40.0%" in t.text and "higher than 100% of trading days since 2000" in t.text
    assert "a high level" in t.text and t.basis is RiskBasis.MARKET_IMPLIED
    low = vix_level(vix_summary(v[::-1].set_axis(idx)))
    assert "a low level" in low.text
    assert "unavailable" in vix_level(None).text


def test_drawdown_sentence_with_and_without_recovery():
    md = MaxDrawdown(-0.242, pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23"), pd.Timestamp("2020-06-08"), 23, 53)
    assert worst_drawdown(md).text == ("The worst fall was −24.2%, from 19 Feb 2020 to 23 Mar 2020 (23 trading days). "
                                       "It took until 08 Jun 2020 (53 more trading days) to recover.")
    open_md = MaxDrawdown(-0.1, pd.Timestamp("2026-01-02"), pd.Timestamp("2026-03-02"), None, 40, None)
    assert "not yet recovered" in worst_drawdown(open_md).text
    assert len(key_takeaways(_rc([1/3] * 3, [0.2] * 3, np.eye(3)), None, md)) == 3
