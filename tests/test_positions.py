import numpy as np
import pandas as pd
import pytest

from ui.positions import A, T, W, add_position, sync_positions


def equal(n=7, total=100_000.0):
    return pd.DataFrame({T: [f"S{i}" for i in range(n)], W: [100 / n] * n, A: [total / n] * n})


def test_amount_edit_recalculates_all_weights():
    prev = equal()
    cur = prev.copy()
    cur.loc[0, A] = 244_411.44
    out, msg, changed = sync_positions(prev, cur)
    total = 244_411.44 + 6 * 100_000 / 7
    assert changed and msg is None
    assert out.loc[0, W] == pytest.approx(244_411.44 / total * 100)  # ~74.04%
    assert out[W].sum() == pytest.approx(100)
    assert out.loc[0, A] == pytest.approx(244_411.44)  # amounts kept as typed


def test_single_weight_edit_rescales_others_and_amounts():
    prev = pd.DataFrame({T: ["A", "B", "C"], W: [50.0, 30.0, 20.0], A: [50_000.0, 30_000.0, 20_000.0]})
    cur = prev.copy()
    cur.loc[0, W] = 60.0
    out, _, changed = sync_positions(prev, cur)
    assert changed
    np.testing.assert_allclose(out[W], [60, 24, 16])  # others keep their 30:20 ratio
    np.testing.assert_allclose(out[A], [60_000, 24_000, 16_000])  # same total value
    over = cur.copy()
    over.loc[0, W] = 150.0
    out2, _, _ = sync_positions(prev, over)
    np.testing.assert_allclose(out2[W], [100, 0, 0])


def test_single_weight_edit_when_others_are_zero():
    prev = pd.DataFrame({T: ["A", "B", "C"], W: [100.0, 0.0, 0.0], A: [10_000.0, 0.0, 0.0]})
    cur = prev.copy()
    cur.loc[0, W] = 40.0
    out, _, _ = sync_positions(prev, cur)
    np.testing.assert_allclose(out[W], [40, 30, 30])


def test_removing_a_row_recalculates_from_amounts():
    prev = equal()
    out, msg, changed = sync_positions(prev, prev.drop(index=3))
    assert changed and "remaining amounts" in msg
    np.testing.assert_allclose(out[W], [100 / 6] * 6)
    assert out[A].sum() == pytest.approx(6 * 100_000 / 7)


def test_added_empty_row_starts_at_zero():
    prev = equal(3)
    cur = pd.concat([prev, pd.DataFrame({T: ["NEW"], W: [np.nan], A: [np.nan]})], ignore_index=True)
    out, msg, _ = sync_positions(prev, cur)
    assert out.loc[3, W] == 0 and out.loc[3, A] == 0 and out[W].sum() == pytest.approx(100)


def test_no_change_and_multi_weight_paste():
    prev = equal(3)
    out, _, changed = sync_positions(prev, prev.copy())
    assert not changed
    cur = prev.copy()
    cur[W] = [10.0, 20.0, 30.0]
    out, _, changed = sync_positions(prev, cur)
    np.testing.assert_allclose(out[W], [10, 20, 30])  # kept as typed; the page warns about 60%
    np.testing.assert_allclose(out[A], [10_000, 20_000, 30_000])


def test_add_position_keeps_equal_portfolios_equal():
    out, msg = add_position(equal(), "7203.T")
    assert len(out) == 8 and out.iloc[-1][T] == "7203.T"
    np.testing.assert_allclose(out[W], 12.5)
    assert out[A].sum() == pytest.approx(100_000)
    uneven = pd.DataFrame({T: ["A", "B"], W: [70.0, 30.0], A: [70.0, 30.0]})
    out2, msg2 = add_position(uneven, "C")
    assert out2.iloc[-1][W] == 0 and "Set its weight" in msg2
