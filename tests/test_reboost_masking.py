import numpy as np
import pandas as pd
import pytest

from leo_decay.constants import SECONDS_PER_DAY
from leo_decay.reboost_masking import (
    REBOOST_RATE_THRESHOLD_M_S, mask_reboosts, reboost_epochs,
)
from leo_decay.tle_conversion import decay_rate_m_s


def test_threshold_is_half_km_per_day():
    assert REBOOST_RATE_THRESHOLD_M_S * SECONDS_PER_DAY == pytest.approx(500.0)


def _iss_series():
    """Daily ISS altitudes decaying 15 m/day with a +1.7 km burn on day 10."""
    epoch = pd.Series(pd.date_range("2026-06-21", periods=25, freq="D"))
    h = 415_000.0 - 15.0 * np.arange(25)
    h[10:] += 1_700.0
    return epoch, pd.Series(h)


def test_detects_single_reboost():
    epoch, h = _iss_series()
    rate = decay_rate_m_s(h, epoch)
    hits = reboost_epochs(rate, epoch)
    assert list(hits) == [epoch.iloc[10]]


def test_masks_symmetric_three_day_window():
    epoch, h = _iss_series()
    rate = decay_rate_m_s(h, epoch)
    masked = mask_reboosts(rate, epoch)
    masked_idx = set(np.flatnonzero(masked.isna().to_numpy()))
    # index 0 is NaN from the diff; days 7..13 are the closed ±3-day window
    assert masked_idx == {0, *range(7, 14)}
    # Everything outside the window is the natural -15 m/day decay
    kept = masked.dropna()
    np.testing.assert_allclose(kept, -15.0 / SECONDS_PER_DAY)


def test_no_reboost_no_mask():
    epoch = pd.Series(pd.date_range("2026-01-01", periods=10, freq="D"))
    rate = decay_rate_m_s(pd.Series(415_000.0 - 20.0 * np.arange(10)), epoch)
    pd.testing.assert_series_equal(mask_reboosts(rate, epoch), rate)
