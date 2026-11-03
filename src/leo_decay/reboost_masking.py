"""ISS thruster-reboost masking (§3.2, Appendix A.1)."""

from __future__ import annotations

import pandas as pd

from .constants import SECONDS_PER_DAY

REBOOST_RATE_THRESHOLD_M_S = 500.0 / SECONDS_PER_DAY   # +0.50 km/day in m/s
REBOOST_BUFFER = pd.Timedelta(days=3)                  # symmetric ±3-day exclusion


def reboost_epochs(decay_rate_m_s: pd.Series, epoch: pd.Series,
                   threshold_m_s: float = REBOOST_RATE_THRESHOLD_M_S) -> pd.Series:
    """Epochs at which the altitude gain rate exceeds ``threshold_m_s``."""
    return pd.to_datetime(epoch)[decay_rate_m_s > threshold_m_s]


def mask_reboosts(decay_rate_m_s: pd.Series, epoch: pd.Series,
                  threshold_m_s: float = REBOOST_RATE_THRESHOLD_M_S,
                  buffer: pd.Timedelta = REBOOST_BUFFER) -> pd.Series:
    """Return a copy of ``decay_rate_m_s`` with NaN within ±``buffer`` of each reboost.

    The window is closed on both ends, matching Appendix A.1.
    """
    epoch = pd.to_datetime(epoch)
    masked = decay_rate_m_s.copy()
    for r_epoch in reboost_epochs(decay_rate_m_s, epoch, threshold_m_s):
        window = (epoch >= r_epoch - buffer) & (epoch <= r_epoch + buffer)
        masked[window] = float("nan")
    return masked
