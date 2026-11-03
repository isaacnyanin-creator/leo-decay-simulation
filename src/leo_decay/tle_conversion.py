"""TLE mean motion -> semi-major axis, altitude, speed and decay rate (§3.2, Appendix A.1).

Everything is SI: metres, seconds, m/s. Mean motion is accepted in the TLE's
native rev/day and converted immediately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import MU_EARTH, R_EARTH, SECONDS_PER_DAY


def mean_motion_to_rad_s(n_rev_day):
    """rev/day -> rad/s."""
    return np.asarray(n_rev_day, dtype=float) * (2.0 * np.pi / SECONDS_PER_DAY)


def semi_major_axis_m(n_rev_day):
    """Kepler's third law inverted: a = (mu / n^2)^(1/3), in metres."""
    n = mean_motion_to_rad_s(n_rev_day)
    return np.cbrt(MU_EARTH / n**2)


def mean_motion_from_semi_major_axis(a_m):
    """Inverse of :func:`semi_major_axis_m`; returns rev/day."""
    n_rad_s = np.sqrt(MU_EARTH / np.asarray(a_m, dtype=float) ** 3)
    return n_rad_s * SECONDS_PER_DAY / (2.0 * np.pi)


def altitude_m(a_m):
    """Mean geodetic altitude h = a - R_E (m)."""
    return np.asarray(a_m, dtype=float) - R_EARTH


def circular_speed_m_s(a_m):
    """Vis-viva circular speed v = sqrt(mu / a) (m/s)."""
    return np.sqrt(MU_EARTH / np.asarray(a_m, dtype=float))


def add_orbital_state(df: pd.DataFrame,
                      mean_motion_col: str = "MEAN_MOTION",
                      epoch_col: str = "EPOCH") -> pd.DataFrame:
    """Return a copy of ``df`` with SI state columns added.

    Adds ``semi_major_axis_m``, ``altitude_m``, ``speed_m_s``. Rows are sorted by
    epoch and duplicate epochs removed, as described in §3.1.
    """
    out = (df.sort_values(epoch_col)
             .drop_duplicates(subset=epoch_col)
             .reset_index(drop=True)
             .copy())
    out["semi_major_axis_m"] = semi_major_axis_m(out[mean_motion_col].to_numpy())
    out["altitude_m"] = altitude_m(out["semi_major_axis_m"].to_numpy())
    out["speed_m_s"] = circular_speed_m_s(out["semi_major_axis_m"].to_numpy())
    return out


def decay_rate_m_s(altitude: pd.Series, epoch: pd.Series) -> pd.Series:
    """Backward finite-difference decay rate (da/dt)_obs,i in m/s.

    da/dt == dh/dt because R_E is constant. The first element is NaN.
    """
    dt_s = pd.to_datetime(epoch).diff().dt.total_seconds()
    return altitude.diff() / dt_s
