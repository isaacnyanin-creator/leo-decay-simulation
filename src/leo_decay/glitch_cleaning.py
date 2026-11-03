"""PCHIP sanitisation of single-step TLE glitches on passive satellites (§3.2, Appendix A.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

GLITCH_THRESHOLD_M = 2_000.0    # |Δh| > 2.0 km flagged as unphysical


def flag_glitches(altitude_m: pd.Series, threshold_m: float = GLITCH_THRESHOLD_M) -> pd.Series:
    """Boolean mask of samples whose single-step jump exceeds ``threshold_m``.

    Matches Appendix A.1: ``altitude.diff().abs() > threshold``.
    """
    return altitude_m.diff().abs() > threshold_m


def clean_glitches(altitude_m: pd.Series,
                   threshold_m: float = GLITCH_THRESHOLD_M) -> pd.Series:
    """Replace flagged samples with NaN and rebuild them by PCHIP interpolation.

    Interpolation is over sample position (the series is on a uniform grid),
    matching ``Series.interpolate(method='pchip')`` in the paper. Uses scipy
    directly so endpoints and behaviour are explicit. Leading/trailing NaNs
    are left as NaN rather than extrapolated.
    """
    values = altitude_m.to_numpy(dtype=float).copy()
    values[flag_glitches(altitude_m, threshold_m).to_numpy()] = np.nan

    x = np.arange(values.size, dtype=float)
    good = ~np.isnan(values)
    if good.sum() < 2:
        return pd.Series(values, index=altitude_m.index, name=altitude_m.name)

    interp = PchipInterpolator(x[good], values[good], extrapolate=False)
    filled = values.copy()
    missing = ~good
    filled[missing] = interp(x[missing])
    return pd.Series(filled, index=altitude_m.index, name=altitude_m.name)
