"""Space-weather-modulated exponential thermospheric density (Appendix A.3, Eq. B.3).

rho = rho0 * exp(-(h - h0) / H) * max(0.2, 1 + 0.008 (F10.7 - 100))
                                * max(0.5, 1 + 0.003 (Ap - 15)),
floored at 1e-18 kg/m^3. Input is semi-major axis in metres.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    AP_BASELINE, AP_SENSITIVITY, F107_BASELINE, F107_SENSITIVITY,
    GEOMAG_FACTOR_MIN, H_0_REF, R_EARTH, RHO_0, RHO_FLOOR,
    SCALE_HEIGHT, SOLAR_FACTOR_MIN,
)


def barometric_density(a_m):
    """Hydrostatic exponential profile rho0 * exp(-(a - R_E - h0) / H), kg/m^3."""
    h = np.asarray(a_m, dtype=float) - R_EARTH
    return RHO_0 * np.exp(-(h - H_0_REF) / SCALE_HEIGHT)


def solar_factor(f107):
    """Linear EUV heating factor, clamped below at 0.2."""
    return np.maximum(SOLAR_FACTOR_MIN,
                      1.0 + F107_SENSITIVITY * (np.asarray(f107, dtype=float) - F107_BASELINE))


def geomagnetic_factor(ap):
    """Linear geomagnetic heating factor, clamped below at 0.5."""
    return np.maximum(GEOMAG_FACTOR_MIN,
                      1.0 + AP_SENSITIVITY * (np.asarray(ap, dtype=float) - AP_BASELINE))


def thermospheric_density(a_m, f107, ap):
    """Neutral mass density rho(a, F10.7, Ap) in kg/m^3."""
    rho = barometric_density(a_m) * solar_factor(f107) * geomagnetic_factor(ap)
    return np.maximum(rho, RHO_FLOOR)
