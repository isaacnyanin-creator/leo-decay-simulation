"""Empirical ballistic-coefficient calibration, Eq. B.4 (Appendix A.3).

B* = argmin_{B* in [1e-6, 0.1]} sum_i [ (da/dt)_obs,i - (-B* rho(a_i, F_i, Ap_i) sqrt(mu a_i)) ]^2

solved with bounded Trust-Region Reflective least squares (scipy curve_fit, method='trf').
All inputs SI: a in m, da/dt in m/s, B* in m^2/kg.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from .constants import MU_EARTH
from .density import thermospheric_density

B_STAR_SEED = 2.297e-3          # m^2/kg, Hubble static estimate (§3.4.2)
B_STAR_BOUNDS = (1.0e-6, 0.10)  # m^2/kg


@dataclass(frozen=True)
class CalibrationResult:
    b_star: float           # m^2/kg
    b_star_sigma: float     # m^2/kg, 1-sigma from the covariance diagonal
    n_obs: int


def model_decay_rate(state, b_star):
    """Right-hand side of Eq. B.1 for observation tuples ``(a_m, f107, ap)``; m/s."""
    a_m, f107, ap = state
    a_m = np.asarray(a_m, dtype=float)
    return -b_star * thermospheric_density(a_m, f107, ap) * np.sqrt(MU_EARTH * a_m)


def calibrate_b_star(a_obs_m, dadt_obs_m_s, f107_obs, ap_obs,
                     p0: float = B_STAR_SEED,
                     bounds: tuple[float, float] = B_STAR_BOUNDS) -> CalibrationResult:
    """Fit B* to observed decay rates. NaN observations are dropped first."""
    a = np.asarray(a_obs_m, dtype=float)
    d = np.asarray(dadt_obs_m_s, dtype=float)
    f = np.broadcast_to(np.asarray(f107_obs, dtype=float), a.shape)
    g = np.broadcast_to(np.asarray(ap_obs, dtype=float), a.shape)

    ok = np.isfinite(a) & np.isfinite(d) & np.isfinite(f) & np.isfinite(g)
    if ok.sum() < 2:
        raise ValueError("Need at least two finite observations to calibrate B*.")

    # SI decay rates are ~1e-4 m/s, so the raw cost (~1e-8) sits below the
    # optimiser's gradient tolerance and TRF stops early. Dividing data and
    # model by one constant leaves the Eq. B.4 argmin and the B* covariance
    # unchanged (absolute_sigma=False rescales pcov by the residual variance).
    scale = float(np.sqrt(np.mean(d[ok] ** 2))) or 1.0

    def scaled_model(state, b_star):
        return model_decay_rate(state, b_star) / scale

    popt, pcov = curve_fit(
        scaled_model, (a[ok], f[ok], g[ok]), d[ok] / scale,
        p0=[p0], bounds=([bounds[0]], [bounds[1]]), method="trf",
    )
    return CalibrationResult(
        b_star=float(popt[0]),
        b_star_sigma=float(np.sqrt(np.diag(pcov))[0]),
        n_obs=int(ok.sum()),
    )
