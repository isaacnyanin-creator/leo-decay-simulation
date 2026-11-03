"""Solar flares: F10.7-dependent occurrence, power-law size spectrum, F10.7 bumps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FlareConfig
from .rates import F107Driver, sample_times

CLASS_THRESHOLDS = (("X", 1e-4), ("M", 1e-5), ("C", 1e-6))   # GOES 1-8 A, W/m^2


def flux_class_letter(flux) -> np.ndarray:
    flux = np.asarray(flux, dtype=float)
    return np.select([flux >= 1e-4, flux >= 1e-5, flux >= 1e-6], ["X", "M", "C"], default="B")


def flux_class_label(flux: float) -> str:
    """e.g. 2.3e-5 -> 'M2.3'."""
    for letter, base in CLASS_THRESHOLDS:
        if flux >= base:
            return f"{letter}{flux / base:.1f}"
    return f"B{flux / 1e-7:.1f}"


def sample_flux(rng: np.random.Generator, n: int, cfg: FlareConfig) -> np.ndarray:
    """Inverse-CDF sample of a truncated power law dN/dS ~ S^-alpha."""
    k = 1.0 - cfg.power_law_index
    lo, hi = cfg.min_flux_w_m2**k, cfg.max_flux_w_m2**k
    return (lo + rng.uniform(0.0, 1.0, n) * (hi - lo)) ** (1.0 / k)


def exceedance_probability(s: float, cfg: FlareConfig) -> float:
    """P(S >= s) under the truncated power law."""
    k = 1.0 - cfg.power_law_index
    s = min(max(s, cfg.min_flux_w_m2), cfg.max_flux_w_m2)
    return (s**k - cfg.max_flux_w_m2**k) / (cfg.min_flux_w_m2**k - cfg.max_flux_w_m2**k)


def delta_f107(flux, cfg: FlareConfig):
    return cfg.delta_f107_at_c1 * (np.asarray(flux, dtype=float) / 1e-6) ** cfg.delta_f107_flux_exponent


def sample_flares(rng: np.random.Generator, t_end_days: float, f107: F107Driver,
                  cfg: FlareConfig) -> pd.DataFrame:
    """Flares on [0, t_end). Columns: start_day, flux_w_m2, class_letter, class_label,
    rise_h, decay_efold_h, duration_h, delta_f107."""
    t = sample_times(rng, t_end_days, f107, cfg.base_rate_per_day,
                     cfg.rate_reference_f107, cfg.rate_f107_exponent)
    n = t.size
    flux = sample_flux(rng, n, cfg)
    tau = cfg.decay_efold_hours_median * np.exp(rng.normal(0.0, cfg.decay_efold_hours_sigma_ln, n))
    return pd.DataFrame({
        "start_day": t,
        "flux_w_m2": flux,
        "class_letter": flux_class_letter(flux),
        "class_label": [flux_class_label(s) for s in flux],
        "rise_h": np.full(n, cfg.rise_hours),
        "decay_efold_h": tau,
        "duration_h": cfg.rise_hours + cfg.duration_efolds * tau,
        "delta_f107": delta_f107(flux, cfg),
    })


def flare_profile(t_days: np.ndarray, flares: pd.DataFrame, n_efolds_cut: float = 10.0) -> np.ndarray:
    """Sum of flare bumps (sfu) on grid ``t_days``: linear rise, exponential decay."""
    t_days = np.asarray(t_days, dtype=float)
    out = np.zeros_like(t_days)
    for row in flares.itertuples(index=False):
        rise = row.rise_h / 24.0
        tau = row.decay_efold_h / 24.0
        t0, t_peak = row.start_day, row.start_day + rise
        lo = np.searchsorted(t_days, t0)
        hi = np.searchsorted(t_days, t_peak + n_efolds_cut * tau, side="right")
        seg = t_days[lo:hi]
        rising = seg < t_peak
        bump = np.where(
            rising,
            row.delta_f107 * (seg - t0) / rise if rise > 0 else row.delta_f107,
            row.delta_f107 * np.exp(-(seg - t_peak) / tau),
        )
        out[lo:hi] += bump
    return out
