"""F10.7-dependent Poisson event times via thinning (Lewis-Shedler)."""

from __future__ import annotations

import numpy as np

# F10.7 driver: a constant (sfu) or a (t_days, f107) pair, linearly interpolated.
F107Driver = float | tuple[np.ndarray, np.ndarray]


def f107_at(t_days, f107: F107Driver):
    if isinstance(f107, tuple):
        return np.interp(t_days, f107[0], f107[1])
    return np.full(np.shape(t_days), float(f107))


def f107_max(f107: F107Driver) -> float:
    return float(np.max(f107[1])) if isinstance(f107, tuple) else float(f107)


def power_rate(f, base_per_day: float, ref_f107: float, exponent: float):
    """rate(F) = base * (F / ref)^exponent, events/day."""
    return base_per_day * (np.asarray(f, dtype=float) / ref_f107) ** exponent


def sample_times(rng: np.random.Generator, t_end_days: float, f107: F107Driver,
                 base_per_day: float, ref_f107: float, exponent: float) -> np.ndarray:
    """Event times (days) on [0, t_end) for rate(F10.7(t)), by thinning.

    Exact for any driver whose maximum is known: propose at the maximum rate,
    accept with probability rate(t) / max rate.
    """
    lam_max = float(power_rate(f107_max(f107), base_per_day, ref_f107, exponent))
    if lam_max <= 0.0 or t_end_days <= 0.0:
        return np.empty(0)
    n = rng.poisson(lam_max * t_end_days)
    t = np.sort(rng.uniform(0.0, t_end_days, n))
    lam = power_rate(f107_at(t, f107), base_per_day, ref_f107, exponent)
    keep = rng.uniform(0.0, 1.0, n) < lam / lam_max
    return t[keep]
