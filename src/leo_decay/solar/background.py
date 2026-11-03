"""Background F10.7: solar-cycle trend + 27-day rotation + AR(1) red noise."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from .config import BackgroundConfig


def cycle_trend(t_days, start_utc: datetime, cfg: BackgroundConfig):
    """sin^2-shaped cycle: f107_min at cycle minimum, f107_max half a period later."""
    offset = (start_utc - cfg.cycle_min_utc).total_seconds() / 86_400.0
    phase = (np.asarray(t_days, dtype=float) + offset) / cfg.cycle_period_days
    return cfg.f107_min + (cfg.f107_max - cfg.f107_min) * np.sin(np.pi * phase) ** 2


def rotation_amplitude(trend, cfg: BackgroundConfig):
    return cfg.rotation_amplitude_min_sfu + cfg.rotation_amplitude_frac * (trend - cfg.f107_min)


def ar1_noise(rng: np.random.Generator, n: int, dt_days: float, cfg: BackgroundConfig):
    """Stationary AR(1) series with std ``noise_sigma_sfu`` and e-folding ``noise_correlation_days``."""
    if cfg.noise_sigma_sfu == 0 or n == 0:
        return np.zeros(n)
    phi = np.exp(-dt_days / cfg.noise_correlation_days)
    innov = rng.normal(0.0, cfg.noise_sigma_sfu * np.sqrt(1.0 - phi**2), n)
    out = np.empty(n)
    out[0] = rng.normal(0.0, cfg.noise_sigma_sfu)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + innov[i]
    return out


def generate_background(rng: np.random.Generator, t_days: np.ndarray,
                        start_utc: datetime, cfg: BackgroundConfig) -> np.ndarray:
    """Background F10.7 (sfu) on a uniform grid ``t_days`` (days since ``start_utc``)."""
    t_days = np.asarray(t_days, dtype=float)
    dt = float(t_days[1] - t_days[0]) if t_days.size > 1 else 1.0
    trend = cycle_trend(t_days, start_utc, cfg)
    rot_phase = rng.uniform(0.0, 2.0 * np.pi)
    rotation = rotation_amplitude(trend, cfg) * np.sin(
        2.0 * np.pi * t_days / cfg.rotation_period_days + rot_phase)
    noise = ar1_noise(rng, t_days.size, dt, cfg)
    return np.maximum(trend + rotation + noise, cfg.f107_floor)
