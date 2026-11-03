"""CMEs and the geomagnetic storms caused by Earth-directed ones."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CmeConfig, StormConfig
from .rates import F107Driver, sample_times


def sample_speeds(rng: np.random.Generator, n: int, cfg: CmeConfig) -> np.ndarray:
    v = cfg.speed_median_km_s * np.exp(rng.normal(0.0, cfg.speed_sigma_ln, n))
    return np.clip(v, cfg.speed_min_km_s, cfg.speed_max_km_s)


def sample_cmes(rng: np.random.Generator, t_end_days: float, f107: F107Driver,
                cfg: CmeConfig) -> pd.DataFrame:
    """CMEs on [0, t_end). Columns: start_day, speed_km_s, earth_directed."""
    t = sample_times(rng, t_end_days, f107, cfg.base_rate_per_day,
                     cfg.rate_reference_f107, cfg.rate_f107_exponent)
    speed = sample_speeds(rng, t.size, cfg)
    directed = rng.uniform(0.0, 1.0, t.size) < cfg.earth_directed_probability
    return pd.DataFrame({"start_day": t, "speed_km_s": speed, "earth_directed": directed})


def transit_lag_hours(speed_km_s, cfg: StormConfig):
    """Lag = 1 AU / speed, clipped to [lag_min_hours, lag_max_hours]."""
    lag = cfg.transit_distance_km / np.asarray(speed_km_s, dtype=float) / 3_600.0
    return np.clip(lag, cfg.lag_min_hours, cfg.lag_max_hours)


def peak_ap(speed_km_s, cfg: StormConfig):
    raw = cfg.peak_ap_at_reference_speed + cfg.peak_ap_per_km_s * (
        np.asarray(speed_km_s, dtype=float) - cfg.reference_speed_km_s)
    return np.clip(raw, cfg.peak_ap_min, cfg.ap_cap)


def storms_from_cmes(cmes: pd.DataFrame, cfg: StormConfig) -> pd.DataFrame:
    """One storm per Earth-directed CME. ``parent_index`` is the CME's row index."""
    d = cmes[cmes["earth_directed"]]
    lag_h = transit_lag_hours(d["speed_km_s"].to_numpy(), cfg)
    return pd.DataFrame({
        "parent_index": d.index.to_numpy(),
        "start_day": d["start_day"].to_numpy() + lag_h / 24.0,
        "lag_h": lag_h,
        "speed_km_s": d["speed_km_s"].to_numpy(),
        "peak_ap": peak_ap(d["speed_km_s"].to_numpy(), cfg),
        "duration_h": np.full(len(d), cfg.main_phase_hours
                              + cfg.duration_efolds * cfg.recovery_efold_hours),
    })


def ap_profile(t_days: np.ndarray, storms: pd.DataFrame, cfg: StormConfig,
               n_efolds_cut: float = 15.0) -> np.ndarray:
    """Ap(t): quiet baseline plus each storm's excess (linear main phase, exponential
    recovery). Overlapping storms add their excesses; the total is capped."""
    t_days = np.asarray(t_days, dtype=float)
    excess = np.zeros_like(t_days)
    main = cfg.main_phase_hours / 24.0
    tau = cfg.recovery_efold_hours / 24.0
    for row in storms.itertuples(index=False):
        amp = row.peak_ap - cfg.quiet_ap
        t0, t_peak = row.start_day, row.start_day + main
        lo = np.searchsorted(t_days, t0)
        hi = np.searchsorted(t_days, t_peak + n_efolds_cut * tau, side="right")
        seg = t_days[lo:hi]
        rising = seg < t_peak
        bump = np.where(rising, amp * (seg - t0) / main if main > 0 else amp,
                        amp * np.exp(-(seg - t_peak) / tau))
        excess[lo:hi] += bump
    return np.minimum(cfg.quiet_ap + excess, cfg.ap_cap)
