"""Compare propagator forecasts with observed TLE altitudes (Protocol 3 check).

Observed altitude follows Appendix A.1 exactly: raw TLE mean motion -> a via
Kepler's third law -> h = a - R_E, with the 2 km PCHIP glitch filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import SECONDS_PER_DAY
from .glitch_cleaning import clean_glitches
from .propagator import propagate
from .tle_conversion import add_orbital_state


def observed_altitudes(gp: pd.DataFrame) -> pd.DataFrame:
    """EPOCH (UTC) and glitch-cleaned altitude_m from GP records."""
    df = add_orbital_state(gp[["EPOCH", "MEAN_MOTION"]].copy())
    df["altitude_raw_m"] = df["altitude_m"]
    df["altitude_m"] = clean_glitches(df["altitude_m"])
    return df[["EPOCH", "altitude_m", "altitude_raw_m"]]


def _seconds(epochs, t0: pd.Timestamp) -> np.ndarray:
    return (pd.to_datetime(epochs, utc=True) - t0).dt.total_seconds().to_numpy() \
        if isinstance(epochs, pd.Series) else \
        np.array([(pd.Timestamp(e) - t0).total_seconds() for e in epochs])


def altitude_at(obs: pd.DataFrame, when: pd.Timestamp, t0: pd.Timestamp) -> dict:
    """Linear interpolation between the bracketing TLEs, plus the nearest TLE."""
    t = _seconds(obs["EPOCH"], t0)
    x = (when - t0).total_seconds()
    h = obs["altitude_m"].to_numpy()
    if not (t[0] <= x <= t[-1]):
        raise ValueError(f"{when} is outside the observed epochs {obs['EPOCH'].iloc[0]} .. "
                         f"{obs['EPOCH'].iloc[-1]}")
    i = int(np.argmin(np.abs(t - x)))
    hi = int(np.searchsorted(t, x))
    lo = max(hi - 1, 0)
    return {
        "interp_m": float(np.interp(x, t, h)),
        "nearest_m": float(h[i]),
        "nearest_epoch": obs["EPOCH"].iloc[i],
        "bracket_gap_h": float((t[min(hi, len(t) - 1)] - t[lo]) / 3600.0),
    }


def tle_noise_m(obs: pd.DataFrame, t0: pd.Timestamp, window_days: float = 30.0) -> float:
    """Std of altitude about a local quadratic fit: a TLE-to-TLE noise estimate."""
    t = _seconds(obs["EPOCH"], t0) / SECONDS_PER_DAY
    sel = t >= t[-1] - window_days
    coef = np.polyfit(t[sel], obs["altitude_m"].to_numpy()[sel], 2)
    resid = obs["altitude_m"].to_numpy()[sel] - np.polyval(coef, t[sel])
    return float(resid.std(ddof=3))


@dataclass(frozen=True)
class ForecastCheck:
    table: pd.DataFrame             # one row per checkpoint
    h0_paper_m: float
    h0_observed_m: float
    observed_rate_km_day: float
    forecast_rate_km_day: float
    noise_m: float
    n_tles: int


def check_forecast(gp: pd.DataFrame, t0: pd.Timestamp, h0_m: float, b_star: float,
                   f107: float, ap: float, checkpoint_days: list[int],
                   paper_km: dict[int, float] | None = None) -> ForecastCheck:
    """Residuals e = h_pred - h_obs (m) at each available checkpoint."""
    obs = observed_altitudes(gp)
    obs = obs[obs["EPOCH"] >= t0 - pd.Timedelta(days=30)].reset_index(drop=True)
    h0_obs = altitude_at(obs, t0, t0)["interp_m"]

    days = [d for d in checkpoint_days
            if t0 + pd.Timedelta(days=d) <= obs["EPOCH"].iloc[-1]]
    t_eval = np.array([0.0] + [d * SECONDS_PER_DAY for d in days])
    span = t_eval[-1] if len(days) else SECONDS_PER_DAY
    pred = propagate(h0_m, span, b_star, f107, ap, t_eval_s=t_eval)
    anchored = propagate(h0_obs, span, b_star, f107, ap, t_eval_s=t_eval)

    rows = []
    for k, d in enumerate(days, start=1):
        when = t0 + pd.Timedelta(days=d)
        o = altitude_at(obs, when, t0)
        h_pred = float(pred.h_m[k])
        rows.append({
            "checkpoint_utc": when, "day": d,
            "h_pred_m": h_pred,
            "h_paper_m": (paper_km or {}).get(d, np.nan) * 1000.0,
            "h_obs_interp_m": o["interp_m"],
            "h_obs_nearest_m": o["nearest_m"],
            "nearest_epoch": o["nearest_epoch"],
            "bracket_gap_h": o["bracket_gap_h"],
            "residual_m": h_pred - o["interp_m"],
            "residual_nearest_m": h_pred - o["nearest_m"],
            "residual_anchored_m": float(anchored.h_m[k]) - o["interp_m"],
        })

    t_days = _seconds(obs["EPOCH"], t0) / SECONDS_PER_DAY
    after = t_days >= 0
    rate = float(np.polyfit(t_days[after], obs["altitude_m"].to_numpy()[after], 1)[0]) / 1000.0
    f_rate = float((pred.h_m[-1] - pred.h_m[0]) / (span / SECONDS_PER_DAY)) / 1000.0
    return ForecastCheck(
        table=pd.DataFrame(rows), h0_paper_m=h0_m, h0_observed_m=h0_obs,
        observed_rate_km_day=rate, forecast_rate_km_day=f_rate,
        noise_m=tle_noise_m(obs, t0), n_tles=int(after.sum()),
    )
