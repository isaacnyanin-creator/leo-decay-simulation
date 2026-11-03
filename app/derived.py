"""Derived quantities for the dashboard, built on the unchanged ``leo_decay`` engine.

Pure functions only (no Streamlit), so they can be unit-tested. SI internally;
``simulate`` converts to display units at the end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from leo_decay.config import Satellite
from leo_decay.constants import MU_EARTH, RHO_FLOOR, SCALE_HEIGHT, SECONDS_PER_DAY
from leo_decay.density import thermospheric_density
from leo_decay.propagator import propagate

M_S_TO_KM_DAY = SECONDS_PER_DAY / 1_000.0
M_S2_TO_KM_DAY2 = SECONDS_PER_DAY**2 / 1_000.0


class SatelliteNotCalibrated(ValueError):
    """Raised for a config entry without mass and calibrated B*."""


def is_simulatable(sat: Satellite) -> bool:
    return sat.mass_kg is not None and sat.b_star_empirical_m2_kg is not None


def rescale_b_star(b_star_cal: float, m_ref_kg: float, m_new_kg: float) -> float:
    """B*_new = B*_cal * m_ref / m_new  (C_D * A held fixed)."""
    if m_ref_kg <= 0 or m_new_kg <= 0:
        raise ValueError("Masses must be positive.")
    return b_star_cal * m_ref_kg / m_new_kg


def decay_rate_m_s(a_m, b_star, f107, ap):
    """Eq. B.1: da/dt = -B* rho sqrt(mu a), in m/s."""
    a_m = np.asarray(a_m, dtype=float)
    return -b_star * thermospheric_density(a_m, f107, ap) * np.sqrt(MU_EARTH * a_m)


def decay_acceleration_m_s2(a_m, b_star, f107, ap):
    """Eq. B.2: d2a/dt2 = -(B*)^2 mu a rho^2 (1/H - 1/(2a)), in m/s^2.

    Valid for constant forcing. Where density sits on the 1e-18 floor its
    gradient is zero, so only the sqrt(a) term remains:
    d2a/dt2 = (da/dt)^2 / (2a).
    """
    a_m = np.asarray(a_m, dtype=float)
    rho = thermospheric_density(a_m, f107, ap)
    b2 = -(b_star**2) * MU_EARTH * a_m * rho**2 * (1.0 / SCALE_HEIGHT - 1.0 / (2.0 * a_m))
    floored = (b_star**2) * MU_EARTH * a_m * rho**2 / (2.0 * a_m)
    return np.where(rho <= RHO_FLOOR, floored, b2)


def drag_force_n(a_m, b_star, mass_kg, f107, ap):
    """F_D = 1/2 rho v^2 C_D A, with v^2 = mu/a and C_D A = B* m. Newtons."""
    a_m = np.asarray(a_m, dtype=float)
    rho = thermospheric_density(a_m, f107, ap)
    return 0.5 * rho * (MU_EARTH / a_m) * b_star * mass_kg


@dataclass(frozen=True)
class SimulationResult:
    frame: pd.DataFrame     # display units, one row per sample
    b_star: float           # m^2/kg actually used
    reentered: bool
    reentry_day: float | None
    forcing: pd.DataFrame | None = None     # day, f107_sfu, ap (both modes)
    events: pd.DataFrame | None = None      # solar event log (generated mode only)


def _frame(traj, b_star, mass_kg, f107, ap) -> pd.DataFrame:
    a = traj.a_m
    return pd.DataFrame({
        "day": traj.t_days,
        "altitude_km": traj.h_m / 1_000.0,
        "decay_rate_km_day": decay_rate_m_s(a, b_star, f107, ap) * M_S_TO_KM_DAY,
        "decay_accel_km_day2": decay_acceleration_m_s2(a, b_star, f107, ap) * M_S2_TO_KM_DAY2,
        "density_kg_m3": thermospheric_density(a, f107, ap),
        "drag_force_n": drag_force_n(a, b_star, mass_kg, f107, ap),
    })


def _b_star_for(sat: Satellite, mass_kg: float) -> float:
    if not is_simulatable(sat):
        raise SatelliteNotCalibrated(f"{sat.name} has no mass / calibrated B* in the config.")
    return rescale_b_star(sat.b_star_empirical_m2_kg, sat.mass_kg, mass_kg)


def simulate(sat: Satellite, mass_kg: float, f107: float, ap: float,
             h0_m: float, days: float, n_points: int = 500) -> SimulationResult:
    """Propagate ``sat`` with mass-rescaled B* and return all five dashboard series."""
    b_star = _b_star_for(sat, mass_kg)
    duration_s = days * SECONDS_PER_DAY
    traj = propagate(h0_m, duration_s, b_star, f107, ap,
                     t_eval_s=np.linspace(0.0, duration_s, n_points))
    frame = _frame(traj, b_star, mass_kg, f107, ap)
    reentry_day = None
    if traj.reentered:
        # With t_eval the event time itself isn't sampled; without t_eval
        # solve_ivp's final step lands exactly on the terminal event.
        reentry_day = float(propagate(h0_m, duration_s, b_star, f107, ap).t_days[-1])
    forcing = pd.DataFrame({"day": frame["day"], "f107_sfu": float(f107), "ap": float(ap)})
    return SimulationResult(frame=frame, b_star=b_star, reentered=traj.reentered,
                            reentry_day=reentry_day, forcing=forcing)


def simulate_scenario(sat: Satellite, mass_kg: float, scenario, h0_m: float,
                      f107_mode: str = "30ma", n_points: int = 500) -> SimulationResult:
    """Propagate through a generated ``SolarScenario`` (time-varying F10.7 and Ap).

    Note: the acceleration column is Eq. B.2, i.e. the barometric term only. With
    time-varying forcing the true d2a/dt2 also has a d(rho)/dt term, which spikes
    at storm onsets.
    """
    from leo_decay.solar import propagate_scenario

    b_star = _b_star_for(sat, mass_kg)
    t_eval = np.linspace(0.0, scenario.duration_s, n_points)
    traj = propagate_scenario(scenario, h0_m, b_star, f107_mode, t_eval_s=t_eval)
    f_series = scenario.f107(f107_mode)
    f_t = np.interp(traj.t_s, scenario.t_s, f_series)
    ap_t = np.interp(traj.t_s, scenario.t_s, scenario.ap)
    frame = _frame(traj, b_star, mass_kg, f_t, ap_t)

    reentry_day = None
    if traj.reentered:
        # Re-run from the last sample without t_eval so the final step lands on the event.
        t0 = float(traj.t_s[-1])
        tail = propagate(
            float(traj.h_m[-1]), scenario.duration_s - t0, b_star,
            lambda t: float(np.interp(t + t0, scenario.t_s, f_series)),
            lambda t: float(np.interp(t + t0, scenario.t_s, scenario.ap)),
        )
        reentry_day = (t0 + float(tail.t_s[-1])) / SECONDS_PER_DAY

    forcing = pd.DataFrame({"day": scenario.t_days, "f107_sfu": f_series, "ap": scenario.ap})
    return SimulationResult(frame=frame, b_star=b_star, reentered=traj.reentered,
                            reentry_day=reentry_day, forcing=forcing, events=scenario.events)
