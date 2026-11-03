"""Scene payload for the 3D view: display geometry built from engine output.

Nothing here integrates the decay ODE. Radius, decay rate and density come
straight from the engine's results; this module only adds display geometry:
  * Sun direction (low-precision almanac formula, ~0.01 deg) and GMST for
    Earth rotation and the day/night terminator;
  * orbit orientation from TLE elements, with optional J2 secular precession
    of RAAN and argument of perigee -- DISPLAY-ONLY, not from the engine;
  * along-track phase from the engine's a(t) via n = sqrt(mu / a^3);
  * the thermosphere iso-density shell, from the engine's own density model;
  * aurora intensity from Ap, and CME tracks from the event log.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from leo_decay.constants import H_0_REF, MU_EARTH, R_EARTH, RHO_0, SCALE_HEIGHT
from leo_decay.density import geomagnetic_factor, solar_factor

ROOT = Path(__file__).resolve().parents[1]
TLE_PATH = ROOT / "config" / "tle.toml"
TLE_EXAMPLE_PATH = ROOT / "config" / "tle.example.toml"

J2 = 1.08262668e-3                  # Earth oblateness (display-only precession)
OMEGA_EARTH = 7.2921158553e-5       # rad/s, Earth rotation rate
SHELL_RHO_REF = 1.0e-12             # kg/m^3, iso-density surface drawn as the shell
GEOMAG_POLE_LAT_DEG = 80.8          # tilted-dipole north pole (approx., placeholder)
GEOMAG_POLE_LON_DEG = -72.7
AURORA_COLAT_DEG = 23.0             # oval ~67 deg geomagnetic latitude
AURORA_AP_THRESHOLD = 30.0
AURORA_AP_FULL = 150.0              # Ap at which the glow is at full strength
PRECESSION_LABEL = ("J2 precession of RAAN and argument of perigee is display-only; "
                    "it is not computed by the engine.")


# ---------------------------------------------------------------- time & Sun
def julian_date(dt: datetime) -> float:
    dt = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    j2000 = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
    return 2451545.0 + (dt - j2000).total_seconds() / 86400.0


def sun_direction_eci(dt: datetime) -> np.ndarray:
    """Unit vector Earth->Sun, mean equator & equinox of date (Astronomical Almanac)."""
    n = julian_date(dt) - 2451545.0
    L = math.radians((280.460 + 0.9856474 * n) % 360.0)
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2 * g)
    eps = math.radians(23.439 - 0.0000004 * n)
    return np.array([math.cos(lam), math.cos(eps) * math.sin(lam), math.sin(eps) * math.sin(lam)])


def gmst_rad(dt: datetime) -> float:
    """Greenwich mean sidereal time (IAU 1982), radians in [0, 2pi)."""
    d = julian_date(dt) - 2451545.0
    t = d / 36525.0
    deg = 280.46061837 + 360.98564736629 * d + 0.000387933 * t**2 - t**3 / 38710000.0
    return math.radians(deg % 360.0)


# ---------------------------------------------------------------- orbit geometry
def orbit_position_eci(r, raan, inc, u):
    """Circular-orbit position for argument of latitude u (radians)."""
    cO, sO, ci, si = np.cos(raan), np.sin(raan), np.cos(inc), np.sin(inc)
    cu, su = np.cos(u), np.sin(u)
    return np.stack([r * (cO * cu - sO * su * ci), r * (sO * cu + cO * su * ci), r * su * si], axis=-1)


def orbit_normal_eci(raan, inc):
    return np.array([math.sin(inc) * math.sin(raan), -math.sin(inc) * math.cos(raan), math.cos(inc)])


def j2_rates(a_m, e, inc):
    """Secular J2 rates (rad/s) of RAAN and argument of perigee. Display-only."""
    a_m = np.asarray(a_m, dtype=float)
    n = np.sqrt(MU_EARTH / a_m**3)
    k = n * J2 * (R_EARTH / (a_m * (1.0 - e**2))) ** 2
    return -1.5 * k * math.cos(inc), 0.75 * k * (5.0 * math.cos(inc) ** 2 - 1.0)


def _cumtrapz(y, x):
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])


# ---------------------------------------------------------------- thermosphere & aurora
def shell_altitude_m(f107, ap, rho_ref: float = SHELL_RHO_REF):
    """Altitude where the engine's density model equals ``rho_ref`` (inverts Eq. B.3's rho)."""
    return H_0_REF + SCALE_HEIGHT * np.log(RHO_0 * solar_factor(f107) * geomagnetic_factor(ap) / rho_ref)


def aurora_intensity(ap):
    ap = np.asarray(ap, dtype=float)
    return np.where(ap > AURORA_AP_THRESHOLD,
                    np.clip((ap - AURORA_AP_THRESHOLD) / (AURORA_AP_FULL - AURORA_AP_THRESHOLD), 0.05, 1.0),
                    0.0)


# ---------------------------------------------------------------- TLE elements
@dataclass(frozen=True)
class Elements:
    norad_id: int
    epoch: datetime
    inclination: float      # rad
    raan: float             # rad
    eccentricity: float
    arg_perigee: float      # rad
    mean_anomaly: float     # rad
    mean_motion_rad_s: float
    source: str
    placeholder: bool


def load_elements(sat_key: str, path: Path | None = None) -> Elements:
    """Elements from config/tle.toml, falling back to the placeholder example file."""
    use = path or (TLE_PATH if TLE_PATH.exists() else TLE_EXAMPLE_PATH)
    with open(use, "rb") as fh:
        data = tomllib.load(fh)
    if sat_key not in data:
        raise KeyError(f"No TLE elements for '{sat_key}' in {use.name}")
    d = data[sat_key]
    placeholder = use.name == TLE_EXAMPLE_PATH.name
    return Elements(
        norad_id=int(d["norad_id"]),
        epoch=datetime.fromisoformat(d["epoch_utc"].replace("Z", "+00:00")),
        inclination=math.radians(d["inclination_deg"]), raan=math.radians(d["raan_deg"]),
        eccentricity=float(d["eccentricity"]), arg_perigee=math.radians(d["arg_perigee_deg"]),
        mean_anomaly=math.radians(d["mean_anomaly_deg"]),
        mean_motion_rad_s=float(d["mean_motion_rev_day"]) * 2 * math.pi / 86400.0,
        source=("PLACEHOLDER elements (config/tle.example.toml)" if placeholder
                else f"{d.get('source', 'Space-Track.org')}, epoch {d['epoch_utc'][:16]} UTC"),
        placeholder=placeholder,
    )


# ---------------------------------------------------------------- CMEs
def _cme_direction(event_id: str) -> tuple[float, float]:
    """Deterministic (offset from Sun->Earth line, azimuth) in degrees for a non-Earth-directed CME."""
    h = hashlib.sha256(event_id.encode()).digest()
    return 50.0 + 100.0 * h[0] / 255.0, 360.0 * h[1] / 255.0


def cme_tracks(events: pd.DataFrame | None, lag_fn=None) -> list[dict]:
    """Launch/arrival days per CME. Earth-directed: arrival = its storm's onset in the log."""
    if events is None or events.empty:
        return []
    storms = events[events["type"] == "storm"].set_index("parent_id")["start_day"]
    out = []
    for row in events[events["type"] == "cme"].itertuples(index=False):
        speed = float(str(row.class_or_speed).split()[0])
        directed = bool(row.earth_directed)
        if directed and row.event_id in storms.index:
            arrival = float(storms.loc[row.event_id])
        else:
            lag_h = float(lag_fn(speed)) if lag_fn else 1.496e8 / speed / 3600.0
            arrival = float(row.start_day) + lag_h / 24.0
        offset, az = (0.0, 0.0) if directed else _cme_direction(row.event_id)
        out.append({"id": row.event_id, "launch_day": float(row.start_day), "arrival_day": arrival,
                    "speed_km_s": speed, "earth_directed": directed,
                    "offset_deg": offset, "azimuth_deg": az})
    return out


# ---------------------------------------------------------------- payload
def build_payload(frame: pd.DataFrame, forcing: pd.DataFrame, events: pd.DataFrame | None,
                  elements: Elements, start_utc: datetime, sat_name: str, run_id: str,
                  lag_fn=None, demo: bool = False) -> dict:
    """Everything the three.js view needs. Times are seconds since ``start_utc``."""
    start_utc = start_utc if start_utc.tzinfo else start_utc.replace(tzinfo=timezone.utc)
    t = frame["day"].to_numpy() * 86400.0
    h = frame["altitude_km"].to_numpy() * 1000.0                    # engine output, unchanged
    a = R_EARTH + h
    n = np.sqrt(MU_EARTH / a**3)                                     # from engine a(t)

    el = elements
    dt_epoch = (start_utc - el.epoch).total_seconds()                # start relative to TLE epoch
    # Elements are carried from the TLE epoch to the run start with the TLE's own a.
    a_tle = (MU_EARTH / el.mean_motion_rad_s**2) ** (1.0 / 3.0)
    raan_dot0, argp_dot0 = j2_rates(a_tle, el.eccentricity, el.inclination)
    m_start = el.mean_anomaly + el.mean_motion_rad_s * dt_epoch
    raan_dot, argp_dot = j2_rates(a, el.eccentricity, el.inclination)
    raan_prec = el.raan + raan_dot0 * dt_epoch + _cumtrapz(raan_dot, t)
    argp_prec = el.arg_perigee + argp_dot0 * dt_epoch + _cumtrapz(argp_dot, t)

    f_t = forcing["day"].to_numpy() * 86400.0
    f107, ap = forcing["f107_sfu"].to_numpy(dtype=float), forcing["ap"].to_numpy(dtype=float)
    sun_days = np.linspace(0.0, t[-1], max(2, int(t[-1] / 86400.0) + 2))
    sun = np.array([sun_direction_eci(start_utc + timedelta(seconds=float(s))) for s in sun_days])

    rate = frame["decay_rate_km_day"].to_numpy()
    rho = frame["density_kg_m3"].to_numpy()
    shell_h = shell_altitude_m(f107, ap)
    return {
        "run_id": run_id, "demo": bool(demo),
        "sat_name": sat_name, "norad_id": el.norad_id,
        "start_utc": start_utc.isoformat(), "duration_s": float(t[-1]),
        "R_E_m": R_EARTH, "omega_earth": OMEGA_EARTH, "gmst0": gmst_rad(start_utc),
        "elements": {"inclination": el.inclination, "eccentricity": el.eccentricity,
                     "raan0": el.raan, "argp0": el.arg_perigee,
                     "source": el.source, "placeholder": el.placeholder},
        "precession_label": PRECESSION_LABEL,
        "t_s": t.tolist(), "h_m": h.tolist(), "mean_anomaly": (m_start + _cumtrapz(n, t)).tolist(),
        "raan_prec": raan_prec.tolist(), "argp_prec": argp_prec.tolist(),
        "decay_rate_km_day": rate.tolist(), "density": rho.tolist(),
        "rate_range": [float(np.abs(rate).min()), float(np.abs(rate).max())],
        "density_range": [float(rho.min()), float(rho.max())],
        "forcing_t_s": f_t.tolist(), "f107": f107.tolist(), "ap": ap.tolist(),
        "shell_h_m": shell_h.tolist(), "shell_rho_ref": SHELL_RHO_REF,
        "aurora": aurora_intensity(ap).tolist(),
        "aurora_pole": {"lat_deg": GEOMAG_POLE_LAT_DEG, "lon_deg": GEOMAG_POLE_LON_DEG,
                        "colat_deg": AURORA_COLAT_DEG},
        "sun_t_s": (sun_days).tolist(), "sun_eci": sun.tolist(),
        "cmes": cme_tracks(events, lag_fn),
    }


def payload_json_size(payload: dict) -> int:
    return len(json.dumps(payload))
