"""3D-view geometry (app/scene.py) and its JavaScript mirror (geometry.js)."""

import json
import math
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leo_decay.constants import R_EARTH
from leo_decay.density import thermospheric_density
from scene import (
    AURORA_AP_THRESHOLD, Elements, aurora_intensity, build_payload, cme_tracks, gmst_rad,
    j2_rates, load_elements, orbit_normal_eci, orbit_position_eci, shell_altitude_m,
    sun_direction_eci,
)

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_JS = ROOT / "app" / "components" / "orbit3d" / "geometry.js"
UTC = timezone.utc


# ---------------------------------------------------------------- Sun & sidereal time
def test_sun_declination_at_solstices_and_equinox():
    dec = lambda dt: math.degrees(math.asin(sun_direction_eci(dt)[2]))
    assert dec(datetime(2026, 6, 21, 8, 24, tzinfo=UTC)) == pytest.approx(23.44, abs=0.02)
    assert dec(datetime(2026, 12, 21, 20, 50, tzinfo=UTC)) == pytest.approx(-23.44, abs=0.02)
    assert dec(datetime(2026, 3, 20, 14, 46, tzinfo=UTC)) == pytest.approx(0.0, abs=0.02)
    v = sun_direction_eci(datetime(2026, 10, 15, tzinfo=UTC))
    assert np.linalg.norm(v) == pytest.approx(1.0)


def test_gmst_reference_values():
    assert math.degrees(gmst_rad(datetime(2000, 1, 1, 12, tzinfo=UTC))) == pytest.approx(280.46061837, abs=1e-6)
    # Vallado, Fundamentals of Astrodynamics, Example 3-5: 1992-08-20 12:14 UT1 -> 152.578788 deg
    assert math.degrees(gmst_rad(datetime(1992, 8, 20, 12, 14, tzinfo=UTC))) == pytest.approx(152.578788, abs=1e-4)


def test_local_noon_faces_sun():
    """At ~12:00 UTC on an equinox, longitude ~0 faces the Sun (equation of time ~ -7 min)."""
    dt = datetime(2026, 3, 20, 12, 7, tzinfo=UTC)
    theta = gmst_rad(dt)
    greenwich = np.array([math.cos(theta), math.sin(theta), 0.0])
    assert np.dot(greenwich, sun_direction_eci(dt)) > 0.999


# ---------------------------------------------------------------- orbit geometry
def test_orbit_position_lies_in_plane_with_radius():
    raan, inc = math.radians(123.0), math.radians(28.47)
    u = np.linspace(0, 2 * np.pi, 50)
    pos = orbit_position_eci(7.0e6, raan, inc, u)
    np.testing.assert_allclose(np.linalg.norm(pos, axis=1), 7.0e6)
    np.testing.assert_allclose(pos @ orbit_normal_eci(raan, inc), 0.0, atol=1e-6)
    assert math.degrees(math.asin(pos[:, 2].max() / 7.0e6)) == pytest.approx(28.47, abs=0.05)


def test_j2_rates_for_hubble():
    a = R_EARTH + 470e3
    raan_dot, argp_dot = j2_rates(a, 0.0002, math.radians(28.47))
    assert math.degrees(raan_dot) * 86400 == pytest.approx(-6.7, abs=0.15)   # deg/day
    assert argp_dot > 0                                                       # i < 63.4 deg
    _, argp_crit = j2_rates(a, 0.0, math.acos(math.sqrt(1 / 5)))
    assert abs(argp_crit) < 1e-15


# ---------------------------------------------------------------- shell & aurora
@pytest.mark.parametrize("f107,ap", [(70.0, 15.0), (145.0, 15.0), (250.0, 120.0)])
def test_shell_is_engine_iso_density(f107, ap):
    h = shell_altitude_m(f107, ap)
    assert thermospheric_density(R_EARTH + h, f107, ap) == pytest.approx(1e-12, rel=1e-12)


def test_shell_rises_with_activity():
    assert shell_altitude_m(250.0, 15.0) > shell_altitude_m(145.0, 15.0) > shell_altitude_m(70.0, 15.0)
    assert shell_altitude_m(145.0, 100.0) > shell_altitude_m(145.0, 15.0)


def test_aurora_only_above_ap_30():
    ap = np.array([0, 15, 30, 30.5, 90, 150, 400])
    a = aurora_intensity(ap)
    assert (a[ap <= AURORA_AP_THRESHOLD] == 0).all()
    assert (a[ap > AURORA_AP_THRESHOLD] > 0).all() and a.max() == 1.0
    assert np.all(np.diff(a[ap > AURORA_AP_THRESHOLD]) >= 0)


# ---------------------------------------------------------------- CMEs & payload
def test_earth_directed_cme_arrives_at_its_storm():
    from leo_decay.solar import generate_scenario
    sc = generate_scenario(60, seed=2026)
    tracks = cme_tracks(sc.events)
    ev = sc.events
    storms = ev[ev["type"] == "storm"].set_index("parent_id")["start_day"]
    directed = [t for t in tracks if t["earth_directed"]]
    assert directed and len(tracks) == (ev["type"] == "cme").sum()
    for t in directed:
        assert t["arrival_day"] == storms.loc[t["id"]]
        assert t["offset_deg"] == 0.0
    others = [t for t in tracks if not t["earth_directed"]]
    assert all(50.0 <= t["offset_deg"] <= 150.0 and t["arrival_day"] > t["launch_day"] for t in others)


def _elements():
    return Elements(norad_id=20580, epoch=datetime(2026, 11, 3, 9, 39, tzinfo=UTC),
                    inclination=math.radians(28.47), raan=math.radians(200.0), eccentricity=0.0002,
                    arg_perigee=math.radians(80.0), mean_anomaly=math.radians(10.0),
                    mean_motion_rad_s=15.27 * 2 * math.pi / 86400, source="test", placeholder=False)


def test_payload_radius_is_engine_output(hubble):
    from derived import simulate
    res = simulate(hubble, hubble.mass_kg, 145.0, 15.0, 469_500.0, 35, n_points=100)
    p = build_payload(res.frame, res.forcing, None, _elements(),
                      datetime(2026, 10, 15, tzinfo=UTC), "HST", "x")
    np.testing.assert_array_equal(p["h_m"], (res.frame["altitude_km"] * 1000).to_numpy())
    assert p["duration_s"] == pytest.approx(35 * 86400)
    m = np.array(p["mean_anomaly"])
    assert np.all(np.diff(m) > 0)
    rev_per_day = (m[-1] - m[0]) / (2 * np.pi) / 35
    assert rev_per_day == pytest.approx(15.2, abs=0.2)
    raan = np.degrees(np.array(p["raan_prec"]))
    assert (raan[-1] - raan[0]) / 35 == pytest.approx(-6.7, abs=0.2)     # display-only J2
    assert p["elements"]["raan0"] == pytest.approx(math.radians(200.0))
    assert "not computed by the engine" in p["precession_label"]
    assert p["cmes"] == [] and set(p["aurora"]) == {0.0}
    json.dumps(p)                                                         # serialisable


def test_load_elements_falls_back_to_placeholder(tmp_path):
    el = load_elements("hubble", ROOT / "config" / "tle.example.toml")
    assert el.placeholder and "PLACEHOLDER" in el.source
    assert el.norad_id == 20580 and el.inclination == pytest.approx(math.radians(28.0))
    with pytest.raises(KeyError):
        load_elements("nope", ROOT / "config" / "tle.example.toml")


# ---------------------------------------------------------------- JS mirror
@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_geometry_js_matches_python():
    raan, inc, u, r = 1.234, math.radians(51.64), 2.71, 1.07
    cases = [
        ["orbitPositionEci", [r, raan, inc, u]],
        ["eciToThree", [[1.0, 2.0, 3.0]]],
        ["displayRadius", [470e3, 10, R_EARTH]],
        ["interp", [[0, 10, 20], [0, 100, 50], 15]],
        ["earthFixedToThreeLocal", [0.0, 0.0]],
        ["earthFixedToThreeLocal", [0.0, 90.0]],
        ["earthFixedToThreeLocal", [90.0, 0.0]],
        ["rotateY", [[1.0, 0.0, 0.0], math.pi / 2]],
        ["cmeDirection", [[1.0, 0.0, 0.0], 0.0, 0.0]],
        ["cmeDirection", [[1.0, 0.0, 0.0], 90.0, 0.0]],
    ]
    out = subprocess.run(["node", str(ROOT / "tests" / "js" / "geometry_cases.mjs"), str(GEOMETRY_JS)],
                         input=json.dumps(cases), capture_output=True, text=True, check=True, timeout=60)
    res = json.loads(out.stdout)
    np.testing.assert_allclose(res[0], orbit_position_eci(r, raan, inc, u), atol=1e-12)
    assert res[1] == [1.0, 3.0, -2.0]                               # ECI -> three (y-up)
    assert res[2] == pytest.approx(1 + 10 * 470e3 / R_EARTH)
    assert res[3] == pytest.approx(75.0)
    # Earth-fixed lon 0 -> +x; lon 90E -> ECI +y -> three -z; north pole -> +y
    np.testing.assert_allclose(res[4], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(res[5], [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(res[6], [0, 1, 0], atol=1e-12)
    # Rotating lon 0 by GMST = 90 deg about +y gives ECI +y (three -z): matches Earth rotation
    np.testing.assert_allclose(res[7], [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(res[8], [-1, 0, 0], atol=1e-12)      # Earth-directed: Sun -> Earth
    assert np.dot(res[9], [-1, 0, 0]) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_earth_rotation_convention_matches_gmst():
    """A ground point at longitude L must sit at ECI angle GMST + L after the Earth group's
    y-rotation by GMST, i.e. JS rotateY(earthFixedToThreeLocal(0, L), GMST) == eciToThree(ECI)."""
    theta, lon = 1.1, 37.0
    cases = [["rotateY", [None, theta]]]
    local = subprocess.run(["node", str(ROOT / "tests" / "js" / "geometry_cases.mjs"), str(GEOMETRY_JS)],
                           input=json.dumps([["earthFixedToThreeLocal", [0.0, lon]]]),
                           capture_output=True, text=True, check=True).stdout
    cases[0][1][0] = json.loads(local)[0]
    rotated = json.loads(subprocess.run(
        ["node", str(ROOT / "tests" / "js" / "geometry_cases.mjs"), str(GEOMETRY_JS)],
        input=json.dumps(cases), capture_output=True, text=True, check=True).stdout)[0]
    ang = theta + math.radians(lon)
    eci = [math.cos(ang), math.sin(ang), 0.0]
    np.testing.assert_allclose(rotated, [eci[0], eci[2], -eci[1]], atol=1e-12)
