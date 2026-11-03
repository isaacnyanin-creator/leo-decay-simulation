"""Dashboard physics helpers (app/derived.py)."""

import numpy as np
import pytest

from derived import (
    M_S2_TO_KM_DAY2, SatelliteNotCalibrated, decay_acceleration_m_s2, decay_rate_m_s,
    drag_force_n, is_simulatable, rescale_b_star, simulate,
)
from leo_decay.constants import MU_EARTH, R_EARTH, RHO_FLOOR, SECONDS_PER_DAY
from leo_decay.density import thermospheric_density
from leo_decay.propagator import decay_rhs, make_forcing

B_CAL = 1.143720e-2
A_P3 = R_EARTH + 469_500.0


def test_rescale_b_star():
    assert rescale_b_star(B_CAL, 11_110.0, 11_110.0) == B_CAL
    assert rescale_b_star(B_CAL, 11_110.0, 22_220.0) == pytest.approx(B_CAL / 2)
    with pytest.raises(ValueError):
        rescale_b_star(B_CAL, 11_110.0, 0.0)


def test_decay_rate_matches_engine_rhs():
    f, g = make_forcing(145.0), make_forcing(15.0)
    assert decay_rate_m_s(A_P3, B_CAL, 145.0, 15.0) == pytest.approx(
        decay_rhs(0.0, [A_P3], B_CAL, f, g)[0])


def test_eq_b2_matches_chain_rule():
    da = 1.0
    rate = lambda a: decay_rate_m_s(a, B_CAL, 145.0, 15.0)
    numeric = (rate(A_P3 + da) - rate(A_P3 - da)) / (2 * da) * rate(A_P3)
    assert decay_acceleration_m_s2(A_P3, B_CAL, 145.0, 15.0) == pytest.approx(numeric, rel=1e-6)


def test_eq_b2_on_density_floor():
    a = R_EARTH + 2_000_000.0
    assert thermospheric_density(a, 145.0, 15.0) == RHO_FLOOR
    rate = decay_rate_m_s(a, B_CAL, 145.0, 15.0)
    assert decay_acceleration_m_s2(a, B_CAL, 145.0, 15.0) == pytest.approx(rate**2 / (2 * a))


def test_drag_force_independent_of_mass_at_fixed_altitude():
    f1 = drag_force_n(A_P3, B_CAL, 11_110.0, 145.0, 15.0)
    f2 = drag_force_n(A_P3, rescale_b_star(B_CAL, 11_110.0, 44_440.0), 44_440.0, 145.0, 15.0)
    assert f1 == pytest.approx(f2)


def test_drag_force_consistent_with_energy_loss():
    # F v = -dE/dt = -(mu m / 2a^2) da/dt  (Appendix B.1)
    m = 11_110.0
    v = np.sqrt(MU_EARTH / A_P3)
    power = -(MU_EARTH * m / (2 * A_P3**2)) * decay_rate_m_s(A_P3, B_CAL, 145.0, 15.0)
    assert drag_force_n(A_P3, B_CAL, m, 145.0, 15.0) * v == pytest.approx(power)


def test_simulate_default_reproduces_protocol3(hubble):
    res = simulate(hubble, hubble.mass_kg, 145.0, 15.0, 469_500.0, 35, n_points=36)
    df = res.frame
    assert res.b_star == B_CAL and not res.reentered
    assert df["altitude_km"].iloc[-1] - df["altitude_km"].iloc[0] == pytest.approx(-2.3947, abs=1e-3)
    assert list(df.columns) == ["day", "altitude_km", "decay_rate_km_day", "decay_accel_km_day2",
                                "density_kg_m3", "drag_force_n"]


def test_simulated_acceleration_matches_trajectory(hubble):
    df = simulate(hubble, hubble.mass_kg, 145.0, 15.0, 469_500.0, 35, n_points=2_001).frame
    numeric = np.gradient(df["decay_rate_km_day"].to_numpy(), df["day"].to_numpy())
    np.testing.assert_allclose(numeric[5:-5], df["decay_accel_km_day2"].to_numpy()[5:-5], rtol=1e-4)


def test_heavier_satellite_decays_slower(hubble):
    light = simulate(hubble, hubble.mass_kg, 145.0, 15.0, 469_500.0, 35)
    heavy = simulate(hubble, 2 * hubble.mass_kg, 145.0, 15.0, 469_500.0, 35)
    assert heavy.frame["altitude_km"].iloc[-1] > light.frame["altitude_km"].iloc[-1]


def test_reentry_day_reported(hubble):
    res = simulate(hubble, hubble.mass_kg, 145.0, 15.0, 200_000.0, 365)
    assert res.reentered
    assert 0 < res.reentry_day < 365
    assert res.frame["day"].iloc[-1] <= res.reentry_day


def test_uncalibrated_satellites(satellites):
    assert is_simulatable(satellites["hubble"])
    for key in ("iss", "noaa15"):
        assert not is_simulatable(satellites[key])
        with pytest.raises(SatelliteNotCalibrated):
            simulate(satellites[key], 1_000.0, 145.0, 15.0, 400_000.0, 10)
