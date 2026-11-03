"""Protocol 3 reproduction (paper §3.4.3 and §4.3) and ODE properties."""

import numpy as np
import pytest

from leo_decay.constants import MU_EARTH, R_EARTH, SCALE_HEIGHT, SECONDS_PER_DAY
from leo_decay.density import thermospheric_density
from leo_decay.propagator import decay_rhs, make_forcing, propagate, weekly_checkpoints_s

# Protocol 3: sealed 35-day forecast, Oct 15 - Nov 19 2026
H0_M = 469_500.0
B_STAR = 1.143720e-2
F107, AP = 145.0, 15.0
DURATION_S = 3_024_000.0

# §4.3 reported weekly altitudes (km): Oct 22, Oct 29, Nov 5, Nov 12, Nov 19
PAPER_WEEKLY_KM = [469.0302, 468.5559, 468.0771, 467.5936, 467.1053]
PAPER_DH35_KM = -2.3947
PAPER_MEAN_RATE_KM_DAY = -0.0684
TOL_KM = 1e-3      # 1 m


@pytest.fixture(scope="module")
def protocol3():
    return propagate(H0_M, DURATION_S, B_STAR, F107, AP,
                     t_eval_s=weekly_checkpoints_s(5))


def test_duration_is_35_days():
    assert DURATION_S == 35 * SECONDS_PER_DAY
    assert weekly_checkpoints_s(5)[-1] == DURATION_S


def test_initial_state(protocol3):
    assert protocol3.a_m[0] == 6_847_637.0
    assert not protocol3.reentered


def test_weekly_checkpoints_match_paper(protocol3):
    h_km = protocol3.h_m[1:] / 1000.0
    np.testing.assert_allclose(h_km, PAPER_WEEKLY_KM, atol=TOL_KM)


def test_35_day_altitude_loss(protocol3):
    dh_km = (protocol3.h_m[-1] - protocol3.h_m[0]) / 1000.0
    assert dh_km == pytest.approx(PAPER_DH35_KM, abs=TOL_KM)
    assert dh_km == pytest.approx(-2.39, abs=0.005)


def test_mean_decay_rate(protocol3):
    rate = (protocol3.h_m[-1] - protocol3.h_m[0]) / 1000.0 / 35.0
    assert rate == pytest.approx(PAPER_MEAN_RATE_KM_DAY, abs=1e-4)


def test_decay_accelerates(protocol3):
    """Weekly losses grow week on week: d^2a/dt^2 < 0 (Eq. B.2)."""
    weekly_loss = -np.diff(protocol3.h_m)
    assert np.all(weekly_loss > 0)
    assert np.all(np.diff(weekly_loss) > 0)


def test_eq_b2_matches_numerical_second_derivative():
    a = R_EARTH + H0_M
    rho = thermospheric_density(a, F107, AP)
    analytic = -(B_STAR**2) * MU_EARTH * a * rho**2 * (1 / SCALE_HEIGHT - 1 / (2 * a))
    f, g = make_forcing(F107), make_forcing(AP)
    da = 1.0     # m
    d_rate_da = (decay_rhs(0, [a + da], B_STAR, f, g)[0]
                 - decay_rhs(0, [a - da], B_STAR, f, g)[0]) / (2 * da)
    numeric = d_rate_da * decay_rhs(0, [a], B_STAR, f, g)[0]
    assert analytic < 0
    assert numeric == pytest.approx(analytic, rel=1e-6)


def test_interpolated_forcing_equals_constant_when_flat():
    t_nodes = np.array([0.0, DURATION_S])
    traj = propagate(H0_M, DURATION_S, B_STAR,
                     (t_nodes, np.array([F107, F107])), (t_nodes, np.array([AP, AP])),
                     t_eval_s=weekly_checkpoints_s(5))
    np.testing.assert_allclose(traj.h_m[1:] / 1000.0, PAPER_WEEKLY_KM, atol=TOL_KM)


def test_higher_flux_decays_faster():
    lo = propagate(H0_M, DURATION_S, B_STAR, 100.0, AP)
    hi = propagate(H0_M, DURATION_S, B_STAR, 200.0, AP)
    assert hi.h_m[-1] < lo.h_m[-1]


def test_reentry_guard_terminates():
    traj = propagate(200_000.0, 365 * SECONDS_PER_DAY, B_STAR, F107, AP)
    assert traj.reentered
    assert traj.a_m[-1] == pytest.approx(R_EARTH, abs=1.0)
    assert traj.t_s[-1] < 365 * SECONDS_PER_DAY
