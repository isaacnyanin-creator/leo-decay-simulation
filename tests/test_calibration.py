import numpy as np
import pytest

from leo_decay.calibration import B_STAR_BOUNDS, calibrate_b_star, model_decay_rate
from leo_decay.constants import R_EARTH, SECONDS_PER_DAY

B_TRUE = 1.143720e-2


def _synthetic(noise_m_s=0.0, seed=0):
    rng = np.random.default_rng(seed)
    a = R_EARTH + np.linspace(539e3, 472e3, 400)
    f107 = rng.uniform(70.0, 200.0, a.size)
    ap = rng.uniform(5.0, 40.0, a.size)
    d = model_decay_rate((a, f107, ap), B_TRUE)
    return a, d + rng.normal(0.0, noise_m_s, a.size), f107, ap


def test_recovers_exact_b_star():
    a, d, f, g = _synthetic()
    res = calibrate_b_star(a, d, f, g)
    assert res.b_star == pytest.approx(B_TRUE, rel=1e-8)
    assert res.n_obs == a.size


def test_recovers_b_star_with_noise_and_reports_sigma():
    noise = 0.005 / SECONDS_PER_DAY      # 5 m/day of rate noise
    a, d, f, g = _synthetic(noise_m_s=noise, seed=1)
    res = calibrate_b_star(a, d, f, g)
    assert res.b_star_sigma > 0
    assert abs(res.b_star - B_TRUE) < 5 * res.b_star_sigma


def test_nans_dropped():
    a, d, f, g = _synthetic()
    d[::10] = np.nan
    res = calibrate_b_star(a, d, f, g)
    assert res.n_obs == a.size - np.isnan(d).sum()
    assert res.b_star == pytest.approx(B_TRUE, rel=1e-8)


def test_respects_bounds():
    a, d, f, g = _synthetic()
    res = calibrate_b_star(a, d * 1e3, f, g)      # would need B* ~ 11 m^2/kg
    assert res.b_star == pytest.approx(B_STAR_BOUNDS[1])


def test_scalar_forcing_broadcasts():
    a = R_EARTH + np.linspace(480e3, 470e3, 50)
    d = model_decay_rate((a, 145.0, 15.0), B_TRUE)
    assert calibrate_b_star(a, d, 145.0, 15.0).b_star == pytest.approx(B_TRUE, rel=1e-8)
