"""Long-run CME and storm statistics must match the config (fixed seeds, 4-sigma bounds)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from leo_decay.solar.cmes import (
    ap_profile, peak_ap, sample_cmes, storms_from_cmes, transit_lag_hours,
)
from leo_decay.solar.rates import power_rate

YEARS_200 = 200 * 365.25


@pytest.fixture(scope="module")
def long_run(solar_cfg):
    cmes = sample_cmes(np.random.default_rng(20), YEARS_200, 150.0, solar_cfg.cmes)
    return cmes, storms_from_cmes(cmes, solar_cfg.storms)


@pytest.mark.parametrize("f107", [80.0, 150.0, 250.0])
def test_cme_rate_rises_with_f107_as_configured(solar_cfg, f107):
    c = solar_cfg.cmes
    n = len(sample_cmes(np.random.default_rng(21), YEARS_200, f107, c))
    expected = power_rate(f107, c.base_rate_per_day, c.rate_reference_f107,
                          c.rate_f107_exponent) * YEARS_200
    assert abs(n - expected) <= 4 * np.sqrt(expected)


def test_earth_directed_fraction(solar_cfg, long_run):
    cmes, _ = long_run
    p = solar_cfg.cmes.earth_directed_probability
    n, k = len(cmes), cmes["earth_directed"].sum()
    assert abs(k - n * p) <= 4 * np.sqrt(n * p * (1 - p))


def test_speeds_within_clip(solar_cfg, long_run):
    cmes, _ = long_run
    c = solar_cfg.cmes
    assert cmes["speed_km_s"].between(c.speed_min_km_s, c.speed_max_km_s).all()
    assert np.median(cmes["speed_km_s"]) == pytest.approx(c.speed_median_km_s, rel=0.02)


def test_one_storm_per_earth_directed_cme(long_run):
    cmes, storms = long_run
    assert len(storms) == cmes["earth_directed"].sum()
    assert cmes.loc[storms["parent_index"], "earth_directed"].all()


def test_storm_lags_within_window_and_match_transit(solar_cfg, long_run):
    cmes, storms = long_run
    s = solar_cfg.storms
    lag_h = (storms["start_day"].to_numpy()
             - cmes.loc[storms["parent_index"], "start_day"].to_numpy()) * 24.0
    assert lag_h.min() >= s.lag_min_hours - 1e-9
    assert lag_h.max() <= s.lag_max_hours + 1e-9
    np.testing.assert_allclose(lag_h, storms["lag_h"], atol=1e-6)
    np.testing.assert_allclose(storms["lag_h"], transit_lag_hours(storms["speed_km_s"], s))


def test_mean_storm_lag_matches_speed_distribution(solar_cfg, long_run):
    _, storms = long_run
    c, s = solar_cfg.cmes, solar_cfg.storms

    def lag_of_z(z):
        v = np.clip(c.speed_median_km_s * np.exp(c.speed_sigma_ln * z), c.speed_min_km_s, c.speed_max_km_s)
        return float(transit_lag_hours(v, s))

    expected, _ = quad(lambda z: lag_of_z(z) * norm.pdf(z), -12, 12, limit=200)
    lags = storms["lag_h"].to_numpy()
    assert abs(lags.mean() - expected) <= 4 * lags.std() / np.sqrt(lags.size)


def test_every_storm_exceeds_ap_30(solar_cfg, long_run):
    _, storms = long_run
    assert (storms["peak_ap"] > 30).all()
    assert (storms["peak_ap"] <= solar_cfg.storms.ap_cap).all()
    assert peak_ap(3_000.0, solar_cfg.storms) > peak_ap(500.0, solar_cfg.storms)


def test_ap_profile_shape_and_return_to_quiet(solar_cfg):
    s = solar_cfg.storms
    storm = pd.DataFrame({"start_day": [5.0], "peak_ap": [120.0]})
    t = np.arange(0, 20, 1 / 60)                       # 24-minute grid
    ap = ap_profile(t, storm, s)
    main, tau = s.main_phase_hours / 24, s.recovery_efold_hours / 24
    assert ap.min() == pytest.approx(s.quiet_ap)
    assert ap[t < 5.0].max() == pytest.approx(s.quiet_ap)
    i_peak = np.argmin(np.abs(t - (5.0 + main)))
    assert ap[i_peak] == pytest.approx(120.0, rel=1e-3)
    for k in (1, 2, 3):
        i = np.argmin(np.abs(t - (5.0 + main + k * tau)))
        assert ap[i] - s.quiet_ap == pytest.approx((120.0 - s.quiet_ap) * np.exp(-k), rel=1e-2)
    assert ap[-1] == pytest.approx(s.quiet_ap, abs=1e-3)   # decays back to quiet Ap = 15


def test_overlapping_storms_add_and_cap(solar_cfg):
    s = solar_cfg.storms
    t = np.arange(0, 10, 1 / 24)
    two = pd.DataFrame({"start_day": [2.0, 2.0], "peak_ap": [100.0, 100.0]})
    one = pd.DataFrame({"start_day": [2.0], "peak_ap": [100.0]})
    excess_two = ap_profile(t, two, s) - s.quiet_ap
    excess_one = ap_profile(t, one, s) - s.quiet_ap
    np.testing.assert_allclose(excess_two, 2 * excess_one)
    huge = pd.DataFrame({"start_day": [2.0] * 3, "peak_ap": [400.0] * 3})
    assert ap_profile(t, huge, s).max() == s.ap_cap


def test_zero_rate_gives_no_cmes(solar_cfg):
    c = replace(solar_cfg.cmes, base_rate_per_day=0.0)
    assert sample_cmes(np.random.default_rng(0), 1000.0, 150.0, c).empty
