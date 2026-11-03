"""Forecast-vs-TLE residuals on synthetic element sets."""

import numpy as np
import pandas as pd
import pytest

from leo_decay.propagator import propagate
from leo_decay.tle_conversion import mean_motion_from_semi_major_axis
from leo_decay.constants import R_EARTH
from leo_decay.validation import altitude_at, check_forecast

T0 = pd.Timestamp("2026-10-15T00:00:00Z")
B = 1.143720e-2


def _synthetic_gp(h0=469_520.0, offset_m=0.0, noise_m=0.0, seed=0):
    """TLE-like records from the engine itself, sampled ~every 8 h, from 20 days before T0."""
    rng = np.random.default_rng(seed)
    t = np.arange(-20 * 86400, 20 * 86400, 8 * 3600.0) + rng.uniform(0, 3600, 120)
    pre = propagate(h0, 20 * 86400, B, 145.0, 15.0, t_eval_s=np.sort(-t[t < 0]))  # backwards proxy
    post = propagate(h0, 20 * 86400, B, 145.0, 15.0, t_eval_s=t[t >= 0])
    # symmetric approx before T0: mirror the decay
    h = np.concatenate([2 * h0 - pre.h_m[::-1], post.h_m]) + offset_m + rng.normal(0, noise_m, t.size)
    a = R_EARTH + h
    return pd.DataFrame({"EPOCH": T0 + pd.to_timedelta(np.sort(t), unit="s"),
                         "MEAN_MOTION": mean_motion_from_semi_major_axis(a)})


def test_altitude_at_interpolates_and_reports_nearest():
    obs = pd.DataFrame({"EPOCH": [T0, T0 + pd.Timedelta(hours=10)], "altitude_m": [1000.0, 2000.0]})
    o = altitude_at(obs, T0 + pd.Timedelta(hours=4), T0)
    assert o["interp_m"] == pytest.approx(1400.0)
    assert o["nearest_m"] == 1000.0 and o["bracket_gap_h"] == pytest.approx(10.0)
    with pytest.raises(ValueError):
        altitude_at(obs, T0 + pd.Timedelta(days=1), T0)


def test_perfect_observations_give_near_zero_residual():
    gp = _synthetic_gp(h0=469_500.0)
    res = check_forecast(gp, T0, 469_500.0, B, 145.0, 15.0, [7, 14, 21, 28, 35])
    assert list(res.table["day"]) == [7, 14]                  # only checkpoints inside the data
    assert res.table["residual_m"].abs().max() < 2.0           # linear-interp error only
    assert res.h0_observed_m == pytest.approx(469_500.0, abs=2.0)


def test_start_offset_shows_in_residual_but_not_anchored():
    # Truth starts 300 m lower (and so decays slightly faster, in denser air).
    gp = _synthetic_gp(h0=469_200.0)
    res = check_forecast(gp, T0, 469_500.0, B, 145.0, 15.0, [7, 14])
    assert res.h0_observed_m == pytest.approx(469_200.0, abs=2.0)
    assert (res.table["residual_m"] > 300.0).all()             # offset grows a little
    assert res.table["residual_anchored_m"].abs().max() < 2.0
    assert res.observed_rate_km_day < res.forecast_rate_km_day  # lower orbit decays faster


def test_noise_estimate():
    gp = _synthetic_gp(noise_m=40.0, seed=3)
    res = check_forecast(gp, T0, 469_500.0, B, 145.0, 15.0, [7])
    assert 25.0 < res.noise_m < 60.0
