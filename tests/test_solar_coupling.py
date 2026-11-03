import numpy as np
import pytest

from leo_decay.propagator import propagate, weekly_checkpoints_s
from leo_decay.solar import generate_scenario, propagate_piecewise, propagate_scenario

B_STAR = 1.143720e-2
H0 = 469_500.0
PAPER_WEEKLY_KM = [469.0302, 468.5559, 468.0771, 467.5936, 467.1053]
DAY = 86_400.0


@pytest.mark.parametrize("mode", ["30ma", "instantaneous"])
def test_quiet_scenario_reproduces_protocol3(flat_solar_cfg, mode):
    sc = generate_scenario(35, seed=1, config=flat_solar_cfg(145.0))
    traj = propagate_scenario(sc, H0, B_STAR, mode, t_eval_s=weekly_checkpoints_s(5))
    np.testing.assert_allclose(traj.t_s, weekly_checkpoints_s(5))
    np.testing.assert_allclose(traj.h_m[1:] / 1000, PAPER_WEEKLY_KM, atol=1e-3)


def test_piecewise_matches_single_call_on_constant_forcing():
    t = np.array([0.0, 35 * DAY])
    ref = propagate(H0, 35 * DAY, B_STAR, 145.0, 15.0, t_eval_s=weekly_checkpoints_s(5))
    pw = propagate_piecewise(H0, 35 * DAY, B_STAR, t, [145.0, 145.0], [15.0, 15.0],
                             breakpoints_s=[3.3 * DAY, 17.77 * DAY], max_segment_s=0.5 * DAY,
                             t_eval_s=weekly_checkpoints_s(5))
    np.testing.assert_allclose(pw.a_m, ref.a_m, atol=1e-3)     # 1 mm


def _spike(t0_day=10.3, width_h=2.0, peak=400.0):
    t = np.arange(0, 35 * DAY + 1, 600.0)
    ap = np.full(t.size, 15.0)
    mid, half = t0_day * DAY, width_h * 1800.0
    on = np.abs(t - mid) < half
    ap[on] = 15.0 + (peak - 15.0) * (1 - np.abs(t[on] - mid) / half)
    window = (t >= mid - half - 600.0) & (t <= mid + half + 600.0)
    return t, np.full(t.size, 145.0), ap, t[window]


def test_breakpoints_resolve_short_spike():
    t, f, ap, bp = _spike()
    pw = propagate_piecewise(H0, 35 * DAY, B_STAR, t, f, ap, breakpoints_s=bp)
    fine = propagate_piecewise(H0, 35 * DAY, B_STAR, t, f, ap, max_segment_s=600.0)
    quiet = propagate(H0, 35 * DAY, B_STAR, 145.0, 15.0)
    assert pw.h_m[-1] == pytest.approx(fine.h_m[-1], abs=1e-3)      # matches 10-min reference
    assert quiet.h_m[-1] - pw.h_m[-1] > 1.0                          # spike costs > 1 m


@pytest.mark.parametrize("mode", ["30ma", "instantaneous"])
def test_scenario_default_segmentation_is_converged(mode):
    """Default (one segment per grid node) matches 10-minute segments to 1 mm,
    and coarser daily segments do not."""
    sc = generate_scenario(35, seed=2026)
    assert len(sc.storms) > 0
    default = propagate_scenario(sc, H0, B_STAR, mode).h_m[-1]
    fine = propagate_scenario(sc, H0, B_STAR, mode, max_segment_s=600.0).h_m[-1]
    daily = propagate_scenario(sc, H0, B_STAR, mode, max_segment_s=DAY).h_m[-1]
    assert default == pytest.approx(fine, abs=1e-3)
    assert abs(daily - fine) > 0.01


def test_storms_increase_decay(solar_cfg, flat_solar_cfg):
    from dataclasses import replace
    stormy_cfg = replace(flat_solar_cfg(145.0), cmes=replace(
        solar_cfg.cmes, base_rate_per_day=0.5, rate_f107_exponent=0.0,
        earth_directed_probability=1.0))
    stormy = generate_scenario(35, seed=5, config=stormy_cfg)
    quiet = generate_scenario(35, seed=5, config=flat_solar_cfg(145.0))
    assert len(stormy.storms) > 0
    h_stormy = propagate_scenario(stormy, H0, B_STAR).h_m[-1]
    h_quiet = propagate_scenario(quiet, H0, B_STAR).h_m[-1]
    assert h_stormy < h_quiet


def test_piecewise_reentry():
    t = np.array([0.0, 365 * DAY])
    traj = propagate_piecewise(200_000.0, 365 * DAY, B_STAR, t, [145.0] * 2, [15.0] * 2)
    assert traj.reentered
    assert traj.t_s[-1] < 365 * DAY
    assert np.all(np.diff(traj.h_m) < 0)


def test_t_eval_validation():
    t = np.array([0.0, DAY])
    with pytest.raises(ValueError):
        propagate_piecewise(H0, DAY, B_STAR, t, [145.0] * 2, [15.0] * 2, t_eval_s=[2 * DAY])
