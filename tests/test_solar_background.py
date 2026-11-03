from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from leo_decay.solar.background import ar1_noise, cycle_trend, generate_background
from leo_decay.solar.generator import centered_moving_average


def test_cycle_trend_min_and_max(solar_cfg):
    b = solar_cfg.background
    start = b.cycle_min_utc
    assert cycle_trend(0.0, start, b) == pytest.approx(b.f107_min)
    assert cycle_trend(b.cycle_period_days / 2, start, b) == pytest.approx(b.f107_max)
    assert cycle_trend(b.cycle_period_days, start, b) == pytest.approx(b.f107_min)
    # Same instant expressed from a different start date gives the same value
    later = start + timedelta(days=1000)
    assert cycle_trend(0.0, later, b) == pytest.approx(cycle_trend(1000.0, start, b))


def test_noiseless_background_is_trend_plus_rotation(solar_cfg):
    b = replace(solar_cfg.background, noise_sigma_sfu=0.0, rotation_amplitude_min_sfu=0.0,
                rotation_amplitude_frac=0.0)
    t = np.arange(0, 400, 1 / 24)
    bg = generate_background(np.random.default_rng(0), t, solar_cfg.start_utc, b)
    np.testing.assert_allclose(bg, cycle_trend(t, solar_cfg.start_utc, b))


def test_27_day_rotation_dominates_detrended_spectrum(solar_cfg):
    b = solar_cfg.background
    t = np.arange(0, 3 * 365.25, 1 / 24)
    bg = generate_background(np.random.default_rng(1), t, solar_cfg.start_utc, b)
    resid = bg - cycle_trend(t, solar_cfg.start_utc, b)
    resid -= resid.mean()
    power = np.abs(np.fft.rfft(resid)) ** 2
    freq = np.fft.rfftfreq(t.size, d=1 / 24)                 # cycles/day
    band = (freq > 1 / 60) & (freq < 1 / 10)                 # periods 10-60 days
    peak_period = 1 / freq[band][np.argmax(power[band])]
    assert peak_period == pytest.approx(b.rotation_period_days, abs=1.0)


def test_ar1_noise_statistics(solar_cfg):
    b = solar_cfg.background
    dt = 0.25
    x = ar1_noise(np.random.default_rng(2), 400_000, dt, b)
    assert x.std() == pytest.approx(b.noise_sigma_sfu, rel=0.03)
    lag1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert lag1 == pytest.approx(np.exp(-dt / b.noise_correlation_days), abs=0.01)


def test_floor_respected(solar_cfg):
    b = replace(solar_cfg.background, f107_min=50.0, f107_max=55.0, noise_sigma_sfu=30.0,
                f107_floor=60.0)
    bg = generate_background(np.random.default_rng(3), np.arange(0, 100, 1 / 24),
                             solar_cfg.start_utc, b)
    assert bg.min() >= 60.0


def test_centered_moving_average():
    np.testing.assert_allclose(centered_moving_average(np.full(50, 7.0), 11), 7.0)
    ramp = np.arange(100, dtype=float)
    out = centered_moving_average(ramp, 21)
    np.testing.assert_allclose(out, ramp[10:-10])          # centered: no lag on a ramp
    with pytest.raises(ValueError):
        centered_moving_average(ramp, 20)
