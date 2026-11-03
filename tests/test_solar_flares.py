"""Long-run flare statistics must match the config (fixed seeds, 4-sigma bounds)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from leo_decay.solar.flares import (
    exceedance_probability, flare_profile, flux_class_label, sample_flares,
)
from leo_decay.solar.rates import power_rate

YEARS_200 = 200 * 365.25


def _within_poisson(count, expected, k=4.0):
    return abs(count - expected) <= k * np.sqrt(expected)


@pytest.mark.parametrize("f107", [100.0, 200.0])
def test_flare_rate_matches_config(solar_cfg, f107):
    c = solar_cfg.flares
    fl = sample_flares(np.random.default_rng(10), YEARS_200, f107, c)
    expected = power_rate(f107, c.base_rate_per_day, c.rate_reference_f107,
                          c.rate_f107_exponent) * YEARS_200
    assert _within_poisson(len(fl), expected)


def test_rate_exponent_zero_disables_f107_dependence(solar_cfg):
    c = replace(solar_cfg.flares, rate_f107_exponent=0.0)
    n_lo = len(sample_flares(np.random.default_rng(11), YEARS_200 / 10, 70.0, c))
    n_hi = len(sample_flares(np.random.default_rng(12), YEARS_200 / 10, 250.0, c))
    expected = c.base_rate_per_day * YEARS_200 / 10
    assert _within_poisson(n_lo, expected) and _within_poisson(n_hi, expected)


def test_time_varying_rate_by_thinning(solar_cfg):
    c = solar_cfg.flares
    t = np.linspace(0, YEARS_200 / 4, 200_001)
    f = 130.0 + 60.0 * np.sin(2 * np.pi * t / 4018.0)
    fl = sample_flares(np.random.default_rng(13), t[-1], (t, f), c)
    lam = power_rate(f, c.base_rate_per_day, c.rate_reference_f107, c.rate_f107_exponent)
    assert _within_poisson(len(fl), np.trapezoid(lam, t))


def test_class_fractions_follow_power_law(solar_cfg):
    c = solar_cfg.flares
    fl = sample_flares(np.random.default_rng(14), YEARS_200, 150.0, c)
    n = len(fl)
    assert fl["flux_w_m2"].between(c.min_flux_w_m2, c.max_flux_w_m2).all()
    for letter, threshold in [("M", 1e-5), ("X", 1e-4)]:
        p = exceedance_probability(threshold, c)
        observed = (fl["flux_w_m2"] >= threshold).sum()
        assert abs(observed - n * p) <= 4 * np.sqrt(n * p * (1 - p)), letter
    counts = fl["class_letter"].value_counts()
    assert counts["C"] > counts["M"] > counts["X"] > 0          # large flares rarer


def test_class_labels():
    assert flux_class_label(2.34e-5) == "M2.3"
    assert flux_class_label(1.0e-6) == "C1.0"
    assert flux_class_label(1.7e-3) == "X17.0"


def test_flare_profile_rises_and_decays(solar_cfg):
    flare = pd.DataFrame({"start_day": [10.0], "rise_h": [0.5], "decay_efold_h": [2.0],
                          "delta_f107": [8.0]})
    t = np.arange(0, 20, 1 / 240)                              # 6-minute grid
    prof = flare_profile(t, flare)
    assert prof[t < 10.0].max() == 0.0
    peak_idx = np.argmax(prof)
    assert t[peak_idx] == pytest.approx(10.0 + 0.5 / 24, abs=1 / 240)
    assert prof[peak_idx] == pytest.approx(8.0, rel=0.01)
    after = t >= 10.0 + (0.5 + 3 * 2.0) / 24                    # 3 e-folds after peak
    assert prof[after].max() <= 8.0 * np.exp(-3) * 1.01
    assert prof[t > 12.0].max() < 1e-3                          # brief: gone within a day
