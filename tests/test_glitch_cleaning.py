import numpy as np
import pandas as pd

from leo_decay.glitch_cleaning import GLITCH_THRESHOLD_M, clean_glitches, flag_glitches


def _decaying_series(n=20, start=500_000.0, step=-100.0):
    return pd.Series(start + step * np.arange(n))


def test_threshold_is_2km_in_metres():
    assert GLITCH_THRESHOLD_M == 2_000.0


def test_flags_only_large_jumps():
    h = _decaying_series()
    h.iloc[10] += 2_500.0
    flags = flag_glitches(h)
    # The spike and the step back down both exceed 2 km (as in Appendix A.1).
    assert flags.tolist() == [i in (10, 11) for i in range(len(h))]


def test_small_steps_untouched():
    h = _decaying_series(step=-1_900.0)
    pd.testing.assert_series_equal(clean_glitches(h), h)


def test_spike_reconstructed_monotonic_without_overshoot():
    clean = _decaying_series()
    h = clean.copy()
    h.iloc[10] += 3_000.0
    fixed = clean_glitches(h)
    assert not fixed.isna().any()
    assert fixed.is_monotonic_decreasing                    # PCHIP preserves monotonicity
    # Linear underlying trend -> PCHIP recovers it exactly
    np.testing.assert_allclose(fixed, clean, atol=1e-6)
