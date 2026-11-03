import numpy as np
import pandas as pd
import pytest

from leo_decay.constants import MU_EARTH, R_EARTH, SECONDS_PER_DAY
from leo_decay.tle_conversion import (
    add_orbital_state, altitude_m, circular_speed_m_s, decay_rate_m_s,
    mean_motion_from_semi_major_axis, semi_major_axis_m,
)


def test_round_trip_mean_motion():
    a = np.array([6_784_137.0, 6_850_337.0, 7_183_537.0])
    n = mean_motion_from_semi_major_axis(a)
    np.testing.assert_allclose(semi_major_axis_m(n), a, rtol=1e-12)


def test_hubble_protocol_initial_states():
    # Protocol 1/2: h0 = 472.200 km -> a0 = 6,850,337.0 m
    # Protocol 3:   h0 = 469.500 km -> a0 = 6,847,637.0 m
    for h0, a0 in [(472_200.0, 6_850_337.0), (469_500.0, 6_847_637.0)]:
        n = mean_motion_from_semi_major_axis(a0)
        assert altitude_m(semi_major_axis_m(n)) == pytest.approx(h0, abs=1e-6)


def test_hubble_mean_motion_is_physical():
    n = mean_motion_from_semi_major_axis(R_EARTH + 472_200.0)
    assert 15.0 < n < 15.4      # rev/day, typical for ~470 km


def test_circular_speed():
    a = 6_850_337.0
    assert circular_speed_m_s(a) == pytest.approx(np.sqrt(MU_EARTH / a))
    assert 7_500 < circular_speed_m_s(a) < 7_800   # §2.1 7.5-7.8 km/s


def test_add_orbital_state_sorts_and_dedupes():
    n = float(mean_motion_from_semi_major_axis(6_850_337.0))
    df = pd.DataFrame({
        "EPOCH": pd.to_datetime(["2026-07-04", "2026-07-01", "2026-07-01"]),
        "MEAN_MOTION": [n, n, n],
    })
    out = add_orbital_state(df)
    assert len(out) == 2
    assert out["EPOCH"].is_monotonic_increasing
    np.testing.assert_allclose(out["altitude_m"], 472_200.0, atol=1e-6)


def test_decay_rate_units_m_s():
    epoch = pd.Series(pd.date_range("2026-01-01", periods=3, freq="3D"))
    h = pd.Series([472_200.0, 472_098.0, 471_996.0])   # -34 m/day
    rate = decay_rate_m_s(h, epoch)
    assert np.isnan(rate.iloc[0])
    np.testing.assert_allclose(rate.iloc[1:], -34.0 / SECONDS_PER_DAY)
