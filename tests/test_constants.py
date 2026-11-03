"""Constants must match Appendix B.2 exactly, and the config must match §3.1."""

from leo_decay import constants as c


def test_geocentric_constants():
    assert c.MU_EARTH == 3.986004418e14
    assert c.R_EARTH == 6_378_137.0


def test_thermospheric_constants():
    assert c.RHO_0 == 3.82e-12
    assert c.H_0_REF == 400_000.0
    assert c.SCALE_HEIGHT == 50_000.0
    assert c.RHO_FLOOR == 1.0e-18


def test_space_weather_constants():
    assert c.F107_BASELINE == 100.0 and c.F107_SENSITIVITY == 0.008
    assert c.AP_BASELINE == 15.0 and c.AP_SENSITIVITY == 0.003


def test_scale_height_dominance_ratio():
    # §3.4.1 / B.1: 2a/H ~ 274 for a ~ 6850 km
    assert round(2 * 6_850_337.0 / c.SCALE_HEIGHT) == 274


def test_config_norad_ids(satellites):
    assert {k: s.norad_id for k, s in satellites.items()} == {
        "hubble": 20580, "iss": 25544, "noaa15": 25338,
    }
    assert satellites["hubble"].passive and satellites["noaa15"].passive
    assert not satellites["iss"].passive


def test_hubble_properties(hubble):
    assert hubble.mass_kg == 11_110.0
    assert hubble.drag_coefficient == 2.2
    assert hubble.b_star_empirical_m2_kg == 1.143720e-2
    # §4.3: empirical / static ratio ~ 4.98
    assert round(hubble.b_star_empirical_m2_kg / hubble.b_star_static_m2_kg, 2) == 4.98
