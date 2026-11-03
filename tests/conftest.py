from dataclasses import replace

import pytest

from leo_decay.config import load_satellites
from leo_decay.solar import load_solar_config


@pytest.fixture(scope="session")
def satellites():
    return load_satellites()


@pytest.fixture(scope="session")
def hubble(satellites):
    return satellites["hubble"]


@pytest.fixture(scope="session")
def solar_cfg():
    return load_solar_config()


@pytest.fixture(scope="session")
def flat_solar_cfg(solar_cfg):
    """Constant F10.7, no rotation/noise, no events: a quiet constant-forcing scenario."""
    def make(f107: float = 145.0):
        return replace(
            solar_cfg,
            background=replace(solar_cfg.background, f107_min=f107, f107_max=f107,
                               rotation_amplitude_min_sfu=0.0, rotation_amplitude_frac=0.0,
                               noise_sigma_sfu=0.0, f107_floor=0.0),
            flares=replace(solar_cfg.flares, base_rate_per_day=0.0),
            cmes=replace(solar_cfg.cmes, base_rate_per_day=0.0),
        )
    return make
