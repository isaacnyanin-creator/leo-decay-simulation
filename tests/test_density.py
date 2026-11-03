import numpy as np
import pytest

from leo_decay.constants import R_EARTH, RHO_0, RHO_FLOOR
from leo_decay.density import geomagnetic_factor, solar_factor, thermospheric_density


def test_reference_density_at_400km_quiet():
    assert thermospheric_density(R_EARTH + 400_000.0, 100.0, 15.0) == pytest.approx(RHO_0)


def test_one_scale_height_up_is_factor_e():
    rho_400 = thermospheric_density(R_EARTH + 400_000.0, 100.0, 15.0)
    rho_450 = thermospheric_density(R_EARTH + 450_000.0, 100.0, 15.0)
    assert rho_400 / rho_450 == pytest.approx(np.e)


def test_solar_and_geomagnetic_scaling():
    assert solar_factor(145.0) == pytest.approx(1.36)
    assert geomagnetic_factor(15.0) == pytest.approx(1.0)
    assert geomagnetic_factor(115.0) == pytest.approx(1.3)


def test_factor_clamps():
    assert solar_factor(0.0) == pytest.approx(0.2)      # 1 + 0.008*(-100) = 0.2
    assert solar_factor(-50.0) == pytest.approx(0.2)
    assert geomagnetic_factor(-500.0) == pytest.approx(0.5)


def test_density_floor():
    assert thermospheric_density(R_EARTH + 5_000_000.0, 100.0, 15.0) == RHO_FLOOR


def test_vectorised():
    a = R_EARTH + np.array([400e3, 469.5e3, 805e3])
    rho = thermospheric_density(a, 145.0, 15.0)
    assert rho.shape == (3,) and np.all(np.diff(rho) < 0)
