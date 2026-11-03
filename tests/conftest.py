import pytest

from leo_decay.config import load_satellites


@pytest.fixture(scope="session")
def satellites():
    return load_satellites()


@pytest.fixture(scope="session")
def hubble(satellites):
    return satellites["hubble"]
