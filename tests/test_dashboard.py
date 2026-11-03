"""Smoke tests for the Streamlit app using streamlit.testing.AppTest."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_default_run_renders(app):
    assert not app.exception
    assert len(app.get("vega_lite_chart")) == 5
    labels = {m.label: m.value for m in app.metric}
    assert labels["Final altitude"] == "467.106 km"
    assert labels["Altitude change"] == "-2.394 km"


def test_satellite_options_labelled_by_norad(app):
    options = app.sidebar.selectbox[0].options
    assert any("20580" in o for o in options)
    assert any("25544" in o for o in options)
    assert any("25338" in o for o in options)


@pytest.mark.parametrize("key", ["iss", "noaa15"])
def test_uncalibrated_satellite_shows_warning(app, key):
    app.sidebar.selectbox[0].set_value(key).run()
    assert not app.exception
    assert app.sidebar.warning
    assert len(app.get("vega_lite_chart")) == 0


def test_doubling_mass_halves_b_star(app):
    app.sidebar.slider(key="mass_hubble").set_value(22_220).run()
    assert not app.exception
    b_star = next(m.value for m in app.metric if m.label == "Effective B*")
    assert b_star.startswith("5.7186e-03")
