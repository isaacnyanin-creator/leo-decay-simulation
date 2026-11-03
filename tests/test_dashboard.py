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


def _generated(app, f107_mode=None):
    app.sidebar.radio(key="solar_mode").set_value("Generated events").run()
    if f107_mode:
        app.sidebar.radio(key="f107_mode").set_value(f107_mode).run()
    return app


def test_generated_mode_renders_events(app):
    _generated(app)
    assert not app.exception
    assert len(app.get("vega_lite_chart")) == 7            # + F10.7 and Ap charts
    assert any("Placeholder solar model" in w.value for w in app.warning)
    labels = {m.label for m in app.metric}
    assert {"Storms", "Peak Ap", "M / X flares", "CMEs (Earth-directed)"} <= labels
    assert len(app.dataframe) == 2                          # event table + table view
    events = app.dataframe[0].value
    assert {"event_id", "type", "peak_ap", "delta_f107"} <= set(events.columns)


def test_generated_mode_is_replayable(app):
    _generated(app)
    first = app.dataframe[0].value.copy()
    app.sidebar.number_input(key="seed").set_value(999).run()
    assert not app.dataframe[0].value.equals(first)
    app.sidebar.number_input(key="seed").set_value(2026).run()
    assert app.dataframe[0].value.equals(first)


def test_instantaneous_mode_flagged(app):
    _generated(app, "instantaneous")
    assert not app.exception
    assert any("outside the B* calibration" in i.value for i in app.info)


def test_3d_page_renders_component(app):
    app.switch_page("views/view3d.py").run()
    assert not app.exception
    assert len(app.get("vega_lite_chart")) == 3                     # linked charts
    assert any(e.type == "component_instance" or "component" in e.type
               for e in app._tree[0].children.values() for e in _walk(e))


def _walk(node):
    yield node
    for child in getattr(node, "children", {}).values():
        yield from _walk(child)


def test_timeline_cursor_on_charts(app):
    app.sidebar.slider(key="timeline_day").set_value(10.0).run()
    assert not app.exception
    assert app.session_state["timeline_day"] == 10.0


def test_3d_view_sync_moves_timeline(app):
    app.session_state["orbit3d"] = {"day": 12.5, "nonce": 1}
    app.run()
    assert app.session_state["timeline_day"] == 12.5
    app.session_state["orbit3d"] = {"day": 99.0, "nonce": 2}          # clamped to run length
    app.run()
    assert app.session_state["timeline_day"] == 35.0


def test_doubling_mass_halves_b_star(app):
    app.sidebar.slider(key="mass_hubble").set_value(22_220).run()
    assert not app.exception
    b_star = next(m.value for m in app.metric if m.label == "Effective B*")
    assert b_star.startswith("5.7186e-03")
