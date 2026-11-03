"""Demo mode: detection, element source, on-screen labels, and no Space-Track code paths."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import demo
from demo import TLE_EXAMPLE_PATH, demo_status, elements_path

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "dashboard.py")


# ---------------------------------------------------------------- detection
@pytest.fixture
def files(tmp_path):
    env, tle = tmp_path / ".env", tmp_path / "tle.toml"
    return env, tle


@pytest.mark.parametrize("have_env,have_tle,active", [
    (False, False, True), (True, False, True), (False, True, True), (True, True, False)])
def test_demo_when_either_file_missing(files, have_env, have_tle, active):
    env, tle = files
    if have_env:
        env.write_text("", encoding="utf-8")
    if have_tle:
        tle.write_text("", encoding="utf-8")
    st = demo_status(env, tle, environ={})
    assert st.active is active and not st.forced
    assert ("no .env file" in st.reasons) is (not have_env)
    assert ("no config/tle.toml" in st.reasons) is (not have_tle)


@pytest.mark.parametrize("value,active", [("1", True), ("true", True), ("ON", True),
                                          ("0", False), ("false", False), ("off", False)])
def test_override(files, value, active):
    env, tle = files                                   # both missing
    if not active:
        assert demo_status(env, tle, environ={"LEO_DEMO_MODE": value}).active is False
    else:
        env.write_text("", encoding="utf-8")
        tle.write_text("", encoding="utf-8")
        st = demo_status(env, tle, environ={"LEO_DEMO_MODE": value})
        assert st.active and st.forced


def test_unrecognised_override_falls_back_to_detection(files):
    env, tle = files
    assert demo_status(env, tle, environ={"LEO_DEMO_MODE": "maybe"}).active is True


def test_env_file_is_never_opened(files, monkeypatch):
    env, tle = files
    env.write_text("SPACETRACK_USER=x\n", encoding="utf-8")
    tle.write_text("", encoding="utf-8")
    opened = []
    real_open = Path.open
    monkeypatch.setattr(Path, "open", lambda self, *a, **k: opened.append(self) or real_open(self, *a, **k))
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: opened.append(self) or "")
    demo_status(env, tle, environ={})
    assert env not in opened


def test_demo_always_uses_example_elements(files):
    env, tle = files
    tle.write_text("[hubble]\n", encoding="utf-8")
    active = demo_status(env, tle, environ={})            # .env missing -> demo
    assert active.active and elements_path(active, tle) == TLE_EXAMPLE_PATH
    full = demo_status(env, tle, environ={"LEO_DEMO_MODE": "0"})
    assert elements_path(full, tle) == tle
    assert elements_path(full, tle.with_name("missing.toml")) == TLE_EXAMPLE_PATH


# ---------------------------------------------------------------- app behaviour
@pytest.fixture
def demo_app(monkeypatch):
    monkeypatch.setenv("LEO_DEMO_MODE", "1")
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def _demo_infos(at):
    return [i.value for i in at.info if "Demo mode" in i.value]


def test_demo_defaults_to_generated_events_with_label(demo_app):
    at = demo_app
    assert not at.exception
    assert at.sidebar.radio(key="solar_mode").value == "Generated events"
    assert len(at.get("vega_lite_chart")) == 7
    assert _demo_infos(at) and "tle.example.toml" in _demo_infos(at)[0]
    # st.badge renders as badge markdown in the sidebar
    assert any("badge[" in m.value and "Demo mode" in m.value for m in at.sidebar.markdown)


def test_demo_keeps_manual_sliders_as_option(demo_app):
    at = demo_app
    at.sidebar.radio(key="solar_mode").set_value("Manual sliders").run()
    assert not at.exception
    assert len(at.get("vega_lite_chart")) == 5
    assert {s.label for s in at.sidebar.slider} >= {"F10.7 (sfu)", "Ap"}
    assert _demo_infos(at)                                 # label stays on


def test_demo_3d_page_label_and_placeholder_elements(demo_app):
    at = demo_app
    at.switch_page("views/view3d.py").run()
    assert not at.exception
    assert _demo_infos(at)
    assert not [w for w in at.warning if "not found" in w.value]   # demo banner replaces it


def test_full_mode_has_no_demo_label(monkeypatch):
    monkeypatch.setenv("LEO_DEMO_MODE", "0")
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception and not _demo_infos(at)
    assert not any("Demo mode" in m.value for m in at.sidebar.markdown)
    assert at.sidebar.radio(key="solar_mode").value == "Manual sliders"


def test_demo_payload_flag(hubble):
    from datetime import datetime, timezone
    from derived import simulate
    from scene import build_payload, load_elements
    res = simulate(hubble, hubble.mass_kg, 145.0, 15.0, 469_500.0, 5, n_points=20)
    el = load_elements("hubble", elements_path(demo.DemoStatus(True, ("x",), True)))
    p = build_payload(res.frame, res.forcing, None, el, datetime(2026, 10, 15, tzinfo=timezone.utc),
                      "HST", "r", demo=True)
    assert p["demo"] is True and p["elements"]["placeholder"] is True


def test_demo_run_never_loads_spacetrack_or_dotenv():
    """Fresh interpreter: render both pages in demo mode, then inspect sys.modules."""
    code = f"""
import os, sys, json
os.environ["LEO_DEMO_MODE"] = "1"
from streamlit.testing.v1 import AppTest
at = AppTest.from_file({APP!r}, default_timeout=120); at.run()
at.switch_page("views/view3d.py").run()
bad = [m for m in ("dotenv", "spacetrack", "leo_decay.spacetrack") if m in sys.modules]
print(json.dumps({{"exception": bool(at.exception), "loaded": bad}}))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                         cwd=ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res == {"exception": False, "loaded": []}
