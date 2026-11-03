"""Sidebar controls, cached simulation runs and the shared timeline (both pages)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import streamlit as st

from derived import is_simulatable, simulate, simulate_scenario
from leo_decay.config import load_satellites
from leo_decay.solar import generate_scenario, load_solar_config
from leo_decay.solar.config import DEFAULT_SOLAR_CONFIG_PATH

DEFAULT_H0_KM = {"hubble": 469.5}       # Protocol 3 start
DEFAULT_F107, DEFAULT_AP, DEFAULT_DAYS = 145.0, 15.0, 35
DEFAULT_SEED = 2026
MANUAL, GENERATED = "Manual sliders", "Generated events"
F107_MODE_LABELS = {"30ma": "30-day average (paper, matches B* calibration)",
                    "instantaneous": "Instantaneous (outside the B* calibration)"}
TIMELINE_KEY, ORBIT3D_KEY = "timeline_day", "orbit3d"


@st.cache_resource
def satellites():
    return load_satellites()


def _solar_config_mtime() -> float:
    return DEFAULT_SOLAR_CONFIG_PATH.stat().st_mtime


@st.cache_data(show_spinner="Integrating Eq. B.3 (RK45)...")
def _run_manual(sat_key: str, mass_kg: float, f107: float, ap: float, h0_km: float, days: int):
    return simulate(satellites()[sat_key], mass_kg, f107, ap, h0_km * 1_000.0, days)


@st.cache_data(show_spinner="Generating solar events and integrating Eq. B.3...")
def _run_generated(sat_key: str, mass_kg: float, seed: int, h0_km: float, days: int,
                   f107_mode: str, cfg_mtime: float):
    # cfg_mtime is only part of the cache key: editing the config invalidates cached runs.
    sc = generate_scenario(days, seed=seed, config=load_solar_config())
    result = simulate_scenario(satellites()[sat_key], mass_kg, sc, h0_km * 1_000.0, f107_mode)
    return result, sc.to_csv_string(), sc.config.is_placeholder, sc.config.config_hash()


@dataclass
class Context:
    sat_key: str
    sat: object
    mass_kg: float
    h0_km: float
    days: int
    solar_mode: str
    f107: float | None
    ap: float | None
    seed: int | None
    f107_mode: str | None
    show_c: bool
    result: object
    csv_text: str | None
    placeholder: bool
    cfg_hash: str | None
    timeline_day: float
    run_id: str


def _sync_timeline_from_3d(days: int) -> None:
    """When the 3D view reports a paused time, move the shared timeline there."""
    value = st.session_state.get(ORBIT3D_KEY)
    if isinstance(value, dict) and value.get("nonce") != st.session_state.get("_orbit3d_nonce"):
        st.session_state["_orbit3d_nonce"] = value.get("nonce")
        st.session_state[TIMELINE_KEY] = float(np.clip(value.get("day", 0.0), 0.0, days))
    if TIMELINE_KEY not in st.session_state:
        st.session_state[TIMELINE_KEY] = 0.0
    st.session_state[TIMELINE_KEY] = float(np.clip(st.session_state[TIMELINE_KEY], 0.0, days))


def sidebar() -> Context:
    """Render the shared sidebar, run (or fetch cached) simulation, return the context."""
    sats = satellites()
    with st.sidebar:
        st.header("Controls")
        sat_key = st.selectbox("Satellite", list(sats),
                               format_func=lambda k: f"{sats[k].name} (NORAD {sats[k].norad_id})")
        sat = sats[sat_key]
        if not is_simulatable(sat):
            st.warning(
                f"{sat.name} has no calibrated B* in the paper, so it can't be simulated yet. "
                "Add `mass_kg` and `b_star_empirical_m2_kg` to its entry in "
                "`config/satellites.toml` to enable it.")
            st.stop()

        m_ref = sat.mass_kg
        mass_kg = st.slider("Mass (kg)", min_value=round(0.25 * m_ref), max_value=round(4 * m_ref),
                            value=round(m_ref), step=10, key=f"mass_{sat_key}",
                            help=f"Reference mass {m_ref:,.0f} kg. B* = B*_cal × m_ref / m_new.")
        h0_km = st.slider("Starting altitude (km)", 150.0, 1_000.0,
                          DEFAULT_H0_KM.get(sat_key, sat.nominal_altitude_m[0] / 1_000.0),
                          step=0.1, key=f"h0_{sat_key}")
        days = st.slider("Simulation length (days)", 1, 730, DEFAULT_DAYS)

        st.divider()
        solar_mode = st.radio("Solar mode", [MANUAL, GENERATED], key="solar_mode")
        f107 = ap = seed = f107_mode = None
        show_c = False
        if solar_mode == MANUAL:
            f107 = st.slider("F10.7 (sfu)", 65.0, 300.0, DEFAULT_F107, step=1.0)
            ap = st.slider("Ap", 0.0, 100.0, DEFAULT_AP, step=1.0)
        else:
            if "seed" not in st.session_state:
                st.session_state.seed = DEFAULT_SEED
            st.number_input("Random seed", min_value=0, max_value=2**31 - 1, step=1, key="seed",
                            help="The same seed and config always replay the same events.")
            st.button("New seed", on_click=lambda: st.session_state.update(
                seed=int(np.random.default_rng().integers(0, 2**31 - 1))))
            f107_mode = st.radio("F10.7 input to density", list(F107_MODE_LABELS),
                                 format_func=F107_MODE_LABELS.get, key="f107_mode")
            show_c = st.checkbox("Show C-class flare markers", value=False,
                                 help="C flares are frequent; hidden by default to keep charts readable.")
            seed = int(st.session_state.seed)

        st.divider()
        _sync_timeline_from_3d(days)
        timeline_day = st.slider("Timeline (day)", 0.0, float(days), step=0.01 if days <= 60 else 0.1,
                                 key=TIMELINE_KEY,
                                 help="Shared by the charts' cursor and the 3D view. Pausing the "
                                      "3D animation moves it.")

    csv_text, placeholder, cfg_hash = None, False, None
    if solar_mode == MANUAL:
        result = _run_manual(sat_key, float(mass_kg), f107, ap, h0_km, days)
    else:
        result, csv_text, placeholder, cfg_hash = _run_generated(
            sat_key, float(mass_kg), seed, h0_km, days, f107_mode, _solar_config_mtime())

    run_id = hashlib.sha256(json.dumps(
        [sat_key, mass_kg, h0_km, days, solar_mode, f107, ap, seed, f107_mode, cfg_hash],
        default=str).encode()).hexdigest()[:16]
    return Context(sat_key, sat, float(mass_kg), h0_km, days, solar_mode, f107, ap, seed, f107_mode,
                   show_c, result, csv_text, placeholder, cfg_hash, float(timeline_day), run_id)
