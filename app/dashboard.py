"""Interactive LEO decay dashboard.

Run with:  python -m streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import altair as alt
import numpy as np
import pandas as pd
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

# Reference palette (validated: scripts/validate_palette.js, light and dark).
# Slot 1 = the data line; slots 2-6 = event markers, fixed per event type.
SERIES = {"light": "#2a78d6", "dark": "#3987e5"}
MARKERS = {   # label: (light, dark, dash). Dash patterns are the secondary encoding.
    "Storm onset":        ("#eb6834", "#d95926", [1, 0]),
    "Earth-directed CME": ("#1baf7a", "#199e70", [6, 3]),
    "X flare":            ("#eda100", "#c98500", [2, 2]),
    "M flare":            ("#e87ba4", "#d55181", [8, 2, 2, 2]),
    "C flare":            ("#008300", "#008300", [1, 3]),
}
MUTED = "#898781"


def _theme() -> str:
    try:
        return st.context.theme.type or "light"
    except AttributeError:
        return "light"


@st.cache_resource
def _satellites():
    return load_satellites()


def _solar_config_mtime() -> float:
    return DEFAULT_SOLAR_CONFIG_PATH.stat().st_mtime


@st.cache_data(show_spinner="Integrating Eq. B.3 (RK45)...")
def _run(sat_key: str, mass_kg: float, f107: float, ap: float, h0_km: float, days: int):
    return simulate(_satellites()[sat_key], mass_kg, f107, ap, h0_km * 1_000.0, days)


@st.cache_data(show_spinner="Generating solar events and integrating Eq. B.3...")
def _run_generated(sat_key: str, mass_kg: float, seed: int, h0_km: float, days: int,
                   f107_mode: str, cfg_mtime: float):
    # cfg_mtime is only part of the cache key: editing the config invalidates cached runs.
    sc = generate_scenario(days, seed=seed, config=load_solar_config())
    result = simulate_scenario(_satellites()[sat_key], mass_kg, sc, h0_km * 1_000.0, f107_mode)
    return result, sc.to_csv_string(), sc.config.is_placeholder, sc.config.config_hash()


def marker_rows(events: pd.DataFrame, days: float, show_c: bool) -> pd.DataFrame:
    """Events to draw as vertical markers, labelled by marker type."""
    if events is None or events.empty:
        return pd.DataFrame(columns=["day", "marker", "event_id", "detail", "start_utc"])
    ev = events
    label = pd.Series(pd.NA, index=ev.index, dtype="object")
    label[ev["type"] == "storm"] = "Storm onset"
    label[(ev["type"] == "cme") & (ev["earth_directed"] == True)] = "Earth-directed CME"  # noqa: E712
    is_flare = ev["type"] == "flare"
    letter = ev["class_or_speed"].astype(str).str[0]
    label[is_flare & (letter == "X")] = "X flare"
    label[is_flare & (letter == "M")] = "M flare"
    if show_c:
        label[is_flare & (letter == "C")] = "C flare"
    detail = ev["class_or_speed"].astype(str)
    detail = detail.where(ev["peak_ap"].isna(), detail + ", peak Ap " + ev["peak_ap"].round(0).astype("Int64").astype(str))
    out = pd.DataFrame({"day": ev["start_day"], "marker": label, "event_id": ev["event_id"],
                        "detail": detail, "start_utc": ev["start_utc"]})
    return out[out["marker"].notna() & (out["day"] <= days)]


def line_chart(df, field: str, title: str, y_title: str, color: str, fmt: str,
               log: bool = False, markers: pd.DataFrame | None = None,
               theme: str = "light") -> alt.LayerChart:
    """One-series line with a pointer crosshair, tooltip and optional event markers."""
    nearest = alt.selection_point(nearest=True, on="pointerover",
                                  fields=["day"], empty=False, clear="pointerout")
    base = alt.Chart(df).encode(x=alt.X("day:Q", title="Day"))
    y = alt.Y(f"{field}:Q", title=y_title,
              scale=alt.Scale(type="log") if log else alt.Scale(zero=False),
              axis=alt.Axis(format=fmt))
    line = base.mark_line(strokeWidth=2, color=color).encode(y=y)
    hit = base.mark_point(size=400, opacity=0).encode(y=y).add_params(nearest)
    dot = base.mark_point(size=80, filled=True, color=color).encode(
        y=y, opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    rule = base.mark_rule(color=MUTED).encode(
        tooltip=[alt.Tooltip("day:Q", title="Day", format=".2f"),
                 alt.Tooltip(f"{field}:Q", title=y_title, format=fmt)],
    ).transform_filter(nearest)
    layers = [line, hit, dot, rule]

    if markers is not None and not markers.empty:
        shown = [m for m in MARKERS if m in set(markers["marker"])]
        idx = 0 if theme == "light" else 1
        layers.append(alt.Chart(markers).mark_rule(strokeWidth=1.5, opacity=0.85).encode(
            x="day:Q",
            color=alt.Color("marker:N", title="Event",
                            scale=alt.Scale(domain=shown, range=[MARKERS[m][idx] for m in shown]),
                            legend=alt.Legend(orient="bottom")),
            strokeDash=alt.StrokeDash("marker:N", title="Event",
                                      scale=alt.Scale(domain=shown, range=[MARKERS[m][2] for m in shown]),
                                      legend=alt.Legend(orient="bottom")),
            tooltip=[alt.Tooltip("event_id:N", title="Event"), alt.Tooltip("marker:N", title="Type"),
                     alt.Tooltip("detail:N", title="Detail"), alt.Tooltip("start_utc:N", title="Start (UTC)"),
                     alt.Tooltip("day:Q", title="Day", format=".2f")],
        ))
    return alt.layer(*layers).properties(title=title, height=260)


st.set_page_config(page_title="LEO decay dashboard", layout="wide")
st.title("LEO orbital decay dashboard")

sats = _satellites()

with st.sidebar:
    st.header("Controls")
    sat_key = st.selectbox(
        "Satellite", list(sats),
        format_func=lambda k: f"{sats[k].name} (NORAD {sats[k].norad_id})",
    )
    sat = sats[sat_key]

    if not is_simulatable(sat):
        st.warning(
            f"{sat.name} has no calibrated B* in the paper, so it can't be simulated yet. "
            "Add `mass_kg` and `b_star_empirical_m2_kg` to its entry in "
            "`config/satellites.toml` to enable it."
        )
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

markers = None
if solar_mode == MANUAL:
    st.caption("Eq. B.3 integrated with RK45 (rtol 1e-9, atol 1e-12), constant space-weather forcing.")
    result = _run(sat_key, float(mass_kg), f107, ap, h0_km, days)
else:
    result, csv_text, placeholder, cfg_hash = _run_generated(
        sat_key, float(mass_kg), int(st.session_state.seed), h0_km, days, f107_mode,
        _solar_config_mtime())
    if placeholder:
        st.warning("**Placeholder solar model.** Event rates and distributions in "
                   "`config/solar_events.toml` are uncalibrated guesses, pending calibration "
                   "against NASA DONKI and NOAA GOES / SWPC data. Treat runs as illustrative.")
    st.caption(f"Eq. B.3 integrated piecewise with RK45 through generated F10.7(t) and Ap(t). "
               f"Seed {st.session_state.seed}, config {cfg_hash}, start "
               f"{load_solar_config().start_utc:%Y-%m-%d}.")
    if f107_mode == "instantaneous":
        st.info("Instantaneous F10.7 is outside the B* calibration, which used the 30-day average.")
    markers = marker_rows(result.events, days, show_c)

df = result.frame
if result.reentered:
    st.error(f"Re-entry (a ≤ R_E) on day {result.reentry_day:.1f}; curves end at the last sample before it.")

h_end = df["altitude_km"].iloc[-1]
dh = h_end - df["altitude_km"].iloc[0]
span = df["day"].iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Final altitude", f"{h_end:,.3f} km")
c2.metric("Altitude change", f"{dh:,.3f} km")
c3.metric("Mean decay rate", f"{dh / span if span else 0.0:.4f} km/day")
c4.metric("Effective B*", f"{result.b_star:.4e} m²/kg",
          help=f"Calibrated {sat.b_star_empirical_m2_kg:.6e} m²/kg at {m_ref:,.0f} kg")

if solar_mode == GENERATED:
    ev = result.events
    in_run = ev[ev["start_day"] <= days]
    flares = in_run[in_run["type"] == "flare"]["class_or_speed"].str[0]
    cmes = in_run[in_run["type"] == "cme"]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Storms", int((in_run["type"] == "storm").sum()))
    d2.metric("Peak Ap", f"{result.forcing['ap'].max():.0f}")
    d3.metric("M / X flares", f"{int((flares == 'M').sum())} / {int((flares == 'X').sum())}")
    d4.metric("CMEs (Earth-directed)",
              f"{len(cmes)} ({int(cmes['earth_directed'].eq(True).sum())})")

theme = _theme()
color = SERIES.get(theme, SERIES["light"])
charts = []
if solar_mode == GENERATED:
    fdf = result.forcing
    charts += [
        (fdf, "f107_sfu", f"F10.7 input ({'30-day average' if f107_mode == '30ma' else 'instantaneous'})",
         "sfu", ",.1f", False),
        (fdf, "ap", "Ap (geomagnetic index)", "Ap", ",.0f", False),
    ]
accel_title = "Decay acceleration (Eq. B.2)" if solar_mode == MANUAL \
    else "Decay acceleration (Eq. B.2, barometric term)"
charts += [
    (df, "altitude_km", "Altitude", "km", ",.2f", False),
    (df, "decay_rate_km_day", "Decay rate (Eq. B.1)", "km/day", ".4f", False),
    (df, "decay_accel_km_day2", accel_title, "km/day²", ".2e", False),
    (df, "density_kg_m3", "Thermospheric density", "kg/m³ (log scale)", ".2e", True),
    (df, "drag_force_n", "Drag force", "N", ".4f", False),
]
cols = st.columns(2)
for i, (data, field, title, unit, fmt, log) in enumerate(charts):
    with cols[i % 2]:
        st.altair_chart(line_chart(data, field, title, unit, color, fmt, log, markers, theme),
                        width="stretch")
        if field == "drag_force_n":
            st.caption("The mass slider holds C_D·A fixed, so at a given altitude the drag force "
                       "doesn't depend on mass. Mass changes the deceleration (F/m), hence how fast "
                       "the satellite descends into denser air; that is the only way it moves this curve.")
        if field == "decay_accel_km_day2" and solar_mode == GENERATED:
            st.caption("Eq. B.2 assumes constant forcing, so this shows the barometric term only. "
                       "The true d²a/dt² also has a dρ/dt term that spikes at storm onsets.")

if solar_mode == GENERATED:
    st.subheader("Solar events")
    ev = result.events
    types = st.multiselect("Event types", ["flare", "cme", "storm"],
                           default=["flare", "cme", "storm"], key="event_types")
    table = ev[ev["type"].isin(types) & (ev["start_day"] <= days)]
    st.caption(f"{len(table):,} events shown. Storm rows link to their CME via parent_id.")
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button("Download event log CSV", csv_text.encode(),
                       file_name=f"solar_events_seed{st.session_state.seed}.csv", mime="text/csv")

with st.expander("Table view"):
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(),
                       file_name=f"{sat_key}_decay.csv", mime="text/csv")
