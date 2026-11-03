"""Interactive LEO decay dashboard.

Run with:  python -m streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import altair as alt
import streamlit as st

from derived import is_simulatable, simulate
from leo_decay.config import load_satellites

DEFAULT_H0_KM = {"hubble": 469.5}       # Protocol 3 start
DEFAULT_F107, DEFAULT_AP, DEFAULT_DAYS = 145.0, 15.0, 35

# Reference palette, single-series charts: categorical slot 1, light/dark steps.
SERIES = {"light": "#2a78d6", "dark": "#3987e5"}
MUTED = "#898781"


def _theme() -> str:
    try:
        return st.context.theme.type or "light"
    except AttributeError:
        return "light"


@st.cache_resource
def _satellites():
    return load_satellites()


@st.cache_data(show_spinner="Integrating Eq. B.3 (RK45)...")
def _run(sat_key: str, mass_kg: float, f107: float, ap: float, h0_km: float, days: int):
    return simulate(_satellites()[sat_key], mass_kg, f107, ap, h0_km * 1_000.0, days)


def line_chart(df, field: str, title: str, y_title: str, color: str,
               fmt: str, log: bool = False) -> alt.LayerChart:
    """One-series line with a pointer crosshair and tooltip."""
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
    return alt.layer(line, hit, dot, rule).properties(title=title, height=260)


st.set_page_config(page_title="LEO decay dashboard", layout="wide")
st.title("LEO orbital decay dashboard")
st.caption("Eq. B.3 integrated with RK45 (rtol 1e-9, atol 1e-12), constant space-weather forcing.")

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
    f107 = st.slider("F10.7 (sfu)", 65.0, 300.0, DEFAULT_F107, step=1.0)
    ap = st.slider("Ap", 0.0, 100.0, DEFAULT_AP, step=1.0)
    h0_km = st.slider("Starting altitude (km)", 150.0, 1_000.0,
                      DEFAULT_H0_KM.get(sat_key, sat.nominal_altitude_m[0] / 1_000.0),
                      step=0.1, key=f"h0_{sat_key}")
    days = st.slider("Simulation length (days)", 1, 730, DEFAULT_DAYS)

result = _run(sat_key, float(mass_kg), f107, ap, h0_km, days)
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

color = SERIES.get(_theme(), SERIES["light"])
charts = [
    ("altitude_km", "Altitude", "km", ",.2f", False),
    ("decay_rate_km_day", "Decay rate (Eq. B.1)", "km/day", ".4f", False),
    ("decay_accel_km_day2", "Decay acceleration (Eq. B.2)", "km/day²", ".2e", False),
    ("density_kg_m3", "Thermospheric density", "kg/m³ (log scale)", ".2e", True),
    ("drag_force_n", "Drag force", "N", ".4f", False),
]
cols = st.columns(2)
for i, (field, title, unit, fmt, log) in enumerate(charts):
    with cols[i % 2]:
        st.altair_chart(line_chart(df, field, title, unit, color, fmt, log),
                        width="stretch")
        if field == "drag_force_n":
            st.caption("The mass slider holds C_D·A fixed, so at a given altitude the drag force "
                       "doesn't depend on mass. Mass changes the deceleration (F/m), hence how fast "
                       "the satellite descends into denser air; that is the only way it moves this curve.")

with st.expander("Table view"):
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(),
                       file_name=f"{sat_key}_decay.csv", mime="text/csv")
