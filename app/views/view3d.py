"""3D view page: three.js scene + compact linked charts. Displays engine output only."""

import pandas as pd
import streamlit as st

from charts_lib import SERIES, line_chart, marker_rows, theme
from common import GENERATED, demo_banner
from components.orbit3d import orbit3d
from demo import elements_path
from leo_decay.solar import load_solar_config
from leo_decay.solar.cmes import transit_lag_hours
from scene import TLE_PATH, build_payload, load_elements

ctx = st.session_state["ctx"]
result = ctx.result

st.title("3D orbit view")
demo_banner(ctx)
try:
    elements = load_elements(ctx.sat_key, elements_path(ctx.demo))
except KeyError as exc:
    st.error(str(exc))
    st.stop()
if elements.placeholder and not ctx.demo.active:
    st.warning(f"`{TLE_PATH.name}` not found, so the orbit plane uses **placeholder elements** from "
               "`config/tle.example.toml`. Run `python tools/fetch_tles.py` to download real ones "
               "from Space-Track.")

solar_cfg = load_solar_config()
payload = build_payload(
    result.frame, result.forcing, result.events, elements, solar_cfg.start_utc,
    ctx.sat.name, ctx.run_id,
    lag_fn=lambda v: float(transit_lag_hours(v, solar_cfg.storms)),
    demo=ctx.demo.active,
)
st.caption(f"Simulation start {solar_cfg.start_utc:%Y-%m-%d %H:%M} UTC. Orbit radius, decay rate and "
           "density are engine output; the scene adds display geometry only (Sun direction, Earth "
           "rotation, orbit orientation from the TLE).")
orbit3d(payload, day=ctx.timeline_day, key="orbit3d")

st.subheader("Linked charts")
mode = theme()
color = SERIES.get(mode, SERIES["light"])
markers = marker_rows(result.events, ctx.days, ctx.show_c) if ctx.solar_mode == GENERATED else None
cols = st.columns(3)
specs = [
    (result.frame, "altitude_km", "Altitude", "km", ",.2f"),
    (result.frame, "decay_rate_km_day", "Decay rate", "km/day", ".4f"),
    (result.forcing, "ap", "Ap", "Ap", ",.0f"),
]
for col, (data, field, title, unit, fmt) in zip(cols, specs):
    with col:
        st.altair_chart(line_chart(data, field, title, unit, color, fmt, markers=markers, mode=mode,
                                   now_day=ctx.timeline_day, height=300, marker_legend=False),
                        width="stretch")
st.caption("Dark vertical line: the shared timeline (pause the 3D view to move it). "
           + ("Event markers use the same colours and dash patterns as the Charts page legend."
              if markers is not None else ""))
