"""Charts page."""

import streamlit as st

from charts_lib import SERIES, line_chart, marker_rows, theme
from common import GENERATED, MANUAL, demo_banner
from leo_decay.solar import load_solar_config

ctx = st.session_state["ctx"]
result, df, days, sat = ctx.result, ctx.result.frame, ctx.days, ctx.sat

st.title("LEO orbital decay dashboard")
demo_banner(ctx)

markers = None
if ctx.solar_mode == MANUAL:
    st.caption("Eq. B.3 integrated with RK45 (rtol 1e-9, atol 1e-12), constant space-weather forcing.")
else:
    if ctx.placeholder:
        st.warning("**Placeholder solar model.** Event rates and distributions in "
                   "`config/solar_events.toml` are uncalibrated guesses, pending calibration "
                   "against NASA DONKI and NOAA GOES / SWPC data. Treat runs as illustrative.")
    st.caption(f"Eq. B.3 integrated piecewise with RK45 through generated F10.7(t) and Ap(t). "
               f"Seed {ctx.seed}, config {ctx.cfg_hash}, start "
               f"{load_solar_config().start_utc:%Y-%m-%d}.")
    if ctx.f107_mode == "instantaneous":
        st.info("Instantaneous F10.7 is outside the B* calibration, which used the 30-day average.")
    markers = marker_rows(result.events, days, ctx.show_c)

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
          help=f"Calibrated {sat.b_star_empirical_m2_kg:.6e} m²/kg at {sat.mass_kg:,.0f} kg")

if ctx.solar_mode == GENERATED:
    ev = result.events
    in_run = ev[ev["start_day"] <= days]
    flares = in_run[in_run["type"] == "flare"]["class_or_speed"].str[0]
    cmes = in_run[in_run["type"] == "cme"]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Storms", int((in_run["type"] == "storm").sum()))
    d2.metric("Peak Ap", f"{result.forcing['ap'].max():.0f}")
    d3.metric("M / X flares", f"{int((flares == 'M').sum())} / {int((flares == 'X').sum())}")
    d4.metric("CMEs (Earth-directed)", f"{len(cmes)} ({int(cmes['earth_directed'].eq(True).sum())})")

mode = theme()
color = SERIES.get(mode, SERIES["light"])
charts = []
if ctx.solar_mode == GENERATED:
    fdf = result.forcing
    charts += [
        (fdf, "f107_sfu", f"F10.7 input ({'30-day average' if ctx.f107_mode == '30ma' else 'instantaneous'})",
         "sfu", ",.1f", False),
        (fdf, "ap", "Ap (geomagnetic index)", "Ap", ",.0f", False),
    ]
accel_title = "Decay acceleration (Eq. B.2)" if ctx.solar_mode == MANUAL \
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
        st.altair_chart(line_chart(data, field, title, unit, color, fmt, log, markers, mode,
                                   now_day=ctx.timeline_day), width="stretch")
        if field == "drag_force_n":
            st.caption("The mass slider holds C_D·A fixed, so at a given altitude the drag force "
                       "doesn't depend on mass. Mass changes the deceleration (F/m), hence how fast "
                       "the satellite descends into denser air; that is the only way it moves this curve.")
        if field == "decay_accel_km_day2" and ctx.solar_mode == GENERATED:
            st.caption("Eq. B.2 assumes constant forcing, so this shows the barometric term only. "
                       "The true d²a/dt² also has a dρ/dt term that spikes at storm onsets.")

if ctx.solar_mode == GENERATED:
    st.subheader("Solar events")
    ev = result.events
    types = st.multiselect("Event types", ["flare", "cme", "storm"],
                           default=["flare", "cme", "storm"], key="event_types")
    table = ev[ev["type"].isin(types) & (ev["start_day"] <= days)]
    st.caption(f"{len(table):,} events shown. Storm rows link to their CME via parent_id.")
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button("Download event log CSV", ctx.csv_text.encode(),
                       file_name=f"solar_events_seed{ctx.seed}.csv", mime="text/csv")

with st.expander("Table view"):
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(),
                       file_name=f"{ctx.sat_key}_decay.csv", mime="text/csv")
