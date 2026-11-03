"""Altair chart helpers shared by the Charts and 3D-view pages."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

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
INK = {"light": "#0b0b0b", "dark": "#f0efec"}


def theme() -> str:
    try:
        return st.context.theme.type or "light"
    except AttributeError:
        return "light"


def marker_rows(events: pd.DataFrame | None, days: float, show_c: bool) -> pd.DataFrame:
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
    detail = detail.where(ev["peak_ap"].isna(),
                          detail + ", peak Ap " + ev["peak_ap"].round(0).astype("Int64").astype(str))
    out = pd.DataFrame({"day": ev["start_day"], "marker": label, "event_id": ev["event_id"],
                        "detail": detail, "start_utc": ev["start_utc"]})
    return out[out["marker"].notna() & (out["day"] <= days)]


def line_chart(df, field: str, title: str, y_title: str, color: str, fmt: str,
               log: bool = False, markers: pd.DataFrame | None = None,
               mode: str = "light", now_day: float | None = None,
               height: int = 260, marker_legend: bool = True) -> alt.LayerChart:
    """One-series line with a pointer crosshair, tooltip, optional event markers and
    a timeline cursor at ``now_day``."""
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
        idx = 0 if mode == "light" else 1
        legend = alt.Legend(orient="bottom") if marker_legend else None
        layers.append(alt.Chart(markers).mark_rule(strokeWidth=1.5, opacity=0.85).encode(
            x="day:Q",
            color=alt.Color("marker:N", title="Event",
                            scale=alt.Scale(domain=shown, range=[MARKERS[m][idx] for m in shown]),
                            legend=legend),
            strokeDash=alt.StrokeDash("marker:N", title="Event",
                                      scale=alt.Scale(domain=shown, range=[MARKERS[m][2] for m in shown]),
                                      legend=legend),
            tooltip=[alt.Tooltip("event_id:N", title="Event"), alt.Tooltip("marker:N", title="Type"),
                     alt.Tooltip("detail:N", title="Detail"),
                     alt.Tooltip("start_utc:N", title="Start (UTC)"),
                     alt.Tooltip("day:Q", title="Day", format=".2f")],
        ))

    if now_day is not None:
        now = pd.DataFrame({"day": [float(now_day)], "label": ["Timeline"]})
        layers.append(alt.Chart(now).mark_rule(color=INK.get(mode, INK["light"]), strokeWidth=1.5,
                                               opacity=0.7).encode(
            x="day:Q", tooltip=[alt.Tooltip("label:N", title=""),
                                alt.Tooltip("day:Q", title="Day", format=".2f")]))
    return alt.layer(*layers).properties(title=title, height=height)
