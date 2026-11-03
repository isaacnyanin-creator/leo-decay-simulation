"""Interactive LEO decay dashboard: entry point with two pages (Charts, 3D view).

Run with:  python -m streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import streamlit as st  # noqa: E402

from common import sidebar  # noqa: E402

st.set_page_config(page_title="LEO decay dashboard", layout="wide")

nav = st.navigation([
    st.Page(str(APP_DIR / "views" / "charts.py"), title="Charts", default=True),
    st.Page(str(APP_DIR / "views" / "view3d.py"), title="3D view", url_path="3d-view"),
])
st.session_state["ctx"] = sidebar()
nav.run()
