"""three.js 3D orbit view as a two-way Streamlit component (static files in this folder).

Value returned: ``{"day": float, "nonce": int}`` when the user pauses playback,
so the shared timeline can follow; ``None`` until then.
"""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT = components.declare_component("orbit3d", path=str(Path(__file__).resolve().parent))


def orbit3d(payload: dict, day: float, height: int = 660, key: str = "orbit3d"):
    return _COMPONENT(payload=payload, day=float(day), height=int(height), key=key, default=None)
