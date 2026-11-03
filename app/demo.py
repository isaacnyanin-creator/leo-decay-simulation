"""Demo mode: run without any Space-Track data or credentials (e.g. on Streamlit Community Cloud).

Demo mode is on when ``.env`` or ``config/tle.toml`` is missing. Only file *existence* is
checked; ``.env`` is never opened. ``LEO_DEMO_MODE=1`` / ``0`` overrides the detection
(on Community Cloud, top-level secrets are exposed as environment variables).

In demo mode the 3D view always uses ``config/tle.example.toml`` (placeholder elements),
and generated solar events are the default solar mode; manual sliders stay available.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
TLE_PATH = ROOT / "config" / "tle.toml"
TLE_EXAMPLE_PATH = ROOT / "config" / "tle.example.toml"
OVERRIDE_VAR = "LEO_DEMO_MODE"
_TRUE, _FALSE = {"1", "true", "yes", "on"}, {"0", "false", "no", "off"}


@dataclass(frozen=True)
class DemoStatus:
    active: bool
    reasons: tuple[str, ...]
    forced: bool

    @property
    def summary(self) -> str:
        return "; ".join(self.reasons)


def demo_status(env_path: Path | None = None, tle_path: Path | None = None,
                environ=None) -> DemoStatus:
    environ = os.environ if environ is None else environ
    raw = str(environ.get(OVERRIDE_VAR, "")).strip().lower()
    if raw in _TRUE:
        return DemoStatus(True, (f"forced by {OVERRIDE_VAR}",), True)
    if raw in _FALSE:
        return DemoStatus(False, (), True)
    env_path = ENV_PATH if env_path is None else env_path
    tle_path = TLE_PATH if tle_path is None else tle_path
    reasons = []
    if not env_path.exists():
        reasons.append("no .env file")
    if not tle_path.exists():
        reasons.append("no config/tle.toml")
    return DemoStatus(bool(reasons), tuple(reasons), False)


def elements_path(status: DemoStatus, tle_path: Path | None = None) -> Path:
    """Orbit-element file to use: always the example file in demo mode."""
    tle_path = TLE_PATH if tle_path is None else tle_path
    if status.active or not tle_path.exists():
        return TLE_EXAMPLE_PATH
    return tle_path
