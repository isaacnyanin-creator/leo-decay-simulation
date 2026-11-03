"""Load satellite definitions from ``config/satellites.toml``."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "satellites.toml"


@dataclass(frozen=True)
class Satellite:
    key: str
    name: str
    norad_id: int
    role: str
    passive: bool
    nominal_altitude_m: tuple[float, float]
    mass_kg: float | None = None
    area_m2: tuple[float, float] | None = None
    drag_coefficient: float | None = None
    b_star_static_m2_kg: float | None = None
    b_star_empirical_m2_kg: float | None = None
    b_star_empirical_sigma_m2_kg: float | None = None


def load_satellites(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Satellite]:
    """Return a mapping of config key (e.g. ``"hubble"``) to :class:`Satellite`."""
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    satellites = {}
    for key, entry in raw.items():
        entry = dict(entry)
        entry["nominal_altitude_m"] = tuple(entry["nominal_altitude_m"])
        if "area_m2" in entry:
            entry["area_m2"] = tuple(entry["area_m2"])
        satellites[key] = Satellite(key=key, **entry)
    return satellites
