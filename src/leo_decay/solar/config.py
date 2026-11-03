"""Load ``config/solar_events.toml`` into frozen dataclasses.

Use ``dataclasses.replace`` to override values (e.g. in tests).
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

DEFAULT_SOLAR_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "solar_events.toml"
)


@dataclass(frozen=True)
class BackgroundConfig:
    cycle_min_utc: datetime
    cycle_period_days: float
    f107_min: float
    f107_max: float
    rotation_period_days: float
    rotation_amplitude_min_sfu: float
    rotation_amplitude_frac: float
    noise_sigma_sfu: float
    noise_correlation_days: float
    f107_floor: float


@dataclass(frozen=True)
class FlareConfig:
    base_rate_per_day: float
    rate_reference_f107: float
    rate_f107_exponent: float
    min_flux_w_m2: float
    max_flux_w_m2: float
    power_law_index: float
    delta_f107_at_c1: float
    delta_f107_flux_exponent: float
    rise_hours: float
    decay_efold_hours_median: float
    decay_efold_hours_sigma_ln: float
    duration_efolds: float


@dataclass(frozen=True)
class CmeConfig:
    base_rate_per_day: float
    rate_reference_f107: float
    rate_f107_exponent: float
    speed_median_km_s: float
    speed_sigma_ln: float
    speed_min_km_s: float
    speed_max_km_s: float
    earth_directed_probability: float


@dataclass(frozen=True)
class StormConfig:
    transit_distance_km: float
    lag_min_hours: float
    lag_max_hours: float
    quiet_ap: float
    ap_cap: float
    peak_ap_min: float
    peak_ap_at_reference_speed: float
    reference_speed_km_s: float
    peak_ap_per_km_s: float
    main_phase_hours: float
    recovery_efold_hours: float
    duration_efolds: float


@dataclass(frozen=True)
class SolarConfig:
    status: str
    start_utc: datetime
    step_hours: float
    background: BackgroundConfig
    flares: FlareConfig
    cmes: CmeConfig
    storms: StormConfig

    @property
    def is_placeholder(self) -> bool:
        return self.status.upper() == "PLACEHOLDER"

    def config_hash(self) -> str:
        """Short SHA-256 of the parsed values (formatting-independent)."""
        blob = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def _section(cls, raw: dict, name: str):
    expected = {f.name for f in fields(cls)}
    got = set(raw)
    if missing := expected - got:
        raise ValueError(f"[{name}] missing keys: {sorted(missing)}")
    if extra := got - expected:
        raise ValueError(f"[{name}] unknown keys: {sorted(extra)}")
    values = {}
    for f in fields(cls):
        v = raw[f.name]
        values[f.name] = datetime.fromisoformat(v) if f.type == "datetime" else float(v)
    return cls(**values)


def _validate(cfg: SolarConfig) -> None:
    b, fl, c, s = cfg.background, cfg.flares, cfg.cmes, cfg.storms
    checks = [
        (cfg.step_hours > 0, "time.step_hours must be > 0"),
        (b.f107_max >= b.f107_min, "background.f107_max must be >= f107_min"),
        (b.noise_correlation_days > 0, "background.noise_correlation_days must be > 0"),
        (fl.base_rate_per_day >= 0 and c.base_rate_per_day >= 0, "rates must be >= 0"),
        (0 < fl.min_flux_w_m2 < fl.max_flux_w_m2, "flare flux bounds invalid"),
        (fl.power_law_index > 1, "flares.power_law_index must be > 1"),
        (0 <= c.earth_directed_probability <= 1, "earth_directed_probability must be in [0, 1]"),
        (0 < c.speed_min_km_s < c.speed_max_km_s, "CME speed bounds invalid"),
        (0 < s.lag_min_hours <= s.lag_max_hours, "storm lag bounds invalid"),
        (s.peak_ap_min > 30, "storms.peak_ap_min must exceed 30"),
        (s.quiet_ap < s.peak_ap_min <= s.ap_cap, "need quiet_ap < peak_ap_min <= ap_cap"),
        (s.recovery_efold_hours > 0 and s.main_phase_hours >= 0, "storm timescales invalid"),
    ]
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def load_solar_config(path: str | Path = DEFAULT_SOLAR_CONFIG_PATH) -> SolarConfig:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    cfg = SolarConfig(
        status=str(raw["status"]),
        start_utc=datetime.fromisoformat(raw["time"]["start_utc"]),
        step_hours=float(raw["time"]["step_hours"]),
        background=_section(BackgroundConfig, raw["background"], "background"),
        flares=_section(FlareConfig, raw["flares"], "flares"),
        cmes=_section(CmeConfig, raw["cmes"], "cmes"),
        storms=_section(StormConfig, raw["storms"], "storms"),
    )
    _validate(cfg)
    return cfg
