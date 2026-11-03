"""Combine background, flares and CMEs into a replayable solar scenario."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .background import generate_background
from .cmes import ap_profile, sample_cmes, storms_from_cmes
from .config import SolarConfig, load_solar_config
from .flares import flare_profile, sample_flares

MA_WINDOW_DAYS = 30.0          # paper: 30-day centered moving average of F10.7
F107_MODES = ("30ma", "instantaneous")

LOG_COLUMNS = ["event_id", "type", "start_utc", "start_day", "duration_h", "class_or_speed",
               "earth_directed", "peak_ap", "delta_f107", "lag_h", "parent_id"]


def centered_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """'valid' centered moving average with an odd window: output length len(x) - window + 1."""
    if window % 2 == 0:
        raise ValueError("window must be odd for a centered average")
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    return (c[window:] - c[:-window]) / window


@dataclass(frozen=True)
class SolarScenario:
    seed: int
    config: SolarConfig
    duration_days: float
    t_s: np.ndarray                 # output grid, s since start_utc, covers [0, duration]
    f107_background: np.ndarray     # sfu
    f107_instantaneous: np.ndarray  # background + flare bumps, sfu
    f107_30ma: np.ndarray           # 30-day centered average of instantaneous, sfu
    ap: np.ndarray
    flares: pd.DataFrame
    cmes: pd.DataFrame
    storms: pd.DataFrame

    @property
    def start_utc(self) -> datetime:
        return self.config.start_utc

    @property
    def duration_s(self) -> float:
        return self.duration_days * 86_400.0

    @property
    def t_days(self) -> np.ndarray:
        return self.t_s / 86_400.0

    def f107(self, mode: str = "30ma") -> np.ndarray:
        if mode not in F107_MODES:
            raise ValueError(f"mode must be one of {F107_MODES}")
        return self.f107_30ma if mode == "30ma" else self.f107_instantaneous

    def forcing(self, mode: str = "30ma"):
        """``(t_s, F10.7)`` and ``(t_s, Ap)`` pairs accepted by ``propagator.propagate``."""
        return (self.t_s, self.f107(mode)), (self.t_s, self.ap)

    @property
    def step_s(self) -> float:
        return self.config.step_hours * 3_600.0

    @property
    def events(self) -> pd.DataFrame:
        """Event log, one row per flare / CME / storm, sorted by start time."""
        return build_event_log(self)

    def to_csv_string(self) -> str:
        """Event log as CSV text with a '#' metadata header (read back with comment='#')."""
        header = (
            "# leo_decay stochastic solar event log\n"
            f"# status={self.config.status}  (rates are placeholders until calibrated)\n"
            f"# seed={self.seed}\n"
            f"# config_sha256={self.config.config_hash()}\n"
            f"# start_utc={self.start_utc.isoformat()}  duration_days={self.duration_days:g}\n"
        )
        return header + self.events.to_csv(index=False, lineterminator="\n")

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.to_csv_string(), encoding="utf-8", newline="")
        return path


def build_event_log(sc: SolarScenario) -> pd.DataFrame:
    f, c, s = sc.flares, sc.cmes, sc.storms
    storm_peak_by_cme = dict(zip(s["parent_index"], s["peak_ap"]))
    parts = [
        pd.DataFrame({
            "type": "flare", "start_day": f["start_day"], "duration_h": f["duration_h"],
            "class_or_speed": f["class_label"], "earth_directed": None,
            "peak_ap": np.nan, "delta_f107": f["delta_f107"], "lag_h": np.nan,
            "_cme": -1, "_parent": -1, "_order": 0,
        }),
        pd.DataFrame({
            "type": "cme", "start_day": c["start_day"], "duration_h": np.nan,
            "class_or_speed": [f"{v:.0f} km/s" for v in c["speed_km_s"]],
            "earth_directed": c["earth_directed"].astype(object),
            "peak_ap": [storm_peak_by_cme.get(i, np.nan) for i in c.index],
            "delta_f107": np.nan, "lag_h": np.nan,
            "_cme": c.index.to_numpy(), "_parent": -1, "_order": 1,
        }),
        pd.DataFrame({
            "type": "storm", "start_day": s["start_day"], "duration_h": s["duration_h"],
            "class_or_speed": [f"{v:.0f} km/s" for v in s["speed_km_s"]],
            "earth_directed": True, "peak_ap": s["peak_ap"], "delta_f107": np.nan,
            "lag_h": s["lag_h"], "_cme": -1, "_parent": s["parent_index"].to_numpy(), "_order": 2,
        }),
    ]
    log = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) \
        else pd.DataFrame(columns=LOG_COLUMNS + ["_cme", "_parent", "_order"])
    log = log.sort_values(["start_day", "_order"], kind="stable").reset_index(drop=True)
    log["event_id"] = [f"E{i + 1:06d}" for i in range(len(log))]
    id_by_cme = dict(zip(log.loc[log["_cme"] >= 0, "_cme"], log.loc[log["_cme"] >= 0, "event_id"]))
    log["parent_id"] = [id_by_cme.get(p, "") if p >= 0 else "" for p in log["_parent"]]
    log["start_utc"] = [(sc.start_utc + timedelta(days=float(d))).isoformat(timespec="seconds")
                        for d in log["start_day"]]
    return log[LOG_COLUMNS]


def generate_scenario(duration_days: float, seed: int | None = None,
                      config: SolarConfig | None = None) -> SolarScenario:
    """Generate F10.7(t), Ap(t) and events for ``duration_days`` from ``config.start_utc``.

    The same ``seed`` and config always give an identical scenario. Background,
    flares and CMEs use independent child streams, so changing one section of
    the config doesn't reshuffle the others.
    """
    cfg = config or load_solar_config()
    ss = np.random.SeedSequence(seed)
    seed = int(ss.entropy)
    rng_bg, rng_fl, rng_cme = (np.random.default_rng(s) for s in ss.spawn(3))

    step = cfg.step_hours / 24.0
    n_out = int(round(duration_days / step))
    if not np.isclose(n_out * step, duration_days):
        raise ValueError("duration_days must be a multiple of time.step_hours")
    window = int(round(MA_WINDOW_DAYS / step)) | 1          # odd, centered
    pad = (window - 1) // 2
    t_full = np.arange(-pad, n_out + pad + 1) * step         # padded for the centered average
    out = slice(pad, pad + n_out + 1)

    bg = generate_background(rng_bg, t_full, cfg.start_utc, cfg.background)
    driver = (t_full, bg)
    flares = sample_flares(rng_fl, duration_days, driver, cfg.flares)
    f_inst = bg + flare_profile(t_full, flares)
    f_30ma = centered_moving_average(f_inst, window)

    cmes = sample_cmes(rng_cme, duration_days, driver, cfg.cmes)
    storms = storms_from_cmes(cmes, cfg.storms)
    t_out = t_full[out]

    return SolarScenario(
        seed=seed, config=cfg, duration_days=float(duration_days),
        t_s=t_out * 86_400.0,
        f107_background=bg[out], f107_instantaneous=f_inst[out], f107_30ma=f_30ma,
        ap=ap_profile(t_out, storms, cfg.storms),
        flares=flares, cmes=cmes, storms=storms,
    )
