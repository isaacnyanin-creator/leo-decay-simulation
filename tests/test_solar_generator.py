from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from leo_decay.solar import generate_scenario, load_solar_config
from leo_decay.solar.config import DEFAULT_SOLAR_CONFIG_PATH
from leo_decay.solar.generator import LOG_COLUMNS


@pytest.fixture(scope="module")
def scenario():
    return generate_scenario(365, seed=42)


def _same(a, b):
    for name in ("t_s", "f107_background", "f107_instantaneous", "f107_30ma", "ap"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    pd.testing.assert_frame_equal(a.events, b.events)


def test_same_seed_replays_exactly(scenario):
    _same(scenario, generate_scenario(365, seed=42))


def test_different_seed_differs(scenario):
    other = generate_scenario(365, seed=43)
    assert not np.array_equal(scenario.f107_instantaneous, other.f107_instantaneous)


def test_unseeded_run_records_replayable_seed():
    a = generate_scenario(30)
    _same(a, generate_scenario(30, seed=a.seed))


def test_streams_independent(solar_cfg):
    a = generate_scenario(200, seed=7, config=solar_cfg)
    cfg2 = replace(solar_cfg, flares=replace(solar_cfg.flares, base_rate_per_day=9.0))
    b = generate_scenario(200, seed=7, config=cfg2)
    pd.testing.assert_frame_equal(a.cmes, b.cmes)          # flare change didn't reshuffle CMEs
    np.testing.assert_array_equal(a.ap, b.ap)
    assert len(b.flares) > len(a.flares)


def test_grid_and_series_shapes(scenario, solar_cfg):
    step_s = solar_cfg.step_hours * 3600
    assert scenario.t_s[0] == 0.0 and scenario.t_s[-1] == pytest.approx(365 * 86400)
    np.testing.assert_allclose(np.diff(scenario.t_s), step_s)
    for arr in (scenario.f107_instantaneous, scenario.f107_30ma, scenario.ap):
        assert arr.shape == scenario.t_s.shape
    assert (scenario.f107_instantaneous >= scenario.f107_background - 1e-12).all()  # flares only add
    assert scenario.f107_30ma.std() < scenario.f107_instantaneous.std()            # smoothing


def test_flat_config_gives_constant_series(flat_solar_cfg):
    sc = generate_scenario(35, seed=1, config=flat_solar_cfg(145.0))
    np.testing.assert_allclose(sc.f107_30ma, 145.0)
    np.testing.assert_allclose(sc.f107_instantaneous, 145.0)
    np.testing.assert_allclose(sc.ap, 15.0)
    assert sc.events.empty


def test_bad_duration():
    with pytest.raises(ValueError):
        generate_scenario(1.01, seed=1)


def test_event_log_csv(scenario, tmp_path):
    path = scenario.to_csv(tmp_path / "events.csv")
    text = path.read_text(encoding="utf-8")
    assert "# seed=42" in text and "status=PLACEHOLDER" in text
    assert f"config_sha256={scenario.config.config_hash()}" in text
    log = pd.read_csv(path, comment="#", keep_default_na=True)
    assert list(log.columns) == LOG_COLUMNS
    assert log["event_id"].is_unique
    assert set(log["type"]) == {"flare", "cme", "storm"}
    assert log["start_day"].is_monotonic_increasing


def test_event_log_links_storms_to_cmes(scenario):
    log = scenario.events
    cmes = log[log["type"] == "cme"].set_index("event_id")
    storms = log[log["type"] == "storm"]
    assert len(storms) > 0
    parents = cmes.loc[storms["parent_id"]]
    assert parents["earth_directed"].eq(True).all()
    lag_h = (storms["start_day"].to_numpy() - parents["start_day"].to_numpy()) * 24
    np.testing.assert_allclose(lag_h, storms["lag_h"], atol=1e-6)
    assert ((lag_h >= 15 - 1e-9) & (lag_h <= 72 + 1e-9)).all()
    np.testing.assert_allclose(parents["peak_ap"], storms["peak_ap"].to_numpy())
    flares = log[log["type"] == "flare"]
    assert flares["delta_f107"].gt(0).all() and flares["peak_ap"].isna().all()


def test_config_marked_placeholder(solar_cfg):
    assert solar_cfg.is_placeholder
    text = DEFAULT_SOLAR_CONFIG_PATH.read_text(encoding="utf-8")
    for section in ("[background]", "[flares]", "[cmes]", "[storms]"):
        line = next(l for l in text.splitlines() if l.startswith(section))
        assert "PLACEHOLDER" in line and "calibrate" in line


def test_config_validation(tmp_path):
    text = DEFAULT_SOLAR_CONFIG_PATH.read_text(encoding="utf-8")
    bad = tmp_path / "bad.toml"
    bad.write_text(text.replace("peak_ap_min = 32.0", "peak_ap_min = 25.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="exceed 30"):
        load_solar_config(bad)
    extra = tmp_path / "extra.toml"
    extra.write_text(text.replace("[cmes]", "[cmes]\nbogus = 1.0", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_solar_config(extra)
