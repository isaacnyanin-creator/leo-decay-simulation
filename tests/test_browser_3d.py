"""Opt-in end-to-end check of the 3D view in headless Chromium:  pytest -m browser"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.browser
ROOT = Path(__file__).resolve().parents[1]


def test_3d_view_renders_and_syncs(tmp_path):
    pytest.importorskip("playwright")
    subprocess.run([sys.executable, str(ROOT / "tools" / "screenshot_3d.py"), str(tmp_path)],
                   check=True, timeout=600, capture_output=True)
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    shots = {s["name"]: s["state"] for s in report["shots"]}
    assert all(s["ready"] and s["textureLoaded"] and s["frames"] > 5 for s in shots.values())
    assert shots["05_generated_storm_aurora"]["auroraVisible"]
    assert not shots["01_manual_day0_x10"]["auroraVisible"]
    assert shots["06_generated_cme_wide"]["activeCmes"] > 0
    assert shots["05_generated_storm_aurora"]["runId"] != shots["01_manual_day0_x10"]["runId"]
    assert not [l for l in report["console"] if l.startswith("[pageerror]")]
    for name in shots:
        assert (tmp_path / f"{name}.png").stat().st_size > 20_000
