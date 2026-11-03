"""Credentials and Space-Track data stay out of git: ignored now, absent from all history."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = [".env", "config/tle.toml", "data/raw/spacetrack/gp_history_20580.json",
             "data/raw/spacetrack/ledger.json", "data/processed/protocol3_residuals.csv"]


def _git():
    exe = shutil.which("git") or next(
        (p for p in [r"C:\Program Files\Git\cmd\git.exe"] if Path(p).exists()), None)
    if exe is None or not (ROOT / ".git").exists():
        pytest.skip("git or repository not available")
    return lambda *args: subprocess.run([exe, *args], cwd=ROOT, capture_output=True, text=True,
                                        check=False, timeout=120)


@pytest.mark.parametrize("path", PROTECTED)
def test_path_is_gitignored(path):
    git = _git()
    assert git("check-ignore", "-q", path).returncode == 0, f"{path} is not gitignored"


def test_nothing_protected_is_tracked():
    tracked = _git()("ls-files").stdout.splitlines()
    bad = [p for p in tracked if p == ".env" or p.endswith("/.env") or p == "config/tle.toml"
           or p.startswith("data/")]
    assert bad == []


def test_never_in_history():
    out = _git()("log", "--all", "--full-history", "--pretty=format:", "--name-only").stdout
    ever = {line for line in out.splitlines() if line}
    bad = sorted(p for p in ever if p == ".env" or p.endswith("/.env") or p == "config/tle.toml"
                 or p.startswith("data/"))
    assert bad == [], f"protected paths appear in history: {bad}"


def test_example_tle_is_placeholder_only():
    text = (ROOT / "config" / "tle.example.toml").read_text(encoding="utf-8")
    assert "PLACEHOLDER" in text and "26001.00000000" in text      # fake epoch, not real elements
