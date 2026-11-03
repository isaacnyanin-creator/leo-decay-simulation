"""Every app, package and tool module imports cleanly in a fresh interpreter.

Catches errors like "cannot import name 'simulate_scenario' from 'derived'" before a
server does. Files are discovered automatically, so new modules are covered.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tests" / "_import_check.py"


def _run(*extra):
    out = subprocess.run([sys.executable, str(CHECKER), str(ROOT), *map(str, extra)],
                         capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout)


@pytest.fixture(scope="module")
def results():
    return _run()


def _expected_files():
    app = [p for p in (ROOT / "app").rglob("*.py") if "vendor" not in p.parts]
    src = list((ROOT / "src" / "leo_decay").rglob("*.py"))
    tools = list((ROOT / "tools").glob("*.py"))
    return sorted(p.relative_to(ROOT).as_posix() for p in app + src + tools)


def test_discovers_all_modules(results):
    names = set(results)
    assert set(_expected_files()) <= names
    for must in ["app/derived.py", "app/common.py", "app/scene.py", "app/charts_lib.py",
                 "app/dashboard.py", "app/views/charts.py", "app/views/view3d.py",
                 "app/components/orbit3d/__init__.py", "src/leo_decay/solar/coupling.py",
                 "tools/fetch_tles.py"]:
        assert must in names


@pytest.mark.parametrize("path", _expected_files())
def test_module_imports(results, path):
    assert results[path] is None, results[path]


def test_checker_catches_missing_name(tmp_path):
    """The exact failure from the stale server must be reported, not silently pass."""
    bad = tmp_path / "bad_page.py"
    bad.write_text("from derived import simulate, does_not_exist\n", encoding="utf-8")
    good = tmp_path / "good_page.py"
    good.write_text("from derived import simulate, simulate_scenario\n", encoding="utf-8")
    res = _run(bad, good)
    assert res[str(good.resolve())] is None
    err = res[str(bad.resolve())]
    assert err and "cannot import name 'does_not_exist' from 'derived'" in err
