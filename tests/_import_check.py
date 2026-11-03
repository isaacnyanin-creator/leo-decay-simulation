"""Import checker run in a fresh interpreter (like a freshly started server).

Usage: python tests/_import_check.py ROOT [EXTRA_FILE ...]
Prints a JSON object {relative_path: error_message_or_null}.

* Library modules (importable without side effects) are imported for real.
* Scripts (Streamlit entry/pages, CLI tools) execute code at top level, so their
  import statements are checked statically: every `from X import name` must
  resolve to an existing module attribute or submodule.
"""

from __future__ import annotations

import ast
import importlib
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
APP, SRC, TOOLS = ROOT / "app", ROOT / "src", ROOT / "tools"
sys.path[:0] = [str(SRC), str(APP)]


def is_script(path: Path) -> bool:
    return path == APP / "dashboard.py" or APP / "views" in path.parents or TOOLS in path.parents


def module_name(path: Path) -> str:
    base = SRC if SRC in path.parents else APP
    parts = list(path.relative_to(base).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def check_script_imports(path: Path) -> None:
    tree = ast.parse(path.read_bytes(), filename=str(path))   # bytes: same decoding rules as Python
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                importlib.import_module(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mod = importlib.import_module(node.module)
            for alias in node.names:
                if alias.name == "*" or hasattr(mod, alias.name):
                    continue
                try:
                    importlib.import_module(f"{node.module}.{alias.name}")
                except ImportError:
                    raise ImportError(f"cannot import name {alias.name!r} from {node.module!r} "
                                      f"(line {node.lineno})") from None


def files() -> list[Path]:
    found = [p for p in APP.rglob("*.py") if "vendor" not in p.parts]
    found += list((SRC / "leo_decay").rglob("*.py")) + list(TOOLS.glob("*.py"))
    found += [Path(p).resolve() for p in sys.argv[2:]]
    return sorted(set(found))


results = {}
for path in files():
    key = path.relative_to(ROOT).as_posix() if ROOT in path.parents else str(path)
    try:
        if is_script(path) or ROOT not in path.parents:
            check_script_imports(path)
        else:
            importlib.import_module(module_name(path))
        results[key] = None
    except Exception as exc:                      # noqa: BLE001 - report every failure
        results[key] = f"{type(exc).__name__}: {exc}\n" + traceback.format_exc(limit=3)
print(json.dumps(results))
