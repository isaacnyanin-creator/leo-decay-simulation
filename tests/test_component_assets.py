"""Every file the three.js component references exists with exact case (Linux hosts are
case-sensitive) and is referenced by a relative URL (so it works behind any host/prefix)."""

import json
import os
import re
from pathlib import Path

import pytest

COMP = Path(__file__).resolve().parents[1] / "app" / "components" / "orbit3d"
IMPORT_RE = re.compile(r"""(?:\bfrom\s*|\bimport\s*\(?\s*)['"]([^'"\n]+)['"]""")
ASSET_RE = re.compile(r"""['"](\./assets/[^'"\n]+)['"]""")


def _html():
    return (COMP / "index.html").read_text(encoding="utf-8")


def _importmap():
    m = re.search(r'<script type="importmap">(.*?)</script>', _html(), re.S)
    return json.loads(m.group(1))["imports"]


def _exists_exact_case(rel: str) -> bool:
    """True only if every path component matches the on-disk name exactly."""
    cur = COMP
    for part in Path(rel).parts:
        if part in (".", ""):
            continue
        if part not in os.listdir(cur):
            return False
        cur = cur / part
    return cur.is_file()


def _resolve(spec: str, from_file: Path, imports: dict) -> str:
    if spec in imports:
        spec = imports[spec]
    else:
        for prefix, target in imports.items():
            if prefix.endswith("/") and spec.startswith(prefix):
                spec = target + spec[len(prefix):]
                break
        else:
            assert spec.startswith("."), f"bare or absolute import {spec!r} in {from_file.name}"
            spec = os.path.relpath((from_file.parent / spec).resolve(), COMP)
    return os.path.normpath(spec).replace("\\", "/")


def _all_references():
    imports = _importmap()
    html = _html()
    refs = set(re.findall(r'(?:href|src)="([^"]+)"', html)) | set(imports.values())
    todo, seen = [COMP / "main.js"], set()
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        text = f.read_text(encoding="utf-8")
        for spec in IMPORT_RE.findall(text):
            if not (spec.startswith(".") or spec.startswith("three")):
                continue                                   # e.g. words in comments
            rel = _resolve(spec, f, imports)
            refs.add(rel)
            if rel.endswith(".js") and _exists_exact_case(rel):
                todo.append(COMP / rel)
        refs |= set(ASSET_RE.findall(text))
    return sorted(os.path.normpath(r).replace("\\", "/") + ("/" if r.endswith("/") else "")
                  for r in refs)


def test_references_are_relative():
    for ref in re.findall(r'(?:href|src)="([^"]+)"', _html()) + list(_importmap().values()):
        assert ref.startswith("./"), f"{ref!r} is not relative; it would break under a URL prefix"
    for f in COMP.glob("*.js"):
        text = f.read_text(encoding="utf-8")
        assert "localhost" not in text and "127.0.0.1" not in text, f.name


def test_found_the_expected_graph():
    refs = _all_references()
    for must in ["main.js", "style.css", "geometry.js", "vendor/three.module.js",
                 "vendor/three.core.js", "vendor/addons/controls/OrbitControls.js",
                 "assets/blue_marble.jpg"]:
        assert must in refs, refs


@pytest.mark.parametrize("ref", _all_references())
def test_reference_exists_with_exact_case(ref):
    if ref.endswith("/"):                                  # import-map prefix
        assert (COMP / ref).is_dir() and Path(ref).parts[-1] in os.listdir((COMP / ref).parent)
    else:
        assert _exists_exact_case(ref), f"{ref} missing or differs in case from the file on disk"
