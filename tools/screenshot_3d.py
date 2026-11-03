"""Capture screenshots of the 3D view (and linked charts) with headless Chromium.

    python tools/screenshot_3d.py OUTPUT_DIR

Starts its own Streamlit server on port 8502, drives the three.js view through
its test hooks (window.__orbit3d), and writes PNGs plus a console-log report.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PORT = 8502
URL = f"http://localhost:{PORT}"


def wait_for_server(timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if urllib.request.urlopen(f"{URL}/_stcore/health", timeout=2).read() == b"ok":
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("Streamlit server did not start")


def scenario_times():
    """Storm peak and a mid-flight Earth-directed CME for the dashboard's default seed."""
    from leo_decay.solar import generate_scenario
    sc = generate_scenario(35, seed=2026)
    ev = sc.events
    storms = ev[ev["type"] == "storm"]
    best = storms.sort_values("peak_ap").iloc[-1]
    peak_day = best["start_day"] + sc.config.storms.main_phase_hours / 24
    cme = ev[ev["event_id"] == best["parent_id"]].iloc[0]
    mid_cme = cme["start_day"] + 0.55 * (best["start_day"] - cme["start_day"])
    return {"storm_id": best["event_id"], "storm_peak_day": float(peak_day),
            "storm_peak_ap": float(best["peak_ap"]), "cme_id": cme["event_id"],
            "cme_mid_day": float(mid_cme)}


def main(out_dir: str) -> int:
    from playwright.sync_api import sync_playwright

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    times = scenario_times()
    # Full mode (manual sliders by default), whether or not .env / config/tle.toml exist here.
    env = {**os.environ, "LEO_DEMO_MODE": "0"}
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "dashboard.py"),
         "--server.headless", "true", "--server.port", str(PORT), "--browser.gatherUsageStats", "false"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logs, report = [], {"times": times, "shots": []}
    try:
        wait_for_server()
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                               "--ignore-gpu-blocklist"])
            page = browser.new_page(viewport={"width": 1600, "height": 1250})
            page.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
            page.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
            page.on("response", lambda r: logs.append(f"[http {r.status}] {r.url}")
                    if r.status >= 400 else None)

            def frame3d(not_run_id=None):
                for _ in range(480):
                    for f in page.frames:
                        if "orbit3d" in f.url:
                            try:
                                st = f.evaluate("window.__orbit3d && window.__orbit3d.state()")
                            except Exception:
                                st = None
                            if (st and st["ready"] and st["textureLoaded"] and st["frames"] > 5
                                    and st["runId"] and st["runId"] != not_run_id):
                                return f
                    time.sleep(0.25)
                raise RuntimeError("3D component did not become ready")

            def wait_idle():
                page.wait_for_timeout(600)
                page.wait_for_function(
                    "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=120_000)
                page.wait_for_timeout(600)

            def shot(name, js, settle_ms=900, full_page=False):
                f = frame3d()
                f.evaluate(js)
                page.wait_for_timeout(settle_ms)
                state = f.evaluate("window.__orbit3d.state()")
                path = out / f"{name}.png"
                if full_page:
                    page.screenshot(path=path, full_page=True)
                else:
                    page.locator("iframe[title*='orbit3d']").first.screenshot(path=path)
                report["shots"].append({"name": name, "state": state})

            page.goto(f"{URL}/3d-view", wait_until="networkidle")
            wait_idle()
            shot("01_manual_day0_x10", "__orbit3d.setExaggeration(10); __orbit3d.view('default')")
            shot("02_manual_day17_x50", "__orbit3d.setDay(17.5); __orbit3d.setExaggeration(50); "
                                        "__orbit3d.view('default')")
            shot("03_manual_night_side", "__orbit3d.setExaggeration(10); __orbit3d.view('night')")
            shot("04_precession_off", "__orbit3d.setPrecession(false); __orbit3d.setDay(30); "
                                      "__orbit3d.view('pole')")

            manual_run = frame3d().evaluate("window.__orbit3d.state().runId")
            page.get_by_text("Generated events").click()
            wait_idle()
            frame3d(not_run_id=manual_run)                     # wait for the generated payload
            shot("05_generated_storm_aurora",
                 f"__orbit3d.setPrecession(true); __orbit3d.setDay({times['storm_peak_day']}); "
                 f"__orbit3d.setExaggeration(10); __orbit3d.view('pole')")
            shot("06_generated_cme_wide",
                 f"__orbit3d.setDay({times['cme_mid_day']}); __orbit3d.view('wide')")
            shot("07_generated_trail_warp",
                 f"__orbit3d.setDay(20); __orbit3d.setWarp(525960); __orbit3d.setExaggeration(25); "
                 f"__orbit3d.view('default')")

            # Pause -> timeline sync -> charts follow.
            f = frame3d()
            f.evaluate("__orbit3d.setDay(20); __orbit3d.pause()")
            wait_idle()
            page.screenshot(path=out / "08_3d_page_after_sync.png", full_page=True)
            report["timeline_after_sync"] = page.evaluate(
                "() => [...document.querySelectorAll('[role=slider]')].map("
                "s => [s.getAttribute('aria-label') || s.closest('[data-testid=stSlider]')?.innerText"
                ".split('\\n')[0], s.getAttribute('aria-valuenow')])")

            page.get_by_role("link", name="Charts").click()
            wait_idle()
            page.wait_for_function(
                "() => [...document.querySelectorAll('[data-testid=\"stMetricValue\"]')]"
                ".some(e => e.textContent.trim().length > 0)", timeout=60_000)
            page.wait_for_timeout(1500)
            page.screenshot(path=out / "09_charts_generated_cursor.png", full_page=True)
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=30)
    report["console"] = logs
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "console"}, indent=1))
    errors = [l for l in logs if l.startswith(("[error]", "[pageerror]"))]
    print(f"console messages: {len(logs)}, errors: {len(errors)}")
    for e in errors[:20]:
        print("  ", e[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "screenshots")))
