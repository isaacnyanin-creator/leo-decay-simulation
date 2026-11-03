# leo-decay

Python reproduction of the Hubble results in *Formulating an Explicit Differential
Equation for Low Earth Orbit Satellite Decay Using Empirical Ballistic Calibration*.
All computation is in SI units (m, kg, s), using the constants from Appendix B.2.

## Setup

```
python -m pip install -e .[test]
python -m pytest                    # browser tests are opt-in: python -m pytest -m browser
```

## Dashboard

```
python -m pip install -e .[dashboard]
python -m streamlit run app/dashboard.py
```

Two pages share the sidebar controls and a **Timeline (day)** slider:

* **Charts**: altitude, decay rate, Eq. B.2 acceleration, density and drag force; in
  "Generated events" solar mode also F10.7, Ap, event markers and the event table.
* **3D view**: three.js scene (Blue Marble Earth with day/night terminator, Sun direction,
  orbit from the TLE, trail coloured by decay rate, thermosphere iso-density shell, CME
  shells, aurora when Ap > 30). It displays engine output only; J2 precession of the orbit
  plane is a labelled, display-only option. Pausing playback moves the shared timeline.

Satellites need `mass_kg` and `b_star_empirical_m2_kg` in `config/satellites.toml`
to be simulated; ISS and NOAA-15 are listed but disabled until calibrated.

Screenshots of the 3D view (headless Chromium):

```
python -m pip install -e .[browser] && python -m playwright install chromium
python tools/screenshot_3d.py OUTPUT_DIR
```

## Module map

| Module | Paper section |
|---|---|
| `constants.py` | Appendix B.2 constants |
| `config.py` + `config/satellites.toml` | §3.1 satellites (Hubble 20580, ISS 25544, NOAA-15 25338) |
| `tle_conversion.py` | §3.2 / A.1: mean motion → a, h, v, da/dt |
| `glitch_cleaning.py` | A.1: \|Δh\| > 2 km → PCHIP |
| `reboost_masking.py` | A.1: > +0.50 km/day → mask ±3 days |
| `density.py` | A.3 density model ρ(a, F10.7, Ap) |
| `calibration.py` | Eq. B.4 bounded TRF B* fit |
| `propagator.py` | Eq. B.3 RK45 (rtol 1e-9, atol 1e-12) with re-entry guard |
| `solar/` | stochastic F10.7 / flare / CME / storm generator (placeholder rates) |
| `spacetrack.py` | policy-aware Space-Track client with on-disk cache |
| `validation.py` | forecast vs observed-TLE residuals |

## Example: Protocol 3

```python
from leo_decay.propagator import propagate, weekly_checkpoints_s
traj = propagate(469_500.0, 35 * 86_400.0, 1.143720e-2, f107=145.0, ap=15.0,
                 t_eval_s=weekly_checkpoints_s(5))
print(traj.h_m / 1000)   # 469.5, 469.0302, 468.5559, 468.0771, 467.5937, 467.1056
```

## Space-Track data

Put `SPACETRACK_USER` and `SPACETRACK_PASSWORD` in `.env` (gitignored), then:

```
python -m pip install -e .[spacetrack]
python tools/fetch_tles.py                                   # latest GP -> config/tle.toml
python tools/fetch_tles.py --history 20580 2026-01-01 2026-11-04 --skip-gp
python tools/validate_protocol3.py                           # residuals from the cache
```

The client follows Space-Track's API Use Guidelines: the `spacetrack` package throttles
to < 30 requests/min and < 300/hour, and a ledger in `data/raw/spacetrack/` enforces GP
at most once per hour and GP_HISTORY once per object ("do not download it again").
**No Space-Track data is committed**: `data/` and `config/tle.toml` are gitignored;
`config/tle.example.toml` documents the format (and is the 3D view's fallback, labelled
as placeholder elements).

## Credits

* Earth texture: NASA Earth Observatory, *Blue Marble: Next Generation* with topography
  and bathymetry (January 2004), downsampled to 4096×2048. NASA imagery is generally not
  subject to copyright in the United States.
* three.js (MIT licence), vendored in `app/components/orbit3d/vendor/`.
