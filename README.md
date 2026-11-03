# leo-decay

Python reproduction of the Hubble results in *Formulating an Explicit Differential
Equation for Low Earth Orbit Satellite Decay Using Empirical Ballistic Calibration*.
All computation is in SI units (m, kg, s), using the constants from Appendix B.2.

## Setup

```
python -m pip install -e .[test]
python -m pytest
```

## Dashboard

```
python -m pip install -e .[dashboard]
python -m streamlit run app/dashboard.py
```

Satellites need `mass_kg` and `b_star_empirical_m2_kg` in `config/satellites.toml`
to be simulated; ISS and NOAA-15 are listed but disabled until calibrated.

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
| `spacetrack.py` | Space-Track download, **stub** |

## Example: Protocol 3

```python
from leo_decay.propagator import propagate, weekly_checkpoints_s
traj = propagate(469_500.0, 35 * 86_400.0, 1.143720e-2, f107=145.0, ap=15.0,
                 t_eval_s=weekly_checkpoints_s(5))
print(traj.h_m / 1000)   # 469.5, 469.0302, 468.5559, 468.0771, 467.5937, 467.1056
```

## Space-Track credentials

Set `SPACETRACK_USER` and `SPACETRACK_PASS` as environment variables. Never commit them.
`spacetrack.fetch_tles` raises `NotImplementedError` until it is implemented.
