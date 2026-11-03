"""Physical constants and model parameters from Appendix B.2 (SI base units)."""

# Geocentric constants
MU_EARTH = 3.986004418e14       # m^3 / s^2, Earth standard gravitational parameter
R_EARTH = 6_378_137.0           # m, WGS84 equatorial radius (re-entry boundary)

# Thermospheric profile
RHO_0 = 3.82e-12                # kg / m^3, quiet-time reference density at H_0_REF
H_0_REF = 400_000.0             # m, reference datum altitude
SCALE_HEIGHT = 50_000.0         # m, hydrostatic scale height H
RHO_FLOOR = 1.0e-18             # kg / m^3, exospheric drag floor

# Space-weather modulation (linear heating factors)
F107_BASELINE = 100.0           # sfu
F107_SENSITIVITY = 0.008        # fractional density change per sfu
SOLAR_FACTOR_MIN = 0.2          # lower clamp on solar factor (Appendix A.3)
AP_BASELINE = 15.0              # dimensionless
AP_SENSITIVITY = 0.003          # fractional density change per Ap unit
GEOMAG_FACTOR_MIN = 0.5         # lower clamp on geomagnetic factor (Appendix A.3)

# Time
SECONDS_PER_DAY = 86_400.0      # s
