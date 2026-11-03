// Pure geometry helpers (no three.js), mirrored by app/scene.py and tested from pytest via node.
// Scene units: 1 = Earth equatorial radius. three.js is y-up, so ECI (x, y, z) -> (x, z, -y).

export function eciToThree(v) {
  return [v[0], v[2], -v[1]];
}

// Linear interpolation of samples (t ascending), clamped at the ends.
export function interp(ts, vs, t) {
  const n = ts.length;
  if (t <= ts[0]) return vs[0];
  if (t >= ts[n - 1]) return vs[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (ts[mid] <= t) lo = mid; else hi = mid;
  }
  const f = (t - ts[lo]) / (ts[hi] - ts[lo]);
  return vs[lo] + f * (vs[hi] - vs[lo]);
}

// Exaggerated display radius in Earth radii: 1 + k * h / R_E (h from the engine).
export function displayRadius(h_m, k, R_E) {
  return 1 + (k * h_m) / R_E;
}

// Circular-orbit ECI position for argument of latitude u (same as scene.orbit_position_eci).
export function orbitPositionEci(r, raan, inc, u) {
  const cO = Math.cos(raan), sO = Math.sin(raan), ci = Math.cos(inc), si = Math.sin(inc);
  const cu = Math.cos(u), su = Math.sin(u);
  return [r * (cO * cu - sO * su * ci), r * (sO * cu + cO * su * ci), r * su * si];
}

export function gmstAt(gmst0, omegaEarth, t_s) {
  return gmst0 + omegaEarth * t_s;
}

// Earth-fixed (geodetic lat/lon, degrees) unit vector in the Earth group's local three.js frame.
// The Earth mesh has longitude 0 on +x and the north pole on +y; the group rotates by GMST about y.
export function earthFixedToThreeLocal(latDeg, lonDeg) {
  const la = (latDeg * Math.PI) / 180, lo = (lonDeg * Math.PI) / 180;
  return [Math.cos(la) * Math.cos(lo), Math.sin(la), -Math.cos(la) * Math.sin(lo)];
}

// Rotate an Earth-group-local vector by the group's y rotation (GMST) into the scene frame.
export function rotateY(v, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  return [v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c];
}

function normalize(v) {
  const n = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / n, v[1] / n, v[2] / n];
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

// Unit direction of a CME from the Sun: toward Earth, tilted by offsetDeg around azimuthDeg.
export function cmeDirection(sunDir, offsetDeg, azimuthDeg) {
  const toEarth = normalize([-sunDir[0], -sunDir[1], -sunDir[2]]);
  const ref = Math.abs(toEarth[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const p = normalize(cross(toEarth, ref));
  const q = cross(toEarth, p);
  const off = (offsetDeg * Math.PI) / 180, az = (azimuthDeg * Math.PI) / 180;
  const side = [p[0] * Math.cos(az) + q[0] * Math.sin(az), p[1] * Math.cos(az) + q[1] * Math.sin(az),
                p[2] * Math.cos(az) + q[2] * Math.sin(az)];
  return normalize([toEarth[0] * Math.cos(off) + side[0] * Math.sin(off),
                    toEarth[1] * Math.cos(off) + side[1] * Math.sin(off),
                    toEarth[2] * Math.cos(off) + side[2] * Math.sin(off)]);
}

// Interpolate between two hex colours (#rrggbb) in linear RGB; f in [0, 1].
export function rampColor(stops, f) {
  const x = Math.min(1, Math.max(0, f)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const g = x - i;
  const a = hexToRgb(stops[i]), b = hexToRgb(stops[i + 1]);
  return [a[0] + g * (b[0] - a[0]), a[1] + g * (b[1] - a[1]), a[2] + g * (b[2] - a[2])];
}

export function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}
