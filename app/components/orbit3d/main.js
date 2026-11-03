// Orbit 3D view: displays engine output only (see app/scene.py for how the payload is built).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as G from './geometry.js';

// ------------------------------------------------------------------ constants
const SUN_DIST = 60;          // Earth radii -- NOT to scale (true: ~23,500)
const SUN_RADIUS = 3;         // Earth radii -- NOT to scale (true: ~109)
const TRAIL_N = 900;
const CME_POOL = 16;
// Sequential ramps from the reference palette, ordered dim -> bright for the dark scene.
const RHO_RAMP = ['#104281', '#1c5cab', '#2a78d6', '#6da7ec', '#b7d3f6'];
const RATE_RAMP = ['#5a2410', '#9c3d17', '#d95926', '#ef8a5c', '#f7c3a8'];
const CME_EARTH = '#199e70';  // matches the "Earth-directed CME" chart marker (dark step)
const CME_OTHER = '#9a9a92';
const AURORA = '#3dff8a';

// ------------------------------------------------------------------ Streamlit protocol
function send(type, data = {}) {
  window.parent.postMessage({ isStreamlitMessage: true, type, ...data }, '*');
}
function setFrameHeight(h) { send('streamlit:setFrameHeight', { height: h }); }
function syncBack() {
  if (!S.payload) return;
  send('streamlit:setComponentValue', {
    value: { day: S.simT / 86400, nonce: Date.now() }, dataType: 'json',
  });
}

// ------------------------------------------------------------------ state
const S = {
  payload: null, runId: null, simT: 0, playing: false, warp: 86400, exag: 10, precess: true,
  lastArgsDay: null, height: 660, ready: false,
};

// ------------------------------------------------------------------ scene setup
const view = document.getElementById('view');
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
} catch (e) {
  const err = document.getElementById('error');
  err.hidden = false;
  err.textContent = 'WebGL is not available in this browser, so the 3D view cannot render.';
  throw e;
}
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
view.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color('#050507');
const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 3000);
camera.position.set(0.0, 2.2, 5.6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.minDistance = 1.15;
controls.maxDistance = 400;

// Stars: fixed random points on a far sphere (decorative backdrop).
{
  const n = 1500, pos = new Float32Array(n * 3);
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < n; i++) {
    const z = 2 * rnd() - 1, th = 2 * Math.PI * rnd(), r = Math.sqrt(1 - z * z);
    pos.set([900 * r * Math.cos(th), 900 * z, 900 * r * Math.sin(th)], i * 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0x8a8a96, size: 1.2, sizeAttenuation: false })));
}

// Earth (rotates by GMST about +y; longitude 0 on +x).
const earthGroup = new THREE.Group();
scene.add(earthGroup);
const dayMap = new THREE.TextureLoader().load('./assets/blue_marble.jpg', () => { S.textureLoaded = true; });
dayMap.colorSpace = THREE.NoColorSpace;
dayMap.anisotropy = renderer.capabilities.getMaxAnisotropy();
const earthMat = new THREE.ShaderMaterial({
  uniforms: { dayMap: { value: dayMap }, sunDir: { value: new THREE.Vector3(1, 0, 0) } },
  vertexShader: `
    varying vec2 vUv; varying vec3 vN;
    void main() {
      vUv = uv; vN = normalize(mat3(modelMatrix) * normal);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`,
  fragmentShader: `
    uniform sampler2D dayMap; uniform vec3 sunDir; varying vec2 vUv; varying vec3 vN;
    void main() {
      vec3 c = texture2D(dayMap, vUv).rgb;
      float d = dot(normalize(vN), normalize(sunDir));
      float day = smoothstep(-0.10, 0.10, d);
      vec3 lit = c * (0.30 + 0.85 * max(d, 0.0));
      vec3 night = c * 0.06 + vec3(0.0, 0.006, 0.02);
      vec3 col = mix(night, lit, day);
      float tw = clamp(1.0 - abs(d) / 0.12, 0.0, 1.0) * 0.22;
      col += vec3(0.35, 0.14, 0.04) * tw * day;
      gl_FragColor = vec4(col, 1.0);
    }`,
});
earthGroup.add(new THREE.Mesh(new THREE.SphereGeometry(1, 160, 80), earthMat));

// Thin atmosphere rim (decorative, true-scale ~100 km).
const rimMat = new THREE.ShaderMaterial({
  uniforms: { sunDir: earthMat.uniforms.sunDir },
  vertexShader: `varying vec3 vN; varying vec3 vV;
    void main(){ vN = normalize(mat3(modelMatrix)*normal);
      vec4 w = modelMatrix*vec4(position,1.0); vV = normalize(cameraPosition - w.xyz);
      gl_Position = projectionMatrix*viewMatrix*w; }`,
  fragmentShader: `uniform vec3 sunDir; varying vec3 vN; varying vec3 vV;
    void main(){ float f = pow(1.0 - abs(dot(vN, vV)), 3.0);
      float lit = smoothstep(-0.2, 0.3, dot(vN, normalize(sunDir)));
      gl_FragColor = vec4(vec3(0.35,0.6,1.0)*f*lit, f*lit*0.8); }`,
  transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.FrontSide,
});
scene.add(new THREE.Mesh(new THREE.SphereGeometry(1.016, 96, 48), rimMat));

// Aurora ovals (in the Earth group so they co-rotate with the geomagnetic pole).
const auroraMeshes = [];
function buildAurora(pole) {
  auroraMeshes.forEach((m) => earthGroup.remove(m));
  auroraMeshes.length = 0;
  const colat = (pole.colat_deg * Math.PI) / 180, R = 1.03;
  for (const sign of [1, -1]) {
    const ax = G.earthFixedToThreeLocal(sign * pole.lat_deg, sign > 0 ? pole.lon_deg : pole.lon_deg + 180);
    const axis = new THREE.Vector3(...ax);
    for (const [tube, op] of [[0.010, 1.0], [0.035, 0.35]]) {
      const m = new THREE.Mesh(
        new THREE.TorusGeometry(R * Math.sin(colat), tube, 10, 200),
        new THREE.MeshBasicMaterial({ color: AURORA, transparent: true, opacity: 0,
          blending: THREE.AdditiveBlending, depthWrite: false }));
      m.position.copy(axis.clone().multiplyScalar(R * Math.cos(colat)));
      m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), axis);
      m.userData.baseOpacity = op;
      m.visible = false;
      earthGroup.add(m);
      auroraMeshes.push(m);
    }
  }
}

// Thermosphere iso-density shell: transparent face-on, brighter at the limb (fresnel), so it
// reads as a shell and doesn't hide the Earth.
const shellMat = new THREE.ShaderMaterial({
  uniforms: { color: { value: new THREE.Color(RHO_RAMP[2]) } },
  vertexShader: `varying vec3 vN; varying vec3 vV;
    void main(){ vN = normalize(mat3(modelMatrix)*normal);
      vec4 w = modelMatrix*vec4(position,1.0); vV = normalize(cameraPosition - w.xyz);
      gl_Position = projectionMatrix*viewMatrix*w; }`,
  fragmentShader: `uniform vec3 color; varying vec3 vN; varying vec3 vV;
    void main(){ float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 3.0);
      gl_FragColor = vec4(color, 0.015 + 0.5 * f); }`,
  transparent: true, depthWrite: false, side: THREE.FrontSide,
});
const shell = new THREE.Mesh(new THREE.SphereGeometry(1, 128, 64), shellMat);
scene.add(shell);

// Sun: light + glowing sphere, NOT to scale.
const sunLight = new THREE.DirectionalLight(0xffffff, 2.0);
scene.add(sunLight, new THREE.AmbientLight(0x404050, 0.6));
const sunMesh = new THREE.Mesh(new THREE.SphereGeometry(SUN_RADIUS, 48, 24),
  new THREE.MeshBasicMaterial({ color: '#fff3c4' }));
scene.add(sunMesh);
const glowTex = (() => {
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d'), grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, 'rgba(255,240,200,1)'); grd.addColorStop(0.25, 'rgba(255,200,120,0.55)');
  grd.addColorStop(1, 'rgba(255,160,60,0)');
  g.fillStyle = grd; g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
})();
const sunGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, blending: THREE.AdditiveBlending,
  depthWrite: false, transparent: true }));
sunGlow.scale.set(28, 28, 1);
scene.add(sunGlow);

// Satellite + trail.
const satMesh = new THREE.Mesh(new THREE.SphereGeometry(0.016, 16, 12), new THREE.MeshBasicMaterial({ color: '#ffffff' }));
scene.add(satMesh);
const trailGeom = new THREE.BufferGeometry();
trailGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL_N * 3), 3));
trailGeom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(TRAIL_N * 3), 3));
const trailLine = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({ vertexColors: true }));
scene.add(trailLine);
// Current orbit (one revolution at the current radius and plane), faint, so the path is
// visible even before any trail has accumulated.
const RING_N = 256;
const ringGeom = new THREE.BufferGeometry();
ringGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(RING_N * 3), 3));
const ring = new THREE.LineLoop(ringGeom, new THREE.LineBasicMaterial({ color: '#c3c2b7', transparent: true, opacity: 0.35 }));
scene.add(ring);

// CME shells: spherical caps around +y, re-oriented per CME.
const cmePool = [];
for (let i = 0; i < CME_POOL; i++) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 12, 0, 2 * Math.PI, 0, (32 * Math.PI) / 180),
    new THREE.MeshBasicMaterial({ color: CME_OTHER, transparent: true, opacity: 0.25,
      side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending }));
  m.visible = false;
  scene.add(m);
  cmePool.push(m);
}

// HTML labels.
const labels = document.getElementById('labels');
function makeLabel(cls, text) {
  const el = document.createElement('div'); el.className = `label ${cls}`; el.textContent = text;
  labels.appendChild(el); return el;
}
const satLabel = makeLabel('sat', '');
const sunLabel = makeLabel('sun', 'Sun · direction true; size & distance not to scale');

// ------------------------------------------------------------------ helpers
const tmpColor = new THREE.Color();
function srgb(hexOrRgb) {
  const [r, g, b] = Array.isArray(hexOrRgb) ? hexOrRgb : G.hexToRgb(hexOrRgb);
  return tmpColor.setRGB(r, g, b, THREE.SRGBColorSpace);
}
const fmt = (x, d) => Number(x).toFixed(d);
const fmtExp = (x) => Number(x).toExponential(2);

function sunDirThree(t) {
  const p = S.payload, v = [0, 1, 2].map((k) => G.interp(p.sun_t_s, p.sun_eci.map((s) => s[k]), t));
  const n = Math.hypot(...v);
  return G.eciToThree([v[0] / n, v[1] / n, v[2] / n]);
}

function satState(t) {
  const p = S.payload, el = p.elements;
  const h = G.interp(p.t_s, p.h_m, t);
  const M = G.interp(p.t_s, p.mean_anomaly, t);
  const raan = S.precess ? G.interp(p.t_s, p.raan_prec, t) : el.raan0;
  const argp = S.precess ? G.interp(p.t_s, p.argp_prec, t) : el.argp0;
  const r = G.displayRadius(h, S.exag, p.R_E_m);
  return { h, pos: G.eciToThree(G.orbitPositionEci(r, raan, el.inclination, argp + M)) };
}

function rateColorFrac(rate) {
  const [lo, hi] = S.payload.rate_range;
  return hi > lo ? (Math.abs(rate) - lo) / (hi - lo) : 0.5;
}
function rhoColorFrac(rho) {
  const [lo, hi] = S.payload.density_range;
  return hi > lo ? (Math.log(rho) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) : 0.5;
}

function project(v3) {
  const v = v3.clone().project(camera);
  const w = renderer.domElement.clientWidth, hgt = renderer.domElement.clientHeight;
  return { x: (v.x * 0.5 + 0.5) * w, y: (-v.y * 0.5 + 0.5) * hgt, visible: v.z < 1 && v.z > -1 };
}
function placeLabel(el, pos) {
  const p = project(pos);
  el.style.display = p.visible ? 'block' : 'none';
  el.style.left = `${p.x}px`; el.style.top = `${p.y}px`;
}

// ------------------------------------------------------------------ payload
function gradient(stops) { return `linear-gradient(to right, ${stops.join(', ')})`; }
document.getElementById('bar-rate').style.background = gradient(RATE_RAMP);
document.getElementById('bar-rho').style.background = gradient(RHO_RAMP);

function loadPayload(p) {
  S.payload = p; S.runId = p.run_id;
  S.simT = Math.min(S.simT, p.duration_s);
  S.meanMotion = p.mean_anomaly.map((m, i, a) =>
    (i ? (m - a[i - 1]) / (p.t_s[i] - p.t_s[i - 1]) : (a[1] - a[0]) / (p.t_s[1] - p.t_s[0])));
  buildAurora(p.aurora_pole);
  satLabel.textContent = `${p.sat_name} (NORAD ${p.norad_id})`;
  const title = document.getElementById('hud-title');
  title.textContent = `${p.sat_name} · NORAD ${p.norad_id}`;
  if (p.demo) {
    const tag = document.createElement('span');
    tag.className = 'demo-tag'; tag.id = 'demo-tag'; tag.textContent = 'Demo mode';
    tag.title = 'Placeholder orbit elements and generated solar events; no Space-Track data.';
    title.append(' ', tag);
  }
  document.getElementById('rate-lo').textContent = fmt(p.rate_range[0], 4);
  document.getElementById('rate-hi').textContent = fmt(p.rate_range[1], 4);
  document.getElementById('rho-lo').textContent = fmtExp(p.density_range[0]);
  document.getElementById('rho-hi').textContent = fmtExp(p.density_range[1]);
  document.getElementById('shell-note').textContent =
    `Shell radius: altitude where ρ = ${fmtExp(p.shell_rho_ref)} kg/m³ (engine density model), × exaggeration.`;
  updateFootnote();
}

function updateFootnote() {
  const p = S.payload; if (!p) return;
  const src = p.elements.placeholder
    ? `<span class="warn">Orbit plane from ${p.elements.source} — run tools/fetch_tles.py for real elements.</span>`
    : `Orbit plane from ${p.elements.source}.`;
  const prec = S.precess ? `<br><span class="warn">${p.precession_label}</span>`
                         : '<br>Orbit plane fixed at the TLE epoch.';
  document.getElementById('footnote').innerHTML =
    `Earth to true scale; satellite altitude and shell × ${S.exag}. Radius, decay rate and density are engine output. ` +
    `CME shells: aqua = Earth-directed (arrives at its storm onset), grey = other; Sun distance compressed. ` +
    `Aurora drawn when Ap &gt; 30. ${src}${prec}`;
}

// ------------------------------------------------------------------ per-frame update
function update() {
  const p = S.payload; if (!p) return;
  const t = S.simT;

  earthGroup.rotation.y = G.gmstAt(p.gmst0, p.omega_earth, t);
  const sd = sunDirThree(t);
  const sunPos = new THREE.Vector3(...sd).multiplyScalar(SUN_DIST);
  earthMat.uniforms.sunDir.value.set(...sd);
  sunMesh.position.copy(sunPos); sunGlow.position.copy(sunPos);
  sunLight.position.copy(sunPos);

  // Satellite & trail (colour = |decay rate| from the engine at each trail time).
  const st = satState(t);
  satMesh.position.set(...st.pos);
  const n = G.interp(p.t_s, S.meanMotion, t) || 0.0011;      // rad/s, from engine a(t)
  const period = (2 * Math.PI) / Math.max(n, 1e-6);
  // Trail window: grows with time warp but is capped at 22 orbits, so 900 points always
  // sample each revolution densely (>= 40 points per orbit; no aliasing).
  const W = Math.min(Math.max(2 * period, S.warp * 15), 22 * period, Math.max(t, 1));
  const pos = trailGeom.attributes.position.array, col = trailGeom.attributes.color.array;
  for (let i = 0; i < TRAIL_N; i++) {
    const ti = Math.max(0, t - (W * i) / (TRAIL_N - 1));
    const s = satState(ti);
    pos.set(s.pos, i * 3);
    const rate = G.interp(p.t_s, p.decay_rate_km_day, ti);
    const c = srgb(G.rampColor(RATE_RAMP, rateColorFrac(rate)));
    col.set([c.r, c.g, c.b], i * 3);
  }
  trailGeom.attributes.position.needsUpdate = true;
  trailGeom.attributes.color.needsUpdate = true;
  trailGeom.setDrawRange(0, t > 0 ? TRAIL_N : 0);

  // Current-orbit ring: same plane and radius as the satellite right now.
  {
    const el = p.elements;
    const raan = S.precess ? G.interp(p.t_s, p.raan_prec, t) : el.raan0;
    const r = G.displayRadius(st.h, S.exag, p.R_E_m);
    const rp = ringGeom.attributes.position.array;
    for (let i = 0; i < RING_N; i++) {
      rp.set(G.eciToThree(G.orbitPositionEci(r, raan, el.inclination, (2 * Math.PI * i) / RING_N)), i * 3);
    }
    ringGeom.attributes.position.needsUpdate = true;
  }

  // Thermosphere shell.
  const hs = G.interp(p.forcing_t_s, p.shell_h_m, t);
  shell.scale.setScalar(G.displayRadius(hs, S.exag, p.R_E_m));
  const rho = G.interp(p.t_s, p.density, t);
  shellMat.uniforms.color.value.copy(srgb(G.rampColor(RHO_RAMP, rhoColorFrac(rho))));

  // Aurora.
  const ai = G.interp(p.forcing_t_s, p.aurora, t);
  const ap = G.interp(p.forcing_t_s, p.ap, t);
  for (const m of auroraMeshes) {
    m.visible = ap > 30 && ai > 0;
    m.material.opacity = m.userData.baseOpacity * (0.25 + 0.75 * ai);
  }

  // CMEs.
  let k = 0;
  const active = p.cmes
    .map((c) => ({ c, prog: (t / 86400 - c.launch_day) / Math.max(c.arrival_day - c.launch_day, 1e-6) }))
    .filter((x) => x.prog > 0 && x.prog < 1.25)
    .sort((a, b) => Number(b.c.earth_directed) - Number(a.c.earth_directed));
  for (const { c, prog } of active) {
    if (k >= CME_POOL) break;
    const m = cmePool[k++];
    const dir = new THREE.Vector3(...G.cmeDirection(sd, c.offset_deg, c.azimuth_deg));
    m.position.copy(sunPos);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    m.scale.setScalar(Math.max(prog, 0.01) * SUN_DIST);
    m.material.color.copy(srgb(c.earth_directed ? CME_EARTH : CME_OTHER));
    m.material.opacity = (c.earth_directed ? 0.38 : 0.2) * (prog < 1 ? 1 : (1.25 - prog) / 0.25);
    m.visible = true;
  }
  for (; k < CME_POOL; k++) cmePool[k].visible = false;
  S.activeCmes = active.length;

  // HUD.
  const date = new Date(Date.parse(p.start_utc) + t * 1000);
  document.getElementById('hud-time').textContent =
    `${date.toISOString().slice(0, 16).replace('T', ' ')} UTC · day ${fmt(t / 86400, 2)}${S.playing ? ' ▶' : ''}`;
  const rows = [
    ['Altitude (engine)', `${fmt(st.h / 1000, 3)} km`],
    ['Decay rate', `${fmt(G.interp(p.t_s, p.decay_rate_km_day, t), 4)} km/day`],
    ['Density at satellite', `${fmtExp(rho)} kg/m³`],
    ['F10.7 input', `${fmt(G.interp(p.forcing_t_s, p.f107, t), 1)} sfu`],
    ['Ap', `${fmt(ap, 0)}${ap > 30 ? ' · aurora' : ''}`],
    ['CMEs in flight', `${S.activeCmes}`],
  ];
  document.getElementById('hud-values').innerHTML =
    rows.map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join('');

  placeLabel(satLabel, satMesh.position);
  placeLabel(sunLabel, sunPos.clone().add(new THREE.Vector3(0, SUN_RADIUS * 1.6, 0)));
}

// ------------------------------------------------------------------ controls
const playBtn = document.getElementById('play');
function setPlaying(on) {
  S.playing = on;
  playBtn.textContent = on ? '⏸ Pause' : '▶ Play';
  playBtn.setAttribute('aria-label', on ? 'Pause' : 'Play');
  if (!on) syncBack();
}
playBtn.addEventListener('click', () => {
  if (!S.playing && S.payload && S.simT >= S.payload.duration_s) S.simT = 0;
  setPlaying(!S.playing);
});
document.getElementById('warp').addEventListener('change', (e) => { S.warp = Number(e.target.value); });
const exagIn = document.getElementById('exag'), exagOut = document.getElementById('exag-out');
exagIn.addEventListener('input', () => { S.exag = Number(exagIn.value); exagOut.textContent = S.exag; updateFootnote(); });
document.getElementById('precess').addEventListener('change', (e) => { S.precess = e.target.checked; updateFootnote(); });

function resize() {
  const w = view.clientWidth || window.innerWidth, h = view.clientHeight || window.innerHeight;
  renderer.setSize(w, h, false);
  renderer.domElement.style.width = `${w}px`; renderer.domElement.style.height = `${h}px`;
  camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

// ------------------------------------------------------------------ main loop
let last = performance.now();
function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.1); last = now;
  if (S.playing && S.payload) {
    S.simT += dt * S.warp;
    if (S.simT >= S.payload.duration_s) { S.simT = S.payload.duration_s; setPlaying(false); }
  }
  controls.update();
  update();
  renderer.render(scene, camera);
  S.frames = (S.frames || 0) + 1;
  requestAnimationFrame(frame);
}

window.addEventListener('message', (ev) => {
  const d = ev.data;
  if (!d || d.type !== 'streamlit:render') return;
  const args = d.args || {};
  if (args.height && args.height !== S.height) { S.height = args.height; }
  document.body.style.height = `${S.height}px`;
  setFrameHeight(S.height);
  if (args.payload && args.payload.run_id !== S.runId) loadPayload(args.payload);
  if (typeof args.day === 'number' && args.day !== S.lastArgsDay) {
    S.lastArgsDay = args.day;
    S.simT = Math.min(Math.max(args.day * 86400, 0), S.payload ? S.payload.duration_s : 0);
  }
  resize();
  S.ready = true;
});

// Hooks for automated screenshots (tools/screenshot_3d.py).
window.__orbit3d = {
  setDay(d) { S.simT = d * 86400; },
  setExaggeration(k) { exagIn.value = k; exagIn.dispatchEvent(new Event('input')); },
  setWarp(w) { S.warp = w; },
  setPrecession(on) { const c = document.getElementById('precess'); c.checked = on; c.dispatchEvent(new Event('change')); },
  play() { setPlaying(true); }, pause() { setPlaying(false); },
  view(name) {
    const sd = S.payload ? new THREE.Vector3(...sunDirThree(S.simT)) : new THREE.Vector3(1, 0, 0);
    controls.target.set(0, 0, 0);
    if (name === 'night') camera.position.copy(sd.clone().multiplyScalar(-4.5).add(new THREE.Vector3(0, 1.2, 0)));
    else if (name === 'day') camera.position.copy(sd.clone().multiplyScalar(4.5).add(new THREE.Vector3(0, 1.2, 0)));
    else if (name === 'pole') camera.position.set(0.01, 4.8, 0.01);
    else if (name === 'wide') {
      const side = new THREE.Vector3().crossVectors(sd, new THREE.Vector3(0, 1, 0)).normalize();
      controls.target.copy(sd.clone().multiplyScalar(SUN_DIST * 0.5));
      camera.position.copy(controls.target.clone().add(side.multiplyScalar(95)).add(new THREE.Vector3(0, 30, 0)));
    } else camera.position.set(0.0, 2.2, 5.6);
    controls.update();
  },
  state() {
    return { ready: S.ready, frames: S.frames || 0, simDay: S.simT / 86400, exag: S.exag,
             demo: !!(S.payload && S.payload.demo),
             elementsPlaceholder: !!(S.payload && S.payload.elements.placeholder),
             textureLoaded: !!S.textureLoaded, activeCmes: S.activeCmes || 0,
             auroraVisible: auroraMeshes.some((m) => m.visible), runId: S.runId };
  },
};

send('streamlit:componentReady', { apiVersion: 1 });
setFrameHeight(S.height);
resize();
requestAnimationFrame(frame);
