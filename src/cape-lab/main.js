// The cape bench: the Warden's cape on spring bones, on a stand-in body with her joints and a walk made in
// code, so the springs can be tuned without the Warden's model. The solver and the settings are the game's
// (src/world/springbones.js, src/world/cape-rig.js); only the body and the walk are stand-ins.
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { applySettings } from '../world/springbones.js';
import { CAPE_SETTINGS, WARDEN_MODEL_SETTINGS } from '../world/cape-rig.js';
import { WALKS, loadBakedWalk } from './baked-walk.js';
import { CapeSim, SCENARIOS, FRAME, PACE } from './sim.js';
import { loadModelBody } from './model-body.js';

const CAPTURE = new URLSearchParams(location.search).has('capture');

// ---- scene ----
const canvas = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: CAPTURE });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputEncoding = THREE.sRGBEncoding;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xcfd3c6);
scene.fog = new THREE.Fog(0xcfd3c6, 400, 900);
const camera = new THREE.PerspectiveCamera(30, 1, 1, 3000);
scene.add(new THREE.HemisphereLight(0xf2f0e6, 0x4a4a3a, 0.75));
const sun = new THREE.DirectionalLight(0xffffff, 0.9);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
Object.assign(sun.shadow.camera, { left: -60, right: 60, top: 60, bottom: -60, near: 1, far: 400 });
scene.add(sun, sun.target);

// a tiled ground, so her movement over it shows
function groundTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  g.fillStyle = '#b3b6a4'; g.fillRect(0, 0, 128, 128);
  g.fillStyle = '#a6aa96'; g.fillRect(0, 0, 64, 64); g.fillRect(64, 64, 64, 64);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(60, 60);
  t.anisotropy = 4;
  t.encoding = THREE.sRGBEncoding;
  return t;
}
const ground = new THREE.Mesh(new THREE.PlaneGeometry(3000, 3000), new THREE.MeshStandardMaterial({ map: groundTexture(), roughness: 1 }));
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = false;
controls.enabled = false;

// ---- the character ----
const settings = { ...CAPE_SETTINGS, colliderScale: 1, slow: 1 };
const opts = { walk: new URLSearchParams(location.search).get('walk') || 'clip', columns: 7, rows: 5, scenario: 'turn', view: 'back3q', showBody: true, showColliders: false, showBones: false, springs: true };
let sim, colliderGroup, boneHelper;
// the body: the stand-in, or the Warden's own model (?body=model, from ?model=<url>, models/warden.glb by default)
const params = new URLSearchParams(location.search);
const MODEL_URL = params.get('model') || 'models/warden.glb';
let modelBody = null;
// the walk the body plays: its own (her clip, or the stand-in's walk made in code) or a baked one
async function applyWalk(body = sim && sim.dummy) {
  if (!body) return;
  body.setWalk(opts.walk === 'clip' ? null : await loadBakedWalk(opts.walk, import.meta.env.BASE_URL));
}
async function setBody(kind) {
  if (kind === 'model') {
    flash('Loading the model…', 60000);
    modelBody = modelBody || await loadModelBody(import.meta.env.BASE_URL + MODEL_URL);
    flash('');
  }
  opts.bodyKind = kind;
  opts.body = kind === 'model' ? modelBody : null;
  await applyWalk(opts.body);
  // each body's own tuning (her cape hangs closer to her than the stand-in's)
  const own = kind === 'model' ? WARDEN_MODEL_SETTINGS : CAPE_SETTINGS;
  Object.assign(settings, { contactSoftness: own.contactSoftness, linkCurve: own.linkCurve });
  for (const f of syncSliders) f();
  build();
}
function build() {
  if (sim) scene.remove(sim.dummy.root);
  if (boneHelper) scene.remove(boneHelper);
  if (sim) sim.build(); else sim = new CapeSim(settings, opts);
  scene.add(sim.dummy.root);
  boneHelper = new THREE.SkeletonHelper(sim.dummy.root);
  boneHelper.visible = opts.showBones;
  scene.add(boneHelper);
  drawColliders();
  showBody();
}
function restart() { sim.restart(); drawColliders(); }
function makeSprings() { sim.makeSprings(); drawColliders(); }
// (not the meshes the model body re-skinned to springs: her own hair under the head is a plain mesh, and showing
// it again drew a rigid copy of her hair through the swinging one)
function showBody() {
  sim.dummy.root.traverse((o) => { if (o.isMesh && !o.isSkinnedMesh && !o.userData.replaced) o.visible = opts.showBody; });
}

// collider capsules, drawn see-through
const colMat = new THREE.MeshBasicMaterial({ color: 0xd2502a, transparent: true, opacity: 0.28, depthWrite: false });
function drawColliders() {
  if (colliderGroup) scene.remove(colliderGroup);
  colliderGroup = new THREE.Group();
  colliderGroup.visible = opts.showColliders;
  for (const c of sim.springs.colliders) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 10), colMat);
    const cyl = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 16, 1, true), colMat);
    const m2 = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 10), colMat);
    const g = new THREE.Group();
    g.add(m, cyl, m2);
    g.userData = { c, m, cyl, m2 };
    colliderGroup.add(g);
  }
  scene.add(colliderGroup);
}
const _up = new THREE.Vector3(0, 1, 0), _d = new THREE.Vector3();
function updateColliderMeshes() {
  if (!colliderGroup || !colliderGroup.visible) return;
  for (const g of colliderGroup.children) {
    const { c, m, cyl, m2 } = g.userData;
    c.refresh();
    const r = c.radius * c.scale;
    m.position.copy(c.wa); m.scale.setScalar(r);
    m2.position.copy(c.wb); m2.scale.setScalar(r);
    _d.subVectors(c.wb, c.wa);
    const len = _d.length();
    cyl.visible = len > 1e-4;
    cyl.position.copy(c.wa).addScaledVector(_d, 0.5);
    cyl.scale.set(r, Math.max(len, 1e-4), r);
    if (len > 1e-4) cyl.quaternion.setFromUnitVectors(_up, _d.normalize());
  }
}

// ---- camera ----
const VIEWS = { back: Math.PI, back3q: Math.PI * 0.75, side: Math.PI / 2, front3q: Math.PI / 4, front: 0, orbit: null };
const camTarget = new THREE.Vector3();
const lastOrbitRoot = new THREE.Vector3();
function placeCamera() {
  // (opts.zoom and opts.lookY frame a close-up, such as the head: the capture's --zoom)
  camTarget.set(sim.x, opts.lookY ?? 34, sim.z);
  const rel = VIEWS[opts.view];
  if (rel === null) {
    // orbit: the controls look at her, and the camera follows her movement
    const moved = new THREE.Vector3(sim.x, 0, sim.z).sub(lastOrbitRoot);
    camera.position.add(moved);
    controls.target.copy(camTarget);
    controls.update();
  } else {
    // the camera holds a fixed bearing from her heading, at the game's own slight tilt from above
    const a = sim.heading + rel, dist = 135 / (opts.zoom || 1);
    camera.position.set(sim.x + Math.sin(a) * dist, camTarget.y + dist * 0.22, sim.z + Math.cos(a) * dist);
    camera.lookAt(camTarget);
  }
  lastOrbitRoot.set(sim.x, 0, sim.z);
  sun.position.set(sim.x + 80, 160, sim.z + 60);
  sun.target.position.set(sim.x, 0, sim.z);
}
function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * renderer.getPixelRatio()) || canvas.height !== Math.round(h * renderer.getPixelRatio())) {
    renderer.setSize(w, h, false);
  }
  camera.aspect = w / Math.max(1, h);
  camera.updateProjectionMatrix();
}
function render() {
  resize();
  placeCamera();
  updateColliderMeshes();
  renderer.render(scene, camera);
}

// ---- the panel ----
const SLIDERS = [
  ['stiffness', 'Stiffness', 0, 900, 5, 'pull back to the hang, units/s²'],
  ['damping', 'Damping', 0, 0.6, 0.005, 'swing lost per 1/60 s (settles a sway)'],
  ['airDrag', 'Air drag', 0, 0.2, 0.002, 'speed lost per 1/60 s (makes it trail)'],
  ['gravity', 'Gravity', 0, 900, 5, 'units/s² (450 is real)'],
  ['linkStiffness', 'Sheet links', 0, 1, 0.01, 'how hard columns keep their spacing'],
  ['linkSlack', 'Link slack', 0, 0.5, 0.01, 'spacing allowed to stretch freely'],
  ['sheetBend', 'Sheet bend', 0, 1, 0.01, 'columns swing together (smooth hem)'],
  ['follow', 'Follow turns', 0, 1, 0.01, 'share of her own turning that carries the cloth'],
  ['maxAngle', 'Swing limit', 0.2, 2, 0.01, 'radians from the hang'],
  ['hitRadius', 'Cloth thickness', 0, 3, 0.05, 'gap kept off the body'],
  ['contactSoftness', 'Soft contact', 0, 4, 0.1, 'band over which the body eases the cloth away'],
  ['colliderScale', 'Collider size', 0.5, 1.5, 0.01, 'scales every body capsule'],
  ['slow', 'Playback speed', 0.1, 1, 0.05, 'slow motion'],
];
const syncSliders = [];
function buildPanel() {
  const host = document.getElementById('sliders');
  for (const [key, label, min, max, step, hint] of SLIDERS) {
    const row = document.createElement('label');
    row.className = 'slider';
    row.innerHTML = `<span class="name">${label}</span><output></output><input type="range" min="${min}" max="${max}" step="${step}"><span class="hint">${hint}</span>`;
    const input = row.querySelector('input'), out = row.querySelector('output');
    input.value = settings[key];
    const show = () => { out.textContent = Number(settings[key]).toFixed(step < 0.05 ? (step < 0.01 ? 3 : 2) : step < 1 ? 2 : 0); };
    show();
    syncSliders.push(() => { input.value = settings[key]; show(); });
    input.addEventListener('input', () => {
      settings[key] = Number(input.value);
      show();
      if (key === 'colliderScale') makeSprings();
      else applySettings(sim.springs, settings);
    });
    host.appendChild(row);
  }
  const sel = (id, options, key, after) => {
    const el = document.getElementById(id);
    for (const [v, l] of options) el.add(new Option(l, v));
    el.value = String(opts[key]);
    el.addEventListener('change', () => { opts[key] = isNaN(Number(el.value)) ? el.value : Number(el.value); after && after(); });
  };
  sel('scenario', Object.entries(SCENARIOS).map(([k, s]) => [k, s.label]), 'scenario', restart);
  sel('camview', [['back', 'Back'], ['back3q', 'Back 3/4'], ['side', 'Side'], ['front3q', 'Front 3/4'], ['front', 'Front'], ['orbit', 'Free (drag)']], 'view', () => {
    controls.enabled = opts.view === 'orbit';
  });
  sel('walksel', Object.entries(WALKS), 'walk', async () => { await applyWalk(); restart(); });
  sel('bodysel', [['standin', 'Stand-in'], ['model', 'Warden model']], 'bodyKind', () => setBody(opts.bodyKind)
    .catch((e) => flash('Model failed to load: ' + e.message)));
  sel('columns', [[5, '5 chains'], [7, '7 chains'], [9, '9 chains']], 'columns', build);
  sel('rows', [[3, '3 rows'], [4, '4 rows'], [5, '5 rows'], [6, '6 rows']], 'rows', build);
  const check = (id, key, after) => {
    const el = document.getElementById(id);
    el.checked = opts[key];
    el.addEventListener('change', () => { opts[key] = el.checked; after(); });
  };
  check('showBody', 'showBody', showBody);
  check('showColliders', 'showColliders', () => { colliderGroup.visible = opts.showColliders; });
  check('showBones', 'showBones', () => { boneHelper.visible = opts.showBones; });
  check('springsOn', 'springs', () => { if (opts.springs) sim.springs.reset(); });
  document.getElementById('restart').addEventListener('click', restart);
  document.getElementById('copy').addEventListener('click', async () => {
    const { colliderScale, slow, ...s } = settings;
    void slow;
    const text = JSON.stringify({ ...s, colliderScale }, null, 2);
    try { await navigator.clipboard.writeText(text); flash('Settings copied'); } catch { flash('Copy failed: ' + text); }
  });
}
let flashTimer = 0;
function flash(text, ms = 2500) {
  const el = document.getElementById('status');
  el.textContent = text;
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => { el.textContent = ''; }, ms);
}
function readout() {
  const el = document.getElementById('readout');
  const local = sim.scenarioTime();
  const drift = sim.drift === null ? 'measuring' : sim.drift.toFixed(3) + ' units';
  el.textContent = `t ${local.toFixed(2)} s · pace ${(sim.speed / PACE).toFixed(2)}` +
    (opts.scenario === 'walk' ? ` · loop drift ${drift}` : '');
}

// ---- run ----
opts.bodyKind = 'standin';
build();
if (!CAPTURE) {
  buildPanel();
  if (params.get('body') === 'model') {
    document.getElementById('bodysel').value = 'model';
    setBody('model').catch((e) => flash('Model failed to load: ' + e.message));
  }
  let last = performance.now(), acc = 0;
  const loop = (now) => {
    acc += Math.min(0.1, (now - last) / 1000) * settings.slow;
    last = now;
    while (acc >= FRAME) { acc -= FRAME; sim.step(); }
    render();
    readout();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
} else {
  // Capture hook (tools/cape-lab/capture.mjs): the script drives the sim and the frames
  document.body.classList.add('capture');
  window.__lab = {
    async setup(o = {}) {
      Object.assign(settings, o.settings || {});
      const kind = o.body || opts.bodyKind;
      const rebuild = (o.columns && o.columns !== opts.columns) || (o.rows && o.rows !== opts.rows);
      Object.assign(opts, o.opts || {}, o.columns ? { columns: o.columns } : {}, o.rows ? { rows: o.rows } : {});
      if (kind !== opts.bodyKind) { await setBody(kind); Object.assign(settings, o.settings || {}); restart(); }
      else if (rebuild) build(); else restart();
      // (the walk is set again whenever it is given: a new body needs it too)
      if (o.opts && o.opts.walk) { await applyWalk(); restart(); }
      colliderGroup.visible = opts.showColliders;
      boneHelper.visible = opts.showBones;
      showBody();
      return true;
    },
    view(v) { opts.view = v; },
    advance(frames) { for (let i = 0; i < frames; i++) sim.step(); },
    render() { render(); },
    sim: () => sim,
    state: () => ({ t: sim.t, frame: sim.frame, speed: sim.speed / PACE, heading: sim.heading, drift: sim.drift }),
  };
  render();
}
