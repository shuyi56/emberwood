// The fabric bench: fixed sheets on fixed bodies (scenes.mjs), solved by the old solver (fabric-core-v1.js, the
// cape's first one) and the new (src/world/fabric-core.js) at draft and final quality, and measured.
//
//   node tools/fabric-bench/bench.mjs [--scenes hang,drape,boot,tasset] [--configs v1,draft,final]
//        [--material linen] [--out DIR (frames for the clay renders)] [--json]
//   node tools/fabric-bench/bench.mjs --cantilever [--material leather] [--hinge 4e-4]
//
// For each scene and solver:
//  - ms/frame: the solver's time for a 60 fps frame (node, one core; the body's own update not counted)
//  - stretch: how much longer than its rest the longest thread is (max) and the threads on average (mean), %
//  - jitter: the mean size of each point's third difference over frames (units): a smooth swing scores low, a
//    flicker high (as tools/cape-lab/check.mjs measures the cape)
//  - in body: the deepest any point goes into the body (units; 39.4 a metre)
//  - settle: hang and drape: the first frame from which no point moves more than 0.02 units a frame
//  - set: boot and tasset: the points that have taken a set (rest curvature moved by more than a 15 cm bend's),
//    and the tightest set, as a radius
//  - crease: fold: the angle the authored crease holds, against the angle asked for
// --cantilever: each material's bending length from the cantilever test (Peirce 1930), against its target
import fs from 'fs';
import path from 'path';
import { performance } from 'perf_hooks';
import { Fabric as FabricV1 } from './fabric-core-v1.js';
import { Fabric, material, MATERIALS } from '../../src/world/fabric-core.js';
import { SCENES, U } from './scenes.mjs';

const arg = (name, def) => { const i = process.argv.indexOf('--' + name); return i > 0 ? process.argv[i + 1] : def; };
const flag = (name) => process.argv.includes('--' + name);

// The solvers. v1: the old core as the cape uses it (bend ties two along, the body after the constraints, no
// weight, no air). draft and final: the new core with the material's hinges, weight, air, tethers, the body
// tested inside the solve; draft at 120 steps a second and 2 walks, final at 240 and 4.
export const CONFIGS = {
  v1: { core: 'v1', substep: 1 / 240, iterations: 2 },
  draft: { core: 'v2', substep: 1 / 240, iterations: 2, passes: 1, contact: 'iteration', edgeContact: true },
  final: { core: 'v2', substep: 1 / 240, iterations: 4, passes: 2, contact: 'iteration', edgeContact: true },
};
const PAD = 0.12;      // the cloth's half thickness over the body

export function build(scene, cfg, matName, over = {}) {
  const g = [0, -9.81 * U, 0];
  const pos = Float64Array.from(scene.pos);
  const at = (c, r) => r * scene.cols + c;
  let f;
  if (cfg.core === 'v1') {
    f = new FabricV1(pos, { gravity: g, iterations: cfg.iterations, friction: 0.5, damping: 0.02 });
    for (const i of scene.pins) f.pin(i);
    f.sheet(scene.cols, scene.rows, at, { wrap: scene.wrap });
  } else {
    const m = { ...material(matName, U), ...over };
    f = new Fabric(pos, { ...m, iterations: cfg.iterations, substep: cfg.substep, contact: cfg.contact, edgeContact: cfg.edgeContact, wind: scene.wind, thickness: scene.thickness ?? 0 });
    for (const i of scene.pins) f.pin(i);
    // (no bend ties two along: the implicit bend (stiffen) bends it, to the material's bending length; hinges only
    // where a crease is authored; shear as the material has it)
    const hinged = !!scene.creases || over.hinges;
    f.sheet(scene.cols, scene.rows, at, { wrap: scene.wrap, bend: null, hinge: hinged ? m.hinge : null, shear: m.shear, flat: !scene.wrap });
    f.weigh(m.density);
    if (!over.noStiff) f.stiffen({ bendingLength: m.bendingLength, density: m.density, setRadius: m.setRadius, creep: m.creep, scale: scene.stiffScale, passes: over.passes ?? cfg.passes, stretchK: over.stretchK ?? cfg.stretchK });
    if (scene.pins.length && !over.noTether) f.tether();
    if (scene.creases) for (const c of scene.creases) f.fold(c.line, c.angle, c);
  }
  return f;
}

// the warp and weft threads of a scene's grid, for the stretch
function threads(scene) {
  const out = [], at = (c, r) => r * scene.cols + c;
  for (let r = 0; r < scene.rows; r++) for (let c = 0; c < scene.cols; c++) {
    if (r + 1 < scene.rows) out.push([at(c, r), at(c, r + 1)]);
    if (c + 1 < scene.cols || scene.wrap) out.push([at(c, r), at((c + 1) % scene.cols, r)]);
  }
  return out;
}
const dist = (x, i, j) => Math.hypot(x[3 * i] - x[3 * j], x[3 * i + 1] - x[3 * j + 1], x[3 * i + 2] - x[3 * j + 2]);

// Run one scene with one solver: returns the numbers, and the frames asked for (keep: frame indices)
export function run(sceneName, cfgName, matName, o = {}) {
  const scene = SCENES[sceneName](), cfg = CONFIGS[cfgName], fps = scene.fps, dt = 1 / fps;
  const f = build(scene, cfg, matName, o.over || {});
  const th = threads(scene), rest = th.map(([i, j]) => dist(scene.pos, i, j));
  const free = [];
  for (let i = 0; i < f.n; i++) if (f.w[i] !== 0) free.push(i);
  let parts = scene.body(0);
  const body = (p, i, ref) => {
    let hit = false;
    for (const b of parts) hit = b.push(p, PAD, ref) > 0 || hit;
    return hit;
  };
  const setPins = (t) => {
    if (!scene.pinAt) return;
    const P = scene.pinAt(t);
    scene.pins.forEach((i, k) => { for (let c = 0; c < 3; c++) { f.pos[3 * i + c] = P[3 * k + c]; f.prev[3 * i + c] = P[3 * k + c]; } });
  };
  // (start from the body's first pose, settled a moment, so no scene starts with a jump)
  let ms = 0, maxStretch = 0, meanStretch = 0, nStretch = 0, inBody = 0, jit = 0, nJit = 0, settle = null;
  const hist = [], kept = [];
  const h = cfg.substep;
  let acc = 0;
  for (let fr = 0; fr < scene.frames; fr++) {
    const t0 = fr * dt;
    const s0 = performance.now();
    acc += dt;
    const steps = Math.floor(acc / h + 1e-9);
    for (let k = 0; k < steps; k++) {
      const t = t0 + (k + 1) * h - (acc - dt);
      if (scene.pinAt || scene.moving !== false) parts = scene.body(t);
      setPins(t);
      f.step(h, parts.length ? body : null, { neutral: true });
    }
    acc -= steps * h;
    ms += performance.now() - s0;
    // the numbers, on the frame as it is shown (after the run's first stride: the cloth starts at rest on a leg
    // already running, and its first frames measure that start, not the cloth)
    const x = f.pos;
    if (fr < (scene.warm || 0)) {
      if (o.keep && o.keep.includes(fr)) kept.push({ frame: fr, t: (fr + 1) * dt, pos: Array.from(x, (v) => +v.toFixed(4)), body: parts.map(partJSON) });
      hist.length = 0;
      continue;
    }
    let mx = 0, sm = 0;
    th.forEach(([i, j], k) => { const e = dist(x, i, j) / rest[k] - 1; mx = Math.max(mx, e); sm += Math.abs(e); });
    maxStretch = Math.max(maxStretch, mx); meanStretch += sm / th.length; nStretch++;
    for (const i of free) for (const b of parts) inBody = Math.max(inBody, b.depth([x[3 * i], x[3 * i + 1], x[3 * i + 2]]));
    hist.push(Float64Array.from(x));
    if (hist.length > 4) hist.shift();
    if (hist.length === 4 && fr > fps / 2) {
      const [a, b, c, d] = hist;
      let s = 0;
      for (const i of free) {
        const q = 3 * i;
        s += Math.hypot(d[q] - 3 * c[q] + 3 * b[q] - a[q], d[q + 1] - 3 * c[q + 1] + 3 * b[q + 1] - a[q + 1], d[q + 2] - 3 * c[q + 2] + 3 * b[q + 2] - a[q + 2]);
      }
      jit += s / free.length; nJit++;
    }
    if (hist.length >= 2) {
      const [c, d] = hist.slice(-2);
      let far = 0;
      for (const i of free) far = Math.max(far, Math.hypot(d[3 * i] - c[3 * i], d[3 * i + 1] - c[3 * i + 1], d[3 * i + 2] - c[3 * i + 2]));
      if (far < 0.02) { if (settle === null) settle = fr; } else settle = null;
    }
    if (o.keep && o.keep.includes(fr)) kept.push({ frame: fr, t: (fr + 1) * dt, pos: Array.from(x, (v) => +v.toFixed(4)), body: parts.map(partJSON), set: setMap(f) });
  }
  const res = {
    scene: sceneName, solver: cfgName, material: cfg.core === 'v1' ? '(v1 settings)' : matName,
    msPerFrame: +(ms / scene.frames).toFixed(3), maxStretchPct: +(100 * maxStretch).toFixed(2), meanStretchPct: +(100 * meanStretch / nStretch).toFixed(3),
    jitter: +(jit / Math.max(1, nJit)).toFixed(4), inBody: +inBody.toFixed(3),
  };
  if (!scene.pinAt) res.settleFrame = settle;
  if (scene.creases && cfg.core !== 'v1') {
    // the authored crease: the angle its hinges are at, against the angle asked for
    const { angle } = f.hingeAngles(), want = scene.creases[0].angle, line = scene.creases[0].line;
    let sum = 0, cnt = 0;
    for (let q = 1; q < line.length; q++) { const k = f._edgeHinge.get(f._key(line[q - 1], line[q])); if (k !== undefined) { sum += Math.abs(angle[k]); cnt++; } }
    res.creaseDeg = +(sum / Math.max(1, cnt) * 180 / Math.PI).toFixed(1); res.creaseWantDeg = +(want * 180 / Math.PI).toFixed(1);
  }
  if (cfg.core !== 'v1' && f.stiff && f.stiff.o.creep > 0) {
    // the set: points whose rest curvature has moved by more than a bend of 15 cm radius (creases taken)
    const k = f.stiff.setCurvature(), lim = 1 / (0.15 * U);
    let set = 0, big = 0;
    for (const v of k) { if (v > lim) set++; big = Math.max(big, v); }
    res.setPoints = set; res.tightestSetCm = big > 0 ? +(100 / (big * U)).toFixed(1) : null;
  }
  if (false) {
    const { rest: r1 } = f.hingeAngles(), r0 = f.hinges.rest;
    let set = 0, big = 0;
    for (let k = 0; k < r1.length; k++) { const d = Math.abs(r1[k] - r0[k]); if (d > 0.15) set++; big = Math.max(big, d); }
    res.creases = set; res.largestSetRad = +big.toFixed(2);
  }
  const tris = cfg.core === 'v1' ? trisOf(scene) : f.tris;
  return { res, frames: kept, tris: Array.from(tris), scene, fabric: f };
}
// each point's set as a share of the tightest a material can take (0 none, 1 a set at its setRadius), for the
// clay renders to darken where the leather has creased
function setMap(f) {
  if (!f.stiff || !(f.stiff.o.setRadius > 0)) return null;
  const k = f.stiff.setCurvature(), r = f.stiff.o.setRadius;
  return Array.from(k, (v) => +Math.min(1, v * r).toFixed(3));
}
function trisOf(scene) {
  const t = [], at = (c, r) => r * scene.cols + (c % scene.cols);
  for (let r = 0; r + 1 < scene.rows; r++) for (let c = 0; c + 1 < scene.cols + (scene.wrap ? 1 : 0); c++) {
    const p = at(c, r), q = at(c + 1, r), u = at(c, r + 1), v = at(c + 1, r + 1);
    if ((c + r) & 1) t.push(p, u, v, p, v, q); else t.push(p, u, q, q, u, v);
  }
  return t;
}
function partJSON(b) {
  if (b.a) return { capsule: [b.a, b.b, b.ra, b.rb] };
  if (b.c) return { sphere: [b.c, b.r] };
  return { floor: b.h };
}

// the cantilever: the strip's bending length (m) for a material's settings
export function cantilever(matName, over = {}, overhang = 0.15) {
  const scene = SCENES.cantilever(overhang), f = build(scene, CONFIGS.final, matName, { noTether: true, ...over });
  for (let k = 0; k < 2 * 240; k++) f.step(1 / 240);
  f.settle(null, { h: 1 / 240, max: 3000, tol: 1e-6, damping: 0.3 });
  const tipRow = scene.rows - 1, mid = Math.floor(scene.cols / 2), tip = 3 * (tipRow * scene.cols + mid);
  const drop = 20 - f.pos[tip + 1], reach = f.pos[tip + 2] - scene.edge;
  const theta = Math.atan2(drop, reach), l = overhang;
  const c = l * Math.cbrt(Math.cos(theta / 2) / (8 * Math.tan(theta)));
  return { theta: +(theta * 180 / Math.PI).toFixed(1), bendingLength: +c.toFixed(4) };
}

if (process.argv[1] && import.meta.url.endsWith(path.basename(process.argv[1]))) {
  if (flag('cantilever')) {
    const mats = (arg('material', Object.keys(MATERIALS).join(','))).split(',');
    for (const m of mats) {
      const over = { ...(arg('hinge') ? { hinge: Number(arg('hinge')) } : {}), ...JSON.parse(arg('over', '{}')) };
      // (the overhang the test is made at: long enough to droop, short enough not to hang straight down)
      const l = Math.min(0.25, Math.max(0.05, 2.5 * MATERIALS[m].bendingLength));
      console.log(m, JSON.stringify({ target: MATERIALS[m].bendingLength, overhang: l, ...cantilever(m, over, l) }));
    }
    process.exit(0);
  }
  const scenes = arg('scenes', 'hang,drape,fold,boot,tasset').split(','), cfgs = arg('configs', 'v1,draft,final').split(',');
  const out = arg('out', null), rows = [];
  for (const sc of scenes) {
    const matName = arg('material', sc === 'boot' ? 'bootLeather' : sc === 'tasset' ? 'leather' : sc === 'fold' ? 'wool' : 'linen');
    for (const c of cfgs) {
      const S = SCENES[sc]();
      const keep = arg('keep') ? arg('keep').split(',').map(Number) : S.pinAt ? [0.25, 0.5, 0.75, 1].map((k) => Math.round(S.frames - 1 - (1 - k) * 0.64 * 60)) : [S.frames - 1];
      const r = run(sc, c, matName, { keep });
      rows.push(r.res);
      if (!flag('json')) console.log(JSON.stringify(r.res));
      if (out) {
        fs.mkdirSync(out, { recursive: true });
        fs.writeFileSync(path.join(out, `${sc}-${c}.json`), JSON.stringify({ scene: sc, solver: c, material: r.res.material, cols: r.scene.cols, rows: r.scene.rows, tris: r.tris, camera: r.scene.camera, frames: r.frames }));
      }
    }
  }
  if (flag('json')) console.log(JSON.stringify(rows, null, 1));
}
