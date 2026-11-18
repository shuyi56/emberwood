// Numbers for the cape springs, without a browser: runs the cape bench's stand-in (src/cape-lab/sim.js) in node
// and reports what a capture can't show at a glance.
//
//   node tools/cape-lab/check.mjs [--settings '{"stiffness":300}'] [--columns 5] [--rows 5] [--walk new]
//
//  - loop drift: on the straight walk, how far any tail lands from where it was one cycle earlier (0 means the
//    cape repeats exactly with the walk, so a looping clip shows no seam)
//  - swing: the hem's largest angle from hanging straight, while walking and in the turns
//  - settle: after she stops, how long until the hem stays within 1 unit of where it comes to rest
//  - frame rate: the hem at 30 and 144 fps against 60, over the start-and-stop (units, largest gap)
//  - clearance: the closest any tail comes to the body's capsules (negative is inside)
//  - cloth in body: the deepest any vertex of the skinned cape (below the shoulders) goes into a body capsule
//    through the sharp turns, where an arm or a leg would show through
//  - jitter: how much the cape shakes from frame to frame (the mean size of each tail's third difference over
//    frames, in her own space: a smooth swing, however fast, scores low; a flicker scores high)
import fs from 'fs';
import * as THREE from 'three';
import { CapeSim, FRAME, CYCLE_FRAMES } from '../../src/cape-lab/sim.js';
import { CAPE_SETTINGS } from '../../src/world/cape-rig.js';

const arg = (name, def) => {
  const i = process.argv.indexOf('--' + name);
  return i > 0 ? process.argv[i + 1] : def;
};
const settings = { ...CAPE_SETTINGS, colliderScale: 1, ...JSON.parse(arg('settings', '{}')) };
// --walk new|main: a baked walk (public/cape-lab/walk-<name>.json) in place of the stand-in's walk made in code
const walkName = arg('walk', null);
const walkData = walkName ? JSON.parse(fs.readFileSync(new URL(`../../public/cape-lab/walk-${walkName}.json`, import.meta.url))) : null;
const base = { columns: Number(arg('columns', 7)), rows: Number(arg('rows', 5)), springs: true, walkData };
const sim = (scenario) => new CapeSim({ ...settings }, { ...base, scenario });
// the hem: each chain's last tail, in her own space
const hem = (s) => {
  const all = s.tails();
  let k = 0;
  return s.springs.chains.map((ch) => { k += ch.joints.length; return all[k - 1]; });
};
const rest = sim('walk');
const restHem = hem(rest);
const swingOf = (s) => Math.max(...hem(s).map((p, i) => {
  const r = restHem[i];
  const top = s.springs.chains[i].joints[0];
  // angle at the chain's top between hanging and now (in her space, the top is fixed)
  const inv = s.dummy.root.matrixWorld.clone().invert();
  const h = top.head.clone().applyMatrix4(inv);
  return p.clone().sub(h).angleTo(r.clone().sub(h));
})) * 180 / Math.PI;
function clearance(s) {
  let worst = Infinity;
  for (const ch of s.springs.chains) for (const j of ch.joints) for (const c of s.springs.colliders) {
    c.refresh();
    const ab = c.wb.clone().sub(c.wa);
    const t = ab.lengthSq() > 1e-9 ? Math.min(1, Math.max(0, j.tail.clone().sub(c.wa).dot(ab) / ab.lengthSq())) : 0;
    const d = j.tail.distanceTo(c.wa.clone().addScaledVector(ab, t)) - c.radius * c.scale;
    worst = Math.min(worst, d);
  }
  return worst;
}

// loop drift and walking swing
const w = sim('walk');
let walkSwing = 0, minClear = Infinity;
for (let i = 0; i < CYCLE_FRAMES * 12; i++) {
  w.step();
  if (i > CYCLE_FRAMES * 4) { walkSwing = Math.max(walkSwing, swingOf(w)); minClear = Math.min(minClear, clearance(w)); }
}
// turns
const tr = sim('turn');
let turnSwing = 0;
for (let i = 0; i < 7 * 60; i++) { tr.step(); turnSwing = Math.max(turnSwing, swingOf(tr)); minClear = Math.min(minClear, clearance(tr)); }
// settle after the stop at 3.5 s (she is still by about 3.9 s)
const ss = sim('startstop');
const track = [];
for (let i = 0; i < 6 * 60; i++) { ss.step(); track.push(hem(ss)); }
const final = track[track.length - 1];
let settled = 3.5;
for (let i = track.length - 1; i >= 0; i--) {
  if (Math.max(...track[i].map((p, k) => p.distanceTo(final[k]))) > 1) { settled = (i + 1) * FRAME; break; }
}
// frame rate: run the start-and-stop at 30 and 144 fps, compare with 60 at the same times
function runAt(fps) {
  const s = sim('startstop');
  const out = [];
  const dt = 1 / fps;
  for (let t = 0; t < 5.5; t += dt) { s.step(dt); out.push([s.t, hem(s)]); }
  return out;
}
const ref = runAt(60);
const at = (series, t) => series.reduce((best, e) => (Math.abs(e[0] - t) < Math.abs(best[0] - t) ? e : best));
function gap(fps) {
  let g = 0;
  for (const [t, h] of runAt(fps)) {
    const [t2, h2] = at(ref, t);
    if (Math.abs(t2 - t) > 0.004) continue;
    g = Math.max(g, ...h.map((p, i) => p.distanceTo(h2[i])));
  }
  return g;
}
// cloth in body, through the turns
function clothInBody() {
  const s = sim('turn');
  const mesh = s.dummy.cape.mesh, pos = mesh.geometry.attributes.position, v = new THREE.Vector3();
  let worst = 0;
  for (let f = 0; f < 7 * 60; f++) {
    s.step();
    if (f % 2) continue;
    mesh.skeleton.update();
    for (const c of s.springs.colliders) c.refresh();
    for (let i = 0; i < pos.count; i++) {
      if (pos.getY(i) > 45) continue;          // the band over the shoulders sits on them by design
      v.fromBufferAttribute(pos, i);
      mesh.boneTransform(i, v);
      v.applyMatrix4(mesh.matrixWorld);
      for (const c of s.springs.colliders) {
        const ab = c.wb.clone().sub(c.wa);
        const t = Math.min(1, Math.max(0, v.clone().sub(c.wa).dot(ab) / Math.max(1e-9, ab.lengthSq())));
        worst = Math.max(worst, c.radius * c.scale - v.distanceTo(c.wa.clone().addScaledVector(ab, t)));
      }
    }
  }
  return worst;
}
function jitter(scenario, secs) {
  const s = sim(scenario);
  const hist = [];
  let sum = 0, n = 0;
  for (let f = 0; f < secs * 60; f++) {
    s.step();
    hist.push(s.tails());
    if (hist.length > 4) hist.shift();
    if (hist.length < 4) continue;
    const [a, b, c, d] = hist;
    for (let i = 0; i < a.length; i++) {
      sum += d[i].clone().sub(c[i].clone().multiplyScalar(3)).add(b[i].clone().multiplyScalar(3)).sub(a[i]).length();
      n++;
    }
  }
  return sum / n;
}
const report = {
  loopDrift: +w.drift.toFixed(4),
  walkSwingDeg: +walkSwing.toFixed(1),
  turnSwingDeg: +turnSwing.toFixed(1),
  settleAfterStopS: +(settled - 3.5).toFixed(2),
  fps30Gap: +gap(30).toFixed(2),
  fps144Gap: +gap(144).toFixed(2),
  minClearance: +minClear.toFixed(2),
  clothInBody: +clothInBody().toFixed(2),
  jitterWalk: +jitter('walk', 4).toFixed(3),
  jitterStartStop: +jitter('startstop', 6).toFixed(3),
  jitterTurn: +jitter('turn', 7).toFixed(3),
};
console.log(JSON.stringify(report, null, 2));
