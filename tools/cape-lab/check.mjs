// Numbers for the cape springs, without a browser: runs the cape bench's stand-in (src/cape-lab/sim.js) in node
// and reports what a capture can't show at a glance.
//
//   node tools/cape-lab/check.mjs [--settings '{"stiffness":300}'] [--columns 5] [--rows 5]
//
//  - loop drift: on the straight walk, how far any tail lands from where it was one cycle earlier (0 means the
//    cape repeats exactly with the walk, so a looping clip shows no seam)
//  - swing: the hem's largest angle from hanging straight, while walking and in the turns
//  - settle: after she stops, how long until the hem stays within 1 unit of where it comes to rest
//  - frame rate: the hem at 30 and 144 fps against 60, over the start-and-stop (units, largest gap)
//  - clearance: the closest any tail comes to the body's capsules (negative is inside)
import { CapeSim, FRAME, CYCLE_FRAMES } from '../../src/cape-lab/sim.js';
import { CAPE_SETTINGS } from '../../src/world/cape-rig.js';

const arg = (name, def) => {
  const i = process.argv.indexOf('--' + name);
  return i > 0 ? process.argv[i + 1] : def;
};
const settings = { ...CAPE_SETTINGS, colliderScale: 1, ...JSON.parse(arg('settings', '{}')) };
const base = { columns: Number(arg('columns', 7)), rows: Number(arg('rows', 5)), springs: true };
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
const report = {
  loopDrift: +w.drift.toFixed(4),
  walkSwingDeg: +walkSwing.toFixed(1),
  turnSwingDeg: +turnSwing.toFixed(1),
  settleAfterStopS: +(settled - 3.5).toFixed(2),
  fps30Gap: +gap(30).toFixed(2),
  fps144Gap: +gap(144).toFixed(2),
  minClearance: +minClear.toFixed(2),
};
console.log(JSON.stringify(report, null, 2));
