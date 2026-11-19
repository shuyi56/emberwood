// the physics the cape should show: her weight, the forces of her walk and run, the cape's weight and the air on it
import fs from 'fs';
import * as THREE from 'three';
import { makeDummy } from '../../src/cape-lab/dummy.js';
const U = 8.6 / 0.218;                 // units a metre, from her head (chin to crown 8.6 units; an adult woman's is about 21.8 cm)
const g = 9.81, rho = 1.225;
const VOL = 3319;                      // her volume, units^3 (anatomy/mannequin-v14.blend: skin, neck, hands, head; Monte Carlo)
const height = (62.8 + 7.4) / U, mass = VOL / U ** 3 * 1010;
console.log('scale', U.toFixed(1), 'u/m; height', height.toFixed(2), 'm; volume', (VOL / U ** 3 * 1000).toFixed(1), 'L; mass', mass.toFixed(1), 'kg; BMI', (mass / height ** 2).toFixed(1));
const legScale = 35.2 / 27.8;          // her leg (thigh + shin, 31 down to -4.2) over the figure the gaits were made on
function gait(file, secs, stride) {
  const d = JSON.parse(fs.readFileSync(file)), n = d.bones.length, T = secs;
  const series = (b, c) => d.bones.map((f) => f[b].p[c] * legScale / U);
  // smooth periodic second derivative: a Fourier fit, harmonics up to 6 of the cycle
  const acc = (y) => { const out = new Array(n).fill(0); for (let k = 1; k <= 2; k++) { let a = 0, b = 0; y.forEach((v, i) => { a += v * Math.cos(2 * Math.PI * k * i / n); b += v * Math.sin(2 * Math.PI * k * i / n); }); a *= 2 / n; b *= 2 / n; const w = 2 * Math.PI * k / T; for (let i = 0; i < n; i++) out[i] -= w * w * (a * Math.cos(2 * Math.PI * k * i / n) + b * Math.sin(2 * Math.PI * k * i / n)); } return out; };
  const hy = series('Pelvis', 1), cy = series('Chest', 1), sx = series('Pelvis', 0);
  const ay = acc(hy), acy = acc(cy), ax = acc(sx);
  const speed = stride * legScale / U / T;
  return { speed, bob: Math.max(...hy) - Math.min(...hy), grfMax: Math.max(...ay.map((a) => (g + a) / g)), grfMin: Math.min(...ay.map((a) => (g + a) / g)),
    chestUp: Math.max(...acy), chestDown: Math.min(...acy), sway: Math.max(...ax.map(Math.abs)) };
}
const walk = gait(new URL('../../public/cape-lab/walk-new.json', import.meta.url), 0.69, 46 / legScale);      // (the bench walks 46 units a cycle in 0.69 s)
const runD = JSON.parse(fs.readFileSync(new URL('../../public/cape-lab/walk-run.json', import.meta.url)));
const run = gait(new URL('../../public/cape-lab/walk-run.json', import.meta.url), runD.seconds_per_cycle, runD.stride);
for (const [k, r] of [['walk', walk], ['run', run]]) console.log(k, 'speed', r.speed.toFixed(2), 'm/s; hip bob', (r.bob * 100).toFixed(1), 'cm; ground force', r.grfMin.toFixed(2), 'to', r.grfMax.toFixed(2), 'bodyweights =', (r.grfMax * mass * g).toFixed(0), 'N peak; chest accel', r.chestDown.toFixed(1), 'to', r.chestUp.toFixed(1), 'm/s2; side', r.sway.toFixed(1));
// the cape: its area from the bench's cape (outer face)
const body = makeDummy({ columns: 7, rows: 5 }); const geo = body.cape.mesh.geometry, p = geo.attributes.position, idx = geo.index;
let area = 0; const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
for (let i = 0; i < idx.count; i += 3) { a.fromBufferAttribute(p, idx.getX(i)); b.fromBufferAttribute(p, idx.getX(i + 1)); c.fromBufferAttribute(p, idx.getX(i + 2)); area += b.sub(a).cross(c.sub(a)).length() / 2; }
const A = area / U ** 2, GSM = 0.45 + 0.1;  // heavy wool (melton, 450 g/m2) and a lining (100 g/m2)
const mc = A * GSM;
console.log('cape area', A.toFixed(2), 'm2; mass', mc.toFixed(2), 'kg; weight', (mc * g).toFixed(1), 'N (', (100 * mc / mass).toFixed(1), '% of her)');
// the air: behind her back the cape is in her wake, so only part of it meets the air; Cd 1.2 (a flat sheet) on 35% of it
const CdA = 1.2 * 0.35 * A;
for (const [k, r] of [['walk', walk], ['run', run]]) {
  const D = 0.5 * rho * CdA * r.speed ** 2, ratio = D / (mc * g);
  // a sheet hanging from a hinge, blown back: weight and drag balance at sin(t) = (D/W) cos^2(t)
  let t = 0; for (let it = 0; it < 60; it++) t = Math.asin(Math.min(1, ratio * Math.cos(t) ** 2));
  console.log(k, 'drag', D.toFixed(2), 'N =', ratio.toFixed(2), 'of its weight; hem lifts back', (t * 180 / Math.PI).toFixed(0), 'deg; the chest shakes it with up to', (mc * Math.max(run.chestUp, -run.chestDown)).toFixed(1), 'N');
}
console.log('quadratic drag a = c v^2: c =', (0.5 * rho * CdA / mc).toFixed(3), '1/m');
