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
// How far weight and drag lift something hanging from a hinge, blown back, for drag/weight ratio r. A flat sheet meets
// the air with only the part of the wind that is normal to it, so its drag falls with cos^2 of the lift:
// sin(t) = r cos^2(t), which is r s^2 + s - r = 0 for s = sin(t). A hanging mass (a bunch of strands, a point on a
// chain) meets the same wind at any angle: tan(t) = r.
const lifts = (r) => { const sheet = Math.asin((-1 + Math.sqrt(1 + 4 * r * r)) / (2 * r || 1)), mass = Math.atan(r);
  return (sheet * 180 / Math.PI).toFixed(0) + ' (as a sheet) to ' + (mass * 180 / Math.PI).toFixed(0) + ' (as a hanging mass) deg'; };
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
  console.log(k, 'drag', D.toFixed(2), 'N =', ratio.toFixed(2), 'of its weight; hem lifts back', lifts(ratio), '; the chest shakes it with up to', (mc * Math.max(run.chestUp, -run.chestDown)).toFixed(1), 'N');
}
console.log('quadratic drag a = c v^2: c =', (0.5 * rho * CdA / mc).toFixed(3), '1/m');

// ---- her hair ----
// A hair is a keratin cylinder: 75 micrometres across, 1300 kg/m3. A head carries about 90000 of them. Seen from the
// air each one is a cylinder in cross flow (Cd 1.2), but in a mass they shield each other: a sheet of them as wide as
// her back is some 30 layers deep, so only a share of them meets the air. 6% is taken here (the outer layer, and the
// gaps in it).
const nHair = 90000, d = 75e-6, rhoH = 1300, Cdh = 1.2, share = 0.06;
const lenH = (62.8 - 44.6) / U + 0.12;               // down her back from the crown to where it ends (the Warden's hair), and over the head
const mh = nHair * lenH * Math.PI * d * d / 4 * rhoH;
const ch = 2 * rho * Cdh * share / (Math.PI * rhoH * d);     // a = ch v^2, per metre
console.log('\nhair: mass', (mh * 1000).toFixed(0), 'g over', lenH.toFixed(2), 'm; drag c =', ch.toFixed(2), '1/m (the cape\'s is', (0.5 * rho * CdA / mc).toFixed(2), ')');
for (const [k, r] of [['walk', walk], ['run', run]]) {
  const ratio = ch * r.speed ** 2 / g;
  console.log(k, 'drag is', ratio.toFixed(2), 'of its weight; hair lifts back', lifts(ratio));
}
// one hair on its own cannot hold itself up: its bending stiffness EI (E 3 GPa dry, I = pi d^4 / 64) against its weight w
// gives the length a free strand holds out before it sags under its own weight, (8 EI / w)^(1/3)
const EI = 3e9 * Math.PI * d ** 4 / 64, w = rhoH * Math.PI * d * d / 4 * g;
console.log('one hair: EI', EI.toExponential(1), 'N m2; holds itself out only', (Math.cbrt(8 * EI / w) * 100).toFixed(1), 'cm: at her hair\'s length it is limp, so what keeps a style is the body it lies on and the strands pressing on each other');
