// The lace's bending length on this solver: the cantilever test (Peirce 1930; ASTM D1388), run as the fabric bench
// runs it (tools/fabric-bench/bench.mjs, cantilever): a strip 2.5 cm wide and 15 cm over a table's edge, held flat
// behind the edge; once it rests, its tip's droop angle theta gives c = l (cos(theta / 2) / (8 tan theta))^(1/3).
//
//   node tools/lace-study/calibrate.mjs                 hinges: the compliance that gives 1.0 cm at each mesh size
//                                                       (they cannot reach it on a fine mesh: why the study bends
//                                                       with the implicit solve instead)
//   node tools/lace-study/calibrate.mjs --check         each lace's bending length with the implicit bend
import { Fabric } from './fabric-core.js';
import { LACE, U } from './lace.js';

export function cantilever({ compliance, res = 0.3, over = 6, implicit = null, g = 981, iterations = 2, substep = 1 / 960 }) {
  const len = over + 4, w = 2.5, cols = Math.max(3, Math.round(w / res) + 1), rows = Math.round(len / res) + 1, d = len / (rows - 1);
  const at = (c, r) => r * cols + c, n = cols * rows, pos = new Float64Array(3 * n);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) { const i = at(c, r); pos[3 * i] = (c / (cols - 1) - 0.5) * w; pos[3 * i + 1] = 20; pos[3 * i + 2] = r * d; }
  const f = new Fabric(pos, { gravity: [0, -g, 0], iterations, substep, stretch: 0, damping: 0.05, order: 'root' });
  const edge = len - over;
  for (let r = 0; r < rows; r++) if (r * d <= edge + 1e-6) for (let c = 0; c < cols; c++) f.pin(at(c, r));
  f.sheet(cols, rows, at, implicit ? { bend: null, hinge: null, shear: 4e-5 } : { bend: null, hinge: compliance ?? 1, shear: 4e-5, flat: true });
  f.weigh(0.05 / (U * U));
  if (implicit) f.stiffen({ bendingLength: implicit, density: 0.05 / (U * U), passes: 2 });
  for (let k = 0; k < 2 / substep; k++) f.step(substep);
  f.settle(null, { h: substep, max: 12000, tol: 1e-7, damping: 0.3 });
  const tip = 3 * at(Math.floor(cols / 2), rows - 1);
  const drop = 20 - f.pos[tip + 1], reach = f.pos[tip + 2] - edge;
  const theta = Math.atan2(drop, reach);
  const c = over * Math.cbrt(Math.cos(theta / 2) / (8 * Math.tan(theta)));
  return { thetaDeg: theta * 180 / Math.PI, c };
}

// The hinge compliance that gives bending length c (cm) on a mesh of spacing res, at the study's solver settings
// (SOLVER in lace.js), by bisection on the ASTM D1388 test: overhang 2c, where the tip should droop 41.5 degrees
export function calibrate(c, res, solver) {
  let lo = -9, hi = 2;
  for (let it = 0; it < 18; it++) {
    const mid = (lo + hi) / 2, r = cantilever({ compliance: 10 ** mid, res, over: 2 * c, ...solver });
    if (r.c > c) lo = mid; else hi = mid;
  }
  const k = 10 ** ((lo + hi) / 2);
  const at2 = cantilever({ compliance: k, res, over: 2 * c, ...solver }), at3 = cantilever({ compliance: k, res, over: 3 * c, ...solver });
  return { c, res, compliance: k, measured: at2.c, thetaDeg: at2.thetaDeg, cAt3c: at3.c };
}

if (process.argv[1] && process.argv[1].endsWith('calibrate.mjs')) {
  const { SOLVER } = await import('./lace.js');
  const rows = [];
  for (const c of [0.6, 1.0, 1.5, 2.5]) for (const res of [0.25, 0.3, 0.4]) {
    const r = calibrate(c, res, SOLVER);
    rows.push(r);
    console.log(JSON.stringify(r));
  }
  // compliance = K c^a res^b, least squares in logs
  const X = rows.map((r) => [1, Math.log(r.c), Math.log(r.res)]), Y = rows.map((r) => Math.log(r.compliance));
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b = [0, 0, 0];
  X.forEach((x, k) => { for (let i = 0; i < 3; i++) { b[i] += x[i] * Y[k]; for (let j = 0; j < 3; j++) A[i][j] += x[i] * x[j]; } });
  for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) { const q = A[j][i] / A[i][i]; for (let k = 0; k < 3; k++) A[j][k] -= q * A[i][k]; b[j] -= q * b[i]; }
  const z = [0, 0, 0];
  for (let i = 2; i >= 0; i--) { let s = b[i]; for (let j = i + 1; j < 3; j++) s -= A[i][j] * z[j]; z[i] = s / A[i][i]; }
  console.log(JSON.stringify({ fit: { K: Math.exp(z[0]), cPow: z[1], resPow: z[2] } }));
}
