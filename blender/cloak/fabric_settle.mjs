// The cloak hung to rest by the project's own fabric solver (src/world/fabric-core.js: the XPBD sheet the game's
// cape and the Face Creator's hair drape run on), not Blender's cloth. blender/cloak/sew.py writes the sewn cloak's
// points, its ties (threads, bias and bend, from each pattern piece's own grid), the pinned points and the body as
// a distance grid; this runs Fabric.settle on them and writes the points back.
//
//   node blender/cloak/fabric_settle.mjs IN.json OUT.bin
//
// IN.json: { pos: "pos.bin" (float64, 3n), ties: "ties.bin" (int32, 3m: i, j, kind 0 thread 1 bias 2 bend),
//            pins: [..], settings: {...}, compliance: [thread, bias, bend], gap,
//            grid: { file: "sdf.bin" (float32), origin: [x, y, z], step, dims: [nx, ny, nz] }, settle: {...} }
import fs from 'fs';
import path from 'path';
import { Fabric } from '../../src/world/fabric-core.js';

const [inPath, outPath] = process.argv.slice(2);
const cfg = JSON.parse(fs.readFileSync(inPath, 'utf8'));
const dir = path.dirname(inPath);
const f64 = (f) => { const b = fs.readFileSync(path.join(dir, f)); return new Float64Array(b.buffer, b.byteOffset, b.byteLength / 8); };
const i32 = (f) => { const b = fs.readFileSync(path.join(dir, f)); return new Int32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const f32 = (f) => { const b = fs.readFileSync(path.join(dir, f)); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };

const pos = Float64Array.from(f64(cfg.pos));
const fab = new Fabric(pos, cfg.settings);
for (const i of cfg.pins) fab.pin(i);
const ties = i32(cfg.ties);
for (let k = 0; k < ties.length; k += 3) fab.tie(ties[k], ties[k + 1], cfg.compliance[ties[k + 2]]);
// the warp: each thread from the neck to the hem held at its length (the solver's held chains, follow the leader)
for (const ch of cfg.chains || []) {
  const x = fab.pos;
  ch.len = ch.slice(1).map((i, k) => Math.hypot(x[3 * i] - x[3 * ch[k]], x[3 * i + 1] - x[3 * ch[k] + 1], x[3 * i + 2] - x[3 * ch[k] + 2]));
  fab.chains.push(ch);
}

// the body: signed distance, trilinear in the grid; a point closer than the gap is pushed out along the gradient
const G = cfg.grid, D = f32(G.file), [nx, ny, nz] = G.dims, h = G.step, [ox, oy, oz] = G.origin;
const at = (i, j, k) => D[(Math.min(nz - 1, Math.max(0, k)) * ny + Math.min(ny - 1, Math.max(0, j))) * nx + Math.min(nx - 1, Math.max(0, i))];
function sdf(x, y, z) {
  const fx = (x - ox) / h, fy = (y - oy) / h, fz = (z - oz) / h;
  if (fx < 0 || fy < 0 || fz < 0 || fx > nx - 1 || fy > ny - 1 || fz > nz - 1) return 1e9;
  const i = Math.floor(fx), j = Math.floor(fy), k = Math.floor(fz), u = fx - i, v = fy - j, w = fz - k;
  const c = (a, b, t) => a + (b - a) * t;
  return c(c(c(at(i, j, k), at(i + 1, j, k), u), c(at(i, j + 1, k), at(i + 1, j + 1, k), u), v),
           c(c(at(i, j, k + 1), at(i + 1, j, k + 1), u), c(at(i, j + 1, k + 1), at(i + 1, j + 1, k + 1), u), v), w);
}
const gap = cfg.gap, e = 0.5 * h;
function body(p) {
  const d = sdf(p[0], p[1], p[2]);
  if (d >= gap) return false;
  let gx = sdf(p[0] + e, p[1], p[2]) - sdf(p[0] - e, p[1], p[2]);
  let gy = sdf(p[0], p[1] + e, p[2]) - sdf(p[0], p[1] - e, p[2]);
  let gz = sdf(p[0], p[1], p[2] + e) - sdf(p[0], p[1], p[2] - e);
  const gl = Math.hypot(gx, gy, gz);
  if (gl < 1e-9) return false;
  gx /= gl; gy /= gl; gz /= gl;
  const push = gap - d;
  p[0] += gx * push; p[1] += gy * push; p[2] += gz * push;
  return true;
}
const t0 = Date.now();
const steps = fab.settle(body, cfg.settle || {});
fs.writeFileSync(outPath, Buffer.from(fab.pos.buffer));
console.log(`fabric: ${fab.n} points, ${ties.length / 3} ties, settled in ${steps} steps (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
