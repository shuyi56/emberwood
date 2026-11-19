// The fabric bench's test scenes: fixed sheets on fixed bodies, so the solver can be timed and measured, the same
// every run, and the old solver set against the new one on the same cloth.
//
//  hang       a 1 m square sheet held at its two top corners, dropped from flat: stretch, settling, speed
//  drape      the same sheet dropped centred on a sphere (a drape tester's table): folds, settling, contact
//  cantilever a strip pushed flat over a table's edge (Peirce 1930): its bending length, to set each material
//  boot       a leather boot shaft round the base figure's shin through her run: the ankle flexing, creases
//  tasset     a belted leather skirt panel over her hips and thighs through her run: weight, the air, the kick
//
// Units are the figures' (39.4 a metre, blender/base, tools/cape-lab/physics.mjs). The body is a set of capsules
// that taper (a radius at each end); the run is public/cape-lab/walk-run.json, the base figure's run, played in her
// own space (she runs on the spot, the air coming at her at her running speed).
import fs from 'fs';

export const U = 39.4;
const RUN = JSON.parse(fs.readFileSync(new URL('../../public/cape-lab/walk-run.json', import.meta.url)));
export const RUN_CYCLE = RUN.seconds_per_cycle;
export const RUN_SPEED = RUN.stride / RUN.seconds_per_cycle;          // units / s (3.1 m/s)

// --- small vector helpers on [x, y, z] ---
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const crs = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const nrm = (a) => mul(a, 1 / (Math.hypot(...a) || 1));
const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
function qrot(q, v) {
  const [x, y, z, w] = q, ix = w * v[0] + y * v[2] - z * v[1], iy = w * v[1] + z * v[0] - x * v[2], iz = w * v[2] + x * v[1] - y * v[0], iw = -x * v[0] - y * v[1] - z * v[2];
  return [ix * w + iw * -x + iy * -z - iz * -y, iy * w + iw * -y + iz * -x - ix * -z, iz * w + iw * -z + ix * -y - iy * -x];
}
function qlerp(a, b, t) {
  const s = dot([a[0], a[1], a[2]], [b[0], b[1], b[2]]) + a[3] * b[3] < 0 ? -1 : 1;
  const q = [0, 1, 2, 3].map((k) => a[k] + (s * b[k] - a[k]) * t), l = Math.hypot(...q);
  return q.map((c) => c / l);
}

// the run at time t (s): each joint's pivot and turn, between the baked frames
export function runPose(t) {
  const F = RUN.bones.length, u = ((t / RUN_CYCLE) % 1 + 1) % 1 * F, i = Math.floor(u) % F, j = (i + 1) % F, k = u - Math.floor(u);
  const out = {};
  for (const name of Object.keys(RUN.bones[i])) out[name] = { p: lerp(RUN.bones[i][name].p, RUN.bones[j][name].p, k), q: qlerp(RUN.bones[i][name].q, RUN.bones[j][name].q, k) };
  return out;
}
// the left leg's points at time t: hip, knee, ankle, toe, heel
export function legPoints(t, side = 'L') {
  const P = runPose(t), hip = P['Thigh.' + side].p, knee = P['Shin.' + side].p, ankle = P['Foot.' + side].p;
  const fq = P['Foot.' + side].q, toe = add(ankle, qrot(fq, mul(nrm([0, -0.45, 1]), 8.6))), heel = add(ankle, qrot(fq, [0, -1.6, -1.6]));
  return { hip, knee, ankle, toe, heel, footQ: fq, pelvis: P.Pelvis };
}

// --- the body: tapered capsules ---
export class Capsule {
  constructor(a, b, ra, rb) {
    Object.assign(this, { a, b, ra, rb });
    this.ab = sub(b, a); this.L2 = dot(this.ab, this.ab);
  }
  // push p (changed in place) out to the surface plus `pad`; returns the depth it was in (0 if outside).
  // ref: where the point was a step ago: a point more than halfway in is pushed out on the side it came from, not
  // through to the far side (a limb moving fast carries the cloth with it instead of passing through it)
  push(p, pad = 0, ref = null) {
    const A = this.a, ab = this.ab, L2 = this.L2;
    // (no arrays made: this runs for every point and edge, every walk)
    const qx = p[0] - A[0], qy = p[1] - A[1], qz = p[2] - A[2];
    let t = L2 > 1e-12 ? (qx * ab[0] + qy * ab[1] + qz * ab[2]) / L2 : 0;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const cx = A[0] + ab[0] * t, cy = A[1] + ab[1] * t, cz = A[2] + ab[2] * t;
    let dx = p[0] - cx, dy = p[1] - cy, dz = p[2] - cz;
    const dl = Math.sqrt(dx * dx + dy * dy + dz * dz), r = this.ra + (this.rb - this.ra) * t + pad;
    if (dl >= r) return 0;
    let ux, uy, uz;
    if (ref && dl < 0.5 * r) {
      let ex = ref[0] - cx, ey = ref[1] - cy, ez = ref[2] - cz;
      const k = (ex * ab[0] + ey * ab[1] + ez * ab[2]) / (L2 || 1);
      ex -= ab[0] * k; ey -= ab[1] * k; ez -= ab[2] * k;
      const el = Math.sqrt(ex * ex + ey * ey + ez * ez);
      if (el > 1e-9) { ux = ex / el; uy = ey / el; uz = ez / el; }
    }
    if (ux === undefined) {
      if (dl > 1e-9) { ux = dx / dl; uy = dy / dl; uz = dz / dl; }
      else { const u = nrm(crs(ab, [0, 0, 1])); ux = u[0]; uy = u[1]; uz = u[2]; }
    }
    p[0] = cx + ux * r; p[1] = cy + uy * r; p[2] = cz + uz * r;
    return r - dl;
  }
  depth(p) { const q = [p[0], p[1], p[2]]; return this.push(q, 0); }
}
export class Sphere {
  constructor(c, r) { Object.assign(this, { c, r }); }
  push(p, pad = 0) {
    const d = sub(p, this.c), dl = Math.hypot(...d), r = this.r + pad;
    if (dl >= r) return 0;
    const u = dl > 1e-9 ? mul(d, 1 / dl) : [0, 1, 0];
    p[0] = this.c[0] + u[0] * r; p[1] = this.c[1] + u[1] * r; p[2] = this.c[2] + u[2] * r;
    return r - dl;
  }
  depth(p) { const q = [...p]; return this.push(q, 0); }
}
// a flat floor (y = h), as a body part
export class Floor {
  constructor(h) { this.h = h; }
  push(p, pad = 0) { const y = this.h + pad; if (p[1] >= y) return 0; const d = y - p[1]; p[1] = y; return d; }
  depth(p) { return Math.max(0, this.h - p[1]); }
}

// a grid of points: cols x rows, pos(c, r) -> [x, y, z]
function grid(cols, rows, at) {
  const pos = new Float64Array(3 * cols * rows);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) pos.set(at(c, r), 3 * (r * cols + c));
  return pos;
}

// The scenes. Each: { name, cols, rows, wrap, pos (Float64Array), pins: [index], pinAt(t) -> Float64Array of the
// pins' positions (or null: they stay), body(t) -> [part], frames, fps, unitsPerMetre, camera (for the renders) }
const SHEET = 40;            // a metre, near enough (1.02 m)
export function hang(res = 25) {
  const cols = res, rows = res, d = SHEET / (res - 1);
  return {
    name: 'hang', cols, rows, pos: grid(cols, rows, (c, r) => [(c - (cols - 1) / 2) * d, 60, r * d - SHEET / 2]),
    pins: [0, cols - 1], body: () => [], frames: 180, fps: 60, wind: [0, 0, 0],
    camera: { at: [0, 40, 0], from: [0, 40, 120], size: 70 },
  };
}
export function drape(res = 25) {
  const cols = res, rows = res, d = SHEET / (res - 1), ball = new Sphere([0, 30, 0], 9), floor = new Floor(0);
  return {
    name: 'drape', cols, rows, pos: grid(cols, rows, (c, r) => [(c - (cols - 1) / 2) * d, 40, (r - (rows - 1) / 2) * d]),
    pins: [], body: () => [ball, floor], frames: 180, fps: 60, wind: [0, 0, 0],
    camera: { at: [0, 26, 0], from: [70, 70, 90], size: 60 },
  };
}
// the cantilever: a strip 2.5 cm wide, `over` metres of it past the edge, held flat on the table behind
export function cantilever(over = 0.15, res = 4) {
  const len = 0.3 * U, cols = res, rows = 49, d = len / (rows - 1), w = 0.025 * U;
  const pos = grid(cols, rows, (c, r) => [(c / (cols - 1) - 0.5) * w, 20, r * d]);
  const pins = [];
  const edge = (0.3 - over) * U;
  for (let r = 0; r < rows; r++) if (r * d <= edge + 1e-6) for (let c = 0; c < cols; c++) pins.push(r * cols + c);
  return { name: 'cantilever', cols, rows, pos, pins, body: () => [], frames: 240, fps: 60, wind: [0, 0, 0], over, edge, camera: { at: [0, 18, 8], from: [60, 18, 8], size: 20 } };
}

// the boot shaft: a cone of leather from the ankle to under the knee, its bottom edge sewn to the foot (pinned in
// the foot's frame), the rest held up by its own stiffness and the calf
export function boot(cols = 24, rows = 16) {
  const L0 = legPoints(0);
  const frame = (P) => {
    // the ankle's frame, where the shaft is sewn to the vamp and quarters: halfway between the foot's turn and the
    // shin's (the foot's: z toward the toe; the shin's: z down the shin turned a right angle forward), as the top
    // of a boot's foot bends with both. x across (her left)
    const zf = nrm(sub(P.toe, P.ankle)), down = nrm(sub(P.ankle, P.knee)), x0 = [1, 0, 0];
    const xs = nrm(crs(crs(down, x0), down)), zs = nrm(crs(xs, down).map((v) => -v));
    const z = nrm(add(zf, zs)), y = nrm(crs(z, x0)), x = crs(y, z);
    return { o: P.ankle, x, y, z };
  };
  const shinAxis = nrm(sub(L0.knee, L0.ankle)), shinLen = Math.hypot(...sub(L0.knee, L0.ankle));
  const side = nrm(crs(shinAxis, [0, 0, 1])), front = crs(side, shinAxis);
  // the shaft: from 8% of the way up the shin (its edge sewn to the vamp just over the ankle bone) to 72%; the radius of the leg plus ease
  // (the top clear of the back of the knee, where the thigh folds down over the shin when the knee bends: a
  // riding boot is cut so, or it is driven down the leg at every stride)
  const top = 0.72, bottom = 0.08;
  // (the leg's own radius, as the capsules below give it, plus the pad and a little ease: a riding boot is cut
  // close, and the calf, wider than the ankle, keeps the shaft from riding up or slipping down)
  const legR = (s) => (s < 0.62 ? 1.75 + 0.8 * s / 0.62 : 2.55 - 0.35 * (s - 0.62) / 0.38);
  // (snug at the calf, as a riding boot is fitted, easier at the ankle)
  const rAt = (s) => legR(s) + 0.12 + (s > 0.45 ? 0.03 : 0.25 - 0.22 * s / 0.45);
  const pos = new Float64Array(3 * cols * rows);
  for (let r = 0; r < rows; r++) {
    const s = top - (top - bottom) * r / (rows - 1), rr = rAt(Math.max(0, s));
    for (let c = 0; c < cols; c++) {
      const a = 2 * Math.PI * c / cols, p = add(add(L0.ankle, mul(shinAxis, s * shinLen)), add(mul(side, Math.cos(a) * rr), mul(front, Math.sin(a) * rr * 1.08)));
      pos.set(p, 3 * (r * cols + c));
    }
  }
  const pins = [];
  for (let c = 0; c < cols; c++) pins.push((rows - 1) * cols + c);
  // the shaft is cut in two weights: stiff leather from the calf up, and a softer, thinner leather over the ankle
  // (the bottom quarter), where a riding boot bends and creases with every step
  const stiffScale = new Float64Array(cols * rows);
  for (let r = 0; r < rows; r++) {
    const s = top - (top - bottom) * r / (rows - 1), k = Math.min(1, Math.max(0, (s - 0.2) / 0.15));
    for (let c = 0; c < cols; c++) stiffScale[r * cols + c] = 0.25 + 0.75 * k * k * (3 - 2 * k);
  }
  const F0 = frame(L0), local = pins.map((i) => { const d = sub(Array.from(pos.subarray(3 * i, 3 * i + 3)), F0.o); return [dot(d, F0.x), dot(d, F0.y), dot(d, F0.z)]; });
  const pinAt = (t) => {
    const F = frame(legPoints(t)), out = new Float64Array(3 * pins.length);
    local.forEach((l, k) => out.set(add(F.o, add(add(mul(F.x, l[0]), mul(F.y, l[1])), mul(F.z, l[2]))), 3 * k));
    return out;
  };
  // (the foot is under the vamp, which is sewn to the shaft's bottom edge and not simulated; it is a body part, so
  // the shaft's soft ankle folds over the instep instead of being pushed down through its own edge)
  const body = (t) => {
    const P = legPoints(t), ax = sub(P.knee, P.ankle);
    const calf = add(P.ankle, mul(ax, 0.62));
    return [
      new Capsule(P.ankle, calf, 1.75, 2.55),
      new Capsule(calf, P.knee, 2.55, 2.2),
      new Capsule(P.knee, lerp(P.knee, P.hip, 0.5), 2.3, 2.9),
      new Capsule(P.heel, P.toe, 1.5, 1.1),
    ];
  };
  return { name: 'boot', cols, rows, wrap: true, pos, pins, pinAt, body, stiffScale, frames: Math.round(3 * RUN_CYCLE * 60), warm: Math.round(RUN_CYCLE * 60), fps: 60, wind: [0, 0, -RUN_SPEED], camera: { at: [3, 12, -2], from: [70, 14, 10], size: 30 } };
}

// the tasset: a leather skirt panel hung from a belt round her hips, from her left side round the front to her
// right (200 degrees), 23 cm deep, over her pelvis and thighs
export function tasset(cols = 25, rows = 11) {
  const P0 = runPose(0), pel = P0.Pelvis;
  const beltY = 4.2, ax = 7.6, az = 5.6, depth = 0.23 * U, span = 200 * Math.PI / 180;
  const pos = new Float64Array(3 * cols * rows), loc = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const a = -span / 2 + span * c / (cols - 1), fl = 1 + 0.18 * r / (rows - 1);
    // (round her front: +z; c runs from her left side to her right)
    const l = [Math.sin(a) * ax * fl, beltY - depth * r / (rows - 1), Math.cos(a) * az * fl];
    loc.push(l);
    pos.set(add(pel.p, qrot(pel.q, l)), 3 * (r * cols + c));
  }
  const pins = [];
  for (let c = 0; c < cols; c++) pins.push(c);
  const pinAt = (t) => { const pl = runPose(t).Pelvis, out = new Float64Array(3 * cols); for (let c = 0; c < cols; c++) out.set(add(pl.p, qrot(pl.q, loc[c])), 3 * c); return out; };
  const body = (t) => {
    const P = runPose(t), pl = P.Pelvis, L = legPoints(t, 'L'), R = legPoints(t, 'R');
    return [
      new Capsule(add(pl.p, qrot(pl.q, [-3.4, 1.2, -0.3])), add(pl.p, qrot(pl.q, [3.4, 1.2, -0.3])), 4.6, 4.6),
      new Capsule(L.hip, L.knee, 3.3, 2.3), new Capsule(R.hip, R.knee, 3.3, 2.3),
    ];
  };
  return { name: 'tasset', cols, rows, pos, pins, pinAt, body, frames: Math.round(3 * RUN_CYCLE * 60), warm: Math.round(RUN_CYCLE * 60), fps: 60, wind: [0, 0, -RUN_SPEED], camera: { at: [0, 26, 0], from: [70, 30, 50], size: 34 } };
}

// an authored fold: a wool panel 50 x 30 cm held along its top edge, with a crease pressed down its middle
// (a box pleat's edge) that should hold at 70 degrees while the rest hangs as it will
export function fold(cols = 25, rows = 15) {
  const w = 0.5 * U, hgt = 0.3 * U, d = w / (cols - 1);
  const pos = grid(cols, rows, (c, r) => [(c - (cols - 1) / 2) * d, 50 - r * (hgt / (rows - 1)), 0]);
  const mid = (cols - 1) / 2, line = [];
  for (let r = 0; r < rows; r++) line.push(r * cols + mid);
  const pins = [];
  for (let c = 0; c < cols; c++) pins.push(c);
  return {
    name: 'fold', cols, rows, pos, pins, body: () => [], frames: 180, fps: 60, wind: [0, 0, 0],
    creases: [{ line, angle: 70 * Math.PI / 180, set: false }],
    camera: { at: [0, 42, 0], from: [60, 50, 90], size: 36 },
  };
}

export const SCENES = { hang, drape, cantilever, boot, tasset, fold };
