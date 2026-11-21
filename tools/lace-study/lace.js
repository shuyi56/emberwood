// Lace and pleating study: a lace material, flat strips cut for each kind of pleat or gather, folded, stitched
// and hung on the project's fabric solver (fabric-core.js, fabric-stiff.js: copied unchanged from the fabric
// bench branch, claude/project-thread-e9qu6c, so the study runs on the game's own solver).
//
// Everything here is in centimetres (U = 100 units a metre), y up. It runs in node (study.mjs) and in the
// browser viewer (viewer/build.mjs inlines it), so it knows nothing about files or pages.
//
// How a strip is made, as a dressmaker makes it:
//  1. cut: a flat strip, u along its length (the cut length, ratio x the finished length) and v down its depth,
//     from the stitched edge (v = 0) to the scalloped lace edge (v = depth). The columns of points run down the
//     strip at the fold lines, so every fold falls on a row of edges;
//  2. fold: the pleat's cross-section, as a list of turns along u (pleatProfile): pressed folds turn at one
//     column, rolled ones (flutes) over the columns of their arc. The strip is laid in that shape;
//  3. stitch: the top row is held where the stitching holds it (on a band, round a cuff, down a jabot's line);
//  4. press: the lace's rest curvature becomes `press` times the curvature it was folded to (1 a hard press or a
//     heat set, 0 only stitched at the top);
//  5. hang: gravity, the lace's weight and its bending stiffness (from its bending length) do the rest.

import { Fabric } from './fabric-core.js';
import * as FIG from './figure.js';

export const U = 100;                      // units a metre: centimetres

// The solver's settings for the study: small steps (Macklin et al. 2019, "Small steps in physics simulation"), two
// walks a step. Lace bends at hinges (Bridson et al. 2003, as XPBD), and a hinge's hold only reaches lace's bending
// length when the walks converge: at 240 steps a second and 4 walks even rigid hinges stop at 0.6 cm on a 3 mm
// mesh; at 960 and 2 they reach 1.5 cm (calibrate.mjs).
export const SOLVER = { substep: 1 / 960, iterations: 2 };
const DEG = Math.PI / 180;

// ------------------------------------------------------------------ the lace
//
// Fine cotton or silk lace of the 1870s to 1890s, the kind engageantes, jabots and collar frills were made of: a
// net ground with denser motifs outlined in a heavier thread (the cordonnet) and a scalloped edge. The numbers and
// where they come from (see docs/clothing-solvers/lace-and-pleating.md):
//  - weight 50 g/m2: Chantilly 45 to 65 g/m2 (TextilePulse), 40 to 90 (Fabric Atlas), 30 to 180 (Tissura);
//  - thickness 0.3 mm: Chantilly 0.2 to 0.5 mm (Fabric Atlas);
//  - bending length 1.0 cm (cantilever, Peirce 1930; ASTM D1388 B = W g c^3): no measured value for lace was found.
//    Ordinary fabrics run 1.6 to 6.35 cm (Peirce's flexometer, in Sanad and Cassidy 2016), and the project's silk
//    is 1.2 cm, measured on the bench; a net ground is limper than the silk it is made of, so 1.0 is an estimate
//    (marked experimental) to be swept, not a measurement. Starched lace (period laundering starched lace frills
//    before fluting: Every Woman's Encyclopaedia, Saffron Walden Museum) is the `starched` variant, 2.5 cm;
//  - the air's drag on a sheet (cd 1.2) times the lace's fill: the net is about half holes.
export const LACE = {
  cotton:   { weight: 0.050, thickness: 0.0003, bendingLength: 0.010, fill: 0.55, shear: 4e-5, hinge: 3e-1, damping: 0.03, friction: 0.3 },
  silk:     { weight: 0.040, thickness: 0.00025, bendingLength: 0.009, fill: 0.5, shear: 4e-5, hinge: 3e-1, damping: 0.03, friction: 0.25 },
  starched: { weight: 0.058, thickness: 0.0003, bendingLength: 0.025, fill: 0.55, shear: 2e-5, hinge: 4e-2, damping: 0.04, friction: 0.4 },
  // not lace: a skirt-weight worsted wool for the pleated skirt (plain cloth, no net, no scallop). Worsted suiting
  // and skirting run about 250 to 300 g/m2 and 0.4 to 0.6 mm; its bending length is set between the project's silk
  // (1.2 cm, measured) and its coat wool (4 cm, fabric-core.js), inside the Peirce flexometer's 1.6 to 6.35 cm
  // range for ordinary fabrics (Sanad and Cassidy 2016): an estimate, like the lace's
  wool:     { weight: 0.28, thickness: 0.0005, bendingLength: 0.025, fill: 1.0, shear: 1e-5, hinge: 1e-2, damping: 0.03, friction: 0.5, plain: true },
};

// The pattern of the lace on the flat strip: 0 net, 1 motif, 2 cordonnet (the outline thread and the scalloped
// edge). The motif repeats with the scallop: a ring of six petals over each scallop, a line of cordonnet just
// inside the edge, and a narrow stitched heading at the top (a plain tape the gathering or the band is sewn
// through). Each part weighs and bends by its own share (partWeight, partStiff), kept so the whole strip's mean is
// the material's, so a cantilever of the whole lace still gives its bending length.
export const PART = { net: 0, motif: 1, cord: 2, heading: 3 };
export const partWeight = [0.7, 1.35, 1.8, 1.4];
export const partStiff = [0.6, 1.5, 3.0, 2.0];

export function scallopEdge(u, depth, scW, scD) {
  // the edge's v at u: arcs a scallop wide, scD deep, bulging out (down)
  const t = ((u / scW) % 1 + 1) % 1;
  return depth - scD + scD * Math.sin(Math.PI * t);
}

export function lacePart(u, v, depth, scW, scD) {
  const edge = scallopEdge(u, depth, scW, scD);
  if (v < 0.35) return PART.heading;
  if (edge - v < 0.18) return PART.cord;
  // the motif: a flower centred over each scallop, its ring of petals outlined
  const cu = (Math.floor(u / scW) + 0.5) * scW, cv = depth - scD - 0.55 * Math.min(depth * 0.45, scW * 0.7);
  const du = u - cu, dv = v - cv, r = Math.hypot(du, dv), a = Math.atan2(dv, du);
  const R = 0.42 * scW * (0.8 + 0.2 * Math.cos(6 * a));
  if (Math.abs(r - R) < 0.08 * scW) return PART.cord;
  if (r < R) return r < 0.12 * scW ? PART.cord : PART.motif;
  // a band of small leaves between the heading and the flowers
  const lv = 0.35 + 0.18 * depth, lu = ((u / (scW / 2)) % 1 + 1) % 1;
  if (Math.abs(v - lv) < 0.12 * depth * Math.sin(Math.PI * lu)) return PART.motif;
  return PART.net;
}

// ------------------------------------------------------------------ the pleats
//
// A pleat's cross-section as the turns along the cut strip. From the pleating references:
//  - knife: per repeat, the face, folded under by the depth, and out again: cut = pitch + 2 x depth, so a pleat
//    whose depth is its pitch takes 3 x its finished length (Wikipedia "Pleat": 3:1; Minerva Patterns: "twice the
//    pleat depth" behind each fold; Sewing With Numbers: B = 3A edge to edge);
//  - box: two knife pleats turned away from each other: cut = pitch + 4 x depth, 3 x when the underlays meet
//    (depth = pitch / 2: Wikipedia, same 3:1 as knife);
//  - accordion: a zigzag of equal faces, each turn the same angle the other way: ratio = 1 / cos(turn / 2); 3 x at
//    141 degrees (Wikipedia: a tighter variation of the knife pleat);
//  - flute (goffered): rolled, not pressed: arcs of `arc` degrees of radius r, turning each way in turn (a fluting
//    iron's rollers: Saffron Walden Museum, Quintessential Clothes Pen): ratio = (arc / 2) / sin(arc / 2),
//    1.57 x for half circles;
//  - gather: no folds pressed: the heading is drawn up on its thread to 1 / ratio of the cut, in a small wave a
//    gathering stitch long; period ruffles 1.25 to 1.75 x (Rosbrugh, historicalsewing.com), sheers up to 2.5 x,
//    modern frills 1.5 to 3 x (The Shapes of Fabric). The lace then finds its own folds below the stitching.
// Pressed folds turn by 175 degrees, not 180: the cloth's thickness keeps the layers apart (blender/shared/folds.py).
export const PLEATS = ['knife', 'box', 'accordion', 'flute', 'gather'];

// Lay out the columns: returns { du (column spacing), cols, turn (per column, radians: the heading turns there),
// ratio (cut / finished), finished, cut, crease (per column: true where a fold is pressed) }.
const accTurn = (p) => 2 * Math.acos(Math.min(0.999, 1 / Math.max(1.0001, p.ratio || 2.5)));

export function pleatProfile(p) {
  const kind = p.kind, F = p.finished;
  const step = p.res || 0.3;
  let unit, seq = null;                              // seq: [{ n: columns straight, turn, arc }] for one repeat
  const pressed = (p.pressAngle ?? 175) * DEG;
  let rep = null, segStart = null, h0 = 0;
  if (kind === 'knife' || kind === 'box') {
    const D = p.depth, P = p.pitch;
    unit = step;
    // (a box repeat begins at the middle of a face, so it ends on a straight run and no fold wraps round it)
    const segs = kind === 'knife'
      ? [{ L: P, turn: -pressed }, { L: D, turn: +pressed }, { L: D, turn: 0 }]
      : [{ L: P / 2, turn: -pressed }, { L: D, turn: +pressed }, { L: 2 * D, turn: +pressed }, { L: D, turn: -pressed }, { L: P / 2, turn: 0 }];
    ({ rep, segStart } = pressedColumns(segs, step, Math.min(step, p.bridge ?? step)));
  } else if (kind === 'accordion') {
    // each face of the zigzag is a = pitch / 2; the repeat starts mid-face so no fold wraps round it, and its
    // folds turn over two columns as the knife's do (at 132 degrees, ratio 2.5, a fold turned at one column tore
    // the strip 35% in the first run)
    const a = p.pitch / 2;
    unit = step;
    // The bridge is a flat run at each fold, so the turn that gives the ratio is wider than 2 acos(1 / ratio): found
    // by bisection on the laid-out repeat
    const build = (T) => { h0 = -T / 2; return pressedColumns([{ L: a / 2, turn: +T }, { L: a, turn: -T }, { L: a / 2, turn: 0 }], step, step, 90 * DEG).rep; };
    const ratioAt = (T) => { const r = build(T); let x = 0, h = -T / 2, L = 0; for (const c of r) { x += c.w * Math.cos(h); L += c.w; h += c.t; } return L / x; };
    const want = Math.max(1.05, p.ratio || 2.5);
    let lo = accTurn({ ratio: 1.0001 }), hi = 176 * DEG;
    for (let it = 0; it < 50; it++) { const mid = (lo + hi) / 2; if (ratioAt(mid) < want) lo = mid; else hi = mid; }
    rep = build((lo + hi) / 2);
  } else if (kind === 'flute') {
    const r = p.radius, arc = (p.arc ?? 180) * DEG, len = r * arc;
    const n = Math.max(3, Math.round(len / step));
    unit = len / n;
    seq = [{ n, arc: +arc }, { n, arc: -arc }];
  } else {
    // gather: a sine wave a stitch long (each stitch takes up one fold) at the ratio, laid out to be found below
    const n = Math.max(p.gatherCols ?? 4, Math.round(p.stitch / step));
    unit = p.stitch / n;
    seq = [{ n, wave: true }];
  }
  // one repeat's columns, then as many repeats as fill the finished length
  if (!rep) rep = [];
  for (const s of seq || []) {
    for (let k = 0; k < s.n; k++) {
      let t = 0, crease = false;
      if (s.arc) { t = s.arc / s.n; crease = true; }
      else if (k === s.n - 1 && s.turn) { t = s.turn; crease = true; }
      rep.push({ t, crease, wave: !!s.wave, arc: !!s.arc });
    }
  }
  // the finished length of one repeat, from the shape laid out (the turns integrated), so the ratio is measured
  // on the folded strip, not assumed
  let tilt = 0;
  const lay = (cols) => {
    const xs = [[0, 0]];
    let h = kind === 'accordion' ? h0 : kind === 'flute' ? -seq[0].arc / 2 : tilt;
    for (let c = 0; c < cols.length; c++) {
      const [x, z] = xs[xs.length - 1];
      const w = cols[c].w ?? unit;
      xs.push([x + w * Math.cos(h), z + w * Math.sin(h)]);
      h += cols[c].t;
    }
    return xs;
  };
  if (kind === 'knife') {
    // a knife pleat's three layers lie as tiles (Minerva Patterns: "each fold laid over the next like tiles"): the
    // under layer becomes the next pleat's face without a turn, so every layer leans by the same small angle, the
    // one at which a repeat ends where it began (no drift out of the band's line). Found by bisection.
    const drift = (phi) => { tilt = phi; const xs = lay(rep); return xs[xs.length - 1][1] - xs[0][1]; };
    let lo = -0.6, hi = 0.6;
    const dlo = drift(lo);
    for (let it = 0; it < 60; it++) { const mid = (lo + hi) / 2, dm = drift(mid); if ((dm > 0) === (dlo > 0)) lo = mid; else hi = mid; }
    tilt = (lo + hi) / 2;
  }
  let finishedRep, turns;
  if (kind === 'gather') {
    const R = Math.max(1.05, p.ratio || 1.5);
    finishedRep = p.stitch / R;
    turns = rep;
  } else {
    const xs = lay(rep);
    finishedRep = xs[xs.length - 1][0] - xs[0][0];
    turns = rep;
  }
  const reps = Math.max(1, Math.round(F / finishedRep));
  const cols = [];
  for (let k = 0; k < reps; k++) cols.push(...turns.map((c) => ({ ...c })));
  cols.push({ t: 0, crease: false, wave: kind === 'gather' });
  // the points of the top edge, laid in the pleat's shape: x along the finished length, z out of it
  let xz;
  if (kind === 'gather') {
    // a sine of the stitch's pitch, its amplitude found so each wave's arc is the stitch
    const R = Math.max(1.05, p.ratio || 1.5), pitch = finishedRep, n = turns.length;
    let A = 0;
    const arcLen = (amp) => { let L = 0; for (let k = 0; k < 64; k++) { const a = 2 * Math.PI * k / 64, b = 2 * Math.PI * (k + 1) / 64; L += Math.hypot(pitch / 64, amp * (Math.sin(b) - Math.sin(a))); } return L; };
    let lo = 0, hi = p.stitch;
    for (let it = 0; it < 50; it++) { A = (lo + hi) / 2; if (arcLen(A) < p.stitch) lo = A; else hi = A; }
    // points spaced evenly along the arc (by arc length), so each edge keeps its cut length
    const tab = [[0, 0, 0]];
    for (let k = 1; k <= 512; k++) { const s = k / 512, ps = tab[k - 1]; const x = s * pitch, z = A * Math.sin(2 * Math.PI * s); tab.push([x, z, ps[2] + Math.hypot(x - ps[0], z - ps[1])]); }
    const total = tab[tab.length - 1][2];
    const at = (s) => { const L = s * total; let k = 1; while (k < tab.length - 1 && tab[k][2] < L) k++; const a = tab[k - 1], b = tab[k], f = (L - a[2]) / ((b[2] - a[2]) || 1); return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]; };
    xz = [];
    for (let c = 0; c < cols.length; c++) { const q = Math.floor(c / n), s = (c % n) / n; const [x, z] = at(s); xz.push([q * pitch + x, z]); }
    void R;
  } else {
    xz = lay(cols.slice(0, -1));
  }
  // (centred on the line the stitching runs along)
  {
    let m = 0; for (const q of xz) m += q[1]; m /= xz.length;
    for (const q of xz) q[1] -= m;
  }
  const finished = xz[xz.length - 1][0] - xz[0][0];
  // each column's place along the cut strip (the fold's bridges are narrower than the rest)
  const uAt = [0];
  for (let c = 0; c < cols.length - 1; c++) uAt.push(uAt[c] + (cols[c].w ?? unit));
  const cut = uAt[uAt.length - 1];
  return { du: unit, uAt, cols: cols.length, turn: cols.map((c) => c.t), crease: cols.map((c) => c.crease && !c.second && !c.arc), split: cols.map((c) => !!c.split && !c.second), xz, finished, cut, ratio: cut / finished, repeats: reps, repeatCols: turns.length };
}

// A pressed pleat's columns from its segments ({ L: length, turn at its end }). A pressed fold turns over two
// columns, half its angle at each, with a column between (`bridge`): turned at one column, a fold of 175 degrees
// folds a point's ring of neighbours in half, and the bend's rotation of it (fabric-stiff.js) has no answer there
// (in the trials the strip tore itself 12 to 18%). The bridge is the fold's round edge, and its width is the gap
// between the pleat's layers. It is a full column wide by default: at 1 mm (a pressed lace fold is nearer 0.3 mm)
// the cotangent weights of so thin a column made a pressed knife strip creep to 14% stretch; at 2 mm, 4 to 6%; at a
// full 3 mm column it rests at 0.6%. The price is the gap: the knife's layers lean (pleatProfile) to take it up.
// Returns the columns ({ w, t, crease, split, second }) and where each
// segment starts.
function pressedColumns(segs, step, bridge, splitAbove = 150 * DEG) {
  const rep = [], segStart = [];
  let carry = 0;
  for (const s of segs) {
    segStart.push(rep.length);
    let L = s.L;
    if (carry) { rep.push({ w: bridge, t: carry, split: true, second: true }); L -= bridge; carry = 0; }
    const k = Math.max(1, Math.round(L / step)), w = L / k;
    for (let j = 0; j < k; j++) rep.push({ w, t: 0 });
    const last = rep[rep.length - 1];
    if (s.turn) {
      last.crease = true;
      if (Math.abs(s.turn) > splitAbove) { last.t += s.turn / 2; last.split = true; carry = s.turn / 2; }
      else last.t += s.turn;
    }
  }
  return { rep, segStart };
}

// ------------------------------------------------------------------ the strip, built and hung
//
// p: { kind, finished (cm), depth (cm, the frill's depth), pitch, depth of pleat (pleatDepth), radius, arc,
//      ratio, stitch, res (column spacing aimed at), rows, mount ('band' | 'cuff' | 'collar' | 'jabot'),
//      lace ('cotton' | 'silk' | 'starched'), bendingLength (cm, overrides), weight (kg/m2, overrides),
//      press (0..1), creaseCompliance, gravity (cm/s2), scallop (cm wide), scallopDepth, iterations, substep }
export const DEFAULTS = {
  kind: 'knife', finished: 24, depth: 7, pitch: 1.2, pleatDepth: 1.2, radius: 0.35, arc: 180, ratio: 2.5,
  stitch: 2.0, res: 0.3, rows: 18, mount: 'band', lace: 'cotton', press: 0.6, creaseCompliance: 1e-3,
  gravity: 981, scallop: 2.4, scallopDepth: 0.45, iterations: 4, substep: 1 / 240, damping: null, cuffRadius: 4.8,
  // the bend's solve holds the threads' lengths too (Projective Dynamics' springs, stretchK) in `passes` rounds: with
  // the threads left to the constraints alone a hanging lace strip stretched 13% (mean 2%); with 300 and 4, 3%
  // (0.3%). The fabric bench left this off for open pieces on a body (contact fights it); these strips hang free.
  stretchK: 300, passes: 4,
};

export function buildStrip(over = {}) {
  const p = { ...DEFAULTS, ...over };
  const mat = { ...LACE[p.lace] };
  if (p.bendingLength != null) mat.bendingLength = p.bendingLength / 100;
  if (p.weight != null) mat.weight = p.weight;
  const kind = p.kind;
  if (onFigure(p)) {
    // a skirt on the figure: its finished length is the yoke's round on her at yokeZ, offset so the pleats' inner
    // layer clears her by 0.3 cm (the pleats' depth found from a first layout)
    const pr0 = pleatProfile({ ...p, finished: 100, depth: p.pleatDepth });
    const inner = Math.min(...pr0.xz.map((q) => q[1]));
    p.yokeOff = 0.3 - inner;
    p.finished = closedLength(FIG.bandRing(p.yokeZ, p.yokeOff / FIG.CM));
    p.cuffRadius = p.finished / (2 * Math.PI);
  }
  const prof = pleatProfile({ ...p, depth: p.pleatDepth });
  const cols = prof.cols, rows = p.rows, du = prof.du;
  // the cut: column c at u = c du, its rows from the heading to the scallop at that u (so the scallop is the edge
  // exactly), indexed column by column so the bend's banded solve stays narrow
  const idx = (c, r) => c * rows + r;
  const n = cols * rows, flat = new Float64Array(2 * n), part = new Uint8Array(n);
  for (let c = 0; c < cols; c++) {
    const u = prof.uAt[c], edge = mat.plain ? p.depth : scallopEdge(u, p.depth, p.scallop, p.scallopDepth);
    for (let r = 0; r < rows; r++) {
      const v = edge * r / (rows - 1), i = idx(c, r);
      flat[2 * i] = u; flat[2 * i + 1] = v;
      part[i] = mat.plain ? (v < (p.headingWidth ?? 0.35) ? PART.heading : PART.net) : lacePart(u, v, p.depth, p.scallop, p.scallopDepth);
    }
  }
  // the flat sheet first: the threads' lengths, the hinges and the bend's rest are the pattern's, as cut
  const pos = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) { pos[3 * i] = flat[2 * i]; pos[3 * i + 1] = -flat[2 * i + 1]; pos[3 * i + 2] = 0; }
  const g = p.gravity;
  const settings = {
    gravity: [0, -g, 0], damping: p.damping ?? mat.damping, iterations: p.iterations, substep: p.substep,
    shear: mat.shear, friction: mat.friction, stretch: 0, order: 'root',
    air: 0.5 * 1.2 * 1.2 * mat.fill / (U * U * U),
  };
  const f = new Fabric(pos, settings);
  const pins = [];
  // the stitching: the heading's top row (`heldRows` holds more; two rows for a gather, as gathers are drawn up on
  // two rows of running stitches, made no difference to the stretch in the trials: 4.6 against 4.3%)
  const held = p.heldRows ?? 1;
  for (let c = 0; c < cols; c++) for (let r = 0; r < Math.min(held, rows - 1); r++) pins.push(idx(c, r));
  for (const i of pins) f.pin(i);
  // the sheet: threads that keep their length and diagonals that give a little on the bias; no ties two along and
  // no hinges: the bend is the implicit one of fabric-stiff.js, at the lace's flexural rigidity from its bending
  // length (B = W g c^3). (Hinges were tried first and cap out below lace's bending length on a fine mesh, and their
  // stiffness changes with the mesh: calibrate.mjs, the study's notes.)
  f.sheet(cols, rows, idx, { bend: null, hinge: null, shear: mat.shear });
  // weight: the material's, each part by its share (normalised so the strip's mean is the material's)
  const density = mat.weight / (U * U);
  const meanW = mean(Array.from(part, (k) => partWeight[k])), meanS = mean(Array.from(part, (k) => partStiff[k]));
  f._areas();
  const kg = new Float64Array(n);
  for (let i = 0; i < n; i++) kg[i] = density * f.area[i] * partWeight[part[i]] / meanW;
  f.massKg(kg);
  const scale = Float64Array.from(part, (k) => partStiff[k] / meanS);
  const B = f.stiffen({ bendingLength: mat.bendingLength * U, density, scale, passes: p.passes ?? 1, stretchK: p.stretchK ?? 0 });
  f._pack();
  // fold: the strip laid in its pleat's shape, the top edge where the stitching holds it (mount), each row below
  // it hanging straight down from it at its cut depth
  const place = mountFn(p, prof);
  for (let c = 0; c < cols; c++) for (let r = 0; r < rows; r++) {
    const i = idx(c, r), P = place(prof.xz[c][0], prof.xz[c][1], flat[2 * i + 1]);
    f.pos[3 * i] = P[0]; f.pos[3 * i + 1] = P[1]; f.pos[3 * i + 2] = P[2];
  }
  f.prev.set(f.pos);
  f.stiff._h = null;
  // press: a pleat pressed (or a flute set with a hot iron, starch and damp) keeps the shape it was pressed in as
  // its rest, so the lace's rest curvature becomes `press` times the curvature it has folded (in each point's rest
  // frame: R^T L x, as the solver's own set does for leather). 1 a hard press that holds, 0 only stitched at the top.
  // A gather is never pressed: its folds are the lace's own.
  setPress(f, kind === 'gather' ? 0 : p.press);
  const creaseCols = [];
  for (let c = 0; c < cols - 3; c++) if (prof.crease[c]) creaseCols.push(c + 1);
  const span = (c) => (prof.split[c - 1] ? 2 : 1);
  const folded = creaseCols.map((c) => foldAngle(f, idx, c, 0, span(c)));
  const creases = creaseCols.map((c, k) => ({ col: c, folded: folded[k] }));
  f.tether();
  if (p.mount === 'skirt' && p.hipRadius == null && !onFigure(p)) {
    // (the hips just inside the pleats' inner layer at the yoke: a body cutting into the pleats as hung tears them)
    let rin = Infinity;
    for (let c = 0; c < cols; c++) { const i = idx(c, 0); rin = Math.min(rin, Math.hypot(f.pos[3 * i], f.pos[3 * i + 2])); }
    p.hipRadius = rin - 0.3;
  }
  return { f, p, mat, prof, cols, rows, idx, flat, part, pins, creases, creaseCols, span, B, body: bodyFn(p) };
}

// set the bend's rest curvature to `press` times the cloth's curvature as it lies now (see buildStrip)
export function setPress(f, press) {
  const st = f.stiff, n = f.n;
  for (let k = 0; k < 8; k++) st._rotations();            // (the turns found from rest, from scratch: a few rounds)
  const Lx = st._lap(f.pos), R = st._R;
  if (!st._b0flat) st._b0flat = Float64Array.from(st.b0);
  for (let i = 0; i < n; i++) {
    const r = 9 * i, a = 3 * i;
    const lx = R[r] * Lx[a] + R[r + 3] * Lx[a + 1] + R[r + 6] * Lx[a + 2], ly = R[r + 1] * Lx[a] + R[r + 4] * Lx[a + 1] + R[r + 7] * Lx[a + 2], lz = R[r + 2] * Lx[a] + R[r + 5] * Lx[a + 1] + R[r + 8] * Lx[a + 2];
    st.b0[a] = st._b0flat[a] + press * (lx - st._b0flat[a]); st.b0[a + 1] = st._b0flat[a + 1] + press * (ly - st._b0flat[a + 1]); st.b0[a + 2] = st._b0flat[a + 2] + press * (lz - st._b0flat[a + 2]);
  }
  st._pressedLx = { Lx, R: Float64Array.from(R) };
}

// a column's fold angle at a row (degrees): how far the strip turns there, across the strip, between the columns
// either side (0 flat, 180 folded back on itself)
export function foldAngle(f, idx, c, r, span = 1) {
  // the turn between the segment into column c and the segment out of column c + span - 1 (span 2: a pressed
  // fold turned over two columns), measured across the strip (less any part along the column's own line)
  const x = f.pos, p = (cc, rr) => { const i = 3 * idx(cc, rr); return [x[i], x[i + 1], x[i + 2]]; };
  const A = p(c - 1, r), Bq = p(c, r), C = p(c + span - 1, r), D = p(c + span, r), E = p(c, r + 1);
  const e = sub(E, Bq), el = Math.hypot(...e) || 1;
  const across = (q) => { const t = dot(q, e) / (el * el); return [q[0] - t * e[0], q[1] - t * e[1], q[2] - t * e[2]]; };
  const uu = across(sub(Bq, A)), vv = across(sub(D, C));
  const cs = dot(uu, vv) / ((Math.hypot(...uu) * Math.hypot(...vv)) || 1);
  return Math.acos(Math.max(-1, Math.min(1, cs))) / DEG;
}

function mean(a) { let s = 0; for (const v of a) s += v; return s / (a.length || 1); }

// Where the stitching holds the top edge, and how the strip hangs from it as laid out: (x, z) the edge's point in
// its pleat's shape, v how far down the strip
export const onFigure = (p) => p.mount === 'skirt' && p.body === 'figure';
const closedLength = (C) => { let L = 0; for (let k = 0; k < C.length; k++) { const a = C[k], b = C[(k + 1) % C.length]; L += Math.hypot(b[0] - a[0], b[1] - a[1]); } return L; };

// A yoke laid on the figure: the pleats' top edge round her ring at yokeZ. Each column steps along the ring by its
// run over (1 + its offset x the ring's curvature there), so the threads keep their cut lengths at every layer, as
// the round mount does; the ring is scaled about her centre line until the edge closes.
export function figureYoke(p, prof) {
  // (the ring starts at her centre back, where the skirt's seam goes)
  const C0 = FIG.bandRing(p.yokeZ, p.yokeOff / FIG.CM, 512, Math.PI), n = C0.length, xz = prof.xz;
  const lay = (g) => {
    const C = C0.map((q) => [q[0] * g, q[1] * g]);
    const S = [0]; for (let k = 1; k <= n; k++) { const a = C[k - 1], b = C[k % n]; S.push(S[k - 1] + Math.hypot(b[0] - a[0], b[1] - a[1])); }
    const L = S[n];
    let turn = 0;
    for (let k = 0; k < n; k++) { const a = C[k], b = C[(k + 1) % n], c = C[(k + 2) % n]; const u = Math.atan2(b[1] - a[1], b[0] - a[0]), w = Math.atan2(c[1] - b[1], c[0] - b[0]); turn += Math.atan2(Math.sin(w - u), Math.cos(w - u)); }
    const sigma = Math.sign(turn);
    const at = (s) => {
      s = ((s % L) + L) % L;
      let k = 0; while (k < n - 1 && S[k + 1] < s) k++;
      const a = C[k], b = C[(k + 1) % n], t = (s - S[k]) / ((S[k + 1] - S[k]) || 1);
      const P = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
      const tx = (b[0] - a[0]) / ((S[k + 1] - S[k]) || 1), tz = (b[1] - a[1]) / ((S[k + 1] - S[k]) || 1);
      // (the ring runs from her front toward her right, which is clockwise from above: outward is the tangent's left)
      let nx = -tz, nz = tx; if (nx * P[0] + nz * P[1] < 0) { nx = -nx; nz = -nz; }
      // (the curvature: the tangent's turn from the segment before to the one after, over the two segments' run)
      const c = C[(k + n - 1) % n], d = C[(k + 2) % n];
      const t0 = Math.atan2(a[1] - c[1], a[0] - c[0]), t2 = Math.atan2(d[1] - b[1], d[0] - b[0]);
      const run = 0.5 * Math.hypot(a[0] - c[0], a[1] - c[1]) + Math.hypot(b[0] - a[0], b[1] - a[1]) + 0.5 * Math.hypot(d[0] - b[0], d[1] - b[1]);
      // (signed: positive where she is convex, negative in the hollow between the seat's two forms)
      const kap = sigma * Math.atan2(Math.sin(t2 - t0), Math.cos(t2 - t0)) / run;
      return { P, N: [nx, nz], kap };
    };
    // each column placed so its step from the last is exactly the cut's: a first guess from the curvature, then
    // the secant method on the step's length (her ring's curvature changes too fast at the sides for the guess alone:
    // it left threads 3% out at the yoke, held there by the stitching)
    const ptAt = (s, z) => { const A = at(s); return [A.P[0] + A.N[0] * z, A.P[1] + A.N[1] * z]; };
    const s = [0];
    for (let c = 1; c < xz.length; c++) {
      const zm = 0.5 * (xz[c][1] + xz[c - 1][1]), dx = xz[c][0] - xz[c - 1][0];
      const d0 = Math.hypot(dx, xz[c][1] - xz[c - 1][1]), Q = ptAt(s[c - 1], xz[c - 1][1]);
      const err = (sv) => { const R = ptAt(sv, xz[c][1]); return Math.hypot(R[0] - Q[0], R[1] - Q[1]) - d0; };
      let a0 = s[c - 1] + dx / (1 + zm * at(s[c - 1]).kap), a1 = a0 + 0.01 * Math.sign(dx || 1), e0 = err(a0), e1 = err(a1);
      for (let it = 0; it < 8 && Math.abs(e1) > 1e-6 && e1 !== e0; it++) { const a2 = a1 - e1 * (a1 - a0) / (e1 - e0); a0 = a1; e0 = e1; a1 = a2; e1 = err(a1); }
      s.push(Math.abs(e1) < Math.abs(e0) ? a1 : a0);
    }
    return { s, L, at };
  };
  let lo = 0.8, hi = 1.25;
  for (let it = 0; it < 40; it++) { const g = (lo + hi) / 2, r = lay(g); if (r.s[r.s.length - 1] > r.L) lo = g; else hi = g; }
  const { s, at } = lay((lo + hi) / 2);
  let col = 0;
  return (x, z, v) => {
    while (col < xz.length - 1 && xz[col][0] !== x) col++;
    if (xz[col][0] !== x) col = xz.findIndex((q) => q[0] === x);
    const A = at(s[Math.max(0, col)]);
    return [A.P[0] + A.N[0] * z, -v, A.P[1] + A.N[1] * z];
  };
}

export const isRound = (p) => p.mount === 'cuff' || p.mount === 'collar' || p.mount === 'skirt';

export function mountFn(p, prof) {
  const F = prof.finished;
  if (onFigure(p)) return figureYoke(p, prof);
  if (isRound(p)) {
    // round a circle the finished length round (an engageante sewn into a sleeve's end; a frill on a neckband),
    // z outward; the cuff hangs down, the collar stands up
    // (each step along the edge turns the angle by its run over the radius it lies at, so the threads keep their
    // cut lengths at every radius; the heading's radius is found so the edge closes round once)
    // (a skirt hangs from its hip yoke as a cuff hangs from a sleeve)
    const xz = prof.xz, up = p.mount === 'collar' ? 1 : -1;
    const angles = (R) => { const a = [0]; for (let c = 1; c < xz.length; c++) a.push(a[c - 1] + (xz[c][0] - xz[c - 1][0]) / (R + 0.5 * (xz[c][1] + xz[c - 1][1]))); return a; };
    let R = F / (2 * Math.PI);
    for (let it = 0; it < 30; it++) { const a = angles(R); R *= a[a.length - 1] / (2 * Math.PI); }
    const A = angles(R), byX = new Map(xz.map((q, c) => [q, c]));
    void byX;
    let col = 0;
    return (x, z, v) => {
      // (called column by column, row by row: find this column's angle by its top point)
      while (col < xz.length - 1 && xz[col][0] !== x) col++;
      if (xz[col][0] !== x) col = xz.findIndex((q) => q[0] === x);
      const a = A[Math.max(0, col)], rr = R + z;
      return [rr * Math.cos(a), up * v, -rr * Math.sin(a)];
    };
  }
  if (p.mount === 'jabot') {
    // down the centre front: the stitched edge is a vertical line (y), the frill stands forward of it (+z) and its
    // pleats fold it from side to side (x)
    return (x, z, v) => [z, -x, v];
  }
  return (x, z, v) => [x - F / 2, -v, z];
}

// What the strip must stay out of: the forearm in a cuff (a cylinder a little inside the frill's top), the neck in a
// collar, the chest in a jabot (a plane behind the stitched line)
export function bodyFn(p) {
  if (p.mount === 'cuff') {
    const R = p.cuffRadius * 0.7;
    return (q) => { const r = Math.hypot(q[0], q[2]); if (r >= R + 0.05 || r < 1e-6) return false; const k = (R + 0.05) / r; q[0] *= k; q[2] *= k; return true; };
  }
  if (onFigure(p) && !p.noBody) return FIG.bodyFn(p.yokeZ, 0.3);
  if (p.mount === 'skirt' && !p.noBody) {
    // the hips under a skirt sewn to a hip yoke (its heading at the hip line, y 0): round just inside the pleats'
    // inner layer for the top 15 cm (hipRadius, set by buildStrip), tapering by 14% to 40 cm down (the thighs, taken
    // together), then nothing
    const R = p.hipRadius ?? 0.93 * p.cuffRadius;
    const at = (y) => (y > 2 ? 0 : y > -15 ? R : y > -40 ? R * (1 - 0.14 * (-15 - y) / 25) : 0);
    return (q) => { const rb = at(q[1]); if (!rb) return false; const r = Math.hypot(q[0], q[2]); if (r >= rb + 0.1 || r < 1e-6) return false; const k = (rb + 0.1) / r; q[0] *= k; q[2] *= k; return true; };
  }
  if (p.mount === 'jabot') {
    return (q) => { if (q[2] > -0.1 + 0.1 * 0 && q[2] >= 0) return false; if (q[2] >= 0) return false; q[2] = 0; return true; };
  }
  return null;
}

// ------------------------------------------------------------------ the solve, watched

// one frame of dt seconds; returns { steps, speed (largest point move a step, cm), strain (largest edge stretch) }
export function frame(S, dt = 1 / 60) {
  const f = S.f, before = Float64Array.from(f.pos);
  const steps = f.advance(dt, { body: S.body, neutral: true });
  let far = 0;
  for (let a = 0; a < f.pos.length; a++) far = Math.max(far, Math.abs(f.pos[a] - before[a]));
  return { steps, speed: far / Math.max(1, steps) };
}

// the edge strains: for every thread (warp, weft) of the grid, its length now over its cut length, minus one
export function strains(S) {
  const { f, cols, rows, idx, flat } = S, x = f.pos, out = new Float64Array(f.n);
  let max = 0, sum = 0, cnt = 0;
  const all = [];
  const edge = (i, j) => {
    if (f.w[i] === 0 && f.w[j] === 0) return;          // (a thread between two held points is the stitching's)
    const r = Math.hypot(flat[2 * i] - flat[2 * j], flat[2 * i + 1] - flat[2 * j + 1]);
    const l = Math.hypot(x[3 * i] - x[3 * j], x[3 * i + 1] - x[3 * j + 1], x[3 * i + 2] - x[3 * j + 2]);
    const e = Math.abs(l / r - 1);
    out[i] = Math.max(out[i], e); out[j] = Math.max(out[j], e);
    max = Math.max(max, e); sum += e; cnt++; all.push(e);
  };
  for (let c = 0; c < cols; c++) for (let r = 0; r < rows; r++) {
    if (c + 1 < cols) edge(idx(c, r), idx(c + 1, r));
    if (r + 1 < rows) edge(idx(c, r), idx(c, r + 1));
  }
  all.sort((a, b) => a - b);
  return { perPoint: out, max, mean: sum / (cnt || 1), p99: all[Math.floor(0.99 * (all.length - 1))] || 0 };
}

// the energies (relative; for the convergence curve): kinetic from the last step's moves, and the height of the
// cloth's centre of mass under the stitching
export function energies(S) {
  const f = S.f, h = f.settings.substep, kg = f.kg;
  let ke = 0, pe = 0, m = 0;
  for (let i = 0; i < f.n; i++) {
    const mi = kg ? kg[i] : 1;
    const vx = (f.pos[3 * i] - f.prev[3 * i]) / h, vy = (f.pos[3 * i + 1] - f.prev[3 * i + 1]) / h, vz = (f.pos[3 * i + 2] - f.prev[3 * i + 2]) / h;
    ke += 0.5 * mi * (vx * vx + vy * vy + vz * vz) / 1e4;       // J (cm to m)
    pe += mi * f.pos[3 * i + 1] / 100 * 9.81;
    m += mi;
  }
  return { kinetic: ke, potential: pe, mass: m };
}

// how far each crease is from the angle it was folded to, top and bottom (degrees): a pressed pleat holds near the
// stitching and opens below as much as the press lets it
export function creaseHold(S) {
  const { f, rows, idx } = S;
  let top = 0, bot = 0, nt = 0, nb = 0;
  for (const c of S.creaseCols) for (let r = 0; r < rows - 1; r++) {
    const ang = foldAngle(f, idx, c, r, S.span(c));
    if (r < rows / 3) { top += ang; nt++; } else if (r > 2 * rows / 3) { bot += ang; nb++; }
  }
  return { topDeg: nt ? top / nt : 0, bottomDeg: nb ? bot / nb : 0 };
}

// Sections: the strip cut across at shares of its depth (0 at the stitching, 1 at the scallop's crest), each a
// polyline of the points at that row in order along the strip, and the numbers that say what shape it is:
// spread (how far the row reaches round or along), ripple (how far it swings in and out of its mean line), lobes
// (the waves it makes), kink (the sharpest turn between two segments that is not an authored crease)
export function sections(S, shares = [0, 0.25, 0.5, 0.75, 1]) {
  const { f, cols, rows, idx, p } = S, x = f.pos, round = isRound(p);
  const out = [];
  for (const s of shares) {
    const r = Math.min(rows - 1, Math.round(s * (rows - 1)));
    const pts = [];
    for (let c = 0; c < cols; c++) { const i = idx(c, r); pts.push([x[3 * i], x[3 * i + 1], x[3 * i + 2]]); }
    let radial;
    if (round) radial = pts.map((q) => Math.hypot(q[0], q[2]));
    else radial = pts.map((q) => q[2]);
    const m = mean(radial);
    const sd = Math.sqrt(mean(radial.map((v) => (v - m) ** 2)));
    // lobes: the swings of the smoothed radial line about its mean, over the pleat repeat's scale
    const win = Math.max(1, Math.floor(S.prof.repeatCols / 2));
    const sm = radial.map((v, k) => { let a = 0, nn = 0; for (let q = Math.max(0, k - win); q <= Math.min(radial.length - 1, k + win); q++) { a += radial[q]; nn++; } return a / nn; });
    let lobes = 0;
    for (let k = 1; k < sm.length; k++) if ((sm[k - 1] - m) * (sm[k] - m) < 0) lobes++;
    let kink = 0;
    for (let k = 1; k < pts.length - 1; k++) {
      if (S.creaseCols.includes(k) || S.creaseCols.includes(k - 1) || S.p.kind === 'flute' || S.p.kind === 'accordion') continue;
      // (in the section's own plane: across the strip, so the scallop's rise and fall is not taken for a kink)
      const a = sub(pts[k], pts[k - 1]), b = sub(pts[k + 1], pts[k]);
      a[1] = 0; b[1] = 0;
      const cs = dot(a, b) / ((Math.hypot(...a) * Math.hypot(...b)) || 1);
      kink = Math.max(kink, Math.acos(Math.max(-1, Math.min(1, cs))) / DEG);
    }
    const ys = pts.map((q) => q[1]);
    out.push({ share: s, row: r, pts, mean: m, ripple: sd, rippleRel: sd / (Math.abs(m) || 1), lobes: Math.round(lobes / 2), kinkDeg: kink, yMean: mean(ys), ySpread: Math.max(...ys) - Math.min(...ys),
      spread: round ? 2 * m : Math.max(...pts.map((q) => q[0])) - Math.min(...pts.map((q) => q[0])) });
  }
  return out;
}

// The flags the contour check raises: a section that is perfectly round or flat where lace should ripple, a kink,
// a thread stretched, the cloth through the body
export function contourFlags(S, sec, st) {
  const flags = [];
  for (const s of sec) {
    if (s.share > 0.2 && s.rippleRel < 0.01 && isRound(S.p)) flags.push(`section ${s.share}: perfectly round (ripple ${(100 * s.rippleRel).toFixed(2)}%)`);
    if (s.share > 0.2 && s.ripple < 0.02 && S.p.mount === 'band') flags.push(`section ${s.share}: flat, no ripple`);
    if (s.kinkDeg > 60) flags.push(`section ${s.share}: kink ${s.kinkDeg.toFixed(0)} deg off a crease`);
  }
  if (st.max > 0.05) flags.push(`thread stretched ${(100 * st.max).toFixed(1)}%`);
  const nan = Array.from(S.f.pos).some((v) => !Number.isFinite(v));
  if (nan) flags.push('blew up (non-finite positions)');
  return flags;
}

// the key the fold cache files a solved strip under: its shape and the solver's settings
export function cacheKey(S) {
  const p = { ...S.p };
  delete p.frames;
  const held = S.pins.length / S.cols;
  return JSON.stringify({ v: 1, p, mat: S.mat, n: S.f.n, ...(held !== 1 ? { held } : {}) });
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
