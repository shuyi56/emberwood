// Clean the game head's mesh (John, 2026-11-20: the faults the face topology check, rq_topo.py, found in the
// approved game heads). Only the head's skin changes ('head part 0', or 'Head shape'); its look stays as it is:
//   - the creator's grid ends (rings of points drawn into one place, never merged: the cause of the torn slivers
//     and the slit at the crown) are rebuilt as real poles, a fan to one point (see rebuildPoles)
//   - degenerate triangles (no area, a corner named twice, or an edge under 1e-4 H) and folded slivers (a triangle turned over against its
//     neighbours, left by the simplifier) are collapsed along their shortest edge, so the surface closes over them;
//     the creator's grid ends at the crown and under the neck in rings of 600 points drawn into one, and these go too
//   - an edge with three or more faces (non-manifold) keeps only its two largest
//   - small holes away from the neck's opening (an open border of at most HOLE_MAX edges, default 32) are filled
//   - vertices no triangle uses are dropped
//   node clean-head.mjs in.glb out.glb
// merge-head.mjs runs the same clean (cleanHead) on the head after it simplifies it.
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';

const HOLE_MAX = +(process.env.HOLE_MAX || 32);

function stats(P, I) {
  // per triangle: area and unit normal; per vertex: area-weighted normal
  const nt = I.length / 3, A = new Float64Array(nt), FN = new Float64Array(nt * 3), VN = new Float64Array(P.length);
  for (let t = 0; t < nt; t++) {
    const a = I[3 * t] * 3, b = I[3 * t + 1] * 3, c = I[3 * t + 2] * 3;
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx, l = Math.hypot(nx, ny, nz);
    A[t] = 0.5 * l;
    if (l > 0) { FN[3 * t] = nx / l; FN[3 * t + 1] = ny / l; FN[3 * t + 2] = nz / l; }
    for (const v of [a, b, c]) { VN[v] += nx; VN[v + 1] += ny; VN[v + 2] += nz; }
  }
  return { A, FN, VN };
}

const ekey = (a, b) => (a < b ? a * 4294967296 + b : b * 4294967296 + a);

/** Border loops from next (place -> the border edges leaving it, each [place it goes to, a, b]): every border
 *  walked edge by edge, and cut into its simple loops wherever the walk meets a point twice (a bow tie). */
function borderLoops(next, most) {
  const loops = [], usedE = new Set();
  for (const [st, outs] of next) for (const e0 of outs) {
    if (usedE.has(e0)) continue;
    const walk = [], pos = new Map();
    let v = st, e = e0;
    while (e && !usedE.has(e) && walk.length <= most) {
      usedE.add(e);
      if (pos.has(v)) {
        const i = pos.get(v);
        loops.push(walk.splice(i));
        for (const q of [...pos.keys()]) if (pos.get(q) >= i) pos.delete(q);
      }
      pos.set(v, walk.length);
      walk.push([...e, v]);
      v = e[0];
      e = (next.get(v) || []).find((x) => !usedE.has(x));
    }
    if (v === st && walk.length) loops.push(walk);
  }
  return loops;
}

/** Clean one indexed triangle list in place of its indices; P the positions (flat). Returns the new indices and a
 *  count of what was done. A collapse maps one corner onto the other; points within 1e-5 H of each other are put at exactly one place. */
export function cleanIndices(P, I0, H) {
  let I = Array.from(I0);
  const n = P.length / 3, rep = { collapsed: 0, nonmanifold: 0, filled: 0 };
  const map = new Int32Array(n).map((_, i) => i);
  const find = (v) => { while (map[v] !== v) { map[v] = map[map[v]]; v = map[v]; } return v; };
  const minA = 1e-9 * H * H;
  // (a vertex's welded twin: the points at the same place, which the creator's UV seams and grid ends split)
  // (points within 1e-5 H of each other, as rq_topo's duplicates: a grid of that size, neighbours included)
  const cell = 1e-5 * H, at = new Map(), twin = new Int32Array(n);
  const ck = (x, y, z) => `${x},${y},${z}`;
  for (let v = 0; v < n; v++) {
    const x = Math.floor(P[3 * v] / cell), y = Math.floor(P[3 * v + 1] / cell), z = Math.floor(P[3 * v + 2] / cell);
    let t = -1;
    for (let dx = -1; dx <= 1 && t < 0; dx++) for (let dy = -1; dy <= 1 && t < 0; dy++) for (let dz = -1; dz <= 1 && t < 0; dz++) {
      for (const u of at.get(ck(x + dx, y + dy, z + dz)) || []) {
        if (Math.hypot(P[3 * u] - P[3 * v], P[3 * u + 1] - P[3 * v + 1], P[3 * u + 2] - P[3 * v + 2]) <= cell) { t = twin[u]; break; }
      }
    }
    twin[v] = t >= 0 ? t : v;
    const k = ck(x, y, z);
    if (!at.has(k)) at.set(k, []);
    at.get(k).push(v);
  }
  // (a seam's split points a hair apart are put at exactly one place: they stay split, for their UVs, but no
  // longer leave a crack or read as duplicates)
  for (let v = 0; v < n; v++) if (twin[v] !== v) for (let j = 0; j < 3; j++) P[3 * v + j] = P[3 * twin[v] + j];
  // (three rounds: closing a slit or dropping a face can leave a new sliver for the next round)
  for (let round = 0; round < 3; round++) {
  const before = I.length;
  for (let pass = 0; pass < 20; pass++) {
    const { A, FN, VN } = stats(P, I);
    let changed = 0;
    for (let t = 0; t < I.length / 3; t++) {
      const a = find(I[3 * t]), b = find(I[3 * t + 1]), c = find(I[3 * t + 2]);
      if (a === b || b === c || a === c) continue;
      let bad = A[t] < minA || twin[a] === twin[b] || twin[b] === twin[c] || twin[a] === twin[c];
      if (!bad) {        // (turned over: its normal against the mean of its corners' normals)
        let s = 0;
        for (const v of [a, b, c]) {
          const x = VN[3 * v], y = VN[3 * v + 1], z = VN[3 * v + 2], l = Math.hypot(x, y, z) || 1;
          s += (FN[3 * t] * x + FN[3 * t + 1] * y + FN[3 * t + 2] * z) / l;
        }
        bad = s / 3 < -0.2;
      }
      const L = (p, q) => Math.hypot(P[3 * p] - P[3 * q], P[3 * p + 1] - P[3 * q + 1], P[3 * p + 2] - P[3 * q + 2]);
      // (an edge under 1e-4 H: the last rings of the creator's grid, drawn in almost to a point at the crown)
      if (!bad) bad = Math.min(L(a, b), L(b, c), L(a, c)) < 1e-4 * H;
      if (!bad) continue;
      const L2 = (p, q) => Math.hypot(P[3 * p] - P[3 * q], P[3 * p + 1] - P[3 * q + 1], P[3 * p + 2] - P[3 * q + 2]);
      const e = [[a, b, L2(a, b)], [b, c, L2(b, c)], [a, c, L2(a, c)]].sort((x, y) => x[2] - y[2])[0];
      map[e[1]] = e[0];
      changed++;
    }
    const J = [];
    for (let t = 0; t < I.length; t += 3) {
      const a = find(I[t]), b = find(I[t + 1]), c = find(I[t + 2]);
      if (a !== b && b !== c && a !== c) J.push(a, b, c);
    }
    // (two triangles on the same three points: one of them goes)
    const seen = new Set(), K = [];
    for (let t = 0; t < J.length; t += 3) {
      const k = [twin[J[t]], twin[J[t + 1]], twin[J[t + 2]]].sort((x, y) => x - y).join(',');
      if (seen.has(k)) { changed++; continue; }
      seen.add(k); K.push(J[t], J[t + 1], J[t + 2]);
    }
    I = K;
    rep.collapsed += changed;
    if (!changed) break;
  }
  // caps: a triangle of no area whose corner lies on its long edge (collapsing its short edges only moves the
  // fault on); the long edge is flipped with the triangle across it, so both become real triangles
  for (let pass = 0; pass < 4; pass++) {
    const { A } = stats(P, I), tri = new Map();
    for (let t = 0; t < I.length / 3; t++) for (let k = 0; k < 3; k++) tri.set(`${twin[I[3 * t + k]]},${twin[I[3 * t + (k + 1) % 3]]}`, [t, k]);
    let flips = 0;
    const done = new Set();
    for (let t = 0; t < I.length / 3; t++) {
      if (A[t] >= minA || done.has(t)) continue;
      const L = (p, q) => Math.hypot(P[3 * p] - P[3 * q], P[3 * p + 1] - P[3 * q + 1], P[3 * p + 2] - P[3 * q + 2]);
      let k = 0, best = -1;
      for (let j = 0; j < 3; j++) { const d = L(I[3 * t + j], I[3 * t + (j + 1) % 3]); if (d > best) { best = d; k = j; } }
      const u = I[3 * t + k], v = I[3 * t + (k + 1) % 3], w = I[3 * t + (k + 2) % 3];
      const o = tri.get(`${twin[v]},${twin[u]}`);
      if (!o || done.has(o[0])) continue;
      const [t2, k2] = o, x = I[3 * t2 + (k2 + 2) % 3];
      if (twin[x] === twin[w]) continue;
      // (the quad u, x, v, w split the other way, on x - w)
      const v2 = I[3 * t2 + k2];        // (the far triangle's own corner at v, with its own UV)
      I[3 * t] = u; I[3 * t + 1] = x; I[3 * t + 2] = w;
      I[3 * t2] = x; I[3 * t2 + 1] = v2; I[3 * t2 + 2] = w;
      done.add(t); done.add(t2);
      flips++;
    }
    rep.flipped = (rep.flipped || 0) + flips;
    if (!flips) break;
  }
  // non-manifold edges (by place, so the UV seams' split points count as one): keep the two largest faces
  {
    const { A } = stats(P, I), faces = new Map();
    for (let t = 0; t < I.length / 3; t++) for (let k = 0; k < 3; k++) {
      const key = ekey(twin[I[3 * t + k]], twin[I[3 * t + (k + 1) % 3]]);
      if (!faces.has(key)) faces.set(key, []);
      faces.get(key).push(t);
    }
    const drop = new Set();
    for (const fs of faces.values()) if (fs.length > 2) {
      fs.sort((x, y) => A[y] - A[x]).slice(2).forEach((t) => drop.add(t));
    }
    if (drop.size) {
      rep.nonmanifold = drop.size;
      I = I.filter((_, i) => !drop.has(Math.floor(i / 3)));
    }
  }
  // small holes: open borders (by place) of up to HOLE_MAX edges, filled with a fan wound as their neighbours
  {
    const half = new Map();
    for (let t = 0; t < I.length / 3; t++) for (let k = 0; k < 3; k++) {
      const a = I[3 * t + k], b = I[3 * t + (k + 1) % 3];
      half.set(`${twin[a]},${twin[b]}`, [a, b]);
    }
    const next = new Map();          // (twin -> the border edges leaving it, walked against their faces)
    for (const [k, [a, b]] of half) {
      const [ta, tb] = k.split(',').map(Number);
      if (!half.has(`${tb},${ta}`)) { if (!next.has(tb)) next.set(tb, []); next.get(tb).push([ta, a, b]); }
    }
    const loops = borderLoops(next, 4 * HOLE_MAX);
    for (const loop of loops) {
      if (loop.length < 3 || loop.length > HOLE_MAX) continue;
      // (each border edge b -> a of the hole, with real vertex ids: the fan from the first)
      const vs = loop.map((x) => x[2]);
      // (a slit: the border's points all but on a line, as the crown's last ring drawn flat; a fan over it would
      // be triangles of no area, so the slit is closed by drawing its points into one instead)
      let area = 0;
      for (let i = 1; i + 1 < vs.length; i++) {
        const a = vs[0] * 3, b = vs[i] * 3, c = vs[i + 1] * 3;
        const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2], vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
        area = Math.max(area, 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx));
      }
      if (area < 1e-7 * H * H) {
        const tw = new Set(vs.map((v) => twin[v]));
        for (let i = 0; i < I.length; i++) if (tw.has(twin[I[i]])) I[i] = vs[0];
        const J = [];
        for (let t = 0; t < I.length; t += 3) if (I[t] !== I[t + 1] && I[t + 1] !== I[t + 2] && I[t] !== I[t + 2]) J.push(I[t], I[t + 1], I[t + 2]);
        I = J;
        rep.slits = (rep.slits || 0) + 1;
        continue;
      }
      for (let i = 1; i + 1 < vs.length; i++) I.push(vs[0], vs[i], vs[i + 1]);
      rep.filled++;
    }
  }
  if (I.length === before && round > 0) break;
  }
  return { I, rep };
}

/** Drop the vertices no triangle uses, from every attribute and morph target of the primitive. */
function compact(doc, prim, I) {
  const n = prim.getAttribute('POSITION').getCount(), keep = new Int32Array(n).fill(-1);
  let m = 0;
  for (const v of I) if (keep[v] < 0) keep[v] = m++;
  const remap = (acc) => {
    const k = acc.getElementSize(), src = acc.getArray(), dst = new src.constructor(m * k);
    for (let v = 0; v < n; v++) if (keep[v] >= 0) for (let j = 0; j < k; j++) dst[keep[v] * k + j] = src[v * k + j];
    acc.setArray(dst);
  };
  const done = new Set();
  for (const s of prim.listSemantics()) { const a = prim.getAttribute(s); if (!done.has(a)) { remap(a); done.add(a); } }
  for (const tg of prim.listTargets()) for (const s of tg.listSemantics()) { const a = tg.getAttribute(s); if (!done.has(a)) { remap(a); done.add(a); } }
  const out = new Uint32Array(I.length);
  for (let i = 0; i < I.length; i++) out[i] = keep[I[i]];
  let idx = prim.getIndices();
  if (!idx || idx.listParents().length > 2) { idx = doc.createAccessor().setType('SCALAR'); prim.setIndices(idx); }
  idx.setArray(out);
  return n - m;
}

/** The creator's grid ends (the crown, and the neck's end when it is not cut off) are rings of points drawn into
 *  one place but never merged; the simplifier then leaves torn slivers round them. The cause is fixed, not its
 *  traces: every place where POLE_MIN or more points meet (to 1e-4 H) is rebuilt as a real pole, the triangles within
 *  POLE_R (0.03 H, under the hair at the crown) taken out and the opening closed by a fan to one new point at its
 *  middle, which takes the attributes of the border point nearest it. Returns the new indices and how many. */
function rebuildPoles(doc, prim, I0, H) {
  const POLE_MIN = 8, R = (+(process.env.POLE_R || 0.03)) * H;
  const pos = prim.getAttribute('POSITION'), P = pos.getArray(), n = pos.getCount();
  const cell = 1e-4 * H, groups = new Map();
  for (let v = 0; v < n; v++) {
    const k = `${Math.round(P[3 * v] / cell)},${Math.round(P[3 * v + 1] / cell)},${Math.round(P[3 * v + 2] / cell)}`;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(v);
  }
  let I = Array.from(I0), made = 0;
  const added = [];
  const sites = [];
  for (const g of groups.values()) {
    if (g.length < POLE_MIN) continue;
    const c = [P[3 * g[0]], P[3 * g[0] + 1], P[3 * g[0] + 2]];
    if (sites.some((q) => Math.hypot(q[0] - c[0], q[1] - c[1], q[2] - c[2]) < 2 * R)) continue;     // (one per pole)
    sites.push(c);
    const near = (v) => Math.hypot(P[3 * v] - c[0], P[3 * v + 1] - c[1], P[3 * v + 2] - c[2]) < R;
    const keep = [];
    for (let t = 0; t < I.length; t += 3) if (!(near(I[t]) || near(I[t + 1]) || near(I[t + 2]))) keep.push(I[t], I[t + 1], I[t + 2]);
    // (the opening's border, by place: the kept triangles' edges near the cut with no triangle across them)
    const key = (v) => `${Math.round(P[3 * v] / (1e-5 * H))},${Math.round(P[3 * v + 1] / (1e-5 * H))},${Math.round(P[3 * v + 2] / (1e-5 * H))}`;
    const halves = (J) => {      // (each directed edge, by place)
      const h = new Map();
      for (let t = 0; t < J.length; t += 3) for (let k = 0; k < 3; k++) {
        const a = J[t + k], b = J[t + (k + 1) % 3];
        h.set(key(a) + '>' + key(b), [a, b]);
      }
      return h;
    };
    const half = halves(keep), cutv = new Set();
    for (let t = 0; t < I.length; t += 3) if (near(I[t]) || near(I[t + 1]) || near(I[t + 2])) for (let j = 0; j < 3; j++) cutv.add(key(I[t + j]));
    const nxt = new Map();
    for (const [k, [a, b]] of half) {        // (the borders the cut made: open edges whose ends a cut triangle had)
      const [ka, kb] = k.split('>');
      if (half.has(kb + '>' + ka) || !cutv.has(ka) || !cutv.has(kb)) continue;
      if (!nxt.has(kb)) nxt.set(kb, []);
      nxt.get(kb).push([ka, a, b]);
    }
    const loops = borderLoops(nxt, Infinity);
    if (!loops.length) continue;
    for (const loop of loops) {
      if (loop.length < 3) continue;
      const vs = loop.map((e) => e[2]);
      const m = [0, 0, 0];
      for (const v of vs) for (let j = 0; j < 3; j++) m[j] += P[3 * v + j] / vs.length;
      let best = vs[0], bd = Infinity;
      for (const v of vs) { const d = Math.hypot(P[3 * v] - m[0], P[3 * v + 1] - m[1], P[3 * v + 2] - m[2]); if (d < bd) { bd = d; best = v; } }
      // (the middle point: the border's mean, raised halfway to the pole's old place, as the dome there)
      const big = vs.length > 0.5 * nxt.size;
      added.push({ from: best, at: big ? [m[0] + 0.5 * (c[0] - m[0]), m[1] + 0.5 * (c[1] - m[1]), m[2] + 0.5 * (c[2] - m[2])] : m });
      const nv = n + added.length - 1;
      for (let i = 0; i < vs.length; i++) keep.push(nv, vs[i], vs[(i + 1) % vs.length]);
    }
    I = keep;
    made++;
  }
  if (added.length) {
    const grow = (acc, isPos, isNrm) => {
      const k = acc.getElementSize(), src = acc.getArray(), dst = new src.constructor((n + added.length) * k);
      dst.set(src);
      added.forEach((a, i) => {
        for (let j = 0; j < k; j++) dst[(n + i) * k + j] = src[a.from * k + j];
        if (isPos) for (let j = 0; j < 3; j++) dst[(n + i) * 3 + j] = a.at[j];
      });
      acc.setArray(dst);
    };
    const done = new Set();
    for (const s of prim.listSemantics()) { const a = prim.getAttribute(s); if (!done.has(a)) { grow(a, s === 'POSITION', s === 'NORMAL'); done.add(a); } }
    for (const tg of prim.listTargets()) for (const s of tg.listSemantics()) { const a = tg.getAttribute(s); if (!done.has(a)) { grow(a, false, false); done.add(a); } }
  }
  return { I, made };
}

/** Clean the head's skin in a document (merge-head.mjs calls this): returns what was done. */
export function cleanHead(doc) {
  const out = [];
  for (const node of doc.getRoot().listNodes()) {
    if (!/^(head part 0|Head shape)$/.test(node.getName()) || !node.getMesh()) continue;
    for (const prim of node.getMesh().listPrimitives()) {
      const P = prim.getAttribute('POSITION').getArray(), Iacc = prim.getIndices();
      const I0 = Iacc ? Iacc.getArray() : Uint32Array.from({ length: P.length / 3 }, (_, i) => i);
      let lo = Infinity, hi = -Infinity;
      for (let i = 1; i < P.length; i += 3) { lo = Math.min(lo, P[i]); hi = Math.max(hi, P[i]); }
      const poles = rebuildPoles(doc, prim, I0, hi - lo);
      const { I, rep } = cleanIndices(prim.getAttribute('POSITION').getArray(), poles.I, hi - lo);
      rep.poles = poles.made;
      rep.unused = compact(doc, prim, I);
      rep.tris = [I0.length / 3, I.length / 3];
      out.push([node.getName(), rep]);
    }
  }
  return out;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [, , inp, outp] = process.argv;
  const io = new NodeIO().registerExtensions(ALL_EXTENSIONS);
  const doc = await io.read(inp);
  for (const [name, r] of cleanHead(doc))
    console.log(`${name}: ${r.poles} grid ends rebuilt as poles, ${r.collapsed} slivers and degenerate triangles collapsed, ${r.flipped || 0} flat triangles flipped, ${r.nonmanifold} faces off non-manifold edges, ` +
                `${r.filled} holes filled, ${r.slits || 0} slits closed, ${r.unused} unused vertices dropped, triangles ${r.tris[0]} -> ${r.tris[1]}`);
  await io.write(outp, doc);
}
