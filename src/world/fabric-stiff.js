// Stiff cloth: bending solved implicitly, as one linear solve a step (fabric-core.js `stiffen`).
//
// Position-based constraints carry a bend one row at a time per walk over them, so a stiff cloth (leather, a
// boot shaft, heavy felt) sags as if it were soft unless it is walked tens of times a step: measured on the
// bench's cantilever, hinges and ties two along both stopped at a bending length of about 2 cm, the stiffness of
// linen, however stiff they were set. The bend is a property of the whole sheet, so here it is solved over the
// whole sheet at once:
//
//  - the bending energy is the discrete Willmore energy of a nearly isometric surface (Bergou, Wardetzky,
//    Harmon, Zorin and Grinspun 2006, "A quadratic bending model for inextensible surfaces"): E = B/2 sum_i
//    |(L x)_i - b_i|^2 / A_i, with L the cotangent Laplacian of the rest mesh, A_i each point's area, and b_i its
//    rest curvature (the rest's L x, turned with the cloth round the point, as Projective Dynamics does: Bouaziz,
//    Martin, Liu, Kavan and Pauly 2014);
//  - a step minimises the inertia plus that energy: (M / h^2 + B L A^-1 L) x = M x~ / h^2 + B L A^-1 b. The
//    matrix only changes with h and the pins, so it is factored once (banded Cholesky after a reverse Cuthill-McKee
//    ordering) and each step is two triangular solves for x, y and z;
//  - B is the cloth's flexural rigidity, from its bending length c (Peirce 1930): B = weight per area x g x c^3,
//    so a material is set by what a cantilever test of it measures, not by a tuned number;
//  - a set (leather that creases where it keeps folding): a point bent past its rest by more than the curvature
//    of `setRadius` moves its rest curvature toward the bend at `creep` a second.
//
// With `stretchK` the edges are held at their length inside the same solve (Projective Dynamics' springs, solved
// with the bend in `passes` local and global steps), which a stiff, heavy sheet needs: leather solved for its bend
// alone, with its stretch left to the constraints after, folded up at 240 steps a second (a boot shaft laid on its
// side sagged 20 units; with the edges in the solve, 0.4). Open pieces are better without: on the bench, linen
// with the edges in the solve shrink-wrapped the ball and fell off it in a wad, and a belted leather panel bunched
// up at the belt (an open question, left for later: the springs and the body's pushes after the solve fight);
// with them left to the constraints both hang as they should. The shear, the hinges, the tethers and the body stay
// with the position-based solve that follows.

const EPS = 1e-12;

export class StiffBend {
  // f: the Fabric (its tris, rest positions now, kg, w); B: flexural rigidity (kg unit^2 / s^2); o: { scale: per
  // point share of B (Float64Array), setRadius (units), creep (1/s) }
  constructor(f, B, o = {}) {
    this.f = f; this.B = B; this.o = o;
    const n = f.n, x = f.pos, t = f.tris;
    // the cotangent Laplacian (as rows of neighbour, weight) and each point's area
    const W = Array.from({ length: n }, () => new Map()), area = new Float64Array(n);
    for (let k = 0; k < t.length; k += 3) {
      const v = [t[k], t[k + 1], t[k + 2]];
      const P = v.map((i) => [x[3 * i], x[3 * i + 1], x[3 * i + 2]]);
      const cr = cross(sub(P[1], P[0]), sub(P[2], P[0])), A2 = Math.hypot(...cr);
      if (A2 < EPS) continue;
      for (let e = 0; e < 3; e++) {
        // the angle at corner e faces the edge between the other two
        const a = P[e], b = P[(e + 1) % 3], c = P[(e + 2) % 3];
        const cot = dot(sub(b, a), sub(c, a)) / A2;
        const i = v[(e + 1) % 3], j = v[(e + 2) % 3];
        W[i].set(j, (W[i].get(j) || 0) + cot / 2); W[j].set(i, (W[j].get(i) || 0) + cot / 2);
        area[v[e]] += A2 / 6;
      }
    }
    this.W = W; this.area = area;
    // the edges, held at their rest length in the same solve (so the bend and the stretch are solved together)
    const seen = new Set(), ea = [], eb = [], el = [];
    for (let k = 0; k < t.length; k += 3) for (let e = 0; e < 3; e++) {
      const i = t[k + e], j = t[k + (e + 1) % 3], key = i < j ? i * n + j : j * n + i;
      if (seen.has(key)) continue;
      seen.add(key); ea.push(i); eb.push(j);
      el.push(Math.hypot(x[3 * i] - x[3 * j], x[3 * i + 1] - x[3 * j + 1], x[3 * i + 2] - x[3 * j + 2]));
    }
    this.ea = Int32Array.from(ea); this.eb = Int32Array.from(eb); this.el = Float64Array.from(el);
    // the rest curvature, in each point's own frame (the rest frame: identity)
    this.nb = W.map((m) => Int32Array.from(m.keys()));
    this.wt = W.map((m) => Float64Array.from(m.values()));
    this.b0 = this._lap(x);
    this.rest = Float64Array.from(x);
    this.q = new Float64Array(4 * n); for (let i = 0; i < n; i++) this.q[4 * i + 3] = 1;
    this.scale = o.scale || new Float64Array(n).fill(1);
    this._h = null;
  }
  // (L x)_i = sum_j w_ij (x_j - x_i), into a 3n array
  _lap(x, out = new Float64Array(x.length)) {
    const n = this.f.n;
    for (let i = 0; i < n; i++) {
      let sx = 0, sy = 0, sz = 0;
      const nb = this.nb[i], wt = this.wt[i];
      for (let k = 0; k < nb.length; k++) { const j = nb[k], w = wt[k]; sx += w * (x[3 * j] - x[3 * i]); sy += w * (x[3 * j + 1] - x[3 * i + 1]); sz += w * (x[3 * j + 2] - x[3 * i + 2]); }
      out[3 * i] = sx; out[3 * i + 1] = sy; out[3 * i + 2] = sz;
    }
    return out;
  }
  // the system for a step h: the free points (the pins are given), in an order that keeps the band narrow
  _factor(h) {
    const f = this.f, n = f.n, w = f.w, kg = f.kg, B = this.B, area = this.area, sc = this.scale;
    const free = [], slot = new Int32Array(n).fill(-1);
    for (let i = 0; i < n; i++) if (w[i] !== 0) free.push(i);
    // K = M / h^2 + B L A^-1 L (with the point's share of B on its own row of A^-1: sum_k L_ik s_k / A_k L_kj)
    const rows = Array.from({ length: n }, () => new Map());
    const Lrow = (k) => { const m = new Map([[k, 0]]); this.nb[k].forEach((j, q) => { m.set(j, this.wt[k][q]); m.set(k, m.get(k) - this.wt[k][q]); }); return m; };
    const L = Array.from({ length: n }, (_, k) => Lrow(k));
    for (let k = 0; k < n; k++) {
      if (area[k] < EPS) continue;
      const c = B * sc[k] / area[k];
      for (const [i, li] of L[k]) for (const [j, lj] of L[k]) rows[i].set(j, (rows[i].get(j) || 0) + c * li * lj);
    }
    const mass = new Float64Array(n);
    for (let i = 0; i < n; i++) mass[i] = kg ? kg[i] : 1;
    for (const i of free) rows[i].set(i, (rows[i].get(i) || 0) + mass[i] / (h * h));
    // the edges' springs: ks (x_j - x_i - d)^2 / 2, with ks a share `stretchK` over each edge's two masses / h^2
    let mm = 0; for (const i of free) mm += mass[i]; mm /= Math.max(1, free.length);
    const ks = (this.o.stretchK ?? 0) * mm / (h * h);
    this.ks = ks;
    for (let e = 0; ks && e < this.ea.length; e++) {
      const i = this.ea[e], j = this.eb[e];
      rows[i].set(i, (rows[i].get(i) || 0) + ks); rows[j].set(j, (rows[j].get(j) || 0) + ks);
      rows[i].set(j, (rows[i].get(j) || 0) - ks); rows[j].set(i, (rows[j].get(i) || 0) - ks);
    }
    // reverse Cuthill-McKee over the free points
    const deg = (i) => rows[i].size, order = [], seen = new Uint8Array(n);
    const byDeg = free.slice().sort((a, b) => deg(a) - deg(b));
    for (const s of byDeg) {
      if (seen[s]) continue;
      seen[s] = 1; order.push(s);
      for (let q = order.length - 1; q < order.length; q++) {
        const nbr = [...rows[order[q]].keys()].filter((j) => w[j] !== 0 && !seen[j]).sort((a, b) => deg(a) - deg(b));
        for (const j of nbr) { seen[j] = 1; order.push(j); }
      }
    }
    order.reverse();
    // (the points' own order, row by row for a sheet, is often narrower still: keep whichever band is narrower)
    const band = (ord) => { const sl = new Int32Array(n).fill(-1); ord.forEach((i, k) => { sl[i] = k; }); let b = 0; for (const i of ord) for (const j of rows[i].keys()) if (sl[j] >= 0) b = Math.max(b, Math.abs(sl[i] - sl[j])); return b; };
    if (band(free) < band(order)) order.splice(0, order.length, ...free);
    order.forEach((i, k) => { slot[i] = k; });
    const m = order.length;
    const bw = band(order);
    // banded Cholesky: G (lower, m x (bw + 1)), G[i][i - j] for j in i-bw..i
    const S = bw + 1, G = new Float64Array(m * S);
    for (const i of order) for (const [j, v] of rows[i]) { const a = slot[i], b = slot[j]; if (b >= 0 && b <= a) G[a * S + (a - b)] = v; }
    for (let i = 0; i < m; i++) {
      const j0 = Math.max(0, i - bw);
      for (let j = j0; j <= i; j++) {
        let s = G[i * S + (i - j)];
        const k0 = Math.max(j0, j - bw);
        for (let k = k0; k < j; k++) s -= G[i * S + (i - k)] * G[j * S + (j - k)];
        if (i === j) G[i * S] = Math.sqrt(Math.max(s, EPS));
        else G[i * S + (i - j)] = s / G[j * S];
      }
    }
    // the pins' part of each free row (moved to the right-hand side every step)
    const pinCols = order.map((i) => [...rows[i]].filter(([j]) => w[j] === 0));
    Object.assign(this, { _h: h, order, slot, G, S, bw, m, pinCols, mass, rhs: new Float64Array(3 * m), _pinsAt: f.w.reduce((s, v, i) => s + (v === 0 ? i + 1 : 0), 0) });
  }
  // the bend for one step: x (f.pos) holds the predicted positions; it is replaced by the solve. The rest
  // curvature is turned by each point's turn, which depends on the answer, so the solve is repeated from it
  // (Projective Dynamics' local and global steps): `passes` times, 2 by default
  step(h) {
    const f = this.f, n = f.n, x = f.pos;
    const pinsAt = f.w.reduce((s, v, i) => s + (v === 0 ? i + 1 : 0), 0);
    if (this._h !== h || this._pinsAt !== pinsAt) this._factor(h);
    const pred = this._pred || (this._pred = new Float64Array(3 * n));
    pred.set(x);
    const passes = this.o.passes ?? 2;
    for (let q = 0; q < passes; q++) this._solve(h, pred);
  }
  _solve(h, pred) {
    const f = this.f, n = f.n, x = f.pos;
    const { order, G, S, bw, m, pinCols, mass, rhs } = this, B = this.B, area = this.area, sc = this.scale;
    // each point's turn from its rest one-ring to now, and its rest curvature turned with it: b_i = R_i b0_i
    const b = this._b || (this._b = new Float64Array(3 * n));
    this._rotations();
    const R = this._R;
    for (let i = 0; i < n; i++) {
      const r = 9 * i, a = 3 * i, b0 = this.b0;
      b[a] = R[r] * b0[a] + R[r + 1] * b0[a + 1] + R[r + 2] * b0[a + 2];
      b[a + 1] = R[r + 3] * b0[a] + R[r + 4] * b0[a + 1] + R[r + 5] * b0[a + 2];
      b[a + 2] = R[r + 6] * b0[a] + R[r + 7] * b0[a + 1] + R[r + 8] * b0[a + 2];
    }
    // v = B s A^-1 b, then L v
    const v = this._v || (this._v = new Float64Array(3 * n));
    for (let i = 0; i < n; i++) { const c = area[i] > EPS ? B * sc[i] / area[i] : 0; v[3 * i] = c * b[3 * i]; v[3 * i + 1] = c * b[3 * i + 1]; v[3 * i + 2] = c * b[3 * i + 2]; }
    const Lv = this._lap(v, this._Lv || (this._Lv = new Float64Array(3 * n)));
    const ks = this.ks, ea = this.ea, eb = this.eb, el = this.el;
    for (let e = 0; ks && e < ea.length; e++) {
      const i = ea[e], j = eb[e], a = 3 * i, c = 3 * j;
      const dx = x[c] - x[a], dy = x[c + 1] - x[a + 1], dz = x[c + 2] - x[a + 2], l = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1, q = ks * el[e] / l;
      Lv[a] -= q * dx; Lv[a + 1] -= q * dy; Lv[a + 2] -= q * dz; Lv[c] += q * dx; Lv[c + 1] += q * dy; Lv[c + 2] += q * dz;
    }
    const h2 = h * h;
    for (let k = 0; k < m; k++) {
      const i = order[k];
      let rx = mass[i] / h2 * pred[3 * i] + Lv[3 * i], ry = mass[i] / h2 * pred[3 * i + 1] + Lv[3 * i + 1], rz = mass[i] / h2 * pred[3 * i + 2] + Lv[3 * i + 2];
      for (const [j, kv] of pinCols[k]) { rx -= kv * x[3 * j]; ry -= kv * x[3 * j + 1]; rz -= kv * x[3 * j + 2]; }
      rhs[3 * k] = rx; rhs[3 * k + 1] = ry; rhs[3 * k + 2] = rz;
    }
    // G G^T y = rhs: forward, then back
    for (let i = 0; i < m; i++) {
      let sx = rhs[3 * i], sy = rhs[3 * i + 1], sz = rhs[3 * i + 2];
      for (let j = Math.max(0, i - bw); j < i; j++) { const g = G[i * S + (i - j)]; sx -= g * rhs[3 * j]; sy -= g * rhs[3 * j + 1]; sz -= g * rhs[3 * j + 2]; }
      const d = G[i * S]; rhs[3 * i] = sx / d; rhs[3 * i + 1] = sy / d; rhs[3 * i + 2] = sz / d;
    }
    for (let i = m - 1; i >= 0; i--) {
      let sx = rhs[3 * i], sy = rhs[3 * i + 1], sz = rhs[3 * i + 2];
      for (let j = i + 1; j <= Math.min(m - 1, i + bw); j++) { const g = G[j * S + (j - i)]; sx -= g * rhs[3 * j]; sy -= g * rhs[3 * j + 1]; sz -= g * rhs[3 * j + 2]; }
      const d = G[i * S]; rhs[3 * i] = sx / d; rhs[3 * i + 1] = sy / d; rhs[3 * i + 2] = sz / d;
    }
    for (let k = 0; k < m; k++) { const i = order[k]; x[3 * i] = rhs[3 * k]; x[3 * i + 1] = rhs[3 * k + 1]; x[3 * i + 2] = rhs[3 * k + 2]; }
  }
  // the set, after the step's constraints: where the cloth is bent past its rest by more than setRadius's
  // curvature, the rest curvature creeps toward the bend
  set(h) {
    const o = this.o;
    if (!(o.creep > 0) || !(o.setRadius > 0)) return;
    const f = this.f, n = f.n, R = this._R, Lx = this._lap(f.pos, this._Lx), b0 = this.b0, area = this.area, kc = Math.min(1, o.creep * h);
    for (let i = 0; i < n; i++) {
      if (f.w[i] === 0) continue;
      const r = 9 * i, a = 3 * i;
      // R^T (L x)_i - b0_i: how far the bend is from its rest, in the rest frame
      const lx = R[r] * Lx[a] + R[r + 3] * Lx[a + 1] + R[r + 6] * Lx[a + 2], ly = R[r + 1] * Lx[a] + R[r + 4] * Lx[a + 1] + R[r + 7] * Lx[a + 2], lz = R[r + 2] * Lx[a] + R[r + 5] * Lx[a + 1] + R[r + 8] * Lx[a + 2];
      const dx = lx - b0[a], dy = ly - b0[a + 1], dz = lz - b0[a + 2], d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      // (|L x| is twice the area times the mean curvature: a bend of radius r is 2 A / r)
      const y = 2 * area[i] / o.setRadius;
      if (d <= y) continue;
      const t = kc * (1 - y / d);
      b0[a] += t * dx; b0[a + 1] += t * dy; b0[a + 2] += t * dz;
    }
  }
  // the turn of each point's one-ring from rest (Mueller, Bender, Chentanez and Macklin 2016), from the last step's
  _rotations() {
    const f = this.f, n = f.n, x = f.pos, x0 = this.rest, Q = this.q, R = this._R || (this._R = new Float64Array(9 * n));
    for (let i = 0; i < n; i++) {
      const nb = this.nb[i];
      let a00 = 0, a01 = 0, a02 = 0, a10 = 0, a11 = 0, a12 = 0, a20 = 0, a21 = 0, a22 = 0;
      for (let k = 0; k < nb.length; k++) {
        const j = nb[k];
        const px = x[3 * j] - x[3 * i], py = x[3 * j + 1] - x[3 * i + 1], pz = x[3 * j + 2] - x[3 * i + 2];
        const rx = x0[3 * j] - x0[3 * i], ry = x0[3 * j + 1] - x0[3 * i + 1], rz = x0[3 * j + 2] - x0[3 * i + 2];
        a00 += px * rx; a01 += px * ry; a02 += px * rz; a10 += py * rx; a11 += py * ry; a12 += py * rz; a20 += pz * rx; a21 += pz * ry; a22 += pz * rz;
      }
      let qx = Q[4 * i], qy = Q[4 * i + 1], qz = Q[4 * i + 2], qw = Q[4 * i + 3];
      for (let it = 0; it < (this.o.turnIterations ?? 3); it++) {
        const r00 = 1 - 2 * (qy * qy + qz * qz), r01 = 2 * (qx * qy - qz * qw), r02 = 2 * (qx * qz + qy * qw);
        const r10 = 2 * (qx * qy + qz * qw), r11 = 1 - 2 * (qx * qx + qz * qz), r12 = 2 * (qy * qz - qx * qw);
        const r20 = 2 * (qx * qz - qy * qw), r21 = 2 * (qy * qz + qx * qw), r22 = 1 - 2 * (qx * qx + qy * qy);
        const ox = (r10 * a20 - r20 * a10) + (r11 * a21 - r21 * a11) + (r12 * a22 - r22 * a12);
        const oy = (r20 * a00 - r00 * a20) + (r21 * a01 - r01 * a21) + (r22 * a02 - r02 * a22);
        const oz = (r00 * a10 - r10 * a00) + (r01 * a11 - r11 * a01) + (r02 * a12 - r12 * a02);
        const dd = Math.abs(r00 * a00 + r10 * a10 + r20 * a20 + r01 * a01 + r11 * a11 + r21 * a21 + r02 * a02 + r12 * a12 + r22 * a22) + 1e-12;
        const wx = ox / dd, wy = oy / dd, wz = oz / dd, ang = Math.sqrt(wx * wx + wy * wy + wz * wz);
        if (ang < 1e-9) break;
        const sh = Math.sin(ang / 2) / ang, dx = wx * sh, dy = wy * sh, dz = wz * sh, dw = Math.cos(ang / 2);
        const nx = dw * qx + dx * qw + dy * qz - dz * qy, ny = dw * qy - dx * qz + dy * qw + dz * qx;
        const nz = dw * qz + dx * qy - dy * qx + dz * qw, nw = dw * qw - dx * qx - dy * qy - dz * qz;
        const ql = Math.hypot(nx, ny, nz, nw);
        qx = nx / ql; qy = ny / ql; qz = nz / ql; qw = nw / ql;
      }
      Q[4 * i] = qx; Q[4 * i + 1] = qy; Q[4 * i + 2] = qz; Q[4 * i + 3] = qw;
      const r = 9 * i;
      R[r] = 1 - 2 * (qy * qy + qz * qz); R[r + 1] = 2 * (qx * qy - qz * qw); R[r + 2] = 2 * (qx * qz + qy * qw);
      R[r + 3] = 2 * (qx * qy + qz * qw); R[r + 4] = 1 - 2 * (qx * qx + qz * qz); R[r + 5] = 2 * (qy * qz - qx * qw);
      R[r + 6] = 2 * (qx * qz - qy * qw); R[r + 7] = 2 * (qy * qz + qx * qw); R[r + 8] = 1 - 2 * (qx * qx + qy * qy);
    }
  }
  // how far each point has set (|b0 - its first rest curvature|, over 2 A: as a curvature, 1/units)
  setCurvature() {
    const n = this.f.n, out = new Float64Array(n), first = this._first || this._lap(this.rest);
    this._first = first;
    for (let i = 0; i < n; i++) { const a = 3 * i; out[i] = this.area[i] > EPS ? Math.hypot(this.b0[a] - first[a], this.b0[a + 1] - first[a + 1], this.b0[a + 2] - first[a + 2]) / (2 * this.area[i]) : 0; }
    return out;
  }
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
