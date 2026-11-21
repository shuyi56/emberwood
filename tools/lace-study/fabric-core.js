// Fabric: a particle solver for cloth and for hair that hangs as a sheet.
//
// Spring bones (springbones.js) swing each bone about its parent and pull it back to the way it was modelled, so a
// cape keeps the shape it was given. Fabric has no modelled shape to return to. It is a sheet of points held
// together by its own weave, and its shape is whatever gravity and the body make of it:
//
//  - warp and weft threads keep their length (`stretch`): the sheet does not grow where it is pulled;
//  - the diagonals keep theirs, less firmly (`shear`): cloth leans on the bias before it stretches;
//  - a thread resists folding (`bend`): each point is tied to the point two along, so the sheet rolls over an edge
//    in a curve and gathers into a few folds instead of creasing at every row;
//  - a thread may also be `slack`: free to change length by that share before it resists. Hair's weft is slack
//    (strands slide past each other and part), cloth's is not.
//
// For cloth that must hold a fold, the sheet can bend at hinges instead (`hinge`): each pair of triangles that
// share an edge keeps the angle between them (Bridson, Marino and Fedkiw 2003, "Simulation of clothing with folds
// and wrinkles"; as a constraint, Mueller et al. 2007, "Position based dynamics"). A hinge has a rest angle, so a
// crease can be authored (`crease`, `fold`), and a stiff cloth can take a set (`yield`, `creep`): bent past its
// rest by more than `yield` it slowly settles toward the bend it is held in, as leather creases at the ankle of a
// boot that keeps flexing there. A distance tie two points along cannot do either: it smooths every fold out.
//
// The cloth has a weight. `weigh(density)` gives each point the mass of the cloth round it (its share of each
// triangle's area); heavy points move light ones more than they are moved. The air's drag (`air`, `wind`) acts on
// each point's area across the flow and is divided by that mass, so linen flies where leather of the same size
// only lifts. Masses are kept relative (the free points' mean is 1), so the compliances mean the same on any
// cloth and any units.
//
// The method is XPBD (Macklin, Mueller and Chentanez 2016, "XPBD: position-based simulation of compliant
// constrained dynamics"), stepped in small fixed steps (Macklin et al. 2019, "Small steps in physics simulation"):
// each step moves every point by its speed and by gravity, then walks the constraints once or a few times,
// moving the points of each toward its rest. A constraint's `compliance` (how soft it is) is divided by the step
// squared, so the cloth is as stiff at 120 steps a second as at 600: stiffness is a property of the fabric, not of
// the step. After the constraints each point is pushed out of the body, and the body's hold on it takes away its
// slide along it (`friction`), so cloth and hair stay where they are laid on a shoulder instead of pouring off.
// With `contact: 'iteration'` the body is tested after every walk over the constraints, not only after the
// last, so a fast limb cannot carry the cloth through itself between two tests.
//
// Stiff cloth needs many walks over the constraints to stop stretching under its own weight. Tethers (`tether`;
// Kim, Chentanez and Mueller 2012, "Long range attachments") tie each point to the pins it hangs from at its
// distance along the cloth, and only stop it going further: one walk then does what tens would.
//
// Known limit: friction and held chains (`hold`) work against each other. With much friction a held sheet that lies
// on a body keeps creeping instead of coming to rest; hair is draped with 0.05, the cape runs with 0.2.
//
// Nothing here knows about three.js, bones or a page: positions are a flat array of x, y, z, the body is a
// function, and the units are the caller's. The cape and the hair in the game run it through fabric.js (on the
// same bones, colliders and bench as the spring bones); the Face Creator runs it once, to rest, to drape a
// hairstyle on the figure that will wear it. tools/fabric-bench measures it.

import { StiffBend } from './fabric-stiff.js';

export const FABRIC_DEFAULTS = {
  gravity: [0, -450, 0],     // units / s^2
  damping: 0.02,             // share of each point's speed lost per 1/60 s
  iterations: 2,             // walks over the constraints per step
  stretch: 0,                // compliance of the threads (0: they keep their length)
  shear: 2e-5,               // compliance of the diagonals
  bend: 2e-4,                // compliance of the ties two points along
  friction: 0.5,             // how well the body holds what lies on it: the slide it stops, as a share of its push
  lead: 0.9,                 // held chains: the share of a point's correction the point above takes the other way
  // (the rest are off unless asked for)
  air: 0,                    // the air's drag: half its density times the drag coefficient (kg / unit^3, with
                             // weigh's density in kg / unit^2); see material()
  wind: [0, 0, 0],           // the air's speed (units / s): minus her speed, when the body is held in place
  yield: Infinity,           // a hinge bent past its rest by more than this (radians) takes a set...
  creep: 0,                  // ...this share of the excess a second
  setMax: 2.6,               // the largest rest angle a set can reach (radians)
  thickness: 0,              // points of the sheet keep this far apart from points they are not tied to (0: off)
  contact: 'step',           // 'step': the body after the constraints; 'iteration': after every walk over them
  edgeContact: false,        // test the middle of every edge against the body too (body(p, -1, ref))
  tetherSlack: 0.01,         // a tether lets a point go this share further than its distance along the cloth
  substep: 1 / 240,          // advance(): the fixed step (s)
  order: 'build',            // 'root': walk the constraints from the pins out (see _orderFromPins)
};

// Cloth by what it is made of, in SI: weight (kg/m2), bending length (m, the cantilever test's, Peirce 1930: how
// far a strip reaches over an edge before it droops), for leather the radius it takes a set at, for boot leather
// `stretchK` (its length held inside the bend's solve too: a closed, stiff, heavy shaft otherwise folds up between
// the solve and the constraints at 240 steps a second; open pieces do better without it, see fabric-stiff.js);
// the rest as FABRIC_DEFAULTS. Bend every one of them with stiffen(). tools/fabric-bench --cantilever checks that the
// solver gives each its bending length.
export const MATERIALS = {
  silk:        { weight: 0.07, bendingLength: 0.012, hinge: 3e-1, shear: 4e-5, friction: 0.3, damping: 0.03 },
  linen:       { weight: 0.18, bendingLength: 0.025, hinge: 4e-2, shear: 2e-5, friction: 0.5, damping: 0.03 },
  wool:        { weight: 0.55, bendingLength: 0.04,  hinge: 1e-2, shear: 1e-5, friction: 0.6, damping: 0.03 },
  // garment leather (about 1 mm): a jerkin, a belted skirt panel, a pouch
  leather:     { weight: 1.0,  bendingLength: 0.07,  hinge: 4e-3, shear: 0,    friction: 0.7, damping: 0.05, setRadius: 0.025, creep: 0.15 },
  // boot leather (2 to 2.5 mm, vegetable tanned): a riding boot's shaft, a scabbard
  bootLeather: { weight: 2.2,  bendingLength: 0.14,  hinge: 4e-3, shear: 0,    friction: 0.7, damping: 0.05, setRadius: 0.025, creep: 0.15, stretchK: 30 },
};

// A material's settings in the caller's units (unitsPerMetre: 39.4 for the figures): `density` for weigh(), `air`
// for the drag, and the material's own. cd: the air's drag coefficient across a sheet
export function material(name, unitsPerMetre = 1, { cd = 1.2, airDensity = 1.2 } = {}) {
  const m = MATERIALS[name];
  if (!m) throw new Error(`fabric: no material ${name}`);
  const { weight, bendingLength, setRadius, ...rest } = m;
  const u = unitsPerMetre;
  return { ...rest, density: weight / (u * u), bendingLength: bendingLength * u, setRadius: setRadius ? setRadius * u : 0, air: 0.5 * airDensity * cd / (u * u * u), gravity: [0, -9.81 * u, 0] };
}

const wrapAngle = (a) => (a > Math.PI ? a - 2 * Math.PI : a < -Math.PI ? a + 2 * Math.PI : a);

export class Fabric {
  // n: the number of points; pos: Float64Array of 3n (taken over, not copied)
  constructor(pos, settings = {}) {
    this.n = pos.length / 3;
    this.pos = pos;
    this.prev = Float64Array.from(pos);
    this.w = new Float64Array(this.n).fill(1);       // inverse mass, relative: 0 pins a point
    this.kg = null;                                  // each point's mass (weigh), for the air's drag
    this.area = new Float64Array(this.n);            // each point's share of the cloth's area
    this.settings = { ...FABRIC_DEFAULTS, ...settings };
    this.ci = []; this.cj = []; this.rest = []; this.comp = []; this.slack = []; this.lambda = null;
    this.tris = [];          // triangles (point indices, three a triangle): the cloth's area, the air, the hinges
    this.hinges = { a: [], b: [], c: [], d: [], rest: [], comp: [], plastic: [] };   // (see hinge())
    this._edgeHinge = new Map();
    this.tethers = null;
    this.chains = [];        // lists of point indices whose segments keep their length exactly (see `hold`)
    this.touch = new Uint8Array(this.n);
    this._packed = null;
    this._acc = 0;
  }
  pin(i) { this.w[i] = 0; }
  // tie points i and j at the distance they are at now (or `rest`); compliance 0 is rigid; slack: the share of
  // the rest length the distance may change freely
  tie(i, j, compliance = 0, slack = 0, rest = null) {
    if (this.w[i] === 0 && this.w[j] === 0) return;
    const p = this.pos;
    this.ci.push(i); this.cj.push(j); this.comp.push(compliance); this.slack.push(slack);
    this.rest.push(rest ?? Math.hypot(p[3 * i] - p[3 * j], p[3 * i + 1] - p[3 * j + 1], p[3 * i + 2] - p[3 * j + 2]));
    this.lambda = null; this._packed = null;
  }
  // a triangle of the cloth (for its area, the air and the hinges)
  triangle(a, b, c) { this.tris.push(a, b, c); this._packed = null; }
  // A hinge on the edge c-d between the triangles (a, c, d) and (b, d, c): keeps the angle between them at `rest`
  // (radians, 0 flat; the angle they are at now if not given). plastic: it may take a set (yield, creep)
  hinge(a, b, c, d, compliance, rest = null, plastic = true) {
    const H = this.hinges;
    H.a.push(a); H.b.push(b); H.c.push(c); H.d.push(d); H.comp.push(compliance); H.plastic.push(plastic ? 1 : 0);
    H.rest.push(rest ?? this._angle(a, b, c, d));
    this._edgeHinge.set(this._key(c, d), H.a.length - 1);
    this._packed = null;
    return H.a.length - 1;
  }
  _key(i, j) { return i < j ? i * this.n + j : j * this.n + i; }
  // hinges over every inner edge of the triangles given since the last call (from `first`, in tris)
  _hingeTriangles(first, compliance, flat) {
    const t = this.tris, opp = new Map();
    for (let k = first; k < t.length; k += 3) {
      for (let e = 0; e < 3; e++) {
        const c = t[k + e], d = t[k + (e + 1) % 3], a = t[k + (e + 2) % 3], key = this._key(c, d);
        const o = opp.get(key);
        // (the second triangle on an edge walks it the other way: (b, d, c))
        if (o) { if (o.length === 3) this.hinge(o[0], a, o[1], o[2], compliance, flat ? 0 : null); o.push(a); }
        else opp.set(key, [a, c, d]);
      }
    }
  }
  // A cloth from a mesh: tris, a flat list of point indices (three a triangle, wound alike). Ties every edge
  // (compliance `stretch`), and bends at a hinge on every inner edge (compliance `hinge`; flat: rest flat, else as
  // the mesh lies now). A pattern piece cut flat and sewn is a mesh: tie its seams with tie().
  mesh(tris, o = {}) {
    const s = this.settings, first = this.tris.length, seen = new Set();
    for (let k = 0; k < tris.length; k += 3) {
      this.triangle(tris[k], tris[k + 1], tris[k + 2]);
      for (let e = 0; e < 3; e++) {
        const i = tris[k + e], j = tris[k + (e + 1) % 3], key = this._key(i, j);
        if (seen.has(key)) continue;
        seen.add(key);
        this.tie(i, j, o.stretch ?? s.stretch, 0);
      }
    }
    this._hingeTriangles(first, o.hinge ?? s.hinge ?? 1e-2, o.flat);
  }
  // A sheet from a grid of point indices, at(c, r) for column c (a thread from the top down) and row r, giving -1
  // where there is no point. Ties the warp (down each column), the weft (across), the diagonals and the bend
  // ties, with the settings' compliances; o: { weftSlack, wrap (the last column meets the first), weft, shear,
  // bend, warpBend, hold (keep each column's segments at their exact length after every step: bones, hair),
  // hinge (bend at hinges with this compliance instead of the ties two along), flat (the hinges' rest is flat) }
  sheet(cols, rows, at, o = {}) {
    const s = this.settings, ok = (c, r) => (r >= 0 && r < rows && (o.wrap || (c >= 0 && c < cols)) ? at(((c % cols) + cols) % cols, r) : -1);
    const link = (a, b, k, slack) => { if (a >= 0 && b >= 0 && k != null) this.tie(a, b, k, slack); };
    const weft = o.weft === undefined ? s.stretch : o.weft, shear = o.shear === undefined ? s.shear : o.shear;
    const hinged = o.hinge != null;
    const bend = o.bend === undefined ? (hinged ? null : s.bend) : o.bend, warpBend = o.warpBend === undefined ? bend : o.warpBend;
    const first = this.tris.length;
    for (let c = 0; c < cols; c++) {
      const chain = [];
      for (let r = 0; r < rows; r++) {
        const p = ok(c, r);
        if (p < 0) continue;
        chain.push(p);
        link(p, ok(c, r + 1), s.stretch, 0);
        link(p, ok(c + 1, r), weft, o.weftSlack || 0);
        link(p, ok(c + 1, r + 1), shear, o.weftSlack || 0);
        link(ok(c + 1, r), ok(c, r + 1), shear, o.weftSlack || 0);
        link(p, ok(c, r + 2), warpBend, 0);
        link(p, ok(c + 2, r), bend, o.weftSlack || 0);
        // the quad's two triangles, split on alternate diagonals so the cloth has no grain to fold along
        const q = ok(c + 1, r), u = ok(c, r + 1), v = ok(c + 1, r + 1);
        if (q >= 0 && u >= 0 && v >= 0) {
          if ((c + r) & 1) { this.triangle(p, u, v); this.triangle(p, v, q); }
          else { this.triangle(p, u, q); this.triangle(q, u, v); }
        }
      }
      if (o.hold && chain.length > 1) {
        const x = this.pos;
        chain.len = chain.slice(1).map((i, k) => Math.hypot(x[3 * i] - x[3 * chain[k]], x[3 * i + 1] - x[3 * chain[k] + 1], x[3 * i + 2] - x[3 * chain[k] + 2]));
        this.chains.push(chain);
      }
    }
    if (hinged) this._hingeTriangles(first, o.hinge, o.flat);
  }
  // An authored crease: the hinge on the edge i-j rests at `angle` (radians; the sign is the side it folds to).
  // o.compliance: how firmly it holds the angle (1e-4); o.soften: that times this; o.set: false keeps it from
  // creeping; o.stiffAt: the share of stiffen()'s bend its two points keep (0.02), so the cloth folds there
  crease(i, j, angle, o = {}) {
    const k = this._edgeHinge.get(this._key(i, j));
    if (k === undefined) return false;
    const H = this.hinges;
    // (the patches of stiff cloth stop at the crease's points: see _patches)
    const cp = this._creasePoints || (this._creasePoints = new Set());
    cp.add(i); cp.add(j);
    if (this.stiff) { this.stiff.scale[i] = this.stiff.scale[j] = o.stiffAt ?? 0.02; this.stiff._h = null; }
    H.rest[k] = angle;
    // (a pressed crease is firm: its hinge's compliance is o.compliance, 1e-4 unless given; o.soften scales it)
    H.comp[k] = o.compliance ?? 1e-4;
    if (o.soften) H.comp[k] *= o.soften;
    if (o.set === false) H.plastic[k] = 0;
    if (this._packed) { this._packed.hRest[k] = H.rest[k]; this._packed.hComp[k] = H.comp[k]; this._packed.hPlastic[k] = H.plastic[k]; }
    return true;
  }
  // a crease along a line of points (each edge of it in turn); returns how many edges it found
  fold(line, angle, o = {}) {
    let k = 0;
    for (let q = 1; q < line.length; q++) k += this.crease(line[q - 1], line[q], angle, o) ? 1 : 0;
    return k;
  }
  // the angle at a hinge (0 flat, signed)
  _angle(a, b, c, d, x = this.pos) {
    const A = 3 * a, B = 3 * b, C = 3 * c, D = 3 * d;
    const ex = x[D] - x[C], ey = x[D + 1] - x[C + 1], ez = x[D + 2] - x[C + 2];
    const n1 = cross(x[A] - x[C], x[A + 1] - x[C + 1], x[A + 2] - x[C + 2], x[A] - x[D], x[A + 1] - x[D + 1], x[A + 2] - x[D + 2]);
    const n2 = cross(x[B] - x[D], x[B + 1] - x[D + 1], x[B + 2] - x[D + 2], x[B] - x[C], x[B + 1] - x[C + 1], x[B + 2] - x[C + 2]);
    const el = Math.hypot(ex, ey, ez) || 1;
    const m = cross(n1[0], n1[1], n1[2], n2[0], n2[1], n2[2]);
    return Math.atan2((m[0] * ex + m[1] * ey + m[2] * ez) / el, n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]);
  }
  // Stiffness: the cloth's bend solved as one implicit solve a step over the whole sheet (fabric-stiff.js), for
  // cloth too stiff for the constraints' walks to hold up (leather, felt, a boot shaft). Call after the triangles
  // (sheet or mesh), the pins and weigh(). o: { bendingLength (units: the cantilever's, Peirce 1930) or B (the
  // flexural rigidity), density (as weigh's; else the cloth's mean), setRadius (units: bent tighter than this it
  // takes a set), creep (1/s), scale (Float64Array: each point's share of the stiffness, for a piece of two
  // weights of leather), passes (solves a step, 2), stretchK (the edges' hold in the solve; 0, or 30 for leather) }. Points of creases made before or after keep only o.stiffAt of the stiffness.
  stiffen(o = {}) {
    this._areas();
    let sigma = o.density;
    if (!sigma) { let m = 0, a = 0; for (let i = 0; i < this.n; i++) { m += this.kg ? this.kg[i] : 1; a += this.area[i]; } sigma = m / (a || 1); }
    const g = Math.hypot(...this.settings.gravity);
    const B = o.B ?? sigma * g * Math.pow(o.bendingLength, 3);
    this.stiff = new StiffBend(this, B, { setRadius: o.setRadius ?? this.settings.setRadius, creep: o.creep ?? this.settings.creep, scale: o.scale, passes: o.passes, stretchK: o.stretchK ?? this.settings.stretchK ?? 0 });
    if (this._creasePoints) for (const i of this._creasePoints) this.stiff.scale[i] = o.stiffAt ?? 0.02;
    return B;
  }
  // Weight: each point weighs the cloth round it, density (mass per square unit) times a third of each triangle it
  // is a corner of; o.extra: [point, mass] pairs added on (a buckle, a hem weight). Pins stay pinned.
  weigh(density, o = {}) {
    this._areas();
    const n = this.n, kg = new Float64Array(n);
    for (let i = 0; i < n; i++) kg[i] = density * this.area[i];
    for (const [i, m] of o.extra || []) kg[i] += m;
    this.massKg(kg);
  }
  // each point's mass, given (kg, or any unit the air's `air` is in)
  massKg(kg) {
    const n = this.n, w = this.w;
    let sum = 0, cnt = 0;
    for (let i = 0; i < n; i++) if (w[i] !== 0 && kg[i] > 0) { sum += kg[i]; cnt++; }
    const mean = cnt ? sum / cnt : 1;
    for (let i = 0; i < n; i++) if (w[i] !== 0) w[i] = kg[i] > 0 ? mean / kg[i] : 1;
    this.kg = kg;
  }
  _areas() {
    const x = this.pos, t = this.tris, ar = this.area.fill(0);
    for (let k = 0; k < t.length; k += 3) {
      const a = 3 * t[k], b = 3 * t[k + 1], c = 3 * t[k + 2];
      const m = cross(x[b] - x[a], x[b + 1] - x[a + 1], x[b + 2] - x[a + 2], x[c] - x[a], x[c + 1] - x[a + 1], x[c + 2] - x[a + 2]);
      const A = Math.hypot(m[0], m[1], m[2]) / 6;
      ar[t[k]] += A; ar[t[k + 1]] += A; ar[t[k + 2]] += A;
    }
  }
  // Tethers: each free point to the nearest pin along the cloth (through its ties of no slack), at that distance;
  // it may come closer, never go further. Call after pinning and tying.
  tether() {
    const n = this.n, w = this.w, adj = Array.from({ length: n }, () => []);
    for (let k = 0; k < this.ci.length; k++) {
      if (this.slack[k]) continue;
      adj[this.ci[k]].push(this.cj[k], this.rest[k]); adj[this.cj[k]].push(this.ci[k], this.rest[k]);
    }
    const dist = new Float64Array(n).fill(Infinity), from = new Int32Array(n).fill(-1), done = new Uint8Array(n);
    // (Dijkstra from every pin at once; n is a few thousand at most, so a plain scan for the nearest will do...
    // but a binary heap keeps a big garment quick)
    const heap = [];
    const push = (d, i) => { heap.push([d, i]); let k = heap.length - 1; while (k) { const p = (k - 1) >> 1; if (heap[p][0] <= heap[k][0]) break; [heap[p], heap[k]] = [heap[k], heap[p]]; k = p; } };
    const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let k = 0; for (;;) { const l = 2 * k + 1, r = l + 1; let m = k; if (l < heap.length && heap[l][0] < heap[m][0]) m = l; if (r < heap.length && heap[r][0] < heap[m][0]) m = r; if (m === k) break; [heap[m], heap[k]] = [heap[k], heap[m]]; k = m; } } return top; };
    for (let i = 0; i < n; i++) if (w[i] === 0) { dist[i] = 0; from[i] = i; push(0, i); }
    while (heap.length) {
      const [d, i] = pop();
      if (done[i]) continue;
      done[i] = 1;
      const a = adj[i];
      for (let q = 0; q < a.length; q += 2) {
        const j = a[q], nd = d + a[q + 1];
        if (nd < dist[j]) { dist[j] = nd; from[j] = from[i]; push(nd, j); }
      }
    }
    const ti = [], tp = [], tl = [];
    for (let i = 0; i < n; i++) if (w[i] !== 0 && from[i] >= 0) { ti.push(i); tp.push(from[i]); tl.push(dist[i]); }
    this.tethers = { i: Int32Array.from(ti), p: Int32Array.from(tp), len: Float64Array.from(tl) };
  }
  // Put the constraints in order from the pins out (each by its point nearer a pin, along the ties), so that one
  // walk over them carries the pins' hold from the root to the free edge, as a stiff cloth needs: walked in the
  // order they were made, a hold travels one row a walk, and leather or a boot shaft sags as if it were soft.
  _orderFromPins() {
    const n = this.n, w = this.w, adj = Array.from({ length: n }, () => []);
    for (let k = 0; k < this.ci.length; k++) { adj[this.ci[k]].push(this.cj[k]); adj[this.cj[k]].push(this.ci[k]); }
    const depth = new Int32Array(n).fill(n), queue = [];
    for (let i = 0; i < n; i++) if (w[i] === 0) { depth[i] = 0; queue.push(i); }
    if (!queue.length) return;
    for (let q = 0; q < queue.length; q++) for (const j of adj[queue[q]]) if (depth[j] > depth[queue[q]] + 1) { depth[j] = depth[queue[q]] + 1; queue.push(j); }
    const permute = (lists, key) => {
      const idx = key.map((v, k) => k).sort((a, b) => key[a] - key[b] || a - b);
      for (const L of lists) { const c = L.slice(); idx.forEach((from, to) => { L[to] = c[from]; }); }
    };
    permute([this.ci, this.cj, this.rest, this.comp, this.slack], this.ci.map((i, k) => Math.min(depth[i], depth[this.cj[k]]) * 2 + Math.max(depth[i], depth[this.cj[k]])));
    const H = this.hinges;
    permute([H.a, H.b, H.c, H.d, H.rest, H.comp, H.plastic], H.c.map((c, k) => Math.min(depth[c], depth[H.d[k]]) * 2 + Math.max(depth[H.a[k]], depth[H.b[k]])));
    this._edgeHinge.clear();
    H.c.forEach((c, k) => this._edgeHinge.set(this._key(c, H.d[k]), k));
    this.lambda = null;
  }
  // the constraints in typed arrays, for the step's loops (built again when any is added)
  _pack() {
    if (this.settings.order === 'root') this._orderFromPins();
    const H = this.hinges, n = this.n;
    const P = {
      ci: Int32Array.from(this.ci), cj: Int32Array.from(this.cj), rest: Float64Array.from(this.rest),
      comp: Float64Array.from(this.comp), slack: Float64Array.from(this.slack),
      ha: Int32Array.from(H.a), hb: Int32Array.from(H.b), hc: Int32Array.from(H.c), hd: Int32Array.from(H.d),
      hRest: Float64Array.from(H.rest), hComp: Float64Array.from(H.comp), hPlastic: Uint8Array.from(H.plastic),
      hScale: new Float64Array(H.a.length), hLam: new Float64Array(H.a.length), hAngle: new Float64Array(H.a.length),
    };
    // (a hinge's stiffness grows with its edge's length over its triangles' area, as the bending energy of a
    // discrete shell does, so a crease costs the same on a fine mesh as on a coarse one)
    const x = this.pos;
    for (let k = 0; k < P.ha.length; k++) {
      const a = 3 * P.ha[k], b = 3 * P.hb[k], c = 3 * P.hc[k], d = 3 * P.hd[k];
      const el2 = (x[d] - x[c]) ** 2 + (x[d + 1] - x[c + 1]) ** 2 + (x[d + 2] - x[c + 2]) ** 2;
      const n1 = cross(x[a] - x[c], x[a + 1] - x[c + 1], x[a + 2] - x[c + 2], x[a] - x[d], x[a + 1] - x[d + 1], x[a + 2] - x[d + 2]);
      const n2 = cross(x[b] - x[d], x[b + 1] - x[d + 1], x[b + 2] - x[d + 2], x[b] - x[c], x[b + 1] - x[c + 1], x[b + 2] - x[c + 2]);
      const area = (Math.hypot(...n1) + Math.hypot(...n2)) / 2;
      P.hScale[k] = el2 > 1e-12 ? area / (3 * el2) : 1;
    }
    if (this.settings.thickness > 0) {
      P.near = new Set();
      for (let k = 0; k < P.ci.length; k++) P.near.add(this._key(P.ci[k], P.cj[k]));
      const t = this.tris;
      for (let k = 0; k < t.length; k += 3) { P.near.add(this._key(t[k], t[k + 1])); P.near.add(this._key(t[k + 1], t[k + 2])); P.near.add(this._key(t[k], t[k + 2])); }
      P.hashSize = 2 * n + 1; P.cellStart = new Int32Array(P.hashSize + 1); P.cellEntry = new Int32Array(n);
    }
    if (this.tris.length) this._areas();
    this.lambda = new Float64Array(P.ci.length);
    this._packed = P;
  }
  // One step of h seconds. body(p, i): pushes the point p ([x, y, z], changed in place) out of whatever it must
  // stay out of and returns true if it moved it; o.neutral: a push by the body is not kept as speed (the cloth
  // comes to rest on the body instead of bouncing off it)
  step(h, body = null, o = {}) {
    if (!this._packed) this._pack();
    const P = this._packed, s = this.settings, n = this.n, x = this.pos, pv = this.prev, w = this.w, g = s.gravity;
    const keep = Math.pow(1 - s.damping, h * 60), m = P.ci.length;
    if (s.air && this.tris.length) this._air(h);
    for (let i = 0; i < n; i++) {
      if (w[i] === 0) continue;
      const a = 3 * i;
      for (let c = 0; c < 3; c++) {
        const v = (x[a + c] - pv[a + c]) * keep;
        pv[a + c] = x[a + c];
        x[a + c] += v + g[c] * h * h;
      }
    }
    if (this.stiff) this.stiff.step(h);
    if (!this.lambda || this.lambda.length !== m) this.lambda = new Float64Array(m);
    const lam = this.lambda.fill(0), ci = P.ci, cj = P.cj, rest = P.rest, comp = P.comp, slack = P.slack, h2 = h * h;
    if (P.ha.length) P.hLam.fill(0);
    const each = body && s.contact === 'iteration';
    // (the walk runs forward one time and backward the next: always the same way, the points solved last get the
    // last word, and the whole sheet creeps that way. Every step walks in the same order, though: two steps that
    // walk differently come to two different answers, and the sheet flickers between them)
    for (let it = 0; it < s.iterations; it++) {
      const back = it & 1;
      for (let kk = 0; kk < m; kk++) {
        const k = back ? m - 1 - kk : kk;
        const i = ci[k], j = cj[k], a = 3 * i, b = 3 * j, wi = w[i], wj = w[j];
        const dx = x[b] - x[a], dy = x[b + 1] - x[a + 1], dz = x[b + 2] - x[a + 2], len = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (len < 1e-9) continue;
        let target = rest[k];
        if (slack[k]) { target = Math.min(rest[k] * (1 + slack[k]), Math.max(rest[k] * (1 - slack[k]), len)); if (target === len) continue; }
        const alpha = comp[k] / h2, C = len - target;
        const dl = (-C - alpha * lam[k]) / (wi + wj + alpha);
        lam[k] += dl;
        const q = dl / len;
        x[a] -= wi * q * dx; x[a + 1] -= wi * q * dy; x[a + 2] -= wi * q * dz;
        x[b] += wj * q * dx; x[b + 1] += wj * q * dy; x[b + 2] += wj * q * dz;
      }
      if (P.ha.length) this._bend(h2, back);
      if (this.tethers) this._tether();
      if (each && it < s.iterations - 1) this._contact(body, o, false);
    }
    if (P.ha.length && s.creep > 0) this._set(h);
    if (this.stiff) this.stiff.set(h);
    if (s.thickness > 0) this._self();
    this.hold();
    if (body) {
      this._contact(body, o, true);
      if (this.chains.length) this.hold(body, o.neutral);
    }
  }
  // the hinges, walked once: C = angle - rest, moved along the angle's gradient at each of the four points
  _bend(h2, back) {
    const P = this._packed, x = this.pos, w = this.w, ha = P.ha, hb = P.hb, hc = P.hc, hd = P.hd, mh = ha.length;
    for (let kk = 0; kk < mh; kk++) {
      const k = back ? mh - 1 - kk : kk;
      const ia = ha[k], ib = hb[k], ic = hc[k], id = hd[k], wa = w[ia], wb = w[ib], wc = w[ic], wd = w[id];
      if (wa + wb + wc + wd === 0) continue;
      const A = 3 * ia, B = 3 * ib, C = 3 * ic, D = 3 * id;
      const ex = x[D] - x[C], ey = x[D + 1] - x[C + 1], ez = x[D + 2] - x[C + 2], el = Math.sqrt(ex * ex + ey * ey + ez * ez);
      if (el < 1e-9) continue;
      // n1 = (a - c) x (a - d), n2 = (b - d) x (b - c)
      const acx = x[A] - x[C], acy = x[A + 1] - x[C + 1], acz = x[A + 2] - x[C + 2];
      const adx = x[A] - x[D], ady = x[A + 1] - x[D + 1], adz = x[A + 2] - x[D + 2];
      const bcx = x[B] - x[C], bcy = x[B + 1] - x[C + 1], bcz = x[B + 2] - x[C + 2];
      const bdx = x[B] - x[D], bdy = x[B + 1] - x[D + 1], bdz = x[B + 2] - x[D + 2];
      const n1x = acy * adz - acz * ady, n1y = acz * adx - acx * adz, n1z = acx * ady - acy * adx;
      const n2x = bdy * bcz - bdz * bcy, n2y = bdz * bcx - bdx * bcz, n2z = bdx * bcy - bdy * bcx;
      const l1 = n1x * n1x + n1y * n1y + n1z * n1z, l2 = n2x * n2x + n2y * n2y + n2z * n2z;
      // (a triangle squashed to a sliver has no plane to turn: its gradient runs off to infinity)
      const e4 = 1e-6 * el * el * el * el;
      if (l1 < e4 || l2 < e4) continue;
      const mx = n1y * n2z - n1z * n2y, my = n1z * n2x - n1x * n2z, mz = n1x * n2y - n1y * n2x;
      const theta = Math.atan2((mx * ex + my * ey + mz * ez) / el, n1x * n2x + n1y * n2y + n1z * n2z);
      P.hAngle[k] = theta;
      const Cv = wrapAngle(theta - P.hRest[k]);
      // the gradients (Bridson et al. 2003)
      const f1 = el / l1, f2 = el / l2;
      const s3a = (adx * ex + ady * ey + adz * ez) / el, s3b = (bdx * ex + bdy * ey + bdz * ez) / el;
      const s4a = (acx * ex + acy * ey + acz * ez) / el, s4b = (bcx * ex + bcy * ey + bcz * ez) / el;
      const u1x = f1 * n1x, u1y = f1 * n1y, u1z = f1 * n1z;
      const u2x = f2 * n2x, u2y = f2 * n2y, u2z = f2 * n2z;
      const g1 = s3a / el, g2 = s3b / el, g3 = s4a / el, g4 = s4b / el;
      const u3x = g1 * u1x + g2 * u2x, u3y = g1 * u1y + g2 * u2y, u3z = g1 * u1z + g2 * u2z;
      const u4x = -g3 * u1x - g4 * u2x, u4y = -g3 * u1y - g4 * u2y, u4z = -g3 * u1z - g4 * u2z;
      const sum = wa * (u1x * u1x + u1y * u1y + u1z * u1z) + wb * (u2x * u2x + u2y * u2y + u2z * u2z)
        + wc * (u3x * u3x + u3y * u3y + u3z * u3z) + wd * (u4x * u4x + u4y * u4y + u4z * u4z);
      const alpha = P.hComp[k] * P.hScale[k] / h2;
      if (sum < 1e-18) continue;
      const dl = (-Cv - alpha * P.hLam[k]) / (sum + alpha);
      P.hLam[k] += dl;
      // (the angle as atan2 above measures it falls along u)
      const q = -dl;
      x[A] += wa * q * u1x; x[A + 1] += wa * q * u1y; x[A + 2] += wa * q * u1z;
      x[B] += wb * q * u2x; x[B + 1] += wb * q * u2y; x[B + 2] += wb * q * u2z;
      x[C] += wc * q * u3x; x[C + 1] += wc * q * u3y; x[C + 2] += wc * q * u3z;
      x[D] += wd * q * u4x; x[D + 1] += wd * q * u4y; x[D + 2] += wd * q * u4z;
    }
  }
  // the set: a hinge held past its rest by more than `yield` moves its rest toward where it is held
  _set(h) {
    const P = this._packed, s = this.settings, k0 = Math.min(1, s.creep * h), y = s.yield, mx = s.setMax;
    for (let k = 0; k < P.ha.length; k++) {
      if (!P.hPlastic[k]) continue;
      const e = wrapAngle(P.hAngle[k] - P.hRest[k]), over = Math.abs(e) - y;
      if (over <= 0) continue;
      const r = P.hRest[k] + Math.sign(e) * over * k0;
      P.hRest[k] = Math.max(-mx, Math.min(mx, r));
    }
  }
  // the tethers: a point further from its pin than its distance along the cloth is put back on that distance
  _tether() {
    const T = this.tethers, x = this.pos, sl = 1 + this.settings.tetherSlack;
    for (let k = 0; k < T.i.length; k++) {
      const a = 3 * T.i[k], b = 3 * T.p[k];
      const dx = x[a] - x[b], dy = x[a + 1] - x[b + 1], dz = x[a + 2] - x[b + 2], d = Math.sqrt(dx * dx + dy * dy + dz * dz), L = T.len[k] * sl;
      if (d <= L) continue;
      const q = L / d;
      x[a] = x[b] + dx * q; x[a + 1] = x[b + 1] + dy * q; x[a + 2] = x[b + 2] + dz * q;
    }
  }
  // the air: drag on each triangle across the flow (the triangle's mean speed through the air along its own
  // normal), shared to its corners by area, as speed lost this step, from the previous position. (Per triangle, not
  // per point: at a pressed fold a point's two faces point opposite ways, their summed normal turns into the cloth's
  // own plane, and a point's drag along it pulled the folds apart; a face's push, -|vn| vn n, is the same whichever
  // way its normal points)
  _air(h) {
    const x = this.pos, pv = this.prev, t = this.tris, n = this.n, s = this.settings, wd = s.wind, w = this.w;
    const acc = this._airAcc || (this._airAcc = new Float64Array(3 * n)), ar = this.area, kg = this.kg;
    acc.fill(0);
    for (let k = 0; k < t.length; k += 3) {
      const i0 = t[k], i1 = t[k + 1], i2 = t[k + 2], a = 3 * i0, b = 3 * i1, c = 3 * i2;
      const m = cross(x[b] - x[a], x[b + 1] - x[a + 1], x[b + 2] - x[a + 2], x[c] - x[a], x[c + 1] - x[a + 1], x[c + 2] - x[a + 2]);
      const ml = Math.hypot(m[0], m[1], m[2]);
      if (ml < 1e-12) continue;
      const ux = m[0] / ml, uy = m[1] / ml, uz = m[2] / ml, A = ml / 6;
      const vx = (x[a] + x[b] + x[c] - pv[a] - pv[b] - pv[c]) / (3 * h) - wd[0];
      const vy = (x[a + 1] + x[b + 1] + x[c + 1] - pv[a + 1] - pv[b + 1] - pv[c + 1]) / (3 * h) - wd[1];
      const vz = (x[a + 2] + x[b + 2] + x[c + 2] - pv[a + 2] - pv[b + 2] - pv[c + 2]) / (3 * h) - wd[2];
      const vn = vx * ux + vy * uy + vz * uz;
      // (the face's mass per area, from its corners'; a = air / that * vn^2, never past stopping the face)
      const sig = kg ? (kg[i0] + kg[i1] + kg[i2]) / ((ar[i0] + ar[i1] + ar[i2]) || 1) : 1;
      const dv = Math.min(Math.abs(vn), s.air / (sig || 1) * vn * vn * h) * Math.sign(vn) * A;
      for (const p of [a, b, c]) { acc[p] += dv * ux; acc[p + 1] += dv * uy; acc[p + 2] += dv * uz; }
    }
    for (let i = 0; i < n; i++) {
      if (w[i] === 0 || !ar[i]) continue;
      const a = 3 * i, q = h / ar[i];
      pv[a] += acc[a] * q; pv[a + 1] += acc[a + 1] * q; pv[a + 2] += acc[a + 2] * q;
    }
  }
  // the sheet against itself: points closer than `thickness` that are not tied together are pushed apart
  // (a spatial hash, Teschner et al. 2003, of cells the thickness across)
  _self() {
    const P = this._packed, x = this.pos, w = this.w, n = this.n, t = this.settings.thickness, size = P.hashSize;
    const start = P.cellStart.fill(0), entry = P.cellEntry, inv = 1 / t;
    const hash = (xi, yi, zi) => Math.abs((xi * 92837111) ^ (yi * 689287499) ^ (zi * 283923481)) % size;
    const cell = new Int32Array(n);
    for (let i = 0; i < n; i++) { const a = 3 * i; cell[i] = hash(Math.floor(x[a] * inv), Math.floor(x[a + 1] * inv), Math.floor(x[a + 2] * inv)); start[cell[i]]++; }
    for (let c = 1; c <= size; c++) start[c] += start[c - 1];
    for (let i = 0; i < n; i++) entry[--start[cell[i]]] = i;
    const t2 = t * t;
    for (let i = 0; i < n; i++) {
      const a = 3 * i, xi = Math.floor(x[a] * inv), yi = Math.floor(x[a + 1] * inv), zi = Math.floor(x[a + 2] * inv);
      for (let ox = -1; ox <= 1; ox++) for (let oy = -1; oy <= 1; oy++) for (let oz = -1; oz <= 1; oz++) {
        const hc = hash(xi + ox, yi + oy, zi + oz);
        for (let e = start[hc]; e < start[hc + 1]; e++) {
          const j = entry[e];
          if (j <= i) continue;
          const b = 3 * j, dx = x[b] - x[a], dy = x[b + 1] - x[a + 1], dz = x[b + 2] - x[a + 2], d2 = dx * dx + dy * dy + dz * dz;
          if (d2 >= t2 || d2 < 1e-18) continue;
          if (P.near.has(this._key(i, j))) continue;
          const ws = w[i] + w[j];
          if (!ws) continue;
          const d = Math.sqrt(d2), q = (t - d) / d / ws;
          x[a] -= w[i] * q * dx; x[a + 1] -= w[i] * q * dy; x[a + 2] -= w[i] * q * dz;
          x[b] += w[j] * q * dx; x[b + 1] += w[j] * q * dy; x[b + 2] += w[j] * q * dz;
        }
      }
    }
  }
  // the body against the middle of each edge of the cloth, so a limb cannot pass between two points by pushing
  // them apart; the push is shared by the edge's two ends (more to the lighter), and not kept as speed. An edge
  // whose two ends both lie on the body is left alone: its middle is inside any curved body by the chord's sag,
  // and pushing it out every step slid a sheet clean off the bench's ball
  _edgeContact(body, o) {
    const P = this._packed;
    if (!P.ea) {
      const t = this.tris, seen = new Set(), ea = [], eb = [];
      for (let k = 0; k < t.length; k += 3) for (let e = 0; e < 3; e++) {
        const i = t[k + e], j = t[k + (e + 1) % 3], key = this._key(i, j);
        if (!seen.has(key)) { seen.add(key); ea.push(i); eb.push(j); }
      }
      P.ea = Int32Array.from(ea); P.eb = Int32Array.from(eb);
    }
    const x = this.pos, pv = this.prev, w = this.w, p = [0, 0, 0], ref = [0, 0, 0];
    for (let k = 0; k < P.ea.length; k++) {
      const i = P.ea[k], j = P.eb[k], a = 3 * i, b = 3 * j, wi = w[i], wj = w[j], ws = wi + wj;
      if (!ws || (this.touch[i] && this.touch[j])) continue;
      for (let c = 0; c < 3; c++) { p[c] = 0.5 * (x[a + c] + x[b + c]); ref[c] = 0.5 * (pv[a + c] + pv[b + c]); }
      const m0 = p[0], m1 = p[1], m2 = p[2];
      if (!body(p, -1, ref)) continue;
      // (moving the ends by d * 2 w / ws moves the middle by d)
      const dx = p[0] - m0, dy = p[1] - m1, dz = p[2] - m2, ki = 2 * wi / ws, kj = 2 * wj / ws;
      x[a] += ki * dx; x[a + 1] += ki * dy; x[a + 2] += ki * dz; x[b] += kj * dx; x[b + 1] += kj * dy; x[b + 2] += kj * dz;
      if (o.neutral) { pv[a] += ki * dx; pv[a + 1] += ki * dy; pv[a + 2] += ki * dz; pv[b] += kj * dx; pv[b + 1] += kj * dy; pv[b + 2] += kj * dz; }
    }
  }
  // the body: each point pushed out, and (on the last test of a step) its slide along the body held back
  _contact(body, o, last) {
    const n = this.n, x = this.pos, pv = this.prev, w = this.w, p = this._p || (this._p = [0, 0, 0]), mu = this.settings.friction, touch = this.touch;
    const ref = this._ref || (this._ref = [0, 0, 0]);
    for (let i = 0; i < n; i++) {
      if (w[i] === 0) continue;
      const a = 3 * i;
      p[0] = x[a]; p[1] = x[a + 1]; p[2] = x[a + 2];
      ref[0] = pv[a]; ref[1] = pv[a + 1]; ref[2] = pv[a + 2];
      const hit = body(p, i, ref) ? 1 : 0;
      if (last) touch[i] = hit;
      if (!hit) continue;
      // the push, and along the body the slide this step: part of the slide is lost
      const nx = p[0] - x[a], ny = p[1] - x[a + 1], nz = p[2] - x[a + 2], nl = Math.sqrt(nx * nx + ny * ny + nz * nz);
      if (o.neutral) { pv[a] += nx; pv[a + 1] += ny; pv[a + 2] += nz; }
      if (last && mu && nl > 1e-12) {
        const ux = nx / nl, uy = ny / nl, uz = nz / nl;
        let sx = p[0] - pv[a], sy = p[1] - pv[a + 1], sz = p[2] - pv[a + 2];
        const sn = sx * ux + sy * uy + sz * uz;
        sx -= sn * ux; sy -= sn * uy; sz -= sn * uz;
        // (Coulomb: a slide smaller than the push times the friction is stopped outright, so cloth lies still on
        // a slope; a larger one loses that much)
        const hold = mu * nl;
        const sl = Math.sqrt(sx * sx + sy * sy + sz * sz), k = sl <= hold ? 1 : hold / sl;
        p[0] -= k * sx; p[1] -= k * sy; p[2] -= k * sz;
      }
      x[a] = p[0]; x[a + 1] = p[1]; x[a + 2] = p[2];
    }
    if (last && this.settings.edgeContact) this._edgeContact(body, o);
  }
  // put every held chain's points back at their segment lengths, from the top down (follow the leader), so a
  // bone or a strand is exactly its length whatever the constraints left; where the body is given, out of it again;
  // still: the moves are not kept as speed (after the caller has moved points itself)
  hold(body = null, neutral = false, still = false) {
    const x = this.pos, pv = this.prev, w = this.w, p = [0, 0, 0], lead = this.settings.lead;
    for (const ch of this.chains) {
      for (let k = 1; k < ch.length; k++) {
        const i = ch[k], a = 3 * i, b = 3 * ch[k - 1];
        if (w[i] === 0) continue;
        const dx = x[a] - x[b], dy = x[a + 1] - x[b + 1], dz = x[a + 2] - x[b + 2], len = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1, q = ch.len[k - 1] / len;
        p[0] = x[b] + dx * q; p[1] = x[b + 1] + dy * q; p[2] = x[b + 2] + dz * q;
        // (the move becomes this point's speed, as any constraint's does, and the point above takes most of it
        // the other way: follow the leader alone moves only the lower point, as if the upper one had no end of
        // mass, and that feeds the chain energy. Mueller et al. 2012, "Fast simulation of inextensible hair and fur")
        if (still) { pv[a] += p[0] - x[a]; pv[a + 1] += p[1] - x[a + 1]; pv[a + 2] += p[2] - x[a + 2]; }
        else if (w[ch[k - 1]] !== 0) { pv[b] += lead * (p[0] - x[a]); pv[b + 1] += lead * (p[1] - x[a + 1]); pv[b + 2] += lead * (p[2] - x[a + 2]); }
        if (body) { const bx = p[0], by = p[1], bz = p[2]; if (body(p, i) && neutral) { pv[a] += p[0] - bx; pv[a + 1] += p[1] - by; pv[a + 2] += p[2] - bz; } }
        x[a] = p[0]; x[a + 1] = p[1]; x[a + 2] = p[2];
      }
    }
  }
  // A frame of dt seconds in fixed steps of settings.substep (the rest carried to the next frame).
  // o.before(t): called before each step with the time it ends at, as a share of the frame (0..1], so the caller
  // can move the pins and the body there; o.body, o.neutral as step's. Returns the steps taken.
  advance(dt, o = {}) {
    const h = this.settings.substep, owed = this._acc;
    this._acc += dt;
    let k = 0;
    const steps = Math.floor(this._acc / h + 1e-9);
    for (; k < steps; k++) {
      if (o.before) o.before(dt > 0 ? ((k + 1) * h - owed) / dt : 1, h);
      this.step(h, o.body || null, o);
    }
    this._acc -= steps * h;
    return k;
  }
  // Drape: run to rest. Steps until the fastest point moves less than `tol` a step (or `max` steps), with the
  // speed damped hard so it settles instead of swinging. Returns the steps taken.
  settle(body, o = {}) {
    const h = o.h || 1 / 120, max = o.max || 600, tol = o.tol || 1e-4, s = this.settings, was = s.damping;
    s.damping = o.damping ?? 0.35;
    let k = 0;
    for (; k < max; k++) {
      this.step(h, body, { neutral: true });
      if (k % 10 === 9) {
        let far = 0;
        const x = this.pos, pv = this.prev;
        for (let a = 0; a < x.length; a++) far = Math.max(far, Math.abs(x[a] - pv[a]));
        if (far < tol) break;
      }
    }
    s.damping = was;
    this.prev.set(this.pos);
    return k;
  }
  // The cloth's state (positions, speeds and the sets its hinges have taken), to keep a drape and start from it
  // again without solving it again (restore). JSON-safe with Array.from on each.
  snapshot() {
    if (!this._packed) this._pack();
    return { pos: Float64Array.from(this.pos), prev: Float64Array.from(this.prev), hRest: Float64Array.from(this._packed.hRest) };
  }
  restore(st) {
    this.pos.set(st.pos); this.prev.set(st.prev ?? st.pos);
    if (!this._packed) this._pack();
    if (st.hRest && st.hRest.length === this._packed.hRest.length) { this._packed.hRest.set(st.hRest); this.hinges.rest = Array.from(st.hRest); }
  }
  // the angle each hinge is at now and rests at (for creases: where the cloth has taken a set)
  hingeAngles() {
    if (!this._packed) this._pack();
    const P = this._packed, out = new Float64Array(P.ha.length), rest = Float64Array.from(P.hRest);
    for (let k = 0; k < P.ha.length; k++) out[k] = this._angle(P.ha[k], P.hb[k], P.hc[k], P.hd[k]);
    return { angle: out, rest };
  }
  // move a point without giving it speed
  shift(i, dx, dy, dz) {
    const a = 3 * i;
    this.pos[a] += dx; this.pos[a + 1] += dy; this.pos[a + 2] += dz;
    this.prev[a] += dx; this.prev[a + 1] += dy; this.prev[a + 2] += dz;
  }
}

function cross(ax, ay, az, bx, by, bz) { return [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx]; }
