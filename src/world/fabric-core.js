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
// The method is XPBD (Macklin, Mueller and Chentanez 2016, "XPBD: position-based simulation of compliant
// constrained dynamics"), stepped in small fixed steps (Macklin et al. 2019, "Small steps in physics simulation"):
// each step moves every point by its speed and by gravity, then walks the constraints once or a few times,
// moving the two points of each toward its rest length. A constraint's `compliance` (how soft it is) is divided by
// the step squared, so the cloth is as stiff at 120 steps a second as at 600: stiffness is a property of the
// fabric, not of the step. After the constraints each point is pushed out of the body, and the body's hold on it
// takes away its slide along it (`friction`), so cloth and hair stay where they are laid on a shoulder instead of pouring off.
//
// Nothing here knows about three.js, bones or a page: positions are a flat array of x, y, z, the body is a
// function, and the units are the caller's. The cape and the hair in the game run it through fabric.js (on the
// same bones, colliders and bench as the spring bones); the Face Creator runs it once, to rest, to drape a
// hairstyle on the figure that will wear it.

export const FABRIC_DEFAULTS = {
  gravity: [0, -450, 0],     // units / s^2
  damping: 0.02,             // share of each point's speed lost per 1/60 s
  iterations: 2,             // walks over the constraints per step
  stretch: 0,                // compliance of the threads (0: they keep their length)
  shear: 2e-5,               // compliance of the diagonals
  bend: 2e-4,                // compliance of the ties two points along
  friction: 0.5,             // how well the body holds what lies on it: the slide it stops, as a share of its push
  lead: 0.9,                 // held chains: the share of a point's correction the point above takes the other way
};

export class Fabric {
  // n: the number of points; pos: Float64Array of 3n (taken over, not copied)
  constructor(pos, settings = {}) {
    this.n = pos.length / 3;
    this.pos = pos;
    this.prev = Float64Array.from(pos);
    this.w = new Float64Array(this.n).fill(1);       // inverse mass: 0 pins a point
    this.settings = { ...FABRIC_DEFAULTS, ...settings };
    this.ci = []; this.cj = []; this.rest = []; this.comp = []; this.slack = []; this.lambda = null;
    this.chains = [];        // lists of point indices whose segments keep their length exactly (see `hold`)
    this.touch = new Uint8Array(this.n);
  }
  pin(i) { this.w[i] = 0; }
  // tie points i and j at the distance they are at now (or `rest`); compliance 0 is rigid; slack: the share of
  // the rest length the distance may change freely
  tie(i, j, compliance = 0, slack = 0, rest = null) {
    if (this.w[i] === 0 && this.w[j] === 0) return;
    const p = this.pos;
    this.ci.push(i); this.cj.push(j); this.comp.push(compliance); this.slack.push(slack);
    this.rest.push(rest ?? Math.hypot(p[3 * i] - p[3 * j], p[3 * i + 1] - p[3 * j + 1], p[3 * i + 2] - p[3 * j + 2]));
    this.lambda = null;
  }
  // A sheet from a grid of point indices, at(c, r) for column c (a thread from the top down) and row r, giving -1
  // where there is no point. Ties the warp (down each column), the weft (across), the diagonals and the bend
  // ties, with the settings' compliances; o: { weftSlack, wrap (the last column meets the first), weft, shear,
  // bend, warpBend, hold (keep each column's segments at their exact length after every step: bones, hair) }
  sheet(cols, rows, at, o = {}) {
    const s = this.settings, ok = (c, r) => (r >= 0 && r < rows && (o.wrap || (c >= 0 && c < cols)) ? at(((c % cols) + cols) % cols, r) : -1);
    const link = (a, b, k, slack) => { if (a >= 0 && b >= 0 && k != null) this.tie(a, b, k, slack); };
    const weft = o.weft === undefined ? s.stretch : o.weft, shear = o.shear === undefined ? s.shear : o.shear;
    const bend = o.bend === undefined ? s.bend : o.bend, warpBend = o.warpBend === undefined ? bend : o.warpBend;
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
      }
      if (o.hold && chain.length > 1) {
        const x = this.pos;
        chain.len = chain.slice(1).map((i, k) => Math.hypot(x[3 * i] - x[3 * chain[k]], x[3 * i + 1] - x[3 * chain[k] + 1], x[3 * i + 2] - x[3 * chain[k] + 2]));
        this.chains.push(chain);
      }
    }
  }
  // One step of h seconds. body(p, i): pushes the point p ([x, y, z], changed in place) out of whatever it must
  // stay out of and returns true if it moved it; o.neutral: a push by the body is not kept as speed (the cloth
  // comes to rest on the body instead of bouncing off it)
  step(h, body = null, o = {}) {
    const s = this.settings, n = this.n, x = this.pos, pv = this.prev, w = this.w, g = s.gravity;
    const keep = Math.pow(1 - s.damping, h * 60), m = this.ci.length;
    for (let i = 0; i < n; i++) {
      if (w[i] === 0) continue;
      const a = 3 * i;
      for (let c = 0; c < 3; c++) {
        const v = (x[a + c] - pv[a + c]) * keep;
        pv[a + c] = x[a + c];
        x[a + c] += v + g[c] * h * h;
      }
    }
    if (!this.lambda || this.lambda.length !== m) this.lambda = new Float64Array(m);
    const lam = this.lambda.fill(0), ci = this.ci, cj = this.cj, rest = this.rest, comp = this.comp, slack = this.slack;
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
        const alpha = comp[k] / (h * h), C = len - target;
        const dl = (-C - alpha * lam[k]) / (wi + wj + alpha);
        lam[k] += dl;
        const q = dl / len;
        x[a] -= wi * q * dx; x[a + 1] -= wi * q * dy; x[a + 2] -= wi * q * dz;
        x[b] += wj * q * dx; x[b + 1] += wj * q * dy; x[b + 2] += wj * q * dz;
      }
    }
    this.hold();
    if (body) {
      const p = [0, 0, 0], mu = s.friction, touch = this.touch;
      for (let i = 0; i < n; i++) {
        if (w[i] === 0) continue;
        const a = 3 * i;
        p[0] = x[a]; p[1] = x[a + 1]; p[2] = x[a + 2];
        touch[i] = body(p, i) ? 1 : 0;
        if (!touch[i]) continue;
        // the push, and along the body the slide this step: part of the slide is lost
        const nx = p[0] - x[a], ny = p[1] - x[a + 1], nz = p[2] - x[a + 2], nl = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (o.neutral) { pv[a] += nx; pv[a + 1] += ny; pv[a + 2] += nz; }
        if (mu && nl > 1e-12) {
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
      if (this.chains.length) this.hold(body, o.neutral);
    }
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
  // move a point without giving it speed
  shift(i, dx, dy, dz) {
    const a = 3 * i;
    this.pos[a] += dx; this.pos[a + 1] += dy; this.pos[a + 2] += dz;
    this.prev[a] += dx; this.prev[a + 1] += dy; this.prev[a + 2] += dz;
  }
}
