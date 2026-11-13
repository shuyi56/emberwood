// Procedural pixel-art sprites for the Emberwood party, shared by the game and the sprite bench.
import { lerp, FW, FH, COLS, ROWS, WALK, ramp, RES, PW, PH, SKIN, CX, GY, HEAD, grain } from './core.js';
import { FEMALE_BODY } from './presets/female-body.js';
import { FEMALE_HEAD, drawFemaleHead } from './presets/female-head.js';
// ---------- Sprites ----------
// The party are pixel-art sprites painted in code. Each figure is posed in 3D (hips, knees, shoulders,
// hands, gear) and projected for eight facings, so every view is consistent and gear stays in the right
// hand from every side. Columns: two idle (breathing) frames, then an eight-step walk. 64x80 pixels a frame.

const BAYER = [[0, 0.5], [0.75, 0.25]];

// The painter works at RES times the design grid: shapes are still described in 64x80 design units,
// but they are rasterised, shaded, dithered and outlined on the finer grid, and dot() can place
// single fine pixels for small details.
function Painter() {
  const buf = new Uint8Array(PW * PH * 4), mask = new Uint8Array(PW * PH);
  const runTop = new Int16Array(PW * PH), runBot = new Int16Array(PW * PH);
  let bx0 = PW, by0 = PH, bx1 = -1, by1 = -1;
  const grow = (x0, y0, x1, y1) => { bx0 = Math.max(0, Math.min(bx0, x0)); by0 = Math.max(0, Math.min(by0, y0)); bx1 = Math.min(PW - 1, Math.max(bx1, x1)); by1 = Math.min(PH - 1, Math.max(by1, y1)); };
  const set = (k, c) => { buf[k * 4] = c[0]; buf[k * 4 + 1] = c[1]; buf[k * 4 + 2] = c[2]; buf[k * 4 + 3] = 255; };
  const P = {
    buf,
    e(cx, cy, rx, ry) {
      rx = Math.max(rx, 0.45); ry = Math.max(ry, 0.45);
      const X0 = Math.max(0, Math.floor((cx - rx) * RES)), X1 = Math.min(PW - 1, Math.ceil((cx + rx) * RES));
      const Y0 = Math.max(0, Math.floor((cy - ry) * RES)), Y1 = Math.min(PH - 1, Math.ceil((cy + ry) * RES));
      for (let Y = Y0; Y <= Y1; Y++) for (let X = X0; X <= X1; X++) {
        const dx = ((X + 0.5) / RES - cx) / rx, dy = ((Y + 0.5) / RES - cy) / ry;
        if (dx * dx + dy * dy <= 1) mask[Y * PW + X] = 1;
      }
      grow(X0, Y0, X1, Y1);
      return P;
    },
    p(pts) {
      let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
      pts.forEach(([x, y]) => { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); });
      const X0 = Math.max(0, Math.floor(x0 * RES)), X1 = Math.min(PW - 1, Math.ceil(x1 * RES));
      const Y0 = Math.max(0, Math.floor(y0 * RES)), Y1 = Math.min(PH - 1, Math.ceil(y1 * RES));
      for (let Y = Y0; Y <= Y1; Y++) {
        const py = (Y + 0.5) / RES;
        for (let X = X0; X <= X1; X++) {
          const px = (X + 0.5) / RES;
          let inside = false;
          for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
            const [xi, yi] = pts[i], [xj, yj] = pts[j];
            if ((yi > py) !== (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi) inside = !inside;
          }
          if (inside) mask[Y * PW + X] = 1;
        }
      }
      grow(X0, Y0, X1, Y1);
      return P;
    },
    // remove a polygon from the current mask (used to cut the face out of the hair)
    cut(pts) {
      let y0 = 1e9, y1 = -1e9;
      pts.forEach(([, y]) => { y0 = Math.min(y0, y); y1 = Math.max(y1, y); });
      const Y0 = Math.max(0, Math.floor(y0 * RES)), Y1 = Math.min(PH - 1, Math.ceil(y1 * RES));
      for (let Y = Y0; Y <= Y1; Y++) {
        const py = (Y + 0.5) / RES;
        for (let X = bx0; X <= bx1; X++) {
          const px = (X + 0.5) / RES;
          let inside = false;
          for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
            const [xi, yi] = pts[i], [xj, yj] = pts[j];
            if ((yi > py) !== (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi) inside = !inside;
          }
          if (inside) mask[Y * PW + X] = 0;
        }
      }
      return P;
    },
    l(x0, y0, x1, y1, t) {
      const vx = x1 - x0, vy = y1 - y0, len2 = vx * vx + vy * vy || 1, rad = Math.max(t / 2, 0.3);
      const X0 = Math.max(0, Math.floor((Math.min(x0, x1) - rad) * RES)), X1 = Math.min(PW - 1, Math.ceil((Math.max(x0, x1) + rad) * RES));
      const Y0 = Math.max(0, Math.floor((Math.min(y0, y1) - rad) * RES)), Y1 = Math.min(PH - 1, Math.ceil((Math.max(y0, y1) + rad) * RES));
      for (let Y = Y0; Y <= Y1; Y++) for (let X = X0; X <= X1; X++) {
        const px = (X + 0.5) / RES - x0, py = (Y + 0.5) / RES - y0;
        const u = Math.max(0, Math.min(1, (px * vx + py * vy) / len2));
        const dx = px - vx * u, dy = py - vy * u;
        if (dx * dx + dy * dy <= rad * rad) mask[Y * PW + X] = 1;
      }
      grow(X0, Y0, X1, Y1);
      return P;
    },
    // a limb segment that narrows from w0 to w1, with rounded joints at both ends
    t(x0, y0, x1, y1, w0, w1) {
      const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy) || 1, nx = -dy / L, ny = dx / L;
      P.p([[x0 + nx * w0 / 2, y0 + ny * w0 / 2], [x1 + nx * w1 / 2, y1 + ny * w1 / 2], [x1 - nx * w1 / 2, y1 - ny * w1 / 2], [x0 - nx * w0 / 2, y0 - ny * w0 / 2]]);
      return P.e(x0, y0, w0 / 2, w0 / 2).e(x1, y1, w1 / 2, w1 / 2);
    },
    // Paint the masked shape with rounded form shading: each row is shaded like a cylinder lit from
    // the upper left, darkening toward the lower right, with a 2x2 dither between neighbouring tones.
    fill(rmp, o) {
      o = o || {};
      if (bx1 < bx0) return P;
      const round = o.round == null ? 1 : o.round;
      // dither: how strongly neighbouring tones are mixed in a 2x2 pattern where they meet (1 = full)
      const dith = o.dither == null ? 1 : o.dither;
      // on: only repaint pixels that are already painted, so details cannot spill past a shape's edge
      // (or a test of the pixel index k, to repaint only pixels of a chosen colour)
      // vertical extent of the shape through every pixel, found in two sweeps per column
      for (let X = bx0; X <= bx1; X++) {
        let start = -1;
        for (let Y = by0; Y <= by1 + 1; Y++) {
          const on = Y <= by1 && mask[Y * PW + X];
          if (on && start < 0) start = Y;
          if (!on && start >= 0) { for (let q = start; q < Y; q++) { runTop[q * PW + X] = start; runBot[q * PW + X] = Y - 1; } start = -1; }
        }
      }
      for (let Y = by0; Y <= by1; Y++) {
        let X = bx0;
        while (X <= bx1) {
          if (!mask[Y * PW + X]) { X++; continue; }
          let X1 = X;
          while (X1 + 1 <= bx1 && mask[Y * PW + X1 + 1]) X1++;
          for (let i = X; i <= X1; i++) {
            const k = Y * PW + i, top = runTop[k], bot = runBot[k];
            const u = (i - X + 0.5) / (X1 - X + 1), v = (Y - top + 0.5) / (bot - top + 1);
            // shade: a function of the fine pixel giving its own brightness (the face's modelled form)
            let b = o.shade ? o.shade(i, Y) : 0.6 - (u - 0.35) * 0.56 * round - (v - 0.45) * 0.28 * round + (o.lift || 0);
            const edge = i === X || i === X1 || Y === top || Y === bot;
            if (o.contour && edge) b -= 0.38;
            else if (!o.flat && !o.shade && (i === X1 || Y === bot)) b -= 0.17;
            if (o.dim) b -= 0.16;
            if (o.pattern) b += o.pattern(i, Y);
            let idx = o.tone != null ? o.tone : Math.round(b * 4 + (BAYER[Y & 1][i & 1] - 0.375) * 0.45 * dith + 0.16875 - 0.22);
            if (o.tone != null && !o.flat && (i === X1 || Y === bot)) idx--;
            if (!o.on || (typeof o.on === 'function' ? o.on(k) : buf[k * 4 + 3])) set(k, rmp[Math.max(0, Math.min(4, idx))]);
          }
          X = X1 + 1;
        }
      }
      for (let Y = by0; Y <= by1; Y++) mask.fill(0, Y * PW + bx0, Y * PW + bx1 + 1);
      bx0 = PW; by0 = PH; bx1 = -1; by1 = -1;
      return P;
    },
    // one design pixel (a RES x RES block)
    px(x, y, c) {
      const X0 = Math.round(x) * RES, Y0 = Math.round(y) * RES;
      for (let Y = Y0; Y < Y0 + RES; Y++) for (let X = X0; X < X0 + RES; X++) if (X >= 0 && Y >= 0 && X < PW && Y < PH) set(Y * PW + X, c);
      return P;
    },
    // a fine pixel that only lands where something is already painted
    dotOn(x, y, c) {
      const X = Math.floor(x * RES), Y = Math.floor(y * RES);
      if (X >= 0 && Y >= 0 && X < PW && Y < PH && buf[(Y * PW + X) * 4 + 3]) set(Y * PW + X, c);
      return P;
    },
    // darken an already painted fine pixel, as a soft cast shadow
    // (cool: how much the shadow leans blue; 0 for a neutral grey shadow)
    darken(x, y, k, cool = 1) {
      const X = Math.floor(x * RES), Y = Math.floor(y * RES), q = (Y * PW + X) * 4;
      if (X >= 0 && Y >= 0 && X < PW && Y < PH && buf[q + 3]) { buf[q] *= k; buf[q + 1] *= k * (1 + 0.01 * cool); buf[q + 2] *= k * (1 + 0.05 * cool); }
      return P;
    },
    // one fine pixel at a design position (fractions select the sub-pixel)
    dot(x, y, c) {
      const X = Math.floor(x * RES), Y = Math.floor(y * RES);
      if (X >= 0 && Y >= 0 && X < PW && Y < PH) set(Y * PW + X, c);
      return P;
    },
    // outline coloured from the pixel it borders (a dark, cool version of it), not flat black
    outline() {
      const src = buf.slice();
      for (let Y = 0; Y < PH; Y++) for (let X = 0; X < PW; X++) {
        const k = Y * PW + X;
        if (src[k * 4 + 3]) continue;
        const n = [X > 0 ? k - 1 : -1, X < PW - 1 ? k + 1 : -1, Y > 0 ? k - PW : -1, Y < PH - 1 ? k + PW : -1].find((q) => q >= 0 && src[q * 4 + 3]);
        if (n == null) continue;
        set(k, [src[n * 4] * 0.34 + 12, src[n * 4 + 1] * 0.32 + 10, src[n * 4 + 2] * 0.34 + 14]);
      }
    }
  };
  return P;
}

const EYE = [46, 40, 38], WHITE = [226, 222, 210];
// Local figure space: a = toward the figure's right, b = forward, h = up from the ground (sprite pixels).
// theta 0 faces the viewer, PI/2 faces screen right, PI faces away.
function View(theta, bob, sway) {
  sway = sway || 0;
  const fx = Math.sin(theta), fz = Math.cos(theta), rx = -Math.cos(theta), rz = Math.sin(theta);
  // The walk's bob and sway move the whole figure. They are snapped to whole fine pixels, so the figure moves
  // rigidly: a shift of a fraction of a pixel lands the shapes and the details snapped to the pixel grid (eyes,
  // mouth) on different sides of a pixel, and the face comes out subtly rearranged in those frames.
  bob = Math.round(bob * RES) / RES;
  const shift = Math.round(sway * rx * RES) / RES;
  const V = {
    fx, fz, rx, rz, bob,
    x: (a, b) => CX + shift + a * rx + b * fx,
    y: (h) => GY - h + bob,
    d: (a, b) => a * rz + b * fz,
    pt: (a, b, h) => [V.x(a, b), V.y(h)],
    w: (A, B) => Math.sqrt(A * A * rx * rx + B * B * fx * fx),
    seam: (A, B, phi, h) => [V.x(A * Math.sin(phi), B * Math.cos(phi)), V.y(h), V.d(A * Math.sin(phi), B * Math.cos(phi))]
  };
  return V;
}
// The face's form, as a brightness for each fine pixel (for P.fill's shade). The head is the union of two
// ellipsoids, the skull and the jaw ([forward offset, height, half-width, half-height, depth radius]),
// lit from the upper left and in front. On top of that light sit the planes of the face: eye sockets under
// the brow ridge, a lit cheekbone with a hollow under it, the nose (a lit ridge, a shadow side and the
// shadow it casts on the lip), the lips and the ball of the chin. All are placed in the head's own
// coordinates, so they turn with it and foreshorten in the three-quarter views.
// the jaw in profile: [forward offset, half-width] for him and for her (height and depth as from the front)
const JAW_PROFILE = { male: [0.9, 4.4], fem: [0.8, 4.1] };
// the jawline in profile: the angle of the jaw below the ear and the point of the chin (forward offset, height)
const JAWLINE = { male: [[-0.8, -4.9], [3.0, -7.7]], fem: [[-0.6, -4.6], [2.8, -7.3]] };
// the front edge of the face in profile, in head space (forward offset, height from the head's centre):
// the bridge and tip of the nose, the upper lip, the lips and the point of the chin
function PROFILE_FRONT(P, V, fem) {
  const pts = fem
    ? [[4.2, 1.6], [6.2, 1.0], [6.4, 0], [6.1, -0.9], [6.4, -2.0], [6.7, -2.9], [6.3, -3.3], [6.0, -3.6], [6.1, -4.4], [5.8, -5.0], [5.9, -5.5], [5.4, -6.0], [5.5, -6.6], [5.2, -7.1], [4.6, -7.45], [3.7, -7.6], [2.8, -7.4], [2.6, -5]]
    : [[4.2, 1.6], [6.5, 1.2], [6.7, 0], [6.4, -0.9], [6.7, -2.0], [7.1, -3.0], [6.6, -3.5], [6.2, -3.8], [6.2, -4.8], [5.9, -5.3], [5.8, -5.8], [5.3, -6.5], [5.0, -7.1], [4.5, -7.65], [3.8, -7.95], [3.0, -7.85], [2.8, -5]];
  const at = ([b, h]) => [V.x(0, b), V.y(HEAD + h)];
  P.p(pts.map(at));
  // the jawline: from the point of the chin it runs back and up to the angle of the jaw below the ear.
  // The rounded bottoms of the skull and the jaw below that line would read as a swollen lower cheek.
  const [angle, chin] = JAWLINE[fem ? 'fem' : 'male'];
  // (only in front of the angle: behind it the head runs into the neck, and cutting there leaves a gap)
  // (the line bows gently down between the two, so the jaw reads as a soft curve rather than a hard edge)
  const line = [];
  for (let k = 0; k <= 6; k++) { const t = k / 6; line.push([lerp(angle[0], chin[0], t), lerp(angle[1], chin[1], t) - 0.5 * Math.sin(Math.PI * t)]); }
  P.cut(line.concat([[chin[0] + 4, chin[1]], [chin[0] + 4, -14], [angle[0], -14]]).map(at));
}
const FACE_LIGHT = (() => { const l = [-0.5, 0.55, 0.67], n = Math.hypot(...l); return l.map((c) => c / n); })();
function faceShade(V, fem, SKULL, JAW) {
  const g = (x, y, sx, sy) => Math.exp(-(x * x) / (sx * sx) - (y * y) / (sy * sy));
  const [Lx, Ly, Lz] = FACE_LIGHT;
  // the light's direction across the head (toward its right, a), so the nose's sides can face it or not
  const La = V.rx * Lx + V.rz * Lz;
  const shade = (X, Y) => {
    const sx = (X + 0.5) / RES, sy = (Y + 0.5) / RES;
    // the two surfaces under this pixel, blended smoothly toward the frontmost (a soft union, so there is
    // no crease where the jaw meets the skull): the view normal and the point in head space
    let wt = 0, nx = 0, ny = 0, nz = 0, xv = 0, zv = 0, cbs = 0;
    [SKULL, JAW].forEach(([cb, ch, w, hh, rd]) => {
      const dx = (sx - V.x(0, cb)) / w, dy = (V.y(HEAD + ch) - sy) / hh, q = 1 - dx * dx - dy * dy;
      if (q < -0.05) return;
      const n = Math.sqrt(Math.max(0, q)), z = cb * V.fz + rd * n, e = Math.exp(z / 0.9) * Math.min(1, (q + 0.05) * 8);
      wt += e; nx += dx / w * 6 * e; ny += dy / hh * 6 * e; nz += n / rd * 6 * e; xv += dx * w * e; zv += rd * n * e; cbs += cb * e;
    });
    if (!wt) return 0.4;
    nx /= wt; ny /= wt; nz /= wt; xv /= wt; zv /= wt;
    const best = { xv, zv, cb: cbs / wt }, nl = Math.hypot(nx, ny, nz) || 1;
    // light with a firm turn into shadow: the core shadow is darkest just past the turn, and the far rim
    // picks up a little light reflected back from the shoulders
    const l = (nx * Lx + ny * Ly + nz * Lz) / nl;
    let b = 0.1 + 0.66 * Math.max(0, Math.min(1, (l + 0.08) / 1.0));
    if (l < -0.08) b += 0.09 * Math.min(1, Math.max(0, (-0.3 - l) * 2.5)) - 0.04 * Math.min(1, (-0.08 - l) * 5);
    // the underside of the jaw turns away from the light into the shadow over the neck
    if (ny / nl < -0.5) b -= (-0.5 - ny / nl) * 0.4;
    // head space: a across (toward the head's right), f forward, h height relative to the head's centre
    const a = V.rx * best.xv + V.rz * best.zv, f = best.cb + V.fx * best.xv + V.fz * best.zv, h = GY + V.bob - sy - HEAD;
    // only the front of the head carries the face
    const front = Math.max(0, Math.min(1, (f - 1.5) / 3));
    if (front <= 0) return b;
    let k = 0;
    const aa = Math.abs(a);
    // eye sockets: shadow under the brow ridge, deepest at the inner corner beside the bridge of the nose
    k -= 0.18 * g(aa - 2.3, h + 0.5, 1.35, 1.05) + 0.1 * g(aa - 1.2, h + 0.2, 0.6, 0.9);
    // the brow ridge and the forehead above it catch the light
    k += 0.09 * g(aa - 2.3, h - 1.8, 2, 0.6) + 0.07 * g(a + 0.6, h - 4.2, 3, 1.6);
    // the cheekbone, lit, and (on him) the hollow of the cheek under it
    k += (fem ? 0.1 : 0.12) * g(aa - 3.4, h + 2.6, 1.3, 0.9);
    if (!fem) k -= 0.08 * g(aa - 4, h + 5, 1.2, 1.3);
    else k += 0.05 * g(aa - 2.8, h + 3.6, 1.6, 1.1);
    // the nose: a lit ridge down the middle, its sides turned toward or away from the light, a lit tip and
    // the shadow it casts down onto the upper lip
    const nose = g(0, (h + 2.2) / 2.1, 1, 1) * g(a, 0, fem ? 0.75 : 0.9, 1);
    k += nose * (0.07 + 0.28 * Math.max(-1, Math.min(1, a / 0.7)) * La);
    k += 0.08 * g(a, h + (fem ? 3.1 : 3.4), 0.5, 0.45);
    k -= 0.17 * g(a - La * 0.5, h + (fem ? 3.95 : 4.25), 1.2, 0.4);
    // the upper lip turns down out of the light, the lower lip and chin turn up into it, with a shadow between
    k -= 0.05 * g(a, h + 4.8, 1.8, 0.5);
    k -= 0.1 * g(a, h + (fem ? 6.3 : 6.4), 1.3, 0.35);
    k += 0.08 * g(a, h + (fem ? 7.1 : 7.6), 1.8, 0.7);
    // the corner where the front of the face turns into its side: a soft step, so the face reads as planes
    k -= 0.08 * Math.max(0, Math.min(1, (aa - 4.4) / 1.2));
    // how much of a flush this pixel carries: a soft patch on the apple of each cheek, below the cheekbone
    // (softer as the head turns, where a round patch on the foreshortened cheek reads as a swelling)
    const flush = front * g(aa - 3.2, h + 3.5, 1.5, 1.05) * (1 - 0.45 * V.fx * V.fx);
    if (flush > 0.02) shade.flush.set(Y * PW + X, flush);
    return b + k * front;
  };
  shade.flush = new Map();
  return shade;
}
// the flushed skin of the cheeks: the skin ramp nudged a little toward rose
const FLUSH = SKIN.map((c) => c.map((v, i) => Math.round(v + ([214, 104, 96][i] - v) * 0.28)));
// mail: rows of interlocking rings, each with a dark centre and a lit upper edge
const mail = (x, y) => { const r = y % 3, sx = (x + (Math.floor(y / 3) % 2) * 2) % 4; return r === 1 && sx === 1 ? -0.34 : r === 0 && (sx === 1 || sx === 2) ? 0.16 : r === 2 && sx === 0 ? -0.1 : 0; };
// hair: strands running down, with a few lighter ones catching the light
// softer strands for long hair: gentle streaks, no bright glints
const strands = (x, y) => { const k = (x * 7 + Math.floor(y / 3) * 3) % 9; return k === 0 ? -0.24 : k === 4 ? 0.16 : 0; };
const weave = (x, y) => ((x + y) % 4 === 0 ? -0.09 : (x - y) % 6 === 0 ? 0.05 : 0);

function drawFigure(P, S, theta, f) {
  // Walk: one smooth cycle over eight frames. swing is the right leg's stride (-1 back .. 1 forward);
  // the body rises as the legs pass and dips as each foot lands, and the weight shifts over the planted foot.
  const walking = f >= 2, step = f - 2, ph = walking ? step / WALK * Math.PI * 2 : 0;
  const swing = walking ? Math.sin(ph) : 0;
  const bob = walking ? 0.4 - 1.5 * (1 - Math.abs(swing)) + 0.6 * Math.abs(swing) : (f === 1 ? 1 : 0);
  const V = View(theta, bob, walking ? Math.cos(ph) * 0.7 : 0);
  const parts = [];
  const add = (depth, draw) => parts.push({ depth, draw });
  // body proportions: shoulder, chest, waist and hip half-widths, depth, limb widths, leg spacing
  const B = Object.assign({ sh: 9, chest: 8.6, waist: 7.2, hip: 7.8, depth: 5.4, arm: 4.8, fore: 4, leg: 6.2, shin: 4.8, gap: 3.8, slope: 3 }, S.body);
  const A = { V, swing, ph, f, walking, B, S };
  // Half-width of the body at height h, eased smoothly between control heights (hip, waist, chest,
  // bust, shoulder), so the silhouette curves instead of joining straight segments.
  const shH = 41.5 - B.slope;
  const ctrl = B.curve ? B.curve.concat([[shH - 2.5, B.sh + 0.4]]) : [[17, B.hipLow || B.hip], [21, B.hip], [B.waistH || 26.5, B.waist], [31, B.chest], [34.5, B.bust || B.chest], [shH - 2.5, B.sh + 0.4]];
  const rawW = (h) => {
    if (h <= ctrl[0][0]) return ctrl[0][1];
    if (h >= ctrl[ctrl.length - 1][0]) return ctrl[ctrl.length - 1][1];
    for (let i = 0; i < ctrl.length - 1; i++) if (h <= ctrl[i + 1][0]) {
      const t = (h - ctrl[i][0]) / (ctrl[i + 1][0] - ctrl[i][0]);
      return lerp(ctrl[i][1], ctrl[i + 1][1], (1 - Math.cos(Math.PI * t)) / 2);
    }
    return ctrl[ctrl.length - 1][1];
  };
  // sample the profile finely and smooth it twice, so the waist flows into chest and hips without kinks
  const H0 = 12, STEP = 0.25, N = Math.ceil((46 - H0) / STEP);
  let prof = Array.from({ length: N + 1 }, (_, i) => rawW(H0 + i * STEP));
  for (let pass = 0; pass < 2; pass++) {
    const R = 5, next = prof.slice();
    for (let i = 0; i <= N; i++) { let sum = 0, n = 0; for (let k = -R; k <= R; k++) { const j = Math.min(N, Math.max(0, i + k)); sum += prof[j]; n++; } next[i] = sum / n; }
    prof = next;
  }
  // body radii at a height: sideways (lw) and front-to-back (ld), in the figure's own units
  A.lw = (h, extra) => {
    const fi = Math.min(N, Math.max(0, (h - H0) / STEP)), i0 = Math.floor(fi), i1 = Math.min(N, i0 + 1);
    return lerp(prof[i0], prof[i1], fi - i0) + (extra || 0);
  };
  A.ld = (w) => B.depth * (0.75 + 0.25 * w / B.chest);
  A.halfW = (h, extra) => { const w = A.lw(h, extra); return V.w(w, A.ld(w)); };
  // the near half of a ring around the body, left to right on screen; hFn(phi) sets the height around it,
  // and the ring is tipped a little toward the camera so it curves like a hem or belt seen from above
  A.ring = (A1, B1, hFn, tilt) => {
    const out = [];
    for (let k = 0; k <= 48; k++) {
      const phi = -Math.PI + k / 48 * Math.PI * 2, q = V.seam(A1, B1, phi, hFn(phi));
      if (q[2] >= -0.05) out.push([q[0], q[1] + q[2] * (tilt == null ? 0.12 : tilt), q[2]]);
    }
    return out.sort((p, q) => p[0] - q[0]);
  };
  // mask the body's outline between two heights, widened by extra
  A.band = (hTop, hBot, extra, flareBot) => {
    const cx = V.x(0, 0), L = [], R = [];
    for (let h = hTop; h >= hBot - 0.01; h -= 0.5) {
      const e = (extra || 0) + (flareBot ? flareBot * Math.max(0, 21 - h) / Math.max(1, 21 - hBot) : 0);
      const w = A.halfW(h, e);
      L.push([cx - w, V.y(h)]); R.push([cx + w, V.y(h)]);
    }
    return P.p(L.concat(R.reverse()));
  };

  // legs: tapered thigh and shin, a rounded boot, a cuff where the boot ends
  [1, -1].forEach((s) => {
    const ps = ph + (s > 0 ? 0 : Math.PI), fb = walking ? Math.sin(ps) * 6.5 : 0, lift = walking ? Math.max(0, Math.cos(ps)) * 3.2 : 0;
    const toeUp = walking ? Math.max(0, Math.sin(ps)) * 0.8 - Math.max(0, -Math.sin(ps)) * 0.6 : 0;
    const hipA = s * B.gap, depth = V.d(hipA, fb), dim = depth < -1;
    add(depth * 0.05 - 4, () => {
      const hip = V.pt(hipA, fb * 0.2, 22), knee = V.pt(hipA * 1.05, fb * 0.5 + 0.8 + lift * 0.9, 11.5 + lift * 0.8), ankle = V.pt(hipA * 1.08, fb, 4 + lift);
      if (!S.robe) {
        P.t(hip[0], hip[1], knee[0], knee[1], B.leg, B.shin + 0.4).t(knee[0], knee[1], ankle[0], ankle[1], B.shin + 0.4, B.shin - 0.8).fill(S.legs, { dim, pattern: S.legPattern });
        if (S.knee && V.d(hipA, fb + 3) > 0) P.e(knee[0] + V.fx, knee[1], 2.4, 2.2).fill(S.knee, { dim });
        if (S.legArmor) S.legArmor(P, { hip, knee, ankle }, dim, A);
      }
      // Shoe: built from slices across the foot, heel to toe, each a section with a flat underside and a
      // rounded top. It is cut roomier than the foot inside it: a full heel, a high instep running straight
      // down from the shaft, and a broad, boxy toe. The foot pitches with the stride: toe up as the heel strikes,
      // heel up as it pushes off. The near end of the foot sits a little lower, as if seen from slightly above.
      const SH = Object.assign({ len: 7.6, back: 2.3, wide: 3.1, heelW: 0.92, toeH: 3, heelH: 4.2, instep: 5.4, sole: 0.9, heelBlock: 0.6, arch: 0.35, point: 0 }, S.shoe);
      const fa = hipA * 1.08, TILT = 0.2, ua = SH.back / SH.len, BALL = 0.7, HEEL = ua * 0.85;
      const pitch = (b) => (toeUp > 0 ? toeUp * 1.6 * (b + SH.back) / SH.len : -toeUp * 1.8 * (SH.len - SH.back - b) / SH.len);
      const footW = (u) => {
        if (u > BALL) { const t = (u - BALL) / (1 - BALL); return SH.wide * Math.max(0.3, Math.sqrt(1 - Math.pow(t, 2 + SH.point * 2))); }
        return SH.wide * lerp(SH.heelW, 1, u / BALL) * Math.sqrt(Math.min(1, 0.6 + u / 0.06 * 0.4));
      };
      const footH = (u) => {
        if (u < ua) return lerp(SH.heelH, SH.instep, u / ua);
        // the instep falls in a nearly straight line to the toe box, not following the curve of the foot
        if (u < 0.7) return lerp(SH.instep, SH.toeH, Math.pow((u - ua) / (0.7 - ua), 0.85));
        // a blunt toe box: full height until near the tip, then turning down square to the sole
        return SH.toeH * Math.sqrt(Math.max(0.2, 1 - Math.pow((u - 0.7) / 0.3, 6)));
      };
      // the arch: the sole lifts off the ground between the heel and the ball of the foot
      const archAt = (u) => (u > HEEL && u < BALL - 0.08 ? Math.sin(Math.PI * (u - HEEL) / (BALL - 0.08 - HEEL)) * SH.arch : 0);
      const slices = [];
      for (let k = 0; k <= 20; k++) {
        const u = k / 20, b = -SH.back + u * SH.len, base = lift + pitch(b), d = V.d(fa, fb + b), arch = archAt(u);
        slices.push({ u, x: V.x(fa, fb + b), sw: Math.max(0.55, footW(u) * Math.abs(V.rx)), g: V.y(base) + d * TILT, y0: V.y(base + arch) + d * TILT, y1: V.y(base + footH(u)) + d * TILT, d });
      }
      // each slice is a flat-bottomed section with a rounded top; neighbouring slices are joined by their
      // convex hull so the profile runs smoothly from heel to toe
      const hull = (pts) => {
        pts.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
        const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]), lo = [], hi = [];
        pts.forEach((p) => { while (lo.length > 1 && cr(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); });
        pts.slice().reverse().forEach((p) => { while (hi.length > 1 && cr(hi[hi.length - 2], hi[hi.length - 1], p) <= 0) hi.pop(); hi.push(p); });
        return lo.slice(0, -1).concat(hi.slice(0, -1));
      };
      const capR = (y0, y1, sw) => Math.min((y0 - y1) * 0.5, sw * 0.4);
      const section = (q, y0, y1, sw) => { const r = capR(y0, y1, sw), pts = [[q - sw, y0], [q + sw, y0], [q + sw, y1 + r], [q - sw, y1 + r]]; for (let k = 1; k < 6; k++) { const a = Math.PI * k / 6; pts.push([q - Math.cos(a) * sw, y1 + r - Math.sin(a) * r]); } return pts; };
      const sweep = (y0f, y1f, swf) => { for (let k = 0; k < slices.length - 1; k++) { const a = slices[k], b = slices[k + 1]; P.p(hull(section(a.x, y0f(a), y1f(a), swf(a)).concat(section(b.x, y0f(b), y1f(b), swf(b))))); } };
      const top = V.pt(hipA * 1.06, fb * 0.75, (S.robe ? 5 : 9.5) + lift);
      // sole first: a welt standing a little proud of the upper, thicker under the heel, lifting at the
      // arch. The upper then sits on it, so from the front the near toe covers the sole of the far heel.
      const SOLE = S.sole || S.boots;
      const soleH = (q) => SH.sole + (q.u < HEEL ? SH.heelBlock : 0);
      sweep((q) => q.y0, (q) => q.y0 - soleH(q) - 0.4, (q) => q.sw + 0.3);
      P.fill(SOLE, { dim, tone: S.sole ? 2 : 1, flat: false });
      // the shaft drops straight to the foot, standing off the ankle rather than hugging it
      P.t(top[0], top[1], ankle[0], ankle[1] + 0.4, 5.4, 5.2);
      sweep((q) => q.y0 - soleH(q), (q) => q.y1, (q) => q.sw);
      P.fill(S.boots, { dim, round: S.bootRound || 1.1, pattern: S.bootPattern });
      // the welt's top edge catches the light where it stands out from under the upper
      slices.forEach((q) => { for (let t = -q.sw - 0.3; t <= q.sw + 0.3; t += 0.5) if (Math.abs(t) > q.sw - 0.2 || q.u > 0.92) P.dotOn(q.x + t, q.y0 - soleH(q) + 0.25, SOLE[dim ? 1 : S.sole ? 3 : 2]); });
      // the toe box: seen from the front, a crease where it meets the instep with light along its rounded
      // top (from the side, the fill's own shading models it)
      const T = (i) => S.boots[Math.max(0, i - (dim ? 1 : 0))];
      if (V.d(0, 1) > 0.3) {
        const q = slices[16], r = capR(q.y0, q.y1, q.sw);
        for (let t = -0.92; t <= 0.92; t += 0.08) { const x = q.x + t * q.sw, y = q.y1 + r - Math.sqrt(1 - t * t) * r; P.dotOn(x, y, T(1)); if (t < 0.4) P.dotOn(x, y + 0.5, T(S.bootRound ? 2 : 3)); }
      }
      const heel = [slices[0].x, slices[0].g], toe = [slices[20].x, slices[20].g];
      if (S.bootDetail) S.bootDetail(P, { top, ankle, slices, heel, toe }, dim, A);
      if (S.footArmor) S.footArmor(P, { heel, toe }, dim, A);
      if (S.cuff && !S.robe) {
        // a folded band (cuffBand: [width, height]) or the default rolled cuff
        if (S.cuffBand) P.e(top[0], top[1] - 1.5, S.cuffBand[0] / 2, S.cuffBand[1] / 2).fill(S.cuff, { dim, round: 0.7 });
        else P.t(top[0], top[1] + 0.5, top[0], top[1] + 1.5, 6.2, 6.2).fill(S.cuff, { dim, round: 0.5 });
        if (S.cuffDetail) S.cuffDetail(P, top, dim, A);
      }
    });
  });

  // pelvis: one rounded form the legs grow out of, with a soft crease where they part
  if (!S.robe) add(-4.05, () => {
    P.e(V.x(0, 0.4), V.y(18.6), V.w(B.hip - 0.4, B.depth - 0.2), 4.6).fill(S.legs, { pattern: S.legPattern, round: 0.8 });
    if (V.fz > 0.4) P.l(V.x(0, B.depth - 0.4), V.y(17), V.x(0, B.depth - 0.6), V.y(14.2), 0.6).fill(S.legs, { flat: true, tone: 1 });
  });

  // cape or cloak, hanging behind; its lining shows at the edges
  if (S.cape) add(V.d(0, -6), () => {
    const sp = f === 1 ? 1 : 0, tr = -Math.abs(swing) * 1.2 - Math.cos(ph * 2) * 0.6 * (walking ? 1 : 0) - (walking ? 2.2 : 0), hem = S.capeHem || 7, cw = S.capeW || 1;
    const q = [V.pt(-8.5 * cw, -4.5, 40 - bob), V.pt(8.5 * cw, -4.5, 40 - bob), V.pt((10.5 + sp) * cw, -6, Math.max(24, hem + 4)), V.pt((12 + sp) * cw, -7 + tr, hem), V.pt(0, -8 + tr, hem - 1.5), V.pt(-(12 + sp) * cw, -7 + tr, hem), V.pt(-(10.5 + sp) * cw, -6, Math.max(24, hem + 4))];
    if (S.capeLining && V.fz > -0.2) P.p(q.map(([x, y]) => [x + (x > CX ? 1 : -1), y + 1])).fill(S.capeLining, { dim: true });
    P.p(q).fill(S.cape, { dim: V.fz > 0.3 });
    [-6, -2, 2, 6].forEach((a) => P.l(V.x(a * 0.8, -5), V.y(36), V.x(a * 1.25, -7 + tr), V.y(hem + 3), 1));
    P.fill(S.cape, { flat: true, tone: 1 });
  });

  // torso: shoulders slope up into the head (no separate neck), chest narrows to the waist and widens at the hips
  add(0, () => {
    const cx = V.x(0, 0), D = B.depth;
    const wN = V.w(3.4, 3), wS = V.w(B.sh, D * 0.8), wD = V.w(B.sh + 0.4, D), wc = V.w(B.chest, D + 0.5), ww = V.w(B.waist, D - 0.3), wh = V.w(B.hip, D);
    const top = 41.5 + B.slope * 0.3 - (B.neckDrop || 0);
    const neckline = [[-wN, top + 1.5], [-wN - 1, top], [-(wN + wS) / 2 - 0.5, top - B.slope * 0.55], [-wS, shH + 0.2]];
    const side = [];
    for (let h = shH - 1; h >= 21; h -= 0.5) side.push([A.halfW(h), h]);
    const outline = neckline.concat(side.map(([w, h]) => [-w, h]), side.slice().reverse().map(([w, h]) => [w, h]), neckline.slice().reverse().map(([x, h]) => [-x, h]));
    P.p(outline.map(([x, h]) => [cx + x, V.y(h)])).e(cx - wD + 1.8, V.y(shH - 1.6), 2.4, 2.6).e(cx + wD - 1.8, V.y(shH - 1.6), 2.4, 2.6).fill(S.torso, { pattern: S.torsoPattern });
    const bl = B.belt || 22.5, wh0 = A.halfW(bl + 1);
    if (S.drawRobe) S.drawRobe(P, A);
    else if (S.robe) {
      // the robe hangs from the shoulders and widens steadily to the hem, with no waist
      const flare = Math.abs(swing) * 1.4, hw = V.w(12.4, 9.2), top = A.halfW(25, 0.8), mid = V.w(11, 8.2);
      const hemY = (k) => 3.4 + Math.sin(k * 1.7 + swing * 0.8) * 0.6;
      const hem = [];
      for (let k = 0; k <= 8; k++) { const t = k / 8; hem.push([cx + (t * 2 - 1) * (hw + flare), V.y(hemY(k))]); }
      P.p([[cx - top, V.y(25)], [cx + top, V.y(25)], [cx + mid + 0.4, V.y(14)]].concat(hem.slice().reverse()).concat([[cx - mid - 0.4, V.y(14)]]))
        .e(cx, V.y(3.8), hw + flare, 2).fill(S.robe, { pattern: weave });
      // long folds: shadowed troughs with a lit ridge beside each, swinging with the stride
      [-0.95, -0.45, 0.05, 0.5, 0.95].forEach((phi, n) => {
        const pts = [[7.8, 6, phi * 0.7, 33], [9.2, 7, phi * 0.9, 22], [11, 8.4, phi + swing * 0.1, 12], [12.2, 9.2, phi + swing * 0.16, 4.6]].map(([A1, B1, ph, h]) => V.seam(A1, B1, ph, h));
        if (pts[0][2] < 0.8) return;
        for (let i = 0; i < pts.length - 1; i++) P.l(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], 0.8 + i * 0.25);
        P.fill(S.robe, { flat: true, tone: 1, pattern: weave });
        for (let i = 1; i < pts.length - 1; i++) P.l(pts[i][0] + 0.7, pts[i][1], pts[i + 1][0] + 0.8, pts[i + 1][1], 0.5);
        P.fill(S.robe, { flat: true, tone: 3 });
      });
      // embroidered hem following the ripple
      for (let k = 0; k < 48; k++) {
        const phi = k / 48 * Math.PI * 2, [x, , d] = V.seam(12.2 + flare, 9, phi, 5);
        const t = Math.max(0, Math.min(8, (x - (cx - hw - flare)) / (2 * (hw + flare)) * 8));
        if (d > -2) P.e(x, V.y(hemY(t) + 1.4), 1.1, 1.4);
      }
      P.fill(S.trim, { round: 0.4 });
    }
    if (S.chest) S.chest(P, A);
    // belt, sitting at the waist (a robe drawn by its own spec ties its own)
    if (S.drawRobe) { /* tied by the robe */ } else if (S.robe) {
      const wl = A.halfW(bl + 2, 1), sag = Math.max(0, V.fz) * 1.6;
      P.p([[cx - wl, V.y(bl + 2.6)], [cx, V.y(bl + 2.6 - sag)], [cx + wl, V.y(bl + 2.6)], [cx + wl, V.y(bl + 0.4)], [cx, V.y(bl + 0.4 - sag)], [cx - wl, V.y(bl + 0.4)]]).fill(S.belt, { round: 0.6 });
    } else {
      // a band around the hips: top and bottom edges follow the body, curving as seen from above
      const wt = A.lw(bl + 3, 0.5), wb = A.lw(bl, 0.7);
      const top = A.ring(wt, A.ld(wt) + 0.3, () => bl + 3), bot = A.ring(wb, A.ld(wb) + 0.4, () => bl);
      P.p(top.map(([x, y]) => [x, y]).concat(bot.map(([x, y]) => [x, y]).reverse())).fill(S.belt, { round: 0.6 });
    }
    if (V.fz > 0.2 && !S.robe) {
      const bx = V.x(0, 5.3), by = V.d(0, A.ld(A.lw(bl + 1.5))) * 0.12;
      P.p([[bx - 1.8 * V.fz - 0.3, V.y(bl + 2.8) + by], [bx + 1.8 * V.fz + 0.3, V.y(bl + 2.8) + by], [bx + 1.8 * V.fz + 0.3, V.y(bl - 0.8) + by], [bx - 1.8 * V.fz - 0.3, V.y(bl - 0.8) + by]]).fill(S.buckle || S.trim, { round: 0.4 });
    }
    if (S.waist) S.waist(P, A);
  });

  // arms: tapered upper arm and forearm, sleeves, cuffs and hands
  [1, -1].forEach((s) => {
    const grip = S.grip && S.grip(s, A);
    const hb = -swing * s * 6;
    const hand = grip || [s * (B.sh + 1.4 + (S.blendArms ? 0.6 : B.armOut == null ? 1.2 : B.armOut)), hb + 0.6, 22 + Math.max(0, hb) * 0.35];
    const sh = [s * (B.sh - 1), 0, 38.8 - B.slope * 0.3];
    const depth = V.d(s * 11, hand[1] * 0.5) - (S.blendArms && Math.abs(V.fz) > 0.6 ? 0.5 : 0), dim = depth < -2;
    A['hand' + (s > 0 ? 'R' : 'L')] = { a: hand[0], b: hand[1], h: hand[2], depth };
    add(depth, () => {
      const elbow = [s * (B.sh + 0.9 + (S.blendArms ? 0.4 : (B.armOut == null ? 1.2 : B.armOut) * 0.5)), hand[1] * 0.45 - 1.4, (sh[2] + hand[2]) / 2 + 0.8];
      const pS = V.pt(...sh), pE = V.pt(...elbow), pH = V.pt(hand[0], hand[1], hand[2] + 1.6);
      const wU = S.sleeveW || B.arm, wF = S.sleeveW ? S.sleeveW + 1.5 : B.fore;
      const outward = Math.sign(pH[0] - V.x(0, 0)) || s;
      if (S.sleeveStyle === 'bell') {
        // a loose upper sleeve that widens into a long bell; the lower edge hangs well past the wrist
        // The sleeve starts narrow at a sloping shoulder (no square cap) and widens down the arm. Its mouth
        // is a soft curve: the cloth sags lowest a little to the outside of the wrist and rounds up at
        // both ends, the way a heavy bell sleeve hangs, rather than coming to a point.
        const wu = B.arm + 1.8, open = 5.2, wy = pH[1] + 1;
        const pS2 = [pS[0] - outward * 0.2, pS[1] + 0.6];
        const mouth = []; for (let k = 0; k <= 10; k++) { const t = k / 10, x = pH[0] + outward * lerp(-open + 0.5, open - 0.3, t); mouth.push([x, wy + lerp(1.0, 1.6, t) + 1.7 * Math.pow(Math.sin(Math.PI * Math.pow(t, 0.75)), 0.8)]); }
        P.t(pS2[0], pS2[1], pE[0], pE[1], wu + 0.2, wu + 0.6)
          // the shoulder: a soft round where the cloth falls over the top of the arm, so the line from the
          // neck runs on down the sleeve with no corner or step
          .e(pS2[0] + outward * 0.2, pS2[1] + 1.2, wu / 2 + 0.5, 2.8)
          .p([[pE[0] - wu / 2 - 0.3, pE[1]], [pE[0] + wu / 2 + 0.3, pE[1]]].concat(outward > 0 ? mouth.slice().reverse() : mouth));
        P.fill(S.sleeve, { round: 0.75, dim: depth < -4, pattern: S.sleevePattern });
        // folds running down the sleeve, and the dark inside of the opening
        P.l(pE[0] - outward * 0.6, pE[1] + 1, pH[0] - outward * 1.6, wy + 1.6, 0.7).l(pE[0] + outward * 0.8, pE[1] + 1.5, pH[0] + outward * 1.8, wy + 2.8, 0.7).fill(S.sleeve, { flat: true, tone: 1 });
        P.l(pE[0] + outward * 0.1, pE[1] + 1.2, pH[0] + outward * 0.2, wy + 1.8, 0.5).fill(S.sleeve, { flat: true, tone: 3 });
        // (just wider than the wrist, so it shows as the dark of the bell round the hand, not a wedge beside it)
        P.e(pH[0], wy - 0.2, Math.min(open - 1.2, 3.3), 1.5).fill(S.sleeveLining || S.sleeve, { flat: true, tone: S.sleeveLining ? 1 : 0 });
        // trim along the hanging edge
        for (let k = 0; k < mouth.length - 1; k++) P.l(mouth[k][0], mouth[k][1] - 0.2, mouth[k + 1][0], mouth[k + 1][1] - 0.2, 0.9);
        P.fill(S.cuffs || S.trim, { round: 0.4 });
      } else if (S.sleeveStyle === 'puff') {
        // a loose shirt sleeve: full through the forearm, gathered into the cuff or bracer
        const mid = [pE[0] + (pH[0] - pE[0]) * 0.55, pE[1] + (pH[1] - pE[1]) * 0.55];
        P.t(pS[0], pS[1], pE[0], pE[1], wU + 1.6, wU + 1.2).t(pE[0], pE[1], mid[0], mid[1], wU + 1.2, wU + 2.4).t(mid[0], mid[1], pH[0], pH[1], wU + 2.4, B.fore - 0.2);
        // the loose cloth hangs below the forearm in a soft bag before the cuff
        P.e(mid[0] + outward * 0.3, mid[1] + 1.6, (wU + 2.2) / 2, 2.4).e(pE[0] + outward * 0.4, pE[1] + 0.8, (wU + 1.6) / 2, 2);
        P.fill(S.sleeve, { dim, pattern: S.sleevePattern, round: 0.8 });
        P.l(mid[0] - outward * 1, mid[1] + 3.2, pH[0] - outward * 0.2, pH[1] - 0.6, 0.6).fill(S.sleeve, { flat: true, tone: 1 });
        // soft folds where the cloth gathers
        P.l(pE[0] + outward * 0.4, pE[1] + 1, mid[0] - outward * 0.6, mid[1] + 1.4, 0.6).l(mid[0] + outward * 0.8, mid[1] - 0.5, pH[0] + outward * 0.4, pH[1] - 1, 0.6).fill(S.sleeve, { flat: true, tone: 1 });
        P.l(pS[0] + outward * 0.4, pS[1] + 2, pE[0] + outward * 0.8, pE[1] - 0.4, 0.5).fill(S.sleeve, { flat: true, tone: 3 });
      } else {
        P.t(pS[0], pS[1], pE[0], pE[1], wU + 0.6, wU - 0.4).t(pE[0], pE[1], pH[0], pH[1], wU - 0.4, wF);
        // blended arms share the body's shading so sleeve and torso read as one garment
        P.fill(S.sleeve, S.blendArms ? { flat: true, round: 0.7, dim: depth < -4, pattern: S.sleevePattern } : { dim, pattern: S.sleevePattern, contour: !S.softArms });
      }
      // sleeve cap: the top of the sleeve overlaps the shoulder so the seam reads as part of the garment
      if (S.softArms) P.e(pS[0], pS[1] + 0.6, (S.sleeveW || B.arm) / 2 + 0.6, 2.4).fill(S.sleeve, { dim, round: 0.7, pattern: S.sleevePattern });
      if (S.shirtCuff) { const pc = V.pt(hand[0], hand[1], hand[2] + 3.2); P.e(pc[0], pc[1], B.fore / 2 + 0.5, 1.2).fill(S.shirtCuff, { dim, round: 0.5 }); }
      if (S.cuffs && S.sleeveStyle !== 'bell') { const pc = V.pt(hand[0], hand[1], hand[2] + 2.8); P.e(pc[0], pc[1], wF / 2 + 0.4, 1.3).fill(S.cuffs, { dim, round: 0.5 }); }
      if (S.bracer && s < 0) { const pa = V.pt(hand[0], hand[1], hand[2] + 5.5); P.t(pa[0], pa[1] - 1.8, pa[0], pa[1] + 1.8, 4.6, 4.6).fill(S.bracer, { dim }); }
      if (S.shoulder) P.e(pS[0], pS[1] - 0.4, 3.8, 3).fill(S.shoulder, { dim, round: 0.7, lift: -0.08 });
      if (S.armArmor) S.armArmor(P, { pS, pE, pH }, dim, s, A);
      const HS = S.hands || SKIN, T = (i) => HS[Math.max(0, i - (dim ? 1 : 0))], hx = pH[0], hy = V.y(hand[2]);
      P.e(hx, hy, 2.3, 2.4).fill(HS, { dim, round: 1.25 });
      // front view: the cuff casts a soft shadow across the top of the hand, the side against the body
      // falls into shadow, the fingers curl under into a darker underside, and the outer knuckle catches the light
      if (V.fz > 0.3) {
        const inward = -outward, rx = 2.3, ry = 2.4, half = (dy) => rx * Math.sqrt(Math.max(0, 1 - (dy * dy) / (ry * ry)));
        for (let dy = -ry + 0.25; dy <= ry - 0.25; dy += 0.5) {
          const w = half(dy), y = hy + dy;
          if (w < 0.6) continue;
          P.dotOn(hx + inward * (w - 0.25), y, T(dy > 1 ? 0 : 1));
          if (dy > -1 && dy < 1.5) P.dotOn(hx + inward * (w - 0.75), y, T(2));
        }
        for (let dx = -1.75; dx <= 1.75; dx += 0.5) { P.dotOn(hx + dx, hy - ry + 0.75, T(dx * inward > 0 ? 1 : 2)); P.dotOn(hx + dx, hy + half(dx) * 0 + ry * Math.sqrt(Math.max(0, 1 - (dx * dx) / (rx * rx))) - 0.25, T(1)); }
        P.dotOn(hx + outward * 1.25, hy - 0.75, T(4)).dotOn(hx + outward * 0.75, hy - 1.25, T(3));
      }
      // a bell sleeve overhangs the hand: its shadow falls across the back of the hand, deepest at the wrist
      if (S.sleeveStyle === 'bell') for (let dx = -2.25; dx <= 2.25; dx += 0.5) { P.darken(hx + dx, hy - 2.15, 0.7, 0.3).darken(hx + dx, hy - 1.65, 0.8, 0.3).darken(hx + dx, hy - 1.15, 0.92, 0.3); }
      // finger lines: short creases across the lower half of the hand, and the crease of the thumb,
      // snapped to the fine grid so each is one clean pixel wide
      const fp = (v) => (Math.floor(v * RES) + 0.5) / RES, gx = fp(hx), gy = fp(hy);
      [-1, 0, 1].forEach((o) => P.dotOn(gx + o, gy + 0.5, T(1)).dotOn(gx + o, gy + 1, T(0)).dotOn(gx + o, gy + 1.5, T(0)));
      if (Math.abs(V.fx) < 0.8) P.dotOn(gx - outward * 1.5, gy - 0.5, T(1)).dotOn(gx - outward * 1, gy, T(0));
    });
  });

  // neck: drawn just behind the body so collars and necklines cover its base, and the head covers its top
  add(-0.5, () => {
    if (Math.abs(V.fx) > 0.3 || V.fz < -0.3 || S.female) {
      // (from the side it reaches up under the jawline, so no gap opens between the jaw and the neck)
      const n0 = V.pt(0, -0.6, 45 + 3 * V.fx * V.fx), n1 = V.pt(0, -0.4, 40.2);
      P.t(n0[0], n0[1], n1[0], n1[1], V.w(S.female ? 3.4 : 4, S.female ? 3.2 : 3.8), V.w(S.female ? 3.8 : 4.4, S.female ? 3.6 : 4)).fill(SKIN, { round: 0.7, lift: -0.14 });
      // the jaw casts a soft shadow onto the top of the neck, separating chin from throat
      if (V.fz > 0.3) P.e(V.x(0, 0.6), V.y(43.4), V.w(S.female ? 2 : 2.4, 2), 1.1).fill(SKIN, { tone: 1, flat: true });
      // seen from the side the shadow runs along the jawline (see PROFILE_FRONT), only on the neck itself,
      // so it does not hang below the chin; in three-quarter view it sits under the chin as before
      if (V.fx * V.fx > 0.9) {
        const [angle, chin] = JAWLINE[S.female ? 'fem' : 'male'], at = ([b, h]) => [V.x(0, b), V.y(HEAD + h - 0.45)];
        // (it starts a little way along from the angle of the jaw, which the hair covers)
        const p0 = at([lerp(angle[0], chin[0], 0.4), lerp(angle[1], chin[1], 0.4)]), p1 = at(chin);
        P.l(p0[0], p0[1], p1[0], p1[1], 0.9).fill(SKIN, { tone: 1, flat: true, on: true });
      } else if (Math.abs(V.fx) > 0.5) P.l(V.x(0, 1), V.y(HEAD - 7.2), V.x(0, 4.4), V.y(HEAD - 8.4), 0.8).fill(SKIN, { tone: 1, flat: true });
    }
  });

  // head and face, always drawn over the body
  add(20, () => {
    // ears: a rim, an inner fold and a lobe, all at the fine resolution
    const ears = [];
    [1, -1].forEach((s) => { if (V.d(s * 6.5, -0.5) > 0.5 && Math.abs(V.fz) < 0.9) { P.e(V.x(s * 6.3, -0.5), V.y(HEAD - 1), 1.5, 2.3); ears.push(s); } });
    P.fill(SKIN, { tone: 1 });
    ears.forEach((s) => {
      const ex = V.x(s * 6.4, -0.3), ey = V.y(HEAD - 1);
      P.dot(ex, ey - 1, SKIN[0]).dot(ex, ey - 0.5, SKIN[0]).dot(ex + 0.5 * s * V.rx, ey, SKIN[0]).dot(ex, ey + 0.5, SKIN[1]).dot(ex - 0.5, ey + 1.5, SKIN[3]);
    });
    const fem = !!S.female;
    // her lower face is rounder: a broader, shorter jaw curve gives a round chin rather than a point
    // the skull and the jaw: [forward offset, height, half-width, half-height, depth radius]
    const SKULL = fem ? [0.3, 0.9, 6.2, 7.2, 6.6] : [0.3, 0.6, 6.6, 7.6, 7];
    const JAW0 = fem ? [1.15, -4.2, 4.85, 4.1, 5.1] : [1.4, -3.3, 5.3, 5.1, 5.6];
    // Seen from the side the jaw is a slimmer shape than seen from the front: the cheek sits back from the
    // plane of the face and the jawline runs back from the chin toward the ear. Using the front view's
    // half-width there makes the lower face swell forward past the eye like a puffed cheek, so as the head
    // turns to profile the jaw narrows and moves back (pf is 0 from the front, 1 in profile).
    const pf = V.fx * V.fx;
    const JAWP = fem ? [JAW_PROFILE.fem[0], JAW0[1], JAW_PROFILE.fem[1], JAW0[3], JAW0[4]] : [JAW_PROFILE.male[0], JAW0[1], JAW_PROFILE.male[1], JAW0[3], JAW0[4]];
    const JAW = JAW0.map((v, i) => lerp(v, JAWP[i], pf));
    [SKULL, JAW].forEach(([cb, ch, w, hh]) => P.e(V.x(0, cb), V.y(HEAD + ch), w, hh));
    // in profile the nose, the lips and the chin carry the face's front edge
    if (pf > 0.9) PROFILE_FRONT(P, V, fem);
    const shade = faceShade(V, fem, SKULL, JAW);
    P.fill(SKIN, { shade: (X, Y) => { face.push(Y * PW + X); return shade(X, Y); }, dither: 0.35 });
    // the flush: pixels of the cheeks swap to the same tone of the rosier ramp, dithered at its edge
    shade.flush.forEach((w, k) => {
      const X = k % PW, Y = (k - X) / PW;
      if (w * 1.6 <= BAYER[Y & 1][X & 1] + 0.15) return;
      const i = SKIN.findIndex((c) => Math.floor(c[0]) === P.buf[k * 4] && Math.floor(c[1]) === P.buf[k * 4 + 1] && Math.floor(c[2]) === P.buf[k * 4 + 2]);
      if (i >= 0) P.dot((X + 0.5) / RES, (Y + 0.5) / RES, FLUSH[i]);
    });
    const brow = S.brow || [86, 68, 56], iris = S.iris || [86, 72, 60];
    const irisDk = iris.map((c) => c * 0.62), irisLt = iris.map((c) => Math.min(255, c * 1.3 + 14));
    const lid = [86, 58, 50], lash = [52, 38, 34], white = [228, 222, 208], blush = [196, 128, 110];
    const mix = (c, k) => SKIN[2].map((v, i) => v + (c[i] - v) * k);
    const D = (x, y, c) => P.dotOn(x, y, c);
    [1, -1].forEach((s) => {
      // the eyes sit on the curve of the face (b inside its radius), so in three-quarter view the far eye
      // turns in from the silhouette and foreshortens instead of sitting on the edge of the face
      const a = s * 2.7, b = 5.4, d = V.d(a, b);
      if (d < 1) return;
      // o points from the eye toward the outer corner on screen
      const o = Math.sign(V.x(s * 6, 6) - V.x(a, b)) || -s;
      const narrow = d < 2.6;
      const cx = Math.round(V.x(a, b) * 2) / 2, cy = Math.round(V.y(HEAD - 1) * 2) / 2;
      // the eye on a fine grid, from the inner corner (i = 0) outward: lid, three rows of eye, soft lower lid
      const at = (i, j, c) => D(cx + o * (i * 0.5 - 0.75), cy + (j - 0.5) * 0.5, c);
      const pupil = [28, 22, 22], catchlight = [244, 240, 230];
      if (narrow) {
        [1, 2].forEach((i) => at(i, -1, mix(lid, 0.85)));
        at(1, 0, iris); at(2, 0, pupil); at(1, 1, irisDk); at(2, 1, irisDk); at(1, 0.6, catchlight);
      } else {
        [0, 1, 2].forEach((i) => at(i, -1, mix(lid, 0.85)));
        at(3, -1, lash); at(4, -0.5, mix(lash, 0.6));
        if (fem) { at(0, -1, lash); at(1, -1, lash); at(2, -1, lash); at(4, -1, lash); at(5, -1.5, mix(lash, 0.7)); at(4, -1.5, mix(lash, 0.5)); }
        at(0, 0, white); at(1, 0, irisLt); at(2, 0, iris); at(3, 0, mix(white, 0.8));
        at(0, 1, mix(white, 0.75)); at(1, 1, pupil); at(2, 1, irisDk); at(3, 1, mix(white, 0.55));
        at(1, 2, irisDk); at(2, 2, mix(irisDk, 0.8));
        at(1, 0, catchlight);
        at(1, 3, mix(lid, 0.3)); at(2, 3, mix(lid, 0.3));
      }
      // brow: thick near the nose, thinning toward the outer corner
      for (let i = -1; i <= 4; i++) {
        const lift = Math.sin(Math.PI * (i + 1) / 6) * 0.5;
        const y = cy - (fem ? 1.8 : 2.3) - lift;
        D(cx + o * (i * 0.5 - 0.5), y, fem ? mix(brow, 0.85) : brow);
        if (i < 2 && !fem) D(cx + o * (i * 0.5 - 0.5), y + 0.5, mix(brow, 0.7));
      }
      // lit cheekbone
      if (!narrow) {
        if (!fem) { D(cx + o * 1, cy + 2, SKIN[3]); D(cx + o * 0.5, cy + 2, SKIN[3]); }
        // a soft flush on the cheek (not on her)
        if (!fem) { D(cx + o * 1, cy + 3, mix(blush, 0.45)); D(cx + o * 0.5, cy + 3, mix(blush, 0.35)); D(cx + o * 1.5, cy + 3, mix(blush, 0.3)); D(cx + o * 1, cy + 3.5, mix(blush, 0.25)); }
      }
    });
    // nose: shown from the front and three-quarter views; in profile it is left to the silhouette
    if (false) {
      const nx = V.x(0, 8.5), ny = V.y(HEAD - 2.6), dir = Math.sign(V.fx);
      D(nx, ny - 1, SKIN[3]); D(nx + dir * 0.5, ny - 0.5, SKIN[3]); D(nx + dir * 0.5, ny, SKIN[2]); D(nx, ny + 0.5, SKIN[1]);
      D(nx - dir * 0.5, ny + 0.5, mix(lid, 0.6)); D(nx, ny + 1, mix(lid, 0.3));
    } else if (V.fz > 0.3) {
      const nx = Math.round(V.x(0, V.fz > 0.9 ? 7.6 : 6.2) * 2) / 2, ny = V.y(HEAD - 1.5);
      if (fem) {
        D(nx, ny + 0.5, SKIN[3]); D(nx, ny + 1, SKIN[3]); D(nx, ny + 1.5, SKIN[4]); D(nx + 0.5, ny + 1.5, SKIN[1]);
        D(nx - 0.5, ny + 2, mix(lid, 0.35)); D(nx + 0.5, ny + 2, mix(lid, 0.45));
      } else {
      D(nx, ny, SKIN[3]); D(nx, ny + 0.5, SKIN[3]); D(nx, ny + 1, SKIN[3]); D(nx, ny + 1.5, SKIN[4]);
      D(nx + 0.5, ny + 0.5, SKIN[1]); D(nx + 0.5, ny + 1, SKIN[1]); D(nx + 0.5, ny + 1.5, SKIN[1]);
      D(nx - 0.5, ny + 2, mix(lid, 0.5)); D(nx + 1, ny + 2, mix(lid, 0.65)); D(nx, ny + 2.5, SKIN[1]); D(nx + 0.5, ny + 2.5, SKIN[1]);
      }
    }
    // mouth: a short upper lip line with soft corners, a lit lower lip and a shadow under it. In three-quarter
    // view it sits a little in from the face's edge, so the lips do not stand out on the outline as a pout
    if (V.fz > 0.3) {
      const front = V.fz > 0.9;
      const mx = Math.round(V.x(0, front ? 7.4 : fem ? 4.6 : 5.0) * 2) / 2, my = Math.round(V.y(HEAD - 5.2) * 2) / 2, half = front ? 1 : 0.5;
      const lipLine = fem ? [150, 76, 74] : [128, 80, 70], lipLo = fem ? [206, 124, 116] : [196, 132, 116];
      if (fem) {
        const up = [184, 104, 98];
        D(mx - 0.5, my - 0.5, mix(up, 0.8)); D(mx + 0.5, my - 0.5, mix(up, 0.8)); D(mx, my - 0.5, mix(up, 0.5));
        for (let x = -half; x <= half + 0.01; x += 0.5) D(mx + x, my + 1, mix(lipLo, 0.75));
        D(mx, my + 1, mix([236, 176, 160], 0.6));
      }
      for (let x = -half; x <= half + 0.01; x += 0.5) D(mx + x, my, lipLine);
      D(mx - half - 0.5, my, mix(lipLine, 0.45)); D(mx + half + 0.5, my, mix(lipLine, 0.45));
      for (let x = -half + 0.5; x <= half - 0.49 || x === -half + 0.5; x += 0.5) D(mx + x, my + 0.5, mix(lipLo, 0.6));
      D(mx, my + (fem ? 1.5 : 1), SKIN[1]);
    }
    // remember the finished face, so whatever is later drawn over it can be found
    faceWas = new Uint32Array(face.map((k) => (P.buf[k * 4] << 16) | (P.buf[k * 4 + 1] << 8) | P.buf[k * 4 + 2]));
  });
  // Hair, fringe and hat brims lie on the face, so they cast a shadow onto it: skin just below and to
  // the right of anything drawn over the face (away from the light) steps down a tone.
  const face = [];
  let faceWas = null;
  add(99, () => {
    if (!faceWas) return;
    const rgb = (k) => (P.buf[k * 4] << 16) | (P.buf[k * 4 + 1] << 8) | P.buf[k * 4 + 2];
    const key = (c) => (Math.floor(c[0]) << 16) | (Math.floor(c[1]) << 8) | Math.floor(c[2]);
    // each skin tone (plain or flushed) and the tone one step darker in the same ramp
    const tone = new Map(SKIN.map((c, i) => [key(c), SKIN[Math.max(0, i - 1)]]).concat(FLUSH.map((c, i) => [key(c), FLUSH[Math.max(0, i - 1)]])));
    const was = new Map(face.map((k, i) => [k, faceWas[i]]));
    const over = (k) => was.has(k) && rgb(k) !== was.get(k);
    const hits = [];
    face.forEach((k) => {
      if (over(k) || !tone.has(rgb(k))) return;
      // the shadow is two fine pixels deep, with a soft dithered third row
      const X = k % PW, Y = (k - X) / PW;
      const near = [[0, -1], [-1, -1], [0, -2], [-1, -2]].some(([dx, dy]) => over((Y + dy) * PW + X + dx));
      const far = !near && ((X + Y) & 1) === 0 && [[0, -3], [-1, -3]].some(([dx, dy]) => over((Y + dy) * PW + X + dx));
      if (near || far) hits.push(k);
    });
    hits.forEach((k) => { const c = tone.get(rgb(k)); P.buf[k * 4] = c[0]; P.buf[k * 4 + 1] = c[1]; P.buf[k * 4 + 2] = c[2]; });
  });

  if (S.gear) S.gear(add, P, A);
  parts.sort((p, q) => p.depth - q.depth).forEach((p) => p.draw());
}

const STEEL = ramp(0x7e827c, 0.08), BLADE = ramp(0x9a9d96, 0.08), GOLD = ramp(0xa8905c, 0.09), LEATHER = ramp(0x594a3d, 0.1), WOOD = ramp(0x74624c, 0.1);

// Knight: bare-headed, short dark hair and a trimmed beard. A mail hauberk to mid-thigh with mail
// sleeves to the elbow, a studded leather cuirass over it with stitched leather pauldrons, a quilted
// red gambeson showing at the forearms, leather bracers and gloves, a wide belt, wool hose and
// tall boots, and a cape.
const MAIL = ramp(0x6f736d, 0.1), HIDE = ramp(0x80603f, 0.1), HIDE_DK = ramp(0x553d2c, 0.09), GAMB = ramp(0x8a4f40, 0.1),
      HOSE = ramp(0x4f4a44, 0.09), KHAIR = ramp(0x654733, 0.1), HERALD = ramp(0x7e4a3d, 0.1), STRAP = ramp(0x47362a, 0.09);
// leather with grain, and round studs that catch light on top and cast a dot of shadow below
const studs = (x, y) => { const sx = x % 6, sy = y % 6; return (sx === 2 && sy === 2) ? 0.42 : (sx === 3 && sy === 3) ? -0.3 : grain(x, y); };
const FIGHTER = {
  body: { curve: [[15, 7.6], [20.5, 7.7], [26, 6.2], [31.5, 7.9], [34.4, 8.1]], sh: 8.2, chest: 7.9, waist: 6.2, hip: 7.6, depth: 5.2, arm: 4.3, fore: 3.7, leg: 5.6, shin: 4.3, gap: 3.5, slope: 3.2, armOut: 0.4, belt: 19.2 },
  torso: MAIL, torsoPattern: mail, sleeve: GAMB, sleevePattern: (x, y) => (y % 3 === 0 ? -0.14 : 0), legs: HOSE,
  boots: HIDE_DK, cuff: HIDE, hands: HIDE_DK, belt: STRAP, buckle: ramp(0x9a968c, 0.09), trim: GOLD, knee: HIDE,
  cape: ramp(0x6b4136, 0.1), capeLining: ramp(0x3e3a3c, 0.09), capeW: 0.82, brow: [70, 52, 40], iris: [96, 70, 50],
  chest(P, A) {
    const V = A.V, cx = V.x(0, 0);
    // hauberk skirt below the cuirass, scalloped at the hem
    A.band(23, 13.2, 1, 1.8).fill(MAIL, { pattern: mail });
    const hw = A.halfW(13.2, 2.8);
    P.e(cx, V.y(13.3), hw, 1.2).fill(MAIL, { pattern: mail });
    for (let x = Math.ceil(cx - hw + 1) * 2; x < (cx + hw - 1) * 2; x += 2) { P.dot(x / 2, V.y(12.1), MAIL[1]); P.dot(x / 2 + 0.5, V.y(11.7), MAIL[0]); }
    if (Math.abs(V.fz) > 0.4) P.l(cx, V.y(18), cx, V.y(12.6), 1).fill(MAIL, { flat: true, tone: 0 });
    // studded leather cuirass, cut away at the shoulders, with stitched edges
    A.band(36.6, 26.4, 0.4).e(cx, V.y(36.2), A.halfW(36.2, -0.8), 2.6).e(cx, V.y(26.6), A.halfW(26.6, 0.4), 1.4);
    P.fill(HIDE, { pattern: studs, round: 0.8 });
    const w0 = A.halfW(26.6, 0.4);
    for (let x = Math.ceil(cx - w0 + 1); x < cx + w0 - 1; x += 1) P.dot(x, V.y(26.6), HIDE[4]);
    // front lacing where the two halves meet
    if (V.fz > 0.25) {
      const fx = V.x(0, A.B.depth + 0.6);
      P.l(fx, V.y(36), fx, V.y(26.4), 1).fill(HIDE, { flat: true, tone: 0 });
      for (let k = 0; k < 5; k++) { const y = V.y(35 - k * 2); P.dot(fx - 1, y, HIDE[4]).dot(fx - 0.5, y + 0.5, HIDE[4]).dot(fx + 0.5, y + 0.5, HIDE[4]).dot(fx + 1, y, HIDE[4]).dot(fx, y + 1, HIDE[3]); }
    }
    // mail collar around the throat
    const cTop = V.w(3.4, 3), cBot = V.w(5.2, 4.4), jx = V.x(0, 1.2);
    P.p([[jx - cTop, V.y(44.6)], [jx + cTop, V.y(44.6)], [cx + cBot, V.y(39.4)], [cx - cBot, V.y(39.4)]]).e(cx, V.y(39.4), cBot, 1.8).fill(MAIL, { pattern: mail, round: 0.8 });
    P.e(cx, V.y(38.2), cBot - 0.6, 0.8).fill(MAIL, { flat: true, tone: 1 });
  },
  waist(P, A) {
    const V = A.V;
    // a pouch on the right hip
    if (V.d(8, 2) > -3) P.e(V.x(8.2, 2.4), V.y(17.2), 2.1, 2.4).fill(HIDE, { dim: V.d(8, 2) < 0 });
  },
  // mail sleeve to the elbow, leather bracer over the gambeson forearm, stitched leather pauldron
  armArmor(P, J, dim, s, A) {
    P.t(J.pS[0], J.pS[1] + 1, J.pE[0], J.pE[1], 4.8, 4.4).fill(MAIL, { dim, pattern: mail });
    const bx0 = J.pE[0] + (J.pH[0] - J.pE[0]) * 0.35, by0 = J.pE[1] + (J.pH[1] - J.pE[1]) * 0.35;
    P.t(bx0, by0, J.pH[0], J.pH[1] - 1.2, 4.4, 4).fill(HIDE, { dim, round: 0.7 });
    const mx = (bx0 + J.pH[0]) / 2, my = (by0 + J.pH[1]) / 2;
    P.l(mx - 2.4, my - 0.6, mx + 2.4, my - 0.6, 0.9).l(mx - 2.4, my + 1.4, mx + 2.4, my + 1.4, 0.9).fill(STRAP, { flat: true, tone: 2 });
    const ox = s * 0.8 * Math.abs(A.V.rx);
    P.e(J.pS[0] + ox * 0.6, J.pS[1] + 1.8, 3.6, 2).fill(HIDE, { dim, round: 0.7, contour: true });
    P.e(J.pS[0] + ox * 0.3, J.pS[1] - 0.2, 3.8, 2.5).fill(HIDE, { dim, round: 0.9, contour: true });
    for (let k = -4; k <= 4; k++) { const ang = k * 0.3; P.dot(J.pS[0] + ox * 0.3 + Math.sin(ang) * 3, J.pS[1] - 0.4 - Math.cos(ang) * 1.9, HIDE[4]); }
  },
  gear(add, P, A) {
    const V = A.V;
    // short hair over the crown and temples, cropped at the back
    add(20.2, () => {
      P.e(V.x(0, -0.8), V.y(HEAD + 5), 6.8, 3.8);
      if (V.fz < 0.3) P.e(V.x(0, -2.8), V.y(HEAD - 0.8), 6.4, 7);
      [1, -1].forEach((s) => { if (V.d(s * 6.2, 0) > -2) P.t(V.x(s * 6.4, 0), V.y(HEAD + 4.4), V.x(s * 6.3, 0.4), V.y(HEAD + 1.6), 2, 1.4); });
      // a few locks falling over the forehead
      [[-3, 4.2], [-0.4, 3.6], [2.4, 4]].forEach(([a, h]) => { if (V.d(a, 6.4) > 0) P.t(V.x(a, 5.8), V.y(HEAD + 6.2), V.x(a + 0.7, 6.4), V.y(HEAD + h), 2, 1); });
      P.fill(KHAIR, { round: 0.8, pattern: strands });
      for (let k = -3; k <= 3; k++) { const a = Math.sin(k * 0.42) * 6.5, b = Math.cos(k * 0.42) * 6.5 - 0.6; if (V.d(a, b) > 1) P.dot(V.x(a, b), V.y(HEAD + 5.4), KHAIR[3]); }
    });
    // trimmed beard along the jaw, shown when the face is
    add(20.4, () => {
      if (V.d(0, 6) < -1) return;
      P.p([V.pt(-5.25, 2.21, HEAD - 2.6), V.pt(-4.6, 4.4, HEAD - 7), V.pt(-2.4, 5.6, HEAD - 9.6), V.pt(0, 5.8, HEAD - 10.2), V.pt(2.4, 5.6, HEAD - 9.6), V.pt(4.6, 4.4, HEAD - 7), V.pt(5.25, 2.21, HEAD - 2.6), V.pt(4.43, 4.23, HEAD - 4.4), V.pt(2.13, 6.26, HEAD - 6.6), V.pt(-2.13, 6.26, HEAD - 6.6), V.pt(-4.43, 4.23, HEAD - 4.4)]).fill(KHAIR, { round: 0.5, lift: 0.05, pattern: (x, y) => ((x * 5 + y * 3) % 7 === 0 ? -0.22 : (x + y * 2) % 9 === 0 ? 0.14 : 0) });
      if (V.d(0, 7) > 2) P.l(V.x(-1.9, 6.8), V.y(HEAD - 4.6), V.x(1.9, 6.8), V.y(HEAD - 4.6), 1).fill(KHAIR, { flat: true, tone: 2 });
    });
  }
};

// Mage: dusty indigo robe over a paler underrobe, embroidered hem, a short mantle with a gilt edge,
// knotted sash with hanging ends, a book at the hip, wide cuffed sleeves and a pointed hat.
const ROBE = ramp(0x55607c, 0.1), UNDER = ramp(0xa29c88, 0.1), HAT = ramp(0x474e64, 0.1), BEARD = ramp(0xcdc9be, 0.08), GEM = ramp(0x84bcae, 0.11), SASH = ramp(0x7a634e, 0.1);
const MAGE = {
  // older, narrow-shouldered build with a fuller middle; sleeves are part of the robe
  body: { curve: [[15, 9.8], [21, 9.4], [27, 9], [32, 8.6], [35, 8.4]], belt: 21, sh: 8, chest: 8.6, waist: 9, hip: 9.4, depth: 6.2, arm: 5, fore: 4.4, slope: 3.6 }, blendArms: true,
  robe: ROBE, torso: ROBE, sleeve: ROBE, sleeveStyle: 'bell', cuffs: GOLD, boots: LEATHER, belt: SASH, buckle: SASH, trim: GOLD, brow: [214, 210, 200], iris: [98, 116, 128],
  chest(P, A) {
    const V = A.V;
    if (V.fz > 0.2) {
      // underrobe showing down the front opening, edged in gold
      const top = V.pt(0, 5.6, 39), bot = V.pt(0, 8.6, 5), w = 2.6 * V.fz + 0.6;
      P.p([[top[0] - w * 0.6, top[1]], [top[0] + w * 0.6, top[1]], [bot[0] + w, bot[1]], [bot[0] - w, bot[1]]]).fill(UNDER, { pattern: weave });
      P.l(top[0] - w * 0.6, top[1], bot[0] - w, bot[1], 1).l(top[0] + w * 0.6, top[1], bot[0] + w, bot[1], 1).fill(GOLD, { flat: true, tone: 2 });
    }
    // mantle over the shoulders with a gilt edge
    const cx = V.x(0, 0), w = V.w(A.B.sh + 1.8, A.B.depth + 0.8);
    P.e(cx, V.y(37.5), w, 4.6).fill(HAT);
    P.e(cx, V.y(34.2), w - 0.8, 1.2).fill(GOLD, { round: 0.4 });
    P.e(cx, V.y(37.8), w - 1.2, 3.6).fill(HAT, { lift: 0.05 });
    if (V.fz > 0.2) P.e(V.x(0, 6.6), V.y(36.5), 1.6, 1.6).fill(GEM);
  },
  waist(P, A) {
    const V = A.V;
    // sash knot and ends at the front left, and a book strapped to the right hip
    if (V.d(-3, 5.5) > 0) {
      const k = V.pt(-3, 5.8, 23.5);
      P.e(k[0], k[1], 2, 1.8).t(k[0], k[1] + 1, V.x(-3.5, 6.8), V.y(13), 1.8, 1.4).t(k[0], k[1] + 1, V.x(-1.5, 7.2), V.y(15), 1.6, 1.2).fill(SASH);
    }
    if (V.d(8, 2) > -2) {
      const bk = V.pt(8.2, 2.5, 19);
      P.p([[bk[0] - 2.4, bk[1] - 3], [bk[0] + 2.4, bk[1] - 3], [bk[0] + 2.4, bk[1] + 3], [bk[0] - 2.4, bk[1] + 3]]).fill(ramp(0x67503f, 0.1), { round: 0.6, dim: V.d(8, 2) < 0 });
      P.l(bk[0] - 2, bk[1] - 2.4, bk[0] - 2, bk[1] + 2.4, 0.8).fill(ramp(0xd2cbb6), { flat: true, tone: 2 });
    }
  },
  gear(add, P, A) {
    const V = A.V;
    add(19.9 + V.d(0, -3) * 0.1, () => { if (V.fz < 0.5) P.e(V.x(0, -2.3), V.y(HEAD - 2.5), 6.7, 6.6).fill(ramp(0x9b978f, 0.09), { lift: -0.1, pattern: (x, y) => ((x * 3 + y) % 5 === 0 ? -0.15 : 0) }); });
    add(20.5 + V.d(0, 6), () => {
      if (V.d(0, 6) < -1) return;
      P.p([V.pt(0, 1.6, HEAD - 2.6), V.pt(0, 6.6, HEAD - 3.4), V.pt(0, 6.8, HEAD - 13), V.pt(0, 6, HEAD - 17), V.pt(0, 3.2, HEAD - 11)]).p([V.pt(-4.47, 5.06, HEAD - 2), V.pt(4.47, 5.06, HEAD - 2), V.pt(3.44, 6.44, HEAD - 9), V.pt(1.29, 6.62, HEAD - 14), V.pt(0.0, 6.44, HEAD - 17 - (A.f === 1 ? 1 : 0)), V.pt(-1.29, 6.62, HEAD - 14), V.pt(-3.44, 6.44, HEAD - 9)]).fill(BEARD);
      if (V.d(0, 7) > 2) { [-2.5, 0, 2.5].forEach((a) => P.l(V.x(a, 7.5), V.y(HEAD - 5), V.x(a * 0.4, 7.2), V.y(HEAD - 13), 0.8)); P.fill(BEARD, { flat: true, tone: 1 }); }
      if (V.d(0, 7) > 2) P.l(V.x(-3, 7.6), V.y(HEAD - 3.6), V.x(3, 7.6), V.y(HEAD - 3.6), 1.4).fill(BEARD, { lift: 0.2 });
    });
    add(22, () => {
      P.e(V.x(0, 0), V.y(HEAD + 5), 13.5, 3.6).fill(HAT);
      const cb = V.x(0, 0), cm = V.x(0, -2.5), yb = V.y(HEAD + 5), ym = V.y(HEAD + 15);
      P.p([[cb - 7, yb], [cb + 7, yb], [cm + 3.6, ym], [V.x(0, -5.5), V.y(HEAD + 20.5)], [V.x(0, -9), V.y(HEAD + 21.5)], [V.x(0, -6.5), V.y(HEAD + 19)], [cm - 3.6, ym]]).fill(HAT);
      P.p([[cb - 6.8, V.y(HEAD + 8.2)], [cb + 6.8, V.y(HEAD + 8.2)], [cb + 6.3, V.y(HEAD + 6.2)], [cb - 6.3, V.y(HEAD + 6.2)]]).fill(GOLD, { round: 0.4 });
      // small stars stitched on the cone
      [[-2.5, 0, 11], [2.5, 0, 14], [0, -2, 17.5], [-1, -1, 13.5]].forEach(([a, b, h]) => { if (V.d(a, b + 3) > -1) { const x = V.x(a, b), y = V.y(HEAD + h); P.dot(x, y, GOLD[4]).dot(x - 0.5, y, GOLD[2]).dot(x + 0.5, y, GOLD[2]).dot(x, y - 0.5, GOLD[2]).dot(x, y + 0.5, GOLD[2]); } });
    });
  }
};

// Ranger: bare-headed, auburn hair cut in a bob to just below the jaw, a short olive mantle with a leaf clasp, a laced leather
// jerkin with stitched edges over a linen shirt, belt with pouch, patched trousers, folded
// boot cuffs and a bracer on the left forearm.
const HAIR = ramp(0x7e4c34, 0.1), MANTLE = ramp(0x56663f, 0.1), JERKIN = ramp(0x765a40, 0.1), SHIRT = ramp(0xaca48d, 0.09),
      BOW = ramp(0x927452, 0.1), QUIVER = ramp(0x63513f, 0.1), STRING = ramp(0xcdc6b2, 0.08);
const RANGER = {
  // the female figure and head presets: slender build, female face (see presets/)
  ...FEMALE_BODY,
  ...FEMALE_HEAD,
  // trousers: a faint diagonal twill, kept quiet so the folds and shading read
  legPattern: (x, y) => ((x + y * 2) % 6 === 0 ? -0.04 : 0),
  // Trouser finish, painted over each leg in the trousers' own tones (no off-ramp darkening, so the
  // cloth stays clean): shadow where the legs meet, a pressed crease catching the light down the front,
  // folds bunching above the knee, a stitched side seam and a small patch sewn over the knee.
  legArmor(P, J, dim, A) {
    const V = A.V, [hx, hy] = J.hip, [kx, ky] = J.knee, T = (i) => RANGER.legs[Math.max(0, i - (dim ? 1 : 0))];
    const ux = (kx - hx) / (ky - hy || 1), wAt = (y) => lerp(A.B.leg, A.B.shin + 0.4, Math.max(0, Math.min(1, (y - hy) / (ky - hy || 1)))) / 2;
    const xAt = (y) => hx + (y - hy) * ux, inner = Math.sign(V.x(0, 0) - hx) || 1, front = V.d(0, 1) > 0.15;
    for (let i = 0, y = V.y(18); y <= ky; i++, y += 0.5) {
      const x = xAt(y), w = wAt(y);
      // the inner edge turns away into the shadow between the legs
      if (Math.abs(V.fx) < 0.5) P.dotOn(x + inner * (w - 0.3), y, T(0)).dotOn(x + inner * (w - 0.8), y, T(1));
      // front crease: a lit ridge left of centre
      if (front && y < ky - 3.6) P.dotOn(x - w * 0.3, y, T(3));
      // stitched side seam on the outer leg, seen most from the side
      if (Math.abs(V.fx) > 0.35 && i % 3 === 0) P.dotOn(x - inner * w * 0.55 * Math.sign(V.fz || 1) + V.fx * 0.6, y, T(1));
    }
    // folds bunching above the knee: a dark crease with a lit edge above it
    [[4.3, 0.75], [5.4, 0.45]].forEach(([up, len]) => {
      const y = ky - up, x = xAt(y), w = wAt(y);
      for (let t = -len; t <= len; t += 0.25) { const fx = x + t * w, fy = y - t * 0.7 * inner; P.dotOn(fx, fy - 0.5, T(3)).dotOn(fx, fy, T(1)); }
    });
    // a small patch sewn over the knee, with stitches round its edge and a shadow under its lower rim
    if (V.d(0, 1) > 0.2 && !dim) {
      const K = RANGER.kneePatch, cx = kx + V.fx * 0.8, cy = ky - 2.2;
      for (let t = -1.1; t <= 1.1; t += 0.5) P.dotOn(cx + t * 0.9, cy + 1.3, T(1));
      P.e(cx, cy, 1.4, 1.1).fill(K, { round: 0.8 });
      for (let k = 0; k < 10; k++) { const a = k / 10 * Math.PI * 2; P.dotOn(cx + Math.cos(a) * 1.15, cy + Math.sin(a) * 0.85, k % 2 ? K[1] : K[3]); }
    }
  },
  // folded-over boot tops: a narrow band with a row of stitches, a shadow under the fold and another
  // cast up onto the trousers
  cuffBand: [6, 2.2],
  cuffDetail(P, top, dim) {
    const [x, y] = top, C = RANGER.cuff, L = RANGER.legs, d = dim ? 1 : 0;
    for (let t = -2.6; t <= 2.6; t += 0.5) P.dotOn(x + t, y - 3, L[1 - d]).dotOn(x + t, y - 0.3, C[0]).dotOn(x + t, y + 0.3, RANGER.boots[1 - d]);
    // fine running stitches in a pale thread, not studs
    for (let t = -2.25; t <= 2.25; t += 0.75) P.dotOn(x + t, y - 1.9, C[3 - d]);
  },
  // soft, worn leather boots: matte with a mottled grain, slouching into creases at the ankle,
  // a seam stitched round where the foot joins the shaft, and the toes scuffed paler
  bootRound: 0.7,
  bootPattern: (x, y) => grain(x, y) * 0.8 + (((x * 3 + y * 7) % 13) === 0 ? -0.1 : 0),
  bootDetail(P, J, dim, A) {
    const V = A.V, B = RANGER.boots, T = (i) => B[Math.max(0, i - (dim ? 1 : 0))], [ax, ay] = J.ankle, [tx, ty] = J.top;
    // slouch creases: short curved folds across the shaft, alternating sides, each dark with a lit lip above
    [[0.45, -0.5, 1.4], [0.75, 0.6, 1.1]].forEach(([t, off, len]) => {
      const cx = lerp(tx, ax, t) + off, cy = lerp(ty, ay, t) + 0.6;
      for (let k = -len; k <= len; k += 0.5) { const yy = cy + (k * k) / (len * len) * 0.6; P.dotOn(cx + k, yy, T(Math.abs(k) < len * 0.6 ? 1 : 2)).dotOn(cx + k, yy - 0.5, T(3)); }
    });
    // stitched seam where the foot meets the shaft, curving round the instep
    const s = J.slices, mid = s[8];
    if (Math.abs(V.fx) > 0.3) for (let k = 3; k <= 11; k++) { const q = s[k]; if (k % 2) P.dotOn(q.x, Math.max(q.y1 + 0.9, mid.y1 + 0.4), T(1)); }
    else for (let t = -0.8; t <= 0.8; t += 0.4) P.dotOn(mid.x + t * mid.sw, mid.y1 + 0.9 + t * t * 0.5, T(1));
    // scuffed toes: a few paler flecks where the leather has worn
    if (V.d(0, 1) > -0.2) [[17, -0.3, 1.1], [18, 0.2, 1.6], [16, 0.5, 1.9]].forEach(([k, o, dy]) => { const q = s[k]; P.dotOn(q.x + o * q.sw, q.y1 + dy, T(3)); });
  },
  torso: SHIRT, sleeve: ramp(0x63704a, 0.1), legs: ramp(0x5b5547, 0.1), kneePatch: ramp(0x7a6c52, 0.1),
  boots: ramp(0x684a32, 0.1), cuff: ramp(0x7a5b40, 0.1), sole: ramp(0x2f2620, 0.08), belt: ramp(0x3d342b, 0.09), buckle: ramp(0x9a978f, 0.09), trim: GOLD,
  hands: SKIN, sleeveStyle: 'puff', shirtCuff: SHIRT,
  chest(P, A) {
    const V = A.V, cx = V.x(0, 0), wc = V.w(A.B.chest + 0.4, A.B.depth + 0.5), wh = V.w(A.B.waist + 0.5, A.B.depth + 0.2);
    // jerkin over the shirt, open at the collar, laced up the front, falling just past the belt
    const hemH = (phi) => 18.6 - 1.3 * Math.pow(Math.max(0, Math.cos(phi)), 4) + 0.3 * Math.abs(Math.sin(phi * 2));
    const wHem = A.lw(19.5, 0.35), hem = A.ring(wHem, A.ld(wHem) + 0.3, hemH);
    const topY = V.y(22.5), wTop = A.halfW(22.5, 0.35);
    A.band(37.5, 22.5, 0.25, 0);
    P.p([[cx - wTop, topY], [cx + wTop, topY]].concat(hem.slice().reverse().map(([x, y]) => [x, y])));
    P.fill(JERKIN, { pattern: grain, round: 0.9 });
    // the flare over the hips catches less light, and a seam runs down each side to the hem
    hem.forEach(([x, y, d]) => { P.darken(x, y - 0.5, 0.86); P.darken(x, y - 1, 0.93); });
    // stitched hem following its curve, and the hem's shadow falling on the trousers below
    hem.forEach(([x, y, d], i) => { if (i % 2 === 0) P.dotOn(x, y - 1.4, JERKIN[4]); P.darken(x, y + 0.6, 0.72); P.darken(x, y + 1.1, 0.84); P.darken(x, y + 1.6, 0.93); });
    if (V.fz > 0.2) { const px = V.x(0, A.ld(wHem) + 0.6); P.l(px, V.y(21), px, V.y(hemH(0)) + V.d(0, A.ld(wHem)) * 0.12 - 0.4, 0.6).fill(JERKIN, { flat: true, tone: 1 }); }
    // ---- finishing: shaping seams, bound edges, stitching and brass rivets
    const onBody = (phi, h, out) => { const w = A.lw(h, 0.3 + (out || 0)); const q = V.seam(w, A.ld(w) + 0.3 + (out || 0), phi, h); return [q[0], q[1] + q[2] * 0.12, q[2]]; };
    const seamPath = (phi) => { const pts = []; for (let h = 35.4; h >= hemH(phi) + 0.8; h -= 0.7) pts.push(onBody(phi, h)); return pts; };
    // princess seams curving over the bust and down to the hem, and a seam down each side
    [-0.62, 0.62, -1.45, 1.45].forEach((phi) => {
      const pts = seamPath(phi);
      if (!pts.length || pts[Math.floor(pts.length / 2)][2] < 0.8) return;
      for (let i = 0; i < pts.length - 1; i++) P.l(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], 0.6);
      P.fill(JERKIN, { flat: true, tone: 0 });
      // a raised edge catching light beside the seam, then a line of stitches
      pts.forEach((q, i) => { P.dotOn(q[0] - 0.5, q[1], JERKIN[3]); if (i % 2 === 0) P.dotOn(q[0] + 0.7, q[1], JERKIN[4]); });
      // a brass rivet where each seam meets the bust line
      const r = pts[0];
      P.dotOn(r[0], r[1], GOLD[3]).dotOn(r[0] + 0.5, r[1] + 0.5, GOLD[1]);
    });
    // bound top edge: a darker leather binding along the neckline with a row of stitches under it
    for (let k = 0; k <= 40; k++) {
      const phi = -Math.PI / 2 + k / 40 * Math.PI, q = onBody(phi, 37.2, 0.1);
      if (q[2] < 0.3) continue;
      P.dotOn(q[0], q[1], JERKIN[0]).dotOn(q[0], q[1] + 0.5, JERKIN[1]);
      if (k % 2 === 0) P.dotOn(q[0], q[1] + 1.2, JERKIN[3]);
    }
    // tooled border above the hem: small pressed diamonds, light on top and dark below
    for (let k = 0; k <= 30; k++) {
      const phi = -Math.PI / 2 + k / 30 * Math.PI, q = onBody(phi, hemH(phi) + 2.2, 0.05);
      if (q[2] < 1 || k % 2) continue;
      P.dotOn(q[0], q[1] - 0.5, JERKIN[3]).dotOn(q[0] - 0.5, q[1], JERKIN[3]).dotOn(q[0] + 0.5, q[1], JERKIN[1]).dotOn(q[0], q[1] + 0.5, JERKIN[0]);
    }
    // bound hem: a darker band of leather following the curve of the hem
    hem.forEach(([x, y, d]) => { if (d > 0.2) { P.dotOn(x, y - 0.3, JERKIN[0]); P.dotOn(x, y - 0.8, JERKIN[1]); } });
    // rivets at the bottom of the seams
    [-0.62, 0.62].forEach((phi) => { const q = onBody(phi, hemH(phi) + 1.1); if (q[2] > 0.8) P.dotOn(q[0], q[1], GOLD[3]).dotOn(q[0] + 0.5, q[1] + 0.5, GOLD[1]); });
    const hemW = A.halfW(17, 0.95);
    if (V.fz > 0.25) {
      const fx = V.x(0, 5.7);
      P.p([[fx - 2.2 * V.fz, V.y(38)], [fx + 2.2 * V.fz, V.y(38)], [fx, V.y(32.5)]]).fill(SHIRT);
      for (let k = 0; k < 5; k++) { const y = V.y(32.5 - k * 2); P.dot(fx - 1, y, SHIRT[3]).dot(fx - 0.5, y + 0.5, SHIRT[3]).dot(fx, y + 1, SHIRT[4]).dot(fx + 0.5, y + 0.5, SHIRT[3]).dot(fx + 1, y, SHIRT[3]); P.dot(fx - 1.5, y - 0.5, GOLD[3]).dot(fx + 1.5, y - 0.5, GOLD[3]); }
    }
    if (V.fz > 0.2) P.e(V.x(4.6, 6.4), V.y(34.6), 1.5, 1.5).fill(GOLD);
  },
  waist(P, A) {
    const V = A.V;
    // belt pouch on the front of the left hip, drawn only when the front of that hip faces the viewer, never from behind
    if (V.d(-7.4, 2.5) > 0.5 && V.fz > -0.1) P.e(V.x(-7.5, 2.8), V.y(20.6), 2.4, 2.6).fill(JERKIN);
  },
  gear(add, P, A) {
    const V = A.V, B = A.B;
    // Short mantle: a ring of cloth around the shoulders, projected for each facing like the body,
    // so it keeps one smooth outline from every side. The hem dips at front and back and lifts over the arms.
    add(9, () => {
      const N = 28, ring = [];
      const hemH = (phi) => 32.6 + 1.6 * Math.sin(phi) * Math.sin(phi) - 0.4 * Math.cos(phi);
      for (let k = 0; k <= N; k++) {
        const phi = k / N * Math.PI * 2;
        const top = V.seam(4.4, 4, phi, 40.2), hem = V.seam(B.sh + 2.2, B.depth + 2, phi, hemH(phi));
        ring.push({ phi, top, hem });
      }
      for (let k = 0; k < N; k++) {
        const a = ring[k], b = ring[k + 1];
        P.p([[a.top[0], a.top[1]], [b.top[0], b.top[1]], [b.hem[0], b.hem[1]], [a.hem[0], a.hem[1]]]);
      }
      P.e(V.x(0, 0), V.y(40.1), V.w(B.sh + 0.6, B.depth + 0.8), 2);
      P.fill(MANTLE, { pattern: weave, round: 0.6, lift: -0.06 });
      // folds: shadowed lines from the shoulders to the hem, each with a lit edge, on the side facing us
      const folds = [-1.35, -0.75, -0.3, 0.3, 0.75, 1.35, Math.PI - 0.5, Math.PI + 0.5];
      folds.forEach((phi) => {
        const t = V.seam(5, 4.4, phi, 40.2), h = V.seam(B.sh + 2.1, B.depth + 1.9, phi, hemH(phi) + 0.3);
        if (h[2] > 2.6 && t[2] > 2) P.l(t[0], t[1], h[0], h[1], 0.7);
      });
      P.fill(MANTLE, { flat: true, tone: 1 });
      folds.forEach((phi) => {
        const t = V.seam(5, 4.4, phi + 0.12, 40), h = V.seam(B.sh + 2.1, B.depth + 1.9, phi + 0.14, hemH(phi + 0.14) + 0.5);
        if (h[2] > 2.6 && t[2] > 2) P.l(t[0], t[1], h[0], h[1], 0.45);
      });
      P.fill(MANTLE, { flat: true, tone: 3 });
      // light across the tops of the shoulders
      [-1, 1].forEach((sd) => { const c = V.seam(B.sh - 0.6, B.depth - 0.6, sd * Math.PI / 2, 40); if (c[2] > 1.5) P.e(c[0], c[1], 1.8, 0.7).fill(MANTLE, { flat: true, tone: 3, pattern: weave }); });
      // stitched hem and a soft shadow cast below it, where the hem faces us
      for (let k = 0; k <= 96; k++) {
        const phi = k / 96 * Math.PI * 2, h = V.seam(B.sh + 2.2, B.depth + 2, phi, hemH(phi));
        if (h[2] < 0.3) continue;
        if (h[2] > 3) P.dotOn(h[0], h[1] - 0.6, MANTLE[3]);
        P.darken(h[0], h[1] + 0.6, 0.78).darken(h[0], h[1] + 1.1, 0.86).darken(h[0], h[1] + 1.6, 0.93);
      }
      // clasp at the throat
      const cl = V.seam(4.6, 4.4, 0, 39);
      if (cl[2] > 1) {
        P.e(cl[0], cl[1], 1.4, 1.3).fill(GOLD, { round: 0.5 });
        P.dot(cl[0] - 0.5, cl[1] - 0.5, GOLD[4]);
        P.dot(cl[0] + 1, cl[1] + 1, MANTLE[3]).dot(cl[0] + 1.5, cl[1] + 1.5, MANTLE[2]);
      }
    });
    // hair: the female head preset's bob with fringe and side parting, in her auburn
    drawFemaleHead(add, P, A, HAIR);
  }
};

// Cleric: the ranger's figure and face in a long white robe that hangs in deep pleats from a black
// corded cincture, a black scapular falling front and back with a sun stitched on the breast, a soft white
// cowl draped round the throat with the hood lying on her back, long bell sleeves edged in black, and her
// hair gathered into a bun. Black and white only: the ramps are neutral greys, so the cloth's form comes
// from its folds and not from colour.
const CL_WHITE = [[124, 124, 128], [162, 162, 165], [196, 196, 197], [222, 222, 221], [241, 241, 238]];
const CL_BLACK = [[18, 18, 21], [33, 33, 37], [50, 50, 54], [72, 72, 76], [100, 100, 104]];
const CL_SOLE = [[12, 12, 14], [22, 22, 25], [34, 34, 37], [48, 48, 52], [66, 66, 70]];
// a faint linen weave, quiet enough that the folds carry the cloth
const linen = (x, y) => ((x * 3 + y) % 7 === 0 ? -0.035 : (x + y * 5) % 11 === 0 ? 0.025 : 0);
const CL = { WAIST: 26, HEM: 2.2, PLEATS: 7, HEM_A: 12.2, HEM_B: 9 };
// The robe as a surface around the body: at each height and angle round the body (phi 0 at the front,
// positive to her right), a point in figure space. Above the cord it follows the body loosely; below it
// the skirt flares toward the hem and falls into pleats that deepen as they fall, and the cloth over
// each leg is carried forward or back with the stride.
function clericSurface(A) {
  const { V, walking, ph } = A;
  const legs = [1, -1].map((s) => ({ s, fb: walking ? Math.sin(ph + (s > 0 ? 0 : Math.PI)) * 6.5 : 0 }));
  const wW = A.lw(CL.WAIST, 0.7), wD = A.ld(wW) + 0.5;
  const fall = (h) => Math.max(0, Math.min(1, (CL.WAIST - h) / (CL.WAIST - CL.HEM)));
  const radii = (h) => {
    if (h >= CL.WAIST) { const w = A.lw(h, 0.55); return [w, A.ld(w) + 0.45]; }
    const t = Math.pow(fall(h), 0.78);
    return [lerp(wW, CL.HEM_A, t), lerp(wD, CL.HEM_B, t)];
  };
  // pleats: the radius swells over each ridge and sinks into each trough, more toward the hem
  const pleat = (phi, h) => 1 + 0.12 * Math.pow(fall(h), 0.8) * (0.7 + 0.3 * Math.sin(3 * phi + 1)) * Math.cos(CL.PLEATS * phi + 0.4);
  const at = (phi, h) => {
    const [ra, rb] = radii(h), m = pleat(phi, h), t = fall(h);
    let a = ra * m * Math.sin(phi), b = rb * m * Math.cos(phi);
    // each leg carries the cloth over it: most at the hem, fading round toward the far side
    legs.forEach(({ s, fb }) => { const near = Math.max(0, Math.min(1, s * Math.sin(phi) * 1.3 + 0.55)); b += fb * 0.42 * Math.pow(t, 1.4) * near; });
    return [V.x(a, b), V.y(h), V.d(a, b)];
  };
  // the hem: rises a little over each ridge (the cloth there hangs from further out) and lifts in front
  // of the stepping leg
  const hemH = (phi) => CL.HEM + 0.45 * Math.cos(CL.PLEATS * phi + 0.4) + legs.reduce((u, { s, fb }) => u + Math.max(0, fb) * 0.16 * Math.max(0, s * Math.sin(phi) + 0.6) * Math.max(0, Math.cos(phi)), 0);
  // where a fold that reaches the hem at phi sits higher up: folds are gathered in toward the front and
  // back under the cord and spread as they fall, so they run in long diagonals rather than straight down
  const gather = (phi, h) => { const c = Math.round(phi / Math.PI) * Math.PI; return c + (phi - c) * lerp(0.62, 1, Math.pow(fall(h), 0.55)); };
  return { at, hemH, fall, legs, gather };
}
function clericRobe(P, A) {
  const W = CL_WHITE, K = CL_BLACK, { at, hemH, gather } = clericSurface(A);
  const N = 66, HS = [37.2, 35, 33, 31, 29, 27.4, CL.WAIST, 24.5, 22.5, 20, 17.5, 15, 12.5, 10, 7.5, 5.2, 3.6];
  const hOf = (h, phi) => Math.max(h, hemH(phi));
  // the robe's body: quads between rings, round the whole body, then the hem row
  for (let r = 0; r < HS.length; r++) for (let k = 0; k < N; k++) {
    const p0 = k / N * Math.PI * 2, p1 = (k + 1) / N * Math.PI * 2, h0 = HS[r], h1 = r + 1 < HS.length ? HS[r + 1] : null;
    const q = h1 == null ? [at(p0, h0), at(p1, h0), at(p1, hemH(p1)), at(p0, hemH(p0))] : [at(p0, h0), at(p1, h0), at(p1, hOf(h1, p1)), at(p0, hOf(h1, p0))];
    P.p(q.map((c) => [c[0], c[1]]));
  }
  P.fill(W, { pattern: linen, round: 1.05, dither: 0.4 });
  // Pleats: each trough a soft shadow that starts as a crease under the cord and widens as it falls,
  // the side of each fold turned from the light a half tone darker, and a lit crest along each ridge.
  // Drawn only where that part of the robe faces us.
  const trough = (k) => (Math.PI * (2 * k + 1) - 0.4) / CL.PLEATS, crest = (k) => (Math.PI * 2 * k - 0.4) / CL.PLEATS;
  const path = (phi, from, step) => { const pts = []; for (let h = from; h > hemH(phi) + 0.2; h -= step) pts.push(at(gather(phi, h), h)); pts.push(at(phi, hemH(phi) + 0.25)); return pts; };
  const stroke = (pts, w0, w1, minD) => { for (let i = 0; i < pts.length - 1; i++) if (pts[i][2] > minD && pts[i + 1][2] > minD) { const t = i / Math.max(1, pts.length - 2); P.l(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], lerp(w0, w1, t)); } };
  // the side of each fold turned from the light: a broad half-tone that widens as the fold opens out
  for (let k = 0; k < CL.PLEATS; k++) stroke(path(trough(k) - 0.13, CL.WAIST - 1, 1.2), 0.5, 3.0, 1.0);
  P.fill(W, { flat: true, tone: 2, on: true, pattern: linen });
  // the deep of each trough, narrower and darker, fading out before it reaches the cord
  for (let k = 0; k < CL.PLEATS; k++) stroke(path(trough(k), CL.WAIST - 3.5, 1.2), 0.5, 1.5, 1.3);
  P.fill(W, { flat: true, tone: 1, on: true });
  // the very bottom of each trough near the hem, where little light reaches
  for (let k = 0; k < CL.PLEATS; k++) { const pts = path(trough(k), 11, 1.2); stroke(pts, 0.5, 0.7, 2); }
  P.fill(W, { flat: true, tone: 0, on: true });
  // the crest of each ridge, catching the light
  for (let k = 0; k < CL.PLEATS; k++) stroke(path(crest(k) + 0.08, CL.WAIST - 2, 1.2), 0.5, 1.0, 1.8);
  P.fill(W, { flat: true, tone: 4, on: true });
  // above the cord the bodice is gathered: short soft folds fanning up from the cord toward the bust
  [-0.9, -0.5, -0.15, 0.2, 0.55, 0.95].forEach((phi, i) => {
    const a = at(phi, CL.WAIST + 0.5), b = at(phi * 0.8, CL.WAIST + 3.2 + (i % 2) * 1.2);
    if (a[2] > 1.5 && b[2] > 1.5) P.l(a[0], a[1], b[0], b[1], 0.5);
  });
  P.fill(W, { flat: true, tone: 2, on: true });
  // the skirt blouses out a little under the cord, which shades the cloth just beneath it
  for (let k = 0; k <= 120; k++) { const phi = k / 120 * Math.PI * 2, q = at(phi, CL.WAIST - 0.9), q2 = at(phi, CL.WAIST - 1.6); if (q[2] > 0.5) { P.darken(q[0], q[1], 0.88, 0); P.darken(q2[0], q2[1], 0.95, 0); } }
  // the hem: a black band following every rise and dip, the inside of the hem showing as a dark lip where
  // the cloth turns back on itself at the troughs, and the robe's shadow on the ground-facing boots below
  for (let k = 0; k <= 160; k++) {
    const phi = k / 160 * Math.PI * 2, q = at(phi, hemH(phi) + 0.35), q2 = at(phi, hemH(phi) + 1.05);
    if (q[2] < -0.4) continue;
    P.dotOn(q[0], q[1], K[q[2] > 3 ? 2 : 1]).dotOn(q2[0], q2[1], K[q[2] > 3 ? 3 : 2]);
  }
}
// the scapular: a black panel hanging from the shoulders to just above the hem, in front and behind,
// lying on the robe; it swings with the cloth under it, with a soft crease down it and a sun stitched on
// the breast in front
function clericScapular(P, A) {
  const V = A.V, K = CL_BLACK, W = CL_WHITE, { at } = clericSurface(A);
  [0, Math.PI].forEach((c0) => {
    const facing = V.d(0, Math.cos(c0)) > 0.18;
    if (!facing) return;
    const half = (h) => (h > CL.WAIST ? 0.52 : 0.46 - 0.12 * (CL.WAIST - h) / CL.WAIST);
    const L = [], R = [];
    for (let h = 36.4; h >= 7.4; h -= 0.8) { const w = half(h); L.push(at(c0 - w, h)); R.push(at(c0 + w, h)); }
    // the bottom edge is cut square, with a small point at the middle
    const tip = at(c0, 6.6);
    P.p(L.map((q) => [q[0], q[1]]).concat([[tip[0], tip[1]]], R.reverse().map((q) => [q[0], q[1]]))).fill(K, { round: 0.8, pattern: linen, lift: 0.04 });
    // a crease down the panel where it bends over the body, lit along its edge
    const crease = []; for (let h = 30; h >= 9; h -= 1.2) crease.push(at(c0 + 0.08, h));
    for (let i = 0; i < crease.length - 1; i++) if (crease[i][2] > 2) P.l(crease[i][0], crease[i][1], crease[i + 1][0], crease[i + 1][1], 0.5);
    P.fill(K, { flat: true, tone: 0, on: true });
    // a white running stitch inset along each long edge and across the foot
    for (let h = 33; h >= 8.4; h -= 1.5) { const w = half(h) - 0.09; [at(c0 - w, h), at(c0 + w, h)].forEach((q) => { if (q[2] > 1.5) P.dotOn(q[0], q[1], W[2]); }); }
    // the sun on the breast: a ring of short rays round a small disc, in white thread
    if (c0 === 0) {
      const s = at(0, 32.4);
      if (s[2] > 3) {
        P.dotOn(s[0], s[1], W[4]).dotOn(s[0] - 0.5, s[1], W[3]).dotOn(s[0] + 0.5, s[1], W[3]).dotOn(s[0], s[1] - 0.5, W[4]).dotOn(s[0], s[1] + 0.5, W[2]);
        for (let k = 0; k < 8; k++) { const a = k / 8 * Math.PI * 2; P.dotOn(s[0] + Math.cos(a) * 1.5 * Math.abs(V.fz), s[1] + Math.sin(a) * 1.5, W[k < 4 ? 2 : 3]); }
      }
    }
  });
}
// the cincture: a black cord round the waist over the scapular, knotted at her left hip, with two ends
// hanging to the knee and finished in small tassels that swing with the stride
function clericCord(P, A) {
  const K = CL_BLACK, { at } = clericSurface(A);
  const ring = []; for (let k = 0; k <= 64; k++) { const phi = k / 64 * Math.PI * 2, q = at(phi, CL.WAIST + 0.2); ring.push(q); }
  for (let k = 0; k < 64; k++) { const a = ring[k], b = ring[k + 1]; if (a[2] > 0.4 && b[2] > 0.4) P.l(a[0], a[1], b[0], b[1], 1.1); }
  P.fill(K, { round: 0.5 });
  // the twist of the cord: a lit fleck every few pixels along it
  for (let k = 0; k < 64; k += 2) { const q = ring[k]; if (q[2] > 1.5) P.dotOn(q[0], q[1] - 0.25, K[4]); }
  const kphi = -0.75, knot = at(kphi, CL.WAIST + 0.1);
  if (knot[2] < 0.4) return;
  const sw = A.walking ? Math.sin(A.ph) * 0.9 : 0;
  [[-0.45, 13.4, 0], [0.35, 15.6, 0.6]].forEach(([o, endH, lag]) => {
    const end = at(kphi + o * 0.12, endH), x1 = end[0] + sw * (1 + lag), mid = [lerp(knot[0], x1, 0.5) + o * 0.6, lerp(knot[1], end[1], 0.5)];
    P.l(knot[0] + o, knot[1] + 0.6, mid[0], mid[1], 0.8).l(mid[0], mid[1], x1, end[1], 0.8);
    // the tassel: a binding, then a little brush of threads
    P.e(x1, end[1] + 0.4, 0.75, 0.5).t(x1, end[1] + 0.8, x1 + sw * 0.3, end[1] + 2.8, 1.5, 1.1);
  });
  P.fill(K, { round: 0.6 });
  [[-0.45, 13.4, 0], [0.35, 15.6, 0.6]].forEach(([o, endH, lag]) => { const end = at(kphi + o * 0.12, endH), x1 = end[0] + sw * (1 + lag); P.dotOn(x1 - 0.25, end[1] + 0.4, K[4]).dotOn(x1, end[1] + 1.6, K[3]).dotOn(x1 + 0.4, end[1] + 2.2, K[3]); });
  P.e(knot[0], knot[1] + 0.2, 1.3, 1.1).fill(K, { round: 0.9 });
  P.dotOn(knot[0] - 0.5, knot[1] - 0.25, K[4]);
}
// the cowl: soft white cloth draped round the throat, falling in loose U-shaped folds over the breast and
// lying on the shoulders; and behind, the hood resting on her back, its black lining showing at the rim
function clericCowl(P, A) {
  const V = A.V, B = A.B, W = CL_WHITE, K = CL_BLACK;
  // the hem dips lowest at the front where the cloth hangs in a deep U, rises over the shoulders
  const hemH = (phi) => 37.2 - 3.4 * Math.pow(Math.max(0, Math.cos(phi)), 2) - 1.4 * Math.pow(Math.max(0, -Math.cos(phi)), 2) - 1.0 * Math.pow(Math.sin(phi), 4);
  const top = (phi) => V.seam(4.3, 4.0, phi, 40.6), bot = (phi) => V.seam(B.sh - 0.5, B.depth + 0.8, phi, hemH(phi));
  const N = 40;
  for (let k = 0; k < N; k++) {
    const p0 = k / N * Math.PI * 2, p1 = (k + 1) / N * Math.PI * 2, a0 = top(p0), a1 = top(p1), b0 = bot(p0), b1 = bot(p1);
    P.p([[a0[0], a0[1]], [a1[0], a1[1]], [b1[0], b1[1]], [b0[0], b0[1]]]);
  }
  P.e(V.x(0, 0), V.y(40.4), V.w(4.6, 4.2), 1.6);
  P.fill(W, { pattern: linen, round: 0.75, dither: 0.4, lift: 0.04 });
  // The drape: the cowl hangs in three loose U-curves across the breast, one inside the other, each a soft
  // band of shadow under a lit roll of cloth, deepest at the middle and easing out toward the shoulders
  // where the cloth lies flat. Then shorter folds run from the throat out over each shoulder.
  const fold = (frac, sag, phiMax, lift = 0) => { const pts = []; for (let k = 0; k <= 26; k++) { const phi = -phiMax + k / 26 * phiMax * 2, c = Math.cos(phi); const h = lerp(40.4, hemH(phi), frac) - sag * c * c + lift; const w = lerp(4.4, B.sh - 0.5, frac), d = lerp(4.1, B.depth + 0.8, frac) + 0.2; pts.push(V.seam(w, d, phi, h)); } return pts; };
  const draw = (pts, minD, w0, w1 = w0) => { for (let i = 0; i < pts.length - 1; i++) if (pts[i][2] > minD && pts[i + 1][2] > minD) { const t = Math.abs(i / (pts.length - 2) - 0.5) * 2; P.l(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], lerp(w1, w0, t)); } };
  const U = [[0.3, 0.7, 0.95], [0.56, 1.1, 1.2], [0.82, 1.5, 1.42]];
  // the soft shadow band under each roll, wide at the middle
  U.forEach(([fr, sag, pm]) => draw(fold(fr, sag, pm), 1.4, 0.5, 1.6));
  P.fill(W, { flat: true, tone: 2, on: true, pattern: linen });
  // the deepest line of each, at the middle only
  U.forEach(([fr, sag, pm]) => draw(fold(fr, sag, pm * 0.7, -0.2), 2.2, 0.5, 0.7));
  P.fill(W, { flat: true, tone: 1, on: true });
  // the lit roll just above each shadow
  U.forEach(([fr, sag, pm]) => draw(fold(fr - 0.12, sag, pm * 0.85), 2, 0.5, 0.8));
  P.fill(W, { flat: true, tone: 4, on: true });
  // folds from the throat out over each shoulder and down the back
  [-1.25, -1.75, -2.4, 1.25, 1.75, 2.4, Math.PI].forEach((phi, i) => {
    const a = top(phi), b = bot(phi + 0.12 * Math.sign(phi)), m = [lerp(a[0], b[0], 0.55), lerp(a[1], b[1], 0.55)];
    if (a[2] > 1 && b[2] > 1) { P.l(a[0], a[1] + 0.8, m[0], m[1], 0.5).l(m[0], m[1], b[0], b[1] - 0.5, i % 3 === 1 ? 0.6 : 0.9); }
  });
  P.fill(W, { flat: true, tone: 2, on: true });
  // the cowl's edge: no hard line, but the cloth turning under itself (a darker lip) with a lit edge just
  // above, and a soft shadow on the robe and sleeves below that is deepest where the cowl hangs lowest
  for (let k = 0; k <= 160; k++) {
    const phi = k / 160 * Math.PI * 2, q = bot(phi), deep = Math.max(0, Math.cos(phi)) ** 2;
    if (q[2] < 0.9) continue;
    // (the lip fades out toward the sides, where the cowl lies smooth over the arm)
    if (q[2] > 2.2) P.dotOn(q[0], q[1] - 0.25, W[q[2] > 3.4 ? 1 : 2]).dotOn(q[0], q[1] - 0.75, W[4]);
    P.darken(q[0], q[1] + 0.3, 0.8 - 0.06 * deep, 0).darken(q[0], q[1] + 0.8, 0.88, 0).darken(q[0], q[1] + 1.3, 0.95, 0);
  }
  // the hood resting on her back, seen from the side and behind: a soft bag of cloth hanging from the
  // cowl to a rounded point between the shoulder blades, its mouth a black crescent of lining at the top
  if (V.fz < 0.3) {
    const D = -B.depth - 1.9, mouthW = B.sh - 1.1;
    // outline: from one shoulder, swelling out and down to the point, back up to the other shoulder
    const side = (s) => { const pts = []; for (let k = 0; k <= 10; k++) { const t = k / 10, a = s * mouthW * (1 + 0.18 * Math.sin(t * Math.PI * 0.9)) * (1 - Math.pow(t, 2.2)), h = lerp(38.8, 28.6, t); pts.push(V.pt(a, D - 0.35 * Math.sin(t * Math.PI), h)); } return pts; };
    const L = side(-1), Rr = side(1).reverse();
    P.p(L.concat(Rr)).fill(W, { pattern: linen, round: 1.1, lift: -0.03 });
    // the mouth: the rim curves up at the shoulders and dips in the middle, the lining showing under it
    const rim = (u, h0, sag) => V.pt(u * mouthW * 0.98, D, h0 - sag * (1 - u * u));
    const top = [], low = [];
    for (let k = 0; k <= 12; k++) { const u = -1 + k / 6; top.push(rim(u, 39.0, 0.9)); low.push(rim(u * 0.9, 38.6, 2.6)); }
    P.p(top.concat(low.reverse())).fill(K, { round: 0.5, on: true });
    // the lit roll of the rim over the lining
    for (let i = 0; i < top.length - 1; i++) P.l(top[i][0], top[i][1] - 0.4, top[i + 1][0], top[i + 1][1] - 0.4, 0.5);
    P.fill(W, { flat: true, tone: 4, on: true });
    // folds sagging from the rim's ends toward the point, and a deep crease down the middle
    [-1, 1].forEach((s) => {
      const p0 = rim(s * 0.75, 38.6, 2.0), p1 = V.pt(s * 0.6, D, 30.2), p2 = rim(s * 0.4, 38.6, 2.6), p3 = V.pt(s * 0.25, D, 31.6);
      P.l(p0[0], p0[1] + 0.5, p1[0], p1[1], 0.7).l(p2[0], p2[1] + 0.6, p3[0], p3[1], 0.45);
    });
    P.fill(W, { flat: true, tone: 1, on: true });
    [-1, 1].forEach((s) => { const p0 = rim(s * 0.92, 38.6, 1.6), p1 = V.pt(s * 1.3, D, 31.4); P.l(p0[0], p0[1] + 0.6, p1[0], p1[1], 0.5); });
    P.fill(W, { flat: true, tone: 4, on: true });
    const pt = V.pt(0, D, 28.6);
    P.dotOn(pt[0], pt[1] - 0.6, W[1]);
  }
}
const CLERIC = {
  ...FEMALE_BODY,
  ...FEMALE_HEAD,
  hairStyle: 'bun',
  // the robe hides the legs; it is drawn by drawRobe below instead of the mage's straight robe
  robe: CL_WHITE, drawRobe(P, A) { clericRobe(P, A); clericScapular(P, A); clericCord(P, A); },
  torso: CL_WHITE, sleeve: CL_WHITE, sleeveStyle: 'bell', sleeveLining: CL_BLACK, cuffs: CL_BLACK, trim: CL_BLACK, sleevePattern: linen,
  boots: CL_BLACK, sole: CL_SOLE, belt: CL_BLACK, buckle: CL_BLACK, hands: SKIN, bootRound: 0.8,
  gear(add, P, A) {
    add(9, () => clericCowl(P, A));
    drawFemaleHead(add, P, A, HAIR);
  }
};

// paint one frame of a character (facing row, frame col) into a PW x PH RGBA buffer, top row first
function paintFrame(S, row, col) {
  const P = Painter();
  drawFigure(P, S, row * Math.PI / 4, col);
  P.outline();
  return P.buf;
}

// paint every facing and frame of one character into a single RGBA sheet (row 0 at the top)
function paintSheet(S) {
  const W = PW * COLS, H = PH * ROWS, data = new Uint8Array(W * H * 4);
  for (let row = 0; row < ROWS; row++) for (let col = 0; col < COLS; col++) {
    const P = Painter();
    drawFigure(P, S, row * Math.PI / 4, col);
    P.outline();
    for (let y = 0; y < PH; y++) {
      const dst = ((H - 1 - (row * PH + y)) * W + col * PW) * 4;
      data.set(P.buf.subarray(y * PW * 4, (y + 1) * PW * 4), dst);
    }
  }
  return { data, W, H };
}
// ---------- end sprites ----------

export { PW, PH, RES, FW, FH, COLS, ROWS, WALK, FIGHTER, MAGE, RANGER, CLERIC, paintSheet, paintFrame };
