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
// legLen: extra sprite pixels of leg. Heights between the ankle (4) and the hip (22) are stretched to make
// room, and everything above the hip rises by legLen, so all the parts placed by height stay together.
const ANKLE_H = 4, HIP_H = 22;
function View(theta, bob, sway, legLen) {
  sway = sway || 0;
  const L = legLen || 0, k = (HIP_H - ANKLE_H + L) / (HIP_H - ANKLE_H);
  const lift = (h) => (h <= ANKLE_H ? h : h <= HIP_H ? ANKLE_H + (h - ANKLE_H) * k : h + L);
  const unlift = (u) => (u <= ANKLE_H ? u : u <= HIP_H + L ? ANKLE_H + (u - ANKLE_H) / k : u - L);
  const fx = Math.sin(theta), fz = Math.cos(theta), rx = -Math.cos(theta), rz = Math.sin(theta);
  // The walk's bob and sway move the whole figure. They are snapped to whole fine pixels, so the figure moves
  // rigidly: a shift of a fraction of a pixel lands the shapes and the details snapped to the pixel grid (eyes,
  // mouth) on different sides of a pixel, and the face comes out subtly rearranged in those frames.
  bob = Math.round(bob * RES) / RES;
  const shift = Math.round(sway * rx * RES) / RES;
  const V = {
    fx, fz, rx, rz, bob,
    x: (a, b) => CX + shift + a * rx + b * fx,
    y: (h) => GY - lift(h) + bob,
    // the height whose point lands on screen row y (the inverse of V.y)
    h: (y) => unlift(GY + bob - y),
    d: (a, b) => a * rz + b * fz,
    pt: (a, b, h) => [V.x(a, b), V.y(h)],
    w: (A, B) => Math.sqrt(A * A * rx * rx + B * B * fx * fx),
    seam: (A, B, phi, h) => [V.x(A * Math.sin(phi), B * Math.cos(phi)), V.y(h), V.d(A * Math.sin(phi), B * Math.cos(phi))]
  };
  return V;
}
// The face's form, as a brightness for each fine pixel (for P.fill's shade). The light comes from the
// sculpted head below (its depth map gives each pixel a normal), lit from the upper left and in front. On top
// of that light sit painted accents for the planes of the face: eye sockets under
// the brow ridge, a lit cheekbone with a hollow under it, the nose (a lit ridge, a shadow side and the
// shadow it casts on the lip), the lips and the ball of the chin. All are placed in the head's own
// coordinates, so they turn with it and foreshorten in the three-quarter views.
// the jawline in profile: the angle of the jaw below the ear and the point of the chin (forward offset, height)
// (matches the back edge of the jaw in HEAD_SHAPE)
const JAWLINE = { male: [[-0.8, -6.0], [3.3, -8.3]], fem: [[-0.6, -5.8], [3.1, -7.9]] };
// ---- The sculpted head ----
// The head is modelled as a stack of horizontal cross-sections in head space (a across toward the head's
// right, f forward, h height from HEAD). Each section is a superellipse, flatter across the front so the face
// reads as planes, with its widest point at f = m. Its half-width W, the front of the face plane Ff and the
// back of the head Fb all change with height, so the jaw narrows to the chin and its back edge rises from the
// chin to the angle below the ear. On the face plane sit the features: the midline profile (brow, the dip at
// the bridge, the nose, the lips with the mouth between them, the chin), the hollows of the eye sockets, the
// cheekbones and the corners of the mouth. The outline in every facing and the light on the face both come
// from this one model, so the views agree with each other.
// piecewise-linear lookup in a table of [h, value] rows, sorted from the top of the head down
const pw = (tab, h) => {
  if (h >= tab[0][0]) return tab[0][1];
  for (let i = 1; i < tab.length; i++) if (h >= tab[i][0]) { const [h0, v0] = tab[i - 1], [h1, v1] = tab[i]; return lerp(v1, v0, (h - h1) / (h0 - h1)); }
  return tab[tab.length - 1][1];
};
const HEAD_SHAPE = {
  fem: {
    top: 8.1, bottom: -8.1,
    W: [[8.1, 0], [7.3, 3.3], [6, 5.0], [4, 5.9], [2, 6.2], [0.5, 6.2], [-1, 6.05], [-2, 5.85], [-3, 5.5], [-4, 5.05], [-5, 4.55], [-6, 3.85], [-7, 3.05], [-7.6, 2.2], [-8.1, 0.8]],
    Ff: [[8.1, 0.4], [7.3, 3.2], [6, 4.8], [4, 5.6], [3, 5.85], [1, 6.05], [0, 6.1], [-1, 5.95], [-2, 5.9], [-3, 5.85], [-4, 5.8], [-5, 5.7], [-6, 5.45], [-7, 5.15], [-7.6, 4.8], [-7.9, 4.4], [-8.1, 3.9]],
    Fb: [[8.1, 0.2], [7.3, -3.6], [6, -5.6], [4, -6.5], [2, -6.7], [0, -6.5], [-2, -5.8], [-3, -4.9], [-4, -3.6], [-5, -2.0], [-5.8, -0.6], [-6.4, 0.6], [-7.0, 1.8], [-7.6, 2.9], [-8.1, 3.4]],
    m: [[8.1, 0.3], [-1, 0.3], [-3, 0.8], [-5, 1.8], [-7, 3.0], [-8.1, 3.6]],
    // the midline profile, the front edge seen from the side
    P: [[3, 5.85], [1.2, 6.1], [0.2, 6.35], [-0.8, 6.15], [-1.9, 6.6], [-2.5, 6.98], [-2.85, 7.18], [-3.2, 7.15], [-3.45, 6.85], [-3.6, 6.3], [-4.3, 6.25], [-4.85, 6.1], [-5.05, 5.9], [-5.5, 6.05], [-6.1, 5.55], [-6.8, 5.75], [-7.4, 5.35], [-7.85, 4.6], [-8.1, 3.9]],
    nose: 0.8
  },
  male: {
    top: 8.2, bottom: -8.7,
    W: [[8.2, 0], [7.4, 3.6], [6, 5.4], [4, 6.3], [2, 6.6], [0.5, 6.6], [-1, 6.45], [-2, 6.3], [-3, 6.0], [-4, 5.6], [-5, 5.2], [-6, 4.6], [-7, 3.8], [-7.9, 2.9], [-8.4, 2.0], [-8.7, 0.8]],
    Ff: [[8.2, 0.4], [7.4, 3.4], [6, 5.0], [4, 5.9], [3, 6.15], [1, 6.4], [0, 6.45], [-1, 6.25], [-2, 6.2], [-3, 6.15], [-4, 6.1], [-5, 6.0], [-6, 5.8], [-7, 5.5], [-7.9, 5.1], [-8.4, 4.6], [-8.7, 4.0]],
    Fb: [[8.2, 0.2], [7.4, -3.8], [6, -5.9], [4, -6.9], [2, -7.1], [0, -6.9], [-2, -6.2], [-3, -5.3], [-4, -4.0], [-5, -2.4], [-6, -0.8], [-6.8, 0.6], [-7.5, 1.9], [-8.1, 3.0], [-8.5, 3.6], [-8.7, 3.8]],
    m: [[8.2, 0.3], [-1, 0.3], [-3, 0.9], [-5, 1.9], [-7, 3.3], [-8.7, 3.9]],
    P: [[3, 6.15], [1.2, 6.45], [0.2, 6.75], [-0.9, 6.4], [-2, 6.95], [-2.7, 7.5], [-3.05, 7.72], [-3.4, 7.68], [-3.7, 7.3], [-3.9, 6.6], [-4.6, 6.55], [-5.15, 6.4], [-5.3, 6.2], [-5.75, 6.4], [-6.4, 5.85], [-7.2, 6.25], [-7.9, 5.95], [-8.4, 5.0], [-8.7, 4.0]],
    nose: 0.95
  }
};
// the superellipse exponent across the front of the head: 2 (a plain ellipse) over the crown, where the hair
// shells are built on a round skull, flattening to NF over the face so it reads as planes
const NF = 2.7;
const nf = (h) => (h > 2 ? 2 : h > -1 ? lerp(NF, 2, (h + 1) / 3) : NF);
function headModel(fem) {
  const T = HEAD_SHAPE[fem ? 'fem' : 'male'];
  const g = (x, y, sx, sy) => Math.exp(-(x * x) / (sx * sx) - (y * y) / (sy * sy));
  // how far the midline profile stands out of the face plane, and how wide across that feature is
  // (the nose is narrow at the bridge and widens down to its tip)
  const featW = (h) => (h > -0.6 ? 2.4 : h > -3.6 ? lerp(0.45, T.nose, Math.min(1, (-0.6 - h) / 2.4)) : h > -5.9 ? 1.5 : 1.9);
  // the front surface: forward offset at (a, h), without the features (base) and with them
  const base = (a, h) => {
    const W = pw(T.W, h), m = pw(T.m, h), Ff = pw(T.Ff, h), u = Math.min(1, Math.abs(a) / Math.max(W, 1e-3));
    const n = nf(h);
    return m + (Ff - m) * Math.pow(1 - Math.pow(u, n), 1 / n);
  };
  // the features, as an offset outward from the section's centre
  const feature = (a, h) => {
    const aa = Math.abs(a), W = pw(T.W, h);
    const front = Math.max(0, 1 - Math.pow(aa / Math.max(W, 1e-3), 4));
    // (the profile table starts at the forehead; above it the crown is left as the plain section)
    let o = h > T.P[0][0] ? 0 : (pw(T.P, h) - pw(T.Ff, h)) * g(a, 0, featW(h), 1);
    // the wings of the nostrils swell out beside the tip
    o += (fem ? 0.4 : 0.5) * g(aa - (fem ? 0.72 : 0.85), h + (fem ? 3.3 : 3.6), 0.38, 0.42);
    // the eye sockets sink in under the brow; the cheekbones stand out below and outside them
    o -= (fem ? 0.4 : 0.5) * g(aa - 2.6, h + 1, 1.05, 0.85);
    o += (fem ? 0.3 : 0.35) * g(aa - 3.9, h + 2.4, 1.1, 0.95);
    // the corners of the mouth turn in, and (on him) the cheek hollows under the cheekbone
    o -= 0.28 * g(aa - 1.55, h + (fem ? 5.05 : 5.3), 0.5, 0.6);
    if (!fem) o -= 0.25 * g(aa - 4.1, h + 4.9, 1, 1.2);
    // the temples flatten a little above the outer corner of the eye
    o -= 0.15 * g(aa - 4.8, h - 0.8, 0.9, 1.4);
    return o * front;
  };
  // one cross-section as a closed loop of [a, f] points (front from right to left, then the back)
  const section = (h) => {
    const W = pw(T.W, h), m = pw(T.m, h), Fb = pw(T.Fb, h), pts = [];
    if (W <= 0.05) return pts;
    // (sampled densely across the front, so the narrow ridge of the nose is caught)
    const N = 72;
    for (let i = 0; i <= N; i++) {
      const u = Math.PI / 2 - i / N * Math.PI, a = W * Math.sin(u), f = base(a, h);
      // push the point outward from the section's centre by the features at it
      const o = feature(a, h), dx = a, df = f - m, r = Math.hypot(dx, df) || 1;
      pts.push([a + dx / r * o, f + df / r * o]);
    }
    for (let i = 1; i < 24; i++) {
      const u = -Math.PI / 2 + i / 24 * Math.PI;
      pts.push([W * Math.sin(u), m - (m - Fb) * Math.cos(u)]);
    }
    return pts;
  };
  const front = (a, h) => { const f = base(a, h), o = feature(a, h), m = pw(T.m, h), df = f - m, r = Math.hypot(a, df) || 1; return [a + a / r * o, f + df / r * o]; };
  return { T, section, front };
}
// Render the head model into the painter's mask and a depth map: for each fine pixel row, the section at
// that height is projected and filled between its extremes on screen; each pixel keeps the depth of the
// nearest surface under it and the head-space point there.
function rasterHead(P, V, model) {
  const { T } = model, map = new Map();
  const Y0 = Math.floor(V.y(HEAD + T.top) * RES), Y1 = Math.ceil(V.y(HEAD + T.bottom) * RES);
  const outline = [];
  for (let Y = Y0; Y <= Y1; Y++) {
    const sy = (Y + 0.5) / RES, h = V.h(sy) - HEAD, pts = model.section(h);
    if (pts.length < 3) continue;
    const scr = pts.map(([a, f]) => [V.x(a, f), V.d(a, f), a, f]);
    let x0 = 1e9, x1 = -1e9;
    scr.forEach(([x]) => { x0 = Math.min(x0, x); x1 = Math.max(x1, x); });
    outline.push([Y, x0, x1]);
    for (let X = Math.floor(x0 * RES); X <= Math.ceil(x1 * RES); X++) {
      const sx = (X + 0.5) / RES;
      if (sx < x0 || sx > x1) continue;
      let best = null;
      for (let i = 0; i < scr.length; i++) {
        const p = scr[i], q = scr[(i + 1) % scr.length];
        if ((p[0] - sx) * (q[0] - sx) > 0 || p[0] === q[0] && p[0] !== sx) continue;
        const t = p[0] === q[0] ? 0.5 : (sx - p[0]) / (q[0] - p[0]), d = lerp(p[1], q[1], t);
        if (!best || d > best.d) best = { d, a: lerp(p[2], q[2], t), f: lerp(p[3], q[3], t) };
      }
      if (best) map.set(Y * PW + X, { ...best, h });
    }
  }
  // fill the outline, one row at a time
  outline.forEach(([Y, x0, x1]) => P.p([[x0, Y / RES], [x1, Y / RES], [x1, (Y + 1) / RES], [x0, (Y + 1) / RES]]));
  return map;
}
const FACE_LIGHT = (() => { const l = [-0.5, 0.55, 0.67], n = Math.hypot(...l); return l.map((c) => c / n); })();
function faceShade(V, fem, map) {
  const g = (x, y, sx, sy) => Math.exp(-(x * x) / (sx * sx) - (y * y) / (sy * sy));
  // In profile the light is turned to come from the side the face points to. From the upper left it grazes a
  // head facing right, lighting only the back of the cheek beside the hair and leaving the front of the face
  // dull, which reads as wrong; mirrored, the right profile is lit the way the left one is. (fx^4: the
  // three-quarter views keep most of the usual light.)
  const turnK = Math.pow(V.fx, 4), sideK = 1 - turnK + turnK * (-Math.sign(V.fx) || 1);
  const Lraw = [FACE_LIGHT[0] * sideK, FACE_LIGHT[1], FACE_LIGHT[2]], Ln = Math.hypot(...Lraw);
  const [Lx, Ly, Lz] = Lraw.map((c) => c / Ln);
  // the light's direction across the head (toward its right, a), so the nose's sides can face it or not
  const La = V.rx * Lx + V.rz * Lz;
  // the slope of the depth map at a pixel, along one screen axis: a central difference where both
  // neighbours are on the head, one-sided at the outline (where the surface turns away steeply)
  const step = 1 / RES, MAXS = 5;
  const slope = (k, dk, here) => {
    const p = map.get(k + dk), q = map.get(k - dk);
    let s = p && q ? (p.d - q.d) / (2 * step) : p ? (p.d - here) / step : q ? (here - q.d) / step : 0;
    // a neighbour on a different surface (across the edge of the nose, say) makes a jump, not a slope
    if (!(p && q) && Math.abs(s) > MAXS) s = Math.sign(s) * MAXS;
    return Math.max(-MAXS, Math.min(MAXS, s));
  };
  // the raw slopes are stepped (each section is sampled on its own), so they are averaged over each pixel's
  // 3x3 neighbourhood on the head before lighting, or the skin comes out mottled
  const raw = new Map();
  map.forEach((pt, k) => raw.set(k, [-slope(k, 1, pt.d), slope(k, PW, pt.d)]));
  const smooth = (k) => {
    let sx = 0, sy = 0, n = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const r = raw.get(k + dy * PW + dx); if (r) { sx += r[0]; sy += r[1]; n++; } }
    return [sx / n, sy / n];
  };
  const shade = (X, Y) => {
    const pk = Y * PW + X, pt = map.get(pk);
    if (!pt) return 0.4;
    // the normal in view space: x to screen right, y up, z toward the viewer
    const [nx, ny] = smooth(pk), nz = 1, nl = Math.hypot(nx, ny, nz);
    const sy = (Y + 0.5) / RES;
    // light with a firm turn into shadow: the core shadow is darkest just past the turn, and the far rim
    // picks up a little light reflected back from the shoulders
    const l = (nx * Lx + ny * Ly + nz * Lz) / nl;
    let b = 0.1 + 0.66 * Math.max(0, Math.min(1, (l + 0.08) / 1.0));
    if (l < -0.08) b += 0.09 * Math.min(1, Math.max(0, (-0.3 - l) * 2.5)) - 0.04 * Math.min(1, (-0.08 - l) * 5);
    // the underside of the jaw turns away from the light into the shadow over the neck
    if (ny / nl < -0.5) b -= (-0.5 - ny / nl) * 0.4;
    // head space: a across (toward the head's right), f forward, h height relative to the head's centre
    const a = pt.a, f = pt.f, h = V.h(sy) - HEAD;
    // only the front of the head carries the face
    const front = Math.max(0, Math.min(1, (f - 1.5) / 3));
    if (front <= 0) return b;
    let k = 0;
    const aa = Math.abs(a);
    // eye sockets: shadow under the brow ridge, deepest at the inner corner beside the bridge of the nose
    k -= 0.18 * g(aa - 2.3, h + 0.5, 1.35, 1.05) + 0.1 * g(aa - 1.2, h + 0.2, 0.6, 0.9);
    // the brow ridge and the forehead above it catch the light
    // (on her, softly: a bright band right under the fringe reads as the rim of a cap)
    k += (fem ? 0.03 : 0.09) * g(aa - 2.3, h - 1.8, 2, 0.6) + 0.07 * g(a + 0.6, h - 4.2, 3, 1.6);
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
    const flush = front * g(aa - 3.2, h + 3.5, 1.5, 1.05) * (1 - 0.7 * V.fx * V.fx);
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

// Seen from the front an arm hangs in the same plane as the torso and reads as a flat strip pasted to its side,
// the more so when sleeve and body are one cloth (mail on mail, white on white). This pass gives it a round
// form and sets it in front of the body: across each row of the arm a lit rim on the side toward the light
// (screen left), a core shadow turning away on the other side, the edge against the body darker still where it
// is hidden from the light, and a crease on the inside of the elbow. Then the arm casts a soft neutral shadow
// onto the body beside it, so a clear line of dark parts the two. under: the frame as it was before the arm.
function armForm(P, under, { pS, pE, pH, outward, cx, bell, shaded }) {
  const buf = P.buf, inward = -outward;
  const changed = (k) => buf[k * 4 + 3] && (buf[k * 4] !== under[k * 4] || buf[k * 4 + 1] !== under[k * 4 + 1] || buf[k * 4 + 2] !== under[k * 4 + 2] || !under[k * 4 + 3]);
  const mix = (k, t, c) => { for (let j = 0; j < 3; j++) buf[k * 4 + j] = buf[k * 4 + j] + (c[j] - buf[k * 4 + j]) * t; };
  const dk = (k, f) => { buf[k * 4] *= f; buf[k * 4 + 1] *= f; buf[k * 4 + 2] *= f; };
  const LIGHT = [255, 246, 222];
  const Y0 = Math.max(0, Math.floor((pS[1] + 2.4) * RES)), Y1 = Math.min(PH - 1, Math.ceil((pH[1] - 0.6) * RES));
  const xMin = Math.max(0, Math.floor((Math.min(pS[0], pE[0], pH[0]) - 6) * RES)), xMax = Math.min(PW - 1, Math.ceil((Math.max(pS[0], pE[0], pH[0]) + 6) * RES));
  const elbowY = pE[1] * RES;
  for (let Y = Y0; Y <= Y1; Y++) {
    // the arm's span in this row: the run of its own pixels nearest the arm's line
    const t = Math.max(0, Math.min(1, (Y / RES - pS[1]) / Math.max(1, pH[1] - pS[1])));
    const ax = (t < 0.5 ? lerp(pS[0], pE[0], t * 2) : lerp(pE[0], pH[0], (t - 0.5) * 2)) * RES;
    let a = Math.round(ax), b;
    if (!changed(Y * PW + a)) { let f = -1; for (let d = 1; d < 6 && f < 0; d++) { if (changed(Y * PW + a + d)) f = a + d; else if (changed(Y * PW + a - d)) f = a - d; } if (f < 0) continue; a = f; }
    b = a;
    while (a - 1 >= xMin && changed(Y * PW + a - 1)) a--;
    while (b + 1 <= xMax && changed(Y * PW + b + 1)) b++;
    const w = b - a + 1;
    if (w < 3) continue;
    const L = Y * PW + a, R = Y * PW + b, innerK = inward > 0 ? R : L;
    if (!bell) {
      // the lit rim, brighter on the arm whose outer side faces the light
      mix(L, outward < 0 ? 0.3 : 0.16, LIGHT);
      if (w > 5) mix(L + 1, outward < 0 ? 0.14 : 0.07, LIGHT);
      // the core shadow on the side turned from the light, and the hidden edge against the body
      { dk(R, 0.74); if (w > 5) dk(R - 1, 0.86); if (w > 8) dk(R - 2, 0.94); }
      dk(innerK, inward > 0 ? 0.88 : 0.82);
      // the crease on the inside of the elbow: a short soft dark notch
      const de = Math.abs(Y - elbowY);
      if (de < 2.5) { const k = innerK - inward * 1; dk(k, de < 1.5 ? 0.82 : 0.9); }
    }
    // the cast shadow on the body beside the arm (the puff sleeve casts its own), fading in below the shoulder
    if (Y > (pS[1] + 3.2) * RES) {
      const fade = Math.min(1, (Y / RES - pS[1] - 3.2) / 2.5);
      [0.62, 0.76, 0.9].forEach((f, j) => {
        const X = (inward > 0 ? b : a) + inward * (j + 1), k = Y * PW + X;
        if (X < 0 || X >= PW || changed(k) || !buf[k * 4 + 3]) return;
        // (only onto the body, not past the far side of it)
        if ((X / RES - cx) * outward < 0) return;
        dk(k, 1 - (1 - f) * fade);
      });
    }
  }
}

// how far down the screen a point drops per unit it stands toward the viewer (legs and shoes)
const LEG_TILT = 0.2;
function drawFigure(P, S, theta, f) {
  // Walk: one smooth cycle over eight frames. swing is the right leg's stride (-1 back .. 1 forward);
  // the body rises as the legs pass and dips as each foot lands, and the weight shifts over the planted foot.
  // (the cycle is sampled half a step off the passing position, so no frame has both feet side by side: from
  // the side the legs would stack into one, and from the front they would close together)
  const walking = f >= 2, step = f - 2, ph = walking ? (step + 0.5) / WALK * Math.PI * 2 : 0;
  const swing = walking ? Math.sin(ph) : 0;
  const bob = walking ? 0.4 - 1.5 * (1 - Math.abs(swing)) + 0.6 * Math.abs(swing) : (f === 1 ? 1 : 0);
  const V = View(theta, bob, walking ? Math.cos(ph) * 0.7 : 0, S.body && S.body.legLen);
  const parts = [];
  const add = (depth, draw) => parts.push({ depth, draw });
  // body proportions: shoulder, chest, waist and hip half-widths, depth, limb widths, leg spacing
  const B = Object.assign({ sh: 9, chest: 8.6, waist: 7.2, hip: 7.8, depth: 5.4, arm: 4.8, fore: 4, leg: 6.2, shin: 4.8, gap: 3.8, slope: 3 }, S.body);
  const A = { V, swing, ph, f, walking, B, S };
  // The resting pose (the idle frames) stands with the weight on the right leg: that foot sits a little back
  // and under the body, the left foot is eased forward and out with its knee soft, and the arms are held a
  // little off the body, bent at the elbow, the right hand further forward than the left. It gives the
  // figure depth from every side, where a pose with both sides alike reads flat.
  // stride(s): how far forward (b) the foot of leg s stands, for the legs and for any cloth that follows them
  // At rest the feet stand well apart, wide to the sides and one ahead of the other: the free leg forward with
  // its knee eased, the weight leg back under the hips. Which foot is forward is chosen per facing (A.free, the
  // side of the free leg): from some three-quarter views the step forward would carry the left foot straight
  // behind the right on screen and the legs would stack into one, so there the other foot steps forward.
  // (the step is longest seen from the side, where it is the only thing parting the legs: there the boots,
  // long from heel to toe, would otherwise overlap and the trousers run together)
  const reach = 1 + 0.5 * V.fx * V.fx, REST_FWD = 5.4 * reach, REST_BACK = -2.6 * reach;
  A.free = -1;
  if (!walking) {
    const sepX = (free) => Math.abs([1, -1].reduce((d, s) => d + s * V.x(s * B.gap * (s === free ? 1.7 : 1.55), s === free ? REST_FWD : REST_BACK), 0));
    if (sepX(1) > sepX(-1) + 1) A.free = 1;
  }
  A.stride = (s) => (walking ? Math.sin(ph + (s > 0 ? 0 : Math.PI)) * 6.5 : s === A.free ? REST_FWD : REST_BACK);
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
    const ps = ph + (s > 0 ? 0 : Math.PI), fb = A.stride(s), lift = walking ? Math.max(0, Math.cos(ps)) * 3.2 : 0;
    // at rest the free (left) leg's knee eases forward and its foot turns out a little; the weight leg is straight
    // The feet stand out from the hips (out: the ankle's distance out, as a multiple of the hip's), about as wide
    // as the shoulders at rest, so from the front and three-quarters a clear gap of ground shows between the legs
    // all the way up to the crotch; the knee lies on the straight line between hip and ankle.
    const free = !walking && s === A.free, ease = free ? 1 : 0, out = walking ? 1.3 : free ? 1.7 : 1.55, kneeOut = (1 + out) / 2;
    const toeUp = walking ? Math.max(0, Math.sin(ps)) * 0.8 - Math.max(0, -Math.sin(ps)) * 0.6 : 0;
    const hipA = s * B.gap, depth = V.d(hipA, fb), dim = depth < -1;
    add(depth * 0.05 - 4, () => {
      // The knee and ankle are tipped down by their depth (LEG_TILT, as the shoe below is), as if seen from a
      // little above: a foot ahead of the other then sits lower on screen, so it reads as ahead from the front too.
      const tip = (a, b, h, k) => { const q = V.pt(a, b, h); q[1] += V.d(a, b) * LEG_TILT * k; return q; };
      // (the hip follows the stride a little and the knee more than half, so seen from the side the thighs part
      // just below the crotch and the shins stand clear of each other)
      const hip = V.pt(hipA, fb * 0.3, 22), knee = tip(hipA * (kneeOut + ease * 0.06), fb * 0.58 + 0.8 + lift * 1.1 + ease * 0.9, 11.5 + lift * 0.8 - ease * 0.3, 0.5), ankle = tip(hipA * out, fb, 4 + lift, 1);
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
      // (the shoe sits under the ankle: the free leg's foot at rest is set further out)
      const fa = hipA * out, TILT = LEG_TILT, ua = SH.back / SH.len, BALL = 0.7, HEEL = ua * 0.85;
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
      // At rest the feet turn out, the free one well out (yaw, in radians): from the front that shoe shows its
      // side and toe at an angle while the weight foot points at us, so the two read as one foot ahead of the
      // other. The slices run along the turned foot, and each is as wide on screen as the foot is across it.
      const yaw = walking ? 0 : free ? 0.45 : 0.12, ya = s * Math.sin(yaw), yb = Math.cos(yaw);
      const across = Math.abs(yb * V.rx - ya * V.fx);
      const slices = [];
      for (let k = 0; k <= 20; k++) {
        const u = k / 20, b = -SH.back + u * SH.len, base = lift + pitch(b), fa2 = fa + ya * b, fb2 = fb + yb * b, d = V.d(fa2, fb2), arch = archAt(u);
        slices.push({ u, x: V.x(fa2, fb2), sw: Math.max(0.55, footW(u) * across), g: V.y(base) + d * TILT, y0: V.y(base + arch) + d * TILT, y1: V.y(base + footH(u)) + d * TILT, d });
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
      // (the shaft stands straight above its own ankle, tipped by depth as the ankle is: from a fixed point by
      // the inner leg it would lean out to a foot set wide, and the boot would look drawn from another angle)
      const top = tip(hipA * (out - 0.02), fb * 0.85, (S.robe ? 5 : 9.5) + lift, 1);
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
    // (a little narrower than the hips and not too tall, so from behind it reads as a seat, not a bulge)
    P.e(V.x(0, 0.4), V.y(18.8), V.w(B.hip - 1.1, B.depth - 0.4), 4.1).fill(S.legs, { pattern: S.legPattern, round: 0.8 });
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
    // At rest the arms hang close from the shoulder, the elbows just off the waist and a little back, and the
    // forearms swing forward, the right further than the left (see A.stride): [out, forward, up] for the hand.
    // (Holding the arms out from the shoulder instead reads stiff, like a mannequin.)
    // (restArms overrides these per character: [right, left])
    const rest = walking ? [0, 0, 0] : S.restArms ? S.restArms[s > 0 ? 0 : 1] : s > 0 ? [0.5, 1.9, 0.3] : [0.3, 1.0, 0.1];
    // On the weight side (right) the hand rests on the hip at rest: the elbow pushes out to the side and the
    // forearm angles back in, leaving a clear space between arm and body that echoes the weight shift in the
    // legs, so the pose reads from the front too. (Bell sleeves keep the arm hanging: on a bent arm the bell
    // would hang wrong.)
    // (a spec with akimbo: false keeps both arms relaxed at rest)
    const akimbo = !walking && !grip && s > 0 && S.sleeveStyle !== 'bell' && S.akimbo !== false;
    // (the resting hands are placed from the hips they rest on or hang beside, not from the shoulders: on a
    // slim figure, placed from the shoulders they float off the body)
    const hangA = B.sh + 1.4 + (S.blendArms ? 0.6 : B.armOut == null ? 1.2 : B.armOut) + rest[0];
    // A relaxed arm at rest is bent a little at the elbow: the elbow sits back by the waist and the forearm comes
    // forward, the hand a little raised. From the front the forearm is then foreshortened and the hand stands in
    // front of the hip rather than beside it, so the arm reads as reaching toward us instead of lying flat
    // against the body. (Bell sleeves hang straight: a bent bell would hang wrong.)
    const bend = !walking && !grip && !akimbo && S.sleeveStyle !== 'bell' ? 1 : 0;
    const hand = grip || (akimbo ? [s * (A.lw(22.6) - 0.4), A.ld(A.lw(22.6)) * 0.45, 22.6] : [s * (walking ? hangA : bend ? Math.min(hangA, A.lw(23) + 1.4) : Math.min(hangA + 0.6, A.lw(21.5) + 3.4)), hb + 0.6 + rest[1] + bend * 2.6, 22 + Math.max(0, hb) * 0.35 + rest[2] + bend * 1.5]);
    const sh = [s * (B.sh - 1), 0, 38.8 - B.slope * 0.3];
    // The hand on the hip rests on the side of it, a little forward, fingers curling over the hip toward the
    // front: seen from the front the whole hand shows on the hip (a hand tucked behind it left only a stub of
    // cuff at the waist, which read as a square from any distance), and the arm stands in front of the body.
    const depth = V.d(s * 11, hand[1] * 0.5) - (S.blendArms && Math.abs(V.fz) > 0.6 ? 0.5 : 0), dim = depth < -2;
    A['hand' + (s > 0 ? 'R' : 'L')] = { a: hand[0], b: hand[1], h: hand[2], depth };
    add(depth, () => {
      // (what lay here before the arm, so the arm's own pixels can be told apart for its form pass below)
      const formPass = V.fz > 0.3, under = formPass ? P.buf.slice() : null;
      // (at rest the elbow sits just off the waist and back, so the arm bends softly instead of hanging straight)
      // At rest the upper arm hangs nearly straight down from the shoulder (the elbow only a little outside it)
      // and the forearm angles slightly out to the hand beside the hip, as a relaxed arm does. An elbow set
      // further out splays the upper arm and makes the whole arm look as if it is dangling.
      // (far enough out that the full sleeve clears the narrow waist, leaving a gap between arm and jerkin)
      // (elbowTuck draws the resting elbows in toward a narrow waist, so the forearm angles softly out to the
      // hand beside the thigh and the arm bends a little at the elbow instead of hanging as one straight slant)
      const restElbow = Math.abs(sh[0]) + 2.6 + (S.blendArms ? 0.3 : 0) - (S.elbowTuck || 0);
      const elbow = akimbo ? [s * (B.sh + 1.7), -2.0, 30.4] : [s * (walking ? B.sh + 0.9 + (S.blendArms ? 0.4 : (B.armOut == null ? 1.2 : B.armOut) * 0.5) : (bend ? Math.max(restElbow, Math.abs(hand[0]) + 1.3) : Math.min(restElbow, Math.abs(hand[0]) - 0.3))), walking ? hand[1] * 0.45 - 1.4 : -0.9 - bend * 1.4, (sh[2] + hand[2]) / 2 + 0.8 - bend * 0.6];
      // a point up the forearm from the hand, dist along it toward the elbow, so the wrist, the cuffs and the
      // bracer follow the forearm when it angles forward instead of sitting straight above the hand
      // Points on the arm are tipped down by their depth like the knees and ankles (LEG_TILT): seen a little from
      // above, a hand held forward sits lower on screen than one held back, so the arms stand at different depths
      // as the feet do instead of lying in one flat plane. The elbow takes half the tilt, the shoulder none.
      const tip = (q, a, b, k) => { q[1] += V.d(a, b) * LEG_TILT * k; return q; };
      const fore = (dist) => { const d = [elbow[0] - hand[0], elbow[1] - hand[1], elbow[2] - hand[2]], L = Math.hypot(...d) || 1, a = hand[0] + d[0] / L * dist, b = hand[1] + d[1] / L * dist; return tip(V.pt(a, b, hand[2] + d[2] / L * dist), a, b, 1); };
      const pS = V.pt(...sh), pE = tip(V.pt(...elbow), elbow[0], elbow[1], 0.5), pH = fore(1.6);
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
        // A loose shirt sleeve, full through the forearm and gathered into the cuff or bracer. Its outline is one
        // smooth taper (bulges added for the bagging cloth made it look crumpled from three-quarters): a slight
        // fullness over the forearm that narrows into the cuff, a lit line down the upper arm, and one fold
        // where the cloth gathers above the cuff.
        // The arm follows one smooth curve from shoulder to wrist that bends round the elbow (passing just inside
        // it), so a bent arm turns in an arc instead of two straight pieces meeting at a point: a quadratic whose
        // control point lies a little past the elbow. Q(t) is a point on it, Tq(t) its direction, W(t) its width.
        const ctl = [pE[0] * 1.1 - (pS[0] + pH[0]) * 0.05, pE[1] * 1.1 - (pS[1] + pH[1]) * 0.05];
        const Q = (t) => [(1 - t) * (1 - t) * pS[0] + 2 * t * (1 - t) * ctl[0] + t * t * pH[0], (1 - t) * (1 - t) * pS[1] + 2 * t * (1 - t) * ctl[1] + t * t * pH[1]];
        const Tq = (t) => { const x = 2 * (1 - t) * (ctl[0] - pS[0]) + 2 * t * (pH[0] - ctl[0]), y = 2 * (1 - t) * (ctl[1] - pS[1]) + 2 * t * (pH[1] - ctl[1]), L = Math.hypot(x, y) || 1; return [x / L, y / L]; };
        const W = (t) => (t < 0.5 ? lerp(wU + 0.3, wU + 0.1, t * 2) : t < 0.75 ? lerp(wU + 0.1, wU - 0.1, (t - 0.5) * 4) : lerp(wU - 0.1, B.fore, (t - 0.75) * 4));
        const NQ = 14, segs = [];
        for (let k = 0; k < NQ; k++) { const t0 = k / NQ, t1 = (k + 1) / NQ; segs.push([Q(t0), Q(t1), W(t0), W(t1), t0]); }
        segs.forEach(([a, b, w0, w1]) => P.t(a[0], a[1], b[0], b[1], w0, w1));
        // the cloth: a soft weave with a little grain, so the sleeve has a surface rather than a flat fill. Seen
        // from the side the thin arm would be lit almost across its width, so there it sits a little darker.
        const side = V.fx * V.fx, cloth = S.sleevePattern || ((x, y) => weave(x, y) * 0.8 + grain(x, y) * 0.5);
        P.fill(S.sleeve, { dim, pattern: cloth, round: 0.9, lift: -0.1 * side });
        // details land on the sleeve's own pixels only, never on the jerkin or the hand beside it
        const SLV = new Set(S.sleeve.map((c) => (Math.floor(c[0]) << 16) | (Math.floor(c[1]) << 8) | Math.floor(c[2])));
        const onSleeve = (k) => SLV.has((P.buf[k * 4] << 16) | (P.buf[k * 4 + 1] << 8) | P.buf[k * 4 + 2]);
        const T = (i) => Math.max(0, i - (dim ? 1 : 0));
        // (the mantle above casts its own shadow onto the top of the sleeve, following its hem; see RANGER.gear)
        // a fold down the upper arm, on the side away from the light
        const u0 = Q(0.14), u1 = Q(0.44);
        P.l(u0[0] - outward * 0.8, u0[1], u1[0] - outward * 0.9, u1[1], 0.5).fill(S.sleeve, { flat: true, tone: T(1), on: onSleeve });
        P.l(u0[0] + outward * 0.6, u0[1] - 0.2, u1[0] + outward * 0.7, u1[1] - 0.2, 0.5).fill(S.sleeve, { flat: true, tone: T(3), on: onSleeve });
        // The inner edge of the sleeve, toward the body, turns away from us into shadow, and the arm casts a soft
        // neutral shadow onto the body just beside it, so the arm reads as standing in front of the torso rather
        // than as one flat shape with it (seen from the front and three-quarters)
        if (V.fz > 0.3) {
          const inward = -outward;
          segs.forEach(([a, b, w0, w1, t0]) => {
            const n = Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) * RES * 2);
            for (let i = 0; i <= n; i++) {
              const t = i / n, x = lerp(a[0], b[0], t), y = lerp(a[1], b[1], t), hw = lerp(w0, w1, t) / 2;
              const kIn = Math.floor(y * RES) * PW + Math.floor((x + inward * (hw - 0.3)) * RES);
              if (onSleeve(kIn)) P.dot(x + inward * (hw - 0.3), y, S.sleeve[T(1)]);
              const kIn2 = Math.floor(y * RES) * PW + Math.floor((x + inward * (hw - 0.8)) * RES);
              if (i % 2 === 0 && onSleeve(kIn2)) P.dot(x + inward * (hw - 0.8), y, S.sleeve[T(2)]);
              [0.35, 0.85].forEach((o, j) => {
                const sx = x + inward * (hw + o), k = Math.floor(y * RES) * PW + Math.floor(sx * RES);
                if (!onSleeve(k) && P.buf[k * 4 + 3]) P.darken(sx, y, j ? 0.9 : 0.8, 0);
              });
              // Below the elbow, whatever shows through the gap between the arm and the body (long hair falling
              // behind the back, mostly) lies deep in their shadow: darkened, it reads as a gap, so the arm
              // stands clear of the jerkin instead of running into hair of nearly the same brown.
              if (t0 >= 0.5) {
                const h = V.h(y), xt = V.x(0, 0) + outward * A.halfW(h, 0.45), xi = x + inward * (hw + 1.2);
                for (let gx = Math.min(xt, xi); gx <= Math.max(xt, xi); gx += 0.5) {
                  const k = Math.floor(y * RES) * PW + Math.floor(gx * RES);
                  if ((gx - xt) * outward > 0 && (xi - gx) * outward > 0 && !onSleeve(k) && P.buf[k * 4 + 3]) P.darken(gx, y, 0.62, 0);
                }
              }
            }
          });
        }
        // gathers across the forearm where the loose cloth bunches above the cuff: two soft U-shaped creases,
        // each a dark trough with a lit lip just above it, the lower one shorter as the sleeve narrows
        [[0.38, (wU + 0.6) / 2 - 0.5, -0.15], [0.72, (wU + 0.4) / 2 - 0.6, 0.2]].forEach(([t, hw2, skew]) => {
          const [cx, cy] = Q(0.5 + t * 0.5), [ux, uy] = Tq(0.5 + t * 0.5), nx = -uy, ny = ux, at = (u, lift) => [cx + nx * u * hw2 + ux * (0.7 * (1 - u * u) - lift), cy + ny * u * hw2 + uy * (0.7 * (1 - u * u) - lift)];
          for (let u = -0.85; u < 0.85; u += 0.15) { const a = at(u + skew * 0.3, 0), b = at(u + 0.15 + skew * 0.3, 0); P.l(a[0], a[1], b[0], b[1], 0.5); }
          P.fill(S.sleeve, { flat: true, tone: T(1), on: onSleeve });
          for (let u = -0.6; u < 0.6; u += 0.15) { const a = at(u + skew * 0.3, 0.6), b = at(u + 0.15 + skew * 0.3, 0.6); P.l(a[0], a[1], b[0], b[1], 0.5); }
          P.fill(S.sleeve, { flat: true, tone: T(3), on: onSleeve });
        });
      } else {
        P.t(pS[0], pS[1], pE[0], pE[1], wU + 0.6, wU - 0.4).t(pE[0], pE[1], pH[0], pH[1], wU - 0.4, wF);
        // blended arms share the body's shading so sleeve and torso read as one garment
        P.fill(S.sleeve, S.blendArms ? { flat: true, round: 0.7, dim: depth < -4, pattern: S.sleevePattern } : { dim, pattern: S.sleevePattern, contour: !S.softArms });
      }
      // sleeve cap: the top of the sleeve overlaps the shoulder so the seam reads as part of the garment
      if (S.softArms) P.e(pS[0], pS[1] + 0.6, (S.sleeveW || B.arm) / 2 + 0.6, 2.4).fill(S.sleeve, { dim, round: 0.7, pattern: S.sleevePattern });
      // (on the hip the cuffs are drawn with the hand, turned to its angle; see below)
      // Cuffs: the end of the sleeve itself, the last few units before the wrist, turned to the cuff's cloth. Its
      // pixels keep the sleeve's own shading, recoloured tone for tone, so the cuff follows the sleeve's form and
      // light from every side and at any angle of the forearm, and can never stand off the arm's outline (a
      // separate band laid over a slanted forearm juts past it like a bandage). A darker rim runs where it meets
      // the hand and a lit edge where it meets the sleeve above.
      const SLV0 = S.sleeve.map((c) => (Math.floor(c[0]) << 16) | (Math.floor(c[1]) << 8) | Math.floor(c[2]));
      const cuffOnSleeve = (C, len) => {
        // (from just above the hand, which covers the sleeve's last stretch, up the forearm)
        const w0 = fore(1.9), w1 = fore(1.9 + len), ex = w1[0] - w0[0], ey = w1[1] - w0[1], L = Math.hypot(ex, ey) || 1, ux = ex / L, uy = ey / L;
        const X0 = Math.floor((Math.min(w0[0], w1[0]) - 4) * RES), X1 = Math.ceil((Math.max(w0[0], w1[0]) + 4) * RES);
        const Y0 = Math.floor((Math.min(w0[1], w1[1]) - 4) * RES), Y1 = Math.ceil((Math.max(w0[1], w1[1]) + 4) * RES);
        for (let Y = Math.max(0, Y0); Y <= Math.min(PH - 1, Y1); Y++) for (let X = Math.max(0, X0); X <= Math.min(PW - 1, X1); X++) {
          const k = Y * PW + X, i = SLV0.indexOf((P.buf[k * 4] << 16) | (P.buf[k * 4 + 1] << 8) | P.buf[k * 4 + 2]);
          if (i < 0) continue;
          const t = ((X + 0.5) / RES - w0[0]) * ux + ((Y + 0.5) / RES - w0[1]) * uy;
          if (t < -0.8 || t > L) continue;
          // (a tone lighter than the sleeve, as the linen is a lighter cloth; the rim at the hand a tone darker)
          const tone = Math.max(0, Math.min(4, i + (t < 0.45 ? 0 : t > L - 0.55 ? 2 : 1)));
          P.dot((X + 0.5) / RES, (Y + 0.5) / RES, C[tone]);
        }
      };
      const cuffC = S.shirtCuff || (S.sleeveStyle !== 'bell' && S.cuffs);
      // (on the hip arm a short cuff: a long pale one there reads as a block standing in for the hand)
      if (cuffC) cuffOnSleeve(cuffC, akimbo ? 1.5 : S.shirtCuff ? 2.7 : 2.4);
      if (S.bracer && s < 0) { const pa = fore(5.5); P.t(pa[0], pa[1] - 1.8, pa[0], pa[1] + 1.8, 4.6, 4.6).fill(S.bracer, { dim }); }
      if (S.shoulder) P.e(pS[0], pS[1] - 0.4, 3.8, 3).fill(S.shoulder, { dim, round: 0.7, lift: -0.08 });
      if (S.armArmor) S.armArmor(P, { pS, pE, pH }, dim, s, A);
      if (formPass) armForm(P, under, { pS, pE, pH, outward, cx: V.x(0, 0), bell: S.sleeveStyle === 'bell', shaded: S.sleeveStyle === 'puff' });
      // (a hand resting on the back of the hip is out of sight from the front: the sleeve ends at the cuff)
      const HS = S.hands || SKIN, T = (i) => HS[Math.max(0, i - (dim ? 1 : 0))];
      // The hand is drawn in its own frame: x across it, y down it toward the fingertips. A hanging hand points
      // straight down; a hand on the hip turns with the forearm (most of its angle, the wrist bending a little
      // against the hip), so the fingers point down and in along the arm instead of straight down.
      let hx = pH[0], hy = V.y(hand[2]) + V.d(hand[0], hand[1]) * LEG_TILT, ux = 0, uy = 1;
      if (akimbo) {
        const c = tip(V.pt(hand[0], hand[1], hand[2]), hand[0], hand[1], 1), dx = c[0] - pE[0], dy = c[1] - pE[1], ang = Math.atan2(dx, dy) * 0.8;
        hx = c[0]; hy = c[1]; ux = Math.sin(ang); uy = Math.cos(ang);
      }
      // (handScale shrinks the hand and everything drawn on it, for slim figures whose forearms are narrower)
      const hk = S.handScale || 1;
      // (a hanging hand rises by what it lost at the top, so it still meets the cuff at the wrist)
      if (!akimbo) hy -= (1 - hk) * 2.4;
      const H = (x, y) => [hx + (x * uy + y * ux) * hk, hy + (-x * ux + y * uy) * hk];
      const Dh = (x, y, c) => { const q = H(x, y); P.dotOn(q[0], q[1], c); };
      const ellipseAt = (cx, cy, rx, ry) => { const pts = []; for (let k = 0; k < 24; k++) { const t = k / 24 * Math.PI * 2; pts.push(H(cx + Math.cos(t) * rx, cy + Math.sin(t) * ry)); } return P.p(pts); };
      if (akimbo) {
        // On the hip. Its own simple shading
        // (the hanging hand's assumes fingers pointing straight down): light along the knuckles at the top, the
        // fingers parted by short creases running down the tilt and darker where they curl round the hip, and
        // the side against the body in shadow.
        // (the same size as the hanging hand)
        const rx = 2.0, ry = 2.4, inward = -outward;
        // the hand sits on the jerkin and casts a soft neutral shadow onto it, below and toward the body
        for (let t = 0; t < Math.PI * 2; t += 0.12) {
          const ex = Math.cos(t), ey = Math.sin(t);
          if (ey < 0.1 && ex * inward < 0.3) continue;
          const q = H(ex * (rx + 0.5) + inward * 0.25, ey * (ry + 0.5) + 0.3);
          P.darken(q[0], q[1], 0.78, 0);
        }
        ellipseAt(0, 0, rx, ry).fill(HS, { dim, round: 1 });
        // the thumb, a small pad on the inner side near the wrist
        ellipseAt(inward * 1.85, -0.85, 0.85, 1.1).fill(HS, { dim, round: 0.8 });
        for (let y = -1.5; y <= -0.2; y += 0.5) Dh(inward * 1.15, y, T(1));
        // a firm edge where the hand overlaps the jerkin: darkest along the bottom and the inner side
        for (let t = 0; t < Math.PI * 2; t += 0.1) {
          const ex = Math.cos(t), ey = Math.sin(t);
          if (ey < -0.55) continue;
          Dh(ex * (rx - 0.25), ey * (ry - 0.25), T(ey > 0.35 || ex * inward > 0.4 ? 0 : 1));
        }
        // the knuckles: a lit ridge across the back of the hand, brightest toward the outer side
        for (let x = -1.5; x <= 1.51; x += 0.5) Dh(x, -0.55 + 0.12 * x * x, T(x * outward > 0.4 ? 4 : 3));
        // four fingers curling down over the hip: each with a lit top just under the knuckle, parted by creases
        // that darken toward the tips where the fingers turn under
        [-1.3, -0.43, 0.43, 1.3].forEach((x) => { Dh(x, 0.25, T(x * outward > 0 ? 3 : 2)); Dh(x, 1.45, T(1)); });
        [-0.86, 0, 0.86].forEach((x) => { for (let y = 0.25; y <= 1.9; y += 0.5) Dh(x, y, T(y > 1.1 ? 0 : 1)); });
      // (a hanging hand is a little narrower than it is long, so at a distance it reads as a hand, not a square)
      } else {
        // its lower half is fuller than an ellipse (a superellipse), so the fingertips end in a soft round edge
        // rather than narrowing to a point
        const pts = [];
        for (let k = 0; k < 32; k++) {
          const t = k / 32 * Math.PI * 2, c = Math.cos(t), sn = Math.sin(t), e = sn > 0 ? 0.8 : 1, ry = sn > 0 ? 2.2 : 2.4;
          pts.push(H(Math.sign(c) * Math.pow(Math.abs(c), e) * 2.0, 0.1 + Math.sign(sn) * Math.pow(Math.abs(sn), e) * ry));
        }
        P.p(pts).fill(HS, { dim, round: 1.25 });
      }
      // front view: the cuff casts a soft shadow across the top of the hand, the side against the body
      // falls into shadow, the fingers curl under into a darker underside, and the outer knuckle catches the light
      if (V.fz > 0.3 && !akimbo) {
        const inward = -outward, rx = 2.0, ry = 2.5, half = (dy) => rx * Math.sqrt(Math.max(0, 1 - (dy * dy) / (ry * ry)));
        for (let dy = -ry + 0.25; dy <= ry - 0.25; dy += 0.5) {
          const w = half(dy);
          if (w < 0.6) continue;
          Dh(inward * (w - 0.25), dy, T(dy > 1 ? 0 : 1));
          if (dy > -1 && dy < 1.5) Dh(inward * (w - 0.75), dy, T(2));
        }
        for (let dx = -1.75; dx <= 1.75; dx += 0.5) { Dh(dx, -ry + 0.75, T(dx * inward > 0 ? 1 : 2)); Dh(dx, ry * Math.sqrt(Math.max(0, 1 - (dx * dx) / (rx * rx))) - 0.25, T(1)); }
        Dh(outward * 1.25, -0.75, T(4)); Dh(outward * 0.75, -1.25, T(3));
      }
      // a bell sleeve overhangs the hand: its shadow falls across the back of the hand, deepest at the wrist
      if (S.sleeveStyle === 'bell') for (let dx = -2.25; dx <= 2.25; dx += 0.5) { P.darken(hx + dx * hk, hy - 2.15 * hk, 0.7, 0.3).darken(hx + dx * hk, hy - 1.65 * hk, 0.8, 0.3).darken(hx + dx * hk, hy - 1.15 * hk, 0.92, 0.3); }
      // finger lines: short creases across the lower half of the hand, and the crease of the thumb, snapped to
      // the fine grid (in the hand's own frame) so each is one clean pixel wide. The middle crease runs longest
      // and the outer ones shorter, as the middle fingers are longer, so the lower edge reads as fingers rather
      // than a mitten; each fingertip catches a little light.
      const fp = (v) => (Math.floor(v * RES) + 0.5) / RES, ox = (fp(hx) - hx) / hk, oy = (fp(hy) - hy) / hk;
      if (!akimbo) [-1, 0, 1].forEach((o) => { if (o === 0) Dh(ox + o, oy, T(1)); Dh(ox + o, oy + 0.5, T(1)); Dh(ox + o, oy + 1, T(0)); Dh(ox + o, oy + 1.5, T(0)); });
      if (!akimbo) [-1.5, -0.5, 0.5, 1.5].forEach((o) => Dh(ox + o, oy + 1.5 - (Math.abs(o) > 1 ? 0.5 : 0), T(o * outward > 0 ? 3 : 2)));
      if (Math.abs(V.fx) < 0.8 && !akimbo) { Dh(ox - outward * 1.5, oy - 0.5, T(1)); Dh(ox - outward * 1, oy, T(0)); }
      // the cuff's shadow across the top of the hand, from every side (the front view's own loop draws it there)
      if (V.fz <= 0.3 && !akimbo) for (let dx = -1.5; dx <= 1.5; dx += 0.5) Dh(dx, -1.65, T(1));
      // in three-quarter and side views the thumb shows on the forward side of the hand: a small rounded pad
      // standing off the hand's edge, parted from it by a crease and lit along its top
      if (Math.abs(V.fx) > 0.3 && V.d(hand[0], hand[1] + 2) > V.d(hand[0], hand[1]) - 0.2) {
        const fwd = Math.sign(V.fx);
        ellipseAt(fwd * 1.85, -0.3, 0.95, 1.35).fill(HS, { dim, round: 1 });
        for (let dy = -0.75; dy <= 0.75; dy += 0.5) Dh(fwd * 1.1, -0.3 + dy, T(1));
        Dh(fwd * 1.85, -1.05, T(3)); Dh(fwd * 2.1, 0.7, T(1));
      }
    });
  });

  // neck: drawn just behind the body so collars and necklines cover its base, and the head covers its top
  add(-0.5, () => {
    if (Math.abs(V.fx) > 0.3 || V.fz < -0.3 || S.female) {
      // (from the side it reaches up under the jawline, so no gap opens between the jaw and the neck)
      // Seen from the side the neck is deeper than it is wide from the front (nearly half the depth of the
      // head), and it leans forward from the shoulders up to the head, so the throat runs up to meet the
      // underside of the jaw instead of leaving a thin stalk with a gap in front of it
      const n0 = V.pt(0, 0.5, 45 + 3 * V.fx * V.fx), n1 = V.pt(0, -0.4, 40.2);
      P.t(n0[0], n0[1], n1[0], n1[1], V.w(S.female ? 3.4 : 4, S.female ? 5.4 : 6), V.w(S.female ? 3.8 : 4.4, S.female ? 5.8 : 6.4)).fill(SKIN, { round: 0.7, lift: -0.14 });
      // the jaw casts a soft shadow onto the top of the neck, separating chin from throat
      if (V.fz > 0.3) P.e(V.x(0, 0.6), V.y(43.4), V.w(S.female ? 2 : 2.4, 2), 1.1).fill(SKIN, { tone: 1, flat: true });
      // seen from the side the shadow runs along the jawline (see HEAD_SHAPE), only on the neck itself,
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
    // the sculpted head (see headModel): her lower face is rounder and shorter, with a softer chin
    const model = headModel(fem), depth = rasterHead(P, V, model);
    const shade = faceShade(V, fem, depth);
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
      // (on the sculpted surface, in the hollow of the socket under the brow)
      const [a, b] = model.front(s * 2.7, -1), d = V.d(a, b);
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
    // Nose and mouth accents, placed on the sculpted surface: a point (a, h) on the face lands where the
    // model puts it in this view, and is skipped where something nearer covers it (the nose, or the near
    // cheek as the head turns), so the far corner of the mouth tucks behind the cheek instead of sitting on
    // the outline as a pout.
    const onFace = (a, h) => {
      const [a2, f2] = model.front(a, h), x = V.x(a2, f2), y = V.y(HEAD + h), d = V.d(a2, f2);
      const X = Math.floor(x * RES), Y = Math.floor(y * RES), hit = depth.get(Y * PW + X);
      return hit && hit.d <= d + 0.3 ? [(X + 0.5) / RES, (Y + 0.5) / RES] : null;
    };
    const put = (a, h, c) => { const q = onFace(a, h); if (q) D(q[0], q[1], c); return q; };
    // the side of the face toward the viewer (its sign in a), and how far the head is turned (0 front, 1 side)
    const near = V.rz >= 0 ? 1 : -1, turn = Math.abs(V.fx);
    if (V.fz > -0.2) {
      // Nose. The model gives its form (the bridge, the tip, the wings of the nostrils); these accents make it
      // read at this size: a lit tip, a shadow down the side of the bridge turned from the light, the curved
      // crease behind each nostril wing we can see, the nostril itself, and the shadow under the tip.
      const tip = fem ? -3.0 : -3.2, base = fem ? -3.45 : -3.75, ala = fem ? 0.78 : 0.92;
      // the side of the nose turned away from the light (+1 is the head's right)
      const dark = Math.sign(-(V.rx * FACE_LIGHT[0] + V.rz * FACE_LIGHT[2])) || 1;
      put(0, tip + 0.45, SKIN[3]); put(0, tip, SKIN[4]);
      if (turn > 0.25 && turn < 0.9) {
        // In three-quarter view the nose stands against the far cheek: the cheek just past the ridge, in the
        // nose's shadow, is a soft line from below the brow down to the wing. It sits one fine pixel beyond
        // the ridge on screen (toward the far side), on the cheek behind it.
        const farDir = Math.sign(-near * V.rx) || 1;
        for (let h = -1.2; h >= tip - 0.1; h -= 0.25) {
          const q = onFace(0, h);
          if (!q) continue;
          const x = q[0] + farDir / RES, k = Math.floor(q[1] * RES) * PW + Math.floor(x * RES);
          if (depth.has(k)) D(x, q[1], h > -1.7 ? SKIN[1] : SKIN[0]);
        }
      }
      [-1, 1].forEach((s) => {
        if (turn > 0.9 && s !== near) return;
        const lit = s !== dark;
        // the crease around the back of the wing: a short curve, open toward the tip
        if (turn > 0.25) [[ala + 0.1, base + 0.55], [ala + 0.2, base + 0.25], [ala + 0.15, base - 0.05]].forEach(([a, h]) => put(s * a, h, lit ? SKIN[1] : mix(lid, 0.35)));
        // the nostril, darkest on the shadow side
        put(s * (fem ? 0.45 : 0.55), base + 0.05, mix(lid, lit ? 0.4 : 0.6));
      });
      // the tip casts a soft shadow down onto the upper lip, offset away from the light
      for (let a = -0.5; a <= 0.5 + 1e-6; a += 0.25) put(a + dark * 0.25, base - 0.3, SKIN[1]);
      // mouth: the line between the lips, darkest in the middle, fading out at the corners; a tinted upper
      // lip; a lit lower lip; and the shadow under it, where the lip turns in toward the chin
      const mh = fem ? -5.05 : -5.3, hw = fem ? 1.15 : 1.3;
      const lipLine = fem ? [150, 76, 74] : [128, 80, 70], lipLo = fem ? [206, 124, 116] : [196, 132, 116], up = [184, 104, 98];
      const seen = new Set(), once = (a, h, c) => { const q = onFace(a, h); if (!q) return; const key = q[0] + ',' + q[1]; if (seen.has(key)) return; seen.add(key); D(q[0], q[1], c); };
      for (let a = -hw; a <= hw + 1e-6; a += 0.1) {
        const edge = Math.abs(a) > hw - 0.35;
        once(a, mh, edge ? mix(lipLine, 0.45) : lipLine);
      }
      for (let a = -0.85; a <= 0.85 + 1e-6; a += 0.1) {
        if (fem) once(a, mh + 0.5, mix(up, Math.abs(a) < 0.3 ? 0.5 : 0.8));
        once(a, mh - 0.5, mix(lipLo, fem ? 0.75 : 0.6));
      }
      if (fem) put(0, mh - 0.5, mix([236, 176, 160], 0.6));
      for (let a = -0.5; a <= 0.5 + 1e-6; a += 0.25) once(a, mh - (fem ? 1.05 : 0.95), SKIN[1]);
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
    const V = A.V, [hx, hy] = J.hip, [kx, ky] = J.knee, [ax, ay] = J.ankle, T = (i) => RANGER.legs[Math.max(0, i - (dim ? 1 : 0))];
    const ux = (kx - hx) / (ky - hy || 1), wAt = (y) => lerp(A.B.leg, A.B.shin + 0.4, Math.max(0, Math.min(1, (y - hy) / (ky - hy || 1)))) / 2;
    const xAt = (y) => hx + (y - hy) * ux, inner = Math.sign(V.x(0, 0) - hx) || 1, front = V.d(0, 1) > 0.15;
    // the shin, from the knee down toward the boot (which covers its lower part)
    const sx = (y) => kx + (ax - kx) * (y - ky) / (ay - ky || 1), sw = (y) => lerp(A.B.shin + 0.4, A.B.shin - 0.8, Math.max(0, Math.min(1, (y - ky) / (ay - ky || 1)))) / 2;
    for (let i = 0, y = V.y(18); y <= ay; i++, y += 0.5) {
      const thigh = y <= ky, x = thigh ? xAt(y) : sx(y), w = thigh ? wAt(y) : sw(y), odd = i % 2;
      // The inner edge turns away into the shadow between the legs: a broad soft band (two fine pixels of the
      // darkest tone, then a dithered half-tone), wide enough to set the legs apart at this size. The edge
      // turned from the light (screen right) falls into shadow too, so each leg reads as round.
      if (Math.abs(V.fx) < 0.5) {
        P.dotOn(x + inner * (w - 0.3), y, T(0)).dotOn(x + inner * (w - 0.8), y, T(0)).dotOn(x + inner * (w - 1.3), y, T(1));
        if (odd) P.dotOn(x + inner * (w - 1.8), y, T(1));
      }
      if (inner < 0 || Math.abs(V.fx) >= 0.5) { P.dotOn(x + w - 0.3, y, T(1)); if (odd) P.dotOn(x + w - 0.8, y, T(1)); }
      if (!thigh) continue;
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
  // both arms hang relaxed at rest, the elbows drawn in toward the waist
  akimbo: false, elbowTuck: 1.4,
  // a little asymmetry so the relaxed pose is not flat: the arms set against the feet (the left foot is
  // forward), the right hand eased out from the body and forward, the left closer in and back. Both are at the
  // same height: they differ on screen only by their depth (see the arms' tilt), as the feet do.
  restArms: [[0.9, 3.0, 0.3], [0.0, -0.8, 0.3]],
  chest(P, A) {
    const V = A.V, cx = V.x(0, 0), wc = V.w(A.B.chest + 0.4, A.B.depth + 0.5), wh = V.w(A.B.waist + 0.5, A.B.depth + 0.2);
    // jerkin over the shirt, open at the collar, laced up the front, falling just past the belt
    const hemH = (phi) => 18.6 - 1.3 * Math.pow(Math.max(0, Math.cos(phi)), 4) + 0.3 * Math.abs(Math.sin(phi * 2));
    // (the hem sits just over the body, so the jerkin follows the hips instead of flaring past them like a skirt)
    const wHem = A.lw(19.5, -0.5), hem = A.ring(wHem, A.ld(wHem) + 0.25, hemH);
    const topY = V.y(22.5), wTop = A.halfW(22.5, 0.15);
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
  },
  waist(P, A) {
    const V = A.V;
    // Belt pouch on the front of the left hip, set on the hip itself (in from its edge, so it never stands off
    // the outline as a disc), drawn only when the front of that hip faces the viewer, never from behind: a
    // rounded body, a darker flap with a stitched edge, and light along its top.
    const hw = A.lw(20.6), pa = -(hw - 1.9), pb = A.ld(hw) * 0.62;
    if (V.d(pa, pb + 1) > 0.5 && V.fz > -0.1) {
      const c = V.pt(pa, pb, 20.4), w = V.w(1.75, 0.7) * 2;
      P.t(c[0], c[1] - 1.4, c[0], c[1] + 1.1, w, w + 0.5).fill(JERKIN, { round: 0.9, pattern: grain });
      P.e(c[0], c[1] - 1.2, w / 2 + 0.15, 1.1).fill(JERKIN, { flat: true, tone: 1 });
      for (let t = -0.8; t <= 0.81; t += 0.4) P.dotOn(c[0] + t * w / 2, c[1] - 0.3 + 0.35 * (1 - t * t), JERKIN[3]);
      P.dotOn(c[0] - w / 2 + 0.6, c[1] - 1.9, JERKIN[3]).dotOn(c[0] - w / 2 + 1.1, c[1] - 2.1, JERKIN[3]);
    }
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
      // From the neck to the hem the cloth follows a rounded shoulder rather than a straight cone: it slopes
      // gently out from the neck and rounds down over the shoulder (outward fast at first, then mostly down),
      // so the shoulders come out sloped and soft instead of a flat top with square corners.
      const M = 6, at = (r, t) => {
        const e = 1 - Math.pow(1 - t, 1.8), R = lerp(4.4, B.sh + 2.2, e), D = lerp(4, B.depth + 2, e);
        return V.seam(R, D, r.phi, lerp(40.2, hemH(r.phi), Math.pow(t, 1.25)));
      };
      for (let k = 0; k < N; k++) {
        const a = ring[k], b = ring[k + 1];
        for (let m = 0; m < M; m++) {
          const a0 = at(a, m / M), a1 = at(a, (m + 1) / M), b0 = at(b, m / M), b1 = at(b, (m + 1) / M);
          P.p([[a0[0], a0[1]], [b0[0], b0[1]], [b1[0], b1[1]], [a1[0], a1[1]]]);
        }
      }
      P.e(V.x(0, 0), V.y(39.9), V.w(6.0, 5.0), 1.6);
      // the cloth lying over each shoulder: a rounded cap, so the mantle covers the top of the sleeve beneath
      // (which otherwise pokes out past the sloped shoulder as a step in the outline) and keeps a soft outline
      [-1, 1].forEach((sd) => { const c = V.seam(B.sh - 0.9, B.depth - 1.4, sd * Math.PI / 2, 37.5); P.e(c[0], c[1], 3.0, 2.7); });
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
      // light across the tops of the shoulders, set on the shoulder's curve and painted only on the mantle's own
      // pixels (placed by a straight cone it stood off the rounded shoulder as a pale fleck on the hair)
      const MSET = new Set(MANTLE.map((c) => (Math.floor(c[0]) << 16) | (Math.floor(c[1]) << 8) | Math.floor(c[2])));
      const onMantle = (k) => MSET.has((P.buf[k * 4] << 16) | (P.buf[k * 4 + 1] << 8) | P.buf[k * 4 + 2]);
      [-1, 1].forEach((sd) => { const c = at({ phi: sd * Math.PI / 2 }, 0.45); if (c[2] > 1.5) P.e(c[0] - sd * Math.abs(V.fz) * 0.4, c[1] + 0.3, 1.5, 0.6).fill(MANTLE, { flat: true, tone: 3, pattern: weave, on: onMantle }); });
      // The hem: a defined edge, so the mantle reads as a layer lying over the sleeves rather than melting into
      // them (they are close greens): a dark line along the hem itself, the stitching lit just above it where
      // the hem faces us, and a lit lip where it turns away over the arms. Below it the mantle casts its shadow,
      // stepping whatever lies under it (sleeve, jerkin, shirt) one tone down its own ramp, following the hem's
      // curve: two rows, the second softened. (A plain darken there turned the cloth blue and muddy.)
      const UNDER = [A.S.sleeve, JERKIN, SHIRT, A.S.belt].map((r) => r.map((c) => (Math.floor(c[0]) << 16) | (Math.floor(c[1]) << 8) | Math.floor(c[2])));
      const stepDown = (x, y) => {
        const X = Math.floor(x * RES), Y = Math.floor(y * RES);
        if (X < 0 || Y < 0 || X >= PW || Y >= PH) return;
        const k = Y * PW + X, c = (P.buf[k * 4] << 16) | (P.buf[k * 4 + 1] << 8) | P.buf[k * 4 + 2];
        for (let r = 0; r < UNDER.length; r++) { const i = UNDER[r].indexOf(c); if (i > 0) { const R = [A.S.sleeve, JERKIN, SHIRT, A.S.belt][r][i - 1]; P.dot(x, y, R); return; } if (i === 0) return; }
      };
      const done = new Set();
      for (let k = 0; k <= 160; k++) {
        const phi = k / 160 * Math.PI * 2, h = V.seam(B.sh + 2.2, B.depth + 2, phi, hemH(phi));
        if (h[2] < 0.3) continue;
        P.dotOn(h[0], h[1] - 0.2, MANTLE[0]);
        if (h[2] > 3) P.dotOn(h[0], h[1] - 0.8, MANTLE[3]);
        else P.dotOn(h[0], h[1] - 0.7, MANTLE[2]);
        [[0.4, 1], [0.9, k % 2]].forEach(([dy, on]) => {
          if (!on) return;
          const key = Math.floor(h[0] * RES) + ',' + Math.floor((h[1] + dy) * RES);
          if (done.has(key)) return;
          done.add(key); stepDown(h[0], h[1] + dy);
        });
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
  const { V } = A;
  const legs = [1, -1].map((s) => ({ s, fb: A.stride(s) }));
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
  const hemH = (phi) => 37.2 - 2.7 * Math.pow(Math.max(0, Math.cos(phi)), 2) - 1.4 * Math.pow(Math.max(0, -Math.cos(phi)), 2) - 1.0 * Math.pow(Math.sin(phi), 4) + 0.35 * Math.cos(5 * phi + 0.5) * Math.max(0, Math.cos(phi));
  const top = (phi) => V.seam(4.3, 4.0, phi, 40.6), bot = (phi) => V.seam(B.sh - 0.5, B.depth + 0.35, phi, hemH(phi));
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
  const fold = (frac, sag, phiMax, lift = 0, off = 0) => { const pts = []; for (let k = 0; k <= 26; k++) { const phi = off - phiMax + k / 26 * phiMax * 2, c = Math.cos(phi - off); const h = lerp(40.4, hemH(phi), frac) - sag * c * c + lift; const w = lerp(4.4, B.sh - 0.5, frac), d = lerp(4.1, B.depth + 0.35, frac) + 0.2; pts.push(V.seam(w, d, phi, h)); } return pts; };
  const draw = (pts, minD, w0, w1 = w0) => { for (let i = 0; i < pts.length - 1; i++) if (pts[i][2] > minD && pts[i + 1][2] > minD) { const t = Math.abs(i / (pts.length - 2) - 0.5) * 2; P.l(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], lerp(w1, w0, t)); } };
  // Three deep swags, each a true U that hangs well below its ends, a little off-centre so they don't
  // stack like rings: [frac down the cowl, sag at the middle, half-reach round the body, centre offset].
  // A soft shadow band under a lit roll for each, thin at its ends.
  const U = [[0.12, 1.6, 0.7, 0.04], [0.36, 2.3, 0.85, -0.05], [0.6, 2.8, 1.0, 0.03]];
  const foldAt = (fr, sag, pm, off, lift = 0) => fold(fr, sag, pm, lift, off);
  U.forEach(([fr, sag, pm, off]) => draw(foldAt(fr, sag, pm, off), 1.4, 0.45, fr > 0.5 ? 1.4 : 1.0));
  P.fill(W, { flat: true, tone: 2, on: true, pattern: linen });
  // the deepest line of the longer folds, at the middle only
  U.forEach(([fr, sag, pm, off]) => { draw(foldAt(fr, sag, pm * 0.6, off, -0.15), 2.2, 0.45, 0.6); });
  P.fill(W, { flat: true, tone: 1, on: true });
  // the lit roll just above each shadow
  U.forEach(([fr, sag, pm, off]) => draw(foldAt(fr - 0.09, sag, pm * 0.8, off), 2, 0.45, 0.6));
  P.fill(W, { flat: true, tone: 4, on: true });
  // The collar: a standing roll of cloth round the neck, low at the throat and rising at the sides and
  // back, flaring a little at its top and finished with a thick lit roll. It gives the neckpiece height
  // and body instead of a flat edge, and its inside shows as a shadowed band where it stands away.
  const cTop = (phi) => 40.5 + 0.7 + 1.5 * Math.pow((1 - Math.cos(phi)) / 2, 0.8);
  const cR = (h) => [lerp(4.55, 5.05, (h - 40) / 3), lerp(4.3, 4.8, (h - 40) / 3)];
  const ringAt = (phi, h) => { const [ra, rb] = cR(h); return V.seam(ra, rb, phi, h); };
  const CN = 36, drawAll = V.fz < 0.3;
  // the inside of the far side, visible from the front above the throat: dark and soft
  if (!drawAll) {
    for (let k = 0; k < CN; k++) {
      const p0 = Math.PI * (0.55 + k / CN * 0.9), p1 = Math.PI * (0.55 + (k + 1) / CN * 0.9), a0 = ringAt(p0, 40.6), a1 = ringAt(p1, 40.6), b0 = ringAt(p0, cTop(p0) - 0.3), b1 = ringAt(p1, cTop(p1) - 0.3);
      P.p([[a0[0], a0[1]], [a1[0], a1[1]], [b1[0], b1[1]], [b0[0], b0[1]]]);
    }
    P.fill(W, { flat: true, tone: 1 });
  }
  for (let k = 0; k < 48; k++) {
    const p0 = k / 48 * Math.PI * 2, p1 = (k + 1) / 48 * Math.PI * 2, m = ringAt((p0 + p1) / 2, 41);
    if (!drawAll && m[2] < -0.5) continue;
    const a0 = ringAt(p0, 40.0), a1 = ringAt(p1, 40.0), b0 = ringAt(p0, cTop(p0)), b1 = ringAt(p1, cTop(p1));
    P.p([[a0[0], a0[1]], [a1[0], a1[1]], [b1[0], b1[1]], [b0[0], b0[1]]]);
  }
  P.fill(W, { pattern: linen, round: 0.9, dither: 0.4, lift: 0.03 });
  // a vertical fold or two in the standing cloth, and the thick roll along the top edge, lit above and
  // shadowed beneath so it reads as rolled cloth
  [-2.4, -1.5, -0.7, 0.7, 1.5, 2.4].forEach((phi) => { const a = ringAt(phi, 40.4), b = ringAt(phi + 0.08, cTop(phi) - 0.5); if (a[2] > 1.2 && b[2] > 1.2) P.l(a[0], a[1], b[0], b[1], 0.45); });
  P.fill(W, { flat: true, tone: 2, on: true });
  for (let k = 0; k <= 96; k++) {
    const phi = k / 96 * Math.PI * 2, q = ringAt(phi, cTop(phi)), q2 = ringAt(phi, cTop(phi) - 0.55), q3 = ringAt(phi, cTop(phi) - 1.0);
    if (q[2] < (drawAll ? -4 : 0.8)) continue;
    P.dotOn(q[0], q[1] + 0.25, W[4]).dotOn(q2[0], q2[1], W[3]);
    P.darken(q3[0], q3[1], 0.9, 0).darken(q3[0], q3[1] + 0.5, 0.94, 0);
  }
  // the seam where the collar meets the cowl: a soft shadow line, and the cowl's shadow cast on it
  for (let k = 0; k <= 80; k++) { const phi = k / 80 * Math.PI * 2, q = ringAt(phi, 40.0); if (q[2] > 1) P.darken(q[0], q[1] + 0.2, 0.82, 0).darken(q[0], q[1] + 0.7, 0.92, 0); }
  // the edge: a narrow black band and white running stitch along the hem where it hangs across the
  // breast and back only. Over the shoulders there is no line at all, just a shadow that fades out, so
  // nothing reads as a pad or a plate on top of the arm.
  for (let k = 0; k <= 200; k++) {
    // g: how squarely this part of the hem faces us, front on the breast or back on the shoulder blades;
    // it is zero in profile, where the hem is seen edge-on and any line would sit on the silhouette
    const phi = k / 200 * Math.PI * 2, q = bot(phi), g = Math.max(0, Math.cos(phi) * (V.fz > 0 ? 1 : -1) * Math.abs(V.fz)), deep = Math.max(0, Math.cos(phi)) ** 2;
    if (q[2] < 0.9) continue;
    if (g > 0.6) P.dotOn(q[0], q[1] - 0.25, K[g > 0.85 ? 3 : 4]);
    if (g > 0.5 && k % 4 === 0) P.dotOn(q[0], q[1] - 1.0, W[1]);
    const f = Math.pow(Math.min(1, g * 1.2), 1.6) * 0.6;
    P.darken(q[0], q[1] + 0.3, 1 - (0.2 + 0.06 * deep) * f, 0).darken(q[0], q[1] + 0.8, 1 - 0.12 * f, 0).darken(q[0], q[1] + 1.3, 1 - 0.05 * f, 0);
  }
  // The hood, thrown back and resting on her shoulder blades: a full bag of cloth gathered up behind the
  // collar. It is built on the same ellipse as the body and wraps round the back of it, so it turns with
  // her: in a three-quarter view the near half is broad and the far half curves away round the shoulder
  // instead of ending in a flat edge. Its opening is a thick rolled edge, black lining showing inside it;
  // below, the cloth bunches into a rounded mass with folds fanning to a soft point.
  if (V.fz < 0.3) {
    const hTop = 39.6, hBot = 25.4, PI = Math.PI;
    const tOf = (h) => Math.max(0, Math.min(1, (hTop - h) / (hTop - hBot)));
    // how far round the back the hood reaches at each height, and how far it stands off the body
    const reach = (h) => { const t = tOf(h); return 1.12 * Math.pow(Math.max(0, 1 - Math.pow(t, 2.1)), 0.8) * (1 + 0.1 * Math.sin(Math.min(1, t * 1.6) * PI * 0.6)); };
    const bulge = (h) => 0.9 * Math.sin(Math.min(1, tOf(h) * 1.3) * PI);
    const rad = (h) => [B.sh + 0.2 + 0.6 * bulge(h), B.depth + 1.0 + bulge(h)];
    const on = (u, h) => { const [ra, rb] = rad(h); return V.seam(ra, rb, PI + u * reach(h), h); };
    // keep the visible part: on the side turned away, the edge follows the body's edge instead of
    // wrapping behind it
    const edge = (sd, h) => { let u = sd; let q = on(u, h); while (q[2] < 0.3 && Math.abs(u) > 0.04) { u -= sd * 0.04; q = on(u, h); } return q; };
    const rows = []; for (let h = hTop - 0.4; h >= hBot - 0.01; h -= 0.7) rows.push(h);
    const L = rows.map((h) => edge(-1, h)), R = rows.map((h) => edge(1, h)).reverse();
    P.p(L.concat(R).map((q) => [q[0], q[1]])).fill(W, { pattern: linen, round: 1.2, dither: 0.4, lift: -0.02, contour: true });
    // its shadow on the robe and the sleeve beneath its lower edge
    L.concat(R).forEach((q, n, all) => { if (n > 2 && n < all.length - 3 && q[1] > V.y(hTop - 3)) { P.darken(q[0], q[1] + 0.4, 0.8, 0).darken(q[0], q[1] + 0.9, 0.88, 0).darken(q[0], q[1] + 1.4, 0.94, 0); } });
    // the folds: a few broad, uneven creases from under the rolled edge, spreading as they fall to the
    // point; a wide soft shade tapering at both ends, a darker line within it, and a lit crest beside it
    const crease = (u0, u1, h0, h1, bow, w0, w1) => {
      let prev = null;
      for (let k = 0; k <= 8; k++) {
        const t = k / 8, h = lerp(h0, h1, t), u = lerp(u0, u1, t) + bow * Math.sin(t * PI), q = on(u, h);
        if (prev && q[2] > 0.6 && prev[2] > 0.6) P.l(prev[0], prev[1], q[0], q[1], lerp(w0, w1, Math.sin(Math.min(1, (k - 0.5) / 8) * PI) * 0.9 + 0.1));
        prev = q;
      }
    };
    const FOLDS = [[-0.8, -0.74, 36.4, 29.4, -0.06, 2.0], [-0.42, -0.3, 36.0, 26.8, 0.05, 2.4], [0.02, 0.06, 36.4, 25.8, -0.04, 1.6], [0.4, 0.34, 36.0, 27.2, 0.06, 2.4], [0.78, 0.66, 36.4, 30.6, -0.05, 2.0]];
    FOLDS.forEach(([u0, u1, h0, h1, bow, w]) => crease(u0, u1, h0, h1, bow, 0.5, w));
    P.fill(W, { flat: true, tone: 2, on: true, pattern: linen });
    FOLDS.forEach(([u0, u1, h0, h1, bow, w]) => crease(u0 - 0.01, u1 - 0.01, h0 - 0.8, h1 + 1.6, bow, 0.4, w * 0.45));
    P.fill(W, { flat: true, tone: 1, on: true });
    FOLDS.forEach(([u0, u1, h0, h1, bow]) => crease(u0 + 0.14, u1 + 0.12, h0 - 0.6, h1 + 2.2, bow, 0.4, 0.8));
    P.fill(W, { flat: true, tone: 4, on: true });
    // the opening: seen from behind, the black lining is a crescent hugging the collar along its top and
    // sagging in the middle to a thick rolled edge below it, tapering to nothing over each shoulder
    const [ra0, rb0] = [B.sh + 0.2, B.depth + 1.0], span = 1.0;
    const up = (u) => hTop + 0.5 - 0.9 * u * u, lo = (u) => up(u) - 3.0 * Math.pow(Math.max(0, 1 - u * u), 0.8);
    const at = (u, h) => V.seam(ra0, rb0, PI + u * span, h);
    const upper = [], lower = [];
    for (let k = 0; k <= 16; k++) { const u = -1 + k / 8; upper.push(at(u, up(u))); lower.push(at(u, lo(u))); }
    const vis = (q) => Math.abs(V.fz) > 0.3 && q[2] > 1.6 * Math.abs(V.fz) + 0.3;
    P.p(upper.filter(vis).concat(lower.filter(vis).reverse()).map((q) => [q[0], q[1]])).fill(K, { round: 0.5, on: true });
    for (let k = 2; k <= 14; k++) { const u = -1 + k / 8, q = at(u, up(u) - 0.6); if (vis(q)) P.dotOn(q[0], q[1], K[2]); }
    // the rolled edge: a thick pale band along the lower curve, lit on top, shaded at its underside
    const roll = (d) => lower.map((q, k) => { const u = -1 + k / 8; return at(u, lo(u) - d); });
    const r0 = roll(0.0), r1 = roll(0.9), r2 = roll(1.6);
    for (let k = 0; k < r0.length - 1; k++) if (vis(r0[k]) && vis(r0[k + 1])) P.p([[r0[k][0], r0[k][1]], [r0[k + 1][0], r0[k + 1][1]], [r1[k + 1][0], r1[k + 1][1]], [r1[k][0], r1[k][1]]]);
    P.fill(W, { flat: true, tone: 3, on: true });
    for (let k = 0; k < r0.length - 1; k++) if (vis(r0[k]) && vis(r0[k + 1])) P.l(r0[k][0], r0[k][1] - 0.1, r0[k + 1][0], r0[k + 1][1] - 0.1, 0.5);
    P.fill(W, { flat: true, tone: 4, on: true });
    for (let k = 0; k < r1.length - 1; k++) if (vis(r1[k]) && vis(r1[k + 1])) { P.l(r1[k][0], r1[k][1], r1[k + 1][0], r1[k + 1][1], 0.5); P.l(r2[k][0], r2[k][1] + 0.2, r2[k + 1][0], r2[k + 1][1] + 0.2, 0.5); }
    P.fill(W, { flat: true, tone: 1, on: true });
    const pt = on(0, hBot + 0.4);
    P.dotOn(pt[0], pt[1], W[1]).dotOn(pt[0], pt[1] - 0.5, W[2]);
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
