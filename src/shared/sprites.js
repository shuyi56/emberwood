// Procedural pixel-art sprites for the Emberwood party, shared by the game and the sprite bench.
const lerp = (a, b, t) => a + (b - a) * t;
// ---------- Sprites ----------
// The party are pixel-art sprites painted in code. Each figure is posed in 3D (hips, knees, shoulders,
// hands, gear) and projected for eight facings, so every view is consistent and gear stays in the right
// hand from every side. Columns: two idle (breathing) frames, then an eight-step walk. 64x80 pixels a frame.
const FW = 64, FH = 80, COLS = 10, ROWS = 8, WALK = 8;

// Palettes: five tones per material. Shadows drift toward cool violet, lights toward warm straw,
// and both lose saturation, so each ramp covers a range of hues instead of one colour scaled.
function rgb2hsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2;
  if (mx === mn) return [0, 0, l];
  const d = mx - mn, s = l > 0.5 ? d / (2 - mx - mn) : d / (mx + mn);
  let h = mx === r ? (g - b) / d + (g < b ? 6 : 0) : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [h / 6, s, l];
}
function hsl2rgb(h, s, l) {
  const f = (n) => { const k = (n + h * 12) % 12, a = s * Math.min(l, 1 - l); return 255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))); };
  return [f(0), f(8), f(4)];
}
function hueToward(h, target, t) { let d = target - h; d -= Math.round(d); return (h + d * t + 1) % 1; }
function ramp(hex, spread) {
  const [h, s0, l] = rgb2hsl((hex >> 16) & 255, (hex >> 8) & 255, hex & 255);
  const s = Math.min(s0, 0.46);
  spread = (spread || 0.1) * 0.86;
  return [-2, -1, 0, 1, 2].map((t) => {
    const hh = t < 0 ? hueToward(h, 0.62, -t * 0.05) : hueToward(h, 0.13, t * 0.045);
    const ss = Math.max(0, Math.min(1, s * (1 - Math.abs(t) * 0.08)));
    const ll = Math.max(0.06, Math.min(0.9, l + t * spread * (t < 0 ? 1.0 : 0.8)));
    return hsl2rgb(hh, ss, ll);
  });
}
const BAYER = [[0, 0.5], [0.75, 0.25]];

// The painter works at RES times the design grid: shapes are still described in 64x80 design units,
// but they are rasterised, shaded, dithered and outlined on the finer grid, and dot() can place
// single fine pixels for small details.
const RES = 2, PW = FW * RES, PH = FH * RES;
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
            let b = 0.6 - (u - 0.35) * 0.56 * round - (v - 0.45) * 0.28 * round + (o.lift || 0);
            const edge = i === X || i === X1 || Y === top || Y === bot;
            if (o.contour && edge) b -= 0.38;
            else if (!o.flat && (i === X1 || Y === bot)) b -= 0.17;
            if (o.dim) b -= 0.16;
            if (o.pattern) b += o.pattern(i, Y);
            let idx = o.tone != null ? o.tone : Math.round(b * 4 + BAYER[Y & 1][i & 1] * 0.45 - 0.22);
            if (o.tone != null && !o.flat && (i === X1 || Y === bot)) idx--;
            set(k, rmp[Math.max(0, Math.min(4, idx))]);
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
    darken(x, y, k) {
      const X = Math.floor(x * RES), Y = Math.floor(y * RES), q = (Y * PW + X) * 4;
      if (X >= 0 && Y >= 0 && X < PW && Y < PH && buf[q + 3]) { buf[q] *= k; buf[q + 1] *= k * 1.01; buf[q + 2] *= k * 1.05; }
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

const SKIN = ramp(0xcf9f84, 0.1), EYE = [46, 40, 38], WHITE = [226, 222, 210];
const CX = 32, GY = 77, HEAD = 52;
// Local figure space: a = toward the figure's right, b = forward, h = up from the ground (sprite pixels).
// theta 0 faces the viewer, PI/2 faces screen right, PI faces away.
function View(theta, bob, sway) {
  sway = sway || 0;
  const fx = Math.sin(theta), fz = Math.cos(theta), rx = -Math.cos(theta), rz = Math.sin(theta);
  const V = {
    fx, fz, rx, rz, bob,
    x: (a, b) => CX + (a + sway) * rx + b * fx,
    y: (h) => GY - h + bob,
    d: (a, b) => a * rz + b * fz,
    pt: (a, b, h) => [V.x(a, b), V.y(h)],
    w: (A, B) => Math.sqrt(A * A * rx * rx + B * B * fx * fx),
    seam: (A, B, phi, h) => [V.x(A * Math.sin(phi), B * Math.cos(phi)), V.y(h), V.d(A * Math.sin(phi), B * Math.cos(phi))]
  };
  return V;
}
// mail: rows of interlocking rings, each with a dark centre and a lit upper edge
const mail = (x, y) => { const r = y % 3, sx = (x + (Math.floor(y / 3) % 2) * 2) % 4; return r === 1 && sx === 1 ? -0.34 : r === 0 && (sx === 1 || sx === 2) ? 0.16 : r === 2 && sx === 0 ? -0.1 : 0; };
const grain = (x, y) => { const h = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453; return ((h - Math.floor(h)) - 0.5) * 0.14; };
// hair: strands running down, with a few lighter ones catching the light
// softer strands for long hair: gentle streaks, no bright glints
const softStrands = (x, y) => { const k = (x * 5 + Math.floor(y / 4) * 2) % 11; return k === 0 ? -0.13 : k === 6 ? 0.06 : grain(x, y) * 0.4; };
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
  const A = { V, swing, f, walking, B };
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
      const heel = V.pt(hipA * 1.08, fb - 1.6, 2 + lift - Math.min(0, toeUp) * 1.5), toe = V.pt(hipA * 1.08, fb + 3.6, 1.8 + lift + Math.max(0, toeUp) * 1.4);
      const top = V.pt(hipA * 1.06, fb * 0.75, (S.robe ? 5 : 9.5) + lift);
      P.t(top[0], top[1], ankle[0], ankle[1] + 1, 5.4, 5).t(heel[0], heel[1], toe[0], toe[1], 4.6, 3.6).fill(S.boots, { dim });
      if (S.footArmor) S.footArmor(P, { heel, toe }, dim, A);
      if (S.cuff && !S.robe) P.t(top[0], top[1] + 0.5, top[0], top[1] + 1.5, 6.2, 6.2).fill(S.cuff, { dim, round: 0.5 });
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
    if (S.robe) {
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
    // belt, sitting at the waist
    if (S.robe) {
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
        const wu = B.arm + 1.8, open = 5.2, wy = pH[1] + 1;
        P.t(pS[0], pS[1], pE[0], pE[1], wu + 0.8, wu + 0.6)
          .p([[pE[0] - wu / 2 - 0.3, pE[1]], [pE[0] + wu / 2 + 0.3, pE[1]], [pH[0] + open, wy], [pH[0] + outward * 3, wy + 5.8], [pH[0] - open, wy + 1.6]]);
        P.fill(S.sleeve, { round: 0.75, dim: depth < -4, pattern: S.sleevePattern });
        // folds running down the sleeve, and the dark inside of the opening
        P.l(pE[0] - outward * 0.6, pE[1] + 1, pH[0] - outward * 1.6, wy + 0.8, 0.7).l(pE[0] + outward * 0.8, pE[1] + 1.5, pH[0] + outward * 1.8, wy + 3, 0.7).fill(S.sleeve, { flat: true, tone: 1 });
        P.l(pE[0] + outward * 0.1, pE[1] + 1.2, pH[0] + outward * 0.2, wy + 1.8, 0.5).fill(S.sleeve, { flat: true, tone: 3 });
        P.e(pH[0], wy + 0.5, open - 1.2, 1.1).fill(S.sleeve, { flat: true, tone: 0 });
        // trim along the hanging edge
        P.l(pH[0] - open, wy + 1.6, pH[0] + outward * 3, wy + 5.8, 0.9).l(pH[0] + outward * 3, wy + 5.8, pH[0] + open, wy, 0.9).fill(S.cuffs || S.trim, { round: 0.4 });
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
      P.e(pH[0], V.y(hand[2]), 2.3, 2.4).fill(S.hands || SKIN, { dim });
    });
  });

  // neck: drawn just behind the body so collars and necklines cover its base, and the head covers its top
  add(-0.5, () => {
    if (Math.abs(V.fx) > 0.3 || V.fz < -0.3 || S.female) {
      const n0 = V.pt(0, -0.6, 45), n1 = V.pt(0, -0.4, 40.2);
      P.t(n0[0], n0[1], n1[0], n1[1], V.w(S.female ? 3.4 : 4, S.female ? 3.2 : 3.8), V.w(S.female ? 3.8 : 4.4, S.female ? 3.6 : 4)).fill(SKIN, { round: 0.7, lift: -0.14 });
      // the jaw casts a soft shadow onto the top of the neck, separating chin from throat
      if (V.fz > 0.3) P.e(V.x(0, 0.6), V.y(43.4), V.w(S.female ? 2 : 2.4, 2), 1.1).fill(SKIN, { tone: 1, flat: true });
      if (Math.abs(V.fx) > 0.5) P.l(V.x(0, 1), V.y(HEAD - 7.2), V.x(0, 4.4), V.y(HEAD - 8.4), 0.8).fill(SKIN, { tone: 1, flat: true });
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
    if (fem) P.e(V.x(0, 0.3), V.y(HEAD + 0.9), 6.2, 7.2).e(V.x(0, 1.1), V.y(HEAD - 3.1), 4.9, 5.2).fill(SKIN);
    else P.e(V.x(0, 0.3), V.y(HEAD + 0.6), 6.6, 7.6).e(V.x(0, 1.4), V.y(HEAD - 3.3), 5.3, 5.1).fill(SKIN);
    const brow = S.brow || [86, 68, 56], iris = S.iris || [86, 72, 60];
    const irisDk = iris.map((c) => c * 0.62), irisLt = iris.map((c) => Math.min(255, c * 1.3 + 14));
    const lid = [86, 58, 50], lash = [52, 38, 34], white = [228, 222, 208], blush = [196, 128, 110];
    const mix = (c, k) => SKIN[2].map((v, i) => v + (c[i] - v) * k);
    const D = (x, y, c) => P.dotOn(x, y, c);
    [1, -1].forEach((s) => {
      const a = s * 2.7, b = 6.4, d = V.d(a, b);
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
      // lit cheekbone and a soft flush
      if (!narrow) {
        D(cx + o * 1, cy + 2, SKIN[3]); D(cx + o * 0.5, cy + 2, SKIN[3]);
        D(cx + o * 1, cy + 3, mix(blush, 0.45)); D(cx + o * 0.5, cy + 3, mix(blush, 0.35)); D(cx + o * 1.5, cy + 3, mix(blush, 0.3)); D(cx + o * 1, cy + 3.5, mix(blush, 0.25));
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
    // mouth: a short upper lip line with soft corners, a lit lower lip and a shadow under it
    if (V.fz > 0.3) {
      const front = V.fz > 0.9;
      const mx = Math.round(V.x(0, front ? 7.4 : 5.8) * 2) / 2, my = Math.round(V.y(HEAD - 5.2) * 2) / 2, half = front ? 1 : 0.5;
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
  // slender, long-limbed build: narrow waist, slimmer arms and legs, legs set closer
  body: { curve: [[15, 8], [20.5, 9.1], [27.6, 3.9], [34, 8.4]], belt: 26, neckDrop: 0.8, sh: 7.8, chest: 7.2, bust: 8.6, waist: 4.1, hip: 9.2, depth: 5, arm: 4, fore: 3.4, leg: 5.8, shin: 4.2, gap: 3.6, slope: 3.4, armOut: 0.2 },
  // cloth wraps wound around the shins, crossing in front
  legArmor(P, J, dim, A) {
    for (let k = 0; k < 3; k++) {
      const t0 = 0.25 + k * 0.22, t1 = t0 + 0.16;
      const ax = J.knee[0] + (J.ankle[0] - J.knee[0]) * t0, ay = J.knee[1] + (J.ankle[1] - J.knee[1]) * t0;
      const bx = J.knee[0] + (J.ankle[0] - J.knee[0]) * t1, by = J.knee[1] + (J.ankle[1] - J.knee[1]) * t1;
      P.l(ax - 2, ay, bx + 2, by, 0.9).l(ax + 2, ay, bx - 2, by, 0.9);
    }
    P.fill(ramp(0x8c8270, 0.09), { dim, round: 0.4 });
  },
  torso: SHIRT, sleeve: ramp(0x63704a, 0.1), legs: ramp(0x5b5547, 0.1), knee: ramp(0x4e493d, 0.09),
  boots: ramp(0x4a3f34, 0.1), cuff: ramp(0x655546, 0.1), belt: ramp(0x3d342b, 0.09), buckle: ramp(0x9a978f, 0.09), trim: GOLD,
  female: true, hands: SKIN, brow: [96, 70, 56], iris: [88, 108, 70], softArms: true, sleeveStyle: 'puff', shirtCuff: SHIRT,
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
    if (V.d(-7.4, 2.5) > -3) P.e(V.x(-7.5, 2.8), V.y(20.6), 2.4, 2.6).fill(JERKIN, { dim: V.d(-7.4, 2.5) < 0 });
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
    const L = A.handL;
    // Hair: shoulder-length and loose, with a side-swept fringe and a side parting.
    // The head is covered by one shell with the face cut out along a hairline that turns with the head
    // (each half of the face cut separately, so it holds together in profile); the length falls as a
    // curtain around the back of the head and over the shoulders.
    // the face opening is narrow from the front (hair frames the cheeks) and opens up as she turns to profile
    const FACE = 0.98 + 0.32 * V.fx * V.fx;
    const hairline = (phi) => {
      const t = Math.min(1, Math.abs(phi) / FACE), s = (phi + FACE) / (2 * FACE);
      // the fringe sweeps from high on the parting side down across the forehead to the far temple
      const tipN = phi * 7.5 + 0.3, tip = Math.abs(tipN - Math.round(tipN)) * 2;   // 0 at a strand tip, 1 between
      const len = 0.5 + 0.5 * Math.sin(Math.round(tipN) * 2.3);                       // tips of different lengths
      return HEAD + 3.3 - 1.5 * s - 0.6 * t * t + (tip * 0.9 - len * 0.7) * (1 - t * 0.7);
    };
    const ends = (phi) => 35.6 + Math.sin(phi * 5 + 0.4) * 0.9;
    const curtainPhis = [];
    for (let k = 0; k <= 22; k++) curtainPhis.push(1.45 + k / 22 * (Math.PI * 2 - 2.9));
    // Bob: the hair falls from the crown to just below the jaw all the way round, swelling over the
    // cheeks and curling under at the ends. It is built as rings around the head and drawn in two
    // passes: the side facing us over the head, the far side behind everything, so it never covers the face.
    // ends: clumps of different lengths, each coming to a soft point
    const clumpN = 13;
    const endH = (phi) => {
      const u = phi / (Math.PI * 2) * clumpN, f = Math.abs(u - Math.round(u)) * 2;
      const len = Math.sin(Math.round(u) * 3.1) * 0.9 + Math.sin(Math.round(u) * 1.3) * 0.5;
      return 37.4 - len * 0.6 - (1 - f) * 0.45 - (1 - Math.cos(phi)) * 0.9 + 4.2 * Math.pow(Math.max(0, Math.cos(phi) + 0.25), 1.4);
    };
    // the outline swells and dips from clump to clump instead of being a smooth shell
    const bulge = (phi) => Math.sin(phi * clumpN) * 0.28 + Math.sin(phi * 5 + 1) * 0.22;
    const bob = (() => {
      const N = 78, rings = [
        // from the crown, out over the back of the skull, in at the nape, then down over the shoulders
        { r: () => 6.95, d: () => 7.3, h: () => HEAD + 1.2 },
        { r: (p) => 7.3 + bulge(p) * 0.6 + 0.6 * Math.max(0, Math.cos(p) + 0.3), d: (p) => 8 + bulge(p) * 0.6, h: (p) => HEAD - 2.2 + Math.sin(p * 3) * 0.3 },
        { r: (p) => 7.1 + bulge(p) * 0.5 + 0.5 * Math.max(0, Math.cos(p) + 0.3), d: (p) => 6 + bulge(p) * 0.4, h: (p) => HEAD - 7.6, k: 0.86 },
        { r: (p) => 7 + bulge(p) * 0.5, d: (p) => 5.8 + bulge(p) * 0.4, h: (p) => endH(p) + 0.9, k: 0.72 },
        { r: (p) => 6.1 + bulge(p) * 0.4, d: (p) => 4.8, h: (p) => endH(p), k: 0.66 }
      ];
      const quads = [];
      for (let k = 0; k < N; k++) {
        const p0 = FACE + k / N * (Math.PI * 2 - 2 * FACE), p1 = FACE + (k + 1) / N * (Math.PI * 2 - 2 * FACE);
        const clump = Math.round((p0 + p1) / 2 / (Math.PI * 2) * clumpN);
        for (let r = 0; r < rings.length - 1; r++) {
          const A0 = rings[r], A1 = rings[r + 1];
          // a ring with k < 1 pulls its angles toward the back of the head
          const g = (A, p) => (A.k ? Math.PI + (p - Math.PI) * A.k : p);
          const at = (A, p) => V.seam(A.r(p), A.d(p), g(A, p), A.h(p));
          const q = [at(A0, p0), at(A0, p1), at(A1, p1), at(A1, p0)];
          const mid = V.seam(7.6, 7.8, (p0 + p1) / 2, HEAD);
          quads.push({ q, facing: mid[2] > -0.4, band: r, clump });
        }
      }
      return quads;
    })();
    const drawBob = (front) => {
      const mine = bob.filter((b) => b.facing === front);
      if (!mine.length) return;
      if (!front) { mine.forEach((b) => P.p(b.q.map((p) => [p[0], p[1]]))); P.fill(HAIR, { pattern: softStrands, round: 1, dim: true }); return; }
      // (the shape itself was added to the head's hair mask and filled with it; here come the details on top)
      // the curled-under ends sit in shadow
      mine.filter((b) => b.band === 3).forEach((b) => P.p(b.q.map((p) => [p[0], p[1]])));
      P.fill(HAIR, { pattern: softStrands, round: 0.6, lift: -0.2 });
      // strands: curved lines down each lock, longer and shorter, not evenly spaced
      for (let k = 0; k <= 26; k++) {
        const phi = FACE + 0.05 + k / 26 * (Math.PI * 2 - 2 * FACE - 0.1) + Math.sin(k * 2.7) * 0.04;
        const top = HEAD + 0.6 - (k % 3) * 0.9, bend = Math.sin(k * 1.7) * 0.08;
        const a = V.seam(7, 7.4, phi, top), m = V.seam(7.35 + bulge(phi) * 0.6, 7.55 + bulge(phi) * 0.6, phi + bend, HEAD - 3.4), b2 = V.seam(7.4, 7.45, phi + bend * 1.6, endH(phi) + 1.2 + (k % 2) * 0.8);
        if (m[2] < 1.2) continue;
        P.l(a[0], a[1], m[0], m[1], 0.5).l(m[0], m[1], b2[0], b2[1], 0.5);
      }
      P.fill(HAIR, { flat: true, tone: 1 });
      // a few lit strands catching the light on the upper part of each lock
      for (let k = 0; k <= 12; k++) {
        const phi = FACE + 0.2 + k / 12 * (Math.PI * 2 - 2 * FACE - 0.4), m = V.seam(7.3 + bulge(phi) * 0.6, 7.5 + bulge(phi) * 0.6, phi + 0.06, HEAD - 2.2);
        const a = V.seam(7.05, 7.45, phi + 0.04, HEAD + 0.2);
        if (m[2] > 2 && (k % 2 === 0)) P.l(a[0] + 0.3, a[1], m[0] + 0.3, m[1], 0.4);
      }
      P.fill(HAIR, { flat: true, tone: 3 });
    };
    add(-1, () => drawBob(false));
    // Crown, fringe and the near side of the bob are one shape, filled in one pass so the shading runs
    // continuously from the bangs down the sides with no seam; the face is cut out of the whole.
    add(20.3, () => {
      P.e(V.x(0, -0.4), V.y(HEAD + 0.6), V.w(6.75, 7.1), 8);
      bob.filter((b) => b.facing).forEach((b) => P.p(b.q.map((p) => [p[0], p[1]])));
      if (V.fz > -0.25) {
        [1, -1].forEach((sd) => {
          const pts = [];
          for (let k = 0; k <= 10; k++) { const phi = sd * k / 10 * FACE; pts.push(V.seam(7.1, 7.5, phi, hairline(phi)).slice(0, 2)); }
          pts.push(V.seam(7.1, 7.5, sd * FACE, HEAD - 3.4).slice(0, 2), V.seam(5.4, 7, sd * FACE * 0.9, HEAD - 12).slice(0, 2), V.seam(5.6, 7.2, 0, HEAD - 12).slice(0, 2));
          P.cut(pts);
        });
      }
      P.fill(HAIR, { pattern: softStrands, round: 1.1 });
      drawBob(true);
      // a soft band of light across the upper left of the crown, following the curve of the head
      for (let k = 0; k <= 18; k++) {
        const phi = -1.5 + k / 18 * 2.2, q = V.seam(6.3, 6.7, phi, HEAD + 5.6 + Math.cos(phi + 0.4) * 0.8);
        if (q[2] > 1.2) { P.dotOn(q[0], q[1], HAIR[3]); P.dotOn(q[0] + 0.5, q[1] - 0.5, HAIR[3]); if (k % 3 === 1) P.dotOn(q[0], q[1] + 0.5, HAIR[4]); }
      }
      // the fringe casts a soft shadow onto the forehead
      if (V.fz > -0.25) for (let k = 0; k <= 30; k++) {
        const phi = -FACE + k / 30 * 2 * FACE, h = hairline(phi), q = V.seam(7.1, 7.5, phi, h - 0.7), q2 = V.seam(7.1, 7.5, phi, h - 1.2);
        if (q[2] > 1) { P.darken(q[0], q[1], 0.86); P.darken(q2[0], q2[1], 0.93); }
      }
      // sheen across the crown
      for (let k = 0; k <= 14; k++) { const phi = -1 + k / 14 * 2, q = V.seam(6.2, 6.6, phi, HEAD + 5.4 + Math.cos(phi) * 0.6); if (q[2] > 2 && k % 3 === 0) P.dotOn(q[0], q[1], HAIR[3]); }
      // side parting
      for (let k = 0; k <= 7; k++) { const q = V.seam(2.6, 6.6 - k * 1.5, 0, HEAD + 7.4 - Math.abs(k - 3) * 0.25); if (q[2] > -3) P.dotOn(V.x(-2.4, 6.6 - k * 1.5), q[1], HAIR[1]); }
      // fringe: strand shadows sweeping across the forehead, and darker tips along the edge
      if (V.fz > -0.25) {
        [-0.9, -0.45, 0, 0.45].forEach((phi) => {
          const a = V.seam(6.6, 7, phi - 0.25, hairline(phi - 0.25) + 2.4), b = V.seam(7.05, 7.45, phi + 0.3, hairline(phi + 0.3) + 0.6);
          if (b[2] > 3 && a[2] > 3) P.l(a[0], a[1], b[0], b[1], 0.5);
        });
        P.fill(HAIR, { flat: true, tone: 1 });
        for (let k = 0; k <= 26; k++) { const phi = -FACE + k / 26 * 2 * FACE, h = V.seam(7.1, 7.5, phi, hairline(phi) + 0.25); if (h[2] > 1) P.dotOn(h[0], h[1], HAIR[1]); }
      }
    });
  }
};

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

export { PW, PH, RES, FW, FH, COLS, ROWS, WALK, FIGHTER, MAGE, RANGER, paintSheet };
