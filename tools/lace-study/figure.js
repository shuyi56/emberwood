// The project's base female figure (blender/base/female.py), its trunk and legs, as a body for the solver to hang
// cloth on and for the bench to draw. The shapes are the figure's own tables, copied unchanged:
//  - TORSO: female.py's half-width, front depth and back depth at each height, eased by a cosine between rows, with
//    torso_point's seat (two soft forms on the back over the hips) and bust;
//  - LEG_R: the Warden's leg profile (blender/warden/warden.py, the figure's bare leg), from 2 above the hip joint
//    down to the ankle, 1.04 times as deep as wide;
//  - HIP_Z, LEG_X, THIGH, SHIN: female.py's joints.
// The figure is in design pixels (one Blender unit), 63 tall to the top of the head. At a woman of 165 cm a unit is
// 2.62 cm (CM), which gives her 88 cm round the hips and 52 round the waist: a slim figure, as the sprites are.
// Blender's frame (x to her left side, -y her front, z up) becomes the solver's (x, z up as y, -y as z), so her
// front faces +z, the bench's front camera.

export const CM = 165 / 63;
// (one name to a declaration: the bench's build exports only the first of a list)
export const HIP_Z = 31.0;
export const LEG_X = 3.3;
export const THIGH = 14.0;
export const SHIN = 13.8;
export const ANKLE_Z = 3.4;
export const TORSO = [
  [53.0, 1.6, 1.5, 1.7], [51.4, 2.8, 2.2, 2.5], [50.4, 5.0, 2.6, 2.9], [49.2, 6.0, 2.8, 3.1], [47.2, 5.6, 2.9, 3.1],
  [44.5, 5.2, 3.0, 3.0], [42.0, 4.35, 2.85, 2.75], [39.4, 3.65, 2.6, 2.5], [37.0, 4.5, 2.75, 2.85], [34.5, 5.9, 3.1, 3.6],
  [32.0, 6.6, 3.3, 4.0], [30.0, 6.3, 3.3, 3.8], [28.5, 5.6, 3.1, 3.4],
];
export const LEG_R = [[2.0, 3.05], [-1.0, 2.95], [-6.0, 2.45], [-THIGH + 1.2, 1.98], [-THIGH, 1.93], [-THIGH - 1.5, 1.9],
  [-THIGH - 4.0, 1.92], [-THIGH - SHIN, 1.3]];

const ease = (t) => (1 - Math.cos(Math.PI * t)) / 2;

export function torsoDims(z) {
  const T = TORSO;
  if (z >= T[0][0]) return T[0].slice(1);
  if (z <= T[T.length - 1][0]) return T[T.length - 1].slice(1);
  for (let k = 0; k < T.length - 1; k++) {
    const a = T[k], b = T[k + 1];
    if (b[0] <= z && z <= a[0]) { const t = ease((a[0] - z) / (a[0] - b[0])); return [1, 2, 3].map((q) => a[q] + (b[q] - a[q]) * t); }
  }
  return T[T.length - 1].slice(1);
}

// female.py's torso_point: the body at height z (units), phi round from the front toward her right, pushed out by
// off; returns Blender (x, y)
export function torsoPoint(z, phi, off = 0, bust = 1) {
  const [rx, fy, by] = torsoDims(z), sx = Math.sin(phi), c = Math.cos(phi);
  const x = -(rx + off) * sx;
  let y;
  if (c >= 0) {
    y = -(fy + off) * c;
    for (const s of [-1, 1]) { const zc = 44.3, sz = z > zc ? 2.3 : 1.6; y -= bust * 1.25 * Math.exp(-(((x - s * 2.2) / 2.2) ** 2) - (((z - zc) / sz) ** 2)) * c; }
  } else {
    y = -(by + off) * c;
    for (const s of [-1, 1]) y -= 0.9 * Math.exp(-(((x - s * 2.6) / 2.3) ** 2) - (((z - 32.3) / 2.3) ** 2)) * c;
  }
  return [x, y];
}

export function legRadius(zr) {
  const T = LEG_R;
  if (zr >= T[0][0]) return T[0][1];
  for (let k = 0; k < T.length - 1; k++) {
    const [za, ra] = T[k], [zb, rb] = T[k + 1];
    if (zb <= zr && zr <= za) return ra + (rb - ra) * ease((za - zr) / (za - zb));
  }
  return T[T.length - 1][1];
}

// a point of the solver's (cm, y up, front +z, the origin at height `z0` units on her centre line) in her units
const toFig = (q, z0) => [q[0] / CM, -q[2] / CM, q[1] / CM + z0];

// The torso's ring at height z as a closed curve in the solver's x-z plane (cm), n points from the front round:
// for laying a waistband or a yoke on her
export function ring(z, off = 0, n = 256, phi0 = 0) {
  const out = [];
  for (let k = 0; k < n; k++) { const [x, y] = torsoPoint(z, phi0 + 2 * Math.PI * k / n, off, 0); out.push([x * CM, -y * CM]); }
  return out;
}

// The ring as cloth lies round it: a band does not follow her into the hollow between the seat's two forms, so the
// ring's convex hull, resampled evenly by arc length and smoothed so its curvature has no corners (a yoke's line)
export function bandRing(z, off = 0, n = 512, phi0 = 0) {
  const R = ring(z, off, 1024, phi0);
  // (the hull, by the monotone chain, kept in the ring's own turning order)
  const pts = R.map((q, k) => [q[0], q[1], k]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], hi = [];
  for (const p of pts) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); }
  for (let i = pts.length - 1; i >= 0; i--) { const p = pts[i]; while (hi.length >= 2 && cross(hi[hi.length - 2], hi[hi.length - 1], p) <= 0) hi.pop(); hi.push(p); }
  const keep = new Set([...lo, ...hi].map((p) => p[2]));
  let H = R.filter((_, k) => keep.has(k));
  // evenly by arc length
  const even = (P, m) => {
    const S = [0]; for (let k = 1; k <= P.length; k++) { const a = P[k - 1], b = P[k % P.length]; S.push(S[k - 1] + Math.hypot(b[0] - a[0], b[1] - a[1])); }
    const out = []; let k = 0;
    for (let q = 0; q < m; q++) { const s = S[P.length] * q / m; while (S[k + 1] < s) k++; const a = P[k], b = P[(k + 1) % P.length], t = (s - S[k]) / ((S[k + 1] - S[k]) || 1); out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]); }
    return out;
  };
  H = even(H, n);
  for (let it = 0; it < 30; it++) H = H.map((q, k) => { const a = H[(k + n - 1) % n], b = H[(k + 1) % n]; return [0.5 * q[0] + 0.25 * (a[0] + b[0]), 0.5 * q[1] + 0.25 * (a[1] + b[1])]; });
  return even(H, n);
}

// The body for the solver: pushes a point (cm) out of her trunk (heights 28.5 to 53) and her legs (the hip joint
// down to the ankle), `pad` cm clear of the skin. z0: the height, in her units, of the solver's y = 0.
export function bodyFn(z0, pad = 0.3) {
  const pu = pad / CM;
  return (q) => {
    const [x, y, z] = toFig(q, z0);
    let hit = false, best = null;
    // the trunk: find the point's own angle on her ellipse-like ring, then compare with her surface there
    if (z <= 53 && z >= 28.5) {
      const r = Math.hypot(x, y);
      if (r > 1e-6) {
        const [rx, fy, by] = torsoDims(z), th = Math.atan2(-x, -y), d = Math.cos(th) >= 0 ? fy : by;
        // (torso_point's phi for a point at polar angle th: tan phi = depth tan th / half-width)
        let phi = Math.atan2(d * Math.sin(th), rx * Math.cos(th));
        let s = torsoPoint(z, phi, pu, 0);
        for (let it = 0; it < 2; it++) { const ts = Math.atan2(-s[0], -s[1]); phi += th - ts; s = torsoPoint(z, phi, pu, 0); }
        const rs = Math.hypot(s[0], s[1]);
        if (r < rs) { best = [x * rs / r, y * rs / r]; hit = true; }
      }
    }
    // the legs: each a round column about its hip joint's line, 1.04 times as deep as wide
    // (only below the trunk's bottom: above it her legs are inside her hips, and a leg's column standing out of the
    // hips' sides pushed the skirt's second row 10% out of line under the yoke)
    if (!hit && z < 29.0 && z > ANKLE_Z) {
      for (const s of [-1, 1]) {
        const cx = s * LEG_X, dx = x - cx, dy = y / 1.04, r = Math.hypot(dx, dy), R = legRadius(z - HIP_Z) + pu;
        if (r < R && r > 1e-6) { best = [cx + dx * R / r, dy * R / r * 1.04]; hit = true; break; }
      }
    }
    if (!hit) return false;
    q[0] = best[0] * CM; q[2] = -best[1] * CM;
    return true;
  };
}

// her trunk and legs as triangle meshes for the bench to draw (cm, the solver's frame)
export function meshes(z0, { top = 50.4 } = {}) {
  const out = [];
  const tube = (zs, pt, n = 48) => {
    const P = [], I = [];
    for (const z of zs) for (let k = 0; k < n; k++) { const [x, y] = pt(z, 2 * Math.PI * k / n); P.push(x * CM, (z - z0) * CM, -y * CM); }
    for (let a = 0; a < zs.length - 1; a++) for (let k = 0; k < n; k++) {
      const i = a * n + k, j = a * n + (k + 1) % n;
      I.push(i, j, i + n, j, j + n, i + n);
    }
    return { positions: Float32Array.from(P), indices: Uint32Array.from(I) };
  };
  const tz = []; for (let z = top; z >= 28.5 - 1e-9; z -= 0.25) tz.push(z);
  out.push(tube(tz, (z, phi) => torsoPoint(z, phi, 0, 1)));
  for (const s of [-1, 1]) {
    const lz = []; for (let z = HIP_Z + 1.5; z >= ANKLE_Z; z -= 0.4) lz.push(z);
    out.push(tube(lz, (z, phi) => { const r = legRadius(z - HIP_Z); return [s * LEG_X - r * Math.sin(phi), -r * 1.04 * Math.cos(phi)]; }, 32));
  }
  return out;
}

// the yoke the pleats hang from, drawn (not solved): her band line from the waist down to the yoke's seam, standing
// out from her by `off0` at the waist and `off1` at the seam (out over the pleats' top, so no step shows)
export function yokeMesh(z0, zTop, zBot, off0, off1, n = 96) {
  const zs = []; for (let z = zTop; z >= zBot - 1e-9; z -= 0.25) zs.push(z);
  const P = [], I = [];
  for (const z of zs) {
    const t = (zTop - z) / (zTop - zBot), off = off0 + (off1 - off0) * t * t;
    for (const q of bandRing(z, off / CM, n, Math.PI)) P.push(q[0], (z - z0) * CM, q[1]);
  }
  for (let a = 0; a < zs.length - 1; a++) for (let k = 0; k < n; k++) { const i = a * n + k, j = a * n + (k + 1) % n; I.push(i, j, i + n, j, j + n, i + n); }
  return { positions: Float32Array.from(P), indices: Uint32Array.from(I) };
}
