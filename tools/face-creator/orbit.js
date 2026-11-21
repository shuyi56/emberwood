/* ======================================================================================================
   The eye socket (orbit), the lids and the eyeball, carved into the head's ring grid (the JavaScript port of
   blender/shared/orbit.py, kept line for line with it: a change there comes here too). Units are the model's,
   in orbit.py's frame: x across, y depth (the face toward -y), z up; the eye at x > 0 (side s mirrors it).
   1. orbitSpec    measure the head and derive the eye from it (life sizes scaled to the head)
   2. orbitBed     the surface the lids rest on: the ball (with the cornea's bulge), past its edge the socket's tissue
   3. orbitCarve   a block of the ring grid round each eye re-laid as concentric loops: the skin of the lids and
                   folds, the margin (lash line, rounded margin, wet line on the ball), and the conjunctiva running
                   back round the ball and closed behind it, so the head has no hole
   4. orbitGlobe   the eyeball mesh, on the same surface the lids were laid on
   5. orbitCaruncle the caruncle in the inner corner
   ====================================================================================================== */
var ORBIT_LIFE = {
  head_h: 219.0, globe_r: 12.0, cornea_r: 7.8, limbus_r: 5.85, iris_back: 0.5, film: 0.06, margin_t: 1.8, lid_t: 1.4,
  cover_up: 1.5, cover_lo: 0.3, lift: 1.3, lateral: 1.1, medial: 0.25, crease: 8.0, crease_d: 0.5, lower_fold: 9.0,
  brow_ahead: 10.0, cheek_behind: 2.0, caruncle_w: 5.0
};
function osm(e0, e1, x) { var t = (x - e0) / (e1 - e0); t = t < 0 ? 0 : t > 1 ? 1 : t; return t * t * (3 - 2 * t); }
function osmax(a, b, k) { var h = 0.5 + 0.5 * (a - b) / k; h = h < 0 ? 0 : h > 1 ? 1 : h; return b + (a - b) * h + k * h * (1 - h); }
function ointerp(x, xp, fp) {       // np.interp: xp increasing, clamped at the ends
  var n = xp.length;
  if (x <= xp[0]) return fp[0];
  if (x >= xp[n - 1]) return fp[n - 1];
  var lo = 0, hi = n - 1;
  while (hi - lo > 1) { var m = (lo + hi) >> 1; if (xp[m] <= x) lo = m; else hi = m; }
  var t = (x - xp[lo]) / (xp[hi] - xp[lo]);
  return fp[lo] + (fp[hi] - fp[lo]) * t;
}
function otab(tab, u) { var f = (Math.min(1, Math.max(-1, u)) + 1) / 2 * (tab.length - 1), i = Math.min(tab.length - 2, Math.floor(f)); return tab[i] + (tab[i + 1] - tab[i]) * (f - i); }

/* 1. the measure. H: chin to crown; ipd: between the pupils; pupilZ: their height; frontY(x, z): the depth of the
   head's front at a point; tabTop, tabBot: the lids' curves (heights in half-widths over the corner line, 401
   samples over u -1..1); openK: the lids' opening against the curve's own (1: as derived from the iris) */
function orbitSpec(H, ipd, pupilZ, frontY, tabTop, tabBot, o) {
  o = o || {};
  var L = Object.assign({}, ORBIT_LIFE, o.life || {}), scale = o.scale || 1, tilt = o.tilt === undefined ? 0.092 : o.tilt;
  var mm = H / L.head_h;
  var e = mm * scale, r = L.globe_r * e, Rc = L.cornea_r * e, lim = L.limbus_r * e;
  var zl = Math.sqrt(r * r - lim * lim), cc = zl - Math.sqrt(Rc * Rc - lim * lim), apex = cc + Rc, px = ipd / 2;
  var brow = Infinity, cheek = Infinity, k;
  for (k = 0; k < 17; k++) brow = Math.min(brow, frontY(px, pupilZ + H * (0.06 + 0.08 * k / 16)));
  for (k = 0; k < 15; k++) cheek = Math.min(cheek, frontY(px, pupilZ - H * (0.06 + 0.07 * k / 14)));
  var apexY = 0.5 * ((brow + L.brow_ahead * mm) + (cheek + L.cheek_behind * mm));
  var med = px - L.medial * ipd * scale, lat = L.lateral * r, a = 0.5 * (med + lat), xc = 0.5 * (lat - med), lift = L.lift * e;
  var u0 = -xc / a, openK = o.openK || 1;
  var kt = ((lim - L.cover_up * e) + lift) / (a * otab(tabTop, u0)) * openK;
  var kb = ((lim - L.cover_lo * e) - lift) / (a * otab(tabBot, u0)) * openK;
  // (o.almond: the opening takes the creator's own almond for its shape: the lids' curves at their own heights
  // over the half-width (so the same proportions as the curves were drawn with in 2D), and the corner line as far
  // below the pupil, in half-widths, as the creator sets it. The corners and the ball stay as measured)
  if (o.almond) { kt = kb = 1; lift = o.almond.liftRatio * a; }
  var S = {
    mm: mm, e: e, H: H, ipd: ipd, scale: scale, cx: px, cy: apexY + apex, cz: pupilZ,
    r: r, cornea_R: Rc, cornea_c: cc, apex: apex, limbus: lim, limbus_z: zl, iris_r: lim, iris_back: L.iris_back * e,
    film: L.film * e, margin_t: L.margin_t * e, lid_t: L.lid_t * e, med: med, lat: lat, a: a, xc: xc, lift: lift,
    tilt: tilt, kt: kt, kb: kb, crease: L.crease * e, crease_d: L.crease_d * e, lower_fold: L.lower_fold * e,
    caruncle_w: L.caruncle_w * e, tabTop: tabTop, tabBot: tabBot,
    measured: { brow_y: brow, cheek_y: cheek, apex_y: apexY, pupil_z: pupilZ, fissure_mm: 2 * a / mm }
  };
  S.measured.aperture_mm = (orbitMarginZ(S, 0, true) - orbitMarginZ(S, 0, false)) / mm;
  return S;
}
/* the wet line in the front view: its z at xl out from the pupil (lateral +) */
function orbitMarginZ(S, xl, top) {
  var u = Math.min(1, Math.max(-1, (xl - S.xc) / S.a)), zline = S.cz - S.lift + S.tilt * xl;
  return top ? zline + S.kt * S.a * otab(S.tabTop, u) : zline - S.kb * S.a * otab(S.tabBot, u);
}

/* 2. the bed */
function orbitGlobeRadius(S, th, k) {
  var sn = Math.sin(Math.min(th, Math.PI / 2)), cc = S.cornea_c, Rc = S.cornea_R;
  var tc = th < Math.PI / 2 ? cc * Math.cos(th) + Math.sqrt(Math.max(0, Rc * Rc - cc * sn * cc * sn)) : 0;
  return osmax(S.r, tc, k === undefined ? 0.35 * S.e : k);
}
function orbitProfile(S, soft) {
  var key = soft ? '_profSoft' : '_prof';
  if (S[key]) return S[key];
  var r = S.r, N = 1200, NT = 4000, D = new Float64Array(N), Y = new Float64Array(N), G = new Float64Array(N);
  var pairs = [], kk = (soft ? 2.5 : 0.9) * S.e, i;
  for (i = 0; i < NT; i++) { var th = Math.PI / 2 * i / (NT - 1), R = orbitGlobeRadius(S, th, kk); pairs.push([R * Math.sin(th), -R * Math.cos(th)]); }
  pairs.sort(function (p, q) { return p[0] - q[0]; });     // (np.argsort: stable on ties, as sort is)
  var dd = new Float64Array(NT), yy = new Float64Array(NT), ddMax = -Infinity;
  for (i = 0; i < NT; i++) { dd[i] = pairs[i][0]; yy[i] = pairs[i][1]; ddMax = Math.max(ddMax, dd[i]); }
  for (i = 0; i < N; i++) {
    D[i] = 1.75 * r * i / (N - 1);
    var ball = D[i] < ddMax ? ointerp(D[i], dd, yy) : 0, tissue = -0.55 * r + 0.22 * D[i] * D[i] / r;
    Y[i] = -osmax(-ball, -tissue, 0.1 * r);
  }
  for (i = 0; i < N; i++) G[i] = i === 0 ? (Y[1] - Y[0]) / (D[1] - D[0]) : i === N - 1 ? (Y[N - 1] - Y[N - 2]) / (D[N - 1] - D[N - 2]) : (Y[i + 1] - Y[i - 1]) / (D[i + 1] - D[i - 1]);
  return (S[key] = { D: D, Y: Y, G: G });
}
function orbitBed(S, x, z, soft) {
  var p = orbitProfile(S, soft), d = Math.hypot(x - S.cx, z - S.cz), Dl = p.D[p.D.length - 1];
  return S.cy + ointerp(d, p.D, p.Y) + (d > Dl ? p.G[p.G.length - 1] * (d - Dl) : 0);
}
function orbitBedSlope(S, x, z) { var p = orbitProfile(S, false); return ointerp(Math.hypot(x - S.cx, z - S.cz), p.D, p.G); }

/* the skin's surface: grids are Float64Array ny * nx, row-major (row j: z, column i: x) */
function oRelax(Y, fixed, ny, nx, iters) {
  var A = new Float64Array(Y.length), cur = Float64Array.from(Y);
  for (var it = 0; it < iters; it++) {
    for (var j = 0; j < ny; j++) {
      var jm = ((j - 1 + ny) % ny) * nx, jp = ((j + 1) % ny) * nx, jo = j * nx;
      for (var i = 0; i < nx; i++) {
        var k = jo + i;
        A[k] = fixed[k] ? cur[k] : 0.25 * (cur[jm + i] + cur[jp + i] + cur[jo + (i - 1 + nx) % nx] + cur[jo + (i + 1) % nx]);
      }
    }
    var t = cur; cur = A; A = t;
  }
  return cur;
}
function oHarmonic(Y, fixed, ny, nx, levels) {
  if (levels === undefined) levels = 5;
  if (levels === 0 || Math.min(ny, nx) < 12) return oRelax(Y, fixed, ny, nx, 600);
  var h = (ny >> 1) * 2, w = (nx >> 1) * 2, cy = h >> 1, cx = w >> 1, yc = new Float64Array(cy * cx), fc = new Uint8Array(cy * cx), j, i;
  for (j = 0; j < cy; j++) for (i = 0; i < cx; i++) {
    var s = 0, cnt = 0, all = 0;
    for (var dj = 0; dj < 2; dj++) for (var di = 0; di < 2; di++) { var k = (2 * j + dj) * nx + 2 * i + di; all += Y[k]; if (fixed[k]) { s += Y[k]; cnt++; } }
    yc[j * cx + i] = cnt > 0 ? s / cnt : all / 4; fc[j * cx + i] = cnt > 0 ? 1 : 0;
  }
  yc = oHarmonic(yc, fc, cy, cx, levels - 1);
  var Y2 = Float64Array.from(Y);
  for (j = 0; j < h; j++) for (i = 0; i < w; i++) { var kk = j * nx + i; if (!fixed[kk]) Y2[kk] = yc[(j >> 1) * cx + (i >> 1)]; }
  return oRelax(Y2, fixed, ny, nx, 120);
}
function oBlur(Y, ny, nx, n) {
  var ker = new Float64Array(2 * n + 1), s = 0, i, j, k;
  for (k = 0; k <= 2 * n; k++) { var t = -2.5 + 5 * k / (2 * n); ker[k] = Math.exp(-t * t); s += ker[k]; }
  for (k = 0; k <= 2 * n; k++) ker[k] /= s;
  var T = new Float64Array(Y.length), O = new Float64Array(Y.length);
  for (j = 0; j < ny; j++) for (i = 0; i < nx; i++) { var a = 0; for (k = -n; k <= n; k++) { var ii = Math.min(nx - 1, Math.max(0, i + k)); a += ker[k + n] * Y[j * nx + ii]; } T[j * nx + i] = a; }
  for (j = 0; j < ny; j++) for (i = 0; i < nx; i++) { var b = 0; for (k = -n; k <= n; k++) { var jj = Math.min(ny - 1, Math.max(0, j + k)); b += ker[k + n] * T[jj * nx + i]; } O[j * nx + i] = b; }
  return O;
}
function oArange(a, b, h) { var n = Math.ceil((b - a) / h), o = new Float64Array(n); for (var i = 0; i < n; i++) o[i] = a + i * h; return o; }
function oSearch(arr, v, right) { var lo = 0, hi = arr.length; while (lo < hi) { var m = (lo + hi) >> 1; if (right ? arr[m] <= v : arr[m] < v) lo = m + 1; else hi = m; } return lo; }
function oResample(P, n) {      // P: [[x, z], ...] -> n points evenly by arc length (both ends kept)
  var s = [0], i;
  for (i = 1; i < P.length; i++) s.push(s[i - 1] + Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]));
  var sx = P.map(function (p) { return p[0]; }), sz = P.map(function (p) { return p[1]; }), out = [];
  for (i = 0; i < n; i++) { var t = s[s.length - 1] * i / (n - 1); out.push([ointerp(t, s, sx), ointerp(t, s, sz)]); }
  return out;
}
/* the skin round the eye as one smooth surface y(x, z) over the box [x0, x1, z0, z1] (orbit.py's skin_field) */
function orbitSkinField(S, headY, box, M, nh) {
  var e = S.e, h = 0.3 * e, x0 = box[0], x1 = box[1], z0 = box[2], z1 = box[3], pad = 6 * h;
  var xs = oArange(x0 - pad, x1 + pad + h, h), zs = oArange(z0 - pad, z1 + pad + h, h), nx = xs.length, ny = zs.length, N = nx * ny;
  var dense = oResample(M.slice(0, nh + 1), 400).concat(oResample(M.slice(nh).concat([M[0]]), 400)), nd = dense.length;
  var dist = new Float64Array(N), nearUp = new Float64Array(N), along = new Float64Array(N), X = new Float64Array(N), Z = new Float64Array(N);
  var i, j, k, q;
  for (j = 0; j < ny; j++) for (i = 0; i < nx; i++) {
    k = j * nx + i; X[k] = xs[i]; Z[k] = zs[j];
    var best = Infinity, bq = 0;
    for (q = 0; q < nd; q++) { var dx = xs[i] - dense[q][0], dz = zs[j] - dense[q][1], d2 = dx * dx + dz * dz; if (d2 < best) { best = d2; bq = q; } }
    var dd = Math.sqrt(best), xl = xs[i] - S.cx, zt = orbitMarginZ(S, xl, true), zb = orbitMarginZ(S, xl, false);
    var inside = xl > -S.med && xl < S.lat && zs[j] < zt && zs[j] > zb;
    dist[k] = inside ? -dd : dd; nearUp[k] = bq < 400 ? 1 : 0; along[k] = Math.abs(Math.sin(Math.PI * (bq % 400) / 399));
  }
  var nrm = new Float64Array(N), L = new Float64Array(N), Hh = new Float64Array(N), out0 = new Uint8Array(N), edge = new Float64Array(N);
  for (k = 0; k < N; k++) {
    var bs = orbitBedSlope(S, X[k], Z[k]);
    nrm[k] = Math.min(1.5, Math.sqrt(1 + bs * bs));
    var thick = S.lid_t + (S.margin_t - S.lid_t) * Math.exp(-Math.pow(Math.max(dist[k], 0) / (1.2 * e), 2));
    L[k] = orbitBed(S, X[k], Z[k], true) - thick * nrm[k];
    Hh[k] = headY(X[k], Z[k]);
    out0[k] = (X[k] < x0 || X[k] > x1 || Z[k] < z0 || Z[k] > z1) ? 1 : 0;
    edge[k] = Math.min(X[k] - x0, x1 - X[k], Z[k] - z0, z1 - Z[k]);
  }
  // the span: a Coons patch over the box, cubic along rows and columns from the head's depth and slope outside it
  function hermiteFill(V, axis) {
    var coord = axis === 0 ? zs : xs, lo = axis === 0 ? z0 : x0, hi = axis === 0 ? z1 : x1, n = coord.length;
    var a_ = Math.max(1, oSearch(coord, lo, false) - 1), b_ = Math.min(n - 2, oSearch(coord, hi, true)), Ln = coord[b_] - coord[a_];
    var O = Float64Array.from(V);
    for (var jj = 0; jj < ny; jj++) for (var ii = 0; ii < nx; ii++) {
      var c = axis === 0 ? jj : ii;
      if (coord[c] < coord[a_] || coord[c] > coord[b_]) continue;
      var t = Math.min(1, Math.max(0, (coord[c] - coord[a_]) / Ln)), t2 = t * t, t3 = t2 * t;
      var at = function (cc) { return axis === 0 ? V[cc * nx + ii] : V[jj * nx + cc]; };
      var va = at(a_), vb = at(b_), sa = (at(a_) - at(a_ - 1)) / h, sb = (at(b_ + 1) - at(b_)) / h;
      O[jj * nx + ii] = (2 * t3 - 3 * t2 + 1) * va + (t3 - 2 * t2 + t) * Ln * sa + (-2 * t3 + 3 * t2) * vb + (t3 - t2) * Ln * sb;
    }
    return O;
  }
  var Lv = hermiteFill(Hh, 0), Lu = hermiteFill(Hh, 1), LuLv = hermiteFill(Lv, 1), span = new Float64Array(N);
  for (k = 0; k < N; k++) span[k] = Lu[k] + Lv[k] - LuLv[k];
  var nb25 = Math.max(2, Math.round(2.5 * e / h));
  span = oBlur(span, ny, nx, nb25);
  for (k = 0; k < N; k++) Hh[k] = Hh[k] + (span[k] - Hh[k]) * osm(0, 6 * e, edge[k]);
  var wu = oBlur(nearUp, ny, nx, Math.max(2, Math.round(2.0 * e / h)));
  var reach = new Float64Array(N), fixed = new Uint8Array(N), core = new Uint8Array(N), Dl = new Float64Array(N), Dm = new Float64Array(N);
  for (k = 0; k < N; k++) {
    // (deeper toward the corners: the canthi are tied back to the bone, the inner one deepest, so there the lid
    // follows the ball down instead of bridging to the face's plane. Held near the plane, the inner third of the
    // opening stood off the ball and showed the socket's lining)
    var sul = (1.5 + 1.0 * wu[k]) * e + (2.5 + 2.5 * osm(0.5 * S.a, -0.5 * S.a, X[k] - S.cx - S.xc)) * e * (1 - osm(0.0, 0.85, along[k]));
    var Lb = L[k], wmed = osm(0.5 * S.a, -0.5 * S.a, X[k] - S.cx - S.xc) * (1 - osm(0.0, 0.85, along[k]));
    L[k] = -osmax(-L[k], -(Hh[k] + sul), 4.0 * e);
    Dm[k] = (Lb - L[k]) * wmed * (1 - osm(0.5 * e, 2.0 * e, dist[k]));      // (the inner corner: the lid apposed to the ball down to the canthus, the lacrimal lake small; added to the surface below: apply-inner-corner.py, apply-inner-corner-blend.py)
    reach[k] = (wu[k] * S.crease + (1 - wu[k]) * 0.8 * S.lower_fold) * (0.3 + 0.7 * Math.pow(along[k], 0.6));
    core[k] = dist[k] <= reach[k] - 2.5 * e ? 1 : 0;
    fixed[k] = core[k] || out0[k];
    Dl[k] = core[k] ? L[k] - Hh[k] : 0;
  }
  Dl = oHarmonic(Dl, fixed, ny, nx);
  var Db = oBlur(Dl, ny, nx, nb25), Y = new Float64Array(N);
  for (k = 0; k < N; k++) {
    Dl[k] = Dl[k] + (Db[k] - Dl[k]) * (1 - out0[k]) * osm(reach[k] - 3.5 * e, reach[k], dist[k]);
    Y[k] = Hh[k] + Dl[k];
    Y[k] = L[k] + (Y[k] - L[k]) * osm(reach[k] - 2.5 * e, reach[k] + 2.5 * e, dist[k]);
  }
  // (the inner corner's extra depth, taken at the margin, softened and faded out over 8e, so the skin beside the corner
  // slopes down with the lid instead of stepping: apply-inner-corner-blend.py)
  var DmB = oBlur(Dm, ny, nx, Math.max(2, Math.round(1.5 * e / h)));
  for (k = 0; k < N; k++) Y[k] += DmB[k] * (1 - osm(1.0 * e, 8 * e, dist[k]));
  var Yb = oBlur(Y, ny, nx, nb25), r = S.r;
  for (k = 0; k < N; k++) {
    Y[k] = Y[k] + (Yb[k] - Y[k]) * (1 - out0[k]) * osm(1.5 * e, 3.5 * e, dist[k]);
    var xl2 = X[k] - S.cx, corner = osm(0, 0.3, Math.min(1, Math.max(0, 1 - Math.abs((xl2 - S.xc) / S.a)))), tb = osm(0, 4 * e, edge[k]);
    var cr = wu[k] * S.crease_d * Math.exp(-Math.pow((dist[k] - S.crease) / (1.1 * e), 2)) * corner;
    var fo = (1 - wu[k]) * 0.35 * S.crease_d * Math.exp(-Math.pow((dist[k] - S.lower_fold) / (1.8 * e), 2)) * corner;
    Y[k] += (cr + fo) * tb;
    var dc = Math.hypot(X[k] - S.cx, Z[k] - S.cz), mc = Math.min(dc, r);
    var ball = S.cy - Math.sqrt(Math.max(0, r * r - mc * mc)), floor = ball - 0.6 * S.lid_t * nrm[k] - 3 * S.film;
    Y[k] = Y[k] + (-osmax(-Y[k], -floor, 0.4 * S.lid_t) - Y[k]) * osm(1.02 * r, 0.9 * r, dc);
    Y[k] = Hh[k] + (Y[k] - Hh[k]) * osm(0, 6 * e, edge[k]);
  }
  function at(x, z) {
    var fx = Math.min(nx - 1.001, Math.max(0, (x - xs[0]) / h)), fz = Math.min(ny - 1.001, Math.max(0, (z - zs[0]) / h));
    var ii = Math.floor(fx), jj = Math.floor(fz), tx = fx - ii, tz = fz - jj, o = jj * nx + ii;
    return Y[o] * (1 - tx) * (1 - tz) + Y[o + 1] * tx * (1 - tz) + Y[o + nx] * (1 - tx) * tz + Y[o + nx + 1] * tx * tz;
  }
  at.grid = { xs: xs, zs: zs, Y: Y };
  return at;
}

/* 3. the carve. gridAt(j, i) -> [x, y, z] (the head's ring grid, rows j up, columns i round; orbit.py's frame);
   block [i0, i1, j0, j1] in grid indices for the eye at x > 0 (side -1: the mirrored block at x < 0). Returns the
   patch's new points (Float64Array, 3 per point), the border as grid [i, j] in order, the quads (4 per quad;
   negative: the border's point -1 - t), and info (lash and wet rings' first index, ring count, nb) */
function oCycle(i0, i1, j0, j1, jc) {
  var c = [], i, j;
  for (j = jc; j < j1; j++) c.push([i1, j]);
  for (i = i1; i > i0; i--) c.push([i, j1]);
  for (j = j1; j > j0; j--) c.push([i0, j]);
  for (i = i0; i < i1; i++) c.push([i, j0]);
  for (j = j0; j < jc; j++) c.push([i1, j]);
  return c;
}
function oUnwrap(a) { var o = [a[0]], off = 0; for (var i = 1; i < a.length; i++) { var d = a[i] - a[i - 1]; if (d > Math.PI) off -= 2 * Math.PI; else if (d < -Math.PI) off += 2 * Math.PI; o.push(a[i] + off); } return o; }
function orbitCarve(gridAt, block, S, headY, side, outRings, inRings) {
  outRings = outRings || 26; inRings = inRings || 9;
  var i0 = block[0], i1 = block[1], j0 = block[2], j1 = block[3], jc = (j0 + j1) >> 1, t, k;
  if (j1 - jc !== jc - j0) throw new Error('the block needs an odd number of rows, its corner row in the middle');
  var cyc = oCycle(i0, i1, j0, j1, jc);
  if (side < 0) cyc = cyc.map(function (c) { return [i0 + i1 - c[0], c[1]]; });
  var nb = cyc.length, nh = nb >> 1, cx = S.cx, cy = S.cy, cz = S.cz;
  var B = cyc.map(function (c) { var p = gridAt(c[1], c[0]); return [side * p[0], p[2]]; });
  // the wet line in the front view, each half resampled to the border's half so each spoke runs border -> lid
  var up = [], lo = [];
  for (k = 0; k < 800; k++) { var xl = S.lat + (-S.med - S.lat) * k / 799; up.push([cx + xl, orbitMarginZ(S, xl, true)]); }
  for (k = 0; k < 800; k++) { var xl2 = -S.med + (S.lat + S.med) * k / 799; lo.push([cx + xl2, orbitMarginZ(S, xl2, false)]); }
  function byBorder(Lc, Bh) {
    var sb = [0], sl = [0], m;
    for (m = 1; m < Bh.length; m++) sb.push(sb[m - 1] + Math.hypot(Bh[m][0] - Bh[m - 1][0], Bh[m][1] - Bh[m - 1][1]));
    for (m = 0; m < sb.length; m++) sb[m] /= sb[sb.length - 1];
    for (m = 1; m < Lc.length; m++) sl.push(sl[m - 1] + Math.hypot(Lc[m][0] - Lc[m - 1][0], Lc[m][1] - Lc[m - 1][1]));
    var slEnd = sl[sl.length - 1]; for (m = 0; m < sl.length; m++) sl[m] /= slEnd;
    var angB = oUnwrap(Bh.map(function (p) { return Math.atan2(p[1] - cz, p[0] - cx); })), angL = oUnwrap(Lc.map(function (p) { return Math.atan2(p[1] - cz, (p[0] - cx) * 0.5); }));
    var fl = angL.map(function (v) { return (v - angL[0]) / (angL[angL.length - 1] - angL[0]); });
    var lx = Lc.map(function (p) { return p[0]; }), lz = Lc.map(function (p) { return p[1]; });
    return Bh.map(function (p, m2) {
      var fa = (angB[m2] - angB[0]) / (angB[angB.length - 1] - angB[0]), f = 0.5 * sb[m2] + 0.5 * ointerp(fa, fl, sl);
      return [ointerp(f, sl, lx), ointerp(f, sl, lz)];
    });
  }
  var Mu = byBorder(up, B.slice(0, nh + 1)), Ml = byBorder(lo, B.slice(nh).concat([B[0]]));
  var M = Mu.slice(0, -1).concat(Ml.slice(0, -1));
  var wr = [];
  for (t = 0; t < nb; t++) { var fr = Math.min(t % nh, nh - t % nh); wr.push(Math.exp(-Math.pow(fr / 4, 2))); }
  for (var it = 0; it < 40; it++) {
    M = M.map(function (p, t2) {
      var a = M[(t2 - 1 + nb) % nb], b = M[(t2 + 1) % nb];
      return [p[0] + 0.5 * wr[t2] * (0.5 * (a[0] + b[0]) - p[0]), p[1] + 0.5 * wr[t2] * (0.5 * (a[1] + b[1]) - p[1])];
    });
  }
  // the skin, from the lash line out to the border: rings eased from the lid's shape to the block's
  var bx0 = Infinity, bx1 = -Infinity, bz0 = Infinity, bz1 = -Infinity;
  B.forEach(function (p) { bx0 = Math.min(bx0, p[0]); bx1 = Math.max(bx1, p[0]); bz0 = Math.min(bz0, p[1]); bz1 = Math.max(bz1, p[1]); });
  var field = orbitSkinField(S, function (x, z) { return headY(side * x, z); }, [bx0, bx1, bz0, bz1], M, nh);
  var skin = [];
  for (k = 0; k < outRings; k++) {
    var rk = k / outRings, fk = Math.pow(rk, 1.35), wb = osm(0.94, 1.0, rk), ring = [];
    for (t = 0; t < nb; t++) {
      var x = M[t][0] + (B[t][0] - M[t][0]) * fk, z = M[t][1] + (B[t][1] - M[t][1]) * fk, y = field(x, z);
      if (wb > 0) y = y + (headY(side * x, z) - y) * wb;
      ring.push([side * x, y, z]);
    }
    skin.push(ring);
  }
  skin.reverse();
  // the margin: lash line (the skin's last ring), a rounded margin, the wet line on the ball
  var inward = [], frac = [], cen = [cx + S.xc, cz - S.lift + S.tilt * S.xc];
  for (t = 0; t < nb; t++) {
    var pa = M[(t + 1) % nb], pb = M[(t - 1 + nb) % nb], tx = pa[0] - pb[0], tz = pa[1] - pb[1], ix = -tz, iz = tx, il = Math.max(1e-9, Math.hypot(ix, iz));
    ix /= il; iz /= il;
    var sg = Math.sign(ix * (cx - M[t][0]) + iz * (cz - M[t][1])); ix *= sg; iz *= sg;
    var f2 = Math.min(t % nh, nh - t % nh) / nh, ccx = cen[0] - M[t][0], ccz = cen[1] - M[t][1], cl = Math.max(1e-9, Math.hypot(ccx, ccz));
    var wc = 1 - osm(0, 0.25, f2);
    ix = ix * (1 - wc) + ccx / cl * wc; iz = iz * (1 - wc) + ccz / cl * wc; il = Math.max(1e-9, Math.hypot(ix, iz));
    inward.push([ix / il, iz / il]); frac.push(f2);
  }
  var PROF = [[0.88, 0.10], [0.65, 0.16], [0.45, 0.17], [0.25, 0.14], [0.08, 0.07]], mt = S.margin_t, rim = [], wet = [];
  var Ay = M.map(function (p) { return field(p[0], p[1]); }), Wy = M.map(function (p) { return orbitBed(S, p[0], p[1]) - S.film; });
  PROF.forEach(function (pr) {
    var ring = [];
    for (t = 0; t < nb; t++) {
      var tp = 0.5 + 0.5 * osm(0, 0.18, frac[t]), xx = M[t][0] + pr[1] * mt * tp * inward[t][0], zz = M[t][1] + pr[1] * mt * tp * inward[t][1];
      var yy = Math.min(Wy[t] + (Ay[t] - Wy[t]) * pr[0], orbitBed(S, xx, zz) - 2 * S.film);
      ring.push([side * xx, yy, zz]);
    }
    rim.push(ring);
  });
  for (t = 0; t < nb; t++) wet.push([side * M[t][0], Wy[t], M[t][1]]);
  // the conjunctiva: from the wet line back round the ball, under the lids, to behind its equator
  var Cx = side * cx, TH_END = 95 * Math.PI / 180, th0 = [], ph0 = [], rad0 = [], phc = [];
  for (t = 0; t < nb; t++) {
    var vx = wet[t][0] - Cx, vy = wet[t][1] - cy, vz = wet[t][2] - cz, rr0 = Math.hypot(vx, vy, vz);
    rad0.push(rr0); th0.push(Math.acos(Math.min(1, Math.max(-1, -vy / rr0)))); ph0.push(Math.atan2(vz, side * vx));
    var ii = cyc[t][0], jj = cyc[t][1]; if (side < 0) ii = i0 + i1 - ii;
    var ua = 2 * (ii - i0) / (i1 - i0) - 1, va = 2 * (jj - j0) / (j1 - j0) - 1;
    var pc = Math.atan2(va * Math.sqrt(1 - ua * ua / 2), ua * Math.sqrt(1 - va * va / 2)), dph = pc - ph0[t];
    dph = Math.atan2(Math.sin(dph), Math.cos(dph));
    phc.push(ph0[t] + dph);
  }
  var sac = [];
  for (k = 1; k <= inRings; k++) {
    var tt = k / inRings, ring2 = [];
    for (t = 0; t < nb; t++) {
      var th = th0[t] + (TH_END - th0[t]) * Math.pow(osm(0, 1, tt), 0.8), ph = ph0[t] + (phc[t] - ph0[t]) * osm(0.35, 1, tt);
      var rr = rad0[t] + (S.r + 3 * S.film - rad0[t]) * osm(0, 1, tt);
      rr = Math.max(rr, orbitGlobeRadius(S, th) + S.film);
      ring2.push([Cx + rr * side * Math.sin(th) * Math.cos(ph), cy - rr * Math.cos(th), cz + rr * Math.sin(th) * Math.sin(ph)]);
    }
    sac.push(ring2);
  }
  // the cap behind the ball: the block's interior grid, its square mapped onto the back of the sphere
  var ni = i1 - i0, nj = j1 - j0, cap = [], capIJ = [], i, j;
  for (j = j0 + 1; j < j1; j++) for (i = i0 + 1; i < i1; i++) {
    var ua_ = 2 * (i - i0) / ni - 1, va_ = 2 * (j - j0) / nj - 1, du = ua_ * Math.sqrt(1 - va_ * va_ / 2), dv = va_ * Math.sqrt(1 - ua_ * ua_ / 2);
    var th2 = Math.PI - (Math.PI - TH_END) * Math.hypot(du, dv), ph2 = Math.atan2(dv, du), R2 = S.r + 3 * S.film;
    cap.push([Cx + R2 * side * Math.sin(th2) * Math.cos(ph2), cy - R2 * Math.cos(th2), cz + R2 * Math.sin(th2) * Math.sin(ph2)]);
    capIJ.push([i, j]);
  }
  var rings = skin.concat(rim, [wet], sac), nP = rings.length * nb + cap.length, P = new Float64Array(nP * 3), o = 0;
  rings.forEach(function (rg) { rg.forEach(function (p) { P[o++] = p[0]; P[o++] = p[1]; P[o++] = p[2]; }); });
  cap.forEach(function (p) { P[o++] = p[0]; P[o++] = p[1]; P[o++] = p[2]; });
  var quads = [], ringIdx = function (r2, t2) { return r2 === 0 ? -1 - t2 : (r2 - 1) * nb + t2; };
  for (var r2 = 0; r2 < rings.length; r2++) for (t = 0; t < nb; t++) {
    var t2 = (t + 1) % nb; quads.push(ringIdx(r2, t), ringIdx(r2, t2), ringIdx(r2 + 1, t2), ringIdx(r2 + 1, t));
  }
  var where = {}, base = rings.length * nb, last = rings.length;
  capIJ.forEach(function (ij, m) { where[ij[0] + ',' + ij[1]] = base + m; });
  cyc.forEach(function (c, m) { var ci = side > 0 ? c[0] : i0 + i1 - c[0]; where[ci + ',' + c[1]] = ringIdx(last, m); });
  for (j = j0; j < j1; j++) for (i = i0; i < i1; i++) quads.push(where[i + ',' + j], where[(i + 1) + ',' + j], where[(i + 1) + ',' + (j + 1)], where[i + ',' + (j + 1)]);
  return { P: P, cyc: cyc, quads: Int32Array.from(quads), info: { lash: (skin.length - 1) * nb, rim: skin.length * nb, wet: (skin.length + rim.length) * nb, sac: (skin.length + rim.length + 1) * nb, cap: base, rings: rings.length, nb: nb, nSkin: skin.length } };
}

/* 4. the eyeball: a sphere on the eye's axis (its pole at the cornea's apex), shaped by orbitGlobeRadius. Points
   (orbit.py's frame) and triangles, wound outward */
function orbitGlobe(S, side, seg, rings) {
  seg = seg || 96; rings = rings || 64;
  var Cx = side * S.cx, P = [], F = [], th = [], k, r;
  for (r = 1; r < rings; r++) th.push(Math.PI * Math.pow(r / rings, 1.35));
  P.push([Cx, S.cy - orbitGlobeRadius(S, 0), S.cz]);
  for (r = 0; r < th.length; r++) {
    var R = orbitGlobeRadius(S, th[r]);
    for (k = 0; k < seg; k++) { var ph = 2 * Math.PI * k / seg; P.push([Cx + side * R * Math.sin(th[r]) * Math.cos(ph), S.cy - R * Math.cos(th[r]), S.cz + R * Math.sin(th[r]) * Math.sin(ph)]); }
  }
  P.push([Cx, S.cy + S.r, S.cz]);
  var n = th.length, idx = function (r2, k2) { return 1 + r2 * seg + (k2 % seg); }, last = P.length - 1;
  for (k = 0; k < seg; k++) F.push([0, idx(0, k + 1), idx(0, k)]);
  for (r = 0; r < n - 1; r++) for (k = 0; k < seg; k++) { F.push([idx(r, k), idx(r, k + 1), idx(r + 1, k + 1)]); F.push([idx(r, k), idx(r + 1, k + 1), idx(r + 1, k)]); }
  for (k = 0; k < seg; k++) F.push([last, idx(n - 1, k), idx(n - 1, k + 1)]);
  return { P: P, F: oFacing(P, F, [Cx, S.cy, S.cz]) };
}
function oFacing(P, F, ref) {
  return F.map(function (f) {
    var a = P[f[0]], b = P[f[1]], c = P[f[2]], ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2], vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
    var nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx, mx = (a[0] + b[0] + c[0]) / 3 - ref[0], my = (a[1] + b[1] + c[1]) / 3 - ref[1], mz = (a[2] + b[2] + c[2]) / 3 - ref[2];
    return nx * mx + ny * my + nz * mz >= 0 ? f : [f[0], f[2], f[1]];
  });
}
/* 5. the caruncle in the inner corner: a soft mound, its outer edge (the plica) a crescent laid on the ball */
function orbitCaruncle(S, side, n, m) {
  n = n || 14; m = m || 8;
  var w = S.caruncle_w, P = [], F = [], i, j;
  for (i = 0; i <= n; i++) for (j = 0; j <= m; j++) {
    var u = i / n, v = j / m, xl = -S.med + w * u * (0.8 + 0.4 * Math.pow(Math.sin(Math.PI * v), 0.8));
    var zt = orbitMarginZ(S, xl, true), zb = orbitMarginZ(S, xl, false), z = zt + (zb - zt) * v;
    var swell = Math.sin(Math.PI * v) * Math.pow(Math.sin(Math.PI * Math.min(1, 0.2 + 0.8 * u)), 0.8) * (1 - u * u * u);
    var y = orbitBed(S, S.cx + xl, z, true) - S.film - Math.min(0.3 * w, 0.6 * S.margin_t) * swell;
    P.push([side * (S.cx + xl), y, z]);
  }
  for (i = 0; i < n; i++) for (j = 0; j < m; j++) { var a = i * (m + 1) + j; F.push([a, a + m + 1, a + m + 2]); F.push([a, a + m + 2, a + 1]); }
  return { P: P, F: oFacing(P, F, [side * S.cx, S.cy, S.cz]) };
}

if (typeof module !== 'undefined') module.exports = { orbitSpec: orbitSpec, orbitCarve: orbitCarve, orbitGlobe: orbitGlobe, orbitCaruncle: orbitCaruncle, orbitBed: orbitBed, orbitMarginZ: orbitMarginZ, orbitSkinField: orbitSkinField };
