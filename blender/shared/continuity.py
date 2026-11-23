"""Continuity: how smoothly two surfaces (or two parts of one surface) run into each other at a join, measured the way
class-A surfacing measures a patch boundary (G0 position, G1 tangent, G2 curvature), plus the shared curvature
function for any profile line (the foot's top line, a lapel's roll, a hem).

Pure numpy, so it runs inside Blender's Python and outside it on dumped meshes (V: (n,3) float, F: (m,3) int).

THE MEASUREMENT (per station along the join)
  The join is a line on the surface (a seam loop, the boundary where one mesh meets another, or a cut). At each of
  `stations` points along it the surface is cut by the plane across the join (its normal is the join's tangent).
  That section is a curve that crosses the join. Each side of the crossing is fitted on its own, a quadratic in arc
  length over `window` (one-sided fits: a fit over the join would smooth the very step being measured; a quadratic's
  curvature is the side's mean over the window, so a curvature that changes smoothly through the join does not read
  as a step), and the two fits are compared at the crossing:
    G0  the gap between the two sides' end points (two meshes) or 0 on one mesh, in mm;
    G1  the angle between the two sides' tangents, in degrees;
    G2  the two sides' curvatures kA, kB (1/cm, signed in the section plane) and Alias's curvature deviation
        d = |kA - kB| / (|kA| + |kB|)   (Autodesk Alias, "Continuity 2: Construction Tolerances"; the same as
        (R1 - R2)/(R1 + R2) in radii), its denominator floored at 2/FLAT_R: near flat both curvatures are tiny and
        the plain ratio jumps to 1 on noise (Alias notes a "flat" fail "is usually not a problem").
  Each step carries its own noise, 1 sigma from the fits' residuals (g1_sig_deg, dk_sig); a step under 3 sigma is
  the mesh's facets and is reported but not graded (g1_real, g2_real).
  The same three numbers are taken again at `control` stations 1.5 windows to each side of the join, on the same
  section. They say how rough the surface is where nothing is joined: a step at the join that is several times the
  control's is the join's (modelling); a step that is as large away from the join is the surface's (tuning:
  smoothing, resolution) or the solver's when the piece is solved.
  On the surface itself the principal curvatures (k1, k2), mean H and Gaussian K are fitted per vertex (a quadric
  over the 2-ring in the vertex's tangent frame) and averaged in a band on each side of the join.

THE LIMITS (LIMITS below; every number is from a source, and where the source is about another kind of product the
derivation is given)
  G0  class A: 0.01 mm (Alias: "a typical Maximum Gap Distance would be between 0.01mm and 0.001mm" for production
      surfaces). Pass for a rendered mesh: the gap opens no more than 0.1 px at the closest camera the queue uses
      (the face crop: rq_views.regions() 'face', 10 u high over --crop-size 384 px, 0.026 u = 0.68 mm per px at this
      figure's scale: 0.07 mm). Fail at a quarter pixel (0.17 mm), where a background line is drawn at 25% coverage.
  G1  class A: 0.1 deg (Alias: "a tangent break of less than 0.1 degrees will normally not be visible on most
      products", glossy, reflective products). Matte (skin, clay, wool): a normal step of dn turns the Lambert
      shade by dI/I = tan(theta) dn; at the queue's key light (theta about 45 deg) dI/I = dn. A luminance edge is
      seen at a Weber contrast of about 1-2% (Weber's law for luminance; Blackwell 1946, J Opt Soc Am 36:624), so a
      matte pass is dn <= 1% = 0.57 deg. Fail: 4 deg, the lowest normal difference people judge even between two
      separate patches with no edge between them (Norman, Todd, Norman, Clayton & McBride 2006, Vision Research
      46:1057, orientation thresholds "between 4 and 10 deg"). Glossy pieces (leather, metal, polished horn) use
      the class-A 0.1 deg as the pass.
  G2  pass d <= 0.10, fail d > 0.23: curvature differences are seen at 10-20% in full-cue viewing (Phillips, Todd,
      Koenderink & Kappers 1997, JEP:HPP 23:1481, "differences on the order of 10-20%"), and 23% was the best
      single observer's ordinal curvedness threshold in Norman et al. 2006 (Table 2). A step in curvature puts a
      kink in the shade's gradient, which the eye sharpens into a Mach band, so the lower end is the pass.
  FLAT_R  1 head height (about 24 cm here): flatter than that on both sides, the deviation is not judged.

UNITS  The model's unit is converted with MM_PER_U (26.19 mm, the figure's 63 u to a 165 cm woman, the scale the foot
study uses); pass mm_per_u to change it.

The one function the foot study and this check share: polyline_curvature (a profile's k(s), smoothed by a local
cubic, the Savitzky-Golay filter written out in numpy) and plane_section (a true plane-mesh intersection, in place
of a slab of vertices)."""
import json
import math
import os

import numpy as np

MM_PER_U = 26.19
FLAT_R_U = 9.39                 # one head height (anatomy.CANON's unit), in model units
LIMITS = {
    'matte': {'g0_mm': (0.07, 0.17), 'g1_deg': (0.57, 4.0), 'g2_dev': (0.10, 0.23)},
    'glossy': {'g0_mm': (0.01, 0.17), 'g1_deg': (0.10, 4.0), 'g2_dev': (0.10, 0.23)},
}
CLASS_A = {'g0_mm': 0.01, 'g1_deg': 0.1}
SOURCES = {
    'g0': 'Autodesk Alias, Continuity 2: Construction Tolerances (max gap 0.01-0.001 mm); 0.1 px / 0.25 px at the '
          'queue face crop (0.68 mm/px)',
    'g1': 'Alias (0.1 deg tangent break not visible); 1% Weber luminance contrast under a 45 deg key (0.57 deg); '
          'Norman et al. 2006 Vision Res 46:1057 (4-10 deg orientation thresholds)',
    'g2': 'Alias curvature deviation (R1-R2)/(R1+R2); Phillips et al. 1997 JEP:HPP 23:1481 (10-20%); Norman et al. '
          '2006 Table 2 (23% best ordinal curvedness)',
}


# ------------------------------------------------------------------ curves

def savgol(y, x_step, window, order=3, deriv=0):
    """Savitzky-Golay smoothing or derivative of evenly spaced samples (numpy only; scipy's savgol_filter with
    mode='interp' at the ends: the end windows use the end's own fit)."""
    y = np.asarray(y, float)
    n = len(y)
    w = max(order + 2, int(round(window / x_step)) | 1)
    if w > n:
        w = n if n % 2 else n - 1
    if w <= order:
        return np.gradient(y, x_step) if deriv == 1 else (np.zeros_like(y) if deriv else y.copy())
    h = w // 2
    t = np.arange(-h, h + 1) * x_step
    A = np.vander(t, order + 1, increasing=True)
    pinv = np.linalg.pinv(A)                       # (order+1, w): row j gives coefficient of t^j
    c = pinv[deriv] * math.factorial(deriv)
    out = np.empty(n)
    if n > 2 * h:
        out[h:n - h] = np.convolve(y, c[::-1], mode='valid')
    for i in list(range(min(h, n))) + list(range(max(n - h, h), n)):
        a = min(max(i - h, 0), n - w)
        coef = np.linalg.lstsq(np.vander((np.arange(a, a + w) - i) * x_step, order + 1, increasing=True),
                               y[a:a + w], rcond=None)[0]
        out[i] = coef[deriv] * math.factorial(deriv)
    return out


def resample(P, step):
    """Points along a polyline every `step` of arc length (the ends kept)."""
    P = np.asarray(P, float)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    if s[-1] <= 0:
        return P[:1], np.zeros(1)
    ss = np.arange(0, s[-1] + 1e-9, step)
    return np.column_stack([np.interp(ss, s, P[:, j]) for j in range(P.shape[1])]), ss


def polyline_curvature(P, step=0.05, window=0.5, order=3, mm_per_u=MM_PER_U, plane_normal=None):
    """The curvature of a profile line, the function the foot study and the join check share.

    P: points along the line (2D, or 3D lying near a plane: plane_normal, else the best-fit plane), any spacing.
    Resampled every `step` u, smoothed by a local cubic over `window` u (Savitzky-Golay), and
    k = (x'y'' - y'x'') / |r'|^3, signed in the plane (positive turning to the left of travel), in 1/cm.
    Returns dict(s_cm, xy (u, in the plane), k (1/cm), dk (1/cm^2), turn_deg (the tangent's direction, deg))."""
    P = np.asarray(P, float)
    if P.shape[1] == 3:
        P = to_plane(P, plane_normal)[0]
    Q, s = resample(P, step)
    x, y = Q[:, 0], Q[:, 1]
    dx, dy = savgol(x, step, window, order, 1), savgol(y, step, window, order, 1)
    ddx, ddy = savgol(x, step, window, order, 2), savgol(y, step, window, order, 2)
    cm = mm_per_u / 10.0
    k = (dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-12) / cm
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return {'s_cm': s * cm, 'xy': np.column_stack([savgol(x, step, window, order), savgol(y, step, window, order)]),
            'xy_raw': Q, 'smooth': window, 'spacing': float(np.median(seg[seg > 1e-9])) if (seg > 1e-9).any() else step,
            'k': k, 'dk': np.gradient(k, step * cm), 'turn_deg': np.degrees(np.unwrap(np.arctan2(dy, dx)))}


def to_plane(P, normal=None, origin=None):
    """Coordinates of 3D points in a plane (u, v), the plane's frame (origin, e1, e2, n)."""
    P = np.asarray(P, float)
    o = P.mean(0) if origin is None else np.asarray(origin, float)
    if normal is None:
        _, _, vt = np.linalg.svd(P - o)
        n = vt[2]
    else:
        n = np.asarray(normal, float) / np.linalg.norm(normal)
    a = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(a, n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    D = P - o
    return np.column_stack([D @ e1, D @ e2]), (o, e1, e2, n)


def one_sided(s, X, at, window, side, order=2, spacing=None):
    """A polynomial fit (quadratic by default: its curvature is the side's mean over the window) of a curve (arc
    length s, points X (n,2)) over [at-window, at] (side -1) or [at, at+window] (side +1), evaluated at `at`:
    position, unit tangent (towards increasing s), signed curvature (1/u), and the 1-sigma uncertainty of the
    tangent's direction (rad) and of the curvature, from the fit's residuals. `spacing` is the distance between
    independent samples (the mesh's crossings): points resampled closer than that are not counted as more data."""
    m = (s >= at - window - 1e-9) & (s <= at + 1e-9) if side < 0 else (s >= at - 1e-9) & (s <= at + window + 1e-9)
    n = int(m.sum())
    if n < order + 2:
        return None
    t = s[m] - at
    deg = min(order, n - 2)
    A = np.vander(t, deg + 1, increasing=True)
    cx, rx = np.linalg.lstsq(A, X[m, 0], rcond=None)[:2]
    cy, ry = np.linalg.lstsq(A, X[m, 1], rcond=None)[:2]
    c2x, c2y = (cx[2], cy[2]) if deg >= 2 else (0.0, 0.0)
    p = np.array([cx[0], cy[0]])
    d1 = np.array([cx[1], cy[1]])
    d2 = 2.0 * np.array([c2x, c2y])
    sp = np.linalg.norm(d1)
    if sp < 1e-12:
        return None
    k = (d1[0] * d2[1] - d1[1] * d2[0]) / sp ** 3
    res = A @ cx - X[m, 0], A @ cy - X[m, 1]
    sig2 = float((res[0] @ res[0] + res[1] @ res[1]) / max(2 * (n - deg - 1), 1))
    n_eff = n if not spacing else max(deg + 2.0, min(n, window / spacing))
    Ci = np.linalg.pinv(A.T @ A) * sig2 * (n / n_eff)
    u = d1 / sp
    sig_t = math.sqrt(Ci[1, 1]) / sp                       # (either component's slope error, across the tangent)
    sig_k = 2.0 * math.sqrt(Ci[2, 2]) / sp ** 2 if deg >= 2 else 0.0
    return p, u, k, sig_t, sig_k


def compare(s, X, at, window, gap_u=0.0, mm_per_u=MM_PER_U, flat_r=FLAT_R_U, order=2, spacing=None):
    """G0/G1/G2 across a point of a section curve: the one-sided fits before and after `at` compared there.
    The deviation's denominator is floored at 2 / FLAT_R, so two nearly flat sides do not read as a 100% step.
    Each step also carries its noise (1 sigma, from the fits' residuals): g1_sig_deg, dk_sig; a step under 3 sigma
    is the mesh's facets, not a break (g1_real, g2_real)."""
    a = one_sided(s, X, at, window, -1, order, spacing)
    b = one_sided(s, X, at, window, +1, order, spacing)
    if a is None or b is None:
        return None
    cm = mm_per_u / 10.0
    pa, ta, ka, sta, ska = a
    pb, tb, kb, stb, skb = b
    g1 = math.degrees(math.acos(max(-1.0, min(1.0, float(ta @ tb)))))
    g1s = math.degrees(math.hypot(sta, stb))
    dks = math.hypot(ska, skb)
    flat = abs(ka) < 1.0 / flat_r and abs(kb) < 1.0 / flat_r
    # (Alias's deviation, its denominator floored at two FLAT_R circles: near flat, where both curvatures are tiny,
    # the plain ratio jumps to 1 on noise; floored, it tends to |dk| FLAT_R / 2)
    dev = abs(ka - kb) / max(abs(ka) + abs(kb), 2.0 / flat_r)
    g0 = max(gap_u, float(np.linalg.norm(pa - pb)))
    return {'g0_mm': round(g0 * mm_per_u, 4), 'g1_deg': round(g1, 3), 'kA': round(ka / cm, 4),
            'kB': round(kb / cm, 4), 'g2_dev': round(float(dev), 3), 'flat': bool(flat),
            'g1_sig_deg': round(g1s, 3), 'dk_sig': round(dks / cm, 4),
            'g1_real': bool(g1 > 3 * g1s), 'g2_real': bool(abs(ka - kb) > 3 * dks)}


def curvature_steps(prof, window_cm=1.0, min_dev=0.10, skip_ends_cm=0.0, flat_r=FLAT_R_U, mm_per_u=MM_PER_U):
    """Every place along a profile (polyline_curvature's result) where the line breaks: the one-sided fits over
    window_cm compared at each sample (on the line smoothed over its facets, about 3.5 of the voxel skin's: what a
    smooth-shaded surface shows; a step wider than the smoothing survives it), and a
    step kept where it is a local maximum, over the pass limit and over 3 sigma of the fits' own noise (so the
    mesh's facets do not read as steps). For a line with no marked join (the foot's top line): where it would show
    a crease, and how big. Returns (rows, steps, noise)."""
    cm = mm_per_u / 10.0
    s = prof['s_cm'] / cm
    X = prof['xy']                 # (smoothed over the facets: a smooth-shaded mesh does not show its facets)
    w = window_cm / cm
    indep = max(prof.get('smooth', 0.0), prof.get('spacing') or 0.0)   # (samples closer than the smoothing are one)
    rows = []
    for i, at in enumerate(s):
        if at < w + skip_ends_cm / cm or at > s[-1] - w:
            continue
        c = compare(s, X, at, w, mm_per_u=mm_per_u, flat_r=flat_r, spacing=indep)
        if c is not None:
            c['s_cm'] = round(at * cm, 2)
            rows.append(c)
    out = []
    lim = LIMITS['matte']
    # (and over 3x the line's own median: the ripple every sample of this mesh shows, so a step stands out of it)
    f2 = 3.0 * float(np.median([r['g2_dev'] for r in rows])) if rows else 0.0
    f1 = 3.0 * float(np.median([r['g1_deg'] for r in rows])) if rows else 0.0
    half = max(3, int(round(w / max(s[1] - s[0], 1e-9) / 2))) if len(s) > 1 else 3
    for j, r in enumerate(rows):
        nb = range(max(0, j - half), min(len(rows), j + half + 1))
        if r['g2_real'] and r['g2_dev'] >= max(min_dev, lim['g2_dev'][0], f2) and \
                all(r['g2_dev'] >= rows[q]['g2_dev'] for q in nb):
            out.append(dict(r, kind='curvature'))
        if r['g1_real'] and r['g1_deg'] >= max(lim['g1_deg'][0], f1) and \
                all(r['g1_deg'] >= rows[q]['g1_deg'] for q in nb):
            out.append(dict(r, kind='tangent'))
    noise = {'g1_median_deg': round(f1 / 3, 3), 'g2_median': round(f2 / 3, 3), 'g1_sig_deg_median': round(float(np.median([r['g1_sig_deg'] for r in rows])), 3) if rows else None,
             'dk_sig_median_per_cm': round(float(np.median([r['dk_sig'] for r in rows])), 4) if rows else None}
    return rows, out, noise


# ------------------------------------------------------------------ meshes

def face_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(a, 1e-20), a[:, 0] * 0.5


def vertex_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])     # (area weighted)
    N = np.zeros_like(V)
    for j in range(3):
        np.add.at(N, F[:, j], n)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-20)


def triangulate(polys):
    """Faces as vertex-index lists (any n) to triangles (fans)."""
    T = []
    for p in polys:
        for j in range(1, len(p) - 1):
            T.append((p[0], p[j], p[j + 1]))
    return np.asarray(T, int).reshape(-1, 3)


def neighbours(F, nv):
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.vstack([E, E[:, ::-1]])
    E = np.unique(E, axis=0)
    starts = np.searchsorted(E[:, 0], np.arange(nv + 1))
    return [E[starts[i]:starts[i + 1], 1] for i in range(nv)]


def principal_curvatures(V, F, idx=None, rings=2, N=None):
    """Per-vertex principal curvatures k1 >= k2 (1/u, positive convex along the outward normal), mean H and
    Gaussian K: a quadric z = a x^2 + b xy + c y^2 + d x + e y fitted over the vertex's `rings`-ring in its tangent
    frame (the shape operator's eigenvalues). idx: only these vertices."""
    V = np.asarray(V, float)
    N = vertex_normals(V, F) if N is None else N
    nb = neighbours(F, len(V))
    idx = np.arange(len(V)) if idx is None else np.asarray(idx)
    k1 = np.full(len(idx), np.nan)
    k2 = np.full(len(idx), np.nan)
    for q, i in enumerate(idx):
        ring = {int(i)}
        front = {int(i)}
        for _ in range(rings):
            front = {int(j) for f in front for j in nb[f]} - ring
            ring |= front
        ring.discard(int(i))
        if len(ring) < 6:
            continue
        n = N[i]
        a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(n, a)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        D = V[list(ring)] - V[i]
        x, y, z = D @ e1, D @ e2, D @ n
        A = np.column_stack([x * x, x * y, y * y, x, y])
        c = np.linalg.lstsq(A, z, rcond=None)[0]
        # (shape operator of the height field at the origin, with the slope (d, e) taken into account)
        E_, F_, G_ = 1 + c[3] ** 2, c[3] * c[4], 1 + c[4] ** 2
        w = math.sqrt(1 + c[3] ** 2 + c[4] ** 2)
        L, M, Nn = 2 * c[0] / w, c[1] / w, 2 * c[2] / w
        S = np.linalg.solve(np.array([[E_, F_], [F_, G_]]), np.array([[L, M], [M, Nn]]))
        ev = np.sort(np.real(np.linalg.eigvals(S)))[::-1]
        k1[q], k2[q] = -ev[1], -ev[0]           # (z along the outward normal: a convex surface bends away, negative)
    k1, k2 = np.maximum(k1, k2), np.minimum(k1, k2)
    return {'k1': k1, 'k2': k2, 'H': 0.5 * (k1 + k2), 'K': k1 * k2}


def normal_deviation(V, F, N=None, iters=3):
    """Per vertex, the angle (deg) between its normal and the normal field smoothed over its neighbourhood (`iters`
    rounds of averaging over the 1-ring): a heat map of where the surface's direction breaks from its
    surroundings (a crease, a ridge, a facet), with the gentle turning of a smooth surface taken out."""
    N = vertex_normals(V, F) if N is None else N
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    deg = np.maximum(np.bincount(E.ravel(), minlength=len(V)), 1)[:, None]
    S = N.copy()
    for _ in range(iters):
        acc = S.copy()
        np.add.at(acc, E[:, 0], S[E[:, 1]])
        np.add.at(acc, E[:, 1], S[E[:, 0]])
        S = acc / (deg + 1)
        S /= np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-20)
    return np.degrees(np.arccos(np.clip((N * S).sum(1), -1, 1)))


def plane_section(V, F, origin, normal):
    """The polylines where the plane through `origin` with `normal` cuts the mesh: a true plane-mesh intersection
    (in place of a slab of vertices, which goes jagged on steep walls), chained through shared edges."""
    V = np.asarray(V, float)
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    return level_set(V, F, (V - np.asarray(origin, float)) @ n)


def level_set(V, F, d):
    """The polylines where a per-vertex scalar d crosses zero on the mesh (a plane cut, or the line where one mesh
    passes through another when d is the signed distance to it), chained through shared edges."""
    V = np.asarray(V, float)
    d = np.where(d == 0, 1e-12, d)
    sd = np.sign(d[F])
    hit = np.where(~((sd > 0).all(1) | (sd < 0).all(1)) & ~((sd == 0).all(1)))[0]
    segs = {}
    pts = {}

    def ekey(a, b):
        return (a, b) if a < b else (b, a)
    for f in hit:
        tri = F[f]
        cut = []
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            da, db = d[a], d[b]
            if (da > 0) != (db > 0) and (da != 0 or db != 0):
                t = da / (da - db) if da != db else 0.5
                k = ekey(int(a), int(b))
                pts[k] = V[a] + t * (V[b] - V[a])
                cut.append(k)
        if len(cut) == 2:
            segs.setdefault(cut[0], []).append((cut[1], f))
            segs.setdefault(cut[1], []).append((cut[0], f))
    seen = set()
    lines = []
    for start in list(segs):
        if start in seen:
            continue
        # walk to one end first, then collect the chain from it
        a, prev = start, None
        while True:
            nxt = [e for e, _ in segs[a] if e != prev]
            if len(segs[a]) < 2 or not nxt or nxt[0] == start:
                break
            prev, a = a, nxt[0]
        chain, prev = [a], None
        seen.add(a)
        while True:
            nxt = [e for e, _ in segs[chain[-1]] if e != prev and e not in seen]
            if not nxt:
                if len(chain) > 2 and any(e == chain[0] for e, _ in segs[chain[-1]]):
                    chain.append(chain[0])
                break
            prev = chain[-1]
            chain.append(nxt[0])
            seen.add(nxt[0])
        lines.append(np.array([pts[k] for k in chain]))
    return lines


def boundary_loops(F):
    """The mesh's open boundary loops, as ordered vertex index arrays."""
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    key = np.sort(E, axis=1)
    u, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bd = E[cnt[inv.ravel()] == 1]
    nxt = {int(a): int(b) for a, b in bd}
    loops, seen = [], set()
    for a in nxt:
        if a in seen:
            continue
        loop = [a]
        seen.add(a)
        while nxt.get(loop[-1]) is not None and nxt[loop[-1]] not in seen:
            loop.append(nxt[loop[-1]])
            seen.add(loop[-1])
        loops.append(np.array(loop))
    return sorted(loops, key=len, reverse=True)


def nearest_on_mesh(V, F, P, cand=16, FN=None):
    """For each point, the nearest point on the mesh, its distance, and the signed distance (positive on the side
    the nearest face's normal points to: outside a closed, outward-facing mesh). Brute force over the `cand` faces
    with the nearest centres, in chunks (crop the mesh to the region first)."""
    V, F, P = np.asarray(V, float), np.asarray(F, int), np.asarray(P, float)
    FN = face_normals(V, F)[0] if FN is None else FN
    C = V[F].mean(1)
    cand = min(cand, len(F))
    Q = np.empty_like(P)
    dist = np.empty(len(P))
    sd = np.empty(len(P))
    for a in range(0, len(P), 512):
        blk = P[a:a + 512]
        D = ((blk[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        near = np.argpartition(D, cand - 1, axis=1)[:, :cand]
        for r, p in enumerate(blk):
            best, bd, bf = None, np.inf, -1
            for f in near[r]:
                q = _closest_tri(p, *V[F[f]])
                dd = float(np.linalg.norm(q - p))
                if dd < bd:
                    best, bd, bf = q, dd, f
            i = a + r
            Q[i], dist[i] = best, bd
            sd[i] = bd if (p - best) @ FN[bf] >= 0 else -bd
    return Q, dist, sd


def _closest_tri(p, a, b, c):
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + d1 / (d1 - d3) * ab
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + d2 / (d2 - d6) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return b + (d4 - d3) / ((d4 - d3) + (d5 - d6)) * (c - b)
    den = 1.0 / (va + vb + vc)
    return a + ab * vb * den + ac * vc * den


# ------------------------------------------------------------------ the join check

def intersection_seam(A, B, near=None, radius=None):
    """Where mesh A passes into mesh B (a head into its neck, an ear into the head, hair roots into the scalp): the
    zero line of A's signed distance to B, the longest piece (or the one nearest `near`)."""
    VA, FA = A
    VB, FB = B
    if near is not None and radius:
        VA, FA, _ = region(VA, FA, np.atleast_2d(near), radius)
    lo, hi = VA.min(0) - 0.5, VA.max(0) + 0.5
    keep = ((VB >= lo) & (VB <= hi)).all(1)
    fm = keep[FB].all(1)
    if not fm.any():
        return None
    _, _, sd = nearest_on_mesh(VB, FB[fm], VA)
    lines = level_set(VA, FA, sd)
    if not lines:
        return None
    if near is not None:
        return min(lines, key=lambda L: np.linalg.norm(L - np.asarray(near), axis=1).min())
    return max(lines, key=len)


def _visible(L, other, p):
    """The part of section L that lies outside mesh `other`, oriented to end where L enters it nearest p."""
    _, _, sd = nearest_on_mesh(*other, L)
    out = sd > 0
    ch = np.nonzero(out[:-1] != out[1:])[0]
    if not len(ch):
        return None
    i = ch[np.argmin(np.linalg.norm(L[ch] - p, axis=1))]
    t = sd[i] / (sd[i] - sd[i + 1])
    x = L[i] + t * (L[i + 1] - L[i])
    if out[i]:                     # (outside before the crossing: run up to it)
        j = i
        while j > 0 and out[j - 1]:
            j -= 1
        return np.vstack([L[j:i + 1], x])
    j = i + 1
    while j < len(L) - 1 and out[j + 1]:
        j += 1
    return np.vstack([x, L[i + 1:j + 1]])[::-1]

def _seam_frame(S, closed):
    T = np.gradient(S, axis=0)
    if closed and len(S) > 2:
        T[0] = S[1] - S[-1]
        T[-1] = S[0] - S[-2]
    return T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)


def _pick(lines, p, radius):
    best, bd = None, np.inf
    for L in lines:
        dd = np.linalg.norm(L - p, axis=1).min()
        if dd < bd:
            best, bd = L, dd
    return best if bd < radius else None


def _cut_at(L, p):
    """Arc length along L of the point of L nearest p (on its segments)."""
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(L, axis=0), axis=1))])
    best, bs = np.inf, 0.0
    for i in range(len(L) - 1):
        a, b = L[i], L[i + 1]
        ab = b - a
        t = np.clip((p - a) @ ab / max(ab @ ab, 1e-20), 0, 1)
        dd = np.linalg.norm(a + t * ab - p)
        if dd < best:
            best, bs = dd, s[i] + t * np.linalg.norm(ab)
    return s, bs, best


def check_join(name, A, seam, B=None, kind='matte', stations=24, window=None, mm_per_u=MM_PER_U, closed=None,
               solved=False, band=None, curv=True, mode=None):
    """One join. A = (V, F) (and B = (V, F) when the join is between two meshes). seam: (n,3) points along the join,
    in order (a seam loop, a boundary, a cut). mode with B: 'intersect' (A passes into B: only the visible part of
    each is measured, and the join is the crease where they meet) or 'boundary' (A's open edge meets B's).
    Returns (report dict: summary, verdict, cause, worst stations; the station rows)."""
    mode = mode or ('intersect' if B is not None else 'seam')
    VA, FA = np.asarray(A[0], float), np.asarray(A[1], int)
    VB, FB = (np.asarray(B[0], float), np.asarray(B[1], int)) if B is not None else (None, None)
    seam = np.asarray(seam, float)
    if closed is None:
        closed = len(seam) > 3 and np.linalg.norm(seam[0] - seam[-1]) < 2 * np.median(
            np.linalg.norm(np.diff(seam, axis=0), axis=1))
    edge = float(np.median(np.linalg.norm(VA[FA[:, 0]] - VA[FA[:, 1]], axis=1)))
    window = window or max(6 * edge, 0.25)
    VA, FA, _ = region(VA, FA, seam, 4 * window)          # (only the surface round the join is measured)
    if B is not None:
        VB, FB, _ = region(VB, FB, seam, 4 * window)
    rep_mode = mode
    lim = LIMITS[kind]
    Sr, s_seam = resample(np.vstack([seam, seam[:1]]) if closed else seam, 1e-3)
    pick = np.linspace(0, len(Sr) - 1, stations + (0 if closed else 0), endpoint=not closed).astype(int)
    S = Sr[pick]
    T = _seam_frame(Sr, closed)[pick]
    rows = []
    for i, (p, t) in enumerate(zip(S, T)):
        LA = _pick(plane_section(VA, FA, p, t), p, 3 * window)
        if LA is None:
            continue
        if B is not None:
            LB = _pick(plane_section(VB, FB, p, t), p, 3 * window)
            if LB is None:
                continue
            if mode == 'intersect':
                la, lb = _visible(LA, (VB, FB), p), _visible(LB, (VA, FA), p)
                if la is None or lb is None:
                    continue
                lb = lb[::-1]
            else:
                # (the two sections joined into one curve at their nearest ends: A runs into the join, B out of it)
                ia = np.argmin(np.linalg.norm(LA[[0, -1]] - p, axis=1))
                ib = np.argmin(np.linalg.norm(LB[[0, -1]] - p, axis=1))
                la = LA if ia == 1 else LA[::-1]
                lb = LB if ib == 0 else LB[::-1]
            gap = float(np.linalg.norm(la[-1] - lb[0]))
            L = np.vstack([la, lb])
            s, at = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(L, axis=0), axis=1))]), None
            at = s[len(la) - 1]
        else:
            gap = 0.0
            s, at, off = _cut_at(LA, p)
            L = LA
        X, fr = to_plane(L, t, origin=p)
        seg = np.diff(s)
        sp = float(np.median(seg[seg > 1e-9])) if (seg > 1e-9).any() else None
        r = compare(s, X, at, window, gap_u=gap, mm_per_u=mm_per_u, spacing=sp)
        if r is None:
            continue
        ctl = [compare(s, X, at + sg * 1.5 * window, window, mm_per_u=mm_per_u, spacing=sp) for sg in (-1, 1)]
        ctl = [c for c in ctl if c is not None]
        # (the control's steps count only past their own noise, as the join's do)
        r['ctl_g1_deg'] = round(max([c['g1_deg'] if c['g1_real'] else 0.0 for c in ctl], default=float('nan')), 3)
        r['ctl_g2_dev'] = round(max([c['g2_dev'] if c['g2_real'] else 0.0 for c in ctl], default=float('nan')), 3)
        r['station'] = i
        r['p'] = [round(float(v), 4) for v in p]
        r['_section'] = (X, s, at, fr)
        r['_win'] = window
        rows.append(r)
    rep = {'name': name, 'kind': kind, 'mode': rep_mode, 'stations': len(rows), 'window_cm': round(window * mm_per_u / 10, 2),
           'edge_mm': round(edge * mm_per_u, 2), 'limits': lim, 'class_a': CLASS_A, 'sources': SOURCES,
           'mm_per_u': mm_per_u}
    if not rows:
        rep.update(verdict='skipped', why='no section crossed the join')
        return rep, rows
    g0 = np.array([r['g0_mm'] for r in rows])
    g1 = np.array([r['g1_deg'] if r['g1_real'] else 0.0 for r in rows])       # (past the fits' noise only)
    g2 = np.array([r['g2_dev'] if r['g2_real'] else 0.0 for r in rows])
    c1 = np.array([r['ctl_g1_deg'] for r in rows])
    c2 = np.array([r['ctl_g2_dev'] for r in rows])

    def grade(v, lo, hi):
        return 'fail' if v > hi else ('warn' if v > lo else 'pass')
    for r in rows:
        # (two meshes that pass into each other meet by construction: their G0 is the sampling of the crossing,
        # not a gap, and is reported but not graded)
        g0g = 'pass' if mode == 'intersect' else grade(r['g0_mm'], *lim['g0_mm'])
        # (a step within 3 sigma of the fits' noise is the mesh's facets: reported, not graded)
        r['grade'] = max((g0g, grade(r['g1_deg'] if r['g1_real'] else 0.0, *lim['g1_deg']),
                          grade(r['g2_dev'] if r['g2_real'] else 0.0, *lim['g2_dev'])), key=['pass', 'warn', 'fail'].index)
    summ = {'g0_mm_max': round(float(g0.max()), 4), 'g1_deg_max': round(float(g1.max()), 3),
            'g1_deg_p90': round(float(np.percentile(g1, 90)), 3), 'g2_dev_max': round(float(g2.max()), 3),
            'g2_dev_p90': round(float(np.percentile(g2, 90)), 3),
            'ctl_g1_deg_p90': round(float(np.nanpercentile(c1, 90)), 3) if np.isfinite(c1).any() else None,
            'ctl_g2_dev_p90': round(float(np.nanpercentile(c2, 90)), 3) if np.isfinite(c2).any() else None,
            'stations_failing': int(sum(r['grade'] == 'fail' for r in rows)),
            'stations_warning': int(sum(r['grade'] == 'warn' for r in rows))}
    verdict = 'fail' if summ['stations_failing'] else ('warn' if summ['stations_warning'] else 'pass')
    # where the join falls on a curvature step, and whose it is
    cause = []
    if mode != 'intersect' and summ['g0_mm_max'] > lim['g0_mm'][0]:
        cause.append('G0: the parts do not meet (a gap of %.3f mm): modelling, how the join is welded or placed'
                     % summ['g0_mm_max'])
    for key, lo, cp, lab in (('g1', lim['g1_deg'][0], summ['ctl_g1_deg_p90'], 'tangent'),
                             ('g2', lim['g2_dev'][0], summ['ctl_g2_dev_p90'], 'curvature')):
        v = summ[key + ('_deg_p90' if key == 'g1' else '_dev_p90')]
        if v <= lo:
            continue
        if cp is None or v > 2.0 * max(cp, 1e-6):
            cause.append(f'{lab} step at the join ({v} vs {cp} away from it): the join itself, modelling (the two '
                         'sides are shaped apart; blend them over a wider zone or build them as one surface)')
        elif solved:
            cause.append(f'{lab} as rough away from the join ({cp}) as at it ({v}): the solver (settle, '
                         'resolution or bend stiffness), not the join')
        else:
            cause.append(f'{lab} as rough away from the join ({cp}) as at it ({v}): tuning (smoothing or mesh '
                         'resolution over the whole area), not the join')
    rep.update(summary=summ, verdict=verdict, cause=cause,
               worst=[{k: v for k, v in r.items() if not k.startswith('_')}
                      for r in sorted(rows, key=lambda r: -(r['g2_dev'] or 0) - r['g1_deg'] / 4)[:5]])
    if curv:
        idx = _band(VA, seam, band or 2 * window)
        if len(idx):
            pc = principal_curvatures(VA, FA, idx)
            cm = mm_per_u / 10
            rep['surface_A'] = {k: round(float(np.nanmedian(v)) / cm, 4) for k, v in pc.items() if k != 'K'}
            rep['surface_A']['K'] = round(float(np.nanmedian(pc['K'])) / cm ** 2, 5)
        if B is not None:
            idx = _band(VB, seam, band or 2 * window)
            if len(idx):
                pc = principal_curvatures(VB, FB, idx)
                cm = mm_per_u / 10
                rep['surface_B'] = {k: round(float(np.nanmedian(v)) / cm, 4) for k, v in pc.items() if k != 'K'}
                rep['surface_B']['K'] = round(float(np.nanmedian(pc['K'])) / cm ** 2, 5)
    return rep, rows


def _band(V, seam, w):
    d = np.full(len(V), np.inf)
    for p in seam[:: max(1, len(seam) // 64)]:
        d = np.minimum(d, np.linalg.norm(V - p, axis=1))
    return np.where(d < w)[0]


def region(V, F, seam, radius):
    """The faces (and their vertices, reindexed) within `radius` of the seam."""
    keep = np.zeros(len(V), bool)
    keep[_band(V, np.asarray(seam, float), radius)] = True
    fm = keep[F].all(1)
    used = np.unique(F[fm])
    remap = -np.ones(len(V), int)
    remap[used] = np.arange(len(used))
    return V[used], remap[F[fm]], used


def write(rep, rows, out, prefix=None, A=None, B=None, seam=None):
    """The report (json) and the contour sheet (png) of one join."""
    os.makedirs(out, exist_ok=True)
    prefix = prefix or 'join-' + rep['name']
    clean = dict(rep)
    clean['rows'] = [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows]
    with open(os.path.join(out, prefix + '.json'), 'w') as f:
        json.dump(clean, f, indent=1)
    if rows:
        import join_sheet
        w = rep['window_cm'] * 10.0 / rep['mm_per_u']
        cut = lambda M: None if M is None else region(np.asarray(M[0], float), np.asarray(M[1], int), seam, 3 * w)[:2]  # noqa: E731
        join_sheet.sheet(rep, rows, os.path.join(out, prefix + '.png'), A=cut(A), B=cut(B), seam=np.asarray(seam))
    return os.path.join(out, prefix + '.json')
