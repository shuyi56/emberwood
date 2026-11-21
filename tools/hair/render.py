"""Hair rebuild, foundation 7 (shading from the fibre) and the sample renders: children grown between the guides,
drawn as real-width fibres, shaded by a Marschner-type model with colour from melanin, over the clay head and body.

    python3 tools/hair/render.py HEAD.json.gz GROOM_DIR OUT_DIR [--children 24000] [--size 900]

Writes clay and shaded turnarounds (front, 3/4, side, back 3/4, back), close-ups of the parting and the hairline, and
the contour sheet (sections at several heights, front and side silhouettes) with its numbers in contours.json.

Sources:
  [MJC03] Marschner et al. 2003, "Light scattering from human hair fibers": three lobes R (surface), TT (through),
         TRT (through and back), each shifted along the hair by the cuticle's scale tilt and widened by roughness.
  [PBR4]  Pharr, Jakob & Humphreys, Physically Based Rendering, 4th ed., 9.9 "Scattering from hair" (the Chiang et
         al. 2016 model): index of refraction 1.55; scale tilt alpha about 2 deg; longitudinal variance
         v0 = (0.726 b + 0.812 b^2 + 3.7 b^20)^2 with v1 = v0 / 4 (TT) and v2 = 4 v0 (TRT); absorption from the two
         melanins, sigma_a = ce (0.419, 0.697, 1.37) + cp (0.187, 0.4, 1.05) per diameter; eumelanin about 1.3 for
         brown hair.
  Simplified here (DRAFT, not a full Marschner evaluation): Gaussian longitudinal lobes, the azimuthal parts as
  smooth weights (R toward the mirror, TT toward the back light, TRT broad), Fresnel at normal incidence, straight
  path lengths of 2 and 4 radii through the fibre for TT and TRT, a multiple-scattering fill and self-shadowing by
  counting the hair between each point and the light (deep opacity).
  [ZINKE08] Zinke, Yuksel, Weber & Keyser 2008, "Dual scattering approximation for fast multiple scattering in
         hair": light through a hair mass is dominated by forward scattering, so the multiply scattered part is shadowed
         far more softly than the single-scattered part.
  Coverage: each drawn child stands for (hairs / children) real hairs of 80 um; its alpha in a pixel is the share of
  the pixel those hairs cover, so the hair's density on screen comes from the budget, not a slider.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalp_field as SF  # noqa: E402
from groom import SPLIT  # noqa: E402

ETA = 1.55                         # [PBR4]
ALPHA_DEG = 2.0                    # [PBR4] cuticle scale tilt
BETA_M = 0.3                       # [PBR4] longitudinal roughness (mid of the range shown; EXPERIMENTAL choice)
EU_SIG = np.array([0.419, 0.697, 1.37])      # [PBR4] eumelanin absorption per diameter
PHEO_SIG = np.array([0.187, 0.4, 1.05])      # [PBR4] pheomelanin
CE, CP = 1.3, 0.6                  # [PBR4] eumelanin about 1.3 for brown; CP set warmer toward the Ranger's tone (#7e4c34)
F0 = ((ETA - 1) / (ETA + 1)) ** 2  # Fresnel at normal incidence
CLUMP = 0.55                       # the creator's clump profile: children gather to 55% of the way to their guide at
                                   # the tip (hair-building-guide.md, "Drape" 5), EXPERIMENTAL
HEAD_CLOUD = None
FRONT_BOOST = 4.0                  # children per hair at the front against the back (drawing density only)
LIGHT0 = np.array([-0.45, -0.55, 0.70])     # key light for the front view: front, high, from the character's right
LIGHT = LIGHT0.copy()                       # turntable: the light turns with the camera (set by each Canvas), so every
                                            # view is lit the same way and the back is judged lit, not in silhouette
FILL_SRGB = (0x7e, 0x4c, 0x34)              # the Ranger's hairTone in the Face Creator (main.js): the hair's colour in the
                                            # game comes only from the creator, so the multiple-scattering albedo is set
                                            # to it; the melanin model still drives the single-scattering lobes
MS_SOFT = 0.2                               # [ZINKE08] dual scattering: light scattered forward through the mass is cut
                                            # far less than the single-scattering shadow; this share of the deep-opacity
                                            # density dims the fill (EXPERIMENTAL)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def grow_children(F, Gm, n, seed=5):
    """Children: roots spread by the scalp's density; each path the root-space blend of its three nearest guides,
    gathered toward its nearest guide along the length (clumping), its length its nearest guide's within a 10% cut
    jitter (a layered cut's soft ends)."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    per = SF.point_area(F)
    p = F['density'] * per
    p = p / p.sum()
    # more, finer children where the face is framed: the front locks and the bangs are drawn with FRONT_BOOST times
    # the children per hair, each carrying that much less hair, so the coverage stays the budget's
    b = np.ones(len(p))
    if 'lock' in Gm:
        gl = Gm['lock'].astype(int)
        near = gl[cKDTree(Gm['roots']).query(F['P'])[1]]
        b[near >= 1] = FRONT_BOOST
    q = p * b / (p * b).sum()
    idx = rng.choice(len(p), size=n, p=q)
    hairs = (p[idx] / q[idx]) / n                    # share of all the hairs each child carries (sums to about 1)
    root = F['P'][idx] + (rng.random((n, 3)) - 0.5) * 0.004
    R, X = Gm['roots'], Gm['X']
    # blend only guides of the same lock: the face-framing hair (in front of the split) on each side of the part, and
    # the hair tucked behind the ears. A child blended across the split hangs halfway, in front of the neck.
    def lock(Q):
        th = SF.theta(Q)
        return np.where(np.abs(th) < SPLIT, np.where(Q[:, 0] < SF.AXIS[0], 1, 2), 0)
    gl, cl = lock(R), lock(root)
    if 'lock' in Gm:                      # the groom's own locks (with the bangs): a child takes its nearest guide's
        gl = Gm['lock'].astype(int)
        if 'clump' in Gm and Gm['clump'].max() > 0:      # the wave locks: a child blended across two waves' phases
            gl = gl * 1000 + Gm['clump'].astype(int)    # would hang half as wavy, so it blends within its own
        cl = gl[cKDTree(R).query(root)[1]]
    d = np.zeros((n, 3)); nb = np.zeros((n, 3), int)
    for g in np.unique(gl):
        gi = np.where(gl == g)[0]; ci = np.where(cl == g)[0]
        if len(ci):
            kq = min(3, len(gi))                 # (a lock of one or two guides: its nearest repeated)
            dd, kk = cKDTree(R[gi]).query(root[ci], k=kq)
            dd, kk = np.atleast_2d(dd.T).T if kq == 1 else dd, np.atleast_2d(kk.T).T if kq == 1 else kk
            d[ci], nb[ci] = dd[:, [0, min(1, kq - 1), kq - 1]], gi[kk[:, [0, min(1, kq - 1), kq - 1]]]
    w = 1.0 / np.maximum(d, 1e-4) ** 2
    w /= w.sum(1, keepdims=True)
    G, S1 = X.shape[:2]
    Xf = np.where(np.isnan(X), np.nan, X)
    # fill each guide's cut tail with its last node so the blend is defined; the child's own length then cuts it
    last = np.array([np.max(np.where(~np.isnan(X[g, :, 0]))[0]) for g in range(G)])
    for g in range(G):
        Xf[g, last[g] + 1:] = X[g, last[g]]
    rel = Xf - R[:, None]
    path = root[:, None] + (rel[nb] * w[:, :, None, None]).sum(1)
    near = nb[:, 0]
    t = np.linspace(0, 1, S1)[None, :, None]
    gpath = Xf[near] - R[near][:, None] + root[:, None]       # the nearest guide carried to this root
    path = path + (Xf[near] - gpath) * 0                       # (root-space offset kept)
    path = path + (Xf[near] - path) * CLUMP * t ** 1.5           # clumping toward the guide
    L = last[near] * (0.9 + 0.1 * rng.random(n))
    keep = np.arange(S1)[None] <= L[:, None]
    path[~keep] = np.nan
    return path, hairs


def view_basis(az_deg):
    """Orthographic camera turned az degrees round the head from the front (0 front, 90 the character's right side)."""
    a = math.radians(az_deg)
    fwd = np.array([math.sin(a) * -1, math.cos(a) * 1, 0.0]) * -1          # from the camera toward the head
    fwd = np.array([-math.sin(a), math.cos(a), 0.0])
    right = np.array([math.cos(a), math.sin(a), 0.0])
    up = np.array([0, 0, 1.0])
    return right, up, fwd


def body_points(body, da=8, dzs=5):
    """The mannequin's skin from its radius table, interpolated between the table's angles and rows; the first row is
    carried up to z 0.2 to close the neck under the head."""
    na = body['na']
    r = np.array(body['r'], float).reshape(-1, na) * 0.001
    nz = r.shape[0]
    zs = np.r_[np.linspace(0.2, body['z0'], 6, endpoint=False), body['z0'] - np.arange(0, nz - 1, 1.0 / dzs) * body['dz']]
    zs = zs[zs > -1.9]
    th = -np.pi + np.arange(na * da) * 2 * np.pi / (na * da)
    fa = (th + np.pi) / (2 * np.pi) * na
    a0 = np.floor(fa).astype(int) % na; u = fa - np.floor(fa); a1 = (a0 + 1) % na
    P, N = [], []
    for z in zs:
        fz = np.clip((body['z0'] - z) / body['dz'], 0, nz - 1.001); r0 = int(fz); v = fz - r0
        rr = (r[r0, a0] * (1 - u) + r[r0, a1] * u) * (1 - v) + (r[r0 + 1, a0] * (1 - u) + r[r0 + 1, a1] * u) * v
        P.append(np.c_[SF.AXIS[0] + rr * np.sin(th), SF.AXIS[1] - rr * np.cos(th), np.full_like(th, z)])
        N.append(np.c_[np.sin(th), -np.cos(th), np.zeros_like(th)])
    P = np.concatenate(P); N = np.concatenate(N)
    # normals from the surface: the outward horizontal is fine for the trunk; the shoulders lean up
    return P, N


class Canvas:
    def __init__(self, size, az, box=(-1.15, 1.15, -1.75, 1.2)):
        self.right, self.up, self.fwd = view_basis(az)
        global LIGHT
        a = math.radians(az)
        LIGHT = np.array([LIGHT0[0] * math.cos(a) - LIGHT0[1] * math.sin(a),
                          LIGHT0[0] * math.sin(a) + LIGHT0[1] * math.cos(a), LIGHT0[2]])
        self.box = box
        self.W = size
        self.H = int(size * (box[3] - box[2]) / (box[1] - box[0]))
        self.px = (box[1] - box[0]) / size            # HH per pixel

    def project(self, X):
        u = X @ self.right; v = X @ self.up; z = X @ self.fwd
        i = ((u - self.box[0]) / self.px).astype(int)
        j = ((self.box[3] - v) / self.px).astype(int)
        ok = (i >= 0) & (i < self.W) & (j >= 0) & (j < self.H)
        return i, j, z, ok


def clay_backdrop(cv, surf, body):
    """The head and the body as grey clay (Lambert, z-buffered splats)."""
    Ph, Nh, _ = surf.sample(260000)
    m = Ph[:, 2] > 0.18
    Ph, Nh = Ph[m], Nh[m]
    # the neck and the jaw's underside from the export itself (the radius-by-direction surface only covers the head)
    if HEAD_CLOUD is not None:
        q = HEAD_CLOUD[HEAD_CLOUD[:, 2] <= 0.2]
        nq = unit(np.c_[q[:, 0] - SF.AXIS[0], q[:, 1] - SF.AXIS[1], np.zeros(len(q))])
        Ph = np.r_[Ph, q]; Nh = np.r_[Nh, nq]
    Pb, Nb = body_points(body)
    P = np.r_[Ph, Pb]; N = np.r_[Nh, Nb]
    i, j, z, ok = cv.project(P)
    L = unit(LIGHT)
    sh = 0.25 + 0.75 * np.clip(N @ L, 0, 1)
    zb = np.full((cv.H, cv.W), np.inf)
    col = np.ones((cv.H, cv.W, 3)) * np.array([0.93, 0.93, 0.95])
    for di in (-1, 0, 1, 2):
        for dj in (-1, 0, 1, 2):
            ii, jj = np.clip(i + di, 0, cv.W - 1), np.clip(j + dj, 0, cv.H - 1)
            np.minimum.at(zb, (jj[ok], ii[ok]), z[ok])
    for di in (-1, 0, 1, 2):
        for dj in (-1, 0, 1, 2):
            ii, jj = np.clip(i + di, 0, cv.W - 1), np.clip(j + dj, 0, cv.H - 1)
            win = ok & (z <= zb[jj, ii] + 1e-9)
            col[jj[win], ii[win]] = (np.array([0.72, 0.70, 0.68]) * sh[win, None])
    return col, zb


def hair_samples(paths, px):
    """Points along every child, one per pixel of length, with their tangent and their place along the hair."""
    A = paths[:, :-1]; B = paths[:, 1:]
    ok = ~(np.isnan(A[..., 0]) | np.isnan(B[..., 0]))
    A, B = A[ok], B[ok]
    seg = np.linalg.norm(B - A, axis=1)
    # one point per pixel of each segment's own length (one long segment must not set the count for all of them)
    k = np.maximum(1, np.ceil(seg / px).astype(int))
    rep = np.repeat(np.arange(len(A)), k)
    first = np.cumsum(k) - k
    f = (np.arange(len(rep)) - first[rep] + 0.5) / k[rep]
    P = A[rep] * (1 - f)[:, None] + B[rep] * f[:, None]
    T = unit(B - A)[rep]
    along = np.nonzero(ok)[1][rep] + f
    child = np.nonzero(ok)[0][rep]
    return P, T, along, child


def shade(P, T, along, view, mode, shadow):
    """mode 'clay': Kajiya diffuse in grey, to judge the form; 'full': the fibre model with melanin colour."""
    L = unit(LIGHT)
    V = -view
    sin_i = T @ L; sin_r = T @ V
    th_i = np.arcsin(np.clip(sin_i, -1, 1)); th_r = np.arcsin(np.clip(sin_r, -1, 1))
    diff = np.sqrt(np.clip(1 - sin_i ** 2, 0, 1))
    if mode == 'clay':
        c = (0.2 + 0.7 * diff * shadow ** MS_SOFT)[:, None] * np.array([0.78, 0.76, 0.74])
        return c
    th_h = 0.5 * (th_i + th_r)
    b = BETA_M
    v0 = (0.726 * b + 0.812 * b * b + 3.7 * b ** 20) ** 2
    s0 = math.sqrt(v0)
    a = math.radians(ALPHA_DEG)
    g = lambda x, s: np.exp(-0.5 * (x / s) ** 2) / (s * math.sqrt(2 * math.pi))
    M_R = g(th_h + 2 * a, s0)
    M_TT = g(th_h - a, 0.5 * s0)
    M_TRT = g(th_h - 4 * a, 2 * s0)
    # azimuth: the angle between the light and the view, in the plane across the hair
    Lp = unit(L - (T * sin_i[:, None])); Vp = unit(V - (T * sin_r[:, None]))
    cphi = np.clip((Lp * Vp).sum(1), -1, 1)
    phi = np.arccos(cphi)
    N_R = 0.25 * np.cos(0.5 * phi).clip(0)
    N_TT = np.exp(-0.5 * ((np.pi - phi) / 0.35) ** 2) * 0.6
    N_TRT = 0.15 + 0.1 * np.cos(phi)
    sig = CE * EU_SIG + CP * PHEO_SIG
    A_R = F0
    A_TT = (1 - F0) ** 2 * np.exp(-2 * sig)
    A_TRT = (1 - F0) ** 2 * F0 * np.exp(-4 * sig) * 6       # (the TRT glints' energy: caustic boost, DRAFT)
    spec = (M_R * N_R * A_R)[:, None] * np.ones(3) + (M_TT * N_TT)[:, None] * A_TT + (M_TRT * N_TRT)[:, None] * A_TRT
    fill = (np.array(FILL_SRGB) / 255.0) ** 2.2 * 1.0           # multiple scattering through the hair mass: the creator's
                                                               # tone, lifted so the lit side reaches it (DRAFT)
    ms = shadow ** MS_SOFT
    root_dark = 0.75 + 0.25 * np.clip(along / 6.0, 0, 1)         # roots darkest (hair-part-roots-research.md)
    c = (spec * (5.0 / np.maximum(np.cos(th_i), 0.2) * np.cos(th_i))[:, None] * shadow[:, None] + fill * (0.3 + 0.7 * diff * ms)[:, None]) * root_dark[:, None]
    return np.clip(c, 0, 4)


def deep_shadow(P, cv_light_px, hairs_per_child):
    """How much hair lies between each point and the light: points binned in the light's view, ranked by depth; the
    light reaching a point falls by the coverage of each hair in front of it."""
    L = unit(LIGHT)
    up = np.array([0, 0, 1.0]); r = unit(np.cross(up, L)); u2 = np.cross(L, r)
    i = np.floor((P @ r) / cv_light_px).astype(np.int64)
    j = np.floor((P @ u2) / cv_light_px).astype(np.int64)
    z = -(P @ L)                                   # smaller: nearer the light
    key = (i - i.min()) * (j.max() - j.min() + 1) + (j - j.min())
    o = np.lexsort((z, key))
    ks = key[o]
    start = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    rank = np.arange(len(o)) - np.repeat(start, np.diff(np.r_[start, len(o)]))
    cover = np.asarray(hairs_per_child, float) * np.ones(len(P)) * 80e-6 / (cv_light_px * SF.HH_MM / 1000)
    # (share of a light texel one sample blocks; each sample stands for one pixel-length of hair) summed over the
    # samples nearer the light in the same texel
    cs = np.cumsum(cover[o])
    before = cs - cover[o] - np.repeat(np.r_[0, cs][start] , np.diff(np.r_[start, len(o)]))
    acc = np.empty(len(o)); acc[o] = before
    return np.exp(-acc * 0.6)


def render_hair(cv, paths, mode, backdrop, zb, hairs_per_child, return_cov=False):
    P, T, along, child = hair_samples(paths, cv.px)
    hairs_per_child = np.asarray(hairs_per_child, float) * np.ones(len(paths))
    i, j, z, ok = cv.project(P)
    vis = ok.copy()
    vis[ok] = z[ok] < zb[j[ok], i[ok]] + 0.004
    P, T, along, i, j, z, child = P[vis], T[vis], along[vis], i[vis], j[vis], z[vis], child[vis]
    hpc = hairs_per_child[child]
    sh = deep_shadow(P, cv.px * 2, hpc)
    c = shade(P, T, along, cv.fwd, mode, sh)
    # coverage of a pixel by one child's sample: hairs_per_child fibres 80 um wide over a pixel's width
    a1 = 1 - (1 - min(1.0, 80e-6 / (cv.px * SF.HH_MM / 1000))) ** hpc
    zmin = np.full((cv.H, cv.W), np.inf)
    np.minimum.at(zmin, (j, i), z)
    w = np.exp(-(z - zmin[j, i]) / 0.02) * a1            # weighted blended OIT: front hair dominates (McGuire 2013)
    acc = np.zeros((cv.H, cv.W, 3)); wsum = np.zeros((cv.H, cv.W)); logT = np.zeros((cv.H, cv.W))
    for k in range(3):
        np.add.at(acc[..., k], (j, i), c[:, k] * w)
    np.add.at(wsum, (j, i), w)
    np.add.at(logT, (j, i), np.log(1 - a1) * np.ones_like(w))      # (a1 per sample)
    Tr = np.exp(logT)[..., None]
    avg = acc / np.maximum(wsum, 1e-12)[..., None]
    img = backdrop * Tr + avg * (1 - Tr)
    if return_cov:
        return img, 1 - Tr[..., 0]
    return img


def to_srgb(img):
    img = np.clip(img, 0, 1)
    return np.where(img <= 0.0031308, 12.92 * img, 1.055 * img ** (1 / 2.4) - 0.055)


def contours(paths, out_png, out_json):
    """Section cuts of the hair mass at several heights (outer radius by angle round the body's axis, with the smooth
    ring fitted to each), and the front and side silhouettes. Flags a section that is too round (a shell, not hair) or
    has a step (a ledge), and a silhouette that jumps between rows."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    P = paths.reshape(-1, 3); P = P[~np.isnan(P[:, 0])]
    th = np.arctan2(P[:, 0] - SF.AXIS[0], -(P[:, 1] - SF.AXIS[1]))
    r = np.hypot(P[:, 0] - SF.AXIS[0], P[:, 1] - SF.AXIS[1])
    heights = [0.95, 0.8, 0.6, 0.4, 0.1, -0.3, -0.7, -1.1, -1.35]
    nb = 72
    fig = plt.figure(figsize=(20, 9), dpi=100)
    rep = {'sections': []}
    for k, zc in enumerate(heights):
        m = np.abs(P[:, 2] - zc) < 0.012
        ax = fig.add_subplot(2, 7, k + 1, projection='polar')
        if m.sum() < 50:
            ax.set_title('z %.2f: empty' % zc); continue
        b = ((th[m] + np.pi) / (2 * np.pi) * nb).astype(int) % nb
        rmax = np.full(nb, np.nan)
        for q in range(nb):
            s = r[m][b == q]
            if len(s):
                rmax[q] = np.percentile(s, 98)
        a = np.linspace(-np.pi, np.pi, nb, endpoint=False) + np.pi / nb
        ok = ~np.isnan(rmax)
        # the smooth ring: the first 4 Fourier terms of the outline
        A = np.c_[np.ones(ok.sum()), *[f(n * a[ok]) for n in range(1, 5) for f in (np.cos, np.sin)]]
        cfit, *_ = np.linalg.lstsq(A, rmax[ok], rcond=None)
        ring = A @ cfit
        dev = rmax[ok] - ring
        jumps = np.abs(np.diff(rmax[ok]))
        ax.scatter(-a[ok] + np.pi / 2, rmax[ok], s=6, c='#5a3020')
        ax.plot(-a[ok] + np.pi / 2, ring, c='#2b6cb0', lw=0.8)
        ax.set_title('z %.2f' % zc, fontsize=9); ax.set_yticklabels([])
        roundness = float(np.std(rmax[ok]) / np.mean(rmax[ok]))
        rep['sections'].append(dict(z=zc, coverage=round(float(ok.mean()), 2), mean_r=round(float(rmax[ok].mean()), 3),
                                    roundness=round(roundness, 3), dev_rms_mm=round(float(np.sqrt((dev ** 2).mean())) * SF.HH_MM, 1),
                                    max_step_mm=round(float(jumps.max()) * SF.HH_MM, 1) if len(jumps) else None))
    for col, (nm, ax_i) in enumerate((('front silhouette', 0), ('side silhouette', 1))):
        ax = fig.add_subplot(2, 7, 13 + col if col == 0 else 14)
        zs = np.arange(-1.5, 1.05, 0.02)
        lo, hi = [], []
        for zc in zs:
            m = np.abs(P[:, 2] - zc) < 0.01
            if m.sum() > 20:
                q = P[m, ax_i]; lo.append(np.percentile(q, 1)); hi.append(np.percentile(q, 99))
            else:
                lo.append(np.nan); hi.append(np.nan)
        lo, hi = np.array(lo), np.array(hi)
        ax.plot(lo, zs, c='#5a3020'); ax.plot(hi, zs, c='#5a3020'); ax.set_aspect('equal'); ax.set_title(nm, fontsize=9)
        step = np.nanmax(np.abs(np.diff(np.r_[lo, hi].reshape(2, -1), axis=1)))
        rep[nm.replace(' ', '_') + '_max_row_step_mm'] = round(float(step) * SF.HH_MM, 1)
    fig.suptitle('Contour sheet: sections (outer 98th percentile radius by angle, front at the top) with the fitted '
                 'smooth ring, and silhouettes')
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    json.dump(rep, open(out_json, 'w'), indent=1)
    return rep


def main():
    import gzip
    a = sys.argv[1:]
    head, gdir, out = a[0], a[1], a[2]
    nch = int(a[a.index('--children') + 1]) if '--children' in a else 24000
    size = int(a[a.index('--size') + 1]) if '--size' in a else 900
    os.makedirs(out, exist_ok=True)
    global HEAD_CLOUD, FILL_SRGB, CE, CP
    if '--tone' in a:                      # the character's hairTone in the creator, e.g. 36251c (the Warden)
        h = a[a.index('--tone') + 1].lstrip('#')
        FILL_SRGB = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    if '--melanin' in a:                   # eumelanin, pheomelanin [PBR4], e.g. 2.4,0.4 for a dark brown
        CE, CP = (float(v) for v in a[a.index('--melanin') + 1].split(','))
    HEAD_CLOUD = SF.load_head(head)
    surf = SF.Surface(HEAD_CLOUD)
    F = SF.build(surf)
    body = json.load(gzip.open(head))['bodies']['mannequin']
    Gm = dict(np.load(os.path.join(gdir, 'groom.npz')))
    paths, share = grow_children(F, Gm, nch)
    total = float(Gm['total']) if 'total' in Gm else float(Gm['n_hairs'].sum())
    hpc = total * share                    # hairs each child carries
    np.savez_compressed(os.path.join(out, 'children.npz'), paths=paths.astype(np.float32))
    from PIL import Image
    rep = {'children': nch, 'hairs_per_child': round(float(hpc.mean()), 2),
           'hairs_per_child_front': round(float(hpc.min()), 2)}
    for mode in ('clay', 'full'):
        row = []
        for az in (0, 45, 90, 135, 180):
            cv = Canvas(size, az)
            bg, zb = clay_backdrop(cv, surf, body)
            img = render_hair(cv, paths, mode, bg, zb, hpc)
            row.append((to_srgb(img) * 255).astype(np.uint8))
        Image.fromarray(np.concatenate(row, 1)).save(os.path.join(out, 'turn-%s.png' % mode))
    # close-ups: the parting and front hairline from above-front, the side over the ear
    for name, az, box in (('closeup-part', 0, (-0.35, 0.35, 0.55, 1.08)), ('closeup-ear', 90, (-0.45, 0.45, 0.2, 1.08)),
                          ('closeup-back', 180, (-0.55, 0.55, -0.6, 1.08))):
        imgs = []
        for mode in ('clay', 'full'):
            cv = Canvas(size, az, box)
            bg, zb = clay_backdrop(cv, surf, body)
            imgs.append((to_srgb(render_hair(cv, paths, mode, bg, zb, hpc)) * 255).astype(np.uint8))
        Image.fromarray(np.concatenate(imgs, 1)).save(os.path.join(out, name + '.png'))
    rep['contours'] = contours(paths, os.path.join(out, 'contours.png'), os.path.join(out, 'contours.json'))
    json.dump(rep, open(os.path.join(out, 'render.json'), 'w'), indent=1)
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
