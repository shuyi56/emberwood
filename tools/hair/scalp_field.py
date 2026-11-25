"""The scalp (hair rebuild, foundations 1 and 2): where hair grows on the creator's head, how densely, which way it
points as it leaves the skin, and at what angle. Every hairstyle is built on this field. A style is an operation on
the natural fall (part, comb, settle, cut), not a hand-placed set of guide curves.

    python3 tools/hair/scalp_field.py HEAD.json.gz OUT_DIR        (HEAD: the creator's head export, rq_hair_export.cjs)

Writes OUT_DIR/scalp_field.npz (points, normals, mask, density, growth direction, exit angle, region weights),
OUT_DIR/scalp_field.json (the measured landmarks and totals) and OUT_DIR/scalp_field_sheet.png.

Units: head heights (HH), chin z 0 to crown z 1, face toward -y. One HH is 219 mm (NOSE_LIFE.head_h in the creator).
The character's right is -x (facing -y with z up).

Sources (every constant names one; EXPERIMENTAL marks a value with no measurement behind it yet):
  [LS20] Leerunyakul & Suchonwanit 2020, BioMed Res Int, PMC7035527. Trichoscopy of 160 adult Thai women.
         Hairs/cm2: frontal 153.9, vertex 162.6, temporoparietal 133.2, occipital 160.1. Shaft about 80 um at every
         site. Sites at 12, 24 and 30 cm from the glabella along the midline part, and 6 cm from the ear canal.
  [KS16] Karaca Saygili et al. 2016, Folia Morphol 75(4):512-517. 100 women: tr-n 66.93 mm, tr-g 51.29 mm
         (straight calliper distances), solved here along this head's forehead slope.
  [W25]  Wu et al. 2025, J Cosmet Dermatol, PMC12319885. Female hairline: the temporal point is where the line from
         the nasal tip through the pupil crosses the line from the earlobe to the frontal hairline's midpoint. A
         transition zone about 2 cm wide behind the leading edge (1-hair units, then 1- and 2-hair). Exit angle
         10-15 deg at the lateral hairline, 15-25 at the frontal hairline.
  [SDK08] Schmidt, Depner & Kabesch 2008, Sci World J 8:848, PMC5848856. 519 children with one parietal whorl:
         right-sided 70.8%, clockwise 83.9%.
  [P23]  Park et al. 2023, Clin Cosmet Investig Dermatol 16:1885, PMC10349607 (citing Unger 2004, Dermatol Surg
         30:829): exit angle 10-20 deg along the anterior hairline, 30-40 deg mid-scalp.
  [K25]  Kim et al. 2025, J Craniofac Surg, PMC12502931. 125 East Asian men: the hairline hair skews right, 55.0 deg
         on the right against 32.8 on the left (22.2 apart), following the clockwise whorl. Male data, used as the
         target until a female measurement turns up.
"""
import gzip
import json
import math
import os
import sys

import numpy as np

HH_MM = 219.0                      # mm per head height (the creator's NOSE_LIFE.head_h)
MM = 1.0 / HH_MM
CENTRE = np.array([0.0, 0.06, 0.56])   # the skull's centre, as the creator places it
AXIS = (0.0, 0.04)                     # (x, y) of the head's vertical axis (the creator's ear clearance)

# ---- the creator's face and ear, read off main.js (the face is built from these, so the hairline follows them)
PUPIL = (0.14, 0.525 + (-0.05 + 0.11) / 8.6)   # (x, z): eyeSpacing 0.14 HH; 0.525 HH + EYE_LIFT + lift, model units
EAR_TOP, EAR_BOTTOM = 0.60, 0.365              # EAR.TOP / EAR.BOTTOM (the lobe's lowest point)
EAR_CANAL = (0.47, 0.04)                       # (z, y): the ear's middle height at its depth EY. APPROXIMATE

# ---- the hairline (foundation 1)
TRICHION_TR_N_MM, TRICHION_TR_G_MM = 66.93, 51.29   # [KS16]
TRANSITION = 20 * MM               # [W25]: the transition zone behind the leading edge, about 2 cm
EAR_CLEAR = 4 * MM                 # EXPERIMENTAL: the hairline over the top of the ear ("a few millimetres", CLAUDE.md)
NAPE_Z = 0.30                      # PROVISIONAL: the nape hairline's lowest point (no source found yet)

# ---- density [LS20], hairs/cm2 at the four sites (women)
DENSITY = {'frontal': 153.9, 'vertex': 162.6, 'temporoparietal': 133.2, 'occipital': 160.1}
SITE_ARC_MM = {'frontal': 120.0, 'vertex': 240.0, 'occipital': 300.0}   # along the midline from the glabella
TP_MM = 60.0                                                            # temporoparietal: above the ear canal
SHAFT_UM = 80.0                                                         # [LS20]

# ---- direction: a tight whorl, the fall away from it, a frontal stream and the temples
WHORL_SIDE = -1                    # [SDK08]: right-sided (the character's right is -x)
WHORL_CLOCKWISE = True             # [SDK08]: clockwise, seen from above
WHORL_OFFSET = 15 * MM             # EXPERIMENTAL: how far right of the midline (no measured distance yet)
WHORL_RADIUS = 20 * MM             # EXPERIMENTAL: the swirl's reach (a crown whorl is a few cm across)
SKEW_TARGET_DEG = 22.2             # [K25]: the left-right difference along the frontal hairline (2x a common lean)
TEMPLE_BACK_DEG = 30.0             # EXPERIMENTAL: temple hair runs down and back; the angle back from straight down
EXIT_HAIRLINE_DEG = 15.0           # [P23] 10-20, [W25] 15-25 frontal: 15 is inside both
EXIT_LATERAL_DEG = 12.5            # [W25] 10-15 at the lateral hairline
EXIT_MID_DEG = 35.0                # [P23] 30-40 mid-scalp
EXIT_RAMP = TRANSITION             # the angle rises across the same transition zone [W25]


def sm(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def load_head(path):
    d = json.load(gzip.open(path))
    return np.array(d['head']['p'], float).reshape(-1, 3)


def theta(P):
    """The angle round the head's axis: 0 at the front, + toward +x (the character's left), +-pi at the back."""
    P = np.atleast_2d(P)
    return np.arctan2(P[:, 0] - AXIS[0], -(P[:, 1] - AXIS[1]))


class Surface:
    """The head's skin as a radius from the skull's centre by direction, read off the exported points (nearest points
    by angle, weighted). The scalp can then be sampled evenly (the export is dense on the face and sparse on top), and
    the same radius is the head's collider: a point is inside the head when it is nearer the centre than the skin."""
    def __init__(self, cloud):
        from scipy.spatial import cKDTree
        q = cloud[cloud[:, 2] > 0.2] - CENTRE
        r = np.linalg.norm(q, axis=1)
        self.u, self.r = q / r[:, None], r
        self.tree = cKDTree(self.u)

    def radius(self, U):
        d, i = self.tree.query(U, k=6)
        w = 1.0 / np.maximum(d, 1e-5) ** 2
        return (self.r[i] * w).sum(1) / w.sum(1)

    def point(self, U):
        return CENTRE + U * self.radius(U)[:, None]

    def depth(self, X):
        """How far points X are inside the skin (HH, along the radius; negative outside)."""
        q = X - CENTRE
        r = np.linalg.norm(q, axis=1)
        return self.radius(q / r[:, None]) - r

    def normals(self, U, e=0.01):
        t1 = np.cross(U, [0, 0, 1.0]); bad = np.linalg.norm(t1, axis=1) < 1e-6; t1[bad] = [1.0, 0, 0]
        t1 = unit(t1); t2 = np.cross(U, t1)
        p = lambda V: self.point(unit(V))
        N = unit(np.cross(p(U + e * t1) - p(U - e * t1), p(U + e * t2) - p(U - e * t2)))
        return N * np.sign((N * U).sum(1))[:, None]

    def sample(self, n):
        """n directions spread evenly over the sphere (Fibonacci), the points on the skin and their normals."""
        k = np.arange(n) + 0.5
        z = 1 - 2 * k / n
        a = np.pi * (1 + 5 ** 0.5) * k
        U = np.c_[np.sqrt(1 - z * z) * np.cos(a), np.sqrt(1 - z * z) * np.sin(a), z]
        return self.point(U), self.normals(U), U

    def arc(self, U0, U1, n=400):
        """The path along the skin between two directions (a great circle seen from the centre), with its arc length."""
        w = np.arccos(np.clip(U0 @ U1, -1, 1))
        t = np.linspace(0, 1, n)[:, None]
        U = (np.sin((1 - t) * w) * U0 + np.sin(t * w) * U1) / np.sin(w)
        P = self.point(U)
        return P, np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]

    def side_x(self, z):
        """The skin point of greatest x at height z (the head's outline seen from the front)."""
        best = None
        for th in np.linspace(1.0, 2.2, 61):
            X = on_skin_at(self, th, z)
            if best is None or X[0] > best[0]:
                best = X
        return best


def along(path, arcs, a):
    i = int(np.clip(np.searchsorted(arcs, a), 1, len(arcs) - 1))
    t = (a - arcs[i - 1]) / max(arcs[i] - arcs[i - 1], 1e-12)
    return path[i - 1] + t * (path[i] - path[i - 1])


def on_skin_at(surf, th, z):
    """The skin point at angle th round the axis and height z (searching along the ray out from the axis)."""
    d = np.array([math.sin(th), -math.cos(th), 0.0])
    lo, hi = 0.0, 0.8
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        X = np.array([[AXIS[0], AXIS[1], z]]) + mid * d
        if surf.depth(X)[0] > 0:
            lo = mid
        else:
            hi = mid
    return np.array([AXIS[0], AXIS[1], z]) + lo * d


def landmarks(surf):
    """The measured points the hairline is built from: glabella, nasion, pronasale, trichion, temporal point."""
    # the midline profile in front, by elevation from the centre
    el = np.linspace(-0.9, 0.9, 900)
    Pm = surf.point(np.c_[np.zeros_like(el), -np.cos(el), np.sin(el)])
    g = Pm[np.argmin(Pm[:, 1] + 10 * ((Pm[:, 2] < 0.6) | (Pm[:, 2] > 0.78)))]          # brow: most forward
    n = Pm[np.argmax(Pm[:, 1] - 10 * ((Pm[:, 2] < 0.53) | (Pm[:, 2] > g[2])))]         # between the eyes: deepest
    prn = Pm[np.argmin(Pm[:, 1] + 10 * ((Pm[:, 2] < 0.3) | (Pm[:, 2] > 0.5)))]         # the nasal tip
    # trichion: the straight distances tr-n and tr-g [KS16], along the forehead above the brow; their mean
    up = Pm[(Pm[:, 2] > g[2]) & (Pm[:, 1] < 0)]
    up = up[np.argsort(up[:, 2])]
    zn = up[np.argmin(np.abs(np.linalg.norm(up - n, axis=1) - TRICHION_TR_N_MM * MM)), 2]
    zg = up[np.argmin(np.abs(np.linalg.norm(up - g, axis=1) - TRICHION_TR_G_MM * MM)), 2]
    ztr = 0.5 * (zn + zg)
    tr = up[np.argmin(np.abs(up[:, 2] - ztr))]
    # the temporal point [W25]: nasal tip -> pupil, crossed with earlobe -> frontal hairline midpoint. Applied in the
    # side view (y, z): in the front view the two lines cross at the tail of the brow, on the face, so the paper's
    # construction can only be the lateral one (its figure is an oblique photo). INTERPRETED, not stated in the text.
    lobe = surf.side_x(EAR_BOTTOM)
    # the pupil's depth: halfway between the brow and the cheek in front of it, as the creator's orbitSpec places the
    # eye between them (the head export has the sockets cut, so the eye itself is not in it)
    def front_y(x, z):
        best = None
        for th in np.linspace(0.05, 0.9, 300):
            X = on_skin_at(surf, th, z)
            if best is None or abs(X[0] - x) < abs(best[0] - x):
                best = X
        return best[1]
    pupil = np.array([PUPIL[0], 0.5 * (front_y(PUPIL[0], PUPIL[1] + 0.07) + front_y(PUPIL[0], PUPIL[1] - 0.07)), PUPIL[1]])
    A0, A1 = np.array([prn[1], prn[2]]), np.array([pupil[1], PUPIL[1]])
    B0, B1 = np.array([lobe[1], EAR_BOTTOM]), np.array([tr[1], tr[2]])
    M = np.c_[A1 - A0, -(B1 - B0)]
    t, s = np.linalg.solve(M, B0 - A0)
    tp2 = A0 + t * (A1 - A0)                       # (y, z) in the side view
    best = None
    for th in np.linspace(0.3, 1.6, 600):          # onto the side of the head at that depth and height
        X = on_skin_at(surf, th, tp2[1])
        if best is None or abs(X[1] - tp2[0]) < abs(best[1] - tp2[0]):
            best = X
    tp = best
    return dict(glabella=g, nasion=n, pronasale=prn, trichion=tr, trichion_z_from_n=zn, trichion_z_from_g=zg,
                temporal_point=tp, temporal_point_side=tp2, earlobe=lobe, pupil=pupil)


def make_hairline(L):
    """The hairline's height by angle round the head, one smooth, shape-keeping curve (PCHIP: no overshoot, flat at
    the front and the back), mirrored so it is even in th. Knots: the trichion [KS16]; a round female contour (the
    most preferred shape in PMC13597767) to the temporal point [W25]; the sideburn down in front of the ear to the
    canal's height; over the ear; behind it to the lobe; and the nape (PROVISIONAL)."""
    from scipy.interpolate import PchipInterpolator
    tr_z = L['trichion'][2]
    tp_th = float(theta(L['temporal_point'])[0]); tp_z = L['temporal_point'][2]
    ear_th = float(theta(np.array([L['earlobe']]))[0])            # the ear's angle round the head
    knots = [(0.0, tr_z), (0.5 * tp_th, tr_z - 0.25 * (tr_z - tp_z) ** 1.0 * 0.4), (0.85 * tp_th, tp_z + 0.35 * (tr_z - tp_z)),
             (tp_th, tp_z), (ear_th - 0.18, EAR_CANAL[0]), (ear_th, EAR_TOP + EAR_CLEAR),
             (ear_th + 0.25, EAR_TOP - 0.02), (ear_th + 0.5, EAR_BOTTOM + 0.03), (2.4, 0.5 * (EAR_BOTTOM + NAPE_Z) + 0.03),
             (np.pi, NAPE_Z)]
    k = np.array(knots)
    x = np.r_[-k[:0:-1, 0], k[:, 0], 2 * np.pi - k[-2::-1, 0][:2]]
    y = np.r_[k[:0:-1, 1], k[:, 1], k[-2::-1, 1][:2]]
    f = PchipInterpolator(x, y)
    return (lambda th: f(np.abs(th))), knots


def build(surf, n=40000):
    P, N, U = surf.sample(n)
    th = theta(P)
    L = landmarks(surf)
    hairline, knots = make_hairline(L)
    g = L['glabella']
    # the midline from the glabella over the top and down the back: the frontal, vertex and occipital sites [LS20]
    paths, arcs = [], [0.0]
    U0 = unit(g - CENTRE)
    for U1 in (unit(np.array([0, 0, 1.0])), unit(np.array([0, 1.0, -0.6]))):
        pp, aa = surf.arc(U0, U1)
        aa = aa + arcs[-1]
        paths.append(pp[1:] if len(paths) else pp)
        arcs = np.r_[arcs, aa[1:]] if len(paths) > 1 else aa
        U0 = U1
    path = np.concatenate(paths)
    sites = {k2: along(path, arcs, v * MM) for k2, v in SITE_ARC_MM.items()}
    # temporoparietal: 60 mm up the side of the head from the ear canal, along the skin (both sides) [LS20]
    Uc = unit(np.array([1.0, EAR_CANAL[1] - CENTRE[1], EAR_CANAL[0] - CENTRE[2]]))
    pp, aa = surf.arc(Uc, unit(np.array([0, EAR_CANAL[1] - CENTRE[1], 1.0])))
    tp = along(pp, aa, TP_MM * MM)
    sites['temporoparietal_L'] = tp
    sites['temporoparietal_R'] = tp * np.array([-1, 1, 1])

    # the scalp: the head above the hairline. behind: how far the point is below the hairline (> 0: bare skin)
    behind = hairline(th) - P[:, 2]
    on = P[:, 2] > 0.25
    mask = on & (behind < 0)
    # distance behind the leading edge along the skin, roughly: the height gap over the surface's slope there
    slope = np.sqrt(np.clip(1 - N[:, 2] ** 2, 0.05, 1))
    inside = np.maximum(-behind, 0) / slope
    edge = sm(0.0, TRANSITION, inside)

    # density: inverse-distance weighting between the sites
    names = ['frontal', 'vertex', 'occipital', 'temporoparietal_L', 'temporoparietal_R']
    vals = np.array([DENSITY['frontal'], DENSITY['vertex'], DENSITY['occipital'],
                     DENSITY['temporoparietal'], DENSITY['temporoparietal']])
    S = np.array([sites[k] for k in names])
    D = np.linalg.norm(P[:, None, :] - S[None], axis=2)
    W = 1.0 / np.maximum(D, 1e-4) ** 2
    region = W / W.sum(1, keepdims=True)
    dens = (region * vals).sum(1) * edge * mask

    # the whorl: on the vertex site's arc (EXPERIMENTAL), offset to the right [SDK08]
    wv = sites['vertex'].copy()
    wv[0] += WHORL_SIDE * WHORL_OFFSET
    whorl = surf.point(unit((wv - CENTRE)[None]))[0]

    def tangent(v):
        v = v - (v * N).sum(1, keepdims=True) * N
        return unit(v)

    def rotate(v, ang):            # about each point's normal, counterclockwise seen from outside
        ang = np.asarray(ang)[:, None]
        return unit(np.cos(ang) * v + np.sin(ang) * np.cross(N, v))

    # 1. the fall: away from the whorl along the skin, everywhere
    away = tangent(P - whorl)
    dist = np.linalg.norm(P - whorl, axis=1)
    # 2. the swirl: a turn of up to 90 deg round the whorl, gone beyond its radius (tight, as a crown whorl is)
    spin = (-1 if WHORL_CLOCKWISE else 1) * (np.pi / 2) * np.exp(-(dist / WHORL_RADIUS) ** 2)
    # 3. the temples: down and back instead of down and forward (EXPERIMENTAL angle, no regional map yet)
    down = tangent(np.tile([0, 0, -1.0], (len(P), 1)))
    back_dir = tangent(np.c_[np.zeros(len(P)), np.ones(len(P)), np.zeros(len(P))])
    a = np.abs(th)
    temple_w = sm(0.7, 1.05, a) * (1 - sm(1.5, 2.0, a)) * (1 - sm(0.70, 0.85, P[:, 2]))
    temple = unit(np.cos(np.radians(TEMPLE_BACK_DEG)) * down + np.sin(np.radians(TEMPLE_BACK_DEG)) * back_dir)
    # 4. the frontal stream: the whorl's rotation carried forward, a common lean toward the right over the front of
    # the scalp, solved so the frontal hairline skews by the measured amount [K25]
    front_w = (1 - sm(0.5, 1.1, a)) * sm(0.6, 0.8, P[:, 2]) * sm(WHORL_RADIUS, 3 * WHORL_RADIUS, dist)

    def field(lean):
        v = rotate(away, spin)
        v = unit(v * (1 - temple_w)[:, None] + temple * temple_w[:, None])
        # the lean toward the character's right (-x): seen from outside the front, that is clockwise
        return rotate(v, -lean * front_w)

    probe_th = np.array([-0.5, 0.5]) * float(theta(L['temporal_point'])[0])     # halfway to the temporal point [K25]

    def skew(v):
        out = []
        for t in probe_th:
            k = np.argmin(np.abs(th - t) + 10 * np.abs(P[:, 2] - (hairline(np.array([t]))[0] + 0.01)) + 10 * (~on))
            n = N[k]
            dn = np.array([0, -0.5, -1.0]); dn -= n * (n @ dn); dn /= np.linalg.norm(dn)
            rt = np.array([-1.0, 0, 0]); rt -= n * (n @ rt) + dn * (dn @ rt); rt /= np.linalg.norm(rt)
            out.append(math.degrees(math.atan2(float(v[k] @ rt), float(v[k] @ dn))))
        return out
    lo, hi = -0.5, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        s = skew(field(mid))
        if s[0] + s[1] < SKEW_TARGET_DEG:
            lo = mid
        else:
            hi = mid
    lean = 0.5 * (lo + hi)
    v = field(lean)
    sk = skew(v)

    # exit angle: low at the hairline, rising to mid-scalp across the transition zone; lower along the sides
    side_w = sm(0.9, 1.3, a) * (1 - sm(1.9, 2.4, a))
    hl = EXIT_HAIRLINE_DEG + (EXIT_LATERAL_DEG - EXIT_HAIRLINE_DEG) * side_w
    exit_deg = hl + (EXIT_MID_DEG - hl) * sm(0.0, EXIT_RAMP, inside)

    return dict(P=P, N=N, mask=mask, density=dens, dir=v, exit_deg=exit_deg, theta=th, region=region,
                region_names=np.array(names), sites=S, whorl=whorl, lean_deg=math.degrees(lean), skew=np.array(sk),
                inside=inside, landmarks=L, knots=knots)


def point_area(F):
    """Each sample's share of the skin's area (HH^2): the sphere's even share scaled by r^2 over the cosine between
    the normal and the direction (skin leaning away from the centre covers more)."""
    P, N = F['P'], F['N']
    U = unit(P - CENTRE)
    r = np.linalg.norm(P - CENTRE, axis=1)
    return 4 * np.pi / len(P) * r ** 2 / np.clip((N * U).sum(1), 0.2, 1)


def streamlines(F, n=420, step=0.008, steps=45, seed=1):
    """Paths that follow the growth direction along the skin, from evenly scattered seeds on the scalp."""
    from scipy.spatial import cKDTree
    P, v, m = F['P'], F['dir'], F['mask']
    t = cKDTree(P)
    rng = np.random.default_rng(seed)
    k = np.where(m & (F['density'] > 60))[0]
    out = []
    for s0 in rng.choice(k, size=min(n, len(k)), replace=False):
        x, L = P[s0].copy(), [P[s0].copy()]
        for _ in range(steps):
            _, i = t.query(x, k=4)
            if not m[i].all():
                break
            d = v[i].mean(0)
            x = x + step * d / max(np.linalg.norm(d), 1e-9)
            _, j = t.query(x)
            x = x - F['N'][j] * ((x - P[j]) @ F['N'][j])
            L.append(x.copy())
        if len(L) > 5:
            out.append(np.array(L))
    return out


VIEWS = [('front', (0, 2), lambda p: p[:, 1] < 0.05, 1),
         ('right side', (1, 2), lambda p: p[:, 0] < 0, -1),
         ('back', (0, 2), lambda p: p[:, 1] > 0.0, -1),
         ('top', (0, 1), lambda p: p[:, 2] > 0.75, 1)]


def sheet(F, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    P, m, dens = F['P'], F['mask'], F['density']
    lines = streamlines(F)
    fig, ax = plt.subplots(1, 4, figsize=(20, 6), dpi=110)
    X = lambda q, i, flip: q[:, i] * (flip if i == 0 else 1)
    for a, (name, (i, j), vis, flip) in zip(ax, VIEWS):
        s = vis(P)
        a.scatter(X(P[s & ~m], i, flip), P[s & ~m][:, j], s=0.5, c='#d9cfc7')
        sc = a.scatter(X(P[s & m], i, flip), P[s & m][:, j], s=1.2, c=dens[s & m], cmap='viridis', vmin=0, vmax=170)
        for Lp in lines:
            if vis(Lp).mean() < 0.6:
                continue
            a.plot(X(Lp, i, flip), Lp[:, j], color='#7b1e1e', lw=0.7)
            a.plot(X(Lp[-1:], i, flip), Lp[-1:, j], 'o', color='#7b1e1e', ms=1.6)
        S = F['sites']; sv = vis(S)
        a.scatter(X(S[sv], i, flip), S[sv][:, j], s=40, marker='x', c='k')
        lm = np.array([F['landmarks'][k] for k in ('trichion', 'temporal_point')])
        lm = np.r_[lm, lm[1:] * [-1, 1, 1]]
        lv = vis(lm)
        a.scatter(X(lm[lv], i, flip), lm[lv][:, j], s=36, marker='D', facecolors='none', edgecolors='#1f5fbf')
        w = F['whorl'][None]
        if vis(w)[0]:
            a.scatter(X(w, i, flip), w[:, j], s=80, marker='*', c='#e67e22')
        a.set_title(name); a.set_aspect('equal'); a.axis('off')
    fig.colorbar(sc, ax=ax, shrink=0.6, label='hairs / cm2')
    fig.suptitle('Scalp v2. Density [Leerunyakul 2020], 2 cm transition [Wu 2025]; tight right clockwise whorl [Schmidt '
                 '2008] + frontal stream, skew %.1f deg [target 22.2]. x sites, diamond trichion / temporal point, star '
                 'whorl' % (F['skew'][0] + F['skew'][1]))
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    head, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    F = build(Surface(load_head(head)))
    per = point_area(F)
    area = float((per * F['mask']).sum() * (HH_MM / 10) ** 2)
    total = float((F['density'] * per).sum() * (HH_MM / 10) ** 2)
    save = {k: v for k, v in F.items() if k not in ('landmarks', 'knots')}
    np.savez_compressed(os.path.join(out, 'scalp_field.npz'), point_area_hh2=per, **save)
    sheet(F, os.path.join(out, 'scalp_field_sheet.png'))
    L = F['landmarks']
    rep = dict(scalp_area_cm2=round(area, 1), hair_count=int(total), whorl=np.round(F['whorl'], 4).tolist(),
               frontal_lean_deg=round(F['lean_deg'], 1), frontal_skew_deg=np.round(F['skew'], 1).tolist(),
               landmarks={k: (np.round(v, 4).tolist() if hasattr(v, 'tolist') else round(float(v), 4)) for k, v in L.items()},
               hairline_knots=[[round(a, 3), round(float(b), 4)] for a, b in F['knots']],
               sites={str(k): np.round(s, 3).tolist() for k, s in zip(F['region_names'], F['sites'])})
    json.dump(rep, open(os.path.join(out, 'scalp_field.json'), 'w'), indent=1)
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
