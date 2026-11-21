"""The scalp's growth field (hair rebuild, foundation 2): where hair grows on the creator's head, how densely, which way
it points as it leaves the skin and at what angle. Every hairstyle is built on this field. A style is then an operation
on the natural fall (part, comb, gather, tie, cut) instead of a hand-placed set of guide curves.

    python3 tools/hair/scalp_field.py HEAD.json.gz OUT_DIR        (HEAD: the creator's head export, rq_hair_export.cjs)

Writes OUT_DIR/scalp_field.npz (points, normals, mask, density, growth direction, exit angle, region weights) and
OUT_DIR/scalp_field_sheet.png (density and flow from the front, side, back and top, plus the measuring sites).

Units: head heights (HH), chin z 0 to crown z 1, face toward -y. One HH is 219 mm (NOSE_LIFE.head_h in the creator).
The character's right is -x (facing -y with z up).

Sources (every constant below names one; EXPERIMENTAL marks a value with no measurement behind it yet):
  [LS20] Leerunyakul & Suchonwanit 2020, BioMed Res Int, PMC7035527. Trichoscopy of 160 adult Thai women.
         Hairs/cm2: frontal 153.9, vertex 162.6, temporoparietal 133.2, occipital 160.1. Shaft about 80 um at every
         site. Sites: 12, 24 and 30 cm from the glabella along the midline part (frontal, vertex, occipital), and
         6 cm from the external ear canal (temporoparietal).
  [KS16] Karaca Saygili et al. 2016, Folia Morphol 75(4):512-517. 100 women: tr-n 66.93 mm, tr-g 51.29 mm
         (straight calliper distances). Solved along this head's forehead slope, trichion z = 0.857 HH.
  [SDK08] Schmidt, Depner & Kabesch 2008, Sci World J 8:848, PMC5848856. 519 children with one parietal whorl:
         right-sided 70.8%, clockwise 83.9%.
  [P23]  Park et al. 2023, Clin Cosmet Investig Dermatol 16:1885, PMC10349607 (citing Unger 2004, Dermatol Surg
         30:829). Exit angle 10-20 deg along the anterior hairline, 30-40 deg mid-scalp.
  [W25]  Wu et al. 2025, J Cosmet Dermatol, PMC12319885. Female hairline: lateral hairline 10-15 deg, frontal
         hairline 15-25 deg. 1-hair follicular units in front, 1- and 2-hair units behind.
  [K25]  Natural asymmetry in hair growth direction, 2025, PMC12502931. 125 East Asian men: the hair along the
         frontal hairline skews right by 22.2 deg on average (right 55.0, left 32.8), following the clockwise whorl.
         Male data, used here as the target for the whorl's swirl until a female measurement is found.
"""
import gzip
import json
import math
import os
import sys

import numpy as np

HH_MM = 219.0                      # mm per head height (the creator's NOSE_LIFE.head_h)
MM = 1.0 / HH_MM

# ---- the hairline (foundation 1). Only the trichion is measured so far; the rest is PROVISIONAL, parked until the
# temple, ear and nape landmarks are derived. Heights are z on the head, by the angle th round the head's axis.
TRICHION_Z = 0.857                 # [KS16]
AXIS = (0.0, 0.04)                 # (x, y) of the head's vertical axis, as the creator's ear clearance uses
EAR_TOP, EAR_BOTTOM = 0.60, 0.365  # the creator's EAR.TOP / EAR.BOTTOM
EAR_CANAL = (0.47, 0.04)           # (z, y) of the ear canal: the ear's middle height, at its depth EY. APPROXIMATE
NAPE_Z = 0.30                      # PROVISIONAL: the nape hairline's lowest point, on the midline at the back
SIDEBURN_Z = 0.50                  # PROVISIONAL: the sideburn's lower end in front of the ear
TRANSITION = 8 * MM                # EXPERIMENTAL: the hairline's transition zone, density 0 to full over this width

# ---- density [LS20], hairs/cm2 at the four sites (women)
DENSITY = {'frontal': 153.9, 'vertex': 162.6, 'temporoparietal': 133.2, 'occipital': 160.1}
SITE_ARC_MM = {'frontal': 120.0, 'vertex': 240.0, 'occipital': 300.0}   # along the midline from the glabella
TP_MM = 60.0                                                            # temporoparietal: above the ear canal
SHAFT_UM = 80.0                                                         # [LS20]

# ---- the whorl and the exit angle
WHORL_SIDE = -1                    # [SDK08]: right-sided (the character's right is -x)
WHORL_CLOCKWISE = True             # [SDK08]: clockwise, seen from above
WHORL_OFFSET = 15 * MM             # EXPERIMENTAL: how far right of the midline (no measured distance yet)
WHORL_ARC_MM = 240.0               # EXPERIMENTAL: on the vertex site's arc [LS20 site] (a parietal whorl)
SKEW_TARGET_DEG = 22.2             # [K25] the left-right difference along the frontal hairline
EXIT_HAIRLINE_DEG = 15.0           # [P23] 10-20, [W25] 15-25 frontal: 15 is inside both
EXIT_LATERAL_DEG = 12.5            # [W25] 10-15 at the lateral hairline
EXIT_MID_DEG = 35.0                # [P23] 30-40 mid-scalp
EXIT_RAMP = 20 * MM                # EXPERIMENTAL: the distance behind the hairline over which the angle rises


def sm(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def load_head(path):
    d = json.load(gzip.open(path))
    return np.array(d['head']['p'], float).reshape(-1, 3)


CENTRE = np.array([0.0, 0.06, 0.56])     # the skull's centre, as the creator places it


class Surface:
    """The head's skin as a radius from the skull's centre by direction, read off the exported points (nearest
    points by angle, weighted), so the scalp can be sampled evenly: the export is dense on the face and sparse on top."""
    def __init__(self, cloud):
        from scipy.spatial import cKDTree
        q = cloud[cloud[:, 2] > 0.2] - CENTRE
        r = np.linalg.norm(q, axis=1)
        self.u, self.r = q / r[:, None], r
        self.tree = cKDTree(self.u)

    def radius(self, U):
        d, i = self.tree.query(U, k=6)
        w = 1.0 / np.maximum(d, 1e-5) ** 2
        # the outer skin: the largest radii among the neighbours carry it (inside the mouth or ear would pull it in)
        return (self.r[i] * w).sum(1) / w.sum(1)

    def point(self, U):
        return CENTRE + U * self.radius(U)[:, None]

    def sample(self, n):
        """n directions spread evenly over the sphere (Fibonacci), the points on the skin and their normals."""
        k = np.arange(n) + 0.5
        z = 1 - 2 * k / n
        a = np.pi * (1 + 5 ** 0.5) * k
        U = np.c_[np.sqrt(1 - z * z) * np.cos(a), np.sqrt(1 - z * z) * np.sin(a), z]
        P = self.point(U)
        # normals from the surface itself: two small steps across the direction
        t1 = np.cross(U, [0, 0, 1.0]); bad = np.linalg.norm(t1, axis=1) < 1e-6; t1[bad] = [1.0, 0, 0]
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True); t2 = np.cross(U, t1)
        e = 0.01
        def p(V):
            V = V / np.linalg.norm(V, axis=1, keepdims=True)
            return self.point(V)
        N = np.cross(p(U + e * t1) - p(U - e * t1), p(U + e * t2) - p(U - e * t2))
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        N *= np.sign((N * U).sum(1))[:, None]
        return P, N, U

    def arc(self, U0, U1, n=400):
        """The path along the skin between two directions (a great circle seen from the centre), with its arc length."""
        w = np.arccos(np.clip(U0 @ U1, -1, 1))
        t = np.linspace(0, 1, n)[:, None]
        U = (np.sin((1 - t) * w) * U0 + np.sin(t * w) * U1) / np.sin(w)
        P = self.point(U)
        return P, np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def theta(P):
    """The angle round the head's axis: 0 at the front, + toward +x (the character's left), +-pi at the back."""
    return np.arctan2(P[:, 0] - AXIS[0], -(P[:, 1] - AXIS[1]))


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def along(path, arcs, a):
    i = int(np.clip(np.searchsorted(arcs, a), 1, len(arcs) - 1))
    t = (a - arcs[i - 1]) / max(arcs[i] - arcs[i - 1], 1e-12)
    return path[i - 1] + t * (path[i] - path[i - 1])


# PROVISIONAL control heights round the head (|th|, z): the measured trichion, a round female contour (the most preferred
# shape in PMC13597767; its prevalence is not yet sourced), the temple, the sideburn in front of the ear, over the ear,
# behind it, and the nape's U. Only the first is measured; the rest wait on foundation 1.
HAIRLINE_KNOTS = [(0.0, TRICHION_Z), (0.4, TRICHION_Z - 0.012), (0.75, TRICHION_Z - 0.05), (1.02, 0.66),
                  (1.2, SIDEBURN_Z), (1.42, EAR_TOP + 0.02), (1.75, EAR_TOP - 0.03), (2.3, 0.42), (np.pi, NAPE_Z)]


def hairline_z(th):
    """PROVISIONAL hairline height by angle: one smooth, shape-keeping curve (PCHIP: no overshoot between knots, flat
    at the front and the back) through HAIRLINE_KNOTS, mirrored so it is even in th."""
    from scipy.interpolate import PchipInterpolator
    k = np.array(HAIRLINE_KNOTS)
    x = np.r_[-k[:0:-1, 0], k[:, 0], 2 * np.pi - k[-2::-1, 0][:2]]
    y = np.r_[k[:0:-1, 1], k[:, 1], k[-2::-1, 1][:2]]
    return PchipInterpolator(x, y)(np.abs(th))


def build(surf, n=40000):
    P, N, U = surf.sample(n)
    th = theta(P)
    # the glabella: the most forward point of the midline brow
    ang = np.linspace(-0.15, 0.6, 300)                  # elevation from the centre, looking forward
    Ug = np.c_[np.zeros_like(ang), -np.cos(ang), np.sin(ang)]
    Pg = surf.point(Ug)
    k = np.argmin(Pg[:, 1] + 10 * (Pg[:, 2] < 0.6))
    g = Pg[k]
    # the midline from the glabella over the top and down the back: the frontal, vertex and occipital sites [LS20]
    path, arcs = [], [0.0]
    U0 = unit(g - CENTRE)
    for U1 in (unit([0, 0, 1.0]), unit([0, 1.0, -0.6])):
        pp, aa = surf.arc(U0, U1)
        if path:
            aa = aa + arcs[-1]
            pp, aa = pp[1:], aa[1:]
        path.append(pp); arcs = np.r_[arcs[:-1], aa] if len(path) == 1 else np.r_[arcs, aa]
        U0 = U1
    path = np.concatenate(path)
    sites = {k2: along(path, arcs, v * MM) for k2, v in SITE_ARC_MM.items()}
    # temporoparietal: 60 mm up the side of the head from the ear canal, along the skin (both sides) [LS20]
    canal = np.array([0.0, EAR_CANAL[1], EAR_CANAL[0]])
    Uc = unit([1.0, EAR_CANAL[1] - CENTRE[1], EAR_CANAL[0] - CENTRE[2]])
    pp, aa = surf.arc(Uc, unit([0, EAR_CANAL[1] - CENTRE[1], 1.0]))
    tp = along(pp, aa, TP_MM * MM)
    sites['temporoparietal_L'] = tp
    sites['temporoparietal_R'] = tp * np.array([-1, 1, 1])
    prof = path

    # the scalp: the head above the hairline, outside the face
    hz = hairline_z(th)
    behind = hz - P[:, 2]                              # > 0 below the hairline
    on = P[:, 2] > 0.25
    edge = sm(0.0, TRANSITION, -behind)                # 0 at the hairline, 1 a transition zone behind it
    mask = on & (behind < 0)

    # density: inverse-distance weighting between the sites (surface-near Euclidean, fine at this spacing)
    names = ['frontal', 'vertex', 'occipital', 'temporoparietal_L', 'temporoparietal_R']
    vals = np.array([DENSITY['frontal'], DENSITY['vertex'], DENSITY['occipital'],
                     DENSITY['temporoparietal'], DENSITY['temporoparietal']])
    S = np.array([sites[k] for k in names])
    D = np.linalg.norm(P[:, None, :] - S[None], axis=2)
    W = 1.0 / np.maximum(D, 1e-4) ** 2
    region = W / W.sum(1, keepdims=True)
    dens = (region * vals).sum(1) * edge * mask

    # the whorl: on the vertex arc, offset to the right
    wv = sites['vertex'].copy()
    wv[0] += WHORL_SIDE * WHORL_OFFSET
    near = P[np.argmin(np.linalg.norm(P - wv, axis=1))]
    whorl = near

    # growth direction: a spiral source at the whorl. Radial (away from the whorl along the skin) turned by an angle
    # that is full at the whorl and fades with distance; the fade is solved so the frontal hairline skews by the
    # measured left-right difference [K25].
    def field(decay):
        d = P - whorl
        d -= (d * N).sum(1, keepdims=True) * N          # onto the tangent plane
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        spin = np.cross(N, d)                            # 90 deg round the normal, counterclockwise seen from outside
        if WHORL_CLOCKWISE:
            spin = -spin
        dist = np.linalg.norm(P - whorl, axis=1)
        turn = (np.pi / 2) * np.exp(-dist / decay)       # 90 deg at the whorl (a pure swirl), radial far away
        v = np.cos(turn)[:, None] * d + np.sin(turn)[:, None] * spin
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    # the two points of [K25]: halfway between the midline and the frontotemporal corner, on the hairline
    probe_th = np.array([-0.45, 0.45])
    def skew(decay):
        v = field(decay)
        out = []
        for t in probe_th:
            k = np.argmin(np.abs(th - t) + 10 * np.abs(P[:, 2] - (hairline_z(np.array([t]))[0] + 0.01)) + 10 * (~on))
            # the direction's lean toward the character's right (-x), against straight down the forehead
            n = N[k]
            down = np.array([0, -0.5, -1.0]); down -= n * (n @ down); down /= np.linalg.norm(down)
            right = np.array([-1.0, 0, 0]); right -= n * (n @ right) + down * (down @ right); right /= np.linalg.norm(right)
            out.append(math.degrees(math.atan2(float(v[k] @ right), float(v[k] @ down))))
        return out
    lo, hi = 0.02, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        s = skew(mid)
        if s[0] + s[1] < SKEW_TARGET_DEG:      # a common lean c on both sides: [K25]'s difference is 2c
            lo = mid
        else:
            hi = mid
    decay = 0.5 * (lo + hi)
    v = field(decay)
    sk = skew(decay)

    # exit angle: low at the hairline, rising to mid-scalp over EXIT_RAMP; lower along the sides [P23, W25]
    side_w = sm(0.9, 1.3, np.abs(th)) * (1 - sm(1.9, 2.4, np.abs(th)))
    hl = EXIT_HAIRLINE_DEG + (EXIT_LATERAL_DEG - EXIT_HAIRLINE_DEG) * side_w
    exit_deg = hl + (EXIT_MID_DEG - hl) * sm(0.0, EXIT_RAMP, -behind)

    return dict(P=P, N=N, mask=mask, density=dens, dir=v, exit_deg=exit_deg, theta=th, region=region,
                region_names=np.array(names), sites=S, whorl=whorl, decay=decay, skew=np.array(sk),
                glabella=g, prof=prof)


def scalp_area_cm2(F):
    """Scalp area from the masked points: each point's share of the head's area, from the mean spacing."""
    # even directions: each point's area is the sphere's share scaled by r^2 over the cosine between the normal and
    # the direction (the skin leaning away from the centre covers more)
    P, N = F['P'], F['N']
    U = (P - CENTRE) / np.linalg.norm(P - CENTRE, axis=1, keepdims=True)
    r = np.linalg.norm(P - CENTRE, axis=1)
    a = 4 * np.pi / len(P) * r ** 2 / np.clip((N * U).sum(1), 0.2, 1)
    return float((a * F['mask']).sum() * (HH_MM / 10) ** 2), a


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
            x = P[j] + (x - P[j]) - F['N'][j] * ((x - P[j]) @ F['N'][j])     # back onto the skin
            L.append(x.copy())
        if len(L) > 5:
            out.append(np.array(L))
    return out


def sheet(F, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    P, m, dens, v = F['P'], F['mask'], F['density'], F['dir']
    views = [('front', (0, 2), lambda p: p[:, 1] < 0.05, 1),
             ('right side', (1, 2), lambda p: p[:, 0] < 0, -1),
             ('back', (0, 2), lambda p: p[:, 1] > 0.0, -1),
             ('top', (0, 1), lambda p: p[:, 2] > 0.75, 1)]
    fig, ax = plt.subplots(1, 4, figsize=(20, 6), dpi=110)
    lines = streamlines(F)
    for a, (name, (i, j), vis, flip) in zip(ax, views):
        s = vis(P)
        a.scatter(P[s & ~m][:, i] * (flip if i == 0 else 1), P[s & ~m][:, j], s=0.5, c='#d9cfc7')
        sc = a.scatter(P[s & m][:, i] * (flip if i == 0 else 1), P[s & m][:, j], s=1.2, c=dens[s & m],
                       cmap='viridis', vmin=0, vmax=170)
        # flow: streamlines traced on the skin along the growth direction, each from a seed to a dot at its end
        for L in lines:
            q = vis(L)
            if q.mean() < 0.6:
                continue
            a.plot(L[:, i] * (flip if i == 0 else 1), L[:, j], color='#7b1e1e', lw=0.7)
            a.plot(L[-1, i] * (flip if i == 0 else 1), L[-1, j], 'o', color='#7b1e1e', ms=1.6)
        S = F['sites']
        sv = vis(S)
        a.scatter(S[sv][:, i] * (flip if i == 0 else 1), S[sv][:, j], s=40, marker='x', c='k')
        w = F['whorl'][None]
        if vis(w)[0]:
            a.scatter(w[:, i] * (flip if i == 0 else 1), w[:, j], s=80, marker='*', c='#e67e22')
        a.set_title(name)
        a.set_aspect('equal')
        a.axis('off')
    fig.colorbar(sc, ax=ax, shrink=0.6, label='hairs / cm2')
    fig.suptitle('Scalp growth field: density [Leerunyakul 2020], flow from a right, clockwise parietal whorl '
                 '[Schmidt 2008], skew %.1f deg [target 22.2]. x: measuring sites, star: whorl'
                 % (F['skew'][0] + F['skew'][1]))
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    head, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    F = build(Surface(load_head(head)))
    area, per = scalp_area_cm2(F)
    total = float((F['density'] * per).sum() * (HH_MM / 10) ** 2)
    np.savez_compressed(os.path.join(out, 'scalp_field.npz'), **{k: v for k, v in F.items()}, point_area_hh2=per)
    sheet(F, os.path.join(out, 'scalp_field_sheet.png'))
    rep = dict(scalp_area_cm2=round(area, 1), hair_count=int(total), whorl=np.round(F['whorl'], 4).tolist(),
               swirl_decay_hh=round(F['decay'], 4), frontal_skew_deg=np.round(F['skew'], 1).tolist(),
               sites={str(k): np.round(s, 3).tolist() for k, s in zip(F['region_names'], F['sites'])},
               glabella=np.round(F['glabella'], 3).tolist())
    json.dump(rep, open(os.path.join(out, 'scalp_field.json'), 'w'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
