"""The redingote's collar: a turned-down collar with a notch and two points, as a tailor cuts it.

Drafted FLAT first (draft()): one piece, the neck edge along the bottom (as long as the bodice's neckline from one
front edge round the back to the other), the stand above it, the roll line, and the fall above that, wider at the fronts
where the points are. Every size is a share of w, the neck's half-width (the vest's unit, as the bodice's are).

Then it is sewn on and folded by the local fabric solver (blender/shared/fabric.py), the way the sleeves and the
petticoat are: the neck edge is sewn to targets along the bodice's neckline, laid standing round the neck, the roll
line is a pressed fold (soft bending there, stiff everywhere else), the fall is laid over the stand and let go on the
bodice's shoulders and the body. No Blender in this file: the arrays in, the arrays out."""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))

# The proportions are measured off John's reference photo (an 1890s habit on a form, 3/4 view), against the form's
# neck half-width (the w here): the stand about 0.6 w at the back, the fall about 0.75 w over the shoulder and about
# 1 w at the points, the points nearly square; the lapel small, the opening breaking about 1.1 w below the neck at the
# top button, the notch about square, and the lapel's point a little shorter than the collar's.
RATIOS = dict(
    stand=0.65,            # the stand's height (neck edge to the roll line), in w
    fall_back=0.75,        # the fall's depth at the back, in w
    fall_front=0.70,       # the fall's depth at the points
    notch=0.35,            # the slant of the front end: the point stands this far past its foot at the neck edge, in w
    gap=0.80,              # the open gap between the two front ends at the neck edge, in w
    lean=0.18,             # how far the stand leans in toward the neck, as a share of its height
    spacing=0.075,         # the grid's point spacing, in w
    stand_front=None,      # the stand at the front ends (None: as at the back). A notch collar's stand dies away toward
                           # the gorge, so its roll line runs on into the lapel's
)


def neck_height(zf, zs, zb):
    """The bodice's neckline height as a function of the angle from the front (bodice.neckline's own form)."""
    a = 0.25 * (zf + zb) + 0.5 * zs
    c, b = a - zs, 0.5 * (zf - zb)
    return lambda ang: a + b * np.cos(ang) + c * np.cos(2 * ang)


def angle_of(P, yc):
    """The angle from the front round toward her right (-x): the bodice's own convention."""
    return np.mod(np.arctan2(-P[:, 0], -(P[:, 1] - yc)), 2 * np.pi)


def neck_edge(V, M, n=120):
    """The bodice's neck edge as a closed curve of n points: the outermost cloth near the neckline height at each angle."""
    f = neck_height(M['z_neck_front'], M['z_neck_side'], M['z_neck_back'])
    ang = angle_of(V, M['yc'])
    near = np.abs(V[:, 2] - f(ang)) < 0.35
    P, A = V[near], ang[near]
    out = np.zeros((n, 3))
    for k in range(n):
        t = 2 * np.pi * k / n
        d = np.abs(np.angle(np.exp(1j * (A - t))))
        sel = d < np.pi / n * 1.5
        Q = P[sel]
        r = np.hypot(Q[:, 0], Q[:, 1] - M['yc'])
        keep = r >= np.percentile(r, 60)
        out[k] = Q[keep].mean(0)
    return out


def draft(E, M, R=RATIOS):
    """The flat pattern: points (s, t) with s along the neck edge from the left front end to the right front end
    (back at 0), t up from it, the triangles, the roll-line's mask and the neck-edge points. E: neck_edge's curve."""
    from scipy.spatial import Delaunay
    w = M['w']
    ang = 2 * np.pi * np.arange(len(E)) / len(E)
    gap_ang = R['gap'] * w / np.mean(np.hypot(E[:, 0], E[:, 1] - M['yc']))
    inside = (ang > gap_ang / 2) & (ang < 2 * np.pi - gap_ang / 2)
    idx = np.nonzero(inside)[0]
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(E[idx], axis=0), axis=1))])
    L = arc[-1]
    st, fb, ff = R['stand'] * w, R['fall_back'] * w, R['fall_front'] * w
    stf = st if R.get('stand_front') is None else R['stand_front'] * w

    def st_of(s):                                     # the stand's height along the collar
        u = np.clip(np.abs(s) / (0.5 * L), 0, 1)
        return st + (stf - st) * u ** 2
    h = R['spacing'] * w

    nd = R['notch'] * w                               # the front end is cut on the slant: the point stands this far past its foot

    def top(s):                                       # the cut edge's height above the neck edge, rising to the points
        u = np.clip(np.abs(s) / (0.5 * L), 0, 1)
        return st_of(s) + fb + (ff - fb) * (u ** 2.0)
    t_end = float(top(0.5 * L))

    def smax(t):                                      # how far out the piece reaches at height t (the slanted end)
        return 0.5 * L - nd * (1 - np.clip(t / t_end, 0, 1))
    ss = np.arange(-0.5 * L + nd, 0.5 * L - nd + 1e-9, h * 0.7)
    bnd = [(s, 0.0) for s in ss] + [(s, float(top(s))) for s in ss if abs(s) < smax(float(top(s)))]
    for sgn in (-1, 1):
        for t in np.arange(0, t_end + 1e-9, h * 0.7):
            bnd.append((sgn * smax(t), t))
    gs = np.arange(-0.5 * L + h, 0.5 * L - h * 0.5, h)
    gt = np.arange(h, t_end, h)
    grid = [(s, t) for s in gs for t in gt if 0.4 * h < t < top(s) - 0.4 * h and abs(s) < smax(t) - 0.5 * h]
    pts = np.unique(np.round(np.array(bnd + grid), 6), axis=0)
    tri = Delaunay(pts)
    mid = pts[tri.simplices].mean(1)
    ok = (mid[:, 1] > 0) & (mid[:, 1] < top(mid[:, 0])) & (np.abs(mid[:, 0]) < smax(mid[:, 1]))
    T = tri.simplices[ok]
    # drop sliver triangles from the boundary
    A = pts[T]
    area = 0.5 * np.abs((A[:, 1, 0] - A[:, 0, 0]) * (A[:, 2, 1] - A[:, 0, 1]) - (A[:, 2, 0] - A[:, 0, 0]) * (A[:, 1, 1] - A[:, 0, 1]))
    T = T[area > 0.01 * h * h]
    used = np.unique(T)
    remap = -np.ones(len(pts), int)
    remap[used] = np.arange(len(used))
    pts, T = pts[used], remap[T]
    return dict(UV=pts, T=T, L=L, stand=st, st_pt=st_of(pts[:, 0]), idx=idx, arc=arc, s0=-0.5 * L, smax=smax)


SETTINGS = dict(
    flare=0.12,            # the stand's flare outward from the neck, in radians off the vertical (a turned-down collar spreads)
    fold=1.10,             # the fall folded over, below the horizontal, in radians
    lift=0.16,             # the fall's cloth stands this far off the stand where it is turned over
    stretch=1e-6, bend=1e-6, flat=1e-6, crease=1e-6,
    gap=0.10,              # the cloth's stand-off from the bodice and the body
    steps=1800,
)


def frame(E, M):
    """Per neck-edge point: up, and the horizontal unit vector toward the neck's axis."""
    n = np.array([-E[:, 0], -(E[:, 1] - M['yc']), np.zeros(len(E))]).T
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return np.array([0, 0, 1.0]), n


def along(E, idx, s, L):
    """The neck edge's point and inward vector at arc position a (0..L) along the open arc idx."""
    seg = E[idx]
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))])
    a = np.clip(s, 0, arc[-1])
    return np.array([np.interp(a, arc, seg[:, k]) for k in range(3)]).T


def design(D, E, M, S=SETTINGS):
    """The collar as worn: (folded rest positions, funnel start positions) of the pattern's points."""
    UV = D['UV']
    s = UV[:, 0] - D['s0']
    t = UV[:, 1]
    Eo = along(E, D['idx'], s, D['L'])
    n = np.array([-Eo[:, 0], -(Eo[:, 1] - M['yc']), np.zeros(len(Eo))]).T
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    u = np.array([0, 0, 1.0])
    g, b = S['flare'], S['fold']
    d1 = u[None] * math.cos(g) - n * math.sin(g)                       # up the stand, spreading
    st = D['st_pt']
    tt = np.minimum(t, st)[:, None]
    dd = np.maximum(t - st, 0)[:, None]
    R = Eo + tt * d1
    out = -n
    d2 = out * math.cos(b) - u[None] * math.sin(b)                     # the fall, turned over and down
    roll = np.minimum(dd / (0.35 * M['w']), 1.0) ** 0.5                 # (a rounded roll, not a hinge: the fall eases off the stand)
    folded = R + dd * d2 + roll * (out * S['lift'] + np.array([0, 0, 0.6 * S['lift']]))
    funnel = R + dd * d1
    return folded, funnel, Eo


def solve(D, E, M, bodyP, bodyN, S=SETTINGS, R_=RATIOS, log=print):
    import fabric as xf
    folded, funnel, Eo = design(D, E, M, S)
    UV, T = D['UV'], D['T']
    fab = xf.Fabric(funnel, dict(walk='gauss-seidel', iterations=4, stretch=0.0, bend=2e-4, damping=0.03, friction=0.4))
    neck = np.nonzero(UV[:, 1] < 1.05 * R_['spacing'] * M['w'])[0]       # (the sewn rows: two of them, so the stand's hinge at the seam is held)
    fab.pin(neck)
    fab.pos[neck] = folded[neck]
    fab.prev[:] = fab.pos
    soft = np.where(np.abs(UV[:, 1] - D['st_pt']) < 0.09 * M['w'], S['crease'], 0.0)
    fab.mesh(T, stretch=S['stretch'], rest_pos=folded, bend=S['bend'], bend_at=soft, flat=S['flat'])
    body = xf.MeshBody(bodyP, bodyN, gap=S['gap'], reach=1.0)
    k = fab.settle(body, h=1 / 240, max_steps=S['steps'], tol=2e-4, log=log)
    log('collar: settled in %d steps' % k)
    return fab.pos.copy(), folded


LAPEL = dict(
    drop=1.60,             # the break point (where the lapel's roll line ends, at the top button) below the neck edge's front, in w
    notch_at=0.50,         # the notch on the collar's front edge: this share of the way from the roll to its point
    notch_angle=80,        # the notch's opening, in degrees
    point_len=0.70,        # the lapel's point as long as this share of the collar's (notch to point)
    belly=0.10,            # the outer edge's curve out past its straight line, in w
    spacing=0.07,
    edge_rounds=8,         # rounds of evening out the cut edges
    whole_steps=1500,      # steps of the solve with the front let go (see _with_front)
    fold_deg=176,          # the lapel's fold: a full turn back over the roll line, pressed (degrees)
    fold_gap=0.12,         # how far the lapel's cloth past the turn rests off the bodice under it: just clear of the front's face (the cloth is 0.088 thick), so the lapel slopes down from its roll (0.32 high) to the chest. At 0.28 it lay level with the roll's top and read as floating (John, 2026-11-20); the value is a judgement, not from a source
    fold_radius=0.16,      # the fold's inside radius: a rounded roll that carries the lapel up to where it rests
    fold_row=0.05,         # rows of points marked parallel to the roll through the turn, this far apart
    # The fold's solve. The lapel hangs from the turn's fine rows: at the cloth's usual step (1/240 s) and give (1e-6)
    # its own weight stretched those rows by a fifth to double (the contour sheet's strain map), and the lapel came out
    # a different shape from its draft. At 1/1920 s and 1e-8 the strain is 2 to 4 %. correct=False: folds.py's
    # correction scales the turn's hinges up when the lapel rests a few degrees open on the chest, and crumples the rows.
    fold_stretch=1e-8,
    bake=dict(h=1.0 / 1920, most=3000, correct=False),
    tilt=25,               # the lapel starts this far turned off the chest (degrees); the solver folds the rest
    steps=1600,
)
# The notch lapel's own gorge and point (in w; x outward from the middle, z up): the notch G from the neck point N, and
# the lapel's point from G. These are the approved lapels' values (they were first read off the turned-down collar's
# solved ends; kept here the lapels no longer depend on a collar's solve, and the collar is drafted to them).
# roll_side: the roll line is aimed one side-stand up the neck from the neck point at the shoulder (the tailor's draft:
# 1 inch, which at her scale is 0.49 w), so it runs on as the collar's roll; it ends on the neck edge where it crosses
# it (roll_cross). The gorge and point stay where they were approved, measured from the neck edge's point at 'gap'.
# point: widened from (0.572, -0.403) to John's guide lines on the 1885-95 habit photo (2026-11-20): the lapel's outer
# edge runs out toward the armhole and down to the bust. The values are read off his sketch by eye: experimental.
NOTCH_LAPEL = dict(gorge=(0.157, -0.474), point=(0.85, -0.403), roll_side=0.49, belly=0.10)   # (1.10, -0.55 with belly 0.25 gave a lobe hanging under the break that peeled off the chest: fold 105 deg)


# The styles. 'turned': the first one (kept as it was approved): a turned-down collar all round, its short points over
# small lapels. 'notch': a tailored notch collar, as in the reference: the collar is sewn on only round the back and
# sides, its stand dying away toward the front so that its roll runs on into the lapels' roll, the fronts open in a
# wide V from the neck's sides down to the top button and turned back as the lapels, the collar's ends lying along the
# gorge, and the notch between the collar's point and the lapel's.
STYLES = dict(
    turned=(dict(RATIOS), dict(LAPEL)),
    notch=(dict(RATIOS, stand=0.65, stand_front=0.12, fall_back=0.75, fall_front=1.00, notch=0.30, gap=1.8),
           dict(LAPEL, drop=1.35, notch_at=0.60, notch_angle=65, point_len=0.85, point_w=0.70, **NOTCH_LAPEL)),
)
STYLE = os.environ.get('COLLAR_STYLE', 'notch')
RATIOS.update(STYLES[STYLE][0])
LAPEL.update(STYLES[STYLE][1])
if os.environ.get('COLLAR_LAPEL'):                  # (a trial's own values, as JSON)
    import json
    LAPEL.update(json.loads(os.environ['COLLAR_LAPEL']))


def lapel_outlines(E, M, D, cpos, L_=None):
    """Each lapel's outline as it lies turned back, in (x, z): (break point B, neck point N, polygon)."""
    L_ = LAPEL if L_ is None else L_
    w = M['w']
    seg = E[D['idx']]
    outlines = []
    for end in (0, -1):
        foot = seg[end]
        sg = 1.0 if foot[0] > 0 else -1.0
        # the lapel as a tailor cuts it: the roll line (break B up to the neck at N), the gorge (N out along the collar's
        # seam to the notch G), the notch's other side up to the lapel's point P, and the outer edge from P back to B
        # in a soft convex curve
        B = np.array([-sg * L_.get('break_x', 0.0), foot[2] - L_['drop'] * w])       # (its front's own edge: past the middle, on the other side)
        N = np.array([foot[0], foot[2]])
        # in harmony with the collar: the notch G sits on the collar's own front edge (as solved), part way from the
        # roll down to the collar's point C, and the lapel's point mirrors C across the notch: the same length, turned
        # down by the notch's angle
        if L_.get('gorge'):                                              # (set outright: see NOTCH_LAPEL)
            G = N + np.array([sg * L_['gorge'][0], L_['gorge'][1]]) * w
            Pp = G + np.array([sg * L_['point'][0], L_['point'][1]]) * w
            if L_.get('roll_side'):
                N = roll_cross(E, M, B, sg, L_)[0]
            if L_.get('from_front', False):     # (tried and dropped: John wants the wide lapel)
                # The lapel is the buttoned front's own top corner turned back (John, 2026-11-20, on the 1890s habit
                # photo: the front edge is one line from the hem to the neck, and the lapel is what lies above the top
                # button, creased on the roll). Buttoned to the neck the front's edge runs straight up from B to the
                # neck edge at K; the cloth between that edge, the neck edge and the roll line is the lapel, and turned
                # back it is that triangle mirrored in the roll line. The gorge is the front's neck edge (N to K'), the
                # collar sewn on it as far as the notch G (notch_at of the way), and the lapel's point is K' itself.
                # The set gorge and point above made a lapel whose point, unfolded, lay 0.95 past the front's edge and
                # 1.8 above the neck: cloth the buttoned front does not have.
                fr = E[E[:, 1] < M['yc']]
                fr = fr[np.argsort(fr[:, 0])]
                K = np.array([B[0], float(np.interp(B[0], fr[:, 0], fr[:, 2]))])
                Pp = _mirror(K[None], B, N)[0]
                G = N + L_.get('notch_at', 0.6) * (Pp - N)
        else:
            UVc = D['UV']
            side = np.sign(cpos[:, 0]) == sg
            endpts = np.nonzero(side & (np.abs(UVc[:, 0]) > D['smax'](UVc[:, 1]) - 0.6 * L_['spacing'] * w) & (UVc[:, 1] >= D['st_pt']))[0]
            C3 = cpos[endpts[np.argmax(UVc[endpts, 1])]]
            R3 = cpos[endpts[np.argmin(UVc[endpts, 1])]]
            C = C3[[0, 2]]
            G = R3[[0, 2]] + L_['notch_at'] * (C - R3[[0, 2]])
            v = (C - G) * L_['point_len']
            if L_.get('point_w'):                                           # (or a set length, in w, the direction kept)
                v = v / max(np.linalg.norm(v), 1e-9) * L_['point_w'] * w
            a = math.radians(L_['notch_angle'])
            cands = [np.array([v[0] * math.cos(t) - v[1] * math.sin(t), v[0] * math.sin(t) + v[1] * math.cos(t)]) for t in (a, -a)]
            Pp = G + max(cands, key=lambda u: sg * u[0])                       # (the turn that takes it outward, the notch opening sideways)
        q = np.linspace(0, 1, 24)[1:-1]
        chord = B - Pp
        bulge = np.array([chord[1], -chord[0]]) / np.linalg.norm(chord) * (0.0 if L_.get('gorge') and L_.get('from_front', False) else L_['belly']) * w   # (the buttoned edge is straight)
        bulge *= np.sign(bulge[0]) * sg
        outer = Pp[None] + q[:, None] * chord[None] + (4 * q * (1 - q))[:, None] * bulge[None]
        poly = np.vstack([B, N, G, Pp, outer])
        outlines.append((B, N, poly))
    return outlines


def lapels(E, M, bodyP, bodyN, D, cpos, L_=LAPEL, log=print):
    """The two lapels: the bodice's fronts turned back below the collar. Each is cut flat as the triangle the roll
    line (from the collar's end at the neck edge down to the break point) leaves outside it, out to the lapel's point,
    and laid back on the chest in the solver, its roll line held. Returns [(points, triangles)] for the two sides."""
    from scipy.spatial import Delaunay, cKDTree
    import fabric as xf
    w = M['w']
    seg = E[D['idx']]
    out = []
    front = bodyP[:, 1] < M['yc']
    tree = cKDTree(bodyP[front][:, [0, 2]])
    FP, FN = bodyP[front], bodyN[front]

    def on_chest(xz, lift):
        _, j = tree.query(xz, k=6)
        y = FP[j, 1].min(1)                            # the frontmost cloth under each point
        n = FN[j].mean(1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        return np.column_stack([xz[:, 0], y, xz[:, 1]]) + lift * n
    outlines = lapel_outlines(E, M, D, cpos, L_)
    for (B, N, poly), end in zip(outlines, (0, -1)):
        foot = seg[end]
        sg = 1.0 if foot[0] > 0 else -1.0
        h = L_['spacing'] * w
        bnd = []
        for k in range(len(poly)):
            a, b = poly[k], poly[(k + 1) % len(poly)]
            n = max(int(np.linalg.norm(b - a) / (0.7 * h)), 1)
            bnd += [a + (b - a) * j / n for j in range(n)]
        bnd = np.array(bnd)
        from matplotlib.path import Path
        path = Path(poly)
        lo, hi = poly.min(0), poly.max(0)
        gx, gz = np.meshgrid(np.arange(lo[0], hi[0], h), np.arange(lo[1], hi[1], h))
        Gr = np.column_stack([gx.ravel(), gz.ravel()])
        Gr = Gr[path.contains_points(Gr)]
        dmin = np.min(np.linalg.norm(Gr[:, None] - bnd[None], axis=2), 1)
        Gr = Gr[dmin > 0.45 * h]
        UV = np.vstack([bnd, Gr])                      # the lapel as it lies when turned back (x, z)
        # The lapel is the bodice's own front, cut as one with it and turned back along the roll line (B to N). So it
        # is cut where it grows: the outline mirrored across the roll line, into the V the front opens (it crosses the
        # middle, as a lapped front does). A strip of the bodice beyond the roll line goes in the same sheet, held
        # where it lies, so the fold has cloth on both sides of it; the solver turns the lapel back over it.
        BN_ = N - B
        e = BN_ / np.linalg.norm(BN_)

        def mirror(X):
            d = X - B
            return B + 2 * (d @ e)[:, None] * e[None] - d

        def dist_line(X):
            t_ = np.clip(((X - B) @ BN_) / (BN_ @ BN_), 0, 1)
            return np.linalg.norm(X - (B + t_[:, None] * BN_), axis=1)
        on_roll = dist_line(UV) < 0.3 * h
        band = UV[(~on_roll) & (dist_line(UV) < 2.3 * h)]              # (the strip: the bodice beyond the roll)
        cutUV = mirror(UV)                                              # the lapel where it is cut
        P2 = np.vstack([cutUV, band])
        nl = len(UV)
        T = Delaunay(P2).simplices
        cpath = Path(mirror(poly))
        mid = P2[T].mean(1)
        keep = cpath.contains_points(mid) | ((dist_line(mid) < 2.0 * h) & path.contains_points(mid))
        T = T[keep]
        # rest: the fold made (the lapel back on the chest over the strip); start: the front as cut, unfolded, tilted
        # a little off the chest at the roll so the solver turns it the right way
        rest = np.vstack([on_chest(UV, 0.10), on_chest(band, 0.02)])
        flat = np.vstack([on_chest(cutUV, 0.02), on_chest(band, 0.02)])
        a0, a1 = on_chest(B[None], 0.02)[0], on_chest(N[None], 0.02)[0]
        ax = (a1 - a0) / np.linalg.norm(a1 - a0)
        th = math.radians(L_['tilt'])
        d = flat - a0
        par = (d @ ax)[:, None] * ax[None]
        perp = d - par
        start = a0 + par + perp * math.cos(th) + np.cross(ax[None], perp) * math.sin(th)
        fwd = np.array([0, -1.0, 0])                                    # (tilt it toward the front, off the body)
        if ((start[:nl] - flat[:nl]) @ fwd).mean() < 0:
            start = a0 + par + perp * math.cos(th) - np.cross(ax[None], perp) * math.sin(th)
        start[nl:] = flat[nl:]
        fab = xf.Fabric(start, dict(walk='gauss-seidel', iterations=4, stretch=0.0, bend=2e-4, damping=0.03, friction=0.4))
        fab.pin(np.concatenate([np.nonzero(on_roll)[0], np.arange(nl, len(P2))]))
        fab.pos[:nl][on_roll] = rest[:nl][on_roll]
        fab.prev[:] = fab.pos
        crease = np.zeros(len(P2))
        fab.mesh(T, stretch=1e-6, rest_pos=rest, bend=1e-6, flat=1e-6, bend_at=crease)
        body = xf.MeshBody(bodyP, bodyN, gap=0.07, reach=1.0)
        for _ in range(240):                                            # (the turn itself: no early stop while it starts)
            fab.step(1 / 240, body, neutral=True)
        k = 240 + fab.settle(body, h=1 / 240, max_steps=L_['steps'], tol=1e-4)
        fold = float(np.degrees(np.arccos(np.clip(((fab.pos[:nl] - a0) @ (rest[:nl] - a0).T).diagonal().sum()
                                               / max(np.linalg.norm(fab.pos[:nl] - a0) * np.linalg.norm(rest[:nl] - a0), 1e-9), -1, 1))))
        log('collar: lapel turned back in %d steps (%d points), %.0f degrees off the pressed fold' % (k, nl, fold))
        TL = T[(T < nl).all(1)]                                         # (the lapel's own faces: the strip is the bodice's)
        out.append((fab.pos[:nl].copy(), TL))
    lapels.outlines = outlines
    return out


def open_front(objs, M, E, D, L_=LAPEL):
    """Open the bodice's front above the break point into the V the lapels turn back from: the faces inside the V
    (and the buttons above the break) are cut away. The V runs from the break point up to the collar's two ends."""
    import bmesh
    seg = E[D['idx']]
    xn = 0.5 * (abs(seg[0][0]) + abs(seg[-1][0]))
    zn = 0.5 * (seg[0][2] + seg[-1][2])
    zb = zn - L_['drop'] * M['w']
    cut = 0
    for o in objs:
        mw = o.matrix_world
        bm = bmesh.new()
        bm.from_mesh(o.data)
        dead = []
        for f in bm.faces:
            c = mw @ f.calc_center_median()
            if c.y < M['yc'] and c.z > zb:
                half = xn * min((c.z - zb) / max(zn - zb, 1e-6), 1.2)
                if abs(c.x) < half or (o.name.endswith('buttons') and c.z > zb - 0.15 * M['w']):
                    dead.append(f)
        bmesh.ops.delete(bm, geom=dead, context='FACES')
        # the cut edge laid straight along the V (a face-by-face cut leaves a staircase that reads as a broken fold)
        inv = mw.inverted()
        for v in bm.verts:
            if not v.is_boundary or o.name.endswith('buttons'):
                continue
            c = mw @ v.co
            if c.y < M['yc'] and zb - 0.3 * M['w'] < c.z < zn + 0.3 * M['w']:
                half = xn * min(max((c.z - zb) / max(zn - zb, 1e-6), 0.0), 1.2)
                if abs(abs(c.x) - half) < 0.45 * M['w']:
                    c.x = math.copysign(half, c.x if c.x != 0 else 1.0)
                    v.co = inv @ c
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
        bm.to_mesh(o.data)
        bm.free()
        o.data.update()
        cut += len(dead)
    return cut


FOLDED = []          # what fold_front last folded, per lapel: flat pattern, folded points, triangles (for the contour check)
FOLD_FABRIC = dict(walk='gauss-seidel', iterations=4, stretch=0.0, bend=2e-4, damping=0.03, friction=0.4)


FRONT_FLOOR = []


def fold_front(o, outlines, M, bodyP, bodyN, L_=LAPEL, log=print):
    """Turn the fronts back as the lapels. The bodice comes with its front open along the two roll lines (front_cut).
    Each front is drafted on past its roll line by its lapel: the lapel's outline mirrored across the roll, which is
    where that cloth lies before it is turned (it runs past the centre front, as a front that laps over the other
    does). That extension is meshed flat in the pattern with rows of points parallel to the roll, joined to the
    bodice's own points on the roll line, and the fabric solver turns it back over the roll onto the chest with
    folds.py's hinges (the bodice's cloth next to the roll held, as the strip the fold turns against). The bodice's
    mesh gains the folded lapels in place. Returns (points folded, [(points, triangles)] per lapel); what was folded
    is kept in FOLDED for the contour sheet."""
    import bmesh
    import shapely
    import shapely.affinity
    import fabric as xf
    import folds as fo
    from scipy.spatial import Delaunay
    w, yc = M['w'], M['yc']
    h, row = L_['spacing'] * w, L_['fold_row']
    mw = np.array(o.matrix_world)
    W = np.array([v.co[:] for v in o.data.vertices]) @ mw[:3, :3].T + mw[:3, 3]
    xs = np.concatenate([q[:, 0] for _, _, q in outlines])
    zs = np.concatenate([q[:, 1] for _, _, q in outlines])
    box = ((np.abs(bodyP[:, 0]) < np.abs(xs).max() + 1.5) & (bodyP[:, 2] > zs.min() - 1.5) & (bodyP[:, 2] < zs.max() + 1.5)
           & (bodyP[:, 1] < yc))
    FRONT_FLOOR[:] = [bodyP[:len(W)].copy(), bodyN[:len(W)].copy()]     # (the bodice's own drafted surface, whole: the front's floor in _with_front)
    own = (np.arange(len(bodyP)) < len(W))[box]       # (bodyP's first rows are the bodice's own points, then the body's)
    bodyP, bodyN = bodyP[box], bodyN[box]             # (only the chest under the lapels: the fold meets nothing else)
    # (the lapel is kept a cloth's thickness off the bodice's cloth, but only a hair off the body itself: the bodice
    # lies nearer the body than that in places, and a body that reaches through it pushes the turn's first rows apart)
    # (and the body's normals are evened over 0.4: the fronts' lap down the centre is a ridge whose side faces
    # sideways, and it pushed the lapel that lies over it along the cloth instead of off it)
    from scipy.spatial import cKDTree
    bodyN = np.array([bodyN[j].mean(0) for j in cKDTree(bodyP).query_ball_point(bodyP, 0.4)])
    bodyN /= np.maximum(np.linalg.norm(bodyN, axis=1, keepdims=True), 1e-9)
    under = ~own & (cKDTree(bodyP[own][:, [0, 2]]).query(bodyP[:, [0, 2]])[0] < 0.25)      # (the body under the bodice; the bare neck keeps the whole gap)
    bodyP = bodyP - np.where(under, L_['fold_gap'] - 0.04, 0.0)[:, None] * bodyN
    del FOLDED[:]
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bm.verts.layers.deform.verify()                   # (made now: adding a layer later would invalidate the points held)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    edge = np.array([v.is_boundary for v in bm.verts])
    ly = bm.verts.layers.int.get('front_layer')
    layer = np.array([v[ly] for v in bm.verts]) if ly is not None else np.zeros(len(W), int)
    fv = [[v.index for v in f.verts] for f in bm.faces]
    C = np.array([W[f].mean(0) for f in fv])
    P = W[:, [0, 2]]
    inv = np.linalg.inv(mw)
    made, moved, parts = [], 0, []
    for k, (B, N, poly) in enumerate(outlines):
        L = np.linalg.norm(N - B)
        u = (N - B) / L
        nperp = np.array([-u[1], u[0]])
        if nperp @ (np.array([0.0, N[1]]) - B) < 0:                      # (across the roll, into the V)
            nperp = -nperp

        def ts(X):
            return (X - B) @ u, (X - B) @ nperp
        t, sd = ts(P)
        want = (1 if N[0] < 0 else 2) if layer.any() else 0          # (this lapel's own front: see front_layers)
        mine = (layer == want) | (layer == 0) if want else np.ones(len(W), bool)
        line = np.nonzero(edge & mine & (W[:, 1] < yc) & (np.abs(sd) < 0.015) & (t > -0.015) & (t < L + 0.015))[0]
        line = line[np.argsort(t[line])]
        tc, sc = ts(C[:, [0, 2]])
        strip = np.nonzero((C[:, 1] < yc) & (sc < 0) & (sc > -2.5 * h) & (tc > -h) & (tc < L + h))[0]
        strip = np.array([f for f in strip if mine[fv[f]].all()], int)
        st_v = np.array(sorted({i for f in strip for i in fv[f]} | set(line.tolist())))
        loc = {g: j for j, g in enumerate(st_v)}
        ns = len(st_v)
        # the extension's pattern: rows parallel to the roll (close through the turn, then at the cloth's own spacing),
        # staggered, inside the outline; the outline itself set with points, so the cut edges are the drafted ones
        # (lengths along the roll are the cloth's own: the roll runs over the chest, longer than its front view, so the
        # roll's points are set out by their distance along it, and the lapel's outline is drawn to that length)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(W[line], axis=0), axis=1))])
        t0 = t[line[0]]
        grow = arc[-1] / (t[line[-1]] - t0)
        line_flat = B[None] + (t0 + arc)[:, None] * u[None]
        mt, ms = ts(_mirror(poly, B, N))
        cut = shapely.Polygon(B[None] + (t0 + (mt - t0) * grow)[:, None] * u[None] + ms[:, None] * nperp[None]).buffer(0)
        # (the turn itself takes cloth: the arc of the fold's radius. The outline is set that much further from the
        # roll, the turn's band between, so the lapel lands on its draft and not an arc short of it.)
        turn = math.radians(L_['fold_deg']) * L_['fold_radius']
        e0, e1 = B + t0 * u, B + (t0 + arc[-1]) * u
        cut = shapely.unary_union([shapely.affinity.translate(cut, *(turn * nperp)),
                                   shapely.Polygon([e0, e1, e1 + turn * nperp, e0 + turn * nperp])]).buffer(1e-6).buffer(-1e-6)
        ct, cs = ts(np.array(cut.exterior.coords))
        nb = int(math.ceil((math.pi * L_['fold_radius'] + 2 * row) / row))
        rows = [row * j for j in range(1, nb + 1)]
        while rows[-1] < cs.max():
            rows.append(rows[-1] + h * 0.87)
        pts = []
        for j, sv in enumerate(rows):
            step = row * 1.4 if j < nb else h
            tt = np.arange(ct.min() + (0.5 * step if j % 2 else 0.0), ct.max(), step)
            pts.append(B[None] + tt[:, None] * u[None] + sv * nperp[None])
        pts = np.vstack(pts)
        # (inside the outline a second line of points follows it, so the cut edge has even triangles all along: rows
        # clipped at the edge leave fans of uneven triangles, and the cloth's thickness shows them as a toothed edge)
        band = turn + 2 * row
        ps = ts(pts)[1]
        pts = pts[shapely.contains_xy(cut.buffer(-1.15 * h), pts[:, 0], pts[:, 1])
                  | ((ps < band) & shapely.contains_xy(cut.buffer(-0.45 * row), pts[:, 0], pts[:, 1]))]
        inner = cut.buffer(-0.6 * h).exterior
        ins = np.array([inner.interpolate(d).coords[0] for d in np.arange(0, inner.length, 0.7 * h)])
        pts = np.vstack([pts, ins[ts(ins)[1] > band]])
        ring = cut.exterior
        rim = np.array([ring.interpolate(d).coords[0] for d in np.arange(0, ring.length, 0.7 * h)] + list(ring.coords[:-1]))
        rim = rim[ts(rim)[1] > 0.5 * row]
        keep = np.ones(len(rim), bool)                                   # (no two of the outline's points nearer than a third of the spacing)
        for i in range(len(rim)):
            if keep[i]:
                keep[i + 1:] &= np.linalg.norm(rim[i + 1:] - rim[i], axis=1) > 0.3 * h
        rim = rim[keep]
        ext = np.vstack([pts, rim])
        allp = np.vstack([line_flat, ext])
        tri = Delaunay(allp).simplices
        cen = allp[tri].mean(1)
        ar = 0.5 * np.abs((allp[tri[:, 1], 0] - allp[tri[:, 0], 0]) * (allp[tri[:, 2], 1] - allp[tri[:, 0], 1])
                          - (allp[tri[:, 1], 1] - allp[tri[:, 0], 1]) * (allp[tri[:, 2], 0] - allp[tri[:, 0], 0]))
        tri = tri[shapely.contains_xy(cut.buffer(1e-4), cen[:, 0], cen[:, 1]) & (ts(cen)[1] > 0) & (ar > 1e-5)]
        nl = len(line)
        to_fab = np.concatenate([[loc[g] for g in line], ns + np.arange(len(ext))]).astype(int)
        T_ext = to_fab[tri]
        T_strip = np.array([(loc[fv[f][0]], loc[fv[f][j]], loc[fv[f][j + 1]]) for f in strip for j in range(1, len(fv[f]) - 1)])
        P2 = np.vstack([P[st_v], ext])
        P2[[loc[g] for g in line]] = line_flat
        T2 = np.vstack([T_strip, T_ext])
        a_, b_, c_ = P2[T2[:, 0]], P2[T2[:, 1]], P2[T2[:, 2]]
        cr = (b_[:, 0] - a_[:, 0]) * (c_[:, 1] - a_[:, 1]) - (b_[:, 1] - a_[:, 1]) * (c_[:, 0] - a_[:, 0])
        T2[cr < 0] = T2[cr < 0][:, [0, 2, 1]]
        # the start: the extension standing off the chest from the roll (tilt), unfolded; the solver makes the fold
        lt = t0 + arc
        te, se = ts(ext)
        base = np.column_stack([np.interp(te, lt, W[line, c]) for c in range(3)])
        ax = W[line[-1]] - W[line[0]]
        ax /= np.linalg.norm(ax)
        dv = np.array([nperp[0], 0.0, nperp[1]])
        dv -= (dv @ ax) * ax
        dv /= np.linalg.norm(dv)
        nrm = np.cross(ax, dv)
        if nrm[1] > 0:
            nrm = -nrm                                                   # (off the body: toward the front)
        th = math.radians(L_['tilt'])
        start = np.vstack([W[st_v], base + se[:, None] * (math.cos(th) * dv + math.sin(th) * nrm)[None]])
        lap = np.arange(len(P2)) >= ns
        roll = np.array([B, N])
        if np.median(fo.Crease(roll, 1.0).frame(P2[lap])[2]) < 0:
            roll = roll[::-1]
        ang = math.radians(L_['fold_deg'])
        fab = xf.Fabric(start, dict(FOLD_FABRIC))
        fab.pin(np.arange(ns))                                           # (the bodice's own cloth: the strip and the roll's points)
        crease = fo.Crease(roll, ang, L_['fold_radius'])
        fab.mesh([tuple(int(i) for i in t_) for t_ in T2], stretch=L_['fold_stretch'], rest_pos=np.column_stack([P2, np.zeros(len(P2))]),
                 bend=L_.get('fold_bend', 1e-6), flat=L_.get('fold_bend', 1e-6), bend_at=fo.soft_at(P2, T2, [crease]))
        # (the cloth right beside the roll is what the lapel turns round: it is not in the lapel's way. Left in, it
        # pushes the turn's own rows a full gap off and the fold is strained apart.)
        beside = np.zeros(len(bodyP), bool)             # (beside either roll: at the break the two turns meet)
        for Bq, Nq, _ in outlines:
            uq = (Nq - Bq) / np.linalg.norm(Nq - Bq)
            dq = bodyP[:, [0, 2]] - Bq
            bt, bs = dq @ uq, dq[:, 0] * uq[1] - dq[:, 1] * uq[0]
            beside |= own & (np.abs(bs) < math.pi * L_['fold_radius'] + 0.1) & (bt > -0.3) & (bt < np.linalg.norm(Nq - Bq) + 0.3)
        use = ~beside & (own if L_.get('cloth_only', False) else True)
        # (the floor under the lapel is for the cloth past the turn: the turn's own rows rise from the roll through
        # the floor's height, and a floor under them pushes the turn open and strains it. They are held by the hinges.)
        past = np.concatenate([np.zeros(ns, bool), se > turn + row])
        floor = xf.MeshBody(bodyP[use], bodyN[use], gap=L_['fold_gap'], reach=1.0)

        def body(x, past=past, floor=floor):
            p_, t_ = x.copy(), np.zeros(len(x), bool)
            p_[past], t_[past] = floor(x[past])
            return p_, t_
        if not L_.get('body', True):
            body = None
        F = fo.Folds(fab, P2, T2)
        F.add(roll, ang, radius=L_['fold_radius'])
        # The gorge is sewn to the collar, which is sewn to the neck: on the notch lapel's steep roll the lapel's top
        # reaches up past the roll's end at X, and left free that corner hangs from nothing and flops over (a second
        # crease across the lapel, its fold 25 to 40 degrees short). Its gorge edge past the turn is held where the
        # fold as drafted lays it; the collar's seam row is set on it in turn (notch_collar).
        sewn = np.zeros(len(P2), bool)
        if L_.get('gorge') and L_.get('gorge_sewn', True):
            i0 = int(np.argmin(np.linalg.norm(poly - N, axis=1)))
            rd = (B - N) / np.linalg.norm(B - N)
            G = min((poly[i0 - 1], poly[(i0 + 1) % len(poly)]), key=lambda q_: abs((q_ - N) @ rd) / max(np.linalg.norm(q_ - N), 1e-9))
            # (the pattern carries the turn's own cloth, so the gorge is found where the fold lays it: the outline's
            # points that land on the drafted gorge, N to G, in front view)
            e2 = np.sort(np.concatenate([T2[:, [0, 1]], T2[:, [1, 2]], T2[:, [2, 0]]]), 1)
            e2, c2 = np.unique(e2, axis=0, return_counts=True)
            rimv = np.zeros(len(P2), bool)
            rimv[np.unique(e2[c2 == 1])] = True
            gv = G - N
            sd_, up_ = F._sides()
            for _, mv_, tg_, _, _ in list(F._targets(fab.pos, sd_, up_, 1.0)):
                mv_ = np.asarray(mv_)
                mv_ = np.nonzero(mv_)[0] if mv_.dtype == bool else mv_
                dq_ = tg_[:, [0, 2]] - N
                tq_ = np.clip(dq_ @ gv / (gv @ gv), 0, 1)
                hit_ = (np.linalg.norm(dq_ - tq_[:, None] * gv[None], axis=1) < 0.08) & rimv[mv_] & past[mv_]
                sewn[mv_[hit_]] = True
                tg_ = tg_[hit_].copy()
                if L_.get('gorge_on_chest', True):
                    # (the drafted fold is flat, the chest is not: toward the shoulder the chest falls away from the
                    # fold's plane, and a gorge held in that plane stood up to 0.6 off the front, the lapel's point
                    # past it up to 1.2 (3 cm): the lapel floated. The gorge seam lies on the coat's chest, and the
                    # lapel is pad-stitched to curl onto it (UKY CT-LMH.176): the held points are set down along the
                    # chest's normal, from the roll's height at the turn to the resting gap by 0.6 along the gorge)
                    jb_ = cKDTree(bodyP[use]).query(tg_)[1]
                    qn_ = bodyN[use][jb_]
                    ht_ = ((tg_ - bodyP[use][jb_]) * qn_).sum(1)
                    top_ = 2 * L_['fold_radius']
                    want_ = top_ - (top_ - L_['fold_gap']) * np.clip(tq_[hit_] * np.linalg.norm(gv) / 0.6, 0, 1)
                    tg_ -= np.maximum(ht_ - want_, 0.0)[:, None] * qn_
                    log('collar: front %d: gorge set on the chest: its held points stood %.2f to %.2f off the front, now %.2f to %.2f'
                        % (k, ht_.min(), ht_.max(), np.minimum(ht_, want_).min(), np.minimum(ht_, want_).max()))
                fab.pos[mv_[hit_]] = tg_
            log('collar: front %d: %d points of the gorge held as sewn to the collar' % (k, int(sewn.sum())))
            fab.prev[:] = fab.pos
            fab.pin(np.nonzero(sewn)[0])
        # (a fold solved before with the same cloth, body and settings is read back: the standing rule for solved folds)
        import hashlib
        import json as _json
        hh = hashlib.sha1()
        hh.update(np.packbits(sewn).tobytes())
        hh.update(np.ascontiguousarray(np.round(fab.pos[sewn], 5)).tobytes())     # (where the gorge is held: a different hold is a different fold)
        for arr in (P2, T2, start, bodyP[use], bodyN[use]):
            hh.update(np.ascontiguousarray(np.round(np.asarray(arr, float), 5)).tobytes())
        hh.update(b'floor past the band')
        hh.update(_json.dumps([{k_: v_ for k_, v_ in L_.items() if k_.startswith('fold_') or k_ in ('bake', 'tilt')},
                              FOLD_FABRIC], sort_keys=True, default=str).encode())
        cfile = os.path.join(fo.cache_dir(), 'lapel-%s.npz' % hh.hexdigest()[:20])
        if os.path.exists(cfile) and not os.environ.get('COLLAR_NO_CACHE'):
            got = np.load(cfile, allow_pickle=True)
            fab.pos[:] = got['pos']
            rep = str(got['report']) + ' (from the fold cache)'
        else:
            rep = F.bake(body=body, log=lambda *a_: None, sheet=False, **L_.get('bake', {}))   # (sheet=False: the lapels' sheet is contours.py here)
            os.makedirs(fo.cache_dir(), exist_ok=True)
            np.savez(cfile, pos=fab.pos, report=str(rep))
        log('collar: front %d: lapel extension of %d points on %d roll points (roll %.2f long on the cloth, %.2f in front view), turned back by the fold hinges: %s'
            % (k, len(ext), nl, arc[-1], t[line[-1]] - t0, str(rep).replace(chr(10), '; ')))
        X = fab.pos.copy()
        made.append((line, X[ns:], tri))
        parts.append(dict(st_v=st_v, P2=P2, T2=T2.copy(), ns=ns, roll=roll, line=line, ext=X[ns:].copy(), use=use, past=past[ns:], sewn=sewn[ns:]))
        FOLDED.append(dict(flat=P2, pos=X, T=T2.copy(), lap=lap, roll=roll, report=str(rep), body=(bodyP[use], bodyN[use])))
        moved += len(ext)
    if L_.get('whole', True):
        Wn = _with_front(W, fv, C, edge, parts, outlines, M, bodyP, bodyN, own, L_, log)
        for i in np.nonzero(np.abs(Wn - W).max(1) > 0)[0]:
            bm.verts[int(i)].co = tuple(Wn[i] @ inv[:3, :3].T + inv[:3, 3])
        made = [(ln, pt['ext'], tri) for (ln, _, tri), pt in zip(made, parts)]
        for Lk, pt in zip(FOLDED, parts):
            Lk['pos'] = np.vstack([Wn[pt['st_v']], pt['ext']])
    for line, X, tri in made:                         # (the folded extensions joined to the bodice on its own roll points)
        Xl = X @ inv[:3, :3].T + inv[:3, 3]
        bm.verts.ensure_lookup_table()
        line = [bm.verts[int(i)] for i in line]
        vs = list(line) + [bm.verts.new(tuple(c)) for c in Xl]
        # (the lapel's points are bound to the bones as the roll points next to them are: points with no weights
        # are left behind when the figure is posed or turned, and the lapel hangs in the air)
        dl = bm.verts.layers.deform.verify()
        lc = np.array([v.co[:] for v in line])
        for v, c in zip(vs[len(line):], Xl):
            for g, wgt in line[int(np.argmin(((lc - c) ** 2).sum(1)))][dl].items():
                v[dl][g] = wgt
        for t_ in tri:
            try:
                bm.faces.new([vs[int(i)] for i in t_]).smooth = True
            except ValueError:
                pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(o.data)
    bm.free()
    o.data.update()
    return moved, [(L_['pos'], L_['T']) for L_ in FOLDED]


FRONT_SOLVE = {}     # the front cloth before and after the solve with the lapels (for the contour sheet's strain map)


def _with_front(W, fv, C, edge, parts, outlines, M, bodyP, bodyN, own, L_, log=print):
    """The folds held while the bodice's front is let go: the lapels as folded (each on its own, against a front held
    still) and the front's cloth round them in one fabric, the front free from the break up between the armholes and
    held at its border (the armholes, shoulders and neck, the buttoned lap, the cloth below). The hinges keep the
    folds; whatever the turned lapels pull on the roll lines is taken up by the front's cloth, not by pins. The
    front's rest is its drafted shape; the lapels' is their flat pattern. Returns the bodice's points, the front's
    moved; parts[k]['ext'] is updated."""
    import fabric as xf
    import folds as fo
    w, yc = M['w'], M['yc']
    zb = min(B[1] for B, _, _ in outlines)
    zn = max(N[1] for _, N, _ in outlines)
    xm = max(np.abs(q[:, 0]).max() for _, _, q in outlines) + 0.8      # (inside the body's own box: see fold_front)
    # (the whole front half, hem to shoulder: the crease's pull is taken up by all of the front's cloth, not by a box
    # round the lapels whose pinned border stood 1.5 w under the break (John, 2026-11-20: "make sure it's being
    # redistributed by the entire body"). COLLAR_FRONT_BOX=1 gives the old box.)
    if os.environ.get('COLLAR_FRONT_BOX'):
        inr = (C[:, 1] < yc) & (np.abs(C[:, 0]) < xm) & (C[:, 2] > zb - 1.5 * w) & (C[:, 2] < zn + 1.0 * w)
    else:
        inr = C[:, 1] < yc
    reg = np.nonzero(inr)[0]
    R = np.array(sorted({i for f in reg for i in fv[f]}))
    outside = {i for f in np.nonzero(~inr)[0] for i in fv[f]}
    loc = -np.ones(len(W), int)
    loc[R] = np.arange(len(R))
    nR = len(R)
    lap = 0.30 * w + 0.3 * w
    rollv = set(np.concatenate([pt['line'] for pt in parts]).tolist())
    # (held: the region's border, the buttoned lap, and the cloth's own cut edges but the roll lines: the armholes
    # and the neck are sewn to the sleeves and the collar)
    # The buttons hold the fronts at their own points only, and none within a button's step of the break (John,
    # 2026-11-20: "the button is not respecting the fold which creates tension"). The lap was held as one block 1.2
    # wide from 0.2 under the break: a rigid edge right under the roll lines, which the turned lapels pulled against.
    import bpy
    from scipy.spatial import cKDTree as _KD
    bq = [np.array([v.co[:] for v in o_.data.vertices]) @ np.array(o_.matrix_world)[:3, :3].T + np.array(o_.matrix_world)[:3, 3]
          for o_ in bpy.data.objects if o_.name.startswith('Redingote bodice') and o_.name.endswith('buttons')]
    bq = np.vstack(bq)[:, [0, 2]] if bq else np.zeros((0, 2))
    bq = bq[bq[:, 1] < zb - L_.get('button_clear', 0.25) * w]
    if L_.get('button_points', True) and len(bq):
        btn = _KD(bq).query(W[R][:, [0, 2]])[0] < 0.06
        btn &= W[R][:, 1] < yc
        log('collar: buttoned at the buttons only: %d front points held under them, the highest at z %.2f (break %.2f)' % (int(btn.sum()), W[R][btn, 2].max(), zb))
    else:
        btn = (np.abs(W[R][:, 0]) < lap) & (W[R][:, 2] < zb - 0.1 * w)
    held = np.array([(i in outside) or bool(btn[j]) or (edge[i] and i not in rollv)
                     for j, i in enumerate(R)])
    offs = np.cumsum([nR] + [len(pt['ext']) for pt in parts])
    pos = np.vstack([W[R]] + [pt['ext'] for pt in parts])
    fab = xf.Fabric(pos.copy(), dict(FOLD_FABRIC))
    fab.pin(np.concatenate([np.nonzero(held)[0]] + [offs[k] + np.nonzero(pt['sewn'])[0] for k, pt in enumerate(parts)]))   # (and each lapel's gorge: sewn to the collar)
    Tr = np.array([(loc[f[0]], loc[f[j]], loc[f[j + 1]]) for k in reg for f in [fv[k]] for j in range(1, len(f) - 1)])
    fab.mesh([tuple(int(i) for i in t) for t in Tr], stretch=L_['fold_stretch'], rest_pos=pos.copy(), bend=1e-6, flat=1e-6)
    Hs, rests, comps, Fs = [], [], [], []
    ang = math.radians(L_['fold_deg'])
    for k, pt in enumerate(parts):
        ns = pt['ns']
        full = np.concatenate([loc[pt['st_v']], offs[k] + np.arange(len(pt['ext']))])
        if (full < 0).any():
            raise ValueError('lapel %d: its strip reaches outside the front that is solved' % k)
        flat = np.column_stack([pos[:, 0], pos[:, 2]])
        flat[full] = pt['P2']
        Tk = full[pt['T2']]
        ext_t = Tk[(pt['T2'] >= ns).any(1)]
        crease = fo.Crease(pt['roll'], ang, L_['fold_radius'])
        fab.mesh([tuple(int(i) for i in t) for t in ext_t], stretch=L_['fold_stretch'],
                 rest_pos=np.column_stack([flat, np.zeros(len(flat))]), bend=L_.get('fold_bend', 1e-6), flat=L_.get('fold_bend', 1e-6),
                 bend_at=fo.soft_at(flat, Tk, [crease]))
        F = fo.Folds(fab, flat, Tk)
        F.add(pt['roll'], ang, radius=L_['fold_radius'])
        Hs.append(np.concatenate(F._H))
        rests.append(np.concatenate(F._rest))
        comps.append(np.concatenate(F._comp))
        Fs.append(F)
    H, rest, comp = np.concatenate(Hs), np.concatenate(rests), np.concatenate(comps)
    batches = fo._colour(H, fab.w == 0)
    # the bodies: under the front's cloth, its own drafted surface (the taut line over the corset and what else is
    # worn under it: the cloth may lift off it or slide on it, not sink through it; on the bare body alone the front
    # fell 0.2 onto the skin); under the lapels, the front and the body as in their own fold
    # (at the lap the fronts lie one on the other: the floor there is the under layer alone. With both layers in it the
    # under front was pushed up onto the top front's drafted surface, 0.09 against the buttons' hold: strain of
    # several hundred percent down the button line once the lap was no longer held as a block)
    oP, oN = (FRONT_FLOOR[0], FRONT_FLOOR[1]) if FRONT_FLOOR and not os.environ.get('COLLAR_FRONT_BOX') else (bodyP[own], bodyN[own])
    oP, oN = oP[oP[:, 1] < yc + 1.0], oN[oP[:, 1] < yc + 1.0]      # (the box round the lapels left the cloth under z 46.4 with no floor at all: it sagged 0.1 and more)
    lapz = np.nonzero(np.abs(oP[:, 0]) < lap + 0.15)[0]
    kd_ = _KD(oP[lapz][:, [0, 2]])
    top_ = np.zeros(len(oP), bool)
    for a_, nb_ in zip(lapz, kd_.query_ball_point(oP[lapz][:, [0, 2]], 0.09)):
        top_[a_] = (oP[lapz[nb_], 1] > oP[a_, 1] + 0.04).any()      # (another layer lies behind it: it is the top one)
    log('collar: the floor at the lap is the under front: %d of %d points of the top front left out' % (int(top_.sum()), len(lapz)))
    skin = xf.MeshBody(oP[~top_], oN[~top_], gap=0.0, reach=1.0)
    use = parts[0]['use']
    for pt in parts[1:]:
        use = use & pt['use']
    cloth = xf.MeshBody(bodyP[use], bodyN[use], gap=L_['fold_gap'], reach=1.0)

    lp = nR + np.nonzero(np.concatenate([pt['past'] for pt in parts]))[0]     # (the lapels' cloth past their turns)

    def body(x):
        p = x.copy()
        t = np.zeros(len(x), bool)
        p[:nR], t[:nR] = skin(x[:nR])
        p[lp], t[lp] = cloth(x[lp])
        return p, t
    h = L_['bake']['h']
    was = fab.settings['damping']
    fab.settings['damping'] = 0.35
    far, step = 0.0, 0
    for step in range(1, L_['whole_steps'] + 1):
        fab.step(h, body, neutral=True)
        lam = np.zeros(len(H))
        for _ in range(L_['bake'].get('passes', 4)):
            Fs[0]._walk(H, rest, comp, h, batches, lam)
        p, touched = body(fab.pos.copy())
        live = touched & (fab.w > 0)
        fab.pos[live] = p[live]
        if step % 20 == 0:
            far = float(np.abs(fab.pos - fab.prev).max())
            if L_.get('debug') and step % 100 == 0:
                Cc = np.abs(fo._wrap(fo.dihedral(fab.pos, H) - rest))
                jj = np.argsort(Cc)[-3:]
                for q in jj:
                    log('    hinge off by %.1f deg (rest %.1f): points %s free %s at %s' % (
                        math.degrees(Cc[q]), math.degrees(rest[q]), H[q].tolist(), (fab.w[H[q]] > 0).tolist(),
                        np.round(fab.pos[H[q]].mean(0), 2)))
            if L_.get('debug') and step % 100 == 0:
                dm = np.linalg.norm(fab.pos - pos, axis=1)
                j = int(np.argmax(dm))
                log('  step %d: fold angles %s; moved most %.3f at point %d (%s) %s from %s; front median %.4f; fastest %.5f' % (
                    step, ', '.join('%.1f' % F.measure()[0][0] for F in Fs), dm[j], j, 'front' if j < nR else 'lapel',
                    np.round(fab.pos[j], 2), np.round(pos[j], 2), np.median(dm[:nR][~held]), far))
            if step > 200 and far < 2e-5:
                break
    fab.settings['damping'] = was
    Wn = W.copy()
    Wn[R] = fab.pos[:nR]
    for k, pt in enumerate(parts):
        pt['ext'] = fab.pos[offs[k]:offs[k + 1]].copy()
    mv = np.linalg.norm(Wn[R] - W[R], axis=1)
    E = np.unique(np.sort(np.vstack([Tr[:, [0, 1]], Tr[:, [1, 2]], Tr[:, [2, 0]]]), 1), axis=0)
    l0 = np.linalg.norm(pos[E[:, 0]] - pos[E[:, 1]], axis=1)
    ok = l0 > 0.02
    st = 100 * (np.linalg.norm(fab.pos[E[:, 0]] - fab.pos[E[:, 1]], axis=1) / np.maximum(l0, 1e-9) - 1)[ok]
    FRONT_SOLVE.clear()
    FRONT_SOLVE.update(P0=pos[:nR].copy(), P1=fab.pos[:nR].copy(), E=E[ok], strain=st, held=held)
    log('collar: the folds held with the front let go: %d front points free of %d, %d steps (fastest point %.5f); the '
        'front moved up to %.3f (median %.4f), its strain 5%% %.2f, 95%% %.2f, worst %.1f / %.1f %%; fold angles %s' % (
            int((~held).sum()), nR, step, far, mv.max(), np.median(mv[~held]), np.percentile(st, 5), np.percentile(st, 95),
            st.min(), st.max(), ', '.join('%.1f' % F.measure()[0][0] for F in Fs)))
    return Wn


# The notch collar itself is cut and sewn in tailor.py. Its roll line is ONE straight line from the lapel's break point
# to a point one side-stand (1 inch) up the neck from the neck point at the shoulder (Fabrics-Store's notched collar
# draft): where that line crosses the bodice's neck edge (X) the lapel's roll ends and the collar's begins.
def side_roll(E, M, sg, L_=None, S0=SETTINGS, lean=0.18):
    """The point the roll line is aimed at: one side-stand (LAPEL['roll_side'], 1 inch at her scale) up the neck from
    the neck point at the shoulder, leaning in as the stand does. sg: +1 her left (+x), -1 her right."""
    L_ = LAPEL if L_ is None else L_
    n = len(E)
    A = E[n // 4 if sg < 0 else 3 * n // 4]
    out = np.array([A[0], A[1] - M['yc'], 0.0])
    out /= np.linalg.norm(out)
    return A + L_['roll_side'] * M['w'] * (np.array([0, 0, 1.0]) * math.cos(S0['flare']) - out * math.sin(lean))


def roll_cross(E, M, B, sg, L_=None):
    """X: where the roll line (the break point B, in (x, z), to side_roll's point) crosses the neck edge's front.
    Returns (X in (x, z), X in space)."""
    C = side_roll(E, M, sg, L_)[[0, 2]]
    n = len(E)
    ks = range(0, n // 4) if sg < 0 else range(n - 1, 3 * n // 4, -1)
    d = C - B
    for k in ks:
        P, Q = E[k % n], E[(k + (1 if sg < 0 else -1)) % n]
        e = (Q - P)[[0, 2]]
        den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < 1e-12:
            continue
        r = P[[0, 2]] - B
        u = (r[0] * e[1] - r[1] * e[0]) / den
        v = (r[0] * d[1] - r[1] * d[0]) / den
        if 0 <= v <= 1 and u > 0:
            X3 = P + v * (Q - P)
            return X3[[0, 2]], X3
    raise ValueError('the roll line does not cross the neck edge')


def over_lapels(pos, D, laps, M, gap=0.09):
    """The collar's fall lies over the lapels (the lapel's gorge is sewn under it): any of the collar's points behind
    a lapel, or nearer it than gap, set out in front of it, and the move eased into the cloth around."""
    import fabric as xf
    P = np.vstack([lp for lp, _ in laps])
    Nn = np.vstack([xf.vertex_normals(lp, lt) for lp, lt in laps])
    Nn[Nn[:, 1] > 0] *= -1                             # (toward the front: -y)
    body = xf.MeshBody(P, Nn, gap=gap, reach=0.6)
    fall = D['UV'][:, 1] > D['st_pt']
    Q, hit = body(pos.copy())
    move = np.where((fall & hit)[:, None], Q - pos, 0.0)
    # ease the move out over the neighbouring points (a few rounds of averaging along the mesh's edges)
    T = D['T']
    E_ = np.unique(np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
    for _ in range(6):
        acc = np.zeros_like(move)
        cnt = np.zeros(len(move))
        np.add.at(acc, E_[:, 0], move[E_[:, 1]])
        np.add.at(acc, E_[:, 1], move[E_[:, 0]])
        np.add.at(cnt, E_[:, 0], 1)
        np.add.at(cnt, E_[:, 1], 1)
        avg = acc / np.maximum(cnt, 1)[:, None]
        move = np.where(np.linalg.norm(avg, axis=1, keepdims=True) > np.linalg.norm(move, axis=1, keepdims=True), avg, move)
        move[~fall] = 0.0
    return pos + move


def smooth_edges(pos, T, rounds=4, k=0.5):
    """Even out the cut edges: each point on the outline moved toward the mean of its two outline neighbours (a few
    rounds), so the solver's last jitter and the triangulation's steps do not leave a ragged edge."""
    E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    E, c = np.unique(E, axis=0, return_counts=True)
    B = E[c == 1]
    nb = {}
    for a, b in B:
        nb.setdefault(a, []).append(b)
        nb.setdefault(b, []).append(a)
    idx = np.array([i for i, v in nb.items() if len(v) == 2])
    J = np.array([nb[i] for i in idx])
    P = pos.copy()
    for _ in range(rounds):
        P[idx] += k * (0.5 * (P[J[:, 0]] + P[J[:, 1]]) - P[idx])
    return P


DRAFTED = {}


def _mirror(X, B, N):
    e = (N - B) / np.linalg.norm(N - B)
    d = X - B
    return B + 2 * (d @ e)[:, None] * e[None] - d


def front_cut(V, faces, M):
    """bodice.FRONT: the bodice front's own cut for the lapels. The collar is drafted on the bodice's neck edge and
    folded, and the lapels are drafted from it (lapel_outlines). The fronts are cut open along the two roll lines,
    from the break up to the neck (fit.trim cuts the faces the line crosses, so the edge is the roll line itself);
    fold_front drafts each lapel on from that edge and turns it back. What was drafted is kept in DRAFTED for build()."""
    import bpy
    import shapely
    import fabric as xf
    import bodice as _bodice
    used = np.array(sorted({i for f in faces for i in f}))
    Vu = V[used]
    M = dict(M)
    nk = _bodice.neckline(M)
    M['z_neck_front'], M['z_neck_side'], M['z_neck_back'] = (float(nk(np.array([t]))[0]) for t in (0.0, np.pi / 2, np.pi))
    E = neck_edge(Vu, M)
    D = draft(E, M)
    Nn = xf.vertex_normals(V, faces)[used]
    rad = np.array([Vu[:, 0], Vu[:, 1] - M['yc']]).T
    Nn = Nn * np.where(((Nn[:, :2] * rad).sum(1) < 0)[:, None], -1, 1)
    body = bpy.data.objects['Skin body']
    bm = np.array(body.matrix_world)
    BV = np.array([v.co[:] for v in body.data.vertices]) @ bm[:3, :3].T + bm[:3, 3]
    BN = np.array([v.normal[:] for v in body.data.vertices]) @ bm[:3, :3].T
    sel = BV[:, 2] > 44
    bodyP, bodyN = np.vstack([Vu, BV[sel]]), np.vstack([Nn, BN[sel]])
    pos = None if LAPEL.get('gorge') else solve(D, E, M, bodyP, bodyN)[0]
    # each front's lapel starts at its own edge at the top button: the right front's edge lies lap past the middle
    # on her left, the left front's (under it) lap past it on her right, so the two roll lines cross above the top
    # button (the V's point) and below it the fronts lie one on the other
    lapw = _bodice.RATIOS['lap'] * M['w']
    outs = lapel_outlines(E, M, D, pos, dict(LAPEL, break_x=lapw))
    r_ = [o for o in outs if o[1][0] < 0][0]         # (the right front's: its neck point on her right)
    l_ = [o for o in outs if o[1][0] >= 0][0]

    def side(P, Bq, Nq):
        return (Nq - Bq)[0] * (P[:, 1] - Bq[1]) - (Nq - Bq)[1] * (P[:, 0] - Bq[0])
    # where the rolls cross
    d1, d2 = r_[1] - r_[0], l_[1] - l_[0]
    tt = np.linalg.solve(np.array([d1, -d2]).T, l_[0] - r_[0])
    A = r_[0] + tt[0] * d1
    # the left lapel is under the right front below the right front's roll: what shows, and what is turned back, is
    # its part past that roll (from the crossing up); under the right front the left front's cloth lies flat
    far = r_[0] + 30.0 * d1 / np.linalg.norm(d1)
    near = r_[0] - 30.0 * d1 / np.linalg.norm(d1)
    sgn = np.sign(side(l_[1][None], r_[0], r_[1])[0])
    nrm = np.array([-d1[1], d1[0]]) / np.linalg.norm(d1) * sgn * 60.0
    half = shapely.Polygon([near, far, far + nrm, near + nrm])
    shown = shapely.Polygon(l_[2]).buffer(0).intersection(half)
    if shown.geom_type != 'Polygon':
        shown = max(shown.geoms, key=lambda g: g.area)
    pl = np.array(shown.exterior.coords[:-1])
    outlines = [(r_[0], r_[1], r_[2]), (A, l_[1], pl)]
    cuts = [shapely.Polygon(_mirror(poly, Bk, Nk)).buffer(0) for Bk, Nk, poly in outlines]
    DRAFTED.update(E=E, D=D, pos=pos, outlines=outlines, bodyP=bodyP, bodyN=bodyN, cuts=cuts, lap=lapw,
                   layers=[1, 2], breaks=(r_[0], l_[0]),
                   point_r=r_[2][3],
                   gorges={sg_: (o_[1], o_[2][2], roll_cross(E, M, o_[0], sg_)[1]) for o_ in outs for sg_ in [1 if o_[1][0] > 0 else -1]})
    yc = M['yc']

    def opening(Bq, Nq, xe):
        top = Bq + 1.05 * (Nq - Bq)
        poly = shapely.Polygon([Bq, top, np.array([xe, top[1]]), np.array([xe, Bq[1]])]).buffer(0)
        lo, hi = np.array(poly.bounds[:2]) - 0.5, np.array(poly.bounds[2:]) + 0.5

        def f(p):
            out = np.ones(len(p))
            nr = (p[:, 1] < yc) & (p[:, 0] > lo[0]) & (p[:, 0] < hi[0]) & (p[:, 2] > lo[1]) & (p[:, 2] < hi[1])
            if nr.any():
                x, z = p[nr, 0], p[nr, 2]
                dist = shapely.distance(poly.boundary, shapely.points(x, z))
                out[nr] = np.where(shapely.contains_xy(poly, x, z), -dist, dist)
            return out
        return f
    # (each front is cut open from its own edge at the top button, along its roll line, up to the neck)
    return dict(right=opening(r_[0], r_[1], r_[0][0] + 0.03), left=opening(l_[0], l_[1], l_[0][0] - 0.03))


def front_layers(o, M, lap):
    """Which front each point of the bodice's front belongs to, kept on the mesh as 'front_layer' (1: the right
    front, on top; 2: the left front with its underlap; 0: elsewhere), and the left front made one cloth: its
    underlap comes from the draft as a piece of its own, lying edge to edge with it, and is joined to it here."""
    import bmesh
    w, yc = M['w'], M['yc']
    mw = np.array(o.matrix_world)
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bm.verts.ensure_lookup_table()
    W = np.array([v.co[:] for v in bm.verts]) @ mw[:3, :3].T + mw[:3, 3]
    lay = bm.verts.layers.int.get('front_layer') or bm.verts.layers.int.new('front_layer')
    bm.verts.ensure_lookup_table()
    zone = (W[:, 1] < yc - 0.5 * w) & (np.abs(W[:, 0]) < 2.5 * w)
    comp = -np.ones(len(W), int)
    n = 0
    for v in bm.verts:                                # (the fronts as islands of the cloth in front: the cut parts them)
        if not zone[v.index] or comp[v.index] >= 0:
            continue
        stack = [v]
        comp[v.index] = n
        while stack:
            q = stack.pop()
            for e in q.link_edges:
                u = e.other_vert(q)
                if zone[u.index] and comp[u.index] < 0:
                    comp[u.index] = n
                    stack.append(u)
        n += 1
    for c in range(n):
        m = comp == c
        if m.sum() < 20:
            continue
        x = W[m, 0]
        inner = np.abs(x).max() < lap + 0.1          # (lies wholly between the lap lines: the underlap)
        val = 2 if inner or np.median(x) > 0 else 1
        for i in np.nonzero(m)[0]:
            bm.verts[int(i)][lay] = val
    joined = [v for v in bm.verts if v.is_boundary and v[lay] == 2 and abs(W[v.index, 0] - lap) < 0.05]
    k0 = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=joined, dist=0.03)
    bm.to_mesh(o.data)
    counts = [int(sum(1 for v in bm.verts if v[lay] == c)) for c in (1, 2)]
    bm.free()
    o.data.update()
    return n, counts, k0 - len(o.data.vertices)


def build():
    """The figure for views.py: the mannequin, the redingote bodice drafted on it (hidden unless SHOW_BODICE), and the
    collar sewn to its neckline and folded in the solver."""
    for d in ('redingote_bodice', 'redingote', 'vest', 'chemise', 'doublet', 'sorceress_pendant', 'base'):
        sys.path.insert(0, os.path.join(HERE, '..', d))
    import bpy
    import fabric as xf
    import redingote_bodice as RB
    import female
    RB.SETTLE = False
    import bodice as _bodice
    pieces = bool(os.environ.get('COLLAR_LAPEL_PIECES')) or STYLE == 'turned'
    _bodice.FRONT = None if pieces else front_cut
    res = RB.build()
    b = bpy.data.objects['Redingote bodice']
    if DRAFTED:
        print('collar: the fronts told apart (%d islands; right front %d points, left front %d), the underlap joined to the left front at %d points'
              % ((lambda r: (r[0], r[1][0], r[1][1], r[2]))(front_layers(b, RB.MEASURES, DRAFTED['lap']))), flush=True)
    mw = np.array(b.matrix_world)
    V = np.array([v.co[:] for v in b.data.vertices]) @ mw[:3, :3].T + mw[:3, 3]
    Fq = [list(p.vertices) for p in b.data.polygons]
    body = bpy.data.objects['Skin body']
    bm = np.array(body.matrix_world)
    BV = np.array([v.co[:] for v in body.data.vertices]) @ bm[:3, :3].T + bm[:3, 3]
    BN = np.array([v.normal[:] for v in body.data.vertices]) @ bm[:3, :3].T
    M = dict(RB.MEASURES)
    if DRAFTED:
        E, D = DRAFTED['E'], DRAFTED['D']
    else:
        E = neck_edge(V, M)
        D = draft(E, M)
    N = xf.vertex_normals(V, Fq)
    rad = np.array([V[:, 0], V[:, 1] - M['yc']]).T
    N = N * np.where(((N[:, :2] * rad).sum(1) < 0)[:, None], -1, 1)
    sel = BV[:, 2] > 44
    if DRAFTED:
        pos = DRAFTED['pos']
        if os.environ.get('COLLAR_FOLD_DUMP'):         # (the fold's inputs, to work on the fold without the build)
            import pickle
            with open(os.environ['COLLAR_FOLD_DUMP'], 'wb') as fh:
                pickle.dump(dict(V=V, F=Fq, outlines=DRAFTED['outlines'], M=M, bodyP=np.vstack([V, BV[sel]]),
                                 bodyN=np.vstack([N, BN[sel]]), skinP=BV, skinF=[list(p.vertices) for p in body.data.polygons],
                                 mods=[(m.type, getattr(m, 'thickness', None), getattr(m, 'offset', None)) for m in b.modifiers]), fh)
        nfl, folded = fold_front(b, DRAFTED['outlines'], M, np.vstack([V, BV[sel]]), np.vstack([N, BN[sel]]))
        print('collar: the bodice front turned back as the lapels (%d faces)' % nfl, flush=True)
        if os.environ.get('COLLAR_FOLD_OUT'):          # (for the contour sheet: python3 blender/collar/contours.py <this> sheet.png)
            import pickle
            Vf = np.array([v.co[:] for v in b.data.vertices]) @ mw[:3, :3].T + mw[:3, 3]
            with open(os.environ['COLLAR_FOLD_OUT'], 'wb') as fh:
                pickle.dump(dict(V=Vf, F=[list(p.vertices) for p in b.data.polygons], skinP=BV, M=M, folded=FOLDED,
                                 skinF=[list(p.vertices) for p in body.data.polygons], outlines=DRAFTED['outlines'],
                                 front=dict(FRONT_SOLVE), E=E, gorges=DRAFTED.get('gorges')), fh)
        if pos is None:                                # (the notch collar: cut flat as a tailor cuts it, sewn and folded)
            import tailor
            Vf = np.array([v.co[:] for v in b.data.vertices]) @ mw[:3, :3].T + mw[:3, 3]
            D = tailor.tailor_collar(E, M, Vf, [list(p.vertices) for p in b.data.polygons], BV,
                                     [list(p.vertices) for p in body.data.polygons], DRAFTED['gorges'], DRAFTED['point_r'])
            pos = D['pos']
        else:
            pos = over_lapels(pos, D, folded, M)
    else:
        pos, _ = solve(D, E, M, np.vstack([V, BV[sel]]), np.vstack([N, BN[sel]]))
        laps = lapels(E, M, np.vstack([V, BV[sel]]), np.vstack([N, BN[sel]]), D, pos)
        pos = over_lapels(pos, D, laps, M)
    pos = smooth_edges(pos, D['T'])
    if os.environ.get('COLLAR_OUT'):                   # (the collar as solved and as shown, for its contour check)
        import pickle
        with open(os.environ['COLLAR_OUT'], 'wb') as fh:
            pickle.dump(dict(pos=pos, solved=D.get('pos', DRAFTED.get('pos')), UV=D['UV'], T=D['T'], st_pt=D['st_pt'], E=E, M=M,
                             design=D['des'] if 'des' in D else design(D, E, M)[0], flat=D.get('flat'), V=V, F=Fq, skinP=BV,
                             strain=D.get('strain'), ed=D.get('ed'), kind=D.get('kind'),
                             skinF=[list(p.vertices) for p in body.data.polygons]), fh)
    me = bpy.data.meshes.new('Collar')
    me.from_pydata([tuple(p) for p in pos], [], [tuple(int(i) for i in t) for t in D['T']])
    me.update()
    o = bpy.data.objects.new('Collar', me)
    bpy.context.scene.collection.objects.link(o)
    root = female.FIG.get('root') if hasattr(female, 'FIG') else None      # (the views turn the figure by its root)
    par = b.parent or root
    if par is not None:
        o.parent = par
        o.matrix_parent_inverse = par.matrix_world.inverted()
    cloth = female.mat(female.mat_def(97, 'collar wool', female.ramp(RB.DYE, 0.10), 0.0, surface='wool', closeup='wool', form=True))
    me.materials.append(cloth)
    for p in me.polygons:
        p.use_smooth = True
    sub = o.modifiers.new('smooth', 'SUBSURF')
    sub.levels = sub.render_levels = 1
    sol = o.modifiers.new('thickness', 'SOLIDIFY')
    sol.thickness = _bodice.RATIOS['thickness'] * M['w']       # (the bodice's own cloth: the collar is as thick as the lapels it meets)
    sol.offset = 0
    for c in bpy.data.objects:                       # (the buttons above the break: the front is open there)
        if c.name.startswith('Redingote bodice') and c.name.endswith('buttons'):
            open_front([c], M, E, DRAFTED.get('D', D))
    if not DRAFTED:                                  # (the earlier way, a lapel laid on as its own piece)
        for i, (lp, lt) in enumerate(laps):
            lme = bpy.data.meshes.new('Lapel %d' % i)
            lme.from_pydata([tuple(p) for p in smooth_edges(lp, lt)], [], [tuple(int(j) for j in t) for t in lt])
            lme.update()
            lo = bpy.data.objects.new('Collar lapel %d' % i, lme)
            bpy.context.scene.collection.objects.link(lo)
            lme.materials.append(cloth)
            m1 = lo.modifiers.new('thickness', 'SOLIDIFY')
            m1.thickness, m1.offset = 0.07 * M['w'], 0
            if par is not None:
                lo.parent = par
                lo.matrix_parent_inverse = par.matrix_world.inverted()
    if os.environ.get('HIDE_COLLAR'):               # (the bodice alone, its front turned back: the collar out of sight)
        o.hide_render = True
        o.visible_camera = o.visible_shadow = o.visible_diffuse = o.visible_glossy = False
    if not os.environ.get('SHOW_BODICE'):
        b.hide_render = True
        for c in bpy.data.objects:
            if c.name.startswith('Redingote bodice'):
                c.hide_render = True
                c.visible_camera = c.visible_shadow = c.visible_diffuse = c.visible_glossy = False
    zc = M['z_neck_side']
    os.environ['HEAD_CZ'] = '%.2f' % (zc + 0.5)
    os.environ['HEAD_H'] = '%.2f' % (8.0 * M['w'])
    bpy.context.view_layer.update()
    return res
