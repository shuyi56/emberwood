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
    edge_rounds=8,
    fold_deg=172,          # the lapel's pressed fold: how far it turns back over the roll line, in degrees
    fold_gap=0.17,         # its stand-off from the bodice under it (over the bodice's own thickness, 0.045 w)
    fold_radius=0.06,      # the fold's inside radius: the cloth's thickness turned round, not a knife edge
         # rounds of evening out the cut edges
    tilt=25,               # the lapel starts this far turned off the chest (degrees), unfolded; the solver folds the rest
    steps=1600,
)


# The styles. 'turned': the first one (kept as it was approved): a turned-down collar all round, its short points over
# small lapels. 'notch': a tailored notch collar, as in the reference: the collar is sewn on only round the back and
# sides, its stand dying away toward the front so that its roll runs on into the lapels' roll, the fronts open in a
# wide V from the neck's sides down to the top button and turned back as the lapels, the collar's ends lying along the
# gorge, and the notch between the collar's point and the lapel's.
STYLES = dict(
    turned=(dict(RATIOS), dict(LAPEL)),
    notch=(dict(RATIOS, stand=0.65, stand_front=0.12, fall_back=0.75, fall_front=1.00, notch=0.30, gap=1.8),
           dict(LAPEL, drop=1.35, notch_at=0.60, notch_angle=65, point_len=0.85, point_w=0.70, belly=0.10)),
)
STYLE = os.environ.get('COLLAR_STYLE', 'notch')
RATIOS.update(STYLES[STYLE][0])
LAPEL.update(STYLES[STYLE][1])


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
    outlines = []
    for end in (0, -1):
        foot = seg[end]
        sg = 1.0 if foot[0] > 0 else -1.0
        # the lapel as a tailor cuts it: the roll line (break B up to the neck at N), the gorge (N out along the collar's
        # seam to the notch G), the notch's other side up to the lapel's point P, and the outer edge from P back to B
        # in a soft convex curve
        B = np.array([0.0, foot[2] - L_['drop'] * w])
        N = np.array([foot[0], foot[2]])
        # in harmony with the collar: the notch G sits on the collar's own front edge (as solved), part way from the
        # roll down to the collar's point C, and the lapel's point mirrors C across the notch: the same length, turned
        # down by the notch's angle
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
        bulge = np.array([chord[1], -chord[0]]) / np.linalg.norm(chord) * L_['belly'] * w
        bulge *= np.sign(bulge[0]) * sg
        outer = Pp[None] + q[:, None] * chord[None] + (4 * q * (1 - q))[:, None] * bulge[None]
        poly = np.vstack([B, N, G, Pp, outer])
        outlines.append((B, N, poly))
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


FOLD_FABRIC = dict(walk='gauss-seidel', iterations=4, stretch=0.0, bend=2e-4, damping=0.03, friction=0.4)


def fold_front(o, outlines, M, bodyP, bodyN, L_=LAPEL, log=print):
    """Turn the bodice's own front back as the lapels: the bodice cloth inside each lapel's outline as cut (the outline
    mirrored across its roll line, in the V above the break) is the flap; it stays joined to the bodice along the roll
    line and the fabric solver turns it back over the roll onto the chest (start: as it lies, tilted a little off the
    body; rest: mirrored across the roll and laid on the chest). The rest of the V is cut away (the opening). The
    bodice's mesh is changed in place. Returns the flaps' faces cut, folded."""
    import bmesh
    import fabric as xf
    import folds as fo
    from matplotlib.path import Path
    from scipy.spatial import cKDTree
    w = M['w']
    h = L_['spacing'] * w
    mw = np.array(o.matrix_world)
    W = np.array([v.co[:] for v in o.data.vertices]) @ mw[:3, :3].T + mw[:3, 3]
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    front = bodyP[:, 1] < M['yc']
    FP, FN = bodyP[front], bodyN[front]
    tree = cKDTree(FP[:, [0, 2]])

    def on_chest(xz, lift):
        _, j = tree.query(xz, k=6)
        y = FP[j, 1].min(1)
        n = FN[j].mean(1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        return np.column_stack([xz[:, 0], y, xz[:, 1]]) + lift[:, None] * n
    C = np.array([f.calc_center_median()[:] for f in bm.faces]) @ mw[:3, :3].T + mw[:3, 3]
    fv = [[v.index for v in f.verts] for f in bm.faces]
    zb = min(B[1] for B, _, _ in outlines)
    zn = max(N[1] for _, N, _ in outlines)
    xn = max(abs(N[0]) for _, N, _ in outlines)
    inV = (C[:, 1] < M['yc']) & (C[:, 2] > zb) & (np.abs(C[:, 0]) < xn * np.clip((C[:, 2] - zb) / (zn - zb), 0, 1.25))
    owner = -np.ones(len(C), int)
    dead = inV.copy()
    moved = 0
    replace = []
    for k, (B, N, poly) in enumerate(outlines):
        e = (N - B) / np.linalg.norm(N - B)

        def mirror(X):
            d = X - B
            return B + 2 * (d @ e)[:, None] * e[None] - d

        def dist_line(X):
            t_ = np.clip(((X - B) @ (N - B)) / ((N - B) @ (N - B)), 0, 1)
            return np.linalg.norm(X - (B + t_[:, None] * (N - B)), axis=1)
        cut = Path(mirror(poly))
        xz = C[:, [0, 2]]
        flap = inV & cut.contains_points(xz) & (owner < 0)
        owner[flap] = k
        dead &= ~flap
        strip = (~inV) & (C[:, 1] < M['yc']) & (dist_line(xz) < 2.5 * h) & Path(poly).contains_points(xz)
        fl_v = sorted({i for f in np.nonzero(flap)[0] for i in fv[f]})
        st_v = sorted({i for f in np.nonzero(strip)[0] for i in fv[f]} - set(fl_v))
        idx = np.array(fl_v + st_v)
        loc = {g: j for j, g in enumerate(idx)}
        nf = len(fl_v)
        faces = [[loc[i] for i in fv[f]] for f in np.nonzero(flap | strip)[0] if all(i in loc for i in fv[f])]
        tris = [(f[0], f[j], f[j + 1]) for f in faces for j in range(1, len(f) - 1)]
        X0 = W[idx]
        # the depth of each point behind the bodice's outer face (none on a sheet; kept for a bodice built thick)
        rest = X0.copy()
        rest[:nf] = on_chest(mirror(X0[:nf][:, [0, 2]]), np.full(nf, 0.16))
        hinge = dist_line(X0[:, [0, 2]]) < 0.6 * h
        a0, a1 = on_chest(B[None], np.zeros(1))[0], on_chest(N[None], np.zeros(1))[0]
        ax = (a1 - a0) / np.linalg.norm(a1 - a0)
        th = math.radians(L_['tilt'])
        d = X0 - a0
        par = (d @ ax)[:, None] * ax[None]
        perp = d - par
        start = X0.copy()
        for sgn in (1, -1):
            cand = a0 + par + perp * math.cos(th) + sgn * np.cross(ax[None], perp) * math.sin(th)
            if (cand[:nf, 1] - X0[:nf, 1]).mean() < 0:                    # (toward the front: off the body)
                start[:nf] = cand[:nf]
                break
        start[hinge] = X0[hinge]
        if os.environ.get('COLLAR_DUMP') and k == 0:                    # (the fold's inputs, for the fold helper's test)
            import json
            os.makedirs(os.path.dirname(os.environ['COLLAR_DUMP']), exist_ok=True)
            np.savez(os.environ['COLLAR_DUMP'], flat=X0[:, [0, 2]], flat3=X0, rest=rest, T=np.array(tris), start=start,
                     pins=np.arange(nf, len(idx)), roll=np.array([B, N]), B=B, N=N, nf=nf, bodyP=bodyP, bodyN=bodyN,
                     gap=L_['fold_gap'], settings=json.dumps(dict(FOLD_FABRIC, mesh=dict(stretch=1e-6, bend=1e-6, flat=1e-6))))
        # the roll line marked on the cloth (a row of points along it, so the fold has edges to turn on), then the
        # pressed fold made by folds.py's hinges: the target eased in, the flap laid over with it, held until the
        # measured angle across the roll is within a degree of the target
        flat2 = X0[:, [0, 2]]
        P2, T2, (st2, x02), kept = fo.mark_line(flat2, np.array(tris), np.array([B, N]), [start, X0])
        lap = np.concatenate([kept < nf, np.zeros(len(P2) - len(kept), bool)])
        line_pts = np.arange(len(kept), len(P2))
        roll = np.array([B, N])
        if np.median(fo.Crease(roll, 1.0).frame(P2[lap])[2]) < 0:
            roll = roll[::-1]
        ang = math.radians(L_['fold_deg'])
        st2[line_pts] = x02[line_pts]
        fab = xf.Fabric(st2, dict(FOLD_FABRIC))
        fab.pin(np.concatenate([np.nonzero(~lap)[0]]))                   # (the strip and the roll's own points)
        crease = fo.Crease(roll, ang, L_['fold_radius'])
        fab.mesh([tuple(t) for t in T2], stretch=1e-6, rest_pos=np.column_stack([P2, np.zeros(len(P2))]), bend=1e-6,
                 flat=1e-6, bend_at=fo.soft_at(P2, T2, [crease]))
        body = xf.MeshBody(bodyP, bodyN, gap=L_['fold_gap'], reach=1.0)
        F = fo.Folds(fab, P2, T2)
        F.add(roll, ang, radius=L_['fold_radius'])
        rep = F.bake(body=body, log=lambda *a_: None)
        log('collar: bodice front %d turned back as the lapel by the fold hinges: %s' % (k, str(rep).replace(chr(10), '; ')))
        replace.append((np.nonzero(flap | strip)[0], fab.pos.copy(), T2))
        moved += int(lap.sum())
    inv = np.linalg.inv(mw)
    gone = set(np.nonzero(dead)[0].tolist())
    for fi, _, _ in replace:
        gone |= set(fi.tolist())
    bmesh.ops.delete(bm, geom=[bm.faces[i] for i in sorted(gone)], context='FACES')
    new_v = []
    for _, X, T2 in replace:                          # (the folded sheet in place of the faces it was cut from)
        Xl = X @ inv[:3, :3].T + inv[:3, 3]
        vs = [bm.verts.new(tuple(c)) for c in Xl]
        for t in T2:
            try:
                bm.faces.new([vs[i] for i in t])
            except ValueError:
                pass
        new_v += vs
    bm.verts.ensure_lookup_table()
    # (welded to the bodice where the sheet's held edge meets it: the same points, held where they lay)
    bmesh.ops.remove_doubles(bm, verts=list(set(new_v) | {v for v in bm.verts if v.is_boundary}), dist=1e-3)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    # the cut edges round the opening and the lapels evened out: a cut along the mesh's faces leaves a staircase
    bm.verts.ensure_lookup_table()
    edge_v = [v for v in bm.verts if v.is_boundary and (mw[:3, :3] @ np.array(v.co[:]) + mw[:3, 3])[1] < M['yc']
              and (mw[:3, :3] @ np.array(v.co[:]) + mw[:3, 3])[2] > zb - 0.4 * w]
    for _ in range(L_['edge_rounds']):
        new = {}
        for v in edge_v:
            nb = [e.other_vert(v) for e in v.link_edges if e.is_boundary]
            if len(nb) == 2:
                new[v] = v.co + 0.5 * ((nb[0].co + nb[1].co) * 0.5 - v.co)
        for v, c in new.items():
            v.co = c
    bm.to_mesh(o.data)
    bm.free()
    o.data.update()
    return int((owner >= 0).sum()), moved


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
    res = RB.build()
    b = bpy.data.objects['Redingote bodice']
    mw = np.array(b.matrix_world)
    V = np.array([v.co[:] for v in b.data.vertices]) @ mw[:3, :3].T + mw[:3, 3]
    Fq = [list(p.vertices) for p in b.data.polygons]
    body = bpy.data.objects['Skin body']
    bm = np.array(body.matrix_world)
    BV = np.array([v.co[:] for v in body.data.vertices]) @ bm[:3, :3].T + bm[:3, 3]
    BN = np.array([v.normal[:] for v in body.data.vertices]) @ bm[:3, :3].T
    M = dict(RB.MEASURES)
    E = neck_edge(V, M)
    D = draft(E, M)
    N = xf.vertex_normals(V, Fq)
    rad = np.array([V[:, 0], V[:, 1] - M['yc']]).T
    N = N * np.where(((N[:, :2] * rad).sum(1) < 0)[:, None], -1, 1)
    sel = BV[:, 2] > 44
    pos, _ = solve(D, E, M, np.vstack([V, BV[sel]]), np.vstack([N, BN[sel]]))
    laps = lapels(E, M, np.vstack([V, BV[sel]]), np.vstack([N, BN[sel]]), D, pos)
    pos = over_lapels(pos, D, laps, M)
    pos = smooth_edges(pos, D['T'])
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
    sol.thickness = 0.07 * M['w']
    sol.offset = 0
    for c in bpy.data.objects:                       # (the buttons above the break: the front is open there)
        if c.name.startswith('Redingote bodice') and c.name.endswith('buttons'):
            open_front([c], M, E, D)
    if os.environ.get('COLLAR_LAPEL_PIECES'):        # (the earlier way, a lapel laid on as its own piece)
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
    else:
        nfl, nv = fold_front(b, lapels.outlines, M, np.vstack([V, BV[sel]]), np.vstack([N, BN[sel]]))
        print('collar: %d faces of the bodice front turned back (%d points)' % (nfl, nv), flush=True)
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
