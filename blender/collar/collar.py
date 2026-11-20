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
    h = R['spacing'] * w

    nd = R['notch'] * w                               # the front end is cut on the slant: the point stands this far past its foot

    def top(s):                                       # the cut edge's height above the neck edge, rising to the points
        u = np.clip(np.abs(s) / (0.5 * L), 0, 1)
        return st + fb + (ff - fb) * (u ** 2.0)
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
    return dict(UV=pts, T=T, L=L, stand=st, idx=idx, arc=arc, s0=-0.5 * L, smax=smax)


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
    st = D['stand']
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
    soft = np.where(np.abs(UV[:, 1] - D['stand']) < 0.09 * M['w'], S['crease'], 0.0)
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
)


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
        endpts = np.nonzero(side & (np.abs(UVc[:, 0]) > D['smax'](UVc[:, 1]) - 0.6 * L_['spacing'] * w) & (UVc[:, 1] >= D['stand']))[0]
        C3 = cpos[endpts[np.argmax(UVc[endpts, 1])]]
        R3 = cpos[endpts[np.argmin(UVc[endpts, 1])]]
        C = C3[[0, 2]]
        G = R3[[0, 2]] + L_['notch_at'] * (C - R3[[0, 2]])
        v = (C - G) * L_['point_len']
        a = math.radians(L_['notch_angle'])
        cands = [np.array([v[0] * math.cos(t) - v[1] * math.sin(t), v[0] * math.sin(t) + v[1] * math.cos(t)]) for t in (a, -a)]
        Pp = G + max(cands, key=lambda u: sg * u[0])                       # (the turn that takes it outward, the notch opening sideways)
        q = np.linspace(0, 1, 24)[1:-1]
        chord = B - Pp
        bulge = np.array([chord[1], -chord[0]]) / np.linalg.norm(chord) * L_['belly'] * w
        bulge *= np.sign(bulge[0]) * sg
        outer = Pp[None] + q[:, None] * chord[None] + (4 * q * (1 - q))[:, None] * bulge[None]
        poly = np.vstack([B, N, G, Pp, outer])
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
        UV = np.vstack([bnd, Gr])
        T = Delaunay(UV).simplices
        T = T[path.contains_points(UV[T].mean(1))]
        # lay it on the chest, the roll line (B to N) held, the rest let go onto the bodice
        X = on_chest(UV, 0.04)
        BN_ = N - B
        t_ = np.clip(((UV - B) @ BN_) / (BN_ @ BN_), 0, 1)
        dist = np.linalg.norm(UV - (B + t_[:, None] * BN_), axis=1)
        fab = xf.Fabric(X, dict(walk='gauss-seidel', iterations=4, stretch=0.0, bend=2e-4, damping=0.03, friction=0.4))
        fab.pin(np.nonzero(dist < 0.3 * h)[0])
        fab.mesh(T, stretch=1e-6, rest_pos=np.column_stack([UV, np.zeros(len(UV))]), bend=1e-6, flat=1e-6)
        body = xf.MeshBody(bodyP, bodyN, gap=0.07, reach=1.0)
        k = fab.settle(body, h=1 / 240, max_steps=600, tol=2e-4)
        log('collar: lapel settled in %d steps, %d points' % (k, len(UV)))
        out.append((fab.pos.copy(), T))
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


def over_lapels(pos, D, laps, M, gap=0.09):
    """The collar's fall lies over the lapels (the lapel's gorge is sewn under it): any of the collar's points behind
    a lapel, or nearer it than gap, set out in front of it, and the move eased into the cloth around."""
    import fabric as xf
    P = np.vstack([lp for lp, _ in laps])
    Nn = np.vstack([xf.vertex_normals(lp, lt) for lp, lt in laps])
    Nn[Nn[:, 1] > 0] *= -1                             # (toward the front: -y)
    body = xf.MeshBody(P, Nn, gap=gap, reach=0.6)
    fall = D['UV'][:, 1] > D['stand']
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
    print('collar: the bodice opened above the break, %d faces cut' % open_front(
        [c for c in bpy.data.objects if c.name.startswith('Redingote bodice') and c.type == 'MESH'], M, E, D), flush=True)
    for i, (lp, lt) in enumerate(laps):
        lme = bpy.data.meshes.new('Lapel %d' % i)
        lme.from_pydata([tuple(p) for p in smooth_edges(lp, lt)], [], [tuple(int(j) for j in t) for t in lt])
        lme.update()
        lo = bpy.data.objects.new('Collar lapel %d' % i, lme)
        bpy.context.scene.collection.objects.link(lo)
        lme.materials.append(cloth)
        for p in lme.polygons:
            p.use_smooth = True
        m1 = lo.modifiers.new('thickness', 'SOLIDIFY')
        m1.thickness, m1.offset = 0.07 * M['w'], 0
        if par is not None:
            lo.parent = par
            lo.matrix_parent_inverse = par.matrix_world.inverted()
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
