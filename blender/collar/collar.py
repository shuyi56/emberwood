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

RATIOS = dict(
    stand=0.55,            # the stand's height (neck edge to the roll line), in w
    fall_back=0.60,        # the fall's depth at the back, in w
    fall_front=1.20,       # the fall's depth at the points
    notch=0.70,            # the slant of the front end: the point stands this far past its foot at the neck edge, in w
    gap=0.45,              # the open gap between the two front ends at the neck edge, in w
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
    ds = np.linalg.norm(np.roll(E, -1, 0) - E, axis=1)
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
    e = [np.linalg.norm(A[:, i] - A[:, (i + 1) % 3], axis=1) for i in range(3)]
    area = 0.5 * np.abs((A[:, 1, 0] - A[:, 0, 0]) * (A[:, 2, 1] - A[:, 0, 1]) - (A[:, 2, 0] - A[:, 0, 0]) * (A[:, 1, 1] - A[:, 0, 1]))
    T = T[area > 0.04 * h * h]
    used = np.unique(T)
    remap = -np.ones(len(pts), int)
    remap[used] = np.arange(len(used))
    pts, T = pts[used], remap[T]
    return dict(UV=pts, T=T, L=L, stand=st, idx=idx, arc=arc, s0=-0.5 * L)


SETTINGS = dict(
    flare=0.30,            # the stand's flare outward from the neck, in radians off the vertical (a turned-down collar spreads)
    fold=1.10,             # the fall folded over, below the horizontal, in radians
    lift=0.06,             # the fall's cloth stands this far off the stand where it is turned over
    stretch=1e-6, bend=1e-6, flat=1e-6, crease=1e-6,
    gap=0.05,              # the cloth's stand-off from the bodice and the body
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
    folded = R + dd * d2 + (dd > 0) * (out * S['lift'] + np.array([0, 0, S['lift']]))
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
    sol.thickness = 0.045 * M['w']
    sol.offset = 0
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
