"""The redingote's bodice: the fitted jacket body of a riding habit of about 1885-1895, drafted on whatever body it
is given (and over whatever is worn under it: the corset).

It is the vest's own draft (vest/fit.py: the body measured as a tailor measures it, one whole piece wrapped taut
round it, the edges cut along measured lines), cut as a tailored habit bodice is:

  the fit       close everywhere: the habit was cut to the corseted figure with almost no ease, and pressed to it
  the neckline  high, a clean edge round the base of the neck, for the collar (another piece) to be sewn to
  the armholes  a clean edge along the armhole loop the sleeve's head is sewn to (redingote/armhole.py, shared with
                the sleeves; until that lands, a stand-in at the same ratios)
  the front     closed, the right front lapped over the left, a row of small domed buttons down the middle from the
                neck to the front's point
  the seams     as an 1880s bodice was cut: two darts each side of the front, run from under the bust to the hip,
                the underarm seam, a curved side-back seam from the back of the armhole into the waist, and the
                centre-back seam. Each is pressed open: a fine groove with the seam allowance standing a hair proud
                either side of it
  the basque    below the waist the bodice runs on as a short skirt: pointed in front, up over the hips, and longer
                and flared behind (the postillion). It is drafted flared and then let go on the posed body in the
                cloth solver (chemise/solver.py) with wool's behaviour, so it falls into its own few soft flutes

Every size is a share of w, the neck's half-width (the vest's unit). The measurements (and the lines the sleeves,
collar and skirt meet) are returned, and kept on the object as custom properties."""
import math
import os
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in ('vest', 'chemise', 'doublet'):
    sys.path.insert(0, os.path.join(HERE, '..', _d))
sys.path.insert(0, HERE)
import fit  # noqa: E402
import solver  # noqa: E402
import cut  # noqa: E402

RATIOS = dict(
    fit.RATIOS,
    ease=0.035,              # body (or corset) to cloth, in w: a tailored habit, pressed to the figure
    thickness=0.045,         # the wool and its lining, in w
    neck_rise=0.10,          # the neckline above the neck's base line (pit, side neck, back neck), in w
    # the armhole (the sleeves thread's numbers: redingote/armhole.py)
    shoulder=0.97,           # the shoulder seam ends at this share of the way out to the shoulder point
    scye_drop=0.35,          # the armhole's bottom below the armpit, in w
    scoop_front=0.30,
    scoop_back=0.18,
    # the basque: the hem's depth below the waist, in w
    point=1.55,              # at the front's point
    point_width=0.95,        # the front point's half-width, in radians round from the middle
    hip=0.70,                # over the hips
    tail=1.80,               # at the back's middle (the postillion)
    tail_from=1.95,          # the postillion's sides, in radians round from the front: it deepens from here...
    tail_to=2.65,            # ...to its full depth here
    flare_front=0.04,        # how far the basque stands off its taut line at the hem, per w below the waist
    flare_side=0.18,
    flare_back=0.28,
    bone_to=1.05,            # the basque is held as drafted (boned) round to this far from the front, in radians...
    bone_free=1.65,          # ...and let go entirely from here round to the back
    # the seams
    groove=0.045,            # a seam's groove: its depth, in w
    groove_w=0.10,           # its half-width, in w (at least two of the grid's steps, or a slanting seam steps along the grid)
    allowance=0.006,         # the pressed allowance's ridge either side, in w
    lap=0.30,                # how far the right front laps past the middle, in w
    # the buttons
    button=0.115,            # a button's radius, in w (about 6 mm)
    button_step=0.50,        # their spacing, in w
    button_dome=0.45,        # its height as a share of its radius
)
# The darts, as a tailor places them: the first runs up to the bust's apex, the second a little outside it; at the
# waist they sit at a third and at three fifths of the way from the middle to the side. Each: (its top, out from
# the apex in w; at the waist, as a share of the waist's half-width; its top below the apex and its foot below the
# waist, in w)
DARTS = ((0.0, 0.34, 0.45, 0.80), (0.75, 0.62, 0.90, 0.65))
# The cloth's behaviour for the solver: a fulled wool broadcloth, heavier and far stiffer in bending than linen, so the
# basque keeps its flare and breaks into a few broad flutes, not narrow folds. (At this figure's scale; see solver.)
# (the basque is interlined, as a habit bodice was below the waist: bending 15 kept the postillion standing, where
# 6 let it fall away from the seat as a loose flap and swing to one side)
WOOL = dict(solver.LINEN, mass=0.9, tension=200.0, compression=60.0, shear=40.0, bending=15.0,
            damping=(10.0, 10.0, 8.0, 3.0), air=1.0, friction=8.0, gap=0.06, self_gap=0.015, quality=8, frames=90)
COARSE = (0.25, 144)         # the solver's own draft of the basque: (grid step, columns round the body)


def apex(P, aw, M):
    """The bust's apex: (half-distance between the apexes, its height), from the torso's front."""
    T = P[(aw < 0.5) & (P[:, 0] > 0.15 * M['w']) & (np.abs(P[:, 2] - M['z_bust']) < 0.6 * M['w']) & (P[:, 1] < M['yc'])]
    i = int(np.argmin(T[:, 1]))
    return float(T[i, 0]), float(T[i, 2])


def neckline(M, ratios=RATIOS):
    """The neckline's height round the body: through the pit of the neck, the side neck points and the back neck point,
    raised by neck_rise. A function of the angle round from the front."""
    a = 0.25 * (M['z_pit'] + M['z_back']) + 0.5 * M['z_side']
    c, b = a - M['z_side'], 0.5 * (M['z_pit'] - M['z_back'])
    r = ratios['neck_rise'] * M['w']
    return lambda ang: a + b * np.cos(ang) + c * np.cos(2 * ang) + r


def hemline(M, ratios=RATIOS):
    """The basque's hem height round the body (angle round from the front)."""
    w, z0 = M['w'], M['z_waist']

    def z(ang):
        a = np.abs(np.asarray(ang, dtype=float))
        pt = (ratios['point'] - ratios['hip']) * np.clip(1.0 - a / ratios['point_width'], 0.0, 1.0) ** 1.6
        tl = (ratios['tail'] - ratios['hip']) * fit._ss(ratios['tail_from'], ratios['tail_to'], a)
        return z0 - w * (ratios['hip'] + pt + tl)
    return z


def _ang(p, yc):
    return np.arctan2(-p[:, 0], -(p[:, 1] - yc))


def _along_loop(f_seam, loops):
    """The shared armhole's cut, made to follow the loop itself. The seam surface's own field (armhole.armhole_cut)
    is smooth where the loop is the hull of where the skin crosses that surface, and the two part at the front of the
    underarm's U: the field left a pointed lobe of cloth running 0.3 past the loop. Here the cloth goes where the loop
    says: it stays if it lies outside the loop seen from the side (y, z), or inside the seam surface."""
    loop = np.asarray(loops['L']['loop'], dtype=float)[:, 1:3]
    a, b = loop, np.roll(loop, -1, 0)
    e = b - a
    ee = np.maximum((e * e).sum(1), 1e-12)

    def f(p):
        p = np.asarray(p, dtype=float)
        q = p[:, 1:3]
        # signed distance to the loop in the side view: positive outside it
        t = np.clip(((q[:, None, :] - a[None]) * e[None]).sum(2) / ee[None], 0.0, 1.0)
        d = np.linalg.norm(q[:, None, :] - (a[None] + t[..., None] * e[None]), axis=2).min(1)
        x1, x2 = a[None, :, 0], b[None, :, 0]
        y1, y2 = a[None, :, 1], b[None, :, 1]
        cross = ((y1 > q[:, None, 1]) != (y2 > q[:, None, 1])) & \
            (q[:, None, 0] < (x2 - x1) * (q[:, None, 1] - y1) / np.where(y2 == y1, 1e-12, y2 - y1) + x1)
        inside = (cross.sum(1) % 2) == 1
        return np.maximum(np.where(inside, -d, d), f_seam(p))
    return f


def armhole_cut(P, T, aw, M, ratios=RATIOS):
    """The armhole's cutting line, positive on the cloth that stays. The sleeves' shared loop when it is there
    (redingote/armhole.py), else a stand-in: the vest's armhole at the same ratios."""
    try:
        import armhole  # noqa: F401  (the sleeves thread's module)
        if hasattr(armhole, 'armhole_cut'):   # (it has its own ratios: change them there, so the sleeve follows)
            loops = armhole.armhole(P, aw, M, T)
            return _along_loop(armhole.armhole_cut(M, loops), loops), 'shared loop'
    except ImportError:
        pass
    x_scye = fit.scye(P, aw, M, ratios)[0]
    lines, _ = fit.trim_lines(M, np.full(fit.COLS, M['z_chin']), x_scye, ratios)
    return dict(lines)['armholes'], 'stand-in'


def seams(M, ap, ratios=RATIOS):
    """The seams as distances over the cloth: a function of points giving, for each, (distance to the nearest seam,
    how strongly that seam shows there: 0 at a dart's tip, 1 along its length)."""
    w, yc = M['w'], M['yc']
    x_ap, z_ap = ap
    z_w = M['z_waist']
    z_arm = M['z_pit_arm']

    def run(z, z_top, z_foot, top, mid, foot):
        """A seam's place along its length: top at z_top, mid at the waist, foot at z_foot (smoothly)."""
        up = np.clip((z - z_w) / max(z_top - z_w, 1e-6), 0.0, 1.0)
        dn = np.clip((z_w - z) / max(z_w - z_foot, 1e-6), 0.0, 1.0)
        return np.where(z >= z_w, mid + (top - mid) * up ** 1.4, mid + (foot - mid) * dn ** 1.2)

    def f(p):
        z, x = p[:, 2], p[:, 0]
        r = np.hypot(p[:, 0], p[:, 1] - yc)
        a = _ang(p, yc)
        aa = np.abs(a)
        front = p[:, 1] < yc
        best = np.full(len(p), 1e9)
        show = np.zeros(len(p))

        def take(d, s):
            nonlocal best, show
            k = d < best
            best = np.where(k, d, best)
            show = np.where(k, s, show)
        # the darts: on the front, a pair each side, from under the bust down past the waist to the hip
        for out, waist_k, drop, foot in DARTS:
            zt, zf = z_ap - drop * w, z_w - foot * w
            x_top, x_w = x_ap + out * w, waist_k * M['waist_half']
            xs = run(z, zt, zf, x_top, x_w, x_w + 0.3 * (x_top - x_w))
            d = np.abs(np.abs(x) - xs)
            s = fit._ss(zt + 0.05 * w, zt - 0.6 * w, z) * fit._ss(zf - 0.05 * w, zf + 0.6 * w, z)
            take(np.where(front & (z < zt) & (z > zf), d, 1e9), s)
        # the underarm seam: straight down from the armhole's foot
        a_u = 1.62 + 0.06 * fit._ss(z_w + 2 * w, z_w - 2 * w, z)
        take(np.where(z < z_arm, r * np.abs(aa - a_u), 1e9), np.ones(len(p)))
        # the side-back seam: from the back of the armhole, curving in to the waist near the middle, and out again
        a_sb = run(z, z_arm + 0.45 * (M['z_sh'] - z_arm), z_w - 2.5 * w, 2.05, 2.62, 2.50)
        take(r * np.abs(aa - a_sb), np.ones(len(p)))
        # the centre-back seam
        take(r * (math.pi - aa), np.ones(len(p)))
        return best, show
    return f


def surface_lift(M, ap, ratios=RATIOS):
    """How far each point of the cut shell is raised off it: the seams' grooves and allowances, and the right front
    lapped over the left (a cloth's thickness proud, ending in a step a little past the middle)."""
    w, yc = M['w'], M['yc']
    sm = seams(M, ap, ratios)
    th = ratios['thickness'] * w
    g, gw, al = ratios['groove'] * w, ratios['groove_w'] * w, ratios['allowance'] * w
    lap = ratios['lap'] * w

    def lift(p):
        d, s = sm(p)
        # (the panels stand the groove's depth proud and the seam lies on the shell, not below it: a groove pressed in
        # under the shell went through the ease into the skin wherever there is no corset under the cloth)
        groove = g * (1.0 - np.exp(-(d / gw) ** 2)) + al * np.exp(-((d - 1.8 * gw) / gw) ** 2)
        front = fit._ss(yc - 0.5 * w, yc - 1.5 * w, p[:, 1])
        over = th * fit._ss(lap + 0.02 * w, lap - 0.04 * w, p[:, 0]) * front
        edge = -0.6 * g * np.exp(-((p[:, 0] - lap - 0.03 * w) / (0.7 * gw)) ** 2) * front
        return g + s * (groove - g) + over + edge
    return lift


def _union_hull(G, M, under, ease):
    """The shell let out wherever something worn under it (the corset) stands further out than it, along the same
    rays from the neck's axis: each point at least `ease` outside the outermost of them."""
    if not under:
        return G, 0
    pts, tris = [], []
    dg = bpy.context.evaluated_depsgraph_get()
    for o in under:
        e = o.evaluated_get(dg)
        me = e.to_mesh()
        me.calc_loop_triangles()
        n0 = len(pts)
        pts += [o.matrix_world @ v.co for v in me.vertices]
        tris += [tuple(n0 + i for i in t.vertices) for t in me.loop_triangles]
        e.to_mesh_clear()
    tree = BVHTree.FromPolygons(pts, tris)
    yc = M['yc']
    # how far each point of the shell has to be let out to clear the corset (0 where it already clears it)
    need = np.zeros(G.shape[:2])
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            p = G[i, j]
            d = Vector((p[0], p[1] - yc, 0.0))
            r = d.length
            if r < 1e-6:
                continue
            d /= r
            hit = tree.ray_cast(Vector((0.0, yc, p[2])) + d * 40.0, -d)[0]
            if hit is not None:
                need[i, j] = max((hit - Vector((0.0, yc, p[2]))).length + ease - r, 0.0)
    moved = int((need > 0).sum())
    if moved:
        # Cloth lies on the corset as a taut sheet, not as a cast of it: the boning's channels, the busk and the
        # edge of its top (a step the thickness of the corset) must not print through the wool. The need is dilated
        # to the width of a channel and then blurred, so the cloth rides over them (and is never inside them), and
        # ramps over the corset's edges.
        n_r, n_c = int(round(1.2 / fit.STEP)), int(round(0.5 / fit.STEP * G.shape[1] / 100.0))
        grown = need
        for ax, n in ((0, n_r), (1, n_c)):
            stack = [np.roll(grown, k, ax) if ax else np.pad(grown, ((n, n), (0, 0)), mode='edge')[n + k:n + k + len(grown)]
                     for k in range(-n, n + 1)]
            grown = np.max(stack, axis=0)
        need = np.maximum(fit._blur(fit._blur(grown, 0.5 * n_r, 0, False), 0.5 * n_c, 1, True), need)
        r_now = np.hypot(G[..., 0], G[..., 1] - yc)
        k = (r_now + need) / np.maximum(r_now, 1e-6)
        G[..., 0] *= k
        G[..., 1] = yc + (G[..., 1] - yc) * k
    return G, moved


def _corset_shell(G, M, corset):
    """The corset's own thickness under the cloth, where the corset is (the body already drawn in to its shape:
    habit.compress_body), read from the corset's shared numbers (shared/habit.py): its edges and its thickness."""
    import habit
    yc = M['yc']
    th = np.arctan2(-G[..., 0], -(G[..., 1] - yc))
    th2 = np.mod(th, 2 * math.pi)
    z = G[..., 2]
    t = habit.CORSET_THICK * corset.H
    edge = 0.04 * corset.H
    k = fit._ss(corset.bottom(th2) - edge, corset.bottom(th2) + edge, z) * fit._ss(corset.top(th2) + edge, corset.top(th2) - edge, z)
    rr = np.hypot(G[..., 0], G[..., 1] - yc)
    s = (rr + t * k) / np.maximum(rr, 1e-6)
    G[..., 0] *= s
    G[..., 1] = yc + (G[..., 1] - yc) * s
    return G


def draft(P, T, aw, M, hb, under, ratios=RATIOS, log=print, corset=None):
    """The whole piece, flared below the waist and let out over what is under it: the corset's mesh when there is
    one (`under`), else the corset's shell from its shared numbers (`corset`: habit.measure's). Returns
    (points, faces)."""
    w, yc = M['w'], M['yc']
    r_ = dict(ratios, hem_drop=ratios['tail'] + 0.3, hem_point=0.0)
    G, _, z_top, _, _ = fit.whole_piece(P, T, aw, M, hb, r_, log)
    G, moved = _union_hull(G, M, under, ratios['ease'] * w)
    if moved:
        log('bodice: let out over what is worn under it at %d points' % moved)
    elif corset is not None:
        G = _corset_shell(G, M, corset)
        log('bodice: over the corset\'s shell (%.3f thick, %.2f to %.2f at the front)' % (
            float(corset.H) * __import__('habit').CORSET_THICK, float(corset.bottom(0.0)), float(corset.top(0.0))))
    # the basque's flare: out from its taut line, by the depth below the waist, most behind
    a = np.arctan2(-G[..., 0], -(G[..., 1] - yc))
    aa = np.abs(a)
    k = (ratios['flare_front'] + (ratios['flare_side'] - ratios['flare_front']) * fit._ss(0.3, 1.5, aa)
         + (ratios['flare_back'] - ratios['flare_side']) * fit._ss(1.6, 2.8, aa))
    below = np.clip((M['z_waist'] - G[..., 2]) / w, 0.0, None)
    out = k * w * below ** 1.4 / (1.0 + 0.15 * below)
    rr = np.hypot(G[..., 0], G[..., 1] - yc)
    s = (rr + out) / np.maximum(rr, 1e-6)
    G[..., 0] *= s
    G[..., 1] = yc + (G[..., 1] - yc) * s
    rows, cols = G.shape[:2]
    V = G.reshape(-1, 3)
    idx = np.arange(rows * cols).reshape(rows, cols)
    faces = [list(q) for q in np.stack([idx[:-1], np.roll(idx[:-1], -1, 1), np.roll(idx[1:], -1, 1), idx[1:]], 2).reshape(-1, 4)]
    return V, faces


def cut_lines(P, T, aw, M, ratios=RATIOS):
    """The cutting lines (each positive on the cloth that stays) and what they are."""
    yc = M['yc']
    neck, hem = neckline(M, ratios), hemline(M, ratios)
    arm, kind = armhole_cut(P, T, aw, M, ratios)
    return [('hem', lambda p: p[:, 2] - hem(_ang(p, yc))),
            ('armholes', arm),
            ('neckline', lambda p: neck(_ang(p, yc)) - p[:, 2])], kind


def _settle_basque(P, T, aw, M, hb, body, rig, colliders, under, lines, ratios, log, corset=None):
    """The basque let go in the cloth solver: a coarse draft of the bodice below the bust, held above the waist,
    hung on the posed body. Returns a function giving the move it made at any point of the fine draft."""
    keep = (fit.STEP, fit.COLS)
    fit.STEP, fit.COLS = COARSE
    try:
        V, faces = draft(P, T, aw, M, hb, under, ratios, lambda *a: None, corset)
    finally:
        fit.STEP, fit.COLS = keep
    w = M['w']
    z_cap = M['z_waist'] + 1.2 * w
    for _, f in lines:
        V, faces = fit.trim(V, faces, f)
    V, faces = fit.trim(V, faces, lambda p: z_cap - p[:, 2])
    used = sorted({i for f in faces for i in f})
    remap = -np.ones(len(V), dtype=int)
    remap[used] = np.arange(len(used))
    V = V[used]
    faces = [[int(remap[i]) for i in f] for f in faces]
    trunk = np.nonzero(aw < 0.5)[0]
    kd = KDTree(len(trunk))
    for i in trunk:
        kd.insert(Vector(P[i]), int(i))
    kd.balance()
    proxy = fit._sheet('bodice basque proxy', [(V, faces)], M, None, body, kd, rig, lift=0.0, thickness=0.0)
    drafted = np.array([v.co[:] for v in proxy.data.vertices])
    # held as drafted above the waist (the bodice is pressed to the corset there), let go by the basque's top; and
    # held all the way down in front, where the bodice was boned along its darts and seams to the point
    a = np.abs(_ang(drafted, M['yc']))
    boned = fit._ss(ratios['bone_free'], ratios['bone_to'], a)
    pin = np.maximum(fit._ss(M['z_waist'] - 0.35 * w, M['z_waist'] + 0.05 * w, drafted[:, 2]), boned)
    solver.settle(proxy, rig, [body, *colliders, *under], pinned=pin, P=WOOL,
                  log=lambda s: log(s.replace('chemise:', 'bodice basque:')))
    settled = np.array([v.co[:] for v in proxy.data.vertices])
    move = settled - drafted
    me = proxy.data
    tris = []
    me.calc_loop_triangles()
    tris = [tuple(t.vertices) for t in me.loop_triangles]
    tree = BVHTree.FromPolygons([Vector(p) for p in drafted], tris)
    bpy.data.objects.remove(proxy)
    bpy.data.meshes.remove(me)
    log('bodice basque: the hem moved %.2f on average (%.2f at most) as it settled' % (
        float(np.linalg.norm(move[drafted[:, 2] < M['z_waist'] - 0.6 * w], axis=1).mean()), float(np.linalg.norm(move, axis=1).max())))

    def at(p):
        out = np.zeros_like(p)
        for k, q in enumerate(p):
            if q[2] > z_cap:
                continue
            loc, _, fi, dist = tree.find_nearest(Vector(q), 1.0)
            if loc is None:
                continue
            a, b, c = (Vector(drafted[i]) for i in tris[fi])
            # (barycentric weights of the nearest point on the triangle)
            v0, v1, v2 = b - a, c - a, loc - a
            d00, d01, d11, d20, d21 = v0.dot(v0), v0.dot(v1), v1.dot(v1), v2.dot(v0), v2.dot(v1)
            den = d00 * d11 - d01 * d01
            if abs(den) < 1e-12:
                out[k] = move[tris[fi][0]]
                continue
            u = (d11 * d20 - d01 * d21) / den
            v = (d00 * d21 - d01 * d20) / den
            out[k] = (1 - u - v) * move[tris[fi][0]] + u * move[tris[fi][1]] + v * move[tris[fi][2]]
        return out
    return at


def buttons(S, M, ratios, hem, neck, log=print):
    """The buttons down the front: small domes on the lapped edge's line, the neck to the front's point."""
    w, yc = M['w'], M['yc']
    rb = ratios['button'] * w
    z1 = float(neck(np.array([0.0]))[0]) - 0.40 * w
    z0 = float(hem(np.array([0.0]))[0]) + 0.55 * w
    n = max(2, int(round((z1 - z0) / (ratios['button_step'] * w))) + 1)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=1.0)
    bv = np.array([v.co[:] for v in bm.verts])
    bf = [[v.index for v in f.verts] for f in bm.faces]
    bm.free()
    bv = bv[:, :] * np.array([1.0, 1.0, ratios['button_dome']])
    pieces = []
    for z in np.linspace(z1, z0, n):
        p, nr = S.cast(Vector((0.0, yc - 40.0, float(z))), Vector((0.0, 1.0, 0.0)))
        if p is None:
            continue
        a, b = cut._frame(nr)
        R = np.array([a[:], b[:], nr[:]]).T          # (local z along the cloth's normal)
        pieces.append(((bv * rb) @ R.T + np.array(p[:]) + np.array(nr[:]) * (0.05 * rb), bf))
    log('bodice: %d buttons down the front' % len(pieces))
    return pieces


def build_bodice(body, head_objs=(), rig=None, cloth=None, button=None, under=(), colliders=(), name='Redingote bodice',
                 ratios=RATIOS, settle=True, corset=None, log=print):
    """Measure the body, draft the whole piece over it (and over the corset: its mesh, `under`, or its shared numbers,
    `corset`, habit.measure's, on a body already drawn in by habit.compress_body), cut it, let the basque settle,
    press the seams, and set the buttons. Returns (object, measures)."""
    P, T, aw = fit.body_arrays(body)
    hp, hb = None, None
    if head_objs:
        pts, tris = [], []
        for o in head_objs:
            base = len(pts)
            pts += [o.matrix_world @ v.co for v in o.data.vertices]
            o.data.calc_loop_triangles()
            tris += [tuple(base + i for i in t.vertices) for t in o.data.loop_triangles]
        hp = np.array([p[:] for p in pts])
        hb = BVHTree.FromPolygons(pts, tris)
    M = fit.measure(P, aw, hp, ratios)
    w = M['w']
    ap = apex(P, aw, M)
    M['apex_x'], M['apex_z'] = ap
    log('bodice measures: w %.2f, waist %.2f (half %.2f), bust %.2f (half %.2f), apex x %.2f, shoulder point %.2f, %.2f'
        % (w, M['z_waist'], M['waist_half'], M['z_bust'], M['bust_half'], ap[0], M['x_sh'], M['z_sh']))
    lines, kind = cut_lines(P, T, aw, M, ratios)
    log('bodice: armhole from the %s' % kind)
    V, faces = draft(P, T, aw, M, hb, list(under), ratios, log, corset)
    for nm, f in lines:
        V, faces = fit.trim(V, faces, f)
        log('bodice: trimmed %s, %d faces left' % (nm, len(faces)))
    if settle and rig is not None:
        move = _settle_basque(P, T, aw, M, hb, body, rig, list(colliders), list(under), lines, ratios, log, corset)
        used = sorted({i for f in faces for i in f})
        V = V.copy()
        V[used] = V[used] + move(V[used])
    trunk = np.nonzero(aw < 0.5)[0]
    kd = KDTree(len(trunk))
    for i in trunk:
        kd.insert(Vector(P[i]), int(i))
    kd.balance()
    th = ratios['thickness'] * w
    o = fit._sheet(name, [(V, faces, surface_lift(M, ap, ratios))], M, cloth, body, kd, rig, lift=0.0, thickness=th)
    girth = 2 * math.pi * float(np.hypot(V[:, 0], V[:, 1] - M['yc'])[np.abs(V[:, 2] - M['z_bust']) < 0.1].mean())
    cut._cloth_uv(o, M, girth / (2 * math.pi))
    neck, hem = neckline(M, ratios), hemline(M, ratios)
    if button is not None:
        pieces = buttons(cut.Surface(o, M, th), M, ratios, hem, neck, log)
        if pieces:
            fit._sheet(name + ' buttons', pieces, M, button, body, kd, rig, lift=0.0, thickness=0.0)
    # the lines the other pieces meet, and the measures, kept on the object
    M['z_hem_front'], M['z_hem_side'], M['z_hem_back'] = (float(hem(np.array([a]))[0]) for a in (0.0, math.pi / 2, math.pi))
    M['z_neck_front'], M['z_neck_side'], M['z_neck_back'] = (float(neck(np.array([a]))[0]) for a in (0.0, math.pi / 2, math.pi))
    M['z_hem'] = M['z_hem_back']
    for k, v in M.items():
        if isinstance(v, (int, float)):
            o['bodice_' + k] = float(v)
    log('bodice: hem %.2f front, %.2f side, %.2f back; neckline %.2f front, %.2f side, %.2f back' % (
        M['z_hem_front'], M['z_hem_side'], M['z_hem_back'], M['z_neck_front'], M['z_neck_side'], M['z_neck_back']))
    return o, M
