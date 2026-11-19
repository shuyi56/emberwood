"""Hose drafted on whatever body it is given: close-fitting legs of bias-cut wool, from the waist to the feet.

Nothing here is placed by hand. The legs are measured first (measure_legs(): the waist, hip, seat, crotch, thigh,
knee, calf, ankle and foot, read off the body's rest shape and its leg bones), and every size of the hose is a share
of the head's height H, the figure's own unit (anatomy.py builds the body from it), so the hose re-fits when the body
or the head changes.

Hose is the body's own surface let out, because that is what hose is: cut on the bias, wool has enough give to lie on
the leg everywhere it is convex. So the garment is made in four steps:

  1. the shell     the skin of the pelvis, legs and feet, each point moved out along the skin's normal by the ease
                   (more over the knee and the ankle, where hose is cut with room to bend), then pulled taut across
                   the hollows cloth cannot follow (the cleft of the seat, behind the knee, beside the heel cord): a
                   membrane that may only move outward, so it bridges a hollow and leaves every convex form alone.
  2. the creases   the room left at the knee and the ankle falls into a few soft creases: U-swags behind the knee,
                   rings over the instep. They only take up the extra ease, so the cloth never touches the skin.
  3. the trims     the top line at the waist (a little lower in front), and with FOOT='open' or 'stirrup' the hem at
                   the ankle and the stirrup under the arch: cuts through the shell along measured lines (fit.trim,
                   the vest's own cutter, so an edge is exact and needs no seam).
  4. the rig       each point keeps the bone weights of the skin point it came from (cut points take their nearest),
                   with the same armature settings as the skin, so the hose bends exactly as the leg bends.

The pattern is laid out as the tailor cuts it: each leg is one piece with a seam up the back of the leg (and, footed,
under the heel and along the sole), the two pieces joined by a seam down the middle of the seat and the front. The
UV map is that pattern (u across the piece, v down the leg, both in model units), so the cloth's weave runs true on
every piece; the seams are given to the shader as the 'seam' attribute (the distance to the nearest seam)."""
import math

import bpy
import numpy as np
from mathutils import Vector

import fit

# Every size as a share of the head's height H (9.4 units on the mannequin; one unit is about 2.6 cm).
RATIOS = {
    'ease': 0.006,           # the gap between skin and hose (about 1.5 mm): close fitting, held by the wool's give
    'thickness': 0.0045,     # the wool's thickness (about 1.1 mm)
    'knee_ease': 0.012,      # the room over the knee, on top of the ease (a bent knee needs it; straight, it creases)
    'knee_span': 0.16,       # how far that room reaches either way from the knee's joint
    'knee_front': 0.26,      # the hose's bend from the thigh to the shin, either way from the knee's joint: over the
    'knee_back': 0.09,       # kneecap a long band (the stretch spread out), behind the knee a short one (it folds)
    'ankle_ease': 0.010,     # the room at the ankle, for the foot to flex
    'ankle_span': 0.13,
    'bridge': 0.09,          # the most the cloth stands off a hollow it bridges
    'top_drop': 0.10,        # the top line in front, below the waist (it rises to the waist at the back)
    'top_back': 0.03,        # ... and above the waist at the back
    'band': 0.11,            # the turned-down band at the top, its depth
    'band_lift': 0.6,        # how far the band stands out, in thicknesses (the turned cloth is doubled)
    'hem_above': 0.17,       # FOOT 'open' or 'stirrup': the hem above the ankle's joint
    'stirrup': 0.12,         # the stirrup's half width along the foot
    'stirrup_at': 0.42,      # where it crosses the foot, as a share of the way from the heel to the toe
    'crease_depth': 0.7,     # how much of the extra ease a crease takes up
    'crease_width': 0.04,    # a crease's half width (soft: a narrower one reads as a seam)
}
FOOT = 'footed'              # 'footed' (period joined hose, the whole foot), 'stirrup' or 'open'
ARM_GROUPS = ('UpperArm', 'Forearm', 'Hand')
RELAX_ROUNDS = 30


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _girth(pts):
    """The girth of a level section as a tape measures it: round its convex hull."""
    if len(pts) < 3:
        return 0.0
    h = np.array(fit._hull(pts))
    return float(np.linalg.norm(h - np.roll(h, 1, 0), axis=1).sum())


def body_data(body):
    """The body's rest shape: points, normals, faces, edges, and its bone weights as a table [point, group]."""
    me = body.data
    n = len(me.vertices)
    P = np.empty(n * 3)
    me.vertices.foreach_get('co', P)
    P = P.reshape(-1, 3)
    N = np.empty(n * 3)
    me.vertices.foreach_get('normal', N)
    N = N.reshape(-1, 3)
    mw = np.array(body.matrix_world)
    P = P @ mw[:3, :3].T + mw[:3, 3]
    N = N @ mw[:3, :3].T
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    E = np.empty(len(me.edges) * 2, dtype=np.int64)
    me.edges.foreach_get('vertices', E)
    E = E.reshape(-1, 2)
    faces = [tuple(p.vertices) for p in me.polygons]
    names = [g.name for g in body.vertex_groups]
    W = np.zeros((n, len(names)), dtype=np.float32)
    for v in me.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    return dict(P=P, N=N, E=E, faces=faces, W=W, groups=names)


def measure_legs(B, rig, H, ratios=RATIOS):
    """The tailor's measurements of the legs, from the skin's rest shape and the leg bones (rest), in model units:

    H                   the head's height, the unit every ratio is a share of
    hip_z, knee_z, ankle_z   the leg's joints (the bones' heads)
    leg_x               the hip joint's distance out from the middle
    waist_z, waist      the narrowest section of the trunk between the hips and the bust (the arms left out), its girth
    hips_z, hips        the widest section of the hips, and its girth
    seat_z              where the seat stands furthest back
    crotch_z            the lowest point of the trunk in the middle
    thigh, knee, calf, calf_z, ankle, ankle_min_z   girths of one leg (the thigh just under the crotch, the knee at its
                        joint, the calf at its fullest, the ankle at its narrowest above the ankle bones)
    heel_y, toe_y, foot_x   the foot's length along y, and its middle across"""
    P, W, names = B['P'], B['W'], B['groups']
    col = {n: i for i, n in enumerate(names)}
    arm = sum(W[:, col[n]] for n in names if n.split('.')[0] in ARM_GROUPS)
    bones = rig.data.bones
    M = {'H': float(H)}
    M['hip_z'] = float(bones['Thigh.R'].head_local.z)
    M['knee_z'] = float(bones['Shin.R'].head_local.z)
    M['ankle_z'] = float(bones['Foot.R'].head_local.z)
    M['leg_x'] = float(abs(bones['Thigh.R'].head_local.x))
    trunk = arm < 0.3
    mid = trunk & (np.abs(P[:, 0]) < 0.25)
    M['crotch_z'] = float(P[mid & (P[:, 2] < M['hip_z']) & (P[:, 2] > M['knee_z']), 2].min())

    def section(z, sel, dz=0.07):
        m = sel & (np.abs(P[:, 2] - z) < dz)
        return P[m][:, :2]
    zs = np.arange(M['hip_z'], M['hip_z'] + 1.4 * H, 0.2)
    g = [_girth(section(z, trunk)) for z in zs]
    i = int(np.argmin(g))
    M['waist_z'], M['waist'] = float(zs[i]), g[i]
    zs = np.arange(M['crotch_z'] + 0.3, M['waist_z'], 0.2)
    g = [_girth(section(z, trunk)) for z in zs]
    i = int(np.argmax(g))
    M['hips_z'], M['hips'] = float(zs[i]), g[i]
    back = [section(z, trunk & (np.abs(P[:, 0]) < 2.0))[:, 1].max() for z in zs]
    M['seat_z'] = float(zs[int(np.argmax(back))])
    right = trunk & (P[:, 0] < 0)
    M['thigh'] = _girth(section(M['crotch_z'] - 0.06 * H, right))
    M['knee'] = _girth(section(M['knee_z'], right))
    zs = np.arange(M['ankle_z'] + 0.2 * H, M['knee_z'] - 0.1 * H, 0.1)
    g = [_girth(section(z, right)) for z in zs]
    i = int(np.argmax(g))
    M['calf_z'], M['calf'] = float(zs[i]), g[i]
    zs = np.arange(M['ankle_z'] + 0.05 * H, M['calf_z'], 0.1)
    g = [_girth(section(z, right)) for z in zs]
    i = int(np.argmin(g))
    M['ankle_min_z'], M['ankle'] = float(zs[i]), g[i]
    foot = right & (P[:, 2] < M['ankle_z'] - 0.05 * H)
    M['heel_y'], M['toe_y'] = float(P[foot, 1].max()), float(P[foot, 1].min())
    M['foot_x'] = float(abs(P[foot, 0].mean()))
    return M


def report(M, log=print):
    """The measurements in head heights and in centimetres (a 165 cm figure 63 units tall), for the canon check."""
    cm = 165.0 / 63.0
    H = M['H']
    parts = []
    for k in ('waist', 'hips', 'thigh', 'knee', 'calf', 'ankle'):
        parts.append('%s %.2f H (%.0f cm)' % (k, M[k] / H, M[k] * cm))
    log('hose measures: ' + ', '.join(parts))
    log('hose heights: ' + ', '.join('%s %.2f' % (k, M[k]) for k in
                                     ('waist_z', 'hips_z', 'seat_z', 'hip_z', 'crotch_z', 'knee_z', 'calf_z',
                                      'ankle_min_z', 'ankle_z')) +
        ', foot %.2f long (%.0f cm)' % (M['heel_y'] - M['toe_y'], (M['heel_y'] - M['toe_y']) * cm))


def _spine(M, s):
    """The middle line of one leg's piece, waist to toe (s = 1 her right, at -x): straight up from the hip over the
    pelvis, down the thigh and shin, and along the foot, its corner at the ankle rounded. Returns points (k, 3)."""
    x = -s * M['leg_x']
    fx = -s * M['foot_x']
    top = M['waist_z'] + 0.4 * M['H']
    pts = [(x, 0.0, top), (x, 0.0, M['hip_z']), (x, 0.0, M['knee_z']), (0.5 * (x + fx), 0.0, M['ankle_z']),
           (fx, M['toe_y'] + 0.15 * (M['heel_y'] - M['toe_y']), M['ankle_z'] - 0.22 * M['H'])]
    P = np.array(pts, dtype=float)
    dense = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(2, int(np.linalg.norm(b - a) / 0.4))
        for t in np.linspace(0, 1, n + 1)[1:]:
            dense.append(a + (b - a) * t)
    P = np.array(dense)
    for _ in range(4):           # (Chaikin: rounds the knee a little and the ankle's corner more)
        Q = 0.75 * P[:-1] + 0.25 * P[1:]
        R = 0.25 * P[:-1] + 0.75 * P[1:]
        P = np.concatenate([P[:1], np.stack([Q, R], 1).reshape(-1, 3), P[-1:]])
    return P


def _pattern(X, spine, s):
    """Where points lie on one leg's piece: (theta, v, r). v is the length down the spine to the nearest point on it,
    theta the angle round the spine from the front of the leg (the top of the foot), positive toward the outside, so
    the seam at +-pi runs up the back of the leg; r the distance from the spine."""
    A, Bv = spine[:-1], spine[1:]
    d = Bv - A
    L = np.linalg.norm(d, axis=1)
    t_ = d / L[:, None]
    cum = np.concatenate([[0.0], np.cumsum(L)])
    # the frames carried down the spine without twist, starting from the front (-y) and the outside (-s x)
    fr = np.zeros_like(t_)
    f = np.array([0.0, -1.0, 0.0])
    for k in range(len(t_)):
        f = f - t_[k] * f.dot(t_[k])
        f /= np.linalg.norm(f)
        fr[k] = f
    best = np.full(len(X), np.inf)
    seg = np.zeros(len(X), dtype=int)
    tt = np.zeros(len(X))
    for k in range(len(A)):
        u = np.clip(((X - A[k]) @ d[k]) / (L[k] ** 2), 0.0, 1.0)
        q = A[k] + u[:, None] * d[k]
        dist = np.linalg.norm(X - q, axis=1)
        m = dist < best
        best[m], seg[m], tt[m] = dist[m], k, u[m]
    q = A[seg] + tt[:, None] * d[seg]
    off = X - q
    fv = fr[seg]
    side = np.cross(t_[seg], fv) * s           # (the outside of the leg: her right is -x)
    th = np.arctan2((off * side).sum(1), (off * fv).sum(1))
    v = cum[seg] + tt * L[seg]
    return th, v, best


def build_hose(body, M, rig=None, material=None, name='Hose', ratios=RATIOS, foot=None, log=print):
    """Draft the hose on the body from its measures M (measure_legs), and hand it to the body's bones.
    Returns the object. The body keeps a vertex group 'Under hose': the skin the hose covers, for a mask."""
    foot = foot or FOOT
    B = body_data(body)
    P, Nn, W, names = B['P'], B['N'], B['W'], B['groups']
    H = M['H']
    r = {k: v * H if k not in ('band_lift', 'stirrup_at', 'crease_depth') else v for k, v in ratios.items()}
    col = {n: i for i, n in enumerate(names)}
    arm = sum(W[:, col[n]] for n in names if n.split('.')[0] in ARM_GROUPS)

    # ---- where the hose runs: the top line, round the body (lower in front), the arms left out
    rad = np.hypot(P[:, 0], P[:, 1]) + 1e-9
    front = np.clip(-P[:, 1] / rad, -1.0, 1.0)            # 1 at the front, -1 at the back
    top_z = M['waist_z'] - r['top_drop'] * _ss(-0.2, 1.0, front) + r['top_back'] * _ss(0.0, -1.0, front)

    def f_top(X):
        rr = np.hypot(X[:, 0], X[:, 1]) + 1e-9
        fr_ = np.clip(-X[:, 1] / rr, -1.0, 1.0)
        return M['waist_z'] - r['top_drop'] * _ss(-0.2, 1.0, fr_) + r['top_back'] * _ss(0.0, -1.0, fr_) - X[:, 2]
    # the part of the skin the shell is made from: a margin past every cut, so each cut falls inside it
    region = (P[:, 2] < top_z + 0.5) & (arm < 0.02)     # (none of the arm: the hanging forearm lies against the hip)
    idx = np.nonzero(region)[0]
    remap = -np.ones(len(P), dtype=np.int64)
    remap[idx] = np.arange(len(idx))
    faces = [f for f in B['faces'] if all(region[i] for i in f)]
    faces = [[int(remap[i]) for i in f] for f in faces]
    X0, Nr, Wr = P[idx], Nn[idx], W[idx]
    E = B['E'][region[B['E'][:, 0]] & region[B['E'][:, 1]]]
    E = remap[E]
    n = len(X0)

    # ---- 1. the shell: the ease, more at the knee and the ankle
    legs = {}
    for s, side in ((1, 'R'), (-1, 'L')):
        sp = _spine(M, s)
        th, v, rr = _pattern(X0, sp, s)
        legs[side] = (sp, th, v, rr)
    z = X0[:, 2]
    knee_w = np.exp(-((z - M['knee_z']) / r['knee_span']) ** 2)
    ankle_w = np.exp(-((z - 0.5 * (M['ankle_z'] + M['ankle_min_z'])) / r['ankle_span']) ** 2)
    extra = r['knee_ease'] * knee_w + r['ankle_ease'] * ankle_w
    X = X0 + Nr * (r['ease'] + extra)[:, None]
    # the taut membrane over the hollows: each pass moves a point toward its neighbours' middle only where that is
    # outward (a hollow), never in, and never further off the skin than the bridge allows
    deg = np.bincount(E.ravel(), minlength=n).astype(float)
    lift = np.zeros(n)
    for _ in range(160):
        acc = np.zeros_like(X)
        np.add.at(acc, E[:, 0], X[E[:, 1]])
        np.add.at(acc, E[:, 1], X[E[:, 0]])
        lap = acc / np.maximum(deg, 1)[:, None] - X
        out = np.maximum((lap * Nr).sum(1), 0.0) * 0.6
        out = np.minimum(out, r['bridge'] - lift)
        lift += out
        X += Nr * out[:, None]
    log('hose: the shell, %d points; bridged %d of them over hollows, by %.3f at most' % (
        n, int((lift > 0.01).sum()), lift.max()))

    # and nowhere nearer the skin than the ease: where the two thighs' skins meet under the crotch, each one's normal
    # carries its points across the other's, so those are put back out from the skin they ended up nearest to
    from mathutils.bvhtree import BVHTree
    tree = BVHTree.FromPolygons([Vector(p) for p in P], B['faces'])
    for _ in range(3):
        moved = 0
        for i in range(n):
            loc, nrm, _, d = tree.find_nearest(Vector(X[i]), 2.0 * r['ease'])
            if loc is None:
                continue
            if (loc - Vector(X0[i])).length > 6.0 * r['ease']:
                continue        # (the other thigh's skin: a point is only ever set off the skin it came from)
            sd = d if (Vector(X[i]) - loc).dot(nrm) >= 0 else -d
            if sd < 0.95 * r['ease']:
                X[i] = np.array(loc + nrm * r['ease'])
                moved += 1
        if not moved:
            break
        log('hose: %d points put back out to the ease' % moved)

    # ---- 2. the creases in the room at the knee and the ankle
    crease = np.zeros(n)
    rng = np.random.default_rng(7)
    for s, side in ((1, 'R'), (-1, 'L')):
        sp, th, v, rr = legs[side]
        mine = (X0[:, 0] * -s > 0.0)
        back = np.clip((-np.cos(th) - 0.05) / 0.95, 0, 1)   # the back half of the leg
        frt = np.clip((np.cos(th) - 0.05) / 0.95, 0, 1)     # the front half
        # behind the knee: three U-swags, deepest in the middle, their ends rising toward the sides, a little off
        # the leg's middle and not evenly spaced
        for dz, depth, tilt in ((-0.55, 1.0, 0.08), (0.05, 0.8, -0.1), (0.7, 0.55, 0.12)):
            ofs = 0.25 * rng.uniform(-1, 1)
            cz = M['knee_z'] + dz * r['knee_span']
            u = th - math.pi * np.sign(th + 1e-9) - ofs                     # (0 at the back)
            zc = cz + 0.35 * u ** 2 + tilt * u
            g = np.exp(-((z - zc) / r['crease_width']) ** 2) * back ** 1.5 * depth
            crease = np.where(mine, np.maximum(crease, g * knee_w / knee_w.max()), crease)
        # over the instep: rings where the cloth gathers as the foot comes up, their ends falling toward the heel
        for dz, depth in ((0.0, 1.0), (0.55, 0.7), (1.1, 0.45)):
            cz = M['ankle_z'] + 0.35 + dz
            u = th
            zc = cz - 0.25 * u ** 2 + 0.06 * rng.uniform(-1, 1) * u
            g = np.exp(-((z - zc) / r['crease_width']) ** 2) * frt ** 1.2 * depth
            crease = np.where(mine, np.maximum(crease, g * ankle_w), crease)
    X -= Nr * (r['crease_depth'] * extra * np.clip(crease, 0, 1))[:, None]

    # ---- 3. the trims
    lines = [('top line', f_top)]
    if foot in ('open', 'stirrup'):
        hem = M['ankle_z'] + r['hem_above']
        if foot == 'open':
            lines.append(('ankle hem', lambda Y: Y[:, 2] - hem))
        else:
            y_mid = M['heel_y'] + ratios['stirrup_at'] * (M['toe_y'] - M['heel_y'])

            def stir(Y):
                a_ = Y[:, 2] - hem
                b_ = r['stirrup'] - np.abs(Y[:, 1] - y_mid)
                k = 0.25
                h = np.clip(0.5 + 0.5 * (a_ - b_) / k, 0, 1)
                return b_ + (a_ - b_) * h + k * h * (1 - h)          # (a smooth max: the two meet in a rounded corner)
            lines.append(('stirrup', stir))
    # A cut is judged on the skin's points (so the line lies exactly where it was measured), and everything a point
    # carries is cut with it: fit.trim interpolates whole rows, so the shell's point, its normal, its bone weights
    # and its place on both pieces' patterns ride along as columns after the skin's x, y, z.
    cols = [X0, X, Nr, Wr]
    for sd in ('R', 'L'):
        _, th, v, rr = legs[sd]
        cols += [np.cos(th)[:, None], np.sin(th)[:, None], v[:, None], rr[:, None]]
    D = np.concatenate(cols, 1)
    fs = faces
    for nm, f in lines:
        D, fs = fit.trim(D, fs, lambda A, f=f: f(A[:, :3]))
        log('hose: cut the %s, %d faces left' % (nm, len(fs)))
    Vs, V, Nr, Wr = D[:, 0:3], D[:, 3:6], D[:, 6:9], D[:, 9:9 + len(names)]
    c = 9 + len(names)
    for sd in ('R', 'L'):
        sp = legs[sd][0]
        legs[sd] = (sp, np.arctan2(D[:, c + 1], D[:, c]), D[:, c + 2], D[:, c + 3])
        c += 4
    Nr /= np.maximum(np.linalg.norm(Nr, axis=1, keepdims=True), 1e-9)
    # the band at the top: the turned-down cloth stands a little proud
    depth_below = f_top(Vs)
    band = _ss(r['band'] + 0.06, r['band'] - 0.02, depth_below)
    V = V + Nr * (band * ratios['band_lift'] * r['thickness'])[:, None]

    # ---- the mesh, its pattern (UVs) and the seams
    used = sorted({i for f in fs for i in f})
    keep = -np.ones(len(V), dtype=np.int64)
    keep[used] = np.arange(len(used))
    me = bpy.data.meshes.new(name)
    me.from_pydata(V[used].tolist(), [], [[int(keep[i]) for i in f] for f in fs])
    me.validate()
    for p in me.polygons:
        p.use_smooth = True
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    if material is not None:
        me.materials.append(material)
    Vk, Sk = V[used], Vs[used]
    th = {sd: legs[sd][1][used] for sd in legs}
    vv = {sd: legs[sd][2][used] for sd in legs}
    rr = {sd: legs[sd][3][used] for sd in legs}
    # the piece's width at each length: the girth there (u = theta times the mean radius, so u is length across)
    rmean = {}
    for sd in legs:
        mine = (Sk[:, 0] < 0) if sd == 'R' else (Sk[:, 0] > 0)
        bins = np.round(vv[sd][mine] / 0.5).astype(int)
        acc = np.bincount(bins, weights=rr[sd][mine], minlength=bins.max() + 2)
        cnt = np.bincount(bins, minlength=bins.max() + 2)
        mean = acc / np.maximum(cnt, 1)
        for i in range(1, len(mean)):
            if cnt[i] == 0:
                mean[i] = mean[i - 1]
        rmean[sd] = np.convolve(np.pad(mean, 3, mode='edge'), np.ones(7) / 7, mode='valid')
    uv = me.uv_layers.new(name='pattern')
    loops_v = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get('vertex_index', loops_v)
    UV = np.zeros((len(me.loops), 2))
    for p in me.polygons:
        ls = list(p.loop_indices)
        vs = loops_v[ls]
        sd = 'R' if Sk[vs, 0].mean() < 0 else 'L'
        t_ = th[sd][vs].copy()
        if t_.max() - t_.min() > math.pi:
            t_[t_ < 0] += 2 * math.pi
        bins = np.clip(np.round(vv[sd][vs] / 0.5).astype(int), 0, len(rmean[sd]) - 1)
        UV[ls, 0] = t_ * rmean[sd][bins] + (0.0 if sd == 'R' else 1000.0)
        UV[ls, 1] = -vv[sd][vs]
    uv.data.foreach_set('uv', UV.ravel())
    # the seams: up the back of each leg (and under the foot), and down the middle of the seat and the front
    sd_v = np.where(Sk[:, 0] < 0, 0, 1)
    th_own = np.where(sd_v == 0, th['R'], th['L'])
    r_own = np.where(sd_v == 0, rr['R'], rr['L'])
    back_d = (math.pi - np.abs(th_own)) * r_own
    centre = np.where(Sk[:, 2] > M['crotch_z'] - 0.6, np.abs(Sk[:, 0]), 1e3)
    # (the foot's sole was its own piece: the back seam stops at the heel, where it turned under)
    back_d = np.where(Sk[:, 2] > M['ankle_z'] - 0.05 * H, back_d, 1e3)
    seam = np.minimum(back_d, centre)
    a = me.attributes.new('seam', 'FLOAT', 'POINT')
    a.data.foreach_set('value', seam.astype(np.float32))
    a = me.attributes.new('top_d', 'FLOAT', 'POINT')        # (how far below the top line: the band, the eyelets)
    a.data.foreach_set('value', depth_below[used].astype(np.float32))
    a = me.attributes.new('ring_s', 'FLOAT', 'POINT')       # (the length round the waist from the front middle)
    a.data.foreach_set('value', (np.arctan2(Sk[:, 0], -Sk[:, 1]) * M['waist'] / (2 * math.pi)).astype(np.float32))
    a = me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
    a.data.foreach_set('vector', Vk.ravel().astype(np.float32))

    # ---- 4. the rig: the skin's own weights, the skin's own armature settings, but for the knee.
    # The skin hands the leg from the thigh to the shin over one even band round the knee, so in a bend the front of
    # the knee is stretched to twice its length and the back squeezed to a third. Cloth spreads its stretch: over the
    # kneecap the hose takes the bend over a long band, behind the knee over a short one (where it folds).
    Wk = Wr[used].copy()
    for sd, s in (('R', 1), ('L', -1)):
        it, ish = col.get('Thigh.' + sd), col.get('Shin.' + sd)
        if it is None or ish is None:
            continue
        share = Wk[:, it] + Wk[:, ish]
        mine = (share > 1e-4) & (Sk[:, 0] * -s > 0)
        frt = 0.5 + 0.5 * np.cos(th[sd])                    # 1 over the kneecap, 0 behind the knee
        wd = r['knee_back'] + (r['knee_front'] - r['knee_back']) * frt ** 1.5
        zz = Sk[:, 2] - M['knee_z']
        lower = _ss(wd, -wd, zz)
        near = mine & (np.abs(zz) < 1.3 * r['knee_front'])
        Wk[near, ish] = share[near] * lower[near]
        Wk[near, it] = share[near] * (1.0 - lower[near])
    for gi, nm in enumerate(names):
        if nm.split('.')[0] in ARM_GROUPS or nm == 'shoulder smooth':
            continue
        w = Wk[:, gi]
        if (w > 1e-4).any():
            g = o.vertex_groups.new(name=nm)
            for val in np.unique(np.round(w[w > 1e-4], 3)):
                g.add([int(i) for i in np.nonzero(np.abs(np.round(w, 3) - val) < 1e-6)[0]], float(val), 'REPLACE')
    if rig is not None:
        src_mod = next((m for m in body.modifiers if m.type == 'ARMATURE'), None)
        m = o.modifiers.new('rig', 'ARMATURE')
        m.object = rig
        if src_mod is not None:
            m.use_deform_preserve_volume = src_mod.use_deform_preserve_volume
            m.use_bone_envelopes = src_mod.use_bone_envelopes
    # where a joint folds the skin's linear blend squeezes the back of the knee and the front of the ankle to a
    # crease and stretches the seat; cloth does neither, it keeps its length. So over the joints the deformed hose is
    # relaxed back toward its rest proportions (a corrective smooth against the rest shape), the rest left exact.
    zk = Vk[:, 2]
    relax = np.maximum.reduce([
        np.exp(-((zk - M['knee_z']) / (1.6 * r['knee_span'])) ** 2),
        np.exp(-((zk - M['ankle_z']) / (1.6 * r['ankle_span'])) ** 2),
        _ss(M['crotch_z'] - 0.3 * H, M['crotch_z'] - 0.1 * H, zk) * _ss(M['waist_z'], M['hips_z'], zk)])
    g = o.vertex_groups.new(name='hose relax')
    for val in np.unique(np.round(relax[relax > 0.02], 2)):
        g.add([int(i) for i in np.nonzero(np.abs(np.round(relax, 2) - val) < 1e-6)[0]], float(val), 'REPLACE')
    m = o.modifiers.new('relax', 'CORRECTIVE_SMOOTH')
    m.vertex_group = 'hose relax'
    m.rest_source = 'ORCO'
    m.smooth_type = 'SIMPLE'          # (length-weighted divides by the cut's sliver edges and throws spikes)
    m.factor = 0.5
    m.iterations = int(ratios.get('relax_rounds', RELAX_ROUNDS))
    m.use_pin_boundary = True
    m = o.modifiers.new('thickness', 'SOLIDIFY')
    m.thickness = r['thickness']
    m.offset = 1.0
    m.use_rim = True
    m.use_even_offset = False         # (even thickness throws spikes off the cut's sliver triangles)
    m.use_quality_normals = True
    # the skin the hose covers, kept a little in from every edge, for a mask on the body (no skin can show through
    # a bent knee or a stretched seat where the hose is)
    cover = region.copy()
    cover[idx] = f_top(X0) > 0.6
    if foot in ('open', 'stirrup'):
        cover[idx] &= X0[:, 2] > M['ankle_z'] + r['hem_above'] + 0.6
    g = body.vertex_groups.get('Under hose') or body.vertex_groups.new(name='Under hose')
    g.add([int(i) for i in np.nonzero(cover)[0]], 1.0, 'REPLACE')
    log('hose: %d points, %d faces; %d skin points under it' % (len(me.vertices), len(me.polygons), int(cover.sum())))
    return o


def mask_body(body):
    """Hide the skin under the hose (the body's 'Under hose' group)."""
    m = body.modifiers.new('under hose', 'MASK')
    m.vertex_group = 'Under hose'
    m.invert_vertex_group = True
    return m


def fit_check(hose, body, label='', log=print, inside_log=None):
    """How the hose sits on the posed body at the current frame: the least and mean gap to the skin (the mask off),
    how many points are inside it, and how far its edges are stretched or squeezed from the rest shape, by part:
    the seat and crotch, the hips, the knees, the ankles."""
    from mathutils.bvhtree import BVHTree
    mask = next((m for m in body.modifiers if m.type == 'MASK'), None)
    if mask is not None:
        mask.show_viewport = False
    sol = hose.modifiers.get('thickness')
    if sol is not None:
        sol.show_viewport = False       # (a solidified mesh has twice the points: measure the cloth's own surface)
    dg = bpy.context.evaluated_depsgraph_get()
    be = body.evaluated_get(dg)
    bm_ = be.to_mesh()
    bm_.calc_loop_triangles()
    pts = [body.matrix_world @ v.co for v in bm_.vertices]
    tris = [tuple(t.vertices) for t in bm_.loop_triangles]
    tb = BVHTree.FromPolygons(pts, tris)
    Pb = np.array([p[:] for p in pts])
    Rb = np.array([v.co[:] for v in body.data.vertices])
    Eb = np.array([e.vertices[:] for e in bm_.edges])
    be.to_mesh_clear()
    groups = [g.name for g in body.vertex_groups]
    arm = np.zeros(len(body.data.vertices))
    for v in body.data.vertices:
        for g in v.groups:
            if groups[g.group].split('.')[0] in ARM_GROUPS:
                arm[v.index] += g.weight
    he = hose.evaluated_get(dg)
    hm = he.to_mesh()
    Q = np.array([(hose.matrix_world @ v.co)[:] for v in hm.vertices])
    Ed = np.array([e.vertices[:] for e in hm.edges])
    Fq = [p.vertices[:3] for p in hm.polygons]
    he.to_mesh_clear()
    if mask is not None:
        mask.show_viewport = True
    if sol is not None:
        sol.show_viewport = True
    R = np.array([v.co[:] for v in hose.data.vertices])
    gaps = np.empty(len(Q))
    on_arm = np.zeros(len(Q), dtype=bool)
    for i, p in enumerate(Q):
        loc, nrm, fi, d = tb.find_nearest(Vector(p))
        gaps[i] = d if (Vector(p) - loc).dot(nrm) >= 0 else -d
        on_arm[i] = arm[list(tris[fi])].mean() > 0.5
    arm_n = int((on_arm & (gaps < 0)).sum())      # (the hanging hand or arm pressing on the hip: not a fault of the fit)
    gaps = gaps[~on_arm]
    lr = np.linalg.norm(R[Ed[:, 0]] - R[Ed[:, 1]], axis=1)
    st = np.linalg.norm(Q[Ed[:, 0]] - Q[Ed[:, 1]], axis=1) / np.maximum(lr, 1e-6)
    Ed = Ed[lr > 0.03]           # (the slivers a cut leaves along its line are too short to measure a stretch by)
    st = st[lr > 0.03]
    zr = R[Ed[:, 0], 2]
    parts = {'seat and crotch': (zr > 22) & (zr < 33) & (np.abs(R[Ed[:, 0], 0]) < 2.5),
             'hips': (zr > 28) & (zr < 40) & (np.abs(R[Ed[:, 0], 0]) >= 2.5),
             'knees': (zr > 13) & (zr < 21), 'ankles': (zr > 0) & (zr < 7)}
    # folded faces: turned over against their rest facing as their own bone carries it (a foot pitched past 90
    # degrees turns every face on it past 90 degrees too, and that is not a fold)
    Ft = np.array(Fq)
    nr = np.cross(R[Ft[:, 1]] - R[Ft[:, 0]], R[Ft[:, 2]] - R[Ft[:, 0]])
    rig = next((m.object for m in hose.modifiers if m.type == 'ARMATURE' and m.object), None)
    if rig is not None:
        gname = {g.index: g.name for g in hose.vertex_groups}
        rot = {}
        for pb in rig.pose.bones:
            mm = (rig.matrix_world @ pb.matrix @ pb.bone.matrix_local.inverted()).to_3x3()
            rot[pb.name] = np.array(mm)
        dom = []
        for v in hose.data.vertices:
            best = max(((g.weight, gname[g.group]) for g in v.groups if gname[g.group] in rot), default=(0, None))
            dom.append(best[1])
        for i, f in enumerate(Ft):
            b = dom[f[0]]
            if b is not None:
                nr[i] = rot[b] @ nr[i]
    npz = np.cross(Q[Ft[:, 1]] - Q[Ft[:, 0]], Q[Ft[:, 2]] - Q[Ft[:, 0]])
    fl = (nr * npz).sum(1) < 0
    flips = int(fl.sum())
    fz = R[Ft[fl, 0], 2]
    where = {'seat': int(((fz > 22) & (fz < 33)).sum()), 'knee': int(((fz > 13) & (fz < 21)).sum()),
             'ankle': int((fz < 7).sum())}
    out = {'flipped': flips, 'gap_min': float(gaps.min()), 'gap_mean': float(gaps.mean()), 'inside': int((gaps < 0).sum()), 'arm': arm_n}
    # the skin's own stretch in the same places, for comparison: the hose can do no better than the leg under it
    if len(Rb) == len(Pb):
        lb = np.linalg.norm(Rb[Eb[:, 0]] - Rb[Eb[:, 1]], axis=1)
        sb = np.linalg.norm(Pb[Eb[:, 0]] - Pb[Eb[:, 1]], axis=1) / np.maximum(lb, 1e-6)
        zb = Rb[Eb[:, 0], 2]
        xb = np.abs(Rb[Eb[:, 0], 0])
        bparts = {'seat and crotch': (zb > 22) & (zb < 33) & (xb < 2.5), 'hips': (zb > 28) & (zb < 40) & (xb >= 2.5),
                  'knees': (zb > 13) & (zb < 21), 'ankles': (zb > 0) & (zb < 7)}
    else:
        bparts = {}
    msg = []
    for k, m in parts.items():
        if m.any():
            out[k] = (float(st[m].min()), float(st[m].max()))
            sk = ''
            bad = int(((st[m] < 0.6) | (st[m] > 1.6)).sum())
            out[k + ' bad'] = bad
            if k in bparts and bparts[k].any():
                sbk = sb[bparts[k]]
                sk = ' (skin %.2f..%.2f, %d bad)' % (np.percentile(sbk, 0.5), np.percentile(sbk, 99.5),
                                                     int(((sbk < 0.6) | (sbk > 1.6)).sum()))
            msg.append('%s %.2f..%.2f, %d bad%s' % (k, np.percentile(st[m], 0.5), np.percentile(st[m], 99.5), bad, sk))
    if inside_log is not None:
        inside_log.extend(Q[~on_arm][gaps < 0].tolist())
    log('hose fit%s: gap min %.3f mean %.3f, %d of %d points inside the skin (%d more under the arm), %d faces folded over (%s); edge stretch (0.5 to 99.5 percent; bad: under 0.6 or over 1.6) %s' % (
        (' ' + label) if label else '', out['gap_min'], out['gap_mean'], out['inside'], len(Q), arm_n, flips, ', '.join('%s %d' % kv for kv in where.items()), ', '.join(msg)))
    return out
