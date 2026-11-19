"""A linen chemise worn under the vest, drafted on whatever body it is given, from the same measurements (fit.measure).

As a seamstress makes a shift:

  * the body     cut from the same whole piece as the vest (fit.whole_piece), with a shirt's small ease in place of
                 the vest's, so it lies on the skin under the leather everywhere, and longer: to the hip. Its
                 neckline is cut clear of the vest's front opening, so the vest shows skin there and not linen.
  * the sleeves  each a plain rectangle of cloth, as a shift's sleeve is: its full width gathered into the armhole
                 at the head, and left open at the hem. The sleeve is set in under the vest's armhole. Under the
                 vest it lies flat on the shoulder; where it comes out from the vest's edge the gathers open, the
                 cloth rests on the top of the shoulder, hangs plumb from the widest part of the upper arm, and its
                 spare width falls into a few deep flutes down the arm.

Linen does not stretch and has no drape of its own to speak of: it hangs straight and breaks in crisp folds. So the
sleeve's fullness is never a balloon. It is all in the folds.

Every size is a share of w (the neck's half-width, fit's unit). The sleeve is laid out on rings round the arm: a fan
of planes turning about the armpit from the armhole seam round to level, then level rings down the arm."""
import math

import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

import fit
import linen_texture

RATIOS = {
    'ease': 0.03,            # the gap between skin and linen on the body, in w (the vest's is 0.06)
    'skin': 0.010,           # the least gap anywhere, in w
    'hem': 3.0,              # the hem below the waist, in w (to the hip)
    'v_margin': 0.20,        # the neckline lies this far under the vest's front edge, in w
    'seam_in': 0.12,         # the armhole seam lies this far under the vest's armhole edge at its deepest scoop, in w
    'sleeve_ease': 0.03,     # the gap between arm and sleeve where the sleeve lies on it, in w
    'rise': 0.55,            # the fullness arrives over this distance out from the vest's edge, in w
    'full': 0.34,            # how far the sleeve stands off the arm's widest place once it is out of the vest, in w
    'bell': 0.30,            # how much wider again it hangs by the hem, in w
    'flute': 0.55,           # the depth of the creases down the arm, in w
    'flutes': 8,             # how many creases round the sleeve
    'gather': 0.085,         # the depth of the gathers at the sleeve's head, in w
    'gathers': 24,           # how many gathers round the head
    'gather_len': 0.9,       # how far the gathers run before they have opened into the flutes, in w
    'cuff': 0.5,             # the sleeve ends this far above the wrist, in w
    'thick': 0.008,          # the linen's thickness, in w
    'vest_clear': 0.14,      # the sleeve keeps this far off the vest's surface, in w
}
STEP = 0.12                  # the sleeve's ring spacing, in units
COLS = 192                   # points round a sleeve
PITCH = 0.0139               # one thread, in units: 70 threads to the inch on a 165 cm figure 63 units tall


def _signed(bvh, p):
    loc, nrm, _, dist = bvh.find_nearest(p)
    if loc is None:
        return None, None, 1e9
    return loc, nrm, (dist if (p - loc).dot(nrm) >= 0 else -dist)


def _off_skin(V, bvh, gap):
    """Push any point nearer the skin than the gap (or inside it) out to the gap."""
    moved = 0
    for i in range(len(V)):
        p = Vector(V[i, :3])
        loc, nrm, s = _signed(bvh, p)
        if loc is not None and s < gap:
            V[i, :3] = loc + nrm * gap
            moved += 1
    return moved


def _body_cloth(P, T, aw, M, seam, ratios, log):
    """The chemise's body: points [x, y, z, u, v] (u, v the place on the flat cloth, in units) and faces."""
    w = M['w']
    r = dict(fit.RATIOS, ease=ratios['ease'], hem_drop=ratios['hem'], hem_point=0.0)
    G, neck_row, z_top, _, _ = fit.whole_piece(P, T, aw, M, None, r, lambda *a: None)
    G = G[:neck_row + 1]
    G = np.concatenate([G, G[:, :1]], 1)         # (the first column again at the end: the cloth's two edges, joined later)
    rows, cols = G.shape[:2]
    # the flat cloth: across by the bust row's own run round the body, up by the rows' spacing
    ib = int(np.argmin(np.abs(G[:, 0, 2] - M['z_bust'])))
    u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(G[ib], axis=0), axis=1))])
    uv = np.stack(np.broadcast_arrays(u[None, :], (np.arange(rows) * fit.STEP)[:, None]), 2)
    V = np.concatenate([G, uv], 2).reshape(-1, 5)
    idx = np.arange(rows * cols).reshape(rows, cols)
    faces = [list(q) for q in np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], 2).reshape(-1, 4)]
    lines, extra = fit.trim_lines(M, z_top, fit.scye(P, aw, M, fit.RATIOS)[0], fit.RATIOS)
    vest_front, z_v = lines[2][1], extra['z_v']
    z_hem = M['z_waist'] - ratios['hem'] * w
    piv, n = seam
    cuts = [('hem', lambda p: p[:, 2] - z_hem),
            # (the vest's own front line, a margin in under it, and only down to the V's point: below it the front is whole)
            ('neckline', lambda p: np.maximum(vest_front(p) - ratios['v_margin'] * w, z_v - ratios['v_margin'] * w - p[:, 2])),
            # (the armhole: the seam's plane above the armpit, each side)
            ('armholes', lambda p: np.maximum(-((np.abs(p[:, 0]) - piv[0]) * n[0] + (p[:, 2] - piv[2]) * n[2]), piv[2] - p[:, 2]))]
    for nm, f in cuts:
        V, faces = fit.trim(V, faces, f)
        log('chemise: body cut at the %s, %d faces left' % (nm, len(faces)))
    return V, faces


def _folds(rng, count, width, sharp=1.5, low=0.6, drift=0.10, wander=0.0):
    """`count` folds round the sleeve at uneven places: (centres, half-widths, depths, drifts, and how each one's
    depth comes and goes down the sleeve's length). width is a fold's half-width as a share of its spacing."""
    gaps = rng.uniform(0.6, 1.4, count)
    c = np.cumsum(gaps) / gaps.sum() * 2 * math.pi + rng.uniform(0, 2 * math.pi)
    return (c, width * math.pi * gaps / gaps.sum(), rng.uniform(low, 1.0, count), rng.uniform(-drift, drift, count),
            rng.uniform(0.6, 1.6, count), rng.uniform(0, 2 * math.pi, count), wander, sharp)


def _ridges(folds, alpha, t):
    """The folds' depth (0 to 1) at angles alpha [.., cols], t [rings, 1] the way down the sleeve (0 to 1)."""
    c, hw, h, drift, freq, phase, wander, sharp = folds
    out = np.zeros(np.broadcast(alpha, t).shape)
    for ci, wi, hi, di, fi, pi_ in zip(c, hw, h, drift, freq, phase):
        d = (alpha - ci - di * t + math.pi) % (2 * math.pi) - math.pi
        live = 1.0 - wander * (0.5 + 0.5 * np.sin(2 * math.pi * fi * t + pi_))
        out = np.maximum(out, hi * live * (0.5 + 0.5 * np.cos(np.clip(d / wi, -1.0, 1.0) * math.pi)) ** sharp)
    return out


def _sleeve(P, T, aw, M, body_bvh, vest_bvh, arm_z, ratios, log):
    """The left sleeve (x > 0): points [rings, COLS, 5] (x, y, z, u, v), and the ring the fan ends at."""
    w = M['w']
    rv = fit.RATIOS
    band = P[(aw > 0.4) & (aw < 0.6)]
    low = band[band[:, 2] <= np.percentile(band[:, 2], 3)]
    y0, y1 = M['scye_y']
    piv = np.array([float(np.abs(low[:, 0]).mean()), 0.5 * (y0 + y1), M['z_pit_arm']])
    x_end = rv['shoulder'] * M['x_sh']
    th_exit = math.atan2(M['z_sh'] - piv[2], x_end - piv[0])
    th_seam = math.atan2(M['z_sh'] - piv[2], x_end - (rv['scoop_front'] + ratios['seam_in']) * w - piv[0])
    z_elbow, z_wrist = arm_z
    z_end = z_wrist + ratios['cuff'] * w
    n_fan = int(math.ceil(th_seam * 1.5 * w / STEP))
    n_str = int(math.ceil((piv[2] - z_end) / STEP))
    K = n_fan + 1 + n_str
    theta = np.concatenate([np.linspace(th_seam, 0.0, n_fan + 1), np.zeros(n_str)])
    drop = np.concatenate([np.zeros(n_fan + 1), (np.arange(n_str) + 1) * (piv[2] - z_end) / n_str])
    alpha = -math.pi + 2 * math.pi * np.arange(COLS) / COLS       # (0 out and up over the arm, -pi/2 the front, +-pi the underarm seam)
    ca, sa = np.cos(alpha), np.sin(alpha)
    D = np.stack([np.cos(theta), np.zeros(K), np.sin(theta)], 1)            # each ring's own "out"
    base_pt = piv[None, :] - np.stack([np.zeros(K), np.zeros(K), drop], 1)  # the pivot, then the point under it
    # each ring's centre: the middle of the body's section in its plane
    arm = P[(aw > 0.5) & (P[:, 0] > 0)]
    cen = np.empty((K, 3))
    for k in range(K):
        if k <= n_fan:
            far = base_pt[k] + 30.0 * D[k]
            hit = body_bvh.ray_cast(Vector(far), Vector(-D[k]))[0]
            rho = 30.0 - (hit - Vector(far)).length if hit is not None else 2.0 * w
            cen[k] = base_pt[k] + 0.5 * rho * D[k]
        else:
            sl = arm[np.abs(arm[:, 2] - base_pt[k, 2]) < 0.3]
            cen[k] = (0.5 * (sl[:, 0].min() + sl[:, 0].max()), 0.5 * (sl[:, 1].min() + sl[:, 1].max()), base_pt[k, 2]) if len(sl) > 8 else cen[k - 1] - (0, 0, STEP)
    for a in (0, 1):
        cen[:, a] = fit._blur(cen[:, a], 4.0, 0, False)
    # the arm as rays see it from the centre, in the ring's plane; in the fan never past the pivot (the underarm)
    R = np.empty((K, COLS))
    yv = np.array([0.0, 1.0, 0.0])
    for k in range(K):
        o = Vector(cen[k])
        r = np.empty(COLS)
        for j in range(COLS):
            d = Vector(ca[j] * D[k] + sa[j] * yv)
            hit = body_bvh.ray_cast(o, d)[0]
            r[j] = (hit - o).length if hit is not None else 1e9
        if k <= n_fan:
            room = float((cen[k] - base_pt[k]) @ D[k]) - ratios['skin'] * w
            lim = np.where(ca < -1e-6, room / np.maximum(-ca, 1e-6), 1e9)
            r = np.minimum(r, lim)
        ok = r < 1e8
        med = float(np.median(r[ok])) if ok.any() else w
        r = np.minimum(r, 1.6 * med)          # (where the arm runs into the body there is no skin between them to find)
        R[k] = fit._taut(r, -ca, -sa)
    R = np.maximum(fit._blur(fit._blur(R, 1.2, 0, False), 2.0, 1, True), R - 0.01 * w)
    lie = R + ratios['sleeve_ease'] * w

    def pts(r):
        return cen[:, None, :] + r[:, :, None] * (ca[None, :, None] * D[:, None, :] + sa[None, :, None] * yv[None, None, :])

    # how far out from under the vest a point is: past the vest's armhole edge, or down the sleeve from where the
    # seam's fan passes the vest's shoulder end, whichever is more
    B0 = pts(lie)
    run = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(B0[:, COLS // 2], axis=0), axis=1))])     # (down the top of the arm)
    k_exit = int(np.argmin(np.abs(theta[:n_fan + 1] - th_exit)))
    vest_arm = fit.trim_lines(M, np.zeros(fit.COLS), fit.scye(P, aw, M, rv)[0], rv)[0][1][1]
    z_b = M['z_pit_arm'] - rv['scye_drop'] * w
    flat = B0.reshape(-1, 3)
    past = np.where(flat[:, 2] > z_b, -vest_arm(flat), -1e3).reshape(K, COLS)
    e = np.maximum(np.maximum(past, (run - run[k_exit])[:, None]), 0.0)
    # the cloth hangs plumb from the widest place above it: down the level rings a point is never nearer the arm's
    # axis than the ring above (not on the underarm side, where the sleeve lies between the arm and the body)
    free = fit._ss(-0.80, -0.20, ca)[None, :]
    hang = lie.copy()
    for k in range(n_fan + 1, K):
        hang[k] = np.maximum(hang[k], hang[k - 1])
    hang = lie + free * (hang - lie)
    rng = np.random.default_rng(3)
    t = (run / run[-1])[:, None]
    gathers = _ridges(_folds(rng, ratios['gathers'], 1.25), alpha[None, :], t)
    creases = _ridges(_folds(rng, ratios['flutes'], 0.42, sharp=1.6, low=0.5, drift=0.35, wander=0.7), alpha[None, :], t)
    rise = fit._ss(0.0, ratios['rise'] * w, e)
    a_g = ratios['gather'] * w * fit._ss(0.0, 0.12 * w, e) * np.exp(-e / (ratios['gather_len'] * w)) * fit._ss(-0.97, -0.6, ca)[None, :]
    a_f = ratios['flute'] * w * rise * (0.30 + 0.70 * fit._ss(0.0, 3.0 * w, e)) * free
    full = (ratios['full'] + ratios['bell'] * t ** 1.5) * w * rise * free
    # (the cloth stands out full, and its spare width is taken up in creases: narrow valleys running down it, each
    # coming and going along its length, never deeper than the arm under it)
    r = np.maximum(hang + full + a_g * gathers - a_f * creases, lie)
    # never through the vest: a point the vest lies between the arm's axis and is drawn in short of it
    if vest_bvh is not None:
        clear, cut = ratios['vest_clear'] * w, 0
        for k in range(K):
            o = Vector(cen[k])
            for j in np.nonzero(e[k] > 0)[0]:
                d = Vector(ca[j] * D[k] + sa[j] * yv)
                hit = vest_bvh.ray_cast(o, d, float(r[k, j]) + clear)[0]
                if hit is not None:
                    lim = max((hit - o).length - clear, float(lie[k, j]))
                    if lim < r[k, j]:
                        r[k, j] = lim
                        cut += 1
        log('chemise: sleeve drawn in clear of the vest at %d points' % cut)
        r = np.maximum(lie, np.minimum(r, fit._blur(fit._blur(r, 1.0, 0, False), 1.0, 1, True) + 0.02 * w))
    S = pts(r)
    # the flat cloth: a rectangle as wide as the hem's own run, the gathers taking it up at the head
    hem = S[-1]
    girth = float(np.linalg.norm(np.roll(hem, -1, 0) - hem, axis=1).sum())
    S = np.concatenate([S, S[:, :1]], 1)         # (the first column again at the end: the underarm seam's two edges)
    uv = np.stack(np.broadcast_arrays((np.arange(COLS + 1) / COLS * girth)[None, :], run[:, None]), 2)
    log('chemise: sleeve %d rings (%d in the fan, the vest left at ring %d), hem girth %.2f against %.2f round the arm at the armpit'
        % (K, n_fan + 1, k_exit, girth, float(np.linalg.norm(np.roll(B0[n_fan], -1, 0) - B0[n_fan], axis=1).sum())))
    return np.concatenate([S, uv], 2), n_fan


def _seam_plane(P, aw, M, ratios):
    """The armhole seam's plane (the first ring of the sleeve's fan): the pivot at the armpit and the plane's normal
    toward the arm, for the left side."""
    w, rv = M['w'], fit.RATIOS
    band = P[(aw > 0.4) & (aw < 0.6)]
    low = band[band[:, 2] <= np.percentile(band[:, 2], 3)]
    piv = np.array([float(np.abs(low[:, 0]).mean()), 0.5 * sum(M['scye_y']), M['z_pit_arm']])
    th = math.atan2(M['z_sh'] - piv[2], rv['shoulder'] * M['x_sh'] - (rv['scoop_front'] + ratios['seam_in']) * w - piv[0])
    return piv, np.array([math.sin(th), 0.0, -math.cos(th)])


def _cloth(name, V, faces, material, body, trees, arm_share, rig, thickness, bones=('Pelvis', 'Chest', 'Neck', 'UpperArm')):
    """A mesh object from points [x, y, z, u, v] and faces, its flat-cloth coordinates as the UV map, handed to the
    body's bones: each point takes the weights of the skin nearest it (arm_share of it from the arm's skin alone)."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    lay = bm.loops.layers.uv.new('UVMap')
    sh = bm.verts.layers.float.new('arm_share')
    vs = [bm.verts.new(p[:3]) for p in V]
    if arm_share is not None:
        for v, a in zip(vs, arm_share):
            v[sh] = float(a)
    for f in faces:
        try:
            face = bm.faces.new([vs[i] for i in f])
        except ValueError:
            continue
        for lp, i in zip(face.loops, f):
            lp[lay].uv = (V[i, 3], V[i, 4])
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)      # (the seam joined: each side keeps its own place on the cloth)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    arm_share = np.empty(len(me.vertices))
    me.attributes['arm_share'].data.foreach_get('value', arm_share)
    for p in me.polygons:
        p.use_smooth = True
    rest = me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
    rest.data.foreach_set('vector', [c for v in me.vertices for c in v.co])
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    if material is not None:
        me.materials.append(material)
    groups = {g.index: o.vertex_groups.new(name=g.name) for g in body.vertex_groups
              if g.name.split('.')[0] in bones}
    bverts = body.data.vertices
    kd_all, kd_arm, arm_ids = trees
    for v in me.vertices:
        share = float(arm_share[v.index])
        acc, tot = {}, 0.0
        for kd, ids, part in ((kd_all, None, 1.0 - share), (kd_arm, arm_ids, share)):
            if part <= 1e-3:
                continue
            got, sub = {}, 0.0
            for _, bi, dist in kd.find_n(v.co, 4):
                k = 1.0 / (dist + 0.05)
                for g in bverts[bi if ids is None else ids[bi]].groups:
                    if g.group in groups:
                        got[g.group] = got.get(g.group, 0.0) + k * g.weight
                        sub += k * g.weight
            for gi, wt in got.items():
                acc[gi] = acc.get(gi, 0.0) + part * wt / max(sub, 1e-9)
                tot += part * wt / max(sub, 1e-9)
        for gi, wt in acc.items():
            if wt / tot > 1e-3:
                groups[gi].add([v.index], wt / tot, 'REPLACE')
    if rig is not None:
        m = o.modifiers.new('rig', 'ARMATURE')
        m.object = rig
        m.use_deform_preserve_volume = True
    if thickness:
        m = o.modifiers.new('thickness', 'SOLIDIFY')
        m.thickness = thickness
        m.offset = 1.0
        m.use_rim = True
    return o


def build_chemise(body, M, rig=None, material=None, vest_objs=(), name='Chemise', ratios=RATIOS, log=print):
    """Draft the chemise on the body (M: fit.measure's measurements, as the vest was drafted from) and hand it to the
    body's bones. vest_objs: the vest's meshes, for the sleeves to keep clear of. Returns (body, sleeves)."""
    P, T, aw = fit.body_arrays(body)
    w = M['w']
    bvh = BVHTree.FromPolygons([Vector(p) for p in P], [tuple(int(i) for i in t) for t in T])
    vest_bvh = None
    if vest_objs:
        pts, polys = [], []
        for o in vest_objs:
            n0 = len(pts)
            pts += [o.matrix_world @ v.co for v in o.data.vertices]
            polys += [tuple(n0 + i for i in p.vertices) for p in o.data.polygons]
        vest_bvh = BVHTree.FromPolygons(pts, polys)
    gi = {g.name: g.index for g in body.vertex_groups}
    fore = [v.index for v in body.data.vertices if any(g.group == gi.get('Forearm.L') and g.weight > 0.5 for g in v.groups)]
    if not fore:
        raise RuntimeError('chemise: the body has no forearm to end the sleeve at')
    arm_z = (float(P[fore, 2].max()), float(P[fore, 2].min()))
    kd_all = KDTree(len(P))
    for i, p in enumerate(P):
        kd_all.insert(Vector(p), i)
    kd_all.balance()
    arm_ids = [int(i) for i in np.nonzero(aw > 0.5)[0]]
    kd_arm = KDTree(len(arm_ids))
    for k, i in enumerate(arm_ids):
        kd_arm.insert(Vector(P[i]), k)
    kd_arm.balance()
    trees = (kd_all, kd_arm, arm_ids)
    gap = ratios['skin'] * w
    # the body
    seam = _seam_plane(P, aw, M, ratios)
    V, faces = _body_cloth(P, T, aw, M, seam, ratios, log)
    # (off the torso's own skin only: at rest the forearm lies against the hip, and is no reason to move the cloth there)
    torso = BVHTree.FromPolygons([Vector(p) for p in P], [tuple(int(i) for i in t) for t in T[(aw[T] < 0.5).all(1)]])
    moved = _off_skin(V, torso, gap)
    log('chemise: body %d points, %d eased off the skin' % (len(V), moved))
    # (the body's cloth follows the trunk and the upper arm, as the vest does: never the forearm lying against the hip)
    o_body = _cloth(name, V, faces, material, body, trees, None, rig, 0.0)
    # the sleeves: the left one drafted, the right its mirror
    S, n_fan = _sleeve(P, T, aw, M, bvh, vest_bvh, arm_z, ratios, log)
    K = S.shape[0]
    L = S.reshape(-1, 5).copy()
    # (off the arm's and the shoulder's skin: where the sleeve hangs against the hip it is the cloth that gives, not the hip)
    arm_bvh = BVHTree.FromPolygons([Vector(p) for p in P], [tuple(int(i) for i in t) for t in T[(aw[T] >= 0.5).any(1)]])
    moved = _off_skin(L, arm_bvh, gap)
    log('chemise: sleeve %d points, %d eased off the skin' % (len(L), moved))
    Rt = L.copy()
    Rt[:, 0] *= -1
    C = S.shape[1]
    idx = np.arange(K * C).reshape(K, C)
    quads = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], 2).reshape(-1, 4)
    faces = [list(q) for q in quads] + [list(q[::-1] + K * C) for q in quads]
    share = np.repeat(fit._ss(n_fan - 10, n_fan + 2, np.arange(K)), C)
    o_sl = _cloth(name + ' sleeves', np.concatenate([L, Rt]), faces, material, body, trees,
                  np.concatenate([share, share]), rig, ratios['thick'] * w,
                  bones=('Chest', 'Neck', 'UpperArm', 'Forearm', 'Hand'))
    return o_body, o_sl


def under_vest(chemise, vest, log=print):
    """How the chemise's body lies under the vest's leather: the points of it that are outside the leather's inner
    face where the leather covers them."""
    pts = [vest.matrix_world @ v.co for v in vest.data.vertices]
    bvh = BVHTree.FromPolygons(pts, [tuple(p.vertices) for p in vest.data.polygons])
    out, near = 0, 0
    for v in chemise.data.vertices:
        loc, nrm, _, dist = bvh.find_nearest(chemise.matrix_world @ v.co, 0.6)
        if loc is None:
            continue
        near += 1
        d = (chemise.matrix_world @ v.co - loc)
        if d.length > 1e-6 and abs(d.normalized().dot(nrm)) > 0.9 and d.dot(nrm) > 0:
            out += 1
    log('chemise under the vest: %d of %d covered points lie outside the leather' % (out, near))
    return out


# The linen's look. From the research (see linen_texture.py): a fine plain weave, faintly streaked both ways by the
# flax yarn's slubs; bleached, so a white just off toward cream; a dry surface with a low lustre and almost no fuzz
# (flax is a long smooth fibre: far less sheen-fuzz than wool, a little more gloss than cotton); thin enough that a
# single layer passes some light.
LINEN = dict(colour=(0.800, 0.775, 0.700), streak=0.16, rough=0.78, specular=0.32, sheen=0.12, through=0.14,
             relief=0.55)


def linen_closeup(name, P=LINEN):
    """The 3D look of the linen on a material from female.mat (its flat sprite look kept beside it)."""
    m = bpy.data.materials.get(name)
    if m is None:
        return
    nt = m.node_tree
    N = nt.nodes.new
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
    flat = next((n for n in nt.nodes if n.type == 'BSDF_DIFFUSE'), None)
    if flat is not None:
        flat.name = 'sprite bsdf'
    uv = N('ShaderNodeUVMap')
    mp = N('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = (1.0 / (linen_texture.THREADS * PITCH),) * 3
    nt.links.new(uv.outputs['UV'], mp.inputs['Vector'])
    tex = N('ShaderNodeTexImage')
    tex.image = bpy.data.images.get('vest linen') or linen_texture.make_image('vest linen')
    tex.interpolation = 'Cubic'
    nt.links.new(mp.outputs['Vector'], tex.inputs['Vector'])
    sep = N('ShaderNodeSeparateColor')
    nt.links.new(tex.outputs['Color'], sep.inputs['Color'])
    height, tone, open_ = sep.outputs['Red'], sep.outputs['Green'], sep.outputs['Blue']

    def rng(sock, a, b, lo, hi):
        r = N('ShaderNodeMapRange')
        r.inputs['From Min'].default_value, r.inputs['From Max'].default_value = a, b
        r.inputs['To Min'].default_value, r.inputs['To Max'].default_value = lo, hi
        nt.links.new(sock, r.inputs['Value'])
        return r.outputs['Result']

    def scaled(colour, fac):
        k = N('ShaderNodeVectorMath')
        k.operation = 'SCALE'
        if isinstance(colour, tuple):
            k.inputs[0].default_value = colour
        else:
            nt.links.new(colour, k.inputs[0])
        nt.links.new(fac, k.inputs['Scale'])
        return k.outputs['Vector']
    # the yarn's own shade, streak by streak, and the folds' hollows a little dimmer (cloth facing cloth)
    col = scaled(P['colour'], rng(tone, 0.15, 0.85, 1.0 - P['streak'], 1.0 + 0.5 * P['streak']))
    ao = N('ShaderNodeAmbientOcclusion')
    ao.inputs['Distance'].default_value = 1.2
    ao.samples = 8
    col = scaled(col, rng(ao.outputs['AO'], 0.0, 1.0, 0.62, 1.0))
    bump = N('ShaderNodeBump')
    bump.inputs['Strength'].default_value, bump.inputs['Distance'].default_value = P['relief'], 0.6 * PITCH
    nt.links.new(height, bump.inputs['Height'])
    pb = N('ShaderNodeBsdfPrincipled')
    nt.links.new(col, pb.inputs['Base Color'])
    nt.links.new(bump.outputs['Normal'], pb.inputs['Normal'])
    pb.inputs['Roughness'].default_value = P['rough']
    pb.inputs['Specular IOR Level'].default_value = P['specular']
    pb.inputs['Sheen Weight'].default_value = P['sheen']
    pb.inputs['Sheen Roughness'].default_value = 0.35
    # light through the single layer, more between the threads; the two looks mixed inside a group, so the material
    # still has one node named 'close bsdf' with a 'BSDF' output for the views to switch to
    g = bpy.data.node_groups.new('linen layer', 'ShaderNodeTree')
    g.interface.new_socket('Lit', in_out='INPUT', socket_type='NodeSocketShader')
    g.interface.new_socket('Colour', in_out='INPUT', socket_type='NodeSocketColor')
    g.interface.new_socket('Through', in_out='INPUT', socket_type='NodeSocketFloat')
    g.interface.new_socket('BSDF', in_out='OUTPUT', socket_type='NodeSocketShader')
    gi, go = g.nodes.new('NodeGroupInput'), g.nodes.new('NodeGroupOutput')
    tr = g.nodes.new('ShaderNodeBsdfTranslucent')
    mx = g.nodes.new('ShaderNodeMixShader')
    g.links.new(gi.outputs['Colour'], tr.inputs['Color'])
    g.links.new(gi.outputs['Through'], mx.inputs['Fac'])
    g.links.new(gi.outputs['Lit'], mx.inputs[1])
    g.links.new(tr.outputs['BSDF'], mx.inputs[2])
    g.links.new(mx.outputs['Shader'], go.inputs['BSDF'])
    close = N('ShaderNodeGroup')
    close.node_tree = g
    close.name = 'close bsdf'
    nt.links.new(pb.outputs['BSDF'], close.inputs['Lit'])
    nt.links.new(col, close.inputs['Colour'])
    nt.links.new(rng(open_, 0.0, 1.0, P['through'], min(1.0, 3.0 * P['through'])), close.inputs['Through'])
    nt.links.new(close.outputs['BSDF'], out.inputs['Surface'])
