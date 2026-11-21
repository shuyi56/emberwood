"""The cloak's pieces cut as cloth and sewn: every pattern piece (pattern.py) becomes a sheet of cloth whose threads
are the piece's own grid, its flat layout kept as the 'pattern' UVs (model units, as cut) and its place on the
figure given by the surface it was drafted on. The skirts carry their fullness as hanging folds (the spread cloth
put back into the width it was spread from). The pieces are then sewn: the two edges of every seam are the same
line of points on the figure, so sewing joins them point for point into one garment, and the seam is left as a
pressed seam (a shallow groove along it). The centre-front edges stay open; the backs are one piece across the
fold. Topstitching (on the front edges, the yoke seam and the hem) is built as real stitches on the cloth's face,
and the collar is closed at the throat with a bronze clasp.

    o = sew.build(M, panels, seams, rig, mats)     # the sewn cloak, skinned to the rig, as fitted (no solver)
    sew.settle(o, rig, colliders, pinned)          # the fabric solver on the cloak alone (remembered by layers.py)
    sew.stitches(o, M, mats)                       # the stitches, laid on the cloak where it ended up"""
import math
import os
import time

import bpy
import numpy as np

import measure as MS
import pattern as PT

# The cloak's cloth for the fabric solver (src/world/fabric-core.js, the game's cape solver, run to rest): a heavy
# wool melton bagged with a lining. Compliances are XPBD's (0 is rigid): the threads barely give, the bias a little,
# and the bend ties are firm, so the fullness falls in a few deep folds and never crumples.
FABRIC = dict(
    # (a hang from the sewn shape to rest, not a drop: a third of the pull and many walks over the ties, so the
    # sheet, which has no collision with itself, eases into its folds instead of crumpling through itself)
    settings=dict(gravity=[0.0, 0.0, -120.0], damping=0.02, iterations=20, friction=0.6),
    compliance=[0.0, 1e-6, 2e-6],          # thread, bias, bend two along
    gap=0.14,                              # kept off the skin (more than the lining's depth, 0.09, under it)
    settle=dict(h=1 / 120, max=400, tol=2e-4, damping=0.3),
    grow={'Hand': 0.25, 'Fist': 0.25},    # the hands made fatter by this (thin: a fold could pass a finger)
    grid=0.12,                             # the body's distance grid, in units
)
BEND_ALONG = (2,)            # the bend ties: two threads on (five on fights the two and throws the sheet)
TIES = {}                    # each sewn cloak's weave (point, point, 0 thread / 1 bias / 2 bend), by its name
THICK = 0.09                 # wool and lining, in units (about 2.5 mm)
PRESS = 0.035                # how deep a pressed seam's groove is
STITCH = dict(length=0.13, gap=0.13, width=0.022, inset=0.6 / MS.CM, lift=0.012)


# ---------------------------------------------------------------- the folds the fullness falls into
def _normals(S):
    du = np.gradient(S, axis=1)
    ds = np.gradient(S, axis=0)
    N = np.cross(du, ds)
    N /= np.linalg.norm(N, axis=2, keepdims=True) + 1e-12
    out = S.copy()
    out[..., 2] = 0
    rad = out / (np.linalg.norm(out, axis=2, keepdims=True) + 1e-12)
    sgn = np.sign((N * rad).sum(2, keepdims=True))
    sgn[sgn == 0] = 1
    return N * sgn


def folded(p):
    """The skirt panel p's points with its fullness put back as hanging folds: across each row the cloth is
    waved out from the surface it was drafted on (never in, so the valleys rest where the plain cloth lay) until
    the row is as long as it was cut; the folds are p.folds waves across the panel, their valleys at the panel's
    seams (and at the fold of the back when the count is whole)."""
    S = p.S.copy()
    nr, nc = S.shape[:2]
    N = _normals(S)
    if p.name.startswith('back'):
        N[:, -1, 0] = 0.0            # (on the fold the cloth moves straight back: the two halves stay one)
    u = np.linspace(0.0, 1.0, nc)
    # (a little off a pure wave, so the folds are not all alike: each shifted and some wider than others)
    ph = 2 * math.pi * p.folds * u + 0.35 * np.sin(2 * math.pi * u * 1.3 + 0.7) * np.sin(math.pi * u)
    w = 0.5 * (1.0 - np.cos(ph))
    w *= (1.0 - 0.25 * np.sin(math.pi * u * 2.0 * p.folds + 1.1) ** 2)
    for i in range(1, nr):
        target = PT._row_len(p.Fs[i])
        base = S[i]
        lo, hi = 0.0, 6.0
        for _ in range(40):
            a = 0.5 * (lo + hi)
            if PT._row_len(base + a * w[:, None] * N[i]) < target:
                lo = a
            else:
                hi = a
        S[i] = base + 0.5 * (lo + hi) * w[:, None] * N[i]
    return S


# ---------------------------------------------------------------- cutting and sewing
def _mirror(P):
    Q = P.copy()
    Q[..., 0] = -Q[..., 0]
    return Q


def build(M, panels, seams, rig, mats, name='Cloak', log=print):
    """The cloak sewn from its pieces, as fitted on the figure (no solver): one mesh, skinned to the rig's Chest.
    mats: (wool, lining) material ids (female.mat_def). Returns the object; its point attributes say which piece
    each point was cut from ('piece'), where the seams run ('seam'), the pin weights for the solver ('pin'), and
    the 'pattern' UVs are each piece as cut (laid side by side, in model units)."""
    import female
    pieces = []          # (name, side, points [nr, nc, 3], flat [nr, nc, 2])
    order = ['front yoke', 'back yoke', 'front skirt', 'back skirt', 'collar']
    x_off = 0.0
    for nm in order:
        p = panels[nm]
        S = folded(p) if nm.endswith('skirt') else p.S
        F = p.Fs - p.Fs.reshape(-1, 2).min(0)
        for side in ('R', 'L'):
            P3 = S if side == 'R' else _mirror(S)
            # (her left side's pieces are cut as the mirror: their columns run the other way round)
            P3 = P3 if side == 'R' else P3[:, ::-1]
            F2 = F.copy() if side == 'R' else np.stack([F[..., 0].max() - F[..., 0], F[..., 1]], -1)[:, ::-1]
            F2 = F2 + [x_off, 0.0]
            x_off = F2[..., 0].max() + 1.0
            pieces.append((nm, side, P3, F2))
    # every point, then the sewing: points of different pieces on the same spot are one point (the seams); the
    # front edges are apart by the gap left between them, and stay open
    pts, uvs, faces, piece_of, line, ties, grids = [], [], [], [], [], [], []
    for k, (nm, side, P3, F2) in enumerate(pieces):
        nr, nc = P3.shape[:2]
        base = len(pts)
        # (the lines the topstitching follows, by the piece's own threads: 1 the front edge, 2 the top of the turned
        # hem, 3 a thread below the yoke seam on the skirt)
        L = np.zeros((nr, nc))
        cfc = 0 if side == 'R' else nc - 1
        if nm.startswith('front'):
            L[:, cfc] = 1
        if nm.endswith('skirt'):
            down = np.linalg.norm(np.diff(P3[:, nc // 2], axis=0), axis=1)
            up = np.cumsum(down[::-1])
            L[nr - 1 - (1 + int(np.searchsorted(up, PT.ALLOW['hem'] * M.H))), :] = 2
            L[1, :] = 3
        line.extend(L.ravel())
        pts.extend(P3.reshape(-1, 3))
        uvs.extend(F2.reshape(-1, 2))
        piece_of.extend([k] * (nr * nc))
        idx = base + np.arange(nr * nc).reshape(nr, nc)
        # (the piece's weave, for the fabric solver: its threads down and across, the bias both ways, and the bend
        # ties two threads along)
        for a, b, kind in ((idx[:, :-1], idx[:, 1:], 0), (idx[:-1], idx[1:], 0),
                           (idx[:-1, :-1], idx[1:, 1:], 1), (idx[:-1, 1:], idx[1:, :-1], 1)):
            ties.append(np.stack([a.ravel(), b.ravel(), np.full(a.size, kind)], 1))
        grids.append((nm, side, idx))
        q = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
        # (faces wound so the normals face out: the right side's grid runs round toward her right)
        faces.extend(q.tolist() if side == 'L' else q[:, ::-1].tolist())
    pts = np.array(pts)
    key = np.round(pts / 1e-4).astype(np.int64)
    uniq, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    first = np.full(len(uniq), -1)
    for i in range(len(pts) - 1, -1, -1):
        first[inv[i]] = i
    V = pts[first]
    sewn = counts[inv] > 1                   # (a point on a seam: two or more pieces met there)
    log('cloak: %d pieces cut, %d points, sewn into %d (%d seam points)' % (
        len(pieces), len(pts), len(V), int((counts > 1).sum())))
    F_ = [[int(inv[i]) for i in f] for f in faces]
    T = np.concatenate(ties)
    T[:, 0], T[:, 1] = inv[T[:, 0]], inv[T[:, 1]]
    T = T[T[:, 0] != T[:, 1]]
    T[:, :2] = np.sort(T[:, :2], axis=1)
    T = np.unique(T, axis=0)
    # (and its warp: each thread down the cloak from the neck to the hem, through the yoke and on down the skirt,
    # kept at its length exactly by the solver's held chains, as a woven warp does not stretch under its weight)
    chains = []
    by = {(nm, side): idx for nm, side, idx in grids}
    for side in ('R', 'L'):
        for part in ('front', 'back'):
            y, k = by[(part + ' yoke', side)], by[(part + ' skirt', side)]
            for j in range(y.shape[1]):
                ch = [int(inv[i]) for i in y[:, j]] + [int(inv[i]) for i in k[1:, j]]
                chains.append(ch)
    # the bend ties: along every thread of the whole cloak, across its seams (round from one front edge to the
    # other, the back's two halves one across the fold; and down from the neck to the hem), each point tied to the
    # points BEND_ALONG threads on, so the cloth folds in a few wide folds and turns smoothly over a seam
    lines = [np.array(c) for c in chains]
    for part in ('yoke', 'skirt'):
        rf, rb = by[('front ' + part, 'R')], by[('back ' + part, 'R')]
        lf, lb = by[('front ' + part, 'L')], by[('back ' + part, 'L')]
        for i in range(rf.shape[0]):
            ring = np.concatenate([rf[i], rb[i, 1:], lb[i, 1:], lf[i, 1:]])
            lines.append(inv[ring])
    bend = []
    for L in lines:
        for k, kind in zip(BEND_ALONG, range(2, 2 + len(BEND_ALONG))):
            if len(L) > k:
                bend.append(np.stack([L[:-k], L[k:], np.full(len(L) - k, kind)], 1))
    B = np.concatenate(bend)
    B = B[B[:, 0] != B[:, 1]]
    B[:, :2] = np.sort(B[:, :2], axis=1)
    T = np.unique(np.concatenate([T, B]), axis=0)
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], F_)
    me.update()
    uv = me.uv_layers.new(name='pattern')
    for poly, f in zip(me.polygons, faces):
        for li, vi in zip(poly.loop_indices, f):
            uv.data[li].uv = uvs[vi]
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    me.materials.append(female.mat(mats[0]))
    me.materials.append(female.mat(mats[1]))
    for poly in me.polygons:
        poly.use_smooth = True
    _face_out(me)
    # which piece, the seams, the pins
    pc = np.zeros(len(V))
    pc[inv] = np.array(piece_of)
    ln = np.zeros(len(V))
    ln[inv] = np.array(line)
    me.attributes.new('line', 'FLOAT', 'POINT').data.foreach_set('value', ln)
    a = me.attributes.new('piece', 'FLOAT', 'POINT')
    a.data.foreach_set('value', pc)
    seam = np.zeros(len(V))
    seam[inv[sewn]] = 1.0
    # (the fold of the back and the collar's fold are not seams: one piece across the centre back)
    seam[np.abs(V[:, 0]) < 1e-3] = 0.0
    me.attributes.new('seam', 'FLOAT', 'POINT').data.foreach_set('value', seam)
    # (the collar and the yokes are held as fitted: interfaced and lined, they lie on the neck and the shoulders as
    # cut, and the skirts hang from the yoke seam. Left to the solver, the yoke sank onto the shoulder in ripples and
    # the shoulder seam's lining showed through)
    pin = np.zeros(len(V))
    held = np.zeros(len(V), bool)
    for k, (nm, side, P3, F2) in enumerate(pieces):
        if nm == 'collar' or nm.endswith('yoke'):
            held[inv[np.array(piece_of) == k]] = True
        if nm == 'collar':
            collar = inv[np.array(piece_of) == k]
    pin[held] = 1.0
    me.attributes.new('pin', 'FLOAT', 'POINT').data.foreach_set('value', pin)
    # the pressed seams: a shallow groove along each, the cloth either side turned down into it a little
    press(o, seam, collar)
    # the rest shape (for the pattern-space looks) and the rig: the cloak rides the chest
    co = np.empty(len(V) * 3)
    me.vertices.foreach_get('co', co)
    me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT').data.foreach_set('vector', co)
    g = o.vertex_groups.new(name='Chest')
    g.add(list(range(len(V))), 1.0, 'REPLACE')
    m = o.modifiers.new('rig', 'ARMATURE')
    m.object = rig
    sol = o.modifiers.new('thickness', 'SOLIDIFY')
    sol.thickness = THICK
    sol.offset = -1.0
    sol.material_offset = 1          # (the inner face is the lining)
    sol.material_offset_rim = 0
    sol.use_even_offset = False
    sol.use_quality_normals = True
    o['cloak_pieces'] = len(pieces)
    TIES[o.name] = (T, chains)
    return o


def _face_out(me):
    """Every face turned to face away from the figure's axis (the wool outside, the lining in)."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    flip = []
    for f in bm.faces:
        c = f.calc_center_median()
        if f.normal.x * c.x + f.normal.y * c.y < 0:
            flip.append(f)
    bmesh.ops.reverse_faces(bm, faces=flip)
    bm.to_mesh(me)
    bm.free()
    me.update()


def press(o, seam, keep=()):
    """Each sewn seam as a pressed seam: the points on it let in by PRESS (the two pieces turn into the seam's
    line), and their neighbours by a share of it. keep: points left as they are (the collar: its fold is no seam)."""
    me = o.data
    n = len(me.vertices)
    co = np.empty(n * 3)
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    me.update()
    nor = np.empty(n * 3)
    me.vertices.foreach_get('normal', nor)
    nor = nor.reshape(-1, 3)
    e = np.empty(len(me.edges) * 2, dtype=np.int64)
    me.edges.foreach_get('vertices', e)
    e = e.reshape(-1, 2)
    near = np.zeros(n)
    np.maximum.at(near, e[:, 0], seam[e[:, 1]])
    np.maximum.at(near, e[:, 1], seam[e[:, 0]])
    depth = PRESS * np.maximum(seam, 0.35 * near * (1 - seam))
    depth[keep] = 0.0
    co -= nor * depth[:, None]
    me.vertices.foreach_set('co', co.ravel())
    me.update()


# ---------------------------------------------------------------- the solver
def _body_grid(objs, lo, hi, step, grow=None):
    """The signed distance to the colliders' surface (posed, as they stand) on a grid over lo..hi: negative inside.
    Points over a unit from the surface are given a unit (the cloth never gets that far in). grow: {name prefix:
    units} makes thin parts fatter (the hands, so a fold dragged over them cannot pass a finger in one step)."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    grow = grow or {}
    dg = bpy.context.evaluated_depsgraph_get()
    groups = {}
    for ob in objs:
        g = next((v for k, v in grow.items() if ob.name.startswith(k)), 0.0)
        verts, polys = groups.setdefault(g, ([], []))
        e = ob.evaluated_get(dg)
        me = e.to_mesh()
        n0 = len(verts)
        mw = ob.matrix_world
        verts += [mw @ v.co for v in me.vertices]
        polys += [[n0 + i for i in p.vertices] for p in me.polygons]
        e.to_mesh_clear()
    xs, ys, zs = (np.arange(lo[k], hi[k] + step, step) for k in range(3))
    D = np.ones((len(zs), len(ys), len(xs)), dtype=np.float32)
    for g, (verts, polys) in groups.items():
        fn = BVHTree.FromPolygons(verts, polys).find_nearest
        for k, z in enumerate(zs):
            row = D[k]
            for j, y in enumerate(ys):
                for i, x in enumerate(xs):
                    p = Vector((x, y, z))
                    hit = fn(p, 1.0)
                    if hit[0] is not None:
                        d = (hit[3] if (p - hit[0]).dot(hit[1]) >= 0 else -hit[3]) - g
                        if d < row[j, i]:
                            row[j, i] = d
    return D, (float(xs[0]), float(ys[0]), float(zs[0])), (len(xs), len(ys), len(zs))


def settle(o, rig, colliders, pinned=None, P=FABRIC, log=print):
    """Hang the sewn cloak to rest on the posed figure with the project's fabric solver (fabric_settle.mjs runs
    src/world/fabric-core.js's Fabric.settle in node): its own weight against its weave, the collar held where it
    was fitted, the body (and anything worn under it) a fixed surface it lies on. Only the cloak is solved. Keeps
    what it settles to as its shape. pinned: a weight for each point, above 0 held (default: the 'pin' attribute)."""
    import json
    import shutil
    import subprocess
    import tempfile
    me = o.data
    n = len(me.vertices)
    start = np.empty(n * 3)
    me.vertices.foreach_get('co', start)
    if pinned is None:
        pinned = np.empty(n)
        me.attributes['pin'].data.foreach_get('value', pinned)
    T, chains = TIES.get(o.name, (None, None))
    node = shutil.which('node')
    if T is None or node is None:
        log('cloak: fabric solver not run (%s): left as sewn' % ('no node' if node is None else 'no weave'))
        return start.reshape(-1, 3)
    P3 = start.reshape(-1, 3)
    lo = P3.min(0) - 1.0
    hi = P3.max(0) + 1.0
    t0 = time.time()
    D, origin, dims = _body_grid(colliders, lo, hi, P['grid'], P.get('grow'))
    t1 = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        start.astype(np.float64).tofile(os.path.join(tmp, 'pos.bin'))
        T.astype(np.int32).tofile(os.path.join(tmp, 'ties.bin'))
        D.tofile(os.path.join(tmp, 'sdf.bin'))
        cfg = dict(pos='pos.bin', ties='ties.bin', chains=chains, pins=[int(i) for i in np.nonzero(pinned > 0)[0]],
                   settings=P['settings'], compliance=P['compliance'], gap=P['gap'], settle=P['settle'],
                   grid=dict(file='sdf.bin', origin=origin, step=P['grid'], dims=dims))
        with open(os.path.join(tmp, 'in.json'), 'w') as fh:
            json.dump(cfg, fh)
        r = subprocess.run([node, os.path.join(here, 'fabric_settle.mjs'), os.path.join(tmp, 'in.json'),
                            os.path.join(tmp, 'out.bin')], capture_output=True, text=True)
        if r.returncode != 0:
            log('cloak: fabric solver failed, left as sewn:\n' + r.stderr[-2000:])
            return P3
        log(r.stdout.strip())
        out = np.fromfile(os.path.join(tmp, 'out.bin'), dtype=np.float64).reshape(-1, 3)
    e = np.empty(len(me.edges) * 2, dtype=np.int64)
    me.edges.foreach_get('vertices', e)
    e = e.reshape(-1, 2)
    l0 = np.linalg.norm(P3[e[:, 0]] - P3[e[:, 1]], axis=1)
    l1 = np.linalg.norm(out[e[:, 0]] - out[e[:, 1]], axis=1)
    mv = np.linalg.norm(out - P3, axis=1)
    log('cloak: hung by the fabric solver (body grid %.0f s, solve %.0f s): moved %.2f on average, %.2f at most; '
        'threads %.1f%% longer on average' % (t1 - t0, time.time() - t1, mv.mean(), mv.max(),
                                             100 * float((l1 / l0).mean() - 1)))
    # (the cloak rides the chest, whose pose is the rest: the settled points are its rest shape as they are)
    me.vertices.foreach_set('co', out.ravel())
    if 'rest' in me.attributes:
        me.attributes['rest'].data.foreach_set('vector', out.ravel())
    me.update()
    return out


# ---------------------------------------------------------------- stitches and the clasp
def _surface_lines(o):
    """The lines the topstitching runs along, as point indices of the cloak in order: each front edge, the hem
    and the yoke seam (from the 'piece' attribute and the grid's boundary)."""
    me = o.data
    n = len(me.vertices)
    co = np.empty(n * 3)
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    pc = np.empty(n)
    me.attributes['piece'].data.foreach_get('value', pc)
    seam = np.empty(n)
    me.attributes['seam'].data.foreach_get('value', seam)
    # boundary edges
    from collections import Counter
    cnt = Counter()
    for p in me.polygons:
        vs = p.vertices
        for a, b in zip(vs, list(vs[1:]) + [vs[0]]):
            cnt[(min(a, b), max(a, b))] += 1
    bnd = [k for k, v in cnt.items() if v == 1]
    return co, pc, seam, bnd


def stitches(o, M, mat, log=print):
    """Topstitching as stitches: short raised dashes of thread on the cloak's face, STITCH['inset'] in from the
    front edges, along the yoke seam on the skirt side, and at the top of the turned hem."""
    import bmesh
    me = o.data
    dg = bpy.context.evaluated_depsgraph_get()
    co, pc, seam, bnd = _surface_lines(o)
    n = len(co)
    me.update()
    nor = np.empty(n * 3)
    me.vertices.foreach_get('normal', nor)
    nor = nor.reshape(-1, 3)
    ln = np.empty(n)
    me.attributes['line'].data.foreach_get('value', ln)
    lines = []
    ang_of = lambda I: np.arctan2(-co[I, 0], -(co[I, 1] - M.axis[1]))  # noqa: E731
    for sgn in (1, -1):
        L = np.nonzero((ln == 1) & (np.sign(co[:, 0]) == sgn))[0]
        L = L[np.argsort(-co[L, 2])]
        if len(L) > 3:
            # (in from the edge, toward her side, on the cloth)
            lines.append(co[L] + np.array([sgn * STITCH['inset'], 0, 0]))
    for code in (2, 3):
        L = np.nonzero(ln == code)[0]
        a = ang_of(L)
        # (round from one front edge to the other: the angle measured from the centre front, cut at the opening)
        L = L[np.argsort(a)]
        Q = co[L]
        # split where the line jumps (across the front opening)
        gaps = np.nonzero(np.linalg.norm(np.diff(Q, axis=0), axis=1) > 1.5)[0]
        for part in np.split(np.arange(len(L)), gaps + 1):
            if len(part) > 3:
                lines.append(Q[part])
    # each line's stitches: dashes along it, pressed onto the cloak's face (the nearest point of the cloak)
    from mathutils.bvhtree import BVHTree
    ev = o.evaluated_get(dg)
    tree = BVHTree.FromObject(ev, dg)
    bm = bmesh.new()
    count = 0
    for Q in lines:
        seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        t = 0.0
        while t + STITCH['length'] < s[-1]:
            a = np.array([np.interp(t, s, Q[:, k]) for k in range(3)])
            b = np.array([np.interp(t + STITCH['length'], s, Q[:, k]) for k in range(3)])
            ends = []
            for p_ in (a, b):
                hit = tree.find_nearest(p_)
                if hit[0] is None:
                    break
                ends.append((np.array(hit[0]), np.array(hit[1])))
            if len(ends) == 2:
                (pa, na), (pb, nb) = ends
                nn = (na + nb) / (np.linalg.norm(na + nb) + 1e-9)
                d = pb - pa
                side = np.cross(nn, d)
                side /= np.linalg.norm(side) + 1e-9
                w = STITCH['width']
                lift = STITCH['lift']
                vs = [bm.verts.new(tuple(p + nn * (lift + h) + side * w * sx))
                      for p in (pa, pb) for (sx, h) in ((-1, 0), (1, 0), (1, w), (-1, w))]
                for f in ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)):
                    bm.faces.new([vs[i] for i in f])
                count += 1
            t += STITCH['length'] + STITCH['gap']
    me2 = bpy.data.meshes.new('Cloak stitches')
    bm.to_mesh(me2)
    bm.free()
    import female
    st = bpy.data.objects.new('Cloak stitches', me2)
    bpy.context.scene.collection.objects.link(st)
    me2.materials.append(female.mat(mat))
    g = st.vertex_groups.new(name='Chest')
    g.add(list(range(len(me2.vertices))), 1.0, 'REPLACE')
    md = st.modifiers.new('rig', 'ARMATURE')
    md.object = o.modifiers['rig'].object
    log('cloak: %d stitches on %d topstitched lines' % (count, len(lines)))
    return st


def clasp(o, M, mat, log=print):
    """The clasp at the throat: a bronze boss on each front at the foot of the collar, and a short chain of oval
    links between them."""
    import bmesh
    import female
    H = M.H
    n0 = M.neck(MS.PT_CF if hasattr(MS, 'PT_CF') else PT.CF_GAP)
    z = M.fnp[2] + 0.35 * PT.COLLAR_H * H
    r = M.radius(0.0, z) + (MS.NECK_EASE + 0.01) * H + THICK
    y = M.axis[1] - r
    bm = bmesh.new()
    boss_r = 0.07 * H
    for sx in (1, -1):
        x = sx * (0.55 * boss_r + 0.25)
        m = bmesh.ops.create_cone(bm, cap_ends=True, segments=20, radius1=boss_r, radius2=0.7 * boss_r,
                                  depth=0.18)['verts']
        bmesh.ops.rotate(bm, verts=m, cent=(0, 0, 0), matrix=_rot_x(math.pi / 2))
        bmesh.ops.translate(bm, verts=m, vec=(x + sx * 0.3, y - 0.09, z))
    # the chain: links between the bosses
    k = 4
    for i in range(k):
        x = -0.6 + 1.2 * (i + 0.5) / k
        tor = bmesh.ops.create_circle(bm, cap_ends=False, segments=10, radius=0.11)['verts']
        bmesh.ops.scale(bm, verts=tor, vec=(1.6, 1.0, 1.0))
        bmesh.ops.rotate(bm, verts=tor, cent=(0, 0, 0), matrix=_rot_x(math.pi / 2 if i % 2 else 0.0))
        bmesh.ops.translate(bm, verts=tor, vec=(x, y - 0.2, z - 0.25 - 0.05 * math.sin(math.pi * (i + 0.5) / k)))
    me = bpy.data.meshes.new('Cloak clasp')
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new('Cloak clasp', me)
    bpy.context.scene.collection.objects.link(ob)
    me.materials.append(female.mat(mat))
    for p in me.polygons:
        p.use_smooth = True
    # (the chain's links as wire: a thin skin round each)
    sk = ob.modifiers.new('wire', 'WIREFRAME')
    sk.thickness = 0.04
    sk.use_replace = False
    g = ob.vertex_groups.new(name='Chest')
    g.add(list(range(len(me.vertices))), 1.0, 'REPLACE')
    md = ob.modifiers.new('rig', 'ARMATURE')
    md.object = o.modifiers['rig'].object
    return ob


def _rot_x(a):
    from mathutils import Matrix
    return Matrix.Rotation(a, 3, 'X')
