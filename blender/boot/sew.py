"""The boot sewn: its pattern pieces (last.PIECES) joined at their seams by the cloth solver, and its stitches.

    sew()       The pieces are put on where they were cut on the last, over the hose. The foot's pieces (the toe
                cap, the vamp, the quarters) are lasted: wetted, stretched over the last and left to dry on it, so
                the shape they hold to is the last's. The shaft is cut flat and rolled round the leg (its flat
                pattern is within a few percent of its lengths on the last, so it is drafted at those). Every lapped
                seam is sewn along both its edges: the top piece's edge to the piece beneath it, the under piece's
                edge to the top piece (SEAMS), each pair held at the distance it was cut at. The upper's lower edge is sewn to the sole and the collar is held up
                by the calf, so both are held where they were cut (HOLD). Then the fabric solver, given leather's
                numbers (LEATHER), draws the seams closed and keeps the leather off the hose. The pieces are sewn as
                one sheet, then each is set back over the one beneath by its layer (its 'lift', a thickness or more).

    stitches()  A row of waxed-thread stitches along every lapped edge, STITCH['in'] in from it, and round the welt:
                each stitch a short rounded thread lying a little sunk into the leather's face, on the same bones.

The pieces are sewn by the game's fabric solver (blender/shared/fabric.py), on the leg at rest where they were cut,
the seams as ties of no length between the paired points."""
import os
import sys

import bpy
import numpy as np
from mathutils.kdtree import KDTree

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('shared',):
    sys.path.insert(0, os.path.join(HERE, '..', d))
import fabric  # noqa: E402
import pattern  # noqa: E402

# Leather for the fabric solver (blender/shared/fabric.py, the game's): it hardly stretches, shears little and bends
# hard, so it lies where it is sewn and sags only a little between its held edges. Units are the figure's.
LEATHER = dict(
    stretch=0.0,
    shear=1e-7,
    bend=4e-6,                # (a stiff hide, not a board: a little softer than 2e-6, so the folds at the ankle are rounder)
    friction=0.6,            # leather on wool
    gravity=(0.0, 0.0, 0.0), # (the boot stands on the leg: leather this stiff does not sag between its seams, and the
                             # solver's ties, walked all at once, cannot hold a sheet a hundred points long up
                             # against its weight in a few walks a step)
    iterations=12,
    gap=0.03,                # how far the leather is kept off the hose
    steps=120,
)
HOLD = {'sole': 0.30, 'collar': 0.60}   # how far up from the upper's lower edge, and down from its top, it is held
# The seams sewn: the top piece, the pieces beneath, the top piece's edge, the under pieces' edge. The foot's seams,
# then each shaft's (last.SHAFTS); a seam whose pieces a boot does not have is passed over.
SEAMS = (
    ('toe cap', ('vamp',), 'G', 'G'),
    ('inner quarter', ('vamp',), 'A inner', 'A inner'),
    ('outer quarter', ('vamp',), 'A outer', 'A outer'),
    ('outer quarter', ('inner quarter',), 'B', 'B'),
    ('tongue', ('vamp',), 'C', 'C'),
    ('inner shaft', ('vamp', 'inner quarter', 'outer quarter'), 'C', 'C'),
    ('outer shaft', ('vamp', 'inner quarter', 'outer quarter'), 'C', 'C'),
    ('inner shaft', ('vamp',), 'A inner', 'A inner'),            # (a shaft run down to the sole: its panels the quarters)
    ('outer shaft', ('vamp',), 'A outer', 'A outer'),
    ('outer shaft', ('inner shaft',), 'D', 'D'),
    ('inner shaft', ('tongue',), 'H', 'H'),
    ('outer shaft', ('tongue',), 'H', 'H'),
    ('inner binding', ('inner shaft',), 'F', 'F'),
    ('outer binding', ('outer shaft',), 'F', 'F'),
    ('inner binding', ('inner shaft',), 'top', None),
    ('outer binding', ('outer shaft',), 'top', None),
)
REACH = 0.35                 # a point is sewn to the nearest point of the other piece, if that is this near (units)
STITCH = {
    'in': 0.11,              # the row, this far in from the edge (about 3 mm)
    'pitch': 0.15,           # one stitch and the gap after it (about 4 mm)
    'length': 0.62,          # the stitch's share of the pitch
    'thread': 0.022,         # the thread's half width (a waxed linen thread, about 1 mm across)
    'flat': 0.55,            # its height over its width (it is pulled flat into the leather)
    'sunk': 0.35,            # how much of its height is sunk into the leather
}


def _edges(p):
    """A piece's seam edges as point masks, the shaft's back seam told apart into its top and under edges."""
    e = dict(p['edges'])
    if 'D' in e and 't' in p:
        e['D top'] = e['D'] & (p['t'] > 0.5)
        e['D under'] = e['D'] & (p['t'] < 0.5)
    return e


def _tree(pts):
    kd = KDTree(len(pts))
    for i, q in enumerate(pts):
        kd.insert(q, i)
    kd.balance()
    return kd


def pairs(pieces, X, log=print):
    """The points to be sewn together, as pairs of indices into the boot's mesh (X, its points)."""
    by = {p['name']: p for p in pieces}
    out = []
    for top, unders, e_top, e_under in SEAMS:
        if top not in by:                                      # (a seam of another shaft than this boot's)
            continue
        pt = by[top]
        if e_top not in _edges(pt):
            continue
        st = np.arange(pt['start'], pt['start'] + len(pt['V']))
        for un in unders:
            if un not in by:
                continue
            pu = by[un]
            su = np.arange(pu['start'], pu['start'] + len(pu['V']))
            if top == un:                                      # (the back seam: one piece, its two edges)
                tt = pt['t']
                a_from, a_to = st[_edges(pt)[e_top]], st[tt < 0.25]
                b_from, b_to = st[_edges(pt)[e_under]], st[tt > 0.75]
                todo = ((a_from, a_to), (b_from, b_to))
            else:
                todo = [(st[_edges(pt)[e_top]], su)]
                if e_under is not None and e_under in _edges(pu):
                    todo.append((su[_edges(pu)[e_under]], st))
            n0 = len(out)
            for frm, to in todo:
                if not len(frm) or not len(to):
                    continue
                kd = _tree(X[to])
                for i in frm:
                    q, j, d = kd.find(X[i])
                    if d < REACH:
                        out.append((int(i), int(to[j])))
            log('boot sewing: %s over %s at %s, %d points sewn' % (top, un, e_top.split()[0], len(out) - n0))
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def _normals(o):
    o.data.update()
    nv = np.empty(len(o.data.vertices) * 3)
    o.data.vertices.foreach_get('normal', nv)
    return nv.reshape(-1, 3)


def _co(o):
    x = np.empty(len(o.data.vertices) * 3)
    o.data.vertices.foreach_get('co', x)
    return x.reshape(-1, 3)


def sew(o, pieces, rig, colliders, P=LEATHER, log=print):
    """Sew the boot o (the upper's pieces, as last.build_boots made them) from its flat pieces on the colliders."""
    uppers = [p for p in pieces if p['name'] != 'sole']
    X, Nv = _co(o), _normals(o)
    n = len(X)
    lift = np.zeros(n)
    for p in uppers:
        lift[p['start']:p['start'] + len(p['V'])] = p['lift']
    base = X - Nv * lift[:, None]                              # (every piece on the last itself, while it is sewn)
    sew_ij = pairs(uppers, base, log=log)
    # held at the sole and the collar, firmly at the edge and less so further in
    z = base[:, 2]
    pin = np.maximum(np.clip(1.0 - (z - z.min()) / HOLD['sole'], 0.0, 1.0),
                     np.clip(1.0 - (z.max() - z) / HOLD['collar'], 0.0, 1.0)) ** 2
    # the hose, at rest, as the body the leather lies on
    from mathutils.bvhtree import BVHTree
    pts, tris = [], []
    for c in colliders:
        was = rig.data.pose_position if rig is not None else None
        if rig is not None:
            rig.data.pose_position = 'REST'
        bpy.context.view_layer.update()
        e = c.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = e.to_mesh()
        me.calc_loop_triangles()
        n0 = len(pts)
        pts += [tuple(c.matrix_world @ v.co) for v in me.vertices]
        tris += [tuple(n0 + i for i in t.vertices) for t in me.loop_triangles]
        e.to_mesh_clear()
        if rig is not None:
            rig.data.pose_position = was
    bpy.context.view_layer.update()
    body = fabric.bvh_body(BVHTree.FromPolygons(pts, tris), P['gap'])
    import time
    t0 = time.time()
    fab = fabric.Fabric(base, {k: P[k] for k in ('stretch', 'shear', 'bend', 'friction', 'iterations', 'gravity')})
    fab.pin(np.arange(n), pin)
    ties = fab.mesh([tuple(p.vertices) for p in o.data.polygons])
    # (each pair held at the distance it was cut at: the two pieces' points along a seam are not spaced alike, so
    # drawing each to the other's nearest point gathers the seam into scallops)
    fab.tie(sew_ij[:, 0], sew_ij[:, 1], 0.0)
    steps = fab.settle(body, most=P['steps'])
    Y = fab.pos.copy()
    log('boot sewing %s: settled by the fabric solver in %d steps (%.0f s): %d threads, %d diagonals, %d bend ties, '
        '%d seam ties' % (o.name, steps, time.time() - t0, ties[0], ties[1], ties[2], len(sew_ij)))
    gap = np.linalg.norm(Y[sew_ij[:, 0]] - Y[sew_ij[:, 1]], axis=1)
    e = np.empty(len(o.data.edges) * 2, dtype=np.int64)
    o.data.edges.foreach_get('vertices', e)
    e = e.reshape(-1, 2)
    k = np.linalg.norm(Y[e[:, 0]] - Y[e[:, 1]], axis=1) / np.maximum(np.linalg.norm(base[e[:, 0]] - base[e[:, 1]], axis=1), 1e-9)
    log('boot sewing %s: the seams close to %.3f on average (%.3f at most); its lengths %.0f%%..%.0f%% of the cut '
        '(5th to 95th percentile)' % (o.name, gap.mean(), gap.max(), 100 * np.percentile(k, 5), 100 * np.percentile(k, 95)))
    for p in uppers:
        sl = slice(p['start'], p['start'] + len(p['V']))
        d = Y[sl] - base[sl]
        log('boot sewing %s: %s moved %.3f on average (%.3f at most; down %.3f, out along its face %.3f)'
            % (o.name, p['name'], np.linalg.norm(d, axis=1).mean(), np.linalg.norm(d, axis=1).max(), -d[:, 2].mean(),
               (d * Nv[sl]).sum(1).mean()))
    # each piece set back over the one beneath
    Z = Y + (X - base)                                         # (by the same lift it was cut with)
    o.data.vertices.foreach_set('co', Z.ravel())
    if 'rest' in o.data.attributes:
        o.data.attributes['rest'].data.foreach_set('vector', Z.ravel())
    o.data.update()
    return {'gap': float(gap.max()), 'lengths': (float(np.percentile(k, 5)), float(np.percentile(k, 95)))}


def _chains(es):
    """Edges (a, b) chained into lines of points."""
    nb = {}
    for a, b in es:
        nb.setdefault(a, []).append(b)
        nb.setdefault(b, []).append(a)
    seen, lines = set(), []
    ends = [v for v, w in nb.items() if len(w) == 1] + list(nb)
    for s in ends:
        if s in seen:
            continue
        line, prev, cur = [s], None, s
        seen.add(s)
        while True:
            step = [v for v in nb[cur] if v != prev and v not in seen]
            if not step:
                break
            prev, cur = cur, step[0]
            seen.add(cur)
            line.append(cur)
        if len(line) > 2:
            lines.append(line)
    return lines


def _rows(o, pieces, sole):
    """The stitches' rows: for each, the points along it [m, 3] (on the leather's face, the row's inset taken), the
    leather's normal there, and whether the row is closed."""
    X, Nv = _co(o), _normals(o)
    by = {p['name']: p for p in pieces}
    tl = o.modifiers['thickness'].thickness if 'thickness' in o.modifiers else 0.0
    rows = []
    for top, _, e_top, _ in SEAMS:
        if e_top == 'top' or top not in by or e_top not in by[top]['edges']:
            continue
        p = by[top]
        sl = slice(p['start'], p['start'] + len(p['V']))
        V, Nn = X[sl], Nv[sl]
        piece = dict(V=V, F=p['F'], edges={e_top: _edges(p)[e_top]})
        third = {}
        for f in p['F']:
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                third[(min(a, b), max(a, b))] = f[(k + 2) % len(f)]
        for loop in pattern._outline(p['F']):
            for es in pattern._seam_runs(piece, loop).values():
                for line in _chains(es):
                    P = V[line]
                    inward = np.zeros_like(P)
                    for k in range(len(line) - 1):
                        a, b = line[k], line[k + 1]
                        c = third.get((min(a, b), max(a, b)))
                        if c is None:
                            continue
                        t = V[b] - V[a]
                        t /= max(np.linalg.norm(t), 1e-9)
                        w = V[c] - V[a]
                        w -= t * np.dot(w, t)
                        inward[k] += w
                        inward[k + 1] += w
                    inward /= np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), 1e-9)
                    rows.append((P + inward * STITCH['in'] + Nn[line] * tl, Nn[line], False))
    if sole is not None and 'stitch' in sole:
        path = np.asarray(sole['stitch'])
        rows.append((path, np.tile([0.0, 0.0, 1.0], (len(path), 1)), True))
    return rows


def stitches(o, pieces, rig=None, material=None, name=None, log=print):
    """The stitches of the boot o (sewn), as their own object on the same bones."""
    sole = next((p for p in pieces if p['name'] == 'sole'), None)
    rows = _rows(o, [p for p in pieces if p['name'] != 'sole'], sole)
    S = STITCH
    half, r, hgt = 0.5 * S['length'] * S['pitch'], S['thread'], S['thread'] * S['flat']
    ring = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    along = ((-0.62, 0.78), (0.0, 1.0), (0.62, 0.78))            # (where the rings sit along it, and their size)
    verts, faces, count = [], [], 0
    for P, Nn, closed in rows:
        if closed:
            P, Nn = np.concatenate([P, P[:1]]), np.concatenate([Nn, Nn[:1]])
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        if arc[-1] < S['pitch']:
            continue
        for s in np.arange(0.5 * S['pitch'], arc[-1] - 0.25 * S['pitch'], S['pitch']):
            k = min(int(np.searchsorted(arc, s)) - 1, len(seg) - 1)
            k = max(k, 0)
            f = (s - arc[k]) / max(seg[k], 1e-9)
            c = P[k] + (P[k + 1] - P[k]) * f
            nrm = Nn[k] + (Nn[k + 1] - Nn[k]) * f
            nrm /= max(np.linalg.norm(nrm), 1e-9)
            t = P[k + 1] - P[k]
            t -= nrm * np.dot(t, nrm)
            t /= max(np.linalg.norm(t), 1e-9)
            b = np.cross(nrm, t)
            c = c + nrm * hgt * (1.0 - 2.0 * S['sunk'])
            n0 = len(verts) * 1
            pts = [c - t * half]
            for u, sz in along:
                for a in ring:
                    pts.append(c + t * u * half + (np.cos(a) * nrm * hgt + np.sin(a) * b * r) * sz)
            pts.append(c + t * half)
            verts.extend(pts)
            base = n0 + 1
            for i in range(8):
                j = (i + 1) % 8
                faces.append([n0, base + j, base + i])
                for g in range(2):
                    faces.append([base + g * 8 + i, base + g * 8 + j, base + (g + 1) * 8 + j, base + (g + 1) * 8 + i])
                faces.append([n0 + 25, base + 16 + i, base + 16 + j])
            count += 1
    V = np.array(verts)
    st = _on_boot(o, V, faces, name or o.name.replace('Boot', 'Boot stitches'), material, rig)
    log('boot stitches %s: %d stitches in %d rows, %d points' % (o.name, count, len(rows), len(V)))
    return st


def _on_boot(o, V, faces, name, material, rig):
    """V, faces as their own object on the boot o's bones: each point follows the leather nearest it."""
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], faces)
    me.validate()
    for p in me.polygons:
        p.use_smooth = True
    st = bpy.data.objects.new(me.name, me)
    bpy.context.scene.collection.objects.link(st)
    if material is not None:
        me.materials.append(material)
    a = me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
    a.data.foreach_set('vector', V.ravel().astype(np.float32))
    # the bones: each stitch's points follow the leather nearest them
    X = _co(o)
    kd = _tree(X)
    near = np.array([kd.find(q)[1] for q in V])
    W = np.zeros((len(X), len(o.vertex_groups)), dtype=np.float32)
    for v in o.data.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    Wk = W[near]
    for gi, g0 in enumerate(o.vertex_groups):
        w = Wk[:, gi]
        if g0.name != 'skive' and (w > 1e-4).any():
            g = st.vertex_groups.new(name=g0.name)
            for val in np.unique(np.round(w[w > 1e-4], 3)):
                g.add([int(i) for i in np.nonzero(np.abs(np.round(w, 3) - val) < 1e-6)[0]], float(val), 'REPLACE')
    if rig is not None:
        src = next((m for m in o.modifiers if m.type == 'ARMATURE'), None)
        m = st.modifiers.new('rig', 'ARMATURE')
        m.object = rig
        if src is not None:
            m.use_deform_preserve_volume = src.use_deform_preserve_volume
    return st


def _tube(path, rad, sides=6, ring=False):
    """A round tube along path [m, 3], rad across, its ends closed (or, ring, the path a closed loop with its last
    point its first). Returns (points, faces)."""
    t = np.gradient(path, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    ref = np.where(np.abs(t[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    u = np.cross(t, ref)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    w = np.cross(t, u)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    V = (path[:, None, :] + rad * (np.cos(ang)[None, :, None] * u[:, None, :] + np.sin(ang)[None, :, None] * w[:, None, :]))
    V = np.concatenate([V.reshape(-1, 3), path[:1], path[-1:]])
    m = len(path)
    F = [[k * sides + i, k * sides + (i + 1) % sides, (k + 1) * sides + (i + 1) % sides, (k + 1) * sides + i]
         for k in range(m - 1) for i in range(sides)]
    if ring:
        return V[:-2], F
    c0, c1 = m * sides, m * sides + 1
    F += [[c0, (i + 1) % sides, i] for i in range(sides)]
    F += [[c1, (m - 1) * sides + i, (m - 1) * sides + (i + 1) % sides] for i in range(sides)]
    return V, F


def lacing(o, pieces, rig=None, ratios=None, H=None, metal=None, cord=None, log=print):
    """The front's eyelets (a brass ring set through each panel at each of its eyelet points) and the lace crossing
    between them over the tongue, as two objects on the boot's bones. Returns (eyelets, lace)."""
    import last
    r = {k: v * H for k, v in (ratios or last.RATIOS).items()}
    X, Nv = _co(o), _normals(o)
    tl = o.modifiers['thickness'].thickness if 'thickness' in o.modifiers else 0.0
    by = {p['name']: p for p in pieces}
    if not any('eyelets' in p for p in pieces):
        return None, None
    rows = {}
    for q in ('inner', 'outer'):
        p = by[q + ' shaft']
        g = p['start'] + p['eyelets']
        rows[q] = (X[g] + Nv[g] * tl, Nv[g])
    # hand-punched, not stepped off by a machine: each eyelet sits a little off its place along the edge and in from
    # it, and the rings differ a little in size
    side_ = o.name.rsplit('.', 1)[-1]
    rng = np.random.default_rng(sum(map(ord, side_)) + 17)
    pitch = r['eyelet_pitch']
    R = r['eyelet']
    for q in rows:
        P, N = rows[q]
        P = P.copy()
        for k in range(len(P)):
            t = P[min(k + 1, len(P) - 1)] - P[max(k - 1, 0)]
            t /= max(np.linalg.norm(t), 1e-9)
            u = np.cross(N[k], t)
            P[k] = P[k] + t * rng.normal(0, 0.07 * pitch) + u * rng.normal(0, 0.22 * R) * (1 if q == 'outer' else -1)
        rows[q] = (P, N)
    sizes = {q: R * (1 + rng.normal(0, 0.05, len(rows[q][0]))) for q in rows}
    # the eyelets: rings lying on the leather, a tube round the hole
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    EV, EF = [], []
    for q, (P, N) in rows.items():
        for c, nrm, Rk in zip(P, N, sizes[q]):
            tube = 0.32 * Rk
            u = np.cross(nrm, [0.0, 0.0, 1.0])
            u /= max(np.linalg.norm(u), 1e-9)
            path = c[None, :] + (Rk - tube) * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * np.cross(nrm, u))
            path = np.concatenate([path, path[:1]]) + nrm * 0.4 * tube
            v, f = _tube(path, tube, 8, ring=True)
            EF += [[i + len(EV) for i in ff] for ff in f]
            EV += list(v)
    eyelets = _on_boot(o, np.array(EV), EF, o.name.replace('Boot', 'Boot eyelets'), metal, rig)
    # the lace: a cord that goes down into each hole, crosses the opening to the next hole up on the other side
    # (one strand over the other at every crossing, as it is threaded), with its own slack and a width that varies
    from mathutils.bvhtree import BVHTree
    tree = BVHTree.FromPolygons([tuple(v) for v in X], [tuple(p.vertices) for p in o.data.polygons])

    def onto(p, off):
        hit = tree.find_nearest(tuple(p))
        return np.array(hit[0]) + np.array(hit[1]) * off if hit[0] is not None else p

    (Pi, Ni), (Po, No) = rows['inner'], rows['outer']
    LV, LF = [], []

    def strand(path, rad):
        v, f = _tube(path, np.asarray(rad)[:, None, None] if np.ndim(rad) else rad, 8)
        LF.extend([[i + len(LV) for i in ff] for ff in f])
        LV.extend(list(v))

    n = min(len(Pi), len(Po))
    rl = r['lace']
    s = np.linspace(0.0, 1.0, 13)[:, None]
    for k in range(n - 1):
        for j, ((A, NA), (B, NB)) in enumerate((((Pi[k], Ni[k]), (Po[k + 1], No[k + 1])),
                                                 ((Po[k], No[k]), (Pi[k + 1], Ni[k + 1])))):
            slack = rng.uniform(0.0, 0.5) * rl
            over = (1.0 if (j + k) % 2 == 0 else -0.2) * rl             # (this strand over the other, or under it)
            up = 0.5 * (NA + NB)
            mid = 0.5 * (A + B) + up * (rl * 0.9 + over + slack)
            mid = mid + (B - A) * rng.normal(0, 0.035)                    # (the crossing drifts a little up or down)
            path = (1 - s) ** 2 * (A - NA * 0.8 * rl) + 2 * s * (1 - s) * mid + s ** 2 * (B - NB * 0.8 * rl)
            # the cord lies over the curve of the shin: every point between the holes is brought onto the tongue's
            # surface and held a cord's width over it, the over strand a cord higher
            lift = rl * (1.1 + slack / rl * 0.5 + np.where(over > 0, 1.0, -0.1))
            for i in range(2, len(path) - 2):
                path[i] = onto(path[i], float(lift) if np.ndim(lift) == 0 else float(lift))
            for _ in range(2):                                     # (smoothed so it runs as one curve)
                path[2:-2] = 0.25 * path[1:-3] + 0.5 * path[2:-2] + 0.25 * path[3:-1]
            wob = 1 + 0.07 * np.sin(2 * np.pi * (s[:, 0] * rng.uniform(1, 2.5) + rng.uniform()))
            ends = np.minimum(1.0, 0.55 + 6 * np.minimum(s[:, 0], 1 - s[:, 0]))      # (the cord thins into the hole)
            strand(path, rl * wob * ends * (1.1 if j == 0 else 1.0))
    # the top: the two ends come out of the top holes and hang loose, long, one over the other, curving over the
    # panels and the foot's top as the leather does
    A, NA, B, NB = Pi[n - 1], Ni[n - 1], Po[n - 1], No[n - 1]
    nn = 0.5 * (NA + NB)
    nn /= np.linalg.norm(nn)
    es = (B - A) / max(np.linalg.norm(B - A), 1e-9)
    et = np.cross(nn, es)
    et *= 1.0 if et[2] > 0 else -1.0
    for sd, P_, N_, Lk in ((-1, A, NA, 5.5), (1, B, NB, 7.5)):
        tt = np.linspace(0, 1, 40)
        Lc = Lk * pitch
        pts = []
        for x in tt:
            q = P_ + sd * es * (0.35 * Lc * x * (1 - 0.5 * x) + 0.1 * Lc * np.sin(3.1 * x)) - et * (Lc * x ** 1.1) \
                + 0.5 * rl * np.sin(9 * x) * es
            pts.append(onto(q, rl * 1.3 + 0.6 * rl * np.sin(np.pi * x)) if x > 0.04 else P_ - N_ * 0.8 * rl)
        pts = np.array(pts)
        for _ in range(3):
            pts[2:-2] = 0.25 * pts[1:-3] + 0.5 * pts[2:-2] + 0.25 * pts[3:-1]
        strand(pts, rl * (1 - 0.1 * tt))
    lace = _on_boot(o, np.array(LV), LF, o.name.replace('Boot', 'Boot lace'), cord, rig)
    log('boot lacing %s: %d eyelets a side, %d crossings of the lace' % (o.name, n, 2 * (n - 1)))
    return eyelets, lace


CREASE = dict(bend=16.0, steps=360, hold_sole=0.45, hold_collar=1.2, gain=0.4)  # (the ankle's bend in degrees, and the holds)


def _deform(o, rig):
    """Each point's deforming matrix on the rig as it is posed now (the bones' blend, as the armature moves it)."""
    groups = {g.index: g.name for g in o.vertex_groups}
    mats = {}
    for pb in rig.pose.bones:
        mats[pb.name] = np.array(rig.matrix_world @ pb.matrix @ pb.bone.matrix_local.inverted() @ rig.matrix_world.inverted())
    out = np.zeros((len(o.data.vertices), 4, 4))
    tot = np.zeros(len(o.data.vertices))
    for v in o.data.vertices:
        for g in v.groups:
            nm = groups.get(g.group)
            if nm in mats and g.weight > 0:
                out[v.index] += g.weight * mats[nm]
                tot[v.index] += g.weight
    none = tot < 1e-6
    out[~none] /= tot[~none][:, None, None]
    out[none] = np.eye(4)
    return out


def crease(o, pieces, rig, colliders, ankle_bend=None, zone=None, P=LEATHER, C=CREASE, log=print):
    """Crease the sewn boot o where a foot bends it, and set the creases in, as a worn boot has them: the ankle is
    bent (dorsiflexed by C['bend'] degrees), the leather carried along by the bones is drawn back to the lengths it
    was cut at by the fabric solver against the hose (so the leather the bend gathers over the front of the ankle
    folds out, with the sole and the collar held), and the folded leather is taken back to the rest pose through
    the bones and kept as the boot's shape there, over the front of the ankle only (zone: its lowest and highest
    height, and the leg's middle front to back)."""
    import time
    from mathutils.bvhtree import BVHTree
    uppers = [p for p in pieces if p['name'] != 'sole']
    X = _co(o)
    n = len(X)
    rest_ij = pairs(uppers, X, log=lambda *a: None)
    side = o.name.rsplit('.', 1)[-1]
    # the folds are cached: the boot's points as sewn, the bend, the zone and the leather's numbers make the key, so
    # a run that changes only the look (the texture, the lights) takes the folds it had and skips the solver
    import hashlib
    key = hashlib.sha1(np.round(X, 4).tobytes() + repr((C, ankle_bend, zone, sorted((k, P[k]) for k in P))).encode()
                       ).hexdigest()[:16]
    cache = os.path.join(os.environ.get('BOOT_CACHE', '/mnt/project-files/hose/boot-cache'), '%s-%s.npy' % (side, key))
    if os.path.exists(cache) and not os.environ.get('BOOT_NOCACHE'):
        Z = np.load(cache)
        if Z.shape == X.shape:
            Z = X + (Z - X) * C.get('gain', 1.0)
            o.data.vertices.foreach_set('co', Z.ravel())
            if 'rest' in o.data.attributes:
                o.data.attributes['rest'].data.foreach_set('vector', Z.ravel())
            o.data.update()
            log('boot crease %s: the folds taken from the cache (%s), no solve' % (o.name, os.path.basename(cache)))
            return {'moved': 0.0, 'cached': True}
    an = bpy.data.objects.get('Ankle.' + side)
    if rig is None or an is None:
        log('boot crease %s: no rig, nothing to bend' % o.name)
        return None
    was_pos, was_rot = rig.data.pose_position, tuple(an.rotation_euler)
    rig.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    rig.data.pose_position = 'POSE'
    bend = np.radians(C['bend'] if ankle_bend is None else ankle_bend)
    an.rotation_euler = (was_rot[0] - bend, was_rot[1], was_rot[2])         # (the toe up)
    bpy.context.view_layer.update()
    Mv = _deform(o, rig)
    Xh = np.concatenate([X, np.ones((n, 1))], 1)
    Y0 = np.einsum('nij,nj->ni', Mv, Xh)[:, :3]
    toe = np.argmin(X[:, 1])
    log('boot crease %s: the ankle bent %.0f deg; the toe goes %+.2f up' % (o.name, np.degrees(bend), Y0[toe, 2] - X[toe, 2]))
    pts, tris = [], []
    for c in colliders:
        e = c.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = e.to_mesh()
        me.calc_loop_triangles()
        n0 = len(pts)
        pts += [tuple(c.matrix_world @ v.co) for v in me.vertices]
        tris += [tuple(n0 + i for i in t.vertices) for t in me.loop_triangles]
        e.to_mesh_clear()
    body = fabric.bvh_body(BVHTree.FromPolygons(pts, tris), P['gap'])
    t0 = time.time()
    fab = fabric.Fabric(X, {k: P[k] for k in ('stretch', 'shear', 'bend', 'friction', 'iterations', 'gravity')})
    z = X[:, 2]
    pin = np.maximum(np.clip(1.0 - (z - z.min()) / C['hold_sole'], 0.0, 1.0),
                     np.clip(1.0 - (z.max() - z) / C['hold_collar'], 0.0, 1.0)) ** 2
    fab.pin(np.arange(n), pin)
    fab.mesh([tuple(p.vertices) for p in o.data.polygons])         # (the ties at the lengths as cut, at rest)
    fab.tie(rest_ij[:, 0], rest_ij[:, 1], 0.0)
    fab.pos[:] = Y0
    fab.prev[:] = Y0
    steps = fab.settle(body, most=C['steps'])
    Y = fab.pos.copy()
    d = np.linalg.norm(Y - Y0, axis=1)
    out = ((Y - Y0) * _normals(o)).sum(1)
    log('boot crease %s: folded by the solver in %d steps (%.0f s): the leather moved %.3f on average, %.3f at most '
        '(%.3f at most out from the leg)' % (o.name, steps, time.time() - t0, d.mean(), d.max(), out.max()))
    # back to the rest pose through each point's own bones, the folds kept
    Yh = np.concatenate([Y, np.ones((n, 1))], 1)
    Z = np.einsum('nij,nj->ni', np.linalg.inv(Mv), Yh)[:, :3]
    an.rotation_euler = was_rot
    rig.data.pose_position = was_pos
    bpy.context.view_layer.update()
    # the folds are kept only where a foot's bend makes them: over the front of the ankle. Anywhere else the
    # solver's moves are its settling (ripples under the cuff, a fold down the heel), not wear.
    if zone is not None:
        z0, z1, y_mid = zone
        wz = _smooth(z0 - 0.5, z0, X[:, 2]) * (1.0 - _smooth(z1, z1 + 1.2, X[:, 2]))
        wy = _smooth(y_mid + 0.3, y_mid - 0.9, X[:, 1])
        Z = X + (Z - X) * (wz * wy)[:, None]
    o.data.vertices.foreach_set('co', Z.ravel())
    if 'rest' in o.data.attributes:
        o.data.attributes['rest'].data.foreach_set('vector', Z.ravel())
    o.data.update()
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, Z)                                          # (whole: the gain is taken on each use)
    except OSError as err:
        log('boot crease %s: the folds could not be cached (%s)' % (o.name, err))
    Z = X + (Z - X) * C.get('gain', 1.0)
    o.data.vertices.foreach_set('co', Z.ravel())
    if 'rest' in o.data.attributes:
        o.data.attributes['rest'].data.foreach_set('vector', Z.ravel())
    o.data.update()
    kept = np.linalg.norm(Z - X, axis=1)
    log('boot crease %s: the folds kept over the front of the ankle: %.3f on average, %.3f at most'
        % (o.name, kept.mean(), kept.max()))
    return {'moved': float(d.max())}


def _smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
