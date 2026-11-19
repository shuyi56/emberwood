"""The boot sewn: its pattern pieces (last.PIECES) joined at their seams by the cloth solver, and its stitches.

    sew()       The pieces are put on where they were cut on the last, over the hose. The foot's pieces (the toe
                cap, the vamp, the quarters) are lasted: wetted, stretched over the last and left to dry on it, so
                the shape they hold to is the last's. The shaft is cut flat and rolled round the leg (its flat
                pattern is within a few percent of its lengths on the last, so it is drafted at those). Every lapped
                seam is sewn along both its edges: the top piece's edge to the piece beneath it, the under piece's
                edge to the top piece (SEAMS). The upper's lower edge is sewn to the sole and the collar is held up
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
    bend=2e-6,
    friction=0.6,            # leather on wool
    gravity=(0.0, 0.0, 0.0), # (the boot stands on the leg: leather this stiff does not sag between its seams, and the
                             # solver's ties, walked all at once, cannot hold a sheet a hundred points long up
                             # against its weight in a few walks a step)
    iterations=12,
    gap=0.03,                # how far the leather is kept off the hose
    steps=120,
)
HOLD = {'sole': 0.30, 'collar': 0.60}   # how far up from the upper's lower edge, and down from its top, it is held
# The seams sewn: the top piece, the pieces beneath, the top piece's edge, the under pieces' edge.
SEAMS = (
    ('toe cap', ('vamp',), 'G', 'G'),
    ('inner quarter', ('vamp',), 'A inner', 'A inner'),
    ('outer quarter', ('vamp',), 'A outer', 'A outer'),
    ('outer quarter', ('inner quarter',), 'B', 'B'),
    ('shaft', ('vamp', 'inner quarter', 'outer quarter'), 'C', 'C'),
    ('shaft', ('shaft',), 'D top', 'D under'),
    ('binding', ('shaft',), 'F', 'F'),
    ('binding', ('shaft',), 'top', None),
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
        pt = by[top]
        st = np.arange(pt['start'], pt['start'] + len(pt['V']))
        for un in unders:
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
    fab.tie(sew_ij[:, 0], sew_ij[:, 1], 0.0, rest=0.0)
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
    o.data.vertices.foreach_set('co', Y.ravel())
    Z = Y + _normals(o) * lift[:, None]
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
        if e_top == 'top':
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
    me = bpy.data.meshes.new(name or o.name.replace('Boot', 'Boot stitches'))
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
        if (w > 1e-4).any():
            g = st.vertex_groups.new(name=g0.name)
            for val in np.unique(np.round(w[w > 1e-4], 3)):
                g.add([int(i) for i in np.nonzero(np.abs(np.round(w, 3) - val) < 1e-6)[0]], float(val), 'REPLACE')
    if rig is not None:
        src = next((m for m in o.modifiers if m.type == 'ARMATURE'), None)
        m = st.modifiers.new('rig', 'ARMATURE')
        m.object = rig
        if src is not None:
            m.use_deform_preserve_volume = src.use_deform_preserve_volume
    log('boot stitches %s: %d stitches in %d rows, %d points' % (o.name, count, len(rows), len(V)))
    return st
