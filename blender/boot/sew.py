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
HOLD = {'sole': 0.30, 'collar': 0.25}
# sewing bends flat sheets round the leg: a stiffer bend than the creases', walked longer, so a sheet wraps in one
# smooth curve and does not ripple
SEWN = dict(LEATHER, bend=2e-7, stretch=5e-4, shear=5e-4, iterations=16, steps=200)   # (the hide gives a few per cent, as leather pulled over a last does, where a flat sheet cannot follow it)
FLAT = ('inner shaft', 'outer shaft', 'tongue')      # the pieces cut and kept as flat sheets (sew)   # how far up from the upper's lower edge, and down from its top, it is held
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



def _even(Y, base, o, n):
    """What each point moved by in the solve, evened with its neighbours' (the solver's walk leaves a fine jitter, a
    millimetre or so from point to point, that a hide does not have): the wrap's shape kept, the jitter taken."""
    ee = np.empty(len(o.data.edges) * 2, dtype=np.int64)
    o.data.edges.foreach_get('vertices', ee)
    ee = ee.reshape(-1, 2)
    dsp = Y - base
    for _ in range(8):
        acc, cnt = np.zeros_like(dsp), np.zeros(n)
        np.add.at(acc, ee[:, 0], dsp[ee[:, 1]])
        np.add.at(acc, ee[:, 1], dsp[ee[:, 0]])
        np.add.at(cnt, ee[:, 0], 1.0)
        np.add.at(cnt, ee[:, 1], 1.0)
        dsp = 0.4 * dsp + 0.6 * acc / np.maximum(cnt, 1.0)[:, None]
    return base + dsp

def sew(o, pieces, rig, colliders, P=None, log=print):
    """Sew the boot o (the upper's pieces, as last.build_boots made them) from its flat pieces on the colliders."""
    P = SEWN if P is None else P
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
    # the leg's pieces (FLAT) are flat sheets: the solver is given each one's lengths as it lies flat on the pattern
    # (pattern.flatten: the piece peeled off the last and pressed flat, at its true size), so on the leg it can only
    # bend, as a sheet of leather does, and what roundness it has comes from being wrapped; the foot's pieces (the
    # vamp, the toe cap) keep the last's lengths, as leather stretched over the last does
    import pattern
    flat = base.copy()
    for p in uppers:
        if p['name'] in FLAT:
            uv, give = pattern.flatten(p)
            sl = slice(p['start'], p['start'] + len(p['V']))
            flat[sl] = np.column_stack([uv, np.zeros(len(uv))])
            log('boot sewing: %s cut flat (flat lengths %.0f%%..%.0f%% of its lengths on the last)'
                % (p['name'], 100 * give[0], 100 * give[2]))
    # the sewn shell is baked: the pieces as cut (on the last and flat), the leather's numbers and the hose make the
    # key, so a later run (a new tongue's bend, a texture, the lights) takes the panels' folds as they are
    import hashlib
    side = o.name.rsplit('.', 1)[-1]
    key = hashlib.sha1(np.round(base, 4).tobytes() + np.round(flat, 4).tobytes() + np.round(np.array(pts), 3).tobytes()
                       + repr(sorted((k, P[k]) for k in P)).encode() + repr(sorted(HOLD.items())).encode()
                       + b'smooth8').hexdigest()[:16]
    cache = os.path.join(os.environ.get('BOOT_CACHE', '/mnt/project-files/hose/boot-cache'), 'sew-%s-%s.npy' % (side, key))
    fab = fabric.Fabric(flat, {k: P[k] for k in ('stretch', 'shear', 'bend', 'friction', 'iterations', 'gravity')})
    fab.pin(np.arange(n), pin)
    ties = fab.mesh([tuple(p.vertices) for p in o.data.polygons])
    fab.pos[:] = base                                           # (then set on the last, to be drawn to those lengths)
    fab.prev[:] = base
    # (each pair held at the distance it was cut at: the two pieces' points along a seam are not spaced alike, so
    # drawing each to the other's nearest point gathers the seam into scallops)
    fab.tie(sew_ij[:, 0], sew_ij[:, 1], 0.0)
    if os.path.exists(cache) and not os.environ.get('BOOT_NOCACHE'):
        Y = np.load(cache)
        log('boot sewing %s: the sewn shell taken from the cache (%s), no solve' % (o.name, os.path.basename(cache)))
        steps = 0
    else:
        steps = fab.settle(body, most=P['steps'])
        Y = _even(fab.pos.copy(), base, o, n)
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.save(cache, Y)
        except OSError:
            pass
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
    # (the two panels' rows pair up from the bottom: an eyelet with none to cross to would stand empty at the top)
    n_pair = min(len(rows['inner'][0]), len(rows['outer'][0]))
    rows = {q: (rows[q][0][:n_pair], rows[q][1][:n_pair]) for q in rows}
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
            P[k] = P[k] + t * rng.normal(0, 0.03 * pitch) + u * rng.normal(0, 0.10 * R) * (1 if q == 'outer' else -1)
        Ns_ = np.array([N[max(0, k - 2):k + 3].mean(0) for k in range(len(N))])     # (the row's normals smoothed: a
        N = Ns_ / np.maximum(np.linalg.norm(Ns_, axis=1, keepdims=True), 1e-9)       # vertex normal alone is noisy)
        rows[q] = (P, N)
    sizes = {q: R * (1 + rng.normal(0, 0.05, len(rows[q][0]))) for q in rows}
    # the eyelets: rings lying on the leather, a tube round the hole
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    EV, EF = [], []
    for q, (P, N) in rows.items():
        for c, nrm, Rk in zip(P, N, sizes[q]):
            tube = 0.17 * Rk                 # (a thin ring: it barely stands over the leather round the hole)
            u = np.cross(nrm, [0.0, 0.0, 1.0])
            u /= max(np.linalg.norm(u), 1e-9)
            path = c[None, :] + (Rk - tube) * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * np.cross(nrm, u))
            path = np.concatenate([path, path[:1]]) + nrm * 0.2 * tube
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
        if hit[0] is None:
            return p
        nrm = np.array(hit[1])
        out = Nv[list(o.data.polygons[hit[2]].vertices)].mean(axis=0)      # (the side the leather's normals face)
        if nrm @ out < 0:
            nrm = -nrm
        return np.array(hit[0]) + nrm * (off + tl)       # (the leather's face: its sheet and its thickness)

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
            slack = rng.uniform(0.3, 1.3) * rl
            # the cord is drawn from hole to hole along the leather's own curve: a Hermite curve that leaves each
            # hole along the surface there (so it arches over the shin as the contour does, not across the chord),
            # a cord's width over it; at every crossing the same strand lies on top, a full cord over the other
            d = B - A
            tA = 0.7 * (d - NA * (d @ NA))
            tB = 0.7 * (d - NB * (d @ NB))
            h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
            path = h00 * A + h10 * tA + h01 * B + h11 * tB
            Ns = (1 - s) * NA + s * NB
            Ns /= np.linalg.norm(Ns, axis=1, keepdims=True)
            w = np.minimum(1.0, 6 * np.minimum(s[:, 0], 1 - s[:, 0]))[:, None]        # (down into the hole at each end)
            # drawn tight, the cord lies on whatever is under it: the panels near the holes, the tongue down in the
            # opening (a leather's thickness and the panels' lap lower), so each point is laid on the face found
            # straight under it, a cord's radius off it; the strand on top rides over the other only where they
            # cross, in the middle. The heights are eased along the cord (a taut cord does not follow every step)
            # but never below the face
            over = 2.0 * rl * np.exp(-((s[:, 0] - 0.5) / 0.2) ** 2) if j == 0 else 0.0
            face = np.empty(len(path))
            for i in range(len(path)):
                q = path[i] + Ns[i] * 8.0 * rl
                hit = tree.ray_cast(tuple(q), tuple(-Ns[i]), 30.0 * rl)
                face[i] = (np.array(hit[0]) - path[i]) @ Ns[i] + tl if hit[0] is not None else rl
            seat = face + rl
            eased = seat.copy()
            for _ in range(3):
                eased[1:-1] = 0.25 * eased[:-2] + 0.5 * eased[1:-1] + 0.25 * eased[2:]
            seat = np.maximum(eased, face + 0.9 * rl) + over
            hole = rl * (1.1 + 0.05 * w[:, 0])
            lift = (w[:, 0] * seat + (1.0 - w[:, 0]) * hole + 0.15 * slack * np.sin(np.pi * s[:, 0]))[:, None]
            path = path + Ns * lift
            wob = 1 + 0.07 * np.sin(2 * np.pi * (s[:, 0] * rng.uniform(1, 2.5) + rng.uniform()))
            ends = np.minimum(1.0, 0.55 + 6 * np.minimum(s[:, 0], 1 - s[:, 0]))      # (the cord thins into the hole)
            strand(path, rl * wob * ends * (1.05 if j == 0 else 1.0))
    # the finish at the top: the two ends come out of the top holes and are tied in a small bow over the opening, its
    # loops spread over the panels and its two tails of unequal length draped down over them, outside the eyelet rows
    # so they do not lie across the cords below. (EXPERIMENTAL: no source gives a bow's proportions; the one rule
    # found, realmenrealstyle.com's "if the ends of the bow reach past the edge of the shoe, your laces are too
    # long", is kept: nothing here goes past the boot's front. Sizes in cm, about 2.6 cm to the unit.)
    cm = 1.0 / 2.619
    E0, E1 = Pi[n - 1], Po[n - 1]
    K0 = 0.5 * (E0 + E1)
    hk = tree.find_nearest(tuple(K0))
    NK = np.array(hk[1])
    if NK @ Nv[list(o.data.polygons[hk[2]].vertices)].mean(axis=0) < 0:
        NK = -NK
    ex = (E1 - E0) - NK * ((E1 - E0) @ NK)
    ex /= max(np.linalg.norm(ex), 1e-9)
    ey = np.array([0.0, 0.0, 1.0]) - NK * NK[2]
    ey /= max(np.linalg.norm(ey), 1e-9)
    K0 = K0 + ey * 0.55 * cm                               # (the knot a little above the top pair of holes)

    def at(a, v, off=1.05):                                # (a point a cm across and v cm up from the knot, on the leather)
        return onto(K0 + ex * a * cm + ey * v * cm, off * rl)

    def run(pts, rad):
        pts = np.array(pts)
        if len(pts) > 3:
            for _ in range(2):
                pts[1:-1] = 0.25 * pts[:-2] + 0.5 * pts[1:-1] + 0.25 * pts[2:]
        strand(pts, rad)
    t_ = np.linspace(0.0, 1.0, 14)
    for E in (E0, E1):                                     # (each end from its hole up to the knot)
        pts = [onto((1 - u) * E + u * K0, 1.05 * rl) for u in t_]
        run(pts, rl * np.minimum(1.0, 0.6 + 5 * t_))
    run([at(-0.35 + 0.7 * u, 0.0, 1.25) for u in np.linspace(0, 1, 5)], 1.35 * rl)       # (the knot)
    # the loops and tails are laid out by hand (a bow's shape), then let fall: the local cloth solver (fabric.py) runs
    # each as a chain of ties (the cord's length kept, a soft bend) pinned at the knot, under gravity, held off the
    # leather by the cord's radius and thickness, so they hang and settle as a cord does on a boot. (EXPERIMENTAL
    # settings: a waxed lace's stiffness is a guess.)
    chains = []                                            # (points, pinned count at the start, at the end)
    for sd, ang, Ll in ((-1.0, 16.0, 2.9), (1.0, 24.0, 2.5)):                           # (the loops)
        ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        ph = np.linspace(0.0, 2 * np.pi, 26)
        bx = 0.5 * Ll * (1 - np.cos(ph))
        by = 0.36 * Ll * np.sin(ph) * np.sqrt(np.abs(np.sin(ph / 2)))
        chains.append((np.array([at(sd * (ca * x - sa * y), sa * x + ca * y, 1.1 + 0.3 * np.sin(0.5 * p_))
                                 for x, y, p_ in zip(bx, by, ph)]), 2, 2))
    for sd, T, wig in ((-1.0, 12.5, 1.3), (1.0, 9.5, -1.0)):                          # (the tails)
        u = np.linspace(0.0, 1.0, 34)
        aa = sd * (2.7 * (1 - np.exp(-4.0 * u)) + 0.5 * u + wig * np.sin(np.pi * 1.6 * u) * u)
        vv = -T * u
        chains.append((np.array([at(a_, v_, 1.1 + 0.2 * uu) for a_, v_, uu in zip(aa, vv, u)]), 2, 0))
    if DRAPE:
        import fabric
        import time as _t
        t0 = _t.time()
        pts = np.concatenate([c[0] for c in chains])
        fb = fabric.Fabric(pts, dict(bend=DRAPE['bend'], stretch=0.0, friction=DRAPE['friction'], iterations=16,
                                          gravity=(0.0, 0.0, -375.0 * DRAPE['g'])))
        at0, pin = 0, []
        for c, p0, p1 in chains:
            m = len(c)
            fb.tie(np.arange(at0, at0 + m - 1), np.arange(at0 + 1, at0 + m))
            fb.tie(np.arange(at0, at0 + m - 2), np.arange(at0 + 2, at0 + m), DRAPE['bend'])
            if p1 == 2 and not DRAPE['loops']:
                pin += list(range(at0, at0 + m))                     # a loop keeps its hand-laid shape
            else:
                pin += list(range(at0, at0 + p0)) + list(range(at0 + m - p1, at0 + m))
            at0 += m
        fb.pin(pin)
        gap = 1.05 * rl + tl

        def body(x):
            out, touch = x.copy(), np.zeros(len(x), dtype=bool)
            for i in range(len(x)):
                hit = tree.find_nearest(tuple(x[i]), 6.0 * gap + 0.5)
                if hit[0] is None:
                    continue
                nrm = np.array(hit[1])
                if nrm @ Nv[list(o.data.polygons[hit[2]].vertices)].mean(axis=0) < 0:
                    nrm = -nrm
                sd_ = (x[i] - np.array(hit[0])) @ nrm
                if sd_ < gap:
                    out[i] = x[i] + nrm * (gap - sd_)
                    touch[i] = True
            return out, touch
        moved0 = fb.pos.copy()
        steps = fb.settle(body, h=1.0 / 120.0, most=DRAPE['steps'], tol=2e-4, damping=0.4)
        d = np.linalg.norm(fb.pos - moved0, axis=1)
        log('boot lacing %s: the bow let fall by the cloth solver in %d steps (%.1f s): the tails and loops moved %.2f cm on average, %.2f at most'
            % (o.name, steps, _t.time() - t0, d.mean() * 2.619, d.max() * 2.619))
        at0 = 0
        out = []
        for c, p0, p1 in chains:
            out.append(fb.pos[at0:at0 + len(c)].copy())
            at0 += len(c)
        chains = [(q, c[1], c[2]) for q, c in zip(out, chains)]
    for k_, (c, p0, p1) in enumerate(chains):
        if k_ < 2:
            strand(c, rl)
        else:
            run(c, rl * np.minimum(1.0, 1.0 - 0.25 * np.clip((np.linspace(0, 1, len(c)) - 0.9) / 0.1, 0, 1)))
    lace = _on_boot(o, np.array(LV), LF, o.name.replace('Boot', 'Boot lace'), cord, rig)
    log('boot lacing %s: %d eyelets a side, %d crossings of the lace' % (o.name, n, 2 * (n - 1)))
    return eyelets, lace


# the tongue's top, where it stands free above the panels, turned forward over a rolled fold (folds.py), as a
# worn tongue's top falls forward off the shin: its angle, the roll's radius, and the line's height over the
# panels' top edge (shares of the tongue's height above it)
DRAPE = dict(bend=2e-4, friction=0.5, steps=300, g=0.4, loops=False)         # the bow's settling (see lacing)
TONGUE_BEND = dict(angle=28.0, radius=0.45, at=0.15)


def bend_tongue(o, pieces, B=TONGUE_BEND, log=print):
    """Bend the sewn boot o's tongue forward above the panels' top edge, with an authored fold (folds.py) on the
    tongue alone: below the fold line it is held as sewn, above it turns over a rolled crease to B's angle. The fold
    is cached (folds.bake_cached) and a contour sheet of it written. Returns the report."""
    import folds
    by = {p['name']: p for p in pieces}
    if 'tongue' not in by:
        return None
    t = by['tongue']
    sl = slice(t['start'], t['start'] + len(t['V']))
    X = _co(o)
    P = X[sl].copy()
    uv, _ = pattern.flatten(t)
    T = []
    for f in t['F']:
        T += [(f[0], f[i], f[i + 1]) for i in range(1, len(f) - 1)]
    T = np.array(T, np.int64)
    # (wound alike on the pattern, its right side up)
    a, b, c = uv[T[:, 0]], uv[T[:, 1]], uv[T[:, 2]]
    neg = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) < 0
    T[neg] = T[neg][:, [0, 2, 1]]
    panels = np.concatenate([X[p['start']:p['start'] + len(p['V'])] for p in pieces if p['name'].endswith('shaft')])
    z_edge = float(np.percentile(panels[:, 2], 99.5))
    free = P[:, 2] > z_edge
    if free.sum() < 12:
        log('boot tongue %s: nothing stands above the panels to bend' % o.name)
        return None
    # the line on the pattern: across the tongue at the height (on the boot) a share B['at'] above the panels' edge
    z_line = z_edge + B['at'] * (P[:, 2].max() - z_edge)
    near = np.abs(P[:, 2] - z_line) < 0.12
    v_line = float(np.median(uv[near, 1])) if near.any() else float(np.median(uv[free, 1]))
    up = 1.0 if np.corrcoef(uv[:, 1], P[:, 2])[0, 1] > 0 else -1.0
    x0, x1 = uv[:, 0].min() - 1.0, uv[:, 0].max() + 1.0
    line = np.array([[x0, v_line], [x1, v_line]]) if up > 0 else np.array([[x1, v_line], [x0, v_line]])
    hold = P[:, 2] < z_line - 2.0 * B['radius']

    def sheet():
        # (its ties at the lengths it was sewn at: the fold turns it, the folds' hinges hold the turn)
        f = fabric.Fabric(P, dict(stretch=0.0, shear=1e-7, bend=2e-7, friction=0.0, iterations=12, gravity=(0.0, 0.0, 0.0)))
        f.pin(np.nonzero(hold)[0])
        f.mesh([tuple(x) for x in T])
        return f
    fab = sheet()
    # (forward: away from the leg; which way that is on the pattern is read off the tongue's normal on the boot)
    nrm = _normals(o)[sl][free].mean(0)
    # (which way the pattern's right side faces on the boot: a triangle wound up on the pattern has its normal on
    # the boot to that side; forward is out, along the leather's outer normal)
    tn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    side_ = 1.0 if float((tn.sum(0)) @ nrm) > 0 else -1.0
    log('boot tongue %s: the pattern faces %s on the boot; folding to side %+d' % (o.name, 'out' if side_ > 0 else 'in', side_))
    crease = dict(line=line, angle=np.radians(B['angle']), radius=B['radius'], side=side_, name='tongue top')
    side = o.name.rsplit('.', 1)[-1]
    pos, rep, how = folds.bake_cached('boot-tongue-%s' % side, fab, uv, T, [crease],
                                      extra=([B['angle'], B['radius'], B['at'], side_], nrm),
                                      log=log, tol_deg=2.0, most=4000)
    out = float(((pos - P)[free].mean(0)) @ nrm)
    log('boot tongue %s: its free top moved %.3f along the outer normal (%s)' % (o.name, out, 'forward' if out > 0 else 'BACK, toward the leg'))
    X[sl] = pos
    o.data.vertices.foreach_set('co', X.ravel())
    o.data.update()
    cr = (rep.get('creases') or [{}])[0]
    log('boot tongue %s: its top turned forward over a %.2f roll: %.1f of %.1f deg (%s, %s); contour sheet %s'
        % (o.name, B['radius'], cr.get('achieved_deg', float('nan')), cr.get('target_deg', B['angle']), how,
           'reached' if rep.get('ok') else 'stopped short: %s' % rep.get('why'), rep.get('sheet')))
    return rep


CREASE = dict(bend=16.0, steps=360, hold_sole=0.45, hold_collar=0.5, gain=0.4)  # (the ankle's bend in degrees, and the holds)


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
