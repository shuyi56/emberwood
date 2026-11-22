"""Differential-geometric gauges for a boot's draft: what a sheet of leather is asked to do, measured on the mesh.

A flat piece of leather can bend without strain but cannot be made to follow a doubly curved surface, or a bend
across one, without stretching, shearing or folding. These gauges say where and by how much, so a distortion is
traced to its cause instead of smoothed where it shows:

  gauss     the Gaussian curvature K at each point (the angle defect over its area). A flat sheet has K = 0 and
            keeps it however it is bent; the integral of |K| over a piece is the turning it must be stretched to
            take (0 for a cylinder or a cone, 2 pi for a hemisphere). Where it is large the piece wants a seam, a
            dart or a lasted (stretched) cut.
  strain    the two principal stretches of each triangle from one state to another (the square roots of the
            eigenvalues of the metric pulled back: the first fundamental form compared). Their product is the area's
            change, their ratio the shear. From the cut to the sewn boot it is what the leather was asked to give;
            from the boot at rest to the boot on its heel it is what the rig's bend asks, and a stretch under 1 across
            a bend is leather with nowhere to go: it folds.
  fold      the dihedral angle at each edge, at each point the largest of its edges': a crease is a line of them.
  mean      the mean curvature's size (the cotangent Laplacian's): a ridge or a bump is a band of it.
  pose_asymmetry  how unevenly the rig bends the two sides: each point's move from rest to posed against its mirror
            partner's. A shell bends both alike; a leg's skin weights do not.
  symmetry  the boot's inner side against its outer, mirrored in the plane through each station's middle: the top's
            height at matched offsets, the ridge's offset from the middle, and the half widths about the foot's axis.

Run on a built figure:
    python3 blender/zipboot/gauge.py --out DIR [--figure zipboot] [--size PX]
writes DIR/gauge.json (the numbers) and DIR/gauge-<name>.png (each gauge as a colour map on the boot, three views).
The numpy half (everything but sheet and main) needs no Blender and can be read into a test."""
import numpy as np


def triangles(faces):
    """Fan triangles of a face list (triangles or quads)."""
    T = []
    for f in faces:
        for i in range(1, len(f) - 1):
            T.append((f[0], f[i], f[i + 1]))
    return np.array(T, dtype=np.int64)


def _areas(X, T):
    return 0.5 * np.linalg.norm(np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]), axis=1)


def boundary(n, T):
    """True at the points on an open edge."""
    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    b = np.zeros(n, bool)
    b[u[c == 1].ravel()] = True
    return b


def gauss(X, T):
    """(K, area): the Gaussian curvature at each point (angle defect over a third of its triangles' area; 0 on an
    open edge, where the defect is the edge's own turning) and that area."""
    n = len(X)
    ang = np.zeros(n)
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        u, v = X[T[:, b]] - X[T[:, a]], X[T[:, c]] - X[T[:, a]]
        cs = (u * v).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-12)
        np.add.at(ang, T[:, a], np.arccos(np.clip(cs, -1.0, 1.0)))
    A = np.zeros(n)
    np.add.at(A, T.ravel(), np.repeat(_areas(X, T) / 3.0, 3))
    K = (2.0 * np.pi - ang) / np.maximum(A, 1e-12)
    K[boundary(n, T)] = 0.0
    return K, A


def strain(X0, X1, T):
    """(s1, s2): each triangle's larger and smaller principal stretch from X0 to X1. s1 * s2 is the area's change,
    s1 / s2 the shear; both 1 where the sheet only bent."""
    def frame(X):
        e1, e2 = X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]
        a = np.linalg.norm(e1, axis=1)
        ux = e1 / np.maximum(a, 1e-12)[:, None]
        b = (e2 * ux).sum(1)
        c = np.linalg.norm(e2 - b[:, None] * ux, axis=1)
        return a, b, c                                     # (the triangle laid flat: (0,0), (a,0), (b,c))
    a0, b0, c0 = frame(X0)
    a1, b1, c1 = frame(X1)
    ok = (a0 > 1e-9) & (c0 > 1e-9)
    a0, c0 = np.where(ok, a0, 1.0), np.where(ok, c0, 1.0)
    # F = [[a1, b1], [0, c1]] @ inv([[a0, b0], [0, c0]])
    f11, f12, f22 = a1 / a0, (b1 - a1 * b0 / a0) / c0, c1 / c0
    c11, c12, c22 = f11 * f11, f11 * f12, f12 * f12 + f22 * f22        # (C = F^T F)
    tr, det = c11 + c22, c11 * c22 - c12 * c12
    disc = np.sqrt(np.maximum(0.25 * tr * tr - det, 0.0))
    s1, s2 = np.sqrt(np.maximum(0.5 * tr + disc, 0.0)), np.sqrt(np.maximum(0.5 * tr - disc, 0.0))
    return np.where(ok, s1, 1.0), np.where(ok, s2, 1.0)


def to_points(v, T, n, X=None, how='mean'):
    """A triangle quantity carried to the points (the area-weighted mean, or the extreme)."""
    if how == 'mean':
        w = _areas(X, T) if X is not None else np.ones(len(T))
        num, den = np.zeros(n), np.zeros(n)
        np.add.at(num, T.ravel(), np.repeat(v * w, 3))
        np.add.at(den, T.ravel(), np.repeat(w, 3))
        return num / np.maximum(den, 1e-12)
    out = np.full(n, -np.inf if how == 'max' else np.inf)
    (np.maximum if how == 'max' else np.minimum).at(out, T.ravel(), np.repeat(v, 3))
    return np.where(np.isfinite(out), out, 0.0)


def fold(X, T):
    """The largest dihedral angle (degrees) among each point's edges: 0 on a flat sheet, a crease where it runs high."""
    nr = np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]])
    nr /= np.maximum(np.linalg.norm(nr, axis=1, keepdims=True), 1e-12)
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    key = np.sort(e, axis=1)
    tri = np.tile(np.arange(len(T)), 3)
    o = np.lexsort((key[:, 1], key[:, 0]))
    key, tri = key[o], tri[o]
    same = (key[1:] == key[:-1]).all(1)
    i, j = tri[:-1][same], tri[1:][same]
    ang = np.degrees(np.arccos(np.clip((nr[i] * nr[j]).sum(1), -1.0, 1.0)))
    out = np.zeros(len(X))
    np.maximum.at(out, key[:-1][same][:, 0], ang)
    np.maximum.at(out, key[:-1][same][:, 1], ang)
    return out


def mean(X, T):
    """The mean curvature's size at each point (half the cotangent Laplacian's length over the point's area)."""
    n = len(X)
    L = np.zeros((n, 3))
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        u, v = X[T[:, a]] - X[T[:, c]], X[T[:, b]] - X[T[:, c]]
        cot = (u * v).sum(1) / np.maximum(np.linalg.norm(np.cross(u, v), axis=1), 1e-12)
        d = (X[T[:, a]] - X[T[:, b]]) * (0.5 * cot)[:, None]
        np.add.at(L, T[:, a], d)
        np.add.at(L, T[:, b], -d)
    A = np.zeros(n)
    np.add.at(A, T.ravel(), np.repeat(_areas(X, T) / 3.0, 3))
    H = 0.5 * np.linalg.norm(L, axis=1) / np.maximum(A, 1e-12)
    H[boundary(n, T)] = 0.0
    return H


def symmetry(X, axis_x, y0, y1, floor, stations=24, offsets=(0.25, 0.5, 0.75)):
    """The boot's two sides compared station by station from y0 (the tip) to y1. For each station: its middle (of
    its extent in x), the half widths about the foot's axis (axis_x), the ridge's offset from the middle (where the
    top is highest), and the top's height at +-offsets of the half width from the middle (inner minus outer). Returns
    rows of dicts; lengths in the mesh's units."""
    rows = []
    ys = np.linspace(y0, y1, stations + 1)
    for a, b in zip(ys[:-1], ys[1:]):
        m = (X[:, 1] >= a) & (X[:, 1] < b) & (X[:, 2] > floor)
        if m.sum() < 12:
            continue
        P = X[m]
        lo, hi = P[:, 0].min(), P[:, 0].max()
        c, hw = 0.5 * (lo + hi), 0.5 * (hi - lo)
        top = P[np.argmax(P[:, 2])]
        row = dict(y=float(0.5 * (a + b) - y0), width=float(2 * hw), middle_off_axis=float(c - axis_x),
                   ridge_off_middle=float(top[0] - c), top=float(top[2] - floor), dz=[])
        for f in offsets:
            hs = []
            for sg in (-1, 1):
                near = np.abs(P[:, 0] - (c + sg * f * hw)) < 0.12 * hw + 0.03
                hs.append(P[near, 2].max() - floor if near.any() else np.nan)
            row['dz'].append(float(hs[1] - hs[0]))
        rows.append(row)
    return rows


def rigid(A, B):
    """The rotation and shift that carry points A onto B best (Kabsch)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    Rm = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return Rm, cb - Rm @ ca


def pose_asymmetry(R, P, floor, stations=40, fit=None):
    """How unevenly the rig bends the boot's two sides (fit: the points of its rigid part, the foot bone's own). Each point at rest is paired with the point mirrored across
    its station's middle (same distance along, same height, the other side); a shell bends both alike, so the two
    points' moves from rest to posed should mirror too. Returns each point's difference from its partner's move
    (mirrored), in the mesh's units; points with no partner within reach (a piece only one side has) get 0."""
    y = R[:, 1]
    ys = np.linspace(y.min(), y.max(), stations + 1)
    mid = np.zeros(stations)
    for k in range(stations):
        m = (y >= ys[k]) & (y <= ys[k + 1]) & (R[:, 2] > floor)
        mid[k] = 0.5 * (R[m, 0].min() + R[m, 0].max()) if m.any() else np.nan
    ok = ~np.isnan(mid)
    c = np.interp(y, 0.5 * (ys[:-1] + ys[1:])[ok], mid[ok])
    M = R.copy()
    M[:, 0] = 2.0 * c - R[:, 0]
    Rm, t_ = rigid(R[fit] if fit is not None else R, P[fit] if fit is not None else P)   # (the figure's own stance taken out:
    D = (P - t_) @ Rm - R                                # the boot read in the frame of its rigid part, fit)
    out = np.zeros(len(R))
    for a in range(0, len(R), 500):
        d2 = ((M[a:a + 500, None, :] - R[None, :, :]) ** 2).sum(2)
        j = d2.argmin(1)
        near = np.sqrt(d2[np.arange(len(j)), j]) < 0.25
        Dm = D[j] * np.array([-1.0, 1.0, 1.0])
        out[a:a + 500] = np.where(near, np.linalg.norm(D[a:a + 500] - Dm, axis=1), 0.0)
    return out


def read(R, P, faces, pieces=None, cut=None, fit=None):
    """Every gauge on a boot: R its points at rest, P posed, faces its faces, pieces [(name, start, count)], cut the
    points as cut (the last, or the flat pattern) where they are known. Returns (per-point maps, summary numbers)."""
    T = triangles(faces)
    n = len(R)
    maps, out = {}, {}
    K, A = gauss(R, T)
    maps['gauss'] = K
    s1, s2 = strain(R, P, T)
    maps['pose_min_stretch'] = to_points(s2, T, n, how='min')
    maps['pose_max_stretch'] = to_points(s1, T, n, how='max')
    maps['pose_area'] = to_points(s1 * s2, T, n, R)
    maps['fold_rest'], maps['fold_posed'] = fold(R, T), fold(P, T)
    maps['mean_rest'], maps['mean_posed'] = mean(R, T), mean(P, T)
    maps['pose_asymmetry'] = pose_asymmetry(R, P, R[:, 2].min() + 0.2, fit=fit)
    if cut is not None:
        c1, c2 = strain(cut, R, T)
        maps['cut_min_stretch'], maps['cut_max_stretch'] = to_points(c2, T, n, how='min'), to_points(c1, T, n, how='max')
    ar = _areas(R, T)
    for name, start, count in (pieces or [('boot', 0, n)]):
        if start < 0:
            continue
        sl = slice(start, start + count)
        inT = (T >= start).all(1) & (T < start + count).all(1)
        d = dict(points=int(count), area=float(ar[inT].sum()),
                 total_curvature=float((np.abs(K[sl]) * A[sl]).sum()),                 # (radians: 0 for a sheet that only bends)
                 pose_stretch_p01=float(np.percentile(s2[inT], 1)), pose_stretch_p99=float(np.percentile(s1[inT], 99)),
                 pose_compressed_area=float(ar[inT][s2[inT] < 0.95].sum() / max(ar[inT].sum(), 1e-9)),
                 fold_rest_p99=float(np.percentile(maps['fold_rest'][sl], 99)),
                 fold_posed_p99=float(np.percentile(maps['fold_posed'][sl], 99)),
                 creased_points_posed=int((maps['fold_posed'][sl] > 25.0).sum()),
                 creased_points_rest=int((maps['fold_rest'][sl] > 25.0).sum()),
                 pose_asymmetry_mean=float(maps['pose_asymmetry'][sl].mean()), pose_asymmetry_p99=float(np.percentile(maps['pose_asymmetry'][sl], 99)))
        if cut is not None:
            d['cut_stretch_p01'], d['cut_stretch_p99'] = float(np.percentile(c2[inT], 1)), float(np.percentile(c1[inT], 99))
        out[name] = d
    return maps, out


RAMPS = {   # name: (low, middle, high) values, and whether the ramp runs two ways about the middle
    'gauss': (-1.0, 0.0, 1.0), 'pose_min_stretch': (0.8, 1.0, 1.2), 'pose_max_stretch': (0.8, 1.0, 1.2),
    'pose_area': (0.8, 1.0, 1.2), 'cut_min_stretch': (0.9, 1.0, 1.1), 'cut_max_stretch': (0.9, 1.0, 1.1),
    'pose_asymmetry': (0.0, 0.15, 0.3), 'fold_rest': (0.0, 15.0, 30.0), 'fold_posed': (0.0, 15.0, 30.0), 'mean_rest': (0.0, 1.5, 3.0), 'mean_posed': (0.0, 1.5, 3.0),
}


def colours(v, name):
    """Blue below the middle, pale at it, red above (a colour per point)."""
    lo, mid, hi = RAMPS[name]
    t = np.where(v < mid, -np.clip((mid - v) / max(mid - lo, 1e-9), 0, 1), np.clip((v - mid) / max(hi - mid, 1e-9), 0, 1))
    c = np.ones((len(v), 4))
    pale, blue, red = np.array([0.82, 0.80, 0.76]), np.array([0.10, 0.25, 0.85]), np.array([0.85, 0.12, 0.08])
    c[:, :3] = np.where(t[:, None] < 0, pale + (blue - pale) * (-t)[:, None], pale + (red - pale) * t[:, None])
    return c


def main():
    import importlib
    import json
    import math
    import os
    import sys
    argv = sys.argv[1:]
    opt = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d  # noqa: E731
    here = os.path.dirname(os.path.abspath(__file__))
    fig = opt('--figure', 'zipboot')
    for d in (fig, 'heelboot', 'base', 'boot'):
        sys.path.insert(0, os.path.join(here, '..', d))
    import bpy
    from mathutils import Vector
    F = importlib.import_module(fig)
    import female
    import last
    out = opt('--out', '/tmp/gauge')
    size = int(opt('--size', 520))
    os.makedirs(out, exist_ok=True)
    female.MANNEQUIN = True
    F.build()
    female.pose(0)
    bpy.context.view_layer.update()
    res = {}
    sc = bpy.context.scene
    side = 'R'
    o = bpy.data.objects['Boot.' + side]
    for m in o.modifiers:
        if m.type == 'SOLIDIFY':
            m.show_viewport = m.show_render = False
    bpy.context.view_layer.update()
    R = np.array([v.co[:] for v in o.data.vertices])
    e = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
    P = np.array([(e.matrix_world @ v.co)[:] for v in e.data.vertices])
    faces = [p.vertices[:] for p in o.data.polygons]
    pieces = [(p['name'], p.get('start', -1), len(p['V'])) for p in last.PIECES[side]]
    gi = o.vertex_groups.get('Foot.' + side)
    wf = np.array([next((g.weight for g in v.groups if gi is not None and g.group == gi.index), 0.0) for v in o.data.vertices])
    maps, res['pieces'] = read(R, P, faces, pieces, fit=wf > 0.98 if (wf > 0.98).sum() > 50 else None)
    M = F.boot.hose.MEASURES
    sgn = -1.0 if side == 'R' else 1.0
    res['symmetry_rest'] = symmetry(R, sgn * M['leg_x'], R[:, 1].min(), R[:, 1].min() + 6.0, M['boot_welt'] + 0.1)
    for k, v in maps.items():
        res.setdefault('maps', {})[k] = dict(min=float(v.min()), p01=float(np.percentile(v, 1)), median=float(np.median(v)),
                                            p99=float(np.percentile(v, 99)), max=float(v.max()))
    json.dump(res, open(os.path.join(out, 'gauge.json'), 'w'), indent=1)
    for name, d in res['pieces'].items():
        print('gauge %s: the two sides bend apart by %.3f on average, %.3f at the 99th' % (name, d['pose_asymmetry_mean'], d['pose_asymmetry_p99']), flush=True)
        print('gauge %s: total curvature %.2f rad over %.1f units2; posed stretch %.2f..%.2f (1st, 99th), %.0f%% of its '
              'area squeezed over 5%%; folds 99th %.0f deg at rest, %.0f posed; creased points %d at rest, %d posed'
              % (name, d['total_curvature'], d['area'], d['pose_stretch_p01'], d['pose_stretch_p99'],
                 100 * d['pose_compressed_area'], d['fold_rest_p99'], d['fold_posed_p99'], d['creased_points_rest'],
                 d['creased_points_posed']), flush=True)
    # ---- the maps, as colour on the posed boot
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = 12
    sc.render.resolution_x = sc.render.resolution_y = size
    sc.render.image_settings.media_type = 'IMAGE'
    sc.render.image_settings.file_format = 'PNG'
    mat = bpy.data.materials.new('gauge')
    mat.use_nodes = True
    nt = mat.node_tree
    at = nt.nodes.new('ShaderNodeAttribute')
    at.attribute_name = 'gauge'
    bs = nt.nodes['Principled BSDF']
    nt.links.new(at.outputs['Color'], bs.inputs['Base Color'])
    bs.inputs['Roughness'].default_value = 0.9
    for ob in bpy.data.objects:
        if ob.type == 'MESH':
            ob.hide_render = ob.name != o.name
    o.data.materials.clear()
    o.data.materials.append(mat)
    col = o.data.color_attributes.get('gauge') or o.data.color_attributes.new('gauge', 'FLOAT_COLOR', 'POINT')
    cam = bpy.data.objects.new('gauge cam', bpy.data.cameras.new('gauge cam'))
    sc.collection.objects.link(cam)
    cam.data.type = 'ORTHO'
    sc.camera = cam
    c = 0.5 * (P.min(0) + P.max(0))
    cam.data.ortho_scale = 1.15 * max(np.ptp(P[:, 1]), np.ptp(P[:, 2]))
    views = {'outer': (Vector((sgn, 0, 0)), (math.pi / 2, 0, sgn * math.pi / 2)), 'front': (Vector((0, -1, 0)), (math.pi / 2, 0, 0)),
             'top': (Vector((0, -0.25, 1)).normalized(), (0.245, 0, 0))}
    from PIL import Image
    for name, v in maps.items():
        col.data.foreach_set('color', colours(v, name).ravel())
        o.data.update()
        tiles = []
        for vn, (dv, rot) in views.items():
            cam.location = Vector(c) + dv * 40.0
            cam.rotation_euler = rot
            sc.render.filepath = os.path.join(out, '_g.png')
            bpy.ops.render.render(write_still=True)
            tiles.append(Image.open(sc.render.filepath).convert('RGB'))
        sheet = Image.new('RGB', (size * len(tiles), size))
        for i, t in enumerate(tiles):
            sheet.paste(t, (i * size, 0))
        sheet.save(os.path.join(out, 'gauge-%s.png' % name))
        print('wrote', os.path.join(out, 'gauge-%s.png' % name), flush=True)


if __name__ == '__main__':
    main()
