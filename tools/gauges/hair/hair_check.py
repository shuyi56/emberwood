"""Pre-render hair check: catches creases, seams and ledges in the fused hair, and hair cutting into the clothes,
before a render is requested. It never renders: it ray-casts the built .blend.

    python3 blender/warden/hair_check.py BLEND [--out DIR] [--views 0-7] [--cols 0,2,4,6] [--hair Hair]

BLEND is a built figure (any figure on the base rig: e.g. `warden.py --build-only --out DIR` writes DIR/warden.blend).
It takes about two minutes and needs no GPU.

For each of the turnaround's eight facings it casts an orthographic grid of rays at the figure (as the camera sees
it) and, wherever the hair is the first thing hit, takes the smooth-shaded normal there. Then, between neighbouring
hair pixels:

  crease  the surface folds inward (concave) more sharply than a fillet of radius CREASE_R. A blended join between
          two hair masses is a broad, shallow valley; a seam is a narrow groove that renders as a dark line.
  ridge   the surface turns outward more sharply than a radius of RIDGE_R: a piece's corner showing as a hard edge.
  ledge   the hair steps over hair: two neighbours both on the hair but at depths more than LEDGE_D apart, with
          the nearer one facing sideways. A lock lying on the fall instead of growing into it shows like this.

Flagged pixels are grouped into blobs; any blob bigger than MIN_BLOB pixels fails the check. It also checks that the
clothes do not come out through the hair's outside in the walk frames (any poke-through fails). It prints each problem with
the facing, height and side, writes `hair-check.png` (the hair's shading per facing with problems in red/yellow),
and exits 1 on failure.
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

STEP = 0.07          # ray spacing (model units; the head is 8.6 tall)
CREASE_R = 0.3       # a concave bend tighter than this radius is a crease
RIDGE_R = 0.15       # a convex bend tighter than this is an edge (a lock's corner showing as a line)
LEDGE_D = 0.3        # hair stepping over hair by more than this is a ledge
MIN_BLOB = 10        # flagged pixels in one blob before it counts
FACING = ['front', 'front-3/4 (her left)', 'profile (her left)', 'back-3/4 (her left)', 'back',
          'back-3/4 (her right)', 'profile (her right)', 'front-3/4 (her right)']
HAIR = 'Hair'        # the hair object (--hair to change)


def is_clothes(o):
    """Anything of the figure that the hair can lie on: every mesh but the hair itself and what rides on the head."""
    head = o.parent is not None and o.parent.name.startswith('Head')
    return o.type == 'MESH' and o.name != HAIR and not head and not o.name.startswith(('Head', 'Hair')) \
        and 'fuzz' not in o.name


def arg(k, d=None):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d


def ints(spec):
    out = []
    for part in spec.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return out


def scene_tree(dg):
    """One BVH over every visible mesh, with each polygon's object name, and the hair's smooth vertex normals."""
    vs, ps, owner, hair = [], [], [], None
    for o in bpy.data.objects:
        if o.type != 'MESH' or o.hide_render or 'fuzz' in o.name or not o.visible_get():
            continue
        me = o.evaluated_get(dg).to_mesh()
        base = len(vs)
        mw = o.matrix_world
        vs += [mw @ v.co for v in me.vertices]
        ps += [[base + i for i in p.vertices] for p in me.polygons]
        owner += [o.name] * len(me.polygons)
        if o.name == HAIR:
            rot = mw.to_3x3()
            hair = (base, np.array([tuple((rot @ v.normal).normalized()) for v in me.vertices]),
                    [list(p.vertices) for p in me.polygons], len(ps) - len(me.polygons))
        o.evaluated_get(dg).to_mesh_clear()
    return BVHTree.FromPolygons(vs, ps), vs, ps, owner, hair


def shade_normal(hit, hit_poly, vs, ps, hair):
    """The smooth-shaded normal at a hit on the hair: the polygon's vertex normals weighted by nearness."""
    base, vn, polys, p0 = hair
    idx = polys[hit_poly - p0]
    w = np.array([1.0 / ((vs[base + i] - hit).length + 1e-4) for i in idx])
    n = (vn[idx] * w[:, None]).sum(0)
    return n / (np.linalg.norm(n) + 1e-9)


def blobs(mask):
    """4-connected blobs of a boolean image: a list of pixel lists."""
    seen = np.zeros_like(mask, bool)
    out = []
    H, W = mask.shape
    for j0, i0 in zip(*np.nonzero(mask)):
        if seen[j0, i0]:
            continue
        stack, px = [(j0, i0)], []
        seen[j0, i0] = True
        while stack:
            j, i = stack.pop()
            px.append((j, i))
            for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = j + dj, i + di
                if 0 <= a < H and 0 <= b < W and mask[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    stack.append((a, b))
        out.append(px)
    return out


def check_view(row, root, X0, X1, Z0, Z1):
    root.rotation_euler = (0, 0, row * math.pi / 4)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    tree, vs, ps, owner, hair = scene_tree(dg)
    W, H = int((X1 - X0) / STEP), int((Z1 - Z0) / STEP)
    depth = np.full((H, W), np.nan)
    nrm = np.zeros((H, W, 3))
    is_hair = np.zeros((H, W), bool)
    any_hit = np.zeros((H, W), bool)
    for j in range(H):
        z = Z1 - j * STEP
        for i in range(W):
            hit = tree.ray_cast(Vector((X0 + i * STEP, -80.0, z)), Vector((0, 1, 0)), 200.0)
            if hit[0] is None:
                continue
            any_hit[j, i] = True
            if owner[hit[2]] != HAIR:
                continue
            is_hair[j, i] = True
            depth[j, i] = hit[0].y
            n = shade_normal(hit[0], hit[2], vs, ps, hair)
            nrm[j, i] = -n if n[1] > 0 else n
    # (away from the hair's outline, against the sky or the figure, where the surface turns under and away from the
    # camera and any bend reads as an edge)
    inner = is_hair.copy()
    for _ in range(5):
        e = inner.copy()
        e[1:] &= inner[:-1]
        e[:-1] &= inner[1:]
        e[:, 1:] &= inner[:, :-1]
        e[:, :-1] &= inner[:, 1:]
        inner = e
    inner &= np.abs(nrm[..., 1]) > 0.35
    crease = np.zeros((H, W), bool)
    ledge = np.zeros((H, W), bool)
    ridge = np.zeros((H, W), bool)
    for dj, di in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = (slice(0, H - dj), slice(max(0, -di), W - max(0, di)))
        b = (slice(dj, H), slice(max(0, di), W + min(0, di)))
        both = is_hair[a] & is_hair[b]
        dy = depth[b] - depth[a]
        dist = STEP * math.hypot(dj, di)
        # the step between them in model space (screen x, depth y, screen z)
        d3 = np.stack([np.full(dy.shape, di * STEP), dy, np.full(dy.shape, -dj * STEP)], -1)
        dn = nrm[b] - nrm[a]
        ang = np.arccos(np.clip((nrm[a] * nrm[b]).sum(-1), -1, 1))
        span = np.sqrt(dist ** 2 + np.nan_to_num(dy) ** 2)
        concave = (dn * d3).sum(-1) < 0
        cont = both & (np.abs(np.nan_to_num(dy, nan=9)) < 3 * dist + 0.05)
        c = cont & inner[a] & inner[b] & concave & (ang / span > 1.0 / CREASE_R)
        crease[a] |= c
        crease[b] |= c
        r = cont & inner[a] & inner[b] & ~concave & (ang / span > 1.0 / RIDGE_R)
        ridge[a] |= r
        ridge[b] |= r
        # a step: the nearer pixel's surface faces across the step (it is an overhanging edge, not a slope)
        # (a steep but continuous slope near the outline also jumps in depth; there the normals predict the jump)
        nm = nrm[a] + nrm[b]
        ny = np.maximum(np.abs(nm[..., 1]), 0.1)
        pred = np.abs(nm[..., 0] * di * STEP - nm[..., 2] * dj * STEP) / ny
        ady = np.abs(np.nan_to_num(dy))
        far_n = np.where((dy > 0)[..., None], nrm[b], nrm[a])
        st = both & (ady > LEDGE_D) & (ady > 2.0 * pred + 0.15) & (np.abs(far_n[..., 1]) > 0.4)
        ledge[a] |= st & (dy > 0)
        ledge[b] |= st & (dy < 0)
    problems = []
    for kind, mask in (('crease', crease), ('ledge', ledge), ('ridge', ridge)):
        for px in blobs(mask):
            if len(px) < MIN_BLOB:
                continue
            js, is_ = zip(*px)
            z = Z1 - np.mean(js) * STEP
            x = X0 + np.mean(is_) * STEP
            problems.append((kind, row, len(px), z, x, min(js), max(js), min(is_), max(is_)))
    # the picture: the hair lit, the rest of the figure grey, problems marked
    L = np.array([-0.55, -0.75, 0.62]) / np.linalg.norm([-0.55, -0.75, 0.62])
    img = np.zeros((H, W, 3)) + (0.16, 0.17, 0.2)
    img[any_hit] = (0.42, 0.42, 0.44)
    lam = 0.35 + 0.65 * np.clip((nrm * L).sum(-1), 0, 1)
    img[is_hair] = (np.array([0.62, 0.45, 0.33])[None] * lam[is_hair][:, None])
    for kind, mask, colr in (('crease', crease, (1.0, 0.1, 0.1)), ('ledge', ledge, (1.0, 0.85, 0.1)), ('ridge', ridge, (0.2, 0.6, 1.0))):
        for px in blobs(mask):
            if len(px) >= MIN_BLOB:
                for j, i in px:
                    img[j, i] = colr
    return problems, img


def poke_through(cols):
    """Clothes coming out through the hair: in each walk frame, the clothes' surfaces crossing the hair's outside.
    The hair's outside is every face of it whose normal looks out into the open (a ray along it meets nothing of
    the figure within 1.5); the rest is its underside, which may touch the robe out of sight."""
    out = []
    sc = bpy.context.scene
    for col in cols:
        sc.frame_set(col + 1)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        cloth, rest_vs, rest_ps = {}, [], []
        hv = hp = None
        for o in bpy.data.objects:
            if o.type != 'MESH' or o.hide_render or not o.visible_get():
                continue
            me = o.evaluated_get(dg).to_mesh()
            vs = [o.matrix_world @ v.co for v in me.vertices]
            ps = [list(p.vertices) for p in me.polygons]
            o.evaluated_get(dg).to_mesh_clear()
            if o.name == HAIR:
                hv, hp = vs, ps
                continue
            b = len(rest_vs)
            rest_vs += vs
            rest_ps += [[b + i for i in p] for p in ps]
            if is_clothes(o):
                cloth[o.name] = BVHTree.FromPolygons(vs, ps)
        figure = BVHTree.FromPolygons(rest_vs, rest_ps)
        outside = []
        for p in hp:
            c = sum((hv[i] for i in p), Vector()) / len(p)
            n = (hv[p[1]] - hv[p[0]]).cross(hv[p[2]] - hv[p[0]])
            if n.length < 1e-12:
                continue
            n.normalize()
            if figure.ray_cast(c + n * 0.02, n, 1.5)[0] is None:
                outside.append(p)
        ht = BVHTree.FromPolygons(hv, outside)
        for name, t in cloth.items():
            ov = ht.overlap(t)
            if len(ov) >= 4:
                zs = [hv[outside[i][0]].z for i, _ in ov]
                out.append((col, name, len(ov), min(zs), max(zs)))
    sc.frame_set(1)
    return out


def main():
    global HAIR
    blend = sys.argv[1]
    HAIR = arg('--hair', HAIR)
    out_dir = arg('--out', os.path.dirname(os.path.abspath(blend)))
    rows = ints(arg('--views', '0-7'))
    cols = ints(arg('--cols', '0,2,4,6'))
    bpy.ops.wm.open_mainfile(filepath=blend)
    bpy.context.scene.frame_set(1)
    root = [o for o in bpy.data.objects if o.parent is None and o.type == 'EMPTY' and o.name != 'Rig'][0]
    hair = bpy.data.objects[HAIR]
    zs = [(hair.matrix_world @ v.co).z for v in hair.data.vertices]
    Z0, Z1 = min(zs) - 1.0, max(zs) + 0.5
    X0, X1 = -9.0, 9.0
    problems, tiles = [], []
    for row in rows:
        p, img = check_view(row, root, X0, X1, Z0, Z1)
        problems += p
        tiles.append(img)
        print(f'  {FACING[row]}: {len(p)} problem(s)', flush=True)
    root.rotation_euler = (0, 0, 0)
    clash = poke_through(cols)
    try:
        from PIL import Image
        im = np.concatenate(tiles, axis=1)
        Image.fromarray((im * 255).clip(0, 255).astype(np.uint8)).resize(
            (im.shape[1] * 2, im.shape[0] * 2), Image.NEAREST).save(os.path.join(out_dir, 'hair-check.png'))
    except ImportError:
        pass
    print('\nHair check')
    for kind, row, n, z, x, *_ in sorted(problems, key=lambda p: -p[2]):
        side = 'centre' if abs(x) < 0.8 else ('screen left' if x < 0 else 'screen right')
        print(f'  FAIL {kind:6s} {FACING[row]:22s} {n:4d} px at height {z:5.1f}, {side} (x {x:+.1f})')
    for col, name, n, z0, z1 in clash:
        print(f'  FAIL clash  walk column {col}: {name} comes out through the hair ({n} faces, height {z0:.1f}-{z1:.1f})')
    ok = not problems and not clash
    print('  PASS: no creases, ledges or clashes' if ok else f'  {len(problems)} surface problem(s), {len(clash)} clash(es)')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
