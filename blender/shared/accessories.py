"""Jewellery and small accessories for any figure built on the base female body (blender/base/female.py):
necklaces (a chain or a bead strand, with or without a pendant), a choker, earrings and a brooch.

Each piece finds its own place on whatever the figure is already wearing, instead of being placed by hand:

- A necklace is a closed chain of a given length, dropped over the head and let fall (a small Verlet simulation
  against the body and the clothes built so far). It comes to rest the way a real one does: on the slope of the
  trapezius at the back and the sides, where the neck runs out into the shoulders, and falling lower in front, where
  the chest is too steep to hold it, into a U (a V with a pendant's weight at the front). It lies on top of every
  layer under it (skin, shirt, robe, collar) and bridges across hollows the way a taut chain does.
- A pendant hangs from the chain's lowest point, straight down, and turns forward about its bail only as far as it
  must to rest on the chest instead of sinking into it.
- A choker is a band fitted to the neck at a given height, with an optional charm at the front.
- Earrings hang from the earlobes (found from the base head's ear, blender/shared/ear.py; pass lobe= for a head
  whose ear sits elsewhere).
- A brooch is pinned to the outermost surface at a point on the front, turned to lie flat on it.

How they follow the walk: every piece is parented to the figure's 'Body' empty, so finish_figure() hands it to the
rig with the trunk's weights (skin_trunk), exactly as the clothes under it are. The chain therefore moves with the
cloth it lies on, and cannot slide into it through the walk. Pieces sitting wholly on the chest (a pendant, a
brooch) get the Chest bone alone and stay rigid; earrings sit above the neck's blend and get the Neck bone alone,
which follows the head. (Nothing is parented under the 'Head' empty: tools/creator-head/merge-head.mjs drops all
of that for the Face Creator's head, and views.py --no-body hides it.)

Use, in a character's build(), after the clothes the piece lies on and before finish_figure() (and before a cape
that should lie over it: build it after, so the cape's own fitting sees the necklace):

    import accessories as acc
    acc.necklace('Pendant necklace', body, length=acc.LENGTHS['matinee'], pendant='drop')
    acc.necklace('Pearls', body, length=acc.LENGTHS['collar'], style='beads', mid=acc.M_PEARL)
    acc.choker('Choker', body, z=52.6, charm=True)
    acc.earrings('Earring', body, style='drop')
    acc.brooch('Brooch', body, x=-3.0, z=47.0)
    ...
    finish_figure(acc.follow)       # (or lambda: (cape_on_arms([...]), acc.follow()))
    acc.clearance_report()          # optional: checks every accessory against the clothes in every sheet column

Lengths are in model units (one design pixel). LENGTHS names the usual ones, measured on this body by where
they come to rest (see ACCESSORIES.md): a collar length round the base of the neck, a princess on the collarbones,
a matinee on the upper chest, an opera down to the bust.
"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

import female
from female import HEAD_H, HEAD_Z, NECK_Z, ellipsoid, mat, mat_def, mesh_obj, ramp

# ---------------------------------------------------------------- materials (ids 80-89, kept for accessories)
M_ACC_GOLD = mat_def(80, 'jewel gold', ramp(0xc29a4a, 0.1), 0.0, closeup='gold')
M_SILVER = mat_def(81, 'jewel silver', ramp(0xb4b6ba, 0.12), 0.0, closeup='jewel')
M_PEARL = mat_def(82, 'pearl', [[178, 170, 162], [206, 199, 190], [226, 220, 210], [238, 234, 226], [248, 246, 240]], 0.0,
                  closeup='jewel')
M_GARNET = mat_def(83, 'garnet', [[70, 12, 22], [104, 20, 34], [138, 30, 46], [176, 52, 66], [214, 98, 106]], 0.0,
                   closeup='jewel')
M_RIBBON = mat_def(84, 'velvet ribbon', ramp(0x2a1a2e, 0.08), 0.0, closeup='jewel')
M_CAMEO = mat_def(85, 'cameo', [[150, 118, 104], [182, 150, 134], [206, 178, 162], [224, 202, 188], [238, 224, 214]], 0.0,
                  closeup='jewel')

# the close-up look of each (the sheet keeps the plain diffuse surface and the palette ramps: see sprite_mode):
# base colour (linear), metallic, roughness, coat, sheen
JEWEL_LOOK = {
    'jewel silver': ((0.62, 0.63, 0.66), 1.0, 0.28, 0.0, 0.0),
    'pearl': ((0.78, 0.74, 0.68), 0.0, 0.22, 1.0, 0.25),          # (a nacre: a soft body under a glassy coat)
    'garnet': ((0.24, 0.012, 0.03), 0.0, 0.04, 1.0, 0.0),
    'velvet ribbon': ((0.025, 0.012, 0.03), 0.0, 0.8, 0.0, 0.9),   # (velvet: dark, with a sheen at the grazing edges)
    'cameo': ((0.62, 0.45, 0.36), 0.0, 0.45, 0.3, 0.0),
}

ACCESSORIES = []   # every mesh this module built, for clearance_report

# where each length comes to rest on the base figure (the front of the chain; the back is held at the nape, h 51.6;
# measured on blender/accessory_demo): collar 51.6 (level round the base of the neck), princess 49.5 (on the
# collarbones), matinee 48.2 (the upper chest, held by the rise of the bust), opera 43.8 (slipped over the bust).
# The body's neck base is about 14 round, so a unit here is roughly 2.5 cm of real chain.
LENGTHS = {'collar': 15.5, 'princess': 19.0, 'matinee': 25.0, 'opera': 30.0}


def _material(mid):
    """The material for mid, with its close-up surface for the 3D views (a 'close bsdf' that sprite_mode swaps)."""
    m = mat(mid)
    look = JEWEL_LOOK.get(m.name)
    nt = m.node_tree
    if look is None or 'close bsdf' in nt.nodes:
        return m
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
    diff = next(n for n in nt.nodes if n.type == 'BSDF_DIFFUSE')
    diff.name = 'sprite bsdf'
    col, metal, rough, coat, sheen = look
    pb = nt.nodes.new('ShaderNodeBsdfPrincipled')
    pb.name = 'close bsdf'
    pb.inputs['Base Color'].default_value = (*col, 1.0)
    pb.inputs['Metallic'].default_value = metal
    pb.inputs['Roughness'].default_value = rough
    pb.inputs['Coat Weight'].default_value = coat
    pb.inputs['Coat Roughness'].default_value = 0.05
    if sheen:
        pb.inputs['Sheen Weight'].default_value = sheen
        pb.inputs['Sheen Tint'].default_value = (0.5, 0.45, 0.5, 1.0)
    nt.links.new(pb.outputs['BSDF'], out.inputs['Surface'])
    return m


RIGID = {}         # the rigid pieces (a pendant, a charm, an earring, a brooch): name -> the point they hang from


BONES = {}         # pieces carried by given bones outright: name -> {bone: weight} (an earring: the head's)


def _obj(name, bm, mid, body, pivot=None, bones=None, **kw):
    kw.setdefault('subdiv', 0)
    o = mesh_obj(name, bm, mid, body, **kw)
    _material(mid)
    ACCESSORIES.append(o.name)
    if pivot is not None:
        RIGID[o.name] = Vector(pivot)
    if bones:
        BONES[o.name] = dict(bones)
    return o


def _join(parts):
    """One bmesh of several."""
    bm = bmesh.new()
    for part in parts:
        tmp = bpy.data.meshes.new('part')
        part.to_mesh(tmp)
        part.free()
        bm.from_mesh(tmp)
        bpy.data.meshes.remove(tmp)
    return bm


def _xform(bm, M):
    bmesh.ops.transform(bm, matrix=M, verts=bm.verts)
    return bm


# ---------------------------------------------------------------- what the piece lies on
def surface_tree(skip=('Hair',), only=None, accessories=True):
    """A BVH of the figure's outer surfaces as they render (every modifier applied: thickness, smoothing), in world
    space, in the pose the scene is in. Skips the hair (it falls over a necklace, never under one), the head's
    paints and eyes (everything under the 'Head' empty but the head itself) and hidden meshes; with
    accessories=False, also the accessories built so far (otherwise a piece lies over the ones before it: pearls
    over a choker). only= names the meshes to use instead."""
    dg = bpy.context.evaluated_depsgraph_get()
    vs, ps = [], []
    for o in bpy.data.objects:
        if o.type != 'MESH' or o.hide_render:
            continue
        if only is not None:
            if not o.name.startswith(only):
                continue
        else:
            if o.name.startswith(skip) or (not accessories and o.name in ACCESSORIES):
                continue
            if o.parent is not None and o.parent.name == 'Head' and o.name != 'Head shape':
                continue
        me = o.evaluated_get(dg).to_mesh()
        M = o.matrix_world
        b = len(vs)
        vs += [M @ v.co for v in me.vertices]
        ps += [[b + i for i in p.vertices] for p in me.polygons]
        o.evaluated_get(dg).to_mesh_clear()
    return BVHTree.FromPolygons(vs, ps)


def _inside(tree, p, cy=0.3):
    """Whether p lies inside the closed surfaces (a ray from it, out away from the body's axis, crosses them an odd
    number of times)."""
    u = Vector((p.x, p.y - cy, 0.0))
    u = u.normalized() if u.length > 1e-6 else Vector((0, -1, 0))
    o, k = p.copy(), 0
    for _ in range(16):
        hit = tree.ray_cast(o, u, 40.0)[0]
        if hit is None:
            break
        k += 1
        o = hit + u * 1e-4
    return k % 2 == 1


def _signed(tree, p, reach=3.0):
    """How far p stands out from the nearest surface (negative inside it), the nearest point, and the way out from
    it (unit). The facing of the nearest face says which side p is on, except where it can mislead: by the open edge
    of a shell (a neckline's thickness) the nearest face can be its inner side, facing in, with p outside. So when
    the face says inside, a ray decides."""
    q, n, _i, d = tree.find_nearest(p, reach)
    if q is None:
        return reach, None, None
    out = (p - q).dot(n) >= 0 or not _inside(tree, p)
    if d < 1e-6:
        return 0.0, q, n
    way = (p - q) / d if out else (q - p) / d
    return (d if out else -d), q, way


# ---------------------------------------------------------------- the necklace's fall
def _drape(length, tree, n=120, clear=0.1, pendant=0.0, z0=None, cy=0.3, steps=1500, gravity=0.004, friction=0.12,
          nape=NECK_Z + 0.6, iters=14):
    """Let a closed chain of `length` fall over the head onto the figure, and return where it comes to rest: n
    points round it, the first at the front. A chain can't stretch but goes slack freely (only its stretch is held);
    its points are pushed out of any surface to `clear` (its own radius) and, touching, lose most of their slide (the
    friction that keeps a real one up on the slope of the shoulders). `pendant` is the weight hung at the front, in
    units of the chain's own: the front is drawn down into a V by it.

    `nape` is the height where the back of the chain catches, at the base of the neck. On a real body the chain is
    held there by the knob of the seventh vertebra and the ridge of the trapezius, higher than the notch between the
    collarbones in front (the neck leans forward, so its base is a ring tilted down at the front); that tilt is why a
    necklace hangs lower in front than behind. The base figure's torso is a stack of level sections without that
    ridge, and a chain let fall on it freely settles almost level, sliding off the back as far as the front. So the
    chain's back point is held at the surface there and the rest falls from it. None lets it fall freely."""
    r0 = length / (2 * math.pi)
    z0 = NECK_Z + 2.7 if z0 is None else z0
    P = np.array([[-r0 * math.sin(2 * math.pi * k / n), cy - r0 * math.cos(2 * math.pi * k / n), z0] for k in range(n)])
    prev = P.copy()
    w = np.ones(n)
    w[0] = 1.0 / (1.0 + pendant)          # (inverse mass: the pendant's point is heavier)
    back = n // 2
    if nape is not None:
        hit = tree.ray_cast(Vector((0, cy + 20.0, nape)), Vector((0, -1, 0)), 40.0)[0]
        if hit is None:
            raise ValueError(f'drape: no surface at the nape (z {nape})')
        P[back] = prev[back] = np.array(hit) + np.array([0, clear, 0])
        w[back] = 0.0                     # (held)
    rest = length / n
    nxt = np.roll(np.arange(n), -1)
    for step in range(steps):
        vel = (P - prev) * 0.93 * (w > 0)[:, None]
        prev = P.copy()
        P = P + vel
        P[:, 2] -= gravity * (w > 0)
        touch = np.zeros(n, dtype=bool)
        # (a little stiffness: a chain lying slack doesn't stay kinked, its links settle into a smooth run)
        lap = 0.5 * (P[np.roll(np.arange(n), 1)] + P[nxt]) - P
        P += 0.04 * lap * (w > 0)[:, None]
        for _ in range(iters):
            # inextensible, slack allowed
            d = P[nxt] - P
            L = np.linalg.norm(d, axis=1)
            over = np.maximum(0.0, L - rest) / np.maximum(L, 1e-9)
            wa, wb = w, w[nxt]
            k = (over / np.maximum(wa + wb, 1e-9))[:, None] * d
            np.add.at(P, np.arange(n), k * wa[:, None])
            np.add.at(P, nxt, -k * wb[:, None])
            # out of the surfaces, losing most of the slide where it touches
            for i in range(n):
                p = Vector(P[i])
                s, q, nn = _signed(tree, p, clear + 1.5)
                if q is not None and s < clear and w[i] > 0:
                    p2 = q + nn * clear
                    # (pushed out of one surface into another: it is wedged in a gap narrower than itself, as between
                    # the skin and a neckline's edge. A chain can't go in there: it is carried on out, over the
                    # outer layer, and slides on down over it)
                    for k_ in range(10):
                        if _signed(tree, p2, clear + 1.0)[0] >= 0.8 * clear:
                            break
                        p2 = p2 + nn * (0.5 * clear)
                    P[i] = p2
                    touch[i] = True
        # (friction: a point lying on the surface loses part of its slide, once a step)
        prev[touch] += (P[touch] - prev[touch]) * friction
        if step > 120 and np.abs(P - prev).max() < 2e-4:
            break
    # (left symmetric: the fall is, but rounding drifts it a hair)
    half = P.copy()
    for i in range(n):
        j = (n - i) % n
        half[i] = 0.5 * (P[i] + P[j] * np.array([-1, 1, 1]))
    return [Vector(p) for p in half]


def drape(length, tree, **kw):
    """Let a closed chain of `length` fall onto the figure: see _drape, which this runs again with the links'
    length corrected until the chain at rest is within 1.5% of `length` (where it lies on the surface, pushing it
    out stretches it a little each step)."""
    ask = length
    for _ in range(4):
        pts = _drape(ask, tree, **kw)
        got = sum((pts[i] - pts[i - 1]).length for i in range(len(pts)))
        if abs(got - length) < 0.015 * length:
            break
        ask *= length / got
    return pts


def _resample(pts, step, closed=True):
    """Points spaced `step` apart along a polyline (by arc length)."""
    loop = pts + ([pts[0]] if closed else [])
    seg = [(a - b).length for a, b in zip(loop[1:], loop)]
    total = sum(seg)
    m = max(3, int(round(total / step)))
    out, i, acc = [], 0, 0.0
    for k in range(m):
        t = k * total / m
        while acc + seg[i] < t and i < len(seg) - 1:
            acc += seg[i]
            i += 1
        f = (t - acc) / seg[i] if seg[i] else 0.0
        out.append(loop[i].lerp(loop[i + 1], f))
    return out


def ring_tube(pts, r, n=8, twist=0.0):
    """A closed tube of radius r through pts (a loop), its sections kept square to the curve. twist > 0 rolls a
    shallow two-strand ridge round it (a rope or curb chain reads as a twisted cord at this size)."""
    bm = bmesh.new()
    m = len(pts)
    rings = []
    up = Vector((0, 0, 1))
    acc = 0.0
    for i, p in enumerate(pts):
        t = (pts[(i + 1) % m] - pts[i - 1]).normalized()
        a = t.cross(up)
        a = a.normalized() if a.length > 1e-6 else Vector((1, 0, 0))
        b = a.cross(t).normalized()
        acc += (pts[(i + 1) % m] - p).length
        ring = []
        for k in range(n):
            th = 2 * math.pi * k / n
            rr = r * (1.0 + twist * math.cos(2 * th + acc / (2.2 * r)))
            ring.append(bm.verts.new(p + a * (rr * math.cos(th)) + b * (rr * math.sin(th))))
        rings.append(ring)
    for i in range(m):
        ra, rb = rings[i], rings[(i + 1) % m]
        for k in range(n):
            bm.faces.new((ra[k], ra[(k + 1) % n], rb[(k + 1) % n], rb[k]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def beads(pts, r, seg=12):
    bm = bmesh.new()
    for p in pts:
        s = bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=seg // 2 + 2, radius=r)
        bmesh.ops.translate(bm, vec=p, verts=s['verts'])
    return bm


def torus(R, r, seg=24, n=8):
    """A ring in the XZ plane (facing -Y), centred on the origin."""
    bm = bmesh.new()
    loops = []
    for i in range(seg):
        t = 2 * math.pi * i / seg
        c = Vector((R * math.sin(t), 0, R * math.cos(t)))
        out = Vector((math.sin(t), 0, math.cos(t)))
        loops.append([bm.verts.new(c + out * (r * math.cos(2 * math.pi * j / n)) + Vector((0, r * math.sin(2 * math.pi * j / n), 0)))
                      for j in range(n)])
    for i in range(seg):
        a, b = loops[i], loops[(i + 1) % seg]
        for j in range(n):
            bm.faces.new((a[j], a[(j + 1) % n], b[(j + 1) % n], b[j]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def teardrop(w, h, depth, seg=32, rings=6):
    """A drop facing -Y, its point up at the origin and its round body below: a domed front and a flatter back,
    `depth` thick in all."""
    bm = bmesh.new()
    outline = []
    for i in range(seg):
        t = 2 * math.pi * i / seg
        # (the drop: round below, drawn up into a soft point)
        x = 0.5 * w * math.sin(t) * (0.55 + 0.45 * (1 - math.cos(t)) / 2) ** 1.3
        z = -0.5 * h * (1 - math.cos(t))
        outline.append((x, z))
    cz = -0.62 * h
    front, back = [], []
    for j in range(rings):
        f = j / rings
        lift = math.sqrt(max(0.0, 1 - (1 - f) ** 2))
        fr, bk = [], []
        for x, z in outline:
            px, pz = x * (1 - f), cz + (z - cz) * (1 - f)
            fr.append(bm.verts.new((px, -0.65 * depth * lift, pz)))
            bk.append(bm.verts.new((px, 0.35 * depth * lift, pz)))
        front.append(fr)
        back.append(bk)
    cf = bm.verts.new((0, -0.65 * depth, cz))
    cb = bm.verts.new((0, 0.35 * depth, cz))
    for layers, c in ((front, cf), (back, cb)):
        for a, b in zip(layers, layers[1:]):
            for i in range(seg):
                bm.faces.new((a[i], a[(i + 1) % seg], b[(i + 1) % seg], b[i]))
        for i in range(seg):
            bm.faces.new((layers[-1][i], layers[-1][(i + 1) % seg], c))
    for i in range(seg):
        bm.faces.new((front[0][i], back[0][i], back[0][(i + 1) % seg], front[0][(i + 1) % seg]))
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def drop_pendant(size=1.0, gem=M_GARNET, metal=M_ACC_GOLD):
    """A teardrop gem in a gold bezel under a bail, facing -Y; the bail's centre (where the chain runs) at the
    origin. Returns {material: bmesh}."""
    s = size
    bail = _xform(torus(0.2 * s, 0.055 * s, 16, 6), Matrix.Rotation(math.pi / 2, 4, 'Z'))   # (its plane across the chain)
    bezel = _xform(teardrop(0.95 * s, 1.35 * s, 0.32 * s), Matrix.Translation((0, 0, -0.2 * s)))
    stone = _xform(teardrop(0.74 * s, 1.08 * s, 0.34 * s), Matrix.Translation((0, -0.07 * s, -0.33 * s)))
    return {metal: _join([bail, bezel]), gem: stone}


def cameo(size=1.0, metal=M_ACC_GOLD):
    """An oval cameo in a beaded gold frame, facing -Y, centred on the origin."""
    s = size
    rim = _xform(torus(1.0, 0.09 * s, 32, 6), Matrix.Diagonal((0.42 * s, 1, 0.55 * s, 1)))
    back = ellipsoid(0.44 * s, 0.07 * s, 0.57 * s, 24, 8)
    face = _xform(ellipsoid(0.36 * s, 0.1 * s, 0.48 * s, 24, 10), Matrix.Translation((0, -0.05 * s, 0)))
    bead = []
    for k in range(18):
        t = 2 * math.pi * k / 18
        b = ellipsoid(0.055 * s, 0.055 * s, 0.055 * s, 8, 6)
        bead.append(_xform(b, Matrix.Translation((0.52 * s * math.sin(t), -0.02, 0.66 * s * math.cos(t)))))
    return {metal: _join([rim, back] + bead), M_CAMEO: face}


def _place(parts, M):
    return {mid: _xform(bm, M) for mid, bm in parts.items()}


def _verts(parts):
    return [Vector(v.co) for bm in parts.values() for v in bm.verts]


def _rest_on(parts, pivot, tree, clear=0.04, axis='X', limit=1.3):
    """Turn a piece hanging from `pivot` about it (forward, about the figure's X) only as far as it must for every
    point of it to stand `clear` off the surface: a pendant that would sink into the chest swings out and rests on
    it; one that hangs free stays plumb. Returns the turned pieces."""
    best = None
    for k in range(60):
        th = limit * k / 59
        R = Matrix.Translation(pivot) @ Matrix.Rotation(-th, 4, axis) @ Matrix.Translation(-pivot)
        # (the bail turns in place about the chain and stands where the chain does: only the body below it is tested)
        lo = min(_signed(tree, R @ p)[0] for p in _verts(parts) if (p - pivot).length > 0.3)
        if best is None or lo > best[0]:
            best = (lo, th, R)
        if lo >= clear:
            break
    parts = _place(parts, best[2])
    if best[0] < clear:
        # (still touching at its best: stood off along the surface's normal at its deepest point, the bail riding
        # out on the chain)
        p = min(_verts(parts), key=lambda v: _signed(tree, v)[0])
        _s, _q, nn = _signed(tree, p)
        if nn is not None:
            parts = _place(parts, Matrix.Translation(nn * (clear - best[0])))
    print(f'rest_on: turned {math.degrees(best[1]):.0f} deg, clearance {best[0]:+.3f}', flush=True)
    return parts


def necklace(name, body, length=LENGTHS['princess'], style='chain', pendant=None, mid=M_ACC_GOLD, radius=None, tree=None,
             pendant_size=1.0):
    """A necklace let fall onto what the figure wears (see drape): style 'chain' (a fine twisted cord, radius 0.055) or
    'beads' (round beads, radius 0.2, mid=M_PEARL for pearls); pendant None, 'drop' (a garnet teardrop in gold) or
    'cameo'. Returns the chain's resting points."""
    r = radius or (0.2 if style == 'beads' else 0.055)
    tree = tree or surface_tree()
    weight = {None: 0.0, 'drop': 6.0, 'cameo': 8.0}[pendant]
    pts = drape(length, tree, clear=r + 0.02, pendant=weight)
    zs = [p.z for p in pts]
    print(f'necklace {name!r}: front at z {pts[0].z:.2f}, back at z {pts[len(pts) // 2].z:.2f}, '
          f'lowest {min(zs):.2f}, highest {max(zs):.2f}', flush=True)
    if style == 'beads':
        _obj(name, beads(_resample(pts, 2.08 * r), r), mid, body)
    else:
        _obj(name, ring_tube(_resample(pts, 0.06), r, n=10, twist=0.12), mid, body)
    if pendant:
        # (the bail stands on the chain: the chain runs through its back, so the bail's centre is a little in front of
        # the chain's lowest point)
        top = pts[0] + Vector((0, -0.2, 0))
        parts = drop_pendant(pendant_size) if pendant == 'drop' else \
            _place(cameo(pendant_size), Matrix.Translation((0, 0, -0.75 * pendant_size)))
        parts = _place(parts, Matrix.Translation(top))
        parts = _rest_on(parts, top, tree)
        for m_, bm in parts.items():
            _obj(f'{name} pendant {female.MATERIALS[m_]["name"]}', bm, m_, body, pivot=top)
    return pts


def _neck_ring(tree, z, n=64, cy=0.3, out=0.0):
    """The neck's outline at height z (rays cast in toward the axis from outside, first hit), pushed out by `out`."""
    ring = []
    for k in range(n):
        phi = 2 * math.pi * k / n
        d = Vector((math.sin(phi), math.cos(phi), 0))     # (inward: toward the axis from the point at phi)
        o = Vector((0, cy, z)) - d * 6.0
        hit, nn, _i, dist = tree.ray_cast(o, d, 6.0)
        p = hit if hit is not None else Vector((0, cy, z)) - d * 1.8
        ring.append(p - d * out)
    return ring


def choker(name, body, z=52.6, width=0.6, thick=0.12, mid=M_RIBBON, charm=True, tree=None):
    """A band snug round the neck at height z, `width` tall and `thick` proud of the skin, its edges rounded; a
    little cameo hung at its front if charm."""
    # (on the outermost of whatever is there: the neck, or the top of the torso where it rises round the neck's base)
    tree = tree or surface_tree()
    n = 64
    # (its cross-section, round from the top edge's inside, over the front of the band and back in under the lower edge:
    # a closed loop, so the band has an inner face against the skin and no caps)
    prof = [(0.5, 0.0), (0.5, 0.25), (0.42, 0.8), (0.25, 1.0), (0.0, 1.0), (-0.25, 1.0), (-0.42, 0.8), (-0.5, 0.25),
            (-0.5, 0.0)]
    bm = bmesh.new()
    rows = []
    for dz, f in prof:
        zz = z + dz * width
        ring = _neck_ring(tree, zz, n, out=0.04 + thick * f)
        rows.append([bm.verts.new(p) for p in ring])
    for a, b in zip(rows, rows[1:] + rows[:1]):
        for k in range(n):
            bm.faces.new((a[k], a[(k + 1) % n], b[(k + 1) % n], b[k]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    _obj(name, bm, mid, body, subdiv=1)
    if charm:
        # a little cameo hung on a jump ring from the band's lower edge at the front
        front = _neck_ring(tree, z - 0.5 * width + 0.05, 64, out=thick + 0.015)[0] + Vector((0, -0.04, 0))
        ring = _xform(torus(0.1, 0.03, 12, 5), Matrix.Rotation(math.pi / 2, 4, 'Z'))
        parts = _place(cameo(0.55), Matrix.Translation((0, -0.03, -0.1 - 0.66 * 0.55 - 0.03)))
        parts[M_ACC_GOLD] = _join([parts[M_ACC_GOLD], ring])
        parts = _place(parts, Matrix.Translation(front))
        parts = _rest_on(parts, front, tree)
        for m_, b_ in parts.items():
            _obj(f'{name} charm {female.MATERIALS[m_]["name"]}', b_, m_, body, pivot=front)


def ear_lobes():
    """Where each earlobe is pierced, in world space at rest ({-1: point, 1: point}, -1 the figure's right):
    a little above the bottom of the lobe on the base head's ear (blender/shared/ear.py), and on its outer face."""
    from ear import ear_grids, side_x_finder
    out = {}
    for e in ear_grids(side_x_finder(female.head_section, female.jaw_relief), HEAD_H, NT=110, NF=24, NB=8):
        P, lobe, rear = e['P'], e['lobe'], e['rear']
        sel = (lobe > 0.9) & (rear < 0.01)
        pts = P[sel]
        zmin = pts[:, 2].min()
        near = pts[pts[:, 2] < zmin + 0.06 * HEAD_H]
        c = near.mean(axis=0)
        out[-e['side']] = Vector((c[0], c[1], c[2] + HEAD_Z))
    return out


def earrings(name, body, style='drop', lobe=None, mid=M_ACC_GOLD, gem=M_GARNET, size=1.0):
    """A pair of earrings at the lobes (ear_lobes(), or lobe={-1: point, 1: point}): 'drop' (a gold post, a short
    link and a small garnet drop) or 'hoop' (a gold hoop through the lobe, in the plane of the head's side). Each
    is nudged out from the head and neck until it clears them."""
    lobes = lobe or ear_lobes()
    tree = surface_tree(only=('Neck', 'Head shape', 'Ear'))
    s = size
    for side, p in lobes.items():
        out = Vector((math.copysign(1.0, p.x), 0, 0))
        if style == 'hoop':
            R = 0.42 * s
            parts = {mid: _xform(torus(R, 0.05 * s, 28, 6), Matrix.Rotation(math.pi / 2, 4, 'Z') @ Matrix.Translation((0, 0, -R + 0.06)))}
        else:
            post = ellipsoid(0.09 * s, 0.09 * s, 0.09 * s, 10, 6)
            link = _xform(torus(0.1 * s, 0.03 * s, 12, 5), Matrix.Rotation(math.pi / 2, 4, 'Z') @ Matrix.Translation((0, 0, -0.18 * s)))
            drop = _place(drop_pendant(0.42 * s, gem, mid), Matrix.Translation((0, 0, -0.34 * s)))
            drop = _place(drop, Matrix.Rotation(-math.pi / 2 * math.copysign(1, p.x), 4, 'Z'))   # (its face to the side)
            parts = {mid: _join([post, link, drop.pop(mid)]), gem: drop[gem]}
        parts = _place(parts, Matrix.Translation(p + out * 0.05))
        for _ in range(20):
            lo = min(_signed(tree, v)[0] for v in _verts(parts))
            if lo >= 0.05:
                break
            parts = _place(parts, Matrix.Translation(out * 0.04))
        tag = 'R' if side < 0 else 'L'
        for m_, b_ in parts.items():
            _obj(f'{name} {female.MATERIALS[m_]["name"]}.{tag}', b_, m_, body, pivot=p, bones={'Neck': 1.0})


def brooch(name, body, x, z, size=1.0, tree=None, lift=0.06):
    """A cameo brooch pinned to the outermost surface at (x, z) on the front, turned to lie flat on it."""
    tree = tree or surface_tree()
    o = Vector((x, -30.0, z))
    hit, nn, _i, _d = tree.ray_cast(o, Vector((0, 1, 0)), 60.0)
    if hit is None:
        raise ValueError(f'brooch {name!r}: nothing at x {x}, z {z}')
    if nn.y > 0:
        nn = -nn
    R = Vector((0, -1, 0)).rotation_difference(nn).to_matrix().to_4x4()
    parts = _place(cameo(size), Matrix.Translation(hit + nn * (0.12 * size + lift)) @ R)
    for m_, b_ in parts.items():
        _obj(f'{name} {female.MATERIALS[m_]["name"]}', b_, m_, body, pivot=hit)


# ---------------------------------------------------------------- following the body
def follow(skip=('Hair',)):
    """Give every accessory the skinning of the surface it lies on, so it moves exactly as that surface does through
    the walk. Call it in finish_figure's after_skin (after skin_trunk and any mantle_on_arms / cape_on_arms):

        finish_figure(lambda: (cape_on_arms([...]), acc.follow()))

    skin_trunk has already given the accessories the trunk's weights by height, but the meshes under them blend
    between their own vertices (the neck is three rings, 2 units apart), so where the bones turn against each other
    (the head against the leaning chest in the walk) the two drift apart and a choker sank into the back of the neck.
    Here each point takes the bone weights of the nearest point of the rest surface under it, read across that
    face from its corners (as Blender's Data Transfer does). A rigid piece (a pendant, a charm, an earring, a brooch)
    takes one set for all its points, read where it hangs from, so it moves as one piece and does not bend. An
    earring hangs from the head, which is not skinned (it rides the 'Head' empty): it is given the Neck bone
    outright, which follows that empty."""
    vs, ps, ws = [], [], []
    for o in bpy.data.objects:
        if o.type != 'MESH' or o.name in ACCESSORIES or o.name.startswith(skip) or not o.vertex_groups:
            continue
        names = {g.index: g.name for g in o.vertex_groups}
        M = o.matrix_world
        b = len(vs)
        for v in o.data.vertices:
            vs.append(M @ v.co)
            ws.append({names[g.group]: g.weight for g in v.groups if g.weight > 1e-4})
        ps += [[b + i for i in p.vertices] for p in o.data.polygons]
    if not ps:
        return
    tree = BVHTree.FromPolygons(vs, ps)

    def weights_at(p):
        # (the nearest face that faces the point: not, say, the cap closing the top of the torso inside the neck)
        near = sorted(tree.find_nearest_range(p, 1.5), key=lambda h: h[3])
        hit = next((h for h in near if (p - h[0]).dot(h[1]) > -1e-3), None) or tree.find_nearest(p, 50.0)
        q, fi = hit[0], hit[2]
        corners = ps[fi]
        # (barycentric over the face's fan of triangles: the triangle the point falls in)
        from mathutils.geometry import barycentric_transform, intersect_point_tri
        best = None
        for k in range(1, len(corners) - 1):
            a, b_, c = (vs[corners[0]], vs[corners[k]], vs[corners[k + 1]])
            u = barycentric_transform(q, a, b_, c, Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1)))
            err = sum(max(0.0, -x) for x in u)
            if best is None or err < best[0]:
                best = (err, (corners[0], corners[k], corners[k + 1]), u)
        out = {}
        for vi, f in zip(best[1], best[2]):
            f = max(0.0, f)
            for bone, w in ws[vi].items():
                out[bone] = out.get(bone, 0.0) + f * w
        tot = sum(out.values()) or 1.0
        return {k: v / tot for k, v in out.items()}

    for nm in ACCESSORIES:
        o = bpy.data.objects[nm]
        M = o.matrix_world
        for g in list(o.vertex_groups):
            o.vertex_groups.remove(g)
        groups = {}
        rigid = BONES.get(nm) or (weights_at(RIGID[nm]) if nm in RIGID else None)
        for v in o.data.vertices:
            for bone, w in (rigid or weights_at(M @ v.co)).items():
                if w <= 1e-4:
                    continue
                if bone not in groups:
                    groups[bone] = o.vertex_groups.new(name=bone)
                groups[bone].add([v.index], w, 'REPLACE')


# ---------------------------------------------------------------- checking it through the walk
def clearance_report(cols=range(10), names=None, clear=0.0):
    """For each sheet column (the idle and the walk), the deepest any accessory's point sinks into the clothes and
    body under it (as posed and skinned), printed and returned as {column: (depth, mesh)}. Negative is inside."""
    names = names or list(ACCESSORIES)
    sc = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    res = {}
    for col in cols:
        sc.frame_set(col + 1)
        dg = bpy.context.evaluated_depsgraph_get()
        tree = surface_tree(accessories=False)
        worst = (9.0, None)
        for nm in names:
            o = bpy.data.objects[nm]
            me = o.evaluated_get(dg).to_mesh()
            M = o.matrix_world
            for v in me.vertices:
                s = _signed(tree, M @ v.co, 1.0)[0]
                if s < worst[0]:
                    worst = (s, nm)
            o.evaluated_get(dg).to_mesh_clear()
        res[col] = worst
        print(f'accessories col {col}: deepest {worst[0]:+.3f} ({worst[1]})', flush=True)
    sc.frame_set(1)
    return res
