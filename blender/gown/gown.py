"""A sample satin gown on the base figure: a test of the build recipe in
/mnt/project-files/gowns/gown-modeling-research.md, not a character.

A fitted dark purple satin bodice with an off-the-shoulder neckline edged in black lace, trumpet sleeves that flare
from the elbow into a bell whose mouth hangs in a long point on the inside, a low belt with a round medallion and a
long panel down the front, and a floor-length skirt in eight deep pipe folds, fanning from the hips. It is worn by a
plain grey dummy (the base figure's body, rig and walk, with an egg for a head), so only the gown is new.

    python blender/gown/gown.py --out DIR --cells 0-7:0,2,4,6
"""
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'base'))
sys.path.insert(0, os.path.join(HERE, '..', 'warden'))
import bpy  # noqa: E402
import bmesh  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402
import female  # noqa: E402
import warden as W  # noqa: E402   (the Warden's head, hair and helpers: pinch, grid, face_out, strip...)
from female import *  # noqa: E402,F401,F403
from female import _profile, _torso_dims  # noqa: E402,F401
from warden import smoothstep, wrap, pinch, grid, face_out, LEG_R, boot_rings  # noqa: E402

# ---------------------------------------------------------------- materials
SATIN = [[20, 11, 28], [42, 24, 58], [68, 42, 92], [104, 74, 134], [170, 142, 200]]
M_SATIN = mat_def(60, 'satin', SATIN, 0.0, closeup='cloth', form=True)
M_LINING = mat_def(61, 'sleeve lining', [[12, 8, 16], [20, 12, 26], [30, 18, 38], [40, 26, 50], [52, 36, 64]], 0.0)
M_LACE = mat_def(62, 'lace', [[14, 10, 16], [24, 18, 28], [34, 26, 40], [48, 38, 56], [64, 52, 72]], 0.0, flat=1)
M_PANEL = mat_def(63, 'panel', ramp(0x77707e, 0.12), 0.0, shadow_on=(60,))
M_HOSE = mat_def(64, 'hose', ramp(0x2a2030, 0.08), 0.0)
M_SHOE = mat_def(65, 'shoe', ramp(0x1e1a22, 0.08), 0.1, surface='leather')
# ---------------------------------------------------------------- who wears it
# By default a sorceress: the base head sculpted to her face, painted, with long dark hair. GOWN_DUMMY=1 puts the
# grey clay dummy with an egg for a head back (only the gown is new).
DUMMY_MODE = bool(os.environ.get('GOWN_DUMMY'))
if DUMMY_MODE:
    DUMMY = [[92, 92, 98], [122, 122, 128], [150, 150, 156], [176, 176, 182], [204, 204, 210]]
    female.MATERIALS[M_SKIN].update(ramp=np.array(DUMMY, dtype=np.float32), alt=None, detail=0.0, closeup=None)
else:
    # Her face, on the base head with the face creator's controls: elegant and composed, a little cold. A narrow,
    # tapering jaw with a clean edge and a small pointed chin; high, sharp cheekbones over hollowed cheeks; a fine,
    # straight nose; a defined brow over slightly deep sockets; a fuller lower lip with the corners lifted a touch
    # (the knowing half-smile); eyes long, tilted up at the outer corners and a little hooded.
    female.FACE.update({
        'jawWidth': -0.08, 'jawDef': 1.3, 'jawSharp': 1.25, 'jawAngle': 0.2, 'chinProj': 1.05, 'chinWidth': 0.82,
        'chinHeight': 1.08, 'cheekWidth': -0.01, 'cheekbones': 1.6, 'cbHeight': 0.008, 'apples': 0.6,
        'cheekFull': 0.3, 'noseProj': 1.0, 'noseWidth': 0.88, 'noseLen': 0.01, 'tipBall': 0.75, 'nostrils': 0.85,
        'browRidge': 1.2, 'sockets': 1.25, 'mouthWidth': 0.96, 'lipUpper': 0.95, 'lipLower': 1.2, 'smile': 0.65,
    })
    female.EYE_OPEN = 0.6
    female.EYE_TILT = 0.13
    # her make-up: a smoky violet-black worked up from the lash line into the crease, deepest in the outer V, a
    # long liner wing, and dark plum lips; brows dark, fine and high, with a sharp arch
    female.MAKEUP.update({'shadow': (62, 34, 74), 'shadowDark': (24, 12, 32), 'shadowLow': (78, 52, 86),
                          'shadowAmt': 0.95, 'shadowInner': 0.0, 'liner': 1.0, 'wing': 1.1, 'lip': (92, 30, 58),
                          'browArch': 1.7, 'browThick': 0.7, 'browHeight': 0.04})
    # pale, cool skin; near-black hair with a plum cast; violet eyes, pale toward the pupil
    PALE = [[150, 110, 112], [186, 148, 146], [210, 178, 172], [226, 200, 194], [240, 224, 218]]
    PALE_FLUSH = [[150, 100, 108], [186, 136, 142], [208, 164, 166], [224, 188, 188], [238, 214, 212]]
    female.MATERIALS[M_SKIN].update(ramp=np.array(PALE, dtype=np.float32), alt=np.array(PALE_FLUSH, dtype=np.float32))
    female.SKIN = PALE       # (the close-up skin's baked colour, skin_tones)
    M_HAIR = mat_def(2, 'hair', ramp(0x22161f, 0.08), 0.24, surface='hair')
    M_BROW = mat_def(17, 'brow', ramp(0x1e161e, 0.08), 0.0, flat=1)
    M_EYE = mat_def(16, 'iris', [[34, 16, 56], [70, 36, 120], [112, 64, 176], [156, 110, 216], [204, 170, 238]], 0.0,
                    flat='painted')
SATIN_IDS = {M_SATIN}
female.GAIT = 0.75     # (a shorter step in a floor-length skirt: 2 units more room in front, 1.5 behind)

def bare_neck(body):
    """The base neck is a short tube made to sit inside a collar: narrower than the head's own neck where they
    meet, so on a bare neck the head's underside showed as a ring round the throat. Hers runs on from the head's
    sections themselves (a hair inside them, so the head's surface is the one seen above) and widens smoothly into
    the base of the neck at the shoulders."""
    old = bpy.data.objects.get('Neck')
    if old is not None:
        bpy.data.objects.remove(old)
    H = HEAD_H
    # the head's own throat is cut off a little above where it closes, and the new neck carries on from just inside
    # its edge, so there is one surface: no rim, no overlap to show a line
    cut = (HEAD_NECK_Z + 0.06) * H
    hs = bpy.data.objects.get('Head shape')
    tree = None
    if hs is not None:
        hbm = bmesh.new()
        hbm.from_mesh(hs.data)
        bmesh.ops.delete(hbm, geom=[v for v in hbm.verts if v.co.z < cut], context='VERTS')
        hbm.to_mesh(hs.data)
        hbm.free()
        from mathutils.bvhtree import BVHTree
        tree = BVHTree.FromObject(hs, bpy.context.evaluated_depsgraph_get())   # (head space: chin at z 0)
    z_top = HEAD_Z + cut + 0.02     # (meeting the head's cut edge, not overlapping it: skin lets light through, and
    # a neck reaching up inside the head showed through it as a ring)
    z_bot = 49.4

    def head_at(zl, phi):
        # where the head's actual surface is (face and jaw relief included), cast out from the neck's axis
        d = Vector((-math.sin(phi), -math.cos(phi), 0.0))
        o = Vector((0.0, 0.15, zl))
        hit = tree.ray_cast(o, d, 20.0) if tree is not None else (None,)
        if hit[0] is None:
            c, s_ = math.cos(phi), -math.sin(phi)
            hx, hy = head_section(zl / H, c, s_)
            return hx * H, hy * H
        return hit[0].x, hit[0].y

    def pt(z, phi):
        zl = max(cut + 0.03, z - HEAD_Z)
        hx, hy = head_at(zl, phi)
        cy = 0.15
        # (from the head's edge widening into a fuller neck by the base, as a neck does, then drawn in under the
        # shoulders' surface so they grow out of it)
        e = smoothstep(HEAD_Z + cut, 51.4, z)
        bx, by = -1.95 * math.sin(phi), -1.7 * math.cos(phi) + 0.15
        g = 1.0 - 0.18 * smoothstep(51.2, 49.6, z)
        # (and flaring out to the sides at its base, into the slope of the trapezius, so the neck curves into the
        # shoulders instead of standing on them as a post with a corner each side)
        fl = NECK_FLARE * smoothstep(52.6, 51.2, z) ** 2 * math.sin(phi) ** 2
        g += fl * (1.0 - smoothstep(50.6, 49.8, z))
        return (hx + (bx - hx) * e) * g, (hy + (by - hy) * e) * g
    zs = [z_top - (z_top - z_bot) * (i / 40) ** 1.3 for i in range(41)]
    bm = face_out(loft_fn(zs, pt, n=160, cap_top=False, cap_bot=False), axis_y=0.0)
    mesh_obj('Neck', bm, M_SKIN, body, subdiv=0)   # (as dense as the head's edge, unsubdivided, so the two meet)


def open_face(push=float(os.environ.get('HAIR_PUSH', '1.3'))):
    """The creator's long waves hang as a curtain close round the face, over the brows and the outer corners of the
    eyes: right for the Warden's hood of hair, but a sorceress shows her brow. The front of the hair is drawn back
    and out from the face, most at the brows and temples and least at the parting and below the jaw, so the
    curtain frames the face instead of covering it."""
    H = HEAD_H
    for nm in ('Hair', 'Hair locks'):
        o = bpy.data.objects.get(nm)
        if o is None:
            continue
        for v in o.data.vertices:
            x, y, z = v.co
            front = smoothstep(0.05 * H, -0.35 * H, y)                       # only the hair in front of the ears
            band = smoothstep(0.2 * H, 0.5 * H, z) * smoothstep(1.0 * H, 0.75 * H, z)   # the brows and temples
            inner = smoothstep(0.55 * H, 0.15 * H, abs(x))                   # near the face, not the far sides
            k = push * front * band * inner
            if k <= 0:
                continue
            sx = max(-1.0, min(1.0, x / (0.1 * H)))      # (continuous across the parting)
            v.co = (x + sx * k, y + 0.45 * k, z + 0.25 * k)
        o.data.update()
    W.HAIR_PTS = None     # (moved: the clearance pass is not run on her)


def tuck_torso_neck(z0=50.8, z1=52.4, k=0.62):
    """The body's trunk runs up into a short neck stub, made for a collar to cover: on her bare neck it stood out
    of the neck as a second, wider ring. Its top is drawn in under the neck."""
    o = bpy.data.objects.get('Skin torso')
    if o is None:
        return
    mw = o.matrix_world
    inv = mw.inverted()
    me = o.data
    blocks = [kb.data for kb in me.shape_keys.key_blocks] if me.shape_keys else [me.vertices]
    for data in blocks:
        for v in data:
            p = mw @ v.co
            if p.z > z0:
                f = 1.0 - (1.0 - k) * smoothstep(z0, z1, p.z)
                v.co = inv @ Vector((p.x * f, (p.y - 0.15) * f + 0.15, p.z))
    me.update()


# ---------------------------------------------------------------- one skin over the shoulders
FUSE_VOXEL = float(os.environ.get('FUSE_VOXEL', '0.14'))
FUSE_FILLET = float(os.environ.get('FUSE_FILLET', '2.0'))     # how far from the join the surface is eased (units)
FUSE_ITERS = int(os.environ.get('FUSE_ITERS', '80'))
ARMPIT_Z = float(os.environ.get('ARMPIT_Z', '46.3'))          # below this the arm hangs free of the body (and is under the sleeve)


def fuse_body():
    """The base figure's trunk and arms are separate closed shapes pushed into each other, which is fine under
    clothes and wrong on bare shoulders: the arm reads as a tube set against the body, with a groove down the join.
    A modelled character has one skin there, the chest running into the deltoid and the trapezius into the top of
    the arm. So the three are made one surface:

    1. below the armpit the body's side is drawn in clear of the arm, so the two only meet at the shoulder;
    2. their union is taken as a volume and re-surfaced (a voxel remesh), which leaves one closed skin with a
       sharp crease along the join;
    3. the crease is eased into a fillet: only the points near both of the old surfaces are relaxed, the rest stay
       where they were (the smooth-union of implicit modelling, done on the mesh);
    4. the new skin is weighted to the trunk or to the arm by which it came from, blended across the join above
       the armpit (wide, so the swing of the arm stretches the skin over the shoulder) and cut clean below it."""
    from mathutils.bvhtree import BVHTree
    torso = bpy.data.objects['Skin torso']
    arms = [bpy.data.objects['Arm.R'], bpy.data.objects['Arm.L']]
    tuck_torso_neck()
    # 1. the body's side clear of each arm, below the armpit
    for v in torso.data.vertices:
        z = v.co.z
        if z > ARMPIT_Z + 1.2 or z < SH_Z - UPPER - FORE - 1.0:      # (all the way down the arm: the forearm hangs
            # against the hip, and fused there it dragged a web of skin after it in the walk)
            continue
        k = smoothstep(ARMPIT_Z + 1.2, ARMPIT_Z, z)
        for sx in (-SH_X, SH_X):
            # (moved in across the body, never round the arm: down at the hips the body is wider than the arm
            # hangs, and pushed straight away from the arm's axis it closed round the forearm)
            dy = v.co.y - 0.2
            need = _profile(female.BARE_ARM_R, max(z - SH_Z, -UPPER - FORE)) + 0.2 + 0.6
            if abs(dy) >= need:
                continue
            lim = abs(sx) - math.sqrt(need * need - dy * dy)      # (the most |x| may be at this depth)
            if v.co.x * sx > 0 and abs(v.co.x) > lim:
                nx = abs(v.co.x) + k * (lim - abs(v.co.x))
                v.co.x = math.copysign(max(0.3, nx), sx)
    torso.data.update()
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()

    def world_bm(o):
        b = bmesh.new()
        b.from_mesh(o.evaluated_get(dg).to_mesh())
        o.evaluated_get(dg).to_mesh_clear()
        b.transform(o.matrix_world)
        edges = [e for e in b.edges if e.is_boundary]
        if edges:
            bmesh.ops.holes_fill(b, edges=edges, sides=0)      # (closed, as a volume has to be)
        bmesh.ops.recalc_face_normals(b, faces=b.faces[:])
        return b

    def tree_of(bms):
        vs, ps = [], []
        for b in bms:
            base = len(vs)
            vs += [v.co.copy() for v in b.verts]
            ps += [[base + v.index for v in f.verts] for f in b.faces]
        return BVHTree.FromPolygons(vs, ps)
    tb = world_bm(torso)
    abs_ = [world_bm(a) for a in arms]
    for b in [tb] + abs_:
        b.verts.index_update()
    t_tree, a_tree = tree_of([tb]), tree_of(abs_)
    # 2. the union, re-surfaced
    me = bpy.data.meshes.new('fuse')
    allb = bmesh.new()
    for b in [tb] + abs_:
        tmp = bpy.data.meshes.new('part')
        b.to_mesh(tmp)
        allb.from_mesh(tmp)
        bpy.data.meshes.remove(tmp)
        b.free()
    allb.to_mesh(me)
    allb.free()
    ob = bpy.data.objects.new('fuse', me)
    link(ob)
    rm = ob.modifiers.new('remesh', 'REMESH')
    rm.mode = 'VOXEL'
    rm.voxel_size = FUSE_VOXEL
    rm.adaptivity = 0.0
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    em = ob.evaluated_get(dg).to_mesh()
    n = len(em.vertices)
    co = np.zeros(n * 3, dtype=np.float64)
    em.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    ed = np.zeros(len(em.edges) * 2, dtype=np.int64)
    em.edges.foreach_get('vertices', ed)
    ed = ed.reshape(-1, 2)
    out = bmesh.new()
    out.from_mesh(em)
    ob.evaluated_get(dg).to_mesh_clear()
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)
    for o in [torso] + arms:
        bpy.data.objects.remove(o)
    # 3. the fillet: relax where both old surfaces are near (above the armpit; below it nothing is fused)
    dT = np.array([t_tree.find_nearest(Vector(p))[3] for p in co])
    dA = np.array([a_tree.find_nearest(Vector(p))[3] for p in co])
    if os.environ.get('FUSE_DEBUG'):
        side = dA < dT
        mixed = side[ed[:, 0]] != side[ed[:, 1]]
        zz_ = co[ed[:, 0], 2]
        lo = mixed & (zz_ < ARMPIT_Z - 0.5)
        print('FUSE mixed edges below armpit:', int(lo.sum()), 'z range', np.round(np.percentile(zz_[lo], [0, 50, 100]), 2) if lo.any() else None)
        if lo.any():
            pts = co[ed[lo, 0]]
            print('FUSE sample pts', np.round(pts[:: max(1, len(pts) // 8)], 2))
    far = np.maximum(dT, dA)
    t = np.clip(1.0 - far / FUSE_FILLET, 0.0, 1.0)
    band = t * t * (3 - 2 * t)
    zz = np.clip((co[:, 2] - (ARMPIT_Z - 0.2)) / 1.0, 0.0, 1.0)
    band *= zz * zz * (3 - 2 * zz)
    deg = np.zeros(n)
    np.add.at(deg, ed[:, 0], 1.0)
    np.add.at(deg, ed[:, 1], 1.0)
    deg = np.maximum(deg, 1.0)[:, None]

    def relax(c, w, iters):
        for _ in range(iters):
            acc = np.zeros_like(c)
            np.add.at(acc, ed[:, 0], c[ed[:, 1]])
            np.add.at(acc, ed[:, 1], c[ed[:, 0]])
            c = c + w[:, None] * 0.5 * (acc / deg - c)
        return c
    co = relax(co, np.full(n, 0.35), 3)          # (the voxels' stair-steps, everywhere, lightly)
    if FUSE_K <= 0:
        co = relax(co, band, FUSE_ITERS)
    else:
        co = smooth_union(co, t_tree, a_tree, dT, dA, relax, band)
    out.verts.ensure_lookup_table()
    for i, v in enumerate(out.verts):
        v.co = co[i]
    bmesh.ops.recalc_face_normals(out, faces=out.faces[:])
    out.normal_update()
    nrm = np.array([v.normal[:] for v in out.verts])
    co = collarbones(co, nrm)
    pre = co.copy()
    co = trapezius_slope(co)
    # (the foot of the fillet, where it runs out above the armpit, eased: points walked out onto it beside points
    # left where they were made small pits there, which the skin's shading showed as specks)
    ax_, z_ = np.abs(co[:, 0]), co[:, 2]
    foot = 0.7 * smoothstep_np(4.0, 5.0, ax_) * smoothstep_np(46.2, 47.0, z_) * (1.0 - smoothstep_np(49.0, 50.0, z_))
    co = relax(co, foot, FUSE_FOOT_ITERS)
    # (the skin is weighted where it was before the shoulder line was let down, where the trunk and arm it was
    # made from are: weighted where it now is, the blend between them slid down the deltoid and the pose bent a
    # crease into it)
    was = {tuple(np.round(c, 4)): Vector(q) for c, q in zip(co, pre)}
    for i, v in enumerate(out.verts):
        v.co = co[i]
    out.normal_update()
    # 4. the weights
    wR, wL = arm_weights('R'), arm_weights('L')

    def weights(p):
        p = was.get(tuple(np.round(np.array(p[:]), 4)), p)
        d_t, d_a = t_tree.find_nearest(p)[3], a_tree.find_nearest(p)[3]
        # below the armpit the skin goes with whichever it came from, cut clean (the arm swings free of the side);
        # over the shoulder it is handed from the trunk to the arm across the top of the joint, by how far out it
        # lies, as a skinned shoulder is: by distance there, the bend fell in one band along the old join and the
        # arm moving out put a crease across the deltoid
        a_cut = smoothstep(-0.3, 0.3, d_t - d_a)
        a_top = smoothstep(FUSE_BLEND_X0, FUSE_BLEND_X1, abs(p.x))
        # (and over the top of the joint only partly with the arm: the arm turns about a pivot level with the top
        # of the deltoid, and lifted the skin beside it into a ridge round the shoulder when it swung out)
        a_top *= 1.0 - FUSE_CAP_LAG * smoothstep(SH_Z - 1.6, SH_Z + 0.8, p.z)
        h = smoothstep(ARMPIT_Z - 0.2, ARMPIT_Z + 1.8, p.z)
        a = a_cut + (a_top - a_cut) * h
        w = {k: v * (1.0 - a) for k, v in trunk_weights(p).items()}
        if a > 0:
            for k, v in (wR if p.x < 0 else wL)(p).items():      # (her right is -x)
                w[k] = w.get(k, 0.0) + v * a
        return w
    return mesh_obj('Skin body', out, M_SKIN, subdiv=0, skin=weights)


def _sdf(tree, p):
    """Signed distance from p to the closed surface in tree (negative inside) and its gradient."""
    loc, nrm, _, d = tree.find_nearest(p)
    v = p - loc
    sg = 1.0 if v.dot(nrm) >= 0.0 else -1.0
    g = v / d * sg if d > 1e-5 else nrm
    return sg * d, g


def fillet_k(z):
    """How wide the smooth union is at height z: full over the top of the shoulder, where the trapezius and the
    chest run into the deltoid, and closing to nothing at the armpit, where the arm hangs free of the side."""
    return FUSE_K * smoothstep(ARMPIT_Z - 0.4, ARMPIT_Z + 1.8, z)


def smooth_union(co, t_tree, a_tree, dT, dA, relax, band):
    """The join of trunk and arm as an implicit modeller makes it: the skin is the zero set of
    smin(d_trunk, d_arm, k), the polynomial smooth minimum of the two signed distances, which equals the plain
    union away from the join and fills the inside corner with a round fillet of width k along it. The re-surfaced
    union lies on min(d_trunk, d_arm) = 0; every point near the join is walked out along the field's gradient onto
    smin = 0 (a few Newton steps), then the points are spread evenly over the fillet and put back on it. Unlike
    relaxing the crease (which shrinks the shoulder as it smooths it) this only adds skin where the corner was."""
    n = len(co)
    ks = np.array([fillet_k(z) for z in co[:, 2]])
    cand = np.nonzero((ks > 1e-3) & (np.abs(dT - dA) < ks) & (np.minimum(dT, dA) < 0.3))[0]

    def project(c, idx, steps=6):
        for i in idx:
            p = Vector(c[i])
            k = ks[i]
            for _ in range(steps):
                a, ga = _sdf(t_tree, p)
                b, gb = _sdf(a_tree, p)
                h = min(1.0, max(0.0, 0.5 + 0.5 * (b - a) / k))
                f = b + (a - b) * h - k * h * (1.0 - h)
                g = gb.lerp(ga, h)
                if g.length < 1e-6:
                    break
                g.normalize()
                p -= g * f
                if abs(f) < 1e-3:
                    break
            c[i] = p
        return c
    co = project(co, cand)
    w = np.zeros(n)
    w[cand] = 1.0

    for _ in range(3):
        # (the walked-out points bunch at the fillet's middle; spread over it and pull back on)
        w2 = relax(w[:, None].repeat(3, 1), np.full(n, 1.0), 2)[:, 0]       # (a soft edge to the spreading)
        co = relax(co, 0.8 * np.clip(w2 * 1.5, 0, 1), 4)
        co = project(co, cand, steps=4)
    return co


def smoothstep_np(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def collarbones(co, nrm):
    """The collarbones, under the skin. Each runs from the notch at the base of the throat out to the top of the
    shoulder, nearly level and a little higher at its outer end; the skin over it is a soft ridge, fullest in its
    inner half where the bone is round and near the surface and fading out toward the shoulder, where it passes
    under the deltoid. Above it lies the hollow between the bone and the neck's muscle, deepest in its middle
    third. The two inner ends stand up a little either side of the notch. All of it is pushed along the skin's
    normal and shown only where the skin faces forward or up, so it fades round the shoulder instead of ending."""
    if CLAV <= 0:
        return co
    x, z = co[:, 0], co[:, 2]
    ax = np.abs(x)
    u = (ax - CLAV_X0) / (CLAV_X1 - CLAV_X0)
    uc = np.clip(u, 0, 1)
    zc = CLAV_Z + CLAV_RISE * uc ** 1.5
    d = z - zc
    along = np.exp(-((u - 0.38) / 0.46) ** 4) * smoothstep_np(-0.1, 0.3, u)      # (eased in at the throat:
    # a full ridge there stood out of the profile as a knob)                    # (fading out at both ends)
    ridge = 0.12 * (1.0 - 0.4 * uc) * along * np.exp(-(d / 0.3) ** 2)
    above = -0.12 * np.exp(-((u - 0.38) / 0.28) ** 2) * np.exp(-((d - 0.55) / 0.3) ** 2)
    # (the hollow only on the front: over the top it read as a notch in the shoulder's outline)
    above *= np.clip((-nrm[:, 1] - 0.35) / 0.35, 0.0, 1.0)
    heads = 0.03 * np.exp(-((ax - CLAV_X0 - 0.15) / 0.45) ** 2 - ((z - CLAV_Z - 0.05) / 0.4) ** 2)
    notch = -0.05 * np.exp(-(x / 0.6) ** 2 - ((z - CLAV_Z - 0.3) / 0.45) ** 2)
    disp = CLAV * (ridge + above + heads + notch)
    face = np.clip((-nrm[:, 1] + 0.6 * np.maximum(nrm[:, 2], 0) - 0.15) / 0.6, 0.0, 1.0)
    disp *= face * face * (3 - 2 * face) * (nrm[:, 1] < 0.2)
    return co + nrm * disp[:, None]


def trapezius_slope(co):
    """The base trunk's shoulders are near level from the neck out to the arm, a yoke that read as boxy once bare:
    a woman's shoulder line falls from the base of the neck along the trapezius and rounds over the deltoid. The
    top of the body is let down, nothing at the neck and more toward the shoulder point (a little more past it, so
    the outer corner opens into a curve), and the drop fades out down the chest and back, so the bodice line and
    the armpit stay where they are."""
    if TRAP_DROP <= 0:
        return co
    ax = np.abs(co[:, 0])
    D = TRAP_DROP * smoothstep_np(1.9, 6.4, ax)     # (level over the arm, which moves down whole: a drop still
    # growing across the arm bent a crease into the side of the deltoid)
    D = D + TRAP_CORNER * smoothstep_np(5.8, 7.8, ax)          # (the outer corner let down further, into a curve)
    w = smoothstep_np(45.4, 50.8, co[:, 2]) ** 1.3      # (spread over a long way down: squeezed into
    # three units it bent a crease round the deltoid)
    co = co.copy()
    co[:, 2] -= D * w
    # the deltoid drawn in a little where it stood out past the sleeve below it, a ledge in the front view
    k = TRAP_NARROW * smoothstep_np(5.4, 7.9, ax) * smoothstep_np(47.9, 49.2, co[:, 2])   # (clear of the lace)
    co[:, 0] -= np.sign(co[:, 0]) * k
    return co


FUSE_BLEND_X0 = float(os.environ.get('FUSE_BLEND_X0', '4.0'))
FUSE_BLEND_X1 = float(os.environ.get('FUSE_BLEND_X1', '6.6'))
FUSE_CAP_LAG = float(os.environ.get('FUSE_CAP_LAG', '0.7'))
FUSE_FOOT_ITERS = int(os.environ.get('FUSE_FOOT_ITERS', '8'))
TRAP_DROP = float(os.environ.get('TRAP_DROP', '1.4'))
TRAP_CORNER = float(os.environ.get('TRAP_CORNER', '0.0'))
TRAP_NARROW = float(os.environ.get('TRAP_NARROW', '0.3'))       # how far the shoulder line falls from the neck to the arm
NECK_FLARE = float(os.environ.get('NECK_FLARE', '0.22'))    # the neck's base widening into it, to the sides
FUSE_K = float(os.environ.get('FUSE_K', '2.6'))           # the width of the shoulder's fillet (0: the old crease relax)
CLAV = float(os.environ.get('CLAV', '1.0'))                # how strongly the collarbones show
CLAV_Z = float(os.environ.get('CLAV_Z', '50.2'))
CLAV_X0, CLAV_X1 = 0.75, 4.7
CLAV_RISE = float(os.environ.get('CLAV_RISE', '0.2'))


SHOULDER_DROP = float(os.environ.get('SHOULDER_DROP', '0.28'))


def round_shoulder(arm, x0):
    """The base arm's top is a round dome standing 0.7 above the body's sloping shoulder where they meet, made for a
    sleeve or a mantle to cover; bare, it read as a ball set on the body with a notch beside it. Its cap is lowered
    toward the outside (none on the inner side, where it meets the body's slope, up to SHOULDER_DROP of the cap's
    height at the outer side), so the line from the neck runs on down over the deltoid as one curve."""
    for v in arm.data.vertices:
        lx, ly, lz = v.co
        if lz <= 0.0:
            continue
        ax = abs(x0 + lx)
        k = 1.0 - SHOULDER_DROP * smoothstep(4.8, 7.6, ax)
        v.co.z = lz * k
    arm.data.update()


def tuck_torso_shoulders(x_in=4.6, z0=48.4):
    """The body's shoulder ends in a rounded edge just inside the arm's top; it showed between them as a ridge
    and a groove. Past x_in it is drawn in and down a little, under the arm's surface."""
    o = bpy.data.objects.get('Skin torso')
    if o is None:
        return
    mw = o.matrix_world
    inv = mw.inverted()
    for v in o.data.vertices:
        p = mw @ v.co
        ax = abs(p.x)
        if ax <= x_in or p.z <= z0:
            continue
        t = smoothstep(z0, z0 + 1.2, p.z)
        nx = x_in + (ax - x_in) * (1.0 - 0.45 * t)
        p = Vector((math.copysign(nx, p.x), p.y, p.z - 0.35 * t * smoothstep(x_in, x_in + 0.8, ax)))
        v.co = inv @ p
    o.data.update()


def darken_brows(k=0.32, tint=(1.0, 0.85, 1.05), dense=1.7):
    """The base brow painting is a warm mid-brown (its colours are fixed in female.brow_image); hers are near-black
    with a cool cast, so the painted image is darkened in place, alpha untouched."""
    img = female.BROW_TEX.get('img')
    if img is None:
        return
    px = np.array(img.pixels[:], dtype=np.float32).reshape(-1, 4)
    px[:, :3] *= k * np.array(tint, dtype=np.float32)
    px[:, 3] = np.clip(px[:, 3] * dense, 0.0, 1.0)     # (and denser: the painting is half see-through, which read as brown)
    # (a new image swapped into the brow material: the render reads an image's packed copy, which editing the
    # pixels of a packed image does not refresh)
    w, h = img.size
    dark = bpy.data.images.new('brows dark', w, h, alpha=True)
    dark.colorspace_settings.name = img.colorspace_settings.name
    dark.alpha_mode = img.alpha_mode
    dark.pixels.foreach_set(px.ravel())
    dark.pack()
    for m in bpy.data.materials:
        if m.node_tree is None:
            continue
        for n in m.node_tree.nodes:
            if n.type == 'TEX_IMAGE' and n.image == img:
                n.image = dark
    female.BROW_TEX['img'] = dark


# ---------------------------------------------------------------- the skirt's shape
BELT_Z = 36.6
HEM = {'front': (11.0, 0.8), 'side': (10.0, 0.5), 'back': (13.5, 0.05)}   # (radius, height) at the hem
N1, N2 = 8, 17     # the deep pipe folds round the skirt, and the shallow series that breaks them up near the hem


def fold_v(z, phi):
    """The skirt's folds at (z, phi): -1 in a pinched valley to +1 on a round crest. They fan out from the hips
    (the phase leans away from each hip as it falls) and wander a little off true."""
    d = max(0.0, 33.0 - z)
    ph = phi + 0.09 * math.sin(phi) * d / 33.0
    v = pinch(N1 * ph + 0.5 + 0.3 * math.sin(0.12 * z))
    k2 = smoothstep(14.0, 4.0, z)
    if k2 > 0:
        v = (1 - 0.25 * k2) * v + 0.25 * k2 * pinch(N2 * ph + 1.9 + 0.06 * z)
    return v


def fold_amp(z, phi):
    """How deep the folds are: none above the hips, full by z 8 (A = 2R/N sqrt(e), with e about 0.25); a quarter
    under the front panel, fuller at the back, and shallow at the sides where the hands hang against it."""
    p = wrap(phi)
    a = 1.45 * smoothstep(32.0, 8.0, z)
    a *= 1.0 - 0.75 * math.exp(-(p / 0.38) ** 2)
    a *= 1.0 + 0.3 * smoothstep(1.8, 2.6, abs(p))
    a *= 1.0 - 0.7 * math.sin(p) ** 4 * smoothstep(22.0, 27.0, z)
    return a


def hem_z(phi):
    c = math.cos(phi)
    h = HEM['front'][1] * max(0.0, c) ** 2 + HEM['side'][1] * math.sin(phi) ** 2 + HEM['back'][1] * max(0.0, -c) ** 2
    # (the hem rises over each crest and dips in each valley, so its edge shows the tubes' S-curves)
    return max(0.03, h + 0.4 * fold_v(1.0, phi) * fold_amp(1.0, phi) / 1.45)


def skirt_pt(z, phi):
    """The skirt at (z, phi): on the hips at +0.4 down to z 32 (the hang line), then flaring to the hem's radii,
    kept in at the sides (inside the hanging hands) down to z 26."""
    zc = max(z, 32.0)
    x0, y0 = torso_point(zc, phi, 0.4, bust=0.0)
    if z < 32.0:
        c, s_ = math.cos(phi), math.sin(phi)
        rf = HEM['front'][0] if c >= 0 else HEM['back'][0]
        x1, y1 = -HEM['side'][0] * s_, -rf * c
        t = ((32.0 - z) / 32.0) ** 1.1
        t_side = smoothstep(26.0, 0.0, z) ** 1.15
        w = s_ ** 4
        t = lerp(t, t_side, w)
        x0, y0 = lerp(x0, x1, t), lerp(y0, y1, t)
    r = math.hypot(x0, y0) or 1.0
    off = fold_amp(z, phi) * fold_v(z, phi)
    return x0 + x0 / r * off, y0 + y0 / r * off


def skirt_crease(z, phi):
    return min(1.0, 0.9 * fold_amp(z, phi) / 1.45 * max(0.0, -fold_v(z, phi)))


def skirt_sheet(rows=120, cols=241):
    """The skirt as one sheet of rows down and columns round. Down each column the cloth never turns back in on
    itself: below the hips its distance from the axis only grows (a fold's valley may not undercut the crest
    above it, which faced the surface down, away from the sky, and drew black blotches in the sheet)."""
    P = {}
    for j in range(cols):
        phi = 2 * math.pi * j / (cols - 1)
        rmax = 0.0
        for i in range(rows):
            v = i / (rows - 1)
            z = lerp(BELT_Z + 0.2, hem_z(phi), v ** 1.05)
            x, y = skirt_pt(z, phi)
            r = math.hypot(x, y)
            if z < 33.0 and r < rmax:
                x, y = x * rmax / r, y * rmax / r
                r = rmax
            rmax = max(rmax, r)
            P[i, j] = (x, y, z)
    return face_out(grid(rows, cols, lambda i, j: P[i, j]), axis_y=0.0)


def skirt_front_y(x, z):
    """The skirt's surface seen from the front at (x, z), for laying the panel on it."""
    best = None
    for k in range(-60, 61):
        phi = 0.9 * k / 60
        px, py = skirt_pt(z, phi)
        if best is None or abs(px - x) < best[0]:
            best = (abs(px - x), py)
    return best[1]


def gown_skirt_weights(co):
    """The pelvis at the top; below the hips each side's thigh takes a share rising to 0.45 at the knee and easing
    back to 0.3 at the hem (full thigh weights would swing the hem 14 units and split it), blended over a wide band
    across the middle so the centre front and back are carried half by each leg."""
    w = trunk_weights(co)
    f = 0.45 * smoothstep(31.0, 17.0, co.z) - 0.15 * smoothstep(17.0, 0.0, co.z)
    # (the back carries less: a thigh swinging forward would tip the back of the skirt forward with it, turning
    # it to face the ground; behind, the legs' proxies push the cloth out where a heel comes up into it)
    f *= lerp(0.3, 1.0, smoothstep(2.0, -2.0, co.y))
    if f <= 1e-4:
        return w
    r = 1.0 / (1.0 + math.exp(max(-40.0, min(40.0, co.x / 3.0))))      # her right is -X
    out = {k: v * (1.0 - f) for k, v in w.items()}
    out['Thigh.R'] = f * r
    out['Thigh.L'] = f * (1.0 - r)
    return out


def panel_weights(co):
    w = trunk_weights(co)
    f = 0.25 * smoothstep(33.0, 14.0, co.z)
    out = {k: v * (1.0 - f) for k, v in w.items()}
    out['Thigh.R'] = out['Thigh.L'] = 0.5 * f
    return out


# ---------------------------------------------------------------- the bodice's neckline
def neck_z(phi):
    """The off-the-shoulder neckline round the body: a soft sweetheart over the bust (dipping at the middle), rising
    to cross under each arm at the sleeves' top, and straight across the back."""
    p = wrap(phi)
    c = math.cos(p)
    front = 45.2 + 0.5 * math.exp(-((abs(p) - 0.42) / 0.25) ** 2) - 0.5 * math.exp(-(p / 0.15) ** 2)
    side = 46.6
    back = 46.0
    if c >= 0:
        return lerp(side, front, c ** 1.5)
    return lerp(side, back, (-c) ** 1.5)


BODICE_OFF = 0.18


CUP_X, CUP_Z = 2.35, 44.1     # the centres of the bodice's two cups
BUST = tuple(float(v) for v in os.environ.get('BUST', '1.0,2.5,2.0,0.8').split(','))   # depth, width, underside, bridge
DRAPE = float(os.environ.get('DRAPE', '0.35'))


def _bodice_shaped(z, phi):
    """The bodice over the body. The base body's bust is two broad overlapping forms that read as one slab once
    the neckline cuts them; the bodice is cut and boned into two distinct cups instead: narrower and fuller, rounder
    underneath (so the cloth turns sharply back to the ribs and a shadow falls under each cup), and spanning the
    cleavage at the body's own depth. It never goes inside the body: a soft max of the two."""
    x, yb = torso_point(z, phi, BODICE_OFF)
    c = math.cos(phi)
    if c <= 0:
        return x, yb
    _, y0 = torso_point(z, phi, BODICE_OFF, bust=0.0)
    amp, wx, sz_lo, bridge = BUST
    # (one soft form per side; between the cups the cloth spans across instead of following the cleavage in, as a
    # boned bodice does: `bridge` of the way the cup's full depth is held toward the middle)
    ax = abs(x)
    dx = ax - CUP_X if ax > CUP_X else (ax - CUP_X) * (1.0 - bridge)
    sz = 2.6 if z > CUP_Z else sz_lo
    cup = amp * math.exp(-(dx / wx) ** 2 - ((z - CUP_Z) / sz) ** 2)
    yc = y0 - cup * c
    k = 0.4   # (soft min: y is negative toward the front, so the further-out of the two wins)
    y = -k * math.log(math.exp(-yb / k) + math.exp(-yc / k))
    return x, y


def bodice_point(z, phi):
    """The bodice, eased: below the bust the cloth falls from the cups toward the waist in a gentle slope (DRAPE
    units in per unit down) instead of tucking in under each breast, so it reads as a fitted gown, not a second
    skin. Soft max again, so the waist still meets the body."""
    x, y = _bodice_shaped(z, phi)
    if z >= CUP_Z or math.cos(phi) <= 0:
        return x, y
    _, ya = _bodice_shaped(CUP_Z, phi)
    line = ya + DRAPE * (CUP_Z - z) ** 1.3
    # (a smooth min whose rounding grows in from nothing at the cups' height, so there is no step where it starts)
    k = 0.6 * smoothstep(CUP_Z, CUP_Z - 1.5, z)
    m = min(y, line)
    if k > 0:
        h = max(k - abs(y - line), 0.0) / k
        m -= h * h * k * 0.25
    return x, m


def _lace_on_bodice(z, phi, lift):
    x, y = bodice_point(z, phi)
    r = math.hypot(x, y)
    return (x * (r + lift) / r, y * (r + lift) / r) if r > 0 else (x, y)


def bodice_sheet(rows=64, cols=161):
    def pt(i, j):
        phi = 2 * math.pi * j / (cols - 1)
        z = lerp(neck_z(phi), BELT_Z - 0.4, i / (rows - 1))
        x, y = bodice_point(z, phi)
        return (x, y, z)
    return face_out(grid(rows, cols, pt), axis_y=0.0)


def lace_band(point_at, edge_z, cols, scallops, up=True, width=0.5, lift=0.08):
    """A narrow band of lace along an edge: from the edge out by `width`, its outer edge scalloped (one round
    scallop every 1.2-1.5 units), lying `lift` off the cloth. point_at(z, phi, lift) -> (x, y)."""
    sg = 1.0 if up else -1.0

    def pt(i, j):
        phi = 2 * math.pi * j / (cols - 1)
        e = edge_z(phi)
        sc = abs(math.sin(scallops * phi / 2))
        z = e - 0.1 * sg + (width + 0.15 * sc) * sg * i / 2
        x, y = point_at(z, phi, lift)
        return (x, y, z)
    return face_out(grid(3, cols, pt), axis_y=0.0)


# ---------------------------------------------------------------- the sheet's satin highlight
_form_shade = female.form_shade
_L = np.array([-0.55, -0.75, 0.62])
_L /= np.linalg.norm(_L)
_H = _L + np.array([0.0, -1.0, 0.0])
_H /= np.linalg.norm(_H)
_AH = math.atan2(_H[0], -_H[1])


def satin_form_shade(ch, light, mid, cov, sun, nz, ny):
    """The form shading, then satin's highlight: its floats run down the skirt, so the highlight depends on which
    way the surface faces round the vertical and hardly on its tilt. Each crest turned toward the half vector gets
    a narrow streak of the top tone; the valleys go deeper than wool's."""
    light = _form_shade(ch, light, mid, cov, sun, nz, ny)
    sel = np.isin(mid, list(SATIN_IDS))
    if not sel.any():
        return light
    nx = female.find(ch, 'Normal.X') / cov
    cr = np.clip(female.find(ch, 'crease') / cov, 0, 1)
    a = np.arctan2(nx, -ny)
    d = np.angle(np.exp(1j * (a - _AH)))
    streak = np.exp(-(d / 0.22) ** 2) * np.clip(1.0 - np.abs(nz) * 1.4, 0, 1)
    shaded = light * 0.72 - 0.05 - 0.3 * cr + 0.9 * streak * (1.0 - cr) * np.clip(sun * 2.0, 0, 1)
    return np.where(sel, shaded, light)


female.form_shade = satin_form_shade


def _lin(rgb):
    return tuple((c / 255) / 12.92 if c / 255 <= 0.04045 else ((c / 255 + 0.055) / 1.055) ** 2.4 for c in rgb)


SATIN_BASE = _lin((44, 16, 72))       # the cloth's colour in diffuse light, a deep violet
SATIN_STREAK = _lin((185, 140, 230))  # the sharp streak: tinted, never white (white reads as metal)
SATIN_SHEEN = _lin((205, 180, 235))   # the soft rim sheen
SLEEVE_SOFT = (0.14, 0.08, 0.08)      # sleeves: streak share, + streak roughness, + broad roughness
BODICE_SOFT = tuple(float(v) for v in os.environ.get('BODICE_SOFT', '0.26,0.28,0.12,0.6').split(','))   # bodice: the same, plus less anisotropy; a fitted bodice is curved both ways, so a hard-edged streak there
# showed the light's edge as a crisp line; rougher and rounder, it fades out softly (tested A-D, C chosen)


def satin_closeup(name='satin'):
    """The satin up close (the 3D views; the sheet keeps its plain diffuse surface). Two lobes, as the
    microcylinder measurements of satin describe it (Sadeghi et al. 2013): the long flat floats give a narrow
    specular streak drawn out along the threads, and under it light scattered through the dyed fibres gives a broad,
    coloured lobe and the diffuse. Principled carries the diffuse, the broad lobe and a little sheen; an anisotropic
    GGX Glossy carries the streak, mixed in at about a third. The streak takes the cloth's hue: a near-white one
    turned the gown to gunmetal in the study (satin_study.py v6). Everything sits in a node group named
    'close bsdf' so sprite_mode can swap it for the sheet's diffuse."""
    m = bpy.data.materials.get(name)
    if m is None or 'close bsdf' not in m.node_tree.nodes:
        return
    nt = m.node_tree
    old = nt.nodes['close bsdf']
    loc = old.location.copy()
    nt.nodes.remove(old)
    g = bpy.data.node_groups.new('satin two-lobe', 'ShaderNodeTree')
    g.interface.new_socket('BSDF', in_out='OUTPUT', socket_type='NodeSocketShader')
    gn = g.nodes
    gl = g.links.new

    def node(kind, **inputs):
        n = gn.new(kind)
        for k, v in inputs.items():
            if hasattr(v, 'is_linked'):
                gl(v, n.inputs[k])
            else:
                n.inputs[k].default_value = v
        return n

    def math(op, a, b, c=None):
        n = gn.new('ShaderNodeMath')
        n.operation = op
        for i, x in enumerate((a, b, c)):
            if x is None:
                continue
            if hasattr(x, 'is_linked'):
                gl(x, n.inputs[i])
            else:
                n.inputs[i].default_value = x
        return n.outputs['Value']

    def vary(lo, hi, scale):
        """The gloss varies a little from patch to patch."""
        r = node('ShaderNodeMapRange', **{'To Min': lo, 'To Max': hi})
        gl(node('ShaderNodeTexNoise', Vector=v, Scale=scale).outputs['Fac'], r.inputs['Value'])
        return r.outputs['Result']

    rest = gn.new('ShaderNodeAttribute')
    rest.attribute_name = 'rest'
    v = rest.outputs['Vector']
    # the sleeves (pass index 1) are a softer satin: their fluted bells turn every flute to the light at once, so
    # at full strength they glare; there the streak is weaker and wider and the broad lobe more matte
    idx = gn.new('ShaderNodeObjectInfo').outputs['Object Index']
    soft = math('MULTIPLY', idx, math('SUBTRACT', 2.0, idx))              # 1 on the sleeves (index 1), else 0
    fit = math('MULTIPLY', math('MULTIPLY', idx, math('SUBTRACT', idx, 1.0)), 0.5)   # 1 on the bodice (index 2)
    # the surface: fine floats running down the cloth and soft rumples in the drape, so no streak runs ruler-straight
    warp = node('ShaderNodeTexWave', Vector=v, Scale=9.0, Distortion=0.4)
    warp.bands_direction = 'X'
    rum = node('ShaderNodeTexNoise', Vector=v, Scale=0.22, Detail=2.0)
    bump = node('ShaderNodeBump', Strength=0.3, Distance=0.6,
                Height=math('MULTIPLY_ADD', warp.outputs['Fac'], math('MULTIPLY_ADD', fit, -0.02, 0.02), rum.outputs['Fac']))
    # (no weave on the bodice: pulled taut, its floats show as ribbing close up)
    nrm = bump.outputs['Normal']
    # the broad lobe and the diffuse
    b = node('ShaderNodeBsdfPrincipled', Roughness=math('MULTIPLY_ADD', fit, BODICE_SOFT[2],
                                                       math('MULTIPLY_ADD', soft, SLEEVE_SOFT[2], vary(0.44, 0.56, 0.6))),
                  Normal=nrm)
    b.inputs['Base Color'].default_value = (*SATIN_BASE, 1)
    b.inputs['Specular IOR Level'].default_value = 0.5
    b.inputs['Specular Tint'].default_value = (*[min(1.0, c * 3.0) for c in SATIN_BASE], 1)
    b.inputs['Sheen Weight'].default_value = 0.2
    b.inputs['Sheen Roughness'].default_value = 0.35
    b.inputs['Sheen Tint'].default_value = (*SATIN_SHEEN, 1)
    # the streak: tangent round the vertical axis (for the Glossy node that lays the streak along the folds;
    # Principled's Anisotropic Rotation is a quarter turn off from this)
    tg = gn.new('ShaderNodeTangent')
    tg.direction_type = 'RADIAL'
    tg.axis = 'Z'
    s_ = node('ShaderNodeBsdfGlossy', Roughness=math('MULTIPLY_ADD', fit, BODICE_SOFT[1],
                                             math('MULTIPLY_ADD', soft, SLEEVE_SOFT[1], vary(0.049, 0.098, 1.1))),
              Anisotropy=math('MULTIPLY_ADD', fit, -BODICE_SOFT[3], 0.92), Rotation=0.0,
              Tangent=tg.outputs['Tangent'], Normal=nrm)
    s_.distribution = 'GGX'
    s_.inputs['Color'].default_value = (*SATIN_STREAK, 1)
    mix = node('ShaderNodeMixShader', Fac=math('MULTIPLY_ADD', fit, BODICE_SOFT[0] - 0.34, math('MULTIPLY_ADD', soft, SLEEVE_SOFT[0] - 0.34, 0.34)))
    gl(b.outputs['BSDF'], mix.inputs[1])
    gl(s_.outputs['BSDF'], mix.inputs[2])
    out = gn.new('NodeGroupOutput')
    gl(mix.outputs['Shader'], out.inputs['BSDF'])
    grp = nt.nodes.new('ShaderNodeGroup')
    grp.node_tree = g
    grp.name = 'close bsdf'
    grp.location = loc


# ---------------------------------------------------------------- build
def build():
    root, body, pelvis, chest = begin_figure('Gown')

    # ---- the body's skin under the neckline, and the bodice over it
    torso_mesh('Skin torso', M_SKIN, body)
    bo = mesh_obj('Bodice', bodice_sheet(), M_SATIN, body, solid=0.1)
    bo.pass_index = 2   # the fitted satin (satin_closeup)
    # (two faint princess seams from the bust to the waist)
    # (and, for the sheet, a soft shadow under each cup and down the cleavage, where the 3D gets it from the form)
    def bodice_crease(co):
        seam = 0.25 * math.exp(-((abs(co.x) - 2.4) / 0.18) ** 2) * smoothstep(44.5, 42.0, co.z)
        under = 0.55 * math.exp(-((abs(co.x) - CUP_X) / 1.5) ** 2 - ((co.z - (CUP_Z - 1.9)) / 0.55) ** 2)
        cleft = 0.35 * math.exp(-(co.x / 0.45) ** 2 - ((co.z - (CUP_Z + 0.4)) / 0.9) ** 2)
        return (co.y < 0) * max(seam, under, cleft)
    crease_attr(bo, bodice_crease)
    mesh_obj('Lace neckline', lace_band(lambda z, p, l: _lace_on_bodice(z, p, l), neck_z, 241, 52, up=True,
                                        width=0.45), M_LACE, body, subdiv=0, solid=0.04)

    if DUMMY_MODE:
        # the dummy's neck and an egg for a head, on the head's pivot
        mesh_obj('Neck', loft([(53.5, 1.4, 1.6, 0, 0.43), (53.0, 1.75, 1.85, 0, 0.3), (51.0, 2.3, 2.15, 0, 0.2)]),
                 M_SKIN, body)
        head = empty('Head', (0, 0, HEAD_Z), body)
        mesh_obj('Head shape', ellipsoid(2.9, 3.2, 4.0, 24, 16), M_SKIN, head, loc=(0, 0.6, 3.9))
    else:
        # the sorceress: the base figure's neck and sculpted head, and the face creator's long waves with a centre
        # parting (the Warden's hairstyle, in her colour; lifted clear of the gown after the rig)
        if os.environ.get('GOWN_NO_HEAD'):
            # (the body only: no head, face or hair, which also saves their three minutes of building)
            head = empty('Head', (0, 0, HEAD_Z), body)
        else:
            head, head_tree = build_head(body)
        bare_neck(body)
        if not os.environ.get('GOWN_NO_HAIR') and not os.environ.get('GOWN_NO_HEAD'):
            W.creator_hair(head)
            open_face()
        darken_brows()

    build_rig(body)

    # ---- the skirt, from the belt to the floor
    sk = mesh_obj('Skirt', skirt_sheet(), M_SATIN, skin=gown_skirt_weights)
    crease_attr(sk, lambda co: skirt_crease(co.z, math.atan2(-co.x, -co.y)))

    # ---- the belt: a band round the hips dipping to a point at the front, with a medallion, and the long panel
    def belt_z(p):
        return BELT_Z - 0.9 * math.exp(-(wrap(p) / 0.7) ** 2)

    def belt_pt(z, p, l=0.0):
        return torso_point(max(z, 32.0), p, 0.62 + l, bust=0.0)
    bt = lace_band(lambda z, p, l: belt_pt(z, p, l), belt_z, 161, 0, up=True, width=0.75, lift=0.0)
    mesh_obj('Belt', bt, M_PANEL, skin=trunk_weights, subdiv=0, solid=0.12)
    by_ = belt_pt(BELT_Z - 0.9, 0.0)[1]
    mesh_obj('Belt medallion', W.medallion(), M_PANEL, loc=(0, by_ - 0.3, BELT_Z - 0.55), subdiv=0, skin=trunk_weights)

    def panel_path(t):
        return (0.0, lerp(BELT_Z - 1.3, 13.0, t))

    def panel_half(z):
        return 0.95 * smoothstep(12.6, 15.5, z) ** 0.5

    pb = bmesh.new()
    rows_ = []
    for i in range(60):
        z = lerp(BELT_Z - 1.1, 12.6, i / 59)
        hw = max(0.02, panel_half(z))
        row = [pb.verts.new((u * hw, skirt_front_y(u * hw, z) - 0.3, z)) for u in (-1.0, -0.5, 0.0, 0.5, 1.0)]
        rows_.append(row)
    for a, b in zip(rows_, rows_[1:]):
        for k in range(4):
            pb.faces.new((a[k], a[k + 1], b[k + 1], b[k]))
    pn = mesh_obj('Panel', face_out(pb), M_PANEL, skin=gown_skirt_weights, subdiv=1, solid=0.08)
    # (the knotwork, as a dark groove down the middle and dark marks across it every 1.6 units)
    crease_attr(pn, lambda co: max(math.exp(-(co.x / 0.18) ** 2), 0.8 * (math.sin(co.z * 3.9) > 0.75)))

    # ---- arms: bare skin under a trumpet sleeve
    def sleeve(side, s, x, w):
        def fore(co, w=w):
            # (the bell hangs from the forearm, but its long hanging drop lags: past the wrist part of it stays with
            # the upper arm, so when the elbow bends in the walk the cloth keeps falling instead of swinging out
            # level with the forearm)
            d = w(co)
            f = d.get(f'Forearm.{side}', 0.0) + d.pop(f'Hand.{side}', 0.0)
            # (across the elbow the cloth is shared out over a long span, so a bent elbow rounds the sleeve
            # instead of folding a crease into it where the upper sleeve meets the bell)
            u_ = d.get(f'UpperArm.{side}', 0.0)
            if u_ + f > 0.0:
                ez = co.z - SH_Z
                share = smoothstep(-UPPER + 3.0, -UPPER - 4.5, ez)
                near = smoothstep(-UPPER + 5.0, -UPPER + 3.0, ez) * smoothstep(-UPPER - 7.0, -UPPER - 4.5, ez)
                fs = lerp(f / (u_ + f), share, near)
                f, d[f'UpperArm.{side}'] = (u_ + f) * fs, (u_ + f) * (1.0 - fs)
            lag = 0.55 * smoothstep(SH_Z - UPPER - FORE + 1.5, SH_Z - UPPER - FORE - 7.0, co.z)
            d[f'Forearm.{side}'] = f * (1.0 - lag)
            d[f'UpperArm.{side}'] = d.get(f'UpperArm.{side}', 0.0) + f * lag
            return d
        ar = bare_arm(side, s, x, w)
        if not DUMMY_MODE:
            round_shoulder(ar, x)
        TOP = -2.4                     # (local height of the sleeve's top: where the neckline crosses the arm)
        WR = -UPPER - FORE             # the wrist
        phi_in = -s * math.pi / 2      # (toward the body)
        phi_under = -s * (math.pi / 2 + 0.7)   # (where the bell hangs longest: inside and a little behind)

        def off_skirt(lx, ly, lz, gap):
            # (the hanging bell lies on the skirt instead of passing into it: pushed out from the skirt's axis to
            # just past its surface, in the rest pose)
            X, Y, Z = x + lx, 0.2 + ly, SH_Z + lz
            if Z > 33.0:
                return lx, ly
            pw = math.atan2(-X, -Y)
            rs = math.hypot(*skirt_pt(Z, pw)) + gap * smoothstep(33.0, 31.0, Z)
            r = math.hypot(X, Y)
            if r >= rs:
                return lx, ly
            return X * rs / r - x, Y * rs / r - 0.2

        def mouth(phi):
            return WR + 0.6 - 9.0 * (0.5 - 0.5 * math.cos(phi - phi_under)) ** 1.5

        def bell(z, phi):
            # (the bell opens gradually from just under the lace at the shoulder, slowly over the upper arm and
            # faster past the elbow, so the sleeve and its bell read as one piece of cloth; a flare that starts at
            # the elbow put a kink there, a fitted tube with a cone stuck on)
            t = min(1.0, max(0.0, (TOP - 1.0 - z) / (TOP - 1.0 - (WR + 1.0))))
            k = t ** 1.7
            # (the cloth hangs: the bell opens front to back, where it falls free, and stays narrow across, out
            # from the body and in toward it, so it does not stand out from the arm like a stiff cone)
            inside = max(0.0, math.cos(phi - phi_in)) ** 2
            r = k ** 1.2 * (1.1 + 1.25 * math.cos(phi) ** 2) * (1.0 - 0.35 * inside)
            r += 0.6 * k * pinch(5 * phi + 0.7 + 0.2 * z)
            return r

        def pt(u, phi, out=0.0):
            z = lerp(TOP, mouth(phi), u)
            zz = max(z, WR + 0.4)
            x_, y_ = arm_xy(zz, phi, s, 0.22 + out + bell(z, phi))
            x_, y_ = off_skirt(x_, y_, z, 0.5 + out)
            return x_, y_, z - u

        us = [i / 70 for i in range(71)]
        mesh_obj(f'Sleeve.{side}', loft_fn(us, pt, n=160, cap_top=False, cap_bot=False), M_SATIN, loc=(x, 0.2, SH_Z),
                 skin=fore)
        so = bpy.data.objects[f'Sleeve.{side}']
        so.pass_index = 1   # the softer satin (satin_closeup)
        crease_attr(so, lambda co: 0.0)
        mesh_obj(f'Sleeve lining.{side}', loft_fn(us[20:], lambda u, p: pt(u, p, -0.1), n=160, cap_top=False, cap_bot=False),
                 M_LINING, loc=(x, 0.2, SH_Z), skin=fore)

        def lace_pt(u, phi):
            # (a band of lace below the mouth, scalloped along its lower edge)
            sc = abs(math.sin(14 * phi / 2))
            z = mouth(phi) + 0.1 - u * (0.5 + 0.15 * sc)
            zz = max(z, WR + 0.4)
            x_, y_ = arm_xy(zz, phi, s, 0.3 + bell(mouth(phi), phi))
            x_, y_ = off_skirt(x_, y_, z, 0.55)
            return x_, y_, z - u
        mesh_obj(f'Lace sleeve.{side}', loft_fn([0.0, 0.5, 1.0], lace_pt, n=160, cap_top=False, cap_bot=False), M_LACE,
                 loc=(x, 0.2, SH_Z), skin=fore, subdiv=0, solid=0.04)
        # the sleeve's top edge: lace too, carrying the neckline over the arm
        def top_pt(u, phi):
            z = TOP + 0.45 - 0.5 * u + 0.12 * abs(math.sin(9 * phi / 2)) * (1 - u)
            x_, y_ = arm_xy(z, phi, s, 0.3)
            return x_, y_, z - u
        mesh_obj(f'Lace shoulder.{side}', loft_fn([0.0, 0.5, 1.0], top_pt, n=96, cap_top=False, cap_bot=False), M_LACE,
                 loc=(x, 0.2, SH_Z), skin=w, subdiv=0, solid=0.04)
    build_arms(chest, sleeve)
    if not DUMMY_MODE and not os.environ.get('GOWN_NO_FUSE'):
        fuse_body()        # (the trunk and the bare arms made one skin over the shoulders)

    # ---- legs: hose and soft shoes (all but hidden), and the proxies that push the skirt out of their way
    proxies = []

    def leg(side, s, x, w):
        zs = [2.0 - 0.2 * i for i in range(int((2.0 + THIGH + SHIN - 3.5) / 0.2) + 1)]
        mesh_obj(f'Leg.{side}', loft_fn(zs, lambda z, p: (-_profile(LEG_R, z) * math.sin(p),
                                                          -_profile(LEG_R, z) * 1.04 * math.cos(p)), n=40),
                 M_SKIN, loc=(x, 0, HIP_Z), skin=w)
        rings = [r for r in boot_rings()[2:] if r[0] <= -SHIN + 1.9]    # (a low shoe: the foot only)
        kz = HIP_Z - THIGH
        mesh_obj(f'Shoe.{side}', loft(rings, n=40, cap_top=False), M_SHOE, loc=(x, 0, kz), skin=w, solid=0.1)
        # (closed, smooth shapes with no flat caps: Shrinkwrap 'Outside' judges inside from the nearest point's
        # normal, and a cap's hard rim misjudged cloth far from the leg as inside and snapped it onto the cap)
        z0, z1 = -8.0, -THIGH - SHIN - 2.6     # (down into the foot's proxy, so the ankle has no gap)
        pz = [lerp(z0, z1, i / 40) for i in range(41)]

        def prox(z, p):
            t = (z - z0) / (z1 - z0)
            k = math.sqrt(max(0.0, 1.0 - (2 * t - 1) ** 8))
            r = (_profile(LEG_R, z) + 0.6) * k
            return -r * math.sin(p), -r * 1.04 * math.cos(p)
        p1 = mesh_obj(f'Proxy leg.{side}', loft_fn(pz, prox, n=24), M_HOSE, loc=(x, 0, HIP_Z), skin=w, subdiv=0)
        p2 = mesh_obj(f'Proxy foot.{side}', ellipsoid(2.2, 3.8, 2.1, 20, 12), M_SHOE, loc=(x, -1.3, 1.7), skin=w, subdiv=0)
        proxies.extend([p1, p2])
    build_legs(pelvis, leg)

    res = finish_figure()
    if not DUMMY_MODE and os.environ.get('GOWN_NO_FUSE'):
        tuck_torso_neck()
        tuck_torso_shoulders()

    # ---- the skirt pushed out of the legs' way: Shrinkwrap 'Outside' after the rig, below the knee-ish
    vg = sk.vertex_groups.new(name='push')
    for v in sk.data.vertices:
        wv = smoothstep(29.0, 24.0, v.co.z)
        if wv > 1e-3:
            vg.add([v.index], wv, 'REPLACE')
    for p in proxies:
        p.hide_render = True
        if os.environ.get('GOWN_NO_PUSH'):
            continue
        p.hide_viewport = False
        m = sk.modifiers.new(f'push {p.name}', 'SHRINKWRAP')
        m.target = p
        m.wrap_method = 'NEAREST_SURFACEPOINT'
        m.wrap_mode = 'OUTSIDE'
        m.offset = 0.35
        m.vertex_group = 'push'
        sk.modifiers.move(len(sk.modifiers) - 1, 1 + proxies.index(p))
    sm = sk.modifiers.new('settle', 'CORRECTIVE_SMOOTH')
    npush = 0 if os.environ.get('GOWN_NO_PUSH') else len(proxies)
    sm.factor = 0.5
    sm.iterations = 8
    sm.vertex_group = 'push'
    sm.rest_source = 'ORCO'
    sk.modifiers.move(len(sk.modifiers) - 1, 1 + npush)
    # (the panel lies on the skirt, wherever the rig and the legs have moved it)
    m = pn.modifiers.new('on skirt', 'SHRINKWRAP')
    m.target = sk
    m.wrap_method = 'NEAREST_SURFACEPOINT'
    m.wrap_mode = 'ABOVE_SURFACE'
    m.offset = 0.12
    pn.modifiers.move(len(pn.modifiers) - 1, 1)
    satin_closeup()
    return res


def span(s, n):
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += range(int(a), int(b) + 1)
        else:
            out.append(int(part))
    return [x for x in out if 0 <= x < n]


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else HERE
    cells = argv[argv.index('--cells') + 1] if '--cells' in argv else '0-7:0,2,4,6'
    os.makedirs(out, exist_ok=True)
    build()
    if '--blend' in argv:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, 'gown.blend'), compress=True)
    rows_s, cols_s = cells.split(':')
    rows, cols = span(rows_s, ROWS), span(cols_s, COLS)
    sc = bpy.context.scene
    sc.render.use_persistent_data = True
    root = bpy.data.objects['Gown']
    sprite_mode(True)
    from PIL import Image
    g = np.zeros((PH * len(rows), PW * len(cols), 4), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        for j, col in enumerate(cols):
            sc.frame_set(col + 1)
            for i, row in enumerate(rows):
                root.rotation_euler = (0, 0, row * math.pi / 4)
                p = os.path.join(tmp, f'f{row}_{col}.exr')
                sc.render.filepath = p
                bpy.ops.render.render(write_still=True)
                px = to_pixels(read_exr(p))
                g[i * PH:(i + 1) * PH, j * PW:(j + 1) * PW] = px
                print(f'rendered row {row} col {col}', flush=True)
    img = Image.fromarray(g, 'RGBA')
    img.save(os.path.join(out, 'gown-cells.png'))
    img.resize((img.width * 3, img.height * 3), Image.NEAREST).save(os.path.join(out, 'gown-cells-3x.png'))


if __name__ == '__main__':
    main()
