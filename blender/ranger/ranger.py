"""The Ranger: the base female figure (blender/base/female.py) in her hair and clothes, rendered to a pixel-art
sprite sheet.

Long auburn hair with curtain bangs, a green quilted mantle with a mail trim and a leaf clasp, green twill
sleeves, a brown leather jerkin with a laced V and horn toggles, a belt and buckle, grey-brown twill trousers
with knee patches, and soft folded-top boots. The body, head, face, eyes, skin, hands, rig, walk cycle and the
sprite pipeline are the base figure's.

Run it with Blender's Python (either works):
    blender -b -P blender/ranger/ranger.py -- [--build-only] [--out DIR]
    python blender/ranger/ranger.py [--build-only] [--out DIR]      (with `pip install bpy OpenEXR`)

It writes blender/ranger/ranger.blend (the model, rigged with pivots and keyed with the 10 frames on the
timeline) and blender/ranger/ranger-blender.png (the sheet).
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'base'))
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
import bpy  # noqa: E402
import bmesh  # noqa: E402
from mathutils import Vector  # noqa: E402
from female import *  # noqa: E402,F401,F403  (the base figure: shapes, materials, rig, walk and sprite pipeline)
from female import _torso_dims, _profile, _gs  # noqa: E402,F401
import female  # noqa: E402

# ---------------------------------------------------------------- her face
# Designed in the face creator (/mnt/project-files/face-creator, the head preset "Ranger"): round and soft, full cheeks
# and apples, a gentle jaw running into a short rounded chin, a short rounded nose, open eyes, a soft rose blush.
female.FACE.update({
    'jawWidth': 0.05, 'jawDef': 0.7, 'jawSharp': 0.8, 'jawAngle': 0.05, 'jawUnder': 0.03, 'chinProj': 0.85, 'chinWidth': 1.12,
    'chinHeight': 0.85, 'chinTaper': 0.15, 'cheekWidth': 0.08, 'cheekbones': 0.8, 'apples': 1.45, 'cheekFull': 1.4,
    'noseProj': 0.88, 'noseWidth': 1.04, 'noseLen': 0.006, 'tipBall': 1.5, 'browRidge': 0.8, 'sockets': 0.8,
    'lipUpper': 1.1, 'lipLower': 1.15,
})
female.EYE_OPEN = 0.78
female.EYE_TILT = 0.085
female.BLUSH = (0.45, (224, 138, 140))

# ---------------------------------------------------------------- her materials (the base's skin, hair, eyes,
# brows and lips are hers as they are)
M_MANTLE = mat_def(3, 'mantle', ramp(0x56663f, 0.1), 0.2, surface='wool', closeup='wool', soft_edge=True)
M_JERKIN = mat_def(4, 'jerkin', ramp(0x765a40, 0.1), 0.24, surface='leather', form=True)
M_SHIRT = mat_def(5, 'shirt', ramp(0xaca48d, 0.09), 0.04)
M_SLEEVE = mat_def(6, 'sleeve', ramp(0x63704a, 0.1), 0.07, surface='twill', form=True)
M_LEGS = mat_def(7, 'trousers', ramp(0x5b5547, 0.1), 0.05, surface='twill', closeup='cloth', form=True)
M_KNEE = mat_def(8, 'knee patch', ramp(0x7a6c52, 0.1), 0.0, closeup='canvas')
M_BOOTS = mat_def(9, 'boots', ramp(0x684a32, 0.1), 0.24, surface='leather', closeup='leather')
M_CUFF = mat_def(10, 'boot cuff', ramp(0x7a5b40, 0.1), 0.2, surface='leather', closeup='suede')
M_SOLE = mat_def(11, 'sole', ramp(0x2f2620, 0.08), 0.0)
M_BELT = mat_def(12, 'belt', ramp(0x3d342b, 0.09), 0.18, surface='leather')
M_STEEL = mat_def(13, 'buckle', ramp(0x9a978f, 0.09), 0.0)
# (the brooch casts a pixel of shadow onto the wool below and right of it, so it sits in the cloth; see to_pixels)
M_GOLD = mat_def(14, 'clasp', ramp(0xa8905c, 0.09), 0.0, closeup='gold', shadow_on=(M_MANTLE,))
M_MAIL = mat_def(15, 'mail trim', ramp(0x7e827c, 0.08), 0.35, surface='checker')
M_LACE = mat_def(19, 'laces', ramp(0x4a3a2c, 0.08), 0.0)
M_STITCH = mat_def(22, 'stitching', ramp(0xb39a74, 0.08), 0.0, flat=2)


# ---------------------------------------------------------------- her hair
def drape_fall(dg):
    """The long hair down her back, draped rather than modelled as a solid: for each strip across it, the surface
    it rests on (the hair at the back of the head, the mantle, her back) is found by casting at the figure from behind, and the
    hair lies on that, then hangs: it may fall away from what is under it only gradually, as hair does over the edge
    of the mantle, never through it. Across, it is thickest down the middle and thins to soft edges where it wraps
    round the shoulders; its end is uneven, a few locks longer than the rest."""
    from mathutils.bvhtree import BVHTree
    off = Vector((0, 0, HEAD_Z))
    vs, ps = [], []
    for o in bpy.data.objects:
        if o.type != 'MESH' or o.parent is None or not (o.parent.name == 'Body' or o.name in ('Head shape', 'Hair fall', 'Hair cap')):
            continue
        me = o.evaluated_get(dg).to_mesh()
        base = len(vs)
        vs += [o.matrix_world @ v.co - off for v in me.vertices]
        ps += [[base + i for i in p.vertices] for p in me.polygons]
        o.evaluated_get(dg).to_mesh_clear()
    body = BVHTree.FromPolygons(vs, ps)

    def back_y(x, z):
        # (a ray that misses at the very edge, past the side of the head or the mantle, takes what lies a little
        # further in: taking 0 there pulled the edges of the fall forward beside her face as thin flaps)
        for k_ in range(8):
            hit = body.ray_cast(Vector((x * 0.94 ** k_, 30.0, z)), Vector((0, -1, 0)), 40.0)
            if hit[0] is not None:
                return hit[0].y
        return 0.0

    W = [(7.2, 2.0), (6.0, 2.9), (4.5, 3.2), (1.5, 3.3), (-1.5, 3.65), (-4.5, 4.3), (-9.0, 4.1), (-13.0, 3.6), (-16.0, 3.0), (-17.5, 2.2)]

    def width(z):
        for (z0, w0), (z1, w1) in zip(W, W[1:]):
            if z1 <= z <= z0:
                return lerp(w0, w1, (z0 - z) / (z0 - z1))
        return W[0][1] if z > W[0][0] else W[-1][1]

    NX, NZ = 241, 120
    # it starts high on the crown, flush with the hair over the head, and leaves it only where it hangs clear of
    # the skull (see the hull below), so the head of hair and the fall are one surface with no edge or fold
    z_top = 7.2
    # at the ends it parts into eight locks, each tapering to its own fine point and gathering in toward its own
    # middle, so the hair runs as one mass down to the last hand's breadth and only there comes apart
    PARTINGS = [-0.74, -0.5, -0.27, -0.04, 0.2, 0.43, 0.66]     # (u of each parting between locks)
    EDGES = [-1.0] + PARTINGS + [1.0]

    def lock_of(u):
        """Which lock u lies in, and how far across it (0 at one parting, 1 at the next)."""
        for k_ in range(len(EDGES) - 1):
            if u <= EDGES[k_ + 1] or k_ == len(EDGES) - 2:
                return k_, (u - EDGES[k_]) / (EDGES[k_ + 1] - EDGES[k_])

    def end_z(u):     # u from -1 (her right) to 1: an uneven end, the middle longest
        # the outer corners curve up into the sides, so the fall ends in a rounded edge, not a square corner;
        # each lock ends in its own fine point (some a little longer than others), the partings between them highest
        k_, f_ = lock_of(u)
        e = -16.2 - 1.0 * (1 - u * u) + 1.5 * abs(u) ** 4 - (0.8 + 0.3 * (k_ % 2) + 0.15 * (k_ % 3 == 1)) * (1.0 - abs(2 * f_ - 1)) ** 1.15
        return e

    inner, outer = [], []
    for i in range(NX):
        u = -1 + 2 * i / (NX - 1)
        zs = [lerp(z_top, end_z(u), j / (NZ - 1)) for j in range(NZ)]
        # (how far down the fall a point is, measured against one smooth end line rather than each column's own
        # end: the shaping toward the ends then runs smoothly across the locks instead of stepping at each parting)
        z_end = -16.2 - 1.0 * (1 - u * u) + 1.5 * abs(u) ** 4 - 0.6
        pts = []
        for j, z in enumerate(zs):
            v = j / (NZ - 1)
            # the last stretch draws in a little at the sides, so the outline turns into the end in a curve
            # toward the ends each lock gathers in toward its own middle, so the partings open as long narrow Vs
            # that run together higher up, the way strands converge into a tip (not as parallel slots)
            k_, _ = lock_of(u)
            uc = 0.5 * (EDGES[k_] + EDGES[k_ + 1])
            g_ = min(1.0, max(0.0, (v - 0.88) / 0.12))
            ue = lerp(u, uc, 0.3 * g_ * g_ * (3 - 2 * g_))
            vs = min(1.0, (z_top - z) / (z_top - z_end))
            x = ue * width(z) * (1.0 - 0.12 * max(0.0, (vs - 0.75) / 0.25) ** 2 * abs(u))
            # thick down the middle, thin to soft edges; thinner toward the end, and rounded off at the tip
            # (never thinner than a few voxels: a sheet thinner than the remesh breaks up into rods and tatters)
            grow = min(1.0, (z_top - z) / 5.0)
            grow = grow * grow * (3 - 2 * grow)
            t_ = (0.95 - 0.3 * vs) * max(0.0, 1 - u * u) ** (0.5 if z < -0.5 else 1.4) * (1.0 if v < 0.92 else math.cos((v - 0.92) / 0.08 * math.pi / 2) ** 0.5)
            # (the tips are the thinnest part: a lock ends in a wisp, not a rounded slab)
            t = max(t_, lerp(0.4, 0.28, min(1.0, max(0.0, (v - 0.75) / 0.25))))
            # (what it rests on, blurred across: the hair bridges the folds of the mantle instead of carrying
            # each fold down its whole length as a groove)
            # (only below the nape: over the round of the head a blurred support would sink into the hair beneath)
            # (the change from lying over the head to hanging over the mantle is eased over a hand's breadth
            # either side of the nape: switched at one height it left a notch across the fall)
            wb = min(1.0, max(0.0, (0.5 - z) / 2.0))
            wb = wb * wb * (3 - 2 * wb)
            sup = back_y(x, z)
            if wb > 0.0:
                sup = lerp(sup, sum(back_y(x + dx_, z) * w_ for dx_, w_ in
                                    ((-0.9, 0.1), (-0.45, 0.2), (0.0, 0.4), (0.45, 0.2), (0.9, 0.1))), wb)
            # the least it can be: flush with the hair over the head at the top, then lying over what is under it.
            # Over the head its edges lie flush with the hair beneath (the extra thickness goes inward, out of
            # sight), so they leave no rim along the sides of the head
            y = lerp(sup + t_ * grow, max(sup + t * grow, sup + 0.52), wb)
            pts.append([x, y, z, t, sup])
        # hair hangs: pulled down by its weight it spans straight from one thing that holds it out to the next (the
        # back of the skull, then the mantle) and never sinks into the hollow between them. So its outside is the
        # hull of those lowest-allowed points, seen from behind: no groove at the nape, no ledge where it meets the
        # mantle, the head of hair and the fall one surface
        # (the hull is taken down to a common height above the ends: taken to each column's own end, the uneven
        # lock tips tilted whole columns and carried every parting up the fall as a groove; below it the last
        # stretch of the hull carries straight on)
        top_ = [k for k, q in enumerate(pts) if q[2] >= -12.5]
        hull = []
        for k in top_:
            q = pts[k]
            while len(hull) >= 2:
                a, b = pts[hull[-2]], pts[hull[-1]]
                # drop b if it lies on or inside the line from a to q
                if (b[1] - a[1]) * (q[2] - a[2]) - (q[1] - a[1]) * (b[2] - a[2]) >= 0:
                    hull.pop()
                else:
                    break
            hull.append(k)
        for h0, h1 in zip(hull, hull[1:]):
            a, b = pts[h0], pts[h1]
            for k in range(h0 + 1, h1):
                f_ = (a[2] - pts[k][2]) / (a[2] - b[2])
                lift = max(0.0, lerp(a[1], b[1], f_) - pts[k][1])
                # (high on the round of the skull the edges keep to the hair under them, or the straightened
                # edge would stand off it as a rim; from below the crown down the whole width hangs)
                g_ = min(1.0, max(0.0, (pts[k][2] - 3.0) / 2.5))
                side = 1.0 - abs(u) ** 2 * g_ * g_ * (3 - 2 * g_)
                pts[k][1] += lift * side
        if len(hull) >= 2:
            a, b = pts[hull[-2]], pts[hull[-1]]
            slope = (b[1] - a[1]) / (a[2] - b[2]) if a[2] > b[2] else 0.0
            for k in range(hull[-1] + 1, len(pts)):
                pts[k][1] = max(pts[k][1], b[1] + slope * (b[2] - pts[k][2]))
        # where the hanging line leaves the round of the head it turned a corner (a bump in profile, high on the
        # back of the head): the outline there is eased round in a curve, a hand's breadth either way
        for _ in range(3):
            ys = [q[1] for q in pts]
            for k, q in enumerate(pts):
                fe = min(1.0, max(0.0, (q[2] - 0.5) / 1.5), max(0.0, (7.4 - q[2]) / 2.4))
                if fe > 0.0 and k >= 8:
                    w_ = ys[k - 8:k + 9]
                    q[1] = lerp(q[1], max(q[4] + 0.15, sum(w_) / len(w_)), fe)
        col_out, col_in = [], []
        for j, (x, y, z, t, sup) in enumerate(pts):
            v = j / (NZ - 1)
            vs = min(1.0, (z_top - z) / (z_top - z_end))
            if vs > 0.8:          # the ends turn in a little toward her back rather than flicking out
                y = lerp(y, max(sup + 0.52, y - 0.6), ((vs - 0.8) / 0.2) ** 2 * 0.7)
            wb = min(1.0, max(0.0, (0.5 - z) / 2.0))
            wb = wb * wb * (3 - 2 * wb)
            y = max(lerp(y - 1.0, sup + 0.42, wb), y)
            col_out.append([x, y, z, t, sup])
            # where it spans a hollow it is the full mass of hair down to what is under it (a thin sheet bridging
            # the nape would show its underside from the side); where it lies on the mantle it is pressed thinner
            gap = y - sup
            th = t if gap < t + 0.3 else max(t, gap - 0.15)
            yi = lerp(y - th, max(y - th, sup + 0.12), wb)
            # its sides are rounded over down onto what it lies on, rather than cut off square: a square side is a
            # flat wall as deep as the hair, seen as a hard corner down the back (and an edge rounded about its
            # middle instead stands proud of the hair beneath as a ridge)
            if abs(u) > 0.45 and y - yi > 0.3:
                f_ = min(1.0, (abs(u) - 0.45) / 0.55)
                y = yi + max(0.15, (y - yi) * math.sqrt(max(0.0, 1.0 - f_ * f_)))
                col_out[-1][1] = y
            col_in.append(Vector((x, yi, z)))
        col_out = [Vector((x, y, z)) for x, y, z, t, sup in col_out]
        inner.append(col_in)
        outer.append(col_out)
    bm = bmesh.new()
    vin = [[bm.verts.new(p) for p in c] for c in inner]
    vout = [[bm.verts.new(p) for p in c] for c in outer]
    for i in range(NX - 1):
        for j in range(NZ - 1):
            bm.faces.new((vout[i][j], vout[i + 1][j], vout[i + 1][j + 1], vout[i][j + 1]))
            bm.faces.new((vin[i][j + 1], vin[i + 1][j + 1], vin[i + 1][j], vin[i][j]))
    for i in range(NX - 1):          # top and bottom edges
        for j in (0, NZ - 1):
            bm.faces.new((vin[i][j], vin[i + 1][j], vout[i + 1][j], vout[i][j]))
    for i in (0, NX - 1):            # the two sides
        for j in range(NZ - 1):
            bm.faces.new((vin[i][j], vout[i][j], vout[i][j + 1], vin[i][j + 1]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


HAIR_PARTS = []


def fuse_hair(head, names, voxel=0.065):
    """Melt the hair's pieces (the cap over the skull, the side locks and the long fall) into one surface: their
    shapes are joined, remeshed as a single volume, and relaxed so each join becomes a soft fillet instead of a
    seam where one piece sits on another."""
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    mesh_obj('Hair drape', drape_fall(dg), M_HAIR, head, subdiv=0)
    names = names + ['Hair drape']
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    bm = bmesh.new()
    for n in names:
        o = bpy.data.objects[n]
        me = o.evaluated_get(dg).to_mesh()
        part = bmesh.new()
        part.from_mesh(me)
        o.evaluated_get(dg).to_mesh_clear()
        bmesh.ops.transform(part, matrix=o.matrix_local, verts=part.verts)
        tmp = bpy.data.meshes.new('part')
        part.to_mesh(tmp)
        part.free()
        bm.from_mesh(tmp)
        bpy.data.meshes.remove(tmp)
        bpy.data.objects.remove(o)
    me = bpy.data.meshes.new('hair-union')
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new('hair-union', me)
    link(ob)
    rm = ob.modifiers.new('fuse', 'REMESH')
    rm.mode = 'VOXEL'
    rm.voxel_size = voxel
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    fused = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)
    bm = bmesh.new()
    bm.from_mesh(fused)
    bpy.data.meshes.remove(fused)
    for _ in range(6):
        bmesh.ops.smooth_laplacian_vert(bm, verts=bm.verts, lambda_factor=0.6, lambda_border=0.0,
                                        use_x=True, use_y=True, use_z=True, preserve_volume=True)
    # below the ears the hair breaks into a few broad clumps: soft vertical ridges across the fall and the sides
    bm.normal_update()
    for v in bm.verts:
        # (and fading out again toward the ends, where the locks gather on their own: ridges there read as pleats)
        fade = min(1.0, max(0.0, (3.0 - v.co.z) / 4.0), max(0.0, (v.co.z + 13.0) / 5.0))
        if fade <= 0.0:
            continue
        arc = math.atan2(v.co.x, v.co.y - 0.5) * max(2.5, math.hypot(v.co.x, v.co.y - 0.5))
        # (shallow, and uneven in spacing: deep regular ridges read as a row of moulded tubes)
        ridge = 0.045 * math.sin(1.5 * (arc + 0.5 * math.sin(v.co.z * 0.25)) + 0.9 + 0.8 * math.sin(0.55 * arc)) \
            + 0.02 * math.sin(3.7 * arc + 1.3)
        n_ = Vector((v.normal.x, v.normal.y, 0.0))      # across the hair only: the ends must not be scalloped
        v.co += n_ * ridge * fade
    # and the band where the hair over the head runs into the fall is relaxed once more, so the two read as one
    # sweep with no trace of where they were joined
    # (and the ends, so the locks run together into one soft mass and only part at the very tips)
    ends = [v for v in bm.verts if v.co.z < -6.0]
    for _ in range(14):
        bmesh.ops.smooth_laplacian_vert(bm, verts=ends, lambda_factor=0.8, lambda_border=0.0,
                                        use_x=True, use_y=True, use_z=False, preserve_volume=True)
    # the side locks run into the hair beside them: the hollow where they meet is relaxed out, so they read as
    # the front of the side hair rather than a separate piece laid on it
    sides = [v for v in bm.verts if -3.5 < v.co.z < 8.2 and abs(v.co.x) > 1.6 and v.co.y < 5.0]
    for k_ in range(30):
        bmesh.ops.smooth_laplacian_vert(bm, verts=sides, lambda_factor=0.7, lambda_border=0.0,
                                        use_x=True, use_y=True, use_z=k_ % 3 == 2, preserve_volume=True)
    band = [v for v in bm.verts if -4.0 < v.co.z < 9.0 and v.co.y > 1.0]
    for _ in range(8):
        bmesh.ops.smooth_laplacian_vert(bm, verts=band, lambda_factor=0.5, lambda_border=0.0,
                                        use_x=True, use_y=True, use_z=True, preserve_volume=True)
    hair = mesh_obj('Hair', bm, M_HAIR, head, subdiv=0, rest=hair_flow)
    strand_colours(hair)
    return hair


def curtain_bang(tree, s):
    """One side of the curtain bangs, as loose strands: each leaves the centre parting from under the cap, arcs
    out and down across the forehead and ends in a fine point, the ones nearest the parting shortest and those
    reaching the temple longest, where they run on under the side lock. They lie on whatever is outermost (the
    forehead or the cap) and drape smoothly down off the cap's edge. Returns the mesh and a lookup for each
    vertex's pattern coordinates."""
    H = HEAD_H
    head_tree, cap_tree = tree
    C = Vector((0.0, 0.5, 0.0))

    def skin(az, z):
        d = Vector((-s * math.sin(az), -math.cos(az), 0.0))
        o = Vector((C.x, C.y, z))
        r = 2.5
        for t_, extra in ((head_tree, 0.0), (cap_tree, 0.4)):
            hit = t_.ray_cast(o + d * 20.0, -d, 19.9)     # (only the near side: stop at the middle)
            if hit[0] is not None:
                r = max(r, (hit[0] - o).length + extra)
        return o, d, r

    rnd = np.random.default_rng(11 if s > 0 else 12)
    bm = bmesh.new()
    rest = {}
    normals = []
    N, steps, seg = 22, 18, 10
    for k in range(N):
        f = (k + rnd.uniform(0.1, 0.9)) / N               # 0 at the parting, 1 at the temple
        az0 = math.radians(-2.0 + f * 7)     # (the inner roots cross the parting a little, so the halves meet)
        # the strands reaching the temple carry on down and back under the side lock, so the bangs run into the
        # side hair instead of ending in a clump of tips at the temple
        g = max(0.0, (f - 0.6) / 0.4)
        az1 = math.radians(12 + f * 80 + 14 * g)
        z0 = (0.93 + 0.07 * rnd.uniform()) * H
        z1 = (0.77 - 0.14 * math.sin(f * math.pi / 2) ** 0.8 - 0.12 * g + rnd.uniform(-0.03, 0.03)) * H
        rad = rnd.uniform(0.16, 0.24) * (1.15 - 0.3 * f)
        layer = 0.03 + 0.1 * (k % 2)        # alternate strands lie a little over their neighbours
        # the path first (centre, outward direction, radius at each step), smoothed so the strand has no kinks
        cs, ds, rrs, ts, r_prev = [], [], [], [], None
        for i in range(steps):
            t = i / (steps - 1)
            az = lerp(az0, az1, t ** 0.75)
            z = lerp(z0, z1, math.sin(t * math.pi / 2) ** 1.1)
            o, d, r = skin(az, z)
            if r_prev is not None:
                r = max(r, r_prev - 0.35)
            r_prev = r
            tuck = min(1.0, t / 0.35)
            tuck = tuck * tuck * (3 - 2 * tuck)     # (the roots surface gradually from under the hair)
            rr = rad * (0.6 + 0.4 * tuck) * max(0.0, 1 - t) ** 0.6 + 0.012
            # (the strands that sweep round onto the side thin out and sink into the side hair as they go, so the
            # fringe runs on into it rather than lying on it as a layer with a step along its edge)
            sink = min(1.0, max(0.0, (f - 0.4) / 0.5)) * max(0.0, (t - 0.2) / 0.8) ** 1.1
            rr *= 1.0 - 0.6 * sink
            # (the tips settle onto the skin; the root starts under the cap)
            cs.append(o + d * (r + 0.04 + rr * (1 - 0.6 * t) + layer * tuck * (1 - t * t) * (1 - sink)
                               - 0.45 * (1 - tuck) - 0.4 * sink))
            ds.append(d)
            rrs.append(rr)
            ts.append(t)
        for _ in range(3):
            cs = [cs[0]] + [(cs[i - 1] + cs[i] * 2 + cs[i + 1]) / 4 for i in range(1, steps - 1)] + [cs[-1]]
        rings, B_prev = [], None
        for i in range(steps):
            T = (cs[min(i + 1, steps - 1)] - cs[max(i - 1, 0)]).normalized()
            d, rr, t, centre = ds[i], rrs[i], ts[i], cs[i]
            Bv = T.cross(d).normalized()
            if B_prev is not None and Bv.dot(B_prev) < 0:
                Bv = -Bv
            B_prev = Bv
            Nv = Bv.cross(T).normalized()
            if Nv.dot(d) < 0:
                Nv = -Nv
            ring = []
            for j in range(seg):
                a_ = 2 * math.pi * j / seg
                q = centre + Bv * (rr * 1.25 * math.cos(a_)) + Nv * (rr * 0.8 * math.sin(a_))
                ring.append(bm.verts.new(q))
                # lit as part of the hair's outer surface: the normals lean out from the head, so a strand's
                # underside and tip do not turn dark against the forehead
                normals.append((d * 0.8 + (Bv * math.cos(a_) + Nv * math.sin(a_)) * 0.35).normalized())
                rest[q.freeze()] = (k * 2.3 + math.cos(a_) * 0.25, 0.0, -t * 3.0)
            rings.append(ring)
        for ra, rb in zip(rings, rings[1:]):
            for j in range(seg):
                bm.faces.new((ra[j], ra[(j + 1) % seg], rb[(j + 1) % seg], rb[j]))
        bm.faces.new(list(reversed(rings[0])))
        bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    from mathutils.kdtree import KDTree
    keys = list(rest)
    kd = KDTree(len(keys))
    for k, co in enumerate(keys):
        kd.insert(co, k)
    kd.balance()
    return bm, lambda co: rest[keys[kd.find(co)[1]]], normals


# ---------------------------------------------------------------- her clothes' folds
# The sleeve's creases, each a single short groove of its own: (height, angle round the arm, half-width as an
# angle, slant, curl at the ends, depth). Angles are mirrored per arm: 0 faces front, +pi/2 straight out from the
# body, -pi/2 toward it. Each sits at its own height and angle, runs only part of the way round, slants its own
# way and curls up at the ends a little, so together they read as cloth crumpling rather than as rings.
EZ = -UPPER
SLEEVE_CREASES = [
    # the armpit: two creases fanning down and out from the pit, over the front and the back of the arm
    (-1.5, -1.05, 0.55, -0.9, 0.0, 0.1), (-2.3, -2.2, 0.5, 0.8, 0.0, 0.08),
    # the inside of the elbow: three uneven crumples, the ends curling up round the arm
    (EZ + 0.9, 0.25, 0.75, 0.35, 0.5, 0.11), (EZ - 0.05, -0.2, 0.95, -0.25, 0.6, 0.13), (EZ - 1.1, 0.4, 0.55, 0.5, 0.4, 0.08),
    # above the cuff: a couple of short, slanting folds where the sleeve settles on the wrist
    (EZ - FORE + 2.3, 0.6, 0.8, 0.45, 0.3, 0.07), (EZ - FORE + 1.6, -0.9, 0.9, -0.35, 0.3, 0.08), (EZ - FORE + 1.9, 2.6, 0.7, 0.3, 0.3, 0.06),
]


# A few long folds down the outside and back of the arm, where the cloth hangs from the shoulder and from the elbow:
# (angle round the arm, as above, top, bottom, half-width as an angle, depth, drift of the line round the arm per
# unit down). Each fades in and out along its length and wanders off true, so they read as hanging cloth, not ribs.
SLEEVE_DRAPE = [
    (1.15, -0.8, -7.5, 0.28, 0.1, 0.035), (2.05, -2.0, -9.5, 0.25, 0.08, -0.03), (1.6, EZ - 1.0, EZ - FORE + 1.0, 0.26, 0.08, 0.05),
    (2.6, EZ - 2.0, EZ - FORE + 1.6, 0.22, 0.06, -0.04), (0.55, EZ - 2.5, EZ - FORE + 2.0, 0.22, 0.05, 0.03),
]


def lock_bangs(trees, s, frame=True):
    """One side of the curtain bangs built as locks, as in the face creator (HAIR_STYLES.ranger, rangerBangs; see
    docs/hair-building-guide.md): curtain locks in two staggered layers, rooted beside the centre parting under the
    cap, lifting off the brow in a C and falling back to the skin below it, the inner edge of the opening concave
    (they drop almost straight off the parting, then flare out over the brow); one wide cover lock over them; and,
    with frame, four long locks framing the face down the front of the side hair. Each lock is a clump of fine flat
    strands that part toward its end and stop at staggered lengths, each tapering to a point.
    Returns the mesh, a lookup for each vertex's pattern coordinates and the vertex normals, as curtain_bang."""
    import locks as L
    head_tree, outer_trees = trees
    hc = Vector(L.HC)

    def cast(tree, d):
        hit = tree.ray_cast(hc + d * 20.0, -d, 19.9)
        return (hit[0] - hc).length if hit[0] is not None else None

    def surface(d):
        d = Vector(d)
        scalp = cast(head_tree, d) or 4.0
        outer = scalp
        for t_ in outer_trees:
            r = cast(t_, d)
            if r is not None:
                outer = max(outer, r)
        return scalp, outer

    strands, guides = L.ranger_bangs(surface, s, frame)
    verts, faces, normals, info = L.ribbons(strands)
    # (coverage is checked on one side's locks only over that side of the head)
    L.report(f'ranger bangs {s}', strands, guides, surface, verts, faces, normals, info, cover=False)
    bare = [b for b in L.check_cover(strands, surface)[1] if b[0] * s > 0.05]
    if len(bare) > 20:
        print(f'hair lock check: bare ranger bangs {s} {len(bare)} directions inside the hairline show skin '
              f'(th {min(a for a, _ in bare):.2f}..{max(a for a, _ in bare):.2f})', flush=True)
    bm = bmesh.new()
    L.to_bmesh(bm, verts, faces)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    # (the pattern laid as on the cap, hair_flow: the locks' own coordinates glittered against it)
    return bm, hair_flow, [Vector(n_) for n_ in normals]


BANGS = 'locks'      # 'locks' (built as the face creator builds them) or 'strands' (curtain_bang); views.py --set BANGS=strands


def sleeve_folds(z, phi, s):
    """(radial offset, crease) of the sleeve at local height z, angle phi (phi 0 faces front)."""
    pm = s * phi                                    # (mirrored: +pi/2 is out from the body on either arm)
    off, cr = 0.0, 0.0
    for z0, p0, wd, slant, curl, depth in SLEEVE_CREASES:
        dp = math.atan2(math.sin(pm - p0), math.cos(pm - p0))
        win = math.exp(-(dp / wd) ** 4)              # (fading out at the ends of the crease, not cut off)
        if win < 1e-3:
            continue
        d = z - (z0 + slant * dp + curl * dp * dp)
        groove = math.exp(-(d / 0.3) ** 2)
        ridge = math.exp(-((d - 0.42) / 0.38) ** 2) + 0.6 * math.exp(-((d + 0.45) / 0.38) ** 2)
        off += depth * win * (0.55 * ridge - groove)
        cr += win * groove * min(1.0, depth / 0.11)
    for p0, zt, zb, wd, depth, drift in SLEEVE_DRAPE:
        if not (zb - 1.0 < z < zt + 1.0):
            continue
        along = min(1.0, max(0.0, (zt - z) / (zt - zb)))
        env = math.sin(math.pi * along) ** 0.8 if zb <= z <= zt else 0.0
        if env <= 0.0:
            continue
        pc = p0 + drift * (zt - z) + 0.06 * math.sin(1.3 * z + 3 * p0)
        dp = math.atan2(math.sin(pm - pc), math.cos(pm - pc))
        # (a rounded ridge with a shallow valley either side)
        g = math.exp(-(dp / wd) ** 2) - 0.45 * math.exp(-((abs(dp) - 1.3 * wd) / (0.6 * wd)) ** 2)
        off += depth * env * g
        cr += 0.6 * env * max(0.0, -g)
    # a long soft fold down each part of the arm, where the cloth hangs from the shoulder and twists down the
    # forearm, faint
    v = fold_ridge(2.0 * pm + 0.55 * z + 0.6)
    k = 0.6 * _gs(z, -5.5, 2.4) + 0.8 * _gs(z, EZ - 4.2, 2.0)
    off += 0.07 * k * v
    cr += 0.3 * k * max(0.0, -v)
    return off, min(1.0, cr)


# The trouser leg's creases, as SLEEVE_CREASES (local height from the hip joint, angle round the leg mirrored per
# leg: 0 front, +pi/2 out from the body, pi behind): a few crumples behind the knee, where the cloth folds as it
# bends, and two faint whiskers running in from the front of the hip toward the crotch.
LEG_CREASES = [
    (-THIGH + 0.9, 3.0, 0.7, 0.3, -0.3, 0.15), (-THIGH + 0.1, -2.9, 0.9, -0.25, -0.35, 0.18), (-THIGH - 0.8, 3.1, 0.6, 0.35, -0.3, 0.12),
    (-1.6, -0.45, 0.55, 1.0, 0.0, 0.06), (-2.6, -0.6, 0.5, 1.1, 0.0, 0.05),
]


def leg_folds(z, phi, s):
    """(radial offset, crease) of the trouser leg at local height z, angle phi (phi 0 faces front): its creases,
    and above the boot the cloth bunched as it is tucked in, standing out a little in a few soft vertical folds,
    most behind and at the sides."""
    pm = s * phi
    off, cr = 0.0, 0.0
    for z0, p0, wd, slant, curl, depth in LEG_CREASES:
        dp = math.atan2(math.sin(pm - p0), math.cos(pm - p0))
        win = math.exp(-(dp / wd) ** 4)
        if win < 1e-3:
            continue
        d = z - (z0 + slant * dp + curl * dp * dp)
        groove = math.exp(-(d / 0.3) ** 2)
        ridge = math.exp(-((d - 0.42) / 0.38) ** 2) + 0.6 * math.exp(-((d + 0.45) / 0.38) ** 2)
        off += depth * win * (0.55 * ridge - groove)
        cr += win * groove * min(1.0, depth / 0.1)
    blouse = _gs(z, -THIGH - 0.6, 0.8) * (0.55 - 0.45 * math.cos(phi))
    v = fold_ridge(7.0 * pm + 0.8 + 0.5 * math.sin(2 * pm))
    off += blouse * (0.16 + 0.07 * v)
    cr += 0.6 * blouse * max(0.0, -v)
    return off, min(1.0, cr)


J_EDGE = 0.12      # where the jerkin's front closure runs (phi round from the front, just to her right)


def jerkin_form(z, phi):
    """(radial offset, crease) of the jerkin's leather at height z, angle phi round from the front."""
    off, cr = 0.0, 0.0
    sx, c = math.sin(phi), math.cos(phi)
    # the waist: above and below the belt the leather is pushed into a few short wrinkles at the sides, where it
    # bends as she moves, running off on a slant toward the front and back (none across the front itself, where
    # rows of them would read as a muscled belly)
    side = abs(sx) ** 3
    for zc, w, sign in ((41.6, 1.0, 1), (37.3, 0.9, -1)):
        band = _gs(z, zc, w) * side
        v = fold_ridge(3.4 * (z - zc) + sign * 2.4 * c)
        off += 0.12 * band * v
        cr += 0.7 * band * max(0.0, -v)
    # the front panels: a seam down each side of the bust to the waist, a fine line drawn into the leather
    for x0 in (-2.9, 2.9):
        a = math.asin(max(-1.0, min(1.0, x0 / 6.0)))
        seam = _gs(phi, -a, 0.045) * max(0.0, c) * _gs(z, 42.6, 2.8)
        off -= 0.05 * seam
        cr += 0.25 * seam
    # under the arms the leather creases where it is drawn in to the waist
    for p0 in (math.pi / 2, -math.pi / 2):
        band = _gs(phi, p0, 0.45) * _gs(z, 46.0, 2.0)
        v = fold_ridge(2.6 * (z - 46.0) + 3.0 * (phi - p0))
        off += 0.1 * band * v
        cr += 0.7 * band * max(0.0, -v)
    # the front closure: from the foot of the shirt's V down to the hem the jerkin's left front laps over the
    # right, a little to her right of the middle (under the brooch). The upper panel stands proud of the one under
    # it by the leather's thickness, ending in a clean edge with a line of shade beneath it
    zf = min(1.0, max(0.0, (42.6 - z) / 0.8))
    if zf > 0.0 and c > 0.0:
        d = math.atan2(math.sin(phi - J_EDGE), math.cos(phi - J_EDGE))
        lap = min(1.0, max(0.0, (d + 1.0) / 0.7)) / (1.0 + math.exp(min(60.0, d / 0.012)))
        off += 0.17 * zf * lap
        cr += 0.7 * zf * math.exp(-((d - 0.035) / 0.03) ** 2)
    # the skirt: below the hips the leather falls in a few broad folds that deepen toward the hem
    k = min(1.0, max(0.0, (33.0 - z) / 3.5)) ** 1.2
    v = fold_ridge(5.0 * phi + 0.8 + 0.4 * math.sin(3 * phi))
    off += 0.32 * k * v
    cr += 0.8 * k * max(0.0, -v)
    # (and under the belt the leather lies smooth and is drawn in a little, so the belt cinches it)
    belt = _gs(z, 39.4, 0.8)
    return off * (1 - belt) - 0.12 * belt, min(1.0, cr * (1 - belt) + 0.5 * belt)


def leaf_etching(u, v, hl, w):
    """How deep the engraving cuts at a point of the leaf's face (0 none, 1 the full cut): a line round the
    rim just inside the edge, a line down each side of the midrib, and the side veins, which leave the midrib
    and sweep out and up toward the edge, stopping short of the border. u runs along the leaf (-1 at the stem
    to 1 at the tip), v across it (-1 to 1); w is the half-width there."""
    def line(d, c, hw):
        return min(1.0, max(0.0, 1.0 - abs(d - c) / hw)) ** 0.6

    x, y = v * w, u * hl
    ax, edge = abs(x), (1 - abs(v)) * w
    border = line(edge, 0.11, 0.032) * min(1.0, max(0.0, (0.94 - abs(u)) / 0.06))
    rib = line(ax, 0.075, 0.026) * min(1.0, max(0.0, (0.9 - abs(u)) / 0.1))
    ph = (y + 0.2 - 0.85 * ax - 0.25 * ax * ax) * 2.3
    f = abs(ph - math.floor(ph) - 0.5)
    vein = min(1.0, max(0.0, 1.0 - (0.5 - f) / 0.055)) ** 0.6
    vein *= min(1.0, max(0.0, (ax - 0.1) / 0.05)) * min(1.0, max(0.0, (edge - 0.17) / 0.05))
    vein *= min(1.0, max(0.0, (0.8 - u) / 0.15)) * min(1.0, max(0.0, (u + 0.85) / 0.15))
    return max(border, rib, vein)


def leaf_brooch(hl=1.4, hw=0.82, t=0.36, nu=128, nv=60):
    """A gold leaf, its length along local Y and its face toward local +Z: a domed blade with a raised midrib,
    pointed at both ends, the tips and edges curling back so they press down into the cloth under it. Its back
    sits 0.14 below the surface it is set on. Its face is engraved (see leaf_etching): the lines are cut into
    the mesh itself, fine enough to hold them, and their depth is kept as the 'etch' attribute, which the
    close-up gold fills with dark tarnish and the world model bakes into its colour."""
    bm = bmesh.new()
    etch = bm.verts.layers.float.new('etch')

    def width(u):
        return hw * max(0.0, 1 - u * u) ** 0.7 * (1 + 0.14 * u)

    def top(u, v):
        dome = t * max(0.0, 1 - v * v) ** 0.55 * max(0.0, 1 - u * u) ** 0.35
        rib = 0.07 * math.exp(-(v / 0.16) ** 2) * max(0.0, 1 - u * u) ** 0.5
        return dome + rib + 0.05 - 0.2 * u * u - 0.12 * v * v

    def back(u, v):
        return -0.14 - 0.2 * u * u - 0.12 * v * v

    us = [-1 + 2 * i / nu for i in range(nu + 1)]
    vs = [-1 + 2 * j / nv for j in range(nv + 1)]
    T = []
    for u in us:
        row = []
        for v in vs:
            e = leaf_etching(u, v, hl, width(u))
            vert = bm.verts.new((v * width(u), u * hl, top(u, v) - 0.045 * e))
            vert[etch] = e
            row.append(vert)
        T.append(row)
    Bk = [[bm.verts.new((v * width(u), u * hl, back(u, v))) for v in vs] for u in us]
    for i in range(nu):
        for j in range(nv):
            bm.faces.new((T[i][j], T[i][j + 1], T[i + 1][j + 1], T[i + 1][j]))
            bm.faces.new((Bk[i][j], Bk[i + 1][j], Bk[i + 1][j + 1], Bk[i][j + 1]))
    # the edge between the two faces, all the way round
    rim = [(i, 0) for i in range(nu)] + [(nu, j) for j in range(nv)] + [(i, nv) for i in range(nu, 0, -1)] + \
          [(0, j) for j in range(nv, 0, -1)]
    for (i0, j0), (i1, j1) in zip(rim, rim[1:] + rim[:1]):
        bm.faces.new((T[i0][j0], Bk[i0][j0], Bk[i1][j1], T[i1][j1]))
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-4)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def place_brooch(mantle, point, phi, z, parent, tilt=0.6):
    """Set the leaf brooch into the outside of the mantle at (phi, z): found by casting at the finished mantle
    from the front, turned to lie flat on it (its long axis tilted tilt radians off upright), and sunk in."""
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    tree = BVHTree.FromObject(mantle, dg)
    p = point(z, phi)
    hit, nrm, _, _ = tree.ray_cast(Vector((p[0], -40.0, z + p[2])), Vector((0, 1, 0)))
    if nrm.y > 0:
        nrm = -nrm
    # the surface normal, averaged over a small patch so the creases round it do not tip the brooch
    acc = Vector((0, 0, 0))
    for ox, oz in ((0, 0), (0.6, 0), (-0.6, 0), (0, 0.6), (0, -0.6)):
        h_ = tree.ray_cast(Vector((hit.x + ox, -40.0, hit.z + oz)), Vector((0, 1, 0)))
        if h_[0] is not None:
            acc += -h_[1] if h_[1].y > 0 else h_[1]
    nrm = acc.normalized()
    up = Vector((math.sin(tilt), 0, math.cos(tilt)))
    yax = (up - nrm * up.dot(nrm)).normalized()
    xax = yax.cross(nrm)
    from mathutils import Matrix
    rot = Matrix((xax, yax, nrm)).transposed()
    bm = leaf_brooch()
    bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=rot, verts=bm.verts)
    inv = rot.inverted()
    # (its pattern coordinates are the leaf's own: x across, y along it, so the veins follow the leaf)
    return mesh_obj('Leaf clasp', bm, M_GOLD, parent, loc=tuple(hit), subdiv=0, rest=lambda co: inv @ co)


def build():
    root, body, pelvis, chest = begin_figure('Ranger')

    # ---- torso: the body in its shirt, then a fitted jerkin over it: shaped to the bust, drawn in at the waist,
    # following the hips and flaring a little below them into a short skirt
    torso_mesh('Shirt', M_SHIRT, body)
    J = 0.35       # the jerkin's leather stands this far off the shirt

    def jerkin_pt(z, phi):
        off = jerkin_form(z, phi)[0]       # (its wrinkles, seams and folds: see jerkin_form)
        if z >= 32.0:
            x, y = torso_point(z, phi, J + off)
            return x, y
        # below the widest point of the hips the skirt falls straight on, opening a little
        x, y = torso_point(32.0, phi, J + (32.0 - z) * 0.06 + off)
        return x, y, -0.9 * math.cos(phi) ** 2 * (32.0 - z) / 4.0   # the hem dips at front and back

    # (its top runs up under the mantle, so the leather shows where the mantle parts below the brooch)
    jz = [49.6 - 0.4 * i for i in range(52)]          # 49.6 down to 29.2
    jb = loft_fn(jz, jerkin_pt, n=192, cap_bot=False)     # (fine enough round to hold the closure's edge)
    vee_pts = []
    for zz, w in ((47.0, 1.25), (45.5, 0.85), (44.0, 0.45), (42.6, 0.0)):
        for xx in ((-w, w) if w else (0.0,)):
            vee_pts.append((xx, front_y(jb, xx, zz) - 0.12, zz))
    lace_y = {z: front_y(jb, 0.0, z) - 0.1 for z in (46.2, 44.9)}
    jo = mesh_obj('Jerkin', jb, M_JERKIN, body, solid=0.3)
    crease_attr(jo, lambda co: jerkin_form(co.z, math.atan2(-co.x, -co.y))[1])
    # a darker leather band binds the jerkin's hem
    mesh_obj('Jerkin hem', loft_fn([30.3, 29.2], lambda z, p: jerkin_pt(z, p), n=96,
                                   cap_top=False, cap_bot=False), M_BELT, body, solid=0.5, subdiv=0)
    # stitching: a running stitch above the hem band (set into the leather, so it shows where the surface faces
    # the viewer and drops out at the silhouette instead of dotting the outline)
    st = []
    for k in range(42):
        phi = 2 * math.pi * k / 42
        p = jerkin_pt(30.65, phi)
        st.append((*outward(p, 0.2), 30.65 + (p[2] if len(p) > 2 else 0.0)))
    mesh_obj('Jerkin stitching', dots(st), M_STITCH, body, subdiv=0)
    # a running stitch down the closure's edge, and three horn toggles fastening it (above the belt, and two on
    # the skirt)
    cst = []
    for i in range(16):
        z = 42.0 - 0.72 * i
        if abs(z - 39.4) < 0.9:
            continue
        p = jerkin_pt(z, J_EDGE - 0.07)
        cst.append((*outward(p, 0.16), z + (p[2] if len(p) > 2 else 0.0)))
    mesh_obj('Closure stitching', dots(cst, 0.3), M_STITCH, body, subdiv=0)
    for i, z in enumerate((41.7, 36.9, 34.4)):
        p = jerkin_pt(z, J_EDGE - 0.16)
        tg = ellipsoid(0.55, 0.25, 0.22, 10, 6)
        mesh_obj(f'Toggle {i}', tg, M_LACE, body, loc=(*outward(p, 0.2), z), subdiv=0)
    # the shirt shows in a narrow V at the front of the jerkin, crossed by its lacing (laid on the jerkin's surface)
    vee = bmesh.new()
    vv = [vee.verts.new(p) for p in vee_pts]
    for a, b, c, d in ((0, 1, 3, 2), (2, 3, 5, 4)):
        vee.faces.new((vv[a], vv[b], vv[d], vv[c]))
    vee.faces.new((vv[4], vv[5], vv[6]))
    mesh_obj('Shirt V', vee, M_SHIRT, body, subdiv=0)
    for i, z in enumerate((46.2, 44.9)):
        w = 1.25 * (z - 42.6) / 4.4
        mesh_obj(f'Lace {i}', box(2 * w + 0.5, 0.3, 0.4), M_LACE, body, loc=(0, lace_y[z], z), subdiv=0)
    # belt and buckle at the waist
    belt = loft_fn([40.1, 39.4, 38.7], lambda z, p: torso_point(z, p, J + 0.3, bust=0.0), cap_top=False, cap_bot=False)
    mesh_obj('Belt', belt, M_BELT, body, solid=0.3, subdiv=0)
    by_ = torso_point(39.4, 0.0, J + 0.3, bust=0.0)[1]
    # the buckle: an open iron frame with its prong, the belt's tail drawn through it and lying along the belt
    # to her left, its end cut round; and a keeper, a narrow leather loop, holding the tail down
    fr = bmesh.new()
    for (cx, cz, sx_, sz_) in ((0, 0.68, 1.7, 0.32), (0, -0.68, 1.7, 0.32), (-0.7, 0, 0.32, 1.68), (0.7, 0, 0.32, 1.68)):
        r_ = bmesh.ops.create_cube(fr, size=1.0)
        bmesh.ops.scale(fr, vec=(sx_, 0.34, sz_), verts=r_['verts'])
        bmesh.ops.translate(fr, vec=(cx, 0, cz), verts=r_['verts'])
    mesh_obj('Buckle', fr, M_STEEL, body, loc=(0, by_ - 0.3, 39.4), subdiv=0)
    mesh_obj('Buckle prong', box(0.95, 0.16, 0.16), M_STEEL, body, loc=(0.25, by_ - 0.42, 39.4), subdiv=0)

    def belt_strip(x0, x1, half, lift, nseg=12, round_end=False):
        bm_ = bmesh.new()
        rows_ = []
        for i in range(nseg + 1):
            xx = lerp(x0, x1, i / nseg)
            hh = half
            if round_end:
                e = max(0.0, (i / nseg - 0.75) / 0.25)
                hh = half * math.sqrt(max(0.05, 1 - e * e))
            ph = -math.asin(max(-1.0, min(1.0, xx / (_torso_dims(39.4)[0] + J + 0.3))))
            yy = torso_point(39.4, ph, J + 0.3 + lift, bust=0.0)[1]
            rows_.append([bm_.verts.new((xx, yy, 39.4 + hh)), bm_.verts.new((xx, yy, 39.4 - hh))])
        for a_, b_ in zip(rows_, rows_[1:]):
            bm_.faces.new((a_[0], b_[0], b_[1], a_[1]))
        bmesh.ops.recalc_face_normals(bm_, faces=bm_.faces)
        return bm_
    mesh_obj('Belt tail', belt_strip(0.5, 3.4, 0.62, 0.32, round_end=True), M_BELT, body, subdiv=0, solid=0.18)
    mesh_obj('Belt keeper', belt_strip(1.35, 1.75, 0.82, 0.52, nseg=2), M_BELT, body, subdiv=0, solid=0.16)

    # ---- neck and head (the base figure's)
    head, head_tree = build_head(body)

    # ---- hair: a cap over the skull cut along a sloping hairline, curtain bangs parted at the centre,
    # side locks framing the face, and the long fall down the back
    cap = ellipsoid(0.395 * HEAD_H, 0.48 * HEAD_H, 0.54 * HEAD_H, 32, 20)
    bmesh.ops.translate(cap, vec=(0, 0.06 * HEAD_H, 0.56 * HEAD_H), verts=cap.verts)
    bmesh.ops.bisect_plane(cap, geom=cap.verts[:] + cap.edges[:] + cap.faces[:], plane_co=(0, -2.1, 6.4),
                           plane_no=Vector((0, -0.45, -0.89)).normalized(), clear_outer=True)
    # (and stopped above the jaw: lower down, beside the face, its edges stood out from behind the side locks)
    bmesh.ops.bisect_plane(cap, geom=cap.verts[:] + cap.edges[:] + cap.faces[:], plane_co=(0, 0, 2.2),
                           plane_no=Vector((0, 0, -1)), clear_outer=True)
    # over the brow the cap lies close to the skull, thinning to nothing at the front hairline, so the front of
    # the head reads as hair growing from the scalp rather than as a cap; the back and the crown keep their volume
    hl_n = Vector((0, -0.45, -0.89)).normalized()
    hc = Vector((0, 0.5, 4.8))
    cap.verts.index_update()
    cap_thick = {}
    for v in cap.verts:
        # (and its thickness thins out to the hairline too, so the edge leaves no lip under the bangs)
        dh_ = -(v.co - Vector((0, -2.1, 6.4))).dot(hl_n)
        ff = min(1.0, max(0.0, (-(v.co - hc).normalized().y - 0.25) / 0.45))
        cap_thick[v.index] = 1.0 - ff * (1.0 - min(1.0, max(0.0, dh_) / 1.8))
        d = (v.co - hc).normalized()
        f = min(1.0, max(0.0, (-d.y - 0.25) / 0.45))      # how squarely it faces forward
        f = f * f * (3 - 2 * f)
        if f <= 0.0:
            continue
        hit = head_tree.ray_cast(hc + d * 20.0, -d, 30.0)
        if hit[0] is None:
            continue
        rs, rc = (hit[0] - hc).length, (v.co - hc).length
        dh = -(v.co - Vector((0, -2.1, 6.4))).dot(hl_n)
        close = rs + 0.02 + 0.16 * math.sqrt(min(1.0, max(0.0, dh) / 1.6))
        v.co = hc + d * lerp(rc, min(rc, close), f)
    cap_tree = bvh_of(cap)
    # (strands run from the centre parting down each side; see hair_flow)
    cap_o = mesh_obj('Hair cap', cap, M_HAIR, head, solid=0.4, rest=hair_flow)
    vg = cap_o.vertex_groups.new(name='thick')
    for i_, w_ in cap_thick.items():
        vg.add([i_], w_, 'REPLACE')
    cap_o.modifiers['thickness'].vertex_group = 'thick'
    cap_o.modifiers['thickness'].thickness_vertex_group = 0.3     # (kept thick enough to fuse; see fuse_hair)
    side_trees = {}
    for s in (-1, 1):
        if BANGS == 'locks':
            # (the side lock is laid first, so the framing locks can lie on it)
            side_trees[s] = bvh_of(loft([(7.0, 0.62, 1.5, s * 2.9, 0.35), (4.5, 0.7, 2.05, s * 3.22, 1.35), (2.0, 0.8, 1.75, s * 2.85, 2.45),
                                         (-1.0, 0.75, 1.7, s * 3.0, 2.55), (-2.0, 0.6, 1.4, s * 3.1, 2.7), (-3.2, 0.4, 0.8, s * 3.1, 3.1), (-4.0, 0.15, 0.3, s * 2.8, 3.2)]))
            bm_, rest_, nrm_ = lock_bangs((head_tree, [cap_tree, side_trees[s]]), s, frame=False)      # (the side lock frames the face here: the face creator's framing locks, laid in front of it, hung off the cheek)      # (s: the side lock at x = s * 2.9 is on the side of +X for s = 1)
        else:
            bm_, rest_, nrm_ = curtain_bang((head_tree, cap_tree), s)
        bang = mesh_obj(f'Bangs {s}', bm_, M_HAIR, head, rest=rest_, subdiv=0)
        bang.data.normals_split_custom_set_from_vertices(nrm_)
        # (the strands cast no shadow of their own: their dark undersides and the shade they threw on the cap
        # marked the bangs off from the rest of the hair; the cap still shades the forehead)
        bang.visible_shadow = False
        strand_colours(bang)
        # (kept slim and close in to the fall, so once fused it reads as the front of the side hair, not a tube on it)
        mesh_obj(f'Side lock {s}', loft([(7.0, 0.62, 1.5, s * 2.9, 0.35), (4.5, 0.7, 2.05, s * 3.22, 1.35), (2.0, 0.8, 1.75, s * 2.85, 2.45),
                                         (-1.0, 0.75, 1.7, s * 3.0, 2.55), (-2.0, 0.6, 1.4, s * 3.1, 2.7), (-3.2, 0.4, 0.8, s * 3.1, 3.1), (-4.0, 0.15, 0.3, s * 2.8, 3.2)]),
                 M_HAIR, head)
    # the hair over the back of the head, down to the nape; the long hair below is draped once the mantle is on
    # (see drape_fall)
    mesh_obj('Hair fall', loft([(7.5, 3.4, 3.4, 0, 0.75), (4.5, 3.75, 4.05, 0, 0.9), (2.0, 3.75, 2.6, 0, 2.1), (0.5, 3.6, 1.8, 0, 2.8),
                                (-0.4, 3.1, 1.2, 0, 2.95),
                                (-1.4, 2.3, 0.7, 0, 2.75)]), M_HAIR, head)
    HAIR_PARTS[:] = [head, ['Hair cap', 'Side lock -1', 'Side lock 1', 'Hair fall']]

    # ---- mantle: a short shoulder cape, its hem lifting over the arms; a mail trim along the hem and a leaf clasp
    # (eased between a few rings; below the shoulders the wool falls in a few deep, slightly uneven folds that
    # grow toward the hem, and the hem itself waves with them)
    # (the shoulder line slopes from the neck and rounds over the deltoids into the arms, one convex curve with no
    # shelf: the wool does not stand out flat at the height of the shoulder and stop, it falls over the round of
    # the arm and ends a little below it at the sides)
    M_HEM = 47.6
    MANTLE_RINGS = [(53.4, 2.3, 2.3, 0.35), (52.4, 3.7, 3.1, 0.35), (51.3, 5.55, 3.8, 0.3), (50.3, 7.0, 4.25, 0.3),
                    (49.3, 8.0, 4.55, 0.3), (48.4, 8.55, 4.7, 0.3), (M_HEM, 8.75, 4.75, 0.3)]

    MANTLE_BACK = [(53.4, 2.65), (52.4, 3.3), (51.3, 3.25), (50.3, 3.3), (49.3, 3.4), (M_HEM, 3.6)]

    def back_y(z):     # the inside of the mantle down the middle of her back, just clear of it
        B = MANTLE_BACK
        for (z0, y0), (z1, y1) in zip(B, B[1:]):
            if z1 <= z <= z0:
                return lerp(y0, y1, (z0 - z) / (z0 - z1))
        return B[0][1] if z > B[0][0] else B[-1][1]

    # the leaf brooch that pins the mantle shut, a little to her right of the middle (phi round from the front,
    # z the height): the wool is gathered in toward it, and below it the two edges part in a short notch, so the
    # brooch closes the mantle over the jerkin rather than sitting on top of the cloth
    B_PHI, B_Z = 0.1, 50.55

    def brooch_pull(z, phi):
        """(inward pull, lift of the hem) at a point of the mantle, from the brooch."""
        dx, dz = (phi - B_PHI) * 6.6, z - B_Z
        r = math.hypot(dx, dz)
        # a shallow dish where the pin draws the cloth in, and round it the wool gathered into tight pleats that
        # fan out from the pin: pinched to a crease in each valley, rounded over each crest, and running further
        # down than up, as the gathered cloth falls from it
        dish = 0.2 * math.exp(-(r / 1.5) ** 2)
        ang = math.atan2(dz, dx)
        re = math.hypot(dx, dz * (0.55 if dz < 0 else 2.6))
        ring = min(1.0, max(0.0, (re - 0.75) / 0.5)) * math.exp(-max(0.0, re - 1.3) / 1.5)
        crease = 0.24 * ring * pinch(9 * ang + 0.5 + 0.35 * math.sin(2 * ang))
        # the parting: from just under the brooch to the hem the two edges draw apart in a narrow, rounded V
        n = math.exp(-((phi - B_PHI) / 0.075) ** 2)
        s = min(1.0, max(0.0, (49.6 - z) / (49.6 - M_HEM)))
        return dish - crease, n * s * 1.35

    def pinch(t):
        """A row of gathered folds: -1 at a sharp, pinched valley, rising to a rounded crest at +1."""
        return 2 * math.sqrt(math.sin(t / 2) ** 2 + 0.02) - 1

    def mantle_pt(z, phi, off=0.0):
        # one smooth curve through the rings (an ease between each pair of rings would flatten the cloth at every
        # ring and stack it into padded bands, like quilting)
        R = MANTLE_RINGS
        rx, ry, cy = R[-1][1:]
        for i in range(len(R) - 1):
            a_, b_ = R[i], R[i + 1]
            if b_[0] <= z <= a_[0]:
                t = (a_[0] - z) / (a_[0] - b_[0])
                p0, p3 = R[max(0, i - 1)], R[min(len(R) - 1, i + 2)]
                t2, t3 = t * t, t * t * t
                rx, ry, cy = (0.5 * (2 * q1 + (q2 - q0) * t + (2 * q0 - 5 * q1 + 4 * q2 - q3) * t2 + (3 * q1 - q0 - 3 * q2 + q3) * t3)
                              for q0, q1, q2, q3 in zip(p0[1:], a_[1:], b_[1:], p3[1:]))
                break
        # across the back the wool is drawn in close over the shoulder blades, so the hair can hang straight past it
        w = max(0.0, -math.cos(phi)) ** 1.5
        ry = lerp(ry, max(0.0, back_y(z) - cy), w)
        f = min(1.0, max(0.0, (51.4 - z) / (51.4 - M_HEM))) ** 1.3
        # the drape: below the tops of the shoulders the wool hangs on the body under it (the chest, the shoulder
        # blades and the round of each arm) a little way off it, falling a touch looser toward the hem, instead of
        # standing out round her as a stiff box; above, it keeps the rings' shape over the neck and shoulder line
        xr, yr = -rx * math.sin(phi), -ry * math.cos(phi)
        a = math.atan2(-xr, -yr)
        rr = math.hypot(xr, yr)
        # (close over the round of each arm, a little looser over the chest and back: standing well off the
        # arms the wool squares the shoulders off into a wide shelf)
        side = math.sin(a) ** 2
        gap = lerp(0.25 + 0.4 * f, 0.1 + 0.12 * f, side)
        hull = body_hull(z, a, cy) + gap
        dw = min(1.0, max(0.0, (53.2 - z) / 1.6))
        dw = dw * dw * (3 - 2 * dw)
        rd = max(lerp(rr, hull, dw), hull) if dw > 0 else rr
        rx, ry = rx * rd / rr, ry * rd / rr
        # the folds: a few deep ones, and finer ruffles between them that grow toward the hem, all pinched at the
        # valleys (gathered cloth, not padded), each wandering a little off true as it runs down
        fold = (0.62 * pinch(6 * phi + 0.4 + 0.3 * math.sin(2.1 * z)) +
                0.38 * f * pinch(14 * phi + 1.3 + 0.5 * z)) * (1 - 0.8 * w)
        pull, lift = brooch_pull(z, phi)
        k = 1 + 0.085 * f * fold * (1 - 0.6 * side) - pull / max(ry, 1.0)
        # (the hem drops at the sides, falling over the round of the arm and ending below it, so the wool caps
        # the shoulder instead of stopping at its corner)
        hem = (-2.0 * math.sin(phi) ** 4 - 0.5 * math.cos(phi)) * min(1.0, max(0.0, (49.6 - z) / 1.7))
        return -(rx + off) * k * math.sin(phi), cy - (ry + off) * k * math.cos(phi), hem + 0.55 * f * fold + lift

    MN = 200     # (fine enough round for the pleats at the brooch)
    mz = [53.4 - 0.2 * i for i in range(29)] + [M_HEM]
    mantle = mesh_obj('Mantle', loft_fn(mz, mantle_pt, n=MN, cap_top=False, cap_bot=False), M_MANTLE, body, solid=0.45)
    mesh_obj('Mail trim', loft_fn([M_HEM + 0.4, M_HEM + 0.1, M_HEM - 0.15], lambda z, p: mantle_pt(z, p, 0.08), n=MN, cap_top=False, cap_bot=False),
             M_MAIL, body, solid=0.25, subdiv=0)
    mesh_obj('Collar', loft([(54.0, 2.0, 2.05, 0, 0.4), (52.8, 2.5, 2.55, 0, 0.35)], cap_top=False, cap_bot=False), M_SHIRT, body, solid=0.3)
    # the cut edges of the wool (the hem and the neck), fulled up a little fluffier: 1 at the edge, fading over a
    # pixel or so (read by the mantle's material)
    edge = mantle.data.attributes.new('edge', 'FLOAT', 'POINT')
    edge.data.foreach_set('value', [max(min(1.0, max(0.0, (M_HEM + 0.85 - mz[i // MN]) / 0.85)),
                                        min(1.0, max(0.0, (mz[i // MN] - 52.9) / 0.5)))
                                    for i in range(len(mantle.data.vertices))])
    place_brooch(mantle, mantle_pt, B_PHI, B_Z, body)
    fuzz_shells(mantle, MATERIALS[M_MANTLE]['ramp'])

    # the hair's pieces are fused once the mantle is on, so the fall can be kept clear of it
    fuse_hair(*HAIR_PARTS)

    # ---- the rig: one bone per limb segment, each following its pivot (the pivots carry the pose; see pose()).
    # Each limb is a single mesh weighted across its joints, so elbows and knees bend in one smooth piece.
    build_rig(body)

    # ---- arms: one sleeve from a rounded cap at the shoulder to the wrist, then the shirt cuff and the hand
    # (a fuller upper arm and forearm than before: the shoulder cap, the swell of the upper arm, a slight narrowing
    # at the elbow, the forearm's muscle just below it, tapering to the wrist)
    # (the cap is fuller and rounder, and the deltoid swells it outward: see deltoid)
    ARM = ARM_PROFILE

    def sleeve(side, s, x, w):
        zs = [1.55 - 0.22 * i for i in range(int((1.55 + UPPER + FORE + 1.1) / 0.22) + 1)]

        def sleeve_w(co, w=w):
            # (the sleeve hangs from the forearm: its end does not tip with the hand)
            d = w(co)
            d[f'Forearm.{side}'] = d.get(f'Forearm.{side}', 0.0) + d.pop(f'Hand.{side}', 0.0)
            return d

        def arm_pt(z, phi, s=s):
            r = _profile(ARM, z) + deltoid(z, phi, s) + sleeve_folds(z, phi, s)[0]
            # the elbow: a little point at the back of the joint
            back = 0.25 * math.exp(-((z + UPPER) / 0.9) ** 2) * max(0.0, -math.cos(phi))
            return -r * math.sin(phi), -(r * 1.03 + back * 0) * math.cos(phi) + back
        so = mesh_obj(f'Sleeve.{side}', loft_fn(zs, arm_pt, n=48, cap_bot=False), M_SLEEVE, loc=(x, 0.2, SH_Z), skin=sleeve_w, solid=0.12)
        crease_attr(so, lambda co, s=s: sleeve_folds(co.z, math.atan2(-co.x, -co.y), s)[1])
        # the shirt's cuff, a narrow band showing just below the end of the sleeve
        mesh_obj(f'Shirt cuff.{side}', loft([(-UPPER - FORE - 0.7, 1.42, 1.46), (-UPPER - FORE - 1.5, 1.42, 1.46)], cap_top=False, cap_bot=False),
                 M_SHIRT, loc=(x, 0.2, SH_Z), subdiv=0, skin=sleeve_w, solid=0.1)
    build_arms(chest, sleeve)   # (the hands are the base figure's)

    # ---- legs: one trouser leg from the hip to the ankle with a rounded kneecap and a patch over it, soft boots
    # with a folded cuff
    LEG = [(2.0, 3.1), (-1.0, 3.0), (-6.0, 2.55), (-THIGH + 1.2, 2.05), (-THIGH, 2.0), (-THIGH - 1.5, 2.0), (-THIGH - 4.0, 2.05),
           (-THIGH - SHIN, 1.35)]

    def leg(side, s, x, w):
        # (the leg ends inside the boot, well above the ankle, so it never shows through the heel as the foot bends)
        zs = [2.0 - 0.2 * i for i in range(int((2.0 + THIGH + SHIN - 3.5) / 0.2) + 1)]

        def leg_pt(z, phi, folds=False, s=s):
            r = _profile(LEG, z) + (leg_folds(z, phi, s)[0] if folds else 0.0)
            cap = 0.4 * math.exp(-((z + THIGH + 0.3) / 1.1) ** 2) * max(0.0, math.cos(phi)) ** 2   # kneecap
            return -r * math.sin(phi), -(r * 1.04 + cap) * math.cos(phi)
        kz = HIP_Z - THIGH
        lo = mesh_obj(f'Leg.{side}', loft_fn(zs, lambda z, p: leg_pt(z, p, True), n=56), M_LEGS, loc=(x, 0, HIP_Z), skin=w)
        crease_attr(lo, lambda co, s=s: leg_folds(co.z, math.atan2(-co.x, -co.y), s)[1])
        # the knee patch: a piece of heavier cloth sewn onto the trouser leg over the knee, cut to a rounded
        # square and laid on the leg's own surface (so it bends with the knee instead of standing off it as a
        # plate), with its edge turned under and a running stitch round it
        def patch_pt(u, v):
            zc, hz = -THIGH + 0.1, 1.5
            hw = 0.78 * (1 - abs(v) ** 4) ** 0.25
            z, phi = zc + v * hz, u * hw
            px, py = leg_pt(z, phi)
            r = math.hypot(px, py)
            return px / r * (r + 0.08), py / r * (r + 0.08), z
        pb, P = bmesh.new(), []
        NU, NV = 14, 12
        for j in range(NV + 1):
            P.append([pb.verts.new(patch_pt(-1 + 2 * i / NU, -1 + 2 * j / NV)) for i in range(NU + 1)])
        for j in range(NV):
            for i in range(NU):
                pb.faces.new((P[j][i], P[j][i + 1], P[j + 1][i + 1], P[j + 1][i]))
        bmesh.ops.recalc_face_normals(pb, faces=pb.faces)
        mesh_obj(f'Knee patch.{side}', pb, M_KNEE, loc=(x, 0, HIP_Z), subdiv=1, skin=w, solid=0.07)
        kst = []
        for k in range(34):
            a_ = 2 * math.pi * k / 34
            u, v = 0.8 * math.cos(a_), 0.8 * math.sin(a_)
            v = math.copysign(abs(v) ** 0.7 * 0.8 ** 0.3, v)
            px, py, pz = patch_pt(u, v)
            r = math.hypot(px, py)
            kst.append((px / r * (r + 0.1), py / r * (r + 0.1), pz))
        mesh_obj(f'Knee stitching.{side}', dots(kst, 0.13), M_STITCH, loc=(x, 0, HIP_Z), subdiv=0, skin=w)
        # the boot: one piece of soft leather from under the knee down over the ankle and into the foot, so the
        # foot grows out of the shaft (it bends at the ankle with the foot's bone, rather than being a separate
        # shoe set under a tube); the shaft is a little slack, with a few soft wrinkles where it bunches over the
        # instep and above the ankle
        S = SHIN

        def wrinkle(z):
            return 0.06 * math.sin(4.2 * z) * math.exp(-((z + S - 1.6) / 1.6) ** 2)
        boot = loft([(-2.0, 2.2, 2.3), (-3.5, 2.2, 2.32), (-6.5, 2.08, 2.18), (-S + 2.6, 1.72 + wrinkle(-S + 2.6), 1.84),
                     (-S + 1.6, 1.62 + wrinkle(-S + 1.6), 1.76), (-S + 0.6, 1.66, 1.84, 0, -0.05),
                     (-S - 0.5, 1.78, 2.0, 0, -0.3), (-S - 1.4, 1.84, 2.8, 0, -1.0), (-S - 2.2, 1.86, 3.3, 0, -1.6),
                     (-S - 2.85, 1.8, 3.4, 0, -1.7)], n=32, cap_top=False)
        mesh_obj(f'Boot.{side}', boot, M_BOOTS, loc=(x, 0, kz), skin=w, solid=0.12)
        # its top folded down into a cuff: the leather turns over at the rim (a rolled edge, not a cut one) and
        # hangs down over the shaft, dipping at the back
        cuff = loft([(-1.9, 2.25, 2.35), (-1.7, 2.5, 2.6), (-1.95, 2.66, 2.76), (-2.9, 2.64, 2.74),
                     (-3.9, 2.58, 2.68, 0, 0, lambda p: 0.45 * math.cos(p))], n=32, cap_top=False, cap_bot=False)
        mesh_obj(f'Boot cuff.{side}', cuff, M_CUFF, loc=(x, 0, kz), subdiv=1, skin=w, solid=0.15)
        # a seam stitched up the back of the boot, from the heel to the cuff
        bst = []
        for z in [-4.4 - 1.0 * i for i in range(int(SHIN - 4.4))]:
            t = (-2.2 - z) / (SHIN - 2.7)
            ry = lerp(2.32, 1.88, min(1.0, max(0.0, t)))
            bst.append((0.0, ry + 0.1, z))
        mesh_obj(f'Boot stitching.{side}', dashes(bst, (0.24, 0.14, 0.55)), M_STITCH, loc=(x, 0, kz), subdiv=0, skin=w)
        az = kz - SHIN
        # the welt: a strip of darker leather sewn round where the upper meets the sole, standing just proud of
        # both, so the foot ends in a clean, defined edge; then the sole under it (held by enough rings that the
        # smoothing keeps its flat, square-edged shape instead of rounding it into a puck)
        mesh_obj(f'Welt.{side}', loft([(-2.25, 1.86, 3.36, 0, -1.65), (-2.4, 1.97, 3.5, 0, -1.7), (-2.8, 2.0, 3.53, 0, -1.7),
                                       (-2.86, 1.92, 3.45, 0, -1.7)], n=40, cap_top=False, cap_bot=False),
                 M_BELT, loc=(x, 0, az), subdiv=1, skin=w, solid=0.06)
        welt = []
        for k in range(44):
            ph = 2 * math.pi * k / 44
            welt.append((-2.0 * math.sin(ph), -1.7 - 3.52 * math.cos(ph), -2.6))
        mesh_obj(f'Welt stitching.{side}', dots(welt, 0.08), M_STITCH, loc=(x, 0, az), subdiv=0, skin=w)
        mesh_obj(f'Sole.{side}', loft([(-2.8, 1.93, 3.47, 0, -1.7), (-2.86, 1.95, 3.5, 0, -1.7), (-3.3, 1.93, 3.47, 0, -1.7),
                                       (-ANKLE_Z, 1.85, 3.38, 0, -1.7)], n=40), M_SOLE, loc=(x, 0, az), subdiv=1, skin=w)
    build_legs(pelvis, leg)
    # the mantle rides on the arms where it lies over them; then the rig, the camera and light, and the keyframes
    return finish_figure(lambda: mantle_on_arms(['Mantle', 'Mail trim']))


if __name__ == '__main__':
    run(build, 'ranger', HERE)
