"""The Rogue: the base female figure (blender/base/female.py) in her hair and clothes, rendered to a pixel-art
sprite sheet.

Dark hair pulled back from the face into a high ponytail, bound with a leather wrap. A slate wool cowl wrapped
about the neck and falling over the shoulders in a few deep swags. A plum linen shirt whose full sleeves gather
into laced leather bracers. An over-bust bodice of dark leather, its panels shaped by stitched seams and closed
down the front with five buckled straps, bound in darker leather at the top and along the pointed hem. Charcoal
twill leggings, a hip belt slung low across them with a pouch on the right hip and a dagger worn across the small
of the back, and thigh-high boots with a slanted, bound top, two buckled straps and a stitched welt.

The body, head, face, eyes, skin, hands, rig, walk cycle and the sprite pipeline are the base figure's.

Run it with Blender's Python (either works):
    blender -b -P blender/rogue/rogue.py -- [--build-only] [--out DIR]
    python blender/rogue/rogue.py [--build-only] [--out DIR]      (with `pip install bpy OpenEXR`)

It writes rogue.blend and rogue-blender.png into blender/rogue/ (or DIR).
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'base'))
import bpy  # noqa: E402
import bmesh  # noqa: E402
from mathutils import Vector, Matrix  # noqa: E402
from female import *  # noqa: E402,F401,F403  (the base figure: shapes, materials, rig, walk and sprite pipeline)
from female import _profile, _gs, _torso_dims  # noqa: E402,F401

# ---------------------------------------------------------------- her materials
# (her hair, brows and eyes are her own colours on the base figure's materials)
M_HAIR = mat_def(2, 'hair', ramp(0x46312b, 0.1), 0.24, surface='hair')
M_BROW = mat_def(17, 'brow', ramp(0x3a2a24, 0.08), 0.0, flat=1)
# a warm hazel-brown iris, ringed dark, amber round the pupil
M_EYE = mat_def(16, 'iris', [[34, 26, 20], [62, 44, 30], [92, 66, 40], [124, 94, 54], [156, 128, 74]], 0.0, flat='painted')
M_COWL = mat_def(30, 'cowl', ramp(0x4a515c, 0.1), 0.2, surface='wool', closeup='wool', soft_edge=True)
M_SHIRT = mat_def(31, 'shirt', ramp(0x5f4a5f, 0.1), 0.06, closeup='canvas', form=True)
M_BODICE = mat_def(32, 'bodice', ramp(0x4e3b33, 0.1), 0.24, surface='leather', closeup='leather', form=True)
M_BIND = mat_def(33, 'binding', ramp(0x2f2624, 0.08), 0.18, surface='leather')
M_BRACER = mat_def(34, 'bracers', ramp(0x6b4a36, 0.1), 0.22, surface='leather', closeup='suede', form=True)
M_LEGS = mat_def(35, 'leggings', ramp(0x403f48, 0.1), 0.05, surface='twill', closeup='cloth', form=True)
M_BOOTS = mat_def(36, 'boots', ramp(0x3e302a, 0.1), 0.24, surface='leather', closeup='leather')
M_SOLE = mat_def(37, 'sole', ramp(0x221c19, 0.08), 0.0)
M_BELT = mat_def(38, 'belt', ramp(0x5c3d2b, 0.1), 0.2, surface='leather')
M_STEEL = mat_def(39, 'steel', ramp(0x9a978f, 0.09), 0.0)
M_STITCH = mat_def(40, 'stitching', ramp(0xa08c76, 0.08), 0.0, flat=2)
M_LACE = mat_def(41, 'laces', ramp(0x2c2420, 0.08), 0.0)


def smoothstep(e0, e1, x):
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def pinch(t):
    """A row of gathered folds: -1 at a sharp, pinched valley, rising to a rounded crest at +1."""
    return 2 * math.sqrt(math.sin(t / 2) ** 2 + 0.02) - 1


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def creases(table, z, pm, norm):
    """(offset, crease) of short single grooves, each (height, angle, half-width, slant, curl, depth), as the
    ranger's sleeve creases: each runs part of the way round, slants and curls, so they read as crumpling."""
    off, cr = 0.0, 0.0
    for z0, p0, wd, slant, curl, depth in table:
        dp = wrap(pm - p0)
        win = math.exp(-(dp / wd) ** 4)
        if win < 1e-3:
            continue
        d = z - (z0 + slant * dp + curl * dp * dp)
        groove = math.exp(-(d / 0.3) ** 2)
        ridge = math.exp(-((d - 0.42) / 0.38) ** 2) + 0.6 * math.exp(-((d + 0.45) / 0.38) ** 2)
        off += depth * win * (0.55 * ridge - groove)
        cr += win * groove * min(1.0, depth / norm)
    return off, cr


def frame_bm(w, h, bar, depth):
    """An open buckle frame in the XZ plane (facing -Y), w across and h tall, its bars bar thick."""
    bm = bmesh.new()
    for (cx, cz, sx_, sz_) in ((0, h / 2 - bar / 2, w, bar), (0, -h / 2 + bar / 2, w, bar),
                               (-w / 2 + bar / 2, 0, bar, h), (w / 2 - bar / 2, 0, bar, h)):
        r_ = bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(sx_, depth, sz_), verts=r_['verts'])
        bmesh.ops.translate(bm, vec=(cx, 0, cz), verts=r_['verts'])
    return bm


def turned(bm, phi):
    """Turn a piece built facing -Y to face out at angle phi round the body (phi 0 front, toward her right)."""
    bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(-phi, 3, 'Z'), verts=bm.verts)
    return bm


def sweep(path, radius, n=24, ends=True):
    """A tube along a path of points: radius(i, angle) -> (r across, r through) at each point, with its cross
    section held level across (its width along X) so a flattened tube lies flat against what it hangs on."""
    bm = bmesh.new()
    rings = []
    N = len(path)
    for i, p in enumerate(path):
        t = (path[min(i + 1, N - 1)] - path[max(i - 1, 0)]).normalized()
        ax = Vector((1, 0, 0))
        ax = (ax - t * ax.dot(t)).normalized()
        dp = t.cross(ax).normalized()
        ring = []
        for k in range(n):
            a = 2 * math.pi * k / n
            rx, ry = radius(i, a)
            ring.append(bm.verts.new(p + ax * (rx * math.cos(a)) + dp * (ry * math.sin(a))))
        rings.append(ring)
    for ra, rb in zip(rings, rings[1:]):
        for k in range(n):
            bm.faces.new((ra[k], ra[(k + 1) % n], rb[(k + 1) % n], rb[k]))
    if ends:
        for ring, p in ((rings[0], path[0]), (rings[-1], path[-1])):
            c = bm.verts.new(p)
            for k in range(n):
                bm.faces.new((ring[k], ring[(k + 1) % n], c))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def catmull(pts, samples):
    """A smooth curve through the points (Catmull-Rom), sampled evenly in its parameter."""
    P = [Vector(p) for p in pts]
    out = []
    segs = len(P) - 1
    for i in range(samples):
        u = i / (samples - 1) * segs
        k = min(int(u), segs - 1)
        t = u - k
        p0, p1, p2, p3 = P[max(k - 1, 0)], P[k], P[k + 1], P[min(k + 2, segs)]
        out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    return out


def body_tree(dg, names=None, off=Vector((0, 0, 0))):
    """A BVH of the figure's meshes carried by the body (and the head's shape), in coordinates less off."""
    from mathutils.bvhtree import BVHTree
    vs, ps = [], []
    for o in bpy.data.objects:
        if o.type != 'MESH' or o.parent is None:
            continue
        if not (o.parent.name == 'Body' or o.name in (names or ())):
            continue
        me = o.evaluated_get(dg).to_mesh()
        base = len(vs)
        vs += [o.matrix_world @ v.co - off for v in me.vertices]
        ps += [[base + i for i in p.vertices] for p in me.polygons]
        o.evaluated_get(dg).to_mesh_clear()
    return BVHTree.FromPolygons(vs, ps)


# ---------------------------------------------------------------- the bodice
# An over-bust bodice: high at the front (its top runs up under the cowl), cut low under the arms so the shirt
# shows there, and coming to a point at the front below the waist. (phi round from the front toward her right.)
B_OFF = 0.32       # the leather stands this far off the shirt
B_ZT, B_ZB = 48.2, 32.4


def bodice_top(phi):
    c = math.cos(phi)
    return 46.4 + 1.6 * max(0.0, c) ** 1.5 + 0.9 * max(0.0, -c) ** 1.5


def bodice_hem(phi):
    c = math.cos(phi)
    point = math.exp(-(wrap(phi) / 0.42) ** 2)
    return 34.3 - 0.4 * max(0.0, -c) - 1.8 * point


STRAPS_Z = (45.7, 43.3, 40.9, 38.5, 36.1)
SEAMS_X = (2.3, 4.6)      # the princess seams: over the bust, and down the side front


def bodice_form(z, phi):
    """(radial offset, crease) of the bodice's leather: the seams that shape its panels, the front opening the
    straps close, short wrinkles at the sides of the waist, and the leather drawn in under each strap."""
    off, cr = 0.0, 0.0
    sx, c = math.sin(phi), math.cos(phi)
    rx = _torso_dims(z)[0] + B_OFF
    # the seams: a fine groove where two panels are sewn together, each panel a little rounded between them
    for x0 in SEAMS_X:
        for sgn in (-1, 1):
            a = math.asin(max(-1.0, min(1.0, sgn * x0 / rx)))
            d = wrap(phi + a)     # (phi of the point where x = -rx sin(phi) = sgn * x0, i.e. phi = -a)
            seam = _gs(d, 0.0, 0.035) * max(0.0, c) ** 0.5
            off -= 0.06 * seam
            cr += 0.45 * seam
    # the front opening: the two front edges meet down the middle with a dark line of shade between them
    gap = _gs(wrap(phi), 0.0, 0.03) * max(0.0, c)
    off -= 0.14 * gap
    cr += 0.9 * gap
    # the waist: a few short wrinkles at the sides where the leather bends as she moves
    side = abs(sx) ** 3
    band = _gs(z, 40.2, 1.3) * side
    v = fold_ridge(3.2 * (z - 40.2) + 2.0 * c)
    off += 0.09 * band * v
    cr += 0.6 * band * max(0.0, -v)
    # under each strap the leather is pulled in a little, and puckers either side of it
    for zs in STRAPS_Z:
        k = _gs(wrap(phi), 0.0, 0.45)
        off -= 0.05 * k * _gs(z, zs, 0.35)
        cr += 0.35 * k * (_gs(z, zs + 0.45, 0.12) + _gs(z, zs - 0.45, 0.12))
    return off, min(1.0, cr)


def bod_xy(z, phi, lift=0.0):
    off = bodice_form(z, phi)[0]
    return torso_point(z, phi, B_OFF + off + lift)


def bodice_pt(zr, phi):
    v = (B_ZT - zr) / (B_ZT - B_ZB)
    z = lerp(bodice_top(phi), bodice_hem(phi), v)
    x, y = bod_xy(z, phi)
    return x, y, z - zr


def band_between(top, bot, lift, rows=3, n=160):
    """A band laid on the bodice from top(phi) down to bot(phi), standing lift off it."""
    zs = [1.0 - i / (rows - 1) for i in range(rows)]

    def pt(u, phi):
        z = lerp(bot(phi), top(phi), u)
        x, y = bod_xy(z, phi, lift)
        return x, y, z - u
    return loft_fn(zs, pt, n=n, cap_top=False, cap_bot=False)


def strap_strip(z, p0, p1, half, lift, nseg=10, round_end=False):
    """A strap across the bodice's front at height z, from phi p0 to p1, laid on the leather."""
    bm = bmesh.new()
    rows = []
    for i in range(nseg + 1):
        ph = lerp(p0, p1, i / nseg)
        hh = half
        if round_end:
            e = max(0.0, (i / nseg - 0.7) / 0.3)
            hh = half * math.sqrt(max(0.05, 1 - e * e))
        x, y = bod_xy(z, ph, lift)
        rows.append([bm.verts.new((x, y, z + hh)), bm.verts.new((x, y, z - hh))])
    for a, b in zip(rows, rows[1:]):
        bm.faces.new((a[0], b[0], b[1], a[1]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


# ---------------------------------------------------------------- the cowl
# A wide tube of soft wool wrapped once about the neck and falling over the shoulders: rolled at the top, lying
# on the slope of the shoulders and on the round of each arm (its hem drops at the sides, so it caps the shoulder
# instead of stopping at its corner), and falling at the front in a few deep, slightly off-centre U-swags.
CW_ZT, CW_ZB = 54.6, 47.7


def cowl_top(phi):
    return 53.9 + 0.7 * (0.5 - 0.5 * math.cos(phi))


def cowl_hem(phi):
    c = math.cos(phi - 0.06)
    return 49.4 - 1.65 * max(0.0, c) ** 1.6 - 0.75 * max(0.0, -math.cos(phi)) ** 1.6


def cowl_form(z, phi, f):
    """(radial offset, crease, lift of the hem) of the cowl at height z, angle phi, f from 0 at its top to 1 at
    its hem."""
    off, cr, lift = 0.0, 0.0, 0.0
    c = math.cos(phi)
    front = max(0.0, math.cos(phi - 0.06)) ** 2
    # the swags: U-shaped folds across the front, rising toward the shoulders, deepest toward the hem
    u = z + 2.1 * wrap(phi - 0.06) ** 2      # (phi wrapped: just left of the front it is near 2 pi)
    v = fold_ridge(2 * math.pi * (u - 48.6) / 2.5 + 0.25 * math.sin(1.7 * wrap(phi)))
    k = front * smoothstep(0.3, 0.75, f) * (1.0 - smoothstep(51.6, 52.6, z))
    off += 0.38 * k * v
    cr += 0.8 * k * max(0.0, -v)
    lift += 0.3 * k * v
    # over the shoulders and down the back, a few soft folds running down from the neck
    side = math.sin(phi) ** 2
    w = fold_ridge(5.0 * phi + 0.7 + 0.4 * math.sin(2.3 * z))
    kk = (0.35 * side + 0.6 * max(0.0, -c) ** 2) * smoothstep(0.1, 0.7, f)
    off += 0.12 * kk * w
    cr += 0.5 * kk * max(0.0, -w)
    # the roll at the top: the wool turned over on itself, a fuller round band about the neck
    roll = _gs(f, 0.05, 0.09)
    off += 0.28 * roll
    cr += 0.4 * _gs(f, 0.17, 0.05)
    return off, min(1.0, cr), lift


def cowl_pt(zr, phi, extra=0.0):
    f = (CW_ZT - zr) / (CW_ZT - CW_ZB)
    z = lerp(cowl_top(phi), cowl_hem(phi), f)
    cy = 0.15
    # the neck: an ellipse round it; below the base of the neck the wool lies on the shoulders and the body under
    # it (see body_hull), a little looser toward the hem and closer over the round of each arm
    rx, ry = 2.55, 2.6
    xr, yr = -rx * math.sin(phi), -ry * math.cos(phi)
    a = math.atan2(-xr, -yr)
    rr = math.hypot(xr, yr)
    side = math.sin(a) ** 2
    gap = lerp(0.3 + 0.35 * f, 0.12 + 0.12 * f, side)
    hull = body_hull(round(z * 20) / 20, a, cy) + gap     # (heights rounded, so the hull's cache is shared)
    dw = smoothstep(0.0, 1.0, (53.3 - z) / 1.5)
    rd = max(lerp(rr, hull, dw), hull) if dw > 0 else max(rr, hull - 0.2)
    off, _, lift = cowl_form(z, phi, f)
    r = rd + off + extra
    return -r * math.sin(a), cy - r * math.cos(a), z - zr + lift * f


# ---------------------------------------------------------------- the hair
# Pulled back from the face over the skull, sleek, and gathered into a high ponytail at the back of the crown,
# bound with a leather wrap; the tail swells out from the wrap and falls down her back over the cowl, breaking at
# its end into a few locks. (Head-local coordinates: the chin at the origin, the face toward -Y.)
HC = Vector((0.0, 0.5, 4.8))     # the skull's centre, for laying the hair on it
TIE_EL = 0.55                    # how high up the back of the head the tail is tied (radians above level)
TIE = {}
FUSE = True       # melt the hair's pieces into one (views.py --set FUSE=False shows them apart)

# the hairline: its height round the head, as an elevation seen from HC (|angle| from the front: the forehead,
# a little higher at the temples, down in front of the ears, behind them, and across the nape)
# (combed back over the ears, which the base head does not model: down past where they would be, so the side of
# the head behind the jaw does not show as bare skin)
HAIRLINE = smooth_curve([(0.0, 0.53), (0.3, 0.5), (0.7, 0.34), (1.0, 0.1), (1.25, -0.2), (1.5, -0.52), (1.8, -0.76),
                         (2.1, -0.82), (2.6, -0.82), (math.pi, -0.8)])


def _dir(th, el):
    return Vector((-math.sin(th) * math.cos(el), -math.cos(th) * math.cos(el), math.sin(el)))


def tie_axis():
    return _dir(math.pi, TIE_EL)


def cap_flow(co):
    """Pattern coordinates for the hair over the skull: across the strands is the angle round the line from the
    skull's centre to the tie (so every strand runs from the hairline back to the tie), along them the arc from
    the front."""
    A = tie_axis()
    d = (Vector(co) - HC).normalized()
    e1, e2 = Vector((1, 0, 0)), A.cross(Vector((1, 0, 0)))
    across = math.atan2(d.dot(e1), -d.dot(e2))       # (its one break falls under the tail, at the nape)
    along = math.acos(max(-1.0, min(1.0, d.dot(A))))
    return across * 3.6, (math.pi - along) * 4.0


def hair_rest(co):
    """Pattern coordinates for the whole head of hair (see hair_flow): the cap's, then down the tail."""
    co = Vector(co)
    path, lens = TIE['path'], TIE['lens']
    L0 = math.pi * 4.0
    _, k, dist = TIE['kd'].find(co)
    if lens[k] > -0.3 and dist < tail_radius(k / (len(path) - 1))[0] * 1.35 + 0.3:
        p = path[k]
        t = (path[min(k + 1, len(path) - 1)] - path[max(k - 1, 0)]).normalized()
        ax = Vector((1, 0, 0))
        ax = (ax - t * ax.dot(t)).normalized()
        dp = t.cross(ax)
        q = co - p
        across = math.atan2(q.dot(ax), -q.dot(dp)) * 2.4
        return Vector((across, 0.0, -0.3 * (L0 + lens[k])))
    across, along = cap_flow(co)
    return Vector((across, 0.0, -0.3 * along))


def hair_cap(head_tree):
    """The hair over the skull, laid on it from the hairline to the crown: a grid round the head (azimuth) and up
    it (from the hairline to the crown), each point cast onto the skull and set just off it. Sleek at the front
    and sides, a little fuller toward the crown, and gathered up into a mound where it is tied, so the tail grows
    out of the head instead of sitting on it; a few broad combed sections run back to the tie."""
    NU, NV = 128, 44
    MOUND = 0.6
    A = tie_axis()
    bm = bmesh.new()
    rows, weights = [], []

    miss = []

    def skull(d):
        hit = head_tree.ray_cast(HC + d * 20.0, -d, 20.0)
        if hit[0] is None:
            miss.append(tuple(d))
        return (hit[0] - HC).length if hit[0] is not None else 4.0

    tie_r = skull(A)
    # (just outside the mound: the skull, the cap's lift, the mound and the hair's thickness)
    TIE['point'] = HC + A * (tie_r + 0.02 + 0.12 + 0.12 * A.z + MOUND + 0.42 + 0.12)
    for j in range(NV + 1):
        t = j / NV
        row, wr = [], []
        for k in range(NU):
            th = 2 * math.pi * k / NU
            el0 = HAIRLINE(abs(wrap(th)))
            el = lerp(el0, 1.5, t)
            d = _dir(th, el)
            rs = skull(d)
            # thickness: nothing at the hairline, easing up over the first finger's breadth, fuller toward the
            # crown; the gathered mound round the tie
            edge = smoothstep(0.0, 0.14, t)
            ang = math.acos(max(-1.0, min(1.0, d.dot(A))))
            mound = MOUND * math.exp(-(ang / 0.36) ** 2)
            combed = 0.04 * math.sin(5.0 * cap_flow(HC + d)[0] / 3.6 + 0.6) * edge * (1.0 - mound)
            r = rs + 0.02 + edge * (0.12 + 0.12 * max(0.0, d.z)) + mound * edge + combed
            row.append(bm.verts.new(HC + d * r))
            wr.append(0.25 + 0.75 * edge)
        rows.append(row)
        weights.append(wr)
    pole = bm.verts.new(HC + Vector((0, 0, 1)) * (skull(Vector((0, 0, 1))) + 0.26))
    for a, b in zip(rows, rows[1:]):
        for k in range(NU):
            bm.faces.new((a[k], a[(k + 1) % NU], b[(k + 1) % NU], b[k]))
    for k in range(NU):
        bm.faces.new((rows[-1][k], rows[-1][(k + 1) % NU], pole))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    f0 = bm.faces[len(bm.faces) // 2]
    flip = f0.normal.dot(f0.calc_center_median() - HC) < 0
    if flip:      # (outward, so its thickness is laid on outside)
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    side = [(v.co - HC).length for v in rows[0][NU // 4 - 2:NU // 4 + 3]]
    print(f'hair cap: {len(miss)} rays missed the skull; normals flipped: {flip}; hairline at the side '
          f'{[round(v.co.z, 2) for v in rows[0][NU // 4 - 2:NU // 4 + 3]]} at r {[round(x, 2) for x in side]}', flush=True)
    bm.verts.index_update()
    w = [x for r in weights for x in r] + [1.0]
    return bm, w


def ponytail_path(tree):
    """The line down the middle of the tail: out from the tie, arcing over and down, then lying on what it falls
    over (the back of the head, the cowl, her back), cast for from behind; hair spans hollows rather than sinking
    into them, so the line is eased and never let in closer than the hair's own thickness."""
    A = tie_axis()
    T = TIE['point']
    knots = [T - A * 0.5, T + A * 0.7, T + A * 1.2 + Vector((0, 0.55, -1.2)), Vector((0.0, T.y + 1.1, T.z - 3.4)),
             Vector((0.05, 4.4, 0.3)), Vector((0.2, 4.4, -4.2)), Vector((0.35, 4.2, -8.6)), Vector((0.5, 4.0, -12.4))]
    path = catmull(knots, 110)
    N = len(path)

    def support(x, z):
        for k_ in range(6):
            hit = tree.ray_cast(Vector((x * 0.9 ** k_, 40.0, z)), Vector((0, -1, 0)), 60.0)
            if hit[0] is not None:
                return hit[0].y
        return -99.0

    lo = []
    for i, p in enumerate(path):
        s = i / (N - 1)
        lo.append(support(p.x, p.z) + tail_radius(s)[1] + 0.22 if p.z < T.z - 1.5 else -99.0)
    ys = [max(p.y, l) for p, l in zip(path, lo)]
    for _ in range(40):
        ys = [ys[0]] + [max(lo[i], 0.5 * ys[i] + 0.25 * (ys[i - 1] + ys[i + 1])) for i in range(1, N - 1)] + [max(lo[-1], ys[-1])]
    path = [Vector((p.x, y, p.z)) for p, y in zip(path, ys)]
    lens = [0.0]
    for a, b in zip(path, path[1:]):
        lens.append(lens[-1] + (b - a).length)
    # (lengths from the tie, not from the start inside the mound)
    k_tie = min(range(N), key=lambda i: (path[i] - T).length)
    lens = [L - lens[k_tie] for L in lens]
    return path, lens


TAIL_R = [(0.0, 0.72), (0.05, 0.66), (0.08, 0.66), (0.14, 1.3), (0.24, 1.75), (0.5, 1.7), (0.75, 1.4), (0.9, 0.9),
          (0.97, 0.4), (1.0, 0.08)]


def tail_radius(s):
    r = _profile([(-a, b) for a, b in TAIL_R], -s)
    return r, 0.62 * r


def build_hair(head, head_tree):
    """The cap, the ponytail and its end locks, fused into one surface (as the ranger's hair is), then the tie."""
    from mathutils.kdtree import KDTree
    cap, w = hair_cap(head_tree)
    cap_o = mesh_obj('Hair cap', cap, M_HAIR, head, solid=0.42, rest=lambda co: Vector((*cap_flow(co), 0.0)).xzy)
    vg = cap_o.vertex_groups.new(name='thick')
    for i_, w_ in enumerate(w):
        vg.add([i_], w_, 'REPLACE')
    cap_o.modifiers['thickness'].vertex_group = 'thick'
    cap_o.modifiers['thickness'].thickness_vertex_group = 0.3
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    tree = body_tree(dg, ('Head shape', 'Hair cap'), Vector((0, 0, HEAD_Z)))
    path, lens = ponytail_path(tree)
    TIE['path'], TIE['lens'] = path, lens
    kd = KDTree(len(path))
    for i, p in enumerate(path):
        kd.insert(p, i)
    kd.balance()
    TIE['kd'] = kd
    N = len(path)

    def tail_r(i, a):
        s = i / (N - 1)
        r, rd = tail_radius(s)
        # a few broad clumps, twisting a little down the tail and parting deeper toward the end
        clump = 1.0 + (0.05 + 0.16 * s) * math.cos(3 * a + 1.6 * s + 0.4) + 0.04 * math.cos(7 * a + 0.5)
        return r * clump, rd * clump
    mesh_obj('Hair tail', sweep(path, tail_r, n=28), M_HAIR, head, subdiv=0)
    # the end breaks into locks: two finer ones peel off either side, each to its own point
    names = ['Hair cap', 'Hair tail']
    for j, (k0, side, reach, drop) in enumerate(((0.68, 1, 0.9, 1.1), (0.74, -1, 0.75, 0.4))):
        i0 = int(k0 * (N - 1))
        p0 = path[i0]
        end = path[-1] + Vector((side * reach, -0.15, drop))
        mid_ = p0.lerp(end, 0.5) + Vector((side * 0.35, 0.1, 0.0))
        lp = catmull([p0, mid_, end], 30)
        mesh_obj(f'Hair lock {j}', sweep(lp, lambda i, a, n_=len(lp): ((0.62 * (1 - i / (n_ - 1)) ** 0.8 + 0.04),
                                                                        0.5 * (1 - i / (n_ - 1)) ** 0.8 + 0.04), n=16),
                 M_HAIR, head, subdiv=0)
        names.append(f'Hair lock {j}')
    if not FUSE:          # (for checking the pieces before they are melted together)
        return None
    hair = fuse_parts(head, names)
    # the tie: a leather thong wound round the root of the tail, in three turns
    k_tie = min(range(N), key=lambda i: (path[i] - TIE['point']).length)
    seg = path[max(0, k_tie - 4):k_tie + 6]

    def tie_r(i, a, n_=len(seg)):
        u = i / (n_ - 1)
        r = 0.8 + 0.07 * abs(math.sin(3 * math.pi * u)) - 0.06 * (2 * u - 1) ** 8
        return r, r * 0.85
    mesh_obj('Hair tie', sweep(seg, tie_r, n=24), M_BELT, head, subdiv=1)
    return hair


def fuse_parts(head, names, voxel=0.065):
    """Melt the hair's pieces into one surface: joined, remeshed as a single volume and relaxed, so every join
    is a soft fillet rather than a seam (see the ranger's fuse_hair)."""
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
    # where the tail leaves the mound the hair is relaxed once more, so the two read as one sweep
    T = TIE['point']
    root = [v for v in bm.verts if (v.co - T).length < 2.2]
    for _ in range(8):
        bmesh.ops.smooth_laplacian_vert(bm, verts=root, lambda_factor=0.5, lambda_border=0.0,
                                        use_x=True, use_y=True, use_z=True, preserve_volume=True)
    hair = mesh_obj('Hair', bm, M_HAIR, head, subdiv=0, rest=hair_rest)
    strand_colours(hair)
    return hair


# ---------------------------------------------------------------- the sleeves and bracers
# The linen sleeve, full on the upper arm (local height from the shoulder joint, radius), puffed where it is
# gathered into the bracer just below the elbow, and slim inside the bracer down to the wrist.
SLEEVE_R = [(1.6, 0.3), (1.35, 1.35), (0.85, 2.0), (0.0, 2.25), (-2.0, 2.32), (-5.5, 2.2), (-UPPER + 0.6, 2.0),
            (-UPPER - 0.55, 2.12), (-UPPER - 1.55, 1.6), (-UPPER - 4.0, 1.45), (-UPPER - FORE + 0.4, 1.3),
            (-UPPER - FORE - 0.4, 1.28)]
BR_TOP, BR_BOT = -UPPER - 1.3, -UPPER - FORE - 0.6
BRACER_R = [(BR_TOP, 1.95), (BR_TOP - 0.6, 1.9), (-UPPER - 5.0, 1.74), (-UPPER - FORE + 0.8, 1.56), (BR_BOT, 1.64)]

R_SLEEVE_CREASES = [
    # the armpit: two creases fanning down and out from the pit
    (-1.4, -1.05, 0.6, -0.9, 0.0, 0.12), (-2.4, -2.2, 0.55, 0.8, 0.0, 0.1),
    # the inside of the elbow, where the full sleeve folds as the arm bends
    (-UPPER + 0.7, 0.25, 0.85, 0.35, 0.5, 0.14), (-UPPER - 0.2, -0.3, 0.9, -0.3, 0.6, 0.12),
]


def sleeve_form(z, phi, s):
    """(radial offset, crease) of the linen sleeve: the armpit and elbow creases, long folds hanging down the
    full upper arm from the shoulder seam, and tight pleats where it is gathered into the top of the bracer."""
    pm = s * phi
    off, cr = creases(R_SLEEVE_CREASES, z, pm, 0.12)
    k = smoothstep(0.2, -1.2, z) * smoothstep(-UPPER - 0.3, -UPPER + 1.5, z)
    v = fold_ridge(5.0 * pm + 0.4 + 0.22 * z)
    off += 0.12 * k * v
    cr += 0.55 * k * max(0.0, -v)
    g = _gs(z, -UPPER - 0.95, 0.55)
    v2 = pinch(10.0 * pm + 1.1 + 0.3 * math.sin(3 * pm))
    off += 0.14 * g * v2
    cr += 0.8 * g * max(0.0, -v2)
    return off, min(1.0, cr)


def bracer_form(z, phi, s):
    """(radial offset, crease) of the bracer's leather: a few soft wrinkles where the wrist bends, and the leather
    drawn in a little either side of the lacing."""
    pm = s * phi
    off, cr = 0.0, 0.0
    k = _gs(z, -UPPER - FORE + 1.4, 1.0) * max(0.0, math.cos(pm - 0.4)) ** 2
    v = fold_ridge(4.0 * (z + UPPER + FORE) + pm)
    off += 0.06 * k * v
    cr += 0.5 * k * max(0.0, -v)
    lace = _gs(wrap(pm - math.pi / 2), 0.0, 0.12)
    off -= 0.07 * lace
    cr += 0.5 * lace
    return off, min(1.0, cr)


def bracer_pt(z, phi, s, lift=0.0):
    r = _profile(BRACER_R, z) + bracer_form(z, phi, s)[0] + lift
    return -r * math.sin(phi), -r * 1.03 * math.cos(phi)


# ---------------------------------------------------------------- the legs
# The leggings (local height from the hip joint, radius): fitted, a little slimmer than the ranger's trousers
LEG_R = [(2.0, 3.0), (-1.0, 2.9), (-6.0, 2.42), (-THIGH + 1.2, 1.95), (-THIGH, 1.9), (-THIGH - 1.5, 1.88),
         (-THIGH - 4.0, 1.9), (-THIGH - SHIN, 1.3)]
R_LEG_CREASES = [
    (-1.6, -0.45, 0.55, 1.0, 0.0, 0.06), (-2.7, -0.6, 0.5, 1.1, 0.0, 0.05), (-4.2, 1.6, 0.5, -0.6, 0.0, 0.04),
    (-THIGH + 0.9, 3.0, 0.7, 0.3, -0.3, 0.12), (-THIGH + 0.1, -2.9, 0.9, -0.25, -0.35, 0.14),
]
BOOT_TOP = 5.2      # the boot's top, above the knee (height from the knee)


def leg_form(z, phi, s):
    pm = s * phi
    off, cr = creases(R_LEG_CREASES, z, pm, 0.1)
    return off, min(1.0, cr)


def boot_rings(s):
    """The thigh boot's rings (height from the knee, half-width, depth, centre x, centre y, hem): drawn over the
    leggings (their radius plus the leather), round the knee, down the calf, a little slack and wrinkled above
    the ankle, then the foot (as the ranger's boot)."""
    S = SHIN
    R = lambda zk: _profile(LEG_R, zk - THIGH)   # noqa: E731

    def top(p):           # the top slants: higher on the outside of the leg
        return 0.8 * math.sin(s * p) - 0.15

    def wr(z):
        return 0.07 * math.sin(4.4 * z + 0.6) * math.exp(-((z + S - 1.8) / 1.3) ** 2)
    return [(BOOT_TOP, R(BOOT_TOP) + 0.34, R(BOOT_TOP) * 1.04 + 0.36, 0, 0, top),
            (BOOT_TOP - 0.5, R(BOOT_TOP - 0.5) + 0.33, R(BOOT_TOP - 0.5) * 1.04 + 0.35, 0, 0, lambda p: top(p) * 0.92),
            (3.0, R(3.0) + 0.28, R(3.0) * 1.04 + 0.3), (1.2, R(1.2) + 0.28, R(1.2) * 1.04 + 0.36, 0, -0.08),
            (0.0, 2.17, 2.5, 0, -0.24), (-1.4, 2.14, 2.26), (-3.6, 2.2, 2.32), (-6.5, 2.06, 2.16),
            (-S + 2.6, 1.74 + wr(-S + 2.6), 1.86), (-S + 1.9, 1.67 + wr(-S + 1.9), 1.8), (-S + 1.2, 1.7 + wr(-S + 1.2), 1.86),
            (-S + 0.6, 1.66, 1.84, 0, -0.05), (-S - 0.5, 1.78, 2.0, 0, -0.3), (-S - 1.4, 1.84, 2.8, 0, -1.0),
            (-S - 2.2, 1.86, 3.3, 0, -1.6), (-S - 2.85, 1.8, 3.4, 0, -1.7)]


def ring_r(rings, z):
    """(half-width, depth, centre y) of a list of loft rings at height z, eased between them."""
    for a, b in zip(rings, rings[1:]):
        if b[0] <= z <= a[0]:
            t = (a[0] - z) / (a[0] - b[0])
            cy0 = a[4] if len(a) > 4 else 0.0
            cy1 = b[4] if len(b) > 4 else 0.0
            return lerp(a[1], b[1], t), lerp(a[2], b[2], t), lerp(cy0, cy1, t)
    r = rings[0] if z > rings[0][0] else rings[-1]
    return r[1], r[2], r[4] if len(r) > 4 else 0.0


# ---------------------------------------------------------------- the rogue
def build():
    root, body, pelvis, chest = begin_figure('Rogue')

    # ---- torso: the body in its linen shirt; the leggings' seat over the hips, under the bodice
    torso_mesh('Shirt', M_SHIRT, body)
    mesh_obj('Leggings seat', loft_fn([37.0 - 0.5 * i for i in range(18)], lambda z, p: torso_point(z, p, 0.12, bust=0.0)),
             M_LEGS, body)

    # ---- the bodice, its binding, seams and buckled straps
    bz = [B_ZT - (B_ZT - B_ZB) * i / 59 for i in range(60)]
    bo = mesh_obj('Bodice', loft_fn(bz, bodice_pt, n=192, cap_top=False, cap_bot=False), M_BODICE, body, solid=0.3)
    crease_attr(bo, lambda co: bodice_form(co.z, math.atan2(-co.x, -co.y))[1])
    mesh_obj('Bodice top binding', band_between(bodice_top, lambda p: bodice_top(p) - 0.55, 0.14), M_BIND, body,
             solid=0.3, subdiv=0)
    mesh_obj('Bodice hem binding', band_between(lambda p: bodice_hem(p) + 0.5, bodice_hem, 0.14), M_BIND, body,
             solid=0.3, subdiv=0)
    # stitching: a running stitch inside each binding, and down both sides of every seam
    st = []
    for k in range(120):
        phi = 2 * math.pi * k / 120
        for z in (bodice_top(phi) - 0.78, bodice_hem(phi) + 0.72):
            x, y = bod_xy(z, phi, 0.17)
            st.append((x, y, z))
    for x0 in SEAMS_X:
        for sgn in (-1, 1):
            for z in [47.6 - 0.62 * i for i in range(24)]:
                if not (bodice_hem(0.4) + 0.9 < z < bodice_top(0.4) - 0.9):
                    continue
                rx = _torso_dims(z)[0] + B_OFF
                a = -math.asin(max(-1.0, min(1.0, sgn * x0 / rx)))
                for da in (-0.05, 0.05):
                    x, y = bod_xy(z, a + da, 0.15)
                    st.append((x, y, z))
    mesh_obj('Bodice stitching', dots(st, 0.2), M_STITCH, body, subdiv=0)
    # the straps that close the front: each crosses from her right panel to a small steel buckle on her left,
    # and its tail runs on past the buckle to a rounded end
    for i, z in enumerate(STRAPS_Z):
        mesh_obj(f'Strap {i}', strap_strip(z, 0.34, -0.13, 0.27, 0.14), M_BELT, body, subdiv=0, solid=0.14)
        mesh_obj(f'Strap tail {i}', strap_strip(z, -0.2, -0.36, 0.25, 0.3, nseg=5, round_end=True), M_BELT, body,
                 subdiv=0, solid=0.12)
        pb = -0.17
        x, y = bod_xy(z, pb, 0.36)
        mesh_obj(f'Strap buckle {i}', turned(frame_bm(0.72, 0.78, 0.15, 0.16), pb), M_STEEL, body, loc=(x, y, z), subdiv=0)
        x, y = bod_xy(z, pb + 0.02, 0.46)
        mesh_obj(f'Strap prong {i}', turned(box(0.5, 0.1, 0.1), pb), M_STEEL, body, loc=(x, y, z), subdiv=0)
        # the stitch where the strap is sewn to her right panel
        sx_ = [(*bod_xy(z + dz, 0.31, 0.32), z + dz) for dz in (-0.14, 0.14)]
        mesh_obj(f'Strap stitching {i}', dots(sx_, 0.16), M_STITCH, body, subdiv=0)

    # ---- the hip belt, slung low across the leggings, higher over her right hip; a round steel buckle a little to
    # her left of the middle, its end hanging free below it; a pouch on her right hip; a dagger across her back
    def belt_z(p):
        return 31.75 + 0.55 * math.sin(p)
    BELT_OFF = 0.5

    def belt_pt(zr, p):
        z = belt_z(p) + zr
        x, y = torso_point(z, p, BELT_OFF, bust=0.0)
        return x, y, belt_z(p)
    mesh_obj('Hip belt', loft_fn([0.55, 0.0, -0.55], belt_pt, n=128, cap_top=False, cap_bot=False), M_BELT, body,
             solid=0.3, subdiv=0)
    bst = []
    for k in range(96):
        p = 2 * math.pi * k / 96
        if abs(wrap(p + 0.32)) < 0.14:
            continue
        for dz in (0.38, -0.38):
            x, y = torso_point(belt_z(p) + dz, p, BELT_OFF + 0.31, bust=0.0)
            bst.append((x, y, belt_z(p) + dz))
    mesh_obj('Belt stitching', dots(bst, 0.16), M_STITCH, body, subdiv=0)
    PB = -0.32
    zb = belt_z(PB)
    x, y = torso_point(zb, PB, BELT_OFF + 0.45, bust=0.0)
    ring = loft([(0.16, 0.95, 0.95), (0.0, 1.05, 1.05), (-0.16, 0.95, 0.95)], n=28, cap_top=False, cap_bot=False)
    inner = loft([(0.16, 0.66, 0.66), (0.0, 0.58, 0.58), (-0.16, 0.66, 0.66)], n=28, cap_top=False, cap_bot=False)
    for bm_ in (ring, inner):
        bmesh.ops.rotate(bm_, cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi / 2, 3, 'X'), verts=bm_.verts)
        turned(bm_, PB)
    mesh_obj('Belt ring', ring, M_STEEL, body, loc=(x, y, zb), subdiv=1)
    mesh_obj('Belt ring inside', inner, M_STEEL, body, loc=(x, y, zb), subdiv=1)
    # its end, drawn through the ring, hangs down over her left thigh, cut round at the tip
    tail = bmesh.new()
    rows_ = []
    for i in range(9):
        t = i / 8
        hh = 0.5 * math.sqrt(max(0.05, 1 - max(0.0, (t - 0.7) / 0.3) ** 2))
        z = zb - 0.3 - 2.3 * t
        xx, yy = torso_point(max(z, 28.6), PB - 0.09, BELT_OFF + 0.3 + 0.12 * t, bust=0.0)
        rows_.append([tail.verts.new((xx + hh * math.cos(PB), yy - hh * math.sin(PB), z)),
                      tail.verts.new((xx - hh * math.cos(PB), yy + hh * math.sin(PB), z))])
    for a_, b_ in zip(rows_, rows_[1:]):
        tail.faces.new((a_[0], b_[0], b_[1], a_[1]))
    bmesh.ops.recalc_face_normals(tail, faces=tail.faces)
    mesh_obj('Belt end', tail, M_BELT, body, subdiv=0, solid=0.16)
    # the pouch: a soft leather bag hanging from the belt, its flap closed with a steel stud
    PP = 0.86
    zp = belt_z(PP) - 1.25
    x, y = torso_point(zp, PP, BELT_OFF + 1.0, bust=0.0)
    pouch = box(2.0, 1.15, 1.9)
    for v in pouch.verts:
        v.co.y *= 1.0 - 0.2 * (v.co.z + 0.95)       # (fuller at the bottom, where it holds its weight)
    mesh_obj('Pouch', turned(pouch, PP), M_BELT, body, loc=(x, y, zp), subdiv=2)
    xf, yf = torso_point(zp + 0.35, PP, BELT_OFF + 1.62, bust=0.0)
    flap = box(2.1, 0.22, 1.15)
    for v in flap.verts:
        v.co.x *= 1.0 - 0.18 * max(0.0, -v.co.z) / 0.575      # (cut narrower toward its rounded lip)
    mesh_obj('Pouch flap', turned(flap, PP), M_BIND, body, loc=(xf, yf, zp + 0.35), subdiv=2)
    xs, ys = torso_point(zp - 0.1, PP, BELT_OFF + 1.78, bust=0.0)
    mesh_obj('Pouch stud', ellipsoid(0.18, 0.12, 0.18, 10, 6), M_STEEL, body, loc=(xs, ys, zp - 0.1), subdiv=0)
    # the dagger across the small of her back, tucked through the belt, its grip to her right hand
    dz_ = 32.9
    # (clear of her back and of the belt where it crosses it, the hips' fullness included)
    yb = max(torso_point(dz_ + dz, math.pi + a, BELT_OFF + 0.3, bust=0.0)[1] for dz in (-0.7, 0.0, 0.7)
             for a in [-0.75 + 0.05 * i for i in range(31)]) + 0.34
    tilt = Matrix.Rotation(0.12, 3, 'Y') @ Matrix(((0, 0, -1), (0, 1, 0), (1, 0, 0)))
    sheath = loft([(0.0, 0.58, 0.3), (-0.4, 0.6, 0.3), (-3.0, 0.5, 0.27), (-5.2, 0.3, 0.2), (-5.9, 0.12, 0.12)], n=20)
    bmesh.ops.transform(sheath, matrix=tilt.to_4x4(), verts=sheath.verts)
    mesh_obj('Dagger sheath', sheath, M_BIND, body, loc=(-2.4, yb, dz_), subdiv=1)
    for nm, rings, mid, sub in (('Sheath locket', [(0.1, 0.66, 0.36), (-0.55, 0.66, 0.36)], M_STEEL, 0),
                                ('Sheath chape', [(-5.0, 0.36, 0.25), (-5.75, 0.18, 0.17), (-6.05, 0.05, 0.05)], M_STEEL, 1),
                                ('Sheath loop', [(-2.0, 0.6, 0.38), (-2.6, 0.6, 0.38)], M_BELT, 0)):
        bm_ = loft(rings, n=20)
        bmesh.ops.transform(bm_, matrix=tilt.to_4x4(), verts=bm_.verts)
        mesh_obj(nm, bm_, mid, body, loc=(-2.4, yb, dz_), subdiv=sub)
    hilt = tilt     # (the grip runs on along the same line, out past the throat toward her right)
    for nm, bm_, mid in (('Dagger guard', box(1.55, 0.4, 0.26), M_STEEL),
                         ('Dagger grip', loft([(1.95, 0.24, 0.24), (1.0, 0.27, 0.27), (0.2, 0.24, 0.24)], n=12), M_LACE),
                         ('Dagger pommel', ellipsoid(0.36, 0.32, 0.36, 12, 8), M_STEEL)):
        if nm == 'Dagger pommel':
            bmesh.ops.translate(bm_, vec=(0, 0, 2.2), verts=bm_.verts)
        bmesh.ops.transform(bm_, matrix=hilt.to_4x4(), verts=bm_.verts)
        mesh_obj(nm, bm_, mid, body, loc=(-2.4 - 0.15, yb, dz_ + 0.02), subdiv=1)

    # ---- neck and head (the base figure's)
    head, head_tree = build_head(body)

    # ---- the cowl, with its cut edges fulled up a little and the loose fibres standing off it (see fuzz_shells)
    CN = 192
    cz = [CW_ZT - (CW_ZT - CW_ZB) * i / 34 for i in range(35)]
    cowl = mesh_obj('Cowl', loft_fn(cz, cowl_pt, n=CN, cap_top=False, cap_bot=False), M_COWL, body, solid=0.42)
    edge = cowl.data.attributes.new('edge', 'FLOAT', 'POINT')
    ev = []
    for i in range(len(cowl.data.vertices)):
        f = (CW_ZT - cz[(i // CN) % len(cz)]) / (CW_ZT - CW_ZB)
        ev.append(max(smoothstep(0.88, 1.0, f), 1.0 - smoothstep(0.0, 0.07, f)))
    edge.data.foreach_set('value', ev)
    fuzz_shells(cowl, MATERIALS[M_COWL]['ramp'])

    # ---- the hair: pulled back into a ponytail, laid over the cowl
    build_hair(head, head_tree)

    # ---- the rig (the base figure's)
    build_rig(body)

    # ---- arms: the full linen sleeve gathered into a laced bracer below the elbow, then the hand
    def sleeve(side, s, x, w):
        def fore(co, w=w):
            # (the sleeve's end and the bracer hang from the forearm: they do not tip with the hand)
            d = w(co)
            d[f'Forearm.{side}'] = d.get(f'Forearm.{side}', 0.0) + d.pop(f'Hand.{side}', 0.0)
            return d
        zs = [1.55 - 0.22 * i for i in range(int((1.55 + UPPER + FORE + 0.4) / 0.22) + 1)]

        def arm_pt(z, phi):
            r = _profile(SLEEVE_R, z) + deltoid(z, phi, s) + sleeve_form(z, phi, s)[0]
            return -r * math.sin(phi), -r * 1.03 * math.cos(phi)
        so = mesh_obj(f'Sleeve.{side}', loft_fn(zs, arm_pt, n=48, cap_bot=False), M_SHIRT, loc=(x, 0.2, SH_Z), skin=fore,
                      solid=0.12)
        crease_attr(so, lambda co: sleeve_form(co.z, math.atan2(-co.x, -co.y), s)[1])
        bzs = [BR_TOP - 0.2 * i for i in range(int((BR_TOP - BR_BOT) / 0.2) + 1)] + [BR_BOT]
        bo_ = mesh_obj(f'Bracer.{side}', loft_fn(bzs, lambda z, p: bracer_pt(z, p, s), n=56, cap_top=False, cap_bot=False),
                       M_BRACER, loc=(x, 0.2, SH_Z), skin=fore, solid=0.16)
        crease_attr(bo_, lambda co: bracer_form(co.z, math.atan2(-co.x, -co.y), s)[1])
        # its edges bound in darker leather, with a running stitch inside each
        for nm, z0, z1 in (('top', BR_TOP + 0.05, BR_TOP - 0.42), ('bottom', BR_BOT + 0.42, BR_BOT - 0.05)):
            mesh_obj(f'Bracer {nm} binding.{side}', loft_fn([z0, (z0 + z1) / 2, z1], lambda z, p: bracer_pt(z, p, s, 0.12), n=56,
                                                           cap_top=False, cap_bot=False),
                     M_BIND, loc=(x, 0.2, SH_Z), skin=fore, solid=0.12, subdiv=0)
        bst_ = []
        for k in range(40):
            p = 2 * math.pi * k / 40
            if abs(wrap(s * p - math.pi / 2)) < 0.4:
                continue
            for z in (BR_TOP - 0.62, BR_BOT + 0.62):
                bst_.append((*bracer_pt(z, p, s, 0.18), z))
        mesh_obj(f'Bracer stitching.{side}', dots(bst_, 0.15), M_STITCH, loc=(x, 0.2, SH_Z), skin=fore, subdiv=0)
        # the lacing down the outside of the forearm: steel eyelets either side of the opening, the thong crossing
        # between them
        holes = []
        zl = [BR_TOP - 0.9 - 1.05 * i for i in range(7)]
        for z in zl:
            holes.append([Vector((*bracer_pt(z, s * (math.pi / 2 + dp), s, 0.1), z)) for dp in (-0.27, 0.27)])
        mesh_obj(f'Bracer eyelets.{side}', dots([p for pair in holes for p in pair], 0.2), M_STEEL, loc=(x, 0.2, SH_Z),
                 skin=fore, subdiv=0)
        lace = bmesh.new()
        for a_, b_ in zip(holes, holes[1:]):
            for p, q in ((a_[0], b_[1]), (a_[1], b_[0])):
                mid_ = (p + q) / 2
                n_ = Vector((mid_.x, mid_.y, 0)).normalized()
                seg = tube([p, mid_ + n_ * 0.12, q], [0.08, 0.09, 0.08], flat=1.0, n=6)
                tmp = bpy.data.meshes.new('lace')
                seg.to_mesh(tmp)
                seg.free()
                lace.from_mesh(tmp)
                bpy.data.meshes.remove(tmp)
        mesh_obj(f'Bracer lace.{side}', lace, M_LACE, loc=(x, 0.2, SH_Z), skin=fore, subdiv=0)
    build_arms(chest, sleeve)   # (the hands are the base figure's)

    # ---- legs: the leggings, and the thigh boots over them
    def leg(side, s, x, w):
        # (the leggings end inside the boot, well above the ankle, so they never show through as the foot bends)
        zs = [2.0 - 0.2 * i for i in range(int((2.0 + THIGH + SHIN - 3.5) / 0.2) + 1)]

        def leg_pt(z, phi):
            r = _profile(LEG_R, z) + leg_form(z, phi, s)[0]
            cap = 0.4 * math.exp(-((z + THIGH + 0.3) / 1.1) ** 2) * max(0.0, math.cos(phi)) ** 2   # kneecap
            return -r * math.sin(phi), -(r * 1.04 + cap) * math.cos(phi)
        kz = HIP_Z - THIGH
        lo = mesh_obj(f'Leg.{side}', loft_fn(zs, leg_pt, n=56), M_LEGS, loc=(x, 0, HIP_Z), skin=w)
        crease_attr(lo, lambda co: leg_form(co.z, math.atan2(-co.x, -co.y), s)[1])
        rings = boot_rings(s)
        mesh_obj(f'Boot.{side}', loft(rings, n=40, cap_top=False), M_BOOTS, loc=(x, 0, kz), skin=w, solid=0.12)
        # the top bound in darker leather, with a running stitch under it
        top = rings[0][5]
        r0, r1 = rings[0], rings[1]
        mesh_obj(f'Boot binding.{side}', loft([(r0[0] + 0.05, r0[1] + 0.12, r0[2] + 0.12, 0, 0, top),
                                               (r0[0] - 0.45, r1[1] + 0.13, r1[2] + 0.13, 0, 0, top)], n=40, cap_top=False, cap_bot=False),
                 M_BIND, loc=(x, 0, kz), subdiv=1, skin=w, solid=0.12)
        bst = []
        for k in range(36):
            p = 2 * math.pi * k / 36
            z = r0[0] - 0.75 + top(p)
            rx, ry, cy = ring_r(rings, r0[0] - 0.75)
            bst.append((-(rx + 0.06) * math.sin(p), cy - (ry + 0.06) * math.cos(p), z))
        # and a seam up the back of the boot, from the heel to the top
        for z in [-SHIN + 1.0 + 0.95 * i for i in range(int((SHIN + BOOT_TOP - 2.2) / 0.95))]:
            rx, ry, cy = ring_r(rings, z)
            bst.append((0.0, cy + ry + 0.06, z))
        mesh_obj(f'Boot stitching.{side}', dots(bst, 0.15), M_STITCH, loc=(x, 0, kz), subdiv=0, skin=w)
        # two buckled straps: round the top of the calf, and round the ankle; each buckle on the outside of the leg
        for j, zs_ in enumerate((-3.6, -SHIN + 1.25)):
            rx, ry, cy = ring_r(rings, zs_)
            mesh_obj(f'Boot strap {j}.{side}', loft([(zs_ + 0.3, rx + 0.12, ry + 0.12, 0, cy), (zs_, rx + 0.15, ry + 0.15, 0, cy),
                                                     (zs_ - 0.3, rx + 0.12, ry + 0.12, 0, cy)], n=40, cap_top=False, cap_bot=False),
                     M_BELT, loc=(x, 0, kz), subdiv=1, skin=w, solid=0.12)
            fb = frame_bm(0.72, 0.8, 0.15, 0.16)
            turned(fb, s * math.pi / 2)
            mesh_obj(f'Boot buckle {j}.{side}', fb, M_STEEL, loc=(x - s * (rx + 0.32), 0.0, kz + zs_), subdiv=0, skin=w)
        az = kz - SHIN
        # the welt and the sole (as the ranger's boots), with a stitch round the welt and one across the vamp
        mesh_obj(f'Welt.{side}', loft([(-2.25, 1.86, 3.36, 0, -1.65), (-2.4, 1.97, 3.5, 0, -1.7), (-2.8, 2.0, 3.53, 0, -1.7),
                                       (-2.86, 1.92, 3.45, 0, -1.7)], n=40, cap_top=False, cap_bot=False),
                 M_BIND, loc=(x, 0, az), subdiv=1, skin=w, solid=0.06)
        welt = []
        for k in range(44):
            ph = 2 * math.pi * k / 44
            welt.append((-2.0 * math.sin(ph), -1.7 - 3.52 * math.cos(ph), -2.6))
        for k in range(15):
            ph = -0.95 + 1.9 * k / 14
            welt.append((-1.84 * 1.06 * math.sin(ph), -1.0 - 2.8 * 1.04 * math.cos(ph), -1.4))
        mesh_obj(f'Welt stitching.{side}', dots(welt, 0.09), M_STITCH, loc=(x, 0, az), subdiv=0, skin=w)
        mesh_obj(f'Sole.{side}', loft([(-2.8, 1.93, 3.47, 0, -1.7), (-2.86, 1.95, 3.5, 0, -1.7), (-3.3, 1.93, 3.47, 0, -1.7),
                                       (-ANKLE_Z, 1.85, 3.38, 0, -1.7)], n=40), M_SOLE, loc=(x, 0, az), subdiv=1, skin=w)
    build_legs(pelvis, leg)
    # the cowl rides on the arms where it lies over them; then the rig, the camera and light, and the keyframes
    return finish_figure(lambda: mantle_on_arms(['Cowl']))


if __name__ == '__main__':
    run(build, 'rogue', HERE)
