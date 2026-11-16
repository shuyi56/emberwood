"""The Warden: the base female figure (blender/base/female.py) in her hair and clothes, rendered to a pixel-art
sprite sheet.

Long, loose, wavy dark-brown hair parted in the middle, falling past the shoulders down her back, and a pale scar
across her right cheek. A cream tabard over a belted brown leather tunic: the tabard runs from the collar to just
above the knee, trimmed in red along its edges and its stepped hem, with a red yoke embroidered at the chest and a
red band down the middle. Under it a leather tunic, belted at the waist, whose skirt falls over the tops of brown
trousers, gambeson sleeves under buckled leather bracers, and tall buckled boots with a turned-down cuff. (No cloak
and no weapons for now.)

The body, head, face, eyes, skin, hands, rig, walk cycle and the sprite pipeline are the base figure's.

Run it with Blender's Python (either works):
    blender -b -P blender/warden/warden.py -- [--build-only] [--out DIR]
    python blender/warden/warden.py [--build-only] [--out DIR]      (with `pip install bpy OpenEXR`)

It writes warden.blend and warden-blender.png into blender/warden/ (or DIR).
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'base'))
import bpy  # noqa: E402
import bmesh  # noqa: E402
from mathutils import Vector, Matrix  # noqa: E402
import female  # noqa: E402
from female import *  # noqa: E402,F401,F403  (the base figure: shapes, materials, rig, walk and sprite pipeline)
from female import _profile, _gs, _torso_dims  # noqa: E402,F401

# ---------------------------------------------------------------- her materials
# (her hair, brows and eyes are her own colours on the base figure's materials)
M_HAIR = mat_def(2, 'hair', ramp(0x3f2b21, 0.1), 0.24, surface='hair')
M_BROW = mat_def(17, 'brow', ramp(0x2f221c, 0.08), 0.0, flat=1)
# a dark grey-brown iris
M_EYE = mat_def(16, 'iris', [[30, 28, 24], [58, 50, 38], [86, 74, 52], [114, 100, 70], [142, 130, 92]], 0.0, flat='painted')
M_TABARD = mat_def(30, 'tabard', ramp(0xd6cbb0, 0.09), 0.05, closeup='canvas', form=True)
M_TRIM = mat_def(31, 'embroidery', ramp(0x9b2f25, 0.09), 0.0, closeup='canvas')
M_TUNIC = mat_def(32, 'tunic', ramp(0x6b4731, 0.1), 0.24, surface='leather', closeup='leather', form=True)
M_BIND = mat_def(33, 'binding', ramp(0x38291f, 0.08), 0.18, surface='leather')
M_SLEEVE = mat_def(34, 'gambeson', ramp(0x4f3c2e, 0.1), 0.07, surface='twill', closeup='cloth', form=True)
M_BRACER = mat_def(35, 'bracers', ramp(0x7a5337, 0.1), 0.22, surface='leather', closeup='suede', form=True)
M_LEGS = mat_def(36, 'trousers', ramp(0x463d33, 0.1), 0.05, surface='twill', closeup='cloth', form=True)
M_BOOTS = mat_def(37, 'boots', ramp(0x6a4630, 0.1), 0.24, surface='leather', closeup='leather')
M_CUFF = mat_def(38, 'boot cuff', ramp(0x7c5639, 0.1), 0.2, surface='leather', closeup='suede')
M_SOLE = mat_def(39, 'sole', ramp(0x221c19, 0.08), 0.0)
M_BELT = mat_def(40, 'belt', ramp(0x4b3123, 0.1), 0.2, surface='leather')
M_GOLD = mat_def(41, 'gold', ramp(0xb08f4f, 0.09), 0.0, closeup='gold', shadow_on=(M_TABARD, M_BELT))
M_STEEL = mat_def(42, 'steel', ramp(0x9a978f, 0.09), 0.0)
M_STITCH = mat_def(43, 'stitching', ramp(0xa8927a, 0.08), 0.0, flat=2)
# the scar: a little paler and pinker than the skin round it
M_SCAR = mat_def(44, 'scar', [[178, 114, 100], [202, 142, 124], [216, 164, 144], [226, 182, 164], [236, 202, 186]], 0.0)


def smoothstep(e0, e1, x):
    """0 at e0, 1 at e1, eased between (e1 may lie below e0)."""
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def pinch(t):
    """A row of gathered folds: -1 at a sharp, pinched valley, rising to a rounded crest at +1."""
    return 2 * math.sqrt(math.sin(t / 2) ** 2 + 0.02) - 1


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


def grid(rows, cols, point):
    """A sheet of quads through point(i, j) for i down its rows and j across its columns."""
    bm = bmesh.new()
    V = [[bm.verts.new(point(i, j)) for j in range(cols)] for i in range(rows)]
    for i in range(rows - 1):
        for j in range(cols - 1):
            bm.faces.new((V[i][j], V[i][j + 1], V[i + 1][j + 1], V[i + 1][j]))
    return bm


def face_out(bm, toward=Vector((0, -1, 0)), axis_y=None):
    """Turn a sheet's faces so they face out: away from the body's axis (axis_y given: the axis is the vertical
    line through (0, axis_y)), or else toward `toward`."""
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    vote = 0.0
    for f in bm.faces:
        c = f.calc_center_median()
        d = Vector((c.x, c.y - axis_y, 0.0)) if axis_y is not None else toward
        vote += f.normal.dot(d) * f.calc_area()
    if vote < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    return bm


def strip(path, half, surface, lift, n_across=3):
    """A flat band laid on a surface: along the polyline path of (x, z) points, half wide either side of it (in
    the x-z plane), its y found by surface(x, z) and brought forward by lift (toward the camera, -y)."""
    P = [Vector((p[0], 0.0, p[1])) for p in path]
    N = len(P)
    bm = bmesh.new()
    rows = []
    for i, p in enumerate(P):
        t = (P[min(i + 1, N - 1)] - P[max(i - 1, 0)]).normalized()
        nrm = Vector((-t.z, 0.0, t.x))       # across the band, in the x-z plane
        row = []
        for k in range(n_across):
            q = p + nrm * (half * (2 * k / (n_across - 1) - 1))
            row.append(bm.verts.new((q.x, surface(q.x, q.z) - lift, q.z)))
        rows.append(row)
    for a, b in zip(rows, rows[1:]):
        for k in range(n_across - 1):
            bm.faces.new((a[k], a[k + 1], b[k + 1], b[k]))
    return face_out(bm)


def polyline(fn, t0, t1, n):
    return [fn(t0 + (t1 - t0) * i / (n - 1)) for i in range(n)]


# ---------------------------------------------------------------- what hangs from the hips: the thighs carry it
def skirt_weights(co):
    """For what hangs over the thighs (the tabard and the tunic's skirt): the trunk's bones at the
    top, handed over going down to the thighs, each side to its own leg (blended across the middle), so the cloth
    is carried forward by the leg that swings under it instead of the leg swinging through it."""
    w = trunk_weights(co)
    f = 0.92 * smoothstep(35.0, 28.5, co.z)
    if f <= 1e-4:
        return w
    r = 1.0 / (1.0 + math.exp(max(-40.0, min(40.0, co.x / 1.0))))     # her right is -X
    out = {k: v * (1.0 - f) for k, v in w.items()}
    out['Thigh.R'] = f * r
    out['Thigh.L'] = f * (1.0 - r)
    return out


# ---------------------------------------------------------------- the torso: tunic and tabard
TUNIC_OFF = 0.3          # the tunic's leather over the body (the body's own surface is the gambeson)
T_OFF = 0.62             # the tabard over the tunic
T_LOW = 1.12             # and over the tunic's skirt, below the waist
SK_OFF = 0.72            # the skirt stands a little further off the hips than the tunic does off the body: the
                         # tops of the thighs fill out under it


def front_at(z, x, off, bust=1.0):
    """The y of the body's front at (x, z), pushed out by off."""
    rx = _torso_dims(z)[0] + off
    phi = math.asin(max(-1.0, min(1.0, -x / rx)))
    return torso_point(z, phi, off, bust)[1]


def tab_hw(z):
    """The tabard's half-width down its length: narrow at the collar, broad over the chest, drawn in at the
    waist, widening a little to the hem."""
    return _profile([(52.6, 1.75), (51.0, 2.9), (48.5, 3.45), (45.0, 3.35), (42.0, 2.75), (39.4, 2.5), (33.0, 2.75),
                     (21.0, 3.15)], z)


def tab_hem(u):
    """The hem's height across the tabard (u -1 to 1): stepped, the middle a tongue hanging lower than the sides,
    its corners rounded."""
    return 22.6 - 1.1 * smoothstep(0.46, 0.3, abs(u))


def tab_y(x, z):
    """The tabard's surface at (x, z). Above the waist it lies on the tunic and spans the hollow between the
    breasts (cloth bridges it); below the hips it hangs, falling a little forward of them toward the hem, with
    two soft folds."""
    if z > 36.0:
        y = front_at(z, x, T_OFF)
        if abs(x) < 2.2:
            y = min(y, front_at(z, 2.2, T_OFF))
        return y
    y = front_at(max(z, 32.0), x, lerp(T_LOW, T_OFF, smoothstep(35.0, 38.0, z)), 0.0)
    if z < 32.0:
        y -= 0.12 * (32.0 - z)
    k = smoothstep(34.0, 26.0, z)
    return y + 0.16 * k * math.sin(1.9 * x + 0.8) + 0.05 * k * x * x / 4.0


def tab_folds(x, z):
    """The crease of the tabard's soft folds (for the 'crease' attribute)."""
    k = smoothstep(34.0, 26.0, z)
    return k * max(0.0, -math.sin(1.9 * x + 0.8)) * 0.5


def skirt_pt(z, phi):
    """The tunic's skirt: on the hips above their widest point, below them flaring a little; a few broad folds
    deepening toward the hem, the panels' tooled seams drawn into the leather."""
    off, cr = skirt_form(z, phi)
    if z >= 32.0:
        x, y = torso_point(z, phi, lerp(TUNIC_OFF, SK_OFF, smoothstep(37.0, 34.0, z)) + off, bust=0.0)
        return x, y, z
    flare = 0.09 * (32.0 - z) * (0.5 + 0.5 * max(0.0, math.cos(phi))) - 0.2 * smoothstep(0.0, -0.8, math.cos(phi)) * smoothstep(32.0, 29.0, z)
    x, y = torso_point(32.0, phi, SK_OFF + flare + off, bust=0.0)
    return x, y, z


def skirt_form(z, phi):
    k = smoothstep(33.0, 25.0, z)
    v = math.sin(5.0 * phi + 0.7 + 0.3 * math.sin(3 * phi))
    off = 0.2 * k * v
    cr = 0.6 * k * max(0.0, -v)
    # the tooled seams between its panels
    for p0 in [math.pi * (k + 0.5) / 6 for k in range(12)]:
        seam = _gs(wrap(phi - p0), 0.0, 0.03) * smoothstep(38.0, 36.0, z)
        off -= 0.05 * seam
        cr += 0.6 * seam
    return off, min(1.0, cr)


def skirt_hem(phi):
    return 26.4 + 0.5 * max(0.0, -math.cos(phi)) - 0.4 * math.cos(phi) ** 2


# ---------------------------------------------------------------- the hair
# Long and loose, parted in the middle: a cap over the skull from a natural hairline, the hair falling down either
# side of the face and over the ears, and the long fall down her back, all fused into one surface; then waves
# pressed into it, a few broad S-curves running across the locks. (Head-local coordinates: the chin at the origin,
# the face toward -Y.)
HC = Vector((0.0, 0.5, 4.8))     # the skull's centre, for laying the hair on it
HAIRLINE = smooth_curve([(0.0, 0.5), (0.2, 0.52), (0.48, 0.56), (0.72, 0.36), (1.0, 0.08), (1.25, -0.2), (1.5, -0.5),
                         (1.8, -0.7), (2.1, -0.78), (2.6, -0.8), (math.pi, -0.8)])


def _dir(th, el):
    return Vector((-math.sin(th) * math.cos(el), -math.cos(th) * math.cos(el), math.sin(el)))


def hair_cap(head_tree):
    """The hair over the skull, laid on it from the hairline to the crown (as the rogue's cap, without the tie): a
    grid round the head and up it, each point cast onto the skull and set just off it. Thin where it leaves the
    hairline, so the edge feathers onto the skin; close over the brow, fuller over the sides and back where the
    long hair falls from; a narrow parting down the middle with the hair lifting a little either side of it, and
    a few broad combed sections running down from the parting."""
    NU, NV = 128, 44
    bm = bmesh.new()
    rows, weights = [], []

    def skull(d):
        hit = head_tree.ray_cast(HC + d * 20.0, -d, 20.0)
        return (hit[0] - HC).length if hit[0] is not None else 4.0

    for j in range(NV + 1):
        t = j / NV
        row, wr = [], []
        for k in range(NU):
            th = 2 * math.pi * k / NU
            dp_ = wrap(th)
            near_part = math.exp(-(dp_ / 0.06) ** 2)
            el0 = HAIRLINE(abs(dp_)) + 0.04 * near_part
            el = lerp(el0, 1.5, t)
            d = _dir(th, el)
            rs = skull(d)
            edge = smoothstep(0.0, 0.2, t)
            front = 1.0 - smoothstep(0.15, 0.55, t)
            part = -0.13 * near_part * front * edge
            lift_ = smoothstep(0.0, 0.12, abs(dp_)) * math.exp(-((abs(dp_) - 0.45) / 0.6) ** 2)
            swell = 0.16 * lift_ * front * smoothstep(0.02, 0.25, t)
            # combed down from the parting: grooves along the lines the comb draws (round the head's front-to-back
            # axis), fading out behind, where the fall takes over
            psi = math.atan2(d.z, -d.y)
            combed = (0.05 * math.sin(10.0 * psi + 0.6) + 0.025 * math.sin(24.0 * psi + 1.3)) * edge * \
                (1.0 - smoothstep(1.9, 2.7, abs(dp_)))
            # fuller over the sides and the back, where the long hair falls from
            full = 0.34 * smoothstep(0.6, 2.2, abs(dp_))
            r = rs - 0.1 * (1.0 - edge) + 0.02 + edge * (0.14 + 0.12 * max(0.0, d.z) + full) + combed + part + swell
            row.append(bm.verts.new(HC + d * r))
            wr.append(edge)
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
    if f0.normal.dot(f0.calc_center_median() - HC) < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.verts.index_update()
    return bm, [x for r in weights for x in r] + [1.0]


def side_lock(s):
    """The hair falling down one side of the face from the parting, over the ear and on down past the jaw to the
    shoulder, behind the line of the cheek; it waves as it falls."""
    def wv(z):
        return s * 0.14 * math.sin(1.25 * z + 0.4)
    return loft([(7.4, 0.7, 1.6, s * 2.75 + wv(7.4), 0.15), (5.0, 0.95, 2.15, s * 3.2 + wv(5.0), 0.95),
                 (2.5, 1.05, 2.0, s * 3.25 + wv(2.5), 1.65), (0.0, 1.0, 1.9, s * 3.4 + wv(0.0), 2.15),
                 (-1.6, 0.85, 1.6, s * 3.6 + wv(-1.6), 2.55), (-3.0, 0.55, 1.0, s * 3.75 + wv(-3.0), 2.9),
                 (-4.0, 0.2, 0.4, s * 3.7, 3.1)])


def drape_fall(dg):
    """The long hair down her back (the ranger's drape, wavier and a little fuller): for each strip across it, the
    surface it rests on is found by casting at the figure from behind, and the hair lies on that, then hangs,
    spanning the hollows rather than sinking into them. Its sides wave in and out as it falls; its end is uneven,
    a few locks longer than the rest."""
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
        for k_ in range(8):
            hit = body.ray_cast(Vector((x * 0.94 ** k_, 30.0, z)), Vector((0, -1, 0)), 40.0)
            if hit[0] is not None:
                return hit[0].y
        return 0.0

    W = [(7.2, 2.1), (6.0, 3.1), (4.5, 3.45), (1.5, 3.6), (-1.5, 4.0), (-4.5, 4.7), (-9.0, 4.55), (-13.0, 4.0), (-16.0, 3.3),
         (-17.5, 2.4)]

    def width(z):
        for (z0, w0), (z1, w1) in zip(W, W[1:]):
            if z1 <= z <= z0:
                return lerp(w0, w1, (z0 - z) / (z0 - z1))
        return W[0][1] if z > W[0][0] else W[-1][1]

    NX, NZ = 241, 120
    z_top = 7.2
    PARTINGS = [-0.74, -0.5, -0.27, -0.04, 0.2, 0.43, 0.66]
    EDGES = [-1.0] + PARTINGS + [1.0]

    def lock_of(u):
        for k_ in range(len(EDGES) - 1):
            if u <= EDGES[k_ + 1] or k_ == len(EDGES) - 2:
                return k_, (u - EDGES[k_]) / (EDGES[k_ + 1] - EDGES[k_])

    def end_z(u):
        k_, f_ = lock_of(u)
        return -16.2 - 1.0 * (1 - u * u) + 1.5 * abs(u) ** 4 - (0.8 + 0.3 * (k_ % 2) + 0.15 * (k_ % 3 == 1)) * (1.0 - abs(2 * f_ - 1)) ** 1.15

    inner, outer = [], []
    for i in range(NX):
        u = -1 + 2 * i / (NX - 1)
        zs = [lerp(z_top, end_z(u), j / (NZ - 1)) for j in range(NZ)]
        z_end = -16.2 - 1.0 * (1 - u * u) + 1.5 * abs(u) ** 4 - 0.6
        pts = []
        for j, z in enumerate(zs):
            v = j / (NZ - 1)
            k_, _ = lock_of(u)
            uc = 0.5 * (EDGES[k_] + EDGES[k_ + 1])
            g_ = min(1.0, max(0.0, (v - 0.88) / 0.12))
            ue = lerp(u, uc, 0.3 * g_ * g_ * (3 - 2 * g_))
            vs_ = min(1.0, (z_top - z) / (z_top - z_end))
            # the waves: the sides swell out and draw in as the hair falls, the two sides out of step
            wave = 1.0 + 0.07 * smoothstep(2.0, -2.0, z) * math.sin(0.85 * z + 1.6 * ue)
            x = ue * width(z) * wave * (1.0 - 0.12 * max(0.0, (vs_ - 0.75) / 0.25) ** 2 * abs(u))
            grow = min(1.0, (z_top - z) / 5.0)
            grow = grow * grow * (3 - 2 * grow)
            t_ = (0.95 - 0.3 * vs_) * max(0.0, 1 - u * u) ** (0.5 if z < -0.5 else 1.4) * (1.0 if v < 0.92 else math.cos((v - 0.92) / 0.08 * math.pi / 2) ** 0.5)
            t = max(t_, lerp(0.4, 0.28, min(1.0, max(0.0, (v - 0.75) / 0.25))))
            wb = min(1.0, max(0.0, (0.5 - z) / 2.0))
            wb = wb * wb * (3 - 2 * wb)
            sup = back_y(x, z)
            if wb > 0.0:
                sup = lerp(sup, sum(back_y(x + dx_, z) * w_ for dx_, w_ in
                                    ((-0.9, 0.1), (-0.45, 0.2), (0.0, 0.4), (0.45, 0.2), (0.9, 0.1))), wb)
            y = lerp(sup + t_ * grow, max(sup + t * grow, sup + 0.52), wb)
            pts.append([x, y, z, t, sup])
        top_ = [k for k, q in enumerate(pts) if q[2] >= -12.5]
        hull = []
        for k in top_:
            q = pts[k]
            while len(hull) >= 2:
                a, b = pts[hull[-2]], pts[hull[-1]]
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
                g_ = min(1.0, max(0.0, (pts[k][2] - 3.0) / 2.5))
                side = 1.0 - abs(u) ** 2 * g_ * g_ * (3 - 2 * g_)
                pts[k][1] += lift * side
        if len(hull) >= 2:
            a, b = pts[hull[-2]], pts[hull[-1]]
            slope = (b[1] - a[1]) / (a[2] - b[2]) if a[2] > b[2] else 0.0
            for k in range(hull[-1] + 1, len(pts)):
                pts[k][1] = max(pts[k][1], b[1] + slope * (b[2] - pts[k][2]))
        for _ in range(3):
            ys = [q[1] for q in pts]
            for k, q in enumerate(pts):
                fe = min(1.0, max(0.0, (q[2] - 0.5) / 1.5), max(0.0, (7.4 - q[2]) / 2.4))
                if fe > 0.0 and k >= 8:
                    w_ = ys[k - 8:k + 9]
                    q[1] = lerp(q[1], max(q[4] + 0.15, sum(w_) / len(w_)), fe)
        col_out, col_in = [], []
        for j, (x, y, z, t, sup) in enumerate(pts):
            vs_ = min(1.0, (z_top - z) / (z_top - z_end))
            if vs_ > 0.8:
                y = lerp(y, max(sup + 0.52, y - 0.6), ((vs_ - 0.8) / 0.2) ** 2 * 0.7)
            wb = min(1.0, max(0.0, (0.5 - z) / 2.0))
            wb = wb * wb * (3 - 2 * wb)
            y = max(lerp(y - 1.0, sup + 0.42, wb), y)
            gap = y - sup
            th = t if gap < t + 0.3 else max(t, gap - 0.15)
            yi = lerp(y - th, max(y - th, sup + 0.12), wb)
            if abs(u) > 0.45 and y - yi > 0.3:
                f_ = min(1.0, (abs(u) - 0.45) / 0.55)
                y = yi + max(0.15, (y - yi) * math.sqrt(max(0.0, 1.0 - f_ * f_)))
            col_out.append(Vector((x, y, z)))
            col_in.append(Vector((x, yi, z)))
        inner.append(col_in)
        outer.append(col_out)
    bm = bmesh.new()
    vin = [[bm.verts.new(p) for p in c] for c in inner]
    vout = [[bm.verts.new(p) for p in c] for c in outer]
    for i in range(NX - 1):
        for j in range(NZ - 1):
            bm.faces.new((vout[i][j], vout[i + 1][j], vout[i + 1][j + 1], vout[i][j + 1]))
            bm.faces.new((vin[i][j + 1], vin[i + 1][j + 1], vin[i + 1][j], vin[i][j]))
    for i in range(NX - 1):
        for j in (0, NZ - 1):
            bm.faces.new((vin[i][j], vin[i + 1][j], vout[i + 1][j], vout[i][j]))
    for i in (0, NX - 1):
        for j in range(NZ - 1):
            bm.faces.new((vin[i][j], vout[i][j], vout[i][j + 1], vin[i][j + 1]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def fuse_hair(head, names, voxel=0.065):
    """Melt the hair's pieces (the cap, the side hair, the back of the head and the long fall) into one surface,
    as the ranger's: joined, remeshed as one volume and relaxed, so every join is a soft fillet. Then the waves."""
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
    ends = [v for v in bm.verts if v.co.z < -6.0]
    for _ in range(14):
        bmesh.ops.smooth_laplacian_vert(bm, verts=ends, lambda_factor=0.8, lambda_border=0.0,
                                        use_x=True, use_y=True, use_z=False, preserve_volume=True)
    sides = [v for v in bm.verts if -3.5 < v.co.z < 8.2 and abs(v.co.x) > 1.6 and v.co.y < 5.0]
    for k_ in range(30):
        bmesh.ops.smooth_laplacian_vert(bm, verts=sides, lambda_factor=0.7, lambda_border=0.0,
                                        use_x=True, use_y=True, use_z=k_ % 3 == 2, preserve_volume=True)
    band = [v for v in bm.verts if -4.0 < v.co.z < 9.0 and v.co.y > 1.0]
    for _ in range(8):
        bmesh.ops.smooth_laplacian_vert(bm, verts=band, lambda_factor=0.5, lambda_border=0.0,
                                        use_x=True, use_y=True, use_z=True, preserve_volume=True)
    # the waves, pressed in last so no smoothing irons them out: below the ears the hair rises and falls across the
    # locks in broad S-curves, each lock a little out of step with the next (a lit crest and a shaded trough every
    # hand's breadth or so, which is what reads as wavy hair at this size), and a finer ripple over them
    bm.normal_update()
    for v in bm.verts:
        fade = smoothstep(5.0, 1.5, v.co.z)
        if fade <= 0.0:
            continue
        arc = math.atan2(v.co.x, v.co.y - 0.5) * max(2.5, math.hypot(v.co.x, v.co.y - 0.5))
        wave = 0.13 * math.sin(0.95 * v.co.z + 0.5 * arc + 0.7 * math.sin(0.33 * arc)) \
            + 0.04 * math.sin(2.3 * v.co.z - 0.8 * arc + 1.1)
        n_ = Vector((v.normal.x, v.normal.y, 0.0))
        # (on the outside of the hair only: pressed into its underside too, the troughs sank it into her back)
        rad = Vector((v.co.x, v.co.y - 0.5, 0.0))
        outside = smoothstep(0.0, 0.5, n_.dot(rad.normalized()) if rad.length > 1e-6 else 0.0)
        v.co += n_ * wave * fade * outside
    hair = mesh_obj('Hair', bm, M_HAIR, head, subdiv=0, rest=hair_flow)
    strand_colours(hair)
    return hair


def cheek_scar(head):
    """A pale scar across her right cheekbone: a short, slightly curved slash from below the outer corner of the
    eye down toward the corner of the nose, laid on the cheek, thin at its ends."""
    H = HEAD_H
    me = bpy.data.objects['Head shape'].data      # (the finished head: the eyes' sculpting reaches the cheek)
    from mathutils.bvhtree import BVHTree
    head_tree = BVHTree.FromPolygons([v.co.copy() for v in me.vertices], [list(p.vertices) for p in me.polygons])
    pts = []
    for i in range(9):
        t = i / 8
        x = lerp(-0.215 * H, -0.15 * H, t)
        z = lerp(0.45 * H, 0.35 * H, t) + 0.12 * math.sin(math.pi * t)
        hit = head_tree.ray_cast(Vector((x, -30.0, z)), Vector((0, 1, 0)), 60.0)
        y = hit[0].y if hit[0] is not None else -2.5
        pts.append((x, y - 0.05, z))
    radii = [0.04 + 0.07 * math.sin(math.pi * i / 8) ** 0.7 for i in range(9)]
    return mesh_obj('Scar', tube(pts, radii, flat=0.6, n=8), M_SCAR, head, subdiv=0)


# ---------------------------------------------------------------- the sleeves and bracers
# The gambeson's sleeve: a quilted sleeve fitted to the arm (local height from the shoulder joint, radius), a
# little full above the bracer, which takes it from below the elbow to the wrist.
SLEEVE_R = [(1.6, 0.3), (1.35, 1.33), (0.85, 1.99), (0.0, 2.2), (-2.0, 2.16), (-5.5, 1.93), (-UPPER, 1.72),
            (-UPPER - 1.0, 1.8), (-UPPER - 2.2, 1.6), (-UPPER - 5.0, 1.42), (-UPPER - FORE + 0.4, 1.26),
            (-UPPER - FORE - 0.4, 1.24)]
BR_TOP, BR_BOT = -UPPER - 1.6, -UPPER - FORE - 0.55
BRACER_R = [(BR_TOP, 2.0), (BR_TOP - 0.5, 1.92), (-UPPER - 5.0, 1.72), (-UPPER - FORE + 0.8, 1.54), (BR_BOT, 1.62)]
BR_STRAPS = (BR_TOP - 1.4, -UPPER - 5.6, -UPPER - FORE + 1.3)

SLEEVE_CREASES = [
    (-1.4, -1.05, 0.6, -0.9, 0.0, 0.1), (-2.4, -2.2, 0.55, 0.8, 0.0, 0.08),
    (-UPPER + 0.7, 0.25, 0.85, 0.35, 0.5, 0.12), (-UPPER - 0.2, -0.3, 0.9, -0.3, 0.6, 0.11),
]


def sleeve_form(z, phi, s):
    """(radial offset, crease) of the gambeson sleeve: its creases at the armpit and elbow, and the quilting, a
    shallow groove round the arm every couple of units on the upper arm."""
    pm = s * phi
    off, cr = creases(SLEEVE_CREASES, z, pm, 0.12)
    q = smoothstep(0.4, -0.8, z) * smoothstep(-UPPER + 0.6, -UPPER + 2.0, z)
    d = (z % 2.2) - 1.1
    g = math.exp(-(d / 0.18) ** 2) * q
    off -= 0.035 * g
    cr += 0.25 * g
    return off, min(1.0, cr)


def bracer_form(z, phi, s):
    """(radial offset, crease) of the bracer's leather: drawn in under each strap, and soft wrinkles where the
    wrist bends."""
    pm = s * phi
    off, cr = 0.0, 0.0
    k = _gs(z, -UPPER - FORE + 1.6, 1.0) * max(0.0, math.cos(pm - 0.4)) ** 2
    v = fold_ridge(4.0 * (z + UPPER + FORE) + pm)
    off += 0.06 * k * v
    cr += 0.5 * k * max(0.0, -v)
    for zs in BR_STRAPS:
        off -= 0.05 * _gs(z, zs, 0.35)
        cr += 0.3 * (_gs(z, zs + 0.42, 0.1) + _gs(z, zs - 0.42, 0.1))
    return off, min(1.0, cr)


def bracer_pt(z, phi, s, lift=0.0):
    r = _profile(BRACER_R, z) + bracer_form(z, phi, s)[0] + lift
    return -r * math.sin(phi), -r * 1.03 * math.cos(phi)


# ---------------------------------------------------------------- the legs
LEG_R = [(2.0, 3.05), (-1.0, 2.95), (-6.0, 2.45), (-THIGH + 1.2, 1.98), (-THIGH, 1.93), (-THIGH - 1.5, 1.9),
         (-THIGH - 4.0, 1.92), (-THIGH - SHIN, 1.3)]
LEG_CREASES = [
    (-1.6, -0.45, 0.55, 1.0, 0.0, 0.06), (-2.7, -0.6, 0.5, 1.1, 0.0, 0.05),
    (-THIGH + 0.9, 3.0, 0.7, 0.3, -0.3, 0.13), (-THIGH + 0.1, -2.9, 0.9, -0.25, -0.35, 0.15),
]
BOOT_TOP = 1.3        # the boot's top at the front, over the knee (height from the knee); lower behind


def leg_form(z, phi, s):
    off, cr = creases(LEG_CREASES, z, s * phi, 0.1)
    return off, min(1.0, cr)


def boot_top(p):
    """The boot's top: highest over the front of the knee, dipping behind, where the knee bends."""
    return -1.1 * (0.5 - 0.5 * math.cos(p)) ** 1.5


def boot_rings():
    """The tall boot's rings (height from the knee, half-width, depth, centre x, centre y, hem): over the trouser
    leg round the knee, down the calf, a little slack and wrinkled above the ankle, then the foot (as the ranger's
    boot)."""
    S = SHIN
    R = lambda zk: _profile(LEG_R, zk - THIGH)   # noqa: E731

    def wr(z):
        return 0.07 * math.sin(4.4 * z + 0.6) * math.exp(-((z + S - 1.8) / 1.3) ** 2)
    return [(BOOT_TOP, R(BOOT_TOP) + 0.36, R(BOOT_TOP) * 1.04 + 0.5, 0, -0.1, boot_top),
            (BOOT_TOP - 0.5, R(BOOT_TOP - 0.5) + 0.34, R(BOOT_TOP - 0.5) * 1.04 + 0.48, 0, -0.1, lambda p: boot_top(p) * 0.92),
            (0.0, 2.2, 2.55, 0, -0.2), (-1.4, 2.16, 2.3), (-3.6, 2.22, 2.34), (-6.5, 2.08, 2.18),
            (-S + 2.6, 1.76 + wr(-S + 2.6), 1.88), (-S + 1.9, 1.69 + wr(-S + 1.9), 1.82), (-S + 1.2, 1.72 + wr(-S + 1.2), 1.88),
            (-S + 0.6, 1.67, 1.85, 0, -0.05), (-S - 0.5, 1.78, 2.0, 0, -0.3), (-S - 1.4, 1.84, 2.8, 0, -1.0),
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


# ---------------------------------------------------------------- the warden
def build():
    root, body, pelvis, chest = begin_figure('Warden')

    # ---- torso: the gambeson (the body's own surface) and over it the leather tunic, to the waist
    torso_mesh('Gambeson', M_SLEEVE, body)
    # a high collar, the gambeson standing up round the neck
    mesh_obj('Collar', loft([(54.0, 2.0, 2.05, 0, 0.4), (52.8, 2.5, 2.55, 0, 0.35)], cap_top=False, cap_bot=False),
             M_SLEEVE, body, solid=0.3)

    def tunic_form(z, phi):
        side = abs(math.sin(phi)) ** 3
        band = _gs(z, 41.6, 1.0) * side
        v = fold_ridge(3.4 * (z - 41.6) + 2.4 * math.cos(phi))
        return 0.1 * band * v, 0.7 * band * max(0.0, -v)

    tz = [50.6 - 0.3 * i for i in range(41)]       # 50.6 down to 38.6, under the belt
    to = mesh_obj('Tunic', loft_fn(tz, lambda z, p: torso_point(z, p, TUNIC_OFF + tunic_form(z, p)[0]), n=128,
                                   cap_top=False, cap_bot=False), M_TUNIC, body, solid=0.22)
    crease_attr(to, lambda co: tunic_form(co.z, math.atan2(-co.x, -co.y))[1])
    # its top edge bound in darker leather, a band round the base of the neck and over the shoulders
    mesh_obj('Tunic binding', loft_fn([50.75, 50.45, 50.15], lambda z, p: torso_point(z, p, TUNIC_OFF + 0.12), n=96,
                                      cap_top=False, cap_bot=False), M_BIND, body, solid=0.2, subdiv=0)

    # ---- the tabard over the chest, to the belt (the part below the belt hangs from the hips; see below)
    def tab_grid(z0, z1, rows, hem=None):
        def pt(i, j):
            u = -1 + 2 * j / 24
            v = i / (rows - 1)
            z = lerp(z0, hem(u) if hem else z1, v)
            x = u * tab_hw(z)
            return (x, tab_y(x, z), z)
        return face_out(grid(rows, 25, pt))
    up = mesh_obj('Tabard', tab_grid(52.6, 38.9, 56), M_TABARD, body, solid=0.12)
    crease_attr(up, lambda co: tab_folds(co.x, co.z))
    # the red trim: a band down each edge, the yoke embroidered at the chest (a V from the collar), and a band
    # down the middle from the foot of the V
    TL = 0.08      # (the embroidery stands this far off the cloth)
    for s in (-1, 1):
        mesh_obj(f'Tabard edge {s}', strip(polyline(lambda z: (s * (tab_hw(z) - 0.28), z), 52.5, 38.9, 40), 0.24,
                                           tab_y, TL), M_TRIM, body, subdiv=0, solid=0.05)
        mesh_obj(f'Tabard yoke {s}', strip(polyline(lambda t: (s * lerp(1.45, 0.0, t), lerp(52.3, 47.2, t)), 0.0, 1.0, 16),
                                           0.3, tab_y, TL), M_TRIM, body, subdiv=0, solid=0.05)
    mesh_obj('Tabard collar', strip(polyline(lambda u: (u * (tab_hw(52.3) - 0.1), 52.3), -1.0, 1.0, 14), 0.24, tab_y, TL),
             M_TRIM, body, subdiv=0, solid=0.05)
    mesh_obj('Tabard band', strip([(0.0, 47.4), (0.0, 43.0), (0.0, 38.9)], 0.32, tab_y, TL), M_TRIM, body, subdiv=0,
             solid=0.05)

    # ---- the belt at the waist, over the tabard, with a gold buckle
    BO = T_OFF + 0.3
    mesh_obj('Belt', loft_fn([40.1, 39.4, 38.7], lambda z, p: torso_point(z, p, BO, bust=0.0), n=96, cap_top=False,
                             cap_bot=False), M_BELT, body, solid=0.28, subdiv=0)
    bst = []
    for k in range(80):
        p = 2 * math.pi * k / 80
        if abs(wrap(p)) < 0.22:
            continue
        for dz in (0.5, -0.5):
            x, y = torso_point(39.4 + dz, p, BO + 0.29, bust=0.0)
            bst.append((x, y, 39.4 + dz))
    mesh_obj('Belt stitching', dots(bst, 0.15), M_STITCH, body, subdiv=0)
    by_ = torso_point(39.4, 0.0, BO, bust=0.0)[1]
    mesh_obj('Buckle', frame_bm(1.6, 1.7, 0.3, 0.3), M_GOLD, body, loc=(0, by_ - 0.42, 39.4), subdiv=0)
    mesh_obj('Buckle prong', box(0.9, 0.16, 0.16), M_GOLD, body, loc=(0.22, by_ - 0.56, 39.4), subdiv=0)

    # ---- neck and head (the base figure's), and the scar on her cheek
    head, head_tree = build_head(body)
    cheek_scar(head)

    # ---- the hair: the cap, the side hair, the back of the head, then the long fall draped and all fused
    cap, w = hair_cap(head_tree)
    cap_o = mesh_obj('Hair cap', cap, M_HAIR, head, solid=0.42)
    vg = cap_o.vertex_groups.new(name='thick')
    for i_, w_ in enumerate(w):
        vg.add([i_], w_, 'REPLACE')
    cap_o.modifiers['thickness'].vertex_group = 'thick'
    cap_o.modifiers['thickness'].thickness_vertex_group = 0.32
    for s in (-1, 1):
        mesh_obj(f'Side lock {s}', side_lock(s), M_HAIR, head)
    mesh_obj('Hair fall', loft([(7.5, 3.5, 3.4, 0, 0.75), (4.5, 3.9, 4.1, 0, 0.9), (2.0, 3.9, 2.7, 0, 2.1), (0.5, 3.75, 1.9, 0, 2.8),
                                (-0.4, 3.25, 1.3, 0, 2.95), (-1.4, 2.4, 0.75, 0, 2.75)]), M_HAIR, head)
    fuse_hair(head, ['Hair cap', 'Side lock -1', 'Side lock 1', 'Hair fall'])

    # ---- the rig (the base figure's)
    build_rig(body)

    # ---- what hangs from the hips, carried by the thighs: the tunic's skirt and the tabard below the belt
    # (rows from 0 at the top to 1 at the hem, which rises and falls round her: see skirt_hem)
    SR = [1.0 - i / 44 for i in range(45)]

    def sk(v, phi):
        z = lerp(skirt_hem(phi), 39.9, v)
        x, y, _ = skirt_pt(z, phi)
        return x, y, z - v
    so = mesh_obj('Tunic skirt', loft_fn(SR, sk, n=128, cap_top=False, cap_bot=False), M_TUNIC, skin=skirt_weights,
                  solid=0.2)
    crease_attr(so, lambda co: skirt_form(co.z, math.atan2(-co.x, -co.y))[1])

    # its hem bound in darker leather, with a running stitch above the binding
    def bind(v, phi):
        z = skirt_hem(phi) + 0.5 * v
        x, y, _ = skirt_pt(z, phi)
        r = math.hypot(x, y) or 1.0
        return x + x / r * 0.1, y + y / r * 0.1, z - v
    mesh_obj('Skirt hem', loft_fn([1.0, 0.0], bind, n=128, cap_top=False, cap_bot=False), M_BIND, skin=skirt_weights,
             solid=0.2, subdiv=0)
    sst = []
    for k in range(64):
        phi = 2 * math.pi * k / 64
        z = skirt_hem(phi) + 0.78
        x, y, _ = skirt_pt(z, phi)
        r = math.hypot(x, y) or 1.0
        sst.append((x + x / r * 0.22, y + y / r * 0.22, z))
    mesh_obj('Skirt stitching', dots(sst, 0.14), M_STITCH, skin=skirt_weights, subdiv=0)

    lo = mesh_obj('Tabard skirt', tab_grid(39.9, None, 58, hem=tab_hem), M_TABARD, skin=skirt_weights, solid=0.12)
    crease_attr(lo, lambda co: tab_folds(co.x, co.z))
    # its trim: down both edges and round the stepped hem, and the band down the middle ending in a stepped cross
    hem_path = [(u * tab_hw(tab_hem(u)), tab_hem(u) + 0.3) for u in [-1 + 2 * i / 60 for i in range(61)]]
    hem_path = [(x * (1 - 0.09 / max(abs(x), 0.09)) if abs(x) > 0.1 else x, z) for x, z in hem_path]
    trims = [('Tabard skirt hem', hem_path, 0.26)]
    for s in (-1, 1):
        trims.append((f'Tabard skirt edge {s}', polyline(lambda z: (s * (tab_hw(z) - 0.28), z), 39.9, tab_hem(s) + 0.5, 40), 0.24))
    trims.append(('Tabard skirt band', [(0.0, 39.9), (0.0, 31.0), (0.0, 24.6)], 0.32))
    for i, (zc, hw_) in enumerate(((27.6, 1.2), (25.9, 0.7))):
        trims.append((f'Tabard skirt bar {i}', [(-hw_, zc), (hw_, zc)], 0.26))
    for nm, path, half in trims:
        mesh_obj(nm, strip(path, half, tab_y, TL), M_TRIM, skin=skirt_weights, subdiv=0, solid=0.05)

    # ---- arms: the gambeson sleeve under a buckled leather bracer, then the hand
    def sleeve(side, s, x, w):
        def fore(co, w=w):
            d = w(co)
            d[f'Forearm.{side}'] = d.get(f'Forearm.{side}', 0.0) + d.pop(f'Hand.{side}', 0.0)
            return d
        zs = [1.55 - 0.22 * i for i in range(int((1.55 + UPPER + FORE + 0.4) / 0.22) + 1)]

        def arm_pt(z, phi):
            r = _profile(SLEEVE_R, z) + deltoid(z, phi, s) + sleeve_form(z, phi, s)[0]
            return -r * math.sin(phi), -r * 1.03 * math.cos(phi)
        so = mesh_obj(f'Sleeve.{side}', loft_fn(zs, arm_pt, n=48, cap_bot=False), M_SLEEVE, loc=(x, 0.2, SH_Z), skin=fore,
                      solid=0.12)
        crease_attr(so, lambda co: sleeve_form(co.z, math.atan2(-co.x, -co.y), s)[1])
        bzs = [BR_TOP - 0.2 * i for i in range(int((BR_TOP - BR_BOT) / 0.2) + 1)] + [BR_BOT]
        bo_ = mesh_obj(f'Bracer.{side}', loft_fn(bzs, lambda z, p: bracer_pt(z, p, s), n=56, cap_top=False, cap_bot=False),
                       M_BRACER, loc=(x, 0.2, SH_Z), skin=fore, solid=0.16)
        crease_attr(bo_, lambda co: bracer_form(co.z, math.atan2(-co.x, -co.y), s)[1])
        for nm, z0, z1 in (('top', BR_TOP + 0.05, BR_TOP - 0.42), ('bottom', BR_BOT + 0.42, BR_BOT - 0.05)):
            mesh_obj(f'Bracer {nm} binding.{side}', loft_fn([z0, (z0 + z1) / 2, z1], lambda z, p: bracer_pt(z, p, s, 0.12), n=56,
                                                           cap_top=False, cap_bot=False),
                     M_BIND, loc=(x, 0.2, SH_Z), skin=fore, solid=0.12, subdiv=0)
        bst_ = []
        for k in range(40):
            p = 2 * math.pi * k / 40
            for z in (BR_TOP - 0.62, BR_BOT + 0.62):
                bst_.append((*bracer_pt(z, p, s, 0.18), z))
        mesh_obj(f'Bracer stitching.{side}', dots(bst_, 0.15), M_STITCH, loc=(x, 0.2, SH_Z), skin=fore, subdiv=0)
        # three straps round it, each buckled on the outside of the forearm
        for j, zs_ in enumerate(BR_STRAPS):
            mesh_obj(f'Bracer strap {j}.{side}', loft_fn([zs_ + 0.28, zs_, zs_ - 0.28], lambda z, p: bracer_pt(z, p, s, 0.1),
                                                         n=56, cap_top=False, cap_bot=False),
                     M_BELT, loc=(x, 0.2, SH_Z), skin=fore, solid=0.12, subdiv=0)
            ph = s * math.pi / 2
            bx, by2 = bracer_pt(zs_, ph, s, 0.3)
            mesh_obj(f'Bracer buckle {j}.{side}', turned(frame_bm(0.62, 0.7, 0.13, 0.14), ph), M_STEEL,
                     loc=(x + bx, 0.2 + by2, SH_Z + zs_), skin=fore, subdiv=0)
    build_arms(chest, sleeve)   # (the hands are the base figure's)

    # ---- legs: the trousers, and the tall boots over them
    def leg(side, s, x, w):
        zs = [2.0 - 0.2 * i for i in range(int((2.0 + THIGH + SHIN - 3.5) / 0.2) + 1)]

        def leg_pt(z, phi):
            r = _profile(LEG_R, z) + leg_form(z, phi, s)[0]
            cap = 0.4 * math.exp(-((z + THIGH + 0.3) / 1.1) ** 2) * max(0.0, math.cos(phi)) ** 2   # kneecap
            return -r * math.sin(phi), -(r * 1.04 + cap) * math.cos(phi)
        kz = HIP_Z - THIGH
        lo_ = mesh_obj(f'Leg.{side}', loft_fn(zs, leg_pt, n=56), M_LEGS, loc=(x, 0, HIP_Z), skin=w)
        crease_attr(lo_, lambda co: leg_form(co.z, math.atan2(-co.x, -co.y), s)[1])
        rings = boot_rings()
        mesh_obj(f'Boot.{side}', loft(rings, n=40, cap_top=False), M_BOOTS, loc=(x, 0, kz), skin=w, solid=0.12)
        # the top turned down into a cuff of paler, softer leather, deepest over the knee
        r0 = rings[0]
        mesh_obj(f'Boot cuff.{side}', loft([(BOOT_TOP + 0.12, r0[1] + 0.1, r0[2] + 0.1, 0, -0.1, boot_top),
                                            (BOOT_TOP - 0.05, r0[1] + 0.28, r0[2] + 0.28, 0, -0.1, boot_top),
                                            (BOOT_TOP - 1.3, r0[1] + 0.22, r0[2] + 0.2, 0, -0.1,
                                             lambda p: boot_top(p) * 0.8 - 0.25 * math.cos(p))],
                                           n=40, cap_top=False, cap_bot=False),
                 M_CUFF, loc=(x, 0, kz), subdiv=1, skin=w, solid=0.12)
        # straps buckled on the outside: round the calf (two) and over the instep at the ankle
        for j, zs_ in enumerate((-2.6, -5.6, -SHIN + 1.0)):
            rx, ry, cy = ring_r(rings, zs_)
            mesh_obj(f'Boot strap {j}.{side}', loft([(zs_ + 0.3, rx + 0.12, ry + 0.12, 0, cy), (zs_, rx + 0.15, ry + 0.15, 0, cy),
                                                     (zs_ - 0.3, rx + 0.12, ry + 0.12, 0, cy)], n=40, cap_top=False, cap_bot=False),
                     M_BELT, loc=(x, 0, kz), subdiv=1, skin=w, solid=0.12)
            fb = frame_bm(0.72, 0.8, 0.15, 0.16)
            turned(fb, s * math.pi / 2)
            mesh_obj(f'Boot buckle {j}.{side}', fb, M_STEEL, loc=(x - s * (rx + 0.32), cy, kz + zs_), subdiv=0, skin=w)
        # a stitched seam up the back of the boot
        bst2 = []
        for z in [-SHIN + 1.0 + 0.95 * i for i in range(int((SHIN - 2.2) / 0.95))]:
            rx, ry, cy = ring_r(rings, z)
            bst2.append((0.0, cy + ry + 0.06, z))
        mesh_obj(f'Boot stitching.{side}', dots(bst2, 0.15), M_STITCH, loc=(x, 0, kz), subdiv=0, skin=w)
        az = kz - SHIN
        mesh_obj(f'Welt.{side}', loft([(-2.25, 1.86, 3.36, 0, -1.65), (-2.4, 1.97, 3.5, 0, -1.7), (-2.8, 2.0, 3.53, 0, -1.7),
                                       (-2.86, 1.92, 3.45, 0, -1.7)], n=40, cap_top=False, cap_bot=False),
                 M_BIND, loc=(x, 0, az), subdiv=1, skin=w, solid=0.06)
        welt = []
        for k in range(44):
            ph = 2 * math.pi * k / 44
            welt.append((-2.0 * math.sin(ph), -1.7 - 3.52 * math.cos(ph), -2.6))
        mesh_obj(f'Welt stitching.{side}', dots(welt, 0.09), M_STITCH, loc=(x, 0, az), subdiv=0, skin=w)
        mesh_obj(f'Sole.{side}', loft([(-2.8, 1.93, 3.47, 0, -1.7), (-2.86, 1.95, 3.5, 0, -1.7), (-3.3, 1.93, 3.47, 0, -1.7),
                                       (-ANKLE_Z, 1.85, 3.38, 0, -1.7)], n=40), M_SOLE, loc=(x, 0, az), subdiv=1, skin=w)
    build_legs(pelvis, leg)
    # the rig, the camera and light, and the keyframes
    return finish_figure()


if __name__ == '__main__':
    run(build, 'warden', HERE)
