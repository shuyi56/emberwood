"""The Warden: the base female figure (blender/base/female.py) in her hair and clothes, rendered to a pixel-art
sprite sheet.

Long, loose, wavy dark-brown hair parted in the middle, falling past the shoulders down her back. A cream tabard over a belted brown leather tunic: the tabard runs from the collar to just
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

# ---------------------------------------------------------------- her face
# Designed in the face creator (/mnt/project-files/face-creator, the head preset "Warden") on the base head, from
# the reference: a stern, weathered face. A firm jaw with a clear corner and edge, a strong, slightly longer chin,
# high cheekbones over leaner cheeks, a heavier brow over deeper sockets, a straighter nose, thinner lips with the
# corners set level instead of lifted, and the eyes a little narrower and less tilted: a level, watchful look.
female.FACE.update({
    'jawWidth': 0.03, 'jawDef': 1.6, 'jawSharp': 0.95, 'jawAngle': 0.6, 'chinProj': 1.3, 'chinWidth': 1.1,
    'chinHeight': 1.15, 'cheekWidth': -0.02, 'cheekbones': 1.5, 'cbHeight': -0.004, 'apples': 0.65,
    'cheekFull': 0.55, 'noseProj': 1.25, 'noseWidth': 1.04, 'noseLen': -0.006, 'tipBall': 0.75, 'nostrils': 1.1,
    'browRidge': 1.6, 'sockets': 1.35, 'mouthWidth': 1.06, 'lipUpper': 0.75, 'lipLower': 0.88, 'smile': 0.3,
    'chinTaper': -0.2, 'eyeSpacing': 0.145,
})
female.EYE_OPEN = 0.64
female.EYE_TILT = 0.06
# her make-up, as the creator's preset: a light plum wash of shadow kept to the lids (a heavier one read as too
# much, and none runs in toward the nose), lined and winged at the outer corners, and wine lips; her brows dark,
# full and set low, rising to a sharp high arch
female.MAKEUP.update({'shadow': (106, 74, 112), 'shadowDark': (52, 36, 58), 'shadowLow': (150, 118, 132), 'shadowAmt': 0.42,
                      'shadowInner': 0.0, 'liner': 0.7, 'wing': 0.7,
                      'browArch': 1.5, 'browThick': 0.8, 'browHeight': -0.04})
female.LIP_COLOURS = {'MID': [124, 50, 64], 'UP': [113, 43, 56], 'HI': [135, 60, 76], 'DEEP': [78, 28, 37]}

# ---------------------------------------------------------------- her materials
# (her hair, brows and eyes are her own colours on the base figure's materials)
M_HAIR = mat_def(2, 'hair', ramp(0x36251c, 0.1), 0.24, surface='hair')
M_BROW = mat_def(17, 'brow', ramp(0x2e211c, 0.08), 0.0, flat=1)
# a violet iris, deep at the rim and paler toward the pupil
M_EYE = mat_def(16, 'iris', [[30, 20, 48], [58, 40, 96], [92, 66, 146], [132, 102, 188], [176, 150, 214]], 0.0, flat='painted')
M_TABARD = mat_def(30, 'tabard', ramp(0xd6cbb0, 0.09), 0.05, closeup='canvas', form=True)
M_TRIM = mat_def(31, 'embroidery', ramp(0x9b2f25, 0.09), 0.0, closeup='canvas')
M_PLATE = mat_def(32, 'plate', ramp(0x8e9196, 0.13), 0.0, closeup='steel')   # (steel_closeup)
M_BIND = mat_def(33, 'binding', ramp(0x38291f, 0.08), 0.18, surface='leather')
M_SLEEVE = mat_def(34, 'gambeson', ramp(0x4f3c2e, 0.1), 0.07, surface='twill', closeup='cloth', form=True)
M_BRACER = mat_def(35, 'bracers', ramp(0x8a5a30, 0.1), 0.26, surface='leather', closeup='leather', form=True)   # (hardened leather, bronze-brown)
M_LEGS = mat_def(36, 'trousers', ramp(0x463d33, 0.1), 0.05, surface='twill', closeup='cloth', form=True)
M_BOOTS = mat_def(37, 'boots', ramp(0x5a3321, 0.1), 0.24, surface='leather', closeup='leather')
M_CUFF = mat_def(38, 'boot cuff', ramp(0x6c4029, 0.1), 0.2, surface='leather', closeup='suede')
M_SOLE = mat_def(39, 'sole', ramp(0x221c19, 0.08), 0.0)
M_BELT = mat_def(40, 'belt', ramp(0x4b3123, 0.1), 0.2, surface='leather')
M_GOLD = mat_def(41, 'gold', ramp(0xb08f4f, 0.09), 0.0, closeup='gold', shadow_on=(M_TABARD, M_BELT))
M_STEEL = mat_def(42, 'steel', ramp(0x9a978f, 0.09), 0.0)
M_STITCH = mat_def(43, 'stitching', ramp(0xa8927a, 0.08), 0.0, flat=2)
M_ROBE = mat_def(46, 'robe', ramp(0x2e1d33, 0.1), 0.1, surface='wool', closeup='wool', form=True)
M_ROBE_TRIM = mat_def(47, 'robe trim', ramp(0x9c7c45, 0.09), 0.0, closeup='canvas')
M_CAPE = mat_def(48, 'cape', ramp(0x23162a, 0.1), 0.12, surface='wool', closeup='wool', form=True)
M_LINING = mat_def(49, 'cape lining', ramp(0x8e2b22, 0.1), 0.05, closeup='cloth', form=True)
M_COLLAR = mat_def(50, 'collar', ramp(0x3b3e45, 0.12), 0.0, closeup='steel')   # (blackened steel: steel_closeup)
M_BRONZE = mat_def(51, 'bronze', ramp(0x9a6438, 0.12), 0.0, closeup='steel')   # (the knees and greaves: steel_closeup in bronze)
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


# ---------------------------------------------------------------- the torso: breastplate, tabard, robe
# From the body out: the gambeson (the body's own surface), a steel breastplate, the tabard hanging over it, and
# the open robe over the tabard, its fronts lying over the tabard's edges.
P_OFF = 0.42             # the breastplate stands this far off the gambeson (its steel 0.18 thick)
TAB_GAP = 0.28           # the tabard over the steel
T_LOW = 0.95             # the tabard below the waist, off the hips
R_SIDE = 0.8             # the robe off the body at the sides and back (over the plate)


def front_at(z, x, off, bust=1.0):
    """The y of the body's front at (x, z), pushed out by off."""
    rx = _torso_dims(z)[0] + off
    phi = math.asin(max(-1.0, min(1.0, -x / rx)))
    return torso_point(z, phi, off, bust)[1]


def plate_pt(z, phi):
    """The breastplate at (z, phi): one rigid shell round the chest, not the body's shape. Across the front it
    spans between the breasts in one rounded form, falls from there to the waist in a single curve (no hollow
    under the bust), and a low keel runs down its middle."""
    x, y = torso_point(z, phi, P_OFF)
    if math.cos(phi) <= 0.0:
        return x, y

    def front(zz, xx):
        yy = front_at(zz, xx, P_OFF)
        if abs(xx) < 2.2:
            yy = min(yy, front_at(zz, math.copysign(2.2, xx) if xx else 2.2, P_OFF))
        return yy
    y = min(y, front(z, x))
    if z < 44.3:
        rx44 = _torso_dims(44.3)[0] + P_OFF
        y44 = front(44.3, max(-rx44, min(rx44, x)))
        y = min(y, lerp(y44, y, smoothstep(44.3, 39.5, z)))
    keel = 0.12 * math.exp(-(x / 1.0) ** 2) * smoothstep(52.5, 48.0, z) * smoothstep(38.5, 41.0, z)
    return x, y - keel


def plate_front(x, z):
    rx = _torso_dims(z)[0] + P_OFF
    phi = math.asin(max(-1.0, min(1.0, -x / rx)))
    return plate_pt(z, phi)[1]


def tab_hw(z):
    """The tabard's half-width down its length: broad enough over the chest to cover the breastplate's front,
    drawn in at the waist under the belt, then narrower below it, widening a little to the hem."""
    return _profile([(52.6, 2.2), (51.0, 4.0), (48.5, 5.0), (45.0, 5.0), (42.0, 4.3), (39.4, 4.0), (36.0, 3.3),
                     (33.0, 3.0), (21.0, 3.35)], z)


def tab_hem(u):
    """The hem's height across the tabard (u -1 to 1): stepped, the middle a tongue hanging lower than the sides,
    its corners rounded."""
    return 22.6 - 1.1 * smoothstep(0.46, 0.3, abs(u))


def tab_y(x, z):
    """The tabard's surface at (x, z). Over the chest it lies on the steel; from the fullest part of the plate it
    hangs straight down, clear of the plate as the plate draws in toward the waist, until the belt gathers it back
    in: so the cloth is held out by the metal and falls in a few folds gathered up from the belt. Below the hips
    it hangs, falling a little forward of them toward the hem, with two soft folds."""
    if z > 36.0:
        y = plate_front(x, z) - TAB_GAP
        if z < 44.0:
            hang = plate_front(x, 44.0) - TAB_GAP + 0.05 * (44.0 - z)
            y = lerp(min(y, hang), y, smoothstep(41.8, 39.6, z))
        g = _gs(z, 41.4, 1.3)
        return y + 0.2 * g * math.sin(2.4 * x + 0.4)
    y = front_at(max(z, 32.0), x, lerp(T_LOW, P_OFF + TAB_GAP, smoothstep(35.0, 36.0, z)), 0.0)
    if z < 32.0:
        y -= 0.12 * (32.0 - z)
    k = smoothstep(34.0, 26.0, z)
    return y + 0.16 * k * math.sin(1.9 * x + 0.8) + 0.05 * k * x * x / 4.0


def tab_folds(x, z):
    """The crease of the tabard's folds (for the 'crease' attribute)."""
    k = smoothstep(34.0, 26.0, z)
    g = _gs(z, 41.4, 1.3)
    return min(1.0, k * max(0.0, -math.sin(1.9 * x + 0.8)) * 0.5 + g * max(0.0, math.sin(2.4 * x + 0.4)) * 0.7)


def robe_edge(z):
    """How far from the middle the robe's fronts are (half the opening that shows the tabard): a narrow opening
    at the standing collar, widening over the chest, and over the tabard's edges below the belt."""
    return _profile([(53.6, 1.3), (52.3, 2.0), (50.5, 2.8), (48.0, 3.2), (44.0, 3.25), (41.0, 3.0), (39.0, 2.9),
                     (33.0, 2.75), (21.0, 3.05)], z)


def robe_hem(phi):
    # (the hem rises and falls with the folds: up at each crest, down in the valleys)
    v = 0.75 * pinch(5.0 * phi + 0.6 + 0.3 * math.sin(8.4)) + 0.25 * pinch(11.0 * phi + 1.9 + 4.2)
    return 20.8 - 0.7 * max(0.0, -math.cos(phi)) + 0.35 * v * (1.0 - 0.6 * max(0.0, math.cos(phi)) ** 2)


def robe_form(z, phi):
    """(radial offset, crease) of the robe's wool: below the belt a few deep folds round the sides and back,
    deepening toward the hem, and soft folds down the back from the shoulders."""
    k = smoothstep(37.5, 23.0, z) * (1.0 - 0.6 * max(0.0, math.cos(phi)) ** 2)
    # (a few deep folds, each a little off true as it falls, and a shallower one between)
    v = 0.75 * pinch(5.0 * phi + 0.6 + 0.3 * math.sin(0.4 * z)) + 0.25 * pinch(11.0 * phi + 1.9 + 0.2 * z)
    off = 0.55 * k * v
    cr = 0.8 * k * max(0.0, -v)
    b = smoothstep(51.0, 47.0, z) * smoothstep(38.0, 41.0, z) * max(0.0, -math.cos(phi)) ** 2
    w = fold_ridge(4.0 * phi + 0.5)
    off += 0.08 * b * w
    cr += 0.4 * b * max(0.0, -w)
    return off, min(1.0, cr)


def robe_pt(z, phi, lift=0.0):
    """The robe at (z, phi): over the plate (pushed out from it) above the hips, flaring over them below; near its
    fronts it is brought forward over the tabard, so the fronts lie on the tabard instead of passing through it."""
    off, _ = robe_form(z, phi)
    # (close over the hips at the sides, where the arms hang against it)
    hug = math.sin(phi) ** 4 * smoothstep(38.0, 35.5, z) * smoothstep(23.0, 27.0, z)
    # (behind, under the cape, the skirt falls nearly straight from the seat instead of flaring out in deep folds:
    # flared, it stood out past the cape's fall below the seat and poked through it, and a cape laid over it belled
    # out into a box)
    tame = 1.0 - 0.8 * max(0.0, -math.cos(phi)) ** 2 * smoothstep(33.0, 30.0, z)
    off *= tame
    extra = (R_SIDE - P_OFF + off) * (1.0 - 0.85 * hug) + lift
    if z >= 32.0:
        x, y = plate_pt(z, phi)
        if z < 36.0:
            x2, y2 = torso_point(z, phi, P_OFF, bust=0.0)
            x, y = lerp(x2, x, smoothstep(32.0, 36.0, z)), lerp(y2, y, smoothstep(32.0, 36.0, z))
    else:
        flare = (0.1 * (32.0 - z) * (0.5 + 0.5 * max(0.0, math.cos(phi))) + 0.05 * (32.0 - z) * max(0.0, -math.cos(phi))) \
            * (1.0 - 0.8 * hug) * tame
        x, y = torso_point(32.0, phi, P_OFF + flare, bust=0.0)
    r = math.hypot(x, y) or 1.0
    x, y = x + x / r * extra, y + y / r * extra
    # (and tucked in behind the forearms hanging at her sides, so the steel vambraces stand clear of it)
    fz = smoothstep(24.0, 27.0, z) * smoothstep(41.0, 38.0, z)
    if fz > 0.0:
        cx, cy_ = x, y
        for _ in range(20):
            if min(math.hypot(cx - ax, cy_ - 0.2) for ax in (-SH_X, SH_X)) >= 2.15:
                break
            cx, cy_ = cx * 0.97, cy_ * 0.97
        x, y = lerp(x, cx, fz), lerp(y, cy_, fz)
    if math.cos(phi) > 0.0:
        hw = tab_hw(z)
        w = smoothstep(hw + 1.2, hw - 0.2, abs(x))
        if w > 0.0:
            xt = max(-hw + 0.01, min(hw - 0.01, x))
            y = lerp(y, min(y, tab_y(xt, z) - 0.24 - lift), w)
    return x, y


def robe_open(z):
    """The half-angle of the robe's front opening at height z."""
    r = (_torso_dims(max(z, 32.0))[0] + R_SIDE) if z >= 32.0 else _torso_dims(32.0)[0] + R_SIDE + 0.1 * (32.0 - z)
    return math.asin(max(0.0, min(1.0, robe_edge(z) / r)))


def robe_sheet(z0, z1, rows, cols=112, hem=None, lift=0.0):
    """The robe from z0 down to z1 (or to its hem), all the way round but for the opening at the front."""
    def pt(i, j):
        v = i / (rows - 1)
        zz = lerp(z0, z1 if hem is None else 21.0, v)
        a = robe_open(zz)
        phi = a + (2 * math.pi - 2 * a) * j / (cols - 1)
        z = lerp(z0, z1 if hem is None else hem(phi), v)
        x, y = robe_pt(z, phi, lift)
        return (x, y, z)
    return face_out(grid(rows, cols, pt), axis_y=0.0)


def robe_y(x, z, lift=0.0):
    """The robe's surface seen from the front at (x, z) (for laying its trim)."""
    r = (_torso_dims(max(z, 32.0))[0] + R_SIDE) if z >= 32.0 else _torso_dims(32.0)[0] + R_SIDE + 0.1 * (32.0 - z)
    phi = math.asin(max(-1.0, min(1.0, -x / r)))
    return robe_pt(z, phi, lift)[1]


# ---------------------------------------------------------------- the hair
# Long and loose, parted in the middle: a cap over the skull from a natural hairline, the hair falling down either
# side of the face and over the ears, and the long fall down her back, all fused into one surface; then waves
# pressed into it, a few broad S-curves running across the locks. (Head-local coordinates: the chin at the origin,
# the face toward -Y.)
# ---------------------------------------------------------------- the cape
# Over the shoulders and hanging to the knees behind, open in front: a dark wool outside lined in red. Over the
# shoulders it lies on the body (the slope of the shoulders, the round of each arm) a little off it; below them it
# hangs free in a few deep folds, close behind the shoulder blades so the hair hangs straight down it. Its fronts
# are pinned at the chest by two gold medallions, and part below them, wider toward the hem.
CAPE_RINGS = [(53.4, 2.3, 2.3, 0.35), (52.4, 3.7, 3.1, 0.35), (51.3, 5.55, 3.8, 0.3), (50.3, 7.0, 4.25, 0.3),
              (49.3, 8.0, 4.55, 0.3), (48.4, 8.55, 4.7, 0.3), (47.0, 8.9, 4.8, 0.3), (42.0, 9.0, 4.85, 0.35),
              (34.0, 9.2, 5.0, 0.4), (26.0, 9.55, 5.2, 0.45), (18.0, 10.0, 5.5, 0.5)]
CAPE_BACK = [(53.4, 2.65), (52.4, 3.3), (51.3, 3.3), (49.0, 3.45), (45.0, 3.65), (38.0, 3.9), (28.0, 4.3), (18.0, 4.8)]
# (at the neck the fronts sweep back under the collar's foot instead of ending in a square corner beside the throat:
# the cloth comes out from under the collar and falls away down the yoke's edge)
CAPE_EDGE = [(53.4, 0.98), (53.0, 0.8), (52.5, 0.64), (52.0, 0.53), (51.5, 0.46), (51.0, 0.43), (50.3, 0.46), (49.3, 0.6), (47.5, 0.95), (45.0, 1.2), (30.0, 1.32), (18.0, 1.4)]
CLASP_Z = 50.3


def _table(T, z):
    for (z0, a), (z1, b) in zip(T, T[1:]):
        if z1 <= z <= z0:
            return lerp(a, b, (z0 - z) / (z0 - z1))
    return T[0][1] if z > T[0][0] else T[-1][1]


def _rings(R, z):
    """(rx, ry, cy) through the rings at z: one smooth (Catmull-Rom) curve, so the cloth is not flattened at each."""
    for i in range(len(R) - 1):
        a_, b_ = R[i], R[i + 1]
        if b_[0] <= z <= a_[0]:
            t = (a_[0] - z) / (a_[0] - b_[0])
            p0, p3 = R[max(0, i - 1)], R[min(len(R) - 1, i + 2)]
            t2, t3 = t * t, t * t * t
            return tuple(0.5 * (2 * q1 + (q2 - q0) * t + (2 * q0 - 5 * q1 + 4 * q2 - q3) * t2 + (3 * q1 - q0 - 3 * q2 + q3) * t3)
                         for q0, q1, q2, q3 in zip(p0[1:], a_[1:], b_[1:], p3[1:]))
    return R[0][1:] if z > R[0][0] else R[-1][1:]


CAPE_PLEATS = 14     # (no longer used: see cape_fold)
CAPE_FOLDS = 11      # hanging folds round the whole cape (five or six across the back)
CAPE_PLEAT_FACE = 0.74   # (the broad face of each pleat; the rest of its width is the return folded under it)


def _pleat_t(phi):
    return (CAPE_PLEATS * (phi + 0.17) / (2 * math.pi)) % 1.0


def pleat_w(z):
    """(The cape was once knife-pleated from the collar; its folds are now hanging folds, see cape_fold.)"""
    return 0.0


def pleat_pl(phi):
    t = _pleat_t(phi)
    a = CAPE_PLEAT_FACE
    return -1.0 + 2.0 * t / a if t < a else math.cos(math.pi * (t - a) / (1.0 - a))


def _cape_swing(z, phi):
    # (each fold a little off true and wandering as it falls, the row of them uneven in width)
    return pinch(CAPE_FOLDS * phi + 0.9 + 0.45 * math.sin(2.0 * phi + 0.4) + 0.22 * math.sin(0.19 * z + 1.3 * phi))


def cape_fold(z, phi):
    """The cape's folds as heavy wool hangs: smooth where it lies on the shoulders and shoulder blades, then a few
    deep, rounded tubes that start where it leaves them and deepen all the way to the hem, pinched at each valley
    (no knife pleats: pressed pleats stood as a row of shallow ridges, which read as stripes, and stiff).
    (offset in -1..1, depth)"""
    f = smoothstep(48.5, 30.0, z)
    hang = smoothstep(49.5, 22.0, z) ** 0.85
    return hang * _cape_swing(z, phi), f


def cape_crease(z, phi):
    """(where the cape folds into shadow: the valleys of its folds, deepening as they fall)"""
    hang = smoothstep(49.5, 22.0, z) ** 0.85
    return min(1.0, 0.85 * hang * max(0.0, -_cape_swing(z, phi)) ** 0.8)


def cape_hem(phi):
    fv, _ = cape_fold(19.0, phi)
    return 19.2 + 3.2 * max(0.0, math.cos(phi)) ** 1.5 + 0.25 * fv


CAPE_UNDER = None   # (z0, dz, n_phi, radii) of the clothes under the cape, once they are built (see under_table)


def clothes_bvh(skip=('Hair',), head=False):
    """One BVH of what is built so far (all but what the head carries, the stand-ins and the names in skip)."""
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    vs, ps = [], []
    for o in bpy.data.objects:
        if o.type != 'MESH' or o.name.startswith(skip) or o.name == 'Cape stand-in':
            continue
        q = o
        while q.parent is not None and q.parent.name != 'Head':
            q = q.parent
        if q.parent is not None and not (head and o.name in ('Head shape', 'Neck')):
            continue
        me = o.evaluated_get(dg).to_mesh()
        b = len(vs)
        vs += [o.matrix_world @ v.co for v in me.vertices]
        ps += [[b + i for i in pl.vertices] for pl in me.polygons]
        o.evaluated_get(dg).to_mesh_clear()
    return BVHTree.FromPolygons(vs, ps)


def under_table(skip=('Hair',)):
    """How far out the clothes already built reach, round the cape's axis at each height (the farthest surface
    along each outward ray), spread a little and softened so the cape falls over them in smooth curves instead of
    following every edge: over the robe's shoulders, its sleeves, the arms hanging at the sides and the skirt."""
    import numpy as np
    T = clothes_bvh(skip)
    z0, dz, nz, nph = 18.0, 0.5, 74, 180
    R = np.zeros((nz, nph))
    for i in range(nz):
        z = z0 + dz * i
        cy = _rings(CAPE_RINGS, z)[2]
        for j in range(nph):
            a = 2 * math.pi * j / nph
            d = Vector((-math.sin(a), -math.cos(a), 0.0))
            org, far, t = Vector((0.0, cy, z)), 0.0, 0.0
            for _ in range(24):
                hit = T.ray_cast(org, d, 16.0 - t)
                if hit[0] is None:
                    break
                t += hit[3] + 1e-3
                far = t
                org = hit[0] + d * 1e-3
            R[i, j] = far
    # spread (a running max over +-1 in height and +-6 degrees round), then soften
    M = R.copy()
    for di in range(-2, 3):
        for dj in range(-3, 4):
            M = np.maximum(M, np.roll(np.pad(R, ((2, 2), (0, 0)), mode='edge'), dj, axis=1)[2 + di:2 + di + nz])
    for _ in range(2):
        M = (M + np.roll(M, 1, 1) + np.roll(M, -1, 1) + np.roll(M, 2, 1) + np.roll(M, -2, 1)) / 5
        P = np.pad(M, ((2, 2), (0, 0)), mode='edge')
        M = (P[:-4] + P[1:-3] + P[2:-2] + P[3:-1] + P[4:]) / 5
    return z0, dz, nph, M


CAPE_UNDER_BACK = None   # (the table the hair was laid on: kept for the back, where the hair lies on the cape)


def _under(T, z, a):
    z0, dz, nph, M = T
    fi = max(0.0, min(M.shape[0] - 1.001, (z - z0) / dz))
    fj = (a % (2 * math.pi)) / (2 * math.pi) * nph
    i, j = int(fi), int(fj)
    u, v = fi - i, fj - j
    j1 = (j + 1) % nph
    j %= nph
    return lerp(lerp(M[i, j], M[i, j1], v), lerp(M[i + 1, j], M[i + 1, j1], v), u)


def under_r(z, a):
    """(behind, where the hair lies on it, the cape keeps to what was there when the hair was laid, so the two
    still meet; toward the sides it falls over the arms and sleeves too)"""
    r = _under(CAPE_UNDER, z, a)
    if CAPE_UNDER_BACK is None:
        return r
    wb = smoothstep(0.3, 0.6, -math.cos(a))
    # (never inside what is there now: the table the hair was laid on was taken before the robe's skirt, so below
    # the seat the cape cut back in under it, a ledge across the back)
    return lerp(r, max(r, _under(CAPE_UNDER_BACK, z, a)), wb) if wb > 0 else r


CAPE_DROP = 0.85   # (how far the cape's neck is let down below the collar, so more of the collar stands clear)


def cape_z(z):
    """Where the cape's row laid out at z hangs: its top let down by CAPE_DROP, easing to nothing by mid-back."""
    return z - CAPE_DROP * smoothstep(45.0, 53.4, z)


def cape_pt(z, phi, off=0.0):
    """The cape at its row z (laid out as before; it hangs at cape_z(z))."""
    return _cape_pt(cape_z(z), phi, off)


def _cape_pt(z, phi, off=0.0):
    rx, ry, cy = _rings(CAPE_RINGS, z)
    w = max(0.0, -math.cos(phi)) ** 1.5
    ry = lerp(ry, max(0.0, _table(CAPE_BACK, z) - cy), w)
    xr, yr = -rx * math.sin(phi), -ry * math.cos(phi)
    a = math.atan2(-xr, -yr)
    rr = math.hypot(xr, yr) or 1.0
    side = math.sin(a) ** 2
    fv, f = cape_fold(z, phi)
    # over the shoulders it lies on the body and arms a little off them; below, it hangs from them, never closer
    gap = lerp(0.3 + 0.3 * f, 0.18 + 0.2 * f, side)
    hull = body_hull(z, a, cy) + gap
    dw = smoothstep(53.4, 52.0, z) * smoothstep(44.0, 47.5, z)
    rd = max(lerp(rr, hull, dw), hull)
    if CAPE_UNDER is not None:
        rd = max(rd, under_r(z, a) + 0.32 + 0.12 * f + 0.25 * f * side)
    # the drape: below the shoulders it hangs from the outermost thing it rests on above (the shoulder blades, the
    # round of each arm), never tucking back in under it, and swings out a little as it falls; laid on what is
    # under it all the way down, it showed the shoulder blades, the small of the back and the seat through the wool
    rd = max(rd, cape_hang(z, a))
    # (the pleats stand out from what the cape lies on: their valleys rest on it, their crests are proud of it)
    k = (rd + off) / rr * (1.0 + 0.14 * fv * (1.0 - 0.2 * w))
    return xr * k, cy + yr * k


CAPE_HANG_TOP = 49.5     # (above this the cape lies on the shoulders; below it, it hangs)
CAPE_FLARE = 0.045       # (how far it swings out for each unit it has fallen)
_HANG = {}


def _support(z, a):
    """How far out the cape would lie on what is under it at (z, a), as _cape_pt lays it (no folds)."""
    cy = _rings(CAPE_RINGS, z)[2]
    side = math.sin(a) ** 2
    f = smoothstep(48.5, 30.0, z)
    r = body_hull(z, a, cy) + lerp(0.3 + 0.3 * f, 0.18 + 0.2 * f, side)
    if CAPE_UNDER is not None:
        r = max(r, under_r(z, a) + 0.32 + 0.12 * f + 0.25 * f * side)
    return r


def _convex_radii(rs):
    """Radii at evenly spaced angles (as phi), pushed out to the convex hull of the points they make."""
    n = len(rs)
    pts = [(-r * math.sin(2 * math.pi * k / n), -r * math.cos(2 * math.pi * k / n)) for k, r in enumerate(rs)]
    poly = female._convex([(round(x, 5), round(y, 5)) for x, y in pts])
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        dx, dy = -math.sin(a), -math.cos(a)
        best = rs[k]
        for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
            ex, ey = x1 - x0, y1 - y0
            den = dx * ey - dy * ex
            if abs(den) < 1e-9:
                continue
            t = (x0 * ey - y0 * ex) / den
            u = (x0 * dy - y0 * dx) / den
            if t > 0 and -1e-6 <= u <= 1 + 1e-6:
                best = max(best, t)
        out.append(best)
    return out


def cape_hang(z, a):
    """The radius the hanging cape keeps to at (z, a): the most it lay out at any height between where it starts
    to hang and z, plus its flare. Tabulated once per build of what lies under it."""
    key = id(CAPE_UNDER)
    if key not in _HANG:
        zs = [CAPE_HANG_TOP - 0.25 * i for i in range(int((CAPE_HANG_TOP - 17.0) / 0.25) + 1)]
        NA = 180
        tab = []
        run = [0.0] * NA
        S = []
        for z_ in zs:
            # (bridging across what is under it, as cloth does: over the seat it spans the two sides in one curve
            # instead of sinking in between them)
            S.append(_convex_radii([_support(z_, 2 * math.pi * k / NA) for k in range(NA)]))
        tab = [[0.0] * NA for _ in zs]
        for k in range(NA):
            # down each line of the cape, the cloth is drawn taut between what holds it out: from the shoulder
            # blades it runs straight to the seat instead of sinking into the small of the back and swelling out
            # again over the hips (the upper hull of the support down the line)
            pts = [(-z_, S[i][k]) for i, z_ in enumerate(zs)]
            hull = []
            for p_ in pts:
                while len(hull) >= 2 and ((hull[-1][0] - hull[-2][0]) * (p_[1] - hull[-2][1]) -
                                          (hull[-1][1] - hull[-2][1]) * (p_[0] - hull[-2][0])) >= 0:
                    hull.pop()
                hull.append(p_)
            j = 0
            run = 0.0
            for i, (u, r) in enumerate(pts):
                while j < len(hull) - 2 and hull[j + 1][0] < u:
                    j += 1
                (u0, r0), (u1, r1) = hull[j], hull[min(j + 1, len(hull) - 1)]
                h = r0 + (r1 - r0) * ((u - u0) / (u1 - u0) if u1 > u0 else 0.0)
                run = max(run, h, r)
                tab[i][k] = run + CAPE_FLARE * (CAPE_HANG_TOP - zs[i])
        _HANG.clear()
        _HANG[key] = (zs, NA, tab)
    zs, NA, tab = _HANG[key]
    if z >= CAPE_HANG_TOP:
        return 0.0
    fi = min(len(zs) - 1.001, (CAPE_HANG_TOP - z) / 0.25)
    i = int(fi)
    t = fi - i
    fa = (a % (2 * math.pi)) / (2 * math.pi) * NA
    j = int(fa) % NA
    u = fa - int(fa)
    j2 = (j + 1) % NA
    r0 = tab[i][j] * (1 - u) + tab[i][j2] * u
    r1 = tab[i + 1][j] * (1 - u) + tab[i + 1][j2] * u
    # (eased in below the shoulders, so the lying and hanging cloth meet without a crease)
    return (r0 * (1 - t) + r1 * t) * smoothstep(CAPE_HANG_TOP, CAPE_HANG_TOP - 1.5, z)


CAPE_CREASE = []   # (the crease at each point of the last sheet built, in its vertex order)


def cape_sheet(off=0.0, rows=96, cols=260):
    CAPE_CREASE.clear()

    def pt(i, j):
        v = i / (rows - 1)
        zz = lerp(53.4, 20.0, v)
        e = _table(CAPE_EDGE, zz)
        phi = e + (2 * math.pi - 2 * e) * j / (cols - 1)
        e = _table(CAPE_EDGE, lerp(53.4, cape_hem(phi), v))
        phi = e + (2 * math.pi - 2 * e) * j / (cols - 1)
        z = lerp(53.4, cape_hem(phi), v ** 1.15)
        x, y = cape_pt(z, phi, off)
        CAPE_CREASE.append(cape_crease(z, phi))
        z = cape_z(z)
        # (over the shoulders the cloth lies nearly flat, so there the pleats rise up off it instead of outward)
        up = 0.2 * pleat_w(z) * (pleat_pl(phi) + 0.6) * smoothstep(45.0, 48.5, z)
        return (x, y, z + up)
    return face_out(grid(rows, cols, pt), axis_y=0.0)


def set_crease(o, values):
    o.data.attributes.new('crease', 'FLOAT', 'POINT').data.foreach_set('value', list(values))


def cape_weights(co):
    """The cape rides on the trunk; toward the hem the legs carry part of it, as they carry the robe's skirt, so a
    thigh swinging back pushes the cloth along instead of the skirt coming through it."""
    w = trunk_weights(co)
    f = 0.85 * smoothstep(35.0, 27.0, co.z)
    if f <= 1e-4:
        return w
    sk = skirt_weights(co)
    keys = set(w) | set(sk)
    return {k: lerp(w.get(k, 0.0), sk.get(k, 0.0), f) for k in keys}


def cape_on_arms(names):
    """Over the round of each arm the cape rides on the arm (as the base's mantle_on_arms); further down, where it
    falls over the arm hanging at her side, the arm carries it as it swings, fading off a little way round, so the
    cloth is pushed along instead of the hand passing through it."""
    for o in [bpy.data.objects[n] for n in names if n in bpy.data.objects]:
        mw = o.matrix_world
        gs = {g.name: g for g in o.vertex_groups}
        for side in ('R', 'L'):
            for b_ in ('UpperArm', 'Forearm'):
                nm = f'{b_}.{side}'
                gs[nm] = gs.get(nm) or o.vertex_groups.new(name=nm)
        ez = SH_Z - UPPER
        for v in o.data.vertices:
            co = mw @ v.co
            for s_, side in ((1, 'R'), (-1, 'L')):
                out = smoothstep(SH_X - 1.6, SH_X + 1.0, -s_ * co.x)
                top = smoothstep(SH_Z + 2.0, SH_Z - 0.5, co.z) * smoothstep(SH_Z - 9.0, SH_Z - 4.0, co.z) * 0.7
                d = math.hypot(co.x + s_ * SH_X, co.y - 0.2)
                # (reaching well out from the arm: the cape hangs well off it, and a narrower reach let the hand swing back through it)
                near = smoothstep(6.6, 3.0, d) * smoothstep(SH_Z - 1.0, SH_Z - 5.0, co.z) * \
                    smoothstep(SH_Z - UPPER - FORE - 8.0, SH_Z - UPPER - FORE - 3.0, co.z)
                wa = max(out * top, near)
                if wa <= 1e-4:
                    continue
                lower = smoothstep(ez + 1.0, ez - 1.5, co.z)
                for g in v.groups:
                    g.weight *= 1.0 - wa
                gs[f'UpperArm.{side}'].add([v.index], wa * (1 - lower), 'REPLACE')
                if lower > 0:
                    gs[f'Forearm.{side}'].add([v.index], wa * lower, 'REPLACE')


def medallion():
    """A gold medallion facing -Y: a domed disc in a raised rim, a boss in the middle and eight rays round it."""
    bm = bmesh.new()
    parts = [ellipsoid(0.82, 0.16, 0.82, 28, 10), ellipsoid(0.3, 0.2, 0.3, 16, 8)]
    for k in range(8):
        a = 2 * math.pi * k / 8
        r = ellipsoid(0.09, 0.08, 0.26, 8, 6)
        bmesh.ops.rotate(r, cent=(0, 0, 0), matrix=Matrix.Rotation(a, 3, 'Y'), verts=r.verts)
        bmesh.ops.translate(r, vec=Vector((0.5 * math.sin(a), -0.1, 0.5 * math.cos(a))), verts=r.verts)
        parts.append(r)
    ring = bmesh.new()
    loops = []
    for i in range(32):
        t = 2 * math.pi * i / 32
        loop = []
        for j in range(8):
            u = 2 * math.pi * j / 8
            rr = 0.82 + 0.1 * math.cos(u)
            loop.append(ring.verts.new((rr * math.sin(t), -0.1 * math.sin(u) - 0.02, rr * math.cos(t))))
        loops.append(loop)
    for i in range(32):
        for j in range(8):
            a_, b_ = loops[i], loops[(i + 1) % 32]
            ring.faces.new((a_[j], a_[(j + 1) % 8], b_[(j + 1) % 8], b_[j]))
    parts.append(ring)
    for part in parts:
        tmp = bpy.data.meshes.new('part')
        part.to_mesh(tmp)
        part.free()
        bm.from_mesh(tmp)
        bpy.data.meshes.remove(tmp)
    return bm


# ---------------------------------------------------------------- the collar
# A stand collar in blackened steel, shaped as a tailor cuts one: a single band standing up from the neckline (its
# foot inside the cape's neck, just over the robe's own collar), leaning in toward the neck as it rises (the band's
# top edge is shorter than its foot), higher behind than in front, open at the throat with its top corners rounded
# off; its top and front edges rolled in brass, and a brass bar hooking the two fronts together.
GC_Y = 0.35
COLLAR_OPEN = 0.2     # (the half-angle of the opening at the throat)
COLLAR_FOOT = 52.9
G_NECK_R = [(0.0, 2.72), (45.0, 2.95), (90.0, 2.82), (135.0, 2.55), (180.0, 2.45)]   # (the foot's radius round from
# the front, in degrees: just over the robe's standing collar, which is fullest over the collarbones)


def g_neck(a):
    d = abs(math.degrees(wrap(a)))
    for (d0, r0), (d1, r1) in zip(G_NECK_R, G_NECK_R[1:]):
        if d <= d1:
            return lerp(r0, r1, smoothstep(d0, d1, d))
    return G_NECK_R[-1][1]


COLLAR_BASE = None   # (the cape's outer radius round the neck at the collar's foot, once the cape is built)


def collar_top(a):
    """The band's top edge: up under the jaw at the sides, highest at the nape."""
    return 55.1 + 0.6 * ((1.0 - math.cos(a)) / 2) ** 1.2


COLLAR_V = (51.55, 0.233)   # (the V the collar's fronts open in: the height its two edges meet at on the chest, and
# how fast they part above it, as an angle round the neck for each unit of height)


def collar_v(z):
    return max(0.0, COLLAR_V[1] * (z - COLLAR_V[0]))


def collar_open(v):
    """The half-angle of the opening at the throat, up the band. The fronts open in a V: one straight line each
    side from the point on the chest (see build_throat_plate, whose fronts carry it down there) up to the band's
    top corners under the jaw, so from the front the collar's outline is a V and not a ring."""
    z = 53.0 + 0.5 * v / FOOT_V if v <= FOOT_V else lerp(COLLAR_STAND, 55.2, (v - FOOT_V) / (1.0 - FOOT_V))
    return collar_v(z)


def collar_base(a):
    n = len(COLLAR_BASE)
    f = (a % (2 * math.pi)) / (2 * math.pi) * n
    i = int(f) % n
    return lerp(COLLAR_BASE[i], COLLAR_BASE[(i + 1) % n], f - int(f))


COLLAR_ON = None     # (a BVH of the cape and robe, for the collar's foot to lie on)
COLLAR_STAND = 53.5  # (where the band turns upright, at the throat)
FOOT_A0, FOOT_A1 = 0.62, 1.0   # (the foot spreads onto the cape only toward the yoke's edge, at about 0.97, and beyond)
FOOT_V = 0.36        # (the part of the band, from its foot, that lies down over the cape)


def cape_top_z(x, y):
    """(the highest the cape or robe stands near (x, y): the pleats' crests over the shoulders included)"""
    best = None
    for dx, dy in ((0, 0), (0.25, 0), (-0.25, 0), (0, 0.25), (0, -0.25), (0.18, 0.18), (-0.18, -0.18), (0.18, -0.18),
                   (-0.18, 0.18)):
        hit = COLLAR_ON.ray_cast(Vector((x + dx, y + dy, 60.0)), Vector((0, 0, -1)), 12.0)
        if hit[0] is not None:
            best = hit[0].z if best is None else max(best, hit[0].z)
    return COLLAR_FOOT if best is None else best


def collar_pt(a, v, out=0.0):
    """The band at a (round from the front) and v (0 at its foot, 1 at its top): its foot spreads over the cape
    round the neck and lies on it (covering where the cape meets the neck, the rim tucked down onto the cloth),
    turns up round the neck, and stands, flaring out again to its top edge, as the reference's. Toward the opening
    at the throat there is no cape to lie on: there the foot drops straight down, a plain front edge that lies on
    the throat plate."""
    neck = g_neck(a) + 0.16
    rim = min(max(collar_base(a) + 0.08, neck + 0.3), neck + 0.8)
    sa, ca = math.sin(a), math.cos(a)
    k = smoothstep(FOOT_A0, FOOT_A1, abs(wrap(a)))
    if v <= FOOT_V:
        t = v / FOOT_V
        r = lerp(neck, lerp(rim, neck, 1.0 - (1.0 - t) ** 2), k)
        # (over the yoke the foot has no lip: its lower edge sinks into the yoke's surface, so the two read as one
        # piece of steel with no step between them)
        r -= 0.12 * (1.0 - k) * (1.0 - smoothstep(0.0, 1.0, t))
        x, y = -r * sa, GC_Y - r * ca
        z = lerp(53.0, COLLAR_STAND, t)
        if k > 0.0:
            zt = cape_top_z(x, y) + 0.26
            # (easing into the upright: the last of the foot rises off the cloth into the band's curve)
            z = lerp(z, zt + smoothstep(0.6, 1.0, t) * 0.25, k)
        r += out
        return Vector((-r * sa, GC_Y - r * ca, z))
    w = (v - FOOT_V) / (1.0 - FOOT_V)
    zb = COLLAR_STAND
    if k > 0.0:
        zb = lerp(zb, cape_top_z(-neck * sa, GC_Y - neck * ca) + 0.45, k)
    z = lerp(zb, collar_top(a), w)
    r = neck + 0.45 * w ** 1.6 + out
    return Vector((-r * sa, GC_Y - r * ca, z))


def cape_base(n=120):
    """The cape's outer radius round the neck from the collar's foot up past the cape's top (the farthest of it
    along each ray in, the folds' crests softened), for the collar's foot to cover."""
    import numpy as np
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    vs, ps = [], []
    for o in bpy.data.objects:
        if o.type != 'MESH' or not o.name.startswith(('Cape', 'Robe')) or o.name == 'Cape stand-in':
            continue
        me = o.evaluated_get(dg).to_mesh()
        b = len(vs)
        vs += [o.matrix_world @ v.co for v in me.vertices]
        ps += [[b + i for i in pl.vertices] for pl in me.polygons]
        o.evaluated_get(dg).to_mesh_clear()
    T = BVHTree.FromPolygons(vs, ps)
    global COLLAR_ON
    COLLAR_ON = T
    R = np.zeros(n)
    for j in range(n):
        a = 2 * math.pi * j / n
        d = Vector((math.sin(a), math.cos(a), 0.0))     # (inward, from outside)
        best = 0.0
        for dz in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9):
            org = Vector((0.0, GC_Y, COLLAR_FOOT + dz)) - d * 12.0
            hit = T.ray_cast(org, d, 12.0)
            if hit[0] is not None:
                best = max(best, 12.0 - hit[3])
        R[j] = best
    M = np.maximum.reduce([np.roll(R, k) for k in range(-2, 3)])
    for _ in range(3):
        M = (np.roll(M, 1) + M + np.roll(M, -1)) / 3
    return list(M)


def build_collar(body):
    global COLLAR_BASE
    COLLAR_BASE = cape_base()
    n, rows = 120, 16

    def ang(v, j):
        o = collar_open(v)
        return lerp(o, 2 * math.pi - o, j / (n - 1))

    def front(i, j):
        # (the fronts' plain edge below the rolled brass is feathered down into the throat plate, so it has no
        # thickness there to draw a line with)
        v = 1.0 - i / (rows - 1)
        e = (1.0 - smoothstep(0.0, 5.0, min(j, n - 1 - j))) * (1.0 - smoothstep(0.42, 0.58, v))
        return collar_pt(ang(v, j), v, -0.13 * e)
    sheet = grid(rows, n, front)
    mesh_obj('Collar', face_out(sheet, axis_y=GC_Y), M_COLLAR, body, solid=0.09, subdiv=1)
    # the rolled edge: up one front, round the top and down the other
    vs_ = [0.66 + 0.34 * k / 6 for k in range(6)]   # (the fronts' rolled edge only above the throat plate, which
    # is joined to them below it)
    path = [collar_pt(ang(v, 0), v, 0.03) for v in vs_]
    path += [collar_pt(ang(1.0, j), 1.0, 0.03) for j in range(n)]
    path += [collar_pt(ang(v, n - 1), v, 0.03) for v in reversed(vs_)]
    mesh_obj('Collar edge', tube(path, [0.1] * len(path), flat=1.0, n=8), M_GOLD, body, subdiv=0)
    # (and the foot's rim rolled too, lying on the cloth)
    rim = [collar_pt(ang(0.0, j), 0.0, 0.02) for j in range(n)]
    # (it thins away to nothing over the yoke, where the foot sinks into it)
    rr = [0.1 * smoothstep(FOOT_A0, FOOT_A1, abs(wrap(ang(0.0, j)))) + 0.005 for j in range(n)]
    mesh_obj('Collar rim', tube(rim, rr, flat=1.0, n=8), M_COLLAR, body, subdiv=0)
    # (a line tooled round it below the top, and a row of studs between)
    line = [collar_pt(ang(0.78, j), 0.78, 0.035) for j in range(3, n - 3)]
    mesh_obj('Collar line', tube(line, [0.045] * len(line), flat=1.0, n=6), M_GOLD, body, subdiv=0)
    studs = [collar_pt(ang(0.89, j), 0.89, 0.06) for j in range(4, n - 4, 6)]
    mesh_obj('Collar studs', dots(studs, 0.14), M_GOLD, body, subdiv=0)
    build_throat_plate(body)


def build_throat_plate(body):
    """The collar's fronts, and the plate at the throat. Below the band the collar does not close into a plate: each front runs
    on down as a tapering strip of the same steel, leaving the ring's curve to lie on the chest, the two drawing
    together to a point on the breastbone, so the opening between them is a V from the jaw to the chest. Both
    edges of each front are rolled in brass, and a small gold medallion sits on the point. In the V, on the neck
    itself, a plate of the same steel fills the V's base, behind the fronts."""
    W0 = 1.05                 # (the fronts' outer edge at their top: at the collar's foot beside the opening)
    ZT, ZP, RP = 53.3, 51.0, 0.14

    def lay(a, z, rad, out=0.0):
        """(a smooth surface, with no ray casts in it: the ring's curve where it leaves the collar, coming forward
        as it falls to lie a little off the tabard over the chest)"""
        r = g_neck(a) + rad
        x, yr = -r * math.sin(a), GC_Y - r * math.cos(a)
        curve = yr - (GC_Y - (g_neck(0.0) + rad))            # (how far the ring curves back at this point)
        lie = tab_y(0.0, min(z, 52.2)) - 0.3 + 0.75 * curve
        # (round the neck itself it keeps to the ring: the chest only takes it over below the neckline)
        lie = lerp(yr + 1.0, lie, smoothstep(53.0, 52.3, z))
        # (a soft minimum: the ring above, the chest below, eased into each other)
        d = yr - lie
        y = min(yr, lie) - 0.12 * math.exp(-(d / 0.35) ** 2)
        return Vector((x, y - out, z))
    rows, cols = 30, 12

    def front(i, j, sg, out=0.0):
        """One front: its top tucked up under the collar's foot, inside it, so it comes out from under the band;
        its inner edge the V's line, its outer edge running in from the collar's foot to the same point."""
        t = i / (rows - 1)
        z = lerp(ZT, ZP, t)
        a = lerp(collar_v(z), W0 * (1.0 - t) + 0.004, j / (cols - 1))
        return lay(sg * a, z, lerp(0.06, RP - 0.04, smoothstep(0.0, 0.4, t)), out)
    for sg, nm in ((1, 'a'), (-1, 'b')):
        mesh_obj(f'Collar front {nm}', face_out(grid(rows, cols, lambda i, j, sg=sg: front(i, j, sg))), M_COLLAR, body,
                 solid=0.08, subdiv=1)
        # (the brass edges swell in from nothing below the collar's rim, so no stub pokes out above it)
        # (the inner edge is joined to the throat plate, so only the outer one is rolled)
        for jj, en in ((cols - 1, 'outer'),):
            i0 = 5
            pe = [front(i, jj, sg, 0.03) for i in range(i0, rows)]
            mesh_obj(f'Collar front edge {en} {nm}',
                     tube(pe, [0.07 * smoothstep(0, 4, k) + 0.008 for k in range(len(pe))], flat=1.0, n=8),
                     M_GOLD, body, subdiv=0)
    tip = lay(0.0, ZP + 0.42, RP, 0.2)
    mb = medallion()
    bmesh.ops.scale(mb, vec=(0.5, 0.5, 0.5), verts=mb.verts)
    # (moved into place in the mesh itself: the collar's pieces are later rebuilt from their vertices alone, which
    # dropped an object's own location and left the medallion at the origin)
    bmesh.ops.translate(mb, vec=tip, verts=mb.verts)
    mesh_obj('Collar medallion', mb, M_GOLD, body, subdiv=1)

    # ---- the throat plate: the V's base filled with steel, one piece with the collar. Its sides are the band's own
    # surface up to the band's front edges, and below the band the fronts' own surface up to their inner edges, so
    # band, fronts and plate read as one shaped piece of steel tapering to the point on the chest, with no seam
    # between them; only its top edge, the bottom of the opening under the chin, is rolled in brass
    PT = 54.25

    def ptop(a):
        return PT - 0.3 * math.exp(-(a / 0.28) ** 2)

    def band_v(z):
        """(the band's v at height z, at its front edge: the inverse of collar_open's mapping)"""
        if z <= COLLAR_STAND:
            return max(0.0, (z - 53.0) / 0.5) * FOOT_V
        return FOOT_V + (z - COLLAR_STAND) / (55.2 - COLLAR_STAND) * (1.0 - FOOT_V)

    def tplate(i, j, out=0.0):
        t = i / (rows - 1)
        u = lerp(-1.0, 1.0, j / (cols2 - 1))
        a0 = collar_v(lerp(PT, ZP, t))
        z = lerp(ptop(u * a0), ZP + 0.25, t)
        # (a hair past the edges it joins, and a hair behind them, so no crack opens along the join)
        a = u * (collar_v(z) + 0.03)
        if z >= ZT:
            return collar_pt(a, band_v(z), out - 0.01)
        tf = (ZT - z) / (ZT - ZP)
        return lay(a, z, lerp(0.06, RP - 0.04, smoothstep(0.0, 0.4, tf)) - 0.01, out)
    cols2 = 25
    mesh_obj('Collar throat plate', face_out(grid(rows, cols2, tplate)), M_COLLAR, body, solid=0.08, subdiv=1)
    te = [tplate(0, j, 0.03) for j in range(cols2)]
    mesh_obj('Collar throat plate edge', tube(te, [0.07] * cols2, flat=1.0, n=8), M_GOLD, body, subdiv=0)
    mesh_obj('Collar throat plate rivets', dots([tplate(4, j, 0.06) for j in (8, 16)], 0.15), M_GOLD, body, subdiv=0)


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
            # (the parting and the combed grooves are pressed in after the fusion, which would fill them: cap_grooves)
            r = rs - 0.1 * (1.0 - edge) + 0.02 + edge * (0.14 + 0.12 * max(0.0, d.z) + full) + swell + 0.0 * (combed + part)
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


def cap_grooves(co):
    """How far the parting and the combed grooves press into the fused hair at co (head space): the cap's own
    grooves, found from the direction of co from the skull's centre."""
    d = (co - HC)
    if d.length < 1e-6:
        return 0.0
    d = d.normalized()
    th = math.atan2(-d.x, -d.y)
    dp_ = wrap(th)
    el = math.asin(max(-1.0, min(1.0, d.z)))
    el0 = HAIRLINE(abs(dp_))
    t = max(0.0, min(1.0, (el - el0) / max(1e-3, 1.5 - el0)))
    edge = smoothstep(0.0, 0.2, t)
    front = 1.0 - smoothstep(0.15, 0.55, t)
    near_part = math.exp(-(dp_ / 0.085) ** 2)
    part = -0.13 * near_part * front * edge
    psi = math.atan2(d.z, -d.y)
    combed = (0.05 * math.sin(10.0 * psi + 0.6) + 0.025 * math.sin(24.0 * psi + 1.3)) * edge * \
        (1.0 - smoothstep(1.9, 2.7, abs(dp_)))
    # (the comb lines all meet at the side of the head, over the ear: faded out well before)
    return (part + combed * (1.0 - smoothstep(0.45, 0.75, abs(d.x)))) * smoothstep(-0.2, 0.15, el)


def side_lock(s):
    """The hair falling down one side of the face from the parting, over the ear and on down past the jaw to the
    shoulder, behind the line of the cheek; it waves as it falls."""
    def wv(z):
        return s * 0.14 * math.sin(1.25 * z + 0.4)
    # (shallow behind, so its back edge stays forward of the fall's side and the two round into one another)
    return loft([(7.4, 0.7, 1.4, s * 2.75 + wv(7.4), -0.05), (5.0, 0.95, 1.85, s * 3.2 + wv(5.0), 0.7),
                 (2.5, 1.0, 1.6, s * 3.2 + wv(2.5), 1.25), (0.0, 0.95, 1.4, s * 3.35 + wv(0.0), 1.65),
                 (-1.6, 0.8, 1.15, s * 3.5 + wv(-1.6), 2.1), (-2.6, 0.7, 0.85, s * 3.5 + wv(-2.6), 2.8),
                 (-3.4, 0.55, 0.55, s * 3.55, 3.35), (-4.2, 0.4, 0.35, s * 3.5, 3.6), (-5.0, 0.2, 0.15, s * 3.4, 3.8)])
    # (below the jaw it sweeps back behind the shoulder and becomes the fall's front edge: ending over the shoulder
    # it stood as a blob whose underside showed from the front)


def drape_fall(dg):
    """The long hair down her back (the ranger's drape, wavier and a little fuller): for each strip across it, the
    surface it rests on is found by casting at the figure from behind, and the hair lies on that, then hangs,
    spanning the hollows rather than sinking into them. Its sides wave in and out as it falls; its end is uneven,
    a few locks longer than the rest."""
    from mathutils.bvhtree import BVHTree
    off = Vector((0, 0, HEAD_Z))
    vs, ps = [], []
    for o in bpy.data.objects:
        if o.type != 'MESH' or not ((o.parent is not None and o.parent.name == 'Body') or
                                    o.name in ('Head shape', 'Hair fall', 'Hair cap') or
                                    o.name.startswith(('Robe', 'Belt', 'Breastplate', 'Gambeson', 'Neck', 'Tabard', 'Cape'))):
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

    W = [(7.2, 2.4), (6.0, 3.5), (4.5, 4.0), (1.5, 4.3), (-1.5, 4.55), (-4.5, 4.7), (-9.0, 4.3), (-13.0, 3.65), (-16.0, 3.1),
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
        top_ = list(range(len(pts)))   # (all the way down: the ends span the belt rather than stepping over it)
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
            y = max(lerp(y - 1.0, sup + 0.6, wb), y)
            gap = y - sup
            th = t if gap < t + 0.3 else max(t, gap - 0.15)
            yi = lerp(y - th, max(y - th, sup + 0.3), wb)
            if abs(u) > 0.45 and y - yi > 0.3:
                f_ = min(1.0, (abs(u) - 0.45) / 0.55)
                y = yi + max(0.3, (y - yi) * math.sqrt(max(0.0, 1.0 - f_ * f_)))   # (no thinner than 0.3)
            col_out.append(Vector((x, y, z)))
            col_in.append(Vector((x, yi, z)))
        inner.append(col_in)
        outer.append(col_out)
    # across the fall, ease each row's depth over its neighbours: each strip spans the back on its own, and side by
    # side their small differences stand as vertical grooves
    for j in range(NZ):
        ys = [outer[i][j].y for i in range(NX)]
        for i in range(NX):
            if outer[i][j].z > 0.5:
                continue
            lo_, hi_ = max(0, i - 30), min(NX, i + 31)
            w_ = [math.exp(-((k - i) / 12.0) ** 2) for k in range(lo_, hi_)]
            dy_ = sum(ys[k] * wk for k, wk in zip(range(lo_, hi_), w_)) / sum(w_) - ys[i]
            f_ = smoothstep(0.5, -1.5, outer[i][j].z) * (1.0 - smoothstep(0.85, 1.0, abs(-1 + 2 * i / (NX - 1))))
            outer[i][j].y = max(outer[i][j].y + dy_ * f_, inner[i][j].y + 0.15)
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


def round_back(bm):
    """Round the hair's cross-section where the side hair meets the fall, below the skull. Fused, the side locks and
    the fall make a box there: flat sides, a flat back and a corner between them that reads as a seam from behind
    and at three-quarters. Behind the side hair's widest point, draw the outside in to an ellipse that runs from that
    width round to the middle of the back, so the side flows round into the back."""
    from mathutils.bvhtree import BVHTree
    bm.verts.ensure_lookup_table()
    tree = BVHTree.FromBMesh(bm)
    ZS = [5.0 - 0.25 * k for k in range(61)]          # 5 down to -10
    YS = [-1.0 + 0.1 * k for k in range(81)]          # -1 to 7
    K = {}
    for side in (-1, 1):
        raw = []
        for zi, z in enumerate(ZS):
            back = tree.ray_cast(Vector((0.0, 30.0, z)), Vector((0, -1, 0)), 40.0)[0]
            outs = []
            for y in YS:
                h = tree.ray_cast(Vector((side * 20.0, y, z)), Vector((-side, 0, 0)), 40.0)[0]
                outs.append(abs(h.x) if h is not None and h.x * side > 0 else 0.0)
            row = [1.0] * len(YS)
            if back is not None:
                yb, yc = back.y, 1.2
                xm = max(o_ for y_, o_ in zip(YS, outs) if 0.4 <= y_ <= 1.6)
                for yi, y in enumerate(YS):
                    if y <= yc or outs[yi] <= 1.6 or xm <= 1.6:
                        continue
                    q = min(1.0, (y - yc) / max(0.5, yb - yc))
                    target = max(1.6, xm * math.sqrt(max(0.0, 1.0 - q * q)))
                    row[yi] = min(1.0, (target - 1.6) / (outs[yi] - 1.6))
            raw.append(row)
        # eased across heights and depths, so neighbouring rows never step
        for zi in range(len(ZS)):
            for yi in range(len(YS)):
                acc = wsum = 0.0
                for dz in range(-4, 5):
                    for dy in range(-3, 4):
                        a_, b_ = zi + dz, yi + dy
                        if 0 <= a_ < len(ZS) and 0 <= b_ < len(YS):
                            wt = math.exp(-(dz / 2.5) ** 2 - (dy / 2.0) ** 2)
                            acc += wt * raw[a_][b_]
                            wsum += wt
                K[(side, zi, yi)] = acc / wsum
    for v in bm.verts:
        x, y, z = v.co
        w = smoothstep(5.0, 3.0, z) * smoothstep(-10.0, -7.0, z)
        if w <= 0.0 or abs(x) <= 1.6:
            continue
        side = 1 if x > 0 else -1
        fz = (5.0 - z) / 0.25
        fy = (y + 1.0) / 0.1
        z0, y0 = int(fz), int(fy)
        acc = wsum = 0.0
        for dz in (0, 1):
            for dy in (0, 1):
                wt = (1 - abs(fz - z0 - dz)) * (1 - abs(fy - y0 - dy))
                acc += wt * K.get((side, z0 + dz, y0 + dy), 1.0)
                wsum += wt
        k = acc / wsum if wsum else 1.0
        if k < 1.0:
            v.co.x = side * (1.6 + (abs(x) - 1.6) * lerp(1.0, k, w))


def hair_join_weight(co):
    """Where the side hair joins the fall (head space): behind the ears, from the temples down past the nape."""
    x, y, z = co
    return smoothstep(6.0, 4.0, z) * smoothstep(-8.0, -5.0, z) * smoothstep(1.0, 2.0, abs(x)) * smoothstep(0.6, 1.6, y)


CLOTHES = ('Robe', 'Breastplate', 'Belt', 'Neck', 'Gambeson', 'Tabard', 'Cape', 'Medallion', 'Clasp', 'Collar')   # (not the swinging arms)


def clothes_tree(space, dg):
    """A BVH of the clothes (and neck) as they are now, in the given object's space."""
    from mathutils.bvhtree import BVHTree
    inv = space.matrix_world.inverted()
    vs, ps = [], []
    for o in bpy.data.objects:
        if o.type != 'MESH' or not o.name.startswith(CLOTHES):
            continue
        me = o.evaluated_get(dg).to_mesh()
        b = len(vs)
        vs += [inv @ (o.matrix_world @ v.co) for v in me.vertices]
        ps += [[b + i for i in p.vertices] for p in me.polygons]
        o.evaluated_get(dg).to_mesh_clear()
    return BVHTree.FromPolygons(vs, ps) if ps else None


def lift_needed(bm, tree, gap=0.22, need=None):
    """Where the hair's underside meets the clothes (or comes within `gap` of them), the move that clears it, by
    vertex: {index: displacement}. Outward from the underside through the hair, the last of the clothes it passes
    (a collar standing up inside the hair is passed twice); else, over the shoulders, where the clothes lie below
    the hair rather than inside it, off the nearest of their outward faces, along its normal."""
    need = need if need is not None else {}
    bm.normal_update()
    bm.verts.ensure_lookup_table()
    for v in bm.verts:
        if v.co.z > 1.0:
            continue
        rad = Vector((v.co.x, v.co.y - HC.y, 0.0))
        if rad.length < 1e-6:
            continue
        rad.normalize()
        d_ = None
        if v.normal.dot(rad) < -0.2:
            o_, far = v.co.copy(), None
            for _k in range(6):
                h_ = tree.ray_cast(o_ + rad * 0.003, rad, 1.6 - (o_ - v.co).length)
                if h_[0] is None:
                    break
                far = o_ = h_[0]
            if far is not None:
                d_ = rad * ((far - v.co).dot(rad) + gap)
        if d_ is None:
            p_, n_, _i, _dist = tree.find_nearest(v.co, gap + 0.6)
            if p_ is None or v.normal.dot(n_) > -0.2 or (n_.dot(rad) < 0.0 and n_.z < 0.3):
                continue
            d = (v.co - p_).dot(n_)
            if d >= gap:
                continue
            d_ = n_ * (gap - d)
        if d_.length > need.get(v.index, Vector()).length:
            need[v.index] = d_.copy()
    return need


def apply_lift(bm, need, reach=1.6):
    """Move the hair by the needed displacements, spread to everything within `reach` of a vertex that needed it
    (the whole thickness of the hair there, so it keeps its shape) and eased off beyond."""
    from mathutils.kdtree import KDTree
    if not need:
        return
    bm.verts.ensure_lookup_table()
    # (measured with height counted at SQ of its length, so a lift spreads up and down the hair further than across
    # it: over the collar's top or the shoulder it eases in over a hand's breadth instead of standing as a ring)
    SQ = 0.4
    sq = lambda c: Vector((c.x, c.y, c.z * SQ))     # noqa: E731
    # (the displacements are pooled in small cells first: neighbouring ones barely differ, and summing every pair
    # over this reach cost most of the build)
    CELL = 0.3
    cells = {}
    for i, d in need.items():
        c = sq(bm.verts[i].co)
        key = (math.floor(c.x / CELL), math.floor(c.y / CELL), math.floor(c.z / CELL))
        e = cells.get(key)
        if e is None:
            cells[key] = [c.copy(), d.copy(), 1]
        else:
            e[0] += c
            e[1] += d
            e[2] += 1
    pts = [(c / n, d / n, n) for c, d, n in cells.values()]
    kd = KDTree(len(pts))
    for k, (p, _d, _n) in enumerate(pts):
        kd.insert(p, k)
    kd.balance()
    moves = []
    for v in bm.verts:
        near = kd.find_range(sq(v.co), reach * 2.0)
        if not near:
            continue
        acc, ws, wmax = Vector(), 0.0, 0.0
        for _co, k, dist in near:
            w = math.exp(-(dist / reach) ** 2)
            acc += pts[k][1] * (w * pts[k][2])
            ws += w * pts[k][2]
            wmax = max(wmax, w)
        moves.append((v, acc / ws * min(1.0, wmax * 1.6)))
    for v, m in moves:
        v.co += m


def keep_off_clothes(bm, head, gap=0.22):
    """At rest, before the rig: the fusion's fillets and the smoothing can let the underside sag into the robe."""
    tree = clothes_tree(head, bpy.context.evaluated_depsgraph_get())
    if tree is not None:
        apply_lift(bm, lift_needed(bm, tree, gap))
        bmesh.ops.smooth_laplacian_vert(bm, verts=[v for v in bm.verts if v.co.z < 1.0], lambda_factor=0.3,
                                        lambda_border=0.0, use_x=True, use_y=True, use_z=True, preserve_volume=True)


def hair_clearance(cols=range(10), gap=0.22):
    """After the rig and the walk: the hair rides on the head, the clothes on the trunk and arms, so in every pose
    of the sheet find where the hair would meet the clothes, and lift it clear of the worst of them."""
    sc = bpy.context.scene
    hair = bpy.data.objects['Hair']
    bm = bmesh.new()
    bm.from_mesh(hair.data)
    need = {}
    for col in cols:
        sc.frame_set(col + 1)
        tree = clothes_tree(hair, bpy.context.evaluated_depsgraph_get())
        if tree is not None:
            lift_needed(bm, tree, gap, need)
    sc.frame_set(1)
    apply_lift(bm, need)
    # the lift bends the hair where it meets the shoulders and the collar; ease that bend into a roll, then lift
    # again wherever the easing let it back down
    ever = set(need)
    for _round in range(3):
        bm.verts.ensure_lookup_table()
        reg = {bm.verts[i] for i in ever}
        for _ in range(8):
            reg |= {e.other_vert(v) for v in list(reg) for e in v.link_edges}
        reg = [v for v in reg if v.co.z < 1.0]
        for _ in range(16):
            bmesh.ops.smooth_laplacian_vert(bm, verts=reg, lambda_factor=0.5, lambda_border=0.0,
                                            use_x=True, use_y=True, use_z=True, preserve_volume=True)
        need = {}
        for col in cols:
            sc.frame_set(col + 1)
            tree = clothes_tree(hair, bpy.context.evaluated_depsgraph_get())
            if tree is not None:
                lift_needed(bm, tree, gap, need)
        sc.frame_set(1)
        apply_lift(bm, need)
        ever |= set(need)
    # a last light easing of what the final lift moved (the margin over the clothes allows for it)
    bm.verts.ensure_lookup_table()
    reg = {bm.verts[i] for i in need}
    for _ in range(4):
        reg |= {e.other_vert(v) for v in list(reg) for e in v.link_edges}
    for _ in range(4):
        bmesh.ops.smooth_laplacian_vert(bm, verts=list(reg), lambda_factor=0.4, lambda_border=0.0,
                                        use_x=True, use_y=True, use_z=True, preserve_volume=True)
    bm.to_mesh(hair.data)
    bm.free()
    hair.data.update()


# ---------------------------------------------------------------- the hair designed in the face creator
# The face creator's "Warden (long, wavy, centre parting)" (HAIR_STYLES.wavyFull), exported by
# tools/face-creator/export-hair.cjs into hair_creator.json.gz beside this file: the cap (the opaque head of hair)
# as a grid, and every strand of every lock as a centre line with its width, thickness, shade and cut-out, in head
# heights. Head space is the creator's own (x across, y back, z up from the chin), HEAD_H to the head.
HAIR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hair_creator.json.gz')
HAIR_PTS = None      # (every point the clearance moves: the cap's vertices, then the strands' centre lines)
HAIR_RUNS = None     # (the strands' runs of points in HAIR_PTS: (first, count))


HAIR_TIPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hair_tips.png')
HAIR_TEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hair_tex.png')     # (the creator's strand colours)
HAIR_BUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hair_bump.png')   # (and its strand bump)
TIP_GAIN = 1.4      # (the creator's tipGain, between 1 up close and 2.5 far off: the head views sit between)
# (the 3D views' lights and tone mapping lift a dark colour about five times as much as the creator's dim viewport
# does: her hair is taken down to this much of its colour so she reads as dark as the creator's Warden; with the
# creator's own HAIR_SHADE, 0.8, on top)
HAIR_ONSCREEN = 0.45


def hair_tip_cut(objs):
    """The creator's layered hair ends on the Blender hair: its hair-tip map (red each hair's tone, green the top
    layer's cut-out, blue the wider hairs of the layers under it), read three times as the creator's shader reads it.
    The top layer; a second shifted round the head and ending a little higher; a third higher again. Through a gap
    between two strands of one you see the next, so the strands join the mass, and skin shows only where all three
    have run out, toward the very tips. A hard cut, as the creator's alphaTest. (Wrapped round the material's own
    output, so the sprite sheet, which wires its own output, keeps solid hair.)"""
    src = objs[0].data.materials[0]
    m = src.copy()
    m.name = 'hair cut'
    nt = m.node_tree
    img = bpy.data.images.load(HAIR_TIPS, check_existing=True)
    img.colorspace_settings.name = 'Non-Color'
    uvn = nt.nodes.new('ShaderNodeUVMap')
    uvn.uv_map = 'tips'
    sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(uvn.outputs['UV'], sep.inputs['Vector'])

    def m_(op, a, b):
        n = nt.nodes.new('ShaderNodeMath')
        n.operation = op
        for i, x in enumerate((a, b)):
            if isinstance(x, (int, float)):
                n.inputs[i].default_value = x
            else:
                nt.links.new(x, n.inputs[i])
        return n.outputs[0]

    def read(du, dv, ch):
        u = m_('ADD', sep.outputs['X'], du)
        v = m_('MINIMUM', m_('MAXIMUM', m_('SUBTRACT', sep.outputs['Y'], dv), 0.002), 0.998)
        cmb = nt.nodes.new('ShaderNodeCombineXYZ')
        nt.links.new(u, cmb.inputs['X'])
        nt.links.new(v, cmb.inputs['Y'])
        tx = nt.nodes.new('ShaderNodeTexImage')
        tx.image = img
        tx.extension = 'REPEAT'
        tx.interpolation = 'Linear'
        nt.links.new(cmb.outputs['Vector'], tx.inputs['Vector'])
        rgb = nt.nodes.new('ShaderNodeSeparateColor')
        nt.links.new(tx.outputs['Color'], rgb.inputs['Color'])
        return rgb.outputs[ch]

    a = m_('MAXIMUM', read(0.0, 0.0, 'Green'), m_('MAXIMUM', read(0.3713, 0.035, 'Blue'), read(0.7129, 0.08, 'Blue')))
    keep = m_('GREATER_THAN', m_('MULTIPLY', a, TIP_GAIN), 0.5)
    clear = nt.nodes.new('ShaderNodeBsdfTransparent')
    for out in [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL']:
        if not out.inputs['Surface'].is_linked:
            continue
        sh = out.inputs['Surface'].links[0].from_socket
        mix = nt.nodes.new('ShaderNodeMixShader')
        mix.name = 'tip cut'      # (sprite_mode takes it out of the way for the sheet, and puts it back)
        nt.links.new(keep, mix.inputs['Fac'])
        nt.links.new(clear.outputs['BSDF'], mix.inputs[1])
        nt.links.new(sh, mix.inputs[2])
        nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
    if hasattr(m, 'blend_method'):
        m.blend_method = 'CLIP'
    # the surface itself, as the creator's: its painted strand texture (fine strands in clumps of their own tone,
    # dark partings between them, a few strands catching the light) times each strand's shade, matte (roughness 1,
    # no sheen: the warm anisotropic sheen of the Ranger-tuned close-up lit every strand's edge into a line and read
    # as slabs), with the creator's strand bump
    hb = nt.nodes['hair bsdf']
    uvh = nt.nodes.new('ShaderNodeUVMap')
    uvh.uv_map = 'hair'
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images.load(HAIR_TEX, check_existing=True)
    tex.extension = 'REPEAT'
    nt.links.new(uvh.outputs['UV'], tex.inputs['Vector'])
    shade = nt.nodes.new('ShaderNodeVertexColor')
    shade.layer_name = 'strands'
    mul = nt.nodes.new('ShaderNodeVectorMath')
    mul.operation = 'MULTIPLY'
    nt.links.new(tex.outputs['Color'], mul.inputs[0])
    nt.links.new(shade.outputs['Color'], mul.inputs[1])
    k = nt.nodes.new('ShaderNodeVectorMath')
    k.operation = 'SCALE'
    k.inputs['Scale'].default_value = 0.8 * HAIR_ONSCREEN
    nt.links.new(mul.outputs['Vector'], k.inputs[0])
    nt.links.new(k.outputs['Vector'], hb.inputs['Base Color'])
    hb.inputs['Roughness'].default_value = 1.0
    hb.inputs['Specular IOR Level'].default_value = 0.1
    hb.inputs['Anisotropic'].default_value = 0.0
    hb.inputs['Sheen Weight'].default_value = 0.0
    bimg = nt.nodes.new('ShaderNodeTexImage')
    bimg.image = bpy.data.images.load(HAIR_BUMP, check_existing=True)
    bimg.image.colorspace_settings.name = 'Non-Color'
    bimg.extension = 'REPEAT'
    nt.links.new(uvh.outputs['UV'], bimg.inputs['Vector'])
    bmp = nt.nodes.new('ShaderNodeBump')
    bmp.inputs['Strength'].default_value = 0.35
    bmp.inputs['Distance'].default_value = 0.05
    nt.links.new(bimg.outputs['Color'], bmp.inputs['Height'])
    nt.links.new(bmp.outputs['Normal'], hb.inputs['Normal'])
    for o in objs:
        o.data.materials[0] = m


def creator_hair(head):
    """Build the exported hairstyle on the head: 'Hair' (the cap, cut off where its strands thin out toward its
    edge, so the locks make the outline) and 'Hair locks' (each strand a flat ribbon tapering to a point where the
    creator fades it out, at its root and its tip)."""
    import gzip
    import json
    global HAIR_PTS, HAIR_RUNS
    with gzip.open(HAIR_FILE, 'rt') as f:
        d = json.load(f)
    H = HEAD_H
    pts, runs = [], []
    # ---- the cap
    C = d['cap']
    nc, nr = C['cols'], C['rows']
    bm = bmesh.new()
    lay = bm.verts.layers.int.new('pt')      # (which point of HAIR_PTS each vertex follows)
    vs = [bm.verts.new(Vector(C['p'][k * 3:k * 3 + 3]) * H) for k in range(nc * nr)]
    pts += [v.co.copy() for v in vs]
    al = C['a']
    for i in range(nc):
        i2 = (i + 1) % nc
        for j in range(nr - 1):
            q = (i * nr + j, i2 * nr + j, i2 * nr + j + 1, i * nr + j + 1)
            if min(al[k] for k in q) < 0.3:
                continue
            if j == 0:
                continue      # (the crown: closed with a fan below)
            bm.faces.new([vs[k] for k in q])
    crown = bm.verts.new(sum((vs[i * nr].co for i in range(nc)), Vector()) / nc)
    for i in range(nc):
        bm.faces.new((crown, vs[((i + 1) % nc) * nr + 1], vs[i * nr + 1]))
    for i in range(nc):
        pts[i * nr] = crown.co.copy()
    bmesh.ops.delete(bm, geom=[vs[i * nr] for i in range(nc)], context='VERTS')
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    for i in range(nc):
        for j in range(1, nr):
            vs[i * nr + j][lay] = i * nr + j
    crown[lay] = 0
    # (the cap wears the hair-tip map too: round the head across it, its cut-out along it)
    uvl = bm.loops.layers.uv.new('tips')
    uvh = bm.loops.layers.uv.new('hair')      # (and its strand texture: round the head across, down the hair along)
    cu = C.get('u', [0.0] * (nc * nr))
    cv = C.get('v', [1.0] * (nc * nr))
    for f in bm.faces:
        ks = [v[lay] for v in f.verts]
        us = [cu[k] if v is not crown else None for k, v in zip(ks, f.verts)]
        real = [u for u in us if u is not None]
        wrap = max(real) - min(real) > 0.5       # (a face across the seam, where u comes round from 1 to 0)
        for lp, k, u in zip(f.loops, ks, us):
            if u is None:
                u = sum(real) / len(real)
            u = u + (1.0 if wrap and u < 0.5 else 0.0)
            lp[uvl].uv = (u, 1.0 if lp.vert is crown else al[k])
            lp[uvh].uv = (u, 1.0 if lp.vert is crown else cv[k])
    cap = mesh_obj('Hair', bm, M_HAIR, head, subdiv=0, solid=0.12, rest=hair_flow)
    cap.modifiers['thickness'].offset = -1.0     # (thickened inward, toward the scalp)
    sa = cap.data.color_attributes.new('strands', 'FLOAT_COLOR', 'POINT')
    sa.data.foreach_set('color', [1.0] * (len(sa.data) * 4))      # (the cap: its strand texture as it is)
    # ---- the locks, strand by strand, one object per layer of the creator's: 'Hair locks' the top layer (and the
    # pieces that have no layer, as the parting's), 'Hair locks under' the darker, broader layer in its gaps
    rmp = MATERIALS[M_HAIR]['ramp'] / 255.0
    lin = np.where(rmp > 0.04045, ((rmp + 0.055) / 1.055) ** 2.4, rmp / 12.92)
    RING = 8
    locks = []
    for layer, name in ((0, 'Hair locks'), (1, 'Hair locks under')):
        bm = bmesh.new()
        lay = bm.verts.layers.int.new('pt')
        uvl = bm.loops.layers.uv.new('tips')
        uvh = bm.loops.layers.uv.new('hair')
        shade = []
        for st in d['strands']:
            if st.get('layer', 0) != layer:
                continue
            n = len(st['t'])
            P = [Vector(st['p'][k * 3:k * 3 + 3]) * H for k in range(n)]
            W = [Vector(st['w'][k * 3:k * 3 + 3]) * H for k in range(n)]
            keep = [k for k in range(n) if st['a'][k] > 0.02]
            if len(keep) < 4:
                continue
            k0, k1 = keep[0], keep[-1]
            u0, u1 = st.get('u', (0.0, 0.03))
            base = len(pts)
            rings, along, down = [], [], []
            for k in range(k0, k1 + 1):
                # (the creator's own width: its tips are cut by the hair-tip map, as in the creator, so the strand
                # only narrows a little toward its faded ends instead of tapering to a point)
                a = min(1.0, st['a'][k])
                w = W[k] * (1.05 * a ** 0.25)
                T = (P[min(n - 1, k + 1)] - P[max(0, k - 1)]).normalized()
                N = T.cross(w)
                N = N.normalized() * max(0.03, st['t'][k] * H * 1.15) * a ** 0.25 if N.length > 1e-9 else Vector()
                # (a lens, not a diamond: a diamond's two sharp edges each caught the light as a bright line down
                # the strand and its flat faces read as slabs; round, its edges roll away into the strands beside it)
                ring = [bm.verts.new(P[k] + w * math.cos(q * math.pi / 4) + N * math.sin(q * math.pi / 4))
                        for q in range(RING)]
                for v in ring:
                    v[lay] = len(pts)
                shade += [st['c'][k] * (1.0 - 0.1 * abs(math.cos(q * math.pi / 4)) ** 4) for q in range(RING)]
                pts.append(P[k].copy())
                rings.append(ring)
                along.append(a)
                down.append(st['v'][k] if 'v' in st else 1.0 - k / n)
            runs.append((base, len(pts) - base))
            for r in range(len(rings) - 1):
                r0, r1 = rings[r], rings[r + 1]
                for q in range(RING):
                    f = bm.faces.new((r0[q], r0[(q + 1) % RING], r1[(q + 1) % RING], r1[q]))
                    # (round the tube across the map, as the creator's: u from the strand's first ring point to its
                    # last; along it the cut-out, 0 at the faded root and tip, 1 where the strand is whole)
                    for lp, (qq, rr) in zip(f.loops, ((q, r), (q + 1, r), (q + 1, r + 1), (q, r + 1))):
                        lp[uvl].uv = (u0 + (u1 - u0) * qq / RING, along[rr])
                        lp[uvh].uv = (u0 + (u1 - u0) * qq / RING, down[rr])
            for f in (bm.faces.new(rings[0][::-1]), bm.faces.new(rings[-1])):
                for lp in f.loops:
                    lp[uvl].uv = (u0, 0.0)
                    lp[uvh].uv = (u0, down[0])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        o = mesh_obj(name, bm, M_HAIR, head, subdiv=0, rest=hair_flow)
        # (each strand's own shade from the creator: lighter over a wave's crest, darker in its trough, the under
        # layer darker than the top one)
        attr = o.data.color_attributes.new('strands', 'FLOAT_COLOR', 'POINT')
        cols = []
        for c in shade:
            # (the creator's own shade on its strand texture: lighter over a wave's crest, darker in its trough)
            cols += [c, c, c, 1.0]
        attr.data.foreach_set('color', cols)
        locks.append(o)
    hair_tip_cut([cap] + locks)
    HAIR_PTS, HAIR_RUNS = pts, runs
    return cap, locks


def creator_hair_clearance(cols=range(10), gap=0.2):
    """After the rig and the walk: the creator's head has no shoulders, so its long hair falls straight through
    the collar, the cape and the shoulders. In each pose of the sheet, find the points of the hair that are inside
    the clothes or closer to them than `gap`, and the move that clears each (out past the cloth along the radius
    from the neck where that is near, as over the collar and down the back; else off the nearest cloth, as on the
    shoulders); take the largest over the poses, ease it along each strand and into the hair round it, and move
    the cap and the strands by it."""
    from mathutils.kdtree import KDTree
    sc = bpy.context.scene
    cap = bpy.data.objects['Hair']
    locks = [o for o in bpy.data.objects if o.name.startswith('Hair locks')]
    P = HAIR_PTS
    need = {}
    clear_ = [1e9] * len(P)      # how close each point comes to the clothes in any pose
    for col in cols:
        sc.frame_set(col + 1)
        tree = clothes_tree(cap, bpy.context.evaluated_depsgraph_get())
        if tree is None:
            continue
        for i, p in enumerate(P):
            if p.z > 1.0:
                continue
            _p, _n, _i, dn = tree.find_nearest(p, 3.0)
            if dn is not None and dn < clear_[i]:
                clear_[i] = dn
            rad = Vector((p.x, p.y - HC.y, 0.0))
            if rad.length < 1e-6:
                continue
            rad.normalize()
            d_ = None
            # over the shoulders the cloth lies below the hair: going up from a point inside it, the first face met is
            # one where the cloth's top faces up. Then the point comes up out of every layer there (robe, breastplate,
            # cape: the last upward-facing face it passes), off that face.
            up, o_, top_ = Vector((0, 0, 1)), p.copy(), None
            for _k in range(8):
                h_ = tree.ray_cast(o_ + up * 0.003, up, 1.5 - (o_ - p).length)
                if h_[0] is None:
                    break
                if _k == 0 and h_[1].z <= 0.2:
                    break
                o_ = h_[0]
                if h_[1].z > 0.2:
                    top_ = (o_.copy(), h_[1].copy())
            if top_ is not None:
                d_ = (top_[0] - p) + top_[1] * gap
            else:
                # out along the radius from the neck: the hair has to lie outside every cloth it passes within reach
                # (a collar standing up round the neck is passed in and out again), so the move is to the last face
                # it leaves; a point deep inside, as where a lock comes down into the shoulder, comes to rest on the
                # nearest cloth instead of being thrown out past the arm
                o_, far, inside = p.copy(), None, False
                for _k in range(8):
                    h_ = tree.ray_cast(o_ + rad * 0.003, rad, 1.6 - (o_ - p).length)
                    if h_[0] is None:
                        break
                    o_ = h_[0]
                    if h_[1].dot(rad) > 0.0:
                        far = o_
                        inside = inside or _k == 0
                if far is not None and (far - p).dot(rad) < 0.9:
                    d_ = rad * ((far - p).dot(rad) + gap)
                elif far is not None and inside:
                    p_, n_, _i, _dist = tree.find_nearest(p, 3.0)
                    if p_ is None:
                        continue
                    d_ = p_ - p
                    d_ = d_ + (d_.normalized() if d_.length > 1e-6 else n_) * gap
                else:
                    p_, n_, _i, _dist = tree.find_nearest(p, gap + 0.4)
                    if p_ is None:
                        continue
                    dd = (p - p_).dot(n_)
                    if dd >= gap or dd < -0.5:
                        continue
                    d_ = n_ * (gap - dd)
            if d_.length > need.get(i, Vector()).length:
                need[i] = d_.copy()
    sc.frame_set(1)
    if not need:
        return
    # spread through the hair round each point that needed it (pooled in small cells, as apply_lift), so the hair
    # keeps its thickness and its locks stay together where it is lifted
    reach, SQ, CELL = 1.2, 0.4, 0.3
    sq = lambda c: Vector((c.x, c.y, c.z * SQ))     # noqa: E731
    cells = {}
    for i, dv in need.items():
        c = sq(P[i])
        key = (math.floor(c.x / CELL), math.floor(c.y / CELL), math.floor(c.z / CELL))
        e = cells.get(key)
        if e is None:
            cells[key] = [c.copy(), dv.copy(), 1]
        else:
            e[0] += c
            e[1] += dv
            e[2] += 1
    pool = [(c / n, dv / n, n) for c, dv, n in cells.values()]
    kd = KDTree(len(pool))
    for k, (c, _d, _n) in enumerate(pool):
        kd.insert(c, k)
    kd.balance()
    D = [Vector() for _ in P]
    for i, p in enumerate(P):
        near = kd.find_range(sq(p), reach * 2.0)
        if not near:
            continue
        acc, ws, wmax = Vector(), 0.0, 0.0
        for _co, k, dist in near:
            w = math.exp(-(dist / reach) ** 2)
            acc += pool[k][1] * (w * pool[k][2])
            ws += w * pool[k][2]
            wmax = max(wmax, w)
        D[i] = acc / ws * min(1.0, wmax * 1.6)
        # (never less than the point's own need)
        # (the spread stays by the clothes: a lift at the collar carried 3 units up the locks beside the face, which
        # stood them off the cheeks; hair more than a little way clear of every cloth keeps its place)
        D[i] *= 1.0 - smoothstep(gap + 0.25, gap + 0.9, clear_[i])
        if i in need and need[i].length > D[i].length:
            D[i] = need[i].copy()
    # eased along each strand, so a strand bends over the cloth instead of stepping
    for b, n in HAIR_RUNS:
        for _ in range(3):
            run = [D[b + k] for k in range(n)]
            for k in range(n):
                m = (run[max(0, k - 1)] + run[k] * 2.0 + run[min(n - 1, k + 1)]) / 4.0
                D[b + k] = m if m.length > need.get(b + k, Vector()).length else D[b + k]
    for o in [cap] + locks:
        lay = o.data.attributes['pt'].data
        for v in o.data.vertices:
            v.co += D[lay[v.index].value]
        o.data.update()
    for i in range(len(P)):
        P[i] += D[i]


def creator_hair_trim(cols=range(10), cut=None, top=57.5, rounds=10):
    """What the clearance by points leaves. Below the collar the cap only hangs behind the locks, a curtain nobody
    sees, but its faces are coarser than the collar's rim and the cape's edge, so one whose corners all lie clear can
    still cut through a thin edge between them, and its underside faces open air in the gap at the neck. So the cap is
    cut away below `cut` (the locks cover it there), and then, pose by pose, any face of it still crossing the
    clothes where the sheet's hair check looks (its outer faces: open air along the normal) is taken out, up to
    `top`, until none is left."""
    import bmesh
    from mathutils.bvhtree import BVHTree
    sc = bpy.context.scene
    cap = bpy.data.objects['Hair']
    M = cap.matrix_world

    def drop(test):
        bm = bmesh.new()
        bm.from_mesh(cap.data)
        kill = [f for f in bm.faces if test(f)]
        if kill:
            bmesh.ops.delete(bm, geom=kill, context='FACES')
            bm.to_mesh(cap.data)
            cap.data.update()
        bm.free()
        return len(kill)

    # (cut=None, the default: the cap runs its full length behind the locks, as the creator's does; cut away below
    # the collar it ended in a line of tips across the shoulders. Only faces that cross the clothes come out.)
    if cut is not None:
        drop(lambda f: max((M @ v.co).z for v in f.verts) < cut)
    # (and where it is cut it ends as the creator's hair ends, in strands: its cut-out along the map runs down to
    # nothing over the last stretch above the cut, so the tip map thins it into points instead of a hem)
    uv = cap.data.uv_layers.get('tips')
    if uv is not None and cut is not None:
        for lp in cap.data.loops:
            z = (M @ cap.data.vertices[lp.vertex_index].co).z
            d = uv.data[lp.index]
            d.uv = (d.uv[0], min(d.uv[1], smoothstep(cut + 0.1, cut + 2.2, z)))
    # (the clothes and the rest of the figure, pose by pose, built once: only the cap changes between rounds, and the
    # cap rides the head, so each pose needs only where the head has carried it)
    poses = []
    for col in cols:
        sc.frame_set(col + 1)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        cloth, rest_vs, rest_ps = [], [], []
        for o in bpy.data.objects:
            if o.type != 'MESH' or o.hide_render or not o.visible_get() or o == cap:
                continue
            me = o.evaluated_get(dg).to_mesh()
            vs = [o.matrix_world @ v.co for v in me.vertices]
            ps = [list(q.vertices) for q in me.polygons]
            o.evaluated_get(dg).to_mesh_clear()
            b = len(rest_vs)
            rest_vs += vs
            rest_ps += [[b + i for i in q] for q in ps]
            if o.name.startswith(CLOTHES) and ps:
                cloth.append(BVHTree.FromPolygons(vs, ps))
        poses.append((cap.matrix_world.copy(), cloth, BVHTree.FromPolygons(rest_vs, rest_ps)))
    sc.frame_set(1)
    bpy.context.view_layer.update()
    for _round in range(rounds):
        nv = len(cap.data.vertices)
        bad = set()
        dg = bpy.context.evaluated_depsgraph_get()
        me = cap.evaluated_get(dg).to_mesh()
        lv = [v.co.copy() for v in me.vertices]
        # (only faces low enough to be taken out are tested: the crown is far above every cloth)
        polys = [list(q.vertices) for q in me.polygons if min((M @ lv[i]).z for i in q.vertices) < top]
        cap.evaluated_get(dg).to_mesh_clear()
        for Mw, cloth, figure in poses:
            hv = [Mw @ v for v in lv]
            outside = []
            for vi in polys:
                c = sum((hv[i] for i in vi), Vector()) / len(vi)
                n = (hv[vi[1]] - hv[vi[0]]).cross(hv[vi[2]] - hv[vi[0]])
                if n.length < 1e-12:
                    continue
                n.normalize()
                if figure.ray_cast(c + n * 0.02, n, 1.5)[0] is None:
                    outside.append(vi)
            if not outside:
                continue
            ht = BVHTree.FromPolygons(hv, outside)
            for t in cloth:
                for fi, _j in ht.overlap(t):
                    # (the solidified copy's vertices follow the cap's own, in order)
                    bad.update(i % nv for i in outside[fi])
        sc.frame_set(1)
        n = drop(lambda f: any(v.index in bad for v in f.verts) and max((M @ v.co).z for v in f.verts) < top)
        print('hair trim round', _round, 'faces out', n, flush=True)
        if not n:
            break


def velvet_closeup(name):
    """The robe up close as velvet rather than felted wool: a dense, short pile that is dark where it faces you and
    glows along every turn away (Principled sheen at full weight and a tight sheen roughness, the base deepened under
    it), its nap smooth instead of fuzzy, with faint crushed patches where the pile lies a different way."""
    m = bpy.data.materials.get(name)
    if m is None or 'close bsdf' not in m.node_tree.nodes:
        return
    nt = m.node_tree
    b = nt.nodes['close bsdf']
    b.inputs['Sheen Weight'].default_value = 1.0
    b.inputs['Sheen Roughness'].default_value = 0.3
    b.inputs['Sheen Tint'].default_value = (0.92, 0.8, 0.94, 1.0)   # (a soft lilac sheen on the plum velvet)
    b.inputs['Roughness'].default_value = 0.85
    b.inputs['Specular IOR Level'].default_value = 0.05
    # the base deepened (the sheen supplies the light) and patchily crushed
    src = b.inputs['Base Color'].links[0].from_socket if b.inputs['Base Color'].is_linked else None
    rest = nt.nodes.new('ShaderNodeAttribute')
    rest.attribute_name = 'rest'
    crush = nt.nodes.new('ShaderNodeTexNoise')
    crush.inputs['Scale'].default_value = 0.9
    crush.inputs['Detail'].default_value = 2.0
    nt.links.new(rest.outputs['Vector'], crush.inputs['Vector'])
    k = nt.nodes.new('ShaderNodeMapRange')
    k.inputs['To Min'].default_value = 0.6
    k.inputs['To Max'].default_value = 0.85
    nt.links.new(crush.outputs['Fac'], k.inputs['Value'])
    if src is not None:
        sc_ = nt.nodes.new('ShaderNodeVectorMath')
        sc_.operation = 'SCALE'
        nt.links.new(src, sc_.inputs['Vector'])
        nt.links.new(k.outputs['Result'], sc_.inputs['Scale'])
        nt.links.new(sc_.outputs['Vector'], b.inputs['Base Color'])
    if b.inputs['Normal'].is_linked:
        bn = b.inputs['Normal'].links[0].from_node
        if bn.type == 'BUMP':
            bn.inputs['Strength'].default_value = 0.12


def patina_closeup(name):
    """The vambraces' hardened leather up close as the reference's: an antique bronze-brown, mottled with dark
    patches of age and handling (broad, soft blotches over a finer speckle), darker still in the tooled hollows."""
    m = bpy.data.materials.get(name)
    if m is None or 'close bsdf' not in m.node_tree.nodes:
        return
    nt = m.node_tree
    b = nt.nodes['close bsdf']
    if not b.inputs['Base Color'].is_linked:
        return
    src = b.inputs['Base Color'].links[0].from_socket
    rest = nt.nodes.new('ShaderNodeAttribute')
    rest.attribute_name = 'rest'
    blot = nt.nodes.new('ShaderNodeTexNoise')
    blot.inputs['Scale'].default_value = 1.4
    blot.inputs['Detail'].default_value = 4.0
    blot.inputs['Roughness'].default_value = 0.6
    nt.links.new(rest.outputs['Vector'], blot.inputs['Vector'])
    k = nt.nodes.new('ShaderNodeMapRange')
    k.inputs['From Min'].default_value = 0.38
    k.inputs['From Max'].default_value = 0.62
    k.inputs['To Min'].default_value = 0.42
    k.inputs['To Max'].default_value = 1.08
    nt.links.new(blot.outputs['Fac'], k.inputs['Value'])
    sc_ = nt.nodes.new('ShaderNodeVectorMath')
    sc_.operation = 'SCALE'
    nt.links.new(src, sc_.inputs['Vector'])
    nt.links.new(k.outputs['Result'], sc_.inputs['Scale'])
    nt.links.new(sc_.outputs['Vector'], b.inputs['Base Color'])


def steel_closeup(name, dark=(0.13, 0.13, 0.14), light=(0.62, 0.63, 0.66), rgh=(0.55, 0.22), use_ao=True):
    """The greaves and the breastplate up close (the sprites keep the plain diffuse surface: see sprite_mode):
    hammered steel, a little worn. Shallow overlapping dents from the hammer break up the light; the tone drifts
    in broad patches; it is darker and duller down in the hollows, where grime collects, and polished brighter on
    the high places that rub."""
    m = bpy.data.materials.get(name)
    if m is None:
        return
    nt = m.node_tree
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
    bsdf = next(n for n in nt.nodes if n.type == 'BSDF_DIFFUSE')
    bsdf.name = 'sprite bsdf'
    rest = nt.nodes.new('ShaderNodeAttribute')
    rest.attribute_name = 'rest'
    vor = nt.nodes.new('ShaderNodeTexVoronoi')
    vor.feature = 'SMOOTH_F1'
    vor.inputs['Scale'].default_value = 2.6
    nt.links.new(rest.outputs['Vector'], vor.inputs['Vector'])
    bump = nt.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.35
    bump.inputs['Distance'].default_value = 0.05
    nt.links.new(vor.outputs['Distance'], bump.inputs['Height'])
    patch = nt.nodes.new('ShaderNodeTexNoise')
    patch.inputs['Scale'].default_value = 0.7
    patch.inputs['Detail'].default_value = 3.0
    nt.links.new(rest.outputs['Vector'], patch.inputs['Vector'])
    ao = nt.nodes.new('ShaderNodeAmbientOcclusion')
    ao.inputs['Distance'].default_value = 0.8
    ao.samples = 8
    tone = nt.nodes.new('ShaderNodeMath')
    tone.operation = 'MULTIPLY'
    if use_ao:
        nt.links.new(ao.outputs['AO'], tone.inputs[0])
    else:
        # (an even tone, not lighter where it stands open: so a flat plate reads as the same steel as the band)
        tone.inputs[0].default_value = 0.7
    nt.links.new(patch.outputs['Fac'], tone.inputs[1])
    col = nt.nodes.new('ShaderNodeMix')
    col.data_type = 'RGBA'
    col.inputs['A'].default_value = (*dark, 1)
    col.inputs['B'].default_value = (*light, 1)
    nt.links.new(tone.outputs['Value'], col.inputs['Factor'])
    rough = nt.nodes.new('ShaderNodeMapRange')
    rough.inputs['To Min'].default_value = rgh[0]
    rough.inputs['To Max'].default_value = rgh[1]
    nt.links.new(tone.outputs['Value'], rough.inputs['Value'])
    pb = nt.nodes.new('ShaderNodeBsdfPrincipled')
    pb.name = 'close bsdf'
    pb.inputs['Metallic'].default_value = 1.0
    nt.links.new(col.outputs['Result'], pb.inputs['Base Color'])
    nt.links.new(rough.outputs['Result'], pb.inputs['Roughness'])
    nt.links.new(bump.outputs['Normal'], pb.inputs['Normal'])
    nt.links.new(pb.outputs['BSDF'], out.inputs['Surface'])


# The bronze of the knees, greaves and vambraces up close, as the reference's browned armour: not a polished
# mirror but a dull satin metal, dark and warm, mottled in broad blotches where it has tarnished unevenly, darker
# still down in the hollows where grime sits, and rubbed back toward the bright alloy on the edges and high places
# that wear. (Metal reflectance stays metal, 0.2 to 0.45 linear in red; the tarnish is a dark warm brown, about
# sRGB 60, 40, 25; satin roughness 0.4 to 0.75, rougher where it is tarnished.) BRONZE_LOOKS are the study's
# candidates (bronze_study.py renders them side by side on stand-in shapes, without building the figure); BRONZE_LOOK is the one the figure wears.
BRONZE_LOOKS = {
    'browned': dict(metal=(0.30, 0.15, 0.06), tarnish=(0.045, 0.022, 0.011), mottle=0.65, rub=0.55, rgh=(0.45, 0.72)),
    'russet': dict(metal=(0.22, 0.10, 0.04), tarnish=(0.032, 0.016, 0.008), mottle=0.75, rub=0.45, rgh=(0.5, 0.78)),
    'aged': dict(metal=(0.45, 0.24, 0.10), tarnish=(0.07, 0.035, 0.015), mottle=0.55, rub=0.6, rgh=(0.4, 0.68)),
    'polished': dict(metal=(0.82, 0.5, 0.29), tarnish=(0.17, 0.075, 0.035), mottle=0.3, rub=0.2, rgh=(0.3, 0.6)),
}
BRONZE_LOOK = 'aged'


def bronze_closeup(name, look=None):
    m = bpy.data.materials.get(name)
    if m is None:
        return
    look = BRONZE_LOOK if look is None else look
    # (by name, or by its place in the list: views.py --set BRONZE_LOOK=2, which needs no quoting)
    L = BRONZE_LOOKS[look] if isinstance(look, str) else list(BRONZE_LOOKS.values())[int(look)]
    nt = m.node_tree
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_DIFFUSE'), None)
    if bsdf is not None:
        bsdf.name = 'sprite bsdf'
    N = nt.nodes.new
    rest = N('ShaderNodeAttribute')
    rest.attribute_name = 'rest'

    def noise(scale, detail, rough=0.55):
        t = N('ShaderNodeTexNoise')
        t.inputs['Scale'].default_value = scale
        t.inputs['Detail'].default_value = detail
        t.inputs['Roughness'].default_value = rough
        nt.links.new(rest.outputs['Vector'], t.inputs['Vector'])
        return t.outputs['Fac']

    def rng(sock, a, b, lo=0.0, hi=1.0):
        r = N('ShaderNodeMapRange')
        r.inputs['From Min'].default_value, r.inputs['From Max'].default_value = a, b
        r.inputs['To Min'].default_value, r.inputs['To Max'].default_value = lo, hi
        nt.links.new(sock, r.inputs['Value'])
        return r.outputs['Result']

    def math_(op, a, b, clamp=True):
        k = N('ShaderNodeMath')
        k.operation = op
        k.use_clamp = clamp
        for i, v in enumerate((a, b)):
            if isinstance(v, float):
                k.inputs[i].default_value = v
            else:
                nt.links.new(v, k.inputs[i])
        return k.outputs[0]
    # the tarnish: broad blotches, broken by a finer speckle
    blot = rng(noise(1.1, 4.0), 0.42, 0.62)     # (blotches a third of a greave across, not wider than the plate)
    speck = rng(noise(7.0, 6.0, 0.6), 0.35, 0.7)
    mott = math_('MULTIPLY', math_('ADD', math_('MULTIPLY', blot, 0.75), math_('MULTIPLY', speck, 0.25)), L['mottle'])
    # grime down in the hollows
    ao = N('ShaderNodeAmbientOcclusion')
    ao.inputs['Distance'].default_value = 0.6
    ao.samples = 8
    dirt = rng(ao.outputs['AO'], 0.95, 0.45)
    # rubbed bright on the edges and high places
    # (where the surface turns sharply: a rounded normal against the true one. Pointiness is per vertex of the
    # base mesh and sits at 0.5 all over a smooth lofted plate, so it found no edges)
    geo = N('ShaderNodeNewGeometry')
    bev = N('ShaderNodeBevel')
    bev.samples = 4
    bev.inputs['Radius'].default_value = 0.09
    dot = N('ShaderNodeVectorMath')
    dot.operation = 'DOT_PRODUCT'
    nt.links.new(bev.outputs['Normal'], dot.inputs[0])
    nt.links.new(geo.outputs['Normal'], dot.inputs[1])
    edge = rng(dot.outputs['Value'], 0.995, 0.9)
    rub = math_('MULTIPLY', math_('MULTIPLY', edge, rng(noise(3.0, 3.0), 0.3, 0.7, 0.4, 1.0)), L['rub'])
    tf = math_('SUBTRACT', math_('ADD', mott, math_('MULTIPLY', dirt, 0.7)), rub)
    col = N('ShaderNodeMix')
    col.data_type = 'RGBA'
    col.inputs['A'].default_value = (*L['metal'], 1)
    col.inputs['B'].default_value = (*L['tarnish'], 1)
    nt.links.new(tf, col.inputs['Factor'])
    # (and the tone drifts a little, so no two plates match exactly)
    var = N('ShaderNodeMix')
    var.data_type, var.blend_type = 'RGBA', 'MULTIPLY'
    var.inputs['Factor'].default_value = 1.0
    nt.links.new(col.outputs['Result'], var.inputs['A'])
    g = rng(noise(1.6, 2.0), 0.3, 0.7, 0.85, 1.12)
    cmb = N('ShaderNodeCombineColor')
    for c in ('Red', 'Green', 'Blue'):
        nt.links.new(g, cmb.inputs[c])
    nt.links.new(cmb.outputs['Color'], var.inputs['B'])
    rough = N('ShaderNodeMapRange')
    rough.inputs['To Min'].default_value, rough.inputs['To Max'].default_value = L['rgh']
    nt.links.new(tf, rough.inputs['Value'])
    # the surface: shallow hammer dents and a fine pitting of the tarnish
    vor = N('ShaderNodeTexVoronoi')
    vor.feature = 'SMOOTH_F1'
    vor.inputs['Scale'].default_value = 2.6
    nt.links.new(rest.outputs['Vector'], vor.inputs['Vector'])
    b1 = N('ShaderNodeBump')
    b1.inputs['Strength'].default_value = 0.25
    b1.inputs['Distance'].default_value = 0.05
    nt.links.new(vor.outputs['Distance'], b1.inputs['Height'])
    b2 = N('ShaderNodeBump')
    b2.inputs['Strength'].default_value = 0.12
    b2.inputs['Distance'].default_value = 0.01
    nt.links.new(noise(22.0, 4.0, 0.7), b2.inputs['Height'])
    nt.links.new(b1.outputs['Normal'], b2.inputs['Normal'])
    pb = N('ShaderNodeBsdfPrincipled')
    pb.name = 'close bsdf'
    pb.inputs['Metallic'].default_value = 1.0
    nt.links.new(var.outputs['Result'], pb.inputs['Base Color'])
    nt.links.new(rough.outputs['Result'], pb.inputs['Roughness'])
    nt.links.new(b2.outputs['Normal'], pb.inputs['Normal'])
    nt.links.new(pb.outputs['BSDF'], out.inputs['Surface'])


def fuse_hair(head, names, voxel=0.065, close=0.6):
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
    if os.environ.get('WARDEN_HAIR_PIECES'):   # (for working on the fusion alone)
        import pickle
        with open(os.environ['WARDEN_HAIR_PIECES'], 'wb') as f_:
            pickle.dump(([tuple(v.co) for v in me.vertices], [tuple(p_.vertices) for p_ in me.polygons]), f_)
    # one signed-distance volume of all the pieces, closed (grown by CLOSE, then shrunk back by it), which fills
    # every groove and corner narrower than that where two pieces meet with a round fillet, then filleted again
    # (the concave curvature flowed out) and turned back into a surface
    ng = bpy.data.node_groups.new('hair fuse', 'GeometryNodeTree')
    ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    N_, L_ = ng.nodes, ng.links
    gi, go = N_.new('NodeGroupInput'), N_.new('NodeGroupOutput')
    sdf = N_.new('GeometryNodeMeshToSDFGrid')
    sdf.inputs['Voxel Size'].default_value = voxel
    sdf.inputs['Band Width'].default_value = int(close / voxel) + 4
    L_.new(gi.outputs[0], sdf.inputs['Mesh'])
    cur = sdf.outputs[0]
    for dist in (close, -close):
        o_ = N_.new('GeometryNodeSDFGridOffset')
        o_.inputs['Distance'].default_value = dist
        L_.new(cur, o_.inputs['Grid'])
        cur = o_.outputs[0]
    fl = N_.new('GeometryNodeSDFGridFillet')
    fl.inputs['Iterations'].default_value = 3
    L_.new(cur, fl.inputs['Grid'])
    gm = N_.new('GeometryNodeGridToMesh')
    gm.inputs['Threshold'].default_value = 0.0
    L_.new(fl.outputs[0], gm.inputs['Grid'])
    L_.new(gm.outputs[0], go.inputs[0])
    ob.modifiers.new('fuse', 'NODES').node_group = ng
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    fused = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)
    bpy.data.node_groups.remove(ng)
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
    tips = [v for v in bm.verts if v.co.z < -13.0]
    for _ in range(10):
        bmesh.ops.smooth_laplacian_vert(bm, verts=tips, lambda_factor=0.6, lambda_border=0.0,
                                        use_x=True, use_y=True, use_z=True, preserve_volume=True)
    sides = [v for v in bm.verts if -3.5 < v.co.z < 8.2 and abs(v.co.x) > 1.6 and v.co.y < 5.0]
    for k_ in range(30):
        bmesh.ops.smooth_laplacian_vert(bm, verts=sides, lambda_factor=0.7, lambda_border=0.0,
                                        use_x=True, use_y=True, use_z=k_ % 3 == 2, preserve_volume=True)
    band = [v for v in bm.verts if -4.0 < v.co.z < 9.0 and v.co.y > 1.0]
    for _ in range(8):
        bmesh.ops.smooth_laplacian_vert(bm, verts=band, lambda_factor=0.5, lambda_border=0.0,
                                        use_x=True, use_y=True, use_z=True, preserve_volume=True)
    round_back(bm)
    # soften the edges left where the side hair's back meets the fall: plain averaging (which wears convex edges down
    # fastest) behind the ears, strongest in the middle of that region and fading out at its borders
    for lo_, n_ in ((0.0, 6), (0.35, 10), (0.7, 14)):
        reg = [v for v in bm.verts if hair_join_weight(v.co) > lo_]
        for _ in range(n_):
            bmesh.ops.smooth_vert(bm, verts=reg, factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    for _ in range(4):
        bmesh.ops.smooth_laplacian_vert(bm, verts=band, lambda_factor=0.5, lambda_border=0.0,
                                        use_x=True, use_y=True, use_z=True, preserve_volume=True)
    keep_off_clothes(bm, head)
    # the waves, pressed in last so no smoothing irons them out: below the ears the hair rises and falls across the
    # locks in broad S-curves, each lock a little out of step with the next (a lit crest and a shaded trough every
    # hand's breadth or so, which is what reads as wavy hair at this size), and a finer ripple over them
    bm.normal_update()
    for v in bm.verts:
        fade = smoothstep(1.5, -2.5, v.co.z)   # (below the jaw: higher, over the side hair, they read as knots)
        if fade <= 0.0:
            continue
        arc = math.atan2(v.co.x, v.co.y - 0.5) * max(2.5, math.hypot(v.co.x, v.co.y - 0.5))
        wave = 0.17 * math.sin(0.95 * v.co.z + 0.5 * arc + 0.7 * math.sin(0.33 * arc)) \
            + 0.04 * math.sin(2.3 * v.co.z - 0.8 * arc + 1.1)
        n_ = Vector((v.normal.x, v.normal.y, 0.0))
        # (on the outside of the hair only: pressed into its underside too, the troughs sank it into her back)
        rad = Vector((v.co.x, v.co.y - 0.5, 0.0))
        outside = smoothstep(0.0, 0.5, n_.dot(rad.normalized()) if rad.length > 1e-6 else 0.0)
        v.co += n_ * wave * fade * outside
    for v in bm.verts:
        g_ = cap_grooves(v.co)
        if g_:
            v.co += v.normal * g_
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
            (-UPPER - 1.0, 1.5), (-UPPER - 2.2, 1.4), (-UPPER - 5.0, 1.2), (-UPPER - FORE + 0.4, 1.06),
            (-UPPER - FORE - 0.4, 1.04)]   # (slim under the vambrace, so it stays inside it as the elbow bends)
BR_TOP, BR_BOT = -UPPER - 1.7, -UPPER - FORE - 0.55
BRACER_R = [(BR_TOP, 2.0), (BR_TOP - 0.5, 1.92), (-UPPER - 5.0, 1.72), (-UPPER - FORE + 0.8, 1.54), (BR_BOT, 1.62)]
BR_STRAPS = (BR_TOP - 1.4, -UPPER - 5.6, -UPPER - FORE + 1.3)
# the steel vambrace over the sleeve (height from the shoulder joint, radius): flared at the elbow, full over the
# forearm's muscle, drawn in to the wrist, then flared a little over it
VAMB_R = [(BR_TOP, 1.84), (BR_TOP - 0.4, 1.72), (BR_TOP - 1.0, 1.62), (-UPPER - 3.0, 1.55), (-UPPER - 4.5, 1.46), (-UPPER - 6.5, 1.3),
          (-UPPER - FORE + 1.0, 1.2), (-UPPER - FORE + 0.2, 1.19), (BR_BOT + 0.2, 1.21), (BR_BOT, 1.27)]
# (fitted: the bare arm's own shape (BARE_ARM_R) and a leather's thickness off it, flaring only at the wrist)

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
BOOT_TOP = -1.0       # the leather boot's top, under the greave (height from the knee)


def leg_form(z, phi, s):
    off, cr = creases(LEG_CREASES, z, s * phi, 0.1)
    return off, min(1.0, cr)


def boot_top(p):
    """The boot's top (under the greave: level)."""
    return 0.0


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
            (-1.4, 2.16, 2.3), (-3.6, 2.22, 2.34), (-6.5, 2.08, 2.18),
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

    # ---- torso: the gambeson (the body's own surface) and over it the breastplate
    torso_mesh('Gambeson', M_SLEEVE, body)
    pz = [52.4 - 0.3 * i for i in range(46)]       # 52.4 down to 38.9, under the belt
    mesh_obj('Breastplate', loft_fn(pz, plate_pt, n=128, cap_top=False, cap_bot=False), M_PLATE, body, solid=0.18)
    # its edges rolled: round the neck and at the waist
    for nm, z_ in (('neck', 52.25), ('waist', 39.05)):
        mesh_obj(f'Breastplate {nm} roll', loft_fn([z_ + 0.2, z_, z_ - 0.2], lambda z, p: outward(plate_pt(z, p), 0.18),
                                                  n=128, cap_top=False, cap_bot=False), M_PLATE, body, solid=0.12)

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
    TL = 0.12      # (the embroidery stands this far off the cloth's own surface: clear of the tabard's thickness,
    # or its straight runs sank into the cloth across the round of the chest and showed in patches)
    for s in (-1, 1):
        mesh_obj(f'Tabard edge {s}', strip(polyline(lambda z: (s * (robe_edge(z) - 0.42), z), 51.8, 38.9, 40), 0.24,
                                           tab_y, TL), M_TRIM, body, subdiv=0, solid=0.05)
        mesh_obj(f'Tabard yoke {s}', strip(polyline(lambda t: (s * lerp(1.45, 0.0, t), lerp(52.3, 47.2, t)), 0.0, 1.0, 40),
                                           0.3, tab_y, TL), M_TRIM, body, subdiv=0, solid=0.05)
    mesh_obj('Tabard collar', strip(polyline(lambda u: (u * (tab_hw(52.3) - 0.1), 52.3), -1.0, 1.0, 14), 0.24, tab_y, TL),
             M_TRIM, body, subdiv=0, solid=0.05)
    mesh_obj('Tabard band', strip([(0.0, 47.4), (0.0, 43.0), (0.0, 38.9)], 0.32, tab_y, TL), M_TRIM, body, subdiv=0,
             solid=0.05)

    # ---- the robe over it all, open down the front: a standing collar, over the shoulders and down to the belt
    # (its skirt, below the belt, hangs from the hips: see below); its fronts and hem trimmed in an embroidered
    # ochre band
    ro = mesh_obj('Robe', robe_sheet(53.6, 38.9, 62), M_ROBE, body, solid=0.15)
    crease_attr(ro, lambda co: robe_form(co.z, math.atan2(-co.x, -co.y))[1])
    RT = 0.07
        # (the trim starts under the choker: higher up it stood out through the band beside the collar)
    for s in (-1, 1):
        mesh_obj(f'Robe front {s}', strip(polyline(lambda z: (s * (robe_edge(z) + 0.36), z), 52.8, 38.9, 44), 0.3,
                                          robe_y, 0.15 + RT), M_ROBE_TRIM, body, subdiv=0, solid=0.05)

    # ---- the belt at the waist, over the robe and the tabard, with a gold buckle
    def BO(p):
        return 1.02 + 0.42 * max(0.0, math.cos(p)) ** 2
    mesh_obj('Belt', loft_fn([40.1, 39.4, 38.7], lambda z, p: torso_point(z, p, BO(p), bust=0.0), n=96, cap_top=False,
                             cap_bot=False), M_BELT, body, solid=0.28, subdiv=0)
    bst = []
    for k in range(80):
        p = 2 * math.pi * k / 80
        if abs(wrap(p)) < 0.22:
            continue
        for dz in (0.5, -0.5):
            x, y = torso_point(39.4 + dz, p, BO(p) + 0.29, bust=0.0)
            bst.append((x, y, 39.4 + dz))
    mesh_obj('Belt stitching', dots(bst, 0.15), M_STITCH, body, subdiv=0)
    by_ = torso_point(39.4, 0.0, BO(0.0), bust=0.0)[1]
    mesh_obj('Buckle', frame_bm(1.6, 1.7, 0.3, 0.3), M_GOLD, body, loc=(0, by_ - 0.42, 39.4), subdiv=0)
    mesh_obj('Buckle prong', box(0.9, 0.16, 0.16), M_GOLD, body, loc=(0.22, by_ - 0.56, 39.4), subdiv=0)

    # ---- neck and head (the base figure's)
    head, head_tree = build_head(body)

    # ---- the hair: the face creator's long waves with a centre parting (lifted clear of the clothes after the rig)
    global CAPE_UNDER
    CAPE_UNDER = under_table()     # (what is built so far, for the cape to lie over)
    if not female.MANNEQUIN:
        creator_hair(head)

    # ---- the rig (the base figure's)
    build_rig(body)

    # ---- what hangs from the hips, carried by the thighs: the tabard below the belt, and the robe's skirt over it
    lo = mesh_obj('Tabard skirt', tab_grid(39.9, None, 58, hem=tab_hem), M_TABARD, skin=skirt_weights, solid=0.12)
    crease_attr(lo, lambda co: tab_folds(co.x, co.z))
    # its trim: down both edges and round the stepped hem, and the band down the middle ending in a stepped cross
    hem_path = [(u * tab_hw(tab_hem(u)), tab_hem(u) + 0.3) for u in [-1 + 2 * i / 60 for i in range(61)]]
    hem_path = [(x * (1 - 0.09 / max(abs(x), 0.09)) if abs(x) > 0.1 else x, z) for x, z in hem_path]
    trims = [('Tabard skirt hem', hem_path, 0.26)]
    for s in (-1, 1):
        trims.append((f'Tabard skirt edge {s}', polyline(lambda z: (s * (robe_edge(z) - 0.42), z), 39.9, tab_hem(s) + 0.5, 40), 0.24))
    trims.append(('Tabard skirt band', [(0.0, 39.9), (0.0, 31.0), (0.0, 24.6)], 0.32))
    for i, (zc, hw_) in enumerate(((27.6, 1.2), (25.9, 0.7))):
        trims.append((f'Tabard skirt bar {i}', [(-hw_, zc), (hw_, zc)], 0.26))
    for nm, path, half in trims:
        mesh_obj(nm, strip(path, half, tab_y, TL), M_TRIM, skin=skirt_weights, subdiv=0, solid=0.05)

    rs = mesh_obj('Robe skirt', robe_sheet(39.9, None, 70, hem=robe_hem), M_ROBE, skin=skirt_weights, solid=0.15)
    crease_attr(rs, lambda co: robe_form(co.z, math.atan2(-co.x, -co.y))[1])
    for s in (-1, 1):
        mesh_obj(f'Robe skirt front {s}', strip(polyline(lambda z: (s * (robe_edge(z) + 0.36), z), 39.9, robe_hem(0.0) + 0.1, 44),
                                                0.3, robe_y, 0.15 + RT), M_ROBE_TRIM, skin=skirt_weights, subdiv=0, solid=0.05)

    # the hem's band, all the way round the skirt to the fronts
    def hem_band(i, j, n=112):
        a = robe_open(21.0)
        phi = a + (2 * math.pi - 2 * a) * j / (n - 1)
        z = robe_hem(phi) + 0.62 * (1 - i)
        x, y = robe_pt(z, phi, 0.15 + RT)
        return (x, y, z)
    mesh_obj('Robe hem', face_out(grid(2, 112, hem_band), axis_y=0.0), M_ROBE_TRIM, skin=skirt_weights, subdiv=0, solid=0.05)

    # ---- arms: the robe's sleeve under a buckled leather bracer, then the hand
    def sleeve(side, s, x, w):
        def fore(co, w=w):
            d = w(co)
            d[f'Forearm.{side}'] = d.get(f'Forearm.{side}', 0.0) + d.pop(f'Hand.{side}', 0.0)
            return d
        zs = [1.55 - 0.22 * i for i in range(int((1.55 + UPPER + FORE + 0.4) / 0.22) + 1)]

        def arm_pt(z, phi):
            r = _profile(SLEEVE_R, z) + deltoid(z, phi, s) + sleeve_form(z, phi, s)[0]
            return -r * math.sin(phi), -r * 1.03 * math.cos(phi)
        # (bare arms under the robe's short sleeves: the base figure's)
        bare_arm(side, s, x, w)
        # the robe's own sleeve over it, short: from the shoulder (its top rounded over the arm, full width there) to
        # a little above the elbow, widening a touch to a soft bell, its end trimmed in the ochre band
        SE = -UPPER + 3.6

        # knife pleats, as a tailor lays them: each one a broad face that turns under in a narrow return at a crisp
        # fold, all lying the same way round the arm; stitched flat into the shoulder seam and opening toward the
        # hem, so the sleeve flares a little as they open (a sawtooth round the sleeve, deepening as it falls)
        NP = 14

        def pleat(z, phi):
            t = (NP * (phi + 0.13) / (2 * math.pi)) % 1.0
            f = t / 0.8 if t < 0.8 else (1.0 - t) / 0.2      # the face rises to the fold, then the return drops away
            return f, smoothstep(0.9, SE, z), t >= 0.8

        def robe_sleeve_pt(z, phi, out=0.0):
            f, k, _ = pleat(z, phi)
            r = _profile(SLEEVE_R, z) + deltoid(z, phi, s) + 0.24 + 0.38 * smoothstep(SE + 3.0, SE, z) + out
            r += 0.26 * k * (f - 0.5)
            return -r * math.sin(phi), -r * 1.03 * math.cos(phi)
        rzs = [1.6 - 0.2 * i for i in range(int((1.6 - SE) / 0.2) + 1)] + [SE]
        rs_ = mesh_obj(f'Robe sleeve.{side}', loft_fn(rzs, robe_sleeve_pt, n=NP * 10, cap_bot=False), M_ROBE,
                       loc=(x, 0.2, SH_Z), skin=w, solid=0.12)
        # (each pleat's return lies in the shadow of the fold over it)
        crease_attr(rs_, lambda co: (lambda f, k, ret: k * (0.9 if ret else 0.25 * max(0.0, 0.3 - f) / 0.3))(
            *pleat(co.z, math.atan2(-co.x, -co.y / 1.03))))
        mesh_obj(f'Robe sleeve hem.{side}', loft_fn([SE + 0.55, SE + 0.25, SE - 0.02], lambda z, p: robe_sleeve_pt(z, p, 0.07),
                                                   n=NP * 10, cap_top=False, cap_bot=False),
                 M_ROBE_TRIM, loc=(x, 0.2, SH_Z), skin=w, solid=0.06, subdiv=0)
        # the vambrace: bronze, as the reference: a tapered cone flared at the elbow and drawn in to the wrist, open
        # up the inside of the forearm and laced shut there through brass grommets, its edge bound in a rolled
        # border. It is built of lames, bands lapping one over the next: a deep cuff at the elbow and four bands
        # below it, each band's upper edge turned out into a rounded rib that rides over the foot of the one above,
        # with rivets under the rib. The steps and their ribs are what read as grooves.
        VAM_EDGES = [BR_TOP, BR_TOP - 2.25] + [lerp(BR_TOP - 2.25, BR_BOT, k / 4) for k in range(1, 5)]
        VLAP, VBEAD = 0.085, 0.045

        def vam_rise(z, z0, z1):
            t = min(1.0, max(0.0, (z0 - z) / (z0 - z1)))
            return VLAP * (1.0 - t) ** 1.3 + VBEAD * math.exp(-((z0 - z - 0.13) / 0.075) ** 2)

        def vam_lame(z):
            for k in range(len(VAM_EDGES) - 1):
                if z >= VAM_EDGES[k + 1] or k == len(VAM_EDGES) - 2:
                    return k
        GAP = 0.3     # (the half-width of the laced opening up the inside, as an angle)
        P0, P1 = -s * math.pi / 2 + GAP, -s * math.pi / 2 + 2 * math.pi - GAP

        def vam_pt(z, phi, out=0.0, k=None):
            """(x, y, z) of the vambrace at its nominal height z and phi: the arm's own surface (arm_xy) a leather's
            thickness off it, so it is full where the forearm's muscle is and an oval at the wrist as the arm is;
            its elbow edge is cut to the arm, rising to a point over the elbow behind and dipping in the crook in
            front so the arm can bend, and its wrist edge runs a little down over the back of the hand."""
            k = vam_lame(z) if k is None else k
            rise = vam_rise(z, VAM_EDGES[k], VAM_EDGES[k + 1])
            t = (z - BR_BOT) / (BR_TOP - BR_BOT)
            c = math.cos(phi)
            zz = z + t * (0.8 * max(0.0, -c) ** 2 - 0.5 * max(0.0, c) ** 2) - (1.0 - t) * 0.12 * max(0.0, s * math.sin(phi)) ** 2
            lip = 0.05 * smoothstep(BR_BOT + 0.5, BR_BOT, z) + 0.03 * smoothstep(BR_TOP - 0.4, BR_TOP, z)
            x_, y_ = arm_xy(zz, phi, s, 0.15 + lip + rise + out)
            return x_, y_, zz

        def vam_n(z, phi):
            a_, b_ = vam_pt(z, phi - 0.02), vam_pt(z, phi + 0.02)
            n_ = Vector((b_[1] - a_[1], a_[0] - b_[0], 0.0)).normalized()
            p_ = vam_pt(z, phi)
            return n_ if n_.x * p_[0] + n_.y * p_[1] > 0 else -n_

        # (each lame has its own rows, closer together toward its ribbed upper edge; the lame above ends on the
        # ground surface right over that edge, so the step is a clean wall)
        VR = 22
        rows = VR * (len(VAM_EDGES) - 1)
        cols = 150

        def vam_grid(i, j):
            k, u = i // VR, (i % VR) / (VR - 1)
            z = lerp(VAM_EDGES[k] - (0.012 if k else 0.0), VAM_EDGES[k + 1], u * u)
            ph = lerp(P0, P1, j / (cols - 1))
            return vam_pt(z, ph, 0.0, k)
        mesh_obj(f'Bracer.{side}', face_out(grid(rows, cols, vam_grid), axis_y=0.0), M_BRONZE, loc=(x, 0.2, SH_Z),
                      skin=fore, solid=0.12)
        # (the rolled border all round its edge: a rounded rectangle in (phi, z), the corners eased)
        cr, cz = 0.14, 0.25
        border = []
        for t_ in [k / 40 for k in range(40)]:
            border.append((lerp(P0 + cr, P1 - cr, t_), BR_TOP))
        for k in range(8):
            u = math.pi / 2 * k / 8
            border.append((P1 - cr + cr * math.sin(u), BR_TOP - cz + cz * math.cos(u)))
        for t_ in [k / 30 for k in range(31)]:
            border.append((P1, lerp(BR_TOP - cz, BR_BOT + cz, t_)))
        for k in range(1, 8):
            u = math.pi / 2 * k / 8
            border.append((P1 - cr + cr * math.cos(u), BR_BOT + cz - cz * math.sin(u)))
        for t_ in [k / 40 for k in range(41)]:
            border.append((lerp(P1 - cr, P0 + cr, t_), BR_BOT))
        for k in range(1, 8):
            u = math.pi / 2 * k / 8
            border.append((P0 + cr - cr * math.sin(u), BR_BOT + cz - cz * math.cos(u)))
        for t_ in [k / 30 for k in range(31)]:
            border.append((P0, lerp(BR_BOT + cz, BR_TOP - cz, t_)))
        for k in range(1, 8):
            u = math.pi / 2 * k / 8
            border.append((P0 + cr - cr * math.cos(u), BR_TOP - cz + cz * math.sin(u)))
        bm = bmesh.new()
        rings_ = []
        m_ = len(border)
        for k, (ph, z) in enumerate(border):
            pa, pb_ = border[(k - 1) % m_], border[(k + 1) % m_]
            c = Vector(vam_pt(z, ph, 0.04))
            tg = Vector(vam_pt(pb_[1], pb_[0], 0.04)) - Vector(vam_pt(pa[1], pa[0], 0.04))
            nn = vam_n(z, ph)
            bb = tg.cross(nn).normalized()
            rings_.append([bm.verts.new(c + nn * (0.11 * math.cos(2 * math.pi * q / 8)) + bb * (0.13 * math.sin(2 * math.pi * q / 8)))
                           for q in range(8)])
        for k in range(m_):
            ra, rb = rings_[k], rings_[(k + 1) % m_]
            for q in range(8):
                bm.faces.new((ra[q], ra[(q + 1) % 8], rb[(q + 1) % 8], rb[q]))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        mesh_obj(f'Bracer border.{side}', bm, M_BIND, loc=(x, 0.2, SH_Z), skin=fore, subdiv=0)
        # brass grommets in pairs either side of the opening, and the thong laced across between them
        NG = 7
        grom, gz = [], [lerp(BR_TOP - 0.75, BR_BOT + 0.75, k / (NG - 1)) for k in range(NG)]
        for z in gz:
            for ph in (P0 + 0.17, P1 - 0.17):
                grom.append(vam_pt(z, ph, 0.05))
        mesh_obj(f'Bracer grommets.{side}', dots(grom, 0.17), M_GOLD, loc=(x, 0.2, SH_Z), skin=fore, subdiv=0)
        hole = [vam_pt(z, ph, 0.09) for z, ph in ((z, ph) for z in gz for ph in (P0 + 0.17, P1 - 0.17))]
        # (the rivets under each rib: on the outside of the forearm and a way round to either side of it)
        vrv = [vam_pt(VAM_EDGES[k] - 0.42, s * math.pi / 2 + d_, 0.13)
               for k in range(len(VAM_EDGES) - 1) for d_ in (-0.95, 0.0, 0.95)]
        mesh_obj(f'Bracer rivets.{side}', dots(vrv, 0.15), M_GOLD, loc=(x, 0.2, SH_Z), skin=fore, subdiv=0)
        mesh_obj(f'Bracer grommet holes.{side}', dots(hole, 0.08), M_BELT, loc=(x, 0.2, SH_Z), skin=fore, subdiv=0)
        # (what shows through the opening: the dark lining of the vambrace, just inside it)
        def lin_pt(i, j):
            z = lerp(BR_TOP - 0.1, BR_BOT + 0.1, i / 19)
            ph = lerp(P1 - 0.06 - 2 * math.pi, P0 + 0.06, j / 5)
            return vam_pt(z, ph, -0.1)
        mesh_obj(f'Bracer gap.{side}', face_out(grid(20, 6, lin_pt), axis_y=0.0), M_BELT, loc=(x, 0.2, SH_Z),
                 skin=fore, subdiv=0)
        lace = bmesh.new()
        for k in range(NG - 1):
            for pa_, pb2 in (((gz[k], P0 + 0.17), (gz[k + 1], P1 - 0.17 - 2 * math.pi)),
                             ((gz[k], P1 - 0.17 - 2 * math.pi), (gz[k + 1], P0 + 0.17))):
                pts = []
                for q in range(9):
                    t_ = q / 8
                    z = lerp(pa_[0], pb2[0], t_)
                    ph = lerp(pa_[1], pb2[1], t_)
                    pts.append(vam_pt(z, ph, 0.08 + 0.04 * math.sin(math.pi * t_)))
                seg = tube(pts, [0.055] * 9, flat=1.0, n=6)
                tmp = bpy.data.meshes.new('lace')
                seg.to_mesh(tmp)
                seg.free()
                lace.from_mesh(tmp)
                bpy.data.meshes.remove(tmp)
        mesh_obj(f'Bracer lacing.{side}', lace, M_BELT, loc=(x, 0.2, SH_Z), skin=fore, subdiv=0)
        # the vambrace's tongue over the back of the hand, from under its cuff toward the knuckles, narrowing
        hz0 = -UPPER - FORE

        def hand_plate(z0, z1, wide0, wide1, lift, rows=10, cols=14):
            def pt(i, j):
                v = i / (rows - 1)
                z = lerp(z0, z1, v)
                hw = lerp(wide0, wide1, v)
                u = lerp(-1.0, 1.0, j / (cols - 1))
                bulge = 0.66 + lift - 0.38 * u * u
                return (s * 0.23 - s * bulge, -0.03 + hw * u, z)
            return face_out(grid(rows, cols, pt), axis_y=0.0)
        mesh_obj(f'Gauntlet plate.{side}', hand_plate(hz0 - 0.1, hz0 - 1.8, 1.15, 0.85, 0.22), M_BRONZE,
                 loc=(x, 0.2, SH_Z), skin=w, solid=0.1, subdiv=1)
    build_arms(chest, sleeve)   # (the hands are the base figure's)

    # ---- the cape over it all: the dark wool, its red lining just inside, an ochre band down each front, and the
    # two gold medallions that pin it at the chest, joined by a chain hanging between them
    global CAPE_UNDER_BACK
    CAPE_UNDER_BACK = CAPE_UNDER
    CAPE_UNDER = under_table()
    cp = mesh_obj('Cape', cape_sheet(), M_CAPE, skin=cape_weights, solid=0.12)
    set_crease(cp, CAPE_CREASE)
    li = mesh_obj('Cape lining', cape_sheet(-0.16), M_LINING, skin=cape_weights)
    set_crease(li, CAPE_CREASE)
    for s_ in (-1, 1):
        def band(i, j, s_=s_, rows=80):
            v = i / (rows - 1)
            z = lerp(53.2, cape_hem(_table(CAPE_EDGE, 20.0)) + 0.1, v ** 1.15)
            e = _table(CAPE_EDGE, z)
            ph = e + 0.07 * j
            x, y = cape_pt(z, ph, 0.08)
            z = cape_z(z)
            return (x, y, z) if s_ > 0 else (-x, y, z)
        mesh_obj(f'Cape front {s_}', face_out(grid(80, 2, band), axis_y=0.0), M_ROBE_TRIM, skin=cape_weights, subdiv=0,
                 solid=0.06)
    clasps = []
    for s_ in (-1, 1):
        ph = _table(CAPE_EDGE, CLASP_Z) + 0.12
        x, y = cape_pt(CLASP_Z, ph, 0.28)
        x = x if s_ > 0 else -x
        a = ph if s_ > 0 else -ph
        cz = cape_z(CLASP_Z)
        mesh_obj(f'Medallion {s_}', turned(medallion(), a), M_GOLD, loc=(x, y, cz), skin=cape_weights, subdiv=1)
        clasps.append(Vector((x, y - 0.1, cz - 0.2)))
    a_, b_ = clasps
    chain = []
    for k in range(17):
        t = k / 16
        p_ = a_.lerp(b_, t)
        p_.z -= 0.9 * math.sin(math.pi * t)
        p_.y = min(p_.y, tab_y(p_.x, p_.z) - 0.25)
        chain.append(p_)
    mesh_obj('Clasp chain', tube(chain, [0.1] * len(chain), n=6), M_GOLD, skin=cape_weights, subdiv=0)
    # ---- the steel stand collar at the neck
    build_collar(body)


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
        # the greave: a short steel shell round the shin over the boot, from under the knee to halfway down the shin,
        # shaped to the calf, with a ridge down the front; its top and bottom edges turned out a little
        GT, GB = -0.75, -6.6
        # Built as the reference's: not one shell with lines cut in it but lames, plates lapping one over the next
        # down the shin like shingles. Each lame starts tucked under the one above and stands out further toward
        # its own lower edge, which is turned out into a rounded bead; so every joint is a step with a rib on it
        # and a shadow under it, and that is what reads as a groove. The joints (height at the sides, how far each
        # drops at the front, a V or a curve): two nested Vs under the knee, their points down the front ridge, and
        # two curving across low on the shin like bands.
        JOINTS = ((-1.3, 1.6, True), (-1.9, 1.6, True), (-4.5, 0.5, False), (-5.1, 0.5, False))
        LAP, BEAD = 0.1, 0.05     # (how far a lame stands off the next at its lower edge; its bead on top of that)

        def greave_pt(z, phi, out=0.0):
            """The greave's ground surface, on which the lames are raised (see lame_off)."""
            rx, ry, cy = ring_r(rings, z)
            # (slim: close over the boot, so the leg keeps its line instead of reading as a stovepipe)
            flare = 0.08 * (smoothstep(GT - 0.7, GT, z) + smoothstep(GB + 0.7, GB, z))
            ridge = 0.14 * math.exp(-(wrap(phi) / 0.3) ** 2)
            # (shaped to the leg: swelling over the calf behind, drawn in toward its foot)
            calf = 0.16 * math.exp(-((z + 3.4) / 1.7) ** 2) * max(0.0, -math.cos(phi)) ** 2
            ex = 0.17 + flare + out + calf - 0.02 * smoothstep(-3.5, GB, z)   # (not drawn in far: the boot's slack
            # wrinkles above the ankle came through the greave's foot)
            return -(rx + ex) * math.sin(phi), cy - (ry + ex + ridge) * math.cos(phi)

        def lame_edges(phi):
            a_ = abs(wrap(phi))
            return [GT] + [zs_ - dip * (max(0.0, 1.0 - a_ / 1.3) if vee else max(0.0, math.cos(a_)) ** 0.8)
                           for zs_, dip, vee in JOINTS] + [GB]

        def lame_rise(z, z0, z1):
            """How far a lame running from z0 down to z1 stands off the ground surface at z."""
            t = min(1.0, max(0.0, (z0 - z) / (z0 - z1)))
            return LAP * t ** 1.3 + BEAD * math.exp(-((z - z1 - 0.13) / 0.075) ** 2)

        def lame_off(z, phi):
            e = lame_edges(phi)
            for k in range(len(e) - 1):
                if z >= e[k + 1] or k == len(e) - 2:
                    return lame_rise(z, e[k], e[k + 1])

        def lamed_pt(z, phi, out=0.0):
            return greave_pt(z, phi, out + lame_off(z, phi))
        # (one surface, its rows following the joints: each lame has its own rows, closer together toward its beaded
        # edge, and the next lame's first row lies right under that edge on the ground surface, so the step is a
        # clean wall along the joint however the joint slants)
        LR, LC = 20, 128

        def lame_grid(i, j):
            k, u = i // LR, (i % LR) / (LR - 1)
            phi = lerp(-math.pi, math.pi, j / LC)
            e = lame_edges(phi)
            z = lerp(e[k] - (0.012 if k else 0.0), e[k + 1], 1.0 - (1.0 - u) ** 2)
            gx, gy = greave_pt(z, phi, lame_rise(z, e[k], e[k + 1]))
            return gx, gy, z
        gbm = face_out(grid(LR * (len(JOINTS) + 1), LC + 1, lame_grid), axis_y=0.0)
        bmesh.ops.remove_doubles(gbm, verts=gbm.verts, dist=1e-4)      # (the seam, up the back)
        mesh_obj(f'Greave.{side}', gbm, M_BRONZE, loc=(x, 0, kz), skin=w, solid=0.09)
        # the knee: a domed oval cop over the kneecap, its foot lapping over the greave's top; a fan-shaped wing on
        # the outside tucked under the cop's edge; a curved lame above, tucked under the cop's top. Each is a shell
        # lofted round the leg itself, so it wraps the knee instead of sitting on it.
        def knee_pt(z, ph, base, out=0.0):
            lx, ly = leg_pt(z - THIGH, ph)
            rl = math.hypot(lx, ly)
            gx, gy = lamed_pt(z, ph)
            need = math.hypot(gx, gy - 0.0) - rl + 0.11
            off = base + max(0.0, need - base) * smoothstep(GT + 0.7, GT - 0.1, z) + out
            return lx * (rl + off) / rl, ly * (rl + off) / rl, z

        def knee_shell(z0, z1, pc, pw, base, rows=18, cols=22, oval=True):
            zc, hz = (z0 + z1) / 2, (z0 - z1) / 2

            def pt(i, j):
                z = lerp(z0, z1, i / (rows - 1))
                u = (z - zc) / hz
                lim = pw * (math.sqrt(max(0.0, 1 - min(1.0, abs(u)) ** 2.4)) * 0.75 + 0.25 if oval else 1.0)
                ph = pc + lim * lerp(-1, 1, j / (cols - 1))
                v = (ph - pc) / pw
                return knee_pt(z, ph, base(u, v))
            return face_out(grid(rows, cols, pt), axis_y=0.0)
        cop = lambda u, v: 0.13 + 0.22 * (1 - u * u) * (1 - v * v)    # noqa: E731
        for nm, args in (('Knee wing', (0.95, -0.85, s * 1.5, 0.55, lambda u, v: 0.11 + 0.06 * (1 - v * v))),
                         ('Knee lame', (2.05, 1.05, 0.0, 1.3, lambda u, v: 0.09 + 0.04 * (1 - v * v))),
                         ('Knee cop', (1.35, -1.1, 0.0, 1.4, cop))):
            mesh_obj(f'{nm}.{side}', knee_shell(*args, oval=nm != 'Knee lame'), M_BRONZE, loc=(x, 0, kz), skin=w,
                     solid=0.13, subdiv=1)
        rv = [knee_pt(0.05, s * 1.55, 0.3), knee_pt(1.55, -1.05, 0.2), knee_pt(1.55, 1.05, 0.2),
              knee_pt(0.0, -1.05, cop(0.0, 0.75) + 0.12), knee_pt(0.0, 1.05, cop(0.0, 0.75) + 0.12)]
        mesh_obj(f'Knee rivets.{side}', dots(rv, 0.17), M_GOLD, loc=(x, 0, kz), skin=w, subdiv=0)
        # the leather straps holding the greave on, round the calf behind
        for j_, zs_ in enumerate((-3.2, -5.85)):
            def strap_pt(z, phi, zs_=zs_):
                # (over the metal behind; at the front it runs under the plate, between it and the boot)
                gx, gy = greave_pt(zs_, phi, lerp(-0.13, lame_off(zs_, phi) + 0.11, smoothstep(1.2, 1.45, abs(wrap(phi)))))
                return gx, gy
            mesh_obj(f'Greave strap {j_}.{side}', loft_fn([zs_ + 0.28, zs_, zs_ - 0.28], strap_pt, n=56, cap_top=False,
                                                         cap_bot=False), M_BELT, loc=(x, 0, kz), skin=w, solid=0.1, subdiv=0)
        # the greave's top and bottom edges rolled outward, and steel rivets down the front either side of the ridge
        # and at each strap's ends
        for z_, sg in ((GT, -1), (GB, 1)):
            rz = [z_ + 0.17 * math.cos(math.pi * k / 4) for k in range(5)]
            mesh_obj(f'Greave roll {"top" if sg < 0 else "foot"}.{side}',
                     loft_fn(rz, lambda z, p, z_=z_: lamed_pt(z_ + 0.02, p, 0.04 + 0.13 * max(0.0, 1 - ((z - z_) / 0.17) ** 2) ** 0.5),
                             n=56, cap_top=False, cap_bot=False), M_BIND, loc=(x, 0, kz), skin=w, subdiv=0)
        grv = []
        # (domed rivets holding each lame to the next, set above its beaded edge: a pair either side of the front
        # ridge and one on the outside of the shin; and one at each strap's ends)
        for ph in (-0.6, 0.6, s * 1.25):
            e = lame_edges(ph)
            for k in range(len(e) - 1):
                zr_ = e[k + 1] + 0.36
                if zr_ > e[k] - 0.12:
                    continue        # (no room on a lame this narrow here)
                gx, gy = lamed_pt(zr_, ph, 0.07)
                grv.append((gx, gy, zr_))
        for zs_ in (-3.2, -5.85):
            for ph in (-1.4, 1.4):
                gx, gy = lamed_pt(zs_, ph, 0.14)
                grv.append((gx, gy, zs_))
        mesh_obj(f'Greave rivets.{side}', dots(grv, 0.17), M_GOLD, loc=(x, 0, kz), skin=w, subdiv=0)
        # the shoe: a toe cap of paler leather sewn over the front of the foot, its seam stitched in a double row,
        # and a strap round the ankle buckled on the outside
        YC = -3.85

        tz0, tz1 = -SHIN - 1.65, -SHIN - 2.3
        toe = []
        for i in range(14):
            z = lerp(tz0, tz1, i / 13)
            rx, ry, cy = ring_r(rings, z)
            ring_ = []
            c = max(-1.0, min(1.0, (cy - YC) / ry))
            p0 = math.acos(c)
            for j in range(24):
                ph = lerp(-p0, p0, j / 23)
                ring_.append((-(rx + 0.05) * math.sin(ph), cy - (ry + 0.05) * math.cos(ph), z))
            toe.append(ring_)
        mesh_obj(f'Toe cap.{side}', face_out(grid(14, 24, lambda i, j: toe[i][j]), axis_y=0.0), M_CUFF, loc=(x, 0, kz),
                 skin=w, subdiv=1, solid=0.05)
        tst = []
        for i in range(10):
            z = lerp(tz0 - 0.05, tz1 + 0.05, i / 9)
            rx, ry, cy = ring_r(rings, z)
            for dy in (0.12, 0.34):
                c = max(-1.0, min(1.0, (cy - YC - dy) / ry))
                p0 = math.acos(c)
                for f in (-1, 1):
                    tst.append((-(rx + 0.1) * math.sin(f * p0), cy - (ry + 0.1) * math.cos(f * p0), z))
        mesh_obj(f'Toe stitching.{side}', dots(tst, 0.08), M_STITCH, loc=(x, 0, kz), skin=w, subdiv=0)
        za = -SHIN - 0.35
        mesh_obj(f'Ankle strap.{side}', loft_fn([za + 0.22, za, za - 0.22],
                                                lambda z, p: (lambda r: (-(r[0] + 0.08) * math.sin(p),
                                                                         r[2] - (r[1] + 0.08) * math.cos(p)))(ring_r(rings, za)),
                                                n=48, cap_top=False, cap_bot=False), M_BELT, loc=(x, 0, kz), skin=w,
                 solid=0.1, subdiv=0)
        ph = s * math.pi / 2
        rx, ry, cy = ring_r(rings, za)
        mesh_obj(f'Ankle buckle.{side}', turned(frame_bm(0.55, 0.52, 0.12, 0.12), ph), M_STEEL,
                 loc=(x - (rx + 0.2) * math.sin(ph), cy - (ry + 0.2) * math.cos(ph), kz + za), skin=w, subdiv=0)
        # the boot below the greave, fitted to the calf and ankle as the reference's: buckled straps up the outside of
        # the calf, a seam stitched up the back, a strap over the instep, and a stacked heel
        def boot_band(zc, out, h=0.22, span=1.15, c=None):
            # (a strap across the outside of the boot only: from the shin round to the back of the calf)
            rx, ry, cy = ring_r(rings, zc)

            def pt(i, j):
                ph = (s * math.pi / 2 if c is None else c) + lerp(-span, span, j / 23)
                z = zc + lerp(h, -h, i / 2)
                return (-(rx + out) * math.sin(ph), cy - (ry + out) * math.cos(ph), z)
            return face_out(grid(3, 24, pt), axis_y=0.0)
        for j_, zc in enumerate((GB - 1.0, GB - 2.7, GB - 4.4)):
            mesh_obj(f'Boot strap {j_}.{side}', boot_band(zc, 0.07), M_BELT, loc=(x, 0, kz), skin=w, solid=0.1, subdiv=0)
            ph = s * math.pi / 2
            rx, ry, cy = ring_r(rings, zc)
            mesh_obj(f'Boot buckle {j_}.{side}', turned(frame_bm(0.5, 0.48, 0.11, 0.12), ph), M_STEEL,
                     loc=(x - (rx + 0.2) * math.sin(ph), cy - (ry + 0.2) * math.cos(ph), kz + zc), skin=w, subdiv=0)
        seam = []
        for k in range(26):
            zz = lerp(GB - 0.3, -SHIN - 0.2, k / 25)
            rx, ry, cy = ring_r(rings, zz)
            for dp in (-0.07, 0.07):
                seam.append((-(rx + 0.06) * math.sin(math.pi + dp), cy - (ry + 0.06) * math.cos(math.pi + dp), zz))
        mesh_obj(f'Boot seam.{side}', dots(seam, 0.08), M_STITCH, loc=(x, 0, kz), skin=w, subdiv=0)
        zi = -SHIN - 1.25
        mesh_obj(f'Instep strap.{side}', boot_band(zi, 0.07, 0.2, 1.75, 0.0), M_BELT, loc=(x, 0, kz), skin=w, solid=0.1, subdiv=0)
        rx, ry, cy = ring_r(rings, zi)
        ph = s * math.pi / 2
        mesh_obj(f'Instep buckle.{side}', turned(frame_bm(0.48, 0.44, 0.11, 0.12), ph), M_STEEL,
                 loc=(x - (rx + 0.2) * math.sin(ph), cy - (ry + 0.2) * math.cos(ph), kz + zi), skin=w, subdiv=0)
        mesh_obj(f'Heel.{side}', loft_fn([-SHIN - 2.88, -SHIN - 2.93, -SHIN - ANKLE_Z + 0.05, -SHIN - ANKLE_Z],
                                         lambda z, p: (-1.3 * math.sin(p), 0.95 - 0.9 * math.cos(p)), n=32),
                 M_BIND, loc=(x, 0, kz), skin=w, subdiv=1)
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
    # the rig, the camera and light, and the keyframes; then the hair kept clear of the clothes in every pose
    res = finish_figure(lambda: cape_on_arms(['Cape', 'Cape lining', 'Cape front -1', 'Cape front 1']))
    steel_closeup('plate')
    bronze_closeup('bronze')
    # (the collar and its plate one dull, even finish: rougher, so the flat plate facing the light does not flare
    # brighter than the band curving round the neck)
    steel_closeup('collar', dark=(0.04, 0.041, 0.045), light=(0.2, 0.205, 0.22), rgh=(0.62, 0.48), use_ao=False)
    velvet_closeup('robe')
    velvet_closeup('cape')
    patina_closeup('bracers')
    if not os.environ.get('WARDEN_NO_CLEARANCE') and not female.MANNEQUIN:
        # (twice: the first pass's easing lets a little back in where a lock crosses the shoulder in the walk)
        import time
        t_ = time.time()
        creator_hair_clearance()
        creator_hair_clearance()
        print(f'hair clearance {time.time() - t_:.0f} s', flush=True)
        t_ = time.time()
        creator_hair_trim()
        print(f'hair trim {time.time() - t_:.0f} s', flush=True)
    hair_contact_shade()
    return res


def hair_contact_shade(reach=1.3, near=0.25, depth=1.0):
    """The shade the hair casts on the skin it lies over. The locks beside the face hang half a unit to a unit in
    front of the cheek and the jaw; lit from the front, their shadow falls behind them, out of sight, and the face
    stayed bright right up to the hair, a pale mask laid in front of it. (The creator's key casts the hair's shadow
    generously onto the cheeks.) Each vertex of the head gets how near the hair is on its open side, as 'hair occ',
    0 to 1, and the skin's close-up colour goes darker and warmer by it: deepest under a lock's edge, gone a hand's
    breadth in. Meshes without the attribute read 0 and keep their colour."""
    from mathutils.bvhtree import BVHTree
    head = bpy.data.objects.get('Head shape')
    if head is None:
        return
    vs, ps = [], []
    for o in bpy.data.objects:
        # (the locks only: the cap lies right over the scalp, and with it the whole hairline went dark, a dirty band
        # across the forehead)
        if o.type != 'MESH' or not o.name.startswith('Hair locks'):
            continue
        b = len(vs)
        vs += [o.matrix_world @ v.co for v in o.data.vertices]
        ps += [[b + i for i in p.vertices] for p in o.data.polygons]
    if not ps:
        return
    tree = BVHTree.FromPolygons(vs, ps)
    # (the neck below takes no shade, so the head's own shade runs out where the head meets it, and no tone step
    # shows along the join)
    neck_t = None
    nk = bpy.data.objects.get('Neck')
    if nk is not None:
        dg = bpy.context.evaluated_depsgraph_get()
        nm = nk.evaluated_get(dg).to_mesh()
        neck_t = BVHTree.FromPolygons([nk.matrix_world @ v.co for v in nm.vertices],
                                      [list(p.vertices) for p in nm.polygons])
        nk.evaluated_get(dg).to_mesh_clear()
    # (the head and what is painted on it: the brows' paint and the eyeshadow are their own meshes, close-up only,
    # coloured by their own materials; left unshaded, a brow under the fringe came out paler than the shaded skin
    # round it, a light streak where the brow should be)
    objs = [head] + [o for o in bpy.data.objects if o.type == 'MESH' and o.parent == head.parent
                     and o.name.startswith(('Brow paint', 'Eyeshadow'))]
    for ob in objs:
        M = ob.matrix_world
        N3 = M.to_3x3().inverted().transposed()
        me = ob.data
        occ = [0.0] * len(me.vertices)
        for v in me.vertices:
            p = M @ v.co
            q, _n, _i, d = tree.find_nearest(p, reach)
            if q is None:
                continue
            n = (N3 @ v.normal).normalized()
            # (only hair out on the side the skin faces: hair behind the ear, past the edge of the jaw, shades
            # nothing)
            facing = smoothstep(-0.35, 0.25, (q - p).dot(n) / max(d, 1e-6))
            # (but hair touching the skin shades it whichever side its nearest point lies: where a fringe lock grazes
            # the brow ridge its nearest point is under the surface, and the shade had a hole there, a pale streak
            # along the brow)
            facing = max(facing, 1.0 - smoothstep(near, near + 0.35, d))
            dn = neck_t.find_nearest(p, 0.8)[3] if neck_t is not None else None
            seam = 1.0 if dn is None else smoothstep(0.1, 0.8, dn)
            occ[v.index] = seam * facing * (1.0 - smoothstep(near, reach, d))
        at = me.attributes.get('hair occ') or me.attributes.new('hair occ', 'FLOAT', 'POINT')
        at.data.foreach_set('value', occ)
        me.update()
    for m in {s_.material for ob in objs for s_ in ob.material_slots if s_.material is not None}:
        _hair_shade_material(m, depth)


def _hair_shade_material(m, depth):
    """Darken and warm a material's colour by the mesh's 'hair occ' (the close-up skin's base colour, or the colour
    of a paint's or the brows' own shader)."""
    nt = m.node_tree
    if nt.nodes.get('hair occ') is not None:
        return
    bsdf = nt.nodes.get('close bsdf')
    if bsdf is None:
        out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL' and n.inputs['Surface'].links), None)
        if out is None:
            print(f'hair contact shade: no output in material {m.name!r}', flush=True)
            return
        bsdf = out.inputs['Surface'].links[0].from_node
    sock = bsdf.inputs.get('Base Color') or bsdf.inputs.get('Color')
    if sock is None:
        print(f'hair contact shade: no colour to shade in material {m.name!r} ({bsdf.type})', flush=True)
        return
    if sock.links:
        src = sock.links[0].from_socket
    else:
        rgb = nt.nodes.new('ShaderNodeRGB')
        rgb.outputs[0].default_value = tuple(sock.default_value)
        src = rgb.outputs[0]
    a = nt.nodes.new('ShaderNodeAttribute')
    a.name = a.label = 'hair occ'
    a.attribute_type = 'GEOMETRY'
    a.attribute_name = 'hair occ'
    k = nt.nodes.new('ShaderNodeMath')
    k.operation = 'MULTIPLY'
    k.use_clamp = True
    k.inputs[1].default_value = depth
    nt.links.new(a.outputs['Fac'], k.inputs[0])
    mix = nt.nodes.new('ShaderNodeMix')
    mix.data_type, mix.blend_type = 'RGBA', 'MULTIPLY'
    mix.inputs['B'].default_value = (0.4, 0.27, 0.23, 1.0)    # (a shadow on skin: darker, and warmer)
    nt.links.new(k.outputs[0], mix.inputs['Factor'])
    nt.links.new(src, mix.inputs['A'])
    nt.links.new(mix.outputs['Result'], sock)
    # (and the shine: a shadow takes the highlight too. Left at full, the brow ridge's top under the fringe kept
    # its highlight in the shaded skin round it, a pale streak across the brows)
    spec = bsdf.inputs.get('Specular IOR Level')
    if spec is not None and not spec.links:
        f = nt.nodes.new('ShaderNodeMath')
        f.operation = 'MULTIPLY_ADD'
        f.use_clamp = True
        f.inputs[1].default_value = -0.9 * spec.default_value
        f.inputs[2].default_value = spec.default_value
        nt.links.new(k.outputs[0], f.inputs[0])
        nt.links.new(f.outputs[0], spec)

if __name__ == '__main__':
    run(build, 'warden', HERE)
