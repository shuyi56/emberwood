"""The Ranger, modelled in Blender and rendered to a pixel-art sprite sheet.

Builds the ranger as a low-poly 3D figure (long auburn hair, a green mantle with a leaf clasp, green sleeves,
a brown leather jerkin and belt, grey-brown trousers with knee patches, soft folded-top boots), poses her for
the idle and walk frames, and renders the 8 facings with Cycles. The renders are then turned into pixel art:
every pixel takes its colour from the five-tone palette ramp of the material it shows (the same ramps as
src/shared/core.js), picked by how strongly it is lit, and the figure gets the painter's coloured outline.

The sheet matches the painted sheets in public/sprites/: 10 columns (2 idle frames, then an 8-frame walk) by
8 facing rows (front, then turning toward screen right: front-3/4, right profile, back-3/4, back, and the
mirror side), each frame 128 x 160 pixels (the 64 x 80 design grid at RES 2), ground at row 154.

Run it with Blender's Python (either works):
    blender -b -P blender/ranger/ranger.py -- [--build-only] [--out DIR]
    python blender/ranger/ranger.py [--build-only] [--out DIR]      (with `pip install bpy OpenEXR`)

It writes blender/ranger/ranger.blend (the model, rigged with pivots and keyed with the 10 frames on the
timeline) and blender/ranger/ranger-blender.png (the sheet). Units: one Blender unit is one design pixel
(two sheet pixels); the figure faces -Y, the camera looks along +Y.
"""
import math
import os
import sys
import tempfile

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Euler

HERE = os.path.dirname(os.path.abspath(__file__))
FW, FH, RES, COLS, ROWS, WALK = 64, 80, 2, 10, 8, 8
PW, PH = FW * RES, FH * RES
GY = 77  # ground row in design pixels from the top of the frame


# ---------------------------------------------------------------- palette (a port of ramp() in core.js)
def _rgb2hsl(r, g, b):
    r, g, b = r / 255, g / 255, b / 255
    mx, mn = max(r, g, b), min(r, g, b)
    l = (mx + mn) / 2
    if mx == mn:
        return 0.0, 0.0, l
    d = mx - mn
    s = d / (2 - mx - mn) if l > 0.5 else d / (mx + mn)
    if mx == r:
        h = (g - b) / d + (6 if g < b else 0)
    elif mx == g:
        h = (b - r) / d + 2
    else:
        h = (r - g) / d + 4
    return h / 6, s, l


def _hsl2rgb(h, s, l):
    def f(n):
        k = (n + h * 12) % 12
        a = s * min(l, 1 - l)
        return 255 * (l - a * max(-1, min(k - 3, 9 - k, 1)))
    return [f(0), f(8), f(4)]


def _hue_toward(h, target, t):
    d = target - h
    d -= round(d)
    return (h + d * t + 1) % 1


def ramp(hexv, spread=0.1):
    h, s0, l = _rgb2hsl((hexv >> 16) & 255, (hexv >> 8) & 255, hexv & 255)
    s = min(s0, 0.46)
    spread *= 0.86
    out = []
    for t in (-2, -1, 0, 1, 2):
        hh = _hue_toward(h, 0.62, -t * 0.05) if t < 0 else _hue_toward(h, 0.13, t * 0.045)
        ss = max(0, min(1, s * (1 - abs(t) * 0.08)))
        ll = max(0.06, min(0.9, l + t * spread * (1.0 if t < 0 else 0.8)))
        out.append(_hsl2rgb(hh, ss, ll))
    return out


SKIN = [[160, 100, 80], [191, 134, 108], [207, 159, 132], [218, 180, 156], [229, 203, 184]]
# id: (name, ramp, detail amplitude, flat tone or None)
MATERIALS = {}


def mat_def(mid, name, rmp, detail=0.0, flat=None):
    MATERIALS[mid] = dict(name=name, ramp=np.array(rmp, dtype=np.float32), detail=detail, flat=flat)
    return mid


M_SKIN = mat_def(1, 'skin', SKIN, 0.0)
M_HAIR = mat_def(2, 'hair', ramp(0x7e4c34, 0.1), 0.22)
M_MANTLE = mat_def(3, 'mantle', ramp(0x56663f, 0.1), 0.06)
M_JERKIN = mat_def(4, 'jerkin', ramp(0x765a40, 0.1), 0.10)
M_SHIRT = mat_def(5, 'shirt', ramp(0xaca48d, 0.09), 0.04)
M_SLEEVE = mat_def(6, 'sleeve', ramp(0x63704a, 0.1), 0.06)
M_LEGS = mat_def(7, 'trousers', ramp(0x5b5547, 0.1), 0.06)
M_KNEE = mat_def(8, 'knee patch', ramp(0x7a6c52, 0.1), 0.0)
M_BOOTS = mat_def(9, 'boots', ramp(0x684a32, 0.1), 0.10)
M_CUFF = mat_def(10, 'boot cuff', ramp(0x7a5b40, 0.1), 0.06)
M_SOLE = mat_def(11, 'sole', ramp(0x2f2620, 0.08), 0.0)
M_BELT = mat_def(12, 'belt', ramp(0x3d342b, 0.09), 0.0)
M_STEEL = mat_def(13, 'buckle', ramp(0x9a978f, 0.09), 0.0)
M_GOLD = mat_def(14, 'clasp', ramp(0xa8905c, 0.09), 0.0)
M_MAIL = mat_def(15, 'mail trim', ramp(0x7e827c, 0.08), 0.35)
M_EYE = mat_def(16, 'iris', [[46, 40, 38], [58, 66, 46], [88, 108, 70], [120, 140, 100], [226, 222, 210]], 0.0, flat=1)
M_LASH = mat_def(20, 'lashes', [[46, 40, 38]] * 5, 0.0, flat=0)
M_SCLERA = mat_def(21, 'eye white', [[226, 222, 210]] * 5, 0.0, flat=0)
M_BROW = mat_def(17, 'brow', ramp(0x60463a, 0.08), 0.0, flat=1)
M_LIPS = mat_def(18, 'lips', [[150, 86, 74], [168, 100, 86], [184, 116, 100], [196, 132, 114], [210, 150, 132]], 0.0, flat=1)
M_LACE = mat_def(19, 'laces', ramp(0x4a3a2c, 0.08), 0.0)


# ---------------------------------------------------------------- scene and materials
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def make_material(mid):
    d = MATERIALS[mid]
    m = bpy.data.materials.new(d['name'])
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfDiffuse')
    mid_rgb = d['ramp'][2] / 255.0
    lin = [((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92 for c in mid_rgb]
    bsdf.inputs['Color'].default_value = (*lin, 1.0)
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    m.diffuse_color = (*lin, 1.0)
    # the material's id, read back after the render to pick its palette ramp
    aid = nt.nodes.new('ShaderNodeOutputAOV')
    aid.aov_name = 'matid'
    val = nt.nodes.new('ShaderNodeValue')
    val.outputs[0].default_value = mid / 100.0
    nt.links.new(val.outputs[0], aid.inputs['Value'])
    # a surface pattern (hair strands, cloth weave, leather mottling), added to the light before it is banded
    adet = nt.nodes.new('ShaderNodeOutputAOV')
    adet.aov_name = 'detail'
    coord = nt.nodes.new('ShaderNodeTexCoord')
    if mid == M_HAIR:
        tex = nt.nodes.new('ShaderNodeTexWave')
        tex.wave_type = 'BANDS'
        tex.bands_direction = 'X'
        tex.inputs['Scale'].default_value = 0.22
        tex.inputs['Distortion'].default_value = 0.6
        tex.inputs['Detail'].default_value = 0.0
        nt.links.new(coord.outputs['Object'], tex.inputs['Vector'])
        nt.links.new(tex.outputs['Fac'], adet.inputs['Value'])
    elif mid == M_MAIL:
        tex = nt.nodes.new('ShaderNodeTexChecker')
        tex.inputs['Scale'].default_value = 1.0
        nt.links.new(coord.outputs['Object'], tex.inputs['Vector'])
        nt.links.new(tex.outputs['Fac'], adet.inputs['Value'])
    else:
        tex = nt.nodes.new('ShaderNodeTexNoise')
        tex.inputs['Scale'].default_value = 0.35
        tex.inputs['Detail'].default_value = 0.0
        nt.links.new(coord.outputs['Object'], tex.inputs['Vector'])
        nt.links.new(tex.outputs['Fac'], adet.inputs['Value'])
    return m


MATS = {}


def mat(mid):
    if mid not in MATS:
        MATS[mid] = make_material(mid)
    return MATS[mid]


def link(obj, parent=None):
    bpy.context.scene.collection.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    return obj


def empty(name, loc, parent=None):
    e = bpy.data.objects.new(name, None)
    e.empty_display_size = 1.5
    e.location = loc
    e.rotation_mode = 'XYZ'
    return link(e, parent)


def mesh_obj(name, bm, mid, parent=None, loc=(0, 0, 0), smooth=True, subdiv=1, solid=0.0):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = smooth
    o = bpy.data.objects.new(name, me)
    o.location = loc
    me.materials.append(mat(mid))
    link(o, parent)
    if solid:
        m = o.modifiers.new('thickness', 'SOLIDIFY')
        m.thickness = solid
        m.offset = 1.0
    if subdiv:
        m = o.modifiers.new('smooth', 'SUBSURF')
        m.levels = subdiv
        m.render_levels = subdiv
    return o


def loft(rings, n=24, cap_top=True, cap_bot=True):
    """A tube through rings, top to bottom. Each ring is (z, rx, ry[, cx, cy[, dz(phi)]]): an ellipse at height z,
    phi measured round from the front (-Y) toward the figure's right (-X), with dz(phi) shaping a hem."""
    bm = bmesh.new()
    loops = []
    for r in rings:
        z, rx, ry = r[0], r[1], r[2]
        cx = r[3] if len(r) > 3 else 0.0
        cy = r[4] if len(r) > 4 else 0.0
        dz = r[5] if len(r) > 5 else None
        loop = []
        for k in range(n):
            phi = 2 * math.pi * k / n
            x = cx - rx * math.sin(phi)
            y = cy - ry * math.cos(phi)
            loop.append(bm.verts.new((x, y, z + (dz(phi) if dz else 0.0))))
        loops.append(loop)
    for a, b in zip(loops, loops[1:]):
        for k in range(n):
            bm.faces.new((a[k], a[(k + 1) % n], b[(k + 1) % n], b[k]))
    if cap_top:
        bm.faces.new(list(reversed(loops[0])))
    if cap_bot:
        bm.faces.new(loops[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def ellipsoid(rx, ry, rz, seg=20, rings=12):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings, radius=1.0)
    bmesh.ops.scale(bm, vec=(rx, ry, rz), verts=bm.verts)
    return bm


def box(sx, sy, sz):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts)
    return bm


# ---------------------------------------------------------------- the figure
# Realistic adult proportions, about 7.3 heads tall: the top of the head at 63 (the painted sprites' height),
# a head 8.6 tall, shoulders at 1.5 heads down, the waist at 3, the hip joint at 3.7 and the knee at 2 heads up.
HEAD_H, HEAD_Z = 8.6, 54.2          # head height and the height of the chin
HIP_Z, SH_Z, ANKLE_Z = 31.0, 49.6, 3.4
THIGH, SHIN = 14.0, 13.8            # hip -> knee -> ankle, a touch longer than straight so the knee stays soft
UPPER, FORE = 11.0, 8.6
LEG_X, SH_X = 3.3, 6.3
STRIDE, LIFT = 8.0, 3.2


def gauss(x, z, x0, z0, sx, sz, amp):
    return amp * math.exp(-((x - x0) / sx) ** 2 - ((z - z0) / sz) ** 2)


def face_relief(x, z):
    """How far the face stands forward of the base head at (x across, z up), in head heights. The planes a
    painter would block in: brow ridge over the eye sockets, the nose (bridge, tip and wings), cheekbones with
    the hollow under them, the lips parted by the mouth line, and the ball of the chin. (They are pushed a little
    past life size, so they still read when the head is 17 pixels tall.)"""
    d = 0.0
    for s in (-1, 1):
        d += gauss(x, z, s * 0.15, 0.62, 0.12, 0.04, 0.03)       # brow ridge
        d += gauss(x, z, s * 0.14, 0.525, 0.08, 0.05, -0.06)     # eye socket
        d += gauss(x, z, s * 0.215, 0.45, 0.08, 0.06, 0.035)     # cheekbone
        d += gauss(x, z, s * 0.2, 0.31, 0.07, 0.06, -0.025)      # hollow under the cheekbone
        d += gauss(x, z, s * 0.05, 0.37, 0.03, 0.025, 0.035)     # wing of the nose
    # the nose: a narrow ridge rising from the bridge to the tip, then turning under sharply
    if z > 0.36:
        t = min(1.0, max(0.0, (0.6 - z) / 0.24))
        h = (0.015 + 0.085 * t ** 1.6) * math.exp(-max(0.0, z - 0.58) ** 2 / 0.0008)
    else:
        h = 0.1 * math.exp(-((0.36 - z) / 0.025) ** 2)
    d += h * math.exp(-(x / (0.04 + 0.03 * max(0.0, 1 - abs(z - 0.37) / 0.1))) ** 2)
    d += gauss(x, z, 0, 0.27, 0.16, 0.1, 0.05)       # the muzzle: the teeth and jaw carry the mouth forward
    d += gauss(x, z, 0, 0.29, 0.07, 0.025, 0.035)    # upper lip
    d += gauss(x, z, 0, 0.225, 0.055, 0.025, 0.03)   # lower lip
    d += gauss(x, z, 0, 0.258, 0.08, 0.01, -0.018)   # mouth line
    d += gauss(x, z, 0, 0.17, 0.06, 0.02, -0.015)    # the crease above the chin
    d += gauss(x, z, 0, 0.09, 0.075, 0.045, 0.04)    # chin
    return d


def _jaw_width(z):
    # full through the cheeks, then the jaw: a clear angle at z 0.2, narrowing to a rounded chin
    if z >= 0.42:
        return 1.0
    if z >= 0.2:
        t = (0.42 - z) / 0.22
        return 1.0 - 0.2 * t ** 1.3
    t = (0.2 - z) / 0.2
    return 0.8 - 0.42 * t ** 1.4


def sculpted_head():
    """The head as a sculpted mesh: a rounded skull, cheeks over a jaw with a defined angle, a rounded chin, and
    the face's planes pushed out of the front (see face_relief). Units are head heights: chin at z 0, crown at
    z 1, the face toward -Y."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=128, v_segments=96, radius=1.0)
    for v in bm.verts:
        sx, sy, sz = v.co
        z = 0.52 + 0.5 * sz
        front = sy < 0
        t = max(0.0, (0.5 - z) / 0.5)
        # the face is flatter than an egg across the front; the back of the jaw tucks in under the ear
        dep = (0.92 - 0.15 * t) if front else (0.98 - 0.6 * t ** 1.3)
        x = sx * 0.36 * _jaw_width(z)
        y = sy * 0.44 * dep
        f = max(0.0, -sy) ** 1.4          # only the parts facing forward carry the face
        y -= face_relief(x, z) * f
        v.co = (x, y + 0.04, max(0.0, z))
    bmesh.ops.scale(bm, vec=(HEAD_H, HEAD_H, HEAD_H), verts=bm.verts)
    return bm


def surface_y(bm, x, z):
    """The front of the head's surface at (x, z): where a ray from the camera side first meets it."""
    from mathutils.bvhtree import BVHTree
    tree = BVHTree.FromBMesh(bm)
    hit = tree.ray_cast(Vector((x, -50, z)), Vector((0, 1, 0)))
    return hit[0].y if hit[0] is not None else 0.0


def build():
    reset()
    sc = bpy.context.scene
    root = empty('Ranger', (0, 0, 0))
    body = empty('Body', (0, 0, 0), root)   # carries the walk's bob

    # ---- torso: the shirt underneath, then the jerkin over it
    mesh_obj('Shirt', loft([(53.0, 1.6, 1.6, 0, 0.4), (51.4, 2.7, 2.4, 0, 0.3), (50.3, 5.0, 2.9), (49.0, 6.3, 3.3), (47.0, 6.2, 3.5),
                            (44.5, 5.9, 3.85), (42.0, 5.1, 3.35), (39.0, 4.5, 2.95), (36.0, 5.3, 3.25), (33.0, 6.3, 3.65),
                            (30.5, 6.5, 3.75), (29.0, 5.4, 3.3)]), M_SHIRT, body)
    jerk = lambda phi: -1.0 * math.cos(phi) ** 2   # the skirt dips a little at front and back
    mesh_obj('Jerkin', loft([(47.6, 6.45, 3.75), (44.5, 6.2, 4.15), (42.0, 5.4, 3.65), (39.5, 4.8, 3.25), (37.5, 5.1, 3.4),
                             (35.0, 6.05, 3.75), (32.5, 6.75, 3.95), (30.0, 7.0, 4.15, 0, 0, jerk), (28.0, 7.15, 4.25, 0, 0, jerk)],
                            cap_bot=False), M_JERKIN, body, solid=0.35)
    # a darker leather band binds the jerkin's hem
    mesh_obj('Jerkin hem', loft([(29.2, 7.12, 4.25, 0, 0, jerk), (28.0, 7.22, 4.32, 0, 0, jerk)], cap_top=False, cap_bot=False),
             M_BELT, body, solid=0.3, subdiv=0)
    # the shirt shows in a narrow V at the front of the jerkin, crossed by its lacing
    vee = bmesh.new()
    vs = [vee.verts.new(p) for p in [(-1.3, -4.18, 47.0), (1.3, -4.18, 47.0), (0, -3.8, 42.4)]]
    vee.faces.new(vs)
    mesh_obj('Shirt V', vee, M_SHIRT, body, subdiv=0)
    for i, z in enumerate((46.0, 44.7, 43.4)):
        w = 1.15 * (z - 42.4) / 4.6
        mesh_obj(f'Lace {i}', box(2 * w + 0.4, 0.3, 0.45), M_LACE, body, loc=(0, -4.25 + 0.08 * i, z), subdiv=0)
    # belt and buckle at the waist
    mesh_obj('Belt', loft([(40.0, 5.0, 3.45), (38.5, 5.05, 3.5)]), M_BELT, body, subdiv=0)
    mesh_obj('Buckle', box(1.7, 0.6, 1.7), M_STEEL, body, loc=(0, -3.55, 39.25), subdiv=0)
    mesh_obj('Buckle hole', box(0.8, 0.3, 0.8), M_BELT, body, loc=(0, -3.8, 39.25), subdiv=0)

    # ---- neck and head
    mesh_obj('Neck', loft([(57.0, 1.45, 1.55, 0, 0.55), (51.0, 1.7, 1.75, 0, 0.45)]), M_SKIN, body)
    head = empty('Head', (0, 0, HEAD_Z), body)
    hb = sculpted_head()
    # the eyes, brows and mouth sit on the sculpted surface (found by casting rays at it)
    feat = {}
    for s in (-1, 1):
        x = s * 0.14 * HEAD_H
        feat[s] = (x, surface_y(hb, x, 0.525 * HEAD_H), surface_y(hb, x, 0.635 * HEAD_H))
    mouth_y = surface_y(hb, 0, 0.258 * HEAD_H)
    fore_y = {s: surface_y(hb, s * 1.4, 0.79 * HEAD_H) for s in (-1, 1)}
    mesh_obj('Head shape', hb, M_SKIN, head, subdiv=0)
    for s in (-1, 1):
        x, ey, by = feat[s]
        ez = 0.525 * HEAD_H
        # an almond of white set into the socket, the iris in the middle, a dark lash line across the top
        mesh_obj(f'Eye white {s}', ellipsoid(0.85, 0.2, 0.4, 12, 8), M_SCLERA, head, loc=(x, ey + 0.16, ez), subdiv=0)
        mesh_obj(f'Iris {s}', ellipsoid(0.4, 0.2, 0.42, 10, 8), M_EYE, head, loc=(x - s * 0.05, ey + 0.02, ez - 0.03), subdiv=0)
        mesh_obj(f'Lash {s}', ellipsoid(0.95, 0.25, 0.16, 12, 6), M_LASH, head, loc=(x + s * 0.06, ey + 0.02, ez + 0.38), subdiv=0)
        brow = ellipsoid(0.9, 0.2, 0.15, 12, 6)
        bmesh.ops.rotate(brow, cent=(0, 0, 0), matrix=Euler((0, -s * 0.1, 0)).to_matrix(), verts=brow.verts)
        mesh_obj(f'Brow {s}', brow, M_BROW, head, loc=(x + s * 0.05, by + 0.2, 0.645 * HEAD_H), subdiv=0)
    mesh_obj('Mouth', ellipsoid(0.55, 0.2, 0.11, 10, 6), M_LIPS, head, loc=(0, mouth_y + 0.05, 0.258 * HEAD_H), subdiv=0)

    # ---- hair: a cap over the skull cut along a sloping hairline, curtain bangs parted at the centre,
    # side locks framing the face, and the long fall down the back
    cap = ellipsoid(0.395 * HEAD_H, 0.48 * HEAD_H, 0.54 * HEAD_H, 32, 20)
    bmesh.ops.translate(cap, vec=(0, 0.06 * HEAD_H, 0.56 * HEAD_H), verts=cap.verts)
    bmesh.ops.bisect_plane(cap, geom=cap.verts[:] + cap.edges[:] + cap.faces[:], plane_co=(0, -2.1, 6.4),
                           plane_no=Vector((0, -0.45, -0.89)).normalized(), clear_outer=True)
    mesh_obj('Hair cap', cap, M_HAIR, head, solid=0.4)
    for s in (-1, 1):
        b = ellipsoid(1.55, 0.5, 1.15)
        bmesh.ops.rotate(b, cent=(0, 0, 0), matrix=Euler((0.35, s * 0.55, 0)).to_matrix(), verts=b.verts)
        mesh_obj(f'Bangs {s}', b, M_HAIR, head, loc=(s * 1.4, fore_y[s] + 0.25, 0.79 * HEAD_H))
        mesh_obj(f'Side lock {s}', loft([(6.6, 0.85, 1.3, s * 3.05, 0.3), (4.5, 0.85, 1.35, s * 3.45, 0.9), (2.0, 0.8, 1.3, s * 3.6, 1.35),
                                         (-1.0, 0.8, 1.25, s * 3.8, 1.5), (-3.2, 0.8, 1.1, s * 4.05, 1.5), (-4.4, 0.35, 0.5, s * 4.15, 1.5)]),
                 M_HAIR, head)
    mesh_obj('Hair fall', loft([(7.5, 3.4, 3.4, 0, 0.75), (4.5, 3.75, 4.05, 0, 0.9), (2.0, 3.75, 2.9, 0, 2.3), (0.5, 3.7, 2.1, 0, 3.3), (-1.5, 3.8, 1.6, 0, 4.4),
                                (-4.5, 4.15, 1.4, 0, 5.45), (-9.0, 4.25, 1.3, 0, 5.4), (-13.0, 3.95, 1.2, 0, 4.6),
                                (-16.0, 3.55, 1.0, 0, 4.4), (-16.8, 2.4, 0.6, 0, 4.4)]), M_HAIR, head)

    # ---- mantle: a short shoulder cape, its hem lifting over the arms; a mail trim along the hem and a leaf clasp
    hem = lambda phi: 1.4 * math.sin(phi) ** 2 - 0.5 * math.cos(phi)
    mesh_obj('Mantle', loft([(53.4, 2.3, 2.3, 0, 0.35), (52.4, 3.6, 3.1, 0, 0.35), (50.8, 6.0, 4.15, 0, 0.3), (49.2, 7.9, 4.8, 0, 0.3),
                             (47.4, 8.6, 5.05, 0, 0.3, hem), (45.4, 8.85, 5.2, 0, 0.3, hem)], cap_top=False, cap_bot=False),
             M_MANTLE, body, solid=0.45)
    mesh_obj('Mail trim', loft([(45.9, 8.95, 5.3, 0, 0.3, hem), (45.0, 9.0, 5.35, 0, 0.3, hem)], cap_top=False, cap_bot=False),
             M_MAIL, body, solid=0.4, subdiv=0)
    mesh_obj('Collar', loft([(54.0, 2.0, 2.05, 0, 0.4), (52.8, 2.5, 2.55, 0, 0.35)], cap_top=False, cap_bot=False), M_SHIRT, body, solid=0.3)
    clasp = ellipsoid(0.9, 0.4, 1.3, 12, 8)
    bmesh.ops.rotate(clasp, cent=(0, 0, 0), matrix=Euler((0, 0.6, 0)).to_matrix(), verts=clasp.verts)
    mesh_obj('Leaf clasp', clasp, M_GOLD, body, loc=(-0.7, -4.85, 50.6), subdiv=0)

    # ---- arms: shoulder -> elbow -> wrist pivots; sleeves with a rounded cap, shirt cuffs and hands
    for side, s in (('R', 1), ('L', -1)):
        x = -s * SH_X   # the figure's right is -X (she faces -Y)
        sh = empty(f'Shoulder.{side}', (x, 0.2, SH_Z), body)
        mesh_obj(f'Sleeve cap.{side}', ellipsoid(2.15, 2.2, 2.2), M_SLEEVE, sh, loc=(0, 0, -0.6))
        mesh_obj(f'Upper sleeve.{side}', loft([(0.0, 2.0, 2.05), (-5.0, 1.8, 1.85), (-UPPER, 1.6, 1.65)]), M_SLEEVE, sh)
        el = empty(f'Elbow.{side}', (0, 0, -UPPER), sh)
        mesh_obj(f'Fore sleeve.{side}', loft([(0.8, 1.6, 1.65), (-3.5, 1.6, 1.65), (-FORE + 0.6, 1.45, 1.5)]), M_SLEEVE, el)
        mesh_obj(f'Shirt cuff.{side}', loft([(-FORE + 1.0, 1.6, 1.65), (-FORE - 0.4, 1.65, 1.7)]), M_SHIRT, el, subdiv=0)
        wr = empty(f'Wrist.{side}', (0, 0, -FORE), el)
        mesh_obj(f'Hand.{side}', ellipsoid(0.8, 1.05, 1.65, 12, 8), M_SKIN, wr, loc=(-s * 0.1, -0.15, -1.55))

    # ---- legs: hip -> knee -> ankle pivots; trousers with knee patches, soft boots with a folded cuff
    for side, s in (('R', 1), ('L', -1)):
        x = -s * LEG_X
        hip = empty(f'Hip.{side}', (x, 0, HIP_Z), body)
        mesh_obj(f'Thigh.{side}', loft([(2.0, 3.1, 3.2), (-1.0, 3.0, 3.1), (-6.0, 2.55, 2.65), (-THIGH, 1.95, 2.05)]), M_LEGS, hip)
        kn = empty(f'Knee.{side}', (0, 0, -THIGH), hip)
        mesh_obj(f'Shin.{side}', loft([(0.7, 1.95, 2.05), (-4.0, 2.0, 2.1), (-SHIN, 1.35, 1.45)]), M_LEGS, kn)
        mesh_obj(f'Knee round.{side}', ellipsoid(2.0, 2.1, 2.2), M_LEGS, kn)   # rounds the joint, so the leg bends without a notch
        mesh_obj(f'Knee patch.{side}', ellipsoid(1.15, 0.5, 1.3, 12, 8), M_KNEE, kn, loc=(0, -1.9, -0.2), subdiv=0)
        # the boot shaft hangs from the shin, from just under the knee; its top folds down into a cuff
        mesh_obj(f'Boot shaft.{side}', loft([(-2.2, 2.3, 2.4), (-6.0, 2.15, 2.25), (-SHIN + 0.5, 1.75, 1.85), (-SHIN - 1.2, 1.8, 1.9)]),
                 M_BOOTS, kn)
        mesh_obj(f'Boot cuff.{side}', loft([(-1.6, 2.6, 2.7), (-3.5, 2.5, 2.6, 0, 0, lambda p: 0.5 * math.cos(p))]),
                 M_CUFF, kn, subdiv=1)
        an = empty(f'Ankle.{side}', (0, 0, -SHIN), kn)
        mesh_obj(f'Foot.{side}', loft([(-0.8, 1.75, 1.95, 0, -0.3), (-2.2, 1.85, 3.3, 0, -1.6), (-2.85, 1.8, 3.4, 0, -1.7)]),
                 M_BOOTS, an)
        mesh_obj(f'Sole.{side}', loft([(-2.75, 1.85, 3.45, 0, -1.7), (-ANKLE_Z, 1.8, 3.35, 0, -1.7)]), M_SOLE, an, subdiv=1)

    # ---- camera and light
    cam_data = bpy.data.cameras.new('Camera')
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = FH          # the frame is 80 design pixels tall
    cam_data.sensor_fit = 'VERTICAL'
    cam = link(bpy.data.objects.new('Camera', cam_data))
    cam.location = (0, -200, GY - FH / 2)   # frame top at h 77 above the ground, ground at row 154
    cam.rotation_euler = (math.pi / 2, 0, 0)
    cam_data.clip_end = 500
    sc.camera = cam
    sun_data = bpy.data.lights.new('Sun', 'SUN')
    sun_data.energy = 3.2
    sun_data.angle = 0.0
    sun = link(bpy.data.objects.new('Sun', sun_data))
    # lit from the upper left and in front, as the painted sprites are (fixed to the camera, not the figure)
    sun.rotation_euler = Vector((0.55, 0.75, -0.62)).to_track_quat('-Z', 'Y').to_euler()
    world = bpy.data.worlds.new('World')
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    bg.inputs['Color'].default_value = (0.32, 0.32, 0.34, 1.0)
    # the fill light is added after the render (see to_pixels): sampled from the sky it would speckle at one sample
    bg.inputs['Strength'].default_value = 0.0
    sc.world = world

    # ---- render settings: Cycles at the sprite size with one sample per pixel at its centre, so every pixel shows
    # exactly one surface (no blended edges), and passes for the sun's light and the surface normal
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = 1
    sc.cycles.use_adaptive_sampling = False
    sc.cycles.use_denoising = False
    sc.cycles.max_bounces = 0
    sc.cycles.pixel_filter_type = 'BOX'
    sc.cycles.filter_width = 0.01
    sc.render.film_transparent = True
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = PW, PH, 100
    vl = sc.view_layers[0]
    vl.use_pass_diffuse_direct = True
    vl.use_pass_normal = True
    for name in ('matid', 'detail'):
        a = vl.aovs.add()
        a.name = name
        a.type = 'VALUE'
    if hasattr(sc.render.image_settings, 'media_type'):   # Blender 5
        sc.render.image_settings.media_type = 'MULTI_LAYER_IMAGE'
        sc.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
    else:
        sc.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
    sc.render.image_settings.color_depth = '32'
    sc.view_settings.view_transform = 'Standard'

    # key the 10 sheet columns onto the timeline (frames 1-10), so the .blend plays the idle and walk
    sc.frame_start, sc.frame_end = 1, COLS
    for col in range(COLS):
        pose(col)
        for o in sc.objects:
            if o.type == 'EMPTY' and o.name != 'Ranger':   # (the root is left free: the renderer turns it per facing)
                o.keyframe_insert('rotation_euler', frame=col + 1)
                o.keyframe_insert('location', frame=col + 1)
    for o in sc.objects:
        if o.animation_data and o.animation_data.action:
            for fc in _fcurves(o.animation_data.action):
                for kp in fc.keyframe_points:
                    kp.interpolation = 'CONSTANT'
    return root


def _fcurves(action):
    if hasattr(action, 'fcurves'):
        try:
            return list(action.fcurves)
        except Exception:
            pass
    out = []
    for layer in getattr(action, 'layers', []):
        for strip in layer.strips:
            for bag in strip.channelbags:
                out.extend(bag.fcurves)
    return out


# ---------------------------------------------------------------- posing
def leg_ik(hip_z, fwd, foot_z):
    """Two-bone IK in the leg's own plane: the thigh and shin angles (forward of straight down) that put the ankle
    `fwd` ahead of the hip and at height foot_z. The knee bends forward."""
    dz = hip_z - foot_z
    D = min(math.hypot(fwd, dz), THIGH + SHIN - 0.05)
    alpha = math.atan2(fwd, dz)
    beta = math.acos(max(-1, min(1, (THIGH ** 2 + D ** 2 - SHIN ** 2) / (2 * THIGH * D))))
    gamma = math.acos(max(-1, min(1, (SHIN ** 2 + D ** 2 - THIGH ** 2) / (2 * SHIN * D))))
    return alpha + beta, alpha - gamma


def pose(col):
    ob = bpy.data.objects
    walking = col >= 2
    # the walk is sampled half a step off the passing pose (as in the painted sheets), so no frame has the feet together
    ph = (col - 2 + 0.5) / WALK * 2 * math.pi if walking else 0.0
    swing = math.sin(ph)
    if walking:
        bob = 0.4 - 1.5 * (1 - abs(swing)) + 0.6 * abs(swing)
    else:
        bob = 1.0 if col == 1 else 0.0
    bob = round(bob * RES) / RES          # whole sheet pixels, so the face never shifts by a fraction
    ob['Body'].location = (0, 0, -bob)
    hip_z = HIP_Z - bob
    ob['Body'].rotation_euler = (0, 0, 0)

    for side, s in (('R', 1), ('L', -1)):
        hip, kn, an = ob[f'Hip.{side}'], ob[f'Knee.{side}'], ob[f'Ankle.{side}']
        sh, el, wr = ob[f'Shoulder.{side}'], ob[f'Elbow.{side}'], ob[f'Wrist.{side}']
        if walking:
            p = ph + (0 if s > 0 else math.pi)
            fwd = math.sin(p) * STRIDE
            lift = LIFT * max(0.0, math.cos(p)) ** 1.5     # the foot travelling forward is off the ground
            out = 0.0
            turn = 0.0
            toe = -0.25 * max(0.0, -math.cos(p)) * max(0.0, -math.sin(p))   # heel peels up behind
        else:
            # hip-width stance: weight on the right leg, under the body; the left foot eased forward, out, and turned out
            fwd = -1.0 if s > 0 else 4.6
            lift = 0.0
            out = 0.0 if s > 0 else 0.6
            turn = 0.0 if s > 0 else 0.35
            toe = 0.0
        hip.location = (-s * (LEG_X + out), 0, HIP_Z)
        th, shn = leg_ik(hip_z, fwd, ANKLE_Z + lift)
        hip.rotation_euler = (-th, 0, 0)
        kn.rotation_euler = (-(shn - th), 0, 0)
        an.rotation_euler = (shn + toe, 0, -s * turn)

        # arms: hang just off the body with a soft bend; in the walk they swing against the legs
        if walking:
            a = -math.sin(ph + (0 if s > 0 else math.pi)) * 0.38
            bend = 0.25 + 0.3 * max(0.0, a)
        else:
            a = 0.12 if s > 0 else -0.04
            bend = 0.32 if s > 0 else 0.2
        sh.rotation_euler = (-a, s * 0.1, 0)
        el.rotation_euler = (-bend, 0, 0)
        wr.rotation_euler = (0.1, 0, 0)


# ---------------------------------------------------------------- render and pixel-art conversion
def read_exr(path):
    import OpenEXR
    out = {}
    with OpenEXR.File(path, separate_channels=True) as f:
        for part in f.parts:     # Blender 5 writes each pass as its own part
            out.update({k: np.asarray(v.pixels, dtype=np.float32) for k, v in part.channels.items()})
    return out


def find(ch, *keys):
    for k in ch:
        if all(s in k for s in keys):
            return ch[k]
    raise KeyError(f'no channel with {keys}: {sorted(ch)}')


# light thresholds between the five tones (tuned so a surface square to the sun reads tone 3, its rim tone 4)
TONE_EDGES = np.array([0.36, 0.62, 0.95, 1.25], dtype=np.float32)


def to_pixels(ch):
    alpha = find(ch, 'Combined', 'A')
    # every pass comes premultiplied by coverage: divide it out, or the edge pixels read as another material
    cov = np.maximum(alpha, 1e-4)
    mid = np.rint(find(ch, 'matid') / cov * 100).astype(int)
    det = find(ch, 'detail') / cov
    sun = sum(find(ch, 'Diffuse Direct.' + c) for c in 'RGB') / 3.0 / cov
    # fill: a soft sky light, brighter on surfaces facing up and toward the camera (normals are in world space)
    nz, ny = find(ch, 'Normal.Z') / cov, find(ch, 'Normal.Y') / cov
    light = sun + 0.30 + 0.12 * nz - 0.08 * ny
    h, w = alpha.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    solid = (alpha > 0.5) & (mid > 0)
    for m, d in MATERIALS.items():
        sel = solid & (mid == m)
        if not sel.any():
            continue
        if d['flat'] is not None:
            tone = np.full(sel.sum(), d['flat'])
        else:
            L = light[sel] * (1.0 + (det[sel] - 0.5) * 2 * d['detail'])
            tone = np.digitize(L, TONE_EDGES)
        out[sel, :3] = d['ramp'][tone]
        out[sel, 3] = 255
    # the painter's outline: each empty pixel beside the figure takes a dark, cool version of the pixel it borders
    src = out.copy()
    filled = src[..., 3] > 0
    for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        nb = np.zeros_like(filled)
        col = np.zeros_like(src[..., :3])
        ys0, ys1 = max(0, -dy), h - max(0, dy)
        xs0, xs1 = max(0, -dx), w - max(0, dx)
        nb[ys0:ys1, xs0:xs1] = filled[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        col[ys0:ys1, xs0:xs1] = src[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx, :3]
        take = (~filled) & nb & (out[..., 3] == 0)
        out[take, 0] = col[take, 0] * 0.34 + 12
        out[take, 1] = col[take, 1] * 0.32 + 10
        out[take, 2] = col[take, 2] * 0.34 + 14
        out[take, 3] = 255
    return np.clip(out, 0, 255).astype(np.uint8)


def render_sheet(out_png, tmp):
    sc = bpy.context.scene
    root = bpy.data.objects['Ranger']
    sheet = np.zeros((PH * ROWS, PW * COLS, 4), dtype=np.uint8)
    for col in range(COLS):
        sc.frame_set(col + 1)
        for row in range(ROWS):
            # row 0 faces the viewer; each row turns her 45 degrees toward screen right
            root.rotation_euler = (0, 0, row * math.pi / 4)
            path = os.path.join(tmp, f'f{row}_{col}.exr')
            sc.render.filepath = path
            bpy.ops.render.render(write_still=True)
            px = to_pixels(read_exr(path))
            sheet[row * PH:(row + 1) * PH, col * PW:(col + 1) * PW] = px
            print(f'rendered row {row} col {col}', flush=True)
    root.rotation_euler = (0, 0, 0)
    from PIL import Image
    Image.fromarray(sheet, 'RGBA').save(out_png)
    return sheet


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    out_dir = argv[argv.index('--out') + 1] if '--out' in argv else HERE
    os.makedirs(out_dir, exist_ok=True)
    build()
    bpy.context.scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, 'ranger.blend'), compress=True)
    if '--build-only' in argv:
        return
    with tempfile.TemporaryDirectory() as tmp:
        render_sheet(os.path.join(out_dir, 'ranger-blender.png'), tmp)
    print('wrote', os.path.join(out_dir, 'ranger-blender.png'))


if __name__ == '__main__':
    main()
