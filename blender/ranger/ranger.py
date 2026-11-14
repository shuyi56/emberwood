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
M_EYE = mat_def(16, 'eye', [[46, 40, 38], [58, 66, 46], [88, 108, 70], [120, 140, 100], [226, 222, 210]], 0.0, flat=1)
M_LASH = mat_def(20, 'lashes', [[46, 40, 38]] * 5, 0.0, flat=0)
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
HIP_Z, SH_Z = 24.0, 39.2
THIGH, SHIN = 10.6, 10.0      # hip -> knee -> ankle (ankle at h 4 when straight-ish)
UPPER, FORE = 7.6, 6.4
LEG_X, SH_X = 3.4, 7.2


def build():
    reset()
    sc = bpy.context.scene
    root = empty('Ranger', (0, 0, 0))
    body = empty('Body', (0, 0, 0), root)   # carries the walk's bob

    # ---- torso: the shirt underneath, then the jerkin over it
    mesh_obj('Shirt', loft([(43.2, 2.2, 2.0), (41.6, 4.6, 3.0), (40.0, 6.6, 3.6), (38.0, 7.2, 4.0), (35.5, 7.0, 4.2),
                            (32.5, 6.0, 3.8), (29.5, 4.4, 3.2), (27.0, 4.8, 3.4), (24.0, 6.8, 4.0), (21.0, 7.2, 4.2)]),
             M_SHIRT, body)
    jerk = lambda phi: -0.9 * math.cos(phi) ** 2   # the skirt dips a little at front and back
    mesh_obj('Jerkin', loft([(36.6, 7.3, 4.45), (34.0, 7.1, 4.45), (31.5, 5.9, 4.0), (29.5, 4.75, 3.5), (27.5, 5.0, 3.6),
                             (25.5, 6.5, 4.15), (23.0, 7.4, 4.6), (21.2, 7.75, 4.85, 0, 0, jerk), (20.2, 7.9, 4.95, 0, 0, jerk)],
                            cap_bot=False), M_JERKIN, body, solid=0.35)
    # a darker leather band binds the jerkin's hem
    mesh_obj('Jerkin hem', loft([(21.3, 7.82, 4.92, 0, 0, jerk), (20.1, 7.98, 5.02, 0, 0, jerk)], cap_top=False, cap_bot=False),
             M_BELT, body, solid=0.3, subdiv=0)
    # the shirt shows in a narrow V at the front of the jerkin, crossed by its lacing
    vee = bmesh.new()
    vs = [vee.verts.new(p) for p in [(-1.5, -4.62, 36.5), (1.5, -4.62, 36.5), (0, -4.05, 31.2)]]
    vee.faces.new(vs)
    mesh_obj('Shirt V', vee, M_SHIRT, body, subdiv=0)
    for i, z in enumerate((35.4, 34.0, 32.6)):
        w = 1.3 * (z - 31.2) / 5.3
        mesh_obj(f'Lace {i}', box(2 * w + 0.4, 0.3, 0.45), M_LACE, body, loc=(0, -4.7 + 0.1 * i, z), subdiv=0)
    # belt and buckle
    mesh_obj('Belt', loft([(29.3, 5.15, 3.85), (27.7, 5.25, 3.9)]), M_BELT, body, subdiv=0)
    mesh_obj('Buckle', box(1.9, 0.6, 1.9), M_STEEL, body, loc=(0, -3.95, 28.5), subdiv=0)
    mesh_obj('Buckle hole', box(0.9, 0.3, 0.9), M_BELT, body, loc=(0, -4.2, 28.5), subdiv=0)

    # ---- neck and head
    mesh_obj('Neck', loft([(48.0, 1.9, 1.9, 0, 0.3), (42.0, 2.2, 2.1, 0, 0.3)]), M_SKIN, body)
    head = empty('Head', (0, 0, 47.0), body)
    mesh_obj('Head shape', loft([(14.3, 0.6, 0.6, 0, 0.3), (13.6, 2.7, 2.8, 0, 0.3), (12.2, 4.3, 4.3, 0, 0.1), (10.0, 5.0, 4.9),
                                 (7.5, 5.1, 5.0), (5.0, 4.6, 4.6, 0, -0.2), (3.2, 3.6, 3.7, 0, -0.5), (1.8, 2.4, 2.6, 0, -1.0),
                                 (0.8, 1.2, 1.2, 0, -1.6)], n=24), M_SKIN, head, subdiv=2)
    mesh_obj('Nose', ellipsoid(0.55, 0.6, 0.9, 10, 6), M_SKIN, head, loc=(0, -4.95, 5.6))
    for s in (-1, 1):
        mesh_obj(f'Eye {s}', ellipsoid(0.72, 0.4, 1.0, 10, 6), M_EYE, head, loc=(s * 2.0, -4.55, 6.7), subdiv=0)
        mesh_obj(f'Lash {s}', ellipsoid(0.9, 0.4, 0.35, 10, 6), M_LASH, head, loc=(s * 2.0, -4.6, 7.55), subdiv=0)
        mesh_obj(f'Brow {s}', ellipsoid(1.1, 0.3, 0.3, 10, 6), M_BROW, head, loc=(s * 2.1, -4.65, 8.5), subdiv=0)
    mesh_obj('Mouth', ellipsoid(0.75, 0.25, 0.2, 10, 6), M_LIPS, head, loc=(0, -4.3, 3.5), subdiv=0)

    # ---- hair: a cap over the skull with curtain bangs parted at the centre, side locks, and the long fall at the back
    # (the cap is a shell over the whole skull with the face cut away along a sloping hairline)
    cap = ellipsoid(5.65, 5.6, 6.2, 28, 16)
    bmesh.ops.translate(cap, vec=(0, 0.25, 8.5), verts=cap.verts)
    bmesh.ops.bisect_plane(cap, geom=cap.verts[:] + cap.edges[:] + cap.faces[:], plane_co=(0, -2.6, 9.9),
                           plane_no=Vector((0, -0.72, -0.69)).normalized(), clear_outer=True)
    mesh_obj('Hair cap', cap, M_HAIR, head, solid=0.5)
    for s in (-1, 1):
        b = ellipsoid(2.7, 1.0, 2.3)
        bmesh.ops.rotate(b, cent=(0, 0, 0), matrix=Euler((0, s * 0.45, 0)).to_matrix(), verts=b.verts)
        mesh_obj(f'Bangs {s}', b, M_HAIR, head, loc=(s * 2.4, -3.75, 10.0))
        mesh_obj(f'Side lock {s}', loft([(10.0, 1.7, 2.0, s * 4.7, -0.6), (7.0, 1.6, 1.9, s * 5.2, -0.2), (2.0, 1.5, 1.8, s * 5.4, 0.2),
                                         (-2.5, 1.5, 1.6, s * 5.6, 0.4), (-4.6, 0.7, 0.8, s * 5.6, 0.4)]),
                 M_HAIR, head)
    mesh_obj('Hair fall', loft([(10.5, 4.8, 3.6, 0, 1.4), (8.0, 5.85, 4.6, 0, 1.4), (3.0, 6.1, 3.2, 0, 3.2), (-3.0, 6.3, 2.0, 0, 5.4),
                                (-8.0, 6.0, 1.7, 0, 5.6), (-13.0, 5.6, 1.55, 0, 5.0), (-16.6, 5.2, 1.45, 0, 4.7),
                                (-17.4, 4.0, 1.0, 0, 4.7)]), M_HAIR, head)

    # ---- mantle: a short shoulder cape, its hem lifting over the arms; a mail trim along the hem and a leaf clasp
    hem = lambda phi: 1.5 * math.sin(phi) ** 2 - 0.5 * math.cos(phi)
    mesh_obj('Mantle', loft([(44.2, 2.9, 2.7, 0, 0.3), (43.2, 4.6, 3.9, 0, 0.3), (41.6, 6.6, 4.9, 0, 0.3), (40.0, 8.5, 5.5, 0, 0.3),
                             (38.0, 9.7, 5.8, 0, 0.3, hem), (35.3, 9.95, 5.95, 0, 0.3, hem)], cap_top=False, cap_bot=False),
             M_MANTLE, body, solid=0.45)
    mesh_obj('Mail trim', loft([(35.8, 10.05, 6.05, 0, 0.3, hem), (34.9, 10.1, 6.1, 0, 0.3, hem)], cap_top=False, cap_bot=False),
             M_MAIL, body, solid=0.4, subdiv=0)
    mesh_obj('Collar', loft([(44.9, 2.6, 2.5, 0, 0.3), (43.6, 3.1, 3.0, 0, 0.3)], cap_top=False, cap_bot=False), M_SHIRT, body, solid=0.3)
    clasp = ellipsoid(1.0, 0.45, 1.5, 12, 8)
    bmesh.ops.rotate(clasp, cent=(0, 0, 0), matrix=Euler((0, 0.6, 0)).to_matrix(), verts=clasp.verts)
    mesh_obj('Leaf clasp', clasp, M_GOLD, body, loc=(-0.8, -5.55, 40.6), subdiv=0)

    # ---- arms: shoulder -> elbow -> wrist pivots; sleeves with a rounded cap, shirt cuffs and hands
    for side, s in (('R', 1), ('L', -1)):
        x = -s * SH_X   # the figure's right is -X (she faces -Y)
        sh = empty(f'Shoulder.{side}', (x, 0.2, SH_Z), body)
        mesh_obj(f'Sleeve cap.{side}', ellipsoid(2.5, 2.6, 2.4), M_SLEEVE, sh, loc=(0, 0, -0.6))
        mesh_obj(f'Upper sleeve.{side}', loft([(0.0, 2.35, 2.4), (-3.5, 2.15, 2.2), (-UPPER, 1.95, 2.0)]), M_SLEEVE, sh)
        el = empty(f'Elbow.{side}', (0, 0, -UPPER), sh)
        mesh_obj(f'Fore sleeve.{side}', loft([(0.8, 1.95, 2.0), (-3.0, 1.95, 2.0), (-FORE + 0.6, 1.8, 1.85)]), M_SLEEVE, el)
        mesh_obj(f'Shirt cuff.{side}', loft([(-FORE + 1.0, 1.95, 2.0), (-FORE - 0.4, 2.0, 2.05)]), M_SHIRT, el, subdiv=0)
        wr = empty(f'Wrist.{side}', (0, 0, -FORE), el)
        mesh_obj(f'Hand.{side}', ellipsoid(1.05, 1.3, 1.65, 12, 8), M_SKIN, wr, loc=(-s * 0.15, -0.2, -1.5))

    # ---- legs: hip -> knee -> ankle pivots; trousers with knee patches, soft boots with a folded cuff
    for side, s in (('R', 1), ('L', -1)):
        x = -s * LEG_X
        hip = empty(f'Hip.{side}', (x, 0, HIP_Z), body)
        mesh_obj(f'Thigh.{side}', loft([(1.5, 3.1, 3.2), (-1.0, 3.0, 3.1), (-5.0, 2.6, 2.7), (-THIGH, 2.2, 2.3)]), M_LEGS, hip)
        kn = empty(f'Knee.{side}', (0, 0, -THIGH), hip)
        mesh_obj(f'Shin.{side}', loft([(0.6, 2.2, 2.3), (-4.0, 2.0, 2.05), (-SHIN, 1.7, 1.8)]), M_LEGS, kn)
        mesh_obj(f'Knee patch.{side}', ellipsoid(1.2, 0.5, 1.3, 12, 8), M_KNEE, kn, loc=(0, -2.15, -0.3), subdiv=0)
        # the boot shaft hangs from the shin; its top folds down into a cuff
        mesh_obj(f'Boot shaft.{side}', loft([(-1.6, 2.35, 2.45), (-5.0, 2.3, 2.4), (-SHIN + 0.5, 2.15, 2.25), (-SHIN - 1.2, 2.2, 2.3)]),
                 M_BOOTS, kn)
        mesh_obj(f'Boot cuff.{side}', loft([(-1.0, 2.75, 2.85), (-2.8, 2.65, 2.75, 0, 0, lambda p: 0.5 * math.cos(p))]),
                 M_CUFF, kn, subdiv=1)
        an = empty(f'Ankle.{side}', (0, 0, -SHIN), kn)
        mesh_obj(f'Foot.{side}', loft([(-1.0, 2.15, 2.3, 0, -0.4), (-2.6, 2.2, 3.4, 0, -1.4), (-3.4, 2.15, 3.5, 0, -1.5)]),
                 M_BOOTS, an)
        mesh_obj(f'Sole.{side}', loft([(-3.3, 2.2, 3.55, 0, -1.5), (-4.0, 2.15, 3.45, 0, -1.5)]), M_SOLE, an, subdiv=1)

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
            fwd = math.sin(p) * 6.0
            lift = 2.6 * max(0.0, math.cos(p)) ** 1.5     # the foot travelling forward is off the ground
            out = 0.0
            turn = 0.0
            toe = -0.25 * max(0.0, -math.cos(p)) * max(0.0, -math.sin(p))   # heel peels up behind
        else:
            # hip-width stance: weight on the right leg, under the body; the left foot eased forward, out, and turned out
            fwd = -0.8 if s > 0 else 3.6
            lift = 0.0
            out = 0.0 if s > 0 else 0.5
            turn = 0.0 if s > 0 else 0.35
            toe = 0.0
        hip.location = (-s * (LEG_X + out), 0, HIP_Z)
        th, shn = leg_ik(hip_z, fwd, 4.0 + lift)
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
