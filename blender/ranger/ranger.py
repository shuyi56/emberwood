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
M_HAIR = mat_def(2, 'hair', ramp(0x7e4c34, 0.1), 0.24)
M_MANTLE = mat_def(3, 'mantle', ramp(0x56663f, 0.1), 0.14)
M_JERKIN = mat_def(4, 'jerkin', ramp(0x765a40, 0.1), 0.24)
M_SHIRT = mat_def(5, 'shirt', ramp(0xaca48d, 0.09), 0.04)
M_SLEEVE = mat_def(6, 'sleeve', ramp(0x63704a, 0.1), 0.07)
M_LEGS = mat_def(7, 'trousers', ramp(0x5b5547, 0.1), 0.05)
M_KNEE = mat_def(8, 'knee patch', ramp(0x7a6c52, 0.1), 0.0)
M_BOOTS = mat_def(9, 'boots', ramp(0x684a32, 0.1), 0.24)
M_CUFF = mat_def(10, 'boot cuff', ramp(0x7a5b40, 0.1), 0.2)
M_SOLE = mat_def(11, 'sole', ramp(0x2f2620, 0.08), 0.0)
M_BELT = mat_def(12, 'belt', ramp(0x3d342b, 0.09), 0.18)
M_STEEL = mat_def(13, 'buckle', ramp(0x9a978f, 0.09), 0.0)
M_GOLD = mat_def(14, 'clasp', ramp(0xa8905c, 0.09), 0.0)
M_MAIL = mat_def(15, 'mail trim', ramp(0x7e827c, 0.08), 0.35)
# the eyes: a moss-green iris dark enough to hold against the white and the skin at sprite size; the lids are
# skin. (No lash line and no painted catchlight: the eye's highlight is the sun's reflection in its glossy surface.)
# The iris ramp runs from the dark ring at its rim, through the green, to the warm amber round the pupil; it is
# painted onto the iris as vertex colours (see iris_colours), and the sprite picks its tone from that painting
M_EYE = mat_def(16, 'iris', [[30, 42, 30], [50, 76, 44], [72, 104, 58], [104, 134, 74], [148, 146, 86]], 0.0, flat='painted')
M_SCLERA = mat_def(21, 'eye white', [[206, 198, 186], [226, 220, 208], [224, 217, 205], [234, 229, 218], [242, 238, 230]], 0.0, flat=1)
M_BROW = mat_def(17, 'brow', ramp(0x60463a, 0.08), 0.0, flat=1)
M_LIPS = mat_def(18, 'lips', [[150, 86, 74], [168, 100, 86], [184, 116, 100], [196, 132, 114], [210, 150, 132]], 0.0, flat=1)
M_LACE = mat_def(19, 'laces', ramp(0x4a3a2c, 0.08), 0.0)
M_STITCH = mat_def(22, 'stitching', ramp(0xb39a74, 0.08), 0.0, flat=2)
# leather is matte, with a pebbled grain pressed into its surface
LEATHER = {M_JERKIN, M_BELT, M_BOOTS, M_CUFF}


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
    if mid in (M_EYE, M_SCLERA):
        # the eye is wet: a glossy coat over the colour, so the sun leaves a real highlight that moves with the eye
        # and the light (the sprites read only the diffuse light, so they are unaffected)
        nt.nodes.remove(bsdf)
        bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.inputs['Base Color'].default_value = (*lin, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.1 if mid == M_EYE else 0.35
        bsdf.inputs['IOR'].default_value = 1.38
        # the film of tears: a clear, mirror-smooth coat over the whole ball, as on a glass marble
        bsdf.inputs['Coat Weight'].default_value = 1.0
        # (the white's coat a little less mirror-like than the cornea's: its surface is far flatter than a real
        # ball's, so a mirror coat threw the sun back as a large hard-edged patch on the lower white)
        bsdf.inputs['Coat Roughness'].default_value = 0.03 if mid == M_EYE else 0.16
        if mid == M_SCLERA:
            bsdf.inputs['Coat Weight'].default_value = 0.6
        bsdf.inputs['Coat IOR'].default_value = 1.38
        nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
        if mid == M_EYE:
            tex = nt.nodes.new('ShaderNodeTexImage')      # its painting (iris_image)
            tex.image = iris_image()
            tex.interpolation = 'Cubic'
            nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
        else:
            tex = nt.nodes.new('ShaderNodeTexImage')      # its painting (sclera_image)
            tex.image = sclera_image()
            tex.interpolation = 'Cubic'
            nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    if mid in LEATHER:
        # the grain: fine pebbling and scattered pores pressed into the surface, so it catches the light in the
        # render itself (and reads as grain at sprite size, not only as a painted pattern)
        rest = nt.nodes.new('ShaderNodeAttribute')
        rest.attribute_name = 'rest'
        peb = nt.nodes.new('ShaderNodeTexVoronoi')
        peb.feature = 'SMOOTH_F1'
        peb.inputs['Scale'].default_value = 2.4
        nt.links.new(rest.outputs['Vector'], peb.inputs['Vector'])
        fine = nt.nodes.new('ShaderNodeTexNoise')
        fine.inputs['Scale'].default_value = 3.0
        fine.inputs['Detail'].default_value = 4.0
        nt.links.new(rest.outputs['Vector'], fine.inputs['Vector'])
        mix = nt.nodes.new('ShaderNodeMath')
        mix.operation = 'MULTIPLY_ADD'
        mix.inputs[1].default_value = 0.6
        nt.links.new(peb.outputs['Distance'], mix.inputs[0])
        nt.links.new(fine.outputs['Fac'], mix.inputs[2])
        bump = nt.nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = 0.55
        bump.inputs['Distance'].default_value = 0.15
        nt.links.new(mix.outputs[0], bump.inputs['Height'])
        nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    if mid == M_HAIR:
        # strands in the surface itself: broad clumps and finer strands, both drawn out along the fall of the hair
        rest = nt.nodes.new('ShaderNodeAttribute')
        rest.attribute_name = 'rest'
        h = None
        for sc_, k_ in (((0.9, 0.9, 0.07), 0.6), ((4.0, 4.0, 0.15), 0.18)):
            st = nt.nodes.new('ShaderNodeVectorMath')
            st.operation = 'MULTIPLY'
            st.inputs[1].default_value = sc_
            nt.links.new(rest.outputs['Vector'], st.inputs[0])
            nz_ = nt.nodes.new('ShaderNodeTexNoise')
            nz_.inputs['Scale'].default_value = 1.0
            nz_.inputs['Detail'].default_value = 2.0
            nt.links.new(st.outputs['Vector'], nz_.inputs['Vector'])
            m_ = nt.nodes.new('ShaderNodeMath')
            m_.operation = 'MULTIPLY_ADD'
            m_.inputs[1].default_value = k_
            nt.links.new(nz_.outputs['Fac'], m_.inputs[0])
            if h is None:
                m_.inputs[2].default_value = 0.0
            else:
                nt.links.new(h, m_.inputs[2])
            h = m_.outputs[0]
        bump = nt.nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = 0.6
        bump.inputs['Distance'].default_value = 0.25
        nt.links.new(h, bump.inputs['Height'])
        nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
        # the ends: the last hand's breadth of the hair is finer and catches the light, so it goes a little lighter
        rsep = nt.nodes.new('ShaderNodeSeparateXYZ')
        nt.links.new(rest.outputs['Vector'], rsep.inputs['Vector'])
        tip = nt.nodes.new('ShaderNodeMapRange')
        tip.name = 'hair tips'
        tip.inputs['From Min'].default_value = -7.6
        tip.inputs['From Max'].default_value = -10.2
        nt.links.new(rsep.outputs['Z'], tip.inputs['Value'])
        light = d['ramp'][3] / 255.0
        tint = nt.nodes.new('ShaderNodeMix')
        tint.data_type = 'RGBA'
        tint.inputs['A'].default_value = (*lin, 1.0)
        tint.inputs['B'].default_value = (*[((c + 0.055) / 1.055) ** 2.4 for c in light], 1.0)
        tip_k = nt.nodes.new('ShaderNodeMath')
        tip_k.operation = 'MULTIPLY'
        tip_k.inputs[1].default_value = 0.6
        nt.links.new(tip.outputs['Result'], tip_k.inputs[0])
        nt.links.new(tip_k.outputs[0], tint.inputs['Factor'])
        nt.links.new(tint.outputs['Result'], bsdf.inputs['Color'])
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
    # patterns are laid out in each mesh's rest position (stored as the 'rest' attribute), so they stay fixed to
    # the cloth while the limbs bend
    coord = nt.nodes.new('ShaderNodeAttribute')
    coord.attribute_name = 'rest'
    vec = coord.outputs['Vector']
    if mid == M_HAIR:
        # a few broad clumps running down the hair (noise stretched along it, so each clump is a long streak with
        # one lit side and one darker crease), plus the sheen: a lit ring where the hair turns from facing up to
        # facing out, the band a painter puts round the crown
        st = nt.nodes.new('ShaderNodeVectorMath')
        st.operation = 'MULTIPLY'
        st.inputs[1].default_value = (0.32, 0.32, 0.035)
        nt.links.new(vec, st.inputs[0])
        clump = nt.nodes.new('ShaderNodeTexNoise')
        clump.inputs['Scale'].default_value = 1.0
        clump.inputs['Detail'].default_value = 0.0
        nt.links.new(st.outputs['Vector'], clump.inputs['Vector'])
        # fine strands: soft bands across the hair, wavering a little
        wave = nt.nodes.new('ShaderNodeTexWave')
        wave.wave_type = 'BANDS'
        wave.bands_direction = 'X'
        wave.inputs['Scale'].default_value = 0.22
        wave.inputs['Distortion'].default_value = 0.6
        wave.inputs['Detail'].default_value = 0.0
        nt.links.new(vec, wave.inputs['Vector'])
        tex = nt.nodes.new('ShaderNodeMix')
        tex.data_type = 'FLOAT'
        tex.inputs['Factor'].default_value = 0.35
        nt.links.new(wave.outputs['Fac'], tex.inputs['A'])
        nt.links.new(clump.outputs['Fac'], tex.inputs['B'])
        geo = nt.nodes.new('ShaderNodeNewGeometry')
        sep = nt.nodes.new('ShaderNodeSeparateXYZ')
        nt.links.new(geo.outputs['Normal'], sep.inputs['Vector'])
        dist = nt.nodes.new('ShaderNodeMath')
        dist.operation = 'SUBTRACT'
        dist.inputs[1].default_value = 0.5
        nt.links.new(sep.outputs['Z'], dist.inputs[0])
        ab = nt.nodes.new('ShaderNodeMath')
        ab.operation = 'ABSOLUTE'
        nt.links.new(dist.outputs[0], ab.inputs[0])
        band = nt.nodes.new('ShaderNodeMapRange')       # 1 at nz = 0.5, fading to 0 by 0.2 either side
        band.inputs['From Min'].default_value = 0.28
        band.inputs['From Max'].default_value = 0.0
        nt.links.new(ab.outputs[0], band.inputs['Value'])
        add = nt.nodes.new('ShaderNodeMath')
        add.operation = 'MULTIPLY_ADD'
        add.inputs[1].default_value = 0.55
        nt.links.new(band.outputs[0], add.inputs[0])
        nt.links.new(tex.outputs['Result'], add.inputs[2])
        # lighter ends in the sprite too
        tadd = nt.nodes.new('ShaderNodeMath')
        tadd.operation = 'MULTIPLY_ADD'
        tadd.inputs[1].default_value = 0.3
        nt.links.new(nt.nodes['hair tips'].outputs['Result'], tadd.inputs[0])
        nt.links.new(add.outputs[0], tadd.inputs[2])
        nt.links.new(tadd.outputs[0], adet.inputs['Value'])
        return m
    elif mid == M_EYE:
        # the iris's painted tone, for the sprite renderer
        sh = nt.nodes.new('ShaderNodeAttribute')
        sh.attribute_name = 'iris_shade'
        nt.links.new(sh.outputs['Fac'], adet.inputs['Value'])
        return m
    elif mid == M_MAIL:
        tex = nt.nodes.new('ShaderNodeTexChecker')
        tex.inputs['Scale'].default_value = 1.0
    elif mid in (M_SLEEVE, M_LEGS):
        # a twill: fine diagonal ribs, with a little drift so they read as woven rather than ruled
        tex = nt.nodes.new('ShaderNodeTexWave')
        tex.wave_type = 'BANDS'
        tex.bands_direction = 'DIAGONAL'
        tex.inputs['Scale'].default_value = 0.22
        tex.inputs['Distortion'].default_value = 0.8
        tex.inputs['Detail'].default_value = 0.0
    elif mid in LEATHER:
        # hide: broad soft patches where the tone drifts, a finer pebble grain, and scattered darker pores
        tex = nt.nodes.new('ShaderNodeTexNoise')
        tex.inputs['Scale'].default_value = 0.3
        tex.inputs['Detail'].default_value = 2.0
        tex.inputs['Roughness'].default_value = 0.55
        nt.links.new(vec, tex.inputs['Vector'])
        vor = nt.nodes.new('ShaderNodeTexVoronoi')
        vor.inputs['Scale'].default_value = 0.9
        nt.links.new(vec, vor.inputs['Vector'])
        pore = nt.nodes.new('ShaderNodeMath')
        pore.operation = 'LESS_THAN'
        pore.inputs[1].default_value = 0.13
        nt.links.new(vor.outputs['Distance'], pore.inputs[0])
        sub = nt.nodes.new('ShaderNodeMath')
        sub.operation = 'MULTIPLY_ADD'
        sub.inputs[1].default_value = -0.45
        nt.links.new(pore.outputs[0], sub.inputs[0])
        nt.links.new(tex.outputs['Fac'], sub.inputs[2])
        nt.links.new(sub.outputs[0], adet.inputs['Value'])
        return m
    elif mid == M_MANTLE:
        # felted wool: soft mottling at two scales
        tex = nt.nodes.new('ShaderNodeTexNoise')
        tex.inputs['Scale'].default_value = 0.8
        tex.inputs['Detail'].default_value = 3.0
        tex.inputs['Roughness'].default_value = 0.6
    else:
        tex = nt.nodes.new('ShaderNodeTexNoise')
        tex.inputs['Scale'].default_value = 0.35
        tex.inputs['Detail'].default_value = 0.0
    nt.links.new(vec, tex.inputs['Vector'])
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


RIG = {}   # the armature, once built (see build_rig)


def mesh_obj(name, bm, mid, parent=None, loc=(0, 0, 0), smooth=True, subdiv=1, solid=0.0, skin=None, rest=None):
    """A mesh object in one material. skin, if given, maps a rest position to {bone: weight}: the mesh is then
    deformed by the rig instead of being parented, which lets a limb bend smoothly at its joints."""
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = smooth
    flow = rest
    rest = me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
    # pattern coordinates: the rest position, or a mapping given for the mesh (rest(co) -> vector), e.g. one that
    # runs the hair's strands from the parting down over the skull
    rest.data.foreach_set('vector', [c for v in me.vertices for c in (flow(v.co) if flow else v.co + Vector(loc))])
    o = bpy.data.objects.new(name, me)
    o.location = loc
    me.materials.append(mat(mid))
    link(o, None if skin else parent)
    if skin:
        groups = {}
        for v in me.vertices:
            for bone, w in skin(Vector(loc) + v.co).items():
                if w <= 1e-4:
                    continue
                if bone not in groups:
                    groups[bone] = o.vertex_groups.new(name=bone)
                groups[bone].add([v.index], w, 'REPLACE')
        m = o.modifiers.new('rig', 'ARMATURE')
        m.object = RIG['arm']
        m.use_deform_preserve_volume = True
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
HIP_Z, SH_Z, ANKLE_Z = 31.0, 49.0, 3.4
THIGH, SHIN = 14.0, 13.8            # hip -> knee -> ankle, a touch longer than straight so the knee stays soft
UPPER, FORE = 11.0, 8.6
LEG_X, SH_X = 3.3, 6.1
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
        d += gauss(x, z, s * 0.14, 0.525, 0.08, 0.05, -0.02)     # eye socket
        d += gauss(x, z, s * 0.215, 0.45, 0.08, 0.06, 0.035)     # cheekbone
        d += gauss(x, z, s * 0.2, 0.31, 0.07, 0.06, -0.025)      # hollow under the cheekbone
        d += gauss(x, z, s * 0.05, 0.37, 0.03, 0.025, 0.035)     # wing of the nose
    # the nose: a narrow ridge rising from the bridge to the tip, then turning under sharply
    if z > 0.36:
        t = min(1.0, max(0.0, (0.6 - z) / 0.24))
        h = (0.015 + 0.085 * t ** 1.6) * math.exp(-max(0.0, z - 0.58) ** 2 / 0.0008)
    else:
        h = 0.1 * math.exp(-((0.36 - z) / 0.025) ** 2)
    # (the ridge widens smoothly toward the tip: a width that eased in by a straight ramp drew a line across it)
    d += h * math.exp(-(x / (0.04 + 0.03 * math.exp(-((z - 0.37) / 0.065) ** 2))) ** 2)
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


def head_front_y(x, z):
    """The front of the sculpted head (see sculpted_head) at (x across, z up), in head heights, worked out
    exactly from its formula rather than from the mesh, or None off the head."""
    sz = (z - 0.52) / 0.5
    sx = x / (0.36 * _jaw_width(z))
    r = 1 - sx * sx - sz * sz
    if r <= 0:
        return None
    sy = -math.sqrt(r)
    t = max(0.0, (0.5 - z) / 0.5)
    y = sy * 0.44 * (0.92 - 0.15 * t)
    y -= face_relief(x, z) * max(0.0, -sy) ** 1.4
    return y + 0.04


def surface_y(bm, x, z):
    """The front of the head's surface at (x, z): where a ray from the camera side first meets it."""
    from mathutils.bvhtree import BVHTree
    tri = bm.copy()       # (built from triangles: FromBMesh can crash on a large n-gon cap)
    bmesh.ops.triangulate(tri, faces=tri.faces[:])
    tri.verts.index_update()
    tree = BVHTree.FromPolygons([v.co.copy() for v in tri.verts], [[v.index for v in f.verts] for f in tri.faces])
    tri.free()
    hit = tree.ray_cast(Vector((x, -50, z)), Vector((0, 1, 0)))
    return hit[0].y if hit[0] is not None else 0.0


def hair_flow(co):
    """Pattern coordinates for the head of hair: across the strands is the angle round the head (seen from
    above), along them is height, so the strands fall straight down the sides and back. (Coordinates that followed
    the curve from the parting drew rings round the back of the head, like the ridges of a shell.)"""
    # (measured as a distance round the head rather than an angle, so near the neck the strands do not crowd
    # into a fan; the angle is taken from the back, so its one break falls at the parting, not down her back)
    return (math.atan2(co.x, co.y - 0.5) * max(2.5, math.hypot(co.x, co.y - 0.5)), 0.0, co.z * 0.6)


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
        hit = body.ray_cast(Vector((x, 30.0, z)), Vector((0, -1, 0)), 40.0)
        return hit[0].y if hit[0] is not None else 0.0

    W = [(4.5, 3.6), (1.5, 3.7), (-1.5, 4.1), (-4.5, 4.3), (-9.0, 4.1), (-13.0, 3.6), (-16.0, 3.0), (-17.5, 2.2)]

    def width(z):
        for (z0, w0), (z1, w1) in zip(W, W[1:]):
            if z1 <= z <= z0:
                return lerp(w0, w1, (z0 - z) / (z0 - z1))
        return W[0][1] if z > W[0][0] else W[-1][1]

    NX, NZ = 41, 70
    z_top = 4.5      # it starts inside the hair over the back of the head and grows out of it, so there is no seam
    def end_z(u):     # u from -1 (her right) to 1: an uneven end, the middle longest
        # a few rounded locks, each ending in a soft point (a smooth curve, so the remesh leaves no tatters)
        lock = 0.5 + 0.5 * math.cos(2 * math.pi * (1.6 * u + 0.1))
        return -16.2 - 1.0 * (1 - u * u) - 0.45 * lock
    inner, outer = [], []
    for i in range(NX):
        u = -1 + 2 * i / (NX - 1)
        zs = [lerp(z_top, end_z(u), j / (NZ - 1)) for j in range(NZ)]
        col_in, col_out, y_prev, z_prev = [], [], None, None
        for j, z in enumerate(zs):
            x = u * width(z)
            v = j / (NZ - 1)
            # thick down the middle, thin to soft edges; thinner toward the end, and rounded off at the tip
            # (never thinner than a few voxels: a sheet thinner than the remesh breaks up into rods and tatters)
            grow = min(1.0, (z_top - z) / 3.0)
            grow = grow * grow * (3 - 2 * grow)
            t = (0.95 - 0.3 * v) * max(0.0, 1 - u * u) ** 0.5 * (1.0 if v < 0.92 else math.cos((v - 0.92) / 0.08 * math.pi / 2) ** 0.5)
            t = max(t, 0.4)
            sup = back_y(x, z)
            # the outside of the hair: it starts flush with the hair over the head and hangs from the back of the
            # skull, easing in only gradually under its weight; below the nape it always lies over what is under it
            y = sup + t * grow
            if y_prev is not None:
                y = min(max(y, y_prev - 0.45 * (z_prev - z)), y_prev + 0.2 * (z_prev - z))
            if z < -0.5:
                y = max(y, sup + 0.52)
            if v > 0.8:          # the ends turn in a little toward her back rather than flicking out
                y = lerp(y, sup + 0.52, ((v - 0.8) / 0.2) ** 2 * 0.7)
            y_prev, z_prev = y, z
            col_out.append([x, y, z, t, sup])
        # and where something lower down holds it out (the mantle), it is already leaning out above it: no ledges
        dz = (z_top - end_z(u)) / (NZ - 1)
        for j in range(NZ - 2, -1, -1):
            col_out[j][1] = max(col_out[j][1], col_out[j + 1][1] - 0.35 * dz)
        for x, y, z, t, sup in col_out:
            # where it lies on the mantle the hair is pressed thinner rather than pushed out
            col_in.append(Vector((x, max(y - t, sup + 0.12) if z < -0.5 else y - t, z)))
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


def fuse_hair(head, names, voxel=0.1):
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
        fade = min(1.0, max(0.0, (3.0 - v.co.z) / 4.0))
        if fade <= 0.0:
            continue
        arc = math.atan2(v.co.x, v.co.y - 0.5) * max(2.5, math.hypot(v.co.x, v.co.y - 0.5))
        ridge = 0.16 * math.sin(1.5 * (arc + 0.5 * math.sin(v.co.z * 0.25)) + 0.9) + 0.04 * math.sin(3.7 * arc + 1.3)
        n_ = Vector((v.normal.x, v.normal.y, 0.0))      # across the hair only: the ends must not be scalloped
        v.co += n_ * ridge * fade
    return mesh_obj('Hair', bm, M_HAIR, head, subdiv=0, rest=hair_flow)


def bvh_of(bm):
    from mathutils.bvhtree import BVHTree
    tri = bm.copy()
    bmesh.ops.triangulate(tri, faces=tri.faces[:])
    tri.verts.index_update()
    tree = BVHTree.FromPolygons([v.co.copy() for v in tri.verts], [[v.index for v in f.verts] for f in tri.faces])
    tri.free()
    return tree


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
            # (the tips settle onto the skin; the root starts under the cap)
            cs.append(o + d * (r + 0.04 + rr * (1 - 0.6 * t) + layer * tuck * (1 - t * t) - 0.45 * (1 - tuck)))
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


# ---------------------------------------------------------------- the torso's shape
# Half-width, front depth and back depth of the body at each height (eased smoothly between), for an adult female
# figure: shoulders, a ribcage that narrows to the waist, the waist itself, and hips that swell out over the seat.
TORSO = [(53.0, 1.6, 1.5, 1.7), (51.4, 2.8, 2.2, 2.5), (50.4, 5.0, 2.6, 2.9), (49.2, 6.0, 2.8, 3.1), (47.2, 5.6, 2.9, 3.1),
         (44.5, 5.2, 3.0, 3.0), (42.0, 4.35, 2.85, 2.75), (39.4, 3.65, 2.6, 2.5), (37.0, 4.5, 2.75, 2.85), (34.5, 5.9, 3.1, 3.6),
         (32.0, 6.6, 3.3, 4.0), (30.0, 6.3, 3.3, 3.8), (28.5, 5.6, 3.1, 3.4)]


def _torso_dims(z):
    T = TORSO
    if z >= T[0][0]:
        return T[0][1:]
    if z <= T[-1][0]:
        return T[-1][1:]
    for a, b in zip(T, T[1:]):
        if b[0] <= z <= a[0]:
            t = (a[0] - z) / (a[0] - b[0])
            t = (1 - math.cos(math.pi * t)) / 2
            return tuple(lerp(p, q, t) for p, q in zip(a[1:], b[1:]))


def lerp(a, b, t):
    return a + (b - a) * t


def torso_point(z, phi, off=0.0, bust=1.0):
    """A point on the body at height z, angle phi round from the front toward the figure's right, pushed out by
    off. The bust is two rounded forms on the front of the ribcage, fuller below their centre than above; the
    seat rounds out the back over the hips."""
    rx, fy, by = _torso_dims(z)
    sx, c = math.sin(phi), math.cos(phi)
    x = -(rx + off) * sx
    if c >= 0:
        y = -(fy + off) * c
        for s in (-1, 1):
            zc = 44.3
            sz = 2.3 if z > zc else 1.6                  # a long soft slope above, a rounder curve under
            y -= bust * 1.25 * math.exp(-((x - s * 2.2) / 2.2) ** 2 - ((z - zc) / sz) ** 2) * c
    else:
        y = -(by + off) * c
        for s in (-1, 1):
            y -= 0.9 * math.exp(-((x - s * 2.6) / 2.3) ** 2 - ((z - 32.3) / 2.3) ** 2) * c
    return x, y


def loft_fn(zs, point, n=40, cap_top=True, cap_bot=True):
    """A tube through rings at the heights zs, each ring traced by point(z, phi) -> (x, y[, dz])."""
    bm = bmesh.new()
    loops = []
    for z in zs:
        loop = []
        for k in range(n):
            phi = 2 * math.pi * k / n
            p = point(z, phi)
            loop.append(bm.verts.new((p[0], p[1], z + (p[2] if len(p) > 2 else 0.0))))
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


def front_y(bm, x, z):
    """Where the front of a mesh is at (x, z), seen from the camera side."""
    return surface_y(bm, x, z)


# ---------------------------------------------------------------- the rig
def _smooth(a, b, z):
    """0 above a, 1 below b, eased between (a > b)."""
    t = min(1.0, max(0.0, (a - z) / (a - b)))
    return t * t * (3 - 2 * t)


def leg_weights(side):
    kz, az = HIP_Z - THIGH, HIP_Z - THIGH - SHIN

    def w(co):
        lower = _smooth(kz + 1.4, kz - 1.4, co.z)      # thigh -> shin across the knee
        foot = _smooth(az + 0.5, az - 0.5, co.z)       # shin -> foot across the ankle
        return {f'Thigh.{side}': 1 - lower, f'Shin.{side}': lower * (1 - foot), f'Foot.{side}': lower * foot}
    return w


def arm_weights(side):
    ez, wz = SH_Z - UPPER, SH_Z - UPPER - FORE

    def w(co):
        lower = _smooth(ez + 1.1, ez - 1.1, co.z)      # upper arm -> forearm across the elbow
        hand = _smooth(wz + 0.4, wz - 0.4, co.z)       # forearm -> hand across the wrist
        return {f'UpperArm.{side}': 1 - lower, f'Forearm.{side}': lower * (1 - hand), f'Hand.{side}': lower * hand}
    return w


def build_rig(body):
    """An armature with a bone for each limb segment, built in the rest pose (limbs straight down). It sits
    unparented at the origin: each bone follows its pivot empty through a Child Of constraint (set up in
    finish_rig, once the pivots exist), so the pivots stay the one place the pose is keyed."""
    arm_data = bpy.data.armatures.new('Rig')
    arm = bpy.data.objects.new('Rig', arm_data)
    link(arm)
    RIG['arm'] = arm
    vl = bpy.context.view_layer
    vl.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm_data.edit_bones
    for side, s in (('R', 1), ('L', -1)):
        lx, ax = -s * LEG_X, -s * SH_X
        segs = [('Thigh', (lx, 0, HIP_Z), (lx, 0, HIP_Z - THIGH)),
                ('Shin', (lx, 0, HIP_Z - THIGH), (lx, 0, HIP_Z - THIGH - SHIN)),
                ('Foot', (lx, 0, ANKLE_Z), (lx, -3.0, 0.5)),
                ('UpperArm', (ax, 0.2, SH_Z), (ax, 0.2, SH_Z - UPPER)),
                ('Forearm', (ax, 0.2, SH_Z - UPPER), (ax, 0.2, SH_Z - UPPER - FORE)),
                ('Hand', (ax, 0.2, SH_Z - UPPER - FORE), (ax, 0.2, SH_Z - UPPER - FORE - 2.5))]
        for name, h, t in segs:
            b = eb.new(f'{name}.{side}')
            b.head, b.tail = h, t
    bpy.ops.object.mode_set(mode='OBJECT')


PIVOT_OF = {'Thigh': 'Hip', 'Shin': 'Knee', 'Foot': 'Ankle', 'UpperArm': 'Shoulder', 'Forearm': 'Elbow', 'Hand': 'Wrist'}


def finish_rig():
    arm = RIG['arm']
    bpy.context.view_layer.update()
    for pb in arm.pose.bones:
        seg, side = pb.name.split('.')
        target = bpy.data.objects[f'{PIVOT_OF[seg]}.{side}']
        c = pb.constraints.new('CHILD_OF')
        c.target = target
        # (taken in the rest pose: the bone then moves exactly as its pivot moves away from rest)
        c.inverse_matrix = target.matrix_world.inverted()
    arm.hide_render = True


def hand_mesh(s):
    """A hand hanging relaxed from the wrist (at the origin, fingers down), made as one fused surface: metaball
    pieces for the palm, the four fingers (two segments each, curled loosely toward the palm, longest at the
    middle) and the thumb lying forward along the index finger melt into each other, so the fingers grow out of
    the palm with webbing between them, and the result is converted to a mesh. The palm faces the body (s = 1
    for the right hand, whose inside faces +X); the front is -Y."""
    inward = s
    mb = bpy.data.metaballs.new('hand')
    mb.resolution = mb.render_resolution = 0.09
    mb.threshold = 0.6
    ob = bpy.data.objects.new('hand-build', mb)
    link(ob)
    K = 1.55     # (a metaball's surface sits well inside its radius at this threshold)

    def capsule(p, q, r):
        p, q = Vector(p), Vector(q)
        e = mb.elements.new(type='CAPSULE')
        e.co = (p + q) / 2
        e.radius = r * K
        e.size_x = (q - p).length / 2
        e.rotation = (q - p).normalized().to_track_quat('X', 'Z')
        e.stiffness = 2.2

    def blob(c, size, r):
        e = mb.elements.new(type='ELLIPSOID')
        e.co = c
        e.radius = r * K
        e.size_x, e.size_y, e.size_z = size
        e.stiffness = 2.0

    # the wrist: rising from inside the cuff and widening into the heel and back of the hand, so the hand grows
    # out of the forearm instead of hanging below it
    # (flattened like a real wrist, wider across the back of the hand than through it, so the hand does not
    # flare out of a round stalk)
    blob((0.0, 0.05, 1.0), (0.42, 0.62, 1.0), 0.75)
    blob((0.0, 0.05, -0.15), (0.44, 0.8, 0.7), 0.85)
    # the palm: thick at the heel, broad at the knuckles
    blob((0.0, 0.05, -0.9), (0.42, 0.95, 0.8), 1.0)
    blob((0.02 * inward, 0.0, -2.1), (0.4, 1.12, 0.6), 1.0)
    # fingers, index at the front to pinky at the back
    for y0, L, r in ((-0.9, 1.95, 0.36), (-0.3, 2.1, 0.37), (0.3, 2.0, 0.35), (0.78, 1.6, 0.32)):
        p = Vector((0.05 * inward, y0, -2.55))
        ang = 0.3
        for frac, extra, rr in ((0.55, 0.0, r), (0.45, 0.6, r * 0.88)):
            ang += extra
            d = Vector((inward * math.sin(ang), 0.0, -math.cos(ang)))
            q = p + d * (L * frac)
            capsule(p, q, rr)
            p = q
    # the base of the hand where the thumb and the fingers share the wrist: a thenar mound swelling from the
    # wrist into the root of the thumb and a smaller hypothenar mound on the little-finger side, so the thumb,
    # the palm and the four fingers all grow out of one fleshy heel instead of meeting in a crease
    blob((0.18 * inward, -0.5, -0.95), (0.5, 0.6, 0.8), 0.9)
    blob((0.1 * inward, 0.5, -1.1), (0.4, 0.4, 0.75), 0.75)
    # the little-finger edge of the palm: one straight ridge from the wrist to the pinky knuckle, so the
    # outline runs smoothly down that side with no dip between the heel of the hand and the pinky
    capsule((0.05 * inward, 0.62, 0.3), (0.05 * inward, 0.72, -2.45), 0.46)
    capsule((0.05 * inward, -0.2, -0.2), (0.22 * inward, -0.75, -1.1), 0.5)
    # the thumb, from the heel of the palm on the inner front edge, its tip bent in toward the fingers
    t0 = Vector((0.2 * inward, -0.7, -0.9))
    t1 = t0 + Vector((0.3 * inward, -0.25, -1.1))
    t2 = t1 + Vector((0.22 * inward, 0.05, -0.8))
    # the web of skin from the thumb's first joint to the root of the index finger
    capsule(t0 + (t1 - t0) * 0.8, (0.1 * inward, -0.85, -2.3), 0.3)
    capsule(t0, t1, 0.46)
    capsule(t1, t2, 0.37)

    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    bpy.data.objects.remove(ob)
    bpy.data.metaballs.remove(mb)
    bm = bmesh.new()
    bm.from_mesh(me)
    bpy.data.meshes.remove(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.scale(bm, vec=(1.22, 1.22, 1.22), verts=bm.verts)   # about the length of the face, as in life
    return bm


def dots(points, size=0.42):
    """Small blocks at the given points: a row of stitches."""
    bm = bmesh.new()
    for p in points:
        r = bmesh.ops.create_cube(bm, size=size)
        bmesh.ops.translate(bm, vec=Vector(p), verts=r['verts'])
    return bm


def outward(p, d):
    """Push a point of a horizontal ring (x, y[, dz]) out from the body's axis by d."""
    x, y = p[0], p[1]
    n = math.hypot(x, y) or 1.0
    return x + x / n * d, y + y / n * d


def tube(pts, radii, flat=0.5, n=10):
    """A tapered tube along a polyline (lash lines, lids, brows): radii[i] across the face at pts[i], flat times
    that through it (along Y, toward the camera), with rounded ends so it never stops in a cut-off disc."""
    bm = bmesh.new()
    pts = [Vector(p) for p in pts]
    rings = []
    for i, p in enumerate(pts):
        t = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
        a = t.cross(Vector((0, 1, 0)))
        a = a.normalized() if a.length > 1e-6 else Vector((0, 0, 1))
        b = a.cross(t).normalized()
        r = radii[i]
        rings.append([bm.verts.new(p + a * (r * math.cos(2 * math.pi * k / n)) + b * (r * flat * math.sin(2 * math.pi * k / n)))
                      for k in range(n)])
    for ra, rb in zip(rings, rings[1:]):
        for k in range(n):
            bm.faces.new((ra[k], ra[(k + 1) % n], rb[(k + 1) % n], rb[k]))
    for ring, p in ((rings[0], pts[0]), (rings[-1], pts[-1])):
        c = bm.verts.new(p)
        for k in range(n):
            bm.faces.new((ring[k], ring[(k + 1) % n], c))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


EYE_RX, EYE_RZ = 0.76, 0.36
EYE_TILT = 0.092       # the outer corner sits higher than the inner one (as in the photograph)
EYE_LIFT = -0.05        # the line between the corners sits this much below the iris's centre (as in the photograph)
EYE_K = 1.3            # the ball reaches this much past the lids' opening, up and down, and turns in under them


# The opening of the eye, constructed from the eyelid's anatomy rather than traced. Measured adult proportions
# (oculoplastic references): the fissure is about 30 mm wide and 10 mm high; the upper lid is highest a little to
# the nose side of the pupil and the lower lid lowest a little to the temple side of it, so the two peaks sit on a
# diagonal; the lids meet at roughly 60 degrees at each corner; and the outer corner sits 2-3 mm above the inner one
# (EYE_TILT). At the inner corner the lids pull away from the ball round the caruncle and meet in a short
# straight wedge before they turn into the arch; at the outer corner they lie on the ball and close more steeply. Artists build the same shape from planes: the upper lid as two side planes and
# a top, the lower lid as two, joined by curves.
#
# Each lid is a short straight run out of the inner corner (the wedge round the caruncle) followed by two cubic
# Hermite segments, run end -> peak -> outer corner, with a level tangent at the peak. The run is short and its
# tangent carries straight on into the first segment, so the lid leaves the corner straight for a moment and then
# bends continuously into the arch (no long straight ramp). At the outer corner the lids close more steeply.
# Heights are in units of the eye's half-width (EYE_RX), above or below the line joining the corners
# (u -1 inner .. 1 outer).
EYE_OPEN = 0.72               # the opening's height, as a fraction of the reference (a relaxed, slightly narrowed eye)
EYE_TOP_PEAK = (-0.18, 0.62)   # where the upper lid is highest (u) and how high (before EYE_OPEN): nasal of centre
EYE_BOT_LOW = (0.30, 0.32)     # where the lower lid is lowest and how low: temporal of centre
EYE_INNER_ANGLES = (50.0, 13.0)    # degrees the upper and lower lids leave the inner corner at (about 63 between them)
EYE_OUTER_ANGLES = (44.0, 30.0)    # degrees they close into the outer corner at
EYE_INNER_RUN = 0.05          # how far the lids run straight out of the inner corner, in units of EYE_RX
EYE_INNER_TANGENT = 0.8       # how far that straight direction carries into the curve (fraction of the chord to the peak)
EYE_PEAK_FULL = (1.25, 1.2)   # peak tangent length (fraction of the chord), upper and lower: how full each arch is
EYE_OUTER_TANGENT = 0.5       # tangent length at the outer corner (fraction of the chord)

_LID_CACHE = {}


def _lid_curve(top):
    """The sampled curve of one lid (u, height in units of EYE_RX), built as above; cached per setting."""
    i = 0 if top else 1
    peak = EYE_TOP_PEAK if top else EYE_BOT_LOW
    key = (top, peak, EYE_OPEN, EYE_INNER_ANGLES, EYE_OUTER_ANGLES, EYE_INNER_RUN, EYE_INNER_TANGENT, EYE_PEAK_FULL, EYE_OUTER_TANGENT)
    if key in _LID_CACHE:
        return _LID_CACHE[key]
    pu, ph = peak[0], peak[1] * EYE_OPEN
    ai, ao = math.radians(EYE_INNER_ANGLES[i]), math.radians(EYE_OUTER_ANGLES[i])
    dx, dy = math.cos(ai), math.sin(ai)

    def hermite(p0, t0, p1, t1, n=240):
        out = []
        for k in range(n):
            s = k / n
            h00, h10, h01, h11 = 2 * s ** 3 - 3 * s * s + 1, s ** 3 - 2 * s * s + s, -2 * s ** 3 + 3 * s * s, s ** 3 - s * s
            out.append((h00 * p0[0] + h10 * t0[0] + h01 * p1[0] + h11 * t1[0], h00 * p0[1] + h10 * t0[1] + h01 * p1[1] + h11 * t1[1]))
        return out
    q = (-1.0 + EYE_INNER_RUN * dx, EYE_INNER_RUN * dy)
    pts = [(-1.0 + EYE_INNER_RUN * dx * k / 20, EYE_INNER_RUN * dy * k / 20) for k in range(20)]
    c0 = math.hypot(pu - q[0], ph - q[1])
    c1 = math.hypot(1 - pu, ph)
    pf = EYE_PEAK_FULL[i]
    pts += hermite(q, (EYE_INNER_TANGENT * c0 * dx, EYE_INNER_TANGENT * c0 * dy), (pu, ph), (pf * c0, 0.0))
    pts += hermite((pu, ph), (pf * c1, 0.0), (1.0, 0.0), (EYE_OUTER_TANGENT * c1 * math.cos(ao), -EYE_OUTER_TANGENT * c1 * math.sin(ao)))
    pts.append((1.0, 0.0))
    pts.sort()
    us = np.array([p[0] for p in pts])
    hs = np.maximum(0.0, np.array([p[1] for p in pts]))
    _LID_CACHE[key] = (us, hs)
    return us, hs


def lid_top(u):
    """How far the upper lid's edge stands above the line between the corners at u (-1 inner corner .. 1 outer
    corner)."""
    us, hs = _lid_curve(True)
    return EYE_RX * float(np.interp(max(-1.0, min(1.0, u)), us, hs))


def lid_bottom(u):
    """The lower lid's edge below the line between the corners."""
    us, hs = _lid_curve(False)
    return EYE_RX * float(np.interp(max(-1.0, min(1.0, u)), us, hs))


def almond(rx, ry, rz, s, seg=192, rings=128):
    """The eyeball: an ellipsoid reshaped to the lids' opening (lid_top over lid_bottom, tapering to a point at
    each corner rather than a rounded oval), a little larger than it, so its edge turns in under the skin of the
    lids all round (see eye_relief) and the lids close over it with no seam."""
    bm = ellipsoid(rx, ry, rz, seg, rings)
    for v in bm.verts:
        u = max(-1.0, min(1.0, v.co.x / rx)) * s          # + toward the outer corner
        half = rz * math.sqrt(max(1e-6, 1 - u * u))
        zn = max(-1.0, min(1.0, v.co.z / half))
        v.co.z = zn * (lid_top(u) if zn > 0 else lid_bottom(u)) * EYE_K
        # its front stays full almost to the corners (an ellipsoid falls away there, so the skin closed over the
        # tips and rounded them off), so the opening's own straight edges meet in a clear point
        v.co.y *= max(1e-3, 1 - u * u) ** -0.3 if abs(u) < 0.999 else 1.0
    return bm


def iris_dome(r, depth, rings=48, seg=96):
    """The iris (and the cornea over it) as one shallow dome facing -Y: rings round its centre, so the painted
    pupil and the rings of the iris follow the mesh, and a smooth curve for the highlight to slide over."""
    bm = bmesh.new()
    tip = bm.verts.new((0, -depth, 0))
    prev = None
    for i in range(1, rings + 1):
        t = i / rings
        ring = [bm.verts.new((r * t * math.cos(2 * math.pi * k / seg), -depth * math.sqrt(max(0.0, 1 - t * t)),
                              r * t * math.sin(2 * math.pi * k / seg))) for k in range(seg)]
        if prev is None:
            for k in range(seg):
                bm.faces.new((tip, ring[(k + 1) % seg], ring[k]))
        else:
            for k in range(seg):
                bm.faces.new((prev[k], prev[(k + 1) % seg], ring[(k + 1) % seg], ring[k]))
        prev = ring
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for f in bm.faces:            # facing the camera
        if f.calc_center_median().y < 0 and f.normal.y > 0:
            f.normal_flip()
    return bm


def _srgb_lin(c):
    return [((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c]


IRIS_TEX = {}


def _vnoise(x, y, px, seed):
    """Value noise on a lattice, smoothly eased, wrapping every px cells in x (so it closes round the iris)."""
    g = np.random.default_rng(seed).uniform(-1, 1, (int(np.ceil(y.max())) + 2, px))
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    tx, ty = x - ix, y - iy
    tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
    x0, x1 = ix % px, (ix + 1) % px
    a = g[iy, x0] + (g[iy, x1] - g[iy, x0]) * tx
    b = g[iy + 1, x0] + (g[iy + 1, x1] - g[iy + 1, x0]) * tx
    return a + (b - a) * ty


def _polar_fbm(th, q, cells, rows, octaves, seed, gain=0.55):
    """Fractal noise in polar coordinates: round the iris (th, 0..1) it has `cells` lattice cells, out along the
    radius (q, 0..1) only `rows`, so each feature is drawn out along the radius into a fibre. Each octave doubles
    both, and wraps exactly round the circle."""
    tot, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        c = cells * 2 ** o
        tot = tot + amp * _vnoise(th * c, q * rows * 2 ** o, c, seed + o)
        norm += amp
        amp *= gain
    return tot / norm


def iris_image():
    """The iris, painted in polar coordinates the way the real one is built. The stroma is fractal noise
    stretched out along the radius into fibres, its angle warped by a second, coarser noise so the fibres wave
    rather than run ruler-straight. Crypts are dark lozenges from cellular (Worley) noise, packed round the
    collarette. The collarette itself is a ragged ring a little out from the pupil, its radius wandering with a
    low-frequency noise; inside it the pupillary zone is warmer and smoother. Faint contraction furrows ring the
    outer zone, broken up by noise. Then the light: the dark limbal ring at the rim, the upper lid's shadow over
    the top, and light gathered low on the far side where the cornea focuses it. One painting serves both eyes
    (u mirrored for the left one); the disc fills the square."""
    if 'img' in IRIS_TEX:
        return IRIS_TEX['img']
    N = 512
    rmp = np.array(MATERIALS[M_EYE]['ramp'], dtype=np.float32)
    X, Z = np.meshgrid((np.arange(N) + 0.5) / N * 2 - 1, 1 - (np.arange(N) + 0.5) / N * 2)
    q = np.minimum(np.hypot(X, Z), 1.0)
    th = (np.arctan2(Z, X) / (2 * np.pi)) % 1.0

    def sm(e0, e1, x):
        t = np.clip((x - e0) / (e1 - e0), 0, 1)
        return t * t * (3 - 2 * t)
    # the fibres: warped angle, then anisotropic fBm (many cells round, few out)
    warp = 0.012 * _polar_fbm(th, q, 6, 3, 3, 11)
    thw = th + warp * (0.4 + q)
    fib = _polar_fbm(thw, q, 64, 2, 4, 21)                 # fine, long fibres
    band = _polar_fbm(thw, q, 14, 1.5, 2, 31)              # broad, lighter and darker wedges
    # the crypts: Worley F1 on points scattered round the collarette ring
    rng = np.random.default_rng(41)
    n_c = 38
    ct = rng.uniform(0, 1, n_c)
    cq = np.clip(rng.normal(0.56, 0.07, n_c), 0.42, 0.78)
    cx, cz = cq * np.cos(ct * 2 * np.pi), cq * np.sin(ct * 2 * np.pi)
    f1 = np.full(X.shape, 9.0)
    for i in range(n_c):
        # lozenges drawn out along the radius: distance squashed across the radius
        rx, rz = cx[i] / cq[i], cz[i] / cq[i]
        dx, dz = X - cx[i], Z - cz[i]
        along = dx * rx + dz * rz
        across = -dx * rz + dz * rx
        f1 = np.minimum(f1, np.hypot(along * 0.55, across * 1.3) / rng.uniform(0.035, 0.06))
    crypt = sm(1.0, 0.45, f1)
    # the collarette: a ragged ring, its radius wandering
    rag = 0.5 + 0.035 * _polar_fbm(th, q, 9, 1, 2, 51)
    coll = np.exp(-((q - rag) / 0.035) ** 2)
    pupillary = sm(rag + 0.02, rag - 0.04, q)               # inside the collarette
    furrow = (0.5 + 0.5 * np.cos(q * 46)) * sm(0.66, 0.78, q) * sm(0.96, 0.86, q) * (0.5 + 0.5 * band)
    t = 0.5 + 0.30 * fib + 0.10 * band
    t -= 0.16 * crypt
    t -= 0.06 * furrow
    t += 0.14 * coll
    t -= 0.10 * pupillary * (0.6 - 0.4 * fib)                # smoother, darker-toned zone round the pupil
    # the light
    t -= 0.5 * sm(0.8, 1.0, q)                                  # the limbal ring
    t -= 0.42 * sm(0.0, 0.8, Z + 0.15 * X * X)                  # the upper lid's shadow
    glow = np.exp(-((X + 0.2) ** 2 + (Z + 0.5) ** 2) / 0.22)
    t += 0.26 * glow * sm(0.36, 0.5, q) * sm(0.95, 0.8, q)      # light gathered by the cornea
    warm = sm(rag + 0.06, rag - 0.08, q) * (0.75 + 0.25 * fib)
    t = np.clip(t + 0.16 * warm, 0, 1)
    k = t * 4
    i0 = np.minimum(3, k.astype(int))
    c = rmp[i0] + (rmp[i0 + 1] - rmp[i0]) * (k - i0)[..., None]
    amber = np.array([156, 134, 70], dtype=np.float32)
    c = c + (amber - c) * (0.45 * warm)[..., None]
    pupil = sm(0.35, 0.31, q)
    c = c + (np.array([20, 18, 18], dtype=np.float32) - c) * pupil[..., None]
    img = bpy.data.images.new('iris', N, N, alpha=False)
    px = np.concatenate([c / 255, np.ones((N, N, 1), dtype=np.float32)], axis=2)[::-1]
    img.pixels.foreach_set(px.ravel().astype(np.float32))
    img.pack()
    IRIS_TEX['img'] = img
    IRIS_TEX['t'] = t
    return img


def iris_colours(me, disc, r, s):
    """Lay the painted iris (iris_image) onto the iris mesh through texture coordinates across the disc, and store
    its tone at each point along the iris ramp ('iris_shade'), which the sprite renderer reads through the detail
    pass."""
    iris_image()
    tt = IRIS_TEX['t']
    N = tt.shape[0]
    uvl = me.uv_layers.new(name='UVMap')
    pos = {}
    for v in me.vertices:
        # from the flat disc the iris was made as, not from where it lies now (laid onto the ball, and taken back
        # under the lids, the points move in depth, and through the iris's tilt that dragged the painting sideways)
        dx, dz = disc[v.index]
        pos[v.index] = ((dx * s / r + 1) / 2, (dz / r + 1) / 2)
    for lp in me.loops:
        uvl.data[lp.index].uv = pos[lp.vertex_index]
    shades = []
    for v in me.vertices:
        u, w = pos[v.index]
        i = min(N - 1, max(0, int((1 - w) * N)))
        j = min(N - 1, max(0, int(u * N)))
        q = math.hypot(u * 2 - 1, w * 2 - 1)
        dark = smooth(0.35, 0.31, q)
        shades.append(float(tt[i, j]) * (1 - dark) + 0.25 * dark)
    sa = me.attributes.new('iris_shade', 'FLOAT', 'POINT')
    sa.data.foreach_set('value', shades)


def smooth(e0, e1, x):
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


SCLERA_TEX = {}


def sclera_image():
    """The white of the eye, painted into a texture laid across the eye's opening (u from the inner corner to the
    outer one across, w from the bottom to the top; one texture for both eyes, mirrored). Never pure white but a
    soft ivory, cooler and greyer toward the corners where it turns away into the socket, the pink of the caruncle
    at the inner corner, the shadow the upper lid casts across the top, and fine red veins: each starts at a corner
    and wanders in toward the iris, thinning and forking as it goes, and fades before it reaches it."""
    if 'img' in SCLERA_TEX:
        return SCLERA_TEX['img']
    W = 512
    H = int(W * EYE_RZ * EYE_K / EYE_RX)
    uu, ww = np.meshgrid((np.arange(W) + 0.5) / W * 2 - 1, 1 - (np.arange(H) + 0.5) / H * 2)
    c = np.zeros((H, W, 3), dtype=np.float32) + np.array([236, 231, 220], dtype=np.float32)

    def blend(col, k):
        c[:] = c + (np.array(col, dtype=np.float32) - c) * k[..., None]

    def sm(e0, e1, x):
        t = np.clip((x - e0) / (e1 - e0), 0, 1)
        return t * t * (3 - 2 * t)
    def field(nx, ny, seed):
        """Smooth noise over the texture in -1..1: random values on an nx by ny lattice, eased between."""
        g = np.random.default_rng(seed).uniform(-1, 1, (ny + 1, nx + 1))
        fx = (np.arange(W) + 0.5) / W * nx
        fy = (np.arange(H) + 0.5) / H * ny
        ix, iy = np.minimum(fx.astype(int), nx - 1), np.minimum(fy.astype(int), ny - 1)
        tx, ty = fx - ix, fy - iy
        tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
        a_ = g[iy][:, ix] * (1 - tx) + g[iy][:, ix + 1] * tx
        b_ = g[iy + 1][:, ix] * (1 - tx) + g[iy + 1][:, ix + 1] * tx
        return a_ * (1 - ty)[:, None] + b_ * ty[:, None]
    # where each point lies between the eye's centre line (0) and the lid's edge (1), above and below
    top = np.array([[lid_top(u) for u in row] for row in uu]) / EYE_RZ
    bot = np.array([[lid_bottom(u) for u in row] for row in uu]) / EYE_RZ
    e = np.where(ww > 0, ww * EYE_K / np.maximum(top, 0.05), -ww * EYE_K / np.maximum(bot, 0.05))
    # the ball's roundness: a marble lit from the front, brightest a little above and inside the middle, its
    # tone turning warmer and darker as it curves away toward each corner, where it sinks into the socket
    ru = np.abs(uu)
    # lit as a ball: the white is part of a sphere a little wider than the opening, and its tone follows how
    # much each point faces a light in front of and above the face (Lambert), so it is brightest just above the
    # middle, turns gradually darker down toward the lower lid and round into each corner, and never shows a
    # ring; the shade is a warm, darkened ivory, never grey or blue
    R = 1.12
    px_, pz_ = uu / R, (ww * EYE_RZ * EYE_K / EYE_RX) / R
    ny_ = np.sqrt(np.clip(1 - px_ * px_ - pz_ * pz_, 0, 1))
    L = np.array([0.0, 0.3, 0.954])                                          # (across, up, toward the viewer)
    lam = np.clip(px_ * L[0] + pz_ * L[1] + ny_ * L[2], 0, 1)
    blend([156, 124, 110], 0.75 * (1 - lam) ** 1.1)
    blend([168, 140, 134], 0.5 * sm(0.75, 1.0, ru))
    blend([214, 150, 138], 0.85 * sm(0.8, 0.97, -uu))                         # the caruncle, at the inner corner
    # the skin of the white: faint warm and cool patches and a fine grain, so it never reads as flat paint
    m = field(9, 4, 11) * 0.6 + field(23, 9, 12) * 0.4
    blend([234, 212, 178], 0.35 * np.clip(m, 0, 1))
    blend([214, 204, 192], 0.3 * np.clip(-m, 0, 1))
    c[:] = c * (1 + 0.04 * field(90, 30, 13) + 0.02 * field(200, 70, 14))[..., None]
    # where it meets the lids: the upper lid shades the top of the ball in a soft, broad band that deepens
    # into the lid's edge, and the lower lid's wet rim is a thin pink line with a little shade above it, so the
    # white passes into the skin through a tone rather than meeting it at a hard edge
    up = np.where(ww > 0, e, 0)
    lo = np.where(ww < 0, e, 0)
    blend([176, 160, 154], 0.6 * sm(0.15, 0.85, up))
    blend([128, 104, 100], 0.75 * sm(0.75, 1.0, up))
    blend([182, 162, 156], 0.45 * sm(0.45, 0.85, lo))
    blend([196, 128, 120], 0.7 * sm(0.82, 1.0, lo))
    # the corners: where the lids close over the ball it is in their shade and wet with the pink of the lid's
    # inner skin, so toward each point the white warms and darkens into the skin's own tone, and the lids'
    # edges continue round it rather than stopping at a pale tip
    blend([204, 154, 132], sm(0.42, 0.88, ru) ** 0.8)
    # the veins: the conjunctival vessels enter from under the lids and the corners and branch inward toward
    # the iris, thinning as they go. They are grown by space colonisation (Runions et al., the algorithm for leaf
    # venation): scattered attraction points, denser toward the corners and thinning out before the iris, pull
    # the nearest tip of the network toward them and are used up when it arrives, so the branches spread to fill
    # the space without crossing. Each vessel's width then follows Murray's law (a parent's radius cubed is the sum
    # of its children's), so the trunks are thick and the twigs hair-fine, and the drawn paths wander with a
    # low-frequency noise so no segment runs straight.
    rng = np.random.default_rng(7)
    PX = W / 2.0                                  # pixels per unit of u (and roughly of the opening's height)
    def inside(u_, h_):
        # within the opening (h_ in the same units as lid_top), with a little margin from the lids' edges
        return (-lid_bottom(u_) * 0.92 < h_ < lid_top(u_) * 0.92) if abs(u_) < 0.97 else False
    att = []
    while len(att) < 1400:
        u_ = rng.uniform(-1, 1)
        h_ = rng.uniform(-0.5, 0.6)
        dens = smooth(0.28, 0.75, abs(u_)) * (0.35 + 0.65 * smooth(0.3, 1.0, abs(h_) / 0.4))
        if rng.random() < dens and inside(u_, h_):
            att.append((u_, h_))
    att = np.array(att)
    # roots: under the lids (above and below, toward the sides) and at both corners
    roots = []
    for side in (-1, 1):
        roots.append((0.985 * side, 0.0))
        for uq in (0.35, 0.5, 0.62, 0.75, 0.88):
            roots.append((uq * side + rng.normal(0, 0.04), lid_top(uq) * 0.97))
            roots.append((uq * side + rng.normal(0, 0.04), -lid_bottom(uq) * 0.97))
    # and scattered seeds within the white itself, so it fills with many short, separate little trees rather
    # than a few long ones running in from the edges
    for k_ in rng.choice(len(att), 36, replace=False):
        roots.append(tuple(att[k_]))
    nodes = [np.array(r_, dtype=np.float64) for r_ in roots]
    parent = [-1] * len(nodes)
    D, DI, DK = 0.007, 0.07, 0.012                # step, reach of an attractor, distance at which it is used up
    alive = np.ones(len(att), dtype=bool)
    for it in range(900):
        if not alive.any():
            break
        N_ = np.array(nodes)
        A = att[alive]
        d = np.linalg.norm(A[:, None, :] - N_[None, :, :], axis=2)
        near = d.argmin(1)
        dmin = d[np.arange(len(A)), near]
        grow_dirs = {}
        for ai in np.nonzero(dmin < DI)[0]:
            v_ = A[ai] - N_[near[ai]]
            grow_dirs.setdefault(near[ai], []).append(v_ / max(1e-9, np.linalg.norm(v_)))
        if not grow_dirs:
            break
        for ni, dirs in grow_dirs.items():
            g_ = np.sum(dirs, axis=0)
            g_ = g_ / max(1e-9, np.linalg.norm(g_)) + rng.normal(0, 0.3, 2)
            g_ /= max(1e-9, np.linalg.norm(g_))
            nodes.append(N_[ni] + g_ * D)
            parent.append(ni)
        N_ = np.array(nodes)
        idx = np.nonzero(alive)[0]
        dd = np.linalg.norm(att[idx][:, None, :] - N_[None, :, :], axis=2).min(1)
        alive[idx[dd < DK]] = False
    # widths from the tips back toward the roots (Murray's law)
    n_ = len(nodes)
    rad = np.full(n_, 0.3)                        # a tip's radius in pixels
    kids = [[] for _ in range(n_)]
    for i_, p_ in enumerate(parent):
        if p_ >= 0:
            kids[p_].append(i_)
    for i_ in range(n_ - 1, -1, -1):
        if kids[i_]:
            rad[i_] = sum(rad[k_] ** 3 for k_ in kids[i_]) ** (1 / 3)
    rad = np.minimum(rad * 1.1, 1.5)
    # draw each segment as a soft capsule, its path wandering with a smooth noise
    wob = lambda q_, sd: 0.006 * (np.sin(q_[0] * 37 + sd) + np.sin(q_[1] * 53 - sd * 1.7) + np.sin((q_[0] + q_[1]) * 91 + sd * 0.3))
    pts = [np.array([(q_[0] + wob(q_, 1.0) + 1) / 2 * W, (1 - (q_[1] + wob(q_, 4.0)) / (EYE_RZ * EYE_K)) / 2 * H]) for q_ in nodes]
    mask = np.zeros((H, W), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for i_, p_ in enumerate(parent):
        if p_ < 0:
            continue
        a_, b_ = pts[p_], pts[i_]
        r_ = 0.5 * (rad[i_] + rad[p_])
        x0, x1 = int(max(0, min(a_[0], b_[0]) - 4)), int(min(W, max(a_[0], b_[0]) + 5))
        y0, y1 = int(max(0, min(a_[1], b_[1]) - 4)), int(min(H, max(a_[1], b_[1]) + 5))
        if x0 >= x1 or y0 >= y1:
            continue
        X_, Y_ = xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
        ab = b_ - a_
        t_ = np.clip(((X_ - a_[0]) * ab[0] + (Y_ - a_[1]) * ab[1]) / max(1e-9, ab @ ab), 0, 1)
        dist = np.hypot(X_ - (a_[0] + t_ * ab[0]), Y_ - (a_[1] + t_ * ab[1]))
        k_ = (0.3 + 0.7 * min(1.0, r_ / 1.4)) * np.exp(-(dist / max(0.55, r_)) ** 2)
        np.maximum(mask[y0:y1, x0:x1], k_, out=mask[y0:y1, x0:x1])
    mask *= sm(0.22, 0.55, np.abs(uu)) * (0.75 + 0.25 * sm(0.6, 0.9, np.abs(uu)))   # faint mid-way, gone before the iris
    # a faint flush round the larger vessels, then the vessels themselves, deeper red where they are thicker
    soft = mask.copy()
    for _ in range(3):
        soft = (soft + np.roll(soft, 1, 0) + np.roll(soft, -1, 0) + np.roll(soft, 1, 1) + np.roll(soft, -1, 1)) / 5
    blend([214, 140, 128], 0.18 * np.clip(soft * 2.0, 0, 1))
    blend([180, 62, 60], 0.68 * mask)
    img = bpy.data.images.new('sclera', W, H, alpha=False)
    px = np.concatenate([c / 255, np.ones((H, W, 1), dtype=np.float32)], axis=2)[::-1]
    img.pixels.foreach_set(px.ravel().astype(np.float32))
    img.pack()
    SCLERA_TEX['img'] = img
    return img


def eye_relief(x, z, s, ex, ez):
    """How far the skin round one eye stands forward (head coordinates; the eye's centre at ex, ez): the lids and
    the folds about them, sculpted into the face's own surface so they flow out of it rather than sitting on it.
    A few broad, soft forms, each following the curve of the opening (lid_top, lid_bottom) and ending at or before
    its corners (lines carried on past the corners, or several narrow bands, read as stray creases): the rim of the
    upper lid rolling over the top of the ball, one crease above it, the rim of the lower lid, and one shallow fold
    under it where the lower lid meets the cheek."""
    def g(t, w):
        return math.exp(-(t / w) ** 2)
    u = (x - ex) * s / EYE_RX
    if abs(u) > 1.9:
        return 0.0
    uc = max(-1.0, min(1.0, u))
    dz = z - ez - u * EYE_RX * EYE_TILT
    T, B = lid_top(uc), lid_bottom(uc)
    inside = smooth(1.08, 0.85, abs(u))                # 1 within the opening, easing out past its corners
    d = 0.0
    # the opening itself: inside the lids' edges the skin is drawn well back behind the ball, so exactly the
    # traced shape of white shows, its edge where the lids meet the ball
    if abs(u) < 1.0:
        d -= 0.12 * smooth(0.0, 0.12, dz + B) * smooth(0.0, 0.12, T - dz)
    # the lid margins: one rounded lip running round the whole opening, the upper and lower lids joining in it at
    # each corner, standing a little proud of the ball and rolling onto it, so the ball fills the opening from
    # behind the lids rather than sitting in a hole with a gap round it
    cornr = smooth(1.06, 0.96, abs(u))
    lip_t = 0.085 * g(dz - (T - 0.012 * min(1.0, T / 0.08)), 0.055) * (0.6 + 0.4 * (1 - uc * uc))
    lip_b = 0.07 * g(dz + (B - 0.012 * min(1.0, B / 0.08)), 0.055) * (0.6 + 0.4 * (1 - uc * uc))
    d += max(lip_t, lip_b) * cornr
    # upper lid: a broad rim along the edge of the opening, fullest over the iris, closing to the corners
    d += 0.06 * g(dz - (T - 0.045), 0.1) * (0.3 + 0.7 * (1 - uc * uc) ** 0.5) * inside
    # the crease of the upper lid: a clear, narrow fold a lid's breadth above the edge, following its curve and
    # fading out toward either corner, with the soft fullness under the brow rolling over it from above (the
    # upper lid is the one that moves, so it carries the eye's one strong fold)
    cz = T + 0.13 + 0.035 * (1 - uc * uc)
    cf = smooth(1.05, 0.55, abs(u + 0.05))
    d -= 0.014 * g(dz - cz, 0.045) * cf
    # lower lid: a soft rim along the lower edge, and one shallow fold where it meets the cheek
    d += 0.04 * g(dz + (B - 0.02), 0.085) * (0.3 + 0.7 * (1 - uc * uc) ** 0.5) * inside
    # past the outer corner: the upper lid laps over the lower and its fold runs on a little way beyond the
    # cusp, a tapering groove whose upper wall faces down into shadow, with a soft hollow round it (as in a real
    # eye, the corner is the deepest, darkest point and the shade fades out along the line of the lid)
    if u > 0.9:
        k = max(0.0, u - 1.0)
        taper = smooth(0.9, 1.0, u) * max(0.0, 1 - k / 0.5) ** 1.6
        d -= 0.03 * g(dz + 0.04 * k, 0.07 * (1 - 0.5 * k)) * taper * smooth(1.0, 1.15, u)
        d -= 0.035 * g(dz + 0.02 * k, 0.14) * smooth(0.85, 1.0, u) * smooth(1.5, 1.05, u)
    lz = -(B + 0.17 + 0.03 * uc)
    d -= 0.007 * g(dz - lz, 0.08) * smooth(0.9, 0.4, abs(u - 0.1))
    return d


def sculpt_eyes(hb, eyes):
    """Sculpt the lids and folds into the head mesh (see eye_relief): the faces round each eye are first cut
    finer, so the folds, a few hundredths of a unit across, have the vertices to hold them."""
    # one patch over the whole middle of the face, from above the brows to below the nose, so its edge (where the
    # finer mesh meets the coarser) is nowhere near the eyes or the nose
    ez = eyes[0][1]
    region = [f for f in hb.faces
              if abs(f.calc_center_median().x) < 3.0 and ez - 1.75 < f.calc_center_median().z < ez + 1.35
              and f.calc_center_median().y < 0]
    inside = set(region)
    edges = list({e for f in region for e in f.edges})
    # the patch's edge: new points along it stay on the straight edges the coarser faces beyond still share
    rim = [(e.verts[0].co.copy(), e.verts[1].co.copy()) for e in edges if any(lf not in inside for lf in e.link_faces)]
    ra = np.array([a for a, b in rim]); rb = np.array([b for a, b in rim])
    res = bmesh.ops.subdivide_edges(hb, edges=edges, cuts=10, use_grid_fill=True, smooth=0.0)
    new = [v for v in res['geom'] if isinstance(v, bmesh.types.BMVert) and v.co.y < 0]
    P = np.array([v.co for v in new])
    rim_d = np.full(len(new), 1e9)
    for a_, b_ in zip(ra, rb):
        ab = b_ - a_
        t = np.clip(((P - a_) @ ab) / (ab @ ab), 0, 1)
        rim_d = np.minimum(rim_d, np.linalg.norm(P - (a_ + t[:, None] * ab), axis=1))
    # the rest lie on the old flat faces: put each back onto the head's own smooth surface before sculpting, easing
    # from the flat faces at the patch's edge to the smooth surface a little way in, so no step shows where the
    # finer mesh meets the coarser (a hard switch drew a line across the nose)
    for v, r in zip(new, rim_d):
        w = smooth(0.0, 0.9, float(r))
        if w <= 0:
            continue
        y = head_front_y(v.co.x / HEAD_H, v.co.z / HEAD_H)
        if y is not None:
            v.co.y += w * (y * HEAD_H - v.co.y)
    for v in hb.verts:
        if v.co.y >= 0:
            continue
        d = sum(eye_relief(v.co.x, v.co.z, s, ex, ez) for ex, ez, s in eyes)
        if d:
            v.co.y -= d


def build_eye(head, head_tree, s, x, ey, ez):
    """One eye, set into the socket of the sculpted head (head coordinates; s is the side, x the eye's centre
    across, ey the face's surface there, ez its height). Sized from life: the eye is about a fifth of the face's
    width and the iris a little over a third of the eye. The parts: the ball, shaped to the lids' opening and a
    little larger, painted with its shading and veins (sclera_image); an iris looking straight ahead, laid onto
    the ball with a slight glossy swell and the pupil and its shading painted on (iris_colours), so the sun leaves
    a real highlight on it; and an arched brow, full at the nose and tapering to a fine tail. The lids and the
    folds round the eye are sculpted into the head (eye_relief). At sprite size this paints a white-iris-white
    row."""
    RX, RY, RZ = EYE_RX, 0.36, EYE_RZ
    TILT = EYE_TILT
    cy = ey + 0.24                                  # the white's front stands 0.12 proud of the skin
    white = almond(RX, RY, RZ, s)
    bmesh.ops.rotate(white, cent=(0, 0, 0), matrix=Euler((0, -s * TILT, 0)).to_matrix(), verts=white.verts)
    bmesh.ops.translate(white, vec=(x, cy, ez), verts=white.verts)
    # bend the white round the face, so it stands the same height proud of the skin all round its edge: a straight
    # one sank under the socket wall at the inner corner (pushing the iris over to the nose side at sprite size) and
    # under the rim below, and stood out past the lids at the outer corner, where the face turns back
    def skin_y(px, pz):
        h = head_tree.ray_cast(Vector((px, -50, pz)), Vector((0, 1, 0)))
        return h[0].y if h[0] is not None else ey
    for v in white.verts:
        v.co.y += skin_y(v.co.x, v.co.z) - ey
    # (the skin is taken back inside the opening, see eye_relief, so the ball is followed against its plain
    # surface; past the lids' edges the ball turns back under the skin, so only the opening shows white)
    for v in white.verts:
        uu_ = (v.co.x - x) * s / RX
        dz_ = v.co.z - ez - uu_ * RX * TILT
        lim = lid_top(uu_) if dz_ > 0 else lid_bottom(uu_)
        over = abs(dz_) / max(lim, 1e-3)
        # (toward the lids the ball is kept full, so its edge meets the lids' margins rather than falling away
        # from them into a dark gap, and then turns back under them)
        v.co.y -= 0.07 * min(1.0, over) ** 2      # (a smooth swell with no onset: a smoothstep left a kink the gloss drew as a crease)
        v.co.y += 1.1 * smooth(0.97, 1.06, over) ** 1.5
    wtree = bvh_of(white)
    # texture coordinates for the painted white (sclera_image): across from the inner corner to the outer one,
    # and up from the centre line, so one painting serves both eyes
    uvl = white.loops.layers.uv.new('UVMap')
    for f in white.faces:
        for lp in f.loops:
            co = lp.vert.co
            u = (co.x - x) * s / RX
            lp[uvl].uv = ((u + 1) / 2, ((co.z - ez - u * RX * TILT) / (RZ * EYE_K) + 1) / 2)
    wo = mesh_obj(f'Eye white {s}', white, M_SCLERA, head, subdiv=0)
    # toward each corner the ball turns back under the skin, so the lids lap over its tips and the white narrows
    # into the corner and passes beneath it, instead of ending at a cut edge level with the skin. (A shape key:
    # the sprite sheet turns it off, since at sprite size the tips are the white pixel beside the iris.)
    wo.shape_key_add(name='Basis')
    tk = wo.shape_key_add(name='tuck', from_mix=False)
    for kv in tk.data:
        kv.co.y += 0.05 * smooth(0.8, 1.0, abs(kv.co.x - x) / RX) ** 1.5
    tk.value = 1.0
    # for the sprite sheet only: the white drawn a little larger round its own centre, so a pixel of it still
    # shows either side of the iris at that size
    sp = wo.shape_key_add(name='sprite', from_mix=False)
    for kv in sp.data:
        kv.co.x = x + (kv.co.x - x) * 1.12
        kv.co.z = ez + (kv.co.z - ez) * 1.0
    sp.value = 0.0

    def front(px, pz):
        """The nearest surface (the face or the white) in front of (px, pz), seen from the camera."""
        best = 1e9
        h = head_tree.ray_cast(Vector((px, -50, pz)), Vector((0, 1, 0)))
        if h[0] is not None:
            best = h[0].y
        h = wtree.ray_cast(Vector((px, -50, pz)), Vector((0, 1, 0)))
        if h[0] is not None:
            best = min(best, h[0].y)
        return best

    def lid_z(u, top):
        """The edge of the eye's opening at u (-1 inner corner .. 1 outer corner across), upper or lower."""
        return ez + (lid_top(u) if top else -lid_bottom(u)) + u * RX * TILT

    # the iris lies on the curve of the white, turned to face the way it faces there (the white bends with the
    # face, so a disc square to the camera sank into it on the nose side and read as a lens with the pupil off
    # centre)
    IR = 0.31                                       # its top under the upper lid, its foot on the lower one
    iz = ez - EYE_LIFT + 0.06
    hits = [wtree.ray_cast(Vector((x + dx, -50, iz + dz)), Vector((0, 1, 0)))
            for dx, dz in ((0, 0), (0.2, 0), (-0.2, 0), (0, 0.15), (0, -0.15))]
    n = sum((h[1] for h in hits if h[0] is not None), Vector()).normalized()
    c = hits[0][0]
    turn = Vector((0, -1, 0)).rotation_difference(n)
    iris = iris_dome(IR, 0.05)
    disc = [(v.co.x, v.co.z) for v in iris.verts]   # (where each point lies on the flat disc, for its painting)
    bmesh.ops.rotate(iris, cent=(0, 0, 0), matrix=turn.to_matrix(), verts=iris.verts)
    # laid onto the ball itself: each point of the iris sits just over the white's surface, with only the slight
    # swell of the cornea in the middle, so it never stands out from the eye (or is cut by it) at any angle
    iris.verts.index_update()
    spr_y = [0.0] * len(iris.verts)
    for v in iris.verts:
        t = min(1.0, math.hypot(v.co.x, v.co.z) / IR)
        # (where the iris runs past the white, under a lid, it is sent back behind the skin, so the lid covers its
        # edge cleanly instead of the iris climbing the lid as a dark wall)
        h = wtree.ray_cast(Vector((c.x + v.co.x, -50, c.z + v.co.z)), Vector((0, 1, 0)))
        if h[0] is not None:
            # a stiff disc where the ball is open to view (part flat, part following the ball: on the ball alone
            # its lower rim turned down steeply and read as a dark crescent), easing to follow the ball exactly
            # where the ball turns back under the lids, so it passes under them with the white and stands out
            # nowhere (pushed back on its own it left walls and a flat-cut edge in the 3/4 views)
            px_, pz_ = c.x + v.co.x, c.z + v.co.z
            uu_ = (px_ - x) * s / RX
            dz_ = pz_ - ez - uu_ * RX * TILT
            lim = lid_top(uu_) if dz_ > 0 else lid_bottom(uu_)
            wf = 0.6 * smooth(0.95, 0.8, abs(dz_) / max(lim, 1e-3))
            flat = -(n.x * v.co.x + n.z * v.co.z) / n.y
            surf = h[0].y - c.y
            ov = abs(dz_) / max(lim, 1e-3)
            lift = (0.04 + 0.05 * (1 - t * t)) * smooth(1.0, 0.86, ov) - 0.04 * smooth(0.94, 1.0, ov)
            v.co.y = min(wf * flat + (1 - wf) * surf, surf) - lift
            # (for the sprite sheet: the iris following the ball more closely, which keeps its pixel showing in
            # the side views; a shape key, see sprite_mode)
            spr_y[v.index] = 0.75 * surf + 0.25 * min(flat, surf) - (0.006 + 0.07 * (1 - t * t))
        else:
            v.co.y = 0.3
            spr_y[v.index] = 0.3
    o = mesh_obj(f'Iris {s}', iris, M_EYE, head, loc=c, subdiv=0)
    o.shape_key_add(name='Basis')
    sk = o.shape_key_add(name='sprite', from_mix=False)
    for kv, y in zip(sk.data, spr_y):
        kv.co.y = y
    sk.value = 0.0
    o.visible_shadow = False        # (it lies on the white; its shadow fell as a grey column down to the lower lid)
    iris_colours(o.data, disc, IR, s)

    def on_face(us, radii, zf, lift, span=RX, n=36):
        """A smooth curve over the face through the heights zf(u) (resampled finely, so a tube along it is
        round rather than a chain of straight pieces), with its radii eased between the given ones."""
        fine = np.linspace(us[0], us[-1], n)
        pts = []
        for u in fine:
            px = x + s * u * span
            pz = zf(u)
            pts.append((px, front(px, pz) - lift, pz))
        return pts, list(np.interp(fine, us, radii))

    # (the lids, their creases and the folds under the eye are sculpted into the skin itself: see eye_relief)
    # the brow: on the brow ridge, starting just outside the bridge of the nose, rising in a soft arch to its peak
    # two thirds of the way out and falling away to a fine tail past the outer corner
    # (for the sprites only: in the 3-D model the brow is painted hair by hair, see brow_image; the solid tube
    # reads better than the painting at a pixel or two)
    bus = list(np.linspace(BROW_US[0], BROW_US[-1], 36))
    brow, br = on_face(bus, [brow_line(u)[1] for u in bus], lambda u: ez - EYE_LIFT + brow_line(u)[0], 0.03, span=1.1)
    bo = mesh_obj(f'Brow tube {s}', tube(brow, br, flat=0.4, n=12), M_BROW, head, subdiv=0)
    bo.hide_render = True
    bo.hide_viewport = True


def _perlin2(x, y, seed=0):
    """Gradient (Perlin) noise in 2-D, about -1..1, vectorised over numpy arrays: a random unit gradient at each
    lattice point, dotted with the offset to it and blended with the quintic fade 6t^5 - 15t^4 + 10t^3 (Perlin's
    improved noise, 2002). Unlike value noise it has no blocky lattice artefacts and its features are all one size,
    which is what fBm and domain warping need."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xi, yi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    xf, yf = x - xi, y - yi

    def grad(ix, iy):
        h = (ix * 374761393 + iy * 668265263 + seed * 1442695041) & 0xFFFFFFFF
        h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
        a = (h & 0xFFFF) / 65536.0 * 2 * np.pi
        return np.cos(a), np.sin(a)

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)
    n = []
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        gx, gy = grad(xi + dx, yi + dy)
        n.append(gx * (xf - dx) + gy * (yf - dy))
    u, v = fade(xf), fade(yf)
    return 1.414 * ((n[0] * (1 - u) + n[1] * u) * (1 - v) + (n[2] * (1 - u) + n[3] * u) * v)


def _fbm2(x, y, seed=0, octaves=5, lac=2.0, gain=0.5):
    """Fractal Brownian motion: octaves of gradient noise, each at twice the frequency and half the amplitude."""
    tot, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        tot = tot + amp * _perlin2(x * lac ** o, y * lac ** o, seed + 17 * o)
        norm += amp
        amp *= gain
    return tot / norm


def _warped_fbm(x, y, seed=0, k=3.0):
    """Domain-warped fBm (Quilez, "Domain warping"): fBm sampled at a point pushed about by two more fBm fields,
    f(p + k q(p)), q = (fbm(p), fbm(p + offset)). The warp drags the noise into the swirled, filamentary shapes of
    smoke instead of the even blobs of plain fBm."""
    qx = _fbm2(x, y, seed + 1, 4)
    qy = _fbm2(x + 5.2, y + 1.3, seed + 2, 4)
    return _fbm2(x + k * qx, y + k * qy, seed + 3, 5)


SHADOW_TEX = {}
SH_U0, SH_U1, SH_V0, SH_V1 = -1.75, 1.5, -0.45, 0.95      # the patch's extent: across (u) and up from the line between the corners


def eyeshadow_image():
    """The eyeshadow, painted into a texture laid over the lid (u across, dz up from the corner line, the same
    mapping as the lids' relief): a smoky, desaturated grey-mauve wash from the lid's edge up over the crease, fuller and taller
    toward the outer corner and gone at the inner one, a charcoal smoked into the outer V and along the upper
    lash line, and a fainter, lighter wash along the lower lid. Colour and opacity in one RGBA image,
    so the skin shows through and the wash fades into it instead of ending at an edge."""
    if 'img' in SHADOW_TEX:
        return SHADOW_TEX['img']
    W, H = 640, 300
    uu, dd = np.meshgrid(SH_U0 + (np.arange(W) + 0.5) / W * (SH_U1 - SH_U0), SH_V1 - (np.arange(H) + 0.5) / H * (SH_V1 - SH_V0))

    def sm(e0, e1, x):
        t = np.clip((x - e0) / (e1 - e0), 0, 1)
        return t * t * (3 - 2 * t)
    uc = np.clip(uu, -1, 1)
    T = np.array([lid_top(u) for u in uc[0]])[None, :] + 0 * uu
    B = np.array([lid_bottom(u) for u in uc[0]])[None, :] + 0 * uu
    above = dd - T
    below = -B - dd
    tall = 0.46 + 0.16 * sm(-0.2, 0.9, uu)                    # how far up the wash reaches: taller toward the outer corner
    across = sm(-1.3, -0.7, uu) * (0.85 + 0.15 * sm(-0.2, 0.8, uu)) * sm(1.5, 1.0, uu) * sm(-1.55, -1.3, uu)
    lid = sm(-0.012, 0.012, above) * (1 - sm(0.3 * tall, tall, above))
    # the deeper colour: smoked along the lash line and into the outer V (a wedge opening up and out from the corner)
    deep = sm(0.1, 1.0, uu) * (1 - sm(0.0, 0.5 * tall, above)) * sm(-0.005, 0.02, above)
    vwedge = sm(0.55, 1.05, uu) * (1 - sm(0.3 * tall, 1.1 * tall, np.abs(dd - (T + 0.07 + 0.2 * np.maximum(0, uu - 0.9)))))
    deep = np.maximum(deep * 0.8, 0.6 * vwedge * sm(1.5, 0.95, uu))
    base = np.array([118, 88, 100], dtype=np.float32)
    dark = np.array([78, 62, 76], dtype=np.float32)            # (a soft charcoal-mauve: the corners deepen the tone, never to near-black)
    col = base + (dark - base) * np.clip(deep, 0, 1)[..., None]
    a = 0.8 * lid * across + 0.1 * deep * lid * across
    # under the lower lashes, outer half only
    low = sm(-0.01, 0.015, below) * (1 - sm(0.06, 0.3, below)) * sm(-1.0, -0.5, uu) * sm(1.35, 1.0, uu)
    col = col + (np.array([128, 96, 112], dtype=np.float32) - col) * low[..., None]
    a = np.maximum(a, 0.5 * low * (0.7 + 0.3 * sm(0.0, 0.9, uu)))
    # the hollow beside the nose, where the lid meets the bridge: the inner corner's own soft shade
    hollow = np.exp(-(((uu + 1.02) / 0.26) ** 2 + ((dd - 0.04) / 0.3) ** 2))
    col = col + (base - col) * np.clip(hollow, 0, 1)[..., None] * (1 - low[..., None]) * 0.6
    a = np.maximum(a, 0.2 * hollow)
    # the far ends, past each corner, thin out rather than pooling into dark patches
    a *= 1 - 0.45 * sm(0.85, 1.35, uu) - 0.4 * sm(-0.85, -1.35, uu)
    a *= sm(SH_V0, SH_V0 + 0.15, dd) * sm(SH_V1, SH_V1 - 0.15, dd) * sm(SH_U0, SH_U0 + 0.2, uu) * sm(SH_U1, SH_U1 - 0.2, uu)   # (no hard edge at the patch's border)
    # nothing inside the opening itself
    a *= 1 - ((above < 0) & (below < 0) & (np.abs(uu) < 1.0))
    rgb = np.array([_srgb_lin(c / 255) for c in col.reshape(-1, 3)], dtype=np.float32).reshape(H, W, 3) if False else col / 255
    # smoke: domain-warped gradient fBm in the opacity (see _warped_fbm), so the wash drifts and thins in soft,
    # swirled wisps instead of grading smoothly or breaking into even blobs
    smoke = _warped_fbm(uu * 2.2, dd * 2.6, seed=9, k=2.5)
    a = a * np.clip(0.95 + 0.6 * smoke, 0.55, 1.3)
    img = bpy.data.images.new('eyeshadow', W, H, alpha=True)
    px = np.concatenate([rgb, np.clip(a, 0, 1)[..., None]], axis=2)[::-1]
    img.pixels.foreach_set(px.ravel().astype(np.float32))
    img.pack()
    SHADOW_TEX['img'] = img
    return img


LASH_TEX = {}


def lash_image():
    """The lashes, painted into a texture over the same patch as the eyeshadow (same mapping), grown by noise rather
    than drawn one by one. Along each lid margin, 1-D fractal noise sets the length and the lean of the strands and
    a second, slower noise gathers them into clumps (neighbouring strands pull toward their clump's direction, as
    real lashes stick together); each strand is a curved, tapering stroke that leaves the lid turned outward and curls
    back up, drawn at twice the size and then softened. Over them go a dense fuzz of short, faint, thin strands
    with random directions, a soft halo (the lashes' blur against the skin) and a grain of fine noise in the
    opacity, so the fringe reads as fuzzy hair rather than a row of spikes, and a smoky liner along the lid edge."""
    if 'img' in LASH_TEX:
        return LASH_TEX['img']
    from PIL import Image, ImageDraw, ImageFilter
    W, H, SS = 1024, 600, 2
    rng = np.random.default_rng(5)

    def px(u, dz):
        return ((u - SH_U0) / (SH_U1 - SH_U0) * W * SS, (SH_V1 - dz) / (SH_V1 - SH_V0) * H * SS)

    layer = Image.new('L', (W * SS, H * SS), 0)          # opacity of the strands
    dr = ImageDraw.Draw(layer)
    cover = Image.new('L', (W * SS, H * SS), 0)          # where a strand is, at full value (to unpremultiply the colour)
    dc = ImageDraw.Draw(cover)
    paint = Image.new('RGB', (W * SS, H * SS), (0, 0, 0))   # each strand's own colour along its length
    dp = ImageDraw.Draw(paint)
    fuzz = Image.new('L', (W * SS, H * SS), 0)
    fd = ImageDraw.Draw(fuzz)
    ROOT = np.array([16, 10, 11], dtype=np.float32)      # the lash at the lid: near-black
    TIP = np.array([58, 40, 34], dtype=np.float32)       # toward the tip it thins and lets light through: a warm brown
    SHEEN = np.array([84, 72, 70], dtype=np.float32)  # the soft highlight where the curve faces the light

    def stroke(u0, z0, ang, length, curl, wid, top, val, steps=14, tone=0.0, shade=True, d=None):
        """One tapering curved stroke: starts at (u0, z0), leaves at ang off straight up (down for a lower
        lash; positive leans toward the outer corner), curls back toward upright by `curl`, and is nudged along
        its length by curl noise (the curl of a gradient-noise potential: a divergence-free flow, so neighbouring
        strands bend together without bunching or crossing). Shaded root to tip, with a sheen part-way along."""
        sign = 1.0 if top else -1.0
        u, z = u0, z0
        p = px(u, z)
        for k in range(steps):
            t = (k + 1) / steps
            a = ang - curl * t if top else ang + 0.25 * curl * t
            e = 0.01
            pu = (_perlin2((u) * 7, (z + e) * 7, 77) - _perlin2(u * 7, (z - e) * 7, 77)) / (2 * e)
            pz = -(_perlin2((u + e) * 7, z * 7, 77) - _perlin2((u - e) * 7, z * 7, 77)) / (2 * e)
            du = math.sin(a) + 0.012 * float(pu) * t
            dz_ = sign * math.cos(a) + 0.012 * float(pz) * t
            n_ = math.hypot(du, dz_) or 1.0
            u += du / n_ * length / steps
            z += dz_ / n_ * length / steps
            q = px(u, z)
            w = max(1, int(wid * (1 - 0.82 * t) * SS))
            (d or dr).line([p, q], fill=val, width=w)
            if shade:
                c = ROOT + (TIP - ROOT) * min(1.0, (t ** 1.8) * (0.7 + 0.5 * tone))
                c = c + (SHEEN - c) * 0.3 * math.exp(-((t - 0.42) / 0.13) ** 2) * (0.5 + 0.5 * tone)
                dc.line([p, q], fill=255, width=w)
                dp.line([p, q], fill=tuple(int(v) for v in c), width=w)
            p = q

    # the roots: blue-noise positions along each lid (1-D Poisson-disk sampling by dart throwing, Bridson-style:
    # no two closer than a minimum spacing), so the lashes are evenly irregular instead of clumping and gapping at
    # random; then gradient-noise fBm along the lid sets each one's length and lean, and a slower one gathers
    # neighbours into clumps that lean together
    def poisson1(n, lo, hi, seed):
        r_ = np.random.default_rng(seed)
        dmin = (hi - lo) / n * 0.6
        pts_, tries = [], 0
        while len(pts_) < n and tries < n * 60:
            c_ = r_.uniform(lo, hi)
            tries += 1
            if all(abs(c_ - q_) >= dmin for q_ in pts_):
                pts_.append(c_)
        return sorted(pts_)
    for top, count in ((True, 110), (False, 55)):
        roots = poisson1(count, -0.88, 1.07, 3) if top else poisson1(count, -0.3, 1.05, 4)
        for u0 in roots:
            edge = lid_top(u0) if top else -lid_bottom(u0)
            outer = smooth(-0.9, 1.0, u0)
            len_n = float(_fbm2(u0 * 3.0, 0.5, 21, 4))
            lean_n = float(_fbm2(u0 * 2.4, 4.5, 31, 4))
            clump = float(_perlin2(u0 * 5.0, 9.5, 41))
            tone = 0.5 + 0.5 * float(_perlin2(u0 * 9.0, 2.5, 51))
            length = ((0.07 + 0.12 * outer ** 1.2 - 0.05 * smooth(0.95, 1.1, u0)) * (1 + 0.35 * len_n) * (0.85 + 0.3 * rng.random())) * (1.0 if top else 0.45)
            lean = (-0.05 + 0.85 * outer ** 1.3 + 0.35 * lean_n + 0.25 * clump) + 0.08 * (rng.random() - 0.5)
            stroke(u0, edge - (0.005 if top else -0.005), lean, length, 0.12 + 0.12 * rng.random(),
                   0.0075 / (SH_U1 - SH_U0) * W if top else 0.005 / (SH_U1 - SH_U0) * W, top, int(200 + 55 * rng.random()), tone=tone)
    # the fuzz: many short faint thin strands in every direction near the margins
    for top in (True, False):
        for _ in range(260 if top else 110):
            u0 = (-0.9 + 2.0 * rng.random()) if top else (-0.35 + 1.45 * rng.random())
            edge = lid_top(u0) if top else -lid_bottom(u0)
            ang = rng.normal(0.35 * smooth(-0.9, 1.0, u0), 0.55)
            stroke(u0, edge + (rng.random() - 0.3) * 0.02 * (1 if top else -1), ang, (0.02 + 0.05 * rng.random()) * (1 if top else 0.6), 0.5, 0.003 / (SH_U1 - SH_U0) * W, top, int(70 + 90 * rng.random()), steps=6, shade=False, d=fd)
    # the smoky liner hugging the lid's edge
    pts = [px(u, lid_top(u) + 0.006) for u in np.linspace(-1.0, 1.08, 90)]
    dr.line(pts, fill=150, width=int(0.016 / (SH_U1 - SH_U0) * W * SS), joint='curve')
    pts = [px(u, -lid_bottom(u) - 0.004) for u in np.linspace(-0.3, 1.0, 60)]
    dr.line(pts, fill=70, width=int(0.008 / (SH_U1 - SH_U0) * W * SS), joint='curve')
    a1 = np.asarray(layer.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.1)), dtype=np.float32) / 255
    a2 = np.asarray(fuzz.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.3)), dtype=np.float32) / 255
    halo = np.asarray(layer.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(7)), dtype=np.float32) / 255
    yy, xx = np.mgrid[0:H, 0:W]
    grain = 0.8 + 0.2 * _fbm2(xx / 6.0, yy / 6.0, 61, 3)       # fine gradient-noise grain in the opacity
    a = np.clip(a1 * 1.5 + a2 * 0.35 * grain + halo * 0.12, 0, 1)
    a *= 0.85 + 0.15 * grain
    # the colour: each strand's painted shading (unpremultiplied by its coverage), dark where only fuzz, halo or
    # liner lie
    cov = np.asarray(cover.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.6)), dtype=np.float32) / 255
    pnt = np.asarray(paint.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.6)), dtype=np.float32)
    strand = pnt / np.maximum(cov, 1e-3)[..., None]
    w_ = np.clip(cov * 2.5, 0, 1)[..., None]
    col = (ROOT + (np.clip(strand, 0, 255) - ROOT) * w_) / 255
    img = bpy.data.images.new('lashes', W, H, alpha=True)
    pxl = np.concatenate([col, np.clip(a, 0, 1)[..., None]], axis=2)[::-1]
    img.pixels.foreach_set(pxl.ravel().astype(np.float32))
    img.pack()
    LASH_TEX['img'] = img
    return img


BROW_TEX = {}
BR_U0, BR_U1, BR_V0, BR_V1 = -1.28, 2.25, 0.62, 1.92     # the brow patch's extent, in the eye patch's units (u, dz)
BROW_US = [-0.7, 1.08]          # the brow's inner and outer ends (brow units)


def brow_line(ub):
    """The brow's centre height above the eye's centre and its half-height, at ub (in the brow's own units: the
    face's x from the eye's centre over 1.1). Both are single smooth closed forms, with no table to interpolate
    (a piecewise-linear one showed its corners as kinks along the brow's edges): the centre is a skewed sine arch,
    s^1.45 pushing its peak about 60% of the way out, so it rises gently from the nose and falls more steeply
    into the tail; the half-height is full at the head, swells a little just past it and thins smoothly to a fine
    tail."""
    u = min(max(ub, BROW_US[0]), BROW_US[-1])
    t = (u - BROW_US[0]) / (BROW_US[-1] - BROW_US[0])
    zc = 0.93 + 0.35 * math.sin(math.pi * t ** 1.45) - 0.05 * t
    rise = math.sin(0.5 * math.pi * min(1.0, t / 0.28))
    r = 0.016 + 0.17 * (1 - t) ** 0.65 * (0.72 + 0.28 * rise)
    return zc, r


def brow_image():
    """The brow, painted hair by hair into a texture over its patch (u across in eye widths, dz up from the line
    between the eye's corners). How a brow grows (as brow artists draw it): at the head, by the nose, the hairs
    stand up and fan a little outward; through the body the lower hairs angle up and out and the upper ones down
    and out, meeting in a soft ridge just above the middle (the herringbone that gives a brow its shape); the tail
    lies flat along the arch. The head is feathered and sparse, the lower edge of the body dense and crisp, the
    top soft. The noise, as for the lashes: hair roots scattered with 2-D Poisson-disk sampling (Bridson's dart
    throwing, so the hairs are evenly irregular), gradient-noise fBm for each hair's length, angle and tone, curl
    noise bending neighbouring hairs together, and each hair shaded dark at the root to a lighter, warmer tip
    with a sheen part-way along. Under the hairs, a sparse powder of skin-toned brown smoked with domain-warped
    fBm, so the skin between hairs doesn't read as gaps."""
    if 'img' in BROW_TEX:
        return BROW_TEX['img']
    from PIL import Image, ImageDraw, ImageFilter
    W, H, SS = 1200, 420, 2
    rng = np.random.default_rng(17)
    K = EYE_RX / 1.1                    # eye-patch u to brow ub

    def centre(u):
        zc, r = brow_line(u * K)
        return zc - EYE_LIFT - u * EYE_RX * EYE_TILT, r

    def px(u, dz):
        return ((u - BR_U0) / (BR_U1 - BR_U0) * W * SS, (BR_V1 - dz) / (BR_V1 - BR_V0) * H * SS)

    U_IN, U_OUT = BROW_US[0] / K, BROW_US[-1] / K
    cover = Image.new('L', (W * SS, H * SS), 0)
    dc = ImageDraw.Draw(cover)
    alpha = Image.new('L', (W * SS, H * SS), 0)
    da = ImageDraw.Draw(alpha)
    paint = Image.new('RGB', (W * SS, H * SS), (0, 0, 0))
    dp = ImageDraw.Draw(paint)
    ROOT = np.array([40, 24, 17], dtype=np.float32)
    TIP = np.array([116, 66, 36], dtype=np.float32)
    SHEEN = np.array([156, 104, 68], dtype=np.float32)

    def slope(u):
        e = 0.02
        return (centre(u + e)[0] - centre(u - e)[0]) / (2 * e)

    def flow(u, dz):
        """The direction a hair grows at (u, dz), as an angle from straight outward (positive: up)."""
        zc, r = centre(u)
        ub = u * K
        tang = math.atan(slope(u))
        side = (dz - (zc + 0.15 * r)) / max(r, 1e-3)              # -1 at the lower edge, +1 at the upper
        body = tang + math.radians(24) * smooth(0.2, -0.9, side) - math.radians(16) * smooth(-0.1, 0.9, side)
        head = math.radians(82 - 10 * smooth(-0.7, -0.35, ub))
        w = smooth(-0.3, -0.55, ub)
        tail = smooth(0.7, 1.0, ub)
        a = body * (1 - w) + head * w
        return a * (1 - tail) + tang * tail

    # roots: 2-D Poisson-disk sampling over the brow's shape (dart throwing on a cell grid)
    dmin = 0.022
    cell = dmin / math.sqrt(2)
    grid = {}
    roots = []
    tries = 0
    while tries < 60000 and len(roots) < 1400:
        tries += 1
        u = rng.uniform(U_IN - 0.05, U_OUT)
        zc, r = centre(u)
        dz = zc + rng.uniform(-1.0, 1.0) * r * 1.05
        # the head is sparse (feathered), the tail thins out
        ub = u * K
        side0 = (dz - zc) / max(r, 1e-3)
        # a rounded, feathered head: the top of the head starts further out than its lower edge
        keep = (0.3 + 0.7 * smooth(-0.72, -0.3, ub)) * smooth(-0.74, -0.55, ub - 0.12 * max(0.0, side0) ** 2)
        keep *= 1 - 0.5 * smooth(0.75, 1.08, ub)
        if rng.random() > keep:
            continue
        gi, gj = int(u / cell), int(dz / cell)
        ok = True
        for di in (-2, -1, 0, 1, 2):
            for dj in (-2, -1, 0, 1, 2):
                q = grid.get((gi + di, gj + dj))
                if q is not None and (q[0] - u) ** 2 + (q[1] - dz) ** 2 < dmin * dmin:
                    ok = False
        if ok:
            grid[(gi, gj)] = (u, dz)
            roots.append((u, dz))
    # draw from the top of the brow down, so the dense lower hairs lie over the upper ones
    roots.sort(key=lambda p: -p[1])
    for u0, z0 in roots:
        zc, r = centre(u0)
        ub = u0 * K
        side = (z0 - zc) / max(r, 1e-3)
        len_n = float(_fbm2(u0 * 4.0, z0 * 4.0, 71, 4))
        ang_n = float(_fbm2(u0 * 3.0, z0 * 3.0 + 7.0, 81, 4))
        tone = 0.5 + 0.5 * float(_perlin2(u0 * 11.0, z0 * 11.0, 91))
        length = (0.11 + 0.05 * smooth(-0.6, 0.3, ub) - 0.05 * smooth(0.65, 1.08, ub)) * (1 + 0.3 * len_n) * (0.85 + 0.3 * rng.random())
        if ub < -0.35:
            length *= 0.8
        wid = 0.0078 * (1 - 0.35 * smooth(0.65, 1.08, ub))
        # opacity: crisp and full at the lower edge of the body, softer at the top, feathered at the head
        val = 255 * (0.7 + 0.3 * smooth(0.8, -0.4, side)) * (0.55 + 0.45 * smooth(-0.75, -0.35, ub))
        steps = 12
        u, z = u0, z0
        p = px(u, z)
        jitter = 0.18 * ang_n + 0.08 * (rng.random() - 0.5)
        for k in range(steps):
            t = (k + 1) / steps
            a = flow(u, z) + jitter
            e = 0.01
            pu = (_perlin2(u * 6, (z + e) * 6, 101) - _perlin2(u * 6, (z - e) * 6, 101)) / (2 * e)
            pz = -(_perlin2((u + e) * 6, z * 6, 101) - _perlin2((u - e) * 6, z * 6, 101)) / (2 * e)
            du = math.cos(a) + 0.01 * float(pu) * t
            dz_ = math.sin(a) + 0.01 * float(pz) * t
            n_ = math.hypot(du, dz_) or 1.0
            u += du / n_ * length / steps
            z += dz_ / n_ * length / steps
            q = px(u, z)
            w = max(1, int(wid / (BR_U1 - BR_U0) * W * SS * (1 - 0.75 * t)))
            c = ROOT + (TIP - ROOT) * min(1.0, t ** 1.5 * (0.65 + 0.6 * tone))
            # the sheen: where the hair turns toward the light, along the upper middle of the brow
            lit = 0.35 * math.exp(-((t - 0.5) / 0.18) ** 2) * (0.4 + 0.6 * smooth(-0.6, 0.6, side)) * (0.5 + 0.5 * tone)
            c = c + (SHEEN - c) * lit
            # the lower hairs and the head sit in the shadow of the brow bone's turn: a touch darker
            c = c * (0.86 + 0.14 * smooth(-1.0, 0.2, side))
            dc.line([p, q], fill=255, width=w)
            da.line([p, q], fill=int(val * (1 - 0.45 * t)), width=w)
            dp.line([p, q], fill=tuple(int(v) for v in c), width=w)
            p = q
    a_h = np.asarray(alpha.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.7)), dtype=np.float32) / 255
    cov = np.asarray(cover.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.7)), dtype=np.float32) / 255
    pnt = np.asarray(paint.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.7)), dtype=np.float32)
    strand = np.clip(pnt / np.maximum(cov, 1e-3)[..., None], 0, 255)
    # the powder under the hairs: the brow's shape, soft at the top and the head, smoked with warped fBm
    uu, dd = np.meshgrid(BR_U0 + (np.arange(W) + 0.5) / W * (BR_U1 - BR_U0), BR_V1 - (np.arange(H) + 0.5) / H * (BR_V1 - BR_V0))
    zc_row = np.array([centre(u)[0] for u in uu[0]])[None, :]
    r_row = np.array([centre(u)[1] for u in uu[0]])[None, :]
    sd = (dd - zc_row) / np.maximum(r_row, 1e-3)
    ubb = uu * K

    def sm(e0, e1, x):
        t = np.clip((x - e0) / (e1 - e0), 0, 1)
        return t * t * (3 - 2 * t)
    shape = sm(-1.15, -0.8, sd) * sm(1.2, 0.6, sd) * sm(BROW_US[0] - 0.08, BROW_US[0] + 0.25, ubb) * sm(BROW_US[-1] + 0.05, BROW_US[-1] - 0.25, ubb)
    smoke = _warped_fbm(uu * 3.0, dd * 3.0, seed=111, k=2.0)
    powder = shape * (0.5 + 0.15 * smoke) * sm(-0.85, -0.55, ubb - 0.12 * np.clip(sd, 0, None) ** 2) * (0.6 + 0.4 * sm(-0.75, -0.3, ubb))
    grain = 0.85 + 0.15 * _fbm2(np.arange(W)[None, :] / 5.0 + 0 * dd, np.arange(H)[:, None] / 5.0 + 0 * uu, 121, 3)
    a = np.clip(1 - (1 - np.clip(a_h * 1.6 * grain, 0, 1)) * (1 - powder), 0, 1)
    POWDER = np.array([104, 62, 38], dtype=np.float32)
    w_ = np.clip(cov * 2.2, 0, 1)[..., None]
    col = (POWDER + (strand - POWDER) * w_) / 255
    img = bpy.data.images.new('brows', W, H, alpha=True)
    pxl = np.concatenate([col, a[..., None]], axis=2)[::-1]
    img.pixels.foreach_set(pxl.ravel().astype(np.float32))
    img.pack()
    BROW_TEX['img'] = img
    return img


def eye_makeup(head, tree, eyes):
    """Eyeshadow and lashes for both eyes, on the finished head (tree: its surface after the lids are sculpted).
    Each is a thin skin of the face itself (a grid of the lid's area, each point cast onto the head and lifted a
    hair), the shadow carrying the painting from eyeshadow_image and the lashes, a hair above it, the fuzzy
    fringe from lash_image."""
    def material(name, image):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new('ShaderNodeOutputMaterial')
        p = nt.nodes.new('ShaderNodeBsdfPrincipled')
        tex = nt.nodes.new('ShaderNodeTexImage')
        tex.image = image
        tex.interpolation = 'Cubic'
        tex.extension = 'EXTEND'
        nt.links.new(tex.outputs['Color'], p.inputs['Base Color'])
        nt.links.new(tex.outputs['Alpha'], p.inputs['Alpha'])
        p.inputs['Roughness'].default_value = 0.55 if name == 'eyeshadow' else 0.8
        if name == 'brow paint':
            p.inputs['Specular IOR Level'].default_value = 0.2     # hair is matte here: a full sheen turned it grey
        nt.links.new(p.outputs['BSDF'], out.inputs['Surface'])
        m.surface_render_method = 'BLENDED'
        return m

    shadow_m = material('eyeshadow', eyeshadow_image())
    lash_m = material('lash paint', lash_image())
    brow_m = material('brow paint', brow_image())

    def surf(px, pz):
        h = tree.ray_cast(Vector((px, -50, pz)), Vector((0, 1, 0)))
        return h[0].y if h[0] is not None else None

    def patch(name, ex, ez, s, off, m, NU=150, NV=84, ext=None):
        U0, U1, V0, V1 = ext or (SH_U0, SH_U1, SH_V0, SH_V1)
        bm = bmesh.new()
        uvl = bm.loops.layers.uv.new('UVMap')
        grid = []
        for j in range(NV + 1):
            row = []
            for i in range(NU + 1):
                u = U0 + (U1 - U0) * i / NU
                dz = V0 + (V1 - V0) * j / NV
                px, pz = ex + s * u * EYE_RX, ez + u * EYE_RX * EYE_TILT + dz
                y = surf(px, pz)
                row.append(bm.verts.new((px, (y if y is not None else 0.0) - off, pz)))
            grid.append(row)
        for j in range(NV):
            for i in range(NU):
                vs = [grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]]
                f = bm.faces.new(vs if s > 0 else vs[::-1])
                for lp in f.loops:
                    k = vs.index(lp.vert)
                    lp[uvl].uv = ((i + (1 if k in (1, 2) else 0)) / NU, (j + (1 if k in (2, 3) else 0)) / NV)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        me = bpy.data.meshes.new(f'{name} {s}')
        bm.to_mesh(me)
        bm.free()
        for pl in me.polygons:
            pl.use_smooth = True
        me.materials.append(m)
        o = bpy.data.objects.new(f'{name} {s}', me)
        link(o, head)
        o.visible_shadow = False

    for ex, ez, s in eyes:
        patch('Eyeshadow', ex, ez, s, 0.022, shadow_m)
        patch('Lashes', ex, ez, s, 0.034, lash_m, NU=200, NV=110)
        patch('Brow paint', ex, ez, s, 0.02, brow_m, NU=200, NV=80, ext=(BR_U0, BR_U1, BR_V0, BR_V1))


def build():
    reset()
    sc = bpy.context.scene
    root = empty('Ranger', (0, 0, 0))
    body = empty('Body', (0, 0, 0), root)   # carries the walk's bob

    # ---- torso: the body in its shirt, then a fitted jerkin over it: shaped to the bust, drawn in at the waist,
    # following the hips and flaring a little below them into a short skirt
    zs = [53.0 - 0.5 * i for i in range(50)]          # 53 down to 28.5
    mesh_obj('Shirt', loft_fn(zs, lambda z, p: torso_point(z, p)), M_SHIRT, body)
    J = 0.35       # the jerkin's leather stands this far off the shirt

    def jerkin_pt(z, phi):
        if z >= 32.0:
            x, y = torso_point(z, phi, J)
            return x, y
        # below the widest point of the hips the skirt falls straight on, opening a little
        x, y = torso_point(32.0, phi, J + (32.0 - z) * 0.06)
        return x, y, -0.9 * math.cos(phi) ** 2 * (32.0 - z) / 4.0   # the hem dips at front and back

    jz = [47.6 - 0.4 * i for i in range(47)]          # 47.6 down to 29.2
    jb = loft_fn(jz, jerkin_pt, cap_bot=False)
    vee_pts = []
    for zz, w in ((47.0, 1.25), (45.5, 0.85), (44.0, 0.45), (42.6, 0.0)):
        for xx in ((-w, w) if w else (0.0,)):
            vee_pts.append((xx, front_y(jb, xx, zz) - 0.12, zz))
    lace_y = {z: front_y(jb, 0.0, z) - 0.1 for z in (46.2, 44.9)}
    mesh_obj('Jerkin', jb, M_JERKIN, body, solid=0.3)
    # a darker leather band binds the jerkin's hem
    mesh_obj('Jerkin hem', loft_fn([30.3, 29.2], lambda z, p: jerkin_pt(z, p),
                                   cap_top=False, cap_bot=False), M_BELT, body, solid=0.5, subdiv=0)
    # stitching: a running stitch above the hem band (set into the leather, so it shows where the surface faces
    # the viewer and drops out at the silhouette instead of dotting the outline)
    st = []
    for k in range(42):
        phi = 2 * math.pi * k / 42
        p = jerkin_pt(30.65, phi)
        st.append((*outward(p, 0.2), 30.65 + (p[2] if len(p) > 2 else 0.0)))
    mesh_obj('Jerkin stitching', dots(st), M_STITCH, body, subdiv=0)
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
    mesh_obj('Buckle', box(1.6, 0.6, 1.6), M_STEEL, body, loc=(0, by_ - 0.45, 39.4), subdiv=0)
    mesh_obj('Buckle hole', box(0.75, 0.3, 0.75), M_BELT, body, loc=(0, by_ - 0.7, 39.4), subdiv=0)

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
    head_tree = bvh_of(hb)
    # (the eyes are set against the head before its lids are sculpted, so the balls follow the plain socket and
    # the lids then close over their edges)
    eyes = [(feat[s][0], 0.525 * HEAD_H + EYE_LIFT, s) for s in (-1, 1)]
    for s in (-1, 1):
        x, ey, by = feat[s]
        build_eye(head, head_tree, s, x, ey, 0.525 * HEAD_H + EYE_LIFT)
    sculpt_eyes(hb, eyes)
    final_tree = bvh_of(hb)
    mesh_obj('Head shape', hb, M_SKIN, head, subdiv=0)
    eye_makeup(head, final_tree, eyes)
    mesh_obj('Mouth', ellipsoid(0.55, 0.2, 0.11, 10, 6), M_LIPS, head, loc=(0, mouth_y + 0.05, 0.258 * HEAD_H), subdiv=0)

    # ---- hair: a cap over the skull cut along a sloping hairline, curtain bangs parted at the centre,
    # side locks framing the face, and the long fall down the back
    cap = ellipsoid(0.395 * HEAD_H, 0.48 * HEAD_H, 0.54 * HEAD_H, 32, 20)
    bmesh.ops.translate(cap, vec=(0, 0.06 * HEAD_H, 0.56 * HEAD_H), verts=cap.verts)
    bmesh.ops.bisect_plane(cap, geom=cap.verts[:] + cap.edges[:] + cap.faces[:], plane_co=(0, -2.1, 6.4),
                           plane_no=Vector((0, -0.45, -0.89)).normalized(), clear_outer=True)
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
    for s in (-1, 1):
        bm_, rest_, nrm_ = curtain_bang((head_tree, cap_tree), s)
        bang = mesh_obj(f'Bangs {s}', bm_, M_HAIR, head, rest=rest_, subdiv=0)
        bang.data.normals_split_custom_set_from_vertices(nrm_)
        # (the strands cast no shadow of their own: their dark undersides and the shade they threw on the cap
        # marked the bangs off from the rest of the hair; the cap still shades the forehead)
        bang.visible_shadow = False
        # (kept slim and close in to the fall, so once fused it reads as the front of the side hair, not a tube on it)
        mesh_obj(f'Side lock {s}', loft([(7.0, 0.6, 1.3, s * 2.9, 0.3), (4.5, 0.62, 1.35, s * 3.3, 0.9), (2.0, 0.6, 1.3, s * 3.45, 1.35),
                                         (-1.0, 0.6, 1.25, s * 3.65, 1.5), (-2.0, 0.5, 1.0, s * 3.75, 1.6), (-2.6, 0.25, 0.45, s * 3.8, 1.7)]),
                 M_HAIR, head)
    # the hair over the back of the head, down to the nape; the long hair below is draped once the mantle is on
    # (see drape_fall)
    mesh_obj('Hair fall', loft([(7.5, 3.4, 3.4, 0, 0.75), (4.5, 3.75, 4.05, 0, 0.9), (2.0, 3.75, 2.6, 0, 2.1), (0.5, 3.6, 1.8, 0, 2.8),
                                (-0.4, 3.2, 1.3, 0, 3.0)]), M_HAIR, head)
    HAIR_PARTS[:] = [head, ['Hair cap', 'Side lock -1', 'Side lock 1', 'Hair fall']]

    # ---- mantle: a short shoulder cape, its hem lifting over the arms; a mail trim along the hem and a leaf clasp
    # (eased between a few rings; below the shoulders the wool falls in a few deep, slightly uneven folds that
    # grow toward the hem, and the hem itself waves with them)
    MANTLE_RINGS = [(53.4, 2.3, 2.3, 0.35), (52.4, 3.6, 3.1, 0.35), (51.2, 6.0, 3.9, 0.3), (50.2, 7.6, 4.35, 0.3),
                    (49.0, 8.2, 4.6, 0.3), (47.9, 8.35, 4.7, 0.3)]

    MANTLE_BACK = [(53.4, 2.65), (52.4, 3.3), (51.2, 3.25), (50.2, 3.3), (49.0, 3.4), (47.9, 3.55)]

    def back_y(z):     # the inside of the mantle down the middle of her back, just clear of it
        B = MANTLE_BACK
        for (z0, y0), (z1, y1) in zip(B, B[1:]):
            if z1 <= z <= z0:
                return lerp(y0, y1, (z0 - z) / (z0 - z1))
        return B[0][1] if z > B[0][0] else B[-1][1]

    def mantle_pt(z, phi, off=0.0):
        R = MANTLE_RINGS
        for a_, b_ in zip(R, R[1:]):
            if b_[0] <= z <= a_[0]:
                t = (a_[0] - z) / (a_[0] - b_[0])
                t = (1 - math.cos(math.pi * t)) / 2
                rx, ry, cy = (lerp(p_, q_, t) for p_, q_ in zip(a_[1:], b_[1:]))
                break
        else:
            rx, ry, cy = R[-1][1:]
        # across the back the wool is drawn in close over the shoulder blades, so the hair can hang straight past it
        w = max(0.0, -math.cos(phi)) ** 1.5
        ry = lerp(ry, max(0.0, back_y(z) - cy), w)
        f = min(1.0, max(0.0, (51.4 - z) / (51.4 - 47.9))) ** 1.3
        fold = (0.7 * math.sin(6 * phi + 0.4) + 0.3 * math.sin(10 * phi + 1.3)) * (1 - 0.8 * w)
        k = 1 + 0.085 * f * fold
        hem = (1.4 * math.sin(phi) ** 2 - 0.5 * math.cos(phi)) * min(1.0, max(0.0, (49.6 - z) / 1.7))
        return -(rx + off) * k * math.sin(phi), cy - (ry + off) * k * math.cos(phi), hem + 0.4 * f * math.sin(6 * phi + 0.4)

    mz = [53.4 - 0.3 * i for i in range(19)] + [47.9]
    mesh_obj('Mantle', loft_fn(mz, mantle_pt, n=72, cap_top=False, cap_bot=False), M_MANTLE, body, solid=0.45)
    mesh_obj('Mail trim', loft_fn([48.3, 48.0, 47.75], lambda z, p: mantle_pt(z, p, 0.08), n=72, cap_top=False, cap_bot=False),
             M_MAIL, body, solid=0.25, subdiv=0)
    mesh_obj('Collar', loft([(54.0, 2.0, 2.05, 0, 0.4), (52.8, 2.5, 2.55, 0, 0.35)], cap_top=False, cap_bot=False), M_SHIRT, body, solid=0.3)
    clasp = ellipsoid(0.9, 0.4, 1.3, 12, 8)
    bmesh.ops.rotate(clasp, cent=(0, 0, 0), matrix=Euler((0, 0.6, 0)).to_matrix(), verts=clasp.verts)
    mesh_obj('Leaf clasp', clasp, M_GOLD, body, loc=(-0.7, -4.85, 50.6), subdiv=0)

    # the hair's pieces are fused once the mantle is on, so the fall can be kept clear of it
    fuse_hair(*HAIR_PARTS)

    # ---- the rig: one bone per limb segment, each following its pivot (the pivots carry the pose; see pose()).
    # Each limb is a single mesh weighted across its joints, so elbows and knees bend in one smooth piece.
    build_rig(body)

    def profile(table, z):
        if z >= table[0][0]:
            return table[0][1]
        for (za, ra), (zb, rb) in zip(table, table[1:]):
            if zb <= z <= za:
                t = (za - z) / (za - zb)
                return lerp(ra, rb, (1 - math.cos(math.pi * t)) / 2)
        return table[-1][1]

    # ---- arms: one sleeve from a rounded cap at the shoulder to the wrist, then the shirt cuff and the hand
    # (a fuller upper arm and forearm than before: the shoulder cap, the swell of the upper arm, a slight narrowing
    # at the elbow, the forearm's muscle just below it, tapering to the wrist)
    ARM = [(1.45, 0.3), (1.2, 1.15), (0.75, 1.7), (0.0, 2.0), (-2.0, 2.05), (-5.5, 1.9), (-UPPER, 1.65), (-UPPER - 2.2, 1.78),
           (-UPPER - 5.0, 1.55), (-UPPER - FORE + 0.6, 1.3)]
    for side, s in (('R', 1), ('L', -1)):
        x = -s * SH_X   # the figure's right is -X (she faces -Y)
        sh = empty(f'Shoulder.{side}', (x, 0.2, SH_Z), body)
        el = empty(f'Elbow.{side}', (0, 0, -UPPER), sh)
        wr = empty(f'Wrist.{side}', (0, 0, -FORE), el)
        w = arm_weights(side)
        zs = [1.35 - 0.35 * i for i in range(int((1.35 + UPPER + FORE - 0.6) / 0.35) + 1)]

        def arm_pt(z, phi, s=s):
            r = profile(ARM, z)
            # the elbow: a little point at the back of the joint
            back = 0.25 * math.exp(-((z + UPPER) / 0.9) ** 2) * max(0.0, -math.cos(phi))
            return -r * math.sin(phi), -(r * 1.03 + back * 0) * math.cos(phi) + back
        mesh_obj(f'Sleeve.{side}', loft_fn(zs, arm_pt, n=28), M_SLEEVE, loc=(x, 0.2, SH_Z), skin=w)
        mesh_obj(f'Shirt cuff.{side}', loft([(-UPPER - FORE + 1.0, 1.45, 1.5), (-UPPER - FORE - 0.4, 1.52, 1.57)]), M_SHIRT,
                 loc=(x, 0.2, SH_Z), subdiv=0, skin=w)
        mesh_obj(f'Hand.{side}', hand_mesh(s), M_SKIN, loc=(x, 0.2, SH_Z - UPPER - FORE - 0.1), subdiv=0, skin=w)

    # ---- legs: one trouser leg from the hip to the ankle with a rounded kneecap and a patch over it, soft boots
    # with a folded cuff
    LEG = [(2.0, 3.1), (-1.0, 3.0), (-6.0, 2.55), (-THIGH + 1.2, 2.05), (-THIGH, 2.0), (-THIGH - 1.5, 2.0), (-THIGH - 4.0, 2.05),
           (-THIGH - SHIN, 1.35)]
    for side, s in (('R', 1), ('L', -1)):
        x = -s * LEG_X
        hip = empty(f'Hip.{side}', (x, 0, HIP_Z), body)
        kn = empty(f'Knee.{side}', (0, 0, -THIGH), hip)
        an = empty(f'Ankle.{side}', (0, 0, -SHIN), kn)
        w = leg_weights(side)
        zs = [2.0 - 0.35 * i for i in range(int((2.0 + THIGH + SHIN) / 0.35) + 1)]

        def leg_pt(z, phi):
            r = profile(LEG, z)
            cap = 0.4 * math.exp(-((z + THIGH + 0.3) / 1.1) ** 2) * max(0.0, math.cos(phi)) ** 2   # kneecap
            return -r * math.sin(phi), -(r * 1.04 + cap) * math.cos(phi)
        kz = HIP_Z - THIGH
        mesh_obj(f'Leg.{side}', loft_fn(zs, leg_pt, n=28), M_LEGS, loc=(x, 0, HIP_Z), skin=w)
        mesh_obj(f'Knee patch.{side}', ellipsoid(1.25, 0.45, 1.45, 12, 8), M_KNEE, loc=(x, -2.4, kz - 0.3), subdiv=0, skin=w)
        # the boot shaft from just under the knee; its top folds down into a cuff
        mesh_obj(f'Boot shaft.{side}', loft([(-2.2, 2.3, 2.4), (-6.0, 2.15, 2.25), (-SHIN + 0.5, 1.75, 1.85), (-SHIN - 1.2, 1.8, 1.9)]),
                 M_BOOTS, loc=(x, 0, kz), skin=w)
        mesh_obj(f'Boot cuff.{side}', loft([(-1.6, 2.6, 2.7), (-3.5, 2.5, 2.6, 0, 0, lambda p: 0.5 * math.cos(p))]),
                 M_CUFF, loc=(x, 0, kz), subdiv=1, skin=w)
        # a seam stitched up the back of the boot, from the heel to the cuff
        bst = []
        for z in [-3.9 - 1.0 * i for i in range(int(SHIN - 3.9))]:
            t = (-2.2 - z) / (SHIN - 2.7)
            ry = lerp(2.4, 1.85, min(1.0, max(0.0, t)))
            bst.append((0.0, ry + 0.02, z))
        mesh_obj(f'Boot stitching.{side}', dots(bst, 0.4), M_STITCH, loc=(x, 0, kz), subdiv=0, skin=w)
        az = kz - SHIN
        mesh_obj(f'Foot.{side}', loft([(-0.8, 1.75, 1.95, 0, -0.3), (-2.2, 1.85, 3.3, 0, -1.6), (-2.85, 1.8, 3.4, 0, -1.7)]),
                 M_BOOTS, loc=(x, 0, az), skin=w)
        mesh_obj(f'Sole.{side}', loft([(-2.75, 1.85, 3.45, 0, -1.7), (-ANKLE_Z, 1.8, 3.35, 0, -1.7)]), M_SOLE, loc=(x, 0, az),
                 subdiv=1, skin=w)
    finish_rig()

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
    # the eyes have a sun of their own, the same light from the same way, whose shadows only the eyes cast: the
    # lids' shadow from the low sun fell across the white as a hard brown band (the lids are sculpted into the
    # head, so at this light it read as a stain), and the main sun leaves the eyes alone
    eyes_ = [o for o in bpy.data.objects if o.name.startswith(('Eye white', 'Iris'))]
    if eyes_:
        only = bpy.data.collections.new('eyes lit')
        for o in eyes_:
            only.objects.link(o)
        skip = bpy.data.collections.new('eyes not lit')
        for o in eyes_:
            skip.objects.link(o)
        for co in skip.collection_objects:
            co.light_linking.link_state = 'EXCLUDE'
        sun.light_linking.receiver_collection = skip
        esun = link(bpy.data.objects.new('Eye sun', sun_data.copy()))
        esun.data.name = 'Eye sun'
        esun.rotation_euler = sun.rotation_euler.copy()
        esun.light_linking.receiver_collection = only
        esun.light_linking.blocker_collection = only
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
        sh.rotation_euler = (-a, s * 0.17, 0)   # held a little off the body, so the waist shows
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
        if d['flat'] == 'painted':
            # the iris: its tone is the one painted on it (the detail pass carries it), kept off the two ends of
            # the ramp so the two or three pixels of iris stay a readable green
            tone = np.clip(np.rint(det[sel] * 4), 1, 3).astype(int)
        elif d['flat'] is not None:
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


def sprite_mode(on):
    """At sprite size the hair's strand relief is finer than a pixel and only adds speckle; the painted strands
    (the detail pass) carry it there instead, so the relief is turned down for the sheet; the iris is drawn larger
    (see below)."""
    m = bpy.data.materials.get('hair')
    for n in m.node_tree.nodes:
        if n.type == 'BUMP':
            n.inputs['Strength'].default_value = 0.1 if on else 0.6
    # the iris is life-size, a pixel and a bit across in the sheet, and in the 3/4 views it fell between pixel
    # centres and the eye showed as a blank white: for the sprites it is drawn larger, as a painter would
    for o in bpy.data.objects:
        if o.name.startswith(('Eyeshadow', 'Lashes', 'Brow paint')):
            o.hide_render = on         # (finer than a sprite pixel: nothing to draw there)
            o.hide_viewport = on
        if o.name.startswith('Brow tube'):
            o.hide_render = not on
            o.hide_viewport = not on
        if o.name.startswith('Iris'):
            o.scale = (1.11,) * 3 if on else (1.0,) * 3
            o.data.shape_keys.key_blocks['sprite'].value = 1.0 if on else 0.0
        if o.name.startswith('Eye white') and o.data.shape_keys:
            o.data.shape_keys.key_blocks['tuck'].value = 0.0 if on else 1.0
            o.data.shape_keys.key_blocks['sprite'].value = 1.0 if on else 0.0


def render_sheet(out_png, tmp):
    sc = bpy.context.scene
    root = bpy.data.objects['Ranger']
    sheet = np.zeros((PH * ROWS, PW * COLS, 4), dtype=np.uint8)
    sprite_mode(True)
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
    bpy.context.preferences.filepaths.save_version = 0   # no .blend1 backups
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, 'ranger.blend'), compress=True)
    if '--build-only' in argv:
        return
    with tempfile.TemporaryDirectory() as tmp:
        render_sheet(os.path.join(out_dir, 'ranger-blender.png'), tmp)
    print('wrote', os.path.join(out_dir, 'ranger-blender.png'))


if __name__ == '__main__':
    main()
