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
M_MANTLE = mat_def(3, 'mantle', ramp(0x56663f, 0.1), 0.2)
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
M_EYE = mat_def(16, 'iris', [[46, 40, 38], [58, 66, 46], [88, 108, 70], [120, 140, 100], [226, 222, 210]], 0.0, flat=1)
M_LASH = mat_def(20, 'lashes', [[46, 40, 38]] * 5, 0.0, flat=0)
M_SCLERA = mat_def(21, 'eye white', [[226, 222, 210]] * 5, 0.0, flat=0)
M_BROW = mat_def(17, 'brow', ramp(0x60463a, 0.08), 0.0, flat=1)
M_LIPS = mat_def(18, 'lips', [[150, 86, 74], [168, 100, 86], [184, 116, 100], [196, 132, 114], [210, 150, 132]], 0.0, flat=1)
M_LACE = mat_def(19, 'laces', ramp(0x4a3a2c, 0.08), 0.0)
M_STITCH = mat_def(22, 'stitching', ramp(0xb39a74, 0.08), 0.0, flat=2)
# leather is matte, with a pebbled grain pressed into its surface
LEATHER = {M_JERKIN, M_BELT, M_BOOTS, M_CUFF}
# the cloth whose form is shaded by its folds and by the shade where it is shut in (see 'shoulders and cloth form')
FORM_SHADED = {M_SLEEVE, M_JERKIN}


# ---------------------------------------------------------------- scene and materials
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _srgb_lin(rgb255):
    return [((c / 255 + 0.055) / 1.055) ** 2.4 if c / 255 > 0.04045 else c / 255 / 12.92 for c in rgb255]


def _node(nt, kind, op=None, **inputs):
    n = nt.nodes.new(kind)
    if op:
        n.operation = op
    if kind == 'ShaderNodeMix':
        n.data_type = 'RGBA'     # (set first: the colour sockets share their names with the float ones)
    for k, v in inputs.items():
        if isinstance(v, bpy.types.NodeSocket):
            nt.links.new(v, n.inputs[k])
        else:
            n.inputs[k].default_value = v
    return n


def _math(nt, op, a, b=None, c=None):
    n = nt.nodes.new('ShaderNodeMath')
    n.operation = op
    for i, v in enumerate((a, b, c)):
        if v is None:
            continue
        if isinstance(v, bpy.types.NodeSocket):
            nt.links.new(v, n.inputs[i])
        else:
            n.inputs[i].default_value = v
    return n.outputs[0]


def wool_closeup(nt, bsdf, out, rmp):
    """What the mantle looks like up close (the sprite renders keep the plain diffuse surface; see sprite_mode):
    a fulled wool, matte, with no gloss at all. Its tone drifts in soft felted patches and is flecked with single
    lighter and darker fibres; the surface carries a fine fibrous nap that runs down the drape; a soft sheen
    lifts it where it turns away from the light, so its outline reads as fuzz rather than a hard rim; and the
    wool is lighter and fluffier along the hem and the neck, where the cut edge has been fulled up (the 'edge'
    attribute, 1 at the cut edges)."""
    bsdf.name = 'sprite bsdf'
    rest = nt.nodes.new('ShaderNodeAttribute')
    rest.attribute_name = 'rest'
    edge = nt.nodes.new('ShaderNodeAttribute')
    edge.attribute_name = 'edge'
    dark, mid, light, lighter = (_srgb_lin(rmp[i]) for i in (1, 2, 3, 4))
    # felted patches
    patch = _node(nt, 'ShaderNodeTexNoise', Vector=rest.outputs['Vector'], Scale=0.55, Detail=3.0, Roughness=0.55)
    pk = _node(nt, 'ShaderNodeMapRange', Value=patch.outputs['Fac'], **{'From Min': 0.3, 'From Max': 0.7})
    base = _node(nt, 'ShaderNodeMix', Factor=pk.outputs['Result'])
    base.inputs['A'].default_value = (*[lerp(a, b, 0.55) for a, b in zip(dark, mid)], 1)
    base.inputs['B'].default_value = (*[lerp(a, b, 0.3) for a, b in zip(mid, light)], 1)
    # the nap: fine fibres, drawn out down the drape (and a little across, so they cross like felt)
    stretch = _node(nt, 'ShaderNodeVectorMath', Vector=rest.outputs['Vector'])
    stretch.operation = 'MULTIPLY'
    stretch.inputs[1].default_value = (1.0, 1.0, 0.35)
    fib = _node(nt, 'ShaderNodeTexNoise', Vector=stretch.outputs['Vector'], Scale=9.0, Detail=6.0, Roughness=0.7)
    fuzz = _node(nt, 'ShaderNodeTexNoise', Vector=rest.outputs['Vector'], Scale=22.0, Detail=4.0, Roughness=0.75)
    # single fibres a shade lighter or darker than the cloth round them
    fleck = _node(nt, 'ShaderNodeMapRange', Value=fuzz.outputs['Fac'], **{'From Min': 0.35, 'From Max': 0.68,
                                                                      'To Min': -1.0, 'To Max': 1.0})
    fl_l = _math(nt, 'MULTIPLY', _math(nt, 'MAXIMUM', fleck.outputs['Result'], 0.0), 0.35)
    fl_d = _math(nt, 'MULTIPLY', _math(nt, 'MAXIMUM', _math(nt, 'MULTIPLY', fleck.outputs['Result'], -1.0), 0.0), 0.3)
    c1 = _node(nt, 'ShaderNodeMix', Factor=fl_l, A=base.outputs['Result'])
    c1.inputs['B'].default_value = (*light, 1)
    c2 = _node(nt, 'ShaderNodeMix', Factor=fl_d, A=c1.outputs['Result'])
    c2.inputs['B'].default_value = (*dark, 1)
    # the fulled edges: lighter and softer
    ek = _math(nt, 'MULTIPLY', edge.outputs['Fac'], 0.45)
    c3 = _node(nt, 'ShaderNodeMix', Factor=ek, A=c2.outputs['Result'])
    c3.inputs['B'].default_value = (*light, 1)
    # shut-in places (under the brooch, in the creases, where it meets the neck) go darker
    ao = _node(nt, 'ShaderNodeAmbientOcclusion', Distance=1.2)
    ao.samples = 8
    aok = _node(nt, 'ShaderNodeMapRange', Value=ao.outputs['AO'], **{'To Min': 0.5})
    shade = _node(nt, 'ShaderNodeVectorMath', 'SCALE', Vector=c3.outputs['Result'], Scale=aok.outputs['Result'])
    # relief: the nap, a finer fuzz over it, and the slow undulation of fulled cloth
    h = _math(nt, 'MULTIPLY_ADD', fib.outputs['Fac'], 0.7, _math(nt, 'MULTIPLY', fuzz.outputs['Fac'], 0.45))
    bump = _node(nt, 'ShaderNodeBump', Height=h, Strength=0.45, Distance=0.12)
    wb = _node(nt, 'ShaderNodeBsdfPrincipled', **{'Base Color': shade.outputs['Vector'], 'Normal': bump.outputs['Normal']})
    wb.name = 'close bsdf'
    wb.inputs['Roughness'].default_value = 1.0
    wb.inputs['Specular IOR Level'].default_value = 0.12
    wb.inputs['Sheen Weight'].default_value = 0.8
    wb.inputs['Sheen Roughness'].default_value = 0.6
    wb.inputs['Sheen Tint'].default_value = (*[min(1.0, 1.6 * c) for c in lighter], 1)
    nt.links.new(wb.outputs['BSDF'], out.inputs['Surface'])


def gold_closeup(nt, bsdf, out):
    """The brooch up close: worn gold, brighter on its raised midrib and rim than down in the hollows, where the
    tarnish and the shadow of the wool collect."""
    bsdf.name = 'sprite bsdf'
    # the engraving (cut into the mesh; its depth is the 'etch' attribute): sharpened, and filled with dark
    # tarnish, so each line reads as a crisp dark groove with the polished gold standing either side of it
    et = nt.nodes.new('ShaderNodeAttribute')
    et.attribute_name = 'etch'
    vein_k = _node(nt, 'ShaderNodeMapRange', Value=et.outputs['Fac'], **{'From Min': 0.2, 'From Max': 0.6}).outputs['Result']
    bump = _node(nt, 'ShaderNodeBump', Height=_math(nt, 'MULTIPLY', vein_k, -1.0), Strength=0.5, Distance=0.03)
    ao = _node(nt, 'ShaderNodeAmbientOcclusion', Distance=0.6)
    ao.samples = 8
    col = _node(nt, 'ShaderNodeMix', Factor=ao.outputs['AO'])
    col.inputs['A'].default_value = (0.20, 0.12, 0.04, 1)
    col.inputs['B'].default_value = (0.78, 0.55, 0.24, 1)
    rough = _node(nt, 'ShaderNodeMapRange', Value=ao.outputs['AO'], **{'To Min': 0.7, 'To Max': 0.32})
    dk = _node(nt, 'ShaderNodeMix', Factor=_math(nt, 'MULTIPLY', vein_k, 0.85), A=col.outputs['Result'])
    dk.inputs['B'].default_value = (0.05, 0.03, 0.012, 1)
    gb = _node(nt, 'ShaderNodeBsdfPrincipled', **{'Base Color': dk.outputs['Result'], 'Roughness': rough.outputs['Result'],
                                                 'Normal': bump.outputs['Normal']})
    gb.name = 'close bsdf'
    gb.inputs['Metallic'].default_value = 1.0
    nt.links.new(gb.outputs['BSDF'], out.inputs['Surface'])


def fuzz_shells(src, rmp, layers=3, step=0.07):
    """Loose fibres standing off the wool, for the close-up view only: a few see-through shells just outside the
    mantle, each speckled with the tips of fibres, sparser the further out it is. Facing the camera they are
    barely there; along the outline, where the eye looks through them edge-on, they gather into a soft fuzz.
    They are hidden for the sprite renders (see sprite_mode) and left out of the world model."""
    m = bpy.data.materials.new('wool fuzz')
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    rest = nt.nodes.new('ShaderNodeAttribute')
    rest.attribute_name = 'rest'
    shell = nt.nodes.new('ShaderNodeAttribute')     # which shell: 0 innermost to 1 outermost
    shell.attribute_type = 'OBJECT'
    shell.attribute_name = 'fuzz_layer'
    fib = _node(nt, 'ShaderNodeTexNoise', Vector=rest.outputs['Vector'], Scale=26.0, Detail=2.0, Roughness=0.6)
    thr = _math(nt, 'MULTIPLY_ADD', shell.outputs['Fac'], 0.1, 0.56)
    mask = _node(nt, 'ShaderNodeMapRange', Value=fib.outputs['Fac'], **{'From Min': 0.0, 'From Max': 0.06})
    nt.links.new(_math(nt, 'SUBTRACT', fib.outputs['Fac'], thr), mask.inputs['Value'])
    lw = _node(nt, 'ShaderNodeLayerWeight', Blend=0.5)
    edge = _math(nt, 'MULTIPLY_ADD', lw.outputs['Facing'], 0.85, 0.15)
    alpha = _math(nt, 'MULTIPLY', mask.outputs['Result'], edge)
    lit = nt.nodes.new('ShaderNodeBsdfPrincipled')
    lit.inputs['Base Color'].default_value = (*[lerp(a, b, 0.5) for a, b in zip(_srgb_lin(rmp[2]), _srgb_lin(rmp[3]))], 1)
    lit.inputs['Roughness'].default_value = 1.0
    lit.inputs['Specular IOR Level'].default_value = 0.1
    lit.inputs['Sheen Weight'].default_value = 0.6
    lit.inputs['Sheen Tint'].default_value = (*[min(1.0, 1.6 * c) for c in _srgb_lin(rmp[4])], 1)
    clear = nt.nodes.new('ShaderNodeBsdfTransparent')
    mix = _node(nt, 'ShaderNodeMixShader', Fac=alpha)
    nt.links.new(clear.outputs['BSDF'], mix.inputs[1])
    nt.links.new(lit.outputs['BSDF'], mix.inputs[2])
    nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
    m.blend_method = 'HASHED' if hasattr(m, 'blend_method') else None
    shells = []
    for i in range(layers):
        o = src.copy()
        o.name = f'Wool fuzz {i}'
        o.data = src.data.copy()
        o.data.materials.clear()
        o.data.materials.append(m)
        bpy.context.scene.collection.objects.link(o)
        o.parent = src.parent
        o['fuzz_layer'] = i / max(1, layers - 1)
        # (eased over the pinched valleys of the pleats first, so the shells bridge them instead of folding into
        # them and showing as dark nicks)
        sm = o.modifiers.new('ease', 'SMOOTH')
        sm.factor = 0.5
        sm.iterations = 4
        d = o.modifiers.new('stand off', 'DISPLACE')
        d.direction = 'NORMAL'
        d.mid_level = 0.0
        d.strength = step * (i + 1)
        o.visible_shadow = False
        shells.append(o)
    return shells


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
        # what the hair looks like up close (the sprite renders keep the plain diffuse surface above; see
        # sprite_mode). Not one even colour: each strand has its own tone, baked into the 'strands' colour (see
        # strand_colours), lighter at the ends; darker where the hair is shut in (between locks, under the fall),
        # which also lets the parts of the hair shade into one another rather than meet at an edge. Its sheen is
        # soft and warm, drawn out across the strands, and a fine fuzz at the edges: not a hard white highlight
        bsdf.name = 'sprite bsdf'
        col = nt.nodes.new('ShaderNodeVertexColor')
        col.layer_name = 'strands'
        ao = nt.nodes.new('ShaderNodeAmbientOcclusion')
        ao.inputs['Distance'].default_value = 1.6
        ao.samples = 8
        aok = nt.nodes.new('ShaderNodeMapRange')
        aok.inputs['To Min'].default_value = 0.45
        nt.links.new(ao.outputs['AO'], aok.inputs['Value'])
        tipc = nt.nodes.new('ShaderNodeMix')
        tipc.data_type = 'RGBA'
        tipc.blend_type = 'SCREEN'
        nt.links.new(tip_k.outputs[0], tipc.inputs['Factor'])
        nt.links.new(col.outputs['Color'], tipc.inputs['A'])
        tipc.inputs['B'].default_value = (0.32, 0.17, 0.10, 1.0)
        shade = nt.nodes.new('ShaderNodeVectorMath')
        shade.operation = 'SCALE'
        nt.links.new(tipc.outputs['Result'], shade.inputs[0])
        nt.links.new(aok.outputs['Result'], shade.inputs['Scale'])
        hb = nt.nodes.new('ShaderNodeBsdfPrincipled')
        hb.name = 'hair bsdf'
        nt.links.new(shade.outputs['Vector'], hb.inputs['Base Color'])
        hb.inputs['Roughness'].default_value = 0.62
        hb.inputs['Specular IOR Level'].default_value = 0.22
        hb.inputs['Specular Tint'].default_value = (0.95, 0.62, 0.42, 1.0)
        hb.inputs['Anisotropic'].default_value = 0.85
        hb.inputs['Sheen Weight'].default_value = 0.35
        hb.inputs['Sheen Roughness'].default_value = 0.45
        hb.inputs['Sheen Tint'].default_value = (0.95, 0.68, 0.5, 1.0)
        nt.links.new(bump.outputs['Normal'], hb.inputs['Normal'])
        # the sheen runs across the strands: the tangent is the direction down the hair, Z flattened onto the surface
        geo_ = nt.nodes.new('ShaderNodeNewGeometry')
        dn = nt.nodes.new('ShaderNodeVectorMath')
        dn.operation = 'DOT_PRODUCT'
        dn.inputs[1].default_value = (0, 0, 1)
        nt.links.new(geo_.outputs['Normal'], dn.inputs[0])
        sc_ = nt.nodes.new('ShaderNodeVectorMath')
        sc_.operation = 'SCALE'
        nt.links.new(geo_.outputs['Normal'], sc_.inputs[0])
        nt.links.new(dn.outputs['Value'], sc_.inputs['Scale'])
        tg = nt.nodes.new('ShaderNodeVectorMath')
        tg.operation = 'SUBTRACT'
        tg.inputs[0].default_value = (0, 0, 1)
        nt.links.new(sc_.outputs['Vector'], tg.inputs[1])
        tn = nt.nodes.new('ShaderNodeVectorMath')
        tn.operation = 'NORMALIZE'
        nt.links.new(tg.outputs['Vector'], tn.inputs[0])
        nt.links.new(tn.outputs['Vector'], hb.inputs['Tangent'])
        nt.links.new(hb.outputs['BSDF'], out.inputs['Surface'])
    if mid == M_MANTLE:
        wool_closeup(nt, bsdf, out, d['ramp'])
    if mid == M_GOLD:
        gold_closeup(nt, bsdf, out)
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
    if mid in FORM_SHADED:
        # how deep in a fold the cloth lies (the 'crease' attribute; see sleeve_folds and jerkin_form)
        acr = nt.nodes.new('ShaderNodeOutputAOV')
        acr.aov_name = 'crease'
        cra = nt.nodes.new('ShaderNodeAttribute')
        cra.attribute_name = 'crease'
        nt.links.new(cra.outputs['Fac'], acr.inputs['Value'])
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
        # fulled wool: soft felted patches, a nap of short fibres running down the drape (a pixel or two long at
        # sprite size, so it reads as texture, not speckle), and the cut edges a shade lighter, fluffed up
        patch = _node(nt, 'ShaderNodeTexNoise', Vector=vec, Scale=0.6, Detail=2.0, Roughness=0.55)
        st = _node(nt, 'ShaderNodeVectorMath', Vector=vec)
        st.operation = 'MULTIPLY'
        st.inputs[1].default_value = (1.0, 1.0, 0.4)
        nap = _node(nt, 'ShaderNodeTexNoise', Vector=st.outputs['Vector'], Scale=1.6, Detail=1.0, Roughness=0.5)
        edge = nt.nodes.new('ShaderNodeAttribute')
        edge.attribute_name = 'edge'
        h = _math(nt, 'MULTIPLY_ADD', nap.outputs['Fac'], 0.55, _math(nt, 'MULTIPLY', patch.outputs['Fac'], 0.45))
        h = _math(nt, 'MULTIPLY_ADD', edge.outputs['Fac'], 0.35, h)
        nt.links.new(h, adet.inputs['Value'])
        return m
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

    W = [(7.2, 2.0), (6.0, 3.0), (4.5, 3.6), (1.5, 3.7), (-1.5, 4.1), (-4.5, 4.3), (-9.0, 4.1), (-13.0, 3.6), (-16.0, 3.0), (-17.5, 2.2)]

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
            grow = min(1.0, (z_top - z) / 3.0)
            grow = grow * grow * (3 - 2 * grow)
            t_ = (0.95 - 0.3 * vs) * max(0.0, 1 - u * u) ** (0.5 if z < -0.5 else 1.4) * (1.0 if v < 0.92 else math.cos((v - 0.92) / 0.08 * math.pi / 2) ** 0.5)
            # (the tips are the thinnest part: a lock ends in a wisp, not a rounded slab)
            t = max(t_, lerp(0.4, 0.28, min(1.0, max(0.0, (v - 0.75) / 0.25))))
            # (what it rests on, blurred across: the hair bridges the folds of the mantle instead of carrying
            # each fold down its whole length as a groove)
            # (only below the nape: over the round of the head a blurred support would sink into the hair beneath)
            sup = back_y(x, z) if z > -0.5 else sum(back_y(x + dx_, z) * w_ for dx_, w_ in
                                                    ((-0.9, 0.1), (-0.45, 0.2), (0.0, 0.4), (0.45, 0.2), (0.9, 0.1)))
            # the least it can be: flush with the hair over the head at the top, then lying over what is under it.
            # Over the head its edges lie flush with the hair beneath (the extra thickness goes inward, out of
            # sight), so they leave no rim along the sides of the head
            y = sup + t_ * grow if z > -0.5 else max(sup + t * grow, sup + 0.52)
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
        col_out, col_in = [], []
        for j, (x, y, z, t, sup) in enumerate(pts):
            v = j / (NZ - 1)
            vs = min(1.0, (z_top - z) / (z_top - z_end))
            if vs > 0.8:          # the ends turn in a little toward her back rather than flicking out
                y = lerp(y, max(sup + 0.52, y - 0.6), ((vs - 0.8) / 0.2) ** 2 * 0.7)
            y = max(sup + 0.42 if z < -0.5 else y - 1.0, y)
            col_out.append([x, y, z, t, sup])
            # where it spans a hollow it is the full mass of hair down to what is under it (a thin sheet bridging
            # the nape would show its underside from the side); where it lies on the mantle it is pressed thinner
            gap = y - sup
            th = t if gap < t + 0.3 else max(t, gap - 0.15)
            col_in.append(Vector((x, max(y - th, sup + 0.12) if z < -0.5 else y - th, z)))
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
    band = [v for v in bm.verts if -4.0 < v.co.z < 7.5 and v.co.y > 1.0]
    for _ in range(4):
        bmesh.ops.smooth_laplacian_vert(bm, verts=band, lambda_factor=0.5, lambda_border=0.0,
                                        use_x=True, use_y=True, use_z=True, preserve_volume=True)
    hair = mesh_obj('Hair', bm, M_HAIR, head, subdiv=0, rest=hair_flow)
    strand_colours(hair)
    return hair


def strand_colours(o):
    """Each strand its own tone: the hair's colour attribute 'strands' (linear RGB), a mix of the darker, middle
    and lighter tones of its palette ramp, streaked along the fall of the hair at three widths (single strands,
    small locks, broad bands), a little darker on the underside of the fall, where it faces her back."""
    from mathutils import noise
    rmp = MATERIALS[M_HAIR]['ramp'] / 255.0
    lin = np.where(rmp > 0.04045, ((rmp + 0.055) / 1.055) ** 2.4, rmp / 12.92)
    dark, mid, light = lin[1], lin[2], lin[3]
    me = o.data
    rest = me.attributes['rest']
    attr = me.color_attributes.new('strands', 'FLOAT_COLOR', 'POINT')
    cols = []
    for v, r in zip(me.vertices, rest.data):
        a, z = r.vector.x, v.co.z
        t = (0.55 * noise.noise(Vector((a * 2.6, z * 0.05, 0.3)))
             + 0.35 * noise.noise(Vector((a * 0.9 + 11.0, z * 0.03, 2.7)))
             + 0.25 * noise.noise(Vector((a * 0.3 + 5.0, z * 0.02, 6.1))))
        t = min(1.0, max(-1.0, 1.6 * t))
        c = mid + (light - mid) * t if t > 0 else mid + (mid - dark) * t
        # a few single strands that catch the light
        if noise.noise(Vector((a * 7.0, z * 0.04, 9.0))) > 0.42:
            c = c + (light - c) * 0.5
        # the side of the fall that lies against her back is in its own shade
        if v.normal.y < -0.3 and z < 0.0:
            c = c * (1.0 - 0.35 * min(1.0, (-v.normal.y - 0.3) / 0.4))
        cols += [*c, 1.0]
    attr.data.foreach_set('color', cols)


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


# ---------------------------------------------------------------- shoulders and cloth form
# The shoulders and the form of the sleeves and the jerkin. The deltoid rounds the top of each arm out past the
# shoulder joint, so the line from the neck runs down over a curve into the arm instead of turning a square corner.
# The cloth itself carries its folds in its shape: the sleeves gather in creases at the armpit, crumple in a few
# curved wrinkles at the inside of the elbow and bunch above the cuff; the jerkin's stiffer leather takes a few
# soft compression wrinkles at the waist above and below the belt, its front panels are shaped by seams, and the
# short skirt falls in a few broad folds. Each fold function returns a radial offset and how deep in a crease the
# point lies (0 on open cloth, 1 at the bottom of a crease), stored on the mesh as 'crease' and read by the sprite
# and world shading as shade where the cloth is shut in.
def _gs(v, c, w):
    return math.exp(-((v - c) / w) ** 2)


def fold_ridge(t):
    """A run of folds along t: -1 at a pinched crease, rising to a broad rounded crest at +1 (cloth folds are
    narrow in the valleys and full over the tops)."""
    return 2 * math.sqrt(math.sin(t / 2) ** 2 + 0.03) - 1


def deltoid(z, phi, s):
    """Extra radius of the upper arm at local height z (0 at the shoulder joint): the deltoid, a rounded cap over
    the outside, front and back of the joint that tapers into the arm about halfway to the elbow."""
    out = s * math.sin(phi)                         # 1 straight out from the body, -1 toward it
    wrap = max(0.0, 0.45 + 0.55 * out) ** 1.6       # (round the front and back too, but not the inside)
    return 0.42 * wrap * math.exp(-((z + 1.4) / 2.1) ** 2) * (1.0 if z < 0.6 else math.exp(-((z - 0.6) / 0.7) ** 2))


def sleeve_folds(z, phi, s):
    """(radial offset, crease) of the sleeve at local height z, angle phi (phi 0 faces front)."""
    out = s * math.sin(phi)
    front = math.cos(phi)
    ez = -UPPER
    off, cr = 0.0, 0.0
    # the armpit: the cloth gathers where the arm meets the body, in two or three creases fanning down and out
    # from the pit over the front and back of the arm
    inner = max(0.0, -out) ** 1.2
    pit = inner * _gs(z, -1.9, 1.7)
    v = fold_ridge(3.6 * (z + 0.9) + 2.4 * abs(front) * (1 if front > 0 else -1) + 1.1)
    off += 0.16 * pit * v
    cr += pit * max(0.0, -v)
    # the elbow: at the inside of the bend the sleeve crumples in a few curved wrinkles (the ends curving up round
    # the arm), at the back it pulls smooth over the point of the elbow
    inside = max(0.0, front) ** 0.8
    band = _gs(z, ez + 0.3, 1.5)
    v = fold_ridge(4.2 * (z - ez) - 1.6 * (1 - front) + 0.5 * out)
    off += 0.3 * inside * band * v
    cr += inside * band * max(0.0, -v)
    # a long soft diagonal fold on each part of the arm, where the cloth hangs from the shoulder and twists down
    # the forearm
    v = fold_ridge(2.0 * phi * s + 0.55 * z + 0.6)
    k = 0.6 * _gs(z, -5.5, 2.4) + 0.8 * _gs(z, ez - 4.2, 2.0)
    off += 0.13 * k * v
    cr += 0.5 * k * max(0.0, -v)
    # the cloth bunched above the cuff, in a couple of uneven rolls
    wz = -UPPER - FORE + 1.6
    band = _gs(z, wz, 1.0)
    v = fold_ridge(5.5 * (z - wz) + 0.9 * math.sin(2 * phi + 0.7 * s))
    off += 0.22 * band * v
    cr += 0.8 * band * max(0.0, -v)
    return off, min(1.0, cr)


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
        seam = _gs(phi, -a, 0.045) * max(0.0, c) * _gs(z, 43.5, 3.6)
        off -= 0.05 * seam
        cr += 0.45 * seam
    # under the arms the leather creases where it is drawn in to the waist
    for p0 in (math.pi / 2, -math.pi / 2):
        band = _gs(phi, p0, 0.45) * _gs(z, 46.0, 2.0)
        v = fold_ridge(2.6 * (z - 46.0) + 3.0 * (phi - p0))
        off += 0.1 * band * v
        cr += 0.7 * band * max(0.0, -v)
    # the skirt: below the hips the leather falls in a few broad folds that deepen toward the hem
    k = min(1.0, max(0.0, (33.0 - z) / 3.5)) ** 1.2
    v = fold_ridge(5.0 * phi + 0.8 + 0.4 * math.sin(3 * phi))
    off += 0.32 * k * v
    cr += 0.8 * k * max(0.0, -v)
    # (and under the belt the leather lies smooth and is drawn in a little, so the belt cinches it)
    belt = _gs(z, 39.4, 0.8)
    return off * (1 - belt) - 0.12 * belt, min(1.0, cr * (1 - belt) + 0.5 * belt)


def crease_attr(o, fn):
    """Store fn(local co) -> crease on the mesh's points, as the 'crease' attribute."""
    a = o.data.attributes.new('crease', 'FLOAT', 'POINT')
    a.data.foreach_set('value', [fn(v.co) for v in o.data.vertices])


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
        off = jerkin_form(z, phi)[0]       # (its wrinkles, seams and folds: see jerkin_form)
        if z >= 32.0:
            x, y = torso_point(z, phi, J + off)
            return x, y
        # below the widest point of the hips the skirt falls straight on, opening a little
        x, y = torso_point(32.0, phi, J + (32.0 - z) * 0.06 + off)
        return x, y, -0.9 * math.cos(phi) ** 2 * (32.0 - z) / 4.0   # the hem dips at front and back

    # (its top runs up under the mantle, so the leather shows where the mantle parts below the brooch)
    jz = [49.6 - 0.4 * i for i in range(52)]          # 49.6 down to 29.2
    jb = loft_fn(jz, jerkin_pt, n=96, cap_bot=False)
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
        strand_colours(bang)
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
        # the folds: a few deep ones, and finer ruffles between them that grow toward the hem, all pinched at the
        # valleys (gathered cloth, not padded), each wandering a little off true as it runs down
        fold = (0.62 * pinch(6 * phi + 0.4 + 0.3 * math.sin(2.1 * z)) +
                0.38 * f * pinch(14 * phi + 1.3 + 0.5 * z)) * (1 - 0.8 * w)
        pull, lift = brooch_pull(z, phi)
        k = 1 + 0.085 * f * fold - pull / max(ry, 1.0)
        # (the hem drops a little at the sides, where it lies over the round of the arm)
        hem = (-0.45 * math.sin(phi) ** 4 - 0.5 * math.cos(phi)) * min(1.0, max(0.0, (49.6 - z) / 1.7))
        return -(rx + off) * k * math.sin(phi), cy - (ry + off) * k * math.cos(phi), hem + 0.4 * f * fold + lift

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
    # (the cap is fuller and rounder, and the deltoid swells it outward: see deltoid)
    ARM = [(1.6, 0.3), (1.35, 1.3), (0.85, 1.95), (0.0, 2.15), (-2.0, 2.1), (-5.5, 1.88), (-UPPER, 1.65), (-UPPER - 2.2, 1.78),
           (-UPPER - 5.0, 1.55), (-UPPER - FORE + 0.6, 1.3)]
    for side, s in (('R', 1), ('L', -1)):
        x = -s * SH_X   # the figure's right is -X (she faces -Y)
        sh = empty(f'Shoulder.{side}', (x, 0.2, SH_Z), body)
        el = empty(f'Elbow.{side}', (0, 0, -UPPER), sh)
        wr = empty(f'Wrist.{side}', (0, 0, -FORE), el)
        w = arm_weights(side)
        zs = [1.55 - 0.22 * i for i in range(int((1.55 + UPPER + FORE - 0.6) / 0.22) + 1)]

        def arm_pt(z, phi, s=s):
            r = profile(ARM, z) + deltoid(z, phi, s) + sleeve_folds(z, phi, s)[0]
            # the elbow: a little point at the back of the joint
            back = 0.25 * math.exp(-((z + UPPER) / 0.9) ** 2) * max(0.0, -math.cos(phi))
            return -r * math.sin(phi), -(r * 1.03 + back * 0) * math.cos(phi) + back
        so = mesh_obj(f'Sleeve.{side}', loft_fn(zs, arm_pt, n=48), M_SLEEVE, loc=(x, 0.2, SH_Z), skin=w)
        crease_attr(so, lambda co, s=s: sleeve_folds(co.z, math.atan2(-co.x, -co.y), s)[1])
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
    # (and ambient occlusion, for the shade in the folds and where the cloth is shut in: see form_shade)
    vl.use_pass_ambient_occlusion = True
    world.light_settings.distance = 2.2
    for name in ('matid', 'detail', 'crease'):
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


def form_shade(ch, light, mid, cov, sun, nz, ny):
    """On the sleeves and the jerkin, the sky's fill is shut out where the cloth is hidden (in the folds, under the
    mantle and the belt, between the arm and the body) and the creases of the folds take a little shade of
    their own, so the cloth reads as rounded forms rather than flat bands."""
    sel = np.isin(mid, list(FORM_SHADED))
    if not sel.any():
        return light
    ao = np.clip(find(ch, 'Ambient Occlusion', 'R') / cov, 0, 1) if any('Ambient Occlusion' in k for k in ch) else np.ones_like(light)
    cr = np.clip(find(ch, 'crease') / cov, 0, 1)
    fill = 0.30 + 0.12 * nz - 0.08 * ny
    # (and the cloth darkens as it turns away from the viewer, so the arms and the body read round)
    turn = (1.0 - np.clip(-ny, 0, 1)) ** 2
    shaded = sun * (1.0 - 0.35 * cr) + fill * (0.2 + 0.8 * ao) - 0.18 * cr - 0.2 * turn
    return np.where(sel, shaded, light)


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
    light = form_shade(ch, light, mid, cov, sun, nz, ny)
    h, w = alpha.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    solid = (alpha > 0.5) & (mid > 0)
    tones = np.full((h, w), -1, dtype=int)
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
        tones[sel] = tone
    # the brooch casts a pixel of shadow onto the wool below and right of it (the light is from the upper left),
    # so it sits in the cloth instead of on top of it
    gold = solid & (mid == M_GOLD)
    near = np.zeros_like(gold)
    near[1:, :] |= gold[:-1, :]
    near[1:, 1:] |= gold[:-1, :-1]
    near[:, 1:] |= gold[:, :-1]
    sh = near & solid & (mid == M_MANTLE) & (tones > 0)
    out[sh, :3] = MATERIALS[M_MANTLE]['ramp'][tones[sh] - 1]
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
    # the wool's outline is soft: where the outline runs beside the mantle, every third pixel or so is a shade
    # closer to the wool, so the edge reads as fuzz rather than a drawn line
    wool = np.zeros((h, w), dtype=bool)
    for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        nb = np.zeros_like(wool)
        ys0, ys1 = max(0, -dy), h - max(0, dy)
        xs0, xs1 = max(0, -dx), w - max(0, dx)
        nb[ys0:ys1, xs0:xs1] = (solid & (mid == M_MANTLE))[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        wool |= nb
    yy, xx = np.mgrid[0:h, 0:w]
    soft = wool & ~filled & (out[..., 3] > 0) & (((xx * 5 + yy * 3) // 2) % 3 == 0)
    out[soft, :3] = out[soft, :3] * 0.55 + MATERIALS[M_MANTLE]['ramp'][0] * 0.45
    return np.clip(out, 0, 255).astype(np.uint8)


def sprite_mode(on):
    """At sprite size the hair's strand relief is finer than a pixel and only adds speckle; the painted strands
    (the detail pass) carry it there instead, so the relief is turned down for the sheet."""
    m = bpy.data.materials.get('hair')
    nt = m.node_tree
    for n in nt.nodes:
        if n.type == 'BUMP':
            n.inputs['Strength'].default_value = 0.1 if on else 0.6
    # (and the sheet is lit from the plain diffuse surface, its tones picked from the palette ramp)
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
    nt.links.new(nt.nodes['sprite bsdf' if on else 'hair bsdf'].outputs['BSDF'], out.inputs['Surface'])
    # the mantle's wool and the brooch's gold, likewise: their close-up surfaces are for the 3D view only
    for name in ('mantle', 'clasp'):
        nt = bpy.data.materials[name].node_tree
        out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
        nt.links.new(nt.nodes['sprite bsdf' if on else 'close bsdf'].outputs['BSDF'], out.inputs['Surface'])
    for o in bpy.data.objects:
        if o.name.startswith('Wool fuzz'):
            o.hide_render = on


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
