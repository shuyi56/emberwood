"""The base female figure, modelled in Blender and rendered to a pixel-art sprite sheet.

Everything a character shares: the body's proportions and shape, the sculpted head and face, the eyes (balls,
lids, irises, brows, lashes and eyeshadow), the skin, the hands, the rig and its walk cycle, the materials
library (skin, hair, wool, leather, cloth, gold) and the sprite pipeline that renders the 8 facings with Cycles
and turns them into pixel art: every pixel takes its colour from the five-tone palette ramp of the material it
shows (the same ramps as src/shared/core.js), picked by how strongly it is lit, and the figure gets the
painter's coloured outline.

A character (blender/ranger/ranger.py, blender/rogue/rogue.py) is this figure plus its hair and its clothes. Its
build() runs the steps below in order, filling in its own parts between them:

    begin_figure('Name')                 the root and the trunk's pivots
    ...the clothes on the torso...
    head = build_head(body)              neck, sculpted head, eyes, brows, lashes, mouth
    ...hair, and anything over the shoulders...
    build_rig(body)
    build_arms(chest, sleeve)            the shoulder, elbow and wrist pivots; sleeve() dresses each arm; the hand
    build_legs(pelvis, leg)              the hip, knee and ankle pivots; leg() dresses each leg
    finish_figure(after_skin)            skins the trunk, then after_skin(), the rig, camera, light, keyframes

The sheet matches the painted sheets in public/sprites/: 10 columns (2 idle frames, then an 8-frame walk) by
8 facing rows (front, then turning toward screen right: front-3/4, right profile, back-3/4, back, and the
mirror side), each frame 128 x 160 pixels (the 64 x 80 design grid at RES 2), ground at row 154. Units: one
Blender unit is one design pixel (two sheet pixels); the figure faces -Y, the camera looks along +Y.
"""
import math
import os
import sys
import tempfile

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Euler  # noqa: F401  (Euler: for the characters)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
from ear import ear_grids, ear_tones, side_x_finder  # noqa: E402  (the ears, sculpted as their own surface)
from ear import EAR_BOTTOM, EAR_Y  # noqa: E402  (where the ear sits: the neck's upper joint is placed from it)

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
# the flushed skin of the cheeks, nose, knuckles and fingertips: the same lights and darks, rosier
SKIN_FLUSH = [[156, 88, 78], [190, 120, 104], [205, 142, 124], [217, 165, 148], [228, 192, 176]]
# id: (name, ramp, detail amplitude, flat tone or None, and how it is drawn; see mat_def)
MATERIALS = {}
# leather is matte, with a pebbled grain pressed into its surface (the materials with surface 'leather')
LEATHER = set()
# the cloth whose form is shaded by its folds and by the shade where it is shut in (see 'shoulders and cloth form')
FORM_SHADED = set()


def mat_def(mid, name, rmp, detail=0.0, flat=None, surface=None, closeup=None, form=False, soft_edge=False,
            shadow_on=(), alt=None):
    """Register material `mid` (1-99: the sprite reads it back from the 'matid' pass as mid / 100).

    surface    the pattern the sprite adds to the light (and the relief the render shades): 'hair' (strands and
               sheen), 'leather' (a pebbled grain pressed in, and mottled hide), 'wool' (felted patches and nap),
               'twill' (fine diagonal ribs), 'checker' (mail), or None (a soft noise)
    closeup    the 3D look kept for the close-up views (the sheet uses the plain diffuse surface; see
               sprite_mode): 'skin', 'wool', 'gold', 'leather', 'suede', 'cloth', 'canvas' or None
    form       its folds are shaded from the 'crease' attribute and where it is shut in (see form_shade)
    soft_edge  the outline beside it is broken up like fuzz (wool)
    shadow_on  materials it casts a pixel of shadow onto, below and right of it (a brooch on the cloth)
    alt        a second ramp, for the pixels the 'flush' pass marks (the skin's cheeks, nose, knuckles)
    A character may define a base material again (the hair, the brows, the iris) before it builds."""
    MATERIALS[mid] = dict(name=name, ramp=np.array(rmp, dtype=np.float32), detail=detail, flat=flat,
                          surface=surface, closeup=closeup, form=form, soft_edge=soft_edge, shadow_on=tuple(shadow_on),
                          alt=None if alt is None else np.array(alt, dtype=np.float32))
    LEATHER.discard(mid)
    FORM_SHADED.discard(mid)
    if surface == 'leather':
        LEATHER.add(mid)
    if form:
        FORM_SHADED.add(mid)
    return mid


# the base figure's own materials (ids 1, 2, 16-18 and 21; the characters number theirs from 3 and from 30)
M_SKIN = mat_def(1, 'skin', SKIN, 0.22, closeup='skin', alt=SKIN_FLUSH)
M_HAIR = mat_def(2, 'hair', ramp(0x7e4c34, 0.1), 0.24, surface='hair')
# the eyes: a moss-green iris dark enough to hold against the white and the skin at sprite size; the lids are
# skin. (No lash line and no painted catchlight: the eye's highlight is the sun's reflection in its glossy surface.)
# The iris ramp runs from the dark ring at its rim, through the green, to the warm amber round the pupil; it is
# painted onto the iris as vertex colours (see iris_colours), and the sprite picks its tone from that painting
M_EYE = mat_def(16, 'iris', [[30, 42, 30], [50, 76, 44], [72, 104, 58], [104, 134, 74], [148, 146, 86]], 0.0, flat='painted')
M_SCLERA = mat_def(21, 'eye white', [[206, 198, 186], [226, 220, 208], [224, 217, 205], [234, 229, 218], [242, 238, 230]], 0.0, flat=1)
M_BROW = mat_def(17, 'brow', ramp(0x60463a, 0.08), 0.0, flat=1)
M_LIPS = mat_def(18, 'lips', [[150, 86, 74], [168, 100, 86], [184, 116, 100], [196, 132, 114], [210, 150, 132]], 0.0, flat=1)


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


def leather_closeup(nt, bsdf, out, rmp, suede=False):
    """Boot leather up close (the sprite renders keep the plain diffuse surface; see sprite_mode): a waxed,
    oil-tanned hide, worn. Its tone drifts in broad patches; the grain is a fine pebbling with scattered pores;
    across the instep and the front of the ankle, where the boot bends with every step, it is pushed into a few
    short creases that have gone lighter and duller along their ridges; the toe and heel are scuffed pale; and the
    foot is darkened by mud and wet up to above the ankle, a soft tide line climbing the shaft. The wax gives the
    smooth places a soft, low sheen, broken up wherever the leather is scuffed or creased. The folded-down cuff
    shows the flesh side of the hide (suede): napped, matte and a shade lighter, with no wax."""
    bsdf.name = 'sprite bsdf'
    rest = nt.nodes.new('ShaderNodeAttribute')
    rest.attribute_name = 'rest'
    sep = _node(nt, 'ShaderNodeSeparateXYZ', Vector=rest.outputs['Vector'])
    zc = sep.outputs['Z']
    deep, dark, mid, light, lighter = (_srgb_lin(rmp[i]) for i in range(5))
    # broad patches where the hide's tone drifts
    patch = _node(nt, 'ShaderNodeTexNoise', Vector=rest.outputs['Vector'], Scale=0.45, Detail=3.0, Roughness=0.6)
    pk = _node(nt, 'ShaderNodeMapRange', Value=patch.outputs['Fac'], **{'From Min': 0.3, 'From Max': 0.7})
    base = _node(nt, 'ShaderNodeMix', Factor=pk.outputs['Result'])
    base.inputs['A'].default_value = (*[lerp(a, b, 0.6) for a, b in zip(dark, mid)], 1)
    base.inputs['B'].default_value = (*[lerp(a, b, 0.25) for a, b in zip(mid, light)], 1)
    # the grain: pebbling, finer noise over it, and pores
    peb = _node(nt, 'ShaderNodeTexVoronoi', Vector=rest.outputs['Vector'], Scale=7.0 if not suede else 14.0)
    peb.feature = 'SMOOTH_F1'
    fine = _node(nt, 'ShaderNodeTexNoise', Vector=rest.outputs['Vector'], Scale=26.0, Detail=5.0, Roughness=0.7)
    pores = _node(nt, 'ShaderNodeTexVoronoi', Vector=rest.outputs['Vector'], Scale=5.0)
    pore = _math(nt, 'LESS_THAN', pores.outputs['Distance'], 0.07)
    if suede:
        # the nap: short fibres brushed down, soft and lighter, and rubbed darker in patches where it is handled
        st = _node(nt, 'ShaderNodeVectorMath', Vector=rest.outputs['Vector'])
        st.operation = 'MULTIPLY'
        st.inputs[1].default_value = (1.0, 1.0, 0.3)
        nap = _node(nt, 'ShaderNodeTexNoise', Vector=st.outputs['Vector'], Scale=30.0, Detail=4.0, Roughness=0.7)
        nk = _node(nt, 'ShaderNodeMapRange', Value=nap.outputs['Fac'], **{'From Min': 0.35, 'From Max': 0.7})
        col = _node(nt, 'ShaderNodeMix', Factor=_math(nt, 'MULTIPLY', nk.outputs['Result'], 0.35), A=base.outputs['Result'])
        col.inputs['B'].default_value = (*lighter, 1)
        h = _math(nt, 'MULTIPLY_ADD', nap.outputs['Fac'], 0.8, _math(nt, 'MULTIPLY', peb.outputs['Distance'], 0.3))
        wear = None
    else:
        # creases where the boot bends: across the instep and the front of the ankle (short, wavering bands that
        # fade out round the sides), and a couple of soft ones behind the knee-high top
        cw = _node(nt, 'ShaderNodeTexWave', Vector=rest.outputs['Vector'], Scale=0.55, Distortion=4.0, Detail=1.5,
                   **{'Detail Scale': 0.6})
        cw.wave_type = 'BANDS'
        cw.bands_direction = 'Z'
        cw.wave_profile = 'SIN'
        # (only on the front of the ankle, where it folds: the rest's y is negative toward the front)
        front = _node(nt, 'ShaderNodeMapRange', Value=sep.outputs['Y'], **{'From Min': 0.2, 'From Max': -1.2}).outputs['Result']
        band = _math(nt, 'MULTIPLY', _node(nt, 'ShaderNodeMapRange', Value=zc, **{'From Min': 3.0, 'From Max': 4.0}).outputs['Result'],
                     _node(nt, 'ShaderNodeMapRange', Value=zc, **{'From Min': 7.0, 'From Max': 5.4}).outputs['Result'])
        band = _math(nt, 'MULTIPLY', band, front)
        # (each crease runs only part of the way across, broken where the leather lies flat)
        brk = _node(nt, 'ShaderNodeTexNoise', Vector=rest.outputs['Vector'], Scale=1.3, Detail=1.0)
        band = _math(nt, 'MULTIPLY', band, _node(nt, 'ShaderNodeMapRange', Value=brk.outputs['Fac'], **{'From Min': 0.38, 'From Max': 0.55}).outputs['Result'])
        ridge = _node(nt, 'ShaderNodeMapRange', Value=cw.outputs['Fac'], **{'From Min': 0.45, 'From Max': 1.0}).outputs['Result']
        crease = _math(nt, 'MULTIPLY', ridge, band)
        # scuffs: pale, rough patches at the toe, heel and the ankle bones, broken up by scratches
        sc_ = _node(nt, 'ShaderNodeTexNoise', Vector=rest.outputs['Vector'], Scale=3.2, Detail=6.0, Roughness=0.75)
        scuff = _node(nt, 'ShaderNodeMapRange', Value=sc_.outputs['Fac'], **{'From Min': 0.64, 'From Max': 0.76}).outputs['Result']
        low = _node(nt, 'ShaderNodeMapRange', Value=zc, **{'From Min': 2.6, 'From Max': 0.8}).outputs['Result']
        scuff = _math(nt, 'MULTIPLY', scuff, _math(nt, 'MULTIPLY_ADD', low, 0.75, 0.25))
        wear = _math(nt, 'MAXIMUM', _math(nt, 'MULTIPLY', crease, 0.35), scuff)
        # (worn leather goes a duller, warmer tan, a little lighter than the hide: never the pale grey of dust or paint)
        c1 = _node(nt, 'ShaderNodeMix', Factor=_math(nt, 'MULTIPLY', wear, 0.3), A=base.outputs['Result'])
        c1.inputs['B'].default_value = (*[lerp(a, b, 0.3) for a, b in zip(mid, light)], 1)
        # mud and wet: the foot darkened, fading up the shaft along a wavering tide line
        tide = _node(nt, 'ShaderNodeTexNoise', Vector=rest.outputs['Vector'], Scale=1.1, Detail=2.0)
        # (kept low, on the foot below the welt's top and up over the toe, with a clean edge, not a smeared fade)
        tz = _math(nt, 'MULTIPLY_ADD', tide.outputs['Fac'], 0.5, zc)
        mud = _node(nt, 'ShaderNodeMapRange', Value=tz, **{'From Min': 1.75, 'From Max': 1.55}).outputs['Result']
        # (and the mud itself, a dull grey-brown caked in specks over the wet leather nearest the sole)
        speck = _node(nt, 'ShaderNodeTexNoise', Vector=rest.outputs['Vector'], Scale=3.5, Detail=4.0, Roughness=0.65)
        caked = _math(nt, 'MULTIPLY', _node(nt, 'ShaderNodeMapRange', Value=speck.outputs['Fac'], **{'From Min': 0.58, 'From Max': 0.66}).outputs['Result'],
                      _node(nt, 'ShaderNodeMapRange', Value=tz, **{'From Min': 1.55, 'From Max': 1.25}).outputs['Result'])
        cw_ = _node(nt, 'ShaderNodeMix', Factor=_math(nt, 'MULTIPLY', mud, 0.45), A=c1.outputs['Result'])
        cw_.inputs['B'].default_value = (*[c * 0.6 for c in deep], 1)
        # (caked mud is the leather's own colour gone darker, not a fixed grey-brown that reads pale on dark hide)
        dk = _node(nt, 'ShaderNodeVectorMath', 'SCALE', Vector=cw_.outputs['Result'], Scale=0.55)
        col = _node(nt, 'ShaderNodeMix', Factor=_math(nt, 'MULTIPLY', caked, 0.7), A=cw_.outputs['Result'])
        nt.links.new(dk.outputs['Vector'], col.inputs['B'])
        mud = _math(nt, 'MAXIMUM', mud, caked)
        # (the grain is a fine, shallow pebble: smooth calf at this distance, not hammered)
        h = _math(nt, 'MULTIPLY_ADD', peb.outputs['Distance'], 0.35, _math(nt, 'MULTIPLY', fine.outputs['Fac'], 0.25))
        h = _math(nt, 'MULTIPLY_ADD', pore, -0.45, h)
        h = _math(nt, 'MULTIPLY_ADD', ridge, _math(nt, 'MULTIPLY', band, 0.6), h)      # (the folds are in the leather itself; the texture's ridges only hint at them)
        h = _math(nt, 'MULTIPLY_ADD', caked, 0.25, h)
    # shut-in places go darker (between the cuff and the shaft, under the stitching, where the foot meets the sole)
    ao = _node(nt, 'ShaderNodeAmbientOcclusion', Distance=0.8)
    ao.samples = 8
    aok = _node(nt, 'ShaderNodeMapRange', Value=ao.outputs['AO'], **{'To Min': 0.45})
    shade = _node(nt, 'ShaderNodeVectorMath', 'SCALE', Vector=col.outputs['Result'], Scale=aok.outputs['Result'])
    bump = _node(nt, 'ShaderNodeBump', Height=h, Strength=0.5 if not suede else 0.3, Distance=0.06)
    lb = _node(nt, 'ShaderNodeBsdfPrincipled', **{'Base Color': shade.outputs['Vector'], 'Normal': bump.outputs['Normal']})
    lb.name = 'close bsdf'
    if suede:
        lb.inputs['Roughness'].default_value = 1.0
        lb.inputs['Specular IOR Level'].default_value = 0.1
        lb.inputs['Sheen Weight'].default_value = 0.7
        lb.inputs['Sheen Roughness'].default_value = 0.5
        lb.inputs['Sheen Tint'].default_value = (*lighter, 1)
    else:
        # the wax: a soft sheen on the smooth leather, gone dull where it is scuffed, creased or muddy
        r = _math(nt, 'MULTIPLY_ADD', wear, 0.3, 0.55)
        r = _math(nt, 'MULTIPLY_ADD', mud, 0.2, r)
        nt.links.new(r, lb.inputs['Roughness'])
        lb.inputs['Specular IOR Level'].default_value = 0.3
    nt.links.new(lb.outputs['BSDF'], out.inputs['Surface'])


def cloth_closeup(nt, bsdf, out, rmp, canvas=False):
    """The trousers up close: a hard-wearing wool twill, its fine diagonal ribs showing in the surface, the yarn
    heathered with lighter and darker fibres, worn paler and a little fuzzy over the front of the thighs and the
    seat, and dulled darker into the creases. The knee patches are a coarser plain-woven canvas (canvas=True),
    worn paler at the middle where the knee rubs through."""
    bsdf.name = 'sprite bsdf'
    rest = nt.nodes.new('ShaderNodeAttribute')
    rest.attribute_name = 'rest'
    deep, dark, mid, light, lighter = (_srgb_lin(rmp[i]) for i in range(5))
    patch = _node(nt, 'ShaderNodeTexNoise', Vector=rest.outputs['Vector'], Scale=0.35, Detail=1.0, Roughness=0.5)
    pk = _node(nt, 'ShaderNodeMapRange', Value=patch.outputs['Fac'], **{'From Min': 0.3, 'From Max': 0.7})
    base = _node(nt, 'ShaderNodeMix', Factor=pk.outputs['Result'])
    base.inputs['A'].default_value = (*[lerp(a, b, 0.75) for a, b in zip(dark, mid)], 1)
    base.inputs['B'].default_value = (*[lerp(a, b, 0.1) for a, b in zip(mid, light)], 1)
    if canvas:
        # a plain weave: threads over and under in both directions
        wa = _node(nt, 'ShaderNodeTexWave', Vector=rest.outputs['Vector'], Scale=7.0, Distortion=0.6)
        wa.bands_direction = 'Z'
        wb = _node(nt, 'ShaderNodeTexWave', Vector=rest.outputs['Vector'], Scale=7.0, Distortion=0.6)
        wb.bands_direction = 'X'
        wc = _node(nt, 'ShaderNodeTexWave', Vector=rest.outputs['Vector'], Scale=7.0, Distortion=0.6)
        wc.bands_direction = 'Y'
        weave = _math(nt, 'MULTIPLY', wa.outputs['Fac'], _math(nt, 'MAXIMUM', wb.outputs['Fac'], wc.outputs['Fac']))
    else:
        # the twill: diagonal ribs, each broken into threads along its length
        weave = None
        tw = _node(nt, 'ShaderNodeTexWave', Vector=rest.outputs['Vector'], Scale=7.0, Distortion=0.5, Detail=1.0)
        tw.bands_direction = 'DIAGONAL'
        thr = _node(nt, 'ShaderNodeTexNoise', Vector=rest.outputs['Vector'], Scale=40.0, Detail=1.0)
        weave = _math(nt, 'MULTIPLY', tw.outputs['Fac'], _math(nt, 'MULTIPLY_ADD', thr.outputs['Fac'], 0.5, 0.75))
    # heathered yarn: single fibres a shade lighter or darker
    fz = _node(nt, 'ShaderNodeTexNoise', Vector=rest.outputs['Vector'], Scale=55.0, Detail=3.0, Roughness=0.7)
    fleck = _node(nt, 'ShaderNodeMapRange', Value=fz.outputs['Fac'], **{'From Min': 0.35, 'From Max': 0.65,
                                                                    'To Min': -1.0, 'To Max': 1.0})
    c1 = _node(nt, 'ShaderNodeMix', Factor=_math(nt, 'MULTIPLY', _math(nt, 'MAXIMUM', fleck.outputs['Result'], 0.0), 0.3),
               A=base.outputs['Result'])
    c1.inputs['B'].default_value = (*light, 1)
    c2 = _node(nt, 'ShaderNodeMix', Factor=_math(nt, 'MULTIPLY', _math(nt, 'MAXIMUM', _math(nt, 'MULTIPLY', fleck.outputs['Result'], -1.0), 0.0), 0.3),
               A=c1.outputs['Result'])
    c2.inputs['B'].default_value = (*dark, 1)
    # wear: paler where the cloth faces forward over the thighs (or the middle of the patch), in soft patches
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    wn = _node(nt, 'ShaderNodeTexNoise', Vector=rest.outputs['Vector'], Scale=0.8, Detail=2.0, Roughness=0.5)
    wk = _node(nt, 'ShaderNodeMapRange', Value=wn.outputs['Fac'], **{'From Min': 0.4, 'From Max': 0.75}).outputs['Result']
    if canvas:
        wear = _math(nt, 'MULTIPLY', wk, 0.6)
    else:
        sep = _node(nt, 'ShaderNodeSeparateXYZ', Vector=rest.outputs['Vector'])
        thigh = _math(nt, 'MULTIPLY', _node(nt, 'ShaderNodeMapRange', Value=sep.outputs['Z'], **{'From Min': 12.0, 'From Max': 20.0}).outputs['Result'],
                      _node(nt, 'ShaderNodeMapRange', Value=sep.outputs['Z'], **{'From Min': 31.0, 'From Max': 25.0}).outputs['Result'])
        wear = _math(nt, 'MULTIPLY', wk, _math(nt, 'MULTIPLY_ADD', thigh, 0.5, 0.15))
    c3 = _node(nt, 'ShaderNodeMix', Factor=_math(nt, 'MULTIPLY', wear, 0.35), A=c2.outputs['Result'])
    c3.inputs['B'].default_value = (*lighter, 1)
    ao = _node(nt, 'ShaderNodeAmbientOcclusion', Distance=1.0)
    ao.samples = 8
    aok = _node(nt, 'ShaderNodeMapRange', Value=ao.outputs['AO'], **{'To Min': 0.45})
    shade = _node(nt, 'ShaderNodeVectorMath', 'SCALE', Vector=c3.outputs['Result'], Scale=aok.outputs['Result'])
    h = _math(nt, 'MULTIPLY_ADD', fz.outputs['Fac'], 0.25, weave)
    bump = _node(nt, 'ShaderNodeBump', Height=h, Strength=0.45 if not canvas else 0.6, Distance=0.05)
    cb = _node(nt, 'ShaderNodeBsdfPrincipled', **{'Base Color': shade.outputs['Vector'], 'Normal': bump.outputs['Normal']})
    cb.name = 'close bsdf'
    cb.inputs['Roughness'].default_value = 1.0
    cb.inputs['Specular IOR Level'].default_value = 0.15
    cb.inputs['Sheen Weight'].default_value = 0.5
    cb.inputs['Sheen Roughness'].default_value = 0.5
    cb.inputs['Sheen Tint'].default_value = (*light, 1)
    nt.links.new(cb.outputs['BSDF'], out.inputs['Surface'])


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
    if d['surface'] == 'leather':
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
    if d['surface'] == 'hair':
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
        # (each piece shades only itself: the fringe darkened the side hair under its edge into a crease)
        ao.only_local = True
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
    close = d['closeup']
    if close == 'wool':
        wool_closeup(nt, bsdf, out, d['ramp'])
    if close == 'gold':
        gold_closeup(nt, bsdf, out)
    if close in ('leather', 'suede'):
        leather_closeup(nt, bsdf, out, d['ramp'], suede=close == 'suede')
    if close in ('cloth', 'canvas'):
        cloth_closeup(nt, bsdf, out, d['ramp'], canvas=close == 'canvas')
    if close == 'skin':
        skin_surface(nt, bsdf, out)
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
    if d['form']:
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
    if d['surface'] == 'hair':
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
    elif mid == M_SKIN:
        # the sprite: the baked shade (under the jaw and the eyes, between the fingers) darkens the light before
        # it is banded, and the flushed parts take the rosier ramp (the 'flush' pass; see to_pixels)
        fx = nt.nodes.new('ShaderNodeVertexColor')
        fx.layer_name = 'skin fx'
        sp = nt.nodes.new('ShaderNodeSeparateColor')
        nt.links.new(fx.outputs['Color'], sp.inputs['Color'])
        d_ = nt.nodes.new('ShaderNodeMath')
        d_.operation = 'MULTIPLY_ADD'
        d_.inputs[1].default_value = -0.5
        d_.inputs[2].default_value = 0.5
        nt.links.new(sp.outputs['Green'], d_.inputs[0])
        nt.links.new(d_.outputs[0], adet.inputs['Value'])
        afl = nt.nodes.new('ShaderNodeOutputAOV')
        afl.aov_name = 'flush'
        nt.links.new(sp.outputs['Red'], afl.inputs['Value'])
        return m
    elif d['surface'] == 'checker':
        tex = nt.nodes.new('ShaderNodeTexChecker')
        tex.inputs['Scale'].default_value = 1.0
    elif d['surface'] == 'twill':
        # a twill: fine diagonal ribs, with a little drift so they read as woven rather than ruled
        tex = nt.nodes.new('ShaderNodeTexWave')
        tex.wave_type = 'BANDS'
        tex.bands_direction = 'DIAGONAL'
        tex.inputs['Scale'].default_value = 0.22
        tex.inputs['Distortion'].default_value = 0.8
        tex.inputs['Detail'].default_value = 0.0
    elif d['surface'] == 'leather':
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
    elif d['surface'] == 'wool':
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


# The skin's close-up shading, as candidate looks (skin_study.py renders them side by side on stand-in shapes,
# without building a figure); SKIN_LOOK is the one the figures wear. Each look lists only what it changes from the
# one before it in the table, so the study shows what every step adds:
# - fresnel: the reflection of real skin. Skin's surface is a dielectric of IOR about 1.4 (F0 about 2.8%), which
#   is the Principled default at Specular IOR Level 0.5; turning that level down (0.38 before) cuts the grazing
#   reflection that rims a face in light. And light inside skin scatters mostly forward (anisotropy about 0.8),
#   which keeps the scattering from going milky.
# - oil: a second, sharper highlight. Measured skin fits two specular lobes: a broad one from the skin itself and a
#   tighter one from the thin film of sebum over it, strongest on the nose, the brow and the chin. The film is the
#   coat here, weighted by the 'skin fx' gloss channel, and it alone carries the full pores and creases (Coat
#   Normal), while the scattered light under it sees a softer relief, as light that has spread inside the skin does.
# - fuzz: the vellus hair (peach fuzz): a faint sheen that lifts the skin where it turns away from the viewer, so
#   the outline of the face goes soft instead of cut out.
# - patchy: how glossy and how reflective the skin is drifts in patches a centimetre or two across, so a highlight
#   breaks up instead of sliding over it as over moulded plastic.
SKIN_LOOKS = {
    'current': dict(spec=0.38, aniso=0.0, rough=(0.56, 0.38), coat=(0.0, 0.0), coat_rough=(0.35, 0.35),
                    base_bump=1.0, sheen=0.0, patch=0.0),
    'fresnel': dict(spec=0.5, aniso=0.8),
    'oil': dict(rough=(0.62, 0.5), coat=(0.08, 0.4), coat_rough=(0.42, 0.26), base_bump=0.45),
    'fuzz': dict(sheen=0.35),
    'patchy': dict(patch=1.0),
    # (the study's pick: all of the above, with the fuzz turned down, as at 0.35 it frosted the outline white, and
    # the oil film thinner and less glossy: on the ranger's face (#125) the full film ran down the nose bridge and
    # the far cheek as a wet streak)
    'final': dict(sheen=0.15, coat=(0.05, 0.22), coat_rough=(0.46, 0.33)),
    # the skin built in layers (skin_layers.py: a flesh base, veins, the epidermis over both), its epidermis the
    # figure's baked 'skin' tone
    'layered': dict(layers=True),
}
for _a, _b in zip(list(SKIN_LOOKS), list(SKIN_LOOKS)[1:]):
    SKIN_LOOKS[_b] = {**SKIN_LOOKS[_a], **SKIN_LOOKS[_b]}
SKIN_LOOK = 'current'
# the face creator's Fair swatch (#efcfb9) for the figure's skin, in place of the ranger's tone: views.py --set SKIN_FAIR=1
SKIN_FAIR = False
SKIN_FAIR_TONE = [239, 207, 185]


def skin_surface(nt, bsdf, out, look=None):
    """What the skin looks like up close (the sprite renders keep the plain diffuse surface; see sprite_mode). Its
    colour is the baked 'skin' tone (see skin_tones): rosier at the cheeks, nose, chin, knuckles and fingertips,
    cooler and a little darker under the eyes and the jaw, warmer where it is shut in. Light passes into it and
    comes out reddened (subsurface scattering, red reaching furthest), which softens the turn from light to shade
    and warms the edge of every shadow.

    Its texture, all laid out in the rest position so it stays fixed to the skin as she moves (and drawn a little
    larger than life, so it shows in a close-up of the head rather than only under a lens):
    - the fine net of creases that covers all skin, small plateaus between shallow grooves, at two sizes;
    - pores, small pits, densest on the nose, the chin and the cheeks beside the nose;
    - creases across the knuckles and faint rings round the neck;
    - colour that is not even: blotches a little redder or sallower, a fine speckle, the grooves and pores a
      shade darker, and a light scatter of freckles over the nose and cheeks and the backs of the hands;
    - a sheen that breaks up over the relief: rougher in the grooves, glossier on the nose, chin and brow.
    The 'skin tex' colour says where (red: freckles, green: creases, blue: pores). The look (SKIN_LOOKS, by name
    or by its place in the table) sets the reflection, the oil film, the fuzz and the patchiness."""
    look = SKIN_LOOK if look is None else look
    K = SKIN_LOOKS[look] if isinstance(look, str) else list(SKIN_LOOKS.values())[int(look)]
    L = nt.links.new
    if K.get('layers'):
        import skin_layers
        bsdf.name = 'sprite bsdf'
        col_ = nt.nodes.new('ShaderNodeVertexColor')
        col_.layer_name = 'skin'
        skin_layers.skin_layers(nt, out, 'all', col_.outputs['Color'])
        return

    def node(kind, **props):
        n = nt.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def math_(op, a, b=None, c=None):
        n = node('ShaderNodeMath', operation=op)
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                L(v, n.inputs[i])
        return n.outputs[0]

    def ramp_(v, lo, hi, to_lo=0.0, to_hi=1.0, smooth=True):
        n = node('ShaderNodeMapRange', interpolation_type='SMOOTHSTEP' if smooth else 'LINEAR')
        L(v, n.inputs['Value'])
        for k, x in (('From Min', lo), ('From Max', hi), ('To Min', to_lo), ('To Max', to_hi)):
            n.inputs[k].default_value = x
        return n.outputs['Result']

    def tex(kind, scale, **props):
        n = node(kind, **props)
        n.inputs['Scale'].default_value = scale
        L(rest.outputs['Vector'], n.inputs['Vector'])
        return n

    def tint(colour, target, fac):
        n = node('ShaderNodeMix', data_type='RGBA')
        L(fac, n.inputs['Factor'])
        L(colour, n.inputs['A'])
        n.inputs['B'].default_value = (*_srgb_lin(target), 1.0)
        return n.outputs['Result']

    bsdf.name = 'sprite bsdf'
    col = node('ShaderNodeVertexColor', layer_name='skin')
    rest = node('ShaderNodeAttribute', attribute_name='rest')
    fx = node('ShaderNodeSeparateColor')
    L(node('ShaderNodeVertexColor', layer_name='skin fx').outputs['Color'], fx.inputs['Color'])
    where = node('ShaderNodeSeparateColor')
    L(node('ShaderNodeVertexColor', layer_name='skin tex').outputs['Color'], where.inputs['Color'])
    freckly, creased, pored = where.outputs['Red'], where.outputs['Green'], where.outputs['Blue']

    # ---- relief
    # the net of fine creases: 1 on the plateaus, falling to 0 in the grooves between them
    net = ramp_(tex('ShaderNodeTexVoronoi', 12.0, feature='DISTANCE_TO_EDGE').outputs['Distance'], 0.0, 0.2)
    net2 = ramp_(tex('ShaderNodeTexVoronoi', 27.0, feature='DISTANCE_TO_EDGE').outputs['Distance'], 0.0, 0.25)
    # pores: a pit at the middle of each cell of a finer pattern, only where the skin is pored
    pv = tex('ShaderNodeTexVoronoi', 13.0, feature='F1')
    pit = math_('MULTIPLY', ramp_(pv.outputs['Distance'], 0.05, 0.3, 1.0, 0.0), pored)
    # creases across the knuckles (and the neck): bands running round the finger, wavering, deepest mid-joint
    wave = tex('ShaderNodeTexWave', 1.7, wave_type='BANDS', bands_direction='Z', wave_profile='SIN')
    wave.inputs['Distortion'].default_value = 2.2
    wave.inputs['Detail'].default_value = 1.5
    wave.inputs['Detail Scale'].default_value = 1.4
    fold = math_('MULTIPLY', ramp_(wave.outputs['Fac'], 0.25, 0.75, 1.0, 0.0), creased)
    height = math_('MULTIPLY_ADD', net, 0.3, math_('MULTIPLY', net2, 0.2))
    height = math_('SUBTRACT', height, math_('MULTIPLY_ADD', pit, 1.6, math_('MULTIPLY', fold, 2.0)))
    bump = node('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.2
    bump.inputs['Distance'].default_value = 0.03
    L(height, bump.inputs['Height'])

    # ---- colour
    # blotches (redder or sallower), then a fine speckle of lighter and darker
    blotch = tex('ShaderNodeTexNoise', 1.3)
    blotch.inputs['Detail'].default_value = 2.0
    c = tint(col.outputs['Color'], [206, 124, 110], ramp_(blotch.outputs['Fac'], 0.52, 0.8, 0.0, 0.22))
    c = tint(c, [212, 176, 128], ramp_(blotch.outputs['Fac'], 0.48, 0.2, 0.0, 0.2))
    fine = tex('ShaderNodeTexNoise', 9.0)
    fine.inputs['Detail'].default_value = 3.0
    fine.inputs['Roughness'].default_value = 0.65
    # (and the grooves and pores hold a little shade)
    k = math_('MULTIPLY', ramp_(fine.outputs['Fac'], 0.25, 0.75, 0.92, 1.06, smooth=False),
              math_('SUBTRACT', math_('MULTIPLY_ADD', net, 0.025, 0.975), math_('MULTIPLY_ADD', pit, 0.1, math_('MULTIPLY', fold, 0.12))))
    sc_ = node('ShaderNodeVectorMath', operation='SCALE')
    L(c, sc_.inputs[0])
    L(k, sc_.inputs['Scale'])
    # freckles: some cells of a scatter carry a soft brown fleck, each its own size and strength
    fv = tex('ShaderNodeTexVoronoi', 5.5, feature='F1')
    fv.inputs['Randomness'].default_value = 1.0
    rnd = node('ShaderNodeSeparateColor')
    L(fv.outputs['Color'], rnd.inputs['Color'])
    size = math_('MULTIPLY_ADD', rnd.outputs['Green'], 0.1, 0.07)
    fleck = ramp_(math_('DIVIDE', fv.outputs['Distance'], size), 0.55, 1.0, 1.0, 0.0)
    some = math_('MULTIPLY', ramp_(rnd.outputs['Red'], 0.3, 0.4), math_('MULTIPLY_ADD', rnd.outputs['Blue'], 0.6, 0.4))
    freckle = math_('MULTIPLY', math_('MULTIPLY', fleck, some), freckly)
    c = tint(sc_.outputs['Vector'], [150, 96, 66], freckle)
    # a soft contact shadow where the hair lies against the skin or hangs just off it, and in the other shut-in
    # places (under the jaw, round the nose), deepening toward a warmer, redder tone the way skin does in shade,
    # so the hair sits on the face instead of being cut out against it
    ao = node('ShaderNodeAmbientOcclusion')
    ao.inputs['Distance'].default_value = 0.9
    ao.samples = 16
    deep = node('ShaderNodeVectorMath', operation='MULTIPLY')
    L(c, deep.inputs[0])
    deep.inputs[1].default_value = (0.42, 0.26, 0.22)
    shut = node('ShaderNodeMix', data_type='RGBA')
    L(ramp_(ao.outputs['AO'], 0.25, 1.0, 0.0, 1.0, smooth=False), shut.inputs['Factor'])
    L(deep.outputs['Vector'], shut.inputs['A'])
    L(c, shut.inputs['B'])
    c = shut.outputs['Result']

    sb = node('ShaderNodeBsdfPrincipled', name='close bsdf')
    L(c, sb.inputs['Base Color'])
    sb.subsurface_method = 'RANDOM_WALK_SKIN'
    sb.inputs['Subsurface Weight'].default_value = 1.0
    # (the mean free paths of measured skin, red 3.7, green 1.4, blue 0.7 mm: Jensen et al. 2001)
    sb.inputs['Subsurface Radius'].default_value = (1.0, 0.36, 0.18)
    sb.inputs['Subsurface Scale'].default_value = 0.12      # (one unit is about 2.7 cm on her)
    sb.inputs['Subsurface Anisotropy'].default_value = K['aniso']
    sb.inputs['IOR'].default_value = 1.4
    # patches a centimetre or two across where the skin is a little glossier or duller, more or less reflective
    pn = tex('ShaderNodeTexNoise', 0.9)
    pn.inputs['Detail'].default_value = 2.0
    patch = math_('MULTIPLY', math_('SUBTRACT', pn.outputs['Fac'], 0.5), 2.0 * K['patch'])     # about -0.3..0.3
    spec = math_('MULTIPLY', K['spec'], math_('MULTIPLY_ADD', patch, 0.5, 1.0))
    # (the pores and grooves hold less light to send back)
    L(math_('MULTIPLY', spec, math_('SUBTRACT', 1.0, math_('MULTIPLY', pit, 0.5 * K['patch']))),
      sb.inputs['Specular IOR Level'])
    # gloss: the 'skin fx' blue channel is how oily the skin is there (nose, chin, brow); the grooves, pores and
    # creases are duller than the plateaus between them, which breaks the highlight up into a skin-like sparkle
    rough = ramp_(fx.outputs['Blue'], 0.0, 1.0, K['rough'][0], K['rough'][1], smooth=False)
    rough = math_('ADD', rough, math_('MULTIPLY_ADD', math_('SUBTRACT', 1.0, net), 0.06,
                                      math_('MULTIPLY_ADD', pit, 0.14, math_('MULTIPLY', fine.outputs['Fac'], 0.08))))
    rough = math_('MULTIPLY_ADD', patch, -0.25, rough)
    L(rough, sb.inputs['Roughness'])
    # the relief: in full for the surface reflections, softer for the light that has spread inside the skin
    if K['base_bump'] < 1.0:
        soft = node('ShaderNodeBump')
        soft.inputs['Strength'].default_value = 0.2 * K['base_bump']
        soft.inputs['Distance'].default_value = 0.03
        L(height, soft.inputs['Height'])
        L(soft.outputs['Normal'], sb.inputs['Normal'])
    else:
        L(bump.outputs['Normal'], sb.inputs['Normal'])
    # the film of oil: a clear coat, thicker and glossier where the skin is oily, broken by the pores
    if K['coat'][1] > 0.0:
        cw = ramp_(fx.outputs['Blue'], 0.0, 1.0, K['coat'][0], K['coat'][1], smooth=False)
        L(math_('MULTIPLY', cw, math_('SUBTRACT', 1.0, math_('MULTIPLY', pit, 0.6))), sb.inputs['Coat Weight'])
        cr = ramp_(fx.outputs['Blue'], 0.0, 1.0, K['coat_rough'][0], K['coat_rough'][1], smooth=False)
        L(math_('MULTIPLY_ADD', patch, -0.15, cr), sb.inputs['Coat Roughness'])
        sb.inputs['Coat IOR'].default_value = 1.45
        L(bump.outputs['Normal'], sb.inputs['Coat Normal'])
    # the fuzz: a soft sheen at grazing angles, a little warm
    if K['sheen'] > 0.0:
        sb.inputs['Sheen Weight'].default_value = K['sheen']
        sb.inputs['Sheen Roughness'].default_value = 0.4
        sb.inputs['Sheen Tint'].default_value = (1.0, 0.86, 0.78, 1.0)
    L(sb.outputs['BSDF'], out.inputs['Surface'])


def _srgb_lin(c):
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)


def _g(u, v, u0, v0, su, sv):
    return np.exp(-((u - u0) / su) ** 2 - ((v - v0) / sv) ** 2)


def _occlusion(co, nrm, tree, rays=12, dist=1.4, seed=5):
    """How shut in each point is: the share of rays out over its surface that meet something within dist."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(rays, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    occ = np.zeros(len(co))
    for i in range(len(co)):
        n = Vector(nrm[i])
        p = Vector(co[i]) + n * 0.04
        hit = 0
        for d in dirs:
            d = Vector(d)
            if d.dot(n) < 0:
                d = -d
            if tree.ray_cast(p, (d + n).normalized(), dist)[0] is not None:
                hit += 1
        occ[i] = hit / rays
    return occ


def skin_tones():
    """Bake the skin's colour into each skin mesh, as two point colours: 'skin', the linear albedo the close-ups
    and the world model show, 'skin fx' (red: flush, green: shade, blue: gloss) for the sprite passes and the
    roughness, and 'skin tex' (red: freckles, green: creases, blue: pores), where the close-up texture goes. Painted from where each point lies on the part: blush across the cheeks and on the nose, the
    chin a little warm, cool and darker under the eyes and on the underside of the jaw, the knuckles and
    fingertips rosy and the thin skin at the wrist cooler; and everywhere the skin is shut in (the eye sockets, under
    the jaw, between the fingers) it goes darker and warmer, the way light that has passed through skin comes
    out red."""
    from mathutils.bvhtree import BVHTree
    base, flushc, cool = _srgb_lin(SKIN_FAIR_TONE if SKIN_FAIR else SKIN[2]), _srgb_lin([204, 128, 112]), _srgb_lin([164, 138, 126])
    deep = _srgb_lin([150, 78, 64])
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()

    def world_tree(names):
        vs, ps = [], []
        for nm in names:
            o = bpy.data.objects.get(nm)
            if o is None:
                continue
            me = o.evaluated_get(dg).to_mesh()
            b = len(vs)
            vs += [o.matrix_world @ v.co for v in me.vertices]
            ps += [[b + i for i in p.vertices] for p in me.polygons]
            o.evaluated_get(dg).to_mesh_clear()
        return BVHTree.FromPolygons(vs, ps)

    def rest_of(o, co):       # the pattern coordinates of a mesh's points (see mesh_obj)
        a = np.zeros(len(co) * 3, dtype=np.float32)
        o.data.attributes['rest'].data.foreach_get('vector', a)
        return a.reshape(-1, 3)

    head_parts = [o.name for o in bpy.data.objects if o.parent and o.parent.name == 'Head' and o.type == 'MESH']
    for o in [o for o in bpy.data.objects if o.type == 'MESH' and o.data.materials
              and o.data.materials[0].name == 'skin']:
        me = o.data
        co = np.array([tuple(v.co) for v in me.vertices])
        nr = np.array([tuple(v.normal) for v in me.vertices])
        n = len(co)
        flush, shade, gloss, cold = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        freck, crease, pores = np.zeros(n), np.zeros(n), np.full(n, 0.3)
        thin = np.zeros(n)       # how thin the skin is, so how much of the veins under it shows (skin_layers.py)
        if o.name == 'Head shape':
            u, w = co[:, 0] / HEAD_H, co[:, 2] / HEAD_H
            front = np.clip(-nr[:, 1] * 1.5, 0, 1)
            flush = front * (0.95 * (_g(u, w, -0.2, 0.365, 0.105, 0.06) + _g(u, w, 0.2, 0.365, 0.105, 0.06))
                             + 0.8 * _g(u, w, 0, 0.35, 0.05, 0.045) + 0.35 * _g(u, w, 0, 0.085, 0.07, 0.05)
                             + 0.3 * _g(u, w, 0, 0.66, 0.12, 0.05))
            cold = front * 0.8 * (_g(u, w, -0.14, 0.47, 0.065, 0.025) + _g(u, w, 0.14, 0.47, 0.065, 0.025))
            cold += np.clip(-nr[:, 2] - 0.25, 0, 1) * np.clip((0.3 - w) / 0.2, 0, 1) * 1.1   # under the jaw
            cold += np.clip((0.02 - w) / 0.1, 0, 1) * 0.7        # and down the throat, in the head's shade
            gloss = front * (_g(u, w, 0, 0.43, 0.04, 0.15) + 0.7 * _g(u, w, 0, 0.085, 0.06, 0.05)
                             + 0.5 * _g(u, w, 0, 0.66, 0.15, 0.06))
            # texture: freckles in a band over the nose and the tops of the cheeks, a few on the brow; pores on
            # the nose, the cheeks beside it, the chin and the brow
            thin = front * np.clip(0.9 * (_g(u, w, -0.215, 0.5, 0.045, 0.075) + _g(u, w, 0.215, 0.5, 0.045, 0.075))
                                   + 0.7 * _g(u, w, 0, 0.5, 0.03, 0.06) + 0.8 * cold, 0, 1)    # the temples, the bridge of the nose, under the eyes
            freck = front * np.clip(_g(u, w, 0, 0.42, 0.3, 0.075) + 0.3 * _g(u, w, 0, 0.68, 0.2, 0.06), 0, 1)
            pores = np.clip(0.3 + front * (gloss + 0.7 * (_g(u, w, -0.1, 0.39, 0.09, 0.07) + _g(u, w, 0.1, 0.39, 0.09, 0.07))), 0, 1)
            occ = _occlusion([o.matrix_world @ Vector(c) for c in co],
                             [(o.matrix_world.to_3x3() @ Vector(c)).normalized() for c in nr],
                             world_tree(head_parts))
        elif o.name in EAR_TONES:
            # the ears: rosier than the face, most of all round the rim and in the lobe, and dark in the bowl,
            # the groove inside the rim and the fold behind (see ear_tones); a faint crease along the rim's roll
            flush, shut = EAR_TONES[o.name]
            thin = np.full(n, 0.6)
            gloss = 0.15 * (1 - shut)
            crease = 0.3 * shut
            occ = np.maximum(shut, _occlusion([o.matrix_world @ Vector(c) for c in co],
                                              [(o.matrix_world.to_3x3() @ Vector(c)).normalized() for c in nr],
                                              world_tree(head_parts), dist=0.8))
        elif o.name == 'Neck':
            t = np.clip((co[:, 2] - 51.0) / 2.5, 0, 1)        # 0 at the collar, 1 where it meets the head's throat
            thin = np.clip(0.5 * np.abs(nr[:, 0]), 0, 1)     # the sides of the neck
            cold = 0.35 + 0.5 * t ** 2       # the neck sits in the head's shade: cooler than the face throughout
            crease = np.full(n, 0.22)        # the faint rings round a neck
            occ = _occlusion([o.matrix_world @ Vector(c) for c in co],
                             [(o.matrix_world.to_3x3() @ Vector(c)).normalized() for c in nr],
                             world_tree(['Neck', 'Head shape', 'Hair']), dist=2.0)
        elif o.name.startswith('Arm.'):
            # a bare arm (see bare_arm; local height from the shoulder joint): the plain skin of the hand's back,
            # a touch warmer on the outside, which catches the light, and cooler on the inside, toward the body,
            # with a faint warmth over the point of the elbow and the thin cool skin coming down to the wrist
            s = 1 if o.name.endswith('R') else -1
            z = co[:, 2]
            outside = np.clip(-s * nr[:, 0], 0, 1)
            inside = np.clip(s * nr[:, 0], 0, 1)
            flush = 0.15 * outside + 0.35 * np.exp(-((z + UPPER) / 0.9) ** 2) * np.clip(nr[:, 1], 0, 1)
            cold = 0.3 * inside + 0.3 * np.clip((-z - UPPER - FORE + 2.0) / 2.0, 0, 1)
            crease = 0.2 * np.exp(-((z + UPPER) / 0.4) ** 2) * np.clip(-nr[:, 1], 0, 1)   # inside the elbow
            thin = np.clip(0.5 * inside + 0.8 * np.clip((-z - UPPER - FORE + 2.0) / 2.0, 0, 1), 0, 1)   # the inside of the arm, the wrist
            pores = np.full(n, 0.08)     # (fine-grained skin: the face's pores read as red grain on an arm)
            local = BVHTree.FromPolygons([v.co for v in me.vertices], [list(p.vertices) for p in me.polygons])
            occ = _occlusion(co, nr, local, dist=0.9)
        else:      # a hand: the wrist at the origin, the fingers down; the back of the hand faces away from her
            s = 1 if o.name.endswith('R') else -1
            z = co[:, 2]
            back = np.clip(-s * nr[:, 0] * 1.4, 0, 1)
            # the knuckles: a rosy bump over each finger's root and middle joint (fingers index to little at
            # y -0.9 to 0.78 before the hand is scaled by 1.22), not a band across the hand
            rows = sum(np.exp(-((co[:, 1] - 1.22 * y0) / 0.24) ** 2) for y0 in (-0.9, -0.3, 0.3, 0.78))
            flush = (back * rows * (0.9 * np.exp(-((z + 3.15) / 0.34) ** 2) + 0.7 * np.exp(-((z + 4.5) / 0.3) ** 2))
                     + 0.9 * np.clip((-z - 5.3) / 0.8, 0, 1))
            flush += 0.35 * np.clip(s * nr[:, 0], 0, 1) * np.clip((-z - 0.5) / 2, 0, 1)   # the palm is pinker
            cold = 0.45 * np.clip(1 - np.abs(z + 0.2) / 1.2, 0, 1) * back                  # the thin skin at the wrist
            gloss = 0.3 * np.clip((-z - 5.3) / 0.8, 0, 1)
            thin = np.clip(0.8 * back + 0.9 * np.clip(1 - np.abs(z + 0.2) / 1.5, 0, 1), 0, 1)   # the back of the hand and the wrist
            # texture: creases over each knuckle (on the back) and fainter all round the fingers' joints, a few
            # freckles on the back of the hand
            joints = np.exp(-((z + 3.15) / 0.4) ** 2) + np.exp(-((z + 4.5) / 0.34) ** 2)
            crease = np.clip(joints * (0.12 + 0.88 * back * rows), 0, 1)
            freck = 0.5 * back * np.clip((-z - 0.3) / 1.0, 0, 1) * np.clip((z + 3.4) / 0.8, 0, 1)
            local = BVHTree.FromPolygons([v.co for v in me.vertices], [list(p.vertices) for p in me.polygons])
            occ = _occlusion(co, nr, local, dist=0.9)
        flush, cold, gloss = np.clip(flush, 0, 1), np.clip(cold, 0, 1), np.clip(gloss, 0, 1)
        c = base[None, :] * np.ones((n, 1))
        c = c + (flushc - base)[None, :] * (0.6 * flush)[:, None]
        c = c + (cool - c) * (0.5 * cold)[:, None]
        # shut in: darker and warmer (a reddened shade rather than a grey one)
        k = np.clip(occ * 1.3, 0, 1)[:, None]
        c = c * (1 - 0.45 * k) + deep[None, :] * 0.18 * k
        if o.name == 'Neck':
            c *= 0.9      # (and a shade darker than the face, which it sits under)
        shade = np.clip(0.9 * occ + 0.6 * cold, 0, 1)
        # (the colour also drifts a little from place to place, redder or sallower: broad enough to show in the
        # world model, whose skin is only this colour)
        from mathutils import noise
        drift = np.array([noise.noise(Vector(q) * 0.75 + Vector((3.1, 7.7, 1.3))) for q in rest_of(o, co)])
        c = c + (_srgb_lin([206, 124, 110]) - c) * np.clip(drift, 0, 1)[:, None] * 0.3
        c = c + (_srgb_lin([212, 176, 128]) - c) * np.clip(-drift, 0, 1)[:, None] * 0.3
        for name, rgb in (('skin', c), ('skin fx', np.stack([flush, shade, gloss], 1)),
                          ('skin tex', np.stack([freck, crease, pores], 1))):
            a = me.color_attributes.new(name, 'FLOAT_COLOR', 'POINT')
            alpha = np.clip(thin, 0, 1)[:, None] if name == 'skin tex' else np.ones((n, 1))      # ('skin tex' alpha: thin skin)
            a.data.foreach_set('color', np.concatenate([rgb, alpha], 1).astype(np.float32).ravel())
        me.color_attributes.active_color = me.color_attributes['skin']
        me.color_attributes.render_color_index = me.color_attributes.find('skin')


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
HEAD_NECK_Z = -0.13                      # the head's sculpt runs down the neck to here (inside the collar), in head heights
HIP_Z, SH_Z, ANKLE_Z = 31.0, 49.0, 3.4
# The toes' joint (the metatarsophalangeal axis) of a foot with a toe bone: (how far ahead of the ankle, how high), or None
# for the feet whose toes are part of the foot (the boots' lasts, the painted sheets). anatomy.patch sets it.
TOE_JOINT = None
# A heel under the foot (the heel study's walk): how far the foot is held pitched toes-down in the walk, radians. 0 for a bare
# foot or a flat sole. heel.build sets it, with the figure raised by the sole's thickness.
HEEL_PITCH = 0.0
# and the points of its sole that meet the floor, (forward of the ankle, height over it) in the foot's own frame: the
# heel's rear tread edge and the ball under the sole. HEEL_UP is how far the figure is raised on the sole. With these
# set, locked_foot stands the ankle at the height that puts the lowest of those points on the floor at the foot's pitch.
HEEL_CONTACT = None
HEEL_AXIS = None      # the toe joint (forward of the ankle, height over it) and how far the sole's bottom hangs under it
HEEL_UP = 0.0
THIGH, SHIN = 14.0, 13.8            # hip -> knee -> ankle, a touch longer than straight so the knee stays soft
UPPER, FORE = 9.8, 9.8              # the elbow halfway down the arm, level with the waist
LEG_X, SH_X = 3.3, 5.75
# the trunk's two pivots for the walk: the pelvis turns about the hip joints, the chest about the waist above it
CHEST_Z, NECK_Z = 41.0, 51.0
# The neck below the head's sculpt: rings (z, rx, ry, cx, cy) from where it meets the head's throat down into the
# collar. NECK_Z, its bottom, is the base of the neck at the front (the pit of the neck), 0.37 head heights under the
# chin, as in a real body (ANSUR II women: suprasternale 0.36 head heights under the menton).
NECK_RINGS = [(53.5, 1.4, 1.6, 0, 0.43), (53.0, 1.75, 1.85, 0, 0.3), (NECK_Z, 2.3, 2.15, 0, 0.2)]

# ---- the neck's two joints (John, 2026-11-19: "why can't the head bend"). The head used to hang off the body's
# frame on one pivot at the chin, so it never rode the spine (it stayed behind when the chest leaned, and the neck
# tore open in the crouch) and it could only spin about a point no joint is at. A real neck bends in two places
# that move differently, and both are placed here from the measured head, not by eye:
#  - ATLAS: the upper cervical joints (the skull rocking on the atlas, the atlas turning on the axis) pivot at the
#    occipital condyles, which sit straight under the ear's opening at the tip of the mastoid, level with the lobe
#    of the ear;
#  - CERVICAL: the lower cervical spine (C2 to T1) bends about the C7-T1 disc at the base of the neck, in the
#    vertebral bodies, which lie behind the neck's middle. Its height is halfway between the base of the neck at the
#    back (the C7 spine, 0.12 head heights under the chin: ANSUR II women, cervicale 2.4 cm under the menton) and at
#    the front (NECK_Z, the pit of the neck), since the base of the neck slopes down to the front.
# NECK_SHARE: the upper joints' share of each motion (nod, tilt, turn), from the ranges per level (White & Panjabi;
# Panjabi's in vivo flexion and extension of the upper joints): bending forward the skull rocks little on the atlas
# (occiput-C1 3.5, C1-C2 11.5 degrees: 15 of about 50), bending back far more (21 and 11: 32 of about 62), lateral
# bend 10 of 61, axial rotation 45 of 77. So a turn of the head is mostly the atlas, a tilt almost all the lower
# neck, and looking up tips the skull back on the atlas instead of bowing the whole neck.
NECK_SHARE = (15 / 50, 10 / 61, 45 / 77)
NECK_SHARE_EXT = 32 / 62
# The rest of each motion is the lower cervical spine's, which bends along its length, not at one point: a third
# joint at the C3-C4 disc takes the share of the C2-C3 and C3-C4 levels (White & Panjabi, of C2-T1's): flexion and
# extension 25 of 91 degrees, lateral bend 21 of 51, rotation 10 of 32. The C7 joint takes the rest. So a nod or a
# tilt curves down the neck instead of leaning it over as a straight column from its base.
MID_SHARE = (25 / 91, 21 / 51, 10 / 32)
C34_ALONG = 4 / 7       # the C3-C4 disc up the column from C7-T1 to the atlas: C7, C6, C5, C4 below it; C3, C2, C1 above
LORDOSIS = 0.04         # how far forward of the line from C7 to the atlas the mid-neck's vertebrae lie, head heights
NECK_ROM = {'flex': 50, 'ext': 60, 'tilt': 45, 'turn': 80}    # the neck's active range, degrees each way
NECK_PIT = 0.36                      # the pit of the neck under the chin, head heights (ANSUR II women: suprasternale)
C7_BACK = 0.12                       # the C7 spine under the chin, head heights (ANSUR II women: cervicale)
VERTEBRA_BACK = 0.38                 # the vertebral bodies' centre, as a share of the neck's depth from the back
NECK = {}                            # the joints as measured on the head the figure wears: see place_neck
# the walk's range of motion: GAIT scales the swing of the hips (and so the stride), ARM_SWING the arms' swing
# either way, in radians (the world model takes a longer stride: see export_glb.py)
GAIT, ARM_SWING = 1.0, 0.2


# The face's shape, as the face creator's sliders (/mnt/project-files/face-creator): multipliers on face_relief's,
# mouth_relief's and jaw_relief's forms, and offsets on the head's width and a few features' heights. These defaults
# are the ranger's face; a figure changes the ones it wants with FACE.update({...}) before it builds.
# A mannequin: the same body and the same head shape, but the head left plain: no eyes, lids, ears, face paint or
# mouth (and each figure skips its hair, see its build). The clothes, armour and gear build and fit exactly as on the
# real figure, in a fraction of the time. For checking clothes that need a body under them (views.py --mannequin).
MANNEQUIN = False
EARS = False       # whether the head gets its sculpted ears (see ear_mesh); off for the ranger, whose hair covers them
FACE = {
    'jawWidth': 0.0, 'jawDef': 1.0, 'jawSharp': 1.0, 'jawAngle': 0.0, 'chinProj': 1.0, 'chinWidth': 1.0, 'chinHeight': 1.0,
    'cheekWidth': 0.0, 'cheekbones': 1.0, 'cbHeight': 0.0, 'apples': 1.0, 'cheekFull': 1.0,
    'noseProj': 1.0, 'noseWidth': 1.0, 'noseLen': 0.0, 'tipBall': 1.0, 'nostrils': 1.0,
    'browRidge': 1.0, 'sockets': 1.0, 'mouthWidth': 1.0, 'lipUpper': 1.0, 'lipLower': 1.0, 'smile': 1.0,
    # (the face creator's: how the jaw runs in to the chin, + a V, - held wide to a square chin; how softly the jaw
    # falls into the neck; the eyes' half-spacing, head heights)
    'chinTaper': 0.0, 'jawUnder': 0.0, 'eyeSpacing': 0.14,
}


def gauss(x, z, x0, z0, sx, sz, amp):
    return amp * math.exp(-((x - x0) / sx) ** 2 - ((z - z0) / sz) ** 2)


def face_relief(x, z):
    """How far the face stands forward of the base head at (x across, z up), in head heights. The planes a
    painter would block in: brow ridge over the eye sockets, the nose (bridge, tip and wings), cheekbones with
    the hollow under them, the lips parted by the mouth line, and the ball of the chin. (They are pushed a little
    past life size, so they still read when the head is 17 pixels tall.)
    The features below the eyes are laid out on a slightly taller lower face and drawn up into the head's
    proportions by FACE_WARP (sculpt round 2)."""
    z = FACE_WARP(z)
    d = 0.0
    F = FACE
    nw, nl, cb = F['noseWidth'], F['noseLen'], F['cbHeight']
    es = F['eyeSpacing'] / 0.14
    for s in (-1, 1):
        # brow ridge: falling away toward the middle and easing down into the sockets (a ridge that ran on across the
        # bridge as a shelf threw a band of shade between the eyes)
        d += gauss(x, z, s * 0.15 * es, 0.62, 0.075 if x * s < 0.15 * es else 0.12, 0.052, 0.03 * F['browRidge'])
        d += gauss(x, z, s * 0.14 * es, 0.525, 0.08, 0.05, -0.06 * F['sockets'])     # eye socket
        d += gauss(x, z, s * 0.23, 0.415 + cb, 0.09, 0.06, 0.028 * F['cheekbones'])    # cheekbone, running back toward the ear
        d += gauss(x, z, s * 0.145, 0.375 + 0.5 * cb, 0.07, 0.06, 0.04 * F['apples'])    # the apple of the cheek, under the eye by the nose
        d += gauss(x, z, s * 0.19, 0.28, 0.11, 0.11, 0.032 * F['cheekFull'])      # the cheek filled out under the cheekbone, no hollow
        d += gauss(x, z, s * 0.18, 0.2, 0.09, 0.09, 0.016 * F['cheekFull'])       # the soft fullness of the lower cheek
        d += gauss(x, z, s * 0.05 * nw, 0.37 + nl, 0.021 * nw, 0.02, 0.018 * F['nostrils'])    # wing of the nose (smaller: it read big from the game's camera)
        d += gauss(x, z, s * 0.037 * nw, 0.378 + nl, 0.008, 0.02, -0.004 * F['nostrils'])  # the groove between the wing and the tip
        d += gauss(x, z, s * 0.235 * es, 0.53, 0.035, 0.085, 0.016)   # the outer rim of the eye socket, brow to cheekbone
    # the nose: a narrow ridge rising from the bridge to the tip, then turning under sharply. Its cross-section is
    # a flat-topped ridge (the dorsum) with steep sides, narrow at the bridge and widening toward the tip (round 9).
    zt = 0.36 + nl
    if z > zt:
        t = min(1.0, max(0.0, (0.6 - z) / (0.6 - zt)))
        h = (0.012 + 0.04 * t ** 1.6) * math.exp(-max(0.0, z - 0.58) ** 2 / 0.0008)
    else:
        h = 0.05 * math.exp(-((zt - z) / 0.028) ** 2)    # (turning under softly, so no hard shadow under the tip)
    h *= F['noseProj']
    w = (0.026 + 0.01 * min(1.0, max(0.0, (0.56 - z) / 0.18)) + 0.012 * max(0.0, 1 - abs(z - (0.37 + nl)) / 0.06)) * nw
    d += h * math.exp(-(x / (1.15 * w)) ** 2)      # (a rounded ridge: steep sides cast a hard shadow line from above)
    d += gauss(x, z, 0, 0.635, 0.07, 0.03, 0.012)    # the glabella between the brows, over the dip at the nose's root
    d += gauss(x, z, 0, 0.27, 0.16, 0.1, 0.018)      # the muzzle: the teeth and jaw carry the mouth forward
    d += mouth_relief(x / F['mouthWidth'], z)
    ch = F['chinHeight']
    d += gauss(x, z, 0, 0.088 * (0.6 + 0.4 * ch), 0.046 * F['chinWidth'], 0.044 * ch, 0.028 * F['chinProj'])   # chin: the ball of it, set back behind the lower lip
    for s in (-1, 1):
        d += gauss(x, z, s * 0.09 * F['chinWidth'], 0.11, 0.045, 0.06, -0.003)    # a soft dip either side, setting the chin off the jaw
    return d


def mouth_relief(x, z):
    """The mouth and the forms around it (sculpt round 2), in head heights forward of the face: the upper lip
    with its cupid's bow and a small tubercle in the middle, a fuller lower lip, the line between them sinking
    into the corners, the philtrum's two ridges up to the nose, the ball of the nose's tip, the fold from the
    nose's wing toward the corner of the mouth, and the crease under the lower lip."""
    ax = abs(x)
    d = 0.0
    across_u = math.exp(-(x / 0.07) ** 4)                 # the upper lip runs nearly full height to its corners
    bow = 0.006 * math.exp(-((ax - 0.019) / 0.011) ** 2)   # the peaks of the cupid's bow
    top = 0.296 + bow - 0.012 * (ax / 0.082) ** 2
    t = (z - 0.258) / max(0.01, top - 0.258)
    if 0.0 < t < 1.25:
        d += 0.016 * FACE['lipUpper'] * across_u * math.sin(min(1.0, t) * math.pi * 0.9) ** 0.8 * (1.0 if t <= 1.0 else max(0.0, 1 - (t - 1) * 4))
    d += gauss(x, z, 0, 0.266, 0.02, 0.01, 0.005 * FACE['lipUpper'])          # the tubercle
    across_l = math.exp(-(x / 0.06) ** 4)
    d += 0.017 * FACE['lipLower'] * across_l * math.exp(-((z - 0.239) / 0.016) ** 2)    # lower lip (less full: it read pouty)
    # the line between the lips, deepest at the corners where the lips tuck in
    # (the line lifts a touch toward the corners, a relaxed half-smile: level, from above, it read as a frown)
    zl = 0.258 + 0.007 * FACE['smile'] * (ax / 0.082) ** 2
    d -= (0.008 + 0.003 * (ax / 0.082) ** 2) * math.exp(-((z - zl) / 0.006) ** 2) * math.exp(-(x / 0.09) ** 6)
    for sx in (-1, 1):
        # (shallow: deeper, each corner was a pit that the game's high camera saw as a dark wedge)
        d += gauss(x, z, sx * 0.086, 0.263, 0.013, 0.012, -0.002)    # the corners, lifted
        d += gauss(x, z, sx * 0.022, 0.33, 0.01, 0.03, 0.006)        # philtrum ridges
        # the fold from the wing of the nose round to beside the corner of the mouth (a crease with a soft roll
        # of cheek just outside it)
        for k in range(6):
            u = k / 5
            px, pz = sx * (0.07 + 0.032 * u), 0.355 - 0.1 * u
            d += gauss(x, z, px, pz, 0.014, 0.02, -0.0015 * (1 - 0.5 * u))
            d += gauss(x, z, px + sx * 0.022, pz, 0.018, 0.022, 0.002 * (1 - 0.5 * u))
    d += gauss(x, z, 0, 0.385 + FACE['noseLen'], 0.028 * FACE['noseWidth'], 0.024, 0.006 * FACE['tipBall'])        # the ball of the nose's tip
    d += gauss(x, z, 0, 0.19, 0.05, 0.014, -0.008)         # the crease under the lower lip
    return d


def smooth_curve(knots):
    """A smooth curve through (z, value) knots: monotone cubic (Fritsch-Carlson), so it never overshoots between
    them and leaves no kink at a knot the way straight segments would."""
    zs = [k[0] for k in knots]
    vs = [k[1] for k in knots]
    n = len(zs)
    d = [(vs[i + 1] - vs[i]) / (zs[i + 1] - zs[i]) for i in range(n - 1)]
    m = [d[0]] + [0.0 if d[i - 1] * d[i] <= 0 else (d[i - 1] + d[i]) / 2 for i in range(1, n - 1)] + [d[-1]]
    for i in range(n - 1):
        if d[i] == 0:
            m[i] = m[i + 1] = 0.0
            continue
        a, b = m[i] / d[i], m[i + 1] / d[i]
        if a * a + b * b > 9:
            t = 3 / math.sqrt(a * a + b * b)
            m[i], m[i + 1] = t * a * d[i], t * b * d[i]

    def f(z):
        if z <= zs[0]:
            return vs[0]
        if z >= zs[-1]:
            return vs[-1]
        i = max(k for k in range(n - 1) if zs[k] <= z)
        h = zs[i + 1] - zs[i]
        t = (z - zs[i]) / h
        h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
        return h00 * vs[i] + h10 * h * m[i] + h01 * vs[i + 1] + h11 * h * m[i + 1]
    return f


# ---- the head's construction (sculpt rounds): the skull and jaw as a stack of horizontal sections, each a
# rounded box (a superellipse) with its own half-width, front, back and widest point, read off smooth curves of
# height. In head heights: chin at z 0, crown at z 1, the face toward -y.
#   below z 0.2 the sections are the jaw: each ends at the front in the chin and at the back in the underside of
#   the jaw, which rises from the chin to the angle of the jaw under the ear;
#   above, the cranium: widest across the parietals at z 0.7, deepest at the occiput behind, the forehead sloping
#   back from the brow; over the top the sections round off along ellipses into the crown.
HEAD_W = smooth_curve([(HEAD_NECK_Z, 0.2), (-0.07, 0.172), (-0.02, 0.15), (0.03, 0.122), (0.07, 0.156), (0.12, 0.197),
                       (0.18, 0.236), (0.25, 0.268), (0.32, 0.296), (0.41, 0.323), (0.52, 0.342), (0.62, 0.353), (0.7, 0.36)])
# (the front: the throat rising to the soft turn under the chin, the floor under the chin running forward nearly
# level, then the chin and the face)
HEAD_F = smooth_curve([(HEAD_NECK_Z, 0.165), (-0.07, 0.14), (-0.035, 0.12), (-0.012, 0.15), (0.005, 0.2), (0.025, 0.268),
                       (0.05, 0.315), (0.08, 0.352), (0.12, 0.376), (0.18, 0.4), (0.25, 0.41), (0.33, 0.415), (0.42, 0.42),
                       (0.52, 0.42), (0.62, 0.42)])
HEAD_B = smooth_curve([(HEAD_NECK_Z, 0.24), (-0.07, 0.245), (0.05, 0.27), (0.2, 0.31), (0.3, 0.35), (0.38, 0.385),
                       (0.46, 0.415), (0.54, 0.435), (0.64, 0.445)])
HEAD_MID = smooth_curve([(HEAD_NECK_Z, 0.045), (0.3, 0.04), (1.0, 0.04)])
# how square each section is (2 is an ellipse): the face is a broad, fairly flat front turning at the cheekbones,
# narrowing to a rounder chin; the cranium a little squarer than an egg through the temples
HEAD_NF = smooth_curve([(HEAD_NECK_Z, 2.0), (-0.02, 2.0), (0.06, 2.1), (0.14, 2.02), (0.3, 1.98), (0.45, 2.12), (0.6, 2.25), (0.8, 2.2)])
HEAD_NB = smooth_curve([(HEAD_NECK_Z, 2.0), (0.3, 2.1), (0.6, 2.3), (0.85, 2.1)])


# the lower face, nose base to chin, as a third of the head rather than the 0.36 the features were first laid out
# on: the face's relief is read at FACE_WARP(z), which draws the nose up to z 0.335, the mouth to 0.232 and the
# ball of the chin to 0.09, leaving the eyes and brows where they are
FACE_WARP = smooth_curve([(0.0, 0.0), (0.09, 0.1), (0.232, 0.258), (0.335, 0.36), (0.45, 0.455), (0.6, 0.6), (1.0, 1.0)])
MOUTH_Z = 0.232


def _dome(f, z, z0, top=1.0):
    """f below z0; above it the value rounds off along a quarter ellipse to nothing at the crown."""
    if z <= z0:
        return f(z)
    t = min(1.0, (z - z0) / (top - z0))
    return f(z0) * math.sqrt(max(0.0, 1 - t * t))


def head_section(z, c, s):
    """The point of the base head (no face relief) at height z and direction (c forward, s across) round it.
    Below the chin the sections are the neck (down to HEAD_NECK_Z, inside the collar), so the throat, the floor under
    the chin and the jaw's underside are one surface with the face: the jaw's edge is a soft step pressed into it
    (see jaw_relief), not a separate shape set on top of a neck. At HEAD_NECK_Z the sections close off in a quarter
    ellipse, out of sight inside the neck's lower part and the collar."""
    if z < HEAD_NECK_Z + 0.02:
        k = math.sqrt(max(0.0, 1 - ((HEAD_NECK_Z + 0.02 - z) / 0.02) ** 2))
        x, y = _section(max(HEAD_NECK_Z, z), c, s)
        yw = HEAD_MID(z)
        return x * k, yw + (y - yw) * k
    return _section(z, c, s)


RAMUS_BACK = 0.11      # the back edge of the jaw's ramus, head heights from the head's middle (see jaw_relief)


def jaw_edge(y):
    """The jaw's lower edge in profile, head heights above the chin at y (head heights, face toward -y): one curve,
    level under the chin and bending up smoothly to the angle under the ear (round 9)."""
    return 0.17 * min(1.0, max(0.0, (y + 0.34) / 0.4)) ** 1.7


def jaw_relief(y, z):
    """How far the jaw stands out of the neck (head heights, along the surface normal): 1 over the jaw, fading to
    0 across its lower edge (from the bottom of the chin back and up to the angle under the ear) and its back edge
    (the ramus), and fading out above into the cheek, which the sections already shape."""
    zl = jaw_edge(y)
    def ss(t):
        t = min(1.0, max(0.0, t))
        return t * t * (3 - 2 * t)
    js = FACE['jawSharp']                      # (narrows the turn over the lower and back edges: a crisper jawline)
    tail = FACE['jawUnder']                    # (lets the jaw fall softly into the neck, with no crease)
    below = ss((z - zl + 0.03 / js + tail) / (0.1 / js + tail))        # above the lower edge of the jaw
    behind = ss((RAMUS_BACK + tail - y) / (RAMUS_BACK / (1 + 0.3 * (js - 1)) + tail))   # in front of the back edge of the ramus (kept soft: crisp, it read as a step)
    up = ss((zl + 0.26 - z) / 0.26)                 # under the cheek
    # the angle of the jaw under the ear, where the lower edge turns up into the ramus: a firm corner on a
    # defined jaw (the masseter over it), set inside both edges so they stay clean
    angle = FACE['jawAngle'] * math.exp(-((y - 0.06) / 0.065) ** 2 - ((z - 0.19) / 0.07) ** 2) * below * behind
    return 0.02 * FACE['jawDef'] * below * behind * up + 0.016 * angle


def _head_w(z):
    """HEAD_W with the face creator's jaw and face width (FACE['jawWidth'], FACE['cheekWidth'])."""
    def ss(e0, e1, v):
        t = min(1.0, max(0.0, (v - e0) / (e1 - e0)))
        return t * t * (3 - 2 * t)
    jw = ss(-0.05, 0.08, z) * ss(0.42, 0.2, z)
    cw = math.exp(-((z - 0.42) / 0.12) ** 2)
    # (chinTaper works only below the mouth, where the jaw narrows, so the cheeks keep their width)
    ct = math.exp(-((z - 0.1) / 0.075) ** 2) * ss(-0.06, 0.03, z)
    return HEAD_W(z) * (1 + FACE['jawWidth'] * jw + FACE['cheekWidth'] * cw - 0.2 * FACE['chinTaper'] * ct)


def _section(z, c, s):
    w = _dome(_head_w, z, 0.7)
    yw = HEAD_MID(z)
    front = _dome(HEAD_F, z, 0.62) + yw
    back = _dome(HEAD_B, z, 0.66) - yw
    n = HEAD_NF(z) if c >= 0 else HEAD_NB(z)

    def sp(t):
        return math.copysign(abs(t) ** (2 / n), t)
    x = w * sp(s)
    y = yw - front * sp(c) if c >= 0 else yw + back * sp(-c)
    return x, y


EAR_TONES = {}     # each ear's (flush, shade) per point, for skin_tones (see ear_mesh)


def ear_mesh(e, name):
    """An ear's grid (ear_grids) as a mesh: rows round the outline joined end to end, the bowl's centre one point,
    the rear face left open where it runs into the skull. Its colour per point goes to EAR_TONES[name]."""
    P = e['P']
    NT, NC = P.shape[:2]
    bm = bmesh.new()
    pole = bm.verts.new(tuple(P[:, 0].mean(0)))
    vs = [[bm.verts.new(tuple(p)) for p in row[1:]] for row in P]
    for i in range(NT):
        a, b = vs[i], vs[(i + 1) % NT]
        bm.faces.new((pole, a[0], b[0]))
        for j in range(NC - 2):
            bm.faces.new((a[j], b[j], b[j + 1], a[j + 1]))
    # (outward: the outer face looks away from the head; the faces' winding, row to row then out along a row)
    j = NC // 4
    wind = np.cross(np.roll(P, -1, axis=0)[:, j] - P[:, j], P[:, j + 1] - P[:, j])[:, 0].sum()
    if wind * e['side'] < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    flush, shade = ear_tones(e)
    EAR_TONES[name] = tuple(np.concatenate([[t[:, 0].mean()], t[:, 1:].ravel()]) for t in (flush, shade))
    return bm


def sculpted_head():
    """The head as a sculpted mesh: the skull and jaw built up from sections (see head_section), then the face's
    planes pushed out of the front (see face_relief). Units are head heights: chin at z 0, crown at z 1, the face
    toward -Y."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=160, v_segments=256, radius=1.0)
    for v in bm.verts:
        sx, sy, sz = v.co
        z = HEAD_NECK_Z + (1.0 - HEAD_NECK_Z) * (0.5 + 0.5 * sz)
        r = math.hypot(sx, sy)
        c, s = (-sy / r, sx / r) if r > 1e-9 else (1.0, 0.0)
        x, y = head_section(z, c, s)
        # the face is pushed out along the base surface's normal, and only as far as that normal faces forward:
        # where the surface turns under (the chin, the jaw's underside) the relief turns with it and fades, so the
        # jawline stays a rounded turn instead of a ledge sheared forward of the underside
        th, e = math.atan2(s, c), 1e-3
        za, zb = min(1.0, z + e), max(HEAD_NECK_Z, z - e)
        xa, ya = head_section(z, math.cos(th + e), math.sin(th + e))
        xz, yz = head_section(za, c, s)
        xb, yb = head_section(zb, c, s)
        t1 = Vector((xa - x, ya - y, 0.0))
        t2 = Vector((xz - xb, yz - yb, za - zb))
        n = t1.cross(t2)
        if n.length < 1e-12:
            v.co = (x, y, z)
            continue
        n.normalize()
        if n.x * x + n.y * (y - 0.04) < 0:
            n = -n
        f = max(0.0, -n.y) ** 1.0       # only the parts facing forward carry the face
        u = min(1.0, max(0.0, (z - 0.005) / 0.04))
        d = face_relief(x, z) * f * u * u * (3 - 2 * u) + jaw_relief(y, z)
        v.co = (x + n.x * d, y + n.y * d, z + n.z * d)
    bmesh.ops.scale(bm, vec=(HEAD_H, HEAD_H, HEAD_H), verts=bm.verts)
    return bm


def head_front_y(x, z):
    """The front of the sculpted head (see sculpted_head) at (x across, z up), in head heights, worked out
    exactly from its formula rather than from the mesh, or None off the head. (The eyes' sculpt puts the finer
    mesh it cuts round the eyes back onto this smooth surface: put onto the coarse mesh's flat faces, each one
    showed as a flat facet, a staircase down the bridge of the nose.)"""
    if 'head_section' in globals():
        return _head_front_y_sections(x, z)
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


def _head_point(z, th):
    """A point of the head built from sections (head_section, face_relief, jaw_relief) at height parameter z and
    angle th round it (0 straight forward): the same displacement sculpted_head applies to each vertex."""
    c, s = math.cos(th), math.sin(th)
    x, y = head_section(z, c, s)
    e = 1e-3
    za, zb = min(1.0, z + e), max(HEAD_NECK_Z, z - e)
    xa, ya = head_section(z, math.cos(th + e), math.sin(th + e))
    xz, yz = head_section(za, c, s)
    xb, yb = head_section(zb, c, s)
    n = Vector((xa - x, ya - y, 0.0)).cross(Vector((xz - xb, yz - yb, za - zb)))
    if n.length < 1e-12:
        return x, y, z
    n.normalize()
    if n.x * x + n.y * (y - 0.04) < 0:
        n = -n
    f = max(0.0, -n.y)
    u = min(1.0, max(0.0, (z - 0.005) / 0.04))
    d = face_relief(x, z) * f * u * u * (3 - 2 * u) + jaw_relief(y, z)
    return x + n.x * d, y + n.y * d, z + n.z * d


def _head_front_y_sections(x, z):
    """head_front_y for the head built from sections: the surface is a function of (height, angle), not of
    (x, z), so solve for the parameters whose point lands on (x, z) (Newton's method, a few steps from the
    undisplaced guess) and return that point's depth."""
    zp, th = z, math.asin(max(-0.95, min(0.95, x / 0.36)))
    for _ in range(12):
        px, py, pz = _head_point(zp, th)
        fx, fz = px - x, pz - z
        if abs(fx) < 1e-6 and abs(fz) < 1e-6:
            break
        h = 1e-4
        ax, _, az = _head_point(zp, th + h)
        bx, _, bz = _head_point(zp + h, th)
        j11, j12 = (ax - px) / h, (bx - px) / h      # d x / d th, d x / d z
        j21, j22 = (az - pz) / h, (bz - pz) / h
        det = j11 * j22 - j12 * j21
        if abs(det) < 1e-12:
            return None
        th -= (fx * j22 - fz * j12) / det
        zp -= (fz * j11 - fx * j21) / det
        if abs(th) > 1.5:
            return None
    else:
        return None
    return py


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
    # (the distance round is taken at one fixed radius: scaled by each point's own distance from the axis, the
    # streaks bent into arcs wherever the hair swells out, as over the side locks)
    return (math.atan2(co.x, co.y - 0.5) * 3.6, 0.0, co.z * 0.6)


def strand_colours(o, mid=M_HAIR):
    """Each strand its own tone: the hair's colour attribute 'strands' (linear RGB), a mix of the darker, middle
    and lighter tones of its palette ramp, streaked along the fall of the hair at three widths (single strands,
    small locks, broad bands), a little darker on the underside of the fall, where it faces her back."""
    from mathutils import noise
    rmp = MATERIALS[mid]['ramp'] / 255.0
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


# the sleeve's radius down the arm (local height from the shoulder joint, radius), before the deltoid and folds
ARM_PROFILE = [(1.6, 0.3), (1.35, 1.3), (0.85, 1.95), (0.0, 2.15), (-2.0, 2.1), (-5.5, 1.88), (-UPPER, 1.65), (-UPPER - 2.2, 1.78),
               (-UPPER - 5.0, 1.55), (-UPPER - FORE + 1.2, 1.36), (-UPPER - FORE - 0.3, 1.42), (-UPPER - FORE - 1.1, 1.5)]
# (the sleeve runs on past the wrist and opens a little, so it falls over the root of the hand: the hand shows from
# under a cuff wider than its wrist instead of hanging below a sleeve as wide as it is)


def _profile(table, z):
    if z >= table[0][0]:
        return table[0][1]
    for (za, ra), (zb, rb) in zip(table, table[1:]):
        if zb <= z <= za:
            t = (za - z) / (za - zb)
            return lerp(ra, rb, (1 - math.cos(math.pi * t)) / 2)
    return table[-1][1]


_HULL = {}


def _convex(points):
    """The convex hull of 2D points, counter-clockwise (Andrew's monotone chain)."""
    pts = sorted(set(points))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def body_hull(z, a, cy):
    """How far out from (0, cy), in the direction a (round from the front toward her right, as phi), cloth laid
    over the body reaches at height z: it spans the convex hull of the jerkin over the torso and the sleeves at the
    tops of the arms (deltoids included, each held a little out from the body as it is in her stance), so it
    bridges the hollow between the arm and the chest instead of sinking into it."""
    key = (round(z, 3), round(cy, 3))
    if key not in _HULL:
        pts = []
        for k in range(96):
            x, y = torso_point(z, 2 * math.pi * k / 96, 0.65, bust=1.0)
            pts.append((round(x, 5), round(y - cy, 5)))
        # the slope of the shoulders, from the base of the neck out to the tops of the arms (the trapezius over the
        # collarbones), so the cloth runs down it in one line instead of sinking in under the neck and rolling up
        # again over the arms in a collar
        t = min(1.0, max(0.0, (53.3 - z) / 3.0))
        t = t * t * (3 - 2 * t)
        if 0.0 < t:
            tw, td = lerp(2.0, SH_X + 0.9, t), lerp(2.1, 2.6, t)
            for k in range(48):
                u = 2 * math.pi * k / 48
                pts.append((round(tw * math.cos(u), 5), round(0.25 - cy + td * math.sin(u), 5)))
        zl = z - SH_Z
        if zl < 1.6:
            R = _profile(ARM_PROFILE, zl) + 0.42 * math.exp(-((zl + 1.4) / 2.1) ** 2) + 0.1
            for s in (-1, 1):
                cx = s * (SH_X + 0.17 * max(0.0, -zl))
                for k in range(48):
                    t = 2 * math.pi * k / 48
                    pts.append((round(cx + R * math.cos(t), 5), round(0.2 - cy + R * math.sin(t), 5)))
        _HULL[key] = _convex(pts)
    poly = _HULL[key]
    dx, dy = -math.sin(a), -math.cos(a)
    best = 0.0
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        ex, ey = x1 - x0, y1 - y0
        den = dx * ey - dy * ex
        if abs(den) < 1e-9:
            continue
        t = (x0 * ey - y0 * ex) / den
        u = (x0 * dy - y0 * dx) / den
        if t > 0 and -1e-6 <= u <= 1 + 1e-6:
            best = max(best, t)
    return best


def crease_attr(o, fn):
    """Store fn(local co) -> crease on the mesh's points, as the 'crease' attribute."""
    a = o.data.attributes.new('crease', 'FLOAT', 'POINT')
    a.data.foreach_set('value', [fn(v.co) for v in o.data.vertices])


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
        # shin -> foot across the ankle: the boot is one piece from the shaft into the foot, so the bend is spread over
        # the instep and the heel (a band 2.6 deep) instead of pinching in a single ring where the foot folds up
        foot = _smooth(az + 1.6, az - 1.0, co.z)
        if TOE_JOINT:
            # the toes ahead of the metatarsal heads take the toe bone, eased over a little more than a toe's width
            # (narrow under the sole, so the ball's pad stays with the foot; wide over the top, where the skin has to
            # slide over the joint: a narrow band there was squashed to a tenth of its area as the toes bent)
            zt = 1.0 - _smooth(1.2, 0.5, co.z)
            ya, yb = -TOE_JOINT[0] + 0.25 + 0.65 * zt, -TOE_JOINT[0] - 0.85 - 1.15 * zt
            toe = _smooth(ya, yb, co.y)
            return {f'Thigh.{side}': 1 - lower, f'Shin.{side}': lower * (1 - foot), f'Foot.{side}': lower * foot * (1 - toe),
                    f'Toe.{side}': lower * foot * toe}
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
    nk = neck_joints()
    at, md, cv = nk['atlas'], nk['mid'], nk['cervical']
    for name, h, t in (('Pelvis', (0, 0, HIP_Z), (0, 0, CHEST_Z - 1.0)), ('Chest', (0, 0, CHEST_Z), (0, 0, NECK_Z)),
                       ('Cervical', cv, md), ('Midneck', md, at), ('Neck', at, (at[0], at[1], at[2] + 3.0))):
        b = eb.new(name)
        b.head, b.tail = h, t
    for side, s in (('R', 1), ('L', -1)):
        lx, ax = -s * LEG_X, -s * SH_X
        segs = [('Thigh', (lx, 0, HIP_Z), (lx, 0, HIP_Z - THIGH)),
                ('Shin', (lx, 0, HIP_Z - THIGH), (lx, 0, HIP_Z - THIGH - SHIN)),
                ('Foot', (lx, 0, ANKLE_Z), (lx, -3.0, 0.5)),
                *((('Toe', (lx, -TOE_JOINT[0], TOE_JOINT[1]), (lx, -TOE_JOINT[0] - 1.8, TOE_JOINT[1])),) if TOE_JOINT else ()),
                ('UpperArm', (ax, 0.2, SH_Z), (ax, 0.2, SH_Z - UPPER)),
                ('Forearm', (ax, 0.2, SH_Z - UPPER), (ax, 0.2, SH_Z - UPPER - FORE)),
                ('Hand', (ax, 0.2, SH_Z - UPPER - FORE), (ax, 0.2, SH_Z - UPPER - FORE - 2.5))]
        for name, h, t in segs:
            b = eb.new(f'{name}.{side}')
            b.head, b.tail = h, t
    bpy.ops.object.mode_set(mode='OBJECT')
    seat_neck()


def seat_neck():
    """The neck's joints as pivots on the spine: Chest -> Cervical (C7-T1) -> Midneck (C3-C4) -> Atlas (the skull's base) -> Head, so
    the head rides the chest and bends at both joints. The Head empty keeps its own frame (the chin at its origin,
    which the Face Creator's head and every head part are built in) and its rest place in the world: only its
    parent changes. Call once, in the rest pose, after the head is built."""
    ob = bpy.data.objects
    if 'Head' not in ob or 'Chest' not in ob or 'Atlas' in ob:
        return
    head = ob['Head']
    rest = head.matrix_world.translation.copy()
    cerv = empty('Cervical', (0, 0, 0), ob['Chest'])
    mid = empty('Midneck', (0, 0, 0), cerv)
    atlas = empty('Atlas', (0, 0, 0), mid)
    head.parent = atlas
    head.matrix_parent_inverse.identity()
    place_neck(rest)


def _head_meshes():
    """The head's own skin as it sits on the body: the sculpted head, or the Face Creator's head (its first part)."""
    ob = bpy.data.objects
    return [o for o in ob if o.type == 'MESH' and o.name.startswith(('Head shape', 'head part 0'))]


def neck_joints():
    """Measure the head the figure wears and place the neck's joints from it (all in the rest pose, world units):
    chin (the lowest point at the front near the middle, as the figures' seat_head finds it), H (chin to crown),
    the head's middle front to back at the ears, then:
      atlas     the occipital condyles: under the ear's opening (EAR_Y), level with the lobe (EAR_BOTTOM)
      pit       the base of the neck at the front, NECK_PIT under the chin
      mid       the C3-C4 disc: C34_ALONG of the way up the column from C7-T1 to the atlas, LORDOSIS forward of it
      cervical  the C7-T1 disc: halfway between the base of the neck at the back (C7_BACK) and the pit in height,
                VERTEBRA_BACK of the neck's depth from its back, measured on the neck at that height
      bottom    the lowest point of the head's rigid sculpt (it runs on down the neck, see HEAD_NECK_Z)
    Falls back to the base figure's head (HEAD_Z, HEAD_H) where there is nothing to measure."""
    bpy.context.view_layer.update()
    pts = np.array([tuple(o.matrix_world @ v.co) for o in _head_meshes() for v in o.data.vertices]).reshape(-1, 3)
    head = bpy.data.objects.get('Head')
    sc = head.matrix_world.to_scale().x if head is not None else 1.0
    hx = head.matrix_world.translation.x if head is not None else 0.0
    front = pts[(np.abs(pts[:, 0] - hx) < 0.5 * sc) & (pts[:, 1] < -1.9 * sc)] if len(pts) else pts
    if len(front):
        chin, top = float(front[:, 2].min()), float(pts[:, 2].max())
        H = top - chin
        band = pts[np.abs(pts[:, 2] - (chin + 0.5 * H)) < 0.05 * H]
        ymid = 0.5 * float(band[:, 1].min() + band[:, 1].max())
        bottom = float(pts[:, 2].min())
    else:
        chin, H, ymid, bottom = HEAD_Z, HEAD_H, 0.0, HEAD_Z + HEAD_NECK_Z * HEAD_H
    c7z = chin - 0.5 * (C7_BACK + NECK_PIT) * H
    # the neck's depth at the C7 height, on its own surface (the neck, or the body's skin where it runs up the neck)
    yf, yb = NECK_RINGS[-1][4] - NECK_RINGS[-1][2], NECK_RINGS[-1][4] + NECK_RINGS[-1][2]
    for nm in ('Skin body', 'Neck'):
        o = bpy.data.objects.get(nm)
        if o is None:
            continue
        q = np.array([tuple(o.matrix_world @ v.co) for v in o.data.vertices]).reshape(-1, 3)
        q = q[(np.abs(q[:, 0] - hx) < 0.5) & (np.abs(q[:, 2] - c7z) < 0.3)]
        if len(q) >= 2:
            yf, yb = float(q[:, 1].min()), float(q[:, 1].max())
            break
    at = (hx, ymid + EAR_Y * H, chin + EAR_BOTTOM * H)
    cv = (hx, yb - VERTEBRA_BACK * (yb - yf), c7z)
    md = tuple(c + C34_ALONG * (a - c) for c, a in zip(cv, at))
    md = (md[0], md[1] - LORDOSIS * H, md[2])
    return {'chin': chin, 'H': H, 'bottom': bottom, 'pit': chin - NECK_PIT * H, 'ymid': ymid,
            'atlas': at, 'mid': md, 'cervical': cv}


def place_neck(rest=None):
    """Measure the head (neck_joints) and set the neck's pivots there, keeping the head where it is. Called by
    seat_neck, and again by a figure that sizes or seats its head after the build (the sorceress), in the rest
    pose: the pivots move, the trunk's neck weights are split again on the new joints (resplit_neck), the rig's
    neck bones take their rest again and the nape hook is weighted again."""
    ob = bpy.data.objects
    if 'Atlas' not in ob:
        return
    head, cerv, mid, atlas = ob['Head'], ob['Cervical'], ob['Midneck'], ob['Atlas']
    keyed = bool(head.animation_data and head.animation_data.action)
    if 'head_rest' in NECK:
        pose(0)          # (the rest of the neck and head: the idle stands the trunk straight)
    bpy.context.view_layer.update()
    rest = Vector(rest) if rest is not None else head.matrix_world.translation.copy()
    NECK.clear()
    NECK.update(neck_joints())
    cv, md, at = Vector(NECK['cervical']), Vector(NECK['mid']), Vector(NECK['atlas'])
    cerv.rotation_euler = mid.rotation_euler = atlas.rotation_euler = head.rotation_euler = (0, 0, 0)
    cerv.location = ob['Chest'].matrix_world.inverted() @ cv
    mid.location = md - cv
    atlas.location = at - md
    head.location = rest - at
    NECK['head_rest'] = tuple(head.location)
    bpy.context.view_layer.update()
    arm = RIG.get('arm')
    if arm is not None:
        for name in ('Cervical', 'Midneck', 'Neck'):
            pb = arm.pose.bones.get(name)
            for c in (pb.constraints if pb else ()):
                if c.type == 'CHILD_OF':
                    c.inverse_matrix = c.target.matrix_world.inverted()
        resplit_neck()
        nape_hooks()
    if keyed and FIG.get('root') is not None:
        key_sheet(FIG['root'])      # (the sheet's keys hold the pivots' old places: keyed again on the new ones)


def resplit_neck():
    """Split each trunk point's share of the chest and neck (its Chest, Cervical, Midneck and Neck weights
    together) again on the neck's joints as they are now placed (neck_shares)."""
    names = ('Chest', 'Cervical', 'Midneck', 'Neck')
    for o in bpy.data.objects:
        if o.type != 'MESH' or not any(m.type == 'ARMATURE' for m in o.modifiers):
            continue
        gi = {g.name: g.index for g in o.vertex_groups if g.name in names}
        if 'Chest' not in gi:
            continue
        gs = {n: (o.vertex_groups[gi[n]] if n in gi else o.vertex_groups.new(name=n)) for n in names}
        idx = {g.index: n for n, g in gs.items()}
        mw = o.matrix_world
        for v in o.data.vertices:
            t = sum(g.weight for g in v.groups if g.group in idx)
            if t <= 1e-6:
                continue
            for n, w in zip(names, [t * w for w in neck_split(mw @ v.co)]):
                if w > 1e-4:
                    gs[n].add([v.index], w, 'REPLACE')
                else:
                    gs[n].remove([v.index])


PIVOT_OF = {'Thigh': 'Hip', 'Shin': 'Knee', 'Foot': 'Ankle', 'Toe': 'Toe', 'UpperArm': 'Shoulder', 'Forearm': 'Elbow', 'Hand': 'Wrist',
            'Pelvis': 'Pelvis', 'Chest': 'Chest', 'Cervical': 'Cervical', 'Midneck': 'Midneck', 'Neck': 'Head'}


def _ease(a, b, v):
    """0 at a, 1 at b, eased between."""
    t = min(1.0, max(0.0, (v - a) / (b - a)))
    return t * t * (3 - 2 * t)


def neck_shares(co):
    """How a rest point of the neck splits between its joints: (lower, middle, upper). lower is the share carried
    past the base of the neck by the lower cervical spine (0 on the chest, 1 on the neck), middle the share of that
    carried on past the mid-neck (reaching 1 where the head's sculpt begins, as lower does), upper the share carried on by
    the skull (1 on the head; the jaw and face always 1). The lower blend runs over the visible lower neck, from a little under its base at the
    front (the skin over the top of the breastbone is drawn up as the head goes back) up to where the head's sculpt
    begins; the upper one over the whole neck, from the pit up to just under the
    jaw's lower edge (jaw_edge) in front of the ramus, and behind it up to the skull's base at the nape (the
    atlas's height), so the nape bends with the neck instead of with the skull, and a turn of the head (mostly at the atlas)
    is shared out along the neck's skin instead of wringing a thin ring of it under the jaw."""
    nk = NECK or neck_joints()
    H = nk['H']
    lower = _ease(nk['pit'] - 0.12 * H, nk['bottom'] - 0.1, co.z)
    middle = _ease(nk['pit'], nk['bottom'] - 0.1, co.z)
    # the skull's own surface starts at the jaw's lower edge in front of the ramus (a little under it, so the edge
    # stays crisp), and behind the ramus rises to the skull's base, level with the atlas, at the nape
    hy = (co.y - nk['ymid']) / H
    top = jaw_edge(hy) - 0.03 + (EAR_BOTTOM + 0.03 - jaw_edge(RAMUS_BACK)) * _ease(RAMUS_BACK, 0.3, hy)
    upper = _ease(nk['pit'], nk['chin'] + top * H, co.z)
    return lower, middle, upper


def neck_split(co):
    """(Chest, Cervical, Midneck, Neck) shares of a rest point's chest-and-neck weight (see neck_shares)."""
    lower, middle, upper = neck_shares(co)
    return (1 - lower, lower * (1 - middle), lower * middle * (1 - upper), lower * middle * upper)


def trunk_weights(co):
    """The trunk's bones for a point of the torso: the pelvis below the waist, the chest above it (blended over
    the waist, so the body twists there in one piece), then the neck: the lower cervical spine over the lower neck
    and the head's bone (Neck, which follows the Head empty) at the top (see neck_shares)."""
    chest = 1.0 - _smooth(CHEST_Z + 2.5, CHEST_Z - 5.5, co.z)
    c, cv, md, nk = neck_split(co)
    w = {'Pelvis': 1.0 - chest, 'Chest': chest * c, 'Cervical': chest * cv, 'Midneck': chest * md, 'Neck': chest * nk}
    if LOCKED_WALK:
        # (the world model) the jerkin's short skirt is carried part of the way by the thigh under it, more toward
        # the hem, so the leg lifting forward lifts the skirt over it instead of pushing out past its front edge.
        # Each side follows its own thigh, blended across the middle, where the two legs' swings cancel.
        share = SKIRT_ON_THIGH * _smooth(HIP_Z + 2.0, HIP_Z - 2.5, co.z)
        if share > 0:
            right = min(1.0, max(0.0, 0.5 - co.x / 3.0))      # (her right is -x)
            p = w.pop('Pelvis')
            w['Pelvis'] = p * (1.0 - share)
            w['Thigh.R'] = p * share * right
            w['Thigh.L'] = p * share * (1.0 - right)
    return w


def skin_trunk(body):
    """Every mesh carried by the body (the shirt, jerkin, belt, mantle, collar, neck...) is handed over to the
    rig's trunk bones, so the hips and shoulders can turn against each other through the walk."""
    for o in [c for c in body.children if c.type == 'MESH']:
        mw = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = mw
        groups = {}
        for v in o.data.vertices:
            for bone, w in trunk_weights(mw @ v.co).items():
                if w <= 1e-4:
                    continue
                if bone not in groups:
                    groups[bone] = o.vertex_groups.new(name=bone)
                groups[bone].add([v.index], w, 'REPLACE')
        m = o.modifiers.new('rig', 'ARMATURE')
        m.object = RIG['arm']
        o.modifiers.move(len(o.modifiers) - 1, 0)    # (deformed first, before its thickness and smoothing)


def mantle_on_arms(names):
    """Where the mantle lies over the round of each arm it rides on the arm: those points are handed over in part
    to the upper arm's bone, more the further out and down the arm they are, so the wool is carried with the arm as
    it swings instead of the arm swinging out through it (and the mantle can lie close over the arm)."""
    for o in [bpy.data.objects[n] for n in names if n in bpy.data.objects] + \
             [o for o in bpy.data.objects if o.name.startswith('Wool fuzz')]:
        mw = o.matrix_world
        gs = {g.name: g for g in o.vertex_groups}
        for s, side in ((1, 'R'), (-1, 'L')):
            name = f'UpperArm.{side}'
            gs[name] = gs.get(name) or o.vertex_groups.new(name=name)
        for v in o.data.vertices:
            co = mw @ v.co
            for s, side in ((1, 'R'), (-1, 'L')):
                out = min(1.0, max(0.0, (-s * co.x - (SH_X - 1.6)) / 2.6))
                down = min(1.0, max(0.0, (SH_Z + 2.0 - co.z) / 2.5))
                wa = (out * out * (3 - 2 * out)) * (down * down * (3 - 2 * down)) * 0.85
                if wa <= 1e-4:
                    continue
                for g in v.groups:
                    g.weight *= 1.0 - wa
                gs[f'UpperArm.{side}'].add([v.index], wa, 'REPLACE')


def finish_rig():
    arm = RIG['arm']
    bpy.context.view_layer.update()
    for pb in arm.pose.bones:
        seg, _, side = pb.name.partition('.')
        target = bpy.data.objects[f'{PIVOT_OF[seg]}.{side}' if side else PIVOT_OF[seg]]
        c = pb.constraints.new('CHILD_OF')
        c.target = target
        # (taken in the rest pose: the bone then moves exactly as its pivot moves away from rest)
        c.inverse_matrix = target.matrix_world.inverted()
    arm.hide_render = True
    nape_hooks()


def nape_hooks():
    """The head's sculpt runs on down the neck below the chin (to HEAD_NECK_Z), and the head is rigid: left so,
    the whole upper neck and the nape swung with the skull, and a nod tore it away from the neck under it. Its
    points below the skull take the lower neck's motion instead, by the same shares as the neck's own surface
    (neck_shares), through a hook on the Midneck pivot (below the head's sculpt the neck is all past the mid-neck: see neck_shares): a hooked point follows that pivot from its rest place, and
    the rest follows the head. Weighted in the rest pose; place_neck weights it again when the head is moved."""
    ob = bpy.data.objects
    if 'Midneck' not in ob or 'Head shape' not in ob:
        return
    o = ob['Head shape']
    if 'nape' in o.modifiers:
        o.modifiers.remove(o.modifiers['nape'])
    if 'nape' in o.vertex_groups:
        o.vertex_groups.remove(o.vertex_groups['nape'])
    bpy.context.view_layer.update()
    mw = o.matrix_world
    lw = [1.0 - neck_shares(mw @ v.co)[2] for v in o.data.vertices]
    idx = [i for i, w in enumerate(lw) if w > 1e-4]
    if not idx:
        return
    g = o.vertex_groups.new(name='nape')
    for i in idx:
        g.add([i], lw[i], 'REPLACE')
    m = o.modifiers.new('nape', 'HOOK')
    m.object = ob['Midneck']
    m.vertex_group = 'nape'
    m.falloff_type = 'NONE'
    o.modifiers.move(len(o.modifiers) - 1, 0)    # (before the head's smoothing)
    rehook_nape()


def rehook_nape():
    """Take the nape hook's rest from the head and the Midneck pivot as they are now (the rest pose)."""
    ob = bpy.data.objects
    o = ob.get('Head shape')
    m = o and o.modifiers.get('nape')
    if not m:
        return
    bpy.context.view_layer.update()
    m.matrix_inverse = m.object.matrix_world.inverted() @ o.matrix_world
    m.center = (0, 0, 0)


# The fingers, index to pinky: place across the palm, length as a share of the middle finger's, thickness, and the
# height of the knuckle below the wrist. The shares and the knuckle arc are a real hand's (index 0.92, ring 0.95,
# pinky 0.76 of the middle; the pinky's knuckle a tenth of the palm nearer the wrist).
FINGER_L = 1.3
FINGERS = ((-0.86, 0.92, 0.26, -2.52), (-0.29, 1.0, 0.27, -2.6), (0.27, 0.95, 0.25, -2.5), (0.8, 0.76, 0.22, -2.32))
PHALANGES = (0.48, 0.30, 0.22)     # first, second and tip bone as shares of a finger's length


def hand_mesh(s, fist=False):
    """A hand hanging relaxed from the wrist (at the origin, fingers down), made as one fused surface: metaball
    pieces for the palm, the four fingers (two segments each, curled loosely toward the palm, longest at the
    middle) and the thumb lying forward along the index finger melt into each other, so the fingers grow out of
    the palm with webbing between them, and the result is converted to a mesh. The palm faces the body (s = 1
    for the right hand, whose inside faces +X); the front is -Y.

    fist: the same wrist and palm with the hand closed. Each finger's first bone turns in at the knuckle, square to
    the palm, its second folds back up along the palm and its tip tucks in against it, so the knuckles stand in a
    row along the hand's lower edge (the top of the fist when the forearm is raised); the thumb lies across the
    front of the first two fingers' second bones."""
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
    # fingers, index at the front to pinky at the back. Lengths are shares of the middle finger's (itself shorter
    # than the palm: on a figure this size, slim fingers any longer read as spidery), and the knuckles lie on an arc,
    # the pinky's well back toward the wrist, so the four tips of the open hand fall on a curve and the four fingers
    # of the fist fold to the same depth in the palm
    for y0, share, r, zk in FINGERS:
        L = FINGER_L * share
        if fist:
            # (the knuckle a little proud of the back of the hand; the first bone turns in square to the palm, the
            # second folds back up along it, the tip tucks in against it, each its share of the finger's length)
            b1, b2, b3 = (L * f for f in PHALANGES)
            k0 = Vector((-0.06 * inward, y0 * 0.94, zk))
            k1 = k0 + Vector((inward * 0.98 * b1, 0.0, -0.2 * b1))
            k2 = k1 + Vector((inward * 0.1 * b2, 0.0, 0.995 * b2))
            k3 = k2 + Vector((-inward * 0.9 * b3, 0.0, 0.43 * b3))
            blob(k0, (0.9, 0.9, 0.9), r * 1.12)
            capsule(k0, k1, r * 1.05)
            capsule(k1, k2, r * 0.98)
            capsule(k2, k3, r * 0.85)
            continue
        # (the two pieces of each finger meet end to end at the middle joint with only a little overlap, and the
        # finger starts a touch lower in the palm: overlapping fully, their fields add up and swell into a knob)
        p = Vector((0.05 * inward, y0, zk - 0.05))
        ang = 0.3
        for frac, extra, rr in ((0.55, 0.0, r), (0.45, 0.6, r * 0.85)):
            ang += extra
            d = Vector((inward * math.sin(ang), 0.0, -math.cos(ang)))
            q = p + d * (L * frac)
            capsule(p + d * (0.12 * rr if extra else 0.0), q - d * 0.12 * rr, rr)
            p = q
    # the base of the hand where the thumb and the fingers share the wrist: a thenar mound swelling from the
    # wrist into the root of the thumb and a smaller hypothenar mound on the little-finger side, so the thumb,
    # the palm and the four fingers all grow out of one fleshy heel instead of meeting in a crease
    blob((0.18 * inward, -0.5, -0.95), (0.5, 0.6, 0.8), 0.9)
    blob((0.1 * inward, 0.5, -1.1), (0.4, 0.4, 0.75), 0.75)
    # the little-finger edge of the palm: one straight ridge from the wrist to the pinky knuckle, so the
    # outline runs smoothly down that side with no dip between the heel of the hand and the pinky
    zp = FINGERS[3][3]
    capsule((0.05 * inward, 0.62, 0.3), (0.05 * inward, 0.7, zp + 0.2), 0.44)
    # (tapering on into the open pinky's first segment, so the edge narrows down into the finger instead of
    # stepping in at the knuckle; the fist has no finger hanging there, so its edge ends at the knuckle)
    if not fist:
        capsule((0.05 * inward, 0.72, zp + 0.3), (0.15 * inward, 0.8, zp - 0.4), 0.3)
    capsule((0.05 * inward, -0.2, -0.2), (0.22 * inward, -0.75, -1.1), 0.5)
    # the thumb, from the heel of the palm on the inner front edge, its tip bent in toward the fingers
    t0 = Vector((0.2 * inward, -0.7, -0.9))
    t1 = t0 + Vector((0.28 * inward, -0.22, -0.95))
    t2 = t1 + Vector((0.2 * inward, 0.05, -0.62))
    if fist:
        # (round the front of the fist and across the first two fingers, outside their second bones)
        t1 = Vector((0.62 * inward, -1.02, -1.75))
        t2 = Vector((0.98 * inward, -0.28, -2.02))
    # the web of skin from the thumb's first joint to the root of the index finger
    capsule(t0 + (t1 - t0) * 0.8, (0.1 * inward, -0.85, -2.3), 0.26)
    capsule(t0, t1, 0.34)
    capsule(t1, t2, 0.25)

    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    bpy.data.objects.remove(ob)
    bpy.data.metaballs.remove(mb)
    bm = bmesh.new()
    bm.from_mesh(me)
    bpy.data.meshes.remove(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    # the metaball pieces leave small bulges where they overlap (a lump on the back of the hand by the index finger,
    # another on the little-finger edge by the wrist): relax the surface so it reads as one smooth skin, holding its
    # volume, without melting the fingers together
    for _ in range(4):
        bmesh.ops.smooth_laplacian_vert(bm, verts=bm.verts, lambda_factor=0.25, lambda_border=0.0,
                                        use_x=True, use_y=True, use_z=True, preserve_volume=True)
    # (a little smaller than life, about two thirds of the face's length: at this size, held out on a swinging arm
    # toward the viewer, a life-size hand reads as far too big against the slim forearm and the sleeve)
    bmesh.ops.scale(bm, vec=(0.98, 0.98, 1.0), verts=bm.verts)
    return bm


# ---- the fist
def set_fists(amount):
    """Close the hands into fists (amount over 0.5) or open them. A fist is a second mesh on the same wrist,
    'Fist.R' and 'Fist.L' beside 'Hand.R' and 'Hand.L' (hand_mesh(s, fist=True): a closed hand rebuilt from the
    hand's own pieces, since the open hand's skin has no finger joints to bend), and a pose shows one of the two.
    A figure built without fist meshes keeps its open hands. The run closes them (pose_run); every other pose
    opens them. (Posed frame by frame this is all it takes; an exporter keying a run clip has to carry the swap
    itself.)"""
    for side in 'RL':
        hand, fist = bpy.data.objects.get(f'Hand.{side}'), bpy.data.objects.get(f'Fist.{side}')
        if hand is None or fist is None:
            continue
        on = amount > 0.5
        fist.hide_render = fist.hide_viewport = not on
        hand.hide_render = hand.hide_viewport = on


def dots(points, size=0.42):
    """Small blocks at the given points: a row of stitches."""
    bm = bmesh.new()
    for p in points:
        r = bmesh.ops.create_cube(bm, size=size)
        bmesh.ops.translate(bm, vec=Vector(p), verts=r['verts'])
    return bm


def dashes(points, size):
    """Short stitches at the given points, each a flat dash (size across, deep, along), for a seam running up and
    down: a running stitch rather than a row of studs."""
    bm = bmesh.new()
    for p in points:
        r = bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=size, verts=r['verts'])
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


def almond(rx, ry, rz, s, seg=128, rings=80):
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


def _srgb_lin01(c):
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
    # cut to points about 0.014 apart, whatever the head's own mesh: ten cuts on the first head's edges (a
    # finer head needs fewer, or the eyes' patch alone runs to half a million points)
    mean_edge = float(np.mean([e.calc_length() for e in edges]))
    cuts = max(2, int(round(mean_edge / 0.0141)) - 1)
    res = bmesh.ops.subdivide_edges(hb, edges=edges, cuts=cuts, use_grid_fill=True, smooth=0.0)
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
    # (the smooth surface is looked up on a grid 0.025 apart, solved once per grid point and read bicubically: solving
    # it exactly for each of the patch's 190,000 points was half the figure's build)
    G = 0.025
    xs_ = np.arange(P[:, 0].min() - G, P[:, 0].max() + 2 * G, G)
    zs_ = np.arange(P[:, 2].min() - G, P[:, 2].max() + 2 * G, G)
    grid_ = {}

    def node(i, j):
        k = (i, j)
        if k not in grid_:
            grid_[k] = head_front_y((xs_[0] + i * G) / HEAD_H, (zs_[0] + j * G) / HEAD_H)
        return grid_[k]

    def cr(p0, p1, p2, p3, t):      # (Catmull-Rom: the surface's curvature kept between grid points)
        return p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)))

    def front_y(x, z):
        fi, fj = (x - xs_[0]) / G, (z - zs_[0]) / G
        i, j = int(math.floor(fi)), int(math.floor(fj))
        tx, tz = fi - i, fj - j
        rows = []
        for dj in (-1, 0, 1, 2):
            c = [node(i + di, j + dj) for di in (-1, 0, 1, 2)]
            if any(y is None for y in c):
                return head_front_y(x / HEAD_H, z / HEAD_H)
            rows.append(cr(*c, tx))
        return cr(*rows, tz)

    for v, r in zip(new, rim_d):
        w = smooth(0.0, 0.9, float(r))
        if w <= 0:
            continue
        y = front_y(v.co.x, v.co.z)
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
# the lips' MID/UP/HI/DEEP tones (sRGB 0-255), as the face creator exports them; empty for the ranger's own
LIP_COLOURS = {}
# The make-up's colours and strengths. These defaults are the ranger's; a figure changes the ones it wants with
# MAKEUP.update({...}) before it builds. shadow, shadowDark and shadowLow are the eyeshadow's wash, its deepest tone
# and the lighter wash under the lower lashes (sRGB 0-255); shadowAmt its opacity; liner a dark line along the upper
# lashes and wing its flick out past the outer corner (0 none .. 1 full); lip the lips' own tone (None: the
# ranger's dusty rose), from which the upper lip, the highlight and the line between them are shaded. (The lips' tones
# themselves are LIP_COLOURS, as the rogue's branch has them.)
# shadowInner is how far the wash runs in past the inner corner toward the nose (1 the ranger's, 0 none: it starts
# on the lid itself). browArch, browThick and browHeight shape the brow (multipliers on its arch and thickness, and
# an offset on its height in brow units).
MAKEUP = {'shadow': (70, 60, 56), 'shadowDark': (34, 29, 28), 'shadowLow': (92, 80, 74), 'shadowAmt': 0.78, 'shadowInner': 1.0,
          'liner': 0.0, 'wing': 0.0, 'browArch': 1.0, 'browThick': 1.0, 'browHeight': 0.0}
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
    inn = MAKEUP['shadowInner']
    across = (sm(-0.92, -0.35, uu) * (1 - inn) + sm(-1.3, -0.7, uu) * sm(-1.55, -1.3, uu) * inn) * (0.85 + 0.15 * sm(-0.2, 0.8, uu)) * sm(1.5, 1.0, uu)
    lid = sm(-0.012, 0.012, above) * (1 - sm(0.3 * tall, tall, above))
    # the deeper colour: smoked along the lash line and into the outer V (a wedge opening up and out from the corner)
    deep = sm(0.1, 1.0, uu) * (1 - sm(0.0, 0.5 * tall, above)) * sm(-0.005, 0.02, above)
    vwedge = sm(0.55, 1.05, uu) * (1 - sm(0.3 * tall, 1.1 * tall, np.abs(dd - (T + 0.07 + 0.2 * np.maximum(0, uu - 0.9)))))
    deep = np.maximum(deep * 0.8, 0.6 * vwedge * sm(1.5, 0.95, uu))
    base = np.array(MAKEUP['shadow'], dtype=np.float32)      # (the ranger's: a warm charcoal, near-black with a little brown, so it
    dark = np.array(MAKEUP['shadowDark'], dtype=np.float32)  # sits on warm skin; deepest in the corners)
    col = base + (dark - base) * np.clip(deep, 0, 1)[..., None]
    a = 0.8 * lid * across + 0.1 * deep * lid * across
    # under the lower lashes, outer half only
    low = sm(-0.01, 0.015, below) * (1 - sm(0.06, 0.3, below)) * sm(-1.0, -0.5, uu) * sm(1.35, 1.0, uu)
    col = col + (np.array(MAKEUP['shadowLow'], dtype=np.float32) - col) * low[..., None]
    a = np.maximum(a, 0.5 * low * (0.7 + 0.3 * sm(0.0, 0.9, uu)))
    # the hollow beside the nose, where the lid meets the bridge: the inner corner's own soft shade
    hollow = inn * np.exp(-(((uu + 1.02) / 0.26) ** 2 + ((dd - 0.04) / 0.3) ** 2))
    col = col + (base - col) * np.clip(hollow, 0, 1)[..., None] * (1 - low[..., None]) * 0.6
    a = np.maximum(a, 0.2 * hollow)
    # the far ends, past each corner, thin out rather than pooling into dark patches
    a *= 1 - 0.45 * sm(0.85, 1.35, uu) - 0.4 * sm(-0.85, -1.35, uu)
    a *= sm(SH_V0, SH_V0 + 0.15, dd) * sm(SH_V1, SH_V1 - 0.15, dd) * sm(SH_U0, SH_U0 + 0.2, uu) * sm(SH_U1, SH_U1 - 0.2, uu)   # (no hard edge at the patch's border)
    # nothing inside the opening itself
    a *= 1 - ((above < 0) & (below < 0) & (np.abs(uu) < 1.0))
    rgb = np.array([_srgb_lin01(c / 255) for c in col.reshape(-1, 3)], dtype=np.float32).reshape(H, W, 3) if False else col / 255
    # smoke: domain-warped gradient fBm in the opacity (see _warped_fbm), so the wash drifts and thins in soft,
    # swirled wisps instead of grading smoothly or breaking into even blobs
    smoke = _warped_fbm(uu * 2.2, dd * 2.6, seed=9, k=2.5)
    a = a * np.clip(0.95 + 0.6 * smoke, 0.55, 1.3)
    img = bpy.data.images.new('eyeshadow', W, H, alpha=True)
    a = a * MAKEUP['shadowAmt']   # (0.78 for the ranger: charcoal reads much stronger than the mauve did at the same opacity)
    # the liner: a dark line drawn along the upper lashes, thickening outward, and its wing, a flick carried up and
    # out past the outer corner along the line the lower lid would take if it went on, tapering to a point
    if MAKEUP['liner'] > 0 or MAKEUP['wing'] > 0:
        line = np.exp(-((above - 0.02) / (0.016 + 0.02 * sm(-0.6, 0.9, uu))) ** 2) * sm(-0.95, -0.4, uu) * sm(1.08, 0.98, uu)
        k = np.clip((uu - 0.95) / 0.55, 0, 1)
        flick = np.exp(-((dd - (0.03 + 0.34 * (uu - 0.95))) / (0.075 * (1 - 0.85 * k) + 0.004)) ** 2) * sm(0.9, 0.98, uu) * sm(1.5, 1.42, uu)
        ink = np.clip(np.maximum(MAKEUP['liner'] * line, MAKEUP['wing'] * flick), 0, 1)
        ink *= 1 - ((above < 0) & (below < 0) & (np.abs(uu) < 1.0))
        rgb = rgb + (dark / 255 * 0.7 - rgb) * ink[..., None]
        a = np.maximum(a, 0.92 * ink)
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
BROW_US = [-0.6, 1.1]          # the brow's inner and outer ends (brow units)


def brow_line(ub):
    """The brow's centre height above the eye's centre and its half-height, at ub (in the brow's own units: the
    face's x from the eye's centre over 1.1). Both are single smooth closed forms, with no table to interpolate
    (a piecewise-linear one showed its corners as kinks along the brow's edges): the centre is a skewed sine arch,
    s^1.6 pushing its peak about two thirds of the way out, so it starts level at the nose and falls more
    steeply into the tail, and a low arch: an inner end set low and slanting up (with upright hairs) read as a
    scowl from the game's high camera; the half-height is lighter at the head, swells a little just past it and
    thins smoothly to a fine tail.
    """
    u = min(max(ub, BROW_US[0]), BROW_US[-1])
    t = (u - BROW_US[0]) / (BROW_US[-1] - BROW_US[0])
    zc = 1.0 + 0.24 * MAKEUP['browArch'] * math.sin(math.pi * t ** 1.6) - 0.08 * t + MAKEUP['browHeight']
    rise = math.sin(0.5 * math.pi * min(1.0, t / 0.28))
    r = (0.016 + 0.17 * (1 - t) ** 0.65 * (0.62 + 0.38 * rise)) * MAKEUP['browThick']
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
        head = math.radians(62 - 12 * smooth(-0.7, -0.35, ub))
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

    def fade_by_facing(img, ext, tag):
        """Thin a painting out where the face turns away from the front. Each patch is laid on by casting
        straight back from the front, so where the skin turns sideways (the socket's outer wall, the side of the
        face past the outer corner) one column of the painting is stretched over a wide band of skin, which seen in
        profile showed as a dark smear under and behind the eye. The painting is shared by both eyes (mirrored), so
        the facing is measured round the right eye."""
        if img.get('faced'):
            return
        from PIL import Image as PImage
        ex, ez, s = next(e for e in eyes if e[2] == 1)
        U0, U1, V0, V1 = ext
        NX, NY = 128, 56
        f = np.ones((NY, NX), dtype=np.float32)
        for j in range(NY):
            dz = V1 - (j + 0.5) / NY * (V1 - V0)
            for i in range(NX):
                u = U0 + (i + 0.5) / NX * (U1 - U0)
                h = tree.ray_cast(Vector((ex + u * EYE_RX, -50, ez + u * EYE_RX * EYE_TILT + dz)), Vector((0, 1, 0)))
                fwd = -h[1].y if h[0] is not None else 0.0
                t = min(1.0, max(0.0, (fwd - 0.4) / 0.3))
                f[j, i] = t * t * (3 - 2 * t)
        W, H = img.size
        fm = np.asarray(PImage.fromarray((f * 255).astype(np.uint8)).resize((W, H), PImage.BILINEAR), dtype=np.float32) / 255
        px = np.array(img.pixels[:], dtype=np.float32).reshape(H, W, 4)
        px[..., 3] *= fm[::-1]
        img.pixels.foreach_set(px.ravel())
        img.pack()
        img['faced'] = 1

    shadow_img, lash_img, brow_img = eyeshadow_image(), lash_image(), brow_image()
    fade_by_facing(shadow_img, (SH_U0, SH_U1, SH_V0, SH_V1), 'shadow')
    fade_by_facing(lash_img, (SH_U0, SH_U1, SH_V0, SH_V1), 'lash')
    fade_by_facing(brow_img, (BR_U0, BR_U1, BR_V0, BR_V1), 'brow')
    shadow_m = material('eyeshadow', shadow_img)
    lash_m = material('lash paint', lash_img)
    brow_m = material('brow paint', brow_img)

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
        patch('Eyeshadow', ex, ez, s, 0.022, shadow_m, NU=90, NV=44)
        patch('Lashes', ex, ez, s, 0.034, lash_m, NU=110, NV=56)
        patch('Brow paint', ex, ez, s, 0.02, brow_m, NU=90, NV=32, ext=(BR_U0, BR_U1, BR_V0, BR_V1))


LIP_TEX = {}
LIP_X0, LIP_X1, LIP_Z0, LIP_Z1 = -0.12, 0.12, 0.165, 0.305     # the lip patch's extent, in head heights (real z)


def _lip_forms(W, H):
    """The lips' forms on the lip patch's grid (H x W), from mouth_relief: across (x, in the sculpt's own across),
    the warped height w, the line between the lips zl, each lip's edge, and how far across each lip a point lies."""
    xs = LIP_X0 + (np.arange(W) + 0.5) / W * (LIP_X1 - LIP_X0)
    zr = LIP_Z1 - (np.arange(H) + 0.5) / H * (LIP_Z1 - LIP_Z0)
    zw = np.array([FACE_WARP(z) for z in zr])               # (mouth_relief is laid out in the warped heights)
    x, w = np.meshgrid(xs / FACE['mouthWidth'], zw)     # (in mouth_relief's own across, like the sculpt)
    ax = np.abs(x)

    def sm(e0, e1, v):
        t = np.clip((v - e0) / (e1 - e0), 0, 1)
        return t * t * (3 - 2 * t)
    # the forms, from mouth_relief: the line between the lips, the upper edge with its cupid's bow, the lower edge
    zl = 0.258 + 0.007 * FACE['smile'] * (ax / 0.082) ** 2
    bow = 0.006 * np.exp(-((ax - 0.019) / 0.011) ** 2)
    top = 0.296 + bow - 0.012 * (ax / 0.082) ** 2
    # (the colour stays inside the sculpted lips, which end at about 0.08: painted wider and taller, the lips read
    # too full and the colour ran out past the corners as dark wedges)
    pinch = np.sqrt(np.clip(1 - (ax / 0.081) ** 2, 0, 1))  # each lip tapers to nothing at the corner
    top = zl + 0.75 * (top - zl) * pinch ** 0.8
    bot = zl - 0.028 * pinch ** 1.4
    upper = sm(top + 0.0022, top - 0.0022, w) * sm(zl - 0.002, zl + 0.001, w)
    lower = sm(bot - 0.007, bot + 0.005, w) * sm(zl + 0.002, zl - 0.001, w)
    # (past the corners both edges meet the line, which would leave a thin band of each running on out)
    upper, lower = upper * sm(0.084, 0.076, ax), lower * sm(0.084, 0.076, ax)
    lip = np.maximum(upper, lower)
    # how far down each lip lies, 0 at the line between them, 1 at its outer edge
    tu = np.clip((w - zl) / np.maximum(top - zl, 1e-4), 0, 1)
    tl = np.clip((zl - w) / np.maximum(zl - bot, 1e-4), 0, 1)
    return dict(x=x, w=w, ax=ax, sm=sm, zl=zl, top=top, bot=bot, upper=upper, lower=lower, lip=lip, tu=tu, tl=tl)


def lip_painting(W=576, H=336):
    """The lips' colour as an RGBA array (H x W, sRGB 0-1, the first row at LIP_Z1), laid over the sculpted mouth
    (mouth_relief) in its own warped coordinates, so the colour follows the forms exactly. A natural, unpainted
    lip: a dusty rose a few shades deeper and redder than the skin, never a pink. The upper lip, turned down away
    from the light, a shade deeper than the lower; the lower lip fuller and a little lighter across its middle,
    where it catches the light; both darkening into the line between them and the corners, where they tuck in.
    The upper lip's edge (the vermilion border) is crisp, with the faint pale rim of skin just above it that gives
    the mouth its drawn line; the lower lip's edge is soft and melts into the skin. Fine vertical creases run
    across both lips, a breath darker. The colour thins out toward the corners so the mouth ends in skin, not in
    a hard point."""
    F = _lip_forms(W, H)
    x, w, ax, sm, zl, top, bot = F['x'], F['w'], F['ax'], F['sm'], F['zl'], F['top'], F['bot']
    upper, lower, lip, tu, tl = F['upper'], F['lower'], F['lip'], F['tu'], F['tl']
    L = LIP_COLOURS
    MID = np.array(L.get('MID', [178, 104, 96]), dtype=np.float32)      # the lip's own tone: a dusty rose on her warm skin
    UP = np.array(L.get('UP', [162, 90, 84]), dtype=np.float32)         # the upper lip, a shade deeper
    HI = np.array(L.get('HI', [194, 124, 114]), dtype=np.float32)       # the fullest part of the lower lip
    DEEP = np.array(L.get('DEEP', [112, 58, 56]), dtype=np.float32)     # into the line and the corners
    col = MID + (UP - MID) * (upper > lower)[..., None] * sm(0.0, 0.35, tu)[..., None]
    full = np.exp(-((tl - 0.5) / 0.28) ** 2) * np.exp(-(x / 0.045) ** 2) * (lower >= upper)
    col = col + (HI - col) * (0.75 * full)[..., None]
    # (the upper lip's own middle, the tubercle, catches a little light too)
    tub = np.exp(-((tu - 0.4) / 0.25) ** 2) * np.exp(-(x / 0.02) ** 2) * (upper > lower)
    col = col + (HI - col) * (0.35 * tub)[..., None]
    # into the line between them and the corners
    inner = np.where(upper > lower, sm(0.35, 0.0, tu), sm(0.3, 0.0, tl))
    corner = sm(0.045, 0.078, ax)
    col = col + (DEEP - col) * np.clip(0.4 * inner + 0.25 * corner, 0, 1)[..., None]
    # the line itself: a dark, soft parting, deepest at the corners
    # (narrow: wider, it read as the lips parted on a dark mouth)
    part = np.exp(-((w - zl) / 0.0019) ** 2) * sm(0.08, 0.06, ax)
    col = col + (np.array([84, 42, 42], dtype=np.float32) - col) * np.clip(0.8 * part, 0, 1)[..., None]
    # the fine vertical creases across the lips (fanning a little outward toward the corners)
    # (irregular: their spacing wanders and each one fades in and out, so they read as texture, not stripes)
    cr = np.sin((x + 0.18 * x * np.abs(w - zl) / 0.04) / 0.0055 * np.pi + 5.0 * _fbm2(x * 30, w * 8, 31, 3))
    cr = np.clip(cr, 0, 1) ** 4 * np.clip(0.5 + 1.5 * _fbm2(x * 18, w * 30, 37, 3), 0, 1)
    col = col * (1 - 0.045 * np.clip(cr, 0, 1) * np.maximum(sm(0.1, 0.6, tu), sm(0.1, 0.6, tl)))[..., None]
    a = lip * (0.88 - 0.5 * corner) + 0.0 * x
    a = np.maximum(a, part * 0.75)
    # the pale rim of skin just over the upper lip's edge
    rim = np.exp(-((w - top - 0.0045) / 0.0028) ** 2) * sm(0.065, 0.03, ax) * (1 - upper)
    col = col + (np.array([232, 200, 180], dtype=np.float32) - col) * rim[..., None] * (1 - lip[..., None])
    a = np.maximum(a, 0.28 * rim)
    a *= sm(LIP_X0, LIP_X0 + 0.01, x) * sm(LIP_X1, LIP_X1 - 0.01, x)
    return np.concatenate([np.clip(col, 0, 255) / 255, np.clip(a, 0, 1)[..., None]], axis=2).astype(np.float32)


def lip_layer_maps(W=576, H=336):
    """The lips' two layers' map on the lip patch (H x W x 4, 0-1, first row at LIP_Z1): red, how wet the surface is
    (most along the parting and on the lower lip's middle, drying toward the border and the corners); green, the
    grooves of the lip print (mostly vertical, fanning wider toward the border, each with its own depth and length,
    LIP_PITCH_MM apart at the parting, measured from the head); blue, the dry flakes toward the border. The same
    pattern as the Face Creator's lipFurrow and lipAt (tools/face-creator/apply-lip-layers.py)."""
    import skin_layers as SLm
    F = _lip_forms(W, H)
    x, w, ax, sm, upper, lower, lip, tu, tl = F['x'], F['w'], F['ax'], F['sm'], F['upper'], F['lower'], F['lip'], F['tu'], F['tl']
    up = (upper > lower).astype(np.float64)
    t = np.where(up > 0, tu, tl)
    head_mm = HEAD_H * SLm.MM_PER_UNIT
    SP = SLm.LIP_PITCH_MM / head_mm / FACE['mouthWidth']       # (x is across in head heights over the mouth's width)
    moist = np.where(up > 0, 0.6 * sm(0.45, 0.1, tu), sm(0.85, 0.35, tl)) * sm(0.075, 0.03, ax) * lip

    def h(k, s_):
        v = np.sin(k * 12.9898 + s_ * 78.233) * 43758.5453
        return v - np.floor(v)
    s = x / (1 + 0.45 * t)
    ph = s / SP + 0.35 * _fbm2(x * 40 + 3 * up, t * 1.6, 61, 2)
    k = np.floor(ph) + 1000 * up
    f = ph - np.floor(ph)
    r1, r2, r3 = h(k, 71), h(k, 73), h(k, 75)
    t0, t1 = 0.06 + 0.35 * r2 * r2, 0.72 + 0.28 * r3
    g = np.exp(-((f - 0.5 - 0.28 * (r3 - 0.5)) / (0.15 + 0.07 * r1)) ** 2) * (0.2 + 0.8 * r1 * r1) * sm(t0, t0 + 0.12, t) * sm(t1, t1 - 0.15, t)
    groove = np.clip(g, 0, 1) * (lip > 0.01)
    fm = SLm.LIP_FLAKE_MM / head_mm
    fn = 0.6 * _fbm2(x * FACE['mouthWidth'] / fm, w / (0.7 * fm), 91, 2) + 0.4 * _fbm2(x * FACE['mouthWidth'] / (0.5 * fm), w / (0.4 * fm), 93, 2)
    flake = sm(0.15, 0.55, fn) * sm(0.45, 0.95, t) * (1 - moist) * (lip > 0.01)
    return np.stack([np.clip(moist, 0, 1), groove, np.clip(flake, 0, 1), np.ones_like(moist)], axis=2).astype(np.float32)


def lip_layers_image():
    if 'layers' in LIP_TEX:
        return LIP_TEX['layers']
    px = lip_layer_maps()
    H, W = px.shape[:2]
    img = bpy.data.images.new('lip layers', W, H, alpha=False)
    img.colorspace_settings.name = 'Non-Color'
    img.pixels.foreach_set(px[::-1].ravel())
    img.pack()
    LIP_TEX['layers'] = img
    return img


def lip_image():
    if 'img' in LIP_TEX:
        return LIP_TEX['img']
    px = lip_painting()
    H, W = px.shape[:2]
    img = bpy.data.images.new('lips', W, H, alpha=True)
    img.pixels.foreach_set(px[::-1].ravel())
    img.pack()
    LIP_TEX['img'] = img
    return img


def lip_paint(head, tree, NU=120, NV=70, off=0.018):
    """The lips' colour (lip_painting) on a thin skin of the face over the mouth: a grid of the patch, each point
    cast straight back onto the finished head and lifted a hair off it, like the eyeshadow. Its material is the lips'
    two layers (skin_layers.lip_material)."""
    m = bpy.data.materials.new('lip paint')
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image = lip_image()
    tex.interpolation = 'Cubic'
    tex.extension = 'EXTEND'
    lay = nt.nodes.new('ShaderNodeTexImage')
    lay.image = lip_layers_image()
    lay.interpolation = 'Cubic'
    lay.extension = 'EXTEND'
    # (two layers, skin_layers.lip_material: a smooth pink under layer with a short red scatter, and a rougher coat
    # on top with the grooves and flakes in its own normal, wet at the parting and dry at the border. It was one
    # flat Principled at roughness 0.55.)
    import skin_layers
    if os.environ.get('LIP_LAYERS', '1') == '0':      # (the flat lips, for a before/after: face_views.py --lips flat)
        p = nt.nodes.new('ShaderNodeBsdfPrincipled')
        nt.links.new(tex.outputs['Color'], p.inputs['Base Color'])
        nt.links.new(tex.outputs['Alpha'], p.inputs['Alpha'])
        p.inputs['Roughness'].default_value = 0.55
        p.inputs['Specular IOR Level'].default_value = 0.3
        nt.links.new(p.outputs['BSDF'], out.inputs['Surface'])
    else:
        skin_layers.lip_material(nt, out, tex, lay)
    m.surface_render_method = 'BLENDED'
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new('UVMap')
    grid = []
    for j in range(NV + 1):
        row = []
        for i in range(NU + 1):
            px = (LIP_X0 + (LIP_X1 - LIP_X0) * i / NU) * HEAD_H
            pz = (LIP_Z0 + (LIP_Z1 - LIP_Z0) * j / NV) * HEAD_H
            h = tree.ray_cast(Vector((px, -50, pz)), Vector((0, 1, 0)))
            row.append(bm.verts.new((px, (h[0].y if h[0] is not None else 0.0) - off, pz)))
        grid.append(row)
    for j in range(NV):
        for i in range(NU):
            vs = [grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]]
            f = bm.faces.new(vs)
            for lp in f.loops:
                k = vs.index(lp.vert)
                lp[uvl].uv = ((i + (1 if k in (1, 2) else 0)) / NU, (j + (1 if k in (2, 3) else 0)) / NV)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new('Lip paint')
    bm.to_mesh(me)
    bm.free()
    for pl in me.polygons:
        pl.use_smooth = True
    me.materials.append(m)
    o = bpy.data.objects.new('Lip paint', me)
    link(o, head)
    o.visible_shadow = False


# ---------------------------------------------------------------- building a figure
# A character's build() calls these in order (see the module's docstring), adding its hair and clothes between them.
FIG = {}   # the figure being built: its name, root and the trunk's pivots


def begin_figure(name):
    """A fresh scene with the figure's root (named for the character; the renderer turns it per facing) and the
    body under it, which carries the walk's bob and sway. The pelvis and chest turn against each other through
    the walk: the legs hang from the one, the arms from the other, and the torso's meshes are skinned across
    both (see skin_trunk)."""
    reset()
    root = empty(name, (0, 0, 0))
    body = empty('Body', (0, 0, 0), root)
    pelvis = empty('Pelvis', (0, 0, HIP_Z), body)
    chest = empty('Chest', (0, 0, CHEST_Z), body)
    FIG.clear()
    FIG.update(name=name, root=root, body=body, pelvis=pelvis, chest=chest)
    return root, body, pelvis, chest


def torso_mesh(name, mid, parent, zs=None):
    """The body's own surface (in whatever is worn closest on it: a shirt, a bodice), from the neckline to the
    top of the thighs."""
    zs = zs or [53.0 - 0.5 * i for i in range(50)]          # 53 down to 28.5
    return mesh_obj(name, loft_fn(zs, lambda z, p: torso_point(z, p)), mid, parent)


def build_head(body):
    """The neck, the sculpted head and face, the eyes, brows, lashes and eyeshadow, and the mouth. Returns the
    head's pivot and a BVH of the finished head (for laying hair on it)."""
    # (the neck above the collar is part of the head's sculpt, see head_section; this is only its lower part,
    # under the collar and the mantle, running on down into the shirt)
    mesh_obj('Neck', loft(NECK_RINGS), M_SKIN, body)
    head = empty('Head', (0, 0, HEAD_Z), body)
    hb = sculpted_head()
    # the eyes, brows and mouth sit on the sculpted surface (found by casting rays at it)
    feat = {}
    for s in (-1, 1):
        x = s * FACE['eyeSpacing'] * HEAD_H
        feat[s] = (x, surface_y(hb, x, 0.525 * HEAD_H), surface_y(hb, x, 0.635 * HEAD_H))
    mouth_y = surface_y(hb, 0, MOUTH_Z * HEAD_H)
    head_tree = bvh_of(hb)
    # (the eyes are set against the head before its lids are sculpted, so the balls follow the plain socket and
    # the lids then close over their edges)
    eyes = [(feat[s][0], 0.525 * HEAD_H + EYE_LIFT, s) for s in (-1, 1)]
    if MANNEQUIN:
        mesh_obj('Head shape', hb, M_SKIN, head, subdiv=0)
        return head, head_tree
    for s in (-1, 1):
        x, ey, by = feat[s]
        build_eye(head, head_tree, s, x, ey, 0.525 * HEAD_H + EYE_LIFT)
    sculpt_eyes(hb, eyes)
    final_tree = bvh_of(hb)
    mesh_obj('Head shape', hb, M_SKIN, head, subdiv=0)
    # the ears, each sculpted as its own surface (blender/shared/ear.py) and set on the head's side
    if EARS:
        for e in ear_grids(side_x_finder(head_section, jaw_relief), HEAD_H):
            name = 'Ear ' + ('L' if e['side'] > 0 else 'R')
            mesh_obj(name, ear_mesh(e, name), M_SKIN, head, subdiv=0)
    eye_makeup(head, final_tree, eyes)
    lip_paint(head, final_tree)
    # (the lips are sculpted (see mouth_relief); this is only the shadowed line between them, set into it)
    mesh_obj('Mouth', ellipsoid(0.66, 0.1, 0.06, 12, 6), M_LIPS, head, loc=(0, mouth_y + 0.1, MOUTH_Z * HEAD_H), subdiv=0)
    return head, head_tree


# the bare arm's radius (local height from the shoulder joint, radius), before the deltoid: the round of the
# shoulder, the upper arm slimming to the elbow, the forearm full just below it and drawn in to a slender wrist
BARE_ARM_R = [(1.5, 0.3), (1.25, 1.2), (0.8, 1.72), (0.0, 1.82), (-2.0, 1.72), (-4.5, 1.56), (-UPPER + 1.2, 1.34),
              (-UPPER, 1.3), (-UPPER - 1.6, 1.42), (-UPPER - 4.0, 1.3), (-UPPER - FORE + 1.0, 1.0), (-UPPER - FORE, 0.95),
              (-UPPER - FORE - 0.4, 0.93)]


def arm_xy(z, phi, s, out=0.0):
    """The bare arm's surface at local height z (from the shoulder joint) and phi, pushed out by `out` (for what is
    worn close on it): flatter front to back than across above the elbow, the deltoid, a soft hollow inside the
    elbow, the forearm's muscle full over its outside and back just below the elbow, and the wrist an oval lying the
    other way (wide front to back, as the hand hangs with its palm to the thigh, and thin across)."""
    r = _profile(BARE_ARM_R, z) + deltoid(z, phi, s)
    r -= 0.08 * math.exp(-((z + UPPER) / 0.8) ** 2) * max(0.0, math.cos(phi)) ** 2    # (the inside of the elbow)
    r += 0.1 * math.exp(-((z + UPPER + 2.4) / 1.8) ** 2) * max(0.0, 0.5 + 0.5 * s * math.sin(phi))
    t = _smooth(-UPPER - 4.0, -UPPER - FORE, z)
    r += out
    return -r * (1.0 - 0.12 * t) * math.sin(phi), -r * (0.92 + 0.12 * t) * math.cos(phi)


def bare_arm(side, s, x, w, top=1.5):
    """A bare arm in skin, from the shoulder (local height top) to just past the wrist, for clothes with short
    sleeves or none: placed at (x, 0.2, SH_Z) like a sleeve (its shape: arm_xy)."""
    zs = [top - 0.2 * i for i in range(int((top + UPPER + FORE + 0.4) / 0.2) + 1)]
    return mesh_obj(f'Arm.{side}', loft_fn(zs, lambda z, phi: arm_xy(z, phi, s), n=40, cap_bot=False), M_SKIN,
                    loc=(x, 0.2, SH_Z), skin=w)


def build_arms(chest, sleeve):
    """Each arm's pivots (shoulder, elbow, wrist, hung from the chest), its clothes (sleeve(side, s, x, w), where
    side is 'R' or 'L', s is 1 for her right arm and -1 for her left, x the shoulder's X and w the arm's bone
    weights; the meshes are placed at (x, 0.2, SH_Z)), then the hand."""
    for side, s in (('R', 1), ('L', -1)):
        x = -s * SH_X   # the figure's right is -X (she faces -Y)
        sh = empty(f'Shoulder.{side}', (x, 0.2, SH_Z - CHEST_Z), chest)
        el = empty(f'Elbow.{side}', (0, 0, -UPPER), sh)
        empty(f'Wrist.{side}', (0, 0, -FORE), el)
        w = arm_weights(side)
        sleeve(side, s, x, w)
        mesh_obj(f'Hand.{side}', hand_mesh(s), M_SKIN, loc=(x, 0.2, SH_Z - UPPER - FORE - 0.1), subdiv=0, skin=w)


def build_legs(pelvis, leg):
    """Each leg's pivots (hip, knee, ankle, hung from the pelvis) and its clothes and footwear (leg(side, s, x,
    w), as for the arms; the meshes are placed at (x, 0, HIP_Z) and below)."""
    for side, s in (('R', 1), ('L', -1)):
        x = -s * LEG_X
        hip = empty(f'Hip.{side}', (x, 0, 0), pelvis)
        kn = empty(f'Knee.{side}', (0, 0, -THIGH), hip)
        an = empty(f'Ankle.{side}', (0, 0, -SHIN), kn)
        if TOE_JOINT:
            empty(f'Toe.{side}', (0, -TOE_JOINT[0], TOE_JOINT[1] - ANKLE_Z), an)
        leg(side, s, x, leg_weights(side))


# How far the shoulder line is let down from the base of the neck to the arm. The body's table (TORSO) is wide for
# its height there, so the line leaves the neck and runs nearly level to the arm (about 16 degrees, where a real
# trapezius falls about twice as steeply), then turns down a square side with the arm's round cap on its corner:
# a box with a ball on it. Everything worn on the shoulders is built from the torso's shape or from the arm's, so
# the whole figure is let down together at the end of the build (see slope_shoulders).
SHOULDER_SLOPE = float(os.environ.get('SHOULDER_SLOPE', '1.4'))


def _ss(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def slope_shoulders(drop=None):
    """Let the shoulder line down: nothing at the neck, `drop` units at the arm (eased in between, level over the
    arm itself so it moves down whole), and the drop fading out down the chest and back (heights 45.4 to 50.8), so
    the bodice line, the armpit and the arm's joint stay where they are. Applied to every mesh worn on the body
    (skin, shirt, jerkin, mantle, collar, sleeves) in world space, after they are weighted to the rig, so each keeps
    the weights it was given where it was; the head and what is on it are left alone."""
    drop = SHOULDER_SLOPE if drop is None else drop
    if drop <= 0:
        return
    bpy.context.view_layer.update()
    for o in [o for o in bpy.data.objects if o.type == 'MESH']:
        p = o.parent
        on_head = False
        while p is not None:
            if p.name == 'Head':
                on_head = True
                break
            p = p.parent
        if on_head or o.data.shape_keys:
            continue
        mw = o.matrix_world
        inv = mw.inverted()
        moved = False
        for v in o.data.vertices:
            w = mw @ v.co
            if w.z < 45.4 or abs(w.x) < 1.9:
                continue
            # (and none above the base of the neck: a collar or gorget standing round the neck stays where it is, and
            # what lies on the shoulders under its edge, a cape's top, stays tucked under it)
            d = drop * _ss(1.9, 6.4, abs(w.x)) * _ss(45.4, 50.8, w.z) ** 1.3 * (1.0 - _ss(51.6, 52.8, w.z))
            if d <= 1e-4:
                continue
            w.z -= d
            v.co = inv @ w
            moved = True
        if moved:
            o.data.update()


def finish_figure(after_skin=None):
    """Hand the trunk's meshes to the rig (then after_skin(), e.g. to let a mantle ride on the arms), finish the
    rig, set up the camera, the light and the render, and key the 10 sheet columns onto the timeline."""
    sc = bpy.context.scene
    root = FIG['root']
    skin_trunk(FIG['body'])
    if after_skin:
        after_skin()
    slope_shoulders()
    finish_rig()
    skin_tones()

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
    # keep the scene loaded in Cycles between renders: only what changed (the pose, the turn) is synced again,
    # so after the first frame a cell takes about half a second instead of twenty (the samples cost nothing)
    sc.render.use_persistent_data = True
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
    for name in ('matid', 'detail', 'crease', 'flush'):
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

    key_sheet(root)
    return root


def key_sheet(root):
    """Key the 10 sheet columns onto the timeline (frames 1-10), so the .blend plays the idle and walk."""
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, COLS
    for col in range(COLS):
        pose(col)
        for o in sc.objects:
            if o.type == 'EMPTY' and o.name != root.name:   # (the root is left free: the renderer turns it per facing)
                o.keyframe_insert('rotation_euler', frame=col + 1)
                o.keyframe_insert('location', frame=col + 1)
    for o in sc.objects:
        if o.animation_data and o.animation_data.action:
            for fc in _fcurves(o.animation_data.action):
                for kp in fc.keyframe_points:
                    kp.interpolation = 'CONSTANT'


def run(build, stem, out_dir):
    """The command line of a character's script: build it, save <stem>.blend, render <stem>-blender.png."""
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    out_dir = argv[argv.index('--out') + 1] if '--out' in argv else out_dir
    os.makedirs(out_dir, exist_ok=True)
    build()
    bpy.context.scene.frame_set(1)
    bpy.context.preferences.filepaths.save_version = 0   # no .blend1 backups
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, f'{stem}.blend'), compress=True)
    if '--build-only' in argv:
        return
    with tempfile.TemporaryDirectory() as tmp:
        render_sheet(os.path.join(out_dir, f'{stem}-blender.png'), tmp)
    print('wrote', os.path.join(out_dir, f'{stem}-blender.png'))


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


# ---- the walk cycle
# Each leg's joints follow the curves of a real walk through one cycle (t from 0 at the heel striking the ground,
# through the stance to toe-off at about 0.6, then the swing back to the next heel strike), in degrees:
# the hip, forward of straight down (furthest forward just before the heel strikes, furthest back at push-off);
# the knee, bent (a little as the heel takes the weight, nearly straight in mid-stance, then folding up to about
# 60 degrees through the swing, so the foot is lifted clear, and staying bent as the thigh comes up in front, so the
# knee is raised as the leg comes through before the lower leg reaches out for the heel strike); and the
# foot's pitch, toes up positive (up at the heel strike, flat through the stance, rolling up onto the toes to push
# off, and lifted again through the swing so the toes clear the ground).
GAIT_HIP = [(0.0, 22), (0.15, 17), (0.35, 0), (0.52, -13), (0.62, -8), (0.71, 10), (0.81, 25), (0.9, 27)]
GAIT_KNEE = [(0.0, 4), (0.12, 19), (0.3, 6), (0.45, 9), (0.6, 38), (0.7, 63), (0.8, 58), (0.9, 25), (0.97, 5)]
GAIT_FOOT = [(0.0, 15), (0.08, 0), (0.4, 0), (0.5, -12), (0.6, -30), (0.7, -4), (0.82, 6), (0.94, 13)]
# The arms swing against the legs: each arm is furthest forward a little after the leg on its own side is furthest
# back. The swing is full and a little exaggerated, so it reads from the side: the upper arm swings about 35 degrees
# forward and as far back, past the hip. The elbow is never straight, opening to a soft bend as the arm swings back and
# folding to about 60 degrees as it swings forward, just behind the shoulder, so the forearm follows through instead of the arm swinging stiff from the shoulder like a pendulum. The arm
# swings and folds in the plane of the walk, so it reads from the side camera: nothing swings in toward the body.
ARM_LAG, ELBOW_LAG = 0.05, 0.03
# How the arm hangs at rest: ARM_OUT, how far the upper arm is held out from the body at the shoulder; ARM_CARRY, the
# forearm's own angle out from the upper arm at the elbow (the carrying angle, which takes the hand past the hip); and
# for each side (1 her right, -1 her left) the upper arm's swing forward and the elbow's bend in the idle stance.
ARM_OUT, ARM_CARRY = 0.17, 0.0
IDLE_ARM = {1: (0.12, 0.32), -1: (-0.04, 0.2)}
STANCE = 0.62
# The trunk (in radians, and design pixels for the sway): the pelvis turns about its vertical axis with the legs, the
# hip of the leg swinging through carried forward with it; it lists, the hip over the standing leg high and the
# other dropping; and the body sways over whichever foot bears the weight. The chest turns the other way, with the
# arms, and tilts against the pelvis, so the shoulders counter the hips; the head holds steady over it all.
PELVIS_TURN, PELVIS_LIST, CHEST_TURN, CHEST_TILT, SWAY, LEAN = 0.2, 0.075, 1.0, 0.6, 0.85, 0.035
FOOT_X = 0.85          # the feet are set a little inside the hips: a walk is narrower than a stance


def cyclic(keys, t):
    """A smooth closed curve through keys [(t, value)] over t in [0, 1)."""
    t %= 1.0
    n = len(keys)
    for i in range(n):
        t0, v1 = keys[i]
        t1, v2 = keys[(i + 1) % n]
        if i == n - 1:
            t1 += 1.0
        tt = t + 1.0 if t < t0 else t
        if t0 <= tt <= t1:
            u = (tt - t0) / (t1 - t0)
            v0, v3 = keys[(i - 1) % n][1], keys[(i + 2) % n][1]
            return 0.5 * (2 * v1 + (v2 - v0) * u + (2 * v0 - 5 * v1 + 4 * v2 - v3) * u * u + (3 * v1 - v0 - 3 * v2 + v3) * u ** 3)
    return keys[0][1]


def gait_leg(t):
    """(thigh angle forward, knee bend, foot pitch toes-up) in radians, at point t of a leg's cycle. (The hip's
    swing is scaled about its middle by GAIT; the knee and foot follow the same shape.)"""
    hip = 6.5 + (cyclic(GAIT_HIP, t) - 6.5) * GAIT
    return math.radians(hip), math.radians(cyclic(GAIT_KNEE, t)), math.radians(cyclic(GAIT_FOOT, t))


def leg_fk(th, kb):
    """(forward, down) of the ankle from the hip for a thigh angle and knee bend."""
    sh = th - kb
    return THIGH * math.sin(th) + SHIN * math.sin(sh), THIGH * math.cos(th) + SHIN * math.cos(sh)


# how far ahead of the ankle the foot pivots as the heel lifts (the ball of a boot; a longer bare foot sets its own)
FOOT_BALL = 3.0


def foot_lift(pitch):
    """How far the ankle stands above its flat height when the foot is tipped onto its toes (pitch < 0) or its
    heel (pitch > 0)."""
    return FOOT_BALL * math.sin(-pitch) if pitch < 0 else 1.0 * math.sin(pitch)


def pelvis_list(t):
    """The pelvis's tilt about the forward axis at walk phase t (of the right leg): positive lifts the right hip.
    It is highest over the standing leg early in its single support, as the other hip drops off it."""
    return PELVIS_LIST * math.sin(2 * math.pi * (t + 0.05))


def walk_drop(t):
    """How far the hips sit below their standing height at walk phase t (t of the right leg): whichever foot bears
    the weight stays on the ground, so the body rises over the straight leg in mid-stance and dips as the legs
    spread. The swing leg's foot is wherever its joints put it, lifted clear of the ground. (The standing leg's
    hip is raised by the pelvis's list, so the body sits that much lower.)"""
    best = None
    lst = pelvis_list(t)
    for tl, s in ((t, 1), (t + 0.5, -1)):
        tl %= 1.0
        if tl >= STANCE:
            continue
        th, kb, pitch = gait_leg(tl)
        down = leg_fk(th, kb)[1]
        need = HIP_Z - ANKLE_Z - foot_lift(pitch)
        d = need - down + s * LEG_X * math.sin(lst)
        best = d if best is None else min(best, d)     # (the higher: no foot sinks into the ground)
    return best if best is not None else 0.0


# ---- the world model's walk: planted feet
# The sheet's walk above drives each joint from its curve, which is right for a cycle seen one frame at a time, but
# its standing foot does not move back at an even pace (it slides forward in mid-stance and again before toe-off),
# which shows as skating the moment the model covers real ground. The world's walk is built the other way round:
# each ankle follows a path on the ground (still while it bears weight, carried back under the body at exactly the
# pace the body moves forward, then swung through on a smooth arc), the hips ride as high as the standing legs
# allow, smoothed into an easy rise and fall, and the thigh and knee are solved to reach the ankle. One cycle then
# covers exactly STRIDE of ground (WORLD_CYCLE in src/world/main.js), so the feet stay put in the world.
LOCKED_WALK = False     # export_glb.py turns it on for the world model; the sheet keeps the walk above
SKIRT_ON_THIGH = 0.45   # (with LOCKED_WALK) how much of the jerkin's hem each thigh carries: see trunk_weights
STRIDE = 46.0           # ground covered by one cycle (two steps), in model units
STANCE_L = 0.6          # the part of a leg's cycle its foot is on the ground (heel strike at 0, toe-off here)
AHEAD = 0.42            # how much of the stance's travel lies ahead of the hip at the heel strike
SWING_LIFT = 4.0        # how high the ankle is lifted as the foot is carried through (early in the swing)
# how the hips' path is rounded off where one step hands over to the next: held low for BOB_SOFT of the cycle after
# each heel strike (the new standing knee gives a little as it takes the weight), blurred by BOB_BLUR, and joined
# back onto the standing leg's reach within BOB_JOIN, so through the rest of the stance the body rides the standing
# leg with its knee nearly straight instead of crouching under a path lowered all round the cycle
BOB_SOFT, BOB_BLUR, BOB_JOIN = 0.03, 0.02, 0.12
# the foot's pitch from level (toes up positive) in the locked walk: up on the heel as it strikes, flat through the
# stance, the heel peeling up from about halfway and rolling well up onto the ball to push off, then easing level
# through the swing and lifting the toes again to reach for the next heel strike
L_GAIT_FOOT = [(0.0, 16), (0.07, 0), (0.30, 0), (0.40, -8), (0.46, -16), (0.5, -24), (0.56, -48), (0.6, -60),
               (0.68, -33), (0.78, -6), (0.88, 6), (0.95, 14)]
L_PELVIS_TURN = 0.08    # the pelvis's turn either way (the hip of the leg reaching forward carried with it)
# the arms in the locked walk: the upper arm swings ARM_SWING either way; the forearm follows a little later and
# folds more as it comes forward, so the swing travels down the arm instead of the arm swinging stiff from the shoulder
L_ARM_LAG, L_ELBOW_LAG, L_BEND, L_FOLD = 0.04, 0.04, 0.22, 0.16
# (John, 2026-11-18: the arms read too loose, the forearm and hand flopping after the swing. So the elbow keeps a
# small, nearly steady bend, 13 to 22 degrees, the forearm follows almost in step, and the wrist barely moves:
# the arm swings from the shoulder as one piece)
L_WRIST, L_WRIST_SWING = 0.05, 0.02
# (and the swing is centred a little behind straight down, as a relaxed arm swings: further back past the hip than
# it comes forward, where the bending elbow already carries the hand ahead)
L_ARM_BACK = 0.07
# (and held a touch further out from the body than at rest, so the forearm clears the hip as the pelvis sways and
# turns under it: on the wide-hipped figures the arm hanging at ARM_OUT passed half a unit into the hip)
L_ARM_CLEAR = 0.05
# the feet in the locked walk: each turned out a little, as a relaxed walk sets them (pointing dead ahead they read as
# turned in), and set nearly under the hips, so the legs do not lean in and the knees do not come together
L_TOE_OUT, L_FOOT_X = 0.13, 0.97


def _hermite(p0, p1, m0, m1, u):
    return ((2 * u ** 3 - 3 * u ** 2 + 1) * p0 + (u ** 3 - 2 * u ** 2 + u) * m0 + (-2 * u ** 3 + 3 * u ** 2) * p1
            + (u ** 3 - u ** 2) * m1)


def locked_foot(t):
    """(forward of the hip's ground point, ankle height, foot pitch toes-up, on the ground) for a leg at point t of
    its cycle. On the ground the ankle moves back at the body's pace; in the swing it is carried forward on a smooth
    curve that leaves and lands moving a little backward (the foot is drawn back just before the heel strikes), and
    lifted most early on, as the toes leave the ground."""
    t %= 1.0
    a = AHEAD * STANCE_L * STRIDE
    pitch = math.radians(cyclic(L_GAIT_FOOT, t)) - HEEL_PITCH
    if t < STANCE_L:
        return a - STRIDE * t + toe_roll(pitch), heel_ankle_z(pitch), pitch, True
    u = (t - STANCE_L) / (1 - STANCE_L)
    span = STRIDE * (1 - STANCE_L)
    off = toe_roll(math.radians(cyclic(L_GAIT_FOOT, STANCE_L)) - HEEL_PITCH)
    fwd = _hermite(a - STANCE_L * STRIDE + off, a, -0.25 * span, -0.15 * span, u)
    return fwd, heel_ankle_z(pitch) + SWING_LIFT * 6.75 * u * (1 - u) ** 2, pitch, False


def heel_ankle_z(pitch):
    """The ankle's height with the foot at this pitch (toes up positive) standing on the ground: ANKLE_Z over a bare foot's
    heel or ball (foot_lift); over a heeled sole, the height that puts the lowest of HEEL_CONTACT on the floor."""
    if HEEL_CONTACT is None:
        return ANKLE_Z + foot_lift(pitch)
    c, s = math.cos(pitch), math.sin(pitch)
    low = max(-(b * c + a * s) for a, b in HEEL_CONTACT)
    if HEEL_AXIS:
        # (rolling onto the toes the toe bone turns the forefoot flat: the sole hangs its full depth under the toe joint;
        # with the toes up it turns with the foot)
        a, b, depth = HEEL_AXIS
        low = max(low, -(b * c + a * s) + depth * (1.0 if pitch < 0 else c))
    return -HEEL_UP + low


def toe_roll(pitch):
    """How far forward the ankle moves as the heel lifts and the foot rolls over the ball of the foot (3 ahead of
    the ankle, as in foot_lift), so the ball stays where it was planted instead of sliding back under her."""
    return FOOT_BALL * (1.0 - math.cos(pitch)) if pitch < 0 else 0.0


def locked_turn(tr):
    """The pelvis's turn at walk phase tr (of the right leg): the right hip furthest forward as its heel strikes."""
    return L_PELVIS_TURN * math.cos(2 * math.pi * (tr + 0.03))


_LOCKED_HIPS = {}


def locked_hips(tr):
    """The hip joints' height at walk phase tr. The body rides the standing leg: as high as each loaded leg reaches
    with its knee just off straight (and the swing leg, at the end of its swing, reaching the ground with the knee
    straight, so the heel lands where it is set down). Where one step hands over to the next that reach dips sharply
    as the leading heel lands; there it is held low a little after the heel strike, blurred, and joined back onto the
    reach by a soft minimum, so the hips never sit above it and nowhere else sit far below it. (A path smoothed and
    lowered all round the cycle leaves every standing knee bent, a crouch.) The trailing leg, rolling up onto its
    ball to push off, carries no weight then and does not hold the body up."""
    key = (STRIDE, STANCE_L, AHEAD, SWING_LIFT, BOB_SOFT, BOB_BLUR, BOB_JOIN, L_PELVIS_TURN, PELVIS_LIST, FOOT_BALL,
           tuple(L_GAIT_FOOT), HEEL_PITCH, HEEL_UP, tuple(HEEL_CONTACT or ()), tuple(HEEL_AXIS or ()))
    if key not in _LOCKED_HIPS:
        n = 192
        reach = np.zeros(n)
        for i in range(n):
            t = i / n
            lst, turn, best = pelvis_list(t), locked_turn(t), 1e9
            for s, off in ((1, 0.0), (-1, 0.5)):
                tl = (t + off) % 1.0
                fwd, az, _, ground = locked_foot(tl)
                if ground and tl < 0.5:
                    leg = (THIGH + SHIN) * 0.997
                elif tl > 0.9:
                    leg = (THIGH + SHIN) * 0.9995
                else:
                    continue
                f = fwd - s * LEG_X * math.sin(turn)
                best = min(best, az + math.sqrt(max(0.0, leg * leg - f * f)) - s * LEG_X * math.sin(lst))
            reach[i] = best
        # (each point lowered to the least reach over the BOB_SOFT before it, so the dip at the heel strike lasts a
        # little into the step; blurred; then the soft minimum with the reach itself)
        w = max(1, int(round(BOB_SOFT * n)))
        low = np.min([np.roll(reach, j) for j in range(w + 1)], axis=0)
        g = np.exp(-0.5 * (np.minimum(np.arange(n), n - np.arange(n)) / (BOB_BLUR * n)) ** 2)
        low = np.real(np.fft.ifft(np.fft.fft(low) * np.fft.fft(g / g.sum())))
        _LOCKED_HIPS[key] = -BOB_JOIN * np.logaddexp(-reach / BOB_JOIN, -low / BOB_JOIN)
    h = _LOCKED_HIPS[key]
    x = (tr % 1.0) * len(h)
    i = int(x)
    return float(h[i % len(h)] + (h[(i + 1) % len(h)] - h[i % len(h)]) * (x - i))


# The head poses of the pose test strip (rq_views.py --poses): (nod, tilt, turn) in degrees from looking level
# ahead, and the upper joints' share (None: NECK_SHARE). Each is near the neck's range (NECK_ROM): 'nod' is the
# skull rocking on the atlas alone, the most that joint gives; the rest are the whole neck.
HEAD_POSES = {
    'nod': (20, 0, 0, (1.0, 1.0, 1.0)),
    'look-down': (45, 0, 0, None),
    'look-up': (-50, 0, 0, None),
    'tilt': (0, 40, 0, None),
    'turn': (0, 0, 70, None),
}


def carry_head(nod=0.0, tilt=0.0, turn=0.0, share=None):
    """Carry the head on the neck, looking where (nod, tilt, turn) say (radians, in the world: nod chin down,
    tilt toward her left, turn toward her left; all 0 is level and straight ahead), whatever the chest under it
    is doing. The neck takes the turn from the chest to there, split between its joints: share is the upper
    joints' (the atlas's) share of nod, tilt and turn (NECK_SHARE by default), and the rest is split between the
    mid-neck and the base of the neck by MID_SHARE. So a leaning, twisting chest carries the head with it, and
    the neck keeps the face level, as a real neck does in a walk."""
    ob = bpy.data.objects
    if 'Midneck' not in ob or 'head_rest' not in NECK:
        return
    chest = ob['Body'].rotation_euler.to_matrix() @ ob['Chest'].rotation_euler.to_matrix()
    rel = chest.transposed() @ Euler((nod, tilt, turn), 'XYZ').to_matrix()
    e = rel.to_euler('XYZ')
    sn, st, sr = share or (NECK_SHARE[0] if e.x >= 0 else NECK_SHARE_EXT,) + NECK_SHARE[1:]
    rest = [(1 - u) * a for u, a in zip((sn, st, sr), e)]
    low = Euler([r * (1 - m) for r, m in zip(rest, MID_SHARE)], 'XYZ')
    mid = Euler([r * m for r, m in zip(rest, MID_SHARE)], 'XYZ')
    ob['Cervical'].rotation_euler = low
    ob['Midneck'].rotation_euler = mid
    ob['Atlas'].rotation_euler = ((low.to_matrix() @ mid.to_matrix()).transposed() @ rel).to_euler('XYZ')
    ob['Head'].location, ob['Head'].rotation_euler = NECK['head_rest'], (0, 0, 0)


def head_pose(name):
    """One of HEAD_POSES, on whatever the body is doing."""
    nod, tilt, turn, share = HEAD_POSES[name]
    carry_head(math.radians(nod), math.radians(tilt), math.radians(turn), share)


GAZE = [0.0]     # the head's nod for the pose being set (the run's; see pose_run)


def pose(col):
    ob = bpy.data.objects
    walking = col >= 2
    GAZE[0] = 0.0
    if 'Atlas' not in ob:      # (before the neck's joints are seated: the head on the body, at rest)
        ob['Head'].location, ob['Head'].rotation_euler = (0, 0, HEAD_Z), (0, 0, 0)
    set_fists(RUN_FIST if walking and RUN else 0.0)       # (and closes the hands)
    if walking and RUN:
        pose_run(col)
    elif walking and LOCKED_WALK:
        pose_locked(col)
    else:
        pose_sheet(col, walking)
    # (the head rides the chest and the neck keeps it looking ahead: level, or the run's slight nod)
    carry_head(GAZE[0])


# The idle's feet, each (forward of the hip, out from it, turned out in radians): the weight on the right leg under the
# body, the left foot eased forward, out, and turned out. A figure may set its own (the heeled boots stand square).
IDLE_STANCE = {'R': (-1.0, 0.0, 0.0), 'L': (4.6, 0.6, 0.35)}


def pose_sheet(col, walking):
    ob = bpy.data.objects
    # the walk is sampled half a step off the passing pose (as in the painted sheets), so no frame has the feet together
    ph = (col - 2 + 0.5) / WALK * 2 * math.pi if walking else 0.0
    # (the right leg's point in its cycle: its heel strikes when its foot is furthest forward)
    tr = ((ph - math.pi / 2) / (2 * math.pi)) % 1.0
    snap = lambda v: round(v * RES) / RES          # whole sheet pixels, so the face never shifts by a fraction
    if walking:
        bob = walk_drop(tr)
        thr, thl = gait_leg(tr)[0], gait_leg(tr + 0.5)[0]
        turn = PELVIS_TURN * (thr - thl)      # the hip of the forward leg carried forward
        lst = pelvis_list(tr)
        sway = -SWAY * math.sin(2 * math.pi * (tr - 0.06))   # over the right foot in its mid-stance (-X is her right)
    else:
        bob = 1.0 if col == 1 else 0.0
        turn = lst = sway = 0.0
    bob, sway = snap(bob), snap(sway)
    ob['Body'].location = (sway, 0, -bob)
    ob['Body'].rotation_euler = (0, 0, 0)
    ob['Pelvis'].rotation_euler = (0, lst, turn)
    ob['Chest'].rotation_euler = (LEAN if walking else 0.0, -CHEST_TILT * lst, -CHEST_TURN * turn)
    hip_z = HIP_Z - bob

    for side, s in (('R', 1), ('L', -1)):
        hip, kn, an = ob[f'Hip.{side}'], ob[f'Knee.{side}'], ob[f'Ankle.{side}']
        sh, el, wr = ob[f'Shoulder.{side}'], ob[f'Elbow.{side}'], ob[f'Wrist.{side}']
        out, toe = 0.0, 0.0
        roll = 0.0
        if walking:
            t = (tr + (0.0 if s > 0 else 0.5)) % 1.0
            th, kb, pitch = gait_leg(t)
            # the leg leans in under the body so the foot keeps to its track, whichever way the body has swayed and
            # the pelvis tilted (the hip's own roll is what is left after the pelvis's)
            down = leg_fk(th, kb)[1]
            hip_x = sway - s * LEG_X * math.cos(lst)
            roll = math.asin(max(-0.5, min(0.5, (hip_x + s * LEG_X * FOOT_X) / down))) - lst
        else:
            # hip-width stance: weight on the right leg, under the body; the left foot eased forward, out, and turned out
            fwd, out, toe = IDLE_STANCE['R' if s > 0 else 'L']
            # (on a heel, HEEL_CONTACT: the foot stands pitched onto it, the ankle as high as the tread sets it)
            pitch = -HEEL_PITCH if HEEL_CONTACT is not None else 0.0
            th, shn = leg_ik(hip_z, fwd, heel_ankle_z(pitch) if HEEL_CONTACT is not None else ANKLE_Z)
            kb = th - shn
        hip.location = (-s * (LEG_X + out), 0, 0)
        hip.rotation_euler = (-th, roll, 0)
        kn.rotation_euler = (kb, 0, 0)
        # (the foot is held at its pitch from level: the ankle takes out the shin's lean, and the foot is turned
        # back against the pelvis's turn, so it points down the line of the walk)
        an.rotation_euler = (th - kb - pitch, 0, -s * toe - turn)
        set_toe(side, pitch)

        # arms: hang just off the body with a soft bend; in the walk they swing against the legs
        if walking:
            a = ARM_SWING * math.cos(2 * math.pi * (t - 0.5 - ARM_LAG))
            fold = 0.5 + 0.5 * math.cos(2 * math.pi * (t - 0.5 - ARM_LAG - ELBOW_LAG))
            bend = 0.42 + 0.2 * fold
            flick = 0.08 + 0.05 * math.cos(2 * math.pi * (t - 0.5 - ARM_LAG - 2 * ELBOW_LAG))
        else:
            a, bend = IDLE_ARM[s]
            flick = 0.1
        # (held a little off the body, so the waist shows, and a little further out as it swings back, past the hip)
        sh.rotation_euler = (-a, s * (ARM_OUT + 0.05 * max(0.0, -a)), 0)
        el.rotation_euler = (-bend, s * ARM_CARRY, 0)
        wr.rotation_euler = (flick, 0, 0)


# How far an ankle bends, degrees: toes drawn up toward the shin (ANKLE_UP) and pointed away from it (ANKLE_DOWN).
# A foot in the air is posed by its pitch from level, which says nothing of the shin above it: with the shin swung
# far back (the walk's swing, and far more the run's heel coming up behind) a nearly level foot was folded 50 to 60
# degrees up against the shin, a bend no ankle makes (John, 2026-11-18, of the run: fix the ankles).
ANKLE_UP, ANKLE_DOWN = 14.0, 38.0


def ankle_limit(pitch, shn):
    """The pitch (toes up positive) of a foot in the air, kept within what the ankle can bend from the shin, whose
    angle forward of straight down is shn (as leg_ik gives it). The limits are eased into, not struck."""
    w = 0.07
    soft = lambda v: w * math.log1p(math.exp(min(30.0, v / w)))       # noqa: E731
    return pitch - soft(pitch - shn - math.radians(ANKLE_UP)) + soft(shn - math.radians(ANKLE_DOWN) - pitch)


def _euler_m(e):
    """The matrix of a Blender XYZ Euler (X applied first)."""
    (ca, sa), (cb, sb), (cc, sc) = [(math.cos(v), math.sin(v)) for v in e]
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rz @ ry @ rx


def set_toe(side, pitch, k=1.0):
    """Bend the toes of a foot with a toe bone: when the foot is pitched toes-down (pitch < 0) to roll over its ball,
    the toes stay as flat as the floor lets them (k = 1), eased off to k in the swing. A bend is the foot's pitch
    taken back out at the toe joint."""
    t = bpy.data.objects.get(f'Toe.{side}')
    if t is not None:
        t.rotation_euler = (min(0.0, pitch) * k, 0, 0)


def swing_toe(u):
    """How much of the push-off's toe bend the toes keep u (0-1) of the way through the swing: all of it as the foot
    leaves the ground, 0.35 of it a quarter of the way on."""
    return 0.35 + 0.65 * max(0.0, 1.0 - u / 0.25)


def foot_level(pelvis, hip, knee, pitch, toe=0.0):
    """The ankle's XYZ Euler that sets the foot pointing straight down the line of the walk, its sole level from
    side to side, pitched only by `pitch` (toes up positive) and turned out by `toe`, whatever the pelvis's turn and list and the leg's
    roll above it. (Taking each of those out one angle at a time, as pose() does, leaves the foot turned and rolled
    by a few degrees once the shin leans far over, most at push-off, where it shows as the foot twisting on its toes.)"""
    parent = _euler_m(pelvis) @ _euler_m(hip) @ _euler_m(knee)
    m = parent.T @ _euler_m((-pitch, 0.0, toe))
    return (math.atan2(m[2, 1], m[2, 2]), -math.asin(max(-1.0, min(1.0, m[2, 0]))), math.atan2(m[1, 0], m[0, 0]))


def pose_locked(col):
    """The world model's walk (LOCKED_WALK) at sheet column col: feet planted on their paths (locked_foot), the
    hips at locked_hips, the legs solved to reach, and the arms swinging against the legs with the forearm following
    through. Same rig and the same trunk motion as pose()."""
    ob = bpy.data.objects
    ph = (col - 2 + 0.5) / WALK * 2 * math.pi
    tr = ((ph - math.pi / 2) / (2 * math.pi)) % 1.0
    hips = locked_hips(tr)
    bob = HIP_Z - hips
    turn, lst = locked_turn(tr), pelvis_list(tr)
    sway = -SWAY * math.sin(2 * math.pi * (tr - 0.06))
    ob['Body'].location = (sway, 0, -bob)
    ob['Body'].rotation_euler = (0, 0, 0)
    ob['Pelvis'].rotation_euler = (0, lst, turn)
    # (a touch more lean as each foot pushes off, easing back as the body rides up over the standing leg)
    lean = LEAN + 0.012 * math.cos(4 * math.pi * (tr + 0.08))
    ob['Chest'].rotation_euler = (lean, -CHEST_TILT * lst, -CHEST_TURN * turn)
    for side, s in (('R', 1), ('L', -1)):
        hip, kn, an = ob[f'Hip.{side}'], ob[f'Knee.{side}'], ob[f'Ankle.{side}']
        sh, el, wr = ob[f'Shoulder.{side}'], ob[f'Elbow.{side}'], ob[f'Wrist.{side}']
        t = (tr + (0.0 if s > 0 else 0.5)) % 1.0
        fwd, az, pitch, ground = locked_foot(t)
        f = fwd - s * LEG_X * math.sin(turn)            # (the pelvis's turn carries the hip forward or back)
        hz = hips + s * LEG_X * math.sin(lst)           # (and its list lifts or drops it)
        # the leg leans in under the body so the foot keeps to its track; the lean lengthens the reach a little
        hip_x = sway - s * LEG_X * math.cos(lst)
        roll = math.asin(max(-0.5, min(0.5, (hip_x + s * LEG_X * L_FOOT_X) / (hz - az)))) - lst
        th, shn = leg_ik(hz, f, hz - (hz - az) / math.cos(roll + lst))
        kb = th - shn
        if not ground:
            pitch = ankle_limit(pitch, shn)
        hip.location = (-s * LEG_X, 0, 0)
        hip.rotation_euler = (-th, roll, 0)
        kn.rotation_euler = (kb, 0, 0)
        an.rotation_euler = foot_level((0, lst, turn), (-th, roll, 0), (kb, 0, 0), pitch, -s * L_TOE_OUT)
        set_toe(side, pitch, 1.0 if ground else swing_toe((t - STANCE_L) / (1 - STANCE_L)))
        a = ARM_SWING * math.cos(2 * math.pi * (t - 0.5 - L_ARM_LAG)) - L_ARM_BACK
        fold = 0.5 + 0.5 * math.cos(2 * math.pi * (t - 0.5 - L_ARM_LAG - L_ELBOW_LAG))
        flick = L_WRIST + L_WRIST_SWING * math.cos(2 * math.pi * (t - 0.5 - L_ARM_LAG - 2 * L_ELBOW_LAG))
        sh.rotation_euler = (-a, s * (ARM_OUT + L_ARM_CLEAR + 0.05 * max(0.0, -a)), 0)
        el.rotation_euler = (-(L_BEND + L_FOLD * fold), s * ARM_CARRY, 0)
        wr.rotation_euler = (flick, 0, 0)


# ---- the run
# Built like the world's walk (planted feet, the legs solved to reach them), but a run is a different gait, not a
# fast walk: each foot is on the ground for only about a third of the cycle, so between one foot pushing off and the
# other landing she is in the air; the body is lowest in mid-stance, the standing knee bent well under it like a
# spring, and highest in mid-flight; the foot lands nearly flat and close under the hip instead of reaching out onto
# the heel; behind her the heel is whipped up toward the seat, the knee folding past a right angle, and the knee is
# then driven forward and up before the lower leg reaches down for the ground, already moving back as it lands.
# The trunk leans into it, the elbows are held at about a right angle, and the arms pump from the shoulder.
RUN = False             # pose() plays the run instead of the walk (views.py --run; an exporter keying a run clip)
RUN_STRIDE = 2.8        # the ground one cycle (two steps) covers, in leg lengths, thigh + shin (see run_stride)
RUN_SECONDS = 0.64      # one cycle at that pace
RUN_STANCE = 0.32       # the part of a leg's cycle its foot is on the ground (the other lands at 0.5: flight between)
RUN_AHEAD = 0.30        # how much of the stance's travel lies ahead of the hip as the foot lands
RUN_SINK = 46.0         # the standing knee's bend in mid-stance, degrees
RUN_RISE = 0.075        # how far the hips rise from mid-stance to mid-flight, in leg lengths
# the hips' height through one step (half a cycle: how far through it, how much of RUN_RISE): already coming down as
# the foot lands, lowest as the standing knee takes the weight, driven up through the push-off (the leg straightening
# and the foot rolling onto its ball lift her far more than the short flight does), and over the top in the air
R_HIPS = [(0.0, 0.34), (0.3, 0.0), (0.64, 0.86), (0.82, 1.0)]
# the foot's pitch from level (toes up positive): landing nearly flat, the heel peeling up from mid-stance and the
# foot rolled far onto the ball as it pushes off; trailing, toes pointed back, as the heel comes up behind; then
# drawn level as the knee drives through, and the toes lifted a little to meet the ground
R_GAIT_FOOT = [(0.0, 6), (0.04, 0), (0.14, 0), (0.21, -14), (0.27, -38), (0.32, -62), (0.42, -70), (0.56, -52),
               (0.72, -20), (0.84, 2), (0.94, 8)]
# the ankle's path through the swing: (how far through the swing, forward of the hip, height above its standing
# height), the last two in leg lengths (thigh + shin), between leaving the ground and landing. Trailing behind and
# rising; the heel at its highest behind the seat; the knee driven through; the lower leg reaching down ahead
R_SWING = [(0.20, -0.66, 0.36), (0.42, -0.30, 0.47), (0.64, 0.17, 0.27), (0.86, 0.33, 0.10)]
RUN_LEAN, RUN_HEAD = 0.15, 0.3      # the chest's lean into the run, and how much of it the head follows
R_PELVIS_TURN, R_PELVIS_LIST, R_SWAY = 0.12, 0.05, 0.35
# the arms: the upper arm pumps R_ARM_SWING either way about a line R_ARM_BACK behind straight down (the elbow is
# drawn well back past the ribs, the hand comes forward to about chest height); the elbow is held near a right angle,
# closing a little as the arm comes forward and opening as it drives back
R_ARM_SWING, R_ARM_BACK, R_ARM_LAG = 0.62, 0.12, 0.02
R_BEND, R_FOLD, R_ARM_CLEAR = 1.38, 0.5, 0.07
# the wrist (John, 2026-11-19: the hand looked off, dead in line with the forearm like a paddle; then, of a wrist
# that drooped: the hands should bend up, not down). A running fist is carried cocked: with the thumb on top it is
# held a little up from the forearm's line (R_WRIST, toward the thumb, so negative about the wrist's X), gives a
# touch with the arm's swing (R_WRIST_LAG), is curled in toward the body (R_WRIST_IN) and turned so the palm faces
# partly down (R_WRIST_TURN). All well inside what a wrist bends: about 20 degrees toward the thumb, 60 folding in.
R_WRIST, R_WRIST_LAG, R_WRIST_IN, R_WRIST_TURN = -0.2, 0.05, 0.14, 0.2
# the ankle through the swing: (how far through the swing, the foot pointed away from the shin, degrees). In the air
# the foot hangs from the shin, loose and pointed as the heel comes up behind, then drawn up to meet the ground. (It
# is eased in from the push-off's pitch and out to the landing's over the first and last R_ANKLE_EASE of the swing.)
R_SWING_ANKLE = [(0.0, 20), (0.2, 26), (0.5, 16), (0.75, 4), (0.9, -5), (1.0, -5)]
R_ANKLE_EASE = 0.16
RUN_FIST = 1.0          # how far the hands are closed into fists in the run (John, 2026-11-19: a closed fist)
R_TOE_OUT, R_FOOT_X = 0.06, 0.8     # the feet point nearly straight ahead and land closer to the line of travel


def run_stride():
    """The ground one cycle of the run covers, in model units (what the world must move her by per cycle)."""
    return RUN_STRIDE * (THIGH + SHIN)


def run_foot(t):
    """(forward of the hip's ground point, ankle height, foot pitch toes-up, on the ground) for a leg at point t of
    its run cycle: on the ground for RUN_STANCE, moving back at the body's pace, then carried through R_SWING."""
    t %= 1.0
    S = run_stride()
    a = RUN_AHEAD * RUN_STANCE * S
    # (on a heel the foot is held pitched onto it, and stands as high as the tread sets it: see heel_ankle_z)
    pitch = math.radians(cyclic(R_GAIT_FOOT, t)) - HEEL_PITCH
    if t < RUN_STANCE:
        return a - S * t + toe_roll(pitch), heel_ankle_z(pitch), pitch, True
    u = (t - RUN_STANCE) / (1 - RUN_STANCE)
    L = THIGH + SHIN
    off = math.radians(cyclic(R_GAIT_FOOT, RUN_STANCE)) - HEEL_PITCH
    keys = ([(0.0, a - RUN_STANCE * S + toe_roll(off), heel_ankle_z(off))]
            + [(k, f * L, ANKLE_Z + z * L) for k, f, z in R_SWING]
            + [(1.0, a, heel_ankle_z(math.radians(cyclic(R_GAIT_FOOT, 0.0)) - HEEL_PITCH))])
    # (a smooth curve through the keys that leaves the ground and meets it again at the ground's own pace)
    span = S * (1 - RUN_STANCE)
    tan = [(-0.3 * span, 0.8 * L)] + [((keys[i + 1][1] - keys[i - 1][1]) / (keys[i + 1][0] - keys[i - 1][0]),
                                  (keys[i + 1][2] - keys[i - 1][2]) / (keys[i + 1][0] - keys[i - 1][0]))
                                 for i in range(1, len(keys) - 1)] + [(-span, -0.12 * L)]
    for i in range(len(keys) - 1):
        if u <= keys[i + 1][0]:
            break
    d = keys[i + 1][0] - keys[i][0]
    v = (u - keys[i][0]) / d
    fwd = _hermite(keys[i][1], keys[i + 1][1], tan[i][0] * d, tan[i + 1][0] * d, v)
    z = _hermite(keys[i][2], keys[i + 1][2], tan[i][1] * d, tan[i + 1][1] * d, v)
    return fwd, max(z, heel_ankle_z(pitch)), pitch, False


_RUN_HIPS = {}


def run_hips(tr):
    """The hip joints' height at run phase tr: lowest in each stance, where the standing knee is bent RUN_SINK
    under the body, and RUN_RISE higher in each flight (R_HIPS). Never above what the standing leg can reach (a
    soft minimum with that reach, as in locked_hips), so a planted foot stays planted on a figure whose legs or feet
    are too short for the full rise at push-off."""
    key = (RUN_STRIDE, RUN_STANCE, RUN_AHEAD, RUN_SINK, RUN_RISE, R_PELVIS_TURN, R_PELVIS_LIST, FOOT_BALL, THIGH, SHIN,
           ANKLE_Z, BOB_JOIN, tuple(R_HIPS), tuple(R_GAIT_FOOT), HEEL_PITCH, HEEL_UP, tuple(HEEL_CONTACT or ()),
           tuple(HEEL_AXIS or ()))
    if key not in _RUN_HIPS:
        n = 192
        L = THIGH + SHIN
        low = heel_ankle_z(-HEEL_PITCH) + math.sqrt(THIGH ** 2 + SHIN ** 2 + 2 * THIGH * SHIN * math.cos(math.radians(RUN_SINK)))
        want, reach = np.zeros(n), np.full(n, 1e3)
        for i in range(n):
            t = i / n
            want[i] = low + RUN_RISE * L * cyclic(R_HIPS, 2 * t)
            turn, lst = run_turn(t), run_list(t)
            for s, off in ((1, 0.0), (-1, 0.5)):
                fwd, az, _, ground = run_foot(t + off)
                if ground:
                    f = fwd - s * LEG_X * math.sin(turn)
                    reach[i] = min(reach[i], az + math.sqrt(max(0.0, (0.992 * L) ** 2 - f * f)) - s * LEG_X * math.sin(lst))
        _RUN_HIPS[key] = -BOB_JOIN * np.logaddexp(-reach / BOB_JOIN, -want / BOB_JOIN)
    h = _RUN_HIPS[key]
    x = (tr % 1.0) * len(h)
    i = int(x)
    return float(h[i % len(h)] + (h[(i + 1) % len(h)] - h[i % len(h)]) * (x - i))


def run_turn(tr):
    """The pelvis's turn at run phase tr: the hip of the leg reaching forward carried forward with it."""
    return R_PELVIS_TURN * math.cos(2 * math.pi * (tr - 0.04))


def run_list(tr):
    """The pelvis's list: it drops on the side of the leg in the air as the standing leg takes the landing."""
    return R_PELVIS_LIST * math.sin(2 * math.pi * (tr + 0.07))


def pose_run(col):
    """The run at sheet column col (the same columns and phase as the walk: the right foot lands at the start of
    the cycle). Same rig as pose_locked."""
    ob = bpy.data.objects
    ph = (col - 2 + 0.5) / WALK * 2 * math.pi
    tr = ((ph - math.pi / 2) / (2 * math.pi)) % 1.0
    hips = run_hips(tr)
    turn, lst = run_turn(tr), run_list(tr)
    sway = -R_SWAY * math.sin(2 * math.pi * (tr - 0.07))
    ob['Body'].location = (sway, 0, hips - HIP_Z)
    ob['Body'].rotation_euler = (0, 0, 0)
    ob['Pelvis'].rotation_euler = (0, lst, turn)
    # (the trunk folds a little further forward as each foot lands and comes up through the flight)
    lean = RUN_LEAN + 0.02 * math.cos(4 * math.pi * (tr - 0.06))
    ob['Chest'].rotation_euler = (lean, -CHEST_TILT * lst, -CHEST_TURN * turn)
    # (the head is carried forward on the leaning chest, and stays nearly upright on it, looking where she runs:
    # see carry_head, which pose() calls after this)
    GAZE[0] = RUN_HEAD * lean
    for side, s in (('R', 1), ('L', -1)):
        hip, kn, an = ob[f'Hip.{side}'], ob[f'Knee.{side}'], ob[f'Ankle.{side}']
        sh, el, wr = ob[f'Shoulder.{side}'], ob[f'Elbow.{side}'], ob[f'Wrist.{side}']
        t = (tr + (0.0 if s > 0 else 0.5)) % 1.0
        fwd, az, pitch, ground = run_foot(t)
        f = fwd - s * LEG_X * math.sin(turn)
        hz = hips + s * LEG_X * math.sin(lst)
        hip_x = sway - s * LEG_X * math.cos(lst)
        roll = math.asin(max(-0.5, min(0.5, (hip_x + s * LEG_X * R_FOOT_X) / max(8.0, hz - az)))) - lst
        th, shn = leg_ik(hz, f, hz - (hz - az) / math.cos(roll + lst))
        kb = th - shn
        if not ground:
            u = (t - RUN_STANCE) / (1 - RUN_STANCE)
            hang = shn - math.radians(float(np.interp(u, [k for k, _ in R_SWING_ANKLE], [v for _, v in R_SWING_ANKLE])))
            e = min(1.0, u / R_ANKLE_EASE, (1 - u) / R_ANKLE_EASE)
            e = e * e * (3 - 2 * e)
            pitch = ankle_limit(pitch + (hang - pitch) * e, shn)
        hip.location = (-s * LEG_X, 0, 0)
        hip.rotation_euler = (-th, roll, 0)
        kn.rotation_euler = (kb, 0, 0)
        an.rotation_euler = foot_level((0, lst, turn), (-th, roll, 0), (kb, 0, 0), pitch, -s * R_TOE_OUT)
        set_toe(side, pitch, 1.0 if ground else swing_toe((t - RUN_STANCE) / (1 - RUN_STANCE)))
        c = math.cos(2 * math.pi * (t - 0.5 - R_ARM_LAG))
        a = R_ARM_SWING * c - R_ARM_BACK
        sh.rotation_euler = (-a, s * (ARM_OUT + R_ARM_CLEAR + 0.06 * max(0.0, -a)), 0)
        el.rotation_euler = (-(R_BEND + R_FOLD * (0.5 + 0.5 * c)), s * ARM_CARRY, 0)
        droop = R_WRIST - R_WRIST_LAG * math.sin(2 * math.pi * (t - 0.5 - R_ARM_LAG - 0.04))
        wr.rotation_euler = (droop, -s * R_WRIST_IN, s * R_WRIST_TURN)


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
FLUSH_EDGE = 0.55     # how flushed the skin must be for a pixel to take the rosier ramp


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
    shaded = sun * (1.0 - 0.25 * cr) + fill * (0.2 + 0.8 * ao) - 0.1 * cr - 0.2 * turn
    return np.where(sel, shaded, light)


def to_pixels(ch):
    alpha = find(ch, 'Combined', 'A')
    # every pass comes premultiplied by coverage: divide it out, or the edge pixels read as another material
    cov = np.maximum(alpha, 1e-4)
    mid = np.rint(find(ch, 'matid') / cov * 100).astype(int)
    det = find(ch, 'detail') / cov
    flush = find(ch, 'flush') / cov
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
        if d['alt'] is not None:
            # the flushed parts (cheeks, nose, knuckles, fingertips) take the rosier ramp, in the same tones
            alt = flush[sel] > FLUSH_EDGE
            px = out[sel, :3]
            px[alt] = d['alt'][tone[alt]]
            out[sel, :3] = px
        out[sel, 3] = 255
        tones[sel] = tone
    # the brooch casts a pixel of shadow onto the wool below and right of it (the light is from the upper left),
    # so it sits in the cloth instead of on top of it
    for m, d in MATERIALS.items():
        for r in d['shadow_on']:
            gold = solid & (mid == m)
            near = np.zeros_like(gold)
            near[1:, :] |= gold[:-1, :]
            near[1:, 1:] |= gold[:-1, :-1]
            near[:, 1:] |= gold[:, :-1]
            sh = near & solid & (mid == r) & (tones > 0)
            out[sh, :3] = MATERIALS[r]['ramp'][tones[sh] - 1]
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
    yy, xx = np.mgrid[0:h, 0:w]
    for m, d in MATERIALS.items():
        if not d['soft_edge']:
            continue
        wool = np.zeros((h, w), dtype=bool)
        for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nb = np.zeros_like(wool)
            ys0, ys1 = max(0, -dy), h - max(0, dy)
            xs0, xs1 = max(0, -dx), w - max(0, dx)
            nb[ys0:ys1, xs0:xs1] = (solid & (mid == m))[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
            wool |= nb
        soft = wool & ~filled & (out[..., 3] > 0) & (((xx * 5 + yy * 3) // 2) % 3 == 0)
        out[soft, :3] = out[soft, :3] * 0.55 + d['ramp'][0] * 0.45
    return np.clip(out, 0, 255).astype(np.uint8)


def sprite_mode(on):
    """At sprite size the hair's strand relief is finer than a pixel and only adds speckle; the painted strands
    (the detail pass) carry it there instead, so the relief is turned down for the sheet; the iris is drawn larger
    (see below)."""
    hair_names = [d['name'] for d in MATERIALS.values() if d['surface'] == 'hair']
    for m in [m for m in bpy.data.materials if m.name in hair_names
              or any(m.name.startswith(n + ' ') for n in hair_names)]:
        nt = m.node_tree
        if 'sprite bsdf' not in nt.nodes or 'hair bsdf' not in nt.nodes:
            continue
        for n in nt.nodes:
            if n.type == 'BUMP':
                n.inputs['Strength'].default_value = 0.1 if on else 0.6
        # (and the sheet is lit from the plain diffuse surface, its tones picked from the palette ramp; a hair cut-out
        # wrapped round the close-up surface ('tip cut', a figure's own copy of the hair) is for the 3D view only)
        out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
        cut = nt.nodes.get('tip cut')
        if on or cut is None:
            nt.links.new(nt.nodes['sprite bsdf' if on else 'hair bsdf'].outputs['BSDF'], out.inputs['Surface'])
        else:
            nt.links.new(nt.nodes['hair bsdf'].outputs['BSDF'], cut.inputs[2])
            nt.links.new(cut.outputs['Shader'], out.inputs['Surface'])
    # the wool, the gold, the leather, the cloth and the skin, likewise: their close-up surfaces are for the 3D
    # view only
    for name in [d['name'] for d in MATERIALS.values() if d['closeup'] and d['name'] in bpy.data.materials]:
        nt = bpy.data.materials[name].node_tree
        out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
        nt.links.new(nt.nodes['sprite bsdf' if on else 'close bsdf'].outputs['BSDF'], out.inputs['Surface'])
    # the iris is life-size, a pixel and a bit across in the sheet, and in the 3/4 views it fell between pixel
    # centres and the eye showed as a blank white: for the sprites it is drawn larger, as a painter would
    for o in bpy.data.objects:
        if o.name.startswith('Wool fuzz'):
            o.hide_render = on
        if o.name.startswith(('Eyeshadow', 'Lashes', 'Brow paint', 'Lip paint', 'Hair paint')):
            o.hide_render = on         # (finer than a sprite pixel: nothing to draw there)
            o.hide_viewport = on
        if o.name.startswith('Brow tube'):
            o.hide_render = not on
            o.hide_viewport = not on
        if o.name == 'Mouth':
            # (the line between the lips is the sprites' mouth; up close the lip paint draws it, and the solid
            # line stood forward of the paint in the corners as a pale sliver with a hard shadow under it)
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
    root = FIG['root']
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
