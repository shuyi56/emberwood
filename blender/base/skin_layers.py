"""Skin built the way a painter builds it, in three layers, each one shown on its own and then all together.

    1  the base: a deep, mottled red-and-brown flesh (the muscle and the capillary bed under the skin)
    2  the vascular overlay: a fine branching network of blue, purple and green veins over layer 1
    3  the epidermis: the skin's own tone, with freckles, pores and a few blemishes, painted mostly transparent

skin_layers(nt, out, show, tone) builds one material into a node tree. show is '1', '2' or '3' for a layer by
itself (a flat diffuse surface, so the colour is read as painted), or 'all' for the combined skin: layer 3
laid over layers 1 and 2 at EPIDERMIS_OPACITY (more where the pigment is: the freckles), on a Principled surface
with subsurface scattering, so what lies under the epidermis shows through it and the veins stay soft.

Like the figure's own skin it reads its patterns from the 'rest' point attribute, and the freckle and pore
masks from the 'skin tex' point colour (red: freckles, blue: pores); see female.skin_tones. Nothing here builds a
figure. skin_study.py --layers renders the four side by side."""
import numpy as np

# how much of the colour of the epidermis lies over what is under it (0 clear, 1 opaque)
EPIDERMIS_OPACITY = 0.45
# the layers' colours (sRGB)
MUSCLE_RED = [162, 56, 50]
MUSCLE_BROWN = [150, 82, 64]
# (the veins' colours are what they let through, multiplied into what lies under them: blood in a vessel absorbs
# the light that crosses it, as the thin absorbing layer between the epidermis and the dermis does in Donner and
# Jensen's layered model, rather than adding paint on top. Lighter than the colour they leave.)
VEIN_BLUE = [120, 150, 240]
VEIN_PURPLE = [190, 130, 215]
VEIN_GREEN = [140, 215, 160]
FRECKLE = [150, 96, 66]
BLEMISH = [168, 98, 84]


def _lin(c):
    c = np.asarray(c, dtype=np.float64) / 255.0
    return tuple(float(x) for x in np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92))


def skin_layers(nt, out, show='all', tone=(207, 159, 132)):
    L = nt.links.new

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

    def tex(kind, scale, vec=None, **props):
        n = node(kind, **props)
        n.inputs['Scale'].default_value = scale
        L(vec if vec is not None else rest.outputs['Vector'], n.inputs['Vector'])
        return n

    def mix(a, b, fac):
        n = node('ShaderNodeMix', data_type='RGBA')
        if isinstance(fac, (int, float)):
            n.inputs['Factor'].default_value = fac
        else:
            L(fac, n.inputs['Factor'])
        for k, v in (('A', a), ('B', b)):
            if isinstance(v, (list, tuple)):
                n.inputs[k].default_value = (*_lin(v), 1.0)
            else:
                L(v, n.inputs[k])
        return n.outputs['Result']

    rest = node('ShaderNodeAttribute', attribute_name='rest')
    where = node('ShaderNodeSeparateColor')
    L(node('ShaderNodeVertexColor', layer_name='skin tex').outputs['Color'], where.inputs['Color'])
    freckly, pored = where.outputs['Red'], where.outputs['Blue']

    # ---- layer 1: the base. Deep red and brown in broad mottled patches, and a fine speckle of capillaries
    patch = tex('ShaderNodeTexNoise', 1.5)
    patch.inputs['Detail'].default_value = 4.0
    patch.inputs['Roughness'].default_value = 0.6
    c1 = mix(MUSCLE_RED, MUSCLE_BROWN, ramp_(patch.outputs['Fac'], 0.4, 0.6))
    cap = tex('ShaderNodeTexNoise', 14.0)
    cap.inputs['Detail'].default_value = 3.0
    k = ramp_(cap.outputs['Fac'], 0.3, 0.7, 0.88, 1.1, smooth=False)
    sc1 = node('ShaderNodeVectorMath', operation='SCALE')
    L(c1, sc1.inputs[0])
    L(k, sc1.inputs['Scale'])
    layer1 = sc1.outputs['Vector']

    # ---- layer 2: the veins. Cell edges of a Voronoi pattern, warped by noise so they wander and fork like a
    # network, at two sizes (the finer branches thinner); each vein is blue, purple or green by where it runs
    warp = tex('ShaderNodeTexNoise', 0.8)
    warp.inputs['Detail'].default_value = 2.0
    off = node('ShaderNodeVectorMath', operation='SUBTRACT')
    L(warp.outputs['Color'], off.inputs[0])
    off.inputs[1].default_value = (0.5, 0.5, 0.5)
    off2 = node('ShaderNodeVectorMath', operation='SCALE')
    L(off.outputs['Vector'], off2.inputs[0])
    off2.inputs['Scale'].default_value = 1.1
    wv = node('ShaderNodeVectorMath', operation='ADD')
    L(rest.outputs['Vector'], wv.inputs[0])
    L(off2.outputs['Vector'], wv.inputs[1])
    big = ramp_(tex('ShaderNodeTexVoronoi', 0.7, wv.outputs['Vector'], feature='DISTANCE_TO_EDGE').outputs['Distance'],
                0.0, 0.03, 1.0, 0.0)
    fine = ramp_(tex('ShaderNodeTexVoronoi', 1.8, wv.outputs['Vector'], feature='DISTANCE_TO_EDGE').outputs['Distance'],
                 0.0, 0.035, 1.0, 0.0)
    # (a vein is not a closed cell: a second noise gates the edges, so each runs a while, then fades out; the
    # finer branches are gated harder, so they spring off the main ones and end)
    gate = tex('ShaderNodeTexNoise', 1.6)
    gate.inputs['Detail'].default_value = 1.0
    gate2 = tex('ShaderNodeTexNoise', 3.1)
    veins = math_('MAXIMUM', math_('MULTIPLY', big, ramp_(gate.outputs['Fac'], 0.42, 0.58)),
                  math_('MULTIPLY', fine, math_('MULTIPLY', 0.7, ramp_(gate2.outputs['Fac'], 0.55, 0.7))))
    # (some stretches show more than others, as veins lie deeper or nearer the surface)
    depth = tex('ShaderNodeTexNoise', 0.9)
    depth.inputs['Detail'].default_value = 2.0
    veins = math_('MULTIPLY', veins, ramp_(depth.outputs['Fac'], 0.3, 0.6, 0.35, 1.0))
    hue = tex('ShaderNodeTexNoise', 1.3)
    hue.inputs['Detail'].default_value = 1.0
    vc = mix(VEIN_BLUE, VEIN_PURPLE, ramp_(hue.outputs['Fac'], 0.4, 0.6))
    hue2 = tex('ShaderNodeTexNoise', 0.7)
    vc = mix(vc, VEIN_GREEN, ramp_(hue2.outputs['Fac'], 0.58, 0.75, 0.0, 0.8))
    l2 = node('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY')
    L(math_('MULTIPLY', veins, 0.7), l2.inputs['Factor'])
    L(layer1, l2.inputs['A'])
    L(vc, l2.inputs['B'])
    layer2 = l2.outputs['Result']

    # ---- layer 3: the epidermis. The skin tone, freckles (small soft flecks, each its own size and strength),
    # a few larger blemishes, and pores a shade darker
    if isinstance(tone, (tuple, list)):
        ep = node('ShaderNodeRGB')
        ep.outputs[0].default_value = (*_lin(tone), 1.0)
        ep = ep.outputs[0]
    else:
        ep = tone       # (a colour output: the figure's baked 'skin' tone, with its blush and shade)
    fv = tex('ShaderNodeTexVoronoi', 5.5, feature='F1')
    fv.inputs['Randomness'].default_value = 1.0
    rnd = node('ShaderNodeSeparateColor')
    L(fv.outputs['Color'], rnd.inputs['Color'])
    size = math_('MULTIPLY_ADD', rnd.outputs['Green'], 0.1, 0.07)
    fleck = ramp_(math_('DIVIDE', fv.outputs['Distance'], size), 0.55, 1.0, 1.0, 0.0)
    some = math_('MULTIPLY', ramp_(rnd.outputs['Red'], 0.3, 0.4), math_('MULTIPLY_ADD', rnd.outputs['Blue'], 0.6, 0.4))
    freckle = math_('MULTIPLY', math_('MULTIPLY', fleck, some), freckly)
    bv = tex('ShaderNodeTexVoronoi', 1.3, feature='F1')
    bv.inputs['Randomness'].default_value = 1.0
    brnd = node('ShaderNodeSeparateColor')
    L(bv.outputs['Color'], brnd.inputs['Color'])
    blem = math_('MULTIPLY', ramp_(math_('DIVIDE', bv.outputs['Distance'], 0.16), 0.4, 1.0, 1.0, 0.0),
                 ramp_(brnd.outputs['Red'], 0.82, 0.9))
    pv = tex('ShaderNodeTexVoronoi', 13.0, feature='F1')
    pit = math_('MULTIPLY', ramp_(pv.outputs['Distance'], 0.05, 0.3, 1.0, 0.0), pored)
    c3 = mix(ep, FRECKLE, math_('MULTIPLY', freckle, 0.85))
    c3 = mix(c3, BLEMISH, math_('MULTIPLY', blem, 0.55))
    sc3 = node('ShaderNodeVectorMath', operation='SCALE')
    L(c3, sc3.inputs[0])
    L(math_('SUBTRACT', 1.0, math_('MULTIPLY', pit, 0.12)), sc3.inputs['Scale'])
    layer3 = sc3.outputs['Vector']

    if show in ('1', '2', '3'):
        d = node('ShaderNodeBsdfDiffuse')
        L({'1': layer1, '2': layer2, '3': layer3}[show], d.inputs['Color'])
        L(d.outputs['BSDF'], out.inputs['Surface'])
        return

    # ---- the three together: the epidermis laid over layers 1 and 2, mostly clear, but opaque where it is
    # pigmented (freckles, blemishes)
    opacity = math_('ADD', EPIDERMIS_OPACITY, math_('MULTIPLY', math_('MAXIMUM', math_('MULTIPLY', freckle, 1.0),
                                                                     math_('MULTIPLY', blem, 0.7)),
                                                     1.0 - EPIDERMIS_OPACITY))
    combined = mix(layer2, layer3, opacity)
    # (and the veins show through it, faintly, as they do under real skin)
    vt = node('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY')
    L(math_('MULTIPLY', veins, 0.45), vt.inputs['Factor'])
    L(combined, vt.inputs['A'])
    L(vc, vt.inputs['B'])
    combined = vt.outputs['Result']
    sb = node('ShaderNodeBsdfPrincipled', name='close bsdf')
    L(combined, sb.inputs['Base Color'])
    sb.subsurface_method = 'RANDOM_WALK_SKIN'
    sb.inputs['Subsurface Weight'].default_value = 1.0
    sb.inputs['Subsurface Radius'].default_value = (1.0, 0.36, 0.18)
    sb.inputs['Subsurface Scale'].default_value = 0.12
    sb.inputs['Subsurface Anisotropy'].default_value = 0.8
    sb.inputs['IOR'].default_value = 1.4
    sb.inputs['Specular IOR Level'].default_value = 0.45
    sb.inputs['Roughness'].default_value = 0.5
    sb.inputs['Coat Weight'].default_value = 0.1
    sb.inputs['Coat Roughness'].default_value = 0.3
    bump = node('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.2
    bump.inputs['Distance'].default_value = 0.03
    L(math_('MULTIPLY', pit, -1.0), bump.inputs['Height'])
    L(bump.outputs['Normal'], sb.inputs['Normal'])
    L(sb.outputs['BSDF'], out.inputs['Surface'])
