"""Skin built the way a painter builds it, in three layers, each one shown on its own and then all together.

    1  the base: a deep, mottled red-and-brown flesh (the muscle and the capillary bed under the skin)
    2  the vascular overlay: a fine branching network of blue, purple and green veins over layer 1
    3  the epidermis: the skin's own tone, with freckles, pores and a few blemishes, painted mostly transparent

skin_layers(nt, out, show, tone) builds one material into a node tree. show is '1', '2' or '3' for a layer by
itself (a flat diffuse surface, so the colour is read as painted), or 'all' for the combined skin: layer 3
laid over layers 1 and 2 at EPIDERMIS_OPACITY (more where the pigment is: the freckles), on a Principled surface
with subsurface scattering, so what lies under the epidermis shows through it and the veins stay soft.

Like the figure's own skin it reads its patterns from the 'rest' point attribute, and the freckle and pore
masks from the 'skin tex' point colour (red: freckles, blue: pores, alpha: how thin the skin is, so how much of
the veins shows: the temples, the bridge of the nose, under the eyes, the back of the hand); see
female.skin_tones. Nothing here builds a
figure. skin_study.py --layers renders the four side by side."""
import numpy as np

# how much of the colour of the epidermis lies over what is under it (0 clear, 1 opaque)
EPIDERMIS_OPACITY = 0.7
# the layers' colours (sRGB)
MUSCLE_RED = [162, 56, 50]
MUSCLE_BROWN = [150, 82, 64]
# (the veins' colours are what they let through, multiplied into what lies under them: blood in a vessel absorbs
# the light that crosses it, as the thin absorbing layer between the epidermis and the dermis does in Donner and
# Jensen's layered model, rather than adding paint on top. Lighter than the colour they leave. These three are
# the veins laid bare on layer 1.)
VEIN_BLUE = [120, 150, 240]
VEIN_PURPLE = [190, 130, 215]
VEIN_GREEN = [140, 215, 160]
# (and these are the same veins seen through the epidermis. A vein under skin is not blue paint: measured, it
# sends back about as much blue as the skin beside it and clearly less red (Kienle and others, 1996), so it is
# only a little darker and a little cooler than the skin, and the eye calls that blue. So: take out red, a
# little green, almost no blue. Greenish where the skin over it is yellower, purplish where it lies shallow.)
VEIN_SEEN_BLUE = [205, 234, 250]
VEIN_SEEN_PURPLE = [222, 222, 248]
VEIN_SEEN_GREEN = [212, 242, 236]
VEIN_SEEN = 0.9                 # how much of that the thinnest skin shows
# the network (1 unit is about 2.7 cm, so a trunk is about 3 mm across with its blur, a branch about 1.5 mm)
VEIN_BEND = (0.35, 1.6)         # the slow bend of the space the veins run in: noise scale, how far it moves them
VEIN_WIGGLE = (1.7, 0.22)       # the small wiggle on top
VEIN_TRUNK_SCALE = 0.5          # trunks: one cell is about 2 units, so a trunk every 5 cm
VEIN_TRUNK_WIDTH = 0.085
VEIN_BRANCH_SCALE = 1.5
VEIN_BRANCH_WIDTH = 0.05
VEIN_BRANCH_REACH = 0.55        # how far from its trunk a branch runs before it fades out
VEIN_BRANCH_STRENGTH = 0.6
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
    thin_skin = node('ShaderNodeVertexColor', layer_name='skin tex').outputs['Alpha']

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

    # ---- layer 2: the veins. A hierarchy, as veins are: a few wide trunks that wander, and thin side branches
    # that spring off the trunks and die out a little way from them. Both are cell edges of a Voronoi pattern
    # read through a twice-warped space (a slow bend and a small wiggle), so no edge runs straight and no cell
    # reads as a polygon; a noise gates the trunks so each runs a while and sinks out of sight
    def warped(scale, amount, vec=None):
        w = tex('ShaderNodeTexNoise', scale, vec)
        w.inputs['Detail'].default_value = 2.0
        o = node('ShaderNodeVectorMath', operation='SUBTRACT')
        L(w.outputs['Color'], o.inputs[0])
        o.inputs[1].default_value = (0.5, 0.5, 0.5)
        o2 = node('ShaderNodeVectorMath', operation='SCALE')
        L(o.outputs['Vector'], o2.inputs[0])
        o2.inputs['Scale'].default_value = amount
        a = node('ShaderNodeVectorMath', operation='ADD')
        L(vec if vec is not None else rest.outputs['Vector'], a.inputs[0])
        L(o2.outputs['Vector'], a.inputs[1])
        return a.outputs['Vector']

    wv = warped(VEIN_WIGGLE[0], VEIN_WIGGLE[1], warped(VEIN_BEND[0], VEIN_BEND[1]))
    td = tex('ShaderNodeTexVoronoi', VEIN_TRUNK_SCALE, wv, feature='DISTANCE_TO_EDGE').outputs['Distance']
    bd = tex('ShaderNodeTexVoronoi', VEIN_BRANCH_SCALE, wv, feature='DISTANCE_TO_EDGE').outputs['Distance']
    # (seen through skin a vein has no edge: a soft core with a wide blurred fall-off, so a squared smooth ramp)
    trunk = ramp_(td, 0.0, VEIN_TRUNK_WIDTH, 1.0, 0.0)
    trunk = math_('MULTIPLY', trunk, trunk)
    branch = ramp_(bd, 0.0, VEIN_BRANCH_WIDTH, 1.0, 0.0)
    branch = math_('MULTIPLY', branch, branch)
    gate = tex('ShaderNodeTexNoise', 1.1)
    gate.inputs['Detail'].default_value = 1.0
    trunk = math_('MULTIPLY', trunk, ramp_(gate.outputs['Fac'], 0.4, 0.6))
    # (a branch shows only near a live trunk, weaker the further it runs from it)
    near = math_('MULTIPLY', ramp_(td, 0.0, VEIN_BRANCH_REACH, 1.0, 0.0), ramp_(gate.outputs['Fac'], 0.4, 0.6))
    gate2 = tex('ShaderNodeTexNoise', 2.7)
    branch = math_('MULTIPLY', math_('MULTIPLY', branch, near),
                   math_('MULTIPLY', VEIN_BRANCH_STRENGTH, ramp_(gate2.outputs['Fac'], 0.45, 0.6)))
    veins = math_('MAXIMUM', trunk, branch)
    # (some stretches show more than others, as veins lie deeper or nearer the surface)
    depth = tex('ShaderNodeTexNoise', 0.9)
    depth.inputs['Detail'].default_value = 2.0
    veins = math_('MULTIPLY', veins, ramp_(depth.outputs['Fac'], 0.3, 0.6, 0.45, 1.0))
    # (the veins are in the skin everywhere, but show only where it is thin: layer 2 on its own shows them all)
    shown = ramp_(thin_skin, 0.15, 0.6)
    hue = tex('ShaderNodeTexNoise', 0.6)
    hue.inputs['Detail'].default_value = 1.0
    hue2 = tex('ShaderNodeTexNoise', 0.45)
    vc = mix(VEIN_BLUE, VEIN_PURPLE, ramp_(hue.outputs['Fac'], 0.45, 0.65))
    vc = mix(vc, VEIN_GREEN, ramp_(hue2.outputs['Fac'], 0.55, 0.75, 0.0, 0.8))
    # (what the same veins take out of the light once the epidermis lies over them: far less, and mostly red)
    vs = mix(VEIN_SEEN_BLUE, VEIN_SEEN_PURPLE, ramp_(hue.outputs['Fac'], 0.45, 0.65))
    vs = mix(vs, VEIN_SEEN_GREEN, ramp_(hue2.outputs['Fac'], 0.55, 0.75, 0.0, 0.8))
    l2 = node('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY')
    L(math_('MULTIPLY', veins, 0.8), l2.inputs['Factor'])
    L(layer1, l2.inputs['A'])
    L(vc, l2.inputs['B'])
    layer2 = l2.outputs['Result']
    veins = math_('MULTIPLY', veins, shown)

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
    combined = mix(layer1, layer3, opacity)
    # (and the veins show through it, faintly, as they do under real skin)
    vt = node('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY')
    L(math_('MULTIPLY', veins, VEIN_SEEN), vt.inputs['Factor'])
    L(combined, vt.inputs['A'])
    L(vs, vt.inputs['B'])
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
