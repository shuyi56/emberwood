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
# the surface of the combined skin. Set for a whole body as much as a face: little oil, an uneven roughness, a soft
# down, and light that travels far enough under the skin to warm the edge of a shadow on a limb
SPECULAR = 0.4
ROUGH = (0.52, 0.7)             # the roughness, from its smoothest patches to its roughest
COAT = (0.03, 0.45)             # the oil film: weight, roughness
SHEEN = (0.2, 0.45)             # the down: weight, roughness
SSS_SCALE = 0.2                 # how far light travels under the skin (red; 1 unit is about 2.7 cm)
GRAIN = 0.0                     # a fine grain on the surface, beside the pores (off: it read as render noise)
# the pores (sizes in cells of the sowing: at PORE_SCALE 27 a cell is about 1 mm)
PORE_SCALE = 27.0
PORE_SIZE = (0.14, 0.34)        # a pore's radius on fine skin, and where the glands are large
PORES = 1.0                     # how deep the pits are (0 none)
FURROWS = 0.3                   # how deep the net of fine lines between them is, beside a pit
PORE_SHADE = 0.3                # how much darker the bottom of a pit is than the skin round it
PORE_HOLD = 0.8                 # how far a pit is kept out of the scattering (0: blurred away with the skin)
PORE_BUMP = (0.8, 0.012)        # the relief: strength, and its height in units (0.012 is a third of a millimetre)
TONE_RED = [226, 150, 136]      # where the tone drifts redder, and yellower
TONE_YELLOW = [232, 200, 160]
TONE_DRIFT = 0.3
# the melanin in the epidermis: how much of each (0 none, 1 as built; freckles and sun spots follow the red of
# 'skin tex', the freckle map; moles fall anywhere)
FRECKLES = 1.0
SUNSPOTS = 1.0
MOLES = 1.0
PIGMENT_HOLD = 0.9             # how far a pigmented mark is kept out of the scattering (0: blurred with the skin)
SUNSPOT = [186, 138, 104]
MOLE = [88, 54, 40]
FRECKLE = [150, 96, 66]
BLEMISH = [168, 98, 84]


# ---- the lips, in two layers (John 2026-11-20: "a smoother pink under layer and a more rough layer on top"). The same
# numbers as the Face Creator's LIP_LAYERS (tools/face-creator/apply-lip-layers.py), which has the research.
#   under layer: the vascular tissue under a thin epithelium: the painted lip colour, smooth (LIP_BASE_ROUGH, dry to
#     wet), a weak specular (LIP_BASE_SPEC of the skin's: the interface lies under the epithelium), only LIP_BASE_FURROW
#     of the grooves in its relief, and a short, red scatter (LIP_SSS_RADIUS, LIP_SSS_SCALE: shallower than the skin's,
#     the blood near the surface takes green first)
#   top layer: the cornified layer as Principled's coat: its weight and roughness from the lips' wetness (LIP_COAT,
#     LIP_COAT_ROUGH: wet at the parting, dry at the border), its own normal carrying the grooves and the dry flakes
# Measured from the head: HEAD_H units of about 27 mm, so the grooves are LIP_PITCH_MM apart at the parting.
LIP_PITCH_MM = 1.1
LIP_FLAKE_MM = 0.45
LIP_BASE_FURROW = 0.35
LIP_BASE_ROUGH = (0.36, 0.24)
LIP_BASE_SPEC = 0.35
LIP_COAT = (0.6, 0.9)
LIP_COAT_ROUGH = (0.46, 0.22)
LIP_COAT_BUMP = 0.004           # the grooves' depth in units (0.1 mm)
LIP_SSS_RADIUS = (1.0, 0.3, 0.22)
LIP_SSS_SCALE = 0.04            # (about 1 mm of red reach: the skin's SSS_SCALE is 0.2)
LIP_SSS_WEIGHT = 0.6
MM_PER_UNIT = 27.0


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
    size = math_('MULTIPLY_ADD', rnd.outputs['Green'], 0.25, 0.2)      # (of a cell: 1 to 2.2 mm)
    fleck = ramp_(math_('DIVIDE', fv.outputs['Distance'], size), 0.55, 1.0, 1.0, 0.0)
    some = math_('MULTIPLY', ramp_(rnd.outputs['Red'], 0.3, 0.4), math_('MULTIPLY_ADD', rnd.outputs['Blue'], 0.6, 0.4))
    freckle = math_('MULTIPLY', math_('MULTIPLY', fleck, some), freckly)
    bv = tex('ShaderNodeTexVoronoi', 1.3, feature='F1')
    bv.inputs['Randomness'].default_value = 1.0
    brnd = node('ShaderNodeSeparateColor')
    L(bv.outputs['Color'], brnd.inputs['Color'])
    blem = math_('MULTIPLY', ramp_(math_('DIVIDE', bv.outputs['Distance'], 0.16), 0.4, 1.0, 1.0, 0.0),
                 ramp_(brnd.outputs['Red'], 0.82, 0.9))
    # ---- the pores. A pore is a small pit at the mouth of a hair follicle or an oil gland, and three things about
    # them make skin read as skin: they are not all one size (each its own, and larger where the glands are large:
    # the nose, the cheeks beside it, the forehead and the chin; the blue of 'skin tex' says how much), they are
    # sown at about one a millimetre and never on a grid, and they are joined by a net of very fine furrows (the
    # skin's primary lines), so the surface between them is a field of tiny raised plates, not a smooth sheet.
    # (the mouth of each pore is bent a little out of round, like the freckles)
    pbent = warped(30.0, 0.012)
    pv = tex('ShaderNodeTexVoronoi', PORE_SCALE, pbent, feature='F1')
    pv.inputs['Randomness'].default_value = 1.0
    prnd = node('ShaderNodeSeparateColor')
    L(pv.outputs['Color'], prnd.inputs['Color'])
    # (the pore's radius, as a part of its cell: from PORE_SIZE[0] on smooth skin to PORE_SIZE[1] on the nose, each
    # pore somewhere between half that and all of it; a soft-sided pit, deepest at the middle)
    prad = math_('MULTIPLY', math_('MULTIPLY_ADD', pored, PORE_SIZE[1] - PORE_SIZE[0], PORE_SIZE[0]),
                 math_('MULTIPLY_ADD', prnd.outputs['Green'], 0.5, 0.5))
    pcup = ramp_(math_('DIVIDE', pv.outputs['Distance'], prad), 0.0, 1.0, 1.0, 0.0)
    # (not every follicle shows: fewer where the skin is fine)
    pshow = ramp_(math_('ADD', prnd.outputs['Red'], math_('MULTIPLY', pored, 0.6)), 0.45, 0.6)
    pit = math_('MULTIPLY', math_('MULTIPLY', pcup, pshow), math_('MULTIPLY_ADD', pored, 0.75, 0.25))
    # (the furrows between them: the edges of the same sowing's cells and of a coarser one, very shallow)
    fur = math_('MAXIMUM',
                ramp_(tex('ShaderNodeTexVoronoi', PORE_SCALE, pbent, feature='DISTANCE_TO_EDGE').outputs['Distance'],
                      0.0, 0.07, 1.0, 0.0),
                ramp_(tex('ShaderNodeTexVoronoi', PORE_SCALE * 0.45, pbent, feature='DISTANCE_TO_EDGE').outputs['Distance'],
                      0.0, 0.05, 1.0, 0.0))
    relief = math_('ADD', math_('MULTIPLY', pit, PORES), math_('MULTIPLY', fur, FURROWS))
    # (the rest of the melanin. Freckles are never one size or evenly sown: a second, finer sowing between the first,
    # both thinned and thickened in drifts by a slow noise, and every fleck's outline bent a little out of round)
    bent = warped(9.0, 0.06)
    fv2 = tex('ShaderNodeTexVoronoi', 11.0, bent, feature='F1')
    fv2.inputs['Randomness'].default_value = 1.0
    rnd2 = node('ShaderNodeSeparateColor')
    L(fv2.outputs['Color'], rnd2.inputs['Color'])
    fine = math_('MULTIPLY', ramp_(math_('DIVIDE', fv2.outputs['Distance'],
                                         math_('MULTIPLY_ADD', rnd2.outputs['Green'], 0.15, 0.15)), 0.5, 1.0, 1.0, 0.0),
                 math_('MULTIPLY', ramp_(rnd2.outputs['Red'], 0.45, 0.55), math_('MULTIPLY_ADD', rnd2.outputs['Blue'], 0.5, 0.25)))
    sown = tex('ShaderNodeTexNoise', 1.4)
    sown.inputs['Detail'].default_value = 2.0
    drifts = ramp_(sown.outputs['Fac'], 0.3, 0.7, 0.35, 1.0)
    freckle = math_('MULTIPLY', math_('MAXIMUM', freckle, math_('MULTIPLY', fine, freckly)), math_('MULTIPLY', drifts, FRECKLES))
    # (sun spots: fewer, larger, paler and soft edged, where the freckles are and a little past them)
    sv = tex('ShaderNodeTexVoronoi', 2.3, bent, feature='F1')
    sv.inputs['Randomness'].default_value = 1.0
    srnd = node('ShaderNodeSeparateColor')
    L(sv.outputs['Color'], srnd.inputs['Color'])
    spot = math_('MULTIPLY', ramp_(math_('DIVIDE', sv.outputs['Distance'],
                                         math_('MULTIPLY_ADD', srnd.outputs['Green'], 0.2, 0.25)), 0.3, 1.0, 1.0, 0.0),
                 math_('MULTIPLY', ramp_(srnd.outputs['Red'], 0.7, 0.8),
                       math_('MULTIPLY', math_('MULTIPLY_ADD', freckly, 0.8, 0.2), SUNSPOTS)))
    # (moles: a handful over the whole body, anywhere, small, dark, sharp edged, each its own size)
    mv = tex('ShaderNodeTexVoronoi', 0.55, feature='F1')
    mv.inputs['Randomness'].default_value = 1.0
    mrnd = node('ShaderNodeSeparateColor')
    L(mv.outputs['Color'], mrnd.inputs['Color'])
    mole = math_('MULTIPLY', ramp_(math_('DIVIDE', mv.outputs['Distance'],
                                         math_('MULTIPLY_ADD', mrnd.outputs['Green'], 0.035, 0.03)), 0.75, 1.0, 1.0, 0.0),
                 math_('MULTIPLY', ramp_(mrnd.outputs['Red'], 0.5, 0.55), MOLES))
    c3 = mix(ep, FRECKLE, math_('MULTIPLY', freckle, 0.85))
    c3 = mix(c3, SUNSPOT, math_('MULTIPLY', spot, 0.5))
    c3 = mix(c3, MOLE, math_('MULTIPLY', mole, 0.92))
    c3 = mix(c3, BLEMISH, math_('MULTIPLY', blem, 0.55))
    sc3 = node('ShaderNodeVectorMath', operation='SCALE')
    L(c3, sc3.inputs[0])
    # (a pore is a little darker than the skin round it: a pit holds its own shadow)
    L(math_('SUBTRACT', 1.0, math_('MULTIPLY', pit, PORE_SHADE * PORES)), sc3.inputs['Scale'])
    layer3 = sc3.outputs['Vector']

    if show in ('1', '2', '3'):
        d = node('ShaderNodeBsdfDiffuse')
        L({'1': layer1, '2': layer2, '3': layer3}[show], d.inputs['Color'])
        L(d.outputs['BSDF'], out.inputs['Surface'])
        return

    # ---- the three together: the epidermis laid over layers 1 and 2, mostly clear, but opaque where it is
    # pigmented (freckles, blemishes)
    opacity = math_('ADD', EPIDERMIS_OPACITY, math_('MULTIPLY', math_('MAXIMUM', math_('MAXIMUM', math_('MULTIPLY', freckle, 1.0), math_('MAXIMUM', math_('MULTIPLY', spot, 0.6), mole)),
                                                                     math_('MULTIPLY', blem, 0.7)),
                                                     1.0 - EPIDERMIS_OPACITY))
    combined = mix(layer1, layer3, opacity)
    # (and the veins show through it, faintly, as they do under real skin)
    vt = node('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY')
    L(math_('MULTIPLY', veins, VEIN_SEEN), vt.inputs['Factor'])
    L(combined, vt.inputs['A'])
    L(vs, vt.inputs['B'])
    combined = vt.outputs['Result']
    # (the tone is never one flat colour: slow drifts a little redder and a little yellower, painted into the
    # colour itself, so they are still there in shade, where the light does none of the work)
    drift = tex('ShaderNodeTexNoise', 0.35)
    drift.inputs['Detail'].default_value = 3.0
    drift.inputs['Roughness'].default_value = 0.6
    dn = node('ShaderNodeSeparateColor')
    L(drift.outputs['Color'], dn.inputs['Color'])
    combined = mix(combined, TONE_RED, ramp_(dn.outputs['Red'], 0.45, 0.75, 0.0, TONE_DRIFT))
    combined = mix(combined, TONE_YELLOW, ramp_(dn.outputs['Green'], 0.5, 0.8, 0.0, TONE_DRIFT * 0.7))
    sb = node('ShaderNodeBsdfPrincipled', name='close bsdf')
    L(combined, sb.inputs['Base Color'])
    sb.subsurface_method = 'RANDOM_WALK_SKIN'
    # (melanin lies in the top of the skin, above where the light spreads: a freckle is a sharp little shadow on the
    # surface. Left to the scattering, every mark smaller than the scatter's reach is blurred away, so where there
    # is pigment the surface scatters less and keeps the mark's own colour and edge)
    pigment = math_('MAXIMUM', freckle, math_('MAXIMUM', math_('MULTIPLY', spot, 0.5), mole))
    # (the pores the same: under the scatter a pit a third of a millimetre across is blurred to nothing, shadow,
    # relief and all, which is why skin with pores built into it can still render as a smooth sheet. In the pit
    # the surface is lit as a surface, so the pit's own shade and its relief show)
    held = math_('MAXIMUM', math_('MULTIPLY', pigment, PIGMENT_HOLD), math_('MULTIPLY', relief, PORE_HOLD))
    L(math_('SUBTRACT', 1.0, math_('MINIMUM', held, 1.0)), sb.inputs['Subsurface Weight'])
    sb.inputs['Subsurface Radius'].default_value = (1.0, 0.36, 0.18)
    sb.inputs['Subsurface Scale'].default_value = SSS_SCALE
    sb.inputs['Subsurface Anisotropy'].default_value = 0.8
    sb.inputs['IOR'].default_value = 1.4
    sb.inputs['Specular IOR Level'].default_value = SPECULAR
    # (the surface is rougher in some patches than others, and rougher in the pores: an even gloss is what reads
    # as plastic, most of all in shade, where the reflection is all there is to see)
    rn = tex('ShaderNodeTexNoise', 2.2)
    rn.inputs['Detail'].default_value = 4.0
    rough = math_('ADD', ramp_(rn.outputs['Fac'], 0.3, 0.7, ROUGH[0], ROUGH[1], smooth=False), math_('MULTIPLY', pit, 0.18))
    L(rough, sb.inputs['Roughness'])
    sb.inputs['Coat Weight'].default_value = COAT[0]
    sb.inputs['Coat Roughness'].default_value = COAT[1]
    # (the fine down on skin: a soft light along every edge that turns from the eye, in place of a mirror's)
    sb.inputs['Sheen Weight'].default_value = SHEEN[0]
    sb.inputs['Sheen Roughness'].default_value = SHEEN[1]
    grain = tex('ShaderNodeTexNoise', 60.0)
    grain.inputs['Detail'].default_value = 2.0
    bump = node('ShaderNodeBump')
    bump.inputs['Strength'].default_value = PORE_BUMP[0]
    bump.inputs['Distance'].default_value = PORE_BUMP[1]
    L(math_('ADD', math_('MULTIPLY', relief, -1.0), math_('MULTIPLY', grain.outputs['Fac'], GRAIN)), bump.inputs['Height'])
    L(bump.outputs['Normal'], sb.inputs['Normal'])
    L(sb.outputs['BSDF'], out.inputs['Surface'])


# ---------------------------------------------------------------- a whole figure in the layered skin
# The colour of the feet (linear): pinker and a little deeper than the leg's tone, most at the toes, the heel and
# along the sole. A body baked in one tone from neck to toe has feet that read white beside the shins.
FOOT_TONE = (0.80, 0.47, 0.42)


def _ss(lo, hi, v):
    t = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _layer(me, name):
    a = me.color_attributes[name]
    c = np.zeros(len(me.vertices) * 4)
    a.data.foreach_get('color', c)
    return a, c.reshape(-1, 4)


def paint_body(body, hands=(), neck=None, wrists=(), floor=None):
    """Paint what the layered skin reads from a bare figure's point colours, on meshes that already carry the base
    figure's 'skin' and 'skin tex' layers. Colours only: no point is moved.

    body    the skin mesh, in its rest shape (feet straight ahead, toes toward -Y), in world units
    hands   the hand meshes (thin skin all over: the veins on the back of the hand)
    neck    the neck mesh, if it is its own (a little thin at the sides)
    wrists  the wrist joints' positions in the body's rest space, for the thin skin of the inner wrist
    floor   the height of the soles (the body's lowest point if not given)

    'skin' gets the feet's tint; the alpha of 'skin tex' gets how thin the skin is (0.04 over the body, more at the
    wrists, on the top of the foot and across the upper chest), which is where the veins show. Call it after the
    body is built and before the first render; calling it twice tints the feet twice."""
    me = body.data
    n = len(me.vertices)
    R = np.zeros(n * 3)
    me.vertices.foreach_get('co', R)
    mw = np.array(body.matrix_world)
    P = R.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    h = z - (z.min() if floor is None else floor)
    fm = h < 3.2
    y0, y1 = y[fm].min(), y[fm].max()
    # the feet: tinted all over, more by the floor, at the toes and at the heel, fading out above the ankle
    a, c = _layer(me, 'skin')
    k = (1 - _ss(2.5, 5.0, h)) * np.clip(0.22 + 0.38 * np.maximum(
        1 - _ss(0.3, 1.4, h), np.maximum(1 - _ss(y0 + 0.8, y0 + 3.2, y), 0.8 * _ss(y1 - 2.2, y1 - 0.5, y))), 0, 1)
    c[:, :3] = c[:, :3] * (1 - k[:, None]) + np.array(FOOT_TONE) * k[:, None]
    a.data.foreach_set('color', c.ravel())
    # thin skin
    t, ct = _layer(me, 'skin tex')
    thin = np.full(n, 0.04)
    for w in wrists:
        thin = np.maximum(thin, 0.5 * np.clip(1 - np.linalg.norm(P - np.array(w), axis=1) / 4.0, 0, 1))
    thin = np.maximum(thin, 0.45 * _ss(0.9, 2.1, h) * (1 - _ss(y1 - 3.5, y1 - 1.5, y)) * fm)      # top of the foot
    chest = np.clip(1 - np.abs(z - (z.max() - 9.0)) / 2.5, 0, 1) * (np.abs(x) < 4.0) * (y < 0)
    thin = np.maximum(thin, 0.25 * chest)
    ct[:, 3] = thin
    t.data.foreach_set('color', ct.ravel())
    me.update()
    for o, v in [(o, 0.6) for o in hands] + ([(neck, 0.22)] if neck is not None else []):
        t, ct = _layer(o.data, 'skin tex')
        ct[:, 3] = v
        t.data.foreach_set('color', ct.ravel())
        o.data.update()


def paint_face(head, body=None, neck=None):
    """Paint where the freckles and the pores sit on a bare figure's head, in the red and blue of its 'skin tex'
    layer, for meshes that already carry it. Colours only: no point is moved. It sets values, so calling it twice is
    harmless.

    head  the head mesh, in its rest shape (face toward -Y), in world units
    body  the skin mesh, for the freckles across the shoulders, the upper chest and the upper back
    neck  the neck mesh, if it is its own (a few freckles)

    Red is the freckle map: a band over the nose and the cheeks, a little on the forehead. Blue is the pore map: the
    most on the nose, then the cheeks, the forehead and the chin, and a little everywhere. The head's thin skin is
    held to 0.3 so the veins stay faint on the face."""
    def world(o):
        me = o.data
        R = np.zeros(len(me.vertices) * 3)
        me.vertices.foreach_get('co', R)
        mw = np.array(o.matrix_world)
        return R.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]

    def g(v, c, w):
        return np.exp(-((v - c) / w) ** 2)

    P = world(head)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    u = (z - z.min()) / (z.max() - z.min())
    front = np.clip((np.median(y) - y) / 1.5, 0, 1)
    t, c = _layer(head.data, 'skin tex')
    c[:, 0] = front * np.maximum(g(u, 0.42, 0.13) * np.clip(1.15 - np.abs(x) / 3.4, 0, 1), 0.35 * g(u, 0.72, 0.1))
    c[:, 2] = np.clip(0.15 + front * np.maximum.reduce([
        g(x, 0, 0.9) * g(u, 0.43, 0.1),                     # the nose
        0.7 * g(np.abs(x), 1.7, 1.0) * g(u, 0.4, 0.1),      # the cheeks
        0.55 * g(x, 0, 2.2) * g(u, 0.68, 0.1),              # the forehead
        0.5 * g(x, 0, 1.2) * g(u, 0.12, 0.08)]), 0, 1)      # the chin
    c[:, 3] = np.minimum(c[:, 3], 0.3)
    t.data.foreach_set('color', c.ravel())
    head.data.update()
    if neck is not None:
        t, c = _layer(neck.data, 'skin tex')
        c[:, 0] = 0.1
        t.data.foreach_set('color', c.ravel())
        neck.data.update()
    if body is not None:
        z = world(body)[:, 2]
        t, c = _layer(body.data, 'skin tex')
        c[:, 0] = 0.45 * np.clip((z - (z.max() - 7.0)) / 3.0, 0, 1)
        t.data.foreach_set('color', c.ravel())
        body.data.update()


def apply(material, show='all'):
    """Make a figure's skin material the layered skin, its tone read from the baked 'skin' point colour (so the
    blush, the shade and paint_body's feet stay). Every mesh in the material takes it; nothing else changes."""
    import bpy  # noqa: F401
    nt = material.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    tone = nt.nodes.new('ShaderNodeVertexColor')
    tone.layer_name = 'skin'
    skin_layers(nt, out, show, tone.outputs['Color'])


def lip_material(nt, out, colour, layers):
    """The lips' two layers on one Principled BSDF. colour: the lip painting's image node (colour and alpha); layers:
    the lip layers image node (red: how wet, green: the grooves, blue: the dry flakes), both on the same patch."""
    L = nt.links.new
    p = nt.nodes.new('ShaderNodeBsdfPrincipled')
    sep = nt.nodes.new('ShaderNodeSeparateColor')
    L(layers.outputs['Color'], sep.inputs['Color'])
    wet, groove, flake = sep.outputs[0], sep.outputs[1], sep.outputs[2]

    def ramp(v, a, b):
        n = nt.nodes.new('ShaderNodeMapRange')
        L(v, n.inputs['Value'])
        n.inputs['To Min'].default_value, n.inputs['To Max'].default_value = a, b
        return n.outputs['Result']

    def add(a, b, k):
        n = nt.nodes.new('ShaderNodeMath')
        n.operation = 'MULTIPLY_ADD'
        L(b, n.inputs[0])
        n.inputs[1].default_value = k
        L(a, n.inputs[2])
        n.use_clamp = True
        return n.outputs[0]
    # the under layer
    L(colour.outputs['Color'], p.inputs['Base Color'])
    L(colour.outputs['Alpha'], p.inputs['Alpha'])
    L(ramp(wet, *LIP_BASE_ROUGH), p.inputs['Roughness'])
    p.inputs['Specular IOR Level'].default_value = SPECULAR * LIP_BASE_SPEC      # (the skin's, cut: the under layer's interface lies under the epithelium)
    p.subsurface_method = 'BURLEY'      # (a disc, not a random walk: the lips are a thin shell over the head)
    p.inputs['Subsurface Weight'].default_value = LIP_SSS_WEIGHT
    p.inputs['Subsurface Radius'].default_value = LIP_SSS_RADIUS
    p.inputs['Subsurface Scale'].default_value = LIP_SSS_SCALE
    base_bump = nt.nodes.new('ShaderNodeBump')
    base_bump.invert = True
    base_bump.inputs['Distance'].default_value = LIP_COAT_BUMP * LIP_BASE_FURROW
    L(groove, base_bump.inputs['Height'])
    L(base_bump.outputs['Normal'], p.inputs['Normal'])
    # the top layer
    L(ramp(wet, *LIP_COAT), p.inputs['Coat Weight'])
    L(add(ramp(wet, *LIP_COAT_ROUGH), flake, 0.3), p.inputs['Coat Roughness'])
    p.inputs['Coat IOR'].default_value = 1.45
    hmix = nt.nodes.new('ShaderNodeMath')
    hmix.operation = 'MULTIPLY_ADD'
    L(flake, hmix.inputs[0])
    hmix.inputs[1].default_value = 0.4
    L(groove, hmix.inputs[2])
    coat_bump = nt.nodes.new('ShaderNodeBump')
    coat_bump.invert = True
    coat_bump.inputs['Distance'].default_value = LIP_COAT_BUMP
    L(hmix.outputs[0], coat_bump.inputs['Height'])
    L(coat_bump.outputs['Normal'], p.inputs['Coat Normal'])
    L(p.outputs['BSDF'], out.inputs['Surface'])
    return p
