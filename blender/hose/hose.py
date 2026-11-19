"""A standalone garment: period hose on the bare mannequin (the anatomy thread's head-driven body), with nothing worn
over them. The hose are drafted in draft.py from the legs' own measurements; this module supplies the body and the
wool's look.

    python3 blender/base/views.py --figure hose --mannequin --turn --clay --out DIR
    python3 blender/base/views.py --figure hose --mannequin --head --set VIEW=2 --out DIR

VIEW frames the --head close-up on one part of the hose, by its place in FRAMES: 0 'legs' (the default: the waist to
the floor), 1 'hips' (the waist to the crotch), 2 'knee', 3 'ankle'. (A number, because views.py evaluates every --set
value and the queue's commands carry no quotes.) FOOT (draft.FOOT) is 'footed', 'stirrup' or 'open'.

Hose in the 14th and 15th centuries were cut from woven wool on the bias, so the cloth's give lies along the leg and
round it, sewn up the back of the leg, footed or strapped under the foot, and tied to the doublet at the waist by
points through eyelets. Knitted hose came later and to the rich first; this is the woven kind."""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for d in ('vest', 'sorceress_pendant', 'shared', 'base'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
# (the bare mannequin: its own plain head, the gown's pieces and the necklace out of sight)
os.environ.setdefault('HEAD_GLB', '')
os.environ.setdefault('NO_DRESS', '1')
os.environ.setdefault('NO_ACC', '1')
import bpy  # noqa: E402
import female  # noqa: E402
import sorceress_pendant as SP  # noqa: E402
import draft  # noqa: E402

FRAMES = ('legs', 'hips', 'knee', 'ankle')
VIEW = 0
FOOT = draft.FOOT
MASK = True                  # hide the skin the hose covers (it cannot show through a bent knee)
MEASURES = {}
# A madder red, the commonest dye for good hose: dulled and a little brown, as a wool dyed in the piece is.
WOOL = dict(colour=(0.30, 0.075, 0.055), pitch=0.028, wale=4, relief=0.5, rough=0.86, sheen=0.75, mottle=0.10,
            seam_ridge=0.07, eyelets=10)
M_WOOL = female.mat_def(74, 'hose wool', female.ramp(0x7c3328, 0.1), 0.0, surface='twill', closeup='hose')


def wool_closeup(name, P=WOOL):
    """The 3D look of the hose's wool on a material from female.mat (its flat sprite look kept beside it): a fine 2/2
    twill on the bias (the threads at 45 degrees to the leg, so the twill's diagonal runs along it), laid out on the
    cut pattern (the 'pattern' UVs, in model units), with the wool's soft nap and sheen, a slow unevenness of the dye,
    the seams' ridge and stitching, and the eyelets round the waist."""
    m = bpy.data.materials.get(name)
    if m is None:
        return
    nt = m.node_tree
    N = nt.nodes.new
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
    flat = next((n for n in nt.nodes if n.type == 'BSDF_DIFFUSE'), None)
    if flat is not None:
        flat.name = 'sprite bsdf'

    def math_(op, a, b=None):
        k = N('ShaderNodeMath')
        k.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                k.inputs[i].default_value = float(v)
            else:
                nt.links.new(v, k.inputs[i])
        return k.outputs[0]

    def rng(sock, a, b, lo, hi):
        r = N('ShaderNodeMapRange')
        r.inputs['From Min'].default_value, r.inputs['From Max'].default_value = a, b
        r.inputs['To Min'].default_value, r.inputs['To Max'].default_value = lo, hi
        nt.links.new(sock, r.inputs['Value'])
        return r.outputs['Result']

    def attr(nm):
        a = N('ShaderNodeAttribute')
        a.attribute_name = nm
        return a
    uv = N('ShaderNodeUVMap')
    uv.uv_map = 'pattern'
    sep = N('ShaderNodeSeparateXYZ')
    nt.links.new(uv.outputs['UV'], sep.inputs['Vector'])
    u, v = sep.outputs['X'], sep.outputs['Y']
    # the threads on the bias: a runs along one, b along the other, in threads
    k = 1.0 / (math.sqrt(2.0) * P['pitch'])
    a = math_('MULTIPLY', math_('ADD', u, v), k)
    b = math_('MULTIPLY', math_('SUBTRACT', v, u), k)
    ia, ib = math_('FLOOR', a), math_('FLOOR', b)
    fa, fb = math_('FRACT', a), math_('FRACT', b)
    # 2/2 twill: a thread of a lies over two of b, then under two, each a step on from the last
    up = math_('LESS_THAN', math_('FLOORED_MODULO', math_('SUBTRACT', ia, ib), 4.0), 2.0)
    yarn_a = math_('POWER', math_('SINE', math_('MULTIPLY', fb, math.pi)), 0.6)     # (a thread of a, across it)
    yarn_b = math_('POWER', math_('SINE', math_('MULTIPLY', fa, math.pi)), 0.6)
    over = N('ShaderNodeMix')
    over.data_type = 'FLOAT'
    nt.links.new(up, over.inputs['Factor'])
    nt.links.new(math_('MULTIPLY', yarn_b, 0.75), over.inputs['A'])
    nt.links.new(math_('ADD', math_('MULTIPLY', yarn_a, 0.75), 0.25), over.inputs['B'])
    weave = over.outputs['Result']
    # the wool's nap: fine fibres over the weave, fuzzing it, and the dye's slow unevenness
    rest = attr('rest')
    nap = N('ShaderNodeTexNoise')
    nap.inputs['Scale'].default_value, nap.inputs['Detail'].default_value = 26.0, 6.0
    nap.inputs['Roughness'].default_value = 0.7
    nt.links.new(rest.outputs['Vector'], nap.inputs['Vector'])
    dye = N('ShaderNodeTexNoise')
    dye.inputs['Scale'].default_value, dye.inputs['Detail'].default_value = 0.35, 3.0
    nt.links.new(rest.outputs['Vector'], dye.inputs['Vector'])
    # the seams: a soft ridge where the two edges are turned in, and the stitches' groove down its middle
    seam = attr('seam').outputs['Fac']
    sr = P['seam_ridge']
    ridge = math_('EXPONENT', math_('MULTIPLY', math_('POWER', math_('DIVIDE', seam, sr), 2.0), -1.0))
    groove = math_('EXPONENT', math_('MULTIPLY', math_('POWER', math_('DIVIDE', seam, 0.22 * sr), 2.0), -1.0))
    # the top: the turned-down band (its lower edge stitched) and the eyelets for the points, in pairs round it
    top_d = attr('top_d').outputs['Fac']
    ring = attr('ring_s').outputs['Fac']
    band_d = float(MEASURES.get('band', 1.0))
    hem_line = math_('EXPONENT', math_('MULTIPLY', math_('POWER', math_('DIVIDE', math_('SUBTRACT', top_d, band_d), 0.035), 2.0), -1.0))
    step = float(MEASURES.get('eyelet_step', 2.0))
    cell = math_('SUBTRACT', math_('FRACT', math_('DIVIDE', ring, step)), 0.5)          # (-0.5..0.5 round each pair)
    ex = math_('SUBTRACT', math_('ABSOLUTE', math_('MULTIPLY', cell, step)), 0.16)       # (two eyelets 0.32 apart)
    ey = math_('SUBTRACT', top_d, 0.5 * band_d)
    er = math_('SQRT', math_('ADD', math_('MULTIPLY', ex, ex), math_('MULTIPLY', ey, ey)))
    hole = math_('LESS_THAN', er, 0.055)
    rim = math_('MULTIPLY', math_('LESS_THAN', er, 0.095), math_('SUBTRACT', 1.0, hole))
    # the colour
    shade = math_('MULTIPLY', rng(dye.outputs['Fac'], 0.3, 0.7, 1.0 - P['mottle'], 1.0 + P['mottle']),
                  rng(weave, 0.0, 1.0, 0.78, 1.08))
    shade = math_('MULTIPLY', shade, rng(nap.outputs['Fac'], 0.3, 0.7, 0.9, 1.08))
    shade = math_('MULTIPLY', shade, math_('SUBTRACT', 1.0, math_('MULTIPLY', groove, 0.45)))
    shade = math_('MULTIPLY', shade, math_('SUBTRACT', 1.0, math_('MULTIPLY', hem_line, 0.35)))
    shade = math_('MULTIPLY', shade, math_('SUBTRACT', 1.0, math_('MULTIPLY', hole, 0.85)))
    shade = math_('MULTIPLY', shade, math_('ADD', 1.0, math_('MULTIPLY', rim, 0.25)))
    ao = N('ShaderNodeAmbientOcclusion')
    ao.inputs['Distance'].default_value = 1.0
    ao.samples = 8
    shade = math_('MULTIPLY', shade, rng(ao.outputs['AO'], 0.0, 1.0, 0.55, 1.0))
    col = N('ShaderNodeVectorMath')
    col.operation = 'SCALE'
    col.inputs[0].default_value = P['colour']
    nt.links.new(shade, col.inputs['Scale'])
    # the relief: the weave, the nap over it, the seams' ridge, the eyelets' worked rims
    h = math_('ADD', math_('MULTIPLY', weave, 1.0), math_('MULTIPLY', nap.outputs['Fac'], 0.6))
    bump = N('ShaderNodeBump')
    bump.inputs['Strength'].default_value, bump.inputs['Distance'].default_value = P['relief'], 0.4 * P['pitch']
    nt.links.new(h, bump.inputs['Height'])
    big = math_('SUBTRACT', math_('ADD', ridge, math_('MULTIPLY', rim, 0.8)), math_('ADD', math_('MULTIPLY', groove, 0.6),
                                                                                      math_('MULTIPLY', hole, 0.8)))
    bump2 = N('ShaderNodeBump')
    bump2.inputs['Strength'].default_value, bump2.inputs['Distance'].default_value = 0.6, 0.03
    nt.links.new(big, bump2.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], bump2.inputs['Normal'])
    pb = N('ShaderNodeBsdfPrincipled')
    nt.links.new(col.outputs['Vector'], pb.inputs['Base Color'])
    nt.links.new(bump2.outputs['Normal'], pb.inputs['Normal'])
    pb.inputs['Roughness'].default_value = P['rough']
    pb.inputs['Specular IOR Level'].default_value = 0.25
    pb.inputs['Sheen Weight'].default_value = P['sheen']
    pb.inputs['Sheen Roughness'].default_value = 0.45
    pb.inputs['Sheen Tint'].default_value = (1.0, 0.72, 0.66, 1.0)
    pb.name = 'close bsdf'
    nt.links.new(pb.outputs['BSDF'], out.inputs['Surface'])


def build():
    res = SP.build()
    body = bpy.data.objects['Skin body']
    rig = bpy.data.objects.get('Rig')
    for o in bpy.data.objects:                   # (the necklace's parts: NO_ACC hides only the chain)
        if o.name.startswith('Necklace'):
            o.hide_render = True
            o.visible_camera = o.visible_shadow = o.visible_diffuse = o.visible_glossy = False
    bpy.context.view_layer.update()
    # the unit: the head's height as it sits on the body, chin to crown (anatomy.head_height measures it)
    H = SP.anatomy.head_height() or SP.anatomy.HEAD
    B = draft.body_data(body)
    M = draft.measure_legs(B, rig, H)
    draft.report(M)
    MEASURES.update(M)
    MEASURES['band'] = draft.RATIOS['band'] * H
    MEASURES['eyelet_step'] = M['waist'] / WOOL['eyelets']
    o = draft.build_hose(body, M, rig=rig, material=female.mat(M_WOOL), foot=FOOT)
    if MASK:
        draft.mask_body(body)
    wool_closeup('hose wool')
    # the --head view on a part of the hose
    floor = float(B['P'][:, 2].min())
    frames = {'legs': (floor - 0.5, M['waist_z'] + 1.5), 'hips': (M['crotch_z'] - 3.0, M['waist_z'] + 1.5),
              'knee': (M['knee_z'] - 3.5, M['knee_z'] + 3.5), 'ankle': (floor - 0.4, M['ankle_z'] + 4.5)}
    lo, hi = frames[FRAMES[VIEW] if isinstance(VIEW, int) else VIEW]
    os.environ.setdefault('HEAD_CZ', '%.2f' % (0.5 * (lo + hi)))
    os.environ.setdefault('HEAD_H', '%.2f' % (hi - lo))
    bpy.context.view_layer.update()
    if os.environ.get('HOSE_CHECK'):
        sc = bpy.context.scene
        for f in [int(x) for x in os.environ['HOSE_CHECK'].split(',')]:
            sc.frame_set(f)
            draft.fit_check(o, body, 'sheet column %d' % f)
        sc.frame_set(0)
    return res

