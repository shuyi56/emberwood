"""A standalone garment: a fitted leather vest with a stand collar, on the bare mannequin (the anatomy thread's
head-driven body), with none of the gown's pieces. The vest itself is drafted in fit.py from the body's own
measurements; this module only supplies the body and the leather.

    python3 blender/base/views.py --figure vest --mannequin --head --clay --angles 3/4 --out DIR

--head frames the bust and neck here (HEAD_CZ, HEAD_H below), which is all this garment needs to be judged by."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'sorceress_pendant'))
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
sys.path.insert(0, os.path.join(HERE, '..', 'base'))
# (the bare mannequin: its own plain head, the gown's pieces and the necklace out of sight)
os.environ.setdefault('HEAD_GLB', '')
os.environ.setdefault('NO_DRESS', '1')
os.environ.setdefault('NO_ACC', '1')
import bpy  # noqa: E402
import female  # noqa: E402
import sorceress_pendant as SP  # noqa: E402
import fit  # noqa: E402

M_VEST = female.mat_def(70, 'vest leather', female.ramp(int(os.environ.get('VEST', '8a5a34'), 16), 0.12), 0.0,
                        surface='leather', closeup='leather')
# (the plate is the warden's bronze: the same ramp for the sprite, and for the 3D look her bronze_closeup itself,
# read out of warden.py, so the two stay one material; importing her module would set her face and hair on this figure)
M_BRASS = female.mat_def(71, 'vest bronze', female.ramp(0x9a6438, 0.12), 0.0, closeup='steel')
BRASS = os.environ.get('VEST_PLAIN') is None     # (VEST_PLAIN=1: the leather alone)
MEASURES = {}

# The glaze: a fired, glassy coat on the bronze, as enamelled armour, on the trims only: a narrow line of teal glaze
# along each lame's showing lower edge, along both edges of the belt, and just inside the rolled border of the plates
# (round the collar, down the opening, at the armholes and the hem). The plates' faces stay bare bronze. The lines are
# laid out by the plates' own coordinates from fit.py ('paint_u' across a plate, 'paint_e' in from its border), so
# they follow the plates as the body changes. What makes it glaze and not paint is in glaze_texture.py's maps:
#   thickness  the glass is translucent: where it lies thin (toward a line's edges, and in the map's thin places) the
#              bronze glows through it, where it pools it is deep and saturated
#   crazing    hairline cracks through the glass, a shade darker (dirt in them) and just felt in the relief
#   pinholes   pits burst through to the bare metal
# over a glossy surface (a clear coat on top of the colour), domed up from the metal with a rounded lip at its edges.
GLAZE = dict(deep=(0.006, 0.050, 0.085), thin=(0.040, 0.230, 0.290),
             lines_u=((0.225, 0.345), (0.78, 0.90)),      # along a lame's lower edge (and, on the belt, its upper edge too:
                                                          # on a lame that place is under the lame above)
             lines_e=((0.50, 1.00),),                     # inside the plates' rolled border
             lip=0.035, gloss=0.14, scale=0.115)
# The vest's bronze is the warden's 'aged' look with the sheen taken down: the lames are domed, so over the bust each one
# caught the key light as a bright pool. A rougher surface (the same tarnish and rub, roughness 0.54 to 0.80 instead
# of 0.40 to 0.68) spreads those pools out and dims them, and keeps a soft satin on the high places.
VEST_LOOK = dict(rgh=(float(os.environ.get('VEST_ROUGH_LO', '0.54')), float(os.environ.get('VEST_ROUGH_HI', '0.80'))))


def warden_bronze():
    """The warden's bronze_closeup and her 'aged' look (with the vest's roughness), without loading her figure."""
    import ast
    path = os.path.join(HERE, '..', 'warden', 'warden.py')
    with open(path) as f:
        tree = ast.parse(f.read())
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name == 'bronze_closeup')
            or (isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') in ('BRONZE_LOOKS', 'BRONZE_LOOK'))]
    ns = {'bpy': bpy}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, 'exec'), ns)
    ns['BRONZE_LOOKS']['vest'] = dict(ns['BRONZE_LOOKS']['aged'], **VEST_LOOK)
    return lambda name: ns['bronze_closeup'](name, 'vest')


def glaze_coat(name, P=GLAZE):
    """Lay the glaze over a bronze material built by bronze_closeup: its colour, gloss and relief mixed in ahead of
    the bronze's own, by a mask from the plates' 'paint_u' coordinate and the glaze texture's maps."""
    import glaze_texture
    m = bpy.data.materials[name]
    nt = m.node_tree
    N = nt.nodes.new
    pb = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')

    def source(sock):
        return sock.links[0].from_socket if sock.is_linked else None
    base, rough, norm = (source(pb.inputs[k]) for k in ('Base Color', 'Roughness', 'Normal'))
    metal0 = float(pb.inputs['Metallic'].default_value)
    rest = N('ShaderNodeAttribute')
    rest.attribute_name = 'rest'
    u = N('ShaderNodeAttribute')
    u.attribute_type, u.attribute_name = 'GEOMETRY', 'paint_u'
    # (the texture laid on by the plates' rest positions, from three sides and blended, so it needs no unwrapping
    # and does not swim when the figure moves)
    mp = N('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = (P['scale'],) * 3
    nt.links.new(rest.outputs['Vector'], mp.inputs['Vector'])
    tex = N('ShaderNodeTexImage')
    tex.image = bpy.data.images.get('vest glaze') or glaze_texture.make_image('vest glaze')
    tex.projection, tex.projection_blend, tex.interpolation = 'BOX', 0.25, 'Cubic'
    nt.links.new(mp.outputs['Vector'], tex.inputs['Vector'])
    sep = N('ShaderNodeSeparateColor')
    nt.links.new(tex.outputs['Color'], sep.inputs['Color'])
    thick, craze, pin = sep.outputs['Red'], sep.outputs['Green'], sep.outputs['Blue']

    def rng(sock, a, b, lo=0.0, hi=1.0):
        r = N('ShaderNodeMapRange')
        r.interpolation_type = 'SMOOTHSTEP'
        r.inputs['From Min'].default_value, r.inputs['From Max'].default_value = a, b
        r.inputs['To Min'].default_value, r.inputs['To Max'].default_value = lo, hi
        nt.links.new(sock, r.inputs['Value'])
        return r.outputs['Result']

    def mth(op, a, b):
        k = N('ShaderNodeMath')
        k.operation, k.use_clamp = op, True
        for i, v in enumerate((a, b)):
            if isinstance(v, float):
                k.inputs[i].default_value = v
            else:
                nt.links.new(v, k.inputs[i])
        return k.outputs[0]

    def mix(fac, a, b, kind='RGBA', blend='MIX'):
        k = N('ShaderNodeMix')
        k.data_type = kind
        if kind == 'RGBA':
            k.blend_type = blend
        ia, ib = ('A', 'B') if kind == 'RGBA' else (2, 3)
        for key, v in ((ia, a), (ib, b)):
            if isinstance(v, tuple):
                k.inputs[key].default_value = (*v, 1)
            elif isinstance(v, float):
                k.inputs[key].default_value = v
            elif v is not None:
                nt.links.new(v, k.inputs[key])
        if isinstance(fac, float):
            k.inputs['Factor'].default_value = fac
        else:
            nt.links.new(fac, k.inputs['Factor'])
        return k.outputs['Result' if kind == 'RGBA' else 0]

    e = N('ShaderNodeAttribute')
    e.attribute_type, e.attribute_name = 'GEOMETRY', 'paint_e'
    # the glass flowed to its own edge in the kiln: a line's edge wanders a little with the thickness map
    wob = mth('MULTIPLY', rng(thick, 0.0, 1.0, -1.0, 1.0), 0.008)
    k_ = N('ShaderNodeMath')
    k_.operation = 'ADD'
    nt.links.new(u.outputs['Fac'], k_.inputs[0])
    nt.links.new(wob, k_.inputs[1])
    uu = k_.outputs[0]
    k_ = N('ShaderNodeMath')
    k_.operation = 'ADD'
    nt.links.new(e.outputs['Fac'], k_.inputs[0])
    nt.links.new(mth('MULTIPLY', wob, 4.0), k_.inputs[1])
    ee = k_.outputs[0]

    # (a lame's line stops short of the border's line, so the two never cross: the border frames the lames' lines)
    clear = rng(ee, P['lines_e'][-1][1] + 0.12, P['lines_e'][-1][1] + 0.30)

    def lines(soft_u, soft_e):
        out = None
        for sock, spans, soft, keep in ((uu, P['lines_u'], soft_u, clear), (ee, P['lines_e'], soft_e, None)):
            for a, b in spans:
                one = mth('MULTIPLY', rng(sock, a, a + soft), rng(sock, b, b - soft))
                if keep is not None:
                    one = mth('MULTIPLY', one, keep)
                out = one if out is None else mth('MAXIMUM', out, one)
        return out
    outer = lines(0.01, 0.04)
    mask = mth('MULTIPLY', outer, mth('SUBTRACT', 1.0, pin))
    # how thick the glass lies: thin on a rounded lip at the line's edges, then the map's pools and thin places
    lip = lines(P['lip'], 4 * P['lip'])
    depth = mth('MULTIPLY', lip, rng(thick, 0.0, 1.0, 0.55, 1.0))
    # its colour: the bronze seen through thin glass, the glass's own deep colour where it pools
    seen = mix(1.0, base, P['thin'], blend='MULTIPLY') if base is not None else P['thin']
    colour = mix(rng(depth, 0.15, 0.85), mix(0.55, seen, P['thin']), P['deep'])
    colour = mix(mth('MULTIPLY', craze, 0.5), colour, (0.02, 0.025, 0.03))          # (dirt in the cracks)
    nt.links.new(mix(mask, base, colour), pb.inputs['Base Color'])
    nt.links.new(mix(mask, rough, mth('ADD', P['gloss'], mth('MULTIPLY', craze, 0.25)), kind='FLOAT'), pb.inputs['Roughness'])
    # (thin glass still shows the metal's shine through it; deep glass does not)
    nt.links.new(mth('MULTIPLY', mth('SUBTRACT', 1.0, mth('MULTIPLY', mask, rng(depth, 0.0, 0.5, 0.55, 1.0))), metal0),
                 pb.inputs['Metallic'])
    for key, v in (('Coat Weight', mask), ('Coat Roughness', 0.12), ('Coat IOR', 1.52)):
        if key in pb.inputs:
            if isinstance(v, float):
                pb.inputs[key].default_value = v
            else:
                nt.links.new(v, pb.inputs[key])
    # the relief: the glass domed up from the metal, its surface gently uneven, a hairline groove at each crack
    height = mth('SUBTRACT', mth('MULTIPLY', mask, mth('ADD', mth('MULTIPLY', lip, 0.8), mth('MULTIPLY', thick, 0.2))),
                 mth('MULTIPLY', craze, 0.12))
    bump = N('ShaderNodeBump')
    bump.inputs['Strength'].default_value, bump.inputs['Distance'].default_value = 0.35, 0.02
    nt.links.new(height, bump.inputs['Height'])
    if norm is not None:
        nt.links.new(norm, bump.inputs['Normal'])
    nt.links.new(bump.outputs['Normal'], pb.inputs['Normal'])
    if 'Coat Normal' in pb.inputs:
        nt.links.new(bump.outputs['Normal'], pb.inputs['Coat Normal'])


def build():
    res = SP.build()
    body = bpy.data.objects['Skin body']
    head = bpy.data.objects.get('Head')
    heads = [o for o in (head.children_recursive if head else []) if o.type == 'MESH']
    for o in bpy.data.objects:                   # (the necklace's parts: NO_ACC hides only the chain)
        if o.name.startswith('Necklace'):
            o.hide_render = True
            o.visible_camera = o.visible_shadow = o.visible_diffuse = o.visible_glossy = False
    bpy.context.view_layer.update()
    vest, m = fit.build_vest(body, heads, rig=bpy.data.objects.get('Rig'), material=female.mat(M_VEST),
                             brass=female.mat(M_BRASS) if BRASS else None)
    MEASURES.update(m)
    if BRASS:
        warden_bronze()('vest bronze')
        if os.environ.get('VEST_GLAZE', '1') != '0':
            glaze_coat('vest bronze')
    # the --head view on the bust and neck: from under the bust to the chin, centred between them
    lo, hi = m['z_bust'] - 2.0 * m['w'], m['z_chin'] + 1.1 * m['w']
    os.environ.setdefault('HEAD_CZ', '%.2f' % (0.5 * (lo + hi)))
    os.environ.setdefault('HEAD_H', '%.2f' % (hi - lo))
    bpy.context.view_layer.update()
    fit.clearance(vest, body, heads)
    return res
