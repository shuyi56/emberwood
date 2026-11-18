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

# The paint: a hand-painted coat over the bronze, as the reference's bands. Each lame carries a band of teal between two
# cream pin-lines, the edges left bare so the bronze shows round it; the belt the same. The bands are laid out across
# each plate by 'paint_u' (0 at a lame's lower edge, 1 at its upper, from fit.py), so they follow the plates as the
# body changes. The paint is a dielectric (not metal) and satin, wobbles a little at its edges, and is chipped away
# in patches where the bronze shows through.
PAINT = dict(teal=(0.019, 0.084, 0.127), cream=(0.61, 0.51, 0.31), outer=(0.21, 0.60), inner=(0.245, 0.565),
             rough=0.55, chip=(0.64, 0.75))


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


def paint_coat(name, P=PAINT):
    """Lay the paint over a bronze material built by bronze_closeup: the paint's colour, roughness and metal mixed in
    ahead of the bronze's own, by a mask from the plates' 'paint_u' coordinate."""
    m = bpy.data.materials[name]
    nt = m.node_tree
    N = nt.nodes.new
    pb = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')

    def source(sock):
        return sock.links[0].from_socket if sock.is_linked else None
    base, rough, norm = (source(pb.inputs[k]) for k in ('Base Color', 'Roughness', 'Normal'))
    metal0 = pb.inputs['Metallic'].default_value
    rest = N('ShaderNodeAttribute')
    rest.attribute_name = 'rest'
    u = N('ShaderNodeAttribute')
    u.attribute_type, u.attribute_name = 'GEOMETRY', 'paint_u'

    def noise(scale, detail):
        t = N('ShaderNodeTexNoise')
        t.inputs['Scale'].default_value, t.inputs['Detail'].default_value = scale, detail
        nt.links.new(rest.outputs['Vector'], t.inputs['Vector'])
        return t.outputs['Fac']

    def rng(sock, a, b, lo=0.0, hi=1.0):
        r = N('ShaderNodeMapRange')
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
    # (the hand's wobble on every edge)
    uu = mth('ADD', u.outputs['Fac'], mth('MULTIPLY', rng(noise(11.0, 3.0), 0.35, 0.65, -1.0, 1.0), 0.012))
    # (a band: on between two edges, each softened over about a pixel)
    def band(a, b, soft=0.012):
        return mth('MULTIPLY', rng(uu, a, a + soft), rng(uu, b, b - soft))
    outer, inner = band(*P['outer']), band(*P['inner'])
    pin = mth('MULTIPLY', outer, mth('SUBTRACT', 1.0, inner))
    chip = rng(noise(3.2, 6.0), *P['chip'], 1.0, 0.0)           # (patches worn through to the bronze)
    mask = mth('MULTIPLY', outer, chip)
    tone = rng(noise(2.2, 2.0), 0.3, 0.7, 0.8, 1.1)
    colour = N('ShaderNodeMix')
    colour.data_type = 'RGBA'
    colour.inputs['A'].default_value = (*P['teal'], 1)
    colour.inputs['B'].default_value = (*P['cream'], 1)
    nt.links.new(pin, colour.inputs['Factor'])
    shade = N('ShaderNodeMix')
    shade.data_type, shade.blend_type = 'RGBA', 'MULTIPLY'
    shade.inputs['Factor'].default_value = 1.0
    nt.links.new(colour.outputs['Result'], shade.inputs['A'])
    cmb = N('ShaderNodeCombineColor')
    for c in ('Red', 'Green', 'Blue'):
        nt.links.new(tone, cmb.inputs[c])
    nt.links.new(cmb.outputs['Color'], shade.inputs['B'])
    out_col = N('ShaderNodeMix')
    out_col.data_type = 'RGBA'
    nt.links.new(mask, out_col.inputs['Factor'])
    if base is not None:
        nt.links.new(base, out_col.inputs['A'])
    nt.links.new(shade.outputs['Result'], out_col.inputs['B'])
    nt.links.new(out_col.outputs['Result'], pb.inputs['Base Color'])
    out_rgh = N('ShaderNodeMix')
    out_rgh.data_type = 'FLOAT'
    nt.links.new(mask, out_rgh.inputs['Factor'])
    if rough is not None:
        nt.links.new(rough, out_rgh.inputs[2])
    out_rgh.inputs[3].default_value = P['rough']
    nt.links.new(out_rgh.outputs[0], pb.inputs['Roughness'])
    nt.links.new(mth('MULTIPLY', mth('SUBTRACT', 1.0, mask), float(metal0)), pb.inputs['Metallic'])
    # (the paint stands a little proud of the metal: a fine step at every edge of it)
    bump = N('ShaderNodeBump')
    bump.inputs['Strength'].default_value, bump.inputs['Distance'].default_value = 0.35, 0.004
    nt.links.new(mask, bump.inputs['Height'])
    if norm is not None:
        nt.links.new(norm, bump.inputs['Normal'])
    nt.links.new(bump.outputs['Normal'], pb.inputs['Normal'])


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
        if os.environ.get('VEST_PAINT', '1') != '0':
            paint_coat('vest bronze')
    # the --head view on the bust and neck: from under the bust to the chin, centred between them
    lo, hi = m['z_bust'] - 2.0 * m['w'], m['z_chin'] + 1.1 * m['w']
    os.environ.setdefault('HEAD_CZ', '%.2f' % (0.5 * (lo + hi)))
    os.environ.setdefault('HEAD_H', '%.2f' % (hi - lo))
    bpy.context.view_layer.update()
    fit.clearance(vest, body, heads)
    return res
