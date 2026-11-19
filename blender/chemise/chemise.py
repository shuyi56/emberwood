"""A standalone garment: a loose linen chemise on the bare mannequin (the anatomy thread's head-driven body), with
none of the gown's pieces and nothing worn over it. The chemise is drafted in drape.py from the body's own
measurements; the linen's maps are made in linen_texture.py; this module supplies the body and the linen's look.

    python3 blender/base/views.py --figure chemise --mannequin --head --out DIR

--head frames the garment here (HEAD_CZ, HEAD_H below): from its hem to the chin."""
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
import numpy as np  # noqa: E402
import female  # noqa: E402
import sorceress_pendant as SP  # noqa: E402
import fit  # noqa: E402
import drape  # noqa: E402
import linen_texture  # noqa: E402

M_LINEN = female.mat_def(72, 'chemise linen', female.ramp(0xdcd8cc, 0.12), 0.0, closeup='linen')
MEASURES = {}
# White cloth shows its folds only by their shadows. The views' warm light from below (the floor's bounce) is strong
# enough to light the inside of a hanging sleeve as brightly as its outside and to fill every fold, so for this
# garment's own views it is turned down, and the fill a little, and the key's shadows are left to do the work.
LIGHT_SCALE = {'Bounce': 0.25, 'Fill': 0.6}
PITCH = 0.0139               # one thread, in units: 70 threads to the inch on a 165 cm figure 63 units tall

# The linen's look (the reasons are in linen_texture.py): bleached, so a white just off toward cream; the weave too
# fine to see from a step back, where the cloth is even but for short faint streaks, mostly weft-wise; pressed flat, so
# a smooth face with the soft polish of line flax, most on the yarns' crowns. It is opaque: a see-through layer let the
# light under the sleeve shine through it, and the folds lost their shadows.
LINEN = dict(colour=(0.800, 0.780, 0.715), streak=0.07, yarn=0.5, rough=(0.52, 0.72), specular=0.45, sheen=0.06,
             relief=0.35)


def linen_closeup(name, P=LINEN):
    """The 3D look of the linen on a material from female.mat (its flat sprite look kept beside it)."""
    m = bpy.data.materials.get(name)
    if m is None:
        return
    nt = m.node_tree
    N = nt.nodes.new
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
    flat = next((n for n in nt.nodes if n.type == 'BSDF_DIFFUSE'), None)
    if flat is not None:
        flat.name = 'sprite bsdf'
    weave_img, streak_img = linen_texture.make_images()
    pitch = P.get('pitch', PITCH)              # (a coarser cloth of the same weave: a wider thread)
    uv = N('ShaderNodeUVMap')

    def image(img, span):
        mp = N('ShaderNodeMapping')
        mp.inputs['Scale'].default_value = (1.0 / span,) * 3
        nt.links.new(uv.outputs['UV'], mp.inputs['Vector'])
        tex = N('ShaderNodeTexImage')
        tex.image, tex.interpolation = img, 'Cubic'
        nt.links.new(mp.outputs['Vector'], tex.inputs['Vector'])
        sep = N('ShaderNodeSeparateColor')
        nt.links.new(tex.outputs['Color'], sep.inputs['Color'])
        return sep.outputs['Red'], sep.outputs['Green'], sep.outputs['Blue']
    height, tone, rub = image(weave_img, linen_texture.WEAVE_THREADS * pitch)
    streak, body, _ = image(streak_img, linen_texture.STREAK_SPAN * pitch)

    def rng(sock, a, b, lo, hi):
        r = N('ShaderNodeMapRange')
        r.inputs['From Min'].default_value, r.inputs['From Max'].default_value = a, b
        r.inputs['To Min'].default_value, r.inputs['To Max'].default_value = lo, hi
        nt.links.new(sock, r.inputs['Value'])
        return r.outputs['Result']

    def math_(op, a, b):
        k = N('ShaderNodeMath')
        k.operation = op
        for i, v in enumerate((a, b)):
            if isinstance(v, float):
                k.inputs[i].default_value = v
            else:
                nt.links.new(v, k.inputs[i])
        return k.outputs[0]

    def scaled(colour, fac):
        k = N('ShaderNodeVectorMath')
        k.operation = 'SCALE'
        if isinstance(colour, tuple):
            k.inputs[0].default_value = colour
        else:
            nt.links.new(colour, k.inputs[0])
        nt.links.new(fac, k.inputs['Scale'])
        return k.outputs['Vector']
    # the shade: the streaks seen from a step back, the yarns' own seen close, and the folds' hollows a little dimmer
    shade = math_('MULTIPLY', rng(streak, 0.0, 1.0, 1.0 - P['streak'], 1.0 + P['streak']),
                  rng(tone, 0.0, 1.0, 1.0 - P['yarn'], 1.0 + P['yarn']))
    ao = N('ShaderNodeAmbientOcclusion')
    ao.inputs['Distance'].default_value = 1.2
    ao.samples = 8
    col = scaled(P['colour'], math_('MULTIPLY', shade, rng(ao.outputs['AO'], 0.0, 1.0, 0.45, 1.0)))
    # the relief: the weave, and the fuller yarns standing a little proud
    bump = N('ShaderNodeBump')
    bump.inputs['Strength'].default_value, bump.inputs['Distance'].default_value = P['relief'], 0.5 * pitch
    nt.links.new(math_('ADD', height, math_('MULTIPLY', body, 0.6)), bump.inputs['Height'])
    pb = N('ShaderNodeBsdfPrincipled')
    nt.links.new(col, pb.inputs['Base Color'])
    nt.links.new(bump.outputs['Normal'], pb.inputs['Normal'])
    nt.links.new(rng(rub, 0.0, 1.0, P['rough'][1], P['rough'][0]), pb.inputs['Roughness'])     # (polished on the crowns)
    pb.inputs['Specular IOR Level'].default_value = P['specular']
    pb.inputs['Sheen Weight'].default_value = P['sheen']
    pb.inputs['Sheen Roughness'].default_value = 0.3
    pb.name = 'close bsdf'
    nt.links.new(pb.outputs['BSDF'], out.inputs['Surface'])


def dress(body, m, under=None):
    """The chemise on a body already built (m: fit.measure's measurements), under another garment or alone (see
    drape.build_chemise). Returns the object and its draft."""
    o, d = drape.build_chemise(body, m, rig=bpy.data.objects.get('Rig'), material=female.mat(M_LINEN),
                               settle=os.environ.get('CHEMISE_DRAWN') is None, under=under,
                               colliders=[c for c in (bpy.data.objects.get('Hand.L'), bpy.data.objects.get('Hand.R')) if c])
    linen_closeup('chemise linen')
    return o, d


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
    P, _, aw = fit.body_arrays(body)
    hp = np.array([(o.matrix_world @ v.co)[:] for o in heads for v in o.data.vertices]) if heads else None
    m = fit.measure(P, aw, hp)
    MEASURES.update(m)
    _, d = dress(body, m)
    # the --head view on the garment: from under its hem to the chin
    lo, hi = d.z_hem_worn - 0.6 * m['w'], m['z_chin'] + 0.8 * m['w']
    os.environ.setdefault('HEAD_CZ', '%.2f' % (0.5 * (lo + hi)))
    os.environ.setdefault('HEAD_H', '%.2f' % (hi - lo))
    bpy.context.view_layer.update()
    return res
