"""A standalone garment: the petticoat under the riding habit (about 1885-1895), on the bare mannequin over the corset,
with nothing worn over it. It is drafted in draft.py from the body's own measurements and the corset's shared numbers
(habit.py), hung by Blender's cloth solver in cloth.py, and finished with its flounce in flounce.py; this module
supplies the body and the cotton's look.

    python3 blender/base/views.py --figure petticoat --mannequin --turn --clay --out DIR
    python3 blender/base/views.py --figure petticoat --mannequin --head --set VIEW=1 --out DIR

VIEW frames the --head close-up on a part, by its place in FRAMES: 0 'skirt' (the band to the floor), 1 'yoke' (the
band to below the hips), 2 'hem' (the flounce and the train). SETTLE=False skips the solver (the drafted hang)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for d in ('sorceress_pendant', 'shared', 'base'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
# (the bare mannequin: its own plain head, the gown's pieces and the necklace out of sight)
os.environ.setdefault('HEAD_GLB', '')
os.environ.setdefault('NO_DRESS', '1')
os.environ.setdefault('NO_ACC', '1')
import bpy  # noqa: E402
import numpy as np  # noqa: E402
import female  # noqa: E402
import sorceress_pendant as SP  # noqa: E402
import draft  # noqa: E402
import cloth  # noqa: E402
import habit  # noqa: E402

FRAMES = ('skirt', 'yoke', 'hem')
VIEW = 0
SETTLE = True
FLOUNCE = True
CORSET = True                # wear the corset under it (False: the skin drawn in, the corset's surface measured but not built)
MEASURES = {}
M_COTTON = female.mat_def(76, 'petticoat cotton', female.ramp(0xe6e2d8, 0.1), 0.0, closeup='cloth')
# A bleached cotton longcloth, starched: white just off toward cream, a fine plain weave too small to see from a step
# back, a soft dull face. Opaque (a petticoat is two layers at the flounce and the light does not come through).
COTTON = dict(colour=(0.80, 0.785, 0.74), pitch=0.012, relief=0.25, rough=0.62, sheen=0.25, tuck_w=0.11, tucks=())
# White cloth shows its folds by their shadows: the floor's warm bounce fills every flute, so it is turned down here.
LIGHT_SCALE = {'Bounce': 0.3, 'Fill': 0.7}


def body_arrays(body):
    me = body.data
    n = len(me.vertices)
    P = np.empty(n * 3)
    me.vertices.foreach_get('co', P)
    P = P.reshape(-1, 3)
    mw = np.array(body.matrix_world)
    P = P @ mw[:3, :3].T + mw[:3, 3]
    names = [g.name for g in body.vertex_groups]
    W = np.zeros((n, len(names)), dtype=np.float32)
    for v in me.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    return P, W, names


def under(objs, H):
    """The points of what is worn under the petticoat (the corset's pieces), evaluated, for the measure."""
    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    for o in objs or ():
        if o is None or o.type != 'MESH':
            continue
        e = o.evaluated_get(dg)
        me = e.to_mesh()
        mw = np.array(o.matrix_world)
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get('co', co)
        pts.append(co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3])
        e.to_mesh_clear()
    return np.concatenate(pts) if pts else None


def corset_surface(HM, H):
    """With no corset built: the corset's outer surface as habit gives it, as points (its own theta runs toward her
    right from the axis (0, cy))."""
    th = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    pts = []
    for t in th:
        for z in np.arange(float(HM.bottom(t)), float(HM.top(t)), 0.1):
            r = float(habit.corseted_section(HM, z, t)) + habit.CORSET_THICK * H
            pts.append((-r * np.sin(t), HM.cy - r * np.cos(t), z))
    return np.array(pts)


def make_object(V, F, info, wts, rig, name='Petticoat'):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(p) for p in V], [], [tuple(f) for f in F])
    for p in me.polygons:
        p.use_smooth = True
    uv = me.uv_layers.new(name='pattern')
    u, v = info['u'], info['v']
    lo = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get('vertex_index', lo)
    # (a ring's u wraps at the centre back: a face that spans the wrap takes its far corners round by the girth)
    uu = u[lo].reshape(-1, 3)
    G = np.array([info['G'][info['row'][i]] for i in lo]).reshape(-1, 3)
    wrap = (uu.max(1) - uu.min(1)) > 0.5 * G.min(1)
    uu = np.where(wrap[:, None] & (uu < 0.5 * G), uu + G, uu)
    uv.data.foreach_set('uv', np.stack([uu.reshape(-1), v[lo]], 1).reshape(-1))
    rest = me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
    rest.data.foreach_set('vector', V.reshape(-1))
    t = me.attributes.new('down', 'FLOAT', 'POINT')
    t.data.foreach_set('value', info['t'])
    # (how far up the cloth from the hem, for the tucks above the flounce)
    up = me.attributes.new('to_hem', 'FLOAT', 'POINT')
    up.data.foreach_set('value', info['v'] / np.maximum(info['t'], 1e-6) * (1.0 - info['t']))
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    me.materials.append(female.mat(M_COTTON))
    for bone, w in wts.items():
        g = o.vertex_groups.new(name=bone)
        for i in np.nonzero(w > 1e-4)[0]:
            g.add([int(i)], float(w[i]), 'REPLACE')
    m = o.modifiers.new('rig', 'ARMATURE')
    m.object = rig
    return o


def finish(o, M):
    m = o.modifiers.new('thickness', 'SOLIDIFY')
    m.thickness = draft.RATIOS['thickness'] * M['H']
    m.offset = -1.0
    m = o.modifiers.new('smooth', 'SUBSURF')
    m.levels = m.render_levels = 1


def cotton_closeup(name, P=COTTON):
    """The 3D look of the cotton on a material from female.mat (its flat sprite look kept beside it): a fine plain
    weave on the cut pattern (the 'pattern' UVs, in model units), its slight unevenness, a soft dull face."""
    m = bpy.data.materials.get(name)
    if m is None:
        return
    nt = m.node_tree
    N = nt.nodes.new
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
    uv = N('ShaderNodeUVMap')
    uv.uv_map = 'pattern'
    sep = N('ShaderNodeSeparateXYZ')
    nt.links.new(uv.outputs['UV'], sep.inputs['Vector'])

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
    k = 1.0 / P['pitch']
    a = math_('MULTIPLY', sep.outputs['X'], k)
    b = math_('MULTIPLY', sep.outputs['Y'], k)
    up = math_('FLOORED_MODULO', math_('ADD', math_('FLOOR', a), math_('FLOOR', b)), 2.0)
    ya = math_('SINE', math_('MULTIPLY', math_('FRACT', b), 3.14159))
    yb = math_('SINE', math_('MULTIPLY', math_('FRACT', a), 3.14159))
    mix = N('ShaderNodeMix')
    mix.data_type = 'FLOAT'
    nt.links.new(up, mix.inputs['Factor'])
    nt.links.new(ya, mix.inputs['A'])
    nt.links.new(yb, mix.inputs['B'])
    weave = mix.outputs['Result']
    noise = N('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value, noise.inputs['Detail'].default_value = 0.6, 3.0
    nt.links.new(uv.outputs['UV'], noise.inputs['Vector'])
    shade = math_('ADD', 0.94, math_('MULTIPLY', noise.outputs['Fac'], 0.1))
    col = N('ShaderNodeVectorMath')
    col.operation = 'SCALE'
    col.inputs[0].default_value = P['colour']
    nt.links.new(shade, col.inputs['Scale'])
    bump = N('ShaderNodeBump')
    bump.inputs['Strength'].default_value, bump.inputs['Distance'].default_value = P['relief'], 0.3 * P['pitch']
    nt.links.new(weave, bump.inputs['Height'])
    # the tucks above the flounce: TUCKS folded ridges round the skirt, each casting a thin shade below it
    th = N('ShaderNodeAttribute')
    th.attribute_name = 'to_hem'
    ridge = None
    shadow = None
    for c in P['tucks']:
        d = math_('SUBTRACT', th.outputs['Fac'], c)
        r = math_('EXPONENT', math_('MULTIPLY', math_('POWER', math_('DIVIDE', d, P['tuck_w']), 2.0), -1.0))
        sh = math_('EXPONENT', math_('MULTIPLY', math_('POWER', math_('DIVIDE', math_('ADD', d, 1.6 * P['tuck_w']),
                                                                                  0.7 * P['tuck_w']), 2.0), -1.0))
        ridge = r if ridge is None else math_('ADD', ridge, r)
        shadow = sh if shadow is None else math_('ADD', shadow, sh)
    if ridge is not None:
        shade = math_('MULTIPLY', shade, math_('SUBTRACT', 1.0, math_('MULTIPLY', shadow, 0.28)))
        nt.links.new(shade, col.inputs['Scale'])
        bump2 = N('ShaderNodeBump')
        bump2.inputs['Strength'].default_value, bump2.inputs['Distance'].default_value = 0.8, 0.08
        nt.links.new(ridge, bump2.inputs['Height'])
        nt.links.new(bump2.outputs['Normal'], bump.inputs['Normal'])
    pb = N('ShaderNodeBsdfPrincipled')
    nt.links.new(col.outputs['Vector'], pb.inputs['Base Color'])
    nt.links.new(bump.outputs['Normal'], pb.inputs['Normal'])
    pb.inputs['Roughness'].default_value = P['rough']
    pb.inputs['Specular IOR Level'].default_value = 0.3
    pb.inputs['Sheen Weight'].default_value = P['sheen']
    pb.inputs['Sheen Roughness'].default_value = 0.5
    pb.name = 'close bsdf'
    nt.links.new(pb.outputs['BSDF'], out.inputs['Surface'])


def wear_on_points(body, rig, H, pts, log=print):
    return wear(body, rig, H, (), log=log, extra=pts)


def wear(body, rig, H, under_objs=(), log=print, extra=None):
    """Put the petticoat on a body that is already corseted (habit.compress_body) with the corset on it: under_objs are
    what it goes over (the corset's meshes). Another garment over it (the habit's skirt) calls this and measures the
    returned objects as the petticoat measures the corset. Returns (objects, M, pattern)."""
    P, W, names = body_arrays(body)
    if extra is None:
        extra = under(under_objs, H)
    M = draft.measure(P, W, names, H, extra=extra)
    draft.report(M, log)
    pat = draft.pattern(M, log=log)
    V, F, info = draft.hang(M, pat, log=log)
    wts = draft.weights(V, info, M, pat)
    o = make_object(V, F, info, wts, rig)
    if SETTLE:
        bpy.context.scene.frame_set(1)
        pinned = (V[:, 2] > pat['band_bot'] - 0.02).astype(float)
        hands = [bpy.data.objects[n] for n in ('Hand.L', 'Hand.R') if n in bpy.data.objects]
        # (it falls on what it is worn over: the body, the hands, and the corset)
        cloth.settle(o, rig, [body] + hands + [u for u in under_objs if u.type == 'MESH'], pinned, M['floor'], log=log)
    objs = [o]
    if FLOUNCE:
        import flounce
        objs.append(flounce.build(o, M, pat, info, rig, female.mat(M_COTTON), log=log))
    for ob in objs:
        finish(ob, M)
    # three tucks a hand's width above the flounce, about 1.2 cm apart (they shorten the cloth and stiffen the hem)
    D = draft.RATIOS['flounce'] * M['H']
    gap = 0.05 * M['H']
    cotton_closeup('petticoat cotton', dict(COTTON, tucks=tuple(D + gap * (1.2 + k) for k in range(3))))
    bpy.context.view_layer.update()
    return objs, M, pat


def build():
    res = SP.build()
    body = bpy.data.objects['Skin body']
    rig = bpy.data.objects.get('Rig') or female.RIG['arm']
    for o in bpy.data.objects:
        if o.name.startswith('Necklace'):
            o.hide_render = True
            o.visible_camera = o.visible_shadow = o.visible_diffuse = o.visible_glossy = False
    bpy.context.view_layer.update()
    H = SP.anatomy.head_height() or SP.anatomy.HEAD
    # the corset's work first (habit.py, the corset thread's): the skin drawn in, the corset put on
    HM = habit.measure(body, head_h=H)
    habit.compress_body(body, HM)
    before = set(bpy.data.objects)
    if CORSET:
        habit.build_corset(body, HM, rig)
    worn = [o for o in bpy.data.objects if o not in before and o.type == 'MESH']
    bpy.context.view_layer.update()
    print('petticoat: corset %s, natural waist %.2f, corseted %.2f' % (
        'worn' if CORSET else 'measured only', HM.natural_waist, HM.girth_waist_corseted))
    if not CORSET:
        # (no corset built: its surface as habit gives it is what the band sits on)
        objs, M, pat = wear_on_points(body, rig, H, corset_surface(HM, H))
    else:
        objs, M, pat = wear(body, rig, H, worn)
    MEASURES.update({k: v for k, v in M.items() if isinstance(v, float)})
    floor = M['floor']
    frames = {'skirt': (floor - 0.5, pat['band_top'] + 1.5), 'yoke': (M['hips_z'] - 4.0, pat['band_top'] + 2.0),
              'hem': (floor - 0.5, floor + 1.4 * H)}
    lo, hi = frames[FRAMES[VIEW] if isinstance(VIEW, int) else VIEW]
    os.environ.setdefault('HEAD_CZ', '%.2f' % (0.5 * (lo + hi)))
    os.environ.setdefault('HEAD_H', '%.2f' % (hi - lo))
    bpy.context.view_layer.update()
    return res
