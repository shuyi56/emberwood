"""A standalone garment: the riding habit's outer skirt (ca. 1885-1895) on the bare mannequin (the anatomy thread's
head-driven body), with nothing worn over it. The skirt is drafted in draft.py from the body's own measurements and
let fall on the legs by the cloth solver (settle.py); this module supplies the body, the rig, the push that keeps the
legs out of the cloth, and the wool's look.

    python3 blender/base/views.py --figure skirt --mannequin --turn --clay --out DIR
    python3 blender/base/views.py --figure skirt --mannequin --head --set VIEW=2 --out DIR
    python3 blender/skirt/check.py --out DIR          (the altered bodies and the walk: see check.py)

VIEW frames the --head close-up on one part of the skirt, by its place in FRAMES: 0 'skirt' (the waist to the floor,
the default), 1 'waist' (the join under the basque), 2 'hem' (the deep hem and the folds), 3 'train'. (A number,
because views.py evaluates every --set value and the queue's commands carry no quotes.)

The skirt is a long, closed A-line of navy wool broadcloth, fitted over the hips and flaring from there to a deep hem,
with a modest train behind. It hangs over the petticoat (its own thread): the ratios in draft.RATIOS that stand for the
petticoat's bulk ('ease_hips', 'ease_hem') are the one place its published measurements go."""
import math
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
import draft  # noqa: E402
import settle  # noqa: E402
import sorceress_pendant as SP  # noqa: E402

FRAMES = ('skirt', 'waist', 'hem', 'train')
VIEW = 0
SETTLE = os.environ.get('SKIRT_SETTLE') == '1'    # let the cloth solver drape it; off by default: the drafted pattern as cut
                             # (John's rule, 2026-11-19: a new layer is seen first without the solver; the solver round
                             # comes only after that render has been looked at, and never on the whole stack by default)
PUSH = True                  # keep the legs out of the cloth through the walk
MEASURES = {}
DRAFT = {}                   # the last draft and its measures (pattern.draft_pieces reads them; see check.py --pattern)
# Navy broadcloth, near black in the shade, blue where the light is on it.
WOOL = dict(colour=(0.012, 0.020, 0.055), pitch=0.030, relief=0.35, rough=0.88, sheen=0.55, mottle=0.08, nap=0.55,
            seam_ridge=0.09)
M_WOOL = female.mat_def(87, 'skirt wool', female.ramp(0x1c2a52, 0.1), 0.0, surface='twill', closeup='skirt')


def wool_closeup(name, P=WOOL):
    """The 3D look of the skirt's wool on a material from female.mat (its flat sprite look kept beside it): a close
    napped broadcloth (a fine 2/2 twill hidden under a short, even nap, with a soft sheen where the nap lies over),
    laid out on the cut pattern ('pattern' UVs, in model units), a slow unevenness of the dye, the gores' seams (a soft
    ridge with a groove of stitches), the deep hem's turn (its stitched line and the roll above it) and the
    waistband's seam."""
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
        return a.outputs['Fac']

    def line(d, centre, width):          # a soft line where the distance d is at centre
        return math_('EXPONENT', math_('MULTIPLY', math_('POWER', math_('DIVIDE', math_('SUBTRACT', d, centre), width),
                                                         2.0), -1.0))
    uv = N('ShaderNodeUVMap')
    uv.uv_map = 'pattern'
    sep = N('ShaderNodeSeparateXYZ')
    nt.links.new(uv.outputs['UV'], sep.inputs['Vector'])
    u, v = sep.outputs['X'], sep.outputs['Y']
    # the threads: warp down the cloth (v), weft round it (u); 2/2 twill, each thread a step on from the last
    a = math_('MULTIPLY', math_('ADD', u, v), 1.0 / (math.sqrt(2.0) * P['pitch']))
    b = math_('MULTIPLY', math_('SUBTRACT', v, u), 1.0 / (math.sqrt(2.0) * P['pitch']))
    ia, ib = math_('FLOOR', a), math_('FLOOR', b)
    fa, fb = math_('FRACT', a), math_('FRACT', b)
    up = math_('LESS_THAN', math_('FLOORED_MODULO', math_('SUBTRACT', ia, ib), 4.0), 2.0)
    over = N('ShaderNodeMix')
    over.data_type = 'FLOAT'
    nt.links.new(up, over.inputs['Factor'])
    nt.links.new(math_('MULTIPLY', math_('POWER', math_('SINE', math_('MULTIPLY', fa, math.pi)), 0.6), 0.75), over.inputs['A'])
    nt.links.new(math_('ADD', math_('MULTIPLY', math_('POWER', math_('SINE', math_('MULTIPLY', fb, math.pi)), 0.6), 0.75), 0.25),
                 over.inputs['B'])
    weave = over.outputs['Result']
    rest = N('ShaderNodeAttribute')
    rest.attribute_name = 'rest'
    nap = N('ShaderNodeTexNoise')
    nap.inputs['Scale'].default_value, nap.inputs['Detail'].default_value = 34.0, 6.0
    nap.inputs['Roughness'].default_value = 0.7
    nt.links.new(rest.outputs['Vector'], nap.inputs['Vector'])
    dye = N('ShaderNodeTexNoise')
    dye.inputs['Scale'].default_value, dye.inputs['Detail'].default_value = 0.3, 3.0
    nt.links.new(rest.outputs['Vector'], dye.inputs['Vector'])
    # the seams, the hem's turn and the waistband
    seam = attr('seam')
    sr = P['seam_ridge']
    ridge = math_('EXPONENT', math_('MULTIPLY', math_('POWER', math_('DIVIDE', seam, sr), 2.0), -1.0))
    groove = math_('EXPONENT', math_('MULTIPLY', math_('POWER', math_('DIVIDE', seam, 0.22 * sr), 2.0), -1.0))
    hem_d, top_d = attr('hem_d'), attr('top_d')
    band = float(MEASURES.get('band', 1.0))
    turn = line(hem_d, band, 0.05)                       # (the hem's turned edge, stitched down)
    roll = line(hem_d, 0.45 * band, 0.5 * band)          # (the soft roll of the doubled cloth)
    stitch = math_('MULTIPLY', turn, math_('GREATER_THAN', math_('FRACT', math_('MULTIPLY', u, 3.2)), 0.35))
    wb = line(top_d, 0.55 * float(MEASURES.get('H', 9.4)) * 0.075, 0.05)
    # the colour
    shade = math_('MULTIPLY', rng(dye.outputs['Fac'], 0.3, 0.7, 1.0 - P['mottle'], 1.0 + P['mottle']),
                  rng(weave, 0.0, 1.0, 0.9, 1.05))
    shade = math_('MULTIPLY', shade, rng(nap.outputs['Fac'], 0.3, 0.7, 0.85, 1.12))
    shade = math_('MULTIPLY', shade, math_('SUBTRACT', 1.0, math_('MULTIPLY', groove, 0.5)))
    shade = math_('MULTIPLY', shade, math_('SUBTRACT', 1.0, math_('MULTIPLY', stitch, 0.4)))
    shade = math_('MULTIPLY', shade, math_('SUBTRACT', 1.0, math_('MULTIPLY', wb, 0.3)))
    ao = N('ShaderNodeAmbientOcclusion')
    ao.inputs['Distance'].default_value = 1.2
    ao.samples = 8
    shade = math_('MULTIPLY', shade, rng(ao.outputs['AO'], 0.0, 1.0, 0.5, 1.0))
    col = N('ShaderNodeVectorMath')
    col.operation = 'SCALE'
    col.inputs[0].default_value = P['colour']
    nt.links.new(shade, col.inputs['Scale'])
    # the relief: the weave under the nap, the seams' ridge, the hem's roll and stitches
    h = math_('ADD', math_('MULTIPLY', weave, 0.5), math_('MULTIPLY', nap.outputs['Fac'], 1.0))
    bump = N('ShaderNodeBump')
    bump.inputs['Strength'].default_value, bump.inputs['Distance'].default_value = P['relief'], 0.4 * P['pitch']
    nt.links.new(h, bump.inputs['Height'])
    big = math_('SUBTRACT', math_('ADD', ridge, math_('MULTIPLY', roll, 1.4)),
                math_('ADD', math_('MULTIPLY', groove, 0.5), math_('MULTIPLY', stitch, 0.5)))
    bump2 = N('ShaderNodeBump')
    bump2.inputs['Strength'].default_value, bump2.inputs['Distance'].default_value = 0.6, 0.05
    nt.links.new(big, bump2.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], bump2.inputs['Normal'])
    pb = N('ShaderNodeBsdfPrincipled')
    nt.links.new(col.outputs['Vector'], pb.inputs['Base Color'])
    nt.links.new(bump2.outputs['Normal'], pb.inputs['Normal'])
    pb.inputs['Roughness'].default_value = P['rough']
    pb.inputs['Specular IOR Level'].default_value = 0.2
    pb.inputs['Sheen Weight'].default_value = P['sheen']
    pb.inputs['Sheen Roughness'].default_value = 0.5
    pb.inputs['Sheen Tint'].default_value = (0.55, 0.65, 1.0, 1.0)
    pb.name = 'close bsdf'
    nt.links.new(pb.outputs['BSDF'], out.inputs['Surface'])


def skirt_weights(M, body=None, near=(0.05, 1.3)):
    """The rig weights of a point of the skirt. The pelvis carries the top; below the hips each side's thigh takes a
    share rising to 0.45 at the knee and easing back at the hem, blended across the middle so the centre front and
    back are carried half by each leg (the back carries less: a thigh swinging forward would tip the back of the
    skirt forward with it), as the gown's skirt is carried. And with the body (its rest shape), cloth near a leg
    follows that leg's bones, the more the nearer it lies to the skin (full within near[0] H of it, none beyond
    near[1] H): a thigh swung forward lifts the cloth it would have gone through with it, and the far flare, which
    the legs never reach, stays on the pelvis."""
    from mathutils.bvhtree import BVHTree
    H = M['H']

    def ss(a, b, x):
        t = min(1.0, max(0.0, (x - a) / (b - a)))
        return t * t * (3 - 2 * t)
    tree = vw = None
    if body is not None:
        me = body.data
        pts = [body.matrix_world @ v.co for v in me.vertices]
        tree = BVHTree.FromPolygons(pts, [tuple(p.vertices) for p in me.polygons])
        names = {g.index: g.name for g in body.vertex_groups}
        vw = [{names[g.group]: g.weight for g in v.groups if names[g.group].split('.')[0] in ('Thigh', 'Shin', 'Foot')}
              for v in me.vertices]
        faces = [tuple(p.vertices) for p in me.polygons]

    def fn(co):
        w = female.trunk_weights(co)
        f = 0.45 * ss(M['hip_z'] + 0.1 * H, M['knee_z'], co.z) - 0.15 * ss(M['knee_z'], M['floor_z'], co.z)
        f *= 0.3 + 0.7 * ss(0.2 * H, -0.2 * H, co.y)
        r = 1.0 / (1.0 + math.exp(max(-40.0, min(40.0, co.x / (0.32 * H)))))      # her right is -X
        out = {k: v * (1.0 - f) for k, v in w.items()}
        out['Thigh.R'] = out.get('Thigh.R', 0.0) + f * r
        out['Thigh.L'] = out.get('Thigh.L', 0.0) + f * (1.0 - r)
        if tree is not None and co.z < M['hip_z'] + 0.3 * H:
            loc, nrm, fi, d = tree.find_nearest(co)
            if loc is not None:
                k = 1.0 - ss(near[0] * H, near[1] * H, d)
                if k > 1e-3:
                    vi = min(faces[fi], key=lambda i: (pts[i] - loc).length_squared)
                    legs = vw[vi]
                    tot = sum(legs.values())
                    if tot > 1e-4:
                        # (k of the point follows the leg's own bones as the skin does, the rest keeps its weights)
                        kk = k * min(1.0, tot)
                        out = {b: v * (1.0 - kk) for b, v in out.items()}
                        for b, v in legs.items():
                            out[b] = out.get(b, 0.0) + kk * v / tot
        return out
    return fn


def leg_proxy(body, rig, M):
    """The body's own legs (and hands) as a light collider, hidden: the skin's points that the leg and hand bones
    carry, thinned out, deformed by the same rig. It fits whatever body it is made from."""
    me = body.data.copy()
    pr = bpy.data.objects.new('Skirt push proxy', me)
    bpy.context.scene.collection.objects.link(pr)
    for m in list(pr.modifiers):
        pr.modifiers.remove(m)
    names = [g.name for g in pr.vertex_groups]
    low = [i for i, n in enumerate(names) if n.split('.')[0] in ('Thigh', 'Shin', 'Foot', 'Forearm', 'Hand')]
    keep = pr.vertex_groups.new(name='low')
    for v in me.vertices:
        w = sum(g.weight for g in v.groups if g.group in low)
        if w > 0.01:
            keep.add([v.index], min(1.0, w), 'REPLACE')
    mk = pr.modifiers.new('keep', 'MASK')
    mk.vertex_group = 'low'
    mk.threshold = 0.5
    dc = pr.modifiers.new('thin', 'DECIMATE')
    dc.ratio = 0.14
    am = pr.modifiers.new('rig', 'ARMATURE')
    am.object = rig
    pr.hide_render = True
    pr.hide_viewport = False
    pr.visible_camera = pr.visible_shadow = pr.visible_diffuse = pr.visible_glossy = False
    return pr


def add_push(sk, body, M):
    """Shrinkwrap 'Outside' against the body's own skin after the rig, so a thigh swung forward or a foot lifted behind
    does not come through the cloth: wide (0.06 H) over the lower half of the skirt, which has room to stand off the
    legs, and close (0.012 H) over the hips and the waist, where the skirt is fitted. Then the floor, then a
    corrective smooth. (These are Blender modifiers: they shape the renders and the sheets; the game plays the rig's
    weights alone, which is why those weights follow the legs, see skirt_weights.)"""
    H = M['H']

    def ss(a, b, x):
        t = min(1.0, max(0.0, (x - a) / (b - a)))
        return t * t * (3 - 2 * t)
    vg = sk.vertex_groups.new(name='push')
    hp = sk.vertex_groups.new(name='push hips')
    z0, z1 = M['crotch_z'] + 0.1 * H, M['crotch_z'] - 0.2 * H      # (the wide push starts below the crotch)
    for v in sk.data.vertices:
        w = ss(z0, z1, v.co.z)
        if w > 1e-3:
            vg.add([v.index], w, 'REPLACE')
        w = 1.0 - ss(z0, z1, v.co.z)
        if w > 1e-3:
            hp.add([v.index], w, 'REPLACE')
    def wrap(name, group, offset):
        m = sk.modifiers.new(name, 'SHRINKWRAP')
        m.target = body
        m.wrap_method = 'NEAREST_SURFACEPOINT'
        m.wrap_mode = 'OUTSIDE'
        m.offset = offset
        m.vertex_group = group
    # (pushed, smoothed, and pushed again: a push on the points that are inside alone leaves a step between them and
    # their neighbours; smoothing evens the step out and the second push keeps the cloth off the skin)
    sg = sk.vertex_groups.new(name='push smooth')
    for v in sk.data.vertices:
        sg.add([v.index], 1.0 - ss(M['hip_z'] + 0.2 * H, M['hip_z'] + 0.05 * H, v.co.z), 'REPLACE')
    wrap('push', 'push', 0.06 * H)
    wrap('push hips', 'push hips', 0.012 * H)
    sm = sk.modifiers.new('push smooth', 'SMOOTH')
    sm.factor, sm.iterations, sm.vertex_group = 0.6, 5, 'push smooth'
    wrap('push again', 'push', 0.06 * H)
    wrap('push hips again', 'push hips', 0.012 * H)
    # the floor: the hem is carried by the pelvis and the thighs, and as the body dips through a step it would sink
    # into the ground; a slab under the soles keeps the cloth on it
    fl = bpy.data.objects.get('Skirt floor')
    if fl is None:
        me = bpy.data.meshes.new('Skirt floor')
        r = 12.0 * H
        z0 = M['floor_z']
        me.from_pydata([(x, y, z) for z in (z0 - 60.0, z0) for x in (-r, r) for y in (-r, r)], [],
                       [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
        me.update()
        fl = bpy.data.objects.new('Skirt floor', me)
        bpy.context.scene.collection.objects.link(fl)
        fl.hide_render = True
        fl.visible_camera = fl.visible_shadow = fl.visible_diffuse = fl.visible_glossy = False
    m2 = sk.modifiers.new('floor', 'SHRINKWRAP')
    m2.target = fl
    m2.wrap_method = 'NEAREST_SURFACEPOINT'
    m2.wrap_mode = 'OUTSIDE'
    m2.offset = 0.02
    m2.vertex_group = 'push'


def smooth_weights(sk, fn, rows, cols, rounds=8):
    """Give the skirt its rig weights from fn, evened out over the cloth: the weights that follow the nearest bit of
    skin jump from one point of the skin to the next, and a jump in the weights is a step in the cloth when the legs
    move. (The points run row by row down the cloth and wrap round it: the sheet's own grid.)"""
    me = sk.data
    pts = [v.co for v in me.vertices]
    ws = [fn(p) for p in pts]
    bones = sorted({b for w in ws for b in w})
    W = np.zeros((len(pts), len(bones)))
    for i, w in enumerate(ws):
        for b, v in w.items():
            W[i, bones.index(b)] = v
    G = W.reshape(rows, cols, -1)
    for _ in range(rounds):
        G = 0.5 * G + 0.125 * (np.roll(G, 1, 1) + np.roll(G, -1, 1)
                               + np.concatenate([G[:1], G[:-1]]) + np.concatenate([G[1:], G[-1:]]))
    W = G.reshape(len(pts), -1)
    groups = {b: sk.vertex_groups.new(name=b) for b in bones}
    for i in range(len(pts)):
        for k, b in enumerate(bones):
            if W[i, k] > 1e-4:
                groups[b].add([i], float(W[i, k]), 'REPLACE')


def make_skirt(body, rig, M, name='Riding skirt', under=None):
    """Draft the skirt on the body from its measures M and hand it to the body's bones. Returns the object and the
    draft (its sizes, for the thickness)."""
    B = draft.body_data(body)
    D = draft.draft(B, M, under=under)
    bm = draft.sheet_mesh(D)
    sk = female.mesh_obj(name, bm, M_WOOL, skin=lambda co: {}, subdiv=0, solid=0.0)
    smooth_weights(sk, skirt_weights(M, body), D['rows'], D['cols'])
    MEASURES['band'] = D['r']['hem_band']
    MEASURES['H'] = M['H']
    return sk, D


def finish_skirt(sk, D):
    """The wool's thickness (doubled in the turned hem: a vertex group weighs it down the cloth), then the surface's
    smoothing: after the rig, the push and the settle."""
    hem = sk.vertex_groups.new(name='hem')
    band = D['r']['hem_band']
    hd = np.empty(len(sk.data.vertices))
    sk.data.attributes['hem_d'].data.foreach_get('value', hd)
    for i, d in enumerate(hd):
        hem.add([i], 0.5 + 0.5 * float(np.exp(-(d / (0.85 * band)) ** 4)), 'REPLACE')
    sol = sk.modifiers.new('thickness', 'SOLIDIFY')
    sol.thickness = 2.0 * D['r']['wool']
    sol.offset = 1.0
    sol.vertex_group = 'hem'
    sol.thickness_vertex_group = 0.0
    sm = sk.modifiers.new('smooth', 'SUBSURF')
    sm.levels = sm.render_levels = 1


def build():
    res = SP.build()
    body = bpy.data.objects['Skin body']
    rig = bpy.data.objects.get('Rig')
    for o in bpy.data.objects:                   # (the necklace's parts: NO_ACC hides only the chain)
        if o.name.startswith('Necklace'):
            o.hide_render = True
            o.visible_camera = o.visible_shadow = o.visible_diffuse = o.visible_glossy = False
    bpy.context.view_layer.update()
    H = SP.anatomy.head_height() or SP.anatomy.HEAD
    B = draft.body_data(body)
    M = draft.measure(B, rig, H)
    draft.report(M)
    MEASURES.update(M)
    sk, D = make_skirt(body, rig, M)
    DRAFT.clear()
    DRAFT.update(D=D, M=M)
    proxy = leg_proxy(body, rig, M)
    if PUSH:
        add_push(sk, body, M)
    if SETTLE:
        hands = [o for o in (bpy.data.objects.get('Hand.L'), bpy.data.objects.get('Hand.R')) if o]
        P = dict(settle.WOOL)
        P['pin_to'], P['pin_from'] = M['hips_z'], M['hips_z'] - 0.2 * H
        bpy.context.scene.frame_set(1)
        settle.settle(sk, rig, [proxy] + hands, M['floor_z'], P)
    finish_skirt(sk, D)
    wool_closeup('skirt wool')
    floor = M['floor_z']
    frames = {'skirt': (floor - 0.5, M['waist_z'] + 1.5), 'waist': (M['waist_z'] - 4.0, M['waist_z'] + 2.5),
              'hem': (floor - 0.4, floor + 8.0), 'train': (floor - 0.4, floor + 12.0)}
    lo, hi = frames[FRAMES[VIEW] if isinstance(VIEW, int) else VIEW]
    os.environ.setdefault('HEAD_CZ', '%.2f' % (0.5 * (lo + hi)))
    os.environ.setdefault('HEAD_H', '%.2f' % (hi - lo))
    bpy.context.view_layer.update()
    return res

