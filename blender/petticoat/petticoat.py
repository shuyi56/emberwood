"""A standalone garment: the petticoat under the riding habit (about 1885-1895), over the corset on the bare mannequin.

Drafted flat first (the project's rule): the body is drawn in and corseted by the corset thread's habit.py, the
petticoat is measured over the corset (draft.py), and its pieces are drafted flat from those measures with every seam
edge paired (pattern.py). Then the pieces are cut as cloth, laid round her, sewn along their paired edges by the
solver's sewing threads with the band held on the corset, and draped (sew.py, cloth.py, the cotton tuned on
swatch.py's bench); the sewn points are joined into one cloth.

    PETTI_OUT=DIR/pattern python3 blender/base/views.py --figure petticoat --mannequin --list

writes the laid-out pattern (DIR/pattern.png) and its pieces and seams (DIR/pattern.json) after the build."""
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
import pattern  # noqa: E402
import habit  # noqa: E402
import sew  # noqa: E402
import cloth  # noqa: E402

SEW = True                   # cut, sew and drape the pieces (False: draft the flat pattern only)
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


def make_object(V, F, UV, wts, rig, name='Petticoat', mat=None):
    """The cloth as a mesh skinned to rig: its flat pattern coordinates as the 'pattern' UVs (one pair per face corner,
    so a seam keeps each side's own), and a 'rest' copy of its points."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(p) for p in V], [], [tuple(int(i) for i in f) for f in F])
    for p in me.polygons:
        p.use_smooth = True
    uv = me.uv_layers.new(name='pattern')
    uv.data.foreach_set('uv', np.asarray(UV, float).reshape(-1))
    rest = me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
    rest.data.foreach_set('vector', np.asarray(V, float).reshape(-1))
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    me.materials.append(mat or female.mat(M_COTTON))
    for bone, w in wts.items():
        g = o.vertex_groups.new(name=bone)
        for i in np.nonzero(w > 1e-4)[0]:
            g.add([int(i)], float(w[i]), 'REPLACE')
    m = o.modifiers.new('rig', 'ARMATURE')
    m.object = rig
    return o


def finish(o, H):
    m = o.modifiers.new('thickness', 'SOLIDIFY')
    m.thickness = draft.RATIOS['thickness'] * H
    m.offset = -1.0
    m = o.modifiers.new('smooth', 'SUBSURF')
    m.levels = m.render_levels = 1


def body_surface(objs):
    """The points and outward normals of what the petticoat lies on, posed as at this frame."""
    dg = bpy.context.evaluated_depsgraph_get()
    pts, nrm = [], []
    for o in objs:
        e = o.evaluated_get(dg)
        me = e.to_mesh()
        n = len(me.vertices)
        co = np.empty(n * 3)
        me.vertices.foreach_get('co', co)
        no = np.empty(n * 3)
        me.vertices.foreach_get('normal', no)
        mw = np.array(o.matrix_world)
        pts.append(co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3])
        nrm.append(no.reshape(-1, 3) @ np.linalg.inv(mw[:3, :3]))
        e.to_mesh_clear()
    return np.concatenate(pts), np.concatenate(nrm)


def _load_points(path):
    d = np.load(path)
    return d if isinstance(d, np.ndarray) else d['X']


def sew_and_drape(body, rig, M, pat, pieces, m, under_objs=(), log=print):
    """Cut the flat pieces, lay them round her, sew them and let the skirt fall in the local solver (the band held
    on the corset); the sewn points are then joined; then the flounce set on it the same way. Returns the skirt's
    object and the flounce's."""
    cuts = sew.cut(pieces)
    placed = sew.lay(cuts, pieces, m, M, pat, log=log)
    bp, ba = sew.band_ring(cuts[('waistband', 'C')], m, M, pat)
    A = sew.assemble(cuts, placed, bp, ba, m, M)
    P, T, S = A['P'], A['T'], A['sew']
    gap0 = np.linalg.norm(P[S[:, 0]] - P[S[:, 1]], axis=1)
    log('petticoat sew: %d points, %d faces, %d sewn pairs (apart %.2f on average, %.2f at most before sewing)' % (
        len(P), len(T), len(S), float(gap0.mean()), float(gap0.max())))
    bpy.context.scene.frame_set(1)
    hands = [bpy.data.objects[n] for n in ('Hand.L', 'Hand.R') if n in bpy.data.objects]
    pts, nrm = body_surface([body] + hands + [u for u in under_objs if u.type == 'MESH'])
    if os.environ.get('PETTI_COLLIDER'):
        np.savez(os.environ['PETTI_COLLIDER'], pts=pts, nrm=nrm, floor=M['floor'])
    if os.environ.get('PETTI_SKIRT_IN'):      # (the layer below as it settled before: not run again)
        X, weld = _load_points(os.environ['PETTI_SKIRT_IN']), sew.gathers(A)[2]
        log('petticoat sew: the skirt as settled before, from %s' % os.environ['PETTI_SKIRT_IN'])
    else:
        X, weld = sew.drape(A, cuts, pts, nrm, M['floor'], log=log)
    # the flounce: the layer above, set on the settled skirt (which it does not move)
    Y, FT, FUV, FR = sew.flounce(A, cuts, X, pieces, M, pts, nrm, M['floor'], log=log)
    # (the drape is found on the body as it stands at frame 1; carried back to the rest shape through the bones)
    o = make_object(X, T, A['UV'][T], sew.weights(X, M), rig, name='Petticoat sewing')
    B = cloth.bone_matrices(o, rig)
    Bi = np.linalg.inv(B)
    X = np.einsum('nij,nj->ni', Bi[:, :3, :3], X) + Bi[:, :3, 3]
    me_old = o.data
    bpy.data.objects.remove(o)
    bpy.data.meshes.remove(me_old)
    Q, F, new = sew.weld(X, T, weld)
    ok = (new[T][:, 0] != new[T][:, 1]) & (new[T][:, 1] != new[T][:, 2]) & (new[T][:, 0] != new[T][:, 2])
    o = make_object(Q, F, A['UV'][T][ok], sew.weights(Q, M), rig)
    log('petticoat sew: joined into one cloth of %d points, %d faces' % (len(Q), len(F)))
    fo = make_object(Y, FT, FUV[FT], sew.weights(Y, M), rig, name='Petticoat flounce sewing')
    B = cloth.bone_matrices(fo, rig)
    Bi = np.linalg.inv(B)
    Y = np.einsum('nij,nj->ni', Bi[:, :3, :3], Y) + Bi[:, :3, 3]
    me_old = fo.data
    bpy.data.objects.remove(fo)
    bpy.data.meshes.remove(me_old)
    Q, FF, new = sew.weld(Y, FT, FR['close'])
    ok = (new[FT][:, 0] != new[FT][:, 1]) & (new[FT][:, 1] != new[FT][:, 2]) & (new[FT][:, 0] != new[FT][:, 2])
    fo = make_object(Q, FF, FUV[FT][ok], sew.weights(Q, M), rig, name='Petticoat flounce')
    log('petticoat sew: the flounce, %d points, %d faces' % (len(Q), len(FF)))
    return o, fo


def draft_on(body, H, under_objs=(), extra=None, log=print):
    """The petticoat's flat pattern for a body that is already corseted (habit.compress_body) with the corset on it:
    under_objs are what it goes over (the corset's meshes). Returns (pieces, seams, measures). The pieces are not sewn
    or draped yet: that waits for John's go (the project's rule: draft flat first)."""
    P, W, names = body_arrays(body)
    if extra is None:
        extra = under(under_objs, H)
    M = draft.measure(P, W, names, H, extra=extra)
    draft.report(M, log)
    pat = draft.pattern(M, log=log)
    pieces, seams, m = pattern.draft_pieces(M, pat, log=log)
    pattern.report(pieces, seams, m, log)
    m['_M'], m['_pat'] = M, pat
    return pieces, seams, m


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
    extra = None if CORSET else corset_surface(HM, H)
    pieces, seams, m = draft_on(body, H, worn, extra=extra)
    M, pat = m.pop('_M'), m.pop('_pat')
    MEASURES.update({'band': m['band'], 'H': m['H']})
    out = os.environ.get('PETTI_OUT')
    if out:
        pattern.layout(pieces, seams, m, out + '.png')
        pattern.save(pieces, seams, m, out + '.json')
    if SEW and not os.environ.get('PETTI_FLAT'):
        for o in sew_and_drape(body, rig, M, pat, pieces, m, worn):
            finish(o, H)
    bpy.context.view_layer.update()
    return res
