"""The zip ankle boot: a sock-fit ankle boot with an inside zip, on the heeled boot's foot, last, sole and heel
(blender/heelboot: the optimized footbed and the stacked heel joined to the sole, in heelboot.STYLE).

The upper is last.cut_zip's (SHAFTS['zip']): one piece from the inner side's zip round the toe and the front to a
back seam, an inner quarter from the back seam to the zip, and a zip guard behind the zip; sewn by sew.sew (the
local fabric solver) and then the zip built as parts along the closed seam (zipper.py). The style and sizes are read
off John's reference photo with tools/refmeasure (see last.ZIP).

    python3 blender/base/views.py --figure zipboot --mannequin --turn --clay --angles front,profile,back3/4 --out DIR
    python3 blender/base/views.py --figure zipboot --mannequin --turn --set DESIGN=True --out DIR     (colour-coded)

DESIGN paints each design part (each pattern piece, the zip's tapes, teeth, slider and pull, the sole, the heel's
lifts) one flat distinct colour, for the colour-coded draft. ENTRY_CHECK prints whether the foot can enter the
boot with the zip open (entry())."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'heelboot'))
import heelboot  # noqa: E402   (first: it sets the environment the figure is built under)
import bpy  # noqa: E402
import numpy as np  # noqa: E402
import boot  # noqa: E402
import female  # noqa: E402
import zipper  # noqa: E402

STYLE = 'waisted'            # the heel's style (heelboot.STYLES): John's pick
HEEL_H = 3.0                 # cm, the heel's height at the seat point (John's pick; None: the optimizer's)
DESIGN = False               # the colour-coded draft
ENTRY_CHECK = True
DRAPE = 64                   # passes drawing the last taut over the foot and leg (last.build_boots: a sock fit)
TOE_SHELL = 150              # passes turning the toe as one smooth box over the toes (last.build_boots)
TOE_EXT = 0.12               # an almond toe: drawn this far past the toes, x the foot's length (the toe allowance;
                             # about 2.4 cm, a fashion last's. EXPERIMENTAL against the photo's slim toe)
TOE_NARROW = 0.30            # and narrowed toward its point by this share at the tip (read off the photo's plan. EXPERIMENTAL)
ROOM = {'room': 0.034, 'gap': 0.034}   # x H: the foot's room and the least room off the skin (the boot's own 0.022
                             # and 0.024 drew the suede onto the foot until the toes and the instep showed: John, too tight)
PITCH_SET = 40               # passes setting the leather at the heel's pitch (set_at_pitch; 0 for none)
THROAT_LINE = 0.45           # the front line from the vamp to the top: the straight chord (Golding) bowed into the
                             # leg-and-foot corner by this share of the corner's depth (last.build_boots). EXPERIMENTAL
THROAT_FROM = 0.30           # the vamp point, where that curve leaves the foot: this share of the length behind the toe
CREASE = False               # the ankle's bend set into the leather as folds (boot.CREASE): off for a first look

M_TEETH = female.mat_def(91, 'zip coil', female.ramp(0x2a211c, 0.08), 0.35)
M_TAPE = female.mat_def(92, 'zip tape', female.ramp(0x31261f, 0.08), 0.1)
M_METAL = female.mat_def(93, 'zip metal', female.ramp(0xb8b6b0, 0.2), 0.6)

# the colour-coded draft: one flat colour for each design part (pieces and parts, not materials)
COLOURS = {
    'vamp and outer quarter': (0.85, 0.55, 0.20), 'inner quarter': (0.25, 0.55, 0.85), 'zip guard': (0.95, 0.85, 0.20),
    'Zip teeth': (0.80, 0.15, 0.55), 'Zip tape': (0.55, 0.30, 0.70), 'Zip slider': (0.85, 0.85, 0.85),
    'Boot stitches': (0.95, 0.95, 0.60), 'Boot sole': (0.35, 0.25, 0.20), 'Heel lift': (0.25, 0.65, 0.35),
}


def _flat(name, rgb):
    m = bpy.data.materials.get('design ' + name) or bpy.data.materials.new('design ' + name)
    m.use_nodes = True
    b = m.node_tree.nodes.get('Principled BSDF')
    b.inputs['Base Color'].default_value = (*rgb, 1.0)
    b.inputs['Roughness'].default_value = 0.7
    return m


def design(pieces):
    """Each design part one flat colour: the boot's faces by the piece they were cut in, the rest by object."""
    for side in 'RL':
        o = bpy.data.objects.get('Boot.' + side)
        if o is None:
            continue
        me = o.data
        me.materials.clear()
        names = [p['name'] for p in pieces[side] if p['name'] != 'sole']
        for n in names:
            me.materials.append(_flat(n, COLOURS.get(n, (0.6, 0.6, 0.6))))
        bounds = [(p['start'], p['start'] + len(p['V'])) for p in pieces[side] if p['name'] != 'sole']
        idx = np.zeros(len(me.polygons), dtype=np.int32)
        for f in me.polygons:
            v = f.vertices[0]
            idx[f.index] = next(i for i, (a, b) in enumerate(bounds) if a <= v < b)
        me.polygons.foreach_set('material_index', idx)
        me.update()
    for o in bpy.data.objects:
        if o.type != 'MESH':
            continue
        for k, rgb in COLOURS.items():
            if o.name.startswith(k) and not o.name.startswith('Boot.'):
                o.data.materials.clear()
                o.data.materials.append(_flat(k, rgb))


def _hull(q):
    """The convex hull of 2D points, in order (Andrew's monotone chain)."""
    pts = sorted(map(tuple, np.round(q, 6)))
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])  # noqa: E731
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return np.array(lo[:-1] + hi[:-1])


def design_heel():
    """The heel's lifts in their design colour (after heelboot.restyle)."""
    for o in bpy.data.objects:
        if o.type == 'MESH' and o.name.startswith('Heel lift'):
            o.data.materials.clear()
            o.data.materials.append(_flat('Heel lift', COLOURS['Heel lift']))


def entry(pieces, M, log=print):
    """Can the foot go in with the zip open? The foot enters heel first through the opening, so the opening has to
    pass the foot's long heel girth (Golding, The Manufacture of Boots and Shoes: from the top of the instep round
    the seat of the last; the measure a pull-on boot's leg must pass). Opened, the zip's two edges part into a V
    from the stop: the opening's girth is the topline's round plus the two edges' run beside the opened V, so the
    least of (topline round + 2 x zip length) must reach the long heel girth with room to slide (EXPERIMENTAL: the
    V counted as fully spread; no source gives the share of the zip's length a foot uses). Closed, the topline
    should stand near the ankle's own girth (the sock fit). Prints both."""
    import pattern
    by = {p['name']: p for p in pieces['R']}
    zi = by['inner quarter']['zip']
    tops = pattern.pairs([p for p in pieces['R'] if p['name'] != 'sole'], log=lambda *a: None).get('top', {})
    girth = sum(tops.values()) + 2 * zi['gap'] * 165 / 63          # (the topline's pieces and the zip between them)
    zl = (zi['top'] - zi['stop']) * 165 / 63
    body = bpy.data.objects['Skin body']
    P = np.array([v.co[:] for v in body.data.vertices])
    foot = P[(P[:, 0] < 0) & (P[:, 2] < M['ankle_z'] + 0.5)]
    # the long heel girth: the foot's section through the instep's top and the back of the heel (a plane through
    # both, square to the foot's side): its taut round
    y_in = M['toe_y'] + 0.55 * (M['heel_y'] - M['toe_y'])      # (the instep's top: Golding's instep point)
    near = foot[np.abs(foot[:, 1] - y_in) < 0.15]
    inst = near[np.argmax(near[:, 2])]
    back = foot[np.argmax(foot[:, 1] - 0.3 * foot[:, 2])]
    d = back - inst
    d /= np.linalg.norm(d)
    nrm = np.cross(d, np.array([1.0, 0.0, 0.0]))
    nrm /= np.linalg.norm(nrm)
    sec = foot[np.abs((foot - inst) @ nrm) < 0.06]
    u = np.cross(nrm, np.array([1.0, 0.0, 0.0]))
    q = np.column_stack([(sec - inst) @ np.array([1.0, 0.0, 0.0]), (sec - inst) @ u])
    h = _hull(q)
    lh = float(np.linalg.norm(np.diff(np.vstack([h, h[:1]]), axis=0), axis=1).sum()) * 165 / 63
    ankle = M['ankle'] * 165 / 63
    log('zipboot entry: the long heel girth %.1f cm; the topline %.1f cm round closed (the ankle %.1f cm: x%.2f); '
        'opened, the topline and the zip\'s two edges (%.1f cm each) give %.1f cm: %s'
        % (lh, girth, ankle, girth / ankle, zl, girth + 2 * zl,
           'the foot goes in' if girth + 2 * zl > 1.05 * lh else 'TOO TIGHT for the foot'), flush=True)
    return dict(long_heel_cm=lh, top_cm=girth, ankle_cm=ankle, zip_cm=zl, open_cm=girth + 2 * zl)


def _posed(o):
    """The boot's points as the rig bends them (the leather's thickness left off)."""
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    e = o.evaluated_get(dg)
    me = e.to_mesh()
    P = np.array([(o.matrix_world @ v.co)[:] for v in me.vertices])
    e.to_mesh_clear()
    return P


def set_at_pitch(o, M, passes, log=print):
    """Set the boot's leather at the heel's pitch, as a boot is lasted: on a last that already stands at the heel's
    height (Golding, The Manufacture of Boots and Shoes: the last carries the pitch, and the upper is drawn smooth
    over it). This boot is cut and sewn on the flat foot and the rig then pitches the foot onto the heel, which
    pinches the leather behind the ankle and folds it in front. So the bent boot is smoothed where the ankle bends
    (Taubin steps, which do not shrink it) and each point carried back through the rig's own bend at that point (its
    Jacobian, found by moving the point a little each way), so the boot at rest is the shape that bends INTO the
    smooth one. The stitches and whatever else rides on the boot take the nearest point's move."""
    solid = [m for m in o.modifiers if m.type == 'SOLIDIFY' and m.show_viewport]
    for m in solid:
        m.show_viewport = False
    me = o.data
    n = len(me.vertices)
    R = np.array([v.co[:] for v in me.vertices])

    def put(X):
        me.vertices.foreach_set('co', X.ravel())
        me.update()
        bpy.context.view_layer.update()
    P = _posed(o)
    eps = 0.01
    J = np.zeros((n, 3, 3))
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        put(R + d)
        J[:, :, k] = (_posed(o) - P) / eps
    put(R)
    ed = np.array([e.vertices[:] for e in me.edges], dtype=np.int64)
    deg = np.bincount(ed.ravel(), minlength=n).astype(float)

    def avg(V):
        return np.stack([np.bincount(ed[:, 0], weights=V[ed[:, 1], k], minlength=n)
                         + np.bincount(ed[:, 1], weights=V[ed[:, 0], k], minlength=n) for k in range(3)], 1) \
            / np.maximum(deg, 1)[:, None]
    az = M['ankle_z']
    z = R[:, 2]
    s01 = lambda a, b, x: np.clip((x - a) / (b - a), 0, 1) ** 2 * (3 - 2 * np.clip((x - a) / (b - a), 0, 1))  # noqa: E731
    w = s01(az - 2.6, az - 1.6, z) * (1.0 - s01(az + 1.4, az + 2.2, z)) * (deg > 2)   # (the ankle's band; edges held)
    Q = P.copy()
    for _ in range(int(passes)):
        Q += (0.5 * w)[:, None] * (avg(Q) - Q)
        Q += (-0.53 * w)[:, None] * (avg(Q) - Q)
    dR = np.linalg.solve(J, (Q - P)[:, :, None])[:, :, 0]
    put(R + dR)
    for m in solid:
        m.show_viewport = True
    side = o.name.rsplit('.', 1)[-1]
    from mathutils.kdtree import KDTree
    kd = KDTree(n)
    for i, p_ in enumerate(R):
        kd.insert(p_, i)
    kd.balance()
    for r_ in bpy.data.objects:
        if r_ is o or r_.type != 'MESH' or not r_.name.endswith('.' + side) or not r_.name.startswith('Boot stitches'):
            continue
        for v in r_.data.vertices:
            v.co = v.co + type(v.co)(dR[kd.find(v.co)[1]])
        r_.data.update()
    mv = np.linalg.norm(Q - P, axis=1)
    log('zipboot %s: set at the heel\'s pitch (%d passes): %d points moved, %.3f on average, %.3f at most'
        % (o.name, int(passes), int((mv > 0.005).sum()), float(mv[mv > 0.005].mean()) if (mv > 0.005).any() else 0.0,
           float(mv.max())), flush=True)


def build():
    heelboot.STYLE, heelboot.HEEL_H = STYLE, HEEL_H
    boot.SHAFT = 'zip'
    boot.last.RATIOS['top_over_welt'] = boot.last.ZIP['top_over_welt']
    boot.last.RATIOS['drape'] = DRAPE
    boot.last.RATIOS['toe_shell'] = TOE_SHELL
    boot.last.RATIOS['toe_ext'], boot.last.RATIOS['toe_narrow'] = TOE_EXT, TOE_NARROW
    boot.CREASE = CREASE
    boot.last.RATIOS.update(ROOM)
    boot.last.RATIOS['throat_line'], boot.last.RATIOS['throat_from'] = THROAT_LINE, THROAT_FROM
    heelboot.CUT = {}                          # (the tall boot's cut of the shaft does not apply)
    res = heelboot.build()
    rig = bpy.data.objects.get('Rig')
    L = boot.hose.MEASURES['heel_y'] - boot.hose.MEASURES['toe_y']
    root = female.FIG['root']
    up = root.location.z
    root.location.z = 0.0                      # (the zip is built on the boot as sewn, standing at rest)
    female.pose(0)
    bpy.context.view_layer.update()
    for side in 'RL':
        o = bpy.data.objects['Boot.' + side]
        if PITCH_SET:
            root.location.z = up               # (as it stands on the heel)
            bpy.context.view_layer.update()
            set_at_pitch(o, boot.hose.MEASURES, PITCH_SET)
            root.location.z = 0.0
            bpy.context.view_layer.update()
        zipper.build(o, boot.last.PIECES[side], rig, L,
                     {'teeth': female.mat(M_TEETH), 'tape': female.mat(M_TAPE), 'metal': female.mat(M_METAL)})
    if ENTRY_CHECK:
        entry(boot.last.PIECES, boot.hose.MEASURES)
    root.location.z = up
    bpy.context.view_layer.update()
    if DESIGN:
        design(boot.last.PIECES)
    return res
