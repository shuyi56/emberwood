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
TOE_EXT = 0.10               # an almond toe: drawn this far past the toes, x the foot's length (the toe allowance;
                             # about 2.4 cm, a fashion last's. EXPERIMENTAL against the photo's slim toe)
TOE_NARROW = 0.5             # and narrowed toward its point by this share at the tip (read off the photo's plan. EXPERIMENTAL)
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


def build():
    heelboot.STYLE, heelboot.HEEL_H = STYLE, HEEL_H
    boot.SHAFT = 'zip'
    boot.last.RATIOS['top_over_welt'] = boot.last.ZIP['top_over_welt']
    boot.last.RATIOS['drape'] = DRAPE
    boot.last.RATIOS['toe_shell'] = TOE_SHELL
    boot.last.RATIOS['toe_ext'], boot.last.RATIOS['toe_narrow'] = TOE_EXT, TOE_NARROW
    boot.CREASE = CREASE
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
        zipper.build(o, boot.last.PIECES[side], rig, L,
                     {'teeth': female.mat(M_TEETH), 'tape': female.mat(M_TAPE), 'metal': female.mat(M_METAL)})
    if ENTRY_CHECK:
        entry(boot.last.PIECES, boot.hose.MEASURES)
    root.location.z = up
    bpy.context.view_layer.update()
    if DESIGN:
        design(boot.last.PIECES)
    return res
