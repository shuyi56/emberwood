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
TOE_BOX = 0.085              # a square toe box (last.build_boots): cut this far ahead of the toes, x the foot's length
TOE_SQ, TOE_FRONT, TOE_WALL = 2.2, 0.72, 0.40   # its sections' squareness (a superellipse's power: a nearly flat top pinched
                             # into the sides), its width at the cut as a share of the ball's, and the front wall's least
                             # height (John: sharp, not round; nearly flat top, pinched)
TOE_EASE = 8                 # passes easing the box after it is drawn (John: smooth, by the photo's lines)
TOE_SIDE = 0.42              # the toe's sides stand upright this high over the sole (about 1.1 cm), a smooth dome over them
TOE_BOW = 0.38               # the front edge bowed back at its corners by this share of the half width (John: round the tips' edges)
TOE_SWEEP = 1.3              # the profile's hollow: low over the toes, then up to the instep (1: a straight ramp)
TOE_CUT_R, TOE_RAKE = 0.20, 0.25   # the front edge's softness, and the front wall's lean back with height
TURNOUT_AT_HIP = True        # the stance's 7 degrees of turn-out made in the hip: at the ankle it twists the shaft on the foot
                             # (gauge.py: the two sides of the instep bent apart by up to 1.8 cm, the fold across the front view)
SHELL = 0.5                  # the ankle's bend averaged across the boot, inner side with outer (bend_at_ball; 0 for the hose's own)
BEND_FAN = 0.8               # and its band's half-width grown by this per unit of height over the sole
BEND = (2.7, 4.1)            # the boot's bend moved back to the ball: this far behind the tip, units (bend_at_ball)
PITCH_SET = 40               # passes setting the leather at the heel's pitch (set_at_pitch; 0 for none)
WELT = 0.006                 # x H: the sole trimmed close to the upper (about 1.5 mm), as a dress boot's is; the boot's
                             # 0.022 is a welted work boot's ledge (John: join the heel and fit)
SEAT_BAND, SEAT_OUT = 0.35, 0.02   # the heel's seat taken from the upper where it meets the sole (not the counter's
                             # fullest, 1.1 up), and trimmed this close to it: the heel runs on from the boot's back
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


def _carry(o, R, dR):
    """Whatever rides on the boot o (its stitches) moved as the boot's nearest point (at R, before) was (dR)."""
    from mathutils.kdtree import KDTree
    side = o.name.rsplit('.', 1)[-1]
    kd = KDTree(len(R))
    for i, p_ in enumerate(R):
        kd.insert(p_, i)
    kd.balance()
    for r_ in bpy.data.objects:
        if r_ is o or r_.type != 'MESH' or not r_.name.endswith('.' + side) or not r_.name.startswith('Boot stitches'):
            continue
        for v in r_.data.vertices:
            v.co = v.co + type(v.co)(dR[kd.find(v.co)[1]])
        r_.data.update()


def true_seam(o, pieces, reach=0.45, passes=40, log=print):
    """The back seam's lapped edge trued to a fair line. The edge is cut straight on the last, but sewn (each of its
    points tied to the nearest point of the quarter under it, which are not spaced alike) it comes out wavy, with a
    notch over the counter where the back turns from the heel's bulge into the hollow above it. A clicker's edge is
    a clean knife line, so the edge's points are drawn to their own smoothed run up the seam, and the leather
    within `reach` of the edge goes with it, less the further off."""
    by = {p['name']: p for p in pieces}
    fr = by['vamp and outer quarter']
    me = o.data
    R = np.array([v.co[:] for v in me.vertices])
    # the edge itself: the piece's open edges (each on one face only) that lie along the seam
    from collections import Counter
    cnt = Counter()
    lo, hi = fr['start'], fr['start'] + len(fr['V'])
    for f in me.polygons:
        vs = list(f.vertices)
        if lo <= vs[0] < hi:
            for k in range(len(vs)):
                a_, b_ = vs[k], vs[(k + 1) % len(vs)]
                cnt[(min(a_, b_), max(a_, b_))] += 1
    rim = np.unique(np.array([e for e, c in cnt.items() if c == 1]).ravel())
    seam = np.nonzero(fr['edges']['B'])[0] + fr['start']
    near = np.sqrt(((R[rim][:, None, :] - R[seam][None, :, :]) ** 2).sum(-1)).min(1) < 0.2
    idx = rim[near]
    idx = idx[np.argsort(R[idx, 2])]
    Q = R[idx].copy()
    for _ in range(passes):
        Q[1:-1, :2] = 0.25 * Q[:-2, :2] + 0.5 * Q[1:-1, :2] + 0.25 * Q[2:, :2]
    dE = Q - R[idx]
    dR = np.zeros_like(R)
    sl = np.arange(fr['start'], fr['start'] + len(fr['V']))
    d2 = ((R[sl][:, None, :] - R[idx][None, :, :]) ** 2).sum(-1)
    j = d2.argmin(1)
    dist = np.sqrt(d2[np.arange(len(sl)), j])
    t = np.clip(1.0 - dist / reach, 0.0, 1.0)
    dR[sl] = dE[j] * (t * t * (3 - 2 * t))[:, None]
    me.vertices.foreach_set('co', (R + dR).ravel())
    me.update()
    _carry(o, R, dR)
    log('zipboot %s: the back seam\'s edge trued: %d points on it moved %.3f on average, %.3f at most'
        % (o.name, len(idx), float(np.linalg.norm(dE, axis=1).mean()), float(np.linalg.norm(dE, axis=1).max())),
        flush=True)


def bend_at_ball(o, band, like=None, log=print):
    """The boot's bend moved back to the ball: the leather takes the hose's weights, so it bends where the toes do,
    across the middle of the toe box, and on the heel that shows as a line across the box's top. A shoe's forepart is
    stiff and flat to the tread line and bends there, under the vamp's rise. The toes' bone is given the box whole to
    band[0] behind the tip and none of it past band[1] (units), each point's other bones keeping their own
    proportions (a point that had none takes the foot's). The sole (like: the upper's points, weights and names)
    takes its weights from the upper's nearest point by the welt, so the two bend as one."""
    me = o.data
    side = o.name.rsplit('.', 1)[-1]
    if like is not None:
        for nm in like[2]:
            if o.vertex_groups.get(nm) is None:
                o.vertex_groups.new(name=nm)
    names = [g.name for g in o.vertex_groups]
    rig = next((m_.object for m_ in o.modifiers if m_.type == 'ARMATURE'), None)
    bones = np.array([nm in rig.data.bones for nm in names])
    n = len(me.vertices)
    W = np.zeros((n, len(names)))
    for v in me.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    W0 = W.copy()
    X = np.array([v.co[:] for v in me.vertices])
    if like is not None:
        Xu, Wu, nu = like
        low = np.nonzero(Xu[:, 2] < Xu[:, 2].min() + 0.6)[0]
        from mathutils.kdtree import KDTree
        kd = KDTree(len(low))
        for j, i in enumerate(low):
            kd.insert((Xu[i, 0], Xu[i, 1], 0.0), j)
        kd.balance()
        near = np.array([low[kd.find((x[0], x[1], 0.0))[1]] for x in X])
        fwd = X[:, 1] < Xu[:, 1].min() + band[1] + 1.5
        for j, nm in enumerate(names):
            if bones[j] and nm in nu:
                W[fwd, j] = Wu[near[fwd], nu.index(nm)]
    else:
        it, i_f = names.index('Toe.' + side), names.index('Foot.' + side)
        y = X[:, 1] - X[:, 1].min()
        # (the band widens with height: the vamp's top stands well over the joint and a narrow bend there shortens
        # it by a third, which folds it; low by the sole the bend stays short, as a sole's is)
        c0, hw = 0.5 * (band[0] + band[1]), 0.5 * (band[1] - band[0]) + BEND_FAN * np.clip(X[:, 2] - X[:, 2].min() - 0.3, 0.0, 2.0)
        t = np.clip((y - (c0 - hw)) / (2.0 * hw), 0.0, 1.0)
        s_new = 1.0 - t * t * (3.0 - 2.0 * t)
        Ot = W * bones[None, :]
        Ot[:, it] = 0.0
        none = Ot.sum(1) < 1e-3
        Ot[none] = 0.0
        Ot[none, i_f] = 1.0
        Ot /= Ot.sum(1, keepdims=True)
        Wn = Ot * (1.0 - s_new)[:, None]
        Wn[:, it] = s_new
        fwd = y < band[1] + 2.5                             # (behind that the toes' bone has no share either way)
        W[np.ix_(fwd, bones)] = Wn[np.ix_(fwd, bones)]
        # ---- and the ankle's bend made a shell's (SHELL). The hose's weights are a leg's: the shin's share differs
        # between the inner and the outer ankle and changes over a few millimetres, so a boot that is the same on
        # both sides at rest stands on its heel bent more on one side than the other, with a fold across the instep
        # (gauge.py: pose asymmetry). Leather is one stiff shell: its bend depends on where a point lies along the
        # foot and how high, not on which side. The shin's share is averaged across the boot on a grid of (along,
        # up), eased, and read back, the foot's bone taking what is left.
        if SHELL and 'Shin.' + side in names:
            i_s = names.index('Shin.' + side)
            zz = X[:, 2] - X[:, 2].min()
            ny, nz = 48, 48
            gy, gz = np.clip((y / (y.max() + 1e-6) * ny).astype(int), 0, ny - 1), np.clip((zz / (zz.max() + 1e-6) * nz).astype(int), 0, nz - 1)
            num, den = np.zeros((ny, nz)), np.zeros((ny, nz))
            np.add.at(num, (gy, gz), W[:, i_s])
            np.add.at(den, (gy, gz), 1.0)
            for _ in range(60):                             # (empty cells filled from their neighbours, all eased)
                pn, pd = np.pad(num, 1, mode='edge'), np.pad(den, 1, mode='edge')
                an = 0.25 * (pn[:-2, 1:-1] + pn[2:, 1:-1] + pn[1:-1, :-2] + pn[1:-1, 2:])
                ad = 0.25 * (pd[:-2, 1:-1] + pd[2:, 1:-1] + pd[1:-1, :-2] + pd[1:-1, 2:])
                k_ = SHELL if _ < 12 else 0.0
                num = np.where(den > 0, (1 - k_) * num + k_ * an, an)
                den = np.where(den > 0, (1 - k_) * den + k_ * ad, ad)
            grid = num / np.maximum(den, 1e-9)
            fy, fz = y / (y.max() + 1e-6) * ny - 0.5, zz / (zz.max() + 1e-6) * nz - 0.5
            y0_, z0_ = np.clip(np.floor(fy).astype(int), 0, ny - 2), np.clip(np.floor(fz).astype(int), 0, nz - 2)
            ty, tz = np.clip(fy - y0_, 0, 1), np.clip(fz - z0_, 0, 1)
            sh_ = ((1 - ty) * (1 - tz) * grid[y0_, z0_] + ty * (1 - tz) * grid[y0_ + 1, z0_]
                   + (1 - ty) * tz * grid[y0_, z0_ + 1] + ty * tz * grid[y0_ + 1, z0_ + 1])
            oth = W[:, bones].sum(1) - W[:, i_s] - W[:, i_f]
            sh_ = np.clip(sh_, 0.0, np.maximum(1.0 - oth, 0.0))
            log('zipboot %s: the ankle\'s bend made a shell\'s: the shin\'s share changed by %.2f at most, %.3f on average'
                % (o.name, float(np.abs(sh_ - W[:, i_s]).max()), float(np.abs(sh_ - W[:, i_s]).mean())), flush=True)
            W[:, i_s] = sh_
            W[:, i_f] = np.maximum(1.0 - oth - sh_, 0.0)
    ch = np.nonzero(np.abs(W - W0).max(1) > 1e-4)[0]
    for g in o.vertex_groups:
        if not bones[g.index]:
            continue
        col = W[:, g.index]
        for i in ch:
            if col[i] > 1e-4:
                g.add([int(i)], float(col[i]), 'REPLACE')
            elif W0[i, g.index] > 0:
                g.remove([int(i)])
    log('zipboot %s: the bend moved back to the ball (%.1f to %.1f behind the tip): %d points re-weighted, by %.2f at most'
        % (o.name, band[0], band[1], len(ch), float(np.abs(W - W0).max())), flush=True)
    return X, W, names


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
    _carry(o, R, dR)
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
    boot.last.RATIOS.update(toe_box=TOE_BOX, toe_sq=TOE_SQ, toe_front=TOE_FRONT, toe_wall=TOE_WALL,
                            toe_cut_r=TOE_CUT_R, toe_rake=TOE_RAKE, sole_blur=4.0, sole_toe=0.7, toe_ease=TOE_EASE, toe_spare=0.08, toe_sweep=TOE_SWEEP, toe_bow=TOE_BOW, toe_side=TOE_SIDE)   # (the sole's outline kept square at the toe)
    boot.last.RATIOS['welt'] = WELT
    heelboot.SEAT_BAND, heelboot.SEAT_OUT = SEAT_BAND, SEAT_OUT
    boot.last.RATIOS['throat_line'], boot.last.RATIOS['throat_from'] = THROAT_LINE, THROAT_FROM
    female.TURNOUT_AT_HIP = TURNOUT_AT_HIP
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
        if BEND:
            up_ = bend_at_ball(o, BEND)
            if bpy.data.objects.get('Boot sole.' + side) is not None:
                bend_at_ball(bpy.data.objects['Boot sole.' + side], BEND, like=up_)
        if PITCH_SET:
            root.location.z = up               # (as it stands on the heel)
            bpy.context.view_layer.update()
            set_at_pitch(o, boot.hose.MEASURES, PITCH_SET)
            root.location.z = 0.0
            bpy.context.view_layer.update()
        true_seam(o, boot.last.PIECES[side])       # (last: the pitch's smoothing ripples an open edge)
        zipper.build(o, boot.last.PIECES[side], rig, L,
                     {'teeth': female.mat(M_TEETH), 'tape': female.mat(M_TAPE), 'metal': female.mat(M_METAL)})
    if ENTRY_CHECK:
        entry(boot.last.PIECES, boot.hose.MEASURES)
    root.location.z = up
    bpy.context.view_layer.update()
    if DESIGN:
        design(boot.last.PIECES)
    return res
