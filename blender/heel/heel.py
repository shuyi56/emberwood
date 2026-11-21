"""A study figure: the bare mannequin's feet raised onto heels derived from the feet themselves (study.py), with the
footbed, the steel shank, the sole and the stacked heel built under each foot, for clay drafts and the design
colour pass. It does not touch the boots (blender/boot): it shows what a heel derived this way looks like on the
foot, and its gauge says how the foot is carried.

    python3 blender/base/views.py --figure heel --mannequin --head --angles profile,back --clay --only 'Skin body,Heel' --out DIR

H_SEAT is the heel's height (the sole of the foot's rise at the seat point, cm); None derives it: the height at which
the rear foot's load centres on the heel's tread (study.best_height). ARCH_FILL and CUP are the footbed's fit (1: it
follows the posed sole of the foot through the waist / round the heel; 0: a straight shank line, a flat seat).
HEEL_DUMP=PATH saves the feet's data (.npz) for tools/heel/sheet.py, which draws the support and tension sheet.

The rig has one bone for the whole foot (no toe joint), so a heel cannot be posed with the rig: here the rest shape
of the body itself is bent about the metatarsal heads' line (study.pose) and the body taken off the rig. The figure
is good for still views only (no --walk)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for d in ('hose', 'vest', 'sorceress_pendant', 'shared', 'base'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
os.environ.setdefault('HEAD_GLB', '')
os.environ.setdefault('NO_DRESS', '1')
os.environ.setdefault('NO_ACC', '1')
import bpy  # noqa: E402
import numpy as np  # noqa: E402
import female  # noqa: E402
import sorceress_pendant as SP  # noqa: E402
import draft  # noqa: E402
import study as S  # noqa: E402

H_SEAT = None
ARCH_FILL = 1.0
CUP = 1.0
TOP_OF_SEAT = 0.85
RESULT = {}

# the design colour pass: one flat colour for each part of the heel (design parts, not materials)
COLOURS = {'insole': (0.20, 0.62, 0.62), 'shank': (0.85, 0.15, 0.15), 'sole': (0.95, 0.55, 0.10),
           'lift a': (0.55, 0.42, 0.75), 'lift b': (0.35, 0.55, 0.85), 'wedge': (0.90, 0.80, 0.20),
           'top lift': (0.20, 0.70, 0.30)}


def _mat(name):
    m = bpy.data.materials.get('Heel ' + name)
    if m is None:
        m = bpy.data.materials.new('Heel ' + name)
        m.use_nodes = True
        b = m.node_tree.nodes.get('Principled BSDF')
        b.inputs['Base Color'].default_value = (*COLOURS[name], 1.0)
        b.inputs['Roughness'].default_value = 0.7
    return m


def _object(name, V, F, mat):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in V], [], [tuple(int(i) for i in f) for f in F])
    me.update()
    o = bpy.data.objects.new(name, me)
    o.data.materials.append(_mat(mat))
    female.link(o, female.FIG['root'])
    return o


def _slab(top, bot, nu, nv):
    """A closed slab from two grids of points [nu, nv, 3] (its top and under faces). Returns (V, F)."""
    V = np.concatenate([top.reshape(-1, 3), bot.reshape(-1, 3)])
    n = nu * nv
    idx = lambda i, j, b=0: b * n + i * nv + j  # noqa: E731
    F = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            F.append([idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)])
            F.append([idx(i, j, 1), idx(i + 1, j, 1), idx(i + 1, j + 1, 1), idx(i, j + 1, 1)])
    ring = [(i, 0) for i in range(nu)] + [(nu - 1, j) for j in range(1, nv)] + \
           [(i, nv - 1) for i in range(nu - 2, -1, -1)] + [(0, j) for j in range(nv - 2, 0, -1)]
    for k in range(len(ring)):
        a, b = ring[k], ring[(k + 1) % len(ring)]
        F.append([idx(*a), idx(*b), idx(*b, 1), idx(*a, 1)])
    return V, F


class Bed:
    """The posed sole of the foot and its footbed, sampled anywhere in plan (nearest cell)."""

    def __init__(self, Pf, zb):
        self.P, self.z = Pf, zb
        self.step = 0.15

    def at(self, x, y):
        q = np.stack([x.ravel(), y.ravel()], 1)
        d2 = ((q[:, None, :] - self.P[None, :, :2]) ** 2).sum(2)
        w = np.exp(-d2 / (2 * self.step ** 2))
        w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
        return (w @ self.z).reshape(x.shape)

    def outline(self, ys, grow):
        xl, xr = [], []
        for y in ys:
            m = np.abs(self.P[:, 1] - y) < 0.12
            if not m.any():
                m = np.argsort(np.abs(self.P[:, 1] - y))[:4]
            xl.append(self.P[m, 0].min() - grow)
            xr.append(self.P[m, 0].max() + grow)
        xl, xr = np.array(xl), np.array(xr)
        # (ahead of the ball the sole holds the ball's breadth: the toes' pads touch in patches, but the sole runs
        # under all of them; the toe is rounded below)
        if ys[-1] < ys[0]:
            ib = int(np.argmax(xr - xl))
            xl[ib:] = np.minimum.accumulate(xl[ib:])
            xr[ib:] = np.maximum.accumulate(xr[ib:])
        # (the cells' edges step by a cell; smoothed along the length so the outline is a curve, not a staircase)
        k = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0
        for _ in range(3):
            xl = np.convolve(np.pad(xl, 2, mode='edge'), k, mode='valid')
            xr = np.convolve(np.pad(xr, 2, mode='edge'), k, mode='valid')
        # (the back and the toe rounded in plan: the width eases to a round over the last half of a heel's breadth)
        s_ = np.abs(ys - ys[0]), np.abs(ys - ys[-1])
        r = 0.5 * float(np.median(xr - xl))
        for dist in s_:
            near = dist < r
            c = 0.5 * (xl + xr)
            f = np.sqrt(np.clip(1 - ((r - dist) / r) ** 2, 0.0, 1.0))
            f = np.where(near, np.maximum(f, 0.25), 1.0)
            xl, xr = c - (c - xl) * f, c + (xr - c) * f
        return xl, xr


def build_heel(side, f, d, out, H, nv=11):
    """The footbed, shank, sole and stacked heel under one foot (out: study.support's result for it)."""
    Pf, zb = out['Pf'], out['zb']
    bed = Bed(Pf, zb)
    welt = 0.022 * H                                     # (the boot's own welt: last.RATIOS['welt'])
    back, toe = Pf[:, 1].max() + 0.5 * bed.step, Pf[:, 1].min() - 0.5 * bed.step - 0.25   # (past the toes' tips)
    ys = np.linspace(back, toe, 46)
    xl, xr = bed.outline(ys, 0.5 * bed.step)
    v = np.linspace(0, 1, nv)

    def grid(xl_, xr_, ys_):
        X = xl_[:, None] + (xr_ - xl_)[:, None] * v[None, :]
        Y = np.repeat(ys_[:, None], nv, 1)
        return X, Y

    X, Y = grid(xl, xr, ys)
    zt = bed.at(np.clip(X, xl[:, None], xr[:, None]), Y)
    up = d['insole'] + d['sole']                         # (the foot stands this far up off the floor, on the sole)
    lift = lambda X_, Y_, Z_: np.stack([X_, Y_, Z_ + up], -1)  # noqa: E731
    objs = []
    Vt, Ft = _slab(lift(X, Y, zt), lift(X, Y, zt - d['insole']), len(ys), nv)
    objs.append(_object('Heel insole.%s' % side, Vt, Ft, 'insole'))
    # the sole: the insole's outline let out by the welt, under it
    Xs, Ys = grid(xl - welt, xr + welt, ys)
    zs = bed.at(np.clip(Xs, xl[:, None], xr[:, None]), Ys) - d['insole']
    Vs, Fs = _slab(lift(Xs, Ys, zs), lift(Xs, Ys, zs - d['sole']), len(ys), nv)
    objs.append(_object('Heel sole.%s' % side, Vs, Fs, 'sole'))
    # the shank: a steel strip under the insole down the middle of the waist
    sy = np.linspace(d['shank_from'], d['shank_to'], 24)
    mid = 0.5 * (np.interp(-sy, -ys, xl) + np.interp(-sy, -ys, xr))
    hw, t = 0.5 * S.EXP['shank_w'] / S.MM, S.EXP['shank_t'] / S.MM
    Xk, Yk = (mid - hw)[:, None] + 2 * hw * np.linspace(0, 1, 3)[None, :], np.repeat(sy[:, None], 3, 1)
    zk = bed.at(Xk, Yk) - d['insole']
    Vk, Fk = _slab(lift(Xk, Yk, zk), lift(Xk, Yk, zk - t), len(sy), 3)
    objs.append(_object('Heel shank.%s' % side, Vk, Fk, 'shank'))
    # the heel: lifts stacked from the floor to the sole's under face behind the breast; its plan narrows from the
    # seat's outline (at the sole) to the tread's (on the floor), centred on the seat point (Golding)
    hy = np.linspace(back, d['breast_y'], 20)
    sl, sr = bed.outline(hy, 0.5 * bed.step + welt)
    cx = 0.5 * (sl + sr)
    c0 = float(cx.mean())
    half_top = 0.5 * d['top_w']
    top_back = d['top_y'] + 0.5 * d['top_len']
    floor = f['floor']
    seat_under = lambda X_, Y_: bed.at(np.clip(X_, sl.min(), sr.max()), Y_)  # noqa: E731   (the sole's under face)
    z_seat = float(seat_under(np.array([[c0]]), np.array([[d['top_y']]]))[0, 0])
    tl = S.LIFT_MM / S.MM
    n = int(np.ceil((z_seat - floor + 1e-6) / tl))
    for i in range(n):
        z0, z1 = floor + i * tl, floor + (i + 1) * tl

        def plan(zz):
            u = np.clip((zz - floor) / max(z_seat - floor, 1e-6), 0, 1)        # 0 on the floor, 1 at the seat
            yb = top_back + (back - top_back) * u                              # (the back face pitched)
            yy = np.linspace(yb, d['breast_y'], len(hy))
            l_, r_ = np.interp(-yy, -hy, sl), np.interp(-yy, -hy, sr)
            w = (r_ - l_)
            cw = np.interp(-yy, -hy, cx)
            wt = np.minimum(w, 2 * half_top * np.sqrt(np.clip(1 - ((yy - d['top_y']) / (0.5 * d['top_len'] + 1e-6)) ** 2, 0.15, 1)))
            ww = wt + (w - wt) * u
            return cw - 0.5 * ww, cw + 0.5 * ww, yy
        l0, r0, y0 = plan(z0)
        l1, r1, y1 = plan(z1)
        Xb, Yb = grid(l0, r0, y0)
        Xt, Yt = grid(l1, r1, y1)
        cap = seat_under(Xt, Yt)
        Zt = np.minimum(np.full(Xt.shape, z1), cap)
        Zb = np.full(Xb.shape, z0)
        Zt = np.maximum(Zt, z0 + 0.02)
        name = 'top lift' if i == 0 else ('wedge' if z1 >= z_seat - 1e-6 or i == n - 1 else ('lift a' if i % 2 else 'lift b'))
        Vh, Fh = _slab(np.stack([Xt, Yt, Zt], -1), np.stack([Xb, Yb, Zb], -1), len(y0), nv)
        objs.append(_object('Heel %s %d.%s' % (name, i, side), Vh, Fh, name))
    return objs


def pose_body(body, feet, ths, up):
    """The body's rest shape bent onto the heels (each side by its own foot), and the body taken off the rig."""
    me = body.data
    n = len(me.vertices)
    X = np.empty(n * 3)
    me.vertices.foreach_get('co', X)
    X = X.reshape(-1, 3)
    out = X.copy()
    for side, (f, *_rest) in feet.items():
        m = (X[:, 0] < 0) if side == 'R' else (X[:, 0] >= 0)
        out[m] = S.pose(X[m], f, ths[side])
    out[:, 2] += up                              # (on the insole and sole)
    if me.shape_keys:
        for kb in me.shape_keys.key_blocks:
            kb.data.foreach_set('co', out.ravel())
    me.vertices.foreach_set('co', out.ravel())
    me.update()
    for mo in list(body.modifiers):
        if mo.type == 'ARMATURE':
            body.modifiers.remove(mo)
    # (off the rig, the body no longer turns with the figure's root: parent it there, as it stands)
    mw = body.matrix_world.copy()
    body.parent = female.FIG['root']
    body.matrix_world = mw


def build():
    res = SP.build()
    for o in bpy.data.objects:                   # (the necklace's parts: NO_ACC hides only the chain)
        if o.name.startswith('Necklace'):
            o.hide_render = True
            o.visible_camera = o.visible_shadow = o.visible_diffuse = o.visible_glossy = False
    bpy.context.view_layer.update()
    body = bpy.data.objects['Skin body']
    rig = bpy.data.objects.get('Rig')
    H = SP.anatomy.head_height() or SP.anatomy.HEAD
    B = draft.body_data(body)
    # (only the ankle is needed here: the hose draft's full leg measure reads the trunk too, and trips on a seam
    # vertex between the thighs that the voxel grid leaves on some builds)
    M = {'ankle_z': float(rig.data.bones['Foot.R'].head_local.z)}
    P, N, W, names = B['P'], B['N'], B['W'], B['groups']
    col = {g: i for i, g in enumerate(names)}
    feet = {}
    for side, sgn in (('R', -1.0), ('L', 1.0)):
        mask = (sgn * P[:, 0] > 0) & (P[:, 2] < M['ankle_z'] + 1.5) & \
               (W[:, col['Foot.' + side]] + W[:, col['Shin.' + side]] > 0.5)
        ankle = np.array(rig.data.bones['Foot.' + side].head_local)
        f = S.measure(P, N, mask, ankle)
        Pl, step = S.plantar(P, N, mask)
        feet[side] = (f, Pl, step, mask, ankle)
    fR, PlR, stepR = feet['R'][:3]
    k, e, bare = S.calibrate(PlR, stepR, fR)
    h = H_SEAT if H_SEAT is not None else S.best_height(PlR, stepR, fR, k, e, arch_fill=ARCH_FILL, cup=CUP)
    print('heel: the bed %.0f kPa per unit, the weight line %.2f cm in front of the ankle; barefoot shares %s'
          % (k, e * S.CM, {a: round(b, 3) for a, b in bare['share'].items()}), flush=True)
    ths, outs = {}, {}
    for side, (f, Pl, step, mask, ankle) in feet.items():
        d = S.derive(f, h, arch_fill=ARCH_FILL, cup=CUP, top_of_seat=TOP_OF_SEAT)
        o = S.support(Pl, step, f, d, k, e)
        ths[side], outs[side] = np.radians(o['theta_deg']), (d, o)
        print('heel %s: %s' % (side, S.summary(f, d, o)), flush=True)
    d0 = outs['R'][0]
    if os.environ.get('HEEL_PARTS', '1') != '0':     # (HEEL_PARTS=0: the bare foot, flat on the floor)
        pose_body(body, feet, ths, d0['insole'] + d0['sole'])
        for side, (d, o) in outs.items():
            build_heel(side, feet[side][0], d, o, H)
    RESULT.update(feet=feet, outs=outs, k=k, e=e, h=h, bare=bare)
    if os.environ.get('HEEL_DUMP'):
        np.savez(os.environ['HEEL_DUMP'], P=P, N=N, maskR=feet['R'][3], maskL=feet['L'][3],
                 ankleR=feet['R'][4], ankleL=feet['L'][4], H=H)
    # the --head view on the feet: the floor to over the ankle
    os.environ.setdefault('HEAD_CZ', '%.2f' % (fR['floor'] + 3.6))
    os.environ.setdefault('HEAD_H', '%.2f' % 12.5)
    bpy.context.view_layer.update()
    return res
