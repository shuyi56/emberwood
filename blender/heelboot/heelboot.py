"""The tall boots on a heel: the boot figure (blender/boot) over the heel study's derived heel. The foot is held pitched
onto the heel in the walk (female.HEEL_PITCH, as in blender/heelwalk); the boot's own flat last and sole are drafted on
the foot at rest, and the heel's stacked lifts (blender/heel) ride the foot's bones under the sole.

    python3 blender/base/views.py --figure heelboot --mannequin --clay-only --walk 16 --angles profile --out DIR"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('heel', 'boot'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
os.environ['HEEL_PARTS'] = '1'
os.environ['HEEL_WALK'] = '1'
import bpy  # noqa: E402
import numpy as np  # noqa: E402
import boot  # noqa: E402   (first: it sets the environment the figure is built under)
import female  # noqa: E402
import heel  # noqa: E402


# The heeled boot's own cut of the shaft (the rest is the boot's: last.RATIOS). John, 2026-11-21: too big at the knees.
CUT = {
    'top_below_knee': 0.20,  # the top edge sits under the knee's swell, clear of its bend (the boot's own: 0.12)
    'top_back': 0.02,        # and stands less high at the back, where a bent knee leaves the calf
    'top_fit': 1.0,          # the top drawn all the way in to the leg's own section
    'top_snug': 0.9,         # with nearly all the ease taken out there
    'ease': 0.015,           # the shaft's room over the leg, shin to calf (the boot's own: 0.035), so it follows the shin
    'calf_swell': 0.005,     # and next to no swell of its own over the calf (0.015)
}


M_THONG = female.mat_def(79, 'boot thong', female.ramp(0x6b4a30, 0.1), 0.3)


M_SOLE = female.mat_def(87, 'boot sole leather', female.ramp(0x3b2a1e, 0.08), 0.3, surface='leather', closeup='leather')
M_LIFT_A = female.mat_def(88, 'heel lift, light', female.ramp(0x70503a, 0.1), 0.3, surface='leather', closeup='leather')
M_LIFT_B = female.mat_def(89, 'heel lift, dark', female.ramp(0x62442c, 0.1), 0.3, surface='leather', closeup='leather')
M_TOP_LIFT = female.mat_def(90, 'heel top lift', female.ramp(0x221d1a, 0.08), 0.45)
WELD = 0.03                  # the stack's seat stands this far up into the sole, so the two are one solid
TAPER = 0.05                 # the stack closes in this share of its size from the seat down to the tread


def join_heel(side, lift):
    """The stacked heel built on the boot's own sole and joined to it: its seat is the sole's under face over the
    whole back of the boot, from the breast (where the study put it) round the sole's edge, let out to the back end
    of the counter; its tread is the study's (the floor, as the pitched foot stands on it); between them lifts of sole
    leather `lift` thick, the lowest the top lift. The study's own stack, cut on the bare foot's outline, is replaced."""
    sole = bpy.data.objects.get('Boot sole.' + side)
    old = [o for o in bpy.data.objects if o.name.startswith('Heel ') and o.name.endswith('.' + side)]
    if sole is None or not old:
        return []
    Sv = np.array([(sole.matrix_world @ v.co)[:] for v in sole.data.vertices])
    na = (len(Sv) - 2) // 5
    # the sole tailored to the boot as sewn: the counter settles out past the edge the sole was cut to, so round the
    # back the sole's rim is let out to the upper's own outline there, plus the welt
    up_ = bpy.data.objects.get('Boot.' + side)
    A0 = np.concatenate([np.array([v.co[:] for v in o.data.vertices]) for o in old])
    y_br = float(A0[:, 1].min())                               # the breast
    c2 = Sv[-2:, :2].mean(0)
    if up_ is not None:
        Uv = np.array([(up_.matrix_world @ v.co)[:] for v in up_.data.vertices])
        Uv = Uv[(Uv[:, 2] > Sv[:, 2].max() - 0.05) & (Uv[:, 2] < Sv[:, 2].max() + 1.1)]
        ang = np.arctan2(Uv[:, 0] - c2[0], -(Uv[:, 1] - c2[1]))
        rr = np.linalg.norm(Uv[:, :2] - c2, axis=1)
        bins = np.floor((ang + np.pi) / (2 * np.pi) * na).astype(int) % na
        env = np.array([rr[bins == b].max() if (bins == b).any() else 0.0 for b in range(na)])
        env = np.maximum(env, np.roll(env, 1))
        env = np.maximum(env, np.roll(env, -2))                # (a point's width either way: no notch between bins)
        r3 = np.linalg.norm(Sv[3 * na:4 * na, :2] - c2, axis=1)
        welt = boot.last.RATIOS['welt'] * boot.hose.MEASURES['H']
        f = np.maximum(1.0, (env + welt) / r3)
        for _ in range(3):
            f = (np.roll(f, 1) + 2 * f + np.roll(f, -1)) / 4
        f = np.maximum(f, 1.0)
        back = np.clip((Sv[3 * na:4 * na, 1] - (y_br - 1.2)) / 1.2, 0, 1)
        f = 1.0 + (f - 1.0) * back * back * (3 - 2 * back)     # (only behind the breast, eased in ahead of it)
        for k_ in range(5):
            Sv[k_ * na:(k_ + 1) * na, :2] = c2 + (Sv[k_ * na:(k_ + 1) * na, :2] - c2) * f[:, None]
        inv = sole.matrix_world.inverted()
        from mathutils import Vector
        for v, x in zip(sole.data.vertices, Sv):
            v.co = inv @ Vector(x)
        sole.data.update()
        print('heelboot %s: the sole let out to the counter: %.2f at most, %.2f at the back'
              % (side, float(((f - 1) * r3).max()), float(((f - 1) * r3)[np.argmax(Sv[3 * na:4 * na, 1])])), flush=True)
    ring = Sv[3 * na:4 * na]                                   # the sole's rim, at its widest
    z_under = float(Sv[:, 2].min())
    y0 = y_br
    # the tread's plane: the study's stack's under face (tilted, in the foot's rest frame, by the heel's pitch)
    low = next(o for o in old if 'top lift 0' in o.name)
    T = np.array([v.co[:] for v in low.data.vertices])
    for _ in range(2):
        G = np.stack([T[:, 1], np.ones(len(T))], 1)
        k = np.linalg.lstsq(G, T[:, 2], rcond=None)[0]
        T = T[T[:, 2] <= G @ k + 1e-4] if len(T) > 8 else T
    tread = lambda y: k[0] * y + k[1]                          # noqa: E731
    for o in old:
        bpy.data.objects.remove(o, do_unlink=True)
    yb = float(ring[:, 1].max())
    arc = ring[ring[:, 1] >= y0][:, :2]
    cx = float(arc[:, 0].mean())
    arc = arc[np.argsort(np.arctan2(arc[:, 0] - cx, arc[:, 1] - y0 + 1e-6))]
    c = np.array([cx, y0 + 0.45 * (yb - y0)])
    z_top = z_under + WELD
    h = z_top - tread(arc[:, 1])
    n = max(1, int(np.ceil(float(h.max()) / lift)))
    mats = [female.mat(M_TOP_LIFT)] + [female.mat(M_LIFT_A if i % 2 else M_LIFT_B) for i in range(1, n)]
    arm = female.RIG['arm']
    out = []
    m = len(arc)

    def level(kk):
        z = np.minimum(tread(arc[:, 1]) + kk * lift, z_top) if kk < n else np.full(m, z_top)
        u = np.clip((z_top - z) / max(float(h.max()), 1e-6), 0, 1)        # 0 at the seat, 1 on the tread
        xy = c + (arc - c) * (1.0 - TAPER * u)[:, None]
        xy[:, 1] = np.maximum(xy[:, 1], y0)                               # (the breast stands straight)
        return np.column_stack([xy, z])

    for i in range(n):
        lo, hi = level(i), level(i + 1)
        V = np.concatenate([lo, hi])
        F = [[a, (a + 1) % m, m + (a + 1) % m, m + a] for a in range(m)]
        F += [list(range(m))[::-1], list(range(m, 2 * m))]
        me = bpy.data.meshes.new('Heel lift %d.%s' % (i, side))
        me.from_pydata(V.tolist(), [], F)
        me.materials.append(mats[i])
        o = bpy.data.objects.new(me.name, me)
        bpy.context.scene.collection.objects.link(o)
        g = o.vertex_groups.new(name='Foot.' + side)
        g.add(list(range(len(V))), 1.0, 'REPLACE')
        mo = o.modifiers.new('rig', 'ARMATURE')
        mo.object = arm
        out.append(o)
    print('heelboot %s: the heel joined to the sole: %d lifts, %.2f high at the back, %.2f at the breast, seat %.2f into the sole'
          % (side, n, float(z_top - tread(yb)), float(z_top - tread(y0)), WELD), flush=True)
    return out


def build():
    # (set on every build: a render worker that keeps its scene loaded resets the environment between jobs)
    os.environ['HEEL_PARTS'] = '1'
    os.environ['HEEL_WALK'] = '1'
    os.environ['FOOT_FORM'] = 'shod'
    res = heel.build()                         # the figure, the derived heel, the lifts on the foot's bones
    up = female.HEEL_UP
    root = female.FIG['root']
    root.location.z = 0.0                      # (the boot is drafted on the foot standing at rest)
    bpy.context.view_layer.update()
    for o in list(bpy.data.objects):           # (the boot has its own insole, sole and welt: only the stack stays)
        if o.name.startswith(('Heel insole', 'Heel sole', 'Heel shank')):
            bpy.data.objects.remove(o, do_unlink=True)
    boot.last.RATIOS.update(CUT)
    sp = boot.hose.SP
    keep = sp.build
    sp.build = lambda: res                      # (the figure is built already)
    try:
        boot.build()
    finally:
        sp.build = keep
    stacks = {side: join_heel(side, heel.S.LIFT_MM / heel.S.MM) for side in 'RL'}
    for side in 'RL':                          # (the sole in sole leather, its edge dressed dark)
        so = bpy.data.objects.get('Boot sole.' + side)
        if so is not None:
            so.data.materials.clear()
            so.data.materials.append(female.mat(M_SOLE))
    # (the lace and the eyelets in the leather's own tone, a shade down: near black, they drew a dark line down the
    # boot's front edge in every side view)
    for o in bpy.data.objects:
        if o.name.startswith(('Boot lace', 'Boot eyelet')) or 'lace' in o.name.lower() or 'eyelet' in o.name.lower():
            o.data.materials.clear()
            o.data.materials.append(female.mat(M_THONG))
    # what the foot stands on: the boot's sole and the stack's tread, behind the toe joint
    az = female.ANKLE_Z
    tread = []
    for o in [bpy.data.objects.get('Boot sole.R')] + stacks['R'][:1]:
        if o is None:
            continue
        Z = np.array([(o.matrix_world @ v.co)[:] for v in o.data.vertices])
        Z = Z[-Z[:, 1] <= female.FOOT_BALL]
        tread += [(float(-z[1]), float(z[2] - az)) for z in Z[::3 if o.name.startswith('Heel') else 7]]
    female.HEEL_CONTACT = tread
    root.location.z = up
    bpy.context.view_layer.update()
    print('heelboot: %d contact points, raised %.2f' % (len(tread), up), flush=True)
    return res
