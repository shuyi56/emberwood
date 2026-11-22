"""The tall boots on a heel: the boot figure (blender/boot) over the heel study's derived heel. The foot is held pitched
onto the heel in the walk (female.HEEL_PITCH, as in blender/heelwalk); the boot's own flat last and sole are drafted on
the foot at rest, and the heel's stacked lifts (blender/heel) ride the foot's bones under the sole.

    python3 blender/base/views.py --figure heelboot --mannequin --clay-only --walk 16 --angles profile --out DIR"""
import math
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
# the stack's edge: lifts of vegetable-tanned sole leather (5 mm each: the atelier's rule, study.LIFT_MM), glassed and
# sanded smooth, dyed to the upper's brown a shade down and waxed (Gentleman's Gazette, finishing a bespoke shoe: an
# alcohol dye matched to the upper, three coats of hard wax ironed in, a last shinier coat). stack_material() draws
# the lifts' joints and the fibre's grain on it.
M_STACK = female.mat_def(88, 'heel stack edge', female.ramp(0x402a1b, 0.1), 0.35, closeup='leather')
M_TOP_LIFT = female.mat_def(90, 'heel top lift', female.ramp(0x221d1a, 0.08), 0.45)
# The heel's shape and height, as tools/heel/optimize.py's gradient descent on the support gauge found them for this
# foot (2026-11-21; three starts agree on a block: straight sides, no rake, 1.8 to 2.0 cm high, the tread a fifth
# shorter than the seat's run). The descent's weights are EXPERIMENTAL (see that file).
SHAPE = dict(h=1.77,        # cm, the rise at the seat point
             width=1.0,     # the tread's width, a share of the seat's (1: straight sides)
             length=0.81,   # the tread's length, a share of the block's
             rake=0.0,      # the tread's middle in front of the seat point, a share of the block's tread length
             arch=0.39,     # the footbed's fill through the waist
             cup=1.0)       # and its cup round the heel
WELD = 0.03                  # the stack's seat stands this far up into the sole, so the two are one solid
SEAT_OUT = 0.06              # how far the sole stands out past the counter round the heel seat (about 1.5 mm)


def stack_material():
    """The stacked heel's edge in the 3D look: one dyed, waxed face (not a hide's pebbled grain) on which each lift
    still reads: a fine dark joint where two lifts meet, each lift a touch lighter or darker than its neighbours
    (the dye takes differently on each piece), and the leather's fibre showing as faint streaks along the lifts.
    Reads the lifts' own 'stack_u' (the height up the stack, in lifts) and 'rest'. EXPERIMENTAL: the strengths are
    set by eye; the sources give the finish, not numbers for it."""
    m = female.mat(M_STACK)
    nt = m.node_tree
    bsdf = nt.nodes.get('close bsdf')
    if bsdf is None or nt.nodes.get('stack joint') is not None:
        return m
    N, L = nt.nodes.new, nt.links.new

    def math(op, a, b=None, name=None):
        n = N('ShaderNodeMath')
        n.operation = op
        for k, v in ((0, a), (1, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[k].default_value = v
            else:
                L(v, n.inputs[k])
        if name:
            n.name = name
        return n.outputs[0]

    u = N('ShaderNodeAttribute')
    u.attribute_name = 'stack_u'
    d = math('PINGPONG', u.outputs['Fac'], 0.5)                 # how far from the nearest joint, in lifts
    joint = N('ShaderNodeMapRange')
    joint.name = 'stack joint'
    joint.interpolation_type = 'SMOOTHSTEP'
    for k, v in (('From Min', 0.0), ('From Max', 0.07), ('To Min', 1.0), ('To Max', 0.0)):
        joint.inputs[k].default_value = v
    L(d, joint.inputs['Value'])
    piece = N('ShaderNodeTexWhiteNoise')                        # each lift its own tone
    piece.noise_dimensions = '1D'
    L(math('FLOOR', u.outputs['Fac']), piece.inputs['W'])
    rest = N('ShaderNodeAttribute')
    rest.attribute_name = 'rest'
    mp = N('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = (1.2, 1.2, 30.0)         # (streaks along the lifts)
    L(rest.outputs['Vector'], mp.inputs['Vector'])
    fibre = N('ShaderNodeTexNoise')
    fibre.inputs['Scale'].default_value = 2.0
    fibre.inputs['Detail'].default_value = 3.0
    L(mp.outputs['Vector'], fibre.inputs['Vector'])
    val = math('MULTIPLY', math('MULTIPLY_ADD', piece.outputs['Value'], 0.22), math('MULTIPLY_ADD', fibre.outputs['Fac'], 0.24))
    for n_, add in ((val.node.inputs[0].links[0].from_node, 0.89), (val.node.inputs[1].links[0].from_node, 0.88)):
        n_.inputs[2].default_value = add
    val = math('MULTIPLY', val, math('MULTIPLY_ADD', joint.outputs['Result'], -0.5))
    val.node.inputs[1].links[0].from_node.inputs[2].default_value = 1.0
    hsv = N('ShaderNodeHueSaturation')
    hsv.inputs['Color'].default_value = bsdf.inputs['Base Color'].default_value
    for l_ in list(bsdf.inputs['Base Color'].links):
        L(l_.from_socket, hsv.inputs['Color'])
    L(val, hsv.inputs['Value'])
    L(hsv.outputs['Color'], bsdf.inputs['Base Color'])
    bump = N('ShaderNodeBump')                                  # the joints a hair's depth in; no pebbling
    bump.inputs['Strength'].default_value = 0.35
    bump.inputs['Distance'].default_value = 0.05
    L(math('MULTIPLY_ADD', joint.outputs['Result'], -1.0), bump.inputs['Height'])
    for l_ in list(bsdf.inputs['Normal'].links):
        nt.links.remove(l_)
    L(bump.outputs['Normal'], bsdf.inputs['Normal'])
    bsdf.inputs['Roughness'].default_value = 0.34               # waxed and ironed: a soft shine, not a mirror
    if 'Coat Weight' in bsdf.inputs:
        bsdf.inputs['Coat Weight'].default_value = 0.25
        bsdf.inputs['Coat Roughness'].default_value = 0.22
    return m


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
        # (a close seat: the welt ends at the breast, and round the heel the sole and the stack are trimmed nearly
        # flush with the counter, SEAT_OUT past it, as a bootmaker finishes a heel seat. John, 2026-11-21: the ledge
        # standing out at the back)
        f = np.maximum((r3 - welt + SEAT_OUT) / r3, (env + SEAT_OUT) / r3)
        for _ in range(3):
            f = (np.roll(f, 1) + 2 * f + np.roll(f, -1)) / 4
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
    z_top = z_under + WELD
    h = z_top - tread(arc[:, 1])
    n = max(1, int(np.ceil(float(h.max()) / lift)))
    mats = [female.mat(M_TOP_LIFT)] + [stack_material()] * (n - 1)
    arm = female.RIG['arm']
    out = []
    m = len(arc)

    # the tread against the seat, from the study's design d (the foot's rest frame): how far its back edge is set in
    # under the seat's, and its front edge behind the breast (the breast then leans back to it)
    d = heel.RESULT['outs'][side][0]
    block = 2.0 * (d['seat_y'] - d['breast_y'])
    mid = d['seat_y'] - SHAPE['rake'] * block
    back_in = max(0.0, (d['seat_y'] + 0.5 * block) - (mid + 0.5 * SHAPE['length'] * block))
    front_in = max(0.0, (mid - 0.5 * SHAPE['length'] * block) - d['breast_y'])

    def level(kk):
        z = np.minimum(tread(arc[:, 1]) + kk * lift, z_top) if kk < n else np.full(m, z_top)
        u = np.clip((z_top - z) / max(float(h.max()), 1e-6), 0, 1)        # 0 at the seat, 1 on the tread
        xy = arc.copy()
        xy[:, 0] = cx + (arc[:, 0] - cx) * (1.0 - (1.0 - SHAPE['width']) * u)
        ya, yz = y0 + front_in * u, yb - back_in * u                       # (the breast and the back, at this height)
        xy[:, 1] = ya + (arc[:, 1] - y0) * (yz - ya) / (yb - y0)
        return np.column_stack([xy, z])

    for i in range(n):
        lo, hi = level(i), level(i + 1)
        V = np.concatenate([lo, hi])
        F = [[a, (a + 1) % m, m + (a + 1) % m, m + a] for a in range(m)]
        F += [list(range(m))[::-1], list(range(m, 2 * m))]
        me = bpy.data.meshes.new('Heel lift %d.%s' % (i, side))
        me.from_pydata(V.tolist(), [], F)
        me.materials.append(mats[i])
        at = me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
        at.data.foreach_set('vector', V.ravel().astype(np.float32))
        at = me.attributes.new('stack_u', 'FLOAT', 'POINT')
        # (the height up the stack in lifts, true where the top lift is cut short at the seat)
        at.data.foreach_set('value', np.concatenate([np.full(m, float(i)), i + (hi[:, 2] - lo[:, 2]) / lift]).astype(np.float32))
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


def stand_check(cols):
    """Print, for each sheet column, where each boot's under side is: the lowest point of its sole and of its heel
    over the floor (0: standing on it; below 0: through it), and the sole's height under the toe and at its back."""
    for col in cols:
        female.pose(col)                       # (the build's frame handler is not on yet)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        for side in 'RL':
            row = []
            for nm in ('Boot sole.' + side, 'Heel lift 0.' + side):
                o = bpy.data.objects.get(nm)
                if o is None:
                    continue
                e = o.evaluated_get(dg)
                X = np.array([(e.matrix_world @ v.co)[:] for v in e.data.vertices])
                fw = X @ np.array([0.0, -1.0, 0.0])
                row.append('%s lowest %.2f (toe end %.2f, back end %.2f; x %.2f..%.2f)'
                           % (nm, X[:, 2].min(), X[np.argmax(fw), 2], X[np.argmin(fw), 2], X[:, 0].min(), X[:, 0].max()))
            print('heelboot stand col %d: %s' % (col, '; '.join(row)), flush=True)
        print('heelboot stand col %d: ankles R %s L %s' % (col, *[tuple(round(v, 2) for v in bpy.data.objects['Ankle.' + sd].matrix_world.translation) for sd in 'RL']), flush=True)
    female.pose(0)


def build():
    # (set on every build: a render worker that keeps its scene loaded resets the environment between jobs)
    os.environ['HEEL_PARTS'] = '1'
    os.environ['HEEL_WALK'] = '1'
    os.environ['FOOT_FORM'] = 'shod'
    heel.H_SEAT, heel.ARCH_FILL, heel.CUP = SHAPE['h'], SHAPE['arch'], SHAPE['cup']
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
    # she stands square in boots: both feet under the hips, each turned out a little (the base idle's left foot eased
    # forward and turned well out put the far boot higher in every side view and read as a foot off the floor. John,
    # 2026-11-21: fix the stand pose). EXPERIMENTAL: 7 degrees a foot, the usual figure for quiet standing, unsourced here.
    female.IDLE_STANCE = {'R': (0.0, 0.0, math.radians(7.0)), 'L': (0.0, 0.0, math.radians(7.0))}
    # and on the floor: the contact points are a sample of the tread, which leaves the sole a hair over the floor;
    # the ankle is let down by what is left under the standing foot
    female.pose(0)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    gap = min(float(min((o.evaluated_get(dg).matrix_world @ v.co).z for v in o.evaluated_get(dg).data.vertices))
              for o in [bpy.data.objects['Boot sole.R']] + stacks['R'])
    female.HEEL_UP += gap
    female.pose(0)
    bpy.context.view_layer.update()
    print('heelboot: the standing foot was %.2f over the floor, let down onto it' % gap, flush=True)
    if os.environ.get('HEELBOOT_CHECK'):       # (how each foot stands at the sheet's columns named: the idle is 0)
        stand_check([int(v) for v in os.environ['HEELBOOT_CHECK'].split(',')])
    return res
