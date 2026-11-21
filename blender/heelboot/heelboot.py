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
}


M_THONG = female.mat_def(79, 'boot thong', female.ramp(0x6b4a30, 0.1), 0.3)


def seat_heel(side):
    """The stacked heel let out to the boot's own sole: its back and sides brought flush with the sole's edge round
    the back end of the boot (the study cuts it on the bare foot's outline, inside the boot's leather and welt).
    The breast, the stack's front face, stays where the study put it."""
    sole = bpy.data.objects.get('Boot sole.' + side)
    stack = [o for o in bpy.data.objects if o.name.startswith('Heel ') and o.name.endswith('.' + side)]
    if sole is None or not stack:
        return
    Sv = np.array([(sole.matrix_world @ v.co)[:] for v in sole.data.vertices])
    X = [np.array([v.co[:] for v in o.data.vertices]) for o in stack]
    A = np.concatenate(X)
    y0, y1 = float(A[:, 1].min()), float(A[:, 1].max())          # the breast, the stack's back
    yb = float(Sv[:, 1].max())                                   # the back end of the sole
    up_ = bpy.data.objects.get('Boot.' + side)
    if up_ is not None:                                          # (the counter stands out past the sole: the boot ends there)
        Uv = np.array([(up_.matrix_world @ v.co)[:] for v in up_.data.vertices])
        yb = max(yb, float(Uv[Uv[:, 2] < Sv[:, 2].max() + 1.2][:, 1].max()))
    ky = (yb - y0) / (y1 - y0)
    top = A[A[:, 2] > A[:, 2].max() - 0.25]                      # (the seat: the stack's top face)
    ym = y0 + 0.45 * (y1 - y0)
    t = top[np.abs(top[:, 1] - ym) < 0.12 * (y1 - y0)]
    ys = y0 + (ym - y0) * ky
    b = Sv[np.abs(Sv[:, 1] - ys) < 0.12 * (yb - y0)]
    cx0, w0 = 0.5 * (t[:, 0].min() + t[:, 0].max()), t[:, 0].max() - t[:, 0].min()
    cx1, w1 = 0.5 * (b[:, 0].min() + b[:, 0].max()), b[:, 0].max() - b[:, 0].min()
    kx = w1 / w0
    for o, V in zip(stack, X):
        V = V.copy()
        V[:, 1] = y0 + (V[:, 1] - y0) * ky
        V[:, 0] = cx1 + (V[:, 0] - cx0) * kx
        o.data.vertices.foreach_set('co', V.ravel())
        o.data.update()
    print('heelboot %s: the heel let out to the sole: %.2f longer at the back, %.0f%% wider'
          % (side, yb - y1, 100 * (kx - 1)), flush=True)


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
    for side in 'RL':
        seat_heel(side)
    leather = female.mat(boot.M_LEATHER)
    # (the lace and the eyelets in the leather's own tone, a shade down: near black, they drew a dark line down the
    # boot's front edge in every side view)
    for o in bpy.data.objects:
        if o.name.startswith(('Boot lace', 'Boot eyelet')) or 'lace' in o.name.lower() or 'eyelet' in o.name.lower():
            o.data.materials.clear()
            o.data.materials.append(female.mat(M_THONG))
    for o in bpy.data.objects:                 # (stacked leather, in the boot's own)
        if o.name.startswith('Heel '):
            o.data.materials.clear()
            o.data.materials.append(leather)
    # what the foot stands on: the boot's sole and the stack's tread, behind the toe joint
    az = female.ANKLE_Z
    tread = []
    for o in bpy.data.objects:
        if o.name.endswith('.R') and (o.name.startswith('Boot sole') or o.name.startswith('Heel top lift 0')):
            Z = np.array([(o.matrix_world @ v.co)[:] for v in o.data.vertices])
            Z = Z[-Z[:, 1] <= female.FOOT_BALL]
            tread += [(float(-z[1]), float(z[2] - az)) for z in Z[::7]]
    female.HEEL_CONTACT = tread
    root.location.z = up
    bpy.context.view_layer.update()
    print('heelboot: %d contact points, raised %.2f' % (len(tread), up), flush=True)
    return res
