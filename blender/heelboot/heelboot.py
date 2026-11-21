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
    sp = boot.hose.SP
    keep = sp.build
    sp.build = lambda: res                      # (the figure is built already)
    try:
        boot.build()
    finally:
        sp.build = keep
    leather = female.mat(boot.M_LEATHER)
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
