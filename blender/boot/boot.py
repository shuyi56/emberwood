"""A standalone garment: tall leather boots over the hose on the bare mannequin. The boots are drafted in last.py from
the legs' own measurements (the foot, ankle, calf and knee), so they re-fit with the body; this module builds the
hose under them (blender/hose) and supplies the leather's look.

    python3 blender/base/views.py --figure boot --mannequin --turn --clay --out DIR
    python3 blender/base/views.py --figure boot --mannequin --head --set VIEW=3 --out DIR

VIEW frames the --head close-up as in hose.py: 0 the legs, 1 the hips, 2 the knee, 3 the ankle."""
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
import female  # noqa: E402
import hose  # noqa: E402
import last  # noqa: E402

VIEW = 0
FOOT = 'footed'
MEASURES = hose.MEASURES
M_LEATHER = female.mat_def(75, 'boot leather', female.ramp(0x4a2f1f, 0.1), 0.24, surface='leather', closeup='leather')


def build():
    hose.VIEW, hose.FOOT = VIEW, FOOT
    res = hose.build()
    body = bpy.data.objects['Skin body']
    rig = bpy.data.objects.get('Rig')
    ho = bpy.data.objects['Hose']
    boots = last.build_boots(body, hose.MEASURES, rig=rig, material=female.mat(M_LEATHER), colliders=[ho])
    if os.environ.get('BOOT_CHECK'):
        sc = bpy.context.scene
        for f in [int(x) for x in os.environ['BOOT_CHECK'].split(',')]:
            sc.frame_set(f)
            for o in boots.values():
                last.clearance(o, ho)
        sc.frame_set(0)
    bpy.context.view_layer.update()
    return res
