"""A standalone garment: tall leather boots over the hose on the bare mannequin. The boots are drafted in last.py from
the legs' own measurements (the foot, ankle, calf and knee) as sheets of leather lapped and stitched on a sole, so
they re-fit with the body; this module builds the hose under them (blender/hose), sews the pieces (sew.py: the cloth
solver, the pieces' flat patterns, the seams drawn together, the stitches) and supplies the leather's look.

    python3 blender/base/views.py --figure boot --mannequin --turn --clay --out DIR
    python3 blender/base/views.py --figure boot --mannequin --head --set VIEW=3 --out DIR

VIEW frames the --head close-up as in hose.py: 0 the legs, 1 the hips, 2 the knee, 3 the ankle; and 4 the boots, from
under the sole to over the knee."""
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


M_THREAD = female.mat_def(76, 'boot thread', female.ramp(0x8a6a3a, 0.1), 0.0)
M_EYELET = female.mat_def(77, 'boot eyelet', female.ramp(0x9a7a3a, 0.15), 0.35)
M_LACE = female.mat_def(78, 'boot lace', female.ramp(0x1c1814, 0.1), 0.0)
SHAFT = 'lace'               # the shaft sewn on the foot (last.SHAFTS)
CREASE = True                # the ankle bent and the folds it makes set into the leather
SEW = True                   # the pieces sewn by the solver, and their stitches; False: the pieces as cut on the last


def build():
    if VIEW == 4:                # the boots alone: from under the sole to over the knee
        knee = female.HIP_Z - female.THIGH
        os.environ.setdefault('HEAD_CZ', '%.2f' % (0.5 * (knee + 1.0) - 0.8))
        os.environ.setdefault('HEAD_H', '%.2f' % (knee + 2.6))
    hose.VIEW, hose.FOOT = min(VIEW, 3), FOOT
    res = hose.build()
    body = bpy.data.objects['Skin body']
    rig = bpy.data.objects.get('Rig')
    ho = bpy.data.objects['Hose']
    leather = female.mat(M_LEATHER)
    boots = last.build_boots(body, hose.MEASURES, rig=rig, material=leather, shaft_name=SHAFT)
    if SEW:
        import sew
        for side in 'RL':
            sew.sew(boots[side], last.PIECES[side], rig, [ho])
            if CREASE:
                M = hose.MEASURES
                yc = 0.5 * (M['heel_y'] + M['toe_y']) + 0.35 * (M['heel_y'] - M['toe_y'])      # (the ankle's middle)
                sew.crease(boots[side], last.PIECES[side], rig, [ho],
                           zone=(M['ankle_z'] - 0.4, M['ankle_z'] + 0.14 * M['H'], yc))
            boots['stitches ' + side] = sew.stitches(boots[side], last.PIECES[side], rig=rig, material=female.mat(M_THREAD))
            boots['eyelets ' + side], boots['lace ' + side] = sew.lacing(
                boots[side], last.PIECES[side], rig=rig, H=hose.MEASURES['H'], metal=female.mat(M_EYELET),
                cord=female.mat(M_LACE))
    if os.environ.get('BOOT_PATTERN'):       # (the flat pattern sheet, saved as an image at this path)
        import pattern
        pattern.pairs(last.PIECES['R'])
        pattern.sheet(last.PIECES['R'], os.environ['BOOT_PATTERN'], title='Tall boot, right foot: the pattern laid flat')
    if os.environ.get('BOOT_CHECK'):
        sc = bpy.context.scene
        for f in [int(x) for x in os.environ['BOOT_CHECK'].split(',')]:
            sc.frame_set(f)
            for s in 'RL':
                last.clearance(boots[s], ho)
        sc.frame_set(0)
    bpy.context.view_layer.update()
    return res
