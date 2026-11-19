"""The Warden's cloak, rebuilt as a tailored garment on the bare mannequin: measured off the body (measure.py),
drafted flat (pattern.py), cut as cloth and sewn (sew.py), then hung by the project's fabric solver
(src/world/fabric-core.js, through fabric_settle.mjs) on its own (the body is a fixed surface under it; nothing
beneath it is re-solved).

    python3 blender/base/views.py --figure cloak --mannequin --tier draft --preset turn-clay --out DIR
                                                     (the cloak as sewn and fitted, no solver: a draft's settle is
                                                      left unsolved unless remembered; --settle runs it)
    CLOAK_OUT=DIR/cloak python3 blender/base/views.py --figure cloak --mannequin ...
                                                     (also writes the cutting sheet DIR/cloak.png and DIR/cloak.json)

The cut is a yoked circular cloak: front and back yoke fitted over the shoulders, front and back skirt slashed and
spread below the yoke for fullness, a stand collar, a bagged lining, topstitched front edges, yoke seam and hem, and
a bronze clasp with a chain at the throat. Every size comes from the measured body and H (the head's height), so a
different body gives a different pattern: CLOAK_BODY=<setting=value,...> changes the mannequin for a refit check
(anatomy's SH_OUT, HIPS_H, WAIST_W, SH_SLOPE, NECK_W ...)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for d in ('sorceress_pendant', 'shared', 'base'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
os.environ.setdefault('HEAD_GLB', '')
os.environ.setdefault('NO_DRESS', '1')
os.environ.setdefault('NO_ACC', '1')
for kv in filter(None, os.environ.get('CLOAK_BODY', '').split(',')):
    k, v = kv.split('=')
    os.environ[k.strip()] = v.strip()
import bpy  # noqa: E402
import numpy as np  # noqa: E402
import female  # noqa: E402
import sorceress_pendant as SP  # noqa: E402
import measure  # noqa: E402
import pattern  # noqa: E402
import sew  # noqa: E402

# her colours: a deep plum wool (apart from her dark-brown hair in hue, see CLAUDE.md), lined in madder red, bronze
M_WOOL = female.mat_def(77, 'cloak wool', female.ramp(0x3b2244, 0.1), 0.1, surface='wool', closeup='wool', form=True)
M_LINING = female.mat_def(78, 'cloak lining', female.ramp(0x8e2b22, 0.1), 0.05, closeup='cloth', form=True)
M_BRONZE = female.mat_def(79, 'cloak clasp', female.ramp(0xa8783a, 0.12), 0.0, closeup='gold')
M_THREAD = female.mat_def(80, 'cloak thread', female.ramp(0x2a1830, 0.06), 0.0, closeup='cloth')
SETTLE = os.environ.get('CLOAK_SOLVER', '1') != '0'
RESULT = {}


def build():
    res = SP.build()
    body = bpy.data.objects['Skin body']
    rig = bpy.data.objects.get('Rig') or female.RIG['arm']
    for o in bpy.data.objects:
        if o.name.startswith('Necklace'):
            o.hide_render = True
            o.visible_camera = o.visible_shadow = o.visible_diffuse = o.visible_glossy = False
    bpy.context.view_layer.update()
    H = SP.anatomy.head_height() or SP.anatomy.HEAD
    worn = [bpy.data.objects[n] for n in ('Skin body', 'Neck', 'Hand.L', 'Hand.R') if n in bpy.data.objects]
    M = measure.take(worn, H, rig)
    RESULT['measures'] = M.report()
    panels, seams = pattern.draft(M)
    RESULT['pattern'] = pattern.report(panels, seams, M)
    out = os.environ.get('CLOAK_OUT')
    if out:
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        pattern.layout(panels, seams, M, out + '.png')
        pattern.save(panels, seams, M, out + '.json', extra={'measures': RESULT['measures']})
    o = sew.build(M, panels, seams, rig, (M_WOOL, M_LINING))
    if SETTLE:
        # (the fabric solver on the cloak alone, against the body as it stands: remembered by layers.py, and left
        # unsolved in a draft unless --settle asks for it)
        sew.settle(o, rig, [body] + [b for b in worn if b is not body])
    sew.stitches(o, M, M_THREAD)
    sew.clasp(o, M, M_BRONZE)
    bpy.context.view_layer.update()
    return res
