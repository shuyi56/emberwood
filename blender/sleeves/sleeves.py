"""A standalone garment: the redingote's sleeves (ca. 1885-1895 riding habit) on the bare mannequin (the anatomy
thread's head-driven body), with nothing else worn. The sleeves are drafted in redingote/sleeve.py from the body's own
measurements and sewn into the shared armhole (redingote/armhole.py); this module supplies the body and the cloth.

    python3 blender/base/views.py --figure sleeves --mannequin --head --angles 3/4 --out DIR

--head frames the sleeves here (HEAD_CZ, HEAD_H below): from the shoulder's top to below the wrist. SLEEVE_SOLVE=1
lets the solver gather the heads (off by default: John's rule, 2026-11-19, a new layer is shown drawn first)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('redingote', 'chemise', 'vest', 'sorceress_pendant', 'shared', 'base'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
# (the bare mannequin: its own plain head, the gown's pieces and the necklace out of sight)
os.environ.setdefault('HEAD_GLB', '')
os.environ.setdefault('NO_DRESS', '1')
os.environ.setdefault('NO_ACC', '1')
import bpy  # noqa: E402
import numpy as np  # noqa: E402
import female  # noqa: E402
import sorceress_pendant as SP  # noqa: E402
import fit  # noqa: E402
import sleeve  # noqa: E402

# a riding habit's navy broadcloth (the reference, MoMu's habit of ca. 1885-1895: a blue so dark it reads near black,
# matte, with the faint sheen of a fulled, pressed face), and small black buttons
NAVY = int(os.environ.get('SLEEVE_NAVY', '1c2438'), 16)
M_WOOL = female.mat_def(90, 'sleeve wool', female.ramp(NAVY, 0.10), 0.0, surface='wool', closeup='wool', soft_edge=True)
M_BUTTON = female.mat_def(91, 'sleeve buttons', female.ramp(0x18181c, 0.10), 0.0, closeup='leather')
MEASURES = {}


def build():
    res = SP.build()
    body = bpy.data.objects['Skin body']
    head = bpy.data.objects.get('Head')
    heads = [o for o in (head.children_recursive if head else []) if o.type == 'MESH']
    for o in bpy.data.objects:                   # (the necklace's parts: NO_ACC hides only the chain)
        if o.name.startswith('Necklace'):
            o.hide_render = True
            o.visible_camera = o.visible_shadow = o.visible_diffuse = o.visible_glossy = False
    bpy.context.view_layer.update()
    P, _, aw = fit.body_arrays(body)
    hp = np.array([(o.matrix_world @ v.co)[:] for o in heads for v in o.data.vertices]) if heads else None
    m = fit.measure(P, aw, hp)
    MEASURES.update(m)
    sleeves, buttons, d = sleeve.build_sleeves(
        body, m, rig=bpy.data.objects.get('Rig'), material=female.mat(M_WOOL), button_material=female.mat(M_BUTTON),
        settle=os.environ.get('SLEEVE_SOLVE') == '1',
        colliders=[o for o in (bpy.data.objects.get('Hand.L'), bpy.data.objects.get('Hand.R')) if o])
    # the --head view on the garment: from the shoulder's top to below the cuff
    lo, hi = d.z_hem - 1.2 * m['w'], float(d.L['loop'][:, 2].max()) + 1.2 * m['w']
    os.environ.setdefault('HEAD_CZ', '%.2f' % (0.5 * (lo + hi)))
    os.environ.setdefault('HEAD_H', '%.2f' % (hi - lo))
    bpy.context.view_layer.update()
    return res
