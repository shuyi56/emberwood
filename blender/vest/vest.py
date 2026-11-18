"""A standalone garment: a fitted leather vest with a stand collar, on the bare mannequin (the anatomy thread's
head-driven body), with none of the gown's pieces. The vest itself is drafted in fit.py from the body's own
measurements; this module only supplies the body and the leather.

    python3 blender/base/views.py --figure vest --mannequin --head --clay --angles 3/4 --out DIR

--head frames the bust and neck here (HEAD_CZ, HEAD_H below), which is all this garment needs to be judged by."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'sorceress_pendant'))
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
sys.path.insert(0, os.path.join(HERE, '..', 'base'))
# (the bare mannequin: its own plain head, the gown's pieces and the necklace out of sight)
os.environ.setdefault('HEAD_GLB', '')
os.environ.setdefault('NO_DRESS', '1')
os.environ.setdefault('NO_ACC', '1')
import bpy  # noqa: E402
import female  # noqa: E402
import sorceress_pendant as SP  # noqa: E402
import fit  # noqa: E402

M_VEST = female.mat_def(70, 'vest leather', female.ramp(int(os.environ.get('VEST', '8a5a34'), 16), 0.12), 0.0,
                        surface='leather', closeup='leather')
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
    vest, m = fit.build_vest(body, heads, rig=bpy.data.objects.get('Rig'), material=female.mat(M_VEST))
    MEASURES.update(m)
    # the --head view on the bust and neck: from under the bust to the chin, centred between them
    lo, hi = m['z_bust'] - 1.6 * m['w'], m['z_chin'] + 1.1 * m['w']
    os.environ.setdefault('HEAD_CZ', '%.2f' % (0.5 * (lo + hi)))
    os.environ.setdefault('HEAD_H', '%.2f' % (hi - lo))
    bpy.context.view_layer.update()
    fit.clearance(vest, body, heads)
    return res
