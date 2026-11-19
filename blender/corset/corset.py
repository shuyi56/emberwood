"""A standalone garment: the redingote's corset, on the bare mannequin (the anatomy thread's head-driven body) with
none of the gown's pieces. The body is measured (habit.measure), drawn in to the corseted shape (habit.compress_body:
the flesh is compressed, not clipped through), and the corset is drafted on that (corset_garment.py).

    python3 blender/base/views.py --figure corset --mannequin --turn --clay --out DIR

Settings (env): WAIST_REDUCTION, WAIST_W / HIPS_H / HEAD_FIT_S (to try an altered body), NO_CORSET=1 (the bare body),
UNCOMPRESSED=1 (the corset on the natural body, to see what the compression does)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'sorceress_pendant'))
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
sys.path.insert(0, os.path.join(HERE, '..', 'base'))
os.environ.setdefault('HEAD_GLB', '')
os.environ.setdefault('NO_DRESS', '1')
os.environ.setdefault('NO_ACC', '1')
import bpy  # noqa: E402
import female  # noqa: E402
import sorceress_pendant as SP  # noqa: E402
import habit  # noqa: E402

if os.environ.get('WAIST_REDUCTION'):
    habit.WAIST_REDUCTION = float(os.environ['WAIST_REDUCTION'])
MEASURES = {}
# the --turn/--head frame: the torso, from the hips to the shoulders
HEAD_CZ = float(os.environ.get('HEAD_CZ', '40.0'))
HEAD_H = float(os.environ.get('HEAD_H', '22.0'))


def build():
    res = SP.build()
    for o in bpy.data.objects:                   # (the necklace's parts: NO_ACC hides only the chain)
        if o.name.startswith('Necklace'):
            o.hide_render = True
            o.visible_camera = o.visible_shadow = o.visible_diffuse = o.visible_glossy = False
    body = bpy.data.objects['Skin body']
    bpy.context.view_layer.update()
    M = habit.measure(body)
    MEASURES.update(M.report())
    print('CORSET measures', {k: round(v, 3) for k, v in MEASURES.items()}, flush=True)
    if not os.environ.get('UNCOMPRESSED'):
        mv = habit.compress_body(body, M)
        print('CORSET compress_body: largest move %.3f' % mv, flush=True)
    if not os.environ.get('NO_CORSET'):
        habit.build_corset(body, M)
    bpy.context.view_layer.update()
    return res
