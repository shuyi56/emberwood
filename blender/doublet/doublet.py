"""A standalone garment: an arming doublet on the bare mannequin (the anatomy thread's head-driven body), with none
of the gown's pieces, nothing under it and nothing over it. The doublet is drafted in cut.py from the body's own
measurements; this module supplies the body and the cloth's look.

    python3 blender/base/views.py --figure doublet --mannequin --head --out DIR

--head frames the garment here (HEAD_CZ, HEAD_H below): from its hem to the chin."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for d in ('vest', 'chemise', 'sorceress_pendant', 'shared', 'base'):
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
import chemise  # noqa: E402
import cut  # noqa: E402

# Fustian (a linen warp, a cotton weft) dyed with madder: the commonest arming doublet. DOUBLET=rrggbb for another dye.
DYE = int(os.environ.get('DOUBLET', '6f2a25'), 16)
M_CLOTH = female.mat_def(73, 'doublet fustian', female.ramp(DYE, 0.12), 0.0, closeup='linen')
M_THREAD = female.mat_def(74, 'doublet thread', female.ramp(0xcdbf9c, 0.12), 0.0, closeup='cloth')
M_LACE = female.mat_def(75, 'doublet lace', female.ramp(0xb9a67c, 0.12), 0.0, closeup='cloth')
M_AIGLET = female.mat_def(76, 'doublet aiglet', female.ramp(0xb08a3c, 0.12), 0.0, closeup='gold')
MEASURES = {}


def _rgb(h):
    return tuple(((h >> s) & 255) / 255.0 for s in (16, 8, 0))


# The cloth's look: the chemise's own weave (linen_texture.py), of a heavier thread (42 to the inch against the
# shift's 70), unbleached under the dye so the yarn's unevenness shows more, and duller: fustian has a soft nap.
FUSTIAN = dict(colour=tuple(c ** 2.2 for c in _rgb(DYE)), streak=0.10, yarn=0.6, rough=(0.74, 0.90), specular=0.25,
               sheen=0.25, relief=0.6, pitch=chemise.PITCH * 70.0 / 42.0)


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
    o, m = cut.build_doublet(body, heads, rig=bpy.data.objects.get('Rig'), cloth=female.mat(M_CLOTH),
                             thread=female.mat(M_THREAD), lace=female.mat(M_LACE), metal=female.mat(M_AIGLET))
    MEASURES.update(m)
    chemise.linen_closeup('doublet fustian', FUSTIAN)
    # the --head view on the garment: from under its hem to the chin
    lo, hi = m['z_hem'] - 0.6 * m['w'], m['z_chin'] + 0.8 * m['w']
    os.environ.setdefault('HEAD_CZ', '%.2f' % (0.5 * (lo + hi)))
    os.environ.setdefault('HEAD_H', '%.2f' % (hi - lo))
    bpy.context.view_layer.update()
    fit.clearance(o, body, heads)
    return res
