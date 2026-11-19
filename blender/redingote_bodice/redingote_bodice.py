"""A standalone garment: the redingote's bodice on the bare mannequin (the anatomy thread's head-driven body), with
none of the gown's pieces and nothing over it. The bodice is drafted in redingote/bodice.py from the body's own
measurements (and over the corset, when one is built under it: CORSET below); this module supplies the body, the
wool's look and the buttons'.

    python3 blender/base/views.py --figure redingote_bodice --mannequin --head --out DIR

--head frames the garment here (HEAD_CZ, HEAD_H below): from under the basque to the chin.

Settings (--set NAME=VALUE): DYE (the wool's colour), SETTLE (True or False: settle the basque in the cloth solver or not; by default only when nothing is layered
under it),
CORSETED (False: on the body as built, without the corset's drawing in). To check that it re-fits, change the
mannequin itself before it builds, through anatomy's own settings in the environment (WAIST_W, HIPS_H, CANON_H)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for d in ('redingote', 'vest', 'chemise', 'doublet', 'sorceress_pendant', 'shared', 'base'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
# (the bare mannequin: its own plain head, the gown's pieces and the necklace out of sight)
os.environ.setdefault('HEAD_GLB', '')
os.environ.setdefault('NO_DRESS', '1')
os.environ.setdefault('NO_ACC', '1')
import bpy  # noqa: E402
import female  # noqa: E402
import sorceress_pendant as SP  # noqa: E402
import fit  # noqa: E402
import bodice  # noqa: E402

# A dark navy wool habit cloth (a fulled broadcloth: no weave to be seen, a soft nap). DYE=0xrrggbb for another.
DYE = 0x121621              # (picked on swatch.py's bench: 1d2436 read slate blue under the views' lights)
# John's rule for layers (2026-11-19): a layer added over others is first shown without the solver, on the layers
# below as they are; only after that is seen is another round run with the solver on both. So the basque is settled
# only on its own (no corset or skirt mesh in the scene) unless SETTLE is set True; False never settles it.
SETTLE = None
CORSET = 'Corset'            # an object of this name in the scene is worn under the bodice, and the bodice fits over it
CORSETED = True              # without one, the body drawn in by the corset's shared numbers (shared/habit.py), and the
                             # bodice over the corset's shell; False: the body as built, uncorseted
SKIRT = 'Riding skirt'       # the skirt thread's mesh: the basque hangs over its waistband
MEASURES = {}


def materials():
    cloth = female.mat_def(90, 'bodice wool', female.ramp(DYE, 0.10), 0.0, surface='wool', closeup='wool', form=True)
    btn = female.mat_def(91, 'bodice button', female.ramp(0x0e0f12, 0.12), 0.0, closeup='leather')    # (black horn)
    return female.mat(cloth), female.mat(btn)


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
    cloth, btn = materials()
    under = [o for o in bpy.data.objects if o.type == 'MESH' and o.name.split('.')[0] == CORSET]
    corset = None
    if CORSETED and not under:
        import habit
        corset = habit.measure(body)
        moved = habit.compress_body(body, corset)
        print('redingote bodice: the body drawn in by the corset, by up to %.2f; waist %.2f -> %.2f' % (
            moved, corset.natural_waist, corset.girth_waist_corseted), flush=True)
    # (the basque settles on the hands and on the skirt's waist, when the skirt is built first: it ends under the basque)
    hands = [o for o in (bpy.data.objects.get('Hand.L'), bpy.data.objects.get('Hand.R'), bpy.data.objects.get(SKIRT)) if o]
    stacked = bool(under) or bpy.data.objects.get(SKIRT) is not None
    settle = (not stacked) if SETTLE is None else bool(SETTLE)
    if stacked and not settle:
        print('redingote bodice: over other layers, so the basque is shown as drafted (SETTLE=True to settle it)', flush=True)
    o, m = bodice.build_bodice(body, heads, rig=bpy.data.objects.get('Rig'), cloth=cloth, button=btn, under=under,
                               colliders=hands, settle=settle, corset=corset)
    MEASURES.update(m)
    lo, hi = m['z_hem'] - 0.6 * m['w'], m['z_chin'] + 0.8 * m['w']
    os.environ.setdefault('HEAD_CZ', '%.2f' % (0.5 * (lo + hi)))
    os.environ.setdefault('HEAD_H', '%.2f' % (hi - lo))
    bpy.context.view_layer.update()
    fit.clearance(o, body, heads)
    return res
