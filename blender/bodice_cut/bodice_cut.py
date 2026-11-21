"""Cut and fit, nothing else (John, 2026-11-21: "proper work flow is cut, fit, then fold, and finally, after the fold
looks right, you button"; "cut the bodice with the lapel in mind then fold"; "the material is wrong it's too thin").

The redingote's bodice on the bare mannequin, cut as a bodice that closes to the neck: two whole fronts, the right
lapped over the left, each with its own top corner (the cloth that will be turned back as its lapel). No lapel is
drafted on, no front is cut open, nothing is folded, and no buttons are set: those come after, in this order:

    python3 blender/base/views.py --figure bodice_cut --mannequin --turn --angles front --size 300 --samples 8 --save CUT.blend --out DIR
    python3 blender/collar/foldfront.py CUT.blend FOLDED.blend [options]      # the fold, then the buttons

THICK (in w, the neck's half-width): the cloth's thickness, a habit's fulled wool with its facing and interlining
(0.085 w, about 4.4 mm; the bodice was 0.045 w, 2.3 mm, and read as shirting at the lapel's edge).
Each front's points are marked on the mesh ('front_layer': 1 the right front, 2 the left front and its underlap)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('redingote_bodice', 'redingote', 'collar', 'vest', 'chemise', 'doublet', 'sorceress_pendant', 'shared', 'base'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
import bpy  # noqa: E402
import redingote_bodice as RB  # noqa: E402
import bodice  # noqa: E402

THICK = float(os.environ.get('BODICE_THICK', 0.085))


def build():
    import collar
    RB.SETTLE = False
    bodice.FRONT = None                               # the fronts whole to the neck: nothing cut open
    bodice.RATIOS['thickness'] = THICK
    bodice.buttons = lambda *a, **k: []               # buttoned last, after the fold
    res = RB.build()
    b = bpy.data.objects['Redingote bodice']
    M = RB.MEASURES
    lap = bodice.RATIOS['lap'] * M['w']
    n, counts, joined = collar.front_layers(b, M, lap)
    print('bodice_cut: two whole fronts (right front %d points, left front %d), the underlap joined at %d points; cloth %.3f thick (%.1f mm)'
          % (counts[0], counts[1], joined, THICK * M['w'], THICK * M['w'] * 26.2), flush=True)
    b['bodice_lap'] = float(lap)
    for k in ('button', 'button_step', 'button_dome'):
        b['bodice_' + k] = float(bodice.RATIOS[k])
    for m in bpy.data.materials:
        if m.name.startswith('bodice'):
            m.use_fake_user = True
    for c_ in bpy.data.objects:
        if c_.name.startswith('Necklace'):
            c_.hide_render = True
    return res
