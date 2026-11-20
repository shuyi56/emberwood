"""The harness as it is put on, layer over layer, on the bare mannequin: the linen chemise, the arming doublet over
it, and over that the bronze-plated vest (the breastplate) and the lamed shoulder plates.

    python3 blender/base/views.py --figure harness --mannequin --head --out DIR

Each layer is fitted over what is under it (fit.whole_piece's `over`): the vest is wrapped round the body and the
doublet together, the shoulder plates raised over the shoulder with everything on it. Nothing is settled again here:
the chemise is the shape it was last kept in (see CLAUDE.md, 'Layered clothes')."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for d in ('doublet', 'vest', 'chemise', 'sorceress_pendant', 'shared', 'base'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
os.environ.setdefault('HEAD_GLB', '')
os.environ.setdefault('NO_DRESS', '1')
os.environ.setdefault('NO_ACC', '1')
import bpy  # noqa: E402
import female  # noqa: E402
import sorceress_pendant as SP  # noqa: E402
import fit  # noqa: E402
import chemise  # noqa: E402
import drape  # noqa: E402
import doublet  # noqa: E402
import vest  # noqa: E402
import spaulder  # noqa: E402

LIGHT_SCALE = chemise.LIGHT_SCALE          # (the white sleeves need the same light: see chemise.py)
MEASURES = {}
# The plates are bare bronze: the glazed trim is left off (HARNESS_GLAZE=1 lays it on again).
GLAZE = os.environ.get('HARNESS_GLAZE') is not None
LACE_ROOM = 0.07           # the room left under the plates for the doublet's lacing, in w
SCYE_DROP = 0.45           # the plates' armhole: a little deeper than the gambeson's, and cut round (see
SCYE_ROUND = 0.20          # fit.trim_lines): this much wider than the arm's own depth front and back, in w
SHOULDERS = os.environ.get('HARNESS_NO_SHOULDERS') is None
LAYER_GAP = drape.LAYER_GAP


ARM_EXTRA = float(os.environ.get('HARNESS_ARM_EXTRA', '0.33'))   # the arms hang this much further out, in radians


def swing_arms(extra):
    """Both arms held further out from the body at the shoulder, by extra radians (about 19 degrees for 0.33). The
    layers' side wall (shirt, gambeson, vest, plates) runs from the torso's side at 4.6 out to 6.3, across an arm that
    hangs from 4.4 to 7.2; held out, the arm clears it and the sleeve is no longer caught under the breastplate.
    The pose only: the body's own shape and every garment's draft (made on the rest shape) are as they were, so the
    shirt's kept settled shape is still the one to use. Edits the shoulders' keyframes, every frame of them."""
    for side, sgn in (('R', 1.0), ('L', -1.0)):
        o = bpy.data.objects.get('Shoulder.' + side)
        if o is None:
            continue
        o.rotation_euler[1] += sgn * extra
        if o.animation_data and o.animation_data.action:
            for fc in female._fcurves(o.animation_data.action):
                if fc.data_path == 'rotation_euler' and fc.array_index == 1:
                    for kp in fc.keyframe_points:
                        kp.co[1] += sgn * extra
                        kp.handle_left[1] += sgn * extra
                        kp.handle_right[1] += sgn * extra
                    fc.update()
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()


def build():
    res = SP.build()
    swing_arms(ARM_EXTRA)
    body = bpy.data.objects['Skin body']
    rig = bpy.data.objects.get('Rig')
    head = bpy.data.objects.get('Head')
    heads = [o for o in (head.children_recursive if head else []) if o.type == 'MESH']
    for o in bpy.data.objects:                   # (the necklace's parts: NO_ACC hides only the chain)
        if o.name.startswith('Necklace'):
            o.hide_render = True
            o.visible_camera = o.visible_shadow = o.visible_diffuse = o.visible_glossy = False
    bpy.context.view_layer.update()
    coat, m, lo = doublet.wear(body, heads, bare=False)
    # the breastplate: the vest, fitted over the doublet and its lacing
    # (over the doublet's cloth alone, with room for its lacing in the ease: wrapped over the laces themselves the
    # plates took a dent at every cord and came out ragged at the hem)
    v, mv = fit.build_vest(body, heads, rig=rig, material=female.mat(vest.M_VEST), brass=female.mat(vest.M_BRASS),
                           over=fit.rest_tree([coat])[0], ratios=dict(fit.RATIOS, ease=fit.RATIOS['ease'] + LACE_ROOM, scye_drop=SCYE_DROP, scye_round=SCYE_ROUND))
    vest.warden_bronze()('vest bronze')
    if GLAZE:
        vest.glaze_coat('vest bronze')
    # the shirt's sleeve is kept outside the breastplate and its belt: hanging beside the body, it was pulled in
    # under the side of the plates (and the plates, raised over it, then fitted to that)
    # (HARNESS_SLEEVE_OUT=1: off by default. The arm hangs inside the breastplate's side wall below the armpit, so
    # sliding the sleeve out of it moves the whole sleeve about 2.5 units and widens the shoulder plates with it:
    # see the cross-sections at z 36, 41 and 44. The fix wants the arm or the plates' sides to give.)
    if os.environ.get('HARNESS_SLEEVE_OUT') is not None:
        chem = bpy.data.objects['Chemise']
        layers = [o for o in (bpy.data.objects.get(n) for n in ('Vest', 'Vest plates', 'Vest studs')) if o]
        drape.lie_outside(chem, layers, mv, gap=2.0 * LAYER_GAP * mv['w'])
    if SHOULDERS:
        worn = [o for o in bpy.data.objects if o.name.startswith(('Doublet', 'Vest', 'Chemise'))]
        spaulder.build_spaulders(body, mv, worn, rig=rig, plate=female.mat(vest.M_BRASS),
                                 sheet=os.environ.get('HARNESS_BLANKS'))
    MEASURES.update(mv)
    lo, hi = lo - 0.6 * m['w'], m['z_chin'] + 0.8 * m['w']
    os.environ.setdefault('HEAD_CZ', '%.2f' % (0.5 * (lo + hi)))
    os.environ.setdefault('HEAD_H', '%.2f' % (hi - lo))
    bpy.context.view_layer.update()
    return res
