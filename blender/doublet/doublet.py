"""An arming doublet over the linen chemise, on the bare mannequin (the anatomy thread's head-driven body), with none
of the gown's pieces and nothing over it. The doublet is drafted in cut.py from the body's own measurements; the
chemise (blender/chemise) is tucked under it and its sleeves and hem settled round it; this module supplies the body
and the doublet's cloth.

    python3 blender/base/views.py --figure doublet --mannequin --head --out DIR
    DOUBLET_BARE=1 ...        the doublet alone, nothing under it
    DOUBLET_SLEEVES=1 ...     with sleeves (the chemise's own sleeves do not yet fit under them: use with BARE)

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
# (the chemise's settled shape is kept here and used again: see drape.build_chemise)
os.environ.setdefault('CHEMISE_CACHE', os.path.join(HERE, '..', 'chemise', 'kept', 'under-doublet.npz'))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
import female  # noqa: E402
import sorceress_pendant as SP  # noqa: E402
import fit  # noqa: E402
import chemise  # noqa: E402
import drape  # noqa: E402
import cut  # noqa: E402
import sleeve  # noqa: E402

# Fustian (a linen warp, a cotton weft) dyed with madder: the commonest arming doublet. DOUBLET=rrggbb for another dye.
DYE = int(os.environ.get('DOUBLET', '6f2a25'), 16)
M_CLOTH = female.mat_def(73, 'doublet fustian', female.ramp(DYE, 0.12), 0.0, closeup='linen')
M_THREAD = female.mat_def(74, 'doublet thread', female.ramp(0xcdbf9c, 0.12), 0.0, closeup='cloth')
M_LACE = female.mat_def(75, 'doublet lace', female.ramp(0xb9a67c, 0.12), 0.0, closeup='cloth')
M_AIGLET = female.mat_def(76, 'doublet aiglet', female.ramp(0xb08a3c, 0.12), 0.0, closeup='gold')
MEASURES = {}
BARE = os.environ.get('DOUBLET_BARE') is not None
SLEEVES = os.environ.get('DOUBLET_SLEEVES') is not None      # (the doublet with its sleeves: see sleeve.py)
if not BARE:
    LIGHT_SCALE = chemise.LIGHT_SCALE          # (the white sleeves need the same light: see chemise.py)


def _rgb(h):
    return tuple(((h >> s) & 255) / 255.0 for s in (16, 8, 0))


# The cloth's look: the chemise's own weave (linen_texture.py), of a heavier thread (42 to the inch against the
# shift's 70), unbleached under the dye so the yarn's unevenness shows more, and duller: fustian has a soft nap.
FUSTIAN = dict(colour=tuple(c ** 2.2 for c in _rgb(DYE)), streak=0.10, yarn=0.6, rough=(0.74, 0.90), specular=0.25,
               sheen=0.25, relief=0.6, pitch=chemise.PITCH * 70.0 / 42.0)


# The shirt worn under the sleeved gambeson: closer to the body, and only a little of it showing, a band below the
# gambeson's cuffs and below its skirt.
SHIRT = dict(
    drape.RATIOS,
    loose=0.22,              # (0.50 alone: under a padded coat the shirt is taken in)
    flare=0.06,
    sleeve_full=0.20,
    sleeve_bell=0.04,
    full=0.16,
    hem=cut.RATIOS['hem_drop'] + 0.45,     # a band of it below the gambeson's skirt
    cuff=0.08,                             # and below its cuffs (the gambeson's ends 0.25 w above the wrist)
    cuff_in=0.30,
    plumb=0.0,               # cut to the body, not hung from the bust (see drape.Draft)
    thick=0.024,             # a heavier linen, three times the shift's: it lies in round folds where the thin one
                             # folded flat on itself, as paper does
    hand=dict(bending=0.6, self_gap=0.045),
)


def wear(body, heads, bare=BARE):
    """The doublet (and, unless bare, the chemise under it) on a body already built. Returns the doublet, the
    measures, and the height of the lowest hem."""
    # (with a sleeve sewn in, the shoulder's end is not laid down onto the shoulder: the sleeve carries the line on,
    # and a shoulder pinched down to the seam leaves the sleeve's head standing above it as a puff)
    ratios = dict(cut.RATIOS, lay=0.0) if SLEEVES else cut.RATIOS
    o, m, covered = cut.build_doublet(body, heads, rig=bpy.data.objects.get('Rig'), cloth=female.mat(M_CLOTH),
                                      thread=female.mat(M_THREAD), lace=female.mat(M_LACE), metal=female.mat(M_AIGLET),
                                      ratios=ratios)
    MEASURES.update(m)
    under = dict(covered=covered, gap=0.5 * cut.RATIOS['ease'] * m['w'], garment=o)
    if SLEEVES:
        sl = sleeve.build_sleeves(body, m, bpy.data.objects.get('Rig'), female.mat(M_CLOTH), o, base=cut.RATIOS)
        # (the shirt's sleeve goes inside the gambeson's: covered from a little above the sleeve's cuff up, and pressed
        # as close to the arm as the sleeve comes, where it lies between the arm and the body)
        z_cuff = min(min(v.co.z for v in s_.data.vertices) for s_ in sl)
        border = cut.RATIOS['border'] * m['w']
        # (and so is whatever of the shirt's body lies out past the gambeson's armhole over the shoulder, under the
        # sleeve's cap)
        hole = sleeve.sleeve_draft.armhole(np.array([v.co[:] for v in o.data.vertices]), [list(q.vertices) for q in o.data.polygons])
        x_in, z_lo = float(np.abs(hole[:, 0]).min()) - 0.3 * m['w'], float(hole[:, 2].min()) - 0.5 * m['w']

        def covered_arm(p, share, z=z_cuff + border):
            # (the shirt's sleeve is held whole, down to its end: it ends just past the gambeson's cuff, and nothing of
            # it hangs free from the cloth held inside)
            under_cap = (np.abs(p[:, 0]) > x_in) & (p[:, 2] > z_lo)
            return np.where(share >= 0.5, 1.0, np.where(under_cap, p[:, 2] - z, -1.0))
        under.update(sleeves=sl, covered_arm=covered_arm,
                     gap_arm=0.8 * sleeve.RATIOS['sleeve_pinch'] * m['w'])
    chemise.linen_closeup('doublet fustian', FUSTIAN)
    lo = m['z_hem']
    if not bare:
        # (the shirt has half the doublet's ease to lie in: the rest is the doublet's own lining and the lacing's pull)
        _, d = chemise.dress(body, m, under=under, ratios=SHIRT if SLEEVES else None)
        lo = min(lo, d.z_hem_worn)
    return o, m, lo


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
    o, m, lo = wear(body, heads)
    # the --head view on the garments: from under the lowest hem to the chin
    lo, hi = lo - 0.6 * m['w'], m['z_chin'] + 0.8 * m['w']
    os.environ.setdefault('HEAD_CZ', '%.2f' % (0.5 * (lo + hi)))
    os.environ.setdefault('HEAD_H', '%.2f' % (hi - lo))
    bpy.context.view_layer.update()
    fit.clearance(o, body, heads)
    return res
