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
# the neckline redrafted to the collar (John, 2026-11-22: "you have to redraft with this new neckline so the collar fits
# better"): round the sides and back the bodice's neck edge is drawn in to this far off the neck (in model units), as
# the notch collar's stand had been drawn in on its own; 0 keeps the old neckline
NECK_HUG = float(os.environ.get('BODICE_NECK_HUG', 0.05))
NECK_DROP = float(os.environ.get('BODICE_NECK_DROP', 2.0))     # fading out over this far down from the neck edge
NECK_FROM = [float(x) for x in os.environ.get('BODICE_NECK_FROM', '0.9,1.4').split(',')]   # (angle round from the front: none before the first, all past the second; the lapels' top corners lie before)


def neck_fit(b, M, hug=NECK_HUG, drop=NECK_DROP, log=print):
    """The neck edge drawn in toward the neck round the sides and back. Each point near the neckline moves level
    toward the neck until it lies hug off it, fully at the edge and fading out drop below it, and not at all over the
    front, where the fronts turn back as the lapels (from the roll lines' tops forward). The move is evened out over
    the mesh so the shoulders take it up as one smooth swell, then kept off the skin."""
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    if hug <= 0:
        return
    sk = bpy.data.objects['Skin body']; sw = np.array(sk.matrix_world)
    SV = np.array([v.co[:] for v in sk.data.vertices]) @ sw[:3, :3].T + sw[:3, 3]
    ST = BVHTree.FromPolygons([Vector(p) for p in SV], [tuple(p.vertices) for p in sk.data.polygons])
    mw = np.array(b.matrix_world); mi = np.linalg.inv(mw)
    W = np.array([v.co[:] for v in b.data.vertices]) @ mw[:3, :3].T + mw[:3, 3]
    yc = M['yc']
    ang = np.arctan2(-W[:, 0], -(W[:, 1] - yc))
    neck = bodice.neckline(M, bodice.RATIOS)
    zn = neck(ang)
    def ss(a, b_, x):
        t = np.clip((x - a) / (b_ - a), 0, 1); return t * t * (3 - 2 * t)
    w = ss(zn - drop, zn, W[:, 2]) * ss(NECK_FROM[0], NECK_FROM[1], np.abs(ang))
    D = np.zeros_like(W)
    for i in np.nonzero(w > 1e-3)[0]:
        q = np.array(ST.find_nearest(Vector(W[i]))[0]); dv = W[i] - q; dv[2] = 0.0; dh = np.linalg.norm(dv)
        if dh > hug: D[i] = -w[i] * (1 - hug / dh) * dv
    # even the move out over the mesh (edge neighbours), so it leaves no ridge
    E = np.array([e.vertices[:] for e in b.data.edges])
    nbr = [[] for _ in range(len(W))]
    for a_, c_ in E: nbr[a_].append(c_); nbr[c_].append(a_)
    act = np.nonzero(w > 1e-3)[0]
    for _ in range(20):
        D2 = D.copy()
        for i in act:
            if nbr[i]: D2[i] = 0.5 * D[i] + 0.5 * D[nbr[i]].mean(0)
        D = D2
    P = W + D
    for i in act:
        loc, nor, _, dist = ST.find_nearest(Vector(P[i]))
        if loc is not None and dist < 0.05:
            d = P[i] - np.array(loc); n = d / dist if dist > 1e-7 else np.array(nor); P[i] = np.array(loc) + n * 0.05
    L = P @ mi[:3, :3].T + mi[:3, 3]
    for i in act: b.data.vertices[i].co = L[i]
    b.data.update()
    mv = np.linalg.norm(P - W, axis=1)
    log('bodice_cut: the neckline redrafted to the collar, drawn in to %.2f off the neck round the sides and back: %d points moved, %.2f at most (%.1f mm)'
        % (hug, int((mv > 1e-4).sum()), mv.max(), 26.2 * mv.max()), flush=True)


def build():
    import collar
    RB.SETTLE = False
    bodice.FRONT = None                               # the fronts whole to the neck: nothing cut open
    bodice.RATIOS['thickness'] = THICK
    bodice.buttons = lambda *a, **k: []               # buttoned last, after the fold
    res = RB.build()
    b = bpy.data.objects['Redingote bodice']
    M = RB.MEASURES
    neck_fit(b, M)
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
