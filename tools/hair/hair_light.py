"""Light through the hair from its own flow and density, for the Face Creator's strands (bangs_creator_asset.py).

The creator lit every strand point with one isotropic number: the strand points crowding a few probes outward from
the head (creator_asset.py's AO). That counts drawn strands, not hairs, ignores which way the light comes from, and so
shades a thin fan and a thick lock alike. Here the shade comes from the hair itself:

  - Per light, a deep opacity map (Lokovic & Veach 2000, deep shadow maps; Yuksel & Keyser 2008, deep opacity maps):
    the hair between a point and that light, counted as the area its hairs block. Each segment blocks
    hairs x diameter x length of the light's view; a point gets T = exp(-K_SCATTER x the blocked share of its light
    texel in front of it). So a dense lock shadows what lies under it on the light's side, a sparse fan lets light
    through, and it is per light: the creator's key, rim and fill (main.js), each lighting the strand by its own
    transmittance.
  - Ambient: the same transmittance averaged over AMB_DIRS directions spread over the sphere, each weighted by how far
    it faces out from the head at that point (the head itself is the creator's own shadow, not the hair's).

The flow enters through the segments: each is laid along its hair, and its length, hairs and diameter set what it
blocks, so a fan where the flow spreads blocks less per area than a lock where it gathers. HAIR_DIAM_UM 70 (scalp hair
50 to 100 um, Robbins, Chemical and Physical Behavior of Human Hair, ch. 1); K_SCATTER 0.6 is render.py's (light
that a hair blocks is partly scattered on forward through it: Marschner 2003's TT lobe), DRAFT.
"""
import numpy as np
import scalp_field as SF

MM = SF.MM
HAIR_DIAM_UM = 70.0
K_SCATTER = 0.6         # DRAFT, as render.py's deep_shadow
TEXEL_MM = 1.0          # DRAFT: the opacity map's texel
BIAS_MM = 1.5           # DRAFT: its depth bias
AMB_DIRS = 26
# the creator's lights (main.js: key at (-9, 12, 16) aimed at (0, 0.5 HH, 0), rim at (4, 14, -20), fill at (10, 2, 12),
# both aimed at the origin), from page (x, y up, z to the viewer) to the hair's frame (x, y back, z up)
HH = 8.6
LIGHTS_PAGE = dict(key=(-9.0, 12.0 - 0.5 * HH, 16.0), rim=(4.0, 14.0, -20.0), fill=(10.0, 2.0, 12.0))


def page_to_hair(v):
    v = np.asarray(v, float)
    return v[[0, 2, 1]] * np.array([1.0, -1.0, 1.0])


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def transmittance(P, block, L, texel=TEXEL_MM * MM):
    """T toward unit direction L (from the point to the light) at points P whose segments block `block` (area, head
    heights^2) each."""
    up = np.array([0, 0, 1.0]) if abs(L[2]) < 0.9 else np.array([1.0, 0, 0])
    r = unit(np.cross(up, L)); u2 = np.cross(L, r)
    i = np.floor((P @ r) / texel).astype(np.int64)
    j = np.floor((P @ u2) / texel).astype(np.int64)
    z = -(P @ L)                                       # smaller: nearer the light
    key = (i - i.min()) * (j.max() - j.min() + 1) + (j - j.min())
    # the hair in front: in the same texel, nearer the light by more than BIAS_MM (a deep opacity map's depth bias: a
    # sheet of hair met at a grazing angle otherwise shades its own outer surface with its neighbours in the texel)
    comp = key.astype(float) * 16.0 + (z - z.min()) / max(z.max() - z.min(), 1e-9) * 8.0
    o = np.argsort(comp)
    cs = np.r_[0, np.cumsum((block / texel ** 2)[o])]
    cso = comp[o]
    zr = max(z.max() - z.min(), 1e-9)
    lo = np.searchsorted(cso, key[o].astype(float) * 16.0, 'left')                 # the column's first point
    hi = np.searchsorted(cso, comp[o] - BIAS_MM * MM / zr * 8.0, 'left')            # the first within the bias
    acc = np.empty(len(o)); acc[o] = cs[np.maximum(hi, lo)] - cs[lo]
    return np.exp(-K_SCATTER * acc)


def light(strands, hairs_per_strand):
    """strands: list of (n_i, 3) polylines (head heights). Returns per point: T per creator light (dict) and ambient."""
    P = np.concatenate(strands)
    seg = []
    for p in strands:
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)
        seg.append(np.r_[d[:1], 0.5 * (d[1:] + d[:-1]), d[-1:]] if len(d) else np.zeros(1))
    seg = np.concatenate(seg)
    h = np.asarray(hairs_per_strand, float)
    if h.ndim:                                         # (per strand: to its points)
        h = np.repeat(h, [len(p) for p in strands])
    block = h * HAIR_DIAM_UM * 1e-3 * MM * seg
    out = {nm: transmittance(P, block, unit(page_to_hair(v))) for nm, v in LIGHTS_PAGE.items()}
    # ambient: directions over the sphere (Fibonacci), weighted by their facing out from the head
    k = np.arange(AMB_DIRS) + 0.5
    ph = np.arccos(1 - 2 * k / AMB_DIRS); th = np.pi * (1 + 5 ** 0.5) * k
    D = np.c_[np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)]
    O = unit(P - SF.CENTRE)
    num = np.zeros(len(P)); den = np.zeros(len(P))
    for d in D:
        w = np.clip(O @ d, 0, None)
        if w.max() <= 0:
            continue
        num += w * transmittance(P, block, d); den += w
    out['ambient'] = num / np.maximum(den, 1e-9)
    return out
