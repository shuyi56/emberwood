"""The cotton's bench: two small swatches in the solver, measured in numbers, in seconds a trial, so COTTON is found
here and only confirmed on the petticoat.

  the strip    a strip held flat at one end, the rest stuck out over the edge (the cloth-tester's cantilever). Its
               end's droop gives Peirce's bending length, the length of cloth that bends under its own weight: fine
               lawn about 1.5 cm, shirt cotton 2 to 3, a starched petticoat longcloth 3 to 4.5, canvas over 5.
  the breadth  a breadth of the cloth hung from a band it is gathered into, as the petticoat's back breadth is (and the
               flounce onto its skirt): how many folds it falls into, how wide and deep, and how crumpled (how much the
               surface turns on its way down a fold, against across it: a clean flute near 0, crumpled paper near 1).

    python3 blender/petticoat/swatch.py                         the garment's COTTON
    python3 blender/petticoat/swatch.py bending=0.6,1.2 mass=0.3,0.45     every pairing, a row each"""
import math
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloth  # noqa: E402

GRID = float(os.environ.get('PETTI_GRID', '0.36'))     # (draft.GRID: the garment's own sampling is the cloth's grain)
CM = 165.0 / 63.0
GATHER = 1.6                 # the breadth's fullness against its band


def _sheet(nx, nz, origin, across, down, spacing=GRID):
    i, j = np.meshgrid(np.arange(nx), np.arange(nz))
    V = np.asarray(origin, float)[None, :] + spacing * (i.reshape(-1, 1) * np.asarray(across, float)[None, :] +
                                                       j.reshape(-1, 1) * np.asarray(down, float)[None, :])
    F = [(b * nx + a, b * nx + a + 1, (b + 1) * nx + a + 1, (b + 1) * nx + a) for b in range(nz - 1) for a in range(nx - 1)]
    return V, F


def _run(V, F, pinned, P, frames, cut=None, collide=None):
    """Settle a sheet put on at V; its rest lengths are V's, or with cut (the cloth as cut), cut's: the sheet then
    starts flat as cut and its held points are drawn to their places in V over the first frames (a shape key), as a
    gathering thread is pulled up."""
    sc = bpy.context.scene
    me = bpy.data.meshes.new('swatch')
    me.from_pydata([tuple(p) for p in (V if cut is None else cut)], [], F)
    o = bpy.data.objects.new('swatch', me)
    sc.collection.objects.link(o)
    if cut is not None:
        o.shape_key_add(name='cut')
        k = o.shape_key_add(name='on')
        k.data.foreach_set('co', np.asarray(V, float).reshape(-1))
        for f, v in ((1, 0.0), (20, 1.0)):
            k.value = v
            k.keyframe_insert('value', frame=f)
    g = o.vertex_groups.new(name='pin')
    g.add([int(i) for i in np.nonzero(pinned > 0)[0]], 1.0, 'REPLACE')
    m = o.modifiers.new('cloth', 'CLOTH')
    s, c = m.settings, m.collision_settings
    cloth.weave(s, P)
    s.vertex_group_mass, s.pin_stiffness = 'pin', 1.0
    c.use_collision = collide is not None
    c.use_self_collision = False
    m.point_cache.frame_start, m.point_cache.frame_end = 1, frames
    sc.frame_end = frames
    for f in range(1, frames + 1):
        sc.frame_set(f)
    ev = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
    out = np.empty(len(V) * 3)
    ev.data.vertices.foreach_get('co', out)
    bpy.data.objects.remove(o)
    bpy.data.meshes.remove(me)
    return out.reshape(-1, 3)


def strip(P, over=5.0, held=1.5, width=1.5, frames=90):
    """Peirce's bending length, in cm, from a strip over long, held flat for held at one end."""
    nx = int(round((over + held) / GRID)) + 1
    nz = int(round(width / GRID)) + 1
    V, F = _sheet(nx, nz, (0.0, 0.0, 10.0), (1, 0, 0), (0, 1, 0))
    pinned = (V[:, 0] <= held + 1e-6).astype(float)
    out = _run(V, F, pinned, P, frames)
    tip = out[V[:, 0] >= V[:, 0].max() - 1e-6].mean(0)
    edge = np.array([held, tip[1], 10.0])
    d = tip - edge
    ang = math.atan2(-d[2], max(d[0], 1e-6))
    ang = max(ang, 1e-3)
    c = over * (math.cos(ang / 2) / (8 * math.tan(ang))) ** (1 / 3)
    return c * CM, math.degrees(ang)


def breadth(P, band=6.0, depth=8.0, frames=110):
    """A breadth GATHER times its band, hung from it: (folds, fold width cm, fold depth cm, crumple)."""
    nx = int(round(band * GATHER / GRID)) + 1
    nz = int(round(depth / GRID)) + 1
    V, F = _sheet(nx, nz, (0.0, 0.0, 20.0), (1, 0, 0), (0, 0, -1))
    # put on: the top drawn in to the band in even gathers (as a gathering thread lays it), the rest let hang
    u = V[:, 0] / V[:, 0].max()
    k = int(round(band / 0.9))
    on = V.copy()
    on[:, 0] = u * band
    amp = 0.5 * (GATHER - 1.0) * band / (k * math.pi) * 1.4
    on[:, 1] = amp * np.sin(2 * math.pi * k * u)
    pinned = (V[:, 2] >= 20.0 - 1e-6).astype(float)
    # (the cloth's rest lengths are the cut's: the solver starts it drawn in and knows the flat sheet)
    out = _run(on, F, pinned, P, frames, cut=V)
    mid = out[np.abs(V[:, 2] - (20.0 - 0.6 * depth)) < 0.5 * GRID]
    mid = mid[np.argsort(mid[:, 0])]
    y = mid[:, 1] - mid[:, 1].mean()
    cross = np.nonzero(np.diff(np.sign(y)) != 0)[0]
    folds = max(1, len(cross) // 2)
    w = (mid[-1, 0] - mid[0, 0]) / folds
    dpt = float(np.ptp(y))
    # crumple: turning of the surface down a fold against across it
    grid = out.reshape(nz, nx, 3)
    dn = np.linalg.norm(np.diff(grid, axis=0), axis=2).mean()
    dz = np.diff(grid[:, :, 1], axis=0)
    dx = np.diff(grid[:, :, 1], axis=1)
    cr = float(np.abs(np.diff(dz, axis=0)).mean() / max(np.abs(np.diff(dx, axis=1)).mean(), 1e-9))
    return folds, w * CM, dpt * CM, cr, dn


def main(argv):
    trials = [{}]
    for a in argv:
        if '=' in a:
            k, vs = a.split('=', 1)
            trials = [dict(t, **{k: float(v)}) for t in trials for v in vs.split(',')]
    for t in trials:
        P = dict(cloth.COTTON, **t)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        c, ang = strip(P)
        f, w, d, cr, _ = breadth(P)
        print('swatch %-34s strip: bending length %4.1f cm (droop %2.0f deg) | breadth: %d folds %4.1f cm wide, '
              '%4.1f cm deep, crumple %.2f' % (' '.join('%s=%g' % kv for kv in t.items()) or 'COTTON', c, ang, f, w, d, cr),
              flush=True)


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
