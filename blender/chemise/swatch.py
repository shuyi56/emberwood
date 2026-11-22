"""A bench for the cloth's hand: two small swatches in the solver, measured in numbers. Seconds a trial, where
settling the whole garment and rendering it takes minutes, so a cloth's settings are found here and only confirmed on
the garment.

  the strip    a strip of the cloth held flat by one end, the rest hanging free over the edge: the cloth-tester's
               cantilever. How far its end droops gives the bending length (Peirce's), the length of cloth that bends
               under its own weight: the one number for how heavy or light a cloth hangs. Fine handkerchief linen is
               about 1.5 to 2.5 cm, shirt linen 2.5 to 3.5, canvas 4 and over. (A '<' before it: limper than the
               shortest strip the grid can cut will show.)
  the curtain  a piece hung from a band along its top and fuller than the band by the garment's own share, as the
               chemise's cloth is gathered into its neckline. Read off it: how many folds it falls into, how wide and
               deep they are, and how crumpled it is (how much the surface wanders on its way down a fold, against how
               much it turns going across the folds: a clean fold runs straight down, near 0; crumpled paper is near 1).

Both are cut on the garment's own grid (drape.VOXEL), since the grid is the cloth's grain to the solver.

    python3 blender/chemise/swatch.py                      the garment's LINEN
    python3 blender/chemise/swatch.py bending=0.3,0.5 mass=0.25,0.4      every pairing of these, a row each
    python3 blender/chemise/swatch.py bending=0.3 --save DIR   and the settled swatches kept as DIR/*.blend"""
import itertools
import math
import os
import sys
import time

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solver  # noqa: E402

GRID = 0.2                   # (drape.VOXEL: not imported, so the bench does not need the body's modules)
CM = 165.0 / 63.0            # centimetres in a unit: the figure is 63 units tall where a woman is 1.65 m
FULL = 0.28                  # the curtain's fullness (drape.RATIOS['full'])


def _grid(nx, nz, origin, across, down):
    """A sheet of nx by nz points, GRID apart: its points [nz * nx, 3] and four-sided faces."""
    i, j = np.meshgrid(np.arange(nx), np.arange(nz))
    V = np.asarray(origin)[None, :] + GRID * (i.reshape(-1, 1) * np.asarray(across)[None, :] + j.reshape(-1, 1) * np.asarray(down)[None, :])
    F = [(j_ * nx + i_, j_ * nx + i_ + 1, (j_ + 1) * nx + i_ + 1, (j_ + 1) * nx + i_) for j_ in range(nz - 1) for i_ in range(nx - 1)]
    return V, F


def hang(V, F, pinned, P, full=None, itself=False, name='swatch'):
    """Let a swatch go and return where its points settle. pinned: 1 for the points held."""
    sc = bpy.context.scene
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(p) for p in V], [], F)
    o = bpy.data.objects.new(name, me)
    sc.collection.objects.link(o)
    g = o.vertex_groups.new(name='pin')
    g.add([int(i) for i in np.nonzero(pinned > 0)[0]], 1.0, 'REPLACE')
    m = o.modifiers.new('cloth', 'CLOTH')
    s, c = m.settings, m.collision_settings
    solver.weave(s, P)
    s.vertex_group_mass, s.pin_stiffness = 'pin', 1.0
    solver.let_out(o, s, full, P)
    c.use_collision = False
    c.use_self_collision, c.self_distance_min, c.self_friction = itself, P['self_gap'], 8.0
    m.point_cache.frame_start, m.point_cache.frame_end = 1, P['frames']
    sc.frame_end = max(sc.frame_end, P['frames'])
    for f in range(1, P['frames'] + 1):
        sc.frame_set(f)
    ev = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
    out = np.empty(len(V) * 3)
    ev.data.vertices.foreach_get('co', out)
    out = out.reshape(-1, 3)
    me.vertices.foreach_set('co', out.reshape(-1))     # (kept as it settled, for --save)
    o.modifiers.remove(m)
    sc.frame_set(1)
    return out


def strip(P, held=1.0):
    """The cantilever: the bending length in units, and the overhang it was read at. As the cloth-tester does it,
    the overhang is found at which the strip's end droops 41.5 degrees; the bending length is half of it."""
    seen = []
    for length in (0.4, 0.6, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6):
        nx, nz = 6, int(round((length + held) / GRID)) + 1
        V, F = _grid(nx, nz, (0.0, -0.5, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))    # (lying flat, running out along x)
        out = hang(V, F, (V[:, 0] <= held + 1e-6).astype(float), P, name='strip %.1f' % length)
        tip = out[V[:, 0] >= V[:, 0].max() - 1e-6].mean(0)
        seen.append((math.degrees(math.atan2(-tip[2], max(tip[0] - held, 1e-9))), length))
        if seen[-1][0] >= 41.5:
            break
    if len(seen) == 1 or seen[-1][0] < 41.5:        # (limper than the shortest strip can show, or stiffer than the longest)
        return 0.5 * seen[-1][1], seen[-1][1] * (-1.0 if len(seen) == 1 else 1.0)
    (a0, l0), (a1, l1) = seen[-2], seen[-1]
    over = l0 + (l1 - l0) * (41.5 - a0) / (a1 - a0)
    return 0.5 * over, over


def curtain(P, width=10.0, drop=7.0):
    """The gathered piece: (folds a unit of the band, a fold's depth in units, how crumpled, how much longer the
    cloth is along its threads than it was let out to, in percent)."""
    nx, nz = int(round(width / GRID)) + 1, int(round(drop / GRID)) + 1
    V, F = _grid(nx, nz, (-0.5 * width, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
    row = np.arange(len(V)) // nx
    band = np.clip(1.0 - (row - 1) / 4.0, 0.0, 1.0)            # (the band: the top two rows, fading out over four more)
    out = hang(V, F, (row == 0).astype(float), P, full=FULL * (1.0 - band), itself=True, name='curtain')
    S = out.reshape(nz, nx, 3)
    y = S[:, :, 1]
    rows = [int(nz * k) for k in (0.45, 0.65, 0.85)]
    folds, depth = [], []
    for r in rows:
        d = y[r, 3:-3] - np.convolve(y[r], np.ones(25) / 25.0, 'same')[3:-3]        # (the folds, less the sheet's slow sway)
        d = np.convolve(d, np.ones(3) / 3.0, 'same')
        up = np.nonzero((d[1:-1] > d[:-2]) & (d[1:-1] >= d[2:]) & (d[1:-1] > 0.03))[0]
        folds.append(len(up) / width)
        depth.append(float(np.percentile(d, 95) - np.percentile(d, 5)))
    body = S[nz // 3:, 3:-3]
    down = np.diff(body, 2, axis=0)[:, 1:-1]
    across = np.diff(body, 2, axis=1)[1:-1]
    crumple = float(np.sqrt((down ** 2).sum(2).mean()) / max(np.sqrt((across ** 2).sum(2).mean()), 1e-9))
    lens = np.linalg.norm(np.diff(S[8:], axis=1), axis=2).mean() / (GRID * (1.0 + FULL))
    return float(np.mean(folds)), float(np.mean(depth)), crumple, 100.0 * (float(lens) - 1.0)


def bench(P, save=None, tag=''):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    t0 = time.time()
    c, over = strip(P)
    per, depth, crumple, stretch = curtain(P)
    if save:
        os.makedirs(save, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(save, 'swatch%s.blend' % tag))
    return dict(bend_cm=c * CM, over_cm=abs(over) * CM, limp=over < 0, folds=per, fold_cm=CM / max(per, 1e-9), depth_cm=depth * CM, crumple=crumple,
                stretch=stretch, seconds=time.time() - t0)


def main(argv):
    save = argv[argv.index('--save') + 1] if '--save' in argv else None
    sweep = {k: [type(solver.LINEN[k])(float(x)) for x in v.split(',')]
             for k, v in (a.split('=') for a in argv if '=' in a)}
    keys = list(sweep)
    print(' '.join('%11s' % k for k in keys) + '   bend cm  at cm   fold cm  depth cm  crumple  stretch %     s')
    for n, vals in enumerate(itertools.product(*sweep.values())):
        r = bench(dict(solver.LINEN, **dict(zip(keys, vals))), save, '-%d' % n if keys else '')
        print(' '.join('%11g' % v for v in vals) + '  %s%7.2f  %5.1f   %7.2f  %8.2f  %7.2f  %9.1f  %4.0f'
              % ('<' if r['limp'] else ' ', r['bend_cm'], r['over_cm'], r['fold_cm'], r['depth_cm'], r['crumple'], r['stretch'], r['seconds']), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
