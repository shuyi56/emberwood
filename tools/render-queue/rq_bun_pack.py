"""Pack a bun hold solve for the clay turnaround (blender/hair/bun_views.py): the head's surface (marching cubes on
the hair test's head field, cut at the neck), and the hair's strands as drawn, settled at rest and at the worst frame
of each motion, every point in the head's frame and head heights, with each strand's part of the bun (for the
colour-coded design pass). Read from the solve cache only: nothing is solved again.

    python3 rq_bun_pack.py STYLES_DIR OUT.npz [--bun JSON] [--poses run,spin,twist] [--tag before] [--style bun|twist]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rq_bun as B  # noqa: E402
import rq_hair as R  # noqa: E402

PARTS = ['cap and gather', 'core', 'collar wrap', 'wrap over the top', 'wrap crossing it', 'closing wrap',
         'escaping end', 'short ends and flyaways', 'held by a pin or the tie']


def head_mesh(W):
    from skimage.measure import marching_cubes
    G = W.head
    v, f, _, _ = marching_cubes(G.v, 0.0, step_size=2)
    v = G.lo + v * G.cell
    keep = v[:, 2] > W.neck_z - 0.02
    idx = -np.ones(len(v), int)
    idx[keep] = np.arange(keep.sum())
    f = idx[f]
    f = f[(f >= 0).all(1)]
    return v[keep].astype(np.float32), f.astype(np.int32)


PARTS_ROLL = ['cap and gather', 'roll core', 'twisted strands', 'seam fold']


def parts_roll(st, runs, K):
    """The French twist's parts: the roll's core, its twisted strands and the seam's fold; everything else is cap and gather."""
    part = np.zeros(len(runs), int)
    for k in K['body']:
        part[k] = 2
    part[K['core']] = 1
    for k in K['body']:
        if k != K['core'] and st['strands'][k]['w'].mean() < 0.012:
            part[k] = 3
    return part


PARTS_CURLY = ['cap and gather', 'core', 'gripped wraps and curls', '', '', '', '', 'loose curls and ends', 'held by a pin or the tie']


def parts_curly(st, runs, K):
    """A curly bun's parts: the core, the wraps the hold grips, and the loose curls and ends the test measures apart (their
    order is not the structural bun's, so its band-by-band split does not apply)."""
    part = np.zeros(len(runs), int)
    for k in K['body']:
        part[k] = 2
    part[K['core']] = 1
    for k in K['free']:
        part[k] = 7
    return part


def parts_of(st, runs, X0, K):
    """Each strand's design part: the knot's bands in the order the creator builds them (core, then 4 bands, each a
    tube or rope with its clump), the escaping end and the short ends; everything else is cap and gather."""
    part = np.zeros(len(runs), int)
    body = [k for k in K['body'] if k != K['core']]
    part[K['core']] = 1
    # (the bands come in order: the collar (one tube and its clump of 6), then three ropes (2 tubes and 6 each))
    sizes = [7, 8, 8, 8]
    k0 = 0
    for b, n in enumerate(sizes):
        for k in body[k0:k0 + n]:
            part[k] = 2 + b
        k0 += n
    lens = [np.linalg.norm(np.diff(X0[r], axis=0), axis=1).sum() for r in runs]
    for k in K['free']:
        part[k] = 6 if lens[k] > 0.11 and lens[k] < 0.25 and st['strands'][k]['w'].mean() > 0.008 else 7
    return part


def main():
    a = sys.argv[1:]

    def opt(k, d=None):
        return a[a.index(k) + 1] if k in a else d
    if opt('--bun'):
        B.configure(**json.loads(opt('--bun')))
    src, out = a[0], a[1]
    world = R.load_world(os.path.join(src, 'head.json.gz'))
    st = R.load_style(os.path.join(src, opt('--style', 'bun') + '.json.gz'))
    cache = R.cache_dir()
    W = R.World(world, st, cache)
    X0, Xs, runs, roots, thick, steps, cached = R.settle(st['strands'], W, cache)
    K = B.knot_of(st['strands'], X0, runs)
    walk = R.walk_keys()
    states = {'drawn': X0, 'rest': Xs}
    for p in opt('--poses', 'run,spin,twist').split(','):
        _, Fp, _ = R.check_pose(p, st['strands'], W, (X0, Xs, runs, roots, thick), walk, cache)
        rows = [B.knot_metrics(Xs if B.BUN['ref'] == 'rest' else X0, Y, K, W, thick) for Y in Fp['frames']]
        states[p] = Fp['frames'][int(np.argmax([r['sag'] + r['deform'] for r in rows]))]
    hv, hf = head_mesh(W)
    held = np.zeros(len(X0), bool)
    if len(B._STATE.get('hold', [])):
        held[B._STATE['hold']] = True
    np.savez_compressed(out, head_v=hv, head_f=hf, counts=np.array([len(r) for r in runs], np.int32),
                        thick=np.asarray(thick, np.float16), part=(parts_roll(st, runs, K) if B.BUN['roll'] else parts_curly(st, runs, K) if B.BUN['free_by'] == 'extent' else parts_of(st, runs, X0, K)).astype(np.int8),
                        held=held, names=np.array(list(states)), parts=np.array(PARTS_ROLL if B.BUN['roll'] else PARTS_CURLY if B.BUN['free_by'] == 'extent' else PARTS),
                        tag=opt('--tag', ''), bun=json.dumps(B.BUN),
                        **{'s_' + k: v.astype(np.float16) for k, v in states.items()})
    print(out, os.path.getsize(out), 'bytes;', len(hv), 'head points,', len(runs), 'strands, states', list(states))


if __name__ == '__main__':
    main()
