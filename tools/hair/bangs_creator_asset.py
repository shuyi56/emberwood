"""The ponytail field's hair with these bangs, as a Face Creator groom asset (the hair fidelity mode's "strands"):
creator_asset.py's format, with the strands taken from the ponytail renderer's fibres (bangs_render.py's lines: the
bangs and their joined side hair) instead of a guide groom. There are no cards yet, so the asset's LOD list is empty
and the page offers only the strands for it.

    python3 tools/hair/bangs_creator_asset.py HEAD.json.gz BANGS_DIR OUT.json --field-cache CACHE [--strands 14000]
    python3 tools/face-creator/apply-hair-fidelity.py IN.html OUT.html ponytail=OUT.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ponytail_flow as PF
import ponytail_render as PR
import bangs_flow as BF
import creator_asset as CA
import cards as CD


def main():
    a = sys.argv[1:]
    head, bdir, out = a[:3]
    nmax = int(a[a.index('--strands') + 1]) if '--strands' in a else 14000
    BF.reuse_field_cache(a[a.index('--field-cache') + 1])
    PF.HEAD = head
    z = np.load(os.path.join(bdir, 'bangs-lines.npz'))
    mine = {int(r): l for r, l in zip(z['roots'], z['lines'])}
    PF.bang_line = lambda B, i, k=32: mine[int(i)]
    side = {}
    if 'tie_roots' in z.files:
        side.update({int(r): l for r, l in zip(z['tie_roots'], z['tie_lines'])})
        side.update({int(r): l for r, l in zip(z['lock_roots'], z['lock_lines'])})
    build = PF.build
    PF.build = lambda h: (lambda B: (BF.apply_side(B, side), B)[1])(build(h))
    paths, hairs, basin, B = PR.fibres(head, per_root=1)
    noise = None
    if '--noise' in a:                                                # the noising round (noise_round.py)
        import noise_round as NR
        paths, noise = NR.apply(paths, surf=B['surf'], scale=float(a[a.index('--noise') + 1]))
        print('noise', noise, file=sys.stderr)
    print('fibres', paths.shape, file=sys.stderr)
    rng = np.random.default_rng(5)
    keep = np.arange(len(paths))
    if len(keep) > nmax:                               # (a share of the fibres, each standing for more hairs)
        keep = np.sort(rng.choice(keep, nmax, replace=False))
    ps = [p[~np.isnan(p[:, 0])] for p in paths[keep]]
    ok = [k for k, p in enumerate(ps) if len(p) >= 2]
    ps = [ps[k] for k in ok]
    keep = keep[ok]
    P = np.concatenate(ps)
    counts = np.array([len(p) for p in ps])
    q = 2.5 / 32767
    from scipy.spatial import cKDTree
    tree = cKDTree(P[::2])
    centre = np.array([0.0, 0.0, 0.0])
    cap_dir = os.path.join(os.path.dirname(os.path.abspath(out)), 'cap')
    os.makedirs(cap_dir, exist_ok=True)
    CD.scalp_cap(head, cap_dir)
    print('cap done', file=sys.stderr)
    V, _, Fc = CA.read_obj(os.path.join(cap_dir, 'scalp-cap.obj'))
    centre = V.mean(0) - np.array([0, 0, 0.15])
    outS = P - np.where(P[:, 2:3] < centre[2], np.array([centre[0], centre[1], 0]) + np.array([0, 0, 1]) * P[:, 2:3], centre)
    outS /= np.maximum(np.linalg.norm(outS, axis=1, keepdims=True), 1e-6)
    cs = sum(tree.query_ball_point(P + outS * d, CA.AO_R, return_length=True) for d in CA.AO_STEPS)
    sao = np.clip(np.exp(-cs / (np.median(cs) / -np.log(CA.SAO_MID))), CA.SAO_MIN, 1.0)
    hsh = lambda v: ((v.astype(np.int64) * 2654435761 + 12345) >> 7) % 251
    ids = np.stack([hsh(basin[keep] * 7919 + keep // 40), hsh(keep)], 1).astype('u1')
    share = float(hairs.sum()) / len(ps)
    rep = dict(lods=[], guides=0, cap=dict(pos=CA.b64(V, '<f4'), idx=CA.b64(Fc, '<u4')),
               strands=dict(n=int(len(ps)), counts=CA.b64(counts, '<u2'), pos=CA.b64(np.round(P / q), '<i2'), q=q,
                            hairs_per_strand=round(share, 2), ao=CA.b64(np.round(sao * 255), 'u1'), ids=CA.b64(ids, 'u1')),
               source=dict(bangs=os.path.abspath(bdir), fibres=int(len(paths))))
    json.dump(rep, open(out, 'w'))
    print(out, os.path.getsize(out) // 1024, 'KB', len(ps), 'strands of', len(paths), 'fibres, %.1f hairs each' % share)


if __name__ == '__main__':
    main()
