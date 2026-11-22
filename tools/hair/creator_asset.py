"""The groom as the Face Creator loads it: the hair fidelity mode (tools/face-creator/apply-hair-fidelity.py) shows a
hairstyle either as the creator sculpts it or as this groom, as game cards at three levels of detail or as strands.

    python3 tools/hair/creator_asset.py HEAD.json.gz GROOM_DIR CARDS_DIR OUT.json [--strands 8000]

Writes one JSON: the cards of each LOD (cards-lod<k>.obj from cards.py: positions, UVs, triangles), the scalp cap,
and the strands (children grown as render.py grows them, with their share of the hair). Positions stay in head
heights in the hair's frame (x across, y back, z up from the chin); the page turns them into its own units
(x 8.6, z up -> its y, y back -> its -z), the inverse of rq_hair_export.cjs. Arrays are base64 little-endian:
float32 for card positions and UVs, uint32 for triangles, int16 for strand points (scaled by `q`).
"""
import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def b64(a, dt):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dt).tobytes()).decode('ascii')


def read_obj(path):
    V, VT, F = [], [], []
    for line in open(path):
        if line.startswith('v '):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('vt '):
            VT.append([float(x) for x in line.split()[1:3]])
        elif line.startswith('f '):
            F.append([int(t.split('/')[0]) - 1 for t in line.split()[1:4]])
    return np.array(V), (np.array(VT) if VT else None), np.array(F)


def main():
    a = sys.argv[1:]
    head, gdir, cdir, out = a[:4]
    nstr = int(a[a.index('--strands') + 1]) if '--strands' in a else 8000
    rep = {'lods': []}
    cj = json.load(open(os.path.join(cdir, 'cards.json')))
    names = ['LOD 0', 'LOD 1', 'LOD 2']
    lod_files = sorted([f for f in os.listdir(cdir) if f.startswith('cards-lod') and f.endswith('.obj')],
                       key=lambda f: -int(f[9:-4]))
    for name, f in zip(names, lod_files):
        V, VT, F = read_obj(os.path.join(cdir, f))
        rep['lods'].append(dict(name=name, cards=int(f[9:-4]), triangles=int(len(F)),
                                pos=b64(V, '<f4'), uv=b64(VT, '<f4'), idx=b64(F, '<u4')))
    V, _, F = read_obj(os.path.join(cdir, 'scalp-cap.obj'))
    rep['cap'] = dict(pos=b64(V, '<f4'), idx=b64(F, '<u4'))
    # the strands: children as render.py grows them (blended within their lock, clumped to their guide, cut)
    import scalp_field as SF
    import render as RD
    surf = SF.Surface(SF.load_head(head))
    F = SF.build(surf)
    Gm = dict(np.load(os.path.join(gdir, 'groom.npz')))
    paths, share = RD.grow_children(F, Gm, nstr)
    keep = []
    for p in paths:
        p = p[~np.isnan(p[:, 0])]
        if len(p) >= 2:
            keep.append(p)
    counts = np.array([len(p) for p in keep])
    P = np.concatenate(keep)
    q = 2.5 / 32767
    rep['strands'] = dict(n=int(len(keep)), counts=b64(counts, '<u2'), pos=b64(np.round(P / q), '<i2'), q=q,
                          hairs_per_strand=round(float(share.mean() * Gm['n_hairs'].sum()), 2))
    rep['guides'] = int(len(Gm['roots']))
    rep['source'] = dict(groom=os.path.abspath(gdir), cards=os.path.abspath(cdir), card_report=cj['lods'])
    json.dump(rep, open(out, 'w'))
    print(out, os.path.getsize(out) // 1024, 'KB', len(keep), 'strands')


if __name__ == '__main__':
    main()
