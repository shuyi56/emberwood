"""The groom as the Face Creator loads it: the hair fidelity mode (tools/face-creator/apply-hair-fidelity.py) shows a
hairstyle either as the creator sculpts it or as this groom, as game cards at three levels of detail or as strands.

    python3 tools/hair/creator_asset.py HEAD.json.gz GROOM_DIR CARDS_DIR OUT.json [--strands 8000] [--body ranger|warden]

Writes one JSON: the cards of each LOD (cards-lod<k>.obj from cards.py: positions, UVs, triangles), the scalp cap,
and the strands (children grown as render.py grows them, with their share of the hair). Positions stay in head
heights in the hair's frame (x across, y back, z up from the chin); the page turns them into its own units
(x 8.6, z up -> its y, y back -> its -z), the inverse of rq_hair_export.cjs. Arrays are base64 little-endian:
float32 for card positions and UVs, uint32 for triangles, int16 for strand points (scaled by `q`).

The bake: each card vertex carries `ao`, the light that reaches it through the hair, baked from the strands. A vertex
deep in the hair (strand points crowd the space just outside it) is dark, one on the outer surface is lit. This is the
usual card bake's occlusion pass (80 Level, "Tips and tricks on hair for games": AO and root-to-tip gradient baked into
the cards), done per vertex because the page has one shared strand texture.
"""
import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


AO_R = 0.03                        # DRAFT: the probe's radius, head heights (about 7 mm)
AO_STEPS = (0.03, 0.07, 0.12)      # DRAFT: probes outward from the vertex, head heights
AO_MID = 0.75                      # STYLE: the light through the hair at the median card vertex
AO_MIN = 0.35                      # STYLE: the deepest hair keeps a third of the light, so it never goes black
# The strands' own self-shadowing: Scheuermann 2004 ("Practical Real-Time Hair Rendering and Shading", SIGGRAPH
# sketch) computes per-vertex ambient occlusion in pre-processing and multiplies it into diffuse and specular alike;
# without it every strand, the outermost and the one a centimetre under it, is lit the same and the hair reads as
# one flat sheet. The amounts below are ours: the paper gives none.
SAO_MID = 0.7                      # STYLE: the light at the median strand point
SAO_MIN = 0.22                     # STYLE: the deepest strand point (darker than the cards': the gaps between locks
                                   # are where the hair's depth shows)


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
    # the body the groom was draped on (the character's sections from the head export: her collar, cape and back), so
    # the page can show the hair lying on her shoulders; --body none leaves it out
    body = a[a.index('--body') + 1] if '--body' in a else None
    rep = {'lods': []}
    cj = json.load(open(os.path.join(cdir, 'cards.json')))
    names = ['LOD 0', 'LOD 1', 'LOD 2']
    lod_files = sorted([f for f in os.listdir(cdir) if f.startswith('cards-lod') and f.endswith('.obj')],
                       key=lambda f: -int(f[9:-4]))
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
    # each strand's lock and clump (its nearest guide): the page tones whole locks and clumps a little lighter or
    # darker, the "ID map" of card hair (80.lv / CGMA: big clumps first, then subclumps)
    from scipy.spatial import cKDTree
    near = cKDTree(Gm['roots']).query(np.array([p[0] for p in keep]))[1]
    lk = Gm['lock'].astype(int) * 1000 + Gm['clump'].astype(int) if 'lock' in Gm and 'clump' in Gm else near * 0
    hsh = lambda v: ((v.astype(np.int64) * 2654435761 + 12345) >> 7) % 251
    ids = np.stack([hsh(lk[near]), hsh(near)], 1).astype('u1')
    counts = np.array([len(p) for p in keep])
    P = np.concatenate(keep)
    q = 2.5 / 32767
    rep['strands'] = dict(n=int(len(keep)), counts=b64(counts, '<u2'), pos=b64(np.round(P / q), '<i2'), q=q,
                          hairs_per_strand=round(float(share.mean() * Gm['n_hairs'].sum()), 2))
    rep['guides'] = int(len(Gm['roots']))
    tree = cKDTree(P[::2])
    V, _, Fc = read_obj(os.path.join(cdir, 'scalp-cap.obj'))
    centre = V.mean(0) - np.array([0, 0, 0.15])        # inside the skull, below the cap's middle
    rep['cap'] = dict(pos=b64(V, '<f4'), idx=b64(Fc, '<u4'))

    def crowd(V):
        out = V - centre
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        return sum(np.array([len(x) for x in tree.query_ball_point(V + out * d, AO_R)]) for d in AO_STEPS)
    # the strands' AO: probes outward from the head; below the skull's middle, level outward from the neck's axis
    # (a probe from the skull's centre would run down along the hanging hair and count the strand's own lock)
    outS = P - np.where(P[:, 2:3] < centre[2], np.array([centre[0], centre[1], 0]) + np.array([0, 0, 1]) * P[:, 2:3], centre)
    outS /= np.maximum(np.linalg.norm(outS, axis=1, keepdims=True), 1e-6)
    cs = sum(np.array([len(x) for x in tree.query_ball_point(P + outS * d, AO_R)]) for d in AO_STEPS)
    sao = np.clip(np.exp(-cs / (np.median(cs) / -np.log(SAO_MID))), SAO_MIN, 1.0)
    rep['strands']['ao'] = b64(np.round(sao * 255), 'u1')
    rep['strands']['ids'] = b64(ids, 'u1')
    print('strand ao mean %.2f, 10th pct %.2f' % (sao.mean(), np.percentile(sao, 10)))
    # the scale: the hair over the median LOD 0 vertex lets through AO_MID of the light
    k = np.median(crowd(read_obj(os.path.join(cdir, lod_files[0]))[0])) / -np.log(AO_MID)
    for name, f in zip(names, lod_files):
        V, VT, Fc = read_obj(os.path.join(cdir, f))
        ao = np.clip(np.exp(-crowd(V) / k), AO_MIN, 1.0)
        rep['lods'].append(dict(name=name, cards=int(f[9:-4]), triangles=int(len(Fc)), pos=b64(V, '<f4'),
                                uv=b64(VT, '<f4'), idx=b64(Fc, '<u4'), ao=b64(ao, '<f4')))
        print(name, 'ao mean %.2f, 10th pct %.2f' % (ao.mean(), np.percentile(ao, 10)))
    if body and body != 'none':
        import gzip
        rep['body'] = dict(name=body, sections=json.load(gzip.open(head))['bodies'][body], cy=0.04, p=3)
    rep['source'] = dict(groom=os.path.abspath(gdir), cards=os.path.abspath(cdir), card_report=cj['lods'])
    json.dump(rep, open(out, 'w'))
    print(out, os.path.getsize(out) // 1024, 'KB', len(keep), 'strands')


if __name__ == '__main__':
    main()
