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


TIE_FROM_MM = 20.0          # the tie is looked for past the hand-over from the head's hair (r90 peaks at 20 mm)
TIE_SEARCH_MM = 80.0        # and this far along the tail from the bundle's start


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
    tiltv = comb = None
    if '--volume' in a:                                               # the side hair in layers off the bed (hair_volume.py)
        import hair_volume as HV
        FF0 = PF.free_fall(B)
        paths, tiltv, comb, vol = HV.layer(paths, basin == 0, BF.Shell(B['surf']), np.asarray(FF0['cl'])[0], float(B['r_tie']),
                                     np.asarray(FF0['tang'])[0])
        print('volume', vol, file=sys.stderr)
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
    share = float(hairs.sum()) / len(ps)
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
    lock_id = basin[keep] * 7919 + keep // 40
    if comb is not None:                               # (a comb lock's hair shares its tone)
        lock_id = np.where(comb[keep] >= 0, 104729 + comb[keep], lock_id)
    ids = np.stack([hsh(lock_id), hsh(keep)], 1).astype('u1')
    # light through the hair from its flow and density (hair_light.py): per creator light, and ambient
    import hair_light as HL
    lt = HL.light(ps, hairs[keep] * len(paths) / len(keep))      # (each drawn strand: its fibre's hairs, scaled up)
    occ = np.stack([lt['key'], lt['rim'], lt['fill'], lt['ambient']], 1)
    print('light: key T median %.2f, 10th pct %.2f; ambient median %.2f, 10th pct %.2f' % (
        np.median(lt['key']), np.percentile(lt['key'], 10), np.median(lt['ambient']), np.percentile(lt['ambient'], 10)))
    # which way is out, per point: on the head, out from the skull (as the page had it); along the tail, out from the
    # tail's own centre line (the free-fall bundle's, FF['cl']). Lit by the head's "out", every hair of the tail faced
    # the same way and the tail shaded as one flat sheet; lit round its own axis it reads as a round bundle.
    FF = PF.free_fall(B)
    cl, sc = np.asarray(FF['cl']), np.asarray(FF['s'], float)
    jn = cKDTree(cl).query(P)[1]
    tg = np.gradient(cl, axis=0); tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-12)
    v = P - cl[jn]
    v -= (v * tg[jn]).sum(1, keepdims=True) * tg[jn]
    dcl = np.linalg.norm(v, axis=1)
    out_tail = v / np.maximum(dcl[:, None], 1e-9)
    sj = np.abs(sc[jn] - sc[0]) * (1.0 if sc.max() > 5 else 1 / BF.MM)       # mm along the tail from the tie
    Rj = np.asarray(FF['R'], float)[jn]
    wt = BF.sm(0.0, 25.0, sj) * (dcl / BF.MM < Rj + 10.0)
    oh = P - np.c_[np.full(len(P), centre[0]), np.minimum(P[:, 1], centre[1]), np.full(len(P), centre[2])]
    oh /= np.maximum(np.linalg.norm(oh, axis=1, keepdims=True), 1e-9)
    if tiltv is not None:                              # the comb furrows' own slopes (hair_volume.py), so they are lit
        oh += np.concatenate([tiltv[i][~np.isnan(paths[i][:, 0])] for i in keep])
        oh /= np.maximum(np.linalg.norm(oh, axis=1, keepdims=True), 1e-9)
    outv = oh * (1 - wt)[:, None] + out_tail * wt[:, None]
    outv /= np.maximum(np.linalg.norm(outv, axis=1, keepdims=True), 1e-9)
    print('tail points %.0f%%' % (100 * (wt > 0.5).mean()), 's range', sc.min(), sc.max(), 'R', Rj.min(), Rj.max())
    # the tie: where the field gathers the hair (the bundle's first section), for the page to wind its thong round
    # (not the bundle's first section, cl[0], which lies inside the hair on the head and where the centre line still
    # rises: the section within TIE_SEARCH_MM along the tail where the drawn hair is gathered tightest)
    best = None
    sm_ = np.abs(sc - sc[0]) * (1.0 if sc.max() > 5 else 1 / BF.MM)
    for j in np.where((sm_ >= TIE_FROM_MM) & (sm_ < TIE_SEARCH_MM))[0]:
        m = (jn == j) & (dcl / BF.MM < Rj + 10.0)
        if m.sum() < 30:
            continue
        r90 = float(np.percentile(dcl[m], 90))
        if best is None or r90 < best[1]:
            best = (j, r90)
    jt, rt = best if best else (0, float(B['r_tie']))
    print('tie at %.0f mm along the tail, radius %.1f mm' % (sm_[jt], rt / BF.MM))
    tie = dict(c=[float(x) for x in cl[jt]], t=[float(x) for x in tg[jt]], r=rt, s_mm=float(abs(sc[jt] - sc[0])))
    rep = dict(lods=[], guides=0, tie=tie, cap=dict(pos=CA.b64(V, '<f4'), idx=CA.b64(Fc, '<u4')),
               strands=dict(n=int(len(ps)), counts=CA.b64(counts, '<u2'), pos=CA.b64(np.round(P / q), '<i2'), q=q,
                            hairs_per_strand=round(share, 2), ao=CA.b64(np.round(sao * 255), 'u1'), ids=CA.b64(ids, 'u1'),
                            light=CA.b64(np.round(np.clip(occ, 0, 1) * 255), 'u1'), light_order='key,rim,fill,ambient',
                            out=CA.b64(np.round(outv * 127), 'i1')),
               source=dict(bangs=os.path.abspath(bdir), fibres=int(len(paths))))
    json.dump(rep, open(out, 'w'))
    print(out, os.path.getsize(out) // 1024, 'KB', len(ps), 'strands of', len(paths), 'fibres, %.1f hairs each' % share)


if __name__ == '__main__':
    main()
