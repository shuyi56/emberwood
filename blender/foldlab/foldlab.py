"""Fold lab: practise one lapel fold on its own, in seconds, outside Blender.

The figure build takes 20 to 30 minutes, and a lapel fold inside it is one step among hundreds. Here one lapel is
taken out of a finished build with the chest under it (capture), and folded again alone with folds.py and fabric.py:
numpy only, no Blender, no render. Each try writes a sheet (front view by height off the chest, sections across the
roll, the angle along it, strain on the pattern) and one line of numbers to a log, so tries compare at a glance.

    python3 blender/foldlab/foldlab.py capture FOLD.pkl LAB_DIR          # once per build: lapel0.npz, lapel1.npz
    python3 blender/foldlab/foldlab.py try LAB_DIR/lapel1.npz OUT_DIR '{"fold_radius": 0.2}' [--name r20] [--coarse 2]
    python3 blender/foldlab/foldlab.py sweep LAB_DIR/lapel1.npz OUT_DIR '[{"name": "a"}, {"name": "b", "tape_cm": 1.5}]'

A try's JSON overrides the lapel settings (LAB, the same keys as collar.LAPEL) and has the lab's own:
  coarse      keep every k-th row of the extension (faster, rougher): 1 is the build's own mesh
  tape_cm     the roll line taped this much short (UKY CT-LMH.176: 1/2 to 3/4 in, 1.3 to 1.9 cm, more for heavy
              cloth): the strip's roll points freed and tied along the roll to (1 - tape/len) of their length
  curl_deg    pad stitching: the lapel past the turn is stitched while held rolled over the hand (UKY), so it rests
              curled toward the chest: a second soft crease of this angle spread over the flap
  gorge_sewn  hold the gorge where the drafted fold lays it (the build's hold)
The capture keeps what the build had: the flat pattern, its triangles, the roll, the strip of front cloth held, and
the chest under the lapel (points and normals)."""
import json
import math
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
import fabric as xf      # noqa: E402
import folds as fo       # noqa: E402

UNIT_CM = 2.62
# collar.LAPEL and collar.FOLD_FABRIC as the build has them (copied: collar.py needs Blender)
LAB = dict(fold_deg=176, fold_gap=0.12, fold_radius=0.16, fold_row=0.05, spacing=0.07, fold_stretch=1e-8,
           fold_bend=1e-6, tilt=25, bake=dict(h=1.0 / 1920, most=3000, correct=False),
           gorge_sewn=True, coarse=1, tape_cm=0.0, curl_deg=0.0, w=1.964)
FOLD_FABRIC = dict(walk='gauss-seidel', iterations=4, stretch=0.0, bend=2e-4, damping=0.03, friction=0.4)


# ------------------------------------------------------------------ capture

def capture(pkl, out):
    """Each lapel of a finished build (COLLAR_FOLD_OUT's pickle) saved for the lab."""
    d = pickle.load(open(pkl, 'rb'))
    os.makedirs(out, exist_ok=True)
    for k, (L, (B, N, poly)) in enumerate(zip(d['folded'], d['outlines'])):
        bP, bN = L['body']
        lap = np.asarray(L['lap'], bool)
        np.savez(os.path.join(out, 'lapel%d.npz' % k), flat=L['flat'], T=L['T'], lap=lap, roll=L['roll'],
                 strip=L['pos'][~lap], built=L['pos'], B=B, N=N, poly=poly, bodyP=bP, bodyN=bN, w=d['M']['w'],
                 report=L['report'])
        print('lapel %d: %d points (%d on the lapel), %d triangles, chest %d points' % (k, len(lap), lap.sum(), len(L['T']), len(bP)))


# ------------------------------------------------------------------ one try

def _frame(B, N, P):
    u = (N - B) / np.linalg.norm(N - B)
    nperp = np.array([-u[1], u[0]])
    if nperp @ (np.array([0.0, N[1]]) - B) < 0:
        nperp = -nperp
    return u, nperp, (P - B) @ u, (P - B) @ nperp


def _coarsen(flat, T, lap, keep_every, row_h):
    """Drop points of the lapel past the turn's band to every k-th (by a grid of k * spacing) and remesh."""
    if keep_every <= 1:
        return flat, T, np.arange(len(flat))
    from scipy.spatial import Delaunay
    g = np.floor(flat / (keep_every * row_h)).astype(int)
    keep = ~lap.copy()
    seen = set()
    for i in np.nonzero(lap)[0]:
        key = tuple(g[i])
        if key not in seen:
            seen.add(key)
            keep[i] = True
    # (the outline and the turn keep every point: their shape is what is being practised)
    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    e, c = np.unique(e, axis=0, return_counts=True)
    keep[np.unique(e[c == 1])] = True
    idx = np.nonzero(keep)[0]
    P = flat[idx]
    tri = Delaunay(P).simplices
    # (only triangles inside the old mesh: their centres inside some old triangle)
    cen = P[tri].mean(1)
    inside = fo._inside_any(cen, flat, T)
    tri = tri[inside]
    a, b, cc = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    cr = (b[:, 0] - a[:, 0]) * (cc[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (cc[:, 0] - a[:, 0])
    tri = tri[np.abs(cr) > 1e-7]
    cr = cr[np.abs(cr) > 1e-7]
    tri[cr < 0] = tri[cr < 0][:, [0, 2, 1]]
    return P, tri, idx


def run(cap, S, log=print):
    t_start = time.time()
    C = dict(np.load(cap, allow_pickle=True))
    flat, T, lap = C['flat'], C['T'], C['lap'].astype(bool)
    B, N, poly = C['B'], C['N'], C['poly']
    bodyP, bodyN = C['bodyP'], C['bodyN']
    ns = int((~lap).sum())
    # (only the chest the lapel can reach: the build's chest box is far wider, and the floor's lookups were most of a try)
    reach = C['built']
    lo_, hi_ = reach.min(0) - 0.6, reach.max(0) + 0.6
    near = ((bodyP > lo_) & (bodyP < hi_)).all(1)
    bodyP, bodyN = bodyP[near], bodyN[near]
    assert not lap[:ns].any(), 'the strip comes first'
    h = S['spacing'] * S['w']
    flat, T, kept = _coarsen(flat, T, lap, int(S['coarse']), h)
    lap = lap[kept]
    strip = C['built'][kept[:ns]]
    u, nperp, t, sd = _frame(B, N, flat)
    turn = math.radians(S['fold_deg']) * S['fold_radius']
    # the start: the strip where the build left it, the lapel standing off it from the roll by tilt, unfolded
    roll_pts = np.nonzero(~lap & (np.abs(sd) < 0.015))[0]
    roll_pts = roll_pts[np.argsort(t[roll_pts])]
    W = strip
    te, se = t[lap], sd[lap]
    base = np.column_stack([np.interp(te, t[roll_pts], W[roll_pts, c]) for c in range(3)])
    ax = W[roll_pts[-1]] - W[roll_pts[0]]
    ax /= np.linalg.norm(ax)
    dv = np.array([nperp[0], 0.0, nperp[1]])
    dv -= (dv @ ax) * ax
    dv /= np.linalg.norm(dv)
    nrm = np.cross(ax, dv)
    if nrm[1] > 0:
        nrm = -nrm
    th = math.radians(S['tilt'])
    start = np.vstack([W, base + se[:, None] * (math.cos(th) * dv + math.sin(th) * nrm)[None]])
    roll = np.asarray(C['roll'])
    if np.median(fo.Crease(roll, 1.0).frame(flat[lap])[2]) < 0:
        roll = roll[::-1]
    ang = math.radians(S['fold_deg'])
    fab = xf.Fabric(start, dict(FOLD_FABRIC, **S.get('fabric', {})))
    held = np.arange(ns)
    tape_len = None
    if S['tape_cm'] > 0 or S.get('tape_pct'):
        # (the tape: the roll's own points let go and tied end to end along the roll, shorter than the cloth; the
        # strip's far rows stay held, as the rest of the front is)
        far = np.nonzero(~lap & (sd < -S.get('hold_rows', 1.5) * h))[0]
        held = far
    fab.pin(held)
    crease = fo.Crease(roll, ang, S['fold_radius'])
    creases = [crease]
    rest = np.column_stack([flat, np.zeros(len(flat))])
    on = np.zeros(len(flat), bool)
    curl_n = 0
    if S.get('stitch_R'):
        # Pad stitching (UKY CT-LMH.176): the lapel is stitched while held rolled over the hand, in the direction it
        # rolls when worn, so its own rest shape is curled. The flap past the turn is wrapped onto a cylinder of
        # radius stitch_R about an axis along the roll: lengths kept (a cylinder is developable), only the cloth's
        # bending rest changes. stitch_sign: which way it curls on the pattern.
        s0 = turn + S['fold_row']
        sa = sd - s0
        on = lap & (sa > 0)
        R_, sg_ = float(S['stitch_R']), float(S.get('stitch_sign', 1))
        ph = sa[on] / R_
        d_in = s0 + R_ * np.sin(ph) - sd[on]
        rest[on, :2] += d_in[:, None] * nperp[None]
        rest[on, 2] = sg_ * R_ * (1 - np.cos(ph))
    fab.mesh([tuple(int(i) for i in t_) for t_ in T], stretch=S['fold_stretch'], rest_pos=rest,
             bend=S['fold_bend'], flat=S['fold_bend'], bend_at=fo.soft_at(flat, T, creases))
    if S['tape_cm'] > 0 or S.get('tape_pct'):
        seg = np.linalg.norm(np.diff(start[roll_pts], axis=0), axis=1)
        L0 = seg.sum()
        # (UKY's 1.3 to 1.9 cm is on a full jacket's roll of about 25 cm: 5 to 7.5 %. The habit's roll is far
        # shorter (a high break), so the tape is taken as that share of the roll's length, tape_pct, not in cm)
        k = 1.0 - (S['tape_pct'] / 100.0 if S.get('tape_pct') else S['tape_cm'] / UNIT_CM / L0)
        fab.ties(roll_pts[:-1], roll_pts[1:], compliance=0.0, rest=seg * k)
        tape_len = (L0, L0 * k)
    past = np.concatenate([np.zeros(ns, bool), se > turn + S['fold_row']])
    floor = xf.MeshBody(bodyP, bodyN, gap=S['fold_gap'], reach=1.0)

    def body(x):
        p_, t_ = x.copy(), np.zeros(len(x), bool)
        p_[past], t_[past] = floor(x[past])
        return p_, t_
    F = fo.Folds(fab, flat, T)
    F.add(roll, ang, radius=S['fold_radius'])
    if S.get('stitch_R'):
        # (the cloth's own bend ties are unsigned lengths: they keep a curve but not which way it curls, so the
        # lapel came out the same curled either way. The pad stitching's curl is given as signed hinges, as the fold
        # is, on the flap past the turn, the same way round as the fold: toward the chest once the lapel lies on it)
        Hc = F.H_all[on[F.H_all].all(1)]
        rc = np.abs(fo.dihedral(rest, Hc)) * np.sign(F._rest[0].mean()) * float(S.get('stitch_sign', 1))
        keep_ = np.abs(rc) > 1e-4
        F._H.append(Hc[keep_]); F._rest.append(rc[keep_])
        F._comp.append(np.full(keep_.sum(), float(S.get('stitch_comp', 1e-5))))
        F._owner.append(np.zeros(keep_.sum(), int))
        curl_n = int(keep_.sum())
    if S['curl_deg'] > 0:
        # (pad stitching: the flap rests curled toward the chest, a soft crease along the roll half a lapel out)
        off = 0.5 * (se.max() + turn)
        cl = np.array([roll[0] + off * nperp, roll[1] + off * nperp])
        F.add(cl, math.radians(S['curl_deg']), radius=S.get('curl_radius', 0.5 * se.max()), compliance=S.get('curl_comp', 1e-4))
    sewn = np.zeros(len(flat), bool)
    if S['gorge_sewn']:
        i0 = int(np.argmin(np.linalg.norm(poly - N, axis=1)))
        rd = (B - N) / np.linalg.norm(B - N)
        G = min((poly[i0 - 1], poly[(i0 + 1) % len(poly)]), key=lambda q_: abs((q_ - N) @ rd) / max(np.linalg.norm(q_ - N), 1e-9))
        e2 = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
        e2, c2 = np.unique(e2, axis=0, return_counts=True)
        rimv = np.zeros(len(flat), bool)
        rimv[np.unique(e2[c2 == 1])] = True
        gv = G - N
        sd_, up_ = F._sides()
        from scipy.spatial import cKDTree
        for _, mv_, tg_, _, _ in list(F._targets(fab.pos, sd_, up_, 1.0)):
            mv_ = np.asarray(mv_)
            mv_ = np.nonzero(mv_)[0] if mv_.dtype == bool else mv_
            dq_ = tg_[:, [0, 2]] - N
            tq_ = np.clip(dq_ @ gv / (gv @ gv), 0, 1)
            hit_ = (np.linalg.norm(dq_ - tq_[:, None] * gv[None], axis=1) < 0.08) & rimv[mv_] & past[mv_]
            sewn[mv_[hit_]] = True
            tg_ = tg_[hit_].copy()
            if len(tg_):
                jb_ = cKDTree(bodyP).query(tg_)[1]
                qn_ = bodyN[jb_]
                ht_ = ((tg_ - bodyP[jb_]) * qn_).sum(1)
                top_ = 2 * S['fold_radius']
                want_ = top_ - (top_ - S['fold_gap']) * np.clip(tq_[hit_] * np.linalg.norm(gv) / 0.6, 0, 1)
                tg_ -= np.maximum(ht_ - want_, 0.0)[:, None] * qn_
            fab.pos[mv_[hit_]] = tg_
        fab.prev[:] = fab.pos
        fab.pin(np.nonzero(sewn)[0])
    rep = F.bake(body=body, log=lambda *a: None, sheet=False, **S['bake'])
    X = fab.pos.copy()
    # numbers
    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    e = np.unique(e, axis=0)
    l0 = np.linalg.norm(flat[e[:, 0]] - flat[e[:, 1]], axis=1)
    l1 = np.linalg.norm(X[e[:, 0]] - X[e[:, 1]], axis=1)
    onlap = lap[e[:, 0]] | lap[e[:, 1]]
    st = (l1 / np.maximum(l0, 1e-9) - 1)[onlap] * 100
    from scipy.spatial import cKDTree
    jb = cKDTree(bodyP).query(X)[1]
    ht = ((X - bodyP[jb]) * bodyN[jb]).sum(1)
    m = F.measure(X)
    segs = F.measure_segments(X) if hasattr(F, 'measure_segments') else None
    res = dict(secs=round(time.time() - t_start, 1), steps=rep.steps, ok=bool(rep.ok),
               fold=[round(a, 1) for a in rep.achieved_deg], target=[round(a, 1) for a in rep.target_deg],
               strain=[round(float(np.percentile(st, 5)), 2), round(float(np.percentile(st, 95)), 2),
                       round(float(st.min()), 1), round(float(st.max()), 1)],
               height_past=[round(float(np.percentile(ht[past], 50)), 3), round(float(ht[past].max()), 3)],
               sewn=int(sewn.sum()), points=len(flat), tape=tape_len, curl_hinges=curl_n if S.get('stitch_R') else 0)
    return dict(X=X, flat=flat, T=T, lap=lap, past=past, ht=ht, roll=roll, nperp=nperp, u=u, B=B, N=N, poly=poly,
                res=res, segs=segs, edges=(e, l0, l1, onlap), bodyP=bodyP, bodyN=bodyN, measure=m, report=str(rep))


# ------------------------------------------------------------------ the sheet

def sheet(R, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    X, flat, T, lap, ht = R['X'], R['flat'], R['T'], R['lap'], R['ht']
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))
    # 1: front view, lapel by height off the chest
    a = axs[0, 0]
    a.tripcolor(X[:, 0], X[:, 2], T, facecolors=np.array([ht[t].mean() if lap[t].all() else np.nan for t in T]), cmap='viridis', vmin=0, vmax=0.6)
    tri_strip = np.array([not lap[t].all() for t in T])
    a.triplot(X[:, 0], X[:, 2], T[tri_strip], lw=0.2, color='0.6')
    a.set_aspect('equal'); a.set_title('front view: lapel height off the chest (0 to 0.6 units)')
    # 2: side view (y, z)
    a = axs[0, 1]
    a.scatter(-X[lap, 1], X[lap, 2], s=1, c=ht[lap], cmap='viridis', vmin=0, vmax=0.6)
    a.scatter(-R['bodyP'][:, 1], R['bodyP'][:, 2], s=0.2, c='0.75')
    a.set_aspect('equal'); a.set_title('side view (front to the right)')
    # 3: sections across the roll at 25, 50, 75 % of its length
    a = axs[0, 2]
    u, nperp, B, N = R['u'], R['nperp'], R['B'], R['N']
    L = np.linalg.norm(N - B)
    t = (flat - B) @ u
    for f, col in zip((0.25, 0.5, 0.75), ('C0', 'C1', 'C2')):
        sel = np.abs(t - f * L) < 0.04
        if sel.sum() < 3:
            continue
        P = X[sel]
        o = X[np.argmin(np.abs(t - f * L) + 10 * np.abs((flat - B) @ nperp))]
        d3 = np.array([nperp[0], 0, nperp[1]])
        xs, ys = (P - o) @ d3, -(P - o)[:, 1]
        order = np.argsort((flat[sel] - B) @ nperp)
        a.plot(xs[order], ys[order], '.', ms=2, color=col, label='%d%% up the roll' % (f * 100))
    a.set_aspect('equal'); a.legend(fontsize=7); a.set_title('sections across the roll (out of the chest is up)')
    # 4: strain on the pattern
    a = axs[1, 0]
    e, l0, l1, onlap = R['edges']
    s = (l1 / np.maximum(l0, 1e-9) - 1) * 100
    segs = np.stack([flat[e[:, 0]], flat[e[:, 1]]], 1)
    lc = LineCollection(segs[onlap], array=np.clip(s[onlap], -10, 10), cmap='coolwarm', linewidths=0.6)
    lc.set_clim(-10, 10)
    a.add_collection(lc); a.autoscale(); a.set_aspect('equal'); a.set_title('strain on the pattern, -10 to +10 %')
    plt.colorbar(lc, ax=a, fraction=0.04)
    # 5: angle along the roll
    a = axs[1, 1]
    sg = R['segs']
    if sg and sg[0]:
        a.plot([0.5 * (r_[0] + r_[1]) for r_ in sg[0]], [r_[2] for r_ in sg[0]], 'o-')
    a.set_xlabel('along the roll (units)')
    a.axhline(R['res']['target'][0], color='k', lw=0.5)
    a.set_title('fold angle along the roll (each segment, degrees)')
    # 6: numbers
    a = axs[1, 2]
    a.axis('off')
    a.text(0, 1, title + '\n\n' + '\n'.join('%s: %s' % kv for kv in R['res'].items()), va='top', family='monospace', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _one(job):
    cap, out, name, over = job
    S = json.loads(json.dumps(LAB))
    for k, v in over.items():
        if k == 'bake':
            S['bake'].update(v)
        else:
            S[k] = v
    R = run(cap, S)
    png = os.path.join(out, '%s.png' % name)
    sheet(R, png, '%s  %s  %s' % (name, os.path.basename(cap), json.dumps(over)))
    np.savez(os.path.join(out, '%s.npz' % name), X=R['X'], flat=R['flat'], T=R['T'], lap=R['lap'])
    return dict(name=name, cap=os.path.basename(cap), over=over, **R['res'])


def sweep(cap, out, variants, procs=4):
    """Several tries at once, one per core (a try is one Python thread of numpy), and a contact sheet of their
    front and section views side by side."""
    from multiprocessing import Pool
    os.makedirs(out, exist_ok=True)
    jobs = [(cap, out, v.pop('name'), v) for v in variants]
    with Pool(min(procs, len(jobs))) as pool:
        lines = pool.map(_one, jobs)
    with open(os.path.join(out, 'log.jsonl'), 'a') as fh:
        for ln in lines:
            fh.write(json.dumps(ln) + '\n')
            print(json.dumps(ln))
    from PIL import Image
    ims = [Image.open(os.path.join(out, ln['name'] + '.png')) for ln in lines]
    W, H = ims[0].size
    cs = Image.new('RGB', (W, H * len(ims)), 'white')
    for i, im in enumerate(ims):
        cs.paste(im, (0, i * H))
    cs.save(os.path.join(out, 'sweep-%s.png' % '-'.join(ln['name'] for ln in lines)[:80]))
    return lines


def main():
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    if sys.argv[1] == 'capture':
        capture(sys.argv[2], sys.argv[3])
        return
    if sys.argv[1] == 'sweep':                   # foldlab.py sweep CAP OUT '[{"name": "a", ...}, ...]'
        sweep(sys.argv[2], sys.argv[3], json.loads(sys.argv[4]))
        return
    cap, out = sys.argv[2], sys.argv[3]
    over = json.loads(sys.argv[4]) if len(sys.argv) > 4 and not sys.argv[4].startswith('--') else {}
    name = sys.argv[sys.argv.index('--name') + 1] if '--name' in sys.argv else time.strftime('%H%M%S')
    S = json.loads(json.dumps(LAB))
    for k, v in over.items():
        if k == 'bake':
            S['bake'].update(v)
        else:
            S[k] = v
    if '--coarse' in sys.argv:
        S['coarse'] = int(sys.argv[sys.argv.index('--coarse') + 1])
    os.makedirs(out, exist_ok=True)
    R = run(cap, S)
    png = os.path.join(out, '%s.png' % name)
    sheet(R, png, '%s  %s  %s' % (name, os.path.basename(cap), json.dumps(over)))
    np.savez(os.path.join(out, '%s.npz' % name), X=R['X'], flat=R['flat'], T=R['T'], lap=R['lap'])
    line = dict(name=name, cap=os.path.basename(cap), over=over, **R['res'])
    with open(os.path.join(out, 'log.jsonl'), 'a') as fh:
        fh.write(json.dumps(line) + '\n')
    print(json.dumps(line))
    print(R['report'].replace('\n', ' | ')[:600])


if __name__ == '__main__':
    main()
