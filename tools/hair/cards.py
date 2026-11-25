"""Hair rebuild, foundation 8: game hair baked from the groom. Cards (ribbons) are fitted to clusters of the settled
guides, so the game's hair has the shape the rod model gave the strands instead of a shape authored separately.

    python3 tools/hair/cards.py GROOM_DIR OUT_DIR [--lods 360,140,60] [--head HEAD.json.gz]

Writes OUT_DIR/cards-lod<k>.obj (positions, UVs: u across the card, v from root to tip, normals) and cards.json
(cards, triangles, and how far each LOD's surface lies from the strands it stands for).

How (the usual game hair pipeline: 80 Level, "Tips and tricks on hair for games"; the creator's own hair cards):
  1. Cluster the guides by root position and settled path (k-means on root plus three points down the path), so a
     card follows hair that really falls together.
  2. The card's centre line is the members' mean path, to the cluster's median length.
  3. Its width at each point is the members' spread across the hair (2.5 standard deviations, at least two guides'
     spacing), so the card covers the strands it replaces; it is turned to face out from the head's centre.
  4. The strand texture (u across, v along) and the alpha cut-out of the tips belong to the material (as in the
     creator); this tool writes the geometry and the UVs.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def kmeans(F, k, iters=30, seed=2):
    rng = np.random.default_rng(seed)
    C = F[rng.choice(len(F), k, replace=False)]
    for _ in range(iters):
        d = ((F[:, None] - C[None]) ** 2).sum(-1)
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = F[m].mean(0)
    return a


ROOT_ROWS = 6                      # DRAFT: guide segments over which the root's width eases into the lock's
CARD_LIFT_MM = 2.0                 # DRAFT: cards lie at least this far over the skin (the cap lies at 1 mm)


CARD_W_MAX = 0.2                   # HH (4.4 cm): the widest a card is drawn below its root rows
CARD_BEND_DEG = 35.0               # a card's mean line turns at most this much per guide segment
FRAME_CLUMP0 = 900                 # groom.py frame_locks: clump ids from here are the face-framing locks


BANG_CARD0 = 10000                 # the bangs' card groups (fixed ids from here)


def bake(X, k, spacing, step=3, centre=np.array([0.0, 0.04, 0.0]), fixed=None):
    """fixed: a group id per guide (-1: none); each group is one card of its own (the face-framing locks, gathered
    in the groom: split among k-means cards or mixed with the hair beside them, they drew as loose wisps), the rest
    k-means clustered into the remaining cards."""
    G, S1 = X.shape[:2]
    last = np.array([np.max(np.where(~np.isnan(X[g, :, 0]))[0]) for g in range(G)])
    Xf = X.copy()
    for g in range(G):
        Xf[g, last[g] + 1:] = X[g, last[g]]
    feat = np.c_[Xf[:, 0], Xf[:, min(10, S1 - 1)], Xf[:, min(25, S1 - 1)], Xf[:, min(45, S1 - 1)]]
    if fixed is not None and (fixed >= 0).any():
        ids = np.unique(fixed[fixed >= 0])
        rest = np.where(fixed < 0)[0]
        lab = np.empty(G, int)
        lab[rest] = kmeans(feat[rest], k - len(ids))
        for i, f in enumerate(ids):
            lab[fixed == f] = k - len(ids) + i
    else:
        lab = kmeans(feat, k)
    cards = []
    for j in range(k):
        m = np.where(lab == j)[0]
        if not len(m):
            continue
        L = int(np.median(last[m]))
        if L < 2:
            continue
        # a member shorter than the card carries on with the others' mean step, not parked at its own end: parked,
        # each member's end pulled the mean back toward it, and a card of mixed lengths (the face frame's graded cut)
        # zigzagged where they ended
        M = Xf[m, :L + 1].copy()
        for i in range(1, L + 1):
            alive = last[m] >= i
            if alive.any() and not alive.all():
                M[~alive, i] = M[~alive, i - 1] + (X[m[alive], i] - X[m[alive], i - 1]).mean(0)
        C = M.mean(0)
        # a sharp corner in the mean line (members parting ways over the shoulder) eased: hair does not corner
        for _ in range(20):
            d = np.diff(C, axis=0)
            u = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
            sharp = np.where((u[:-1] * u[1:]).sum(1) < np.cos(np.radians(CARD_BEND_DEG)))[0] + 1
            if not len(sharp):
                break
            C2 = C.copy()
            for i in sharp:
                for j in (i - 1, i, i + 1):
                    if 0 < j < len(C) - 1:
                        C2[j] = 0.5 * C[j] + 0.25 * (C[j - 1] + C[j + 1])
            C = C2
        # the mean of hairs lying over a curved scalp lies inside the curve (a lock 4 cm wide over the crown dips
        # about 1.5 mm into the skull): set each point back out to the members' mean distance from the head's centre
        rc = np.linalg.norm(C - centre, axis=1)
        rm = np.linalg.norm(Xf[m, :L + 1] - centre, axis=2).mean(0)
        C = centre + (C - centre) * (np.maximum(rc, rm) / np.maximum(rc, 1e-9))[:, None]
        T = np.gradient(C, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        # the card faces out from the head's centre: over the crown it lies flat on the hair, down the sides it faces
        # out horizontally. (Facing out from the body's vertical axis stood the crown's cards on edge, like fins.)
        out = C - np.asarray(centre)
        if centre[2] == 0.0:
            out[:, 2] = 0
        r0 = np.maximum(np.linalg.norm(out, axis=1), 1e-9)
        out = out - T * (out * T).sum(1, keepdims=True)
        # far below the head, or where hair lies out along the shoulder, the card runs nearly along "out from the
        # head's centre": that direction has little or no part across the card, and its remainder swings round or
        # reverses from row to row, so the strip crossed itself (a bowtie, drawn as a zigzag of strands). There a row
        # keeps the row before's facing, carried along the card
        nrm = np.linalg.norm(out, axis=1)
        for i in range(1, len(out)):
            v = out[i - 1] - T[i] * (out[i - 1] @ T[i])
            nv = np.linalg.norm(v)
            if nv < 1e-6:
                continue
            if nrm[i] < 0.35 * r0[i] or (out[i] @ v) < 0.5 * nrm[i] * nv:
                out[i] = v / nv * max(nrm[i], 1e-6)
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        B = np.cross(T, out)
        B /= np.linalg.norm(B, axis=1, keepdims=True)
        spread = np.array([np.std(((Xf[m, i] - C[i]) * B[i]).sum(1)) if len(m) > 1 else 0 for i in range(L + 1)])
        # the spread read smooth along the card (a median of three rows, then a mean of five) and held to CARD_W_MAX:
        # row by row it jumped (a framing lock's members wave in and out of step: 0.05, 0.19, 0.06 HH), and the card's
        # edges drew as a staircase
        p = np.pad(spread, 1, mode='edge')
        spread = np.median(np.stack([p[:-2], p[1:-1], p[2:]]), 0)
        p = np.pad(spread, 2, mode='edge')
        spread = np.mean(np.stack([p[i:i + len(spread)] for i in range(5)]), 0)
        w = np.minimum(np.maximum(2.5 * spread, 2 * spacing), CARD_W_MAX)
        # at the root the card covers the members' roots across the hair, with a guide's spacing to spare on each
        # side, so neighbouring cards meet over the scalp (the part and the crown otherwise showed the cap)
        rspan = np.ptp(((Xf[m, 0] - C[0]) * B[0]).sum(1)) + 2 * spacing
        w = np.maximum(w, rspan * np.clip(1 - np.arange(len(w)) / ROOT_ROWS, 0, 1))
        # taper the last 15% to a point, so the card ends in the hair's soft tip shape
        t = np.linspace(0, 1, L + 1)
        w = w * (1 - 0.85 * np.clip((t - 0.85) / 0.15, 0, 1))
        keep = np.unique(np.r_[np.arange(0, L + 1, step), L])      # rows every `step` guide segments: straight hair
        C, B, N, w = C[keep], B[keep], out[keep], w[keep]
        # the members' roots are spread along the hair too, not only across it: start the card a guide's spacing
        # behind the rearmost member's root, or the scalp behind the card's mean root is bare under it
        back = -np.min(((Xf[m, 0] - C[0]) * T[0]).sum(1)) + spacing
        if back > spacing:
            C = np.r_[[C[0] - T[0] * back], C]
            B, N, w = np.r_[B[:1], B], np.r_[N[:1], N], np.r_[w[:1], w]
        cards.append(dict(C=C, B=B, N=N, w=w, members=m))  # straight hair needs few rows
    return cards


def write_obj(cards, path, surf=None):
    V, VT, VN, Fc = [], [], [], []
    for c in cards:
        n = len(c['C'])
        base = len(V)
        for i in range(n):
            for s in (-0.5, 0.5):
                V.append(c['C'][i] + c['B'][i] * c['w'][i] * s)
                VT.append((s + 0.5, i / (n - 1)))
                VN.append(c['N'][i])
        for i in range(n - 1):
            a, b, cc, d = base + 2 * i, base + 2 * i + 1, base + 2 * i + 2, base + 2 * i + 3
            Fc += [(a, b, d), (a, d, cc)]
    if surf is not None:
        # nothing under the scalp cap: a ribbon is flat across, so over the curved scalp its edges (and the roots of
        # the centre line) sit in the skull, under the cap, and the cap shows through where the hair should be
        import scalp_field as SF
        V = np.array(V)
        q = V - SF.CENTRE
        r = np.linalg.norm(q, axis=1)
        need = surf.radius(q / r[:, None]) + CARD_LIFT_MM / SF.HH_MM
        d = np.where(V[:, 2] > 0.2, np.maximum(need - r, 0.0), 0.0)
        # the lift eased along each card (the largest of a row and its neighbours, then a mean over three rows, both
        # edges alike): lifted vertex by vertex, a card passing over the ear had some rows out and some not, a
        # staircase (the Warden's sides once they fell over her ears)
        o = 0
        for c in cards:
            n = len(c['C'])
            e = d[o:o + 2 * n].reshape(n, 2).max(1)
            p = np.pad(e, 1, mode='edge')
            e = np.maximum.reduce([p[:-2], p[1:-1], p[2:]])
            p = np.pad(e, 1, mode='edge')
            e = np.maximum((p[:-2] + p[1:-1] + p[2:]) / 3, d[o:o + 2 * n].reshape(n, 2).max(1))
            d[o:o + 2 * n] = np.repeat(e, 2)
            o += 2 * n
        V = SF.CENTRE + q * ((r + d) / r)[:, None]
    with open(path, 'w') as f:
        f.write('# hair cards baked from the settled groom (tools/hair/cards.py); units: head heights, z up\n')
        for v in V:
            f.write('v %.5f %.5f %.5f\n' % tuple(v))
        for t in VT:
            f.write('vt %.4f %.4f\n' % t)
        for n in VN:
            f.write('vn %.4f %.4f %.4f\n' % tuple(n))
        for a, b, c in Fc:
            f.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (a + 1, a + 1, a + 1, b + 1, b + 1, b + 1, c + 1, c + 1, c + 1))
    return len(V), len(Fc)


def card_points(cards, step=0.004):
    """Points over the cards' surfaces (for renders and for the distance check)."""
    P = []
    for c in cards:
        C, B, w = c['C'], c['B'], c['w']
        seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
        s_ = np.r_[0, np.cumsum(seg)]
        t = np.arange(0, s_[-1], step)                      # rows filled in along the length too
        Ci = np.stack([np.interp(t, s_, C[:, q]) for q in range(3)], 1)
        Bi = np.stack([np.interp(t, s_, B[:, q]) for q in range(3)], 1)
        wi = np.interp(t, s_, w)
        for i in range(len(t)):
            k = max(2, int(wi[i] / step))
            for sv in np.linspace(-0.5, 0.5, k):
                P.append(Ci[i] + Bi[i] * wi[i] * sv)
    return np.array(P)


# The cap's edge: cut across each cell where the hair's share of the scalp map crosses one half (marching squares),
# so it runs on the hairline's own curve. Cells kept whole or dropped (3 degrees: 4.6 mm at the temple) showed as a dark
# stair along the temple once the side hair was swept back off it (2026-11-24).
CAP_STEP_DEG = 3.0
CAP_EDGE_K = 12              # scalp-map points the hair's share at a grid point is read from


def scalp_cap(head, out, step_deg=CAP_STEP_DEG, lift_mm=1.0):
    """The scalp cap: a shell 1 mm over the hair-bearing scalp, under the cards, carrying the hair's root colour so
    gaps between cards (the crown whorl, the part) show hair, not skin (the usual game hair setup: 80 Level, "Tips and
    tricks on hair for games"). Cells of a direction grid on the head where the scalp map has hair, cut on the
    hairline's curve at the edge (CAP_EDGE_K)."""
    from scipy.spatial import cKDTree
    import scalp_field as SF
    surf = SF.Surface(SF.load_head(head))
    F = SF.build(surf)
    el = np.radians(np.arange(-40, 90 + 1e-6, step_deg))
    az = np.radians(np.arange(0, 360, step_deg))
    E, A = np.meshgrid(el, az, indexing='ij')
    U = np.c_[(np.cos(E) * np.sin(A)).ravel(), (-np.cos(E) * np.cos(A)).ravel(), np.sin(E).ravel()]
    lift = lift_mm / SF.HH_MM
    P = surf.point(U) + surf.normals(U) * lift
    tree = cKDTree(F['P'])
    _, nb = tree.query(P, k=CAP_EDGE_K)
    fv = F['mask'][nb].astype(float).mean(1)
    ne, na = E.shape
    idx = np.arange(ne * na).reshape(ne, na)
    extra, ekey = [], {}

    def cut(a, b):
        """The point on the grid edge a-b where the hair's share crosses one half (a new vertex, shared by both cells)."""
        k = (min(a, b), max(a, b))
        if k not in ekey:
            t = np.clip((0.5 - fv[a]) / (fv[b] - fv[a]), 0, 1)
            u = U[a] + t * (U[b] - U[a])
            u /= np.linalg.norm(u)
            ekey[k] = len(P) + len(extra)
            extra.append(surf.point(u[None])[0] + surf.normals(u[None])[0] * lift)
        return ekey[k]

    Fc = []
    for i in range(ne - 1):
        for j in range(na):
            q = [idx[i, j], idx[i, (j + 1) % na], idx[i + 1, (j + 1) % na], idx[i + 1, j]]
            h = fv[q] >= 0.5
            if h.all():
                Fc += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
                continue
            if not h.any():
                continue
            poly = []
            for t in range(4):
                a, b = q[t], q[(t + 1) % 4]
                if h[t]:
                    poly.append(a)
                if h[t] != h[(t + 1) % 4]:
                    poly.append(cut(a, b))
            Fc += [(poly[0], poly[t], poly[t + 1]) for t in range(1, len(poly) - 1)]
    if extra:
        P = np.r_[P, np.array(extra)]
    used = np.unique(np.array(Fc).ravel())
    remap = -np.ones(len(P), int); remap[used] = np.arange(len(used))
    with open(os.path.join(out, 'scalp-cap.obj'), 'w') as f:
        f.write('# scalp cap under the hair cards (tools/hair/cards.py); units: head heights, z up\n')
        for v in P[used]:
            f.write('v %.5f %.5f %.5f\n' % tuple(v))
        for a, b, c in Fc:
            f.write('f %d %d %d\n' % (remap[a] + 1, remap[b] + 1, remap[c] + 1))
    # points over the cap for the render
    T = np.array(Fc)
    r = np.random.default_rng(0).random((len(T), 40, 2))
    r = np.where(r.sum(-1, keepdims=True) > 1, 1 - r, r)
    Q = P[T[:, 0]][:, None] + (P[T[:, 1]] - P[T[:, 0]])[:, None] * r[..., :1] + (P[T[:, 2]] - P[T[:, 0]])[:, None] * r[..., 1:]
    return Q.reshape(-1, 3), len(used), len(Fc)


def render_cards(head, out, lods, cap=None, size=600):
    """Clay turnaround of each LOD's cards over the head and body (render.py's camera and backdrop)."""
    import gzip
    import render as RD
    import scalp_field as SF
    from PIL import Image
    RD.HEAD_CLOUD = SF.load_head(head)
    surf = SF.Surface(RD.HEAD_CLOUD)
    body = json.load(gzip.open(head))['bodies']['mannequin']
    rows = []
    for k in lods:
        P = np.load(os.path.join(out, 'cards-lod%d.npz' % k))['P'].astype(float)
        Pc = np.zeros((0, 3)) if cap is None else cap
        row = []
        for az in (0, 45, 90, 135, 180):
            cv = RD.Canvas(size, az)
            bg, zb = RD.clay_backdrop(cv, surf, body)
            img = bg.copy(); zc = zb.copy()
            for Q, col in ((Pc, np.array([0.40, 0.30, 0.25])), (P, np.array([0.55, 0.42, 0.34]))):
                img, zc = splat(cv, Q, col, img, zc)
            row.append((RD.to_srgb(img) * 255).astype(np.uint8))
        rows.append(np.concatenate(row, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(out, 'cards-turn.png'))


def splat(cv, P, col, img, zc):
    """Flat-colour z-buffered splats (the cap darker than the cards, as the root colour is)."""
    i, j, z, ok = cv.project(P)
    for di in (0, 1):
        for dj in (0, 1):
            ii, jj = np.clip(i + di, 0, cv.W - 1), np.clip(j + dj, 0, cv.H - 1)
            np.minimum.at(zc, (jj[ok], ii[ok]), z[ok])
    for di in (0, 1):
        for dj in (0, 1):
            ii, jj = np.clip(i + di, 0, cv.W - 1), np.clip(j + dj, 0, cv.H - 1)
            win = ok & (z <= zc[jj, ii] + 1e-9)
            img[jj[win], ii[win]] = col
    return img, zc


def main():
    a = sys.argv[1:]
    gdir, out = a[0], a[1]
    lods = [int(x) for x in a[a.index('--lods') + 1].split(',')] if '--lods' in a else [360, 140, 60]
    steps = {lods[0]: 2}                   # the closest LOD keeps a row every 2 cm, the others every 3 cm
    os.makedirs(out, exist_ok=True)
    G = np.load(os.path.join(gdir, 'groom.npz'))
    X = G['X']
    from scipy.spatial import cKDTree
    R = G['roots']
    spacing = float(np.median(cKDTree(R).query(R, k=2)[0][:, 1]))
    S = X.reshape(-1, 3); S = S[~np.isnan(S[:, 0])]
    rep = {'guides': int(len(X)), 'lods': []}
    import scalp_field as SF
    centre = SF.CENTRE
    surf = SF.Surface(SF.load_head(a[a.index('--head') + 1])) if '--head' in a else None
    for k in lods:
        fixed = np.where(G['clump'] >= FRAME_CLUMP0, G['clump'], -1) if 'clump' in G else None
        if 'bang' in G and G['bang'].any():
            # the bangs carded among themselves (their share of the cards): mixed with the hair beside them, a
            # card averaged the fan and the side hair and stood off the forehead as a see-through veil
            fixed = np.full(len(X), -1) if fixed is None else fixed.copy()
            bi = np.where(G['bang'].astype(bool) & (fixed < 0))[0]
            kb = max(2, int(round(k * len(bi) / len(X))))
            Xb = X[bi].copy()
            last = (~np.isnan(Xb[..., 0])).sum(1) - 1
            for j in range(len(bi)):
                Xb[j, last[j] + 1:] = Xb[j, last[j]]
            S1 = Xb.shape[1]
            fb = np.c_[Xb[:, 0], Xb[:, min(5, S1 - 1)], Xb[:, min(10, S1 - 1)], Xb[:, min(20, S1 - 1)]]
            fixed[bi] = BANG_CARD0 + kmeans(fb, kb)
        cards = bake(X, k, spacing, steps.get(k, 3), centre, fixed)
        nv, nf = write_obj(cards, os.path.join(out, 'cards-lod%d.obj' % k), surf)
        P = card_points(cards)
        d, _ = cKDTree(P).query(S)            # how far each strand point is from the nearest card
        rep['lods'].append(dict(cards=len(cards), vertices=nv, triangles=nf,
                                strand_to_card_mm_p50=round(float(np.median(d)) * 219, 1),
                                strand_to_card_mm_p95=round(float(np.percentile(d, 95)) * 219, 1)))
        np.savez_compressed(os.path.join(out, 'cards-lod%d.npz' % k), P=P.astype(np.float32))
    if '--head' in a:
        cap, nv, nf = scalp_cap(a[a.index('--head') + 1], out)
        rep['scalp_cap'] = dict(vertices=nv, triangles=nf)
        render_cards(a[a.index('--head') + 1], out, lods, cap)
    json.dump(rep, open(os.path.join(out, 'cards.json'), 'w'), indent=1)
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
