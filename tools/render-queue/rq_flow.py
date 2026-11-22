"""The flow check (John, 2026-11-19): the stress and strain field across each garment after its build or solve, read
before an expensive render. A smooth field means the cut and the body agree. Jagged peaks mark faults: dart tips and
seam corners where curvature concentrates, paired seam edges of unequal length, grain or bias errors, too little ease
at the armholes, shoulders or seat. The check is report-only: it never stops a render, it flags the likely
rejections early and leaves a heatmap beside the render.

The queue puts this file in a job's worktree beside rq_views.py, which runs it right after the build (before the first
render) on outfit and pose-strip jobs, and in every pose of a pose strip (no render needed: the garments are posed by
the rig and read back). Any other views job asks for it with --flow. Results, in the job's folder:

    flow.json              per garment: the numbers below, its flags, and its worst spots with their places
    flow-heat.png          the heatmap: strain, jaggedness and bending, front, 3/4, profile and back, and the flat
                           pattern when the garment has one. The worst spots are ringed and numbered as in flow.json.
    flow-poses.json        with --poses: the same per pose; flow-heat-pose-NAME.png for each pose that is flagged

What is measured, per garment (a mesh that is not the base figure's), against its reference: the flat pattern when
the mesh carries one (a UV layer 'pattern', 'flat' or 'cut', or any UV layer in model units), otherwise the garment
as built in its rest pose (so only what the pose does to it shows):

    stretch_p99, squash_p1   the principal stretches of each triangle (1.00 = as cut), area-weighted percentiles
    shear_p99                degrees the threads' right angle has turned (on the pattern: the warp against the weft,
                             so a grain or bias error shows as shear; otherwise the largest angle change)
    strain_p50, strain_p99   the Green strain's size, |E| = |(F^T F - I) / 2|
    bend_p99                 bending energy density per edge (discrete shells: theta^2 |e|^2 / (A1 + A2)^2), in
                             1/unit^2: the curvature squared (a tube of radius r gives 1/r^2), so 1/sqrt is a fold
                             radius.
                             fold_radius_p1 is that radius, in units, at the tightest 1% of the cloth
    jag_p99                  how far a triangle's strain stands above its neighbours' mean, at the worst 1%
    roughness                rms of that excess over the field's own spread: near 0 smooth, near 1 noise
    grad_p99                 how fast strain changes between neighbouring triangles, per unit of length
    peaks                    isolated strain peaks: local maxima standing well clear of their neighbourhood, one
                             per place (peaks closer than PEAK_SPACING are one)
    peak_max                 how far the sharpest peak stands above its neighbourhood, in strain
    seam_mismatch_pct        the worst seam: paired edges' lengths on the reference, |A - B| / max, in percent.
                             Seams are found three ways: UV seams inside one mesh (both sides of the cut), a
                             'seam_mate' point attribute (each point's partner index, -1 for none) that a garment
                             script can write, and edges of separate pieces lying on each other after the sew

The limits come from the garments John approved (flow_baseline.json: run on main's hose, boots, chemise and vest),
each the worst approved value with a margin. A garment past a limit is flagged 'watch'; well past it, or with several
peaks over the band, 'likely rejection'. Nothing blocks.

Cached: each garment's result is stored under a key made of its posed points, its reference, its triangles, this
file's version and the limits (RQ_FLOW_CACHE, else RQ_HOME/flow-cache). The fold cache gives a re-run the same
settled points, so a texture-only or material-only run gets the same key and the check is read back, not computed.

The core is plain numpy (analyse(), heatmap()); collect() reads a Blender object. Run outside Blender for its self
test: python3 rq_flow.py --selftest OUT_DIR"""
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

VERSION = 7
HERE = os.path.dirname(os.path.abspath(__file__))
CM = 165.0 / 63.0              # a unit in centimetres (the figure: 165 cm, 63 units)
PATTERN_UV = ('pattern', 'flat', 'cut')
# the modifiers that only move points: kept when the garment is read; the rest (solidify, subdivision, bevel...)
# are switched off for the read, so the cloth is read as the sheet it was cut and sewn as, point for point
DEFORM = {'ARMATURE', 'SHRINKWRAP', 'SURFACE_DEFORM', 'LATTICE', 'CORRECTIVE_SMOOTH', 'SMOOTH', 'LAPLACIANSMOOTH',
          'DISPLACE', 'MESH_DEFORM', 'HOOK', 'CAST', 'WARP', 'WAVE', 'SIMPLE_DEFORM', 'CLOTH', 'MESH_CACHE',
          'MESH_SEQUENCE_CACHE', 'SOFT_BODY'}
SKIP_NAMES = ('stitch', 'eyelet', 'button', 'buckle', 'plate', 'rivet', 'aglet', 'medallion', 'pendant', 'lace ')
# Until flow_baseline.json is written: loose, so nothing is flagged on guesses.
DEFAULT_LIMITS = {'stretch_p99': 1.35, 'squash_p1': 0.65, 'shear_p99': 25.0, 'strain_p99': 0.45, 'jag_p99': 0.12,
                  'roughness': 0.9, 'peaks': 12, 'peak_max': 0.5, 'seam_mismatch_pct': 8.0, 'fold_radius_p1': 0.05,
                  'slit_len': 2.0, 'holes': 3, 'open_seam_len': 3.0}
LOWER_IS_WORSE = ('squash_p1', 'fold_radius_p1')
LAYER_LIMITS = {'poke_pct': 1.0, 'poke_max': 0.08}     # a layer through the one over it: share of its near points, depth
SLIVER_DEG = 4.0          # triangles with a smaller angle than this on the reference are left out
STRAIN_CAP = 2.0          # the jaggedness reads strain up to this (Green strain 2: the cloth twice its length)
PEAK_JUMP = 0.05          # a peak stands this far (in strain) above its neighbourhood at least
PEAK_SPACING = 1.5        # units: peaks closer than this are one peak


def limits(pose='rest'):
    """The approved band for a pose: flow_baseline.json beside this file ('poses'[pose], else its 'limits', the
    rest pose's), else DEFAULT_LIMITS. Each pose has its own band, from the approved garments in that pose: the rig
    strains even an approved garment in arms-up, so a new garment is flagged only when it does worse than they do."""
    for p in (os.path.join(HERE, 'flow_baseline.json'),
              os.path.join(HERE, '..', '..', 'tools', 'render-queue', 'flow_baseline.json')):
        try:
            b = json.load(open(p))
            return dict(DEFAULT_LIMITS, **(b.get('poses', {}).get(pose) or b['limits'])), p
        except (OSError, ValueError, KeyError):
            continue
    return dict(DEFAULT_LIMITS), None


# ---------------------------------------------------------------------------------------------------- the mesh

def components(n, a, b):
    """Connected components of n nodes joined by edges a-b: a label per node (min-label propagation with pointer
    jumping, numpy only)."""
    lab = np.arange(n)
    if len(a) == 0:
        return lab
    for _ in range(4 * int(math.log2(n + 2)) + 200):
        m = np.minimum(lab[a], lab[b])
        new = lab.copy()
        np.minimum.at(new, a, m)
        np.minimum.at(new, b, m)
        new = new[new]
        new = new[new]
        if np.array_equal(new, lab):
            break
        lab = new
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def edges_of(tris):
    """Each triangle's three edges (corner k to k+1) as ids into the unique edges, the unique edges [e, 2], and for
    each the triangles on it (interior edges have two: t0, t1; t1 is -1 on a boundary)."""
    m = len(tris)
    ev = np.stack([tris, np.roll(tris, -1, axis=1)], axis=2).reshape(-1, 2)
    key = np.sort(ev, axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    tri_of = np.repeat(np.arange(m), 3)
    order = np.argsort(inv, kind='stable')
    s_inv, s_tri = inv[order], tri_of[order]
    first = np.r_[True, s_inv[1:] != s_inv[:-1]]
    t0 = np.full(len(uniq), -1)
    t1 = np.full(len(uniq), -1)
    t0[s_inv[first]] = s_tri[first]
    second = ~first
    t1[s_inv[second]] = s_tri[second]
    return inv.reshape(m, 3), uniq, t0, t1


def _frames(P):
    """2D coordinates of each triangle's corners [m, 3, 2] in its own plane (P: [m, 3, 3])."""
    e1 = P[:, 1] - P[:, 0]
    e2 = P[:, 2] - P[:, 0]
    l1 = np.linalg.norm(e1, axis=1)
    u = e1 / np.maximum(l1, 1e-12)[:, None]
    nrm = np.cross(e1, e2)
    w = np.cross(nrm / np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)[:, None], u)
    out = np.zeros(P.shape[:2] + (2,))
    out[:, 1, 0] = l1
    out[:, 2, 0] = (e2 * u).sum(1)
    out[:, 2, 1] = (e2 * w).sum(1)
    return out


def _wpct(v, w, q):
    """Area-weighted percentile."""
    if len(v) == 0:
        return None
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][min(len(v) - 1, np.searchsorted(c, q / 100.0 * c[-1]))])


# ---------------------------------------------------------------------------------------------------- the check

def analyse(name, X, tris, ref, ref_kind, regions=None, mate=None, lim=None, uv_island=None, exempt=None):
    """The flow of one garment.
    X [n, 3]        its points as they will render (world, posed)
    tris [m, 3]     its triangles
    ref [m, 3, 2]   each triangle's corners on the reference (the flat pattern, in units), or [n, 3] reference
                    points (the garment as built), from which each triangle's own plane is taken
    ref_kind        'pattern', 'pattern-scaled' or 'rest-shape' (said in the report)
    regions         place(p) -> a name such as 'left armhole, back' (see body_places)
    mate [n]        optional seam partner of each point (-1 none)
    uv_island [m]   the pattern piece of each triangle (UV islands), for the pattern seams and the sheet
    Returns (summary dict, fields dict of per-triangle arrays for the heatmap)."""
    lim = lim or limits()[0]
    X = np.asarray(X, float)
    tris = np.asarray(tris, np.int64)
    m = len(tris)
    P = X[tris]
    Dm = ref if ref.ndim == 3 else _frames(np.asarray(ref, float)[tris])
    A = Dm[:, 1] - Dm[:, 0]
    B = Dm[:, 2] - Dm[:, 0]
    det = A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]
    area_ref = 0.5 * np.abs(det)
    good = area_ref > 1e-9 * max(1e-12, np.median(area_ref) if m else 1)
    good &= area_ref > 1e-10
    # (slivers out: a trim's cut leaves needle triangles whose strain is noise, not cloth)
    e0, e1, e2 = A, B, B - A
    l2 = np.stack([(e0 * e0).sum(1), (e1 * e1).sum(1), (e2 * e2).sum(1)], 1)
    good &= 2 * area_ref > np.sin(np.radians(SLIVER_DEG)) * np.sqrt(l2.max(1) * np.sort(l2, 1)[:, 1])
    Ds1, Ds2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    sd = np.where(good, det, 1.0)
    # F = Ds Dm^-1, Dm^-1 = [[B1, -B0], [-A1, A0]] / det
    Fu = (Ds1 * B[:, 1:2] - Ds2 * A[:, 1:2]) / sd[:, None]     # dx/du (along the weft on a pattern)
    Fv = (-Ds1 * B[:, 0:1] + Ds2 * A[:, 0:1]) / sd[:, None]    # dx/dv (along the warp)
    c00, c11, c01 = (Fu * Fu).sum(1), (Fv * Fv).sum(1), (Fu * Fv).sum(1)
    tr, dif = 0.5 * (c00 + c11), np.sqrt(0.25 * (c00 - c11) ** 2 + c01 ** 2)
    lam1 = np.sqrt(np.maximum(tr + dif, 0))
    lam2 = np.sqrt(np.maximum(tr - dif, 0))
    if ref_kind.startswith('pattern'):
        cosang = c01 / np.maximum(np.sqrt(c00 * c11), 1e-12)
        shear = np.degrees(np.abs(np.arcsin(np.clip(cosang, -1, 1))))   # (90 - the angle between the threads)
    else:
        shear = 90.0 - 2.0 * np.degrees(np.arctan2(lam2, np.maximum(lam1, 1e-12)))
    E00, E11, E01 = 0.5 * (c00 - 1), 0.5 * (c11 - 1), 0.5 * c01
    strain = np.sqrt(E00 ** 2 + E11 ** 2 + 2 * E01 ** 2)
    for f in (lam1, lam2, shear, strain):
        f[~good] = np.nan

    cur_n = np.cross(Ds1, Ds2)
    area_cur = 0.5 * np.linalg.norm(cur_n, axis=1)
    unit_n = cur_n / np.maximum(2 * area_cur, 1e-12)[:, None]
    cen = P.mean(1)
    tri_e, E, t0, t1 = edges_of(tris)
    inner = t1 >= 0
    a_, b_ = t0[inner], t1[inner]

    # bending: discrete shells, against a flat sheet (the pattern) or the built shape
    elen = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
    cosd = np.clip((unit_n[a_] * unit_n[b_]).sum(1), -1, 1)
    sind = np.linalg.norm(np.cross(unit_n[a_], unit_n[b_]), axis=1)
    theta = np.arctan2(sind, cosd)
    if ref.ndim == 2:
        R = np.asarray(ref, float)
        rn = np.cross(R[tris][:, 1] - R[tris][:, 0], R[tris][:, 2] - R[tris][:, 0])
        rn /= np.maximum(np.linalg.norm(rn, axis=1), 1e-12)[:, None]
        theta0 = np.arctan2(np.linalg.norm(np.cross(rn[a_], rn[b_]), axis=1), np.clip((rn[a_] * rn[b_]).sum(1), -1, 1))
        theta = np.abs(theta - theta0)
    asum = np.maximum(area_cur[a_] + area_cur[b_], 1e-12)
    bend_e = theta ** 2 * elen[inner] ** 2 / asum ** 2        # (discrete shells' 3 theta^2 |e|^2 / A, over A / 3, / 9)
    bend_e[theta > 2.6] = 0.0               # (a fold back on itself: a solidified shell's rim, not cloth bending)
    bend_t = np.zeros(m)
    np.maximum.at(bend_t, a_, bend_e)
    np.maximum.at(bend_t, b_, bend_e)

    # jaggedness: each triangle's strain against its neighbours'
    s = np.where(good, np.minimum(strain, STRAIN_CAP), 0.0)    # (past the cap it is torn: how far does not matter)
    ok_pair = good[a_] & good[b_]
    pa, pb = a_[ok_pair], b_[ok_pair]
    nsum, ncnt, nmax = np.zeros(m), np.zeros(m), np.full(m, -np.inf)
    np.add.at(nsum, pa, s[pb]); np.add.at(nsum, pb, s[pa])
    np.add.at(ncnt, pa, 1); np.add.at(ncnt, pb, 1)
    np.maximum.at(nmax, pa, s[pb]); np.maximum.at(nmax, pb, s[pa])
    nmean = np.where(ncnt > 0, nsum / np.maximum(ncnt, 1), s)
    # (and the ring beyond: the neighbourhood a peak must stand clear of)
    n2sum, n2cnt = np.zeros(m), np.zeros(m)
    np.add.at(n2sum, pa, nmean[pb]); np.add.at(n2sum, pb, nmean[pa])
    np.add.at(n2cnt, pa, 1); np.add.at(n2cnt, pb, 1)
    ring = np.where(n2cnt > 0, n2sum / np.maximum(n2cnt, 1), nmean)
    excess = np.where(good & (ncnt > 0), s - nmean, 0.0)
    dist = np.maximum(np.linalg.norm(cen[pa] - cen[pb], axis=1), 1e-6)
    grad = np.abs(s[pa] - s[pb]) / dist
    w = np.where(good, area_ref, 0.0)
    spread = math.sqrt(max(_wvar(s, w), 0.0))
    rough = float(math.sqrt(np.average(excess ** 2, weights=w + 1e-12)) / (spread + 0.005)) if w.sum() else None
    mad = float(np.median(np.abs(excess[good]))) if good.any() else 0.0
    peak = good & (ncnt > 0) & (s >= nmax) & (s - ring > max(PEAK_JUMP, 6.0 * mad)) & \
        (s > (_wpct(s[good], w[good], 95) or 0))
    # (one peak per place: the strongest, the others within PEAK_SPACING of it folded into it)
    keep = np.zeros(m, bool)
    taken = []
    for t in np.where(peak)[0][np.argsort(-(s - ring)[peak])][:2000]:
        if taken and np.min(np.linalg.norm(np.array(taken) - cen[t], axis=1)) < PEAK_SPACING:
            continue
        taken.append(cen[t])
        keep[t] = True
    peak_tris, peak = int(peak.sum()), keep

    seams = seam_check(X, tris, tri_e, E, t0, t1, Dm, mate, uv_island)
    gaps = openings(X, tris, E, t1, exempt=exempt)
    gw = w[good]
    S = {
        'reference': ref_kind, 'tris': int(m), 'area': round(float(area_cur.sum()), 2),
        'stretch_p99': _r(_wpct(lam1[good], gw, 99)), 'squash_p1': _r(_wpct(lam2[good], gw, 1)),
        'shear_p99': _r(_wpct(shear[good], gw, 99), 2), 'strain_p50': _r(_wpct(strain[good], gw, 50), 4),
        'strain_p99': _r(_wpct(strain[good], gw, 99), 4),
        'bend_p99': _r(_wpct(bend_t, area_cur, 99), 3),
        'fold_radius_p1': _r(1.0 / math.sqrt(max(_wpct(bend_t, area_cur, 99) or 0, 1e-9)), 3),
        'jag_p99': _r(_wpct(np.abs(excess[good]), gw, 99), 4), 'roughness': _r(rough, 3),
        'grad_p99': _r(float(np.percentile(grad, 99)) if len(grad) else None, 4),
        'smooth_pct': _r(100.0 * gw[np.abs(excess[good]) < 0.02].sum() / max(gw.sum(), 1e-12), 1),
        'peaks': int(peak.sum()), 'peak_tris': peak_tris,
        'peak_max': _r(float((s - ring)[peak].max()) if peak.any() else 0.0, 4),
        'seam_mismatch_pct': _r(max([c['mismatch_pct'] for c in seams] or [0.0]), 2), 'seams': seams[:8],
        'slit_len': _r(sum(h['length'] for h in gaps['holes']), 2), 'holes': len(gaps['holes']),
        'open_seam_len': _r(gaps['open_len'], 2), 'openings': gaps,
    }
    F = {'X': X, 'tris': tris, 'strain': np.where(good, strain, 0.0), 'excess': excess, 'bend': bend_t,
         'normal': unit_n, 'peak': peak, 'ref2d': Dm, 'island': uv_island, 'ref_kind': ref_kind}
    S['spots'] = spots(S, F, lam1, lam2, shear, E, t1, tri_e, regions, lim)
    S['flags'], S['verdict'] = judge(S, lim)
    return S, F


def _wvar(v, w):
    if w.sum() <= 0:
        return 0.0
    mu = np.average(v, weights=w)
    return float(np.average((v - mu) ** 2, weights=w))


def _r(v, k=3):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), k)


def seam_check(X, tris, tri_e, E, t0, t1, Dm, mate, uv_island):
    """Paired seam edges: their lengths on the reference, side A against side B, per seam (a chain of edges)."""
    m = len(tris)
    pairs = []          # (edge's 3D midpoint, length on side A, length on side B, the two sides' piece ids)

    def ref_len(t, k):
        """The reference length of triangle t's edge k (corner k to k+1)."""
        return np.linalg.norm(Dm[t, (k + 1) % 3] - Dm[t, k], axis=-1)
    # (1) pattern seams inside one mesh: an edge whose two triangles put it at different places on the pattern
    if uv_island is not None:
        inner = np.where(t1 >= 0)[0]
        ta, tb = t0[inner], t1[inner]
        ka = np.argmax(tri_e[ta] == inner[:, None], axis=1)
        # the same two points on both sides? (each side's corners, matched by vertex)
        va0, va1 = tris[ta, ka], tris[ta, (ka + 1) % 3]
        ua0, ua1 = Dm[ta, ka], Dm[ta, (ka + 1) % 3]
        kb0 = np.argmax(tris[tb] == va0[:, None], axis=1)
        kb1 = np.argmax(tris[tb] == va1[:, None], axis=1)
        ub0, ub1 = Dm[tb, kb0], Dm[tb, kb1]
        cut = (np.linalg.norm(ua0 - ub0, axis=1) > 1e-4) | (np.linalg.norm(ua1 - ub1, axis=1) > 1e-4)
        if cut.any():
            la = np.linalg.norm(ua1 - ua0, axis=1)[cut]
            lb = np.linalg.norm(ub1 - ub0, axis=1)[cut]
            mid = 0.5 * (X[va0[cut]] + X[va1[cut]])
            # (which copy is side A: the one nearer the start of the seam's spread on the pattern)
            pa_ = 0.5 * (ua0 + ua1)[cut]
            pb_ = 0.5 * (ub0 + ub1)[cut]
            ia, ib = uv_island[ta[cut]], uv_island[tb[cut]]
            pairs.append((E[inner[cut]], mid, la, lb, ia, ib, pa_, pb_))
    # (2) and (3) separate pieces: a declared mate, or boundary edges lying on each other after the sew
    bnd = np.where(t1 < 0)[0]
    if len(bnd):
        bt = t0[bnd]
        bk = np.argmax(tri_e[bt] == bnd[:, None], axis=1)
        bv0, bv1 = tris[bt, bk], tris[bt, (bk + 1) % 3]
        blen = ref_len(bt, bk)
        mt = np.asarray(mate) if mate is not None else infer_mates(X, tris, bv0, bv1)
        if mt is not None:
            look = {}
            for e, (p, q) in enumerate(zip(bv0, bv1)):
                look[(min(p, q), max(p, q))] = e
            A_e, B_e = [], []
            for e, (p, q) in enumerate(zip(bv0, bv1)):
                if mt[p] < 0 or mt[q] < 0:
                    continue
                f = look.get((min(mt[p], mt[q]), max(mt[p], mt[q])))
                if f is not None and f != e and (min(p, q) < min(mt[p], mt[q])):
                    A_e.append(e); B_e.append(f)
            if A_e:
                A_e, B_e = np.array(A_e), np.array(B_e)
                isl = uv_island if uv_island is not None else np.zeros(m, np.int64)
                mid = 0.5 * (X[bv0[A_e]] + X[bv1[A_e]])
                pairs.append((np.stack([bv0[A_e], bv1[A_e]], 1), mid, blen[A_e], blen[B_e], isl[bt[A_e]],
                              isl[bt[B_e]], None, None))
    out = []
    for ev, mid, la, lb, ia, ib, pa_, pb_ in pairs:
        lab = components(int(ev.max()) + 1, ev[:, 0], ev[:, 1])[ev[:, 0]]
        for c in np.unique(lab):
            k = lab == c
            if k.sum() < 3:
                continue
            a, b = la[k].copy(), lb[k].copy()
            if pa_ is not None:            # (one copy of the seam on each side, consistently)
                d0 = (pb_[k] - pa_[k])
                ref_d = d0[np.argmax(np.linalg.norm(d0, axis=1))]
                flip = (d0 @ ref_d) < 0
                a[flip], b[flip] = lb[k][flip], la[k][flip]
            A_, B_ = float(a.sum()), float(b.sum())
            if max(A_, B_) < 1e-6:
                continue
            per = np.abs(a - b) / np.maximum(np.maximum(a, b), 1e-9)
            out.append({'edges': int(k.sum()), 'length_a': round(A_, 3), 'length_b': round(B_, 3),
                        'mismatch_pct': round(100.0 * abs(A_ - B_) / max(A_, B_), 2),
                        'edge_mismatch_p95': round(100.0 * float(np.percentile(per, 95)), 2),
                        'pieces': [int(np.bincount(ia[k]).argmax()), int(np.bincount(ib[k]).argmax())],
                        'at': [round(float(v), 2) for v in mid[k].mean(0)], 'mid': mid[k]})
    out.sort(key=lambda c: -c['mismatch_pct'])
    return out


def openings(X, tris, E, t1, closed=0.06, reach=0.8, exempt=None, exempt_r=1.2):
    """Where the cloth is open that should not be. Each loop of boundary edges is a hem, a neckline, a cuff, a slit or
    a hole: a slit's two sides lie close together all along it (its width, the median distance from each point to
    the loop's far side, is small for its length), a hole is a short loop. And separate pieces whose edges run near
    each other but do not meet are an open seam (closer than reach, farther than closed). exempt [k, 3]: points
    (the lacing's) whose neighbourhood is open on purpose: the panels' edges across a lacing gap are not an open
    seam."""
    bnd = np.where(t1 < 0)[0]
    out = {'holes': [], 'open_len': 0.0, 'open_gap_max': 0.0}
    if not len(bnd):
        return out
    be = E[bnd]
    nb = {}
    for a, b in be:
        nb.setdefault(int(a), []).append(int(b))
        nb.setdefault(int(b), []).append(int(a))
    seen = set()
    for v0 in nb:
        if v0 in seen or len(nb[v0]) != 2:
            continue
        loop, prev, v = [v0], None, v0
        seen.add(v0)
        while True:
            nxt = [u for u in nb[v] if u != prev]
            if not nxt or len(nb[v]) != 2:
                break
            prev, v = v, nxt[0]
            if v == v0 or v in seen:
                break
            seen.add(v)
            loop.append(v)
        if len(loop) < 6:
            continue
        P = X[loop]
        seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
        L = float(seg.sum())
        s_ = np.cumsum(seg) - seg
        k = len(loop)
        # the far side: points at least a quarter of the loop away along it
        far = np.abs(s_[:, None] - s_[None, :])
        far = np.minimum(far, L - far) > 0.25 * L
        step = max(1, k // 400)
        D = np.linalg.norm(P[::step, None] - P[None, :], axis=2)
        D[~far[::step]] = np.inf
        width = float(np.median(D.min(1)))
        if L < 4.0 or width < 0.5:
            c = P.mean(0)
            out['holes'].append({'length': round(L / 2 if width < 0.5 else L, 2), 'width': round(width, 3),
                                 'kind': 'slit' if width < 0.5 and L >= 4.0 else 'hole',
                                 'at': [round(float(x), 2) for x in c]})
    out['holes'].sort(key=lambda h: -h['length'])
    # open seams: boundary points of one piece near another piece's boundary, but not on it
    piece = components(len(X), np.r_[tris[:, 0], tris[:, 1]], np.r_[tris[:, 1], tris[:, 2]])
    bv = np.unique(be)
    if len(np.unique(piece[bv])) > 1:
        cell = {}
        q = np.floor(X[bv] / reach).astype(np.int64)
        for i, c in zip(bv, map(tuple, q)):
            cell.setdefault(c, []).append(i)
        gap = np.full(len(X), np.inf)
        for i, c in zip(bv, map(tuple, q)):
            best = np.inf
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for j in cell.get((c[0] + dx, c[1] + dy, c[2] + dz), ()):
                            if piece[j] != piece[i]:
                                best = min(best, float(np.linalg.norm(X[i] - X[j])))
            gap[i] = best
        if exempt is not None and len(exempt):
            near = np.array([np.linalg.norm(exempt - X[i], axis=1).min() < exempt_r for i in bv])
            gap[bv[near]] = np.inf
        op = (gap > closed) & (gap <= reach)
        e_open = op[be[:, 0]] & op[be[:, 1]]
        out['open_len'] = float(np.linalg.norm(X[be[e_open, 0]] - X[be[e_open, 1]], axis=1).sum())
        out['open_gap_max'] = round(float(gap[op].max()) if op.any() else 0.0, 3)
        if e_open.any():            # (where: the open stretches by 2-unit cells, the longest first)
            mid_ = 0.5 * (X[be[e_open, 0]] + X[be[e_open, 1]])
            ln_ = np.linalg.norm(X[be[e_open, 0]] - X[be[e_open, 1]], axis=1)
            cells = {}
            for c_, l_, m_ in zip(map(tuple, np.floor(mid_ / 2.0).astype(int)), ln_, mid_):
                e_ = cells.setdefault(c_, [0.0, np.zeros(3)])
                e_[0] += l_
                e_[1] += l_ * m_
            top = sorted(cells.values(), key=lambda e_: -e_[0])[:4]
            out['open_at'] = [{'length': round(float(l_), 2), 'at': [round(float(x), 2) for x in m_ / max(l_, 1e-9)]}
                              for l_, m_ in top]
    return out


def infer_mates(X, tris, bv0, bv1, tol=0.06):
    """Boundary points of separate pieces lying on each other after the sew (within tol units): partners."""
    piece = components(len(X), np.r_[tris[:, 0], tris[:, 1]], np.r_[tris[:, 1], tris[:, 2]])
    bverts = np.unique(np.r_[bv0, bv1])
    if len(np.unique(piece[bverts])) < 2:
        return None
    cell = {}
    q = np.floor(X[bverts] / tol).astype(np.int64)
    for i, c in zip(bverts, map(tuple, q)):
        cell.setdefault(c, []).append(i)
    mate = np.full(len(X), -1)
    for i, c in zip(bverts, map(tuple, q)):
        best, bd = -1, tol
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in cell.get((c[0] + dx, c[1] + dy, c[2] + dz), ()):
                        if piece[j] == piece[i]:
                            continue
                        d = float(np.linalg.norm(X[i] - X[j]))
                        if d < bd:
                            best, bd = j, d
        mate[i] = best
    return mate if (mate >= 0).sum() >= 6 else None


def spots(S, F, lam1, lam2, shear, E, t1, tri_e, place, lim, most=8):
    """The worst places on the garment, each with where it is and what it most likely is."""
    X, tris, s, ex = F['X'], F['tris'], F['strain'], F['excess']
    cen = X[tris].mean(1)
    on_edge = np.zeros(len(tris), bool)
    bnd = np.where(t1 < 0)[0]
    on_edge[np.isin(tri_e, bnd).any(1)] = True
    seam_pts = np.concatenate([c['mid'] for c in S['seams']]) if S['seams'] else np.zeros((0, 3))
    score = np.nan_to_num(np.maximum(s / max(lim['strain_p99'], 1e-6), np.abs(ex) / max(lim['jag_p99'], 1e-6)))
    score[F['peak']] *= 1.5
    order = np.argsort(-score)
    out, taken = [], []
    for t in order[:4000]:
        if len(out) >= most or score[t] < 0.5:
            break
        if any(np.linalg.norm(cen[t] - c) < 2.0 for c in taken):
            continue
        taken.append(cen[t])
        near_seam = bool(len(seam_pts)) and float(np.min(np.linalg.norm(seam_pts - cen[t], axis=1))) < 1.0
        where = place(cen[t]) if place else None
        hint = []
        if F['peak'][t] and (on_edge[t] or near_seam):
            hint.append('a seam end or dart tip: the curvature concentrates here')
        elif F['peak'][t]:
            hint.append('an isolated peak: the cut and the body disagree here')
        if math.isfinite(shear[t]) and shear[t] > lim['shear_p99'] and shear[t] > 40 * max(lam1[t] - 1, 1e-3):
            hint.append('the threads shear: grain or bias')
        reg = (where or '').split(',')[0]
        if math.isfinite(lam1[t]) and lam1[t] > lim['stretch_p99'] and \
                any(k in reg for k in ('armhole', 'shoulder', 'hips', 'bust', 'elbow', 'knee', 'seat')):
            hint.append(f'too little ease at the {reg.split()[-1]}')
        elif math.isfinite(lam1[t]) and lam1[t] > lim['stretch_p99']:
            hint.append('stretched past the approved band')
        if math.isfinite(lam2[t]) and lam2[t] < lim['squash_p1']:
            hint.append('cloth crowded: too much ease, or a short paired edge')
        out.append({'n': len(out) + 1, 'tri': int(t), 'at': [round(float(v), 2) for v in cen[t]], 'where': where,
                    'strain': _r(s[t], 4), 'excess': _r(ex[t], 4), 'stretch': _r(lam1[t]), 'squash': _r(lam2[t]),
                    'shear': _r(shear[t], 1), 'peak': bool(F['peak'][t]), 'hint': '; '.join(hint) or 'high strain'})
    for c in S['seams']:
        c.pop('mid', None)
        if place:
            c['where'] = place(np.array(c['at']))
    return out


def judge(S, lim):
    """Flags against the approved band: 'watch' past a limit, 'likely rejection' well past it."""
    flags, worst = [], 'ok'
    for k, v in lim.items():
        if k in LAYER_LIMITS or k in LACE_LIMITS:
            continue
        x = S.get(k)
        if x is None or v is None:
            continue
        low = k in LOWER_IS_WORSE
        over = (x < v) if low else (x > v)
        if not over:
            continue
        far = (x < v / 1.5) if low else (x > (v * 1.5 if k not in ('peaks', 'holes') else v + max(3, v)))
        lvl = 'likely rejection' if far else 'watch'
        flags.append({'metric': k, 'value': x, 'limit': v, 'level': lvl})
        if lvl == 'likely rejection' or worst == 'ok':
            worst = lvl
    # (the worst first: rejections, then by how far past the limit)
    flags.sort(key=lambda f: (f['level'] != 'likely rejection',
                              -(f['limit'] / max(f['value'], 1e-9) if f['metric'] in LOWER_IS_WORSE
                                else f['value'] / max(f['limit'], 1e-9))))
    return flags, worst


# ---------------------------------------------------------------------------------------------------- lacework

LACE_NAMES = ('lace',)        # objects whose name holds one of these are lacing: checked as laces, not as cloth
# float, sink and dev in the lace's own radii (a lace lifted over another at a crossing stands 2 to 3 radii off the
# contour; the contour path is the Hermite curve over the panels, so a gap between two panels is bridged)
LACE_LIMITS = {'lace_float_max': 4.5, 'lace_sink_max': 0.5, 'lace_dev_max': 4.5, 'lace_tension_min': 0.97,
               'lace_slack_max': 1.25, 'lace_arch_min': 0.5, 'lace_flips': 0, 'lace_hits': 0}
LACE_LOWER = ('lace_tension_min', 'lace_arch_min')


def lace_segments(X, T, bins=16):
    """A lace mesh's runs, each one tube (one connected part) between two eyelets: its centre line [k, 3] (the
    tube's points binned along its own long axis, each bin's mean, ends included) and its radius."""
    part = components(len(X), np.r_[T[:, 0], T[:, 1]], np.r_[T[:, 1], T[:, 2]])
    out = []
    for c in np.unique(part):
        P = X[part == c]
        if len(P) < 12:
            continue
        mu = P.mean(0)
        _, _, vt = np.linalg.svd(P - mu, full_matrices=False)
        t = (P - mu) @ vt[0]
        edges = np.linspace(t.min(), t.max(), bins + 1)
        k = np.clip(np.searchsorted(edges, t, side='right') - 1, 0, bins - 1)
        C = np.array([P[k == i].mean(0) for i in range(bins) if (k == i).any()])
        if len(C) < 3:
            continue
        A, B = C[:-1], C[1:]                  # (radius: each point's distance to the centre line's segments)
        AB = B - A
        t_ = np.clip(((P[:, None] - A[None]) * AB[None]).sum(2) / np.maximum((AB * AB).sum(1), 1e-12)[None], 0, 1)
        Q = A[None] + t_[..., None] * AB[None]
        rad = float(np.median(np.min(np.linalg.norm(P[:, None] - Q, axis=2), axis=1)))
        out.append((C, rad))
    return out


def _len(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def _hermite(A, B, nA, nB, s):
    """The contour path between two seats on a surface: a cubic Hermite curve that leaves each seat along the
    surface there (the chord projected on its tangent plane, a chord long), so it arches over a curved shape as the
    shape does, and across a gap between two panels it bridges at the panels' height instead of dipping onto
    whatever lies in the gap. s [k, 1] in 0..1. Returns the points [k, 3] and the normals [k, 3] along it."""
    d = B - A
    L = float(np.linalg.norm(d))

    def tang(n_):
        t = d - n_ * float(d @ n_)
        return t / max(float(np.linalg.norm(t)), 1e-12) * L
    tA, tB = tang(nA), tang(nB)
    h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    H = h00 * A + h10 * tA + h01 * B + h11 * tB
    N = (1 - s) * nA + s * nB
    return H, N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def lace_check(name, X, T, nearest, place=None, lim=None):
    """How one lace object's runs lie on what they cross. nearest(p, across=None) -> (point on the surface, its
    normal outward) or None, for the surfaces under the lace (the boot, the leg); with across (a direction) it
    skips faces standing along it: the walls of an opening's edge, which face across the gap. Per run (one tube
    between two eyelets), the contour path is a curve on the panels between its two seats (_hermite): what a lace
    drawn over the shape lies on, bridging a gap at the panels' height. Along its centre line:
      float   how far it stands off that path (less its radius), in lace radii
      sink    how far it sinks into the surface itself (the tongue, the leg), in lace radii
      dev     how far it leaves the contour path, in lace radii
      tension its length over the contour path's: under 1 it is shorter than the shape it crosses, so it must cut
              into it; well over 1 it is slack
      arch    how far its middle rises over its chord, over how far the contour path's does: 1 follows the curve,
              near 0 a straight taut lace across a curved shape (judged only where the path rises over 2 radii)
    and over the runs: where two cross, the one on top (it should be the same lace all the way up: flips counts the
    crossings that disagree), and hits, two runs passing through each other at a crossing. The loose ends (runs
    much longer than the rest) are left out, and counted."""
    lim = dict(LACE_LIMITS, **(lim or {}))
    segs = [(np.asarray(C), rad) for C, rad in lace_segments(X, T)]
    if not segs:
        return {'runs': 0, 'crossings': 0, 'lace_flips': 0, 'lace_hits': 0, 'loose_ends': 0, 'spots': [], 'flags': [],
                'verdict': 'ok'}, []
    mids = np.array([C[len(C) // 2] for C, _ in segs])
    up = np.linalg.svd(mids - mids.mean(0), full_matrices=False)[2][0]
    up = up if up[2] >= 0 else -up
    segs = [(C if (C[-1] - C[0]) @ up >= 0 else C[::-1], rad) for C, rad in segs]       # (every run climbs)
    lens = np.array([_len(C) for C, _ in segs])
    typical = float(np.median(lens))
    loose = [n for n in range(len(segs)) if lens[n] > 1.7 * typical]
    segs = [sg for n, sg in enumerate(segs) if n not in loose]
    chords = np.array([C[-1] - C[0] for C, _ in segs])
    flat = chords - np.outer(chords @ up, up)
    across = np.linalg.svd(flat, full_matrices=False)[2][0] if len(segs) > 1 else np.array([1.0, 0, 0])
    runs = []
    for C, rad in segs:
        k = len(C)
        d_ = C[-1] - C[0]
        dh = d_ / max(float(np.linalg.norm(d_)), 1e-12)
        a, b = nearest(C[0], dh), nearest(C[-1], dh)
        if a is None or b is None:
            continue
        (A, nA), (B, nB) = a, b
        if (C[0] - A) @ nA < 0 and (C[-1] - B) @ nB < 0:       # (a surface wound inward: its normals point away)
            nA, nB = -nA, -nB
            flip = -1.0
        else:
            flip = 1.0
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
        s = (arc / max(arc[-1], 1e-12))[:, None]
        H, N = _hermite(A, B, nA, nB, s)
        cont = H + N * rad
        gap_env = (((C - H) * N).sum(1) - rad) / rad            # (off the contour path, along its normal)
        dev = np.linalg.norm(C - cont, axis=1) / rad
        sink = np.zeros(k)
        for i in range(1, k - 1):
            q = nearest(C[i], dh)
            if q is not None:
                sink[i] = -((float(np.dot(C[i] - q[0], q[1])) * flip - rad) / rad)
        inner = slice(1, k - 1)                                  # (the ends sit in their eyelets)
        Lc = _len(cont)
        mid = k // 2
        chord = (1 - s) * C[0] + s * C[-1]
        sag_l = float(np.dot(C[mid] - chord[mid], N[mid]))
        sag_c = float(np.dot(cont[mid] - chord[mid], N[mid]))
        arch = sag_l / sag_c if sag_c > 2 * rad else 1.0
        runs.append({'centre': C, 'radius': rad, 'float': float(max(gap_env[inner].max(), 0)),
                     'sink': float(max(sink[inner].max(), 0)), 'dev': float(dev[inner].max()),
                     'tension': _len(C) / max(Lc, 1e-9), 'arch': float(arch), 'mid': C[mid], 'len': _len(C),
                     'normals': N, 'cont': cont, 'sag': sag_c / rad})
    # crossings: runs going opposite ways across the lacing whose centre lines pass near each other
    flips, hits, crossings = 0, 0, []
    dirs = [np.sign(float((r['centre'][-1] - r['centre'][0]) @ across)) or 1.0 for r in runs]
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            A, B = runs[i]['centre'][4:-4], runs[j]['centre'][4:-4]     # (crossings are mid-run)
            if not len(A) or not len(B) or dirs[i] == dirs[j]:
                continue
            D = np.linalg.norm(A[:, None] - B[None], axis=2)
            a, b = np.unravel_index(np.argmin(D), D.shape)
            reach = 2 * (runs[i]['radius'] + runs[j]['radius'])
            if D[a, b] > reach:
                continue
            n_ = runs[i]['normals'][a + 4] + runs[j]['normals'][b + 4]
            n_ = n_ / max(float(np.linalg.norm(n_)), 1e-12)
            top = i if np.dot(A[a] - B[b], n_) > 0 else j
            crossings.append((dirs[top], 0.5 * (A[a] + B[b])))
            if D[a, b] < 0.9 * (runs[i]['radius'] + runs[j]['radius']):        # (through each other, not touching)
                hits += 1
    if crossings:
        main = np.sign(sum(d for d, _ in crossings)) or 1.0
        flips = sum(1 for d, _ in crossings if d != main)
    S = {'runs': len(runs), 'loose_ends': len(loose), 'crossings': len(crossings), 'lace_flips': int(flips),
         'lace_hits': int(hits)}
    if runs:
        S.update(lace_float_max=_r(max(r['float'] for r in runs)), lace_sink_max=_r(max(r['sink'] for r in runs)),
                 lace_dev_max=_r(max(r['dev'] for r in runs)), lace_tension_min=_r(min(r['tension'] for r in runs)),
                 lace_slack_max=_r(max(r['tension'] for r in runs)), lace_arch_min=_r(min(r['arch'] for r in runs)),
                 radius=_r(float(np.median([r['radius'] for r in runs])), 4))
    # each run judged: its worst trouble, as a share of its limit
    spots = []
    for n, r in enumerate(runs):
        why = []
        if r['float'] > lim['lace_float_max']:
            why.append(f"floats {r['float']:.1f} lace radii off the contour")
        if r['sink'] > lim['lace_sink_max']:
            why.append(f"sinks {r['sink']:.1f} lace radii into the surface")
        if r['tension'] < lim['lace_tension_min']:
            why.append(f"taut: {r['tension']:.3f} of the contour's length, so it cuts across the shape")
        if r['tension'] > lim['lace_slack_max']:
            why.append(f"slack: {r['tension']:.2f} of the contour's length")
        if r['dev'] > lim['lace_dev_max']:
            why.append(f"leaves the contour by {r['dev']:.1f} lace radii")
        if r['arch'] < lim['lace_arch_min']:
            why.append(f"straight across the curve (arch {r['arch']:.2f} of the contour's)")
        r['bad'] = max(r['float'] / max(lim['lace_float_max'], 1e-6), r['sink'] / max(lim['lace_sink_max'], 1e-6),
                       r['dev'] / max(lim['lace_dev_max'], 1e-6),
                       (1 - r['tension']) / max(1 - lim['lace_tension_min'], 1e-6),
                       (1 - r['arch']) / max(1 - lim['lace_arch_min'], 1e-6))
        if why:
            spots.append({'run': n, 'at': [round(float(x), 2) for x in r['mid']], 'where': place(r['mid']) if place else None,
                          'why': '; '.join(why), 'bad': round(float(r['bad']), 2)})
    spots.sort(key=lambda x: -x['bad'])
    for i, sp in enumerate(spots):
        sp['n'] = i + 1
    S['spots'] = spots[:8]
    flags, worst = [], 'ok'
    for k, v in lim.items():
        x = S.get(k)
        if x is None:
            continue
        low = k in LACE_LOWER
        if (x < v) if low else (x > v):
            far = (x < v - 2 * (1 - v)) if low else (x > (v * 1.5 if v else 1))
            lvl = 'likely rejection' if far else 'watch'
            flags.append({'metric': k, 'value': x, 'limit': v, 'level': lvl})
            worst = lvl if lvl == 'likely rejection' or worst == 'ok' else worst
    S['flags'], S['verdict'] = flags, worst
    return S, runs


def lace_heatmap(surf, laces, path, lim, panel=320):
    """The lacing over the clay surfaces it crosses, in the four views, framed on the lacing: each run coloured by
    its worst trouble as a share of its limit (the strain ramp: 1.0 at the limit), flagged runs ringed and numbered
    as in flow.json's laces. surf: [(X, tris)]; laces: [(name, S, runs)]."""
    from PIL import Image, ImageDraw
    allC = np.concatenate([r['centre'] for _, _, runs in laces for r in runs])
    lo, hi = allC.min(0) - 1.0, allC.max(0) + 1.0
    cx, cy, cz = 0.5 * (lo + hi)
    span = max(hi[2] - lo[2], np.linalg.norm(hi[:2] - lo[:2])) * 1.05
    k = (panel - 20) / max(span, 1e-6)
    W, H = panel * len(VIEWS) + 160, panel + 40
    sheet = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(sheet)
    d.text((8, 6), 'lacing: runs near or past a limit coloured by their worst trouble / limit (dark: well inside); '
                   'thin: under the surface; rings: flagged runs', fill=INK)
    P = np.concatenate([X[T] for X, T in surf]) if surf else np.zeros((0, 3, 3))
    for vi, (vname, ang) in enumerate(VIEWS):
        a = math.radians(ang)

        def proj(Q):
            xr = (Q[..., 0] - cx) * math.cos(a) + (Q[..., 1] - cy) * math.sin(a)
            yr = -(Q[..., 0] - cx) * math.sin(a) + (Q[..., 1] - cy) * math.cos(a)
            return np.stack([panel / 2 + xr * k, panel / 2 + (cz - Q[..., 2]) * k], -1), yr
        if len(P):
            P2, dep = proj(P)
            keep = ((P2[..., 0] > -20) & (P2[..., 0] < panel + 20) & (P2[..., 1] > -20) & (P2[..., 1] < panel + 20)).any(1)
            ids = _ids(P2[keep], dep[keep].mean(1), panel, panel)
            nrm = np.cross(P[keep][:, 1] - P[keep][:, 0], P[keep][:, 2] - P[keep][:, 0])
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)[:, None]
            nx = nrm[:, 0] * math.cos(a) + nrm[:, 1] * math.sin(a)
            ny = -nrm[:, 0] * math.sin(a) + nrm[:, 1] * math.cos(a)
            lit = 0.55 + 0.45 * np.abs(-0.45 * nx - 0.8 * ny + 0.4 * nrm[:, 2])
            col = np.array(CLAY, float)[None] * lit[:, None]
            img = _paint(ids, col)
            zbuf = np.full(ids.shape, np.inf)
            okp = ids >= 0
            zbuf[okp] = dep[keep].mean(1)[ids[okp]]
        else:
            img = np.zeros((panel, panel, 3), np.uint8) + np.array(BG, np.uint8)
            zbuf = np.full((panel, panel), np.inf)
        tile = Image.fromarray(img)
        td = ImageDraw.Draw(tile)
        rings = []
        for name, S, runs in laces:
            flagged = {sp['run']: sp['n'] for sp in S['spots']}
            for n, r in enumerate(runs):
                q, dq = proj(r['centre'])
                c = tuple(int(v) for v in ramp(np.array([r['bad']]))[0]) if r['bad'] >= 0.8 else (60, 52, 44)
                away = -r['normals'][:, 0] * math.sin(a) + r['normals'][:, 1] * math.cos(a) > 0.15
                pts, under = [], []
                for (u, v), z, aw in zip(q, dq, away):
                    iu, iv = int(u), int(v)
                    behind = zbuf[iv, iu] if 0 <= iu < panel and 0 <= iv < panel else np.inf
                    under.append(behind < z - 0.5 * r['radius'])
                    # (hidden: facing away, or well behind another surface, as the far wall of the boot)
                    pts.append(None if aw or behind < z - 12 * r['radius'] else (float(u), float(v)))
                for i, (p0, p1) in enumerate(zip(pts, pts[1:])):
                    if p0 and p1:       # (a lace sunk under the surface it crosses: drawn thin through it)
                        td.line((p0, p1), fill=c, width=1 if under[i] and under[i + 1] else 3)
                if n in flagged and pts[len(pts) // 2]:
                    u, v = pts[len(pts) // 2]
                    near = [g for g in rings if abs(g[0] - u) < 10 and abs(g[1] - v) < 10]
                    if near:            # (two runs flagged at one crossing: one ring, both numbers)
                        near[0][2].append(flagged[n])
                    else:
                        rings.append((u, v, [flagged[n]], tag(name)))
        for u, v, ns, tg in rings:
            td.ellipse((u - 8, v - 8, u + 8, v + 8), outline=RING, width=2)
            td.text((u + 10, v - 6), tg + ','.join(str(x) for x in sorted(ns)), fill=RING)
        sheet.paste(tile, (vi * panel, 30))
        d.text((vi * panel + 6, 18), vname, fill=INK)
    lx, ly = panel * len(VIEWS) + 8, H - 34
    for i in range(120):
        d.line((lx + i, ly, lx + i, ly + 10), fill=tuple(int(v) for v in ramp(np.array([i / 80.0]))[0]))
    d.text((lx, ly + 12), '0       1.0 = limit  1.5', fill=INK)
    sheet.save(path)
    return path


def _bvh_nearest(geoms, reach=1.5):
    """nearest(p, across=None) over several surfaces (mathutils BVH trees): the closest point on any, and its
    normal. With across (a direction), faces standing along it (the walls of an opening's edge, which face across
    the gap) are passed over for the nearest face that does not."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    trees = [BVHTree.FromPolygons([tuple(p) for p in X], [tuple(int(i) for i in t) for t in T]) for X, T in geoms]

    def nearest(p, across=None):
        best = None
        for tr in trees:
            loc, nrm, _, dist = tr.find_nearest(Vector(p), reach)
            if loc is None:
                continue
            nrm = np.array(nrm)
            if across is not None and abs(float(nrm @ across)) > 0.6:
                for cush in (0.1, 0.3, 0.8):
                    hits = [h for h in tr.find_nearest_range(Vector(p), min(dist + cush, reach))
                            if abs(float(np.array(h[1]) @ across)) <= 0.6]
                    if hits:
                        h = min(hits, key=lambda h: h[3])
                        loc, nrm, dist = h[0], np.array(h[1]), h[3]
                        break
                else:
                    continue
            if best is None or dist < best[2]:
                best = (np.array(loc), nrm, dist)
        return None if best is None else (best[0], best[1])
    return nearest


def _mesh(o, dg):
    ev = o.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        n, m = len(me.vertices), len(me.loop_triangles)
        X = np.empty(n * 3)
        me.vertices.foreach_get('co', X)
        Mw = np.array(ev.matrix_world)
        X = X.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]
        T = np.empty(m * 3, np.int64)
        me.loop_triangles.foreach_get('vertices', T)
        return X, T.reshape(-1, 3)
    finally:
        ev.to_mesh_clear()


# ---------------------------------------------------------------------------------------------------- the band

FLOORS = {'stretch_p99': 1.03, 'squash_p1': 0.97, 'shear_p99': 2.0, 'strain_p99': 0.02, 'jag_p99': 0.01,
          'roughness': 0.3, 'peaks': 2, 'peak_max': 0.05, 'seam_mismatch_pct': 1.0, 'fold_radius_p1': None,
          'slit_len': 0.5, 'holes': 1, 'open_seam_len': 0.5}
MARGIN = 1.25


def badness(k, v):
    """How bad a value is, 0 for perfect, the same way up for every metric."""
    return v - 1 if k == 'stretch_p99' else 1 - v if k == 'squash_p1' else 1 / max(v, 1e-9) \
        if k == 'fold_radius_p1' else v


def calibrate(approved, outliers=None):
    """The limits from approved garments' summaries ({name: summary}): each metric's worst approved value with a
    margin (x 1.25 on how far it is from perfect; peaks: the most seen + 2), never tighter than FLOORS, so a
    garment as good as the approved ones is not flagged. A garment that carries no strain at all (a rigid sole) says
    nothing and is left out. One approved garment far worse than all the others on a metric (over 3x the next worst)
    is a fault John did not see (the vest's sides under the hanging arm): it is left out of that metric's band and
    listed in outliers (a dict to fill: metric -> [garment, value])."""
    live = {n: S for n, S in approved.items() if (S.get('strain_p99') or 0) > 1e-4}
    out = {}
    for k in DEFAULT_LIMITS:
        vals = sorted(((badness(k, S[k]), n, S[k]) for n, S in live.items() if S.get(k) is not None), reverse=True)
        if not vals:
            out[k] = DEFAULT_LIMITS[k]
            continue
        if len(vals) >= 3 and vals[0][0] > 3 * max(vals[1][0], 1e-6) and k not in ('peaks', 'holes'):
            if outliers is not None:
                outliers.setdefault(k, []).append([vals[0][1], vals[0][2]])
            vals = vals[1:]
        worst = vals[0][0]
        if k in ('peaks', 'holes'):
            v = worst + 2
        elif k == 'stretch_p99':
            v = 1 + worst * MARGIN
        elif k == 'squash_p1':
            v = 1 - worst * MARGIN
        elif k == 'fold_radius_p1':
            v = 1 / (worst * MARGIN)
        else:
            v = worst * MARGIN
        f = FLOORS.get(k)
        if f is not None:
            v = min(v, f) if k in LOWER_IS_WORSE else max(v, f)
        out[k] = round(float(v), 4)
    return out


# ---------------------------------------------------------------------------------------------------- places

def body_places(f):
    """place(p) for the base figure module f (female.py's sizes at this commit): 'left armhole, back'. The figure
    faces -y; its right side is -x."""
    sh, hip, ank, up, fore = f.SH_Z, f.HIP_Z, f.ANKLE_Z, f.UPPER, f.FORE
    knee, wrist, chest, neck, shx = hip - f.THIGH, sh - up - fore, f.CHEST_Z, f.NECK_Z, f.SH_X
    waist = 0.5 * (chest + hip)
    bands = [('neck', neck - 1.5, 99), ('shoulder', sh - 1.2, neck - 1.5), ('bust', chest - 2.5, sh - 1.2),
             ('waist', waist - 2.5, chest - 2.5), ('hips', hip - 3.0, waist - 2.5), ('seat', hip - 6.0, hip - 3.0),
             ('thigh', knee + 2.0, hip - 6.0), ('knee', knee - 2.0, knee + 2.0), ('calf', ank + 4.0, knee - 2.0),
             ('ankle', ank - 1.5, ank + 4.0), ('foot', -99, ank - 1.5)]

    def place(p):
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        side = 'right' if x < 0 else 'left'
        face = 'front' if y < 0 else 'back'
        # (the arm: outside the shoulder's width and above the hips; lower down the hanging hand and the hip overlap,
        # and cloth there is the hip's)
        if abs(x) > 0.85 * shx and wrist - 1.0 < z < sh + 1.0 and z > hip + 3.0:
            r = 'armhole' if z > sh - 3.5 and abs(x) < 1.2 * shx else 'upper arm' if z > sh - up + 1.5 else \
                'elbow' if z > sh - up - 1.5 else 'forearm' if z > wrist + 1.0 else 'wrist'
            return f'{side} {r}, {face}'
        if abs(x) < 0.12 * shx and z < hip:
            side = 'middle'
        r = next((n for n, lo, hi in bands if lo <= z < hi), 'hem')
        if z < hip - 6.0 and abs(x) < 0.12 * shx:
            r = 'crotch' if z > hip - 9.0 else 'between the legs'
        if abs(x) < 0.12 * shx:
            return f'{r}, {face} middle'
        return f'{side} {r}, {face}'
    return place


# ---------------------------------------------------------------------------------------------------- heatmap

CLAY, BG, INK, RING = (214, 212, 206), (246, 245, 242), (40, 40, 44), (205, 48, 40)
RAMP = [(0.0, (214, 212, 206)), (0.35, (158, 197, 244)), (0.7, (85, 152, 231)), (1.0, (37, 106, 191)),
        (1.5, (13, 54, 107))]      # (one hue, light to dark: 1.0 is the approved limit, past it the darkest step)
VIEWS = [('front', 0.0), ('3/4', 45.0), ('profile', 90.0), ('back', 180.0)]


def ramp(t):
    t = np.clip(np.nan_to_num(t), 0, RAMP[-1][0])
    xs = np.array([a for a, _ in RAMP])
    cs = np.array([c for _, c in RAMP], float)
    return np.stack([np.interp(t, xs, cs[:, k]) for k in range(3)], -1)


def _ids(P2, depth, W, H):
    """An id buffer: each pixel the index of the nearest triangle on it (-1 for none). P2 [m, 3, 2] pixel corners,
    depth [m] (larger is farther): drawn far to near, so the near ones cover."""
    from PIL import Image, ImageDraw
    im = Image.new('I', (W, H), -1)
    d = ImageDraw.Draw(im)
    for t in np.argsort(-depth):
        p = P2[t]
        d.polygon([(float(p[0, 0]), float(p[0, 1])), (float(p[1, 0]), float(p[1, 1])), (float(p[2, 0]), float(p[2, 1]))],
                  fill=int(t))
    return np.array(im, dtype=np.int64)


def _paint(ids, col):
    img = np.zeros(ids.shape + (3,)) + BG
    k = ids >= 0
    img[k] = col[ids[k]]
    return img.astype(np.uint8)


def tag(name):
    """A garment's short label on the heatmap: the initials of its name and its side, 'Boot.L' -> 'BL'."""
    base, _, side = name.partition('.')
    return ''.join(w[0] for w in base.split() if w).upper() + (side[:1].upper() if side else '')


def heatmap(items, path, lim, title='', panel=260):
    """The heatmap sheet: one row per field (strain, jaggedness, bending), the four views and the flat pattern,
    every garment of the build together. items: [(name, summary, fields)]."""
    from PIL import Image, ImageDraw
    rows = [('strain', 'strain / approved limit', lambda F: F['strain'] / max(lim['strain_p99'], 1e-6)),
            ('jag', 'jaggedness / approved limit', lambda F: np.abs(F['excess']) / max(lim['jag_p99'], 1e-6)),
            ('bend', 'bending (fold radius)', lambda F: np.sqrt(F['bend']) * max(lim['fold_radius_p1'], 1e-6))]
    allX = np.concatenate([F['X'] for _, _, F in items])
    zlo, zhi = allX[:, 2].min(), allX[:, 2].max()
    r = np.linalg.norm(allX[:, :2] - allX[:, :2].mean(0), axis=1).max()
    Hpx = panel
    k = (Hpx - 24) / max(zhi - zlo, 1e-6)
    Wpx = int(min(panel, max(120, 2 * r * k + 24)))
    k = min(k, (Wpx - 12) / max(2 * r, 1e-6))
    pat = any(F['ref_kind'].startswith('pattern') for _, _, F in items)
    pw = int(panel * 1.4) if pat else 0
    W, H = len(VIEWS) * Wpx + pw + 150, len(rows) * (Hpx + 18) + 40
    sheet = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(sheet)
    d.text((8, 6), title[:160], fill=INK)
    cx, cy = allX[:, 0].mean(), allX[:, 1].mean()
    allP = np.concatenate([F['X'][F['tris']] for _, _, F in items])
    allN = np.concatenate([F['normal'] for _, _, F in items])
    views = []          # (each view's id buffer, drawn once and painted for every field)
    for vname, ang in VIEWS:
        a = math.radians(ang)
        xr = (allP[..., 0] - cx) * math.cos(a) + (allP[..., 1] - cy) * math.sin(a)
        yr = -(allP[..., 0] - cx) * math.sin(a) + (allP[..., 1] - cy) * math.cos(a)
        P2 = np.stack([Wpx / 2 + xr * k, 12 + (zhi - allP[..., 2]) * k], -1)
        nx = allN[:, 0] * math.cos(a) + allN[:, 1] * math.sin(a)
        ny = -allN[:, 0] * math.sin(a) + allN[:, 1] * math.cos(a)
        lit = 0.55 + 0.45 * np.abs(-0.45 * nx - 0.8 * ny + 0.4 * allN[:, 2])
        views.append((_ids(P2, yr.mean(1), Wpx, Hpx), lit))
    pat_ids = pattern_ids(items, pw, Hpx) if pat else None
    for ri, (key, label, fn) in enumerate(rows):
        y0 = 30 + ri * (Hpx + 18)
        d.text((len(VIEWS) * Wpx + pw + 8, y0 + 4), label, fill=INK)
        for vi, (vname, ang) in enumerate(VIEWS):
            ids, lit = views[vi]
            col = np.concatenate([ramp(fn(F)) for _, _, F in items]) * lit[:, None]
            sheet.paste(Image.fromarray(_paint(ids, np.clip(col, 0, 255))), (vi * Wpx, y0 + 14))
            a = math.radians(ang)
            if ri == 0:
                d.text((vi * Wpx + 6, y0), vname, fill=INK)
            # the worst spots, ringed and numbered, in the views where they show (not hidden behind the cloth)
            if ri == 2:
                continue
            base = 0
            for name, S, F in items:
                for sp in S['spots'][:5]:
                    g = base + sp['tri']
                    yy, xx = np.nonzero(ids == g)
                    if len(xx) == 0:
                        continue
                    X0, Y0 = vi * Wpx + xx.mean(), y0 + 14 + yy.mean()
                    d.ellipse((X0 - 7, Y0 - 7, X0 + 7, Y0 + 7), outline=RING, width=2)
                    d.text((X0 + 8, Y0 - 6), f"{tag(name)}{sp['n']}", fill=RING)
                base += len(F['tris'])
        if pat_ids is not None:
            col = np.concatenate([ramp(fn(F)) for _, _, F in items])
            sheet.paste(Image.fromarray(_paint(pat_ids, col)), (len(VIEWS) * Wpx, y0 + 14))
            if ri == 0:
                d.text((len(VIEWS) * Wpx + 6, y0), 'pattern (flat, as cut)', fill=INK)
    # the legend: the ramp, and what 1.0 means
    lx, ly = len(VIEWS) * Wpx + pw + 8, H - 34
    for i in range(120):
        c = tuple(int(v) for v in ramp(np.array([i / 80.0]))[0])
        d.line((lx + i, ly, lx + i, ly + 10), fill=c)
    d.text((lx, ly + 12), '0       1.0 = limit  1.5', fill=INK)
    d.text((lx, ly - 30), 'rings: worst spots, as', fill=INK)
    d.text((lx, ly - 18), 'numbered in flow.json', fill=INK)
    sheet.save(path)
    return path


def pattern_ids(items, W, H):
    """The flat pattern pieces side by side, as an id buffer over all the items' triangles (offsets as in heatmap):
    where on the cut the fault lies."""
    pieces, base = [], 0
    for name, S, F in items:
        if F['ref_kind'].startswith('pattern') and F['island'] is not None:
            for isl in np.unique(F['island']):
                k = np.where(F['island'] == isl)[0]
                if len(k) >= 8:
                    pieces.append((F['ref2d'][k], base + k))
        base += len(F['tris'])
    if not pieces:
        return None
    # (lay the pieces in a row, each at its own place: a pattern's islands may sit far apart on the UV)
    off, laid = 0.0, []
    for P2, ids in pieces:
        Q = P2 - P2.reshape(-1, 2).min(0)
        laid.append((Q + np.array([off, 0.0]), ids))
        off += Q.reshape(-1, 2)[:, 0].max() + 1.0
    span = np.concatenate([q.reshape(-1, 2) for q, _ in laid]).max(0)
    k = min((W - 12) / max(span[0], 1e-6), (H - 12) / max(span[1], 1e-6))
    P = np.concatenate([q for q, _ in laid])
    ids = np.concatenate([i for _, i in laid])
    px = np.stack([6 + P[..., 0] * k, H - 6 - P[..., 1] * k], -1)
    buf = _ids(px, np.zeros(len(px)), W, H)
    return np.where(buf >= 0, ids[np.maximum(buf, 0)], -1)


# ---------------------------------------------------------------------------------------------------- Blender

def is_cloth(o, tris_n, pieces_n):
    nm = o.name.lower()
    if any(s in nm for s in SKIP_NAMES) or any(s.strip() and s.strip() in nm
                                               for s in os.environ.get('RQ_FLOW_SKIP', '').split(',')):
        return False
    return tris_n >= 200 and pieces_n <= max(8, tris_n // 400)     # (many tiny parts: hardware, stitches)


def collect(o, dg):
    """One Blender mesh object as analyse() takes it: the posed points, triangles, the reference and its kind,
    the pattern pieces, any declared seam mates. None when it is not cloth or has no reference."""
    ev = o.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        n, m = len(me.vertices), len(me.loop_triangles)
        if not n or not m:
            return None
        X = np.empty(n * 3)
        me.vertices.foreach_get('co', X)
        Mw = np.array(ev.matrix_world)
        X = X.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]
        T = np.empty(m * 3, np.int64)
        me.loop_triangles.foreach_get('vertices', T)
        T = T.reshape(-1, 3)
        Lp = np.empty(m * 3, np.int64)
        me.loop_triangles.foreach_get('loops', Lp)
        Lp = Lp.reshape(-1, 3)
        pieces = components(n, np.r_[T[:, 0], T[:, 1]], np.r_[T[:, 1], T[:, 2]])
        if not is_cloth(o, m, int(pieces.max()) + 1):
            return None
        uvl = next((me.uv_layers[nm] for nm in PATTERN_UV if nm in me.uv_layers), None)
        named = uvl is not None
        uvl = uvl or me.uv_layers.active
        R3 = None
        if len(o.data.vertices) == n:
            R3 = np.empty(n * 3)
            o.data.vertices.foreach_get('co', R3)
            Mo = np.array(o.matrix_world)
            R3 = R3.reshape(-1, 3) @ Mo[:3, :3].T + Mo[:3, 3]
        mate = None
        if 'seam_mate' in me.attributes and me.attributes['seam_mate'].domain == 'POINT':
            mate = np.empty(n, np.int64)
            me.attributes['seam_mate'].data.foreach_get('value', mate)
        if uvl is not None:
            UV = np.empty(len(me.loops) * 2)
            uvl.data.foreach_get('uv', UV)
            ref = UV.reshape(-1, 2)[Lp]
            d1, d2 = ref[:, 1] - ref[:, 0], ref[:, 2] - ref[:, 0]
            a_uv = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]).sum()
            S3 = (R3 if R3 is not None else X)[T]
            a_3d = 0.5 * np.linalg.norm(np.cross(S3[:, 1] - S3[:, 0], S3[:, 2] - S3[:, 0]), axis=1).sum()
            ratio = math.sqrt(a_uv / max(a_3d, 1e-12))
            kind = 'pattern'
            if not (0.7 < ratio < 1.4) or not np.isfinite(ratio):
                if not named and R3 is not None:
                    uvl = None           # (a texture's UV, not a pattern: use the garment as built instead)
                else:
                    ref = ref / max(ratio, 1e-9)
                    kind = 'pattern-scaled'
            why = None
            if uvl is not None:
                # the pattern's pieces: triangles joined across edges where the pattern agrees
                isl = uv_islands(T, ref)
                why = pattern_fault(T, ref, R3, isl) if R3 is not None else None
                if why is None:
                    return dict(X=X, tris=T, ref=ref, kind=kind, mate=mate, island=isl, uv=uvl.name)
        if R3 is None:
            return None
        return dict(X=X, tris=T, ref=R3, kind='rest-shape', mate=mate, island=None, uv=None,
                    pattern_rejected=why if uvl is not None else None)
    finally:
        ev.to_mesh_clear()


def pattern_fault(T, ref, R3, isl):
    """Why a UV is not the garment's flat pattern, or None if it is. A pattern laid on the body as built takes it with
    a smooth, moderate strain; a texture map (banded, stretched or folded where it was never cut) does not."""
    S, _ = analyse('pattern', R3, T, ref, 'pattern', None, lim=dict(DEFAULT_LIMITS), uv_island=isl)
    if (S['strain_p50'] or 0) > 0.08 or (S['strain_p99'] or 0) > 1.0 or (S['roughness'] or 0) > 0.6:
        return (f"the UV is not a flat pattern of the garment as built (strain median {S['strain_p50']}, "
                f"99% {S['strain_p99']}, roughness {S['roughness']}): measured against the garment as built")
    return None


def uv_islands(T, ref):
    tri_e, E, t0, t1 = edges_of(T)
    inner = np.where(t1 >= 0)[0]
    ta, tb = t0[inner], t1[inner]
    ka = np.argmax(tri_e[ta] == inner[:, None], axis=1)
    va0 = T[ta, ka]
    kb0 = np.argmax(T[tb] == va0[:, None], axis=1)
    same = np.linalg.norm(ref[ta, ka] - ref[tb, kb0], axis=1) < 1e-4
    return components(len(T), ta[same], tb[same])


def cache_dir():
    d = os.environ.get('RQ_FLOW_CACHE') or (os.path.join(os.environ['RQ_HOME'], 'flow-cache')
                                             if os.environ.get('RQ_HOME') else
                                             os.path.join(os.path.expanduser('~'), '.cache', 'emberwood-flow'))
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except OSError:
        return None


def cached_analyse(name, g, place, lim):
    """analyse() through the cache: the key is the posed points, the reference, the triangles, the version and the
    limits, so the same settled folds (the fold cache) give the same answer without computing it."""
    h = hashlib.sha1()
    for a in (np.round(g['X'], 4), np.round(np.asarray(g['ref']), 4), g['tris']):
        h.update(np.ascontiguousarray(a).tobytes())
    if g.get('exempt') is not None:
        h.update(np.ascontiguousarray(np.round(g['exempt'], 3)).tobytes())
    h.update(repr((VERSION, g['kind'], sorted(lim.items()))).encode())
    key = h.hexdigest()[:20]
    d = cache_dir()
    p = os.path.join(d, key) if d else None
    if p and os.path.exists(p + '.json') and os.path.exists(p + '.npz'):
        try:
            S = json.load(open(p + '.json'))
            z = np.load(p + '.npz', allow_pickle=False)
            F = {k: z[k] for k in z.files}
            F['island'] = F['island'] if F['island'].size else None
            F['ref_kind'] = g['kind']
            S['cached'] = True
            return S, F
        except (OSError, ValueError, KeyError):
            pass
    S, F = analyse(name, g['X'], g['tris'], np.asarray(g['ref'], float), g['kind'], place, g.get('mate'), lim=lim,
                   uv_island=g.get('island'), exempt=g.get('exempt'))
    S['key'] = key
    if p:
        try:
            json.dump(S, open(p + '.json', 'w'))
            np.savez_compressed(p + '.npz', **{k: (v if v is not None else np.zeros(0)) for k, v in F.items()
                                               if k != 'ref_kind'})
        except OSError:
            pass
    S['cached'] = False
    return S, F


def run(objects, out, female=None, stem='flow', title='', heat='always', write=True, pose='rest'):
    """The flow check on the given Blender objects (the garments): DIR/flow.json (or STEM.json) and the heatmap.
    Returns the report. Never raises: a garment that cannot be read is listed with why. heat: 'always', or
    'flagged' (a heatmap only when something is flagged); write=False returns the report without writing it."""
    import bpy
    t = time.time()
    lim, src = limits(pose)
    lim = dict(LAYER_LIMITS, **dict(LACE_LIMITS, **lim))
    place = body_places(female) if female is not None else None
    off = []
    for o in objects:
        for md in getattr(o, 'modifiers', ()):
            if md.type not in DEFORM and md.show_viewport:
                md.show_viewport = False
                off.append(md)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    rep = {'version': VERSION, 'pose': pose, 'limits': lim, 'limits_from': os.path.basename(src) if src else 'defaults',
           'garments': {}, 'skipped': {}}
    items, geoms, lace_obs = [], [], []
    lace_pts = [_mesh(o, dg)[0][::6] for o in objects if any(k in o.name.lower() for k in LACE_NAMES)]
    exempt = np.concatenate(lace_pts) if lace_pts else None
    for o in objects:
        if any(k in o.name.lower() for k in LACE_NAMES):
            lace_obs.append(o)
            continue
        try:
            g = collect(o, dg)
            if g is not None:
                g['exempt'] = exempt
            if g is None:
                rep['skipped'][o.name] = 'not cloth, or no reference to measure against'
                continue
            S, F = cached_analyse(o.name, g, place, lim)
            geoms.append((o.name, g['X'], g['tris']))
            S['uv'] = g.get('uv')
            if g.get('pattern_rejected'):
                S['pattern_rejected'] = g['pattern_rejected']
            rep['garments'][o.name] = S
            items.append((o.name, S, F))
        except Exception as e:          # (a check must never fail the render)
            rep['skipped'][o.name] = f'{type(e).__name__}: {e}'
    if len(geoms) > 1:
        try:
            rep['layers'] = layer_check(geoms, place)
            for r in rep['layers']:
                lvl = 'likely rejection' if r['poke_pct'] > 3 * lim['poke_pct'] or r['poke_max'] > 3 * lim['poke_max'] \
                    else 'watch' if r['poke_pct'] > lim['poke_pct'] or r['poke_max'] > lim['poke_max'] else 'ok'
                r['verdict'] = lvl
        except Exception as e:
            rep['layers_error'] = f'{type(e).__name__}: {e}'
    for md in off:
        md.show_viewport = True
    if off:
        bpy.context.view_layer.update()
    off = []
    if lace_obs:
        try:
            # (the surfaces the lace lies on as rendered, thickness included: the lace sits on the leather's face)
            dg = bpy.context.evaluated_depsgraph_get()
            surf = [_mesh(bpy.data.objects[g[0]], dg) if g[0] in bpy.data.objects else (g[1], g[2]) for g in geoms]
            body = bpy.data.objects.get('Skin body')
            if body is not None:
                surf.append(_mesh(body, dg))
            nearest = _bvh_nearest(surf)
            laces = []
            rep['laces'] = {}
            for o in lace_obs:
                X, T = _mesh(o, dg)
                S, runs = lace_check(o.name, X, T, nearest, place, {k: lim[k] for k in LACE_LIMITS if k in lim})
                rep['laces'][o.name] = S
                rep['skipped'].pop(o.name, None)
                laces.append((o.name, S, runs))
            try:        # (the lacing and the surfaces round it, kept for working on the check without Blender)
                allC = np.concatenate([r['centre'] for _, _, runs in laces for r in runs])
                keep = {}
                for n_, (Xs, Ts) in enumerate(surf):
                    c = Xs[Ts].mean(1)
                    m = (np.linalg.norm(c[:, None] - allC[None, ::3], axis=2).min(1) < 2.5)
                    keep['sX%d' % n_], keep['sT%d' % n_] = Xs, Ts[m]
                for o in lace_obs:
                    keep['lX_' + o.name], keep['lT_' + o.name] = _mesh(o, dg)
                np.savez_compressed(os.path.join(out, stem + '-lace-dump.npz'), **keep)
            except Exception:
                pass
            if laces and (heat == 'always' or any(S['verdict'] != 'ok' for _, S, _ in laces)):
                lb = [sf for g, sf in zip(geoms, surf) if not g[0].lower().startswith(('hose',))]
                rep['lace_heatmap'] = os.path.basename(lace_heatmap(lb or surf, laces,
                                                                    os.path.join(out, stem + '-lace.png'), lim))
        except Exception as e:
            rep['laces_error'] = f'{type(e).__name__}: {e}'
    for md in off:
        md.show_viewport = True
    if off:
        bpy.context.view_layer.update()
    order = {'ok': 0, 'watch': 1, 'likely rejection': 2}
    # (the layer check is reported but not yet in the verdict: on the riding habit it reads hems and floors as pokes;
    # it joins the verdict once it is calibrated against renders where a layer really comes through)
    rep['verdict'] = max([S['verdict'] for S in rep['garments'].values()] +
                         [S['verdict'] for S in rep.get('laces', {}).values()] or ['ok'], key=order.get)
    if items and (heat == 'always' or rep['verdict'] != 'ok'):
        try:
            rep['heatmap'] = os.path.basename(heatmap(items, os.path.join(out, stem + '-heat.png'), lim,
                                                      title or f'flow: {rep["verdict"]}'))
        except Exception as e:
            rep['heatmap_error'] = f'{type(e).__name__}: {e}'
    rep['seconds'] = round(time.time() - t, 2)
    if write:
        with open(os.path.join(out, stem + '.json'), 'w') as f:
            json.dump(rep, f, indent=1)
    print(f'rq flow{"" if stem == "flow" else " " + stem}: ' + line(rep), flush=True)
    return rep


def layer_check(geoms, place=None, reach=1.0, through=0.03, most=20000, deepest=0.5):
    """Layers poking through the layer over them (the petticoat's gathers through the corset, a skirt through the
    bodice). For each pair of garments, the points of one that lie within reach of the other's surface: if most of
    them are on its inside, it is worn under it, and those standing out through it (more than through past its
    surface) poke through. geoms: [(name, X [n, 3], tris)], every surface's normals facing out. Uses mathutils."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    trees = {n: BVHTree.FromPolygons([tuple(p) for p in X], [tuple(int(i) for i in t) for t in T])
             for n, X, T in geoms}
    # (a point past the over layer's edge is beside it, not through it: faces on a boundary don't count)
    rim = {}
    for n, X, T in geoms:
        tri_e, _, _, t1 = edges_of(T)
        rim[n] = (t1[tri_e] < 0).any(1)
    out = []
    for over, Xo, To in geoms:
        lo, hi = Xo.min(0) - reach, Xo.max(0) + reach
        for under, Xu, Tu in geoms:
            if under == over:
                continue
            P = Xu[np.all((Xu > lo) & (Xu < hi), axis=1)]
            P = P[::max(1, len(P) // most)]
            if len(P) < 50:
                continue
            d, at, edge = [], [], []
            for p in P:
                loc, nrm, fi, dist = trees[over].find_nearest(Vector(p), reach)
                if loc is None:
                    continue
                d.append((Vector(p) - loc).dot(nrm))
                at.append(p)
                edge.append(bool(rim[over][fi]))
            if len(d) < 50:
                continue
            d, at = np.array(d), np.array(at)
            inside = (d < 0).mean()
            if inside < 0.7:            # (mostly outside it: worn over it, or beside it; not this pair's way round)
                continue
            # through: just outside the over layer (not far past it, which is another surface), under its cloth
            poke = (d > through) & (d < deepest) & ~np.array(edge)
            r = {'under': under, 'over': over, 'near': int(len(d)), 'poke_pct': round(100.0 * poke.mean(), 2),
                 'poke_max': round(float(d[poke].max()) if poke.any() else 0.0, 3), 'spots': []}
            taken = []
            for i in np.argsort(-np.where(poke, d, -np.inf))[:2000]:
                if not poke[i] or len(taken) >= 5:
                    break
                if taken and np.min(np.linalg.norm(np.array(taken) - at[i], axis=1)) < 1.5:
                    continue
                taken.append(at[i])
                r['spots'].append({'at': [round(float(x), 2) for x in at[i]], 'out': round(float(d[i]), 3),
                                   'where': place(at[i]) if place else None})
            out.append(r)
    out.sort(key=lambda r: -r['poke_pct'])
    return out


def line(rep):
    """One line for the log and the queue's reply."""
    bits = []
    for name, S in rep['garments'].items():
        if S['verdict'] == 'ok':
            continue
        f = S['flags'][0]
        sp = next((p for p in S['spots'] if p.get('where')), None)
        bits.append(f"{name} {S['verdict']}: {f['metric']} {f['value']} (limit {f['limit']})"
                    + (f", worst at the {sp['where']} ({sp['hint']})" if sp else ''))
    for name, S in rep.get('laces', {}).items():
        if S['verdict'] != 'ok':
            sp = S['spots'][0] if S['spots'] else {}
            bits.append(f"{name} {S['verdict']}: " + (f"run {sp.get('n')} at the {sp.get('where')}: {sp.get('why')}"
                                                       if sp else ', '.join(f['metric'] for f in S['flags'])))
    for r in rep.get('layers', [])[:0]:
        if r['verdict'] != 'ok':
            sp = r['spots'][0] if r['spots'] else {}
            bits.append(f"{r['under']} pokes through {r['over']} ({r['poke_pct']}% of it near, {r['poke_max']} out)"
                        + (f" at the {sp['where']}" if sp.get('where') else ''))
    n = len(rep['garments'])
    return f"{rep['verdict']}, {n} garment{'s' * (n != 1)} checked in {rep.get('seconds', 0)} s" + \
        ('; ' + '; '.join(bits) if bits else '')


# ---------------------------------------------------------------------------------------------------- self test

def _sleeve(n_u=48, n_v=40, r=2.2, length=16.0, bump=0.0, short_seam=0.0, slit=False):
    """A sleeve: a flat rectangle rolled into a tube, its two long edges sewn (a pattern seam). bump stretches
    a small patch (an isolated fault); short_seam cuts one seam edge shorter on the pattern than the other."""
    W = 2 * math.pi * r
    us, vs = np.linspace(0, W, n_u + 1), np.linspace(0, length, n_v + 1)
    X = np.array([(r * math.cos(u / r), r * math.sin(u / r), -v) for v in vs for u in us[:-1]])
    idx = lambda i, j: j * n_u + (i % n_u)      # noqa: E731
    tris, ref = [], []
    for j in range(n_v):
        for i in range(n_u):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            ua, ub = us[i], us[i + 1]
            va, vb = vs[j], vs[j + 1]
            if slit and i == 12 and 12 <= j < 28:      # (a strip of cloth left out: a slit 6.4 units long)
                continue
            f = (lambda v: 1 - short_seam * v / length) if i == n_u - 1 else (lambda v: 1.0)
            tris += [(a, b, c), (a, c, d)]
            ref += [[(ua, -va), (ub, -va * f(va)), (ub, -vb * f(vb))], [(ua, -va), (ub, -vb * f(vb)), (ua, -vb)]]
    X = X.copy()
    if bump:
        p0 = np.array([-r, 0.0, -8.0])       # (opposite the seam)
        dd = np.linalg.norm(X - p0, axis=1)
        X += (bump * np.exp(-(dd / 0.9) ** 2))[:, None] * (X - np.array([0, 0, -8.0])) / r
    return X, np.array(tris), np.array(ref, float)


def _tube(C, rad, sides=6):
    """A closed tube round the centre line C [k, 3] (the self test's laces)."""
    V, F = [], []
    for i, c in enumerate(C):
        t = C[min(i + 1, len(C) - 1)] - C[max(i - 1, 0)]
        t = t / max(np.linalg.norm(t), 1e-12)
        u = np.cross(t, [0.0, 0.0, 1.0])
        u = u / max(np.linalg.norm(u), 1e-12)
        v = np.cross(t, u)
        for a in np.linspace(0, 2 * np.pi, sides, endpoint=False):
            V.append(c + rad * (np.cos(a) * u + np.sin(a) * v))
    k = len(C)
    for i in range(k - 1):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            F += [(a, b, b + sides), (a, b + sides, a + sides)]
    return np.array(V), np.array(F)


def _shin_lacing(kind, R=2.2, rad=0.06, levels=4, half=math.radians(35), dz=1.0, k=24, recess=0.0):
    """Criss-cross lacing on a cylinder 'shin' (axis z, front -y as the figure's), and the cylinder's nearest(). kind:
      good     every run on the curve, the runs going one way lifted over the others at the crossings
      taut     every run a straight chord between its eyelets, cutting across the curve
      flipped  as good, but which lace lies on top changes from one crossing to the next
      sunk     as good, but sunk 0.7 radii into the shin
    recess dips the surface between the two eyelet rows (a tongue below the panels): a lace bridging the gap at the
    panels' height must still pass."""
    V, F = [], []
    for lv in range(levels):
        z0 = lv * dz
        for sgn in (1, -1):
            s = np.linspace(0, 1, k)
            th = sgn * (-half + 2 * half * s)
            top = sgn == 1 if kind != 'flipped' else (sgn == 1) == (lv % 2 == 0)
            rho = R + rad + (2.2 * rad * np.exp(-((s - 0.5) / 0.12) ** 2) if top else 0.0)
            if kind == 'sunk':
                rho = rho - 0.7 * rad
            C = np.stack([rho * np.sin(th), -rho * np.cos(th), z0 + dz * s], 1)
            if kind == 'taut':
                C = (1 - s)[:, None] * C[0] + s[:, None] * C[-1]
            v, f = _tube(C, rad)
            F.append(f + sum(len(x) for x in V))
            V.append(v)

    def nearest(p, across=None):
        q = np.array([p[0], p[1], 0.0])
        n = q / max(np.linalg.norm(q), 1e-12)
        r_ = R - (recess if abs(math.atan2(n[0], -n[1])) < 0.9 * half else 0.0)
        return np.array([r_ * n[0], r_ * n[1], p[2]]), n
    return np.concatenate(V), np.concatenate(F), nearest


def _cylinder(R=2.2, z0=-0.5, z1=4.5, n=48, m=20):
    a = np.linspace(-0.9, 0.9, n)
    z = np.linspace(z0, z1, m)
    A, Z = np.meshgrid(a, z, indexing='ij')
    X = np.stack([R * np.sin(A), -R * np.cos(A), Z], -1).reshape(-1, 3)
    T = []
    for i in range(n - 1):
        for j in range(m - 1):
            q = i * m + j
            T += [(q, q + m, q + m + 1), (q, q + m + 1, q + 1)]
    return X, np.array(T)


def selftest(out):
    """Four sleeves: a smooth one (the approved band is calibrated from it), one with a stretched patch, one whose
    seam edges were drawn 10% apart on the pattern, and one with a slit. The smooth one must pass, the others be
    flagged."""
    os.makedirs(out, exist_ok=True)
    X, T, R = _sleeve()
    S0, _ = analyse('smooth', X, T, R, 'pattern', None, lim=dict(DEFAULT_LIMITS), uv_island=uv_islands(T, R))
    lim = calibrate({'smooth': S0})
    res, items = {}, []
    for name, kw in (('smooth', {}), ('peak', {'bump': 0.5}), ('seam', {'short_seam': 0.1}), ('slit', {'slit': True})):
        X, T, R = _sleeve(**kw)
        X = X + np.array([7.0 * len(items), 0.0, 0.0])
        S, F = analyse(name, X, T, R, 'pattern', None, lim=lim, uv_island=uv_islands(T, R))
        res[name] = S
        items.append((name, S, F))
        print(name, S['verdict'], {k: S[k] for k in ('strain_p99', 'jag_p99', 'roughness', 'peaks',
                                                       'seam_mismatch_pct', 'fold_radius_p1')}, flush=True)
    heatmap(items, os.path.join(out, 'selftest-heat.png'), lim, 'self test: smooth, peak, short seam')
    assert res['smooth']['verdict'] == 'ok', res['smooth']['flags']
    assert res['peak']['peaks'] >= 1 and res['peak']['verdict'] != 'ok', res['peak']['flags']
    assert res['seam']['seam_mismatch_pct'] > 5 and res['seam']['verdict'] != 'ok', res['seam']['seams']
    assert res['slit']['holes'] == 1 and res['slit']['verdict'] != 'ok', res['slit']['openings']
    # the lacing on a shin: the good lacing must pass, the taut, flipped and sunk ones be flagged
    lres, laces = {}, []
    for n, kind in enumerate(('good', 'taut', 'flipped', 'sunk', 'bridge')):
        X, T, nearest = _shin_lacing('good' if kind == 'bridge' else kind, recess=0.15 if kind == 'bridge' else 0.0)
        S, runs = lace_check(kind, X, T, nearest)
        lres[kind] = S
        print('lace', kind, S['verdict'], {k: S.get(k) for k in LACE_LIMITS}, flush=True)
        if kind in ('good', 'taut'):
            laces.append((kind, S, runs))
    lace_heatmap([_cylinder()], laces[:1], os.path.join(out, 'selftest-lace-good.png'), LACE_LIMITS)
    lace_heatmap([_cylinder()], laces[1:], os.path.join(out, 'selftest-lace-taut.png'), LACE_LIMITS)
    assert lres['good']['verdict'] == 'ok' and lres['good']['crossings'] == 4, lres['good']
    assert lres['taut']['lace_arch_min'] < 0.2 and lres['taut']['lace_tension_min'] < 0.97, lres['taut']
    assert lres['taut']['verdict'] != 'ok', lres['taut']['flags']
    assert lres['flipped']['lace_flips'] == 2 and lres['flipped']['verdict'] != 'ok', lres['flipped']
    assert lres['sunk']['lace_sink_max'] > 0.5 and lres['sunk']['verdict'] != 'ok', lres['sunk']['flags']
    assert lres['bridge']['verdict'] == 'ok', lres['bridge']        # (bridged over a recessed tongue: no fault)
    res['laces'] = lres
    json.dump({'limits': lim, 'results': res}, open(os.path.join(out, 'selftest.json'), 'w'), indent=1)
    print('self test passed', flush=True)


def main():
    """python3 rq_flow.py --selftest DIR
    python3 rq_flow.py --calibrate OUT.json FLOW.json... : the approved band from approved garments' flow.json
    files (every garment in them, and every pose of a flow-poses.json), written as flow_baseline.json."""
    a = sys.argv[1:]
    if '--selftest' in a:
        selftest(a[a.index('--selftest') + 1])
    elif '--calibrate' in a:
        i = a.index('--calibrate')
        dst, srcs = a[i + 1], a[i + 2:]
        by_pose, per, lay, lac = {}, {}, {}, {}
        for p in srcs:
            rep = json.load(open(p))
            reps = rep['poses'].items() if 'poses' in rep else [(rep.get('pose', 'rest'), rep)]
            for pose, r in reps:
                for x in r.get('layers', []):
                    lay.setdefault(pose, []).append(x)
                for S in r.get('laces', {}).values():
                    lac.setdefault(pose, []).append(S)
                for name, S in r['garments'].items():
                    by_pose.setdefault(pose, {})[name + ' / ' + os.path.basename(os.path.dirname(os.path.abspath(p)))] = S
                    per.setdefault(pose, {})[name] = {k: S.get(k) for k in DEFAULT_LIMITS}
        outl = {}
        poses = {}
        for pose, g in sorted(by_pose.items()):
            o = {}
            poses[pose] = calibrate(g, o)
            if o:
                outl[pose] = o
        for pose, xs in lay.items():          # (the layers' band: the approved layers' worst, with the margin)
            if pose in poses:
                poses[pose]['poke_pct'] = round(max(LAYER_LIMITS['poke_pct'], MARGIN * max(x['poke_pct'] for x in xs)), 3)
                poses[pose]['poke_max'] = round(max(LAYER_LIMITS['poke_max'], MARGIN * max(x['poke_max'] for x in xs)), 3)
        for pose, xs in lac.items():          # (the lacing's band: the approved laces' worst, with the margin)
            band = poses.setdefault(pose, dict(poses.get('rest', {})))
            for k, v in LACE_LIMITS.items():
                got = [S[k] for S in xs if S.get(k) is not None]
                if not got:
                    continue
                if k in LACE_LOWER:
                    band[k] = round(min(v, min(got) - (1 - min(got)) * (MARGIN - 1) - 0.01), 3)
                elif k in ('lace_flips', 'lace_hits'):
                    band[k] = max(v, max(got))
                else:
                    band[k] = round(max(v, MARGIN * max(got)), 3)
        # (every approved garment judged again on the band it made: none may be flagged but on its outlier metrics)
        for pose, g in by_pose.items():
            for name, S in g.items():
                bad = {k for k, v in outl.get(pose, {}).items() if any(x[0] == name for x in v)}
                fl, verdict = judge(S, poses[pose])
                left = [f for f in fl if f['metric'] not in bad]
                assert not left, (pose, name, left)
        json.dump({'_doc': 'The approved band for rq_flow.py, per pose: each limit is the worst value among the '
                           'garments John approved (listed under approved) in that pose, with a margin '
                           '(rq_flow.calibrate). limits is the rest pose. Re-run rq_flow.py --calibrate on new '
                           'approved flow-poses.json files to widen or tighten it.',
                   'from': [os.path.basename(os.path.dirname(os.path.abspath(p))) + '/' + os.path.basename(p)
                            for p in srcs],
                   'limits': poses.get('rest') or next(iter(poses.values())), 'poses': poses,
                   'outliers': outl, 'approved': per},
                  open(dst, 'w'), indent=1)
        print('wrote', dst, flush=True)


if __name__ == '__main__':
    main()
