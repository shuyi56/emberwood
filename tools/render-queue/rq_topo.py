"""The face topology check (John, 2026-11-20): before a render of a changed face or head, read the head mesh's
topology and say whether it is clean. Report-only, like the flow check (rq_flow.py) and the layer gap check: it never
stops a render; it writes what it found and a sheet to put beside the render.

    topo.json        per head: the head's measures and the zones derived from them, the counts, the flags against
                     the approved band, the 'clean' standard and what fails it, and the worst spots with their places
    topo-sheet.png   clay with curvature shading: front, 3/4 and profile, then close-ups of the eyes, the mouth and
                     the nose with the wireframe over them (poles marked, the edge loops found around each feature
                     drawn), then the contour roughness as a heatmap. The worst spots are ringed and numbered as in
                     topo.json.

The head is measured first and every zone comes from that measure (project rule: the face sets the ratios), in the
Head frame (chin at the origin, the face toward -y, z up):

    H            chin to crown (the chin: the lowest point of the front near the middle, as female.neck_joints)
    eyes         the eyeballs' centres (the small meshes in the eye band), else the orbits' deepest point
    IPD          between the eyes' centres; eye width = IPD / 2 (an eye's width apart, the canon)
    nose tip     the most forward point of the midline between the eyes and the mouth
    stomion      the recess between the two lips on the midline profile (else the canon 0.232 H, female.MOUTH_Z)
    mouth width  the length of the lip line's crease (else 0.8 IPD)
    zones        ellipses on the front: each eye's orbicularis (0.85 x 0.62 eye widths about its centre), the mouth's
                 orbicularis (0.75 x 0.48 mouth widths about the stomion), the nose (0.3 x 0.42 IPD about its tip).
                 These are where the face deforms, so where loops must run round and poles must not sit.

What is measured, per head:

    counts       verts, faces, the share of quads, triangles and ngons (a glTF is triangles only: its triangles are
                 joined back into quads first, as Blender's Tris to Quads, and that is said)
    poles        interior vertices with 3, 5 or 6+ edges (a quad mesh's irregular points), overall and in each zone
    integrity    non-manifold edges (3+ faces) and vertices (bow ties), holes (open borders other than the neck),
                 loose vertices and wire edges, stray islands (floating pieces under 0.5% of the head), duplicate
                 vertices (within 1e-5 H of another, as Merge by Distance would remove), degenerate faces, flipped
                 normals (neighbouring faces wound against each other; the whole mesh inside out)
    shape        sharp: the angle between neighbouring faces (Mesh Analysis 'Sharp'), p99 and max, on the face;
                 distortion: how far a quad or ngon is from flat (Mesh Analysis 'Distortion'), p99 and max;
                 smooth_p99 / roughness: the contour's smoothness, how far each point's mean curvature (cotangent
                 Laplacian, times H) stands from its neighbours' (near 0 smooth; roughness near 1 is noise);
                 sym_p999: how far the mirrored head lies from itself (surface distance / H, at the worst 0.1%), sym_match_pct: the
                 share of vertices with a mirror twin (topological symmetry)
    loops        per feature (each eye, the mouth): edge loops walked as Blender's Alt-click does (straight through
                 every 4-edge vertex) from seeds at 30-100% of the zone; loops_around counts the distinct closed loops
                 that run round the feature and stay near it (concentric loops), broken counts walks that stop at a
                 pole, a triangle or an ngon inside the zone; alignment is the share of the zone's edge length
                 running round or straight out from the feature (within 20 degrees): about 0.45 for a mesh laid
                 with no regard to the feature, near 1 for concentric loops with radial spokes

Two verdicts:

    verdict      against the approved heads (topo_baseline.json): a head John approved before (by its name, or
                 --label) on its own band, any other on its kind's ('source', a quad mesh as built; 'game', the
                 decimated triangles the world loads): 'watch' past the band, 'likely rejection' well past it.
                 This is the regression check: a face change that makes the mesh worse than what John passed.
    clean        the absolute standard of clean face topology: no non-manifold, holes, strays, duplicates,
                 degenerate or flipped faces; no ngons on the face; no poles inside the eye and mouth zones; at
                 least 2 concentric loops round each eye and the mouth with alignment 0.7 or better; symmetric to
                 0.002 H. 'game' heads are held only to the integrity part (decimation leaves no loops to keep).

Run it:

    in a views job       rq_views.py runs it after the build, before the first render, whenever the job's commit
                         changes a face or head file against main (or --topo; --topo-only stops after it;
                         RQ_TOPO=0 turns it off)
    on any glb or blend  python3 rq_topo.py --glb HEAD.glb [--object NAME] --out DIR [--label TEXT]
                         (bpy): after export-head.js or merge-head.mjs, on the head they wrote
    calibrate            python3 rq_topo.py --calibrate topo_baseline.json DIR/topo.json ...
    self test            python3 rq_topo.py --selftest DIR (no Blender: a grid head and a loop head)

The core (measure, analyse, judge, calibrate) is numpy and scipy; collect() reads a Blender object and sheet()
renders through Cycles when bpy is there (a numpy rasteriser otherwise)."""
import json
import math
import os
import re
import sys
import time
import warnings

import numpy as np

VERSION = 1
HERE = os.path.dirname(os.path.abspath(__file__))

# the head and face's files: a change to one of these (against main) runs the check in a views job
FACE_PATHS = re.compile(r'(^|/)(face|head|ear|eye|lid|lip|mouth|nose|brow|jaw|cheek|creator)[\w-]*\.(py|json|js|mjs)$'
                        r'|(^|/)(creator-head|face-creator|ranger-face|ranger-eyes|ranger-lips)/', re.I)
# and in female.py (or a figure's own file) a change inside one of these functions or settings
FACE_DEFS = re.compile(r'head|face|eye|lid|lash|brow|mouth|lip|nose|nostril|ear|jaw|chin|cheek|skull|iris|teeth', re.I)

ZONES = ('eye R', 'eye L', 'mouth', 'nose')
LOOP_FEATURES = ('eye R', 'eye L', 'mouth')

# the band's metrics; True: lower is worse
METRICS = {'quad_pct': True, 'ngons_face': False, 'poles_zone': False, 'nonmanifold_edges': False,
           'nonmanifold_verts': False, 'holes': False, 'loose': False, 'stray_islands': False,
           'duplicate_verts': False, 'degenerate_faces': False, 'flipped_edges': False,
           'sharp_p99': False, 'sharp_max': False, 'distortion_p99': False, 'smooth_p99': False,
           'roughness': False, 'sym_p999': False, 'sym_match_pct': True,
           'loops_min': True, 'broken_max': False, 'alignment_min': True}
COUNTS = ('ngons_face', 'poles_zone', 'nonmanifold_edges', 'nonmanifold_verts', 'holes', 'loose', 'stray_islands',
          'duplicate_verts', 'degenerate_faces', 'flipped_edges', 'broken_max')
STRUCTURE = ('quad_pct', 'ngons_face', 'poles_zone', 'loops_min', 'broken_max', 'alignment_min')
DEFAULT_LIMITS = {'quad_pct': 90.0, 'ngons_face': 0, 'poles_zone': 0, 'nonmanifold_edges': 0, 'nonmanifold_verts': 0,
                  'holes': 0, 'loose': 0, 'stray_islands': 0, 'duplicate_verts': 0, 'degenerate_faces': 0,
                  'flipped_edges': 0, 'sharp_p99': 25.0, 'sharp_max': 80.0, 'distortion_p99': 8.0,
                  'smooth_p99': 6.0, 'roughness': 0.6, 'sym_p999': 0.003, 'sym_match_pct': 95.0,
                  'loops_min': 2, 'broken_max': 0, 'alignment_min': 0.7}
# the clean standard (absolute: what clean face topology is, whatever the approved heads do)
CLEAN = {'nonmanifold_edges': 0, 'nonmanifold_verts': 0, 'holes': 0, 'loose': 0, 'stray_islands': 0,
         'duplicate_verts': 0, 'degenerate_faces': 0, 'flipped_edges': 0, 'inside_out': 0,
         'ngons_face': 0, 'poles_zone': 0, 'loops_min': 2, 'alignment_min': 0.7, 'sym_p999': 0.002}
CLEAN_GAME = ('nonmanifold_edges', 'nonmanifold_verts', 'holes', 'loose', 'stray_islands', 'duplicate_verts',
              'degenerate_faces', 'flipped_edges', 'inside_out', 'sym_p999')
MARGIN = 1.25
DUP_TOL = 1e-5            # duplicate vertices: within this of another, in head heights
ALIGN_DEG = 20.0


# ------------------------------------------------------------------------------------------------ the measure

def front_half(Y, H):
    """The points in the front part of the head (y toward -y): the front 60% of its depth (all of it when it has
    almost none, a face plate)."""
    lo, hi = float(Y.min()), float(Y.max())
    return Y < lo + 0.6 * (hi - lo) if hi - lo > 0.3 * H else np.ones(len(Y), bool)


def depth_map(X, N, H, step=0.005):
    """The front of the head as a depth map: D[i, j] the most forward y (the least, the face is toward -y) of the
    front-facing points over x = x0 + i step H, z = j step H. Empty cells filled from their neighbours."""
    f = (N[:, 1] < -0.15) & front_half(X[:, 1], H)
    P = X[f] / H
    x0, z0 = -0.6, -0.2
    nx, nz = int(1.2 / step) + 1, int(1.4 / step) + 1
    i = np.clip(((P[:, 0] - x0) / step).astype(int), 0, nx - 1)
    j = np.clip(((P[:, 2] - z0) / step).astype(int), 0, nz - 1)
    D = np.full((nx, nz), np.inf)
    np.minimum.at(D, (i, j), P[:, 1])
    D[~np.isfinite(D)] = np.nan
    for _ in range(3):         # (fill single empty cells from their neighbours: a coarse mesh leaves gaps)
        m = np.isnan(D)
        if not m.any():
            break
        pad = np.pad(D, 1, constant_values=np.nan)
        nb = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fill = np.nanmean(nb, 0)
        D[m] = fill[m]
    return D, x0, z0, step


def _dm(D, x0, z0, step, x, z):
    i = int(round((x - x0) / step))
    j = int(round((z - z0) / step))
    if 0 <= i < D.shape[0] and 0 <= j < D.shape[1]:
        return D[i, j]
    return np.nan


def measure(X, N, eyes=None):
    """The head's measures (see the module's doc), in the Head frame. X [n, 3] points, N [n, 3] vertex normals,
    eyes: [(x, y, z), (x, y, z)] the eyeballs' centres when known."""
    top = float(X[:, 2].max())
    depth = float(X[:, 1].max() - X[:, 1].min())
    front = X[(np.abs(X[:, 0]) < 0.08 * depth) & (X[:, 1] < X[:, 1].min() + 0.3 * depth)]
    chin = float(front[:, 2].min()) if len(front) else float(X[:, 2].min())
    H = top - chin
    Xn = (X - [0, 0, chin]) / H          # (head heights, chin at 0)
    D, x0, z0, st = depth_map(X - [0, 0, chin], N, H)
    src = {}
    if eyes is not None and len(eyes) == 2:
        E = sorted([((e[0]) / H, (e[1]) / H, (e[2] - chin) / H) for e in eyes])
        src['eyes'] = 'eyeballs'
    else:          # the orbits: the deepest front point per side in the eye band
        E = []
        for s in (-1, 1):
            best = None
            for x in np.arange(0.06, 0.26, st):
                for z in np.arange(0.44, 0.64, st):
                    y = _dm(D, x0, z0, st, s * x, z)
                    if np.isfinite(y) and (best is None or y > best[1]):
                        best = (s * x, y, z)
            E.append(best or (s * 0.15, -0.4, 0.53))
        src['eyes'] = 'orbit recess'
    ipd = abs(E[1][0] - E[0][0])
    ez = 0.5 * (E[0][2] + E[1][2])
    # the midline profile: the most forward y at each height
    zs = np.arange(0.02, ez, st)
    prof = np.array([np.nanmean([_dm(D, x0, z0, st, dx, z) for dx in (-st, 0, st)]) for z in zs])
    ok = np.isfinite(prof)
    zs, prof = zs[ok], prof[ok]
    band = (zs > 0.25) & (zs < ez - 0.05)
    tip_z = float(zs[band][np.argmin(prof[band])]) if band.any() else 0.36
    tip_y = float(prof[band].min()) if band.any() else float(np.nanmin(D))
    # the lips: two protrusions under the nose, the stomion the recess between them
    k = 3
    sm = np.convolve(prof, np.ones(k) / k, mode='same')
    lo, hi = 0.1, tip_z - 0.06
    idx = [i for i in range(1, len(zs) - 1) if lo < zs[i] < hi]
    mins = [i for i in idx if sm[i] <= sm[i - 1] and sm[i] <= sm[i + 1]]
    stom, src['stomion'] = 0.232, 'canon (female.MOUTH_Z)'
    if len(mins) >= 2:
        mins = sorted(mins, key=lambda i: sm[i])[:2]
        a, b = sorted(mins)
        if b - a >= 2:
            r = a + int(np.argmax(sm[a:b + 1]))
            if 0.15 < zs[r] < 0.33:
                stom, src['stomion'] = float(zs[r]), 'profile'
    # the mouth's width: where the lip line's crease (deeper than the lips over and under it) fades out
    def crease(x):
        y0 = _dm(D, x0, z0, st, x, stom)
        ya, yb = _dm(D, x0, z0, st, x, stom + 0.025), _dm(D, x0, z0, st, x, stom - 0.025)
        return y0 - 0.5 * (ya + yb) if np.isfinite(y0 + ya + yb) else np.nan
    c0 = crease(0.0)
    mw, src['mouth_width'] = 0.8 * ipd, 'canon (0.8 IPD)'
    if np.isfinite(c0) and c0 > 0.002:
        xs = np.arange(0, 0.6 * ipd, st)
        cs = np.array([np.nanmean([crease(x), crease(-x)]) for x in xs])
        out = np.nonzero(~(cs > 0.3 * c0))[0]
        if len(out) and 0.5 * ipd < 2 * xs[out[0]] < 1.2 * ipd:
            mw, src['mouth_width'] = float(2 * xs[out[0]]), 'lip line crease'
    we = ipd / 2
    zones = {'eye R': (E[0][0], E[0][2], 0.85 * we, 0.62 * we), 'eye L': (E[1][0], E[1][2], 0.85 * we, 0.62 * we),
             'mouth': (0.0, stom, 0.75 * mw, 0.48 * mw), 'nose': (0.0, tip_z, 0.3 * ipd, 0.42 * ipd)}
    return {'H': round(H, 4), 'chin_z': round(chin, 4), 'crown_z': round(top, 4),
            'eyes': [[round(v, 4) for v in e] for e in E], 'ipd': round(ipd, 4), 'eye_z': round(ez, 4),
            'nose_tip': [0.0, round(tip_y, 4), round(tip_z, 4)], 'stomion_z': round(stom, 4),
            'mouth_width': round(mw, 4), 'zones': {k: [round(float(v), 4) for v in z] for k, z in zones.items()},
            'ratios': {'ipd/H': round(ipd, 3), 'eye_z/H': round(ez, 3), 'nose_tip_z/H': round(tip_z, 3),
                       'stomion_z/H': round(stom, 3), 'mouth/ipd': round(mw / max(ipd, 1e-6), 3)},
            'from': src, '_Xn': Xn}


def zone_d(Xn, Nn, M):
    """Each point's elliptic distance to each zone (1.0 on the zone's edge), inf for points not on the front."""
    front = (Nn[:, 1] < -0.1) & front_half(Xn[:, 1], 1.0)
    out = {}
    for k, (cx, cz, a, b) in M['zones'].items():
        d = np.hypot((Xn[:, 0] - cx) / a, (Xn[:, 2] - cz) / b)
        d[~front] = np.inf
        out[k] = d
    return out


def place_fn(M):
    """place(p) for a point in head heights: 'left eye, upper lid'. The face is toward -y; its right is -x."""
    z = M['zones']
    ez, st = M['eye_z'], M['stomion_z']

    def place(p):
        x, y, zz = float(p[0]), float(p[1]), float(p[2])
        side = 'right' if x < 0 else 'left'
        for k in ZONES:
            cx, cz, a, b = z[k]
            d = math.hypot((x - cx) / a, (zz - cz) / b)
            if d < 1.0 and y < 0.05:
                if k.startswith('eye'):
                    part = 'upper lid' if zz > cz + 0.25 * b else 'lower lid' if zz < cz - 0.25 * b else \
                        ('inner corner' if abs(x) < abs(cx) else 'outer corner')
                    return f'{side} eye, {part}'
                if k == 'mouth':
                    part = 'corner' if abs(x - cx) > 0.55 * a else 'upper lip' if zz > cz else 'lower lip'
                    return f'mouth, {part}' + (f', {side}' if part == 'corner' else '')
                return 'nose'
        if zz > 0.97:
            return 'crown'
        if zz < -0.02:
            return 'neck' + (', front' if y < 0 else ', back')
        if y > 0.15:
            return 'back of the head' + (', low' if zz < 0.45 else '')
        if abs(x) > 0.3:
            return f'{side} side' + (' (ear)' if 0.3 < zz < 0.6 else '')
        if zz > ez + 0.08:
            return 'forehead' if zz > ez + 0.18 else f'{side} brow'
        if zz > st + 0.05:
            return f'{side} cheek' if abs(x) > 0.12 else 'nose'
        return 'chin' if abs(x) < 0.1 else f'{side} jaw'
    return place


# ------------------------------------------------------------------------------------------------ the topology

class Topo:
    """The mesh's connectivity from its faces (CSR: fo offsets, fv vertex indices), in numpy."""

    def __init__(self, n, fo, fv):
        self.n, self.fo, self.fv = n, np.asarray(fo, np.int64), np.asarray(fv, np.int64)
        self.nf = len(self.fo) - 1
        self.size = np.diff(self.fo)
        self.face_of = np.repeat(np.arange(self.nf), self.size)
        k = np.arange(len(self.fv))
        start = self.fo[self.face_of]
        self.nxt = start + (k - start + 1) % self.size[self.face_of]
        self.prv = start + (k - start - 1) % self.size[self.face_of]
        a, b = self.fv, self.fv[self.nxt]
        key = np.minimum(a, b) * n + np.maximum(a, b)
        ukey, self.he_e, cnt = np.unique(key, return_inverse=True, return_counts=True)
        self.E = np.stack([ukey // n, ukey % n], 1)
        self.ecount = cnt
        m = len(ukey)
        order = np.argsort(self.he_e, kind='stable')
        first = np.zeros(m + 1, np.int64)
        first[1:] = np.cumsum(cnt)
        self.he_order, self.he_first = order, first
        h0 = order[first[:-1]]
        h1 = np.where(cnt >= 2, order[np.minimum(first[:-1] + 1, len(order) - 1)], -1)
        self.ef = np.stack([self.face_of[h0], np.where(h1 >= 0, self.face_of[np.maximum(h1, 0)], -1)], 1)
        self.eh = np.stack([h0, h1], 1)
        # (two faces wound the same way along their shared edge: one of them is flipped)
        two = cnt == 2
        same = np.zeros(m, bool)
        same[two] = self.fv[h0[two]] == self.fv[h1[two]]
        self.flipped_e = same
        # vertex -> edges
        ve = np.concatenate([self.E[:, 0], self.E[:, 1]])
        eid = np.concatenate([np.arange(m), np.arange(m)])
        o = np.argsort(ve, kind='stable')
        self.ve_e = eid[o]
        self.ve_o = np.zeros(n + 1, np.int64)
        np.add.at(self.ve_o, ve + 1, 1)
        self.ve_o = np.cumsum(self.ve_o)
        self.valence = np.diff(self.ve_o)
        self.boundary_v = np.zeros(n, bool)
        be = self.E[cnt == 1]
        self.boundary_v[be.ravel()] = True
        self.nm_v = np.zeros(n, bool)
        self.nm_v[self.E[cnt > 2].ravel()] = True
        self.vfaces = np.bincount(self.fv, minlength=n)
        mx = np.zeros(n, np.int64)
        np.maximum.at(mx, self.fv, self.size[self.face_of])
        mn = np.full(n, 99, np.int64)
        np.minimum.at(mn, self.fv, self.size[self.face_of])
        self.all_quads = (mx == 4) & (mn == 4)

    def bowties(self):
        """Vertices whose faces form more than one fan (non-manifold vertices), from the corners' connectivity."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        h0, h1 = self.eh[:, 0], self.eh[:, 1]
        ok = (h1 >= 0) & (self.ecount == 2)
        h0, h1 = h0[ok], h1[ok]
        same = self.fv[h0] == self.fv[h1]
        # corner of the edge's first vertex (fv[h0]) and second in each face
        a0, b0 = h0, self.nxt[h0]
        a1 = np.where(same, h1, self.nxt[h1])
        b1 = np.where(same, self.nxt[h1], h1)
        r = np.concatenate([a0, b0])
        c = np.concatenate([a1, b1])
        K = len(self.fv)
        g = coo_matrix((np.ones(len(r)), (r, c)), shape=(K, K))
        _, lab = connected_components(g, directed=False)
        pair = np.unique(np.stack([self.fv, lab], 1), axis=0)
        fans = np.bincount(pair[:, 0], minlength=self.n)
        return (fans > 1) & ~self.nm_v

    def corner(self, f, v):
        s = self.fo[f]
        for k in range(s, self.fo[f + 1]):
            if self.fv[k] == v:
                return k
        return -1

    def other_edge(self, f, e, v):
        """In face f, the edge at vertex v that is not e."""
        k = self.corner(f, v)
        if k < 0:
            return -1
        a, b = self.he_e[k], self.he_e[self.prv[k]]
        return b if a == e else a

    def walk(self, e, v, limit=6000):
        """An edge loop from edge e toward its vertex v, straight through every regular vertex (4 edges, 4 quads).
        Returns (vertices, edges, closed, why it stopped)."""
        e0, vs, es = e, [int(self.E[e, 0] if self.E[e, 1] == v else self.E[e, 1])], [int(e)]
        for _ in range(limit):
            vs.append(int(v))
            if self.boundary_v[v]:
                return vs, es, False, 'border'
            if self.valence[v] != 4 or not self.all_quads[v]:
                return vs, es, False, ('pole' if self.all_quads[v] else 'triangle or ngon')
            f0, f1 = self.ef[e]
            x0, x1 = self.other_edge(f0, e, v), self.other_edge(f1, e, v) if f1 >= 0 else -1
            nxt = [int(q) for q in self.ve_e[self.ve_o[v]:self.ve_o[v + 1]] if q not in (e, x0, x1)]
            if len(nxt) != 1:
                return vs, es, False, 'pole'
            e = nxt[0]
            if e == e0:
                return vs, es, True, 'closed'
            es.append(e)
            a, b = self.E[e]
            v = b if a == v else a
        return vs, es, False, 'too long'


def components(n, a, b):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    return connected_components(g, directed=False)


def face_normals(X, T):
    """Newell normals and areas of every face (CSR)."""
    P = X[T.fv]
    Q = X[T.fv[T.nxt]]
    c = np.stack([(P[:, 1] - Q[:, 1]) * (P[:, 2] + Q[:, 2]), (P[:, 2] - Q[:, 2]) * (P[:, 0] + Q[:, 0]),
                  (P[:, 0] - Q[:, 0]) * (P[:, 1] + Q[:, 1])], 1)
    n = np.zeros((T.nf, 3))
    np.add.at(n, T.face_of, c)
    a = np.linalg.norm(n, axis=1)
    return n / np.maximum(a, 1e-30)[:, None], 0.5 * a


def triangles(T):
    """A fan of triangles for every face, and each triangle's face."""
    k = np.arange(len(T.fv))
    start = T.fo[T.face_of]
    keep = (k - start >= 1) & (k - start <= T.size[T.face_of] - 2)
    kk = k[keep]
    tri = np.stack([T.fv[start[keep]], T.fv[kk], T.fv[T.nxt[kk]]], 1)
    return tri, T.face_of[kk]


def curvature(X, tri, n):
    """Mean curvature per vertex (cotangent Laplacian / 2 area), signed (+ convex), and the vertex normals."""
    from scipy.sparse import coo_matrix
    A, B, C = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    fn = np.cross(B - A, C - A)
    ar = 0.5 * np.linalg.norm(fn, axis=1)
    good = ar > 1e-14
    tri, A, B, C, fn, ar = tri[good], A[good], B[good], C[good], fn[good], ar[good]

    def cot(u, v):
        return np.clip(np.einsum('ij,ij->i', u, v) / np.maximum(np.linalg.norm(np.cross(u, v), axis=1), 1e-14), -50, 50)
    ca, cb, cc = cot(B - A, C - A), cot(C - B, A - B), cot(A - C, B - C)
    i, j, k = tri[:, 0], tri[:, 1], tri[:, 2]
    r = np.concatenate([j, k, k, i, i, j])
    c = np.concatenate([k, j, i, k, j, i])
    w = 0.5 * np.concatenate([ca, ca, cb, cb, cc, cc])
    L = coo_matrix((w, (r, c)), shape=(n, n)).tocsr()
    deg = np.asarray(L.sum(1)).ravel()
    LX = L @ X - deg[:, None] * X
    area = np.zeros(n)
    np.add.at(area, tri.ravel(), np.repeat(ar / 3, 3))
    vn = np.zeros((n, 3))
    np.add.at(vn, tri.ravel(), np.repeat(fn, 3, axis=0))
    vn /= np.maximum(np.linalg.norm(vn, axis=1), 1e-30)[:, None]
    k = 0.5 * np.linalg.norm(LX, axis=1) / np.maximum(area, 1e-14)
    k *= np.sign(-np.einsum('ij,ij->i', LX, vn))
    k[area <= 1e-14] = 0
    return k, vn


def _pct(v, q):
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if len(v) else 0.0


def analyse(name, X, fo, fv, eyes=None, loose_edges=0, joined=False, kind=None, nearest=None):
    """The topology check on one head. X [n, 3] in the Head frame (face toward -y), faces as CSR (fo, fv).
    nearest: a function (points [m, 3]) -> distances to the surface, for the geometric symmetry (else vertices)."""
    t0 = time.time()
    n = len(X)
    T = Topo(n, fo, fv)
    tri, tri_f = triangles(T)
    kap, vn = curvature(X, tri, n)
    M = measure(X, vn, eyes)
    H = M['H']
    Xn = M.pop('_Xn')
    place = place_fn(M)
    zd = zone_d(Xn, vn, M)
    face_zone = (Xn[:, 2] > -0.02) & (Xn[:, 2] < M['eye_z'] + 0.25) & (vn[:, 1] < -0.1) & \
        front_half(Xn[:, 1], 1.0)
    in_zone = np.zeros(n, bool)
    for k in ('eye R', 'eye L', 'mouth'):
        in_zone |= zd[k] < 1.0
    used = np.zeros(n, bool)
    used[T.fv] = True
    S = {'name': name, 'version': VERSION, 'measures': M}
    sizes = T.size
    nq, nt, ng = int((sizes == 4).sum()), int((sizes == 3).sum()), int((sizes > 4).sum())
    quad_pct = 100.0 * nq / max(T.nf, 1)
    if kind is None:
        kind = 'source' if quad_pct >= 50 else 'game'
    S['kind'] = kind
    fnrm, farea = face_normals(X, T)
    fc = np.zeros((T.nf, 3))
    np.add.at(fc, T.face_of, X[T.fv])
    fc /= sizes[:, None]
    fcn = (fc - [0, 0, M['chin_z']]) / H
    face_f = np.zeros(T.nf, bool)
    np.logical_or.at(face_f, T.face_of, face_zone[T.fv])
    S['counts'] = {'verts': n, 'faces': int(T.nf), 'quads': nq, 'tris': nt, 'ngons': ng,
                   'quad_pct': round(quad_pct, 2), 'tri_pct': round(100.0 * nt / max(T.nf, 1), 2),
                   'joined_from_triangles': bool(joined)}
    # poles: interior vertices of the quad mesh with other than 4 edges
    interior = used & ~T.boundary_v & ~T.nm_v
    pole = interior & (T.valence != 4) & (T.vfaces > 0)
    if kind == 'game':          # (on triangles, a regular vertex has 6: count the irregular ones)
        pole = interior & (T.valence != 6)
    S['poles'] = {'3': int((pole & (T.valence == 3)).sum()), '5': int((pole & (T.valence == 5)).sum()),
                  '6+': int((pole & (T.valence >= 6)).sum()) if kind == 'source' else int((pole & (T.valence >= 7)).sum()),
                  'zones': {k: int((pole & (zd[k] < 1.0)).sum()) for k in ZONES}}
    poles_zone = int((pole & in_zone).sum())
    # integrity
    nm_e = int((T.ecount > 2).sum())
    bow = T.bowties()
    bnd = np.nonzero(T.ecount == 1)[0]
    holes = []
    if len(bnd):
        ncomp, lab = components(n, T.E[bnd, 0], T.E[bnd, 1])
        el = np.linalg.norm(X[T.E[bnd, 0]] - X[T.E[bnd, 1]], axis=1)
        loops = {}
        for e, L in zip(range(len(bnd)), el):
            loops.setdefault(int(lab[T.E[bnd[e], 0]]), [0.0, []])
            loops[int(lab[T.E[bnd[e], 0]])][0] += L
            loops[int(lab[T.E[bnd[e], 0]])][1].append(bnd[e])
        rows = []
        for c, (L, es) in loops.items():
            vv = np.unique(T.E[es].ravel())
            rows.append({'edges': len(es), 'length': round(L / H, 4), 'z': round(float(Xn[vv, 2].mean()), 3),
                         'zmin': float(Xn[vv, 2].min()), 'at': Xn[vv].mean(0), 'verts': vv})
        neck = min(rows, key=lambda r: r['zmin'])
        for r in rows:
            if r is neck and r['zmin'] < 0.05:
                continue
            holes.append(r)
    loose = int((~used).sum()) + int(loose_edges)
    ncomp, lab = components(n, T.E[:, 0], T.E[:, 1])
    lab = np.where(used, lab, -1)
    sz = np.bincount(lab[lab >= 0], minlength=ncomp)
    main = int(np.argmax(sz))
    stray = [c for c in range(ncomp) if c != main and 0 < sz[c] < 0.005 * n]
    from scipy.spatial import cKDTree
    tree = cKDTree(X)
    pairs = tree.query_pairs(DUP_TOL * H, output_type='ndarray')
    dup_v = 0
    dup_at = []
    if len(pairs):
        nc, dl = components(n, pairs[:, 0], pairs[:, 1])
        inv = np.unique(pairs.ravel())
        dup_v = int(len(inv) - len(np.unique(dl[inv])))
        for c in np.unique(dl[inv])[:2000]:
            m = inv[dl[inv] == c]
            dup_at.append((len(m), Xn[m[0]]))
    rep_idx = np.zeros(T.nf, bool)        # (a face that names a vertex twice)
    rep = T.fv == T.fv[T.nxt]
    np.logical_or.at(rep_idx, T.face_of, rep)
    degen = (farea < 1e-9 * H * H) | rep_idx
    flipped = T.flipped_e
    # the whole mesh inside out: the normals pointing in, on the skull (above the eyes, where it is convex)
    cz = M['chin_z'] + 0.5 * H
    cen = np.array([0.0, float(np.median(X[:, 1])), cz])
    skull = fcn[:, 2] > M['eye_z'] + 0.12
    outward = np.einsum('ij,ij->i', fnrm[skull], fc[skull] - cen)
    inside_out = int(np.sum(farea[skull] * np.sign(outward)) < 0)
    S['integrity'] = {'nonmanifold_edges': nm_e, 'nonmanifold_verts': int(bow.sum()), 'holes': len(holes),
                      'loose': loose, 'stray_islands': len(stray), 'duplicate_verts': dup_v,
                      'degenerate_faces': int(degen.sum()), 'flipped_edges': int(flipped.sum()),
                      'inside_out': inside_out}
    # shape: sharp edges, distortion, the contour's smoothness, symmetry
    two = (T.ecount == 2) & (T.ef[:, 1] >= 0)
    f0, f1 = T.ef[two, 0], T.ef[two, 1]
    good = ~degen[f0] & ~degen[f1]
    dots = np.einsum('ij,ij->i', fnrm[f0], fnrm[f1])
    dots = np.where(flipped[two], -dots, dots)
    ang = np.degrees(np.arccos(np.clip(dots, -1, 1)))
    e_face = face_f[f0] | face_f[f1]
    sharp = ang[good & e_face] if (good & e_face).any() else ang[good]
    big = sizes > 3
    k = np.nonzero(big[T.face_of])[0]
    P, Pn, Pp = X[T.fv[k]], X[T.fv[T.nxt[k]]], X[T.fv[T.prv[k]]]
    cn = np.cross(Pn - P, Pp - P)
    cl = np.linalg.norm(cn, axis=1)
    cosd = np.einsum('ij,ij->i', cn, fnrm[T.face_of[k]]) / np.maximum(cl, 1e-30)
    cd = np.degrees(np.arccos(np.clip(cosd, -1, 1)))
    cd[cl < 1e-14] = 0
    dist = np.zeros(T.nf)
    np.maximum.at(dist, T.face_of[k], cd)
    dmask = big & ~degen & face_f
    distort = dist[dmask] if dmask.any() else dist[big & ~degen]
    kH = kap * H
    nb = np.zeros(n)
    cnt = np.zeros(n)
    np.add.at(nb, T.E[:, 0], kH[T.E[:, 1]])
    np.add.at(nb, T.E[:, 1], kH[T.E[:, 0]])
    np.add.at(cnt, T.E[:, 0], 1)
    np.add.at(cnt, T.E[:, 1], 1)
    exc = kH - nb / np.maximum(cnt, 1)
    exc[~used | T.boundary_v] = 0
    fz = face_zone & used & ~T.boundary_v
    ex = np.abs(exc[fz]) if fz.any() else np.abs(exc[used])
    kz = kH[fz] if fz.any() else kH[used]
    rough = float(np.sqrt(np.mean(ex ** 2)) / max(np.std(kz), 1e-9))
    Xm = X * [-1, 1, 1]
    dv, _ = tree.query(Xm)
    ds = nearest(Xm) if nearest is not None else dv
    S['shape'] = {'sharp_p99': round(_pct(sharp, 99), 2), 'sharp_max': round(float(sharp.max()) if len(sharp) else 0, 1),
                  'distortion_p99': round(_pct(distort, 99), 2) if len(distort) else 0.0,
                  'distortion_max': round(float(distort.max()), 1) if len(distort) else 0.0,
                  'smooth_p99': round(_pct(ex, 99), 3), 'roughness': round(rough, 3),
                  'sym_p999': round(_pct(ds, 99.9) / H, 5), 'sym_match_pct': round(float(100.0 * np.mean(dv < 1e-4 * H)), 2)}
    # the loops round the eyes and the mouth
    loops, loop_draw = {}, {}
    ends = []
    for fk in LOOP_FEATURES:
        cx, cz_, a, b = M['zones'][fk]
        d = zd[fk]
        found, broken, seen = [], 0, set()
        cand = np.nonzero(np.isfinite(d) & used)[0]
        if len(cand) == 0:
            loops[fk] = {'loops_around': 0, 'broken': 0, 'alignment': 0.0, 'seeds': 0}
            continue
        ptree = cKDTree(Xn[cand][:, [0, 2]])
        seeds = 0
        for t in ((0.3, 0.45, 0.6, 0.8, 1.0) if kind == 'source' else ()):    # (triangles: no loops to walk)
            for ang0 in (90, 270, 0, 180):
                q = (cx + t * a * math.cos(math.radians(ang0)), cz_ + t * b * math.sin(math.radians(ang0)))
                _, j = ptree.query(q)
                v = int(cand[j])
                tang = np.array([-math.sin(math.radians(ang0)) * a, math.cos(math.radians(ang0)) * b])
                tang /= np.linalg.norm(tang)
                best, bv = -1, 0.0
                for e in T.ve_e[T.ve_o[v]:T.ve_o[v + 1]]:
                    u = Xn[T.E[e, 1], [0, 2]] - Xn[T.E[e, 0], [0, 2]]
                    c = abs(float(u @ tang)) / max(np.linalg.norm(u), 1e-12)
                    if c > bv:
                        best, bv = int(e), c
                if best < 0 or (best, v) in seen:
                    continue
                seeds += 1
                seen.add((best, v))
                res = []
                for vv in T.E[best]:
                    res.append(T.walk(best, int(vv)))
                vs0, es0, closed, why = res[0]
                if closed:
                    vs, es = vs0, es0
                else:
                    vs = list(reversed(res[1][0])) + vs0
                    es = list(reversed(res[1][1])) + es0
                key = frozenset(es)
                P2 = Xn[vs][:, [0, 2]]
                dd = np.hypot((P2[:, 0] - cx) / a, (P2[:, 1] - cz_) / b)
                if closed:
                    w = P2 - [cx, cz_]
                    th = np.unwrap(np.arctan2(w[:, 1], w[:, 0]))
                    around = abs(th[-1] - th[0]) > 1.5 * math.pi and float(dd.max()) < 2.2 and \
                        bool(np.all(np.isfinite(d[vs])))
                    if around and key not in {f[0] for f in found}:
                        found.append((key, vs))
                else:
                    for r in res:
                        if not r[2] and r[3] in ('pole', 'triangle or ngon'):
                            ve = r[0][-1]
                            if d[ve] < 1.0:
                                broken += 1
                                ends.append((fk, int(ve), r[3]))
                if not closed and len(loop_draw.get(fk, [])) < 12:
                    loop_draw.setdefault(fk, []).append(('broken', vs[:400]))
        for key, vs in found:
            loop_draw.setdefault(fk, []).append(('loop', vs))
        # alignment of the zone's edges with the feature (round it or out of it)
        ez_ = (d[T.E[:, 0]] < 1.0) & (d[T.E[:, 1]] < 1.0)
        if ez_.any():
            E2 = T.E[ez_]
            u = Xn[E2[:, 1]][:, [0, 2]] - Xn[E2[:, 0]][:, [0, 2]]
            mid = 0.5 * (Xn[E2[:, 1]][:, [0, 2]] + Xn[E2[:, 0]][:, [0, 2]])
            g = np.stack([(mid[:, 0] - cx) / a ** 2, (mid[:, 1] - cz_) / b ** 2], 1)
            L = np.linalg.norm(u, axis=1)
            c = np.abs(np.einsum('ij,ij->i', u, g)) / np.maximum(L * np.linalg.norm(g, axis=1), 1e-12)
            th = np.degrees(np.arccos(np.clip(c, 0, 1)))
            al = float(np.sum(L * ((th < ALIGN_DEG) | (th > 90 - ALIGN_DEG))) / max(L.sum(), 1e-12))
        else:
            al = 0.0
        loops[fk] = {'loops_around': len(found), 'broken': broken, 'alignment': round(al, 3), 'seeds': seeds}
    S['loops'] = loops
    S['summary'] = {
        'quad_pct': round(quad_pct, 2), 'ngons_face': int((sizes[face_f] > 4).sum()), 'poles_zone': poles_zone,
        **{k: S['integrity'][k] for k in ('nonmanifold_edges', 'nonmanifold_verts', 'holes', 'loose',
                                          'stray_islands', 'duplicate_verts', 'degenerate_faces', 'flipped_edges',
                                          'inside_out')},
        **{k: S['shape'][k] for k in ('sharp_p99', 'sharp_max', 'distortion_p99', 'smooth_p99', 'roughness',
                                      'sym_p999', 'sym_match_pct')},
        'loops_min': min(v['loops_around'] for v in loops.values()),
        'broken_max': max(v['broken'] for v in loops.values()),
        'alignment_min': min(v['alignment'] for v in loops.values())}
    # the worst spots, one per place
    sp = []

    def add(kind_, idx, value, what):
        sp.append({'kind': kind_, 'v': int(idx), 'value': value, 'what': what})
    for e in np.nonzero(T.ecount > 2)[0][:50]:
        add('non-manifold edge', T.E[e, 0], int(T.ecount[e]), f'an edge with {int(T.ecount[e])} faces')
    for v in np.nonzero(bow)[0][:50]:
        add('non-manifold vertex', v, 1, 'faces meeting at a point only (a bow tie)')
    for h in holes[:20]:
        add('hole', h['verts'][0], h['length'], f"an open border {h['edges']} edges long" +
            (' (collapsed: its edges have no length)' if h['length'] < 1e-4 else ''))
    for cnt_, at in sorted(dup_at, key=lambda x: -x[0])[:20]:
        j = int(np.argmin(np.linalg.norm(Xn - at, axis=1)))
        add('duplicate vertices', j, cnt_, f'{cnt_} vertices in one place')
    for f in np.nonzero(flipped)[0][:200:10]:
        add('flipped faces', T.E[f, 0], 1, 'neighbouring faces wound against each other')
    if inside_out:
        add('inside out', int(np.argmax(Xn[:, 2])), 1, 'the normals point into the head')
    for f in np.nonzero(degen & ~rep_idx)[0][:100:5]:
        add('degenerate face', T.fv[T.fo[f]], 1, 'a face with no area')
    for fk, v, why in ends[:40]:
        add('broken loop', v, why, f'the loop round the {fk.split()[0]} stops at a {why}')
    pz = np.nonzero(pole & in_zone & (kind == 'source'))[0]
    for v in pz[np.argsort(-np.abs(T.valence[pz] - 4))][:40]:
        add('pole in a deformation zone', v, int(T.valence[v]), f'{int(T.valence[v])} edges meet inside a zone that moves')
    ef_idx = np.nonzero(two)[0][good & e_face]
    if len(ef_idx):
        lim_s = max(DEFAULT_LIMITS['sharp_p99'], _pct(sharp, 99.9))
        for e in ef_idx[np.argsort(-ang[good & e_face])][:10]:
            a_ = float(ang[np.searchsorted(np.nonzero(two)[0], e)])
            if a_ > lim_s:
                add('sharp edge', T.E[e, 0], round(a_, 1), f'neighbouring faces at {a_:.0f} degrees')
    if fz.any():
        idx = np.nonzero(fz)[0]
        for v in idx[np.argsort(-np.abs(exc[idx]))][:30]:
            add('rough contour', v, round(float(abs(exc[v])), 2), 'the curvature jumps against its neighbours')
    if S['shape']['sym_p999'] > CLEAN['sym_p999']:
        j = int(np.argmax(ds))
        add('asymmetry', j, round(float(ds[j] / H), 4), 'the mirrored side is this far off (head heights)')
    order = {'non-manifold edge': 0, 'non-manifold vertex': 0, 'hole': 1, 'inside out': 1, 'flipped faces': 1,
             'duplicate vertices': 2, 'degenerate face': 2, 'pole in a deformation zone': 3, 'broken loop': 3,
             'sharp edge': 4, 'asymmetry': 4, 'rough contour': 5}
    sp.sort(key=lambda s: order.get(s['kind'], 9))
    kept = []
    for s in sp:
        p = Xn[s['v']]
        if any(k_['kind'] == s['kind'] and np.linalg.norm(Xn[k_['v']] - p) < 0.04 for k_ in kept):
            continue
        s['at'] = [round(float(x), 3) for x in p]
        s['where'] = place(p)
        kept.append(s)
        if len(kept) >= 14:
            break
    for i, s in enumerate(kept):
        s['n'] = i + 1
    S['spots'] = kept
    S['seconds'] = round(time.time() - t0, 2)
    F = {'X': X, 'Xn': Xn, 'T': T, 'tri': tri, 'tri_f': tri_f, 'kH': kH, 'exc': exc, 'vn': vn, 'pole': pole,
         'zd': zd, 'loops': loop_draw, 'ends': ends}
    return S, F


# ------------------------------------------------------------------------------------------------ the verdicts

def limits(kind, name=None):
    """The band a head is judged on: its own, when John approved this head before (topo_baseline.json's heads, by
    name: a face change must leave it no worse than he passed it), else its kind's (the worst of every approved
    head of that kind). Returns (limits, file, which)."""
    p = os.path.join(HERE, 'topo_baseline.json')
    try:
        b = json.load(open(p))
    except (OSError, ValueError):
        return dict(DEFAULT_LIMITS), None, 'defaults'
    if name and name in b.get('heads', {}):
        return dict(DEFAULT_LIMITS, **b['heads'][name]), p, f'its own approved band ({name})'
    if kind in b.get('kinds', {}):
        return dict(DEFAULT_LIMITS, **b['kinds'][kind]), p, f'every approved {kind} head'
    return dict(DEFAULT_LIMITS), None, 'defaults'


def judge(S, lim, kind):
    """Flags against the approved band: 'watch' past a limit, 'likely rejection' well past it."""
    flags, worst = [], 'ok'
    sm = S['summary']
    for k, low in METRICS.items():
        if kind == 'game' and k in STRUCTURE:
            continue
        x, v = sm.get(k), lim.get(k)
        if x is None or v is None:
            continue
        over = x < v if low else x > v
        if not over:
            continue
        if k in COUNTS:
            far = x > 2 * v + 3
        else:
            far = x < v / 1.5 if low else x > v * 1.5
        lvl = 'likely rejection' if far else 'watch'
        flags.append({'metric': k, 'value': x, 'limit': v, 'level': lvl})
        if lvl == 'likely rejection' or worst == 'ok':
            worst = lvl
    flags.sort(key=lambda f: f['level'] != 'likely rejection')
    return flags, worst


def clean(S, kind):
    """The absolute standard: what fails it, in words."""
    sm, fails = S['summary'], []
    words = {'nonmanifold_edges': 'non-manifold edges', 'nonmanifold_verts': 'non-manifold vertices',
             'holes': 'holes', 'loose': 'loose vertices or wire edges', 'stray_islands': 'stray islands',
             'duplicate_verts': 'duplicate vertices', 'degenerate_faces': 'degenerate faces',
             'flipped_edges': 'edges between faces wound against each other', 'inside_out': 'inside out',
             'ngons_face': 'ngons on the face', 'poles_zone': 'poles inside the eye and mouth zones'}
    for k, v in CLEAN.items():
        if kind == 'game' and k not in CLEAN_GAME:
            continue
        x = sm.get(k)
        if x is None:
            continue
        if k == 'loops_min' and x < v:
            fails.append(f"fewer than {v} concentric loops round " +
                         ', '.join(f for f, L in S['loops'].items() if L['loops_around'] < v) +
                         f" (fewest {x})")
        elif k == 'alignment_min' and x < v:
            fails.append(f"edges not following the eyes and mouth (alignment {x}, needs {v})")
        elif k == 'sym_p999' and x > v:
            fails.append(f'asymmetric: the mirrored head {x} H off (p99.9; needs {v})')
        elif k not in ('loops_min', 'alignment_min', 'sym_p999') and x > v:
            fails.append(f'{x} {words.get(k, k)}')
    return {'ok': not fails, 'fails': fails}


def badness(k, v):
    return -v if METRICS[k] else v


def calibrate(approved, by='kind'):
    """Per kind, each metric's worst approved value: counts as they are (any more is flagged), measures with the
    margin (x 1.25 on the measure, / 1.25 where lower is worse), never tighter than the defaults' sense of
    perfect."""
    out = {}
    groups = ('source', 'game') if by == 'kind' else sorted({S['name'] for S in approved.values()})
    for kind in groups:
        rows = {n: S for n, S in approved.items() if (S['kind'] if by == 'kind' else S['name']) == kind}
        if not rows:
            continue
        lim = {}
        for k, low in METRICS.items():
            vals = [S['summary'][k] for S in rows.values() if S['summary'].get(k) is not None]
            if not vals:
                continue
            w = min(vals) if low else max(vals)
            if k in COUNTS:
                v = int(w)
            elif k == 'quad_pct':
                v = round(max(0.0, w - 2.0), 2)
            elif k == 'sym_match_pct':
                v = round(max(0.0, w - 5.0), 2)
            elif k == 'loops_min':
                v = int(w)
            elif low:
                v = round(w / MARGIN, 4)
            else:
                v = round(max(w * MARGIN, 1e-6), 5)
            if k in ('sharp_p99', 'sharp_max'):
                v = min(v, 180.0)
            lim[k] = v
        out[kind] = lim
    return out


def line(rep):
    bits = []
    for n, S in rep['heads'].items():
        c = S['clean']
        bits.append(f"{n} [{S['kind']}]: {S['verdict']}" +
                    (f" ({'; '.join(f['metric'] + ' ' + str(f['value']) + ' vs ' + str(f['limit']) for f in S['flags'][:3])})"
                     if S['flags'] else '') + f", clean: {'yes' if c['ok'] else 'no, ' + '; '.join(c['fails'][:3])}")
    return ' | '.join(bits) or 'no head found'


# ------------------------------------------------------------------------------------------------ Blender

def head_objects():
    """The head's own skin in the scene (as female._head_meshes), the Head empty, and the eyeballs' centres."""
    import bpy
    ob = bpy.data.objects
    heads = [o for o in ob if o.type == 'MESH' and o.name.startswith(('Head shape', 'head part 0'))]
    head = ob.get('Head')
    if not heads:      # (a creator export: the largest mesh under the Head node that is not the hair)
        kids = [o for o in ob if o.type == 'MESH' and o.parent is not None and o.parent.name.startswith('Head')
                and 'hair' not in o.name.lower()]
        if kids:
            heads = [max(kids, key=lambda o: len(o.data.vertices))]
            head = heads[0].parent
    return heads, head


def eye_centres(head_ob, head, H_guess=None):
    """The eyeballs' centres in the Head frame, found by place, not by name: the points of the other small meshes
    (under a quarter of the head high) that lie in the eye band in front, either side of the middle, averaged per
    side. (A merged game head joins both eyes' whites and irises into one mesh, so a mesh's own centre will not
    do.)"""
    import bpy
    inv = np.array(head.matrix_world.inverted()) if head is not None else np.eye(4)

    def pts(o):
        X = np.empty(len(o.data.vertices) * 3)
        o.data.vertices.foreach_get('co', X)
        m = inv @ np.array(o.matrix_world)
        return X.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
    Xh = pts(head_ob)
    top = Xh[:, 2].max()
    depth = Xh[:, 1].max() - Xh[:, 1].min()
    fr = Xh[(np.abs(Xh[:, 0]) < 0.08 * depth) & (Xh[:, 1] < Xh[:, 1].min() + 0.3 * depth)]
    chin = fr[:, 2].min() if len(fr) else Xh[:, 2].min()
    H = top - chin
    yfront = Xh[:, 1].min()
    side = {-1: [], 1: []}
    for o in bpy.data.objects:
        if o.type != 'MESH' or o is head_ob or len(o.data.vertices) < 40:
            continue
        nm = o.name.lower()
        if any(w in nm for w in ('lash', 'brow', 'shadow', 'paint', 'lid', 'hair', 'bang')):
            continue
        P = pts(o)
        if P[:, 2].max() - P[:, 2].min() > 0.25 * H and len(P) > 200000:
            continue
        z = (P[:, 2] - chin) / H
        k = (z > 0.42) & (z < 0.64) & (np.abs(P[:, 0]) > 0.04 * H) & (np.abs(P[:, 0]) < 0.3 * H) & \
            (P[:, 1] < yfront + 0.35 * depth)
        if not k.any():
            continue
        Q = P[k]
        # (only a mesh that is mostly in that band: an eye, not a hood or a collar that passes through it)
        if k.mean() < 0.6:
            continue
        for s_ in (-1, 1):
            q = Q[np.sign(Q[:, 0]) == s_]
            if len(q):
                side[s_].append(q)
    if side[-1] and side[1]:
        return [tuple(np.concatenate(side[s_]).mean(0)) for s_ in (-1, 1)]
    return None


def collect(o, head=None):
    """The head mesh as built (its modifiers applied, in its rest shape), in the Head frame: points, faces (CSR),
    loose edges, and a surface distance function for the symmetry."""
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    ev = o.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        n = len(me.vertices)
        X = np.empty(n * 3)
        me.vertices.foreach_get('co', X)
        X = X.reshape(-1, 3)
        m = (head.matrix_world.inverted() @ o.matrix_world) if head is not None else o.matrix_world
        Mx = np.array(m)
        X = X @ Mx[:3, :3].T + Mx[:3, 3]
        fo = np.zeros(len(me.polygons) + 1, np.int64)
        tot = np.empty(len(me.polygons), np.int64)
        me.polygons.foreach_get('loop_total', tot)
        fo[1:] = np.cumsum(tot)
        lv = np.empty(len(me.loops), np.int64)
        me.loops.foreach_get('vertex_index', lv)
        ev_ = np.empty(len(me.edges) * 2, np.int64)
        me.edges.foreach_get('vertices', ev_)
        ev_ = ev_.reshape(-1, 2)
        ek = set((np.minimum(lv, lv[np.r_[1:len(lv), 0]]) * n + np.maximum(lv, lv[np.r_[1:len(lv), 0]])).tolist())
        loose_edges = int(sum(1 for a, b in ev_ if min(a, b) * n + max(a, b) not in ek)) if len(ev_) < 2_000_000 else 0
    finally:
        ev.to_mesh_clear()
    joined = False
    if len(tot) and (tot == 3).all():          # (a glTF: triangles only; back to the quads it was cut from)
        X, fo, lv, joined = join_tris(X, fo, lv)
    bvh = BVHTree.FromPolygons([tuple(p) for p in X.tolist()],
                               [tuple(lv[fo[i]:fo[i + 1]].tolist()) for i in range(len(fo) - 1)], epsilon=0.0)

    def nearest(P):
        out = np.empty(len(P))
        for i, p in enumerate(P.tolist()):
            r = bvh.find_nearest(Vector(p))
            out[i] = r[3] if r[0] is not None else np.inf
        return out
    return X, fo, lv, loose_edges, joined, nearest


def join_tris(X, fo, lv):
    """A triangle mesh (a glTF) back to the quads it was cut from, when it was: an exporter (three.js, Blender's
    glTF) splits each quad into two triangles one after the other, so when nearly every consecutive pair shares an
    edge (98%), each pair is joined back into its quad exactly. Otherwise the triangles are the mesh itself (a
    decimated game head) and are left as they are. Returns (X, fo, lv, joined)."""
    T = lv.reshape(-1, 3)
    m = len(T) // 2 * 2
    A, B = T[0:m:2], T[1:m:2]
    inb = (A[:, :, None] == B[:, None, :]).any(2)          # (each of A's corners: is it in B?)
    shared = inb.sum(1) == 2
    if m == 0 or shared.mean() < 0.98:
        return X, fo, lv, False
    quads = []
    for k in range(m // 2):
        a, b = A[k], B[k]
        if not shared[k]:
            quads.append(list(a))
            quads.append(list(b))
            continue
        i = int(np.nonzero(~inb[k])[0][0])       # (a's corner not on the shared edge: the edge runs i+1 -> i+2)
        d = int(b[~(b[:, None] == a[None, :]).any(1)][0])
        quads.append([int(a[i]), int(a[(i + 1) % 3]), d, int(a[(i + 2) % 3])])
    if len(T) % 2:
        quads.append(list(T[-1]))
    fo2 = np.zeros(len(quads) + 1, np.int64)
    fo2[1:] = np.cumsum([len(q) for q in quads])
    return X, fo2, np.array([v for q in quads for v in q], np.int64), True


def run(out, objects=None, head=None, stem='topo', title='', label=None):
    """The topology check on the head(s) in the scene: DIR/topo.json and DIR/topo-sheet.png. Never raises."""
    t = time.time()
    rep = {'version': VERSION, 'heads': {}, 'skipped': {}}
    try:
        if objects is None:
            objects, head = head_objects()
    except Exception as e:
        rep['skipped']['scene'] = f'{type(e).__name__}: {e}'
        objects = []
    items = []
    for o in objects:
        try:
            X, fo, fv, le, joined, nearest = collect(o, head)
            eyes = eye_centres(o, head)
            S, F = analyse(label or o.name, X, fo, fv, eyes=eyes, loose_edges=le, joined=joined, nearest=nearest)
            lim, src, which = limits(S['kind'], S['name'])
            S['limits_from'] = (os.path.basename(src) + ': ' if src else '') + which
            S['flags'], S['verdict'] = judge(S, lim, S['kind'])
            S['clean'] = clean(S, S['kind'])
            S['limits'] = lim
            rep['heads'][S['name']] = S
            items.append((S, F))
        except Exception as e:      # (a check must never fail the render)
            import traceback
            traceback.print_exc()
            rep['skipped'][o.name] = f'{type(e).__name__}: {e}'
    order = {'ok': 0, 'watch': 1, 'likely rejection': 2}
    rep['verdict'] = max([S['verdict'] for S in rep['heads'].values()] or ['ok'], key=order.get)
    rep['clean'] = all(S['clean']['ok'] for S in rep['heads'].values()) if rep['heads'] else None
    for S, F in items:
        try:
            p = os.path.join(out, stem + ('-sheet.png' if len(items) == 1 else f"-sheet-{_slug(S['name'])}.png"))
            sheet(S, F, p, title or f"topology: {S['name']}")
            S['sheet'] = os.path.basename(p)
        except Exception as e:
            import traceback
            traceback.print_exc()
            S['sheet_error'] = f'{type(e).__name__}: {e}'
    rep['seconds'] = round(time.time() - t, 1)
    with open(os.path.join(out, stem + '.json'), 'w') as f:
        json.dump(rep, f, indent=1, default=_js)
    print('rq topo: ' + line(rep), flush=True)
    return rep


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _slug(s):
    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')


# ------------------------------------------------------------------------------------------------ the sheet

BG, INK, RING = (246, 245, 242), (40, 40, 44), (205, 48, 40)
CLAY = np.array([214, 212, 206], float)
RAMP = [(0.0, (214, 212, 206)), (0.35, (158, 197, 244)), (0.7, (85, 152, 231)), (1.0, (37, 106, 191)),
        (1.5, (13, 54, 107))]
POLE_COL = {3: (230, 140, 20), 5: (150, 60, 190), 6: (205, 48, 40)}
LOOP_COL, BROKEN_COL, WIRE = (30, 150, 80), (205, 48, 40), (70, 70, 78)


def ramp(t):
    t = np.clip(np.nan_to_num(t), 0, RAMP[-1][0])
    xs = np.array([a for a, _ in RAMP])
    cs = np.array([c for _, c in RAMP], float)
    return np.stack([np.interp(t, xs, cs[:, k]) for k in range(3)], -1)


class View:
    """An orthographic view about the Head frame's z axis: az degrees from the front (0: the face, toward -y)."""

    def __init__(self, az, centre, half, W, Hh=None):
        self.az, self.c, self.half, self.W, self.H = math.radians(az), np.asarray(centre, float), half, W, Hh or W

    def project(self, P):
        a = self.az
        x = (P[..., 0] - self.c[0]) * math.cos(a) + (P[..., 1] - self.c[1]) * math.sin(a)
        dep = -(P[..., 0] - self.c[0]) * math.sin(a) + (P[..., 1] - self.c[1]) * math.cos(a)
        s = self.W / (2 * self.half)
        return np.stack([self.W / 2 + x * s, self.H / 2 - (P[..., 2] - self.c[2]) * s], -1), dep

    def toward(self):
        return np.array([math.sin(self.az), -math.cos(self.az), 0.0])     # (from the head toward the camera)


def raster(V, P, tri, col, vn=None):
    """A z-buffered image of the triangles (numpy, a triangle's pixels at a time for the big ones, a point splat for
    the small ones) and its depth: the cheap draft render, no Blender needed. col: [n, 3] per vertex."""
    W, H = V.W, V.H
    p2, dep = V.project(P)
    img = np.zeros((H, W, 3)) + BG
    zb = np.full((H, W), np.inf)
    A, B, C = p2[tri[:, 0]], p2[tri[:, 1]], p2[tri[:, 2]]
    x0 = np.floor(np.minimum(np.minimum(A[:, 0], B[:, 0]), C[:, 0])).astype(int)
    x1 = np.ceil(np.maximum(np.maximum(A[:, 0], B[:, 0]), C[:, 0])).astype(int)
    y0 = np.floor(np.minimum(np.minimum(A[:, 1], B[:, 1]), C[:, 1])).astype(int)
    y1 = np.ceil(np.maximum(np.maximum(A[:, 1], B[:, 1]), C[:, 1])).astype(int)
    vis = (x1 >= 0) & (x0 < W) & (y1 >= 0) & (y0 < H)
    if vn is not None:      # (back faces culled: a splat of one behind would show through the fine front)
        tw = vn @ V.toward()
        vis &= tw[tri].max(1) > -0.05
    small = vis & ((x1 - x0) <= 2) & ((y1 - y0) <= 2)
    # the small ones: their centre, nearest wins
    s = np.nonzero(small)[0]
    if len(s):
        cx = ((A[s, 0] + B[s, 0] + C[s, 0]) / 3).astype(int)
        cy = ((A[s, 1] + B[s, 1] + C[s, 1]) / 3).astype(int)
        dz = dep[tri[s]].mean(1)
        ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
        cx, cy, dz, s = cx[ok], cy[ok], dz[ok], s[ok]
        o = np.argsort(-dz)
        cx, cy, dz, s = cx[o], cy[o], dz[o], s[o]
        flat = cy * W + cx
        _, first = np.unique(flat[::-1], return_index=True)
        k = len(flat) - 1 - first
        zf = zb.ravel()
        better = dz[k] < zf[flat[k]]
        k = k[better]
        zf[flat[k]] = dz[k]
        img.reshape(-1, 3)[flat[k]] = col[tri[s[k]]].mean(1)
    for t in np.nonzero(vis & ~small)[0]:
        xa, xb, ya, yb = max(x0[t], 0), min(x1[t], W - 1), max(y0[t], 0), min(y1[t], H - 1)
        gx, gy = np.meshgrid(np.arange(xa, xb + 1) + 0.5, np.arange(ya, yb + 1) + 0.5)
        a, b, c = A[t], B[t], C[t]
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-12:
            continue
        l1 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / den
        l2 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / den
        l3 = 1 - l1 - l2
        inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        if not inside.any():
            continue
        dz = l1 * dep[tri[t, 0]] + l2 * dep[tri[t, 1]] + l3 * dep[tri[t, 2]]
        sub = zb[ya:yb + 1, xa:xb + 1]
        w = inside & (dz < sub)
        sub[w] = dz[w]
        cc = l1[..., None] * col[tri[t, 0]] + l2[..., None] * col[tri[t, 1]] + l3[..., None] * col[tri[t, 2]]
        img[ya:yb + 1, xa:xb + 1][w] = cc[w]
    # (pixels a splat missed inside the silhouette: filled from their neighbours)
    for _ in range(2):
        m = ~np.isfinite(zb)
        pad = np.pad(np.isfinite(zb), 1)
        n4 = pad[:-2, 1:-1].astype(int) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
        hole = m & (n4 >= 3)
        if not hole.any():
            break
        ys, xs = np.nonzero(hole)
        acc, za, cnt = np.zeros((len(ys), 3)), np.zeros(len(ys)), np.zeros(len(ys))
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = np.clip(ys + dy, 0, H - 1), np.clip(xs + dx, 0, W - 1)
            ok = np.isfinite(zb[yy, xx])
            acc[ok] += img[yy[ok], xx[ok]]
            za[ok] += zb[yy[ok], xx[ok]]
            cnt[ok] += 1
        img[ys, xs] = acc / cnt[:, None]
        zb[ys, xs] = za / cnt
    return np.clip(img, 0, 255).astype(np.uint8), zb


def shade(F, V, extra=None):
    """Clay lit from the front-left, with curvature shading (convex lighter, concave darker), per vertex."""
    vn = F['vn']
    L = np.array([-0.45, -0.8, 0.45])
    a = V.az
    Lr = np.array([L[0] * math.cos(a) - L[1] * math.sin(a), L[0] * math.sin(a) + L[1] * math.cos(a), L[2]])
    Lr /= np.linalg.norm(Lr)
    lit = 0.35 + 0.65 * np.clip(vn @ Lr, 0, 1)
    kH = F['kH']
    sc = max(_pct(np.abs(kH), 95), 1e-6)
    cv = np.clip(kH / sc, -1.5, 1.5)
    base = CLAY[None, :] if extra is None else extra
    return base * lit[:, None] * (1 + 0.22 * cv)[:, None]


def sheet(S, F, path, title=''):
    """The sheet: clay with curvature (front, 3/4, profile), the close-ups with the wireframe, and the contour's
    roughness. The worst spots ringed and numbered as in topo.json."""
    from PIL import Image, ImageDraw
    Xn, T, tri = F['Xn'], F['T'], F['tri']
    M = S['measures']
    big = 300
    cen = np.array([0.0, float(np.median(Xn[:, 1])), 0.5])
    full = [('front', View(0, cen, 0.62, big)), ('3/4', View(45, cen, 0.62, big)), ('profile', View(90, cen, 0.62, big))]
    close = []
    for k in ('eye R', 'eye L', 'mouth', 'nose'):
        cx, cz, a, b = M['zones'][k]
        close.append((k, View(0, (cx, cen[1], cz), 1.25 * max(a, b), big), k))
    lim = S['limits']
    rows = 3
    Wsh = 4 * big + 20
    Hsh = rows * (big + 22) + 70
    im = Image.new('RGB', (Wsh, Hsh), BG)
    d = ImageDraw.Draw(im)
    d.text((8, 6), title[:150], fill=INK)
    d.text((8, 20), f"{S['kind']}: {S['counts']['verts']} verts, {S['counts']['quad_pct']}% quads"
           f"{' (joined from triangles)' if S['counts']['joined_from_triangles'] else ''}, verdict {S['verdict']}, clean "
           f"{'yes' if S['clean']['ok'] else 'no'}; loops round eyes/mouth {[v['loops_around'] for v in S['loops'].values()]}, "
           f"alignment {[v['alignment'] for v in S['loops'].values()]}", fill=INK)

    def ring_spots(V, x0, y0, zb):
        for s in S['spots']:
            p = Xn[s['v']]
            q, dep = V.project(p[None])
            X0, Y0 = q[0]
            if not (0 <= X0 < V.W and 0 <= Y0 < V.H):
                continue
            if dep[0] > zb[int(Y0), int(X0)] + 0.02:       # (hidden behind the head)
                continue
            d.ellipse((x0 + X0 - 8, y0 + Y0 - 8, x0 + X0 + 8, y0 + Y0 + 8), outline=RING, width=2)
            d.text((x0 + X0 + 9, y0 + Y0 - 7), str(s['n']), fill=RING)

    y = 40
    for i, (nm, V) in enumerate(full):
        col = shade(F, V)
        img, zb = raster(V, Xn, tri, col, F['vn'])
        im.paste(Image.fromarray(img), (i * big, y))
        d.text((i * big + 6, y + 2), nm + ' (clay, curvature)', fill=INK)
        if nm == 'front':
            for k, (cx, cz, a, b) in M['zones'].items():
                q, _ = V.project(np.array([[cx, 0, cz]]))
                s_ = V.W / (2 * V.half)
                d.ellipse((i * big + q[0, 0] - a * s_, y + q[0, 1] - b * s_, i * big + q[0, 0] + a * s_,
                           y + q[0, 1] + b * s_), outline=(120, 120, 130))
        ring_spots(V, i * big, y, zb)
    # the measures beside the views
    tx = 3 * big + 10
    lines_ = [f"H {M['H']}  IPD {M['ipd']} H", f"eye line {M['eye_z']} H", f"nose tip {M['nose_tip'][2]} H",
              f"stomion {M['stomion_z']} H ({M['from']['stomion'].split()[0]})",
              f"mouth {M['mouth_width']} H ({M['from']['mouth_width'].split()[0]})", '',
              'flags:'] + [f" {f['metric']} {f['value']} vs {f['limit']}" for f in S['flags'][:6]] + \
        ['', 'not clean:' if not S['clean']['ok'] else 'clean'] + [' ' + f[:44] for f in S['clean']['fails'][:6]] + \
        ['', 'spots:'] + [f" {s['n']} {s['kind']}: {s['where']}"[:46] for s in S['spots'][:8]]
    for j, L in enumerate(lines_):
        d.text((tx, y + 2 + 12 * j), L, fill=INK)
    y += big + 22
    for i, (nm, V, fk) in enumerate(close):
        col = shade(F, V)
        img, zb = raster(V, Xn, tri, col, F['vn'])
        tile = Image.fromarray(img)
        td = ImageDraw.Draw(tile)
        # the wireframe of the faces turned to the camera; on a mesh too fine to draw (edges under 4 pixels) every
        # Nth edge loop instead, walked through the panel in both directions: the mesh's flow, as it runs
        p2, dep = V.project(Xn)
        tw = F['vn'] @ V.toward()
        E = T.E

        def seen(v):
            X0, Y0 = p2[v]
            return 0 <= X0 < V.W and 0 <= Y0 < V.H and tw[v] > 0.05 and dep[v] <= zb[int(Y0), int(X0)] + 0.01
        inwin = ((p2[E[:, 0], 0] > -5) & (p2[E[:, 0], 0] < V.W + 5) & (p2[E[:, 0], 1] > -5) & (p2[E[:, 0], 1] < V.H + 5))
        front = inwin & (tw[E[:, 0]] > 0.05) & (tw[E[:, 1]] > 0.05)
        el = np.linalg.norm(p2[E[front, 0]] - p2[E[front, 1]], axis=1)
        fine = len(el) and float(np.median(el)) < (2.0 if S['kind'] == 'game' else 4.0)
        if S['kind'] == 'game':         # (triangles: no loops to walk; the wire as it is, when it can be seen)
            if not fine:
                for e in np.nonzero(front)[0]:
                    a, b = E[e]
                    if seen(a):
                        td.line((p2[a, 0], p2[a, 1], p2[b, 0], p2[b, 1]), fill=WIRE, width=1)
        elif not fine:
            for e in np.nonzero(front)[0]:
                a, b = E[e]
                if seen(a):
                    td.line((p2[a, 0], p2[a, 1], p2[b, 0], p2[b, 1]), fill=WIRE, width=1)
        else:
            from scipy.spatial import cKDTree
            cand = np.nonzero((p2[:, 0] >= 0) & (p2[:, 0] < V.W) & (p2[:, 1] >= 0) & (p2[:, 1] < V.H) & (tw > 0.05))[0]
            kt = cKDTree(p2[cand])
            step = 14
            for gy in np.arange(step / 2, V.H, step):
                for gx in np.arange(step / 2, V.W, step):
                    _, j = kt.query((gx, gy))
                    v = int(cand[j])
                    for e in T.ve_e[T.ve_o[v]:T.ve_o[v + 1]]:
                        u = p2[E[e, 1]] - p2[E[e, 0]]
                        horiz = abs(u[0]) > abs(u[1])
                        # (from a seed on the left edge, the loops that run across; from one on the top, down)
                        if (horiz and gx > step) or (not horiz and gy > step):
                            continue
                        for w in E[e]:
                            vs = T.walk(int(e), int(w), limit=1500)[0]
                            pts = [tuple(p2[q]) for q in vs if seen(q)]
                            if len(pts) > 1:
                                td.line(pts, fill=WIRE, width=1)
            d.text((i * big + 6, y + big + 6), f'(mesh too fine to draw: every {step} px an edge loop)', fill=INK)
        for kind_, vs in F['loops'].get(fk, []):        # (the loops found round the feature, green)
            pts = [tuple(p2[v]) for v in vs if tw[v] > 0.05]
            if kind_ == 'loop' and len(pts) > 1:
                td.line(pts, fill=LOOP_COL, width=2)
        for fk2, v, why in F['ends']:          # (where a loop walk from the feature broke off, a red cross)
            if fk2 == fk and seen(v):
                X0, Y0 = p2[v]
                td.line((X0 - 4, Y0 - 4, X0 + 4, Y0 + 4), fill=BROKEN_COL, width=2)
                td.line((X0 - 4, Y0 + 4, X0 + 4, Y0 - 4), fill=BROKEN_COL, width=2)
        for v in np.nonzero(F['pole'] & (tw > 0.05) & (S['kind'] == 'source'))[0]:
            X0, Y0 = p2[v]
            if 0 <= X0 < V.W and 0 <= Y0 < V.H and dep[v] <= zb[int(Y0), int(X0)] + 0.01:
                c = POLE_COL.get(min(int(T.valence[v]), 6) if T.valence[v] != 3 else 3, POLE_COL[6])
                td.ellipse((X0 - 2.5, Y0 - 2.5, X0 + 2.5, Y0 + 2.5), fill=c)
        im.paste(tile, (i * big, y))
        L = S['loops'].get(fk)
        d.text((i * big + 6, y + 2), f"{nm}: " + (f"{L['loops_around']} loops, {L['broken']} broken, align {L['alignment']}"
                                                  if L else 'wireframe'), fill=INK)
        ring_spots(V, i * big, y, zb)
    y += big + 22
    for i, (nm, V) in enumerate(full):
        heat = ramp(np.abs(F['exc']) / max(lim.get('smooth_p99') or 1e-6, 1e-6))
        col = shade(F, V, heat)
        img, zb = raster(V, Xn, tri, col, F['vn'])
        im.paste(Image.fromarray(img), (i * big, y))
        d.text((i * big + 6, y + 2), nm + ': contour roughness / limit', fill=INK)
        ring_spots(V, i * big, y, zb)
    lx, ly = 3 * big + 10, y + 40
    for i in range(120):
        c = tuple(int(v) for v in ramp(np.array([i / 80.0]))[0])
        d.line((lx + i, ly, lx + i, ly + 10), fill=c)
    d.text((lx, ly + 12), '0       1.0 = limit  1.5', fill=INK)
    d.text((lx, ly + 34), 'wire: green = loop found round', fill=INK)
    d.text((lx, ly + 46), 'the feature, red x = loop broke', fill=INK)
    d.text((lx, ly + 58), 'dots: orange 3-pole, purple', fill=INK)
    d.text((lx, ly + 70), '5-pole, red 6+; rings: spots', fill=INK)
    im.save(path)
    return path


# ------------------------------------------------------------------------------------------------ the trigger

def face_changed(repo='.', base='origin/main', argv=()):
    """Whether the commit checked out at repo changes a face or head file against base (its merge base): a file
    whose path names a face part, or a change inside a face or head function or setting of a Python file (female.py,
    a figure's own file). Also a --set of a face setting. Returns (changed, why)."""
    import subprocess
    for i, a in enumerate(argv):
        if a == '--set' and i + 1 < len(argv) and FACE_DEFS.search(argv[i + 1].split('=')[0]):
            return True, f'--set {argv[i + 1]}'

    def git(*a):
        return subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True).stdout
    mb = git('merge-base', 'HEAD', base).strip()
    if not mb:
        return False, f'no merge base with {base}'
    names = [n for n in git('diff', '--name-only', mb, 'HEAD').split('\n') if n]
    for n in names:
        if FACE_PATHS.search(n):
            return True, n
    for n in names:
        if not n.endswith('.py') or not os.path.exists(os.path.join(repo, n)):
            continue
        hunks = re.findall(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', git('diff', '-U0', mb, 'HEAD', '--', n), re.M)
        if not hunks:
            continue
        src = open(os.path.join(repo, n)).read().split('\n')
        for start, count in hunks:
            s = int(start)
            for ln in range(s, s + max(int(count or 1), 1)):
                # the enclosing def or the assignment on that line
                j = min(ln - 1, len(src) - 1)
                if j >= 0 and re.match(r'^[A-Z_][A-Z0-9_]*\s*[,=]', src[j]) and FACE_DEFS.search(src[j].split('=')[0]):
                    return True, f'{n}: {src[j].split("=")[0].strip()}'
                while j >= 0 and not re.match(r'^(def|class) ', src[j]):
                    if src[j] and not src[j][0].isspace() and not src[j].startswith(('#', ')', ']', '}')):
                        break
                    j -= 1
                if j >= 0 and re.match(r'^(def|class) ', src[j]):
                    nm = re.match(r'^(?:def|class) (\w+)', src[j]).group(1)
                    if FACE_DEFS.search(nm):
                        return True, f'{n}: {nm}()'
    return False, 'no face or head change'


# ------------------------------------------------------------------------------------------------ test heads

def grid_head(nr=120, ns=160, bump=0.0, hole=False, flip=False, dup=False):
    """A lat-long grid head (as the creator's and the sculpt's: rings up, segments round), chin at 0, face -y."""
    zs = np.linspace(-0.15, 1.0, nr)
    th = np.linspace(-math.pi, math.pi, ns, endpoint=False)
    P = []
    for z in zs:
        r = 0.42 * math.sqrt(max(1e-4, 1 - ((z - 0.5) / 0.62) ** 2))
        for t in th:
            x, y = r * math.sin(t), -r * math.cos(t) * 1.15
            if bump and y < 0:
                y -= bump * math.exp(-((x - 0.1) ** 2 + (z - 0.4) ** 2) / 0.0004)
            P.append((x, y, z))
    P = np.array(P)
    fo, fv = [0], []
    for j in range(nr - 1):
        for i in range(ns):
            if hole and j == nr // 2 and i == 0:
                continue
            a, b, c, d = j * ns + i, j * ns + (i + 1) % ns, (j + 1) * ns + (i + 1) % ns, (j + 1) * ns + i
            q = [a, b, c, d] if not (flip and j == nr // 3 and i == ns // 2) else [d, c, b, a]
            fv.extend(q)
            fo.append(len(fv))
    if dup:
        P = np.vstack([P, P[:5]])
    return P, np.array(fo), np.array(fv)


def loop_head(rings=10, ns=48):
    """A head whose front has concentric loops round two eyes and a mouth: a disc of rings about each feature,
    stitched into nothing (a flat face plate); enough to test the loop walk."""
    P, fo, fv = [], [0], []
    for (cx, cz, a, b) in ((-0.16, 0.53, 0.1, 0.07), (0.16, 0.53, 0.1, 0.07), (0.0, 0.24, 0.2, 0.12)):
        base = len(P)
        for r in range(rings):
            s = 0.3 + 1.2 * r / (rings - 1)
            for i in range(ns):
                t = 2 * math.pi * i / ns
                x, z = cx + s * a * math.cos(t), cz + s * b * math.sin(t)
                P.append((x, -0.45 + 0.2 * (x * x + (z - 0.5) ** 2), z))
        for r in range(rings - 1):
            for i in range(ns):
                a_, b_ = base + r * ns + i, base + r * ns + (i + 1) % ns
                c_, d_ = base + (r + 1) * ns + (i + 1) % ns, base + (r + 1) * ns + i
                fv.extend([a_, d_, c_, b_])
                fo.append(len(fv))
    # (a crown and chin so the measure finds the head's height)
    P.append((0.0, -0.4, -0.0))
    P.append((0.0, 0.0, 1.0))
    return np.array(P), np.array(fo), np.array(fv)


def selftest(out):
    os.makedirs(out, exist_ok=True)
    rep = {}
    cases = [('grid', grid_head(), {}), ('grid bump', grid_head(bump=0.02), {}), ('grid hole', grid_head(hole=True), {}),
             ('grid flip', grid_head(flip=True), {}), ('grid dup', grid_head(dup=True), {})]
    for nm, (P, fo, fv), _ in cases:
        S, F = analyse(nm, P, fo, fv, eyes=[(-0.16, -0.35, 0.53), (0.16, -0.35, 0.53)], kind='source')
        S['flags'], S['verdict'] = judge(S, DEFAULT_LIMITS, 'source')
        S['clean'] = clean(S, 'source')
        S['limits'] = DEFAULT_LIMITS
        rep[nm] = S
        print(nm, S['summary'], S['clean']['fails'][:4])
        if nm == 'grid':
            sheet(S, F, os.path.join(out, 'selftest-grid.png'), 'selftest: grid head')
    g, h, fl, du, bu = (rep[k]['summary'] for k in ('grid', 'grid hole', 'grid flip', 'grid dup', 'grid bump'))
    assert g['loops_min'] == 0 and g['holes'] == 1 and g['flipped_edges'] == 0 and g['duplicate_verts'] == 0, g
    assert h['holes'] == 2, h      # (the test grid is open at the crown: one hole there)
    assert fl['flipped_edges'] == 4, fl
    assert du['duplicate_verts'] == 5 and du['loose'] == 5, du      # (copies in no face: loose too)
    assert bu['smooth_p99'] > g['smooth_p99'] and bu['sym_p999'] > g['sym_p999'], (bu, g)
    P, fo, fv = loop_head()
    T = Topo(len(P), fo, fv)
    S, F = analyse('loops', P, fo, fv, eyes=[(-0.16, -0.4, 0.53), (0.16, -0.4, 0.53)], kind='source')
    print('loops', {k: v for k, v in S['loops'].items()})
    assert all(v['loops_around'] >= 2 for v in S['loops'].values()), S['loops']
    assert all(v['alignment'] > 0.75 for v in S['loops'].values()), S['loops']
    assert T.valence.max() <= 4
    print('selftest passed', flush=True)


# ------------------------------------------------------------------------------------------------ main

def main():
    a = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d  # noqa: E731
    if '--selftest' in a:
        selftest(opt('--selftest'))
    elif '--calibrate' in a:
        i = a.index('--calibrate')
        dst, srcs = a[i + 1], a[i + 2:]
        approved = {}
        for p in srcs:
            rep = json.load(open(p))
            for n, S in rep['heads'].items():
                approved[f'{n} / {os.path.basename(os.path.dirname(os.path.abspath(p)))}'] = S
        kinds = calibrate(approved)
        heads = calibrate(approved, by='name')
        for n, S in approved.items():        # (every approved head judged on the bands it made: none is flagged)
            for lim in (kinds[S['kind']], heads[S['name']]):
                fl, _ = judge(S, dict(DEFAULT_LIMITS, **lim), S['kind'])
                assert not fl, (n, fl)
        json.dump({'_doc': 'The approved bands for rq_topo.py. heads: each approved head\'s own band, by its name '
                           '(a head checked under that name is judged on it: no worse than John passed it); kinds: '
                           'for any other head, the worst of every approved head of its kind (source: a quad mesh '
                           'as built; game: the decimated triangles the world loads). Counts are as they are, '
                           'measures with a 25% margin (rq_topo.calibrate). Re-run rq_topo.py --calibrate on the '
                           'approved topo.json files (listed under from) when John approves a face.',
                   'from': [os.path.basename(os.path.dirname(os.path.abspath(p))) + '/' + os.path.basename(p)
                            for p in srcs],
                   'kinds': kinds, 'heads': heads,
                   'approved': {n: {'kind': S['kind'], 'summary': S['summary'], 'clean': S['clean']}
                                for n, S in approved.items()}},
                  open(dst, 'w'), indent=1)
        print('wrote', dst, flush=True)
    elif '--glb' in a or '--blend' in a:
        import bpy
        out = opt('--out', '.')
        os.makedirs(out, exist_ok=True)
        if '--glb' in a:
            bpy.ops.wm.read_factory_settings(use_empty=True)
            bpy.ops.import_scene.gltf(filepath=opt('--glb'))
        else:
            bpy.ops.wm.open_mainfile(filepath=opt('--blend'))
        objs, head = head_objects()
        if opt('--object'):
            objs = [bpy.data.objects[opt('--object')]]
            head = objs[0].parent if head is None else head
        run(out, objs, head, title=opt('--title', ''), label=opt('--label'))
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
