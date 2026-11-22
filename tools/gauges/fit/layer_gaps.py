"""The layer gap check: how far each garment stands off the one worn under it, in every pose of the pose strip, and
where a lower layer comes through (a negative gap: show-through). Report-only: it never moves a garment or fails a
render.

The stack and the ease each layer should have come from the outfit's spec (riding_habit.json, ease.py derives it).
The check reads the stack along the skin's normals: from each sampled point of the body a ray goes out along the
skin's normal and meets every layer in turn. A layer's inner face is its first hit, its outer face the next hit of
the same layer within a few of its thicknesses (a sheet with no thickness has one face). At each point

    gap(L) = inner(L) - outer(the outermost lower layer met within reach)

so a corset coming through the bodice gives a negative gap at that point. The ray stops where it enters the body
again (an arm beside the trunk), so a layer on another limb is never read as lying under this one.

    import layer_gaps, ease
    rep = layer_gaps.check(ease.load(), poses=('rest', 'arms-up', 'step'), apply=rq_views_apply, out=DIR)

Writes DIR/layer_gaps.json: per pose, per pair (layer over layer beneath), per region (and front, side, back on the
trunk): how many points, the smallest gap and the 5th percentile in mm, the share of points with a negative gap,
the share under the spec's ease, and the worst point (where it is on the rest body). 'show_through' lists every
region with a negative gap past the tolerance, worst first."""
import json
import math
import os
import time

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

import ease

# a negative gap smaller than this (in mm) is the meshes' own faceting, not show-through
TOLERANCE_MM = 0.3
# a lower layer met further out than this past the layer's outer face is not under it (another fold, another limb)
REACH_U = 0.6
# how far out along the normal a ray looks for layers, in units (a skirt's hem stands far off the leg)
RAY_U = 8.0
# every STEP-th vertex of the skin is sampled
STEP = int(os.environ.get("RQ_GAP_STEP", "5"))
# the skin's parts that count as one surface: a hit is read only over the limb its ray started from
GROUPS = {'neck': 'trunk', 'trunk': 'trunk', 'arm': 'arm', 'forearm': 'arm', 'hand': 'arm',
          'leg': 'leg', 'shin': 'leg', 'foot': 'leg'}


def _layer_objects(spec):
    """[(layer, [objects])] for the spec's layers present in the scene, in wearing order."""
    out = []
    for L in spec['layers']:
        obs = []
        for o in bpy.data.objects:
            if o.type != 'MESH' or o.hide_render:
                continue
            n = o.name
            if not any(n == p or n.startswith(p + '.') or n.startswith(p + ' ') or n.startswith(p) for p in L['objects']):
                continue
            low = n.lower()
            if any(x in low for x in L.get('exclude', ())):
                continue
            obs.append(o)
        if obs:
            out.append((L, obs))
    return out


def _tree(objs, dg):
    pts, tris = [], []
    for o in objs:
        e = o.evaluated_get(dg)
        me = e.to_mesh()
        me.calc_loop_triangles()
        mw = e.matrix_world
        n0 = len(pts)
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get('co', co)
        co = co.reshape(-1, 3)
        R = np.array(mw.to_3x3())
        co = co @ R.T + np.array(mw.translation)
        pts.extend(map(tuple, co))
        lt = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
        me.loop_triangles.foreach_get('vertices', lt)
        tris.extend(map(tuple, (lt.reshape(-1, 3) + n0)))
        e.to_mesh_clear()
    return BVHTree.FromPolygons(pts, tris, all_triangles=True) if tris else None


def _hits(tree, o, d, far, thick):
    """(inner, outer) distances along the ray, or None."""
    h = tree.ray_cast(o, d, far)
    if h[0] is None:
        return None
    t1 = h[3]
    o2 = o + d * (t1 + 1e-4)
    h2 = tree.ray_cast(o2, d, 3.0 * thick + 0.05)
    t2 = t1 + 1e-4 + h2[3] if h2[0] is not None else t1
    return t1, t2


def _neighbours(rest, radius):
    """For each sample, the samples within `radius` on the rest body (a KD tree): the span the upper cloth bridges."""
    from mathutils.kdtree import KDTree
    kd = KDTree(len(rest))
    for i, p in enumerate(rest):
        kd.insert(p, i)
    kd.balance()
    return [np.array([j for _, j, _ in kd.find_range(p, radius)]) for p in rest]


def _relief(nbr, inner, below, m):
    """The relief of the layer beneath and of the upper layer's inner face at the points m: each one's stand-off from
    the skin less its mean over the span the upper cloth bridges (nan where there are too few neighbours)."""
    ix = np.nonzero(m)[0]
    hb, hu = np.full(len(ix), np.nan), np.full(len(ix), np.nan)
    for k, i in enumerate(ix):
        nb = nbr[i]
        nb = nb[~np.isnan(below[nb]) & ~np.isnan(inner[nb])]
        if len(nb) >= 4:
            hb[k] = below[i] - below[nb].mean()
            hu[k] = inner[i] - inner[nb].mean()
    return hb, hu


def _print_through(nbr, inner, below, m, cm):
    """How much of the relief of the layer beneath (its stand-off from the skin, less its mean over the span the
    upper cloth bridges) comes through into the upper layer's inner face: the slope of the one on the other, 0 when
    the upper cloth bridges it, 1 when it copies every ridge and step. relief_mm is the relief's spread (5th to
    95th percentile) where it is under this layer."""
    ix = np.nonzero(m)[0]
    if len(ix) < 8:
        return None
    hb, hu = [], []
    for i in ix:
        nb = nbr[i]
        nb = nb[~np.isnan(below[nb]) & ~np.isnan(inner[nb])]
        if len(nb) < 4:
            continue
        hb.append(below[i] - below[nb].mean())
        hu.append(inner[i] - inner[nb].mean())
    if len(hb) < 8:
        return None
    hb, hu = np.array(hb), np.array(hu)
    var = float((hb * hb).mean())
    if var < 1e-10:
        return None
    return {'relief_mm': round(float(np.percentile(hb, 95) - np.percentile(hb, 5)) * cm * 10.0, 2),
            'print_through': round(float((hb * hu).mean()) / var, 2)}


def _sector(th):
    a = abs(((th + math.pi) % (2 * math.pi)) - math.pi)
    return 'front' if a < math.pi / 4 else ('back' if a > 3 * math.pi / 4 else 'side')


def check(spec, poses=('rest',), apply=None, out=None, log=print):
    t0 = time.time()
    body = bpy.data.objects.get('Skin body')
    layers = _layer_objects(spec)
    rep = {'layers': [L['name'] for L, _ in layers], 'objects': {L['name']: [o.name for o in obs] for L, obs in layers},
           'tolerance_mm': TOLERANCE_MM, 'poses': {}, 'show_through': [], 'print_through': []}
    if body is None or len(layers) < 2:
        rep['skipped'] = 'fewer than two layers of the spec in the scene' if body is not None else 'no body'
        return _write(rep, out, log, t0)
    cm = spec['cm_per_unit']
    lab = ease.label_body(body)
    idx = np.arange(0, len(body.data.vertices), STEP)
    idx = idx[lab['part'][idx] != 'other']
    region, sector = lab['region'][idx], lab['theta'][idx]
    rest = lab['rest'][idx]
    # (which limb each skin face is on: a layer's hit counts only over the sample's own limb)
    limb = np.array([GROUPS.get(p, p) for p in lab['part']], dtype=object)
    own = limb[idx]
    fv = np.empty(len(body.data.polygons), dtype=np.int64)
    body.data.polygons.foreach_get('loop_start', fv)
    lv = np.empty(len(body.data.loops), dtype=np.int64)
    body.data.loops.foreach_get('vertex_index', lv)
    side = lab['side']
    poly_part = np.array([f'{limb[v]}{side[v]}' if limb[v] != 'trunk' else 'trunk' for v in lv[fv]], dtype=object)
    own = np.array([f'{own[k]}{side[idx[k]]}' if own[k] != 'trunk' else 'trunk' for k in range(len(idx))], dtype=object)
    sewn = {(L['name'], s_['to']): set(s_['regions']) for L, _ in layers for s_ in L.get('sewn_to', ())}
    rep['seams'] = []
    need = {}                       # (layer, region) -> the spec's stand-off over the layer beneath, in units
    for name, regs in (spec.get('derived', {}).get('layers') or {}).items():
        for r, v in regs.items():
            if isinstance(v, dict) and 'radial_u' in v:
                need[(name, r)] = v['radial_u']
    thick = {L['name']: spec['fabrics'][L['fabric']]['thick_u'] for L, _ in layers}
    # (a pair is read only where both layers are worn: the regions each one covers, so a ray from the arm that
    # crosses the bodice's side is not read as the bodice over the arm)
    covered = [np.isin(region, list(L['covers']) + list(L.get('also', ()))) for L, _ in layers]
    nbr = None
    maps = {}
    for pose in poses:
        if apply is not None:
            apply(pose)
        dg = bpy.context.evaluated_depsgraph_get()
        be = body.evaluated_get(dg)
        me = be.to_mesh()
        n = len(me.vertices)
        if n != len(body.data.vertices):
            be.to_mesh_clear()
            rep['skipped'] = 'the evaluated body has a different vertex count (a modifier that adds points)'
            return _write(rep, out, log, t0)
        co = np.empty(n * 3)
        me.vertices.foreach_get('co', co)
        nr = np.empty(n * 3)
        me.vertices.foreach_get('normal', nr)
        be.to_mesh_clear()
        mw = be.matrix_world
        R = np.array(mw.to_3x3())
        P = (co.reshape(-1, 3) @ R.T + np.array(mw.translation))[idx]
        N = nr.reshape(-1, 3) @ R.T
        N = N[idx] / np.maximum(np.linalg.norm(N[idx], axis=1, keepdims=True), 1e-9)
        btree = BVHTree.FromObject(body, dg)
        trees = [(L['name'], _tree(obs, dg)) for L, obs in layers]
        # stack[k][i] = (inner, outer) of layer k at sample i, or None
        H = np.full((len(trees), len(idx), 2), np.nan)
        for i in range(len(idx)):
            p, d = Vector(P[i]), Vector(N[i])
            o = p - d * 0.03
            hb = btree.ray_cast(p + d * 0.08, d, RAY_U)
            far = (hb[3] + 0.08 + 0.03) if hb[0] is not None else RAY_U
            for k, (name, tr) in enumerate(trees):
                if tr is None:
                    continue
                h = _hits(tr, o, d, far, thick[name])
                if h is None:
                    continue
                # (the hit has to lie over this skin, not over another part the ray passed: an arm's ray meets the
                # bodice's side, which is nearer the trunk's skin than the arm's)
                q = btree.find_nearest(o + d * h[0])
                if q[0] is not None and q[2] is not None and q[2] < len(poly_part) and poly_part[q[2]] != own[i]:
                    continue
                H[k, i] = h
        H -= 0.03          # (distances from the skin)
        if nbr is None:
            nbr = _neighbours(rest, max(spec['fabrics'][L['fabric']].get('bridge_u', 0.5) for L, _ in layers))
        res = {}
        for k in range(1, len(trees)):
            name = trees[k][0]
            inner, outer = H[k, :, 0].copy(), H[k, :, 1]
            inner[~covered[k]] = np.nan
            below = np.full(len(idx), np.nan)
            who = np.full(len(idx), -1)
            for j in range(k):
                oj = H[j, :, 1].copy()
                oj[~covered[j]] = np.nan
                ok = ~np.isnan(oj) & ~np.isnan(inner) & (oj < outer + REACH_U) & (np.isnan(below) | (oj > below))
                below[ok], who[ok] = oj[ok], j
            gap = inner - below
            have = ~np.isnan(gap)
            # (for the relief, where nothing is under the layer the skin is: a lower layer's edge is a step in what is
            # beneath, and the step the upper cloth makes over it is print-through as much as a bone channel is)
            bfill = np.where(np.isnan(below) & ~np.isnan(inner), 0.0, below)
            worn = ~np.isnan(inner)
            if not have.any():
                continue
            for j in sorted(set(who[have].tolist())):
                pair = f'{name} over {trees[j][0]}'
                sel = have & (who == j)
                rows = {}
                for r in sorted(set(region[sel].tolist())):
                    for sec in ('all', 'front', 'side', 'back'):
                        m = sel & (region == r)
                        if sec != 'all':
                            if not ease.REGIONS_TRUNK.get(r):
                                continue
                            m &= np.array([_sector(t) == sec for t in sector])
                        if m.sum() < 3:
                            continue
                        g = gap[m] * cm * 10.0           # mm
                        w = int(np.argmin(g))
                        want = need.get((name, r))
                        row = {'n': int(m.sum()), 'min_mm': round(float(g.min()), 2),
                               'p5_mm': round(float(np.percentile(g, 5)), 2),
                               'median_mm': round(float(np.median(g)), 2),
                               'negative_pct': round(100.0 * float((g < -TOLERANCE_MM).mean()), 1),
                               'worst_at': [round(float(c), 2) for c in rest[m][w]]}
                        pt = _print_through(nbr, inner, bfill, m, cm)
                        if pt:
                            row.update(pt)
                        if want is not None:
                            row['spec_mm'] = round(want * cm * 10.0, 2)
                            row['under_spec_pct'] = round(100.0 * float((g < want * cm * 10.0).mean()), 1)
                        rows[r if sec == 'all' else f'{r}/{sec}'] = row
                        bridge = spec['fabrics'][layers[k][0]['fabric']].get('bridge_u', 0.5) * cm * 10.0
                        where = (sec != 'all') if ease.REGIONS_TRUNK.get(r) else (sec == 'all')
                        if where and row.get('relief_mm', 0) > 2 * TOLERANCE_MM and row.get('print_through', 0) > 0.5:
                            rep['print_through'].append(dict(pose=pose, pair=pair, region=r if sec == 'all' else f'{r}/{sec}',
                                                             relief_mm=row['relief_mm'],
                                                             print_through=row['print_through'],
                                                             bridges_mm=round(bridge, 1)))
                        if r in sewn.get((name, trees[j][0]), ()):
                            if sec == 'all' and row['negative_pct'] > 0:
                                rep['seams'].append(dict(pose=pose, pair=pair, region=r, negative_pct=row['negative_pct'],
                                                         min_mm=row['min_mm'], note='sewn together here (the spec)'))
                            rows[r if sec == 'all' else f'{r}/{sec}'] = dict(row, sewn=True)
                            continue
                        if sec != 'all' and row['negative_pct'] > 0 or (sec == 'all' and not ease.REGIONS_TRUNK.get(r)
                                                                         and row['negative_pct'] > 0):
                            rep['show_through'].append(dict(pose=pose, pair=pair, region=(r if sec == 'all' else f'{r}/{sec}'),
                                                            negative_pct=row['negative_pct'], min_mm=row['min_mm'],
                                                            worst_at=row['worst_at']))
                res[pair] = rows
                tm = worn & (sel | np.isnan(below)) & np.isin(region, list(ease.REGIONS_TRUNK) + ['seat'])
                if tm.sum() > 20:
                    hb, hu = _relief(nbr, inner, bfill, tm)
                    maps.setdefault(pair, {})[pose] = (sector[tm], rest[tm, 2], (inner - bfill)[tm] * cm * 10.0,
                                                       hb * cm * 10.0, hu * cm * 10.0)
        rep['poses'][pose] = res
        log(f'layer gaps: {pose}: ' + '; '.join(
            f"{pr} min {min(v['min_mm'] for v in rows.values()):.1f} mm" for pr, rows in res.items() if rows))
    if apply is not None:
        apply('rest')
    if out:
        try:
            rep['maps'] = _maps(maps, need, cm, out)
        except ImportError:
            pass
    rep['show_through'].sort(key=lambda s: (-s['negative_pct'], s['min_mm']))
    rep['print_through'].sort(key=lambda s: -s['relief_mm'] * s['print_through'])
    return _write(rep, out, log, t0)


def _maps(maps, need, cm, out, px=5):
    """A picture per pair, the trunk unrolled (her back at the edges, the front in the middle, up is up), in the pose
    with the smallest gap. Three panels: the gap (red where the layer beneath comes through, amber 0 to 1 mm, pale to
    grey as it grows to 6 mm), the relief of the layer beneath and the relief of the upper layer's inner face (blue
    hollow, orange proud, +-1.5 mm, over the span the upper cloth bridges). Where the last two panels match, the
    layer beneath prints through. DIR/gaps-<pair>.png. Needs Pillow."""
    from PIL import Image, ImageDraw
    files = []

    def div(v):
        if np.isnan(v):
            return None
        k = max(-1.0, min(1.0, v / 1.5))
        return (int(130 + 110 * k), int(130 + 40 * k - 40 * abs(k)), int(130 - 110 * k)) if k > 0 else \
            (int(130 + 100 * k), int(130 + 20 * k), int(130 - 110 * k))

    def gapc(g):
        if g < -TOLERANCE_MM:
            return (220, 40, 40)
        if g < 1.0:
            return (235, 170, 40)
        k = min(1.0, (g - 1.0) / 5.0)
        return (int(240 - 110 * k), int(232 - 100 * k), int(210 - 70 * k))
    for pair, by_pose in maps.items():
        pose = min(by_pose, key=lambda p: float(np.min(by_pose[p][2])))
        th, z, g, hb, hu = by_pose[pose]
        W, z0, z1 = 300, float(z.min()) - 0.5, float(z.max()) + 0.5
        Hh = int((z1 - z0) / 22.0 * W * 1.6) + 20
        im = Image.new('RGB', (3 * W + 20, Hh + 34), (40, 42, 48))
        dr = ImageDraw.Draw(im)
        lower, upper = pair.split(' over ')[1], pair.split(' over ')[0]
        for n, (title, col, vals) in enumerate((('gap', gapc, g), (f'{lower} relief', div, hb), (f'{upper} relief', div, hu))):
            x0 = n * (W + 10)
            dr.text((x0 + 4, 18), title, fill=(220, 220, 220))
            for a, zz, v in zip(th, z, vals):
                c = col(v)
                if c is None:
                    continue
                x = x0 + int((a + math.pi) / (2 * math.pi) * (W - 1))
                y = int((z1 - zz) / (z1 - z0) * (Hh - 1)) + 32
                dr.rectangle([x - px // 2, y - px // 2, x + px // 2, y + px // 2], fill=c)
        dr.text((4, 3), f'{pair}, {pose} (back | front | back). gap: red under 0, amber 0-1 mm, then pale to grey to 6 mm. '
                        f'relief: blue hollow, orange proud, +-1.5 mm', fill=(220, 220, 220))
        f = os.path.join(out, 'gaps-' + pair.replace(' ', '-') + '.png')
        im.save(f)
        files.append(os.path.basename(f))
    return files


def _worst(items):
    """One entry per pair and region (its worst pose), in the order given."""
    seen, out = set(), []
    for it in items:
        k = (it['pair'], it['region'])
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def _write(rep, out, log, t0):
    rep['seconds'] = round(time.time() - t0, 1)
    if out:
        with open(os.path.join(out, 'layer_gaps.json'), 'w') as f:
            json.dump(rep, f, indent=1)
    st = _worst(rep.get('show_through') or [])
    pt = _worst(rep.get('print_through') or [])
    if pt:
        log('layer gaps: relief printing through: ' + '; '.join(
            f"{s['pose']} {s['pair']} {s['region']} {s['relief_mm']} mm x{s['print_through']}" for s in pt[:6]), flush=True)
    log('layer gaps: ' + (rep.get('skipped') or (f'{len(st)} show-through regions' + (
        ': ' + '; '.join(f"{s['pose']} {s['pair']} {s['region']} {s['negative_pct']}% (min {s['min_mm']} mm)"
                         for s in st[:6]) if st else ''))), flush=True)
    return rep
