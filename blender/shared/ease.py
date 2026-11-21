"""Garment ease, set from the measured body instead of by feel (John, 2026-11-19).

The outfit's spec (riding_habit.json beside this file) lists every layer, under to over, with its fabric and how it
is cut. This module measures the body the outfit is built on, at rest and in the pose strip's extreme poses (arms up,
arms forward, hard elbows, a deep step, a crouch, a twist), and derives from that the ease each layer needs over the
layer beneath, region by region:

    ease over beneath = standing ease (the cut, fits[...])
                      + movement ease: the most the body's girth there grows in any pose, less what the cloth gives
                        (stretch * girth); none where a held layer beneath (the boned corset) keeps the body still
                      + turn of cloth: pi * own thickness * turn (the inner face of a thick cloth is shorter than its
                        outer face, and the cloth needs that back to close round the figure)
    ease over body    = the thickness of every layer beneath (2 pi t each; a layer pressed by a held one over it
                        counts only its thickness, times its gathers) + each lower layer's own ease where it is not
                        pressed + this layer's ease over beneath

`radial_u` and `radial_w` are the ease over beneath as a stand-off from the layer beneath (ease / 2 pi), in scene
units and in w (the neck's half-width, the vest's and the bodice's draft unit): what a draft that offsets a shell over
the layer beneath sets as its ease.

    import ease
    spec = ease.load()
    d = ease.derive(spec, apply=pose_fn)      # in a built scene; pose_fn(name) puts the rig in that pose
    ease.save(spec, d)                        # into riding_habit.json's "derived" block

Girths are a tape round a band of skin: the convex hull of the band's points in the band's own plane (a tape
bridges the hollows). The band is the same skin in every pose (the same vertices), so a change is the body's own.
Everything here reads the base figure's landmarks (female.py) and the body's own vertex groups, so another body
gives other numbers with nothing changed."""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, 'riding_habit.json')

REGIONS_TRUNK = {'neck_base': True, 'bust': True, 'under_bust': True, 'waist': True, 'high_hip': True, 'hip': True,
                 'across_back': True}
# the skin's parts, from its dominant vertex group
PARTS = {'Pelvis': 'trunk', 'Chest': 'trunk', 'Cervical': 'trunk', 'Midneck': 'neck', 'Neck': 'neck',
         'UpperArm': 'arm', 'Forearm': 'forearm', 'Hand': 'hand', 'Thigh': 'leg', 'Shin': 'shin', 'Foot': 'foot'}
# a limb part's segment: (from joint, to joint) objects of the rig
SEGMENTS = {'arm': ('Shoulder', 'Elbow'), 'forearm': ('Elbow', 'Wrist'), 'leg': ('Hip', 'Knee'), 'shin': ('Knee', 'Ankle')}
BAND_U = 0.35            # a band's half-height (trunk) or half-length (limb), in units


def load(path=SPEC):
    with open(path) as f:
        return json.load(f)


def save(spec, derived, path=SPEC):
    spec = dict(spec, derived=derived)
    with open(path, 'w') as f:
        json.dump(spec, f, indent=2)
        f.write('\n')
    return path


# ---------------------------------------------------------------- the body
def _evaluated(body):
    import bpy
    dg = bpy.context.evaluated_depsgraph_get()
    be = body.evaluated_get(dg)
    me = be.to_mesh()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', co)
    be.to_mesh_clear()
    mw = be.matrix_world
    return co.reshape(-1, 3) @ np.array(mw.to_3x3()).T + np.array(mw.translation)


def _joint(name):
    import bpy
    o = bpy.data.objects.get(name)
    return np.array(o.matrix_world.translation) if o is not None else None


def landmarks(body, P=None):
    """Heights of the trunk's girths, measured on this body (rest pose): the bust's apex is the front-most point of
    the chest, the hip the widest section below the waist, the waist the narrowest between them."""
    import female
    P = _evaluated(body) if P is None else P
    trunk = np.abs(P[:, 0]) < female.SH_X - 1.0
    zs = np.arange(female.HIP_Z - 4.0, female.SH_Z, 0.25)
    front, width = [], []
    for z in zs:
        m = trunk & (np.abs(P[:, 2] - z) < 0.2)
        front.append(-P[m, 1].min() if m.any() else np.nan)
        width.append(np.ptp(P[m, 0]) if m.sum() > 2 else np.nan)
    front, width = np.array(front), np.array(width)
    hi = zs > female.CHEST_Z
    bust = float(zs[hi][np.nanargmax(front[hi])])
    try:                                   # (the corset's waist line, when the habit's numbers are on this branch)
        import habit
        waist = float(habit.measure(body).z_waist)
    except Exception:
        mid = (zs > female.HIP_Z) & (zs < bust - 1.0)
        waist = float(zs[mid][np.nanargmin(width[mid])])
    lo = zs < waist - 1.0
    hip = float(zs[lo][np.nanargmax(width[lo])])
    try:
        neck = [g.index for g in body.vertex_groups if g.name in ('Neck', 'Midneck', 'Cervical')]
        nz = [P[v.index, 2] for v in body.data.vertices if any(g.group in neck and g.weight > 0.5 for g in v.groups)]
        neck_base = float(np.percentile(nz, 5)) + BAND_U if nz else float(female.NECK_Z - 1.0)
    except Exception:
        neck_base = float(female.NECK_Z - 1.0)
    return {'NECK_BASE': neck_base, 'BUST': bust, 'UNDER_BUST': waist + 0.55 * (bust - waist),
            'WAIST': waist, 'HIGH_HIP': waist - 0.4 * (waist - hip), 'HIP': hip,
            'ACROSS_BACK': 0.5 * (bust + female.SH_Z)}


def label_body(body):
    """Each skin vertex's part (trunk, neck, arm ...), side, region (the nearest of the spec's regions), angle round the
    trunk (0 at the front, as habit.py's theta) and rest position. Call in the rest pose."""
    P = _evaluated(body)
    n = len(P)
    names = {g.index: g.name for g in body.vertex_groups}
    best = np.zeros(n)
    group = np.full(n, -1)
    for v in body.data.vertices:
        for g in v.groups:
            if g.weight > best[v.index] and names.get(g.group, '').split('.')[0] in PARTS:
                best[v.index], group[v.index] = g.weight, g.group
    part = np.array(['other'] * n, dtype=object)
    side = np.array([''] * n, dtype=object)
    for gi, nm in names.items():
        base, _, sd = nm.partition('.')
        if base in PARTS:
            part[group == gi] = PARTS[base]
            side[group == gi] = sd
    L = landmarks(body, P)
    spec = load()
    region = np.array([''] * n, dtype=object)
    trunk_regs = [(r, L[d['z']]) for r, d in spec['regions'].items()
                  if not r.startswith('_') and d['part'] == 'trunk' and d['kind'] == 'girth']
    m = part == 'trunk'
    zr = np.array([z for _, z in trunk_regs])
    k = np.argmin(np.abs(P[m, 2][:, None] - zr[None, :]), axis=1)
    region[m] = np.array([r for r, _ in trunk_regs], dtype=object)[k]
    region[m & (P[:, 2] > L['NECK_BASE'] + 1.0)] = 'neck'
    region[m & (P[:, 2] < L['HIP'] - 2.0)] = 'seat'
    region[part == 'neck'] = 'neck_base'
    for pt, (a, b) in SEGMENTS.items():
        regs = [(r, d['at']) for r, d in spec['regions'].items()
                if not r.startswith('_') and d['part'] == pt and d['kind'] == 'girth']
        for sd in ('L', 'R'):
            m = (part == pt) & (side == sd)
            A, B = _joint(f'{a}.{sd}'), _joint(f'{b}.{sd}')
            if A is None or B is None or not m.any() or not regs:
                continue
            t = ((P[m] - A) @ (B - A)) / max(float((B - A) @ (B - A)), 1e-9)
            at = np.array([x for _, x in regs])
            region[m] = np.array([r for r, _ in regs], dtype=object)[np.argmin(np.abs(t[:, None] - at[None, :]), axis=1)]
    for pt in ('hand', 'foot'):
        region[part == pt] = pt
    cy = float(np.median(P[part == 'trunk', 1])) if (part == 'trunk').any() else 0.0
    theta = np.arctan2(-P[:, 0], -(P[:, 1] - cy))
    return {'part': part, 'side': side, 'region': region, 'theta': theta, 'rest': P, 'landmarks': L, 'cy': cy}


def _hull(xy):
    """Convex hull of 2-D points (Andrew's monotone chain): the indices, counter-clockwise."""
    o = np.lexsort((xy[:, 1], xy[:, 0]))
    pts = xy[o]

    def half(seq):
        h = []
        for i in seq:
            while len(h) >= 2:
                a, b = pts[h[-2]], pts[h[-1]]
                c = pts[i]
                if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 0:
                    h.pop()
                else:
                    break
            h.append(i)
        return h
    lower, upper = half(range(len(pts))), half(range(len(pts) - 1, -1, -1))
    return o[np.array(lower[:-1] + upper[:-1])]


def _plane(X, normal=None):
    c = X.mean(0)
    if normal is None:
        _, _, vt = np.linalg.svd(X - c, full_matrices=False)
        normal = vt[2]
    normal = normal / np.linalg.norm(normal)
    u = np.cross(normal, [0.0, 1.0, 0.0] if abs(normal[1]) < 0.9 else [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return np.stack([(X - c) @ u, (X - c) @ v], 1)


def _girth(X, normal=None, keep=None):
    """A tape round the points X (n x 3): the convex hull's perimeter in the band's plane. With keep (a mask), only
    the hull's edges between kept points (the back's width)."""
    if len(X) < 4:
        return float('nan')
    xy = _plane(X, normal)
    h = _hull(xy)
    a, b = h, np.roll(h, -1)
    seg = np.linalg.norm(xy[a] - xy[b], axis=1)
    if keep is not None:
        seg = seg[keep[a] & keep[b]]
    return float(seg.sum())


def bands(lab, spec):
    """region -> [(name, vertex indices, kind)]: the skin each girth is taken round, at rest."""
    P, part, side, L = lab['rest'], lab['part'], lab['side'], lab['landmarks']
    out = {}
    for r, d in spec['regions'].items():
        if r.startswith('_'):
            continue
        if d['part'] in ('trunk', 'neck'):
            m = (part == d['part']) & (np.abs(P[:, 2] - L[d['z']]) < BAND_U)
            out[r] = [(r, np.nonzero(m)[0], d['kind'], None)]
            continue
        a, b = SEGMENTS[d['part']]
        per = []
        for sd in ('L', 'R'):
            A, B = _joint(f'{a}.{sd}'), _joint(f'{b}.{sd}')
            if A is None or B is None:
                continue
            seg = B - A
            ln = float(np.linalg.norm(seg))
            m = (part == d['part']) & (side == sd)
            t = ((P - A) @ seg) / (ln * ln)
            m &= np.abs(t - d['at']) * ln < BAND_U
            # (and near the limb's axis: not the skin of the trunk or the other limb the weights reach)
            ax = (P - A) - np.outer(t, seg)
            rad = np.linalg.norm(ax, axis=1)
            if m.any():
                m &= rad < 1.6 * np.median(rad[m])
            per.append((np.nonzero(m)[0], (f'{a}.{sd}', f'{b}.{sd}')))
        if d['kind'] == 'pair':
            out[r] = [(r, np.concatenate([ix for ix, _ in per]) if per else np.array([], int), 'pair', None)]
        else:
            out[r] = [(f'{r}.{sd}', ix, 'girth', ab) for ix, ab in per]
    return out


def measure(lab, spec, P):
    """region -> girth (units) on the posed skin P, the worse side for paired limbs."""
    res = {}
    cy = lab['cy']
    for r, items in bands(lab, spec).items():
        vals = []
        for _, ix, kind, ab in items:
            if len(ix) < 6:
                continue
            X = P[ix]
            if kind == 'pair':
                vals.append(_girth(X, normal=np.array([0.0, 0.0, 1.0])))
            elif kind == 'width':
                vals.append(_girth(X, normal=np.array([0.0, 0.0, 1.0]), keep=lab['rest'][ix, 1] > cy))
            elif ab is not None:            # (a limb: the tape square to the bone as posed)
                A, B = _joint(ab[0]), _joint(ab[1])
                vals.append(_girth(X, normal=B - A))
            else:
                vals.append(_girth(X, normal=np.array([0.0, 0.0, 1.0])))
        res[r] = max(vals) if vals else float('nan')
    return res


# ---------------------------------------------------------------- the ease
def derive(spec, apply=None, log=print, w=None):
    """Measure the body in every pose of spec['poses'] and derive each layer's ease, region by region."""
    import bpy
    body = bpy.data.objects['Skin body']
    if apply is not None:
        apply('rest')
    lab = label_body(body)
    cm = spec['cm_per_unit']
    girths = {}
    for pose in spec['poses']:
        if apply is not None:
            apply(pose)
        girths[pose] = measure(lab, spec, _evaluated(body))
    if apply is not None:
        apply('rest')
    rest = girths['rest']
    motion = {r: max((girths[p][r] - rest[r]) for p in girths if not math.isnan(girths[p][r])) for r in rest}
    worst = {r: max(girths, key=lambda p: girths[p][r] if not math.isnan(girths[p][r]) else -1) for r in rest}
    held = set()
    out_layers, stack = {}, {}           # stack[r]: [(layer, thickness girth, own ease, pressed)]
    names = [L['name'] for L in spec['layers']]
    for L in spec['layers']:
        fab = spec['fabrics'][L['fabric']]
        fit = spec['fits'][L['fit']]
        t = fab['thick_u']
        rows = {}
        for r in L['covers']:
            g0 = rest.get(r, float('nan'))
            if math.isnan(g0):
                continue
            below = stack.get(r, [])
            thick_below = sum(x[1] for x in below)
            ease_below = sum(x[2] for x in below if not x[3])
            if L['fit'] == 'corset':
                own = 2 * math.pi * fit['clear_u']
                row = {'note': 'the corset sets the body (habit.py): clear of the skin by clear_u'}
                mv = 0.0
            else:
                stand = fit.get(r, 0.0) / cm
                mv = 0.0 if r in held else max(0.0, motion[r] - fab.get('stretch', 0.0) * g0)
                turn = math.pi * t * fab.get('turn', 1.0)
                own = stand + mv + turn
                row = {'standing_cm': round(stand * cm, 2), 'movement_cm': round(mv * cm, 2),
                       'movement_pose': worst[r] if mv > 0 else None,
                       'body_growth_cm': round(motion[r] * cm, 2), 'stretch_credit_cm': round(fab.get('stretch', 0.0) * g0 * cm, 2),
                       'turn_cm': round(turn * cm, 2), 'held_by_corset': r in held}
            row.update({'girth_beneath_cm': round((g0 + thick_below + ease_below) * cm, 1),
                        'thickness_below_cm': round(thick_below * cm, 2),
                        'ease_over_beneath_cm': round(own * cm, 2),
                        'ease_over_body_cm': round((thick_below + ease_below + own) * cm, 2),
                        'radial_u': round(own / (2 * math.pi), 4)})
            if w:
                row['radial_w'] = round(own / (2 * math.pi) / w, 4)
            rows[r] = row
        out_layers[L['name']] = rows
        for r in L['covers']:
            if r not in rows:
                continue
            pressed_by = L.get('pressed_by')
            pressed = bool(pressed_by) and pressed_by in names and r in next(
                x for x in spec['layers'] if x['name'] == pressed_by)['covers']
            k = fit.get('pressed_layers', 1.0) if pressed else 1.0
            stack.setdefault(r, []).append((L['name'], 2 * math.pi * t * k,
                                            rows[r]['ease_over_beneath_cm'] / cm, pressed))
        held |= set(L.get('holds', ()))
    log('ease: ' + '; '.join(f'{r} {rest[r] * cm:.1f} cm (+{motion[r] * cm:.1f} in {worst[r]})' for r in rest))
    return {'_doc': 'written by ease.derive on the measured body; do not edit by hand',
            'body': {'landmarks_u': {k: round(v, 3) for k, v in lab['landmarks'].items()},
                     'girth_cm': {p: {r: round(v * cm, 2) for r, v in g.items()} for p, g in girths.items()},
                     'w_u': w},
            'layers': out_layers}
