"""views.py with the render queue's extras: crops of named problem areas and the pose test strip (John, 2026-11-19).
The queue puts this file in a job's worktree as blender/base/rq_views.py and runs it instead of views.py whenever a
request asks for --crop or --poses. It loads views.py's own code at that commit (without running it), runs views.py's
main exactly as views.py would (the build, then any --turn, --head, --walk ... asked for), and then, on the same
build:

--crop SPECS   square close-ups of problem areas, each a row over --crop-angles (default: --angles, else 3/4), from
               the same fixed camera and lights as --head, so crops of two commits line up pixel for pixel:
               DIR/crop-NAME.png (and crop-NAME-clay.png with --clay or --clay-only alone). SPECS is a comma list of:
                 NAME          a region of the body (below), measured from the base figure's own sizes at this commit
                 NAME+PAD      the region widened by PAD (0.3 is 30%)
                 box:X0:Y0:X1:Y1[@S]   a box in pixels of one tile of the turn (DIR/turn.png at --size S, default
                               360: each tile S wide and 1.75 S high), so a fault seen in a turn is cropped as it is
--crop-size N  the crops' height in pixels (default 384; a wide region such as the hem is wider than it is high)
--poses NAMES  the pose test strip: the clay figure at the joints' extremes, one tile per pose, from --pose-angle
               (default 3/4), on one fixed camera: DIR/poses.png and DIR/pose-NAME.png. NAMES is all, or some of
               rest, arms-up, arms-fwd, elbows, step, crouch, twist, and the head poses the base figure defines
               (female.HEAD_POSES: nod, look-down, look-up, tilt, turn). The head rides the chest and is held
               looking level ahead in the body poses (female.carry_head), as a real neck holds it.
--pose-crop REGIONS  with --poses: also a crop of each region (as --crop names them) in every pose, from the same
               camera as the strip: DIR/pose-crop-REGION.png, one tile per pose (default height 300, --pose-size)

Regions (centred on the figure's axis, so one crop serves every angle; both sides show):
  neck, shoulder, armhole, bust, elbow, waist, wrist (the cuff), hips (the seat), hand, knee, hem (floor length),
  ankle, face, head.
"""
import json
import math
import os
import sys
import tempfile
import time

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
VIEWS_PY = os.path.join(HERE, 'views.py')
src = open(VIEWS_PY).read()
cut = src.rstrip().rfind('\nmain()')
if cut < 0:                 # (a views.py that does not end in main(): run it as it is, with no extras)
    import runpy
    print('rq_views: views.py does not end in main(): running it as it is', flush=True)
    runpy.run_path(VIEWS_PY, run_name='__main__')
    sys.exit(0)
V = {'__file__': VIEWS_PY, '__name__': 'views_lib'}
exec(compile(src[:cut], VIEWS_PY, 'exec'), V)
T_LOADED = time.time()
bpy, female, opt, argv = V['bpy'], V['female'], V['opt'], V['argv']

POSES = ('rest', 'arms-up', 'arms-fwd', 'elbows', 'step', 'crouch', 'twist') + tuple(getattr(female, 'HEAD_POSES', {}))
ALL_VIEWS = [('front', 0), ('3/4', 45), ('profile', 90), ('back 3/4', 135), ('back', 180)]


def regions():
    """name -> (centre z, height[, width / height]): from the base figure's sizes, so they follow the body at this
    commit."""
    f = female
    sh, hip, ank, up, fore = f.SH_Z, f.HIP_Z, f.ANKLE_Z, f.UPPER, f.FORE
    knee = hip - f.THIGH
    wrist = sh - up - fore
    return {
        'head': (getattr(f, 'HEAD_Z', 55.5) + 0.5, 17.0), 'face': (getattr(f, 'HEAD_Z', 55.5) + 0.5, 10.0),
        'neck': (f.NECK_Z, 9.0), 'shoulder': (sh - 0.5, 2 * f.SH_X + 6.0), 'armhole': (sh - 3.0, 2 * f.SH_X + 5.0),
        'bust': (f.CHEST_Z + 2.0, 2 * f.SH_X + 4.0), 'elbow': (sh - up, 2 * f.SH_X + 6.0),
        'waist': (0.5 * (f.CHEST_Z + hip) + 1.0, 14.0), 'wrist': (wrist + 1.0, 2 * f.SH_X + 7.0),
        'hand': (wrist - 2.5, 2 * f.SH_X + 8.0), 'hips': (hip, 16.0, 1.4), 'knee': (knee, 12.0, 1.6),
        # (a floor-length hem is far wider than it is deep: a wide, low strip from just above the floor)
        'hem': (ank + 6.0, 20.0, 2.8), 'ankle': (ank, 10.0, 1.6),
    }


def crop_frames(specs):
    """[(stem, centre, height)] for --crop."""
    reg, out = regions(), []
    for spec in [s.strip() for s in specs.split(',') if s.strip()]:
        if spec.startswith('box:'):
            body, _, at = spec[4:].partition('@')
            x0, y0, x1, y1 = (float(v) for v in body.split(':'))
            S = float(at or 360)
            W, H = S, 1.75 * S
            k = 68.0 / H                       # (the turn: ortho height 68 about z 32.5)
            cx, cz = ((x0 + x1) / 2 - W / 2) * k, 32.5 + (H / 2 - (y0 + y1) / 2) * k
            bh = max(2.0, abs(y1 - y0) * k * 1.15)
            out.append((f'box-{int(x0)}-{int(y0)}-{int(x1)}-{int(y1)}', (cx, 0, cz), bh,
                        max(0.4, min(4.0, abs(x1 - x0) / max(1.0, abs(y1 - y0))))))
            continue
        name, _, pad = spec.partition('+')
        if name not in reg:
            raise SystemExit(f'rq_views: --crop {name}: no such region (have {", ".join(sorted(reg))}, or box:...)')
        z, h, aspect = (reg[name] + (1.0,))[:3]
        out.append((name + (f'-p{pad}' if pad else ''), (0, 0, z), h * (1 + float(pad or 0)), aspect))
    return out


def angles(flag, default):
    want = [a.strip().replace(' ', '') for a in (opt(flag) or opt('--angles') or default).split(',') if a.strip()]
    return [v for v in ALL_VIEWS if v[0].replace(' ', '') in want] or [ALL_VIEWS[1]]


def leg(side, th, kb, pitch=0.0):
    ob = bpy.data.objects
    hip, kn, an = ob[f'Hip.{side}'], ob[f'Knee.{side}'], ob[f'Ankle.{side}']
    hip.rotation_euler = (-th, 0, 0)
    kn.rotation_euler = (kb, 0, 0)
    an.rotation_euler = female.foot_level(tuple(ob['Pelvis'].rotation_euler), (-th, 0, 0), (kb, 0, 0), pitch)


def drop(*legs):
    """How far the hips come down for the lower leg to keep its foot on the floor."""
    T, S = female.THIGH, female.SHIN
    return T + S - max(T * math.cos(th) + S * math.cos(th - kb) for th, kb in legs)


def apply(name):
    ob = bpy.data.objects
    female.pose(0)
    arms = {'R': 1, 'L': -1}
    if name == 'arms-up':
        for side, s in arms.items():
            ob[f'Shoulder.{side}'].rotation_euler = (0, s * 1.35, 0)
            ob[f'Elbow.{side}'].rotation_euler = (-0.15, 0, 0)
    elif name == 'arms-fwd':
        for side, s in arms.items():
            ob[f'Shoulder.{side}'].rotation_euler = (-1.45, s * female.ARM_OUT, 0)
            ob[f'Elbow.{side}'].rotation_euler = (-0.1, 0, 0)
    elif name == 'elbows':
        for side, s in arms.items():
            ob[f'Shoulder.{side}'].rotation_euler = (-0.45, s * (female.ARM_OUT + 0.1), 0)
            ob[f'Elbow.{side}'].rotation_euler = (-2.1, 0, 0)
    elif name == 'step':
        r, l_ = (0.75, 0.25), (-0.45, 0.15)
        ob['Body'].location = (0, 0, -drop(r, l_))
        leg('R', *r); leg('L', *l_, pitch=-0.25)
    elif name == 'crouch':
        th, kb = 1.35, 2.1
        ob['Body'].location = (0, 0, -drop((th, kb)))
        ob['Chest'].rotation_euler = (0.45, 0, 0)
        for side, s in arms.items():
            leg(side, th, kb)
            ob[f'Shoulder.{side}'].rotation_euler = (-0.7, s * female.ARM_OUT, 0)
            ob[f'Elbow.{side}'].rotation_euler = (-0.6, 0, 0)
    elif name == 'twist':
        ob['Pelvis'].rotation_euler = (0, 0, -0.15)
        ob['Chest'].rotation_euler = (0, 0, 0.55)
    elif name != 'rest' and name not in getattr(female, 'HEAD_POSES', {}):
        raise SystemExit(f'rq_views: no pose {name!r} (have {", ".join(POSES)})')
    # (the head on the neck: a head pose, or held level ahead on whatever the chest is doing)
    if name in getattr(female, 'HEAD_POSES', {}):
        female.head_pose(name)
    elif hasattr(female, 'carry_head'):
        female.carry_head()
    bpy.context.view_layer.update()


def crops(out):
    frames = crop_frames(opt('--crop'))
    size = int(opt('--crop-size', 384))
    cam = V['look_3d'](int(opt('--samples', 32)))
    keep = V['VIEWS']
    V['VIEWS'] = angles('--crop-angles', '3/4')
    looks = ('-clay',) if '--clay-only' in argv else ('', '-clay') if '--clay' in argv else ('',)
    saved = {}
    with tempfile.TemporaryDirectory() as tmp:
        for look in looks:
            V['clay'](look == '-clay', saved)
            for stem, centre, h, aspect in frames:
                V['row'](cam, os.path.join(out, f'crop-{stem}{look}.png'), tmp, (int(size * aspect), size), centre, h)
                print(f'rq crop {stem}{look}: centre z {centre[2]:.1f}, height {h:.1f}', flush=True)
        V['clay'](False, saved)
    V['VIEWS'] = keep


def poses(out):
    from PIL import Image, ImageDraw
    sc = bpy.context.scene
    for o in sc.objects:      # (the sheet's columns are keyed on the timeline: pose by hand instead, as the walk does)
        if o.animation_data:
            o.animation_data_clear()
    names = list(POSES) if opt('--poses') in (None, 'all') else [p for p in opt('--poses').split(',') if p]
    size = int(opt('--pose-size', 300))
    cam = V['look_3d'](int(opt('--samples', 32)))
    keep = V['VIEWS']
    V['VIEWS'] = angles('--pose-angle', '3/4')[:1]
    saved = {}
    V['clay'](True, saved)
    ims = []
    frames = crop_frames(opt('--pose-crop')) if opt('--pose-crop') else []
    cims = {f[0]: [] for f in frames}
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            apply(name)
            p = os.path.join(out, f'pose-{name}.png')
            # one fixed camera and framing for every pose: tall enough for the raised arms, low enough for the crouch
            V['row'](cam, p, tmp, (size, int(size * 1.5)), (0, 0, 35.0), 76.0)
            ims.append((name, Image.open(p).convert('RGB')))
            for stem, centre, hh, aspect in frames:
                # (the region follows the body down in the crouch and the step: the region's height on the pelvis)
                z = centre[2] + bpy.data.objects['Body'].location.z
                q = os.path.join(tmp, f'pc-{stem}-{name}.png')
                V['row'](cam, q, tmp, (int(size * aspect), size), (centre[0], centre[1], z), hh)
                cims[stem].append((name, Image.open(q).convert('RGB')))
            print(f'rq pose {name}', flush=True)
    V['clay'](False, saved)
    V['VIEWS'] = keep
    female.pose(0)

    def lay(row, path):
        w, h = row[0][1].size
        strip = Image.new('RGB', (w * len(row), h + 22), (40, 42, 48))
        d = ImageDraw.Draw(strip)
        for i, (name, im) in enumerate(row):
            strip.paste(im, (i * w, 22))
            d.text((i * w + 6, 5), name, fill=(230, 230, 230))
        strip.save(path)
        print('wrote', path, flush=True)
    lay(ims, os.path.join(out, 'poses.png'))
    for stem, row in cims.items():
        lay(row, os.path.join(out, f'pose-crop-{stem}.png'))


def garments():
    """The meshes showing that are not the base figure's (a material id from FIGURE_MATS up, not under the head)."""
    ids = {d['name']: mid for mid, d in female.MATERIALS.items()}
    first = V.get('FIGURE_MATS', 30)
    out = []
    for o in bpy.data.objects:
        if o.type != 'MESH' or o.hide_render or o.active_material is None:
            continue
        if ids.get(o.active_material.name.split('.')[0], first) < first:
            continue
        q = o
        while q.parent is not None and q.parent.name != 'Head':
            q = q.parent
        if q.parent is None:
            out.append(o)
    return out


def geometry(out):
    """Per-garment geometry checks on every build (John, 2026-11-19), in the pose the views were rendered in:
    how much of each garment sits inside the body (penetration), how far its edges are stretched or squashed from
    the garment as built (stretch), and the body's own size (so a drift in its ratios shows night to night).
    Written to DIR/geometry.json for the queue's manifest and the nightly report."""
    import numpy as np
    from mathutils.bvhtree import BVHTree
    t = time.time()
    dg = bpy.context.evaluated_depsgraph_get()
    body = bpy.data.objects.get('Skin body')
    res = {'body': {}, 'garments': {}}
    tree, inv = None, None
    if body is not None:
        tree = BVHTree.FromObject(body, dg)
        inv = body.matrix_world.inverted()
        be = body.evaluated_get(dg)
        bm = be.to_mesh()
        P = np.array([(be.matrix_world @ v.co)[:] for v in bm.vertices])
        be.to_mesh_clear()
        if len(P):
            sh = P[np.abs(P[:, 2] - female.SH_Z) < 1.0]
            hp = P[np.abs(P[:, 2] - female.HIP_Z) < 1.0]
            res['body'] = {'height': round(float(P[:, 2].max() - P[:, 2].min()), 3),
                           'shoulder_width': round(float(np.ptp(sh[:, 0])), 3) if len(sh) else None,
                           'hip_width': round(float(np.ptp(hp[:, 0])), 3) if len(hp) else None,
                           'hip_depth': round(float(np.ptp(hp[:, 1])), 3) if len(hp) else None}
    for o in garments():
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        n = len(me.vertices)
        g = {'verts': n}
        if tree is not None and n:
            M = inv @ ev.matrix_world
            inside, deep = 0, 0.0
            for v in me.vertices:
                p = M @ v.co
                loc, nrm, _, _ = tree.find_nearest(p, 3.0)
                if loc is None:
                    continue
                d = (p - loc).dot(nrm)
                if d < -0.05:
                    inside += 1
                    deep = max(deep, -d)
            g.update(inside=inside, inside_pct=round(100.0 * inside / n, 2), max_depth=round(deep, 3))
        rest = o.data
        if n and len(rest.vertices) == n and len(me.edges) == len(rest.edges) and len(me.edges):
            a = np.array([[e.vertices[0], e.vertices[1]] for e in me.edges])
            pe = np.array([v.co[:] for v in me.vertices])
            pr = np.array([v.co[:] for v in rest.vertices])
            le, lr = np.linalg.norm(pe[a[:, 0]] - pe[a[:, 1]], axis=1), np.linalg.norm(pr[a[:, 0]] - pr[a[:, 1]], axis=1)
            ok = lr > 1e-6
            r = le[ok] / lr[ok]
            if len(r):
                g.update(stretch_p99=round(float(np.percentile(r, 99)), 3), squash_p1=round(float(np.percentile(r, 1)), 3),
                         over_25pct=int((r > 1.25).sum()), under_25pct=int((r < 0.75).sum()))
        ev.to_mesh_clear()
        res['garments'][o.name] = g
    res['seconds'] = round(time.time() - t, 2)
    with open(os.path.join(out, 'geometry.json'), 'w') as f:
        json.dump(res, f, indent=1)
    worst = sorted(res['garments'].items(), key=lambda kv: -kv[1].get('inside_pct', 0))[:3]
    print('rq geometry: ' + '; '.join(f"{k} {v.get('inside_pct', '-')}% inside, stretch p99 {v.get('stretch_p99', '-')}"
                                       for k, v in worst), flush=True)


def precheck(out):
    """The early rejection check (John, 2026-11-19): the geometry checks run right after the build, before any
    render. With RQ_PRECHECK set (JSON: {"inside_pct": 5, "stretch_p99": 1.6, "squash_p1": 0.5}, thresholds the
    nightly job calibrates from the rejections), a garment past one of them stops the job before it renders, and
    DIR/precheck.json says which garment and why. Without it the numbers are only recorded."""
    geometry(out)
    lim = json.loads(os.environ.get('RQ_PRECHECK') or '{}')
    if not lim:
        return
    g = json.load(open(os.path.join(out, 'geometry.json')))
    bad = []
    for name, v in g['garments'].items():
        if 'inside_pct' in lim and (v.get('inside_pct') or 0) > lim['inside_pct']:
            bad.append(f"{name}: {v['inside_pct']}% inside the body (limit {lim['inside_pct']}%)")
        if 'stretch_p99' in lim and (v.get('stretch_p99') or 0) > lim['stretch_p99']:
            bad.append(f"{name}: stretched {v['stretch_p99']}x (limit {lim['stretch_p99']}x)")
        if 'squash_p1' in lim and v.get('squash_p1') is not None and v['squash_p1'] < lim['squash_p1']:
            bad.append(f"{name}: squashed {v['squash_p1']}x (limit {lim['squash_p1']}x)")
    if bad:
        with open(os.path.join(out, 'precheck.json'), 'w') as f:
            json.dump({'failed': bad, 'limits': lim}, f, indent=1)
        print('rq precheck: failed, not rendering: ' + '; '.join(bad), flush=True)
        raise SystemExit('rq precheck: ' + '; '.join(bad))
    print('rq precheck: passed', flush=True)


# the poses the layer gap check reads when the request names none (--poses): the joints' extremes
GAP_POSES = ('rest', 'arms-up', 'arms-fwd', 'elbows', 'step', 'crouch', 'twist')


def layer_gaps(out):
    """The layer gap check (John, 2026-11-19): when two or more layers of an outfit spec (blender/shared/
    riding_habit.json) are in the scene, how far each stands off the one beneath it in every pose, and where a lower
    layer comes through: DIR/layer_gaps.json, and its summary in geometry.json. Report-only. --derive-ease also
    measures the body in the poses and writes the ease the spec derives from it (DIR/ease.json). RQ_LAYER_GAPS=0
    turns it off; --gaps-only stops the job after the checks, before any render."""
    try:
        import ease
        import layer_gaps as LG
    except ImportError:
        return
    spec = ease.load()
    names = opt('--poses')
    poses = [p for p in (names.split(',') if names and names != 'all' else GAP_POSES) if p in POSES and p not in
             getattr(female, 'HEAD_POSES', {})]
    if 'rest' not in poses:
        poses.insert(0, 'rest')
    if '--derive-ease' in argv:
        w = (getattr(V['F'], 'MEASURES', None) or {}).get('w')
        d = ease.derive(spec, apply=apply, w=w)
        with open(os.path.join(out, 'ease.json'), 'w') as f:
            json.dump(d, f, indent=1)
        spec['derived'] = d
    rep = LG.check(spec, poses=poses, apply=apply, out=out)
    gp = os.path.join(out, 'geometry.json')
    if os.path.exists(gp) and 'skipped' not in rep:
        g = json.load(open(gp))
        g['layer_gaps'] = {'layers': rep['layers'], 'poses': poses, 'seconds': rep['seconds'],
                           'show_through': rep['show_through'][:20]}
        with open(gp, 'w') as f:
            json.dump(g, f, indent=1)


def main():
    out = opt('--out', os.path.join(tempfile.gettempdir(), 'rq-views'))
    os.makedirs(out, exist_ok=True)
    timing = {'load': round(T_LOADED - T0, 1)}
    built = V['F'].build
    checked = {}
    sp = sys.modules.get('sorceress_pendant')      # (the bare body every standalone garment starts from: timed apart)
    if sp is not None and hasattr(sp, 'build'):
        body_build = sp.build

        def timed_body(*a, **k):
            t = time.time()
            r = body_build(*a, **k)
            timing['body'] = round(timing.get('body', 0) + time.time() - t, 1)
            return r
        sp.build = timed_body

    def build_then_check(*a, **k):      # (the geometry checks between the build and the first render)
        r = built(*a, **k)
        if os.environ.get('RQ_GEOMETRY', '1') != '0' and '--list' not in argv and not opt('--save'):
            t = time.time()
            try:
                precheck(out)
            except SystemExit:
                raise
            except Exception as e:      # (a check must never fail the render)
                print(f'rq geometry: skipped ({type(e).__name__}: {e})', flush=True)
            checked['s'] = round(time.time() - t, 1)
        if os.environ.get('RQ_LAYER_GAPS', '1') != '0' and '--list' not in argv and not opt('--save'):
            t = time.time()
            try:
                layer_gaps(out)
            except Exception as e:      # (report-only: never fails the render)
                import traceback
                traceback.print_exc()
                print(f'rq layer gaps: skipped ({type(e).__name__}: {e})', flush=True)
            checked['gaps'] = round(time.time() - t, 1)
            if '--gaps-only' in argv:
                print('rq layer gaps: --gaps-only, stopping before the renders', flush=True)
                sys.stdout.flush()
                os._exit(0)
        return r
    V['F'].build = build_then_check
    t = time.time()
    V['main']()             # views.py itself: the build and whatever it was asked for
    timing['views'] = round(time.time() - t, 1)
    if '--list' in argv or opt('--save'):
        return
    if 's' in checked:
        timing['geometry'] = checked['s']      # (inside views.py's own 'build N s', which includes it)
    if 'gaps' in checked:
        timing['layer_gaps'] = checked['gaps']
    for name, fn, on in (('crops', crops, bool(opt('--crop'))), ('poses', poses, '--poses' in argv)):
        if on:
            t = time.time()
            fn(out)
            timing[name] = round(time.time() - t, 1)
    with open(os.path.join(out, 'timing.json'), 'w') as f:
        json.dump(timing, f)


main()
