"""Join diagnostics: G0 gap, G1 tangent and G2 curvature continuity across a join, a curvature comb, a normal-
deviation heat map, zebra and matcap views, and pass/fail against researched limits (blender/shared/continuity.py
says where each limit comes from). Runs outside Blender on a dump of the built meshes (tools/joins/dump.py), and
inside a build as the queue's optional pre-render check (rq_views.py --joins).

    # a join between two meshes (one passing into the other: a head into its neck, an ear into the head)
    python3 tools/joins/joins.py join --npz scene.npz --a "Head shape" --b "Skin body" --near 0,0,53.2 --out DIR

    # one mesh, a seam you name: the plane that cuts it (origin and normal), kept within --clip
    python3 tools/joins/joins.py join --npz scene.npz --a "Skin body" --seam-plane -2.8,-1.3,2.6,0,0.7,0.7 \\
        --clip "abs(x+2.8)<1.2" --out DIR

    # a profile line with no marked join (the foot's top line): curvature, steps, comb, views
    python3 tools/joins/joins.py profile --npz scene.npz --a "Skin body" --plane x=-2.8 --polar -2,1,-5,100 \\
        --clip "z<6,y<1" --out DIR

    # a preset from presets.json (the joins the project checks: neck, foot-top, ankle-front, ...)
    python3 tools/joins/joins.py preset neck foot-top --npz scene.npz --out DIR

Every run writes join-NAME.json (the numbers: per station and the summary, the verdict, the cause) and
join-NAME.png (the contour sheet). The exit code is 0 when nothing fails, 1 when a join fails (pass --report-only
to always exit 0)."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'blender', 'shared'))
import continuity as C  # noqa: E402
import join_sheet  # noqa: E402

PRESETS = os.path.join(HERE, 'presets.json')


def load(npz, name):
    d = np.load(npz) if isinstance(npz, str) else npz
    if name + '/V' not in d:
        raise SystemExit(f'joins: no mesh {name!r} in the dump (has: {sorted({k.split("/")[0] for k in d})})')
    return np.asarray(d[name + '/V'], float), np.asarray(d[name + '/F'], int)


def vec(s):
    return np.array([float(v) for v in s.split(',')])


def clip_mask(P, clip):
    if not clip:
        return np.ones(len(P), bool)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    m = np.ones(len(P), bool)
    for c in clip.split(','):
        m &= eval(c, {'abs': np.abs, 'np': np}, {'x': x, 'y': y, 'z': z})      # (the CLI's own option, not data)
    return m


def longest_run(ok, closed=True):
    idx = np.nonzero(ok)[0]
    if not len(idx):
        return idx
    runs = np.split(idx, np.nonzero(np.diff(idx) != 1)[0] + 1)
    if closed and ok[0] and ok[-1] and len(runs) > 1:
        runs = [np.concatenate([runs[-1], runs[0]])] + runs[1:-1]
    return max(runs, key=len)


def plane_line(A, plane, clip=None, polar=None):
    """The section of mesh A by a plane 'x=-2.8' (or 'ox,oy,oz,nx,ny,nz'), the longest run of it inside clip (and
    inside the polar fan 'cy,cz,a0,a1' round a point in the y-z plane, as the foot study takes the top line)."""
    if '=' in plane:
        ax, v = plane.split('=')
        n = np.eye(3)['xyz'.index(ax)]
        o = n * float(v)
    else:
        q = vec(plane)
        o, n = q[:3], q[3:]
    lines = C.plane_section(A[0], A[1], o, n)
    if not lines:
        raise SystemExit('joins: the plane misses the mesh')
    best = None
    for L in lines:
        ok = clip_mask(L, clip)
        if polar:
            cy, cz, a0, a1 = vec(polar)
            ang = np.degrees(np.arctan2(L[:, 2] - cz, -(L[:, 1] - cy)))
            ok &= (ang > a0) & (ang < a1)
        r = longest_run(ok, closed=np.linalg.norm(L[0] - L[-1]) < 1e-6)
        if best is None or len(r) > len(best[1]):
            best = (L, r)
    P = best[0][best[1]]
    if len(P) > 1 and P[0, 2] > P[-1, 2]:
        P = P[::-1]          # (from the low end up: the toe tip first on a foot's top line)
    return P, n


def run_join(name, A, B=None, seam=None, mode=None, kind='matte', near=None, radius=4.0, stations=24, window=None,
             out='.', solved=False, mm_per_u=C.MM_PER_U):
    if seam is None:
        if B is None:
            raise SystemExit('joins: one mesh needs a seam (--seam-plane, --seam-edges, --boundary)')
        seam = C.intersection_seam(A, B, near=near, radius=radius)
        if seam is None:
            raise SystemExit(f'joins: {name}: the meshes do not cross near {near}')
    rep, rows = C.check_join(name, A, seam, B=B, kind=kind, stations=stations, window=window, mode=mode,
                             solved=solved, mm_per_u=mm_per_u)
    C.write(rep, rows, out, A=A, B=B, seam=seam)
    return rep


def run_profile(name, A, plane, clip=None, polar=None, window_cm=1.0, smooth=0.5, out='.', joins_at=(),
                skip_ends_cm=0.0, mm_per_u=C.MM_PER_U):
    P, n = plane_line(A, plane, clip, polar)
    prof = C.polyline_curvature(P, step=0.05, window=smooth, plane_normal=n, mm_per_u=mm_per_u)
    rows, steps, noise = C.curvature_steps(prof, window_cm=window_cm, skip_ends_cm=skip_ends_cm, mm_per_u=mm_per_u)
    k = prof['k']
    keep = prof['s_cm'] >= skip_ends_cm
    rep = {'name': name, 'mode': 'profile', 'plane': plane, 'clip': clip, 'polar': polar,
           'length_cm': round(float(prof['s_cm'][-1]), 2), 'smooth_u': smooth, 'window_cm': window_cm,
           'rms_k_per_cm': round(float(np.sqrt((k[keep] ** 2).mean())), 4),
           'max_abs_k_per_cm': round(float(np.abs(k[keep]).max()), 4),
           'rms_dk_per_cm2': round(float(np.sqrt((prof['dk'][keep] ** 2).mean())), 4),
           'g1_deg_max': round(max(r['g1_deg'] for r in rows), 3) if rows else None,
           'g2_dev_p90': round(float(np.percentile([r['g2_dev'] for r in rows], 90)), 3) if rows else None,
           'noise': noise, 'steps': steps, 'skip_ends_cm': skip_ends_cm, 'limits': C.LIMITS['matte'], 'sources': C.SOURCES}
    # a join named on the line (s from the low end, cm): does it fall on a step?
    rep['joins'] = []
    joins_at = [resolve_at(js, prof, P) for js in joins_at]
    for js in joins_at:
        near = [st for st in steps if abs(st['s_cm'] - js) <= window_cm]
        rep['joins'].append({'s_cm': js, 'on_step': bool(near), 'steps': near})
    bad = [s for s in steps if s['g1_deg'] > C.LIMITS['matte']['g1_deg'][1] or s['g2_dev'] > C.LIMITS['matte']['g2_dev'][1]]
    warn = [s for s in steps if s not in bad]
    rep['verdict'] = 'fail' if bad else ('warn' if warn else 'pass')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f'join-{name}.json'), 'w') as f:
        json.dump(rep, f, indent=1)
    # the views: the surface round the line, from the side the line is seen from (along the plane's normal)
    Vr, Fr, _ = C.region(A[0], A[1], P, 1.5)
    look = -np.asarray(n, float)
    mid = P[len(P) // 2]
    if (Vr.mean(0) - mid) @ look < 0:
        look = -look
    join_sheet.profile_sheet(name, prof, rows, steps, os.path.join(out, f'join-{name}.png'), A=(Vr, Fr), line3=P,
                             look=look, joins=joins_at, rep=rep)
    return rep


def resolve_at(js, prof, P):
    """A join's place on a profile: a number (cm from the low end) or 'kmax:AXIS:LO:HI', the sample of greatest
    |k| whose AXIS coordinate lies in [LO, HI] (the ankle's front: where the shin turns into the instep)."""
    if isinstance(js, (int, float)):
        return float(js)
    _, ax, lo, hi = js.split(':')
    s3 = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    c = np.interp(prof['s_cm'] / (prof['s_cm'][-1] or 1) * s3[-1], s3, P[:, 'xyz'.index(ax)])
    m = (c >= float(lo)) & (c <= float(hi))
    m &= (prof['s_cm'] > 1.0) & (prof['s_cm'] < prof['s_cm'][-1] - 1.0)     # (not the smoothing's ends)
    if not m.any():
        return float('nan')
    i = np.nonzero(m)[0][np.argmax(np.abs(prof['k'][m]))]
    return round(float(prof['s_cm'][i]), 2)


def seam_from_plane(A, spec, clip=None):
    q = vec(spec)
    lines = C.plane_section(A[0], A[1], q[:3], q[3:])
    best = None
    for L in lines:
        r = longest_run(clip_mask(L, clip), closed=np.linalg.norm(L[0] - L[-1]) < 1e-6)
        if best is None or len(r) > len(best):
            best = L[r]
    return best


def preset(names, npz, out, report_only=False):
    P = json.load(open(PRESETS))['joins']
    reps = []
    for nm in names:
        p = P[nm]
        A = load(npz, p['a'])
        if p['type'] == 'profile':
            r = run_profile(nm, A, p['plane'], p.get('clip'), p.get('polar'), p.get('window_cm', 1.0),
                            p.get('smooth_u', 0.5), out, p.get('joins_at', ()), p.get('skip_ends_cm', 0.0))
        else:
            B = load(npz, p['b']) if p.get('b') else None
            seam = seam_from_plane(A, p['seam_plane'], p.get('clip')) if p.get('seam_plane') else None
            r = run_join(nm, A, B, seam, p.get('mode'), p.get('kind', 'matte'),
                         vec(p['near']) if p.get('near') else None, p.get('radius', 4.0), p.get('stations', 24),
                         p.get('window'), out, p.get('solved', False))
        reps.append(r)
        print(summary_line(r), flush=True)
    return reps


def summary_line(r):
    if r.get('mode') == 'profile':
        return (f"joins {r['name']}: {r['verdict'].upper()}  rms k {r['rms_k_per_cm']}/cm  max |k| "
                f"{r['max_abs_k_per_cm']}/cm  G1 max {r['g1_deg_max']} deg  steps {len(r['steps'])}")
    s = r.get('summary', {})
    return (f"joins {r['name']}: {r['verdict'].upper()}  G0 max {s.get('g0_mm_max')} mm  G1 p90 {s.get('g1_deg_p90')}"
            f" deg (away {s.get('ctl_g1_deg_p90')})  G2 p90 {s.get('g2_dev_p90')} (away {s.get('ctl_g2_dev_p90')})"
            + ('  | ' + '; '.join(r.get('cause', [])) if r.get('cause') else ''))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    j = sub.add_parser('join')
    j.add_argument('--npz', required=True)
    j.add_argument('--a', required=True)
    j.add_argument('--b')
    j.add_argument('--mode', choices=['intersect', 'boundary', 'seam'])
    j.add_argument('--near')
    j.add_argument('--radius', type=float, default=4.0)
    j.add_argument('--seam-plane')
    j.add_argument('--clip')
    j.add_argument('--kind', default='matte', choices=list(C.LIMITS))
    j.add_argument('--stations', type=int, default=24)
    j.add_argument('--window', type=float, help='one-sided fit length, model units (default 6 edges)')
    j.add_argument('--solved', action='store_true', help='the piece comes from a solver (cause wording)')
    j.add_argument('--name', default='join')
    j.add_argument('--out', default='.')
    pr = sub.add_parser('profile')
    pr.add_argument('--npz', required=True)
    pr.add_argument('--a', required=True)
    pr.add_argument('--plane', required=True)
    pr.add_argument('--clip')
    pr.add_argument('--polar')
    pr.add_argument('--window-cm', type=float, default=1.0)
    pr.add_argument('--smooth', type=float, default=0.5, help='Savitzky-Golay window, model units')
    pr.add_argument('--joins-at', default='', help='arc lengths (cm from the low end) of joins on the line')
    pr.add_argument('--skip-ends-cm', type=float, default=0.0)
    pr.add_argument('--name', default='profile')
    pr.add_argument('--out', default='.')
    ps = sub.add_parser('preset')
    ps.add_argument('names', nargs='+')
    ps.add_argument('--npz', required=True)
    ps.add_argument('--out', default='.')
    for p in (j, pr, ps):
        p.add_argument('--report-only', action='store_true')
    a = ap.parse_args(argv)
    if a.cmd == 'join':
        A = load(a.npz, a.a)
        B = load(a.npz, a.b) if a.b else None
        seam = seam_from_plane(A, a.seam_plane, a.clip) if a.seam_plane else None
        reps = [run_join(a.name, A, B, seam, a.mode, a.kind, vec(a.near) if a.near else None, a.radius, a.stations,
                         a.window, a.out, a.solved)]
        print(summary_line(reps[0]))
    elif a.cmd == 'profile':
        reps = [run_profile(a.name, load(a.npz, a.a), a.plane, a.clip, a.polar, a.window_cm, a.smooth, a.out,
                            [float(v) for v in a.joins_at.split(',') if v], a.skip_ends_cm)]
        print(summary_line(reps[0]))
    else:
        reps = preset(a.names, a.npz, a.out)
    failed = [r['name'] for r in reps if r.get('verdict') == 'fail']
    return 0 if a.report_only or not failed else 1


if __name__ == '__main__':
    sys.exit(main())
