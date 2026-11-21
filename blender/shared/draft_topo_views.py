"""The draft topology check as a views job: build a figure as drafted (no cloth solver, nothing sewn by the solver), read
its draft, run blender/shared/drafttopo.py's checks on it and write the report and the diagram sheet; --clay adds cheap
orthographic clay views of the drafted pieces (8 samples, front, 3/4, profile, back 3/4, back) to look at beside it.

    python3 tools/render-queue/rq_run.py --quality check -- blender/shared/draft_topo_views.py --figure boot --mannequin \
        [--lace tucked|bow] [--clay] [--size 360] --out DIR
    python3 tools/render-queue/rq_run.py --quality check -- blender/shared/draft_topo_views.py --figure redingote_bodice \
        --mannequin [--styles 'buttoned to the neck,notch,modern'] [--clay] --out DIR

Writes DIR/draft-topo[-NAME].json, DIR/draft-topo[-NAME]-sheet.png and, with --clay, DIR/turn-clay.png. The figures it
knows (an adapter in tools/render-queue/rq_drafttopo.py reads each one's draft): boot (last.PIECES and the lacing),
redingote_bodice (the closing edge and its buttons, against each lapel style's break). Report-only: it never fails."""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
opt = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d      # noqa: E731
FIG = opt('--figure', 'boot')
OUT = opt('--out', os.path.join(ROOT, 'draft-topo'))
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'render-queue'))
import rq_drafttopo as T  # noqa: E402

VIEWS_PY = os.path.join(ROOT, 'blender', 'base', 'views.py')
sys.argv = ['views.py', '--figure', FIG] + (['--mannequin'] if '--mannequin' in argv else [])
src = open(VIEWS_PY).read()
cut = src.rstrip().rfind('\nmain()')
V = {'__file__': VIEWS_PY, '__name__': 'draft_topo_views'}
exec(compile(src[:cut], VIEWS_PY, 'exec'), V)
bpy, female, F = V['bpy'], V['female'], V['F']
if '--mannequin' in argv:
    female.MANNEQUIN = True
# as drafted: nothing settled or sewn by a solver (the draft stage)
for k, v in (('SEW', False), ('SETTLE', False), ('CREASE', False)):
    if hasattr(F, k):
        setattr(F, k, v)
t0 = time.time()
F.build()
print('draft topo: %s built as drafted in %.0f s' % (FIG, time.time() - t0), flush=True)

import numpy as np  # noqa: E402

if FIG == 'boot':
    import last
    lace = opt('--lace', 'tucked')
    for side in opt('--sides', 'R'):
        dr = T.boot_draft(last.PIECES[side], dict(last.RATIOS), lace=lace, side=side)
        T.write(dr, OUT, stem='draft-topo-boot-%s-%s' % (side, lace))
elif FIG == 'redingote_bodice':
    M = dict(F.MEASURES)
    import bodice
    btn = bpy.data.objects.get('Redingote bodice buttons')
    body = bpy.data.objects.get('Redingote bodice')
    B = np.array([btn.matrix_world @ v.co for v in btn.data.vertices])
    z = np.sort(B[:, 2])
    cuts = np.r_[0, np.nonzero(np.diff(z) > 0.3)[0] + 1, len(z)]
    zs = [float(z[a:c].mean()) for a, c in zip(cuts[:-1], cuts[1:])]
    Vb = np.array([body.matrix_world @ v.co for v in body.data.vertices])
    x0 = float(np.median(B[:, 0]))
    col = Vb[np.abs(Vb[:, 0] - x0) < 0.08]
    edge = []
    for zz in np.linspace(M['z_neck_front'], M['z_hem_front'], 120):
        q = col[np.abs(col[:, 2] - zz) < 0.1]
        if len(q):
            edge.append([x0, float(q[:, 1].min()), float(zz)])
    for st in opt('--styles', ','.join(T.LAPEL_DROP)).split(','):
        dr = T.bodice_draft(M, dict(bodice.RATIOS), sorted(zs, reverse=True), np.array(edge), st)
        T.write(dr, OUT, stem='draft-topo-bodice-' + st.split()[0])
else:
    print('draft topo: no adapter for the figure %s (boot, redingote_bodice)' % FIG, flush=True)

if '--clay' in argv:
    import tempfile
    size = int(opt('--size', 360))
    cam = V['look_3d'](int(opt('--samples', 8)))
    saved = {}
    with tempfile.TemporaryDirectory() as tmp:
        V['clay'](True, saved)
        shown = V['isolate']() if (opt('--only') or opt('--hide')) else None
        centre, h = V['fit'](shown, 1 / 1.75) if shown is not None else ((0, 0, 32.5), 68.0)
        if FIG == 'boot':
            centre, h = (0.0, 0.0, 8.0), 20.0
        V['row'](cam, os.path.join(OUT, 'turn-clay.png'), tmp, (size, int(size * 1.75)), centre, h)
        V['clay'](False, saved)
    print('draft topo: clay views in %s' % OUT, flush=True)
