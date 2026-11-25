"""The coupled sweep's builder: one lapel (foldfront) and a set of notch collars on it, each gauged for feasible.py.

    python3 blender/tension/sweep.py CUT.blend SWEEP_DIR LAPEL_NAME 'FOLDFRONT_OPTS' 'COLLAR_GRID' [--par 2]

COLLAR_GRID is a JSON object of lists, crossed: '{"notch_angle":[30,40,50],"notch_at":[0.35,0.5,0.65],"point_len":[0.55,0.7,0.85]}'.
Writes SWEEP_DIR/LAPEL_NAME/ (lapel.blend and its npz, fold.log) and c_a<angle>_g<at>_p<len>[...]/ for each collar
(collar.log, out.blend.collar.npz, collar_gauge.npz, gauge.json, join.json); the collars' .blend files are deleted
(20 MB each): rebuild the chosen one. LAPEL_NAME must start w<width*100>_p<point*100> for feasible.py.
Run from the repository's root."""
import os, sys, json, itertools, subprocess, argparse
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
SHORT = dict(notch_angle='a', notch_at='g', point_len='p', tip_round='t', stand_run='sr', roll_r='rr', end_taper='et')

ap = argparse.ArgumentParser(); ap.add_argument('cut'); ap.add_argument('sweep'); ap.add_argument('lapel')
ap.add_argument('fold'); ap.add_argument('grid'); ap.add_argument('--par', type=int, default=2)
a = ap.parse_args()
L = os.path.abspath(os.path.join(a.sweep, a.lapel)); os.makedirs(L, exist_ok=True)
if not os.path.exists(os.path.join(L, 'lapel.blend.lapels.npz')):
    with open(os.path.join(L, 'fold.log'), 'w') as f:
        subprocess.run(['python3', '-I', 'blender/collar/foldfront.py', a.cut, os.path.join(L, 'lapel.blend'), a.fold], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
G = json.loads(a.grid); keys = list(G)


def one(vals):
    o = dict(zip(keys, vals))
    name = 'c_' + '_'.join('%s%s' % (SHORT.get(k, k), ('%d' % v if k == 'notch_angle' else '%d' % round(100 * v) if k == 'tip_round' else '%g' % v)) for k, v in o.items())
    D = os.path.join(L, name); os.makedirs(D, exist_ok=True)
    for f in ('lapel.blend.lapels.npz', 'lapel.blend.strain.npz'):
        if not os.path.exists(os.path.join(D, f)): os.symlink(os.path.join(L, f), os.path.join(D, f))
    with open(os.path.join(D, 'collar.log'), 'w') as f:
        subprocess.run(['python3', '-I', 'blender/collar/notchcollar.py', os.path.join(L, 'lapel.blend'), os.path.join(D, 'out.blend'), json.dumps(o)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    with open(os.path.join(D, 'gauge.json'), 'w') as f:
        subprocess.run(['python3', '-I', os.path.join(HERE, 'collar_gauge.py'), os.path.join(D, 'out.blend.collar.npz'), os.path.join(D, 'collar_gauge.npz')], stdout=f)
    subprocess.run(['python3', '-I', os.path.join(HERE, 'join.py'), D, os.path.join(D, 'join.json')], stdout=subprocess.DEVNULL)
    if os.path.exists(os.path.join(D, 'out.blend')): os.remove(os.path.join(D, 'out.blend'))
    print('sweep: built', a.lapel, name, flush=True)


with ThreadPoolExecutor(a.par) as ex:
    list(ex.map(one, itertools.product(*G.values())))
