"""The collar's finishing passes as one post-process (John, 2026-11-21: "save some of the steps of finishing you did as a
post processing"). Run after a build, on its fold pickle (COLLAR_FOLD_OUT) and the .blend it wrote:

    python3 blender/collar/finish.py FOLD.pkl BUILD.blend OUT.blend [WORKDIR]

Steps, in order (each is its own script and can be run alone):
  1  collarlab.py        solve the standalone collar: swing 0.2 (the outer edge's pull stays under 2 %), the notch
                         drafted 85 deg against each lapel's folded top edge (fit_fold), gorge carried 0.5 along the
                         lapel's top edge, collar end 1.3, a light press
  2  collarlab.py smooth Taubin-fair the collar's free edges, front seam, roll and tips
  3  swapcollar.py       put that collar into the build's blend
  4  lapsmooth.py        fair the lapels' free edges in the bodice mesh
  5  roundpress.py       round the fold beside the neck; press her left collar's front onto the lapel
  6  edgein.py           fair the ragged outer edge at her left, move the collar in 1.2 mm
  7  backsmooth.py       fair the fold line and outer edge across the back through the centre, weld the halves
Not in the post-process: pushout.py (stands the back of the collar off the neck; kept out by default, run it after
this if wanted) and tailor.py's stretch_out option (eased the crease only slightly).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
STYLE = {'COLLAR_STYLE': 'modern', 'OMP_NUM_THREADS': '1'}
SOLVE = {"swing": 0.2, "gorge_ext": 0.5, "end": 1.3, "notch": 85, "fit_fold": True, "press": 0.006}
SMOOTH = {"rounds": 150, "band": 1.0, "inner_rounds": 60, "front_y": -1.5, "tip": 2.0}
ROUNDPRESS = {"round": {"rounds": 40, "band": 1.0}, "push": {"amount": 0.5, "sides": [1]}}
EDGEIN = {"side": 1, "x0": 1.0, "y1": 1.8, "y0": -1.2, "in": 0.12, "rounds": 150}
BACK = {"y0": -1.2, "chain": 300, "surf": 20, "band": 1.0}


def run(script, *args, **env):
    e = dict(os.environ, **env)
    print('finish:', script, flush=True)
    subprocess.run([PY, os.path.join(HERE, script), *map(str, args)], check=True, env=e)


def main(fold, blend, out, work=None):
    work = work or os.path.join(os.path.dirname(os.path.abspath(out)), 'finish')
    os.makedirs(work, exist_ok=True)
    w = lambda n: os.path.join(work, n)
    run('collarlab.py', fold, w('solved.pkl'), json.dumps(SOLVE), **STYLE)
    run('collarlab.py', 'smooth', w('solved.pkl'), w('smoothed.pkl'), json.dumps(SMOOTH), **STYLE)
    run('swapcollar.py', blend, w('smoothed.pkl'), w('a.blend'))
    run('lapsmooth.py', w('a.blend'), w('b.blend'), COLLAR_FOLD_PKL=fold)
    run('roundpress.py', w('b.blend'), w('smoothed.pkl'), w('c.blend'), json.dumps(ROUNDPRESS))
    run('edgein.py', w('c.blend'), w('c.pkl'), w('d.blend'), json.dumps(EDGEIN))
    run('backsmooth.py', w('d.blend'), w('d.pkl'), out, json.dumps(BACK))


if __name__ == '__main__':
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:5])
