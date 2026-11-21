"""Collar lab: the tailored top collar alone, on the solver, outside the build (John, 2026-11-21: "build the standalone
on a solver"). A build's COLLAR_FOLD_OUT pickle has everything the collar meets (the bodice with its lapels turned
back, the body, the neck edge, the gorges); tailor.tailor_collar is run on it with a try's settings, in about 3
minutes, and the result saved for a clay render (swap into the build's .blend) and its numbers printed.

    COLLAR_STYLE=modern python3 blender/collar/collarlab.py FOLD.pkl OUT.pkl '{"swing": 0.0, "gorge_ext": 0.5}'

A try's JSON is tailor.SETTINGS' keys (swing, end, notch, lay_r, crease_on_lapel, ...) and the lab's own:
  gorge_ext   the gorge seam carried on along the lapel's top edge by this share of it (notch toward the lapel's
              point): the collar's front reaches further down the chest and its end grows with it, as on the 1890s
              habit, whose collar takes the top 40 % of the V. 0 is the build's notch.
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('collar', 'shared', 'base', 'redingote_bodice', 'redingote', 'vest', 'chemise', 'sorceress_pendant'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
import tailor      # noqa: E402


def main():
    f = pickle.load(open(sys.argv[1], 'rb'))
    st = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    ext = float(st.pop('gorge_ext', 0.0))
    gorges = dict(f['gorges'])
    outs = {(1 if o[1][0] > 0 else -1): o for o in f['outlines']}
    Pp = np.asarray(outs[-1][2][3], float)
    if ext:
        for sg, (X, G, X3) in list(gorges.items()):
            # (the lapel's own point on that side: the left lapel's outline is clipped where the right lies over it, so
            # the right's is mirrored)
            P_ = Pp * np.array([-sg, 1.0]) if sg == 1 else Pp
            P_ = np.array([abs(Pp[0]) * sg, Pp[1]])
            G2 = np.asarray(G) + ext * (P_ - np.asarray(G))
            gorges[sg] = (X, G2, X3)
        print('collarlab: notch moved %.0f %% along the lapel\'s top edge: gorge %.2f -> %.2f'
              % (100 * ext, np.linalg.norm(np.asarray(f['gorges'][-1][1]) - f['gorges'][-1][0]), np.linalg.norm(gorges[-1][1] - gorges[-1][0])))
    D = tailor.tailor_collar(f['E'], f['M'], f['V'], f['F'], f['skinP'], f['skinF'], gorges, Pp, settings=st)
    pickle.dump(D, open(sys.argv[2], 'wb'))
    print('collarlab: ' + D['report'])


if __name__ == '__main__':
    main()
