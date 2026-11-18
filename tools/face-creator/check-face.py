"""Check a face creator export (tools/face-creator/export-face.cjs) against the creator, without rendering.

    python3 tools/face-creator/check-face.py blender/faces/*.json.gz

For each export: how closely the Blender lookup of its surface (blender/base/creator.py) lands on the creator's
exact points, off the grid (the export carries 3000 of them, 2000 over the face), and how far the head Blender
would build from the same sliders on its own (female.py's sections and reliefs) lies from the creator's: the
drift that building from the surface removes. Distances in head heights and in model units (x HEAD_H).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'blender', 'base'))
import female  # noqa: E402
import creator  # noqa: E402

DEFAULT_FACE = dict(female.FACE)
for path in sys.argv[1:]:
    doc = creator.load(path)
    surf = creator.Surface(doc['surface'])
    C = np.array(doc['check'])
    got = surf.points(C[:, 0], C[:, 1])
    err = np.linalg.norm(got - C[:, 2:5], axis=1)
    face = slice(0, 2000)
    # the head female.py builds from the same sliders (no surface)
    female.HEAD_SOURCE = None
    female.FACE.clear(); female.FACE.update(DEFAULT_FACE)
    for k in female.FACE:
        if k in doc['settings']:
            female.FACE[k] = doc['settings'][k]
    own = np.array([female._head_point(z, th) for z, th in C[:, :2]])
    drift = np.linalg.norm(own - C[:, 2:5], axis=1)
    worst = int(np.argmax(drift[face]))
    print(f"{os.path.basename(path)} ({doc.get('face')}, sex {doc['settings'].get('sex', 0)}):"
          f"\n  lookup vs creator: face max {err[face].max():.5f} mean {err[face].mean():.6f}, head max {err.max():.5f}"
          f" head heights (max {err.max() * female.HEAD_H:.4f} units)"
          f"\n  female.py's own build vs creator: face max {drift[face].max():.4f} mean {drift[face].mean():.5f}"
          f" (worst at z {C[worst, 0]:.3f}, x {C[worst, 2]:+.3f})")
