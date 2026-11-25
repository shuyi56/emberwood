"""Contour sheet for a lapel and collar build, before against after on the same axes (John's rule: inspect the
contours of every solve before rendering):

    python3 blender/tension/contour_sheet.py BEFORE.npz AFTER.npz SHEET.png

(the .npz from dump_build.py). Panels: level sections through the V and the collar at six heights (top view),
the sections square across each roll line at three stations (the fold's own profile: one smooth hairpin, the lapel
parallel to the front), and the side profile through each lapel's point. Grey is before, black is after."""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def section(V, T, n, c):
    d = V @ n - c
    D = d[T]
    hit = (D.min(1) < 0) & (D.max(1) > 0)
    segs = []
    for t, dd in zip(T[hit], D[hit]):
        p = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (dd[a] < 0) != (dd[b] < 0):
                s = dd[a] / (dd[a] - dd[b]); p.append(V[t[a]] + s * (V[t[b]] - V[t[a]]))
        if len(p) == 2:
            segs.append(p)
    return np.array(segs).reshape(-1, 2, 3)


def load(path):
    D = np.load(path)
    out = [(D['body_X'], D['body_T'])]
    if 'collar_X' in D.files:
        out.append((D['collar_X'], D['collar_T']))
    return out, eval(str(D['M'][0]))


def draw(ax, parts, n, c, axes, col, lw, box):
    for V, T in parts:
        S = section(V, T, n, c)
        if len(S) == 0:
            continue
        S2 = S[..., axes]
        keep = ((S2[..., 0] > box[0]) & (S2[..., 0] < box[1]) & (S2[..., 1] > box[2]) & (S2[..., 1] < box[3])).all(1)
        ax.add_collection(LineCollection(S2[keep], colors=col, linewidths=lw))


if __name__ == '__main__':
    A, M = load(sys.argv[1]); B, _ = load(sys.argv[2])
    zs = [M['z_bust'] - 1.0, M['z_bust'], M['z_bust'] + 1.5, M['z_bust'] + 3.0, M['z_bust'] + 4.5, M['z_neck_front'] + 0.6]
    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    for i, z in enumerate(zs):
        ax = axs.flat[i]; box = (-4.5, 4.5, -5.5, 1.0)
        draw(ax, A, np.array([0, 0, 1.0]), z, [0, 1], '0.65', 2.0, box)
        draw(ax, B, np.array([0, 0, 1.0]), z, [0, 1], 'k', 0.8, box)
        ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3]); ax.set_aspect('equal')
        ax.set_title('level section z %.1f (top view, front down)' % z); ax.invert_yaxis()
    # side profiles through each lapel's middle and the collar's point region
    for j, x in enumerate((-1.4, 1.4, -2.6, 2.6)):
        ax = axs.flat[6 + j]; box = (-6.0, 1.0, 42.0, 53.0)
        draw(ax, A, np.array([1.0, 0, 0]), x, [1, 2], '0.65', 2.0, box)
        draw(ax, B, np.array([1.0, 0, 0]), x, [1, 2], 'k', 0.8, box)
        ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3]); ax.set_aspect('equal')
        ax.set_title('side section x %.1f (%s), front to the left' % (x, 'her right' if x < 0 else 'her left'))
    # front outline: the sections at the collar's height, front view
    for j, y in enumerate((-2.6, -3.2)):
        ax = axs.flat[10 + j]; box = (-4.5, 4.5, 42.0, 53.0)
        draw(ax, A, np.array([0, 1.0, 0]), y, [0, 2], '0.65', 2.0, box)
        draw(ax, B, np.array([0, 1.0, 0]), y, [0, 2], 'k', 0.8, box)
        ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3]); ax.set_aspect('equal')
        ax.set_title('coronal section y %.1f (front view)' % y)
    # the polygon check (polycheck.py) of the after build, if it is a lapel and collar build: side counts, every
    # mismatched edge, and the curvature against the riding-habit photo (John, 2026-11-23)
    import os
    Bd = os.path.dirname(os.path.abspath(sys.argv[2]))
    for ax in axs.flat[12:]: ax.axis('off')
    if os.path.exists(Bd + '/lapel.blend.lapels.npz'):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import polycheck
        r = polycheck.check(Bd); axs.flat[12].axis('on'); polycheck.draw(axs.flat[12], r)
        axs.flat[13].text(0, 1, polycheck.report(r), va='top', family='monospace', fontsize=7, transform=axs.flat[13].transAxes)
        print(polycheck.report(r).splitlines()[0])
    fig.suptitle('grey: before (%s)   black: after (%s)' % (sys.argv[1].split('/')[-2], sys.argv[2].split('/')[-2]))
    plt.tight_layout(); plt.savefig(sys.argv[3], dpi=60)
    print('contour_sheet: wrote', sys.argv[3])
