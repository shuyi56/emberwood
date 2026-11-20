"""The contours of a built figure, to be looked at before any render: level sections at several heights and upright
profiles, every garment's outline drawn over the body's. A round that is too perfect, a step or a kink in an outline
shows here plainly, where a shaded render hides it.

    python3 blender/base/contours.py FIGURE.blend --out SHEET.png --objects "Skin body" Doublet "Doublet sleeve.L" \\
        --z 47.5 46 44 40 36 31 --front 0.2 --side 5.75 [--window x0 x1 y0 y1 z0 z1]

--z: heights of the level sections; --front: the y of an upright cut seen from the front (x by z); --side: the x of
an upright cut seen from the side (y by z). --window: the part of the figure to draw (default: the left arm and
shoulder). Objects are matched by the start of their names and cut as posed, with their thickness."""
import sys

import bpy
import numpy as np

COLOURS = [(150, 150, 150), (170, 40, 34), (30, 90, 170), (30, 140, 70), (190, 130, 20), (120, 50, 150)]


def _cut(tri, axis, c):
    """The segments [n, 2, 3] where triangles [m, 3, 3] cross the plane axis = c."""
    d = tri[:, :, axis] - c
    out = []
    for a, b, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        # (the corner k alone on its side: the plane crosses the two edges that meet at it)
        m = (d[:, k] * d[:, a] < 0) & (d[:, k] * d[:, b] < 0)
        if not m.any():
            continue
        K, A, B = tri[m, k], tri[m, a], tri[m, b]
        ta = (d[m, k] / (d[m, k] - d[m, a]))[:, None]
        tb = (d[m, k] / (d[m, k] - d[m, b]))[:, None]
        out.append(np.stack([K + ta * (A - K), K + tb * (B - K)], 1))
    return np.concatenate(out) if out else np.zeros((0, 2, 3))


def sheet(names, zs, front, side, window, path, scale=46.0):
    from PIL import Image, ImageDraw, ImageFont
    try:
        f = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    except OSError:
        f = ImageFont.load_default()
    dg = bpy.context.evaluated_depsgraph_get()
    tris = []
    for n in names:
        for o in bpy.data.objects:
            if o.type == 'MESH' and o.name.startswith(n):
                e = o.evaluated_get(dg)
                me = e.to_mesh()
                me.calc_loop_triangles()
                co = np.array([(o.matrix_world @ v.co)[:] for v in me.vertices])
                T = np.array([t.vertices[:] for t in me.loop_triangles])
                tris.append((n, co[T]))
                e.to_mesh_clear()
    x0, x1, y0, y1, z0, z1 = window
    panels = [('level cut at z %.1f' % z, 2, z, (0, 1), (x0, x1, y0, y1)) for z in zs]
    panels += [('upright cut at y %.1f, from the front' % c, 1, c, (0, 2), (x0, x1, z0, z1)) for c in front]
    panels += [('upright cut at x %.1f, from the side' % c, 0, c, (1, 2), (y0, y1, z0, z1)) for c in side]
    sizes = [(int((p[4][1] - p[4][0]) * scale) + 30, int((p[4][3] - p[4][2]) * scale) + 50) for p in panels]
    W = sum(s[0] for s in sizes) + 20
    H = max(s[1] for s in sizes) + 40
    im = Image.new('RGB', (W, H), (250, 248, 242))
    dr = ImageDraw.Draw(im)
    x = 10
    for (title, axis, c, (u, v), (u0, u1, v0, v1)), (pw, ph) in zip(panels, sizes):
        dr.text((x + 6, 6), title, fill=(40, 30, 20), font=f)
        dr.rectangle([x, 30, x + pw - 10, 30 + ph - 40], outline=(215, 208, 195))
        for k, (n, tri) in enumerate(tris):
            col = COLOURS[[nn for nn in names].index(n) % len(COLOURS)]
            for s in _cut(tri, axis, c):
                if not (u0 <= s[0, u] <= u1 and v0 <= s[0, v] <= v1):
                    continue
                a = (x + 10 + (s[0, u] - u0) * scale, 30 + ph - 50 - (s[0, v] - v0) * scale)
                b = (x + 10 + (s[1, u] - u0) * scale, 30 + ph - 50 - (s[1, v] - v0) * scale)
                dr.line([a, b], fill=col, width=2)
        x += pw
    lx = 16
    for k, n in enumerate(names):
        dr.text((lx, H - 26), n, fill=COLOURS[k % len(COLOURS)], font=f)
        lx += int(dr.textlength(n, font=f)) + 30
    im.save(path)


if __name__ == '__main__':
    a = sys.argv

    def opt(k, n=None, default=None):
        if k not in a:
            return default
        i = a.index(k) + 1
        j = i
        while j < len(a) and not a[j].startswith('--'):
            j += 1
        return a[i:j]
    bpy.ops.wm.open_mainfile(filepath=a[1])
    bpy.context.scene.frame_set(1)
    sheet(opt('--objects'), [float(v) for v in opt('--z', default=[])], [float(v) for v in opt('--front', default=[])],
          [float(v) for v in opt('--side', default=[])],
          [float(v) for v in opt('--window', default=['1.5', '10.5', '-5', '5.5', '28', '53'])], opt('--out')[0])
    print('contours:', opt('--out')[0])
