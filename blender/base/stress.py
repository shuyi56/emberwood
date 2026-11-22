"""A garment's stress and bend, drawn on its flat cloth (its UV map, which drape.py and the doublet lay out in units
of the cloth as cut): each thread's length as worn over its length on the flat (white the same, red stretched, blue
crushed), and each fold's sharpness (the angle between the two faces at an edge: white flat, dark a sharp crease).
Linen that folds like paper shows as dark lines of sharp creases and blue crushed cloth packed between them.

    python3 blender/base/stress.py FIGURE.blend --object Chemise --out SHEET.png [--z 30 40]

--z: only the cloth worn between these heights. The garment is taken as posed, without its SUBSURF and SOLIDIFY."""
import math
import sys

import bpy
import numpy as np


def maps(o, z_range=None):
    off = [m for m in o.modifiers if m.type in ('SUBSURF', 'SOLIDIFY', 'BEVEL')]
    for m in off:
        m.show_viewport = False
    dg = bpy.context.evaluated_depsgraph_get()
    e = o.evaluated_get(dg)
    me = e.to_mesh()
    co = np.array([(o.matrix_world @ v.co)[:] for v in me.vertices])
    uv = np.zeros((len(co), 2))
    lay = me.uv_layers.active.data
    for p in me.polygons:
        for li, vi in zip(p.loop_indices, p.vertices):
            uv[vi] = lay[li].uv
    ed = np.array([x.vertices[:] for x in me.edges])
    l3 = np.linalg.norm(co[ed[:, 0]] - co[ed[:, 1]], axis=1)
    l2 = np.linalg.norm(uv[ed[:, 0]] - uv[ed[:, 1]], axis=1)
    ok = (l2 > 1e-4) & (l2 < 1.0)                 # (an edge across the flat's seam is no thread)
    strain = np.where(ok, l3 / np.maximum(l2, 1e-9), 1.0)
    fn = np.array([p.normal[:] for p in me.polygons])
    faces_of = {}
    for fi, p in enumerate(me.polygons):
        for k in p.edge_keys:
            faces_of.setdefault(k, []).append(fi)
    bend = np.zeros(len(ed))
    for i, (a, b) in enumerate(ed):
        f = faces_of.get((min(a, b), max(a, b)), [])
        if len(f) == 2:
            bend[i] = math.degrees(math.acos(float(np.clip(np.dot(fn[f[0]], fn[f[1]]), -1, 1))))
    e.to_mesh_clear()
    for m in off:
        m.show_viewport = True
    mid = 0.5 * (co[ed[:, 0]] + co[ed[:, 1]])
    keep = ok.copy()
    if z_range:
        keep &= (mid[:, 2] >= z_range[0]) & (mid[:, 2] <= z_range[1])
    return 0.5 * (uv[ed[:, 0]] + uv[ed[:, 1]])[keep], strain[keep], bend[keep]


def sheet(name, at, strain, bend, path, scale=40.0):
    from PIL import Image, ImageDraw, ImageFont
    try:
        f = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    except OSError:
        f = ImageFont.load_default()
    lo, hi = at.min(0), at.max(0)
    pw, ph = int((hi[0] - lo[0]) * scale) + 40, int((hi[1] - lo[1]) * scale) + 60
    im = Image.new('RGB', (pw, 2 * ph), (250, 248, 242))
    dr = ImageDraw.Draw(im)
    sharp = float((bend > 30).mean())
    dr.text((20, 8), '%s, stress: thread length worn / flat. 2%% %.2f, median %.2f, 98%% %.2f (white 1, red +20%%, blue -20%%)' % (
        name, *np.percentile(strain, [2, 50, 98])), fill=(40, 30, 20), font=f)
    dr.text((20, ph + 8), '%s, bend: angle between faces. median %.1f deg, 98%% %.1f deg, %.1f%% of edges past 30 deg (dark: sharp crease)' % (
        name, float(np.median(bend)), float(np.percentile(bend, 98)), 100 * sharp), fill=(40, 30, 20), font=f)
    for (u, v), s_, b_ in zip(at, strain, bend):
        x, y = 20 + (u - lo[0]) * scale, 40 + (hi[1] - v) * scale
        t = float(np.clip((s_ - 1.0) / 0.2, -1, 1))
        c = (255, int(255 * (1 - t)), int(255 * (1 - t))) if t >= 0 else (int(255 * (1 + t)), int(255 * (1 + t)), 255)
        dr.point((x, y), fill=c)
        k = int(255 * (1 - min(b_ / 45.0, 1.0)))
        dr.point((x, y + ph), fill=(k, k, k))
    im.save(path)
    return dict(strain=np.percentile(strain, [2, 50, 98]), bend_median=float(np.median(bend)),
                bend98=float(np.percentile(bend, 98)), sharp=sharp)


if __name__ == '__main__':
    a = sys.argv
    bpy.ops.wm.open_mainfile(filepath=a[1])
    bpy.context.scene.frame_set(1)
    name = a[a.index('--object') + 1]
    zr = [float(a[a.index('--z') + 1]), float(a[a.index('--z') + 2])] if '--z' in a else None
    at, s, b = maps(bpy.data.objects[name], zr)
    r = sheet(name, at, s, b, a[a.index('--out') + 1])
    print('stress: strain 2/50/98%% %.2f %.2f %.2f; bend median %.1f, 98%% %.1f deg, %.1f%% past 30 deg' % (
        *r['strain'], r['bend_median'], r['bend98'], 100 * r['sharp']))
