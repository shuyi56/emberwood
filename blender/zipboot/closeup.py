"""Close-ups of one boot (the right) on fixed orthographic cameras, the figure built once: each heel style in turn
(heelboot.STYLES; only the stack is rebuilt) from each view, in flat clay and/or the design colours.

    python3 blender/zipboot/closeup.py --figure zipboot --out DIR [--styles block,ferragamo] [--views outer,inner,back]
        [--looks clay,design] [--h CM] [--span UNITS] [--focus heel|boot] [--size PX] [--samples N]

Writes DIR/closeup-<look>[-hCM].png: a row per view, a column per style, a 1 cm bar on each tile. --focus heel frames
the heel (--span 4 units, about 10.5 cm), boot frames the whole boot. Also writes DIR/model.json: the boot's
landmarks (the toe, the heel, the topline, the ankle, the zip) in the outer-side view, for tools/refmeasure."""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]
opt = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d  # noqa: E731
FIG = opt('--figure', 'zipboot')
for d in (FIG, 'heelboot', 'base'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402
import importlib  # noqa: E402
F = importlib.import_module(FIG)
import heelboot as HB  # noqa: E402
import female  # noqa: E402

CM = 165.0 / 63.0
VIEWS = {   # name: (camera offset direction, rotation): orthographic, looking at the right boot
    'outer': (Vector((-1, 0, 0)), (math.pi / 2, 0, -math.pi / 2)),
    'inner': (Vector((1, 0, 0)), (math.pi / 2, 0, math.pi / 2)),
    'back': (Vector((0, 1, 0)), (math.pi / 2, 0, math.pi)),
    'front': (Vector((0, -1, 0)), (math.pi / 2, 0, 0)),
    'inner3q': (Vector((0.7, 0.7, 0)), (math.pi / 2, 0, math.radians(135))),
}


def flat(name, rgb):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get('Principled BSDF')
    b.inputs['Base Color'].default_value = (*rgb, 1)
    b.inputs['Roughness'].default_value = 0.6
    return m


def mine(o, side='R'):
    n = o.name
    return n.endswith('.' + side) and n.startswith(('Boot', 'Heel lift', 'Zip'))


def pts(prefixes, side='R'):
    dg = bpy.context.evaluated_depsgraph_get()
    X = []
    for o in bpy.data.objects:
        if o.type == 'MESH' and o.name.startswith(prefixes) and o.name.endswith('.' + side):
            e = o.evaluated_get(dg)
            X += [(e.matrix_world @ v.co)[:] for v in e.data.vertices]
    return np.array(X)


def landmarks():
    """The boot's landmarks as the reference sheet names them, seen from the outer side (image x = y: the toe to
    the left, as in the photo; image y = -z), in hundredths of a unit. The topline is the pieces' top edges, the
    ankle the boot's front and back at the photo's ankle height (0.489 of the length over the floor)."""
    import last
    dg = bpy.context.evaluated_depsgraph_get()
    o = bpy.data.objects['Boot.R'].evaluated_get(dg)
    X = np.array([(o.matrix_world @ v.co)[:] for v in o.data.vertices])
    B = np.concatenate([X, pts(('Boot sole.',))])
    Hh = pts(('Heel lift',))
    Zt = pts(('Zip teeth',))
    top = np.concatenate([X[p['start']:p['start'] + len(p['V'])][p['edges']['top']] for p in last.PIECES['R']
                          if 'top' in p.get('edges', {})])
    toe = B[np.argmin(B[:, 1])]
    floor = min(B[:, 2].min(), Hh[:, 2].min() if len(Hh) else 9.0)
    lo_h = Hh[Hh[:, 2] < floor + 0.08] if len(Hh) else B
    L = lo_h[:, 1].max() - toe[1]
    hb = Hh[np.argmax(Hh[:, 1])] if len(Hh) else B[np.argmax(B[:, 1])]
    ank = X[np.abs(X[:, 2] - (floor + 0.489 * L)) < 0.1]
    P = {
        'toe_tip': (toe[1], toe[2]), 'heel_back_tread': (lo_h[:, 1].max(), floor), 'floor_back': (lo_h[:, 1].max(), floor),
        'heel_back_seat': (hb[1], Hh[:, 2].max() if len(Hh) else floor), 'breast_tread': (lo_h[:, 1].min(), floor),
        'breast_top': (lo_h[:, 1].min(), Hh[Hh[:, 1] < lo_h[:, 1].min() + 0.1][:, 2].max() if len(Hh) else floor),
        'top_front': (top[np.argmin(top[:, 1]), 1], top[np.argmin(top[:, 1]), 2]),
        'top_back': (top[np.argmax(top[:, 1]), 1], top[np.argmax(top[:, 1]), 2]),
        'ankle_front': (ank[:, 1].min(), floor + 0.489 * L), 'ankle_back': (ank[:, 1].max(), floor + 0.489 * L),
    }
    if len(Zt):
        P['zip_top'] = (Zt[np.argmax(Zt[:, 2]), 1], Zt[:, 2].max())
        P['zip_bottom'] = (Zt[np.argmin(Zt[:, 2]), 1], Zt[:, 2].min())
        P['far_floor'] = (lo_h[:, 1].max(), floor)                 # (the zip's own boot: this one)
        P['far_top_back'] = P['top_back']
    return {k: [float(y) * 100.0, float(-z) * 100.0] for k, (y, z) in P.items()}


def main():
    out = opt('--out', '/tmp/closeup')
    os.makedirs(out, exist_ok=True)
    styles = opt('--styles', 'block').split(',')
    views = opt('--views', 'outer,inner,back').split(',')
    looks = opt('--looks', 'clay').split(',')
    size = int(opt('--size', 520))
    if opt('--h'):
        F.HEEL_H = float(opt('--h')) if hasattr(F, 'HEEL_H') else None
        HB.HEEL_H = float(opt('--h'))
    if 'design' in looks and hasattr(F, 'DESIGN'):
        F.DESIGN = True
    female.MANNEQUIN = True
    F.build()
    female.pose(0)
    bpy.context.view_layer.update()
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = int(opt('--samples', 16))
    sc.cycles.use_denoising = False
    sc.render.resolution_x = sc.render.resolution_y = size
    if hasattr(sc.render.image_settings, 'media_type'):
        sc.render.image_settings.media_type = 'IMAGE'
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGB'
    w = sc.world or bpy.data.worlds.new('w')
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes['Background'].inputs['Color'].default_value = (0.32, 0.33, 0.36, 1)
    w.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.7
    for o in [o for o in bpy.data.objects if o.type in ('LIGHT', 'CAMERA')]:
        bpy.data.objects.remove(o, do_unlink=True)
    for rot, en in (((50, 10, 35), 3.0), ((60, 0, 215), 1.2)):
        sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN'))
        sun.data.energy = en
        sun.rotation_euler = tuple(math.radians(a) for a in rot)
        sc.collection.objects.link(sun)
    cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.type = 'ORTHO'
    clay = flat('clay', (0.42, 0.40, 0.38))
    saved = {o.name: [s.material for s in o.material_slots] for o in bpy.data.objects if o.type == 'MESH'}
    from PIL import Image, ImageDraw
    tiles = {}
    marks = None
    for st in styles:
        if st != HB.STYLE or len(styles) > 1:
            HB.restyle(st)
            if hasattr(F, 'design') and 'design' in looks:
                F.design_heel()
        for o in bpy.data.objects:
            if o.type == 'MESH':
                o.hide_render = not mine(o)
                if '--nosolid' in argv:               # (the leather's thickness off: a debugging view)
                    for m_ in o.modifiers:
                        if m_.type == 'SOLIDIFY':
                            m_.show_render = m_.show_viewport = False
                if '--rest' in argv:                  # (the boot as cut and sewn, before the rig bends it)
                    for m_ in o.modifiers:
                        if m_.type == 'ARMATURE':
                            m_.show_render = m_.show_viewport = False
        bpy.context.view_layer.update()
        B = pts(('Boot.', 'Boot sole.', 'Heel lift'))
        if opt('--focus', 'boot') == 'heel':
            Hh = pts(('Heel lift',))
            c = 0.5 * (Hh.min(0) + Hh.max(0)) + np.array([0, 0.0, 0.8])
            span = float(opt('--span', 6.5))
        else:
            c = 0.5 * (B.min(0) + B.max(0))
            span = float(opt('--span', 1.1 * max(np.ptp(B[:, 1]), np.ptp(B[:, 2]))))
        cam.data.ortho_scale = span
        if marks is None:
            marks = landmarks()
        for look in looks:
            for o in bpy.data.objects:
                if o.type == 'MESH' and o.name in saved:
                    for s_, m_ in zip(o.material_slots, saved[o.name] if look == 'design' else [clay] * len(o.material_slots)):
                        s_.material = m_
                    if look == 'clay':
                        for s_ in o.material_slots:
                            s_.material = clay
            if look == 'clay':
                for o in bpy.data.objects:
                    if o.type == 'MESH' and o.name.startswith('Heel lift'):
                        o.data.materials.clear()
                        o.data.materials.append(clay)
            for v in views:
                d, rot = VIEWS[v]
                cam.location = Vector(c) + d * 60.0
                cam.rotation_euler = rot
                p = os.path.join(out, 'v-%s-%s-%s.png' % (look, st, v))
                sc.render.filepath = p
                bpy.ops.render.render(write_still=True)
                tiles[(look, st, v)] = Image.open(p).convert('RGB')
                print('rendered', look, st, v, flush=True)
    pad = 30
    px_cm = size / (span * CM)
    for look in looks:
        sheet = Image.new('RGB', (len(styles) * size, len(views) * (size + pad)), (40, 41, 45))
        dr = ImageDraw.Draw(sheet)
        for i, st in enumerate(styles):
            for j, v in enumerate(views):
                y0 = j * (size + pad)
                sheet.paste(tiles[(look, st, v)], (i * size, y0 + pad))
                dr.text((i * size + 8, y0 + 8), '%s heel, %s side, %s' % (st, v, look), fill=(235, 235, 235))
                yb = y0 + pad + size - 14
                dr.line([(i * size + 10, yb), (i * size + 10 + px_cm, yb)], fill=(235, 235, 235), width=3)
                dr.text((i * size + 14 + px_cm, yb - 6), '1 cm', fill=(235, 235, 235))
        stem = 'closeup-%s' % look + ('-h%s' % opt('--h') if opt('--h') else '')
        sheet.save(os.path.join(out, stem + '.png'))
        print('wrote', os.path.join(out, stem + '.png'), flush=True)
    json.dump(dict(landmarks=marks), open(os.path.join(out, 'model.json'), 'w'), indent=1)


main()
