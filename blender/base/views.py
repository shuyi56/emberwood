"""Views of any character built on the base figure, for checking work without the full sheet.

    python3 blender/base/views.py --figure rogue --out DIR [--cells 0-7:0,3,6] [--turn] [--head]
                                  [--size PX] [--samples N] [--frame COL]

--figure   the character: blender/<figure>/<figure>.py (ranger, rogue)
--cells    sprite cells of the sheet, ROWS:COLS (rows 0-7 the facings, cols 0-1 idle and 2-9 walk; lists or
           ranges): DIR/cell-rR-cC.png each at its real size and DIR/cells.png, all of them laid out 3x
--turn     the whole figure in its 3D look (sprite_mode off): front, front 3/4, profile, back 3/4 and back, in a
           row (DIR/turn.png), posed at sheet column --frame (default 0, the idle)
--head     the head and shoulders from the same five angles (DIR/head.png)
--clay     with --turn or --head: the same views again with every surface one matte grey, the hair included and
           the painted layers (makeup, brows, painted strands, fuzz) hidden, so only the forms and how they meet
           show (DIR/turn-clay.png, DIR/head-clay.png)
--set      NAME=VALUE: set a module setting of the character before it builds (e.g. FUSE=False), repeatable
"""
import importlib
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
opt = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d  # noqa: E731
FIGURE = opt('--figure', 'ranger')
sys.path.insert(0, os.path.join(HERE, '..', FIGURE))
sys.path.insert(0, HERE)
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402
from PIL import Image  # noqa: E402
import female  # noqa: E402
F = importlib.import_module(FIGURE)

VIEWS = [('front', 0), ('3/4', 45), ('profile', 90), ('back 3/4', 135), ('back', 180)]


def span(s, n):
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += range(int(a), int(b) + 1)
        else:
            out.append(int(part))
    return [x for x in out if 0 <= x < n]


def cells(out, spec):
    rows_s, cols_s = spec.split(':')
    rows, cols = span(rows_s, female.ROWS), span(cols_s, female.COLS)
    sc = bpy.context.scene
    root = female.FIG['root']
    female.sprite_mode(True)
    PW, PH = female.PW, female.PH
    grid = np.zeros((PH * len(rows), PW * len(cols), 4), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        for j, col in enumerate(cols):
            sc.frame_set(col + 1)
            for i, row in enumerate(rows):
                root.rotation_euler = (0, 0, row * math.pi / 4)
                p = os.path.join(tmp, f'f{row}_{col}.exr')
                sc.render.filepath = p
                bpy.ops.render.render(write_still=True)
                px = female.to_pixels(female.read_exr(p))
                grid[i * PH:(i + 1) * PH, j * PW:(j + 1) * PW] = px
                Image.fromarray(px, 'RGBA').save(os.path.join(out, f'cell-r{row}-c{col}.png'))
                print(f'rendered row {row} col {col}', flush=True)
    root.rotation_euler = (0, 0, 0)
    img = Image.fromarray(grid, 'RGBA')
    img.resize((img.width * 3, img.height * 3), Image.NEAREST).save(os.path.join(out, 'cells.png'))


def look_3d(samples):
    """Switch the scene from the sprite render to a lit 3D look: the close-up materials, a key, fill and rim
    light, a dim backdrop. Returns the camera."""
    sc = bpy.context.scene
    female.sprite_mode(False)
    sc.render.use_persistent_data = False
    sc.cycles.samples = samples
    sc.cycles.use_denoising = False
    sc.cycles.max_bounces = 4
    sc.cycles.pixel_filter_type = 'GAUSSIAN'
    sc.cycles.filter_width = 1.5
    sc.render.film_transparent = False
    if hasattr(sc.render.image_settings, 'media_type'):
        sc.render.image_settings.media_type = 'IMAGE'
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_depth = '8'
    names = [i.identifier for i in sc.view_settings.bl_rna.properties['view_transform'].enum_items]
    sc.view_settings.view_transform = 'AgX' if 'AgX' in names else 'Filmic'
    bg = sc.world.node_tree.nodes.get('Background')
    bg.inputs['Color'].default_value = (0.18, 0.19, 0.21, 1)
    bg.inputs['Strength'].default_value = 0.35
    for o in list(bpy.data.objects):
        if o.type == 'LIGHT':
            bpy.data.objects.remove(o)

    def area(name, loc, energy, size_, colour=(1, 1, 1), aim=(0, 0, 40)):
        d = bpy.data.lights.new(name, 'AREA')
        d.energy, d.size, d.color = energy, size_, colour
        o = bpy.data.objects.new(name, d)
        sc.collection.objects.link(o)
        o.location = loc
        o.rotation_euler = (Vector(aim) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
        return o
    area('Key', (-40, -48, 80), 52000, 24, (1, 0.96, 0.9))
    area('Fill', (46, -36, 50), 12000, 30, (0.85, 0.9, 1))
    area('Rim', (20, 60, 70), 26000, 18)
    cam_d = bpy.data.cameras.new('View')
    cam_d.clip_start, cam_d.clip_end = 1, 1000
    cam = bpy.data.objects.new('View', cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam
    return cam


def row(cam, out, tmp, size, centre, height, ortho=True, dist=200.0):
    sc = bpy.context.scene
    root = female.FIG['root']
    sc.render.resolution_x, sc.render.resolution_y = size
    cam.data.type = 'ORTHO' if ortho else 'PERSP'
    if ortho:
        cam.data.ortho_scale = height
        cam.data.sensor_fit = 'VERTICAL'
    else:
        cam.data.lens = 85
    tiles = []
    for i, (name, yaw) in enumerate(VIEWS):
        root.rotation_euler = (0, 0, math.radians(yaw))
        cam.location = Vector(centre) + Vector((0, -dist, 0))
        cam.rotation_euler = (math.pi / 2, 0, 0)
        p = os.path.join(tmp, f'v{i}.png')
        sc.render.filepath = p
        bpy.ops.render.render(write_still=True)
        tiles.append(np.array(Image.open(p).convert('RGB')))
        print('rendered', name, flush=True)
    root.rotation_euler = (0, 0, 0)
    Image.fromarray(np.concatenate(tiles, axis=1)).save(out)
    print('wrote', out)


def clay(on, saved):
    """Every mesh in one matte grey (on), or back as it was (off)."""
    if on:
        m = bpy.data.materials.get('clay') or bpy.data.materials.new('clay')
        m.use_nodes = True
        b = m.node_tree.nodes.get('Principled BSDF')
        b.inputs['Base Color'].default_value = (0.42, 0.40, 0.38, 1)
        b.inputs['Roughness'].default_value = 0.55
        for o in bpy.data.objects:
            if o.type != 'MESH':
                continue
            saved[o.name] = ([s.material for s in o.material_slots], o.hide_render)
            if o.name.startswith(('Wool fuzz', 'Eyeshadow', 'Lashes', 'Brow paint', 'Hair paint')):
                o.hide_render = True
            for s in o.material_slots:
                s.material = m
    else:
        for name, (mats, hide) in saved.items():
            o = bpy.data.objects[name]
            for s, mm in zip(o.material_slots, mats):
                s.material = mm
            o.hide_render = hide
        saved.clear()


def main():
    out = opt('--out', os.path.join(tempfile.gettempdir(), f'{FIGURE}-views'))
    os.makedirs(out, exist_ok=True)
    for i, a in enumerate(argv):
        if a == '--set':
            k, v = argv[i + 1].split('=', 1)
            setattr(F, k, eval(v))
    F.build()
    sc = bpy.context.scene
    sc.frame_set(1)
    if opt('--cells'):
        sc.render.use_persistent_data = True
        cells(out, opt('--cells'))
    if '--turn' in argv or '--head' in argv:
        sc.frame_set(int(opt('--frame', 0)) + 1)
        size = int(opt('--size', 360))
        cam = look_3d(int(opt('--samples', 32)))
        saved = {}
        with tempfile.TemporaryDirectory() as tmp:
            for look in ('', '-clay') if '--clay' in argv else ('',):
                clay(look == '-clay', saved)
                if '--turn' in argv:
                    row(cam, os.path.join(out, f'turn{look}.png'), tmp, (size, int(size * 1.75)), (0, 0, 32.5), 68.0)
                if '--head' in argv:
                    row(cam, os.path.join(out, f'head{look}.png'), tmp, (size, size), (0, 0, 55.5), 17.0)
            clay(False, saved)


main()
