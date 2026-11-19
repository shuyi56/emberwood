"""Swatches of the bodice's wool and buttons: each dye in --dyes on a sphere and on a bodice-sized open shell (a waisted
tube), side by side in one picture, under the figure views' own lights and world (as warden/bronze_study.py). Nothing
of the figure is built, so it renders in seconds: the cloth's colour is judged here, not on a figure render.

    python3 blender/redingote_bodice/swatch.py [--out DIR] [--size 300] [--samples 64] [--dyes 1d2436,161b2b] [--dry]

Writes DIR/swatches.png. The first dye is the module's own (DYE), unless --dyes says otherwise."""
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), 'base')]
import bpy  # noqa: E402
import bmesh  # noqa: E402
from mathutils import Vector  # noqa: E402

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]


def opt(flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def shape(name, bm, loc, mat):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    at = me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
    at.data.foreach_set('vector', [c for v in me.vertices for c in (Vector(loc) + v.co)])
    me.materials.append(mat)
    return o


def shell(r=1.6, h=6.0, n=64, rows=40):
    """An open tube about a waist's size, drawn in at its middle as a bodice is."""
    bm = bmesh.new()
    ring = []
    for i in range(rows + 1):
        t = i / rows
        rr = r * (1.0 - 0.18 * math.sin(math.pi * t))
        ring.append([bm.verts.new((rr * math.cos(2 * math.pi * j / n), 0.8 * rr * math.sin(2 * math.pi * j / n), h * (0.5 - t)))
                     for j in range(n)])
    for i in range(rows):
        for j in range(n):
            bm.faces.new((ring[i][j], ring[i + 1][j], ring[i + 1][(j + 1) % n], ring[i][(j + 1) % n]))
    return bm


def main():
    import female
    import redingote_bodice as RB      # (the module only: nothing is built on import)
    dyes = [int(d, 16) for d in opt('--dyes', '%06x' % RB.DYE).split(',')]
    out = opt('--out', os.path.join(tempfile.gettempdir(), 'bodice-wool'))
    os.makedirs(out, exist_ok=True)
    size, samples = int(opt('--size', 300)), int(opt('--samples', 64))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = samples
    sc.cycles.use_denoising = False
    sc.cycles.max_bounces = 4
    names = [i.identifier for i in sc.view_settings.bl_rna.properties['view_transform'].enum_items]
    sc.view_settings.view_transform = 'AgX' if 'AgX' in names else 'Filmic'
    sc.world = bpy.data.worlds.new('World')
    sc.world.use_nodes = True
    bg = sc.world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (0.18, 0.19, 0.21, 1)
    bg.inputs['Strength'].default_value = 0.35
    aim = Vector((0, 0, 40))
    for name, loc, energy, size_, colour in (('Key', (-40, -48, 80), 52000, 24, (1, 0.96, 0.9)),
                                             ('Fill', (46, -36, 50), 12000, 30, (0.85, 0.9, 1)),
                                             ('Rim', (20, 60, 70), 26000, 18, (1, 1, 1))):
        d = bpy.data.lights.new(name, 'AREA')
        d.energy, d.size, d.color = energy, size_, colour
        o = bpy.data.objects.new(name, d)
        sc.collection.objects.link(o)
        o.location = loc
        o.rotation_euler = (aim - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    female.sprite_mode(False)
    step = 4.6
    x0 = -step * (len(dyes) - 1) / 2
    for i, dye in enumerate(dyes):
        RB.DYE = dye
        female.MATS.pop(94, None)
        female.MATS.pop(95, None)
        cloth, btn = RB.materials()
        cloth.name = 'bodice wool %06x' % dye
        x = x0 + step * i
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=1.7)
        shape('Sphere %06x' % dye, bm, (x, 0, 44.2), cloth)
        s = shape('Shell %06x' % dye, shell(), (x, 0, 38.2), cloth)
        s.modifiers.new('thickness', 'SOLIDIFY').thickness = 0.09
        for k in range(3):
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=0.23)
            b = shape('Button %06x %d' % (dye, k), bm, (x, -1.32, 39.8 - 0.9 * k), btn)
            b.scale = (1.0, 0.45, 1.0)
    female.sprite_mode(False)
    cam_d = bpy.data.cameras.new('View')
    cam_d.type = 'ORTHO'
    w = step * len(dyes) + 0.6
    cam_d.ortho_scale = max(w, 12.6)
    cam_d.sensor_fit = 'HORIZONTAL' if w >= 12.6 else 'VERTICAL'
    cam = bpy.data.objects.new('View', cam_d)
    sc.collection.objects.link(cam)
    cam.location = (0, -200, 40.6)
    cam.rotation_euler = (math.pi / 2, 0, 0)
    sc.camera = cam
    sc.render.resolution_x = int(size * len(dyes))
    sc.render.resolution_y = int(size * len(dyes) * 12.6 / w) if w >= 12.6 else int(size * len(dyes))
    if hasattr(sc.render.image_settings, 'media_type'):     # (P004: off multilayer EXR before the format is chosen)
        sc.render.image_settings.media_type = 'IMAGE'
    sc.render.image_settings.file_format = 'PNG'
    print('swatches:', ', '.join('%06x' % d for d in dyes), flush=True)
    if '--dry' in argv:
        return
    sc.render.filepath = os.path.join(out, 'swatches.png')
    bpy.ops.render.render(write_still=True)
    try:
        from PIL import Image, ImageDraw
        im = Image.open(sc.render.filepath).convert('RGB')
        dr = ImageDraw.Draw(im)
        for i, dye in enumerate(dyes):
            dr.text((int(im.width * (i + 0.5) / len(dyes)) - 18, 6), '#%06x' % dye, fill=(235, 235, 235))
        im.save(sc.render.filepath)
    except ImportError:
        pass


main()
