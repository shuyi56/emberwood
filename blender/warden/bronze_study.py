"""Swatches of the warden's bronze: every look in BRONZE_LOOKS on a sphere and on a greave-sized open shell, side by
side in one picture, under the figure views' own lights and world. Nothing of the figure is built, so it renders in
seconds: a material change is judged here, not on a figure render.

    python3 blender/warden/bronze_study.py [--out DIR] [--size 300] [--samples 64] [--looks browned,russet] [--dry]

Writes DIR/swatches.png (default DIR: the system temp dir + 'warden-bronze'). --dry builds the scene and stops
before rendering."""
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
    # ('rest': the figure's meshes carry their rest positions for the procedural textures to ride on)
    at = me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
    at.data.foreach_set('vector', [c for v in me.vertices for c in (Vector(loc) + v.co)])
    me.materials.append(mat)
    return o


def shell(r=1.3, h=6.0, n=64, rows=40):
    """An open tube the size of a greave, a little waisted, its rims turned out, as the greave's are."""
    bm = bmesh.new()
    ring = []
    for i in range(rows + 1):
        t = i / rows
        z = h * (0.5 - t)
        rr = r * (1.0 - 0.12 * math.sin(math.pi * t)) + 0.14 * (max(0.0, 1 - t / 0.1) ** 2 + max(0.0, 1 - (1 - t) / 0.1) ** 2)
        ring.append([bm.verts.new((rr * math.cos(2 * math.pi * j / n), rr * math.sin(2 * math.pi * j / n), z))
                     for j in range(n)])
    for i in range(rows):
        for j in range(n):
            bm.faces.new((ring[i][j], ring[i + 1][j], ring[i + 1][(j + 1) % n], ring[i][(j + 1) % n]))
    return bm


def main():
    import warden as W      # (the module only: its build runs under __main__)
    looks = [k.strip() for k in opt('--looks', ','.join(W.BRONZE_LOOKS)).split(',')]
    out = opt('--out', os.path.join(tempfile.gettempdir(), 'warden-bronze'))
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
    # (the world and the three lights of views.py's look_3d, aimed at the swatches as they are at the figure)
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
    step = 4.6
    x0 = -step * (len(looks) - 1) / 2
    for i, look in enumerate(looks):
        m = bpy.data.materials.new(f'bronze {look}')
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        o_ = nt.nodes.new('ShaderNodeOutputMaterial')
        d_ = nt.nodes.new('ShaderNodeBsdfDiffuse')
        nt.links.new(d_.outputs['BSDF'], o_.inputs['Surface'])
        W.bronze_closeup(m.name, look)
        x = x0 + step * i
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=1.7)
        shape(f'Sphere {look}', bm, (x, 0, 44.2), m)
        s = shape(f'Shell {look}', shell(), (x, 0, 38.2), m)
        s.modifiers.new('thickness', 'SOLIDIFY').thickness = 0.14
        s.rotation_euler = (0, 0, math.radians(25))
    cam_d = bpy.data.cameras.new('View')
    cam_d.type = 'ORTHO'
    cam_d.ortho_scale = max(step * len(looks) + 0.6, 12.6 * 1.0)
    cam = bpy.data.objects.new('View', cam_d)
    sc.collection.objects.link(cam)
    cam.location = (0, -200, 40.6)
    cam.rotation_euler = (math.pi / 2, 0, 0)
    sc.camera = cam
    w = step * len(looks) + 0.6
    cam_d.ortho_scale = max(w, 12.6)
    sc.render.resolution_x = int(size * len(looks))
    sc.render.resolution_y = int(size * len(looks) * 12.6 / max(w, 12.6)) if w >= 12.6 else int(size * len(looks))
    cam_d.sensor_fit = 'HORIZONTAL' if w >= 12.6 else 'VERTICAL'
    sc.render.image_settings.file_format = 'PNG'
    print('swatches:', ', '.join(looks), flush=True)
    if '--dry' in argv:
        return
    sc.render.filepath = os.path.join(out, 'swatches.png')
    bpy.ops.render.render(write_still=True)
    try:
        from PIL import Image, ImageDraw
        im = Image.open(sc.render.filepath).convert('RGB')
        dr = ImageDraw.Draw(im)
        for i, look in enumerate(looks):
            dr.text((int(im.width * (i + 0.5) / len(looks)) - 3 * len(look), 6), look, fill=(235, 235, 235))
        im.save(sc.render.filepath)
    except ImportError:
        pass


main()
