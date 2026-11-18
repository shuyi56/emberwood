"""Swatches of the skin: every look in female.SKIN_LOOKS side by side in one picture, each on three stand-ins at the
figure's own scale (one unit is about 2.7 cm): a head-sized ball (the cheek and brow), a thin ear-sized lobe and a
finger. They stand under the figure views' own world and key, fill and rim lights (views.py look_3d), which come
from behind the shapes as well as in front, so the scattering shows through the lobe and the finger. Nothing of a
figure is built, so it renders in seconds: a skin material change is judged here, not on a figure render.

    python3 blender/base/skin_study.py [--out DIR] [--size 300] [--samples 64] [--looks current,oil] [--denoise]
                                       [--tone fair] [--vs-main] [--dry]

--layers   instead of the looks: the skin built in three layers, one column each, then all together (see
           skin_layers.py): 1 the muscle and capillary base, 2 the veins over it, 3 the epidermis alone, and the
           three combined on a subsurface surface
--tone     the skin's base colour: a face creator swatch (ranger, fair, light, olive, tan, brown, deep) or a hex
           colour such as efcfb9; default the figures' own (female.SKIN[2])

Writes DIR/swatches.png (default DIR: the system temp dir + 'skin-study'), and with --denoise also
DIR/swatches-denoised.png from the same scene, to show how much of the grain is render noise. --dry builds the
scene and stops before rendering.

--vs-main  the regression check: the 'current' look rendered once with this checkout's skin_surface and once,
           in the same scene with the same samples, with origin/main's (read with git show), into
           DIR/vs-main.png (branch | main | the difference, scaled up 8 times); the largest and mean difference
           are printed. 0 means 'current' reproduces main's skin.

The ball carries the skin's maps the way a face does: oily at the top (the brow and nose), drier at the bottom,
pored all over, a light scatter of freckles on its right half, flushed a little in the middle. The finger is
creased round its joints."""
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bpy  # noqa: E402
import bmesh  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]


def opt(flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


# the face creator's skin swatches (SKIN_SWATCHES in its main.js)
TONES = {'ranger': 'cf9f84', 'fair': 'efcfb9', 'light': 'e2b597', 'olive': 'c39a6e', 'tan': 'b07a55',
         'brown': '8a573a', 'deep': '5c3826'}
TONE = None


def tone_rgb(t):
    h = TONES.get(t.lower(), t).lstrip('#')
    if len(h) != 6 or any(ch not in '0123456789abcdefABCDEF' for ch in h):
        sys.exit(f'skin_study.py: --tone takes {", ".join(TONES)} or a hex colour (got {t})')
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def shape(name, bm, loc, mat, maps, scale=(1, 1, 1), rot=(0, 0, 0)):
    """A stand-in with the skin's point colours: maps(local co) -> (flush, shade, gloss, freckles, creases, pores)."""
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    o.location, o.scale, o.rotation_euler = loc, scale, rot
    bpy.context.view_layer.update()
    co = [o.matrix_world @ v.co for v in me.vertices]
    # ('rest': the figure's meshes carry their rest positions for the procedural textures to ride on)
    me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT').data.foreach_set('vector', [c for v in co for c in v])
    import female
    base = female._srgb_lin(TONE or female.SKIN[2])
    vals = np.array([maps(v.co) for v in me.vertices], dtype=np.float64)
    skin = np.tile([*base, 1.0], (len(co), 1))
    fx = np.column_stack([vals[:, 0], vals[:, 1], vals[:, 2], np.ones(len(co))])
    tx = np.column_stack([vals[:, 3], vals[:, 4], vals[:, 5], np.ones(len(co))])
    for nm, arr in (('skin', skin), ('skin fx', fx), ('skin tex', tx)):
        me.color_attributes.new(nm, 'FLOAT_COLOR', 'POINT').data.foreach_set('color', arr.ravel())
    me.materials.append(mat)
    return o


def ball_maps(c):
    up = max(0.0, min(1.0, 0.5 + c.z / 2.0))      # (c: the ball's own unit-sphere coordinates)
    return (0.35 * math.exp(-(c.z / 0.5) ** 2), 0.0, 0.15 + 0.75 * up,
            0.8 if c.x > 0.15 else 0.0, 0.0, 1.0)


def vs_main(sc, out, female):
    import importlib.util
    import subprocess
    root = os.path.dirname(os.path.dirname(HERE))
    src = subprocess.run(['git', '-C', root, 'show', 'origin/main:blender/base/female.py'], check=True,
                         capture_output=True, text=True).stdout
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'female_main.py')
    with open(path, 'w') as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location('female_main', path)
    fm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fm)
    if '--dry' in argv:
        return
    from PIL import Image
    shots = []
    for who in ('branch', 'main'):
        if who == 'main':
            m = bpy.data.materials['skin current']
            nt = m.node_tree
            nt.nodes.clear()
            o_ = nt.nodes.new('ShaderNodeOutputMaterial')
            d_ = nt.nodes.new('ShaderNodeBsdfDiffuse')
            nt.links.new(d_.outputs['BSDF'], o_.inputs['Surface'])
            fm.skin_surface(nt, d_, o_)
        sc.render.filepath = os.path.join(tmp, f'{who}.png')
        bpy.ops.render.render(write_still=True)
        shots.append(np.asarray(Image.open(sc.render.filepath).convert('RGB')).astype(np.int16))
    diff = np.abs(shots[0] - shots[1])
    print(f'vs main: largest difference {int(diff.max())} of 255, mean {diff.mean():.4f}', flush=True)
    row = np.concatenate([shots[0], shots[1], np.minimum(255, diff * 8)], axis=1).astype(np.uint8)
    Image.fromarray(row).save(os.path.join(out, 'vs-main.png'))


def main():
    import female       # (the module only: nothing of a figure is built)
    looks = [k.strip() for k in opt('--looks', ','.join(female.SKIN_LOOKS)).split(',')]
    if '--vs-main' in argv:
        looks = ['current']
    if '--layers' in argv:
        looks = ['1 base', '2 + veins', '3 epidermis', 'all three']
    bad = [k for k in looks if k not in female.SKIN_LOOKS and '--layers' not in argv]
    if bad:
        sys.exit(f'skin_study.py: no such look {", ".join(bad)} (looks: {", ".join(female.SKIN_LOOKS)})')
    global TONE
    TONE = tone_rgb(opt('--tone')) if opt('--tone') else None
    out = opt('--out', os.path.join(tempfile.gettempdir(), 'skin-study'))
    os.makedirs(out, exist_ok=True)
    size, samples = int(opt('--size', 300)), int(opt('--samples', 64))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = samples
    sc.cycles.use_denoising = False
    sc.cycles.max_bounces = 4
    sc.cycles.pixel_filter_type = 'GAUSSIAN'
    sc.cycles.filter_width = 1.5
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
    step = 7.6
    x0 = -step * (len(looks) - 1) / 2
    for i, look in enumerate(looks):
        m = bpy.data.materials.new(f'skin {look}')
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        o_ = nt.nodes.new('ShaderNodeOutputMaterial')
        d_ = nt.nodes.new('ShaderNodeBsdfDiffuse')
        nt.links.new(d_.outputs['BSDF'], o_.inputs['Surface'])
        if '--layers' in argv:
            import skin_layers
            skin_layers.skin_layers(nt, o_, ('1', '2', '3', 'all')[i], TONE or female.SKIN[2])
        else:
            female.skin_surface(nt, d_, o_, look)
        x = x0 + step * i
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=96, v_segments=48, radius=1.0)
        shape(f'Ball {look}', bm, (x, 0, 44.6), m, ball_maps, scale=(3.0, 3.0, 3.3))
        # an ear's rim: a thin lobe, 0.2 thick (about 5 mm), turned a little to the light behind it
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=1.0)
        shape(f'Lobe {look}', bm, (x - 1.7, 0, 37.6), m, lambda c: (0.5, 0.0, 0.3, 0.0, 0.0, 0.4),
              scale=(1.3, 0.1, 2.0), rot=(0, 0, math.radians(-30)))
        # a finger, 0.65 across (about 1.8 cm), creased at two joints
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=64, radius=1.0)
        shape(f'Finger {look}', bm, (x + 1.9, 0, 37.6), m,
              lambda c: (0.6, 0.0, 0.2, 0.0, max(0.0, 1 - abs(abs(c.z) - 0.35) / 0.12), 0.3),
              scale=(0.33, 0.33, 2.3), rot=(0, math.radians(12), 0))
    cam_d = bpy.data.cameras.new('View')
    cam_d.type = 'ORTHO'
    w, h = step * len(looks), 14.4
    cam_d.ortho_scale = max(w, h)
    cam_d.sensor_fit = 'HORIZONTAL' if w >= h else 'VERTICAL'
    cam = bpy.data.objects.new('View', cam_d)
    sc.collection.objects.link(cam)
    cam.location = (0, -200, 41.4)
    cam.rotation_euler = (math.pi / 2, 0, 0)
    sc.camera = cam
    sc.render.resolution_x = int(size * w / step)
    sc.render.resolution_y = int(size * h / step)
    if hasattr(sc.render.image_settings, 'media_type'):
        sc.render.image_settings.media_type = 'IMAGE'
    sc.render.image_settings.file_format = 'PNG'
    print('swatches:', ', '.join(looks), 'tone', TONE or female.SKIN[2], flush=True)
    if '--vs-main' in argv:
        vs_main(sc, out, female)
        return
    if '--dry' in argv:
        return
    files = ['swatches.png']
    if '--denoise' in argv:
        files.append('swatches-denoised.png')
    for f in files:
        sc.cycles.use_denoising = f.endswith('denoised.png')
        if sc.cycles.use_denoising:
            sc.cycles.denoiser = 'OPENIMAGEDENOISE'
            sc.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
        sc.render.filepath = os.path.join(out, f)
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
