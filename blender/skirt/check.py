"""Checks of the skirt on a body that is not the default one, and through the walk, from one build.

    python3 blender/skirt/check.py --out DIR [--walk 8] [--angles 3/4,profile] [--size 300] [--samples 12]
    python3 blender/skirt/check.py --out DIR --pattern --walk 0      (the flat pattern pieces and their paired edges)
    SKIRT_SETTLE=1 python3 blender/skirt/check.py --out DIR --solve  (every walk frame solved against the posed legs, cached)
    CANON_H=10.4 WAIST_W=4.6 python3 blender/skirt/check.py --out DIR          (an altered body: see anatomy.py)

The build is the skirt's own (skirt.build(), on the mannequin). The numbers it prints are the checks: the measures
the skirt was drafted from, and for every walk frame how far the cloth stands off the legs, the hands and the floor
(`inside` counts the cloth's points that are inside the body or a hand: it has to be 0, and `closest` is the nearest the
cloth comes to the skin). The picture, DIR/walk.png, has a row for each angle and a column for each frame of the
walk (the world's: female.LOCKED_WALK, the planted-feet cycle the world model plays), one cycle in all."""
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
opt = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d  # noqa: E731
import skirt  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402
import female  # noqa: E402
from mathutils import Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402

class check:
    where = {}


ANGLES = {'front': 0, '3/4': 45, 'profile': 90, 'back3/4': 135, 'back': 180}


def lights():
    sc = bpy.context.scene

    def area(name, loc, energy, size, colour, aim):
        ld = bpy.data.lights.new(name, 'AREA')
        ld.energy, ld.size, ld.color = energy, size, colour
        o = bpy.data.objects.new(name, ld)
        sc.collection.objects.link(o)
        o.location = loc
        o.rotation_euler = (Vector(aim) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
        return o
    area('Key', (-45, -70, 80), 30000, 60, (1.0, 0.96, 0.9), (0, 0, 28))
    area('Fill', (60, -50, 40), 9000, 60, (0.85, 0.9, 1.0), (0, 0, 28))
    area('Rim', (20, 70, 70), 12000, 40, (1, 1, 1), (0, 0, 28))
    w = bpy.data.worlds.new('w')
    w.use_nodes = True
    w.node_tree.nodes['Background'].inputs[0].default_value = (0.3, 0.3, 0.32, 1)
    sc.world = w
    cd = bpy.data.cameras.new('V')
    cd.type, cd.sensor_fit = 'ORTHO', 'VERTICAL'
    cam = bpy.data.objects.new('V', cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    return cam


def clearance(sk, body, hands):
    """How the cloth stands against the body and hands at this frame: the points inside them (below the hip joint, and
    all of them), the nearest to the skin, and the lowest point."""
    dg = bpy.context.evaluated_depsgraph_get()
    e = sk.evaluated_get(dg)
    me = e.to_mesh()
    pts = [sk.matrix_world @ v.co for v in me.vertices]
    e.to_mesh_clear()
    verts, polys = [], []
    for o in [body] + hands:
        oe = o.evaluated_get(dg)
        m = oe.to_mesh()
        n0 = len(verts)
        verts += [o.matrix_world @ v.co for v in m.vertices]
        polys += [tuple(n0 + i for i in p.vertices) for p in m.polygons]
        oe.to_mesh_clear()
    tree = BVHTree.FromPolygons(verts, polys)
    inside, low_inside, closest = 0, 0, 1e9
    where = {}
    for p in pts[::2]:
        loc, nrm, _, d = tree.find_nearest(p, 3.0)
        if loc is None:
            continue
        if (p - loc).dot(nrm) < 0:
            inside += 1
            low_inside += p.z < skirt.MEASURES['hip_z']
            b = int(p.z // 5) * 5
            n_, dm = where.get(b, (0, 0.0))
            where[b] = (n_ + 1, max(dm, d))
        closest = min(closest, d)
    check.where = where
    return inside, low_inside, closest, min(p.z for p in pts)


def main():
    out = opt('--out', os.path.join(tempfile.gettempdir(), 'skirt-check'))
    os.makedirs(out, exist_ok=True)
    n = int(opt('--walk', 8))
    size = int(opt('--size', 300))
    names = [a for a in opt('--angles', '3/4,profile').split(',') if a != 'none']
    female.MANNEQUIN = True
    female.LOCKED_WALK, female.ARM_SWING = True, 0.24        # (the world model's walk, as export_glb.py sets it)
    skirt.build()
    if '--pattern' in argv:
        # (the flat pattern pieces and their paired edges: drafted from the same skirt, nothing sewn)
        import pattern
        pieces, pairs, extra = pattern.draft_pieces(skirt.DRAFT['D'], skirt.DRAFT['M'])
        pattern.report(pieces, pairs)
        print('wrote', pattern.flat_sheet(pieces, os.path.join(out, 'pattern.png')))
        if n == 0:
            return
    sc = bpy.context.scene
    sk = bpy.data.objects['Riding skirt']
    body = bpy.data.objects['Skin body']
    hands = [o for o in (bpy.data.objects.get('Hand.L'), bpy.data.objects.get('Hand.R')) if o]
    # (what the sheet keyed on the timeline: the walk is posed by hand here, a frame at a time)
    rest = '--rest' in argv                     # (the pose the skirt was built and settled in: frame 1, as views.py shows it)
    if rest:
        sc.frame_set(1)
    for o in sc.objects:
        if o.animation_data and not rest:
            o.animation_data_clear()
    cam = lights()
    sc.render.engine = 'CYCLES'
    female.sprite_mode(False)                    # (the 3D look: the close-up materials, as views.py's look_3d)
    sc.render.film_transparent = False
    if hasattr(sc.render.image_settings, 'media_type'):
        sc.render.image_settings.media_type = 'IMAGE'
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_depth = '8'
    sc.cycles.max_bounces = 4
    sc.view_settings.view_transform = 'AgX'
    sc.cycles.samples = int(opt('--samples', 12))
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = size, int(size * 1.75)
    root = female.FIG['root']
    rest0 = [v.co.copy() for v in sk.data.vertices]
    rig_obj = bpy.data.objects['Rig']
    for o in ('Riding skirt stitches', 'Riding skirt seams'):
        if '--solve' in argv and bpy.data.objects.get(o):
            bpy.data.objects[o].hide_render = True
    floor = skirt.MEASURES['floor_z']
    cam.data.ortho_scale = (64.0 - floor) * 1.06
    from PIL import Image
    grid = [[None] * n for _ in names]
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(n):
            if not rest:
                female.pose(2 - 0.5 + female.WALK + k * female.WALK / n)
            bpy.context.view_layer.update()
            if '--solve' in argv:
                # (the cloth solved against the legs as they stand in this pose, from the rest drape: John's per-pose solve)
                import settle
                for v, co in zip(sk.data.vertices, rest0):
                    v.co = co
                sk.data.update()
                M, H = skirt.MEASURES, skirt.MEASURES['H']

                def stand_off(p):
                    t = min(1.0, max(0.0, (p[2] - (M['crotch_z'] + 0.1 * H)) / (-0.3 * H)))
                    t = t * t * (3 - 2 * t)
                    return 0.012 * H + (0.06 * H - 0.012 * H) * t
                settle.settle_xpbd(sk, rig_obj, [body] + hands, floor, pin=(M['hips_z'] - 0.2 * H, M['hips_z']), gap=stand_off,
                                   floor_hard=True)
                bpy.context.view_layer.update()
            if '--stacks' in argv:
                # (the push's own work: the same frame with the modifiers that move the cloth on, one more at a time)
                for stack in (['rig'], ['rig', 'push', 'push hips'], ['rig', 'push', 'push hips', 'push smooth', 'push again', 'push hips again', 'floor']):
                    for m in sk.modifiers:
                        m.show_viewport = m.name in stack or m.name in ('thickness', 'smooth')
                    bpy.context.view_layer.update()
                    r = clearance(sk, body, hands)
                    print('    modifiers %-30s inside %d (%d below the hip joint), deepest below z 10: %s' % (
                        '+'.join(stack), r[0], r[1], {b: round(d, 2) for b, (a, d) in sorted(check.where.items()) if b < 10}),
                        flush=True)
                for m in sk.modifiers:
                    m.show_viewport = True
                bpy.context.view_layer.update()
            ins, lins, near, low = clearance(sk, body, hands)
            print('walk frame %d: cloth inside the body/hands %d points (%d below the hip joint), closest %.3f off the '
                  'skin, lowest %.2f (floor %.2f)' % (k, ins, lins, near, low, floor), flush=True)
            print('    inside by height (z: points, deepest): ' + ', '.join(
                '%d: %d, %.2f' % (b, a, d) for b, (a, d) in sorted(check.where.items())), flush=True)
            for i, nm in enumerate(names):
                root.rotation_euler = (0, 0, math.radians(ANGLES[nm]))
                cam.location = Vector((0, -200, (64.0 + floor) / 2))
                cam.rotation_euler = (math.pi / 2, 0, 0)
                p = os.path.join(tmp, 'w%d-%d.png' % (k, i))
                sc.render.filepath = p
                bpy.ops.render.render(write_still=True)
                grid[i][k] = Image.open(p).convert('RGB')
    if not names:
        return
    w, h = grid[0][0].size
    sheet = Image.new('RGB', (w * n, h * len(names)))
    for i, frames in enumerate(grid):
        for k, im in enumerate(frames):
            sheet.paste(im, (k * w, i * h))
    sheet.save(os.path.join(out, 'walk.png'))
    print('wrote', os.path.join(out, 'walk.png'))


main()
