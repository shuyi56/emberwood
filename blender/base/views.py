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
           show (DIR/turn-clay.png, DIR/head-clay.png). --clay-only: those alone, the look not rendered (a cheap
           draft of the forms).
--angles   NAMES: render --turn, --head and --pieces from only these angles (front, 3/4, profile, back3/4, back;
           comma separated) instead of all five. One angle first, the full turn once that view is confirmed.
--mannequin  build the figure's body and a plain head only: no eyes, lids, ears, face paint, mouth or hair (most of
           a build), with the clothes, armour and gear built and fitted exactly as on the real figure. For clothes
           changes that need a body under them.
--walk N   the world's walk (female.LOCKED_WALK, the planted-feet cycle the world model plays) in the 3D look:
           N evenly spaced frames of one cycle from each of --angles, on a ground plane, laid out with a row per angle
           (DIR/walk.png) and played as a loop per angle (DIR/walk-ANGLE.gif, one cycle in --walk-seconds, default
           0.69 s: the world's pace, 2.2 a second over a cycle of 46 model units). Use it with --mannequin.
--run N    the run (female.RUN: see pose_run) the same way, in a wider frame for its longer reach: DIR/run.png and
           DIR/run-ANGLE.gif, one cycle in --run-seconds (default female.RUN_SECONDS).
--set      NAME=VALUE: set a module setting of the character before it builds (e.g. FUSE=False), repeatable; a base
           figure setting such as SKIN_LOOK=5 (by its place in female.SKIN_LOOKS) or AHEAD=0.4 is set on the base
           module too
--denoise  with --turn or --head: denoise the 3D look (OpenImageDenoise, albedo and normal passes)

Rendering pieces on their own (any of the views above):
--only     PREFIXES: render only the meshes whose names start with one of these (comma separated), e.g.
           --only 'Gorget,Cape' or --only Bracer; everything else is hidden, so the pieces show with nothing in front
           of them or under them. Matching is case-sensitive and a prefix that matches nothing is an error.
--hide     PREFIXES: hide these meshes and render the rest
--no-body  hide the base figure itself (the body, head, face, eyes and hair: every mesh in a base material, below
           FIGURE_MATS, and every mesh parented under the 'Head' empty: the face's paints and the hair), leaving the
           clothes and gear standing as if on a dress form. So head-worn gear (a hood, circlet, earring, mask) must be
           parented elsewhere, or it is hidden too.
--fit      frame --turn and --head on what is left showing instead of the whole figure (the default with --only);
           --no-fit keeps the usual framing
--pieces   PREFIXES: after the other views, from the same build, the same five-angle turn of just these meshes
           framed on them (DIR/pieces.png): the full figure and its pieces alone for the cost of one build
--list     print every mesh's name, material and vertex count after the build, and stop
--save     PATH: with --turn or --head, save the built figure with the 3D look's camera and lights as a .blend
           instead of rendering
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
if opt('--angles'):
    _want = [a.strip().replace(' ', '') for a in opt('--angles').split(',') if a.strip()]
    _bad = [a for a in _want if a not in [n.replace(' ', '') for n, _ in VIEWS]]
    if _bad or not _want:
        sys.exit(f'views.py: --angles takes front, 3/4, profile, back3/4, back (got {",".join(_bad) or "nothing"})')
    VIEWS = [v for v in VIEWS if v[0].replace(' ', '') in _want]


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
    # (--denoise: OpenImageDenoise with the albedo and normal passes. Without it the skin's subsurface scattering
    # stays grainy at these sample counts, which reads as a stucco texture that is not in the material)
    sc.cycles.use_denoising = '--denoise' in argv
    if sc.cycles.use_denoising:
        sc.cycles.denoiser = 'OPENIMAGEDENOISE'
        sc.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
    sc.cycles.max_bounces = 4
    sc.cycles.transparent_max_bounces = int(os.environ.get('TRANSP_BOUNCES', '16'))     # (cut-out hair: a ray through overlapping strand tips crosses many walls; at 8 it ended black)
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
    # light off the floor: big, warm and from below, so the shaded undersides (legs, jaw, the far side of the body)
    # keep some colour instead of going grey. It is not seen by the camera.
    area('Bounce', (-15, -70, -60), 30000, 120, (1.0, 0.72, 0.6), aim=(0, 0, 20)).visible_camera = False
    # (a figure may dim or raise a light for its own views, by name: LIGHT_SCALE = {'Bounce': 0.25}. None does by default.)
    for name, k in getattr(F, 'LIGHT_SCALE', {}).items():
        bpy.data.objects[name].data.energy *= k
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


FIGURE_MATS = 30   # (the base figure's own materials have ids below this; every outfit's start at it)


def prefixes(flag):
    v = opt(flag)
    return tuple(p.strip() for p in v.split(',') if p.strip()) if v else ()


_ISOLATED = set()    # (the meshes isolate() itself hid)


def isolate(only=None):
    """Hide what --only, --hide and --no-body leave out (hide_render: it neither shows nor casts shadows).
    Returns the meshes still showing."""
    if only is not None:
        # (--pieces picks from the whole figure, not from what --only left showing)
        for o in bpy.data.objects:
            if o.type == 'MESH' and o.name in _ISOLATED:
                o.hide_render = False
    only, hide = (prefixes('--only') if only is None else only), prefixes('--hide')
    ids = {d['name']: mid for mid, d in female.MATERIALS.items()}
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    for p in only + hide:
        if not any(o.name.startswith(p) for o in meshes):
            sys.exit(f'views.py: no mesh starts with {p!r} (see --list)')
    shown = []
    for o in meshes:
        m = o.active_material
        base = m is not None and ids.get(m.name.split('.')[0], FIGURE_MATS) < FIGURE_MATS
        # (and all the head carries: the face's paints and the hair have materials of their own, not the figure's)
        q = o
        while q.parent is not None and q.parent.name != 'Head':
            q = q.parent
        base = base or q.parent is not None
        off = (only and not o.name.startswith(only)) or (hide and o.name.startswith(hide)) or \
            ('--no-body' in argv and base)
        if off and not o.hide_render:
            _ISOLATED.add(o.name)
        o.hide_render = bool(off) or o.hide_render
        if not o.hide_render:
            shown.append(o)
    if not shown:
        sys.exit('views.py: nothing left to render')
    print('showing', len(shown), 'meshes:', ', '.join(sorted(o.name for o in shown)[:40]), flush=True)
    return shown


def fit(shown, aspect):
    """(centre, ortho height) that frame the shown meshes from every angle the views turn the figure through:
    the figure turns about its own vertical axis, so the width needed is twice the farthest any point lies from it."""
    dg = bpy.context.evaluated_depsgraph_get()
    zs, rr = [], 0.0
    for o in shown:
        me = o.evaluated_get(dg).to_mesh()
        mw = o.matrix_world
        for v in me.vertices:
            c = mw @ v.co
            zs.append(c.z)
            rr = max(rr, math.hypot(c.x, c.y))
        o.evaluated_get(dg).to_mesh_clear()
    z0, z1 = min(zs), max(zs)
    h = max((z1 - z0) * 1.12, 2.0 * rr * 1.12 / aspect, 4.0)
    return (0, 0, (z0 + z1) / 2), h


def list_meshes():
    for o in sorted((o for o in bpy.data.objects if o.type == 'MESH'), key=lambda o: o.name):
        m = o.active_material
        print(f'{o.name:40s} {m.name if m else "-":24s} {len(o.data.vertices)}')


def walk(n, out, stem='walk', seconds=0.69, wide=1.0):
    """N frames of the world's walk cycle (or, with female.RUN set, the run: stem 'run') from each of VIEWS (see
    --walk and --run), on a ground plane. `wide` widens the frame for a gait that reaches further."""
    sc = bpy.context.scene
    size = int(opt('--size', 360))
    cam = look_3d(int(opt('--samples', 32)))
    for o in sc.objects:          # (the sheet's columns are keyed on the timeline: pose each frame by hand instead)
        if o.animation_data:
            o.animation_data_clear()
    floor = bpy.data.meshes.new('Walk floor')
    fz = female.ANKLE_Z - getattr(female, 'SOLE_UNDER_ANKLE', 3.4)   # (the sole is 3.4 under the ankle, or the measured foot's ankle height: the floor, wherever the figure's legs put it)
    floor.from_pydata([(-60, -60, fz), (60, -60, fz), (60, 60, fz), (-60, 60, fz)], [], [(0, 1, 2, 3)])
    fo = bpy.data.objects.new('Walk floor', floor)
    sc.collection.objects.link(fo)
    fm = bpy.data.materials.new('walk floor')
    fm.use_nodes = True
    fm.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value = (0.32, 0.31, 0.29, 1)
    fo.data.materials.append(fm)
    root = female.FIG['root']
    sc.render.resolution_x, sc.render.resolution_y = int(size * wide), int(size * 1.75)
    cam.data.type, cam.data.ortho_scale, cam.data.sensor_fit = 'ORTHO', (64.0 - fz) * 1.06, 'VERTICAL'
    grid = [[None] * n for _ in VIEWS]
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(n):
            female.pose(2 - 0.5 + female.WALK + k * female.WALK / n)
            for i, (name, yaw) in enumerate(VIEWS):
                root.rotation_euler = (0, 0, math.radians(yaw))
                cam.location = Vector((0, -200, (64.0 + fz) / 2))
                cam.rotation_euler = (math.pi / 2, 0, 0)
                p = os.path.join(tmp, f'w{k}-{i}.png')
                sc.render.filepath = p
                bpy.ops.render.render(write_still=True)
                grid[i][k] = Image.open(p).convert('RGB')
                print(f'rendered {stem} frame {k} {name}', flush=True)
    root.rotation_euler = (0, 0, 0)
    w, h = grid[0][0].size
    sheet = Image.new('RGB', (w * n, h * len(VIEWS)))
    for i, frames in enumerate(grid):
        for k, im in enumerate(frames):
            sheet.paste(im, (k * w, i * h))
        ms = int(round(1000 * float(opt(f'--{stem}-seconds', seconds)) / n))
        frames[0].save(os.path.join(out, f'{stem}-{VIEWS[i][0].replace("/", "").replace(" ", "")}.gif'), save_all=True,
                       append_images=frames[1:], duration=max(20, ms), loop=0)
    sheet.save(os.path.join(out, f'{stem}.png'))
    print('wrote', os.path.join(out, f'{stem}.png'))


def main():
    out = opt('--out', os.path.join(tempfile.gettempdir(), f'{FIGURE}-views'))
    os.makedirs(out, exist_ok=True)
    for i, a in enumerate(argv):
        if a == '--set':
            k, v = argv[i + 1].split('=', 1)
            setattr(F, k, eval(v))
            if hasattr(female, k):
                setattr(female, k, eval(v))     # (a base figure setting, such as SKIN_LOOK or the walk's)
    if opt('--walk') or opt('--run'):
        female.LOCKED_WALK, female.ARM_SWING = True, 0.24     # (as export_glb.py sets them for the world model)
    if '--mannequin' in argv:
        female.MANNEQUIN = True
    import time
    t0 = time.time()
    F.build()
    print(f'build {time.time() - t0:.0f} s' + (' (mannequin)' if female.MANNEQUIN else ''), flush=True)
    sc = bpy.context.scene
    sc.frame_set(1)
    if '--list' in argv:
        list_meshes()
        return
    isolating = bool(opt('--only') or opt('--hide') or '--no-body' in argv)
    shown = isolate() if isolating else None
    framed = shown is not None and '--no-fit' not in argv and ('--fit' in argv or opt('--only'))
    if opt('--cells'):
        sc.render.use_persistent_data = True
        cells(out, opt('--cells'))
    if '--turn' in argv or '--head' in argv:
        sc.frame_set(int(opt('--frame', 0)) + 1)
        size = int(opt('--size', 360))
        cam = look_3d(int(opt('--samples', 32)))
        if isolating:
            # (again: the close-up look shows the face's paints, which the sprite look hides, over what was hidden)
            shown = isolate()
        if opt('--save'):
            # (the built figure with the 3D look's camera and lights, as a scene file, instead of rendering)
            sb = bpy.data.objects.get('Skin body')
            if sb is not None:
                print('save: Skin body attributes', [(x.name, x.domain, x.data_type) for x in sb.data.attributes], flush=True)
            print('save: objects', [(o.name, o.type) for o in bpy.data.objects if o.type in ('CAMERA', 'LIGHT', 'EMPTY') and o.parent is None], flush=True)
            bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(opt('--save')), compress=True)
            print('saved', opt('--save'), bpy.app.version_string, flush=True)
            return
        saved = {}
        with tempfile.TemporaryDirectory() as tmp:
            looks = ('-clay',) if '--clay-only' in argv else ('', '-clay') if '--clay' in argv else ('',)
            for look in looks:
                clay(look == '-clay', saved)
                if '--turn' in argv:
                    centre, h = fit(shown, 1 / 1.75) if framed else ((0, 0, 32.5), 68.0)
                    row(cam, os.path.join(out, f'turn{look}.png'), tmp, (size, int(size * 1.75)), centre, h)
                if '--head' in argv:
                    centre, h = fit(shown, 1.0) if framed else ((float(os.environ.get('HEAD_CX', 0)), float(os.environ.get('HEAD_CY', 0)), float(os.environ.get('HEAD_CZ', 55.5))), float(os.environ.get('HEAD_H', 17.0)))
                    row(cam, os.path.join(out, f'head{look}.png'), tmp, (size, size), centre, h)
            clay(False, saved)
    if opt('--walk'):
        walk(int(opt('--walk')), out)
    if opt('--run'):
        female.RUN = True
        walk(int(opt('--run')), out, 'run', female.RUN_SECONDS, 1.45)
        female.RUN = False
    if opt('--pieces'):
        if not ('--turn' in argv or '--head' in argv):
            cam = look_3d(int(opt('--samples', 32)))
            sc.frame_set(int(opt('--frame', 0)) + 1)
        size = int(opt('--size', 360))
        shown = isolate(prefixes('--pieces'))
        centre, h = fit(shown, 1 / 1.75)
        with tempfile.TemporaryDirectory() as tmp:
            row(cam, os.path.join(out, 'pieces.png'), tmp, (size, int(size * 1.75)), centre, h)


main()
