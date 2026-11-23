"""Named view presets, quality tiers, framed close-ups, the matcap look and the label on every image, for views.py.

views.py keeps the flags; this module turns a preset name into those flags, frames a named area of the body from
the figure's own measurements, and labels each image with the quality it was really rendered at. A request stays
one views.py command line, so the render queue's cache and its check quality work as before:

    blender/base/views.py --figure rogue --mannequin --preset turn-clay              (draft unless --tier says)
    blender/base/views.py --figure rogue --preset closeup --area shoulders --tier check

Every image views.py writes through row() (turn, head, area, pieces) and the walk and run sheets gets a strip
under it naming its tier, samples, denoising, look, angles, figure and commit, and DIR/views.json records the same
with the exact camera of each image, which tools/render-queue/rq_compare.py uses to refuse a before/after whose
cameras differ.
"""
import json
import os
import subprocess
import sys

# ------------------------------------------------------------------ presets and tiers
FOUR = 'front,3/4,profile,back'

# name: (what it is, the flags it stands for). Flags given on the command line win over a preset's.
PRESETS = {
    'view-clay':   ('one 3/4 clay view of the whole figure: the first look at a change',
                    ['--turn', '--look', 'clay', '--angles', '3/4']),
    'turn-clay':   ('clay turnaround, orthographic: front, 3/4, profile, back',
                    ['--turn', '--look', 'clay', '--angles', FOUR]),
    'turn-matcap': ('matcap turnaround: unlit, noise-free form at any sample count',
                    ['--turn', '--look', 'matcap', '--angles', FOUR]),
    'turn-lit':    ('the lit 3D look, turnaround', ['--turn', '--look', 'full', '--angles', FOUR]),
    'head-clay':   ('head and shoulders in clay, four angles', ['--head', '--look', 'clay', '--angles', FOUR]),
    'head-lit':    ('head and shoulders, lit, one 3/4 view', ['--head', '--look', 'full', '--angles', '3/4']),
    'closeup':     ('a named area (--area, see AREAS), lit and clay, one 3/4 view',
                    ['--look', 'full,clay', '--angles', '3/4']),
    'walk':        ('the world walk on the mannequin, 6 frames, 3/4 and profile',
                    ['--mannequin', '--walk', '6', '--angles', '3/4,profile']),
    'run':         ('the run on the mannequin, 6 frames, profile', ['--mannequin', '--run', '6', '--angles', 'profile']),
}

# draft: low samples, no denoiser, small (the default for every preset). check: what threads already send for a
# review look (32 samples, denoised, 480; with the request's "quality": "check" the queue drops it to 8). final: the
# script's own samples and size, denoised, for John's sign-off. A tier only fills in flags the command leaves out, and
# only when --preset or --tier is given, so no existing request changes or gets a new cache key.
TIERS = {
    'draft': {'--samples': '8', '--size': '320', '--look': 'clay'},
    'check': {'--samples': '32', '--size': '480', '--denoise': None},
    'final': {'--denoise': None},
}
VALUE_FLAGS = ('--look', '--angles', '--walk', '--run', '--samples', '--size', '--area', '--tier', '--preset')


def expand(argv):
    """argv with --preset and --tier spelled out as views.py's own flags (flags already given are kept)."""
    out = list(argv)
    name = out[out.index('--preset') + 1] if '--preset' in out else None
    if name is not None:
        if name not in PRESETS:
            raise SystemExit(f'views.py: no preset {name!r}; presets: {", ".join(PRESETS)}')
        flags = PRESETS[name][1]
        i = 0
        while i < len(flags):
            f = flags[i]
            has_value = f in VALUE_FLAGS
            if f not in out:
                out += flags[i:i + 2] if has_value else [f]
            i += 2 if has_value else 1
        if name == 'closeup' and '--area' not in out:
            raise SystemExit(f'views.py: --preset closeup needs --area NAME ({", ".join(AREA_NAMES)} or piece:PREFIX)')
    tier = out[out.index('--tier') + 1] if '--tier' in out else ('draft' if name else None)
    if tier is not None:
        if tier not in TIERS:
            raise SystemExit(f'views.py: --tier takes {", ".join(TIERS)} (got {tier!r})')
        for f, v in TIERS[tier].items():
            if f not in out:
                out += [f] if v is None else [f, v]
    return out


def tier_asked(argv):
    """The tier this command runs at (argv already expanded): --tier, or draft for a preset, or None."""
    if '--tier' in argv:
        return argv[argv.index('--tier') + 1]
    return 'draft' if '--preset' in argv else None


def looks(argv):
    """The looks to render, as file suffixes: '' (the lit look), '-clay', '-matcap'."""
    if '--look' in argv:
        names = [x.strip() for x in argv[argv.index('--look') + 1].split(',') if x.strip()]
        bad = [x for x in names if x not in ('full', 'clay', 'matcap')]
        if bad or not names:
            raise SystemExit(f'views.py: --look takes full, clay, matcap (got {",".join(bad) or "nothing"})')
        return tuple('' if x == 'full' else '-' + x for x in names)
    if '--clay-only' in argv:
        return ('-clay',)
    return ('', '-clay') if '--clay' in argv else ('',)


# ------------------------------------------------------------------ named areas
AREA_NAMES = ('face', 'head', 'shoulders', 'torso', 'waist', 'hips', 'hands', 'legs', 'knees', 'feet', 'full')


def areas(f):
    """Each area as (bottom z, top z, half width), from the base figure's measurements (female.py: the head's
    height and the chin, shoulder, chest, neck, hip and ankle heights, the limb lengths). Margins are in head
    heights, so a figure with other proportions is framed the same way. Clothes never move these, so an area's
    camera is the same before and after any garment change."""
    H, chin = f.HEAD_H, f.HEAD_Z
    crown, sh, hip = chin + H, f.SH_Z, f.HIP_Z
    knee, ankle = hip - f.THIGH, f.ANKLE_Z
    floor = ankle - 3.4                      # (the sole is 3.4 under the ankle, as views.py's walk floor)
    elbow = sh - f.UPPER
    wrist = elbow - f.FORE
    return {
        'face':      (chin - 0.15 * H, crown + 0.10 * H, 0.70 * H),
        'head':      (sh - 0.23 * H, crown + 0.14 * H, 0.70 * H),     # (views.py --head's own framing: 47 to 64)
        'shoulders': (f.CHEST_Z - 0.25 * H, chin + 0.35 * H, 1.05 * H),
        'torso':     (hip - 0.35 * H, f.NECK_Z + 0.25 * H, 1.05 * H),
        'waist':     (hip - 0.60 * H, f.CHEST_Z + 0.20 * H, 1.00 * H),
        'hips':      (hip - 0.50 * f.THIGH, hip + 0.60 * H, 1.00 * H),
        'hands':     (wrist - 0.75 * H, wrist + 0.65 * H, 1.05 * H),
        'legs':      (floor - 0.10 * H, hip + 0.30 * H, 0.90 * H),
        'knees':     (knee - 0.60 * H, knee + 0.60 * H, 0.80 * H),
        'feet':      (floor - 0.15 * H, ankle + 0.60 * H, 0.90 * H),
        'full':      (floor - 0.10 * H, crown + 0.15 * H, 1.10 * H),
    }


def area_frame(f, name):
    """(centre, ortho height) of a square frame on a named area."""
    table = areas(f)
    if name not in table:
        raise SystemExit(f'views.py: no area {name!r}; areas: {", ".join(AREA_NAMES)} or piece:PREFIX')
    z0, z1, hw = table[name]
    return (0.0, 0.0, (z0 + z1) / 2), max(z1 - z0, 2 * hw)


# ------------------------------------------------------------------ matcap
def matcap_material(bpy):
    """An unlit material shaded by the surface's normal as the camera sees it: light where it faces the camera and
    up, darker where it turns away, a soft rim at the silhouette. No lights, no noise: forms read at any sample
    count, so it is the cheapest honest look at a shape."""
    m = bpy.data.materials.get('matcap')
    if m is not None:
        return m
    m = bpy.data.materials.new('matcap')
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    n = nt.nodes.new
    geo = n('ShaderNodeNewGeometry')
    vt = n('ShaderNodeVectorTransform')
    vt.vector_type, vt.convert_from, vt.convert_to = 'NORMAL', 'WORLD', 'CAMERA'
    sep = n('ShaderNodeSeparateXYZ')
    nt.links.new(geo.outputs['Normal'], vt.inputs['Vector'])
    nt.links.new(vt.outputs['Vector'], sep.inputs['Vector'])

    def math(op, a, b=None, clamp=False):
        x = n('ShaderNodeMath')
        x.operation, x.use_clamp = op, clamp
        for k, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                x.inputs[k].default_value = v
            else:
                nt.links.new(v, x.inputs[k])
        return x.outputs[0]
    # (camera space: +z faces the camera, +y is up)
    facing = math('MAXIMUM', sep.outputs['Z'], 0.0)
    rim = math('MULTIPLY', math('POWER', math('SUBTRACT', 1.0, facing), 3.0), 0.22)
    v = math('ADD', math('ADD', 0.16, math('MULTIPLY', facing, 0.62)), math('MULTIPLY', sep.outputs['Y'], 0.14))
    v = math('ADD', v, rim, clamp=True)
    ramp = n('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color = (0.10, 0.09, 0.085, 1)
    ramp.color_ramp.elements[1].color = (0.86, 0.83, 0.79, 1)
    nt.links.new(v, ramp.inputs['Fac'])
    em = n('ShaderNodeEmission')
    nt.links.new(ramp.outputs['Color'], em.inputs['Color'])
    out = n('ShaderNodeOutputMaterial')
    nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
    return m


_MATCAP_SAVED = {}


def set_look(bpy, look, saved, clay):
    """Switch every mesh to a look: '' as built, '-clay' views.py's clay, '-matcap' the clay's hiding with the
    matcap material, the Standard view transform and no bounces (so the emission is seen exactly as shaded)."""
    sc = bpy.context.scene
    if _MATCAP_SAVED:                                  # (leaving the matcap: the scene settings back)
        sc.view_settings.view_transform = _MATCAP_SAVED.pop('view')
        sc.cycles.max_bounces = _MATCAP_SAVED.pop('bounces')
    clay(False, saved)
    if look == '-clay':
        clay(True, saved)
    elif look == '-matcap':
        clay(True, saved)
        mc = matcap_material(bpy)
        for o in bpy.data.objects:
            if o.type == 'MESH':
                for s in o.material_slots:
                    if s.material is not None and s.material.name == 'clay':
                        s.material = mc
        _MATCAP_SAVED['view'], _MATCAP_SAVED['bounces'] = sc.view_settings.view_transform, sc.cycles.max_bounces
        sc.view_settings.view_transform = 'Standard'
        sc.cycles.max_bounces = 0


# ------------------------------------------------------------------ labels and the record of each image
IMAGES = []          # one entry per image written: path, look, angles, camera, the samples it really rendered at
STATE = {'look': ''}


def tier_of(samples, denoised, requested=None):
    """The tier an image was really rendered at: never denoised is a draft whatever was asked; otherwise the tier
    asked for, or with none asked, check at up to 16 samples (rq_run.py --quality check renders 8) and final above."""
    if not denoised:
        return 'draft'
    if requested in ('check', 'final'):
        return requested
    return 'check' if samples <= 16 else 'final'


def recorded(row, bpy, angles):
    """views.py's row() wrapped to record each image it writes: its camera and the quality it rendered at."""
    def wrapped(cam, out, tmp, size, centre, height, ortho=True, dist=200.0):
        row(cam, out, tmp, size, centre, height, ortho, dist)
        sc = bpy.context.scene
        IMAGES.append({'file': os.path.basename(out), 'look': STATE['look'].lstrip('-') or 'full',
                       'angles': [a for a, _ in angles()], 'size': list(size),
                       'camera': {'centre': [round(float(c), 4) for c in centre], 'ortho_height': round(float(height), 4),
                                  'ortho': ortho, 'dist': dist},
                       'samples': int(sc.cycles.samples), 'denoised': bool(sc.cycles.use_denoising)})
    return wrapped


def note(bpy, out, angles, look='full'):
    """Record an image written outside row() (the walk and run sheets)."""
    sc = bpy.context.scene
    IMAGES.append({'file': os.path.basename(out), 'look': look, 'angles': list(angles), 'camera': None,
                   'samples': int(sc.cycles.samples), 'denoised': bool(sc.cycles.use_denoising)})


COLOURS = {'draft': ((214, 140, 40), (20, 14, 6)), 'check': ((60, 120, 200), (245, 248, 252)),
           'final': ((60, 150, 80), (245, 252, 246))}


def commit():
    try:
        return subprocess.run(['git', 'rev-parse', '--short=9', 'HEAD'], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def stamp(path, text, tier):
    """A strip under the image: the tier in its colour and what the image is, so a draft is never taken for a final.
    The text is wrapped at its ' | ' breaks onto as many lines as the image's width needs, so nothing is cut off."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(path).convert('RGB')
    h = max(16, im.width // 70)
    try:
        font = ImageFont.load_default(size=int(h * 0.72))
    except TypeError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(im)
    lines, cur = [], ''
    for part in [p.strip() for p in text.split('|') if p.strip()]:
        trial = f'{cur}  |  {part}' if cur else part
        if cur and draw.textlength(trial, font=font) > im.width - 12:
            lines.append(cur)
            cur = part
        else:
            cur = trial
    lines.append(cur)
    bg, fg = COLOURS[tier]
    out = Image.new('RGB', (im.width, im.height + h * len(lines)), bg)
    out.paste(im, (0, 0))
    d = ImageDraw.Draw(out)
    for i, line in enumerate(lines):
        d.text((6, im.height + h * (i + 0.12)), line, fill=fg, font=font)
    out.save(path)
    return h * len(lines)


def finish(out, argv, figure):
    """Label every recorded image and write DIR/views.json."""
    sha = commit()
    preset = argv[argv.index('--preset') + 1] if '--preset' in argv else None
    requested = argv[argv.index('--tier') + 1] if '--tier' in argv else None
    try:
        import layers
        cloth = layers.summary()
    except ImportError:
        cloth = ''
    for im in IMAGES:
        im['tier'] = tier_of(im['samples'], im['denoised'], requested)
        what = f"{im['samples']} samples, {'denoised' if im['denoised'] else 'not denoised'}"
        text = (f"{im['tier'].upper()}  {what}  |  {im['look']}  |  {' / '.join(im['angles'])}  |  {figure}"
                + ('  mannequin' if '--mannequin' in argv else '') + (f'  @ {sha}' if sha else '')
                + (f'  |  preset {preset}' if preset else '') + (f'  |  {cloth}' if cloth else ''))
        p = os.path.join(out, im['file'])
        if os.path.exists(p):
            im['label_px'] = stamp(p, text, im['tier'])
    rec = {'figure': figure, 'commit': sha, 'preset': preset,
           'requested_tier': requested,
           'mannequin': '--mannequin' in argv, 'argv': argv, 'images': IMAGES,
           'cloth': [{'garment': n, 'state': st, 'seconds': round(t, 1)} for n, st, t in getattr(
               sys.modules.get('layers'), 'LOG', [])]}
    with open(os.path.join(out, 'views.json'), 'w') as fh:
        json.dump(rec, fh, indent=1)
    print('wrote', os.path.join(out, 'views.json'), flush=True)
