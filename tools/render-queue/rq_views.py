"""views.py with the render queue's extras: crops of named problem areas and the pose test strip (John, 2026-11-19).
The queue puts this file in a job's worktree as blender/base/rq_views.py and runs it instead of views.py whenever a
request asks for --crop or --poses. It loads views.py's own code at that commit (without running it), runs views.py's
main exactly as views.py would (the build, then any --turn, --head, --walk ... asked for), and then, on the same
build:

--crop SPECS   square close-ups of problem areas, each a row over --crop-angles (default: --angles, else 3/4), from
               the same fixed camera and lights as --head, so crops of two commits line up pixel for pixel:
               DIR/crop-NAME.png (and crop-NAME-clay.png with --clay or --clay-only alone). SPECS is a comma list of:
                 NAME          a region of the body (below), measured from the base figure's own sizes at this commit
                 NAME+PAD      the region widened by PAD (0.3 is 30%)
                 box:X0:Y0:X1:Y1[@S]   a box in pixels of one tile of the turn (DIR/turn.png at --size S, default
                               360: each tile S wide and 1.75 S high), so a fault seen in a turn is cropped as it is
--crop-size N  the crops' height in pixels (default 384; a wide region such as the hem is wider than it is high)
--poses NAMES  the pose test strip: the clay figure at the joints' extremes, one tile per pose, from --pose-angle
               (default 3/4), on one fixed camera: DIR/poses.png and DIR/pose-NAME.png. NAMES is all, or some of
               rest, arms-up, arms-fwd, elbows, step, crouch, twist.

Regions (centred on the figure's axis, so one crop serves every angle; both sides show):
  neck, shoulder, armhole, bust, elbow, waist, wrist (the cuff), hips (the seat), hand, knee, hem (floor length),
  ankle, face, head.
"""
import math
import os
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWS_PY = os.path.join(HERE, 'views.py')
src = open(VIEWS_PY).read()
cut = src.rstrip().rfind('\nmain()')
if cut < 0:
    raise SystemExit('rq_views: views.py at this commit does not end in main()')
V = {'__file__': VIEWS_PY, '__name__': 'views_lib'}
exec(compile(src[:cut], VIEWS_PY, 'exec'), V)
bpy, female, opt, argv = V['bpy'], V['female'], V['opt'], V['argv']

POSES = ('rest', 'arms-up', 'arms-fwd', 'elbows', 'step', 'crouch', 'twist')
ALL_VIEWS = [('front', 0), ('3/4', 45), ('profile', 90), ('back 3/4', 135), ('back', 180)]


def regions():
    """name -> (centre z, height[, width / height]): from the base figure's sizes, so they follow the body at this
    commit."""
    f = female
    sh, hip, ank, up, fore = f.SH_Z, f.HIP_Z, f.ANKLE_Z, f.UPPER, f.FORE
    knee = hip - f.THIGH
    wrist = sh - up - fore
    return {
        'head': (getattr(f, 'HEAD_Z', 55.5) + 0.5, 17.0), 'face': (getattr(f, 'HEAD_Z', 55.5) + 0.5, 10.0),
        'neck': (f.NECK_Z, 9.0), 'shoulder': (sh - 0.5, 2 * f.SH_X + 6.0), 'armhole': (sh - 3.0, 2 * f.SH_X + 5.0),
        'bust': (f.CHEST_Z + 2.0, 2 * f.SH_X + 4.0), 'elbow': (sh - up, 2 * f.SH_X + 6.0),
        'waist': (0.5 * (f.CHEST_Z + hip) + 1.0, 14.0), 'wrist': (wrist + 1.0, 2 * f.SH_X + 7.0),
        'hand': (wrist - 2.5, 2 * f.SH_X + 8.0), 'hips': (hip, 16.0, 1.4), 'knee': (knee, 12.0, 1.6),
        # (a floor-length hem is far wider than it is deep: a wide, low strip from just above the floor)
        'hem': (ank + 6.0, 20.0, 2.8), 'ankle': (ank, 10.0, 1.6),
    }


def crop_frames(specs):
    """[(stem, centre, height)] for --crop."""
    reg, out = regions(), []
    for spec in [s.strip() for s in specs.split(',') if s.strip()]:
        if spec.startswith('box:'):
            body, _, at = spec[4:].partition('@')
            x0, y0, x1, y1 = (float(v) for v in body.split(':'))
            S = float(at or 360)
            W, H = S, 1.75 * S
            k = 68.0 / H                       # (the turn: ortho height 68 about z 32.5)
            cx, cz = ((x0 + x1) / 2 - W / 2) * k, 32.5 + (H / 2 - (y0 + y1) / 2) * k
            bh = max(2.0, abs(y1 - y0) * k * 1.15)
            out.append((f'box-{int(x0)}-{int(y0)}-{int(x1)}-{int(y1)}', (cx, 0, cz), bh,
                        max(0.4, min(4.0, abs(x1 - x0) / max(1.0, abs(y1 - y0))))))
            continue
        name, _, pad = spec.partition('+')
        if name not in reg:
            raise SystemExit(f'rq_views: --crop {name}: no such region (have {", ".join(sorted(reg))}, or box:...)')
        z, h, aspect = (reg[name] + (1.0,))[:3]
        out.append((name + (f'-p{pad}' if pad else ''), (0, 0, z), h * (1 + float(pad or 0)), aspect))
    return out


def angles(flag, default):
    want = [a.strip().replace(' ', '') for a in (opt(flag) or opt('--angles') or default).split(',') if a.strip()]
    return [v for v in ALL_VIEWS if v[0].replace(' ', '') in want] or [ALL_VIEWS[1]]


def leg(side, th, kb, pitch=0.0):
    ob = bpy.data.objects
    hip, kn, an = ob[f'Hip.{side}'], ob[f'Knee.{side}'], ob[f'Ankle.{side}']
    hip.rotation_euler = (-th, 0, 0)
    kn.rotation_euler = (kb, 0, 0)
    an.rotation_euler = female.foot_level(tuple(ob['Pelvis'].rotation_euler), (-th, 0, 0), (kb, 0, 0), pitch)


def drop(*legs):
    """How far the hips come down for the lower leg to keep its foot on the floor."""
    T, S = female.THIGH, female.SHIN
    return T + S - max(T * math.cos(th) + S * math.cos(th - kb) for th, kb in legs)


def apply(name):
    ob = bpy.data.objects
    female.pose(0)
    arms = {'R': 1, 'L': -1}
    if name == 'arms-up':
        for side, s in arms.items():
            ob[f'Shoulder.{side}'].rotation_euler = (0, s * 1.35, 0)
            ob[f'Elbow.{side}'].rotation_euler = (-0.15, 0, 0)
    elif name == 'arms-fwd':
        for side, s in arms.items():
            ob[f'Shoulder.{side}'].rotation_euler = (-1.45, s * female.ARM_OUT, 0)
            ob[f'Elbow.{side}'].rotation_euler = (-0.1, 0, 0)
    elif name == 'elbows':
        for side, s in arms.items():
            ob[f'Shoulder.{side}'].rotation_euler = (-0.45, s * (female.ARM_OUT + 0.1), 0)
            ob[f'Elbow.{side}'].rotation_euler = (-2.1, 0, 0)
    elif name == 'step':
        r, l_ = (0.75, 0.25), (-0.45, 0.15)
        ob['Body'].location = (0, 0, -drop(r, l_))
        leg('R', *r); leg('L', *l_, pitch=-0.25)
    elif name == 'crouch':
        th, kb = 1.35, 2.1
        ob['Body'].location = (0, 0, -drop((th, kb)))
        ob['Chest'].rotation_euler = (0.45, 0, 0)
        for side, s in arms.items():
            leg(side, th, kb)
            ob[f'Shoulder.{side}'].rotation_euler = (-0.7, s * female.ARM_OUT, 0)
            ob[f'Elbow.{side}'].rotation_euler = (-0.6, 0, 0)
    elif name == 'twist':
        ob['Pelvis'].rotation_euler = (0, 0, -0.15)
        ob['Chest'].rotation_euler = (0, 0, 0.55)
    elif name != 'rest':
        raise SystemExit(f'rq_views: no pose {name!r} (have {", ".join(POSES)})')
    bpy.context.view_layer.update()


def crops(out):
    frames = crop_frames(opt('--crop'))
    size = int(opt('--crop-size', 384))
    cam = V['look_3d'](int(opt('--samples', 32)))
    keep = V['VIEWS']
    V['VIEWS'] = angles('--crop-angles', '3/4')
    looks = ('-clay',) if '--clay-only' in argv else ('', '-clay') if '--clay' in argv else ('',)
    saved = {}
    with tempfile.TemporaryDirectory() as tmp:
        for look in looks:
            V['clay'](look == '-clay', saved)
            for stem, centre, h, aspect in frames:
                V['row'](cam, os.path.join(out, f'crop-{stem}{look}.png'), tmp, (int(size * aspect), size), centre, h)
                print(f'rq crop {stem}{look}: centre z {centre[2]:.1f}, height {h:.1f}', flush=True)
        V['clay'](False, saved)
    V['VIEWS'] = keep


def poses(out):
    from PIL import Image, ImageDraw
    sc = bpy.context.scene
    for o in sc.objects:      # (the sheet's columns are keyed on the timeline: pose by hand instead, as the walk does)
        if o.animation_data:
            o.animation_data_clear()
    names = list(POSES) if opt('--poses') in (None, 'all') else [p for p in opt('--poses').split(',') if p]
    size = int(opt('--pose-size', 300))
    cam = V['look_3d'](int(opt('--samples', 32)))
    keep = V['VIEWS']
    V['VIEWS'] = angles('--pose-angle', '3/4')[:1]
    saved = {}
    V['clay'](True, saved)
    ims = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            apply(name)
            p = os.path.join(out, f'pose-{name}.png')
            # one fixed camera and framing for every pose: tall enough for the raised arms, low enough for the crouch
            V['row'](cam, p, tmp, (size, int(size * 1.5)), (0, 0, 35.0), 76.0)
            ims.append((name, Image.open(p).convert('RGB')))
            print(f'rq pose {name}', flush=True)
    V['clay'](False, saved)
    V['VIEWS'] = keep
    female.pose(0)
    w, h = ims[0][1].size
    strip = Image.new('RGB', (w * len(ims), h + 22), (40, 42, 48))
    d = ImageDraw.Draw(strip)
    for i, (name, im) in enumerate(ims):
        strip.paste(im, (i * w, 22))
        d.text((i * w + 6, 5), name, fill=(230, 230, 230))
    strip.save(os.path.join(out, 'poses.png'))
    print('wrote', os.path.join(out, 'poses.png'), flush=True)


def main():
    out = opt('--out', os.path.join(tempfile.gettempdir(), 'rq-views'))
    V['main']()             # views.py itself: the build and whatever it was asked for
    if opt('--crop'):
        crops(out)
    if opt('--poses') or '--poses' in argv:
        poses(out)


main()
