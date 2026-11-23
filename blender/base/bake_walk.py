"""Bake the base figure's walk or run, as blender/base/female.py poses it now, to the cape bench's JSON
(public/cape-lab/walk-<name>.json, played by src/cape-lab/baked-walk.js). The figure is built and posed in Blender
itself (the walk thread's bake_walk.py did it with Blender stubbed out, which the neck's joints and the fists have
since outgrown). Per frame, each rig joint's pivot gives `q`: its turn in glTF axes away from the rest pose (all
pivots unturned), xyzw, and `p`: where the pivot is, in model units.

    python3 blender/base/bake_walk.py walk public/cape-lab/walk-new.json [--frames 24]
    python3 blender/base/bake_walk.py run  public/cape-lab/walk-run.json [--frames 24]

The run also says that it closes the hands (`fists`), which the bench shows with the glb's Fist meshes."""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# (the bare mannequin: no head model, no dress, no accessories; only the rig's pivots matter)
os.environ.setdefault('HEAD_GLB', '')
os.environ.setdefault('NO_DRESS', '1')
os.environ.setdefault('NO_ACC', '1')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'sorceress_pendant'))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
import female  # noqa: E402
import sorceress_pendant as SP  # noqa: E402

B2G = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)   # Blender axes -> glTF axes
PIVOT = {'Pelvis': 'Pelvis', 'Chest': 'Chest', 'Cervical': 'Cervical', 'Midneck': 'Midneck', 'Neck': 'Head', 'Thigh': 'Hip', 'Shin': 'Knee', 'Foot': 'Ankle',
         'UpperArm': 'Shoulder', 'Forearm': 'Elbow', 'Hand': 'Wrist'}
# (the neck's own joints too, Chest -> Cervical -> Midneck -> Neck: left at the bind pose, the neck's skin stood up
# out of a collar fitted to the neck as it is posed)
BONES = ['Pelvis', 'Chest', 'Cervical', 'Midneck', 'Neck'] + [f'{b}.{s}' for s in 'LR' for b in ('Thigh', 'Shin', 'Foot', 'UpperArm', 'Forearm', 'Hand')]


def quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return [x, y, z, w]


def pivot(b):
    seg, _, side = b.partition('.')
    return bpy.data.objects.get(f'{PIVOT[seg]}.{side}' if side else PIVOT[seg])


def rotations():
    bpy.context.view_layer.update()
    return {b: np.array(pivot(b).matrix_world)[:3, :3] for b in BONES if pivot(b) is not None}


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    kind, out = argv[0], argv[1]
    n = int(argv[argv.index('--frames') + 1]) if '--frames' in argv else 24
    female.MANNEQUIN = True
    SP.build()
    female.LOCKED_WALK, female.ARM_SWING = True, 0.24        # (as export_glb.py and views.py --walk set them)
    female.RUN = kind == 'run'
    # the rest: every pivot unturned (the pose the rig is bound in)
    bones = [b for b in BONES if pivot(b) is not None]
    objs = {pivot(b) for b in bones} | {o for o in bpy.data.objects if o.type == 'EMPTY'}
    keep = {o: tuple(o.rotation_euler) for o in objs}
    for o in objs:
        o.rotation_euler = (0, 0, 0)
    rest = rotations()
    for o, r in keep.items():
        o.rotation_euler = r
    frames = []
    for k in range(n):
        female.pose(2 - 0.5 + female.WALK + k * female.WALK / n)
        R = rotations()
        fr = {}
        for b in bones:
            g = B2G @ (R[b] @ rest[b].T) @ B2G.T
            p = B2G @ np.array(pivot(b).matrix_world.translation)
            fr[b] = {'q': [round(v, 6) for v in quat(g)], 'p': [round(float(v), 4) for v in p]}
        frames.append(fr)
    d = {'about': f"The base figure's {kind} (blender/base/female.py as it poses it now), baked by blender/base/bake_walk.py.",
         'frames_per_cycle': n, 'bones': frames}
    if kind == 'run':
        leg = female.THIGH + female.SHIN
        d.update(stride=round(female.RUN_STRIDE * leg, 2), seconds_per_cycle=female.RUN_SECONDS, leg=round(leg, 2),
                 fists=female.RUN_FIST > 0.5)
    else:
        d.update(stride=female.STRIDE, seconds_per_cycle_in_world=46 * (2.2 / 80 * 1.2) / 2.2)
    with open(out, 'w') as f:
        json.dump(d, f)
    print('wrote', out, n, 'frames')


if __name__ == '__main__':
    main()
