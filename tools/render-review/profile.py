"""Surface profile of one object of a built figure, without rendering: where its front and back surfaces lie.

smoke.py calls measure() at the moment the first render would start, when RQ_PROFILE is set:

    RQ_PROFILE='Leggings seat|0,0.6,1.2,2,3.2|26:33:0.5|/tmp/p.json' \\
        python3 tools/render-review/smoke.py blender/base/views.py --figure rogue --turn --out /tmp/x

The spec is OBJECT|X positions|Z0:Z1:STEP|output file. For each x and height it ray-casts the object from the
front and from the back (figure at rest, facing -y) and records the y of the first surface met. Run it on two
commits and compare: it shows which surface actually moved, and whether a change dug a trough (the centre
falling behind its neighbours) or left a lobe (one column standing forward of both sides). See PITFALLS P017.

    python3 tools/render-review/profile.py A.json B.json      prints the two side by side
"""
import json
import sys


def measure(spec):
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    name, xs, zr, out = spec.split("|")
    z0, z1, step = (float(v) for v in zr.split(":"))
    sc = bpy.context.scene
    frame0 = sc.frame_current
    root = [o for o in bpy.data.objects if o.parent is None and o.type == "EMPTY" and o.name != "Rig"][0]
    rot0 = tuple(root.rotation_euler)
    sc.frame_set(1)
    root.rotation_euler = (0, 0, 0)
    bpy.context.view_layer.update()
    o = bpy.data.objects[name]
    tree = BVHTree.FromObject(o, bpy.context.evaluated_depsgraph_get())
    mw, inv = o.matrix_world, o.matrix_world.inverted()
    rot = inv.to_3x3()
    res = {"object": name, "front": {}, "back": {}}
    for side, y0, d in (("front", -60.0, Vector((0, 1, 0))), ("back", 60.0, Vector((0, -1, 0)))):
        for x in (float(v) for v in xs.split(",")):
            col = []
            z = z1
            while z >= z0 - 1e-6:
                hit = tree.ray_cast(inv @ Vector((x, y0, z)), (rot @ d).normalized(), 200.0)
                col.append([round(z, 2), round((mw @ hit[0]).y, 3) if hit[0] is not None else None])
                z -= step
            res[side][str(x)] = col
    root.rotation_euler = rot0
    sc.frame_set(frame0)
    with open(out, "w") as f:
        json.dump(res, f)
    return res


def compare(a, b):
    lines = []
    for side in ("front", "back"):
        lines.append(f"{side} surface y ({'more negative' if side == 'front' else 'more positive'} is further out): A, B, B-A")
        for x in a[side]:
            lines.append(f"  x = {x}")
            for (z, ya), (_, yb) in zip(a[side][x], b[side].get(x, [])):
                if ya is None and yb is None:
                    continue
                d = f"{yb - ya:+.2f}" if ya is not None and yb is not None else ""
                lines.append(f"    z {z:5.1f}  {ya}  {yb}  {d}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(compare(json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))))
