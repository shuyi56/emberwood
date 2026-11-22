"""Write a measurement sheet for a Blender render, with landmarks projected from 3D.

Clicking landmarks on a render works (measure.html, or `refmeasure.py set`), but a model has exact points,
so project them instead. Use inside Blender, after the camera is set up and before or after rendering:

    import sys; sys.path.append("tools/refmeasure")
    from blender_landmarks import write_sheet
    pts = {
        "shoulder_l": body.matrix_world @ body.data.vertices[i_sh_l].co,
        "shoulder_r": ...,
        "neck_front": ...,
    }
    write_sheet("/tmp/x/front.png", pts, view="front 3/4 left", template="refs/habit.measure.json")

`template` copies the unit, plumb landmark, segments and ratios from the reference sheet, so both sheets
measure the same things. Then, outside Blender:

    python3 tools/refmeasure/refmeasure.py compare refs/habit.measure.json /tmp/x/front.measure.json

Match the camera to the photo first (same view, and roughly the same lens: a museum photo is a long
lens, close to ortho). Ratios of widths taken from different viewing angles are not comparable.
"""
import json
import os

import bpy
from bpy_extras.object_utils import world_to_camera_view


def project(points, scene=None, camera=None):
    """{name: world Vector} -> {name: [x, y]} in render pixels, y down. Points behind the camera are dropped."""
    scene = scene or bpy.context.scene
    camera = camera or scene.camera
    r = scene.render
    w = r.resolution_x * r.resolution_percentage / 100.0
    h = r.resolution_y * r.resolution_percentage / 100.0
    out = {}
    for name, co in points.items():
        v = world_to_camera_view(scene, camera, co)
        if v.z <= 0:
            continue
        out[name] = [round(v.x * w, 2), round((1.0 - v.y) * h, 2)]
    return out


def write_sheet(image_path, points, view="", template=None, note="", scene=None, camera=None):
    """Write <image>.measure.json with projected landmarks. Keeps landmarks already in the sheet."""
    scene = scene or bpy.context.scene
    camera = camera or scene.camera
    r = scene.render
    size = [round(r.resolution_x * r.resolution_percentage / 100), round(r.resolution_y * r.resolution_percentage / 100)]
    sheet = os.path.splitext(image_path)[0] + ".measure.json"
    s = {"version": 1, "image": os.path.basename(image_path), "size": size,
         "camera": {"view": view, "projection": "ortho" if camera.data.type == "ORTHO" else "persp",
                    "lens_mm": round(camera.data.lens, 2), "note": note or camera.name},
         "unit": None, "plumb": None, "landmarks": {}, "segments": {}, "ratios": {}}
    if os.path.exists(sheet):
        with open(sheet) as f:
            old = json.load(f)
        s["landmarks"] = old.get("landmarks", {})
        for k in ("unit", "plumb", "segments", "ratios"):
            s[k] = old.get(k) or s[k]
    if template:
        with open(template) as f:
            t = json.load(f)
        for k in ("unit", "plumb", "segments", "ratios"):
            s[k] = t.get(k)
    s["landmarks"].update(project(points, scene, camera))
    with open(sheet, "w") as f:
        json.dump(s, f, indent=1)
        f.write("\n")
    return sheet
