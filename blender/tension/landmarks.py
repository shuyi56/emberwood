"""Project a build's coat landmarks into a front ortho render and write its refmeasure sheet (tools/refmeasure,
names from `refmeasure.py names`), so a derived build and the current one are measured the same way against the
same reference sheet.

    python3 blender/tension/landmarks.py BUILD.blend LAPEL.blend.lapels.npz COLLAR.npz RENDER.png REF.measure.json

The render must be clay.py's "front" view (az 0, ortho). Landmarks: shoulder points, neck front, each roll line's
top (roll_top), the break (break_point), each lapel's point (the turned cloth's farthest point from its line),
each collar point (the fall's lowest front point), the waist."""
import sys, os, json, math
import numpy as np
import bpy
from mathutils import Vector

blend, lap, col, png, ref = sys.argv[-5:]
bpy.ops.wm.open_mainfile(filepath=blend)
sys.path.insert(0, os.environ.get('REFMEASURE', 'tools/refmeasure'))
from blender_landmarks import write_sheet  # noqa: E402

spec = json.load(open(os.path.splitext(png)[0] + '.view.json'))
sc = bpy.context.scene
cam = bpy.data.objects.get('View') or bpy.data.objects.get('Camera'); sc.camera = cam
cam.data.type = 'ORTHO'; cam.data.ortho_scale = spec['scale']
th = math.radians(spec.get('az', 0)); at = spec.get('at', 0.0)
cam.location = (200 * math.sin(th) + at * math.cos(th), -200 * math.cos(th) + at * math.sin(th), spec['z'])
cam.rotation_euler = (math.radians(90), 0, th)
sc.render.resolution_x, sc.render.resolution_y = spec['res']; sc.render.resolution_percentage = 100
cam.data.clip_end = 1000; bpy.context.view_layer.update()
b = bpy.data.objects['Redingote bodice']
M = {k[7:]: float(b[k]) for k in b.keys() if k.startswith('bodice_')}
L = np.load(lap); C = np.load(col)
pts = {}
y0 = -4.0
pts['shoulder_l'] = Vector((-M['x_sh'], y0, M['z_sh'])); pts['shoulder_r'] = Vector((M['x_sh'], y0, M['z_sh']))
pts['waist_l'] = Vector((-M['waist_half'], y0, M['z_waist'])); pts['waist_r'] = Vector((M['waist_half'], y0, M['z_waist']))
pts['neck_front'] = Vector((0.0, y0, M['z_neck_front']))
for k in (1, 2):
    cr = L['crease%d' % k]; ct = L['ct%d' % k]
    side = 'r' if cr[:, 0].mean() < 0 else 'l'          # image left/right: her right is image left (x < 0)
    side = 'l' if side == 'r' else 'r'
    top = cr[np.argmin(ct)]; bot = cr[np.argmax(ct)]
    pts['roll_top_' + side] = Vector(top)
    if k == 1:
        pts['break_point'] = Vector(bot)
    P = L['pts%d' % k]; d = L['d%d' % k]
    pts['lapel_point_' + side] = Vector(P[np.argmax(d)])
    Q = C['pts']; s = (Q[:, 0] < 0) if top[0] < 0 else (Q[:, 0] > 0)
    fr = Q[s & (Q[:, 1] < M['yc'])]
    pts['collar_point_' + side] = Vector(fr[np.argmin(fr[:, 2])])
write_sheet(png, pts, view='front (ortho)', template=ref)
print('landmarks: wrote', os.path.splitext(png)[0] + '.measure.json')
