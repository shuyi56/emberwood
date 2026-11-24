"""Cheap draft renders of a built bodice for the tension solver's comparisons: clay (one matte grey on everything)
or a colour-coded pass (each design part its own flat colour: the fronts, each lapel, the collar, the buttons).

    python3 blender/tension/clay.py BUILD.blend OUTDIR '{"mode": "clay", "samples": 12, "views": {
        "front": {"z": 47.5, "scale": 14, "res": [700, 700], "az": 0, "at": 0.0}}}'

az turns the camera toward her left (degrees), `at` shifts it sideways (model units) so close-ups can aim off the
centre line, scale is the orthographic width. Draft settings: low samples, no denoising, Eevee-free Cycles CPU.
Lapel points are read from BUILD's sibling LAPEL.blend.lapels.npz when "lapels" names it (code mode)."""
import bpy, math, sys, os, json
import numpy as np

blend, out, spec = sys.argv[-3], sys.argv[-2], json.loads(sys.argv[-1])
os.makedirs(out, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=blend)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.samples = spec.get('samples', 12); sc.cycles.device = 'CPU'
sc.cycles.use_denoising = False
sc.render.film_transparent = False


def mat(name, rgb, rough=0.8):
    m = bpy.data.materials.new(name); m.use_nodes = True
    bs = m.node_tree.nodes.get('Principled BSDF')
    bs.inputs['Base Color'].default_value = (*rgb, 1.0); bs.inputs['Roughness'].default_value = rough
    return m


CLAY = mat('clay', (0.62, 0.62, 0.60))
CODE = {'front': mat('c_front', (0.55, 0.58, 0.66)), 'lapel1': mat('c_lapel1', (0.90, 0.45, 0.15)),
        'lapel2': mat('c_lapel2', (0.95, 0.75, 0.15)), 'collar': mat('c_collar', (0.20, 0.55, 0.85)),
        'buttons': mat('c_buttons', (0.85, 0.15, 0.55)), 'rest': mat('c_rest', (0.70, 0.70, 0.68)),
        'skin': mat('c_skin', (0.80, 0.70, 0.62))}
mode = spec.get('mode', 'clay')
for o in bpy.data.objects:
    if o.type != 'MESH':
        continue
    o.data.materials.clear()
    if mode == 'clay':
        o.data.materials.append(CLAY); continue
    if o.name == 'Redingote collar':
        o.data.materials.append(CODE['collar'])
    elif o.name == 'Redingote bodice buttons':
        o.data.materials.append(CODE['buttons'])
    elif o.name == 'Redingote bodice':
        for k in ('front', 'lapel1', 'lapel2'):
            o.data.materials.append(CODE[k])
        lp = spec.get('lapels')
        if lp and os.path.exists(lp):
            from scipy.spatial import cKDTree
            L = np.load(lp); mw = np.array(o.matrix_world)
            V = np.array([v.co[:] for v in o.data.vertices]) @ mw[:3, :3].T + mw[:3, 3]
            tag = np.zeros(len(V), int)
            for k in (1, 2):
                d, _ = cKDTree(L['pts%d' % k]).query(V)
                tag[d < 1e-4] = k
            for p in o.data.polygons:
                t = tag[list(p.vertices)]
                p.material_index = int(t.min()) if t.min() == t.max() else 0
    elif o.name.startswith(('Skin', 'Neck')):
        o.data.materials.append(CODE['skin'])
    else:
        o.data.materials.append(CODE['rest'])
cam = bpy.data.objects.get('View') or bpy.data.objects.get('Camera'); sc.camera = cam
cam.data.type = 'ORTHO'
for name, v in spec['views'].items():
    th = math.radians(v.get('az', 0)); z = v['z']; d = 200; at = v.get('at', 0.0)
    cam.data.ortho_scale = v['scale']
    cam.location = (d * math.sin(th) + at * math.cos(th), -d * math.cos(th) + at * math.sin(th), z)
    cam.rotation_euler = (math.radians(90), 0, th)
    cam.data.clip_end = 1000
    sc.render.resolution_x, sc.render.resolution_y = v['res']
    hid = []
    for o in bpy.data.objects:
        if o.type == 'MESH' and any(o.name.startswith(h) for h in v.get('hide', spec.get('hide', []))) and not o.hide_render:
            o.hide_render = True; hid.append(o)
    sc.render.filepath = os.path.join(out, name + '.png')
    json.dump(v, open(os.path.join(out, name + '.view.json'), 'w'))       # (landmarks.py projects through the same camera)
    bpy.ops.render.render(write_still=True)
    for o in hid:
        o.hide_render = False
    print('clay: rendered', name, flush=True)
