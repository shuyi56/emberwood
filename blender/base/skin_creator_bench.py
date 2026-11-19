"""The skin bench: a Face Creator head rendered in Blender with the layered skin (skin_layers.py), on the creator's own
cameras and lights, beside the creator's render of the same view. It is the check that the creator's port of the
layered skin looks as Blender's does; it builds no figure.

    node tools/face-creator/dump-head.cjs face-creator.html DUMP [Ranger]
    python3 blender/base/skin_creator_bench.py DUMP --out DIR [--views front,three,cheek,nose,temple,profile]
        [--samples 32] [--scale 0.5] [--before DIR_OF_BEFORE_PNGS] [--denoise] [--plain] [--reuse]

What is the same on both sides: the head's mesh, its painted face (the epidermis's tone: flush, shade, freckles,
brows, lashes) and its occlusion; the layers' rest positions, so Blender's noise and Voronoi land where the creator's
rebuilt ones do; skin_tones' thin-skin and freckle zones; the creator's three lights (suns of the same colour and
intensity: three.js r155+ and Cycles both give a matte white surface intensity / pi), only the key casting shadows;
its hemisphere light and room environment as a world measured from them on a matte white sphere; ACES tone mapping
afterwards, as three.js does. Light does not bounce between surfaces (as in three.js), but it does scatter under the
skin. What differs is the point of the bench: Blender scatters light under the skin where the creator wraps it, and
Blender sows its own pores where the creator keeps its bump map. The creator's freckles are in the painted tone, so
skin_layers' own freckles are off here (FRECKLES = 0). Lips, brows and lashes (the mask's green) take the painted
colour on a plain surface, as the creator keeps them out of the layers.

Writes blender-<view>.png and compare-<view>.png ([before |] creator | Blender), and with each a close crop."""
import base64
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skin_layers  # noqa: E402


def arg(k, d=None):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d


def aces_three(rgb, exposure):
    """three.js r160 ACESFilmicToneMapping, then the sRGB transfer (linear in, display sRGB 0..1 out)."""
    c = rgb * (exposure / 0.6)
    i_m = np.array([[0.59719, 0.35458, 0.04823], [0.07600, 0.90834, 0.01566], [0.02840, 0.13383, 0.83777]])
    o_m = np.array([[1.60475, -0.53108, -0.07367], [-0.10208, 1.10813, -0.00605], [-0.00327, -0.07276, 1.07602]])
    c = c @ i_m.T
    a = c * (c + 0.0245786) - 0.000090537
    b = c * (0.983729 * c + 0.4329510) + 0.238081
    c = np.clip((a / b) @ o_m.T, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def image_from_data_url(name, url, colour='sRGB'):
    raw = base64.b64decode(url.split(',', 1)[1])
    path = os.path.join(bpy.app.tempdir or '/tmp', name + '.png')
    with open(path, 'wb') as f:
        f.write(raw)
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = colour
    img.pack()
    return img


def build_mesh(name, m):
    V = np.array(m['verts'], dtype=np.float64).reshape(-1, 3)
    I = np.array(m['index'], dtype=np.int64).reshape(-1, 3)
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], I.tolist())
    me.update()
    if 'uv' in m:
        uv = np.array(m['uv']).reshape(-1, 2)
        lay = me.uv_layers.new(name='UVMap')
        loops = np.zeros(len(me.loops), dtype=np.int64)
        me.loops.foreach_get('vertex_index', loops)
        lay.data.foreach_set('uv', uv[loops].ravel())
    if 'color' in m:
        c = np.array(m['color']).reshape(-1, 3)
        a = me.color_attributes.new('occ', 'FLOAT_COLOR', 'POINT')
        a.data.foreach_set('color', np.concatenate([c, np.ones((len(c), 1))], 1).ravel())
    for p in me.polygons:
        p.use_smooth = True
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    return o, V, me


AMB = {}


def add_ambient(mat):
    """the creator's ambient on a material: its Principled base colour (the layered skin's own combined colour) times
    the ambient map at the surface's normal, as an emission beside the surface"""
    nt = mat.node_tree
    outn = [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output][0]
    surf = outn.inputs['Surface'].links[0].from_socket
    bsdfs = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED']
    b = next((n for n in bsdfs if n.name == 'close bsdf'), bsdfs[0])
    bc = b.inputs['Base Color']
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    env = nt.nodes.new('ShaderNodeTexEnvironment')
    env.image = AMB['img']
    env.interpolation = 'Linear'
    nt.links.new(geo.outputs['Normal'], env.inputs['Vector'])
    mul = nt.nodes.new('ShaderNodeMix')
    mul.data_type = 'RGBA'
    mul.blend_type = 'MULTIPLY'
    mul.inputs['Factor'].default_value = 1.0
    if bc.is_linked:
        nt.links.new(bc.links[0].from_socket, mul.inputs['A'])
    else:
        mul.inputs['A'].default_value = bc.default_value
    nt.links.new(env.outputs['Color'], mul.inputs['B'])
    envk = nt.nodes.new('ShaderNodeTexEnvironment')
    envk.image = AMB['spec']
    envk.interpolation = 'Linear'
    nt.links.new(geo.outputs['Normal'], envk.inputs['Vector'])
    addc = nt.nodes.new('ShaderNodeMix')
    addc.data_type = 'RGBA'
    addc.blend_type = 'ADD'
    addc.inputs['Factor'].default_value = 1.0
    nt.links.new(mul.outputs['Result'], addc.inputs['A'])
    nt.links.new(envk.outputs['Color'], addc.inputs['B'])
    em = nt.nodes.new('ShaderNodeEmission')
    nt.links.new(addc.outputs['Result'], em.inputs['Color'])
    add = nt.nodes.new('ShaderNodeAddShader')
    nt.links.new(surf, add.inputs[0])
    nt.links.new(em.outputs['Emission'], add.inputs[1])
    nt.links.new(add.outputs['Shader'], outn.inputs['Surface'])


def head_material(face_img, mask_img):
    """the layered skin, its epidermis the creator's painted face times its occlusion; plain where the mask is green"""
    mat = bpy.data.materials.new('creator skin')
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image = face_img
    occ = nt.nodes.new('ShaderNodeVertexColor')
    occ.layer_name = 'occ'
    tone = nt.nodes.new('ShaderNodeMix')
    tone.data_type = 'RGBA'
    tone.blend_type = 'MULTIPLY'
    tone.inputs['Factor'].default_value = 1.0
    nt.links.new(tex.outputs['Color'], tone.inputs['A'])
    nt.links.new(occ.outputs['Color'], tone.inputs['B'])
    # skin_layers writes its BSDF into an output node: give it one of its own, then take the shader from it
    tmp = nt.nodes.new('ShaderNodeOutputMaterial')
    tmp.is_active_output = False
    if '--plain' in sys.argv:       # (a check of the light alone: the painted tone on a plain surface)
        pb = nt.nodes.new('ShaderNodeBsdfPrincipled')
        if '--white' in sys.argv:
            pb.inputs['Base Color'].default_value = (0.4969, 0.4969, 0.4969, 1)      # (#bbbbbb)
        else:
            nt.links.new(tone.outputs['Result'], pb.inputs['Base Color'])
        pb.inputs['Roughness'].default_value = 0.56
        pb.inputs['IOR'].default_value = 1.5      # (three.js's standard surface: F0 0.04)
        nt.links.new(pb.outputs['BSDF'], tmp.inputs['Surface'])
    else:
        skin_layers.skin_layers(nt, tmp, 'all', tone.outputs['Result'])
    layered = tmp.inputs['Surface'].links[0].from_socket
    plain = nt.nodes.new('ShaderNodeBsdfPrincipled')
    nt.links.new(tone.outputs['Result'], plain.inputs['Base Color'])
    plain.inputs['Roughness'].default_value = 0.4
    mk = nt.nodes.new('ShaderNodeTexImage')
    mk.image = mask_img
    sep = nt.nodes.new('ShaderNodeSeparateColor')
    nt.links.new(mk.outputs['Color'], sep.inputs['Color'])
    mix = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(sep.outputs['Green'], mix.inputs['Fac'])
    nt.links.new(layered, mix.inputs[1])
    nt.links.new(plain.outputs['BSDF'], mix.inputs[2])
    nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
    out.is_active_output = True
    nt.nodes.remove(tmp)
    return mat


def main():
    dump = sys.argv[1]
    out = arg('--out', 'skin-bench')
    os.makedirs(out, exist_ok=True)
    views = arg('--views', 'front,three,cheek,nose,temple,profile').split(',')
    samples, scale, before = int(arg('--samples', '32')), float(arg('--scale', '0.5')), arg('--before')
    D = json.load(open(os.path.join(dump, 'head.json')))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    skin_layers.FRECKLES = 0.0      # (the creator's freckles are in the painted tone)

    # ---- the head, with the layered skin's lookups
    h = D['head']
    head, V, me = build_mesh('Creator head', h)
    rest = me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
    rest.data.foreach_set('vector', np.array(h['skinRest'], dtype=np.float64))
    zones = np.array(h['skinZones']).reshape(-1, 2)
    N = np.zeros(len(V) * 3)
    me.vertex_normals.foreach_get('vector', N)
    N = N.reshape(-1, 3)
    HH = 8.6
    u, w = V[:, 0] / HH, V[:, 2] / HH
    front = np.clip(-N[:, 1] * 1.5, 0, 1)

    def g(a, b, a0, b0, sa, sb):
        return np.exp(-((a - a0) / sa) ** 2 - ((b - b0) / sb) ** 2)
    gloss = front * (g(u, w, 0, 0.43, 0.04, 0.15) + 0.7 * g(u, w, 0, 0.085, 0.06, 0.05) + 0.5 * g(u, w, 0, 0.66, 0.15, 0.06))
    pores = np.clip(0.3 + front * (gloss + 0.7 * (g(u, w, -0.1, 0.39, 0.09, 0.07) + g(u, w, 0.1, 0.39, 0.09, 0.07))), 0, 1)
    st = me.color_attributes.new('skin tex', 'FLOAT_COLOR', 'POINT')
    st.data.foreach_set('color', np.stack([zones[:, 0], np.zeros(len(V)), pores, zones[:, 1]], 1).ravel())
    face_img = image_from_data_url('creator face', D['face'])
    mask_img = image_from_data_url('creator mask', D['mask'], 'Non-Color')
    head.data.materials.append(head_material(face_img, mask_img))

    for k, e in enumerate(D['eyes']):
        o, _, _ = build_mesh('Eye %d' % k, e)
        m = bpy.data.materials.new('eye %d' % k)
        m.use_nodes = True
        b = m.node_tree.nodes['Principled BSDF']
        t = m.node_tree.nodes.new('ShaderNodeTexImage')
        t.image = image_from_data_url('eye %d' % k, e['map'])
        m.node_tree.links.new(t.outputs['Color'], b.inputs['Base Color'])
        b.inputs['Roughness'].default_value = 0.22
        b.inputs['Coat Weight'].default_value = 1.0
        b.inputs['Coat Roughness'].default_value = 0.04
        o.data.materials.append(m)
    for k, e in enumerate(D['ears']):
        if not e['index']:
            continue
        o, _, _ = build_mesh('Ear %d' % k, e)
        m = bpy.data.materials.new('ear %d' % k)
        m.use_nodes = True
        b = m.node_tree.nodes['Principled BSDF']
        vc = m.node_tree.nodes.new('ShaderNodeVertexColor')
        vc.layer_name = 'occ'
        m.node_tree.links.new(vc.outputs['Color'], b.inputs['Base Color'])
        b.inputs['Roughness'].default_value = 0.62
        o.data.materials.append(m)

    # ---- the creator's lights
    for L in D['lights']:
        ld = bpy.data.lights.new(L['name'], 'SUN')
        ld.energy = L['intensity']
        ld.color = L['color']
        ld.angle = math.radians(2.0)
        ld.use_shadow = bool(L['shadow'])
        lo = bpy.data.objects.new(L['name'], ld)
        sc.collection.objects.link(lo)
        lo.rotation_euler = (-Vector(L['dir'])).to_track_quat('-Z', 'Y').to_euler()
    if D.get('bounce'):     # (the creator's bounce from below, views.py's big warm light: a wide sun)
        Bn = D['bounce']
        ld = bpy.data.lights.new('bounce', 'SUN')
        ld.energy = 1.0
        ld.color = Bn['color']
        ld.angle = math.radians(60.0)
        ld.use_shadow = False
        lo = bpy.data.objects.new('bounce', ld)
        sc.collection.objects.link(lo)
        lo.rotation_euler = (-Vector(Bn['dir'])).to_track_quat('-Z', 'Y').to_euler()
    A = D['ambient']
    W, H = A['W'], A['H']

    def amb_image(name, data):
        amb = np.array(data, dtype=np.float64).reshape(H, W, 3)
        for r in range(H):
            hole = amb[r, :, 0] < 0
            if hole.any() and (~hole).any():
                amb[r, hole] = amb[r, ~hole].mean(axis=0)
        for r in (0, H - 1):        # (the poles' rows, which the sphere never showed: their neighbours')
            if (amb[r, :, 0] < 0).all():
                amb[r] = amb[1 if r == 0 else H - 2]
        amb = np.maximum(amb, 0)
        img = bpy.data.images.new(name, W, H, float_buffer=True)
        img.colorspace_settings.name = 'Non-Color'
        img.pixels.foreach_set(np.concatenate([amb, np.ones((H, W, 1))], 2).astype(np.float32).ravel())
        return img, amb
    # (three.js's hemisphere light and room environment light every surface from every side, with no occlusion, and
    # its hemisphere is brighter above than any real sky can make it: no world gives its gradient. So, as three.js
    # does, each surface gets it directly, as an emission: its colour times what a matte surface takes at its normal
    # (the white sphere less the black), plus the room's reflection off the surface (the black sphere))
    _, white = amb_image('creator ambient white', A['white'])
    img_k, black = amb_image('creator ambient spec', A['black'])
    img, _ = amb_image('creator ambient', (white - black).ravel())
    AMB['spec'] = img_k
    AMB['img'] = img
    world = bpy.data.worlds.new('creator')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (0, 0, 0, 1)
    sc.world = world
    for o in sc.objects:
        for ms in getattr(o, 'material_slots', []):
            add_ambient(ms.material)

    # ---- render settings: linear out, no bounces between surfaces (as in three.js), transparent film
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = samples
    sc.cycles.use_denoising = '--denoise' in sys.argv
    sc.cycles.max_bounces = 4
    sc.cycles.diffuse_bounces = 0
    sc.cycles.glossy_bounces = 0
    sc.cycles.transmission_bounces = 0
    sc.cycles.transparent_max_bounces = 2
    sc.render.film_transparent = True
    sc.view_settings.view_transform = 'Standard'
    sc.view_settings.look = 'None'
    sc.render.image_settings.file_format = 'OPEN_EXR'
    sc.render.image_settings.color_depth = '32'
    Wc, Hc = D['size']
    sc.render.resolution_x, sc.render.resolution_y = int(Wc * scale), int(Hc * scale)
    sc.render.resolution_percentage = 100
    cd = bpy.data.cameras.new('View')
    cam = bpy.data.objects.new('View', cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    C = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))      # (three.js axes to Blender's)
    from PIL import Image
    for vname in views:
        c = D['cams'][vname]
        M3 = Matrix([c['matrixWorld'][r::4] for r in range(4)])
        cam.matrix_world = C @ M3
        cd.sensor_fit = 'VERTICAL'
        cd.angle_y = math.radians(c['fov'])
        cd.clip_start, cd.clip_end = 1, 200
        path = os.path.join(out, 'blender-%s.exr' % vname)
        sc.render.filepath = path
        if not ('--reuse' in sys.argv and os.path.exists(path)):      # (--reuse: keep Blender's renders, redo the sheets)
            bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(path)
        px = np.array(im.pixels[:], dtype=np.float64).reshape(im.size[1], im.size[0], 4)[::-1]
        bpy.data.images.remove(im)
        rgb = aces_three(px[..., :3], D.get('toneExposure', 1.05))
        cre = Image.open(os.path.join(dump, 'creator-%s.png' % vname)).convert('RGB')
        bg = np.array([0x2b, 0x30, 0x2e]) / 255.0      # (the stage behind the creator's canvas, --stage)
        a = px[..., 3:4]
        comp = (rgb * a + bg * (1 - a)) * 255
        bl = Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8)).resize(cre.size, Image.LANCZOS)
        bl.save(os.path.join(out, 'blender-%s.png' % vname))
        panels = ([Image.open(os.path.join(before, 'b-ranger-%s.png' % vname)).convert('RGB')] if before else []) + [cre, bl]
        sheet = Image.new('RGB', (cre.width * len(panels), cre.height))
        for i, pnl in enumerate(panels):
            sheet.paste(pnl, (i * cre.width, 0))
        sheet.save(os.path.join(out, 'compare-%s.png' % vname))
        print('view', vname, 'done', flush=True)


if __name__ == '__main__':
    main()
