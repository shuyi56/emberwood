"""The hair bench: a hair asset (tools/hair-asset/FORMAT.md) rendered in Blender on the face creator's own cameras,
lights and tone mapping, beside the creator's reference renders of the same style, with a difference map. It is the
check that Blender draws a hairstyle as the creator does; it builds no figure, only the hair and the creator's head.

    python3 blender/shared/hair_bench.py ASSET.hair.glb --head head.glb --ref DIR --out DIR
        [--views front,three,profile,back3,back] [--looks clay,full] [--size 512] [--samples 16] [--denoise]
        [--crops] [--legacy WARDEN_HAIR_JSON]

ASSET, the head and the reference renders come from tools/hair-asset/export.cjs (--views, --head, --probe). For each
view and look it writes blender-<view>-<look>.png and compare-<view>-<look>.png (creator | Blender | difference, the
difference x4, red where Blender is lighter, blue where darker), and with --crops close-ups of the hairline and the
ends. It prints the mean and 95th percentile difference over the hair (0..255 sRGB; where Blender's hair is in front,
from a second pass with the head held out), and how much of the frame the hair and head cover in each.

How the creator's light is rebuilt (each part checked light by light against three.js on a sphere and on the hair): its three directional lights as suns of the same colour and intensity (three.js
r155+ and Cycles both give a matte white surface intensity / pi), the key alone casting shadows; its hemisphere light
and its room environment (scaled by the materials' envMapIntensity, 0.25 for hair and clay alike) as light that no
geometry occludes, as in three.js, which has no ambient occlusion: an emission of what three.js's own shading takes
from them at the surface's normal (the exporter's environmentResponse and environmentSpecular maps, a matte white
and a matte black sphere lit by the room alone, so a surface of colour c gets c x (white - black) + black; plus the
hemisphere's own formula). The hair takes no shadow (hair_asset.unshadow); the head does. The render is
linear (view transform Standard) and tone-mapped afterwards with three.js's ACES filmic curve and exposure. Light does
not bounce between surfaces (no diffuse or glossy bounces), as in three.js: with bounces, the inside of the hair, which
faces away from the lights, was lit by the hair in front of it.
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hair_asset  # noqa: E402


def arg(k, d=None):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d


# ---------------------------------------------------------------- three.js's tone mapping
def aces_three(rgb, exposure):
    """three.js r160 ACESFilmicToneMapping, then the sRGB transfer (linear in, display sRGB 0..1 out)."""
    c = rgb * (exposure / 0.6)
    i_m = np.array([[0.59719, 0.35458, 0.04823], [0.07600, 0.90834, 0.01566], [0.02840, 0.13383, 0.83777]])
    o_m = np.array([[1.60475, -0.53108, -0.07367], [-0.10208, 1.10813, -0.00605], [-0.00327, -0.07276, 1.07602]])
    c = c @ i_m.T
    a = c * (c + 0.0245786) - 0.000090537
    b = c * (0.983729 * c + 0.4329510) + 0.238081
    c = (a / b) @ o_m.T
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


# ---------------------------------------------------------------- the creator's unoccluded light
def _map_image(name, data, W, H):
    data = data.reshape(H, W, 3).copy()
    for r in range(H):      # (bins the sphere never showed, at the poles: the rest of their row)
        hole = data[r].sum(axis=1) <= 0
        if hole.any() and (~hole).any():
            data[r, hole] = data[r, ~hole].mean(axis=0)
    img = bpy.data.images.new(name, W, H, alpha=True, float_buffer=True)
    img.colorspace_settings.name = 'Non-Color'
    img.pixels.foreach_set(np.concatenate([data, np.ones((H, W, 1))], 2).astype(np.float32).ravel())
    img.pack()
    return img


def ambient_group(white, black, W, H, env_k, hemi, name='creator ambient'):
    """A node group: Normal in; out the creator's ambient light for a surface of colour c, as c x Light + Spec.
    Light: the room environment's response three.js gives a matte surface at that normal (the exporter's maps:
    white minus black) plus the hemisphere light, mix(ground, sky, 0.5 n.up + 0.5) x intensity / pi; Spec: the part
    that does not take the surface's colour (the black surface's response: its sheen of the room). Both times the
    material's envMapIntensity. Unoccluded, as three.js has no ambient occlusion."""
    g = bpy.data.node_groups.new(name, 'ShaderNodeTree')
    g.interface.new_socket('Normal', in_out='INPUT', socket_type='NodeSocketVector')
    g.interface.new_socket('Light', in_out='OUTPUT', socket_type='NodeSocketColor')
    g.interface.new_socket('Spec', in_out='OUTPUT', socket_type='NodeSocketColor')
    nt, L = g.nodes, g.links
    gi, go = nt.new('NodeGroupInput'), nt.new('NodeGroupOutput')
    sep = nt.new('ShaderNodeSeparateXYZ')
    L.new(gi.outputs['Normal'], sep.inputs['Vector'])
    x, y, z = sep.outputs['X'], sep.outputs['Y'], sep.outputs['Z']

    def m(op, a, b=0.0):
        n = nt.new('ShaderNodeMath')
        n.operation = op
        for k, v in enumerate((a, b)):
            if isinstance(v, (int, float)):
                n.inputs[k].default_value = v
            else:
                L.new(v, n.inputs[k])
        return n.outputs[0]

    def vm(op, a, b, k=None):
        n = nt.new('ShaderNodeVectorMath')
        n.operation = op
        L.new(a, n.inputs[0])
        if isinstance(b, (int, float)):
            n.inputs['Scale'].default_value = b
        else:
            L.new(b, n.inputs[1])
        return n.outputs[0]
    # (the maps' layout: u = atan2(y, -x) / 2pi + 0.5, v = atan2(z, |xy|) / pi + 0.5)
    u = m('ADD', m('MULTIPLY', m('ARCTAN2', y, m('MULTIPLY', x, -1.0)), 1 / (2 * math.pi)), 0.5)
    v = m('ADD', m('MULTIPLY', m('ARCTAN2', z, m('SQRT', m('ADD', m('MULTIPLY', x, x), m('MULTIPLY', y, y)))), 1 / math.pi), 0.5)
    uv = nt.new('ShaderNodeCombineXYZ')
    L.new(u, uv.inputs[0])
    L.new(v, uv.inputs[1])
    out = {}
    for key, data in (('white', white), ('black', black)):
        tx = nt.new('ShaderNodeTexImage')
        tx.image = _map_image('creator ambient ' + key, data, W, H)
        tx.interpolation = 'Linear'
        tx.extension = 'EXTEND'
        L.new(uv.outputs[0], tx.inputs['Vector'])
        out[key] = vm('SCALE', tx.outputs['Color'], env_k)
    s = vm('SUBTRACT', out['white'], out['black'])
    if hemi:
        sky = np.array(hemi['color']) * hemi['intensity'] / math.pi
        gr = np.array(hemi['groundColor']) * hemi['intensity'] / math.pi
        w = m('ADD', m('MULTIPLY', z, 0.5), 0.5)
        hv = nt.new('ShaderNodeCombineXYZ')
        for c in range(3):
            L.new(w, hv.inputs[c])
        ma = nt.new('ShaderNodeVectorMath')
        ma.operation = 'MULTIPLY_ADD'
        ma.inputs[0].default_value = tuple(float(q) for q in sky - gr)
        L.new(hv.outputs[0], ma.inputs[1])
        ma.inputs[2].default_value = tuple(float(q) for q in gr)
        s = vm('ADD', s, ma.outputs[0])
    L.new(s, go.inputs['Light'])
    L.new(out['black'], go.inputs['Spec'])
    return g


def add_ambient(mat, group):
    """Add the creator's unoccluded ambient to a material: an emission of its Principled base colour times the
    group's Light, plus its Spec, beside the BSDF (inside any cut-out mix, so cut-out parts stay clear)."""
    nt = mat.node_tree
    for b in [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED']:
        bc = b.inputs['Base Color']
        src = bc.links[0].from_socket if bc.is_linked else None
        geo = nt.nodes.new('ShaderNodeNewGeometry')
        gn = nt.nodes.new('ShaderNodeGroup')
        gn.node_tree = group
        nt.links.new(geo.outputs['Normal'], gn.inputs['Normal'])
        mul = nt.nodes.new('ShaderNodeVectorMath')
        mul.operation = 'MULTIPLY_ADD'
        if src:
            nt.links.new(src, mul.inputs[0])
        else:
            mul.inputs[0].default_value = tuple(bc.default_value)[:3]
        nt.links.new(gn.outputs['Light'], mul.inputs[1])
        nt.links.new(gn.outputs['Spec'], mul.inputs[2])
        em = nt.nodes.new('ShaderNodeEmission')
        nt.links.new(mul.outputs[0], em.inputs['Color'])
        add = nt.nodes.new('ShaderNodeAddShader')
        users = [l_.to_socket for l_ in nt.links if l_.from_socket == b.outputs['BSDF']]
        nt.links.new(b.outputs['BSDF'], add.inputs[0])
        nt.links.new(em.outputs['Emission'], add.inputs[1])
        for s in users:
            nt.links.new(add.outputs['Shader'], s)


def creator_lights(V):
    """The creator's directional lights as suns; returns the hemisphere light (or None)."""
    hemi = None
    for L in V['lights']:
        if L['type'] == 'HemisphereLight':
            hemi = L
            continue
        if L['type'] != 'DirectionalLight':
            continue
        d = bpy.data.lights.new('creator ' + str(len(bpy.data.lights)), 'SUN')
        d.energy = L['intensity']
        d.color = L['color']
        d.angle = math.radians(1.5)
        d.use_shadow = bool(L.get('castShadow'))
        o = bpy.data.objects.new(d.name, d)
        bpy.context.scene.collection.objects.link(o)
        t = Vector(hair_asset.to_blender(L['toLight'], 1.0)).normalized()
        o.rotation_euler = t.to_track_quat('Z', 'Y').to_euler()
    return hemi


def build_head(path, scale=1.0):
    A = hair_asset.Asset.__new__(hair_asset.Asset)
    import json
    import struct
    with open(path, 'rb') as f:
        buf = f.read()
    off, A.json, A.bin = 12, None, None
    while off < len(buf):
        ln, ty = struct.unpack_from('<II', buf, off)
        body = buf[off + 8:off + 8 + ln]
        if ty == 0x4E4F534A:
            A.json = json.loads(body.decode('utf8'))
        else:
            A.bin = body
        off += 8 + ln
    pr = A.json['meshes'][0]['primitives'][0]
    pos = hair_asset.to_blender(A.acc(pr['attributes']['POSITION']), scale)
    nrm = hair_asset.to_blender(A.acc(pr['attributes']['NORMAL']), 1.0)
    idx = A.acc(pr['indices'])[:, 0]
    n = len(pos)
    me = hair_asset._mesh('creator head', pos, nrm, np.zeros((n, 2)), np.ones((n, 4)), idx, {})
    o = bpy.data.objects.new('creator head', me)
    bpy.context.scene.collection.objects.link(o)
    m = bpy.data.materials.new('clay')
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    g = (0x9a / 255.0 + 0.055) / 1.055
    b.inputs['Base Color'].default_value = (g ** 2.4, g ** 2.4, g ** 2.4, 1.0)      # (the creator's clay, 0x9a9a9a)
    b.inputs['Roughness'].default_value = 1.0
    b.inputs['Specular IOR Level'].default_value = 0.5
    me.materials.append(m)
    return o


def legacy_hair(A):
    """The hair as Blender builds it today, for the before: blender/warden/warden.py's creator_hair from the older
    export beside it (hair_creator.json.gz: the cap and the strands' centre lines, rebuilt as ribbons, with its own
    material). Built in the bench's frame (the creator's 8.6 head), at the head's place."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, '..', 'warden'))
    import warden  # noqa: E402
    head = bpy.data.objects.new('legacy head', None)
    bpy.context.scene.collection.objects.link(head)
    head.scale = (8.6 / warden.HEAD_H,) * 3
    cap, locks = warden.creator_hair_legacy(head)
    objs = [cap] + list(locks)
    for o in objs:
        o['receive_shadow'] = True       # (it takes the scene's shadows, as on the figure)
    return objs


def camera(c):
    cd = bpy.data.cameras.new('bench')
    cd.sensor_fit = 'VERTICAL'
    cd.angle_y = math.radians(c['fovY'])
    cd.clip_start, cd.clip_end = 1.0, 200.0
    o = bpy.data.objects.new('bench', cd)
    bpy.context.scene.collection.objects.link(o)
    o.location = Vector(hair_asset.to_blender(c['position'], 1.0))
    tgt = Vector(hair_asset.to_blender(c['target'], 1.0))
    o.rotation_euler = (o.location - tgt).to_track_quat('Z', 'Y').to_euler()
    return o


def read_png(p):
    img = bpy.data.images.load(p)
    W, H = img.size
    a = np.array(img.pixels[:], dtype=np.float32).reshape(H, W, 4)[::-1]
    bpy.data.images.remove(img)
    return a


def write_png(p, a):
    H, W = a.shape[:2]
    img = bpy.data.images.new(os.path.basename(p), W, H, alpha=True)
    img.pixels.foreach_set(np.ascontiguousarray(a[::-1]).astype(np.float32).ravel())
    img.filepath_raw = p
    img.file_format = 'PNG'
    img.save()
    bpy.data.images.remove(img)


def over(a, bg):
    return a[..., :3] * a[..., 3:4] + bg * (1 - a[..., 3:4])


def main():
    av = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    asset = av[0]
    out = arg('--out', 'hair-bench')
    os.makedirs(out, exist_ok=True)
    ref = arg('--ref', os.path.dirname(asset))
    views = arg('--views', 'front,three,profile,back3,back').split(',')
    looks = arg('--looks', 'clay,full').split(',')
    size = int(arg('--size', '512'))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = int(arg('--samples', '16'))
    sc.cycles.use_denoising = '--denoise' in sys.argv
    # (three.js lights each surface by itself, with no light bouncing between surfaces: no diffuse or glossy bounces,
    # only the light passing through the cut-out)
    sc.cycles.max_bounces = 64
    sc.cycles.diffuse_bounces = sc.cycles.glossy_bounces = sc.cycles.transmission_bounces = sc.cycles.volume_bounces = 0
    sc.cycles.transparent_max_bounces = 64
    sc.render.film_transparent = True
    sc.render.filter_size = 1.0
    sc.render.resolution_x = sc.render.resolution_y = size
    sc.view_settings.view_transform = 'Standard'
    sc.view_settings.look = 'None'
    sc.render.image_settings.file_format = 'OPEN_EXR'
    w = bpy.data.worlds.new('none')
    w.use_nodes = True
    w.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.0
    sc.world = w
    A = hair_asset.load(asset)
    V = A.ext['viewer']
    hemi = creator_lights(V)
    # the unoccluded ambient: the room environment at the hair's envMapIntensity plus the hemisphere light
    env_k = next((m.get('envMapIntensity', 1.0) for m in A.materials if m['kind'] == 'cut'), 0.25)
    er, es = V.get('environmentResponse'), V.get('environmentSpecular')
    if er:
        white = A.acc(er['accessor']).astype(np.float64)
        black = A.acc(es['accessor']).astype(np.float64) if es else np.zeros_like(white)
        W_, H_ = er['width'], er['height']
    else:
        print('the asset has no environmentResponse (export with --probe): the room environment is left out')
        white = black = np.zeros((2, 3))
        W_, H_ = 2, 1
    grp = ambient_group(white, black, W_, H_, env_k, hemi)
    head = build_head(arg('--head')) if arg('--head') else None
    if head:
        add_ambient(head.data.materials[0], grp)
    stats = []
    if '--legacy' in sys.argv:
        looks = looks + ['legacy']
    for look in looks:
        if look == 'legacy':
            objs = legacy_hair(A)
            ref_look = 'full'
        else:
            objs = hair_asset.build(A, look=look, name='Hair ' + look)
            ref_look = look
            for o in objs:
                for m in o.data.materials:
                    if m.get('emberwood') == 'cut':
                        add_ambient(m, grp)
        hair_asset.unshadow(objs)
        for vname in views:
            c = A.ext['cameras'].get(vname)
            if not c:
                print('no camera for', vname)
                continue
            cam = camera(c)
            sc.camera = cam
            dist = (Vector(c['position']) - Vector(c['target'])).length
            hair_asset.set_view(objs, A, hair_asset.pixel_size(cam, dist, size))
            exr = os.path.join(out, 'blender-%s-%s.exr' % (vname, look))
            sc.render.filepath = exr
            bpy.ops.render.render(write_still=True)
            img = bpy.data.images.load(exr)
            px = np.array(img.pixels[:], dtype=np.float64).reshape(size, size, 4)[::-1]
            bpy.data.images.remove(img)
            os.remove(exr)
            # (Cycles' alpha is premultiplied: tone-map the colour unpremultiplied, as three.js draws it over clear)
            al = np.clip(px[..., 3:4], 1e-6, 1)
            rgb = aces_three(px[..., :3] / al, V.get('exposure', 1.0))
            bl = np.concatenate([rgb, px[..., 3:4]], axis=2)
            write_png(os.path.join(out, 'blender-%s-%s.png' % (vname, look)), bl)
            # (where the hair is, for the numbers: the same shot with the head held out, at a few samples)
            hair_a = None
            if head:
                keep_m, keep_s = head.data.materials[0], sc.cycles.samples
                hold = bpy.data.materials.get('holdout') or bpy.data.materials.new('holdout')
                if not hold.node_tree or 'Holdout' not in hold.node_tree.nodes:
                    hold.use_nodes = True
                    hold.node_tree.nodes.clear()
                    ho = hold.node_tree.nodes.new('ShaderNodeHoldout')
                    hold.node_tree.links.new(ho.outputs[0], hold.node_tree.nodes.new('ShaderNodeOutputMaterial').inputs['Surface'])
                head.data.materials[0] = hold
                sc.cycles.samples = 4
                bpy.ops.render.render(write_still=True)
                img = bpy.data.images.load(exr)
                hair_a = np.array(img.pixels[:], dtype=np.float64).reshape(size, size, 4)[::-1, :, 3]
                bpy.data.images.remove(img)
                os.remove(exr)
                head.data.materials[0], sc.cycles.samples = keep_m, keep_s
            rp = os.path.join(ref, 'creator-%s-%s.png' % (vname, ref_look))
            if os.path.exists(rp):
                cr = read_png(rp)
                bg = np.array([0.16, 0.18, 0.18])
                a1, a2 = over(cr, bg), over(bl, bg)
                d = a2 - a1
                lum = d.mean(axis=2)
                dm = np.stack([np.clip(0.5 + 2 * lum, 0, 1), np.clip(0.5 - 2 * np.abs(lum), 0, 1), np.clip(0.5 - 2 * lum, 0, 1)], 2)
                hairmask = (hair_a > 0.5) if hair_a is not None else ((cr[..., 3] > 0.5) | (bl[..., 3] > 0.5))
                e = np.abs(d).mean(axis=2)[hairmask] * 255
                stats.append((vname, look, float(e.mean()), float(np.percentile(e, 95)), float((cr[..., 3] > 0.5).mean()), float((bl[..., 3] > 0.5).mean())))
                print('%-8s %-5s  mean %.1f  p95 %.1f  coverage creator %.3f blender %.3f' % stats[-1], flush=True)
                row = np.concatenate([a1, a2, dm], axis=1)
                write_png(os.path.join(out, 'compare-%s-%s.png' % (vname, look)), np.concatenate([row, np.ones(row.shape[:2] + (1,))], 2))
                if '--crops' in sys.argv:
                    # the hairline (upper middle) and the ends (lower part), 3x
                    for cname, (y0, x0, s) in (('hairline', (int(size * 0.12), int(size * 0.3), int(size * 0.4))),
                                               ('ends', (int(size * 0.62), int(size * 0.1), int(size * 0.4)))):
                        crop = np.concatenate([a1[y0:y0 + s, x0:x0 + s], a2[y0:y0 + s, x0:x0 + s]], axis=1)
                        crop = crop.repeat(2, 0).repeat(2, 1)
                        write_png(os.path.join(out, 'crop-%s-%s-%s.png' % (vname, look, cname)), np.concatenate([crop, np.ones(crop.shape[:2] + (1,))], 2))
            bpy.data.objects.remove(cam)
        for o in objs:
            o.hide_render = True
        for L in [o for o in sc.objects if o.get('hair_twin')]:
            bpy.data.objects.remove(L)
    with open(os.path.join(out, 'stats.txt'), 'w') as f:
        for s in stats:
            f.write('%-8s %-5s  mean %.1f  p95 %.1f  coverage creator %.3f blender %.3f\n' % s)


if __name__ == '__main__':
    main()
