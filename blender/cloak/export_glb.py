"""Export the cloak on the mannequin as a glTF model for the cape bench and the game's cape rig
(public/models/cloak-mannequin.glb): the bare mannequin (its skin thinned for the web), its rig, and the cloak as
fitted and settled. The cloak's mesh (wool outside, lining in) is named 'Cape', so the bench's model body
(src/cape-lab/model-body.js) re-skins it to the cape's spring-bone chains exactly as it does the Warden's cape, and
src/world/cape-rig.js (fabric walking, spring bones running) drives it.

    python3 blender/cloak/export_glb.py [--out public/models/cloak-mannequin.glb]
then open  cape-lab.html?body=model&model=models/cloak-mannequin.glb&capespan=3.0&walk=new"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bpy  # noqa: E402
import cloak  # noqa: E402
import female  # noqa: E402


def lin(c):
    return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92


def plain_materials():
    for mid, d in female.MATERIALS.items():
        m = bpy.data.materials.get(d['name'])
        if m is None or not m.use_nodes:
            continue
        nt = m.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new('ShaderNodeOutputMaterial')
        p = nt.nodes.new('ShaderNodeBsdfPrincipled')
        rgb = d['ramp'][2] / 255.0
        p.inputs['Base Color'].default_value = (*[lin(c) for c in rgb], 1.0)
        metal = d['name'] == 'cloak clasp'
        p.inputs['Metallic'].default_value = 0.8 if metal else 0.0
        p.inputs['Roughness'].default_value = 0.45 if metal else 0.88
        nt.links.new(p.outputs['BSDF'], out.inputs['Surface'])


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    out = os.path.abspath(argv[argv.index('--out') + 1] if '--out' in argv else
                          os.path.join(HERE, '..', '..', 'public', 'models', 'cloak-mannequin.glb'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    female.MANNEQUIN = True
    cloak.build()
    sc = bpy.context.scene
    keep = {'Skin body', 'Neck', 'Hand.L', 'Hand.R', 'Head shape', 'Cloak', 'Cloak clasp', 'Rig'}
    for o in list(sc.objects):
        if o.type == 'MESH' and o.name not in keep:
            bpy.data.objects.remove(o)
    for o in sc.objects:
        if o.animation_data:
            o.animation_data_clear()
    # (the cloak is the cape the bench re-skins; its stitches ride the chest and would part from a swinging hem,
    # so the game model goes without them)
    bpy.data.objects['Cloak'].name = 'Cape'
    sk = bpy.data.objects['Skin body']
    d = sk.modifiers.new('lighter', 'DECIMATE')
    d.ratio = 0.25
    plain_materials()
    for o in sc.objects:
        if o.type == 'MESH':
            for a in list(o.data.color_attributes):
                o.data.color_attributes.remove(a)
    bpy.ops.export_scene.gltf(
        filepath=out, export_format='GLB', export_apply=True, export_yup=True, export_animations=False,
        export_skins=True, export_attributes=False, export_texcoords=True, export_materials='EXPORT',
        export_cameras=False, export_lights=False)
    print('wrote', out, os.path.getsize(out) // 1024, 'KB')


if __name__ == '__main__':
    main()
