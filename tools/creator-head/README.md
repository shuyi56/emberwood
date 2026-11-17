# The Face Creator's head on the world model

`public/models/ranger.glb` carries the Face Creator's Ranger head and hair in place of the Blender head. The body,
clothes, skinning and the walk and idle animations are the Blender export (`blender/ranger/export_glb.py`) as before.

1. `node export-head.js <ranger-body-viewer.html> creator-head.glb [settings.json]` runs the Ranger body viewer
   (project files: `face-creator/body-viewer/`, rebuilt from the creator's current `main.js`) in headless Chromium
   and writes its head and hair as a glb, in the frame of the model's Head node (chin at 0, face toward +z). A
   settings JSON copied from the creator ("Copy settings (JSON)") exports that face instead of the Ranger preset.
   The hair strands' alpha maps are baked into the alpha of their colour maps (alphaMode MASK); glTF has no alpha
   map. The creator's shader tweaks (onBeforeCompile) do not export, so the model gets each plain material and its
   painted maps. So the export repaints the face without the creator's fine skin texture and blotches, which only its
   shader balances (in the world they read as a grey-green mottle). For the world it also flattens the lips to a painted
   shape (lips 0.3 and 0.35, no gloss), and leaves out the fine strand clumps laid over the locks behind the front of
   the head (FRONT_Z, default 1.3): at the world's size they only speckle the hair. The bangs and the locks framing
   the face stay. Only the game model changes; the creator keeps its own lips and strands.
2. `node merge-head.mjs ranger.glb creator-head.glb out.glb [headRatio] [hairRatio]` drops the Blender head, hair,
   face paint and eye animation channels, simplifies the creator's parts for the world (the head to about 25k
   triangles, the joined strands to about 0.4) and hangs them under the Head node. It drops the face's roughness map (oily patches in the world's light) and checks
   the face's mean colour against the Blender head's skin. It blurs the hair texture across the strands and softens
   its cut-out tips, which speckled in the world (hairSoft, default 5). Also needs `sharp` (0.33.5). Needs `@gltf-transform/core`,
   `@gltf-transform/extensions`, `@gltf-transform/functions` (4.1.1) and `meshoptimizer` (0.22.0).

Rerun both after a new Blender export (export_glb.py writes the Blender head back) or a change to the creator.
