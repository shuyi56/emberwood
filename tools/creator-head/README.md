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
   shader balances (in the world they read as a grey-green mottle). For the world it also softens the mouth (lips 0.55
   and 0.6, no gloss, a little wider with the corners up, slimmer cheeks round it), and leaves out the fine strand clumps laid over the locks behind the front of
   the head (FRONT_Z, default 1.3): at the world's size they only speckle the hair. The bangs and the locks framing
   the face stay. Only the game model changes; the creator keeps its own lips and strands.
2. `node merge-head.mjs ranger.glb creator-head.glb out.glb [headRatio] [hairRatio]` drops the Blender head, hair,
   face paint and eye animation channels, simplifies the creator's parts for the world (the head, alone, to 0.25 of
   its triangles, about 67k; the joined strands to about 0.4) and hangs them under the Head node. It smooths the
   normals over the mouth (MOUTH_ITERS, MOUTH_MAX), so the sculpted folds round it stop drawing a frown in the world's
   light, and leaves out the ears (under her hair) and the hair's unlit veil (flat orange patches at the temples). It drops the face's roughness map (oily patches in the world's light) and checks
   the face's mean colour against the Blender head's skin. It blurs the hair texture across the strands and softens
   its cut-out tips, which speckled in the world (hairSoft, default 5), and evens out and lifts the hair's shading
   (HAIR_EVEN 0.85, HAIR_LIFT 0.45: a little of its own colour as emission), which went dark and streaky in the sun. Also needs `sharp` (0.33.5). Needs `@gltf-transform/core`,
   `@gltf-transform/extensions`, `@gltf-transform/functions` (4.1.1) and `meshoptimizer` (0.22.0).

Rerun both after a new Blender export (export_glb.py writes the Blender head back) or a change to the creator. After a
change to a face, check its topology on what they wrote before the render (report-only; `--label` an approved head's
name, as in `tools/render-queue/topo_baseline.json`, to judge it on its own approved band):

    python3 tools/render-queue/rq_topo.py --glb creator-head.glb --out DIR --label "creator head (Ranger), raw export"

`merge-head.mjs` runs `clean-head.mjs` on the head after the simplify, so a game head comes out without holes, folds,
duplicate or unused vertices (the simplify leaves those at the poles and on thin slivers). It also runs on its own:
`node clean-head.mjs in.glb out.glb`.

## The Warden

Her body and clothes come from `blender/warden/export_glb.py` (it needs `blender/warden/warden.py` and the base
figure, and bpy with Pillow and SciPy). Her head is the creator's "Warden" preset (`warden-face.json`, dumped from the
creator), exported with her own mouth and softer make-up for the world, and merged with her body cut down for the game:

    python blender/warden/export_glb.py --out warden-blender.glb
    WORLD='{"smile":0.6,"mouthWidth":1.06,"lipUpper":0.6,"lipLower":0.7,"apples":0.65,"cheekFull":0.55,"chinProj":1.3,"flushAmt":0.7,"blush":0.15,"noseFlush":0.1}' \
      node export-head.js ranger-body-viewer.html warden-head.glb warden-face.json
    BODY_TRIS=12000 node merge-head.mjs warden-blender.glb warden-head.glb public/models/warden.glb

With no vertex colours on her Blender head, the face's skin is matched to her `skin` material (her neck and hands).
