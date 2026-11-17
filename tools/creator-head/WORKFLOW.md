# Putting a character in the world

A character in the 3D world is two halves, made in two places and joined in one step:

- **Blender, for the body and clothes only.** `blender/<character>/<character>.py` builds the figure on the
  base female body; `blender/<character>/export_glb.py` exports it with the rig, the walk (frames 1-25) and the
  idle (frame 40). Whatever head and hair the Blender script builds are thrown away in the merge.
- **The Face Creator, for the face and hair only.** The character's preset (`FACES.<Name>` and its hairstyle in
  `/mnt/project-files/face-creator/main.js`) is the one and only source of her face, skin, make-up and hair in
  the game. Change a face or a hairstyle there, never in Blender.

## Steps

1. **Clothes.** Export the body from the character's Blender branch (bpy 5.x works headless:
   `pip install bpy pillow scipy numpy OpenEXR`):

       python blender/<character>/export_glb.py --out <character>-blender.glb

2. **Face and hair.** Rebuild the body viewer from the current creator (`python3 build-viewer.py` in
   `/mnt/project-files/face-creator/body-viewer/`), dump the preset's settings (click its face button and save
   `window.__fc.P`, as `warden-face.json` here was), then export the head:

       WORLD='{...}' node export-head.js ranger-body-viewer.html <character>-head.glb <character>-face.json

   `export-head.js` already turns off what only the creator's shader balances (skin texture, blotches, lip
   gloss) and softens the mouth. `WORLD` overrides those defaults per character. Put back her own mouth shape
   (`smile`, `mouthWidth`, `lipUpper`, `lipLower`, cheeks) and halve strong make-up (`flushAmt`, `blush`):
   the world's plain light makes both read twice as strong.

3. **Merge.**

       BODY_TRIS=12000 node merge-head.mjs <character>-blender.glb <character>-head.glb public/models/<character>.glb

   It drops the Blender head and hair, hangs the creator's under the `Head` node and matches the face's skin to
   the Blender head (or to the `skin` material). It also smooths the mouth's shading, evens out the hair's, and
   caps each body part at `BODY_TRIS`. A cape lining (`LINING`) is turned to face inward, and the embroidery
   is lifted off the cloth (`TRIM_LIFT`).

4. **Roster.** Add her to `ROSTER` in `src/shared/party.js` with `model: 'models/<character>.glb'`.

5. **Check in the world.** Run `npx vite --port 5173 --host 127.0.0.1`, then

       node tools/world-capture/capture.mjs --member <character> --views front,front3q,side,back3q --dist 5

   Look at all four views and a few walk frames (`--frames 12`), and zoom into the face (`--dist 3`).

## What to look for

- **Hair against the clothes.** Anything framing the hair (cape, robe, hood, collar) must differ from the hair in
  hue, not only in value. Check it in the side and back views.
- **Layers that break through.** Two sheets of cloth close together (a lining under a cape, trim on a tabard)
  z-fight at the world's distance and push through each other when she walks. Turn an inner layer to face
  inward and draw it single-sided, and lift trims off the cloth. Don't try to fix this with offsets alone.
- **Face.** The skin should match the neck and hands, the blush should read as soft, and the mouth should
  have no dark creases around it.
- **Hair.** Look for speckle and dark streaks on the bangs, and for parts that don't carry over (the creator's
  unlit hair veil, ears poking through).
- **Size.** Keep each model under about 50 MB and 1M triangles.
