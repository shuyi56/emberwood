# Cape bench

The Warden's cape on spring bones, on a stand-in body, so the springs can be tuned without her model. The solver
(`src/world/springbones.js`) and the cape's settings and body colliders (`src/world/cape-rig.js`) are the ones the
game will use; only the body and the walk are stand-ins (`src/cape-lab/dummy.js`). The stand-in has the Warden's
joint names and rest positions from `public/models/warden.glb`, and its cape her cape's size: from the collar
(52.7) over the shoulders to just under the knees (19), flaring to 14.6 either side.

    npm run dev        # then open http://localhost:5173/cape-lab.html

The page has the motions (straight walk, start and stop, sharp turns, run, turn on the spot), five camera views
and a free orbit, sliders for every setting, and toggles for the body, the colliders and the bones. "Copy
settings" puts the tuned values on the clipboard as JSON, to paste into `CAPE_SETTINGS`.

The Body menu swaps the stand-in for the Warden's own model (`public/models/warden.glb`, or another glb under
`public/` given as `?model=<path>`), played the way the world plays it. Her own cape meshes are re-skinned in the
page to `cape_<c>_<r>` chains under her Chest joint (`src/cape-lab/model-body.js`), the same layout and weights
`warden.py` would give them, so the springs run on her real cape and body before the Blender side is ported.
Captures take `--body model` (add `?model=...` to `--url` for another glb). A head with the creator's `ponytail` style (exported with
`FRONT_Z=-100`, so `export-head.js` keeps the tail's strands, and merged with `merge-head.mjs`) gets its tail on
one chain, `hairT_0_<row>`, from the tie; the rest of her hair rides her head.

The Body menu also has the new mannequin (`?body=mannequin`, `--body mannequin` for captures):
`public/models/mannequin.glb`, the bare figure of the project's `anatomy/mannequin-v14.blend` on its own rig. It wears
the stand-in's cape and plays the baked walk (`--walk new`), as it has neither of its own.
It also wears the Face Creator head and hair of another model (the Warden's glb by default, `?head=<glb under public/>`, or
`?head=none` for its own blank head), and the hair hangs on spring-bone chains as the Warden's does.

## Springs or fabric

The cape runs on either of two solvers over the same bones, colliders and fitting. Spring bones
(`src/world/springbones.js`) swing each bone about its parent and pull it back to its modelled direction. Fabric
(`src/world/fabric.js`, on the solver in `src/world/fabric-core.js`) solves every tail as a point of one sheet of
cloth: threads that keep their length, diagonals, ties that make it roll into folds, and friction where it lies
on the body. Open the page with `?solver=fabric`, and pass `--solver fabric` to `check.mjs` and `capture.mjs`.
Its settings are `CAPE_FABRIC_SETTINGS` in `src/world/cape-rig.js`; in the game, `attachCapeFabric(model)` takes
the place of `attachCapeSprings(model)`.

## The cape rig

Each column of the cape is a chain `cape_<column>_0 … cape_<column>_<rows>` under `Chest`, plus an empty
`cape_<column>_tip` at the hem that gives the last bone its length. Row 0 is the band from the collar over the
shoulders: it rides the chest and is not simulated. The rows under it swing. The bench defaults to 7 columns (the
outer two run down the front edges) and 5 swinging rows; other counts are on the page to compare.

The motions turn as the game turns a hero (the heading closes on its goal at 12 a second, so a reversal starts at
over 30 radians a second), which is the hardest case for the cape.

## Captures and numbers

    node tools/cape-lab/capture.mjs --url http://127.0.0.1:5173/cape-lab.html --out /tmp/cape \
         [--scenarios turn,startstop,walk,run,spin] [--views back3q,side,back] [--every 3] [--colliders]
    node tools/cape-lab/check.mjs [--settings '{"damping":0.25}'] [--columns 7 --rows 6]

`capture.mjs` writes a GIF, a sheet of 8 frames and the frames themselves for each motion and view. `check.mjs`
runs the same simulation in node in a few seconds and prints:

- the loop drift (how far the cape is from where it was one walk cycle earlier; 0 means it repeats exactly);
- the hem's largest swing walking and turning, and how long it takes to settle after a stop;
- how far the hem lands at 30 and 144 fps from where it does at 60;
- how close any bone comes to the body's capsules, and how deep any vertex of the cloth goes into them through
  the sharp turns (where an arm would show through);
- the jitter: the mean frame-to-frame jerk of the cape walking, starting and stopping, and turning.

Corrections that move the cloth (collisions, the swing limit, sheet bend) are applied to its last position too,
so they never add speed. On reset the cloth settles against the body for a second of steps without gaining
speed, so it doesn't jump on the first frames.

What the settings do, from sweeps with `check.mjs`:

- Damping is the main control for sway (0.1 keeps swinging 2 s after a stop; 0.3 settles in about 0.6 s).
- Follow turns stops the game's snap turns from slinging the cape: at 0 the hem swings 110° in a reversal, at
  0.5 about 45°.
- Sheet bend keeps the hem a smooth curve instead of a jagged edge with points at each column. It moves the
  cloth without adding speed, but above about 0.6 the cape gets stiff and boxy.
- Soft contact eases the body's push in over a band (2 units) instead of snapping the cloth flat against a leg
  every step. Hard contact (0) is what made the cape shiver against the legs on every stride.
- Air drag sets how far it trails while walking. Stiffness hardly matters next to gravity on a cape this long; it
  matters more for short chains such as hair.

- Link curve (her model only, the first two swinging rows): the cloth between two columns is taken on the
  curve through the columns either side, as it wraps her shoulders and arms. On the straight line it lay most of
  a unit inside its own cloth, and an arm swinging back pushed the cape out into a hump behind her.

Collisions test each bone at its middle and its tip, and the cloth between neighbouring columns at the hem and
the middle of each panel, so an arm or a leg can't slip between bone tips. The colliders must sit clear of the
cape at rest, or the cape puffs out where they overlap; the stand-in's cape is shaped to clear them.

## Putting it on the Warden

1. In `blender/warden/warden.py`, add the bones above with the same layout (the stand-in's `makeCape` places them
   on the cloth's inner face), and weight the cape and its lining to them alike: two columns either side by angle,
   rigid in the middle of a bone and blended at its joints, as `makeCape` does.
2. In `src/world/main.js`, after the model loads: `hero.cape = attachCapeSprings(m)` and
   `hero.hair = attachHairSprings(m)` (after the cape, so the hair finds the cape's bones and rests on them).
   Each frame, after `poseModel(h)`: `m.updateMatrixWorld(true); hero.cape.update(dt); hero.hair.update(dt)`,
   the cape first. Call `reset()` on both after a teleport. Before them, `hero.headMotion = HeadMotion.attach(m)`
   (`src/world/head-motion.js`) and `hero.headMotion.update(dt)` give her head a nod, a sway and a turn on a
   spring over the neck (the walk holds it upright), which the hair then hangs from.
3. Her hair from the Face Creator is modelled lying a little inside the cape's collar band at the back. The
   bench lifts it out over the band when it re-skins her hair (`clearOfCape` in `src/cape-lab/model-body.js`);
   the lasting fix is the hairstyle in the Face Creator, or the band sitting closer to her neck.
