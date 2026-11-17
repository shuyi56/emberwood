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

## The cape rig

Each column of the cape is a chain `cape_<column>_0 … cape_<column>_<rows>` under `Chest`, plus an empty
`cape_<column>_tip` at the hem that gives the last bone its length. Row 0 is the band from the collar over the
shoulders: it rides the chest and is not simulated. The rows under it swing. The bench defaults to 5 columns and
5 swinging rows; 7 columns and 6 rows are on the page to compare.

## Captures and numbers

    node tools/cape-lab/capture.mjs --url http://127.0.0.1:5173/cape-lab.html --out /tmp/cape \
         [--scenarios turn,startstop,walk,run,spin] [--views back3q,side,back] [--every 3] [--colliders]
    node tools/cape-lab/check.mjs [--settings '{"damping":0.25}'] [--columns 7 --rows 6]

`capture.mjs` writes a GIF, a sheet of 8 frames and the frames themselves for each motion and view. `check.mjs`
runs the same simulation in node in half a second and prints the loop drift (how far the cape is from where it
was one walk cycle earlier; 0 means it repeats exactly), the hem's largest swing walking and turning, how long it
takes to settle after a stop, how far the hem lands at 30 and 144 fps from where it does at 60, and how close
any bone comes to the body's capsules.

What the settings do, from a sweep with `check.mjs`: damping is the main control (0.1 swings 30° and keeps
moving 2 s after a stop; 0.3 swings 14° and settles in 0.75 s). Air drag sets how far it trails while walking.
Stiffness hardly matters next to gravity on a cape this long; it matters more for short chains such as hair.

## Putting it on the Warden

1. In `blender/warden/warden.py`, add the bones above with the same layout (the stand-in's `makeCape` places them
   on the cloth's inner face), and weight the cape and its lining to them alike: two columns either side by angle,
   rigid in the middle of a bone and blended at its joints, as `makeCape` does.
2. In `src/world/main.js`, after the model loads: `hero.cape = attachCapeSprings(m)`. Each frame, after
   `poseModel(h)`: `m.updateMatrixWorld(true); hero.cape.update(dt)`. Call `hero.cape.reset()` after a teleport.
