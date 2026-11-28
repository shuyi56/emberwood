# Emberwood: notes for sprite work

Sprites are painted in code in `src/shared/sprites.js` (figure, outfits, specs) and
`src/shared/presets/` (female body and head, hairstyles). Check every change visually before
committing. Render all 8 facings and a few walk frames (columns 2, 4 and 6), and zoom into the
area you changed. One good front view proves nothing: most problems show up in the 3/4 and
profile rows.

## Start every project and thread with the solvers and gauges index

Before any build, read `tools/INDEX.md`. Find your kind of project in its decision guide, take the solvers and gauges
it names, and read their entries (`tools/solvers/README.md`, `tools/gauges/README.md`) before you start. New solvers
and gauges go in `tools/solvers/<domain>/` or `tools/gauges/<domain>/`, documented in the same shape, with a row in
the index. Edit the real files under `tools/`; the files left at old paths are shims.

Every new solver or render driver caches its stages on disk from its first version (the rule and the helpers to copy
are at the top of `tools/solvers/README.md`): a change to one stage must not rebuild the stages before it.

## Shoulders: how to blend them into the arms

The neck-to-arm line has to read as one continuous curve. The usual ways it breaks are a square
corner (it looks like shoulder pads), a step, or a notch. Rules:

- **Never make the sleeve top narrower than whatever lies over the shoulder.** If a collar, cowl,
  mantle or pauldron ends at the shoulder and the sleeve starts narrower or lower underneath it,
  the silhouette gets a step. Start the sleeve at the shoulder point, at full width or wider.
- **Give the sleeve a rounded cap.** Add an ellipse at the top of the sleeve (about half the
  sleeve width plus 0.5 across, and 2.5–3 high) in the sleeve's own ramp. The cloth then falls
  over the top of the arm instead of being hinged onto the torso. See the `bell` branch in
  `drawFigure`.
- **A layer over the shoulder should lie on the cap, not stop at its corner.** Drop that layer's
  hem lower at the sides (the cleric cowl uses `- 1.0 * sin(phi)^4`), and keep its radius at the
  sides no wider than `B.sh`. A radius past `B.sh` makes a ledge that sticks out beyond the arm.
- **Fade hard edge lines out toward the silhouette.** A dark lip or trim line that runs all the
  way to the outline leaves a dark tick on the shoulder. Draw it only where the surface faces the
  camera (depth roughly > 2) and lighten it as it turns away.
- **Keep anything that should sit under a layer starting below it.** For example, the scapular
  starts under the cowl (h 36.4, not 38.6). Otherwise its corner pokes out at the far shoulder in
  the 3/4 views.
- **Use cast shadows, not outlines, to separate white-on-white layers.** Darken one or two rows
  below the upper layer with `P.darken(x, y, k, 0)`. The 4th argument `0` keeps the shadow
  neutral grey. The default leans blue, which is wrong on black-and-white cloth.

## Other lessons

- Snap animated offsets to the pixel grid (see `View`). Fractional bob or sway misaligns grid-snapped
  features and warps faces on single frames.
- Sleeve ends and hems should end in soft curves. A polygon that comes to a point reads as sharp
  and triangular.
- The inside of a wide sleeve should be a dark shape just wider than the wrist (`sleeveLining`).
  Add a cast shadow across the back of the hand. A wide grey ellipse leaves wedges beside the hand.
- Drapery folds read best as a few deep, slightly off-centre U-swags. Several shallow bands stack
  into stripes and look layered.
- At this resolution a hair feature is about 10 pixels across. Use a few clear shapes, each with
  one lit band and one dark crease. Many fine dots turn into noise.
- A piece added to the head (bun, ponytail) needs a joining shape where it meets the skull, so it
  grows out of the head instead of sitting on it.

## Game characters: Blender for clothes, the Face Creator for faces

- In the game, a character's face and hair come only from the Face Creator. The Blender scripts
  (`blender/<character>/`) are for the body and the clothes. Change a face or a hairstyle in the
  creator, never in Blender.
- Keep the clothes that frame the hair (cape, robe, hood, collar) clearly apart from the hair's colour
  in hue, not only in value. Dark brown cloth next to dark brown hair made the Warden's hair disappear
  into her cape from the side and back. Keep the colours in harmony too: slate blue against brown hair
  clashed, and forest green read badly (and vanished on the grass), so her robe and cape are a deep plum. Check the cape against the hair in the side and back views.

## Hair grooms: the Face Creator's lines are the ground truth

Build and check every groom (`tools/hair/`) against the creator's own hairstyle, not against a look
or a measurement alone.

1. Export the creator's style and head: `node tools/render-queue/rq_hair_export.cjs face-creator.html DIR <style>`
   (working copy in `hair-stress/code`). It writes `DIR/<style>.json.gz` (the cap grid and the strand clumps) and
   `DIR/head.json.gz`.
2. Groom with the creator traced: `python3 tools/hair/groom.py HEAD OUT --trace DIR/<style>.json.gz`. Styled hair
   (bangs, face-framing locks) is combed along the creator's strands and ends where they end. It is not solved:
   a lone hair holds a shape over only 4.5 cm, so a solve lets styled hair fall straight. Add
   `--reuse OLD/groom.npz` so that only the guides that changed are solved (seconds, not 20 minutes).
3. Check against the creator after every change: `python3 tools/hair/creator_truth.py HEAD DIR/<style>.json.gz OUT
   CHECK`. It draws the creator's outline (red) and strands (blue) over the groom from five views, marks magenta
   where the creator has hair and the groom does not, and cyan where the groom adds hair. The numbers per view are
   in truth.json. Fix the largest magenta area first: a hole in the creator's coverage is the worst fault.
4. Where a cited measurement and the creator disagree, follow the creator and note the measurement it overrides.
   Example: the bang section's depth is 60 mm from the creator, not the guides' 32 mm.
5. The hairline has no gaps: from ear to ear, no bare scalp behind it. `groom.py` gauges every groom it writes
   (`tools/hair/hairline_gap.py HEAD GROOM --map OUT.png`) and fails the build on a bare cell. Fix the hair, not the
   threshold.
6. Then render (`render.py`, `--size 750` keeps it inside memory), bake the cards (`cards.py`), and log each
   rejected look with `rq_reject.py`.

## Boots: lessons from the zip ankle boot

See `docs/clothing-solvers/zip-ankle-boot-retrospective.md` for the numbers.

- Measure with `tools/gauges/footwear/leather_gauge.py` (the old `blender/zipboot/gauge.py` still runs it) before and after every change to a boot's shape, weights or pose. Pose
  asymmetry over 0.02 units, or stretch outside 0.9 to 1.15, means the rig is distorting the leather.
- Make a stance's turn-out in the hip (`female.TURNOUT_AT_HIP`), never at the ankle: at the ankle it twists any shaft.
- On a heel the toes pitch: move the leather's bend back to the ball (`zipboot.bend_at_ball`).
- Render with `--rest` before blaming the shape: a line that is only there posed is the rig.
- Fit clearance by region (toes, heel, shaft), and when one region tightens, check the next one.
