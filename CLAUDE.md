# Emberwood: notes for sprite work

## Research before you guess

Research a modelling or painting technique (reference photos, tutorials, how the real thing is made) before
changing the model whenever you are not 100% certain how to build it, or when the same feature has been rejected
more than once. Two rejections in a row means the approach is wrong, not the numbers: stop tuning, research, and come
back with a different construction. Write down what the reference shows before writing code.

Sprites are painted in code in `src/shared/sprites.js` (figure, outfits, specs) and
`src/shared/presets/` (female body and head, hairstyles). Check every change visually before
committing. Render all 8 facings and a few walk frames (columns 2, 4 and 6), and zoom into the
area you changed. One good front view proves nothing: most problems show up in the 3/4 and
profile rows.

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

## Face checks: no renders

Check a face with `python blender/facecheck/facecheck.py <figure> --before origin/main`, not with a render. It
runs the figure's own face code without Blender (about 17 s), draws front, 3/4, profile, eyes, lips and the
game-scale view, and measures the eyes, features, lip paint and symmetry. See `blender/facecheck/README.md`.
Ask the render queue only for what it cannot show: hair, cloth, final lighting and the sprite palette.

## Hair: build it in layers

The full step-by-step guide, with the reasons, is `docs/hair-building-guide.md` (worked examples: the face creator's
`HAIR_STYLES.ponytail`, `ponytailFull` and `ranger`). In order:

1. **Cap.** The shell from the scalp outward with a depth profile, hanging straight below the skull, clear of the ears,
   ending in the strand cut-out. Raise its front edge a little when locks make the fringe, so the locks set the outline.
2. **Guide curves.** One per lock, rooted on the crown, lifting off the forehead in a C and turning back in at the tip.
   All locks comb the same way; lengths follow a designed outline (arch, inverted V) with a small jitter.
3. **Clumps of fine strands** (`hairClump`). 6 to 11 flat sub-strands per lock, together at the root, parting toward the
   end, stopping at staggered lengths, each a slightly different tone. Never one thick tube per lock.
4. **Sheer layer under each clump** (`hairVeil`). A soft, warm-tinted ribbon of hair colour, fading in at the root and out
   before the strands end, sized to the gap it fills (1.3x the clump for open sets, up to 3x for a closed fringe).
5. **Root and tip fades.** Fade strands in from the cap and out at the tips; no shadows from fine fringe strands.
6. **Coverage.** Two staggered layers of locks so no skin shows between them. Make thin areas with fewer, finer
   strands, never with an alpha-test density cut (it breaks strands into dashes).
7. **Face-framing locks.** Several long clumps per side down the front edge of the side hair, built the same way.
8. **Check.** Front, 3/4, close-up and profile every time; colour each layer flat in a debug pass to find which one
   causes a problem.
9. **Hair falls.** Every lock is one piece from its root (on the part line for a parting) over the head and down to its
   end; ends are staggered and tapered, come back to the head, and every lock rides the cap of the style it is in.
10. **Check before rendering.** Run the face creator's `hairlint.js` first. It finds buried locks, stiff flaps, ledges,
    roots on bare skin and kinks in seconds, and names the lock.
