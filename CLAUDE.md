# Emberwood: notes for sprite work

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

## Game characters: Blender for clothes, the Face Creator for faces

- In the game, a character's face and hair come only from the Face Creator. The Blender scripts
  (`blender/<character>/`) are for the body and the clothes. Change a face or a hairstyle in the
  creator, never in Blender.
- Keep the clothes that frame the hair (cape, robe, hood, collar) clearly apart from the hair's colour
  in hue, not only in value. Dark brown cloth next to dark brown hair made the Warden's hair disappear
  into her cape from the side and back. Keep the colours in harmony too: slate blue against brown hair
  clashed, so her robe and cape are forest green. Check the cape against the hair in the side and back views.
