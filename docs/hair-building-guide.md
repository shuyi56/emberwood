# Building hair in the face creator

These are the techniques that worked on the Rogue ponytail and its full fringe, written as the order to build a
hairstyle in. Each layer sits on the one before it. The code lives in the face creator's `main.js`: the cap is
`buildHair()`, the shared lock helpers are `hairClump()` and `hairVeil()`, and the worked examples are
`HAIR_STYLES.ponytail.extras`, `HAIR_STYLES.ponytailFull` and `HAIR_STYLES.ranger.extras`.

## Rule zero: research before guessing

Research the technique (reference photos of the real cut, hair-card and groom tutorials, stylist guides) before you
change the model whenever either of these is true:

- You are not 100% certain how the real thing is built or how to model it.
- The same feature has been rejected more than once. Two rejections in a row means the approach is wrong, not the
  numbers. Stop tuning, research how the effect is made, and come back with a different construction.

Write down what the reference shows (shape, direction, thickness, where it is thin) before writing code. On the
Rogue fringe, tuning the cap's edge line went through seven rejections; the fix came only after researching how
shaped bangs and hair cards are built, which led to real locks.

## Layer 1: the cap (the head of hair)

1. Build the shell from the scalp outward: every point is the scalp's own point at that height and angle, lifted by
   the style's depth. Below the widest part of the skull it hangs straight down instead of following the head back in.
2. Give it a depth profile: thin at the hairline and the ends, full over the crown and the back.
3. Keep it clear of the ears with a wide, soft swell, never a ledge.
4. End it with the strand cut-out (alpha-tested strand texture) so the edge is hair ends, not a hard rim.
5. When locks will form the fringe, raise the cap's edge a little at the front so the locks, not the cap, make the
   outline. A cap edge alone always reads as a bowl or a clean cut.

## Layer 2: guide curves for each lock

1. Each lock follows a guide curve through 5 to 7 knots, each knot an angle round the head, an elevation and a lift
   off the scalp: `onScalp(rgDir(th, el), lift)`, smoothed with `catmull`.
2. Shape it like a blow-dried lock: rooted back on the crown, lifting off the forehead in a C (root lift, a bulge over
   the brow bone), with the tip turning back in to touch the skin.
3. Give the set a comb direction. All locks drift the same way as they fall (for a side-swept fringe, add a term that
   grows with `pow(along, 1.3)`), and hooked locks curve over their last third.
4. Vary the lock lengths along a designed outline (an arch, an inverted V for curtain bangs, longer toward the
   temples) and add a small jitter per lock so the outline is soft but not ragged.
5. Give each lock its own lift so overlapping locks never cut through each other.

## Layer 3: clumps of fine strands (`hairClump`)

1. Never draw a lock as one thick tube. Draw it as a clump of 6 to 11 fine sub-strands along the guide.
2. The strands sit together at the root and part toward the end (the spread grows with the distance along the lock).
3. Each strand stops at its own length (staggered by 15 to 25 percent), the outer ones more than the inner ones.
   That stagger is what makes the ends wispy.
4. Each strand gets a slightly different tone (plus or minus 5 percent) so the clump has texture.
5. Strands are flat ribbons (depth about 0.15 of their width), not round tubes.
6. When the fringe must read as one mass, spread the strands of each clump across the whole gap to the next clump.
   Clumps that stay narrow read as separate thick ropes with skin between them.

## Layer 4: the sheer layer under each clump (`hairVeil`)

1. Lay a soft ribbon of hair colour under every clump: alpha highest down the middle (`1 - x²`), fading in at the
   root and out before the strands end.
2. Make it warmer than the hair (red up, blue down). Over skin a sheer brown otherwise mixes to grey.
3. Keep the canvas alpha when blending (`blendSrcAlpha` Zero, `blendDstAlpha` One), or the page shows through pale.
4. Size it to the gaps it must fill: about 1.3 times the clump width when clumps are far apart, up to about 3 times
   for a closed fringe. A veil much wider than the hair it supports turns the skin beside it grey.
5. No shadows from the veil, no depth writes, drawn before the strands.

## Layer 5: root and tip fades

1. Root: fade each strand in over its first stretch (`rootLen` about 0.2) so it grows out of the cap instead of
   sitting on it.
2. Tip: fade the last stretch out (`tipLen` 0.15 to 0.3) through the strand cut-out, so ends are fine points.
3. Turn shadows off for fine fringe strands (`noShadow`). Their shadow on the forehead reads as a pale ghost lock.

## Layer 6: coverage and overlap

1. Build the fringe in two staggered layers (the second offset by half a step and a little lower), so every gap in
   one layer is covered by a lock of the other.
2. Check that no skin shows between locks unless the style wants it (an open parting, a wispy centre).
3. Thin areas come from fewer, finer strands, never from lowering density with the alpha test: that cuts strands into
   dashes.

## Layer 7: face-framing locks

1. Add long locks down the front edge of the side hair, from the temple to the jaw or past it, built the same way:
   guide curve, clump, veil, root fade.
2. For a full frame use several (4 to 5) per side at slightly different angles and lengths, not one.

## Layer 8: check it

1. Render front, 3/4, a close-up of the area you changed and the profile, every time.
2. For a debug pass, colour each layer flat (one colour for clumps, another for hooked locks) to see which layer
   causes a problem before you change anything.
3. Rebuild the test page from the built file before every render. A stale page shows no change and wastes a round.
4. Syntax-check the source (`node --check main.js`) and fail the render on any page error. A broken page leaves the
   previous images on disk, and they look like a real result.
5. To round the top of a curtain parting, shape the cap's edge as a broad arch there (a wide bump, not a narrow
   peak). Moving the inner locks outward early does not do it: they then hang straight down and the opening turns
   into a box.
6. Volume comes from lift: raise the inner locks' C (more lift at the root and over the brow, tip still turning in).
7. Compare against the reference photo side by side.

## Wavy hair

Researched for the Warden's long waves (`HAIR_STYLES.wavy.extras`), from how wavy hair is painted (the ribbon
method: Gurney's "hair ribbon secret", value-mapping guides) and how it is built for games (hair cards in layers,
stylized sculpts built lock by lock).

1. A wavy lock is a ribbon, not a wiggly line. Move the guide from side to side in an S and, a quarter turn behind
   it, out and in from the head, so the lock winds down. Turn the ribbon's face toward the side it is travelling to.
   A side-to-side wiggle alone is flat and reads as crimped straight hair.
2. No wave over the skull. The hair's weight pulls the roots straight, so the wave comes in below the ears, grows
   down the length, and the waves sit a little closer together toward the ends.
3. Few large waves. About 0.6 of a head per wave at this scale. A wave of 0.45 or less with locks this narrow reads
   as curls or a plait.
4. Neighbouring locks are a little out of step (about a tenth of a wave from one to the next, plus a small random
   part), so the waves travel across the hair as one body and the locks still cross and part. Locks fully out of
   step weave into a braid pattern; locks in step ripple as one corrugated sheet.
5. Layers: the cap is the opaque base. Over it a darker under layer of broad locks (tone about 0.75), then a top
   layer of narrower locks sitting in the under layer's gaps and out of step with it. The dark layer showing between
   the top locks is what gives the waves depth.
6. Light and shade run across the lock, not along it. Lighten each strand over a crest (where the ribbon faces out),
   darken it in the trough and where it turns under, and darken the troughs more than you lighten the crests
   (`WAVY.bandUp`, `WAVY.bandDown`). The highlight then breaks into a patch on every wave, which is what reads as
   wavy from a distance.
7. A wavy lock gathers to a point. Keep the strands together (spread about 0.12, against 0.6 for a straight fringe),
   narrow the clump over its last third and stagger the strand lengths, so each lock ends in a soft point and the
   outline is a row of points at different heights.
8. End the cap above the locks' ends so the outline is made by the locks.
9. Over hair, make the sheer layer under a clump darker than the hair (tone about 0.5) and no wider than the clump.
   The warm, wide veil used over skin shows as pale patches and a halo round long hair.
10. Ride every lock on the cap's own surface: find the cap's radius down the lock's column the way `buildHair` does
    (scalp, ear clearance, depth, the hang rule) and add the lock's lift. Locks placed by their own estimate cut
    through the cap.

## Ponytail: a hierarchy of locks, not one mass

A real tail is gathered hair, so build it the way it is gathered (hair-card and groom breakdowns build tails in
layered clumps, with longer cards defining volume and break-up cards on top).

1. **Core.** A slim, darker tube down the middle, about half the tail's width. It is the shade inside the tail, seen
   between the locks, and it stops the tail reading as see-through.
2. **Primary locks.** About seven round the tail. They leave the tie together, bloom out just past it (the hair pushed
   out by the tie), fall gathered through the middle, and part again toward the end. Each twists a little round the
   tail and ends at its own length (86 to 100 percent).
3. **Secondary locks.** Each primary is three thinner locks that lie together near the tie and split apart over the
   lower half, tapering to points at staggered lengths.
4. **Ends.** Cut the last stretch of every secondary lock into strands (a longer tip fade, about a third of a head), so
   the end is many fine points, not one blunt tip.
5. **Tone.** Each primary lock gets its own tone, and each secondary a small variation on it. The core is darker.
6. **Check** from the side, the back 3/4 and the front 3/4, with the whole tail and tie in frame.
