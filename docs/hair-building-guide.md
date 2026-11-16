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
4. Compare against the reference photo side by side.
