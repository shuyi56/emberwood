# Accessories: necklaces, chokers, earrings, brooches

`accessories.py` adds jewellery to any figure built on the base female body. Accessories count as clothes, so they
are made in Blender (faces and hair still come only from the Face Creator). `blender/accessory_demo/` is a plain
mannequin that wears two sample sets, for checking changes:

    python3 blender/base/views.py --figure accessory_demo --mannequin --head --angles front --out DIR
    python3 blender/base/views.py --figure accessory_demo --mannequin --set "LOOK='B'" --head --angles front --out DIR
    python3 blender/base/views.py --figure accessory_demo --mannequin --list     # build + clearance report only

## How real jewellery sits, and what the code does about it

**A necklace is a closed chain that hangs from the base of the neck.** At the back it is caught at the nape, on the
knob of the seventh vertebra and the ridge of the trapezius. The base of the neck is a ring tilted down at the front
(the neck leans forward), and below the collarbones the chest is too steep to hold the chain, so the front falls
lower than the back, into a U. A pendant's weight draws that U into a V. Length decides where the front lands:
a collar length goes round the base of the neck, a princess length lies on the collarbones, a matinee on the upper
chest, an opera down over the bust. A chain bridges hollows instead of following them, and it lies over whatever is
worn under it.

`necklace()` does this by simulation. A chain of the given length is dropped over the head and let fall (Verlet:
inextensible but free to go slack, friction where it touches, a little stiffness so slack links don't stay kinked).
It collides with every surface already built, so it lies on the outermost layer. The base body's torso is a stack of
level sections with no trapezius ridge, so a freely falling chain settled almost level. The code therefore holds the
chain's back point at the nape (`nape=`, h 51.6) and lets the rest fall from there. `LENGTHS` gives the usual lengths
as measured on this body:

| name     | length | front of the chain comes to rest at |
|----------|--------|-------------------------------------|
| collar   | 15.5   | h 51.6, level round the base of the neck |
| princess | 19     | h 49.5, on the collarbones |
| matinee  | 25     | h 48.2, the upper chest, held by the rise of the bust |
| opera    | 30     | h 43.8, slipped over the bust |

The neck base is about 14 units round, so one unit is roughly 2.5 cm of real chain.

**A pendant hangs plumb from the chain's lowest point** and swings forward only as far as it must to rest on the
chest. `_rest_on` turns it about its bail until its body (not the bail, which stands where the chain does) clears
the surface.

**A choker is fitted, not draped.** It is a band at a fixed height, snug on the outermost surface there (rays cast in
toward the neck), with a closed cross-section so it has no caps. A charm hangs from its lower edge on a jump ring.

**Earrings hang from the lobes** (`ear_lobes()` reads them off the base head's ear in `blender/shared/ear.py`; pass
`lobe=` for a head whose ear sits elsewhere, e.g. the Face Creator's if it differs). A hoop lies in the plane of the
head's side, the way it passes front to back through the lobe. Each earring is nudged out until it clears the head.

**A brooch is pinned to the outermost surface** at a point on the front, turned to lie flat on it.

## Following the body through the walk

This is the part that matters for clipping. Every piece is parented to the `Body` empty, so `skin_trunk` skins it
with the trunk's weights. That alone is not enough: the meshes under it blend between their own vertices (the neck
is three rings two units apart), so where the bones turn against each other (the head against the leaning chest)
the two drift apart, and a choker sank 0.2 into the back of the neck mid-stride. `follow()`, called in
`finish_figure`'s `after_skin`, gives every accessory point the bone weights of the nearest surface point under it
(interpolated across that face, as Blender's Data Transfer does, using the nearest face that faces the point so a
hidden cap inside the neck is never picked). Rigid pieces (pendant, charm, brooch) take one set of weights read
where they hang from, so they never bend. Earrings get the `Neck` bone outright, which follows the head.

Nothing is parented under the `Head` empty: `tools/creator-head/merge-head.mjs` drops everything there for the Face
Creator's head, and `views.py --no-body` hides it.

`clearance_report()` checks each sheet column (the idle and the walk): how deep any accessory point sinks into the
clothes and body, as posed and skinned. On the demo both sets stay within 0.003 in every column.

## Using it on a character

In `build()`, after the clothes the piece lies on and before `finish_figure`. Build a cape after the necklace, so the
cape's own fitting sees it, or leave it out of the necklace's surfaces if the necklace should lie over the cape.

    import accessories as acc
    acc.necklace('Pendant necklace', body, length=acc.LENGTHS['princess'], pendant='drop')
    acc.earrings('Earring', body, style='hoop')
    ...
    finish_figure(lambda: (cape_on_arms([...]), acc.follow()))
    acc.clearance_report()

Materials use ids 80-85 (`jewel gold`, `jewel silver`, `pearl`, `garnet`, `velvet ribbon`, `cameo`), each with a
close-up look for the 3D views and a palette ramp for the sprites. At sprite size a chain is about a pixel wide; the
pieces are mostly for the 3D world model and close-ups.

## Pitfalls found on the way

- Nearest-face tests lie at the open edge of a garment's thickness: the nearest face there can be its inner side,
  facing in, with the point outside. `_signed` falls back to a ray-parity test whenever the face says "inside".
- A chain wedged between the skin and a neckline's edge is carried on out over the outer layer, not left in the gap.
- A thick garment lip catches a chain whose resting curve happens to match the neckline; it then reads as a trim.
  Keep clothes under jewellery close-fitting where the chain should pass over them.
- Pushing points out of surfaces stretches the chain a little each step; `drape` reruns with the links' length
  corrected until the resting chain is within 1.5% of the asked length.
