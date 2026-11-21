# Compression

## Spare cloth rolls up

A garment drafted longer or wider than the room left for it has nowhere to go. Above a held line the cloth is
pinned to the body; below it the extra cloth bunches into a roll. The shirt under the gambeson stood 0.34 off the
skin where it was held and 0.91 just below it, and the difference showed as a roll pushing up at the centre front.

Fix: draft the cloth to the room it is pressed into (the gap between the body and the outer layer), so no spare
length is left to roll. Do not ask the solver to absorb it.

## Plumb hang crushes cloth

Drafting a hem to hang plumb from the widest point pushes cloth past the body, then the body pushes it back.
Threads were crushed to 0.09 of their length and folded back 175 degrees, which reads as paper.
Cut the cloth to the body instead (`plumb=0` in `drape.RATIOS`). The crush figure rose to 0.28.

## Compressed cloth needs more thickness

When cloth is squeezed between layers, a thin sheet creases hard. Making the linen thicker (`thick=0.024`) and
adding bending stiffness settings (`bending=0.6`, `self_gap=0.045`) kept the hem soft.

## Layers clear each other, thickness included

A cloth settle collides with a thin sheet. The garment's thickness is added after, so a layer can look clear and
still sit inside the outer garment. Check with `blender/base/spots.py`.

## Sleeves

- Sleeve room comes in gradually from the seam. A full-width cap at the seam stands up as a puff.
- Next to the body limit the ring width to half the gap between arm and torso.
- Pulling the cap toward the flat lengths over the whole width crumples the armpit. Do it on the outer columns only.
- A shirt sleeve ending at the gambeson cuff, held whole, avoids cuff shards.
