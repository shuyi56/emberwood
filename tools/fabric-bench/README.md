# Fabric bench

Fixed sheets on fixed bodies, so the local fabric solver (`src/world/fabric-core.js`, `src/world/fabric-stiff.js`) can
be timed and measured the same way every run, and the old solver (`fabric-core-v1.js`, the cape's first core, kept
here only for the comparison) set against the new one on the same cloth.

    node tools/fabric-bench/bench.mjs                       # every scene, old solver and new at draft and final
    node tools/fabric-bench/bench.mjs --cantilever          # each material's bending length against its target
    node tools/fabric-bench/bench.mjs --out /tmp/fb         # also the frames, for the clay views
    python3 blender/fabric-bench/views.py --frames /tmp/fb --out /tmp/fbv      # clay, Cycles 8 samples, undenoised
    python3 tools/fabric-bench/blender_cloth_timing.py      # Blender's cloth on the same sheets, timed

Scenes (`scenes.mjs`, in the figures' units, 39.4 a metre): `hang` (1 m sheet held at two corners), `drape` (the
same sheet dropped on a ball), `fold` (a wool panel with a pressed crease), `boot` (a boot-leather shaft round the
base figure's shin through her run) and `tasset` (a belted leather panel over her hips and thighs through her run).
The run is `public/cape-lab/walk-run.json`, played in her own space with the air coming at her at 3.1 m/s.

The solvers: `v1` (old core: bend ties two along, the body after the constraints, no weight, no air), `draft`
(new core, 240 steps a second, 2 walks, 1 bend solve a step) and `final` (240, 4 walks, 2 bend solves).

## What changed in the solver

- **Stiffness from what the cloth is.** `stiffen()` solves the bend as one implicit solve over the whole sheet each
  step (a quadratic bending energy, Bergou et al. 2006, with its rest curvature turned with the cloth, as Projective
  Dynamics does; banded Cholesky, factored once). Its stiffness is the cloth's flexural rigidity from its bending
  length, B = weight x g x c^3, so `MATERIALS` lists what a cantilever test measures, not tuned numbers. The old bend
  ties and hinges could not pass about 2 cm of bending length (linen) however stiff they were set: the constraint
  walks carry a bend one row at a time. Now all five materials land within 5% of their target:

  | material | target | measured |
  | --- | --- | --- |
  | silk | 1.2 cm | 1.1 |
  | linen | 2.5 | 2.3 |
  | wool | 4.0 | 3.8 |
  | leather (1 mm) | 7.0 | 6.7 |
  | boot leather (2.5 mm) | 14.0 | 13.3 |

- **Weight.** `weigh(density)` gives each point the mass of the cloth round it; heavy points move light ones more.
  The air's drag (`air`, `wind`) acts on each point's area across the flow and is divided by that mass, so linen
  floats down at about 1.6 m/s (as a flat sheet of it does) and leather of the same size barely lifts.
- **Creases.** Hinges at the triangles' edges with a rest angle: `crease(i, j, angle)` and `fold(line, angle)` press
  one in (the `fold` scene holds 62 of the 70 degrees asked). Leather takes a set: where it stays bent tighter than
  its `setRadius`, its rest curvature creeps toward the bend (`creep`), and a crease stays.
- **Contact.** The body can be tested after every walk (`contact: 'iteration'`), and at the middle of every edge
  (`edgeContact`), so a limb cannot pass between two points. The body gets the point's last position too
  (`body(p, i, ref)`), so a point pushed more than halfway in goes back out the side it came from.
- **Tethers** (`tether()`): each point may not go further from its pin than its distance along the cloth.
- `advance(dt, o)` (fixed steps with a callback to move pins and body to each step's time), `snapshot()` /
  `restore()` (keep a drape, the sets included), `mesh(tris)` for any triangle mesh (pattern pieces).

## Numbers (2026-11-19, one core of the cloud machine)

| scene | solver | ms a frame | longest thread stretch | mean | jitter | in body |
| --- | --- | --- | --- | --- | --- | --- |
| hang | v1 | 0.8 | 109% | 1.5% | 0.010 | 0 |
| hang | final | 5.6 | 12% | 0.8% | 0.002 | 0 |
| hang | Blender cloth, quality 5 | 14.5 | 23% | | | |
| drape | v1 | 1.0 | 23% | 3.0% | 0.010 | 0 |
| drape | final | 6.8 | 17% | 1.0% | 0.009 | 0 |
| drape | Blender cloth, quality 5 | 9.8 | 26% | | | |
| tasset (leather, run) | v1 | 0.5 | 400% | 9.0% | 0.158 | 0.07 |
| tasset | final | 3.0 | 49% | 2.1% | 0.365 | 0 |
| boot (leather, run) | v1 | 0.8 | 779% | 38% | 0.744 | 0.89 |
| boot | final | 4.6 | 201% | 5.1% | 0.570 | 0.03 |

The full list, with draft, is `bench-numbers.jsonl` in the project's `fabric-bench/` folder. Blender's cloth is
multithreaded C on 4 cores; the bench is one thread of JavaScript.

## Known limits

- The boot shaft stays on the leg through the run at final quality only, and it rides down the shin over the
  strides; over 20 strides it comes loose. Draft lets it fall. The tasset is sound at both.
- `stretchK` (holding the edges inside the bend's solve) is on for boot leather only: on open pieces it made the
  linen shrink-wrap the ball and the leather panel bunch up at the belt. Why is still open.
- Self-collision (`thickness`) works but made the drape worse; it is off in every scene.
