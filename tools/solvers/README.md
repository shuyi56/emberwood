# Solvers

A solver finds a shape from forces, constraints or a cost: the fold a lapel takes, the drape of a sheet, a strand's
fall, a heel's height. Read [`../INDEX.md`](../INDEX.md) first for which ones your project needs. Every solver run is
followed by its contour inspection ([`../gauges/README.md`](../gauges/README.md#contours)) before any render.

**Every solver and render driver caches its stages.** A stage whose inputs did not change is read from disk, not
rebuilt: key each stage on its input files' hashes and on the code and constants it depends on (`folds.cache_key`,
`ponytail_flow`'s `OUT/../cache`, `photo_waves._cached` with `_code_key`), keep the stages apart so that a change late
in the chain (a cut, the locks, a camera) does not redo the early ones (the scalp field, the comb, the backdrops), give
the driver a `--quick` look for iteration, and write the seconds per stage into its report. A driver that starts from
zero on every run is a fault: the photo hair's first driver did, and every change cost 10 minutes (2026-11-28).

Each entry: **what it is for**, **in and out**, **run**, **trouble looks like**, **use it when**, **grounded in**.
Known limits and their fixes for the cloth solvers are kept in
[`docs/clothing-solvers/solver-issues.md`](../../docs/clothing-solvers/solver-issues.md); read it before calling a
flaw a solver fault.

---

## Folds

### Authored folds: `folds/folds.py`

**What it is for.** Baking a fold into a sheet of the local fabric solver to a measured angle: a lapel turned back
over its roll line, a collar turned down at its roll, a pressed hem or pleat. The fabric's own bending only ever pulls
a sheet back toward its cut shape, so it cannot hold a fold (a lapel ended 19 degrees short of target that way); here
the fold is its own constraint, a hinge on each inner edge holding the angle between its two triangles.

**In and out.** A `Fabric` sheet (`blender/shared/fabric.py`), its flat pattern `[n, 2]` and triangles, and creases
(a line on the pattern, an angle in radians, a radius: 0 for a pressed fold, the roll's radius for a rolled one, and
a side). Out: the folded points, and a report (`ok`, `achieved_deg`, `target_deg`, `steps`, and where each flap lands
against its draft, `land_along`, `land_off`, `land_worst`). `bake_cached` adds the fold cache: the same piece, creases,
settings and body load the solved points instead of solving.

**Run.**
```python
import sys; sys.path.insert(0, 'blender/shared')    # fabric.py; folds and contours through their shims
import fabric as xf, folds as fo
F = fo.Folds(fab, flat, T)
F.add(line, angle, radius=0.0, side=+1)
report = F.bake(body=None, h=1/240, tol_deg=1.0)   # writes the contour sheet when it converges
pos, report, state = fo.bake_cached('lapel', fab, flat, T, creases, body=body, extra=(body_points,))
```
Set `EMBER_CLOTH_CACHE` to keep the cache somewhere lasting (default: the temp folder).

**Trouble looks like.**
- `report.ok` false, "stopped short": it ran out of `most` steps before the angle reached tolerance. The result is not
  cached. A fold that geometry blocks will not move however many passes it gets (solver-issues G1: a rolled fold on a
  curved roll line needs more cloth at the outer edge, a draft change).
- The angle on target but `|land_along|` over one row: the turn took less cloth than its arc and the flap sits off its
  line (S2). Treat it as a failed fold.
- Strain jumping inside the roll once a floor or body holds the flap (worst edges 300 to 500 %): pass
  `body_band=False` (S1).
- A fold that closes past target on a curved front: `correct=True` meeting G1, not a hinge fault.

**Use it when.** Any pressed or rolled fold: lapels, collars, cuffs, hems, pleats, a boot tongue's turn
(`blender/boot/sew.py`). Do not use Blender's cloth for authored folds (project rule).

**Grounded in.** Bridson, Marino and Fedkiw 2003, "Simulation of clothing with folds and wrinkles" (the hinge's
dihedral angle and its gradient); Mueller et al. 2007, "Position based dynamics", and Macklin, Mueller and Chentanez
2016, XPBD (the hinge as a compliant constraint); the rest angles measured on the fold as a tailor makes it (the flat
pattern turned about the fold line through a cylinder of the fold's radius: a pressed fold has the cloth's thickness
as radius, a rolled fold spreads its turn over `r * angle`).

**Note.** Moved byte for byte. Its cache key hashes its own source, so do not edit the file just to tidy it: an edit
re-solves every cached fold.

---

## Cloth

### Local fabric solver, Python: `blender/shared/fabric.py` (stays until PR #85 lands)

**What it is for.** The game's particle cloth (`src/world/fabric-core.js`) for the Blender scripts: settle, sew and
fold sheets the way the game does. Project rule: simple pieces and authored folds use this, not Blender's cloth.

**In and out.** Points `[n, 3]` (z up, the caller's units; the figure is 63 units for 1.65 m) and settings
(`FABRIC_DEFAULTS`). `mesh(faces)` makes threads, shear and bend ties (rest lengths from the flat pattern with
`rest_pos=`); `pin`, `tie`, `tether`, `sew`, `flatten`, `hold_chain`. Bodies are callables `body(P) -> (pushed P,
touched)`: `MeshBody` (scipy) or `bvh_body` (Blender). Out: `f.pos`, and `settle()` returns the steps taken.

**Run.** See the module docstring's API block; `tools/solvers/tests/` has three small worked scenes.

**Trouble looks like.** Tears in body creases (the armpit): `MeshBody(planes=8)` (S3), but check the armhole cut
first. Strain frozen between step 500 and 2000 with the fastest point steady, next to a cloth collider's rim:
`MeshBody(tangent=1.5 * spacing)` (S5). NaN on the first step: pins set after `mesh()` (S4, fixed; pin first anyway).
Normals of a cloth collider must have one sign for the whole mesh.

**Use it when.** Cloaks, capes, sheets, petticoats, skirts' draft settles, any sewn piece before a final render.

**Grounded in.** XPBD (Macklin, Mueller and Chentanez 2016) in small fixed steps (Macklin et al. 2019); Coulomb
friction against the body; long-range attachments (Kim, Chentanez and Mueller 2012) for `tether`; follow-the-leader
chains (Mueller et al. 2012) for `hold_chain`.

### Local fabric solver, game: `src/world/fabric-core.js`, `fabric-stiff.js` (PR #83)

The runtime twin of `fabric.py`, used by the cape, the cloak's settle (`blender/cloak/fabric_settle.mjs`) and the
lace study. Adds per-point mass (`weigh`), air drag, hinges with authored creases and creep, and an implicit bend
solve (`stiffen`: a quadratic bending energy after Bergou et al. 2006, its stiffness from the cloth's bending length,
B = weight x g x c^3). On the fabric bench all five materials land within 5 % of their target bending length (silk
1.2 cm, linen 2.5, wool 4.0, leather 7.0, boot leather 14.0). Docs: `tools/fabric-bench/README.md` on that PR.

### Blender cloth drivers (stay, one per garment)

`blender/chemise/solver.py` (linen), `blender/petticoat/cloth.py` (starched cotton), and on PR #91
`blender/skirt/settle.py` (broadcloth). Each puts the drafted piece on the posed body, lets Blender's cloth settle it,
and carries the result back to the garment's rest shape through its bones. The figure is 63 units for 1.65 m and
Blender takes a unit for a metre, so each cloth's numbers are set at this scale on a swatch bench
(`blender/petticoat/swatch.py`; PR #91 `blender/chemise/swatch.py`, `blender/redingote/bench.py`: the cantilever's
bending length after Peirce 1930, and a gathered curtain's folds), never copied from a preset. Use Blender's cloth
only for final renders and structured pieces such as the chemise (project rule). These stay with their garments
because each is that garment's own settings and hand-off.

### Solver regression tests: `tests/`

`test_folds_band.py` (S1 and S2: the floor under a rolled fold), `test_crease_body.py` (S3: the armpit crease, planes
1 against 8), `test_sheet_edge.py` (S5: cloth above an open sheet's rim). Numpy and scipy, no Blender, seconds each:
```
python3 tools/solvers/tests/test_folds_band.py
```
Run all three after any change to `fabric.py` or `folds.py`; the numbers in `solver-issues.md` are their output.

---

## Footwear

### Boot last and pieces: `blender/boot/last.py` (stays: the boot is built from it)

The last a bootmaker works the leather over (level sections of the leg pulled taut, closing in only as fast as a stiff
hide can, never narrower at the ankle than the heel needs to pass) and the pieces cut from it: toe cap, vamp, two
quarters, two shaft panels, tongue, bindings, sole, each with its seam edges paired by letter (A to H, L for the
lacing). Every size is a share of the head's height H. Measure it with
[`leather_gauge.py`](../gauges/README.md#leather-gauges).

### Heel optimizer: `footwear/heel/optimize.py`

**What it is for.** The heel's height and shape found by gradient descent on the support gauge
(`gauges/footwear/heel/study.py`), not chosen by eye.

**In and out.** `DUMP.npz`, the feet a build saved (`HEEL_DUMP=PATH` with the heel figure). Varies the height (cm),
the tread's width, length and rake, the footbed's arch fill and heel cup. Out: `--out OUT.json` (the optimum and each
term) and `--plot OUT.png`.

**Run.**
```
HEEL_DUMP=/tmp/heel.npz python3 tools/render-review/smoke.py blender/base/views.py --figure heel --mannequin --head --out /tmp/hv
python3 tools/solvers/footwear/heel/optimize.py /tmp/heel.npz --out opt.json --plot opt.png [--steps 60] [--shapes]
```

**Trouble looks like.** A variable pinned at its bound (the cost wants past what the bounds allow: check the bound's
source before widening it); a term that dominates the sum (its weight is EXPERIMENTAL unless the term names a source).

**Use it when.** Any heeled shoe or boot, before the heel is modelled (`blender/heelboot/heelboot.py` takes its
numbers from it).

**Grounded in.** The gauge's physics and sources (Golding c. 1902, Cavanagh et al. 1987, Chen et al. 2020 and others,
listed under the heel gauge); the gradient by central differences (the gauge is a contact solve, not a formula) and
Adam steps within bounds. Which readings count, and how much (`WEIGHTS`), is EXPERIMENTAL unless a term names a source.

### Stylized heel: `footwear/heel/stylize.py`

**What it is for.** The optimizer's seat, height and footbed kept, and the heel's body given a flare and a waist
(Louis, French, Ferragamo-style), scored on the same gauge plus the neck's stress at heel strike.

**Run.** `python3 tools/solvers/footwear/heel/stylize.py DUMP.npz --out sty.json --plot sty.png [--base opt.json]`

**Trouble looks like.** A style whose neck stress exceeds the straight block's at the optimizer's height. That limit
is EXPERIMENTAL: a ratio, not a strength (no source gives a breaking stress for a stacked leather heel; a real
waisted heel is built round a stiffener or cut in wood).

**Grounded in.** Roark (elliptical section: area pi w l / 4, modulus pi w l^2 / 32); Perry and Burnfield 2010
(vertical peak about 110 % of body weight); Winter (fore-aft force peaks near 0.2 body weight); American Duchess 2024
and SATRA heel styles for the shapes.

### Lace tension solver: `blender/shared/lace_tension.py` (PR #76)

Finds a laced opening's proportions (gap, eyelet inset, row offset) from the forces: the cord's tension
T = N p / (2 cos theta), the capstan friction exp(mu * turn) at each eyelet, a plane-stress cell of leather per pitch,
and the cord bridging the ankle's hollow. Stored energy is least at a row spacing D = sqrt(2) p (frictionless). Gives
ratios to the eyelet pitch and to H, so they carry to any boot. Docs: `docs/clothing-solvers/lace-tension-solver.md`
on that PR.

---

## Laces and pleats

### Lace and pleating study: `tools/lace-study/` (PR #79)

Lace strips cut, folded (pressed or fluted), stitched, pressed and hung on the game's fabric solver, as a dressmaker
makes them; `calibrate.mjs` sets each lace's bending length with the cantilever test (Peirce 1930; ASTM D1388);
`study.mjs` solves, inspects contours (spread, ripple, waves, flags for too round, flat, kinked, stretched) and keeps
the fold cache. Docs: `docs/clothing-solvers/lace-and-pleating.md` on that PR.

---

## Hair

### Hair foundations: `tools/hair/` (PR #88)

The Ranger's hair rebuild, foundations 1 to 8 (read `/mnt/project-files/hair-rebuild/ranger-foundations.md`):
- `scalp_field.py`: where hair grows on the creator's head, its density (Leerunyakul and Suchonwanit 2020: 133 to
  163 hairs/cm2 by region), growth direction and exit angle. Every style is built on it.
- `groom.py`: strands on the scalp settled under their own weight by an inextensible Kirchhoff rod (bending energy
  EI/2 times curvature squared; Young's modulus 2.9 GPa, do Amaral 2022; density 1.312 g/cm3; diameter 80 um),
  checked against the cantilever sag w L^4 / (8 EI) (`--selftest`). A style is operations on the natural fall (part,
  comb, settle, cut), never hand-placed guides.
- `cards.py`: game cards fitted to clusters of the settled guides, with LODs.
- Always paired with the tracer, `creator_truth.py` (see gauges): the creator's own outline is the ground truth
  (John's hair rule).

### Wavy hair as wave locks: `tools/hair/wavy_flow.py`

**Purpose.** The Warden's long wavy hair (the creator's `wavyFull`) rebuilt with the ponytail's generation: fibres
grown from the approved groom on the head, then below z 0.3 each of about 120 wave locks solved as a Kirchhoff rod
with the creator's own wave as its rest shape, the hairs riding their lock (drawn in, then opening into sub-locks),
finished with the mid layer, the noising round, fairing, light through the hair, strand isolation and the 0.4 mm
scalp cap. Writes the Face Creator asset (`asset-wavyFull.json`), the fibres (`fibres.npz`) and every gauge
(`wavy.json`: wave height against the creator, rod report, kinks, light, hairline, join cover).

**Run.** `python3 tools/hair/wavy_flow.py HEAD.json.gz GROOM_DIR wavyFull.json.gz OUT --noise 1.0 --rest-gain 1.4`
(about 10 minutes, most of it the rods), then `apply-hair-fidelity.py ... wavyFull=OUT/asset-wavyFull.json` and
`node tools/hair/creator_shots.cjs PAGE OUT_PREFIX wavyFull Warden` for the five views.

**Trouble looks like.** `wave_groom_after_mm` off the creator's `wave_creator_mm` by more than 1.5 mm in a band (the
rest gain is calibrated: re-calibrate when the body, the cut or the lock count changes); rods `residual_mm` over 0.5;
`kinks.faired` over 20 deg; hairline bare cells; join cover under 1; `sides.hair` left and right front tips apart
by more than 0.03 HH where the creator's are level (front locks take the creator's strands of both sides, mirrored,
and are cut as well as grown); `locks.hairs_min` far under the median (a thin lock on its own rod dangles alone:
locks under a quarter of the median join a neighbour). The asset's `look` tints its highlight and glints toward
the hair's hue (white ones read ashy on a wavy fall), and a lock's far half has its "out" mirrored outward (lit
by its own lock normal it showed as a black inner layer). `curtain` (empty 4 mm cells round the body inside the
hair's span, by height, and the hem's step): high when the hair hangs as separate ropes. The locks are sections of
a brushed-out set, not ropes: each hair takes the smooth lines of its nearest locks (`--blend`, default 0.8) and
its own lock's wave.

**Grounded in.** The creator's wave (main.js `wavyMass`: 0.034 / 0.040 HH axes, wavelength 0.5 + 0.08 x section HH,
grown in from z 0.3 to -0.05) as ground truth; the settled wave under its own hanging weight A0 / (1 + h / (Lb^3 k^2))
(small-amplitude Kirchhoff rod), so the rest curvature is preloaded by that factor (gravity preloading, as in film
long-hair rigs); the last inch left straight (stylists); rod constants from Robbins ch. 9. CALIBRATED: the rest gain
(the cape and back flatten the wave's outward half, so the linear preload alone settles low). DRAFT: lock split 30 mm,
packing 0.1, the clump and opening amounts.

---

### Wavy hair from a reference photo: `tools/hair/photo_waves.py`, `tools/hair/photo_render.py` (PR #99)

**Purpose.** Long layered waves with a curtain fringe built from a photo, in the ponytail recipe's order: measure,
flow field, gauges, look, then hair. `build` combs every root over the scalp (gravity, a push off the part, the
sides directed diagonally back) and lifts it onto one outer surface (`ENV_MM`) on an arc that leaves the root at an
angle; `field2` hangs the hair below the head as a front fall and a back fall per side (rank kept, faired, one S wave
a side by height) and builds each side's bang as one smooth S sheet (`bang_sheet`); `lock_fall` groups the falls'
lines into locks (about 120 lines each) that gather, take their own share of the wave and open into 3 sub-locks;
`photo_render.py` draws the lines as fibres with `render.py` and saves the front close-up beside the photo.

**Run.** `python3 tools/hair/photo_render.py HEAD.json.gz OUT --lines 16000 --per-root 2 --size 750 --photo PHOTO`
(about 10 minutes). The field alone, drawn as lines: `field2` then `draw2` (driver in the project files,
`hair-rebuild/wavy-photo/field/field2_driver.py`; about 2 minutes at 8,000 lines).

**Trouble looks like.** `smooth_falls` or `smooth_fringe` not ok (`gauges/hair/flow_smooth.py`; it reads the course
before the S wave, so a waved line under 45 mm is the wave, not a fault); rank moves over 3% above about 2% of a
fall; `lines_turning_up` over 0; half widths off the photo's (0.45 / 0.54 / 0.63 at z 0.52 / 0.2 / -0.15). On the
head: a node turning over 20 degrees (the comb steps along the skin and then lifts: `_fair_head` pins the root only).
A step in height between sections means the lift is not one surface. A bare middle of the forehead: the bang's
inner end (`BANG_END`) is too far out.

**Grounded in.** Sectioning and direction: NY cosmetology study guide 8.5, Sam Villa, Red Seal 6.4. Lift as an angle
at the root: SideFX Houdini Lift Guides ("applies a rotation"), Behind the Chair ("overdirect each section to a
90-degree angle"), Yuksel et al. 2009 Hair Meshes (strands traced through an outer volume, C1 joined). Bend bounds: a
hair's bending length about 45 mm (Goldstein, Warren, Ball 2012), a fringe dried on a 35 mm brush (ghd). Locks: the
ponytail recipe's step 6. MEASURED off the photo: widths, the fringe's ends, the front fall's band. Overridden by John
(2026-11-28): the lift height (32 then 27 mm against the photo's 9), the fringe's depth (66 mm). DRAFT: lock size,
lock wave depth, copy offset.

---

## Accessories

### Chain drop: inside `blender/shared/accessories.py` (stays)

A necklace is a closed chain of a given length dropped over the head and let fall (a small Verlet simulation against
the body and the clothes built so far): it comes to rest on the trapezius at the back and falls into a U (a V with a
pendant) in front. The pendant hangs from the chain's lowest point and turns about its bail only as far as it must.

## Hair

### Endpoint flow: `hair/endpoint_flow.py` (PR #99)

**What it is for.** A hair line from a set start to a set end: the smoothest line between a root (a point and the way
it leaves the skin) and an end (a point and the way it arrives), kept off the head. It was written for the photo hair's
fringe, which had been a sheet solved apart from the rest of the flow and then a heading authored by height; with the
two ends set, nothing in between is authored.

**In and out.** `solve(P0, T0, P1, T1, floor, m0, m1, N)`: start and end points `[n, 3]`, unit directions there, how
far the line keeps each direction (`m0`, `m1`, lengths), and `floor(X)`, which returns points pushed out to the body
plus its clearance. Out: lines `[n, N, 3]` and a report (`passes`, `pushed`, `worst_inside`, `bend_radius_min`,
`bend_radius_p1`, `bend_radius_median`, in the caller's units). The line minimises the summed squared second
differences with two nodes held at each end; nodes inside the floor are held softly to the floor under them and the
line solved again, the targets sliding along the body from pass to pass.

**Run.** `photo_waves.fringe_flow(root, layer, surf, side)` sets the ends (`FRINGE_ENDS`, `FRINGE_END_DIR`, `FRINGE_M`)
and calls it; `field2` caches each side's result (`fringe+1`, `fringe-1`). Look at it with
`python3 tools/hair/overlays/head_flow.py HEAD CACHE OUT.png`.

**Trouble looks like.** `bend_radius_min` far under the median with a sampled (voxel) body: the hold is too hard and the
line is copying the body's bumps (a hard snap gave 0.03 mm); `worst_inside` larger than the clearance can spare; lines
crossing the middle: `m1` too large (2.2 x the chord closed the curtain; 1.0 is right).

**Use it when.** A styled piece has a known resting place: a fringe, a face-framing piece, a tucked lock. Not for hair
that hangs free: that is the falls' flow field.

**Grounded in.** The minimiser of the bending energy with both ends' points and directions fixed is the cubic Hermite
curve; contact is an active set on top of it. John 2026-11-29: "set the start and end points then set a solver to build
the smooth flow".

### Tail lock rods: `hair/tail_lock_rods.py` (PR #97)

**Purpose.** The locks of a ponytail in 3D: each lock is one Kirchhoff rod clamped at the tie and free at the tip,
with natural curl, bent by its weight and pushed off the other locks and the head. Locks part as they fall because
nothing below the tie holds them together.

**Inputs.** Per lock: root (mm), frame at the tie (d1, d2, tangent), length, hairs, radius; optionally a head
function (points, radii) -> (depth inside, outward normal). **Outputs.** Node positions, frames, arc lengths, radii
and a report.

**Run.** `import tail_lock_rods as TR; out = TR.solve(roots, frames, lengths, n_hairs, radius, head=head)`. The
creator asset drives it: `tools/hair/bangs_creator_asset.py ... --tail-locks` (hairs ride their lock's rod as a flat
ribbon that rolls). About 30 s for 33 locks of 75 segments.

**Trouble looks like.** `residual_mm` over 0.5 or `imbalance_rad_max` over 0.5 (not settled: raise ITERS);
`gap_under_half_share` over 0.01 (locks inside each other: raise CONTACT_K, which needs a smaller RELAX);
`rank_swaps_outer` above 0 (outer locks crossed round the axis); a plain fixed-point iteration (RELAX near 1) never
settles, since a 300 mm lock is far past its 45 mm bending length.

**When.** Any tail, braid end or loose hank below a tie. Not for hair lying on the head (that is the flow field).
Also the hang of loose long hair: `wavy_flow.py` (below) clamps one rod per wave lock where the hair leaves the head
and passes each lock its own natural wave through `solve(..., rest=f)`, where `f(s)` returns the rest curvature per
node, the signed turn rate and the phase per lock (the tail's fixed helix is the default).

**Grounded in.** Kirchhoff rod statics; Bertails et al. 2006, Super-Helices (hair as an inextensible rod with
natural curvature under gravity); Bergou et al. 2008, Discrete Elastic Rods (the Bishop frame). E 3.9 GPa, density
1.3 g/cm3, 70 um (Robbins ch. 9). DRAFT: GJ/EI 0.7, the curl (radius 12 mm, pitch 100 mm, off the reference photo),
contact stiffness, lock packing 0.1.
