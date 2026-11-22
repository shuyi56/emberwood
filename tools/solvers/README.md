# Solvers

A solver finds a shape from forces, constraints or a cost: the fold a lapel takes, the drape of a sheet, a strand's
fall, a heel's height. Read [`../INDEX.md`](../INDEX.md) first for which ones your project needs. Every solver run is
followed by its contour inspection ([`../gauges/README.md`](../gauges/README.md#contours)) before any render.

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

---

## Accessories

### Chain drop: inside `blender/shared/accessories.py` (stays)

A necklace is a closed chain of a given length dropped over the head and let fall (a small Verlet simulation against
the body and the clothes built so far): it comes to rest on the trapezius at the back and falls into a U (a V with a
pendant) in front. The pendant hangs from the chain's lowest point and turns about its bail only as far as it must.
