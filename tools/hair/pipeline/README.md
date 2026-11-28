# The hair pipeline: a hairstyle is a parameter file

One runner, one graph, every style. A new hairstyle is a new file in `styles/`, not new code: the file holds
everything that belongs to the style (the reference's measurements, the part, the root lift, the falls and where
they end, the wave, the bangs, the finish), every number with its source, and the runner feeds those numbers to the
same tools the Warden's ponytail and waves were built with.

```
python3 tools/hair/pipeline/run.py tools/hair/pipeline/styles/warden-ponytail.yaml OUT          # run to the gate
python3 tools/hair/pipeline/run.py STYLE.yaml OUT --graph        # the nodes, their inputs and outputs
python3 tools/hair/pipeline/run.py STYLE.yaml OUT --bindings     # the tool constants the style sets, and which differ
python3 tools/hair/pipeline/run.py STYLE.yaml OUT --code ../pr99 # run a PR's tools (the waves need PR #99's)
python3 -m pytest tools/hair/pipeline/tests -q
```

## What "a parameter file, not a rewrite" means

Before, a style lived in the tools' code: the ponytail's photo ratios were a `PH` dict at the top of
`ponytail_flow.py`, its bangs a preset in `bangs_flow.py`, the waves' lift and wave constants at the top of
`photo_waves.py`. A second ponytail meant editing those constants in place, which changed the first one too, and the
value the approved field was built with lived only in a commit (the round 6 photo ratios never reached main: main's
`PH` was still the first, wrong sheet). Now the tools keep their defaults and the style file says what this style
is. Two styles can share the tools and differ only in their files; an approved style stays reproducible because its
numbers are in its file.

## The graph

A family is a kind of style with its own graph; both run the recipe's order, **measure, field, gauge, look, then
hair**, and nothing is groomed until John has seen the field.

| node | kind | tied (ponytail) | loose (waves) |
| --- | --- | --- | --- |
| provenance | gauge | every number names a flag | same |
| measure | stage | the refmeasure sheet's ratios against the style's; `ref:` sources checked | same (no sheet yet: a known fault) |
| field | stage | `ponytail_flow.py` (funnel, gravity, ear, lift as the stack, handover, free fall) | `photo_waves.field2` (combed head lines, fringe sheet, falls) |
| field_gauges | gauge | the recipe's bar: flux, packing, funnel, rank, hairline, handover, loose share, curl, slack, photo gaps | rank moves, lines turning up, smoothness, widths and ends against the photo, the root lift's angle |
| smoothness | gauge | `flow_smooth` on the bundle's middle line before its wave | (inside field_gauges: falls and fringe) |
| wave | gauge | the S wave on the middle line | the falls' wave: length, both amplitudes, twist, neighbours in phase |
| field_review | gate | stops until `approved.field` is the field's digest | same |
| look | stage | (`stages.look`) the field as fibres, `ponytail_render.py` | |
| bangs, bangs_gauges | stage, gauge | `bangs_flow.py`, its constraints and `flow_smooth` (set hair) | (the fringe is part of the field) |
| asset / hair | stage | `bangs_creator_asset.py` with `finish.steps` | `wavy_flow.py` photo mode (PR #99) |
| asset_gauges / hair_gauges | gauge | join cover | hairline, join cover |

Each node has typed inputs and outputs (`graph.TYPES`: a head export, a refmeasure sheet, a tied or loose field
folder, a field cache, bangs, a strand asset ...), checked on disk when the node finishes. Each node runs its tool in
a fresh process (`bind.py`) with the style's constants bound onto the tool's module, so no stage inherits another's
module state. Each node's key hashes the parameters it reads, its inputs' keys and the tools' code; a rerun into the
same folder reuses every node whose key is unchanged. `OUT/run.json` lists every node, what it read and made, its
time and every check.

A gauge's failing check stops the run where it stands unless the style lists it under `gauges.allow_fail` with the
reason: a known fault is reported in every run ("report it, do not tune the field to hide it"), never silenced. The
gate prints the field's digest and the figures to show John; when he approves, the digest goes into
`approved.field`. Any change that moves the field changes its digest, so an approval never carries over to a field
he has not seen.

## The parameter file

YAML (JSON loads the same way). A value is bare or `{v: VALUE, flag: FLAG, src: "where it comes from"}`; a mapping's
`_flag` and `_src` are inherited by its children. The flags (`params.FLAGS`):

| flag | meaning |
| --- | --- |
| SOURCED | from a cited source (paper, craft, anatomy) |
| MEASURED | off the reference (refmeasure sheet, photo crop) or the head |
| CREATOR | the Face Creator's own value (hair ground truth) |
| CALIBRATED | fitted so a named gauge matches the reference |
| JOHN | John's direct instruction (applied, still checked against research) |
| DRAFT | no source yet: a placeholder (experimental) |
| STYLE | an artistic choice of this style |
| DERIVED | computed from other parameters (formula in src) |
| TOOL | sampling, steps, seeds: cost, not look |

The provenance gauge fails a file with an unflagged number. A `src` of `ref:NAME` (or `ref:NAME*0.5`,
`ref:NAME*crown` for times 1 + head_top_hair) says the value is the reference's ratio, and the measure stage checks
it is. The sections, the same names in every family where the idea is the same:

- `inputs`: the head export, the refmeasure sheet(s). Paths are relative to the style file, then the project files.
- `reference`: the measured ratios in heads, the sheet's names for them, the tolerance, and `kept` (a value kept off
  the sheet, with the reason).
- `part`: where the hair parts (`line: centre`, a push away from it; for a tied style the bangs' `part_mm`).
- `root_lift`: tied: `model: stack` (the lift is the stack's thickness, recipe 1.6). Loose: the outer surface's height
  (`outer_mm`) or the leaving angle (`emergence_deg`), and the relaxation length (`relax_mm`) over which a hair turns
  from its root onto that surface: height T(1 - (1 - s/L)^2), leaving angle atan(2T/L), bend radius L^2/2T.
- `fall`: lengths and end targets (hem, layers, grade), the bending length and envelope (tied), the bands the falls
  land in (loose).
- `wave`: two perpendicular sinusoids, `out = A_out sin(k s + phi)`, `side = A_side sin(k s + phi + delta)` (`wave.py`):
  `length_hh`, `amp_side_hh`, `amp_out_hh`, `twist_deg` (delta: 0 a planar sine-generated meander, 90 a helix), where it
  grows in, and how locks share phase. Tied: `mass` (the tail's planar S) and `locks` (helical curls, each lock's
  phase). Loose: one wave with neighbours in phase (by height), checked by the wave gauge.
- `bangs`: the section (`section_rad`, `depth_mm` or `depth_hh`), the curve (tied: bangs_flow's B/D blend preset;
  loose: the S sheet's ends, handles, span and fairing), lift and feather.
- `loose_locks`, `framing`, `locks`, `finish`, `shading`: the rest of each family's stages.
- `gauges`: bounds (each flagged) and `allow_fail`.
- `tools`: `{module: {CONST: value}}` for a tool constant the vocabulary does not name yet. Use it sparingly; when a
  second style needs the same constant, give it a name in the family's `VOCAB`.
- `approved`: the field digest John approved.

## Adding a new style

1. **Pick the family.** Gathered to a tie (ponytail, bun, half-up): `tied`. Hanging from the head (loose, layered,
   wavy, straight): `loose`. A style neither builds (braids, a crop) needs a new family, see below.
2. **Copy the nearest style** in `styles/` to `styles/<character>-<style>.yaml`; change `style`, `title`, `inputs`.
3. **Measure the reference first** (project rule): refmeasure the photo, unit head top to chin, plumb through the ear
   canal; point `inputs.reference` at the sheet and write the ratios into `reference.ratios`. The measure stage
   checks them against the sheet.
4. **Set the style's numbers**, each with its flag and source: part, root lift, lengths and ends, wave (length,
   amplitudes, twist from the reference: a planar S has twist 0, a curl 90), bangs. A number you could not ground is
   DRAFT, never bare.
5. `--bindings` shows which tool constants change; `--graph` shows the run.
6. **Run to the gate** (`run.py STYLE OUT`). Fix failing gauges in the field's parameters, not by widening a bound;
   list a fault you cannot fix yet under `gauges.allow_fail` with the reason, and stop by the project's stopping rule.
7. **Show John the field** (the gate prints the figures). After he approves, put the digest in `approved.field` and
   run again: the hair stages run.
8. Commit the style file with the run's `run.json` numbers in the PR.

## Adding a family or a stage

A family is a module in `families/` with `VOCAB` (parameter path to tool constant, set before or after the tool's
`derive()`), `validate(style)` (what its tools cannot build, said before anything runs), `binder()` and `graph()`
(the nodes in order). A tool that computes constants from others gets a `derive()` (as `ponytail_flow.derive`) so a
bound primary carries through. A stage that has no `main` gets a small driver in `drivers.py`. Register the family in
`run.py FAMILIES`.

## Reproduction (2026-11-28)

| style | what was compared | result |
| --- | --- | --- |
| warden-ponytail | the runner's field against `ponytail_flow.py` run directly on the same head and sheet | `gauges.json`, every `field.npz` array and the views PNG byte-identical |
| warden-photo-waves | the runner's field against the wavy thread's field v14 (`field2_driver.py`, head-v14, n 8000) | `gauges.json` identical to the posted v14 gauges; drawings pixel-identical |

`--bindings` reports no constant changed from the tools' defaults for either file. The ponytail field matches
today's tools, not the 2026-11-25 `field6` folder, which was solved before later scalp changes (76,462 hairs then,
76,601 now; see the PR).

The bangs changed on purpose after the field was approved: `bangs_flow.brush` holds set hair to the round brush it was
dried on (`BRUSH_R_MM`, 19 mm, read over 10 mm as `flow_smooth` reads it). The crest of the fringe's lift on the convex
brow had bent 2.4% of the curtain's nodes to 12 to 17 mm; now none is under 17.5 mm (tightest 19.8). The brush moves a
line at most 2.1 mm (0.6 mm on average); the photo's fringe edge and side ends read the same (-0.003 and +0.013 head).

## Sources for the model

- Two perpendicular sinusoids with a phase offset: a circular helix is the rest shape of a Kirchhoff rod of constant
  curvature and torsion (Bertails et al. 2006, Super-Helices; Bergou et al. 2008, Discrete Elastic Rods); the planar
  case is Langbein and Leopold's sine-generated curve (1966, USGS Professional Paper 422-H).
- Neighbours in phase: a wave set is brushed out "to form a uniform wave pattern" (saloncentric.com; behindthechair.com).
- The rest of the numbers: in the style files' `src` and the tools' docstrings (the ponytail recipe's sources).
