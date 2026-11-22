# Render request pitfalls

What has gone wrong with render requests in this project, and how each one is fixed. The reviewer
(`review.py check`) reads this file on every request. An entry with a **Pattern** is matched
automatically against the lines a branch adds under `blender/` (and skipped for a file that matches its
**Unless**); the rest are listed in every review
for the reviewer to keep in mind while reading the diff.

Add an entry whenever a request is sent back or a render comes out wrong:

    python3 tools/render-review/review.py learn --title "..." --spot "..." --fix "..." [--pattern REGEX [--unless REGEX]] [--seen "..."]

Keep each entry to what was seen, how to spot it before rendering, and the fix. Refine an entry in
place when it turns out to be too broad; do not add a near copy.

## P001 The full sprite sheet requested by a thread

- **Seen:** John's rule since 2026-11-15. The sheet takes far longer than any quick job and blocks the queue.
- **Spot:** `sheet` in outputs; a character script (`blender/ranger/ranger.py`, `blender/rogue/rogue.py`) run as a view; `frames` or `--cells` covering 60 or more of the 80 cells.
- **Fix:** Ask for the cells that show the change (`0-7:2,4,6` is the usual set). The full sheet, `.blend` and world model run only as the scheduled task; send `{"branch": "...", "ready": true}` to be included.

## P002 A before render, or a faked one

- **Seen:** Requests #14, #18 and #26 rendered main as a "before"; #29 asked for a before by hiding the new lip paint on the branch.
- **Spot:** The commit is already on main; the note says before or baseline; a view hides or switches off exactly the layer the branch adds (`--hide`, `--set X=False`) only to imitate the old look. Hiding a layer to diagnose a problem in the new work is fine.
- **Fix:** Ask only for the after views. Compare with a cached render of main under `render-queue/results/` (the queue's refusal lists them).

## P003 A name used but never defined, or any crash during the build

- **Seen:** Request #5 died 45 s in with `NameError: name '_jaw_width' is not defined` after a merge dropped the helper that `head_front_y` still called. Main still carries that reference in a path that no longer runs, which is why it is ignored there.
- **Spot:** pyflakes "undefined name" that main does not have; a function moved between files (`ranger.py` to `base/female.py`) whose helpers stayed behind; a merge that takes one side's deletion and the other side's caller. The smoke run (`smoke.py`) builds the scene for real and settles it.
- **Fix:** Define the name or restore what the merge or move dropped. Run `python3 tools/render-review/smoke.py <script> <args> --out /tmp/x` before asking.

## P004 PNG output set while the scene is still on multilayer EXR

- **Seen:** Request #2 failed after 60 s: `enum "PNG" not found in ('OPEN_EXR_MULTILAYER')` in `face_views.py`. The build leaves the scene's output on the multilayer media type, where PNG is not a choice.
- **Spot:** A view script that sets `image_settings.file_format = 'PNG'` (or JPEG) after `build()` without setting `image_settings.media_type = 'IMAGE'` first.
- **Fix:** Before choosing the format add `if hasattr(sc.render.image_settings, 'media_type'): sc.render.image_settings.media_type = 'IMAGE'`.
- **Pattern:** `file_format\s*=\s*['"](PNG|JPEG|TIFF)['"]`
- **Unless:** `media_type`

## P005 A function defined twice after a clean merge

- **Seen:** Main has two `_srgb_lin` in `blender/ranger/ranger.py` since the skin merge (#10). Git merged both sides without a conflict; Python uses the second for every caller.
- **Spot:** The same top-level `def` or `class` name twice in one file with different bodies, after merging main. Both sides added a helper with the same name, or one side moved a function the other edited.
- **Fix:** Keep one definition. If both are needed, rename one and update its callers. A duplicate that main already has is main's to fix, not the branch's.

## P006 The branch is not pushed, or conflicts with main

- **Seen:** Requests #22, #35 and #36 were held: main had gained merges that edit `ranger.py` where the branch had moved code into `base/female.py`.
- **Spot:** The branch or commit is missing on origin; `git merge-tree` of main and the commit reports conflicts in code. A branch that moves code between files conflicts with every later edit of the old location.
- **Fix:** `git fetch origin main && git merge origin/main`, port main's edits to where the code now lives, push, and ask again.

## P007 Rendered files changed on a feature branch

- **Seen:** README rule: feature branches carry code only, so merges never conflict on binaries and nothing is re-rendered after a rebase.
- **Spot:** `blender/ranger/ranger-blender.png`, `blender/ranger/ranger.blend` or `public/models/ranger.glb` differ from main on the branch.
- **Fix:** `git checkout origin/main -- blender/ranger/ranger-blender.png blender/ranger/ranger.blend public/models/ranger.glb` and commit.

## P008 Asking for isolation

- **Seen:** A night of requests marked `mode: isolated` ran one at a time (2026-11-15).
- **Spot:** `"mode": "isolated"` in the request.
- **Fix:** Leave it out. It is ignored: branches are combined whenever their code does not overlap, and up to three quick jobs run side by side.

## P009 Hair that does not blend into the head

- **Seen:** Rogue rounds 5 to 12: a ledge at the hairline, temple locks standing off the cheek like blades with a slit of shadow under them, a ridge where hair shells meet, side locks reading as one thick strand. Each took a render round to find because the lit view hides them.
- **Spot:** The branch edits hair functions but asks for no clay view. In the code: a hair piece added without a joining shape where it meets the skull; a shell whose inner edge is not pulled flush to the skin; strands in one flat tone; a new piece's root starting outside the surface it grows from.
- **Fix:** Ask for `--head --clay` together with the lit view, so form and seam are judged in the same round. Give every added piece a joining shape, end edges flush with the skin, and group painted strands into locks with their own tone.

## P010 A view flag the script does not read

- **Seen:** View scripts parse `sys.argv` by hand (`opt('--size')`, `'--turn' in argv`). A misspelt or unsupported flag is ignored without a word, and the render shows the default instead of what was asked.
- **Spot:** A `--flag` in the view command that never appears as a string in the script at that commit; a view script path that is not in the commit (for example `blender/base/views.py` on a branch that predates the base figure).
- **Fix:** Use the flags the script reads (the review lists them), or add the flag to the script in the same commit.

## P011 Asking again for code that was already rendered or already failed

- **Seen:** Identical code is never rendered twice; a job that failed stays failed for that exact code.
- **Spot:** The job's key already has a folder under `render-queue/results/` (done, or `-failed`), or the same request is pending.
- **Fix:** Read the existing result or `render.log`. Push a new commit before asking again.

## P012 Shell characters in a view command

- **Seen:** The queue runs each view command through a shell exactly as written.
- **Spot:** `;`, `&`, `|`, `<`, `>`, `$`, backticks, a backslash or `..` in a view command, an absolute path, or an `--out` of its own.
- **Fix:** Give a script path relative to the repo root and plain arguments. Quotes around one argument are fine (`--hide 'Eyeshadow,Lashes'`).

## P013 Checking a change from the front only

- **Seen:** CLAUDE.md: one good front view proves nothing; most problems show up in the 3/4 and profile rows.
- **Spot:** `cells` or `--cells` with fewer than all 8 facing rows, for a change that alters the silhouette, hair, shoulders or clothing.
- **Fix:** Use rows `0-7` with a few columns (`0-7:2,4,6`, or `0-7:0,3,6` to include the idle).

## P014 A face check sent to the render queue

- **Seen:** John's rule since 2026-11-16: every figure face check goes through the face check API, not a render. Of the first 41 requests, 17 were face views of a face-only change (eyes, brows, lips, nose, jaw), each costing minutes of render time.
- **Spot:** The change since the branch was last rendered touches only face sculpt or face paint (`face_relief`, `mouth_relief`, `jaw_relief`, lids, `eye_relief`, iris, sclera, eyeshadow, lashes, brows, lips) and the request asks only for face views (`face_views.py`, `sculpt_views.py`, or `views.py --head` without `--cells` or `--turn`).
- **Fix:** Run `python blender/facecheck/facecheck.py <figure> --before origin/main` (no Blender, about 17 s, and a before is free there). Ask the queue only for what it cannot show: hair, cloth, final lighting, subsurface scattering and the sprite palette; say which in `"needs_render"`.

# Proposed from logged rejections (nightly, 2026-11-19)

These come from `render-queue/rejections.jsonl` (37 rejections on 2026-11-18, six garment branches). Each one
repeated on more than one garment. The reviewer applies them once John merges this.

## R001 The underarm and armhole edge left to chance

- **Seen:** 5 rejections on 3 garments: a crumpled wedge at the front of the armhole (sleeves, m2odto), a ragged slit
  and a knot of cloth at the bottom of each armhole and torn linen at both armholes (chemise, mjx1su), and a slit at the
  armpit with skin showing in the final turn (bodice #178, ffan3z). In each case a shared armhole cut ended in a flat
  edge or hard corner at the underarm.
- **Spot:** The diff adds or changes an armhole, a sleeve seam or a side slit (`armhole`, `sleeve`, `slit`, `underarm`
  in the garment script), and the request has no view of the underarm.
- **Fix:** Close the armhole as one rounded loop with no corner under the arm, and check the build's clearance report
  round that loop. Ask first for a 3/4 `--head` view of that side, with the arm at rest and raised, before any turn.

## R002 A long hem in the walk with no look at the feet

- **Seen:** 4 rejections on 2 skirts: toes through the hem at toe-off, the hem sinking into the floor when the body
  dips (riding skirt, fva9xp), and the front hem catching on the toes and folding between the feet (petticoat,
  ybma5q).
- **Spot:** A skirt, petticoat, robe or gown whose hem reaches below the knee, asked for with `--walk` or `--run`
  but without a profile view, or a hem change checked only standing.
- **Fix:** Give the hem a foot clearance and a floor limit in the build. Check the walk in profile, because frames 6
  and 7 (toe-off) and the body's lowest frame are where it fails.

## R003 Cloth solver numbers guessed instead of measured

- **Seen:** 4 rejections on 2 garments: cloth hung like tissue with a bending length of about 0 cm (petticoat,
  ybma5q), fabric too heavy with thick rolled folds, and crumpled like paper (chemise, mjx1su). The numbers were at
  the wrong scale until they were found on the swatch bench.
- **Spot:** The diff changes cloth settings (`bending`, `compression`, `tension`, `mass`, weight per point) and
  no swatch-bench result at that commit is cited.
- **Fix:** Find the numbers on the swatch bench first (a strip of the cloth, rendered in seconds), cite the bending
  length it shows, and only then render the garment.

## R004 Foot and toe meshes with inward normals or a melted forefoot

- **Seen:** 3 rejections on the foot thread (2026-11-21, 06:56 to 07:47Z): toes flat and melted into a plate, a dark
  notch at the toes' base in profile and 3/4, and a sheet of skin between the shins with inward normals on folded
  forefoot sections.
- **Spot:** The diff rebuilds or smooths the forefoot, toes or heel (SDF fillet, section lofts) and no normals or
  toe-separation check is in the build output.
- **Fix:** Check that every section loft has outward normals and no folded sections before the fillet, and render
  the profile and 3/4 crop of the toes (the notch and the plate show there first).

## R005 Render framing that does not match the request

- **Seen:** 3 rejections in the day: `--zoom` read as closeness when it is the view's width (the lips came out about
  100 px across), a body left unparented so every angle showed one side, and a figure rebuilt with the wrong parts
  because a module-level set was lost when the worker reused the module.
- **Spot:** A view or crop request that sets `--zoom`, takes a part off the rig, or relies on a module-level setting
  made at import.
- **Fix:** State the visible width in the request and check the first frame's size; keep the body parented to the
  rig; set per-figure parts inside the build, not at import.

## R006 Free lace ends longer than the reference line

- **Seen:** 2 rejections on the boot thread (2026-11-21, 02:25 and 02:38Z): free ends of 22/17 cm ran past the sole
  and hung in the air, and after the first fix 15/12 cm still ran below the mid-shin line John drew on the front view.
- **Spot:** The diff sets a free end length (bow tails, tie ends) as a fixed number, or sets it without a line to
  compare against on the garment.
- **Fix:** Derive the length from a landmark on the garment (mid-shin, hem, knee), put that line on the clay view,
  and check the ends against it in the first round, not after John's circle.

## R007 Lapels and collars laid on, not cut from the front

- **Seen:** 2 rejections from John on the redingote (2026-11-20, 14:37 and 15:47Z): lapels added as ready-shaped
  triangles laid on the chest, and a bodice front fold that came out as a flat striped flap with ragged cut edges.
- **Spot:** The diff adds a lapel, collar or revers as its own flat piece placed over the chest instead of
  drafting it as part of the front panel with a fold line.
- **Fix:** Draft the lapel as the folded-back part of the front panel (roll line, break point, return), solve the
  fold, and check the 3/4 view for a flap standing off the chest before sending it.

## P015 Colours that touch but do not separate

- **Seen:** 2026-11-20: a near-black lace darkened the tongue; a first linen replacement matched the brass eyelets' hue
- **Spot:** Run tools/render-review/colorcheck.py FIGURE before a render that changes a material colour: it flags touching parts with a small lightness gap and the same hue, and a cord darker than the shadow of what it lies on
- **Fix:** Pick the colour from the check's table, not by eye: keep dL 12+ or dE 25+, and 20 degrees of hue when the values are close; a lace over a dark opening must be lighter than the surface's shadow tone

## R008 Hair guides placed or combed without a check against the scalp and the face

- **Seen:** 11 of the day's 17 hair rejections, all on one thread (2026-11-21, 18:38 to 20:21Z, logged by the thread
  itself): guides rooted behind the crown that flipped forward over the face, bang guides planted on bare skin in front
  of it, sideburn guides whose first segment ended 5 mm inside the head, and four bang rounds (hung straight like a
  sheet, then swept out like antennae, then the inner tips still over the pupil, then dipped 15 mm into the eye socket).
- **Spot:** The diff plants, combs or sets hair guides (a bang section, a part, a sideburn, a framing lock) and the
  first render of the round is the first time anyone looks at where they ended up.
- **Fix:** Before the render, check each guide numerically: its root lies inside the hair-bearing scalp mask, its first
  segment clears the skin, and no node lies inside the eye socket; for bangs compare the inner tip's |x| with the pupil's
  and the bang's outward angle with the head's width at that height, and fix the set in one pass from a styled target
  shape instead of one parameter per round.
