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
