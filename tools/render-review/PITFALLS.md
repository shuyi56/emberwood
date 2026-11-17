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

## P015 Creases, ridges and ledges in fused hair, and clothes through the hair

- **Seen:** John circled seams on the warden's hair where pieces met; the rogue's side locks leave creases of about 45 px where they join the cap at the temples (rounds 9 and 17). Fusing three wisps into the cap took the rogue from 1 flag to 21 in one commit (c2081d7), and nothing showed it until the render.
- **Spot:** `hair_check.py` (run by the review inside the headless build whenever a commit touches hair code): a crease is a concave bend tighter than radius 0.3, a ridge a convex edge tighter than 0.15, a ledge hair stepping over hair by 0.3 or more, each as a blob of 10 or more pixels in one of the 8 facings; a clash is another mesh coming out through the hair's outside in a walk frame. The marked picture is `hair-check.png`. A flag of 25 px or more that the branch's last measured commit did not have sends the request back.
- **Fix:** Crease: widen the overlap between the two pieces and relax the join after fusing. Ridge: round the piece's corner or sink it into the mass. Ledge: root the lock inside the hair it grows from instead of laying it on top. Clash: move the hair clear in that pose or weight it to follow the body. Check before asking: `RQ_HAIRCHECK=/tmp/h python3 tools/render-review/smoke.py <view script> <args> --out /tmp/x`.

## P016 Painted strands that do not match the hair

- **Seen:** Rogue rounds 13 to 15: the painted side locks were half-transparent, the skin showed through and tinted them pink (78% of the painted texels half-transparent at b14e416, 32% once each strand was opaque), and their colour had been shifted away from the hair's ramp to compensate. John rejected the painted locks after three render rounds.
- **Spot:** The strand paint part of `hair_check.py`: more than 45% of a strand texture's painted texels with alpha between 0.08 and 0.7, or its mean colour further than 0.045 in chromaticity from `MATERIALS[M_HAIR]['ramp']`. The texture over a skin tone is saved as `paint-<name>.png`.
- **Fix:** Paint each strand opaque and thin a lock by the gaps between its strands. Take the tones from the hair's ramp unchanged; if the lock renders darker or redder than the hair above it, fix the material or the lighting it sits in, not the paint.

## P017 Turning one knob further when the last render barely moved

- **Seen:** Rogue rounds 19 and 20 (2026-11-16): a crotch mound 'shrank only slightly' after the seat's front-centre recession went to 0.22, so round 20 doubled it to 0.42. Measured headless, the centre was already 0.1 behind the thigh fronts and round 20 made a trough 1 deep; the mound that stayed was the inner-thigh fill from another function (leg_pt), which neither round touched.
- **Spot:** A request whose diff only raises the same constant again, with a note that the previous round changed little. Build both commits headless and ray-cast the surface profile where the problem is: if the last change already moved that surface, the visible problem is made by something else.
- **Fix:** Before another round, measure which surface is actually forward at the spot (a ray-cast profile at a few x across it, from the built scene) and change the function that makes it. Do not raise the same number again.
- **Also seen:** Round 22: the front factor went from 0.35 * inner to 0.55 * inner ** 3 and the surface moved 0.11; at the back the marked lobe and the columns beside it moved in together, so the lobe stood as proud as before between deeper grooves. Measure the problem relative to its neighbours (`tools/render-review/profile.py`), not by how far one point moved.

## P018 A new part built under a shell that already covers its place

- **Seen:** Rogue ears, request #56 (2026-11-16): the ears were added at th 1.45 to 1.95 while the hair cap's HAIRLINE still ran down past them, as it was drawn for a head without ears. The render showed only a lobe tip. The hair check passed, because a part hidden inside the hair is not a clash.
- **Spot:** A commit adds a mesh (ears, jewellery, a collar) and leaves the shell over that spot unchanged: compare the new part's span with the curve that bounds the covering shell (HAIRLINE for the cap, hems for cloth) in the same coordinates. In the headless build, ray-cast from outside toward the new part in profile and 3/4: the first hit must be the new part over most of its area.
- **Fix:** Open the covering shell round the new part in the same commit, with clearance, and state both spans in the request note (the part's and the opening's). Ask for the render only once the ray-cast shows the part outermost.

## P019 A build that takes many minutes before the first render

- **Seen:** Warden with the cape, db7668d (2026-11-16): the scene took 12 to 13 minutes to build on the requester's machine and 21 minutes in the review with the hair check (bb142f7, before the cape: 8 minutes; the rogue: 3). The review's limit was 15 minutes and reported a crash that was not one. The queue pays the same build before every render of that figure.
- **Spot:** The smoke line in the review: 'reached its first render in N s' above 600, or 'no render reached' with no Python error in the log. In the diff: a dense grid built more than once (an outside and a lining), a per-vertex Python loop over every mesh for weights or clearance, a lookup table rebuilt per piece.
- **Fix:** Profile one build (python3 -X importtime is no help; time each build step) and cut the slow step: build the sheet once and offset it for the lining, halve grid resolution where a subdivision follows, cache tables. A real crash shows a traceback; a timeout does not, so rerun with RQ_SMOKE_TIMEOUT raised before sending the request back as broken.

## P020 A new skin mesh that falls into another part's tone branch

- **Seen:** Warden bare arms, 02a9e43 (2026-11-16): skin_tones picks its colouring by object name (head, ears, neck, then everything else as a hand). The new Arm.R and Arm.L took the hand's branch and its fingertip flush turned the whole arm pink. It cost one render to see.
- **Spot:** A commit adds a mesh in M_SKIN (mesh_obj(..., M_SKIN, ...)) under a name skin_tones has no branch for. Read the if and elif chain on o.name in skin_tones at that commit: an unnamed mesh lands in the last branch.
- **Fix:** Add the new name's own branch to skin_tones in the same commit, and match its tone to the neighbours where they meet (wrist, shoulder, neck).

## P021 Hair strands or ribbons wound inside out

- **Seen:** Hair locks, f4f54d6 (2026-11-17): the strand ribbons of the new lock clumps were wound inward, so the fringe rendered black on one side and pale on the other on both figures. The build and the hair check passed, because both look at shape, not at which way a face points. It cost a render of four jobs.
- **Spot:** A commit that builds hair (or any thin shell) from its own vertex loops: new grid, ribbon or tube code in blender/shared/locks.py or a figure's hair functions, with no recalc of normals after it. In the headless build, take each hair face's normal against the direction from the head's centre to the face: more than a few percent pointing inward on a piece means it is wound the wrong way.
- **Fix:** Wind the faces so the normal points away from the head, or call bmesh.ops.recalc_face_normals on the piece before it is joined, and check the share of inward faces in the smoke build before asking.
- **Also seen:** Rogue temple locks, f13d5e2 (2026-11-17): the lock check reported flap and float on every temple wisp after locks.hang_down, which makes them hang straight past the jaw at John's request. A lock meant to hang free below the jaw stands off the head by design, so those lines are not a fault: forward, and have the requester measure such locks against their hang line so the check stays quiet when right and loud when wrong. Kink, inward, glint, facing and bare lines still block.

## P022 A lit view command beside its own --clay twin

- **Seen:** Warden head turn, 37f53b3 and c75919e (2026-11-17): the reviewer itself added a second command, the same views with or without --clay, to get both looks. views.py --clay renders the lit views first and then the clay ones (turn.png, head.png, turn-clay.png, head-clay.png), so the pair built the figure twice and rendered the lit views twice. The queue caught it and dropped the duplicate.
- **Spot:** Two commands in "views" that are equal once --clay is taken out.
- **Fix:** Keep only the --clay command. To add clay to a request, add the flag to the existing command; never add a second command.

## P023 A build killed for memory, read as a crash

- **Seen:** Creator faces on the Warden, 44308e5 (2026-11-17): five Warden builds ran side by side in the review (four views of one request and one of another). Each peaks near 7 GB, the container has 15 GB, and two were killed (exit 137, the log ends in "Killed" with no traceback). The review reported them as build failures. Rerun two at a time, both passed.
- **Spot:** Exit 137 or a bare "Killed" as the last log line, with no Python traceback. dmesg shows "Memory cgroup out of memory".
- **Fix:** Never send this back as a broken build. Rerun the killed views with nothing else building (the review now runs two builds at a time, and a second review must wait for the first). Tell the queue the build's peak so it does not run three side by side, and tell the requester when one build passes about 5 GB.
