# Ranger render queue

One thread, "Centralize render requests across threads", is the only thing in the project that renders the
ranger (Blender sheet, .blend, world model, close-ups). Other threads never run `ranger.py`, `export_glb.py`,
`sculpt_views.py` or any other Blender render themselves. They push their code and send a request.

## The code and the running copy

This folder (`tools/render-queue/` on `main`) is the source of the queue. The running queue is a copy of it in the
project's shared folder, `/mnt/project-files/render-queue/`, with its state beside it (`queue.json`, `results/`,
`events.jsonl`, `stats/`). Change the code here through a pull request; once it merges, the render-queue thread copies
it to the shared folder. To run this copy against the shared state, set `RQ_HOME=/mnt/project-files/render-queue`.

| Piece | What it does |
| --- | --- |
| `rq.py` | the queue: `add`, `plan`, `serve`, `list`, `export`, `ready`, `cancel`, conflicts and the cache |
| `rq_run.py` | the one way a Blender script is run: `python3 rq_run.py [--quality check] [--memo DIR] [--keep-scene] -- SCRIPT ARGS` |
| `rq_frames.py` | renders the sprite cells a `frames` request names |
| `rq_stats.py` | nightly throughput, wait and idle figures |
| `rq_flow.py` | the flow check: each garment's strain field, seams and jaggedness, with a heatmap, before the render |
| `rq_jit.py`, `rq_hot.py` | region rendering and build reuse: kept as options, not used by the queue's jobs |

A view job is any script in the requesting branch, run through `rq_run.py` with the request's arguments plus
`--out DIR`. The standard figure views are `blender/base/views.py` (`--figure`, `--mannequin`, `--turn`/`--head`,
`--angles`, `--clay`, `--walk N`/`--run N`, `--size`, `--samples`, `--denoise`).

## How to ask for a render

Run `rq.py ping --thread <your id>` first (see "Ping the queue before you ask for more" below). Then, by the kind of request
(John, 2026-11-19, to cut the messages between sessions):

* **Check quality** (`--quality check`: a crop, a pose strip, a clay or lit look you only need to see): add it yourself,
  with no reviewer and no message to anyone. Push your commit and run

  ```
  python3 /mnt/project-files/render-queue/rq.py add --thread <your id> --branch <branch> --sha <sha> \
      --outputs views --view "<views.py command>" --quality check --note "what you are checking"
  ```

  The request goes into the queue at once. The answer to `add` prints your warnings.
* **Final quality, `frames`, `glb`, or `ready`**: send the **reviewer thread**, "Review render requests before queueing"
  (cmsg_01CnnKqGvuWEUi2G9X9cb9nBJkqDv7Ktra4a31euzvMxpx), a message with `message_thread` holding one JSON block. The
  reviewer reads it first (see "Review before the queue" below) and passes it on to the queue, or sends it back to you
  with fixes. `rq.py add` refuses a final-quality request that the reviewer has not passed.

Do not message the render-queue thread to ask for a render or to ask whether it is done. The JSON block:

```json
{
  "branch": "claude/project-thread-xxxxxx",
  "sha": "optional: a pushed commit; the branch tip if left out",
  "outputs": ["frames", "views", "glb"],
  "cells": "0-7:2,4,6",
  "views": ["blender/ranger/sculpt_views.py --looks clay,full --size 512"],
  "mode": "combined",
  "note": "what you are checking"
}
```

Add `"quality": "check"` for a quick look: every `views` job of the request renders at 8 Cycles samples with the
denoiser, about four times faster per image, and is cached apart from the final render. Leave it out (or say
`"final"`) for anything John signs off on. `frames`, `glb` and the sheet always render at full quality.

| Output | What you get | Priority |
| --- | --- | --- |
| `frames` | The sprite cells named in `cells` (`ROWS:COLS`, rows 0-7 are the facings, cols 0-1 idle and 2-9 walk), each cell as a PNG and all of them on one 3x contact sheet | first |
| `views` | Any close-up script that lives in your branch, run with the arguments you give plus `--out DIR` | first |
| `glb` | `ranger.glb` from `export_glb.py` | after frames and views |

**Never ask for the full sprite sheet.** It is not a requestable output. The full sheet, `.blend` and world model
are rendered once a day by a scheduled task (2:41 AM Toronto), only while no other job is waiting, and they stop
to make room the moment a request arrives. To have your branch included in that run, send
`{"branch": "...", "ready": true}` once the work is finished.

**Never ask for a "before" render.** Main's code is rendered once and reused. A request whose commit is already
on `main` is answered from the cache, and it is refused if that exact job is not cached. The refusal lists the
cached renders of main to compare against. Don't fake a before on your branch either, for example with a view that
hides your new layer. Ask only for your after views and compare them with a cached main render.

Every request is combined with the other waiting branches whenever it merges cleanly with them and changes
different functions of the scripts (see "Batching" below for sets and the collation window). One render then covers all of them, and the reply says what else was in it.
`mode: isolated` is still accepted but ignored. Up to three quick jobs render side by side.

## Ping the queue before you ask for more

Run this before every render request, and again before a second one (John, 2026-11-19). It is read-only, costs one
tool call and wakes no other thread, so use it instead of asking the queue thread "is my render done?":

```
python3 /mnt/project-files/render-queue/rq.py ping --thread <your thread id>
```

It prints how busy the queue is (idle, light or busy), what is rendering and for how long, what is waiting and
when each request should start, any announced intents, and about when a new request would start. With `--thread`
it adds your own requests, your last result and where its files are, your hogging warnings, and advice. Add `--json`
for a machine-readable answer.

What to do with the answer:
- If you already have a request rendering or waiting, wait for it, or send a newer one for the same views, which
  replaces the waiting one. Do not stack a second request on the first.
- If the queue is busy, send a check-quality crop first and keep full renders for when it has room.
- If your last result is already back, read it before you send another request.
- If you are flagged for hogging, change the request (a cheaper crop) or check that your edit changes the build.

`rq.py add` reminds a thread that has not pinged in the last 15 minutes (it logs an `unpinged` event, which the
nightly stats count). The reminder never blocks the request. The reviewer thread pings on a thread's behalf when it
forwards a request, and tells the thread to ping when the queue is busy.

## Batching: sets, intents and the collation window

John's rules (2026-11-19). The queue does all of this itself; a thread only has to announce its work.

**Announce a render before you do the work.** When you start on a change you will want rendered, send the
reviewer thread an intent along with your plan, and the reviewer passes it to the queue:

```json
{"intent": true, "branch": "claude/project-thread-xxxxxx", "piece": "sleeves", "quality": "check",
 "eta_minutes": 10, "view": "blender/base/views.py --figure sleeves --mannequin --head --angles 3/4 --clay-only"}
```

The queue records it with `rq.py intent --thread T --branch B --piece FIGURE [--set NAME] [--view CMD]
[--quality check|final] [--eta MINUTES] [--note TEXT]` (`rq.py intents` lists the open ones). `piece` is the
figure module, `blender/<piece>/<piece>.py`, and its set comes from `sets.json`. The intent is closed when your
request comes. A newer intent from you for the same piece replaces the old one.

**Pieces of one set render together, always.** When pieces of one set are waiting together (in
`sets.json`: the riding habit is chemise, corset, petticoat, skirt, redingote bodice and sleeves), the queue
renders them in one build, with every piece on one body, in wearing order. The job is `views.py --figure outfit`,
using the queue's own `rq_outfit.py` with `OUTFIT=skirt,sleeves`, and every piece's thread gets the same
pictures. The views and angles are the union of what was asked, and the size and samples are the largest. The
`--head` close-up is framed on the pieces that asked for it. Each piece is still fitted by its own module, so
the combined render shows where the layers meet.

**The collation window.** The clock starts when the first piece of a set is ready (its request). The queue holds
that job for up to **5 minutes** (`READY_CUTOFF_S`) while another piece of the set or another view of the same
figure is announced and not yet sent. When the window ends, the batch renders with whichever pieces are ready.
A slower piece goes in the next batch, and its thread is not held up either. Once every announced piece has
come, the job starts at once. A job nothing is announced for waits only 45 s (`WINDOW_S`) after its newest
request, to catch requests sent together, and never longer.

**Other batching rules:**
- **One build per scene.** Views of one figure on one commit, with the same body, pose (`--frame`), env and
  quality, run as one `views.py` call when that renders no more images than the separate calls would.
- **Drafts with drafts.** A check render never shares a job with a final, and checks run before finals.
- **Identical requests render once.** A job with the same key serves every request that asks for it, and the
  cache serves repeats.
- **Edits coalesce.** A new request for the same branch with the same views replaces one that is still waiting,
  whether pinned to a commit or not, so a stale commit is never rendered.
- **Caps.** A merged commit holds at most 4 branches (`MAX_BATCH`) and a combined job at most 6 requests
  (`MAX_MERGE`). If a combined job fails (other than by being killed at 30 minutes), each of its requests is
  retried alone, so one broken piece does not lose the lot.

Every job start writes a `batch` line to `events.jsonl`: what it combined, how long its first request waited,
what it was held for, and the announced pieces it left out. Coalesced requests and solo retries are logged
there too. The nightly job's `rq_stats.py` reports these numbers (builds saved, duplicates, coalesced
requests, holds that gained a partner, stragglers and how late they came) and proposes changes to the window,
the cutoff and the caps from them.

## Crops and the pose test strip

Two cheap views the queue adds to `views.py` (its own `rq_views.py`, put in the job's worktree). Both run at check
quality unless the request says `"quality": "final"`. Both batch with the figure's other views and with the
other pieces of its set: the crops and the poses come from the same build as the turn.

**Crops: render only the problem area.** Use `"views": ["crop shoulder,armhole --figure sleeves --mannequin"]`, or
add `--crop SPECS` to a `views.py` line. Each spec gives `crop-NAME.png`, a row over `--crop-angles` (default:
`--angles`, else 3/4). Crops use the same fixed camera and lights as `--head`, so crops of two commits line up
pixel for pixel. A spec is:
- **A region by name:** `head`, `face`, `neck`, `shoulder`, `armhole`, `bust`, `elbow`, `waist`, `wrist`,
  `hand`, `hips`, `knee`, `hem` or `ankle`. Each is measured from the base figure's sizes at that commit and
  centred on the figure, so both sides show at every angle.
- **A widened region:** add `+PAD`, as in `hem+0.3` for 30% more.
- **A box from a turn you already have:** `box:X0:Y0:X1:Y1@S`, in pixels of one tile of a `turn.png` rendered at
  `--size S`. A fault seen in a turn is cropped exactly where it is.

`--crop-size N` sets the height in pixels (default 384). `--clay` and `--clay-only` apply to crops as well.

**Pose test strip: joint extremes before a full turn.** Use `"views": ["pose-strip --figure sleeves"]`. It
renders one clay tile per pose from one fixed 3/4 camera, laid out as `poses.png`. The poses:
- `rest`
- `arms-up` (arms raised out to the side)
- `arms-fwd` (arms reached forward)
- `elbows` (hard bend)
- `step` (deep stride)
- `crouch`
- `twist` (chest against hips)

`--poses arms-up,step` picks some poses, and `--pose-angle front` changes the camera. Shoulder, armhole and
hem faults show here first. On the riding habit's skirt and sleeves (`views-57cafbc602ee`), the arms-up pose
tore the sleeve caps off the shoulder, and the skirt folded into itself in the step and the crouch. Check a
garment's pose strip before asking for a final turn. The rig itself is weighted for the walk, so a pose that
tears the body as well as the cloth is a rig limit, not a fault of yours.

The nightly report counts crops, pose strips, how long jobs with only crops or poses take, and the rejections
that came out of a pose strip.

## Telemetry and the early geometry check

John's telemetry (2026-11-19). Every `views.py` job now runs through the queue's `rq_views.py`, which runs
`views.py`'s own code unchanged and adds measurements. The job's key is unchanged, so cached results stay valid.

**Per job, in `manifest.json`:**
- `queue_wait`, the time from the first request to the start
- `phases`: Blender's startup, loading the scripts, the build, the geometry checks, the renders, the crops and
  the poses
- `settings`: quality, size, samples, the scene or set, and the mannequin
- `solver`: the git blob of every solver, settle, sew, xpbd or drape script at that commit
- `cause`, a short reason when the job failed
- `geometry`, the geometry check's numbers

The same phases and cause go on the job's `done` or `failed` line in `events.jsonl`.

**The geometry checks run after the build, before the first render,** and are saved as `geometry.json`:
- `inside_pct` and `max_depth`: the share of each garment's vertices inside the body, and the deepest. This is
  penetration and clipping through the body.
- `stretch_p99` and `squash_p1`: how far the garment's edges are stretched or squashed from the garment as
  built.
- The body's height, shoulder width and hip width and depth, so a drift in its ratios shows from commit to
  commit.

The checks never fail a render on their own. `RQ_PRECHECK` holds thresholds, for example
`{"inside_pct": 5, "stretch_p99": 1.6}`. With it set, a garment past a threshold stops the job before it
renders, and `precheck.json` says why. The thresholds stay unset until the nightly job has matched the
numbers against the logged rejections. Seam-length mismatch needs each garment's seam pairs, so it is not
checked yet.

**The nightly report** (`rq_stats.py`, section "Telemetry") gives:
- the phase medians and totals
- failures by cause
- the cache-hit rate (jobs already rendered when a request was first planned)
- requests by quality (drafts against finals)
- each thread's cost: requests, checks and finals, and render time, with a shared job split among its requests
- garments past the geometry bands (over 2% inside, stretched past 1.3x, squashed below 0.7x)
- body-ratio drift past 1% from the median
- rejections by tag, garment and who, and the rounds of requests per piece

## The flow check: strain before the render

John's request (2026-11-19): after a build or solve and before the render, read the stress and strain across each
garment. A smooth field means the cut and the body agree; jagged peaks mark the faults that come back as
rejections: dart tips and seam corners where curvature concentrates, paired seam edges of unequal length, grain or
bias errors, too little ease at the armholes, shoulders or seat. `rq_flow.py` does it, and `rq_views.py` runs it
right after the build, beside the geometry checks:

- **On by itself for outfit jobs** (`--figure outfit`) **and pose strips** (`--poses`), where it also runs in every
  pose (the rig poses the garment and it is read back, so the poses cost no render). Any other views job asks for
  it with `--flow`. `RQ_FLOW=0` turns it off.
- **Report-only.** It never stops a render. It flags `watch` (past the approved band) or `likely rejection` (well
  past it, or a sharp peak at a seam end), and the log, the manifest's `flow` and the nightly report say so.
- **Files in the result:** `flow.json` (per garment: the numbers, the flags and the worst spots with where they are
  on the body, such as "left armhole, back", and what they most likely are), `flow-heat.png` (strain, jaggedness
  and bending in four views and on the flat pattern, the worst spots ringed and numbered as in `flow.json`), and
  with poses `flow-poses.json` and `flow-heat-pose-NAME.png` for each flagged pose. Attach the heatmap beside the
  render it came with.
- **What it measures:** against the flat pattern when the mesh carries one (a UV layer `pattern`, `flat` or `cut`
  that really is the cut: laid on the garment as built it must give a smooth, moderate strain), otherwise against
  the garment as built, in the rig's rest position (then what the pose does to it shows): principal stretch and
  squash, shear, Green strain, bending energy density (as a fold radius), how far each triangle's strain stands
  above its neighbours (jaggedness, roughness, gradient, isolated peaks counted one per place, the sharpest
  peak), and each seam's paired edge lengths. A garment script can declare its seams exactly with a `seam_mate`
  point attribute (each point's partner, -1 for none); otherwise they are found from the UV seams and from
  separate pieces' edges that lie on each other. Slivers from trims and hardware (eyelets, studs, stitches,
  plates) are left out. None of main's garments carries its flat pattern yet: the hose's and the chemise's UVs
  are texture maps (banded in steps of 0.5 down the leg, folded at the shoulders), so they and the rest are
  measured against their built shape. A garment built from flat pieces, as John's construction rule asks,
  should keep its pieces' flat coordinates as a UV named `cut`; then the check reads the cut against the body.
- **The band comes from John's approved garments, per pose.** `flow_baseline.json` holds the limits for the rest
  pose and for every pose of the strip, calibrated by
  `python3 rq_flow.py --calibrate flow_baseline.json RESULT/flow-poses.json ...` on main's hose, boots, chemise
  and vest (no-render jobs `views-0a8c28aac239`, `views-05159b056b65`, `views-0e767d28ffaa`): each limit is the
  worst approved value with a 25% margin, so a garment is flagged only when it does worse than what John already
  passed, in that pose (the rig strains even approved cloth in arms-up). An approved garment more than 3x worse
  than every other on a metric is left out of that metric's band and listed under `outliers`: so far that is the
  vest, whose side panels under the arm stretch 2.7x in the rest pose (hidden by the arm in its renders) and tear
  in arms-up and arms-fwd. The band is loose where the approved garments are rough (the chemise has one sharp
  peak on the right forearm, the left boot shears at the ankle crease); fixing them and re-running the
  calibration tightens it. Re-run it when John approves more garments.
- **Cached.** Each garment's result is stored under a key made of its posed points, its reference, its triangles,
  the check's version and the band (`RQ_FLOW_CACHE`, else `RQ_HOME/flow-cache`). The fold cache gives a
  texture-only or material-only re-run the same settled points, so it gets the same key and costs nothing.
- **Measure without rendering:** `tools/render-queue/rq_views.py --figure F --mannequin --poses all --flow-only`
  builds, checks in every pose and renders nothing. `python3 rq_flow.py --selftest DIR` checks the metrics on
  three test sleeves (smooth, a stretched patch, a seam drawn 10% short) outside Blender.

## Review before the queue

The reviewer thread checks each final-quality, `frames`, `glb` or `ready` request without rendering anything, in about
two minutes. Check-quality requests skip it: a thread adds those itself with `rq.py add --quality check`.

* the request itself: outputs, cells, view commands; no full sheet in any form (not as an output, not a character
  script run as a view, not `frames` covering most of the sheet); no before of main that is not cached
* the code: the branch is pushed and merges with main, nothing is defined twice, no name is used that nothing
  defines, each view script exists and reads every flag it is given
* a headless run of each job up to its first render (imports, arguments and the whole scene build), so a crash
  is found here instead of minutes into the queue
* the pitfalls collected so far (`tools/render-review/PITFALLS.md` in the repo), and a read of the diff since your
  branch was last rendered

**Face checks are never rendered.** Check a face with
`python blender/facecheck/facecheck.py <figure> --before origin/main` (no Blender, about 17 s, and a before is
free there; on branch `claude/project-thread-k27az6` until PR #26 merges). A request that only checks a face (only
face sculpt or face paint changed, only face views asked) is answered by the reviewer with that tool's views and
measurements under `render-queue/facechecks/`, and does not reach the queue. If you need what it cannot show
(hair, cloth, final lighting, subsurface scattering, the sprite palette), add
`"needs_render": "<what you need to see>"` to the request.

**Hair is checked before it is rendered.** When your commit changes hair code, the reviewer runs the hair check
inside its headless build: creases, ridges and ledges in the hair in all 8 facings, clothes coming through the hair
in the walk, and the strand paint (half-transparent strands, colour away from the hair's ramp). A flag of 25 px or
more that your branch's last measured commit did not have sends the request back with the marked picture under
`render-queue/review/hair/`; flags that were already there are listed and the request goes through. Run it yourself
with `RQ_HAIRCHECK=/tmp/h python3 tools/render-review/smoke.py <view script> <args> --out /tmp/x`.

**A material change renders the material, not the figure** (John's rule, 2026-11-17). When the commits since your
branch was last rendered only change a material (shader nodes, colour, roughness, metalness, texture, a look
table such as `BRONZE_LOOKS`), ask for a swatch render: a script in your branch that puts the material on a few
stand-in shapes (a sphere, and a cylinder or slab at the size of the piece it dresses), lights them with the
figure's own light and world, and renders them in seconds without building the figure. Several looks go side by
side in one picture from one run, not one job per look. Import the material function from the figure module
without running its build; if the module cannot be imported that way, split the material out first. The reviewer
sends back a figure render (`views.py`, `--turn`, `--only`, `--no-body`) whose diff is material-only, with this
rule as the fix. If you need to see the material on the figure (against the hair, in the figure's own shadows),
add `"needs_render": "<what you need to see>"` to the request; that runs as one figure job, never one per look.
For scale: the four-look bronze study (#109) built the whole warden four times, about 15 minutes, to show armour
that rendered in seconds.

**One view first, then the turn** (John's rule, 2026-11-17). A change is first rendered from one angle, the one
that shows it best (`views.py --angles front`, or `3/4`, `profile`, `back3/4`, `back`). Look at it, and only when it
is right ask for the full five-angle `--turn`, with `"confirmed": "<what the one view showed>"` in the request. The
reviewer sends back a `--turn` with no `confirmed`, and a `confirmed` turn must be at the commit of the one view
or a later commit that changes nothing it showed.

**Clothes go on a mannequin** (John's rule, 2026-11-17). A clothes change that needs a body under it renders on the
mannequin (`views.py --mannequin`): the figure's body shape with a plain head and no face sculpt, face paint, ears
or hair, which are most of a build. Cloth, armour and gear are built and fitted exactly as on the real figure. Ask
for the real figure only with `"needs_render"` saying why (the clothes against the hair, a hood on the head), and
material-only changes still go to a swatch.

A request that passes is forwarded to the queue unchanged, pinned to the commit that was reviewed, and you are
told in one line with any warnings. A request that does not pass comes back with what is wrong, where, and the
fix; push the fix and send the request again. To check your own branch before asking, run
`python3 tools/render-review/smoke.py <script> <args> --out /tmp/x` (it renders nothing; needs `pip install bpy`).

The render-queue thread still does everything it did: planning, combining, rendering, the cache, conflicts it
can resolve, and delivering results to your thread.

## Log every rejected render

John's rule (2026-11-19): when a render is rejected, by John, by the reviewer or by your own look at it, log it with
the reason before you fix it:

```
python3 /mnt/project-files/render-queue/rq_reject.py add --request 177 --tags crease,seam \
    --reason "side seam shows through the sleeve cap in the 3/4 and profile" \
    --fix "pushed the sleeve cap 2 mm out over the seam" --by John \
    [--file /mnt/project-files/render-queue/results/views-KEY/head.png] [--artifacts black_patches] [--region x0,y0,x1,y1]
```

`--request` is the queue's number, which fills in the branch, commit, views and result folders. Tags: clipping, gap,
crease, spike, stretch, seam, silhouette, proportion, pose, shading, colour, noise, framing, blank, other (`rq_reject.py
--help` says what each means). `--artifacts` takes the post-render flag ids when an image check could have caught it.
The nightly job reads the new rejections, proposes PITFALLS.md rules and flag thresholds for the tags that repeat, and
opens a pull request for John.

## Speed-ups every job gets

Every script runs under `rq_run.py`, which remembers the head's surface between runs: the eyes' sculpt and the
ears solve `head_front_y` and the head's side for every vertex round them, about 70% of a figure's build. The
answers are stored under a key made of `female.py`, `ear.py` and the female module's settings and functions, so
they are reused only while the head's code is unchanged, and the stored value is exactly what the function returns.

Since 2026-11-18 every job also keeps Cycles' copy of the scene loaded between its renders (`--keep-scene`), so a
five-angle turn loads the figure once instead of five times; only the turned figure and the camera are updated.
It changes no pixel (checked on the warden with her hair: a view rendered fresh and again from the kept scene are
identical) and took a mannequin turn from 180 s to 137 s. So several angles in one job cost little more than one:
when the one view is confirmed, ask for the turn as one job, not one job per angle.

Region rendering (`rq_jit.py`: a 4-sample probe finds the pixels a change touches, only that region is re-rendered
and pasted onto main's image) is built and exact, but at the queue's sizes (480 px, 48 samples) a region costs
about as much as the whole frame, because loading the scene, not the pixels, is most of a render. It is kept for
big, high-sample renders and is not used by the queue's jobs.

Since 2026-11-19, final renders of the 3D look use Cycles' adaptive sampling (`--adaptive 0.01`, set by
`RQ_ADAPTIVE`, where 0 turns it off). Each pixel stops sampling once its noise is under the threshold. On a
sleeves final (two angles, 600 px, 48 samples, denoised) the renders took 191 s instead of 213 s, and no pixel
moved by more than 1 level in 255. Check renders are already 8 samples and denoised, so they are unchanged.

**Where a check job's time goes** (the first telemetry, 2026-11-19). A mannequin check job of about 140 s spends
about 85 s building the bare body (`sorceress_pendant.build`: the anatomy, the skin and the head), 30 to 40 s
building the garment, and under 20 s rendering. Blender's start and loading the scripts take 2 to 10 s. So a warm
Blender process saves only those few seconds. The large saving is the body: it is the same for every garment
on one main, and caching it would cut a check job by about 60%. Forking a Blender process after the body is
built deadlocks in the child (tested: the garment's build hangs in a thread wait). Caching the body needs the
scene saved after the body build, plus the base modules' Python state restored. That is the next speed-up to
build.

## Hogging warnings

The queue warns, and never blocks or reorders, when one thread takes more than its share. `rq.py add` prints an
`rq warning:` line to the thread that sent the request and logs a `hog` event; the dashboard shows the warnings in
force under "Queue hogging"; and the nightly stats (`rq_stats.py`) list each thread's share of the day's render time
and every warning it would have raised, replayed every 15 minutes. `rq.py hogs` prints the warnings in force now.

The rules (`HOG` in `rq.py`), picked from the telemetry of 19 November, when the boots and the sleeves each sent the
same view eight times in an afternoon:

| Rule | Fires when |
|------|------------|
| `hour_share` | a thread used over 50% of the last hour's render time (and at least 10 minutes) while another thread waited 2 minutes or more for a slot |
| `day_share` | the same over the last 24 hours, at 35% and at least 30 minutes |
| `repeat` | a thread asked for the same views 4 times in 3 hours (superseded requests don't count) |
| `streak` | a thread sent 5 requests in a row while another thread had one waiting or rendering (the parts of one set, sent within 15 s, count as one) |

Render time is split among the requests a job served, so a shared set build is not charged twice. Being alone on the
queue is never hogging: the share rules need a second thread that actually waited.

## Time limits

John's rule (2026-11-17): a job running past **10 minutes** is reported to John and to the requesting thread as it
crosses the line, and a job running past **30 minutes** is killed and comes back as failed, with its log ending
`[killed: over the 30-minute limit]`. The limits are per job, on this machine and on John's desktop alike. If
your job is that slow, split it (one view first, the mannequin, check quality, fewer frames) rather than asking
again unchanged.


**Size a final turn to fit** (nightly, 2026-11-19). A final job's time is its image count times about 1.5 to 2
minutes at 600 px and 48 samples with `--denoise`, plus about 3.5 minutes of setup. A turn is five images and `--head`
adds five more, so a turn with a head is ten images and runs about 18 minutes (#188, 1110 s). At 720 px and 128 samples
an image takes 5 to 6 minutes, and that turn was killed at 30 minutes (#187). To stay under the 10-minute warning, send
the turn and the head as separate jobs, or split the turn (`--angles front,3/4,profile` and `--angles back3/4,back`).
Check a navy or near-black dye on a swatch first: three rejections were a navy that read slate blue under the figure's
light.

## John's desktop (GPU worker)

John's PC (RTX 4070 Super, Ryzen 3800X, 16 GB) can take **light jobs only**: mannequin and head views, which peak
at about 2 GB. Measured 2026-11-18 on the same mannequin 3/4 head view: 84 and 89 s there against 72 s in the
cloud, because the figure build runs on the CPU and is slower there (57 s against 31 s); the GPU render took
about 25 s, the same as the cloud's CPU for a small view. A full warden build needs 7 GB or more and was killed
twice by the memory stop, so full-figure jobs stay in the cloud. The GPU would pay off only for big or
high-sample renders, such as the nightly sheet.

It renders in its own clone (never John's working clone), on the GPU with OptiX, and pushes the output to the
`render-results` branch (images only, never merged). Its idle rule: no game running and 9 GB free, which John may
waive for a job he asks for himself; a 1.5 GB memory stop, the 10-minute warning and the 30-minute kill always
apply. A busy or failed job there simply runs in the cloud. The head memo is only saved by a build that exits
normally, so it fills on the first full-figure job that finishes there.

## What comes back

**Your result is not sent to you as a message.** When a request finishes, it appears in `rq.py ping --thread <your id>`
under "last result", with the result folder `/mnt/project-files/render-queue/results/<key>/`: the images,
`render.log`, and `manifest.json` (the commit rendered, every branch combined into it, file sizes and git blob
ids). Ping, then open the images yourself. The render-queue thread messages you only when something needs a
decision: a failed or killed job, a merge conflict, a hogging warning, or a request it had to refuse. Identical code
is never rendered twice: an unchanged request is answered from the cache at once.

## Conflicts

Every request is merged with the current `main` first (generated files never count: they are re-rendered).

* A code conflict with `main` is resolved here when the intent of both sides is clear. You get a git bundle under
  `resolutions/` and the command to fast-forward your branch onto it. Review the merge before pushing it.
* When the right answer depends on what you meant, you get the files and hunks and are asked to rebase.
* The queue also catches what git does not: a function defined twice with different bodies after a clean merge
  (the later one silently replaces the earlier for every caller), and two branches editing the same function
  (they are rendered in separate batches, never combined).

## Rendered files in pull requests

Feature branches carry code only. Keep `blender/ranger/ranger-blender.png`, `blender/ranger/ranger.blend` and
`public/models/ranger.glb` at main's version (`git checkout origin/main -- <those files>`), so a merge never
conflicts on binaries and no one re-renders after a rebase. The daily run refreshes them on `main` through a
pull request from the render-queue thread, which John merges.
