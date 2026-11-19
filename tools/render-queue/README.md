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
| `rq_jit.py`, `rq_hot.py` | region rendering and build reuse: kept as options, not used by the queue's jobs |

A view job is any script in the requesting branch, run through `rq_run.py` with the request's arguments plus
`--out DIR`. The standard figure views are `blender/base/views.py` (`--figure`, `--mannequin`, `--turn`/`--head`,
`--angles`, `--clay`, `--walk N`/`--run N`, `--size`, `--samples`, `--denoise`).

## How to ask for a render

Push your commit, then send the **reviewer thread**, "Review render requests before queueing"
(cmsg_01CnnKqGvuWEUi2G9X9cb9nBJkqDv7Ktra4a31euzvMxpx), a message with `message_thread` holding one JSON block.
Do not send requests straight to the render-queue thread any more. The reviewer reads every request first (see
"Review before the queue" below) and passes it on to the queue, or sends it back to you with fixes:

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
different functions of the scripts. One render then covers all of them, and the reply says what else was in it.
`mode: isolated` is still accepted but ignored. Up to three quick jobs render side by side.

## Review before the queue

The reviewer thread checks each request without rendering anything, in about two minutes:

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

## Time limits

John's rule (2026-11-17): a job running past **10 minutes** is reported to John and to the requesting thread as it
crosses the line, and a job running past **30 minutes** is killed and comes back as failed, with its log ending
`[killed: over the 30-minute limit]`. The limits are per job, on this machine and on John's desktop alike. If
your job is that slow, split it (one view first, the mannequin, check quality, fewer frames) rather than asking
again unchanged.


**Size a final turn to fit** (nightly, 2026-11-19). A five-angle final turn at 720 px and 128 samples was killed at 30
minutes (#187, 5 to 6 minutes an angle), and one at 800 px took 10 minutes (#178). With `--denoise`, 600 px and 48
samples look the same at review size and take 8 to 10 minutes for the whole turn. Go past that only as two jobs
(`--angles front,3/4,profile` and `--angles back3/4,back`). Check a navy or near-black dye on a swatch first: three
rejections were a navy that read slate blue under the figure's light.

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

A reply in your thread with the result folder `/mnt/project-files/render-queue/results/<key>/`: the images,
`render.log`, and `manifest.json` (the commit rendered, every branch combined into it, file sizes and git blob
ids). Identical code is never rendered twice: an unchanged request is answered from the cache at once.

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
