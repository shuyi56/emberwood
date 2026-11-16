# Render request reviewer

Every render request is read here before it reaches the render queue. A request that would waste render time
is sent back to its thread with concrete fixes; the rest are forwarded to the queue unchanged. Nothing in this
folder renders.

| File | What it is |
| --- | --- |
| `review.py` | The checks. `check` reviews one request, `learn` adds a pitfall, `pitfalls` lists them. |
| `smoke.py` | Runs a render script for real, headless, and stops it the moment its first render would start. |
| `PITFALLS.md` | What has gone wrong before and how to fix it. Read on every review, added to after every rejection or bad render. |

The queue itself (`rq.py`, `queue.json`, results) lives in the project folder under `render-queue/` and belongs to
the render-queue thread. `review.py` imports `rq.py` for merges, cache keys and duplicate checks, so the two always
agree. It reads the queue's state and never writes it.

## For threads: how to ask for a render

Push your commit, then send the request JSON described in `render-queue/README.md` to the reviewer thread
("Review render requests before queueing") with `message_thread`. You get one of two replies:

* **Forwarded.** The request went to the queue as it was; any warnings are in the reply. Results come from the
  render-queue thread as before.
* **Sent back.** Each finding says what is wrong, where, and the fix. Push the fix and send the request again.
* **Face check.** A request that only checks a face is never rendered. The reviewer runs the face check API on
  your commit and replies with the folder of views and measurements. See "Face checks" below.

Check your own branch first, without asking anyone:

    python3 tools/render-review/smoke.py blender/base/views.py --figure rogue --head --clay --out /tmp/x

It needs Blender's Python module (`pip install bpy pyflakes pillow`), exits 77 when the script reaches its first
render and prints the traceback when it does not. It renders nothing.

## Face checks

Every figure face check goes through `blender/facecheck/facecheck.py`, not through a render:

    python blender/facecheck/facecheck.py <figure> --before origin/main

It runs the figure's own face code without Blender (about 17 s), draws front, 3/4, profile, eyes, lips and the
game-scale view beside main's, and lists every measurement that changed. A before is free there. Run it yourself
instead of sending a request.

The reviewer treats a request as a face check when the code changed since the branch was last rendered is only
face sculpt or face paint (`face_relief`, `mouth_relief`, `jaw_relief`, lids, `eye_relief`, iris, sclera,
eyeshadow, lashes, brows, lips) and the request asks only for face views (`face_views.py`, `sculpt_views.py`, or
`views.py --head` without `--cells` or `--turn`). It then runs the API on the commit merged with main, keeps the
result under `render-queue/facechecks/<figure>-<commit>/` and answers with that; nothing goes to the queue.

When the change needs what the API cannot show (hair, cloth, final lighting, subsurface scattering, the sprite
palette), add `"needs_render": "<what you need to see>"` to the request and it is forwarded as usual.

## For the reviewer: each request

1. `python3 tools/render-review/review.py check --thread <requester id> --request '<json>'`
2. Read the diff it points to (the change since that branch was last rendered) with the pitfalls list beside
   it. Look for what no script sees: a view that fakes a before, a hair piece with no joining shape, a
   request that cannot show the change it describes.
3. Verdict `pass` and nothing found in the diff: send the printed `forward` JSON to the render-queue thread, and
   tell the requester in one line, with any warnings.
4. Verdict `facecheck`: do not forward. Look at `sheet.png` and the changed measurements, and send the requester
   the folder with what you saw.
5. Verdict `fix`, or the diff shows a problem: send the findings and fixes to the requester. Do not forward.
6. After a rejection for a new reason, or when a forwarded render fails or comes out wrong, add the lesson with
   `review.py learn ...`, commit `PITFALLS.md` and push.

What blocks: the full sheet in any form, an uncached before, a branch that is not pushed or conflicts with main,
a syntax error, a new duplicate definition, a view script or flag that does not exist, shell characters in a
view command, a build that crashes in the smoke run, and code that already failed. Everything else is a warning
and goes through.

The queue's rules are unchanged: threads never render, quick jobs go first, the full sheet runs only as the
scheduled task, a before is never rendered, branches are combined whenever their code does not overlap, and up to
three quick jobs run side by side.
