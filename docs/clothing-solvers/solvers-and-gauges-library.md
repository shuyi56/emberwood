# Solvers and gauges library: how it was gathered, and lessons

The project's solvers and gauges had grown up beside the garments that needed them (`blender/shared`,
`blender/zipboot`, `blender/collar`, `blender/heel`, `blender/warden`, `tools/heel`, `tools/contours`,
`tools/render-review`, `tools/face-creator`) and on 28 open draft PRs. A new thread could not tell what existed, so
checks were rewritten or skipped. They are now in one place, `tools/solvers/` and `tools/gauges/`, grouped by domain,
with `tools/INDEX.md` as the catalogue and decision guide, and CLAUDE.md tells every thread to start there.

## What was done

- 19 files moved from main into `tools/solvers/{folds,footwear/heel,tests}` and
  `tools/gauges/{contours,fit,footwear,tailoring,color,hair}`, in a commit of pure renames (git sees each as 100 %
  similar, so `git log --follow` keeps the history), then a second commit with the shims and the path fixes.
- A shim at every old path. It loads the real file with `importlib.util.spec_from_file_location` and
  `module_from_spec` and puts it in `sys.modules` under the old name before running it (the "importing a source file
  directly" recipe in the Python `importlib` documentation). So `import folds` gets the one real module with all its
  state, `python3 blender/collar/basecheck.py ...` runs the real file's `__main__` block, and pickles keep their
  module names.
- Path fixes only where a file found its neighbours relative to its own folder: the leather gauge (`blender/<fig>`),
  the colour check and the boot dump (`blender/`), the heel tools (the support gauge), the ease spec (it stays beside
  the outfit in `blender/shared/riding_habit.json`), the solver tests (`fabric.py`). Sibling imports for callers who
  come in by the new path are appended to `sys.path`, so an old path already on it still wins, as before.

## What was left in place, and why

- **Files edited by an open PR**: `blender/shared/fabric.py` (PR #85), `blender/chemise/solver.py` (PR #91),
  `tools/render-review/review.py` and `smoke.py` (PR #27). With a shim left at the old path git sees a modify/modify
  conflict, not a rename, so moving them now would break each of those PRs' merges. They move when their PR lands.
- **Tools that live only on open PRs** (refmeasure, joins, draft topology, lace tension, lace study, hair foundations,
  flow check, face checks and others): catalogued with their PR and path; they come into the library when merged.
- **Pipeline code** (`tools/render-queue/`, `tools/render-review/`): it runs the gauges rather than being one, most
  open PRs edit it, and the queue's live copy runs from `/mnt/project-files/render-queue/`.
- **Garment-specific drivers** (Blender cloth per garment, `blender/boot/last.py`, the chain drop in
  `accessories.py`): each is its garment's own settings, catalogued and linked from the index.

## Numbers that mattered

- `folds.py` hashes its own source into the fold cache key. Moved byte for byte, the key for a test sheet was the
  same before and after (`af61cac112f91c4473461d8c`), the bake the same (90.0 of 90.0 degrees in 80 steps), and the
  contour sheet's PNG and JSON identical. One edited line in that file would have re-solved every approved fold.
- The three solver tests print byte-identical output on main and on the branch, through the old paths and the new.
- The heel optimizer, stylizer and sheet, the colour check, and the leather gauge were run on main and on the branch
  on the same build and compared (see the PR for the results).

## Lessons

1. Before moving shared code, list every open PR's edits to it (`git diff --name-only origin/main...<branch>`). A
   shim at the old path turns their merges into conflicts; a file an open PR edits waits for that PR.
2. Look for caches keyed on source before touching a file. A cache key that hashes code makes a "no-change" move a
   behaviour change if a single byte differs.
3. Commit pure renames apart from edits, so history follows the move.
4. Check a move by running the tools, not by reading imports: the same input through the old path, the new path and
   main, outputs compared byte for byte.
5. New solvers and gauges start in the library with the entry shape (purpose, in and out, run, trouble, when,
   sources), so the next thread can find them.
