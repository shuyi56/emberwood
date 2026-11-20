# Inspection and stopping

- Before any render, inspect contours: `blender/base/contours.py` (level cuts, front and side profiles).
- For cloth feel, run `blender/base/stress.py`: strain against the UV flat (2nd, 50th, 98th percentile) and the bend
  angle per edge (median, 98th percentile, share past 30 degrees). Paper-like cloth shows as sharp folds and crushed threads.
- Sleeves: `SLEEVE_STRAIN=path` writes a strain map on the flat pattern.
- An outline that is perfectly round, stepped or kinked has a cause. Fix the cause, then render.
- Render on grey clay, not skin.
- Stop iterating when two rounds in a row improve only slightly, when a round makes things worse, when the same flaw
  returns, or at five rounds. Render the best clay draft with its contour sheet and report each round, the numbers,
  the likely cause (tuning, modelling or solver), and the next idea. John decides from there.
