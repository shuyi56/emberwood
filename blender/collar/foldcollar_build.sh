#!/bin/bash
# foldcollar_build.sh LAPEL.blend OUTDIR ['{foldcollar options}']: the folded collar on a foldfront lapel, in one go.
#   OUTDIR/out.blend            the coat with the folded collar ('Redingote collar' replaced)
#   OUTDIR/collar.npz(.json)    the collar: pts, tri, uv (the shaped cut), uv_rect (the rectangle), strain
#   OUTDIR/join.json            the join gauge (blender/tension/join.py) on it
# notchcollar is run first only for what foldcollar reads from it: the neckline and the stand's up along it.
# Experimental (phases 9 and 10 of docs/clothing-solvers/lapel-tension-solver.md); the options are the photo fit.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; T="$HERE/../tension"; L="$1"; O="$2"; mkdir -p "$O"
FIT='{"width":2.05,"stand":0.7,"fold_angle":21,"open":5,"corner":0.5,"seam_from":-16,"fold_k":3,"push_even":600,"push_grow":20,"lift_run":0.3,"shape_cut":1,"bone":1.0,"lift":0.19,"clear":0.27,"push_rounds":3,"gorge_flush":1.6,"gorge_level":0.12}'
for s in "" .lapels.npz .strain.npz; do cp "$L$s" "$O/lapel.blend$s"; done
python3 -I "$HERE/notchcollar.py" "$O/lapel.blend" "$O/out.blend" '{}' > "$O/notchcollar.log" 2>&1
python3 -I "$T/dump_build.py" "$O/out.blend" "$O/build.npz" > /dev/null 2>&1
python3 "$HERE/foldcollar.py" "$O" "$O/collar.npz" "${3:-$FIT}"
python3 -I "$HERE/foldcollar_blend.py" "$O/lapel.blend" "$O/collar.npz" "$O/out.blend" > "$O/blend.log" 2>&1
cp "$O/out.blend.collar.npz" "$O/notchcollar.collar.npz"; cp "$O/collar.npz" "$O/out.blend.collar.npz"
python3 "$T/foldcollar_gauge.py" "$O" "$O/collar.npz" "$O/collar_gauge.json"
python3 "$T/join.py" "$O" "$O/join.json" > /dev/null 2>&1 && echo "foldcollar_build: $O/out.blend, join gauge in $O/join.json"
