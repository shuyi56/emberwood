"""Quick sprite preview for the render queue: renders chosen cells of the ranger sheet, not the whole sheet.

Run from a worktree root:  python3 rq_frames.py --out DIR --cells ROWS:COLS
ROWS and COLS are comma lists or ranges, e.g. 0-7:2,4,6 (all 8 facings at walk columns 2, 4 and 6) or 0,2:0.
Writes DIR/cells.png (the chosen cells laid out as a small sheet, rows x cols in the order given, scaled 3x
nearest-neighbour) and DIR/cell-rR-cC.png for each cell at its real size.
"""
import math, os, sys, tempfile

sys.path.insert(0, os.path.join(os.getcwd(), "blender", "ranger"))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
import ranger as R  # noqa: E402
from PIL import Image  # noqa: E402


def span(s, n):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += range(int(a), int(b) + 1)
        else:
            out.append(int(part))
    return [x for x in out if 0 <= x < n]


def main():
    argv = sys.argv[1:]
    out = argv[argv.index("--out") + 1]
    rows_s, cols_s = argv[argv.index("--cells") + 1].split(":")
    rows, cols = span(rows_s, R.ROWS), span(cols_s, R.COLS)
    os.makedirs(out, exist_ok=True)
    R.build()
    sc = bpy.context.scene
    sc.render.use_persistent_data = True   # only the first cell syncs the whole scene (as in ranger.py)
    root = bpy.data.objects["Ranger"]
    R.sprite_mode(True)
    grid = np.zeros((R.PH * len(rows), R.PW * len(cols), 4), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        for j, col in enumerate(cols):
            sc.frame_set(col + 1)
            for i, row in enumerate(rows):
                root.rotation_euler = (0, 0, row * math.pi / 4)
                p = os.path.join(tmp, f"f{row}_{col}.exr")
                sc.render.filepath = p
                bpy.ops.render.render(write_still=True)
                px = R.to_pixels(R.read_exr(p))
                grid[i * R.PH:(i + 1) * R.PH, j * R.PW:(j + 1) * R.PW] = px
                Image.fromarray(px, "RGBA").save(os.path.join(out, f"cell-r{row}-c{col}.png"))
                print(f"rendered row {row} col {col}", flush=True)
    img = Image.fromarray(grid, "RGBA")
    img.resize((img.width * 3, img.height * 3), Image.NEAREST).save(os.path.join(out, "cells.png"))


main()
