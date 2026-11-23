#!/usr/bin/env python3
"""Before and after on a fixed camera, from results the queue already has. Nothing here renders.

    python3 tools/render-queue/rq_compare.py AFTER [--before DIR] [--image NAME]... [--out DIR]

AFTER is a result folder (results/<key>/, or its key alone). The before is, unless --before names one, the newest
finished result of the exact same view command (so the same preset, flags and camera) at another commit,
preferring a commit already on main: the queue's own rule is that a before is never rendered, only taken from the
cache. When views.json is in both folders (blender/base/views.py writes it), each image's camera is compared and a
pair whose cameras differ is refused, so a before/after can never mix framings.

For every image in both folders, OUT/<name> holds the before above the after, each with its tier label, and a
third row marking in red every pixel that changed; OUT/compare.json lists the changed share of each angle.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

DATA = os.environ.get("RQ_HOME", "/mnt/project-files/render-queue")
RESULTS = os.path.join(DATA, "results")
REPO = os.environ.get("RQ_REPO", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
SKIP = ("cell-",)          # (sprite cells are compared as cells.png)
THRESH = 6                 # a channel must change by more than this (of 255) to count: denoiser and sampling noise


def load(d):
    def j(name):
        p = os.path.join(d, name)
        return json.load(open(p)) if os.path.exists(p) else None
    return j("manifest.json") or {}, j("views.json")


def on_main(c):
    if not c:
        return False
    return subprocess.run(["git", "-C", REPO, "merge-base", "--is-ancestor", c, "origin/main"],
                          capture_output=True).returncode == 0


def find_before(after_dir, man):
    cands = []
    for mp in glob.glob(os.path.join(RESULTS, "*", "manifest.json")):
        d = os.path.dirname(mp)
        if os.path.abspath(d) == os.path.abspath(after_dir):
            continue
        try:
            m = json.load(open(mp))
        except (OSError, ValueError):
            continue
        if m.get("ok") and m.get("param") == man.get("param") and m.get("commit") != man.get("commit"):
            cands.append((on_main(m.get("commit")), m.get("finished", ""), d))
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][2]


def strip(img, rec, name):
    """The image without its label strip, and the strip (None for images from before labels existed)."""
    px = 0
    for im in (rec or {}).get("images", []):
        if im["file"] == name:
            px = im.get("label_px", 0)
    a = np.array(img.convert("RGB"))
    return (a[:-px], a[-px:]) if px else (a, None)


def camera(rec, name):
    for im in (rec or {}).get("images", []):
        if im["file"] == name:
            return im.get("camera"), len(im.get("angles") or []) or 1
    return None, 1


def banner(text, w, bg=(40, 40, 44)):
    from PIL import ImageDraw, ImageFont
    h = max(16, w // 70)
    try:
        font = ImageFont.load_default(size=int(h * 0.72))
    except TypeError:
        font = ImageFont.load_default()
    im = Image.new("RGB", (w, h), bg)
    ImageDraw.Draw(im).text((6, h * 0.12), text, fill=(235, 235, 235), font=font)
    return np.array(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("after")
    ap.add_argument("--before")
    ap.add_argument("--image", action="append", help="only these image names (default: every image in both)")
    ap.add_argument("--out")
    a = ap.parse_args()
    after = a.after if os.path.isdir(a.after) else os.path.join(RESULTS, a.after)
    am, av = load(after)
    before = a.before if a.before is None or os.path.isdir(a.before) else os.path.join(RESULTS, a.before)
    before = before or find_before(after, am)
    if not before:
        sys.exit(f"no cached before: no other finished result of {am.get('param')!r}. A before is never rendered; "
                 "this result becomes the before for the next change of the same view.")
    bm, bv = load(before)
    if am.get("param") and bm.get("param") and am["param"] != bm["param"]:
        print(f"warning: different view commands:\n  before {bm['param']}\n  after  {am['param']}", file=sys.stderr)
    out = a.out or os.path.join(os.getcwd(), f"before-after-{os.path.basename(os.path.normpath(after))}")
    os.makedirs(out, exist_ok=True)
    names = sorted(set(os.listdir(after)) & set(os.listdir(before)))
    names = [n for n in names if n.endswith(".png") and not n.startswith(SKIP) and (not a.image or n in a.image)]
    report = {"before": before, "before_commit": bm.get("commit"), "before_on_main": on_main(bm.get("commit")),
              "after": after, "after_commit": am.get("commit"), "param": am.get("param"), "images": {}}
    for n in names:
        ca, tiles = camera(av, n)
        cb, _ = camera(bv, n)
        if ca is not None and cb is not None and ca != cb:
            report["images"][n] = {"refused": "the cameras differ", "before": cb, "after": ca}
            print(f"{n}: refused, the cameras differ ({cb} against {ca})", file=sys.stderr)
            continue
        if ca is None or cb is None:
            report["images"].setdefault(n, {})["warning"] = "no camera record on one side: same command assumed"
        b, bl = strip(Image.open(os.path.join(before, n)), bv, n)
        f, fl = strip(Image.open(os.path.join(after, n)), av, n)
        if b.shape != f.shape:
            report["images"][n] = {"refused": f"sizes differ ({b.shape[1]}x{b.shape[0]} against {f.shape[1]}x{f.shape[0]})"}
            continue
        changed = (np.abs(b.astype(int) - f.astype(int)) > THRESH).any(axis=2)
        mark = (f * 0.35).astype(np.uint8)
        mark[changed] = (230, 40, 40)
        w = f.shape[1]
        tw = w // tiles
        share = [round(float(changed[:, i * tw:(i + 1) * tw].mean()), 4) for i in range(tiles)]
        bc, fc = (bm.get("commit") or "?")[:9], (am.get("commit") or "?")[:9]
        rows = [banner(f"BEFORE  {bc}" + ("  (main)" if report["before_on_main"] else ""), w), b]
        rows += [bl] if bl is not None else []
        rows += [banner(f"AFTER  {fc}", w), f] + ([fl] if fl is not None else [])
        rows += [banner("CHANGED  " + "  ".join(f"{s * 100:.1f}%" for s in share), w, (90, 20, 20)), mark]
        Image.fromarray(np.concatenate(rows, axis=0)).save(os.path.join(out, n))
        report["images"].setdefault(n, {}).update({"changed_share": share, "out": os.path.join(out, n)})
        print(f"{n}: changed " + ", ".join(f"{s * 100:.1f}%" for s in share))
    with open(os.path.join(out, "compare.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("before", before, "\nwrote", out)


if __name__ == "__main__":
    main()
