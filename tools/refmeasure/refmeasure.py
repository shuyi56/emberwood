#!/usr/bin/env python3
"""Comparative measurement of a reference image, and of a render against it.

The method is the atelier one (see README.md for sources):
  * pick ONE unit (head height, shoulder width, ...) and express every size as a multiple of it;
    never chain one measurement off another, because small errors grow;
  * prefer plumb (vertical) and level (horizontal) offsets to angles, they are read more accurately;
  * compare size, position and angle; and
  * only compare two images taken from the same viewing angle (foreshortening changes every ratio).

A measurement sheet is a JSON file next to the image (``photo.jpg`` -> ``photo.measure.json``):

    {
      "image": "photo.jpg", "size": [w, h],
      "camera": {"view": "front 3/4 left", "projection": "persp", "note": "..."},
      "unit":   {"name": "head", "a": "head_top", "b": "chin", "axis": "y"},
      "plumb":  "neck_front",                      # origin and plumb line for positions
      "landmarks": {"head_top": [x, y], ...},      # image pixels, y down
      "segments":  {"shoulder_w": {"a": "shoulder_l", "b": "shoulder_r", "axis": "x"}, ...},
      "ratios":    {"lapel_to_shoulder": ["lapel_w", "shoulder_w"], ...},
      "computed":  {...}                           # written by `report`, never edit by hand
    }

``axis`` is "len" (straight-line distance, the default), "x" (horizontal span only) or "y" (vertical
span only). Run ``refmeasure.py -h`` for the commands. Every command works headless.
"""
import argparse
import json
import math
import os
import sys

VERSION = 1

# Suggested landmark names, so photo and model sheets use the same ones. Add to it, don't rename.
LANDMARKS = {
    "head": ["head_top", "brow", "nose_base", "chin", "ear_top", "ear_bottom"],
    "neck": ["neck_l", "neck_r", "neck_front", "neck_back", "jugular_notch", "c7"],
    "torso": ["shoulder_l", "shoulder_r", "armpit_l", "armpit_r", "bust_l", "bust_r",
              "waist_l", "waist_r", "waist_front", "hip_l", "hip_r", "crotch"],
    "arm": ["elbow_l", "elbow_r", "wrist_l", "wrist_r"],
    "leg": ["knee_l", "knee_r", "ankle_l", "ankle_r", "heel_l", "toe_l", "floor"],
    "coat": ["collar_top", "collar_point_l", "collar_point_r", "gorge_l", "gorge_r",
             "notch_l", "notch_r", "lapel_point_l", "lapel_point_r", "lapel_outer_l",
             "lapel_outer_r", "roll_top_l", "roll_top_r", "break_point", "top_button",
             "bottom_button", "hem_front", "hem_side_l", "hem_side_r", "sleeve_cap_l",
             "sleeve_cap_r", "cuff_l", "cuff_r"],
}
AXES = ("len", "x", "y")


# ----------------------------------------------------------------------------- sheet i/o

def sheet_path(path):
    """photo.jpg -> photo.measure.json; a .json path is returned as is."""
    if path.endswith(".json"):
        return path
    return os.path.splitext(path)[0] + ".measure.json"


def load(path):
    p = sheet_path(path)
    with open(p) as f:
        s = json.load(f)
    s.setdefault("landmarks", {})
    s.setdefault("segments", {})
    s.setdefault("ratios", {})
    s["_path"] = p
    return s


def save(s):
    p = s.pop("_path")
    out = {k: v for k, v in s.items() if not k.startswith("_")}
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    s["_path"] = p
    return p


def image_of(s):
    """Absolute path of the sheet's image (stored relative to the sheet)."""
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(s["_path"])), s["image"]))


def new_sheet(image, view="", projection="persp", note=""):
    from PIL import Image
    w, h = Image.open(image).size
    p = sheet_path(image)
    s = {"version": VERSION, "image": os.path.relpath(os.path.abspath(image),
                                                     os.path.dirname(os.path.abspath(p))),
         "size": [w, h],
         "camera": {"view": view, "projection": projection, "note": note},
         "unit": None, "plumb": None, "landmarks": {}, "segments": {}, "ratios": {}}
    if os.path.exists(p):
        old = load(p)
        old["size"] = [w, h]
        return old
    s["_path"] = p
    return s


# ----------------------------------------------------------------------------- geometry

def _pt(s, name):
    try:
        return s["landmarks"][name]
    except KeyError:
        raise SystemExit(f"no landmark '{name}' in {s['_path']} (have: {', '.join(sorted(s['landmarks']))})")


def span(s, a, b, axis="len"):
    (x0, y0), (x1, y1) = _pt(s, a), _pt(s, b)
    if axis == "x":
        return abs(x1 - x0)
    if axis == "y":
        return abs(y1 - y0)
    return math.hypot(x1 - x0, y1 - y0)


def seg_spec(v):
    """Segments may be written ["a", "b"], ["a", "b", "x"] or {"a":..,"b":..,"axis":..}."""
    if isinstance(v, dict):
        return v["a"], v["b"], v.get("axis", "len")
    return v[0], v[1], (v[2] if len(v) > 2 else "len")


def unit_px(s):
    u = s.get("unit")
    if not u:
        raise SystemExit("no unit set: run `refmeasure.py unit SHEET A B --name head --axis y`")
    px = span(s, u["a"], u["b"], u.get("axis", "len"))
    if px <= 0:
        raise SystemExit("unit has zero length")
    return px


def angle_from_vertical(s, a, b):
    """Degrees the line a->b leans from plumb; + leans right (image x grows) going down."""
    (x0, y0), (x1, y1) = _pt(s, a), _pt(s, b)
    if y1 < y0:
        x0, y0, x1, y1 = x1, y1, x0, y0
    return math.degrees(math.atan2(x1 - x0, y1 - y0))


def compute(s):
    """Everything as multiples of the unit. Sizes, positions and angles, as Keller sorts them."""
    u = unit_px(s)
    origin = s.get("plumb") or s["unit"]["a"]
    ox, oy = _pt(s, origin)
    c = {"unit_px": round(u, 2), "unit": s["unit"], "origin": origin,
         "segments": {}, "ratios": {}, "positions": {}}
    for name, v in s["segments"].items():
        a, b, axis = seg_spec(v)
        d = span(s, a, b, axis)
        e = {"units": round(d / u, 4), "px": round(d, 1), "axis": axis}
        if axis == "len":
            e["deg_from_vertical"] = round(angle_from_vertical(s, a, b), 2)
        c["segments"][name] = e
    for name, (p, q) in s["ratios"].items():
        dp = c["segments"].get(p, {}).get("px")
        dq = c["segments"].get(q, {}).get("px")
        if dp is None or dq is None:
            c["ratios"][name] = {"error": f"needs segments {p} and {q}"}
            continue
        c["ratios"][name] = {"value": round(dp / dq, 4), "of": [p, q]}
    for name, (x, y) in s["landmarks"].items():
        # + right of plumb, + below origin, in units
        c["positions"][name] = {"x": round((x - ox) / u, 4), "y": round((y - oy) / u, 4)}
    lm = s["landmarks"]
    if lm:
        ys = [p[1] for p in lm.values()]
        c["whole_height_units"] = round((max(ys) - min(ys)) / u, 3)
    s["computed"] = c
    return c


# ----------------------------------------------------------------------------- printing

def print_report(s, c):
    cam = s.get("camera") or {}
    print(f"# {s['image']}  ({cam.get('view','?')}, {cam.get('projection','?')})")
    u = c["unit"]
    print(f"unit = {u.get('name','unit')} = {u['a']}->{u['b']} ({u.get('axis','len')}) = {c['unit_px']} px;"
          f"  origin/plumb = {c['origin']};  landmarks span {c.get('whole_height_units','?')} units tall")
    if c["segments"]:
        print("\nSIZES (units)")
        for n, e in sorted(c["segments"].items()):
            ang = f"  {e['deg_from_vertical']:+6.1f} deg from plumb" if "deg_from_vertical" in e else ""
            print(f"  {n:<24} {e['units']:7.3f}  [{e['axis']:>3}]{ang}")
    if c["ratios"]:
        print("\nRATIOS")
        for n, e in sorted(c["ratios"].items()):
            if "error" in e:
                print(f"  {n:<24} -- {e['error']}")
            else:
                print(f"  {n:<24} {e['value']:7.3f}  = {e['of'][0]} / {e['of'][1]}")
    print("\nPOSITIONS (units from plumb; x + right, y + down)")
    for n, p in sorted(c["positions"].items(), key=lambda kv: kv[1]["y"]):
        print(f"  {n:<24} x {p['x']:+7.3f}   y {p['y']:+7.3f}")
    # Alignments are what plumb lines are for: pairs that sit nearly on one vertical or level.
    al = alignments(c)
    if al:
        print("\nALIGNMENTS (within 0.05 unit)")
        for kind, a, b, d in al:
            print(f"  {kind:<5} {a} ~ {b}  ({d:+.3f})")


def alignments(c, tol=0.05):
    pos = list(c["positions"].items())
    out = []
    for i, (a, pa) in enumerate(pos):
        for b, pb in pos[i + 1:]:
            if abs(pa["x"] - pb["x"]) <= tol and abs(pa["y"] - pb["y"]) > 0.25:
                out.append(("plumb", a, b, pb["x"] - pa["x"]))
            if abs(pa["y"] - pb["y"]) <= tol and abs(pa["x"] - pb["x"]) > 0.25:
                out.append(("level", a, b, pb["y"] - pa["y"]))
    return out


# ----------------------------------------------------------------------------- overlay

def _font(size):
    from PIL import ImageFont
    for f in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_overlay(s, out, sub=4, levels=True, crop=None, scale=1.0):
    """Unit grid (with `sub` subdivisions) anchored on the plumb landmark, the plumb line, a level line
    through every landmark, segments and labels. `crop` = (x0, y0, x1, y1) in image pixels."""
    from PIL import Image, ImageDraw
    im = Image.open(image_of(s)).convert("RGB")
    W, H = im.size
    x0, y0, x1, y1 = crop or (0, 0, W, H)
    im = im.crop((x0, y0, x1, y1))
    if scale != 1.0:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    T = lambda x, y: ((x - x0) * scale, (y - y0) * scale)
    lw = max(1, round(min(W, H) / 900 * scale))
    fs = max(12, round(min(im.size) / 45))
    fnt = _font(fs)
    have_unit = bool(s.get("unit")) and all(k in s["landmarks"] for k in (s["unit"]["a"], s["unit"]["b"]))
    if have_unit:
        u = unit_px(s)
        org = s.get("plumb") or s["unit"]["a"]
        ox, oy = _pt(s, org)
        step = u / sub
        k0, k1 = math.floor((x0 - ox) / step), math.ceil((x1 - ox) / step)
        for k in range(k0, k1 + 1):
            X, _ = T(ox + k * step, 0)
            major = k % sub == 0
            d.line([(X, 0), (X, im.height)], fill=(0, 190, 255, 150 if major else 55), width=lw)
            if major and k:
                d.text((X + 3, 3), f"{k // sub:+d}", fill=(0, 190, 255, 255), font=fnt)
        k0, k1 = math.floor((y0 - oy) / step), math.ceil((y1 - oy) / step)
        for k in range(k0, k1 + 1):
            _, Y = T(0, oy + k * step)
            major = k % sub == 0
            d.line([(0, Y), (im.width, Y)], fill=(0, 190, 255, 150 if major else 55), width=lw)
            if major and Y > 1.5 * fs:
                d.text((3, Y + 2), f"{k // sub:+d}", fill=(0, 190, 255, 255), font=fnt)
        X, _ = T(ox, 0)
        d.line([(X, 0), (X, im.height)], fill=(255, 40, 40, 230), width=lw * 2)   # the plumb line
    else:
        # No unit yet: a plain pixel grid with labels, so points can be read off the picture.
        # Pick a 1/2/5 x 10^n step whose labels (about 4 glyphs wide) do not collide.
        need = 3.6 * fs / scale
        step = next(m * 10 ** e for e in range(0, 6) for m in (1, 2, 5) if m * 10 ** e >= need)
        for gx in range(int(x0 // step * step), int(x1) + 1, int(step)):
            X, _ = T(gx, 0)
            d.line([(X, 0), (X, im.height)], fill=(0, 190, 255, 90), width=lw)
            d.text((X + 2, 2), str(gx), fill=(0, 190, 255, 255), font=fnt)
        for gy in range(int(y0 // step * step), int(y1) + 1, int(step)):
            _, Y = T(0, gy)
            d.line([(0, Y), (im.width, Y)], fill=(0, 190, 255, 90), width=lw)
            if Y > 1.5 * fs:
                d.text((2, Y + 2), str(gy), fill=(0, 190, 255, 255), font=fnt)
    if levels:
        for n, (x, y) in s["landmarks"].items():
            _, Y = T(x, y)
            d.line([(0, Y), (im.width, Y)], fill=(255, 220, 0, 70), width=lw)
    for n, v in s["segments"].items():
        a, b, axis = seg_spec(v)
        if a not in s["landmarks"] or b not in s["landmarks"]:
            continue
        (ax, ay), (bx, by) = s["landmarks"][a], s["landmarks"][b]
        if axis == "x":
            bx2, by2 = bx, ay
        elif axis == "y":
            bx2, by2 = ax, by
        else:
            bx2, by2 = bx, by
        d.line([T(ax, ay), T(bx2, by2)], fill=(80, 255, 120, 230), width=lw * 2)
        mx, my = T((ax + bx2) / 2, (ay + by2) / 2)
        d.text((mx + 4, my - 4), n, fill=(80, 255, 120, 255), font=fnt)
    r = 3 * lw + 1
    for n, (x, y) in s["landmarks"].items():
        X, Y = T(x, y)
        d.ellipse([X - r, Y - r, X + r, Y + r], outline=(255, 60, 200, 255), width=lw)
        d.line([(X - 2 * r, Y), (X + 2 * r, Y)], fill=(255, 60, 200, 255), width=1)
        d.line([(X, Y - 2 * r), (X, Y + 2 * r)], fill=(255, 60, 200, 255), width=1)
        d.text((X + r + 2, Y - r - 12), n, fill=(255, 255, 255, 255), font=fnt,
               stroke_width=2, stroke_fill=(0, 0, 0, 255))
    Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB").save(out)
    return out


# ----------------------------------------------------------------------------- compare

def compare(ref, mod, tol=0.05, angle_tol=3.0):
    """Gaps between two computed sheets. Sizes and positions in units, ratios relative, angles deg."""
    cr, cm = compute(ref), compute(mod)
    rows = []
    for n, e in cr["segments"].items():
        m = cm["segments"].get(n)
        if not m:
            rows.append(("size", n, e["units"], None, None, "missing in model"))
            continue
        g = m["units"] - e["units"]
        rows.append(("size", n, e["units"], m["units"], g, "GAP" if abs(g) / max(e["units"], 1e-9) > tol else ""))
        if "deg_from_vertical" in e and "deg_from_vertical" in m:
            ga = m["deg_from_vertical"] - e["deg_from_vertical"]
            rows.append(("angle", n, e["deg_from_vertical"], m["deg_from_vertical"], ga,
                         "GAP" if abs(ga) > angle_tol else ""))
    for n, e in cr["ratios"].items():
        m = cm["ratios"].get(n)
        if "value" not in e:
            continue
        if not m or "value" not in m:
            rows.append(("ratio", n, e["value"], None, None, "missing in model"))
            continue
        g = m["value"] - e["value"]
        rows.append(("ratio", n, e["value"], m["value"], g, "GAP" if abs(g) / max(e["value"], 1e-9) > tol else ""))
    for n, e in cr["positions"].items():
        m = cm["positions"].get(n)
        if not m:
            rows.append(("pos", n, e["y"], None, None, "missing in model"))
            continue
        for ax in ("x", "y"):
            g = m[ax] - e[ax]
            rows.append((f"pos.{ax}", n, e[ax], m[ax], g, "GAP" if abs(g) > tol else ""))
    warn = []
    if cr["unit"]["a"] != cm["unit"]["a"] or cr["unit"]["b"] != cm["unit"]["b"] or \
            cr["unit"].get("axis", "len") != cm["unit"].get("axis", "len"):
        warn.append("the two sheets use different units; ratios against the unit are not comparable")
    if cr["origin"] != cm["origin"]:
        warn.append("the two sheets use different plumb origins; positions are not comparable")
    va, vb = (ref.get("camera") or {}), (mod.get("camera") or {})
    pa, pb = va.get("projection"), vb.get("projection")
    if va.get("view") != vb.get("view") or (pa != pb and "photo" not in (pa, pb)):
        warn.append(f"camera differs (ref {va.get('view')!r}/{pa!r}, model {vb.get('view')!r}/{pb!r}): "
                    "foreshortened widths will not match; render the model from the photo's view")
    return rows, warn


def print_compare(rows, warn, only_gaps=False):
    for w in warn:
        print("WARNING:", w)
    print(f"{'kind':<7} {'name':<24} {'ref':>8} {'model':>8} {'gap':>8} {'gap%':>7}")
    rank = sorted(rows, key=lambda r: -(abs(r[4]) / max(abs(r[2]), 1e-9) if r[4] is not None and not r[0].startswith(("pos", "angle")) else abs(r[4] or 0)))
    for k, n, a, b, g, flag in rank:
        if only_gaps and not flag:
            continue
        if b is None:
            print(f"{k:<7} {n:<24} {a:8.3f} {'-':>8} {'-':>8} {'':>7}  {flag}")
            continue
        pct = "" if k.startswith(("pos", "angle")) else f"{100 * g / max(abs(a), 1e-9):+6.1f}%"
        print(f"{k:<7} {n:<24} {a:8.3f} {b:8.3f} {g:+8.3f} {pct:>7}  {flag}")
    n = sum(1 for r in rows if r[5])
    print(f"\n{n} of {len(rows)} measurements outside tolerance")


def side_by_side(ref, mod, out, unit_px_out=300, margin=0.4):
    """Both overlays scaled so one unit is `unit_px_out` pixels, cropped to the same window in units around
    the plumb origin (union of both sheets' landmarks plus `margin`), plumb lines and origins level."""
    import tempfile
    from PIL import Image
    cs = [compute(s) for s in (ref, mod)]
    xs = [p["x"] for c in cs for p in c["positions"].values()]
    ys = [p["y"] for c in cs for p in c["positions"].values()]
    ux0, ux1, uy0, uy1 = min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin
    panels = []
    for s, c in zip((ref, mod), cs):
        u = c["unit_px"]
        ox, oy = _pt(s, c["origin"])
        tmp = tempfile.mktemp(suffix=".png")
        draw_overlay(s, tmp, crop=(ox + ux0 * u, oy + uy0 * u, ox + ux1 * u, oy + uy1 * u), scale=unit_px_out / u)
        panels.append(Image.open(tmp).convert("RGB"))
        os.remove(tmp)
    W, H = max(p.width for p in panels), max(p.height for p in panels)
    canvas = Image.new("RGB", (2 * W + 10, H), (255, 255, 255))
    for i, p in enumerate(panels):
        canvas.paste(p, (i * (W + 10), 0))
    canvas.save(out)
    return out


# ----------------------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("new", help="start a sheet next to an image (keeps an existing one)")
    p.add_argument("image"); p.add_argument("--view", default=""); p.add_argument("--projection", default="persp",
                                                                                   choices=["persp", "ortho", "photo"])
    p.add_argument("--note", default="")

    p = sp.add_parser("set", help="place landmarks: set SHEET name x y [name x y ...]")
    p.add_argument("sheet"); p.add_argument("items", nargs="+")

    p = sp.add_parser("rm", help="remove landmarks or segments by name")
    p.add_argument("sheet"); p.add_argument("names", nargs="+")

    p = sp.add_parser("unit", help="set the unit to the span between two landmarks")
    p.add_argument("sheet"); p.add_argument("a"); p.add_argument("b")
    p.add_argument("--name", default="unit"); p.add_argument("--axis", default="len", choices=AXES)

    p = sp.add_parser("plumb", help="set the landmark the plumb line and positions hang from")
    p.add_argument("sheet"); p.add_argument("name")

    p = sp.add_parser("seg", help="name a segment: seg SHEET name a b [--axis x|y|len]")
    p.add_argument("sheet"); p.add_argument("name"); p.add_argument("a"); p.add_argument("b")
    p.add_argument("--axis", default="len", choices=AXES)

    p = sp.add_parser("ratio", help="name a ratio of two segments: ratio SHEET name seg_p seg_q")
    p.add_argument("sheet"); p.add_argument("name"); p.add_argument("p"); p.add_argument("q")

    p = sp.add_parser("report", help="print and save every size, ratio, position and alignment in units")
    p.add_argument("sheet"); p.add_argument("--json", action="store_true")

    p = sp.add_parser("probe", help="quick look: probe SHEET a b [c d] -> span in units, offsets, angle, ratio")
    p.add_argument("sheet"); p.add_argument("pts", nargs="+")

    p = sp.add_parser("overlay", help="draw the unit grid, plumb line, levels, segments and landmarks")
    p.add_argument("sheet"); p.add_argument("--out"); p.add_argument("--sub", type=int, default=4)
    p.add_argument("--crop", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--scale", type=float, default=1.0); p.add_argument("--no-levels", action="store_true")

    p = sp.add_parser("grid", help="pixel-labelled zoom of a region, to read coordinates for `set`")
    p.add_argument("image"); p.add_argument("x0", type=float); p.add_argument("y0", type=float)
    p.add_argument("x1", type=float); p.add_argument("y1", type=float)
    p.add_argument("--out"); p.add_argument("--scale", type=float, default=0)

    p = sp.add_parser("compare", help="gaps between a reference sheet and a model sheet")
    p.add_argument("ref"); p.add_argument("model"); p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--angle-tol", type=float, default=3.0); p.add_argument("--gaps", action="store_true")
    p.add_argument("--json"); p.add_argument("--png", help="side-by-side overlay, both scaled to one unit size")

    p = sp.add_parser("template", help="copy unit, plumb, segments and ratios from one sheet to another")
    p.add_argument("src"); p.add_argument("dst")

    p = sp.add_parser("flip", help="write a left-right mirrored copy of an image (render turned the other way)")
    p.add_argument("image"); p.add_argument("--out")

    sp.add_parser("names", help="list the suggested landmark names")

    a = ap.parse_args(argv)

    if a.cmd == "names":
        for k, v in LANDMARKS.items():
            print(f"{k}: {' '.join(v)}")
        return
    if a.cmd == "flip":
        from PIL import Image, ImageOps
        out = a.out or os.path.splitext(a.image)[0] + ".flipped.png"
        ImageOps.mirror(Image.open(a.image)).save(out)
        print(out)
        return
    if a.cmd == "new":
        s = new_sheet(a.image, a.view, a.projection, a.note)
        print(save(s))
        return
    if a.cmd == "grid":
        s = {"image": os.path.abspath(a.image), "landmarks": {}, "segments": {}, "unit": None,
             "_path": os.path.abspath(a.image)}
        sc = a.scale or max(1.0, 900 / max(1, a.x1 - a.x0))
        out = a.out or os.path.splitext(a.image)[0] + f".grid_{int(a.x0)}_{int(a.y0)}.png"
        if os.path.exists(sheet_path(a.image)):
            s = load(a.image)
            s = dict(s, unit=None)   # pixel labels, but show placed landmarks
        print(draw_overlay(s, out, crop=(a.x0, a.y0, a.x1, a.y1), scale=sc, levels=False))
        return
    if a.cmd == "compare":
        ref, mod = load(a.ref), load(a.model)
        rows, warn = compare(ref, mod, a.tol, a.angle_tol)
        print_compare(rows, warn, a.gaps)
        if a.png:
            print(side_by_side(ref, mod, a.png))
        if a.json:
            with open(a.json, "w") as f:
                json.dump({"ref": ref["_path"], "model": mod["_path"], "warnings": warn,
                           "rows": [dict(zip(("kind", "name", "ref", "model", "gap", "flag"), r)) for r in rows]},
                          f, indent=1)
        return
    if a.cmd == "template":
        src, dst = load(a.src), load(a.dst)
        for k in ("unit", "plumb", "segments", "ratios"):
            dst[k] = src.get(k)
        print(save(dst))
        return

    s = load(a.sheet)
    if a.cmd == "set":
        it = a.items
        if len(it) % 3:
            raise SystemExit("set takes triples: name x y")
        for i in range(0, len(it), 3):
            s["landmarks"][it[i]] = [float(it[i + 1]), float(it[i + 2])]
        save(s)
    elif a.cmd == "rm":
        for n in a.names:
            s["landmarks"].pop(n, None); s["segments"].pop(n, None); s["ratios"].pop(n, None)
        save(s)
    elif a.cmd == "unit":
        s["unit"] = {"name": a.name, "a": a.a, "b": a.b, "axis": a.axis}
        s.setdefault("plumb", None)
        save(s)
    elif a.cmd == "plumb":
        s["plumb"] = a.name
        save(s)
    elif a.cmd == "seg":
        s["segments"][a.name] = {"a": a.a, "b": a.b, "axis": a.axis}
        save(s)
    elif a.cmd == "ratio":
        s["ratios"][a.name] = [a.p, a.q]
        save(s)
    elif a.cmd == "report":
        c = compute(s)
        save(s)
        if a.json:
            print(json.dumps(c, indent=1))
        else:
            print_report(s, c)
    elif a.cmd == "probe":
        probe(s, a.pts)
    elif a.cmd == "overlay":
        out = a.out or os.path.splitext(image_of(s))[0] + ".measure.png"
        print(draw_overlay(s, out, sub=a.sub, levels=not a.no_levels, crop=a.crop, scale=a.scale))


def probe(s, pts):
    """probe a b        -> span of a-b in units (len, x, y), offsets and lean from plumb
       probe a b c d    -> also span of c-d and the ratio (a-b)/(c-d), each axis"""
    if len(pts) not in (2, 4):
        raise SystemExit("probe takes two landmarks, or four for a ratio")
    u = unit_px(s) if s.get("unit") else None
    unit = f"{s['unit'].get('name', 'unit')}s" if u else "px"

    def one(a, b):
        (x0, y0), (x1, y1) = _pt(s, a), _pt(s, b)
        L, dx, dy = math.hypot(x1 - x0, y1 - y0), x1 - x0, y1 - y0
        k = u or 1.0
        print(f"{a} -> {b}: len {L / k:.3f} {unit}   dx {dx / k:+.3f}   dy {dy / k:+.3f}   "
              f"lean {angle_from_vertical(s, a, b):+.1f} deg from plumb")
        return L, abs(dx), abs(dy)

    r1 = one(pts[0], pts[1])
    if len(pts) == 4:
        r2 = one(pts[2], pts[3])
        lab = ("len", "x", "y")
        print("ratio (first / second): " + "   ".join(
            f"{lab[i]} {r1[i] / r2[i]:.3f}" if r2[i] > 1e-9 else f"{lab[i]} -" for i in range(3)))


if __name__ == "__main__":
    main()
