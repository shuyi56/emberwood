#!/usr/bin/env python3
"""Post-render artifact flags: cheap checks of a finished job's files (image statistics, .glb structure, the log),
run by rq.py's finish() after every job. No Blender, no scipy: PIL and numpy only.

    rq_flags.py check DIR [--write]          flag one result folder (reads DIR/manifest.json for kind and param)
    rq_flags.py scan RESULTS [--only ID,..]  flag every result folder under RESULTS and print a table
    rq_flags.py calibrate --rejections FILE [--results DIR] [--write]
                                             re-measure rejected files and propose threshold edits to flags.json
    rq_flags.py accept KEY [--by NAME] [--reason TEXT]
                                             a job failed only by its flags was fine: move it to results/KEY and
                                             record the override (calibrate reads it as a false alarm)

Every flag id is also the artifact tag a rejection carries (rq_reject.py --artifacts), so the nightly calibrate run
can line up what John rejected with what the checks saw. Thresholds live in flags.json beside this file; each check
is `metric op limit` with a warn and/or a fail limit. A fail flag turns the job into a failed result (results/KEY-failed)
when flags.json says fail_marks_job_failed; warn flags only report.
"""
import argparse, fnmatch, glob, json, os, re, shlex, struct, subprocess, sys

import numpy as np
from PIL import Image, ImageSequence

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.environ.get("RQ_FLAGS", os.path.join(HERE, "flags.json"))
DATA = os.environ.get("RQ_HOME", HERE)
OVERRIDES = os.path.join(DATA, "flag-overrides.jsonl")
ROWS, COLS = 8, 10
WALK_COLS = range(2, 10)          # the sheet's walk columns (0-1 are the idle)


def load_cfg(path=CFG_PATH):
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ helpers
def span(s, n):
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-"); out += range(int(a), int(b) + 1)
        else:
            out.append(int(part))
    return [x for x in out if 0 <= x < n]


def opt(words, k, d=None):
    return words[words.index(k) + 1] if k in words and words.index(k) + 1 < len(words) else d


def view_words(kind, param):
    """The script and its arguments of a views job (with the queue's env prefix and quality suffix taken off)."""
    if kind != "views" or not param:
        return None, []
    p = param.replace("  [quality check]", "")
    try:
        words = shlex.split(p)
    except ValueError:
        return None, []
    if words and words[0] == "env":
        words = words[1:]
        while words and "=" in words[0]:
            words = words[1:]
    return (words[0] if words else None), words[1:]


def layout_for(name, cfg):
    for lay in cfg["layouts"]:
        if fnmatch.fnmatch(name, lay["glob"]):
            return lay
    return {"panels": "none", "figure": False}


def lum_of(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def neigh(a, fn):
    """fn over the 8 neighbours of every pixel (edge pixels padded with themselves)."""
    p = np.pad(a, 1, mode="edge")
    h, w = a.shape
    parts = [p[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3) if (dy, dx) != (1, 1)]
    return fn(parts)


def erode(mask, r):
    for _ in range(r):
        mask = mask & neigh(mask, lambda ps: np.logical_and.reduce(ps))
    return mask


def changed_fraction(a, b, tol):
    if a.shape != b.shape:
        return None
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=-1)
    return float((d > tol).mean())


def read_rgba(path):
    im = Image.open(path)
    has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
    return np.asarray(im.convert("RGBA")), has_alpha


# ------------------------------------------------------------------ panels and expected sizes
def views_geometry(words, cfg):
    vp = cfg["views_py"]
    size = int(opt(words, "--size", vp["default_size"]))
    angles = opt(words, "--angles")
    n = len([a for a in angles.split(",") if a.strip()]) if angles else len(vp["angles"])
    return size, n


def panels_of(name, shape, lay, kind, param, cfg):
    """Panel boxes (x0, y0, x1, y1) and, for walk/run sheets, the grid (rows, cols)."""
    h, w = shape[:2]
    mode = lay.get("panels", "none")
    script, words = view_words(kind, param)
    size, n = views_geometry(words, cfg) if script else (None, None)
    if mode == "single":
        return [(0, 0, w, h)], None
    if mode in ("turn", "head"):
        ph = int(size * 1.75) if (size and mode == "turn") else (size or h)
        cols = max(1, round(w / (size or (h / 1.75 if mode == "turn" else h))))
        pw = w // cols
        return [(i * pw, 0, (i + 1) * pw, h) for i in range(cols)], (1, cols)
    if mode in ("walk", "run"):
        N = opt(words, "--" + mode)
        if not (N and N.isdigit() and size):
            return [(0, 0, w, h)], None
        cols, ph = int(N), int(size * 1.75)
        rows = max(1, round(h / ph))
        pw = w // cols
        return [(c * pw, r * ph, (c + 1) * pw, (r + 1) * ph) for r in range(rows) for c in range(cols)], (rows, cols)
    return [(0, 0, w, h)], None


def expected_size(name, kind, param, cfg):
    """(w, h) the request asked for, with None for a side the queue cannot know; None when nothing is known."""
    script, words = view_words(kind, param)
    cw, ch = cfg["cell_size"]
    if re.fullmatch(r"cell-r\d+-c\d+\.png", name):
        return (cw, ch)
    if name == "cells.png":
        spec = param if kind == "frames" else opt(words, "--cells")
        if spec and ":" in spec:
            r, c = spec.split(":")
            return (3 * cw * len(span(c, COLS)), 3 * ch * len(span(r, ROWS)))
        return None
    if not script or os.path.basename(script) != os.path.basename(cfg["views_py"]["script"]):
        return None
    size, n = views_geometry(words, cfg)
    stem = name[:-4].replace("-clay", "")
    if stem in ("turn", "pieces"):
        return (size * n, int(size * 1.75))
    if stem == "head":
        return (size * n, size)
    if stem == "walk" and (opt(words, "--walk") or "").isdigit():
        return (size * int(opt(words, "--walk")), int(size * 1.75) * n)
    if stem == "run" and (opt(words, "--run") or "").isdigit():
        return (None, int(size * 1.75) * n)
    return None


def expected_files(kind, param, cfg):
    """File names (relative to the result folder) the job must have written."""
    if kind == "glb":
        return ["ranger.glb"]
    if kind == "sheet":
        return ["ranger-blender.png", "ranger.blend", "ranger.glb"]
    if kind == "frames" and param and ":" in param:
        r, c = param.split(":")
        return [f"cell-r{i}-c{j}.png" for j in span(c, COLS) for i in span(r, ROWS)] + ["cells.png"]
    script, words = view_words(kind, param)
    if not script or os.path.basename(script) != os.path.basename(cfg["views_py"]["script"]):
        return []
    if "--list" in words or "--save" in words:
        return []
    out = []
    looks = ("", "-clay") if "--clay" in words else ("",)
    angles = opt(words, "--angles")
    names = [a.strip().replace("/", "").replace(" ", "") for a in angles.split(",")] if angles else \
        [a.replace("/", "") for a in cfg["views_py"]["angles"]]
    for look in looks:
        if "--turn" in words:
            out.append(f"turn{look}.png")
        if "--head" in words:
            out.append(f"head{look}.png")
    if "--pieces" in words:
        out.append("pieces.png")
    for stem in ("walk", "run"):
        if (opt(words, "--" + stem) or "").isdigit():
            out.append(f"{stem}.png")
            out += [f"{stem}-{a}.gif" for a in names]
    cells = opt(words, "--cells")
    if cells and ":" in cells:
        r, c = cells.split(":")
        out += [f"cell-r{i}-c{j}.png" for j in span(c, COLS) for i in span(r, ROWS)] + ["cells.png"]
    return out


# ------------------------------------------------------------------ image metrics
def image_metrics(path, name, kind, param, cfg):
    ic = cfg["image"]
    lay = layout_for(name, cfg)
    arr, has_alpha = read_rgba(path)
    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3].astype(np.float32) / 255
    m = {}
    # whole-image statistics on the picture as seen (transparent pixels composited over mid grey)
    seen = rgb * alpha[..., None] + 80 * (1 - alpha[..., None])
    L = lum_of(seen)
    m["lum_p99"], m["lum_p01"], m["lum_std"] = (float(np.percentile(L, 99)), float(np.percentile(L, 1)),
                                                float(L.std()))
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    vis = alpha > ic["fg_min_alpha"] / 255
    m["magenta_fraction"] = float(((r > 200) & (b > 200) & (g < 60) & vis).mean())
    if not lay.get("alpha"):
        m["alpha_fraction"] = float((arr[..., 3] < 255).mean()) if has_alpha else 0.0
    exp = expected_size(name, kind, param, cfg)
    if exp:
        w, h = arr.shape[1], arr.shape[0]
        m["size_mismatch"] = float((exp[0] is not None and exp[0] != w) or (exp[1] is not None and exp[1] != h))
        m["size_expected"] = [exp[0], exp[1]]
    m["size"] = [arr.shape[1], arr.shape[0]]
    if lay.get("figure", True) is False or lay.get("panels") == "none":
        return m, arr
    boxes, grid = panels_of(name, arr.shape, lay, kind, param, cfg)
    fgf, touch, noise, ff_n, ff_px, black, black_px = [], [], [], 0, 0, 0, 0
    edges = lay.get("edges", "")
    worst_touch = None
    for pi, (x0, y0, x1, y1) in enumerate(boxes):
        P = arr[y0:y1, x0:x1]
        if P.size == 0:
            continue
        prgb = P[..., :3].astype(np.float32)
        if lay.get("alpha"):
            fg = P[..., 3] > ic["fg_min_alpha"]
        else:
            border = np.concatenate([prgb[0], prgb[-1], prgb[:, 0], prgb[:, -1]])
            bg = np.median(border, axis=0)
            diff = np.abs(prgb - bg).max(axis=-1)
            diff = (neigh(diff, lambda ps: sum(ps)) + diff) / 9.0     # (a 3x3 mean: render noise is not figure)
            fg = diff > ic["bg_tolerance"]
            if lay.get("panels") in ("walk", "run"):
                # the walk's ground plane fills the bottom of every frame: judge the figure above it
                rows_fg = fg.mean(axis=1)
                floor = np.where(rows_fg > 0.9)[0]
                if len(floor):
                    fg[floor.min():] = False
        fgf.append(float(fg.mean()))
        band = ic["edge_band"]
        sides = {"t": fg[:band], "b": fg[-band:], "l": fg[:, :band], "r": fg[:, -band:]}
        for e in edges:
            t = float(sides[e].mean())
            if worst_touch is None or t > worst_touch[0]:
                worst_touch = (t, pi, e)
        interior = erode(fg, 2)
        if interior.sum() < 50 or lay.get("surface") is False:   # (sprite cells are pixel art: no surface checks)
            continue
        PL = lum_of(prgb)
        mean8 = neigh(PL, lambda ps: sum(ps) / 8.0)
        resid = np.abs(PL - mean8)[interior]
        noise.append(float(np.median(resid)))
        max8 = neigh(PL, lambda ps: np.maximum.reduce(ps))
        ff = (PL - max8 > ic["firefly_jump"]) & (PL > ic["firefly_min_lum"]) & interior
        ff_n += int(ff.sum()); ff_px += int(interior.sum())
        pb = (prgb.max(axis=-1) <= ic["pure_black_max"]) & interior
        black += int(erode(pb, 1).sum()); black_px += int(interior.sum())
    if fgf:
        m["fg_fraction_min"] = min(fgf)
        m["fg_fraction_panel"] = int(np.argmin(fgf))
    if worst_touch:
        m["edge_touch_max"], m["edge_touch_panel"], m["edge_touch_side"] = worst_touch
    if noise:
        m["noise_p50"] = float(np.median(noise))
    if ff_px:
        # (a couple of sparks on a metal edge are not an artifact: below firefly_min_count the rate counts as 0)
        m["fireflies_count"] = ff_n
        m["fireflies_per_mpx"] = ff_n / ff_px * 1e6 if ff_n >= ic.get("firefly_min_count", 8) else 0.0
    if black_px:
        m["pure_black_fraction"] = black / black_px
    if lay.get("frames") and grid and grid[1] > 1:
        tol = ic["same_pixel_tolerance"]
        worst = None
        for rr in range(grid[0]):
            row = [arr[y0:y1, x0:x1] for (x0, y0, x1, y1) in boxes[rr * grid[1]:(rr + 1) * grid[1]]]
            ch = [changed_fraction(row[i], row[i + 1], tol) for i in range(len(row) - 1)]
            ch = [c for c in ch if c is not None]
            if ch and (worst is None or max(ch) < worst):
                worst = max(ch)      # the most any frame of this row moves; a frozen row barely moves at all
        if worst is not None:
            m["frames_changed_min"] = worst
    return m, arr


def gif_metrics(path, cfg):
    tol = cfg["image"]["same_pixel_tolerance"]
    im = Image.open(path)
    frames = [np.asarray(f.convert("RGB")) for f in ImageSequence.Iterator(im)]
    m = {"gif_frames": len(frames)}
    if len(frames) > 1:
        ch = [changed_fraction(frames[i], frames[i + 1], tol) for i in range(len(frames) - 1)]
        m["frames_changed_min"] = max(c for c in ch if c is not None)
    else:
        m["frames_changed_min"] = 0.0
    return m


# ------------------------------------------------------------------ .glb
def _read_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 20 or data[:4] != b"glTF":
        return None, None
    j, binc, off = None, b"", 12
    while off + 8 <= len(data):
        ln, ct = struct.unpack_from("<I4s", data, off)
        chunk = data[off + 8: off + 8 + ln]
        if ct == b"JSON":
            j = json.loads(chunk)
        elif ct == b"BIN\x00":
            binc = chunk
        off += 8 + ln
    return j, binc


_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _accessor(j, binc, i):
    a = j["accessors"][i]
    if "bufferView" not in a:
        return None
    bv = j["bufferViews"][a["bufferView"]]
    dt = np.dtype(_CT[a["componentType"]])
    nc = _NC[a["type"]]
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride") or dt.itemsize * nc
    count = a["count"]
    if count == 0:
        return np.zeros((0, nc), dt)
    need = stride * (count - 1) + dt.itemsize * nc
    buf = np.frombuffer(binc, dtype=np.uint8, count=need, offset=start)
    out = np.lib.stride_tricks.as_strided(buf, shape=(count, nc * dt.itemsize), strides=(stride, 1))
    out = np.ascontiguousarray(out).view(dt).reshape(count, nc)
    if a.get("normalized") and dt != np.float32:
        out = out.astype(np.float32) / np.iinfo(dt).max
    return out


def glb_metrics(path, cfg):
    gc = cfg["glb"]
    j, binc = _read_glb(path)
    if j is None:
        return {"glb_empty": 1.0, "glb_error": "not a glTF binary"}
    meshes = [mm.get("name", f"mesh{i}") for i, mm in enumerate(j.get("meshes", []))]
    anims = [a.get("name", f"anim{i}") for i, a in enumerate(j.get("animations", []))]
    skins = j.get("skins", [])
    joints = max((len(s.get("joints", [])) for s in skins), default=0)
    bad, pts, bad_w, n_w = 0, 0, 0, 0
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for mesh in j.get("meshes", []):
        for prim in mesh.get("primitives", []):
            at = prim.get("attributes", {})
            if "POSITION" in at:
                p = _accessor(j, binc, at["POSITION"])
                if p is not None and len(p):
                    p = p.astype(np.float64)
                    finite = np.isfinite(p).all(axis=1)
                    extreme = np.abs(np.where(np.isfinite(p), p, 0)).max(axis=1) > gc["max_abs_position"]
                    bad += int((~finite | extreme).sum()); pts += len(p)
                    ok = p[finite & ~extreme]
                    if len(ok):
                        lo, hi = np.minimum(lo, ok.min(0)), np.maximum(hi, ok.max(0))
            if "WEIGHTS_0" in at and skins:
                w = _accessor(j, binc, at["WEIGHTS_0"])
                jn = _accessor(j, binc, at["JOINTS_0"]) if "JOINTS_0" in at else None
                if w is not None and len(w):
                    s = w.astype(np.float64).sum(axis=1)
                    wrong = ~np.isfinite(s) | (np.abs(s - 1) > gc["weight_tolerance"])
                    if jn is not None and len(jn) == len(w):
                        wrong |= ((jn >= joints) & (w > 0)).any(axis=1)
                    bad_w += int(wrong.sum()); n_w += len(w)
    m = {"glb_meshes": meshes, "glb_animations": anims, "glb_joints": joints, "glb_points": pts,
         "glb_bad_vertex_count": float(bad),
         "glb_bad_weight_fraction": bad_w / n_w if n_w else 0.0,
         "glb_empty": float(not meshes or not j.get("nodes") or pts == 0)}
    if np.isfinite(lo).all():
        m["glb_extent"] = [round(float(x), 4) for x in (hi - lo)]
    m["glb_skin_problem"] = float(gc["require_skin"] and joints == 0)
    m["glb_anims_lost"] = float(gc["require_animation"] and not anims) * 999 + \
        len([a for a in gc.get("expected_animations", []) if a not in anims])
    m["glb_meshes_lost"] = 0.0
    return m


def glb_compare(m, ref):
    """Fill the metrics that need the last good export of the same model."""
    lost = [x for x in ref.get("glb_meshes", []) if x not in m.get("glb_meshes", [])]
    m["glb_meshes_lost"] = float(len(lost)); m["glb_meshes_lost_names"] = lost[:20]
    al = [x for x in ref.get("glb_animations", []) if x not in m.get("glb_animations", [])]
    if al:
        m["glb_anims_lost"] = m.get("glb_anims_lost", 0) + len(al); m["glb_anims_lost_names"] = al[:20]
    if ref.get("glb_joints") and m.get("glb_joints", 0) < ref["glb_joints"]:
        m["glb_skin_problem"] = 1.0
    if ref.get("glb_extent") and m.get("glb_extent"):
        r = [max(a, 1e-6) / max(b, 1e-6) for a, b in zip(m["glb_extent"], ref["glb_extent"])]
        m["glb_extent_ratio"] = float(max(max(r), 1 / min(r)))


# ------------------------------------------------------------------ references (main's render of the same job)
def _on_main(commit, repo):
    if not commit or not repo or not os.path.isdir(repo):
        return False
    r = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=repo,
                       capture_output=True)
    return r.returncode == 0


def references(results, manifest, repo, want_file=None, kinds=None):
    """(main_dir, previous_dir): the latest other good result of the same job whose commit is on main, and the latest
    other good result of it at all. For a glb, any glb or sheet result counts as the same job."""
    kinds = kinds or (manifest.get("kind"),)
    cands = []
    for mp in glob.glob(os.path.join(results, "*", "manifest.json")):
        d = os.path.dirname(mp)
        if d.endswith("-failed") or os.path.basename(d) == manifest.get("key"):
            continue
        try:
            o = json.load(open(mp))
        except (OSError, ValueError):
            continue
        if o.get("kind") not in kinds or (kinds == (manifest.get("kind"),) and o.get("param") != manifest.get("param")):
            continue
        if o.get("commit") == manifest.get("commit"):
            continue
        if want_file and not os.path.exists(os.path.join(d, want_file)):
            continue
        if manifest.get("finished") and o.get("finished", "") > manifest["finished"]:
            continue                       # only renders from before this one
        cands.append((o.get("finished", ""), d, o.get("commit")))
    cands.sort(reverse=True)
    main = next((d for _, d, c in cands if _on_main(c, repo)), None)
    prev = cands[0][1] if cands else None
    return main, prev


# ------------------------------------------------------------------ the log
def log_metrics(path, cfg):
    try:
        text = open(path, errors="replace").read()
    except OSError:
        return {}
    ex = re.findall(r"\[exit (-?\d+) after", text)
    rc = int(ex[-1]) if ex else None
    errs = []
    pats = [re.compile(p, re.M) for p in cfg["log"]["error_patterns"]]
    benign = [re.compile(p) for p in cfg["log"]["benign_patterns"]]
    for line in text.splitlines():
        if any(p.search(line) for p in pats) and not any(b.search(line) for b in benign):
            errs.append(line.strip()[:200])
    saved = re.findall(r"(?:Saved: '([^'\n]+)'|^wrote (\S+))", text, re.M)
    saved = [a or b for a, b in saved]
    return {"exit": rc, "log_errors": float(len(errs)) if rc == 0 else 0.0, "log_error_lines": errs[:5],
            "saved_paths": saved}


# ------------------------------------------------------------------ evaluate
def quality_of(manifest):
    p = (manifest or {}).get("param") or ""
    return "denoised" if ("[quality check]" in p or "--denoise" in p) else "raw"


def evaluate(metrics, cfg, quality="raw"):
    """metrics: {file or '': {metric: value}} -> flags."""
    flags = []
    for fname, mm in metrics.items():
        for cid, c in cfg["checks"].items():
            c = {**c, **c.get("by_quality", {}).get(quality, {})}
            v = mm.get(c["metric"])
            if not isinstance(v, (int, float)) or v is None:
                continue
            hit = (lambda lim: v < lim) if c["op"] == "<" else (lambda lim: v > lim)
            level = "fail" if "fail" in c and hit(c["fail"]) else "warn" if "warn" in c and hit(c["warn"]) else None
            if not level:
                continue
            f = {"id": cid, "level": level, "file": fname or None, "value": round(float(v), 6),
                 "limit": c[level], "msg": c["msg"]}
            extra = {"figure_missing": ("fg_fraction_panel",), "figure_cropped": ("edge_touch_panel", "edge_touch_side"),
                     "wrong_size": ("size", "size_expected"), "missing_output": ("missing_files",),
                     "log_error": ("log_error_lines",), "glb_no_mesh": ("glb_meshes_lost_names",),
                     "glb_no_animation": ("glb_anims_lost_names",),
                     "same_as_main": ("reference",), "same_as_previous": ("reference_prev",)}.get(cid, ())
            for k in extra:
                if k in mm:
                    f[k.replace("fg_fraction_", "").replace("edge_touch_", "")] = mm[k]
            flags.append(f)
    order = {"fail": 0, "warn": 1}
    flags.sort(key=lambda f: (order[f["level"]], f["id"], f["file"] or ""))
    return flags


def measure(out, manifest, results=None, repo=None, cfg=None):
    """Every metric of a result folder: {file name or '' for the job: {metric: value}}."""
    cfg = cfg or load_cfg()
    kind, param = manifest.get("kind"), manifest.get("param")
    names = sorted(os.path.relpath(p, out) for p in glob.glob(os.path.join(out, "**", "*"), recursive=True)
                   if os.path.isfile(p))
    job = {}
    lm = log_metrics(os.path.join(out, "render.log"), cfg)
    job.update({k: v for k, v in lm.items() if k != "saved_paths"})
    want = set(expected_files(kind, param, cfg))
    outs_real = [n for n in names if n not in ("manifest.json", "render.log")]
    missing = sorted(w for w in want if w not in names)
    for p in lm.get("saved_paths", []):
        if "/out-" in p:                                   # written into the job's own --out folder
            rel = p.split("/out-", 1)[1].split("/", 1)
            # (a name starting with _ or . is a script's scratch file, removed on purpose)
            if len(rel) == 2 and rel[1] not in names and not os.path.basename(rel[1]).startswith(("_", ".")):
                missing.append(rel[1])
    if not outs_real and kind:
        missing.append("(no output at all)")
    job["missing_count"] = float(len(set(missing))); job["missing_files"] = sorted(set(missing))
    metrics = {"": job}
    tol = cfg["image"]["same_pixel_tolerance"]
    refs = {}
    if results and kind in ("views", "frames"):
        refs["main"], refs["prev"] = references(results, manifest, repo)
    cells = {}
    for n in outs_real:
        p = os.path.join(out, n)
        mm = {"bytes": os.path.getsize(p)}
        lim = next((v for g, v in cfg["file_min_bytes"].items() if fnmatch.fnmatch(os.path.basename(n), g)), None)
        mm["bytes_short"] = float(lim is not None and mm["bytes"] < lim)
        base = os.path.basename(n)
        try:
            if base.endswith(".png"):
                im_m, arr = image_metrics(p, base, kind, param, cfg)
                mm.update(im_m)
                rc = re.fullmatch(r"cell-r(\d+)-c(\d+)\.png", base)
                if rc:
                    cells[(int(rc.group(1)), int(rc.group(2)))] = (n, arr)
                for which, metric, refkey in (("main", "changed_vs_main", "reference"),
                                              ("prev", "changed_vs_prev", "reference_prev")):
                    d = refs.get(which)
                    if which == "prev" and refs.get("main"):
                        continue            # a main render to compare with says more than an earlier branch one
                    if d and os.path.exists(os.path.join(d, n)):
                        ref, _ = read_rgba(os.path.join(d, n))
                        cf = changed_fraction(arr, ref, tol)
                        if cf is not None:
                            mm[metric] = cf; mm[refkey] = os.path.basename(d)
            elif base.endswith(".gif"):
                mm.update(gif_metrics(p, cfg))
            elif base.endswith(".glb"):
                gm = glb_metrics(p, cfg)
                if results:
                    _, prev = references(results, manifest, repo, want_file=n, kinds=("glb", "sheet"))
                    if prev:
                        ref = glb_metrics(os.path.join(prev, n), cfg)
                        glb_compare(gm, ref); gm["reference"] = os.path.basename(prev)
                mm.update(gm)
            elif base.endswith(".blend"):
                with open(p, "rb") as f:
                    head = f.read(8)
                mm["blend_bad_header"] = float(not (head.startswith(b"BLENDER") or head.startswith(b"\x28\xb5\x2f\xfd")
                                                    or head.startswith(b"\x1f\x8b")))
                mm["bytes_short"] = max(mm["bytes_short"], mm["blend_bad_header"])
        except Exception as e:                              # a file the checks cannot read is itself a finding
            mm["unreadable"] = f"{type(e).__name__}: {e}"[:200]
            mm["bytes_short"] = 1.0
        metrics[n] = mm
    # same as main / as before: judged on the whole job, so a change that shows in only some views is not flagged
    imgs = [mm for n, mm in metrics.items() if n.endswith(".png")]
    for metric, job_metric, refkey in (("changed_vs_main", "changed_fraction", "reference"),
                                       ("changed_vs_prev", "changed_fraction_prev", "reference_prev")):
        vals = [mm[metric] for mm in imgs if metric in mm]
        if imgs and len(vals) == len(imgs):
            job[job_metric] = max(vals); job[refkey] = imgs[0][refkey]
    # a frames job's walk columns: the same facing at different walk columns must differ
    by_row = {}
    for (r, c), (n, arr) in cells.items():
        if c in WALK_COLS:
            by_row.setdefault(r, []).append((c, n, arr))
    worst = None
    for r, lst in by_row.items():
        lst.sort()
        ch = [changed_fraction(lst[i][2], lst[i + 1][2], tol) for i in range(len(lst) - 1)]
        ch = [x for x in ch if x is not None]
        if ch and (worst is None or max(ch) < worst[0]):
            worst = (max(ch), r)
    if worst:
        job["frames_changed_min"] = worst[0]
    return metrics


def summarize(flags):
    fails = [f for f in flags if f["level"] == "fail"]
    warns = [f for f in flags if f["level"] == "warn"]
    lvl = "fail" if fails else "warn" if warns else "ok"
    ids = sorted({f["id"] for f in flags})
    return {"level": lvl, "fail": len(fails), "warn": len(warns), "ids": ids}


def log_lines(flags):
    if not flags:
        return ["[flags] ok: no artifacts found"]
    out = [f"[flags] {summarize(flags)['level']}: {len(flags)} flag(s)"]
    for f in flags:
        where = f" {f['file']}" if f.get("file") else ""
        extra = "".join(f" {k}={f[k]}" for k in ("panel", "side", "reference", "missing_files", "size", "size_expected")
                        if k in f)
        out.append(f"[flags] {f['level'].upper():4s} {f['id']}{where}: {f['msg']} (value {f['value']:g}, "
                   f"limit {f['limit']:g}){extra}")
    return out


def compact(metrics):
    """The numbers worth keeping in manifest.json (calibrate and the nightly job read them)."""
    keep = {}
    for n, mm in metrics.items():
        keep[n] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in mm.items()
                   if isinstance(v, (int, float)) and k != "bytes"}
    return keep


def run(out, manifest, results=None, repo=None, cfg=None):
    """Flag a finished job's output folder; returns what finish() adds to the manifest."""
    cfg = cfg or load_cfg()
    metrics = measure(out, manifest, results, repo, cfg)
    flags = evaluate(metrics, cfg, quality_of(manifest))
    return {"flags": flags, "flags_summary": summarize(flags), "flag_metrics": compact(metrics),
            "flags_version": cfg.get("version", 1)}


# ------------------------------------------------------------------ commands
def cmd_check(a):
    m = json.load(open(os.path.join(a.dir, "manifest.json")))
    results = a.results or os.path.dirname(os.path.abspath(a.dir))
    rep = run(a.dir, m, results, a.repo)
    if a.write:
        m.update(rep)
        json.dump(m, open(os.path.join(a.dir, "manifest.json"), "w"), indent=2)
        with open(os.path.join(a.dir, "render.log"), "a") as f:
            f.write("\n" + "\n".join(log_lines(rep["flags"])) + "\n")
    if a.json:
        print(json.dumps({k: rep[k] for k in ("flags", "flags_summary")}, indent=2))
    else:
        print("\n".join(log_lines(rep["flags"])))


def cmd_scan(a):
    only = set(a.only.split(",")) if a.only else None
    dirs = sorted(d for d in glob.glob(os.path.join(a.results, "*")) if os.path.exists(os.path.join(d, "manifest.json")))
    counts, rows = {}, []
    for d in dirs[: a.limit or None]:
        m = json.load(open(os.path.join(d, "manifest.json")))
        rep = run(d, m, a.results if a.refs else None, a.repo)
        fl = [f for f in rep["flags"] if not only or f["id"] in only]
        for f in fl:
            counts[(f["id"], f["level"])] = counts.get((f["id"], f["level"]), 0) + 1
        rows.append({"dir": os.path.basename(d), "kind": m.get("kind"), "summary": summarize(fl), "flags": fl})
        if fl and not a.quiet:
            print(os.path.basename(d), (m.get("param") or "")[:90])
            for line in log_lines(fl)[1:]:
                print("   ", line[8:])
    print(json.dumps({"results": len(rows), "flagged": sum(1 for r in rows if r["flags"]),
                      "counts": {f"{k[0]}:{k[1]}": v for k, v in sorted(counts.items())}}, indent=1))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


# tags a reviewer gives a rejection (rq_reject.py --tags) that an image check could have caught
TAG_HINTS = {"blank": ["blank_frame", "black_frame", "white_frame", "figure_missing"],
             "framing": ["figure_cropped", "figure_missing", "wrong_size"],
             "noise": ["noise", "fireflies"], "shading": ["black_patches", "missing_texture"],
             "colour": ["missing_texture"]}


def _resolve(path, results):
    if not path:
        return None
    for p in (path, os.path.join(os.path.dirname(results), path), os.path.join(results, path),
              os.path.join(results, path.split("results/", 1)[-1])):
        if os.path.exists(p):
            return p
    return None


def _population(results, cfg, skip=(), limit=120):
    """Metrics of the newest results nobody rejected: how often a proposed limit would fire on accepted work. Results
    rendered before the flags existed are measured here (a few seconds each)."""
    pop = []
    mps = sorted(glob.glob(os.path.join(results, "*", "manifest.json")), key=os.path.getmtime, reverse=True)
    for mp in mps:
        d = os.path.dirname(mp)
        if os.path.basename(d) in skip or d.endswith("-failed"):
            continue
        try:
            m = json.load(open(mp))
        except (OSError, ValueError):
            continue
        fm = m.get("flag_metrics")
        if fm is None:
            try:
                fm = compact(measure(d, m, None, None, cfg))
            except Exception:                                   # noqa: BLE001
                continue
        pop += list(fm.values())
        limit -= 1
        if limit <= 0:
            break
    return pop


def cmd_calibrate(a):
    """Line up rejections with the checks: for each artifact tag on a rejected file, did its check fire? Propose a
    limit that would have caught a miss, but only if it fires on few results nobody rejected; list rejections whose
    human tags hint at a check that carries no artifact tag; and list flag overrides (false alarms)."""
    cfg = load_cfg(a.config)
    results = a.results
    rej = []
    if os.path.exists(a.rejections):
        for line in open(a.rejections):
            line = line.strip()
            if line:
                try:
                    rej.append(json.loads(line))
                except ValueError:
                    pass
    rejected = {r.get("key") for r in rej} | {k for r in rej for k in (r.get("jobs") or [])}
    pop = _population(results, cfg, skip=rejected, limit=a.population)
    hits, misses, hints, proposals = [], [], [], {}
    for r in rej:
        tags = r.get("artifacts") or []
        files = []
        f = _resolve(r.get("file"), results)
        if f:
            files.append(f)
        else:
            for k in ([r["key"]] if r.get("key") else []) + list(r.get("jobs") or []):
                d = _resolve(os.path.join("results", k), results)
                if d:
                    files += [p for p in glob.glob(os.path.join(d, "*")) if p.endswith((".png", ".gif", ".glb"))]
        if not tags:
            sug = sorted({c for t in r.get("tags", []) for c in TAG_HINTS.get(t, [])})
            if sug:
                hints.append({"at": r.get("at"), "request": r.get("request"), "tags": r.get("tags"),
                              "reason": r.get("reason"), "checks_that_might_apply": sug})
            continue
        for p in files:
            d = os.path.dirname(p)
            mpath = os.path.join(d, "manifest.json")
            man = json.load(open(mpath)) if os.path.exists(mpath) else {}
            mets = measure(d, man, None, None, cfg)
            base = os.path.relpath(p, d)
            mm = dict(mets.get(base, {}))
            mm.update({k: v for k, v in mets.get("", {}).items() if k not in mm})
            if r.get("region") and base.endswith(".png"):
                # the region the reviewer marked: measure it as one panel of its own
                x0, y0, x1, y1 = [int(v) for v in r["region"]]
                arr, _ = read_rgba(p)
                crop = os.path.join(a.tmp, "region.png")
                os.makedirs(a.tmp, exist_ok=True)
                Image.fromarray(arr[y0:y1, x0:x1]).save(crop)
                reg, _ = image_metrics(crop, "cell-r0-c0.png" if base.startswith("cell") else "turn.png",
                                       "region", None, {**cfg, "layouts": [{"glob": "*", "panels": "single",
                                                                           "edges": "", "alpha": base.startswith("cell")}]})
                for k in ("noise_p50", "fireflies_per_mpx", "pure_black_fraction", "magenta_fraction"):
                    if k in reg:
                        mm[k] = max(mm.get(k, 0), reg[k])
            fired = {f["id"] for f in evaluate({base: mm}, cfg, quality_of(man))}
            for t in tags:
                c = cfg["checks"].get(t)
                if not c:
                    continue
                q = quality_of(man)
                bucket = q if q in c.get("by_quality", {}) else None   # a limit of its own for this quality
                c = {**c, **c.get("by_quality", {}).get(q, {})}
                v = mm.get(c["metric"])
                row = {"at": r.get("at"), "request": r.get("request"), "file": base, "artifact": t, "value": v}
                if t in fired:
                    hits.append(row); continue
                misses.append(row)
                if not isinstance(v, (int, float)):
                    continue
                # the limit that would have caught it, a hair past the value
                new = v * (1.02 if c["op"] == "<" else 0.98)
                vals = [x.get(c["metric"]) for x in pop if isinstance(x.get(c["metric"]), (int, float))]
                fires = [x for x in vals if (x < new if c["op"] == "<" else x > new)]
                rate = len(fires) / len(vals) if vals else None
                pk = f"{t}@{bucket}" if bucket else t
                cur = proposals.get(pk, {}).get("warn")
                better = cur is None or (new > cur if c["op"] == "<" else new < cur)
                if better:
                    proposals[pk] = {"check": t, "quality": bucket, "metric": c["metric"], "op": c["op"], "warn_now": c.get("warn"), "warn": round(new, 6),
                                    "would_fire_on_accepted": rate, "accepted_measured": len(vals)}
    overrides = []
    if os.path.exists(OVERRIDES):
        overrides = [json.loads(l) for l in open(OVERRIDES) if l.strip()]
    # a false alarm John overrode: suggest moving the fail limit past its value
    for o in overrides:
        for f in o.get("flags", []):
            c = cfg["checks"].get(f["id"])
            if c and f["level"] == "fail":
                p = proposals.setdefault(f["id"], {"metric": c["metric"], "op": c["op"]})
                nv = f["value"] * (0.98 if c["op"] == "<" else 1.02)
                p.setdefault("fail_now", c.get("fail")); p["fail"] = round(nv, 6)
    accept = {k: v for k, v in proposals.items()
              if v.get("would_fire_on_accepted") is None or v["would_fire_on_accepted"] <= a.max_rate}
    held = {k: v for k, v in proposals.items() if k not in accept}
    report = {"rejections": len(rej), "with_artifacts": sum(1 for r in rej if r.get("artifacts")),
              "caught": hits, "missed": misses, "untagged_hints": hints, "overrides": len(overrides),
              "proposed": accept, "held_back_too_noisy": held}
    if a.write and accept:
        for k, v in accept.items():
            c = cfg["checks"][v.get("check", k)]
            if v.get("quality"):
                c = c["by_quality"][v["quality"]]
            for lvl in ("warn", "fail"):
                if lvl in v:
                    c[lvl] = v[lvl]
        cfg["version"] = cfg.get("version", 1) + 1
        with open(a.config, "w") as f:
            json.dump(cfg, f, indent=2); f.write("\n")
        report["written"] = a.config
    print(json.dumps(report, indent=1, default=str))


def cmd_accept(a):
    results = os.path.join(DATA, "results")
    src, dst = os.path.join(results, a.key + "-failed"), os.path.join(results, a.key)
    m = json.load(open(os.path.join(src, "manifest.json")))
    if not m.get("failed_by_flags"):
        sys.exit("only a job failed by its flags alone can be accepted")
    m["ok"] = True; m["flags_accepted"] = {"by": a.by, "reason": a.reason}
    json.dump(m, open(os.path.join(src, "manifest.json"), "w"), indent=2)
    with open(os.path.join(src, "render.log"), "a") as f:
        f.write(f"\n[flags] accepted by {a.by}: {a.reason}\n")
    os.rename(src, dst)
    with open(OVERRIDES, "a") as f:
        f.write(json.dumps({"key": a.key, "by": a.by, "reason": a.reason,
                            "flags": [x for x in m.get("flags", []) if x["level"] == "fail"]}) + "\n")
    print(dst)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("dir"); c.add_argument("--write", action="store_true")
    c.add_argument("--json", action="store_true"); c.add_argument("--results"); c.add_argument("--repo",
                                                                                                default=os.environ.get("RQ_REPO", "/home/user/emberwood"))
    s = sub.add_parser("scan"); s.add_argument("results"); s.add_argument("--only"); s.add_argument("--limit", type=int)
    s.add_argument("--out"); s.add_argument("--quiet", action="store_true")
    s.add_argument("--refs", action="store_true", help="also compare with main's render (same_as_main)")
    s.add_argument("--repo", default=os.environ.get("RQ_REPO", "/home/user/emberwood"))
    k = sub.add_parser("calibrate"); k.add_argument("--rejections", required=True)
    k.add_argument("--results", default=os.path.join(DATA, "results")); k.add_argument("--write", action="store_true")
    k.add_argument("--config", default=CFG_PATH); k.add_argument("--max-rate", type=float, default=0.02,
                                                                 help="hold back a limit that fires on more than this share of accepted results")
    k.add_argument("--tmp", default="/tmp/rq-flags")
    k.add_argument("--population", type=int, default=120, help="how many accepted results to test a new limit on")
    ac = sub.add_parser("accept"); ac.add_argument("key"); ac.add_argument("--by", default="John")
    ac.add_argument("--reason", required=True)
    a = p.parse_args()
    {"check": cmd_check, "scan": cmd_scan, "calibrate": cmd_calibrate, "accept": cmd_accept}[a.cmd](a)


if __name__ == "__main__":
    main()
