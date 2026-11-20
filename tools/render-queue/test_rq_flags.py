#!/usr/bin/env python3
"""Planted-artifact tests for rq_flags.py: each case builds a result folder with one known artifact and checks that
its flag (and no fail flag it should not have) comes out. No Blender; runs in a few seconds.

    python3 tools/render-queue/test_rq_flags.py
"""
import json, os, shutil, struct, sys, tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq_flags as F  # noqa: E402

BG = (79, 81, 85)
SIZE = 120
PARAM = f"blender/base/views.py --figure warden --turn --angles front,3/4 --size {SIZE} --samples 48"


def figure_panel(seed=0, shift=0):
    """A soft-shaded figure (head and body ellipses with a little cloth texture) on the views' grey."""
    rng = np.random.default_rng(seed)
    h, w = int(SIZE * 1.75), SIZE
    img = np.zeros((h, w, 3), np.float32) + BG
    yy, xx = np.mgrid[0:h, 0:w]
    cx = w / 2 + shift
    for (cy, ry, rx, col) in ((0.15 * h, 0.07 * h, 0.12 * w, (220, 180, 160)), (0.55 * h, 0.36 * h, 0.28 * w, (120, 60, 50))):
        d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        m = d < 1
        shade = (1.0 - 0.45 * (xx - cx + rx) / (2 * rx))[..., None]
        tex = 1 + 0.03 * np.sin(xx / 3.0)[..., None] * np.sin(yy / 4.0)[..., None]
        img[m] = (np.array(col) * shade * tex)[m]
    img += rng.normal(0, 0.6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def turn(panels):
    return np.concatenate(panels, axis=1)


def write_result(root, key, files, param=PARAM, kind="views", log_extra="", rc=0, commit="b" * 40, finished="2026-11-19T06:00:00Z"):
    d = os.path.join(root, key)
    os.makedirs(d, exist_ok=True)
    for name, data in files.items():
        p = os.path.join(d, name)
        if isinstance(data, np.ndarray):
            Image.fromarray(data).save(p)
        elif isinstance(data, list):                    # gif frames
            ims = [Image.fromarray(x) for x in data]
            ims[0].save(p, save_all=True, append_images=ims[1:], duration=60, loop=0)
        else:
            open(p, "wb").write(data)
    with open(os.path.join(d, "render.log"), "w") as f:
        f.write(f"$ python3 rq_run.py -- {param} --out /tmp/rq-work/out-{key}\n{log_extra}\n[exit {rc} after 30s]\n")
    m = {"ok": rc == 0, "key": key, "kind": kind, "param": param, "commit": commit, "combines": [],
         "seconds": 30, "files": {}, "finished": finished}
    json.dump(m, open(os.path.join(d, "manifest.json"), "w"))
    return d, m


def glb_bytes(positions, skins=True, anims=True, weights=None):
    pos = np.asarray(positions, np.float32)
    binc = pos.tobytes()
    acc = [{"bufferView": 0, "componentType": 5126, "count": len(pos), "type": "VEC3"}]
    views = [{"buffer": 0, "byteOffset": 0, "byteLength": len(binc)}]
    attrs = {"POSITION": 0}
    if weights is not None:
        w = np.asarray(weights, np.float32); j = np.zeros((len(w), 4), np.uint8)
        off = len(binc); binc += w.tobytes()
        views.append({"buffer": 0, "byteOffset": off, "byteLength": w.nbytes})
        acc.append({"bufferView": 1, "componentType": 5126, "count": len(w), "type": "VEC4"})
        off = len(binc); binc += j.tobytes()
        views.append({"buffer": 0, "byteOffset": off, "byteLength": j.nbytes})
        acc.append({"bufferView": 2, "componentType": 5121, "count": len(j), "type": "VEC4"})
        attrs.update({"WEIGHTS_0": 1, "JOINTS_0": 2})
    while len(binc) % 4:
        binc += b"\0"
    j = {"asset": {"version": "2.0"}, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
         "meshes": [{"name": "Body", "primitives": [{"attributes": attrs}]}],
         "accessors": acc, "bufferViews": views, "buffers": [{"byteLength": len(binc)}]}
    if skins:
        j["skins"] = [{"name": "Rig", "joints": [0]}]
    if anims:
        j["animations"] = [{"name": "Walk", "channels": [], "samplers": []}]
    js = json.dumps(j).encode()
    while len(js) % 4:
        js += b" "
    body = struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(binc), b"BIN\x00") + binc
    data = struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body
    return data + b"\0" * max(0, 1_100_000 - len(data))   # (padded past the size floor; only content is judged)


def main():
    cfg = F.load_cfg()
    root = tempfile.mkdtemp(prefix="rq-flags-test-")
    good = turn([figure_panel(1), figure_panel(2)])
    cases = []

    def case(name, files, expect, absent=(), group=None, **kw):
        # each case is its own results folder, except a reference and the case compared with it (same group)
        r = os.path.join(root, group or name)
        d, m = write_result(r, name, files, **kw)
        cases.append((name, d, m, set(expect), set(absent), r))

    noisy = np.clip(good.astype(np.int16) + np.random.default_rng(3).normal(0, 14, good.shape), 0, 255).astype(np.uint8)
    fire = good.copy()
    rng = np.random.default_rng(4)
    ys, xs = rng.integers(int(SIZE * 0.35), int(SIZE * 1.6), 60), rng.integers(SIZE * 0.4, SIZE * 0.6, 60)
    fire[ys, xs] = 255
    magenta = good.copy(); magenta[100:130, 50:70] = (255, 0, 255)
    holes = good.copy(); holes[90:130, 45:75] = 0
    flat = np.zeros_like(good) + np.array(BG, np.uint8)
    off = turn([figure_panel(1, shift=SIZE * 0.55), figure_panel(2)])
    gone = turn([np.zeros((int(SIZE * 1.75), SIZE, 3), np.uint8) + np.array(BG, np.uint8), figure_panel(2)])

    case("good", {"turn.png": good}, expect=())
    case("black", {"turn.png": np.zeros_like(good)}, expect={"black_frame"})
    case("white", {"turn.png": np.full_like(good, 255)}, expect={"white_frame"})
    case("flat", {"turn.png": flat}, expect={"blank_frame", "figure_missing"})
    case("off_camera", {"turn.png": off}, expect={"figure_cropped"})
    case("one_panel_empty", {"turn.png": gone}, expect={"figure_missing"})
    case("noise", {"turn.png": noisy}, expect={"noise"})
    case("fireflies", {"turn.png": fire}, expect={"fireflies"})
    case("magenta", {"turn.png": magenta}, expect={"missing_texture"})
    case("black_patches", {"turn.png": holes}, expect={"black_patches"})
    case("wrong_size", {"turn.png": good[:, :SIZE + 7]}, expect={"wrong_size"})
    case("missing_output", {"other.png": good}, expect={"missing_output"})
    case("tiny", {"turn.png": good}, expect={"tiny_file"}, absent={"black_frame"})
    tiny = os.path.join(root, "tiny", "tiny", "turn.png")
    open(tiny, "wb").write(open(tiny, "rb").read()[:800])
    case("log_error", {"turn.png": good}, expect={"log_error"},
         log_extra="Traceback (most recent call last):\n  File x\nValueError: boom\nERROR Draco is not available")
    case("alpha", {"turn.png": np.dstack([good, np.full(good.shape[:2], 128, np.uint8)])}, expect={"unexpected_alpha"})
    wparam = f"blender/base/views.py --figure warden --mannequin --walk 3 --angles profile --size {SIZE}"
    fr = [figure_panel(1, shift=s) for s in (-8, 0, 8)]
    case("walk_ok", {"walk.png": turn(fr), "walk-profile.gif": fr}, expect=(), param=wparam)
    same = [figure_panel(1)] * 3
    case("walk_frozen", {"walk.png": turn(same), "walk-profile.gif": same}, expect={"frozen_animation"}, param=wparam)
    # same as main: a main render of this exact job at an earlier commit (the test repo makes it count as main)
    case("main_ref", {"turn.png": good}, expect=(), commit="a" * 40, finished="2026-11-18T00:00:00Z", group="same")
    case("same_as_before", {"turn.png": good}, expect={"same_as_previous"}, commit="c" * 40, group="same")
    case("changed_ref", {"turn.png": good}, expect=(), commit="a" * 40, finished="2026-11-18T00:00:00Z", group="chg")
    case("changed", {"turn.png": turn([figure_panel(1, shift=6), figure_panel(2)])}, expect=(), commit="c" * 40,
         group="chg")
    tri = [[0, 0, 0], [0, 170, 0], [40, 0, 0]]
    w_ok = [[1, 0, 0, 0]] * 3
    case("glb_ok", {"ranger.glb": glb_bytes(tri, weights=w_ok)}, expect=(), kind="glb", param=None,
         finished="2026-11-18T00:00:00Z", commit="a" * 40, group="glb")
    case("glb_nan", {"ranger.glb": glb_bytes([[0, 0, 0], [np.nan, 1, 0], [1e9, 0, 0]], weights=w_ok)},
         expect={"glb_bad_vertices"}, kind="glb", param=None)
    case("glb_bare", {"ranger.glb": glb_bytes(tri, skins=False, anims=False)},
         expect={"glb_no_skin", "glb_no_animation"}, kind="glb", param=None)
    case("glb_weights", {"ranger.glb": glb_bytes(tri, weights=[[0.3, 0, 0, 0]] * 3)}, expect={"glb_bad_weights"},
         kind="glb", param=None)
    case("glb_scale", {"ranger.glb": glb_bytes([[0, 0, 0], [0, 1700, 0], [400, 0, 0]], weights=w_ok)},
         expect={"glb_bounds"}, kind="glb", param=None, group="glb")
    case("glb_garbage", {"ranger.glb": b"not a glb" * 200000}, expect={"glb_empty"}, kind="glb", param=None)

    bad = 0
    for name, d, m, expect, absent, results in cases:
        rep = F.run(d, m, results=results, repo=None, cfg=cfg)
        got = {f["id"] for f in rep["flags"]}
        fails = {f["id"] for f in rep["flags"] if f["level"] == "fail"}
        ok = expect <= got and not (absent & got) and (expect or not got)
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name:16s} expected {sorted(expect) or '-'} got {sorted(got) or '-'}"
              + (f" (fail: {sorted(fails)})" if fails else ""))
    shutil.rmtree(root)
    print(f"{len(cases) - bad}/{len(cases)} cases pass")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
