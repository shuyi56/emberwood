"""Headless check: python3 tools/refmeasure/test_refmeasure.py  (needs Pillow)."""
import json
import os
import subprocess
import sys
import tempfile

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import refmeasure as rm  # noqa: E402

TOOL = [sys.executable, os.path.join(HERE, "refmeasure.py")]


def run(*a):
    return subprocess.run(TOOL + list(a), check=True, capture_output=True, text=True).stdout


def main():
    d = tempfile.mkdtemp()
    for name, sh_r, waist_r in (("ref.png", 300, 250), ("model.png", 330, 250)):
        p = os.path.join(d, name)
        Image.new("RGB", (400, 600), (200, 200, 200)).save(p)
        run("new", p, "--view", "front")
        sheet = rm.sheet_path(p)
        run("set", sheet, "head_top", "200", "50", "chin", "200", "150", "neck_front", "200", "170",
            "shoulder_l", "100", "200", "shoulder_r", str(sh_r), "200", "waist_l", "150", "350", "waist_r", str(waist_r), "350")
    ref, mod = (rm.sheet_path(os.path.join(d, n)) for n in ("ref.png", "model.png"))
    run("unit", ref, "head_top", "chin", "--name", "head", "--axis", "y")
    run("plumb", ref, "neck_front")
    run("seg", ref, "shoulder_w", "shoulder_l", "shoulder_r", "--axis", "x")
    run("seg", ref, "waist_w", "waist_l", "waist_r", "--axis", "x")
    run("ratio", ref, "waist_to_shoulder", "waist_w", "shoulder_w")
    run("template", ref, mod)

    c = rm.compute(rm.load(ref))
    assert c["segments"]["shoulder_w"]["units"] == 2.0, c["segments"]
    assert c["ratios"]["waist_to_shoulder"]["value"] == 0.5
    assert c["positions"]["chin"] == {"x": 0.0, "y": -0.2}

    out = run("probe", ref, "shoulder_l", "shoulder_r", "waist_l", "waist_r")
    assert "len 2.000 heads" in out and "ratio (first / second): len 2.000" in out, out

    rows, warn = rm.compare(rm.load(ref), rm.load(mod))
    gaps = {(k, n): g for k, n, a, b, g, f in rows if f}
    assert set(gaps) == {("size", "shoulder_w"), ("ratio", "waist_to_shoulder"), ("pos.x", "shoulder_r")}, gaps
    assert abs(gaps[("size", "shoulder_w")] - 0.3) < 1e-6 and not warn

    for args in (["overlay", ref], ["grid", os.path.join(d, "ref.png"), "50", "20", "350", "400"]):
        png = run(*args).strip()
        Image.open(png).verify()
    json.load(open(ref))
    print("ok", d)


if __name__ == "__main__":
    main()
