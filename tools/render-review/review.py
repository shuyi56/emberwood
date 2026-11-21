#!/usr/bin/env python3
"""Render request reviewer: reads a render request before it reaches the render queue.

The reviewer thread runs this on every request a thread sends. Nothing here renders. A request that passes is
forwarded to the render-queue thread unchanged; one that does not is sent back to its thread with the findings
below as concrete fixes, so no render time is spent on it.

    review.py check --thread ID --request '{"branch": ..., "outputs": [...], ...}' [--no-smoke] [--json]
    review.py pitfalls                      print the pitfalls file
    review.py learn --title T --spot S --fix F [--pattern REGEX [--unless REGEX]] [--seen WHERE]
                                            add a pitfall (after a rejection or a render that came out wrong)

What `check` does, cheapest first:
  shape     outputs, cells and view commands are well formed; the full sheet is never requestable (not as an
            output, not as a character script run as a view, not as frames covering most of the sheet)
  commit    the branch and commit exist on origin
  before    a commit already on main is served from the cache only (the queue's own rule, asked early)
  merge     the branch merged with main: code conflicts, names defined twice, names used but never defined
  views     each view script exists in the commit and knows every flag it is given
  pitfalls  every pitfall in PITFALLS.md that carries a pattern is matched against the lines the branch adds
  smoke     each job is run for real, headless, up to its first render (smoke.py): imports, argument parsing
            and the whole scene build. A crash here is a crash the queue would have hit minutes in.
  facecheck a request that only checks a face (face sculpt or face paint changed, face views asked) is not
            forwarded: blender/facecheck/facecheck.py draws and measures it here, without Blender, in seconds
  hair      when the commit touches hair code, hair_check.py runs inside that same build: creases, ridges and
            ledges in the hair in all 8 facings, clothes coming through it in the walk, and the strand paint.
            New flags since the branch's last measured commit send the request back; old ones are listed.
  cache     jobs the queue already holds are reported, so the requester knows the answer is immediate

Findings are 'block' (send back), 'warn' (forward, and tell the requester) or 'info'. The verdict is 'fix' when
anything blocks, 'facecheck' when the face check API answered the request, otherwise 'pass'. The reviewer then reads the diff (saved beside the report) and may still
send a request back for something no script can see.

The queue's own helpers (rq.py, beside queue.json in the project folder) are imported, never copied, so the
reviewer and the queue always agree on merges, cache keys and what counts as a duplicate definition. The
reviewer only reads the queue's state; it never writes queue.json.
"""
import argparse
import ast
import concurrent.futures
import datetime
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
RQ_DIR = os.environ.get("RQ_DIR", "/mnt/project-files/render-queue")
os.environ.setdefault("RQ_WORK", "/tmp/rq-review-work")   # never the queue's own work folder
sys.path.insert(0, RQ_DIR)
import rq  # noqa: E402  (the render queue's own module)

WORK = os.environ["RQ_WORK"]
PRESETS = {"view-clay", "turn-clay", "turn-matcap", "turn-lit", "head-clay", "head-lit", "closeup", "walk", "run"}
AREAS = ("face", "head", "shoulders", "torso", "waist", "hips", "hands", "legs", "knees", "feet", "full")   # view_tools.AREA_NAMES
REQ_ENV = {}   # the request's "env" (string variables the job needs); set in main, used by smoke builds
PITFALLS = os.environ.get("RQ_PITFALLS", os.path.join(HERE, "PITFALLS.md"))
LOG = os.path.join(RQ_DIR, "review", "reviews.jsonl")
SMOKE_TIMEOUT = int(os.environ.get("RQ_SMOKE_TIMEOUT", "1800"))
SHEET_ROWS, SHEET_COLS = 8, 10
SHELL_META = re.compile(r"[;&|<>$`\\\n\r]")
MATERIAL = re.compile(r"(_closeup|_material|_shade|_tones|_image|_tex|_painting|_paint|_LOOKS?|_TONE|_TINT|TEX)$"
                      r"|^M_[A-Z0-9_]+$|^(mat_def|ramp|MATERIALS|MAKEUP|LIP_COLOURS|SKIN|SKIN_FLUSH)$")
MATLINE = re.compile(r"mat_def\(|_closeup\(|\bramp\(|\bM_[A-Z0-9_]+\b|_LOOKS?\b|default_value|node_tree|\.nodes\b"
                     r"|\.links\b|Sheen|Roughness|Metallic|Base Color|Specular|\bmaterials\b")


def material_only(since, tip):
    """The material things a diff changes, if it changes nothing else under blender/ (else []). A hunk inside a
    material function counts whole; elsewhere every changed line must itself be a material line, a comment, or the
    continuation of one (a look table's rows, the body of a new material function)."""
    files = rq.git("diff", "--name-only", since, tip, "--", "blender/", check=False).stdout.split()
    if not files or any(not f.endswith((".py", ".png", ".jpg", ".exr")) for f in files):
        return []
    found, skip, block = set(), False, None        # block: indent of the accepted line a continuation hangs from
    for ln in rq.git("diff", "-U0", since, tip, "--", "blender/", check=False).stdout.splitlines():
        if ln.startswith(("diff ", "index ", "--- ", "+++ ", "Binary ", "new file", "deleted file", "similarity")):
            continue
        if ln.startswith("@@"):
            m = re.search(r"\bdef (\w+)", ln.split("@@")[-1])
            skip, block = bool(m and MATERIAL.search(m.group(1))), None
            if skip:
                found.add(m.group(1))
            continue
        if skip or not ln[:1] in "+-":
            continue
        raw = ln[1:]
        t = raw.strip()
        ind = len(raw) - len(raw.lstrip())
        if not t or t.startswith("#"):
            continue
        if block is not None and (ind > block or re.fullmatch(r"[)\]}]+,?", t)):
            continue
        block = None
        m = re.match(r"def (\w+)", t)
        if m and MATERIAL.search(m.group(1)):
            found.add(m.group(1))
            block = ind
            continue
        hit = MATLINE.search(t)
        if not hit:
            return []
        found.add(hit.group(0).rstrip("("))
        block = ind
    return sorted(found)


HAIR = re.compile(r"hair|bang|fringe|lock(?!ed)|pony|wisp|strand|braid|bun(?![a-z])", re.I)
# what blender/facecheck/facecheck.py draws and measures without Blender: the sculpted face and its paintings
FACE = re.compile(r"face|mouth|jaw|lid|eye|iris|sclera|brow|lash|lip|nose|nostril|chin|cheek|philtrum|"
                  r"head_section|head_front|sculpted_head|build_head", re.I)
NOT_FACE = re.compile(r"skin|tone|sss|scatter|mat(erial)?_|shader|light|sprite|palette|pixel|cloth|" + HAIR.pattern, re.I)
FACE_VIEW_SCRIPTS = ("face_views.py", "sculpt_views.py")
FACECHECK = "blender/facecheck/facecheck.py"
FACECHECK_REFS = ("origin/main", "origin/claude/project-thread-k27az6")   # the second only until PR #26 merges
FACECHECK_OUT = os.path.join(RQ_DIR, "facechecks")
HAIR_DIR = os.path.join(RQ_DIR, "review", "hair")       # hair-check results, one folder per commit measured
HAIR_NEW_PX = 25           # a flag this big that the branch's last measured commit did not have sends it back
HAIR_SEMI = 0.45           # share of a strand texture's painted texels that are half-transparent
HAIR_TONE = 0.045          # chromaticity distance between the strand paint and the hair's ramp
KEEP_AT_MAIN = ("blender/ranger/ranger-blender.png", "blender/ranger/ranger.blend", "public/models/ranger.glb")


class Report:
    def __init__(self):
        self.findings = []

    def add(self, level, check, text, fix=None, pitfall=None):
        self.findings.append({"level": level, "check": check, "text": text, "fix": fix, "pitfall": pitfall})

    def block(self, *a, **k): self.add("block", *a, **k)
    def warn(self, *a, **k): self.add("warn", *a, **k)
    def info(self, *a, **k): self.add("info", *a, **k)

    @property
    def blocked(self):
        return any(f["level"] == "block" for f in self.findings)


# ------------------------------------------------------------------ pitfalls file
def read_pitfalls(path=None):
    """PITFALLS.md as a list of {id, title, seen, spot, fix, pattern}. One '## P012 Title' section each."""
    out = []
    try:
        text = open(path or PITFALLS).read()
    except FileNotFoundError:
        return out
    for m in re.finditer(r"^## (P\d+) (.+?)\n(.*?)(?=^## |\Z)", text, re.M | re.S):
        p = {"id": m[1], "title": m[2].strip()}
        for key in ("seen", "spot", "fix", "pattern", "unless"):
            f = re.search(rf"^- \*\*{key.capitalize()}:\*\* (.+?)(?=^- \*\*|\Z)", m[3], re.M | re.S)
            p[key] = " ".join(f[1].split()) if f else None
        for key in ("pattern", "unless"):
            if p[key]:
                p[key] = p[key].strip("`")
        out.append(p)
    return out


def cmd_learn(a):
    ps = read_pitfalls()
    for rx in (a.pattern, a.unless):
        if rx:
            re.compile(rx)
    if any(p["title"].lower() == a.title.lower() for p in ps):
        sys.exit("a pitfall with this title already exists; edit PITFALLS.md to refine it")
    pid = f"P{max([int(p['id'][1:]) for p in ps] or [0]) + 1:03d}"
    seen = a.seen or datetime.date.today().isoformat()
    entry = f"\n## {pid} {a.title}\n\n- **Seen:** {seen}\n- **Spot:** {a.spot}\n- **Fix:** {a.fix}\n"
    if a.pattern:
        entry += f"- **Pattern:** `{a.pattern}`\n"
    if a.unless:
        entry += f"- **Unless:** `{a.unless}`\n"
    with open(PITFALLS, "a") as f:
        f.write(entry)
    print(f"added {pid} to {PITFALLS}; commit and push it so the next reviewer has it")


def cmd_pitfalls(a):
    for p in read_pitfalls():
        print(f"{p['id']}  {p['title']}\n      spot: {p['spot']}\n      fix:  {p['fix']}"
              + (f"\n      pattern: {p['pattern']}" if p["pattern"] else ""))


# ------------------------------------------------------------------ small helpers
def cells_of(spec):
    """'0-7:2,4,6' -> (rows, cols), or raises ValueError."""
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise ValueError("cells must be ROWS:COLS, e.g. 0-7:2,4,6")
    out = []
    for part, n, what in zip(spec.split(":"), (SHEET_ROWS, SHEET_COLS), ("row", "column")):
        vals = set()
        for x in part.split(","):
            m = re.fullmatch(r"(\d+)(?:-(\d+))?", x.strip())
            if not m:
                raise ValueError(f"cannot read {x!r} in cells {spec!r}")
            lo, hi = int(m[1]), int(m[2] or m[1])
            if lo > hi or hi >= n:
                raise ValueError(f"{what} {x} is outside the sheet ({what}s run 0-{n - 1})")
            vals.update(range(lo, hi + 1))
        out.append(sorted(vals))
    return out


def check_cells(rep, spec, where):
    try:
        rows, cols = cells_of(spec)
    except ValueError as e:
        rep.block("shape", f"{where}: {e}", "Use ROWS:COLS with rows 0-7 (the facings) and columns 0-9 "
                  "(0-1 idle, 2-9 walk), for example 0-7:2,4,6.")
        return
    n = len(rows) * len(cols)
    if n >= 0.75 * SHEET_ROWS * SHEET_COLS:
        rep.block("shape", f"{where}: {spec} is {n} of the sheet's {SHEET_ROWS * SHEET_COLS} cells, which is the "
                  "full sprite sheet under another name.",
                  "Ask for the cells that show your change, for example 0-7:2,4,6 (24 cells). The full sheet "
                  'runs only as the scheduled task; send {"branch": "...", "ready": true} to be included.', "P001")
    elif n > 32:
        rep.warn("shape", f"{where}: {spec} is {n} cells; recent requests needed 24 to 32.",
                 "Drop the columns you will not look at.")
    if len(rows) < SHEET_ROWS:
        rep.warn("shape", f"{where}: only facings {','.join(map(str, rows))} are requested. Most problems show "
                 "in the 3/4 and profile rows.", "Use rows 0-7 unless the change cannot be seen from the others.",
                 "P013")


def flags_known_to(src):
    """Every '--flag' string literal in a script: what its argparse or hand-written argv parsing can read."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    return {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and re.fullmatch(r"--[A-Za-z][\w-]*", n.value)}


def show(commit, path):
    r = rq.git("show", f"{commit}:{path}", check=False)
    return r.stdout if r.returncode == 0 else None


def added_lines(base, tip):
    """[(path, line number, text)] for every line the branch adds to a render input."""
    out, path, ln = [], None, 0
    d = rq.git("diff", "-U0", base, tip, "--", *rq.INPUTS, check=False).stdout
    for line in d.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("@@"):
            ln = int(re.search(r"\+(\d+)", line)[1])
        elif line.startswith("+") and path and not rq.gen(path):
            out.append((path, ln, line[1:])); ln += 1
    return out


def flake_names(commit, paths=None):
    """{'name': 'path:line'} for names used but never defined (pyflakes), by name so line shifts do not matter."""
    try:
        from pyflakes.api import check
        from pyflakes.reporter import Reporter
        import io
    except ImportError:
        return None
    hits = {}
    for path in rq.git("ls-tree", "-r", "--name-only", commit, "--", "blender/").stdout.split():
        if not path.endswith(".py"):
            continue
        buf = io.StringIO()
        check(show(commit, path) or "", path, Reporter(buf, io.StringIO()))
        for ln in buf.getvalue().splitlines():
            m = re.match(r"(.+?):(\d+):\d+:? (undefined name '(.+)'|local variable '(.+)' .*referenced before assignment)", ln)
            if m:
                hits.setdefault(m[4] or m[5], f"{m[1]}:{m[2]}")
    return hits


def last_rendered(branch, tip):
    """The newest commit of this branch the queue has already rendered and that `tip` builds on, or None."""
    for r in reversed(rq.load()["requests"]):
        t = r.get("tip")
        if r["branch"] == branch and r["state"] == "done" and t and t != tip and \
                rq.git("merge-base", "--is-ancestor", t, tip, check=False).returncode == 0:
            return t
    return None


# ------------------------------------------------------------------ face checks: no render at all
def face_only(fresh, outs, parsed):
    """(figure, why) when the request only checks a face: every changed definition is face sculpt or face
    paint, and every view is a face view. Such a request goes to facecheck.py, not to the render queue."""
    if outs != ["views"] or not parsed:
        return None, None
    figure = None
    for _, tok in parsed:
        name = os.path.basename(tok[0])
        if name in FACE_VIEW_SCRIPTS:
            fig = "ranger"
        elif name == "views.py" and "--head" in tok and "--cells" not in tok and "--turn" not in tok:
            fig = tok[tok.index("--figure") + 1] if "--figure" in tok and tok.index("--figure") + 1 < len(tok) else "ranger"
        else:
            return None, None
        if figure and fig != figure:
            return None, None
        figure = fig
    defs = sorted({d for p, ds in fresh.items() if p.endswith(".py") and not p.startswith("blender/facecheck/")
                   for d in ds if d not in ("<module>", "Expr", "Import", "ImportFrom")})
    if any(not p.endswith(".py") for p in fresh if not p.startswith("blender/facecheck/")) or not defs:
        return None, None
    other = [d for d in defs if not FACE.search(d) or NOT_FACE.search(d)]
    if other:
        return None, None
    return figure, defs


def facecheck(commit, figure, before="origin/main"):
    """Run the face check API on `commit` (compared with main) and keep its images and report in the project
    folder. Returns {ok, out, seconds, changed, notes, tail}."""
    wt = os.path.join(WORK, "fc-" + commit[:12])
    rq.git("worktree", "remove", "--force", wt, check=False)
    rq.git("worktree", "add", "--detach", wt, commit)
    try:
        if not os.path.exists(os.path.join(wt, FACECHECK)):          # the commit predates the API: borrow it
            src = next((r for r in FACECHECK_REFS if show(r, FACECHECK)), None)
            if not src:
                return {"ok": False, "tail": f"{FACECHECK} is not on any of {FACECHECK_REFS}"}
            os.makedirs(os.path.join(wt, os.path.dirname(FACECHECK)), exist_ok=True)
            open(os.path.join(wt, FACECHECK), "w").write(show(src, FACECHECK))
        out = os.path.join(FACECHECK_OUT, f"{figure}-{commit[:9]}")
        t0 = datetime.datetime.now()
        r = subprocess.run([sys.executable, os.path.join(wt, FACECHECK), figure, "--before", before, "--out", out],
                           capture_output=True, text=True, timeout=SMOKE_TIMEOUT)
        res = {"ok": r.returncode == 0, "out": out, "seconds": round((datetime.datetime.now() - t0).total_seconds()),
               "tail": "\n".join((r.stdout + r.stderr).strip().splitlines()[-8:]) if r.returncode else ""}
        if res["ok"]:
            rpt = json.load(open(os.path.join(out, "report.json")))
            res["changed"] = rpt.get("changed", [])
            res["notes"] = rpt["reports"]["now"].get("notes")
        return res
    finally:
        rq.git("worktree", "remove", "--force", wt, check=False)


# ------------------------------------------------------------------ the smoke run
def smoke_command(kind, param, out):
    q = shlex.quote
    runner = f"python3 {q(os.path.join(HERE, 'smoke.py'))}"
    if kind == "frames":
        return f"{runner} {q(os.path.join(RQ_DIR, 'rq_frames.py'))} --out {q(out)} --cells {q(param)}"
    if kind == "glb":
        return f"{runner} blender/ranger/export_glb.py --out {q(os.path.join(out, 'ranger.glb'))}"
    return f"{runner} {param} --out {q(out)}"


def hair_findings(rep, now, before, before_tip, override):
    """Turn a hair-check result into findings: clothes through the hair and flags the branch's last measured
    commit did not have send the request back; flags that were already there are listed as warnings."""
    def same(p, q):
        return p["kind"] == q["kind"] and p["facing"] == q["facing"] and abs(p["height"] - q["height"]) < 0.7 \
            and abs(p["x"] - q["x"]) < 0.9

    def line(p):
        return f"{p['kind']} in the {p['facing']} view, {p['pixels']} px at height {p['height']}, {p['side']} (x {p['x']:+.1f})"
    level = rep.warn if override else rep.block
    pic = f" Marked in {now['picture']} (red crease, blue ridge, yellow ledge)." if now.get("picture") else ""
    if now.get("note"):
        rep.info("hair", f"Hair check: {now['note']}.")
        return
    for c in now["clashes"]:
        level("hair", f"In walk column {c['walk_column']}, {c['object']} comes out through the hair "
              f"({c['faces']} faces, height {c['height'][0]} to {c['height'][1]}).",
              "Move the hair clear of it in that pose, or weight the hair there to follow the body.", "P015")
    probs = now["problems"]
    old = (before or {}).get("problems")
    if old is None:
        if probs:
            rep.warn("hair", f"Hair check, first measurement of this branch: {len(probs)} flag(s). "
                     + "; ".join(line(p) for p in probs[:6]) + ("; ..." if len(probs) > 6 else "") + "." + pic,
                     "A crease is a seam where two hair pieces meet: widen their overlap and relax the join after "
                     "fusing. A ridge is a piece's corner: round it or sink it. A ledge is hair lying on hair: "
                     "root it inside the mass it grows from.", "P015")
    else:
        new = [p for p in probs if p["pixels"] >= HAIR_NEW_PX and not any(same(p, q) for q in old)]
        grown = [(p, q) for p in probs for q in old if same(p, q) and p["pixels"] >= max(HAIR_NEW_PX, 1.6 * q["pixels"])]
        kept = [p for p in probs if p not in new and not any(p is g for g, _ in grown)]
        if new or grown:
            level("hair", f"This commit adds hair problems that {before_tip[:9]} (the last commit of this branch "
                  "measured) did not have: " + "; ".join([line(p) for p in new[:6]] +
                  [f"{line(p)}, up from {q['pixels']} px" for p, q in grown[:4]]) + "." + pic,
                  "A crease is a seam where two hair pieces meet: widen their overlap and relax the join after "
                  "fusing. A ridge is a piece's corner: round it or sink it. A ledge is hair lying on hair: root "
                  "it inside the mass it grows from. Check with RQ_HAIRCHECK=/tmp/h python3 "
                  "tools/render-review/smoke.py <view script> <args> --out /tmp/x before asking again.", "P015")
        if kept:
            rep.warn("hair", f"Hair check: {len(kept)} other flag(s), already there at {before_tip[:9]} or small: "
                     + "; ".join(line(p) for p in kept[:5]) + ("; ..." if len(kept) > 5 else "") + "." + pic, None, "P015")
        gone = [q for q in old if not any(same(p, q) for p in probs)]
        if gone:
            rep.info("hair", f"Hair check: {len(gone)} flag(s) from {before_tip[:9]} are gone.")
    if not probs and not now["clashes"]:
        rep.info("hair", "Hair check: no creases, ridges, ledges or clothes through the hair in any of the 8 facings.")
    for name, p in (now.get("paint") or {}).items():
        if p.get("half_transparent", 0) > HAIR_SEMI:
            rep.warn("hair", f"Strand texture '{name}': {p['half_transparent']:.0%} of its painted texels are "
                     "half-transparent, so the skin shows through and tints the strands.",
                     "Paint each strand opaque and thin the lock by the gaps between strands.", "P016")
        if p.get("tone_distance", 0) > HAIR_TONE:
            rep.warn("hair", f"Strand texture '{name}': its colour {p['mean_rgb']} sits away from the hair's ramp "
                     f"{p['hair_ramp_rgb']}, so the painted strands will not match the hair they grow from.",
                     "Take the strand tones from MATERIALS[M_HAIR]['ramp'] unchanged.", "P016")


def hair_before(branch, tip):
    """(result, tip) of the newest hair check of this branch on a commit `tip` builds on, or (None, None)."""
    try:
        index = json.load(open(os.path.join(HAIR_DIR, "index.json")))
    except (OSError, ValueError):
        return None, None
    # (entries from before baselines were kept per figure sit under the bare branch name)
    for e in reversed(index.get(branch, []) or index.get(branch.split("#")[0], [])):
        if e["tip"] != tip and rq.git("merge-base", "--is-ancestor", e["tip"], tip, check=False).returncode == 0:
            try:
                return json.load(open(os.path.join(e["dir"], "hair-check.json"))), e["tip"]
            except (OSError, ValueError):
                continue
    return None, None


def hair_record(branch, tip, folder):
    os.makedirs(HAIR_DIR, exist_ok=True)
    p = os.path.join(HAIR_DIR, "index.json")
    try:
        index = json.load(open(p))
    except (OSError, ValueError):
        index = {}
    rows = [e for e in index.get(branch, []) if e["tip"] != tip] + [{"tip": tip, "dir": folder}]
    index[branch] = rows
    json.dump(index, open(p, "w"), indent=1)


def smoke(commit, kind, param, hair_dir=None):
    """Run one job up to its first render in a worktree of `commit`. Returns {ok, seconds, tail, cached}.
    With hair_dir, the hair is checked in the same run (hair_check.py) and its result left there."""
    key = rq.job_key(commit, "smoke-" + kind, param)
    key += "-" + hashlib.sha1(open(os.path.join(HERE, "smoke.py"), "rb").read()).hexdigest()[:6]
    if REQ_ENV:
        key += "-e" + hashlib.sha1(json.dumps(REQ_ENV, sort_keys=True).encode()).hexdigest()[:6]
    memo = os.path.join(WORK, "smoke", key + ".json")
    need_hair = hair_dir and not os.path.exists(os.path.join(hair_dir, "hair-check.json"))
    if os.path.exists(memo) and not need_hair:
        return {**json.load(open(memo)), "cached": True}
    wt, out = os.path.join(WORK, "wt-" + key), os.path.join(WORK, "out-" + key)
    os.makedirs(out, exist_ok=True)
    rq.git("worktree", "remove", "--force", wt, check=False)
    rq.git("worktree", "add", "--detach", wt, commit)
    t0 = datetime.datetime.now()
    cmd = smoke_command(kind, param, out)
    try:
        env = dict(os.environ, **REQ_ENV, **({"RQ_HAIRCHECK": hair_dir} if need_hair else {})) if (need_hair or REQ_ENV) else None
        r = subprocess.run(cmd, shell=True, cwd=wt, capture_output=True, text=True, timeout=SMOKE_TIMEOUT, env=env)
        rc, log = r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        rc, log = -1, ((e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(errors="replace")
                       + f"\n[no render reached after {SMOKE_TIMEOUT}s]")
    rq.git("worktree", "remove", "--force", wt, check=False)
    subprocess.run(["rm", "-rf", out])
    res = {"ok": rc in (0, 77), "reached_render": rc == 77, "exit": rc,
           "seconds": round((datetime.datetime.now() - t0).total_seconds()),
           "tail": "\n".join(log.strip().splitlines()[-8:]) if rc not in (0, 77) else "",
           # (the lines the build's own hair checks print, blender/shared/locks.py report(): a line that does not
           # say 'clean' is a fault)
           "lock_lines": sorted({ln.split("hair lock check:", 1)[1].strip() for ln in log.splitlines()
                                 if "hair lock check:" in ln})}
    os.makedirs(os.path.dirname(memo), exist_ok=True)
    if rc != -1:
        json.dump(res, open(memo, "w"))
    return {**res, "cached": False}


# ------------------------------------------------------------------ the review
def review(req, thread, do_smoke=True):
    rep = Report()
    facts = {}
    branch, sha = req.get("branch"), req.get("sha") or None
    outs = req.get("outputs") or []
    if isinstance(outs, str):
        outs = [o for o in outs.split(",") if o]
    views = req.get("views") or []
    if isinstance(views, str):
        views = [views]
    cells = req.get("cells")

    # ---- shape
    if not branch or not isinstance(branch, str):
        rep.block("shape", "The request names no branch.", 'Send {"branch": "claude/project-thread-xxxxxx", ...}.')
        return rep, facts
    if req.get("ready") and not outs:
        rep.info("shape", "A 'ready' note for the scheduled sheet run: nothing to render now.")
        facts["ready_only"] = True
    if "sheet" in outs or any(re.search(r"full.?sheet|whole sheet", str(o), re.I) for o in outs):
        rep.block("shape", "The full sprite sheet is not a requestable output.",
                  'Ask for "frames" with the cells that show your change (for example "cells": "0-7:2,4,6"). The '
                  'full sheet runs only as the scheduled task; send {"branch": "...", "ready": true} once the work '
                  "is finished to be included in it.", "P001")
    bad = [o for o in outs if o not in rq.REQUESTABLE and o != "sheet"]
    if bad or (not outs and not req.get("ready")):
        rep.block("shape", f"outputs must be some of {list(rq.REQUESTABLE)}" + (f"; got {bad}" if bad else "") + ".",
                  'For example "outputs": ["frames", "views"].')
    if "views" in outs and not views:
        rep.block("shape", '"views" is requested but no view command is given.',
                  'Add "views": ["blender/base/views.py --figure rogue --head --clay"] or drop "views".')
    if views and "views" not in outs:
        rep.warn("shape", 'View commands are given but "views" is not in outputs, so they will not run.',
                 'Add "views" to outputs.')
    if "frames" in outs:
        if not cells:
            rep.block("shape", '"frames" is requested without "cells".', 'Add "cells": "0-7:2,4,6".')
        else:
            check_cells(rep, cells, "cells")
    if req.get("quality") not in (None, "check", "final"):
        rep.block("shape", f'"quality" is {req.get("quality")!r}; the queue reads only "check" or "final".',
                  'Use "quality": "check" for work in progress on a 3D look (8 samples, denoised, views only) and '
                  'leave it out, or say "final", for anything John signs off on.')
    if req.get("mode") == "isolated":
        rep.info("shape", "mode: isolated is ignored; the queue combines every branch whose code does not overlap.")
    note = str(req.get("note") or "")
    if re.search(r"\bbefore\b|\bbaseline\b", note, re.I):
        rep.warn("before", f"The note mentions a before or baseline ({note[:80]!r}). A before is never rendered, and "
                 "must not be faked on the branch by hiding or switching off the new work.",
                 "Ask only for the after views and compare them with a cached render of main "
                 "(the queue lists them, or look under render-queue/results/).", "P002")

    parsed = []
    for v in views:
        if not isinstance(v, str) or SHELL_META.search(v) or ".." in v:
            rep.block("views", f"View command {v!r} holds shell characters or '..'. It is run by a shell as "
                      "written.", "Give only a script path inside the repo and its arguments.", "P012")
            continue
        try:
            tok = shlex.split(v)
        except ValueError as e:
            rep.block("views", f"View command {v!r} does not parse: {e}.", "Balance the quotes.")
            continue
        if tok and tok[0] in ("pose-strip", "crop"):
            # the queue's own views (README "Crops and the pose test strip"): checked as the views.py build they run on
            rest = tok[1:]
            if tok[0] == "crop" and rest and not rest[0].startswith("--"):
                rest = rest[1:]                                  # the region list
            clean, skip = [], 0
            for t in rest:
                if skip:
                    skip -= 1
                elif t in ("--poses", "--pose-angle", "--pose-crop", "--crop-size", "--crop-angles", "--crop"):
                    skip = 1
                else:
                    clean.append(t)
            if "--angles" not in clean:
                clean += ["--angles", "3/4"]
            tok = ["blender/base/views.py"] + clean + ["--turn"]
            v = " ".join(shlex.quote(t) for t in tok)
        if not tok or not tok[0].endswith(".py") or os.path.isabs(tok[0]):
            rep.block("views", f"View command {v!r} must start with a .py path relative to the repo root.",
                      "For example blender/base/views.py --figure rogue --head.")
            continue
        m = re.fullmatch(r"blender/(\w+)/(\w+)\.py", tok[0])
        if m and m[1] == m[2] and "--build-only" not in tok:
            rep.block("views", f"{tok[0]} is a character script: run as a view it renders the full sprite sheet.",
                      "Use a view script (blender/base/views.py, sculpt_views.py, face_views.py) or ask for "
                      '"frames".', "P001")
            continue
        if "--out" in tok:
            rep.block("views", f"View command {v!r} sets --out; the queue adds its own.", "Remove --out.")
            continue
        if "--cells" in tok and tok.index("--cells") + 1 < len(tok):
            check_cells(rep, tok[tok.index("--cells") + 1], f"{tok[0]} --cells")
        parsed.append((v, tok))

    # ---- commit
    rq.git("fetch", "--quiet", "--prune", "origin", check=False)
    main = rq.git("rev-parse", "origin/main").stdout.strip()
    tip_of_branch = rq.git("rev-parse", "--verify", "--quiet", f"origin/{branch}", check=False).stdout.strip() \
        if branch != "main" else main
    if not tip_of_branch:
        rep.block("commit", f"Branch {branch} is not on origin.", "Push the branch, then send the request again.",
                  "P006")
        return rep, facts
    tip = tip_of_branch
    if sha:
        tip = rq.git("rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}", check=False).stdout.strip()
        if not tip:
            rep.block("commit", f"Commit {sha} is not on origin (it may not be pushed yet).",
                      "Push it, or leave sha out to use the branch tip.", "P006")
            return rep, facts
        if rq.git("merge-base", "--is-ancestor", tip, tip_of_branch, check=False).returncode != 0:
            rep.warn("commit", f"Commit {tip[:9]} is not on {branch} (tip {tip_of_branch[:9]}).",
                     "Check the sha; a rebased branch leaves old commits behind.")
        elif tip != tip_of_branch:
            rep.info("commit", f"The request is pinned to {tip[:9]}; the branch tip is now {tip_of_branch[:9]}.")
    facts.update(tip=tip, main=main)
    if facts.get("ready_only") or (rep.blocked and not outs):
        return rep, facts

    # ---- before: main's own code is served from the cache only (the queue's rule, applied here first)
    on_main = branch == "main" or rq.git("merge-base", "--is-ancestor", tip, main, check=False).returncode == 0
    if on_main and all(o in rq.REQUESTABLE for o in outs):
        try:
            rq.before_only_from_cache(types.SimpleNamespace(branch=branch, sha=tip, view=views, cells=cells), outs)
            rep.info("before", "This commit is already on main and every job is cached: answered at once.")
        except SystemExit as e:
            rep.block("before", str(e), "Compare against one of the cached renders listed above; do not ask for a "
                      "new render of main. For a face, the face check API draws a before for free: "
                      f"python {FACECHECK} <figure> --before origin/main", "P002")
            return rep, facts

    # ---- merge with main
    merged, conflicts = rq.onto_main(branch, tip, main)
    if not merged:
        semantic = [c for c in conflicts if ": " in c]
        if semantic:
            rep.block("merge", "After merging main, " + "; ".join(semantic) + ". Python keeps only the later "
                      "definition, for every caller.", "Keep one definition (rename the other if both are needed), "
                      "push, and send the request again.", "P005")
        else:
            t = rq.git("merge-tree", "--write-tree", "--no-messages", main, tip, check=False).stdout.splitlines()[0]
            hunks = {p: (show(t, p) or "").count("\n<<<<<<<") for p in conflicts}
            rep.block("merge", "The branch conflicts with main in " +
                      ", ".join(f"{p} ({n} hunk{'s' if n != 1 else ''})" for p, n in hunks.items()) + ".",
                      "Run git fetch origin main && git merge origin/main, resolve those hunks, push, and send the "
                      "request again. If the right resolution depends only on the queue's judgment, say so in the "
                      "note and the request is forwarded for the queue to resolve.", "P006")
            facts["conflict"] = hunks
        return rep, facts
    facts["merged"] = merged
    base = rq.git("merge-base", main, tip).stdout.strip()
    behind = int(rq.git("rev-list", "--count", f"{tip}..{main}").stdout.strip() or 0)
    if behind:
        rep.info("merge", f"The branch is {behind} commit(s) behind main and merges cleanly; the render uses the "
                 "merged code.")

    unparsable = [h for h in rq.duplicate_defs_all(merged) if h.endswith("does not parse")]
    for h in unparsable:
        path = h.split(":")[0]
        try:
            ast.parse(show(merged, path) or "")
            err = ""
        except SyntaxError as e:
            err = f" (line {e.lineno}: {e.msg})"
        rep.block("merge", f"{path} does not parse{err}.", "Fix the syntax error and push.", "P003")
    on_main_dups = [h for h in rq.duplicate_defs_all(main) if not h.endswith("does not parse")]
    for h in on_main_dups:
        rep.info("merge", f"main itself has {h} (not this branch's doing).")

    new_names = {}
    names, main_names = flake_names(merged), flake_names(main)
    if names is None:
        rep.info("merge", "pyflakes is not installed here: undefined names were not checked.")
    else:
        new_names = {n: w for n, w in names.items() if n not in main_names}

    fp = rq.footprint(main, merged)
    prev = last_rendered(branch, tip)
    since = prev or base
    fresh = rq.footprint(since, tip)          # what changed since this branch was last looked at
    facts["changes"] = {p: sorted(d) for p, d in fp.items()}
    if not fp:
        rep.info("merge", "The branch changes no render input compared with main.")
    carried = [p for p in KEEP_AT_MAIN
               if rq.git("diff", "--quiet", main, tip, "--", p, check=False).returncode != 0]
    if carried and tip != main:
        rep.warn("merge", "The branch carries its own " + ", ".join(carried) + ".",
                 "Keep them at main's version: git checkout origin/main -- " + " ".join(carried), "P007")

    # ---- view scripts and their flags
    for v, tok in parsed:
        src = show(merged, tok[0])
        if src is None:
            near = [p for p in rq.git("ls-tree", "-r", "--name-only", merged, "--", "blender/").stdout.split()
                    if os.path.basename(p) == "views.py"]
            rep.block("views", f"{tok[0]} is not in the commit.", "View scripts in this commit: " + ", ".join(near),
                      "P010")
            continue
        known = flags_known_to(src)
        if known:
            unknown = [t for t in tok[1:] if t.startswith("--") and t.split("=")[0] not in known]
            if unknown:
                rep.block("views", f"{tok[0]} does not read {', '.join(unknown)}. A script that parses its own "
                          "arguments ignores an unknown flag without a word and renders something else.",
                          f"Flags it reads: {' '.join(sorted(known - {'--out'}))}", "P010")
        # (a painted texture changes colour, not form: clay would show nothing new)
        hair = sorted({d for defs in fresh.values() for d in defs
                       if HAIR.search(d) and not re.search(r"(_image|_tex|_painting|_paint|_shade|_material|TEX)$", d)})
        if hair and known and "--clay" in known and not any("--clay" in t for _, t in parsed):
            rep.warn("views", "This commit changes hair code (" + ", ".join(hair[:6]) + ") but no view asks for "
                     "--clay. Seams, ledges and ridges where hair meets the head only show in clay.",
                     "Add --head --clay to the view command.", "P009")
            break

    # ---- a lit command beside its own --clay twin (--clay renders the lit views first, then the clay ones)
    cmds = [shlex.split(v) for v in views if isinstance(v, str)]
    for c in cmds:
        if "--clay" in c and [t for t in c if t != "--clay"] in [o for o in cmds if "--clay" not in o]:
            rep.block("views", "The same view command is asked with and without --clay. --clay already writes the "
                      "lit views (turn.png, head.png) as well as the clay ones, so the second command builds and "
                      "renders the lit views twice.", "Keep only the --clay command.", "P022")
            break

    # ---- pitfall patterns against the lines the branch adds
    lines = added_lines(main, merged) if fp else []
    for p in read_pitfalls():
        if not p["pattern"]:
            continue
        hits = [(path, ln, t) for path, ln, t in lines if re.search(p["pattern"], t)
                and not (p["unless"] and re.search(p["unless"], show(merged, path) or ""))]
        if hits:
            rep.warn("pitfalls", f"{p['id']} {p['title']}: " +
                     "; ".join(f"{path}:{ln} `{t.strip()[:90]}`" for path, ln, t in hits[:4]) +
                     (f" (+{len(hits) - 4} more)" if len(hits) > 4 else ""), p["fix"], p["id"])

    # ---- a material change renders the material, not the figure (John's rule, 2026-11-17; P024)
    mat_defs = material_only(since, tip)
    figure_jobs = [v for v, tok in parsed if tok and os.path.basename(tok[0]) == "views.py"]
    if mat_defs and figure_jobs:
        if not req.get("needs_render"):
            rep.block("material", "Since this branch was last rendered only material code changed (" +
                      ", ".join(mat_defs[:6]) + "), and the request builds the whole figure to show it.",
                      "Ask for a swatch render: a script in your branch that puts the material on a sphere and a "
                      "cylinder or slab at the piece's size, under the figure's own light and world, with every look "
                      "side by side in one picture from one run and no figure build. If the material must be seen on "
                      'the figure, add "needs_render": "<what you need to see>" and ask for one figure job.', "P024")
        elif len(figure_jobs) > 1:
            rep.block("material", f"A material-only change with needs_render runs as one figure job, not "
                      f"{len(figure_jobs)}.", "Keep one figure command; put the other looks on a swatch.", "P024")

    # ---- one view first, then the turn (P025); clothes go on a mannequin (P026). John's rules, 2026-11-17
    changed = sorted({d for path, defs in fresh.items() if path.startswith("blender/") for d in defs})
    hairy = any(HAIR.search(d) for d in changed)
    for v, tok in parsed:
        if not tok or not os.path.basename(tok[0]) == "views.py":
            continue
        known = flags_known_to(show(merged, tok[0]) or "") or set()
        one = "--angles" in tok and tok.index("--angles") + 1 < len(tok) \
            and "," not in tok[tok.index("--angles") + 1]
        multi = [f for f in ("--turn", "--head", "--pieces") if f in tok]
        def tval(flag):
            return tok[tok.index(flag) + 1] if flag in tok and tok.index(flag) + 1 < len(tok) else ""
        preset = tval("--preset")
        try:
            cost = int(tval("--size") or 360) ** 2 * int(tval("--samples") or 32)
        except ValueError:
            cost = 0
        if multi and cost > 40_000_000 and "--angles" not in tok:
            rep.warn("views", f"A five-angle {' '.join(multi)} at size {tval('--size') or 360} and {tval('--samples') or 32} samples "
                     f"({cost / 1e6:.0f} million sample-pixels per angle). The queue killed a 720 px, 128 sample turn at its "
                     "30-minute limit; 600 px with 48 samples and --denoise finished.",
                     "Use --size 600 --samples 48 --denoise for a final turn, or ask for one angle at the larger size.", "P028")
        if preset in ("turn-clay", "turn-matcap", "turn-lit", "head-clay"):   # four angles each (view_tools.PRESETS)
            multi.append("--preset " + preset)
        if "--preset" in tok:
            bad = []
            if preset not in PRESETS:
                bad.append(f"--preset {preset or '(none)'} (one of {', '.join(sorted(PRESETS))})")
            if tval("--tier") and tval("--tier") not in ("draft", "check", "final"):
                bad.append(f"--tier {tval('--tier')} (draft, check or final)")
            for lk in [x for x in tval("--look").split(",") if x]:
                if lk not in ("full", "clay", "matcap"):
                    bad.append(f"--look {lk} (full, clay or matcap)")
            if preset == "closeup" and not tval("--area"):
                bad.append("--preset closeup without --area")
            for ar in [x for x in tval("--area").split(",") if x]:
                if ar not in AREAS and not ar.startswith("piece:"):
                    bad.append(f"--area {ar} (one of {', '.join(AREAS)}, or piece:PREFIX)")
            if bad:
                rep.block("views", "views.py preset flags the presets do not know: " + "; ".join(bad) + ".",
                          "Use the names in blender/base/view_tools.py (PRESETS, TIERS, AREA_NAMES).", "P010")
        if not any(f in tok for f in ("--turn", "--head", "--pieces", "--cells", "--list", "--walk", "--run", "--save", "--preset")):
            rep.block("views", "views.py command with none of --turn, --head, --pieces, --cells, --list, --walk, --run, --save or --preset: it builds "
                      "the figure and renders nothing (--angles only picks the angles of --turn, --head and --pieces).",
                      "Add --turn (one view: --turn --angles front) or --head, --pieces, --cells, --walk N or --run N.", "P027")
        if multi and not one and not str(req.get("confirmed") or "").strip():
            if "--angles" in known:
                rep.block("views", f"All five angles of {' '.join(multi)} with no \"confirmed\" in the request.",
                          "Render one angle first, the one that shows the change best: add --angles front (or 3/4, "
                          "profile, back3/4, back) to the command. When it is right, ask for all angles with "
                          '"confirmed": "<what the one view showed>".', "P025")
            else:
                rep.warn("views", "All five angles with no \"confirmed\". views.py at this commit has no --angles yet, "
                         "so one view cannot be asked for; let through with the gap noted.",
                         "Merge the commit that adds --angles and ask for one angle first.", "P025")
        clothes = changed and not hairy and not mat_defs and "--head" not in tok and "--face" not in tok \
            and "--mannequin" not in tok and not req.get("needs_render")
        if clothes:
            pieces = "--only" in tok or "--no-body" in tok
            if "--mannequin" in known and not pieces:
                rep.block("views", "A clothes change rendered on the real figure, which builds the face sculpt, the "
                          "face paints, the ears and the hair to show cloth.",
                          "Add --mannequin to the command (the body shape with a plain head). Ask for the real figure "
                          'only with "needs_render": "<why the hair or face must be there>".', "P026")
            elif "--mannequin" in known:
                rep.warn("views", "Pieces alone still build the whole figure under them.",
                         "Add --mannequin: the pieces fit the same and the face and hair are not built.", "P026")
            else:
                rep.warn("views", "A clothes change on the real figure. views.py at this commit has no --mannequin "
                         "yet; let through with the gap noted.", "Merge the commit that adds --mannequin.", "P026")
        break

    # ---- a face-only request never reaches the queue: the face check API answers it here
    figure, face_defs = face_only(fresh, outs, parsed)
    if figure and req.get("needs_render"):
        rep.info("facecheck", f"Face-only change, forwarded to the queue because the request says a render is "
                 f"needed: {str(req['needs_render'])[:120]}")
    elif figure and not rep.blocked:
        res = facecheck(merged, figure) if do_smoke else {"ok": None}
        facts["facecheck"] = {"figure": figure, "defs": face_defs, **res}
        if res["ok"] is False:
            rep.block("facecheck", f"The face check of {figure} fails on this commit:\n{res['tail']}",
                      f"Fix the error and push. Reproduce: python {FACECHECK} {figure} --before origin/main", "P014")
        else:
            rep.add("facecheck", "facecheck", f"This request only checks the face of {figure} (changed: "
                    + ", ".join(face_defs[:8]) + ("..." if len(face_defs) > 8 else "") + "). Face checks are not "
                    "rendered: the face check API drew and measured it instead"
                    + (f", in {res['seconds']}s: {res['out']}/ (sheet.png, one PNG per view, report.json)."
                       if res["ok"] else "."),
                    f"Next time run it yourself: python {FACECHECK} {figure} --before origin/main. If you need "
                    'what it cannot show (hair, cloth, final lighting, subsurface scattering, the sprite palette), '
                    'send the request again with "needs_render": "<what you need to see>".', "P014")
        return rep, facts

    # ---- jobs: cached already, or smoke-run
    jobs = [(o, v) for o in outs if o in rq.REQUESTABLE
            for v in ([t for t, _ in parsed] if o == "views" else [cells if o == "frames" else None])]
    facts["jobs"] = []
    todo = []
    for kind, param in jobs:
        key = rq.job_key(merged, kind, param)
        j = {"key": key, "kind": kind, "param": param, "cached": rq.cached(key), "failed_before": rq.failed(key)}
        facts["jobs"].append(j)
        if j["cached"]:
            rep.info("cache", f"{kind} {param or ''} is already rendered: results/{key}/ (answered at once).")
        elif j["failed_before"]:
            rep.block("cache", f"{kind} {param or ''} already failed for this exact code: results/{key}-failed/"
                      "render.log.", "Fix what the log shows and push a new commit; the same code is not rendered "
                      "again.", "P011")
        elif not rep.blocked:
            todo.append(j)
    smoked_ok = False
    # the hair check rides on the first job's build, for every request whose commit touches hair code
    hair_defs = sorted({d for defs in fresh.values() for d in defs if HAIR.search(d)})
    # (measured on the figure the first job builds, and compared only with that figure's last measurement: a branch
    # that carries two figures has two baselines)
    # (a view that hides part of the figure, --hair-only, --only, --hide, --no-body, is no basis for the check: with
    # the clothes hidden nothing can come through the hair and the hair's hidden parts count as flags)
    whole = [j for j in (todo or facts["jobs"]) if not re.search(r"--(hair-only|only|hide|no-body)\b", j.get("param") or "")]
    first = (whole or [{}])[0]
    m = re.search(r"--figure\s+(\w+)", first.get("param") or "") or re.search(r"blender/(\w+)/", first.get("param") or "")
    figure_ = m.group(1) if m and m.group(1) != "base" else "ranger"
    hair_key = f"{branch}#{figure_}"
    hair_dir = os.path.join(HAIR_DIR, f"{tip[:9]}-{merged[:9]}-{figure_}") if hair_defs and whole else None
    if hair_dir and not todo and not rep.blocked and do_smoke and not os.path.exists(os.path.join(hair_dir, "hair-check.json")) \
            and facts["jobs"]:
        todo.append(first)          # the renders are cached, but this hair was never measured
    if do_smoke and todo:
        with concurrent.futures.ThreadPoolExecutor(2) as ex:
            for j, res in zip(todo, ex.map(lambda j: smoke(merged, j["kind"], j["param"],
                                                           hair_dir if j is first else None), todo)):
                j["smoke"] = res
                faults = [ln for ln in res.get("lock_lines", []) if " clean (" not in ln]
                if faults:
                    rep.block("hair", f"{j['kind']} {j['param'] or ''}: the build's own hair lock check reports "
                              + "; ".join(faults) + ".",
                              "Fix what the line names (kink: smooth that lock's guide points so no one point bends it sharply; inward: rewind the strand faces; glint: give the strands "
                              "the cap's outward normal; facing: flip normals that point at the head; bare: add "
                              "locks or widen the cap where skin shows inside the hairline) and push.", "P021")
                elif res.get("lock_lines"):
                    rep.info("hair", f"{j['kind']} {j['param'] or ''}: hair lock check " + "; ".join(res["lock_lines"]) + ".")
                if res["ok"]:
                    rep.info("smoke", f"{j['kind']} {j['param'] or ''}: built headless and "
                             f"{'reached its first render' if res['reached_render'] else 'finished'} in "
                             f"{res['seconds']}s" + (" (checked earlier)" if res["cached"] else "") + ".")
                else:
                    rep.block("smoke", f"{j['kind']} {j['param'] or ''} fails before its first render "
                              f"(exit {res['exit']}, {res['seconds']}s):\n{res['tail']}",
                              "Fix the error in the last lines above and push. Reproduce without rendering: "
                              f"python3 tools/render-review/smoke.py {j['param'] or '<script> <args>'} --out /tmp/x",
                              "P003")
        smoked_ok = all(j["smoke"]["ok"] for j in todo)
        if hair_dir and os.path.exists(os.path.join(hair_dir, "hair-check.json")):
            now = json.load(open(os.path.join(hair_dir, "hair-check.json")))
            before, before_tip = hair_before(hair_key, tip)
            facts["hair"] = {"dir": hair_dir, "changed": hair_defs, "problems": len(now.get("problems", [])),
                             "clashes": len(now.get("clashes", [])), "compared_with": before_tip}
            hair_findings(rep, now, before, before_tip, bool(req.get("needs_render")))
            hair_record(hair_key, tip, hair_dir)
        elif hair_dir and smoked_ok:
            err = os.path.join(hair_dir, "hair-check-error.txt")
            rep.warn("hair", "This commit changes hair code (" + ", ".join(hair_defs[:6]) + ") but the hair check "
                     "left no result." + (" Its error: " + open(err).read().strip().splitlines()[-1] if os.path.exists(err) else ""), "Look at the build log; run RQ_HAIRCHECK=/tmp/h python3 "
                     "tools/render-review/smoke.py <view script> <args> --out /tmp/x.")
    elif todo:
        rep.info("smoke", "Smoke run skipped.")
    for n, where in sorted(new_names.items()):
        text = f"{where}: {n} is used but never defined (main does not have this)."
        fix = f"Define {n}, or restore what a merge or a move between files dropped."
        if smoked_ok:
            rep.warn("merge", text + " The build did not reach it this time.", fix, "P003")
        else:
            rep.block("merge", text + " The render stops with a NameError when it gets there.", fix, "P003")

    # ---- what the reviewer should read: the change since this branch was last rendered
    facts["read"] = {"since": since[:9], "since_is": "last rendered commit of this branch" if prev else "merge base with main",
                     "stat": rq.git("diff", "--stat", since, tip, "--", *rq.INPUTS, check=False).stdout.strip().splitlines()[-1:],
                     "diff_command": f"git -C {rq.REPO} diff {since[:9]} {tip[:9]} -- {' '.join(rq.INPUTS)}"}
    same = [r["id"] for r in rq.load()["requests"] if r["state"] in ("pending", "rendering")
            and r.get("tip") == tip and r["outputs"] == outs and r.get("views", []) == views and r.get("cells") == cells]
    if same:
        rep.block("cache", f"The queue is already working on this exact request (#{', #'.join(map(str, same))}).",
                  "Wait for its reply; a second copy would only be merged into it.", "P011")
    return rep, facts


def cmd_check(a):
    try:
        req = json.loads(a.request if a.request != "-" else sys.stdin.read())
        REQ_ENV.update({str(k): str(v) for k, v in (req.get("env") or {}).items()} if isinstance(req, dict) else {})
        if not isinstance(req, dict):
            raise ValueError("the request must be one JSON object")
    except ValueError as e:
        rep, facts, req = Report(), {}, {}
        rep.block("shape", f"The request is not valid JSON: {e}.",
                  'Send one JSON block: {"branch": ..., "outputs": [...], "cells": ..., "views": [...], "note": ...}')
    else:
        rep, facts = review(req, a.thread, do_smoke=not a.no_smoke)
    verdict = "fix" if rep.blocked else "facecheck" if any(f["level"] == "facecheck" for f in rep.findings) else "pass"
    forward = None
    if verdict == "pass":
        forward = {k: v for k, v in req.items() if v not in (None, "", [])}
        forward["sha"] = facts.get("tip", forward.get("sha"))     # the commit that was reviewed is the one rendered
        forward["for_thread"] = a.thread
        forward["reviewed"] = True
    pitfalls = read_pitfalls()
    result = {"verdict": verdict, "thread": a.thread, "request": req, "findings": rep.findings, "facts": facts,
              "forward": forward, "pitfalls_read": len(pitfalls),
              "at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a") as f:
            f.write(json.dumps({k: result[k] for k in ("at", "verdict", "thread", "request")}
                               | {"findings": [f for f in rep.findings if f["level"] != "info"],
                                  "tip": facts.get("tip")}) + "\n")
    except OSError:
        pass
    if a.json:
        print(json.dumps(result, indent=1)); return
    what = {"pass": "forward to the render queue", "fix": "send back with the fixes below",
            "facecheck": "do not forward: answer the requester with the face check below"}[verdict]
    print(f"VERDICT: {verdict.upper()}  ({what})")
    fc = facts.get("facecheck") or {}
    if fc.get("ok"):
        print("\nface check: measurements changed since main:\n  " + ("\n  ".join(fc["changed"]) or "nothing measured changed")
              + (f"\n  notes: {fc['notes']}" if fc.get("notes") else ""))
    for level in ("block", "facecheck", "warn", "info"):
        for f in rep.findings:
            if f["level"] == level:
                print(f"\n[{level}] {f['check']}" + (f" ({f['pitfall']})" if f["pitfall"] else "") + f": {f['text']}")
                if f["fix"]:
                    print(f"    fix: {f['fix']}")
    if facts.get("changes"):
        print("\nchanges vs main:")
        for p, d in facts["changes"].items():
            print(f"  {p}: {', '.join(d[:12])}{' ...' if len(d) > 12 else ''}")
    if facts.get("read"):
        r = facts["read"]
        print(f"\nread before deciding (since the {r['since_is']}, {r['since']}): {' '.join(r['stat'])}\n  {r['diff_command']}")
    print(f"\npitfalls to keep in mind ({len(pitfalls)}, from {PITFALLS}):")
    for p in pitfalls:
        print(f"  {p['id']} {p['title']}")
    if forward:
        print("\nforward to the render-queue thread:\n" + json.dumps(forward, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--thread", required=True, help="the requesting thread's id")
    c.add_argument("--request", required=True, help="the request's JSON, or - to read it from stdin")
    c.add_argument("--no-smoke", action="store_true"); c.add_argument("--json", action="store_true")
    sub.add_parser("pitfalls")
    q = sub.add_parser("learn")
    q.add_argument("--title", required=True); q.add_argument("--spot", required=True)
    q.add_argument("--fix", required=True); q.add_argument("--pattern"); q.add_argument("--unless")
    q.add_argument("--seen")
    a = p.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
