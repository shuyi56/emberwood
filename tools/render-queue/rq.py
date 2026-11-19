#!/usr/bin/env python3
"""Render queue for shuyi56/emberwood: the only thing in the project that renders the ranger.

Threads never render. They message the render-queue thread, which records each request here, plans it, renders,
and sends the requester its results. Nothing is pushed anywhere: results are files under results/<key>/.

    rq.py add --thread ID --branch BR [--sha SHA] --outputs frames,views,glb
              [--cells ROWS:COLS] [--view "path/in/repo.py ARGS"]... [--mode combined|isolated] [--note TEXT]
    rq.py ready --branch BR       the branch is done: include it in the next scheduled sheet run
    rq.py sheet-due               called by the scheduled task: the full sheets may run once the queue is idle
    rq.py list | plan | cancel ID
    rq.py serve                   run jobs until one request (or one sheet) is finished, then exit
    rq.py resolve BRANCH / resolved BRANCH   merge main into BRANCH in a worktree; commit the hand resolution

Job kinds and priority (lower runs first; ties go to the older request)
    0 frames  a few cells of the sprite sheet (rq_frames.py), e.g. 0-7:2,4,6
    0 views   a close-up script that lives in the branch (e.g. sculpt_views.py)
    1 glb     the world model (export_glb.py)
    9 sheet   the full sheet + .blend + glb (ranger.py, export_glb.py). Never requested by a thread, never run per
              merge. Only the scheduled task marks it due; it then runs for main and the branches marked ready, only
              while nothing else is waiting, and is stopped (to rerun later) the moment a quicker job arrives.

Planning rules
  * Generated files (.png .blend .glb ...) never count as conflicts: they are re-rendered, so a merge takes main's.
  * A branch with code conflicts against main is held as 'conflict' (files and hunk counts) until it is rebased or
    resolved here with rq.py resolve; a resolution is written as a git bundle for the owner under resolutions/.
  * Clean requests are stacked onto main into one batch while they merge cleanly with each other AND change
    different top-level functions (two clean edits of one function can still give a model neither thread meant).
    Overlapping ones get their own batch. Requests with no code changes are never combined. mode=isolated is
    accepted but ignored: every branch is combined whenever its code does not overlap.
  * Every job is cached by the hash of the script inputs plus its parameters: the same code is never rendered twice.

Batching rules (John, 2026-11-19)
  * Sets: pieces of one set (sets.json) waiting at the same time render together, always: one build of the body with
    every piece on it (rq_outfit.py, run as views.py --figure outfit), in wearing order, delivered to each piece's
    thread. Angles, views and looks are the union; size and samples the largest.
  * One build per scene: views.py jobs of one figure on one commit, with the same body (--mannequin), pose (--frame),
    env and quality, run as one views.py call when that renders no more images than the separate calls would.
  * Drafts with drafts, finals with finals: a check render never shares a job with a final; checks run first.
  * Identical requests are one job (same key), whatever batch they come from; the cache serves repeats.
  * Coalescing: a new request for the same branch and the same views replaces one still waiting (pinned or not).
  * Intent: a thread says what it is about to send (rq.py intent) before it does the work. A waiting job that an
    open intent could join (same figure, or a piece of the same set) is held for it: the collation window, 5 minutes
    (READY_CUTOFF_S) from the first piece ready (its request). When it ends the batch renders with whichever pieces
    are ready; a slower piece goes in the next batch. When every announced piece has come, it starts at once.
  * Without an intent a job still waits WINDOW_S (45 s) after its newest request for a related one, never longer.
  * Caps: at most MAX_BATCH (4) branches in a merged commit and MAX_MERGE (6) requests in a combined job. A combined
    job that fails (not one killed at 30 minutes) is retried with each of its requests on its own.
"""
import argparse, glob, ast, hashlib, json, os, re, shlex, shutil, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("RQ_HOME", HERE)   # queue.json, results/, events.jsonl: next to the code unless RQ_HOME says
STATE = os.path.join(DATA, "queue.json")
RESULTS = os.path.join(DATA, "results")
RESOLUTIONS = os.path.join(DATA, "resolutions")
REPO = os.environ.get("RQ_REPO", "/home/user/emberwood")
WORK = os.environ.get("RQ_WORK", "/tmp/rq-work")
GEN = (".png", ".jpg", ".webp", ".blend", ".blend1", ".glb", ".gltf", ".bin", ".exr")
INPUTS = ("blender/", "src/shared/core.js")   # what a ranger render reads
# under blender/ but never read by a render: left out of the job key, so adding them does not invalidate the cache
NOT_INPUTS = ("blender/facecheck/",)
RENDERED = ("blender/ranger/ranger-blender.png",)   # the pngs the queue itself renders
REQUESTABLE = ("frames", "views", "glb")
PRIORITY = {"frames": 0, "views": 0, "glb": 1, "sheet": 9}
MAX_BATCH = int(os.environ.get("RQ_MAX_BATCH", "4"))      # branches merged into one commit
MAX_MERGE = int(os.environ.get("RQ_MAX_MERGE", "6"))      # requests in one combined views.py job
WINDOW_S = int(os.environ.get("RQ_WINDOW", "45"))         # a job waits this long after its newest request
READY_CUTOFF_S = int(os.environ.get("RQ_READY_CUTOFF", "300"))   # John: a 5-minute collation window from the first piece ready
INTENT_TTL_S = 1800                                       # an intent nobody followed lapses after its ETA + this
VIEWS_PY = "blender/base/views.py"
OUTFIT = os.path.join(HERE, "rq_outfit.py")
ANGLES = ("front", "3/4", "profile", "back3/4", "back")
# views.py options a combined job can take the union of (True: takes a value); any other option keeps a job apart
MERGEABLE = {"--figure": True, "--angles": True, "--size": True, "--samples": True, "--frame": True,
             "--mannequin": False, "--turn": False, "--head": False, "--clay": False, "--clay-only": False,
             "--denoise": False}


def sets():
    for d in (DATA, HERE):
        try:
            return {k: v for k, v in json.load(open(os.path.join(d, "sets.json"))).items() if not k.startswith("_")}
        except (OSError, ValueError):
            continue
    return {}


def ts(t):
    import calendar
    return calendar.timegm(time.strptime(t, "%Y-%m-%dT%H:%M:%SZ"))


def parse_view(param):
    """A views.py job's parts, or None when it is not one a combined job can take."""
    check = param.endswith(CHECK)
    words = shlex.split(param[:-len(CHECK)] if check else param)
    env = []
    if words and words[0] == "env":
        words = words[1:]
        while words and "=" in words[0] and not words[0].startswith("-"):
            env.append(words.pop(0))
    if not words or words[0] != VIEWS_PY:
        return None
    a, i = {}, 1
    while i < len(words):
        w = words[i]
        if w not in MERGEABLE or (MERGEABLE[w] and i + 1 >= len(words)):
            return None
        a[w] = words[i + 1] if MERGEABLE[w] else True
        i += 2 if MERGEABLE[w] else 1
    if "--figure" not in a or not ("--turn" in a or "--head" in a) or any(e.startswith("OUTFIT") for e in env):
        return None
    return {"env": env, "check": check, "a": a, "group": (tuple(sorted(env)), check, "--mannequin" in a,
                                                          a.get("--frame"))}


def view_images(a):
    ang = len(a["--angles"].split(",")) if "--angles" in a else 5
    return (("--turn" in a) + ("--head" in a)) * ang * (1 if "--clay-only" in a else 2 if "--clay" in a else 1)


def union_args(parts, figure):
    a = {"--figure": figure}
    for f in ("--mannequin", "--turn", "--head", "--denoise"):
        if any(f in p for p in parts):
            a[f] = True
    if all("--clay-only" in p for p in parts):
        a["--clay-only"] = True
    elif any("--clay" in p or "--clay-only" in p for p in parts):
        a["--clay"] = True
    if all("--angles" in p for p in parts):
        want = {x for p in parts for x in p["--angles"].split(",")}
        a["--angles"] = ",".join([x for x in ANGLES if x in want] + sorted(want - set(ANGLES)))
    for k in ("--size", "--samples"):
        vals = [int(p[k]) for p in parts if k in p]
        if vals:
            a[k] = str(max(vals))
    if parts[0].get("--frame"):
        a["--frame"] = parts[0]["--frame"]
    return a


def view_param(env, a, check):
    words = [VIEWS_PY]
    for k, v in a.items():
        words += [k] if v is True else [k, v]
    return ("env " + " ".join(env) + " " if env else "") + shlex.join(words) + (CHECK if check else "")


def combine_views(bjobs, solo):
    """John's batching rules on one batch's views.py jobs: pieces of one set in one build, and one build per
    figure where that renders no more than the separate builds would."""
    keep, pool = [], []
    for j in bjobs:
        pv = parse_view(j["param"]) if j["kind"] == "views" and not (set(j["requests"]) & solo) else None
        (pool if pv else keep).append((j, pv))
    out = [j for j, _ in keep]
    groups = {}
    for j, pv in pool:
        groups.setdefault(pv["group"], []).append((j, pv))
    for g, items in groups.items():
        # one build per figure
        by_fig = {}
        for j, pv in items:
            by_fig.setdefault(pv["a"]["--figure"], []).append([(j, pv)])
        for fig, lots in by_fig.items():
            merged = [lots[0]]
            for lot in lots[1:]:
                cur = merged[-1]
                parts = [pv["a"] for _, pv in cur + lot]
                same = all(p.get("--size") == parts[0].get("--size") and p.get("--samples") == parts[0].get("--samples")
                           for p in parts)
                n_req = len({i for j, _ in cur + lot for i in j["requests"]})
                if same and n_req <= MAX_MERGE and view_images(union_args(parts, fig)) <= sum(map(view_images, parts)):
                    merged[-1] = cur + lot
                else:
                    merged.append(lot)
            by_fig[fig] = merged
        # pieces of one set: always one build, the set with the most waiting pieces first
        lots = [(fig, lot) for fig, ls in by_fig.items() for lot in ls]
        done = set()
        for name, members in sorted(sets().items(), key=lambda kv: -len(set(kv[1]) & set(by_fig))):
            have = [i for i, (fig, _) in enumerate(lots) if fig in members and i not in done]
            if len({lots[i][0] for i in have}) < 2:
                continue
            have.sort(key=lambda i: members.index(lots[i][0]))
            while have:
                take, n = [], 0
                for i in have:
                    k = len({x for j, _ in lots[i][1] for x in j["requests"]})
                    if take and n + k > MAX_MERGE:
                        break
                    take.append(i); n += k
                have = have[len(take):]
                if len({lots[i][0] for i in take}) < 2:
                    continue
                figs = list(dict.fromkeys(lots[i][0] for i in take))
                items2 = [x for i in take for x in lots[i][1]]
                out.append(_combined(items2, "outfit", g, figs, name))
                done.update(take)
        for i, (fig, lot) in enumerate(lots):
            if i not in done:
                out.append(lot[0][0] if len(lot) == 1 else _combined(lot, fig, g, None, None))
    return out


def _combined(items, fig, group, pieces, set_name):
    env, check = list(group[0]), group[1]
    a = union_args([pv["a"] for _, pv in items], fig)
    if pieces:
        rev = hashlib.sha1(open(OUTFIT, "rb").read()).hexdigest()[:8]
        # the --head close-up frames the pieces whose own requests asked for it (a skirt's waist-to-floor framing
        # would lose the sleeves' close-up)
        heads = [p for p in pieces if any(pv["a"]["--figure"] == p and "--head" in pv["a"] for _, pv in items)]
        env = [f"OUTFIT={','.join(pieces)}"] + ([f"OUTFIT_FRAME={','.join(heads)}"] if heads else []) + \
              [f"OUTFIT_REV={rev}"] + env
    j0 = items[0][0]
    param = view_param(env, a, check)
    reqs = sorted({i for j, _ in items for i in j["requests"]})
    return dict(j0, param=param, requests=reqs, key=job_key(j0["commit"], "views", param),
                order=min(j["order"] for j, _ in items), merged_from=[j["param"] for j, _ in items],
                **({"set": set_name, "pieces": pieces} if pieces else {}))


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git(*a, check=True, cwd=REPO):
    r = subprocess.run(["git", "-C", cwd, *a], capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"git {' '.join(a)}: {r.stderr.strip()}")
    return r


def load():
    try:
        with open(STATE) as f:
            s = json.load(f)
    except FileNotFoundError:
        s = {"next": 1, "requests": []}
    s.setdefault("ready", []); s.setdefault("sheet_due", False); s.setdefault("sheets_done", {})
    return s


def save(s):
    """Compact JSON, written and then read back: the shared folder is a network mount, and a replace on it once
    left the file cut short. A local copy is kept too."""
    s = {**s, "requests": [{k: v for k, v in r.items() if not k.startswith("_")} for r in s["requests"]]}
    try:     # (keep requests another process added since s was loaded: plan() holds s through slow git merges)
        with open(STATE) as f:
            disk = json.load(f)
        have = {r["id"] for r in s["requests"]}
        extra = [r for r in disk.get("requests", []) if r["id"] not in have and r["id"] >= min(have | {10 ** 9})]
        if extra:
            s["requests"] = sorted(s["requests"] + extra, key=lambda r: r["id"])
            s["next"] = max(s["next"], disk.get("next", 1))
    except (OSError, ValueError, KeyError):
        pass
    data = json.dumps(s, separators=(",", ":"))
    os.makedirs(WORK, exist_ok=True)
    with open(os.path.join(WORK, "queue.local.json"), "w") as f:
        f.write(data)
    for _ in range(5):
        with open(STATE, "w") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        time.sleep(0.5)
        with open(STATE) as f:
            if f.read() == data:
                return
    sys.exit(f"could not write {STATE} intact; the local copy is in {WORK}/queue.local.json")


def gen(p):
    return p.lower().endswith(GEN)


# ------------------------------------------------------------------ what a branch changes
def defs_touched(base, tip, path):
    """Top-level functions/assignments of a .py file whose lines differ between base and tip."""
    out = set()
    d = git("diff", "-U0", base, tip, "--", path, check=False).stdout
    old_lines, new_lines = [], []
    for m in re.finditer(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", d, re.M):
        a, an, b, bn = int(m[1]), int(m[2] or 1), int(m[3]), int(m[4] or 1)
        old_lines += range(a, a + max(an, 1)); new_lines += range(b, b + max(bn, 1))
    for rev, lines in ((base, old_lines), (tip, new_lines)):
        src = git("show", f"{rev}:{path}", check=False).stdout
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return {"<unparsable>"}
        spans = []
        for n in tree.body:
            name = getattr(n, "name", None) or (ast.unparse(n.targets[0]) if isinstance(n, ast.Assign) else type(n).__name__)
            spans.append((n.lineno, n.end_lineno, name))
        for ln in lines:
            out.add(next((nm for s, e, nm in spans if s <= ln <= e), "<module>"))
    return out


def footprint(base, tip):
    fp = {}
    for p in git("diff", "--name-only", base, tip).stdout.split():
        if not gen(p):
            fp[p] = defs_touched(base, tip, p) if p.endswith(".py") else {"*"}
    return fp


def overlap(a, b):
    hits = []
    for p in set(a) & set(b):
        common = (a[p] & b[p]) - {"<module>"}   # comments and blank lines between defs
        if "*" in a[p] or "*" in b[p] or common:
            hits.append(f"{p}: {', '.join(sorted(common)) or 'same file'}")
    return hits


# ------------------------------------------------------------------ merging (no checkout, nothing pushed)
def merge(ours, theirs, msg):
    """Merge two commits; generated-file conflicts take `ours`. Returns (commit or None, code conflicts)."""
    r = git("merge-tree", "--write-tree", "--name-only", "--no-messages", ours, theirs, check=False)
    lines = r.stdout.splitlines()
    tree, conflicted = lines[0], [p for p in lines[1:] if p]
    code = [p for p in conflicted if not gen(p)]
    if code:
        return None, code
    if conflicted:
        os.makedirs(WORK, exist_ok=True)
        idx = os.path.join(WORK, "index.tmp")
        env = dict(os.environ, GIT_INDEX_FILE=idx)
        subprocess.run(["git", "-C", REPO, "read-tree", tree], env=env, check=True)
        for p in conflicted:
            ent = git("ls-tree", ours, "--", p).stdout.split()
            args = ["--cacheinfo", f"{ent[0]},{ent[2]},{p}"] if ent else ["--force-remove", p]
            subprocess.run(["git", "-C", REPO, "update-index", *args], env=env, check=True)
        tree = subprocess.run(["git", "-C", REPO, "write-tree"], env=env, capture_output=True, text=True,
                              check=True).stdout.strip()
        os.remove(idx)
    return git("commit-tree", tree, "-p", ours, "-p", theirs, "-m", msg).stdout.strip(), []


def job_key(commit, kind, param):
    h = hashlib.sha1()
    for line in git("ls-tree", "-r", commit, "--", *INPUTS).stdout.splitlines():
        p = line.split("\t", 1)[1]
        # (a picture a script reads, such as the warden's hair textures, is an input; the renders are not)
        if (not gen(p) or (p.endswith(".png") and p not in RENDERED)) and not p.startswith(NOT_INPUTS):
            h.update(line.encode())
    h.update(json.dumps([kind, param]).encode())
    return f"{kind}-{h.hexdigest()[:12]}"


def duplicate_defs(commit, base=None):
    """duplicate_defs_all, less the names main itself already defines twice (in any file): those are main's own
    problem, carried faithfully by every branch, and are reported to John instead of holding the branch."""
    hits = duplicate_defs_all(commit)
    if not hits or not base:
        return hits
    known = {h.split(": ", 1)[1] for h in duplicate_defs_all(base)}
    return [h for h in hits if h.split(": ", 1)[1] not in known]


def duplicate_defs_all(commit):
    """Top-level names defined twice with different bodies in a .py file: git merges these cleanly, but the
    later definition silently replaces the earlier one for every caller (a semantic conflict)."""
    hits = []
    for path in git("ls-tree", "-r", "--name-only", commit, "--", "blender/").stdout.split():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(git("show", f"{commit}:{path}").stdout)
        except SyntaxError:
            hits.append(f"{path}: does not parse"); continue
        seen = {}
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                body = ast.dump(n)
                if n.name in seen and seen[n.name] != body:
                    hits.append(f"{path}: {n.name} defined twice, differently")
                seen[n.name] = body
    return hits


def undefined_names(commit):
    """Names a merged script calls but nothing defines (one side removed or renamed what the other still calls).
    Reported to the requester as a warning, not held: the call may sit behind a guard that never runs."""
    hits = []
    try:
        from pyflakes.api import check
        from pyflakes.reporter import Reporter
        import io
    except ImportError:
        return hits
    for path in git("ls-tree", "-r", "--name-only", commit, "--", "blender/").stdout.split():
        if path.endswith(".py"):
            buf = io.StringIO()
            check(git("show", f"{commit}:{path}").stdout, path, Reporter(buf, io.StringIO()))
            hits += [f"{path}:{ln.split(':', 2)[1]}: {ln.split(': ', 1)[-1]}" for ln in buf.getvalue().splitlines()
                     if "undefined name" in ln]
    return hits


def cached(key):
    return os.path.exists(os.path.join(RESULTS, key, "manifest.json"))


def failed(key):
    return os.path.exists(os.path.join(RESULTS, key + "-failed", "manifest.json")) and not cached(key)


# ------------------------------------------------------------------ planning
def onto_main(branch, tip, main):
    """The branch merged with main (or our hand resolution of it), or (None, conflicts)."""
    if branch == "main" or tip == main:
        return main, []
    res = git("rev-parse", "--verify", "--quiet", f"refs/rq/resolved/{branch}", check=False).stdout.strip()
    if res and all(git("merge-base", "--is-ancestor", x, res, check=False).returncode == 0 for x in (tip, main)):
        return res, []
    c, code = merge(main, tip, f"rq: main + {branch}")
    dup = duplicate_defs(c, main) if c else []
    return (None, dup) if dup else (c, code)


def cells_union(specs):
    rows, cols = set(), set()
    for s in specs:
        r, c = s.split(":")
        for part, dst in ((r, rows), (c, cols)):
            for x in part.split(","):
                a, _, b = x.partition("-")
                dst.update(range(int(a), int(b or a) + 1))
    return f"{','.join(map(str, sorted(rows)))}:{','.join(map(str, sorted(cols)))}"


def plan(fetch=True):
    if fetch:
        git("fetch", "--quiet", "--prune", "origin")
    main = git("rev-parse", "origin/main").stdout.strip()
    s = load()
    pend = [r for r in s["requests"] if r["state"] in ("pending", "conflict", "rendering")]
    held, ready = [], []
    for r in pend:
        tip = r["sha"] or git("rev-parse", "--verify", "--quiet", f"origin/{r['branch']}", check=False).stdout.strip()
        if not tip:
            r["state"], r["why"] = "conflict", "branch not found on origin"; held.append(r); continue
        r["tip"] = tip
        # a request keeps the main it was first planned on, so a merge to main mid-request does not re-render it
        if r.get("base_for") != tip:
            r["base"], r["base_for"] = main, tip
        base = r["base"]
        c, code = onto_main(r["branch"], tip, base)
        if code:
            r.pop("base_for", None)   # a held request is re-planned on the newest main
            t = git("merge-tree", "--write-tree", "--no-messages", base, tip, check=False).stdout.splitlines()[0]
            r["state"] = "conflict"
            r["why"] = {"conflicts_with_main": {p: (git("show", f"{t}:{p}", check=False).stdout.count("\n<<<<<<<")
                                                    if ":" not in p else "semantic") for p in code}}
            held.append(r); continue
        if r["state"] == "conflict":
            r["state"] = "pending"
        r.pop("why", None)
        r["merged"] = c
        r["warnings"] = undefined_names(c)
        r["_footprint"] = {k: sorted(v) for k, v in footprint(base, c).items()}
        ready.append(r)
    batches = []
    for r in sorted(ready, key=lambda r: r["id"]):
        fp = {k: set(v) for k, v in r["_footprint"].items()}
        r.pop("not_combined_with", None)
        placed = False
        # a request already rendering stays in the batch it started in: planned on its own again, its other jobs
        # got a new commit and were rendered a second time (#137 and #138, 2026-11-18)
        if r["state"] == "rendering" and r.get("batch_commit"):
            b = next((b for b in batches if b["commit"] == r["batch_commit"]), None)
            if b is None:
                batches.append({"commit": r["batch_commit"], "requests": [r], "fp": fp, "isolated": True})
            else:
                b["requests"].append(r)
            continue
        for b in batches:
            if r.get("solo") or b.get("solo"):
                continue     # (a request retried on its own after a combined job failed)
            if all(x["tip"] == r["tip"] for x in b["requests"]):
                b["requests"].append(r); placed = True; break     # same commit asked twice: one render
            # (mode=isolated is no longer honoured: John wants branches combined whenever their code does not
            #  overlap; befores, the one case isolation served, now come only from the cache)
            # a batch that is already rendering is closed: adding to it would change its commit and waste the
            # jobs in flight
            if len({x["branch"] for x in b["requests"]} | {r["branch"]}) > MAX_BATCH:
                continue     # John: a cap on batch size, so one failure does not lose the lot
            if b["isolated"] or not fp or b["requests"][0]["base"] != r["base"] or r["state"] == "rendering" \
                    or any(x["state"] == "rendering" for x in b["requests"]):
                continue
            clash = overlap(b["fp"], fp)
            c, code = (None, clash) if clash else merge(b["commit"], r["merged"], f"rq: + {r['branch']}")
            dup = duplicate_defs(c, r["base"]) if c else []
            if dup:
                c, code = None, dup
            if code:
                r.setdefault("not_combined_with", {})[b["requests"][0]["branch"]] = code; continue
            b["commit"] = c; b["requests"].append(r)
            for k, v in fp.items():
                b["fp"].setdefault(k, set()).update(v)
            placed = True; break
        if not placed:
            batches.append({"commit": r["merged"], "requests": [r], "fp": fp,
                            "isolated": not fp, "solo": bool(r.get("solo"))})
    jobs = []
    solo = {r["id"] for r in ready if r.get("solo")}
    for b in batches:
        bjobs = []
        want = {}
        for r in b["requests"]:
            for o in r["outputs"]:
                for v in (r["views"] if o == "views" else [None]):
                    if o == "views" and r.get("quality") == "check":
                        v += CHECK      # a check render is its own job, cached apart from the final one
                    want.setdefault((o, v), []).append(r)
        for (kind, param), rs in want.items():
            if kind == "frames":
                param = cells_union([r["cells"] for r in rs])
            bjobs.append({"kind": kind, "param": param, "commit": b["commit"], "requests": [r["id"] for r in rs],
                          "combines": [{"id": r["id"], "branch": r["branch"], "sha": r["tip"]} for r in b["requests"]],
                          "key": job_key(b["commit"], kind, param), "prio": PRIORITY[kind],
                          "order": min(r["id"] for r in rs)})
        jobs += combine_views(bjobs, solo)
    # identical jobs (the same key) from different batches are one job for all their requests
    seen = {}
    for j in jobs:
        if j["key"] in seen:
            seen[j["key"]]["requests"] = sorted(set(seen[j["key"]]["requests"]) | set(j["requests"]))
            seen[j["key"]]["dups"] = seen[j["key"]].get("dups", 0) + 1
        else:
            seen[j["key"]] = j
    jobs = list(seen.values())
    for r in ready:
        r["jobs"] = [j["key"] for j in jobs if r["id"] in j["requests"]]
    # full sheets: only once the scheduled task has marked them due; main first, then the branches marked ready
    if s["sheet_due"]:
        for br in ["main"] + s["ready"]:
            tip = git("rev-parse", "--verify", "--quiet", f"origin/{br}", check=False).stdout.strip()
            c, _ = onto_main(br, tip, main) if tip else (None, None)
            if c:
                jobs.append({"kind": "sheet", "param": None, "commit": c, "requests": [], "prio": 9, "order": 0,
                             "key": job_key(c, "sheet", None), "combines": [{"id": None, "branch": br, "sha": tip}]})
    for j in jobs:
        j["cached"] = cached(j["key"])
    # cheap drafts before expensive finals
    jobs.sort(key=lambda j: (j["prio"], 0 if (j["param"] or "").endswith(CHECK) else 1, j["order"]))
    save(s)
    return main, held, batches, jobs


def show(main, held, batches, jobs):
    print(json.dumps({
        "main": main[:9],
        "held": [{"id": r["id"], "branch": r["branch"], "thread": r["thread"], "why": r["why"]} for r in held],
        "batches": [{"combines": [{"id": r["id"], "branch": r["branch"], "sha": r["tip"][:9], "changes": r["_footprint"],
                                   "not_combined_with": r.get("not_combined_with")} for r in b["requests"]]}
                    for b in batches],
        "jobs": [{k: j[k] for k in ("key", "kind", "param", "prio", "requests", "cached")} for j in jobs]}, indent=2))


# ------------------------------------------------------------------ running
CHECK = "  [quality check]"
MEMO = os.path.join(WORK, "memo")


def command(job, out):
    """Every script runs under rq_run.py: it remembers the head's surface between runs (the eyes' sculpt and the
    ears, most of a build) and, for a check render, drops Cycles to 8 denoised samples."""
    q = shlex.quote
    # (--keep-scene: Cycles keeps the scene between a job's renders; pixel-identical, see rq_run.py)
    run = f"python3 {q(os.path.join(HERE, 'rq_run.py'))} --memo {q(MEMO)} --keep-scene"
    glb = f"{run} -- blender/ranger/export_glb.py --out {q(os.path.join(out, 'ranger.glb'))}"
    if job["kind"] == "sheet":
        return f"{run} -- blender/ranger/ranger.py --out {q(out)} && {glb}"
    if job["kind"] == "glb":
        return glb
    if job["kind"] == "frames":
        return f"{run} -- {q(os.path.join(HERE, 'rq_frames.py'))} --out {q(out)} --cells {q(job['param'])}"
    view = job["param"]                         # views: a script in the branch, with its own arguments
    if view.endswith(CHECK):
        view, run = view[:-len(CHECK)], f"{run} --quality check"
    env = ""
    if view.startswith("env "):                 # a request's "env": "env K=V ... script args" (part of the key)
        words = shlex.split(view)[1:]
        sets = [w for w in words[:next((i for i, w in enumerate(words) if "=" not in w), len(words))]]
        view = shlex.join(words[len(sets):])
        env = "env " + " ".join(q(w) for w in sets) + " "
    return f"{env}{run} -- {view} --out {q(out)}"


def start(job):
    tmp_out = os.path.join(WORK, "out-" + job["key"])
    wt = os.path.join(WORK, "wt-" + job["key"])
    shutil.rmtree(tmp_out, ignore_errors=True); os.makedirs(tmp_out)
    git("worktree", "remove", "--force", wt, check=False)
    git("worktree", "add", "--detach", wt, job["commit"])
    if job.get("pieces") or "--figure outfit" in (job["param"] or ""):
        os.makedirs(os.path.join(wt, "blender", "outfit"), exist_ok=True)
        shutil.copy(OUTFIT, os.path.join(wt, "blender", "outfit", "outfit.py"))
    log = open(os.path.join(tmp_out, "render.log"), "w")
    cmd = command(job, tmp_out)
    log.write(f"$ {cmd}\n"); log.flush()
    p = subprocess.Popen(cmd, shell=True, cwd=wt, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return {"proc": p, "log": log, "wt": wt, "out": tmp_out, "t0": time.time(), "job": job}


def finish(run):
    job, rc = run["job"], run["proc"].returncode
    run["log"].write(f"\n[exit {rc} after {time.time() - run['t0']:.0f}s]\n"); run["log"].close()
    git("worktree", "remove", "--force", run["wt"], check=False)
    files = {}
    for root, _, fs in os.walk(run["out"]):
        for f in fs:
            p = os.path.join(root, f)
            files[os.path.relpath(p, run["out"])] = {"bytes": os.path.getsize(p),
                                                       "git_blob": git("hash-object", p).stdout.strip()}
    m = {"ok": rc == 0, "key": job["key"], "kind": job["kind"], "param": job["param"], "commit": job["commit"],
         "combines": job["combines"], "seconds": round(time.time() - run["t0"]), "files": files, "finished": now(),
         "requests": job["requests"], **{k: job[k] for k in ("merged_from", "set", "pieces") if k in job}}
    with open(os.path.join(run["out"], "manifest.json"), "w") as f:
        json.dump(m, f, indent=2)
    dst = os.path.join(RESULTS, job["key"] + ("" if m["ok"] else "-failed"))
    shutil.rmtree(dst, ignore_errors=True); shutil.copytree(run["out"], dst)   # only complete results land here
    shutil.rmtree(run["out"], ignore_errors=True)
    return m


def stop(run):
    os.killpg(run["proc"].pid, signal.SIGTERM)
    run["proc"].wait()
    run["log"].close()
    git("worktree", "remove", "--force", run["wt"], check=False)
    shutil.rmtree(run["out"], ignore_errors=True)


def settle(s):
    """Mark requests whose jobs are all finished. Returns the requests that just finished."""
    done = []
    for r in s["requests"]:
        if r["state"] in ("pending", "rendering") and r.get("jobs"):
            bad = [k for k in r["jobs"] if failed(k)]
            if bad and not r.get("solo") and any(_retry_alone(k) for k in bad):
                r["solo"] = True; r.pop("jobs", None); r.pop("batch_commit", None); r["state"] = "pending"
                r["retried_alone"] = bad
                event("retry_alone", bad[0], request=r["id"])
                continue
            if bad or all(cached(k) for k in r["jobs"]):
                r["state"] = "failed" if bad else "done"
                r["results"] = {k: os.path.join(RESULTS, k + ("-failed" if k in bad else "")) for k in r["jobs"]}
                r["finished"] = now(); done.append(r)
    return done


def _retry_alone(key):
    """A failed job that combined several requests (a set, merged views or merged branches), unless it was killed
    at the time limit (alone it would be killed again)."""
    try:
        m = json.load(open(os.path.join(RESULTS, key + "-failed", "manifest.json")))
    except (OSError, ValueError):
        return False
    many = len(m.get("requests") or []) > 1 or len(m.get("combines") or []) > 1 or bool(m.get("merged_from"))
    return many and m.get("seconds", 0) < KILL_S


PARALLEL = int(os.environ.get("RQ_PARALLEL", "3"))
def _mem_gb():
    """What the jobs may use together: the container's own limit where it has one (13.4 GB here, under the
    machine's 15: a third warden-sized build was killed at it), else the machine's memory; less a little."""
    import glob as _g
    lim = []
    for f in _g.glob("/sys/fs/cgroup/**/claude-code-bash/memory.limit_in_bytes", recursive=True) + \
            _g.glob("/sys/fs/cgroup/**/claude-code-bash/memory.max", recursive=True) + ["/sys/fs/cgroup/memory.max"]:
        try:
            lim.append(int(open(f).read()))
        except (OSError, ValueError):
            pass
    with open("/proc/meminfo") as f:
        lim.append(int(f.readline().split()[1]) * 1024)
    return min(lim) / 2 ** 30 - 0.5


MEM_GB = _mem_gb()
EVENTS = os.path.join(DATA, "events.jsonl")


def event(ev, key, **kw):
    """One line per job event, for rq_stats.py: start, done, failed, warn, killed, stopped (a sheet preempted)."""
    with open(EVENTS, "a") as f:
        f.write(json.dumps({"t": now(), "ev": ev, "key": key, **kw}) + "\n")


WARN_S, KILL_S = 600, 1800    # John's rule: a job past 10 minutes is reported, one past 30 is killed


def job_gb(job):
    """About how much memory a job's build peaks at: the warden (771 hair strands, cape, armour) peaks at 7.6 GB (one of two side
    by side was killed here); the other figures stay under 4."""
    p = job.get("param") or ""
    if "OUTFIT=" in p:
        return min(6.0, 2.0 + 0.6 * len(re.search(r"OUTFIT=(\S+)", p)[1].split(",")))
    if "--mannequin" in p:
        return 2.0                                     # (no head, face or hair: a light build)
    return 7.6 if "warden" in p else 4.0


def cmd_serve(a):
    """Run jobs in priority order until a request (or a sheet) finishes, then exit so the thread can deliver."""
    os.makedirs(WORK, exist_ok=True)
    lock = os.path.join(WORK, "serve.pid")
    if os.path.exists(lock):
        try:
            os.kill(int(open(lock).read()), 0); sys.exit("serve is already running")
        except (ProcessLookupError, ValueError):
            pass
    open(lock, "w").write(str(os.getpid()))
    running = {}     # key -> run: quick jobs run side by side (each build is single-threaded Python, so
    #                  PARALLEL of them overlap their builds on the 4 cores); a full sheet always runs alone
    finished = []
    waits_seen = {}   # key -> what a job was held for (the batching stats)
    try:
        while True:
            main, held, batches, jobs = plan(fetch=not running)
            s = load()
            fin = settle(s)
            finished += [{"id": r["id"], "thread": r["thread"], "state": r["state"]} for r in fin]
            todo = [j for j in jobs if not j["cached"] and not failed(j["key"]) and j["key"] not in running]
            waits = {j["key"]: hold_until(j, s) for j in todo}
            if any(waits.values()):
                for j in todo:
                    if waits[j["key"]] and j["key"] not in s.setdefault("holding", {}):
                        s["holding"][j["key"]] = waits[j["key"]]
                        print(f"{now()} hold {j['key']} for {waits[j['key']]['for']} until "
                              f"{time.strftime('%H:%M:%SZ', time.gmtime(waits[j['key']]['until']))}", flush=True)
            for k in list(s.get("holding", {})):
                if not waits.get(k):
                    s["holding"].pop(k)
            for k, w in waits.items():
                if w:
                    waits_seen.setdefault(k, w["for"])
            held_jobs = [j for j in todo if waits[j["key"]]]
            todo = [j for j in todo if not waits[j["key"]]]
            if s["sheet_due"] and not running and not any(j["kind"] == "sheet" for j in todo):
                s["sheet_due"] = False                     # every due sheet is rendered
            save(s)
            if finished and not running:
                print(json.dumps({"finished": finished})); return
            if not todo and not running and held_jobs:
                time.sleep(max(1, min(10, min(waits[j["key"]]["until"] for j in held_jobs) - time.time())))
                continue                                   # (a held job: wait for its partner, then plan again)
            if not todo and not running:
                print(json.dumps({"idle": True, "held": [r["id"] for r in held]})); return
            # start more jobs while there is room, unless a request has just finished (then let the running ones
            # end and exit, so the thread can deliver)
            sheet_running = any(r["job"]["kind"] == "sheet" for r in running.values())
            while todo and not finished and not sheet_running and len(running) < PARALLEL:
                job = todo.pop(0)
                if job["kind"] == "sheet" and running:
                    break                                      # the sheet waits for the quick jobs to end
                if running and sum(job_gb(r["job"]) for r in running.values()) + job_gb(job) > MEM_GB:
                    todo.insert(0, job)                        # it would not fit in memory beside the running ones
                    break
                s = load()
                ids = {c["id"] for c in job["combines"]}
                for r in s["requests"]:
                    if r["id"] in job["requests"]:
                        r["state"] = "rendering"
                    if r["id"] in ids and r["state"] in ("pending", "rendering") and job["kind"] != "sheet":
                        r["batch_commit"] = job["commit"]      # (its batch is closed: see plan)
                save(s)
                print(f"{now()} start {job['key']} for {[c['branch'] for c in job['combines']]}", flush=True)
                running[job["key"]] = start(job)
                event("start", job["key"], requests=job["requests"], param=job["param"],
                      parallel=len(running), waiting=len(todo))
                if job["kind"] != "sheet":
                    batch_event(job, s, waits_seen.pop(job["key"], None))
                if job["kind"] == "sheet":
                    break
            mtime = os.path.getmtime(STATE)
            while running and all(r["proc"].poll() is None for r in running.values()):
                time.sleep(10)
                for r in running.values():                 # John's rule: warn past 10 minutes, kill at 30
                    took = time.time() - r["t0"]
                    if took > KILL_S and not r.get("killed"):
                        r["killed"] = True
                        r["log"].write(f"\n[killed: over the {KILL_S // 60}-minute limit]\n"); r["log"].flush()
                        print(f"{now()} KILLED {r['job']['key']} after {took:.0f}s (limit {KILL_S // 60} min)", flush=True)
                        os.killpg(r["proc"].pid, signal.SIGKILL)
                        event("killed", r["job"]["key"], seconds=round(took))
                    elif took > WARN_S and not r.get("warned"):
                        r["warned"] = True
                        event("warn", r["job"]["key"], seconds=round(took))
                        print(f"{now()} WARN slow {r['job']['key']} running {took:.0f}s (over {WARN_S // 60} min)",
                              flush=True)
                sheet = next((r for r in running.values() if r["job"]["kind"] == "sheet"), None)
                if sheet and os.path.getmtime(STATE) != mtime:   # a full sheet gives way to anything quicker
                    mtime = os.path.getmtime(STATE)
                    if any(r["state"] == "pending" for r in load()["requests"]):
                        print(f"{now()} sheet {sheet['job']['key']} stopped for a quicker job; it reruns when idle", flush=True)
                        stop(sheet); running.pop(sheet["job"]["key"])
                        event("stopped", sheet["job"]["key"])
                if not finished and len(running) < PARALLEL and os.path.getmtime(STATE) != mtime:
                    break                                  # a new request arrived: plan again and fill the free slot
            for key in [k for k, r in running.items() if r["proc"].poll() is not None]:
                run = running.pop(key); job = run["job"]
                m = finish(run)
                event("done" if m["ok"] else "failed", job["key"], seconds=m["seconds"])
                print(f"{now()} {'done' if m['ok'] else 'FAILED'} {job['key']} in {m['seconds']}s", flush=True)
                if job["kind"] == "sheet":
                    s = load(); s["sheets_done"][job["combines"][0]["branch"]] = job["key"]; save(s)
                    print(json.dumps({"sheet_finished": job["combines"][0]["branch"], "ok": m["ok"], "key": job["key"]}))
                    return
    finally:
        os.remove(lock)


def batch_event(job, s, held_for):
    """One line per job started, for the nightly batching numbers (rq_stats.py): what it combined, how long its
    first request waited, what it was held for, and the announced pieces it left out (the stragglers)."""
    rs = [r for r in s["requests"] if r["id"] in job["requests"]]
    first = min((ts(r["created"]) for r in rs), default=time.time())
    figs = job_figures(job)
    strag = [i["id"] for i in open_intents(s) if any(related(f, i["piece"]) for f in figs)
             and not (i["thread"] in {r["thread"] for r in rs} and i["branch"] in {r["branch"] for r in rs})]
    event("batch", job["key"], requests=job["requests"], pieces=job.get("pieces") or figs, set=job.get("set"),
          parts=len(job.get("merged_from") or [job["param"]]), branches=len({c["branch"] for c in job["combines"]}),
          waited=round(time.time() - first), held_for=held_for, stragglers=strag, dups=job.get("dups", 0),
          solo=any(r.get("solo") for r in rs), check=(job["param"] or "").endswith(CHECK))


def related(fig_a, fig_b):
    if fig_a == fig_b:
        return True
    return any(fig_a in m and fig_b in m for m in sets().values())


def job_figures(job):
    p = job.get("param") or ""
    if job.get("pieces"):
        return list(job["pieces"])
    m = re.search(r"--figure\s+(\S+)", p)
    return [m[1]] if m else []


def hold_until(job, s):
    """When a waiting job may start: None now, else {"until": t, "for": why}. John: a short window for related
    requests, longer only for an announced intent, and never past the readiness cutoff (a slow piece goes in the
    next batch)."""
    if job["kind"] == "sheet" or not job["requests"]:
        return None
    rs = [r for r in s["requests"] if r["id"] in job["requests"]]
    if not rs or any(r.get("solo") for r in rs):
        return None
    t0 = min(ts(r["created"]) for r in rs)
    t1 = max(ts(r["created"]) for r in rs)
    cutoff = t0 + READY_CUTOFF_S
    figs = job_figures(job)
    best = None
    for i in open_intents(s):
        if i["thread"] in {r["thread"] for r in rs} and i["branch"] in {r["branch"] for r in rs}:
            continue       # (the job's own thread announcing its next step does not hold this one)
        if any(related(f, i["piece"]) for f in figs) and (not i.get("quality") or i["quality"] == rs[0].get("quality")):
            eta = cutoff       # John: a 5-minute collation window from the first piece ready; stragglers wait for the next
            if eta > time.time() and (best is None or eta > best["until"]):
                best = {"until": eta, "for": f"intent #{i['id']} ({i['piece']}, {i['thread'][-6:]})"}
    win = t1 + WINDOW_S
    if win > time.time() and win <= cutoff and (best is None or win > best["until"]):
        best = {"until": win, "for": f"the {WINDOW_S} s window"}
    return best


def open_intents(s):
    t = time.time()
    return [i for i in s.get("intents", []) if i["state"] == "open" and ts(i["created"]) + 0 <= t
            and i["eta"] + INTENT_TTL_S > t]


def cmd_intent(a):
    """A thread says what it is about to send, before it does the work, so the queue can hold a related job for it."""
    s = load()
    s.setdefault("intents", [])
    s.setdefault("next_intent", 1)
    piece = a.piece or (re.search(r"--figure\s+(\S+)", a.view or "") or [None, None])[1]
    if not piece:
        sys.exit("intent needs --piece FIGURE (blender/<figure>/<figure>.py) or a --view with --figure")
    st = next((n for n, m in sets().items() if piece in m), None)
    for o in s["intents"]:      # a thread's newer intent for the same piece replaces its older one
        if o["state"] == "open" and o["thread"] == a.thread and o["piece"] == piece:
            o["state"] = "replaced"
    i = {"id": s["next_intent"], "thread": a.thread, "branch": a.branch, "piece": piece, "set": a.set or st,
         "view": a.view, "quality": a.quality, "eta": time.time() + 60 * a.eta, "created": now(),
         "note": a.note, "state": "open"}
    s["next_intent"] += 1; s["intents"].append(i); save(s)
    print(json.dumps(dict(i, eta=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(i["eta"]))), indent=2))
    pend = [r for r in s["requests"] if r["state"] == "pending" and any(
        related(piece, f) for v in r.get("views") or [] for f in re.findall(r"--figure\s+(\S+)", v))]
    if pend:
        print(f"waiting now that this could join: {[r['id'] for r in pend]} (held up to "
              f"{READY_CUTOFF_S // 60} min after each one's request; later, {piece} goes in the next batch)")


def cmd_intents(a):
    s = load()
    for i in s.get("intents", []):
        if a.all or i["state"] == "open":
            print(f"intent #{i['id']:<3} {i['state']:<10} {i['piece']:<18} {i.get('set') or '-':<14} "
                  f"eta {time.strftime('%H:%M:%SZ', time.gmtime(i['eta']))}  {i['branch']}  {i.get('note') or ''}")


def cmd_drop_intent(a):
    s = load()
    for i in s.get("intents", []):
        if i["id"] == a.id:
            i["state"] = "dropped"
    save(s)


# ------------------------------------------------------------------ small commands
def before_only_from_cache(a, outs):
    """John's rule: a "before" is never rendered. A request whose code is main's (the branch is main, or the commit
    is already on main) is served only from the cache; if any of its jobs is not cached it is refused, and the
    refusal lists the cached renders of main's code that the thread can compare against instead."""
    git("fetch", "--quiet", "origin", "main", check=False)
    c = git("rev-parse", "--verify", "--quiet", f"{a.sha or 'origin/' + a.branch}^{{commit}}", check=False).stdout.strip()
    if not c or (a.branch != "main" and
                 git("merge-base", "--is-ancestor", c, "origin/main", check=False).returncode != 0):
        return
    want = [(o, v) for o in outs for v in (a.view if o == "views" else [a.cells if o == "frames" else None])]
    missing = [(k, p) for k, p in want if not cached(job_key(c, k, p))]
    if not missing:
        return
    have = []
    for m in glob.glob(os.path.join(RESULTS, "*", "manifest.json")):
        try:
            j = json.load(open(m))
        except (OSError, ValueError):
            continue
        if j.get("ok") and j.get("commit") and git("merge-base", "--is-ancestor", j["commit"], "origin/main",
                                                    check=False).returncode == 0:
            have.append(f"  {os.path.dirname(m)}  ({j['kind']} {j.get('param') or ''}, main {j['commit'][:7]})")
    sys.exit("refused: a before render of main's code is never rendered, only taken from the cache, and these are "
             "not cached:\n" + "\n".join(f"  {k} {p or ''}" for k, p in missing) +
             "\ncompare against one of the cached renders of main instead:\n" + "\n".join(sorted(have)))


def cmd_add(a):
    s = load()
    outs = [o for o in a.outputs.split(",") if o]
    if "sheet" in outs:
        sys.exit("threads never request the full sheet: ask for frames (a few cells) instead; "
                 "the full sheet runs only on the scheduled task")
    bad = [o for o in outs if o not in REQUESTABLE]
    if bad or not outs:
        sys.exit(f"outputs must be some of {REQUESTABLE}")
    if "views" in outs and not a.view:
        sys.exit("views needs at least one --view 'path/in/repo.py args'")
    if "frames" in outs and not a.cells:
        sys.exit("frames needs --cells ROWS:COLS, e.g. 0-7:2,4,6")
    before_only_from_cache(a, outs)
    r = {"id": s["next"], "thread": a.thread, "branch": a.branch, "sha": a.sha, "outputs": outs,
         "views": a.view or [], "cells": a.cells, "mode": a.mode, "note": a.note, "state": "pending", "created": now(),
         "quality": a.quality}
    for o in s["requests"]:   # coalescing: a newer request for the same piece replaces one still waiting
        # (the same thread and branch following the branch tip, or the same branch asking for the same views and
        #  cells at another commit, pinned or not: the stale one is dropped, never rendered)
        if o["state"] in ("pending", "conflict") and o["branch"] == a.branch and o["outputs"] == outs and (
                (o["thread"] == a.thread and not a.sha and not o.get("sha"))
                or (sorted(o.get("views") or []) == sorted(r["views"]) and o.get("cells") == r["cells"]
                    and o.get("quality") == r["quality"])):
            o["state"] = f"superseded by {r['id']}"
            event("coalesced", None, dropped=o["id"], by=r["id"], branch=a.branch)
    figs = {f for v in r["views"] for f in re.findall(r"--figure\s+(\S+)", v)}
    for i in s.get("intents", []):    # the request an intent announced has come
        if i["state"] == "open" and i["branch"] == a.branch and (i["piece"] in figs or not figs):
            i["state"] = f"sent as #{r['id']}"
    s["next"] += 1; s["requests"].append(r); save(s)
    print(json.dumps(r, indent=2))


def cmd_ready(a):
    s = load()
    if a.branch not in s["ready"]:
        s["ready"].append(a.branch)
    save(s)


def cmd_sheet_due(a):
    s = load(); s["sheet_due"] = True; save(s)


def cmd_list(a):
    s = load()
    print(f"sheet due: {s['sheet_due']}   ready for the next sheet: {s['ready']}")
    for r in s["requests"]:
        print(f"#{r['id']:<3} {r['state']:<16} {r['thread'][:14]:<14} {r['branch']:<32} "
              f"{','.join(r['outputs']):<18} {r.get('note') or ''}")


def cmd_plan(a):
    show(*plan())


THREAD_NAMES = {
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBGVwbvHb2eMywHyFAReFP87": "Skin study",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBK4u2f9dGFpTToijqneW7Pp": "Walk on mannequin",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nB6RET1QoPpSXTrbYhhywz7b": "Eyes",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBMNGnniQFBvLZARCJwcRN6z": "Walk",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBUsN38NMgogiusf2HDh4BrB": "Face",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBEeHsM6Q8wvoEWwWz3CHuEo": "Clothes",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBSQLhQLoJJHsp7fdxZwKVin": "John (render queue)",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBVQEmZByDtBCP6HWt9RcPH1": "Base female / rogue",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBCMZ5NsWAZKEeG5j6twrjo2": "Lip colour",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nB8yN8piTmjhz8SVxLp6ztyJ": "Skin",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBLSp9vr6CauC2NFvnDaWTW4": "Warden",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBJkqDv7Ktra4a31euzvMxpx": "Reviewer",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nB58A32xihrD4suX2WgSKk5h": "World walk",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBU9x4GTj8QLYGorRA6gLuHc": "Creator heads",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBUhdiu12Gy6W5kaV1SvyEVc": "Face creator",
}


def cmd_export(a):
    """Write the queue as JSON rows for the tracking page: one row per request, newest first."""
    s = load()
    rows = []
    for r in s["requests"]:
        secs, fin, kinds = 0, r.get("finished"), []
        for k in r.get("jobs") or []:
            for d in (os.path.join(RESULTS, k), os.path.join(RESULTS, k + "-failed")):
                mp = os.path.join(d, "manifest.json")
                if os.path.exists(mp):
                    m = json.load(open(mp))
                    secs += m.get("seconds", 0); fin = max(fin or "", m.get("finished", ""))
                    break
        state = r["state"].split(" ")[0]
        rows.append({
            "id": r["id"], "requester": THREAD_NAMES.get(r["thread"], r["thread"]), "branch": r["branch"],
            "commit": (r.get("sha") or r.get("tip") or "")[:7],
            "outputs": ", ".join(r["outputs"]), "mode": r.get("mode", "combined"), "state": state,
            "note": r.get("note") or "", "created": r.get("created"),
            "finished": fin if state in ("done", "failed") else None,
            "render_seconds": secs or None,
            "results": ", ".join(os.path.join("render-queue/results", os.path.basename(p))
                                 for p in (r.get("results") or {}).values()),
            "warnings": (f"{len(r['warnings'])} undefined name(s) after the merge" if r.get("warnings") else ""),
        })
    rows.sort(key=lambda x: -x["id"])
    sched = [{"item": "Full sheet, .blend and world model", "when": "Daily, 2:41 AM Toronto, only when the queue is idle",
              "sheet_due": s["sheet_due"], "ready_branches": ", ".join(s["ready"]) or "none"}]
    with open(a.out, "w") as f:
        json.dump({"rows": rows, "meta": [{"label": "Written by", "value": "rq.py export, from the render queue's queue.json"},
                                          {"label": "Exported", "value": now()}]}, f, indent=1)
    with open(a.out.replace(".json", "-schedule.json"), "w") as f:
        json.dump({"rows": sched, "meta": [{"label": "Exported", "value": now()}]}, f, indent=1)
    print(a.out, len(rows), "rows")


def cmd_cancel(a):
    s = load()
    for r in s["requests"]:
        if r["id"] == a.id:
            r["state"] = "cancelled"
    save(s)


def cmd_resolve(a):
    git("fetch", "--quiet", "origin")
    wt = os.path.join(WORK, "resolve-" + a.branch.replace("/", "_"))
    git("worktree", "remove", "--force", wt, check=False)
    git("worktree", "add", "--detach", wt, f"origin/{a.branch}")
    git("merge", "--no-ff", "--no-commit", "origin/main", cwd=wt, check=False)
    conflicted = git("diff", "--name-only", "--diff-filter=U", cwd=wt).stdout.split()
    for p in conflicted:
        if gen(p):
            git("checkout", "origin/main", "--", p, cwd=wt); git("add", p, cwd=wt)
    print(json.dumps({"worktree": wt, "code_conflicts": [p for p in conflicted if not gen(p)]}, indent=2))


def cmd_resolved(a):
    wt = os.path.join(WORK, "resolve-" + a.branch.replace("/", "_"))
    if git("grep", "-n", "^<<<<<<<", "--", "*.py", "*.js", cwd=wt, check=False).stdout:
        sys.exit("conflict markers are still there")
    git("add", "-A", cwd=wt)
    git("-c", "user.name=Render queue", "-c", "user.email=render-queue@localhost", "commit", "-q", "-m",
        f"Merge main into {a.branch} (conflicts resolved by the render queue)", cwd=wt)
    c = git("rev-parse", "HEAD", cwd=wt).stdout.strip()
    git("update-ref", f"refs/rq/resolved/{a.branch}", c)
    os.makedirs(RESOLUTIONS, exist_ok=True)
    bundle = os.path.join(RESOLUTIONS, a.branch.replace("/", "_") + ".bundle")
    git("bundle", "create", bundle, f"refs/rq/resolved/{a.branch}", f"^origin/{a.branch}", "^origin/main")
    git("worktree", "remove", "--force", wt, check=False)
    print(json.dumps({"commit": c, "bundle": bundle,
                      "owner_applies_with": f"git fetch {bundle} refs/rq/resolved/{a.branch} && git merge --ff-only FETCH_HEAD"}, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("add")
    q.add_argument("--thread", required=True); q.add_argument("--branch", required=True); q.add_argument("--sha")
    q.add_argument("--outputs", required=True); q.add_argument("--view", action="append"); q.add_argument("--cells")
    q.add_argument("--mode", choices=("combined", "isolated"), default="combined"); q.add_argument("--note")
    q.add_argument("--quality", choices=("final", "check"), default="final")
    q = sub.add_parser("intent")
    q.add_argument("--thread", required=True); q.add_argument("--branch", required=True)
    q.add_argument("--piece", help="the figure module (blender/<piece>/<piece>.py) the render will be of")
    q.add_argument("--set", help="the set it belongs to (default: from sets.json)")
    q.add_argument("--view", help="the views.py command it expects to send, if known")
    q.add_argument("--quality", choices=("final", "check"))
    q.add_argument("--eta", type=float, default=5, help="minutes until the request will be sent")
    q.add_argument("--note")
    sub.add_parser("intents").add_argument("--all", action="store_true")
    sub.add_parser("drop-intent").add_argument("id", type=int)
    sub.add_parser("ready").add_argument("--branch", required=True)
    for n in ("list", "plan", "serve", "sheet-due"):
        sub.add_parser(n)
    for n in ("resolve", "resolved"):
        sub.add_parser(n).add_argument("branch")
    sub.add_parser("cancel").add_argument("id", type=int)
    sub.add_parser("export").add_argument("out")
    a = p.parse_args()
    globals()["cmd_" + a.cmd.replace("-", "_")](a)


if __name__ == "__main__":
    main()
