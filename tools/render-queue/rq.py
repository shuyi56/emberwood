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
  * Without an intent a job still waits WINDOW_S (22 s) after its newest request for a related one, never longer.
  * Caps: at most MAX_BATCH (4) branches in a merged commit and MAX_MERGE (6) requests in a combined job. A combined
    job that fails (not one killed at 30 minutes) is retried with each of its requests on its own.
"""
import argparse, functools, glob, ast, hashlib, json, os, re, shlex, shutil, signal, statistics, subprocess, sys, time, uuid

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
# John, 2026-11-19: a thread that asks for the same views again and again loses priority each time. The request that
# repeats the previous one (same thread, same views, same quality, not interrupted by a different request, and not
# more than `quiet_s` after it) is `repeat` steps behind, up to `max`; each `age_s` it waits wins one step back, so
# it is never starved, and an idle queue runs it at once. Hogging warnings stay as they were.
PEN = {"quiet_s": 2700, "max": 3, "age_s": 600}
MAX_BATCH = int(os.environ.get("RQ_MAX_BATCH", "4"))      # branches merged into one commit
MAX_MERGE = int(os.environ.get("RQ_MAX_MERGE", "6"))      # requests in one combined views.py job
WINDOW_S = int(os.environ.get("RQ_WINDOW", "22"))         # a job waits this long after its newest request
READY_CUTOFF_S = int(os.environ.get("RQ_READY_CUTOFF", "300"))   # John: a 5-minute collation window from the first piece ready
INTENT_TTL_S = 1800                                       # an intent nobody followed lapses after its ETA + this
VIEWS_PY = "blender/base/views.py"
RQ_VIEWS = "blender/base/rq_views.py"        # views.py plus crops and the pose strip (the queue's rq_views.py)
RQ_VIEWS_SRC = os.path.join(HERE, "rq_views.py")
OUTFIT = os.path.join(HERE, "rq_outfit.py")
ANGLES = ("front", "3/4", "profile", "back3/4", "back")
# views.py options a combined job can take the union of (True: takes a value); any other option keeps a job apart
MERGEABLE = {"--figure": True, "--angles": True, "--size": True, "--samples": True, "--frame": True,
             "--mannequin": False, "--turn": False, "--head": False, "--clay": False, "--clay-only": False,
             "--denoise": False, "--crop": True, "--crop-size": True, "--crop-angles": True, "--poses": True,
             "--pose-angle": True, "--pose-size": True}


def sets():
    for d in (DATA, HERE):
        try:
            return {k: v for k, v in json.load(open(os.path.join(d, "sets.json"))).items() if not k.startswith("_")}
        except (OSError, ValueError):
            continue
    return {}


@functools.lru_cache(maxsize=None)
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
    if not words or words[0] not in (VIEWS_PY, RQ_VIEWS):
        return None
    a, i, sets_ = {}, 1, []
    while i < len(words):
        w = words[i]
        if w == "--set" and i + 1 < len(words):     # (a figure setting: jobs share a build only with the same ones)
            sets_.append(words[i + 1]); i += 2; continue
        if w not in MERGEABLE or (MERGEABLE[w] and i + 1 >= len(words)):
            return None
        a[w] = words[i + 1] if MERGEABLE[w] else True
        i += 2 if MERGEABLE[w] else 1
    if "--figure" not in a or not ("--turn" in a or "--head" in a or "--crop" in a or "--poses" in a) \
            or any(e.startswith("OUTFIT") for e in env):
        return None
    if sets_:
        a["--set"] = sorted(sets_)
    return {"env": env, "check": check, "a": a, "group": (tuple(sorted(env)), check, "--mannequin" in a,
                                                          a.get("--frame"), tuple(sorted(sets_)))}


def view_images(a):
    ang = len(a["--angles"].split(",")) if "--angles" in a else 5
    looks = 1 if "--clay-only" in a else 2 if "--clay" in a else 1
    n = (("--turn" in a) + ("--head" in a)) * ang * looks
    if "--crop" in a:
        cang = len((a.get("--crop-angles") or a.get("--angles") or "3/4").split(","))
        n += len(a["--crop"].split(",")) * cang * looks
    if "--poses" in a:
        n += 7 if a["--poses"] == "all" else len(a["--poses"].split(","))
    return n


def _union_list(parts, key, order=None):
    vals = [x for p in parts if key in p for x in p[key].split(",") if x]
    vals = list(dict.fromkeys(vals))
    return ",".join(sorted(vals, key=order.index) if order and all(v in order for v in vals) else vals)


def _views_part(p):
    return "--turn" in p or "--head" in p


def union_args(parts, figure):
    # a crop's or a pose strip's own angles are made explicit first, so a turn beside it keeps its own
    parts = [dict(p, **({"--crop-angles": p.get("--angles") or "3/4"} if "--crop" in p and "--crop-angles" not in p
                        else {})) for p in parts]
    a = {"--figure": figure}
    for f in ("--mannequin", "--turn", "--head", "--denoise"):
        if any(f in p for p in parts):
            a[f] = True
    looked = [p for p in parts if "--turn" in p or "--head" in p or "--crop" in p] or parts   # (a pose strip is clay)
    if all("--clay-only" in p for p in looked):
        a["--clay-only"] = True
    elif any("--clay" in p or "--clay-only" in p for p in looked):
        a["--clay"] = True
    vparts = [p for p in parts if _views_part(p)]
    if vparts and all("--angles" in p for p in vparts):
        want = {x for p in vparts for x in p["--angles"].split(",")}
        a["--angles"] = ",".join([x for x in ANGLES if x in want] + sorted(want - set(ANGLES)))
    if any("--crop" in p for p in parts):
        a["--crop"] = _union_list(parts, "--crop")
        if any("--crop-angles" in p for p in parts):
            a["--crop-angles"] = _union_list(parts, "--crop-angles", list(ANGLES))
    if any("--poses" in p for p in parts):
        a["--poses"] = "all" if any(p.get("--poses") == "all" for p in parts) else _union_list(parts, "--poses")
        pa = next((p["--pose-angle"] for p in parts if "--pose-angle" in p), None)
        if pa:
            a["--pose-angle"] = pa
    for k in ("--size", "--samples", "--crop-size", "--pose-size"):
        vals = [int(p[k]) for p in (vparts if k in ("--size", "--samples") else parts) if k in p]
        if vals:
            a[k] = str(max(vals))
    if parts[0].get("--frame"):
        a["--frame"] = parts[0]["--frame"]
    if parts[0].get("--set"):
        a["--set"] = parts[0]["--set"]          # (the same in every part: they are in one group)
    return a


def view_param(env, a, check):
    words = [RQ_VIEWS if ("--crop" in a or "--poses" in a) else VIEWS_PY]
    for k, v in a.items():
        if k == "--set":
            for x in v:
                words += ["--set", x]
        else:
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
                vp = [p for p in parts if _views_part(p)]
                same = all(p.get("--size") == vp[0].get("--size") and p.get("--samples") == vp[0].get("--samples")
                           for p in vp)
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


LOCK = os.path.join(DATA, "queue.lock")
INBOX = os.path.join(DATA, "inbox")   # threads drop one new file per add or cancel here; only the serving machine writes queue.json
SERVING = False
_lock_depth = 0


class qlock:
    """One writer at a time for queue.json, across processes (an O_EXCL lock file; a lock older than 60 s is a
    dead process's and is broken). Re-entrant, so a locked command can call save()."""
    def __enter__(self):
        global _lock_depth
        if _lock_depth == 0:
            end = time.time() + 45
            while True:
                try:
                    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, f"{os.getpid()} {now()}".encode()); os.close(fd)
                    break
                except FileExistsError:
                    try:
                        if time.time() - os.path.getmtime(LOCK) > 60:
                            os.unlink(LOCK)
                            continue
                    except OSError:
                        pass
                    if time.time() > end:
                        break                      # never block the queue for good: go on without the lock
                    time.sleep(0.05)
        _lock_depth += 1
        return self

    def __exit__(self, *e):
        global _lock_depth
        _lock_depth -= 1
        if _lock_depth == 0:
            try:
                os.unlink(LOCK)
            except OSError:
                pass


def locked(fn):
    @functools.wraps(fn)
    def w(*a, **k):
        with qlock():
            return fn(*a, **k)
    return w


_LISTS = ("requests", "intents")


def _clean(o):
    return json.dumps({k: v for k, v in o.items() if not k.startswith("_")}, sort_keys=True)


def load(raw=False):
    try:
        with open(STATE) as f:
            s = json.load(f)
    except FileNotFoundError:
        s = {"next": 1, "requests": []}
    if SERVING:
        # the serving machine reads the shared folder through a mount that lags its own writes (2026-11-21: a stale
        # read numbered two requests #337 and the first one vanished): its own latest copy wins when it is newer
        try:
            with open(os.path.join(WORK, "queue.local.json")) as f:
                loc = json.load(f)
            if loc.get("seq", 0) > s.get("seq", 0):
                s = loc
        except (OSError, ValueError):
            pass
    s.setdefault("ready", []); s.setdefault("sheet_due", False); s.setdefault("sheets_done", {})
    # what this process saw, so save() can tell its own changes from other processes' (a three-way merge)
    s["_base"] = {"lists": {k: {o["id"]: _clean(o) for o in s.get(k, [])} for k in _LISTS},
                  "top": {k: json.dumps(v, sort_keys=True) for k, v in s.items() if k not in _LISTS and not k.startswith("_")}}
    if not raw and not SERVING:
        _merge_inbox(s)
    return s


def _inbox_ops():
    ops = []
    for f in glob.glob(os.path.join(INBOX, "*.json")):
        try:
            with open(f) as fh:
                ops.append((f, json.load(fh)))
        except (OSError, ValueError):
            pass                                     # (half-written or just taken: next time)
    ops.sort(key=lambda x: (x[1].get("created", ""), x[1].get("token", "")))
    return ops


def _merge_inbox(s):
    """A read-only view: requests threads have dropped in the inbox but the serving machine has not taken yet show
    up as pending, marked _inbox so that save() never writes them."""
    have = {r.get("token") for r in s["requests"] if r.get("token")}
    nxt = max([s["next"]] + [r["id"] + 1 for r in s["requests"]])
    for f, d in _inbox_ops():
        if d.get("op") == "add":
            r = dict(d["request"])
            if r.get("token") in have:
                continue
            have.add(r.get("token"))
            r["id"] = nxt; nxt += 1; r["_inbox"] = True
            s["requests"].append(r)
        elif d.get("op") == "cancel":
            for r in s["requests"]:
                if (d.get("token") and r.get("token") == d["token"]) or (not d.get("token") and r["id"] == d.get("id")):
                    r["state"] = "cancelled"


def admit(s, r):
    """Put a request that arrived in the inbox into the queue: number it, drop a waiting request it replaces, mark the
    intent it answers."""
    r["id"] = max([s["next"]] + [o["id"] + 1 for o in s["requests"]])
    s["next"] = r["id"]
    outs = r["outputs"]
    for o in s["requests"]:   # coalescing: a newer request for the same piece replaces one still waiting
        # (the same thread and branch following the branch tip, or the same branch asking for the same views and
        #  cells at another commit, pinned or not: the stale one is dropped, never rendered)
        if o["state"] in ("pending", "conflict") and o["branch"] == r["branch"] and o["outputs"] == outs and (
                (o["thread"] == r["thread"] and not r.get("sha") and not o.get("sha"))
                or (sorted(o.get("views") or []) == sorted(r["views"]) and o.get("cells") == r["cells"]
                    and o.get("quality") == r["quality"])):
            o["state"] = f"superseded by {r['id']}"
            event("coalesced", None, dropped=o["id"], by=r["id"], branch=r["branch"])
    figs = {f for v in r["views"] for f in re.findall(r"--figure\s+(\S+)", v)}
    for i in s.get("intents", []):    # the request an intent announced has come
        if i["state"] == "open" and i["branch"] == r["branch"] and (i["piece"] in figs or not figs):
            i["state"] = f"sent as #{r['id']}"
    s["next"] += 1
    s["requests"].append(r)


def ingest():
    """Only the serving machine runs this: take what threads dropped in the inbox into queue.json. Threads never
    write queue.json for an add or a cancel, so no other machine's write can lose theirs (2026-11-21: three adds
    and a cancel vanished that way, each thread seeing the shared folder through its own mount)."""
    ops = _inbox_ops()
    if not ops:
        return 0
    with qlock():
        s = load(raw=True)
        toks = {r.get("token") for r in s["requests"] if r.get("token")}
        for f, d in ops:
            if d.get("op") == "add":
                r = dict(d["request"])
                if r.get("token") not in toks:
                    toks.add(r.get("token")); admit(s, r)
            elif d.get("op") == "cancel":
                for r in s["requests"]:
                    if (d.get("token") and r.get("token") == d["token"]) or (not d.get("token") and r["id"] == d.get("id")):
                        r["state"] = "cancelled"
        save(s)
        for f, d in ops:
            try:
                os.unlink(f)
            except OSError:
                pass
    return len(ops)


def drop_in_inbox(op):
    os.makedirs(INBOX, exist_ok=True)
    name = f"{op['op']}-{op['created'].replace(':', '')}-{op.get('token') or uuid.uuid4().hex[:8]}.json"
    tmp = os.path.join(INBOX, "." + name)
    with open(tmp, "w") as f:
        json.dump(op, f); f.flush(); os.fsync(f.fileno())
    os.rename(tmp, os.path.join(INBOX, name))


def merge_disk(s, disk):
    """s is this process's copy (loaded some time ago, perhaps before a slow step); disk is the file now.
    Keep what this process changed, and take everything else from the file: a request another thread added, a
    cancel, a state another process set. Without this the last writer won and lost the others' changes."""
    base = s.get("_base")
    if base is None:
        return s
    out = {k: v for k, v in s.items() if not k.startswith("_")}
    for k, v in disk.items():
        if k in _LISTS or k.startswith("_"):
            continue
        if k in ("next", "next_intent"):
            out[k] = max(out.get(k, 1), v)
        elif k not in out or json.dumps(out[k], sort_keys=True) == base["top"].get(k):
            out[k] = v                                   # unchanged here: the file's value is newer
    for k in _LISTS:
        if k not in s and k not in disk:
            continue
        b = base["lists"].get(k, {})
        mine = {o["id"]: o for o in s.get(k, [])}
        theirs = {o["id"]: o for o in disk.get(k, [])}
        merged = []
        for i in sorted(set(mine) | set(theirs)):
            m, d = mine.get(i), theirs.get(i)
            if m is not None and (i not in b or _clean(m) != b[i]):
                if (k == "requests" and d is not None and d.get("state") == "cancelled" and i in b
                        and json.loads(b[i]).get("state") != "cancelled" and m.get("state") in ("pending", "rendering", "conflict")):
                    merged.append(d)                     # cancelled meanwhile: the cancel stands
                else:
                    merged.append(m)                     # changed or added here
            elif d is not None:
                merged.append(d)                         # unchanged here, or added elsewhere: the file's version
            elif m is not None and i not in b:
                merged.append(m)
        out[k] = merged
    return out


def save(s):
    """Compact JSON, written and then read back: the shared folder is a network mount, and a replace on it once
    left the file cut short. A local copy is kept too. Under the lock, merged with what the file holds now."""
    orig = s
    base = s.get("_base")
    s = {**s, "requests": [{k: v for k, v in r.items() if not k.startswith("_")} for r in s["requests"]
                           if not r.get("_inbox")]}
    os.makedirs(WORK, exist_ok=True)
    for _ in range(5):
        with qlock():
            try:
                with open(STATE) as f:
                    disk = json.load(f)
                s = merge_disk({**s, "_base": base}, disk)
                if s["requests"]:
                    s["next"] = max(s["next"], max(r["id"] for r in s["requests"]) + 1)
            except (OSError, ValueError, KeyError):
                pass
            s["seq"] = s.get("seq", 0) + 1
            data = json.dumps({k: v for k, v in s.items() if not k.startswith("_")}, separators=(",", ":"))
            with open(os.path.join(WORK, "queue.local.json"), "w") as f:
                f.write(data)
            with open(STATE, "w") as f:
                f.write(data); f.flush(); os.fsync(f.fileno())
            time.sleep(0.2)
            with open(STATE) as f:
                if f.read() == data:
                    w = json.loads(data)        # what the file holds now is this process's new base
                    orig["seq"] = w["seq"]
                    orig["_base"] = {"lists": {k: {o["id"]: _clean(o) for o in w.get(k, [])} for k in _LISTS},
                                     "top": {k: json.dumps(v, sort_keys=True) for k, v in w.items()
                                             if k not in _LISTS and not k.startswith("_")}}
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


def same_blob(c1, c2, path):
    a = git("rev-parse", "--verify", "--quiet", f"{c1}:{path}", check=False).stdout.strip()
    return bool(a) and a == git("rev-parse", "--verify", "--quiet", f"{c2}:{path}", check=False).stdout.strip()


def _defs_src(rev, path):
    src = git("show", f"{rev}:{path}", check=False).stdout
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    out = {}
    for n in tree.body:
        name = getattr(n, "name", None) or (ast.unparse(n.targets[0]) if isinstance(n, ast.Assign) else type(n).__name__)
        out.setdefault(name, []).append(ast.get_source_segment(src, n))
    return out


def same_defs(c1, c2, hit):
    """An overlap whose shared definitions read the same on both sides (each side brought the same change)."""
    path, _, names = hit.partition(": ")
    if not path.endswith(".py") or names in ("", "same file"):
        return False
    a, b = _defs_src(c1, path), _defs_src(c2, path)
    return a is not None and b is not None and all(a.get(n) == b.get(n) for n in names.split(", "))


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
    if param and RQ_VIEWS in param:          # (the queue's own script, put in the worktree: its code is an input)
        h.update(open(RQ_VIEWS_SRC, "rb").read())
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
            # (a file both sides changed to the very same text is no clash: two branches that each merged the same
            # main, or share a new helper, as the sleeves and the bodice share armhole.py, 2026-11-19)
            clash = [h for h in clash if not same_blob(b["commit"], r["merged"], h.split(":", 1)[0])
                     and not same_defs(b["commit"], r["merged"], h)]
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
                    if o == "views" and v in (r.get("parts_done") or {}):
                        continue        # (rendered already, in a job it shared with a request that has finished)
                    want.setdefault((o, v), []).append(r)
        for (kind, param), rs in want.items():
            if kind == "frames":
                param = cells_union([r["cells"] for r in rs])
            bjobs.append({"kind": kind, "param": param, "commit": b["commit"], "requests": [r["id"] for r in rs],
                          "combines": [{"id": r["id"], "branch": r["branch"], "sha": r["tip"]} for r in b["requests"]],
                          "key": job_key(b["commit"], kind, param), "prio": PRIORITY[kind],
                          "pen": min(eff_penalty(r) for r in rs),
                          "order": min(r["id"] for r in rs)})
        jobs += combine_views(bjobs, solo)
    # identical jobs (the same key) from different batches are one job for all their requests
    seen = {}
    for j in jobs:
        if j["key"] in seen:
            seen[j["key"]]["requests"] = sorted(set(seen[j["key"]]["requests"]) | set(j["requests"]))
            seen[j["key"]]["pen"] = min(seen[j["key"]].get("pen", 0), j.get("pen", 0))
            seen[j["key"]]["dups"] = seen[j["key"]].get("dups", 0) + 1
        else:
            seen[j["key"]] = j
    jobs = list(seen.values())
    for r in ready:
        r["jobs"] = [j["key"] for j in jobs if r["id"] in j["requests"]]
        r["jobs"] += [k for k in dict.fromkeys((r.get("parts_done") or {}).values()) if k not in r["jobs"]]
        if "cache_hits" not in r and r["jobs"]:       # (the cache-hit rate: what it found ready when first planned)
            r["cache_hits"] = sum(1 for k in r["jobs"] if cached(k))
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
    jobs.sort(key=lambda j: (j["prio"] + j.get("pen", 0), 0 if (j["param"] or "").endswith(CHECK) else 1, j["order"]))
    save(s)
    return main, held, batches, jobs


def show(main, held, batches, jobs):
    print(json.dumps({
        "main": main[:9],
        "held": [{"id": r["id"], "branch": r["branch"], "thread": r["thread"], "why": r["why"]} for r in held],
        "batches": [{"combines": [{"id": r["id"], "branch": r["branch"], "sha": r["tip"][:9], "changes": r["_footprint"],
                                   "not_combined_with": r.get("not_combined_with")} for r in b["requests"]]}
                    for b in batches],
        "jobs": [{k: j[k] for k in ("key", "kind", "param", "prio", "pen", "requests", "cached")} for j in jobs]}, indent=2))


# ------------------------------------------------------------------ running
CHECK = "  [quality check]"
MEMO = os.path.join(WORK, "memo")
ADAPTIVE = os.environ.get("RQ_ADAPTIVE", "0.01") if os.environ.get("RQ_ADAPTIVE", "0.01") not in ("", "0") else None


def command(job, out):
    """Every script runs under rq_run.py: it remembers the head's surface between runs (the eyes' sculpt and the
    ears, most of a build) and, for a check render, drops Cycles to 8 denoised samples."""
    q = shlex.quote
    # (--keep-scene: Cycles keeps the scene between a job's renders; pixel-identical, see rq_run.py)
    run = f"python3 {q(os.path.join(HERE, 'rq_run.py'))} --memo {q(MEMO)} --keep-scene"
    if ADAPTIVE and job["kind"] == "views":       # (finals of the 3D look only: see rq_run.py --adaptive)
        run += f" --adaptive {ADAPTIVE}"
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
    w = shlex.split(view)
    if w and w[0] == VIEWS_PY:          # (every views.py job runs through rq_views.py: geometry checks, timing)
        view = shlex.join([RQ_VIEWS] + w[1:])
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
    if job["kind"] == "views" and (RQ_VIEWS in (job["param"] or "") or VIEWS_PY in (job["param"] or "")):
        shutil.copy(RQ_VIEWS_SRC, os.path.join(wt, RQ_VIEWS))
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
    tele = telemetry(run, rc)
    m = {"ok": rc == 0, "key": job["key"], "kind": job["kind"], "param": job["param"], "commit": job["commit"],
         **tele,
         "combines": job["combines"], "seconds": round(time.time() - run["t0"]), "files": files, "finished": now(),
         "requests": job["requests"], **{k: job[k] for k in ("merged_from", "set", "pieces") if k in job}}
    with open(os.path.join(run["out"], "manifest.json"), "w") as f:
        json.dump(m, f, indent=2)
    dst = os.path.join(RESULTS, job["key"] + ("" if m["ok"] else "-failed"))
    shutil.rmtree(dst, ignore_errors=True); shutil.copytree(run["out"], dst)   # only complete results land here
    shutil.rmtree(run["out"], ignore_errors=True)
    return m


def failure_cause(log, rc):
    """Why a job failed, in a word or two, from the end of its log (for the nightly failure log)."""
    tail = log[-6000:]
    if "[killed: over the" in tail:
        return "killed at the time limit"
    if rc in (-9, 137) or "MemoryError" in tail or "Killed" in tail.splitlines()[-1:]:
        return "out of memory"
    if "Segmentation fault" in tail or rc in (-11, 139):
        return "Blender crashed"
    m = re.findall(r"^(\w*(?:Error|Exception)): (.*)$", tail, re.M)
    if m:
        return f"{m[-1][0]}: {m[-1][1][:120]}"
    m = re.findall(r"^SystemExit: (.*)$|^(?:views\.py|rq_views|outfit|pose_strip): (.*)$", tail, re.M)
    if m:
        return (m[-1][0] or m[-1][1])[:140]
    return f"exit {rc}"


def telemetry(run, rc):
    """John's per-job telemetry (2026-11-19): the queue wait, the phases (Blender's start, loading the scripts, the
    build, the renders, crops, poses, the geometry checks), the settings and solver code it ran with, the cause
    of a failure, and the geometry checks' numbers."""
    job, out = run["job"], run["out"]
    total = time.time() - run["t0"]
    try:
        log = open(os.path.join(out, "render.log"), errors="replace").read()
    except OSError:
        log = ""
    t = {}
    try:
        t = json.load(open(os.path.join(out, "timing.json")))
    except (OSError, ValueError):
        pass
    b = re.search(r"^build (\d+) s", log, re.M)
    geo_s = t.get("geometry") or 0
    phases = {"startup": round(max(0.0, total - sum(t.get(k) or 0 for k in ("load", "views", "crops", "poses"))), 1)
              if t else None, "load": t.get("load"), "build": round(int(b[1]) - geo_s, 1) if b else None,
              "geometry": t.get("geometry"), "body": t.get("body"),
              "render": round(t["views"] - int(b[1]), 1) if t.get("views") is not None and b else None,
              "crops": t.get("crops"), "poses": t.get("poses")}
    s = load()
    created = [ts(r["created"]) for r in s["requests"] if r["id"] in job["requests"] and r.get("created")]
    p = job["param"] or ""
    settings = {"quality": "check" if p.endswith(CHECK) else "final",
                "size": (re.search(r"--size (\d+)", p) or [None, None])[1],
                "samples": "8 (check)" if p.endswith(CHECK) else (re.search(r"--samples (\d+)", p) or [None, "default"])[1],
                "scene": (re.search(r"--figure (\S+)", p) or [None, job["kind"]])[1],
                "pieces": job.get("pieces"),
                "mannequin": "--mannequin" in p}
    solver = {}
    for path in git("ls-tree", "-r", "--name-only", job["commit"], "--", "blender/").stdout.split():
        if re.search(r"(solver|settle|sew|xpbd|drape)\w*\.py$", path):
            solver[path] = git("rev-parse", f"{job['commit']}:{path}").stdout.strip()[:10]
    geo = None
    try:
        g = json.load(open(os.path.join(out, "geometry.json")))
        geo = {"body": g.get("body"), "seconds": g.get("seconds"),
               "garments": {k: {x: v.get(x) for x in ("verts", "inside_pct", "max_depth", "stretch_p99", "squash_p1")}
                            for k, v in (g.get("garments") or {}).items()}}
    except (OSError, ValueError):
        pass
    return {"queue_wait": round(run["t0"] - min(created)) if created else None, "phases": phases,
            "settings": settings, "solver": solver, "geometry": geo,
            "cause": None if rc == 0 else failure_cause(log, rc),
            "warnings": len(re.findall(r"^(?:WARNING|Warning|warning)\b|rq geometry: skipped", log, re.M))}


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
            # the parts of this request a finished job covered: when the job it shared with another request is
            # done and that request leaves, this one is planned again, and without this its part rendered twice
            # (#216 and #217, 2026-11-19: the sleeves pose strip, in 286db5 with #216's turn and again alone)
            for k in r["jobs"]:
                if cached(k):
                    try:
                        m = json.load(open(os.path.join(RESULTS, k, "manifest.json")))
                    except (OSError, ValueError):
                        continue
                    for v in (m.get("merged_from") or [m.get("param")]):
                        if m.get("kind") == "views" and v in [x + (CHECK if r.get("quality") == "check" else "")
                                                              for x in r.get("views") or []]:
                            r.setdefault("parts_done", {})[v] = k
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
    if "HABIT_PIECES" in p or "habit_all" in p or "petticoat" in p:
        return 7.0                  # (nightly 2026-11-20: two habit/petticoat builds side by side, #255 and #256, one killed at 142 s)
    if "--mannequin" in p:
        return 2.0                                     # (no head, face or hair: a light build)
    return 7.6 if "warden" in p else 4.0


def cmd_serve(a):
    """Run jobs in priority order until a request (or a sheet) finishes, then exit so the thread can deliver."""
    global SERVING
    SERVING = True      # (this process sees only what ingest() has taken into queue.json)
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
            ingest()
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
                event("done" if m["ok"] else "failed", job["key"], seconds=m["seconds"], cause=m.get("cause"),
                      phases=m.get("phases"), quality=(m.get("settings") or {}).get("quality"),
                      threads=sorted({r["thread"] for r in load()["requests"] if r["id"] in job["requests"]}))
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
          solo=any(r.get("solo") for r in rs), check=(job["param"] or "").endswith(CHECK),
          crops=len((re.search(r"--crop\s+(\S+)", job["param"] or "") or [None, ""])[1].split(",")) if "--crop " in (job["param"] or "") else 0,
          poses="--poses " in (job["param"] or ""),
          only_extras=not re.search(r"--(turn|head|walk|run|cells)\b", job["param"] or ""))


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


@locked
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


@locked
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


def expand_view(v):
    """The named views a thread may ask for: "pose-strip ARGS" and "crop SPECS ARGS", and views.py with --crop or
    --poses, all run as the queue's rq_views.py."""
    w = shlex.split(v)
    env = []
    while w and (w[0] == "env" or ("=" in w[0] and not w[0].startswith("-"))):
        env.append(w.pop(0))
    if env and env[0] != "env":
        env.insert(0, "env")
    if not w:
        return v
    if w[0] == "pose-strip":
        rest = w[1:]
        if "--poses" not in rest:
            rest += ["--poses", "all"]
        if "--mannequin" not in rest:
            rest.append("--mannequin")
        return shlex.join(env + [RQ_VIEWS] + rest)
    if w[0] == "crop":
        if len(w) < 2:
            sys.exit("crop needs the regions: crop shoulder,hem --figure F (see rq_views.py for the names)")
        return shlex.join(env + [RQ_VIEWS] + w[2:] + ["--crop", w[1]])
    if w and w[0] == VIEWS_PY and ("--crop" in w or "--poses" in w):
        return shlex.join(env + [RQ_VIEWS] + w[1:])
    return v


def is_extra(v):
    w = shlex.split(v)
    return RQ_VIEWS in w and not any(f in w for f in ("--turn", "--head", "--walk", "--run", "--cells"))


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
    a.view = [expand_view(v) for v in (a.view or [])] or None
    if a.quality is None:     # crops and pose strips are cheap drafts unless the request says final
        a.quality = "check" if a.view and all(is_extra(v) for v in a.view) else "final"
    unreviewed = False
    if a.quality == "final" and not a.reviewed:
        # John, 2026-11-19: with nothing rendering or waiting, a request may skip the reviewer (a crash costs nobody a wait)
        busy = [o["id"] for o in s["requests"] if o["state"] in ("pending", "rendering", "conflict")]
        if busy:
            sys.exit("a final-quality request goes through the reviewer thread (see the README, 'How to ask for a render'), "
                     f"unless the queue is idle; #{', #'.join(map(str, busy))} is rendering or waiting now. "
                     "A check-quality request you can add yourself: add --quality check")
        unreviewed = True
    before_only_from_cache(a, outs)
    tentative = max([s["next"]] + [o["id"] + 1 for o in s["requests"]])
    r = {"id": tentative, "token": uuid.uuid4().hex[:12], "thread": a.thread, "branch": a.branch, "sha": a.sha,
         "outputs": outs, "views": a.view or [], "cells": a.cells, "mode": a.mode, "note": a.note, "state": "pending",
         "created": now(), "quality": a.quality}
    r["repeat"] = repeat_count(s, a.thread, r["views"], r["quality"], time.time()) if r["views"] else 0
    for w in hog_warnings(s, thread=a.thread):     # (the requesting thread hears it in the add's answer)
        print(f"rq warning: {w['text']}", flush=True)
        event("hog", None, thread=a.thread, rule=w["rule"], value=w["value"], request=r["id"])
    # One new file in the inbox, never a write to queue.json: the serving machine takes it in within seconds and
    # numbers it. (Each thread sees the shared folder through its own mount, and a write to queue.json from here lost
    # to another machine's three times on 2026-11-21.)
    drop_in_inbox({"op": "add", "created": r["created"], "token": r["token"], "request": r})
    if unreviewed:
        event("unreviewed", None, thread=a.thread, request=r["id"])
        print("rq note: the queue was idle, so this final request skipped the reviewer", flush=True)
    print(json.dumps(r, indent=2))
    print(f"rq note: queued (token {r['token']}). The renderer takes it into the queue within seconds and numbers it; the "
          f"id above can change, so read the real id by ping. To cancel before then: cancel {r['id']} --token {r['token']}", flush=True)
    if r["repeat"]:
        steps = min(r["repeat"], PEN["max"])
        print(f"rq note: this is repeat {r['repeat']} of the same views: it runs {steps} step{'s' * (steps > 1)} behind other "
              f"requests (each {PEN['age_s'] // 60} minutes it waits wins one step back; an idle queue runs it at once; "
              f"a different request, or {PEN['quiet_s'] // 60} quiet minutes, resets it)", flush=True)
        event("penalty", None, thread=a.thread, request=r["id"], repeat=r["repeat"], steps=steps)
    lp = last_ping(a.thread)
    if lp is None or time.time() - lp > PING_FRESH_S:
        print(f"rq note: no ping from this thread in the last {PING_FRESH_S // 60} minutes. Before the next request, run "
              f"`python3 /mnt/project-files/render-queue/rq.py ping --thread {a.thread}` to see what is rendering, "
              f"what is waiting and whether your last result is back", flush=True)
        event("unpinged", None, thread=a.thread, request=r["id"])
    for w in hog_warnings(s, thread=a.thread):     # (the requesting thread hears it in the add's answer)
        print(f"rq warning: {w['text']}", flush=True)
        event("hog", None, thread=a.thread, rule=w["rule"], value=w["value"], request=r["id"])


# John, 2026-11-19: warn when one thread hogs the queue (the boots: seven near-identical 3/4 renders in an afternoon).
# A warning only: nothing is blocked or reordered. Thresholds from the 2026-11-19 telemetry, when four threads
# shared the queue and the busiest hour gave one thread 68% of 1,300 s of rendering.
HOG = {"wait_s": 120, "hour_share": 0.5, "hour_min_s": 600,     # over half the last hour's render time, and at least 10 minutes
       "day_share": 0.35, "day_min_s": 1800,     # over 35% of the last day's (fair share with three threads or
       #                                           more is a third), and at least 30 minutes
       "repeat": 4, "repeat_h": 3,               # the same views asked 4 times in 3 hours
       "streak": 5}                              # 5 submissions in a row while another thread had one waiting


_SECS = {}      # job key -> (seconds, requests it served): a finished job's manifest never changes


def _job_seconds(k):
    if k in _SECS:
        return _SECS[k]
    for d in (k, k + "-failed"):
        mp = os.path.join(RESULTS, d, "manifest.json")
        if os.path.exists(mp):
            try:
                m = json.load(open(mp))
            except ValueError:
                return (0.0, 1)        # (being written: not cached, read again next time)
            _SECS[k] = (m.get("seconds", 0), max(1, len(m.get("requests") or [1])))
            return _SECS[k]
    return (0.0, 1)


def request_seconds(r):
    """A request's share of render time: each of its jobs' seconds, split among the requests that job served."""
    t = 0.0
    for k in r.get("jobs") or []:
        sec, n = _job_seconds(k)
        t += sec / n
    return t


def repeat_count(s, thread, views, quality, at):
    """How many requests in a row, just before this one, this thread sent for the same views at the same quality.
    A different request from the thread, or a gap over PEN["quiet_s"], ends the run."""
    key = (sorted(views or []), quality)
    n, prev = 0, at
    for o in sorted((o for o in s["requests"] if o["thread"] == thread and o.get("created")), key=lambda o: -o["id"]):
        if o["state"].startswith(("superseded", "cancelled")):
            continue
        if (sorted(o.get("views") or []), o.get("quality")) != key or prev - ts(o["created"]) > PEN["quiet_s"]:
            break
        n += 1; prev = ts(o["created"])
    return n


def eff_penalty(r, at=None):
    """The steps a request is behind now: its repeat count (at most PEN["max"]) less one for every PEN["age_s"] waited."""
    n = min(r.get("repeat", 0), PEN["max"])
    if not n or r["state"] not in ("pending", "conflict") or not r.get("created"):
        return n if r["state"] in ("pending", "conflict") else 0
    return max(0, n - int(((at or time.time()) - ts(r["created"])) // PEN["age_s"]))


HOG_RULES = ("over {hour_share:.0%} of the last hour's render time (at least {hour_min_s} s) or {day_share:.0%} of the "
             "day's (at least {day_min_s} s) while another thread waited {wait_s} s or more; the same views {repeat} times "
             "in {repeat_h} h; {streak} requests in a row while another thread had one waiting. Warnings only: the "
             "queue never blocks or reorders.")
_STARTS = {}


def _starts():
    """Request id to the time its first render job started, from the event log (cached by file size)."""
    try:
        size = os.path.getsize(EVENTS)
    except OSError:
        return {}
    if _STARTS.get("size") != size:
        st = {}
        with open(EVENTS) as f:
            for line in f:
                if '"start"' not in line:
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("ev") == "start":
                    for i in e.get("requests") or []:
                        st.setdefault(i, ts(e["t"]))
        _STARTS.update(size=size, map=st)
    return _STARTS["map"]


def hog_warnings(s, at=None, thread=None):
    """The warnings in force at time `at` (now), for one thread or all: share of the last hour's and the last day's
    render time while another thread waited at least 2 minutes for a slot, the same views asked again and again, and long runs of requests in
    a row while another thread waits."""
    at = at or time.time()
    rs = [r for r in s["requests"] if r.get("created") and ts(r["created"]) <= at]
    name = lambda t: THREAD_NAMES.get(t, t[-6:])   # noqa: E731
    starts = _starts()
    out = []
    for label, span, share, floor in (("hour", 3600, HOG["hour_share"], HOG["hour_min_s"]),
                                      ("day", 86400, HOG["day_share"], HOG["day_min_s"])):
        win = [r for r in rs if ts(r["created"]) > at - span]
        secs = {}
        for r in win:
            secs[r["thread"]] = secs.get(r["thread"], 0) + request_seconds(r)
        tot = sum(secs.values())
        waited = {r["thread"] for r in win if r["id"] in starts and starts[r["id"]] - ts(r["created"]) >= HOG["wait_s"]}
        if len(secs) < 2 or not tot:
            continue          # (alone on the queue is no hogging)
        for t, v in secs.items():
            if v / tot > share and v >= floor and waited - {t}:   # (only while someone else waited for a slot)
                out.append({"thread": t, "name": name(t), "rule": f"{label}_share", "value": round(v / tot, 2),
                            "seconds": round(v), "of": round(tot),
                            "text": f"{name(t)} used {v / tot:.0%} of the last {label}'s render time "
                                    f"({fmt_s(v)} of {fmt_s(tot)}) while {", ".join(sorted(name(o) for o in waited - {t}))} waited for a slot"})
    rep = {}
    for r in rs:
        if ts(r["created"]) > at - HOG["repeat_h"] * 3600 and r.get("views") and not r["state"].startswith("superseded"):
            rep.setdefault((r["thread"], tuple(sorted(r["views"]))), []).append(r["id"])
    for (t, _), ids in rep.items():
        if len(ids) >= HOG["repeat"]:
            out.append({"thread": t, "name": name(t), "rule": "repeat", "value": len(ids), "requests": ids,
                        "text": f"{name(t)} asked for the same views {len(ids)} times in {HOG['repeat_h']} hours "
                                f"(#{', #'.join(map(str, ids))}): check a cheaper crop, or what changed, before the next"})
    srt = sorted(rs, key=lambda r: r["id"])
    # (one submission: the requests a thread sent within 15 s, such as the three parts of one set fit test)
    subs = []
    for r in srt:
        if subs and subs[-1]["thread"] == r["thread"] and ts(r["created"]) - ts(subs[-1]["created"]) <= 15:
            continue
        subs.append(r)
    run, best = [], {}
    for r in subs:
        run = run + [r] if run and run[-1]["thread"] == r["thread"] else [r]
        hi = ts(r["created"])
        # (another thread had a request waiting or rendering at the moment one of the run's requests was sent)
        others = any(o["thread"] != r["thread"] and o.get("created") and not o["state"].startswith(("cancelled", "superseded"))
                     and ts(o["created"]) <= ts(x["created"]) and
                     (not o.get("finished") or ts(o["finished"]) > ts(x["created"])) for o in srt for x in run)
        if len(run) >= HOG["streak"] and others and hi > at - 3600 * HOG["repeat_h"]:
            best[r["thread"]] = [x["id"] for x in run]
    for t, ids in best.items():
        out.append({"thread": t, "name": name(t), "rule": "streak", "value": len(ids), "requests": ids,
                    "text": f"{name(t)} sent {len(ids)} requests in a row (#{ids[0]} to #{ids[-1]}) while another "
                            f"thread had one waiting or rendering"})
    return [w for w in out if thread is None or w["thread"] == thread]


def fmt_s(v):
    v = round(v)
    return f"{v} s" if v < 90 else f"{v // 60} min {v % 60:02d} s" if v < 3600 else f"{v // 3600} h {v % 3600 // 60:02d} min"


def cmd_hogs(a):
    s = load()
    ws = hog_warnings(s)
    print(json.dumps(ws, indent=2) if ws else "no thread is hogging the queue")


# John, 2026-11-19: a ping any thread can run, and threads are told to run it before they ask for more renders.
PING_FRESH_S = 900    # a request sent without a ping from its thread in the last 15 minutes gets a reminder


def _est_seconds(r, done):
    """About how long a request will render: the median of the last finished requests like it (same quality and
    the same kind of render), else of the same quality, else 3 minutes."""
    kind = tuple(_view_kinds(r.get("views"), r.get("outputs") or []))
    for pick in (lambda o: o.get("quality") == r.get("quality") and tuple(_view_kinds(o.get("views"), o.get("outputs") or [])) == kind,
                 lambda o: o.get("quality") == r.get("quality")):
        xs = [x for x in (request_seconds(o) for o in done[-60:] if pick(o)) if x > 0]
        if len(xs) >= 3:
            return statistics.median(xs)
    return 180.0


def queue_state(s, thread=None, at=None):
    """What the queue is doing right now, and where a thread's own requests stand: the answer to a ping."""
    at = at or time.time()
    starts = _starts()
    name = lambda t: THREAD_NAMES.get(t, t[-6:])   # noqa: E731
    done = [r for r in s["requests"] if r["state"] == "done"]
    rendering, waiting = [], []
    for r in s["requests"]:
        if r["state"] == "rendering":
            st = starts.get(r["id"], ts(r["created"]))
            est = _est_seconds(r, done)
            rendering.append({"id": r["id"], "thread": name(r["thread"]), "mine": r["thread"] == thread,
                              "running_s": round(at - st), "left_s": round(max(30, est - (at - st))), "note": r.get("note")})
        elif r["state"] == "pending" or r["state"] == "conflict":
            waiting.append({"id": r["id"], "thread": name(r["thread"]), "mine": r["thread"] == thread,
                            "state": r["state"], "waited_s": round(at - ts(r["created"])), "est_s": round(_est_seconds(r, done)),
                            "penalty": eff_penalty(r, at), "repeat": r.get("repeat", 0),
                            "views": r.get("views"), "note": r.get("note")})
    # (a rough clock: PARALLEL slots, each job taking its estimate; the queue starts jobs in id order)
    slots = sorted([x["left_s"] for x in rendering] + [0] * max(0, PARALLEL - len(rendering)))[:PARALLEL] or [0]
    for w in sorted(waiting, key=lambda w: (w["penalty"], w["id"])):    # (the queue starts the least penalised first)
        if w["state"] != "pending":
            continue
        i = min(range(len(slots)), key=slots.__getitem__)
        w["starts_in_s"] = slots[i]; slots[i] += w["est_s"]
    new_wait = min(slots)
    busy = "idle" if not rendering and not waiting else "light" if len(rendering) + len(waiting) <= PARALLEL else "busy"
    out = {"at": now(), "busy": busy, "slots": PARALLEL, "rendering": rendering, "waiting": waiting,
           "new_request_starts_in_s": round(new_wait), "sheet_due": s.get("sheet_due", False),
           "open_intents": [{"id": i["id"], "thread": name(i["thread"]), "piece": i.get("piece")}
                            for i in s.get("intents", []) if i.get("state") == "open"]}
    if thread:
        mine = [r for r in s["requests"] if r["thread"] == thread]
        last = next((r for r in reversed(mine) if r["state"] in ("done", "failed")), None)
        out["you"] = {
            "name": name(thread),
            "waiting": [w["id"] for w in waiting if w["mine"]],
            "rendering": [x["id"] for x in rendering if x["mine"]],
            "last_result": last and {"id": last["id"], "state": last["state"], "finished": last.get("finished"),
                                     "results": [f"render-queue/results/{k}/" for k in last.get("jobs") or []],
                                     "note": last.get("note")},
            "warnings": [w["text"] for w in hog_warnings(s, at=at, thread=thread)],
        }
        adv = []
        if out["you"]["rendering"] or out["you"]["waiting"]:
            adv.append(f"you already have #{', #'.join(map(str, out['you']['rendering'] + out['you']['waiting']))} in the queue: "
                       "wait for it, or send a newer request for the same views, which replaces a waiting one")
        pens = [w for w in waiting if w["mine"] and w["penalty"]]
        if pens:
            adv.append("your waiting request is behind others because it repeats the same views: " +
                       ", ".join(f"#{w['id']} by {w['penalty']}" for w in pens) +
                       "; change the views, or wait, and it moves up one step every "
                       f"{PEN['age_s'] // 60} minutes")
        if out["you"]["warnings"]:
            adv.append("you are flagged for hogging: send a cheaper crop or check that your edit changes the build before the next full render")
        if busy == "busy":
            adv.append(f"the queue is busy: a new request would start in about {fmt_s(new_wait)}; prefer a check-quality crop")
        if busy == "idle":
            adv.append("the queue is idle: a final-quality request can skip the reviewer, add it yourself with rq.py add")
        out["you"]["advice"] = adv or ["the queue has room: go ahead"]
    return out


def last_ping(thread):
    """When this thread last pinged, from the event log (None if never)."""
    t = None
    try:
        with open(EVENTS) as f:
            for line in f:
                if '"ping"' in line and thread in line:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if e.get("ev") == "ping" and e.get("thread") == thread:
                        t = ts(e["t"])
    except OSError:
        pass
    return t


def cmd_ping(a):
    for i in range(5):    # (another container may be mid-write on the shared mount: read again)
        try:
            s = load(); break
        except ValueError:
            time.sleep(1 + i)
    else:
        sys.exit("the queue file is being written: ping again in a few seconds")
    st = queue_state(s, thread=a.thread)
    if a.thread:
        try:
            event("ping", None, thread=a.thread)
        except OSError:
            pass      # (a ping is read-only for the queue; a thread that cannot append to the log still gets its answer)
    if a.json:
        print(json.dumps(st, indent=2)); return
    print(f"render queue at {st['at']}: {st['busy']}, {len(st['rendering'])} rendering and "
          f"{sum(w['state'] == 'pending' for w in st['waiting'])} waiting on {st['slots']} slots; "
          f"a new request would start in about {fmt_s(st['new_request_starts_in_s'])}")
    for x in st["rendering"]:
        print(f"  rendering #{x['id']} {x['thread']}: {fmt_s(x['running_s'])} in, about {fmt_s(x['left_s'])} left  {x['note'] or ''}")
    for w in st["waiting"]:
        when = f"starts in about {fmt_s(w['starts_in_s'])}" if "starts_in_s" in w else w["state"]
        pen = f" (priority -{w['penalty']}: repeat {w['repeat']})" if w["penalty"] else ""
        print(f"  waiting   #{w['id']} {w['thread']}: {when}{pen}  {w['note'] or ''}")
    if st["open_intents"]:
        print("  announced: " + "; ".join(f"intent #{i['id']} {i['thread']} ({i['piece']})" for i in st["open_intents"]))
    if st["sheet_due"]:
        print("  the full sheet is due and runs when the queue goes idle")
    y = st.get("you")
    if y:
        lr = y["last_result"]
        print(f"you ({y['name']}): rendering {y['rendering'] or 'none'}, waiting {y['waiting'] or 'none'}; last result "
              + (f"#{lr['id']} {lr['state']} {lr['finished'] or ''} {' '.join(lr['results'])}" if lr else "none"))
        for w in y["warnings"]:
            print(f"  warning: {w}")
        for v in y["advice"]:
            print(f"  advice: {v}")


@locked
def cmd_ready(a):
    s = load()
    if a.branch not in s["ready"]:
        s["ready"].append(a.branch)
    save(s)


@locked
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
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBA3b81CneeuYShgwNpSXbXe": "Sleeves",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBQkfRWVrSdTJkEPgPFsn9m6": "Hose and boots",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBBfxoRfhjYNDgEMgK2GDcKJ": "Riding skirt",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nB4hFTQqChHhnYciZn4Te2QE": "Vest, chemise and doublet",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBWHNmAC96iG7dkSuFhMvCv4": "Head and neck",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nB8K2NHFFcGop9VHSKeLiCHR": "Bodice",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBPFTNW92soZDZD4MFuoM2uf": "Petticoat",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nB4anKPy38sg1qeMmhbwzNnt": "Corset",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBYUips3shtbgvsGwsUspf95": "Cloak",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBQCTaqo5NACgyA2VUZDR8Rq": "Necklaces",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBPFdbEQxN22vaxHSoyHyetD": "Ranger hair",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBG3u7qgVcdd4PjmWwFqtjfQ": "Cape physics",
    "cmsg_01CnnKqGvuWEUi2G9X9cb9nBDY4MDu8mY63XYh5NxKA37m": "Mannequin anatomy",
}


def _view_kinds(views, outputs):
    """What a request renders, in plain words, for the tracking page's filter."""
    out = [o for o in outputs if o != "views"]
    for v in views or []:
        w = v.split()
        if w and w[0] == "pose-strip" or "--poses" in w:
            out.append("pose strip")
        if w and w[0] == "crop" or "--crop" in w:
            out.append("crop")
        if "--turn" in w:
            out.append("turn")
        if "--head" in w:
            out.append("close-up")
        if "--walk" in w or "--run" in w:
            out.append("walk")
    return list(dict.fromkeys(out)) or list(outputs)


def _figures(views):
    out = []
    for v in views or []:
        w = v.split()
        if "--figure" in w and w.index("--figure") + 1 < len(w):
            out.append(w[w.index("--figure") + 1])
    return list(dict.fromkeys(out))


def cmd_export(a):
    """Write the queue as JSON rows for the tracking page: one row per request, newest first."""
    s = load()
    rows = []
    users = {}
    for r in s["requests"]:
        for k in r.get("jobs") or []:
            users.setdefault(k, set()).add(r["id"])
    shared = {k for k, ids in users.items() if len(ids) > 1}
    sets_of = {k for k in users if "OUTFIT=" in k or os.path.exists(os.path.join(RESULTS, k, "manifest.json"))
               and json.load(open(os.path.join(RESULTS, k, "manifest.json"))).get("set")}
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
            "piece": ", ".join(_figures(r.get("views"))),
            "quality": r.get("quality") or ("final" if "views" in r["outputs"] else ""),
            "repeat": r["repeat"] if "repeat" in r else (repeat_count({"requests": [o for o in s["requests"] if o["id"] < r["id"]]},
                                                                       r["thread"], r.get("views"), r.get("quality"), ts(r["created"]))
                                                        if r.get("views") and r.get("created") else 0),
            "priority": -eff_penalty(r) if state in ("pending", "conflict") else 0,
            "kind": ", ".join(_view_kinds(r.get("views"), r["outputs"])),
            "combined": "set" if any(k in sets_of for k in r.get("jobs") or []) else
                        ("shared" if any(k in shared for k in r.get("jobs") or []) else "alone"),
        })
    rows.sort(key=lambda x: -x["id"])
    sched = [{"item": "Full sheet, .blend and world model", "when": "Daily, 2:41 AM Toronto, only when the queue is idle",
              "sheet_due": s["sheet_due"], "ready_branches": ", ".join(s["ready"]) or "none"}]
    with open(a.out, "w") as f:
        json.dump({"rows": rows, "meta": [{"label": "Written by", "value": "rq.py export, from the render queue's queue.json"},
                                          {"label": "Exported", "value": now()}]}, f, indent=1)
    hogs = [{k: w[k] for k in ("name", "rule", "value", "text")} for w in hog_warnings(s)]
    with open(a.out.replace(".json", "-hogs.json"), "w") as f:
        json.dump({"rows": hogs, "meta": [{"label": "Rules", "value": HOG_RULES.format(**HOG)}, {"label": "Exported", "value": now()}]}, f, indent=1)
    with open(a.out.replace(".json", "-schedule.json"), "w") as f:
        json.dump({"rows": sched, "meta": [{"label": "Exported", "value": now()}]}, f, indent=1)
    print(a.out, len(rows), "rows")


def cmd_cancel(a):
    drop_in_inbox({"op": "cancel", "created": now(), "id": a.id, "token": a.token})


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
    q.add_argument("--quality", choices=("final", "check"), default=None, help="default: check for crops and pose strips alone, final otherwise")
    q.add_argument("--reviewed", action="store_true", help="the reviewer thread passed this request (needed for final quality)")
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
    cp = sub.add_parser("cancel")
    cp.add_argument("id", type=int)
    cp.add_argument("--token", default=None, help="the token add printed, when the id may have changed")
    sub.add_parser("export").add_argument("out")
    sub.add_parser("hogs")
    q = sub.add_parser("ping", help="what the queue is doing now, and where your own requests stand (read-only)")
    q.add_argument("--thread", help="your thread id, for your own requests, last result and warnings")
    q.add_argument("--json", action="store_true")
    a = p.parse_args()
    globals()["cmd_" + a.cmd.replace("-", "_")](a)


if __name__ == "__main__":
    main()
