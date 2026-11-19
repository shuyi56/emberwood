"""Throughput, waiting and idle statistics for the render queue.

    python3 rq_stats.py [--hours N] [--json OUT]

Reads queue.json (the requests), results/*/manifest.json (every job that finished: how long it ran and when it
ended) and events.jsonl (starts, warnings, kills and sheet preemptions, written by serve since 2026-11-18).
--hours limits everything to the last N hours (default: all history). --json also writes the numbers as JSON.

What the numbers mean:
  wait        a request's time from being queued to its first job starting. This is the time a thread is blocked
              on the queue before anything happens for it. A request answered from the cache waits 0.
  turnaround  queued to finished: what the thread actually waits for its pictures.
  busy        time with at least one job running; idle is the rest of the window.
  idle while waiting
              time with no job running although a request was pending. Should be zero; anything here is a
              scheduler gap (a serve not restarted, a thread slow to queue).
  blocked     time a request waited while every slot (or the memory guard) was taken.
"""
import argparse
import datetime as dt
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("RQ_HOME", HERE)   # queue.json, results/, events.jsonl: next to the code unless RQ_HOME says
STATE = os.path.join(DATA, "queue.json")
RESULTS = os.path.join(DATA, "results")
EVENTS = os.path.join(DATA, "events.jsonl")


def ts(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp() if s else None


def fmt(sec):
    if sec is None:
        return "-"
    sec = int(round(sec))
    if sec < 90:
        return f"{sec} s"
    if sec < 5400:
        return f"{sec / 60:.1f} min"
    return f"{sec / 3600:.1f} h"


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def union(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def clip(intervals, lo, hi):
    return [(max(a, lo), min(b, hi)) for a, b in intervals if b > lo and a < hi]


def total(intervals):
    return sum(b - a for a, b in intervals)


def load_jobs():
    jobs = {}
    for mp in glob.glob(os.path.join(RESULTS, "*", "manifest.json")):
        try:
            m = json.load(open(mp))
        except (OSError, ValueError):
            continue
        end = ts(m.get("finished"))
        if end is None:
            continue
        jobs[m["key"]] = {"key": m["key"], "kind": m.get("kind"), "param": m.get("param") or "", "ok": m.get("ok"),
                          "start": end - m.get("seconds", 0), "end": end, "seconds": m.get("seconds", 0),
                          "requests": [c["id"] for c in m.get("combines", [])], "dir": os.path.dirname(mp)}
    return jobs


def load_events():
    ev = []
    if os.path.exists(EVENTS):
        for line in open(EVENTS):
            try:
                ev.append(json.loads(line))
            except ValueError:
                pass
    return ev


def label(job):
    p = job["param"]
    if job["kind"] == "sheet":
        return "full sheet"
    for flag in ("--figure ",):
        if flag in p:
            fig = p.split(flag)[1].split()[0]
            bits = [b for b in ("--mannequin", "--turn", "--head", "--walk", "--pieces", "--angles", "--only", "--no-body")
                    if b in p]
            return f"{fig} {' '.join(b.lstrip('-') for b in bits)}".strip()
    return os.path.basename(p.split()[0]) if p else job["kind"]


def compute(hours=None, names=None):
    names = names or {}
    now = dt.datetime.now(dt.timezone.utc).timestamp()
    lo = now - hours * 3600 if hours else 0
    s = json.load(open(STATE))
    jobs = load_jobs()
    events = load_events()
    reqs = [r for r in s["requests"] if r.get("created") and ts(r["created"]) >= lo]
    if not hours and reqs:
        lo = min(ts(r["created"]) for r in reqs)
    window = now - lo

    # job intervals: finished jobs from the manifests, running ones from start events with no end yet
    ended = {e["key"] for e in events if e["ev"] in ("done", "failed", "killed", "stopped")}
    running = {e["key"]: e for e in events if e["ev"] == "start" and e["key"] not in ended and e["key"] not in jobs}
    iv = [(j["start"], j["end"]) for j in jobs.values()] + [(ts(e["t"]), now) for e in running.values()]
    busy = union(clip(iv, lo, now))

    per_req = {}
    for j in jobs.values():
        for rid in j["requests"]:
            per_req.setdefault(rid, []).append(j)
    waits, turns, rows = [], [], []
    pending_iv = []
    for r in reqs:
        c, f = ts(r["created"]), ts(r.get("finished"))
        js = per_req.get(r["id"], [])
        first = min((j["start"] for j in js), default=None)
        if r["state"] == "cancelled":
            continue
        if first is None and f is not None:
            first = c                                   # answered from the cache
        if first is not None and first < c:
            first = c                                   # its job was already running for another request
        wait = (first - c) if first is not None else (now - c if r["state"] in ("pending", "rendering") else None)
        turn = (f - c) if f else None
        if wait is not None and r["state"] not in ("pending",):
            waits.append(wait)
        if turn is not None:
            turns.append(turn)
        pending_iv.append((c, first if first is not None else now))
        rows.append({"id": r["id"], "thread": names.get(r["thread"], r["thread"][-6:]), "state": r["state"],
                     "wait": wait, "turnaround": turn, "jobs": len(js),
                     "render": sum(j["seconds"] for j in js)})
    waiting = union(clip(pending_iv, lo, now))
    idle_waiting = total(waiting) - sum(total(clip(busy, a, b)) for a, b in waiting)

    # concurrency while busy
    pts = sorted([(a, 1) for a, b in clip(iv, lo, now)] + [(b, -1) for a, b in clip(iv, lo, now)])
    conc, last, acc, peak = 0, None, 0.0, 0
    for t, d in pts:
        if last is not None and conc > 0:
            acc += conc * (t - last)
        conc += d; peak = max(peak, conc); last = t

    jl = [j for j in jobs.values() if j["end"] >= lo]
    secs = [j["seconds"] for j in jl if j["ok"]]
    by_thread = {}
    for row in rows:
        b = by_thread.setdefault(row["thread"], {"requests": 0, "wait": 0.0, "turnaround": 0.0, "render": 0, "max_wait": 0})
        b["requests"] += 1; b["wait"] += row["wait"] or 0; b["turnaround"] += row["turnaround"] or 0
        b["render"] += row["render"]; b["max_wait"] = max(b["max_wait"], row["wait"] or 0)
    blocked_now = [row for row in rows if row["state"] in ("pending", "rendering")]
    evc = lambda k: sum(1 for e in events if e["ev"] == k and ts(e["t"]) >= lo)  # noqa: E731
    kinds = {}
    for j in jl:
        k = label(j)
        d = kinds.setdefault(k, [])
        d.append(j["seconds"])
    return {
        "window": {"from": dt.datetime.fromtimestamp(lo, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "hours": round(window / 3600, 1)},
        "requests": {"total": len(rows), "done": sum(r["state"] == "done" for r in rows),
                     "failed": sum(r["state"] == "failed" for r in rows),
                     "from_cache": sum(1 for r in rows if r["jobs"] == 0 and r["state"] == "done"),
                     "per_day": round(len(rows) / max(window / 86400, 1 / 24), 1)},
        "wait": {"median": statistics.median(waits) if waits else None, "p90": pct(waits, 0.9),
                 "max": max(waits, default=None), "total": sum(waits)},
        "turnaround": {"median": statistics.median(turns) if turns else None, "p90": pct(turns, 0.9),
                       "max": max(turns, default=None)},
        "machine": {"busy": total(busy), "idle": window - total(busy), "utilisation": total(busy) / window if window else 0,
                    "idle_while_waiting": max(0.0, idle_waiting), "avg_parallel": acc / total(busy) if busy else 0,
                    "peak_parallel": peak},
        "jobs": {"count": len(jl), "failed": sum(1 for j in jl if not j["ok"]), "render_total": sum(j["seconds"] for j in jl),
                 "median": statistics.median(secs) if secs else None,
                 "slowest": sorted(({"key": j["key"], "what": label(j), "seconds": j["seconds"]} for j in jl),
                                   key=lambda x: -x["seconds"])[:5],
                 "fastest": sorted(({"key": j["key"], "what": label(j), "seconds": j["seconds"]} for j in jl if j["ok"]),
                                   key=lambda x: x["seconds"])[:5],
                 "by_kind": {k: {"count": len(v), "median": statistics.median(v), "max": max(v)}
                             for k, v in sorted(kinds.items(), key=lambda kv: -sum(kv[1]))},
                 "warned": max(evc("warn"), sum(1 for j in jl if j["seconds"] > 600)), "killed": evc("killed"), "sheet_preempted": evc("stopped")},
        "threads": by_thread,
        "blocked_now": blocked_now,
        "batching": batching(lo, events, json.load(open(STATE)) if os.path.exists(STATE) else {}),
        "telemetry": telemetry(lo, jl, json.load(open(STATE)) if os.path.exists(STATE) else {}),
        "hogging": hogging(lo, now, events),
    }


def hogging(lo, now, events):
    """John, 2026-11-19: which threads hogged the queue. The warnings rq.py gives (hog_warnings), replayed every
    15 minutes over the window, with each thread's share of the render time and the warnings the threads were told."""
    try:
        sys.path.insert(0, HERE)
        import rq
    except Exception as e:       # (the stats still run without it)
        return {"error": repr(e)}
    s = json.load(open(STATE))
    seen = {}
    t = lo
    while t <= now:
        for w in rq.hog_warnings(s, t):
            k = (w["name"], w["rule"])
            x = seen.setdefault(k, {"thread": w["name"], "rule": w["rule"], "first": t, "last": t, "peak": w["value"]})
            x["last"] = t; x["peak"] = max(x["peak"], w["value"])
        t += 900
    secs = {}
    for r in s["requests"]:
        if r.get("created") and lo <= ts(r["created"]) <= now:
            n = rq.THREAD_NAMES.get(r["thread"], r["thread"][-6:])
            secs[n] = secs.get(n, 0) + rq.request_seconds(r)
    tot = sum(secs.values()) or 1
    told = {}
    for e in events:
        if e["ev"] == "hog" and ts(e["t"]) >= lo:
            n = rq.THREAD_NAMES.get(e.get("thread", ""), e.get("thread", "")[-6:])
            told[n] = told.get(n, 0) + 1
    pings, unpinged = {}, {}
    for e in events:
        if e["ev"] in ("ping", "unpinged") and ts(e["t"]) >= lo:
            n = rq.THREAD_NAMES.get(e.get("thread", ""), e.get("thread", "")[-6:])
            d = pings if e["ev"] == "ping" else unpinged
            d[n] = d.get(n, 0) + 1
    pen = {}      # the priority penalty: requests that ran behind others, and the most steps, per thread
    for r in s["requests"]:
        if r.get("created") and lo <= ts(r["created"]) <= now and r.get("views"):
            rep = r["repeat"] if "repeat" in r else rq.repeat_count(          # (older requests: counted afterwards)
                {"requests": [o for o in s["requests"] if o["id"] < r["id"]]}, r["thread"], r["views"], r.get("quality"), ts(r["created"]))
            if not rep:
                continue
            n = rq.THREAD_NAMES.get(r["thread"], r["thread"][-6:])
            d = pen.setdefault(n, {"requests": 0, "steps": 0, "max_repeat": 0})
            d["requests"] += 1; d["steps"] += min(rep, rq.PEN["max"]); d["max_repeat"] = max(d["max_repeat"], rep)
    return {"rules": rq.HOG, "penalty_rules": rq.PEN, "penalised": pen, "pings": pings, "unpinged": unpinged,
            "share": {k: round(v / tot, 2) for k, v in sorted(secs.items(), key=lambda kv: -kv[1])},
            "seconds": {k: round(v) for k, v in secs.items()},
            "warnings": [dict(v, first=dt.datetime.fromtimestamp(v["first"], dt.timezone.utc).strftime("%H:%MZ"),
                              last=dt.datetime.fromtimestamp(v["last"], dt.timezone.utc).strftime("%H:%MZ"))
                         for v in seen.values()],
            "told": told}


def build_seconds(key):
    """How long a job's build took (views.py prints 'build N s'), from its render.log."""
    import re
    for d in (os.path.join(RESULTS, key), os.path.join(RESULTS, key + "-failed")):
        try:
            m = re.search(r"^build (\d+) s", open(os.path.join(d, "render.log")).read(), re.M)
        except OSError:
            continue
        if m:
            return int(m[1])
    return None


def _rejections(lo):
    try:
        rows = [json.loads(x) for x in open(os.path.join(DATA, "rejections.jsonl")) if x.strip()]
    except OSError:
        return []
    return [r for r in rows if ts(r.get("at")) and ts(r["at"]) >= lo]


def batching(lo, events, state):
    """John's batching numbers (2026-11-19): what was combined, what it saved, who was left out, and how long
    the collation window held work; with tuning proposals for the window, the straggler cutoff and the caps."""
    import re
    src = open(os.path.join(HERE, "rq.py")).read()
    knob = lambda n, d: int(re.search(rf'{n} = int\(os.environ.get\("[A-Z_]+", "(\d+)"\)\)', src)[1]) if re.search(  # noqa: E731
        rf'{n} = int\(os.environ.get\("[A-Z_]+", "(\d+)"\)\)', src) else d
    window, cutoff = knob("WINDOW_S", 45), knob("READY_CUTOFF_S", 300)
    cap_batch, cap_merge = knob("MAX_BATCH", 4), knob("MAX_MERGE", 6)
    ev = [e for e in events if ts(e["t"]) >= lo]
    bt = [e for e in ev if e["ev"] == "batch"]
    done = {e["key"]: e for e in ev if e["ev"] in ("done", "failed", "killed")}
    combined = [e for e in bt if e.get("parts", 1) > 1 or e.get("set")]
    sets_ = [e for e in bt if e.get("set")]
    saved = 0
    for e in combined:
        b = build_seconds(e["key"])
        if b:
            saved += (e["parts"] - 1) * b      # one build instead of `parts`
    held = [e for e in bt if e.get("held_for")]
    held_window = [e["waited"] for e in held if "window" in (e["held_for"] or "")]
    held_intent = [e["waited"] for e in held if "intent" in (e["held_for"] or "")]
    gained = [e for e in held if e.get("parts", 1) > 1 or e.get("branches", 1) > 1 or len(e.get("requests") or []) > 1]
    strag = [i for e in bt for i in (e.get("stragglers") or [])]
    intents = {i["id"]: i for i in state.get("intents", [])}
    reqs = {r["id"]: r for r in state.get("requests", [])}
    late = []      # how long after the batch started a straggler's request came
    for e in bt:
        for i in e.get("stragglers") or []:
            it = intents.get(i)
            m = re.match(r"sent as #(\d+)", (it or {}).get("state", ""))
            if m and int(m[1]) in reqs:
                late.append(ts(reqs[int(m[1])]["created"]) - ts(e["t"]))
    fails = [e for e in combined if done.get(e["key"], {}).get("ev") in ("failed", "killed")]
    at_cap = [e for e in bt if e.get("branches", 1) >= cap_batch or len(e.get("requests") or []) >= cap_merge]
    out = {"jobs": len(bt), "combined": len(combined), "set_renders": len(sets_),
           "pieces_per_set": [len(e.get("pieces") or []) for e in sets_],
           "requests_per_job": statistics.mean([len(e.get("requests") or []) for e in bt]) if bt else None,
           "time_saved": saved, "duplicates_dropped": sum(e.get("dups", 0) for e in bt),
           "coalesced": sum(1 for e in ev if e["ev"] == "coalesced"),
           "retried_alone": sum(1 for e in ev if e["ev"] == "retry_alone"),
           "held": len(held), "held_gained_partner": len(gained),
           "wait_in_window": {"median": statistics.median(held_window) if held_window else None,
                              "max": max(held_window, default=None)},
           "wait_for_intent": {"median": statistics.median(held_intent) if held_intent else None,
                               "max": max(held_intent, default=None)},
           "stragglers": len(strag), "straggler_late": {"median": statistics.median(late) if late else None,
                                                        "max": max(late, default=None)},
           "combined_failed": len(fails), "at_cap": len(at_cap),
           "knobs": {"WINDOW_S": window, "READY_CUTOFF_S": cutoff, "MAX_BATCH": cap_batch, "MAX_MERGE": cap_merge}}
    tune = []
    if len(held) >= 5:
        rate = len(gained) / len(held)
        if rate < 0.1 and window > 15:
            tune.append(f"WINDOW_S {window} -> {max(15, window // 2)}: only {len(gained)} of {len(held)} held jobs "
                        f"gained a partner in the window; the wait costs every lone job")
        elif rate > 0.4 and window < 120:
            tune.append(f"WINDOW_S {window} -> {min(120, window * 2)}: {len(gained)} of {len(held)} held jobs gained "
                        f"a partner; a longer window may catch more")
    if len(late) >= 3:
        p75 = sorted(late)[int(0.75 * (len(late) - 1))]
        if p75 < 600 and p75 > 0:
            tune.append(f"READY_CUTOFF_S {cutoff} -> {min(900, cutoff + int(p75) // 60 * 60 + 60)}: three quarters of "
                        f"the stragglers came within {fmt(p75)} of their batch starting, and each cost a second build")
    if len(held_intent) >= 5 and not strag and max(held_intent) < cutoff * 0.6:
        tune.append(f"READY_CUTOFF_S {cutoff} -> {max(120, int(max(held_intent)) // 60 * 60 + 60)}: every announced "
                    f"piece came within {fmt(max(held_intent))}; a shorter cutoff loses nothing")
    if combined and len(fails) / len(combined) > 0.2 and cap_merge > 2:
        tune.append(f"MAX_MERGE {cap_merge} -> {cap_merge - 1}: {len(fails)} of {len(combined)} combined jobs failed "
                    f"and were retried alone")
    elif len(at_cap) >= 3 and not fails:
        tune.append(f"MAX_MERGE {cap_merge} -> {cap_merge + 2}, MAX_BATCH {cap_batch} -> {cap_batch + 1}: {len(at_cap)} "
                    f"jobs hit the cap and no combined job failed")
    crop_jobs = [e for e in bt if e.get("crops")]
    pose_jobs = [e for e in bt if e.get("poses")]
    secs = {e["key"]: e.get("seconds") for e in ev if e["ev"] == "done"}
    extra_secs = [secs[e["key"]] for e in bt if e.get("only_extras") and secs.get(e["key"])]
    out["extras"] = {"crop_jobs": len(crop_jobs), "crops": sum(e["crops"] for e in crop_jobs),
                     "pose_strips": len(pose_jobs),
                     "extras_only_median": statistics.median(extra_secs) if extra_secs else None,
                     "rejected_after_pose_strip": sum(
                         1 for r in _rejections(lo) if set(r.get("jobs") or []) & {e["key"] for e in pose_jobs})}
    out["tuning"] = tune or ["no change: too few batches, or the numbers sit inside the bands"]
    return out


def telemetry(lo, jl, state):
    """John's telemetry (2026-11-19), from the manifests the queue writes: phase times, failures by cause, the
    cache-hit rate, drafts against finals, what each thread costs, the garments' geometry checks and the body's
    ratios commit to commit, and the rejections by tag, garment, who and rounds."""
    import re
    ms = []
    for j in jl:
        try:
            ms.append(json.load(open(os.path.join(j["dir"], "manifest.json"))))
        except (OSError, ValueError, KeyError):
            continue
    ms = [m for m in ms if m.get("phases") is not None or m.get("cause")]
    ph = {}
    for m in ms:
        for k, v in (m.get("phases") or {}).items():
            if isinstance(v, (int, float)):
                ph.setdefault(k, []).append(v)
        if m.get("queue_wait") is not None:
            ph.setdefault("queue_wait", []).append(m["queue_wait"])
    phases = {k: {"median": statistics.median(v), "p90": pct(v, 0.9), "total": sum(v), "n": len(v)} for k, v in ph.items()}
    causes = {}
    for m in ms:
        if m.get("cause"):
            causes.setdefault(m["cause"], []).append(m["key"])
    reqs = [r for r in state.get("requests", []) if ts(r.get("created")) and ts(r["created"]) >= lo]
    planned = [r for r in reqs if r.get("jobs")]
    hits = sum(r.get("cache_hits", 0) for r in planned)
    total_jobs = sum(len(r["jobs"]) for r in planned)
    quality = {}
    for r in reqs:
        quality[r.get("quality") or "final"] = quality.get(r.get("quality") or "final", 0) + 1
    # what each thread costs: requests, rounds on one piece, render time, check and final
    cost = {}
    secs = {j["key"]: j["seconds"] for j in jl}
    for r in reqs:
        c = cost.setdefault(r["thread"], {"requests": 0, "render": 0, "check": 0, "final": 0, "pieces": {}})
        c["requests"] += 1
        c[r.get("quality") or "final"] = c.get(r.get("quality") or "final", 0) + 1
        for k in r.get("jobs") or []:
            n = sum(1 for x in reqs if k in (x.get("jobs") or []))
            c["render"] += secs.get(k, 0) / max(1, n)      # (a shared job's time is split among its requests)
        for v in r.get("views") or []:
            for f in re.findall(r"--figure (\S+)", v):
                c["pieces"][f] = c["pieces"].get(f, 0) + 1
    # geometry: the latest numbers per garment, flagged past the bands; the body's size per commit
    geo, body = {}, []
    for m in sorted(ms, key=lambda m: m.get("finished", "")):
        g = m.get("geometry") or {}
        for name, v in (g.get("garments") or {}).items():
            geo[name] = dict(v, key=m["key"])
        if g.get("body"):
            body.append((m["key"], m.get("commit", "")[:9], g["body"]))
    flags = [f"{n}: {v['inside_pct']}% of its vertices inside the body (deepest {v.get('max_depth')})"
             for n, v in geo.items() if (v.get("inside_pct") or 0) > 2.0] + \
            [f"{n}: edges stretched to {v['stretch_p99']}x (99th percentile)" for n, v in geo.items()
             if (v.get("stretch_p99") or 0) > 1.3] + \
            [f"{n}: edges squashed to {v['squash_p1']}x (1st percentile)" for n, v in geo.items()
             if v.get("squash_p1") is not None and v["squash_p1"] < 0.7]
    drift = []
    for k in ("height", "shoulder_width", "hip_width", "hip_depth"):
        vals = [b[k] for _, _, b in body if b.get(k)]
        if len(vals) >= 3:
            med = statistics.median(vals)
            for key, c, b in body:
                if b.get(k) and abs(b[k] - med) / med > 0.01:
                    drift.append(f"{k} {b[k]} at {c} ({key}): {100 * (b[k] - med) / med:+.1f}% from the median {med}")
    rej = _rejections(lo)
    by = lambda f: dict(sorted(((k, sum(1 for r in rej if f(r) == k)) for k in {f(r) for r in rej}), key=lambda kv: -kv[1]))  # noqa: E731
    piece = lambda r: (re.findall(r"--figure (\S+)", " ".join(r.get("views") or []) if isinstance(r.get("views"), list) else str(r.get("views") or "")) or [r.get("branch") or "-"])[0]  # noqa: E731
    rounds = {}
    for r in reqs:
        for v in r.get("views") or []:
            for f in re.findall(r"--figure (\S+)", v):
                rounds.setdefault(f, set()).add(r["id"])
    return {"phases": phases, "failures": {k: len(v) for k, v in causes.items()},
            "cache_hit_rate": (hits / total_jobs) if total_jobs else None, "cache_hits": hits, "jobs_planned": total_jobs,
            "quality": quality, "thread_cost": cost, "geometry": geo, "geometry_flags": flags, "body_drift": drift,
            "rejections": {"by_tag": by(lambda r: (r.get("tags") or ["-"])[0]), "by_garment": by(piece),
                           "by_who": by(lambda r: r.get("by") or "-"),
                           "rounds_per_piece": {f: len(v) for f, v in sorted(rounds.items(), key=lambda kv: -len(kv[1]))},
                           "rejected_per_piece": by(piece)}}


def report(st):
    w, m, j = st["wait"], st["machine"], st["jobs"]
    out = [f"Render queue, last {st['window']['hours']} h (since {st['window']['from']})", "",
           f"Requests: {st['requests']['total']} ({st['requests']['done']} done, {st['requests']['failed']} failed, "
           f"{st['requests']['from_cache']} from the cache), {st['requests']['per_day']} a day",
           f"Wait before the first job starts: median {fmt(w['median'])}, 90% under {fmt(w['p90'])}, longest {fmt(w['max'])}",
           f"Turnaround (queued to pictures): median {fmt(st['turnaround']['median'])}, "
           f"90% under {fmt(st['turnaround']['p90'])}, longest {fmt(st['turnaround']['max'])}",
           f"Machine: busy {fmt(m['busy'])} ({m['utilisation']:.0%}), idle {fmt(m['idle'])}, "
           f"idle while a request waited {fmt(m['idle_while_waiting'])}",
           f"Parallel jobs while busy: {m['avg_parallel']:.2f} on average, {m['peak_parallel']} at most",
           f"Jobs: {j['count']} ({j['failed']} failed), {fmt(j['render_total'])} of rendering, median {fmt(j['median'])}; "
           f"{j['warned']} over 10 min, {j['killed']} killed at 30 min, {j['sheet_preempted']} sheet runs preempted",
           "", "Slowest jobs:"]
    out += [f"  {fmt(x['seconds']):>8}  {x['what']}  ({x['key']})" for x in j["slowest"]]
    out += ["Fastest jobs:"] + [f"  {fmt(x['seconds']):>8}  {x['what']}  ({x['key']})" for x in j["fastest"]]
    out += ["", "By kind of job (count, median, longest):"]
    out += [f"  {k}: {v['count']}, {fmt(v['median'])}, {fmt(v['max'])}" for k, v in list(j["by_kind"].items())[:12]]
    out += ["", "Threads (requests, total wait, longest wait, total turnaround, render time):"]
    for t, b in sorted(st["threads"].items(), key=lambda kv: -kv[1]["wait"]):
        out.append(f"  {t}: {b['requests']}, {fmt(b['wait'])}, {fmt(b['max_wait'])}, {fmt(b['turnaround'])}, {fmt(b['render'])}")
    b = st.get("batching")
    if b:
        out += ["", "Batching (sets, merged views, the collation window):",
                f"  {b['combined']} of {b['jobs']} jobs combined work ({b['set_renders']} set renders, pieces "
                f"{b['pieces_per_set'] or '-'}), {b['requests_per_job'] or 0:.2f} requests a job; about "
                f"{fmt(b['time_saved'])} of builds saved",
                f"  {b['duplicates_dropped']} duplicate jobs dropped, {b['coalesced']} stale requests coalesced, "
                f"{b['retried_alone']} retried alone after a combined job failed ({b['combined_failed']} failed)",
                f"  held: {b['held']} jobs, {b['held_gained_partner']} gained a partner; wait in the window median "
                f"{fmt(b['wait_in_window']['median'])}, for an intent median {fmt(b['wait_for_intent']['median'])} "
                f"(longest {fmt(b['wait_for_intent']['max'])})",
                f"  stragglers left out: {b['stragglers']}, arriving a median {fmt(b['straggler_late']['median'])} "
                f"after their batch started; {b['at_cap']} jobs at the size cap",
                f"  knobs: {b['knobs']}"]
        x = b.get("extras") or {}
        out.append(f"  crops: {x.get('crops', 0)} in {x.get('crop_jobs', 0)} jobs; pose strips: {x.get('pose_strips', 0)}; "
                   f"crop/pose-only jobs median {fmt(x.get('extras_only_median'))}; rejections that came from a pose "
                   f"strip: {x.get('rejected_after_pose_strip', 0)}")
        out += ["  tuning: " + t for t in b["tuning"]]
    t = st.get("telemetry")
    if t:
        out += ["", "Telemetry (per job, from the manifests):"]
        out.append("  phases (median / 90% / total): " + "; ".join(
            f"{k} {fmt(v['median'])} / {fmt(v['p90'])} / {fmt(v['total'])}" for k, v in t["phases"].items()) if t["phases"]
            else "  phases: none yet (recorded from 2026-11-19)")
        out.append(f"  cache hits: {t['cache_hits']} of {t['jobs_planned']} jobs planned"
                   + (f" ({t['cache_hit_rate']:.0%})" if t["cache_hit_rate"] is not None else "")
                   + f"; requests by quality: {t['quality']}")
        out.append(f"  failures by cause: {t['failures'] or 'none'}")
        out.append("  thread cost (requests, check/final, render time): " + "; ".join(
            f"{k[-6:]} {v['requests']}, {v.get('check', 0)}/{v.get('final', 0)}, {fmt(v['render'])}"
            for k, v in sorted(t["thread_cost"].items(), key=lambda kv: -kv[1]["render"])[:8]))
        out += ["  geometry: " + f for f in t["geometry_flags"]] or ["  geometry: every garment inside the bands"]
        out += ["  body drift: " + d for d in t["body_drift"][:6]] or ["  body drift: none past 1%"]
        r = t["rejections"]
        out.append(f"  rejections by tag {r['by_tag']}, by garment {r['by_garment']}, by who {r['by_who']}")
        out.append(f"  rounds (requests) per piece: {dict(list(r['rounds_per_piece'].items())[:8])}")
    h = st.get("hogging")
    if h and "error" not in h:
        out += ["", "Hogging (one thread's share of the render time; warnings replayed every 15 min):",
                "  share: " + "; ".join(f"{k} {v:.0%} ({fmt(h['seconds'][k])})" for k, v in list(h["share"].items())[:8])]
        out += [f"  {w['thread']}: {w['rule']} from {w['first']} to {w['last']}, peak {w['peak']}" for w in h["warnings"]] \
            or ["  no thread passed a warning"]
        out.append(f"  warnings told to threads at request time: {h['told'] or 'none'}; rules {h['rules']}")
        out.append("  repeat penalty (requests that ran behind others, total steps, longest repeat run): "
                   + ("; ".join(f"{k}: {v['requests']}, {v['steps']}, {v['max_repeat']}" for k, v in h["penalised"].items()) or "none")
                   + f"; rules {h['penalty_rules']}")
        out.append(f"  pings by thread: {h['pings'] or 'none'}; requests sent without a recent ping: {h['unpinged'] or 'none'}")
    out += ["", f"Blocked right now: {len(st['blocked_now'])}"]
    out += [f"  #{r['id']} {r['thread']} {r['state']}, waiting {fmt(r['wait'])}" for r in st["blocked_now"]]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float)
    ap.add_argument("--json")
    a = ap.parse_args()
    try:
        import ast
        src = open(os.path.join(HERE, "rq.py")).read()
        names = ast.literal_eval("{" + src.split("THREAD_NAMES = {", 1)[1].split("\n}\n", 1)[0] + "\n}")
    except (OSError, ValueError, SyntaxError, IndexError):
        names = {}
    st = compute(a.hours, names)
    print(report(st))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(st, f, indent=1)
