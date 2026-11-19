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
                          "requests": [c["id"] for c in m.get("combines", [])]}
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
    }


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
