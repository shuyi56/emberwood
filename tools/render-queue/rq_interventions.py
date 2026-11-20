"""How often John steps into a thread, and how much progress a thread makes before he does.

    python3 rq_interventions.py [--hours N] [--json OUT]       (rq_stats.py also prints this as a section)
    python3 rq_interventions.py extract FILE... >> new.jsonl    (the nightly job: saved read_thread outputs to lines)
    python3 rq_interventions.py add-activity new.jsonl          (the nightly job: merge them into the log)

Inputs, all in the queue's folder:
  stats/thread-activity.jsonl   one line per thread message, timestamps and kinds only, never the text:
        {"id": "cmsg_...", "thread": "cmsg_<root>", "at": "2026-11-20T21:45:37Z", "who": "john"|"agent",
         "kind": "root"|"msg"|"result"|"stop"}
        plus one {"thread": "cmsg_<root>", "title": "...", "read_from": "<UTC>"} line per thread; read_from (optional)
        says messages before it were never read, so the run that started before it is left out. The nightly job writes it from
        read_thread (see the README, "John's interventions"); add-activity merges a new file in, keyed on id.
        kind for an agent message: "stop" = a stop-rule report, "result" = a progress report (a reply of 280
        characters or more: results, renders, contour sheets), "msg" = a short reply or ack (not counted as
        progress). Checklist updates, activity notes and errors are left out. `extract` applies these rules.
  rejections.jsonl              rejections; --by John counts as an intervention
  queue.json                    every render or solver request a thread sent (one progress step each)

Definitions:
  intervention  John stepping into a thread after its opening ask: a message he writes there or a rejection logged
                --by John. His messages and rejections less than 15 minutes apart in one thread are one
                intervention (a burst of quick follow-ups is one stepping-in); the raw counts are kept as well.
  iteration     one step of the thread's own progress: a request to the queue (render or solver run), a self- or
                reviewer-logged rejection, a posted result or a stop-rule report. Steps in one thread less than
                10 minutes apart are one iteration (a request, its result post and the rejection of it are one round).
  run           the iterations between two interventions (or between the opening ask and the first one). A run that
                ends with the thread still going is "open" and not in the median.
Days are Toronto days (the nightly job's clock).
"""
import argparse
import datetime as dt
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("RQ_HOME", HERE)
ACTIVITY = os.path.join(DATA, "stats", "thread-activity.jsonl")
SAME_INTERVENTION_S = 15 * 60
SAME_ROUND_S = 10 * 60
try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("America/Toronto")
except Exception:                                   # no tz database: fixed EST
    TZ = dt.timezone(dt.timedelta(hours=-5))


def ts(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp() if s else None


def day(t):
    return dt.datetime.fromtimestamp(t, TZ).strftime("%Y-%m-%d")


def _jsonl(path):
    try:
        return [json.loads(x) for x in open(path) if x.strip()]
    except OSError:
        return []


def load_activity(path=ACTIVITY, read_from=None):
    msgs, titles = {}, {}
    for r in _jsonl(path):
        if "title" in r and "at" not in r:
            titles[r["thread"]] = r["title"]
            if read_from is not None and r.get("read_from"):
                read_from[r["thread"]] = r["read_from"]
        elif r.get("id") and r.get("at"):
            msgs[r["id"]] = r
    return list(msgs.values()), titles


def add_activity(src, dst=ACTIVITY):
    """Merge FILE into the activity log, keyed on message id (titles: last one wins). Refuses text fields."""
    rf = {}
    old, titles = load_activity(dst, rf)
    have = {m["id"]: m for m in old}
    added = 0
    for r in _jsonl(src):
        if "body" in r or "text" in r:
            sys.exit("activity lines must not carry message text")
        if "title" in r and "at" not in r:
            titles[r["thread"]] = r["title"]
            if r.get("read_from"):
                rf[r["thread"]] = min(r["read_from"], rf.get(r["thread"], r["read_from"]))
        elif r.get("id") and r.get("at") and r["id"] not in have:
            have[r["id"]] = {k: r[k] for k in ("id", "thread", "at", "who", "kind") if k in r}
            added += 1
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    with open(tmp, "w") as f:
        for t, title in sorted(titles.items()):
            f.write(json.dumps({"thread": t, "title": title, **({"read_from": rf[t]} if t in rf else {})}) + "\n")
        for m in sorted(have.values(), key=lambda m: (m["at"], m["id"])):
            f.write(json.dumps(m) + "\n")
    os.replace(tmp, dst)
    return added


JOHN_ID = "user_01FcQUGjW7zmrnu3zzSoTrE4"
STOP_RE = re.compile(r"stop(ped|ping)? (iterating|here|the iteration|and report)|stop rule|stopping point|"
                     r"two rounds in a row|third round", re.I)
RESULT_CHARS = 280


def extract(paths):
    """read_thread results (the JSON object, or a saved tool-result file holding it) -> activity lines, no text."""
    out = {}
    for p in paths:
        s = open(p).read()
        d = json.loads(s[s.index('{"thread_id"'):s.rindex('}') + 1])
        for m in d["messages"]:
            t, b = m["thread_id"], (m.get("body") or "")
            if m["id"] == t:
                who, k = "john", "root"
            elif m.get("author") == "user":
                if m.get("author_id") != JOHN_ID:
                    continue
                who, k = "john", "msg"
            elif b.startswith('{"') and '"kind"' in b[:200]:
                continue                                    # checklist, activity note, error
            else:
                who = "agent"
                k = "stop" if STOP_RE.search(b) else "result" if len(b) >= RESULT_CHARS else "msg"
            out[m["id"]] = {"id": m["id"], "thread": t, "at": m["created_at"], "who": who, "kind": k}
    return sorted(out.values(), key=lambda m: m["at"])


def _branch_threads(state):
    bt = {}
    for r in state.get("requests", []):
        if r.get("branch") and r.get("thread"):
            bt.setdefault(r["branch"], r["thread"])
    return bt


def compute(hours=None, names=None, now=None):
    names = dict(names or {})
    now = now or dt.datetime.now(dt.timezone.utc).timestamp()
    lo = now - hours * 3600 if hours else 0
    rf = {}
    msgs, titles = load_activity(read_from=rf)
    names.update({t: v for t, v in titles.items() if t not in names})
    try:
        state = json.load(open(os.path.join(DATA, "queue.json")))
    except (OSError, ValueError):
        state = {}
    b2t = _branch_threads(state)
    rej = _jsonl(os.path.join(DATA, "rejections.jsonl"))
    rej_thread = lambda r: r.get("thread") or b2t.get(r.get("branch"))   # noqa: E731

    root, john, steps = {}, {}, {}
    for m in msgs:
        t, at = m["thread"], ts(m["at"])
        if m.get("kind") == "root" or m["id"] == t:
            root[t] = at
        elif m.get("who") == "john":
            john.setdefault(t, []).append((at, "msg"))
        elif m.get("kind") in ("result", "stop"):
            steps.setdefault(t, []).append((at, m["kind"]))
    for r in rej:
        t = rej_thread(r)
        if not t or not r.get("at"):
            continue
        if r.get("by") == "John":
            john.setdefault(t, []).append((ts(r["at"]), "rejection"))
        else:
            steps.setdefault(t, []).append((ts(r["at"]), "rejection"))
    for r in state.get("requests", []):
        if r.get("thread") and r.get("created") and r.get("state") != "cancelled":
            steps.setdefault(r["thread"], []).append((ts(r["created"]), "request"))

    covered = set(root) | {m["thread"] for m in msgs}     # threads whose messages were read: the only ones measured
    per_thread, per_day = {}, {}
    all_runs, all_gaps = [], []
    for t in sorted(covered):
        iv = []                                             # merge a John rejection into a nearby John message
        for at, k in sorted(john.get(t, [])):
            if iv and at - iv[-1][0] <= SAME_INTERVENTION_S:
                iv[-1][1].add(k)
                continue
            iv.append((at, {k}))
        rounds = []
        for at, k in sorted(steps.get(t, [])):
            if rounds and at - rounds[-1] < SAME_ROUND_S:
                continue
            rounds.append(at)
        cut = ts(rf[t]) if t in rf else None
        start = root.get(t) if t in root and (cut is None or root[t] >= cut) else cut
        if start is None:
            start = min([a for a, _ in iv] + rounds + [now])
        iv = [(a, k) for a, k in iv if a >= start]
        rounds = [r for r in rounds if r >= start]
        marks = [start] + [a for a, _ in iv]
        runs = []                                           # (ends_at, iterations), closed by an intervention
        for i, (a, b) in enumerate(zip(marks, marks[1:])):
            if i == 0 and start == cut:
                continue                                    # began before the messages we read
            runs.append((b, sum(1 for r in rounds if a <= r < b), b - a))
        open_run = sum(1 for r in rounds if r >= marks[-1])
        iv_w = [(a, k) for a, k in iv if lo <= a <= now]
        runs_w = [n for b, n, _ in runs if lo <= b <= now]
        gaps = [g for b, _, g in runs if lo <= b <= now]
        if not iv_w and not runs_w and not any(lo <= r <= now for r in rounds):
            continue
        name = names.get(t, t[-6:])
        raw = [k for a, k in john.get(t, []) if lo <= a <= now and a >= start]
        per_thread[name] = {
            "interventions": len(iv_w), "john_messages": raw.count("msg"), "john_rejections": raw.count("rejection"), "messages": sum(1 for _, k in iv_w if "msg" in k),
            "rejections": sum(1 for _, k in iv_w if "rejection" in k),
            "avg_gap_s": statistics.mean(gaps) if gaps else None,
            "iterations": sum(1 for r in rounds if lo <= r <= now),
            "median_run": statistics.median(runs_w) if runs_w else None, "longest_run": max(runs_w, default=None),
            "open_run": open_run}
        all_runs += [(n, name) for n in runs_w]
        all_gaps += gaps
        for a, k in iv_w:
            d = per_day.setdefault(day(a), {"interventions": 0, "threads": set(), "runs": [], "gaps": []})
            d["interventions"] += 1
            d["threads"].add(name)
        for b, n, g in runs:
            if lo <= b <= now:
                per_day[day(b)]["runs"].append(n)
                per_day[day(b)]["gaps"].append(g)
    longest = max(all_runs, default=(None, None))
    days = {d: {"interventions": v["interventions"], "threads": len(v["threads"]),
                "avg_gap_s": statistics.mean(v["gaps"]) if v["gaps"] else None,
                "median_run": statistics.median(v["runs"]) if v["runs"] else None, "longest_run": max(v["runs"], default=None)}
            for d, v in sorted(per_day.items())}
    return {"covered_threads": len(covered), "activity_lines": len(msgs),
            "interventions": sum(v["interventions"] for v in per_thread.values()),
            "john_messages": sum(v["john_messages"] for v in per_thread.values()),
            "rejections_by_john": sum(v["john_rejections"] for v in per_thread.values()),
            "threads_touched": sum(1 for v in per_thread.values() if v["interventions"]),
            "avg_gap_s": statistics.mean(all_gaps) if all_gaps else None,
            "median_run": statistics.median([n for n, _ in all_runs]) if all_runs else None,
            "longest_run": longest[0], "longest_run_thread": longest[1],
            "untouched_runs": sum(1 for n, _ in all_runs if n == 0),
            "days": days,
            "threads": dict(sorted(per_thread.items(), key=lambda kv: (-kv[1]["interventions"], kv[0])))}


def _fmt(sec):
    if sec is None:
        return "-"
    return f"{sec / 60:.0f} min" if sec < 5400 else f"{sec / 3600:.1f} h"


def report_lines(x, top=8):
    """A short block for rq_stats.py's report."""
    if not x or not x.get("activity_lines"):
        return ["John's interventions: no thread activity logged yet (stats/thread-activity.jsonl)"]
    n = lambda v: "-" if v is None else (f"{v:g}" if isinstance(v, (int, float)) else str(v))   # noqa: E731
    out = ["John's interventions (his messages and rejections after a thread's opening ask, 15 min bursts as one; "
           "iterations = queue requests, progress reports, stop reports, own rejections, 10 min apart):",
           f"  {x['interventions']} interventions ({x['john_messages']} messages, {x['rejections_by_john']} rejections) in "
           f"{x['threads_touched']} threads, "
           f"average gap {_fmt(x['avg_gap_s'])}; iterations before he stepped in: median {n(x['median_run'])}, longest "
           f"{n(x['longest_run'])} ({x['longest_run_thread'] or '-'}), {x['untouched_runs']} times straight after the last"]
    if len(x["days"]) > 1:
        out.append("  by day (interventions, threads, avg gap, median/longest run): " + "; ".join(
            f"{d[5:]} {v['interventions']}, {v['threads']}, {_fmt(v['avg_gap_s'])}, {n(v['median_run'])}/{n(v['longest_run'])}"
            for d, v in x["days"].items()))
    out.append("  by thread (interventions, avg gap, median/longest run, open run now): " + "; ".join(
        f"{k} {v['interventions']}, {_fmt(v['avg_gap_s'])}, {n(v['median_run'])}/{n(v['longest_run'])}, {v['open_run']}"
        for k, v in list(x["threads"].items())[:top] if v["interventions"]))
    return out


def _names():
    try:
        import ast
        src = open(os.path.join(HERE, "rq.py")).read()
        return ast.literal_eval("{" + src.split("THREAD_NAMES = {", 1)[1].split("\n}\n", 1)[0] + "\n}")
    except (OSError, ValueError, SyntaxError, IndexError):
        return {}


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "extract":
        for m in extract(sys.argv[2:]):
            print(json.dumps(m))
        sys.exit(0)
    if len(sys.argv) > 2 and sys.argv[1] == "add-activity":
        print(f"added {add_activity(sys.argv[2])} messages to {ACTIVITY}")
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float)
    ap.add_argument("--json")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    x = compute(a.hours, _names())
    print("\n".join(report_lines(x, a.top)))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(x, f, indent=1)
