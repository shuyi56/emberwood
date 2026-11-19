"""Log a rejected render, with the reason, so the nightly job can turn repeated faults into rules and flags.

    python3 /mnt/project-files/render-queue/rq_reject.py add --request 177 --tags clipping,seam \\
        --reason "the bodice's side seam shows through the sleeve cap in the 3/4 and profile" \\
        --fix "pushed the sleeve cap 2 mm out over the seam" [--by "John"|"self"|"reviewer"] [--image FILE]

    rq_reject.py list [--since 2026-11-19]
    rq_reject.py summary [--since DATE] [--mark]     (the nightly job: what is new since the last summary)
    rq_reject.py tags                                 (the tags used so far, most common first)

A rejection is one line of JSON appended to rejections.jsonl beside the queue's state (RQ_HOME, or this folder). The
request number fills in the branch, commit, views and result folder from queue.json, so a thread gives only what it
saw. Use the tags below where one fits, and a new short tag only when none does: the tags are what the rules are
built from. --artifacts takes the post-render flag ids of rq_flags.py (black_frame, figure_cropped, fireflies,
black_patches, same_as_main and the rest) when the fault is one an image check could catch, and --file and --region
say which image and where, so the nightly calibration can re-measure it.

    clipping      one layer comes through another (cloth through body, body through cloth, cloth through hair)
    gap           a layer stands off what it should lie on, or a seam opens
    crease        a hard fold, ridge, ledge or step that should be smooth
    spike         a vertex or few shoot out: spikes, tears, torn faces
    stretch       texture or mesh stretched, pinched or collapsed at a joint
    seam          a visible seam, edge line or UV join that should not show
    silhouette    the outline reads wrong: lumps, square shoulders, a notch
    proportion    a length, width or scale is off
    pose          the pose or the animation is wrong (a joint bent the wrong way, a foot sliding)
    shading       light, shadow, normals or smoothing look wrong (faceting, black patches, plastic skin)
    colour        a colour or material is off, or two layers read as one
    noise         fireflies, grain or denoiser smear
    framing       the camera, crop or size hides what was asked for
    blank         an empty, black or unchanged image
    other         none of these (say what in --reason)
"""
import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("RQ_HOME", HERE)
LOG = os.path.join(DATA, "rejections.jsonl")
STATE = os.path.join(DATA, "queue.json")
CURSOR = os.path.join(DATA, "stats", "rejections.cursor")
TAGS = ("clipping", "gap", "crease", "spike", "stretch", "seam", "silhouette", "proportion", "pose", "shading",
        "colour", "noise", "framing", "blank", "other")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read():
    try:
        with open(LOG) as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def request(rid):
    try:
        with open(STATE) as f:
            for r in json.load(f)["requests"]:
                if r["id"] == rid:
                    return r
    except (OSError, ValueError, KeyError):
        pass
    return None


def add(a):
    tags = [t.strip().lower() for t in a.tags.split(",") if t.strip()]
    if not tags:
        sys.exit("give at least one --tags value (see rq_reject.py --help)")
    e = {"at": now(), "request": a.request, "tags": tags, "reason": a.reason.strip(), "fix": (a.fix or "").strip(),
         "by": a.by or "self"}
    r = request(a.request) if a.request is not None else None
    if a.request is not None and r is None:
        sys.exit(f"no request #{a.request} in {STATE}")
    if r:
        e.update(thread=r.get("thread"), branch=r.get("branch"), sha=r.get("sha") or r.get("commit"),
                 views=r.get("views") or r.get("view"), quality=r.get("quality"),
                 jobs=r.get("jobs") or [])
    if a.artifacts:
        e["artifacts"] = [x.strip() for x in a.artifacts.split(",") if x.strip()]
    if a.region:
        e["region"] = [int(v) for v in a.region.split(",")]
    if a.file:
        e["file"] = a.file
        k = next((part for part in a.file.replace("\\", "/").split("/") if part.startswith(("views-", "frames-", "glb-", "sheet-"))), None)
        if k:
            e["key"] = k
    for k in ("branch", "sha", "view", "image", "key"):
        v = getattr(a, k, None)
        if v:
            e[k] = v
    new = [t for t in tags if t not in TAGS]
    line = json.dumps(e, separators=(",", ":")) + "\n"
    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)   # (one write per line: lines never mix)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)
    n = len(read())
    print(f"rejection {n} logged" + (f" (new tag: {', '.join(new)})" if new else ""))


def since_filter(rows, since):
    return [e for e in rows if not since or e["at"] >= since]


def cmd_list(a):
    for e in since_filter(read(), a.since):
        print(f'{e["at"]}  #{e.get("request", "-")}  {",".join(e["tags"]):<20} {e.get("branch", "")}  {e["reason"][:90]}')


def cmd_tags(a):
    c = collections.Counter(t for e in read() for t in e["tags"])
    for t, n in c.most_common():
        print(f"{n:4}  {t}{'' if t in TAGS else '  (new)'}")


def cmd_summary(a):
    rows = read()
    since = a.since
    if since is None:
        try:
            since = open(CURSOR).read().strip()
        except OSError:
            since = ""
    new = [e for e in rows if e["at"] > since] if since else rows
    out = {"since": since, "until": now(), "count": len(new), "total": len(rows),
           "by_tag": collections.Counter(t for e in new for t in e["tags"]).most_common(),
           "all_time_by_tag": collections.Counter(t for e in rows for t in e["tags"]).most_common(),
           "by_branch": collections.Counter(e.get("branch") or "-" for e in new).most_common(),
           "entries": new}
    print(json.dumps(out, indent=1))
    if a.mark:
        os.makedirs(os.path.dirname(CURSOR), exist_ok=True)
        with open(CURSOR, "w") as f:
            f.write(out["until"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("add", help="log one rejected render")
    q.add_argument("--request", type=int, help="the queue's request number (#177): fills in branch, commit, views")
    q.add_argument("--tags", required=True, help="comma-separated fault tags, e.g. clipping,seam")
    q.add_argument("--reason", required=True, help="what was wrong, where, and in which angle or frame")
    q.add_argument("--fix", help="what was changed, or tried, to fix it")
    q.add_argument("--by", help="who rejected it: John, self (the thread) or reviewer")
    q.add_argument("--branch"); q.add_argument("--sha"); q.add_argument("--view")
    q.add_argument("--image", help="a picture of the fault, if one was made")
    q.add_argument("--file", help="the rendered image rejected, e.g. results/views-3ce723bc5393/head.png")
    q.add_argument("--key", help="the result folder's key (views-...), if not given by --file or --request")
    q.add_argument("--artifacts", help="post-render flag ids that match the fault (rq_flags.py), e.g. black_patches")
    q.add_argument("--region", help="x0,y0,x1,y1 of the fault in --file, in pixels")
    q.set_defaults(f=add)
    q = sub.add_parser("list"); q.add_argument("--since"); q.set_defaults(f=cmd_list)
    q = sub.add_parser("tags"); q.set_defaults(f=cmd_tags)
    q = sub.add_parser("summary"); q.add_argument("--since"); q.add_argument("--mark", action="store_true")
    q.set_defaults(f=cmd_summary)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
