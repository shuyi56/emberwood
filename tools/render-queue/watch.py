import json, sys, time, hashlib, subprocess, os
sys.path.insert(0, "/mnt/project-files/render-queue")
import rq
last = None
since = None        # when a pending request was first seen with no serve running
cool = 0
def serving():
    return subprocess.run(["pgrep", "-f", "rq.py serve"], capture_output=True).returncode == 0
while True:
    try:
        s = rq.load()
        sig = [[r["id"], r["state"]] for r in s["requests"]] + sorted(w["text"] for w in rq.hog_warnings(s)) \
              + [[i["id"], i["state"]] for i in s.get("intents", [])]
        h = hashlib.md5(json.dumps(sig).encode()).hexdigest()
        if last is not None and h != last:
            ch = [f"#{r['id']} {r['state'].split()[0]}" for r in s["requests"] if r["state"] in ("rendering", "done", "failed", "pending", "conflict") and r["id"] >= s["next"] - 6]
            print("queue changed: " + ", ".join(ch[-4:]), flush=True)
        last = h
        # 2026-11-22: start serve when a request waits and nothing is serving (an 18 min wait on 2026-11-21 came from serve being down)
        waiting = any(r["state"].split()[0] == "pending" for r in s["requests"])
        if waiting and not serving():
            since = since or time.time()
            if time.time() - since > 15 and time.time() > cool:
                out = open(f"/tmp/rq-work/serve-n{int(time.time())}.log", "w")
                subprocess.Popen(["python3", "rq.py", "serve"], cwd="/mnt/project-files/render-queue", stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
                cool = time.time() + 60
                print("watch: started serve for a waiting request", flush=True)
        else:
            since = None
    except Exception as e:
        pass
    time.sleep(10)
