import sys, json, gzip, hashlib, numpy as np
sys.path.insert(0, 'tools/hair')
import scalp_field as SF, photo_waves as PW, wavy_flow as WF
from scipy.spatial import cKDTree
head = sys.argv[1]; cache = sys.argv[2]
hk = hashlib.md5(open(head, 'rb').read()).hexdigest()[:12]
src = lambda m: hashlib.md5(open(m.__file__, 'rb').read()).hexdigest()[:12]
pts = SF.load_head(head)
surf, F = PW._cached(cache, 'head', (hk, src(SF)), lambda: None)
body = json.load(gzip.open(head))['bodies']['mannequin']
coll, depth = WF.colliders(surf, body)
Fd = PW.field2(F, surf, depth, 16000, cache=cache, head_key=hk, log=lambda t: None)
def so(p):
    q = p - SF.CENTRE; r = np.linalg.norm(q, axis=1); return (r - surf.radius(q / r[:, None])) / PW.MM
def step(B, O, name):
    # (matched by direction from the head's centre, not by distance: a bang 10 mm under the hair beside it is still
    # beside it. The surface beside it: the highest of the 8 nearest by direction.)
    un = lambda p: (p - SF.CENTRE) / np.linalg.norm(p - SF.CENTRE, axis=1, keepdims=True)
    d, ix = cKDTree(un(O)).query(un(B), k=8)
    near = d[:, -1] < 0.05
    st = so(B[near]) - so(O)[ix[near]].max(1)
    return {name: dict(pairs=int(near.sum()), step_mm_median=round(float(np.median(st)), 1), step_mm_p10=round(float(np.percentile(st, 10)), 1),
                       step_mm_p90=round(float(np.percentile(st, 90)), 1))}
O = np.concatenate([l[:60] for l, b in zip(Fd['line'], Fd['basin']) if b > 0])
O = O[O[:, 2] > 0.7]
rep = {}
rep.update(step(np.concatenate([l[:14] for l, b in zip(Fd['line'], Fd['basin']) if b == 0]), O, 'at the roots'))
# (the joint: the whole of each bang hair that lies over the scalp, against the hair beside and behind it)
J = np.concatenate([l[l[:, 2] > 0.80] for l, b in zip(Fd['line'], Fd['basin']) if b == 0])
rep.update(step(J, O, 'along the joint'))
print(json.dumps(rep))
