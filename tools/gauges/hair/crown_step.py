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
B = np.concatenate([l[:14] for l, b in zip(Fd['line'], Fd['basin']) if b == 0])
O = np.concatenate([l[:40] for l, b in zip(Fd['line'], Fd['basin']) if b > 0])
O = O[O[:, 2] > 0.8]
d, ix = cKDTree(O).query(B)
near = d < 4 * PW.MM
step = so(B[near]) - so(O[ix[near]])
print(json.dumps(dict(pairs=int(near.sum()), step_mm_median=round(float(np.median(step)), 1), step_mm_p10=round(float(np.percentile(step, 10)), 1), step_mm_p90=round(float(np.percentile(step, 90)), 1))))
