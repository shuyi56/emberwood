import sys, json, gzip, hashlib, numpy as np
sys.path.insert(0, 'tools/hair')
import scalp_field as SF, photo_waves as PW, wavy_flow as WF
head = sys.argv[1]; cache = sys.argv[2]
hk = hashlib.md5(open(head, 'rb').read()).hexdigest()[:12]
src = lambda m: hashlib.md5(open(m.__file__, 'rb').read()).hexdigest()[:12]
surf, F = PW._cached(cache, 'head', (hk, src(SF)), lambda: None)
body = json.load(gzip.open(head))['bodies']['mannequin']
coll, depth = WF.colliders(surf, body)
Fd = PW.field2(F, surf, depth, 16000, cache=cache, head_key=hk, log=lambda t: None)
A = np.concatenate([l[:60] for l in Fd['line']])
for name, sel in (('top y -0.1..0.1', (np.abs(A[:, 1] - 0.0) < 0.1) & (A[:, 2] > 0.85)), ('back of top y 0.15..0.35', (A[:, 1] > 0.15) & (A[:, 1] < 0.35) & (A[:, 2] > 0.8))):
    B = A[sel]; q = B - SF.CENTRE; r = np.linalg.norm(q, axis=1); so = (r - surf.radius(q / r[:, None])) / PW.MM
    xm = np.abs(B[:, 0]) / PW.MM
    print(name, [(int(a), round(float(np.percentile(so[(xm >= a) & (xm < a + 4)], 95)), 1) if ((xm >= a) & (xm < a + 4)).sum() > 5 else None) for a in range(0, 48, 4)])
R = np.array([l[0] for l in Fd['line']]); print('roots nearest the part mm', np.round(np.sort(np.abs(R[R[:, 2] > 0.9, 0]))[:5] / PW.MM, 1))
