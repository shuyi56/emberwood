"""The collar's best flat cut (ARAP, tension.flatten) and its principal stretches:

    python3 blender/tension/collar_gauge.py OUT.blend.collar.npz GAUGE.npz      (prints a JSON line of the numbers)

join.py reads GAUGE.npz beside a build for the collar's side of the seam."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tension as TS  # noqa: E402

c = np.load(sys.argv[1]); X = c['pts']; T = c['tri'].astype(np.int64)
Xs, Ts, ids = TS.submesh(X, T, TS.areas(X, T) > 1e-7)
U = TS.flatten(Xs, Ts, iters=200)
s1, s2, _, _ = TS.principal(Xs, Ts, U); A = TS.areas(Xs, Ts)
r = dict(n=len(Xs), energy=TS.energy(Xs, Ts, U), s1_50=np.percentile(s1, 50), s1_95=np.percentile(s1, 95), s1_max=s1.max(),
         s2_5=np.percentile(s2, 5), s2_min=s2.min(), area_tension_2pct=float(A[s1 > 1.02].sum() / A.sum()),
         area_compress_2pct=float(A[s2 < 0.98].sum() / A.sum()))
print(json.dumps({k: round(float(v), 4) for k, v in r.items()}))
np.savez(sys.argv[2], X=Xs, T=Ts, U=U, s1=s1, s2=s2)
