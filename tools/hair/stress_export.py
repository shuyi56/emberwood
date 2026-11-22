"""groom.npz -> the hair stress test's style json.gz (tools/render-queue/rq_hair.py): same frame (HH, x across, y back, z up),
    python3 tools/hair/stress_export.py GROOM_DIR CREATOR_STYLE.json.gz OUT.json.gz STYLE_KEY [STRIDE]
STYLE_KEY picks the body the test drapes on (ranger, wavyFull). Guides only, lock -1 per strand like the creator export."""
import sys, gzip, json, numpy as np
gdir, capsrc, out, name = sys.argv[1:5]
step = int(sys.argv[5]) if len(sys.argv) > 5 else 1
G = dict(np.load(gdir + '/groom.npz'))
cap = json.load(gzip.open(capsrc))['cap']
st = []
for i in range(0, len(G['X']), step):
    P = G['X'][i]; P = P[~np.isnan(P[:, 0])]
    if len(P) < 3: continue
    n = len(P); f = np.linspace(0, 1, n)
    w = 0.0086 * np.sqrt(np.clip(np.minimum(f * 8, 1), 0.15, 1)) * (1 - 0.5 * f)   # creator-like taper
    st.append({'p': np.round(P.ravel(), 4).tolist(), 'w': np.round(w, 4).tolist(), 't': [0.0021] * n, 'lock': -1, 'layer': 0})
json.dump({'style': name, 'name': name, 'cap': cap, 'strands': st}, gzip.open(out, 'wt'))
print(name, len(st), 'strands')
