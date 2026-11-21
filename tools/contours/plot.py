import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
src, dst = sys.argv[1], sys.argv[2]
D = np.load(src, allow_pickle=True)
X, N, hose = D['X'], D['N'], D['hose']
names = list(D['names'])
CM = 165.0 / 63.0
P = {}
for nm in names:
    s = int(D['start_' + nm]); V = D['V_' + nm]
    tri = [tuple(f) for f in D['F_' + nm] if len(set(f)) == 3]
    for q in D['Q_' + nm]:
        if len(set(q)) == 4: tri += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    P[nm] = dict(V=V, Y=X[s:s + len(V)], Nn=N[s:s + len(V)], uv=D['uv_' + nm], tri=np.array(tri))
shaft = [n for n in names if 'shaft' in n or n == 'tongue']
allY = np.concatenate([P[n]['Y'] for n in names]); allV = np.concatenate([P[n]['V'] for n in names])
side = np.sign(np.median(allY[:, 0]))
leg = hose[(hose[:, 0] * side > 0)]
z0, z1 = allY[:, 2].min(), max(P[n]['Y'][:, 2].max() for n in shaft if 'shaft' in n)
fig = plt.figure(figsize=(17, 11), dpi=110)
fig.suptitle('Boot (left): contours of the bent pieces after the solver, before any render', fontsize=14, x=0.02, ha='left')
# 1. sections
hs = np.linspace(z0 + 0.22 * (z1 - z0), z1 - 0.9, 8)
for k, z in enumerate(hs):
    ax = fig.add_axes([0.02 + 0.118 * k, 0.60, 0.112, 0.30]); ax.set_aspect('equal'); ax.axis('off')
    def ring(Q, **kw):
        q = Q[np.abs(Q[:, 2] - z) < 0.12]
        if len(q) < 8: return
        c = q[:, :2].mean(0); a = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]); o = np.argsort(a)
        ax.plot(q[o, 0], -q[o, 1], **kw)
    ring(leg, color='#b9b9b9', lw=1.0, ls='-')
    ring(allV, color='#d0762a', lw=0.9, ls='--')
    for n in shaft:
        q = P[n]['Y'][np.abs(P[n]['Y'][:, 2] - z) < 0.12]
        if len(q) < 4: continue
        c = allY[np.abs(allY[:, 2] - z) < 0.12][:, :2].mean(0)
        a = np.unwrap(np.sort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])))
        o = np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))
        # break the line at the piece's own gap
        qq = q[o]; d = np.linalg.norm(np.diff(qq[:, :2], axis=0), axis=1); cut = np.nonzero(d > 0.5)[0]
        st = 0
        for cidx in list(cut) + [len(qq) - 1]:
            ax.plot(qq[st:cidx + 1, 0], -qq[st:cidx + 1, 1], color={'tongue': '#2a7f62'}.get(n, '#1f4e8c'), lw=1.6)
            st = cidx + 1
    ax.set_title('%.0f cm up' % ((z - z0) * CM), fontsize=9)
fig.text(0.02, 0.925, 'Sections (seen from above, front at the bottom): grey the leg, orange dashes the last, blue the panels, green the tongue', fontsize=10)
# 2. profiles
def prof(ax, Q, col, axis, **kw):
    zs = np.linspace(Q[:, 2].min(), Q[:, 2].max(), 90); lo, hi, zz = [], [], []
    for a, b in zip(zs[:-1], zs[1:]):
        q = Q[(Q[:, 2] >= a) & (Q[:, 2] < b)]
        if len(q) < 3: continue
        lo.append(q[:, axis].min()); hi.append(q[:, axis].max()); zz.append(0.5 * (a + b))
    sg = -1 if axis == 1 else 1
    ax.plot(sg * np.array(lo), zz, color=col, **kw); ax.plot(sg * np.array(hi), zz, color=col, **kw)
for k, (axis, ttl) in enumerate(((0, 'Front profile'), (1, 'Side profile (front at the right)'))):
    ax = fig.add_axes([0.03 + 0.15 * k, 0.04, 0.14, 0.50]); ax.set_aspect('equal'); ax.axis('off'); ax.set_title(ttl, fontsize=10)
    prof(ax, leg[(leg[:, 2] < z1 + 1.2)], '#b9b9b9', axis, lw=1.0)
    prof(ax, allV, '#d0762a', axis, lw=0.9, ls='--')
    prof(ax, allY, '#1f4e8c', axis, lw=1.6)
# 3. bend map on the flat patterns
vmax = None; maps = []
for n in shaft:
    p = P[n]; t = p['tri']; Y = p['Y']
    fn = np.cross(Y[t[:, 1]] - Y[t[:, 0]], Y[t[:, 2]] - Y[t[:, 0]]); fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    vn = np.zeros_like(Y)
    for c in range(3): np.add.at(vn, t[:, c], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    k = np.linalg.norm(vn[e[:, 0]] - vn[e[:, 1]], axis=1) / np.maximum(np.linalg.norm(Y[e[:, 0]] - Y[e[:, 1]], axis=1), 1e-9)
    acc = np.zeros(len(Y)); cnt = np.zeros(len(Y))
    for c in range(2): np.add.at(acc, e[:, c], k); np.add.at(cnt, e[:, c], 1)
    kv = acc / np.maximum(cnt, 1)
    for _ in range(3):                                  # (smoothed over the neighbours)
        a2 = np.zeros(len(Y)); c2 = np.zeros(len(Y))
        np.add.at(a2, e[:, 0], kv[e[:, 1]]); np.add.at(a2, e[:, 1], kv[e[:, 0]]); np.add.at(c2, e[:, 0], 1); np.add.at(c2, e[:, 1], 1)
        kv = 0.5 * kv + 0.5 * a2 / np.maximum(c2, 1)
    maps.append((n, p['uv'] * CM, t, kv / CM))             # bend as 1/cm
vmax = float(np.percentile(np.concatenate([m[3] for m in maps]), 98))
lev = np.linspace(0, vmax, 9)
x = 0.36
for n, uv, t, kv in maps:
    w = 0.21 if 'shaft' in n else 0.08
    ax = fig.add_axes([x, 0.09, w, 0.43]); ax.set_aspect('equal'); ax.axis('off')
    cs = ax.tricontourf(uv[:, 0], uv[:, 1], t, np.clip(kv, 0, vmax), levels=lev, cmap='YlGnBu')
    ax.tricontour(uv[:, 0], uv[:, 1], t, np.clip(kv, 0, vmax), levels=lev[1::2], colors='k', linewidths=0.35)
    ax.text(0.5, -0.03, '%s, laid flat\nbend: half under %.2f, most under %.2f' % (n.replace('_', ' '), np.median(kv), np.percentile(kv, 95)), fontsize=9, ha='center', va='top', transform=ax.transAxes)
    x += w + 0.01
cax = fig.add_axes([0.36, 0.555, 0.3, 0.012]); cb = fig.colorbar(cs, cax=cax, orientation='horizontal'); cb.ax.tick_params(labelsize=7)
fig.text(0.67, 0.553, 'bend, 1 / radius in cm (0 flat; 0.2 is a 5 cm radius)', fontsize=9)
fig.savefig(dst)
print('wrote', dst)
