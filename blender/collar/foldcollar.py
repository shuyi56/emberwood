"""foldcollar.py BUILD OUT.npz ['{options}']: the collar as a tailor cuts it: a flat rectangle, folded on its roll line.

John, 2026-11-24: "Begin with a rectangle and deform", "Rounded corners on the rectangle as well", "draw only the collar
and get that correct", "Research how this type of collar is done with the sharp triangular bend".

Sources: a one-piece collar is drafted as a rectangle, the collar's width high (2.5 to 4 in) and the neckline long,
its neck edge straight or slightly convex (dresspatternmaking.com, basic collar); the roll line lies about 1 in up from
the neck edge and the stand is the part between the two (thecreativecurator.com/collars); the back collar 2 3/4 in
(blog.fabrics-store.com/?p=13206); the crease is set by pad stitching, a bias undercollar and a tape drawn up along
the roll line (minervapatterns.com, collar and lapel construction).

The triangular bend (my own reading of the geometry, not from a source): at the back the roll line runs parallel to
the neck edge, the stand's height up; at the front it must run down to the neck edge's end, where the lapel's roll
takes over, so it crosses the band's end at an angle. A rectangle folded on a line that crosses it shows its turned
part as a triangle, and its square end comes out turned by twice that angle.

Method (closed form, then a push off the coat):
  1. the seam: the neckline from her left lapel's corner G round the back to her right's (BUILD's collar.npz carries
     the neckline and the stand's up along it; the lapels' corners come from lapels.npz)
  2. the flat band: x along the seam, v across it, 0 at the neck edge to `width`; rounded corners (`corner`)
  3. the roll line on the band: v = h(x), `stand` high round the back, running down to the neck edge at each end at
     `fold_angle` to it
  4. the stand is laid up the neck from the seam; the fall is the band past the roll line, turned over it (mirrored
     across the roll line on the flat, so every length is kept) and opened off the stand by `open` degrees
  5. the fall is pushed off the coat and the lapels (BUILD's build.npz) by the cloth's thickness

Options: width 2.2, stand 0.9, fold_angle 45, corner 0.5, open 25, mesh 0.06, lift 0.1, clear 0.14.
Shaping to the neck (off by default): fold_k (the roll line's turn spread, 3), lift_run (the crease's turn, 0.3),
push_even/push_grow (600/20), shape_cut (the flat cut solved from the lay; saves uv and uv_rect).
Units are the build's (1 = 2.62 cm)."""
import sys, json, math
import numpy as np
from scipy.spatial import cKDTree, Delaunay
from scipy.ndimage import gaussian_filter1d
from matplotlib.path import Path


def build(B, opt):
    Wd = opt.get('width', 2.2); h0 = opt.get('stand', 0.9); al = math.radians(opt.get('fold_angle', 45.0))
    rc = opt.get('corner', 0.5); op = math.radians(opt.get('open', 25.0)); hm = opt.get('mesh', 0.06)
    lift = opt.get('lift', 0.1); clr = opt.get('clear', 0.14)
    C = np.load(B + '/out.blend.collar.npz'); E, upE = C['neckline'], C['neck_up']
    L = np.load(B + '/lapel.blend.lapels.npz'); M = np.load(B + '/build.npz', allow_pickle=True)
    # 1. the seam: from G (her left, front 2) round the back to G (her right, front 1)
    def corner_G(s):
        m = L['inner' + s] if ('inner' + s) in L.files else L['edge' + s]
        return L['pts' + s][m][np.argmax(L['d' + s][m])] if ('inner' + s) in L.files else L['crease' + s][0]
    iL = int(cKDTree(E).query(corner_G('2'))[1]); iR = int(cKDTree(E).query(corner_G('1'))[1])
    iL += int(opt.get('seam_from', 0)); iR -= int(opt.get('seam_from', 0))
    P = E[iL:iR + 1].copy(); U = upE[iL:iR + 1].copy()
    sl = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]); Ls = sl[-1]
    xs = np.arange(0, Ls, 0.02); P = np.column_stack([np.interp(xs, sl, P[:, k]) for k in range(3)])
    U = np.column_stack([np.interp(xs, sl, U[:, k]) for k in range(3)])
    P = gaussian_filter1d(P, 8, axis=0, mode='nearest'); T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1)[:, None]
    U -= (U * T).sum(1)[:, None] * T; U /= np.linalg.norm(U, axis=1)[:, None]
    U = gaussian_filter1d(U, 8, axis=0, mode='nearest'); U /= np.linalg.norm(U, axis=1)[:, None]
    cen = P.mean(0); Nn = np.cross(T, U); Nn /= np.linalg.norm(Nn, axis=1)[:, None]
    if np.mean(((P - cen) * Nn).sum(1)) < 0: Nn = -Nn                     # (outward, away from the neck)
    def emb(x, v, off=0.0):
        """A point of the band's own plane at the seam: x along the seam (carried on straight past its ends), v up."""
        xc = np.clip(x, 0, xs[-1]); k = np.clip(xc / 0.02, 0, len(xs) - 1.001); i = k.astype(int); f = (k - i)[:, None]
        p = (1 - f) * P[i] + f * P[i + 1]; u = (1 - f) * U[i] + f * U[i + 1]; n = (1 - f) * Nn[i] + f * Nn[i + 1]; t = (1 - f) * T[i] + f * T[i + 1]
        return p + (x - xc)[:, None] * t + v[:, None] * u + off * n, t, u, n
    # 2. the flat band, corners rounded
    def arc(c, r, a0, a1, n=10):
        a = np.linspace(a0, a1, n); return np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)])
    r0 = min(rc, 0.45 * Wd); rn = opt.get('corner_neck', 0.08)
    poly = np.vstack([arc((rn, rn), rn, np.pi, 1.5 * np.pi), arc((Ls - rn, rn), rn, 1.5 * np.pi, 2 * np.pi),
                      arc((Ls - r0, Wd - r0), r0, 0, 0.5 * np.pi), arc((r0, Wd - r0), r0, 0.5 * np.pi, np.pi)])
    def densify(Q, h):
        o = []
        for a, b in zip(Q, np.vstack([Q[1:], Q[:1]])):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / h))); o += [a + (b - a) * i / n for i in range(n)]
        return np.array(o)
    bnd = densify(poly, hm); path = Path(poly)
    # 3. the roll line on the band
    xf = np.arange(0, Ls + 1e-9, 0.01); hf = np.minimum(h0, np.tan(al) * np.minimum(xf, Ls - xf))
    hf = gaussian_filter1d(hf, opt.get('fold_round', 8), mode='nearest'); hf[0] = hf[-1] = 0.0
    if opt.get('fold_k'):
        # (the turn spread along the band: the same angle at the end, the stand reached gradually. A roll line that
        # turns in a short run folds its fall onto itself there, a wedge of twice the angle; spread, the same wedge is
        # a few percent of length the iron takes up. k large is the sharp turn again)
        k = float(opt['fold_k']); tt = np.tan(al) * np.minimum(xf, Ls - xf) / h0; hf = h0 * tt / (1 + tt ** k) ** (1 / k)
    F = np.column_stack([xf, hf]); tF = np.gradient(F, axis=0); tF /= np.linalg.norm(tF, axis=1)[:, None]
    gx, gy = np.meshgrid(np.arange(0, Ls, hm), np.arange(0, Wd, hm * 0.866)); gx = gx + (np.arange(gx.shape[0])[:, None] % 2) * 0.5 * hm
    G2 = np.column_stack([gx.ravel(), gy.ravel()]); G2 = G2[path.contains_points(G2)]
    G2 = G2[cKDTree(bnd).query(G2)[0] > 0.6 * hm]; G2 = G2[cKDTree(F).query(G2)[0] > 0.5 * hm]
    Fm = F[::max(1, int(hm / 0.01))]; Fm = Fm[path.contains_points(Fm) & (cKDTree(bnd).query(Fm)[0] > 0.5 * hm)]   # (the roll line's own points: the crease is an edge of the mesh)
    UV = np.vstack([bnd, Fm, G2]); tri = Delaunay(UV).simplices; tri = tri[path.contains_points(UV[tri].mean(1))]
    A2 = UV[tri]; ar = 0.5 * np.abs((A2[:, 1, 0] - A2[:, 0, 0]) * (A2[:, 2, 1] - A2[:, 0, 1]) - (A2[:, 2, 0] - A2[:, 0, 0]) * (A2[:, 1, 1] - A2[:, 0, 1]))
    tri = tri[ar > 0.03 * hm * hm]
    # 4. the stand up the neck, the fall turned over the roll line
    hx = np.interp(UV[:, 0], xf, hf); fall = UV[:, 1] > hx + 1e-9
    X3 = np.zeros((len(UV), 3)); X3[~fall] = emb(UV[~fall, 0], UV[~fall, 1])[0]
    _, jf = cKDTree(F).query(UV[fall]); Fq, tq = F[jf], tF[jf]; dq = UV[fall] - Fq
    mir = Fq + 2 * (dq * tq).sum(1)[:, None] * tq - dq                         # (mirrored across the roll line, on the flat)
    Pm, _, _, nm = emb(mir[:, 0], mir[:, 1], lift)                              # (laid against the stand, the cloth's thickness off it)
    if opt.get('lift_run'):                                                     # (the turn of the cloth at the crease: the thickness reached over a short run, not in one edge)
        rho = np.linalg.norm(dq, axis=1); Pm = Pm - (lift * (1 - np.clip(rho / opt['lift_run'], 0, 1)))[:, None] * nm
    Pf, tfx, ufx, nfx = emb(Fq[:, 0], Fq[:, 1], 0.0 if opt.get('lift_run') else 0.5 * lift)
    ax = tq[:, :1] * tfx + tq[:, 1:] * ufx; ax /= np.linalg.norm(ax, axis=1)[:, None]      # (the roll line's own direction in space)
    r = Pm - Pf; c_, s_ = math.cos(op), math.sin(op)
    rot = lambda sg: Pf + r * c_ + sg * np.cross(ax, r) * s_ + ax * (ax * r).sum(1)[:, None] * (1 - c_)
    Ra, Rb = rot(1.0), rot(-1.0)                                                # (opened off the stand: the way that goes outward)
    out_a = ((Ra - Pm) * nm).sum(1).mean(); X3[fall] = Ra if out_a > 0 else Rb
    # 5. off the coat: the body and the lapels, by their own outward normals
    BX, BT = M['body_X'], M['body_T']; fn = np.cross(BX[BT[:, 1]] - BX[BT[:, 0]], BX[BT[:, 2]] - BX[BT[:, 0]])
    vn = np.zeros_like(BX)
    for k in range(3): np.add.at(vn, BT[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1), 1e-12)[:, None]
    near = np.linalg.norm(BX - P.mean(0), axis=1) < 9.0; KX, Kn = BX[near], vn[near]; cx = np.array([0.0, float(P[:, 1].mean()), 0.0])
    rad = KX - cx; rad[:, 2] = 0; Kn = np.where(((Kn * rad).sum(1) < 0)[:, None] & (np.abs(Kn[:, 2]) < 0.9)[:, None], Kn, Kn)   # (as the mesh has them)
    tree = cKDTree(KX); idx = np.nonzero(fall)[0]; moved = np.zeros(len(UV))
    # (pushed out along the band's own outward normal there, a smooth field, and by an amount evened over the cloth:
    # pushing each point along the coat's nearest normal sent neighbours different ways over the lapels' layers and
    # crumpled the ends)
    ed_ = np.unique(np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1), axis=0)
    from scipy.sparse import coo_matrix
    nA = len(UV); Adj = coo_matrix((np.ones(2 * len(ed_)), (np.concatenate([ed_[:, 0], ed_[:, 1]]), np.concatenate([ed_[:, 1], ed_[:, 0]]))), shape=(nA, nA)).tocsr()
    dg = np.maximum(np.asarray(Adj.sum(1)).ravel(), 1); dirn = np.zeros((nA, 3)); dirn[idx] = nm
    for _ in range(opt.get('push_rounds', 2)):
        d, j = tree.query(X3[idx]); sd = ((X3[idx] - KX[j]) * Kn[j]).sum(1)
        need = np.zeros(nA); need[idx] = np.where((d < 0.6) & (sd < clr), clr - sd, 0.0)
        for _s in range(opt.get('push_grow', 6)):                # (the envelope: every point at least what its neighbours need)
            nb_max = np.zeros(nA); np.maximum.at(nb_max, ed_[:, 0], need[ed_[:, 1]]); np.maximum.at(nb_max, ed_[:, 1], need[ed_[:, 0]])
            need = np.maximum(need, 0.9 * nb_max) * fall
        for _s in range(opt.get('push_even', 30)): need = (0.5 * need + 0.5 * (Adj @ need) / dg) * fall     # (then evened)
        X3 += need[:, None] * dirn; moved += need
    # the strain the laying left (each edge against its length on the flat)
    ed = np.unique(np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1), axis=0)
    l0 = np.linalg.norm(UV[ed[:, 0]] - UV[ed[:, 1]], axis=1); l1 = np.linalg.norm(X3[ed[:, 0]] - X3[ed[:, 1]], axis=1); st = l1 / l0 - 1
    info = dict(seam_cm=2.62 * Ls, width_cm=2.62 * Wd, stand_cm=2.62 * h0, fold_angle=math.degrees(al), strain_mean=float(np.abs(st).mean()),
                strain_p95=float(np.percentile(np.abs(st), 95)), pushed=int((moved > 0).sum()), points=len(UV), faces=len(tri))
    return dict(uv=UV, pts=X3, tri=tri, fall=fall, fold=F, strain=st, edges=ed), info


def relax(R, B, opt):
    """Let the cloth take up what the laying stretched: edges back toward their flat lengths, the seam and stand held."""
    UV, X, ed, fall = R['uv'], R['pts'].copy(), R['edges'], R['fall']
    l0 = np.linalg.norm(UV[ed[:, 0]] - UV[ed[:, 1]], axis=1); free = fall.astype(float)
    X0 = X.copy()
    from scipy.sparse import coo_matrix
    n = len(X); A = coo_matrix((np.ones(2 * len(ed)), (np.concatenate([ed[:, 0], ed[:, 1]]), np.concatenate([ed[:, 1], ed[:, 0]]))), shape=(n, n)).tocsr()
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    for it in range(opt.get('relax_rounds', 150)):
        d = X[ed[:, 1]] - X[ed[:, 0]]; l = np.linalg.norm(d, axis=1); c = ((l - l0) / np.maximum(l, 1e-9))[:, None] * d
        wa, wb = free[ed[:, 0]], free[ed[:, 1]]; ws = np.maximum(wa + wb, 1e-9)
        dx = np.zeros_like(X); cnt = np.zeros(len(X))
        np.add.at(dx, ed[:, 0], c * (wa / ws)[:, None]); np.add.at(dx, ed[:, 1], -c * (wb / ws)[:, None])
        np.add.at(cnt, ed[:, 0], 1); np.add.at(cnt, ed[:, 1], 1)
        X += 0.9 * dx / np.maximum(cnt, 1)[:, None]
        # (the cloth's stiffness: what each point has moved is evened out with its neighbours', or a sheet with no
        # bending resistance takes up compression by crumpling)
        dl = X - X0; av = (A @ dl) / deg[:, None]; X = X0 + dl + opt.get('bend', 0.6) * free[:, None] * (av - dl)
    l1 = np.linalg.norm(X[ed[:, 0]] - X[ed[:, 1]], axis=1); st = l1 / l0 - 1
    return X, st


def shape_cut(R, opt):
    """The cut shaped to the neck: the flat pattern whose lengths are the laid collar's own.

    John, 2026-11-24: "Adjust the shape more for the neck deformation". The rectangle laid round a neck that flares to
    the shoulders is stretched and squeezed (the stand's top edge runs a shorter way round than the seam, the fall's
    outer edge a longer one). A tailor does not leave that to the cloth: the band is cut curved, so each line of it
    already has the length its place on the neck asks (dresspatternmaking.com: the straighter the neck edge the higher
    the collar stands, the more curved the flatter it lies). Here the curve is not drawn by eye but solved: the flat
    points are moved until every edge on the flat is as long as it lies on the neck (sparse least squares, the
    rectangle as the start). What is left is what no flat cut can give, the iron's share."""
    from scipy.optimize import least_squares
    from scipy.sparse import coo_matrix
    UV, X, ed = R['uv'], R['pts'], R['edges']; n = len(UV); m = len(ed)
    l3 = np.linalg.norm(X[ed[:, 0]] - X[ed[:, 1]], axis=1); w = 1e-3
    def res(z):
        Q = z.reshape(n, 2); d = Q[ed[:, 0]] - Q[ed[:, 1]]
        return np.concatenate([np.linalg.norm(d, axis=1) - l3, w * (z - UV.ravel())])
    def jac(z):
        Q = z.reshape(n, 2); d = Q[ed[:, 0]] - Q[ed[:, 1]]; g = d / np.maximum(np.linalg.norm(d, axis=1), 1e-12)[:, None]
        r = np.repeat(np.arange(m), 4); c = np.column_stack([2 * ed[:, 0], 2 * ed[:, 0] + 1, 2 * ed[:, 1], 2 * ed[:, 1] + 1]).ravel()
        v = np.column_stack([g[:, 0], g[:, 1], -g[:, 0], -g[:, 1]]).ravel()
        r = np.concatenate([r, m + np.arange(2 * n)]); c = np.concatenate([c, np.arange(2 * n)]); v = np.concatenate([v, np.full(2 * n, w)])
        return coo_matrix((v, (r, c)), shape=(m + 2 * n, 2 * n)).tocsr()
    sol = least_squares(res, UV.ravel(), jac=jac, method='trf', tr_solver='lsmr', max_nfev=opt.get('cut_rounds', 60))
    Q = sol.x.reshape(n, 2); tri = R['tri']
    a0 = lambda Z: (Z[tri[:, 1], 0] - Z[tri[:, 0], 0]) * (Z[tri[:, 2], 1] - Z[tri[:, 0], 1]) - (Z[tri[:, 2], 0] - Z[tri[:, 0], 0]) * (Z[tri[:, 1], 1] - Z[tri[:, 0], 1])
    flipped = int((np.sign(a0(Q)) != np.sign(a0(UV))).sum())
    st = l3 / np.maximum(np.linalg.norm(Q[ed[:, 0]] - Q[ed[:, 1]], axis=1), 1e-9) - 1
    # the cut's measures: how far the neck edge and the outer edge bow off their chords, and their lengths
    Ls = UV[:, 0].max(); Wd = UV[:, 1].max(); out = {}
    for nm_, sel in (('neck', UV[:, 1] < 1e-6), ('outer', UV[:, 1] > Wd - 1e-6)):
        i = np.nonzero(sel)[0]; i = i[np.argsort(UV[i, 0])]; E_ = Q[i]; ch = E_[-1] - E_[0]; L_ = np.linalg.norm(ch); nrm = np.array([-ch[1], ch[0]]) / L_
        off = (E_ - E_[0]) @ nrm; out[nm_ + '_bow_cm'] = float(2.62 * off[np.argmax(np.abs(off))])
        out[nm_ + '_len_cm'] = float(2.62 * np.linalg.norm(np.diff(E_, axis=0), axis=1).sum()); out[nm_ + '_len_rect_cm'] = float(2.62 * (UV[i, 0].max() - UV[i, 0].min()))
    out.update(cut_strain_mean=float(np.abs(st).mean()), cut_strain_p95=float(np.percentile(np.abs(st), 95)), cut_flipped=flipped)
    return Q, st, out


if __name__ == '__main__':
    B = sys.argv[1].rstrip('/'); out = sys.argv[2]; opt = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    R, info = build(B, opt)
    if opt.get('relax', True):
        X, st = relax(R, B, opt); info['strain_mean_relaxed'] = float(np.abs(st).mean()); info['strain_p95_relaxed'] = float(np.percentile(np.abs(st), 95))
        info['moved_mean'] = float(np.linalg.norm(X - R['pts'], axis=1).mean()); R['pts'] = X; R['strain'] = st
    if opt.get('shape_cut'):
        Q, st, ci = shape_cut(R, opt); info.update(ci); R['uv_rect'] = R['uv']; R['uv'] = Q; R['strain'] = st
    np.savez(out, **R); json.dump(dict(opt=opt, info=info), open(out + '.json', 'w'), indent=1)
    print('foldcollar: ' + ', '.join('%s %s' % (k, ('%.3f' % v) if isinstance(v, float) else v) for k, v in info.items()))
