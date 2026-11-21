"""A small tension solver for a laced opening: two leather panels whose front edges stand a gap apart, an eyelet row
in each, and a cord criss-crossing between the rows over the tongue. It finds the lacing's proportions (the gap, the
eyelets' inset from the edge, the offset of one row against the other) from the forces instead of by eye, and gives
them as ratios to the eyelet pitch and to the head's height H, so they carry to any boot size.

The lacing has one job: to hold the shaft closed round the leg, a hoop pull N (force per unit length of the opening)
set by how firmly the boot grips (fit pressure q times the leg's radius). Everything else follows from statics:

  the cord     Each diagonal runs from an eyelet to the next one up on the other side. With the rows D apart
               (D = gap + 2 * inset) and the pitch p, a diagonal leans at theta from the across direction,
               tan(theta) = p / D, and only cos(theta) of its tension closes the opening. So the cord tension is
               T = N p / (2 cos(theta)). Steep diagonals (rows too close) need a large tension for the same hold:
               they run along the opening and do no structural work. Flat diagonals (rows too far apart) are long,
               so they stretch more under the same hold, and the cord turns almost a hairpin at every eyelet, so
               friction there grows. The cord's stored energy per pitch, T^2 l / (2 EA), is least at D = sqrt(2) p
               (frictionless), where the closure is stiffest: the cord's lateral stiffness EA D^2 / l^3 peaks there.
  friction     At each eyelet the cord turns (in the plane, and over the ring's rim), so the tension of the diagonal
               above is exp(mu * turn) times the one below (capstan). The difference pulls the eyelet along the
               edge: the sideways pull F_t. It also spreads the diagonals' tensions either side of an eyelet.
  the leather  Each eyelet hands the leather a lateral pull F_x = N p and the sideways pull F_t. A small plane-stress
               finite-element cell (one pitch of panel, periodic up the opening, free at the front edge, the ring
               bearing on the side of its hole it is pulled against) gives the leather's strain energy for unit
               loads as a function of the inset. It changes only a few per cent from a thin ligament to a deep one.
  the front    The cord lies on the tongue where the front is convex along it and bridges where it is concave
               (the hollow at the front of the ankle). Its normal curvature along a diagonal is
               cos^2(theta) / R_across + sin^2(theta) * k_along; where it is concave the cord stands off the tongue
               by k l_span^2 / 8 over the span of the gap. Steep diagonals feel more of the ankle's hollow.

The cost is the stored elastic energy per unit length of the opening (cord plus leather, in N mm per mm), times one
plus the penalties: a diagonal whose closing share cos(theta) falls below share_min (it runs along the opening and
no longer does its part), a cord that lifts off the tongue by more than its radius, a ligament thinner than the
rule. A grid over (gap, inset, offset) is searched; it is cheap (numpy, about ten seconds with the leather cells).

The leather's energy turns out nearly flat in the inset (a few per cent from a thin ligament to a deep one: the
eyelet's pull spreads into the panel either way, and at a boot's loads the ligament is far from tearing), so the
inset is the leatherworker's rule (ligament_min: leather at least half a ring's radius wide between the ring and the
edge), the shallowest it allows, which shows the most tongue. At that inset the gap and the offset are the least cost.
The report also gives the window of gaps within `tol` of it: the cost is flat near its least, so anything in the
window is as good, and past it the cost climbs, either way.

    python3 blender/shared/lace_tension.py                  # the boot's numbers, report and plots
    python3 blender/shared/lace_tension.py --out DIR        # where the plots go
    python3 blender/shared/lace_tension.py --set mu=0.3 --set q=4000

From Python (also inside Blender: numpy only, matplotlib only for the plots):

    import lace_tension as lt
    res = lt.solve()                 # or lt.solve(dict(lt.DEFAULTS, pitch=0.08))
    res['ratios']                    # last.RATIOS names: lace_gap, eyelet_in, ... as shares of H
"""
import argparse
import json
import math
import os
import sys

import numpy as np

# Lengths are shares of the head's height H (as in blender/boot/last.py RATIOS), forces in newtons, H in mm.
DEFAULTS = dict(
    H=250.0,              # mm: the head's height (the boot's leather is 0.014 H, about 3.5 mm)
    pitch=0.085,          # the eyelets' pitch up the opening (last.RATIOS eyelet_pitch, about 2.1 cm)
    eyelet=0.018,         # an eyelet ring's outer radius (last.RATIOS eyelet)
    lace=0.004,           # the cord's radius (1.0 mm, the hose thread's)
    leather=0.014,        # the leather's thickness (last.RATIOS leather)
    q=3000.0,             # Pa: the grip of the laced shaft on the leg (a firm boot, about 22 mmHg)
    leg_r=0.18,           # the leg's radius where the lacing sits (about 4.5 cm)
    E_leather=150.0,      # MPa: vegetable-tanned boot leather in its plane
    nu=0.4,               # its Poisson ratio
    EA_cord=400.0,        # N: the cord's axial stiffness (a 2 mm leather thong; waxed cotton is about 1000)
    mu=0.2,               # the cord's friction on a brass ring
    rim=0.5 * math.pi,    # rad: the turn over the ring's rims, on top of the turn in the plane
    R_across=0.16,        # the front's radius across the opening (the shin, about 4 cm)
    k_along=(0.0, -1.0 / 0.30, 1.0 / 0.6),   # its curvature up the opening (per H) at the shin, the ankle's hollow,
                                             # the instep: the worst one governs the cord's lift
    share_min=0.5,        # the least share of a diagonal's tension that must close the opening (cos theta: 0.5 is
                          # 60 degrees from across), before a penalty
    ligament_min=0.5,     # the least leather between the ring and the edge, as a share of the ring's radius
    gap_min=0.0,          # a tongue at least this wide must show (0: no floor)
    tol=0.03,             # the window of gaps reported: within this share of the least cost
)

COMPARE = {                                   # designs to mark on the plots (gap, inset, as shares of H)
    'boot on main': (0.035, 0.028),
    'hose thread (2.4 cm)': (0.096, 0.028),
}


# ---------------------------------------------------------------------------------------------------------------
# the leather: one pitch of panel, plane stress, periodic up the opening

def leather_cell(p, a, e, E=1.0, nu=0.4, t=1.0, n_up=32, width=1.25, field=False):
    """Strain energy of one pitch of panel loaded at its eyelet, for unit loads. The panel is x in [0, e + width p]
    (x = 0 the free front edge, the back held), y in [0, p) periodic; the ring (radius a, centre e in from the edge, at
    mid-pitch) is a hole, and the ring bears on the half of its rim it is pulled against. Returns (cx, cy): the
    energy for a unit pull across (towards the gap) and along the edge, times E t (so dimensionless). With
    field=True also the mesh and the strain energy density of the across load, for the contour plot."""
    h = p / n_up
    nx = int(round((e + width * p) / h)) + 1           # the back held width p behind the eyelet, whatever the inset
    ny = n_up
    xs, ys = np.arange(nx) * h, np.arange(ny) * h
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    nid = np.arange(nx * ny).reshape(nx, ny)
    tris = []
    for i in range(nx - 1):
        for j in range(ny):
            j2 = (j + 1) % ny
            a0, a1, a2, a3 = nid[i, j], nid[i + 1, j], nid[i + 1, j2], nid[i, j2]
            tris += [(a0, a1, a2), (a0, a2, a3)]
    tris = np.array(tris)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    # coordinates of each triangle, unwrapped across the periodic seam
    C = P[tris].copy()
    for k in (1, 2):
        wrap = C[:, k, 1] - C[:, 0, 1] < -0.5 * p
        C[wrap, k, 1] += p
    D = E / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, 0.5 * (1 - nu)]])
    x1, y1, x2, y2, x3, y3 = C[:, 0, 0], C[:, 0, 1], C[:, 1, 0], C[:, 1, 1], C[:, 2, 0], C[:, 2, 1]
    A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    b = np.stack([y2 - y3, y3 - y1, y1 - y2], 1) / A2[:, None]
    c = np.stack([x3 - x2, x1 - x3, x2 - x1], 1) / A2[:, None]
    B = np.zeros((len(tris), 3, 6))
    B[:, 0, 0::2], B[:, 1, 1::2], B[:, 2, 0::2], B[:, 2, 1::2] = b, c, c, b
    Ke = t * 0.5 * np.abs(A2)[:, None, None] * np.einsum('eki,kl,elj->eij', B, D, B)
    # the hole: no leather inside the ring. The ring bears on the half of the hole's rim it is pulled against (the
    # half facing the edge for the pull across, the upper half for the pull along); the nodes there move with it, as
    # one rigid body (two master dofs). The back edge is held. Each node's dofs map straight to the reduced system's
    # (-1: held or in the hole), so the stiffness is assembled already reduced.
    r_ = np.hypot(P[:, 0] - e, P[:, 1] - 0.5 * p)
    cen = C.mean(axis=1)
    solid = np.hypot(cen[:, 0] - e, cen[:, 1] - 0.5 * p) > a
    tris, C, B, Ke = tris[solid], C[solid], B[solid], Ke[solid]
    used = np.zeros(len(P), bool)
    used[tris.ravel()] = True
    rim = used & (r_ < a + 0.75 * h)
    back = P[:, 0] > (nx - 1) * h - 1e-9
    U, cs = [], []
    for case, contact in ((0, rim & (P[:, 0] <= e + 1e-9)), (1, rim & (P[:, 1] >= 0.5 * p - 1e-9))):
        free = used & ~contact & ~back
        fidx = np.nonzero(free)[0]
        nf = 2 * len(fidx)
        m = -np.ones(2 * len(P), dtype=np.int64)
        m[2 * fidx], m[2 * fidx + 1] = np.arange(0, nf, 2), np.arange(1, nf, 2)
        m[2 * np.nonzero(contact)[0]], m[2 * np.nonzero(contact)[0] + 1] = nf, nf + 1
        dofs = m[np.stack([2 * tris, 2 * tris + 1], 2).reshape(len(tris), 6)]
        keep = (dofs[:, :, None] >= 0) & (dofs[:, None, :] >= 0)
        rows = np.broadcast_to(dofs[:, :, None], Ke.shape)[keep]
        cols = np.broadcast_to(dofs[:, None, :], Ke.shape)[keep]
        Kr = np.zeros((nf + 2, nf + 2))
        np.add.at(Kr, (rows, cols), Ke[keep])
        Fv = np.zeros(nf + 2)
        Fv[nf + case] = -1.0 if case == 0 else 1.0       # across: towards the gap (x = 0); along: up the edge
        V = np.linalg.solve(Kr, Fv)
        cs.append(0.5 * Fv @ V * E * t)
        U.append(np.where(m >= 0, V[np.maximum(m, 0)], 0.0).reshape(-1, 2))
    cx, cy = cs
    if not field:
        return cx, cy
    u = U[0]
    ue = u[tris].reshape(len(tris), 6)
    eps = np.einsum('eij,ej->ei', B, ue)
    w = 0.5 * np.einsum('ei,ij,ej->e', eps, D, eps)
    return cx, cy, dict(P=P, tris=tris, C=C, w=w, u=u)


def leather_table(p, a, E, nu, t, insets):
    """cx, cy at each inset (shares of H, like p and a)."""
    cx, cy = [], []
    for e in insets:
        x, y = leather_cell(p, a, e, E, nu, t)
        cx.append(x)
        cy.append(y)
    return np.array(cx), np.array(cy)


# ---------------------------------------------------------------------------------------------------------------
# the cord and the cost

def lacing_forces(prm, gap, inset, offset):
    """Everything about the lacing at these proportions (arrays broadcast; lengths in mm, forces in N).

    The rows are D = gap + 2 inset apart; the right row stands offset higher than the left. The diagonals going up to
    the right (family a) rise p + offset in a pitch, those going up to the left (b) p - offset."""
    H = prm['H']
    p, gap, inset, offset = prm['pitch'] * H, gap * H, inset * H, offset * H
    N = prm['q'] * 1e-6 * prm['leg_r'] * H                 # N/mm: the hoop pull the lacing holds
    D = gap + 2 * inset
    dya, dyb = p + offset, p - offset
    la, lb = np.hypot(D, dya), np.hypot(D, dyb)
    ca, cb, sa, sb = D / la, D / lb, dya / la, dyb / lb
    # the turn at an eyelet, in the plane (between the diagonal in and the one out) and over the rims
    turn = np.arccos(np.clip((dya * dyb - D * D) / (la * lb), -1, 1)) + prm['rim']
    f = np.exp(prm['mu'] * turn)
    rf = np.sqrt(f)
    # the tension: the two eyelets of a pitch (one each side) together hold 2 N p
    T = 2 * N * p / ((ca + cb) * (rf + 1 / rf))
    Fx_R, Ft_R = T / rf * ca + T * rf * cb, T * rf * sb - T / rf * sa      # right eyelet: a comes in, b goes out
    Fx_L, Ft_L = T / rf * cb + T * rf * ca, T * rf * sa - T / rf * sb      # left eyelet: b comes in, a goes out
    share = np.minimum(ca, cb)                    # the share of a diagonal's tension that closes the opening
    # the cord's energy in a pitch (a and b, with the friction spread above and below)
    U_cord = T ** 2 * (la + lb) / (2 * prm['EA_cord']) * 0.5 * (f + 1 / f)
    # the cord's lift off the tongue over the gap, at the worst station
    R = prm['R_across'] * H
    lift = np.zeros(np.broadcast(D, offset).shape)
    for k in prm['k_along']:
        for c_, l_ in ((ca, la), (cb, lb)):
            kn = c_ ** 2 / R + (1 - c_ ** 2) * k / H
            span = l_ * np.clip(gap / np.maximum(D, 1e-9), 0, 1)
            lift = np.maximum(lift, np.maximum(-kn, 0) * span ** 2 / 8)
    return dict(p=p, N=N, D=D, la=la, lb=lb, ca=ca, cb=cb, theta_a=np.degrees(np.arctan2(dya, D)),
                theta_b=np.degrees(np.arctan2(dyb, D)), turn=turn, f=f, T=T, Fx_R=Fx_R, Ft_R=Ft_R, Fx_L=Fx_L,
                Ft_L=Ft_L, share=share, U_cord=U_cord, lift=lift)


def cost(prm, gap, inset, offset, cx, cy):
    """The cost at each point and its parts. cx, cy: the leather cell's energies at each inset (same shape)."""
    L = lacing_forces(prm, gap, inset, offset)
    Et = prm['E_leather'] * prm['leather'] * prm['H']                     # N/mm
    U_leather = (cx * (L['Fx_R'] ** 2 + L['Fx_L'] ** 2) + cy * (L['Ft_R'] ** 2 + L['Ft_L'] ** 2)) / Et
    U = (L['U_cord'] + U_leather) / L['p']                                 # N mm per mm of opening
    r_c = prm['lace'] * prm['H']
    lig = (inset - prm['eyelet']) / prm['eyelet']
    pen_share = 25.0 * np.maximum(prm['share_min'] - L['share'], 0) ** 2 / prm['share_min'] ** 2
    pen_lift = np.maximum(L['lift'] / r_c - 1.0, 0) ** 2
    pen_lig = 100.0 * np.maximum(prm['ligament_min'] - lig, 0) ** 2
    pen_gap = 100.0 * np.maximum(prm['gap_min'] - gap, 0) ** 2 / prm['pitch'] ** 2
    J = U * (1 + pen_share + pen_lift + pen_lig + pen_gap)
    L.update(U=U, U_cord_l=L['U_cord'] / L['p'], U_leather_l=U_leather / L['p'], J=J, pen_share=pen_share,
             pen_lift=pen_lift, pen_lig=pen_lig, ligament=lig)
    return L


def solve(prm=None, n_gap=121, n_inset=13, n_offset=21, log=print):
    """Search (gap, inset, offset). Returns a dict: 'best' (the widest gap within tol of the least cost), 'least'
    (the least cost), 'ratios' (last.RATIOS names), and the grids for the plots."""
    prm = dict(DEFAULTS, **(prm or {}))
    p, a = prm['pitch'], prm['eyelet']
    insets = np.unique(np.concatenate([a * np.linspace(1.1, 3.5, n_inset), [a * (1 + prm['ligament_min'])]]))
    cx, cy = leather_table(p, a, prm['E_leather'], prm['nu'], prm['leather'], insets)
    gaps = p * np.linspace(0.0, 3.0, n_gap)
    offsets = p * np.linspace(-0.3, 0.3, n_offset)
    G, I, O = np.meshgrid(gaps, insets, offsets, indexing='ij')
    CX, CY = np.broadcast_to(cx[None, :, None], G.shape), np.broadcast_to(cy[None, :, None], G.shape)
    R = cost(prm, G, I, O, CX, CY)
    J = R['J']
    k = np.unravel_index(np.argmin(J), J.shape)
    # the inset: the rule's (the leather's energy hardly changes with it, so the shallowest the rule allows, which
    # shows the most tongue); the gap and the offset: the least cost there
    ii = int(np.argmin(np.abs(insets - prm['eyelet'] * (1 + prm['ligament_min']))))
    gi, oi = np.unravel_index(np.argmin(J[:, ii, :]), J[:, ii, :].shape)
    kb = (gi, ii, oi)
    ok = np.nonzero(J[:, ii, oi] <= (1 + prm['tol']) * J[kb])[0]
    window = (float(gaps[ok.min()]), float(gaps[ok.max()]))

    def point(idx):
        d = {name: float(np.asarray(v)[idx] if np.ndim(v) == 3 else v) for name, v in R.items()}
        d.update(gap=float(G[idx]), inset=float(I[idx]), offset=float(O[idx]))
        return d

    best, least = point(kb), point(k)
    ratios = dict(lace_gap=best['gap'], eyelet_in=best['inset'], eyelet_pitch=p, eyelet=a, lace=prm['lace'])
    cord = best['T'] / prm['EA_cord']
    res = dict(prm=prm, best=best, least=least, ratios=ratios, window=window,
               per_pitch=dict(gap=best['gap'] / p, inset=best['inset'] / p, rows=best['D'] / best['p'],
                              offset=best['offset'] / p),
               rest_length=1.0 / (1.0 + cord),
               grids=dict(gaps=gaps, insets=insets, offsets=offsets, cx=cx, cy=cy, J=J, R=R, kb=kb, k=k))
    if log:
        log(report(res))
    return res


def report(res):
    prm, b, l = res['prm'], res['best'], res['least']
    H = prm['H']
    out = ['lace tension solve: pitch %.1f mm, ring %.1f mm across, cord %.1f mm across, hold %.3f N/mm'
           % (prm['pitch'] * H, 2 * prm['eyelet'] * H, 2 * prm['lace'] * H, b['N'])]
    for name, d in (('best (inset at the ligament rule, gap and offset at the least cost there)', b),
                    ('least cost anywhere (the inset free: it covers the tongue for a saving of %.1f%%)'
                     % (100 * (1 - l['J'] / b['J'])), l)):
        out.append('  %s' % name)
        out.append('    gap %.1f mm (%.3f H, %.2f pitch), inset %.1f mm (%.3f H, %.2f pitch), rows %.1f mm apart '
                   '(%.2f pitch), offset %.2f pitch'
                   % (d['gap'] * H, d['gap'], d['gap'] / prm['pitch'], d['inset'] * H, d['inset'],
                      d['inset'] / prm['pitch'], d['D'], d['D'] / d['p'], d['offset'] / prm['pitch']))
        out.append('    diagonals at %.0f / %.0f deg from across, cord tension %.2f N, eyelet pull %.2f N across, '
                   '%.2f N along (sideways)' % (d['theta_a'], d['theta_b'], d['T'], 0.5 * (d['Fx_R'] + d['Fx_L']),
                                                0.5 * (abs(d['Ft_R']) + abs(d['Ft_L']))))
        out.append('    energy %.4f N (cord %.4f, leather %.4f), weakest share %.2f, lift %.2f mm, ligament %.2f ring '
                   'radii, cost %.4f' % (d['U'], d['U_cord_l'], d['U_leather_l'], d['share'], d['lift'],
                                         d['ligament'], d['J']))
    w = res['window']
    out.append('  gaps within %d%% of the best: %.1f to %.1f mm (%.2f to %.2f pitch; rows %.2f to %.2f pitch apart)'
               % (round(100 * prm['tol']), w[0] * H, w[1] * H, w[0] / prm['pitch'], w[1] / prm['pitch'],
                  (w[0] + 2 * b['inset']) / prm['pitch'], (w[1] + 2 * b['inset']) / prm['pitch']))
    out.append('  the cord, drawn up: rest length %.4f of its laid path (it stretches %.2f%%)'
               % (res['rest_length'], 100 * (1 / res['rest_length'] - 1)))
    out.append('  ratios: ' + json.dumps({k: round(v, 4) for k, v in res['ratios'].items()}))
    return '\n'.join(out)


def evaluate(prm, gap, inset, offset=0.0):
    """The cost and its parts at one design (shares of H)."""
    prm = dict(DEFAULTS, **(prm or {}))
    cx, cy = leather_cell(prm['pitch'], prm['eyelet'], inset, prm['E_leather'], prm['nu'], prm['leather'])
    R = cost(prm, np.array(gap), np.array(inset), np.array(offset), np.array(cx), np.array(cy))
    return {k: float(v) for k, v in R.items()}


# ---------------------------------------------------------------------------------------------------------------
# the plots

def plot(res, out_dir, name='lace-tension'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.tri import Triangulation

    prm, g, b = res['prm'], res['grids'], res['best']
    H, p, a = prm['H'], prm['pitch'], prm['eyelet']
    gi, ii, oi = g['kb']
    R = g['R']
    fig = plt.figure(figsize=(16, 10), dpi=110)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1])

    # 1. the lacing as solved, laid flat: leather strain energy round each eyelet, cords by tension
    ax = fig.add_subplot(gs[:, 0])
    _, _, F = leather_cell(p, a, b['inset'], prm['E_leather'], prm['nu'], prm['leather'], field=True)
    n_rows = 5
    pmm, gap, e, off = p * H, b['gap'] * H, b['inset'] * H, b['offset'] * H
    wmax = np.percentile(F['w'], 99)
    for side in (-1, 1):
        for k in range(n_rows):
            yk = k * pmm + (off if side > 0 else 0.0) - 0.5 * pmm
            Cx = side * (0.5 * gap + F['C'][..., 0] * H)
            Cy = F['C'][..., 1] * H + yk
            tri = Triangulation(Cx.ravel(), Cy.ravel(), np.arange(Cx.size).reshape(-1, 3))
            ax.tripcolor(tri, facecolors=np.minimum(F['w'], wmax), cmap='magma', shading='flat',
                         vmin=0, vmax=wmax, rasterized=True)
    ax.add_patch(plt.Rectangle((-0.5 * gap, -0.5 * pmm), gap, n_rows * pmm, color='#b08850', zorder=1))
    eyes = {s: [(s * (0.5 * gap + e), k * pmm + (off if s > 0 else 0.0)) for k in range(n_rows)] for s in (-1, 1)}
    segs = [[eyes[s_][k], eyes[-s_][k + 1]] for k in range(n_rows - 1) for s_ in (-1, 1)]
    ax.add_collection(LineCollection(segs, colors='#2a6fdb', linewidths=4, zorder=3))
    for s_ in (-1, 1):
        Fx, Ft = (b['Fx_R'], b['Ft_R']) if s_ > 0 else (b['Fx_L'], b['Ft_L'])
        for (x, y) in eyes[s_]:
            ax.add_patch(plt.Circle((x, y), a * H, fill=False, color='#e0c060', lw=2, zorder=4))
            ax.arrow(x, y, -s_ * Fx * 3, Ft * 3, width=0.5, color='#40e040', zorder=5, length_includes_head=True)
    ax.text(0, -0.5 * pmm + 1, 'cord %.2f N (x %.2f either side of an eyelet: friction)\ngreen: the pull on each '
            'eyelet, 3 mm per N' % (b['T'], math.sqrt(b['f'])), ha='center', va='bottom', fontsize=8, color='w',
            zorder=6)
    ax.set_xlim(-0.5 * gap - 1.2 * pmm, 0.5 * gap + 1.2 * pmm)
    ax.set_ylim(-0.6 * pmm, (n_rows - 0.4) * pmm)
    ax.set_aspect('equal')
    ax.set_title('best: gap %.1f mm, inset %.1f mm, rows %.2f pitch apart\nleather strain energy (dark low, bright '
                 'high) and cord tension' % (gap, e, b['D'] / b['p']), fontsize=10)
    ax.set_xlabel('mm across the opening')
    ax.set_ylabel('mm up the opening')

    # 2. cost against the gap at the best inset and offset, and its parts
    ax = fig.add_subplot(gs[0, 1])
    x = g['gaps'] / p
    ax.plot(x, R['J'][:, ii, oi], 'k', lw=2, label='cost')
    ax.plot(x, R['U_cord_l'][:, ii, oi], '#3070c0', label='cord energy')
    ax.plot(x, R['U_leather_l'][:, ii, oi], '#c06030', label='leather energy')
    best_per_gap = R['J'].min(axis=(1, 2))
    ax.plot(x, best_per_gap, 'k:', label='cost, inset and offset free')
    ax.axvline(b['gap'] / p, color='g', lw=1)
    ax.axhline((1 + prm['tol']) * best_per_gap.min(), color='g', lw=0.8, ls='--')
    for lbl, (gg, ee) in COMPARE.items():
        v = evaluate(prm, gg, ee)
        ax.plot(gg / p, v['J'], 'o', label='%s: %.3f' % (lbl, v['J']))
    ax.set_ylim(0, 3 * best_per_gap.min())
    ax.set_xlabel('gap / pitch (inset %.2f pitch)' % (b['inset'] / p))
    ax.set_ylabel('N mm per mm of opening')
    ax.legend(fontsize=7)
    ax.set_title('cost along the gap: green line, the answer', fontsize=10)

    # 3. the cost map over gap and inset (offset at its best), with the rows' spacing D / p as contours
    ax = fig.add_subplot(gs[0, 2])
    GG, II = np.meshgrid(g['gaps'] / p, g['insets'] / p, indexing='ij')
    Jm = R['J'].min(axis=2)
    lev = Jm.min() * np.array([1.0, 1.03, 1.1, 1.25, 1.5, 2, 3, 5])
    from matplotlib.colors import BoundaryNorm
    lv = np.concatenate([lev, [max(Jm.max(), lev[-1] * 1.01)]])
    cs = ax.contourf(GG, II, Jm, levels=lv, cmap='viridis_r', norm=BoundaryNorm(lv, 256))
    ax.contour(GG, II, Jm, levels=lev[1:2], colors='w', linewidths=1.5)
    Dp = GG + 2 * II
    cl = ax.contour(GG, II, Dp, levels=[1.0, 1.2, 1.414, 1.7, 2.0, 2.5, 3.0], colors='k', linewidths=0.6)
    ax.clabel(cl, fmt='D/p %.2f', fontsize=6)
    ax.plot(b['gap'] / p, b['inset'] / p, 'r*', ms=14)
    for lbl, (gg, ee) in COMPARE.items():
        ax.plot(gg / p, ee / p, 'o', mfc='none', mew=2, ms=9)
        ax.annotate(lbl, (gg / p, ee / p), fontsize=7, xytext=(4, 4), textcoords='offset points')
    fig.colorbar(cs, ax=ax, fraction=0.04).set_label('cost / least')
    cs.colorbar.set_ticks(lev)
    cs.colorbar.set_ticklabels(['%.2f' % (v / Jm.min()) for v in lev])
    ax.set_xlabel('gap / pitch')
    ax.set_ylabel('inset / pitch')
    ax.set_title('cost map (white: within %d%%), rows spacing contours' % round(100 * prm['tol']), fontsize=10)

    # 4. the forces against the rows' spacing
    ax = fig.add_subplot(gs[1, 1])
    Dl = R['D'][:, ii, oi] / R['p']
    ax.plot(Dl, R['T'][:, ii, oi], label='cord tension (N)')
    ax.plot(Dl, np.abs(R['Ft_R'][:, ii, oi]), label='sideways pull at an eyelet (N)')
    ax.plot(Dl, R['lift'][:, ii, oi], label='cord lift off the tongue (mm)')
    ax.axvline(math.sqrt(2), color='grey', ls=':', lw=1)
    ax.axvline(b['D'] / b['p'], color='g', lw=1)
    ax.set_xlabel('rows apart / pitch  (dotted: sqrt 2, the frictionless stiffest)')
    ax.set_ylim(0, None)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_title('forces along the rows\' spacing', fontsize=10)
    axs = ax.twinx()
    axs.plot(Dl, R['share'][:, ii, oi], 'g--')
    axs.axhline(prm['share_min'], color='g', lw=0.6, ls=':')
    axs.set_ylim(0, 1)
    axs.set_ylabel('closing share cos theta (dashed; dotted: the floor)', color='g')

    # 5. the offset of the rows, and the leather against the inset
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(g['offsets'] / p, R['J'][gi, ii, :] / R['J'][gi, ii, :].min(), 'k', label='cost against row offset')
    ax.set_xlabel('right row offset / pitch')
    ax.set_ylabel('cost / least')
    ax2 = ax.twiny()
    ax2.plot((g['insets'] - a) / a, g['cx'] / g['cx'].min(), '#c06030', label='leather, across load')
    ax2.plot((g['insets'] - a) / a, g['cy'] / g['cy'].min(), '#c06030', ls='--', label='leather, along load')
    ax2.set_xlabel('ligament (ring radii), leather energy over its least', color='#c06030')
    ax.set_ylim(0.95, 1.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper center')

    fig.suptitle('Lace tension solve: pitch %.1f mm, ring %.1f mm, cord %.1f mm, grip %.1f kPa, mu %.2f'
                 % (p * H, 2 * a * H, 2 * prm['lace'] * H, prm['q'] / 1000, prm['mu']))
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + '.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def _parse_set(items):
    out = {}
    for it in items or ():
        k, v = it.split('=', 1)
        if k not in DEFAULTS:
            raise SystemExit('unknown setting %r (one of %s)' % (k, ', '.join(DEFAULTS)))
        out[k] = json.loads(v) if v.strip().startswith(('[', '(')) else float(v)
        if isinstance(out[k], list):
            out[k] = tuple(out[k])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--set', action='append', help='name=value, one of DEFAULTS')
    ap.add_argument('--out', default=None, help='folder for the plot (default: no plot)')
    ap.add_argument('--json', default=None, help='write the result here')
    args = ap.parse_args(argv)
    res = solve(_parse_set(args.set))
    if args.out:
        print('plot:', plot(res, args.out))
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(dict(best=res['best'], least=res['least'], ratios=res['ratios'], per_pitch=res['per_pitch'],
                           rest_length=res['rest_length'], prm={k: v for k, v in res['prm'].items()}), fh, indent=1)
    return res


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
