"""Fold, then button (John, 2026-11-21: "proper work flow is cut, fit, then fold, and finally, after the fold looks
right, you button"; "don't weld: cut the bodice with the lapel in mind then fold").

    python3 blender/collar/foldfront.py CUT.blend OUT.blend ['{"foot": 47.2, "neck_x": 2.2, "buttons": true}']

CUT.blend is bodice_cut's: the bodice cut and fitted whole to the neck, two fronts, no lapel piece, no buttons.
How it is done properly (Fabrics-Store, "How to draft and sew a notched collar"; FreeSewing, Jaeger instructions;
Historical Tailoring Masterclasses, the frock coat's forepart; CLO's forum on blazer lapels):
  the cut     the lapel is no piece of its own: it is the front panel itself, running on past the roll line. The roll
              line is ONE line marked on the front, from the neck beside the shoulder to the break at the top button
              (the V "1-2 in above bust level").
  the fit     the front is fitted flat, unfolded ("remove any fold angles (back to 180 deg)"), and only then folded on
              that one line, on a fine mesh.
  the fold    a roll, held by a tape "about half a centimeter shorter" than the line: never a pressed crease.
  buttons     "Add buttonholes and buttons" is the last step.
So here: each front's own cloth is cut along its roll line (a clean line of edges in the mesh: nothing is taken out
and nothing is added), every point of that front past the line is turned back over it on a roll, lying on the front
with the cloth's thickness, and then the buttons are set down the front below the break.
"""
import bpy, bmesh, sys, json, numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

a = sys.argv
has = a[-1].lstrip().startswith('{')
opt = json.loads(a[-1]) if has else {}
blend, out = a[-3:-1] if has else a[-2:]
# The saved recipes: '{"recipe": "stand_collar"}' (any other option given beside it overrides the recipe's own).
#   stand_collar   John, 2026-11-21: "save the current recipe for building a beautiful stand collar" (the sheet
#                  bodice-button-on-curved-fold.png). The bodice's own neck edge stands round the neck as the collar,
#                  and each front is turned back from it on a curved roll line that sweeps in to the top button:
#                  the break 0.15 above the bust line, the line bowed 0.5 away from the V (a rounded V), a 1.3 cm roll,
#                  the top button just under where the two folds meet, cloth 4.4 mm thick (bodice_cut's THICK 0.085).
#                  See blender/collar/STAND_COLLAR.md.
RECIPES = {
    'stand_collar': dict(foot_over_bust=0.15, bow=-0.5, roll_radius=0.24, break_over=0.03, neck_x=2.2, sheet=6, fair=40,
                         ease_chest=True, ease_rounds=150, tail=0, bow_band=1.0, bow_over=1.5, lie=0.02, thick_out=1.0, t0=0.03),
}
if opt.get('recipe'):
    opt = dict(RECIPES[opt['recipe']], **{k: v for k, v in opt.items() if k != 'recipe'})
# the softened cusp is part of the traced lapel (not of the saved stand collar recipe, which carries 'bow'); the shape
# before it is kept as the ground truth (lapel-ground-truth/truth.npz: pass '{"truth": path}' to measure against it,
# '{"cusp": 0}' to build it again)
if 'cusp' not in opt and 'bow' not in opt: opt['cusp'] = [0.55, 0.7]
bpy.ops.wm.open_mainfile(filepath=blend)
b = bpy.data.objects['Redingote bodice']
M = {k[7:]: b[k] for k in b.keys() if k.startswith('bodice_')}
w, yc, lap = M['w'], M['yc'], M['lap']
sol = [m for m in b.modifiers if m.type == 'SOLIDIFY'][0]
th = sol.thickness
# The fold and the buttons (John: "deeper fold; the buttons should not be fighting the fold; the fold should leave a
# line long for buttons to hold in place"; "it should be right on the fold line and curved"): the fold line runs deep,
# to under the bust line, and it is no straight chord: it bows into the V and turns down to run INTO the front's
# closing edge, tangent to it. So the V closes well above the fold's foot, and from there down the front is one straight
# closed line, which the buttons hold; the top button sits just under where the V closes, on the line the fold becomes.
foot = opt.get('foot', M['z_bust'] + opt.get('foot_over_bust', -1.35))
belly = opt.get('belly', 0.0 if 'bow' in opt else 0.25)    # the cut line's swing out from the chord (a true curved cut)
bow = opt.get('bow', 0.0 if belly else None)           # (the earlier way: the cloth drawn along the surface; None: tangent to the edge)
# The lapel's outline, traced off John's photo of a camp-collar shirt (2026-11-21: "trace these lines for the lapel"), in
# the fold's own measure: (how far down the fold from the neck, how far out from it), both as shares of the fold's
# length. Read off both lapels of the photo: the top edge leaves the fold just under the neck and runs out to the
# lapel's point, 0.27 of the way down and 0.48 of the fold's length out; from the point the outer edge runs straight
# in to the foot of the fold at the top button. John's own outline drawn on our render puts the point 0.38 out, so
# 0.40 is used. The front is CUT to this outline (the cloth past the roll line is the outline mirrored over it), so
# the lapel is still the front's own cloth; where the bodice's corner is smaller than the outline, the front is cut
# wider there, as a tailor draws the lapel onto the front's pattern.
outline = opt.get('outline', None if 'bow' in opt else [[0.04, 0.0], [0.27, 0.40], [1.0, 0.0]])
if outline and not belly: belly = 1e-6
def bowdev(t):
    t = np.clip(t, 0, 1); return bow * 6.75 * t * t * (1 - t)
def ss01(a_, b_, x_):
    t_ = np.clip((x_ - a_) / (b_ - a_), 0, 1); return t_ * t_ * (3 - 2 * t_)
neck_x = opt.get('neck_x', 2.2)
t0 = opt.get('t0', 0.03)
r0 = opt.get('roll_radius', 0.24)
mw = np.array(b.matrix_world); inv = np.linalg.inv(mw)
sk = bpy.data.objects['Skin body']; sw = np.array(sk.matrix_world)
SV = np.array([v.co[:] for v in sk.data.vertices]) @ sw[:3, :3].T + sw[:3, 3]
SP = [tuple(p.vertices) for p in sk.data.polygons]

bm = bmesh.new(); bm.from_mesh(b.data)
lay = bm.verts.layers.int['front_layer']
def world():
    return np.array([v.co[:] for v in bm.verts]) @ mw[:3, :3].T + mw[:3, 3]
W = world(); L_ = np.array([v[lay] for v in bm.verts])
# 0. the fit, eased: the front's taut canvas meets the round of the shoulder on a line (flat on one side, curved on the
# other), which thick cloth shows as a ridge from the neck down to the bust. The upper chest's depth is faired across it.
bm.verts.ensure_lookup_table()
if opt.get('ease_chest', True):
    def ss(a_, b_, x_):
        t_ = np.clip((x_ - a_) / (b_ - a_), 0, 1); return t_ * t_ * (3 - 2 * t_)
    ax = np.abs(W[:, 0])
    wt = (W[:, 1] < yc) * ss(0.75, 1.1, ax) * (1 - ss(3.2, 3.9, ax)) * ss(M['z_bust'] + 0.2, M['z_bust'] + 1.2, W[:, 2])
    idx = np.nonzero(wt > 1e-3)[0]
    nbs = [[e.other_vert(bm.verts[int(i)]).index for e in bm.verts[int(i)].link_edges] for i in idx]
    Y = W[:, 1].copy(); y0 = Y.copy()
    for r_ in range(opt.get('ease_rounds', 150)):
        A = np.array([Y[n_].mean() for n_ in nbs]); Y[idx] += 0.6 * wt[idx] * (A - Y[idx])
    W[:, 1] = Y
    for v in bm.verts: v.co = tuple(W[v.index] @ inv[:3, :3].T + inv[:3, 3])
    print('foldfront: the upper chest eased across the canvas line: %d points, moved up to %.3f' % (len(idx), np.abs(Y - y0).max()))
# 1. the roll lines, marked on each front and cut into its mesh as a line of edges
sides = []
for layer, sg in ((1, -1.0), (2, 1.0)):               # (the right front's neck point is at x < 0; it laps to +x, on top)
    bd = np.array([v.index for v in bm.verts if v.is_boundary and v[lay] == layer and W[v.index, 1] < yc
                   and W[v.index, 2] > M['z_neck_front'] - 0.3 and abs(W[v.index, 0] - sg * neck_x) < 0.25])
    i = bd[np.argmin(np.abs(W[bd, 0] - sg * neck_x))]
    N = W[i][[0, 2]]; B = np.array([-sg * lap, foot]); ln = np.linalg.norm(B - N); u = (B - N) / ln
    n = np.array([-u[1], u[0]]); n = n if n[0] * -sg > 0 else -n       # toward the V
    sides.append(dict(layer=layer, sg=sg, N=N, B=B, u=u, n=n, ln=ln))
    if bow is None: bow = opt.get('tangent', 0.9) * abs(u[0] / u[1]) * ln / 6.75
    # the line as it is cut: straight, or (belly) a true curve that swings out from the chord, away from the V, so
    # that more of the front lies past it and the lapel is wider (John: "the lapel is too thin needs to be wider").
    # A curve is cut as K short straight cuts, each through its own stretch of the cloth.
    K = opt.get('cut_segments', 16) if belly else 1
    ts = np.linspace(0, 1, K + 1)
    Cp = N[None] + (ts * ln)[:, None] * u[None] - (belly * 6.75 * ts * ts * (1 - ts))[:, None] * n[None]
    sides[-1]['Cp'] = Cp
    for k in range(K):
        lo, hi = (-9.0 if k == 0 else ts[k]), (9.0 if k == K - 1 else ts[k + 1])
        geom = []
        for f in bm.faces:
            if any(v[lay] != layer for v in f.verts): continue
            if min(W[v.index, 2] for v in f.verts) <= foot - 0.4 or max(W[v.index, 1] for v in f.verts) >= yc: continue
            c = f.calc_center_median()
            tc = ((c.x - N[0]) * u[0] + (c.z - N[1]) * u[1]) / ln
            if lo <= tc < hi: geom.append(f)
        ge = list({e for f in geom for e in f.edges}); gv = list({v for f in geom for v in f.verts})
        dk = Cp[k + 1] - Cp[k]; nk = np.array([-dk[1], dk[0]]); nk = nk / np.linalg.norm(nk); nk = nk if nk @ n > 0 else -nk
        co = Vector(tuple(inv[:3, :3] @ np.array([Cp[k][0], 0, Cp[k][1]]) + inv[:3, 3])); no = Vector((nk[0], 0, nk[1]))
        r = bmesh.ops.bisect_plane(bm, geom=geom + ge + gv, dist=1e-5, plane_co=co, plane_no=no)
        for g in r['geom_cut']:
            if isinstance(g, bmesh.types.BMVert): g[lay] = layer
        bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table()
        W = world(); L_ = np.array([v[lay] for v in bm.verts])
    print('foldfront: front %d: roll line marked from the neck (%.2f, %.2f) to the break (%.2f, %.2f), %.1f cm' % (layer, N[0], N[1], B[0], B[1], 2.62 * ln))
bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table()
W = world(); L_ = np.array([v[lay] for v in bm.verts])
top, und = sides
zx = top['N'][1] + top['u'][1] * (0 - top['N'][0]) / top['u'][0]
# 2. what turns back: each front's own points past its line
q = W[:, [0, 2]]
from scipy.spatial import cKDTree
for s in sides:
    sd = (q - s['N']) @ s['n']; tl = ((q - s['N']) @ s['u']) / s['ln']
    if belly:                                         # distance past the CURVE, and where along it
        td = np.linspace(0, 1, 1200)
        Cd = s['N'][None] + (td * s['ln'])[:, None] * s['u'][None] - (belly * 6.75 * td * td * (1 - td))[:, None] * s['n'][None]
        tg = np.gradient(Cd, axis=0); nd = np.column_stack([-tg[:, 1], tg[:, 0]]); nd /= np.linalg.norm(nd, axis=1)[:, None]
        nd = np.where(((nd @ s['n']) > 0)[:, None], nd, -nd)
        _, jn = cKDTree(Cd).query(q)
        sdc = ((q - Cd[jn]) * nd[jn]).sum(1)
        mine = (L_ == s['layer']) & (W[:, 1] < yc) & (W[:, 2] > foot - 0.5)
        snap = mine & (np.abs(sdc) < opt.get('snap', 0.012)) & (tl > -0.02) & (tl < 1.02)
        q[snap] -= sdc[snap][:, None] * nd[jn][snap]; W[snap, 0], W[snap, 2] = q[snap, 0], q[snap, 1]; sdc[snap] = 0.0
        chord = tl.copy(); sd, tl = sdc, td[jn]
        f = mine & (sd > 1e-6) & (W[:, 2] > foot - 1e-3) & (chord > -0.15)
        s.update(sd=sd, tl=tl, fold=f, crease=snap, C=Cd[jn], nl=nd[jn], Cd=Cd, nd=nd, td=td)
        continue
    f = (L_ == s['layer']) & (W[:, 1] < yc) & (sd > 1e-6) & (W[:, 2] > foot - 1e-3) & (tl > -0.15)
    s.update(sd=sd, tl=tl, fold=f, crease=(L_ == s['layer']) & (W[:, 1] < yc) & (np.abs(sd) <= 1e-6) & (tl > -0.02) & (tl < 1.02))
if outline:
    from scipy.spatial import Delaunay
    OT = np.array(outline, float)
    rec = []
    for s in sides:
        ci = np.nonzero(s['crease'])[0]; ci = ci[np.argsort(s['tl'][ci])]
        rec.append(([bm.verts[int(i)] for i in ci], s['tl'][ci].copy(), W[ci, 1].copy()))
    ref_ny = np.median([f.normal.y for f in bm.faces if all(v[lay] == 1 for v in f.verts) and f.calc_center_median().y < yc - 1][:2000])
    bmesh.ops.delete(bm, geom=[bm.verts[int(i)] for i in np.nonzero(top['fold'] | und['fold'])[0]], context='VERTS')
    info = {}
    hm = opt.get('lapel_mesh', 0.07)
    for si, s in enumerate(sides):
        cv, ct, cy = rec[si]; Lc = s['ln']
        Dfun = lambda t_, Lc=Lc: np.interp(t_, OT[:, 0], OT[:, 1] * Lc, left=0.0, right=0.0)
        pts = [(OT[1, 0], OT[1, 1] * Lc)]
        for t_ in np.arange(OT[0, 0], 1.0 + 1e-9, hm / Lc):
            D = float(Dfun(t_)); n_ = int(np.ceil(D / hm))
            pts += [(t_, D * j / n_) for j in range(1, n_ + 1)]
        pts = np.array(pts)
        keep = np.nonzero((ct >= OT[0, 0] - 1e-6) & (ct <= 1 + 1e-6))[0]; nc = len(keep)
        allp = np.vstack([np.column_stack([ct[keep] * Lc, np.zeros(nc)]), np.column_stack([pts[:, 0] * Lc, pts[:, 1]])])
        tri = Delaunay(allp).simplices
        tt = np.concatenate([ct[keep], pts[:, 0]]); dd = np.concatenate([np.zeros(nc), pts[:, 1]])
        Cq = np.column_stack([np.interp(tt, s['td'], s['Cd'][:, k]) for k in (0, 1)])
        nq = np.column_stack([np.interp(tt, s['td'], s['nd'][:, k]) for k in (0, 1)])
        flat = Cq + dd[:, None] * nq                    # where the cloth lies as cut, before it is turned
        vs = [cv[int(i)] for i in keep]
        for k in range(nc, len(tt)):
            v = bm.verts.new(tuple(np.array([flat[k, 0], np.interp(tt[k], ct, cy), flat[k, 1]]) @ inv[:3, :3].T + inv[:3, 3]))
            v[lay] = s['layer']; vs.append(v); info[v] = (si, dd[k], tt[k])
        nf = 0
        for a_, b_, c_ in tri:
            if max(a_, b_, c_) < nc: continue
            e1, e2 = flat[b_] - flat[a_], flat[c_] - flat[a_]; ny = e1[1] * e2[0] - e1[0] * e2[1]
            if abs(ny) < 1e-9: continue
            order = (a_, b_, c_) if ny * ref_ny > 0 else (a_, c_, b_)
            try: bm.faces.new([vs[int(i)] for i in order]); nf += 1
            except ValueError: pass
        s['Dfun'] = Dfun
        print('foldfront: front %d: cut to the traced lapel: point %.2f of the way down the fold, %.1f cm out; %d points, %d faces of its own cloth'
              % (s['layer'], OT[1, 0], 2.62 * OT[1, 1] * Lc, len(tt) - nc, nf))
    # the seam between the lapel and the rest of its front must be whole. Where a point of the front's cut edge was not
    # counted as on the line (it sat a hair off it), the lapel was joined to its neighbours past it and a slit was left
    # open in the cloth: the dark ticks beside the top button. Every small hole along the line is closed.
    if opt.get('close_slits', 'bow' not in opt):
        cvs = set(v for r_ in rec for v in r_[0]); newv = set(info)
        be = [e for e in bm.edges if len(e.link_faces) == 1 and any(v in cvs for v in e.verts)
              and not all(v in newv for v in e.verts)]
        nf0 = len(bm.faces); bmesh.ops.holes_fill(bm, edges=be, sides=opt.get('slit_sides', 8))
        for f_ in list(bm.faces)[nf0:]:
            for v in f_.verts:
                if v not in info and v not in cvs: pass
        bmesh.ops.recalc_face_normals(bm, faces=[f_ for f_ in bm.faces if any(v in cvs for v in f_.verts)]) if opt.get('slit_recalc') else None
        print('foldfront: %d slits along the roll lines closed (%d open edges looked at)' % (len(bm.faces) - nf0, len(be)))
    bm.verts.index_update(); bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table(); bm.normal_update()
    W = world(); L_ = np.array([v[lay] for v in bm.verts]); q = W[:, [0, 2]]; nV = len(W)
    for si, s in enumerate(sides):
        cv, ct, cy = rec[si]
        sd, tl, f, cr = np.zeros(nV), np.zeros(nV), np.zeros(nV, bool), np.zeros(nV, bool)
        for v, t_ in zip(cv, ct): cr[v.index] = True; tl[v.index] = t_
        for v, (sj, d_, t_) in info.items():
            if sj == si: f[v.index] = True; sd[v.index] = d_; tl[v.index] = t_
        C = np.column_stack([np.interp(tl, s['td'], s['Cd'][:, k]) for k in (0, 1)])
        nl = np.column_stack([np.interp(tl, s['td'], s['nd'][:, k]) for k in (0, 1)])
        s.update(sd=sd, tl=tl, fold=f, crease=cr, C=C, nl=nl)
n0 = len(W)
def caster(skip, layers=None):
    """Rays from the front onto the cloth that stays (the faces with no point of `skip`; of these layers only) and the body."""
    polys = [tuple(v.index for v in f.verts) for f in bm.faces if not any(skip[v.index] for v in f.verts)
             and (layers is None or all(L_[v.index] in layers for v in f.verts))]
    tree = BVHTree.FromPolygons([Vector(p) for p in np.vstack([W, SV])], polys + ([tuple(i + n0 for i in p) for p in SP] if layers is None or 0 in layers else []))
    def cast(X):
        o = np.full(len(X), np.nan)
        for k, (x, z) in enumerate(X):
            h = tree.ray_cast(Vector((x, -40.0, z)), Vector((0, 1, 0)))
            if h[0] is not None: o[k] = h[0][1]
        return o
    return cast
# the line curved: the cloth by the line is drawn along its own surface so that the cut line becomes the curve
band_w = opt.get('bow_band', 1.0)
for s_ in (sides if bow else []):
    own = caster(np.zeros(len(W), bool), (s_['layer'],))
    sel = (L_ == s_['layer']) & (W[:, 1] < yc) & ~s_['fold'] & (s_['sd'] <= 1e-6) & (s_['sd'] > -band_w) & (s_['tl'] > -0.1) & (s_['tl'] < 1.1)
    jj = np.nonzero(sel)[0]
    dv = bowdev(s_['tl'][jj]) * (1 - ss01(0, band_w, -s_['sd'][jj]))
    nq = q[jj] + dv[:, None] * s_['n'][None]; ny = own(nq)
    s_['warp'] = (jj, nq, np.where(np.isfinite(ny), ny, W[jj, 1]))
for s_ in (sides if bow else []):
    jj, nq, ny = s_['warp']; W[jj, 0], W[jj, 2], W[jj, 1] = nq[:, 0], nq[:, 1], ny
out_side = opt.get('thick_out', 1.0) * th              # (how much of the cloth's thickness stands outside the mesh's own surface)
for s in (und, top):                                  # the under front first: the upper front's lapel then lies on what is there
    N, u, n, ln, sd, tl = s['N'], s['u'], s['n'], s['ln'], s['sd'], s['tl']
    # the under front's lapel lies on its own front (it runs on under the upper front, which covers its foot); the
    # upper front's on whatever is under it by then
    ground = caster(top['fold'] | und['fold'], (2, 0)) if s is und else caster(top['fold'])
    cover = caster(top['fold'], (1,)) if s is und else None
    ii = np.nonzero(s['fold'])[0]; ci = np.nonzero(s['crease'])[0]
    if not len(ii): continue
    d, t = sd[ii], np.clip(tl[ii], 0, 1)
    # the lapel's width at each height, read off the cloth itself
    bins = np.linspace(0, 1, 25); wb = np.array([d[(t >= lo) & (t <= hi)].max() if ((t >= lo) & (t <= hi)).any() else 0 for lo, hi in zip(bins[:-1], bins[1:])])
    wloc = np.maximum(np.interp(t, 0.5 * (bins[:-1] + bins[1:]), wb), 1e-3)
    if outline: wloc = np.maximum(s['Dfun'](t), 1e-3)
    # the roll: a half turn of radius rr at the break (open all the way down, tapering at the foot), then the lapel
    # bends back toward the chest on a wide arc so its edge lies on the front
    pr = (opt.get('press') if 'press' in opt else ({} if 'bow' in opt else {'left': 0.85})).get('left' if s['layer'] == 2 else 'right', 0.0)
    rr = r0 * np.clip((1 - t) / 0.12, 0.15, 1) * np.clip((t + 0.05) / 0.1, 0.4, 1)
    rr = rr * (1 - 0.7 * pr)                           # (pressed: the roll itself is ironed lower, so the lapel lies flat from the fold out)
    rr = np.minimum(rr, wloc / np.pi * 0.9 + 0.01)
    # the cusp (John: "soften the folding near the cusp, too sharp"): toward the foot the roll used to shrink to a
    # hard crease (its radius tapered, and capped by the lapel's width). Softened, the radius keeps a floor there: a
    # strip narrower than the half turn then only turns part of the way (its width over the radius), so the fold opens
    # out of the flat front like a cone instead of pinching shut.
    if opt.get('cusp'):
        c0, fl = opt['cusp']; rr = rr + ss01(c0, min(c0 + 0.12, 1.0), t) * (np.maximum(rr, r0 * fl * (1 - 0.7 * pr)) - rr)
        # and over the very last of it the radius opens right out, so the needle-thin end of the lapel is hardly turned
        # at all: folded hard over, cloth that thin and this thick made a spike and a dark notch at the foot
        if opt.get('cusp_end'): e0, Re = opt['cusp_end']; rr = rr + ss01(e0, 1.0, t) ** 2 * np.maximum(Re - rr, 0)      # (tried: it tore the front beside the button; off)
    arc = np.pi * rr; ang = np.minimum(d, arc) / rr
    ex = rr * np.sin(ang); hh = rr * (1 - np.cos(ang))
    rest = np.maximum(d - arc, 0)
    lie = th + opt.get('lie', 0.02)                    # the turned cloth's own surface over the front's: a thickness, and a hair
    Rc = np.maximum((wloc - arc) ** 2 / (4 * rr), 0.6)
    ph = rest / Rc; phm = np.arccos(np.clip(1 - np.maximum(2 * rr - lie, 0) / Rc, -1, 1))
    ex = ex - Rc * np.sin(np.minimum(ph, phm)) - np.maximum(ph - phm, 0) * Rc
    hh = np.where(rest > 0, np.maximum(2 * rr - Rc * (1 - np.cos(np.minimum(ph, phm))), np.minimum(lie, 2 * rr)), hh)
    P = q[ii]
    X = P - d[:, None] * n[None] + ex[:, None] * n[None]
    if belly: X = s['C'][ii] + ex[:, None] * s['nl'][ii]            # (turned over the curve: out from its own nearest point)
    X = X + (bowdev(t) * np.exp(-((ex - d) / opt.get('bow_over', 1.5)) ** 2))[:, None] * n[None]      # (the lapel goes with its curved line)
    # where the cloth past the line is only a sliver (the tail running down to the foot beside the top button, and the
    # first of it at the neck) it is not turned at all: a strip that thin cannot roll, and it showed as a ragged fringe.
    # The fold opens out of the flat front as the lapel gains width.
    g_ = np.ones(len(ii))
    if opt.get('tail', (0.12, 0.45)) and not outline:
        g_ = ss01(*opt.get('tail', (0.12, 0.45)), wloc)
    gy = ground(X)
    yc_t = np.interp(t, tl[ci][np.argsort(tl[ci])], W[ci, 1][np.argsort(tl[ci])]) if len(ci) > 1 else np.nanmedian(gy)
    gy = np.where(np.isfinite(gy), gy, yc_t)
    gy = np.where((gy > yc_t + opt.get('fell', 0.15)) & (d < arc + 0.2), yc_t, gy)       # (a ray that fell past the line into the V: the lapel lies on its own front, at the line's depth)
    if opt.get('cusp'):
        # at the cusp the two lapels cross and the tail of one lies over the roll of the other: read point by point
        # that ground is a row of steps (a ray hits the other lapel's edge, the next one misses it), and the tail took
        # the steps as creases. It is laid on a smooth cover drawn over them instead.
        from scipy.spatial import cKDTree as _KD
        kt = _KD(X); r1, r2 = opt.get('cusp_cover', (0.22, 0.3))
        env = np.array([gy[j].min() for j in kt.query_ball_point(X, r1)])
        gs = np.array([env[j].mean() for j in kt.query_ball_point(X, r2)])
        gy = gy + ss01(opt['cusp'][0] - 0.05, opt['cusp'][0] + 0.05, t) * (np.minimum(gs, gy) - gy)
    k = np.clip(d / (arc + 0.35), 0, 1); k = k * k * (3 - 2 * k)
    base = (1 - k) * yc_t + k * gy
    y = base - out_side * k - hh
    y = np.where(d > arc, np.minimum(y, gy - out_side - np.minimum(lie, 2 * rr)), y)
    # one smooth sheet: faired over its own mesh, never sunk into what it lies on
    nb = {int(i): [e.other_vert(bm.verts[int(i)]).index for e in bm.verts[int(i)].link_edges] for i in ii}
    Y = W[:, 1].copy(); Y[ii] = y; lim = y.copy(); wgt = np.clip(d / 0.08, 0, 1)
    for rep in range(opt.get('sheet', 6)):
        for r_ in range(opt.get('fair', 40)):
            A = np.array([Y[nb[int(i)]].mean() for i in ii]); Y[ii] += 0.7 * wgt * (A - Y[ii])
        Y[ii] = np.minimum(Y[ii], lim)
    if cover is not None:                             # under the upper front: pressed flat behind it
        cy = cover(X); under_ = np.isfinite(cy); Y[ii[under_]] = np.maximum(Y[ii[under_]], cy[under_] + opt.get('under_gap', 0.02 if 'bow' in opt else th + 0.02))
        print('foldfront: %d of the under lapel\'s points lie under the upper front, pressed behind it' % under_.sum())
    # pressed (John: "press the right lapel", the one on the right of the picture: her left): past the roll the lapel is
    # laid down onto the front, to the cloth's own thickness; the roll at the break is left as it is
    if pr:
        wp = pr * ss01(0.0, opt.get('press_over', 0.7), d - arc); Y[ii] = Y[ii] + wp * ((gy - out_side - lie) - Y[ii])
    if opt.get('cusp'):
        # the cloth's own reach: a point d from the line along the cloth cannot stand further off the line than d. Where
        # the lapel's tail was laid over the other lapel's roll at the cusp it was lifted further than that (edges
        # stretched to three times their length: the crumple beside the top button); held to its reach it lies quiet.
        reach = np.sqrt(np.maximum(d ** 2 - ex ** 2, 0)) * opt.get('reach', 1.0)
        over = (yc_t - Y[ii]) > reach; Y[ii] = np.maximum(Y[ii], yc_t - reach)
        print('foldfront: front %d: %d points were lifted past the cloth\'s own reach, held back' % (s['layer'], over.sum()))
        cz = ss01(opt['cusp'][0] - 0.05, opt['cusp'][0] + 0.05, t) * wgt          # (and the tail faired once more, so the hold leaves no kink)
        for r_ in range(opt.get('cusp_fair', 40)):
            A = np.array([Y[nb[int(i)]].mean() for i in ii]); Y[ii] += 0.6 * cz * (A - Y[ii])
        Y[ii] = np.maximum(Y[ii], yc_t - reach)
    if opt.get('dump'): np.savez(opt['dump'] + '.%d.npz' % s['layer'], t=t, d=d, ex=ex, X=X, Y=Y[ii], gy=gy, yc=yc_t, rr=rr, wloc=wloc, g=g_, C=s['C'][ii])
    far = d > arc + 0.3; gap = gy[far] - Y[ii][far]
    print('foldfront: front %d: the lapel stands %.2f off the front on average, %.2f at most (the cloth alone is %.2f)' % (s['layer'], gap.mean(), gap.max(), out_side + lie))
    new = np.column_stack([X[:, 0], Y[ii], X[:, 1]])
    W[ii] = (1 - g_)[:, None] * W[ii] + g_[:, None] * new
    # the fold's end. Turned hard over right down to its needle point, the cloth flipped face within a millimetre: a
    # spike and a dark notch at the foot, the worst strain on the front. The turn now runs out before the foot: the
    # cloth bends round the same radius through an angle that falls from the half turn to nothing (a true part turn:
    # a blend of flat and turned cloth passes through the line itself and crushes the cloth onto it), and the last of
    # the tail lies flat in the front's own plane.
    if opt.get('cusp') and opt.get('cusp_open'):          # (tried and left off: the part-turned tail stood out as a wing)
        a_, b_ = opt['cusp_open']; go = 1 - ss01(a_, b_, t)
        Th = np.pi * go; a1 = np.minimum(d, rr * Th) / rr; run = np.maximum(d - rr * Th, 0)
        exp_ = rr * np.sin(a1) + run * np.cos(Th); hp = rr * (1 - np.cos(a1)) + run * np.sin(Th)
        Xp = s['C'][ii] + exp_[:, None] * s['nl'][ii]; Yp = yc_t - hp
        back = (exp_ < d) & (Th > np.pi / 2); Yp = np.where(back, np.minimum(Yp, W[ii, 1] + 0 * Yp), Yp)
        wf = ss01(0.85, 1.0, go)[:, None]
        W[ii] = (1 - wf) * np.column_stack([Xp[:, 0], Yp, Xp[:, 1]]) + wf * W[ii]
        s['turn'] = (t.copy(), np.degrees(Th), d.copy())
    # the strain of the turned cloth: each edge's length as it lies against its length as cut (flat, on the roll line's
    # own measure). Turning cloth over a curved line and laying it on a round chest cannot keep every length.
    inset = s['fold'] | s['crease']
    flat2 = (s['C'] + s['sd'][:, None] * s['nl']) if 'C' in s else q        # (a straight line: the flat cut is the front view itself)
    ea = np.array([(e.verts[0].index, e.verts[1].index) for v_ in (bm.verts[int(i)] for i in ii) for e in v_.link_edges
                   if inset[e.verts[0].index] and inset[e.verts[1].index]])
    ea = np.unique(np.sort(ea, axis=1), axis=0)
    l0 = np.linalg.norm(flat2[ea[:, 0]] - flat2[ea[:, 1]], axis=1)
    # the cloth relaxed (John: "do the cloth relaxation"). The rules above place the lapel; they do not keep its lengths.
    # Here every edge of the turned cloth is drawn back toward its length as cut, all at once and over and over, the
    # way cloth settles: stretch is resisted hard, compression less (cloth gives by easing), the fold line is held, the
    # cloth may not sink into what it lies on, and a light hold on where it was placed keeps the traced shape.
    rx = opt.get('relax', None if 'bow' in opt else {})
    if rx is not None and 'C' in s:
        it_, anch, kc_, ks_ = rx.get('iters', 1500), rx.get('anchor', 0.0005), rx.get('comp', 0.7), rx.get('stretch', 1.0)
        l1_ = np.linalg.norm(W[ea[:, 0]] - W[ea[:, 1]], axis=1); okr = l0 > 1e-4
        print('foldfront: front %d before relaxing: strain mean |%.1f%%|, 95%% under |%.1f%%|' % (s['layer'], 100 * np.abs(l1_[okr] / l0[okr] - 1).mean(), 100 * np.percentile(np.abs(l1_[okr] / l0[okr] - 1), 95)))
        A_, B_ = ea[okr, 0], ea[okr, 1]; L0 = l0[okr]
        free = np.zeros(len(W)); free[ii] = 1.0
        if cover is not None: free[ii[under_]] = 0.0          # (the part hidden behind the upper front stays pressed there)
        P0 = W.copy(); Pr = W.copy()
        kk_ = np.clip(d / (arc + 0.35), 0, 1); kk_ = kk_ * kk_ * (3 - 2 * kk_); clear = (out_side + lie) * kk_
        mob = free[ii] > 0
        nbi = [np.array(nb[int(i)]) for i in ii]
        for r_ in range(it_):
            dv = Pr[B_] - Pr[A_]; ln_ = np.maximum(np.linalg.norm(dv, axis=1), 1e-9); Cn = ln_ - L0
            kq = np.where(Cn > 0, ks_, kc_); wa, wb = free[A_], free[B_]; ws = np.maximum(wa + wb, 1e-9)
            cor = (kq * Cn / ln_ / ws)[:, None] * dv
            dP = np.zeros_like(Pr); cnt = np.zeros(len(Pr))
            np.add.at(dP, A_, wa[:, None] * cor); np.add.at(dP, B_, -wb[:, None] * cor); np.add.at(cnt, A_, 1); np.add.at(cnt, B_, 1)
            Pr += rx.get('omega', 1.3) * dP / np.maximum(cnt, 1)[:, None]
            Pr[ii] += anch * free[ii][:, None] * (P0[ii] - Pr[ii])
            if r_ % 4 == 3:                                   # (bending: the move itself is kept smooth, so the cloth eases and does not crinkle)
                dl = Pr - P0; av = np.array([dl[j].mean(axis=0) for j in nbi]); Pr[ii] += rx.get('bend', 0.25) * free[ii][:, None] * (av - dl[ii])
            if r_ % 10 == 9 or r_ == it_ - 1:                 # (and it may not sink into what it lies on)
                g2 = ground(Pr[ii][:, [0, 2]]); okg = np.isfinite(g2) & mob & (g2 < yc_t + opt.get('fell', 0.15))
                Pr[ii[okg], 1] = np.minimum(Pr[ii[okg], 1], g2[okg] - clear[okg])
        mv = np.linalg.norm(Pr[ii] - P0[ii], axis=1)
        print('foldfront: front %d relaxed over %d rounds: its points moved %.1f mm on average, %.1f mm at most' % (s['layer'], it_, 26.2 * mv.mean(), 26.2 * mv.max()))
        W[:] = Pr
    l1 = np.linalg.norm(W[ea[:, 0]] - W[ea[:, 1]], axis=1)
    okl = l0 > 1e-4; st = l1[okl] / l0[okl] - 1
    tm = 0.5 * (s['tl'][ea[:, 0]] + s['tl'][ea[:, 1]])[okl]
    s['strain'] = dict(st=st, mid=0.5 * (W[ea[:, 0]] + W[ea[:, 1]])[okl], t=tm)
    cusp = tm > opt.get('cusp_from', 0.8)
    print('foldfront: front %d: strain over %d edges: mean |%.1f%%|, 95%% under |%.1f%%|, worst %+.1f%% / %+.1f%%; at the cusp (the last fifth of the fold) mean |%.1f%%|, worst %+.1f%% / %+.1f%%'
          % (s['layer'], len(st), 100 * np.abs(st).mean(), 100 * np.percentile(np.abs(st), 95), 100 * st.min(), 100 * st.max(),
             100 * np.abs(st[cusp]).mean(), 100 * st[cusp].min(), 100 * st[cusp].max()))
    s['lapel_pts'] = W[ii].copy(); s['lapel_t'] = t.copy()
    print('foldfront: front %d: %d of its own points turned back on the roll line (%d on the line); the lapel %.1f cm wide, the roll %.1f cm high'
          % (s['layer'], len(ii), len(ci), 2.62 * d.max(), 2.62 * 2 * rr.max()))
for v in bm.verts: v.co = tuple(W[v.index] @ inv[:3, :3].T + inv[:3, 3])
# the foot of each fold: where the roll line runs out into the front's edge the cut left faces with no area, facing the
# wrong way; given the cloth's thickness they stood out as a spike with a dark notch beside it. They are taken out.
if opt.get('clean_foot', 'bow' not in opt):
    ne0 = len(bm.faces)
    for s_ in sides:
        Bf = np.array([s_['Cd'][-1, 0], s_['Cd'][-1, 1]]) if 'Cd' in s_ else None
        if Bf is None: continue
        near = [e for e in bm.edges if all(np.hypot(W[v.index, 0] - Bf[0], W[v.index, 2] - Bf[1]) < opt.get('foot_reach', 0.35) and W[v.index, 1] < yc for v in e.verts)]
        bmesh.ops.dissolve_degenerate(bm, dist=opt.get('foot_dist', 0.012), edges=near)
    bm.verts.index_update(); bm.normal_update()
    print('foldfront: the feet of the folds cleaned: %d faces with no area taken out' % (ne0 - len(bm.faces)))
bm.to_mesh(b.data); bm.free()
for p in b.data.polygons: p.use_smooth = True
b.data.update()
# 3. the buttons, last: down the front below the break
if opt.get('buttons', True):
    rb = M['button'] * w; step = opt.get('button_step', M['button_step']) * w
    # the top button right at the fold's foot (John: "it should be right on the fold line"): just under where the two
    # curved folds meet at the middle
    tt = np.linspace(0, 1, 400); cx = top['N'][0] + tt * top['ln'] * top['u'][0] + bowdev(tt) * top['n'][0]
    cz = top['N'][1] + tt * top['ln'] * top['u'][1] + bowdev(tt) * top['n'][1]
    if belly: cx, cz = top['Cd'][:, 0], top['Cd'][:, 1]
    zcross = float(np.interp(0.0, cx * np.sign(cx[-1] - cx[0]), cz))
    z1 = zcross - rb - opt.get('break_over', 0.45); z0 = M['z_hem_front'] + 0.55 * w
    # John: "move down the top button and let more strain transfer": the top button sits clear below the foot of the
    # fold, so the fold runs out into the front over its whole length before anything holds the cloth
    if opt.get('button_under_foot', None if 'bow' in opt else 0.15) is not None:
        z1 = min(z1, foot - rb - opt.get('button_under_foot', 0.15))
    nbt = max(2, int(round((z1 - z0) / step)) + 1)
    Wb = np.array([v.co[:] for v in b.data.vertices]) @ mw[:3, :3].T + mw[:3, 3]
    fb = BVHTree.FromPolygons([Vector(p) for p in Wb], [tuple(p.vertices) for p in b.data.polygons])
    sb = bmesh.new(); bmesh.ops.create_uvsphere(sb, u_segments=16, v_segments=10, radius=1.0)
    bv = np.array([v.co[:] for v in sb.verts]) * np.array([1, M['button_dome'], 1]) * rb; bf = [[v.index for v in f.verts] for f in sb.faces]; sb.free()
    P_, F_ = [], []
    for z in np.linspace(z1, z0, nbt):
        h = fb.ray_cast(Vector((0.0, -40.0, float(z))), Vector((0, 1, 0)))
        if h[0] is None: continue
        base = len(P_); P_ += [tuple(p + np.array([0.0, h[0][1] - out_side - 0.05 * rb, z])) for p in bv]; F_ += [[i + base for i in f] for f in bf]
    # what the buttons impose (John: "measure properly the strain imposed by buttons"). A button holds both fronts
    # together at one point, flat. So at each button: how much fold is left at its height (it should be none: a button
    # set where the cloth is still turning fights the fold), how much turned cloth lies under its footprint, the gap
    # between the two fronts under it against the cloth's thickness, and the strain of the cloth around it.
    zb = np.linspace(z1, z0, nbt); Wf = np.array([v.co[:] for v in b.data.vertices]) @ mw[:3, :3].T + mw[:3, 3]
    Lb = np.array([d_.value for d_ in b.data.attributes['front_layer'].data])
    def surf(layer, z):
        P = [tuple(p.vertices) for p in b.data.polygons if all(Lb[i] == layer for i in p.vertices)]
        return BVHTree.FromPolygons([Vector(p) for p in Wf], P).ray_cast(Vector((0.0, -40.0, float(z))), Vector((0, 1, 0)))[0]
    rep = []
    for k_, z in enumerate(zb[:3] if belly else []):
        row = dict(z=float(z))
        for s_ in sides:
            tz = float(np.interp(z, s_['Cd'][::-1, 1], np.linspace(1, 0, len(s_['Cd'])))) if z > foot else 1.0
            tt_, th_, dd_ = s_.get('turn', (s_['lapel_t'], np.full(len(s_['lapel_t']), 180.0), np.ones(len(s_['lapel_t']))))
            near = np.abs(tt_ - tz) < 0.03
            row['turn%d' % s_['layer']] = float(th_[near].max()) if z > foot and near.any() else 0.0
            lp = s_['lapel_pts']; row['under%d' % s_['layer']] = int(((lp[:, 0] ** 2 + (lp[:, 2] - z) ** 2) < (1.15 * rb) ** 2).sum())
            m_ = s_['strain']['mid']; nr = (m_[:, 0] ** 2 + (m_[:, 2] - z) ** 2) < (2.5 * rb) ** 2
            row['st%d' % s_['layer']] = (float(np.abs(s_['strain']['st'][nr]).mean()), float(np.abs(s_['strain']['st'][nr]).max()), int(nr.sum())) if nr.any() else (0.0, 0.0, 0)
        h1, h2 = surf(1, z), surf(2, z); row['gap'] = float(h2[1] - h1[1]) if h1 is not None and h2 is not None else float('nan')
        rep.append(row)
        print('foldfront: button %d at z %.2f: fold left at its height %.0f / %.0f degrees (upper / under front); turned cloth under its footprint %d / %d points; '
              'strain of the cloth within 2.5 radii: upper mean |%.1f%%| worst |%.1f%%| (%d edges), under mean |%.1f%%| worst |%.1f%%| (%d edges); the fronts %.2f apart there (the cloth is %.2f)'
              % (k_ + 1, z, row['turn1'], row['turn2'], row['under1'], row['under2'], 100 * row['st1'][0], 100 * row['st1'][1], row['st1'][2],
                 100 * row['st2'][0], 100 * row['st2'][1], row['st2'][2], row['gap'], th))
    json.dump(rep, open(out + '.buttons.json', 'w'), indent=1)
    me = bpy.data.meshes.new('Redingote bodice buttons'); me.from_pydata(P_, [], F_); me.update()
    for p in me.polygons: p.use_smooth = True
    old = bpy.data.objects.get('Redingote bodice buttons')
    if old: bpy.data.objects.remove(old, do_unlink=True)
    o = bpy.data.objects.new('Redingote bodice buttons', me); bpy.context.scene.collection.objects.link(o)
    mat = bpy.data.materials.get('bodice button')
    if mat: me.materials.append(mat)
    if b.parent: o.parent = b.parent; o.matrix_parent_inverse = b.parent.matrix_world.inverted()
    print('foldfront: buttoned last: %d buttons from z %.2f (the folds meet at the middle at z %.2f, the break is at %.2f) down to z %.2f, %.1f cm apart' % (nbt, z1, zcross, foot, z0, 2.62 * (z1 - z0) / (nbt - 1)))
# the ground truth (John: "maintain this shape as ground truth"): the turned lapels' own points, kept, and any later
# build measured against them
if opt.get('save_truth'):
    np.savez(opt['save_truth'], **{'pts%d' % s_['layer']: s_['lapel_pts'] for s_ in sides}, **{'t%d' % s_['layer']: s_['lapel_t'] for s_ in sides})
    print('foldfront: ground truth saved to', opt['save_truth'])
if opt.get('truth'):
    T_ = np.load(opt['truth'])
    for s_ in sides:
        dd_, jj_ = cKDTree(T_['pts%d' % s_['layer']]).query(s_['lapel_pts']); body_ = s_['lapel_t'] < opt.get('cusp_from', 0.8)
        print('foldfront: front %d against the ground truth: above the cusp mean %.3f, worst %.3f (%.1f mm); at the cusp mean %.3f, worst %.3f (%.1f mm)'
              % (s_['layer'], dd_[body_].mean(), dd_[body_].max(), 26.2 * dd_[body_].max(), dd_[~body_].mean(), dd_[~body_].max(), 26.2 * dd_[~body_].max()))
np.savez(out + '.strain.npz', **{'st%d' % s_['layer']: s_['strain']['st'] for s_ in sides}, **{'mid%d' % s_['layer']: s_['strain']['mid'] for s_ in sides})
bpy.ops.wm.save_as_mainfile(filepath=out)
