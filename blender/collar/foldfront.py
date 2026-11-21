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
                         ease_chest=True, ease_rounds=150, bow_band=1.0, bow_over=1.5, lie=0.02, thick_out=1.0, t0=0.03),
}
if opt.get('recipe'):
    opt = dict(RECIPES[opt['recipe']], **{k: v for k, v in opt.items() if k != 'recipe'})
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
foot = opt.get('foot', M['z_bust'] + opt.get('foot_over_bust', -0.6))
belly = opt.get('belly', 0.0 if 'bow' in opt else 0.7)    # the cut line's swing out from the chord (a true curved cut)
bow = opt.get('bow', 0.0 if belly else None)           # (the earlier way: the cloth drawn along the surface; None: tangent to the edge)
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
        s.update(sd=sd, tl=tl, fold=f, crease=snap, C=Cd[jn], nl=nd[jn], Cd=Cd)
        continue
    f = (L_ == s['layer']) & (W[:, 1] < yc) & (sd > 1e-6) & (W[:, 2] > foot - 1e-3) & (tl > -0.15)
    s.update(sd=sd, tl=tl, fold=f, crease=(L_ == s['layer']) & (W[:, 1] < yc) & (np.abs(sd) <= 1e-6) & (tl > -0.02) & (tl < 1.02))
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
    # the roll: a half turn of radius rr at the break (open all the way down, tapering at the foot), then the lapel
    # bends back toward the chest on a wide arc so its edge lies on the front
    rr = r0 * np.clip((1 - t) / 0.12, 0.15, 1) * np.clip((t + 0.05) / 0.1, 0.4, 1)
    rr = np.minimum(rr, wloc / np.pi * 0.9 + 0.01)
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
    g_ = np.ones(len(ii))
    gy = ground(X)
    yc_t = np.interp(t, tl[ci][np.argsort(tl[ci])], W[ci, 1][np.argsort(tl[ci])]) if len(ci) > 1 else np.nanmedian(gy)
    gy = np.where(np.isfinite(gy), gy, yc_t)
    gy = np.where((gy > yc_t + opt.get('fell', 0.15)) & (d < arc + 0.2), yc_t, gy)       # (a ray that fell past the line into the V: the lapel lies on its own front, at the line's depth)
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
        cy = cover(X); under_ = np.isfinite(cy); Y[ii[under_]] = np.maximum(Y[ii[under_]], cy[under_] + opt.get('under_gap', 0.02))
        print('foldfront: %d of the under lapel\'s points lie under the upper front, pressed behind it' % under_.sum())
    new = np.column_stack([X[:, 0], Y[ii], X[:, 1]])
    W[ii] = (1 - g_)[:, None] * W[ii] + g_[:, None] * new
    print('foldfront: front %d: %d of its own points turned back on the roll line (%d on the line); the lapel %.1f cm wide, the roll %.1f cm high'
          % (s['layer'], len(ii), len(ci), 2.62 * d.max(), 2.62 * 2 * rr.max()))
for v in bm.verts: v.co = tuple(W[v.index] @ inv[:3, :3].T + inv[:3, 3])
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
    me = bpy.data.meshes.new('Redingote bodice buttons'); me.from_pydata(P_, [], F_); me.update()
    for p in me.polygons: p.use_smooth = True
    old = bpy.data.objects.get('Redingote bodice buttons')
    if old: bpy.data.objects.remove(old, do_unlink=True)
    o = bpy.data.objects.new('Redingote bodice buttons', me); bpy.context.scene.collection.objects.link(o)
    mat = bpy.data.materials.get('bodice button')
    if mat: me.materials.append(mat)
    if b.parent: o.parent = b.parent; o.matrix_parent_inverse = b.parent.matrix_world.inverted()
    print('foldfront: buttoned last: %d buttons from z %.2f (the folds meet at the middle at z %.2f, the break is at %.2f) down to z %.2f, %.1f cm apart' % (nbt, z1, zcross, foot, z0, 2.62 * (z1 - z0) / (nbt - 1)))
bpy.ops.wm.save_as_mainfile(filepath=out)
