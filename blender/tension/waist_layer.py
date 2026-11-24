"""Layer the waist as a tailor does: the skirt on the petticoat, the coat's basque over the skirt's band. EXPERIMENTAL.

    python3 blender/tension/waist_layer.py HABIT.blend OUT.blend ['{"gap": 0.03, "taper": 1.5}']

The habit as built stacks corset, petticoat, COAT, skirt (waist.py: the coat's inner face is 0.4 to 0.7 units inside the
skirt's outer at 97% of the rays), so the skirt's top edge is a hard line over the coat and wherever the basque's hem is
outside the skirt it comes through as a flap. The skirt also stands 0.3 off the petticoat at the waist (the close push
holds it 1.5 cm off the skin whatever lies under it) and the petticoat comes through it in front at 36 to 38.

Two passes, each on what the build already has:
    skirt   at the waist the skirt is fitted to the petticoat (a shrinkwrap on the Petticoat, offset 0.01, in a new
            vertex group 'push waist' that takes over from 'push' between z 35.5 and 38.5), not to the skin's 1.5 cm. The draft
            leaves the skirt 0.3 off the petticoat there (its hull allows the bulk and the wool), so the shrinkwrap runs
            in OUTSIDE_SURFACE mode: it brings the cloth in as well as out.
    coat    the basque's points are moved horizontally outward from the body's axis to lie `gap` outside the skirt's
            outer face, in full from the hem to the skirt's top edge and tapering to none over `taper` units above it;
            the displacement is evened over the mesh, and never less than the clearance needs. The buttons move with it.
"""
import bpy, sys, json, numpy as np
from mathutils import Vector, kdtree
sys.path.insert(0, __import__('os').path.dirname(__file__))
import waist as W

OPT = dict(wrap='OUTSIDE_SURFACE', gap=0.03, taper=1.5, z_hem=35.5, z_full=38.5, z_clear=34.0, freeze=1, z_seams=35.6, petti_off=0.01, smooth_iters=6, fit_skirt=1)
if len(sys.argv) > 3 and sys.argv[-1].startswith('{'):
    OPT.update(json.loads(sys.argv[-1]))
src, dst = [a for a in sys.argv[1:] if a.endswith('.blend')][:2]
bpy.ops.wm.open_mainfile(filepath=src)
ss = lambda t: np.clip(t, 0, 1) ** 2 * (3 - 2 * np.clip(t, 0, 1))
sk = bpy.data.objects[W.SKIRT]; coat = bpy.data.objects[W.COAT]


def fit_skirt():
    """The waist's skirt on the petticoat."""
    z0, z1 = OPT['z_hem'], OPT['z_full']
    vg = sk.vertex_groups.new(name='push waist')
    vp, vs = sk.vertex_groups['push'], sk.vertex_groups['push smooth']
    for v in sk.data.vertices:
        z = v.co.z
        if z <= z0: continue
        w = float(ss((z - z0) / (z1 - z0)))
        vg.add([v.index], w, 'REPLACE')
        vp.add([v.index], 1 - w, 'REPLACE')
        vs.add([v.index], 1.0, 'REPLACE')
    names = [m.name for m in sk.modifiers]

    def add(after, name):
        m = sk.modifiers.new(name, 'SHRINKWRAP')
        m.target, m.wrap_method, m.wrap_mode = bpy.data.objects[W.PETTI], 'NEAREST_SURFACEPOINT', OPT['wrap']
        m.offset, m.vertex_group = OPT['petti_off'], 'push waist'
        i = names.index(after) + 1
        with bpy.context.temp_override(object=sk, active_object=sk):
            bpy.ops.object.modifier_move_to_index(modifier=m.name, index=i)
        names.insert(i, name)
    add('push hips', 'waist fit')
    add('push hips again', 'waist fit again')
    # and last, whatever came before: the cloth is outside the petticoat (it hugs it only from the ramp up, but the
    # petticoat's tucks must not come through it below)
    vc = sk.vertex_groups.new(name='push clear')
    for v in sk.data.vertices:
        if v.co.z > OPT['z_clear']: vc.add([v.index], 1.0, 'REPLACE')
    m = sk.modifiers.new('waist clear', 'SHRINKWRAP')
    m.target, m.wrap_method, m.wrap_mode, m.offset, m.vertex_group = bpy.data.objects[W.PETTI], 'NEAREST_SURFACEPOINT', 'OUTSIDE', OPT['petti_off'], 'push clear'
    with bpy.context.temp_override(object=sk, active_object=sk):
        bpy.ops.object.modifier_move_to_index(modifier=m.name, index=names.index('push final') + 1)


def skirt_outer(bvh, x, y, z, zc):
    """The skirt's outer radius along the ray from the axis through (x, y), at height min(z, zc)."""
    r = np.hypot(x, y - W.YC); phi = np.arctan2(x, -(y - W.YC))
    q = W.radii(bvh, min(z, zc), phi)
    return (q[0] if q else None), r


def lift_coat(bvh_skirt, z_top):
    """Lift the basque over the band as one stack. The shift is a function of the direction and the height (a polar grid of
    rays: the outermost coat layer's inner face against the skirt's outer face), and every point at that direction and
    height moves by it, whatever its depth in the folds. (Pushing each point to the skirt on its own flattened the
    waist's tucks and the lap: their inner limbs lie 0.7 inside the surface and were thrown out to it, edges stretched
    5 times, the two fronts onto one surface.)"""
    from scipy.ndimage import gaussian_filter
    me = coat.data; n = len(me.vertices)
    P = np.array([v.co[:] for v in me.vertices])
    R = np.zeros(n * 3); me.attributes['rest'].data.foreach_get('vector', R); R = R.reshape(-1, 3)
    th = [m for m in coat.modifiers if m.type == 'SOLIDIFY'][0].thickness
    bc = W.solid_bvh(W.COAT)
    zg = np.arange(34.8, z_top + OPT['taper'] + 0.01, 0.2); pg = np.radians(np.arange(0, 360, 1.0))
    G = np.zeros((len(zg), len(pg))); ok = np.zeros(G.shape, bool)
    for i, z in enumerate(zg):
        for j, ph in enumerate(pg):
            c = W.radii(bc, z, ph); q = W.radii(bvh_skirt, min(z, z_top - 0.05), ph)
            if c is None or q is None: continue
            ok[i, j] = True
            G[i, j] = max(0.0, q[0] + OPT['gap'] - (c[0] - th)) * (1.0 - ss((z - z_top) / OPT['taper']))
    # (below the hem there is no coat to measure: the shift of the nearest row above it goes down, so the hem, a V in
    # front, takes its full shift and does not saw between the shifted and the unshifted)
    for i in range(len(zg) - 2, -1, -1):
        G[i] = np.where(ok[i], G[i], G[i + 1]); ok[i] = ok[i] | ok[i + 1]
    G0 = G.copy()
    G = gaussian_filter(G, sigma=(1.5, 2.0), mode=('nearest', 'wrap'))
    G = np.maximum(G, G0)

    def lookup(x, y, z):
        ph = np.degrees(np.arctan2(x, -(y - W.YC))) % 360.0
        fi = np.clip((z - zg[0]) / 0.2, 0, len(zg) - 1.001); i0 = int(fi); ti = fi - i0
        fj = ph; j0 = int(fj) % 360; j1 = (j0 + 1) % 360; tj = fj - int(fj)
        a = G[i0, j0] * (1 - tj) + G[i0, j1] * tj; b = G[i0 + 1, j0] * (1 - tj) + G[i0 + 1, j1] * tj
        return a * (1 - ti) + b * ti

    zone = np.where((P[:, 2] < zg[-1]) & (P[:, 2] > zg[0]))[0]
    d = np.zeros(n); u = np.zeros((n, 2))
    for i in zone:
        x, y, z = P[i]; r = np.hypot(x, y - W.YC)
        d[i] = lookup(x, y, z); u[i] = np.array([x, y - W.YC]) / r
    E = np.array([e.vertices[:] for e in me.edges])
    Q = P.copy(); Q[:, 0] += d * u[:, 0]; Q[:, 1] += d * u[:, 1]
    me.vertices.foreach_set('co', Q.ravel()); me.update()
    idx = zone
    kd = kdtree.KDTree(len(idx))
    for k, i in enumerate(idx): kd.insert(Vector(P[i]), k)
    kd.balance()
    for o in bpy.data.objects:
        if o.type == 'MESH' and o.name.startswith('Redingote bodice buttons'):
            for v in o.data.vertices:
                _, k, dist = kd.find(Vector(v.co))
                i = idx[k]; v.co.x += d[i] * u[i, 0]; v.co.y += d[i] * u[i, 1]
            o.data.update()
    e = np.linalg.norm(Q[E[:, 0]] - Q[E[:, 1]], axis=1) / np.maximum(np.linalg.norm(R[E[:, 0]] - R[E[:, 1]], axis=1), 1e-9)
    m = np.isin(E[:, 0], idx) & np.isin(E[:, 1], idx)
    print('coat lifted: %d points in the zone, shift mean %.3f max %.3f; edge stretch vs the draft in the zone: '
          '5/50/95/max pct %s' % (len(idx), d[idx].mean(), d.max(), np.round(np.percentile(e[m], [5, 50, 95, 100]), 3)))
    return d


if OPT['fit_skirt']:
    fit_skirt()
bpy.context.view_layer.update()
bv = W.solid_bvh(W.SKIRT)
z_top = max(v.co.z for v in bpy.data.objects[W.SKIRT].evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh().vertices if v.co.z < 45)
lift_coat(bv, z_top)
if OPT['freeze']:
    # the skirt with its extra shrinkwraps on the petticoat's 300 000 faces needs 14 GB to render: its evaluated mesh
    # goes in as plain geometry (the rig is at rest, so nothing is lost for a still)
    e = sk.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me2 = bpy.data.meshes.new_from_object(e)
    sk.modifiers.clear(); sk.vertex_groups.clear(); old = sk.data; sk.data = me2
    print('skirt frozen: %d points' % len(me2.vertices))
# the skirt's seam allowances and stitches stand where the skirt was drafted (0.3 off the fitted cloth at the band), and
# under the basque they would come through it as dark slits: what lies under the coat's hem is dropped from them
import bmesh
for nm in ('Riding skirt seams', 'Riding skirt stitches'):
    o = bpy.data.objects.get(nm)
    if not o: continue
    ev = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me3 = bpy.data.meshes.new_from_object(ev)
    bm = bmesh.new(); bm.from_mesh(me3)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z > OPT['z_seams']], context='VERTS')
    n0 = len(me3.vertices); bm.to_mesh(me3); bm.free()
    o.modifiers.clear(); o.vertex_groups.clear(); o.data = me3
    print('%s: %d points, %d kept (below z %.1f)' % (nm, n0, len(me3.vertices), OPT['z_seams']))
bpy.ops.wm.save_as_mainfile(filepath=dst)
print('saved', dst)
