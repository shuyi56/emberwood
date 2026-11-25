"""The waist's layers, measured: how the corset, petticoat, riding skirt and coat (the basque) stack at the waist.

    python3 blender/tension/waist.py HABIT.blend [OUT.json]

Rays go in at each height and angle round the body's axis; each solid (as the modifiers evaluate it) gives its outer and
inner radius there. From these:

    skirt - petticoat   the skirt's inner face less the petticoat's outer: the gap (must be >= 0, or the skirt is through it)
    coat - skirt        the coat's inner face less the skirt's outer face over the waist: the basque lies OVER the
                        skirt's band only when this is >= 0 (negative: the coat is wedged under the skirt's top edge and
                        the band shows over it, the hard line round the waist; where the coat's hem is still outside, it
                        pokes out through the skirt there, the flap)
    inside              the share of the basque's points (z in the band's reach) that lie inside the skirt's cloth

The skirt's top height and the coat's hem height are read from the meshes. Units are the scene's (1 = 2.62 cm)."""
import bpy, sys, json, numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

YC = -0.3                                  # the body's axis in y at the waist (x = 0)
COAT, SKIRT, PETTI, CORSET, SKIN = 'Redingote bodice', 'Riding skirt', 'Petticoat', 'Corset', 'Skin body'


def solid_bvh(name):
    dg = bpy.context.evaluated_depsgraph_get()
    o = bpy.data.objects[name]; e = o.evaluated_get(dg); m = e.to_mesh()
    M = o.matrix_world
    bm = BVHTree.FromPolygons([M @ v.co for v in m.vertices], [tuple(p.vertices) for p in m.polygons])
    return bm


def radii(bvh, z, phi):
    """(outer, inner) radius of the solid at height z along angle phi (0 = front, toward -y, turning toward +x), or None."""
    u = np.array([np.sin(phi), -np.cos(phi)])
    o = Vector((u[0] * 25, YC + u[1] * 25, z)); d = Vector((-u[0], -u[1], 0))
    hits = []
    pos = o
    for _ in range(12):
        h = bvh.ray_cast(pos, d)
        if h[0] is None: break
        s_ = h[0].x * u[0] + (h[0].y - YC) * u[1]
        if s_ <= 0: break                  # (past the axis: the far wall)
        hits.append(float(s_))
        pos = h[0] + d * 1e-4
    return (hits[0], hits[1] if len(hits) > 1 else hits[0]) if hits else None      # (the top layer's outer face and its back face)


def measure(path, zs=None, phis=None):
    bpy.ops.wm.open_mainfile(filepath=path)
    names = [COAT, SKIRT, PETTI, CORSET, SKIN]
    bv = {n: solid_bvh(n) for n in names}
    sk = bpy.data.objects[SKIRT].evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
    z_top = max(v.co.z for v in sk.vertices if v.co.z < 45)
    zs = zs if zs is not None else np.arange(35.5, 41.6, 0.5)
    phis = phis if phis is not None else np.radians(np.arange(0, 360, 30))
    R = {n: np.full((len(zs), len(phis), 2), np.nan) for n in names}
    for i, z in enumerate(zs):
        for j, p in enumerate(phis):
            for n in names:
                r = radii(bv[n], z, p)
                if r: R[n][i, j] = r
    return dict(z=zs, phi=phis, R=R, z_top=z_top)


def report(m):
    zs, phis, R, z_top = m['z'], m['phi'], m['R'], m['z_top']
    c_in, s_out, s_in, p_out = R[COAT][..., 1], R[SKIRT][..., 0], R[SKIRT][..., 1], R[PETTI][..., 0]
    # (hits sorted by distance: a solid's outer face is the first, its back face the second; deeper hits are layers under it)
    print('skirt top z %.2f' % z_top)
    print('  z     coat-skirt min / mean   skirt-petticoat min / mean   (radial, units; coat-skirt >= 0 is basque over band)')
    for i, z in enumerate(zs):
        a, b = c_in[i] - s_out[i], s_in[i] - p_out[i]
        print('%5.1f   %7.3f / %7.3f        %7.3f / %7.3f' % (z, np.nanmin(a) if np.isfinite(a).any() else np.nan,
              np.nanmean(a) if np.isfinite(a).any() else np.nan, np.nanmin(b) if np.isfinite(b).any() else np.nan,
              np.nanmean(b) if np.isfinite(b).any() else np.nan))
    for lab, k in (('front', 0), ('side L', 3), ('back', 6), ('side R', 9)):
        print(' %s (phi %d): coat in / skirt out / skirt in / petticoat out, z=%s' % (lab, np.degrees(phis[k]), ' '.join('%.1f' % z for z in zs[::2])))
        for nm, arr in (('coat in', c_in), ('skirt out', s_out), ('skirt in', s_in), ('petti out', p_out)):
            print('   %-9s ' % nm + ' '.join('%6.2f' % x for x in arr[::2, k]))
    cover = np.where(zs <= z_top + 0.05)[0]
    a = (c_in - s_out)[cover]
    print('over the band (z <= top): coat-skirt < 0 at %.0f%% of rays, worst %.3f' % (100 * np.mean(a[np.isfinite(a)] < 0), np.nanmin(a)))


if __name__ == '__main__':
    m = measure(sys.argv[-2] if sys.argv[-1].endswith('.json') else sys.argv[-1])
    report(m)
    if sys.argv[-1].endswith('.json'):
        json.dump({k: (v.tolist() if hasattr(v, 'tolist') else {n: a.tolist() for n, a in v.items()} if isinstance(v, dict) else v)
                   for k, v in m.items()}, open(sys.argv[-1], 'w'))
