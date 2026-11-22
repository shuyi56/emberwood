"""The upper lip's shading, fixed at its two causes (combine thread, 2026-11-21).

    python3 tools/face-creator/apply-shade-fix.py face-creator.html [main.js ...]   (in place; every anchor once)

1. The baked shut-in shade (the vertex colour's cavity term) measured how far each point lies below its neighbours
   two and seven grid steps away, with weights tuned on the 600-column grid. The nose's hybrid grid has 300 columns
   (each step twice as wide, so four times the offset on a curved surface) and re-lays the points over the nose, and
   the mouth's loops have their own ring spacing, so the same steps no longer meant the same distance: the upper lip
   got a dark trapezoid and the sides of the nose dark bands. The shade is now measured in world space, on discs of
   fixed radius round each point, area weighted, so it no longer depends on the mesh. The radii keep the old
   calibration: on the 600-column grid a step-2 neighbour sat about 0.035 units away and a step-7 one about 0.125
   (measured on the cheek and lip). Four neighbours at distance h give a normal offset of H h^2 / 2 (H the mean
   curvature); an area-weighted disc of radius r gives H r^2 / 8 (the mean of x^2 over a disc is r^2 / 4), so the
   same reading needs r = 2h: radii 0.07 and 0.25 units (1.9 and 6.8 mm at 27 mm a unit). Surfaces facing away
   (inside the lips, the far side of a nostril) are left out.
2. The key light's shadow had a penumbra about 0.55 mm wide (PCFSoftShadowMap ignores the radius: about 3 texels of a
   2048 map over 14 units). Penumbra width = gap x light size / light distance (Fernando 2005, PCSS). The figure's
   Blender key, which this creator's lights copy (SKIN_LAYERS.LIGHTS), is an area light 24 across at about 64 from the
   head: 0.375. The nose's shadow on the lip lies 3 to 5 mm below the wing, so its penumbra is 1.1 to 1.9 mm: PCF with
   a radius of 4 texels (2 x 4 x 0.0068 units = 1.5 mm).
"""
import sys

OLD_LOOP = """    var occ = 0;
    [[2, 70], [7, 22]].forEach(function (st) {
      var s = st[0], jm = Math.max(0, j - s), jp = Math.min(NR - 1, j + s), im = (i - s + NS) % NS, ip = (i + s) % NS;
      var ax = 0, ay = 0, az = 0;
      [jm * NS + i, jp * NS + i, j * NS + im, j * NS + ip].forEach(function (m) { if (BLK[m] && !BLK[k]) m = k; ax += pos[m * 3]; ay += pos[m * 3 + 1]; az += pos[m * 3 + 2]; });
      ax = ax / 4 - pos[k * 3]; ay = ay / 4 - pos[k * 3 + 1]; az = az / 4 - pos[k * 3 + 2];
      occ += st[1] * (ax * nrm[k * 3] + ay * nrm[k * 3 + 1] + az * nrm[k * 3 + 2]);
    });
    occ = clamp(occ, 0, 1);"""
NEW_LOOP = """    var occ = CAV[k];      // (world-space discs: see cavityWorld)"""

OLD_HEAD = """  var BLK = new Uint8Array(NR * NS);"""
NEW_HEAD = """  var BLK = new Uint8Array(NR * NS);
  var CAV = cavityWorld(pos, nrm, idx, [[0.07, 70], [0.25, 22]]);"""

OLD_MOUTH = """      var kk = MOUTH.slots[v], rr = v <= MOUTH.nh ? 0 : 1 + Math.floor((v - MOUTH.nh - 1) / nbm), tt = v <= MOUTH.nh ? v : (v - MOUTH.nh - 1) % nbm, occ = 0;
      [[2, 70], [7, 22]].forEach(function (st) {
        var s = st[0], ax = 0, ay = 0, az = 0;
        [MOUTH.vid(rr - s, tt), MOUTH.vid(rr + s, tt), MOUTH.vid(rr, tt - s), MOUTH.vid(rr, tt + s)].forEach(function (mm) { ax += pos[mm * 3]; ay += pos[mm * 3 + 1]; az += pos[mm * 3 + 2]; });
        ax = ax / 4 - pos[kk * 3]; ay = ay / 4 - pos[kk * 3 + 1]; az = az / 4 - pos[kk * 3 + 2];
        occ += st[1] * (ax * nrm[kk * 3] + ay * nrm[kk * 3 + 1] + az * nrm[kk * 3 + 2]);
      });"""
NEW_MOUTH = """      var kk = MOUTH.slots[v], occ = CAV[kk];      // (the same world-space discs as the grid's)"""

FUNC_ANCHOR = "function buildHeadGeometry() {"
FUNC = """/* the shut-in shade's cavity term, measured in world space (combine thread, 2026-11-21): for each point, the
   area-weighted mean of the surface within a disc of radius r round it, minus the point, along its normal, at each
   [radius, weight]; surfaces facing away are left out. It does not depend on how the mesh is laid out (the nose's
   hybrid grid, the mouth's loops), only on the shape. Radii 0.07 and 0.25 give the old 600-column grid's step-2 and
   step-7 readings (r = 2h). */
function cavityWorld(pos, nrm, idx, scales) {
  var n = pos.length / 3, area = new Float32Array(n), out = new Float32Array(n), q, a, b, c;
  for (q = 0; q < idx.length; q += 3) {
    a = idx[q]; b = idx[q + 1]; c = idx[q + 2];
    var ux = pos[b * 3] - pos[a * 3], uy = pos[b * 3 + 1] - pos[a * 3 + 1], uz = pos[b * 3 + 2] - pos[a * 3 + 2];
    var vx = pos[c * 3] - pos[a * 3], vy = pos[c * 3 + 1] - pos[a * 3 + 1], vz = pos[c * 3 + 2] - pos[a * 3 + 2];
    var ar = 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx) / 3;
    area[a] += ar; area[b] += ar; area[c] += ar;
  }
  var R = 0; scales.forEach(function (s) { R = Math.max(R, s[0]); });
  var cells = new Map(), key = function (x, y, z) { return (x * 73856093) ^ (y * 19349663) ^ (z * 83492791); };
  for (var k = 0; k < n; k++) {
    if (!area[k]) continue;
    var h = key(Math.floor(pos[k * 3] / R), Math.floor(pos[k * 3 + 1] / R), Math.floor(pos[k * 3 + 2] / R)), L = cells.get(h);
    if (!L) cells.set(h, L = []); L.push(k);
  }
  var S = scales.length, sw = new Float64Array(S), sx = new Float64Array(S), sy = new Float64Array(S), sz = new Float64Array(S);
  for (k = 0; k < n; k++) {
    if (!area[k]) continue;
    var px = pos[k * 3], py = pos[k * 3 + 1], pz = pos[k * 3 + 2], nx = nrm[k * 3], ny = nrm[k * 3 + 1], nz = nrm[k * 3 + 2];
    var cx = Math.floor(px / R), cy = Math.floor(py / R), cz = Math.floor(pz / R), e;
    for (e = 0; e < S; e++) { sw[e] = sx[e] = sy[e] = sz[e] = 0; }
    for (var dx = -1; dx <= 1; dx++) for (var dy = -1; dy <= 1; dy++) for (var dz = -1; dz <= 1; dz++) {
      var C = cells.get(key(cx + dx, cy + dy, cz + dz)); if (!C) continue;
      for (var t = 0; t < C.length; t++) {
        var m = C[t];
        if (nrm[m * 3] * nx + nrm[m * 3 + 1] * ny + nrm[m * 3 + 2] * nz <= 0) continue;
        var ox = pos[m * 3] - px, oy = pos[m * 3 + 1] - py, oz = pos[m * 3 + 2] - pz, d2 = ox * ox + oy * oy + oz * oz, w = area[m];
        for (e = 0; e < S; e++) if (d2 <= scales[e][0] * scales[e][0]) { sw[e] += w; sx[e] += w * ox; sy[e] += w * oy; sz[e] += w * oz; }
      }
    }
    var occ = 0;
    for (e = 0; e < S; e++) if (sw[e] > 0) occ += scales[e][1] * (sx[e] * nx + sy[e] * ny + sz[e] * nz) / sw[e];
    out[k] = clamp(occ, 0, 1);
  }
  return out;
}
""" + FUNC_ANCHOR

SHADOW = [("renderer.shadowMap.type = THREE.PCFSoftShadowMap;",
           "renderer.shadowMap.type = THREE.PCFShadowMap;      // (PCF: the radius below sets the penumbra; PCFSoft ignores it)"),
          ("key.shadow.radius = 3;",
           "key.shadow.radius = 4;      // (penumbra 2 x 4 texels = 1.5 mm: the figure's key, 24 across at 64, is 0.375 of its distance across; the nose's shadow lies 3-5 mm below the wing (Fernando 2005, PCSS))")]

for f in sys.argv[1:]:
    s = open(f).read()
    for a, b in [(OLD_LOOP, NEW_LOOP), (OLD_HEAD, NEW_HEAD), (OLD_MOUTH, NEW_MOUTH), (FUNC_ANCHOR, FUNC)] + SHADOW:
        c = s.count(a)
        if c == 0 and b.split('\n')[0] in s and a != FUNC_ANCHOR:
            continue      # (already applied)
        assert c == 1, (f, c, a[:80])
        s = s.replace(a, b)
    open(f, 'w').write(s)
    print('ok', f)
