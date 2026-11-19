"""The boot's pattern, laid flat: each piece last.py cut on the last (last.PIECES), peeled off it and flattened, as a
bootmaker tapes a last, draws the seams on the tape, cuts along them and presses the pieces flat on the hide.

    flatten()   one piece's flat shape, at its true size (angle-based unwrapping, scaled to the piece's own area),
                turned so its grain (the piece's 'up' on the boot) runs up the sheet, the grain side facing.
    sheet()     every piece flat on one sheet, as an image: its outline, each seam edge in its letter's colour
                (an edge is sewn to the edge of the same letter on another piece), the grain's arrow, its size, and
                how much the leather has to give to take the last's shape (flat lengths against lengths on the last).
    pairs()     each seam's length on either piece, to check that the edges to be sewn together agree.

The laps are in the pieces as cut: the quarters' front edges, the outer quarter's back edge and the shaft's back and
lower edges each reach last.RATIOS['lap'] past the seam's line. No stitching is done here."""
import bpy
import numpy as np

CM = 165.0 / 63.0            # a unit in centimetres (a 165 cm figure is 63 units tall)
SEAMS = {'A': ('vamp to quarter', (196, 60, 50)), 'B': ('heel back seam', (40, 110, 190)),
         'C': ('ankle seam', (30, 140, 80)), 'D': ('shaft back seam', (150, 70, 170)),
         'E': ('upper to sole', (210, 130, 20)), 'F': ('binding', (90, 90, 90)),
         'G': ('toe cap to vamp', (20, 150, 160))}


def flatten(piece):
    """The piece flat: its points [n, 2] in units. Returns (points, the flat lengths of its edges over their lengths
    on the last, as the 5th, 50th and 95th percentiles)."""
    V, F = np.asarray(piece['V'], dtype=float), piece['F']
    if 'flat' in piece:                                 # (cut flat already: the sole)
        uv = V[:, :2] * np.array([1.0, -1.0])
        return _upright(uv, V, piece), (1.0, 1.0, 1.0)
    me = bpy.data.meshes.new('flat')
    me.from_pydata(V.tolist(), [], F)
    o = bpy.data.objects.new('flat', me)
    bpy.context.scene.collection.objects.link(o)
    for ob in bpy.context.view_layer.objects:
        ob.select_set(False)
    bpy.context.view_layer.objects.active = o
    o.select_set(True)
    me.uv_layers.new(name='flat')
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.0)
    bpy.ops.object.mode_set(mode='OBJECT')
    uv = np.zeros((len(V), 2))
    lay = me.uv_layers['flat'].data
    for lp in me.loops:
        uv[lp.vertex_index] = lay[lp.index].uv
    area3 = sum(p.area for p in me.polygons)
    area2, signed = 0.0, 0.0
    for p in me.polygons:
        q = uv[list(p.vertices)]
        a = 0.5 * float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]))
        area2 += abs(a)
        signed += a
    uv *= np.sqrt(area3 / max(area2, 1e-12))
    if signed < 0:                                      # (the grain side facing)
        uv[:, 0] *= -1.0
    e = np.array([ed.vertices[:] for ed in me.edges])
    l3 = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    l2 = np.linalg.norm(uv[e[:, 0]] - uv[e[:, 1]], axis=1)
    give = tuple(float(x) for x in np.percentile(l2[l3 > 1e-6] / l3[l3 > 1e-6], [5, 50, 95]))
    bpy.data.objects.remove(o)
    bpy.data.meshes.remove(me)
    return _upright(uv, V, piece), give


def _upright(uv, V, piece):
    """Turned so the piece's 'up' on the boot runs up the sheet."""
    up = np.asarray(piece.get('up', (0, 0, 1)), dtype=float)
    h = V @ (up / np.linalg.norm(up))
    A = np.concatenate([uv - uv.mean(0), np.ones((len(uv), 1))], 1)
    g = np.linalg.lstsq(A, h - h.mean(), rcond=None)[0][:2]
    g /= max(np.linalg.norm(g), 1e-9)
    R = np.array([[g[1], -g[0]], [g[0], g[1]]])         # (g to +y)
    out = (uv - uv.mean(0)) @ R.T
    return out - out.min(0)


def _outline(F):
    """The piece's cut edges: the edges only one face has, as loops of point indices."""
    count = {}
    for f in F:
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            count[(min(a, b), max(a, b))] = count.get((min(a, b), max(a, b)), 0) + 1
    nxt = {}
    for (a, b), c in count.items():
        if c == 1:
            nxt.setdefault(a, []).append(b)
            nxt.setdefault(b, []).append(a)
    loops, seen = [], set()
    for s in nxt:
        if s in seen:
            continue
        loop, prev, cur = [s], None, s
        seen.add(s)
        while True:
            step = [v for v in nxt[cur] if v != prev and v not in seen]
            if not step:
                break
            prev, cur = cur, step[0]
            seen.add(cur)
            loop.append(cur)
        loops.append(loop)
    return loops


def _seam_runs(piece, loop):
    """The outline's edges that lie on a seam: (letter and side, [(a, b), ...])."""
    out = {}
    for k in range(len(loop)):
        a, b = loop[k], loop[(k + 1) % len(loop)]
        for lab, on in piece['edges'].items():
            if on[a] and on[b]:
                out.setdefault(lab, []).append((a, b))
                break
    return out


def pairs(pieces, log=print):
    """Each seam's length on each piece that has it, in cm."""
    tot = {}
    for p in pieces:
        V = np.asarray(p['V'])
        if 'flat' in p:
            ring = list(p['flat'])
            tot.setdefault('E', {})[p['name']] = float(sum(np.linalg.norm(V[ring[k]] - V[ring[(k + 1) % len(ring)]])
                                                           for k in range(len(ring)))) * CM
            continue
        for loop in _outline(p['F']):
            for lab, es in _seam_runs(p, loop).items():
                d = tot.setdefault(lab, {})
                d[p['name']] = d.get(p['name'], 0.0) + float(sum(np.linalg.norm(V[a] - V[b]) for a, b in es)) * CM
    for lab in sorted(tot):
        log('boot pattern: seam %s (%s): %s' % (lab, SEAMS[lab[0]][0], ', '.join('%s %.1f cm' % kv for kv in tot[lab].items())))
    return tot


def sheet(pieces, path, px=22.0, title='Boot pattern', log=print):
    """Every piece flat on one sheet, saved as an image at px pixels to the centimetre."""
    from PIL import Image, ImageDraw, ImageFont
    flats = []
    for p in pieces:
        uv, give = flatten(p)
        flats.append((p, uv * CM, give))
        log('boot pattern: %s, %.1f by %.1f cm flat; its flat lengths are %.0f%% to %.0f%% of those on the last'
            % (p['name'], uv[:, 0].max() * CM, uv[:, 1].max() * CM, 100 * give[0], 100 * give[2]))
    pad, head = 3.0, 4.0                                # (cm)
    width = max(60.0, max(f[1][:, 0].max() for f in flats) + 2 * pad)
    x, y, row, spots = pad, head + pad, 0.0, []
    for p, uv, give in sorted(flats, key=lambda f: -f[1][:, 1].max()):
        w, h = uv[:, 0].max(), uv[:, 1].max()
        if x + w + pad > width and x > pad:
            x, y, row = pad, y + row + pad + 2.5, 0.0
        spots.append((p, uv, give, x, y))
        x, row = x + w + pad, max(row, h)
    height = y + row + pad + 7.2
    im = Image.new('RGB', (int(width * px), int(height * px)), (247, 243, 234))
    dr = ImageDraw.Draw(im)
    f_big, f_mid, f_small = (ImageFont.load_default(size=s) for s in (int(1.3 * px), int(0.85 * px), int(0.7 * px)))
    dr.text((pad * px, 1.0 * px), title, fill=(40, 30, 20), font=f_big)
    for p, uv, give, x0, y0 in spots:
        h = uv[:, 1].max()

        def at(i):
            return ((x0 + uv[i, 0]) * px, (y0 + h - uv[i, 1]) * px)
        loops = [list(p['flat'])] if 'flat' in p else _outline(p['F'])
        for loop in loops:
            dr.polygon([at(i) for i in loop], fill=(214, 180, 140), outline=(70, 50, 35))
            dr.line([at(i) for i in loop] + [at(loop[0])], fill=(70, 50, 35), width=3)
            runs = {'E': [(loop[k], loop[(k + 1) % len(loop)]) for k in range(len(loop))]} if 'flat' in p else _seam_runs(p, loop)
            for lab, es in runs.items():
                colr = SEAMS[lab[0]][1]
                for a, b in es:
                    dr.line([at(a), at(b)], fill=colr, width=7)
                mid = np.mean([at(a) for a, _ in es], 0)
                cen = np.mean([at(i) for i in loop], 0)
                pos = mid + (cen - mid) / max(np.linalg.norm(cen - mid), 1e-6) * 1.1 * px
                dr.text((pos[0] - 0.3 * px, pos[1] - 0.5 * px), lab[0], fill=colr, font=f_mid)
        cx, w = x0 + 0.5 * uv[:, 0].max(), uv[:, 0].max()
        # the grain's arrow
        if h > 5.0:
            ax, a0, a1 = cx * px, (y0 + 0.68 * h) * px, (y0 + 0.32 * h) * px
            dr.line([(ax, a0), (ax, a1)], fill=(70, 50, 35), width=2)
            dr.polygon([(ax, a1 - 0.5 * px), (ax - 0.3 * px, a1 + 0.3 * px), (ax + 0.3 * px, a1 + 0.3 * px)], fill=(70, 50, 35))
        dr.text((x0 * px, (y0 + h + 0.4) * px), '%s  %.0f x %.0f cm' % (p['name'], w, h), fill=(40, 30, 20), font=f_mid)
        if 'flat' not in p:
            dr.text((x0 * px, (y0 + h + 1.5) * px), 'gives %.0f to %.0f%%' % (100 * give[0], 100 * give[2]),
                    fill=(110, 95, 80), font=f_small)
    # the key and the scale
    ky = (height - 5.8) * px
    kx = pad * px
    for lab in sorted(SEAMS):
        nm, colr = SEAMS[lab]
        if kx + (len(nm) * 0.42 + 4.5) * px > width * px:
            kx, ky = pad * px, ky + 1.1 * px
        dr.line([(kx, ky + 0.5 * px), (kx + 1.6 * px, ky + 0.5 * px)], fill=colr, width=7)
        dr.text((kx + 2.0 * px, ky), '%s  %s' % (lab, nm), fill=(40, 30, 20), font=f_small)
        kx += (len(nm) * 0.42 + 4.5) * px
    sy = (height - 2.2) * px
    dr.line([(pad * px, sy), ((pad + 10) * px, sy)], fill=(40, 30, 20), width=4)
    for k in range(11):
        dr.line([((pad + k) * px, sy - 0.25 * px), ((pad + k) * px, sy + 0.25 * px)], fill=(40, 30, 20), width=2)
    dr.text(((pad + 10.6) * px, sy - 0.45 * px), '10 cm.  Same letter, same seam: those edges are sewn together. Laps of 1 cm are '
            'included. Arrows show the grain.', fill=(40, 30, 20), font=f_small)
    im.save(path)
    log('boot pattern: sheet saved, %d by %d' % im.size)
    return path
