"""The Face Creator's Ranger as ground truth for the groom: its silhouette and its flow lines, traced from the
creator's own hair (the export of face-creator.html: the cap, a grid of points with alpha, and its strand clumps),
drawn from the same cameras as render.py and laid over the groom's render.

    python3 tools/hair/creator_truth.py HEAD.json.gz CREATOR_STYLE.json.gz GROOM_DIR OUT_DIR [--children 24000]

Writes truth-<view>.png (the groom's render, the creator's outline in red, its flow lines in blue, magenta where the
creator has hair and the groom does not, cyan where the groom has hair outside the creator's) and truth.json (per
view: overlap, the area the groom misses and the area it adds, in cm2 on the image plane, and the outline's mean and
95th-percentile distance in mm).
"""
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render as RD  # noqa: E402
import scalp_field as SF  # noqa: E402

VIEWS = (0, 45, 90, 135, 180)
CAP_ALPHA = 0.5                    # a cap cell counts as hair where its alpha is at least this (the creator's cut-out)


def load(path):
    d = json.load(gzip.open(path))
    c = d['cap']
    P = np.array(c['p'], float).reshape(c['rows'], c['cols'], 3)
    A = np.array(c['a'], float).reshape(c['rows'], c['cols'])
    strands = [dict(p=np.array(s['p'], float).reshape(-1, 3), w=np.array(s['w'], float)) for s in d['strands']]
    return P, A, strands


def creator_points(P, A, strands, sub=10):
    """Points over the creator's hair: each cap quad that passes the alpha, subdivided, and each strand clump across its
    width (the hair as solid surfaces, for the depth test against the head and body)."""
    rows, cols = A.shape
    f = (np.arange(sub) + 0.5) / sub
    Q = []
    for r in range(rows - 1):
        c = np.arange(cols); c1 = (c + 1) % cols
        a = (A[r, c] + A[r, c1] + A[r + 1, c] + A[r + 1, c1]) / 4
        ok = a >= CAP_ALPHA
        p00, p01, p10, p11 = P[r, c][ok], P[r, c1][ok], P[r + 1, c][ok], P[r + 1, c1][ok]
        for u in f:
            for v in f:
                Q.append(p00 * (1 - u) * (1 - v) + p01 * u * (1 - v) + p10 * (1 - u) * v + p11 * u * v)
    for s in strands:
        p = s['p']; n = len(p)
        w = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(s['w'])), s['w'])
        T = np.gradient(p, axis=0); T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
        out = p - SF.CENTRE; out[:, 2] = 0
        B = np.cross(T, out); B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
        # (resampled to about 1 mm along)
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        t = np.arange(0, L[-1], 1.0 / SF.HH_MM)
        pi = np.stack([np.interp(t, L, p[:, q]) for q in range(3)], 1)
        Bi = np.stack([np.interp(t, L, B[:, q]) for q in range(3)], 1)
        wi = np.interp(t, L, w)
        for k in np.linspace(-1, 1, 7):
            Q.append(pi + Bi * (wi * k)[:, None])
    return np.concatenate(Q)


def creator_mask(cv, Q, zb):
    """The creator's hair as seen from cv: its points depth-tested against the head and body, splatted, closed."""
    from scipy import ndimage
    i, j, z, ok = cv.project(Q)
    vis = ok.copy(); vis[ok] = z[ok] < zb[j[ok], i[ok]] + 0.004
    m = np.zeros((cv.H, cv.W), bool)
    m[j[vis], i[vis]] = True
    m = ndimage.binary_closing(m, iterations=3)
    return m


def flow_lines(cv, P, A, strands, zb, every=3):
    """The creator's flow, where it is seen: every few of its strands, broken where the head or body hides them."""
    L = []
    def add(X):
        i, j, z, ok = cv.project(X)
        vis = ok.copy(); vis[ok] = z[ok] < zb[j[ok], i[ok]] + 0.004
        uv = np.c_[(X @ cv.right - cv.box[0]) / cv.px, (cv.box[3] - X @ cv.up) / cv.px]
        run = []
        for q, v in zip(uv, vis):
            if v:
                run.append(tuple(q))
            elif len(run) > 1:
                L.append(run); run = []
            else:
                run = []
        if len(run) > 1:
            L.append(run)
    for s in strands[::every]:
        add(s['p'])
    return L


def outline(mask):
    m = mask
    e = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    return e


def main():
    a = sys.argv[1:]
    head, style, gdir, out = a[:4]
    nch = int(a[a.index('--children') + 1]) if '--children' in a else 24000
    os.makedirs(out, exist_ok=True)
    from PIL import Image, ImageDraw
    from scipy.spatial import cKDTree
    RD.HEAD_CLOUD = SF.load_head(head)
    surf = SF.Surface(RD.HEAD_CLOUD)
    F = SF.build(surf)
    body = json.load(gzip.open(head))['bodies']['mannequin']
    Gm = dict(np.load(os.path.join(gdir, 'groom.npz')))
    paths, share = RD.grow_children(F, Gm, nch)
    total = float(Gm['n_hairs'].sum())
    P, A, strands = load(style)
    Q = creator_points(P, A, strands)
    rep = {}
    box = (-0.75, 0.75, -1.6, 1.15)
    for az in VIEWS:
        cv = RD.Canvas(700, az, box)
        bg, zb = RD.clay_backdrop(cv, surf, body)
        img, cov = RD.render_hair(cv, paths, 'full', bg, zb, total * share, return_cov=True)
        ours = cov > 0.5
        truth = creator_mask(cv, Q, zb)
        mm2 = (cv.px * SF.HH_MM) ** 2 / 100.0           # cm2 per pixel
        miss = truth & ~ours; extra = ours & ~truth
        eo, et = outline(ours), outline(truth)
        d = cKDTree(np.argwhere(eo)).query(np.argwhere(et))[0] * cv.px * SF.HH_MM if eo.any() and et.any() else np.array([0])
        rep['az%d' % az] = dict(iou=round(float((truth & ours).sum() / max(1, (truth | ours).sum())), 3),
                                missed_cm2=round(float(miss.sum() * mm2), 1), added_cm2=round(float(extra.sum() * mm2), 1),
                                outline_mm_mean=round(float(d.mean()), 1), outline_mm_p95=round(float(np.percentile(d, 95)), 1))
        rgb = RD.to_srgb(img)
        rgb[miss] = rgb[miss] * 0.5 + np.array([1.0, 0.0, 1.0]) * 0.5
        rgb[extra] = rgb[extra] * 0.5 + np.array([0.0, 0.9, 0.9]) * 0.5
        rgb[et] = [0.9, 0.05, 0.05]
        im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(im)
        for l in flow_lines(cv, P, A, strands, zb):
            dr.line(l, fill=(40, 90, 230), width=1)
        im.save(os.path.join(out, 'truth-az%d.png' % az))
    ims = [Image.open(os.path.join(out, 'truth-az%d.png' % az)) for az in VIEWS]
    W = sum(i.width for i in ims); sheet = Image.new('RGB', (W, ims[0].height), 'white'); x = 0
    for i in ims:
        sheet.paste(i, (x, 0)); x += i.width
    sheet.save(os.path.join(out, 'truth-sheet.png'))
    json.dump(rep, open(os.path.join(out, 'truth.json'), 'w'), indent=1)
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
