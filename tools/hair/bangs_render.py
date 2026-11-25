"""The bangs as fibres, in clay: ponytail_render.py's renderer and cameras, with the bangs from bangs_flow.py in place
of the field's old fringe. Nothing else changes: the ponytail's lines come from its cache.

    python3 tools/hair/bangs_render.py HEAD.json.gz BANGS_DIR OUT_DIR --field-cache CACHE [--size 750] [--before RENDER5]

BANGS_DIR is a bangs_flow.py output folder (bangs-lines.npz). Clay only (first renders are clay, project rule).
--before puts the old render's clay close-ups beside the new ones on the same cameras."""
import os, sys, json, gzip
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalp_field as SF
import ponytail_flow as PF
import ponytail_render as PR
import render as RD
import bangs_flow as BF


def main():
    from PIL import Image
    a = sys.argv[1:]
    head, bdir, out = a[0], a[1], a[2]
    size = int(a[a.index('--size') + 1]) if '--size' in a else 750
    os.makedirs(out, exist_ok=True)
    BF.reuse_field_cache(a[a.index('--field-cache') + 1])
    PF.HEAD = head
    z = np.load(os.path.join(bdir, 'bangs-lines.npz'))
    mine = {int(r): l for r, l in zip(z['roots'], z['lines'])}
    PF.bang_line = lambda B, i, k=32: mine[int(i)]
    paths, hairs, basin, B = PR.fibres(head, per_root=1)
    RD.HEAD_CLOUD = SF.load_head(head)
    RD.SCALP_F = B['F']
    surf = B['surf']
    body = json.load(gzip.open(head))['bodies']['mannequin']
    row = []
    for az in (0, 45, 90, 135, 180):
        cv = RD.Canvas(size, az)
        bg, zb = RD.clay_backdrop(cv, surf, body, 'clay')
        row.append((RD.to_srgb(RD.render_hair(cv, paths, 'clay', bg, zb, hairs)) * 255).astype(np.uint8))
    Image.fromarray(np.concatenate(row, 1)).save(os.path.join(out, 'turn-clay.png'))
    shots = (('closeup-front', 0, (-0.5, 0.5, 0.25, 1.1)), ('closeup-34', 40, (-0.5, 0.5, 0.25, 1.1)),
             ('closeup-side', 90, (-0.55, 0.75, 0.1, 1.12)))
    for name, az, box in shots:
        cv = RD.Canvas(size, az, box)
        bg, zb = RD.clay_backdrop(cv, surf, body, 'clay')
        im = (RD.to_srgb(RD.render_hair(cv, paths, 'clay', bg, zb, hairs)) * 255).astype(np.uint8)
        Image.fromarray(im).save(os.path.join(out, name + '.png'))
        if '--before' in a:
            p = os.path.join(a[a.index('--before') + 1], name + '.png')
            if os.path.exists(p):
                old = np.asarray(Image.open(p).convert('RGB'))
                old = old[:, :old.shape[1] // 2]                      # the clay half
                if old.shape[0] == im.shape[0]:
                    Image.fromarray(np.concatenate([old, im], 1)).save(os.path.join(out, name + '-before-after.png'))
    json.dump(dict(fibres=len(paths), hairs=round(float(hairs.sum())), bang_fibres=int((basin == 1).sum())),
              open(os.path.join(out, 'render.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
