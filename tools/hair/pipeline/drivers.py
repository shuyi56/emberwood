"""Small drivers for tools that have no main of their own for a stage (run inside bind.py, so the style is bound)."""
import gzip
import json
import os

import numpy as np


def loose_field(head, out, n=8000, seed=11, body='mannequin'):
    """photo_waves.field2 as the photo-waves thread runs it (hair-rebuild/wavy-photo/field/field2_driver.py, field
    v14): the scalp, the colliders of the bare mannequin (or the character's body), the field, its gauges and the
    drawings. Writes out/field.npz (lines nan-padded, basin, release, hairs, sec, kind), gauges.json, flow-field.png
    and flow-field-face.png."""
    import scalp_field as SF
    import photo_waves as PW
    import wavy_flow as WF
    import groom as GR
    os.makedirs(out, exist_ok=True)
    pts = SF.load_head(head)
    surf = SF.Surface(pts)
    F = SF.build(surf)
    bodies = json.load(gzip.open(head))['bodies']
    B = bodies['mannequin'] if body == 'mannequin' else GR.character_body(bodies['warden'], bodies['mannequin'])
    coll, depth = WF.colliders(surf, B)
    Fd = PW.field2(F, surf, depth, int(n), seed=int(seed))
    json.dump(Fd['gauges'], open(os.path.join(out, 'gauges.json'), 'w'), indent=1)
    m = max(len(l) for l in Fd['line'])
    L = np.full((len(Fd['line']), m, 3), np.nan)
    for i, l in enumerate(Fd['line']):
        L[i, :len(l)] = l
    np.savez_compressed(os.path.join(out, 'field.npz'), lines=L, basin=Fd['basin'], release=Fd['release'],
                        hairs=Fd['hairs'], sec=Fd['sec'], kind=Fd['kind'], root=Fd['root'])
    # the course before the wave: the same field with the wave's amplitude at 0 (nothing else reads it, and the roots
    # and the course take no random draw from the wave), node for node, for the wave gauge's base
    w0 = dict(PW.WAVE)
    PW.WAVE = dict(w0, amp_hh=0.0)
    Fb = PW.field2(F, surf, depth, int(n), seed=int(seed), log=lambda t: None)
    PW.WAVE = w0
    Lb = np.full(L.shape, np.nan)
    for i, l in enumerate(Fb['line']):
        Lb[i, :len(l)] = l
    np.savez_compressed(os.path.join(out, 'field-course.npz'), lines=Lb)
    bt, bz, bR = PW.body_radius(depth, nth=160, zs=np.arange(0.18, -1.0, -0.012))
    T, Z = np.meshgrid(bt, bz, indexing='ij')
    bp = np.stack([SF.AXIS[0] + bR * np.sin(T), SF.AXIS[1] - bR * np.cos(T), Z], -1).reshape(-1, 3)
    PW.draw2(Fd, np.r_[pts[pts[:, 2] > 0.15], bp], depth, os.path.join(out, 'flow-field.png'))
    PW.draw2(Fd, np.r_[pts[pts[:, 2] > 0.15], bp[bp[:, 2] > -0.6]], depth, os.path.join(out, 'flow-field-face.png'),
             views=((0, 4, 'front'), (35, 6, 'three quarter'), (70, 4, 'side on')), every=3, px=700)
