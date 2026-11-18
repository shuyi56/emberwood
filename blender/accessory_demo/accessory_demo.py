"""A stand for checking accessories (blender/shared/accessories.py): the base female figure in a plain scoop-necked
top, trousers and flat shoes, wearing one of two sets of jewellery. Meant to be built as a mannequin (no face or
hair; the Face Creator supplies those in the game), so it builds in seconds:

    python3 blender/base/views.py --figure accessory_demo --mannequin --turn --angles front --out DIR
    python3 blender/base/views.py --figure accessory_demo --mannequin --set "LOOK='B'" --head --out DIR

LOOK  'A': a gold chain at princess length with a garnet drop pendant, garnet drop earrings, a cameo brooch.
      'B': a velvet choker with a cameo charm, a short pearl strand at the base of the neck, gold hoops.
CHECK True prints accessories.clearance_report() after the build: how deep any piece sinks into the clothes in
      each sheet column (the idle and the walk).
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'base'))
import female  # noqa: E402
from female import *  # noqa: E402,F401,F403
from female import _profile  # noqa: E402
import accessories as acc  # noqa: E402  (blender/shared, on the path once female is imported)

LOOK = 'A'
CHECK = True

M_TOP = mat_def(30, 'top', ramp(0x5a6e7a, 0.1), 0.05, surface='twill', closeup='cloth')
M_TROUSERS = mat_def(31, 'trousers', ramp(0x3e3832, 0.1), 0.05, surface='twill', closeup='cloth')
M_SHOE = mat_def(32, 'shoes', ramp(0x4a3022, 0.1), 0.2, surface='leather', closeup='leather')

LEG_R = [(2.0, 3.0), (-1.0, 2.9), (-6.0, 2.42), (-THIGH + 1.2, 1.95), (-THIGH, 1.9), (-THIGH - 1.5, 1.88),
         (-THIGH - 4.0, 1.9), (-THIGH - SHIN, 1.3)]


def top_edge(phi):
    """The top's neckline: across the shoulders at the sides and back, scooped to under the collarbones in front."""
    c = max(0.0, math.cos(phi))
    return 50.7 - 2.6 * c ** 2.2


def build():
    root, body, pelvis, chest = begin_figure('AccessoryDemo')
    torso_mesh('Torso', M_SKIN, body)

    # the top: a layer over the torso from its neckline to the hips (rows run 0 at the neckline to 1 at the hem)
    n_rows = 60

    def top_pt(v, phi):
        z = lerp(top_edge(phi), 27.0, v)
        x, y = torso_point(z, phi, 0.05)
        return x, y, z - v
    mesh_obj('Top', loft_fn([i / (n_rows - 1) for i in range(n_rows)], top_pt, n=96, cap_top=False, cap_bot=False),
             M_TOP, body, solid=0.06)

    build_head(body)
    build_rig(body)

    # ---- the jewellery, on what is built so far
    if LOOK == 'A':
        acc.necklace('Necklace', body, length=acc.LENGTHS['princess'], pendant='drop')
        acc.earrings('Earring', body, style='drop')
        acc.brooch('Brooch', body, x=-3.2, z=46.6, size=0.8)
    else:
        acc.choker('Choker', body, z=52.5, charm=True)
        acc.necklace('Pearls', body, length=18.0, style='beads', mid=acc.M_PEARL)
        acc.earrings('Earring', body, style='hoop')

    def sleeve(side, s, x, w):
        bare_arm(side, s, x, w)
    build_arms(chest, sleeve)

    def leg(side, s, x, w):
        zs = [2.0 - 0.2 * i for i in range(int((2.0 + THIGH + SHIN) / 0.2) + 1)]

        def leg_pt(z, phi):
            r = _profile(LEG_R, z) + 0.15
            return -r * math.sin(phi), -r * 1.04 * math.cos(phi)
        mesh_obj(f'Leg.{side}', loft_fn(zs, leg_pt, n=40), M_TROUSERS, loc=(x, 0, HIP_Z), skin=w)
        mesh_obj(f'Shoe.{side}', loft([(-0.6, 1.5, 1.6, 0, -0.4), (-1.6, 1.9, 2.9, 0, -1.4), (-2.9, 1.95, 3.3, 0, -1.6),
                                       (-ANKLE_Z, 1.85, 3.2, 0, -1.6)], n=32), M_SHOE,
                 loc=(x, 0, ANKLE_Z), skin=w)
    build_legs(pelvis, leg)

    res = finish_figure(acc.follow)
    if CHECK:
        acc.clearance_report()
    return res


if __name__ == '__main__':
    run(build, 'accessory_demo', HERE)
