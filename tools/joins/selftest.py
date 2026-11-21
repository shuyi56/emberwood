"""Known shapes with known answers, so a change to the measurement can be checked in a second (numpy only):

    python3 tools/joins/selftest.py

  cylinder into a sphere cap, one mesh: G1 0 (tangent), kA 0, kB 1/R: a G1-not-G2 join, G2 deviation 1
  cylinder into a cone, one mesh: a 20 deg crease
  cylinder (r 1) into a sphere (R 5), two meshes: the crease angle 90 - asin(1/5) = 78.46 deg, kA 0, kB 1/R
  the profile function on a circle: k = 1/R everywhere, no steps"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'blender', 'shared'))
import continuity as C  # noqa: E402

CM = C.MM_PER_U / 10


def grid(fn, nu, nv):
    us = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    V = np.array([fn(u, v) for v in np.linspace(0, 1, nv) for u in us])
    F = []
    for j in range(nv - 1):
        for i in range(nu):
            a, b = j * nu + i, j * nu + (i + 1) % nu
            F += [(a, b, b + nu), (a, b + nu, a + nu)]
    return V, np.array(F)


def ring(r, z, n=64):
    return np.array([(r * math.cos(u), r * math.sin(u), z) for u in np.linspace(0, 2 * np.pi, n, endpoint=False)])


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {label}: {got:.4f} (want {want:.4f} +- {tol})")
    return ok


def main():
    good = True

    def cap(u, v):
        z = v * 8 - 4
        if z <= 0:
            return (2 * math.cos(u), 2 * math.sin(u), z)
        a = min(z / 2, 1) * math.pi / 2 * 0.9
        return (2 * math.cos(a) * math.cos(u), 2 * math.cos(a) * math.sin(u), 2 * math.sin(a))
    rep, rows = C.check_join('cap', grid(cap, 96, 321), ring(2, 0), stations=8)
    good &= check('cylinder-cap G1 deg', rep['summary']['g1_deg_max'], 0.0, 0.3)
    good &= check('cylinder-cap |kB| 1/cm', abs(rows[0]['kB']), 1 / (2 * CM), 0.02)
    good &= check('cylinder-cap G2 deviation', rep['summary']['g2_dev_max'], 1.0, 0.05)

    t20 = math.tan(math.radians(20))
    rep, _ = C.check_join('cone', grid(lambda u, v: ((2 if v <= .5 else 2 - t20 * (v * 8 - 4)) * math.cos(u),
                                                       (2 if v <= .5 else 2 - t20 * (v * 8 - 4)) * math.sin(u),
                                                       v * 8 - 4), 96, 321), ring(2, 0), stations=8)
    good &= check('cylinder-cone G1 deg', rep['summary']['g1_deg_max'], 20.0, 0.2)

    cyl = grid(lambda u, v: (math.cos(u), math.sin(u), v * 4 - 1), 96, 161)
    sph = grid(lambda u, v: (5 * math.sin(v * math.pi * .6) * math.cos(u), 5 * math.sin(v * math.pi * .6) * math.sin(u),
                             6 - 5 * math.cos(v * math.pi * .6)), 192, 241)
    seam = C.intersection_seam(cyl, sph)
    rep, rows = C.check_join('cyl-sphere', cyl, seam, B=sph, stations=8)
    good &= check('cylinder-into-sphere G1 deg', rep['summary']['g1_deg_max'], 90 - math.degrees(math.asin(0.2)), 0.2)
    good &= check('cylinder-into-sphere |kB| 1/cm', abs(rows[0]['kB']), 1 / (5 * CM), 0.005)

    th = np.linspace(0, math.pi, 400)
    prof = C.polyline_curvature(np.column_stack([3 * np.cos(th), 3 * np.sin(th)]))
    k = np.abs(prof['k'][20:-20])
    good &= check('circle profile k 1/cm', float(k.mean()), 1 / (3 * CM), 0.002)
    _, steps, _ = C.curvature_steps(prof)
    good &= check('circle profile steps', len(steps), 0, 0)
    print('selftest:', 'passed' if good else 'FAILED')
    return 0 if good else 1


if __name__ == '__main__':
    sys.exit(main())
