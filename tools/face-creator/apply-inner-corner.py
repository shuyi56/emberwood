"""The inner corner of the eye: the lid apposed to the ball down to the canthus (combine thread, 2026-11-21).

    python3 tools/face-creator/apply-inner-corner.py face-creator.html main.js orbit.js   (in place; every anchor once)

From a 3/4 view the near eye showed dark red flesh between the ball and the lids at the inner corner. Measured on the
Ranger: over the inner 4.5 mm of the opening the lid's lash line sat about 1 mm behind the corneal apex while its wet
edge sat 11 mm behind it, so the margin there was a 10 mm deep tunnel standing off the ball, and the medial skin
showed through it (in shade, read as flesh). The cause is the clamp that holds the lid no deeper than the face's plane
plus a sulcus (orbitSkinField): at the inner corner the face's plane (the side of the nose) stands far forward, so the
lid was held there instead of on the ball.

Anatomy: the lids are well apposed to the globe (StatPearls, Lateral Canthoplasty), and the medial canthus, where the
two margins meet, is separated from the eyeball only by the lacrimal lake, a small triangular space holding the
caruncle (Elsevier, Medial angle of eye). So toward the inner corner the lid now lies on the ball (its bed, less
its thickness) down to the canthus, with the clamp eased out on the same weight the sulcus already uses for the
corner (the medial half of the opening, near the line out of the corner). The skin between the face and the margin
then slopes down into the canthus as a hollow, and the caruncle fills the lake.
Mirror in blender/shared/orbit.py (sulcus, L).
"""
import sys

A = "    L[k] = -osmax(-L[k], -(Hh[k] + sul), 4.0 * e);"
B = """    var Lb = L[k], wmed = osm(0.5 * S.a, -0.5 * S.a, X[k] - S.cx - S.xc) * (1 - osm(0.0, 0.85, along[k]));
    L[k] = -osmax(-L[k], -(Hh[k] + sul), 4.0 * e);
    L[k] += (Lb - L[k]) * wmed;      // (the inner corner: the lid apposed to the ball down to the canthus, the lacrimal lake small: apply-inner-corner.py)"""
for f in sys.argv[1:]:
    s = open(f).read()
    if B in s:
        print('already', f); continue
    assert s.count(A) == 1, (f, s.count(A))
    open(f, 'w').write(s.replace(A, B)); print('ok', f)
