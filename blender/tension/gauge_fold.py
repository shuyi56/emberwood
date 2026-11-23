"""The sharpest dihedral angle (degrees) between neighbouring triangles of a set: blender/zipboot/gauge.py's fold
gauge, reduced to one number for a patch."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'zipboot'))
import gauge as G  # noqa: E402


def max_fold(X, T):
    return float(G.fold(X, T).max()) if len(T) else 0.0
