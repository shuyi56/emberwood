"""The mannequin's bare feet: the measured foot (blender/sorceress_pendant/foot.py) flat on the floor, framed by the
heel study's --head view. The heel gauge still runs and prints its barefoot validation."""
import os
import sys

os.environ['HEEL_PARTS'] = '0'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'heel'))
import heel  # noqa: E402


def build():
    # (set on every build: a render worker that keeps its scene loaded resets the environment between jobs)
    os.environ['HEEL_PARTS'] = '0'
    return heel.build()
