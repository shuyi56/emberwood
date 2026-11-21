"""The mannequin in the heel study's heels, for the walk: the heel parts ride the foot's bones and the foot is held
pitched onto the heel (female.HEEL_PITCH). Use it with views.py --walk N; the bare foot is blender/barefoot."""
import os
import sys

os.environ['HEEL_PARTS'] = '1'
os.environ['HEEL_WALK'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'heel'))
import heel  # noqa: E402


def build():
    # (set on every build: a render worker that keeps its scene loaded resets the environment between jobs)
    os.environ['HEEL_PARTS'] = '1'
    os.environ['HEEL_WALK'] = '1'
    return heel.build()
