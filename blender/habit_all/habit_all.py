"""The whole riding habit (ca. 1885-1895) on one mannequin: every piece built by its own module, layered in the order it is
worn, over one body.

    python3 blender/base/views.py --figure habit_all --mannequin --turn --clay-only --out DIR

Each piece is a standalone figure (blender/<piece>/<piece>.py with a build() that makes the body, then its garment).
Here the body is built once (sorceress_pendant.build is made to return the same result on every later call) and each
piece's build() then only adds its garment. Pieces share file names (draft.py, pattern.py, sew.py are in several folders),
so every piece is imported and run with its own copies of those modules.

The order, innermost first: the hose and boots, the chemise, the corset (its measures draw the body in, the petticoat is
drafted on that), the petticoat, the riding skirt, the bodice, the sleeves. Settings: PIECES (a list of the names below, to
build some), and each piece module's own settings through SETTINGS ({piece: {NAME: value}}), by default the layers
over others shown as drafted (John's rule: a layer is first shown without the solver, on the layers below as they are)."""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.realpath(os.path.join(HERE, '..'))
for d in ('sorceress_pendant', 'shared', 'base'):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault('HEAD_GLB', '')
os.environ.setdefault('NO_DRESS', '1')
os.environ.setdefault('NO_ACC', '1')

# (the piece, the module's folder, the folders it needs on its path: innermost first)
ORDER = [
    ('boot', 'boot', ['boot', 'hose', 'vest']),
    ('chemise', 'chemise', ['chemise', 'vest']),
    ('petticoat', 'petticoat', ['petticoat']),
    ('skirt', 'skirt', ['skirt']),
    ('redingote_bodice', 'redingote_bodice', ['redingote_bodice', 'redingote', 'vest', 'chemise', 'doublet']),
    ('sleeves', 'sleeves', ['sleeves', 'redingote', 'chemise', 'vest']),
]
PIECES = [x for x in os.environ.get('HABIT_PIECES', '').split(',') if x] or None      # None: all of ORDER
SETTINGS = {
    'skirt': {'SETTLE': 0},      # (shown as drafted over the petticoat: the solver comes after this is seen)
}
# The chemise is the loosest layer and nothing yet draws it in under the bodice and the sleeves (they fit over the corset
# and the body, not over it), so it bursts out through both. It is built, so the layers over it can be fitted to it, but
# not shown until they are: HABIT_SHOW=chemise shows it.
HIDDEN = [x for x in os.environ.get('HABIT_HIDE', 'chemise').split(',') if x and x not in os.environ.get('HABIT_SHOW', '').split(',')]
SHARED = ('sorceress_pendant', 'shared', 'base')
PIECE_DIRS = tuple(os.path.join(ROOT, p) + os.sep for p in {d for _, _, ds in ORDER for d in ds} | {'corset', 'gown'})


def _own(m):
    f = getattr(m, '__file__', None)
    return bool(f) and os.path.realpath(f).startswith(PIECE_DIRS)


def _strip():
    """Take every piece's own modules out of sys.modules; returns them."""
    out = {k: m for k, m in list(sys.modules.items()) if _own(m)}
    for k in out:
        del sys.modules[k]
    return out


class Piece:
    def __init__(self, name, module, dirs):
        self.name, self.dirs = name, [os.path.join(ROOT, d) for d in dirs]
        _strip()
        sys.path[:0] = self.dirs
        try:
            self.mod = importlib.import_module(module)
        finally:
            self.owned = _strip()
            for d in self.dirs:
                if d in sys.path:
                    sys.path.remove(d)

    def run(self):
        _strip()
        sys.modules.update(self.owned)
        sys.path[:0] = self.dirs
        try:
            for k, v in SETTINGS.get(self.name, {}).items():
                setattr(self.mod, k, v)
            return self.mod.build()
        finally:
            self.owned = _strip()
            for d in self.dirs:
                if d in sys.path:
                    sys.path.remove(d)


def build():
    import sorceress_pendant as SP
    res = SP.build()
    SP.build = lambda: res                      # (the body is built once)
    names = PIECES or [p for p, _, _ in ORDER]
    for name, module, dirs in ORDER:
        if name not in names:
            continue
        print('habit_all: %s' % name, flush=True)
        import bpy
        before = set(bpy.data.objects)
        Piece(name, module, dirs).run()
        if name in HIDDEN:
            new = [o for o in bpy.data.objects if o not in before]
            for o in new:
                o.hide_render = True
                o.hide_viewport = True
            print('habit_all: %s built and hidden (%d objects)' % (name, len(new)), flush=True)
    import bpy
    bpy.context.view_layer.update()
    return res
