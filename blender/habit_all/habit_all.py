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
if os.environ.get('HABIT_SOLVE'):    # the solver on every layer that has one (the petticoat's is always on)
    SETTINGS = {'skirt': {'SETTLE': 1}, 'redingote_bodice': {'SETTLE': True}}
# The chemise is the loosest layer and nothing yet draws it in under the bodice and the sleeves (they fit over the corset
# and the body, not over it), so it bursts out through both. It is built, so the layers over it can be fitted to it, but
# not shown until they are: HABIT_SHOW=chemise shows it.
HIDDEN = [x for x in os.environ.get('HABIT_HIDE', 'chemise').split(',') if x and x not in os.environ.get('HABIT_SHOW', '').split(',')]
# What the skirt must clear: the layers worn under it (by name prefix). Its draft stands its cloth off the larger of the
# body's sections and theirs, so the corset's lower edge, which flares over the hips, is tucked under the skirt rather than
# standing out through it.
SKIRT_OVER = tuple(x for x in os.environ.get('HABIT_UNDER', 'Corset,Petticoat').split(',') if x)
# One wool for the whole habit. The pieces were each coloured on their own bench (skirt 1c2a52, sleeves 161b2a, bodice
# 121621), and together the skirt read as a brighter blue than the bodice above it. They are dyed alike here (the
# bodice's: the darkest navy, picked on its swatch bench); HABIT_DYE=rrggbb for another.
DYE = int(os.environ.get('HABIT_DYE', '121621'), 16)
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
            if self.name == 'skirt' and SKIRT_OVER:
                self._fit_over_layers()
            for k, v in SETTINGS.get(self.name, {}).items():
                setattr(self.mod, k, v)
            return self.mod.build()
        finally:
            self.owned = _strip()
            for d in self.dirs:
                if d in sys.path:
                    sys.path.remove(d)


def _points(prefixes):
    """The worn layers' points in world space, as they stand now (the rig, the modifiers)."""
    import bpy
    import numpy as np
    dg = bpy.context.evaluated_depsgraph_get()
    out = []
    for o in bpy.data.objects:
        if o.type != 'MESH' or o.hide_render or not o.name.startswith(tuple(prefixes)):
            continue
        e = o.evaluated_get(dg)
        me = e.to_mesh()
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get('co', co)
        mw = np.array(o.matrix_world)
        out.append(co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3])
        e.to_mesh_clear()
    return np.concatenate(out)


def _fit_over_layers(self):
    """The skirt module's make_skirt, handed what lies under it, so its draft clears it."""
    make, draft = self.mod.make_skirt, self.mod.draft
    pts = _points(SKIRT_OVER)
    pts = pts[::max(1, len(pts) // 40000)]            # (its sections need an outline, not every point: 570 000 of them ran out of memory)
    print('habit_all: the skirt fitted over %d points of %s' % (len(pts), ', '.join(SKIRT_OVER)), flush=True)
    self.mod.make_skirt = lambda body, rig, M, **kw: make(body, rig, M, under=draft.under_body(pts), **kw)


Piece._fit_over_layers = _fit_over_layers


def _dye():
    import female
    female.mat_def(87, 'skirt wool', female.ramp(DYE, 0.1), 0.0, surface='twill', closeup='skirt')
    female.mat_def(96, 'sleeve wool', female.ramp(DYE, 0.10), 0.0, surface='wool', closeup='wool', soft_edge=True)
    SETTINGS.setdefault('redingote_bodice', {})['DYE'] = DYE


def build():
    import sorceress_pendant as SP
    res = SP.build()
    _dye()
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
