"""Moved to tools/gauges/color/colorcheck.py (the solvers and gauges library: tools/INDEX.md). This shim keeps the old
`import colorcheck` and `python3 tools/render-review/colorcheck.py ...` working: it runs the real file under this module's name, so
the module, its state and its command line are the same object either way. Change the real file, not this one."""
import importlib.util as _iu, os as _os, sys as _sys
_path = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../gauges/color/colorcheck.py'))
_spec = _iu.spec_from_file_location(__name__, _path)
_mod = _iu.module_from_spec(_spec)
_sys.modules[__name__] = _mod
_spec.loader.exec_module(_mod)
