#!/usr/bin/env python3
"""Run a render script for real, headless, up to the moment it would start its first render, then stop.

    python3 smoke.py path/to/script.py [its arguments...]

Everything before the first render runs exactly as it will in the render queue: the imports, the argument
parsing, the whole scene build. The first render never starts: a render_init handler ends the process there.

With RQ_HAIRCHECK=DIR in the environment, the hair is checked at that moment too (hair_check.py: creases,
ridges, ledges, clothes through the hair, the strand paint) and the result is written to DIR.

With RQ_PROFILE='OBJECT|xs|z0:z1:step|FILE', one object's front and back surface is ray-cast there too
(profile.py), to compare two commits' geometry in numbers.

Exit codes: 77 the script reached its first render (good); 0 it finished without rendering (good, e.g. the glb
export); anything else is the script's own failure, with its traceback on stderr.
"""
import os
import runpy
import sys

import bpy

REACHED_RENDER = 77


def _tool(name):
    """A module from this folder, loaded by its path: a branch may carry a file of the same name (the warden has
    its own hair_check.py) and the standard library has a `profile`, and neither must be picked up instead."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("render_review_" + name,
                                                  os.path.join(os.path.dirname(os.path.abspath(__file__)), name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@bpy.app.handlers.persistent          # survives the read_factory_settings() every build starts with
def _stop_at_first_render(*_):
    if os.environ.get("RQ_HAIRCHECK"):                 # the scene is built: measure the hair before leaving
        try:
            _tool("hair_check").check_scene(os.environ["RQ_HAIRCHECK"])
        except Exception:                               # a failed measurement must not fail the build check
            import traceback
            traceback.print_exc()
            os.makedirs(os.environ["RQ_HAIRCHECK"], exist_ok=True)
            with open(os.path.join(os.environ["RQ_HAIRCHECK"], "hair-check-error.txt"), "w") as f:
                traceback.print_exc(file=f)
    if os.environ.get("RQ_PROFILE"):                   # and, when asked, where one object's surface lies
        try:
            _tool("profile").measure(os.environ["RQ_PROFILE"])
        except Exception:
            import traceback
            traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(REACHED_RENDER)


def main():
    script = sys.argv[1]
    bpy.app.handlers.render_init.append(_stop_at_first_render)
    sys.argv = [script] + sys.argv[2:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
