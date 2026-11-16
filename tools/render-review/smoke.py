#!/usr/bin/env python3
"""Run a render script for real, headless, up to the moment it would start its first render, then stop.

    python3 smoke.py path/to/script.py [its arguments...]

Everything before the first render runs exactly as it will in the render queue: the imports, the argument
parsing, the whole scene build. The first render never starts: a render_init handler ends the process there.

Exit codes: 77 the script reached its first render (good); 0 it finished without rendering (good, e.g. the glb
export); anything else is the script's own failure, with its traceback on stderr.
"""
import os
import runpy
import sys

import bpy

REACHED_RENDER = 77


@bpy.app.handlers.persistent          # survives the read_factory_settings() every build starts with
def _stop_at_first_render(*_):
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
