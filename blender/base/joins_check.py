"""The join check as a view script, for a request before the queue's own copy of rq_views.py has --joins:

    blender/base/joins_check.py --figure FIGURE [--mannequin] --joins neck,foot-top [--joins-only] [views.py flags]

It runs this commit's tools/render-queue/rq_views.py as the queue would (in place of views.py: the same build and
views, then the geometry checks, the layer gaps and the join check, tools/joins). Once the queue's copy carries
--joins, ask for views.py with --joins directly."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'tools', 'render-queue', 'rq_views.py')
code = compile(open(SRC).read(), os.path.join(HERE, 'rq_views.py'), 'exec')
exec(code, {'__file__': os.path.join(HERE, 'rq_views.py'), '__name__': '__main__'})
