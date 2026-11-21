# Clothing solvers: lessons learned

Notes from building the gambeson, sleeves and linen shirt (`blender/doublet/`, `blender/chemise/`).
Read these before running a cloth solver on a layered figure.

- [compression.md](compression.md): what happens when cloth is pressed between layers, and how to draft for it.
- [cloth-solver.md](cloth-solver.md): holds, colliders, blow-ups, stiffness and thickness settings.
- [inspection.md](inspection.md): the contour and stress tools, and when to stop iterating.

Short version:

1. Draft the cloth to the room it will have. The solver settles a draft, it does not fix one.
2. Build a new layer without the solver over the kept lower layers. Run the solver only on request.
3. Hold whole garments, never part of one.
4. Look at contours and stress before any render.
