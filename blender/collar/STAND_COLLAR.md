# The stand collar with turned-back fronts: the saved recipe

John, 2026-11-21: "Save the current recipe for building a beautiful stand collar." This is the build behind
`bodice-button-on-curved-fold.png`.

The order is the tailor's: cut, fit, fold, and button last.

## 1. Cut and fit

```
python3 blender/base/views.py --figure bodice_cut --mannequin --turn --angles front --size 300 --samples 8 \
    --save CUT.blend --out DIR
```

`blender/bodice_cut/bodice_cut.py` builds the redingote bodice whole to the neck: two fronts, the right lapped over the
left, no lapel piece, no front cut open, no buttons. The cloth is 0.085 w thick (4.4 mm). The bodice's own neck edge
stands round the neck: that is the stand collar.

## 2. Fold, then button

```
python3 blender/collar/foldfront.py CUT.blend OUT.blend '{"recipe": "stand_collar"}'
```

What the recipe does, in order:

1. Eases the upper chest across the line where the flat chest canvas meets the round of the shoulder (150 rounds).
2. Marks each front's roll line into its own mesh, from the neck edge at 2.2 from the middle down to the front's edge
   0.15 above the bust line. Nothing is added to the mesh and nothing is welded.
3. Curves the line: bowed 0.5 away from the V, fullest two thirds of the way down, so it falls steeply from the neck
   and sweeps in to the button. The V is rounded.
4. Turns each front's own cloth past the line back over it on a roll 1.3 cm high (radius 0.24), lying on the front with
   the cloth's thickness. The under front is folded first, on its own cloth, and pressed behind the upper front where
   that covers it.
5. Sets the buttons last: the top one just under where the two folds meet at the middle (0.03 clear), the rest at the
   bodice's own spacing down to the point.

Any option given beside the recipe overrides it, for example `'{"recipe": "stand_collar", "foot": 43.9}'` for a
deeper fold.

## What it does not do yet

No sleeves or skirt are in this build. There are small ticks along the fold lines and at their neck ends.
