# Lace tension solver

`blender/shared/lace_tension.py` finds a laced opening's proportions from the forces instead of by eye: the gap between
the panels, the eyelets' inset from the edge, the offset of one row against the other, and the cord's rest length. It
answers the boot question "too far apart and the laces pull, too close and they do no structural work" with a number.
It is numpy only (matplotlib for the plot), runs in about ten seconds, and works inside Blender too.

```
python3 blender/shared/lace_tension.py --out /tmp/lt              # the boot's numbers: report and plot
python3 blender/shared/lace_tension.py --set pitch=0.10 --set mu=0.3
```

```python
import lace_tension as lt
res = lt.solve()          # res['ratios']: lace_gap, eyelet_in, eyelet_pitch, eyelet, lace (shares of H, as last.RATIOS)
lt.evaluate({}, gap=0.096, inset=0.028)   # the cost and its parts at one design
```

## The model

The lacing holds the shaft closed round the leg: a hoop pull `N = q * leg_r` per unit length of the opening (the
boot's grip on the leg times the leg's radius). With the rows `D = gap + 2 * inset` apart and the eyelet pitch `p`, a
diagonal leans `theta` from the across direction, `tan(theta) = p / D`, and only `cos(theta)` of its tension closes
the opening.

| part | what it does | which way it pushes |
|---|---|---|
| cord | tension `T = N p / (2 cos theta)`, stored energy `T^2 l / 2EA` per diagonal | least at `D = sqrt(2) p` (stiffest closure) |
| friction | capstan at each eyelet, `exp(mu * turn)`, turn in the plane plus over the ring's rims | rows a little closer (smaller hairpin) |
| sideways pull | tension difference across an eyelet pulls it along the edge | loads the leather along the edge |
| leather | plane-stress FE cell, one pitch of panel, the ring bearing on its hole | prefers a deeper inset, but only slightly |
| front | cord's normal curvature `cos^2/R_across + sin^2 * k_along`, lift over the gap where concave | penalises steep diagonals at the ankle hollow |

Cost = (cord + leather energy per mm of opening) x (1 + penalties). The penalties are a diagonal closing with less
than `share_min` of its tension (cos theta < 0.5, steeper than 60 degrees), a cord lifting off the tongue by more than
its radius, a ligament thinner than `ligament_min` ring radii, and an optional least tongue width `gap_min`.

The inset is set by the ligament rule, not by the optimiser. The leather FE shows the leather's energy changes by only
a few per cent from a thin ligament to a deep one, and at a boot's loads (about 3 N an eyelet) the ligament is far from
tearing. So the shallowest inset the rule allows is the best, because it shows the most tongue. At that inset the gap
and the offset are the least cost.

## Results (the boot: pitch 0.085 H, ring 9 mm across, cord 2 mm across)

- **Rows apart / pitch = 1.39.** This is the ratio to carry, and it hardly moves: 1.41 without friction (exactly
  sqrt 2), 1.36 at mu 0.3, 1.37 to 1.38 at pitch 0.07 H to 0.108 H. The grip, the cord's stiffness and the cord's
  thickness scale out. Only friction and the rule for the inset shift the gap.
- **Gap = 1.39 p - 2 inset.** For the boot that is 15.9 mm (0.064 H, 0.75 pitch) with the inset at 6.7 mm (0.027 H).
- Within 3% of the least cost: gaps of 10.6 to 22.8 mm. Within 10%: 6.9 to 30.3 mm. Past about 3 cm the cord
  stretches and the eyelets turn near hairpins (the "pull" side). Under about 7 mm the diagonals run along the opening
  and need much more tension for the same hold (the "no structural work" side).
- Main's boot (gap 0.035 H, rows 1.07 pitch apart, diagonals 43 degrees) and the hose thread's 2.4 cm (rows 1.79
  pitch, 29 degrees) both cost about 4% over the best, on opposite sides of it.
- To show a wider tongue without leaving the best ratio, widen the pitch with it: a 2.4 cm gap at the same inset wants
  a pitch of about 0.108 H (2.7 cm), which means fewer eyelets.
- Row offset: 0 (rows facing). An offset only trades the sideways pull of one row against the other.
- The cord's lift off the tongue never triggers at these proportions: the shin's curvature across the opening
  outweighs the ankle's hollow for any diagonal flatter than about 60 degrees.
- Cord rest length: 0.9958 of its laid path (it stretches 0.4% under the hold). Give the cloth or rope solver that.

## Limits

- The front is treated as locally developable (a cylinder or cone laid flat). The lift check uses curvatures at
  three stations, not the boot's mesh.
- Friction is local (each eyelet against its neighbours), as after a wearer cinches each crossing in turn. A single
  pull from the top end would leave the bottom much looser.
- The leather cell's mesh is coarse (about 0.66 mm), so its energy jitters by a few per cent between insets. That is
  below the decision, since the inset comes from the rule.
- The weights are physical (energies in N mm per mm). Only the penalty multipliers are judgment calls.

## Next uses

The same set-up (unknown ratios, a physical cost, constraints for the look, a window rather than a point) could be
used for other proportion questions, such as the nose ratios. It has not been started for them.
