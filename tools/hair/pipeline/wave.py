"""The wave model every style shares, and the gauge that measures it on built lines.

The model (John 2026-11-28, from the coordinator's brief). A wave is two perpendicular sinusoids along the hair,
laid on a lock's frame (out from the body, and across the fall):
    out(s)  = A_out  sin(k s + phi)
    side(s) = A_side sin(k s + phi + delta),        k = 2 pi / lambda
  - delta = +-90 deg and A_out = A_side: a round helix (a curl), its handedness the sign of delta;
    A_out != A_side: an elliptical helix (the Face Creator's wavyMass: 0.034 out, 0.040 across);
  - delta = 0 or one amplitude 0: a planar meander, a sine-generated S in one plane (the ponytail's S wave);
  - "twist" is delta: how far the wave turns out of its plane. 0 is flat, 90 deg is a full helix.
A curve of constant curvature and torsion is a circular helix (Kirchhoff rods at rest: Bertails et al. 2006,
Super-Helices; Bergou et al. 2008, Discrete Elastic Rods), so the helix case is the rod solver's natural shape; the
planar case is Langbein and Leopold's sine-generated curve (1966, USGS Prof. Paper 422-H, "River meanders: theory of
minimum variance": the direction swings as a sine of arc length, the bend of least total variance for a line of
given length between two points), at small amplitude the same as a sine offset.

Neighbouring locks share wavelength and phase. A set is made in sections and then brushed or combed out "to form a
uniform wave pattern" (saloncentric.com; behindthechair.com), so neighbours swing together: phase is a smooth
function round the head (one value per section, or a slow drift), plus a small jitter per lock. The gauge measures
that sharing as the phase step between neighbouring lines.

    offsets(s, lam, a_out, a_side, phi, delta, grow=None) -> (out, side)
    measure(lines, ...) -> dict: wavelength, the two amplitudes, twist, neighbour phase step
"""
import math

import numpy as np


def offsets(s, lam, a_out, a_side, phi=0.0, delta=math.pi / 2, grow=None):
    """The wave's two offsets at arc (or height) s; grow (same shape as s, 0 to 1) fades it in, as at a tie or a
    crown where the set has not taken."""
    k = 2 * math.pi / lam
    g = 1.0 if grow is None else grow
    return g * a_out * np.sin(k * s + phi), g * a_side * np.sin(k * s + phi + delta)


def helix_curvature(lam, a):
    """A round helix of radius a and pitch lam: curvature a / (a^2 + c^2), torsion c / (a^2 + c^2), c = lam / 2 pi."""
    c = lam / (2 * math.pi)
    return a / (a * a + c * c), c / (a * a + c * c)


def _frame(p, axis):
    """Per node: the fall's direction t (smoothed), 'out' (radial from the body's axis, square to t) and 'side'."""
    t = np.gradient(p, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    r = p.copy(); r[:, 0] -= axis[0]; r[:, 1] -= axis[1]; r[:, 2] = 0.0
    out = r - (r * t).sum(1, keepdims=True) * t
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    side = np.cross(t, out)
    return t, out, side


def fit_line(p, lam_range, step, axis=(0.0, 0.04), trend=None, along='arc'):
    """One line's wave: the residual from its own trend (a Gaussian smoothing over a wavelength, or `trend` if the
    base is known), projected on out and side, fitted by least squares with a sinusoid over the wavelengths in
    lam_range. along: 'arc' reads the wave along its base's arc (a wave laid by arc, as on the ponytail's middle line),
    'height' along the drop in height (a wave laid by height, as photo_waves lays the falls' so neighbours swing
    together). Returns (lam, a_out, a_side, phase_out, phase_side) or None."""
    from scipy.ndimage import gaussian_filter1d
    p = np.asarray(p, float)
    if trend is not None and len(trend) == len(p):
        # (the base is known node for node, as a builder that lays the wave on a base line has it: the residual is
        # read node by node and spaced by the base's own arc, so the wave's extra length does not shift its phase)
        trend = np.asarray(trend, float)
        _, out, side = _frame(trend, axis)
        res = p - trend
        L = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(trend, axis=0), axis=1))] if along == 'arc' else trend[0, 2] - trend[:, 2]
        if np.any(np.diff(L) <= 0) or L[-1] < 1.5 * lam_range[1]:
            return None
        s = np.arange(0, L[-1], step)
        yo = np.interp(s, L, (res * out).sum(1))
        ys = np.interp(s, L, (res * side).sum(1))
    else:
        # (no base: a cubic along the line stands for it, and is fitted again with the sinusoid, so a fall that is
        # only two or three waves long is still read: a smoothing over a wavelength would eat its ends)
        L = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        if L[-1] < 1.5 * lam_range[1]:
            return None
        u = np.linspace(0, L[-1], max(int(L[-1] / step), 8))
        q = np.stack([np.interp(u, L, p[:, c]) for c in range(3)], 1)
        x = u / L[-1]
        base = np.stack([np.polyval(np.polyfit(x, q[:, c], 3), x) for c in range(3)], 1)
        _, out, side = _frame(base, axis)
        res = q - base
        # read along the base, not the hair (the hair's arc is longer by the wave's take-up, about (k A)^2 / 4)
        Lb = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(base, axis=0), axis=1))] if along == 'arc' else base[0, 2] - base[:, 2]
        if np.any(np.diff(Lb) <= 0) or Lb[-1] < 1.5 * lam_range[1]:
            return None
        s = np.arange(0, Lb[-1], step)
        yo, ys = np.interp(s, Lb, (res * out).sum(1)), np.interp(s, Lb, (res * side).sum(1))
        poly = np.c_[np.ones_like(s), s, s * s, s ** 3]
    best = None
    if trend is not None and len(trend) == len(p):
        poly = np.ones((len(s), 1))
    Y = np.c_[yo, ys]
    for lam in np.linspace(lam_range[0], lam_range[1], 41):
        k = 2 * math.pi / lam
        M = np.c_[np.sin(k * s), np.cos(k * s), poly]
        # (normal equations on a handful of columns: many small lstsq calls are slow under a threaded BLAS)
        A = M.T @ M
        co, cs = np.linalg.solve(A + 1e-12 * np.trace(A) * np.eye(len(A)), M.T @ Y).T
        pw = co[0] ** 2 + co[1] ** 2 + cs[0] ** 2 + cs[1] ** 2
        if best is None or pw > best[0]:
            best = (pw, lam, co, cs)
    _, lam, co, cs = best
    return lam, math.hypot(co[0], co[1]), math.hypot(cs[0], cs[1]), math.atan2(co[1], co[0]), math.atan2(cs[1], cs[0])


def measure(lines, lam_range, step=None, axis=(0.0, 0.04), trends=None, order=None, along='arc'):
    """The wave on a set of lines (head heights). order: the lines' order round the head (neighbours adjacent), for
    the phase-sharing measure. Returns wavelength, amplitudes (out, side), twist (deg: the phase of side against out,
    0 planar, 90 helix) as medians, and the median phase step between neighbours (deg, of the stronger component)."""
    step = step or lam_range[0] / 24
    fits = []
    for j, p in enumerate(lines):
        f = fit_line(np.asarray(p), lam_range, step, axis, None if trends is None else trends[j], along)
        fits.append(f)
    ok = [f for f in fits if f is not None]
    if not ok:
        return dict(lines=0)
    F = np.array(ok)
    strong_side = np.median(F[:, 2]) >= np.median(F[:, 1])
    tw = np.degrees(np.angle(np.exp(1j * (F[:, 4] - F[:, 3]))))
    tw_ok = (F[:, 1] > 0.25 * np.median(F[:, 1])) & (F[:, 2] > 0.25 * np.median(F[:, 2]))
    out = dict(lines=len(ok), wavelength=float(np.median(F[:, 0])), amp_out=float(np.median(F[:, 1])),
               amp_side=float(np.median(F[:, 2])),
               twist_deg=float(np.median(np.abs(tw[tw_ok]))) if tw_ok.any() else 0.0)
    if order is not None:
        ph = np.array([(f[4] if strong_side else f[3]) if f is not None else np.nan for f in fits])
        lam = np.array([f[0] if f is not None else np.nan for f in fits])
        o = np.asarray(order)
        steps = []
        for a, b in zip(o[:-1], o[1:]):
            if np.isfinite(ph[a]) and np.isfinite(ph[b]) and abs(lam[a] - lam[b]) < 0.05 * lam[a]:
                steps.append(abs(np.degrees(np.angle(np.exp(1j * (ph[b] - ph[a]))))))
        out['neighbour_phase_step_deg'] = float(np.median(steps)) if steps else None
        out['neighbour_pairs'] = len(steps)
    return out
