"""build4.py name 'json': the lower nose rebuilt the way sculptors block it (a wedge with a tilted bottom plane, the ball as its
rounded end, crescent wings on its sides running into the cheek, bean nostrils carved last), on the i1 block's bridge."""
import sys, json, subprocess, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
name = sys.argv[1]
W4 = dict(off=1.0, tilt=3.0, show=0.17, colw=0.2, wback=0.45, wfront=1.1, lean=0.35, pf=1.8, kball=0.9, kbase=1.5, top=0.5, kjoin=5.0,
          wc=0.30, wl=0.75, wh=0.55, wp=0.5, arch=0.06, kw=2.0, wall=0.17, ncol=0.3, nb=0.12, nh=1.3, kwb=3.0, nf=1.0, goode=0, showmm=0, knos=1.2, archf=0, tear=0, tback=1.15, kcr=4.2, wasw=1, kflo=3.2, tipb=0, cbox=0, blg=0, blc=0.3, blw=0.35, cres=0, cf0=0.0, cf1=0.75, ctp=0.8, cwp=1.6, chb=1.0, chf=0.6, cgr=0.3)
W4.update(json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
b = open('nose/nose-block-i1.js').read()
start = b.index('    // the tip: two domes')
end = b.index('    o[2] = (nE - 1) * Math.min(nHw, nHl, nHn) * 0.8; o[3] = ns - 0.35 * nHn;\n') + len('    o[2] = (nE - 1) * Math.min(nHw, nHl, nHn) * 0.8; o[3] = ns - 0.35 * nHn;\n')
NEW = r'''    // ---- the lower nose as a sculptor blocks it: one wedge (its bottom plane tilted up toward the tip, its sides leaning in
    // as they rise, its front the lobule falling from the tip to the columella's break), the ball as the wedge's rounded end,
    // a crescent wing on each side of it whose back runs into the cheek, and the nostrils carved last into the bottom plane.
    // (f forward of the subnasale, u up from it) ----
    var f = ySn - y, u = z - zSn;
    var uB = W4.off * mm + f * tanB;
    var colW = W4.colw * alarHalf;
    var hang = (W4.showmm ? W4.showmm * mm : W4.show * alarHalf) * sm(-0.05 * Tp, 0.25 * Tp, f) * (1 - sm(fBr, fBr + 0.4 * (Tp - fBr), f)) * sm(colW + 1.2 * mm, colW - 0.6 * mm, ax);
    var ub = uB - hang;
    var Wb = lerp(W4.wback * alarHalf, W4.wfront * rEnv, sm(0, Tp, f)), W = Wb - W4.lean * Math.max(0, u - uB);
    var eS = (ax - W) / Math.sqrt(1 + W4.lean * W4.lean);
    var q = clamp((hP - u) / Math.max(1e-6, hP - uBr), 0, 1.6), fF = Tp - (Tp - fBr) * Math.pow(q, W4.pf);
    var eF = (f - fF) * 0.8, eB = ub - u;
    var dL = -nSmin(-eS, -eF, W4.kball * rEnv);
    dL = -nSmin(-dL, -eB, W4.kbase * mm);
    dL = -nSmin(-dL, -(u - (hP + W4.top * rEnv)), 3 * mm);
    var dN = nSmin(dD, dL, W4.kjoin * mm);
    // the wing: a sail on the wedge's side, its outline an arc from the cheek round to the lobule, thickest at its foot, its
    // lower rim arched up between its two ends
    var ef = (f - wC) / wRf, uBw = uB + W4.arch * alarHalf * Math.max(0, 1 - ef * ef) * (W4.archf ? (1 - sm(0.0, 0.8, ef)) : 1);
    var uw = Math.max(0, u - uBw) / wRu, r2 = ef * ef + uw * uw;
    var X = W + wT * Math.pow(Math.max(0, 1 - r2), W4.wp);
    var dW = -nSmin(-(ax - X) * 0.7, -(Math.sqrt(r2) - 1) * Math.min(wRf, wRu) * 0.6, W4.kw * mm);
    var Tt = 0;
    if (W4.cres) {
      // (the wing as a crescent: thickest at its base by the cheek, tapering forward until it runs into the lobule's side;
      // its height falls from back to front, its cross-section eases off gently, and a shallow groove marks its top edge)
      var cf0 = W4.cf0 * alarHalf, cf1 = W4.cf1 * Tp, tc = clamp((f - cf0) / Math.max(1e-6, cf1 - cf0), 0, 1);
      var Wbc0 = lerp(W4.wback * alarHalf, W4.wfront * rEnv, sm(0, Tp, cf0));
      Tt = (alarHalf - Wbc0) * Math.pow(1 - tc, W4.ctp) * sm(cf0 - 0.4 * alarHalf, cf0, f);
      Tt += W4.blg * mm * Math.exp(-Math.pow((tc - W4.blc) / W4.blw, 2)) * sm(cf0 - 0.4 * alarHalf, cf0, f);   // (the alar lobule: the widest point, al)
      var Hc = wRu * lerp(W4.chb, W4.chf, tc), vc = Math.max(0, u - uBw) / Hc;
      var gv = Math.pow(Math.max(0, 1 - vc * vc), W4.cwp);
      var gr = W4.cgr * mm * Math.exp(-Math.pow((vc - 1) / 0.22, 2)) * (1 - sm(0.6, 1.0, tc)) * sm(cf0 - 0.4 * alarHalf, cf0, f);
      X = W + Tt * gv - gr;
      dW = -nSmin(-(ax - X) * 0.7, -(W - 0.5 * mm - ax), 1.0 * mm);   // (only the shell outside the wedge's side)
      if (W4.cbox) {      // (and only where the wing is: below its top and behind its front, so where the wedge's sides meet high up
        // the shell's two bounds don't both fall to zero and leave a sheet of near-surface out in front of the bridge)
        dW = -nSmin(-dW, -(u - uBw - Hc), 1.0 * mm); dW = -nSmin(-dW, -(f - cf1 - 1.0 * mm), 1.0 * mm); }
    }
    dW = -nSmin(-dW, -(uBw - u), W4.kbase * mm);
    o[0] = dN; o[1] = dW;
    // the nostril: a bean in the bottom plane between the columella and the wing's wall, domed up into the nose
    var fN0 = W4.nb * Tp, fN1 = W4.nf * fBr, hl = 0.5 * (fN1 - fN0), fNc = 0.5 * (fN0 + fN1);
    var Wb1 = W - 0 * mm, Xo = Wb1 + (W4.cres ? Tt : wT * Math.pow(Math.max(0, 1 - ef * ef), W4.wp)) - W4.wall * alarHalf;
    var xi = colW + W4.ncol * mm, hw = Math.max(0.6 * mm, 0.5 * (Xo - xi)), xc = 0.5 * (xi + Xo), ns = u - uB, nHn = nR[2] * W4.nh;
    // (teardrop: wide at the back by the alar base, narrowing toward the tip with its inner edge kept along the columella,
    // so the long axis leans in toward the tip)
    var tN = clamp((f - fN0) / Math.max(1e-6, fN1 - fN0), 0, 1);
    if (W4.tear) { var hwT = hw * lerp(W4.tback, W4.tear, Math.pow(tN, 1.3)); xc = lerp(xc, xi + hwT, tN); hw = Math.max(0.5 * mm, hwT); }
    var nE = Math.sqrt(Math.pow((ax - xc) / hw, 2) + Math.pow((f - fNc) / hl, 2) + Math.pow(Math.max(0, ns) / nHn, 2));
    o[2] = (nE - 1) * Math.min(hw, hl, nHn) * 0.8; o[3] = ns - 0.35 * nHn;
'''
b = b[:start] + NEW + b[end:]
old = "  Tp = nSmin(Tp, 1.24 * aH0, 1.0 * mm) * (1 - 0); yPrn = ySn - Tp;"
assert b.count(old) == 1
if W4['goode']:
    b = b.replace(old, """  // (Goode: the tip's projection from the alar crease is a set share g of the nasion-to-tip length; solved for the tip's
  // reach from the subnasale, with the crease where the wing meets the cheek and the nasion where the bridge starts)
  (function () { var g = %s, fc = (%s - %s) * aH0, fn = -(yN - %s * mm - ySn), H = (zN - zSn) - (zPrn - zSn);
    var A = 1 - g * g, Bq = -2 * (fc - g * g * fn), Cq = fc * fc - g * g * (fn * fn + H * H);
    Tp = (-Bq + Math.sqrt(Math.max(0, Bq * Bq - 4 * A * Cq))) / (2 * A); })();
  yPrn = ySn - Tp;""" % (W4['goode'], W4['wc'], W4['wl'], 'NL.radix * np'))


# the tip set from the face's own bridge (John: the Ranger's and the Warden's profiles crooked): the tip stands a set
# distance in front of the bridge's most forward point above it, instead of at one ratio for every face
if W4['tipb']:
    tb = r"""
  (function () { var zSup_ = zPrn + 0.14 * L, hump_ = (1.4 * sx - 0.25) * mm * (0.7 + 0.3 * np), yRad_ = yN - NL.radix * np * mm;
    function drop(T) { var ySup_ = ySn - T + 1.3 * mm, best = 1e9;
      for (var q = 0; q <= 60; q++) { var tc = 0.25 + 0.7 * q / 60;
        var yl = lerp(ySup_, yRad_, tc) - hump_ * Math.sin(Math.PI * tc) + 4.2 * mm * Math.sin(Math.PI * tc) * Math.pow(tc, 1.5) - 0.6 * mm * Math.exp(-Math.pow((tc - 0.38) / 0.16, 2));
        best = Math.min(best, yl); }
      return T - (ySn - best); }
    var lo = 0.6 * Tp, hi = 1.1 * Tp, tgt = %s * mm;
    if (typeof console !== 'undefined') { var dd = []; for (var qq = 0; qq <= 6; qq++) { var tc2 = qq / 6; dd.push(((ySn - (lerp(ySn - Tp + 1.3 * mm, yRad_, tc2) - hump_ * Math.sin(Math.PI * tc2) + 4.2 * mm * Math.sin(Math.PI * tc2) * Math.pow(tc2, 1.5))) / mm).toFixed(1)); } console.log('NOSE tipdrop Tp ' + (Tp / mm).toFixed(1) + ' drop ' + (drop(Tp) / mm).toFixed(2) + ' yl fwd ' + dd.join(',') + ' yRad ' + ((ySn - yRad_) / mm).toFixed(1)); }
    for (var it = 0; it < 30; it++) { var mid = 0.5 * (lo + hi); if (drop(mid) > tgt) hi = mid; else lo = mid; }
    Tp = 0.5 * (lo + hi); })();
  yPrn = ySn - Tp;""" % W4['tipb']
    assert b.count('  yPrn = ySn - Tp;') == 1
    b = b.replace('  yPrn = ySn - Tp;', '  yPrn = ySn - Tp;' + tb, 1)
# the frame's constants, set once
pre = '''  var W4 = %s;
  var hP = zPrn - zSn, tanB = Math.tan(Math.atan2(Math.max(0, hP - rT - 2.0 * mm - W4.off * mm), Tp0 - rT) + W4.tilt * Math.PI / 180);
  var fBr = vBr, uBr = W4.off * mm + fBr * tanB;
  var wC = W4.wc * alarHalf, wRf = W4.wl * alarHalf, wRu = W4.wh * alarHalf, wT = alarHalf - lerp(W4.wback * alarHalf, W4.wfront * rEnv, sm(0, Tp, wC));
  function parts(x, y, z, o) {''' % json.dumps(W4)
assert b.count('  function parts(x, y, z, o) {') == 1
b = b.replace('  function parts(x, y, z, o) {', pre)
assert b.count('    if (1) {') == 1
b = b.replace('    if (1) {', '    if (0) {')
b = b.replace('* sm(zSn + 0.5 * mm, zSn + 3.0 * mm, z);', '* sm(zSn + 0.5 * mm, zSn + 3.0 * mm, z) * %s;' % W4['wasw'])
b = b.replace('lerp(3.2 * mm, 5.0 * mm', 'lerp(%s * mm, 5.0 * mm' % W4['kflo'])
b = b.replace('4.2 * mm, wa)', '%s * mm, wa)' % W4['kcr'])
b = b.replace('kNos = 1.2 * mm', 'kNos = %s * mm' % W4['knos'])
b = b.replace('var dN = nSmin(PT[0], PT[1], 3.0 * mm);', 'var dN = nSmin(PT[0], PT[1], %s * mm);' % W4['kwb'])
open('var/blk/nose-block.js', 'w').write(b)
s = open('nose/patch_nose.py').read().replace("os.path.join(HERE, 'nose-block.js')", "'var/blk/nose-block.js'")
open('var/blk/patch.py', 'w').write(s)
subprocess.run(['python3', 'var/blk/patch.py', 'nose/main-base.js', 'var/%s.js' % name, 'nose/fc-base.html', 'var/%s.html' % name], check=True, stdout=subprocess.DEVNULL)
