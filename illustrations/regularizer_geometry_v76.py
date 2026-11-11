"""Revision 76: the admissible regularizers of Proposition 7 of the paper and the Pinsker constants of Sect. 4.1
in pictures (for Appendix D.3 of the Online Resource; the visuals of the verification companion, redrawn).

(a) Convexity along a generalized geodesic for H_omega = KL + chi^2 (omega = 1) with the prior N(0, I) in the
    plane: Gaussian laws mu_theta = N(m_theta, M_theta^2) from the base N(b, 0.6^2 I), with the chord and the
    bound of 1-convexity.
(b) A prior that is not Gaussian, Gamma = exp(-W) with W(x) = x^2/2 + 0.6 sin x + log Z (lambda_W = 0.4), and the
    laws along the quantile interpolation between a Gaussian and a mixture.
(c) For the integrand G(v) = v^3/2 with omega = 1/2 on that prior: the regularizer less its chord, its relative
    entropy part, and the bound of lambda_W-convexity.
(d) The condition of Proposition 7 on the integrand, s -> exp(s) G(exp(-s)) convex and nonincreasing, for the power
    integrands with r = 3/2, 2 and 3.
(e) The slope G(v)/v for two convex integrands with G(0) = 0, nondecreasing, and for the integrand of relative
    entropy, which has G(0) = 1.
(f) The squared total-variation distance to the prior N(0,1) against the regularizer for Gaussian laws, for
    relative entropy and for H_omega with omega = 1, with the Pinsker bounds c_H H, c_H = 2/(1 + 2 omega).
Output: regularizer-geometry-v76 (pdf/png) and regularizer-geometry-v76.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, panel_label, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

out = {}
fig, axes = plt.subplots(2, 3, figsize=(TEXTW, 4.9), gridspec_kw=dict(wspace=0.62, hspace=1.05))
BELOW = dict(loc="upper center", bbox_to_anchor=(0.5, -0.30))
LG = dict(fontsize=FS - 2, handlelength=1.6, labelspacing=0.25)

# ---- (a) KL + chi^2 along a generalized geodesic of Gaussian laws in the plane
def rot(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s], [s, c]])


def kl_std(m, C):
    return 0.5 * (np.trace(C) + m @ m - 2 - np.log(np.linalg.det(C)))


def chi2_std(m, C):
    """chi^2(N(m, C) || N(0, I)) in the plane, finite when 2 C^{-1} - I is positive definite."""
    Ci = np.linalg.inv(C); A = 2 * Ci - np.eye(2)
    assert np.linalg.eigvalsh(A).min() > 0
    h = 2 * Ci @ m
    return float(np.exp(0.5 * h @ np.linalg.solve(A, h) - m @ Ci @ m) / (np.linalg.det(C) * np.sqrt(np.linalg.det(A))) - 1)


th = np.linspace(0, 1, 101)
a0, a1 = np.array([-0.8, 0.2]), np.array([0.8, 0.4])
S0 = np.diag([1.1, 0.6]); S1 = rot(60) @ S0 @ rot(60).T
wg = float(((a0 - a1) ** 2).sum() + ((S0 - S1) ** 2).sum())                 # the coupling distance through the base
Hk = np.array([kl_std((1 - t) * a0 + t * a1, ((1 - t) * S0 + t * S1) @ ((1 - t) * S0 + t * S1)) for t in th])
Hc = np.array([chi2_std((1 - t) * a0 + t * a1, ((1 - t) * S0 + t * S1) @ ((1 - t) * S0 + t * S1)) for t in th])
H = Hk + Hc; chord = (1 - th) * H[0] + th * H[-1]; bound = chord - 0.5 * th * (1 - th) * wg
assert (H <= bound + 1e-12).all()
ax = axes[0, 0]
ax.plot(th, chord, color=MUTED, lw=1.0, ls=":", label="chord")
ax.plot(th, bound, color=PAL[2], lw=1.2, ls="--", label="bound of $1$-convexity")
ax.plot(th, H, color=PAL[0], lw=1.5, label="$\\mathcal{H}_\\omega$, $\\omega=1$")
ax.plot(th, Hc, color=PAL[3], lw=1.0, ls="-.", label="its part $\\chi^2$")
ax.set_xlim(0, 1); ax.set_ylim(0, None); ax.set_xlabel("position $\\vartheta$"); ax.set_ylabel("$\\mathcal{H}_\\omega(\\mu_\\vartheta)$")
ax.legend(**BELOW, **LG); panel_label(ax, "a")
out["a"] = dict(coupling_distance_squared=wg, H_ends=[float(H[0]), float(H[-1])], H_half=float(H[50]), bound_half=float(bound[50]), chord_half=float(chord[50]))

# ---- (b), (c) a prior that is not Gaussian and the integrand G(v) = v^3/2
LAM = 0.4
Z = quad(lambda x: np.exp(-x ** 2 / 2 - 0.6 * np.sin(x)), -np.inf, np.inf)[0]
W = lambda x: x ** 2 / 2 + 0.6 * np.sin(x) + np.log(Z)
OM, R = 0.5, 3.0
xq = np.linspace(-9, 9, 360001); uq = (np.arange(40000) + 0.5) / 40000
mix = lambda x, comps: sum(w * norm.pdf(x, m, s) for w, m, s in comps)
quant = lambda comps: np.interp(uq, sum(w * norm.cdf(xq, m, s) for w, m, s in comps), xq)
c0 = [(1.0, -1.0, 0.7)]; c1 = [(0.5, 0.4, 0.6), (0.5, 1.6, 0.6)]
Q0, Q1 = quant(c0), quant(c1); dQ0, dQ1 = 1 / mix(Q0, c0), 1 / mix(Q1, c1)


def reg(t):
    Q = (1 - t) * Q0 + t * Q1; dQ = (1 - t) * dQ0 + t * dQ1
    k = np.mean(-np.log(dQ) + W(Q)); f = np.exp(W(Q)) / dQ                     # f is the density ratio to the prior at Q
    return k, k + OM * (np.mean(f ** (R - 1)) / (R - 1) - 1 / (R - 1))


vals = np.array([reg(t) for t in th]); K2, H2 = vals[:, 0], vals[:, 1]
w2 = float(np.mean((Q1 - Q0) ** 2)); conv = (1 - th) * H2[0] + th * H2[-1]; kconv = (1 - th) * K2[0] + th * K2[-1]
low = -LAM / 2 * th * (1 - th) * w2
assert (H2 - conv <= low + 1e-6).all() and (K2 - kconv <= low + 1e-6).all()
ax = axes[0, 1]
xg = np.linspace(-4, 4, 801)
ax.plot(xg, np.exp(-W(xg)), color=MUTED, lw=1.1, ls=":", label="prior")
for t, c, ls in ((0, PAL[0], "-"), (0.5, PAL[2], "--"), (1, PAL[3], "-.")):
    ax.plot((1 - t) * Q0 + t * Q1, 1 / ((1 - t) * dQ0 + t * dQ1), color=c, lw=1.2, ls=ls, label=f"$\\mu_{{{t:g}}}$")
ax.set_xlim(-3.5, 3.5); ax.set_xlabel("$x$"); ax.set_ylabel("density"); ax.legend(ncol=2, **BELOW, **LG); panel_label(ax, "b")
ax = axes[0, 2]
ax.plot(th, H2 - conv, color="k", lw=1.5, label="$\\mathcal{H}_{\\omega,G}$ less its chord")
ax.plot(th, K2 - kconv, color=PAL[0], lw=1.0, ls="-.", label="the same for $D_{\\mathrm{KL}}$")
ax.plot(th, low, color=PAL[7], lw=1.2, ls="--", label="bound of $\\lambda_W$-convexity")
ax.axhline(0, color=MUTED, lw=0.6, ls=":")
ax.set_xlim(0, 1); ax.set_xlabel("position $\\vartheta$"); ax.set_ylabel("difference from the chord"); ax.legend(**BELOW, **LG); panel_label(ax, "c")
out["bc"] = dict(lambda_W=LAM, omega=OM, r=R, W2_squared=w2, H_half=float(H2[50]), chord_half=float(conv[50]), bound_half=float((conv + low)[50]))

# ---- (d), (e) the condition on the integrand
ax = axes[1, 0]
ss = np.linspace(-1.5, 2.5, 300)
for r, c, ls in ((1.5, PAL[2], "-"), (2.0, PAL[0], "--"), (3.0, PAL[7], "-.")):
    ax.plot(ss, np.exp(ss) * np.exp(-r * ss) / (r - 1), color=c, lw=1.3, ls=ls, label=f"$r={r:g}$")
ax.set_ylim(0, 6); ax.set_xlabel("$s$"); ax.set_ylabel("$e^sG(e^{-s})$"); ax.legend(loc="upper right", **LG); panel_label(ax, "d")
ax = axes[1, 1]
vv = np.linspace(0.02, 3, 400)
ax.plot(vv, vv, color=PAL[0], lw=1.3, label="$G(v)=v^2$")
ax.plot(vv, 2 * np.sqrt(vv), color=PAL[2], lw=1.3, ls="--", label="$G(v)=2v^{3/2}$")
ax.plot(vv, (vv * np.log(vv) - vv + 1) / vv, color="k", lw=1.0, ls=":", label="$G(v)=v\\log v-v+1$")
ax.set_ylim(0, 6.2); ax.set_xlabel("$v$"); ax.set_ylabel("$G(v)/v$"); ax.legend(loc="upper right", **LG); panel_label(ax, "e")

# ---- (f) the Pinsker constants for Gaussian laws against the prior N(0,1)
pts = []
xs = np.linspace(-12, 12, 48001)
for m in np.linspace(-1.5, 1.5, 13):
    for s in np.linspace(0.45, 1.3, 12):
        tv = float(np.sum(np.abs(norm.pdf(xs, m, s) - norm.pdf(xs))) * (xs[1] - xs[0]))
        pts.append((tv ** 2, (s ** 2 + m ** 2 - 1 - np.log(s ** 2)) / 2, np.exp(m ** 2 / (2 - s ** 2)) / (s * np.sqrt(2 - s ** 2)) - 1))
pts = np.array(pts); ok = pts[:, 1] > 1e-6
assert (pts[:, 0] <= 2 * pts[:, 1] + 1e-9).all() and (pts[:, 0] <= 2 / 3 * (pts[:, 1] + pts[:, 2]) + 1e-9).all()
ax = axes[1, 2]
hh = np.geomspace(1e-3, 30, 50)
ax.loglog(pts[ok, 1], pts[ok, 0], "o", ms=2.6, color=PAL[0], mfc="none", mew=0.6, label="$\\omega=0$")
ax.loglog(pts[ok, 1] + pts[ok, 2], pts[ok, 0], "s", ms=2.6, color=PAL[3], mfc="none", mew=0.6, label="$\\omega=1$")
ax.loglog(hh, 2 * hh, color=PAL[0], lw=1.0, ls="--", label="$2\\,\\mathcal{H}$")
ax.loglog(hh, 2 / 3 * hh, color=PAL[3], lw=1.0, ls="-.", label="$\\frac{2}{3}\\mathcal{H}$")
ax.axhline(4, color=MUTED, lw=0.6, ls=":")
ax.set_xlim(1e-3, 30); ax.set_ylim(1e-3, 6); ax.set_xlabel("regularizer $\\mathcal{H}_\\omega(\\mu)$"); ax.set_ylabel("$\\|\\mu-\\Gamma\\|_{\\mathrm{TV}}^2$")
ax.legend(loc="lower right", **LG); panel_label(ax, "f")
out["f"] = dict(laws=int(ok.sum()), largest_ratio_to_KL=float((pts[ok, 0] / pts[ok, 1]).max()), largest_ratio_to_H1=float((pts[ok, 0] / (pts[ok, 1] + pts[ok, 2])).max()))
(ROOT / "regularizer-geometry-v76.json").write_text(json.dumps(out, indent=1) + "\n")
save(fig, "regularizer-geometry-v76", ROOT)
print(json.dumps(out, indent=1))
