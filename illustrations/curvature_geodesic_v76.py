"""Revision 76: the curvature budget of the convergence theorem along a geodesic of laws (the figure of Sect. 4 of the
paper, in place of the toy function of concept_figures.py).

The model is the decision problem of the analytical example of the paper with another feature. A unit has one real
parameter x, the policy is the exponential tilt with the tilt c(mu) = int varphi dmu, and the clipped value is
V_m(mu) = V_m(c(mu)) with the value curve of the example (regularizer-flow.json of the solvable-flow folder), which
increases in the tilt. The feature is varphi = g * N(0, W^2), the smoothing by a Gaussian kernel of
    g(x) = BETA (1 + cos(OM0 (x - XC)))/2 + (1 - BETA) (1 - 2 frac((x - X1)/ELL))^2,
with frac the fractional part. g has values in [0, 1], so varphi has values in [0, 1] and is smooth with bounded
derivatives. The second term of g is a parabola repeated with the period ELL. It has sharp peaks at X1 + k ELL and
parabolic valleys between them, and the first term is a slow envelope that makes the peaks differ in height. Sharp
peaks and parabolic valleys give -V_m sharp wells and broad barriers, the shape with the least concavity for barriers
of a given height. For a Gaussian law the tilt has the closed form
    c(N(a, s^2)) = BETA (1 + exp(-OM0^2 t/2) cos(OM0 (a - XC)))/2
                   + (1 - BETA) (1/3 + (4/pi^2) sum_k exp(-(2 pi k/ELL)^2 t/2) cos(2 pi k (a - X1)/ELL) / k^2),  t = W^2 + s^2,
from the Fourier series (1 - 2 frac(u))^2 = 1/3 + (4/pi^2) sum_k cos(2 pi k u)/k^2. The prior is Gamma = N(0, 1) and the
regularizer is kappa KL(mu || Gamma), with KL(N(a, s^2) || Gamma) = (a^2 + s^2 - 1)/2 - log s.

The geodesic joins the Gaussian laws mu_0 = N(A0, S0^2) and mu_1 = N(A1, S1^2) and passes through the prior at the
position PRIOR. In one dimension the generalized geodesic between two laws, with any base, is the Wasserstein-2
geodesic, here mu_theta = N((1-theta) A0 + theta A1, ((1-theta) S0 + theta S1)^2), and
W_2(mu_0, mu_1)^2 = (A1-A0)^2 + (S1-S0)^2. A functional phi is lambda-convex along the geodesic when
d^2/dtheta^2 phi(mu_theta) >= lambda W_2^2; the relative entropy is 1-convex along it (lambda_W = 1).
(a) -V_m and the regularized objective along the geodesic, at 0.3 times and at 1.5 times kappa_geo, the smallest
    strength at which the objective is convex along this geodesic. The threshold of the convergence theorem makes the
    objective convex along every generalized geodesic, so kappa_geo is at most that threshold. The strip below shows
    the density of mu_theta at five positions, each with the name of its law.
(b) The position of the minimizer of the regularized objective among the laws of the geodesic, against the strength.
Output: curvature-geodesic-v76 (pdf/png) and curvature-geodesic-v76.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, panel_label, legend_below, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

PRIOR, LENGTH, S1, MODERATE = 0.15, 24.0, 0.7, 0.3
BETA, W = 0.70, 0.3                                          # weight of the envelope, width of the smoothing kernel
SPACING, FIRST, CENTRE, PERIOD = 0.33, 0.25, 0.5, 1.8       # in units of the position: spacing of the peaks, the first peak, centre and period of the envelope
A0, A1 = -PRIOR * LENGTH, (1 - PRIOR) * LENGTH               # the mean is 0 at the position PRIOR
S0 = 1 + PRIOR * (1 - S1) / (1 - PRIOR)                      # the standard deviation is 1 at the position PRIOR
W2SQ = (A1 - A0) ** 2 + (S1 - S0) ** 2
OM0, ELL = 2 * np.pi / (PERIOD * LENGTH), SPACING * LENGTH
XC, X1 = (CENTRE - PRIOR) * LENGTH, (FIRST - PRIOR) * LENGTH
d = json.loads((ROOT.parent / "solvable-flow" / "regularizer-flow.json").read_text())
CURVE_C, CURVE_V = np.array(d["curve"]["c"]), np.array(d["curve"]["V_m"])
assert (np.diff(CURVE_V) > 0).all()                           # the value increases in the tilt
mean = lambda th: (1 - th) * A0 + th * A1
sdev = lambda th: (1 - th) * S0 + th * S1
KS = np.arange(1, 400)[:, None]


def smoothed(a, t):
    """int g(a + sqrt(t) z) N(dz): the feature varphi at a for t = W^2, the tilt of N(a, s^2) for t = W^2 + s^2."""
    a = np.atleast_1d(np.asarray(a, float)); t = np.broadcast_to(np.asarray(t, float), a.shape)
    peaks = 1 / 3 + (4 / np.pi ** 2) * (np.exp(-(2 * np.pi * KS / ELL) ** 2 * t / 2) * np.cos(2 * np.pi * KS * (a - X1) / ELL) / KS ** 2).sum(0)
    return BETA * (1 + np.exp(-OM0 ** 2 * t / 2) * np.cos(OM0 * (a - XC))) / 2 + (1 - BETA) * peaks


feature = lambda x: smoothed(x, W ** 2)
tilt = lambda th: smoothed(mean(th), W ** 2 + sdev(th) ** 2)
neg_value = lambda th: -np.interp(tilt(th), CURVE_C, CURVE_V)                         # -V_m(mu_theta)
entropy = lambda th: (mean(th) ** 2 + sdev(th) ** 2 - 1) / 2 - np.log(sdev(th))       # KL(mu_theta || Gamma)
assert abs(mean(PRIOR)) < 1e-12 and abs(sdev(PRIOR) - 1) < 1e-12 and abs(entropy(PRIOR)) < 1e-12
# the feature against a direct smoothing of g, and the closed form of the tilt against quadrature, at three positions
Z = np.linspace(-10.0, 10.0, 400001); PHI = np.exp(-Z ** 2 / 2) / np.sqrt(2 * np.pi)
g_raw = lambda x: BETA * (1 + np.cos(OM0 * (x - XC))) / 2 + (1 - BETA) * (1 - 2 * np.mod((x - X1) / ELL, 1.0)) ** 2
XS = np.linspace(A0 - 12.0, A1 + 12.0, 6001); FEAT = feature(XS)
assert FEAT.min() >= 0 and FEAT.max() <= 1
for th_ in (0.0, 0.37, 1.0):
    assert abs(np.trapezoid(g_raw(mean(th_) + W * Z) * PHI, Z) - feature(mean(th_))[0]) < 1e-6
    assert abs(np.trapezoid(np.interp(mean(th_) + sdev(th_) * Z, XS, FEAT) * PHI, Z) - tilt(np.array([th_]))[0]) < 1e-6

TH = np.linspace(0.0, 1.0, 8001)
f = neg_value(TH); h = entropy(TH)
f2 = np.gradient(np.gradient(f, TH), TH); h2 = np.gradient(np.gradient(h, TH), TH)
inner = slice(40, -40)                                        # the numerical second derivative is not used at the ends
assert (h2[inner] >= W2SQ - 1e-4).all()                       # relative entropy is 1-convex along the geodesic
deficit = float(-f2[inner].min() / W2SQ)                      # curvature deficit of -V_m along this geodesic, per unit of W_2^2
k_star = float((-f2[inner] / h2[inner]).max())                # the smallest strength at which the sum is convex along it
local_min = lambda y: np.where((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:]))[0] + 1

fig = plt.figure(figsize=(TEXTW, 3.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.2], hspace=0.72, wspace=0.32)
ax = fig.add_subplot(gs[0, 0]); strip = fig.add_subplot(gs[1, 0], sharex=ax)
# the strip below panel (a): the density of mu_theta at five positions, on the scale of the parameter x. The mean of mu_theta is
# A0 + theta (A1 - A0), so the parameter axis and the position axis are aligned and each density stands below its position.
to_u = lambda x: (x - A0) / (A1 - A0)
SHOWN = (0.0, PRIOR, 0.42, 0.7, 1.0)
peak = 1 / (S1 * np.sqrt(2 * np.pi))
for th in SHOWN:
    xs = mean(th) + sdev(th) * np.linspace(-3.2, 3.2, 200)
    dens = np.exp(-(xs - mean(th)) ** 2 / (2 * sdev(th) ** 2)) / (sdev(th) * np.sqrt(2 * np.pi))
    col = TEXT2 if abs(th - PRIOR) < 1e-9 else PAL[0]
    strip.fill_between(to_u(xs), 0, dens / peak, color=col, alpha=0.25, lw=0, clip_on=False)
    strip.plot(to_u(xs), dens / peak, color=col, lw=1.0, clip_on=False)
    name = {0.0: "$\\mu_0$", PRIOR: "$\\Gamma$", 1.0: "$\\mu_1$"}.get(th, f"$\\mu_{{{th:g}}}$")
    strip.text(th, S1 / sdev(th) + 0.18, name, ha="center", va="bottom", fontsize=FS, color=col, clip_on=False)
strip.set_ylim(0, 1.0)
for side in ("left", "right", "bottom"):
    strip.spines[side].set_visible(False)
strip.grid(False); strip.yaxis.set_visible(False); strip.tick_params(axis="x", bottom=False, labelbottom=False)
sec = strip.secondary_xaxis(-0.08, functions=(lambda u: A0 + u * (A1 - A0), to_u))
sec.set_xticks([0, 10, 20]); sec.tick_params(labelsize=FS - 1, length=2, pad=1, colors=TEXT2); sec.spines["bottom"].set_color(MUTED)
sec.set_xlabel("parameter $x$", fontsize=FS - 1, color=TEXT2, labelpad=2)
ax.plot(TH, f - f.min(), color=TEXT2, lw=1.2, label="$-V_m(\\mu_\\vartheta)$")
k0 = local_min(f); ax.plot(TH[k0], (f - f.min())[k0], marker="o", color=TEXT2, ms=3.5, ls="none")
marked = {"0": dict(local_minimizers=[round(float(x), 3) for x in TH[k0]])}
for mult, col, ls, lab in ((MODERATE, PAL[1], "--", f"$-V_m+\\kappa\\mathcal{{H}}$, $\\kappa={MODERATE:g}\\,\\kappa_{{\\mathrm{{geo}}}}$: not convex,\none minimizer"),
                           (1.5, PAL[0], "-.", "$-V_m+\\kappa\\mathcal{H}$, $\\kappa=1.5\\,\\kappa_{\\mathrm{geo}}$: convex")):
    y = f + mult * k_star * h
    ax.plot(TH, y - y.min(), color=col, lw=1.2, ls=ls, label=lab)
    k = local_min(y)
    ax.plot(TH[k], (y - y.min())[k], marker="o", color=col, ms=3.5, ls="none")
    y2 = np.gradient(np.gradient(y, TH), TH)[inner]
    marked[f"{mult:g}"] = dict(convex=bool((y2 >= -1e-5).all()), local_minimizers=[round(float(x), 3) for x in TH[k]],
                               global_minimizer=round(float(TH[int(np.argmin(y))]), 3))
assert not marked[f"{MODERATE:g}"]["convex"] and len(marked[f"{MODERATE:g}"]["local_minimizers"]) == 1
assert marked["1.5"]["convex"] and len(marked["1.5"]["local_minimizers"]) == 1 and len(k0) == 3
ax.axvline(PRIOR, color=MUTED, lw=0.7, ls=":")
YTOP = 1.6 * float((f - f.min()).max())
ax.set_ylim(0, YTOP); ax.set_xlim(-0.08, 1.08); ax.text(PRIOR + 0.02, 0.92 * YTOP, "prior", fontsize=FS, color=TEXT2)
ax.set_xticks([0.0, PRIOR, 0.5, 1.0]); ax.set_xticklabels(["$0$", "$0.15$", "$0.5$", "$1$"])
ax.set_xlabel("position $\\vartheta$ on the geodesic $\\vartheta\\mapsto\\mu_\\vartheta$", labelpad=2); ax.set_ylabel("objective (shifted)")
panel_label(ax, "(a)")
ax = fig.add_subplot(gs[0, 1]); panel_label(ax, "(b)")
kappas = np.geomspace(1e-3 * k_star, 10 * k_star, 1600)
mins = np.array([TH[int(np.argmin(f + k * h))] for k in kappas])
ax.plot(kappas / k_star, mins, color=PAL[0])
jumps = [dict(strength_in_units_of_kappa_geo=round(float(kappas[i] / k_star), 4), from_position=round(float(mins[i]), 3), to_position=round(float(mins[i + 1]), 3))
         for i in np.where(np.abs(np.diff(mins)) > 0.08)[0]]
ax.axvline(1.0, color=PAL[1], lw=0.8, ls=":")
ax.text(0.8, 0.84, "convex along the\ngeodesic from $\\kappa_{\\mathrm{geo}}$ on", fontsize=FS, color=PAL[1], ha="right")
ax.axhline(PRIOR, color=MUTED, lw=0.7, ls=":"); ax.text(1.3e-3, PRIOR - 0.09, "prior $\\Gamma$", fontsize=FS, color=TEXT2)
ax.axhline(float(TH[np.argmin(f)]), color=MUTED, lw=0.7, ls="--")
ax.axhline(float(TH[k0[0]]), color=MUTED, lw=0.7, ls="--")
ax.text(1.3e-3, float(TH[k0[0]]) + 0.025, "nearest well", fontsize=FS, color=TEXT2, va="bottom")
ax.text(1.3e-3, float(TH[np.argmin(f)]) + 0.025, "minimizer of $-V_m$", fontsize=FS, color=TEXT2, va="bottom")
ax.set_xscale("log"); ax.set_ylim(0, 1.03)
ax.set_xlabel("strength $\\kappa/\\kappa_{\\mathrm{geo}}$"); ax.set_ylabel("minimizing position $\\vartheta$")
# the legend of panel (a) stands under panel (b), beside the strip
fig.canvas.draw(); box = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(box.x0 + 0.02, box.y0 - 0.03), ncol=1, frameon=False, fontsize=FS, handlelength=1.8)
R = float(f.max() - f.min()); height = lambda i: round(float((f[i] - f.min()) / R), 3)
at = lambda mult: int(np.argmin(f + mult * k_star * h))
out = dict(feature="g * N(0, W^2), g(x) = BETA (1 + cos(OM0 (x - XC)))/2 + (1 - BETA) (1 - 2 frac((x - X1)/ELL))^2",
           feature_parameters=dict(BETA=BETA, W=W, OM0=round(OM0, 5), ELL=round(ELL, 4), XC=round(XC, 4), X1=round(X1, 4)),
           heights_as_shares_of_the_range=dict(prior=height(int(np.argmin(np.abs(TH - PRIOR)))), wells=[height(i) for i in k0],
                                               minimizer_at_the_moderate_strength=height(at(MODERATE)), minimizer_at_1_5_times_kappa_geo=height(at(1.5))), mu_0=[round(A0, 4), round(S0, 4)], mu_1=[round(A1, 4), S1], prior_position=PRIOR, positions_of_the_densities=list(SHOWN),
           W2_squared=round(W2SQ, 4), range_of_minus_V=round(float(f.max() - f.min()), 4), curvature_deficit_along_the_geodesic=round(deficit, 4),
           kappa_geo=round(k_star, 4), marked=marked, jumps_of_the_minimizer=jumps)
(ROOT / "curvature-geodesic-v76.json").write_text(json.dumps(out, indent=1) + "\n")
save(fig, "curvature-geodesic-v76", ROOT)
print(json.dumps(out))
