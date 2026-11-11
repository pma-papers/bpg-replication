"""Revision 76: the setup of the analytical example in one figure (no new runs).

(a) The decision: one action a in [0,1] with utility a; the budget is exhausted when a < eps s_2 for the uniform
    draw s_2, so the failure probability of the action a is max(0, 1 - a/eps). The represented policies are the
    exponential tilts pi_c(da) ~ exp(c a) da of the uniform reference, c in [0,1); the densities at c = 0 and c = 1.
(b) The parametrization: the feature varphi vanishes on [-1,1] (the plateau); the prior N(0,1), the initial law
    (the prior conditioned on [-1/2,1/2]) and the minimizers of the regularized objective at two strengths
    (relative entropy alone; the final densities of the stored flows, regularizer-flow.json).
(c) The outcome as a function of the tilt c = int varphi dmu: clipped value V_m and geometric risk q_beta, with the
    tilts of the initial law, the prior and the two minimizers marked.
Reads regularizer-flow.json (written by the convergence test). Output: example-setup-v76 (pdf/png) and
example-setup-v76.json with the marked values.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, panel_label, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from density_flow import chi as feature, phi_gauss  # noqa: E402

d = json.loads((ROOT / "regularizer-flow.json").read_text())
EPS = d["params"]["eps"]
curve = {k: np.array(v) for k, v in d["curve"].items()}
RUNS = [("k1/KL", "$\\kappa=1$", PAL[0], "s"), ("k0.1/KL", "$\\kappa=0.1$", PAL[1], "^")]

fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.05), gridspec_kw=dict(wspace=0.78))

# (a) the decision
ax = axes[0]
a = np.linspace(0, 1, 401)
ax.plot(a, a, color=TEXT2, lw=1.2, label="utility $u=a$")
ax.plot(a, np.maximum(0, 1 - a / EPS), color=PAL[7], lw=1.2, ls="--", label="failure probability")
ax.axvspan(0, EPS, color=PAL[7], alpha=0.07, lw=0)
for c, ls in ((0.0, ":"), (1.0, "-.")):
    dens = np.exp(c * a) * (c / np.expm1(c) if c > 0 else 1.0)
    ax.plot(a, dens, color=PAL[0], lw=1.0, ls=ls, label=f"action density, $c={c:g}$")
ax.set_xlabel("first action $a$"); ax.set_ylim(0, 1.7); ax.set_xlim(0, 1)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), fontsize=FS - 1.5, ncol=1, handlelength=1.6)
panel_label(ax, "a")

# (b) the parametrization
ax = axes[1]
x = np.linspace(-4.5, 4.5, 1801)
ax.plot(x, feature(x), color="k", lw=1.2, label="feature $\\varphi$")
ax.plot(x, phi_gauss(x), color=MUTED, lw=1.0, ls="--", label="prior $N(0,1)$")
p0 = np.where(np.abs(x) <= 0.5, phi_gauss(x), 0.0); p0 /= np.trapezoid(p0, x)
ax.fill_between(x, 0, p0, color=MUTED, alpha=0.35, lw=0, label="initial law")
marks = [("initial law", 0.0, MUTED, "o"), ("prior", d["c_Gamma"], TEXT2, "D")]
for key, lab, col, mk in RUNS:
    r = d["runs"][key]
    th = np.array(r["density"]["theta"]); p = np.array(r["density"]["p_final"])
    ax.plot(th, p, color=col, lw=1.1, label=f"minimizer, {lab}")
    marks.append((f"minimizer, {lab}", r["records"][-1]["c"], col, mk))
ax.set_xlim(-4.5, 4.5); ax.set_ylim(0, 1.12); ax.set_xlabel("parameter $x$ of a unit"); ax.set_xticks([-4, -2, 0, 2, 4])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), fontsize=FS - 1.5, ncol=1, handlelength=1.6)
panel_label(ax, "b")

# (c) clipped value and risk against the tilt
ax = axes[2]
ax.plot(curve["c"], curve["V_m"], color="k", lw=1.2)
ax.set_xlabel("tilt $c=\\int\\varphi\\,d\\mu$"); ax.set_ylabel("clipped value $V_m$")
ax2 = ax.twinx(); ax2.spines["right"].set_visible(True); ax2.grid(False)
ax2.plot(curve["c"], curve["q"], color=PAL[7], lw=1.2, ls="--")
ax2.set_ylabel("geometric risk $q_\\beta$", color=PAL[7]); ax2.tick_params(axis="y", colors=PAL[7])
out = {}
for lab, c, col, mk in marks:
    v = float(np.interp(c, curve["c"], curve["V_m"])); q = float(np.interp(c, curve["c"], curve["q"]))
    ax.plot([c], [v], marker=mk, ms=5, color=col, mec="white", mew=0.6, ls="none", label=lab, zorder=4)
    out[lab] = dict(c=float(c), V_m=v, q_beta=q, U=float(np.interp(c, curve["c"], curve["U"])))
ax.set_xlim(-0.03, 1.0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), fontsize=FS - 1.5, ncol=1, handlelength=1.0)
panel_label(ax, "c")
out["ends"] = dict(c0=dict(V_m=float(curve["V_m"][0]), q_beta=float(curve["q"][0]), U=float(curve["U"][0])),
                   c1=dict(V_m=float(curve["V_m"][-1]), q_beta=float(curve["q"][-1]), U=float(curve["U"][-1])))
(ROOT / "example-setup-v76.json").write_text(json.dumps(out, indent=1) + "\n")
save(fig, "example-setup-v76", ROOT)
print(json.dumps(out, indent=1))
