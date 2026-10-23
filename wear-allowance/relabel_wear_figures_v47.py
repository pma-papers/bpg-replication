"""Revision 47: the wear-allowance figures of revision 41 under the manuscript's names, with the paper's name for
the action relative entropy toward the incumbent rule ("action regularization" instead of "action anchor").

Revision 68: the renamed labels, the print-width layout and the sans-serif lettering are now in the revision-41
modules themselves (analyze_v41.py, anchor_analyze_v41.py), so this script only draws the three figures with those
modules and saves them under the revision-47 names: wear-anchor-v47 (the parameter-anchor figure of the paper),
wear-dual-v47 (the multiplier figure of Online Resource 1) and wear-frontier-v47 (the wear-allowance figure of the
paper). The SHA-256 digest of every plotted array is printed, so that a run can be compared with the stored
revision-41 digests (wear-digests-v41.json) and with earlier runs. Nothing else in this folder is written: the
modules' other outputs go to _scratch/.
Run: ~/.venvs/constrained-rl/bin/python relabel_wear_figures_v47.py
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import anchor_analyze_v41 as anchor  # noqa: E402
import analyze_v41 as wear  # noqa: E402


def fig_digest(fig):
    h = hashlib.sha256()
    for ax in fig.axes:
        for ln in ax.get_lines():
            for a in (ln.get_xdata(), ln.get_ydata()):
                a = np.ascontiguousarray(np.asarray(a, dtype=float)); h.update(a.tobytes())
        for c in ax.collections:
            off = np.ascontiguousarray(np.asarray(c.get_offsets(), dtype=float)); h.update(off.tobytes())
            for p in c.get_paths():
                h.update(np.ascontiguousarray(p.vertices, dtype=float).tobytes())
    return h.hexdigest()


def run(module, call, name_old, name_new):
    captured = {}

    def save(fig, name, root):
        if name == name_old:
            captured["digest"] = fig_digest(fig)
            captured["labels"] = sorted({l.get_label() for ax in fig.axes for l in ax.get_lines() if not l.get_label().startswith("_")})
            fig.savefig(HERE / f"{name_new}.pdf", bbox_inches="tight")
            fig.savefig(HERE / f"{name_new}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    module.save = save
    module.ROOT = HERE / "_scratch"
    module.ROOT.mkdir(exist_ok=True)
    call(module)
    print(name_new, "data digest", captured["digest"][:16], "labels:", captured["labels"])


def call_anchor(m):
    m.main()


def call_frontier(m):
    bench, protocol, runs = m.load()
    summary, _ = m.summarize(bench, protocol, runs)
    m.fig_frontier(bench, summary)


def call_dual(m):
    bench, protocol, runs = m.load()
    summary, _ = m.summarize(bench, protocol, runs)
    m.fig_dual(summary, runs, protocol)


if __name__ == "__main__":
    # anchor_analyze reads the matched campaign from ROOT/wear-anchor: provide it in the scratch root
    (HERE / "_scratch").mkdir(exist_ok=True)
    link = HERE / "_scratch" / "wear-anchor"
    if not link.exists():
        link.symlink_to(HERE / "wear-anchor")
    run(anchor, call_anchor, "wear-anchor-v41", "wear-anchor-v47")
    run(wear, call_dual, "wear-dual-v41", "wear-dual-v47")
    run(wear, call_frontier, "wear-frontier-v41", "wear-frontier-v47")
