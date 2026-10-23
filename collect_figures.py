"""Copy the regenerated figures of the paper into a manuscript figure folder under the manuscript's names.

Usage:  python collect_figures.py <figures folder of the manuscript>
The mapping is that of the README's regeneration table; the three illustrations (logic, identity, curvature) come
from illustrations/. Every source must exist; nothing else in the target folder is touched.
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAP = {
    "logic": "illustrations/logic.pdf", "identity": "illustrations/identity.pdf", "curvature": "illustrations/curvature.pdf",
    "wear-frontier": "wear-allowance/wear-frontier-v47.pdf", "parameter-anchor": "wear-allowance/wear-anchor-v47.pdf",
    "wear-multiplier": "wear-allowance/wear-dual-v47.pdf", "wear-margin": "wear-allowance/wear-margin-v41.pdf",
    "wear-noise": "wear-allowance/wear-noise-v60.pdf", "fuel-tank": "fuel-tank/lunar-campaign-v41.pdf",
    "flow-test": "solvable-flow/regularizer-flow-v41.pdf", "prior": "solvable-flow/prior-v47.pdf",
    "prior-sweep": "solvable-flow/prior-flow.pdf", "particle-flow": "solvable-flow/particle-flow-v58.pdf",
    "roles-flow": "solvable-flow/roles-flow-v59.pdf", "corollary-bound": "solvable-flow/corollary-bound-v60.pdf",
    "safety": "output/figures/safety.pdf",
}

if __name__ == "__main__":
    target = Path(sys.argv[1]).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    for name, src in MAP.items():
        shutil.copyfile(HERE / src, target / f"{name}.pdf")
        print(f"{name}.pdf  <-  {src}")
