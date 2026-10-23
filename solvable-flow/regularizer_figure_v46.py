"""Revision 46: the flow-test figure of revision 41 with the paper's name for the regularizer
("relative entropy" instead of "KL") in the legend. Plotted data unchanged: the v41 module is imported,
only its label dictionary and its output directory are replaced."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "solvable-flow"))
import regularizer_figure_v41 as m  # noqa: E402
m.LABEL = {"KL": "relative entropy", "r1.5": "rel. entropy $+\\,\\omega G$, $r=3/2$",
           "r2": "rel. entropy $+\\,\\omega G$, $r=2$", "r3": "rel. entropy $+\\,\\omega G$, $r=3$"}
m.ROOT = HERE
if __name__ == "__main__":
    m.main()
