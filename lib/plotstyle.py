"""Shared figure style: validated categorical palette (fixed slot order), thin marks, recessive grid.

Revision 68: the figures are drawn at the journal's print width (the Springer template's text width,
372 pt = 5.15 in) with sans-serif lettering (Arial, falling back to Helvetica and DejaVu Sans) of 9 pt or
more on the canvas, so that a figure included at 0.92 of the text width prints its lettering at 8.3 pt or
more; the amsart documents, whose text width is 6.5 in, enlarge it. No panel carries a title (the
journal's rule; the captions carry the panel descriptions). Every series that a panel distinguishes by
colour also differs by marker or line style (`LS` and `MK` give the slot order).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # blue ramp light->dark
TEXT, TEXT2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"
NEUTRAL = "#8a8984"
LS = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1)), (0, (6, 2)), (0, (3, 1, 1, 1, 1, 1))]   # line styles by slot
MK = ["o", "s", "^", "D", "v", "P", "X", "*"]                                                        # markers by slot
FS = 9.0          # smallest lettering on the canvas (ticks, legends, annotations)
rcParams.update({
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS + 0.5, "legend.fontsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "lines.linewidth": 1.2, "lines.markersize": 4, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.borderaxespad": 0.2, "legend.labelspacing": 0.3,
    "legend.handletextpad": 0.5, "legend.columnspacing": 1.2,
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "mathtext.fontset": "stixsans",
    "axes.titleweight": "normal", "axes.titlelocation": "left", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})
TEXTW = 372 / 72.27   # the journal's text width in inches (Springer template, 372 pt)


def panel_label(ax, s, x=-0.02, y=1.03):
    ax.text(x, y, s, transform=ax.transAxes, ha="right", va="bottom", fontsize=FS + 1, fontweight="bold", color=TEXT)


def legend_below(fig, handles, labels=None, ncol=3, gap=0.03, axes=None, **kw):
    """One legend for the whole figure (or for the panels `axes`), in rows below them: the panels are too
    narrow at print width for legends of several entries. The legend is centred under the panels and placed
    under their lowest axis label."""
    if labels is None:
        labels = [h.get_label() for h in handles]
    axes = list(fig.axes) if axes is None else list(axes)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(r).transformed(fig.transFigure.inverted()) for ax in axes]
    bottom = min(b.y0 for b in boxes)
    xc = (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2
    opts = dict(loc="upper center", bbox_to_anchor=(xc, bottom - gap), ncol=ncol, frameon=False, fontsize=FS,
                handlelength=1.8, columnspacing=1.2, handletextpad=0.5)
    opts.update(kw)
    return fig.legend(handles, labels, **opts)


def save(fig, name, figdir):
    fig.savefig(figdir / f"{name}.pdf")
    fig.savefig(figdir / f"{name}.png")
    plt.close(fig)
