"""Shared matplotlib style for the poster figures (same font files as poster.tex)."""
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

POSTER = Path(__file__).resolve().parents[1]
ROOT = POSTER.parent
FIG = POSTER / "figures"

ACCENT = "#1F4E79"        # deep blue: our systems
ACCENT_LIGHT = "#7FA3C6"  # second "ours" series (e.g. router, crops)
GREY_DARK = "#6E6E6E"     # reference baseline
GREY = "#B4B4B4"          # other baselines
INK = "#1A1A1A"

# Figure widths in inches, matched to the poster column (A1 landscape, 3 columns).
COL_W = 10.0
HALF_W = 4.85

FONT_SIZE = 19


def apply():
    for f in (POSTER / "fonts").glob("FiraSans-*.ttf"):
        font_manager.fontManager.addfont(str(f))
    mpl.rcParams.update({
        "font.family": "Fira Sans",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=110)  # preview only
