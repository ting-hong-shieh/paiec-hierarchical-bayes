"""One design system for every figure of the exp214 paper (IEEE and Kaggle versions).

Colours (also in figstyle.tex for the TikZ figures):
- ink / navy: titles, the pooled primary estimate, the retention heatmap scale;
- arms (Okabe-Ito): summary = reddish purple, no compaction = blue;
- workspace states (Okabe-Ito): passing = bluish green, edited-not-passing = vermilion, not yet edited = sky blue;
- greys for exploratory rows, grids and notes.

Font: Noto Sans inside figures (serif body text in the paper). The TrueType files come from the TeX Live
`noto` package that tectonic already downloads, so TikZ and matplotlib use the same font and the PDFs embed
TrueType (Type 42), not Type 3. If the files are missing, matplotlib falls back to DejaVu Sans.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

from matplotlib import font_manager

NAVY = "#1F3A5F"
NAVY_LIGHT = "#E8EDF4"   # card bands, primary row band
BAND_GREY = "#F3F4F6"    # exploratory row band
GRID = "#DCE1E8"
AXIS = "#6B7785"
TEXT = "#222222"
NOTE = "#5A6472"

BLUE = "#0072B2"         # no-compaction arm
PURPLE = "#CC79A7"       # summary arm / compaction
PURPLE_DARK = "#8E4F75"  # purple text
GREEN = "#009E73"        # passing
VERMILION = "#D55E00"    # edited, not passing
SKY = "#56B4E9"          # not yet edited
GREY = "#7A7A7A"         # exploratory rows

STATE_COLOUR = {"primary_not_passed": NAVY, "edited_not_passed": VERMILION, "not_edited": SKY, "passed": GREEN}
MINUS = "\u2013"  # Noto Sans 2.005 has no U+2212 minus; the en dash stands in, as is usual in sans figures


def _register_noto() -> str:
    """Register Noto Sans (all weights found) with matplotlib and return the family name to use."""
    dirs = [os.environ.get("NOTO_SANS_DIR", ""), str(Path(__file__).resolve().parent / "fonts")]
    dirs += glob.glob(os.path.expanduser("~/.cache/tectonic/bundles/data/*"))
    for d in dirs:
        if d and Path(d, "NotoSans-Regular.ttf").exists():
            for f in sorted(Path(d).glob("NotoSans-*.ttf")):
                font_manager.fontManager.addfont(str(f))
            return "Noto Sans"
    return "DejaVu Sans"


FAMILY = _register_noto()


def rc(fs: float) -> dict:
    """matplotlib rcParams for one base font size (pt)."""
    return {
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": [FAMILY, "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": fs, "axes.labelsize": fs, "axes.titlesize": fs,
        "xtick.labelsize": fs, "ytick.labelsize": fs, "legend.fontsize": fs,
        "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": AXIS, "ytick.color": TEXT,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
        "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlecolor": NAVY,
        "axes.titlepad": 4.0,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "hatch.color": "#B8BEC6", "hatch.linewidth": 0.5,
        "figure.constrained_layout.use": True,
        "savefig.pad_inches": 0.02,
        "axes.unicode_minus": False,  # the font has no U+2212; explicit tick labels use the en dash
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }


def pp(x: float, signed: bool = True) -> str:
    """Percentage points with one decimal, a real minus sign, and plain 0.0 for zero (as in Table II)."""
    s = f"{x:+.1f}" if signed else f"{x:.1f}"
    if s in ("+0.0", "-0.0"):
        s = "0.0"
    return s.replace("-", MINUS)
