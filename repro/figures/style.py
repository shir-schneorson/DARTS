"""The figures' shared look: ink, muted axes, light grid, 8 pt type, embedded TrueType fonts."""

import matplotlib.pyplot as plt

INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
BLUE, ORANGE = "#2a6fdb", "#e8743b"
BLUES = ("#9ecae1", "#6baed6", "#3182bd", "#08519c")
WIDTH = 7.16  # inches, a two-column figure


def apply() -> None:
    """Set the shared rcParams."""
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False,
                         "legend.fontsize": 7, "pdf.fonttype": 42})
