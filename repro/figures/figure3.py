"""Fig. 3: partial AUC and PD at PF = 1e-3 against the target amplitude on the synthetic spatial
scene, mean over seeds with the standard deviation as a band: the exact detectors dashed or dotted,
DART and DARTS solid, each with its Rao test and its GLRT.

Usage: python -m figures.figure3 [--results DIR] [--out FILE]
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.cli import REPRO
from figures import style

PURPLE = "#945ecf"
LINES = (("oracle/rao", "Rao, exact scene density", style.MUTED, "--", None),
         ("oracle/glrt", "GLRT, exact scene density", style.INK, "--", None),
         ("oracle_context/rao", "Rao given the true context, normalized", style.INK, ":", None),
         ("dart/rao", "DART, Rao", PURPLE, "-", "o"),
         ("dart/glrt", "DART, GLRT", PURPLE, "-.", "s"),
         ("darts/rao", "DARTS, Rao", style.BLUE, "-", "o"),
         ("darts/glrt", "DARTS, GLRT", style.ORANGE, "-.", "s"))
METRICS = (("pauc", r"partial AUC to $P_{\mathrm{F}}=10^{-3}$", (0.495, 1.005)),
           ("pd", r"$P_{\mathrm{D}}$ at $P_{\mathrm{F}}=10^{-3}$", (-0.02, 1.02)))


def load(folder: Path) -> dict:
    """Every seed file's rows by (row, alpha), metrics as floats; the seeds found are printed."""
    rows, paths = defaultdict(list), sorted(folder.glob("seed*.csv"))
    for path in paths:
        with path.open() as f:
            for r in csv.DictReader(f):
                rows[(r["row"], float(r["alpha"]))].append({k: float(r[k])
                                                            for k in ("pd", "pauc")})
    print(f"averaging over {len(paths)} seeds: {[p.stem for p in paths]}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=REPRO / "results" / "experiment3",
                        help="Experiment 3's output directory")
    parser.add_argument("--out", type=Path, default=REPRO / "results" / "figure3.pdf",
                        help="the figure file")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = load(args.results)
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH, 3.1), constrained_layout=True)
    for ax, (metric, label, limits) in zip(axes, METRICS, strict=True):
        for row, name, color, ls, marker in LINES:
            alphas = sorted(a for r, a in rows if r == row)
            values = [np.array([x[metric] for x in rows[(row, a)]]) for a in alphas]
            mean = np.array([v.mean() for v in values])
            ax.plot(alphas, mean, ls=ls, color=color, marker=marker, ms=3.2, lw=1.2, label=name)
            if marker and min(len(v) for v in values) > 1:
                sd = np.array([v.std(ddof=1) for v in values])
                ax.fill_between(alphas, mean - sd, mean + sd, color=color, alpha=0.12, lw=0)
        ax.set_xlabel(r"target amplitude $\theta$")
        ax.set_ylabel(label)
        ax.set_ylim(*limits)
        ax.grid(True, color=style.GRID, lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=6.8)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
