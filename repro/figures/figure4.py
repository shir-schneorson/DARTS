"""Fig. 4: PD at PF = 1e-3 and partial AUC of the DARTS Rao test, the DARTS GLRT and the AMF against
the implant amplitude, mean over target types with the standard deviation as a band.

Usage: python -m figures.figure4 [--results DIR] [--out FILE]
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.cli import REPRO
from experiments.experiment4 import PLAN
from figures import style

BACKGROUNDS = {"pavia": "Pavia University (9 classes)",
               "muufl": "MUUFL campus 1 (4 cloth signatures)"}
LINES = {"Rao": ("DARTS, Rao", style.BLUE, "-", "o"),
         "GLRT": ("DARTS, GLRT", style.ORANGE, "-", "s"),
         "AMF": ("AMF", style.MUTED, "--", None)}
METRICS = (("pd", r"$P_{\mathrm{D}}$ at $P_{\mathrm{F}}=10^{-3}$", (-0.02, 1.02)),
           ("pauc", r"partial AUC to $P_{\mathrm{F}}=10^{-3}$", (0.495, 1.005)))


def panel(ax: plt.Axes, rows: dict, metric: str) -> list[float]:
    """One background's curves of `metric`; returns the amplitudes."""
    alphas = []
    for detector, (name, color, ls, marker) in LINES.items():
        alphas = sorted({a for d, a in rows if d == detector})
        values = [np.array([r[metric] for r in rows[(detector, a)]]) for a in alphas]
        mean = np.array([v.mean() for v in values])
        ax.plot(alphas, mean, ls=ls, color=color, marker=marker, ms=3.5,
                lw=1.0 if detector == "AMF" else 1.3, label=name)
        if detector != "AMF":
            sd = np.array([v.std(ddof=1) for v in values])
            ax.fill_between(alphas, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), color=color,
                            alpha=0.12, lw=0)
    return alphas


def load(path: Path, types: int) -> dict:
    """Rows of an Experiment 4 CSV by (detector, alpha), metrics as floats; refused unless every
    (detector, alpha) has all `types` target types."""
    rows = defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            rows[(r["detector"], float(r["alpha"]))].append({k: float(r[k])
                                                             for k in ("pd", "pauc")})
    if short := sorted(k for k, v in rows.items() if len(v) != types):
        raise ValueError(f"{path} is incomplete: {short} lack target types")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=REPRO / "results" / "experiment4",
                        help="Experiment 4's output directory")
    parser.add_argument("--out", type=Path, default=REPRO / "results" / "figure4.pdf",
                        help="the figure file")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    style.apply()
    fig, axes = plt.subplots(2, 2, figsize=(style.WIDTH, 4.2), constrained_layout=True)
    for col, (background, title) in enumerate(BACKGROUNDS.items()):
        rows = load(args.results / f"{background}.csv", len(PLAN[background]["types"]))
        for row, (metric, label, limits) in enumerate(METRICS):
            ax = axes[row, col]
            alphas = panel(ax, rows, metric)
            ax.set_xscale("log", base=2)
            ax.set_xticks(alphas, [f"{a:g}" for a in alphas])
            ax.minorticks_off()
            ax.grid(True, color=style.GRID, lw=0.5)
            ax.set_ylabel(label)
            ax.set_ylim(*limits)
        axes[0, col].set_title(title, fontsize=8)
        axes[1, col].set_xlabel(r"implant amplitude $\theta$")
    axes[0, 0].legend(loc="upper left")
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
