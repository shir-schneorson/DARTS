"""Fig. 2: pAUC and PD at PF = 1e-3 against the target amplitude on the synthetic scene
(Experiment 2), mean over seeds with one standard deviation as a band. The exact detectors are
grey or black: the scene density's Rao test and GLRT dashed, the Rao test given the true region
r unsmoothed (dash-dot-dot) and smoothed at the matched level (dotted). DART (purple) and DARTS
(green) are solid (Rao test) and dash-dotted (plug-in GLRT), the GLRT in a lighter tint.

Usage, from repro/: python -m paper.figure2 [--results DIR] [--out FILE]
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper import style

REPRO = Path(__file__).resolve().parents[1]
PURPLE, LIGHT_PURPLE, GREEN, LIGHT_GREEN = "#945ecf", "#b994e6", "#1b9e77", "#7fc9b0"
LINES = (("oracle Rao", r"Rao, $p_0(\mathbf{x})$", style.MUTED, "--", None),
         ("oracle GLRT", r"GLRT, $p_0(\mathbf{x})$", style.INK, "--", None),
         ("region Rao oracle", r"Rao, $p_0(\mathbf{x}\mid\mathbf{r})$", style.MUTED,
          (0, (5, 1.5, 1, 1.5, 1, 1.5)), None),
         ("region Rao oracle, matched level",
          r"Rao, $p_{\sigma_{\mathrm{m}}}(\mathbf{x}\mid\mathbf{r})$", style.INK, ":", None),
         ("DART Rao", "DART, Rao", PURPLE, "-", "o"),
         ("DART GLRT", "DART, GLRT", LIGHT_PURPLE, "-.", "s"),
         ("DARTS Rao", "DARTS, Rao", GREEN, "-", "o"),
         ("DARTS GLRT", "DARTS, GLRT", LIGHT_GREEN, "-.", "s"))
METRICS = (("pauc", r"partial AUC to $P_{\mathrm{F}}=10^{-3}$", (0.495, 1.005)),
           ("pd", r"$P_{\mathrm{D}}$ at $P_{\mathrm{F}}=10^{-3}$", (-0.02, 1.02)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=REPRO / "results" / "experiment2")
    parser.add_argument("--out", type=Path, default=REPRO / "outputs" / "paper" / "figure2.pdf")
    args = parser.parse_args()
    values = defaultdict(list)  # (detector, alpha, metric) -> one value per seed
    for r in csv.DictReader((args.results / "runs.csv").open()):
        for metric, _, _ in METRICS:
            values[r["detector"], float(r["alpha"]), metric].append(float(r[metric]))
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH, 3.1), layout="constrained")
    for ax, (metric, label, limits) in zip(axes, METRICS, strict=True):
        for detector, name, color, ls, marker in LINES:
            alphas = sorted(a for d, a, m in values if d == detector and m == metric)
            seeds = np.array([values[detector, a, metric] for a in alphas])  # (alphas, seeds)
            mean, sd = seeds.mean(1), seeds.std(1, ddof=1)
            cells = zip(alphas, mean, sd, strict=True)
            print(f"{metric} {detector}:", *(f"{a:g} {m:.3f}±{s:.3f}" for a, m, s in cells))
            ax.plot(alphas, mean, ls=ls, color=color, marker=marker, ms=3.2, lw=1.2, label=name)
            if marker:
                ax.fill_between(alphas, mean - sd, mean + sd, color=color, alpha=0.12, lw=0)
        ax.set_xlabel(r"target amplitude $\theta$")
        ax.set_ylabel(label)
        ax.set_ylim(*limits)
        ax.grid(True, color=style.GRID, lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=6.8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
