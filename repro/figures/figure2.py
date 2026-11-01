"""Fig. 2: (a) where the KL sits across noise levels for a fine-scale and a coarse-scale error;
(b) the log mass ratio of two modes recovered by each route, against the exact value; (c) the
GLRT's PD on three separated clusters by each route, and the learned Rao test's.

Usage: python -m figures.figure2 [--results DIR] [--out FILE]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.cli import REPRO
from experiments.experiment2 import FRACTIONS
from figures import style

GREEN, PURPLE = "#19a979", "#945ecf"
ROUTES = (("direct@0.02", "direct\n" + r"$\sigma_0{=}.02$", style.ORANGE),
          ("direct@0.3", "direct\n" + r"$\sigma_0{=}.3$", style.ORANGE),
          ("rect_energy@0.3", "rect.\n" + r"$\sigma_0{=}.3$", style.BLUE),
          ("classifier@0.3", "classif.\n" + r"$\sigma_0{=}.3$", GREEN),
          ("itd@0", "denois.\nerrors", PURPLE))
GLRTS = (("oracle@0.3", "exact GLRT", style.INK, "--"),
         ("direct@0.3", "direct GLRT", style.ORANGE, "-"),
         ("rect_energy@0.3", "rectangle GLRT", style.BLUE, "-"),
         ("classifier@0.3", "classifier GLRT", GREEN, "-"), ("rao@0.3", "Rao", PURPLE, ":"))


def kl_panel(ax: plt.Axes, kl: dict) -> None:
    """(a) Each error's share of the KL per unit log sigma."""
    for key, name, color in (("ripple", "fine-scale error (ripple)", style.ORANGE),
                             ("mass", "coarse-scale error (relative mass)", style.BLUE)):
        ax.plot(kl["sigmas"], kl[key], color=color, lw=1.3, label=name)
    ax.set_xscale("log")
    ax.set_xlabel(r"noise level $\sigma$")
    ax.set_ylabel(r"share of KL per unit $\log\sigma$")
    ax.legend(loc="upper left")
    ax.grid(True, which="major", color=style.GRID, lw=0.5)


def mass_panel(ax: plt.Axes, runs: list[dict]) -> None:
    """(b) Each route's log mass ratio per run, open for N = 2,000 and filled for 20,000."""
    for i, (key, _, color) in enumerate(ROUTES):
        for n, face, dx in ((2000, "white", -0.12), (20000, color, 0.12)):
            values = [r[key] for r in runs if r["N"] == n]
            ax.plot([i + dx] * len(values), values, "o", ms=4, mfc=face, mec=color, mew=1.0)
    ax.axhline(np.log(0.25), color=style.INK, lw=0.9, ls="--")
    ax.set_xticks(range(len(ROUTES)), [r[1] for r in ROUTES], fontsize=6.5)
    ax.set_ylabel("log mass ratio of the two modes")
    handles = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=style.MUTED, ms=4,
                          label=r"$N=2{,}000$"),
               plt.Line2D([], [], marker="o", ls="", color=style.MUTED, ms=4,
                          label=r"$N=20{,}000$"),
               plt.Line2D([], [], color=style.INK, ls="--", lw=0.9, label="exact")]
    ax.legend(handles=handles, loc="lower right")
    ax.grid(True, axis="y", color=style.GRID, lw=0.5)


def glrt_panel(ax: plt.Axes, runs: list[dict]) -> None:
    """(c) Each route's GLRT PD, and the Rao test's, mean and standard deviation over seeds."""
    for key, name, color, ls in GLRTS:
        v = np.array([r["pd"][key] for r in runs])
        ax.errorbar(FRACTIONS, v.mean(0), yerr=v.std(0, ddof=1), color=color, ls=ls, lw=1.2,
                    ms=3.5, marker="o" if ls != "--" else None, capsize=1.5, elinewidth=0.6,
                    label=name)
    ax.set_xlabel("amplitude / cluster distance")
    ax.set_ylabel(r"$P_{\mathrm{D}}$ at $P_{\mathrm{F}}=10^{-2}$")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=6.0, handlelength=1.4, borderaxespad=0.2,
              labelspacing=0.25)
    ax.grid(True, color=style.GRID, lw=0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=REPRO / "results" / "experiment2",
                        help="Experiment 2's output directory")
    parser.add_argument("--out", type=Path, default=REPRO / "results" / "figure2.pdf",
                        help="the figure file")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(style.WIDTH, 2.35), constrained_layout=True)
    kl_panel(axes[0], dict(np.load(args.results / "kl.npz")))
    mass_panel(axes[1], json.loads((args.results / "toy1d.json").read_text()))
    glrt_panel(axes[2], json.loads((args.results / "toy_gmm.json").read_text()))
    for ax, tag in zip(axes, "abc", strict=True):
        ax.set_title(f"({tag})", fontsize=8, loc="left")
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
