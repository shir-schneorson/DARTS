"""Fig. 5: against the noise level, the likelihood-ratio rate r(sigma)/alpha on noised held-out
samples, the non-convex share over PF (= beta) on clean held-out samples, and the Rao test's PD
at PF; the exact score dashed, the models fitted on n samples solid (mean over seeds). Dots mark
each model's sigma_hat and the PD there, the diamond the exact score's.

Usage: python -m figures.figure5 [--results FILE] [--out FILE]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.cli import REPRO
from experiments.experiment5 import NS, SEEDS
from figures import style


def panel(ax: plt.Axes, curves: dict, field: str, scale: float) -> None:
    """One field's curves: exact dashed, each n solid; Rao PD values are drawn as they are, the
    rates floored at 1e-3 for the log axis."""
    sigmas = curves["sigmas"]
    floor = (lambda v: v) if field == "pd" else (lambda v: np.maximum(v, 1e-3))
    ax.plot(sigmas, floor(curves[f"exact_{field}"] * scale), color=style.INK, lw=1.2, ls="--",
            label="exact")
    for n, color in zip(NS, style.BLUES, strict=True):
        mean = np.mean([curves[f"n{n}_s{s}_{field}"] for s in SEEDS], 0) * scale
        ax.plot(sigmas, floor(mean), color=color, lw=1.2, label=f"$n={n:,}$".replace(",", "{,}"))
        if field == "pd":
            for s in SEEDS:
                ax.plot(curves[f"n{n}_s{s}_sigma"], curves[f"n{n}_s{s}_pd_at_sigma"], "o", ms=4,
                        color=color, mec="white", mew=0.6, zorder=3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path,
                        default=REPRO / "results" / "experiment5" / "curves.npz",
                        help="Experiment 5's curves")
    parser.add_argument("--out", type=Path, default=REPRO / "results" / "figure5.pdf",
                        help="the figure file")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    curves = dict(np.load(args.results))
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(style.WIDTH, 2.4), constrained_layout=True)
    panels = (("lr_rate", r"likelihood-ratio rate $r(\sigma)/\alpha$", 1.0),
              ("share", r"non-convex share / $P_{\mathrm{F}}$", 1 / float(curves["pf"])),
              ("pd", r"$P_{\mathrm{D}}$ of the Rao test", 1.0))
    for ax, (field, label, scale) in zip(axes, panels, strict=True):
        panel(ax, curves, field, scale)
        if field == "pd":
            ax.plot(curves["exact_sigma"], curves["exact_pd_at_sigma"], "D", ms=4, color=style.INK,
                    mec="white", mew=0.6, zorder=4)
            shown = np.concatenate([lines.get_ydata() for lines in ax.get_lines()])
            ax.set_ylim(min(0.55, shown.min() - 0.01), max(0.8, shown.max() + 0.01))
        else:
            ax.set_yscale("log")
            ax.axhline(1, color=style.MUTED, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"noise level $\sigma$")
        ax.set_ylabel(label)
        ax.grid(True, which="major", color=style.GRID, lw=0.5)
    axes[0].legend(fontsize=7, loc="upper right")
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
