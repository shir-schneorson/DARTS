"""Fig. 4: the noise-level rule on one signature per background (Experiment 4's figure4.json), one
column per background, three amplitudes each, one Spectral_r shade per amplitude from the weakest.
Top: the screened response against sigma, divided by its maximum, of the scene (solid) and of the
matched reference r_{theta-hat,rho-hat} (dashed), over the range of the candidate implants (gray).
Bottom: the scene's PD at PF = 1e-3 (solid), the held-out PD averaged over [theta-hat / 2,
theta-hat] that the rule maximizes (dashed), and sigma-hat (dot).

Usage, from repro/: python -m paper.figure4 [--results DIR] [--out FILE]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator

from paper import style

REPRO = Path(__file__).resolve().parents[1]
BAND, DASH = "#eeeeee", (0, (3.5, 2))


def draw(top: plt.Axes, bottom: plt.Axes, d: dict, shades: tuple) -> None:
    """One background's column: the responses on top, the detection probabilities below."""
    sigmas = np.array(d["sigmas"])
    top.fill_between(sigmas[1:], d["band"][0], d["band"][1], color=BAND, lw=0)
    for c, color in zip(d["columns"], shades, strict=True):
        top.plot(sigmas[1:], c["reference"], color=color, lw=1.0, ls=DASH, alpha=0.9)
        top.plot(sigmas[1:], c["response"], color=color, lw=1.4, label=f"θ = {c['alpha']:g}")
        bottom.plot(sigmas, c["band_pd"], color=color, lw=1.3, ls=DASH, alpha=0.9)
        bottom.plot(sigmas, c["scene_pd"], color=color, lw=1.8)
        k = int(np.argmin(np.abs(sigmas - c["sigma_hat"])))
        bottom.plot(sigmas[k], c["scene_pd"][k], "o", ms=6.5, color=color, mec="white",
                    mew=1.2, zorder=5)
        print(f"{d['title']} theta {c['alpha']:g}: theta-hat {c['theta_hat']:.4f} rho-hat "
              f"{c['rho_hat']} sigma-hat {c['sigma_hat']:.4f} | scene PD there "
              f"{c['scene_pd'][k]:.3f}, best {max(c['scene_pd']):.3f}")
    top.set_title(d["title"], fontsize=9, color=style.INK, pad=6)
    top.legend(loc="lower left", handlelength=1.4, borderaxespad=0.3)
    for ax in (top, bottom):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["0.01", "0.1", "1"])
        ax.set_ylim(-0.03, 1.05)
        ax.grid(True, color=style.GRID, lw=0.5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=REPRO / "results" / "experiment4")
    parser.add_argument("--out", type=Path, default=REPRO / "outputs" / "paper" / "figure4.pdf")
    args = parser.parse_args()
    data = json.loads((args.results / "figure4.json").read_text())
    style.apply()
    plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "custom",
                         "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                         "mathtext.bf": "Arial:bold"})
    shades = tuple(plt.get_cmap("Spectral_r")(u) for u in (0.18, 0.68, 0.84))
    fig, axes = plt.subplots(2, len(data), figsize=(style.WIDTH, 4.4), sharex=True, sharey="row",
                             layout="constrained", gridspec_kw={"height_ratios": [1.1, 1]})
    for g, d in enumerate(data.values()):
        draw(axes[0, g], axes[1, g], d, shades)
    fig.supxlabel(r"Noise level $\sigma$", fontsize=8)
    axes[0, 0].set_ylabel("Loud fraction")
    axes[1, 0].set_ylabel(r"$P_\mathrm{D}$")
    key = [Line2D([], [], color="k", lw=1.8, label="Scene"),
           Line2D([], [], color="k", lw=1.3, ls=DASH, label="Held-out implants (rule)"),
           Line2D([], [], color="k", marker="o", ls="", ms=6.5, mec="white", mew=1.2,
                  label=r"$\hat\sigma$"),
           Patch(color=BAND, label="Range of candidate implants")]
    fig.legend(handles=key, loc="outside upper center", ncol=4, handlelength=2.2,
               columnspacing=1.8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
