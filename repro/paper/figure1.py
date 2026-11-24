"""Fig. 1: detection maps on San Diego at PF = 1e-3 for DARTS, E-CEM and CS-TTD.

DARTS's map is its Rao statistic at the rule's sigma-hat (--results rule.csv), recomputed from the
Experiment 1 checkpoint, so San Diego's data file is needed: the shipped checkpoint, or the
retrained one when --results holds a run with --train (checkpoints/sandiego.pt). E-CEM's and
CS-TTD's maps are stored (results/experiment1/sandiego-maps.npz).
Top: the scene, with the zoom box; bottom: the region of the three aircraft. Each map is in its
own background standard deviations, gray from -3 to 1.5 times its threshold (darker is larger).
The threshold is the (k + 1)-th largest background value, k = floor(1e-3 n_background): the k
pixels above it are the false alarms, the target pixels at or below it the misses.

Usage, from repro/: python -m paper.figure1 [--results DIR] [--data DIR] [--out FILE]
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from dartsdet import data, statistics, whitening
from dartsdet.metrics import PF
from dartsdet.model import DARTS, pixels
from paper import style

REPRO = Path(__file__).resolve().parents[1]
GREEN, ORANGE, BLUE = "#009E73", "#D55E00", "#0072B2"


def maps(results: Path, scene: data.Scene) -> dict[str, np.ndarray]:
    """{detector: map (H, W)} on San Diego: DARTS's Rao statistic at the sigma-hat of `results`'s
    rule.csv, from that run's checkpoint if it trained one, else the shipped one; E-CEM's and
    CS-TTD's stored maps."""
    sigma = float(next(r for r in csv.DictReader((results / "rule.csv").open())
                       if r["scene"] == "sandiego")["sigma_hat"])
    retrained = results / "checkpoints" / "sandiego.pt"
    model = DARTS.load(retrained if retrained.exists() else
                       REPRO / "checkpoints" / "experiment1" / "sandiego.pt")
    recipe = json.loads((REPRO / "configs" / "darts.json").read_text())
    inputs = whitening.prepare(scene, recipe["screen"])
    z, mask = model.image(inputs.fx, inputs.usable, inputs.grid)
    c = statistics.contexts(model, z, mask, sigma)
    H, W = scene.cube.shape[:2]
    stored = np.load(REPRO / "results" / "experiment1" / "sandiego-maps.npz")  # never recomputed
    return {"DARTS": statistics.rao(model, pixels(z), c, inputs.t, sigma).double().numpy()
            .reshape(W, H).T, "E-CEM": stored["E-CEM"].T, "CS-TTD": stored["CS-TTD"].T}


def draw(ax: plt.Axes, score: np.ndarray, labels: np.ndarray, zoom: bool) -> None:
    """One map (H, W) with the label outlines, false alarms and misses at PF."""
    target, background = labels == 1, labels == 0
    k = int(np.floor(PF * background.sum()))
    threshold = np.sort(score[background])[::-1][k]
    mu, sd = score[background].mean(), score[background].std()
    ax.imshow(np.clip((score - mu) / sd, -3, None), cmap="Greys", vmin=-3,
              vmax=1.5 * (threshold - mu) / sd, interpolation="nearest")
    ax.contour(target.astype(float), levels=[0.5], colors=GREEN, linewidths=0.8 if zoom else 0.5)
    width = 0.9 if zoom else 0.6
    halo = [patheffects.withStroke(linewidth=width + 1.2, foreground="white")]
    rows, cols = np.nonzero(background & (score > threshold))
    ax.scatter(cols, rows, s=22 if zoom else 8, facecolors="none", edgecolors=ORANGE,
               linewidths=width, path_effects=halo)
    rows, cols = np.nonzero(target & (score <= threshold))
    ax.scatter(cols, rows, s=20 if zoom else 5, marker="x", c=BLUE, linewidths=width,
               path_effects=halo)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=REPRO / "results" / "experiment1")
    parser.add_argument("--data", type=Path, default=REPRO.parent / "data")
    parser.add_argument("--out", type=Path, default=REPRO / "outputs" / "paper" / "figure1.pdf")
    args = parser.parse_args()
    registry = json.loads((REPRO / "configs" / "scenes.json").read_text())
    scene = data.load(registry["sandiego"], args.data)
    H, W = scene.cube.shape[:2]
    labels = scene.labels.reshape(W, H).T
    tr, tc = np.nonzero(labels == 1)
    r0, r1 = max(tr.min() - 5, 0), min(tr.max() + 6, H)
    c0, c1 = max(tc.min() - 5, 0), min(tc.max() + 6, W)
    ratio = (r1 - r0) / (c1 - c0)
    style.apply()
    fig, axes = plt.subplots(2, 3, figsize=(style.WIDTH, style.WIDTH / 3 * (1 + ratio) + 0.35),
                             layout="constrained", gridspec_kw={"height_ratios": [1, ratio]})
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.02, hspace=0.02)
    for col, (name, score) in enumerate(maps(args.results, scene).items()):
        draw(axes[0, col], score, labels, zoom=False)
        draw(axes[1, col], score, labels, zoom=True)
        axes[1, col].set_xlim(c0 - 0.5, c1 - 0.5)
        axes[1, col].set_ylim(r1 - 0.5, r0 - 0.5)
        axes[0, col].add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False,
                                         ec=style.INK, lw=0.6, ls=(0, (4, 3))))
        axes[0, col].set_title(name, fontsize=8, color=style.INK)
    handles = [Line2D([], [], color=GREEN, lw=1.1, label="labeled target"),
               Line2D([], [], ls="none", marker="o", mfc="none", mec=ORANGE, mew=1.0, ms=5,
                      label="false alarm"),
               Line2D([], [], ls="none", marker="x", color=BLUE, mew=1.0, ms=4.5,
                      label="missed target pixel")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7, frameon=False,
               columnspacing=2.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
