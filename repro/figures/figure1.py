"""Fig. 1: held-out tail calibration of the tractable families on the hyperspectral scenes. Per
scene, the Gaussian, the single Student-t, and the GMM and t mixture with K >= 2 of highest
held-out log-likelihood: the held-out upper-tail share along t at 1e-3 over its nominal value,
and the share of random directions rejected at 1e-2; filled for the scene, open for the
reference, where the family is exactly right. A value below the plot's floor is marked "not
converged" when its fit did not converge (from runs.csv), else "collapsed".

Usage: python -m figures.figure1 [--results FILE] [--out FILE]
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.cli import REPRO
from figures import style

GREEN, PURPLE = "#19a979", "#945ecf"
NAMES = {"sandiego": "San Diego", "abu-airport-2": "ABU-airport", "abu-beach-4": "ABU-beach",
         "muufl-campus1-220": "MUUFL"}
MARKS = {"Gaussian": (style.ORANGE, "o"), "Student-t": (PURPLE, "s"), "GMM": (style.BLUE, "D"),
         "t mixture": (GREEN, "^")}
FLOOR = 0.08


def picks(lines: list[dict]) -> list[tuple[dict, dict, str]]:
    """Per scene: the Gaussian, the single t, and the GMM and t mixture with K >= 2 of highest
    held-out log-likelihood, each as (scene line, reference line, name)."""
    by_key = {(ln["scene"], ln["data"], ln["family"], int(ln["components"])): ln for ln in lines}
    rows = []
    for name in dict.fromkeys(ln["scene"] for ln in lines):
        scene = [ln for ln in lines if ln["scene"] == name and ln["data"] == "scene"]
        chosen = [("gmm", 1, "Gaussian"), ("student", 1, "Student-t")]
        for family, label in (("gmm", "GMM"), ("student", "t mixture")):
            mixtures = [ln for ln in scene if ln["family"] == family and int(ln["components"]) > 1]
            best = max(mixtures, key=lambda ln: float(ln["ll_test"]))
            chosen.append((family, int(best["components"]), label))
        rows += [(by_key[name, "scene", f, k], by_key[name, "reference", f, k], label)
                 for f, k, label in chosen]
    return rows


def value(line: dict, tail_ratio: bool) -> float:
    """The upper-tail ratio along t at 1e-3, or the percentage of random directions rejected at
    1e-2, averaged over the two sides."""
    if tail_ratio:
        return float(line["upper_t_0.001"])
    return 50 * (float(line["upper_rejected_0.01"]) + float(line["lower_rejected_0.01"]))


def panel(ax: plt.Axes, rows: list, scenes: list[str], tail_ratio: bool) -> None:
    """One panel: per scene and family, the reference (open) joined to the scene (filled)."""
    for i, scene in enumerate(scenes):
        group = [r for r in rows if r[0]["scene"] == scene]
        for j, (data, reference, family) in enumerate(group):
            x = i + (j - 1.5) * 0.17
            color, marker = MARKS[family]
            v, w = value(data, tail_ratio), value(reference, tail_ratio)
            if tail_ratio:
                v, w = max(v, FLOOR), max(w, FLOOR)
            ax.plot([x, x], [w, v], color=color, lw=0.8, alpha=0.5, zorder=1)
            ax.plot(x, w, marker, ms=4.5, mfc="white", mec=color, mew=1.0, zorder=2)
            ax.plot(x, v, marker, ms=4.5, color=color, mec="white", mew=0.5, zorder=3,
                    label=family if i == 0 else None)
            if tail_ratio and value(data, True) < FLOOR:
                note = "not converged" if "no" in data["converged"] else "collapsed"
                ax.annotate(note, (x, FLOOR), xytext=(0, -9), textcoords="offset points",
                            ha="center", fontsize=6, color=style.MUTED)
    ax.set_xticks(range(len(scenes)), [NAMES.get(s, s) for s in scenes])
    ax.axhline(1, color=style.MUTED, lw=0.8)
    ax.grid(True, axis="y", which="major", color=style.GRID, lw=0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path,
                        default=REPRO / "results" / "experiment1" / "runs.csv",
                        help="Experiment 1's runs.csv")
    parser.add_argument("--out", type=Path, default=REPRO / "results" / "figure1.pdf",
                        help="the figure file")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open() as f:
        rows = picks(list(csv.DictReader(f)))
    scenes = list(dict.fromkeys(r[0]["scene"] for r in rows))
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH, 2.45), constrained_layout=True)
    panel(axes[0], rows, scenes, tail_ratio=True)
    axes[0].set_ylabel(r"tail share along $\mathbf{t}$ at $10^{-3}$ / nominal")
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.05, 20)
    panel(axes[1], rows, scenes, tail_ratio=False)
    axes[1].set_ylabel(r"random directions rejected at $10^{-2}$ (%)")
    axes[1].set_ylim(0, 104)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="white", mec=style.MUTED, ms=4.5))
    labels.append("reference (family exactly right)")
    axes[0].legend(handles, labels, loc="lower right", ncol=1)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
