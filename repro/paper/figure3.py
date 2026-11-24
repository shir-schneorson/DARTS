"""Fig. 3: PD at PF = 1e-3 against the implant amplitude in Experiment 3, mean over target
signatures with half a standard deviation (over signatures) as a band; rows Pavia University and
MUUFL, one column per score model, one line per noise level, light to dark with the level:
- GMM, Rao test: the mixture smoothed exactly at sigma (results/experiment3/<bg>-gmm.csv);
- LRao objective: the DARTS network trained with LRao's objective at input noise sigma, the better
  of its last and best held-out checkpoints; stored rows (<bg>-lrao-objective.csv);
- DARTS: its Rao test at the grid levels nearest those sigmas (<bg>-darts.csv), at the rule's
  sigma-hat (orange) and at sigma-u, the level without the amplitude estimate (dashed), each
  scene at its own level (<bg>-rule.csv).
The legend's sigma is the GMM's and LRao's, in whitened coordinates; DARTS's levels are its own
grid's, in its standardized coordinates (whitened bands over their training standard deviation,
0.81-1.00), so a column's "sigma = 0.1" is 0.11 there.

Usage, from repro/: python -m paper.figure3 [--results DIR] [--out FILE]
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from paper import style

REPRO = Path(__file__).resolve().parents[1]
BACKGROUNDS = {"pavia": "Pavia University", "muufl": "MUUFL"}
LEVELS = (0.0, 0.1, 0.3, 1.0)  # GMM and LRao noise levels
DARTS_LEVELS = (0.005, 0.11, 0.2659, 1.0)  # the levels of the rule's grid nearest LEVELS
BLUES, ORANGE = ("#9ecae1", "#6baed6", "#3182bd", "#08519c"), "#e8743b"
BAND = 0.5  # standard deviations over signatures


def by_amplitude(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(amplitudes, mean PD, its standard deviation over signatures)."""
    if not rows:
        raise ValueError("no rows at this level: the grid and the results disagree")
    cells = defaultdict(list)
    for r in rows:
        cells[float(r["alpha"])].append(float(r["pd"]))
    alphas = np.array(sorted(cells))
    return (alphas, np.array([np.mean(cells[a]) for a in alphas]),
            np.array([np.std(cells[a]) for a in alphas]))


def columns(results: Path, bg: str) -> dict[str, list[tuple[str, tuple]]]:
    """Column title: [(line label, by_amplitude curve)], the levels in order, then for DARTS the
    rule ("rule") and sigma-u ("simple")."""
    read = {name: list(csv.DictReader((results / f"{bg}-{name}.csv").open()))
            for name in ("gmm", "darts", "rule")}
    stored = REPRO / "results" / "experiment3" / f"{bg}-lrao-objective.csv"  # never recomputed
    read["lrao-objective"] = list(csv.DictReader(stored.open()))
    rao = {(r["target"], r["alpha"], r["sigma"]): r for r in read["darts"] if r["test"] == "Rao"}
    darts = [(f"{s:g}", by_amplitude([r for r in rao.values()
                                      if abs(float(r["sigma"]) - s) < 1e-3]))
             for s in DARTS_LEVELS]
    for label, key in (("rule", "sigma_hat"), ("simple", "sigma_u")):
        darts.append((label, by_amplitude([rao[r["target"], r["alpha"], r[key]]
                                           for r in read["rule"]])))
    return {"GMM, Rao test": [(f"{s:g}", by_amplitude(
                [r for r in read["gmm"] if float(r["sigma"]) == s])) for s in LEVELS],
            "LRao objective (DARTS network)": [(f"{s:g}", by_amplitude(
                [r for r in read["lrao-objective"] if float(r["noise"]) == s])) for s in LEVELS],
            "DARTS (DSM), Rao test": darts}


def draw(ax: plt.Axes, lines: list[tuple[str, tuple]]) -> None:
    """One panel: the levels light to dark, the rule in orange, sigma-u dashed, each with a band."""
    for k, (label, (a, m, sd)) in enumerate(lines):
        simple, rule = label == "simple", label == "rule"
        color = style.INK if simple else ORANGE if rule else BLUES[k]
        ax.fill_between(a, np.clip(m - BAND * sd, 0, 1), np.clip(m + BAND * sd, 0, 1),
                        color=color, alpha=0.1, lw=0, zorder=1)
        if simple:
            ax.plot(a, m, color=color, lw=1.3, ls="--", marker="o", ms=2.6, zorder=3)
        else:
            ax.plot(a, m, color=color, lw=2.0 if rule else 1.5, marker="o",
                    ms=3.2 if rule else 2.6, zorder=4 if rule else 2 + 0.1 * k)
        print(f"  {label:6s} " + " ".join(f"{x:g} {y:.3f}±{s:.3f}" for x, y, s in zip(a, m, sd,
                                                                                    strict=True)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=REPRO / "results" / "experiment3")
    parser.add_argument("--out", type=Path, default=REPRO / "outputs" / "paper" / "figure3.pdf")
    args = parser.parse_args()
    style.apply()
    fig, axes = plt.subplots(2, 3, figsize=(style.WIDTH, 3.6), sharex="row", sharey=True,
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.03, hspace=0.03)
    for r, (bg, name) in enumerate(BACKGROUNDS.items()):
        for c, (title, lines) in enumerate(columns(args.results, bg).items()):
            print(name, title)
            ax = axes[r, c]
            draw(ax, lines)
            alphas = lines[0][1][0]
            ax.set_xscale("log", base=2)
            ax.set_xticks(alphas, [f"{a:g}" for a in alphas], fontsize=6.5)
            ax.minorticks_off()
            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.tick_params(axis="y", labelsize=6.5)
            ax.grid(True, color=style.GRID, lw=0.5)
            if r == 0:
                ax.set_title(title, fontsize=8, color=style.INK)
            if c == 0:
                ax.set_ylabel(f"{name}\n" + r"$P_\mathrm{D}$")
    handles = [Line2D([], [], color=BLUES[k], lw=1.5, marker="o", ms=2.6, label=rf"$\sigma={s:g}$")
               for k, s in enumerate(LEVELS)]
    handles += [Line2D([], [], color=ORANGE, lw=2.0, marker="o", ms=3.2,
                       label=r"DARTS at $\hat\sigma$ (rule)"),
                Line2D([], [], color=style.INK, lw=1.3, ls="--", marker="o", ms=2.6,
                       label=r"DARTS at $\hat\sigma_{\mathrm{u}}$ (no amplitude estimate)")]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), fontsize=7,
               frameon=False, columnspacing=1.2, handlelength=2.2)
    fig.supxlabel(r"Implant amplitude $\theta$", fontsize=8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
