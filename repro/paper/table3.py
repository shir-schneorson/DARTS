"""Table III: mean PD at PF = 1e-3 of DARTS for each choice of the noise level, per set of scenes
(Experiment 4's levels.csv), as the paper's tabular: the proposed rule's row in bold. One level per
signature is "---" for Experiment 1, where each signature has one scene and it coincides with the
best level.

Usage, from repro/: python -m paper.table3 [--results DIR] [--out FILE]
"""

import argparse
import csv
from pathlib import Path

import numpy as np

REPRO = Path(__file__).resolve().parents[1]
SETS = {"Pavia University": "Pavia U.", "MUUFL": "MUUFL", "Experiment 1": r"Exp.~1"}
ROWS = {"sigma_min": "Smallest level", "fixed": r"One level per signature $\dagger$",
        "uniform": "No amplitude estimate", "matched": r"Matched level $\ddagger$",
        "rule": r"\textbf{Proposed rule ($\hat\sigma$)}",
        "best": r"Best level per scene $\dagger$"}


def means(rows: list[dict]) -> dict[str, list[str]]:
    """{choice: printed mean PD per set} from the rows of levels.csv."""
    return {choice: ["---" if choice == "fixed" and name == "Experiment 1" else
                     f"{np.mean([float(r[f'pd_{choice}']) for r in rows if r['set'] == name]):.3f}"
                     for name in SETS] for choice in ROWS}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=REPRO / "results" / "experiment4")
    parser.add_argument("--out", type=Path, default=REPRO / "outputs" / "paper" / "table3.tex")
    args = parser.parse_args()
    rows = list(csv.DictReader((args.results / "levels.csv").open()))
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             "Noise level & " + " & ".join(SETS.values()) + r" \\", r"\midrule"]
    for (choice, label), cells in zip(ROWS.items(), means(rows).values(), strict=True):
        if choice == "rule":
            lines += [f"{label} &", " & ".join(rf"\textbf{{{c}}}" for c in cells) + r" \\"]
        else:
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
