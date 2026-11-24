"""Table II: AUC, PD at PF 1e-3 and the standardized pAUC per scene, and the mean seconds per map.

Rows, in three groups: the model-based detectors (results/experiment1/detectors.csv) and E-CEM and
hCEM; the learned detectors run with their authors' code (both from baselines.csv); and DARTS: its
Rao test at the rule's sigma-hat (rule.csv), and, as a reference marked with a dagger, its plug-in
GLRT at sigma-hat_G, the level the rule's second stage picks from the GLRT's held-out PD. Bold and
underlined: the best and second-best printed PD of each scene among the rows without a dagger.

Usage, from repro/: python -m paper.table2 [--results DIR] [--out FILE]
"""

import argparse
import csv
from pathlib import Path

import numpy as np

REPRO = Path(__file__).resolve().parents[1]
SCENES = {"sandiego": "San Diego", "abu-beach-4": "ABU-beach-4",
          "pavia-u-a0.75": "Pavia University", "muufl-campus1-220": "MUUFL"}
DAGGER = r"\textsuperscript{\dag}"
METRICS = ("auc", "pd", "pauc")
GROUPS = (("AMF", "ACE", "CEM", "Local AMF", "Local ACE", "GMM-Rao", "GMM-GLRT", "GMM-Rao, window",
           "GMM-GLRT, window", "E-CEM", "hCEM"),
          ("LRao", "OS-VAE", "TSTTD", "CS-TTD", "HTD-Mamba"))


def gather(results: Path) -> list[tuple[int, str, list[tuple[float, ...]], float, bool]]:
    """(group, name, (auc, pd, pauc) per scene, mean seconds, label-free) per row."""
    rows = list(csv.DictReader((results / "detectors.csv").open()))
    stored = REPRO / "results" / "experiment1" / "baselines.csv"  # never recomputed
    rows += [r | {"sigma": ""} for r in csv.DictReader(stored.open())]
    rule = {r["scene"]: r for r in csv.DictReader((results / "rule.csv").open())}

    def pick(name: str, scene: str, sigma: str | None = None) -> dict:
        found = [r for r in rows if r["detector"] == name and r["scene"] == scene
                 and (sigma is None or r["sigma"] == sigma)]
        return max(found, key=lambda r: tuple(float(r[k]) for k in ("pd", "pauc", "auc")))

    def line(group: int, label: str, chosen: list[dict], free: bool, timing: list[dict]) -> tuple:
        cells = [tuple(float(r[k]) for k in METRICS) for r in chosen]
        return group, label, cells, float(np.mean([float(r["time"]) for r in timing])), free

    out = []
    for group, names in enumerate(GROUPS):
        for name in names:
            chosen = [pick(name, s) for s in SCENES]
            out.append(line(group, name, chosen, True, chosen))
    rao = [pick("DARTS Rao", s, rule[s]["sigma_hat"]) for s in SCENES]
    glrt = [pick("DARTS GLRT", s, rule[s]["sigma_hat_g"]) for s in SCENES]
    out.append(line(2, "DARTS", rao, True, rao))
    out.append(line(2, rf"DARTS, GLRT at $\hat\sigma_{{\mathrm G}}${DAGGER}", glrt, False, glrt))
    return out


def seconds(t: float) -> str:
    return f"{t:.3f}" if t < 0.01 else f"{t:.2f}" if t < 1 else f"{t:.1f}" if t < 10 else f"{t:.0f}"


def printed(value: float, k: int) -> str:
    """A metric as the table prints it: AUC (k = 0) to 4 decimals, PD and pAUC to 3."""
    return f"{value:.4f}" if k == 0 else f"{value:.3f}"


def ranked(rows: list[tuple]) -> list[list[str]]:
    """Per scene, the distinct printed PDs of the rows without a dagger, best first."""
    return [sorted({printed(r[2][j][1], 1) for r in rows if r[4]}, key=float, reverse=True)
            for j in range(len(SCENES))]


def tabular(rows: list[tuple]) -> str:
    """The LaTeX tabular of the rows."""
    best = ranked(rows)
    tex = [r"\footnotesize", r"\setlength{\tabcolsep}{2.6pt}",
           r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "ccc" * len(SCENES) + "c@{}}",
           r"\toprule",
           " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{h}}}" for h in SCENES.values())
           + r" & \\",
           "".join(rf"\cmidrule(lr){{{2 + 3 * j}-{4 + 3 * j}}}" for j in range(len(SCENES))),
           "Detector & " + " & ".join([r"AUC & $\PD$ & pAUC"] * len(SCENES)) + r" & Time (s) \\",
           r"\midrule"]
    group = 0
    for g, name, cells, t, free in rows:
        if g != group:
            tex.append(r"\midrule[\heavyrulewidth]")
            group = g
        out = []
        for j, values in enumerate(cells):
            for k, v in enumerate(values):
                s = printed(v, k)
                if k == 1 and free and s == best[j][0]:
                    s = rf"\textbf{{{s}}}"
                elif k == 1 and free and s == best[j][1]:
                    s = rf"\underline{{{s}}}"
                out.append(s)
        tex.append(name + " & " + " & ".join(out) + " & " + seconds(t) + r" \\")
    return "\n".join(tex + [r"\bottomrule", r"\end{tabular*}"]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=REPRO / "results" / "experiment1")
    parser.add_argument("--out", type=Path, default=REPRO / "outputs" / "paper" / "table2.tex")
    args = parser.parse_args()
    text = tabular(gather(args.results))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
