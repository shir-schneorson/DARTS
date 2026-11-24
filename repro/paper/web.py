"""The project page's generated parts (site/): Figs. 1-4 as PNG, drawn by paper.figure1-4 from
results/, and Tables II and III as HTML, written between the <!-- name:begin --> and
<!-- name:end --> markers of site/index.html. Every number on the page's tables is read from
results/, as the paper's tables are.

Usage, from repro/: python -m paper.web [--data DIR]
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

from paper import table2, table3

REPRO = Path(__file__).resolve().parents[1]
SITE = REPRO / "site"
TABLE3_LABELS = {"sigma_min": "Smallest level", "fixed": "One level per signature<sup>†</sup>",
                 "uniform": "No amplitude estimate", "matched": "Matched level<sup>‡</sup>",
                 "rule": r"Proposed rule (\(\hat\sigma\))",
                 "best": "Best level per scene<sup>†</sup>"}


def figures(data: Path) -> None:
    """Draw Figs. 1-4 into site/assets/figureN.png with the paper's scripts."""
    for n in range(1, 5):
        command = [sys.executable, "-m", f"paper.figure{n}",
                   "--out", str(SITE / "assets" / f"figure{n}.png")]
        subprocess.run(command + (["--data", str(data)] if n == 1 else []), check=True, cwd=REPRO)


def html_label(tex: str) -> str:
    """A Table II row label in HTML: $...$ becomes KaTeX's \\(...\\), the dagger a superscript."""
    return re.sub(r"\$(.+?)\$", r"\\(\1\\)", tex.replace(table2.DAGGER, "<sup>†</sup>"))


def table2_html(rows: list[tuple]) -> str:
    """Table II's rows (table2.gather) as an HTML table, one tbody per group."""
    best = table2.ranked(rows)
    head = "".join(f'<th colspan="3" scope="colgroup">{h}</th>' for h in table2.SCENES.values())
    metrics = "".join("<th>AUC</th><th>P<sub>D</sub></th><th>pAUC</th>" for _ in table2.SCENES)
    out = ['<table class="results t2">', "<thead>",
           f'<tr><th rowspan="2" scope="col">Detector</th>{head}'
           '<th rowspan="2" scope="col">Time (s)</th></tr>', f"<tr>{metrics}</tr>", "</thead>"]
    group = None
    for g, name, cells, t, free in rows:
        if g != group:
            out.append("</tbody>" * (group is not None) + f'<tbody class="group{g}">')
            group = g
        tds = []
        for j, values in enumerate(cells):
            for k, v in enumerate(values):
                s = table2.printed(v, k)
                rank = (best[j].index(s) if k == 1 and free and s in best[j][:2] else None)
                tds.append(f'<td class="{("best", "second")[rank]}">{s}</td>' if rank is not None
                           else f"<td>{s}</td>")
        out.append(f'<tr><th scope="row">{html_label(name)}</th>{"".join(tds)}'
                   f"<td>{table2.seconds(t)}</td></tr>")
    return "\n".join(out + ["</tbody>", "</table>"]) + "\n"


def table3_html(rows: list[dict]) -> str:
    """Table III's means (table3.means) as an HTML table, the proposed rule's row in bold."""
    head = "".join(f'<th scope="col">{h}</th>' for h in ("Pavia U.", "MUUFL", "Exp. 1"))
    out = ['<table class="results t3">', f'<thead><tr><th scope="col">Noise level</th>{head}'
           "</tr></thead>", "<tbody>"]
    for choice, cells in table3.means(rows).items():
        row = "".join(f"<td>{c.replace('---', '—')}</td>" for c in cells)
        tag = ' class="rule"' if choice == "rule" else ""
        out.append(f'<tr{tag}><th scope="row">{TABLE3_LABELS[choice]}</th>{row}</tr>')
    return "\n".join(out + ["</tbody>", "</table>"]) + "\n"


def inject(page: str, name: str, block: str) -> str:
    """The page with the text between the markers of `name` replaced by `block`."""
    pattern = re.compile(rf"(<!-- {name}:begin -->\n).*?(<!-- {name}:end -->)", re.S)
    page, count = pattern.subn(lambda m: m[1] + block + m[2], page)
    if count != 1:
        raise ValueError(f"site/index.html needs one pair of {name} markers, found {count}")
    return page


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPRO.parent / "data")
    args = parser.parse_args()
    figures(args.data)
    results = REPRO / "results"
    levels = list(csv.DictReader((results / "experiment4" / "levels.csv").open()))
    index = SITE / "index.html"
    page = inject(index.read_text(), "table2", table2_html(table2.gather(results / "experiment1")))
    index.write_text(inject(page, "table3", table3_html(levels)))
    print("wrote", SITE / "assets", "and the tables of", index)


if __name__ == "__main__":
    main()
