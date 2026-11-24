import csv
from pathlib import Path

import pytest

from paper import table2, table3, web

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_inject_replaces_only_between_markers() -> None:
    page = "a\n<!-- t:begin -->\nold\n<!-- t:end -->\nb"
    assert web.inject(page, "t", "new\n") == "a\n<!-- t:begin -->\nnew\n<!-- t:end -->\nb"
    with pytest.raises(ValueError):
        web.inject(page, "missing", "new\n")


def test_tables_carry_the_latex_tables_numbers() -> None:
    rows = table2.gather(RESULTS / "experiment1")
    html = web.table2_html(rows)
    best = table2.ranked(rows)
    for j in range(len(table2.SCENES)):
        assert f'<td class="best">{best[j][0]}</td>' in html
    levels = list(csv.DictReader((RESULTS / "experiment4" / "levels.csv").open()))
    page3 = web.table3_html(levels)
    for cells in table3.means(levels).values():
        assert all(f"<td>{c.replace('---', '—')}</td>" in page3 for c in cells)
