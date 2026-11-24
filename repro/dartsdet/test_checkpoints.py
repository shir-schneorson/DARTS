"""The shipped checkpoints: each matches checkpoints/SHA256SUMS and loads as a model."""

from pathlib import Path

import pytest

from dartsdet.fetch import sha256
from dartsdet.model import DARTS

REPRO = Path(__file__).resolve().parents[1]
SUMS = dict(reversed(line.split(maxsplit=1)) for line in
            (REPRO / "checkpoints" / "SHA256SUMS").read_text().splitlines())


def test_every_checkpoint_is_listed() -> None:
    assert {str(p.relative_to(REPRO)) for p in (REPRO / "checkpoints").rglob("*.pt")} == set(SUMS)


@pytest.mark.parametrize("name", sorted(SUMS))
def test_checkpoint_matches_its_checksum_and_loads(name: str) -> None:
    assert sha256(REPRO / name) == SUMS[name]
    model = DARTS.load(REPRO / name)
    assert model.record and len(model.mean) > 0
