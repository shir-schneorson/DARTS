"""The fetch script's checks, without the network: files already in place are kept when their
checksum matches, and a missing or wrong file placed by hand is reported."""

import hashlib
from pathlib import Path

from dartsdet.fetch import REPRO, SOURCES, checksums, fetch


def test_every_source_has_a_checksum() -> None:
    assert set(SOURCES) <= set(checksums(REPRO / "SHA256SUMS"))


def test_matching_files_are_kept_and_others_reported(tmp_path: Path) -> None:
    good, bad = tmp_path / "a.mat", tmp_path / "b.mat"
    good.write_bytes(b"background")
    bad.write_bytes(b"not the file")
    sums = {name: hashlib.sha256(b"background").hexdigest() for name in ("a.mat", "b.mat", "c.mat")}
    problems = fetch(tmp_path, sums, {"a.mat": None, "b.mat": None, "c.mat": None})
    assert [p.split(":")[0] for p in problems] == ["b.mat", "c.mat"]
    assert "wrong checksum" in problems[0] and "missing" in problems[1]
