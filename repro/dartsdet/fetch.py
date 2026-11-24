"""Download the scenes from their original sources and check each file against SHA256SUMS (paths
relative to the data directory's parent). Nothing is redistributed with the package.

San Diego and ABU-beach-4 come from the HTD-Mamba repository, MUUFL Gulfport from GatorSense's.
The source of Pavia University (GIC, UPV/EHU) refuses scripted downloads, so its two files are
placed by hand and only checked here. A file already present with the right checksum is kept.

Usage, from repro/: python -m dartsdet.fetch [--data DIR]
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
# Pinned to the commits whose files match SHA256SUMS (checked 2026-11-24), not to branch heads.
HTD_MAMBA = ("https://raw.githubusercontent.com/shendb2022/HTD-Mamba/"
             "8af780211e3809b647a5baaec1f9660f9e3f2ac3/datasets/")
MUUFL = ("https://raw.githubusercontent.com/GatorSense/MUUFLGulfport/"
         "144fb301b2194ce7ee139aa97e60f592f3c3612b/MUUFLGulfportDataCollection/")
PAVIA = "https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes"
SOURCES = {"Sandiego.mat": HTD_MAMBA + "Sandiego.mat",
           "abu-beach-4.mat": HTD_MAMBA + "abu-beach-4.mat",
           "MUUFLGulfport/MUUFLGulfportDataCollection/muufl_gulfport_campus_w_lidar_1.mat":
               MUUFL + "muufl_gulfport_campus_w_lidar_1.mat",
           "MUUFLGulfport/MUUFLGulfportDataCollection/tgt_lab_spectra.mat":
               MUUFL + "tgt_lab_spectra.mat",
           "pavia_university/PaviaU.mat": None,
           "pavia_university/PaviaU_gt.mat": None}


def checksums(path: Path) -> dict[str, str]:
    """{file relative to the data directory: SHA-256} from a `shasum -a 256` listing whose paths
    start with data/."""
    out = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        out[name.removeprefix("data/")] = digest
    return out


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fetch(data: Path, sums: dict[str, str], sources: dict[str, str | None]) -> list[str]:
    """Bring every file of `sources` (URL, or None for a file placed by hand) under `data` and
    check it against `sums`; the problems left, one line each (none when all files match)."""
    problems = []
    for name, url in sources.items():
        path = data / name
        if path.exists() and sha256(path) == sums[name]:
            print(f"ok          {name}")
            continue
        if url is None:
            problems.append(f"{name}: {'wrong checksum' if path.exists() else 'missing'}; "
                            f"download it from {PAVIA} and place it at {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(url, part)
        if sha256(part) != sums[name]:
            problems.append(f"{name}: downloaded from {url}, but its SHA-256 is {sha256(part)}, "
                            f"not {sums[name]}; kept as {part}")
            continue
        part.rename(path)
        print(f"downloaded  {name}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPRO.parent / "data")
    args = parser.parse_args()
    problems = fetch(args.data, checksums(REPRO / "SHA256SUMS"), SOURCES)
    print("\n".join(problems) if problems else f"every file is in {args.data}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
