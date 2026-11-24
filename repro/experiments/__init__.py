"""The experiments of the paper, one driver per experiment, and what they share."""

import csv
from pathlib import Path

import torch


def write(path: Path, rows: list[dict]) -> None:
    """rows as a CSV file, the first row's keys as the header."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def default_device() -> str:
    """The device --train uses unless told otherwise: MPS, then CUDA, then the CPU."""
    return ("mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu")
