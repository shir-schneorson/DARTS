"""The command-line options the experiments share."""

import argparse
from pathlib import Path

import torch

REPRO = Path(__file__).resolve().parents[1]


def default_device() -> str:
    """cuda if there is one, else mps (Apple silicon, where the paper's models were trained), else
    the cpu."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def arguments(doc: str, out: str, device: bool, data: bool,
              checkpoints: bool) -> argparse.ArgumentParser:
    """A parser described by the first paragraph of `doc`, with --out (default results/<out>)
    and, as asked, --device, --data and --checkpoints."""
    parser = argparse.ArgumentParser(description=doc.split("\n\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    if device:
        parser.add_argument("--device", default=default_device(),
                            help="torch device for training and reads")
    if data:
        parser.add_argument("--data", type=Path, default=REPRO / "data",
                            help="directory holding the scene files")
    if checkpoints:
        parser.add_argument("--checkpoints", type=Path, default=REPRO / "checkpoints",
                            help="where models are loaded from, or saved to once trained")
    parser.add_argument("--out", type=Path, default=REPRO / "results" / out,
                        help="output directory")
    return parser
