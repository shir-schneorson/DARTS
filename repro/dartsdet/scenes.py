"""The scenes, with their labels, and the additive implants of a known target.

A scene of H x W pixels has P = H W of them. Pixel vectors, shape (P,), are raveled in Fortran
(column-major) order, as the hyperspectral baselines dump their maps: pixel p is row p % H,
column p // H.

Real-target scenes (San Diego, ABU) are shipped as `{data, map}` .mat files; the target spectrum is
the mean of the labeled target pixels. Implanted scenes (`IMPLANT`) add alpha d to sites of a
background scene, d a known spectrum: a class mean of Pavia University, or a lab signature of MUUFL.
Real instances of the implanted material are never implanted on and are labeled `IGNORED`, as are
pixels that do not exist (Pavia's saturated pixels).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

from dartsdet import muufl

IGNORED = -1  # the label of a pixel scored neither as target nor as background
# The implant protocol: one implant per 100 scored background pixels, at least 3 pixels apart
# (Chebyshev), sites drawn with seed 0.
IMPLANT = {"background_per_target": 100, "spacing": 3, "seed": 0}
PAVIA_CEILING = 8000  # Pavia University is stored clipped at this DN; such pixels do not exist


@dataclass(frozen=True)
class Scene:
    """A scene as the detectors see it.

    cube: (H, W, B) float64. labels: (P,) 1 target, 0 background, `IGNORED` neither.
    exists: (P,) bool. target: (B,) the target spectrum. additive: implanted as x + alpha target
    (else real targets, which replace the background). masked: MUUFL's labels with every emplaced
    cloth's halo also ignored, or None.
    """

    cube: np.ndarray
    labels: np.ndarray
    exists: np.ndarray
    target: np.ndarray
    additive: bool
    masked: np.ndarray | None = None

    def spectra(self) -> np.ndarray:
        """(P, B) float64, in pixel order."""
        return self.cube.reshape(-1, self.cube.shape[2], order="F")


def fortran(mask: np.ndarray) -> np.ndarray:
    """An (H, W) map as a (P,) pixel vector."""
    return mask.ravel(order="F")


def implant_count(pixels: int, ignored: int, background_per_target: int) -> int:
    """Implants that leave `background_per_target` scored background pixels for each one."""
    return round((pixels - ignored) / (1 + background_per_target))


def implant_sites(allowed: np.ndarray, count: int, spacing: int,
                  rng: np.random.Generator) -> np.ndarray:
    """`count` sites (count, 2) as (row, col) from `allowed` (H, W) bool: candidates in random
    order, each kept if its Chebyshev distance to every kept site is at least `spacing`."""
    kept = np.empty((0, 2), dtype=np.int64)
    for site in rng.permutation(np.argwhere(allowed)):
        if len(kept) == count:
            break
        if not len(kept) or np.abs(kept - site).max(axis=1).min() >= spacing:
            kept = np.vstack([kept, site])
    if len(kept) < count:
        raise ValueError(f"only {len(kept)} of {count} sites fit at spacing {spacing}")
    return kept


def implanted(cube: np.ndarray, d: np.ndarray, real: np.ndarray, gone: np.ndarray,
              alpha: float) -> Scene:
    """`cube` (H, W, B) with alpha d added at sites away from the real instances `real` and the
    non-existing pixels `gone`, (H, W) bool each, under the `IMPLANT` protocol."""
    count = implant_count(real.size, int((real | gone).sum()), IMPLANT["background_per_target"])
    rng = np.random.default_rng(IMPLANT["seed"])
    rows, cols = implant_sites(~real & ~gone, count, IMPLANT["spacing"], rng).T
    data = cube.astype(np.float64)
    data[rows, cols] += alpha * d
    labels = np.zeros(real.shape)
    labels[rows, cols] = 1
    labels[real | gone] = IGNORED
    return Scene(data, fortran(labels), fortran(~gone), d, additive=True)


def real_targets(path: Path) -> Scene:
    """A `{data, map}` scene file whose targets are its labeled pixels."""
    mat = sio.loadmat(path)
    cube, targets = mat["data"].astype(np.float64), fortran(mat["map"] == 1)
    target = cube.reshape(-1, cube.shape[2], order="F")[targets].mean(axis=0)
    return Scene(cube, targets.astype(np.float64), np.ones(targets.size, dtype=bool), target,
                 additive=False)


def pavia(directory: Path, target_class: int, alpha: float) -> Scene:
    """Pavia University with the mean of class `target_class` implanted at amplitude alpha."""
    cube = sio.loadmat(directory / "PaviaU.mat")["paviaU"]
    classes = sio.loadmat(directory / "PaviaU_gt.mat")["paviaU_gt"]
    gone = (cube >= PAVIA_CEILING).any(axis=-1)
    real = classes == target_class
    d = cube[real & ~gone].astype(np.float64).mean(axis=0)
    return implanted(cube, d, real, gone, alpha)


def muufl_campus(collection: Path, scene: str, columns: int, signature: int,
                 alpha: float) -> Scene:
    """A MUUFL campus crop with lab signature `signature` implanted at amplitude alpha. The real
    instances are the halos of the cloths that signature detects; `masked` also ignores every
    other cloth's halo."""
    campus = muufl.campus(collection, scene, columns)
    shape = campus["cube"].shape[:2]
    wanted = muufl.LAB_TARGET_TYPES[campus["names"][signature]]
    mine = np.isin(campus["type"], wanted)
    real = muufl.halo_mask(shape, campus["row"][mine], campus["col"][mine], campus["size"][mine])
    out = implanted(campus["cube"], campus["signatures"][:, signature], real,
                    np.zeros(shape, dtype=bool), alpha)
    halos = fortran(muufl.halo_mask(shape, campus["row"], campus["col"], campus["size"]))
    masked = np.where((out.labels == 0) & halos, IGNORED, out.labels)
    return Scene(out.cube, out.labels, out.exists, out.target, out.additive, masked)


def without_targets(scene: Scene) -> Scene:
    """An implanted scene's labels without its implants, the signature and the ignored and
    excluded pixels kept. Requires a scene implanted at alpha = 0, whose cube is the source's: the
    cube is kept as it is."""
    labels = np.where(scene.labels == 1, 0, scene.labels)
    masked = None if scene.masked is None else np.where(scene.masked == 1, 0, scene.masked)
    return Scene(scene.cube, labels, scene.exists, scene.target, scene.additive, masked)


def from_entry(entry: dict, data_dir: Path) -> Scene:
    """The scene a registry entry describes, its files under `data_dir`. An implanted entry's
    `target` is a Pavia class or a MUUFL lab signature (column), `alpha` its amplitude."""
    if entry["kind"] == "mat":
        return real_targets(data_dir / entry["file"])
    if entry["kind"] == "pavia":
        return pavia(data_dir / entry["directory"], entry["target"], entry["alpha"])
    if entry["kind"] == "muufl":
        return muufl_campus(data_dir / entry["collection"], entry["scene"], entry["columns"],
                            entry["target"], entry["alpha"])
    raise ValueError(f"unknown scene kind {entry['kind']!r}")


def load(name: str, registry: Path, data_dir: Path) -> Scene:
    """Scene `name` of the registry file (configs/scenes.json)."""
    return from_entry(json.loads(registry.read_text())[name], data_dir)
