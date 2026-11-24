"""The scenes, their labels, and the additive implants of a known target (Section VI-A).

A scene of H x W pixels has P = H W of them. Pixel vectors, shape (P,), are raveled in Fortran
(column-major) order, as the hyperspectral baselines write their maps: pixel p is row p % H,
column p // H.

San Diego and ABU-beach-4 contain real targets. Their files hold `data` (H, W, B) and `map`
(H, W), and the target spectrum is the mean of the labeled target pixels. Pavia University and
MUUFL campus 1 receive additive implants alpha d at one site per 100 scored background pixels, d a
class mean of Pavia University or a laboratory cloth signature of MUUFL. Real instances of the
implanted material are never implanted on and are not scored (`IGNORED`), nor are pixels that do
not exist (Pavia's saturated pixels). On MUUFL, the background pixels in the border of any emplaced
cloth are not scored either; implants avoid only the borders of the cloths their signature is
scored against, so the few that land in another cloth's border (12-20 of about 705 per signature)
are scored as targets.

MUUFL Gulfport: P. Gader, A. Zare, R. Close, J. Aitken, G. Tuell, University of Florida, Tech.
Rep. REP-2013-570, 2013. Its ground truth is one marked pixel per cloth; the border around each
mark (`halo_mask`) is the scoring window of Bullwinkle (T. Glenn, A. Zare, P. Gader,
D. Dranishnikov, GatorSense/MUUFLGulfport, 2016).
"""

import dataclasses
from pathlib import Path

import numpy as np
import scipy.io as sio

IGNORED = -1  # the label of a pixel scored neither as target nor as background
# One implant per 100 scored background pixels, at least 3 pixels apart (Chebyshev), sites drawn
# with seed 0.
IMPLANT = {"background_per_target": 100, "spacing": 3, "seed": 0}
PAVIA_CEILING = 8000  # Pavia University is stored clipped at this DN; such pixels do not exist

# MUUFL's 64-band convention drops the first and last four of the 72 delivered bands, as
# `hsi.info.remove_band_index` of the collection's campus-1 label file records.
MUUFL_DROP_BANDS = (0, 1, 2, 3, 68, 69, 70, 71)
# demo.m's lab_tgt_inds = [4 2 3 5], 1-based into tgt_lab_spectra.mat, and the emplaced cloth
# types each lab signature is scored against. The lab set has no faux vineyard green; Vineyard
# Green is scored against the faux cloths, since campus 1's first 220 columns hold no true one.
MUUFL_LAB_ORDER = (3, 1, 2, 4)
MUUFL_TYPES = {"Soil Brown": ("brown",), "Solid Dark Green": ("dark green",),
               "Vineyard Green": ("faux vineyard green",), "Pea Green": ("pea green",)}
HALO = 2  # Bullwinkle's default border, pixels


@dataclasses.dataclass(frozen=True)
class Scene:
    """A scene as the detectors see it.

    cube: (H, W, B) float64. labels: (P,) 1 target, 0 background, `IGNORED` not scored.
    exists: (P,) bool. target: (B,) the target spectrum. additive: implanted as x + alpha target
    (else real targets, which replace the background).
    """

    cube: np.ndarray
    labels: np.ndarray
    exists: np.ndarray
    target: np.ndarray
    additive: bool

    def spectra(self) -> np.ndarray:
        """(P, B) float64, in pixel order."""
        return self.cube.reshape(-1, self.cube.shape[2], order="F")


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
    scored = real.size - int((real | gone).sum())
    count = round(scored / (1 + IMPLANT["background_per_target"]))
    rng = np.random.default_rng(IMPLANT["seed"])
    rows, cols = implant_sites(~real & ~gone, count, IMPLANT["spacing"], rng).T
    data = cube.astype(np.float64)
    data[rows, cols] += alpha * d
    labels = np.zeros(real.shape)
    labels[rows, cols] = 1
    labels[real | gone] = IGNORED
    return Scene(data, labels.ravel(order="F"), ~gone.ravel(order="F"), d, additive=True)


def halo_mask(shape: tuple[int, int], rows: np.ndarray, cols: np.ndarray,
              sizes: np.ndarray) -> np.ndarray:
    """(H, W) bool: the union of the cloths' borders, clipped to the image. A cloth `size` metres
    across gets Bullwinkle's square of side 2 HALO + 1 (0.5 and 1 m cloths) or 2 HALO + 3 (3 m),
    or a (2 HALO + 13) x (2 HALO + 11) rectangle (the 6 x 10 m calibration cloths)."""
    extents = {0.5: (1, 1), 1.0: (1, 1), 3.0: (3, 3), 6.0: (13, 11)}
    mask = np.zeros(shape, dtype=bool)
    for row, col, size in zip(rows, cols, sizes, strict=True):
        if size not in extents:
            raise ValueError(f"no border for a cloth of {size} m: the collection has 0.5, 1, 3, 6")
        height, width = (2 * HALO + e for e in extents[size])
        mask[max(0, row - height // 2):min(shape[0], row + height // 2 + 1),
             max(0, col - width // 2):min(shape[1], col + width // 2 + 1)] = True
    return mask


def real_targets(path: Path) -> Scene:
    """A `{data, map}` scene file whose targets are its labeled pixels."""
    mat = sio.loadmat(path)
    cube, targets = mat["data"].astype(np.float64), (mat["map"] == 1).ravel(order="F")
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


def muufl(collection: Path, file: str, columns: int, signature: int, alpha: float) -> Scene:
    """The first `columns` columns of a MUUFL campus scene, with lab signature `signature`
    (0-3, `MUUFL_LAB_ORDER`) implanted at amplitude alpha. The lab spectrum is resampled to the
    nearest lab wavelength of each band. Implants avoid the borders of the cloths that signature
    is scored against; background pixels in the border of any cloth are not scored, implants there
    are."""
    load = {"struct_as_record": False, "squeeze_me": True}
    hsi = sio.loadmat(collection / file, **load)["hsi"]
    keep = np.ones(hsi.Data.shape[2], dtype=bool)
    keep[list(MUUFL_DROP_BANDS)] = False
    if not np.asarray(hsi.valid_mask)[:, :columns].all():
        raise ValueError(f"{file}'s first {columns} columns contain invalid pixels")
    cube = np.asarray(hsi.Data, dtype=np.float64)[:, :columns][:, :, keep]
    truth = hsi.groundTruth
    col = np.asarray(truth.Targets_colIndices, dtype=np.int64) - 1
    inside = col < columns
    row = (np.asarray(truth.Targets_rowIndices, dtype=np.int64) - 1)[inside]
    col, size = col[inside], np.asarray(truth.Targets_Size, dtype=np.float64)[inside]
    kind = np.array([str(t) for t in truth.Targets_Type])[inside]
    lab = sio.loadmat(collection / "tgt_lab_spectra.mat", **load)["spectralSignatures"]
    lab = lab[MUUFL_LAB_ORDER[signature]]
    wavelength = np.asarray(hsi.info.wavelength)[keep]
    nearest = np.abs(np.asarray(lab.wavelengths)[None, :] - wavelength[:, None]).argmin(axis=1)
    d = np.asarray(lab.reflectance, dtype=np.float64)[nearest]
    mine = np.isin(kind, MUUFL_TYPES[str(lab.name)])
    real = halo_mask(cube.shape[:2], row[mine], col[mine], size[mine])
    scene = implanted(cube, d, real, np.zeros(cube.shape[:2], dtype=bool), alpha)
    borders = halo_mask(cube.shape[:2], row, col, size).ravel(order="F")
    labels = np.where((scene.labels == 0) & borders, IGNORED, scene.labels)
    return dataclasses.replace(scene, labels=labels)


def load(entry: dict, data_dir: Path) -> Scene:
    """The scene a `configs/scenes.json` entry describes, its files under `data_dir`. An implanted
    entry's `target` is a Pavia class or a MUUFL lab signature, `alpha` its amplitude."""
    if entry["kind"] == "mat":
        return real_targets(data_dir / entry["file"])
    if entry["kind"] == "pavia":
        return pavia(data_dir / entry["directory"], entry["target"], entry["alpha"])
    if entry["kind"] == "muufl":
        return muufl(data_dir / entry["collection"], entry["file"], entry["columns"],
                     entry["target"], entry["alpha"])
    raise ValueError(f"unknown scene kind {entry['kind']!r}")


def without_targets(scene: Scene) -> Scene:
    """An implanted scene's labels without its implants. Requires a scene implanted at
    alpha = 0, whose cube is the source's: the cube is kept as it is."""
    return dataclasses.replace(scene, labels=np.where(scene.labels == 1, 0, scene.labels))
