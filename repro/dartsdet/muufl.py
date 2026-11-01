"""MUUFL Gulfport: the campus cube, its lab target signatures and its emplaced cloth targets.

Data: P. Gader, A. Zare, R. Close, J. Aitken, G. Tuell, "MUUFL Gulfport Hyperspectral and LiDAR
Airborne Data Set," University of Florida, Tech. Rep. REP-2013-570, 2013.

The ground truth is one marked pixel per emplaced cloth, not a target mask; the Bullwinkle halo
around each mark (`halo_mask`) is the collection's allowance for registration error. The halos
port Bullwinkle's scoring windows: T. Glenn, A. Zare, P. Gader, D. Dranishnikov. (2016).
Bullwinkle: Scoring Code for Sub-pixel Targets (Version 1.0) [Software]. GatorSense/MUUFLGulfport.
"""

from pathlib import Path

import numpy as np
import scipy.io as sio

SCENES = {"campus1": "muufl_gulfport_campus_w_lidar_1.mat"}

# The 64-band convention: the first four and last four of the 72 delivered bands are dropped, as
# `hsi.info.remove_band_index` in MUUFLGulfportSceneLabels/muufl_gulfport_campus_1_hsi_220_label.mat
# records them (that file is not needed here).
DROP_BANDS = (0, 1, 2, 3, 68, 69, 70, 71)

# demo.m's lab_tgt_inds = [4 2 3 5], 1-based into tgt_lab_spectra.mat, and the emplaced cloth types
# each lab signature is scored against. The lab set holds no faux vineyard green measurement; we
# score Vineyard Green against the faux cloths, since campus 1's first 220 columns hold none of the
# true ones.
LAB_ORDER = (3, 1, 2, 4)
LAB_TARGET_TYPES = {"Soil Brown": ("brown",), "Solid Dark Green": ("dark green",),
                    "Vineyard Green": ("faux vineyard green",), "Pea Green": ("pea green",)}

HALO = 2  # Bullwinkle's default halo, pixels


def _loadmat(path: Path) -> dict:
    return sio.loadmat(path, struct_as_record=False, squeeze_me=True)


def campus(collection: Path, scene: str, columns: int) -> dict:
    """The first `columns` columns of a campus scene.

    Returns:
        `cube` (H, W, 64) float64 reflectance; `signatures` (64, 4), the lab spectra
        nearest-wavelength resampled onto the scene's bands; `names` of the signatures; and the
        targets inside the crop: `row`, `col` (0-based), `size` (m) and `type`.
    """
    hsi = _loadmat(collection / SCENES[scene])["hsi"]
    keep = np.ones(hsi.Data.shape[2], dtype=bool)
    keep[list(DROP_BANDS)] = False
    if not np.asarray(hsi.valid_mask)[:, :columns].all():
        raise ValueError(f"{scene}'s first {columns} columns contain invalid pixels")
    truth = hsi.groundTruth
    col = np.asarray(truth.Targets_colIndices, dtype=np.int64) - 1
    inside = col < columns
    signatures, names = lab_signatures(collection, np.asarray(hsi.info.wavelength)[keep])
    return {"cube": np.asarray(hsi.Data, dtype=np.float64)[:, :columns][:, :, keep],
            "signatures": signatures, "names": names,
            "row": (np.asarray(truth.Targets_rowIndices, dtype=np.int64) - 1)[inside],
            "col": col[inside],
            "size": np.asarray(truth.Targets_Size, dtype=np.float64)[inside],
            "type": np.array([str(t) for t in truth.Targets_Type])[inside]}


def lab_signatures(collection: Path, wavelength: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Lab cloth reflectances (B, 4) at the nearest lab wavelength to each band, and their names."""
    spectra = _loadmat(collection / "tgt_lab_spectra.mat")["spectralSignatures"]
    lab = np.asarray(spectra[0].wavelengths, dtype=np.float64)
    nearest = np.abs(lab[None, :] - wavelength[:, None]).argmin(axis=1)
    signatures = np.stack([np.asarray(spectra[i].reflectance, dtype=np.float64)[nearest]
                           for i in LAB_ORDER], axis=1)
    return signatures, [str(spectra[i].name) for i in LAB_ORDER]


def halo_extent(size: float) -> tuple[int, int]:
    """Halo height and width in pixels of a target `size` metres across: Bullwinkle's square of
    side 2 HALO + 1 for the 0.5 and 1 m cloths, 2 HALO + 3 for the 3 m ones, and a
    (2 HALO + 13) x (2 HALO + 11) rectangle for the 6 x 10 m calibration cloths."""
    extents = {0.5: (1, 1), 1.0: (1, 1), 3.0: (3, 3), 6.0: (13, 11)}
    if size not in extents:
        raise ValueError(f"no halo for a target of {size} m: the collection has 0.5, 1, 3 and 6")
    height, width = extents[size]
    return 2 * HALO + height, 2 * HALO + width


def halo_mask(shape: tuple[int, int], rows: np.ndarray, cols: np.ndarray,
              sizes: np.ndarray) -> np.ndarray:
    """(H, W) bool: the union of the targets' halos, clipped to the image."""
    mask = np.zeros(shape, dtype=bool)
    for row, col, size in zip(rows, cols, sizes, strict=True):
        height, width = halo_extent(size)
        mask[max(0, row - height // 2):min(shape[0], row + height // 2 + 1),
             max(0, col - width // 2):min(shape[1], col + width // 2 + 1)] = True
    return mask
