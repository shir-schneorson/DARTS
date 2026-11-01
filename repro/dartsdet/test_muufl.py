"""The MUUFL halos, Bullwinkle's scoring windows."""

import numpy as np
import pytest

from dartsdet import muufl


def test_halo_mask_sizes_and_clipping() -> None:
    assert muufl.halo_extent(0.5) == (5, 5) and muufl.halo_extent(3) == (7, 7)
    assert muufl.halo_extent(6) == (17, 15)
    mask = muufl.halo_mask((20, 20), np.array([0, 10]), np.array([0, 10]), np.array([1.0, 3.0]))
    assert mask[:3, :3].all() and mask.sum() == 9 + 49


def test_unknown_target_size_is_refused() -> None:
    with pytest.raises(ValueError, match="no halo"):
        muufl.halo_extent(2.0)
