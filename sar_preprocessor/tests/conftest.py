import numpy as np
import pytest


@pytest.fixture
def single_band_image():
    """Synthetic single-band SAR image."""
    rng = np.random.default_rng(42)
    return rng.random((64, 64), dtype=np.float32)


@pytest.fixture
def multi_band_image():
    """Synthetic 3-band SAR image."""
    rng = np.random.default_rng(42)
    return rng.random((3, 64, 64), dtype=np.float32)
