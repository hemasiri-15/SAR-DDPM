"""
structdiff/utils/structure_tensor_multiscale.py
================================================
A10: Multi-Scale Structure Tensor — pure-NumPy computation utility.

Extends ``structdiff/utils/structure_tensor.py`` (A3) by computing
``compute_structure_tensor`` at three integration scales and stacking
the results into a single [9, H, W] array or returning them as a list
of three [3, H, W] arrays.

No new logic is introduced: this module is a thin loop over
``compute_structure_tensor``, which owns all numerical code.

Design contract
---------------
- ``compute_structure_tensor`` is NOT modified.
- This module has no PyTorch dependency (CPU DataLoader workers).
- Default scales match the A10 plan: sigma1=1.0, sigma2=2.5, sigma3=4.5.
- rho (pre-smoothing) is shared across all scales; it suppresses speckle
  before differentiation and is independent of integration scale.

Return convention
-----------------
``compute_structure_tensor_multiscale`` returns a list of three arrays:

    [s1, s2, s3]

where each sN is shape [3, H, W], float32, values in [-1, 1].
The list is consumed by ``MultiScaleStructTensorDataset.__getitem__``
and stacked into three separate tensors for the DataLoader.

Tensor stacking (for the encoder) is done inside
``MultiScaleStructTensorEncoder.forward``, not here.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# Re-use A3 computation — no code duplication.
from utils.structure_tensor import compute_structure_tensor  # noqa: E402


# ---------------------------------------------------------------------------
# Default scale triplet (ablation parameters)
# ---------------------------------------------------------------------------

#: Fine integration scale (pixels).  Captures sharp edges.
SIGMA_FINE: float = 1.0

#: Medium integration scale (pixels).  Captures mid-range structures.
SIGMA_MEDIUM: float = 2.5

#: Coarse integration scale (pixels).  Captures broad orientation fields.
SIGMA_COARSE: float = 4.5

#: Default (sigma_fine, sigma_medium, sigma_coarse) triplet.
DEFAULT_SIGMAS: Tuple[float, float, float] = (SIGMA_FINE, SIGMA_MEDIUM, SIGMA_COARSE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_structure_tensor_multiscale(
    image: np.ndarray,
    rho: float = 1.0,
    sigmas: Tuple[float, float, float] = DEFAULT_SIGMAS,
    normalise: bool = True,
) -> List[np.ndarray]:
    """Compute structure tensors at three integration scales.

    Calls ``compute_structure_tensor`` once per scale.  All other
    parameters (rho, normalise) are forwarded unchanged.

    Parameters
    ----------
    image:
        2-D float32 array of shape [H, W], range [0, 1].
        Squeeze the channel axis before calling (same contract as A3).
    rho:
        Pre-smoothing Gaussian std (pixels) shared across all scales.
        Default 1.0 (same as A3 default).
    sigmas:
        Three integration Gaussian stds (pixels): (fine, medium, coarse).
        Default (1.0, 2.5, 4.5).  Each value must be > 0.
    normalise:
        Passed to ``compute_structure_tensor``.  Default True.

    Returns
    -------
    List[np.ndarray]
        ``[s1, s2, s3]`` — one array per scale.
        Each array: shape [3, H, W], dtype float32, range [-1, 1].
        s1 = fine scale   (sigmas[0])
        s2 = medium scale (sigmas[1])
        s3 = coarse scale (sigmas[2])

    Raises
    ------
    ValueError
        If ``image`` is not 2-D, or any sigma <= 0, or len(sigmas) != 3.

    Examples
    --------
    >>> img = np.random.rand(256, 256).astype(np.float32)
    >>> tensors = compute_structure_tensor_multiscale(img)
    >>> len(tensors)
    3
    >>> tensors[0].shape
    (3, 256, 256)
    >>> all(t.dtype == np.float32 for t in tensors)
    True
    >>> all(np.all(t >= -1.0) and np.all(t <= 1.0) for t in tensors)
    True
    """
    # --- input validation ---
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(
            f"image must be 2-D [H, W], got shape {image.shape}. "
            "Squeeze the channel dimension before calling."
        )
    if len(sigmas) != 3:
        raise ValueError(
            f"sigmas must be a 3-tuple (fine, medium, coarse), got length {len(sigmas)}."
        )
    for i, s in enumerate(sigmas):
        if s <= 0.0:
            raise ValueError(
                f"sigmas[{i}] must be > 0, got {s}."
            )

    # --- compute at each scale (delegates entirely to A3 function) ---
    result: List[np.ndarray] = [
        compute_structure_tensor(image, rho=rho, sigma=s, normalise=normalise)
        for s in sigmas
    ]
    # result[0]: s1  [3, H, W]  fine
    # result[1]: s2  [3, H, W]  medium
    # result[2]: s3  [3, H, W]  coarse
    return result
