"""
structdiff/data/spectral_tensor_dataset.py
============================================
A11: SpectralTensorDataset — dataset conditioning module.

Subclasses ``MultiScaleStructTensorDataset`` (A10,
structdiff/data/multiscale_struct_tensor_dataset.py, UNCHANGED).
Appends A11 spectral features (eigenvalues, anisotropy, coherence,
per A10 scale) computed from A10's structure tensors via
``compute_spectral_features_multiscale``
(structdiff/utils/tensor_spectral_features.py).

Design contract
----------------
- ``MultiScaleStructTensorDataset`` (A10) is NOT modified and is called
  via ``super().__getitem__(idx)`` -- not reproduced. Unlike A3 -> A10
  (which had no hook point and required body reproduction), A10's
  ``__getitem__`` returns a COMPLETE 7-tuple including ``(s1, s2, s3)``,
  so A11 has everything it needs without re-deriving images, speckle,
  or gradients.
- ``compute_structure_tensor`` / ``compute_structure_tensor_multiscale``
  (A3 / A10 gradient computation) are NOT called again -- A11 operates
  purely on A10's already-computed ``s1``/``s2``/``s3`` tensors.
- This class is a pure dataset; it owns no training-loop logic.

Return tuple
------------
``__getitem__`` returns an 8-tuple::

    (clean_tensor, noisy_tensor, image_filename, look_num,
     struct_tensor_s1, struct_tensor_s2, struct_tensor_s3,
     spectral_tensor)

    clean_tensor      [C, H, W]   float32  in [-1, 1]
    noisy_tensor      [C, H, W]   float32  in [-1, 1]
    image_filename    str
    look_num          int         (collated by DataLoader -> [B] int64)
    struct_tensor_s1  [3, H, W]   float32  in [-1, 1]   sigma1 (fine)
    struct_tensor_s2  [3, H, W]   float32  in [-1, 1]   sigma2 (medium)
    struct_tensor_s3  [3, H, W]   float32  in [-1, 1]   sigma3 (coarse)
    spectral_tensor   [12, H, W]  float32  -- see compute_spectral_features_multiscale
                                  for channel layout (4 features x 3 scales,
                                  concatenated fine -> medium -> coarse)

After DataLoader collation, ``spectral_tensor`` -> [B, 12, H, W].

Consumed by the A11-patched ``train_util.py``, which unpacks this
8-tuple and passes ``model_kwargs['spectral_tensor'] = spectral_tensor``
in addition to A10's ``model_kwargs['struct_tensors'] = (s1, s2, s3)``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from structdiff.data.multiscale_structure_tensor_dataset import (
    MultiScaleStructTensorDataset,
)
from structdiff.utils.spectral_tensor_features import (
    compute_spectral_features_multiscale,
    _EPS,
    _DEFAULT_CLIP_RANGE,
)


class SpectralTensorDataset(MultiScaleStructTensorDataset):
    """A10 ``MultiScaleStructTensorDataset`` subclass adding A11 spectral features.

    Every sample additionally returns a ``spectral_tensor`` of shape
    [12, H, W] -- per-scale (lambda1, lambda2, anisotropy, coherence)
    derived from A10's ``s1``/``s2``/``s3`` structure tensors via
    ``compute_spectral_features_multiscale``. A10's
    ``(clean, noisy, fname, look_num, s1, s2, s3)`` 7-tuple is obtained
    via ``super().__getitem__(idx)`` and returned unchanged as the
    first seven elements.

    Parameters
    ----------
    *args, **kwargs:
        Forwarded unchanged to ``MultiScaleStructTensorDataset.__init__``
        (e.g. ``dataset_path``, ``train``, ``num_channels``,
        ``crop_size``, ``length``, ``seed``, ``looks``, ``rho``,
        ``sigma``, ``sigma1``, ``sigma2``, ``sigma3``).
    spectral_eps : float, optional
        Denominator regulariser for anisotropy/coherence (A11 spec
        default ``1e-8``). Forwarded to
        ``compute_spectral_features_multiscale``.
    spectral_clip_range : tuple[float, float] | None, optional
        Clip range applied to anisotropy/coherence after computation
        (default ``(-10.0, 10.0)``; see
        structdiff/utils/tensor_spectral_features.py "Numerical note").
        Set to ``None`` to disable clipping.

    Attributes
    ----------
    spectral_eps : float
        Stored value of the above.
    spectral_clip_range : tuple[float, float] | None
        Stored value of the above.
    """

    def __init__(
        self,
        *args,
        spectral_eps: float = _EPS,
        spectral_clip_range: Optional[Tuple[float, float]] = _DEFAULT_CLIP_RANGE,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.spectral_eps: float = spectral_eps
        self.spectral_clip_range: Optional[Tuple[float, float]] = spectral_clip_range

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        str,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return the A10 7-tuple plus an A11 ``spectral_tensor`` -- an 8-tuple.

        Calls ``MultiScaleStructTensorDataset.__getitem__`` (A10,
        unmodified) for ``(clean, noisy, fname, look_num, s1, s2, s3)``,
        then computes ``spectral_tensor = compute_spectral_features_multiscale(
        s1, s2, s3)`` from those structure tensors -- no image loading,
        augmentation, speckle synthesis, or gradient computation is
        repeated.
        """
        clean, noisy, fname, look_num, s1, s2, s3 = super().__getitem__(idx)

        # s1/s2/s3 are torch.Tensor [3, H, W], float32 (A10 output).
        # compute_spectral_features_multiscale is pure NumPy.
        spectral_np = compute_spectral_features_multiscale(
            s1.numpy(),
            s2.numpy(),
            s3.numpy(),
            eps=self.spectral_eps,
            clip_range=self.spectral_clip_range,
        )  # [12, H, W], float32

        spectral_tensor = torch.from_numpy(spectral_np)

        return clean, noisy, fname, look_num, s1, s2, s3, spectral_tensor
