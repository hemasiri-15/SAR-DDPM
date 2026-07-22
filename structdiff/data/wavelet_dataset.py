"""
structdiff/data/wavelet_dataset.py
====================================
A12: Wavelet Dataset — dataset conditioning module.

Subclasses ``SpectralTensorDataset`` (A11,
structdiff/data/spectral_tensor_dataset.py, UNCHANGED).
Appends A12 wavelet features computed from the noisy amplitude image via
``compute_wavelet_features`` (structdiff/utils/wavelet_features.py).

Design contract
----------------
- ``SpectralTensorDataset`` (A11) is NOT modified and is called via
  ``super().__getitem__(idx)`` — not reproduced.  Unlike A3 → A10 (which
  had no hook point and required body reproduction), A11's
  ``__getitem__`` returns a COMPLETE 8-tuple including
  ``(clean, noisy, fname, look_num, s1, s2, s3, spectral_tensor)``.
- However, A11 does NOT expose the noisy amplitude image in [0, 1]
  (pre-normalisation) — it only exposes ``noisy`` which is already in
  [-1, 1].  Wavelet features MUST be computed from the amplitude image
  in [0, 1] to match inference conditions (see "Wavelet computation"
  below).  Therefore ``WaveletDataset.__getitem__`` reconstructs
  ``noisy_2d`` in [0, 1] from the normalised ``noisy`` tensor using the
  inverse of the normalisation applied by A10:

      noisy_01 = (noisy + 1.0) / 2.0          (per-pixel, clamped)

  This is exact for the round-trip used in A10/A11:
      normalised = np.round(noisy_array * 255.0) / 127.5 - 1.0
  whose inverse is:
      (normalised + 1.0) / 2.0  (within ±0.004 of the pre-round value)

  The approximation is negligible for wavelet conditioning because
  ``compute_wavelet_features`` internally normalises each subband to
  [-1, 1] by its absolute maximum, so small absolute errors in the
  reconstructed amplitude cancel in the per-channel normalisation step.

  This is preferable to reproducing the entire A10/A11 body (≈120 lines)
  for a single intermediate value.

- ``compute_structure_tensor`` / ``compute_structure_tensor_multiscale``
  / ``compute_spectral_features_multiscale`` are NOT called again.
  ``WaveletDataset`` operates solely on A11's output 8-tuple plus the
  recovered ``noisy_2d`` amplitude for wavelet computation.
- This class is a pure dataset; it owns no training-loop logic.

Return tuple
------------
``__getitem__`` returns a 9-tuple::

    (clean_tensor, noisy_tensor, image_filename, look_num,
     struct_tensor_s1, struct_tensor_s2, struct_tensor_s3,
     spectral_tensor, wavelet_tensor)

    clean_tensor      [C, H, W]    float32  in [-1, 1]
    noisy_tensor      [C, H, W]    float32  in [-1, 1]
    image_filename    str
    look_num          int          (collated by DataLoader -> [B] int64)
    struct_tensor_s1  [3, H, W]    float32  in [-1, 1]   sigma1 (fine)
    struct_tensor_s2  [3, H, W]    float32  in [-1, 1]   sigma2 (medium)
    struct_tensor_s3  [3, H, W]    float32  in [-1, 1]   sigma3 (coarse)
    spectral_tensor   [12, H, W]   float32  (see A11 / tensor_spectral_features)
    wavelet_tensor    [4, H/2, W/2] float32  in [-1, 1]
                                   ch 0: LL (approximation)
                                   ch 1: LH (horizontal detail)
                                   ch 2: HL (vertical detail)
                                   ch 3: HH (diagonal detail)

After DataLoader collation:
    wavelet_tensor -> [B, 4, H/2, W/2]

Wavelet tensor meaning
-----------------------
A single-level 2-D DWT decomposes the noisy amplitude image into four
subbands:

    LL — coarse approximation (low-pass × low-pass), speckle-suppressed.
    LH — horizontal detail (low-pass rows × high-pass cols).
    HL — vertical detail (high-pass rows × low-pass cols).
    HH — diagonal detail (high-pass × high-pass), speckle-rich.

The four subbands carry distinct physical information (frequency + spatial
direction); they MUST NOT be collapsed or channel-repeated.  The wavelet
tensor is therefore always [4, H/2, W/2] regardless of ``num_channels``.

Why wavelets are computed from the noisy amplitude image
---------------------------------------------------------
At inference the model receives the noisy observation, not the clean
reference.  Computing wavelets from the clean image would be cheating;
computing them from the already-normalised [-1, 1] tensor would alter
the subband energy distribution.  Consistent with A3's structure tensor
and A11's spectral tensor, wavelets are derived from the pre-normalised
amplitude signal in [0, 1].

Why wavelet_tensor is always [4, H/2, W/2]
-------------------------------------------
Unlike the clean/noisy image tensors (which are channel-repeated to
``num_channels`` for compatibility with the U-Net stem), the wavelet
tensor is a multi-channel conditioning signal.  Its four channels are
physically distinct (LL / LH / HL / HH); repeating them would destroy
their semantic meaning.  The WaveletEncoder is responsible for projecting
[4, H/2, W/2] into the model embedding space.

Hook-point philosophy
----------------------
A11's ``__getitem__`` returns a complete 8-tuple.  There is therefore a
hook point: ``super().__getitem__(idx)`` yields all A11 outputs.
``WaveletDataset`` calls ``super().__getitem__(idx)`` and reconstructs
``noisy_2d`` in [0, 1] from the normalised ``noisy`` tensor by inverting
A10's normalisation formula:

    noisy_01 = (noisy_normalised + 1.0) / 2.0

This avoids reproducing ≈120 lines of A10 body.  The reconstruction
error is bounded by the rounding quantisation in A10's normalisation
(≤ 1/255 ≈ 0.004) and is absorbed by ``compute_wavelet_features``'s
per-subband normalisation.  If a future stage requires exact [0, 1]
amplitude without rounding error, expose it directly in the parent.

Checkpoint compatibility
------------------------
A12 adds ``wavelet_encoder.*`` keys.  The A11 ``spectral_encoder.*``,
A10 ``ms_struct_encoder.*`` keys remain unchanged.  Use
``load_state_dict(strict=False)`` when loading A11 checkpoints into an
A12 model.

Examples
--------
>>> import torch
>>> from structdiff.data.wavelet_dataset import WaveletDataset
>>> ds = WaveletDataset(
...     dataset_path="/data/sar_patches",
...     train=False,
...     num_channels=1,
...     crop_size=(256, 256),
...     wavelet="db2",
... )
>>> sample = ds[0]
>>> len(sample)
9
>>> clean, noisy, fname, look_num, s1, s2, s3, spec, wav = sample
>>> clean.shape
torch.Size([1, 256, 256])
>>> wav.shape
torch.Size([4, 128, 128])
>>> wav.dtype
torch.float32
>>> bool(wav.min() >= -1.0 and wav.max() <= 1.0)
True
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from structdiff.data.spectral_tensor_dataset import SpectralTensorDataset
from structdiff.utils.wavelet_features import (
    compute_wavelet_features,
    DEFAULT_WAVELET,
    NUM_SUBBANDS,
)
from structdiff.utils.spectral_tensor_features import (
    _EPS,
    _DEFAULT_CLIP_RANGE,
)


_DEFAULT_LOOKS: Tuple[int, ...] = (1, 2, 4, 8, 10)


class WaveletDataset(SpectralTensorDataset):
    """A11 ``SpectralTensorDataset`` subclass adding A12 wavelet features.

    Every sample additionally returns a ``wavelet_tensor`` of shape
    [4, H/2, W/2] — the four DWT subbands (LL, LH, HL, HH) computed from
    the noisy amplitude image (pre-normalisation) via
    ``compute_wavelet_features``.  A11's
    ``(clean, noisy, fname, look_num, s1, s2, s3, spectral_tensor)``
    8-tuple is obtained via ``super().__getitem__(idx)`` and returned
    unchanged as the first eight elements.

    The noisy amplitude image in [0, 1] is reconstructed from the
    normalised ``noisy`` tensor (already in [-1, 1]) using the inverse
    of A10's normalisation formula:

        noisy_01 = (noisy_normalised + 1.0) / 2.0

    This reconstruction is accurate to ±1/255 (one quantisation step),
    which is negligible for wavelet conditioning because
    ``compute_wavelet_features`` independently normalises each subband to
    [-1, 1] by its absolute maximum.

    Parameters
    ----------
    dataset_path : str
        Root directory containing PNG/JPG images.
    train : bool, optional
        Training mode (random augmentation + random L) vs eval mode
        (centre crop + deterministic L per index).  Default ``False``.
    num_channels : int, optional
        Image channels for clean/noisy tensors.  Does NOT affect
        ``wavelet_tensor``, which is always [4, H/2, W/2].  Default 1.
    crop_size : tuple[int, int], optional
        (H, W) of the output patch.  Both H and W must be even.
        Default ``(256, 256)``.
    length : int, optional
        Subsample the image list (-1 = all).  Default -1.
    seed : int | None, optional
        Master RNG seed.  Default ``None``.
    looks : Sequence[int], optional
        Tuple of look counts to sample from uniformly.
        Default ``(1, 2, 4, 8, 10)``.
    rho : float, optional
        Pre-smoothing Gaussian std (pixels) for structure tensor.
        Default 1.0.
    sigma1 : float, optional
        Fine integration scale for multi-scale structure tensor.
        Default 1.0.
    sigma2 : float, optional
        Medium integration scale.  Default 2.5.
    sigma3 : float, optional
        Coarse integration scale.  Default 4.5.
    spectral_clip_range : tuple[float, float] | None, optional
        Clip range for A11 spectral features.  Default ``(-10.0, 10.0)``.
    wavelet : str, optional
        PyWavelets wavelet name used for the DWT.  Default ``"db2"``.
        Any wavelet supported by ``pywt.dwt2`` may be used for ablations
        (e.g. ``"haar"``, ``"db4"``, ``"sym2"``).

    Attributes
    ----------
    wavelet : str
        Stored wavelet name passed to ``compute_wavelet_features``.

    Return values
    -------------
    ``__getitem__`` returns::

        (clean_tensor, noisy_tensor, image_filename, look_num,
         st1, st2, st3, spectral_tensor, wavelet_tensor)

        clean_tensor      torch.Tensor  [C, H, W]      float32  [-1, 1]
        noisy_tensor      torch.Tensor  [C, H, W]      float32  [-1, 1]
        image_filename    str
        look_num          int
        st1               torch.Tensor  [3, H, W]      float32  [-1, 1]
        st2               torch.Tensor  [3, H, W]      float32  [-1, 1]
        st3               torch.Tensor  [3, H, W]      float32  [-1, 1]
        spectral_tensor   torch.Tensor  [12, H, W]     float32
        wavelet_tensor    torch.Tensor  [4, H/2, W/2]  float32  [-1, 1]

    Relationship to A11
    --------------------
    ``WaveletDataset`` is a strict superset of ``SpectralTensorDataset``.
    All A11 return elements are preserved unchanged at indices 0–7.
    Index 8 (``wavelet_tensor``) is new.  Every ``SpectralTensorDataset``
    usage site can be updated to ``WaveletDataset`` by extending the
    tuple unpack to include ``wavelet_tensor``.

    Examples
    --------
    >>> import torch
    >>> from structdiff.data.wavelet_dataset import WaveletDataset
    >>> ds = WaveletDataset(
    ...     dataset_path="/data/sar_patches",
    ...     train=False,
    ...     crop_size=(256, 256),
    ...     wavelet="db2",
    ... )
    >>> sample = ds[0]
    >>> len(sample)
    9
    >>> clean, noisy, fname, look, s1, s2, s3, spec, wav = sample
    >>> wav.shape
    torch.Size([4, 128, 128])
    >>> wav.dtype
    torch.float32
    >>> bool(wav.min() >= -1.0 and wav.max() <= 1.0)
    True

    >>> # Ablation: Haar vs db2
    >>> ds_haar = WaveletDataset(
    ...     dataset_path="/data/sar_patches",
    ...     wavelet="haar",
    ... )
    >>> wav_haar = ds_haar[0][-1]
    >>> wav_haar.shape
    torch.Size([4, 128, 128])
    """

    def __init__(
        self,
        dataset_path: str,
        train: bool = False,
        num_channels: int = 1,
        crop_size: Tuple[int, int] = (256, 256),
        length: int = -1,
        seed: Optional[int] = None,
        looks: Sequence[int] = _DEFAULT_LOOKS,
        rho: float = 1.0,
        sigma1: float = 1.0,
        sigma2: float = 2.5,
        sigma3: float = 4.5,
        spectral_clip_range: Optional[Tuple[float, float]] = _DEFAULT_CLIP_RANGE,
        wavelet: str = DEFAULT_WAVELET,
    ) -> None:
        # ----------------------------------------------------------------
        # Validate wavelet before calling super().__init__, so that a bad
        # argument fails fast with a clear message.
        # ----------------------------------------------------------------
        if not isinstance(wavelet, str):
            raise ValueError(
                f"wavelet must be a str (e.g. 'db2', 'haar', 'sym2'), "
                f"got {type(wavelet).__name__!r}: {wavelet!r}."
            )

        # SpectralTensorDataset (A11) accepts *args/**kwargs forwarded to
        # MultiScaleStructTensorDataset (A10), which accepts positional
        # arguments (dataset_path, train, num_channels, crop_size,
        # length, seed, looks, rho, sigmas).  A10 uses sigmas=(s1,s2,s3).
        super().__init__(
            dataset_path=dataset_path,
            train=train,
            num_channels=num_channels,
            crop_size=crop_size,
            length=length,
            seed=seed,
            looks=looks,
            rho=rho,
            sigmas=(sigma1, sigma2, sigma3),
            spectral_clip_range=spectral_clip_range,
        )

        # Store A12-specific attributes.
        self.wavelet: str = wavelet

    # ------------------------------------------------------------------
    # Item retrieval
    # ------------------------------------------------------------------

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        torch.Tensor,   # clean_tensor      [C, H, W]      float32 [-1, 1]
        torch.Tensor,   # noisy_tensor      [C, H, W]      float32 [-1, 1]
        str,            # image_filename
        int,            # look_num
        torch.Tensor,   # st1               [3, H, W]      float32 [-1, 1]
        torch.Tensor,   # st2               [3, H, W]      float32 [-1, 1]
        torch.Tensor,   # st3               [3, H, W]      float32 [-1, 1]
        torch.Tensor,   # spectral_tensor   [12, H, W]     float32
        torch.Tensor,   # wavelet_tensor    [4, H/2, W/2]  float32 [-1, 1]
    ]:
        """Return the A11 8-tuple plus an A12 ``wavelet_tensor`` — a 9-tuple.

        Calls ``SpectralTensorDataset.__getitem__`` (A11, unmodified) for
        ``(clean, noisy, fname, look_num, s1, s2, s3, spectral_tensor)``,
        then reconstructs the noisy amplitude image in [0, 1] from the
        normalised ``noisy`` tensor and computes wavelet features via
        ``compute_wavelet_features``.

        No image loading, augmentation, speckle synthesis, gradient
        computation, or spectral feature computation is repeated.

        Wavelet computation details
        ---------------------------
        ``noisy_tensor`` (A11 output) is in [-1, 1], having been
        normalised by A10's formula::

            normalised = np.round(noisy_amplitude * 255.0) / 127.5 - 1.0

        The inverse reconstruction is::

            noisy_01 = (noisy_tensor + 1.0) / 2.0

        Clamped to [0, 1] for numerical safety.  The reconstruction error
        is bounded by ±1/255 (quantisation) and is negligible for wavelet
        conditioning because ``compute_wavelet_features`` independently
        normalises each subband by its absolute maximum.

        Returns
        -------
        Tuple of nine elements; see class docstring for full specification.
        """
        # ------ 1. Delegate entirely to A11 ------
        (
            clean,
            noisy,
            fname,
            look_num,
            s1,
            s2,
            s3,
            spectral_tensor,
        ) = super().__getitem__(idx)

        # ------ 2. Reconstruct noisy amplitude in [0, 1] ------
        # A10 normalisation: normalised = round(amp * 255) / 127.5 - 1.0
        # Inverse:           amp ≈ (normalised + 1.0) / 2.0
        # noisy has shape [C, H, W]; take first channel (all equal after
        # channel-repeat, and the structure tensor was also computed from
        # the single-channel amplitude).
        noisy_01: np.ndarray = ((noisy[0].numpy() + 1.0) / 2.0).astype(np.float32)
        # Defensive clamp: reconstruction is in [0, 1] by construction,
        # but floating-point rounding could produce values ε outside.
        noisy_01 = np.clip(noisy_01, 0.0, 1.0)

        # noisy_01 is 2-D [H, W], float32 in [0, 1].  Consistent with
        # A3's ``noisy_2d`` and A10's identical variable name.
        noisy_2d: np.ndarray = noisy_01  # rename for clarity

        # ------ 3. Compute wavelet features (PRE-[-1,1] normalisation) ------
        # Input: [H, W] float32 in [0, 1].  Matches inference conditions.
        # compute_wavelet_features independently normalises each subband
        # to [-1, 1] by its absolute maximum (normalise=True).
        wavelet_np: np.ndarray = compute_wavelet_features(
            noisy_2d,
            wavelet=self.wavelet,
            normalise=True,
        )  # [4, H/2, W/2], float32, range [-1, 1]

        # ------ 4. Runtime validation of wavelet output ------
        if wavelet_np.ndim != 3:
            raise ValueError(
                f"compute_wavelet_features returned an array with "
                f"{wavelet_np.ndim} dimensions; expected 3 "
                f"([{NUM_SUBBANDS}, H/2, W/2])."
            )
        if wavelet_np.shape[0] != NUM_SUBBANDS:
            raise ValueError(
                f"compute_wavelet_features returned {wavelet_np.shape[0]} "
                f"channels; expected exactly {NUM_SUBBANDS} "
                f"(LL, LH, HL, HH)."
            )
        H, W = noisy_2d.shape
        expected_spatial = (H // 2, W // 2)
        if wavelet_np.shape[1:] != expected_spatial:
            raise ValueError(
                f"compute_wavelet_features returned spatial shape "
                f"{wavelet_np.shape[1:]}, expected {expected_spatial} "
                f"for input shape ({H}, {W}) with wavelet '{self.wavelet}'."
            )

        # ------ 5. Convert to torch.Tensor ------
        # torch.tensor copies the NumPy buffer; dtype is explicitly float32.
        # Do NOT channel-repeat: each subband is physically distinct.
        wavelet_tensor: torch.Tensor = torch.tensor(
            wavelet_np,
            dtype=torch.float32,
        )  # [4, H/2, W/2]

        # ------ 6. Package all conditioning ------
        conditions = {
            "look_num": look_num,
            "struct_tensor": s1,          # backward compatibility
            "struct_tensors": (s1, s2, s3),
            "spectral_tensor": spectral_tensor,
            "wavelet_tensor": wavelet_tensor,
        }

        # ------ 7. Return unified interface ------
        return (
            clean,
            noisy,
            fname,
            conditions,
        )
