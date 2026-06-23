"""
structdiff/data/multiscale_struct_tensor_dataset.py
====================================================
A10: Multi-Scale Structure Tensor Dataset — dataset component.

Subclasses ``StructTensorDataset`` (A3) and replaces the single-scale
structure tensor computation with three-scale computation via
``compute_structure_tensor_multiscale``.

Design contract
---------------
- ``StructTensorDataset`` / ``MultiLookDataset`` are NOT modified.
- The U-Net, TrainLoop, GaussianDiffusion are NOT modified here.
- ``compute_structure_tensor`` (A3) is NOT called directly here;
  ``compute_structure_tensor_multiscale`` wraps it.
- The sole behavioural change vs the parent is:
    (a) Three structure tensor maps are computed at scales
        sigma1, sigma2, sigma3 instead of one.
    (b) The 5-tuple return is replaced by a 7-tuple return:
        (clean, noisy, fname, look_num, st1, st2, st3)

Return tuple
------------
(clean_tensor, noisy_tensor, image_filename, look_num, st1, st2, st3)

clean_tensor   : torch.Tensor  [C, H, W]   float32  [-1, 1]
noisy_tensor   : torch.Tensor  [C, H, W]   float32  [-1, 1]
image_filename : str
look_num       : int   (collated to [B] int64 by DataLoader)
st1            : torch.Tensor  [3, H, W]   float32  [-1, 1]  fine scale
st2            : torch.Tensor  [3, H, W]   float32  [-1, 1]  medium scale
st3            : torch.Tensor  [3, H, W]   float32  [-1, 1]  coarse scale

Each stN corresponds to sigmas[N-1] passed at construction time.
DataLoader collates stN to [B, 3, H, W].

Structure tensor computation
-----------------------------
Input to ``compute_structure_tensor_multiscale``: the noisy amplitude
array [H, W] float32 in [0, 1], extracted pre-normalisation.
This is identical to the A3 contract in ``StructTensorDataset``.

Checkpoint compatibility
------------------------
A10 adds ``ms_struct_encoder.*`` keys.  The A3 ``struct_encoder.*``
keys are absent (StructTensorEncoder is not instantiated here; it lives
in the encoder module).  Use ``load_state_dict(strict=False)`` when
loading A3 checkpoints into an A10 model.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Path resolution: inherit from StructTensorDataset in structdiff/data/
# ---------------------------------------------------------------------------
import os
import sys

_STRUCTDIFF_DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__)))
_STRUCTDIFF_DIR = os.path.normpath(os.path.join(_STRUCTDIFF_DATA_DIR, ".."))
if _STRUCTDIFF_DIR not in sys.path:
    sys.path.insert(0, _STRUCTDIFF_DIR)

from data.struct_tensor_dataset import StructTensorDataset          # noqa: E402
from structdiff.utils.multiscale_structure_tensor import (          # noqa: E402
    compute_structure_tensor_multiscale,
    DEFAULT_SIGMAS,
)


_DEFAULT_LOOKS: Tuple[int, ...] = (1, 2, 4, 8, 10)


class MultiScaleStructTensorDataset(StructTensorDataset):
    """SAR synthetic dataset with multi-look speckle and multi-scale structure tensors.

    Extends :class:`StructTensorDataset` for Stage A10 of StructDiff-SAR.
    Every sample returns three structure tensor maps computed at fine,
    medium, and coarse integration scales.

    Return tuple
    ------------
    (clean_tensor, noisy_tensor, image_filename, look_num, st1, st2, st3)

    st1 : torch.Tensor, shape [3, H, W], float32, range [-1, 1]  — fine
    st2 : torch.Tensor, shape [3, H, W], float32, range [-1, 1]  — medium
    st3 : torch.Tensor, shape [3, H, W], float32, range [-1, 1]  — coarse

    Parameters
    ----------
    dataset_path:
        Root directory containing PNG/JPG images.
    train:
        Training mode vs eval mode.
    num_channels:
        Image channels for clean/noisy tensors.  Does NOT affect stN.
    crop_size:
        (H, W) of the output patch.
    length:
        Subsample the image list (-1 = all).
    seed:
        Master RNG seed.
    looks:
        Tuple of look counts to sample from uniformly.
    rho:
        Pre-smoothing Gaussian std (pixels) shared across all scales.
        Default 1.0.
    sigmas:
        Three integration Gaussian stds (fine, medium, coarse).
        Default (1.0, 2.5, 4.5).  Each must be > 0.
    """

    def __init__(
        self,
        dataset_path: str,
        train: bool = False,
        num_channels: int = 1,
        crop_size: Tuple[int, int] = (256, 256),
        length: int = -1,
        seed: int | None = None,
        looks: Sequence[int] = _DEFAULT_LOOKS,
        rho: float = 1.0,
        sigmas: Tuple[float, float, float] = DEFAULT_SIGMAS,
    ) -> None:
        # Pass rho and sigma (A3 parent stores them as self.rho, self.sigma).
        # We pass sigma=sigmas[0] to satisfy the parent's __init__ signature;
        # self.sigma is not used in our overridden __getitem__.
        super().__init__(
            dataset_path=dataset_path,
            train=train,
            num_channels=num_channels,
            crop_size=crop_size,
            length=length,
            seed=seed,
            looks=looks,
            rho=rho,
            sigma=sigmas[0],   # satisfies parent validation; not used in A10 path
        )

        # Validate and store A10 sigma triplet.
        if len(sigmas) != 3:
            raise ValueError(
                f"sigmas must be a 3-tuple (fine, medium, coarse), "
                f"got length {len(sigmas)}."
            )
        for i, s in enumerate(sigmas):
            if s <= 0.0:
                raise ValueError(f"sigmas[{i}] must be > 0, got {s}.")

        self.sigmas: Tuple[float, float, float] = tuple(sigmas)  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Item retrieval
    # ------------------------------------------------------------------

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        torch.Tensor, torch.Tensor, str, int,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Return (clean, noisy, filename, look_num, st1, st2, st3).

        Reproduces the pipeline of ``StructTensorDataset.__getitem__``
        verbatim up to step 5, then replaces single-scale computation
        with ``compute_structure_tensor_multiscale``.

        Steps 1–5 (image loading, augmentation, look sampling, speckle
        synthesis) are identical to A3; we cannot call super().__getitem__
        because the parent exits with a 5-tuple before the normalisation
        step we need to intercept.
        """
        # ------ 1. Lazy RNG initialisation (training only) ------
        if self.train and not self.loaded_rng:
            self._load_rng()

        # ------ 2. Image loading ------
        image_filename: str = self.images_list[idx]
        image = PILImage.open(image_filename).convert("L")

        # ------ 3. Spatial augmentations (training) / centre crop (val) ------
        if self.train:
            rand_nums = self.transform_rng.integers(
                low=0, high=np.iinfo(np.int32).max, size=5
            )
            image = self.horizontal_flip(image, rand_nums[0])
            image = self.vertical_flip(image, rand_nums[1])
            image = self.rotation(image, rand_nums[2])
            image = self.crop(image, rand_nums[3], rand_nums[4])
        else:
            image = self.center_crop(image)
            self.gamma_rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, idx])
            )
            self.look_rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, idx, 1_000_003])
            )

        # ------ 4. Sample look count L ------
        look_num: int = int(
            self.look_rng.choice(np.array(self.looks, dtype=np.int64))
        )

        # ------ 5. Speckle synthesis ------
        clean_image = np.float32(image)
        clean_image = clean_image[np.newaxis, :, :]  # [1, H, W]

        noisy_array = (clean_image / 255.0) ** 2  # power domain

        gamma_noise = self.gamma_rng.gamma(
            size=noisy_array.shape,
            shape=float(look_num),
            scale=1.0 / float(look_num),
        ).astype(noisy_array.dtype)

        # Amplitude: [1, H, W], range [0, 1]
        noisy_array = np.clip(np.sqrt(noisy_array * gamma_noise), 0.0, 1.0)

        # ------ 6. Multi-scale structure tensors (PRE-normalisation) ------
        # Input: 2-D float32 in [0, 1], squeezed from [1, H, W].
        noisy_2d: np.ndarray = noisy_array[0]  # [H, W]

        st_list = compute_structure_tensor_multiscale(
            noisy_2d,
            rho=self.rho,
            sigmas=self.sigmas,
            normalise=True,
        )
        # st_list[0]: [3, H, W] float32 fine
        # st_list[1]: [3, H, W] float32 medium
        # st_list[2]: [3, H, W] float32 coarse

        # ------ 7. Normalise clean and noisy to [-1, 1] ------
        clean_array = np.round(clean_image) / 127.5 - 1.0
        noisy_array = np.round(noisy_array * 255.0) / 127.5 - 1.0

        # ------ 8. Channel repetition (clean/noisy only) ------
        if self.num_channels > 1:
            clean_array = np.repeat(clean_array, self.num_channels, axis=0)
            noisy_array = np.repeat(noisy_array, self.num_channels, axis=0)
        # stN tensors stay [3, H, W] — NOT channel-repeated.

        # ------ 9. Return 7-tuple ------
        return (
            torch.tensor(clean_array),
            torch.tensor(noisy_array),
            image_filename,
            look_num,
            torch.tensor(st_list[0]),   # st1: fine    [3, H, W]
            torch.tensor(st_list[1]),   # st2: medium  [3, H, W]
            torch.tensor(st_list[2]),   # st3: coarse  [3, H, W]
        )
