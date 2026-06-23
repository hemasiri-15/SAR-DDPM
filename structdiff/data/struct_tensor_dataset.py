"""
structdiff/data/struct_tensor_dataset.py
=========================================
A3: Structure Tensor Dataset — dataset component.

Subclasses MultiLookDataset (structdiff/data/multi_look_dataset.py) and
appends a pre-computed structure tensor to every returned sample.

Design contract
---------------
- MultiLookDataset is NOT modified.
- The U-Net, TrainLoop, GaussianDiffusion are NOT modified here.
- The only behavioural change vs the parent is:
    (a) ``compute_structure_tensor`` is called on the noisy amplitude
        image (before [-1,1] normalisation) at dataset time.
    (b) A 5th element ``struct_tensor: torch.Tensor [3, H, W]`` is
        appended to the return tuple.
- At A3 fine-tuning start, ``StructTensorEncoder.proj`` is near-zero,
  so the struct signal contributes approximately zero to ``emb`` and
  the model starts from A2 behaviour.

Return tuple
------------
(clean_tensor, noisy_tensor, image_filename, look_num, struct_tensor)

clean_tensor   : torch.Tensor [C, H, W]  float32  [-1, 1]
noisy_tensor   : torch.Tensor [C, H, W]  float32  [-1, 1]
image_filename : str
look_num       : int   (collated to [B] int64 by DataLoader)
struct_tensor  : torch.Tensor [3, H, W]  float32  [-1, 1]

The structure tensor is always 3 channels regardless of num_channels
(it is not channel-repeated), because its 3 components (J11, J12, J22)
carry distinct physical information.

Structure tensor computation
-----------------------------
Input to ``compute_structure_tensor``: the noisy amplitude array
[H, W] float32 in [0, 1], extracted from ``noisy_array`` before the
final [-1,1] normalisation step.  This matches inference conditions:
at inference the model receives a noisy image, not the clean reference,
so the structure tensor must be computed from the noisy signal.

Default scales: ρ=1.0 (pre-smooth), σ=5.0 (integration).
Both are constructor parameters for ablation studies.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Path resolution: inherit from MultiLookDataset in structdiff/data/
# ---------------------------------------------------------------------------
import os
import sys

_STRUCTDIFF_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__))
)
_STRUCTDIFF_DIR = os.path.normpath(os.path.join(_STRUCTDIFF_DATA_DIR, ".."))
if _STRUCTDIFF_DIR not in sys.path:
    sys.path.insert(0, _STRUCTDIFF_DIR)

from data.multilook_dataset import MultiLookDataset  # noqa: E402
from utils.structure_tensor import compute_structure_tensor  # noqa: E402


_DEFAULT_LOOKS: Tuple[int, ...] = (1, 2, 4, 8, 10)


class StructTensorDataset(MultiLookDataset):
    """SAR synthetic dataset with multi-look speckle and structure tensor.

    Extends :class:`MultiLookDataset` for Stage A3 of StructDiff-SAR.
    Every sample additionally returns a pre-computed structure tensor
    map derived from the noisy amplitude image.

    Return tuple
    ------------
    (clean_tensor, noisy_tensor, image_filename, look_num, struct_tensor)

    struct_tensor : torch.Tensor, shape [3, H, W], float32, range [-1, 1]
        Structure tensor components (J11, J12, J22), computed from the
        noisy amplitude image before [-1,1] normalisation.  Each channel
        is independently normalised to [-1, 1] by its absolute maximum.

    Parameters
    ----------
    dataset_path:
        Root directory containing PNG/JPG images.
    train:
        Training mode (random augmentation + random L) vs eval mode
        (centre crop + deterministic L per index).
    num_channels:
        Image channels for clean/noisy tensors.  Does NOT affect
        struct_tensor, which is always [3, H, W].
    crop_size:
        (H, W) of the output patch.
    length:
        Subsample the image list (-1 = all).
    seed:
        Master RNG seed.
    looks:
        Tuple of look counts to sample from uniformly.
    rho:
        Pre-smoothing Gaussian std (pixels) for structure tensor.
        Default 1.0.  Ablation parameter.
    sigma:
        Integration Gaussian std (pixels) for structure tensor.
        Default 5.0.  Ablation parameter.
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
        sigma: float = 5.0,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            train=train,
            num_channels=num_channels,
            crop_size=crop_size,
            length=length,
            seed=seed,
            looks=looks,
        )

        if rho < 0.0:
            raise ValueError(f"rho must be >= 0, got {rho}.")
        if sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma}.")

        self.rho: float = rho
        self.sigma: float = sigma

    # ------------------------------------------------------------------
    # Item retrieval
    # ------------------------------------------------------------------

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, str, int, torch.Tensor]:
        """Return one (clean, noisy, filename, look_num, struct_tensor) tuple.

        All image loading, augmentation, look sampling, and speckle
        synthesis logic is reproduced from MultiLookDataset verbatim.
        The sole addition is step 5: structure tensor computation on
        the noisy amplitude array before [-1,1] normalisation.

        We do NOT call ``super().__getitem__`` because the parent returns
        a 4-tuple and exits; there is no hook to inject tensor computation
        between speckle synthesis and normalisation.  Reproducing the body
        here is the correct pattern for this kind of mid-pipeline override.
        """
        from torch.utils.data import get_worker_info

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

        # ------ 6. Structure tensor (computed PRE-normalisation) ------
        # Input: 2-D float32 in [0, 1], matching inference conditions.
        # squeeze to [H, W] for compute_structure_tensor.
        noisy_2d: np.ndarray = noisy_array[0]  # [H, W]
        struct_np: np.ndarray = compute_structure_tensor(
            noisy_2d,
            rho=self.rho,
            sigma=self.sigma,
            normalise=True,
        )  # [3, H, W], float32, range [-1, 1]

        # ------ 7. Normalise clean and noisy to [-1, 1] ------
        clean_array = np.round(clean_image) / 127.5 - 1.0
        noisy_array = np.round(noisy_array * 255.0) / 127.5 - 1.0

        # ------ 8. Channel repetition (clean/noisy only) ------
        if self.num_channels > 1:
            clean_array = np.repeat(clean_array, self.num_channels, axis=0)
            noisy_array = np.repeat(noisy_array, self.num_channels, axis=0)
        # struct_tensor stays [3, H, W] — NOT channel-repeated.

        # ------ 9. Return 5-tuple ------
        # struct_tensor is float32 [3, H, W]; DataLoader collates to [B, 3, H, W].
        return (
            torch.tensor(clean_array),
            torch.tensor(noisy_array),
            image_filename,
            look_num,
            torch.from_numpy(struct_np),
        )
