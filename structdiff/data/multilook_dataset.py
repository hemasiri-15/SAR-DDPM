"""
structdiff/data/multi_look_dataset.py
======================================
A1: Multi-Look Training — dataset component.

Subclasses SynthSARDataset (scripts/datasets.py) and replaces the fixed
single-look speckle model with a randomly sampled L-look model.

Design contract
---------------
- SynthSARDataset is NOT modified.
- The U-Net and TrainLoop are NOT modified at this stage (A1).
- The only behavioural change vs the parent is:
    (a) L is drawn uniformly from ``looks`` on every training call.
    (b) Gamma(shape=L, scale=1/L) replaces Gamma(shape=1, scale=1).
    (c) A 4th element ``look_num: int`` is appended to the return tuple.
- At L=1 the distribution is mathematically identical to the parent
  (Gamma(1, 1) == Exponential(1)), so any A0 checkpoint can be fine-tuned
  with MultiLookDataset without a warm-up period or learning-rate reset.

Speckle physics reminder
------------------------
For an L-look SAR intensity image the speckle factor G satisfies:

    G ~ Gamma(shape=L, scale=1/L),   E[G] = 1,  Var[G] = 1/L.

The parent uses Gamma(shape=1, scale=1) — the L=1 (single-look) special
case.  More looks → lower variance → less speckle → smoother image.
This implementation works in amplitude:

    noisy_amplitude = sqrt(clean_intensity * G)
                    = sqrt((clean / 255)^2 * G)

which matches the parent's convention exactly at L=1.

RNG design
----------
The parent maintains two lazy RNGs:
    gamma_rng      — draws speckle noise
    transform_rng  — draws spatial augmentation parameters

Both are initialised in _load_rng(), which is called lazily on the first
__getitem__ call in training mode.  DataLoader workers fork *after*
__init__, so lazy init correctly gives each worker an independent stream.

MultiLookDataset adds a third RNG:
    look_rng       — draws the look count L for each sample

look_rng is seeded from a third child of the same SeedSequence used by the
parent, ensuring it is independent from gamma_rng and transform_rng.
For validation, look_rng is re-seeded deterministically per index (with a
large prime offset from gamma_rng's seed to keep the two streams distinct).

Checkpoint compatibility (A0 → A1)
------------------------------------
MultiLookDataset changes only the dataset, not the model.  A0 model weights
load without modification.  At L=1 the speckle distribution is identical to
A0, so fine-tuning can start immediately from any A0 checkpoint.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence, Tuple

import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import get_worker_info

# ---------------------------------------------------------------------------
# Locate and import SynthSARDataset from scripts/datasets.py.
#
# scripts/ is not a package (no __init__.py), so we add it to sys.path.
# We insert at position 0 to take priority over any other datasets.py that
# might be on the path (e.g. the HuggingFace one).
# ---------------------------------------------------------------------------
_SCRIPTS_DIR: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from datasets import SynthSARDataset  # noqa: E402  (after sys.path patch)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default look set from the StructDiff-SAR design doc § A1.
#: Standard multi-look counts used in SAR processing literature.
#: Stored as a sorted tuple; callers may override via the ``looks`` argument.
_DEFAULT_LOOKS: Tuple[int, ...] = (1, 2, 4, 8, 10)

#: Prime offset added to the validation look_rng SeedSequence to guarantee
#: independence from gamma_rng even though both are seeded with (seed, idx).
_LOOK_RNG_PRIME_OFFSET: int = 1_000_003


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class MultiLookDataset(SynthSARDataset):
    """SAR synthetic dataset with randomly sampled multi-look speckle.

    Extends :class:`SynthSARDataset` for Stage A1 of StructDiff-SAR.

    During **training** the number of looks *L* is drawn uniformly from
    ``looks`` for every sample, giving the model exposure to the full range
    of speckle levels in a single training run.

    During **validation / test** *L* is assigned deterministically per index
    (using a seeded draw) so that evaluation metrics are reproducible across
    runs and between processes.

    Return tuple
    ------------
    ``(clean_tensor, noisy_tensor, image_filename, look_num)``

    clean_tensor : torch.Tensor  —  shape [C, H, W], float32, range [−1, 1]
        Clean reference image.  Identical to :class:`SynthSARDataset` output.
    noisy_tensor : torch.Tensor  —  shape [C, H, W], float32, range [−1, 1]
        Speckled image drawn with Gamma(L, 1/L) speckle factor.
    image_filename : str
        Absolute path to the source image.
    look_num : int
        The look count *L* used to generate this sample.
        DataLoader's default ``collate_fn`` converts a list of Python ints
        into a ``torch.Tensor`` of shape ``[B]``, dtype ``torch.int64``.
        At A1 the training script logs this value.
        At A2 it becomes a conditioning signal fed into ``LookEmbedding``.

    Parameters
    ----------
    dataset_path : str
        Root directory containing PNG/JPG images (searched recursively).
    train : bool
        ``True``  → random augmentations, L drawn stochastically per sample.
        ``False`` → centre crop, L assigned deterministically per index.
    num_channels : int
        Number of output channels.  Single-channel images are repeated.
    crop_size : Tuple[int, int]
        Output patch size ``(H, W)``.
    length : int
        Subsample the image list to this many entries.  ``-1`` means all.
    seed : int or None
        Master seed for all RNGs.  ``None`` yields non-deterministic runs.
    looks : Sequence[int]
        Look counts to sample from uniformly.
        Default: ``(1, 2, 4, 8, 10)``.
        Duplicates are silently removed; values are sorted canonically.
        Every element must be >= 1.
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
    ) -> None:
        # -------------------------------------------------------------------
        # 1. Initialise the parent in full.
        #
        # After super().__init__() the following attributes are available:
        #   self.images_list       — list[str], all image paths
        #   self.rng_rng           — Generator used to seed child RNGs
        #   self.gamma_rng         — None until _load_rng() is called
        #   self.transform_rng     — None until _load_rng() is called
        #   self.loaded_rng        — bool, False until _load_rng() is called
        #   self.seed              — the master seed (int or 0 if None)
        #   self.train             — bool
        #   self.num_channels      — int
        #   self.horizontal_flip,  self.vertical_flip, self.rotation,
        #   self.crop, self.center_crop  — torchvision / albumentations transforms
        # -------------------------------------------------------------------
        super().__init__(
            dataset_path=dataset_path,
            train=train,
            num_channels=num_channels,
            crop_size=crop_size,
            length=length,
            seed=seed,
        )

        # -------------------------------------------------------------------
        # 2. Validate and store the look set.
        #
        # We convert to a sorted tuple so the ordering is canonical regardless
        # of the caller's container type, and de-duplicate via set() so that
        # passing (1, 1, 2) is treated the same as (1, 2).
        # -------------------------------------------------------------------
        if len(looks) == 0:
            raise ValueError("`looks` must contain at least one element.")
        if any(l < 1 for l in looks):
            raise ValueError(
                f"All look counts must be >= 1.  Received: {list(looks)}"
            )
        self.looks: Tuple[int, ...] = tuple(sorted(set(int(l) for l in looks)))

        # -------------------------------------------------------------------
        # 3. Declare look_rng.
        #
        # Set to None here and populated by _load_rng() on the first
        # __getitem__ call (training) or inline in __getitem__ (validation).
        # Declaring it on the instance mirrors the parent's pattern for
        # gamma_rng and transform_rng and keeps type-checkers happy.
        # -------------------------------------------------------------------
        self.look_rng: np.random.Generator | None = None

    # -----------------------------------------------------------------------
    # RNG initialisation
    # -----------------------------------------------------------------------

    def _load_rng(self) -> None:
        """Initialise gamma_rng, transform_rng, and look_rng atomically.

        Overrides the parent's ``_load_rng``.

        The parent spawns 2 child SeedSequences from a single draw on
        ``rng_rng``.  We spawn 3 children so that ``look_rng`` has its own
        independent entropy stream, while keeping the ``rng_rng`` consumption
        rate identical to the parent (one ``size=2`` draw).

        **Worker safety**
        DataLoader forks workers *after* ``__init__``.  RNG state is therefore
        cloned into each worker.  The parent incorporates ``worker_id`` into
        the seed so each worker produces a disjoint stream.  We follow the
        same pattern for ``look_rng``.

        Seeding hierarchy (mirrors the parent for indices 0 and 1)::

            rng_rng  ──draws──►  [rand_0, rand_1]
                                         │
                            SeedSequence([seed, worker_id, rand_0, rand_1])
                                         │
                         ┌───────┬───────┴───────┐
                    spawn(0)  spawn(1)        spawn(2)
                       │          │                │
                  gamma_rng  transform_rng    look_rng
        """
        worker_info = get_worker_info()
        worker_id: int = worker_info.id if worker_info is not None else 0

        # Consume exactly size=2 from rng_rng — same as the parent.
        rand_nums = self.rng_rng.integers(
            low=0, high=np.iinfo(np.int32).max, size=2
        )

        # Spawn 3 children:
        #   [0] → gamma_rng      (same seed path as parent child 0)
        #   [1] → transform_rng  (same seed path as parent child 1)
        #   [2] → look_rng       (new, independent entropy stream)
        seed_seq = np.random.SeedSequence(
            [self.seed, worker_id, rand_nums[0], rand_nums[1]]
        ).spawn(3)

        self.gamma_rng = np.random.default_rng(seed_seq[0])
        self.transform_rng = np.random.default_rng(seed_seq[1])
        self.look_rng = np.random.default_rng(seed_seq[2])

        self.loaded_rng = True

    # -----------------------------------------------------------------------
    # Item retrieval
    # -----------------------------------------------------------------------

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        """Return one ``(clean, noisy, filename, look_num)`` tuple.

        All image loading, augmentation, and normalisation logic is reproduced
        from the parent with the sole change being multi-look speckle synthesis.

        We do **not** call ``super().__getitem__`` because the parent returns a
        3-tuple and there is no hook point to inject look sampling between
        speckle generation and the return statement.  Reproducing the body here
        is the correct pattern for this type of attribute-level override.

        Parameters
        ----------
        idx : int
            Index into ``self.images_list``.

        Returns
        -------
        clean_tensor : torch.Tensor
            Shape ``[C, H, W]``, float32, range ``[−1, 1]``.
        noisy_tensor : torch.Tensor
            Shape ``[C, H, W]``, float32, range ``[−1, 1]``.
        image_filename : str
            Absolute path to the source image.
        look_num : int
            Look count *L* used to generate this sample.
        """

        # -------------------------------------------------------------------
        # Step 1 — Lazy RNG initialisation (training mode only).
        #
        # In validation mode the RNGs are seeded inline per-index (step 3b).
        # In training mode we initialise once per worker, lazily, so that
        # DataLoader's fork has already happened and workers get distinct seeds.
        # -------------------------------------------------------------------
        if self.train and not self.loaded_rng:
            self._load_rng()

        # -------------------------------------------------------------------
        # Step 2 — Image loading.
        #
        # Convert to greyscale ("L" mode) — identical to the parent.
        # -------------------------------------------------------------------
        image_filename: str = self.images_list[idx]
        image = PILImage.open(image_filename).convert("L")

        # -------------------------------------------------------------------
        # Step 3 — Spatial transforms.
        #
        # 3a  Training: random flip / rotation / crop via transform_rng.
        # 3b  Validation: centre crop + per-index deterministic RNG seeding.
        # -------------------------------------------------------------------
        if self.train:
            # Draw 5 integers for the augmentation transforms.
            rand_nums = self.transform_rng.integers(
                low=0, high=np.iinfo(np.int32).max, size=5
            )
            image = self.horizontal_flip(image, rand_nums[0])
            image = self.vertical_flip(image, rand_nums[1])
            image = self.rotation(image, rand_nums[2])
            image = self.crop(image, rand_nums[3], rand_nums[4])
        else:
            # Validation / test: deterministic centre crop.
            image = self.center_crop(image)

            # Re-seed gamma_rng per index so that the same validation image
            # always receives the same noise draw.  Mirrors the parent exactly.
            self.gamma_rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, idx])
            )

            # Re-seed look_rng per index with a prime-offset entropy value to
            # guarantee independence from gamma_rng even though both are seeded
            # with (seed, idx).  The same validation image therefore always gets
            # the same L, making evaluation fully reproducible across runs.
            self.look_rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, idx, _LOOK_RNG_PRIME_OFFSET])
            )

        # -------------------------------------------------------------------
        # Step 4 — Sample look count L.
        #
        # Uniform draw from self.looks on every call.
        # For training: look_rng is the worker-safe independent stream.
        # For val/test: look_rng was just re-seeded deterministically above.
        #
        # np.array wrapping ensures correct dtype for rng.choice (int64).
        # int() unwraps the numpy scalar to a plain Python int so that
        # DataLoader's default collate_fn can batch it into torch.int64.
        # -------------------------------------------------------------------
        look_num: int = int(
            self.look_rng.choice(np.array(self.looks, dtype=np.int64))
        )

        # -------------------------------------------------------------------
        # Step 5 — Speckle synthesis.
        #
        # Power-domain model: the Gamma factor multiplies intensity, not
        # amplitude.  We therefore convert to intensity, apply speckle, then
        # take the square root back to amplitude.
        #
        #   clean_intensity  = (pixel / 255)^2        ∈ [0, 1]
        #   G                ~ Gamma(shape=L, scale=1/L)
        #   noisy_amplitude  = sqrt(clean_intensity * G)  ∈ [0, ∞)
        #   noisy_amplitude  = clip(noisy_amplitude, 0, 1)
        #
        # At L=1: Gamma(1, 1) == Exponential(1), which is exactly what the
        # parent uses.  This ensures A0-checkpoint compatibility at L=1.
        # -------------------------------------------------------------------
        clean_image = np.float32(image)           # shape: [H, W]
        clean_image = clean_image[np.newaxis, :]  # shape: [1, H, W]

        # Intensity domain.
        noisy_array = (clean_image / 255.0) ** 2  # [0, 1]

        # L-look speckle factor.  Key properties:
        #   E[G]   = shape * scale = L * (1/L) = 1   (mean-preserving)
        #   Var[G] = shape * scale² = L * (1/L²) = 1/L  (falls with L)
        gamma_noise: np.ndarray = self.gamma_rng.gamma(
            shape=float(look_num),          # α = L
            scale=1.0 / float(look_num),   # β = 1/L
            size=noisy_array.shape,
        ).astype(noisy_array.dtype)

        # Back to amplitude, clipped to [0, 1].
        noisy_array = np.clip(np.sqrt(noisy_array * gamma_noise), 0.0, 1.0)

        # -------------------------------------------------------------------
        # Step 6 — Normalisation to [−1, 1].
        #
        # Matches the parent's rounding convention exactly, ensuring any
        # consumer expecting values in [−1, 1] (TrainLoop, noise schedule,
        # ConditionedSuperResModel) receives identically formatted tensors.
        # -------------------------------------------------------------------
        # Clean: uint8 pixel values → round (noop for integer pixels) → scale.
        clean_array: np.ndarray = np.round(clean_image) / 127.5 - 1.0

        # Noisy: amplitude in [0, 1] → scale to [0, 255] → round → scale.
        noisy_array = np.round(noisy_array * 255.0) / 127.5 - 1.0

        # -------------------------------------------------------------------
        # Step 7 — Channel repetition.
        #
        # The SR U-Net expects ``in_channels`` input channels.  Single-channel
        # SAR images are repeated along axis 0.
        # -------------------------------------------------------------------
        if self.num_channels > 1:
            clean_array = np.repeat(clean_array, self.num_channels, axis=0)
            noisy_array = np.repeat(noisy_array, self.num_channels, axis=0)

        # -------------------------------------------------------------------
        # Step 8 — Return 4-tuple.
        #
        # look_num is a plain Python int.  DataLoader's default collate_fn
        # collects a list of ints and converts it to torch.Tensor([...],
        # dtype=torch.int64) of shape [B].  No custom collate_fn is required.
        # -------------------------------------------------------------------
        return (
            torch.tensor(clean_array, dtype=torch.float32),
            torch.tensor(noisy_array, dtype=torch.float32),
            image_filename,
            look_num,
        )
