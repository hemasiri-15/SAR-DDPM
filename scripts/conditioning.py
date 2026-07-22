"""
conditioning.py

This module is the ONLY place in the repository that generates conditioning
tensors for the SAR-DDPM UNet. Adding a future ablation (A15, A16, ...)
means adding/editing a class in scripts/features/ and wiring it into
ConditionGenerator.compute_flags / ConditionGenerator.generate below —
nothing in datasets.py or guided_diffusion/train_util.py should ever need
to change again for a new conditioning signal.

Usage:
    conditioner = ConditionGenerator(args)
    conditions_dict = conditioner.generate(clean_array)
    # conditions_dict is a flat dict of torch Tensors (or tuples of
    # Tensors), suitable for **conditions in model_kwargs after default
    # collation.
"""

import numpy as np

from scripts.features.look import LookCondition
from scripts.features.structure import StructureCondition
from scripts.features.spectral import SpectralCondition
from scripts.features.wavelet import WaveletCondition


class ConditionGenerator:
    def __init__(
        self,
        args=None,
        enable_multilook: bool = True,
        enable_structure: bool = True,
        enable_spectral: bool = True,
        enable_wavelet: bool = True,
        look_min: int = 1,
        look_max: int = 4,
        structure_kernels=(3, 5, 9),
        wavelet_type: str = "haar",
        rng: np.random.Generator = None,
    ):
        """
        args: optional config/namespace object. If provided and it has
            matching attributes (e.g. args.enable_multilook,
            args.look_min, args.structure_kernels, ...), those override
            the keyword defaults above. This lets ablations be driven
            entirely from experiment config rather than code edits.
        rng: shared np.random.Generator for reproducibility. Pass the
            dataset's seeded RNG (see datasets.py) rather than leaving
            this None, so runs are reproducible across restarts.
        """
        def _get(name, default):
            if args is not None and hasattr(args, name):
                return getattr(args, name)
            return default

        self.enable_multilook = _get("enable_multilook", enable_multilook)
        self.enable_structure = _get("enable_structure", enable_structure)
        self.enable_spectral = _get("enable_spectral", enable_spectral)
        self.enable_wavelet = _get("enable_wavelet", enable_wavelet)

        look_min = _get("look_min", look_min)
        look_max = _get("look_max", look_max)
        structure_kernels = _get("structure_kernels", structure_kernels)
        wavelet_type = _get("wavelet_type", wavelet_type)

        rng = rng if rng is not None else np.random.default_rng()

        self._modules = {}
        if self.enable_multilook:
            self._modules["look"] = LookCondition(
                look_min=look_min, look_max=look_max, rng=rng
            )
        if self.enable_structure:
            self._modules["structure"] = StructureCondition(kernels=structure_kernels)
        if self.enable_spectral:
            self._modules["spectral"] = SpectralCondition()
        if self.enable_wavelet:
            self._modules["wavelet"] = WaveletCondition(wavelet=wavelet_type)

    def generate(self, clean_array: np.ndarray) -> dict:
        """
        clean_array: normalized single-channel array, shape (1, H, W) —
            the exact array used to build clean_tensor in the dataset,
            so every conditioning signal is computed from the same
            preprocessing pass (no duplicated/divergent preprocessing).

        Returns a flat dict merging every enabled feature module's
        output, e.g.:
            {
              "look_num": Tensor,
              "struct_tensor": Tensor,
              "struct_tensors": (Tensor, Tensor, Tensor),
              "spectral_tensor": Tensor,
              "wavelet_tensor": Tensor,
            }
        Only keys from *enabled* modules are present — a disabled
        ablation simply omits its key rather than passing a zero tensor,
        so the UNet's forward signature should treat these as optional
        kwargs (matching your existing behavior where these were always
        passed together; if the UNet requires all keys unconditionally,
        set every enable_* flag to True).
        """
        conditions = {}
        for module in self._modules.values():
            conditions.update(module.compute(clean_array))
        return conditions
