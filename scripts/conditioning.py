"""
Conditioning tensor generation for SAR-DDPM.

All conditioning signals are generated from the same normalized clean image.
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
        def _get(name, default):
            if args is not None and hasattr(args, name):
                return getattr(args, name)
            return default

        self.enable_multilook = _get(
            "enable_multilook", enable_multilook
        )
        self.enable_structure = _get(
            "enable_structure", enable_structure
        )
        self.enable_spectral = _get(
            "enable_spectral", enable_spectral
        )
        self.enable_wavelet = _get(
            "enable_wavelet", enable_wavelet
        )

        look_min = _get("look_min", look_min)
        look_max = _get("look_max", look_max)
        structure_kernels = _get(
            "structure_kernels", structure_kernels
        )
        wavelet_type = _get("wavelet_type", wavelet_type)

        rng = (
            rng
            if rng is not None
            else np.random.default_rng()
        )

        self._modules = {}

        if self.enable_multilook:
            self._modules["look"] = LookCondition(
                look_min=look_min,
                look_max=look_max,
                rng=rng,
            )

        if self.enable_structure:
            self._modules["structure"] = StructureCondition(
                kernels=structure_kernels
            )

        if self.enable_spectral:
            self._modules["spectral"] = SpectralCondition()

        if self.enable_wavelet:
            self._modules["wavelet"] = WaveletCondition(
                wavelet=wavelet_type
            )

    def generate(self, clean_array: np.ndarray) -> dict:
        """
        Generate all enabled conditioning tensors.

        Spectral conditioning depends on the three structure tensors,
        so structure conditioning is explicitly computed before the
        spectral module.
        """

        conditions = {}

        # ---------------------------------------------------------
        # 1. Multi-look conditioning
        # ---------------------------------------------------------
        if self.enable_multilook:
            conditions.update(
                self._modules["look"].compute(clean_array)
            )

        # ---------------------------------------------------------
        # 2. Structure conditioning
        # ---------------------------------------------------------
        # Spectral conditioning depends on these tensors.
        if self.enable_structure:
            conditions.update(
                self._modules["structure"].compute(clean_array)
            )

        # ---------------------------------------------------------
        # 3. Tensor-spectral / eigenvalue conditioning
        # ---------------------------------------------------------
        if self.enable_spectral:
            struct_tensors = conditions.get("struct_tensors")

            conditions.update(
                self._modules["spectral"].compute(
                    clean_array,
                    struct_tensors=struct_tensors,
                )
            )

        # ---------------------------------------------------------
        # 4. Wavelet conditioning
        # ---------------------------------------------------------
        if self.enable_wavelet:
            conditions.update(
                self._modules["wavelet"].compute(clean_array)
            )

        return conditions
