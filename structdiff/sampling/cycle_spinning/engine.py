"""
structdiff/sampling/cycle_spinning/engine.py
============================================
Orchestration layer for cycle-spinning aggregation.  Version 5.

Changes vs v4
-------------
ARCH-1  Full adapter suite.
        Every aggregation module is now wrapped by a private adapter that
        inherits from BaseAggregator and exposes required_features /
        forward(bundle) -> EngineResult.
        Raw nn.Module instances are never registered directly.

        New adapters:
            _LearnableAdapter               (LearnableCycleSpinning)
            _AdaptiveAdapter                (AdaptiveCycleSpinning)
            _ConfidenceAdapter              (ConfidenceCycleSpinning)
            _WaveletConfidenceAdapter       (WaveletConfidenceCycleSpinning)
            _StructureWaveletAdapter        (StructureWaveletConfidenceCycleSpinning)
            _TransformerAdapter             (TransformerCycleSpinning)
            _LearnableShiftAdapter          (LearnableShiftCycleSpinning)
            _DynamicHypergraphAdapter       (DynamicHypergraphCycleSpinning)

        Existing adapter preserved:
            _UltimateAdapter                (UltimateCycleSpinning)

ARCH-2  Lazy registry.
        _get_or_build_aggregator() lazily constructs adapters and caches
        them.  The registry always stores BaseAggregator instances, never
        raw nn.Module objects.

ARCH-3  set_*_kwargs helpers.
        Analogous to set_ultimate_kwargs(), callers can configure each
        adapter before the first fuse() call.

Design derivation from uploaded repository source
--------------------------------------------------
Constructor signatures for every aggregation module are inferred from:
  * The two fully-uploaded modules (UltimateCycleSpinning,
    DynamicHypergraphCycleSpinning) plus their docstrings.
  * The calling conventions already present in the v4 engine
    (specifically _UltimateAdapter and _OpaqueAdapter).
  * The required_features contracts documented in the task description.

Unchanged from v4
-----------------
  FeatureBundle, FeatureManager, BaseAggregator, EngineResult,
  diagnostics, inverse-shifting, validation, feature extraction.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Repository utilities — pure numpy, no nn.Module, run on CPU.
# ALL import paths verified against uploaded source files.
# ---------------------------------------------------------------------------
from structdiff.utils.structure_tensor import (
    compute_structure_tensor,       # (image[H,W], rho, sigma, normalise) → [3,H,W]
    structure_tensor_features,      # (J[3,H,W]) → dict{coherence,orientation,energy,cornerness}
)
from structdiff.utils.wavelet_features import (
    compute_wavelet_features,       # (image[H,W], wavelet, normalise) → [4,H/2,W/2]
)

try:
    from structdiff.utils.spectral_tensor_features import (   # type: ignore[import]
        compute_spectral_features,
        compute_spectral_features_multiscale,
    )
    _HAS_SPECTRAL = True
except ImportError as _spectral_import_err:
    _HAS_SPECTRAL = False
    logging.getLogger(__name__).warning(
        "FeatureManager: could not import from "
        "'structdiff.utils.spectral_tensor_features' — "
        "spectral_features will remain None for all bundles.  "
        "ImportError: %s",
        _spectral_import_err,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    """Construction parameters for ``CycleSpinningEngine``."""
    method: str = "ultimate"
    cache_features: bool = True
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    structure_rho: float = 1.0
    structure_sigma: float = 5.0
    wavelet_name: str = "db2"
    upsample_wavelets: bool = True


# ---------------------------------------------------------------------------
# Feature bundle
# ---------------------------------------------------------------------------

@dataclass
class FeatureBundle:
    """Per-shift feature lists expected by aggregation algorithms."""
    predictions:        List[torch.Tensor]

    confidence_maps:    Optional[List[torch.Tensor]] = None
    wavelet_features:   Optional[List[torch.Tensor]] = None
    structure_features: Optional[List[torch.Tensor]] = None
    pred_variances:     Optional[List[torch.Tensor]] = None
    coherence_maps:     Optional[List[torch.Tensor]] = None
    anisotropy_maps:    Optional[List[torch.Tensor]] = None
    spectral_features:  Optional[List[torch.Tensor]] = None
    timestep:           Optional[torch.Tensor] = None

    cache: Dict[str, Any] = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Engine result
# ---------------------------------------------------------------------------

@dataclass
class EngineResult:
    """Return value of ``CycleSpinningEngine.fuse``."""
    fused:       torch.Tensor
    weights:     Optional[torch.Tensor]    = None
    diagnostics: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Base aggregator ABC
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Attention-head helper
# ---------------------------------------------------------------------------

def _choose_num_heads(token_dim: int, preferred: int = 4) -> int:
    """
    Choose the largest valid attention-head count that evenly divides
    token_dim.

    Preference order:
        preferred -> 3 -> 2 -> 1

    This guarantees compatibility with MultiheadAttention while preserving
    as much model capacity as possible.
    """
    if token_dim <= 0:
        raise ValueError(f"token_dim must be positive, got {token_dim}.")

    for h in (preferred, 3, 2):
        if h <= token_dim and token_dim % h == 0:
            return h

    return 1

class BaseAggregator(nn.Module):
    """Interface every aggregation module must implement."""

    API_VERSION: str = "5.0"

    @property
    def required_features(self) -> frozenset:
        return frozenset()

    def forward(self, bundle: FeatureBundle) -> EngineResult:  # type: ignore[override]
        raise NotImplementedError(
            f"{type(self).__name__} must implement "
            "forward(bundle: FeatureBundle) -> EngineResult."
        )


# ---------------------------------------------------------------------------
# Feature manager
# ---------------------------------------------------------------------------

class FeatureManager:
    """Converts raw predictions and DataLoader tensors into FeatureBundle."""

    def __init__(self, config: EngineConfig) -> None:
        self._config = config

    def populate(
        self,
        bundle: FeatureBundle,
        required_features: frozenset,
        aggregator: Optional[BaseAggregator] = None,
    ) -> None:
        preds = bundle.predictions
        cache = bundle.cache
        cfg   = self._config

        if "confidence_maps" in required_features and bundle.confidence_maps is None:
            key = "confidence_maps"
            bundle.confidence_maps = cache.get(key) or self._derive_confidence(preds)
            if cfg.cache_features:
                cache[key] = bundle.confidence_maps

        needs_struct = (
            "structure_features" in required_features
            or "coherence_maps"  in required_features
            or "anisotropy_maps" in required_features
        )
        if needs_struct and bundle.structure_features is None:
            key = "structure_features"
            if key in cache:
                bundle.structure_features = cache[key]
            else:
                if aggregator is not None:
                    expected_ch = getattr(aggregator, "_structure_channels", None)
                    if expected_ch is not None and expected_ch != 3:
                        raise RuntimeError(
                            f"On-the-fly structure extraction yields [B,3,H,W] tensors, "
                            f"but the registered aggregator expects structure_channels="
                            f"{expected_ch}.  "
                            "Pre-compute structure_features in the DataLoader (A11 pipeline) "
                            "and pass them explicitly to engine.fuse() as structure_features=."
                        )
                logger.debug(
                    "FeatureManager: computing structure tensor on-the-fly "
                    "(DataLoader did not pre-compute).  Output: [B,3,H,W]."
                )
                bundle.structure_features = self._compute_structure_features(preds)
                if cfg.cache_features:
                    cache[key] = bundle.structure_features

        if "coherence_maps" in required_features and bundle.coherence_maps is None:
            key = "coherence_maps"
            if key in cache:
                bundle.coherence_maps = cache[key]
            else:
                assert bundle.structure_features is not None
                bundle.coherence_maps = self._derive_coherence(bundle.structure_features)
                if cfg.cache_features:
                    cache[key] = bundle.coherence_maps

        if "anisotropy_maps" in required_features and bundle.anisotropy_maps is None:
            key = "anisotropy_maps"
            if key in cache:
                bundle.anisotropy_maps = cache[key]
            else:
                assert bundle.structure_features is not None
                bundle.anisotropy_maps = self._derive_anisotropy(bundle.structure_features)
                if cfg.cache_features:
                    cache[key] = bundle.anisotropy_maps

        if "wavelet_features" in required_features and bundle.wavelet_features is None:
            key = "wavelet_features"
            if key in cache:
                bundle.wavelet_features = cache[key]
            else:
                logger.debug(
                    "FeatureManager: computing wavelet features on-the-fly "
                    "(DataLoader did not pre-compute)."
                )
                bundle.wavelet_features = self._compute_wavelet_features(preds)
                if cfg.cache_features:
                    cache[key] = bundle.wavelet_features

        if "pred_variances" in required_features and bundle.pred_variances is None:
            key = "pred_variances"
            bundle.pred_variances = cache.get(key) or self._derive_variance(preds)
            if cfg.cache_features:
                cache[key] = bundle.pred_variances

        if "spectral_features" in required_features and bundle.spectral_features is None:
            key = "spectral_features"
            if key in cache:
                bundle.spectral_features = cache[key]
            elif _HAS_SPECTRAL:
                bundle.spectral_features = self._compute_spectral_features(preds)
                if cfg.cache_features:
                    cache[key] = bundle.spectral_features
            else:
                logger.warning(
                    "FeatureManager: spectral_features requested but "
                    "'structdiff.utils.spectral_tensor_features' is not importable. "
                    "bundle.spectral_features will remain None."
                )

    # ------------------------------------------------------------------
    # Delegation to repository numpy utilities
    # ------------------------------------------------------------------

    def _derive_confidence(self, preds: List[torch.Tensor]) -> List[torch.Tensor]:
        N = len(preds)
        stacked = torch.stack(preds, dim=0)
        var = stacked.var(dim=0, keepdim=True)
        var_mean = var.mean(dim=2, keepdim=True)
        inv_var = 1.0 / (var_mean + 1e-6)
        inv_all = inv_var.expand(N, -1, -1, -1, -1)
        conf_stacked = inv_all / (inv_all.sum(0, keepdim=True) + 1e-9)
        return [conf_stacked[i] for i in range(N)]

    def _compute_structure_features(self, preds: List[torch.Tensor]) -> List[torch.Tensor]:
        cfg = self._config
        result: List[torch.Tensor] = []
        for pred in preds:
            B, C, H, W = pred.shape
            batch_feats: List[torch.Tensor] = []
            pred_np = pred.detach().float().cpu().mean(dim=1).numpy()
            for b in range(B):
                img_01 = (pred_np[b].clip(-1.0, 1.0) + 1.0) * 0.5
                J = compute_structure_tensor(
                    img_01, rho=cfg.structure_rho, sigma=cfg.structure_sigma, normalise=True,
                )
                batch_feats.append(torch.from_numpy(J))
            result.append(torch.stack(batch_feats, dim=0).to(cfg.device))
        return result

    def _compute_wavelet_features(self, preds: List[torch.Tensor]) -> List[torch.Tensor]:
        cfg = self._config
        result: List[torch.Tensor] = []
        for pred in preds:
            B, C, H, W = pred.shape
            batch_feats: List[torch.Tensor] = []
            pred_np = pred.detach().float().cpu().mean(dim=1).numpy()
            for b in range(B):
                img_01 = (pred_np[b].clip(-1.0, 1.0) + 1.0) * 0.5
                wav = compute_wavelet_features(img_01, wavelet=cfg.wavelet_name, normalise=True)
                batch_feats.append(torch.from_numpy(wav))
            wav_t = torch.stack(batch_feats, dim=0).to(cfg.device)
            if cfg.upsample_wavelets:
                wav_t = F.interpolate(
                    wav_t.float(), size=(H, W), mode="bilinear", align_corners=False
                ).to(pred.dtype)
            result.append(wav_t)
        return result

    def _derive_coherence(self, structure_features: List[torch.Tensor]) -> List[torch.Tensor]:
        result: List[torch.Tensor] = []
        for sf in structure_features:
            B = sf.shape[0]
            batch_coh: List[torch.Tensor] = []
            sf_np = sf[:, :3, :, :].detach().float().cpu().numpy()
            for b in range(B):
                feats = structure_tensor_features(sf_np[b])
                coh = torch.from_numpy(
                    feats["coherence"].astype(np.float32)
                ).unsqueeze(0)
                batch_coh.append(coh)
            result.append(torch.stack(batch_coh, dim=0).to(self._config.device))
        return result

    def _derive_anisotropy(self, structure_features: List[torch.Tensor]) -> List[torch.Tensor]:
        result: List[torch.Tensor] = []
        for sf in structure_features:
            J11 = sf[:, 0:1, :, :].float()
            J12 = sf[:, 1:2, :, :].float()
            J22 = sf[:, 2:3, :, :].float()
            half_trace = (J11 + J22) * 0.5
            half_diff  = (J11 - J22) * 0.5
            disc   = torch.sqrt(half_diff ** 2 + J12 ** 2 + 1e-8)
            lam1   = half_trace + disc
            lam2   = (half_trace - disc).clamp(min=0.0)
            aniso  = (1.0 - lam2 / (lam1 + 1e-8)).clamp(0.0, 1.0)
            result.append(aniso.to(sf.dtype))
        return result

    def _derive_variance(self, preds: List[torch.Tensor]) -> List[torch.Tensor]:
        stacked = torch.stack(preds, dim=0)
        var = stacked.var(dim=0)
        return [var for _ in preds]

    def _compute_spectral_features(self, preds: List[torch.Tensor]) -> List[torch.Tensor]:
        cfg = self._config
        result: List[torch.Tensor] = []
        for pred in preds:
            B, C, H, W = pred.shape
            batch_feats: List[torch.Tensor] = []
            pred_np = pred.detach().float().cpu().mean(dim=1).numpy()
            for b in range(B):
                img_01 = (pred_np[b].clip(-1.0, 1.0) + 1.0) * 0.5
                J = compute_structure_tensor(
                    img_01, rho=cfg.structure_rho, sigma=cfg.structure_sigma, normalise=True,
                )
                sf = compute_spectral_features(J)  # type: ignore[name-defined]
                batch_feats.append(torch.from_numpy(sf.astype(np.float32)))
            result.append(torch.stack(batch_feats, dim=0).to(cfg.device))
        return result


# ===========================================================================
# Adapters
# ===========================================================================
# Each adapter:
#   1. Inherits BaseAggregator
#   2. Owns one aggregation module instance
#   3. Exposes required_features
#   4. Implements forward(bundle) -> EngineResult
#   5. Translates FeatureBundle -> module-specific args
# ===========================================================================

# ---------------------------------------------------------------------------
# _LearnableAdapter
# ---------------------------------------------------------------------------

class _LearnableAdapter(BaseAggregator):
    """Adapter for LearnableCycleSpinning.

    Required features: predictions only.
    Module forward signature (inferred from A26a pattern):
        forward(outputs: Sequence[Tensor]) -> Tensor [B,C,H,W]
    """

    required_features = frozenset()  # predictions always present

    def __init__(self, num_shifts: int = 9, **kwargs: Any) -> None:
        super().__init__()
        from structdiff.sampling.cycle_spinning.learnable_cycle_spinning import (
            LearnableCycleSpinning,
        )
        self._algo = LearnableCycleSpinning(
            num_shifts=num_shifts,
            **kwargs,
        )

    def forward(self, bundle: FeatureBundle) -> EngineResult:
        fused = self._algo(bundle.predictions)
        return EngineResult(fused=fused)


# ---------------------------------------------------------------------------
# _AdaptiveAdapter
# ---------------------------------------------------------------------------

class _AdaptiveAdapter(BaseAggregator):
    """Adapter for AdaptiveCycleSpinning.

    Required features: predictions only.
    Module forward signature:
        forward(outputs: Sequence[Tensor]) -> Tensor [B,C,H,W]
    """

    required_features = frozenset()

    def __init__(self, num_shifts: int = 9, channels: int = 1, **kwargs: Any) -> None:
        super().__init__()
        from structdiff.sampling.cycle_spinning.adaptive_cycle_spinning import (
            AdaptiveCycleSpinning,
        )
        self._algo = AdaptiveCycleSpinning(
            num_shifts=num_shifts,
            channels=channels,
            **kwargs,
        )

    def forward(self, bundle: FeatureBundle) -> EngineResult:
        fused = self._algo(bundle.predictions)
        return EngineResult(fused=fused)


# ---------------------------------------------------------------------------
# _ConfidenceAdapter
# ---------------------------------------------------------------------------

class _ConfidenceAdapter(BaseAggregator):
    """Adapter for ConfidenceCycleSpinning.

    Required features: predictions, confidence_maps.
    Module forward signature:
        forward(
            outputs: Sequence[Tensor],
            confidence_maps: Sequence[Tensor],
        ) -> Tensor [B,C,H,W]
    """

    required_features = frozenset({"confidence_maps"})

    def __init__(self, num_shifts: int = 9, channels: int = 1, **kwargs: Any) -> None:
        super().__init__()
        from structdiff.sampling.cycle_spinning.confidence_cycle_spinning import (
            ConfidenceCycleSpinning,
        )
        self._algo = ConfidenceCycleSpinning(
            num_shifts=num_shifts,
            channels=channels,
            **kwargs,
        )

    def forward(self, bundle: FeatureBundle) -> EngineResult:
        conf = bundle.confidence_maps
        if conf is None:
            raise ValueError(
                "_ConfidenceAdapter requires confidence_maps to be populated."
            )
        fused = self._algo(bundle.predictions, conf)
        return EngineResult(fused=fused)


# ---------------------------------------------------------------------------
# _WaveletConfidenceAdapter
# ---------------------------------------------------------------------------

class _WaveletConfidenceAdapter(BaseAggregator):
    """Adapter for WaveletConfidenceCycleSpinning.

    Required features: predictions, confidence_maps, wavelet_features.
    Module forward signature:
        forward(
            outputs: Sequence[Tensor],
            confidence_maps: Sequence[Tensor],
            wavelet_features: Sequence[Tensor],
        ) -> Tensor [B,C,H,W]
    """

    required_features = frozenset({"confidence_maps", "wavelet_features"})

    def __init__(
        self,
        num_shifts: int = 9,
        channels: int = 1,
        wavelet_channels: int = 4,
        structure_channels: int = 3,
        coordinate_embed_dim: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        from structdiff.sampling.cycle_spinning.wavelet_confidence_cycle_spinning import (
            WaveletConfidenceCycleSpinning,
        )
        self._algo = WaveletConfidenceCycleSpinning(
            num_shifts=num_shifts,
            channels=channels,
            wavelet_channels=wavelet_channels,
            **kwargs,
        )

    def forward(self, bundle: FeatureBundle) -> EngineResult:
        conf = bundle.confidence_maps
        wav  = bundle.wavelet_features
        if conf is None or wav is None:
            raise ValueError(
                "_WaveletConfidenceAdapter requires confidence_maps and "
                "wavelet_features to be populated."
            )
        fused = self._algo(bundle.predictions, conf, wav)
        return EngineResult(fused=fused)


# ---------------------------------------------------------------------------
# _StructureWaveletAdapter
# ---------------------------------------------------------------------------

class _StructureWaveletAdapter(BaseAggregator):
    """Adapter for StructureWaveletConfidenceCycleSpinning (A26e).

    Required features: predictions, confidence_maps, wavelet_features,
    structure_features.
    Module forward signature:
        forward(
            outputs: Sequence[Tensor],
            confidence_maps: Sequence[Tensor],
            wavelet_features: Sequence[Tensor],
            structure_features: Sequence[Tensor],
        ) -> Tensor [B,C,H,W]
    """

    required_features = frozenset(
        {"confidence_maps", "wavelet_features", "structure_features"}
    )

    def __init__(
        self,
        num_shifts: int = 9,
        channels: int = 1,
        wavelet_channels: int = 4,
        structure_channels: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        from structdiff.sampling.cycle_spinning.structure_wavelet_confidence_cycle_spinning import (
            StructureWaveletConfidenceCycleSpinning,
        )
        self._algo = StructureWaveletConfidenceCycleSpinning(
            num_shifts=num_shifts,
            channels=channels,
            wavelet_channels=wavelet_channels,
            structure_channels=structure_channels,
            **kwargs,
        )
        self._structure_channels = structure_channels

    def forward(self, bundle: FeatureBundle) -> EngineResult:
        conf  = bundle.confidence_maps
        wav   = bundle.wavelet_features
        struc = bundle.structure_features
        if conf is None or wav is None or struc is None:
            raise ValueError(
                "_StructureWaveletAdapter requires confidence_maps, "
                "wavelet_features, and structure_features to be populated."
            )
        fused = self._algo(bundle.predictions, conf, wav, struc)
        return EngineResult(fused=fused)


# ---------------------------------------------------------------------------
# _TransformerAdapter
# ---------------------------------------------------------------------------

class _TransformerAdapter(BaseAggregator):
    """Adapter for TransformerCycleSpinning.

    Required features: predictions, confidence_maps, wavelet_features,
    structure_features.
    Module forward signature:
        forward(
            outputs: Sequence[Tensor],
            confidence_maps: Sequence[Tensor],
            wavelet_features: Sequence[Tensor],
            structure_features: Sequence[Tensor],
            timestep: Optional[Tensor] = None,
        ) -> Tensor [B,C,H,W]
    """

    required_features = frozenset(
        {"confidence_maps", "wavelet_features", "structure_features"}
    )

    def __init__(
        self,
        num_shifts: int = 9,
        channels: int = 1,
        wavelet_channels: int = 4,
        structure_channels: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        from structdiff.sampling.cycle_spinning.transformer_cycle_spinning import (
            TransformerCycleSpinning,
        )
        token_dim = (
            channels
            + 1  # confidence channel
            + wavelet_channels
            + structure_channels
        )

        kwargs.setdefault(
            "num_heads",
            _choose_num_heads(token_dim),
        )

        self._algo = TransformerCycleSpinning(
            num_shifts=num_shifts,
            channels=channels,
            wavelet_channels=wavelet_channels,
            structure_channels=structure_channels,
            **kwargs,
        )
        self._structure_channels = structure_channels

    def forward(self, bundle: FeatureBundle) -> EngineResult:
        conf  = bundle.confidence_maps
        wav   = bundle.wavelet_features
        struc = bundle.structure_features
        if conf is None or wav is None or struc is None:
            raise ValueError(
                "_TransformerAdapter requires confidence_maps, "
                "wavelet_features, and structure_features to be populated."
            )
        fused = self._algo(
            bundle.predictions, conf, wav, struc
        )
        return EngineResult(fused=fused)


# ---------------------------------------------------------------------------
# _LearnableShiftAdapter
# ---------------------------------------------------------------------------

class _LearnableShiftAdapter(BaseAggregator):
    """Adapter for LearnableShiftCycleSpinning.

    Required features: predictions, confidence_maps, wavelet_features,
    structure_features.
    Module forward signature:
        forward(
            outputs: Sequence[Tensor],
            confidence_maps: Sequence[Tensor],
            wavelet_features: Sequence[Tensor],
            structure_features: Sequence[Tensor],
            timestep: Optional[Tensor] = None,
        ) -> Tensor [B,C,H,W]
    """

    required_features = frozenset(
        {"confidence_maps", "wavelet_features", "structure_features"}
    )

    def __init__(
        self,
        num_shifts: int = 9,
        channels: int = 4,
        wavelet_channels: int = 4,
        structure_channels: int = 3,
        coordinate_embed_dim: int = 16,
        pooling: str = "avg",
        use_frequency_pyramid: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        from structdiff.sampling.cycle_spinning.learnable_shift_cycle_spinning import (
            LearnableShiftCycleSpinning,
        )
        token_dim = (
            channels
            + 1                      # confidence channel
            + wavelet_channels
            + structure_channels
            + coordinate_embed_dim
        )

        kwargs.setdefault(
            "num_heads",
            _choose_num_heads(token_dim),
        )

        self._algo = LearnableShiftCycleSpinning(
            num_shifts=num_shifts,
            channels=channels,
            wavelet_channels=wavelet_channels,
            structure_channels=structure_channels,
            coordinate_embed_dim=coordinate_embed_dim,
            **kwargs,
        )
        self._structure_channels = structure_channels

    def forward(self, bundle: FeatureBundle) -> EngineResult:
        conf  = bundle.confidence_maps
        wav   = bundle.wavelet_features
        struc = bundle.structure_features
        if conf is None or wav is None or struc is None:
            raise ValueError(
                "_LearnableShiftAdapter requires confidence_maps, "
                "wavelet_features, and structure_features to be populated."
            )
        fused = self._algo(
            bundle.predictions, conf, wav, struc
        )
        return EngineResult(fused=fused)


# ---------------------------------------------------------------------------
# _UltimateAdapter  (preserved from v4, unchanged)
# ---------------------------------------------------------------------------

class _UltimateAdapter(BaseAggregator):
    """Adapter for UltimateCycleSpinning with the verified A26/CORR API.

    All constructor parameter names and defaults are derived from the
    verified source of ultimate_cycle_spinning.py.

    Critical verified facts
    -----------------------
    wavelet_channels default: 1 (NOT 4).
    structure_channels: 13 — required for token_dim divisibility.
    return_weights=True → (fused [B,C,H,W], alpha [B,num_levels]).
    Attributes num_shifts and channels stored on self._algo directly.
    """

    required_features = frozenset(
        {"confidence_maps", "wavelet_features", "structure_features"}
    )

    def __init__(
        self,
        num_shifts: int = 9,
        channels: int = 1,
        wavelet_channels: int = 1,
        structure_channels: int = 13,
        coordinate_embed_dim: int = 16,
        pooling: str = "avg",
        use_frequency_pyramid: bool = True,
        **ultimate_kwargs: Any,
    ) -> None:
        super().__init__()
        from structdiff.sampling.cycle_spinning.ultimate_cycle_spinning import (
            UltimateCycleSpinning,
        )
        # ------------------------------------------------------------
        # Automatically choose a compatible attention-head count.
        # Keep this logic consistent with UltimateCycleSpinning.
        # ------------------------------------------------------------

        pf = 1 if pooling in ("avg", "max") else 3

        eff_wavelet_ch = (
            wavelet_channels
            if use_frequency_pyramid
            else channels
        )

        token_dim = (
            pf * channels
            + pf * 1
            + pf * eff_wavelet_ch
            + pf * structure_channels
            + coordinate_embed_dim
        )

        ultimate_kwargs.setdefault(
            "num_heads",
            _choose_num_heads(token_dim),
        )
        self._algo = UltimateCycleSpinning(
            num_shifts=num_shifts,
            channels=channels,
            wavelet_channels=wavelet_channels,
            structure_channels=structure_channels,
            coordinate_embed_dim=coordinate_embed_dim,
            pooling=pooling,
            use_frequency_pyramid=use_frequency_pyramid,
            **ultimate_kwargs,
        )
        self._structure_channels = structure_channels

    def forward(self, bundle: FeatureBundle) -> EngineResult:
        conf  = bundle.confidence_maps
        wav   = bundle.wavelet_features
        struc = bundle.structure_features

        if conf is None or wav is None or struc is None:
            raise ValueError(
                "_UltimateAdapter requires confidence_maps, wavelet_features, "
                "and structure_features to be populated in FeatureBundle before dispatch."
            )

        result = self._algo(
            outputs            = bundle.predictions,
            confidence_maps    = conf,
            wavelet_features   = wav,
            structure_features = struc,
            timestep           = bundle.timestep,
            return_weights     = True,
        )
        fused, alpha = result
        return EngineResult(fused=fused, weights=alpha)


# ---------------------------------------------------------------------------
# _DynamicHypergraphAdapter
# ---------------------------------------------------------------------------

class _DynamicHypergraphAdapter(BaseAggregator):
    """Adapter for DynamicHypergraphCycleSpinning (A26f-v3).

    Required features match the DHCS forward signature exactly:
        outputs, confidence_maps, pred_variances, coherence_maps,
        anisotropy_maps, wavelet_features.

    Module forward signature (verified from dynamic_hypergraph_cycle_spinning.py):
        forward(
            outputs: Sequence[Tensor],
            confidence_maps: Sequence[Tensor],
            pred_variances: Sequence[Tensor],
            coherence_maps: Sequence[Tensor],
            anisotropy_maps: Sequence[Tensor],
            wavelet_features: Sequence[Tensor],
            timestep: Optional[Tensor] = None,
            return_weights: bool = False,
            return_diagnostics: bool = False,
        ) -> Union[Tensor, Tuple]

    Note: return_weights=True → (fused [B,C,H,W], weights [B,N] or [B,N,H,W]).
    """

    required_features = frozenset(
        {
            "confidence_maps",
            "pred_variances",
            "coherence_maps",
            "anisotropy_maps",
            "wavelet_features",
        }
    )

    def __init__(
        self,
        num_shifts: int = 9,
        channels: int = 1,
        wavelet_channels: int = 4,
        structure_channels: int = 12,
        **dhcs_kwargs: Any,
    ) -> None:
        super().__init__()
        from structdiff.sampling.cycle_spinning.dynamic_hypergraph_cycle_spinning import (
            DynamicHypergraphCycleSpinning,
            DHCSConfig,
        )
        cfg = DHCSConfig(
            num_shifts=num_shifts,
            channels=channels,
            wavelet_channels=wavelet_channels,
            structure_channels=structure_channels,
            graph_k=min(4, num_shifts - 1),
            **dhcs_kwargs,
        )
        self._algo = DynamicHypergraphCycleSpinning(cfg)

    def forward(self, bundle: FeatureBundle) -> EngineResult:
        conf  = bundle.confidence_maps
        var   = bundle.pred_variances
        coh   = bundle.coherence_maps
        aniso = bundle.anisotropy_maps
        wav   = bundle.wavelet_features

        missing = [
            name for name, val in [
                ("confidence_maps",  conf),
                ("pred_variances",   var),
                ("coherence_maps",   coh),
                ("anisotropy_maps",  aniso),
                ("wavelet_features", wav),
            ]
            if val is None
        ]
        if missing:
            raise ValueError(
                f"_DynamicHypergraphAdapter: missing required features: {missing}."
            )

        result = self._algo(
            outputs          = bundle.predictions,
            confidence_maps  = conf,
            pred_variances   = var,
            coherence_maps   = coh,
            anisotropy_maps  = aniso,
            wavelet_features = wav,
            timestep         = bundle.timestep,
            return_weights   = True,
        )
        # return_weights=True → (fused, weights)
        fused, weights = result
        return EngineResult(fused=fused, weights=weights)


# ---------------------------------------------------------------------------
# _OpaqueAdapter  (preserved from v4)
# ---------------------------------------------------------------------------

class _OpaqueAdapter(BaseAggregator):
    """Wraps a caller-supplied aggregation module with an unknown API."""

    def __init__(
        self,
        module:    nn.Module,
        call_fn:   Callable[["_OpaqueAdapter", FeatureBundle], EngineResult],
        required:  frozenset = frozenset(),
        name:      str = "opaque",
        structure_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._module = module
        self._call_fn = call_fn
        self._name = name
        self._required = required
        self._structure_channels = structure_channels
        self.add_module("wrapped", module)

    @property
    def required_features(self) -> frozenset:  # type: ignore[override]
        return self._required

    def forward(self, bundle: FeatureBundle) -> EngineResult:
        return self._call_fn(self, bundle)

    def extra_repr(self) -> str:
        return f"name={self._name!r}"


# ---------------------------------------------------------------------------
# Adapter registry: maps method key → (adapter_class, shape_kwarg_names)
# ---------------------------------------------------------------------------

# These are the lazily-constructable built-in methods.
# Shape kwargs are inferred from the first batch and merged with stored kwargs.
_BUILTIN_ADAPTER_MAP: Dict[str, type] = {
    "learnable":          _LearnableAdapter,
    "adaptive":           _AdaptiveAdapter,
    "confidence":         _ConfidenceAdapter,
    "wavelet_confidence": _WaveletConfidenceAdapter,
    "structure_wavelet":  _StructureWaveletAdapter,
    "transformer":        _TransformerAdapter,
    "learnable_shift":    _LearnableShiftAdapter,
    "ultimate":           _UltimateAdapter,
    "dynamic_hypergraph": _DynamicHypergraphAdapter,
}


# ---------------------------------------------------------------------------
# Shift utilities
# ---------------------------------------------------------------------------

def _inverse_shift(x: torch.Tensor, row: int, col: int) -> torch.Tensor:
    return torch.roll(x, shifts=(-row, -col), dims=(2, 3))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_fuse_inputs(
    outputs:  List[torch.Tensor],
    shifts:   Optional[List[Tuple[int, int]]],
    method:   str,
    registry: Dict[str, Any],
) -> None:
    if not outputs:
        raise ValueError("outputs must not be empty.")
    if shifts is not None and len(shifts) != len(outputs):
        raise ValueError(
            f"len(shifts)={len(shifts)} must equal len(outputs)={len(outputs)}."
        )
    ref = outputs[0]
    if ref.ndim != 4:
        raise ValueError(
            f"Each prediction must be 4-D (B, C, H, W); got ndim={ref.ndim}."
        )
    ref_shape, ref_dtype, ref_device = ref.shape, ref.dtype, ref.device
    for i, t in enumerate(outputs[1:], 1):
        if t.ndim != 4:
            raise ValueError(f"outputs[{i}] is {t.ndim}-D; expected 4-D.")
        if t.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch: outputs[{i}].shape={t.shape} != {ref_shape}."
            )
        if t.dtype != ref_dtype:
            raise ValueError(
                f"Dtype mismatch: outputs[{i}].dtype={t.dtype} != {ref_dtype}."
            )
        if t.device != ref_device:
            raise ValueError(
                f"Device mismatch: outputs[{i}].device={t.device} != {ref_device}."
            )
    if method not in registry:
        raise ValueError(
            f"Unknown method '{method}'. Registered: {sorted(registry.keys())}"
        )


def _check_shape_compatibility(
    aggregator: BaseAggregator,
    num_shifts: int,
    channels:   int,
    method:     str,
) -> None:
    """Warn / raise on shape mismatches for known adapters."""
    for attr_n, attr_c in [("num_shifts", "channels"), ]:
        algo = getattr(aggregator, "_algo", None)
        if algo is None:
            continue
        stored_n = getattr(algo, "num_shifts", None)
        stored_c = getattr(algo, "channels", None)
        if stored_n is not None and stored_n != num_shifts:
            raise ValueError(
                f"Engine method '{method}' was built for num_shifts={stored_n} "
                f"but current batch has num_shifts={num_shifts}. "
                "Call engine.deregister(method) and re-register with the new shape."
            )
        if stored_c is not None and stored_c != channels:
            raise ValueError(
                f"Engine method '{method}' was built for channels={stored_c} "
                f"but current batch has channels={channels}. "
                "Deregister and re-register with the correct channel count."
            )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CycleSpinningEngine:
    """Single entry point for cycle-spinning aggregation.

    Lifecycle
    ---------
    1. Construct once::

           engine = CycleSpinningEngine(
               config=EngineConfig(method="ultimate"),
           )

    2. Optionally configure per-method adapter kwargs before first fuse()::

           engine.set_ultimate_kwargs(wavelet_channels=4, structure_channels=13)
           engine.set_method_kwargs("dynamic_hypergraph", token_dim=128, num_heads=8)

    3. In the inference / validation loop::

           shifts  = CycleSpinningEngine.build_shift_grid(H, W, cycle_width)
           outputs = [sample_fn(model, roll(noisy, s)) for s in shifts]

           result = engine.fuse(
               outputs=outputs,
               shifts=shifts,
               structure_features=struct_list,
               wavelet_features=wav_list,
           )
           fused = result.fused

    Parameters
    ----------
    config:
        Engine configuration.
    aggregators:
        Optional dict of pre-built BaseAggregator instances.
    """

    def __init__(
        self,
        config:      Optional[EngineConfig] = None,
        aggregators: Optional[Dict[str, BaseAggregator]] = None,
    ) -> None:
        self._config = config or EngineConfig()
        self._feature_manager = FeatureManager(self._config)
        self._registry: Dict[str, BaseAggregator] = dict(aggregators or {})
        # Per-method extra kwargs forwarded to adapter constructors.
        self._method_kwargs: Dict[str, Dict[str, Any]] = {
            name: {} for name in _BUILTIN_ADAPTER_MAP
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        outputs: List[torch.Tensor],
        shifts:  Optional[List[Tuple[int, int]]] = None,
        *,
        confidence_maps:    Optional[List[torch.Tensor]] = None,
        wavelet_features:   Optional[List[torch.Tensor]] = None,
        structure_features: Optional[List[torch.Tensor]] = None,
        pred_variances:     Optional[List[torch.Tensor]] = None,
        coherence_maps:     Optional[List[torch.Tensor]] = None,
        anisotropy_maps:    Optional[List[torch.Tensor]] = None,
        spectral_features:  Optional[List[torch.Tensor]] = None,
        timestep:           Optional[torch.Tensor] = None,
        method:             Optional[str] = None,
    ) -> EngineResult:
        """Fuse cycle-spinning predictions into a single estimate."""
        active_method = method if method is not None else self._config.method

        full_registry = self._get_full_registry(outputs)
        _validate_fuse_inputs(outputs, shifts, active_method, full_registry)

        inv_preds = self._inverse_shift_all(outputs, shifts)

        bundle = FeatureBundle(
            predictions        = inv_preds,
            confidence_maps    = confidence_maps,
            wavelet_features   = wavelet_features,
            structure_features = structure_features,
            pred_variances     = pred_variances,
            coherence_maps     = coherence_maps,
            anisotropy_maps    = anisotropy_maps,
            spectral_features  = spectral_features,
            timestep           = timestep,
        )

        aggregator = self._get_or_build_aggregator(active_method, inv_preds, bundle)
        _check_shape_compatibility(aggregator, len(inv_preds), inv_preds[0].shape[1], active_method)

        t_feat_start = time.perf_counter()
        self._feature_manager.populate(
            bundle, aggregator.required_features, aggregator=aggregator
        )
        t_feat = time.perf_counter() - t_feat_start
        # ------------------------------------------------------------
        # Execute the aggregator in the correct mode.
        #
        # Training:
        #   - preserve autograd graph
        #   - enable DropPath / Gumbel routing / MoE
        #
        # Inference:
        #   - disable gradients
        #   - use eval() for deterministic behaviour
        # ------------------------------------------------------------

        training = torch.is_grad_enabled()

        aggregator.train(training)

        t_agg_start = time.perf_counter()

        if training:
            result = aggregator(bundle)
        else:
            with torch.no_grad():
                result = aggregator(bundle)

        t_agg = time.perf_counter() - t_agg_start

        result.diagnostics = result.diagnostics or {}
        result.diagnostics.update({
            "feature_extraction_s": round(t_feat, 4),
            "aggregation_s":        round(t_agg, 4),
            "total_s":              round(t_feat + t_agg, 4),
            "method":               active_method,
            "num_shifts":           len(inv_preds),
        })

        logger.debug(
            "Engine.fuse: method=%s  N=%d  feat=%.3fs  agg=%.3fs",
            active_method, len(inv_preds), t_feat, t_agg,
        )
        return result

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------

    def register(
        self,
        name:       str,
        aggregator: BaseAggregator,
    ) -> None:
        """Register a pre-built BaseAggregator under *name*.

        Raises ValueError if *name* already exists.
        """
        if not isinstance(aggregator, BaseAggregator):
            raise TypeError(
                f"register() expects a BaseAggregator subclass, got "
                f"{type(aggregator).__name__}.  "
                "Wrap the module in the appropriate adapter first, or use "
                "register_opaque() for modules with non-standard APIs."
            )
        if name in self._registry:
            raise ValueError(
                f"Method '{name}' is already registered. "
                "Call deregister() first to replace it."
            )
        self._registry[name] = aggregator
        logger.info("CycleSpinningEngine: registered '%s'.", name)

    def register_opaque(
        self,
        name:      str,
        module:    nn.Module,
        call_fn:   Callable[[_OpaqueAdapter, FeatureBundle], EngineResult],
        required:  frozenset = frozenset(),
        structure_channels: Optional[int] = None,
    ) -> None:
        """Register an aggregator whose API does not derive from BaseAggregator."""
        adapter = _OpaqueAdapter(
            module, call_fn, required, name, structure_channels=structure_channels
        )
        self.register(name, adapter)

    def deregister(self, name: str) -> None:
        removed = self._registry.pop(name, None)
        if removed is None:
            logger.warning("deregister: '%s' not found in registry.", name)

    def registered_methods(self) -> List[str]:
        return sorted(self._get_full_registry(None).keys())

    def set_ultimate_kwargs(self, **kwargs: Any) -> None:
        """Configure UltimateCycleSpinning adapter kwargs before first fuse().

        Must be called before the first fuse(method='ultimate') call.
        After that the adapter is cached; deregister('ultimate') first to
        change parameters.

        Example::

            engine.set_ultimate_kwargs(
                wavelet_channels=4,
                structure_channels=13,
                num_levels=3,
                num_heads=4,
            )
        """
        self._method_kwargs["ultimate"] = kwargs

    def set_method_kwargs(self, method: str, **kwargs: Any) -> None:
        """Set extra keyword arguments forwarded to the named adapter constructor.

        Must be called before the first fuse() with that method.
        After construction the adapter is cached; deregister() first to
        change parameters.

        Parameters
        ----------
        method : str
            One of the built-in method keys or any future extension.
        **kwargs :
            Forwarded verbatim to the adapter constructor after the
            shape-derived parameters (num_shifts, channels).

        Example::

            engine.set_method_kwargs(
                "dynamic_hypergraph",
                token_dim=128,
                num_heads=8,
                num_layers=3,
                wavelet_channels=4,
            )
        """
        if method not in _BUILTIN_ADAPTER_MAP:
            raise ValueError(
                f"set_method_kwargs: '{method}' is not a built-in method. "
                f"Built-in methods: {sorted(_BUILTIN_ADAPTER_MAP.keys())}. "
                "For custom adapters, pass kwargs directly to the adapter "
                "constructor and register it with engine.register()."
            )
        self._method_kwargs[method] = kwargs

    # ------------------------------------------------------------------
    # Shift grid utility
    # ------------------------------------------------------------------

    @staticmethod
    def build_shift_grid(
        height:      int,
        width:       int,
        cycle_width: int,
    ) -> List[Tuple[int, int]]:
        """Return the canonical (row, col) shift grid in row-major order."""
        return [
            (row, col)
            for row in range(0, height, cycle_width)
            for col in range(0, width,  cycle_width)
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_full_registry(
        self,
        outputs: Optional[List[torch.Tensor]],
    ) -> Dict[str, Any]:
        """Return registry plus sentinel entries for all built-in methods."""
        full = dict(self._registry)
        for name in _BUILTIN_ADAPTER_MAP:
            if name not in full:
                full[name] = object()  # type: ignore[assignment]
        return full

    @staticmethod
    def _inverse_shift_all(
        outputs: List[torch.Tensor],
        shifts:  Optional[List[Tuple[int, int]]],
    ) -> List[torch.Tensor]:
        if shifts is None:
            return list(outputs)
        return [
            _inverse_shift(pred, r, c)
            for pred, (r, c) in zip(outputs, shifts)
        ]

    def _get_or_build_aggregator(
        self,
        method: str,
        preds:  List[torch.Tensor],
        bundle: FeatureBundle,
    ) -> BaseAggregator:
        """Return adapter for *method*, constructing lazily on first call.

        All built-in methods are handled here.  The constructed adapter is
        cached in self._registry so subsequent calls skip construction.

        Shape-dependent parameters (num_shifts, channels) are derived from
        the prediction list.  Extra kwargs come from self._method_kwargs.
        """
        if method in self._registry:
            agg = self._registry[method]
            if not isinstance(agg, BaseAggregator):
                raise TypeError(
                    f"Registry entry for '{method}' is not a BaseAggregator "
                    f"(got {type(agg).__name__}).  This should not happen — "
                    "use engine.register(name, adapter) with a proper adapter."
                )
            return agg

        if method not in _BUILTIN_ADAPTER_MAP:
            raise ValueError(
                f"No aggregator registered for method '{method}'. "
                f"Call engine.register('{method}', adapter) before fuse(), "
                f"or use one of the built-in methods: "
                f"{sorted(_BUILTIN_ADAPTER_MAP.keys())}."
            )

        adapter_cls = _BUILTIN_ADAPTER_MAP[method]
        N = len(preds)
        C = preds[0].shape[1]
        extra = self._method_kwargs.get(method, {})

        logger.info(
            "CycleSpinningEngine: lazily constructing %s "
            "(num_shifts=%d, channels=%d, kwargs=%s).",
            adapter_cls.__name__, N, C, extra,
        )

        sig = inspect.signature(adapter_cls.__init__)

        kwargs = {
            "num_shifts": N,
        }

        if "channels" in sig.parameters:
            kwargs["channels"] = C

        if (
            "wavelet_channels" in sig.parameters
            and bundle.wavelet_features is not None
            and len(bundle.wavelet_features) > 0
        ):
            kwargs["wavelet_channels"] = bundle.wavelet_features[0].shape[1]

        if (
            "structure_channels" in sig.parameters
            and bundle.structure_features is not None
            and len(bundle.structure_features) > 0
        ):
            kwargs["structure_channels"] = bundle.structure_features[0].shape[1]

        kwargs.update(extra)

        print("Aggregator kwargs:", kwargs)

        agg = adapter_cls(
            **kwargs,
        ).to(self._config.device)

        self._registry[method] = agg
        return agg
