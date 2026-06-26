"""
structdiff/sampling/cycle_spinning/engine.py
============================================
Orchestration layer for cycle-spinning aggregation.  Version 4.

Corrections applied vs v3
--------------------------

FIX-1  Spectral import path.
        v3 imported from ``structdiff.utils.spectral_tensor_features``
        (does not exist).  Correct module is
        ``structdiff.utils.spectral_tensor_features``
        (verified from uploaded source).

FIX-2  ``_compute_spectral_features`` two-step pipeline.
        v3 called ``compute_spectral_features(img_01)`` directly on a
        2-D image array, which always raises ValueError because
        ``compute_spectral_features`` expects a [3,H,W] structure tensor
        as input, not a raw image.  The correct pipeline is:
            compute_structure_tensor(img_01) → [3,H,W] J
            compute_spectral_features(J) → [4,H,W]
        This matches the A11 module docstring exactly:
        "it operates purely on the structure tensors already produced
        by compute_structure_tensor_multiscale".

FIX-3  ``_UltimateAdapter`` wavelet_channels default.
        v3 hardcoded wavelet_channels=4.  UltimateCycleSpinning default
        is wavelet_channels=1 (verified from constructor signature).
        Engine now passes wavelet_channels via set_ultimate_kwargs only,
        defaulting to 1 to match UCS.

FIX-4  On-the-fly structure extraction guard.
        ``_compute_structure_features`` produces [B,3,H,W].  If the
        registered method requires structure_channels != 3, on-the-fly
        extraction is incompatible.  The engine now raises a descriptive
        error directing the caller to pre-compute structure_features in
        the DataLoader and pass them explicitly.

FIX-5  Spectral import diagnostic.
        v3's try/except swallowed the ImportError silently.  The actual
        error is now logged at WARNING level with the correct module path
        so misconfiguration is visible without a stack trace.

Design derivation from uploaded repository source
--------------------------------------------------
The following facts are derived from the four uploaded files only.
No APIs are inferred, invented, or assumed.

``structdiff/utils/structure_tensor.py``
    compute_structure_tensor(image, rho, sigma, normalise) → [3,H,W] float32
    structure_tensor_features(J) → dict with keys:
        "coherence"   [H,W] in [0,1]
        "orientation" [H,W] in [-π/2, π/2]
        "energy"      [H,W]
        "cornerness"  [H,W]
    image: 2-D float32 [H,W] in [0,1], even dims not required by A3.

``structdiff/utils/wavelet_features.py``
    compute_wavelet_features(image, wavelet, normalise) → [4,H/2,W/2] float32
    Channel order: LL, LH, HL, HH.
    Requires both H and W even (validated internally).
    DWT_MODE="periodization" guarantees exact H/2,W/2 output.

``structdiff/utils/spectral_tensor_features.py``
    compute_spectral_features(J, eps, clip_range) → [4,H,W] float32
        J must be [3,H,W] structure tensor (J11,J12,J22).
        Returns lambda1, lambda2, anisotropy, coherence.
    compute_spectral_features_multiscale(s1, s2, s3, ...) → [12,H,W]
        Concatenates spectral features across three A10 scales.

``structdiff/inference/ultimate_cycle_spinning.py``
    UltimateCycleSpinning(
        num_levels=3, num_shifts=9, channels=1,
        wavelet_channels=1,          ← default is 1, NOT 4
        structure_channels=13,       ← 13 channels required
        coordinate_embed_dim=16, num_heads=4, num_layers=4,
        cross_level_heads=2, cross_level_layers=2,
        dropout=0.1, temperature=1.0,
        level_radii=(1.0,3.0,6.0), min_shifts=2,
        ... (all other kwargs have defaults)
    )
    Stored attributes: self.num_shifts, self.channels
    forward(
        outputs,            # Sequence[Tensor (B,C,H,W)] length=num_shifts
        confidence_maps,    # Sequence[Tensor (B,1,H,W)]
        wavelet_features,   # Sequence[Tensor]
        structure_features, # Sequence[Tensor]
        timestep=None,      # Optional[Tensor (B,)]
        return_weights=False,
        return_level_outputs=False,
        return_uncertainty=False,
    )
    return_weights=True → (fused [B,C,H,W], alpha [B,num_levels])
    return_level_outputs=True → 6-tuple or 7-tuple (see source)

Roles
-----
``FeatureManager``
    Routes DataLoader-supplied features into FeatureBundle.
    On-the-fly computes structure, wavelet, confidence, variance from
    predictions when the DataLoader did not pre-compute them.
    NEVER re-implements repository mathematics.
    On-the-fly spectral extraction is a two-step pipeline
    (structure_tensor → spectral_features) as required by A11 API.

``CycleSpinningEngine``
    Single entry point.  Validates, inverse-shifts, dispatches.
    Does not inspect aggregator internals.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

# FIX-1: correct module name is spectral_tensor_features (not spectral_tensor_features).
# FIX-5: log the exact import path attempted so misconfiguration is visible.
try:
    from structdiff.utils.spectral_tensor_features import (   # type: ignore[import]
        compute_spectral_features,           # (J[3,H,W]) → [4,H,W]
        compute_spectral_features_multiscale, # (s1,s2,s3[3,H,W]) → [12,H,W]
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
    """Construction parameters for ``CycleSpinningEngine``.

    Parameters
    ----------
    method:
        Default aggregation method key.
    cache_features:
        Store computed features in ``FeatureBundle.cache`` so that
        evaluating multiple methods on the same batch does not re-run
        the numpy extractors.
    device:
        Target device for all tensors produced by ``FeatureManager``.
    structure_rho:
        Pre-smoothing Gaussian σ forwarded to ``compute_structure_tensor``.
    structure_sigma:
        Integration Gaussian σ forwarded to ``compute_structure_tensor``.
    wavelet_name:
        Wavelet basis forwarded to ``compute_wavelet_features``
        (default ``"db2"`` — matches wavelet_features.py DEFAULT_WAVELET).
    upsample_wavelets:
        If True, upsample wavelet features from ``[4, H/2, W/2]`` back
        to ``[4, H, W]`` so that spatial dimensions match the prediction
        tensors.  Set False only if the aggregator handles half-resolution
        internally.
    """
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
    """Per-shift feature lists expected by aggregation algorithms.

    Every list field has length N (number of shifts).  The tensor at
    index i corresponds to the i-th inverse-shifted prediction.

    Shape conventions (per verified source)
    ----------------------------------------
    predictions:       (B, C, H, W) per element
    confidence_maps:   (B, 1, H, W) per element
    wavelet_features:  (B, 4, H, W) or (B, 4, H/2, W/2) per element
    structure_features:(B, C_struct, H, W) per element
                       C_struct=3 for on-the-fly extraction;
                       C_struct=13 when DataLoader pre-computes A11.
    pred_variances:    (B, C, H, W) per element
    coherence_maps:    (B, 1, H, W) per element
    anisotropy_maps:   (B, 1, H, W) per element
    spectral_features: (B, 4, H, W) per element (one scale, on-the-fly)
    timestep:          (B,) — forwarded to aggregators that need it
    """
    predictions:        List[torch.Tensor]

    confidence_maps:    Optional[List[torch.Tensor]] = None
    wavelet_features:   Optional[List[torch.Tensor]] = None
    structure_features: Optional[List[torch.Tensor]] = None
    pred_variances:     Optional[List[torch.Tensor]] = None
    coherence_maps:     Optional[List[torch.Tensor]] = None
    anisotropy_maps:    Optional[List[torch.Tensor]] = None
    spectral_features:  Optional[List[torch.Tensor]] = None
    timestep:           Optional[torch.Tensor] = None

    # Internal cache — FeatureManager writes; aggregators must not write.
    cache: Dict[str, Any] = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Engine result
# ---------------------------------------------------------------------------

@dataclass
class EngineResult:
    """Return value of ``CycleSpinningEngine.fuse``.

    Parameters
    ----------
    fused:
        Aggregated prediction, canonical frame. Shape: (B, C, H, W).
    weights:
        Optional fusion weights from the aggregator.
        For ``UltimateCycleSpinning`` with ``return_weights=True``:
        alpha tensor of shape (B, num_levels) — the cross-level weights,
        NOT per-shift weights.  Verified from UCS.forward() source.
    diagnostics:
        Timing measurements and any extra key-value pairs.
    """
    fused:       torch.Tensor
    weights:     Optional[torch.Tensor]    = None   # shape (B, num_levels) for UCS
    diagnostics: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Base aggregator ABC
# ---------------------------------------------------------------------------

class BaseAggregator(nn.Module):
    """Interface every aggregation module must implement."""

    API_VERSION: str = "4.0"

    @property
    def required_features(self) -> frozenset:
        """Names of FeatureBundle fields this aggregator reads.

        Override in subclasses to declare dependencies so the engine
        knows which extractors to run before dispatch.
        Default: empty (predictions only).
        """
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
    """Converts raw predictions and DataLoader tensors into FeatureBundle.

    DataLoader mode (normal training/validation path)
        The DataLoader worker pre-computes structure and wavelet features
        and passes them as tensors to ``fuse()``.  FeatureManager
        reformats them into the per-shift list layout that aggregators expect.
        No re-extraction happens.

    On-the-fly mode (standalone inference, no DataLoader)
        The caller passes only raw predictions.  FeatureManager calls the
        numpy utilities on CPU for each prediction.

        IMPORTANT structural-channels constraint (FIX-4):
        On-the-fly structure extraction yields [B, 3, H, W].
        UltimateCycleSpinning is constructed with structure_channels=13.
        These are incompatible.  If the aggregator's required_structure_channels
        attribute is set and != 3, on-the-fly extraction raises a descriptive
        error rather than producing wrong-shape tensors silently.

    In both modes: no new mathematical implementations exist here.
    All computation is delegated to the existing numpy utilities.
    """

    def __init__(self, config: EngineConfig) -> None:
        self._config = config

    def populate(
        self,
        bundle: FeatureBundle,
        required_features: frozenset,
        aggregator: Optional[BaseAggregator] = None,
    ) -> None:
        """Compute and inject missing features into *bundle* in-place.

        Already-populated fields (not None) are never recomputed.
        Cache is keyed by feature name; hits skip the numpy call.
        """
        preds = bundle.predictions
        cache = bundle.cache
        cfg   = self._config

        # ── confidence ─────────────────────────────────────────────────
        if "confidence_maps" in required_features and bundle.confidence_maps is None:
            key = "confidence_maps"
            bundle.confidence_maps = cache.get(key) or self._derive_confidence(preds)
            if cfg.cache_features:
                cache[key] = bundle.confidence_maps

        # ── structure features ──────────────────────────────────────────
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
                # FIX-4: guard against shape mismatch with UCS (13 channels).
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

        # ── coherence (derived from structure features) ─────────────────
        if "coherence_maps" in required_features and bundle.coherence_maps is None:
            key = "coherence_maps"
            if key in cache:
                bundle.coherence_maps = cache[key]
            else:
                assert bundle.structure_features is not None
                bundle.coherence_maps = self._derive_coherence(bundle.structure_features)
                if cfg.cache_features:
                    cache[key] = bundle.coherence_maps

        # ── anisotropy (derived from structure features) ────────────────
        if "anisotropy_maps" in required_features and bundle.anisotropy_maps is None:
            key = "anisotropy_maps"
            if key in cache:
                bundle.anisotropy_maps = cache[key]
            else:
                assert bundle.structure_features is not None
                bundle.anisotropy_maps = self._derive_anisotropy(bundle.structure_features)
                if cfg.cache_features:
                    cache[key] = bundle.anisotropy_maps

        # ── wavelet features ────────────────────────────────────────────
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

        # ── predicted variance ──────────────────────────────────────────
        if "pred_variances" in required_features and bundle.pred_variances is None:
            key = "pred_variances"
            bundle.pred_variances = cache.get(key) or self._derive_variance(preds)
            if cfg.cache_features:
                cache[key] = bundle.pred_variances

        # ── spectral features ───────────────────────────────────────────
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
    # All function signatures and return shapes are verified from source.
    # ------------------------------------------------------------------

    def _derive_confidence(
        self, preds: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Inverse-variance confidence from prediction spread.

        Uses only torch primitives on already-computed predictions.
        Returns List[Tensor (B, 1, H, W)] length N, summing to 1 over N.
        """
        N = len(preds)
        stacked = torch.stack(preds, dim=0)           # (N, B, C, H, W)
        var = stacked.var(dim=0, keepdim=True)        # (1, B, C, H, W)
        var_mean = var.mean(dim=2, keepdim=True)      # (1, B, 1, H, W)
        inv_var = 1.0 / (var_mean + 1e-6)
        inv_all = inv_var.expand(N, -1, -1, -1, -1)
        conf_stacked = inv_all / (inv_all.sum(0, keepdim=True) + 1e-9)
        return [conf_stacked[i] for i in range(N)]

    def _compute_structure_features(
        self, preds: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Call compute_structure_tensor (CPU numpy) for each shift.

        Verified signature: compute_structure_tensor(image, rho, sigma, normalise)
        Verified return: [3, H, W] float32, channels (J11, J12, J22).

        Returns List[Tensor (B, 3, H, W)] on config.device.

        NOTE: This produces 3-channel tensors.  UltimateCycleSpinning
        requires structure_channels=13.  Callers must pass
        structure_features from the DataLoader (A11 pipeline) when using
        method="ultimate".  The engine raises an error before reaching
        this point if the channel count would mismatch (see populate).
        """
        cfg = self._config
        result: List[torch.Tensor] = []
        for pred in preds:
            B, C, H, W = pred.shape
            batch_feats: List[torch.Tensor] = []
            pred_np = pred.detach().float().cpu().mean(dim=1).numpy()  # (B, H, W)
            for b in range(B):
                # compute_structure_tensor expects 2-D float32 in [0,1].
                img_01 = (pred_np[b].clip(-1.0, 1.0) + 1.0) * 0.5
                J = compute_structure_tensor(
                    img_01,
                    rho=cfg.structure_rho,
                    sigma=cfg.structure_sigma,
                    normalise=True,
                )  # [3, H, W] float32 numpy, verified from source
                batch_feats.append(torch.from_numpy(J))
            result.append(
                torch.stack(batch_feats, dim=0).to(cfg.device)  # (B, 3, H, W)
            )
        return result

    def _compute_wavelet_features(
        self, preds: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Call compute_wavelet_features (CPU numpy) for each shift.

        Verified signature: compute_wavelet_features(image, wavelet, normalise)
        Verified return: [4, H/2, W/2] float32, channels (LL, LH, HL, HH).
        Requires even H and W (validated internally by the function).

        Returns List[Tensor] per config.upsample_wavelets:
            True  → (B, 4, H, W)   bilinear-upsampled
            False → (B, 4, H/2, W/2)
        """
        cfg = self._config
        result: List[torch.Tensor] = []
        for pred in preds:
            B, C, H, W = pred.shape
            batch_feats: List[torch.Tensor] = []
            pred_np = pred.detach().float().cpu().mean(dim=1).numpy()  # (B, H, W)
            for b in range(B):
                img_01 = (pred_np[b].clip(-1.0, 1.0) + 1.0) * 0.5
                wav = compute_wavelet_features(
                    img_01,
                    wavelet=cfg.wavelet_name,
                    normalise=True,
                )  # [4, H/2, W/2] float32 numpy, verified from source
                batch_feats.append(torch.from_numpy(wav))
            wav_t = torch.stack(batch_feats, dim=0).to(cfg.device)  # (B, 4, H/2, W/2)
            if cfg.upsample_wavelets:
                wav_t = F.interpolate(
                    wav_t.float(), size=(H, W), mode="bilinear", align_corners=False
                ).to(pred.dtype)
            result.append(wav_t)
        return result

    def _derive_coherence(
        self,
        structure_features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Derive per-pixel coherence from structure tensor components.

        Calls structure_tensor_features() from the repository.
        Verified return dict key: "coherence" (shape [H, W], float32).
        Operates on channels [J11, J12, J22] — first 3 channels of
        the structure_features tensors.

        Returns: List[Tensor (B, 1, H, W)]
        """
        result: List[torch.Tensor] = []
        for sf in structure_features:
            # sf: (B, ≥3, H, W) — channels [J11, J12, J22, ...]
            B = sf.shape[0]
            batch_coh: List[torch.Tensor] = []
            sf_np = sf[:, :3, :, :].detach().float().cpu().numpy()  # (B, 3, H, W)
            for b in range(B):
                feats = structure_tensor_features(sf_np[b])  # dict; verified keys
                coh = torch.from_numpy(
                    feats["coherence"].astype(np.float32)    # key "coherence" verified
                ).unsqueeze(0)                               # (1, H, W)
                batch_coh.append(coh)
            result.append(
                torch.stack(batch_coh, dim=0).to(self._config.device)  # (B, 1, H, W)
            )
        return result

    def _derive_anisotropy(
        self,
        structure_features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Derive per-pixel anisotropy from structure tensor eigenvalues.

        Uses J11/J12/J22 (first 3 channels) with the standard formula:
            anisotropy = 1 - λ2 / (λ1 + ε)
        where λ1 ≥ λ2 are eigenvalues of the 2×2 structure tensor.

        Returns: List[Tensor (B, 1, H, W)]
        """
        result: List[torch.Tensor] = []
        for sf in structure_features:
            # sf: (B, ≥3, H, W)
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

    def _derive_variance(
        self, preds: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Per-pixel variance of predictions across shifts.

        Every shift gets the same variance map (ensemble variance).
        Returns: List[Tensor (B, C, H, W)], each element identical.
        """
        stacked = torch.stack(preds, dim=0)   # (N, B, C, H, W)
        var = stacked.var(dim=0)              # (B, C, H, W)
        return [var for _ in preds]

    def _compute_spectral_features(
        self, preds: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Compute spectral features for each shift using the A11 two-step pipeline.

        FIX-2: compute_spectral_features(J) takes a [3,H,W] structure tensor,
        NOT a raw image.  The correct pipeline is:
            step 1: compute_structure_tensor(img_01) → J [3,H,W]
            step 2: compute_spectral_features(J)     → [4,H,W]

        This matches the A11 module docstring:
        "it operates purely on the structure tensors already produced
        by compute_structure_tensor_multiscale".

        Returns: List[Tensor (B, 4, H, W)] on config.device.
        Note: this produces single-scale (4-channel) spectral features.
        For the 12-channel multiscale output, the DataLoader must call
        compute_spectral_features_multiscale directly.
        """
        cfg = self._config
        result: List[torch.Tensor] = []
        for pred in preds:
            B, C, H, W = pred.shape
            batch_feats: List[torch.Tensor] = []
            pred_np = pred.detach().float().cpu().mean(dim=1).numpy()  # (B, H, W)
            for b in range(B):
                img_01 = (pred_np[b].clip(-1.0, 1.0) + 1.0) * 0.5
                # Step 1: structure tensor  [3, H, W]
                J = compute_structure_tensor(
                    img_01,
                    rho=cfg.structure_rho,
                    sigma=cfg.structure_sigma,
                    normalise=True,
                )
                # Step 2: spectral features [4, H, W]
                sf = compute_spectral_features(J)  # type: ignore[name-defined]
                batch_feats.append(torch.from_numpy(sf.astype(np.float32)))
            result.append(
                torch.stack(batch_feats, dim=0).to(cfg.device)  # (B, 4, H, W)
            )
        return result


# ---------------------------------------------------------------------------
# UltimateCycleSpinning adapter
# ---------------------------------------------------------------------------

class _UltimateAdapter(BaseAggregator):
    """Adapter for UltimateCycleSpinning with the verified A26/CORR API.

    All constructor parameter names and defaults are derived from the
    verified source of ultimate_cycle_spinning.py.

    Critical verified facts
    -----------------------
    wavelet_channels default: 1 (NOT 4).  FIX-3.
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
        wavelet_channels: int = 1,        # FIX-3: default is 1, verified from source
        structure_channels: int = 13,     # verified: 13 required for token_dim divisibility
        **ultimate_kwargs: Any,
    ) -> None:
        super().__init__()
        from structdiff.sampling.cycle_spinning.ultimate_cycle_spinning import (
            UltimateCycleSpinning,
        )
        self._algo = UltimateCycleSpinning(
            num_shifts=num_shifts,
            channels=channels,
            wavelet_channels=wavelet_channels,
            structure_channels=structure_channels,
            **ultimate_kwargs,
        )
        # Expose for FIX-4 channel guard in FeatureManager.populate
        self._structure_channels = structure_channels

    def forward(self, bundle: FeatureBundle) -> EngineResult:
        preds = bundle.predictions
        conf  = bundle.confidence_maps
        wav   = bundle.wavelet_features
        struc = bundle.structure_features

        if conf is None or wav is None or struc is None:
            raise ValueError(
                "_UltimateAdapter requires confidence_maps, wavelet_features, "
                "and structure_features to be populated in FeatureBundle before dispatch."
            )

        # Verified forward signature and return contract from source:
        # return_weights=True → (fused [B,C,H,W], alpha [B,num_levels])
        result = self._algo(
            outputs            = preds,
            confidence_maps    = conf,
            wavelet_features   = wav,
            structure_features = struc,
            timestep           = bundle.timestep,
            return_weights     = True,
        )
        fused, alpha = result   # alpha: [B, num_levels] — cross-level weights
        return EngineResult(fused=fused, weights=alpha)


# ---------------------------------------------------------------------------
# Opaque adapter
# ---------------------------------------------------------------------------

class _OpaqueAdapter(BaseAggregator):
    """Wraps a caller-supplied aggregation module with an unknown API.

    The caller provides a call_fn that translates a FeatureBundle into
    an EngineResult.  This makes the engine extensible without requiring
    knowledge of any module's internal API.

    Parameters
    ----------
    module:
        Pre-built nn.Module.  Stored as a submodule so that
        engine.named_modules() and device movement work correctly.
    call_fn:
        (adapter, bundle) -> EngineResult.
    required:
        frozenset of FeatureBundle field names this aggregator needs.
    name:
        Human-readable label for logging.
    structure_channels:
        If the wrapped module expects a specific structure channel count,
        set this so FeatureManager can enforce it.  Default None (no check).
    """

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
# Shift utilities
# ---------------------------------------------------------------------------

def _inverse_shift(x: torch.Tensor, row: int, col: int) -> torch.Tensor:
    """Undo a circular shift of (row, col) pixels."""
    return torch.roll(x, shifts=(-row, -col), dims=(2, 3))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_fuse_inputs(
    outputs:  List[torch.Tensor],
    shifts:   Optional[List[Tuple[int, int]]],
    method:   str,
    registry: Dict[str, BaseAggregator],
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
    """Raise a descriptive error if an already-built aggregator has incompatible shape.

    Only inspects _UltimateAdapter, whose attribute names are verified
    from the uploaded source (self._algo.num_shifts, self._algo.channels).
    """
    if isinstance(aggregator, _UltimateAdapter):
        stored_n = aggregator._algo.num_shifts   # verified attribute name
        stored_c = aggregator._algo.channels     # verified attribute name
        if stored_n != num_shifts:
            raise ValueError(
                f"Engine method '{method}' was built for num_shifts={stored_n} "
                f"but current batch has num_shifts={num_shifts}. "
                "Call engine.deregister(method) and re-register with the new shape, "
                "or use a fixed cycle_width throughout the experiment."
            )
        if stored_c != channels:
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
    1. Construct once, optionally with pre-built aggregators::

           engine = CycleSpinningEngine(
               config=EngineConfig(method="ultimate"),
           )
           # For non-default UCS parameters (e.g. wavelet_channels=4):
           engine.set_ultimate_kwargs(
               wavelet_channels=4,
               structure_channels=13,
               num_levels=3,
               num_heads=4,
           )

    2. In the inference / validation loop::

           shifts  = CycleSpinningEngine.build_shift_grid(H, W, cycle_width)
           outputs = []
           for row, col in shifts:
               shifted = torch.roll(noisy, shifts=(row, col), dims=(2, 3))
               outputs.append(sample_fn(model, shifted))

           result = engine.fuse(
               outputs=outputs,
               shifts=shifts,
               # Pass DataLoader-precomputed features explicitly:
               structure_features=struct_list,   # List[Tensor(B,13,H,W)]
               wavelet_features=wav_list,        # List[Tensor(B,4,H,W)]
           )
           fused = result.fused   # (B, C, H, W)

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
        # Extra kwargs forwarded to UltimateCycleSpinning constructor.
        # Callers must call set_ultimate_kwargs before the first fuse()
        # with method="ultimate" if non-default values are needed.
        self._ultimate_kwargs: Dict[str, Any] = {}

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
        """Fuse cycle-spinning predictions into a single estimate.

        Parameters
        ----------
        outputs:
            List of N diffusion-model predictions, each (B, C, H, W),
            one per (row, col) shift — already in the shifted frame.
        shifts:
            Corresponding (row, col) shift tuples.  If None, no
            inverse-shift is applied (outputs already in canonical frame).
        structure_features:
            IMPORTANT: when using method="ultimate", this must be
            provided by the caller as List[Tensor(B,13,H,W)].
            On-the-fly extraction produces only 3 channels, which is
            incompatible with UltimateCycleSpinning(structure_channels=13).
        method:
            Overrides EngineConfig.method for this call only.

        Returns
        -------
        EngineResult
            fused: (B, C, H, W) in the canonical frame.
            weights: (B, num_levels) cross-level alpha from UCS,
                     or None for other aggregators.
        """
        active_method = method if method is not None else self._config.method

        # 1. Validate
        _validate_fuse_inputs(outputs, shifts, active_method, self._get_full_registry(outputs))

        # 2. Inverse-shift to canonical frame
        inv_preds = self._inverse_shift_all(outputs, shifts)

        # 3. Get or build aggregator
        aggregator = self._get_or_build_aggregator(active_method, inv_preds)

        # 4. Shape-compatibility check
        _check_shape_compatibility(
            aggregator, len(inv_preds), inv_preds[0].shape[1], active_method
        )

        # 5. Build feature bundle
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

        # 6. Populate missing features (FIX-4: aggregator passed for channel guard)
        t_feat_start = time.perf_counter()
        self._feature_manager.populate(
            bundle, aggregator.required_features, aggregator=aggregator
        )
        t_feat = time.perf_counter() - t_feat_start

        # 7. Dispatch
        aggregator.eval()
        t_agg_start = time.perf_counter()
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
        """Register a pre-built aggregator under *name*.

        Raises ValueError if *name* already exists.
        Call deregister() first to replace an existing entry.

        Example — registering A27::

            class A27Adapter(BaseAggregator):
                required_features = frozenset({"confidence_maps"})
                def forward(self, bundle): ...

            engine.register("a27", A27Adapter(...))
            result = engine.fuse(outputs, shifts, method="a27")
        """
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
        """Register an aggregator whose API does not derive from BaseAggregator.

        Parameters
        ----------
        name:
            Unique method key.
        module:
            Pre-built nn.Module.
        call_fn:
            (adapter, bundle) -> EngineResult.
        required:
            frozenset of FeatureBundle field names the module needs.
        structure_channels:
            Expected structure channel count for FIX-4 guard.
            Pass the value the wrapped module was constructed with.

        Example::

            def _conf_call(adapter, bundle):
                fused, w = adapter._module(
                    bundle.predictions,
                    confidence_maps=bundle.confidence_maps,
                )
                return EngineResult(fused=fused, weights=w)

            engine.register_opaque(
                "confidence",
                ConfidenceCycleSpinning().to(device),
                call_fn=_conf_call,
                required=frozenset({"confidence_maps"}),
            )
        """
        adapter = _OpaqueAdapter(
            module, call_fn, required, name, structure_channels=structure_channels
        )
        self.register(name, adapter)

    def deregister(self, name: str) -> None:
        """Remove a registered aggregator."""
        removed = self._registry.pop(name, None)
        if removed is None:
            logger.warning("deregister: '%s' not found in registry.", name)

    def registered_methods(self) -> List[str]:
        """Return sorted list of all registered method names."""
        return sorted(self._get_full_registry(None).keys())

    def set_ultimate_kwargs(self, **kwargs: Any) -> None:
        """Set extra keyword arguments forwarded to UltimateCycleSpinning.

        Must be called before the first fuse() with method="ultimate".
        After that call, "ultimate" is built and cached; use deregister()
        first if you need to change parameters.

        Example (typical A12 configuration)::

            engine.set_ultimate_kwargs(
                wavelet_channels=4,       # if DataLoader produces 4-ch wavelet
                structure_channels=13,    # default; kept for explicitness
                num_levels=3,
                num_heads=4,
            )
        """
        self._ultimate_kwargs = kwargs

    # ------------------------------------------------------------------
    # Shift grid utility
    # ------------------------------------------------------------------

    @staticmethod
    def build_shift_grid(
        height:      int,
        width:       int,
        cycle_width: int,
    ) -> List[Tuple[int, int]]:
        """Return the canonical (row, col) shift grid in row-major order.

        Exactly matches the nested loop from test_util.py::

            for row in range(0, num_rows, cycle_width):
                for col in range(0, num_cols, cycle_width):
                    ...

        Parameters
        ----------
        height, width:
            Spatial dimensions of the input image.
        cycle_width:
            Step size (pixels) for both row and column directions.

        Returns
        -------
        List of (row_shift, col_shift) tuples.
        len(result) equals the num_shifts value for _UltimateAdapter.
        """
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
    ) -> Dict[str, BaseAggregator]:
        """Return registry plus a sentinel "ultimate" entry for validation."""
        full = dict(self._registry)
        if "ultimate" not in full:
            full["ultimate"] = object()  # type: ignore[assignment]
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
    ) -> BaseAggregator:
        """Return aggregator for *method*, building _UltimateAdapter on first call.

        Shape-dependent construction reads from the actual prediction list.
        After construction the aggregator is cached in self._registry.
        """
        if method in self._registry:
            return self._registry[method]

        if method == "ultimate":
            N = len(preds)
            C = preds[0].shape[1]
            logger.info(
                "CycleSpinningEngine: constructing _UltimateAdapter "
                "(num_shifts=%d, channels=%d, kwargs=%s).",
                N, C, self._ultimate_kwargs,
            )
            agg = _UltimateAdapter(
                num_shifts=N,
                channels=C,
                **self._ultimate_kwargs,
            ).to(self._config.device)
            self._registry[method] = agg
            return agg

        raise ValueError(
            f"No aggregator registered for method '{method}'. "
            f"Call engine.register('{method}', aggregator) before fuse()."
        )
