"""
sampling/statistics_extractor.py
=================================
Unified image statistics extraction for the Ultimate Sampling Framework.

This module is the single entry point for all spatial feature computation.
SamplingController calls:

    stats = extractor(x)         → StatisticsMaps

and then passes stats fields into AdaptiveBetaController and SamplingState
without knowing which sub-module produced each map.

Architecture
------------
StatisticsExtractor internally orchestrates:

    MultiScaleExtractor   → edge, entropy, enl, cv, texture, freq_energy,
                            cv_variance, skewness, kurtosis,
                            gradient_coherence
    StructureTensorModule → lambda1, lambda2, coherence, anisotropy,
                            orientation, cos2theta, sin2theta,
                            orientation_confidence, lambda_sum,
                            lambda_ratio, weighted_orientation,
                            eigenvalue_entropy, scale_consistency

Cache
-----
Single-entry cache keyed on (data_ptr, _version, shape, device).
Using _version (PyTorch's internal storage-mutation counter) prevents the
old data_ptr-only key from returning stale results when a tensor is
modified in-place or when PyTorch reuses the same memory address.

Lazy execution
--------------
StatisticsExtractor accepts a ``requested_fields`` set.  When provided,
only the sub-module outputs actually needed are computed.  Both sub-modules
are skipped if none of their fields are requested.  This enables ablation
stages that use only a subset of statistics to avoid paying for the full
computation.

If ``requested_fields`` is None (default), all enabled modules are run.

AMP safety
----------
All computation is done in float32 internally.  Output dtype matches the
input tensor dtype if ``restore_dtype=True`` (default False for backward
compatibility).

Profiling
---------
When ``enable_profiling=True``, forward() records wall-clock runtimes for
each sub-module in ``self.profile_ms`` (dict: module_name → ms).  This
adds minimal overhead (two time.perf_counter() calls per module).

Optional modules
----------------
Respects config.enable_structure_tensor and config.enable_multiscale_maps.
When a module is disabled, its output fields are None.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import torch
import torch.nn as nn

from .config            import SamplingConfig
from .multiscale_maps   import MultiScaleExtractor, MultiScaleMaps
from .structure_tensor  import StructureTensorModule, StructureTensorMaps


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class StatisticsMaps:
    """
    Unified output of StatisticsExtractor.

    All Optional[torch.Tensor] fields have shape [N,1,H,W] and dtype
    matching the extractor's output policy (float32 by default).
    None indicates that the producing sub-module was disabled or the field
    was not requested via ``requested_fields``.

    Multi-scale maps (from MultiScaleExtractor)
    --------------------------------------------
    edge               — fused edge magnitude
    entropy            — fused MAD-based entropy proxy
    enl                — fused Equivalent Number of Looks
    cv                 — fused local coefficient of variation  (A7)
    texture            — fused Laplacian variance
    freq_energy        — fused local FFT power spectrum energy (A12)
    cv_variance        — Var(CV across scales) per pixel
    skewness           — fused local skewness
    kurtosis           — fused local kurtosis
    gradient_coherence — mean cosine similarity of neighbouring gradients

    Structure tensor maps (from StructureTensorModule)
    ---------------------------------------------------
    lambda1              — major eigenvalue
    lambda2              — minor eigenvalue
    coherence            — (λ₁-λ₂)/(λ₁+λ₂+ε) ∈ [0,1]
    anisotropy           — alias for coherence (same tensor)
    orientation          — dominant edge angle ∈ [-π/2, π/2]
    cos2theta            — cos(2θ), π-ambiguity-free orientation encoding
    sin2theta            — sin(2θ), π-ambiguity-free orientation encoding
    orientation_confidence — coherence² (∈ [0,1])
    lambda_sum           — λ₁+λ₂ (total gradient energy)
    lambda_ratio         — λ₁/(λ₂+ε)
    weighted_orientation — coherence·orientation
    eigenvalue_entropy   — Shannon entropy of (λ₁,λ₂) distribution
    scale_consistency    — Var(λ₁ across integration scales)

    Injected from external modules (populated after forward())
    ----------------------------------------------------------
    spin_variance        — Var(predictions across cycle-spin passes)  (A26)
    confidence_entropy   — H(confidence map) per pixel                (A9/A14)
    """
    # ── Multi-scale maps ───────────────────────────────────────────────────
    edge:               Optional[torch.Tensor] = None
    entropy:            Optional[torch.Tensor] = None
    enl:                Optional[torch.Tensor] = None
    cv:                 Optional[torch.Tensor] = None
    texture:            Optional[torch.Tensor] = None
    freq_energy:        Optional[torch.Tensor] = None
    cv_variance:        Optional[torch.Tensor] = None
    skewness:           Optional[torch.Tensor] = None
    kurtosis:           Optional[torch.Tensor] = None
    gradient_coherence: Optional[torch.Tensor] = None

    # ── Structure tensor maps ──────────────────────────────────────────────
    lambda1:               Optional[torch.Tensor] = None
    lambda2:               Optional[torch.Tensor] = None
    coherence:             Optional[torch.Tensor] = None
    anisotropy:            Optional[torch.Tensor] = None
    orientation:           Optional[torch.Tensor] = None
    cos2theta:             Optional[torch.Tensor] = None
    sin2theta:             Optional[torch.Tensor] = None
    orientation_confidence:Optional[torch.Tensor] = None
    lambda_sum:            Optional[torch.Tensor] = None
    lambda_ratio:          Optional[torch.Tensor] = None
    weighted_orientation:  Optional[torch.Tensor] = None
    eigenvalue_entropy:    Optional[torch.Tensor] = None
    scale_consistency:     Optional[torch.Tensor] = None

    # ── Injected from external modules ────────────────────────────────────
    spin_variance:      Optional[torch.Tensor] = None
    confidence_entropy: Optional[torch.Tensor] = None


# Field membership sets — used by lazy-execution logic
_MS_FIELDS: Set[str] = {
    "edge", "entropy", "enl", "cv", "texture", "freq_energy",
    "cv_variance", "skewness", "kurtosis", "gradient_coherence",
}
_ST_FIELDS: Set[str] = {
    "lambda1", "lambda2", "coherence", "anisotropy",
    "orientation", "cos2theta", "sin2theta",
    "orientation_confidence", "lambda_sum", "lambda_ratio",
    "weighted_orientation", "eigenvalue_entropy", "scale_consistency",
}


# ---------------------------------------------------------------------------
# StatisticsExtractor
# ---------------------------------------------------------------------------

class StatisticsExtractor(nn.Module):
    """
    Orchestrates MultiScaleExtractor and StructureTensorModule.

    Parameters
    ----------
    config : SamplingConfig
        Reads:
            enable_multiscale_maps    : bool
            enable_structure_tensor   : bool
            sigma_gradient            : float
            sigma_integration         : float
    ms_extractor : Optional[MultiScaleExtractor]
        Pre-built multi-scale extractor.  If None and
        enable_multiscale_maps=True, one is constructed with default
        parameters.
    st_module : Optional[StructureTensorModule]
        Pre-built structure tensor module.  If None and
        enable_structure_tensor=True, one is constructed with
        config.sigma_gradient / sigma_integration.
    enable_profiling : bool
        When True, sub-module runtimes are recorded in self.profile_ms.
    restore_dtype : bool
        When True, output tensors are cast back to the input's original
        dtype.  Default False (always float32) for backward compatibility.
    """

    def __init__(
        self,
        config:          SamplingConfig,
        ms_extractor:    Optional[MultiScaleExtractor]   = None,
        st_module:       Optional[StructureTensorModule] = None,
        enable_profiling: bool = False,
        restore_dtype:   bool = False,
    ) -> None:
        super().__init__()
        self.config           = config
        self.enable_profiling = enable_profiling
        self.restore_dtype    = restore_dtype

        # Profiling accumulator: module_name → last runtime in ms
        self.profile_ms: Dict[str, float] = {}

        # ── Multi-scale extractor ─────────────────────────────────────
        if config.enable_multiscale_maps:
            self.ms: Optional[MultiScaleExtractor] = (
                ms_extractor if ms_extractor is not None
                else MultiScaleExtractor()
            )
        else:
            self.ms = None

        # ── Structure tensor module ───────────────────────────────────
        if config.enable_structure_tensor:
            self.st: Optional[StructureTensorModule] = (
                st_module if st_module is not None
                else StructureTensorModule(
                    sigma_gradient     = config.sigma_gradient,
                    integration_sigmas = (
                        config.sigma_integration / 2.0,
                        config.sigma_integration,
                        config.sigma_integration * 2.0,
                    ),
                )
            )
        else:
            self.st = None

        # Single-entry cache: (data_ptr, _version, shape, device) → StatisticsMaps
        self._cache_key:    Optional[tuple] = None
        self._cache_result: Optional[StatisticsMaps] = None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _make_cache_key(self, x: torch.Tensor) -> tuple:
        """
        Build a cache key that is invalidated by in-place mutations.

        _version is PyTorch's internal storage mutation counter.
        It increments on every in-place op, so same data_ptr but
        different content → different key.  Prevents stale cache hits
        that occurred with the old (data_ptr, shape, device) key.
        """
        return (x.data_ptr(), x._version, tuple(x.shape), str(x.device))

    def _cache_hit(self, x: torch.Tensor) -> bool:
        return self._make_cache_key(x) == self._cache_key

    def _update_cache(self, x: torch.Tensor, result: StatisticsMaps) -> None:
        self._cache_key    = self._make_cache_key(x)
        self._cache_result = result

    def invalidate_cache(self) -> None:
        """Explicitly clear the cache (e.g. between independent sample runs)."""
        self._cache_key    = None
        self._cache_result = None

    # ------------------------------------------------------------------
    # Profiling helper
    # ------------------------------------------------------------------

    def _timed(self, name: str, fn):
        """Run fn() and optionally record its wall-clock time."""
        if not self.enable_profiling:
            return fn()
        t0 = time.perf_counter()
        result = fn()
        self.profile_ms[name] = (time.perf_counter() - t0) * 1e3
        return result

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        requested_fields: Optional[Set[str]] = None,
    ) -> StatisticsMaps:
        """
        Extract all enabled spatial statistics from x.

        Parameters
        ----------
        x : torch.Tensor  [N,C,H,W]
            Current noisy sample.  Any dtype; cast to float32 internally.
        requested_fields : Optional[Set[str]]
            If provided, only fields in this set are computed.  Fields
            that belong to a disabled or un-requested sub-module are None.
            If None, all enabled sub-modules are run.

        Returns
        -------
        StatisticsMaps
            Disabled or un-requested fields are None.
            Present fields are float32 (or input dtype if restore_dtype=True).
        """
        if self._cache_hit(x):
            assert self._cache_result is not None
            return self._cache_result

        in_dtype = x.dtype
        xf = x.to(dtype=torch.float32)

        # ── Lazy execution: determine which sub-modules to run ────────
        need_ms = (
            self.ms is not None
            and (
                requested_fields is None
                or bool(requested_fields & _MS_FIELDS)
            )
        )
        need_st = (
            self.st is not None
            and (
                requested_fields is None
                or bool(requested_fields & _ST_FIELDS)
            )
        )

        # ── Multi-scale maps ──────────────────────────────────────────
        ms_maps: Optional[MultiScaleMaps] = None
        if need_ms:
            ms_maps = self._timed("multiscale", lambda: self.ms(xf))

        # ── Structure tensor maps ─────────────────────────────────────
        st_maps: Optional[StructureTensorMaps] = None
        if need_st:
            st_maps = self._timed("structure_tensor", lambda: self.st(xf))

        # ── Assemble output ───────────────────────────────────────────
        def _ms(attr: str) -> Optional[torch.Tensor]:
            """Return ms_maps.attr if present and requested, else None."""
            if ms_maps is None:
                return None
            if requested_fields is not None and attr not in requested_fields:
                return None
            return getattr(ms_maps, attr, None)

        def _st(attr: str) -> Optional[torch.Tensor]:
            """Return st_maps.attr if present and requested, else None."""
            if st_maps is None:
                return None
            if requested_fields is not None and attr not in requested_fields:
                return None
            return getattr(st_maps, attr, None)

        result = StatisticsMaps(
            # Multi-scale
            edge               = _ms("edge"),
            entropy            = _ms("entropy"),
            enl                = _ms("enl"),
            cv                 = _ms("cv"),
            texture            = _ms("texture"),
            freq_energy        = _ms("freq_energy"),
            cv_variance        = _ms("cv_variance"),
            skewness           = _ms("skewness"),
            kurtosis           = _ms("kurtosis"),
            gradient_coherence = _ms("gradient_coherence"),
            # Structure tensor
            lambda1               = _st("lambda1"),
            lambda2               = _st("lambda2"),
            coherence             = _st("coherence"),
            anisotropy            = _st("anisotropy"),
            orientation           = _st("orientation"),
            cos2theta             = _st("cos2theta"),
            sin2theta             = _st("sin2theta"),
            orientation_confidence= _st("orientation_confidence"),
            lambda_sum            = _st("lambda_sum"),
            lambda_ratio          = _st("lambda_ratio"),
            weighted_orientation  = _st("weighted_orientation"),
            eigenvalue_entropy    = _st("eigenvalue_entropy"),
            scale_consistency     = _st("scale_consistency"),
            # Injected fields — always None until set externally
            spin_variance      = None,
            confidence_entropy = None,
        )

        # ── Optional dtype restore ────────────────────────────────────
        if self.restore_dtype and in_dtype != torch.float32:
            result = self._cast_result(result, in_dtype)

        self._update_cache(x, result)
        return result

    # ------------------------------------------------------------------
    # Dtype restore helper
    # ------------------------------------------------------------------

    @staticmethod
    def _cast_result(result: StatisticsMaps, dtype: torch.dtype) -> StatisticsMaps:
        """Cast all non-None tensor fields to ``dtype``."""
        d = {}
        for f_name, val in result.__dict__.items():
            if isinstance(val, torch.Tensor):
                d[f_name] = val.to(dtype=dtype)
            else:
                d[f_name] = val
        return StatisticsMaps(**d)

    # ------------------------------------------------------------------
    # Convenience: apply to SamplingState
    # ------------------------------------------------------------------

    def apply_to_state(
        self,
        state,
        device: torch.device,
        requested_fields: Optional[Set[str]] = None,
    ):
        """
        Convenience: extract stats from state.xt and return a new state
        with all statistics map fields populated.

        Parameters
        ----------
        state             : SamplingState
        device            : torch.device
        requested_fields  : Optional[Set[str]]
            Forwarded to forward().  None → compute everything.

        Returns
        -------
        SamplingState  — new instance with maps filled in.
        """
        stats = self.forward(state.xt.to(device), requested_fields)
        return state.replace(
            edge_map             = stats.edge,
            entropy_map          = stats.entropy,
            enl_map              = stats.enl,
            cv_map               = stats.cv,
            texture_map          = stats.texture,
            freq_energy_map      = stats.freq_energy,
            cv_variance_map      = stats.cv_variance,
            skewness_map         = stats.skewness,
            kurtosis_map         = stats.kurtosis,
            gradient_coherence_map = stats.gradient_coherence,
            lambda1_map          = stats.lambda1,
            lambda2_map          = stats.lambda2,
            coherence_map        = stats.coherence,
            anisotropy_map       = stats.anisotropy,
            orientation_map      = stats.orientation,
            cos2theta_map        = stats.cos2theta,
            sin2theta_map        = stats.sin2theta,
            orientation_confidence_map = stats.orientation_confidence,
            eigenvalue_entropy_map = stats.eigenvalue_entropy,
            scale_consistency_map  = stats.scale_consistency,
        )
