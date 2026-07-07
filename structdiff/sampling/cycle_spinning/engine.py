"""
structdiff/sampling/cycle_spinning/engine.py
<<<<<<< HEAD
=============================================
Unified Cycle Spinning Engine.

Orchestrates every existing cycle-spinning aggregation module
(``learnable``, ``adaptive``, ``confidence``, ``wavelet_confidence``,
``structure_wavelet``, ``transformer``, ``learnable_shift``,
``ultimate``) behind a single, stable, strongly-typed interface so
``guided_diffusion/test_util.py`` (or any other inference script) can
switch aggregation strategy by changing one field
(``EngineConfig.method``) instead of hand-wiring a different module
each time.

No existing aggregation algorithm is modified, subclassed, or
monkey-patched. This file only *instantiates* and *dispatches to* the
modules that already live under ``structdiff/sampling/cycle_spinning/``.

Provenance / verification notes
--------------------------------
Every adapter below is now built from real, verbatim source for all
eight modules -- there are no remaining inferred signatures:

    learnable_cycle_spinning.py                    (A26a) -- VERIFIED
    adaptive_cycle_spinning.py                     (A26b) -- VERIFIED
    confidence_cycle_spinning.py                   (A26c) -- VERIFIED
    wavelet_confidence_cycle_spinning.py           (A26d) -- VERIFIED
    structure_wavelet_confidence_cycle_spinning.py (A26e) -- VERIFIED
    transformer_cycle_spinning.py                  (A26f) -- VERIFIED
    learnable_shift_cycle_spinning.py              (A26g) -- VERIFIED
    ultimate_cycle_spinning.py                     -- VERIFIED

(An earlier revision of this file inferred the constructor/forward
shapes of A26a, A26b, and A26f from the consistent A26-series contract
documented in the other modules, before their source was available.
Cross-checking against the real source afterwards confirmed every
inferred shape was correct except one: the ``learnable`` adapter was
not forwarding ``EngineConfig.temperature`` into
``LearnableCycleSpinning``'s ``temperature`` constructor argument.
That has been fixed below.)

Extensibility
-------------
New aggregation methods do not require editing this file's internals.
Call the module-level :func:`register_aggregator` once (e.g. from an
extension module) with a fresh :class:`BaseAggregator` subclass and it
becomes available as ``EngineConfig(method="...")`` immediately.
=======
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
>>>>>>> cycle-engine
"""

from __future__ import annotations

import logging
<<<<<<< HEAD
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Type, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports of the existing, unmodified aggregation modules.
# ---------------------------------------------------------------------------
from .learnable_cycle_spinning import LearnableCycleSpinning
from .adaptive_cycle_spinning import AdaptiveCycleSpinning
from .confidence_cycle_spinning import ConfidenceCycleSpinning
from .wavelet_confidence_cycle_spinning import WaveletConfidenceCycleSpinning
from .structure_wavelet_confidence_cycle_spinning import (
    StructureWaveletConfidenceCycleSpinning,
)
from .transformer_cycle_spinning import TransformerCycleSpinning
from .learnable_shift_cycle_spinning import LearnableShiftCycleSpinning
from .ultimate_cycle_spinning import UltimateCycleSpinning


# =============================================================================
# Public configuration / result types
# =============================================================================

#: The full set of feature kinds any adapter might require. The engine only
#: validates and forwards the subset a given method actually declares via
#: ``BaseAggregator.required_features``.
FeatureName = str
_ALL_FEATURE_NAMES: Tuple[FeatureName, ...] = (
    "confidence_maps",
    "wavelet_features",
    "structure_features",
    "spectral_features",
    "pred_variances",
    "coherence_maps",
    "anisotropy_maps",
)


@dataclass(frozen=True)
class EngineConfig:
    """Immutable configuration selecting and parameterising one aggregation method.

    Switching aggregation strategy is a one-line change::

        EngineConfig(method="ultimate")  ->  EngineConfig(method="transformer")

    Parameters
    ----------
    method:
        One of ``"learnable"``, ``"adaptive"``, ``"confidence"``,
        ``"wavelet_confidence"``, ``"structure_wavelet"``,
        ``"transformer"``, ``"learnable_shift"``, ``"ultimate"``.
    num_shifts:
        Number of cycle-spin shifts *N*. Must match ``len(outputs)`` at
        call time and the shift-grid used by the caller's inference loop.
    channels:
        Number of channels in each shifted prediction tensor (e.g. 1 for
        grayscale SAR amplitude images).
    wavelet_channels:
        Channel count of each wavelet feature tensor. Required only by
        methods that consume ``wavelet_features``.
    structure_channels:
        Channel count of each structure-tensor feature. Required only by
        methods that consume ``structure_features``.
    hidden_dim:
        MLP hidden width for the MLP-based methods (``adaptive``,
        ``confidence``, ``wavelet_confidence``, ``structure_wavelet``).
    temperature:
        Softmax temperature shared by every method's weight predictor.
    pooling:
        Spatial pooling mode (``"avg"`` or ``"max"``) shared by the
        MLP/Transformer-based methods.
    eps:
        Numerical-stability epsilon shared by every method.
    num_heads, num_layers, dropout:
        Transformer hyperparameters, used by ``transformer``,
        ``learnable_shift``, and (with different field names) ``ultimate``.
    coordinate_embed_dim, max_shift_radius, radius_lambda, repulsion_lambda:
        ``learnable_shift``-specific hyperparameters.
    ultimate_kwargs:
        Free-form dict of additional constructor kwargs forwarded only to
        :class:`UltimateCycleSpinning`, which has many hyperparameters
        (MoE, frequency pyramid, deformable attention, ...) that do not
        generalise to the other seven methods. Anything not given here
        falls back to ``UltimateCycleSpinning``'s own defaults.
    device:
        Optional explicit device string/``torch.device``. If ``None``,
        the engine infers the device from the first call's ``outputs[0]``
        and moves the (lazily constructed) aggregator there once.

    Examples
    --------
    >>> cfg = EngineConfig(method="wavelet_confidence", num_shifts=9,
    ...                     channels=1, wavelet_channels=4)
    >>> cfg.method
    'wavelet_confidence'
    """

    method: str
    num_shifts: int = 9
    channels: int = 1
    wavelet_channels: int = 4
    structure_channels: int = 12
    hidden_dim: int = 128
    temperature: float = 1.0
    pooling: str = "avg"
    eps: float = 1e-8
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    coordinate_embed_dim: int = 16
    max_shift_radius: float = 3.0
    radius_lambda: float = 1e-4
    repulsion_lambda: float = 1e-3
    ultimate_kwargs: Dict[str, Any] = field(default_factory=dict)
    device: Optional[Union[str, torch.device]] = None


@dataclass(frozen=True)
class FeatureBundle:
    """Container for every optional per-shift feature sequence the engine accepts.

    Each field, if provided, must be a sequence of length ``num_shifts``
    of 4-D tensors ``[B, C_feature, H_feature, W_feature]`` aligned
    index-for-index with ``outputs``. Only the fields actually required
    by the selected aggregation method are validated and forwarded; the
    rest are accepted but ignored (so callers do not need to know in
    advance which method will consume which features).

    Parameters
    ----------
    confidence_maps, wavelet_features, structure_features:
        Forwarded, when required, to the matching constructor argument
        of the underlying aggregation module.
    spectral_features, pred_variances, coherence_maps, anisotropy_maps:
        Reserved for future aggregation methods. No currently-registered
        method declares these as required, so they are accepted (for
        forward API stability) but never forwarded today. If a future
        method needs one of these, only that method's adapter needs to
        declare it in ``required_features`` and consume it in ``_call``.
    """

    confidence_maps: Optional[Sequence[torch.Tensor]] = None
    wavelet_features: Optional[Sequence[torch.Tensor]] = None
    structure_features: Optional[Sequence[torch.Tensor]] = None
    spectral_features: Optional[Sequence[torch.Tensor]] = None
    pred_variances: Optional[Sequence[torch.Tensor]] = None
    coherence_maps: Optional[Sequence[torch.Tensor]] = None
    anisotropy_maps: Optional[Sequence[torch.Tensor]] = None

    def get(self, name: FeatureName) -> Optional[Sequence[torch.Tensor]]:
        """Look up a feature sequence by its string name.

        Parameters
        ----------
        name:
            One of the field names listed in this class.

        Returns
        -------
        Optional[Sequence[torch.Tensor]]
            The corresponding sequence, or ``None`` if not supplied.

        Raises
        ------
        AttributeError
            If ``name`` is not a recognised feature name.
        """
        return getattr(self, name)


@dataclass(frozen=True)
class EngineResult:
    """Return value of :meth:`CycleSpinningEngine.fuse`.

    Parameters
    ----------
    fused:
        The aggregated prediction, shape ``[B, C, H, W]``.
    metadata:
        Diagnostic and provenance information about the call:

        ``"method"``
            The method name that was actually dispatched to.
        ``"weights"``
            Per-shift aggregation weights ``[B, num_shifts]`` if the
            underlying method produced any (every registered method
            does), else ``None``.
        ``"required_features"``
            The set of feature names this method declared and consumed.
        ``"num_shifts"``
            Number of shifts used for this call.
        ``"extra"``
            Method-specific extras (e.g. ``UltimateCycleSpinning``'s
            per-level outputs / cross-level alpha / uncertainty, when
            requested) as a free-form dict. Empty for methods that do
            not produce anything beyond ``fused`` and ``weights``.
    """

    fused: torch.Tensor
    metadata: Dict[str, Any]


# =============================================================================
# Adapter pattern
# =============================================================================


class BaseAggregator:
    """Adapter base class: normalises one aggregation module behind a common call shape.

    Subclasses wrap exactly one existing ``nn.Module`` aggregation class.
    They never reimplement aggregation logic; they only:

    1. Declare which :class:`FeatureBundle` fields they need
       (:meth:`required_features`).
    2. Build the wrapped module from an :class:`EngineConfig`
       (:meth:`build`).
    3. Translate the engine's generic ``(outputs, features)`` call shape
       into the wrapped module's specific positional/keyword signature
       (:meth:`call`).

    This indirection is what lets :class:`CycleSpinningEngine` treat all
    eight methods identically without any ``if method == ...`` branches
    in the hot path.
    """

    #: Registry key this adapter is responsible for. Set by subclasses.
    method_name: str = ""

    def required_features(self, config: EngineConfig) -> Set[FeatureName]:
        """Return the :class:`FeatureBundle` field names this method needs.

        Parameters
        ----------
        config:
            The active engine configuration (some adapters' requirements
            do not depend on config, but the signature is kept uniform).

        Returns
        -------
        Set[FeatureName]
            Subset of the names in :data:`_ALL_FEATURE_NAMES`.
        """
        raise NotImplementedError

    def build(self, config: EngineConfig) -> nn.Module:
        """Construct the wrapped aggregation module from ``config``.

        Parameters
        ----------
        config:
            Engine configuration supplying every constructor argument
            this method's underlying module needs.

        Returns
        -------
        nn.Module
            A freshly constructed, ``eval()``-ready aggregation module.
            The engine instantiates this exactly once per method and
            caches it for reuse (see ``CycleSpinningEngine._get_aggregator``).
        """
        raise NotImplementedError

    def call(
        self,
        module: nn.Module,
        outputs: Sequence[torch.Tensor],
        features: FeatureBundle,
        timestep: Optional[torch.Tensor],
        return_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Invoke the wrapped module's ``forward`` with the right argument order.

        Parameters
        ----------
        module:
            The module previously returned by :meth:`build` (and cached
            by the engine).
        outputs:
            Sequence of *N* per-shift prediction tensors, each
            ``[B, C, H, W]``.
        features:
            The full :class:`FeatureBundle`; only the fields named in
            :meth:`required_features` are read.
        timestep:
            Optional diffusion timestep tensor. Only
            :class:`UltimateCycleSpinning` currently consumes this;
            other adapters ignore it.
        return_weights:
            Whether to ask the wrapped module for its per-shift weights.

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]
            ``(fused, weights_or_None, extra_metadata)``.
        """
        raise NotImplementedError


class _LearnableAdapter(BaseAggregator):
    """Adapter for A26a :class:`LearnableCycleSpinning` (VERIFIED).

    ``LearnableCycleSpinning.__init__`` is
    ``(num_shifts, init_mode="uniform", temperature=1.0, manual_logits=None)``
    and ``forward`` is ``(outputs, return_weights=False)`` -- no
    per-shift conditioning features; a single global softmax logit
    vector shared across the batch. ``init_mode="uniform"`` is used
    here (zero-initialised logits, reproducing the original SAR-DDPM
    equal-weight average at step 0), matching every other method's
    near-uniform initialisation guarantee.
    """

    method_name = "learnable"

    def required_features(self, config: EngineConfig) -> Set[FeatureName]:
        return set()

    def build(self, config: EngineConfig) -> nn.Module:
        return LearnableCycleSpinning(
            num_shifts=config.num_shifts,
            init_mode="uniform",
            temperature=config.temperature,
        )

    def call(self, module, outputs, features, timestep, return_weights):
        result = module(outputs, return_weights=return_weights)
        if return_weights:
            fused, weights = result
        else:
            fused, weights = result, None
        return fused, weights, {}


class _AdaptiveAdapter(BaseAggregator):
    """Adapter for A26b :class:`AdaptiveCycleSpinning` (VERIFIED)."""

    method_name = "adaptive"

    def required_features(self, config: EngineConfig) -> Set[FeatureName]:
        return set()

    def build(self, config: EngineConfig) -> nn.Module:
        return AdaptiveCycleSpinning(
            num_shifts=config.num_shifts,
            channels=config.channels,
            hidden_dim=config.hidden_dim,
            temperature=config.temperature,
            pooling=config.pooling,
            eps=config.eps,
        )

    def call(self, module, outputs, features, timestep, return_weights):
        result = module(outputs, return_weights=return_weights)
        if return_weights:
            fused, weights = result
        else:
            fused, weights = result, None
        return fused, weights, {}


class _ConfidenceAdapter(BaseAggregator):
    """Adapter for A26c :class:`ConfidenceCycleSpinning` (VERIFIED)."""

    method_name = "confidence"

    def required_features(self, config: EngineConfig) -> Set[FeatureName]:
        return {"confidence_maps"}

    def build(self, config: EngineConfig) -> nn.Module:
        return ConfidenceCycleSpinning(
            num_shifts=config.num_shifts,
            channels=config.channels,
            hidden_dim=config.hidden_dim,
            temperature=config.temperature,
            pooling=config.pooling,
            eps=config.eps,
        )

    def call(self, module, outputs, features, timestep, return_weights):
        result = module(
            outputs, features.confidence_maps, return_weights=return_weights
        )
        if return_weights:
            fused, weights = result
        else:
            fused, weights = result, None
        return fused, weights, {}


class _WaveletConfidenceAdapter(BaseAggregator):
    """Adapter for A26d :class:`WaveletConfidenceCycleSpinning` (VERIFIED)."""

    method_name = "wavelet_confidence"

    def required_features(self, config: EngineConfig) -> Set[FeatureName]:
        return {"confidence_maps", "wavelet_features"}

    def build(self, config: EngineConfig) -> nn.Module:
        return WaveletConfidenceCycleSpinning(
            num_shifts=config.num_shifts,
            channels=config.channels,
            wavelet_channels=config.wavelet_channels,
            hidden_dim=config.hidden_dim,
            temperature=config.temperature,
            pooling=config.pooling,
            eps=config.eps,
        )

    def call(self, module, outputs, features, timestep, return_weights):
        result = module(
            outputs,
            features.confidence_maps,
            features.wavelet_features,
            return_weights=return_weights,
        )
        if return_weights:
            fused, weights = result
        else:
            fused, weights = result, None
        return fused, weights, {}


class _StructureWaveletAdapter(BaseAggregator):
    """Adapter for A26e :class:`StructureWaveletConfidenceCycleSpinning` (VERIFIED)."""

    method_name = "structure_wavelet"

    def required_features(self, config: EngineConfig) -> Set[FeatureName]:
        return {"confidence_maps", "wavelet_features", "structure_features"}

    def build(self, config: EngineConfig) -> nn.Module:
        return StructureWaveletConfidenceCycleSpinning(
            num_shifts=config.num_shifts,
            channels=config.channels,
            wavelet_channels=config.wavelet_channels,
            structure_channels=config.structure_channels,
            hidden_dim=config.hidden_dim,
            temperature=config.temperature,
            pooling=config.pooling,
            eps=config.eps,
        )

    def call(self, module, outputs, features, timestep, return_weights):
        result = module(
            outputs,
            features.confidence_maps,
            features.wavelet_features,
            features.structure_features,
            return_weights=return_weights,
        )
        if return_weights:
            fused, weights = result
        else:
            fused, weights = result, None
        return fused, weights, {}


class _TransformerAdapter(BaseAggregator):
    """Adapter for A26f :class:`TransformerCycleSpinning` (VERIFIED).

    Token dimension is ``channels + 1 + wavelet_channels +
    structure_channels`` (confirmed: no coordinate embedding and no
    learnable shift coordinates at this stage -- those are introduced
    later, in A26g/:class:`LearnableShiftCycleSpinning`). A CLS token
    is prepended and a learnable positional embedding is added before
    the stock ``nn.TransformerEncoder`` stack.
    """

    method_name = "transformer"

    def required_features(self, config: EngineConfig) -> Set[FeatureName]:
        return {"confidence_maps", "wavelet_features", "structure_features"}

    def build(self, config: EngineConfig) -> nn.Module:
        return TransformerCycleSpinning(
            num_shifts=config.num_shifts,
            channels=config.channels,
            wavelet_channels=config.wavelet_channels,
            structure_channels=config.structure_channels,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            temperature=config.temperature,
            pooling=config.pooling,
            eps=config.eps,
        )

    def call(self, module, outputs, features, timestep, return_weights):
        result = module(
            outputs,
            features.confidence_maps,
            features.wavelet_features,
            features.structure_features,
            return_weights=return_weights,
        )
        if return_weights:
            fused, weights = result
        else:
            fused, weights = result, None
        return fused, weights, {}


class _LearnableShiftAdapter(BaseAggregator):
    """Adapter for A26g :class:`LearnableShiftCycleSpinning` (VERIFIED)."""

    method_name = "learnable_shift"

    def required_features(self, config: EngineConfig) -> Set[FeatureName]:
        return {"confidence_maps", "wavelet_features", "structure_features"}

    def build(self, config: EngineConfig) -> nn.Module:
        return LearnableShiftCycleSpinning(
            num_shifts=config.num_shifts,
            channels=config.channels,
            wavelet_channels=config.wavelet_channels,
            structure_channels=config.structure_channels,
            coordinate_embed_dim=config.coordinate_embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            temperature=config.temperature,
            max_shift_radius=config.max_shift_radius,
            radius_lambda=config.radius_lambda,
            repulsion_lambda=config.repulsion_lambda,
            pooling=config.pooling,
            eps=config.eps,
        )

    def call(self, module, outputs, features, timestep, return_weights):
        result = module(
            outputs,
            features.confidence_maps,
            features.wavelet_features,
            features.structure_features,
            return_weights=return_weights,
        )
        if return_weights:
            fused, weights = result
        else:
            fused, weights = result, None
        return fused, weights, {}


class _UltimateAdapter(BaseAggregator):
    """Adapter for :class:`UltimateCycleSpinning` (VERIFIED)."""

    method_name = "ultimate"

    def required_features(self, config: EngineConfig) -> Set[FeatureName]:
        return {"confidence_maps", "wavelet_features", "structure_features"}

    def build(self, config: EngineConfig) -> nn.Module:
        kwargs: Dict[str, Any] = dict(
            num_shifts=config.num_shifts,
            channels=config.channels,
            wavelet_channels=config.wavelet_channels,
            structure_channels=config.structure_channels,
            coordinate_embed_dim=config.coordinate_embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            temperature=config.temperature,
            pooling=config.pooling,
            eps=config.eps,
        )
        # UltimateCycleSpinning has many extra hyperparameters
        # (num_layers, cross_level_heads, level_radii, MoE flags, ...)
        # that have no equivalent in the other seven methods. Anything
        # explicitly supplied via EngineConfig.ultimate_kwargs overrides
        # the shared defaults above and UltimateCycleSpinning's own
        # built-in defaults for everything else.
        kwargs.update(config.ultimate_kwargs)
        return UltimateCycleSpinning(**kwargs)

    def call(self, module, outputs, features, timestep, return_weights):
        result = module(
            outputs,
            features.confidence_maps,
            features.wavelet_features,
            features.structure_features,
            timestep=timestep,
            return_weights=return_weights,
        )
        # UltimateCycleSpinning.forward returns:
        #   return_weights=False -> fused
        #   return_weights=True  -> (fused, alpha)
        if return_weights:
            fused, weights = result
        else:
            fused, weights = result, None
        return fused, weights, {}


# =============================================================================
# Registry pattern
# =============================================================================

#: method name -> adapter class. No if/elif chain anywhere in the engine;
#: adding a ninth method means adding one BaseAggregator subclass and one
#: entry here.
_REGISTRY: Dict[str, Type[BaseAggregator]] = {
    "learnable": _LearnableAdapter,
    "adaptive": _AdaptiveAdapter,
    "confidence": _ConfidenceAdapter,
    "wavelet_confidence": _WaveletConfidenceAdapter,
    "structure_wavelet": _StructureWaveletAdapter,
    "transformer": _TransformerAdapter,
    "learnable_shift": _LearnableShiftAdapter,
    "ultimate": _UltimateAdapter,
}


def register_aggregator(name: str, adapter_cls: Type[BaseAggregator]) -> None:
    """Register a new aggregation method without editing this module.

    Lets downstream code add a ninth (tenth, ...) cycle-spinning method
    by writing a single :class:`BaseAggregator` subclass and calling
    this function once at import time -- no change to
    ``CycleSpinningEngine``, ``_REGISTRY``, or any existing adapter is
    required.

    Parameters
    ----------
    name:
        The method name to register (the value callers will pass as
        ``EngineConfig(method=name)``).
    adapter_cls:
        A :class:`BaseAggregator` subclass implementing
        ``required_features``, ``build``, and ``call``.

    Raises
    ------
    ValueError
        If ``name`` is already registered (overwriting a built-in or
        previously-registered method is almost always a mistake; build
        a differently-named adapter instead).
    TypeError
        If ``adapter_cls`` is not a subclass of :class:`BaseAggregator`.

    Examples
    --------
    >>> class _MyAdapter(BaseAggregator):
    ...     method_name = "my_method"
    ...     def required_features(self, config):
    ...         return set()
    ...     def build(self, config):
    ...         import torch.nn as nn
    ...         return nn.Identity()
    ...     def call(self, module, outputs, features, timestep, return_weights):
    ...         fused = sum(outputs) / len(outputs)
    ...         return fused, None, {}
    >>> register_aggregator("my_method", _MyAdapter)
    >>> "my_method" in CycleSpinningEngine.available_methods()
    True
    """
    if not (isinstance(adapter_cls, type) and issubclass(adapter_cls, BaseAggregator)):
        raise TypeError(
            f"adapter_cls must be a subclass of BaseAggregator, got {adapter_cls!r}."
        )
    if name in _REGISTRY:
        raise ValueError(
            f"Method {name!r} is already registered "
            f"({_REGISTRY[name].__name__}). Choose a different name."
        )
    _REGISTRY[name] = adapter_cls
    logger.info("CycleSpinningEngine: registered new method %r -> %s.", name, adapter_cls.__name__)


def _resolve_adapter(method: str) -> BaseAggregator:
    """Factory function: look up and instantiate the adapter for ``method``.

    Parameters
    ----------
    method:
        One of the keys in :data:`_REGISTRY`.

    Returns
    -------
    BaseAggregator
        A fresh adapter instance (adapters are stateless; only the
        wrapped ``nn.Module`` they build is cached, by the engine).

    Raises
    ------
    ValueError
        If ``method`` is not a registered key. The message lists every
        valid option.
    """
    try:
        adapter_cls = _REGISTRY[method]
    except KeyError as exc:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown cycle-spinning method {method!r}. "
            f"Valid methods are: {valid}."
        ) from exc
    return adapter_cls()


# =============================================================================
# Validation layer
# =============================================================================


def _validate_outputs(outputs: Sequence[torch.Tensor], config: EngineConfig) -> torch.Tensor:
    """Validate the raw per-shift outputs before any aggregator sees them.

    Parameters
    ----------
    outputs:
        Sequence of per-shift prediction tensors.
    config:
        Active engine configuration (supplies expected ``num_shifts``
        and ``channels``).

    Returns
    -------
    torch.Tensor
        ``outputs[0]``, returned for convenience.

    Raises
    ------
    ValueError
        If ``outputs`` is empty, has the wrong length, or any tensor has
        the wrong rank, channel count, or is inconsistent with the
        others in shape/dtype/device.
    """
    if len(outputs) == 0:
        raise ValueError("outputs must be a non-empty sequence of tensors.")
    if len(outputs) != config.num_shifts:
        raise ValueError(
            f"len(outputs)={len(outputs)} does not match "
            f"EngineConfig.num_shifts={config.num_shifts}. Either pass "
            f"exactly num_shifts tensors, or build a new EngineConfig "
            f"with the correct num_shifts."
        )
    ref = outputs[0]
    if ref.dim() != 4:
        raise ValueError(
            f"Each output tensor must be 4-D [B, C, H, W]; outputs[0] has "
            f"shape {tuple(ref.shape)} (ndim={ref.dim()})."
        )
    if ref.shape[1] != config.channels:
        raise ValueError(
            f"outputs[0] has {ref.shape[1]} channels but "
            f"EngineConfig.channels={config.channels}."
        )
    for idx, t in enumerate(outputs[1:], start=1):
        if t.shape != ref.shape:
            raise ValueError(
                f"outputs[{idx}].shape={tuple(t.shape)} does not match "
                f"outputs[0].shape={tuple(ref.shape)}."
            )
        if t.dtype != ref.dtype:
            raise ValueError(
                f"outputs[{idx}].dtype={t.dtype} does not match "
                f"outputs[0].dtype={ref.dtype}."
            )
        if t.device != ref.device:
            raise ValueError(
                f"outputs[{idx}].device={t.device} does not match "
                f"outputs[0].device={ref.device}."
            )
    return ref


def _validate_required_features(
    required: Set[FeatureName],
    features: FeatureBundle,
    config: EngineConfig,
    reference: torch.Tensor,
) -> None:
    """Ensure every feature the selected method needs is present and well-formed.

    Parameters
    ----------
    required:
        Feature names declared by the active adapter's
        ``required_features``.
    features:
        The caller-supplied :class:`FeatureBundle`.
    config:
        Active engine configuration.
    reference:
        ``outputs[0]``, used to cross-check batch size and shift count.

    Raises
    ------
    ValueError
        If a required feature is missing, has the wrong length, or any
        tensor within it has the wrong rank or batch size. Spatial
        dimensions and channel counts are intentionally *not*
        cross-checked here -- the wrapped modules themselves perform
        that more detailed validation (e.g. wavelet/structure tensors
        are explicitly allowed to be at a different spatial resolution
        than ``outputs``), so duplicating it here would only risk the
        two checks drifting out of sync.
    """
    batch_size = reference.shape[0]
    for name in sorted(required):
        seq = features.get(name)
        if seq is None:
            raise ValueError(
                f"Method requires '{name}' but FeatureBundle.{name} is None. "
                f"Required features for this method: {sorted(required)}."
            )
        if len(seq) != config.num_shifts:
            raise ValueError(
                f"len(FeatureBundle.{name})={len(seq)} does not match "
                f"num_shifts={config.num_shifts}."
            )
        for idx, t in enumerate(seq):
            if t.dim() != 4:
                raise ValueError(
                    f"FeatureBundle.{name}[{idx}] must be 4-D [B, C, H, W]; "
                    f"got ndim={t.dim()}."
                )
            if t.shape[0] != batch_size:
                raise ValueError(
                    f"FeatureBundle.{name}[{idx}] has batch size "
                    f"{t.shape[0]}, expected {batch_size} (from outputs[0])."
                )


# =============================================================================
# CycleSpinningEngine
# =============================================================================


class CycleSpinningEngine:
    """Unified interface over every registered cycle-spinning aggregation method.

    Construct once with an :class:`EngineConfig`, then call
    :meth:`fuse` per-batch from the inference loop. The underlying
    aggregation module (e.g. :class:`WaveletConfidenceCycleSpinning`)
    is built lazily on the first :meth:`fuse` call and cached for the
    lifetime of this engine instance -- it is never rebuilt on
    subsequent calls, and never copies feature tensors beyond what the
    wrapped module itself does internally.
=======
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
            bundle.predictions, conf, wav, struc, timestep=bundle.timestep
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
            bundle.predictions, conf, wav, struc, timestep=bundle.timestep
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
>>>>>>> cycle-engine

    Parameters
    ----------
    config:
<<<<<<< HEAD
        Engine configuration selecting the method and its
        hyperparameters.

    Attributes
    ----------
    config : EngineConfig
        The configuration this engine was constructed with.

    Examples
    --------
    >>> import torch
    >>> engine = CycleSpinningEngine(
    ...     config=EngineConfig(method="confidence", num_shifts=4, channels=1)
    ... )
    >>> outputs = [torch.randn(2, 1, 16, 16) for _ in range(4)]
    >>> confs = [torch.rand(2, 1, 16, 16) for _ in range(4)]
    >>> result = engine.fuse(
    ...     outputs=outputs,
    ...     features=FeatureBundle(confidence_maps=confs),
    ... )
    >>> result.fused.shape
    torch.Size([2, 1, 16, 16])

    Switching methods requires changing only the config::

        engine = CycleSpinningEngine(config=EngineConfig(method="ultimate", ...))
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config: EngineConfig = config
        self._adapter: BaseAggregator = _resolve_adapter(config.method)
        self._module: Optional[nn.Module] = None
        self._module_device: Optional[torch.device] = None

    @staticmethod
    def available_methods() -> Tuple[str, ...]:
        """Return every registered method name, sorted.

        Returns
        -------
        Tuple[str, ...]
            E.g. ``("adaptive", "confidence", "learnable", ...)``.
        """
        return tuple(sorted(_REGISTRY))

    def _get_aggregator(self, device: torch.device) -> nn.Module:
        """Return the cached aggregation module, building (once) if needed.

        Parameters
        ----------
        device:
            Device of the current call's ``outputs[0]``. Used only to
            place a freshly-built module; an already-built module is
            never re-moved or rebuilt, even if a later call arrives on
            a different device (mixed-device calls are not supported by
            a single engine instance -- construct a second
            ``CycleSpinningEngine`` for a second device).

        Returns
        -------
        nn.Module
            The (possibly newly constructed) aggregator, in ``eval()``
            mode.
        """
        if self._module is None:
            module = self._adapter.build(self.config)
            target_device = (
                torch.device(self.config.device)
                if self.config.device is not None
                else device
            )
            module = module.to(target_device)
            module.eval()
            self._module = module
            self._module_device = target_device
            logger.info(
                "CycleSpinningEngine: built '%s' aggregator on %s.",
                self.config.method,
                target_device,
            )
        return self._module

    def fuse(
        self,
        outputs: Sequence[torch.Tensor],
        features: Optional[FeatureBundle] = None,
        timestep: Optional[torch.Tensor] = None,
        return_weights: bool = True,
    ) -> EngineResult:
        """Aggregate ``outputs`` using the configured method.

        Parameters
        ----------
        outputs:
            Sequence of *N* per-shift prediction tensors
            (``N == self.config.num_shifts``), each ``[B, C, H, W]``,
            already inverse-shifted -- i.e. exactly what the existing
            row/col cycle-spin loop in
            ``guided_diffusion/test_util.py`` produces per shift before
            its current ``(1.0/N) * sample`` accumulation.
        features:
            Optional :class:`FeatureBundle` carrying whichever of
            confidence maps / wavelet features / structure features
            (etc.) the active method requires. Fields the method does
            not need may be left as ``None`` even if supplied elsewhere
            in the pipeline -- the engine only validates and forwards
            what :meth:`BaseAggregator.required_features` declares.
        timestep:
            Optional diffusion timestep, forwarded only to
            :class:`UltimateCycleSpinning` (ignored by every other
            method).
        return_weights:
            If ``True`` (default), ask the underlying method for its
            per-shift weights and include them in
            ``EngineResult.metadata["weights"]``.

        Returns
        -------
        EngineResult
            ``fused`` has shape ``[B, C, H, W]``. See
            :class:`EngineResult` for ``metadata`` contents.

        Raises
        ------
        ValueError
            If ``outputs`` is invalid (see ``_validate_outputs``), or if
            a feature required by the active method is missing or
            malformed (see ``_validate_required_features``).

        Examples
        --------
        >>> engine = CycleSpinningEngine(EngineConfig(method="ultimate"))
        >>> # ... build outputs / features for 9 shifts ...
        >>> # result = engine.fuse(outputs=outputs, features=features)
        >>> # pred_tensor = result.fused
        """
        if features is None:
            features = FeatureBundle()

        reference = _validate_outputs(outputs, self.config)
        required = self._adapter.required_features(self.config)
        _validate_required_features(required, features, self.config, reference)

        module = self._get_aggregator(reference.device)

        with torch.no_grad() if not module.training else _null_context():
            fused, weights, extra = self._adapter.call(
                module, outputs, features, timestep, return_weights
            )

        metadata: Dict[str, Any] = {
            "method": self.config.method,
            "weights": weights,
            "required_features": sorted(required),
            "num_shifts": self.config.num_shifts,
            "extra": extra,
        }
        return EngineResult(fused=fused, metadata=metadata)

    def train(self) -> "CycleSpinningEngine":
        """Switch the cached aggregator (if built) into training mode.

        Returns
        -------
        CycleSpinningEngine
            ``self``, for chaining.
        """
        if self._module is not None:
            self._module.train()
        return self

    def eval(self) -> "CycleSpinningEngine":
        """Switch the cached aggregator (if built) into eval mode.

        Returns
        -------
        CycleSpinningEngine
            ``self``, for chaining.
        """
        if self._module is not None:
            self._module.eval()
        return self

    @property
    def module(self) -> Optional[nn.Module]:
        """The cached aggregation module, or ``None`` before the first :meth:`fuse` call.

        Useful for accessing method-specific diagnostics (e.g.
        ``engine.module.save_statistics(...)``) without reaching into
        engine internals.
        """
        return self._module


class _null_context:
    """No-op context manager used when the cached module is in train mode.

    ``torch.no_grad()`` must not wrap a training-mode forward pass (it
    would silently disable gradient computation for parameters the
    caller intends to update), so :meth:`CycleSpinningEngine.fuse`
    selects between a real ``torch.no_grad()`` and this no-op based on
    ``module.training``.
    """

    def __enter__(self) -> "_null_context":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False
=======
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
        aggregator = self._get_or_build_aggregator(active_method, inv_preds, bundle)
        _check_shape_compatibility(aggregator, len(inv_preds), inv_preds[0].shape[1], active_method)

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

        t_feat_start = time.perf_counter()
        self._feature_manager.populate(
            bundle, aggregator.required_features, aggregator=aggregator
        )
        t_feat = time.perf_counter() - t_feat_start

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
>>>>>>> cycle-engine
