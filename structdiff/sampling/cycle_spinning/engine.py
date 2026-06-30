"""
structdiff/sampling/cycle_spinning/engine.py
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
"""

from __future__ import annotations

import logging
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

    Parameters
    ----------
    config:
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
