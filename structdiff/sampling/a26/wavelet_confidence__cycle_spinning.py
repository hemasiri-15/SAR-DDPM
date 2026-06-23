"""
structdiff/inference/wavelet_confidence_cycle_spinning.py
==========================================================
A26d: WaveletConfidenceCycleSpinning — wavelet-guided, confidence-adaptive
softmax aggregation of cycle-shifted diffusion outputs for SAR despeckling.

Background
----------
The original SAR-DDPM cycle-spinning implementation (see
``inference_sar.py`` / ``inference_sar_unet.py``) applies a denoiser
to *N* shifted copies of the input, inverse-shifts each result, and
averages them with **fixed equal weights**::

    pred_tensor += (1.0 / N) * sample

A26a (:class:`~structdiff.inference.learnable_cycle_spinning.LearnableCycleSpinning`)
replaced these fixed coefficients with a single set of *global*
trainable scalar logits, shared across every image in the dataset::

    w_i = softmax(a_i / τ)
    x̂   = Σ_i w_i • x_i

A26b (:class:`~structdiff.inference.adaptive_cycle_spinning.AdaptiveCycleSpinning`)
made the weights *image-adaptive* by predicting them from pooled
descriptors of the shifted predictions themselves::

    z_i = GAP(x_i)                              z_i ∈ R^{B × C}
    z   = concat(z_1, …, z_N)                   z   ∈ R^{B × (N•C)}
    a   = MLP(z)                                a   ∈ R^{B × N}
    w   = softmax(a / τ, dim=1)                 w   ∈ R^{B × N}

A26c (:class:`~structdiff.inference.confidence_cycle_spinning.ConfidenceCycleSpinning`)
extended A26b by additionally conditioning the weight predictor on
a per-shift **confidence map** ``σ_i``::

    z_i = GAP(x_i)                              z_i ∈ R^{B × C}
    c_i = GAP(σ_i)                              c_i ∈ R^{B × 1}
    d_i = concat(z_i, c_i)                      d_i ∈ R^{B × (C+1)}
    d   = concat(d_1, …, d_N)                   d   ∈ R^{B × N(C+1)}
    a   = MLP(d)                                a   ∈ R^{B × N}
    w   = softmax(a / τ, dim=1)                 w   ∈ R^{B × N}

Wavelet-guided aggregation (A26d)
-----------------------------------
A26d extends A26c by additionally conditioning the weight predictor on
a per-shift **wavelet tensor** ``W_i``, one multi-channel feature map
per shift derived from the DWT subbands (LL, LH, HL, HH) introduced in
A12 (:class:`~structdiff.conditioning.wavelet_encoder.WaveletEncoder`
and :mod:`structdiff.utils.wavelet_features`). The intuition is direct:

* The LL (approximation) subband captures low-frequency, speckle-suppressed
  content; shifts dominated by LL energy are often smoother and more
  faithful to the underlying scene reflectivity.
* The LH, HL, HH (detail) subbands capture horizontal, vertical, and
  diagonal high-frequency structure; shifts rich in these subbands
  carry more edge information but also more speckle energy.
* By conditioning the weight predictor on wavelet descriptors, the
  module can learn to modulate aggregation based on the directional
  texture and frequency profile of each shifted prediction — information
  that neither the raw image content (A26b) nor the scalar confidence
  map (A26c) alone can fully convey.

Formally, for shift ``i`` with prediction ``x_i ∈ R^{B×C×H×W}``,
confidence map ``σ_i ∈ R^{B×1×H×W}``, and wavelet tensor
``W_i ∈ R^{B×Cw×Hw×Ww}``::

    z_i = GAP(x_i)                              z_i ∈ R^{B × C}
    c_i = GAP(σ_i)                              c_i ∈ R^{B × 1}
    v_i = GAP(W_i)                              v_i ∈ R^{B × Cw}
    d_i = concat(z_i, c_i, v_i)                d_i ∈ R^{B × (C+1+Cw)}
    D   = concat(d_1, …, d_N)                   D   ∈ R^{B × N(C+1+Cw)}
    h   = GELU(Linear(D))                       h   ∈ R^{B × hidden_dim}
    a   = Linear(h)                              a   ∈ R^{B × N}
    w   = softmax(a / τ, dim=1)                 w   ∈ R^{B × N}
    x̂   = Σ_i w_i • x_i                         x̂   ∈ R^{B × C × H × W}

Note that the *fusion* itself still combines only the image content
``x_i`` (matching all prior A26 stages and the original SAR-DDPM
average); the confidence maps ``σ_i`` and wavelet tensors ``W_i`` are
used exclusively to help *predict* the weights, not to directly reweight
pixels.

The wavelet channel count ``Cw`` is **not hardcoded** — it is inferred
from the ``wavelet_channels`` constructor argument, allowing the module
to work with any DWT configuration (e.g. 4 subbands from a single-level
transform, 12 from a three-level transform, or any other variant from
A12).

Initialization guarantee
-------------------------
As in A26b and A26c, the final linear layer of the weight-predictor MLP
is initialised with very small weights (``std=1e-3``) and a zero bias.
Because the logits ``a`` are therefore approximately zero for any input
at step 0, the softmax output is approximately uniform::

    a ≈ 0  →  w_i ≈ 1/N  for all i

This means **at initialisation A26d approximately reproduces the
original SAR-DDPM equal-weight average** (and is approximately
equivalent to A26a, A26b, and A26c's uniform initialisation), regardless
of the content of the supplied confidence maps or wavelet tensors.

Checkpoint compatibility
-------------------------
``WaveletConfidenceCycleSpinning`` is a new module. Its
``weight_predictor`` has a different input dimensionality
(``N * (channels + 1 + wavelet_channels)``) than A26c's
(``N * (channels + 1)``), so **A26c checkpoints cannot be loaded
directly into A26d's ``weight_predictor`` weights** — the shapes do
not match. When migrating from an A26c run, load everything except
``weight_predictor.*`` with::

    model.load_state_dict(checkpoint, strict=False)

and allow ``weight_predictor.*`` to keep its freshly initialised,
near-zero values, which — per the initialisation guarantee above —
still preserves approximately equal-weight averaging behaviour at the
start of A26d training.

Wavelet motivation
-------------------
SAR images exhibit strong directional textures arising from azimuth
focusing, range resolution, and scene-specific scattering mechanisms
(urban layover, vegetation canopy, ocean swell). Cycle-shifted
denoiser outputs differ in how well they preserve these directional
structures, since each shift introduces a different spatial phase
offset relative to the underlying scattering geometry. Wavelet
subbands provide a computationally cheap, spatially localised
frequency decomposition that quantifies:

* **LL**: fraction of energy in the low-frequency approximation
  (higher → smoother, speckle-suppressed shift).
* **LH/HL**: energy in directional (horizontal/vertical) edge bands
  (higher → edge-rich, possibly better for urban or coastline scenes).
* **HH**: energy in diagonal detail bands (higher → more speckle or
  fine texture, wavelength-dependent).

By pooling wavelet descriptors per shift and feeding them to the MLP
alongside image content and confidence, the weight predictor can learn
a richer per-shift quality criterion than either A26b or A26c alone,
and can specialise its aggregation strategy to scene type.

Future roadmap
---------------
This module is the next link in the A26 series:

* **A26e** — Structure Tensor Fusion: further extend the conditioning
  signal to ``w = f(x, σ, W, S)`` by conditioning on structure-tensor
  eigenvalue features (A10/A11), so edge- and texture-rich regions can
  steer aggregation differently from smooth regions.
* **A26f** — Transformer Fusion: replace the MLP with cross-attention
  over the stack of *N* shifted outputs (and their confidence and wavelet
  maps), allowing each shift to attend to the others before weights are
  produced.
* **A26g** — Learnable Shift Coordinates: jointly learn the (row, col)
  shift grid rather than using a fixed uniform grid, feeding shift
  geometry as an additional conditioning signal.
* **A26h** — Hierarchical Cycle Spinning: nested coarse + fine shift
  pyramids with independent wavelet-confidence weight predictors per
  level.
* **A26i** — Full Adaptive Cycle-Spinning Transformer: integrates
  A26e-h into a unified transformer-based aggregation module that
  consumes confidence, wavelet, and structure-tensor features jointly.
* **A26j** — Bayesian Cycle Spinning: model per-image shift weights as
  a Dirichlet distribution and estimate uncertainty over the
  aggregation weights themselves, enabling principled confidence
  intervals over the fused prediction for journal-level uncertainty
  quantification in SAR despeckling.
* **A26k** — Meta-Learned Cycle Spinning: learn weight-prediction
  policies across datasets, allowing zero-shot transfer of the
  wavelet-guided aggregator to unseen SAR sensor configurations and
  domains.
* **A26l** — Reinforcement-Learned Shift Selection: use a policy
  network to decide adaptively which shifts to even evaluate for a
  given image, reducing inference cost while preserving despeckling
  quality.
* **A26m** — Timestep-Adaptive Fusion: extend the weight predictor to
  consume timestep-dependent features, exploiting the diffusion
  trajectory to modulate per-image aggregation strength at each noise
  level.
* **A26n** — Dynamic Shift Count: learn how many shifts are actually
  necessary for a given image, replacing the fixed *N* with an
  adaptive per-image shift budget.

The interface of this module (``forward``, ``get_weights``,
``entropy``, ``entropy_regularizer``, ``effective_num_shifts``,
``weight_variance``, ``max_weight_index``, ``min_weight_index``,
``uniform_weights``, ``kl_to_uniform``, ``js_to_uniform``,
``set_temperature``, ``freeze``, ``unfreeze``, ``is_frozen``,
``weight_statistics``, ``summary``, ``save_statistics``,
``extra_repr``) deliberately matches
:class:`~structdiff.inference.confidence_cycle_spinning.ConfidenceCycleSpinning`
exactly, with the sole functional difference being that every method
which predicts or consumes weights now also accepts ``wavelet_features``
alongside ``outputs`` and ``confidence_maps``. This keeps downstream
logging, training, and ablation code interchangeable across A26c and
A26d, and ensures future extensions (A26e-i) can inherit or compose
from this module without breaking changes.

References
----------
Coifman, R.R. & Donoho, D.L. (1995). Translation-Invariant
De-Noising. *Wavelets and Statistics*, Springer.

Mallat, S. (1999). *A Wavelet Tour of Signal Processing*. Academic Press.

Notes
-----
* All computation is performed in PyTorch; no NumPy, no CPU transfer,
  no in-place operations, full autograd support.
* The module is device-agnostic: all submodules move with
  ``model.to(device)``.
* Weights are predicted per batch element, so two images in the same
  batch may receive entirely different aggregation strategies, each
  informed by its own per-shift confidence maps and wavelet descriptors.
* The wavelet channel count ``Cw`` is inferred from the
  ``wavelet_channels`` constructor argument and is not hardcoded,
  allowing compatibility with any DWT configuration produced by A12.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Supported spatial pooling modes for descriptor extraction.
_VALID_POOLING_MODES: frozenset = frozenset({"avg", "max"})

#: Standard deviation used to initialise the final linear layer of the
#: weight-predictor MLP. Kept small so that the predicted logits start
#: near zero, producing a near-uniform softmax distribution and
#: maximising compatibility with the SAR-DDPM heuristic baseline and
#: the A26a / A26b / A26c uniform initialisation.
_FINAL_LAYER_INIT_STD: float = 1e-3

#: Epsilon added inside the entropy logarithm for numerical stability.
#: Must satisfy _LOG_EPS << 1/N for any practical N.
_LOG_EPS: float = 1e-8

#: Number of channels expected in every confidence map (one scalar
#: confidence value per spatial location).
_CONFIDENCE_CHANNELS: int = 1


# ---------------------------------------------------------------------------
# WaveletConfidenceCycleSpinning
# ---------------------------------------------------------------------------


class WaveletConfidenceCycleSpinning(nn.Module):
    """Wavelet-guided, confidence-adaptive softmax aggregation of cycle-shifted
    diffusion outputs.

    Extends A26c
    (:class:`~structdiff.inference.confidence_cycle_spinning.ConfidenceCycleSpinning`)
    by additionally conditioning the per-image weight predictor on a
    per-shift wavelet tensor, in addition to the shifted prediction
    content and confidence maps. Each of the *N* shifted prediction
    tensors is spatially pooled to a ``[B, C]`` descriptor, each of the
    *N* accompanying confidence maps is spatially pooled to a ``[B, 1]``
    descriptor, and each of the *N* wavelet tensors is spatially pooled
    to a ``[B, Cw]`` descriptor; all three are concatenated per shift to
    ``[B, C+1+Cw]``, all *N* per-shift descriptors are concatenated to
    ``[B, N•(C+1+Cw)]``, and a two-layer MLP maps this to per-shift
    logits ``[B, N]``. A temperature-scaled softmax over the shift
    dimension then yields per-image aggregation weights that sum to 1
    across shifts for every batch element.

    At construction, the final linear layer of the MLP is initialised
    with very small weights and zero bias, so the predicted logits
    start near zero and the softmax output starts near-uniform —
    closely matching the original SAR-DDPM equal-weight average and
    the A26a / A26b / A26c uniform initialisation, regardless of the
    content of the supplied confidence maps or wavelet tensors.

    Parameters
    ----------
    num_shifts:
        Total number of cycle-spin shifts *N*. Must be a positive
        integer. Corresponds to the number of (row, col) shift pairs
        in the nested loop of the existing SAR-DDPM inference code.
    channels:
        Number of channels *C* in each shifted prediction tensor.
        Must be a positive integer. Used to size part of the MLP input
        dimension.
    wavelet_channels:
        Number of channels *Cw* in each wavelet tensor. Must be a
        positive integer. Inferred from the A12 DWT configuration
        (e.g. 4 for a single-level transform with LL, LH, HL, HH
        subbands). NOT hardcoded; any value > 0 is accepted.
    hidden_dim:
        Width of the MLP's hidden layer. Must be a positive integer.
        Default 128.
    temperature:
        Softmax temperature τ > 0 applied to the predicted logits
        before the softmax. Lower values sharpen the per-image weight
        distribution; higher values flatten it. Default 1.0.
    pooling:
        Spatial pooling mode used to compute the per-shift image
        descriptors ``z_i = pool(x_i)``, confidence descriptors
        ``c_i = pool(σ_i)``, and wavelet descriptors
        ``v_i = pool(W_i)``. One of:

        ``"avg"`` (default)
            Global average pooling (``nn.AdaptiveAvgPool2d(1)``).
        ``"max"``
            Global max pooling (``nn.AdaptiveMaxPool2d(1)``).

        Raises ``ValueError`` for any other string.
    eps:
        Small positive constant used for numerical stability in
        entropy-style computations. Must be strictly positive.
        Default 1e-8.

    Attributes
    ----------
    num_shifts : int
        Number of cycle-spin shifts registered at construction.
    channels : int
        Number of channels expected in each shifted prediction tensor.
    wavelet_channels : int
        Number of channels expected in each wavelet tensor.
    hidden_dim : int
        Width of the weight-predictor MLP's hidden layer.
    temperature : float
        Softmax temperature used when converting logits to weights.
    pooling : str
        Spatial pooling mode used for descriptor extraction.
    eps : float
        Numerical-stability constant used in entropy computations.
    pool : nn.Module
        The instantiated pooling layer (``AdaptiveAvgPool2d(1)`` or
        ``AdaptiveMaxPool2d(1)``), shared by the image, confidence,
        and wavelet descriptor branches.
    weight_predictor : nn.Sequential
        The two-layer MLP (``Linear`` → ``GELU`` → ``Linear``) that
        maps concatenated pooled image, confidence, and wavelet
        descriptors to per-shift logits.

    Examples
    --------
    >>> import torch
    >>> from structdiff.inference.wavelet_confidence_cycle_spinning import (
    ...     WaveletConfidenceCycleSpinning,
    ... )
    >>> wccs = WaveletConfidenceCycleSpinning(
    ...     num_shifts=9, channels=1, wavelet_channels=4
    ... )
    >>> wccs.num_shifts
    9
    >>> outputs = [torch.randn(2, 1, 64, 64) for _ in range(9)]
    >>> confidence_maps = [torch.rand(2, 1, 64, 64) for _ in range(9)]
    >>> wavelet_features = [torch.randn(2, 4, 32, 32) for _ in range(9)]
    >>> fused = wccs(outputs, confidence_maps, wavelet_features)
    >>> fused.shape
    torch.Size([2, 1, 64, 64])

    >>> # return_weights=True yields per-image weights
    >>> fused, weights = wccs(
    ...     outputs, confidence_maps, wavelet_features, return_weights=True
    ... )
    >>> weights.shape
    torch.Size([2, 9])
    >>> bool(torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-5))
    True
    """

    def __init__(
        self,
        num_shifts: int,
        channels: int,
        wavelet_channels: int,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        pooling: str = "avg",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        # ----------------------------------------------------------------
        # Input validation
        # ----------------------------------------------------------------
        if not isinstance(num_shifts, int) or num_shifts <= 0:
            raise ValueError(
                f"num_shifts must be a positive integer, got {num_shifts!r}."
            )
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError(
                f"channels must be a positive integer, got {channels!r}."
            )
        if not isinstance(wavelet_channels, int) or wavelet_channels <= 0:
            raise ValueError(
                f"wavelet_channels must be a positive integer, "
                f"got {wavelet_channels!r}."
            )
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be a positive integer, got {hidden_dim!r}."
            )
        if temperature <= 0.0:
            raise ValueError(
                f"temperature must be strictly positive, got {temperature}."
            )
        if pooling not in _VALID_POOLING_MODES:
            raise ValueError(
                f"pooling must be one of {sorted(_VALID_POOLING_MODES)}, "
                f"got {pooling!r}."
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be strictly positive, got {eps}.")

        # ----------------------------------------------------------------
        # Attributes
        # ----------------------------------------------------------------
        self.num_shifts: int = num_shifts
        self.channels: int = channels
        self.wavelet_channels: int = wavelet_channels
        self.hidden_dim: int = hidden_dim
        self.temperature: float = temperature
        self.pooling: str = pooling
        self.eps: float = eps

        # ----------------------------------------------------------------
        # Spatial pooling layer
        #
        # Shared by all three branches:
        #   - Reduces each shifted prediction [B, C, H, W] to [B, C, 1, 1].
        #   - Reduces each confidence map [B, 1, H, W] to [B, 1, 1, 1].
        #   - Reduces each wavelet tensor [B, Cw, Hw, Ww] to [B, Cw, 1, 1].
        # ----------------------------------------------------------------
        self.pool: nn.Module
        if pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:  # pooling == "max"
            self.pool = nn.AdaptiveMaxPool2d(1)

        # ----------------------------------------------------------------
        # Weight-predictor MLP
        #
        # Input : [B, num_shifts * (channels + 1 + wavelet_channels)]
        #          (concatenated image + confidence + wavelet descriptors)
        # Hidden: [B, hidden_dim]
        # Output: [B, num_shifts]              (per-shift logits)
        # ----------------------------------------------------------------
        mlp_input_dim: int = num_shifts * (
            channels + _CONFIDENCE_CHANNELS + wavelet_channels
        )
        self.weight_predictor: nn.Sequential = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_shifts),
        )

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        """(Re-)initialise the weight-predictor MLP.

        The first linear layer uses Kaiming (He) initialisation for
        its weight matrix, which is appropriate given the following
        GELU nonlinearity, and a zero bias.

        The second (final) linear layer — whose output directly
        becomes the softmax logits — is initialised with weights
        drawn from ``Normal(0, _FINAL_LAYER_INIT_STD)`` and a zero
        bias. Because ``_FINAL_LAYER_INIT_STD`` is small (1e-3), the
        predicted logits ``a`` start very close to zero for any input
        (image content, confidence map, or wavelet tensor), so::

            w = softmax(a / τ) ≈ [1/N, …, 1/N]

        reproducing the original SAR-DDPM equal-weight average and the
        A26a / A26b / A26c uniform initialisation at step 0.

        Examples
        --------
        >>> wccs = WaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=2, wavelet_channels=4
        ... )
        >>> wccs.reset_parameters()  # re-draw initial weights
        >>> first_linear = wccs.weight_predictor[0]
        >>> isinstance(first_linear, torch.nn.Linear)
        True
        """
        first_linear: nn.Linear = self.weight_predictor[0]  # type: ignore[assignment]
        final_linear: nn.Linear = self.weight_predictor[2]  # type: ignore[assignment]

        # First layer: standard Kaiming init for a layer feeding into GELU.
        nn.init.kaiming_normal_(
            first_linear.weight, nonlinearity="linear"
        )
        nn.init.zeros_(first_linear.bias)

        # Final layer: near-zero init so initial logits ≈ 0 and the
        # softmax output ≈ uniform, matching the SAR-DDPM / A26a / A26b /
        # A26c baseline regardless of confidence or wavelet content.
        nn.init.normal_(
            final_linear.weight, mean=0.0, std=_FINAL_LAYER_INIT_STD
        )
        nn.init.zeros_(final_linear.bias)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_outputs(self, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """Validate the ``outputs`` sequence and return the reference tensor.

        Checks performed (in order):

        * ``outputs`` is non-empty.
        * ``len(outputs) == self.num_shifts``.
        * Every tensor is 4-dimensional ``[B, C, H, W]``.
        * Every tensor's channel dimension equals ``self.channels``.
        * Every tensor shares the same shape, dtype, and device as the
          first tensor in the sequence.

        Parameters
        ----------
        outputs:
            Candidate sequence of cycle-shifted prediction tensors.

        Returns
        -------
        torch.Tensor
            The first tensor in ``outputs`` (the reference tensor),
            returned for convenience so callers do not need to index
            into ``outputs`` again.

        Raises
        ------
        ValueError
            If any of the checks above fail, with a descriptive
            message identifying which tensor and which property
            triggered the failure.
        """
        if len(outputs) == 0:
            raise ValueError(
                "outputs must be a non-empty sequence of tensors, got length 0."
            )
        if len(outputs) != self.num_shifts:
            raise ValueError(
                f"len(outputs) must equal num_shifts={self.num_shifts}, "
                f"got {len(outputs)}."
            )

        reference: torch.Tensor = outputs[0]

        if reference.ndim != 4:
            raise ValueError(
                f"Each output tensor must be 4-dimensional [B, C, H, W]; "
                f"outputs[0] has shape {reference.shape} (ndim={reference.ndim})."
            )
        if reference.shape[1] != self.channels:
            raise ValueError(
                f"Each output tensor must have channels={self.channels}; "
                f"outputs[0] has {reference.shape[1]} channels."
            )

        ref_shape: torch.Size = reference.shape
        ref_dtype: torch.dtype = reference.dtype
        ref_device: torch.device = reference.device

        for idx, tensor in enumerate(outputs[1:], start=1):
            if tensor.ndim != 4:
                raise ValueError(
                    f"Each output tensor must be 4-dimensional [B, C, H, W]; "
                    f"outputs[{idx}] has shape {tensor.shape} "
                    f"(ndim={tensor.ndim})."
                )
            if tensor.shape != ref_shape:
                raise ValueError(
                    f"All output tensors must have the same shape; "
                    f"outputs[0].shape={ref_shape} but "
                    f"outputs[{idx}].shape={tensor.shape}."
                )
            if tensor.dtype != ref_dtype:
                raise ValueError(
                    f"All output tensors must have the same dtype; "
                    f"outputs[0].dtype={ref_dtype} but "
                    f"outputs[{idx}].dtype={tensor.dtype}."
                )
            if tensor.device != ref_device:
                raise ValueError(
                    f"All output tensors must reside on the same device; "
                    f"outputs[0].device={ref_device} but "
                    f"outputs[{idx}].device={tensor.device}."
                )

        return reference

    def _validate_confidence_maps(
        self,
        confidence_maps: Sequence[torch.Tensor],
        reference: torch.Tensor,
    ) -> None:
        """Validate the ``confidence_maps`` sequence against the reference output.

        Checks performed (in order):

        * ``confidence_maps`` is non-empty.
        * ``len(confidence_maps) == self.num_shifts``.
        * Every confidence map is 4-dimensional ``[B, 1, H, W]``.
        * Every confidence map has exactly one channel.
        * Every confidence map's batch size, spatial dimensions,
          dtype, and device match ``reference``.

        Parameters
        ----------
        confidence_maps:
            Candidate sequence of per-shift confidence maps.
        reference:
            The validated reference tensor from ``outputs`` (typically
            the return value of ``_validate_outputs``), used as the
            ground truth for batch size, spatial dimensions, dtype,
            and device.

        Raises
        ------
        ValueError
            If any of the checks above fail, with a descriptive
            message identifying which confidence map and which
            property triggered the failure.
        """
        if len(confidence_maps) == 0:
            raise ValueError(
                "confidence_maps must be a non-empty sequence of tensors, "
                "got length 0."
            )
        if len(confidence_maps) != self.num_shifts:
            raise ValueError(
                f"len(confidence_maps) must equal num_shifts={self.num_shifts}, "
                f"got {len(confidence_maps)}."
            )

        ref_batch_size: int = reference.shape[0]
        ref_height: int = reference.shape[2]
        ref_width: int = reference.shape[3]
        ref_dtype: torch.dtype = reference.dtype
        ref_device: torch.device = reference.device

        for idx, conf in enumerate(confidence_maps):
            if conf.ndim != 4:
                raise ValueError(
                    f"Each confidence map must be 4-dimensional "
                    f"[B, 1, H, W]; confidence_maps[{idx}] has shape "
                    f"{conf.shape} (ndim={conf.ndim})."
                )
            if conf.shape[1] != _CONFIDENCE_CHANNELS:
                raise ValueError(
                    f"Each confidence map must have exactly "
                    f"{_CONFIDENCE_CHANNELS} channel; "
                    f"confidence_maps[{idx}] has {conf.shape[1]} channels."
                )
            if conf.shape[0] != ref_batch_size:
                raise ValueError(
                    f"Each confidence map must have the same batch size as "
                    f"outputs; outputs[0].shape[0]={ref_batch_size} but "
                    f"confidence_maps[{idx}].shape[0]={conf.shape[0]}."
                )
            if conf.shape[2] != ref_height or conf.shape[3] != ref_width:
                raise ValueError(
                    f"Each confidence map must have the same spatial "
                    f"dimensions as outputs; outputs[0].shape[2:]="
                    f"({ref_height}, {ref_width}) but "
                    f"confidence_maps[{idx}].shape[2:]="
                    f"({conf.shape[2]}, {conf.shape[3]})."
                )
            if conf.dtype != ref_dtype:
                raise ValueError(
                    f"Each confidence map must have the same dtype as "
                    f"outputs; outputs[0].dtype={ref_dtype} but "
                    f"confidence_maps[{idx}].dtype={conf.dtype}."
                )
            if conf.device != ref_device:
                raise ValueError(
                    f"Each confidence map must reside on the same device as "
                    f"outputs; outputs[0].device={ref_device} but "
                    f"confidence_maps[{idx}].device={conf.device}."
                )

    def _validate_wavelet_features(
        self,
        wavelet_features: Sequence[torch.Tensor],
        reference: torch.Tensor,
    ) -> None:
        """Validate the ``wavelet_features`` sequence against the reference output.

        Checks performed (in order):

        * ``wavelet_features`` is non-empty.
        * ``len(wavelet_features) == self.num_shifts``.
        * Every wavelet tensor is 4-dimensional ``[B, Cw, Hw, Ww]``.
        * Every wavelet tensor has exactly ``self.wavelet_channels`` channels.
        * Every wavelet tensor's batch size, dtype, and device match
          ``reference`` (spatial dimensions may differ from ``outputs``
          since wavelet tensors are typically half-resolution).

        Parameters
        ----------
        wavelet_features:
            Candidate sequence of per-shift wavelet tensors, each
            ``[B, Cw, Hw, Ww]``. The spatial dimensions ``(Hw, Ww)``
            need not match those of ``outputs``; they are typically
            half the resolution of the SAR image (one DWT level) or
            smaller (multi-level DWT). What matters is that all wavelet
            tensors in the sequence share the same shape.
        reference:
            The validated reference tensor from ``outputs`` (typically
            the return value of ``_validate_outputs``), used as the
            ground truth for batch size, dtype, and device.

        Raises
        ------
        ValueError
            If any of the checks above fail, with a descriptive
            message identifying which wavelet tensor and which
            property triggered the failure.
        """
        if len(wavelet_features) == 0:
            raise ValueError(
                "wavelet_features must be a non-empty sequence of tensors, "
                "got length 0."
            )
        if len(wavelet_features) != self.num_shifts:
            raise ValueError(
                f"len(wavelet_features) must equal num_shifts={self.num_shifts}, "
                f"got {len(wavelet_features)}."
            )

        ref_batch_size: int = reference.shape[0]
        ref_dtype: torch.dtype = reference.dtype
        ref_device: torch.device = reference.device

        # Grab the first wavelet tensor as a shape reference for the sequence.
        first_wav: torch.Tensor = wavelet_features[0]

        if first_wav.ndim != 4:
            raise ValueError(
                f"Each wavelet tensor must be 4-dimensional [B, Cw, Hw, Ww]; "
                f"wavelet_features[0] has shape {first_wav.shape} "
                f"(ndim={first_wav.ndim})."
            )
        if first_wav.shape[1] != self.wavelet_channels:
            raise ValueError(
                f"Each wavelet tensor must have wavelet_channels="
                f"{self.wavelet_channels}; wavelet_features[0] has "
                f"{first_wav.shape[1]} channels."
            )
        if first_wav.shape[0] != ref_batch_size:
            raise ValueError(
                f"Each wavelet tensor must have the same batch size as "
                f"outputs; outputs[0].shape[0]={ref_batch_size} but "
                f"wavelet_features[0].shape[0]={first_wav.shape[0]}."
            )
        if first_wav.dtype != ref_dtype:
            raise ValueError(
                f"Each wavelet tensor must have the same dtype as outputs; "
                f"outputs[0].dtype={ref_dtype} but "
                f"wavelet_features[0].dtype={first_wav.dtype}."
            )
        if first_wav.device != ref_device:
            raise ValueError(
                f"Each wavelet tensor must reside on the same device as "
                f"outputs; outputs[0].device={ref_device} but "
                f"wavelet_features[0].device={first_wav.device}."
            )

        wav_ref_shape: torch.Size = first_wav.shape

        for idx, wav in enumerate(wavelet_features[1:], start=1):
            if wav.ndim != 4:
                raise ValueError(
                    f"Each wavelet tensor must be 4-dimensional [B, Cw, Hw, Ww]; "
                    f"wavelet_features[{idx}] has shape {wav.shape} "
                    f"(ndim={wav.ndim})."
                )
            if wav.shape != wav_ref_shape:
                raise ValueError(
                    f"All wavelet tensors must have the same shape; "
                    f"wavelet_features[0].shape={wav_ref_shape} but "
                    f"wavelet_features[{idx}].shape={wav.shape}."
                )
            if wav.dtype != ref_dtype:
                raise ValueError(
                    f"All wavelet tensors must have the same dtype as outputs; "
                    f"outputs[0].dtype={ref_dtype} but "
                    f"wavelet_features[{idx}].dtype={wav.dtype}."
                )
            if wav.device != ref_device:
                raise ValueError(
                    f"All wavelet tensors must reside on the same device as "
                    f"outputs; outputs[0].device={ref_device} but "
                    f"wavelet_features[{idx}].device={wav.device}."
                )

    # ------------------------------------------------------------------
    # Descriptor extraction
    # ------------------------------------------------------------------

    def _extract_descriptors(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute and concatenate pooled per-shift image, confidence, and wavelet descriptors.

        For each shift ``i``:

        * The image tensor ``x_i`` of shape ``[B, C, H, W]`` is spatially
          pooled to ``[B, C, 1, 1]`` and flattened to ``[B, C]``.
        * The confidence map ``σ_i`` of shape ``[B, 1, H, W]`` is spatially
          pooled to ``[B, 1, 1, 1]`` and flattened to ``[B, 1]``.
        * The wavelet tensor ``W_i`` of shape ``[B, Cw, Hw, Ww]`` is
          spatially pooled to ``[B, Cw, 1, 1]`` and flattened to ``[B, Cw]``.

        The three per-shift descriptors are concatenated to ``[B, C+1+Cw]``.
        The *N* per-shift concatenated descriptors are then concatenated
        along the channel dimension to produce a single descriptor of shape
        ``[B, N * (C + 1 + Cw)]`` suitable as input to ``weight_predictor``.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``. Assumed to
            have already been validated by ``_validate_outputs``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``. Assumed to
            have already been validated by ``_validate_confidence_maps``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``. Assumed to
            have already been validated by ``_validate_wavelet_features``.

        Returns
        -------
        torch.Tensor
            Concatenated descriptor of shape ``[B, N * (C + 1 + Cw)]``.
        """
        descriptors: List[torch.Tensor] = []
        for tensor, conf, wav in zip(outputs, confidence_maps, wavelet_features):
            pooled_image: torch.Tensor = self.pool(tensor)   # [B, C, 1, 1]
            pooled_conf: torch.Tensor = self.pool(conf)      # [B, 1, 1, 1]
            pooled_wav: torch.Tensor = self.pool(wav)        # [B, Cw, 1, 1]

            batch_size: int = pooled_image.shape[0]
            image_descriptor: torch.Tensor = pooled_image.reshape(
                batch_size, -1
            )  # [B, C]
            conf_descriptor: torch.Tensor = pooled_conf.reshape(
                batch_size, -1
            )  # [B, 1]
            wav_descriptor: torch.Tensor = pooled_wav.reshape(
                batch_size, -1
            )  # [B, Cw]

            descriptors.append(
                torch.cat(
                    [image_descriptor, conf_descriptor, wav_descriptor], dim=1
                )
            )  # [B, C+1+Cw]

        return torch.cat(descriptors, dim=1)  # [B, N * (C+1+Cw)]

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def get_weights(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Predict per-image softmax aggregation weights from outputs, confidence, and wavelets.

        Computes pooled image, confidence, and wavelet descriptors for
        every shift, concatenates them per shift and then across shifts,
        passes the result through ``weight_predictor`` to obtain
        per-shift logits, and applies a temperature-scaled softmax over
        the shift dimension.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``, giving a
            per-pixel confidence/reliability signal for the
            corresponding entry of ``outputs``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``, giving
            per-shift DWT subband features (LL, LH, HL, HH, or any
            other wavelet decomposition produced by A12).

        Returns
        -------
        torch.Tensor
            Shape ``[B, num_shifts]``. Every row sums to 1.0 and every
            entry is strictly positive. Retains the autograd graph;
            never detached.

        Raises
        ------
        ValueError
            If ``self.temperature`` is not strictly positive, or if any
            of the input sequences fail validation.

        Notes
        -----
        The temperature τ controls the sharpness of each row of the
        distribution::

            w_i = exp(a_i / τ) / Σ_j exp(a_j / τ)

        * τ → 0⁺ : winner-takes-all per image.
        * τ = 1.0 : standard softmax.
        * τ → ∞ : uniform distribution (1/N) per image.

        Examples
        --------
        >>> wccs = WaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4
        ... )
        >>> outs = [torch.randn(3, 1, 16, 16) for _ in range(4)]
        >>> confs = [torch.rand(3, 1, 16, 16) for _ in range(4)]
        >>> wavs = [torch.randn(3, 4, 8, 8) for _ in range(4)]
        >>> w = wccs.get_weights(outs, confs, wavs)
        >>> w.shape
        torch.Size([3, 4])
        >>> bool(torch.allclose(w.sum(dim=1), torch.ones(3), atol=1e-5))
        True
        """
        if self.temperature <= 0.0:
            raise ValueError(
                f"temperature must be strictly positive, got {self.temperature}."
            )
        reference: torch.Tensor = self._validate_outputs(outputs)
        self._validate_confidence_maps(confidence_maps, reference)
        self._validate_wavelet_features(wavelet_features, reference)

        descriptors: torch.Tensor = self._extract_descriptors(
            outputs, confidence_maps, wavelet_features
        )  # [B, N*(C+1+Cw)]

        # The MLP's Linear layers are fp32 by default; cast descriptors
        # to match so mixed-precision (fp16) callers do not error out.
        predictor_dtype: torch.dtype = self.weight_predictor[0].weight.dtype
        if descriptors.dtype != predictor_dtype:
            descriptors = descriptors.to(predictor_dtype)

        logits: torch.Tensor = self.weight_predictor(descriptors)  # [B, N]
        weights: torch.Tensor = F.softmax(logits / self.temperature, dim=1)
        return weights

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Aggregate cycle-shifted diffusion outputs with wavelet-confidence-guided weights.

        Parameters
        ----------
        outputs:
            A sequence of *N* tensors, one per cycle-spin shift. Each
            tensor must have shape ``[B, C, H, W]``.

            All tensors must satisfy:

            * identical shape,
            * identical dtype,
            * identical device,
            * channel dimension equal to ``self.channels``.

            Corresponds to the per-shift ``sample`` tensors produced
            inside the row/col loop in the existing SAR-DDPM inference
            code, *after* the inverse shift has been applied.
        confidence_maps:
            A sequence of *N* tensors, one per cycle-spin shift, each
            ``[B, 1, H, W]``, providing a per-pixel confidence signal
            for the corresponding entry of ``outputs``. Must match
            ``outputs`` in batch size, spatial dimensions, dtype, and
            device.
        wavelet_features:
            A sequence of *N* tensors, one per cycle-spin shift, each
            ``[B, Cw, Hw, Ww]``, providing DWT subband features
            (e.g. LL, LH, HL, HH from A12) for the corresponding
            shifted prediction. Must match ``outputs`` in batch size,
            dtype, and device. Spatial dimensions ``(Hw, Ww)`` may
            differ from ``(H, W)`` (wavelet tensors are typically
            half-resolution or smaller).
        return_weights:
            If ``False`` (default), return only the fused tensor.
            If ``True``, return ``(fused, weights)`` where ``weights``
            has shape ``[B, num_shifts]``.

        Returns
        -------
        torch.Tensor or tuple of (torch.Tensor, torch.Tensor)
            * ``return_weights=False``: fused tensor, shape
              ``[B, C, H, W]``.
            * ``return_weights=True``: ``(fused, weights)`` where
              ``fused`` has shape ``[B, C, H, W]`` and ``weights`` has
              shape ``[B, num_shifts]``.

        Raises
        ------
        ValueError
            If ``outputs`` is empty, or ``len(outputs) != self.num_shifts``.
        ValueError
            If any output tensor is not 4-dimensional, or its channel
            dimension does not equal ``self.channels``.
        ValueError
            If output tensors have inconsistent shapes, dtypes, or devices.
        ValueError
            If ``confidence_maps`` is empty, or
            ``len(confidence_maps) != self.num_shifts``.
        ValueError
            If any confidence map is not 4-dimensional, does not have
            exactly one channel, or its batch size, spatial dimensions,
            dtype, or device does not match ``outputs``.
        ValueError
            If ``wavelet_features`` is empty, or
            ``len(wavelet_features) != self.num_shifts``.
        ValueError
            If any wavelet tensor is not 4-dimensional, does not have
            ``self.wavelet_channels`` channels, or its batch size,
            dtype, or device does not match ``outputs``.

        Notes
        -----
        The aggregation is::

            z_i = GAP_or_GMP(x_i)                        [B, C]
            c_i = GAP_or_GMP(σ_i)                         [B, 1]
            v_i = GAP_or_GMP(W_i)                         [B, Cw]
            d_i = concat(z_i, c_i, v_i)                   [B, C+1+Cw]
            D   = concat(d_1, …, d_N)                      [B, N*(C+1+Cw)]
            a   = weight_predictor(D)                      [B, N]
            w   = softmax(a / temperature, dim=1)          [B, N]
            x̂   = Σ_i w_i • x_i                            [B, C, H, W]

        Only the image content ``x_i`` is summed into the fused output;
        the confidence maps ``σ_i`` and wavelet tensors ``W_i`` are
        consumed solely by the weight predictor.

        Weights are broadcast as ``[B, N, 1, 1, 1]`` over a
        ``[B, N, C, H, W]``-permuted stack of the shifted outputs, then
        summed along the shift dimension to produce ``[B, C, H, W]``.
        No in-place operations are used; the autograd graph is preserved
        throughout.

        Because the final layer of ``weight_predictor`` is initialised
        with very small weights, at construction time the predicted
        logits are approximately zero for any input, so ``w_i ≈ 1/N``
        and the output is approximately identical to the original
        SAR-DDPM equal-weight average, regardless of confidence map or
        wavelet tensor content.

        Examples
        --------
        >>> import torch
        >>> from structdiff.inference.wavelet_confidence_cycle_spinning import (
        ...     WaveletConfidenceCycleSpinning,
        ... )
        >>> wccs = WaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4
        ... )
        >>> outputs = [torch.ones(2, 1, 8, 8) * (i + 1.0) for i in range(4)]
        >>> confidence_maps = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavelet_features = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> fused = wccs(outputs, confidence_maps, wavelet_features)
        >>> fused.shape
        torch.Size([2, 1, 8, 8])

        >>> # return_weights=True
        >>> fused, w = wccs(
        ...     outputs, confidence_maps, wavelet_features, return_weights=True
        ... )
        >>> w.shape
        torch.Size([2, 4])
        >>> bool(torch.allclose(w.sum(dim=1), torch.ones(2), atol=1e-5))
        True
        """
        # ----------------------------------------------------------------
        # Validate all input sequences
        # ----------------------------------------------------------------
        reference: torch.Tensor = self._validate_outputs(outputs)
        self._validate_confidence_maps(confidence_maps, reference)
        self._validate_wavelet_features(wavelet_features, reference)
        batch_size: int = reference.shape[0]

        # ----------------------------------------------------------------
        # Compute per-image softmax weights
        #
        # Shape: [B, num_shifts]
        # ----------------------------------------------------------------
        weights: torch.Tensor = self.get_weights(
            outputs, confidence_maps, wavelet_features
        )  # [B, N]

        # ----------------------------------------------------------------
        # Stack outputs and aggregate
        #
        # Stack    : list of N x [B, C, H, W]  ->  [N, B, C, H, W]
        # Permute  : [N, B, C, H, W]            ->  [B, N, C, H, W]
        # Weights  : [B, N] -> [B, N, 1, 1, 1]  (broadcast over C, H, W)
        # Product  : [B, N, C, H, W] * [B, N, 1, 1, 1] -> [B, N, C, H, W]
        # Sum      : dim=1                       ->  [B, C, H, W]
        # ----------------------------------------------------------------
        stacked: torch.Tensor = torch.stack(list(outputs), dim=0)  # [N, B, C, H, W]
        stacked = stacked.permute(1, 0, 2, 3, 4)  # [B, N, C, H, W]

        weights_broadcast: torch.Tensor = weights.view(
            batch_size, self.num_shifts, 1, 1, 1
        )  # [B, N, 1, 1, 1]

        # Promote stacked dtype to match weights if necessary (e.g. fp16
        # inputs with fp32 weight predictor) so the multiplication is
        # well-defined. The original input dtype is saved and restored
        # so that fp16 callers receive fp16 output, preserving full
        # compatibility with FP16 training.
        input_dtype: torch.dtype = stacked.dtype
        if stacked.dtype != weights_broadcast.dtype:
            stacked = stacked.to(weights_broadcast.dtype)

        fused: torch.Tensor = (stacked * weights_broadcast).sum(dim=1)  # [B, C, H, W]

        # Restore the caller's original dtype (e.g. fp16 -> fp16).
        fused = fused.to(input_dtype)

        # ----------------------------------------------------------------
        # Return
        # ----------------------------------------------------------------
        if return_weights:
            return fused, weights
        return fused

    # ------------------------------------------------------------------
    # Entropy
    # ------------------------------------------------------------------

    def entropy(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the batch-averaged Shannon entropy of the predicted weights.

        For each batch element ``b``, the per-image entropy is::

            H_b = -Σ_i w_{b,i} • log(w_{b,i} + eps)

        This method returns the mean of ``H_b`` over the batch
        dimension, giving a single scalar summary suitable for
        logging or use as a regularisation term.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.
            Together with ``outputs`` and ``confidence_maps``, used to
            predict the weight distribution via ``get_weights``.

        Returns
        -------
        torch.Tensor
            Scalar tensor (shape ``[]``), dtype matching the weight
            predictor's output dtype. Retains the autograd graph; can
            be added directly to a loss.

        Notes
        -----
        Entropy is maximised (= log N) when all weights for a given
        image are equal (1/N) and is zero when the distribution for
        that image is a delta (one weight = 1). It can be used as a
        regularisation term in the training loss to encourage or
        discourage peaked, image-specific weight distributions
        depending on the sign of the regularisation coefficient.

        Examples
        --------
        >>> wccs = WaveletConfidenceCycleSpinning(
        ...     num_shifts=8, channels=1, wavelet_channels=4
        ... )
        >>> outs = [torch.randn(2, 1, 16, 16) for _ in range(8)]
        >>> confs = [torch.rand(2, 1, 16, 16) for _ in range(8)]
        >>> wavs = [torch.randn(2, 4, 8, 8) for _ in range(8)]
        >>> h = wccs.entropy(outs, confs, wavs)
        >>> bool(h.item() > 0.0)
        True
        """
        weights: torch.Tensor = self.get_weights(
            outputs, confidence_maps, wavelet_features
        )  # [B, N]
        per_image_entropy: torch.Tensor = -(
            weights * torch.log(weights.clamp(min=self.eps))
        ).sum(dim=1)  # [B]
        return per_image_entropy.mean()

    # ------------------------------------------------------------------
    # Entropy regularizer
    # ------------------------------------------------------------------

    def entropy_regularizer(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
        coefficient: float = 1.0,
    ) -> torch.Tensor:
        """Entropy regularization term for use directly in a training loss.

        Returns ``coefficient * H`` where H is the batch-averaged
        Shannon entropy of the predicted weight distributions (see
        ``entropy``).

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.
            Together with ``outputs`` and ``confidence_maps``, used to
            predict the weight distribution via ``get_weights``.
        coefficient:
            Scalar multiplier applied to the entropy.

            * Positive value -> maximise entropy -> encourage uniform,
              image-agnostic weights.
            * Negative value -> minimise entropy -> encourage sparse,
              wavelet-specific weights.

            Default 1.0.

        Returns
        -------
        torch.Tensor
            Scalar tensor, retains the autograd graph. Can be added
            directly to a training loss::

                loss = diffusion_loss + wccs.entropy_regularizer(
                    outs, confs, wavs, lambda_ent
                )

        Examples
        --------
        >>> wccs = WaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4
        ... )
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> reg = wccs.entropy_regularizer(outs, confs, wavs, coefficient=0.01)
        >>> reg.shape
        torch.Size([])
        """
        return coefficient * self.entropy(outputs, confidence_maps, wavelet_features)

    # ------------------------------------------------------------------
    # Effective number of shifts
    # ------------------------------------------------------------------

    def effective_num_shifts(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the batch-averaged effective number of active shifts.

        Defined as::

            N_eff = exp(H)

        where H is the batch-averaged Shannon entropy of the predicted
        weight distribution.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.

        Returns
        -------
        torch.Tensor
            Scalar tensor.

            * Uniform weights for every image (all equal 1/N):
              N_eff = num_shifts.
            * One dominant weight per image (-> 1): N_eff -> 1.

        Examples
        --------
        >>> wccs = WaveletConfidenceCycleSpinning(
        ...     num_shifts=8, channels=1, wavelet_channels=4
        ... )
        >>> outs = [torch.randn(2, 1, 16, 16) for _ in range(8)]
        >>> confs = [torch.rand(2, 1, 16, 16) for _ in range(8)]
        >>> wavs = [torch.randn(2, 4, 8, 8) for _ in range(8)]
        >>> n_eff = wccs.effective_num_shifts(outs, confs, wavs)
        >>> bool(0.0 < n_eff.item() <= 8.0 + 1e-3)
        True
        """
        return torch.exp(
            self.entropy(outputs, confidence_maps, wavelet_features)
        )

    # ------------------------------------------------------------------
    # Weight variance
    # ------------------------------------------------------------------

    def weight_variance(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the variance of the predicted weight distribution.

        Computes the population variance (``unbiased=False``) over
        the flattened ``[B, num_shifts]`` weight tensor.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.

        Returns
        -------
        torch.Tensor
            Scalar tensor (population variance, ``unbiased=False``).
            Retains the autograd graph.

        Examples
        --------
        >>> wccs = WaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4
        ... )
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> v = wccs.weight_variance(outs, confs, wavs)
        >>> v.shape
        torch.Size([])
        >>> bool(v.item() >= 0.0)
        True
        """
        weights: torch.Tensor = self.get_weights(
            outputs, confidence_maps, wavelet_features
        )  # [B, N]
        return weights.var(unbiased=False)

    # ------------------------------------------------------------------
    # Index utilities
    # ------------------------------------------------------------------

    def max_weight_index(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Return, per batch element, the index of the highest-weight shift.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B]``, dtype ``int64``. Detached from the autograd graph.

        Examples
        --------
        >>> wccs = WaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4
        ... )
        >>> outs = [torch.randn(3, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(3, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(3, 4, 4, 4) for _ in range(4)]
        >>> idx = wccs.max_weight_index(outs, confs, wavs)
        >>> idx.shape
        torch.Size([3])
        """
        weights: torch.Tensor = self._weights_no_grad(
            outputs, confidence_maps, wavelet_features
        )
        return weights.argmax(dim=1)

    def min_weight_index(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Return, per batch element, the index of the lowest-weight shift.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B]``, dtype ``int64``. Detached from the autograd graph.

        Examples
        --------
        >>> wccs = WaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4
        ... )
        >>> outs = [torch.randn(3, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(3, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(3, 4, 4, 4) for _ in range(4)]
        >>> idx = wccs.min_weight_index(outs, confs, wavs)
        >>> idx.shape
        torch.Size([3])
        """
        weights: torch.Tensor = self._weights_no_grad(
            outputs, confidence_maps, wavelet_features
        )
        return weights.argmin(dim=1)

    # ------------------------------------------------------------------
    # Uniform reference distribution
    # ------------------------------------------------------------------

    def uniform_weights(self, batch_size: int) -> torch.Tensor:
        """Return the uniform weight matrix ``1/N`` for a given batch size.

        Creates a tensor of shape ``[batch_size, num_shifts]`` where
        every entry equals ``1 / num_shifts``, placed on the same
        device and with the same dtype as the weight-predictor MLP's
        first linear layer.

        Parameters
        ----------
        batch_size:
            Number of rows ``B`` in the returned uniform distribution.
            Must be a positive integer.

        Returns
        -------
        torch.Tensor
            Shape ``[batch_size, num_shifts]``, all entries equal to
            ``1 / num_shifts``. Not connected to the autograd graph.

        Raises
        ------
        ValueError
            If ``batch_size`` is not a positive integer.

        Examples
        --------
        >>> wccs = WaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4
        ... )
        >>> u = wccs.uniform_weights(batch_size=2)
        >>> u.shape
        torch.Size([2, 4])
        >>> bool(torch.allclose(u, torch.full((2, 4), 0.25)))
        True
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive integer, got {batch_size!r}."
            )
        reference_param: torch.Tensor = self.weight_predictor[0].weight  # type: ignore[index]
        return torch.full(
            (batch_size, self.num_shifts),
            1.0 / self.num_shifts,
            device=reference_param.device,
            dtype=reference_param.dtype,
        )

    # ------------------------------------------------------------------
    # Divergence measures
    # ------------------------------------------------------------------

    def kl_to_uniform(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the batch-averaged KL divergence to the uniform distribution.

        For each batch element ``b``::

            KL_b(w ‖ u) = Σ_i w_{b,i} • [log(w_{b,i}) - log(1/N)]
                        = Σ_i w_{b,i} • log(N • w_{b,i})

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.

        Returns
        -------
        torch.Tensor
            Scalar tensor >= 0. Zero iff every row of ``w`` is exactly
            uniform. Retains the autograd graph.

        Examples
        --------
        >>> wccs = WaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4
        ... )
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> kl = wccs.kl_to_uniform(outs, confs, wavs)
        >>> kl.shape
        torch.Size([])
        >>> bool(kl.item() >= -1e-6)
        True
        """
        weights: torch.Tensor = self.get_weights(
            outputs, confidence_maps, wavelet_features
        )  # [B, N]
        uniform: torch.Tensor = self.uniform_weights(weights.shape[0])
        per_image_kl: torch.Tensor = (
            weights
            * (
                torch.log(weights.clamp(min=self.eps))
                - torch.log(uniform)
            )
        ).sum(dim=1)  # [B]
        return per_image_kl.mean()

    def js_to_uniform(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the batch-averaged Jensen-Shannon divergence to uniform.

        The JSD is a symmetric, bounded alternative to KL divergence.
        For each batch element ``b``::

            M_b  = (w_b + u) / 2
            JSD_b(w_b ‖ u) = [KL(w_b ‖ M_b) + KL(u ‖ M_b)] / 2

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.

        Returns
        -------
        torch.Tensor
            Scalar tensor in ``[0, log(2)]``. Zero iff every row of
            ``w`` equals uniform. Retains the autograd graph.

        Examples
        --------
        >>> wccs = WaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4
        ... )
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> jsd = wccs.js_to_uniform(outs, confs, wavs)
        >>> jsd.shape
        torch.Size([])
        >>> bool(jsd.item() >= -1e-6)
        True
        """
        weights: torch.Tensor = self.get_weights(
            outputs, confidence_maps, wavelet_features
        )  # [B, N]
        uniform: torch.Tensor = self.uniform_weights(weights.shape[0])
        mixture: torch.Tensor = 0.5 * (weights + uniform)

        kl_w_m: torch.Tensor = (
            weights
            * (
                torch.log(weights.clamp(min=self.eps))
                - torch.log(mixture.clamp(min=self.eps))
            )
        ).sum(dim=1)  # [B]
        kl_u_m: torch.Tensor = (
            uniform
            * (
                torch.log(uniform.clamp(min=self.eps))
                - torch.log(mixture.clamp(min=self.eps))
            )
        ).sum(dim=1)  # [B]

        per_image_jsd: torch.Tensor = 0.5 * (kl_w_m + kl_u_m)  # [B]
        return per_image_jsd.mean()

    # ------------------------------------------------------------------
    # Temperature control
    # ------------------------------------------------------------------

    def set_temperature(self, temperature: float) -> None:
        """Update the softmax temperature in-place with validation.

        Parameters
        ----------
        temperature:
            New temperature value τ > 0.

        Raises
        ------
        ValueError
            If ``temperature`` is not strictly positive.

        Examples
        --------
        >>> wccs = WaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4, temperature=1.0
        ... )
        >>> wccs.set_temperature(0.5)
        >>> wccs.temperature
        0.5
        >>> try:
        ...     wccs.set_temperature(-1.0)
        ... except ValueError as e:
        ...     print("caught:", e)
        caught: temperature must be strictly positive, got -1.0.
        """
        if temperature <= 0.0:
            raise ValueError(
                f"temperature must be strictly positive, got {temperature}."
            )
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Gradient control
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Disable gradient updates for the entire weight-predictor MLP.

        After calling ``freeze()``, none of ``weight_predictor``'s
        parameters will receive gradients during backpropagation.

        Examples
        --------
        >>> wccs = WaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4
        ... )
        >>> wccs.freeze()
        >>> wccs.is_frozen()
        True
        """
        for param in self.weight_predictor.parameters():
            param.requires_grad_(False)

    def unfreeze(self) -> None:
        """Enable gradient updates for the entire weight-predictor MLP.

        Restores gradient computation for every parameter in
        ``weight_predictor`` after a prior call to ``freeze()``.

        Examples
        --------
        >>> wccs = WaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4
        ... )
        >>> wccs.freeze()
        >>> wccs.unfreeze()
        >>> wccs.is_frozen()
        False
        """
        for param in self.weight_predictor.parameters():
            param.requires_grad_(True)

    def is_frozen(self) -> bool:
        """Return ``True`` if every weight-predictor parameter is frozen.

        Returns
        -------
        bool
            ``True`` iff ``requires_grad`` is ``False`` for *all*
            parameters in ``weight_predictor``.

        Examples
        --------
        >>> wccs = WaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4
        ... )
        >>> wccs.is_frozen()
        False
        >>> wccs.freeze()
        >>> wccs.is_frozen()
        True
        """
        return all(
            not param.requires_grad
            for param in self.weight_predictor.parameters()
        )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _weights_no_grad(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-image softmax weights without retaining the autograd graph.

        Used internally by logging and diagnostic methods
        (``weight_statistics``, ``summary``, ``save_statistics``,
        ``max_weight_index``, ``min_weight_index``) that need the
        weight values purely for inspection, not for gradient
        computation.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, num_shifts]``, detached from the autograd graph.
        """
        with torch.no_grad():
            return self.get_weights(outputs, confidence_maps, wavelet_features)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def weight_statistics(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
    ) -> Dict[str, float]:
        """Return useful statistics about the predicted weight distribution.

        All statistics are computed from a single ``get_weights()``
        call (under ``torch.no_grad()``) to avoid redundant forward
        passes.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.

        Returns
        -------
        Dict[str, float]
            A dictionary with the following ``str`` keys and ``float``
            values:

            ``"entropy"``
                Batch-averaged Shannon entropy.
            ``"effective_num_shifts"``
                N_eff = exp(H).
            ``"max_weight"``
                Maximum weight value across the whole ``[B, N]`` tensor.
            ``"min_weight"``
                Minimum weight value across the whole ``[B, N]`` tensor.
            ``"std_weight"``
                Population standard deviation of all weight values.

        Examples
        --------
        >>> wccs = WaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4
        ... )
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> stats = wccs.weight_statistics(outs, confs, wavs)
        >>> set(stats.keys()) == {
        ...     "entropy", "effective_num_shifts",
        ...     "max_weight", "min_weight", "std_weight"
        ... }
        True
        """
        weights: torch.Tensor = self._weights_no_grad(
            outputs, confidence_maps, wavelet_features
        )
        per_image_entropy: torch.Tensor = -(
            weights * torch.log(weights.clamp(min=self.eps))
        ).sum(dim=1)  # [B]
        entropy_val: torch.Tensor = per_image_entropy.mean()

        return {
            "entropy": float(entropy_val.item()),
            "effective_num_shifts": float(torch.exp(entropy_val).item()),
            "max_weight": float(weights.max().item()),
            "min_weight": float(weights.min().item()),
            "std_weight": float(weights.std(unbiased=False).item()),
        }

    def summary(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
    ) -> Dict[str, float]:
        """Return a comprehensive diagnostic summary of the module's behaviour.

        Extends ``weight_statistics()`` with the batch-averaged weight
        variance. All quantities are computed from a single
        ``get_weights()`` call.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.

        Returns
        -------
        Dict[str, float]
            A dictionary with the following ``str`` keys and ``float``
            values:

            ``"entropy"``
                Batch-averaged Shannon entropy.
            ``"effective_num_shifts"``
                N_eff = exp(H).
            ``"max_weight"``
                Maximum weight value across the whole ``[B, N]`` tensor.
            ``"min_weight"``
                Minimum weight value across the whole ``[B, N]`` tensor.
            ``"weight_variance"``
                Population variance of all weight values.

        Examples
        --------
        >>> wccs = WaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4
        ... )
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> s = wccs.summary(outs, confs, wavs)
        >>> set(s.keys()) == {
        ...     "entropy", "effective_num_shifts",
        ...     "max_weight", "min_weight", "weight_variance"
        ... }
        True
        """
        weights: torch.Tensor = self._weights_no_grad(
            outputs, confidence_maps, wavelet_features
        )
        per_image_entropy: torch.Tensor = -(
            weights * torch.log(weights.clamp(min=self.eps))
        ).sum(dim=1)  # [B]
        entropy_val: torch.Tensor = per_image_entropy.mean()

        return {
            "entropy": float(entropy_val.item()),
            "effective_num_shifts": float(torch.exp(entropy_val).item()),
            "max_weight": float(weights.max().item()),
            "min_weight": float(weights.min().item()),
            "weight_variance": float(weights.var(unbiased=False).item()),
        }

    # ------------------------------------------------------------------
    # Checkpoint-friendly statistics snapshot
    # ------------------------------------------------------------------

    def save_statistics(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
    ) -> Dict[str, float]:
        """Return a detached statistics snapshot for checkpoint logging.

        Intended to be stored alongside a model checkpoint so that the
        behaviour of the wavelet-confidence-guided aggregation weights
        can be inspected from the log without reloading the model.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.

        Returns
        -------
        Dict[str, float]
            A dictionary with the following keys:

            ``"entropy"``
                Batch-averaged Shannon entropy.
            ``"effective_num_shifts"``
                N_eff = exp(H).
            ``"kl_to_uniform"``
                Batch-averaged KL(w ‖ uniform).
            ``"max_weight"``
                Maximum weight value across the whole ``[B, N]`` tensor.
            ``"min_weight"``
                Minimum weight value across the whole ``[B, N]`` tensor.
            ``"weight_variance"``
                Population variance of all weight values.
            ``"max_weight_index"``
                Index of the highest-weight shift for the *first*
                batch element (as a float, for JSON/CSV serialisation).
            ``"min_weight_index"``
                Index of the lowest-weight shift for the *first*
                batch element (as a float).

        Examples
        --------
        >>> wccs = WaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4
        ... )
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> s = wccs.save_statistics(outs, confs, wavs)
        >>> set(s.keys()) == {
        ...     "entropy", "effective_num_shifts", "kl_to_uniform",
        ...     "max_weight", "min_weight", "weight_variance",
        ...     "max_weight_index", "min_weight_index",
        ... }
        True
        """
        weights: torch.Tensor = self._weights_no_grad(
            outputs, confidence_maps, wavelet_features
        )
        per_image_entropy: torch.Tensor = -(
            weights * torch.log(weights.clamp(min=self.eps))
        ).sum(dim=1)  # [B]
        entropy_val: torch.Tensor = per_image_entropy.mean()

        uniform: torch.Tensor = self.uniform_weights(weights.shape[0])
        per_image_kl: torch.Tensor = (
            weights
            * (
                torch.log(weights.clamp(min=self.eps))
                - torch.log(uniform)
            )
        ).sum(dim=1)  # [B]
        kl_val: torch.Tensor = per_image_kl.mean()

        return {
            "entropy": float(entropy_val.item()),
            "effective_num_shifts": float(torch.exp(entropy_val).item()),
            "kl_to_uniform": float(kl_val.item()),
            "max_weight": float(weights.max().item()),
            "min_weight": float(weights.min().item()),
            "weight_variance": float(weights.var(unbiased=False).item()),
            "max_weight_index": float(weights[0].argmax().item()),
            "min_weight_index": float(weights[0].argmin().item()),
        }

    # ------------------------------------------------------------------
    # Module representation
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Return a concise parameter summary for ``print(module)``.

        Returns
        -------
        str
            Human-readable representation of the module's configuration.

        Examples
        --------
        >>> wccs = WaveletConfidenceCycleSpinning(
        ...     num_shifts=9, channels=1, wavelet_channels=4,
        ...     hidden_dim=128, temperature=0.5, pooling="avg",
        ... )
        >>> print(wccs)  # doctest: +ELLIPSIS
        WaveletConfidenceCycleSpinning(
          ...
        )
        """
        return (
            f"num_shifts={self.num_shifts}, "
            f"channels={self.channels}, "
            f"wavelet_channels={self.wavelet_channels}, "
            f"hidden_dim={self.hidden_dim}, "
            f"temperature={self.temperature}, "
            f"pooling={self.pooling}"
        )
