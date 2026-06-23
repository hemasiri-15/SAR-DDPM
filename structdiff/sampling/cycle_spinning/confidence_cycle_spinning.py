"""
structdiff/inference/confidence_cycle_spinning.py
====================================================
A26c: ConfidenceCycleSpinning — confidence-guided, image-adaptive
softmax aggregation of cycle-shifted diffusion outputs for SAR
despeckling.

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

Both A26a and A26b condition the aggregation weights solely on the
*content* of the shifted predictions. Neither uses any explicit
measure of how *reliable* each shifted prediction is.

Confidence-guided aggregation (A26c)
--------------------------------------
A26c extends A26b by additionally conditioning the weight predictor on
a per-shift **confidence map** ``σ_i``, one scalar-channel map per
shift, indicating how trustworthy the diffusion model considers each
spatial location of that shift's prediction to be (e.g. derived from
predicted variance, ensemble disagreement, or any other per-pixel
uncertainty signal upstream of this module). The intuition is direct:

* High-confidence shifts should receive **larger** aggregation weights.
* Low-confidence shifts should contribute **less** to the fused output.

Formally, for shift ``i`` with prediction ``x_i ∈ R^{B×C×H×W}`` and
confidence map ``σ_i ∈ R^{B×1×H×W}``::

    z_i = GAP(x_i)                              z_i ∈ R^{B × C}
    c_i = GAP(σ_i)                              c_i ∈ R^{B × 1}
    d_i = concat(z_i, c_i)                       d_i ∈ R^{B × (C+1)}
    d   = concat(d_1, …, d_N)                    d   ∈ R^{B × N(C+1)}
    h   = GELU(Linear(d))                        h   ∈ R^{B × hidden_dim}
    a   = Linear(h)                               a   ∈ R^{B × N}
    w   = softmax(a / τ, dim=1)                   w   ∈ R^{B × N}
    x̂   = Σ_i w_i • x_i                           x̂   ∈ R^{B × C × H × W}

Note that the *fusion* itself still combines only the image content
``x_i`` (matching A26a/A26b and the original SAR-DDPM average); the
confidence maps ``σ_i`` are used exclusively to help *predict* the
weights, not to directly reweight pixels. Per-pixel (spatially
adaptive) use of confidence is left to a later stage of the A26
roadmap.

Initialization guarantee
-------------------------
As in A26b, the final linear layer of the weight-predictor MLP is
initialised with very small weights (``std=1e-3``) and a zero bias.
Because the logits ``a`` are therefore approximately zero for any
input at step 0, the softmax output is approximately uniform::

    a ≈ 0  →  w_i ≈ 1/N  for all i

This means **at initialisation A26c approximately reproduces the
original SAR-DDPM equal-weight average** (and is approximately
equivalent to A26a's and A26b's uniform initialisation), regardless of
the content of the supplied confidence maps, so training begins close
to the known-good heuristic baseline.

Checkpoint compatibility
-------------------------
``ConfidenceCycleSpinning`` is a new module. Its ``weight_predictor``
has a different input dimensionality (``N * (channels + 1)``) than
A26b's (``N * channels``), so **A26b checkpoints cannot be loaded
directly into A26c's ``weight_predictor`` weights** — the shapes do
not match. When migrating from an A26b run, load everything except
``weight_predictor.*`` with::

    model.load_state_dict(checkpoint, strict=False)

and allow ``weight_predictor.*`` to keep its freshly initialised,
near-zero values, which — per the initialisation guarantee above —
still preserves approximately equal-weight averaging behaviour at the
start of A26c training.

Future roadmap
---------------
This module is the next link in the A26 series:

* **A26d** — Wavelet-Guided Fusion: extend the conditioning signal to
  ``w = f(x, σ, W)``, deriving additional per-shift descriptors in the
  wavelet domain (LL/LH/HL/HH subbands from A12) so the weight
  predictor can react to frequency-domain artefacts as well as pixel-
  domain content and confidence.
* **A26e** — Structure Tensor Fusion: further extend to
  ``w = f(x, σ, W, S)`` by conditioning on structure-tensor eigenvalue
  features (A10/A11), so edge- and texture-rich regions can steer
  aggregation differently from smooth regions.
* **A26f** — Transformer Fusion: replace the MLP with cross-attention
  over the stack of *N* shifted outputs (and their confidence maps),
  allowing each shift to attend to the others before weights are
  produced.
* **A26g** — Learnable Shift Coordinates: jointly learn the (row, col)
  shift grid rather than using a fixed uniform grid, feeding shift
  geometry as an additional conditioning signal.
* **A26h** — Hierarchical Cycle Spinning: nested coarse + fine shift
  pyramids with independent confidence-guided weight predictors per
  level.
* **A26i** — Full Adaptive Cycle-Spinning Transformer: integrates
  A26d-h into a unified transformer-based aggregation module that
  consumes confidence, wavelet, and structure-tensor features jointly.
* **A26j** — Bayesian Cycle Spinning: model per-image shift weights as
  a Dirichlet distribution and estimate uncertainty over the
  aggregation weights themselves, enabling principled confidence
  intervals over the fused prediction for journal-level uncertainty
  quantification in SAR despeckling.
* **A26k** — Meta-Learned Cycle Spinning: learn weight-prediction
  policies across datasets, allowing zero-shot transfer of the
  confidence-guided aggregator to unseen SAR sensor configurations and
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
:class:`~structdiff.inference.adaptive_cycle_spinning.AdaptiveCycleSpinning`
exactly, with the sole functional difference being that every method
which predicts or consumes weights now also accepts ``confidence_maps``
alongside ``outputs``. This keeps downstream logging, training, and
ablation code interchangeable across A26b and A26c, and ensures future
extensions (A26d-i) can inherit or compose from this module without
breaking changes.

References
----------
Coifman, R.R. & Donoho, D.L. (1995). Translation-Invariant
De-Noising. *Wavelets and Statistics*, Springer.

Notes
-----
* All computation is performed in PyTorch; no NumPy, no CPU transfer,
  no in-place operations, full autograd support.
* The module is device-agnostic: all submodules move with
  ``model.to(device)``.
* Weights are predicted per batch element, so two images in the same
  batch may receive entirely different aggregation strategies, each
  informed by its own per-shift confidence maps.
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
#: the A26a / A26b uniform initialisation.
_FINAL_LAYER_INIT_STD: float = 1e-3

#: Epsilon added inside the entropy logarithm for numerical stability.
#: Must satisfy _LOG_EPS << 1/N for any practical N.
_LOG_EPS: float = 1e-8

#: Number of channels expected in every confidence map (one scalar
#: confidence value per spatial location).
_CONFIDENCE_CHANNELS: int = 1


# ---------------------------------------------------------------------------
# ConfidenceCycleSpinning
# ---------------------------------------------------------------------------


class ConfidenceCycleSpinning(nn.Module):
    """Confidence-guided, image-adaptive softmax aggregation of cycle-shifted
    diffusion outputs.

    Extends A26b
    (:class:`~structdiff.inference.adaptive_cycle_spinning.AdaptiveCycleSpinning`)
    by additionally conditioning the per-image weight predictor on a
    per-shift confidence map, in addition to the shifted prediction
    content itself. Each of the *N* shifted prediction tensors is
    spatially pooled to a ``[B, C]`` descriptor, and each of the *N*
    accompanying confidence maps is spatially pooled to a ``[B, 1]``
    descriptor; the two are concatenated per shift to ``[B, C+1]``,
    all *N* per-shift descriptors are concatenated to ``[B, N•(C+1)]``,
    and a two-layer MLP maps this to per-shift logits ``[B, N]``. A
    temperature-scaled softmax over the shift dimension then yields
    per-image aggregation weights that sum to 1 across shifts for
    every batch element.

    At construction, the final linear layer of the MLP is initialised
    with very small weights and zero bias, so the predicted logits
    start near zero and the softmax output starts near-uniform —
    closely matching the original SAR-DDPM equal-weight average and
    the A26a / A26b uniform initialisation, regardless of the content
    of the supplied confidence maps.

    Parameters
    ----------
    num_shifts:
        Total number of cycle-spin shifts *N*. Must be a positive
        integer. Corresponds to the number of (row, col) shift pairs
        in the nested loop of the existing SAR-DDPM inference code.
    channels:
        Number of channels *C* in each shifted prediction tensor.
        Must be a positive integer. Used to size the input dimension
        of the weight-predictor MLP (``N * (channels + 1)``).
    hidden_dim:
        Width of the MLP's hidden layer. Must be a positive integer.
        Default 128.
    temperature:
        Softmax temperature τ > 0 applied to the predicted logits
        before the softmax. Lower values sharpen the per-image weight
        distribution; higher values flatten it. Default 1.0.
    pooling:
        Spatial pooling mode used to compute both the per-shift image
        descriptors ``z_i = pool(x_i)`` and confidence descriptors
        ``c_i = pool(σ_i)``. One of:

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
        ``AdaptiveMaxPool2d(1)``), shared by both the image and
        confidence descriptor branches.
    weight_predictor : nn.Sequential
        The two-layer MLP (``Linear`` → ``GELU`` → ``Linear``) that
        maps concatenated pooled image-and-confidence descriptors to
        per-shift logits.

    Examples
    --------
    >>> import torch
    >>> from structdiff.inference.confidence_cycle_spinning import (
    ...     ConfidenceCycleSpinning,
    ... )
    >>> ccs = ConfidenceCycleSpinning(num_shifts=9, channels=1)
    >>> ccs.num_shifts
    9
    >>> outputs = [torch.randn(2, 1, 64, 64) for _ in range(9)]
    >>> confidence_maps = [torch.rand(2, 1, 64, 64) for _ in range(9)]
    >>> fused = ccs(outputs, confidence_maps)
    >>> fused.shape
    torch.Size([2, 1, 64, 64])

    >>> # return_weights=True yields per-image weights
    >>> fused, weights = ccs(outputs, confidence_maps, return_weights=True)
    >>> weights.shape
    torch.Size([2, 9])
    >>> bool(torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-5))
    True
    """

    def __init__(
        self,
        num_shifts: int,
        channels: int,
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
        self.hidden_dim: int = hidden_dim
        self.temperature: float = temperature
        self.pooling: str = pooling
        self.eps: float = eps

        # ----------------------------------------------------------------
        # Spatial pooling layer
        #
        # Shared by both branches:
        #   - Reduces each shifted prediction [B, C, H, W] to [B, C, 1, 1].
        #   - Reduces each confidence map [B, 1, H, W] to [B, 1, 1, 1].
        # ----------------------------------------------------------------
        self.pool: nn.Module
        if pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:  # pooling == "max"
            self.pool = nn.AdaptiveMaxPool2d(1)

        # ----------------------------------------------------------------
        # Weight-predictor MLP
        #
        # Input : [B, num_shifts * (channels + 1)]
        #          (concatenated image + confidence descriptors)
        # Hidden: [B, hidden_dim]
        # Output: [B, num_shifts]              (per-shift logits)
        # ----------------------------------------------------------------
        self.weight_predictor: nn.Sequential = nn.Sequential(
            nn.Linear(num_shifts * (channels + _CONFIDENCE_CHANNELS), hidden_dim),
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
        (image content or confidence map), so::

            w = softmax(a / τ) ≈ [1/N, …, 1/N]

        reproducing the original SAR-DDPM equal-weight average and the
        A26a / A26b uniform initialisation at step 0.

        Examples
        --------
        >>> ccs = ConfidenceCycleSpinning(num_shifts=4, channels=2)
        >>> ccs.reset_parameters()  # re-draw initial weights
        >>> first_linear = ccs.weight_predictor[0]
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
        # softmax output ≈ uniform, matching the SAR-DDPM / A26a / A26b
        # baseline regardless of confidence-map content.
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
          dtype, and device match ``reference`` (the validated
          ``outputs[0]`` tensor).

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

    # ------------------------------------------------------------------
    # Descriptor extraction
    # ------------------------------------------------------------------

    def _extract_descriptors(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute and concatenate pooled per-shift image+confidence descriptors.

        For each shift ``i``, the image tensor ``x_i`` of shape
        ``[B, C, H, W]`` is spatially pooled to ``[B, C, 1, 1]`` and
        flattened to ``[B, C]``; the confidence map ``σ_i`` of shape
        ``[B, 1, H, W]`` is spatially pooled to ``[B, 1, 1, 1]`` and
        flattened to ``[B, 1]``. The two are concatenated to
        ``[B, C+1]``. The *N* per-shift descriptors are concatenated
        along the channel dimension to produce a single descriptor of
        shape ``[B, N * (C+1)]`` suitable as input to
        ``weight_predictor``.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``. Assumed to
            have already been validated by ``_validate_outputs``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``. Assumed to
            have already been validated by ``_validate_confidence_maps``.

        Returns
        -------
        torch.Tensor
            Concatenated descriptor of shape ``[B, N * (C + 1)]``.
        """
        descriptors: List[torch.Tensor] = []
        for tensor, conf in zip(outputs, confidence_maps):
            pooled_image: torch.Tensor = self.pool(tensor)  # [B, C, 1, 1]
            pooled_conf: torch.Tensor = self.pool(conf)  # [B, 1, 1, 1]
            batch_size: int = pooled_image.shape[0]
            image_descriptor: torch.Tensor = pooled_image.reshape(
                batch_size, -1
            )  # [B, C]
            conf_descriptor: torch.Tensor = pooled_conf.reshape(
                batch_size, -1
            )  # [B, 1]
            descriptors.append(
                torch.cat([image_descriptor, conf_descriptor], dim=1)
            )  # [B, C+1]
        return torch.cat(descriptors, dim=1)  # [B, N * (C+1)]

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def get_weights(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Predict per-image softmax aggregation weights from outputs and confidence.

        Computes pooled image and confidence descriptors for every
        shift, concatenates them per shift and then across shifts,
        passes the result through ``weight_predictor`` to obtain
        per-shift logits, and applies a temperature-scaled softmax
        over the shift dimension.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``, giving a
            per-pixel confidence/reliability signal for the
            corresponding entry of ``outputs``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, num_shifts]``. Every row sums to 1.0 and every
            entry is strictly positive. Retains the autograd graph;
            never detached.

        Raises
        ------
        ValueError
            If ``self.temperature`` is not strictly positive (checked
            here in addition to ``__init__`` to guard against external
            mutation of the attribute), or if ``outputs`` or
            ``confidence_maps`` fail validation (see
            ``_validate_outputs`` and ``_validate_confidence_maps``).

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
        >>> ccs = ConfidenceCycleSpinning(num_shifts=4, channels=1)
        >>> outs = [torch.randn(3, 1, 16, 16) for _ in range(4)]
        >>> confs = [torch.rand(3, 1, 16, 16) for _ in range(4)]
        >>> w = ccs.get_weights(outs, confs)
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

        descriptors: torch.Tensor = self._extract_descriptors(
            outputs, confidence_maps
        )  # [B, N*(C+1)]

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
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Aggregate cycle-shifted diffusion outputs with confidence-guided weights.

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
            If output tensors have inconsistent shapes, dtypes, or
            devices.
        ValueError
            If ``confidence_maps`` is empty, or
            ``len(confidence_maps) != self.num_shifts``.
        ValueError
            If any confidence map is not 4-dimensional, does not have
            exactly one channel, or its batch size, spatial dimensions,
            dtype, or device does not match ``outputs``.

        Notes
        -----
        The aggregation is::

            z_i = GAP_or_GMP(x_i)                        [B, C]
            c_i = GAP_or_GMP(σ_i)                         [B, 1]
            d_i = concat(z_i, c_i)                         [B, C+1]
            d   = concat(d_1, …, d_N)                      [B, N*(C+1)]
            a   = weight_predictor(d)                      [B, N]
            w   = softmax(a / temperature, dim=1)          [B, N]
            x̂   = Σ_i w_i • x_i                            [B, C, H, W]

        Note that only the image content ``x_i`` is summed into the
        fused output; the confidence maps ``σ_i`` are consumed solely
        by the weight predictor. Weights are broadcast as
        ``[B, N, 1, 1, 1]`` over a ``[B, N, C, H, W]``-permuted stack
        of the shifted outputs, then summed along the shift dimension
        to produce ``[B, C, H, W]``. No in-place operations are used;
        the autograd graph is preserved throughout.

        Because the final layer of ``weight_predictor`` is initialised
        with very small weights, at construction time the predicted
        logits are approximately zero for any input, so ``w_i ≈ 1/N``
        and the output is approximately identical to the original
        SAR-DDPM equal-weight average, regardless of the confidence
        map content.

        Examples
        --------
        >>> import torch
        >>> from structdiff.inference.confidence_cycle_spinning import (
        ...     ConfidenceCycleSpinning,
        ... )
        >>> ccs = ConfidenceCycleSpinning(num_shifts=4, channels=1)
        >>> outputs = [torch.ones(2, 1, 8, 8) * (i + 1.0) for i in range(4)]
        >>> confidence_maps = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> fused = ccs(outputs, confidence_maps)
        >>> fused.shape
        torch.Size([2, 1, 8, 8])

        >>> # return_weights=True
        >>> fused, w = ccs(outputs, confidence_maps, return_weights=True)
        >>> w.shape
        torch.Size([2, 4])
        >>> bool(torch.allclose(w.sum(dim=1), torch.ones(2), atol=1e-5))
        True
        """
        # ----------------------------------------------------------------
        # Validate outputs and confidence_maps sequences
        # ----------------------------------------------------------------
        reference: torch.Tensor = self._validate_outputs(outputs)
        self._validate_confidence_maps(confidence_maps, reference)
        batch_size: int = reference.shape[0]

        # ----------------------------------------------------------------
        # Compute per-image softmax weights
        #
        # Shape: [B, num_shifts]
        # ----------------------------------------------------------------
        weights: torch.Tensor = self.get_weights(outputs, confidence_maps)  # [B, N]

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
            Together with ``outputs``, used to predict the weight
            distribution via ``get_weights``.

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
        >>> ccs = ConfidenceCycleSpinning(num_shifts=8, channels=1)
        >>> outs = [torch.randn(2, 1, 16, 16) for _ in range(8)]
        >>> confs = [torch.rand(2, 1, 16, 16) for _ in range(8)]
        >>> h = ccs.entropy(outs, confs)
        >>> bool(h.item() > 0.0)
        True
        """
        weights: torch.Tensor = self.get_weights(outputs, confidence_maps)  # [B, N]
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
            Together with ``outputs``, used to predict the weight
            distribution via ``get_weights``.
        coefficient:
            Scalar multiplier applied to the entropy.

            * Positive value -> maximise entropy -> encourage uniform,
              image-agnostic weights.
            * Negative value -> minimise entropy -> encourage sparse,
              confidence-specific weights.

            Default 1.0.

        Returns
        -------
        torch.Tensor
            Scalar tensor, retains the autograd graph. Can be added
            directly to a training loss::

                loss = diffusion_loss + ccs.entropy_regularizer(
                    outs, confs, lambda_ent
                )

        Notes
        -----
        Useful for A26f (Transformer Fusion) and A26i (Full Adaptive
        Cycle-Spinning Transformer) where controlling per-image weight
        sparsity is important for training stability.

        Examples
        --------
        >>> ccs = ConfidenceCycleSpinning(num_shifts=4, channels=1)
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> reg = ccs.entropy_regularizer(outs, confs, coefficient=0.01)
        >>> reg.shape
        torch.Size([])
        """
        return coefficient * self.entropy(outputs, confidence_maps)

    # ------------------------------------------------------------------
    # Effective number of shifts
    # ------------------------------------------------------------------

    def effective_num_shifts(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the batch-averaged effective number of active shifts.

        Defined as::

            N_eff = exp(H)

        where H is the batch-averaged Shannon entropy of the predicted
        weight distribution (``self.entropy(outputs, confidence_maps)``).

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
            Together with ``outputs``, used to predict the weight
            distribution via ``get_weights``.

        Returns
        -------
        torch.Tensor
            Scalar tensor.

            * Uniform weights for every image (all equal 1/N):
              N_eff = num_shifts.
            * One dominant weight per image (-> 1): N_eff -> 1.

        Notes
        -----
        N_eff is a standard information-theoretic measure of
        distribution peakedness, analogous to the perplexity of a
        language model. It is useful for monitoring whether training
        is collapsing to a single shift position (per image, on
        average) or maintaining a spread distribution.

        Retains the autograd graph; can be used as a loss term.

        Examples
        --------
        >>> ccs = ConfidenceCycleSpinning(num_shifts=8, channels=1)
        >>> outs = [torch.randn(2, 1, 16, 16) for _ in range(8)]
        >>> confs = [torch.rand(2, 1, 16, 16) for _ in range(8)]
        >>> n_eff = ccs.effective_num_shifts(outs, confs)
        >>> bool(0.0 < n_eff.item() <= 8.0 + 1e-3)
        True
        """
        return torch.exp(self.entropy(outputs, confidence_maps))

    # ------------------------------------------------------------------
    # Weight variance
    # ------------------------------------------------------------------

    def weight_variance(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the variance of the predicted weight distribution.

        Computes the population variance (``unbiased=False``) over
        the flattened ``[B, num_shifts]`` weight tensor, i.e. across
        both the batch and shift dimensions jointly.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
            Together with ``outputs``, used to predict the weight
            distribution via ``get_weights``.

        Returns
        -------
        torch.Tensor
            Scalar tensor (population variance, ``unbiased=False``).
            Retains the autograd graph.

        Notes
        -----
        High variance indicates that the predicted weights are peaked
        (a few shifts dominate, for some or all images in the batch).
        Low variance indicates a near-uniform distribution. Useful for
        analysis and for constructing regularisation terms that
        penalise extreme peaking, and is the foundation for the
        variance-aware loss terms anticipated in A26f (Transformer
        Fusion).

        Examples
        --------
        >>> ccs = ConfidenceCycleSpinning(num_shifts=4, channels=1)
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> v = ccs.weight_variance(outs, confs)
        >>> v.shape
        torch.Size([])
        >>> bool(v.item() >= 0.0)
        True
        """
        weights: torch.Tensor = self.get_weights(outputs, confidence_maps)  # [B, N]
        return weights.var(unbiased=False)

    # ------------------------------------------------------------------
    # Index utilities
    # ------------------------------------------------------------------

    def max_weight_index(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Return, per batch element, the index of the highest-weight shift.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
            Together with ``outputs``, used to predict the weight
            distribution via ``get_weights``.

        Returns
        -------
        torch.Tensor
            Shape ``[B]``, dtype ``int64``. Entry ``b`` is the index
            ``i`` such that ``get_weights(outputs, confidence_maps)[b, i]``
            is maximal for that batch element. Detached from the
            autograd graph.

        Notes
        -----
        Returns one index *per image* in the batch, since A26c
        predicts a separate weight vector per image. Useful for
        visualising which shift position each individual SAR image
        relies on most, and for inspecting whether the model is in
        fact deferring to the supplied confidence signal.

        Examples
        --------
        >>> ccs = ConfidenceCycleSpinning(num_shifts=4, channels=1)
        >>> outs = [torch.randn(3, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(3, 1, 8, 8) for _ in range(4)]
        >>> idx = ccs.max_weight_index(outs, confs)
        >>> idx.shape
        torch.Size([3])
        """
        weights: torch.Tensor = self._weights_no_grad(outputs, confidence_maps)
        return weights.argmax(dim=1)

    def min_weight_index(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Return, per batch element, the index of the lowest-weight shift.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
            Together with ``outputs``, used to predict the weight
            distribution via ``get_weights``.

        Returns
        -------
        torch.Tensor
            Shape ``[B]``, dtype ``int64``. Entry ``b`` is the index
            ``i`` such that ``get_weights(outputs, confidence_maps)[b, i]``
            is minimal for that batch element. Detached from the
            autograd graph.

        Notes
        -----
        Complements ``max_weight_index()``. The pair
        ``(min_weight_index, max_weight_index)`` identifies, for every
        image in the batch independently, which shift positions the
        model trusts least and most — useful for per-image ablation
        studies and qualitative visualisation across a SAR scene
        dataset.

        Examples
        --------
        >>> ccs = ConfidenceCycleSpinning(num_shifts=4, channels=1)
        >>> outs = [torch.randn(3, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(3, 1, 8, 8) for _ in range(4)]
        >>> idx = ccs.min_weight_index(outs, confs)
        >>> idx.shape
        torch.Size([3])
        """
        weights: torch.Tensor = self._weights_no_grad(outputs, confidence_maps)
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

        Notes
        -----
        Centralising this construction avoids the repeated
        ``torch.full`` pattern that would otherwise appear in
        ``kl_to_uniform``, ``js_to_uniform``, and any future
        uniform-reference diagnostics (e.g. for A26d / A26f).

        Examples
        --------
        >>> ccs = ConfidenceCycleSpinning(num_shifts=4, channels=1)
        >>> u = ccs.uniform_weights(batch_size=2)
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
    ) -> torch.Tensor:
        """Compute the batch-averaged KL divergence to the uniform distribution.

        For each batch element ``b``::

            KL_b(w ‖ u) = Σ_i w_{b,i} • [log(w_{b,i}) - log(1/N)]
                        = Σ_i w_{b,i} • log(N • w_{b,i})

        where ``u_i = 1/N`` is the uniform distribution and
        ``w_{b,i} = get_weights(outputs, confidence_maps)[b, i]``.
        This method returns the mean of ``KL_b`` over the batch
        dimension.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
            Together with ``outputs``, used to predict the weight
            distribution via ``get_weights``.

        Returns
        -------
        torch.Tensor
            Scalar tensor >= 0. Zero iff every row of ``w`` is exactly
            uniform. Retains the autograd graph; can be added to a
            training loss to penalise deviation from equal-weight
            averaging.

        Notes
        -----
        ``KL_b(w ‖ u) = log(N) - H_b(w)``, so minimising the batch-
        averaged KL term is equivalent to maximising the batch-
        averaged entropy returned by ``entropy()``. The explicit form
        is provided because it gives a physically interpretable
        magnitude: it is approximately zero near initialisation and
        grows as the predicted weights become more confidence-specific
        and peaked. Anticipated for use by A26d (Wavelet-Guided Fusion)
        and A26f (Transformer Fusion) as a stabilising regulariser.

        Examples
        --------
        >>> ccs = ConfidenceCycleSpinning(num_shifts=4, channels=1)
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> kl = ccs.kl_to_uniform(outs, confs)
        >>> kl.shape
        torch.Size([])
        >>> bool(kl.item() >= -1e-6)
        True
        """
        weights: torch.Tensor = self.get_weights(outputs, confidence_maps)  # [B, N]
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
    ) -> torch.Tensor:
        """Compute the batch-averaged Jensen-Shannon divergence to uniform.

        The Jensen-Shannon divergence (JSD) is a symmetric, bounded
        alternative to KL divergence. For each batch element ``b``::

            M_b  = (w_b + u) / 2
            JSD_b(w_b ‖ u) = [KL(w_b ‖ M_b) + KL(u ‖ M_b)] / 2

        where ``u`` is the uniform distribution (1/N) and ``M_b`` is
        the per-image mixture. This method returns the mean of
        ``JSD_b`` over the batch dimension.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
            Together with ``outputs``, used to predict the weight
            distribution via ``get_weights``.

        Returns
        -------
        torch.Tensor
            Scalar tensor in ``[0, log(2)]``. Zero iff every row of
            ``w`` equals uniform (approximately true near
            initialisation). Retains the autograd graph.

        Notes
        -----
        Because JSD is symmetric and bounded regardless of ``N``, it
        produces cleaner training curves than KL divergence and is
        preferred for publication figures comparing aggregation
        strategies across configurations with different shift counts.

        Examples
        --------
        >>> ccs = ConfidenceCycleSpinning(num_shifts=4, channels=1)
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> jsd = ccs.js_to_uniform(outs, confs)
        >>> jsd.shape
        torch.Size([])
        >>> bool(jsd.item() >= -1e-6)
        True
        """
        weights: torch.Tensor = self.get_weights(outputs, confidence_maps)  # [B, N]
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

            * Decrease τ to sharpen the per-image weight distribution
              (encourage the model to specialise on fewer, higher-
              confidence shift positions per image).
            * Increase τ to flatten the distribution (encourage
              uniform averaging; at τ → ∞ this recovers the SAR-DDPM
              baseline for every image regardless of confidence).

        Raises
        ------
        ValueError
            If ``temperature`` is not strictly positive.

        Notes
        -----
        Temperature annealing is a common technique for controlling
        learning dynamics in softmax-parametrised models. This method
        is anticipated for use by A26f (Transformer Fusion), which is
        expected to schedule τ over the course of training (e.g.
        annealing from a high, exploratory temperature down to a
        sharper, more confidence-decisive one).

        Examples
        --------
        >>> ccs = ConfidenceCycleSpinning(num_shifts=4, channels=1, temperature=1.0)
        >>> ccs.set_temperature(0.5)
        >>> ccs.temperature
        0.5
        >>> try:
        ...     ccs.set_temperature(-1.0)
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
        parameters will receive gradients during backpropagation. The
        module still participates in the forward pass and produces
        valid per-image weights; only the parameter updates are
        suppressed.

        Useful for ablation studies where the confidence-guided
        aggregation should be held fixed at its current behaviour
        (e.g. after convergence, or when evaluating the near-uniform
        baseline by freezing immediately after initialisation).

        See Also
        --------
        unfreeze : Re-enable gradient updates.
        is_frozen : Query the current frozen state.

        Examples
        --------
        >>> ccs = ConfidenceCycleSpinning(num_shifts=4, channels=1)
        >>> ccs.freeze()
        >>> ccs.is_frozen()
        True
        """
        for param in self.weight_predictor.parameters():
            param.requires_grad_(False)

    def unfreeze(self) -> None:
        """Enable gradient updates for the entire weight-predictor MLP.

        Restores gradient computation for every parameter in
        ``weight_predictor`` after a prior call to ``freeze()``.
        Newly constructed modules have ``requires_grad=True`` by
        default; calling ``unfreeze()`` on them is a no-op.

        See Also
        --------
        freeze : Disable gradient updates.
        is_frozen : Query the current frozen state.

        Examples
        --------
        >>> ccs = ConfidenceCycleSpinning(num_shifts=4, channels=1)
        >>> ccs.freeze()
        >>> ccs.unfreeze()
        >>> ccs.is_frozen()
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
            parameters in ``weight_predictor``. If the module is in a
            mixed state (some parameters frozen, others not — which
            should not occur via ``freeze()``/``unfreeze()`` alone,
            but could result from manual external manipulation),
            this returns ``False``, since at least one parameter can
            still receive gradients.

        Notes
        -----
        Useful in ablation scripts to assert module state before
        training begins::

            assert not ccs.is_frozen(), "weight_predictor must be trainable"

        or to confirm that a baseline frozen run is correctly
        configured::

            ccs.freeze()
            assert ccs.is_frozen()

        See Also
        --------
        freeze : Disable gradient updates.
        unfreeze : Re-enable gradient updates.

        Examples
        --------
        >>> ccs = ConfidenceCycleSpinning(num_shifts=4, channels=1)
        >>> ccs.is_frozen()
        False
        >>> ccs.freeze()
        >>> ccs.is_frozen()
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
    ) -> torch.Tensor:
        """Compute per-image softmax weights without retaining the autograd graph.

        Used internally by logging and diagnostic methods
        (``weight_statistics``, ``summary``, ``save_statistics``,
        ``max_weight_index``, ``min_weight_index``) that need the
        weight values purely for inspection, not for gradient
        computation. A single call here avoids redundant forward
        passes through ``weight_predictor`` across multiple logging
        accessors.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, num_shifts]``, detached from the autograd graph.
        """
        with torch.no_grad():
            return self.get_weights(outputs, confidence_maps)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def weight_statistics(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
    ) -> Dict[str, float]:
        """Return useful statistics about the predicted weight distribution.

        All statistics are computed from a single ``get_weights()``
        call (under ``torch.no_grad()``) to avoid redundant forward
        passes, and are averaged or reduced over the batch dimension
        as noted below.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
            Together with ``outputs``, used to predict the weight
            distribution via ``get_weights``.

        Returns
        -------
        Dict[str, float]
            A dictionary with the following ``str`` keys and ``float``
            values:

            ``"entropy"``
                Batch-averaged Shannon entropy
                H = mean_b[-Σ_i w_{b,i} log(w_{b,i})].
            ``"effective_num_shifts"``
                N_eff = exp(H).
            ``"max_weight"``
                Maximum weight value across the whole ``[B, N]`` tensor.
            ``"min_weight"``
                Minimum weight value across the whole ``[B, N]`` tensor.
            ``"std_weight"``
                Standard deviation of all weight values (population
                std, ``unbiased=False``), computed over the flattened
                ``[B, N]`` tensor.

        Notes
        -----
        All values are detached scalars; this method is intended for
        logging and does not retain the autograd graph.

        Examples
        --------
        >>> ccs = ConfidenceCycleSpinning(num_shifts=4, channels=1)
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> stats = ccs.weight_statistics(outs, confs)
        >>> set(stats.keys()) == {
        ...     "entropy", "effective_num_shifts",
        ...     "max_weight", "min_weight", "std_weight"
        ... }
        True
        """
        weights: torch.Tensor = self._weights_no_grad(outputs, confidence_maps)
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
    ) -> Dict[str, float]:
        """Return a comprehensive diagnostic summary of the module's behaviour.

        Extends ``weight_statistics()`` with the batch-averaged
        weight variance for a more complete picture of the predicted
        distribution. All quantities are computed from a single
        ``get_weights()`` call.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
            Together with ``outputs``, used to predict the weight
            distribution via ``get_weights``.

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
                Population variance of all weight values, computed
                over the flattened ``[B, N]`` tensor.

        Notes
        -----
        All values are detached scalars; this method is intended for
        logging and does not retain the autograd graph.

        Examples
        --------
        >>> ccs = ConfidenceCycleSpinning(num_shifts=4, channels=1)
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> s = ccs.summary(outs, confs)
        >>> set(s.keys()) == {
        ...     "entropy", "effective_num_shifts",
        ...     "max_weight", "min_weight", "weight_variance"
        ... }
        True
        """
        weights: torch.Tensor = self._weights_no_grad(outputs, confidence_maps)
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
    ) -> Dict[str, float]:
        """Return a detached statistics snapshot for checkpoint logging.

        Intended to be stored alongside a model checkpoint so that the
        behaviour of the confidence-guided aggregation weights — for
        the particular ``outputs`` / ``confidence_maps`` batch used to
        call this method — can be inspected from the log without
        reloading the model and re-running inference.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
            Together with ``outputs``, used to predict the weight
            distribution via ``get_weights``.

        Returns
        -------
        Dict[str, float]
            A dictionary with the following keys:

            ``"entropy"``
                Batch-averaged Shannon entropy
                H = mean_b[-Σ_i w_{b,i} log(w_{b,i})].
            ``"effective_num_shifts"``
                N_eff = exp(H).
            ``"kl_to_uniform"``
                Batch-averaged KL(w ‖ uniform).
            ``"max_weight"``
                Maximum weight value across the whole ``[B, N]``
                tensor.
            ``"min_weight"``
                Minimum weight value across the whole ``[B, N]``
                tensor.
            ``"weight_variance"``
                Population variance of all weight values, computed
                over the flattened ``[B, N]`` tensor.
            ``"max_weight_index"``
                Index of the highest-weight shift for the *first*
                batch element (as a float, for JSON/CSV serialisation
                compatibility). For per-image indices across the full
                batch, call ``max_weight_index()`` directly.
            ``"min_weight_index"``
                Index of the lowest-weight shift for the *first*
                batch element (as a float). For per-image indices
                across the full batch, call ``min_weight_index()``
                directly.

        Notes
        -----
        All values are detached scalars. This method is intentionally
        separate from ``summary()`` so that logging code can call it
        without worrying about autograd side-effects. Because A26c's
        weights are batch-dependent, the index fields here summarise
        only the first batch element; richer per-image summaries
        should use ``max_weight_index()`` / ``min_weight_index()``
        directly and aggregate externally (e.g. for a histogram across
        a full test set).

        Examples
        --------
        >>> ccs = ConfidenceCycleSpinning(num_shifts=4, channels=1)
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> s = ccs.save_statistics(outs, confs)
        >>> set(s.keys()) == {
        ...     "entropy", "effective_num_shifts", "kl_to_uniform",
        ...     "max_weight", "min_weight", "weight_variance",
        ...     "max_weight_index", "min_weight_index",
        ... }
        True
        """
        weights: torch.Tensor = self._weights_no_grad(outputs, confidence_maps)
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
            Human-readable representation of the module's
            configuration, formatted to match the style of
            ``nn.Linear``, ``nn.Conv2d``, and the other encoders /
            aggregation modules in this codebase (cf.
            :class:`~structdiff.inference.adaptive_cycle_spinning.AdaptiveCycleSpinning`).

        Examples
        --------
        >>> ccs = ConfidenceCycleSpinning(
        ...     num_shifts=9, channels=1, hidden_dim=128,
        ...     temperature=0.5, pooling="avg",
        ... )
        >>> print(ccs)
        ConfidenceCycleSpinning(
          (pool): AdaptiveAvgPool2d(output_size=1)
          (weight_predictor): Sequential(
            (0): Linear(in_features=18, out_features=128, bias=True)
            (1): GELU(approximate='none')
            (2): Linear(in_features=128, out_features=9, bias=True)
          )
        )
        """
        return (
            f"num_shifts={self.num_shifts}, "
            f"channels={self.channels}, "
            f"hidden_dim={self.hidden_dim}, "
            f"temperature={self.temperature}, "
            f"pooling={self.pooling}"
        )
