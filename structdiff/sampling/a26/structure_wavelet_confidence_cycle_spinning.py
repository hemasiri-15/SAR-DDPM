"""
structdiff/inference/structure_wavelet_confidence_cycle_spinning.py
====================================================================
A26e: StructureWaveletConfidenceCycleSpinning — structure-tensor-guided,
wavelet-guided, confidence-adaptive softmax aggregation of cycle-shifted
diffusion outputs for SAR despeckling.

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
extended A26b by conditioning on per-shift **confidence maps** ``σ_i``::

    d_i = concat(GAP(x_i), GAP(σ_i))            d_i ∈ R^{B × (C+1)}

A26d (:class:`~structdiff.inference.wavelet_confidence_cycle_spinning.WaveletConfidenceCycleSpinning`)
further extended A26c by conditioning on per-shift **wavelet tensors**
``W_i`` (DWT subbands LL, LH, HL, HH from A12)::

    d_i = concat(GAP(x_i), GAP(σ_i), GAP(W_i)) d_i ∈ R^{B × (C+1+Cw)}

Structure-tensor-guided aggregation (A26e)
------------------------------------------
A26e extends A26d by additionally conditioning the weight predictor on
a per-shift **structure tensor descriptor** ``S_i``, a multi-channel
feature map derived from the structure tensor eigenvalue analysis
introduced in A10 (:mod:`structdiff.utils.structure_tensor_multiscale`)
and A11 (:class:`~structdiff.conditioning.tensor_spectral_encoder.TensorSpectralEncoder`).

The structure tensor at a spatial location encodes the local image
geometry: its eigenvalues ``λ1 ≥ λ2 ≥ 0`` and derived quantities
(anisotropy, coherence, orientation) reveal whether a patch lies in a
uniform region (λ1 ≈ λ2 ≈ 0), on an edge (λ1 >> λ2 ≈ 0), or at a
corner / junction (λ1 ≈ λ2 >> 0). By conditioning the aggregation MLP
on pooled structure tensor descriptors, the weight predictor can:

* Favour shifts whose denoised output better preserves the orientation
  and coherence of the dominant edges in the scene.
* Assign lower aggregation weight to shifts that smear or misalign
  sharp edges due to look-direction mismatch or azimuth spectral leakage.
* Respond differently to smooth uniform regions (coherence ≈ 1,
  anisotropy ≈ 0) versus strongly textured or edge-rich regions
  (anisotropy ≈ 1), where incorrect shift aggregation causes the most
  perceptible artefacts.

The structure tensor descriptors complement wavelet descriptors (which
are frequency-domain, global) with local, geometry-sensitive, scale-
aware statistics. Together, ``(x, σ, W, S)`` form the richest per-shift
feature set available prior to the transformer-based A26f stage.

Formally, for shift ``i`` with prediction ``x_i ∈ R^{B×C×H×W}``,
confidence map ``σ_i ∈ R^{B×1×H×W}``, wavelet tensor
``W_i ∈ R^{B×Cw×Hw×Ww}``, and structure tensor descriptor
``S_i ∈ R^{B×Cs×Hs×Ws}``::

    z_i = GAP(x_i)                              z_i ∈ R^{B × C}
    c_i = GAP(σ_i)                              c_i ∈ R^{B × 1}
    v_i = GAP(W_i)                              v_i ∈ R^{B × Cw}
    s_i = GAP(S_i)                              s_i ∈ R^{B × Cs}
    d_i = concat(z_i, c_i, v_i, s_i)           d_i ∈ R^{B × (C+1+Cw+Cs)}
    D   = concat(d_1, …, d_N)                   D   ∈ R^{B × N(C+1+Cw+Cs)}
    h   = GELU(Linear(D))                       h   ∈ R^{B × hidden_dim}
    a   = Linear(h)                              a   ∈ R^{B × N}
    w   = softmax(a / τ, dim=1)                 w   ∈ R^{B × N}
    x̂   = Σ_i w_i • x_i                         x̂   ∈ R^{B × C × H × W}

Note that the *fusion* itself still combines only the image content
``x_i`` (matching all prior A26 stages and the original SAR-DDPM
average); the confidence maps ``σ_i``, wavelet tensors ``W_i``, and
structure tensor descriptors ``S_i`` are used exclusively to help
*predict* the weights.

The structure channel count ``Cs`` is **not hardcoded** — it is inferred
from the ``structure_channels`` constructor argument, allowing the module
to work with any structure tensor configuration (e.g. 12 channels from
A11's three-scale eigenvalue decomposition, 3 channels from a single-
scale λ1/λ2/coherence triple, or any other layout produced upstream).

Initialization guarantee
-------------------------
As in A26b, A26c, and A26d, the final linear layer of the weight-predictor
MLP is initialised with very small weights (``std=1e-3``) and a zero bias.
Because the logits ``a`` are therefore approximately zero for any input
at step 0, the softmax output is approximately uniform::

    a ≈ 0  →  w_i ≈ 1/N  for all i

This means **at initialisation A26e approximately reproduces the
original SAR-DDPM equal-weight average** (and is approximately
equivalent to A26a–A26d's uniform initialisation), regardless of the
content of the supplied confidence maps, wavelet tensors, or structure
tensor descriptors.

Checkpoint compatibility
-------------------------
``StructureWaveletConfidenceCycleSpinning`` is a new module. Its
``weight_predictor`` has a different input dimensionality
(``N * (channels + 1 + wavelet_channels + structure_channels)``) than
A26d's (``N * (channels + 1 + wavelet_channels)``), so **A26d
checkpoints cannot be loaded directly into A26e's ``weight_predictor``
weights** — the shapes do not match. When migrating from an A26d run,
load everything except ``weight_predictor.*`` with::

    model.load_state_dict(checkpoint, strict=False)

and allow ``weight_predictor.*`` to keep its freshly initialised,
near-zero values, which — per the initialisation guarantee above —
still preserves approximately equal-weight averaging behaviour at the
start of A26e training.

Structure tensor motivation
----------------------------
The 2-D structure tensor of an image ``I`` at location ``(x, y)``
with Gaussian window ``w_σ`` is::

    J = w_σ ★ [∇I ∇I^T]
      = w_σ ★ [[I_x²,  I_x I_y],
                [I_x I_y, I_y²  ]]

Its eigenvalues ``λ1 ≥ λ2 ≥ 0`` encode the principal gradient energies
along the two local image directions, and derived features include:

* **Anisotropy** = (λ1 − λ2) / (λ1 + λ2 + ε): near 1 on edges,
  near 0 in uniform regions and at corners.
* **Coherence** = ((λ1 − λ2) / (λ1 + λ2 + ε))²: strongly positive
  where gradient direction is consistent within the window.
* **Orientation** = 0.5 · arctan2(2 I_x I_y, I_x² − I_y²): principal
  edge direction.

In SAR imagery, cycle-spin shifts alter the apparent phase of
directional structures (layover, shadow, azimuth compression artefacts)
in a shift-dependent way. Shifts misaligned with the dominant edge
direction may degrade coherence or rotate the apparent orientation of
the fused output. By conditioning on pooled structure tensor
descriptors, the aggregation network can learn to detect and penalise
such misalignments per image, per shift, and per scene type.

This is the last MLP-based fusion stage. A26f replaces the MLP with
a cross-attention transformer over the stack of per-shift descriptors.

Future roadmap
---------------
* **A26f** — Transformer Fusion: replace the MLP weight predictor with
  cross-attention over the stack of *N* per-shift descriptors
  ``(z_i, c_i, v_i, s_i)``, allowing each shift to attend to all
  others before its aggregation weight is predicted. This introduces
  global, shift-to-shift interactions not available in the MLP.
* **A26g** — Learnable Shift Coordinates: jointly learn the (row, col)
  shift grid rather than using a fixed uniform grid, feeding shift
  geometry as an additional conditioning signal alongside
  ``(x, σ, W, S)``.
* **A26h** — Hierarchical Cycle Spinning: nested coarse + fine shift
  pyramids with independent structure-wavelet-confidence weight
  predictors per level.
* **A26i** — Full Adaptive Cycle-Spinning Transformer: integrates
  A26f-h into a unified transformer-based aggregation module that
  consumes confidence, wavelet, structure-tensor, and shift-geometry
  features jointly across all scales.
* **A26j** — Bayesian Cycle Spinning: model per-image shift weights as
  a Dirichlet distribution and estimate uncertainty over the
  aggregation weights themselves, enabling principled confidence
  intervals over the fused prediction for journal-level uncertainty
  quantification in SAR despeckling.
* **A26k** — Meta-Learned Cycle Spinning: learn weight-prediction
  policies across datasets, allowing zero-shot transfer of the
  structure-wavelet-confidence aggregator to unseen SAR sensor
  configurations and domains.
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
:class:`~structdiff.inference.wavelet_confidence_cycle_spinning.WaveletConfidenceCycleSpinning`
exactly, with the sole functional difference being that every method
which predicts or consumes weights now also accepts
``structure_features`` alongside ``outputs``, ``confidence_maps``, and
``wavelet_features``. This keeps downstream logging, training, and
ablation code interchangeable across A26d and A26e, and ensures future
extensions (A26f-i) can inherit or compose from this module without
breaking changes.

References
----------
Coifman, R.R. & Donoho, D.L. (1995). Translation-Invariant
De-Noising. *Wavelets and Statistics*, Springer.

Mallat, S. (1999). *A Wavelet Tour of Signal Processing*. Academic Press.

Bigun, J. & Granlund, G.H. (1987). Optimal orientation detection of
linear symmetry. *Proc. ICCV*, 433–438.

Förstner, W. & Gülch, E. (1987). A fast operator for detection and
precise location of distinct points, corners and centres of circular
features. *Proc. ISPRS Intercommission Conf. on Fast Processing of
Photogrammetric Data*, 281–305.

Notes
-----
* All computation is performed in PyTorch; no NumPy, no CPU transfer,
  no in-place operations, full autograd support.
* The module is device-agnostic: all submodules move with
  ``model.to(device)``.
* Weights are predicted per batch element, so two images in the same
  batch may receive entirely different aggregation strategies, each
  informed by its own per-shift confidence maps, wavelet descriptors,
  and structure tensor features.
* Neither ``wavelet_channels`` nor ``structure_channels`` is hardcoded;
  any positive integer value is accepted for both, allowing full
  compatibility with any upstream DWT or structure-tensor configuration.
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
#: the A26a / A26b / A26c / A26d uniform initialisation.
_FINAL_LAYER_INIT_STD: float = 1e-3

#: Epsilon added inside the entropy logarithm for numerical stability.
#: Must satisfy _LOG_EPS << 1/N for any practical N.
_LOG_EPS: float = 1e-8

#: Number of channels expected in every confidence map (one scalar
#: confidence value per spatial location).
_CONFIDENCE_CHANNELS: int = 1


# ---------------------------------------------------------------------------
# StructureWaveletConfidenceCycleSpinning
# ---------------------------------------------------------------------------


class StructureWaveletConfidenceCycleSpinning(nn.Module):
    """Structure-tensor-guided, wavelet-guided, confidence-adaptive softmax
    aggregation of cycle-shifted diffusion outputs.

    Extends A26d
    (:class:`~structdiff.inference.wavelet_confidence_cycle_spinning.WaveletConfidenceCycleSpinning`)
    by additionally conditioning the per-image weight predictor on a
    per-shift structure tensor descriptor ``S_i``, alongside the shifted
    prediction content, confidence maps, and wavelet tensors already
    introduced in A26c and A26d.

    For each of the *N* shifted predictions, four descriptors are pooled
    and concatenated:

    * ``z_i = GAP(x_i)``  — image content descriptor    ``[B, C]``
    * ``c_i = GAP(σ_i)``  — confidence descriptor        ``[B, 1]``
    * ``v_i = GAP(W_i)``  — wavelet descriptor            ``[B, Cw]``
    * ``s_i = GAP(S_i)``  — structure tensor descriptor   ``[B, Cs]``

    Per-shift concatenation: ``d_i = [z_i, c_i, v_i, s_i] ∈ R^{B×(C+1+Cw+Cs)}``.
    All *N* per-shift descriptors are concatenated to
    ``D ∈ R^{B × N(C+1+Cw+Cs)}``, and a two-layer MLP maps this to
    per-shift logits ``[B, N]``. A temperature-scaled softmax over the
    shift dimension then yields per-image aggregation weights that sum
    to 1 across shifts for every batch element.

    At construction, the final linear layer of the MLP is initialised
    with very small weights and zero bias, so the predicted logits
    start near zero and the softmax output starts near-uniform —
    closely matching the original SAR-DDPM equal-weight average and the
    A26a / A26b / A26c / A26d uniform initialisation, regardless of the
    content of the supplied confidence maps, wavelet tensors, or
    structure tensor descriptors.

    Parameters
    ----------
    num_shifts:
        Total number of cycle-spin shifts *N*. Must be a positive
        integer. Corresponds to the number of (row, col) shift pairs
        in the nested loop of the existing SAR-DDPM inference code.
    channels:
        Number of channels *C* in each shifted prediction tensor.
        Must be a positive integer.
    wavelet_channels:
        Number of channels *Cw* in each wavelet tensor. Must be a
        positive integer. Inferred from the A12 DWT configuration
        (e.g. 4 for a single-level transform). NOT hardcoded.
    structure_channels:
        Number of channels *Cs* in each structure tensor descriptor.
        Must be a positive integer. Inferred from the A10/A11
        configuration (e.g. 12 for four features × three scales from
        A11, or 3 for a single-scale λ1/λ2/coherence triple).
        NOT hardcoded.
    hidden_dim:
        Width of the MLP's hidden layer. Must be a positive integer.
        Default 128.
    temperature:
        Softmax temperature τ > 0 applied to the predicted logits
        before the softmax. Lower values sharpen the per-image weight
        distribution; higher values flatten it. Default 1.0.
    pooling:
        Spatial pooling mode used to compute all four per-shift
        descriptors. One of:

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
    structure_channels : int
        Number of channels expected in each structure tensor descriptor.
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
        ``AdaptiveMaxPool2d(1)``), shared by all four descriptor
        branches.
    weight_predictor : nn.Sequential
        The two-layer MLP (``Linear`` → ``GELU`` → ``Linear``) that
        maps concatenated pooled descriptors to per-shift logits.

    Examples
    --------
    >>> import torch
    >>> from structdiff.inference.structure_wavelet_confidence_cycle_spinning import (
    ...     StructureWaveletConfidenceCycleSpinning,
    ... )
    >>> swccs = StructureWaveletConfidenceCycleSpinning(
    ...     num_shifts=9, channels=1, wavelet_channels=4, structure_channels=12
    ... )
    >>> swccs.num_shifts
    9
    >>> outputs = [torch.randn(2, 1, 64, 64) for _ in range(9)]
    >>> confidence_maps = [torch.rand(2, 1, 64, 64) for _ in range(9)]
    >>> wavelet_features = [torch.randn(2, 4, 32, 32) for _ in range(9)]
    >>> structure_features = [torch.randn(2, 12, 64, 64) for _ in range(9)]
    >>> fused = swccs(outputs, confidence_maps, wavelet_features, structure_features)
    >>> fused.shape
    torch.Size([2, 1, 64, 64])

    >>> # return_weights=True yields per-image weights
    >>> fused, weights = swccs(
    ...     outputs, confidence_maps, wavelet_features, structure_features,
    ...     return_weights=True
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
        structure_channels: int,
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
        if not isinstance(structure_channels, int) or structure_channels <= 0:
            raise ValueError(
                f"structure_channels must be a positive integer, "
                f"got {structure_channels!r}."
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
        self.structure_channels: int = structure_channels
        self.hidden_dim: int = hidden_dim
        self.temperature: float = temperature
        self.pooling: str = pooling
        self.eps: float = eps

        # ----------------------------------------------------------------
        # Spatial pooling layer
        #
        # Shared by all four descriptor branches:
        #   - Image:     [B, C,  H,  W]  -> [B, C,  1, 1]
        #   - Confidence:[B, 1,  H,  W]  -> [B, 1,  1, 1]
        #   - Wavelet:   [B, Cw, Hw, Ww] -> [B, Cw, 1, 1]
        #   - Structure: [B, Cs, Hs, Ws] -> [B, Cs, 1, 1]
        # AdaptiveAvgPool2d(1) / AdaptiveMaxPool2d(1) handles any
        # spatial resolution, so wavelet and structure tensors need not
        # share spatial dimensions with the SAR image outputs.
        # ----------------------------------------------------------------
        self.pool: nn.Module
        if pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:  # pooling == "max"
            self.pool = nn.AdaptiveMaxPool2d(1)

        # ----------------------------------------------------------------
        # Weight-predictor MLP
        #
        # Input : [B, num_shifts * (channels + 1 + wavelet_channels
        #                           + structure_channels)]
        # Hidden: [B, hidden_dim]
        # Output: [B, num_shifts]              (per-shift logits)
        # ----------------------------------------------------------------
        mlp_input_dim: int = num_shifts * (
            channels
            + _CONFIDENCE_CHANNELS
            + wavelet_channels
            + structure_channels
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
        (image content, confidence map, wavelet tensor, or structure
        tensor descriptor), so::

            w = softmax(a / τ) ≈ [1/N, …, 1/N]

        reproducing the original SAR-DDPM equal-weight average and the
        A26a / A26b / A26c / A26d uniform initialisation at step 0.

        Examples
        --------
        >>> swccs = StructureWaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1,
        ...     wavelet_channels=4, structure_channels=12
        ... )
        >>> swccs.reset_parameters()
        >>> first_linear = swccs.weight_predictor[0]
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
        # A26c / A26d baseline regardless of any conditioning signal.
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
            The first tensor in ``outputs`` (the reference tensor).

        Raises
        ------
        ValueError
            If any of the checks above fail.
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
            Validated reference tensor from ``_validate_outputs``.

        Raises
        ------
        ValueError
            If any of the checks above fail.
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
          ``reference`` (spatial dimensions may differ).
        * All wavelet tensors in the sequence share the same shape.

        Parameters
        ----------
        wavelet_features:
            Candidate sequence of per-shift wavelet tensors.
        reference:
            Validated reference tensor from ``_validate_outputs``.

        Raises
        ------
        ValueError
            If any of the checks above fail.
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

    def _validate_structure_features(
        self,
        structure_features: Sequence[torch.Tensor],
        reference: torch.Tensor,
    ) -> None:
        """Validate the ``structure_features`` sequence against the reference output.

        Checks performed (in order):

        * ``structure_features`` is non-empty.
        * ``len(structure_features) == self.num_shifts``.
        * Every structure tensor is 4-dimensional ``[B, Cs, Hs, Ws]``.
        * Every structure tensor has exactly ``self.structure_channels``
          channels.
        * Every structure tensor's batch size, dtype, and device match
          ``reference`` (spatial dimensions may differ from those of
          ``outputs`` — structure tensors may be at the same resolution
          as the SAR image, at half resolution, or multi-scale).
        * All structure tensors in the sequence share the same shape.

        Parameters
        ----------
        structure_features:
            Candidate sequence of per-shift structure tensor descriptors,
            each ``[B, Cs, Hs, Ws]``. Spatial dimensions ``(Hs, Ws)``
            need not match those of ``outputs``; they may equal
            ``(H, W)`` (same resolution) or differ if structure tensors
            are computed at a downsampled scale.
        reference:
            Validated reference tensor from ``_validate_outputs``, used
            as the ground truth for batch size, dtype, and device.

        Raises
        ------
        ValueError
            If any of the checks above fail, with a descriptive message
            identifying which structure tensor and which property
            triggered the failure.
        """
        if len(structure_features) == 0:
            raise ValueError(
                "structure_features must be a non-empty sequence of tensors, "
                "got length 0."
            )
        if len(structure_features) != self.num_shifts:
            raise ValueError(
                f"len(structure_features) must equal "
                f"num_shifts={self.num_shifts}, "
                f"got {len(structure_features)}."
            )

        ref_batch_size: int = reference.shape[0]
        ref_dtype: torch.dtype = reference.dtype
        ref_device: torch.device = reference.device

        first_st: torch.Tensor = structure_features[0]

        if first_st.ndim != 4:
            raise ValueError(
                f"Each structure tensor must be 4-dimensional "
                f"[B, Cs, Hs, Ws]; structure_features[0] has shape "
                f"{first_st.shape} (ndim={first_st.ndim})."
            )
        if first_st.shape[1] != self.structure_channels:
            raise ValueError(
                f"Each structure tensor must have structure_channels="
                f"{self.structure_channels}; structure_features[0] has "
                f"{first_st.shape[1]} channels."
            )
        if first_st.shape[0] != ref_batch_size:
            raise ValueError(
                f"Each structure tensor must have the same batch size as "
                f"outputs; outputs[0].shape[0]={ref_batch_size} but "
                f"structure_features[0].shape[0]={first_st.shape[0]}."
            )
        if first_st.dtype != ref_dtype:
            raise ValueError(
                f"Each structure tensor must have the same dtype as outputs; "
                f"outputs[0].dtype={ref_dtype} but "
                f"structure_features[0].dtype={first_st.dtype}."
            )
        if first_st.device != ref_device:
            raise ValueError(
                f"Each structure tensor must reside on the same device as "
                f"outputs; outputs[0].device={ref_device} but "
                f"structure_features[0].device={first_st.device}."
            )

        st_ref_shape: torch.Size = first_st.shape

        for idx, st in enumerate(structure_features[1:], start=1):
            if st.ndim != 4:
                raise ValueError(
                    f"Each structure tensor must be 4-dimensional "
                    f"[B, Cs, Hs, Ws]; structure_features[{idx}] has shape "
                    f"{st.shape} (ndim={st.ndim})."
                )
            if st.shape != st_ref_shape:
                raise ValueError(
                    f"All structure tensors must have the same shape; "
                    f"structure_features[0].shape={st_ref_shape} but "
                    f"structure_features[{idx}].shape={st.shape}."
                )
            if st.dtype != ref_dtype:
                raise ValueError(
                    f"All structure tensors must have the same dtype as "
                    f"outputs; outputs[0].dtype={ref_dtype} but "
                    f"structure_features[{idx}].dtype={st.dtype}."
                )
            if st.device != ref_device:
                raise ValueError(
                    f"All structure tensors must reside on the same device "
                    f"as outputs; outputs[0].device={ref_device} but "
                    f"structure_features[{idx}].device={st.device}."
                )

    # ------------------------------------------------------------------
    # Descriptor extraction
    # ------------------------------------------------------------------

    def _extract_descriptors(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
        structure_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute and concatenate pooled per-shift descriptors from all four branches.

        For each shift ``i``:

        * Image tensor ``x_i``  ``[B, C, H, W]``   pooled to ``[B, C]``.
        * Confidence map ``σ_i`` ``[B, 1, H, W]``   pooled to ``[B, 1]``.
        * Wavelet tensor ``W_i`` ``[B, Cw, Hw, Ww]`` pooled to ``[B, Cw]``.
        * Structure tensor ``S_i`` ``[B, Cs, Hs, Ws]`` pooled to ``[B, Cs]``.

        All four are concatenated to ``[B, C+1+Cw+Cs]`` and the *N*
        per-shift descriptors are concatenated to
        ``[B, N * (C+1+Cw+Cs)]``.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``. Pre-validated.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``. Pre-validated.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``. Pre-validated.
        structure_features:
            Sequence of *N* tensors, each ``[B, Cs, Hs, Ws]``. Pre-validated.

        Returns
        -------
        torch.Tensor
            Concatenated descriptor of shape ``[B, N * (C + 1 + Cw + Cs)]``.
        """
        descriptors: List[torch.Tensor] = []
        for tensor, conf, wav, st in zip(
            outputs, confidence_maps, wavelet_features, structure_features
        ):
            pooled_image: torch.Tensor = self.pool(tensor)   # [B, C,  1, 1]
            pooled_conf: torch.Tensor = self.pool(conf)      # [B, 1,  1, 1]
            pooled_wav: torch.Tensor = self.pool(wav)        # [B, Cw, 1, 1]
            pooled_st: torch.Tensor = self.pool(st)          # [B, Cs, 1, 1]

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
            st_descriptor: torch.Tensor = pooled_st.reshape(
                batch_size, -1
            )  # [B, Cs]

            descriptors.append(
                torch.cat(
                    [image_descriptor, conf_descriptor, wav_descriptor, st_descriptor],
                    dim=1,
                )
            )  # [B, C+1+Cw+Cs]

        return torch.cat(descriptors, dim=1)  # [B, N*(C+1+Cw+Cs)]

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def get_weights(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
        structure_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Predict per-image softmax aggregation weights from all four conditioning signals.

        Computes pooled image, confidence, wavelet, and structure tensor
        descriptors for every shift, concatenates them per shift and
        then across shifts, passes the result through ``weight_predictor``
        to obtain per-shift logits, and applies a temperature-scaled
        softmax over the shift dimension.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.
        structure_features:
            Sequence of *N* tensors, each ``[B, Cs, Hs, Ws]``,
            providing per-shift structure tensor descriptors (e.g.
            λ1, λ2, anisotropy, coherence from A10/A11).

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

        Examples
        --------
        >>> swccs = StructureWaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1,
        ...     wavelet_channels=4, structure_channels=12
        ... )
        >>> outs = [torch.randn(3, 1, 16, 16) for _ in range(4)]
        >>> confs = [torch.rand(3, 1, 16, 16) for _ in range(4)]
        >>> wavs = [torch.randn(3, 4, 8, 8) for _ in range(4)]
        >>> sts = [torch.randn(3, 12, 16, 16) for _ in range(4)]
        >>> w = swccs.get_weights(outs, confs, wavs, sts)
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
        self._validate_structure_features(structure_features, reference)

        descriptors: torch.Tensor = self._extract_descriptors(
            outputs, confidence_maps, wavelet_features, structure_features
        )  # [B, N*(C+1+Cw+Cs)]

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
        structure_features: Sequence[torch.Tensor],
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Aggregate cycle-shifted diffusion outputs with structure-wavelet-confidence-guided weights.

        Parameters
        ----------
        outputs:
            A sequence of *N* tensors, one per cycle-spin shift. Each
            tensor must have shape ``[B, C, H, W]``, with identical
            shape, dtype, device, and channel dimension equal to
            ``self.channels``.
        confidence_maps:
            A sequence of *N* tensors, one per cycle-spin shift, each
            ``[B, 1, H, W]``. Must match ``outputs`` in batch size,
            spatial dimensions, dtype, and device.
        wavelet_features:
            A sequence of *N* tensors, one per cycle-spin shift, each
            ``[B, Cw, Hw, Ww]``. Must match ``outputs`` in batch size,
            dtype, and device. Spatial dimensions ``(Hw, Ww)`` may
            differ from ``(H, W)``.
        structure_features:
            A sequence of *N* tensors, one per cycle-spin shift, each
            ``[B, Cs, Hs, Ws]``, providing per-shift structure tensor
            descriptors derived from A10/A11 (e.g. eigenvalues λ1, λ2,
            anisotropy, coherence, orientation at one or more scales).
            Must match ``outputs`` in batch size, dtype, and device.
            Spatial dimensions ``(Hs, Ws)`` may differ from ``(H, W)``.
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
            If any input sequence is empty or has the wrong length.
        ValueError
            If any output tensor has wrong dimensionality or channel count.
        ValueError
            If any confidence map has wrong dimensionality, channel count,
            spatial dimensions, dtype, or device.
        ValueError
            If any wavelet tensor has wrong dimensionality, channel count,
            dtype, or device.
        ValueError
            If any structure tensor has wrong dimensionality, channel
            count, dtype, or device.

        Notes
        -----
        The aggregation is::

            z_i = GAP_or_GMP(x_i)                        [B, C]
            c_i = GAP_or_GMP(σ_i)                         [B, 1]
            v_i = GAP_or_GMP(W_i)                         [B, Cw]
            s_i = GAP_or_GMP(S_i)                         [B, Cs]
            d_i = concat(z_i, c_i, v_i, s_i)              [B, C+1+Cw+Cs]
            D   = concat(d_1, …, d_N)                      [B, N*(C+1+Cw+Cs)]
            a   = weight_predictor(D)                      [B, N]
            w   = softmax(a / temperature, dim=1)          [B, N]
            x̂   = Σ_i w_i • x_i                            [B, C, H, W]

        Only the image content ``x_i`` is summed into the fused output.

        Examples
        --------
        >>> import torch
        >>> from structdiff.inference.structure_wavelet_confidence_cycle_spinning import (
        ...     StructureWaveletConfidenceCycleSpinning,
        ... )
        >>> swccs = StructureWaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1,
        ...     wavelet_channels=4, structure_channels=12
        ... )
        >>> outputs = [torch.ones(2, 1, 8, 8) * (i + 1.0) for i in range(4)]
        >>> confidence_maps = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavelet_features = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> structure_features = [torch.randn(2, 12, 8, 8) for _ in range(4)]
        >>> fused = swccs(outputs, confidence_maps, wavelet_features, structure_features)
        >>> fused.shape
        torch.Size([2, 1, 8, 8])

        >>> fused, w = swccs(
        ...     outputs, confidence_maps, wavelet_features, structure_features,
        ...     return_weights=True
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
        self._validate_structure_features(structure_features, reference)
        batch_size: int = reference.shape[0]

        # ----------------------------------------------------------------
        # Compute per-image softmax weights  [B, N]
        # ----------------------------------------------------------------
        weights: torch.Tensor = self.get_weights(
            outputs, confidence_maps, wavelet_features, structure_features
        )

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
        stacked = stacked.permute(1, 0, 2, 3, 4)                   # [B, N, C, H, W]

        weights_broadcast: torch.Tensor = weights.view(
            batch_size, self.num_shifts, 1, 1, 1
        )  # [B, N, 1, 1, 1]

        # Promote stacked dtype to match weights if necessary (e.g. fp16
        # inputs with fp32 weight predictor). The original input dtype is
        # saved and restored so that fp16 callers receive fp16 output,
        # preserving full compatibility with FP16 training.
        input_dtype: torch.dtype = stacked.dtype
        if stacked.dtype != weights_broadcast.dtype:
            stacked = stacked.to(weights_broadcast.dtype)

        fused: torch.Tensor = (stacked * weights_broadcast).sum(dim=1)  # [B, C, H, W]

        # Restore the caller's original dtype (e.g. fp16 -> fp16).
        fused = fused.to(input_dtype)

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
        structure_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the batch-averaged Shannon entropy of the predicted weights.

        For each batch element ``b``, the per-image entropy is::

            H_b = -Σ_i w_{b,i} • log(w_{b,i} + eps)

        Returns the mean of ``H_b`` over the batch dimension.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.
        structure_features:
            Sequence of *N* tensors, each ``[B, Cs, Hs, Ws]``.

        Returns
        -------
        torch.Tensor
            Scalar tensor. Retains the autograd graph.

        Examples
        --------
        >>> swccs = StructureWaveletConfidenceCycleSpinning(
        ...     num_shifts=8, channels=1,
        ...     wavelet_channels=4, structure_channels=12
        ... )
        >>> outs = [torch.randn(2, 1, 16, 16) for _ in range(8)]
        >>> confs = [torch.rand(2, 1, 16, 16) for _ in range(8)]
        >>> wavs = [torch.randn(2, 4, 8, 8) for _ in range(8)]
        >>> sts = [torch.randn(2, 12, 16, 16) for _ in range(8)]
        >>> h = swccs.entropy(outs, confs, wavs, sts)
        >>> bool(h.item() > 0.0)
        True
        """
        weights: torch.Tensor = self.get_weights(
            outputs, confidence_maps, wavelet_features, structure_features
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
        structure_features: Sequence[torch.Tensor],
        coefficient: float = 1.0,
    ) -> torch.Tensor:
        """Entropy regularization term for use directly in a training loss.

        Returns ``coefficient * H`` where H is the batch-averaged
        Shannon entropy of the predicted weight distributions.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.
        structure_features:
            Sequence of *N* tensors, each ``[B, Cs, Hs, Ws]``.
        coefficient:
            Scalar multiplier (positive → encourage uniform weights;
            negative → encourage sparse, geometry-specific weights).
            Default 1.0.

        Returns
        -------
        torch.Tensor
            Scalar tensor, retains the autograd graph.

        Examples
        --------
        >>> swccs = StructureWaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1,
        ...     wavelet_channels=4, structure_channels=12
        ... )
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(2, 12, 8, 8) for _ in range(4)]
        >>> reg = swccs.entropy_regularizer(outs, confs, wavs, sts, coefficient=0.01)
        >>> reg.shape
        torch.Size([])
        """
        return coefficient * self.entropy(
            outputs, confidence_maps, wavelet_features, structure_features
        )

    # ------------------------------------------------------------------
    # Effective number of shifts
    # ------------------------------------------------------------------

    def effective_num_shifts(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
        structure_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the batch-averaged effective number of active shifts (exp(H)).

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.
        structure_features:
            Sequence of *N* tensors, each ``[B, Cs, Hs, Ws]``.

        Returns
        -------
        torch.Tensor
            Scalar tensor in ``(0, num_shifts]``.

        Examples
        --------
        >>> swccs = StructureWaveletConfidenceCycleSpinning(
        ...     num_shifts=8, channels=1,
        ...     wavelet_channels=4, structure_channels=12
        ... )
        >>> outs = [torch.randn(2, 1, 16, 16) for _ in range(8)]
        >>> confs = [torch.rand(2, 1, 16, 16) for _ in range(8)]
        >>> wavs = [torch.randn(2, 4, 8, 8) for _ in range(8)]
        >>> sts = [torch.randn(2, 12, 16, 16) for _ in range(8)]
        >>> n_eff = swccs.effective_num_shifts(outs, confs, wavs, sts)
        >>> bool(0.0 < n_eff.item() <= 8.0 + 1e-3)
        True
        """
        return torch.exp(
            self.entropy(outputs, confidence_maps, wavelet_features, structure_features)
        )

    # ------------------------------------------------------------------
    # Weight variance
    # ------------------------------------------------------------------

    def weight_variance(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
        structure_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the population variance of the predicted weight distribution.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.
        structure_features:
            Sequence of *N* tensors, each ``[B, Cs, Hs, Ws]``.

        Returns
        -------
        torch.Tensor
            Scalar tensor (population variance, ``unbiased=False``).
            Retains the autograd graph.

        Examples
        --------
        >>> swccs = StructureWaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1,
        ...     wavelet_channels=4, structure_channels=12
        ... )
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(2, 12, 8, 8) for _ in range(4)]
        >>> v = swccs.weight_variance(outs, confs, wavs, sts)
        >>> v.shape
        torch.Size([])
        >>> bool(v.item() >= 0.0)
        True
        """
        weights: torch.Tensor = self.get_weights(
            outputs, confidence_maps, wavelet_features, structure_features
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
        structure_features: Sequence[torch.Tensor],
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
        structure_features:
            Sequence of *N* tensors, each ``[B, Cs, Hs, Ws]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B]``, dtype ``int64``. Detached from autograd graph.

        Examples
        --------
        >>> swccs = StructureWaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1,
        ...     wavelet_channels=4, structure_channels=12
        ... )
        >>> outs = [torch.randn(3, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(3, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(3, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(3, 12, 8, 8) for _ in range(4)]
        >>> idx = swccs.max_weight_index(outs, confs, wavs, sts)
        >>> idx.shape
        torch.Size([3])
        """
        weights: torch.Tensor = self._weights_no_grad(
            outputs, confidence_maps, wavelet_features, structure_features
        )
        return weights.argmax(dim=1)

    def min_weight_index(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
        structure_features: Sequence[torch.Tensor],
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
        structure_features:
            Sequence of *N* tensors, each ``[B, Cs, Hs, Ws]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B]``, dtype ``int64``. Detached from autograd graph.

        Examples
        --------
        >>> swccs = StructureWaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1,
        ...     wavelet_channels=4, structure_channels=12
        ... )
        >>> outs = [torch.randn(3, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(3, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(3, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(3, 12, 8, 8) for _ in range(4)]
        >>> idx = swccs.min_weight_index(outs, confs, wavs, sts)
        >>> idx.shape
        torch.Size([3])
        """
        weights: torch.Tensor = self._weights_no_grad(
            outputs, confidence_maps, wavelet_features, structure_features
        )
        return weights.argmin(dim=1)

    # ------------------------------------------------------------------
    # Uniform reference distribution
    # ------------------------------------------------------------------

    def uniform_weights(self, batch_size: int) -> torch.Tensor:
        """Return the uniform weight matrix ``1/N`` for a given batch size.

        Parameters
        ----------
        batch_size:
            Number of rows ``B`` in the returned tensor. Must be a
            positive integer.

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
        >>> swccs = StructureWaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1,
        ...     wavelet_channels=4, structure_channels=12
        ... )
        >>> u = swccs.uniform_weights(batch_size=2)
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
        structure_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the batch-averaged KL divergence KL(w ‖ uniform).

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.
        structure_features:
            Sequence of *N* tensors, each ``[B, Cs, Hs, Ws]``.

        Returns
        -------
        torch.Tensor
            Scalar tensor >= 0. Retains the autograd graph.

        Examples
        --------
        >>> swccs = StructureWaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1,
        ...     wavelet_channels=4, structure_channels=12
        ... )
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(2, 12, 8, 8) for _ in range(4)]
        >>> kl = swccs.kl_to_uniform(outs, confs, wavs, sts)
        >>> kl.shape
        torch.Size([])
        >>> bool(kl.item() >= -1e-6)
        True
        """
        weights: torch.Tensor = self.get_weights(
            outputs, confidence_maps, wavelet_features, structure_features
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
        structure_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the batch-averaged Jensen-Shannon divergence JSD(w ‖ uniform).

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.
        structure_features:
            Sequence of *N* tensors, each ``[B, Cs, Hs, Ws]``.

        Returns
        -------
        torch.Tensor
            Scalar tensor in ``[0, log(2)]``. Retains the autograd graph.

        Examples
        --------
        >>> swccs = StructureWaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1,
        ...     wavelet_channels=4, structure_channels=12
        ... )
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(2, 12, 8, 8) for _ in range(4)]
        >>> jsd = swccs.js_to_uniform(outs, confs, wavs, sts)
        >>> jsd.shape
        torch.Size([])
        >>> bool(jsd.item() >= -1e-6)
        True
        """
        weights: torch.Tensor = self.get_weights(
            outputs, confidence_maps, wavelet_features, structure_features
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
        >>> swccs = StructureWaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1,
        ...     wavelet_channels=4, structure_channels=12, temperature=1.0
        ... )
        >>> swccs.set_temperature(0.5)
        >>> swccs.temperature
        0.5
        >>> try:
        ...     swccs.set_temperature(-1.0)
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

        Examples
        --------
        >>> swccs = StructureWaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1,
        ...     wavelet_channels=4, structure_channels=12
        ... )
        >>> swccs.freeze()
        >>> swccs.is_frozen()
        True
        """
        for param in self.weight_predictor.parameters():
            param.requires_grad_(False)

    def unfreeze(self) -> None:
        """Enable gradient updates for the entire weight-predictor MLP.

        Examples
        --------
        >>> swccs = StructureWaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1,
        ...     wavelet_channels=4, structure_channels=12
        ... )
        >>> swccs.freeze()
        >>> swccs.unfreeze()
        >>> swccs.is_frozen()
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
        >>> swccs = StructureWaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1,
        ...     wavelet_channels=4, structure_channels=12
        ... )
        >>> swccs.is_frozen()
        False
        >>> swccs.freeze()
        >>> swccs.is_frozen()
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
        structure_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-image softmax weights without retaining the autograd graph.

        Used internally by all logging and diagnostic methods.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.
        structure_features:
            Sequence of *N* tensors, each ``[B, Cs, Hs, Ws]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, num_shifts]``, detached from the autograd graph.
        """
        with torch.no_grad():
            return self.get_weights(
                outputs, confidence_maps, wavelet_features, structure_features
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def weight_statistics(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
        structure_features: Sequence[torch.Tensor],
    ) -> Dict[str, float]:
        """Return useful statistics about the predicted weight distribution.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.
        structure_features:
            Sequence of *N* tensors, each ``[B, Cs, Hs, Ws]``.

        Returns
        -------
        Dict[str, float]
            Keys: ``"entropy"``, ``"effective_num_shifts"``,
            ``"max_weight"``, ``"min_weight"``, ``"std_weight"``.

        Examples
        --------
        >>> swccs = StructureWaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1,
        ...     wavelet_channels=4, structure_channels=12
        ... )
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(2, 12, 8, 8) for _ in range(4)]
        >>> stats = swccs.weight_statistics(outs, confs, wavs, sts)
        >>> set(stats.keys()) == {
        ...     "entropy", "effective_num_shifts",
        ...     "max_weight", "min_weight", "std_weight"
        ... }
        True
        """
        weights: torch.Tensor = self._weights_no_grad(
            outputs, confidence_maps, wavelet_features, structure_features
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
        structure_features: Sequence[torch.Tensor],
    ) -> Dict[str, float]:
        """Return a comprehensive diagnostic summary of the module's behaviour.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.
        structure_features:
            Sequence of *N* tensors, each ``[B, Cs, Hs, Ws]``.

        Returns
        -------
        Dict[str, float]
            Keys: ``"entropy"``, ``"effective_num_shifts"``,
            ``"max_weight"``, ``"min_weight"``, ``"weight_variance"``.

        Examples
        --------
        >>> swccs = StructureWaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1,
        ...     wavelet_channels=4, structure_channels=12
        ... )
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(2, 12, 8, 8) for _ in range(4)]
        >>> s = swccs.summary(outs, confs, wavs, sts)
        >>> set(s.keys()) == {
        ...     "entropy", "effective_num_shifts",
        ...     "max_weight", "min_weight", "weight_variance"
        ... }
        True
        """
        weights: torch.Tensor = self._weights_no_grad(
            outputs, confidence_maps, wavelet_features, structure_features
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
        structure_features: Sequence[torch.Tensor],
    ) -> Dict[str, float]:
        """Return a detached statistics snapshot for checkpoint logging.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``.
        structure_features:
            Sequence of *N* tensors, each ``[B, Cs, Hs, Ws]``.

        Returns
        -------
        Dict[str, float]
            Keys: ``"entropy"``, ``"effective_num_shifts"``,
            ``"kl_to_uniform"``, ``"max_weight"``, ``"min_weight"``,
            ``"weight_variance"``, ``"max_weight_index"``,
            ``"min_weight_index"``.

        Examples
        --------
        >>> swccs = StructureWaveletConfidenceCycleSpinning(
        ...     num_shifts=4, channels=1,
        ...     wavelet_channels=4, structure_channels=12
        ... )
        >>> outs = [torch.randn(2, 1, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(2, 12, 8, 8) for _ in range(4)]
        >>> s = swccs.save_statistics(outs, confs, wavs, sts)
        >>> set(s.keys()) == {
        ...     "entropy", "effective_num_shifts", "kl_to_uniform",
        ...     "max_weight", "min_weight", "weight_variance",
        ...     "max_weight_index", "min_weight_index",
        ... }
        True
        """
        weights: torch.Tensor = self._weights_no_grad(
            outputs, confidence_maps, wavelet_features, structure_features
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
        >>> swccs = StructureWaveletConfidenceCycleSpinning(
        ...     num_shifts=9, channels=1, wavelet_channels=4,
        ...     structure_channels=12, hidden_dim=128,
        ...     temperature=0.5, pooling="avg",
        ... )
        >>> print(swccs)  # doctest: +ELLIPSIS
        StructureWaveletConfidenceCycleSpinning(
          ...
        )
        """
        return (
            f"num_shifts={self.num_shifts}, "
            f"channels={self.channels}, "
            f"wavelet_channels={self.wavelet_channels}, "
            f"structure_channels={self.structure_channels}, "
            f"hidden_dim={self.hidden_dim}, "
            f"temperature={self.temperature}, "
            f"pooling={self.pooling}"
        )
