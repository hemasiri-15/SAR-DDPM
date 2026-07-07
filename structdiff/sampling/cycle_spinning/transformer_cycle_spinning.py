"""
structdiff/inference/transformer_cycle_spinning.py
====================================================
A26f: TransformerCycleSpinning — cross-shift attention-based softmax
aggregation of cycle-shifted diffusion outputs for SAR despeckling.

Background
----------
The original SAR-DDPM cycle-spinning implementation (see
``inference_sar.py`` / ``inference_sar_unet.py``) applies a denoiser
to *N* shifted copies of the input, inverse-shifts each result, and
averages them with **fixed equal weights**::

    pred_tensor += (1.0 / N) * sample

A26a (:class:`~structdiff.inference.learnable_cycle_spinning.LearnableCycleSpinning`)
replaced these fixed coefficients with a single set of *global*
trainable scalar logits, shared across every image in the dataset.

A26b (:class:`~structdiff.inference.adaptive_cycle_spinning.AdaptiveCycleSpinning`)
made the weights *image-adaptive* by predicting them from pooled
descriptors of the shifted predictions via an MLP.

A26c (:class:`~structdiff.inference.confidence_cycle_spinning.ConfidenceCycleSpinning`)
extended A26b by additionally conditioning on per-shift confidence maps.

A26d (:class:`~structdiff.inference.wavelet_confidence_cycle_spinning.WaveletConfidenceCycleSpinning`)
extended A26c by conditioning on per-shift wavelet (DWT subband) tensors.

A26e (:class:`~structdiff.inference.structure_wavelet_confidence_cycle_spinning.StructureWaveletConfidenceCycleSpinning`)
extended A26d by conditioning on per-shift structure tensor descriptors
(eigenvalues λ1, λ2, anisotropy, coherence from A10/A11).

Transformer-based aggregation (A26f)
--------------------------------------
A26e, like all prior MLP-based stages, constructs the weight predictor
by concatenating all per-shift descriptors into a single flat vector
and feeding it through a two-layer MLP.  This design has a critical
limitation: **shifts cannot attend to one another**.  The predicted
importance of shift *i* is computed from the concatenated evidence
of all shifts in a flat, permutation-unaware way, and the MLP cannot
selectively compare pairs of shifts or reason about the geometry of
the full N-shift ensemble.

A26f replaces the MLP weight predictor with a **Transformer encoder**.
The shift ensemble is modelled as a sequence of *N* tokens, one per
shift, each token being the pooled four-branch descriptor
``d_i = [z_i, c_i, v_i, s_i] ∈ R^{B × D}`` introduced in A26e.
A learnable CLS token is prepended to the sequence.  After positional
embedding, the full ``[B, N+1, D]`` token matrix is processed by a
stack of TransformerEncoderLayers with self-attention, allowing every
shift token to **attend to every other shift token**.  The CLS token
representation is then passed through a two-layer MLP head to produce
per-shift logits ``[B, N]``, which are converted to aggregation weights
via temperature-scaled softmax.

The key insight is that SAR cycle-spinning produces shifts with
correlated artefacts: two shifts that are close in (row, col) space
may produce similar ghosting artefacts or similar speckle patterns,
while distant shifts may be complementary.  The Transformer can learn
these inter-shift relationships globally, whereas the MLP in A26e must
infer them implicitly from the flat concatenation.

Formally, for *N* shifts with predictions ``x_i``, confidence maps
``σ_i``, wavelet tensors ``W_i``, and structure tensor descriptors
``S_i``::

    z_i  = GAP(x_i)                          z_i  ∈ R^{B × C}
    c_i  = GAP(σ_i)                          c_i  ∈ R^{B × 1}
    v_i  = GAP(W_i)                          v_i  ∈ R^{B × Cw}
    s_i  = GAP(S_i)                          s_i  ∈ R^{B × Cs}
    d_i  = concat(z_i, c_i, v_i, s_i)        d_i  ∈ R^{B × D}
           where D = C + 1 + Cw + Cs

    T    = stack(d_1, …, d_N, dim=1)          T    ∈ R^{B × N × D}

    # Prepend CLS token
    cls  = cls_token.expand(B, 1, D)          cls  ∈ R^{B × 1 × D}
    tokens = concat([cls, T], dim=1)           tokens ∈ R^{B × (N+1) × D}

    # Positional embedding
    tokens = tokens + pos_embed               tokens ∈ R^{B × (N+1) × D}

    # Transformer encoder (num_layers layers, num_heads heads)
    encoded = TransformerEncoder(tokens)       encoded ∈ R^{B × (N+1) × D}

    # Extract CLS feature
    cls_feat = encoded[:, 0, :]               cls_feat ∈ R^{B × D}

    # MLP head
    h = GELU(Linear(cls_feat))                h ∈ R^{B × D}
    a = Linear(h)                             a ∈ R^{B × N}
    w = softmax(a / τ, dim=1)                 w ∈ R^{B × N}

    # Fusion (image content only)
    x̂ = Σ_i w_i • x_i                         x̂ ∈ R^{B × C × H × W}

The *fusion* still combines only the image content ``x_i``, exactly as
in A26a–A26e and the original SAR-DDPM average; the confidence maps,
wavelet tensors, and structure tensor descriptors are used solely to
construct the token sequence that is fed to the Transformer weight
predictor.

Why Transformer aggregation outperforms MLP aggregation
---------------------------------------------------------
1. **Cross-shift attention**: each shift token can query and attend to
   every other shift token before its aggregation logit is predicted.
   The Transformer can learn that two visually similar (correlated)
   shifts should share weight and that a high-confidence outlier shift
   should dominate.
2. **Permutation equivariance (broken by positional embedding)**: the
   Transformer without positional embedding is permutation-equivariant
   over shifts, a useful inductive bias.  Adding a learnable positional
   embedding allows the network to exploit shift-order information if
   it is informative.
3. **Depth with residual connections**: each TransformerEncoderLayer
   contains a self-attention sublayer and a feed-forward sublayer, both
   with residual connections and LayerNorm.  This provides a deeper,
   more expressive path from descriptors to logits than the two-layer
   MLP in A26e, while remaining well-conditioned via LayerNorm.
4. **CLS token aggregation**: the CLS token attends to all *N* shift
   tokens and acts as a global summary of the full ensemble, from which
   the per-shift logits are then predicted.  This allows the model to
   form a holistic view of the ensemble before committing to weights.

Initialization guarantee
-------------------------
The final linear layer of the MLP head is initialised with weights
drawn from ``Normal(0, 1e-3)`` and a zero bias.  Because the logits
``a`` are therefore approximately zero for any input, the softmax
output is approximately uniform::

    a ≈ 0  →  w_i ≈ 1/N  for all i

This means **at initialisation A26f approximately reproduces the
original SAR-DDPM equal-weight average**, regardless of the Transformer
attention patterns, and is approximately equivalent to A26a–A26e at
step 0.

Checkpoint compatibility
-------------------------
``TransformerCycleSpinning`` is a new module.  Its architecture differs
from A26e's ``weight_predictor`` in two ways:

1. A26f adds ``cls_token``, ``pos_embed``, ``transformer_encoder``, and
   ``head`` parameters not present in A26e.
2. A26f removes A26e's ``weight_predictor`` Sequential (the MLP).

Therefore **A26e checkpoints cannot be loaded directly into A26f** — the
parameter names and shapes do not match.  When transitioning from A26e,
use::

    model.load_state_dict(checkpoint, strict=False)

Missing ``transformer_encoder.*``, ``cls_token``, ``pos_embed``, and
``head.*`` keys will be reported as missing and kept at their freshly
initialised values.  The missing ``weight_predictor.*`` keys from A26e
are simply absent in A26f.  Per the initialisation guarantee above, the
model starts from approximately equal-weight averaging regardless.

Future roadmap
---------------
* **A26g** — Learnable Shift Coordinates: jointly learn the (row, col)
  shift grid rather than using a fixed uniform grid, feeding shift
  geometry as an additional per-token conditioning signal alongside the
  four-branch descriptor ``(x, σ, W, S)``.
* **A26h** — Hierarchical Cycle Spinning: nested coarse + fine shift
  pyramids with independent Transformer-based weight predictors per
  level, allowing the model to aggregate shifts at multiple spatial
  scales.
* **A26i** — Full Adaptive Cycle-Spinning Transformer: integrates
  A26f-h into a unified Transformer-based aggregation module that
  consumes confidence, wavelet, structure-tensor, and shift-geometry
  features jointly across all scales.
* **A26j** — Bayesian Cycle Spinning: model per-image shift weights as
  a Dirichlet distribution and estimate uncertainty over the
  aggregation weights themselves, enabling principled confidence
  intervals over the fused prediction for journal-level uncertainty
  quantification in SAR despeckling.
* **A26k** — Meta-Learned Cycle Spinning: learn weight-prediction
  policies across datasets, allowing zero-shot transfer of the
  Transformer-based aggregator to unseen SAR sensor configurations and
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
:class:`~structdiff.inference.structure_wavelet_confidence_cycle_spinning.StructureWaveletConfidenceCycleSpinning`
exactly, with the sole architectural difference being that the MLP
weight predictor is replaced by a Transformer encoder + CLS-based MLP
head.  This ensures downstream logging, training, and ablation code
remain interchangeable across A26e and A26f, and all future extensions
(A26g-n) can inherit or compose from this module without breaking
changes.

References
----------
Coifman, R.R. & Donoho, D.L. (1995). Translation-Invariant
De-Noising. *Wavelets and Statistics*, Springer.

Mallat, S. (1999). *A Wavelet Tour of Signal Processing*. Academic Press.

Bigun, J. & Granlund, G.H. (1987). Optimal orientation detection of
linear symmetry. *Proc. ICCV*, 433–438.

Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS*.

Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding. *NAACL-HLT*.

Notes
-----
* All computation is performed in PyTorch; no NumPy, no CPU transfer,
  no in-place operations, full autograd support.
* The module is device-agnostic: all submodules and parameters move
  with ``model.to(device)``.
* Weights are predicted per batch element, so two images in the same
  batch may receive entirely different aggregation strategies.
* The Transformer's ``batch_first=True`` flag is used throughout,
  consistent with modern PyTorch practice.
* ``nn.TransformerEncoderLayer`` with ``norm_first=False`` (Post-LN)
  is used, which is the standard configuration for small-scale
  Transformers and matches the original "Attention Is All You Need"
  design.
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
#: MLP head.  Kept small so that the predicted logits start near zero,
#: producing a near-uniform softmax distribution and maximising
#: compatibility with the SAR-DDPM heuristic baseline and A26a–A26e.
_FINAL_LAYER_INIT_STD: float = 1e-3

#: Epsilon added inside the entropy logarithm for numerical stability.
#: Must satisfy _LOG_EPS << 1/N for any practical N.
_LOG_EPS: float = 1e-8

#: Number of channels expected in every confidence map (one scalar
#: confidence value per spatial location).
_CONFIDENCE_CHANNELS: int = 1

#: Feed-forward multiplier for the Transformer inner dimension.
#: Following the "Attention Is All You Need" convention, the FFN
#: hidden size is 4 × the model dimension D.
_FFN_MULTIPLIER: int = 4


# ---------------------------------------------------------------------------
# TransformerCycleSpinning
# ---------------------------------------------------------------------------


class TransformerCycleSpinning(nn.Module):
    """Cross-shift attention-based softmax aggregation of cycle-shifted
    diffusion outputs for SAR despeckling.

    Extends A26e
    (:class:`~structdiff.inference.structure_wavelet_confidence_cycle_spinning.StructureWaveletConfidenceCycleSpinning`)
    by replacing the flat MLP weight predictor with a Transformer
    encoder that allows every shift to attend to every other shift
    before the aggregation weights are predicted.

    Each of the *N* cycle-shifted predictions contributes one token to
    the Transformer's input sequence.  The token for shift *i* is the
    concatenation of four globally pooled descriptors::

        d_i = [GAP(x_i), GAP(σ_i), GAP(W_i), GAP(S_i)] ∈ R^{B × D}

    where ``D = C + 1 + Cw + Cs``.  A learnable CLS token is prepended
    and a learnable positional embedding is added to the full
    ``[B, N+1, D]`` token matrix.  After processing by the Transformer
    encoder, the CLS token representation is fed through a two-layer
    MLP head to produce per-shift logits, which are converted to
    softmax aggregation weights.  The weighted sum over shift image
    content yields the fused output.

    Parameters
    ----------
    num_shifts:
        Total number of cycle-spin shifts *N*. Must be a positive
        integer.
    channels:
        Number of channels *C* in each shifted prediction tensor.
        Must be a positive integer.
    wavelet_channels:
        Number of channels *Cw* in each wavelet tensor. Must be a
        positive integer.  NOT hardcoded; any value > 0 is accepted.
    structure_channels:
        Number of channels *Cs* in each structure tensor descriptor.
        Must be a positive integer.  NOT hardcoded; any value > 0 is
        accepted.
    num_heads:
        Number of attention heads in each TransformerEncoderLayer.
        Must be a positive integer and must evenly divide
        ``D = channels + 1 + wavelet_channels + structure_channels``.
        Default 4.
    num_layers:
        Number of TransformerEncoderLayer stacked in the encoder.
        Must be a positive integer.  Default 2.
    dropout:
        Dropout probability applied inside TransformerEncoderLayer
        (both attention dropout and feed-forward dropout).  Must be in
        ``[0.0, 1.0)``.  Default 0.1.
    temperature:
        Softmax temperature τ > 0 applied to the predicted logits.
        Lower values sharpen the weight distribution; higher values
        flatten it toward uniform.  Default 1.0.
    pooling:
        Spatial pooling mode for descriptor extraction.  One of:

        ``"avg"`` (default)
            Global average pooling (``nn.AdaptiveAvgPool2d(1)``).
        ``"max"``
            Global max pooling (``nn.AdaptiveMaxPool2d(1)``).

        Raises ``ValueError`` for any other string.
    eps:
        Small positive constant for numerical stability in entropy
        computations.  Default 1e-8.

    Attributes
    ----------
    num_shifts : int
        Number of cycle-spin shifts.
    channels : int
        Number of channels expected in each shifted prediction.
    wavelet_channels : int
        Number of channels expected in each wavelet tensor.
    structure_channels : int
        Number of channels expected in each structure tensor descriptor.
    token_dim : int
        Dimensionality of each shift token: ``C + 1 + Cw + Cs``.
    num_heads : int
        Number of Transformer attention heads.
    num_layers : int
        Number of stacked TransformerEncoderLayers.
    dropout : float
        Dropout probability used inside the Transformer.
    temperature : float
        Softmax temperature.
    pooling : str
        Spatial pooling mode.
    eps : float
        Numerical-stability constant.
    pool : nn.Module
        The instantiated pooling layer, shared by all four descriptor
        branches.
    cls_token : nn.Parameter
        Learnable CLS token of shape ``[1, 1, token_dim]``.
    pos_embed : nn.Parameter
        Learnable positional embedding of shape ``[1, N+1, token_dim]``.
    transformer_encoder : nn.TransformerEncoder
        Stack of ``num_layers`` TransformerEncoderLayers with
        ``batch_first=True``, GELU activation, and LayerNorm.
    head : nn.Sequential
        Two-layer MLP head that maps the CLS feature ``[B, token_dim]``
        to per-shift logits ``[B, num_shifts]``.

    Examples
    --------
    >>> import torch
    >>> from structdiff.inference.transformer_cycle_spinning import (
    ...     TransformerCycleSpinning,
    ... )
    >>> tcs = TransformerCycleSpinning(
    ...     num_shifts=9, channels=1, wavelet_channels=4,
    ...     structure_channels=12,
    ... )
    >>> tcs.num_shifts
    9
    >>> outputs = [torch.randn(2, 1, 64, 64) for _ in range(9)]
    >>> confidence_maps = [torch.rand(2, 1, 64, 64) for _ in range(9)]
    >>> wavelet_features = [torch.randn(2, 4, 32, 32) for _ in range(9)]
    >>> structure_features = [torch.randn(2, 12, 64, 64) for _ in range(9)]
    >>> fused = tcs(outputs, confidence_maps, wavelet_features, structure_features)
    >>> fused.shape
    torch.Size([2, 1, 64, 64])

    >>> fused, weights = tcs(
    ...     outputs, confidence_maps, wavelet_features, structure_features,
    ...     return_weights=True,
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
        num_heads: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
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
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                f"num_heads must be a positive integer, got {num_heads!r}."
            )
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer, got {num_layers!r}."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0.0, 1.0), got {dropout}."
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

        token_dim: int = channels + _CONFIDENCE_CHANNELS + wavelet_channels + structure_channels

        if token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim (= channels + 1 + wavelet_channels + "
                f"structure_channels = {token_dim}) must be divisible by "
                f"num_heads={num_heads}.  Adjust num_heads or channel counts."
            )

        # ----------------------------------------------------------------
        # Attributes
        # ----------------------------------------------------------------
        self.num_shifts: int = num_shifts
        self.channels: int = channels
        self.wavelet_channels: int = wavelet_channels
        self.structure_channels: int = structure_channels
        self.token_dim: int = token_dim
        self.num_heads: int = num_heads
        self.num_layers: int = num_layers
        self.dropout: float = dropout
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
        # ----------------------------------------------------------------
        self.pool: nn.Module
        if pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:  # pooling == "max"
            self.pool = nn.AdaptiveMaxPool2d(1)

        # ----------------------------------------------------------------
        # Learnable CLS token
        #
        # Shape: [1, 1, token_dim].  Expanded to [B, 1, token_dim] at
        # forward time and prepended to the N shift tokens, giving a
        # sequence of length N+1 for the Transformer.
        # ----------------------------------------------------------------
        self.cls_token: nn.Parameter = nn.Parameter(
            torch.zeros(1, 1, token_dim)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ----------------------------------------------------------------
        # Learnable positional embedding
        #
        # Shape: [1, N+1, token_dim].  Added to the full token sequence
        # (CLS + N shift tokens) before the Transformer encoder.
        # ----------------------------------------------------------------
        self.pos_embed: nn.Parameter = nn.Parameter(
            torch.zeros(1, num_shifts + 1, token_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ----------------------------------------------------------------
        # Transformer encoder
        #
        # Each TransformerEncoderLayer uses:
        #   - Self-attention with num_heads heads, head_dim = D / num_heads
        #   - Feed-forward inner dimension: _FFN_MULTIPLIER * D
        #   - GELU activation (matching the MLP-based stages)
        #   - batch_first=True (modern PyTorch convention)
        #   - LayerNorm (Post-LN, standard "Attention Is All You Need")
        #   - dropout applied to attention weights and FFN outputs
        # ----------------------------------------------------------------
        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=_FFN_MULTIPLIER * token_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer_encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # ----------------------------------------------------------------
        # MLP head
        #
        # Converts the CLS token representation [B, token_dim] to
        # per-shift logits [B, num_shifts].
        #
        # Linear(D, D) → GELU → Linear(D, N)
        #
        # The final layer is initialised with near-zero weights and zero
        # bias so that at initialisation all logits ≈ 0 and the softmax
        # output ≈ 1/N, preserving the SAR-DDPM / A26a–A26e baseline.
        # ----------------------------------------------------------------
        self.head: nn.Sequential = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_shifts),
        )

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        """(Re-)initialise the MLP head.

        The first linear layer of the head uses Kaiming (He) uniform
        initialisation for its weight matrix and a zero bias.

        The second (final) linear layer — whose output directly becomes
        the softmax logits — is initialised with weights drawn from
        ``Normal(0, _FINAL_LAYER_INIT_STD)`` and a zero bias.  Because
        ``_FINAL_LAYER_INIT_STD = 1e-3`` is small, the predicted logits
        ``a`` start very close to zero for any input, so::

            w = softmax(a / τ) ≈ [1/N, …, 1/N]

        reproducing the original SAR-DDPM equal-weight average and the
        A26a / A26b / A26c / A26d / A26e uniform initialisation at step 0,
        regardless of the Transformer attention patterns.

        The CLS token and positional embedding are initialised with
        truncated Normal (std=0.02) in ``__init__`` and are not reset
        here to avoid losing any learned positional structure when
        ``reset_parameters`` is called manually after construction.

        Examples
        --------
        >>> tcs = TransformerCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, num_heads=4,
        ... )
        >>> tcs.reset_parameters()
        >>> first_linear = tcs.head[0]
        >>> isinstance(first_linear, torch.nn.Linear)
        True
        """
        first_linear: nn.Linear = self.head[0]  # type: ignore[assignment]
        final_linear: nn.Linear = self.head[2]  # type: ignore[assignment]

        nn.init.kaiming_normal_(first_linear.weight, nonlinearity="linear")
        nn.init.zeros_(first_linear.bias)

        nn.init.normal_(final_linear.weight, mean=0.0, std=_FINAL_LAYER_INIT_STD)
        nn.init.zeros_(final_linear.bias)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_outputs(self, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """Validate the ``outputs`` sequence and return the reference tensor.

        Checks (in order):

        * Non-empty sequence.
        * ``len(outputs) == self.num_shifts``.
        * Every tensor is 4-dimensional ``[B, C, H, W]``.
        * Every tensor's channel dimension equals ``self.channels``.
        * Every tensor shares the same shape, dtype, and device as
          ``outputs[0]``.

        Parameters
        ----------
        outputs:
            Candidate sequence of cycle-shifted prediction tensors.

        Returns
        -------
        torch.Tensor
            The first tensor in ``outputs`` (reference tensor).

        Raises
        ------
        ValueError
            If any check fails, with a descriptive message.
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

        Checks (in order):

        * Non-empty.
        * ``len(confidence_maps) == self.num_shifts``.
        * Every map is 4-dimensional ``[B, 1, H, W]``.
        * Exactly one channel per map.
        * Batch size, spatial dimensions, dtype, device match ``reference``.

        Parameters
        ----------
        confidence_maps:
            Candidate sequence of per-shift confidence maps.
        reference:
            Validated reference tensor from ``_validate_outputs``.

        Raises
        ------
        ValueError
            If any check fails.
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

        ref_batch: int = reference.shape[0]
        ref_h: int = reference.shape[2]
        ref_w: int = reference.shape[3]
        ref_dtype: torch.dtype = reference.dtype
        ref_device: torch.device = reference.device

        for idx, conf in enumerate(confidence_maps):
            if conf.ndim != 4:
                raise ValueError(
                    f"Each confidence map must be 4-dimensional [B, 1, H, W]; "
                    f"confidence_maps[{idx}] has shape {conf.shape} "
                    f"(ndim={conf.ndim})."
                )
            if conf.shape[1] != _CONFIDENCE_CHANNELS:
                raise ValueError(
                    f"Each confidence map must have exactly "
                    f"{_CONFIDENCE_CHANNELS} channel; "
                    f"confidence_maps[{idx}] has {conf.shape[1]} channels."
                )
            if conf.shape[0] != ref_batch:
                raise ValueError(
                    f"Each confidence map must have the same batch size as "
                    f"outputs; outputs[0].shape[0]={ref_batch} but "
                    f"confidence_maps[{idx}].shape[0]={conf.shape[0]}."
                )
            if conf.shape[2] != ref_h or conf.shape[3] != ref_w:
                raise ValueError(
                    f"Each confidence map must have the same spatial dimensions "
                    f"as outputs; outputs[0].shape[2:]=({ref_h}, {ref_w}) but "
                    f"confidence_maps[{idx}].shape[2:]=({conf.shape[2]}, "
                    f"{conf.shape[3]})."
                )
            if conf.dtype != ref_dtype:
                raise ValueError(
                    f"Each confidence map must have the same dtype as outputs; "
                    f"outputs[0].dtype={ref_dtype} but "
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

        Checks (in order):

        * Non-empty.
        * ``len(wavelet_features) == self.num_shifts``.
        * Every tensor is 4-dimensional ``[B, Cw, Hw, Ww]``.
        * Exactly ``self.wavelet_channels`` channels per tensor.
        * Batch size, dtype, device match ``reference`` (spatial dims may differ).
        * All wavelet tensors share the same shape.

        Parameters
        ----------
        wavelet_features:
            Candidate sequence of per-shift wavelet tensors.
        reference:
            Validated reference tensor from ``_validate_outputs``.

        Raises
        ------
        ValueError
            If any check fails.
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

        ref_batch: int = reference.shape[0]
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
        if first_wav.shape[0] != ref_batch:
            raise ValueError(
                f"Each wavelet tensor must have the same batch size as outputs; "
                f"outputs[0].shape[0]={ref_batch} but "
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
                f"Each wavelet tensor must reside on the same device as outputs; "
                f"outputs[0].device={ref_device} but "
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

        Checks (in order):

        * Non-empty.
        * ``len(structure_features) == self.num_shifts``.
        * Every tensor is 4-dimensional ``[B, Cs, Hs, Ws]``.
        * Exactly ``self.structure_channels`` channels per tensor.
        * Batch size, dtype, device match ``reference`` (spatial dims may differ).
        * All structure tensors share the same shape.

        Parameters
        ----------
        structure_features:
            Candidate sequence of per-shift structure tensor descriptors,
            each ``[B, Cs, Hs, Ws]``.
        reference:
            Validated reference tensor from ``_validate_outputs``.

        Raises
        ------
        ValueError
            If any check fails.
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

        ref_batch: int = reference.shape[0]
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
        if first_st.shape[0] != ref_batch:
            raise ValueError(
                f"Each structure tensor must have the same batch size as "
                f"outputs; outputs[0].shape[0]={ref_batch} but "
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
                    f"All structure tensors must have the same dtype as outputs; "
                    f"outputs[0].dtype={ref_dtype} but "
                    f"structure_features[{idx}].dtype={st.dtype}."
                )
            if st.device != ref_device:
                raise ValueError(
                    f"All structure tensors must reside on the same device as "
                    f"outputs; outputs[0].device={ref_device} but "
                    f"structure_features[{idx}].device={st.device}."
                )

    # ------------------------------------------------------------------
    # Token construction
    # ------------------------------------------------------------------

    def _build_token_sequence(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
        structure_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Construct the Transformer input token sequence from per-shift descriptors.

        For each shift *i*, four branch descriptors are pooled and
        concatenated::

            z_i  = pool(x_i).reshape(B, C)     image descriptor
            c_i  = pool(σ_i).reshape(B, 1)     confidence descriptor
            v_i  = pool(W_i).reshape(B, Cw)    wavelet descriptor
            s_i  = pool(S_i).reshape(B, Cs)    structure descriptor
            d_i  = concat(z_i, c_i, v_i, s_i)  [B, D]

        The *N* per-shift descriptors are stacked into a token matrix
        ``[B, N, D]``, the learnable CLS token (expanded to ``[B, 1, D]``)
        is prepended, and the learnable positional embedding
        ``[1, N+1, D]`` is added, yielding the final token sequence
        ``[B, N+1, D]``.

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
            Token sequence of shape ``[B, N+1, D]``, in the dtype of the
            Transformer encoder parameters (fp32 by default; mixed-precision
            inputs are cast to match).
        """
        shift_tokens: List[torch.Tensor] = []
        for x_i, sig_i, wav_i, st_i in zip(
            outputs, confidence_maps, wavelet_features, structure_features
        ):
            pooled_x: torch.Tensor = self.pool(x_i)    # [B, C,  1, 1]
            pooled_s: torch.Tensor = self.pool(sig_i)  # [B, 1,  1, 1]
            pooled_w: torch.Tensor = self.pool(wav_i)  # [B, Cw, 1, 1]
            pooled_t: torch.Tensor = self.pool(st_i)   # [B, Cs, 1, 1]

            batch_size: int = pooled_x.shape[0]
            z_i: torch.Tensor = pooled_x.reshape(batch_size, -1)   # [B, C]
            c_i: torch.Tensor = pooled_s.reshape(batch_size, -1)   # [B, 1]
            v_i: torch.Tensor = pooled_w.reshape(batch_size, -1)   # [B, Cw]
            s_i: torch.Tensor = pooled_t.reshape(batch_size, -1)   # [B, Cs]

            d_i: torch.Tensor = torch.cat([z_i, c_i, v_i, s_i], dim=1)  # [B, D]
            shift_tokens.append(d_i)

        # Stack to [B, N, D]
        token_matrix: torch.Tensor = torch.stack(shift_tokens, dim=1)  # [B, N, D]

        # Cast to Transformer parameter dtype (typically fp32)
        encoder_dtype: torch.dtype = next(self.transformer_encoder.parameters()).dtype
        if token_matrix.dtype != encoder_dtype:
            token_matrix = token_matrix.to(encoder_dtype)

        batch_size_final: int = token_matrix.shape[0]

        # Expand CLS token: [1, 1, D] -> [B, 1, D]
        cls_expanded: torch.Tensor = self.cls_token.expand(
            batch_size_final, 1, self.token_dim
        )

        # Prepend CLS: [B, N+1, D]
        tokens: torch.Tensor = torch.cat([cls_expanded, token_matrix], dim=1)

        # Add positional embedding: [1, N+1, D] broadcast over B
        tokens = tokens + self.pos_embed

        return tokens  # [B, N+1, D]

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
        """Predict per-image softmax aggregation weights via Transformer attention.

        Constructs the shift token sequence, processes it through the
        Transformer encoder, extracts the CLS token representation,
        passes it through the MLP head to obtain per-shift logits, and
        applies a temperature-scaled softmax over the shift dimension.

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
            Shape ``[B, num_shifts]``. Every row sums to 1.0 and every
            entry is strictly positive. Retains the autograd graph.

        Raises
        ------
        ValueError
            If ``self.temperature`` is not strictly positive, or if any
            input sequence fails validation.

        Examples
        --------
        >>> tcs = TransformerCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, num_heads=4,
        ... )
        >>> outs = [torch.randn(3, 4, 16, 16) for _ in range(4)]
        >>> confs = [torch.rand(3, 1, 16, 16) for _ in range(4)]
        >>> wavs = [torch.randn(3, 4, 8, 8) for _ in range(4)]
        >>> sts = [torch.randn(3, 4, 16, 16) for _ in range(4)]
        >>> w = tcs.get_weights(outs, confs, wavs, sts)
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

        # Build token sequence: [B, N+1, D]
        tokens: torch.Tensor = self._build_token_sequence(
            outputs, confidence_maps, wavelet_features, structure_features
        )

        # Transformer encoder: [B, N+1, D] -> [B, N+1, D]
        encoded: torch.Tensor = self.transformer_encoder(tokens)

        # Extract CLS token (position 0): [B, D]
        cls_feat: torch.Tensor = encoded[:, 0, :]

        # MLP head: [B, D] -> [B, N]
        logits: torch.Tensor = self.head(cls_feat)

        # Temperature-scaled softmax: [B, N]
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
        """Aggregate cycle-shifted diffusion outputs with Transformer-predicted weights.

        Parameters
        ----------
        outputs:
            A sequence of *N* tensors, one per cycle-spin shift, each
            ``[B, C, H, W]`` with identical shape, dtype, device, and
            channel count equal to ``self.channels``.
        confidence_maps:
            A sequence of *N* tensors, each ``[B, 1, H, W]``, providing
            per-pixel confidence maps for the corresponding shifted
            prediction.  Must match ``outputs`` in batch size, spatial
            dimensions, dtype, and device.
        wavelet_features:
            A sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``,
            providing DWT subband features for each shifted prediction.
            Must match ``outputs`` in batch size, dtype, and device.
            Spatial dimensions ``(Hw, Ww)`` may differ from ``(H, W)``.
        structure_features:
            A sequence of *N* tensors, each ``[B, Cs, Hs, Ws]``,
            providing structure tensor descriptors (λ1, λ2, anisotropy,
            coherence, etc.) for each shifted prediction.  Must match
            ``outputs`` in batch size, dtype, and device.  Spatial
            dimensions may differ from ``(H, W)``.
        return_weights:
            If ``False`` (default), return only the fused tensor.
            If ``True``, return ``(fused, weights)``.

        Returns
        -------
        torch.Tensor or tuple of (torch.Tensor, torch.Tensor)
            * ``return_weights=False``: fused tensor, shape ``[B, C, H, W]``.
            * ``return_weights=True``: ``(fused, weights)`` where ``fused``
              has shape ``[B, C, H, W]`` and ``weights`` has shape
              ``[B, num_shifts]``.

        Raises
        ------
        ValueError
            If any input sequence fails validation (length, shape, dtype,
            device, or channel count).

        Notes
        -----
        The aggregation is::

            tokens  = [CLS | d_1 | … | d_N] + pos_embed    [B, N+1, D]
            encoded = TransformerEncoder(tokens)             [B, N+1, D]
            cls     = encoded[:, 0, :]                       [B, D]
            h       = GELU(Linear(cls))                      [B, D]
            a       = Linear(h)                              [B, N]
            w       = softmax(a / τ, dim=1)                  [B, N]
            x̂       = Σ_i w_i • x_i                          [B, C, H, W]

        Only the image content ``x_i`` enters the weighted sum.

        Examples
        --------
        >>> import torch
        >>> from structdiff.inference.transformer_cycle_spinning import (
        ...     TransformerCycleSpinning,
        ... )
        >>> tcs = TransformerCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, num_heads=4,
        ... )
        >>> outputs = [torch.ones(2, 4, 8, 8) * float(i + 1) for i in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> fused = tcs(outputs, confs, wavs, sts)
        >>> fused.shape
        torch.Size([2, 4, 8, 8])

        >>> fused, w = tcs(outputs, confs, wavs, sts, return_weights=True)
        >>> w.shape
        torch.Size([2, 4])
        >>> bool(torch.allclose(w.sum(dim=1), torch.ones(2), atol=1e-5))
        True
        """
        reference: torch.Tensor = self._validate_outputs(outputs)
        self._validate_confidence_maps(confidence_maps, reference)
        self._validate_wavelet_features(wavelet_features, reference)
        self._validate_structure_features(structure_features, reference)
        batch_size: int = reference.shape[0]

        # Predict per-image softmax weights via Transformer: [B, N]
        weights: torch.Tensor = self.get_weights(
            outputs, confidence_maps, wavelet_features, structure_features
        )

        # Stack outputs: list of N×[B,C,H,W] -> [N,B,C,H,W] -> [B,N,C,H,W]
        stacked: torch.Tensor = torch.stack(list(outputs), dim=0)  # [N, B, C, H, W]
        stacked = stacked.permute(1, 0, 2, 3, 4)                   # [B, N, C, H, W]

        # Broadcast weights: [B, N] -> [B, N, 1, 1, 1]
        weights_broadcast: torch.Tensor = weights.view(
            batch_size, self.num_shifts, 1, 1, 1
        )

        # Promote stacked dtype to match weights if needed (fp16 inputs)
        input_dtype: torch.dtype = stacked.dtype
        if stacked.dtype != weights_broadcast.dtype:
            stacked = stacked.to(weights_broadcast.dtype)

        # Weighted sum over shift dimension: [B, C, H, W]
        fused: torch.Tensor = (stacked * weights_broadcast).sum(dim=1)

        # Restore caller's original dtype (e.g. fp16 -> fp16)
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

        For each batch element ``b``::

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
        >>> tcs = TransformerCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, num_heads=4,
        ... )
        >>> outs = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> h = tcs.entropy(outs, confs, wavs, sts)
        >>> bool(h.item() > 0.0)
        True
        """
        weights: torch.Tensor = self.get_weights(
            outputs, confidence_maps, wavelet_features, structure_features
        )
        per_image_entropy: torch.Tensor = -(
            weights * torch.log(weights.clamp(min=self.eps))
        ).sum(dim=1)
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
            negative → encourage peaked, shift-specific weights).
            Default 1.0.

        Returns
        -------
        torch.Tensor
            Scalar tensor. Retains the autograd graph.

        Examples
        --------
        >>> tcs = TransformerCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, num_heads=4,
        ... )
        >>> outs = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> reg = tcs.entropy_regularizer(outs, confs, wavs, sts, coefficient=0.01)
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
        >>> tcs = TransformerCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, num_heads=4,
        ... )
        >>> outs = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> n_eff = tcs.effective_num_shifts(outs, confs, wavs, sts)
        >>> bool(0.0 < n_eff.item() <= 4.0 + 1e-3)
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
        >>> tcs = TransformerCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, num_heads=4,
        ... )
        >>> outs = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> v = tcs.weight_variance(outs, confs, wavs, sts)
        >>> v.shape
        torch.Size([])
        >>> bool(v.item() >= 0.0)
        True
        """
        weights: torch.Tensor = self.get_weights(
            outputs, confidence_maps, wavelet_features, structure_features
        )
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
            Shape ``[B]``, dtype ``int64``. Detached from the autograd graph.

        Examples
        --------
        >>> tcs = TransformerCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, num_heads=4,
        ... )
        >>> outs = [torch.randn(3, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(3, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(3, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(3, 4, 8, 8) for _ in range(4)]
        >>> idx = tcs.max_weight_index(outs, confs, wavs, sts)
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
            Shape ``[B]``, dtype ``int64``. Detached from the autograd graph.

        Examples
        --------
        >>> tcs = TransformerCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, num_heads=4,
        ... )
        >>> outs = [torch.randn(3, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(3, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(3, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(3, 4, 8, 8) for _ in range(4)]
        >>> idx = tcs.min_weight_index(outs, confs, wavs, sts)
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
            Number of rows ``B``. Must be a positive integer.

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
        >>> tcs = TransformerCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, num_heads=4,
        ... )
        >>> u = tcs.uniform_weights(batch_size=2)
        >>> u.shape
        torch.Size([2, 4])
        >>> bool(torch.allclose(u, torch.full((2, 4), 0.25)))
        True
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive integer, got {batch_size!r}."
            )
        reference_param: torch.Tensor = next(self.head.parameters())
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
        >>> tcs = TransformerCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, num_heads=4,
        ... )
        >>> outs = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> kl = tcs.kl_to_uniform(outs, confs, wavs, sts)
        >>> kl.shape
        torch.Size([])
        >>> bool(kl.item() >= -1e-6)
        True
        """
        weights: torch.Tensor = self.get_weights(
            outputs, confidence_maps, wavelet_features, structure_features
        )
        uniform: torch.Tensor = self.uniform_weights(weights.shape[0])
        per_image_kl: torch.Tensor = (
            weights
            * (
                torch.log(weights.clamp(min=self.eps))
                - torch.log(uniform)
            )
        ).sum(dim=1)
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
        >>> tcs = TransformerCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, num_heads=4,
        ... )
        >>> outs = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> jsd = tcs.js_to_uniform(outs, confs, wavs, sts)
        >>> jsd.shape
        torch.Size([])
        >>> bool(jsd.item() >= -1e-6)
        True
        """
        weights: torch.Tensor = self.get_weights(
            outputs, confidence_maps, wavelet_features, structure_features
        )
        uniform: torch.Tensor = self.uniform_weights(weights.shape[0])
        mixture: torch.Tensor = 0.5 * (weights + uniform)

        kl_w_m: torch.Tensor = (
            weights
            * (
                torch.log(weights.clamp(min=self.eps))
                - torch.log(mixture.clamp(min=self.eps))
            )
        ).sum(dim=1)
        kl_u_m: torch.Tensor = (
            uniform
            * (
                torch.log(uniform.clamp(min=self.eps))
                - torch.log(mixture.clamp(min=self.eps))
            )
        ).sum(dim=1)

        per_image_jsd: torch.Tensor = 0.5 * (kl_w_m + kl_u_m)
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
        >>> tcs = TransformerCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, num_heads=4, temperature=1.0,
        ... )
        >>> tcs.set_temperature(0.5)
        >>> tcs.temperature
        0.5
        >>> try:
        ...     tcs.set_temperature(-1.0)
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
        """Disable gradient updates for all Transformer and head parameters.

        Freezes ``cls_token``, ``pos_embed``, ``transformer_encoder``,
        and ``head``.  The pooling layer has no parameters.

        Examples
        --------
        >>> tcs = TransformerCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, num_heads=4,
        ... )
        >>> tcs.freeze()
        >>> tcs.is_frozen()
        True
        """
        self.cls_token.requires_grad_(False)
        self.pos_embed.requires_grad_(False)
        for param in self.transformer_encoder.parameters():
            param.requires_grad_(False)
        for param in self.head.parameters():
            param.requires_grad_(False)

    def unfreeze(self) -> None:
        """Enable gradient updates for all Transformer and head parameters.

        Examples
        --------
        >>> tcs = TransformerCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, num_heads=4,
        ... )
        >>> tcs.freeze()
        >>> tcs.unfreeze()
        >>> tcs.is_frozen()
        False
        """
        self.cls_token.requires_grad_(True)
        self.pos_embed.requires_grad_(True)
        for param in self.transformer_encoder.parameters():
            param.requires_grad_(True)
        for param in self.head.parameters():
            param.requires_grad_(True)

    def is_frozen(self) -> bool:
        """Return ``True`` if all learnable parameters are frozen.

        Returns
        -------
        bool
            ``True`` iff ``requires_grad`` is ``False`` for the CLS token,
            positional embedding, and all Transformer encoder and head
            parameters.

        Examples
        --------
        >>> tcs = TransformerCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, num_heads=4,
        ... )
        >>> tcs.is_frozen()
        False
        >>> tcs.freeze()
        >>> tcs.is_frozen()
        True
        """
        trainable_params = (
            [self.cls_token, self.pos_embed]
            + list(self.transformer_encoder.parameters())
            + list(self.head.parameters())
        )
        return all(not p.requires_grad for p in trainable_params)

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
        >>> tcs = TransformerCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, num_heads=4,
        ... )
        >>> outs = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> stats = tcs.weight_statistics(outs, confs, wavs, sts)
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
        ).sum(dim=1)
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
        >>> tcs = TransformerCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, num_heads=4,
        ... )
        >>> outs = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> s = tcs.summary(outs, confs, wavs, sts)
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
        ).sum(dim=1)
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
        >>> tcs = TransformerCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, num_heads=4,
        ... )
        >>> outs = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> s = tcs.save_statistics(outs, confs, wavs, sts)
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
        ).sum(dim=1)
        entropy_val: torch.Tensor = per_image_entropy.mean()

        uniform: torch.Tensor = self.uniform_weights(weights.shape[0])
        per_image_kl: torch.Tensor = (
            weights
            * (
                torch.log(weights.clamp(min=self.eps))
                - torch.log(uniform)
            )
        ).sum(dim=1)
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
        >>> tcs = TransformerCycleSpinning(
        ...     num_shifts=9, channels=1, wavelet_channels=4,
        ...     structure_channels=12, num_heads=4, num_layers=2,
        ...     dropout=0.1, temperature=0.5, pooling="avg",
        ... )
        >>> print(tcs)  # doctest: +ELLIPSIS
        TransformerCycleSpinning(
          ...
        )
        """
        return (
            f"num_shifts={self.num_shifts}, "
            f"channels={self.channels}, "
            f"wavelet_channels={self.wavelet_channels}, "
            f"structure_channels={self.structure_channels}, "
            f"token_dim={self.token_dim}, "
            f"num_heads={self.num_heads}, "
            f"num_layers={self.num_layers}, "
            f"dropout={self.dropout}, "
            f"temperature={self.temperature}, "
            f"pooling={self.pooling}"
        )
