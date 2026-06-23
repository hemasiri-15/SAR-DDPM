"""
structdiff/inference/learnable_shift_cycle_spinning.py
=======================================================
A26g: LearnableShiftCycleSpinning — jointly learnable shift geometry and
Transformer-based softmax aggregation of cycle-shifted diffusion outputs
for SAR despeckling.

Background
----------
The original SAR-DDPM cycle-spinning implementation (see
``inference_sar.py`` / ``inference_sar_unet.py``) applies a denoiser
to *N* shifted copies of the input, inverse-shifts each result, and
averages them with **fixed equal weights** over a fixed, hand-designed
shift grid::

    # Fixed 3×3 grid
    shifts = [(-1,-1), (-1,0), (-1,1),
              ( 0,-1), ( 0,0), ( 0,1),
              ( 1,-1), ( 1,0), ( 1,1)]
    pred_tensor += (1.0 / N) * sample

A26a–A26f introduced progressively richer weight predictors (global
logits → image-adaptive MLP → confidence-conditioned MLP → wavelet-
conditioned MLP → structure-tensor-conditioned MLP → Transformer with
cross-shift self-attention) while keeping the **shift positions fixed**.

A26g — Learnable Shift Geometry
---------------------------------
A26g removes the fixed-grid assumption entirely. The model learns the
shift coordinates

    (Δr_i, Δc_i)   for i = 0, …, N-1

as an ``nn.Parameter`` (``raw_shift_coords``, shape ``[N, 2]``). The
actual (bounded) coordinates are obtained via a tanh activation::

    coords = max_shift_radius · tanh(raw_shift_coords)

so that all shifts satisfy::

    −max_shift_radius ≤ Δr_i ≤ max_shift_radius
    −max_shift_radius ≤ Δc_i ≤ max_shift_radius

always, without clipping or detaching. Gradients flow through tanh into
``raw_shift_coords`` at every backward pass.

Because fractional (sub-pixel) shifts are now possible, the integer
``torch.roll`` used in the original SAR-DDPM code is replaced by
``torch.nn.functional.grid_sample`` with bilinear interpolation and
reflection padding, which is differentiable with respect to the sampling
coordinates and therefore with respect to ``raw_shift_coords``.

The Transformer weight predictor from A26f is retained unchanged. A26g
**adds** a lightweight coordinate embedding that encodes the current
shift geometry as an additional per-token signal, so the Transformer
learns both:

* *Where* to shift  (via ``raw_shift_coords``), and
* *How much to weight* each shift  (via the Transformer head),

jointly and end-to-end.

Architecture
------------
For shift *i*::

    coords   = max_shift_radius · tanh(raw_shift_coords)    [N, 2]
    g_i      = coordinate_embedding(coords[i])               [B, Ge]
               (broadcast over the batch dimension)

    z_i      = GAP(x_i)                                     [B, C]
    c_i      = GAP(σ_i)                                     [B, 1]
    v_i      = GAP(W_i)                                     [B, Cw]
    s_i      = GAP(S_i)                                     [B, Cs]
    d_i      = concat(z_i, c_i, v_i, s_i, g_i)             [B, D]
               where D = C + 1 + Cw + Cs + Ge

The stacked token sequence, CLS token, positional embedding, and
Transformer blocks are exactly as in A26f, with the only change being
the extended token dimension D.

The differentiable shift is applied via ``apply_shift`` and reversed via
``inverse_shift`` (both implemented with ``grid_sample``). In the
forward pass the caller provides pre-shifted, pre-denoised outputs (the
``outputs`` argument); A26g's ``apply_shift`` / ``inverse_shift`` helpers
exist as public utilities for callers that want to perform the shifting
inside this module or for unit-testing gradient flow.

Differentiable shifting
-----------------------
``apply_shift(x, shift_row, shift_col)`` constructs a normalised
sampling grid that is offset by (shift_row, shift_col) pixels relative
to the identity grid::

    # Identity grid in [-1, 1]² (align_corners=False convention)
    grid_y = linspace(-1 + 1/H, 1 - 1/H, H)   # row coords
    grid_x = linspace(-1 + 1/W, 1 - 1/W, W)   # col coords
    base_grid[..., 0] = grid_x   # columns = x-axis in grid_sample
    base_grid[..., 1] = grid_y   # rows    = y-axis in grid_sample

    # A positive shift_row moves content downward; to sample content
    # from (row - shift_row) we subtract the normalised shift:
    grid[..., 1] -= shift_row * 2.0 / H   # row shift
    grid[..., 0] -= shift_col * 2.0 / W   # col shift

``inverse_shift(x, shift_row, shift_col)`` is identical but with the
sign of both offsets negated, i.e. it applies (-shift_row, -shift_col).

All grid construction uses only differentiable PyTorch ops; no NumPy,
no CPU transfer, no detach.

Coordinate regularization
--------------------------
Two regularization terms encourage the learned shift geometry to remain
well-spread and compact:

**Radius regularizer** (``radius_lambda``)::

    L_r = Σ_i ‖coords_i‖²

Penalises large shifts, encouraging the learned grid to stay within a
compact region.

**Repulsion regularizer** (``repulsion_lambda``)::

    L_rep = Σ_{i≠j} exp(-d_{ij})

where d_{ij} = ‖coords_i − coords_j‖₂. The exponential falls off with
distance, so the repulsion is strongest for nearby shift pairs and
gently encourages diversity across the learned grid.

Both terms are exposed via ``radius_regularizer()``,
``repulsion_regularizer()``, and ``coordinate_regularizer()`` for
flexible integration into the training loss.

Initialization guarantee
-------------------------
If ``num_shifts == 9`` the ``raw_shift_coords`` parameter is initialised
so that after the tanh activation the coordinates reproduce the standard
3×3 grid::

    tanh^{-1}(x / max_shift_radius)   for x in {-1, 0, 1}

For any other value of ``num_shifts`` the coordinates are initialised
uniformly in [-1, 1] (after tanh, which maps them to approximately
[-0.76, 0.76] relative to max_shift_radius).

The MLP head's final linear layer is initialised with near-zero weights
(``std=1e-3``) and zero bias, so the predicted logits start close to
zero and::

    softmax(a / τ) ≈ [1/N, …, 1/N]

at step 0 regardless of the shift geometry or Transformer attention
patterns, reproducing the SAR-DDPM equal-weight average.

Checkpoint compatibility
-------------------------
A26f checkpoints can be loaded with ``strict=False``::

    model.load_state_dict(a26f_checkpoint, strict=False)

The keys that will be missing (and therefore initialised from scratch)
are:

* ``raw_shift_coords`` — the learnable shift coordinates.
* ``coordinate_embedding.*`` — the two-layer coordinate embedding MLP.
* ``pos_embed``, ``cls_token``, ``blocks.*``, ``head.*``, ``pool`` —
  already present in A26f checkpoints and will load normally.

Future roadmap
--------------
* **A26h** — Hierarchical Shift Pyramid: nested coarse + fine learnable
  shift grids with independent Transformer-based weight predictors per
  level, enabling multi-scale translation-invariant despeckling.
* **A26i** — Unified Adaptive Transformer: integrates A26f-g into a
  single module where the shift geometry, feature conditioning, and
  weight prediction are jointly trained with a shared Transformer
  backbone.
* **A26j** — Bayesian Shift Coordinates: model ``raw_shift_coords`` as
  a distribution (e.g. Gaussian with learnable mean and variance) and
  estimate uncertainty over both the shift geometry and the aggregation
  weights via ELBO training.
* **A26k** — Meta-Learned Coordinates: learn the shift geometry as a
  meta-parameter across a family of SAR sensor configurations, allowing
  zero-shot adaptation of the shift grid to unseen scenes.
* **A26l** — Reinforcement Shift Selection: use a policy network to
  decide which of the (now differentiably positioned) shifts to evaluate,
  reducing inference cost while preserving despeckling quality.
* **A26m** — Timestep-Adaptive Coordinates: modulate ``raw_shift_coords``
  as a function of the diffusion timestep so the shift grid adapts
  dynamically over the denoising trajectory.
* **A26n** — Dynamic Shift Count: jointly learn the number of shifts *N*
  and their positions, replacing the fixed budget with an adaptive per-
  image allocation.

References
----------
Coifman, R.R. & Donoho, D.L. (1995). Translation-Invariant De-Noising.
*Wavelets and Statistics*, Springer.

Jaderberg, M. et al. (2015). Spatial Transformer Networks. *NeurIPS*.

Dai, J. et al. (2017). Deformable Convolutional Networks. *ICCV*.

Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS*.

Dosovitskiy, A. et al. (2021). An Image is Worth 16×16 Words:
Transformers for Image Recognition at Scale. *ICLR*.

Bigun, J. & Granlund, G.H. (1987). Optimal orientation detection of
linear symmetry. *Proc. ICCV*, 433–438.

Notes
-----
* All computation is in PyTorch; no NumPy, no CPU transfer, no in-place
  operations, full autograd support throughout.
* The module is device-agnostic; all submodules and parameters move with
  ``model.to(device)``.
* Mixed-precision (fp16/bf16) compatible; the Transformer blocks operate
  in fp32 by default and inputs are cast accordingly.
* DDP compatible; no state outside ``nn.Parameter`` and ``nn.Module``.
* ``torch.compile`` compatible; no data-dependent Python control flow
  beyond list iteration over a fixed-length sequence.
* PyTorch ≥ 2.2 required (for ``grid_sample`` gradient support and
  ``nn.MultiheadAttention`` with ``average_attn_weights=False``).

Examples
--------
>>> import torch
>>> from structdiff.inference.learnable_shift_cycle_spinning import (
...     LearnableShiftCycleSpinning,
... )
>>> lscs = LearnableShiftCycleSpinning(
...     num_shifts=9,
...     channels=1,
...     wavelet_channels=4,
...     structure_channels=12,
...     coordinate_embed_dim=16,
...     num_heads=1,
...     num_layers=2,
... )
>>> lscs.num_shifts
9
>>> outputs = [torch.randn(2, 1, 64, 64) for _ in range(9)]
>>> confs   = [torch.rand(2, 1, 64, 64) for _ in range(9)]
>>> wavs    = [torch.randn(2, 4, 32, 32) for _ in range(9)]
>>> sts     = [torch.randn(2, 12, 64, 64) for _ in range(9)]
>>> fused   = lscs(outputs, confs, wavs, sts)
>>> fused.shape
torch.Size([2, 1, 64, 64])
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Supported spatial pooling modes.
_VALID_POOLING_MODES: frozenset = frozenset({"avg", "max"})

#: Supported FFN nonlinearities (mirroring A26f).
_VALID_ACTIVATIONS: frozenset = frozenset({"gelu", "relu", "silu"})

#: Dispatch table from activation name to functional implementation.
_ACTIVATION_FN: Dict[str, object] = {
    "gelu": F.gelu,
    "relu": F.relu,
    "silu": F.silu,
}

#: Near-zero std for MLP head final layer init; gives softmax ≈ 1/N at step 0.
_FINAL_LAYER_INIT_STD: float = 1e-3

#: Truncated-Normal std for CLS token init (tighter than ViT's 0.02).
_CLS_INIT_STD: float = 0.01

#: Truncated-Normal std for positional embedding init.
_POS_INIT_STD: float = 0.01

#: Epsilon inside entropy logarithm for numerical stability.
_LOG_EPS: float = 1e-8

#: Number of channels expected in every confidence map.
_CONFIDENCE_CHANNELS: int = 1

#: Small epsilon used in repulsion distance denominator.
_DIST_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# TransformerBlock (identical to A26f — reproduced for self-containedness)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Single Transformer encoder block with configurable norm order and full
    per-head attention weight exposure.

    Identical in behaviour to the ``TransformerBlock`` in A26f
    (``transformer_cycle_spinning.py``). Reproduced here so that
    ``learnable_shift_cycle_spinning.py`` is self-contained and does not
    import from the A26f module, keeping each stage independent for
    ablation studies.

    Parameters
    ----------
    d_model:
        Token dimensionality. Must be divisible by ``num_heads``.
    num_heads:
        Number of self-attention heads.
    ffn_multiplier:
        FFN hidden-dimension multiplier. Default 4.
    dropout:
        Dropout probability in ``[0.0, 1.0)``. Default 0.1.
    activation:
        FFN nonlinearity: ``"gelu"`` (default), ``"relu"``, or ``"silu"``.
    norm_first:
        If ``True``, use Pre-LN (GPT-2 style). If ``False`` (default),
        use Post-LN ("Attention Is All You Need" style).

    Examples
    --------
    >>> blk = TransformerBlock(d_model=16, num_heads=4)
    >>> x = torch.randn(3, 9, 16)
    >>> y = blk(x)
    >>> y.shape
    torch.Size([3, 9, 16])
    >>> w = blk.get_attention_weights()
    >>> w.shape
    torch.Size([3, 4, 9, 9])
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = False,
    ) -> None:
        super().__init__()

        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {d_model!r}.")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"num_heads must be a positive integer, got {num_heads!r}.")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}."
            )
        if not isinstance(ffn_multiplier, int) or ffn_multiplier <= 0:
            raise ValueError(
                f"ffn_multiplier must be a positive integer, got {ffn_multiplier!r}."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")
        if activation not in _VALID_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_VALID_ACTIVATIONS)}, "
                f"got {activation!r}."
            )

        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.ffn_multiplier: int = ffn_multiplier
        self.norm_first: bool = norm_first
        self._act_fn = _ACTIVATION_FN[activation]
        self.activation: str = activation

        dim_ff: int = ffn_multiplier * d_model

        self.self_attn: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff1: nn.Linear = nn.Linear(d_model, dim_ff)
        self.ff2: nn.Linear = nn.Linear(dim_ff, d_model)
        self.norm1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm2: nn.LayerNorm = nn.LayerNorm(d_model)
        self.drop1: nn.Dropout = nn.Dropout(dropout)
        self.drop2: nn.Dropout = nn.Dropout(dropout)
        self.drop3: nn.Dropout = nn.Dropout(dropout)

        self._last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one Transformer encoder block.

        Parameters
        ----------
        x:
            ``[B, seq_len, d_model]``.

        Returns
        -------
        torch.Tensor
            ``[B, seq_len, d_model]``.
        """
        if self.norm_first:
            x_n: torch.Tensor = self.norm1(x)
            attn_out, attn_w = self.self_attn(
                x_n, x_n, x_n,
                need_weights=True,
                average_attn_weights=False,
            )
            x = x + self.drop1(attn_out)
            x_n2: torch.Tensor = self.norm2(x)
            x = x + self.drop2(
                self.ff2(self.drop3(self._act_fn(self.ff1(x_n2))))
            )
        else:
            attn_out, attn_w = self.self_attn(
                x, x, x,
                need_weights=True,
                average_attn_weights=False,
            )
            x = self.norm1(x + self.drop1(attn_out))
            x = self.norm2(x + self.drop2(
                self.ff2(self.drop3(self._act_fn(self.ff1(x))))
            ))
        self._last_attn_weights = attn_w.detach()
        return x

    def get_attention_weights(self) -> torch.Tensor:
        """Return per-head attention weights from the most recent forward pass.

        Returns
        -------
        torch.Tensor
            Shape ``[B, num_heads, seq_len, seq_len]``. Detached.

        Raises
        ------
        RuntimeError
            If called before the first forward pass.
        """
        if self._last_attn_weights is None:
            raise RuntimeError(
                "No attention weights available. Call forward() at least once."
            )
        return self._last_attn_weights

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"num_heads={self.num_heads}, "
            f"ffn_multiplier={self.ffn_multiplier}, "
            f"activation={self.activation!r}, "
            f"norm_first={self.norm_first}"
        )


# ---------------------------------------------------------------------------
# LearnableShiftCycleSpinning
# ---------------------------------------------------------------------------

class LearnableShiftCycleSpinning(nn.Module):
    """Jointly learnable shift geometry and Transformer-based softmax
    aggregation of cycle-shifted diffusion outputs for SAR despeckling.

    Extends A26f
    (:class:`~structdiff.inference.transformer_cycle_spinning.TransformerCycleSpinning`)
    by making the (row, col) shift coordinates themselves learnable
    ``nn.Parameter`` values, bounded by a tanh activation, and adding a
    lightweight coordinate embedding that feeds the current shift geometry
    into each shift's Transformer token.

    The module learns *where* to shift (via ``raw_shift_coords``) and
    *how much to weight* each shift (via the Transformer head) jointly
    and end-to-end through standard autograd.

    Parameters
    ----------
    num_shifts:
        Total number of cycle-spin shifts *N*. Must be a positive integer.
    channels:
        Number of channels *C* in each shifted prediction tensor. Must be
        a positive integer.
    wavelet_channels:
        Number of channels *Cw* in each wavelet tensor. Must be a positive
        integer. NOT hardcoded.
    structure_channels:
        Number of channels *Cs* in each structure tensor descriptor. Must
        be a positive integer. NOT hardcoded.
    coordinate_embed_dim:
        Output dimension *Ge* of the coordinate embedding MLP. The token
        dimension becomes D = C + 1 + Cw + Cs + Ge. Must be a positive
        integer. Default 16.
    num_heads:
        Number of self-attention heads per :class:`TransformerBlock`. Must
        evenly divide D. Default 4.
    num_layers:
        Number of stacked :class:`TransformerBlock` modules. Default 2.
    dropout:
        Dropout probability in ``[0.0, 1.0)`` inside every block. Default
        0.1.
    temperature:
        Softmax temperature τ > 0. Default 1.0.
    max_shift_radius:
        Maximum absolute shift in pixels. All learned coordinates satisfy
        |Δr|, |Δc| ≤ max_shift_radius. Must be strictly positive. Default
        3.0.
    radius_lambda:
        Coefficient for the radius regularization term. Default 1e-4.
    repulsion_lambda:
        Coefficient for the repulsion regularization term. Default 1e-3.
    pooling:
        Spatial pooling mode: ``"avg"`` (default) or ``"max"``.
    eps:
        Small constant for numerical stability. Default 1e-8.

    Attributes
    ----------
    num_shifts : int
    channels : int
    wavelet_channels : int
    structure_channels : int
    coordinate_embed_dim : int
    num_heads : int
    num_layers : int
    dropout : float
    temperature : float
    max_shift_radius : float
    radius_lambda : float
    repulsion_lambda : float
    pooling : str
    eps : float
    token_dim : int
        D = C + 1 + Cw + Cs + Ge.
    raw_shift_coords : nn.Parameter
        Shape ``[num_shifts, 2]``. Learnable raw coordinates; passed
        through tanh·max_shift_radius to obtain bounded coordinates.
    coordinate_embedding : nn.Sequential
        Two-layer MLP: ``Linear(2→32) → GELU → Linear(32→Ge)``.
    pool : nn.Module
    cls_token : nn.Parameter
        Shape ``[1, 1, token_dim]``.
    pos_embed : nn.Parameter
        Shape ``[1, num_shifts+1, token_dim]``.
    blocks : nn.ModuleList
        ``num_layers`` :class:`TransformerBlock` instances.
    head : nn.Sequential
        ``Linear(D,D) → GELU → Linear(D,N)``.

    Examples
    --------
    >>> import torch
    >>> from structdiff.inference.learnable_shift_cycle_spinning import (
    ...     LearnableShiftCycleSpinning,
    ... )
    >>> lscs = LearnableShiftCycleSpinning(
    ...     num_shifts=9, channels=1, wavelet_channels=4,
    ...     structure_channels=12, coordinate_embed_dim=16, num_heads=1,
    ... )
    >>> outputs = [torch.randn(2, 1, 64, 64) for _ in range(9)]
    >>> confs   = [torch.rand(2, 1, 64, 64) for _ in range(9)]
    >>> wavs    = [torch.randn(2, 4, 32, 32) for _ in range(9)]
    >>> sts     = [torch.randn(2, 12, 64, 64) for _ in range(9)]
    >>> fused   = lscs(outputs, confs, wavs, sts)
    >>> fused.shape
    torch.Size([2, 1, 64, 64])
    >>> fused, w = lscs(outputs, confs, wavs, sts, return_weights=True)
    >>> w.shape
    torch.Size([2, 9])
    >>> bool(torch.allclose(w.sum(dim=1), torch.ones(2), atol=1e-5))
    True
    """

    def __init__(
        self,
        num_shifts: int,
        channels: int,
        wavelet_channels: int,
        structure_channels: int,
        coordinate_embed_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 1.0,
        max_shift_radius: float = 3.0,
        radius_lambda: float = 1e-4,
        repulsion_lambda: float = 1e-3,
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
        if not isinstance(coordinate_embed_dim, int) or coordinate_embed_dim <= 0:
            raise ValueError(
                f"coordinate_embed_dim must be a positive integer, "
                f"got {coordinate_embed_dim!r}."
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
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")
        if temperature <= 0.0:
            raise ValueError(
                f"temperature must be strictly positive, got {temperature}."
            )
        if max_shift_radius <= 0.0:
            raise ValueError(
                f"max_shift_radius must be strictly positive, "
                f"got {max_shift_radius}."
            )
        if radius_lambda < 0.0:
            raise ValueError(
                f"radius_lambda must be non-negative, got {radius_lambda}."
            )
        if repulsion_lambda < 0.0:
            raise ValueError(
                f"repulsion_lambda must be non-negative, got {repulsion_lambda}."
            )
        if pooling not in _VALID_POOLING_MODES:
            raise ValueError(
                f"pooling must be one of {sorted(_VALID_POOLING_MODES)}, "
                f"got {pooling!r}."
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be strictly positive, got {eps}.")

        token_dim: int = (
            channels
            + _CONFIDENCE_CHANNELS
            + wavelet_channels
            + structure_channels
            + coordinate_embed_dim
        )
        if token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim (= channels + 1 + wavelet_channels + "
                f"structure_channels + coordinate_embed_dim = {token_dim}) "
                f"must be divisible by num_heads={num_heads}. "
                f"Adjust num_heads or coordinate_embed_dim."
            )

        # ----------------------------------------------------------------
        # Attributes
        # ----------------------------------------------------------------
        self.num_shifts: int = num_shifts
        self.channels: int = channels
        self.wavelet_channels: int = wavelet_channels
        self.structure_channels: int = structure_channels
        self.coordinate_embed_dim: int = coordinate_embed_dim
        self.num_heads: int = num_heads
        self.num_layers: int = num_layers
        self.dropout: float = dropout
        self.temperature: float = temperature
        self.max_shift_radius: float = max_shift_radius
        self.radius_lambda: float = radius_lambda
        self.repulsion_lambda: float = repulsion_lambda
        self.pooling: str = pooling
        self.eps: float = eps
        self.token_dim: int = token_dim

        # ----------------------------------------------------------------
        # Learnable shift coordinates
        #
        # Shape [num_shifts, 2].  Column 0 = row shift; column 1 = col shift.
        # Actual coordinates: max_shift_radius * tanh(raw_shift_coords).
        # ----------------------------------------------------------------
        self.raw_shift_coords: nn.Parameter = nn.Parameter(
            torch.empty(num_shifts, 2, dtype=torch.float32)
        )
        self._init_shift_coords()

        # ----------------------------------------------------------------
        # Coordinate embedding MLP
        #
        # Maps the 2-D (row, col) coordinate of shift i to a vector of
        # dimension coordinate_embed_dim, which is appended to the other
        # per-shift descriptors before forming the Transformer token.
        #
        # Architecture: Linear(2→32) → GELU → Linear(32→Ge)
        # No BatchNorm (batch-size independent, compatible with DDP).
        # ----------------------------------------------------------------
        self.coordinate_embedding: nn.Sequential = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, coordinate_embed_dim),
        )

        # ----------------------------------------------------------------
        # Spatial pooling layer (shared by all four feature branches)
        # ----------------------------------------------------------------
        self.pool: nn.Module
        if pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

        # ----------------------------------------------------------------
        # Learnable CLS token: [1, 1, token_dim]
        # ----------------------------------------------------------------
        self.cls_token: nn.Parameter = nn.Parameter(
            torch.empty(1, 1, token_dim)
        )
        nn.init.trunc_normal_(self.cls_token, std=_CLS_INIT_STD)

        # ----------------------------------------------------------------
        # Learnable positional embedding: [1, N+1, token_dim]
        # ----------------------------------------------------------------
        self.pos_embed: nn.Parameter = nn.Parameter(
            torch.empty(1, num_shifts + 1, token_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=_POS_INIT_STD)

        # ----------------------------------------------------------------
        # Transformer encoder (ModuleList for per-layer attention access)
        # ----------------------------------------------------------------
        self.blocks: nn.ModuleList = nn.ModuleList([
            TransformerBlock(
                d_model=token_dim,
                num_heads=num_heads,
                ffn_multiplier=4,
                dropout=dropout,
                activation="gelu",
                norm_first=False,
            )
            for _ in range(num_layers)
        ])

        # ----------------------------------------------------------------
        # MLP head: [B, token_dim] → [B, num_shifts]
        # Linear(D,D) → GELU → Linear(D,N)
        # Final layer near-zero init → softmax ≈ 1/N at step 0.
        # ----------------------------------------------------------------
        self.head: nn.Sequential = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_shifts),
        )
        self._init_head()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_shift_coords(self) -> None:
        """Initialise ``raw_shift_coords`` from the desired bounded coordinates.

        If ``num_shifts == 9``, initialise so that after the tanh
        activation the coordinates reproduce the standard 3×3 grid::

            (-1,-1), (-1,0), (-1,1),
            ( 0,-1), ( 0,0), ( 0,1),
            ( 1,-1), ( 1,0), ( 1,1)

        The inverse-tanh transform is used::

            raw = atanh(coord / max_shift_radius)
              where atanh(x) = 0.5 · log((1+x)/(1-x))

        For the zero coordinate (centre shift), the raw value is exactly
        0.0 so tanh(0) = 0 and the shift is exactly zero.

        For any other ``num_shifts``, initialise uniformly in [-1, 1]
        in raw space, which maps to approximately
        [-0.76, 0.76] × max_shift_radius in bounded space.
        """
        with torch.no_grad():
            if self.num_shifts == 9:
                # Standard 3×3 grid in bounded coordinate space.
                bounded: List[List[float]] = [
                    [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
                    [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0],
                    [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0],
                ]
                bounded_t: torch.Tensor = torch.tensor(
                    bounded, dtype=torch.float32
                )
                # Inverse tanh: atanh(coord / max_shift_radius).
                # Avoid atanh(0) = 0 and atanh(±1) = ±inf by clamping;
                # ±1 / max_shift_radius is safe as long as max_shift_radius > 1.
                scaled: torch.Tensor = bounded_t / self.max_shift_radius
                # atanh is numerically stable for |x| < 1.
                self.raw_shift_coords.data.copy_(torch.atanh(scaled))
            else:
                # Uniform in raw space → tanh maps to ≈ [-0.76, 0.76] × R.
                nn.init.uniform_(self.raw_shift_coords, -1.0, 1.0)

    def _init_head(self) -> None:
        """Initialise the MLP head.

        First layer: Kaiming-normal weight, zero bias.
        Final layer: Normal(0, 1e-3) weight, zero bias — ensures
        logits ≈ 0 at step 0, giving softmax ≈ 1/N.
        """
        first_linear: nn.Linear = self.head[0]  # type: ignore[assignment]
        final_linear: nn.Linear = self.head[2]  # type: ignore[assignment]
        nn.init.kaiming_normal_(first_linear.weight, nonlinearity="linear")
        nn.init.zeros_(first_linear.bias)
        nn.init.normal_(final_linear.weight, mean=0.0, std=_FINAL_LAYER_INIT_STD)
        nn.init.zeros_(final_linear.bias)

    def reset_parameters(self) -> None:
        """(Re-)initialise the MLP head weights.

        Does not reset ``raw_shift_coords``, ``cls_token``, or
        ``pos_embed`` to avoid overwriting learned shift geometry.
        Call ``_init_shift_coords()`` explicitly if a full reset is
        required.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> lscs.reset_parameters()
        """
        self._init_head()

    # ------------------------------------------------------------------
    # Bounded coordinate accessor
    # ------------------------------------------------------------------

    def get_shift_coordinates(self) -> torch.Tensor:
        """Return the current bounded shift coordinates.

        Returns
        -------
        torch.Tensor
            Shape ``[num_shifts, 2]`` (fp32). Entry ``[i, 0]`` is the
            row shift Δr_i and ``[i, 1]`` is the column shift Δc_i.
            Values lie strictly in
            ``(-max_shift_radius, max_shift_radius)``.

        Notes
        -----
        Gradients flow through this method into ``raw_shift_coords`` via
        the tanh activation. Do not call ``.detach()`` on the return
        value if you need to backpropagate through the coordinates.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=9, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=1,
        ... )
        >>> coords = lscs.get_shift_coordinates()
        >>> coords.shape
        torch.Size([9, 2])
        >>> bool((coords.abs() < lscs.max_shift_radius).all())
        True
        """
        return self.max_shift_radius * torch.tanh(self.raw_shift_coords)

    # ------------------------------------------------------------------
    # Differentiable shifting
    # ------------------------------------------------------------------

    def apply_shift(
        self,
        x: torch.Tensor,
        shift_row: torch.Tensor,
        shift_col: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a differentiable spatial shift to a 4-D tensor.

        Implements a fractional-pixel, gradient-safe spatial
        transformation using ``torch.nn.functional.grid_sample`` with
        bilinear interpolation and reflection padding.

        A positive ``shift_row`` moves the image content **downward**
        (the content that was at row *r* appears at row *r + shift_row*),
        which is achieved by sampling from row (*r − shift_row*).

        Parameters
        ----------
        x:
            Input tensor, shape ``[B, C, H, W]``.
        shift_row:
            Scalar tensor: row shift in pixels. May be fractional.
            Gradients flow through this argument into the caller's
            computation graph (e.g. into ``raw_shift_coords`` via the
            tanh activation in ``get_shift_coordinates``).
        shift_col:
            Scalar tensor: column shift in pixels. May be fractional.

        Returns
        -------
        torch.Tensor
            Shifted tensor, shape ``[B, C, H, W]``. Same dtype and device
            as ``x``.

        Notes
        -----
        The normalised coordinate convention used by ``grid_sample``
        (``align_corners=False``) maps pixel index *k* in a dimension of
        size *S* to normalised coordinate::

            u_k = -1 + (2k + 1) / S

        A shift of *Δ* pixels corresponds to a normalised offset of
        ``2·Δ/S``. Subtracting this offset from the sampling grid
        samples the image at position (k − Δ), achieving a shift of +Δ.

        ``padding_mode="reflection"`` avoids border discontinuities that
        would occur with zero-padding, which is particularly important
        for SAR images where the shift may sample beyond the image
        boundary.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> x = torch.randn(2, 1, 16, 16)
        >>> sr = torch.tensor(1.5)
        >>> sc = torch.tensor(-0.5)
        >>> y = lscs.apply_shift(x, sr, sc)
        >>> y.shape
        torch.Size([2, 1, 16, 16])
        """
        B, C, H, W = x.shape
        device: torch.device = x.device
        dtype: torch.dtype = x.dtype

        # Build base identity grid in [-1, 1] (align_corners=False).
        # grid[..., 0] = x-axis (columns), grid[..., 1] = y-axis (rows).
        lin_x: torch.Tensor = torch.linspace(
            -1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device=device, dtype=dtype
        )
        lin_y: torch.Tensor = torch.linspace(
            -1.0 + 1.0 / H, 1.0 - 1.0 / H, H, device=device, dtype=dtype
        )
        grid_y, grid_x = torch.meshgrid(lin_y, lin_x, indexing="ij")
        # base_grid: [H, W, 2]
        base_grid: torch.Tensor = torch.stack([grid_x, grid_y], dim=-1)
        # Expand to [B, H, W, 2]
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)

        # Normalised shift offsets.
        # Subtracting the offset samples from (coord − offset), i.e. shifts
        # the image content in the positive direction.
        norm_row: torch.Tensor = shift_row * (2.0 / H)  # scalar
        norm_col: torch.Tensor = shift_col * (2.0 / W)  # scalar

        # Build shifted grid (no in-place ops).
        shifted_x: torch.Tensor = base_grid[..., 0] - norm_col
        shifted_y: torch.Tensor = base_grid[..., 1] - norm_row
        grid: torch.Tensor = torch.stack([shifted_x, shifted_y], dim=-1)

        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=False,
        )

    def inverse_shift(
        self,
        x: torch.Tensor,
        shift_row: torch.Tensor,
        shift_col: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the inverse of a spatial shift.

        Equivalent to ``apply_shift(x, -shift_row, -shift_col)``.

        Parameters
        ----------
        x:
            Input tensor, shape ``[B, C, H, W]``.
        shift_row:
            Row shift in pixels (the *original* shift, not its negation).
        shift_col:
            Column shift in pixels (the *original* shift, not its negation).

        Returns
        -------
        torch.Tensor
            Inverse-shifted tensor, shape ``[B, C, H, W]``.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> x = torch.randn(2, 1, 16, 16)
        >>> sr = torch.tensor(1.0)
        >>> sc = torch.tensor(2.0)
        >>> y = lscs.inverse_shift(x, sr, sc)
        >>> y.shape
        torch.Size([2, 1, 16, 16])
        """
        return self.apply_shift(x, -shift_row, -shift_col)

    # ------------------------------------------------------------------
    # Coordinate regularization
    # ------------------------------------------------------------------

    def radius_regularizer(self) -> torch.Tensor:
        """Compute the radius regularization loss.

        Penalises large shift magnitudes::

            L_r = Σ_i ‖coords_i‖²

        scaled by ``self.radius_lambda``.

        Returns
        -------
        torch.Tensor
            Scalar tensor. Retains the autograd graph so that
            ``L_r.backward()`` propagates gradients into
            ``raw_shift_coords`` through the tanh activation.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=9, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=1,
        ... )
        >>> lr = lscs.radius_regularizer()
        >>> lr.shape
        torch.Size([])
        >>> bool(lr.item() >= 0.0)
        True
        """
        coords: torch.Tensor = self.get_shift_coordinates()  # [N, 2]
        return self.radius_lambda * (coords * coords).sum()

    def repulsion_regularizer(self) -> torch.Tensor:
        """Compute the repulsion regularization loss.

        Encourages shift diversity by penalising nearby pairs::

            L_rep = Σ_{i≠j} exp(-d_{ij})

        where ``d_{ij} = ‖coords_i − coords_j‖₂``, scaled by
        ``self.repulsion_lambda``.

        The exponential decays with distance, so the penalty is strongest
        for nearly coincident shifts. This gently encourages the learned
        grid to be spread out without imposing a rigid geometry.

        Returns
        -------
        torch.Tensor
            Scalar tensor. Retains the autograd graph.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=9, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=1,
        ... )
        >>> lr = lscs.repulsion_regularizer()
        >>> lr.shape
        torch.Size([])
        >>> bool(lr.item() >= 0.0)
        True
        """
        coords: torch.Tensor = self.get_shift_coordinates()  # [N, 2]
        # Pairwise squared distances: [N, N]
        diff: torch.Tensor = coords.unsqueeze(0) - coords.unsqueeze(1)  # [N, N, 2]
        sq_dist: torch.Tensor = (diff * diff).sum(dim=-1)  # [N, N]
        dist: torch.Tensor = torch.sqrt(sq_dist + _DIST_EPS)  # [N, N]
        # Mask the diagonal (i == j) by zeroing it.
        mask: torch.Tensor = 1.0 - torch.eye(
            self.num_shifts, device=coords.device, dtype=coords.dtype
        )
        repulsion: torch.Tensor = (torch.exp(-dist) * mask).sum()
        return self.repulsion_lambda * repulsion

    def coordinate_regularizer(self) -> torch.Tensor:
        """Return the total coordinate regularization loss.

        Sum of the radius and repulsion terms::

            L_coord = L_r + L_rep

        This is the quantity to add to the despeckling training loss
        when jointly optimising the shift geometry::

            loss = diffusion_loss + lscs.coordinate_regularizer()

        Returns
        -------
        torch.Tensor
            Scalar tensor. Retains the autograd graph.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=9, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=1,
        ... )
        >>> lc = lscs.coordinate_regularizer()
        >>> lc.shape
        torch.Size([])
        """
        return self.radius_regularizer() + self.repulsion_regularizer()

    # ------------------------------------------------------------------
    # Coordinate statistics
    # ------------------------------------------------------------------

    def shift_distance_matrix(self) -> torch.Tensor:
        """Return the pairwise Euclidean distance matrix between shift coordinates.

        Returns
        -------
        torch.Tensor
            Shape ``[num_shifts, num_shifts]``. Entry [i, j] is
            ‖coords_i − coords_j‖₂. Diagonal entries are 0. Detached.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=9, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=1,
        ... )
        >>> D = lscs.shift_distance_matrix()
        >>> D.shape
        torch.Size([9, 9])
        >>> bool((D.diagonal() == 0.0).all())
        True
        """
        with torch.no_grad():
            coords: torch.Tensor = self.get_shift_coordinates()
            diff: torch.Tensor = coords.unsqueeze(0) - coords.unsqueeze(1)
            sq_dist: torch.Tensor = (diff * diff).sum(dim=-1)
            return torch.sqrt(sq_dist + _DIST_EPS) - math.sqrt(_DIST_EPS)

    def average_shift_radius(self) -> float:
        """Return the mean Euclidean distance of all shifts from the origin.

        Returns
        -------
        float
            Mean ‖coords_i‖₂ over i. Detached scalar.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=9, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=1,
        ... )
        >>> r = lscs.average_shift_radius()
        >>> isinstance(r, float)
        True
        """
        with torch.no_grad():
            coords: torch.Tensor = self.get_shift_coordinates()
            radii: torch.Tensor = torch.sqrt((coords * coords).sum(dim=-1))
            return float(radii.mean().item())

    def max_shift_radius_used(self) -> float:
        """Return the maximum Euclidean distance of any shift from the origin.

        Returns
        -------
        float
            Max ‖coords_i‖₂ over i. Detached scalar.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=9, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=1,
        ... )
        >>> r = lscs.max_shift_radius_used()
        >>> isinstance(r, float)
        True
        """
        with torch.no_grad():
            coords: torch.Tensor = self.get_shift_coordinates()
            radii: torch.Tensor = torch.sqrt((coords * coords).sum(dim=-1))
            return float(radii.max().item())

    def min_pairwise_distance(self) -> float:
        """Return the minimum pairwise Euclidean distance between any two shifts.

        Returns
        -------
        float
            Min d_{ij} for i ≠ j. Detached scalar. Useful for monitoring
            whether two shifts have collapsed to the same position.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=9, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=1,
        ... )
        >>> d = lscs.min_pairwise_distance()
        >>> isinstance(d, float)
        True
        """
        D: torch.Tensor = self.shift_distance_matrix()  # already no_grad
        # Mask diagonal with a large value before taking the min.
        large: float = float(D.max().item()) + 1.0
        D_off: torch.Tensor = D + torch.eye(
            self.num_shifts, device=D.device, dtype=D.dtype
        ) * large
        return float(D_off.min().item())

    def coordinate_variance(self) -> float:
        """Return the population variance of the shift coordinates.

        Computes the variance over all 2N scalar coordinate values.

        Returns
        -------
        float
            Population variance (``unbiased=False``). Detached scalar.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=9, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=1,
        ... )
        >>> v = lscs.coordinate_variance()
        >>> isinstance(v, float)
        True
        """
        with torch.no_grad():
            coords: torch.Tensor = self.get_shift_coordinates()
            return float(coords.var(unbiased=False).item())

    def coordinate_entropy(self) -> float:
        """Return the differential entropy proxy of the shift coordinate distribution.

        Computed as the log of the coordinate variance (a rough measure
        of how spread out the shift grid is). Not to be confused with the
        Shannon entropy of the aggregation weights.

        Returns
        -------
        float
            0.5 · log(variance + eps). Detached scalar.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=9, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=1,
        ... )
        >>> e = lscs.coordinate_entropy()
        >>> isinstance(e, float)
        True
        """
        var: float = self.coordinate_variance()
        return 0.5 * math.log(var + self.eps)

    def save_shift_statistics(self) -> Dict[str, float]:
        """Return a detached statistics snapshot of the current shift geometry.

        Returns
        -------
        Dict[str, float]
            Keys:

            ``"avg_radius"``
                Mean ‖coords_i‖₂.
            ``"max_radius"``
                Max ‖coords_i‖₂.
            ``"coord_variance"``
                Population variance of all 2N coordinate values.
            ``"min_pairwise_distance"``
                Minimum pairwise distance d_{ij}, i ≠ j.
            ``"mean_row"``
                Mean row coordinate (Δr) over all shifts.
            ``"mean_col"``
                Mean column coordinate (Δc) over all shifts.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=9, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=1,
        ... )
        >>> s = lscs.save_shift_statistics()
        >>> set(s.keys()) == {
        ...     "avg_radius", "max_radius", "coord_variance",
        ...     "min_pairwise_distance", "mean_row", "mean_col"
        ... }
        True
        """
        with torch.no_grad():
            coords: torch.Tensor = self.get_shift_coordinates()
            radii: torch.Tensor = torch.sqrt((coords * coords).sum(dim=-1))
        return {
            "avg_radius": float(radii.mean().item()),
            "max_radius": float(radii.max().item()),
            "coord_variance": self.coordinate_variance(),
            "min_pairwise_distance": self.min_pairwise_distance(),
            "mean_row": float(coords[:, 0].mean().item()),
            "mean_col": float(coords[:, 1].mean().item()),
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_shift_coordinates(self) -> matplotlib.figure.Figure:
        """Return a scatter-plot figure of the current learned shift coordinates.

        Plots each shift as a point in (row, col) space with index
        annotation. The origin (0, 0) is highlighted. Equal-aspect ratio
        and grid are enabled so the geometry is visually faithful.

        Returns
        -------
        matplotlib.figure.Figure
            A matplotlib Figure that can be displayed with
            ``plt.show()`` or saved with ``fig.savefig(...)``.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=9, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=1,
        ... )
        >>> fig = lscs.plot_shift_coordinates()
        >>> import matplotlib.figure
        >>> isinstance(fig, matplotlib.figure.Figure)
        True
        >>> import matplotlib.pyplot as plt
        >>> plt.close(fig)
        """
        with torch.no_grad():
            coords: torch.Tensor = self.get_shift_coordinates().cpu()
        rows: List[float] = coords[:, 0].tolist()
        cols: List[float] = coords[:, 1].tolist()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(cols, rows, s=80, zorder=3)
        ax.scatter([0.0], [0.0], s=120, marker="+", color="red",
                   linewidths=2, zorder=4, label="origin")
        for idx, (r, c) in enumerate(zip(rows, cols)):
            ax.annotate(str(idx), (c, r), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)
        lim: float = float(self.max_shift_radius) * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlabel("Column shift (Δc)")
        ax.set_ylabel("Row shift (Δr)")
        ax.set_title(f"A26g Learned Shift Coordinates (N={self.num_shifts})")
        ax.legend()
        fig.tight_layout()
        return fig

    def save_shift_plot(self, path: str) -> None:
        """Save the shift-coordinate scatter plot to a PNG file.

        Parameters
        ----------
        path:
            File path (including ``.png`` extension). Created or
            overwritten.

        Examples
        --------
        >>> import tempfile, os
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     out = os.path.join(tmp, "shifts.png")
        ...     lscs.save_shift_plot(out)
        ...     saved = os.path.exists(out)
        >>> saved
        True
        """
        fig: matplotlib.figure.Figure = self.plot_shift_coordinates()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Freeze utilities (coordinates only)
    # ------------------------------------------------------------------

    def freeze_coordinates(self) -> None:
        """Freeze ``raw_shift_coords`` so gradients are not computed for it.

        Leaves all other parameters (Transformer blocks, head, embeddings)
        unaffected.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> lscs.freeze_coordinates()
        >>> lscs.coordinates_frozen()
        True
        """
        self.raw_shift_coords.requires_grad_(False)

    def unfreeze_coordinates(self) -> None:
        """Unfreeze ``raw_shift_coords`` so gradients are computed for it.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> lscs.freeze_coordinates()
        >>> lscs.unfreeze_coordinates()
        >>> lscs.coordinates_frozen()
        False
        """
        self.raw_shift_coords.requires_grad_(True)

    def coordinates_frozen(self) -> bool:
        """Return ``True`` if ``raw_shift_coords.requires_grad`` is ``False``.

        Returns
        -------
        bool

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> lscs.coordinates_frozen()
        False
        >>> lscs.freeze_coordinates()
        >>> lscs.coordinates_frozen()
        True
        """
        return not self.raw_shift_coords.requires_grad

    # ------------------------------------------------------------------
    # Freeze utilities (full module)
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Disable gradient updates for all learnable parameters.

        Freezes ``raw_shift_coords``, ``coordinate_embedding``,
        ``cls_token``, ``pos_embed``, ``blocks``, and ``head``.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> lscs.freeze()
        >>> lscs.is_frozen()
        True
        """
        self.raw_shift_coords.requires_grad_(False)
        for param in self.coordinate_embedding.parameters():
            param.requires_grad_(False)
        self.cls_token.requires_grad_(False)
        self.pos_embed.requires_grad_(False)
        for param in self.blocks.parameters():
            param.requires_grad_(False)
        for param in self.head.parameters():
            param.requires_grad_(False)

    def unfreeze(self) -> None:
        """Enable gradient updates for all learnable parameters.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> lscs.freeze()
        >>> lscs.unfreeze()
        >>> lscs.is_frozen()
        False
        """
        self.raw_shift_coords.requires_grad_(True)
        for param in self.coordinate_embedding.parameters():
            param.requires_grad_(True)
        self.cls_token.requires_grad_(True)
        self.pos_embed.requires_grad_(True)
        for param in self.blocks.parameters():
            param.requires_grad_(True)
        for param in self.head.parameters():
            param.requires_grad_(True)

    def is_frozen(self) -> bool:
        """Return ``True`` if every learnable parameter has ``requires_grad=False``.

        Returns
        -------
        bool

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=1, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> lscs.is_frozen()
        False
        >>> lscs.freeze()
        >>> lscs.is_frozen()
        True
        """
        all_params = (
            [self.raw_shift_coords, self.cls_token, self.pos_embed]
            + list(self.coordinate_embedding.parameters())
            + list(self.blocks.parameters())
            + list(self.head.parameters())
        )
        return all(not p.requires_grad for p in all_params)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_outputs(self, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """Validate the ``outputs`` sequence and return the reference tensor.

        Checks: non-empty, correct length, every tensor 4-D with
        ``self.channels`` channels, consistent shape / dtype / device.

        Parameters
        ----------
        outputs:
            Candidate sequence of cycle-shifted prediction tensors.

        Returns
        -------
        torch.Tensor
            ``outputs[0]`` (the reference tensor).

        Raises
        ------
        ValueError
            If any check fails.
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
                    f"outputs[{idx}] has ndim={tensor.ndim}."
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
        """Validate ``confidence_maps`` against the reference tensor.

        Checks: non-empty, correct length, every map 4-D with exactly
        one channel, consistent batch size, spatial dims, dtype, device.

        Parameters
        ----------
        confidence_maps:
            Candidate sequence of per-shift confidence maps.
        reference:
            Validated reference from ``_validate_outputs``.

        Raises
        ------
        ValueError
            If any check fails.
        """
        if len(confidence_maps) == 0:
            raise ValueError(
                "confidence_maps must be a non-empty sequence, got length 0."
            )
        if len(confidence_maps) != self.num_shifts:
            raise ValueError(
                f"len(confidence_maps) must equal num_shifts={self.num_shifts}, "
                f"got {len(confidence_maps)}."
            )
        ref_B: int = reference.shape[0]
        ref_H: int = reference.shape[2]
        ref_W: int = reference.shape[3]
        ref_dtype: torch.dtype = reference.dtype
        ref_device: torch.device = reference.device
        for idx, conf in enumerate(confidence_maps):
            if conf.ndim != 4:
                raise ValueError(
                    f"Each confidence map must be 4-D [B,1,H,W]; "
                    f"confidence_maps[{idx}] has ndim={conf.ndim}."
                )
            if conf.shape[1] != _CONFIDENCE_CHANNELS:
                raise ValueError(
                    f"Each confidence map must have exactly 1 channel; "
                    f"confidence_maps[{idx}] has {conf.shape[1]}."
                )
            if conf.shape[0] != ref_B:
                raise ValueError(
                    f"Batch size mismatch: outputs[0].shape[0]={ref_B} but "
                    f"confidence_maps[{idx}].shape[0]={conf.shape[0]}."
                )
            if conf.shape[2] != ref_H or conf.shape[3] != ref_W:
                raise ValueError(
                    f"Spatial dimension mismatch: outputs[0] has ({ref_H},{ref_W}) "
                    f"but confidence_maps[{idx}] has "
                    f"({conf.shape[2]},{conf.shape[3]})."
                )
            if conf.dtype != ref_dtype:
                raise ValueError(
                    f"dtype mismatch: outputs[0].dtype={ref_dtype} but "
                    f"confidence_maps[{idx}].dtype={conf.dtype}."
                )
            if conf.device != ref_device:
                raise ValueError(
                    f"Device mismatch: outputs[0].device={ref_device} but "
                    f"confidence_maps[{idx}].device={conf.device}."
                )

    def _validate_wavelet_features(
        self,
        wavelet_features: Sequence[torch.Tensor],
        reference: torch.Tensor,
    ) -> None:
        """Validate ``wavelet_features`` against the reference tensor.

        Checks: non-empty, correct length, every tensor 4-D with
        ``self.wavelet_channels`` channels, consistent batch size /
        dtype / device, all tensors share the same shape (spatial dims
        may differ from outputs).

        Parameters
        ----------
        wavelet_features:
            Candidate sequence of per-shift wavelet tensors.
        reference:
            Validated reference from ``_validate_outputs``.

        Raises
        ------
        ValueError
            If any check fails.
        """
        if len(wavelet_features) == 0:
            raise ValueError(
                "wavelet_features must be a non-empty sequence, got length 0."
            )
        if len(wavelet_features) != self.num_shifts:
            raise ValueError(
                f"len(wavelet_features) must equal num_shifts={self.num_shifts}, "
                f"got {len(wavelet_features)}."
            )
        ref_B: int = reference.shape[0]
        ref_dtype: torch.dtype = reference.dtype
        ref_device: torch.device = reference.device
        first_wav: torch.Tensor = wavelet_features[0]
        if first_wav.ndim != 4:
            raise ValueError(
                f"Each wavelet tensor must be 4-D; "
                f"wavelet_features[0] has ndim={first_wav.ndim}."
            )
        if first_wav.shape[1] != self.wavelet_channels:
            raise ValueError(
                f"wavelet_features[0] has {first_wav.shape[1]} channels; "
                f"expected wavelet_channels={self.wavelet_channels}."
            )
        if first_wav.shape[0] != ref_B:
            raise ValueError(
                f"Batch size mismatch in wavelet_features[0]."
            )
        if first_wav.dtype != ref_dtype or first_wav.device != ref_device:
            raise ValueError(
                f"dtype/device mismatch in wavelet_features[0]."
            )
        wav_ref_shape: torch.Size = first_wav.shape
        for idx, wav in enumerate(wavelet_features[1:], start=1):
            if wav.shape != wav_ref_shape:
                raise ValueError(
                    f"All wavelet tensors must share the same shape; "
                    f"wavelet_features[0].shape={wav_ref_shape} but "
                    f"wavelet_features[{idx}].shape={wav.shape}."
                )
            if wav.dtype != ref_dtype or wav.device != ref_device:
                raise ValueError(
                    f"dtype/device mismatch in wavelet_features[{idx}]."
                )

    def _validate_structure_features(
        self,
        structure_features: Sequence[torch.Tensor],
        reference: torch.Tensor,
    ) -> None:
        """Validate ``structure_features`` against the reference tensor.

        Checks: non-empty, correct length, every tensor 4-D with
        ``self.structure_channels`` channels, consistent batch size /
        dtype / device, all tensors share the same shape (spatial dims
        may differ from outputs).

        Parameters
        ----------
        structure_features:
            Candidate sequence of per-shift structure tensor descriptors.
        reference:
            Validated reference from ``_validate_outputs``.

        Raises
        ------
        ValueError
            If any check fails.
        """
        if len(structure_features) == 0:
            raise ValueError(
                "structure_features must be a non-empty sequence, got length 0."
            )
        if len(structure_features) != self.num_shifts:
            raise ValueError(
                f"len(structure_features) must equal num_shifts={self.num_shifts}, "
                f"got {len(structure_features)}."
            )
        ref_B: int = reference.shape[0]
        ref_dtype: torch.dtype = reference.dtype
        ref_device: torch.device = reference.device
        first_st: torch.Tensor = structure_features[0]
        if first_st.ndim != 4:
            raise ValueError(
                f"Each structure tensor must be 4-D; "
                f"structure_features[0] has ndim={first_st.ndim}."
            )
        if first_st.shape[1] != self.structure_channels:
            raise ValueError(
                f"structure_features[0] has {first_st.shape[1]} channels; "
                f"expected structure_channels={self.structure_channels}."
            )
        if first_st.shape[0] != ref_B:
            raise ValueError(
                f"Batch size mismatch in structure_features[0]."
            )
        if first_st.dtype != ref_dtype or first_st.device != ref_device:
            raise ValueError(
                f"dtype/device mismatch in structure_features[0]."
            )
        st_ref_shape: torch.Size = first_st.shape
        for idx, st in enumerate(structure_features[1:], start=1):
            if st.shape != st_ref_shape:
                raise ValueError(
                    f"All structure tensors must share the same shape; "
                    f"structure_features[0].shape={st_ref_shape} but "
                    f"structure_features[{idx}].shape={st.shape}."
                )
            if st.dtype != ref_dtype or st.device != ref_device:
                raise ValueError(
                    f"dtype/device mismatch in structure_features[{idx}]."
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
        """Construct the Transformer input token sequence.

        For each shift *i*, computes five branch descriptors and
        concatenates them to form the shift token ``d_i``::

            z_i = pool(x_i).reshape(B, C)          image descriptor
            c_i = pool(σ_i).reshape(B, 1)          confidence descriptor
            v_i = pool(W_i).reshape(B, Cw)         wavelet descriptor
            s_i = pool(S_i).reshape(B, Cs)         structure descriptor
            g_i = coord_embed(coords[i])            coordinate descriptor

            d_i = concat(z_i, c_i, v_i, s_i, g_i)  [B, D]

        Stacks the N shift tokens to ``[B, N, D]``, prepends the CLS
        token, and adds the positional embedding.

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
            Token sequence of shape ``[B, N+1, D]``, cast to the
            Transformer block parameter dtype (fp32 by default).
        """
        # Get bounded coordinates: [N, 2]  (differentiable)
        coords: torch.Tensor = self.get_shift_coordinates()

        # Cast coords to the block's dtype for the embedding MLP.
        block_dtype: torch.dtype = next(self.blocks[0].parameters()).dtype
        coords_cast: torch.Tensor = coords.to(block_dtype)

        shift_tokens: List[torch.Tensor] = []
        for i, (x_i, sig_i, wav_i, st_i) in enumerate(
            zip(outputs, confidence_maps, wavelet_features, structure_features)
        ):
            batch_size: int = x_i.shape[0]

            z_i: torch.Tensor = self.pool(x_i).reshape(batch_size, -1)     # [B, C]
            c_i: torch.Tensor = self.pool(sig_i).reshape(batch_size, -1)   # [B, 1]
            v_i: torch.Tensor = self.pool(wav_i).reshape(batch_size, -1)   # [B, Cw]
            s_i: torch.Tensor = self.pool(st_i).reshape(batch_size, -1)    # [B, Cs]

            # Coordinate embedding: [2] → [Ge], broadcast to [B, Ge]
            g_i: torch.Tensor = self.coordinate_embedding(
                coords_cast[i]
            ).unsqueeze(0).expand(batch_size, -1)                           # [B, Ge]

            d_i: torch.Tensor = torch.cat([z_i, c_i, v_i, s_i, g_i], dim=1)  # [B, D]
            shift_tokens.append(d_i)

        # Stack: [B, N, D]
        token_matrix: torch.Tensor = torch.stack(shift_tokens, dim=1)

        # Cast inputs to block dtype if needed (fp16 inputs → fp32 tokens).
        if token_matrix.dtype != block_dtype:
            token_matrix = token_matrix.to(block_dtype)

        batch_size_final: int = token_matrix.shape[0]

        # Expand CLS token: [1, 1, D] → [B, 1, D]
        cls_expanded: torch.Tensor = self.cls_token.expand(
            batch_size_final, 1, self.token_dim
        )

        # Prepend CLS and add positional embedding: [B, N+1, D]
        tokens: torch.Tensor = (
            torch.cat([cls_expanded, token_matrix], dim=1) + self.pos_embed
        )
        return tokens

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

        Builds the shift token sequence (including coordinate embeddings),
        passes it through all :class:`TransformerBlock` layers, extracts
        the CLS token representation, applies the MLP head to obtain
        per-shift logits, and returns a temperature-scaled softmax over
        the shift dimension.

        Gradients flow through the CLS token → Transformer → head →
        softmax → (forward pass only; the weights are not attached to
        raw_shift_coords here). To train raw_shift_coords, include
        ``coordinate_regularizer()`` in the total loss and rely on
        ``apply_shift`` / ``inverse_shift`` being called with the
        differentiable coordinates by the outer training loop.

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
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> outs  = [torch.randn(3, 4, 16, 16) for _ in range(4)]
        >>> confs = [torch.rand(3, 1, 16, 16) for _ in range(4)]
        >>> wavs  = [torch.randn(3, 4, 8, 8) for _ in range(4)]
        >>> sts   = [torch.randn(3, 4, 16, 16) for _ in range(4)]
        >>> w = lscs.get_weights(outs, confs, wavs, sts)
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

        tokens: torch.Tensor = self._build_token_sequence(
            outputs, confidence_maps, wavelet_features, structure_features
        )  # [B, N+1, D]

        for block in self.blocks:
            tokens = block(tokens)

        cls_feat: torch.Tensor = tokens[:, 0, :]     # [B, D]
        logits: torch.Tensor = self.head(cls_feat)   # [B, N]
        return F.softmax(logits / self.temperature, dim=1)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        outputs: Sequence[torch.Tensor],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted sum of the shifted outputs.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``.
        weights:
            Softmax weights, shape ``[B, N]``.

        Returns
        -------
        torch.Tensor
            Fused tensor ``x̂ = Σ_i w_i · x_i``, shape ``[B, C, H, W]``.
        """
        batch_size: int = weights.shape[0]
        # Stack: [N, B, C, H, W] → [B, N, C, H, W]
        stacked: torch.Tensor = torch.stack(list(outputs), dim=0).permute(
            1, 0, 2, 3, 4
        )
        weights_broadcast: torch.Tensor = weights.view(
            batch_size, self.num_shifts, 1, 1, 1
        )
        input_dtype: torch.dtype = stacked.dtype
        if stacked.dtype != weights_broadcast.dtype:
            stacked = stacked.to(weights_broadcast.dtype)
        fused: torch.Tensor = (stacked * weights_broadcast).sum(dim=1)
        return fused.to(input_dtype)

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
        """Aggregate cycle-shifted diffusion outputs using learnable shifts and Transformer weights.

        Parameters
        ----------
        outputs:
            Sequence of *N* tensors, each ``[B, C, H, W]``, with
            identical shape, dtype, device, and
            ``shape[1] == self.channels``.

            These are the per-shift denoiser outputs *after* the inverse
            shift has been applied (matching the convention in
            ``inference_sar.py``). A26g's ``apply_shift`` /
            ``inverse_shift`` methods are provided as utilities for
            callers that want differentiable shifting inside this module.
        confidence_maps:
            Sequence of *N* tensors, each ``[B, 1, H, W]``, matching
            ``outputs`` in batch size, spatial dims, dtype, and device.
        wavelet_features:
            Sequence of *N* tensors, each ``[B, Cw, Hw, Ww]``, matching
            ``outputs`` in batch size, dtype, and device. Spatial dims
            may differ.
        structure_features:
            Sequence of *N* tensors, each ``[B, Cs, Hs, Ws]``, matching
            ``outputs`` in batch size, dtype, and device. Spatial dims
            may differ.
        return_weights:
            If ``False`` (default), return only the fused tensor.
            If ``True``, return ``(fused, weights)``.

        Returns
        -------
        torch.Tensor or tuple of (torch.Tensor, torch.Tensor)
            * ``return_weights=False``: fused tensor ``[B, C, H, W]``.
            * ``return_weights=True``: ``(fused, weights)`` where
              ``fused`` has shape ``[B, C, H, W]`` and ``weights`` has
              shape ``[B, num_shifts]``.

        Raises
        ------
        ValueError
            If any input sequence fails length, shape, dtype, device, or
            channel-count validation.

        Notes
        -----
        The full forward pass::

            coords   = max_shift_radius · tanh(raw_shift_coords)   [N, 2]
            g_i      = coordinate_embedding(coords[i])             [B, Ge]
            d_i      = concat(GAP(x_i), GAP(σ_i), GAP(W_i),
                               GAP(S_i), g_i)                      [B, D]
            tokens   = [CLS | d_1 | … | d_N] + pos_embed          [B, N+1, D]
            for blk in blocks:
                tokens = blk(tokens)                               [B, N+1, D]
            cls      = tokens[:, 0, :]                             [B, D]
            a        = head(cls)                                   [B, N]
            w        = softmax(a / τ, dim=1)                       [B, N]
            x̂        = Σ_i w_i · x_i                              [B, C, H, W]

        Only the image content ``x_i`` enters the weighted sum; the
        confidence, wavelet, structure, and coordinate descriptors are
        used solely to drive the Transformer weight predictor.

        Examples
        --------
        >>> import torch
        >>> from structdiff.inference.learnable_shift_cycle_spinning import (
        ...     LearnableShiftCycleSpinning,
        ... )
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> outputs = [torch.ones(2, 4, 8, 8) * float(i + 1) for i in range(4)]
        >>> confs   = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs    = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts     = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> fused   = lscs(outputs, confs, wavs, sts)
        >>> fused.shape
        torch.Size([2, 4, 8, 8])
        >>> fused, w = lscs(outputs, confs, wavs, sts, return_weights=True)
        >>> w.shape
        torch.Size([2, 4])
        >>> bool(torch.allclose(w.sum(dim=1), torch.ones(2), atol=1e-5))
        True
        """
        reference: torch.Tensor = self._validate_outputs(outputs)
        self._validate_confidence_maps(confidence_maps, reference)
        self._validate_wavelet_features(wavelet_features, reference)
        self._validate_structure_features(structure_features, reference)

        weights: torch.Tensor = self.get_weights(
            outputs, confidence_maps, wavelet_features, structure_features
        )
        fused: torch.Tensor = self._aggregate(outputs, weights)

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

            H_b = -Σ_i w_{b,i} · log(w_{b,i} + eps)

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
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> outs  = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs  = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts   = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> h = lscs.entropy(outs, confs, wavs, sts)
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

        Returns ``coefficient * H`` where H is the batch-averaged Shannon
        entropy of the predicted weight distributions.

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
            Scalar multiplier. Positive → encourage uniform weights.
            Negative → encourage peaked weights. Default 1.0.

        Returns
        -------
        torch.Tensor
            Scalar tensor. Retains the autograd graph.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> outs  = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs  = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts   = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> reg = lscs.entropy_regularizer(outs, confs, wavs, sts, 0.01)
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
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> outs  = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs  = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts   = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> n_eff = lscs.effective_num_shifts(outs, confs, wavs, sts)
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
            Scalar tensor. Retains the autograd graph.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> outs  = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs  = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts   = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> v = lscs.weight_variance(outs, confs, wavs, sts)
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
        """Return the index of the highest-weight shift per batch element.

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
            Shape ``[B]``, dtype ``int64``. Detached.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> outs  = [torch.randn(3, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(3, 1, 8, 8) for _ in range(4)]
        >>> wavs  = [torch.randn(3, 4, 4, 4) for _ in range(4)]
        >>> sts   = [torch.randn(3, 4, 8, 8) for _ in range(4)]
        >>> idx = lscs.max_weight_index(outs, confs, wavs, sts)
        >>> idx.shape
        torch.Size([3])
        """
        return self._weights_no_grad(
            outputs, confidence_maps, wavelet_features, structure_features
        ).argmax(dim=1)

    def min_weight_index(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
        structure_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Return the index of the lowest-weight shift per batch element.

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
            Shape ``[B]``, dtype ``int64``. Detached.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> outs  = [torch.randn(3, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(3, 1, 8, 8) for _ in range(4)]
        >>> wavs  = [torch.randn(3, 4, 4, 4) for _ in range(4)]
        >>> sts   = [torch.randn(3, 4, 8, 8) for _ in range(4)]
        >>> idx = lscs.min_weight_index(outs, confs, wavs, sts)
        >>> idx.shape
        torch.Size([3])
        """
        return self._weights_no_grad(
            outputs, confidence_maps, wavelet_features, structure_features
        ).argmin(dim=1)

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
            Shape ``[batch_size, num_shifts]``, all entries ``1/N``.
            Not connected to the autograd graph.

        Raises
        ------
        ValueError
            If ``batch_size`` is not a positive integer.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> u = lscs.uniform_weights(batch_size=3)
        >>> u.shape
        torch.Size([3, 4])
        >>> bool(torch.allclose(u, torch.full((3, 4), 0.25)))
        True
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive integer, got {batch_size!r}."
            )
        ref_param: torch.Tensor = next(self.head.parameters())
        return torch.full(
            (batch_size, self.num_shifts),
            1.0 / self.num_shifts,
            device=ref_param.device,
            dtype=ref_param.dtype,
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
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> outs  = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs  = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts   = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> kl = lscs.kl_to_uniform(outs, confs, wavs, sts)
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
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> outs  = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs  = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts   = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> jsd = lscs.js_to_uniform(outs, confs, wavs, sts)
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
        return (0.5 * (kl_w_m + kl_u_m)).mean()

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
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> lscs.set_temperature(0.5)
        >>> lscs.temperature
        0.5
        >>> try:
        ...     lscs.set_temperature(0.0)
        ... except ValueError as e:
        ...     print("caught:", e)
        caught: temperature must be strictly positive, got 0.0.
        """
        if temperature <= 0.0:
            raise ValueError(
                f"temperature must be strictly positive, got {temperature}."
            )
        self.temperature = temperature

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
        """Compute weights without retaining the autograd graph.

        Used internally by logging and diagnostic methods.

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
            Shape ``[B, num_shifts]``, detached.
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
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> outs  = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs  = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts   = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> stats = lscs.weight_statistics(outs, confs, wavs, sts)
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
        """Return a comprehensive diagnostic summary.

        Extends ``weight_statistics`` with coordinate geometry metrics.

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
            ``"max_weight"``, ``"min_weight"``, ``"weight_variance"``,
            ``"avg_shift_radius"``, ``"max_shift_radius"``,
            ``"coordinate_variance"``, ``"min_pairwise_distance"``.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> outs  = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs  = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts   = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> s = lscs.summary(outs, confs, wavs, sts)
        >>> set(s.keys()) == {
        ...     "entropy", "effective_num_shifts",
        ...     "max_weight", "min_weight", "weight_variance",
        ...     "avg_shift_radius", "max_shift_radius",
        ...     "coordinate_variance", "min_pairwise_distance"
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
            "avg_shift_radius": self.average_shift_radius(),
            "max_shift_radius": self.max_shift_radius_used(),
            "coordinate_variance": self.coordinate_variance(),
            "min_pairwise_distance": self.min_pairwise_distance(),
        }

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
            ``"min_weight_index"``, ``"avg_radius"``, ``"max_radius"``,
            ``"coord_variance"``, ``"min_pairwise_distance"``,
            ``"mean_row"``, ``"mean_col"``.

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=4, channels=4, wavelet_channels=4,
        ...     structure_channels=4, coordinate_embed_dim=4, num_heads=4,
        ... )
        >>> outs  = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> confs = [torch.rand(2, 1, 8, 8) for _ in range(4)]
        >>> wavs  = [torch.randn(2, 4, 4, 4) for _ in range(4)]
        >>> sts   = [torch.randn(2, 4, 8, 8) for _ in range(4)]
        >>> s = lscs.save_statistics(outs, confs, wavs, sts)
        >>> set(s.keys()) == {
        ...     "entropy", "effective_num_shifts", "kl_to_uniform",
        ...     "max_weight", "min_weight", "weight_variance",
        ...     "max_weight_index", "min_weight_index",
        ...     "avg_radius", "max_radius", "coord_variance",
        ...     "min_pairwise_distance", "mean_row", "mean_col",
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

        coord_stats: Dict[str, float] = self.save_shift_statistics()

        return {
            "entropy": float(entropy_val.item()),
            "effective_num_shifts": float(torch.exp(entropy_val).item()),
            "kl_to_uniform": float(kl_val.item()),
            "max_weight": float(weights.max().item()),
            "min_weight": float(weights.min().item()),
            "weight_variance": float(weights.var(unbiased=False).item()),
            "max_weight_index": float(weights[0].argmax().item()),
            "min_weight_index": float(weights[0].argmin().item()),
            "avg_radius": coord_stats["avg_radius"],
            "max_radius": coord_stats["max_radius"],
            "coord_variance": coord_stats["coord_variance"],
            "min_pairwise_distance": coord_stats["min_pairwise_distance"],
            "mean_row": coord_stats["mean_row"],
            "mean_col": coord_stats["mean_col"],
        }

    # ------------------------------------------------------------------
    # Module representation
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Return a concise parameter summary for ``print(module)``.

        Returns
        -------
        str

        Examples
        --------
        >>> lscs = LearnableShiftCycleSpinning(
        ...     num_shifts=9, channels=1, wavelet_channels=4,
        ...     structure_channels=12, coordinate_embed_dim=16,
        ...     num_heads=1, num_layers=2, temperature=1.0,
        ... )
        >>> print(lscs)  # doctest: +ELLIPSIS
        LearnableShiftCycleSpinning(
          ...
        )
        """
        return (
            f"num_shifts={self.num_shifts}, "
            f"channels={self.channels}, "
            f"wavelet_channels={self.wavelet_channels}, "
            f"structure_channels={self.structure_channels}, "
            f"coordinate_embed_dim={self.coordinate_embed_dim}, "
            f"token_dim={self.token_dim}, "
            f"num_heads={self.num_heads}, "
            f"num_layers={self.num_layers}, "
            f"max_shift_radius={self.max_shift_radius}, "
            f"radius_lambda={self.radius_lambda}, "
            f"repulsion_lambda={self.repulsion_lambda}, "
            f"temperature={self.temperature}, "
            f"pooling={self.pooling!r}"
        )
