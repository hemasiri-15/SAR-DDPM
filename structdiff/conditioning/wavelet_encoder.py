"""
structdiff/conditioning/wavelet_encoder.py
==========================================
A12: WaveletEncoder — conditioning encoder for DWT subband features.

Purpose
-------
Maps a wavelet tensor produced by ``WaveletDataset``
(structdiff/data/wavelet_dataset.py) into a conditioning embedding
vector that lives in the same vector space as the time embedding, look
embedding, structure tensor embedding, multi-scale structure tensor
embedding, and spectral tensor embedding.

The full embedding sum consumed by the U-Net is::

    emb = time_emb + look_emb + struct_emb + ms_struct_emb
          + spectral_emb + wavelet_emb

``wavelet_emb`` is produced by this module.

Architecture
------------
A lightweight three-stage convolutional encoder followed by global
average pooling and a single linear projection.  The design mirrors
``StructTensorEncoder`` (A3) and ``TensorSpectralEncoder`` (A11) exactly:

    Conv2d(4  → 64,  3×3, pad=1)  + GroupNorm(8,  64)  + SiLU
    Conv2d(64 → 128, 3×3, s=2)    + GroupNorm(8, 128)  + SiLU
    Conv2d(128→ D,   3×3, s=2)    + GroupNorm(g,   D)  + SiLU
    Global Average Pool  →  [B, D]
    Linear(D → D)        →  [B, D]

where D = ``time_embed_dim`` and g is chosen dynamically as the largest
divisor of D that is ≤ 8 (so GroupNorm is valid for any D > 0).

Input tensor shape:  [B, 4, H/2, W/2]
Output tensor shape: [B, time_embed_dim]

Channel meanings (input):
    ch 0 — LL: approximation (low-pass × low-pass), speckle-suppressed.
    ch 1 — LH: horizontal detail (low-pass rows × high-pass cols).
    ch 2 — HL: vertical detail  (high-pass rows × low-pass cols).
    ch 3 — HH: diagonal detail  (high-pass × high-pass), speckle-rich.

Initialization
--------------
Convolutional layers use PyTorch's default initialisation (Kaiming
uniform for Conv2d weights).

The projection layer (``proj``) uses near-zero initialisation::

    nn.init.normal_(proj.weight, mean=0.0, std=0.02)
    nn.init.zeros_(proj.bias)

This ensures that at iteration 0 of A12 fine-tuning the wavelet branch
contributes approximately zero to ``emb``, preserving the behaviour of
the loaded A11 checkpoint.  The network then gradually learns to
exploit wavelet information over the course of A12 training.

Checkpoint compatibility
------------------------
A11 checkpoints do not contain any ``wavelet_encoder.*`` parameter keys.
Loading an A11 checkpoint into an A12 model therefore requires::

    model.load_state_dict(ckpt, strict=False)

With ``strict=False`` the missing ``wavelet_encoder.*`` keys are silently
ignored and the freshly initialised weights are kept.  Because the
projection is near-zero, the model starts from A11 behaviour and
gradually adapts to wavelet conditioning during A12 fine-tuning.

Mathematical role
-----------------
Let W ∈ ℝ^{B × 4 × H/2 × W/2} be the wavelet tensor.
``WaveletEncoder`` computes a deterministic function f_θ such that::

    wavelet_emb = f_θ(W) ∈ ℝ^{B × D},   D = time_embed_dim

``wavelet_emb`` is then added element-wise to the accumulated embedding
``emb`` before it is injected into every ResBlock of the U-Net via the
standard ``emb_out = self.emb_layers(emb)`` pathway (unchanged from A0).

Examples
--------
>>> import torch
>>> from structdiff.conditioning.wavelet_encoder import WaveletEncoder
>>> enc = WaveletEncoder(time_embed_dim=768)
>>> W = torch.randn(4, 4, 128, 128)   # [B=4, 4 subbands, H/2=128, W/2=128]
>>> v = enc(W)
>>> v.shape
torch.Size([4, 768])
>>> v.dtype
torch.float32

>>> # Minimal sanity: output is finite and not all-zero (post near-zero init)
>>> import torch
>>> enc = WaveletEncoder(512)
>>> W = torch.randn(2, 4, 64, 64)
>>> out = enc(W)
>>> bool(torch.isfinite(out).all())
True
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Module-level constants — mirrors StructTensorEncoder and TensorSpectralEncoder.
# ---------------------------------------------------------------------------

#: Number of GroupNorm groups used in stages 1 and 2.  Stage 3 uses a
#: dynamically computed group count to handle arbitrary time_embed_dim
#: values (see ``_compute_gn_groups``).
_GN_GROUPS: int = 8

#: Standard deviation for near-zero projection initialisation.
#: Ensures the wavelet branch contributes ~0 at A12 fine-tuning start,
#: preserving A11 checkpoint behaviour.  Mirrors A3 and A11.
_PROJ_INIT_STD: float = 0.02

#: Number of input channels (one per DWT subband: LL, LH, HL, HH).
_IN_CHANNELS: int = 4

#: Intermediate channel widths for the two strided convolution stages.
_HIDDEN_C1: int = 64
_HIDDEN_C2: int = 128


# ---------------------------------------------------------------------------
# Helper: dynamic GroupNorm group count
# ---------------------------------------------------------------------------


def _compute_gn_groups(num_channels: int, max_groups: int = _GN_GROUPS) -> int:
    """Return the largest valid GroupNorm group count ≤ ``max_groups``.

    GroupNorm requires ``num_channels % num_groups == 0``.  For stage 3
    the output channel count equals ``time_embed_dim``, which may not be
    divisible by 8.  This helper tries ``max_groups``, ``max_groups - 1``,
    … until it finds a valid divisor, guaranteeing correctness for any
    ``time_embed_dim > 0``.

    Parameters
    ----------
    num_channels:
        The channel dimension that GroupNorm will normalise.
    max_groups:
        Upper bound for the group count (default ``_GN_GROUPS`` = 8).

    Returns
    -------
    int
        Largest ``g`` in ``[1, max_groups]`` such that
        ``num_channels % g == 0``.

    Raises
    ------
    ValueError
        If no valid group count exists in ``[1, max_groups]``.  This
        cannot happen for ``max_groups >= 1`` because 1 always divides
        any positive integer.

    Examples
    --------
    >>> _compute_gn_groups(768)
    8
    >>> _compute_gn_groups(320)
    8
    >>> _compute_gn_groups(48)
    8
    >>> _compute_gn_groups(7)
    1
    """
    g = min(max_groups, num_channels)
    while g > 0:
        if num_channels % g == 0:
            return g
        g -= 1
    # Unreachable: g=1 always satisfies the condition.
    raise ValueError(
        f"Could not find a valid GroupNorm group count for "
        f"num_channels={num_channels} with max_groups={max_groups}."
    )


# ---------------------------------------------------------------------------
# WaveletEncoder
# ---------------------------------------------------------------------------


class WaveletEncoder(nn.Module):
    """Lightweight convolutional encoder for DWT subband conditioning.

    Maps a wavelet tensor of shape ``[B, 4, H/2, W/2]`` (produced by
    ``WaveletDataset``) into a conditioning embedding of shape
    ``[B, time_embed_dim]`` that is added to the U-Net's accumulated
    ``emb`` vector alongside the time, look, structure tensor, multi-scale
    structure tensor, and spectral tensor embeddings.

    Architecture summary::

        Conv2d(4  → 64,  3×3, pad=1) + GroupNorm(8,  64) + SiLU
        Conv2d(64 → 128, 3×3, s=2)   + GroupNorm(8, 128) + SiLU
        Conv2d(128→ D,   3×3, s=2)   + GroupNorm(g,   D) + SiLU
        Global Average Pool  →  [B, D]
        Linear(D → D, near-zero init) →  [B, D]

    where ``D = time_embed_dim`` and ``g`` is the largest divisor of ``D``
    that is ≤ 8, computed dynamically by ``_compute_gn_groups``.

    Parameters
    ----------
    time_embed_dim:
        Dimensionality of the output embedding vector.  Must match the
        U-Net's ``model_channels * 4``.  Must be > 0.

    Attributes
    ----------
    conv1 : nn.Conv2d
        Stage 1 convolution: 4 → 64 channels, 3×3, padding=1, no stride.
    norm1 : nn.GroupNorm
        Stage 1 GroupNorm: 8 groups, 64 channels.
    conv2 : nn.Conv2d
        Stage 2 convolution: 64 → 128 channels, 3×3, stride=2.
    norm2 : nn.GroupNorm
        Stage 2 GroupNorm: 8 groups, 128 channels.
    conv3 : nn.Conv2d
        Stage 3 convolution: 128 → time_embed_dim channels, 3×3, stride=2.
    norm3 : nn.GroupNorm
        Stage 3 GroupNorm: g groups (dynamic), time_embed_dim channels.
    proj : nn.Linear
        Linear projection: time_embed_dim → time_embed_dim, near-zero init.

    Examples
    --------
    >>> import torch
    >>> from structdiff.conditioning.wavelet_encoder import WaveletEncoder
    >>> enc = WaveletEncoder(time_embed_dim=768)
    >>> W = torch.randn(4, 4, 128, 128)
    >>> v = enc(W)
    >>> v.shape
    torch.Size([4, 768])
    >>> v.dtype
    torch.float32

    >>> # Works for non-power-of-2 embed dims
    >>> enc2 = WaveletEncoder(320)
    >>> v2 = enc2(torch.randn(2, 4, 64, 64))
    >>> v2.shape
    torch.Size([2, 320])

    >>> # Near-zero init: projection output is small at init
    >>> import torch; from structdiff.conditioning.wavelet_encoder import WaveletEncoder
    >>> enc = WaveletEncoder(512)
    >>> W = torch.randn(1, 4, 128, 128)
    >>> out = enc(W)
    >>> bool(out.abs().max().item() < 5.0)   # not exactly zero but small
    True
    """

    def __init__(self, time_embed_dim: int) -> None:
        super().__init__()

        # ----------------------------------------------------------------
        # Input validation
        # ----------------------------------------------------------------
        if time_embed_dim <= 0:
            raise ValueError(
                f"time_embed_dim must be a positive integer, "
                f"got {time_embed_dim}."
            )

        # ----------------------------------------------------------------
        # Stage 1: 4 → 64, no spatial downsampling.
        # Input  : [B, 4,  H/2, W/2]
        # Output : [B, 64, H/2, W/2]
        # ----------------------------------------------------------------
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=_IN_CHANNELS,
            out_channels=_HIDDEN_C1,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1: nn.GroupNorm = nn.GroupNorm(
            num_groups=_GN_GROUPS,
            num_channels=_HIDDEN_C1,
        )

        # ----------------------------------------------------------------
        # Stage 2: 64 → 128, stride=2 (spatial downsampling ×2).
        # Input  : [B, 64,  H/2, W/2]
        # Output : [B, 128, H/4, W/4]
        # ----------------------------------------------------------------
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=_HIDDEN_C1,
            out_channels=_HIDDEN_C2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.norm2: nn.GroupNorm = nn.GroupNorm(
            num_groups=_GN_GROUPS,
            num_channels=_HIDDEN_C2,
        )

        # ----------------------------------------------------------------
        # Stage 3: 128 → time_embed_dim, stride=2.
        # GroupNorm group count is computed dynamically because
        # time_embed_dim may not be divisible by 8.
        # Input  : [B, 128,           H/4, W/4]
        # Output : [B, time_embed_dim, H/8, W/8]
        # ----------------------------------------------------------------
        gn_groups_3: int = _compute_gn_groups(time_embed_dim)

        self.conv3: nn.Conv2d = nn.Conv2d(
            in_channels=_HIDDEN_C2,
            out_channels=time_embed_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.norm3: nn.GroupNorm = nn.GroupNorm(
            num_groups=gn_groups_3,
            num_channels=time_embed_dim,
        )

        # ----------------------------------------------------------------
        # Global average pool: [B, time_embed_dim, H/8, W/8] → [B, time_embed_dim]
        # Implemented in forward as x.mean(dim=(2, 3)).
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        # Projection: time_embed_dim → time_embed_dim.
        # Near-zero initialisation ensures wavelet_emb ≈ 0 at A12 start,
        # preserving A11 checkpoint behaviour (see module docstring).
        # ----------------------------------------------------------------
        self.proj: nn.Linear = nn.Linear(time_embed_dim, time_embed_dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=_PROJ_INIT_STD)
        nn.init.zeros_(self.proj.bias)

    # ------------------------------------------------------------------
    # Property
    # ------------------------------------------------------------------

    @property
    def time_embed_dim(self) -> int:
        """Output embedding dimensionality (equals ``proj.out_features``)."""
        return self.proj.out_features

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, wavelet_tensor: torch.Tensor) -> torch.Tensor:
        """Encode a batch of wavelet tensors into conditioning embeddings.

        Parameters
        ----------
        wavelet_tensor:
            DWT subband tensor produced by ``WaveletDataset``.

            Shape: ``[B, 4, H/2, W/2]``

            Channel layout::

                ch 0 — LL  approximation (low-pass × low-pass)
                ch 1 — LH  horizontal detail (low-pass rows × high-pass cols)
                ch 2 — HL  vertical detail  (high-pass rows × low-pass cols)
                ch 3 — HH  diagonal detail  (high-pass × high-pass)

            Expected range: ``[-1, 1]``, dtype ``float32``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, time_embed_dim]``, dtype ``float32``.
            Ready to be added to the accumulated U-Net embedding ``emb``.

        Raises
        ------
        ValueError
            If ``wavelet_tensor`` is not 4-dimensional, or if its channel
            count (``wavelet_tensor.shape[1]``) is not exactly 4.

        Examples
        --------
        >>> import torch
        >>> from structdiff.conditioning.wavelet_encoder import WaveletEncoder
        >>> enc = WaveletEncoder(768)
        >>> W = torch.randn(4, 4, 128, 128)
        >>> out = enc(W)
        >>> out.shape
        torch.Size([4, 768])
        >>> out.dtype
        torch.float32

        >>> # Single-sample batch (B=1)
        >>> out1 = enc(torch.randn(1, 4, 64, 64))
        >>> out1.shape
        torch.Size([1, 768])

        >>> # Wrong number of dimensions raises ValueError
        >>> try:
        ...     enc(torch.randn(4, 128, 128))
        ... except ValueError as e:
        ...     print("caught:", e)
        caught: wavelet_tensor must be 4-dimensional [B, 4, H/2, W/2], got shape torch.Size([4, 128, 128]) (ndim=3).

        >>> # Wrong channel count raises ValueError
        >>> try:
        ...     enc(torch.randn(2, 3, 64, 64))
        ... except ValueError as e:
        ...     print("caught:", e)
        caught: wavelet_tensor must have exactly 4 channels (LL, LH, HL, HH), got shape torch.Size([2, 3, 64, 64]) with 3 channels.
        """
        # ----------------------------------------------------------------
        # Runtime validation
        # ----------------------------------------------------------------
        if wavelet_tensor.ndim != 4:
            raise ValueError(
                f"wavelet_tensor must be 4-dimensional [B, 4, H/2, W/2], "
                f"got shape {wavelet_tensor.shape} (ndim={wavelet_tensor.ndim})."
            )
        if wavelet_tensor.shape[1] != _IN_CHANNELS:
            raise ValueError(
                f"wavelet_tensor must have exactly {_IN_CHANNELS} channels "
                f"(LL, LH, HL, HH), got shape {wavelet_tensor.shape} "
                f"with {wavelet_tensor.shape[1]} channels."
            )

        # ----------------------------------------------------------------
        # Stage 1: [B, 4, H/2, W/2] → [B, 64, H/2, W/2]
        # ----------------------------------------------------------------
        x: torch.Tensor = self.conv1(wavelet_tensor)
        x = self.norm1(x)
        x = torch.nn.functional.silu(x)

        # ----------------------------------------------------------------
        # Stage 2: [B, 64, H/2, W/2] → [B, 128, H/4, W/4]
        # ----------------------------------------------------------------
        x = self.conv2(x)
        x = self.norm2(x)
        x = torch.nn.functional.silu(x)

        # ----------------------------------------------------------------
        # Stage 3: [B, 128, H/4, W/4] → [B, time_embed_dim, H/8, W/8]
        # ----------------------------------------------------------------
        x = self.conv3(x)
        x = self.norm3(x)
        x = torch.nn.functional.silu(x)

        # ----------------------------------------------------------------
        # Global Average Pool: [B, time_embed_dim, H/8, W/8] → [B, time_embed_dim]
        # ----------------------------------------------------------------
        x = x.mean(dim=(2, 3))

        # ----------------------------------------------------------------
        # Linear projection: [B, time_embed_dim] → [B, time_embed_dim]
        # Near-zero init ensures wavelet_emb ≈ 0 at A12 fine-tuning start.
        # ----------------------------------------------------------------
        x = self.proj(x)

        return x
