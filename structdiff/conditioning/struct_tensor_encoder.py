"""
structdiff/conditioning/struct_tensor_encoder.py
=================================================
A3: Structure Tensor Encoder — conditioning module.

Maps a batch of structure tensor maps [B, 3, H, W] to dense embedding
vectors [B, time_embed_dim] that are added to the combined timestep +
look embedding inside ``ConditionedUNetModel.forward``:

    emb = time_emb + look_emb(look_num) + struct_encoder(struct_tensor)

The combined vector is distributed to every ``ResBlock`` via its
``emb_layers``, providing FiLM-style (scale + shift) conditioning.

Architecture
------------
Three convolutional stages with GroupNorm + SiLU, progressively
downsampling the spatial map, followed by global average pooling and
a linear projection:

    Input:  [B, 3,    H,    W]
    Conv1:  [B, 64,   H,    W]   (stride 1, padding 1)
    Conv2:  [B, 128,  H/2,  W/2] (stride 2, padding 1)
    Conv3:  [B, D,    H/4,  W/4] (stride 2, padding 1)  D = time_embed_dim
    GAP:    [B, D]
    Proj:   [B, D]                (linear, initialised near-zero)

GroupNorm uses 8 groups throughout (compatible with odd channel counts).

Initialisation
--------------
The final linear projection (``proj``) is initialised to near-zero
(std=0.02), identical to ``LookEmbedding``.  This ensures that at the
start of A3 fine-tuning from an A2 checkpoint, the structure tensor
signal contributes approximately zero to ``emb``, preserving A2
behaviour exactly.  All conv layers use PyTorch default (Kaiming uniform).

Checkpoint compatibility
------------------------
All parameters live under the ``struct_encoder.*`` key prefix.
A2 checkpoints contain no such keys, so ``load_state_dict(strict=False)``
loads A2 weights cleanly, leaving ``struct_encoder`` at its initialised
(near-zero output) state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Number of GroupNorm groups.  8 divides 64 and 128 cleanly.
_GN_GROUPS: int = 8

# Standard deviation for near-zero initialisation of the projection layer.
_PROJ_INIT_STD: float = 0.02


class StructTensorEncoder(nn.Module):
    """Lightweight CNN that encodes a structure tensor map into an embedding.

    Parameters
    ----------
    time_embed_dim:
        Output dimensionality — must match ``UNetModel``'s ``time_embed``
        output (``model_channels * 4``).  For the default SAR-DDPM
        configuration (``--num_channels 192``) this is ``768``.

    Attributes
    ----------
    conv1, conv2, conv3 : nn.Conv2d
        Three convolutional stages.
    norm1, norm2, norm3 : nn.GroupNorm
        Per-stage group normalisation.
    proj : nn.Linear
        Final projection from global-average-pooled features to
        ``time_embed_dim``.  Initialised near-zero.

    Examples
    --------
    >>> enc = StructTensorEncoder(time_embed_dim=768)
    >>> J = torch.randn(4, 3, 256, 256)   # [B, 3, H, W]
    >>> v = enc(J)                         # [B, 768]
    >>> v.shape
    torch.Size([4, 768])
    """

    def __init__(self, time_embed_dim: int) -> None:
        super().__init__()

        if time_embed_dim <= 0:
            raise ValueError(
                f"time_embed_dim must be a positive integer, got {time_embed_dim}."
            )

        D = time_embed_dim

        # ------------------------------------------------------------------
        # Stage 1: 3 → 64 channels, full spatial resolution
        # ------------------------------------------------------------------
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=_GN_GROUPS, num_channels=64)

        # ------------------------------------------------------------------
        # Stage 2: 64 → 128 channels, spatial /2
        # ------------------------------------------------------------------
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=_GN_GROUPS, num_channels=128)

        # ------------------------------------------------------------------
        # Stage 3: 128 → time_embed_dim channels, spatial /4
        # ------------------------------------------------------------------
        self.conv3 = nn.Conv2d(128, D, kernel_size=3, stride=2, padding=1, bias=False)
        # GroupNorm: use min(8, D) groups so D need not be divisible by 8
        gn3_groups = min(_GN_GROUPS, D)
        # D must be divisible by gn3_groups; find the largest valid divisor ≤ 8
        while D % gn3_groups != 0 and gn3_groups > 1:
            gn3_groups -= 1
        self.norm3 = nn.GroupNorm(num_groups=gn3_groups, num_channels=D)

        # ------------------------------------------------------------------
        # Final projection: D → D, initialised near-zero for warm-start
        # ------------------------------------------------------------------
        self.proj = nn.Linear(D, D)
        nn.init.normal_(self.proj.weight, mean=0.0, std=_PROJ_INIT_STD)
        nn.init.zeros_(self.proj.bias)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def time_embed_dim(self) -> int:
        """Output embedding dimensionality."""
        return self.proj.out_features

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, struct_tensor: torch.Tensor) -> torch.Tensor:
        """Encode a batch of structure tensor maps into embedding vectors.

        Parameters
        ----------
        struct_tensor:
            Float tensor of shape ``[B, 3, H, W]``.
            Channel 0: J11, Channel 1: J12, Channel 2: J22.
            Expected range [-1, 1] (as produced by
            ``compute_structure_tensor(..., normalise=True)``).

        Returns
        -------
        torch.Tensor
            Float tensor of shape ``[B, time_embed_dim]``.
            Add directly to the timestep embedding:

                emb = emb + struct_encoder(struct_tensor)

        Raises
        ------
        ValueError
            If ``struct_tensor`` is not 4-D or does not have 3 channels.
        """
        if struct_tensor.ndim != 4:
            raise ValueError(
                f"struct_tensor must be 4-D [B, 3, H, W], "
                f"got shape {tuple(struct_tensor.shape)}."
            )
        if struct_tensor.shape[1] != 3:
            raise ValueError(
                f"struct_tensor must have exactly 3 channels (J11, J12, J22), "
                f"got {struct_tensor.shape[1]}."
            )

        # Stage 1
        x = F.silu(self.norm1(self.conv1(struct_tensor)))   # [B, 64,  H,   W]

        # Stage 2
        x = F.silu(self.norm2(self.conv2(x)))               # [B, 128, H/2, W/2]

        # Stage 3
        x = F.silu(self.norm3(self.conv3(x)))               # [B, D,   H/4, W/4]

        # Global average pool → [B, D]
        x = x.mean(dim=(2, 3))

        # Linear projection (near-zero at init → warm-start from A2)
        return self.proj(x)                                  # [B, D]
