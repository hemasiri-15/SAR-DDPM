"""
structdiff/conditioning/tensor_spectral_encoder.py
====================================================
A11: Tensor Spectral Encoder — conditioning module.

Maps a batch of A11 spectral-feature maps [B, 12, H, W] (eigenvalues +
anisotropy + coherence, concatenated across A10's three scales -- see
structdiff/utils/tensor_spectral_features.py) to dense embedding
vectors [B, time_embed_dim], added to the combined embedding inside
``ConditionedUNetModel.forward``:

    emb = time_emb + look_emb(look_num)
          + struct_encoder(struct_tensor)          # A3, if enabled
          + ms_struct_encoder(struct_tensors)       # A10, if enabled
          + spectral_encoder(spectral_tensor)       # A11, this module

Architecture
------------
Mirrors A3's ``StructTensorEncoder`` exactly, with in_channels=12
instead of 3 (per A11 spec: 4 features x 3 scales):

    Input:  [B, 12,   H,    W]
    Conv1:  [B, 64,   H,    W]   (stride 1, padding 1) + GroupNorm + SiLU
    Conv2:  [B, 128,  H/2,  W/2] (stride 2, padding 1) + GroupNorm + SiLU
    Conv3:  [B, D,    H/4,  W/4] (stride 2, padding 1) + GroupNorm + SiLU
    GAP:    [B, D]
    Proj:   [B, D]                (linear, initialised near-zero)

GroupNorm uses 8 groups for stages 1-2; stage 3 uses min(8, D) groups
adjusted down to the largest divisor of D, matching A3's
StructTensorEncoder convention exactly.

Initialisation
--------------
``proj`` is initialised near-zero (std=0.02, zero bias), identical to
``StructTensorEncoder`` (A3) and ``LookEmbedding`` (A2). At the start
of A11 fine-tuning from an A10 checkpoint, ``spectral_encoder``
contributes approximately zero to ``emb``, so A11 starts from A10
behaviour exactly.

Checkpoint compatibility
------------------------
All parameters live under the ``spectral_encoder.*`` key prefix. A10
checkpoints contain no such keys, so ``load_state_dict(strict=False)``
loads A10 weights cleanly, leaving ``spectral_encoder`` at its
initialised (near-zero output) state. No key remapping is needed (see
checkpoint_a10_to_a11.py) -- this is a brand-new module with no A10
analog to warm-start from.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


#: Input channels: 4 features (lambda1, lambda2, anisotropy, coherence)
#: x 3 scales (fine, medium, coarse), per A11 spec.
_IN_CHANNELS: int = 12

# Number of GroupNorm groups for stages 1-2. 8 divides 64 and 128 cleanly.
_GN_GROUPS: int = 8

# Standard deviation for near-zero initialisation of the projection layer.
_PROJ_INIT_STD: float = 0.02


class TensorSpectralEncoder(nn.Module):
    """Lightweight CNN that encodes A11 spectral-feature maps into an embedding.

    Parameters
    ----------
    time_embed_dim:
        Output dimensionality -- must match ``UNetModel``'s
        ``time_embed`` output (``model_channels * 4``), i.e. the same
        ``D`` used by ``LookEmbedding``, ``StructTensorEncoder`` (A3),
        and ``MultiScaleStructTensorEncoder`` (A10).

    Attributes
    ----------
    conv1, conv2, conv3 : nn.Conv2d
        Three convolutional stages.
    norm1, norm2, norm3 : nn.GroupNorm
        Per-stage group normalisation.
    proj : nn.Linear
        Final projection from global-average-pooled features to
        ``time_embed_dim``. Initialised near-zero.

    Examples
    --------
    >>> enc = TensorSpectralEncoder(time_embed_dim=768)
    >>> spectral = torch.randn(4, 12, 256, 256)  # [B, 12, H, W]
    >>> v = enc(spectral)                         # [B, 768]
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
        # Stage 1: 12 -> 64 channels, full spatial resolution
        # ------------------------------------------------------------------
        self.conv1 = nn.Conv2d(_IN_CHANNELS, 64, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=_GN_GROUPS, num_channels=64)

        # ------------------------------------------------------------------
        # Stage 2: 64 -> 128 channels, spatial /2
        # ------------------------------------------------------------------
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=_GN_GROUPS, num_channels=128)

        # ------------------------------------------------------------------
        # Stage 3: 128 -> time_embed_dim channels, spatial /4
        # ------------------------------------------------------------------
        self.conv3 = nn.Conv2d(128, D, kernel_size=3, stride=2, padding=1, bias=False)
        # GroupNorm: use min(8, D) groups so D need not be divisible by 8
        gn3_groups = min(_GN_GROUPS, D)
        while D % gn3_groups != 0 and gn3_groups > 1:
            gn3_groups -= 1
        self.norm3 = nn.GroupNorm(num_groups=gn3_groups, num_channels=D)

        # ------------------------------------------------------------------
        # Final projection: D -> D, initialised near-zero for warm-start
        # ------------------------------------------------------------------
        self.proj = nn.Linear(D, D)
        nn.init.normal_(self.proj.weight, mean=0.0, std=_PROJ_INIT_STD)
        nn.init.zeros_(self.proj.bias)

    @property
    def time_embed_dim(self) -> int:
        """Output embedding dimensionality."""
        return self.proj.out_features

    def forward(self, spectral_tensor: torch.Tensor) -> torch.Tensor:
        """Encode a batch of A11 spectral-feature maps into embedding vectors.

        Parameters
        ----------
        spectral_tensor:
            Float tensor of shape ``[B, 12, H, W]``. Channel layout:
            4 channels (lambda1, lambda2, anisotropy, coherence) x 3
            scales (fine, medium, coarse), as produced by
            ``compute_spectral_features_multiscale``.

        Returns
        -------
        torch.Tensor
            Float tensor of shape ``[B, time_embed_dim]``. Add
            directly to the combined embedding:

                emb = emb + spectral_encoder(spectral_tensor)

        Raises
        ------
        ValueError
            If ``spectral_tensor`` is not 4-D or does not have 12
            channels.
        """
        if spectral_tensor.ndim != 4:
            raise ValueError(
                f"spectral_tensor must be 4-D [B, 12, H, W], "
                f"got shape {tuple(spectral_tensor.shape)}."
            )
        if spectral_tensor.shape[1] != _IN_CHANNELS:
            raise ValueError(
                f"spectral_tensor must have exactly {_IN_CHANNELS} channels "
                f"(4 features x 3 scales), got {spectral_tensor.shape[1]}."
            )

        # Stage 1
        x = F.silu(self.norm1(self.conv1(spectral_tensor)))  # [B, 64,  H,   W]

        # Stage 2
        x = F.silu(self.norm2(self.conv2(x)))                # [B, 128, H/2, W/2]

        # Stage 3
        x = F.silu(self.norm3(self.conv3(x)))                # [B, D,   H/4, W/4]

        # Global average pool -> [B, D]
        x = x.mean(dim=(2, 3))

        # Linear projection (near-zero at init -> warm-start from A10)
        return self.proj(x)                                  # [B, D]
