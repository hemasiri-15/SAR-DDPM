"""
Reusable Vision-Transformer-style block operating on [B, C, H, W] feature maps.

This module implements a single Transformer encoder block (pre-norm,
multi-head self-attention + feed-forward network, both with residual
connections) that preserves the spatial feature-map interface expected by
convolutional diffusion UNets (e.g. OpenAI's guided-diffusion / SAR-DDPM).

Phase 1 scope
-------------
This file is intentionally self-contained. It does NOT modify, import from,
or get imported by ``guided_diffusion/unet.py``. It is a standalone building
block intended to be inserted into the UNet bottleneck in a later phase.

Phase 1.1 (this revision)
--------------------------
Adds forward-compatibility scaffolding for later research branches
(confidence-guided attention, wavelet-guided attention, structure-tensor
attention) without changing current numerical behaviour:

* LayerScale (trainable per-channel gates ``gamma1``/``gamma2``,
  initialized to 1e-5) on both residual branches.
* Optional DropPath / stochastic depth (default off).
* Optional keyword-only extension hooks (``attention_bias``,
  ``confidence_map``, ``wavelet_features``, ``structure_features``) that are
  accepted and documented but currently ignored -- they exist so later
  phases can add these signals without breaking this module's call sites.
* Optional ``return_attention`` flag to retrieve attention weights.

None of these additions change the block's behaviour at default settings:
``drop_path=0.0`` and the extension hooks all default to ``None``/``False``,
and because both residual branches are still zero-initialized (see
``_init_weights``), the block remains an exact identity function at
initialization, exactly as before.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic depth: randomly drops entire residual branches per sample.

    During training, each sample in the batch has probability ``drop_prob``
    of having its input zeroed out (and the surviving samples are rescaled
    by ``1 / (1 - drop_prob)`` to preserve the expected value). During
    evaluation, or when ``drop_prob == 0``, this module is the identity
    function.

    Parameters
    ----------
    drop_prob : float, optional
        Probability of dropping the branch for a given sample. Must be in
        ``[0, 1)``. Default is ``0.0`` (no-op, identity).
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not (0.0 <= drop_prob < 1.0):
            raise ValueError(
                f"`drop_prob` must be in the range [0, 1), got {drop_prob!r}."
            )
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # One random value per sample, broadcast over all remaining dims.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()  # binarize: 0 (drop) or 1 (keep)
        return x.div(keep_prob) * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


class TransformerBlock(nn.Module):
    """A pre-norm Transformer encoder block for [B, C, H, W] feature maps.

    The block flattens the spatial dimensions of the input feature map into
    a sequence, applies a standard pre-norm Transformer encoder layer
    (LayerNorm -> Multi-Head Self-Attention -> LayerScale -> DropPath ->
    residual, then LayerNorm -> Feed-Forward Network -> LayerScale ->
    DropPath -> residual), and reshapes the result back to the original
    ``[B, C, H, W]`` layout. This makes the block a drop-in, shape-preserving
    component that can be inserted anywhere a convolutional feature map is
    available (e.g. a UNet bottleneck).

    Pipeline
    --------
    ``[B, C, H, W]``
        -> flatten spatial dims -> ``[B, H*W, C]``
        -> LayerNorm -> Multi-Head Self-Attention -> LayerScale (gamma1)
           -> DropPath -> residual add
        -> LayerNorm -> Feed-Forward Network -> LayerScale (gamma2)
           -> DropPath -> residual add
        -> reshape back -> ``[B, C, H, W]``

    Parameters
    ----------
    channels : int
        Number of input/output channels ``C``. Must be positive and must be
        divisible by ``num_heads`` (a requirement of multi-head attention,
        which splits ``channels`` evenly across heads).
    num_heads : int
        Number of attention heads used by the multi-head self-attention
        layer. Must be positive.
    mlp_ratio : float, optional
        Expansion ratio for the hidden dimension of the feed-forward
        network, i.e. the FFN hidden size is ``int(channels * mlp_ratio)``.
        Default is ``4.0``, following standard ViT/DiT/Transformer
        conventions. Note: prior to this revision the hidden size used
        ``round()``; it now uses ``int()`` (truncation), per updated spec.
        For ratios/channel counts where ``channels * mlp_ratio`` is already
        an integer (the common case, e.g. integer channels with ratio 4.0),
        this produces an identical hidden size to before.
    dropout : float, optional
        Dropout probability applied inside the attention layer and inside
        the feed-forward network (after each linear projection where
        specified). Default is ``0.0``.
    drop_path : float, optional
        Stochastic-depth probability applied independently to each of the
        two residual branches (attention and FFN). Must be in ``[0, 1)``.
        Default is ``0.0``, which makes ``DropPath`` an identity function
        and reproduces the exact previous behaviour.

    Attributes
    ----------
    norm1 : nn.LayerNorm
        Pre-attention normalization, applied over the channel dimension.
    attn : nn.MultiheadAttention
        Multi-head self-attention module, configured with
        ``batch_first=True`` so it directly consumes ``[B, N, C]`` tensors.
    norm2 : nn.LayerNorm
        Pre-FFN normalization, applied over the channel dimension.
    mlp : nn.Sequential
        Feed-forward network: Linear -> GELU -> Dropout -> Linear -> Dropout.
    gamma1 : nn.Parameter
        LayerScale gate for the attention residual branch, shape
        ``[channels]``, initialized to ``1e-5``.
    gamma2 : nn.Parameter
        LayerScale gate for the FFN residual branch, shape ``[channels]``,
        initialized to ``1e-5``.
    drop_path1, drop_path2 : DropPath
        Stochastic-depth modules for the attention and FFN branches,
        respectively.

    Notes
    -----
    Residual connections
        Both sub-layers (attention and FFN) are wrapped in residual
        connections around their *normalized, gated* input (pre-norm
        formulation): ``x = x + DropPath(gamma1 * Attention(LayerNorm(x)))``
        and ``x = x + DropPath(gamma2 * FFN(LayerNorm(x)))``.
        Pre-norm is used (rather than post-norm) because it produces more
        stable gradients in deep stacks and is the formulation used by
        modern diffusion Transformers (e.g. DiT), which matters here since
        this block may later be stacked or embedded inside a deep UNet.

    LayerScale
        ``gamma1``/``gamma2`` are trainable per-channel scalars initialized
        to a small value (``1e-5``), following the LayerScale technique
        (Touvron et al., "Going deeper with Image Transformers"). This lets
        each residual branch start with a near-zero contribution and grow
        its influence gradually during training, which further stabilizes
        training once the block is stacked or embedded in a deeper network.
        Combined with the zero-initialized output projections described
        below, the branch contribution is *exactly* zero at initialization
        regardless of the ``gamma`` value, so adding LayerScale does not
        change the block's behaviour at initialization.

    DropPath
        Stochastic depth is a regularizer that randomly skips a residual
        branch per-sample during training. At ``drop_path=0.0`` (the
        default), ``DropPath`` is the identity function and training/eval
        behaviour is unchanged from before this revision.

    Extension hooks (forward-compatibility only)
        ``attention_bias``, ``confidence_map``, ``wavelet_features``, and
        ``structure_features`` are accepted by ``forward`` but are currently
        **ignored** -- they exist purely so later research branches
        (confidence-guided attention, wavelet-guided attention,
        structure-tensor attention, and additive attention-score biasing)
        can be wired in without changing this module's public call
        signature again. See ``forward`` for per-argument documentation of
        their *intended* future use.

    Mixed precision
        The module performs no manual dtype or device casting anywhere in
        its ``forward`` method. All tensors keep whatever dtype/device they
        arrive in, so the block is safe to use under
        ``torch.cuda.amp.autocast()``: autocast will handle casting of the
        internal ``nn.Linear``/``nn.LayerNorm``/``nn.MultiheadAttention``
        operations automatically.

    Examples
    --------
    >>> block = TransformerBlock(channels=192, num_heads=8)
    >>> x = torch.randn(2, 192, 32, 32)
    >>> y = block(x)
    >>> y.shape
    torch.Size([2, 192, 32, 32])
    >>> y, attn_weights = block(x, return_attention=True)
    >>> attn_weights.shape
    torch.Size([2, 32, 32])
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        if not isinstance(channels, int) or channels <= 0:
            raise ValueError(
                f"`channels` must be a positive integer, got {channels!r}."
            )
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                f"`num_heads` must be a positive integer, got {num_heads!r}."
            )
        if channels % num_heads != 0:
            raise ValueError(
                "`channels` must be divisible by `num_heads` so that "
                f"multi-head attention can split channels evenly across "
                f"heads, got channels={channels} and num_heads={num_heads} "
                f"(channels % num_heads = {channels % num_heads})."
            )
        if mlp_ratio <= 0:
            raise ValueError(
                f"`mlp_ratio` must be positive, got {mlp_ratio!r}."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"`dropout` must be in the range [0, 1), got {dropout!r}."
            )
        if not (0.0 <= drop_path < 1.0):
            raise ValueError(
                f"`drop_path` must be in the range [0, 1), got {drop_path!r}."
            )

        self.channels = channels
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.drop_path_prob = drop_path

        hidden_dim = int(channels * mlp_ratio)

        # --- Attention sub-layer ------------------------------------------------
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # --- Feed-forward sub-layer ----------------------------------------------
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels),
            nn.Dropout(dropout),
        )

        # --- LayerScale gates (Touvron et al., CaiT) -----------------------------
        self.gamma1 = nn.Parameter(1e-5 * torch.ones(channels))
        self.gamma2 = nn.Parameter(1e-5 * torch.ones(channels))

        # --- Stochastic depth -----------------------------------------------------
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize parameters using a stable scheme for diffusion models.

        LayerNorms are initialized to the identity transform (weight=1,
        bias=0). Linear layers use Xavier/Glorot-uniform initialization for
        weights and zero for biases, which keeps activation variance stable
        at initialization -- important for diffusion models, where the
        denoiser is applied recursively across many timesteps and poorly
        scaled activations can compound into training instability.

        The output projection of the attention layer (``attn.out_proj``) and
        the final projection of the feed-forward network (the second
        ``nn.Linear`` in ``mlp``) are additionally zero-initialized. This
        makes the *entire block* an identity function at the start of
        training (both residual branches contribute zero before the
        residual add), which is the standard "zero-init residual branch"
        trick used throughout diffusion-model architectures (e.g. DiT, ADM)
        to stabilize early training. This holds regardless of the
        ``gamma1``/``gamma2`` LayerScale values, since ``gamma * 0 == 0``.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Zero-init the final linear layer of the MLP so its branch starts
        # as an identity mapping with respect to its residual connection.
        final_linear = self.mlp[-2]
        assert isinstance(final_linear, nn.Linear)
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

        # Zero-init the attention output projection so the attention branch
        # also starts as an identity mapping with respect to its residual
        # connection (nn.MultiheadAttention stores this as `out_proj`).
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        confidence_map: Optional[torch.Tensor] = None,
        wavelet_features: Optional[torch.Tensor] = None,
        structure_features: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Apply the Transformer block to a spatial feature map.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape ``[B, C, H, W]``, where ``B`` is the
            batch size, ``C`` must equal ``self.channels``, and ``H``/``W``
            are the spatial height/width.
        attention_bias : torch.Tensor, optional
            Reserved extension hook. **Currently ignored.** In a future
            phase this will be an additive bias applied to the raw
            attention scores immediately before the softmax (e.g. shape
            ``[B, num_heads, H*W, H*W]`` or a broadcastable variant), for
            things like relative-position or physically-informed biasing.
            Default is ``None``.
        confidence_map : torch.Tensor, optional
            Reserved extension hook. **Currently ignored.** Intended for a
            future confidence-guided attention branch, where a per-pixel
            confidence map (e.g. despeckling confidence) would modulate
            attention weights or the value projection. Default is ``None``.
        wavelet_features : torch.Tensor, optional
            Reserved extension hook. **Currently ignored.** Intended for a
            future wavelet-guided attention branch, where multi-scale
            wavelet sub-band features would be fused into the attention
            computation. Default is ``None``.
        structure_features : torch.Tensor, optional
            Reserved extension hook. **Currently ignored.** Intended for a
            future structure-tensor-guided attention branch, where local
            structure-tensor features (orientation/coherence) would bias or
            gate attention. Default is ``None``.
        return_attention : bool, optional
            If ``True``, also return the attention weights produced by the
            self-attention layer, averaged across heads (the default
            reduction of ``nn.MultiheadAttention``), with shape
            ``[B, H*W, H*W]``. Default is ``False``.

        Returns
        -------
        torch.Tensor or tuple of (torch.Tensor, torch.Tensor or None)
            If ``return_attention`` is ``False`` (default): the output
            feature map, shape ``[B, C, H, W]``, identical to the input
            shape.
            If ``return_attention`` is ``True``: a tuple
            ``(output, attention_weights)`` where ``output`` is as above and
            ``attention_weights`` has shape ``[B, H*W, H*W]``.

        Raises
        ------
        ValueError
            If ``x`` is not a 4-D tensor or its channel dimension does not
            match ``self.channels``.
        """
        # NOTE: attention_bias, confidence_map, wavelet_features, and
        # structure_features are accepted for forward-compatibility with
        # later research branches but are intentionally not used yet.
        del attention_bias, confidence_map, wavelet_features, structure_features

        if x.dim() != 4:
            raise ValueError(
                f"Expected a 4-D input tensor [B, C, H, W], got tensor "
                f"with {x.dim()} dimensions and shape {tuple(x.shape)}."
            )

        batch_size, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(
                f"Input channel dimension ({channels}) does not match "
                f"the channel dimension this block was constructed with "
                f"({self.channels})."
            )

        # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        seq = x.reshape(batch_size, channels, height * width).transpose(1, 2)

        # --- Self-attention sub-layer (pre-norm + LayerScale + DropPath + residual) ---
        normed = self.norm1(seq)
        attn_out, attn_weights = self.attn(
            normed, normed, normed, need_weights=return_attention, average_attn_weights=False
        )
        seq = seq + self.drop_path1(self.gamma1 * attn_out)

        # --- Feed-forward sub-layer (pre-norm + LayerScale + DropPath + residual) ---
        normed = self.norm2(seq)
        mlp_out = self.mlp(normed)
        seq = seq + self.drop_path2(self.gamma2 * mlp_out)

        # [B, H*W, C] -> [B, C, H*W] -> [B, C, H, W]
        out = seq.transpose(1, 2).reshape(batch_size, channels, height, width)

        if return_attention:
            return out, attn_weights
        return out


if __name__ == "__main__":
    x = torch.randn(2, 192, 32, 32)
    block = TransformerBlock(channels=192, num_heads=8)
    output = block(x)

    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(output.shape)}")

    assert x.shape == output.shape, (
        f"Shape mismatch: input {tuple(x.shape)} vs output "
        f"{tuple(output.shape)}"
    )

    print("TransformerBlock smoke test passed.")
