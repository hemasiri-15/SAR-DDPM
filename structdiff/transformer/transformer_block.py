"""
Reusable Vision-Transformer-style blocks operating on [B, C, H, W] feature maps.

This module implements pre-norm Transformer encoder blocks (LayerNorm ->
Multi-Head Self-Attention -> LayerScale -> DropPath -> residual, then
LayerNorm -> Feed-Forward Network -> LayerScale -> DropPath -> residual)
that preserve the spatial feature-map interface expected by convolutional
diffusion UNets (e.g. OpenAI's guided-diffusion / SAR-DDPM).

Two concrete blocks are provided:

* ``TransformerBlock`` -- generic self-attention via ``nn.MultiheadAttention``.
  Numerically and structurally identical to the original standalone
  implementation, including its state_dict key layout.
* ``PhysicsTransformerBlock`` -- identical skeleton, but attention is
  performed by ``PhysicsAwareAttention``, which accepts an optional
  additive ``physics_attention_bias`` term.

Shared architecture
--------------------
Both blocks share every sub-layer *except* the attention implementation:
LayerNorm, MLP, LayerScale gates, DropPath, residual connections, spatial
reshaping, and the initialization scheme are defined exactly once in
``_BaseTransformerBlock`` and inherited (not composed) by both concrete
blocks. Because this uses ordinary subclassing rather than a
``self._base = ...`` composition, submodules created in
``_BaseTransformerBlock.__init__`` (e.g. ``self.norm1``) become direct
attributes of the subclass instance -- so ``TransformerBlock``'s
state_dict keys (``norm1.*``, ``attn.*``, ``norm2.*``, ``mlp.*``,
``gamma1``, ``gamma2``, ``drop_path1``, ``drop_path2``) are byte-for-byte
identical to the original standalone class, and existing checkpoints load
with no changes.

The only behavior that differs between the two blocks -- constructing the
attention module, running it, and zero-initializing its output
projection -- is isolated behind three small hooks
(``_build_attention``, ``_attend``, ``_zero_init_attention_output``) that
subclasses implement. This keeps the forward-pass skeleton
(``_forward_impl``) defined exactly once, so there is no duplicated
Transformer-block logic between the two classes.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .physics_aware_attention import PhysicsAwareAttention


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
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


class _BaseTransformerBlock(nn.Module):
    """Shared pre-norm Transformer-block skeleton for [B, C, H, W] inputs.

    This class is not meant to be instantiated directly. It owns every
    sub-layer that is common to all attention variants: LayerNorm, the
    feed-forward network, LayerScale gates, DropPath, the residual
    connections, and spatial reshaping, plus the shared parameter
    initialization scheme. It does **not** know how to construct or run
    attention -- that is delegated to subclasses via three hooks:

    ``_build_attention(channels, num_heads, dropout)``
        Construct and return the attention submodule; assigned to
        ``self.attn``. Called at the same point in ``__init__`` as the
        original ``TransformerBlock`` constructed its ``nn.MultiheadAttention``
        (immediately after ``norm1``), so that RNG consumption order --
        and therefore actual parameter values under a fixed seed -- is
        preserved for ``TransformerBlock``.
    ``_attend(normed, return_attention, **attn_kwargs)``
        Run ``self.attn`` on the normalized sequence and return
        ``(attn_output, attn_weights_or_None)``. Subclasses are
        responsible for normalizing their attention module's return value
        into this shape.
    ``_zero_init_attention_output()``
        Zero-initialize the attention module's output projection, so the
        attention residual branch starts as an identity mapping (combined
        with LayerScale and the zero-initialized final MLP projection,
        this makes the whole block an identity function at
        initialization, exactly as before).

    Parameters
    ----------
    channels : int
        Number of input/output channels ``C``. Must be positive and must
        be divisible by ``num_heads``.
    num_heads : int
        Number of attention heads. Must be positive.
    mlp_ratio : float, optional
        Expansion ratio for the FFN hidden dimension, i.e. hidden size is
        ``int(channels * mlp_ratio)``. Default is ``4.0``.
    dropout : float, optional
        Dropout probability applied inside attention and the FFN. Default
        is ``0.0``.
    drop_path : float, optional
        Stochastic-depth probability applied to each residual branch.
        Must be in ``[0, 1)``. Default is ``0.0``.

    Attributes
    ----------
    norm1, norm2 : nn.LayerNorm
        Pre-attention / pre-FFN normalization.
    attn : nn.Module
        Attention submodule, constructed by ``_build_attention``.
    mlp : nn.Sequential
        Linear -> GELU -> Dropout -> Linear -> Dropout.
    gamma1, gamma2 : nn.Parameter
        LayerScale gates, shape ``[channels]``, initialized to ``1e-5``.
    drop_path1, drop_path2 : DropPath
        Stochastic-depth modules for the attention and FFN branches.
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

        # --- Attention sub-layer -------------------------------------------
        self.norm1 = nn.LayerNorm(channels)
        self.attn = self._build_attention(channels, num_heads, dropout)

        # --- Feed-forward sub-layer -----------------------------------------
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels),
            nn.Dropout(dropout),
        )

        # --- LayerScale gates (Touvron et al., CaiT) -------------------------
        self.gamma1 = nn.Parameter(1e-5 * torch.ones(channels))
        self.gamma2 = nn.Parameter(1e-5 * torch.ones(channels))

        # --- Stochastic depth -------------------------------------------------
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

        self._init_weights()

    # ------------------------------------------------------------------ #
    # Hooks subclasses must implement
    # ------------------------------------------------------------------ #
    def _build_attention(
        self, channels: int, num_heads: int, dropout: float
    ) -> nn.Module:
        """Construct the attention submodule (assigned to ``self.attn``)."""
        raise NotImplementedError

    def _zero_init_attention_output(self) -> None:
        """Zero-initialize the attention module's output projection."""
        raise NotImplementedError

    def _attend(
        self,
        normed: torch.Tensor,
        return_attention: bool,
        **attn_kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run attention on ``normed`` and return ``(output, weights_or_None)``."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Shared logic
    # ------------------------------------------------------------------ #
    def _init_weights(self) -> None:
        """Initialize parameters using a stable scheme for diffusion models.

        LayerNorms are initialized to the identity transform (weight=1,
        bias=0). Linear layers use Xavier/Glorot-uniform initialization
        for weights and zero for biases. The final projection of the
        feed-forward network and the attention module's output projection
        are additionally zero-initialized, making the entire block an
        identity function at the start of training (the standard
        "zero-init residual branch" trick used throughout diffusion-model
        architectures, e.g. DiT, ADM), regardless of the ``gamma1``/
        ``gamma2`` LayerScale values, since ``gamma * 0 == 0``.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        final_linear = self.mlp[-2]
        assert isinstance(final_linear, nn.Linear)
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

        self._zero_init_attention_output()

    def _forward_impl(
        self,
        x: torch.Tensor,
        return_attention: bool,
        **attn_kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Shared forward skeleton: reshape, attend, FFN, reshape back.

        Subclasses' public ``forward`` methods validate/accept their own
        argument signature and then delegate to this method, passing
        whatever attention-specific keyword arguments (if any) their
        ``_attend`` implementation expects.
        """
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
        attn_out, attn_weights = self._attend(normed, return_attention, **attn_kwargs)
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


class TransformerBlock(_BaseTransformerBlock):
    """A pre-norm Transformer encoder block using standard self-attention.

    Behaviorally and structurally identical to the original standalone
    implementation, including state_dict key layout
    (``norm1``, ``attn``, ``norm2``, ``mlp``, ``gamma1``, ``gamma2``,
    ``drop_path1``, ``drop_path2``) and initialized-parameter values under
    a fixed random seed.

    Extension hooks (forward-compatibility only)
        ``attention_bias``, ``confidence_map``, ``wavelet_features``, and
        ``structure_features`` are accepted by ``forward`` but are
        currently **ignored** -- they exist purely so later research
        branches can be wired in without changing this module's public
        call signature again.

    Examples
    --------
    >>> block = TransformerBlock(channels=192, num_heads=8)
    >>> x = torch.randn(2, 192, 32, 32)
    >>> y = block(x)
    >>> y.shape
    torch.Size([2, 192, 32, 32])
    >>> y, attn_weights = block(x, return_attention=True)
    >>> attn_weights.shape
    torch.Size([2, 1024, 1024])
    """

    def _build_attention(
        self, channels: int, num_heads: int, dropout: float
    ) -> nn.Module:
        return nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def _zero_init_attention_output(self) -> None:
        # nn.MultiheadAttention stores its output projection as `out_proj`.
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def _attend(
        self,
        normed: torch.Tensor,
        return_attention: bool,
        **_unused,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.attn(
            normed,
            normed,
            normed,
            need_weights=return_attention,
            average_attn_weights=True,
        )

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
            Input feature map of shape ``[B, C, H, W]``.
        attention_bias, confidence_map, wavelet_features, structure_features : torch.Tensor, optional
            Reserved extension hooks. Currently ignored. See class
            docstring.
        return_attention : bool, optional
            If ``True``, also return attention weights (averaged across
            heads), shape ``[B, H*W, H*W]``. Default is ``False``.

        Returns
        -------
        torch.Tensor or tuple of (torch.Tensor, torch.Tensor or None)
            Output feature map, shape ``[B, C, H, W]``, optionally paired
            with attention weights.
        """
        del attention_bias, confidence_map, wavelet_features, structure_features
        return self._forward_impl(x, return_attention)


class PhysicsTransformerBlock(_BaseTransformerBlock):
    """A pre-norm Transformer encoder block using ``PhysicsAwareAttention``.

    Identical skeleton to ``TransformerBlock`` (same LayerNorm/MLP/
    LayerScale/DropPath/residual/reshaping logic, same initialization
    scheme), but self-attention is performed by ``PhysicsAwareAttention``,
    which accepts an optional additive ``physics_attention_bias`` term
    injected into the pre-softmax attention scores.

    Examples
    --------
    >>> block = PhysicsTransformerBlock(channels=192, num_heads=8)
    >>> x = torch.randn(2, 192, 32, 32)
    >>> y = block(x)
    >>> y.shape
    torch.Size([2, 192, 32, 32])
    >>> bias = torch.randn(2, 32 * 32, 32 * 32)
    >>> y = block(x, physics_attention_bias=bias)
    >>> y.shape
    torch.Size([2, 192, 32, 32])
    """

    def _build_attention(
        self, channels: int, num_heads: int, dropout: float
    ) -> nn.Module:
        return PhysicsAwareAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
        )

    def _zero_init_attention_output(self) -> None:
        # PhysicsAwareAttention wraps its own nn.MultiheadAttention at
        # `self.attn.attn`, so its output projection lives one level
        # deeper than in TransformerBlock.
        nn.init.zeros_(self.attn.attn.out_proj.weight)
        nn.init.zeros_(self.attn.attn.out_proj.bias)

    def _attend(
        self,
        normed: torch.Tensor,
        return_attention: bool,
        physics_attention_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # PhysicsAwareAttention returns a bare tensor when
        # return_attention=False (unlike nn.MultiheadAttention, which
        # always returns a tuple); normalize to the (output, weights)
        # shape _forward_impl expects.
        result = self.attn(
            normed,
            physics_attention_bias=physics_attention_bias,
            return_attention=return_attention,
        )
        if return_attention:
            return result
        return result, None

    def forward(
        self,
        x: torch.Tensor,
        physics_attention_bias: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Apply the physics-aware Transformer block to a spatial feature map.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape ``[B, C, H, W]``.
        physics_attention_bias : torch.Tensor, optional
            Additive pre-softmax attention bias of shape
            ``[B, H*W, H*W]``, forwarded only to ``PhysicsAwareAttention``.
            If ``None`` (default), reduces to ordinary self-attention.
        return_attention : bool, optional
            If ``True``, also return per-head attention weights, shape
            ``[B, num_heads, H*W, H*W]``. Default is ``False``.

        Returns
        -------
        torch.Tensor or tuple of (torch.Tensor, torch.Tensor or None)
            Output feature map, shape ``[B, C, H, W]``, optionally paired
            with attention weights.
        """
        return self._forward_impl(
            x, return_attention, physics_attention_bias=physics_attention_bias
        )


if __name__ == "__main__":
    x = torch.randn(2, 192, 32, 32)

    block = TransformerBlock(channels=192, num_heads=8)
    output = block(x)
    assert x.shape == output.shape, (
        f"Shape mismatch: input {tuple(x.shape)} vs output {tuple(output.shape)}"
    )
    out_attn, weights = block(x, return_attention=True)
    assert weights.shape == (2, 32 * 32, 32 * 32)
    print("TransformerBlock smoke test passed.")

    physics_block = PhysicsTransformerBlock(channels=192, num_heads=8)
    out_no_bias = physics_block(x)
    assert out_no_bias.shape == x.shape
    bias = torch.randn(2, 32 * 32, 32 * 32)
    out_with_bias = physics_block(x, physics_attention_bias=bias)
    assert out_with_bias.shape == x.shape
    out_with_attn, phys_weights = physics_block(x, return_attention=True)
    assert phys_weights.shape == (2, 8, 32 * 32, 32 * 32)
    print("PhysicsTransformerBlock smoke test passed.")
