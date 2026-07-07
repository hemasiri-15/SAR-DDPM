"""
Physics-aware multi-head self-attention wrapper.

This module implements ``PhysicsAwareAttention``: a self-attention layer
that behaves identically to standard multi-head self-attention, but
additionally accepts an optional externally-computed additive attention
bias (``physics_attention_bias``).

Scope
-----
This file is completely standalone. It does not import from, modify, or
get imported by ``transformer_block.py``, ``guided_diffusion/unet.py``, or
any physics-bias-computation module (e.g. ``physics_bias_fusion.py``). It
depends only on ``torch``.

It is intended to be consumed later by a ``PhysicsTransformerBlock`` that
swaps this module in for ``nn.MultiheadAttention`` inside an otherwise
shared Transformer-block implementation.

Design rationale
----------------
Why an *additive* attention bias
    Standard scaled-dot-product attention computes
    ``softmax(QK^T / sqrt(d))``. An additive bias term ``B`` folded in as
    ``softmax(QK^T / sqrt(d) + B)`` is a strict superset of ordinary
    attention: when ``B`` is all-zeros (or ``None``), the result is
    numerically identical to unbiased attention. This is exactly the
    mechanism PyTorch's own ``attn_mask`` argument already implements
    (it is added to the raw attention scores before the softmax), and it
    is the same mechanism used for relative-position biases (e.g. T5,
    Swin) and physically-informed biases in the wider literature. Because
    the bias is purely additive and optional, this module is a drop-in
    replacement for standard self-attention -- no architectural change is
    required elsewhere to start feeding in a bias, and omitting the bias
    exactly reproduces unbiased behaviour.

Why wrap ``nn.MultiheadAttention`` instead of reimplementing attention
    Reimplementing scaled-dot-product attention by hand would duplicate
    well-tested, highly-optimized logic (including PyTorch's fused
    SDPA/FlashAttention backends) for no numerical benefit, since PyTorch
    already exposes an ``attn_mask`` hook that accepts exactly the
    additive-bias formulation described above. Wrapping
    ``nn.MultiheadAttention`` keeps this module small, keeps its numerics
    identical to plain multi-head attention in the unbiased case, and
    automatically inherits any future PyTorch attention-kernel
    improvements.

Why this enables future physics-aware attention without changing the
Transformer architecture
    Because the only new surface area is a single optional forward
    argument with a well-defined additive semantics, any component that
    already knows how to call ordinary self-attention can be extended to
    call this module instead with no other changes: pass ``None`` to get
    identical behaviour to before, or pass a ``[B, N, N]`` tensor to
    inject physics-derived structure (e.g. from a structure tensor,
    wavelet coherence map, or other physically-motivated computation)
    into the attention scores. The Transformer block that hosts this
    module, its residual/normalization structure, and its training
    behaviour do not need to know anything about *how* the bias was
    computed.

Examples
--------
>>> attn = PhysicsAwareAttention(embed_dim=192, num_heads=4)
>>> x = torch.randn(2, 128, 192)
>>> out = attn(x)
>>> out.shape
torch.Size([2, 128, 192])
>>> bias = torch.randn(2, 128, 128)
>>> out = attn(x, physics_attention_bias=bias)
>>> out.shape
torch.Size([2, 128, 192])
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class PhysicsAwareAttention(nn.Module):
    """Multi-head self-attention with an optional additive physics bias.

    Parameters
    ----------
    embed_dim : int
        Total embedding dimension ``D``. Must be positive and divisible by
        ``num_heads``.
    num_heads : int
        Number of attention heads. Must be positive.
    dropout : float, optional
        Dropout probability applied inside the underlying
        ``nn.MultiheadAttention``. Default is ``0.0``.

    Attributes
    ----------
    attn : nn.MultiheadAttention
        The underlying multi-head attention module, configured with
        ``batch_first=True`` so it directly consumes ``[B, N, D]`` tensors.

    Notes
    -----
    This module owns no learnable physics parameters. It is a thin,
    stateless-with-respect-to-physics wrapper: any physics-derived signal
    must be computed elsewhere (e.g. by a ``PhysicsBiasFusion``-style
    module) and passed in as ``physics_attention_bias`` on each call.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(
                f"`embed_dim` must be a positive integer, got {embed_dim!r}."
            )
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                f"`num_heads` must be a positive integer, got {num_heads!r}."
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                "`embed_dim` must be divisible by `num_heads` so that "
                f"multi-head attention can split channels evenly across "
                f"heads, got embed_dim={embed_dim} and num_heads={num_heads} "
                f"(embed_dim % num_heads = {embed_dim % num_heads})."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"`dropout` must be in the range [0, 1), got {dropout!r}."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def _expand_bias_to_attn_mask(
        self, physics_attention_bias: torch.Tensor, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        """Convert a ``[B, N, N]`` additive bias into PyTorch's attn_mask format.

        ``nn.MultiheadAttention`` (non-batched call via a batch-first
        module) expects a float ``attn_mask`` of shape either
        ``[N, N]`` (broadcast across batch and heads) or
        ``[B * num_heads, N, N]`` (one mask per batch element and head).
        Since the physics bias here is per-batch-element but shared across
        heads, it is expanded across the head dimension and reshaped/cast
        to satisfy that contract.

        Parameters
        ----------
        physics_attention_bias : torch.Tensor
            Additive bias of shape ``[B, N, N]``.
        batch_size : int
            Batch size ``B``, used to validate the bias shape.
        seq_len : int
            Sequence length ``N``, used to validate the bias shape.

        Returns
        -------
        torch.Tensor
            Float tensor of shape ``[B * num_heads, N, N]`` suitable for
            ``nn.MultiheadAttention``'s ``attn_mask`` argument.
        """
        if physics_attention_bias.dim() != 3:
            raise ValueError(
                "`physics_attention_bias` must be a 3-D tensor of shape "
                f"[B, N, N], got tensor with "
                f"{physics_attention_bias.dim()} dimensions and shape "
                f"{tuple(physics_attention_bias.shape)}."
            )
        bias_batch, bias_n1, bias_n2 = physics_attention_bias.shape
        if bias_batch != batch_size or bias_n1 != seq_len or bias_n2 != seq_len:
            raise ValueError(
                "`physics_attention_bias` shape must be "
                f"[B, N, N] = [{batch_size}, {seq_len}, {seq_len}] to match "
                f"the input `x`, got {tuple(physics_attention_bias.shape)}."
            )

        # [B, N, N] -> [B, 1, N, N] -> [B, num_heads, N, N] -> [B * num_heads, N, N]
        attn_mask = physics_attention_bias.unsqueeze(1).expand(
            -1, self.num_heads, -1, -1
        )
        attn_mask = attn_mask.reshape(
            batch_size * self.num_heads, seq_len, seq_len
        )
        return attn_mask.to(dtype=physics_attention_bias.dtype)

    def forward(
        self,
        x: torch.Tensor,
        physics_attention_bias: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Apply physics-aware self-attention to a sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape ``[B, N, D]``, where ``D`` must equal
            ``self.embed_dim``.
        physics_attention_bias : torch.Tensor, optional
            Additive pre-softmax attention bias of shape ``[B, N, N]``.
            If ``None`` (default), this module performs ordinary
            self-attention, numerically identical to
            ``nn.MultiheadAttention`` with no mask.
        return_attention : bool, optional
            If ``True``, also return the attention weights, per head
            (``nn.MultiheadAttention``'s ``average_attn_weights=False``
            semantics), with shape ``[B, num_heads, N, N]``. Default is
            ``False``.

        Returns
        -------
        torch.Tensor or tuple of (torch.Tensor, torch.Tensor or None)
            If ``return_attention`` is ``False`` (default): the output
            sequence, shape ``[B, N, D]``.
            If ``return_attention`` is ``True``: a tuple
            ``(output, attention_weights)``.

        Raises
        ------
        ValueError
            If ``x`` is not a 3-D tensor, its embedding dimension does not
            match ``self.embed_dim``, or ``physics_attention_bias`` (when
            provided) does not have shape ``[B, N, N]``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected a 3-D input tensor [B, N, D], got tensor with "
                f"{x.dim()} dimensions and shape {tuple(x.shape)}."
            )

        batch_size, seq_len, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Input embedding dimension ({embed_dim}) does not match "
                f"the embedding dimension this module was constructed with "
                f"({self.embed_dim})."
            )

        attn_mask: Optional[torch.Tensor] = None
        if physics_attention_bias is not None:
            attn_mask = self._expand_bias_to_attn_mask(
                physics_attention_bias, batch_size, seq_len
            )

        attn_out, attn_weights = self.attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        if return_attention:
            return attn_out, attn_weights
        return attn_out


if __name__ == "__main__":
    attn = PhysicsAwareAttention(embed_dim=192, num_heads=4)
    x = torch.randn(2, 128, 192)
    bias = torch.randn(2, 128, 128)

    out_no_bias = attn(x)
    assert out_no_bias.shape == (2, 128, 192), (
        f"Shape mismatch (no bias): got {tuple(out_no_bias.shape)}"
    )

    out_with_bias = attn(x, physics_attention_bias=bias)
    assert out_with_bias.shape == (2, 128, 192), (
        f"Shape mismatch (with bias): got {tuple(out_with_bias.shape)}"
    )

    out_with_attn, weights = attn(x, return_attention=True)
    assert out_with_attn.shape == (2, 128, 192)
    assert weights.shape == (2, 4, 128, 128), (
        f"Attention weights shape mismatch: got {tuple(weights.shape)}"
    )

    print("PhysicsAwareAttention smoke test passed.")
