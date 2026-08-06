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
swaps this module in for standard self-attention inside an otherwise
shared Transformer-block implementation.

Design rationale
----------------
Why an *additive* attention bias
    Standard scaled-dot-product attention computes
    ``softmax(QK^T / sqrt(d))``. An additive bias term ``B`` folded in as
    ``softmax(QK^T / sqrt(d) + B)`` is a strict superset of ordinary
    attention: when ``B`` is all-zeros (or ``None``), the result is
    numerically identical to unbiased attention. This is exactly the
    mechanism PyTorch's fused ``scaled_dot_product_attention`` already
    implements via its ``attn_mask`` argument (a float mask is added to
    the raw attention scores before the softmax), and it is the same
    mechanism used for relative-position biases (e.g. T5, Swin) and
    physically-informed biases in the wider literature. Because the bias
    is purely additive and optional, this module is a drop-in replacement
    for standard self-attention -- no architectural change is required
    elsewhere to start feeding in a bias, and omitting the bias exactly
    reproduces unbiased behaviour.

Why Q/K/V projections are manual rather than a black-box attention module
    This module owns its own ``q_proj`` / ``k_proj`` / ``v_proj`` /
    ``out_proj`` linear layers and calls
    ``torch.nn.functional.scaled_dot_product_attention`` directly (falling
    back to a hand-written softmax-attention implementation only when the
    caller needs the raw attention weights back, which the fused SDPA
    kernel does not expose). This keeps the fast path on PyTorch's fused
    SDPA/FlashAttention backend -- including automatic inheritance of
    future kernel improvements -- while still allowing exact reproduction
    of unbiased multi-head attention when ``physics_attention_bias`` is
    ``None``.

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

import math
import torch.nn.functional as F

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
        Dropout probability applied to the post-softmax attention weights.
        Default is ``0.0``.

    Attributes
    ----------
    q_proj, k_proj, v_proj, out_proj : nn.Linear
        The learned linear projections for queries, keys, values, and the
        post-attention output projection. There is no separate
        ``nn.MultiheadAttention`` submodule -- attention itself is computed
        either via ``torch.nn.functional.scaled_dot_product_attention``
        (fast path) or a manual softmax implementation (used when raw
        attention weights, or the attention-shift research metric, are
        needed -- see ``track_attention_shift`` below).

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
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout_layer = nn.Dropout(dropout)

        self.use_sdpa = True  # Enable PyTorch SDPA / Flash Attention backend

        # ==========================================================
        # Research metrics
        # ==========================================================
        self.track_attention_shift = False
        self.last_attention_shift = None

    def enable_sdpa(self, enabled: bool = True) -> None:
        """
        Enable or disable the SDPA backend.

        Parameters
        ----------
        enabled : bool
            If True, the module will use the fused SDPA implementation
            whenever possible. Otherwise it always falls back to the
            manual attention implementation.
        """
        self.use_sdpa = enabled

    def _validate_physics_bias(
        self, physics_attention_bias: torch.Tensor, batch_size: int, seq_len: int
    ) -> None:
        """Validate that an additive physics bias matches the expected shape.

        ``physics_attention_bias`` must be ``[B, N, N]`` so it can be
        broadcast across the head dimension (both the SDPA and manual
        attention paths add it as ``[B, 1, N, N]`` against ``[B, H, N, N]``
        scores). This check runs unconditionally in ``forward`` -- it's
        pure shape/metadata inspection (no CUDA sync), so it's cheap enough
        to always run and gives a clear error instead of an opaque failure
        deep inside matmul/SDPA.
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
            self-attention, numerically identical to unbiased attention.
        return_attention : bool, optional
            If ``True``, also return the attention weights, per head, with
            shape ``[B, num_heads, N, N]``. These are the raw, pre-dropout
            softmax probabilities (dropout is applied afterward, only to
            the weights actually used for the output matmul, so it doesn't
            affect what's returned here). Forces the manual attention path,
            since the fused SDPA kernel does not expose weights.
            Default is ``False``.

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

        if physics_attention_bias is not None:
            self._validate_physics_bias(physics_attention_bias, batch_size, seq_len)

        B, N, D = x.shape
        H = self.num_heads
        Hd = self.head_dim

        # ==========================================================
        # Fast SDPA path:
        # Used during normal training/inference. Falls back to the manual
        # implementation when attention weights are explicitly requested,
        # or when the attention-shift research metric is being tracked --
        # both require the actual softmax weights, which the fused SDPA
        # kernel does not expose.
        # ==========================================================
        use_sdpa = (
            self.use_sdpa
            and not return_attention
            and not self.track_attention_shift
            and hasattr(F, "scaled_dot_product_attention")
        )

        # ----------------------------
        # Q K V
        # ----------------------------
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, N, H, Hd).transpose(1, 2)
        k = k.view(B, N, H, Hd).transpose(1, 2)
        v = v.view(B, N, H, Hd).transpose(1, 2)

        if use_sdpa:
            attn_mask = None
            if physics_attention_bias is not None:
                attn_mask = physics_attention_bias.unsqueeze(1).to(q.dtype)

            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
            attn_weights = None

        else:
            # Scores/softmax computed in fp32 for numerical stability
            # regardless of q/k/v's dtype (matches standard mixed-precision
            # attention practice).
            scores = torch.matmul(q.float(), k.transpose(-2, -1).float())
            scores /= math.sqrt(Hd)

            if physics_attention_bias is not None:
                scores = scores + physics_attention_bias.unsqueeze(1).float()

            weights_fp32 = F.softmax(scores, dim=-1)

            # ==========================================================
            # Research Metric: Attention Shift
            # Measures how much the physics prior changes the attention
            # distribution compared to standard self-attention. Computed
            # from the fp32 softmax output (before any precision-narrowing
            # cast below) so the metric itself isn't degraded by whatever
            # dtype v happens to be in.
            # ==========================================================
            if self.track_attention_shift and physics_attention_bias is not None:
                with torch.no_grad():
                    scores_no_bias = torch.matmul(
                        q.float(), k.transpose(-2, -1).float()
                    ) / math.sqrt(Hd)
                    weights_no_bias = F.softmax(scores_no_bias, dim=-1)
                    attention_shift = (weights_fp32 - weights_no_bias).abs().mean()
                    self.last_attention_shift = attention_shift.detach()

            # Weights returned to the caller (when requested) are the raw,
            # pre-dropout, fp32 softmax probabilities -- dropout is a
            # training-time regularization artifact, not part of the
            # underlying attention distribution, so it shouldn't be baked
            # into weights someone inspects or visualizes later. The matmul
            # below still uses the post-dropout, cast weights, so training
            # behavior is unaffected.
            attn_weights = weights_fp32 if return_attention else None

            weights = self.dropout_layer(weights_fp32)
            weights = weights.to(v.dtype)  # cast once, immediately before the AV matmul

            attn_out = torch.matmul(weights, v)

        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(B, N, D)
        )

        attn_out = self.out_proj(attn_out)

        if return_attention:
            return attn_out, attn_weights
        return attn_out

    def get_attention_shift(self):
        """
        Return the latest physics attention shift metric.
        """
        return self.last_attention_shift

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

    attn.track_attention_shift = True
    _ = attn(x, physics_attention_bias=bias)
    assert attn.get_attention_shift() is not None, (
        "Attention-shift metric was not populated when track_attention_shift=True"
    )

    print("PhysicsAwareAttention smoke test passed.")
