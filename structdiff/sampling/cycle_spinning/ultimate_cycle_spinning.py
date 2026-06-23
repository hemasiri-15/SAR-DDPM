"""
structdiff/inference/ultimate_cycle_spinning.py
================================================
UltimateCycleSpinning — Post-freeze Corrections Applied
(Integrates A26i through A26q; corrections CORR-1 through CORR-8.)

Post-freeze corrections
-----------------------
CORR-1  Shift-wise frequency pyramid.
         _decompose_input now called per shift (not only on outputs[0]).
         level_wav is a real per-shift list of frequency-band tensors.

CORR-2  Level ID embedding.
         nn.Embedding(num_levels, token_dim) injected before shared_blocks
         so the backbone distinguishes fine/medium/coarse levels.

CORR-3  Automatic Gumbel temperature annealing.
         gumbel_decay stored; step_gumbel_temperature() auto-decays.
         Callers no longer need to track temperature externally.

CORR-4  Sample-specific deformable offsets.
         OffsetMLP output is [B, N, 2]; RelativeCoordinateBias accepts
         offsets and returns [B, H, N, N] bias — no batch averaging.

CORR-5  MoE router entropy regulariser.
         router_entropy_lambda (independent of moe_lambda) penalises
         expert collapse. MoE.forward() returns 3 values.

CORR-6  Cross-level diversity uses coord + base_radius + entropy.
         Previously used mean coordinate alone.

CORR-7  feature_diversity_threshold default 0.8 -> 0.6.

CORR-8  radius_descriptor extended to 5-D (adds structure coherence).

Frozen Decisions (post-correction)
-----------------------------------
Frequency pyramid     : per-shift Haar (CORR-1)
Level ID embedding    : nn.Embedding(L, token_dim) pre-backbone (CORR-2)
Shift range           : min_shifts=2, max_shifts=16
Shift count training  : Soft Gumbel-softmax
Shift count inference : Straight-through
Gumbel annealing      : Auto via gumbel_decay=0.99995 (CORR-3)
Deformable offsets    : Per-sample [B,N,2] (CORR-4)
MoE experts           : 4 (texture, homogeneous, edge, high-freq)
MoE routing           : topk=2
MoE router entropy    : Independent router_entropy_lambda (CORR-5)
feature_div threshold : 0.6 (CORR-7)
radius_descriptor dim : 5 with structure coherence (CORR-8)
Cross-level diversity : coord + radius + entropy (CORR-6)
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as grad_ckpt

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_VALID_POOLING_MODES: frozenset = frozenset({"avg", "max", "multi"})
_FINAL_LAYER_INIT_STD: float = 1e-3
_LOG_EPS: float = 1e-8
_CONFIDENCE_CHANNELS: int = 1
_DIST_EPS: float = 1e-6
_LEVEL_STAT_DIM: int = 8
_MULTI_POOL_FACTOR: int = 3
_MIN_SHIFTS_HARD: int = 2
_MAX_SHIFTS_HARD: int = 16
_MOE_NUM_EXPERTS: int = 4
_MOE_TOPK: int = 2
_GUMBEL_TEMP_INIT: float = 1.0
_GUMBEL_TEMP_MIN: float = 0.1
_GUMBEL_DECAY_DEFAULT: float = 0.99995   # CORR-3
_RADIUS_DESCRIPTOR_DIM: int = 5          # CORR-8 (was 4)

_LEVEL_PLOT_COLORS: Tuple[str, ...] = (
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
)

_HAS_FLASH_ATTN: bool = hasattr(F, "scaled_dot_product_attention")

_SOBEL_X: torch.Tensor = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32,
).reshape(1, 1, 3, 3)

_SOBEL_Y: torch.Tensor = torch.tensor(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32,
).reshape(1, 1, 3, 3)

_HAAR_LL: torch.Tensor = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32).reshape(1,1,2,2)
_HAAR_LH: torch.Tensor = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32).reshape(1,1,2,2)
_HAAR_HL: torch.Tensor = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32).reshape(1,1,2,2)
_HAAR_HH: torch.Tensor = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32).reshape(1,1,2,2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pool_factor(pooling: str) -> int:
    return _MULTI_POOL_FACTOR if pooling == "multi" else 1


@torch.no_grad()
def _sobel_energy(x: torch.Tensor) -> torch.Tensor:
    gray = x.detach().float().mean(dim=1, keepdim=True)
    device = gray.device
    gx = F.conv2d(gray, _SOBEL_X.to(device), padding=1)
    gy = F.conv2d(gray, _SOBEL_Y.to(device), padding=1)
    return (gx ** 2 + gy ** 2).mean(dim=[1, 2, 3])


@torch.no_grad()
def _structure_coherence(x: torch.Tensor) -> torch.Tensor:
    """Per-image (lambda1 - lambda2)/(lambda1 + lambda2) via Sobel structure tensor.

    CORR-8: 5th slot in the radius conditioning descriptor.
    Returns [B] in [0, 1], no gradient.
    """
    gray = x.detach().float().mean(dim=1, keepdim=True)
    device = gray.device
    gx = F.conv2d(gray, _SOBEL_X.to(device), padding=1)
    gy = F.conv2d(gray, _SOBEL_Y.to(device), padding=1)
    j11 = (gx * gx).mean(dim=[1, 2, 3])
    j12 = (gx * gy).mean(dim=[1, 2, 3])
    j22 = (gy * gy).mean(dim=[1, 2, 3])
    T = j11 + j22
    D = j11 * j22 - j12 * j12
    inner = ((T / 2) ** 2 - D).clamp(min=0.0).sqrt()
    lam1 = T / 2 + inner
    lam2 = (T / 2 - inner).clamp(min=0.0)
    return (lam1 - lam2) / (lam1 + lam2 + _DIST_EPS)


# ---------------------------------------------------------------------------
# Frequency Pyramid
# ---------------------------------------------------------------------------

class HaarPyramid(nn.Module):
    """Single-level 2-D Haar wavelet decomposition (depthwise, no learnable params)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.register_buffer("_ker_ll", _HAAR_LL.expand(channels, 1, 2, 2).contiguous())
        self.register_buffer("_ker_lh", _HAAR_LH.expand(channels, 1, 2, 2).contiguous())
        self.register_buffer("_ker_hl", _HAAR_HL.expand(channels, 1, 2, 2).contiguous())
        self.register_buffer("_ker_hh", _HAAR_HH.expand(channels, 1, 2, 2).contiguous())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[2] % 2 != 0 or x.shape[3] % 2 != 0:
            x = F.pad(x, (0, x.shape[3] % 2, 0, x.shape[2] % 2))
        dtype = x.dtype
        x_f = x.float()
        ll = F.conv2d(x_f, self._ker_ll.float(), stride=2, groups=self.channels).to(dtype)
        lh = F.conv2d(x_f, self._ker_lh.float(), stride=2, groups=self.channels).to(dtype)
        hl = F.conv2d(x_f, self._ker_hl.float(), stride=2, groups=self.channels).to(dtype)
        hh = F.conv2d(x_f, self._ker_hh.float(), stride=2, groups=self.channels).to(dtype)
        return ll, lh, hl, hh

    def extra_repr(self) -> str:
        return f"channels={self.channels}"


# ---------------------------------------------------------------------------
# DropPath
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * mask / keep

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


# ---------------------------------------------------------------------------
# Multi-stat pooling
# ---------------------------------------------------------------------------

class MultiStatPool2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._avg = nn.AdaptiveAvgPool2d(1)
        self._max = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        avg = self._avg(x).reshape(B, -1)
        mx = self._max(x).reshape(B, -1)
        std = x.flatten(2).std(dim=-1)
        return torch.cat([avg, mx, std], dim=1)


def _build_pool(pooling: str) -> nn.Module:
    if pooling == "avg":
        return nn.AdaptiveAvgPool2d(1)
    elif pooling == "max":
        return nn.AdaptiveMaxPool2d(1)
    return MultiStatPool2d()


def _apply_pool(pool: nn.Module, x: torch.Tensor) -> torch.Tensor:
    B = x.shape[0]
    out = pool(x)
    if isinstance(pool, MultiStatPool2d):
        return out
    return out.reshape(B, -1)


# ---------------------------------------------------------------------------
# Learnable Radius
# ---------------------------------------------------------------------------

class LearnableRadius(nn.Module):
    """Softplus radius conditioned on a descriptor.  CORR-8: descriptor_dim=5."""

    def __init__(self, init_radius: float, descriptor_dim: int) -> None:
        super().__init__()
        if init_radius <= 0.0:
            raise ValueError(f"init_radius must be > 0, got {init_radius}.")
        raw_init = math.log(math.exp(init_radius) - 1.0)
        self.raw_radius = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        self.cond_mlp = nn.Sequential(
            nn.Linear(descriptor_dim, 32), nn.GELU(), nn.Linear(32, 1),
        )
        nn.init.normal_(self.cond_mlp[2].weight, std=_FINAL_LAYER_INIT_STD)
        nn.init.zeros_(self.cond_mlp[2].bias)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        delta = self.cond_mlp(descriptor).squeeze(-1)
        return F.softplus(self.raw_radius + delta)

    def base_radius(self) -> float:
        with torch.no_grad():
            return float(F.softplus(self.raw_radius).item())

    def extra_repr(self) -> str:
        return f"base_radius≈{self.base_radius():.3f}"


# ---------------------------------------------------------------------------
# Dynamic Shift Count Predictor
# ---------------------------------------------------------------------------

class ShiftCountPredictor(nn.Module):
    """Gumbel-softmax shift count predictor.  CORR-3: auto-annealing via step_temperature()."""

    def __init__(
        self,
        descriptor_dim: int,
        min_shifts: int = _MIN_SHIFTS_HARD,
        max_shifts: int = _MAX_SHIFTS_HARD,
        temperature: float = _GUMBEL_TEMP_INIT,
        gumbel_decay: float = _GUMBEL_DECAY_DEFAULT,
    ) -> None:
        super().__init__()
        if min_shifts < _MIN_SHIFTS_HARD:
            raise ValueError(f"min_shifts={min_shifts} violates hard floor {_MIN_SHIFTS_HARD}.")
        if max_shifts > _MAX_SHIFTS_HARD:
            raise ValueError(f"max_shifts={max_shifts} violates hard ceiling {_MAX_SHIFTS_HARD}.")
        if min_shifts >= max_shifts:
            raise ValueError(f"min_shifts={min_shifts} must be < max_shifts={max_shifts}.")
        self.min_shifts = min_shifts
        self.max_shifts = max_shifts
        self.num_choices = max_shifts - min_shifts + 1
        self.temperature = temperature
        self.gumbel_decay = gumbel_decay
        self.mlp = nn.Sequential(
            nn.Linear(descriptor_dim, 64), nn.GELU(), nn.Linear(64, self.num_choices),
        )
        nn.init.normal_(self.mlp[2].weight, std=_FINAL_LAYER_INIT_STD)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        B = descriptor.shape[0]
        logits = self.mlp(descriptor)
        if self.training:
            gumbel = -torch.empty_like(logits).exponential_().log()
            soft_probs = F.softmax((logits + gumbel) / self.temperature, dim=-1)
            count_vals = torch.arange(
                self.min_shifts, self.max_shifts + 1, dtype=logits.dtype, device=logits.device,
            )
            expected_k = (soft_probs * count_vals).sum(dim=-1)
            positions = torch.arange(
                self.max_shifts, dtype=logits.dtype, device=logits.device,
            ).unsqueeze(0)
            return torch.sigmoid((expected_k.unsqueeze(1) - positions) / self.temperature)
        else:
            k_idx = logits.argmax(dim=-1)
            k = k_idx + self.min_shifts
            positions = torch.arange(self.max_shifts, device=logits.device).unsqueeze(0).expand(B, -1)
            return (positions < k.unsqueeze(1)).float()

    def step_temperature(self) -> float:
        """CORR-3: auto-decay. Call once per training step."""
        self.temperature = max(self.temperature * self.gumbel_decay, _GUMBEL_TEMP_MIN)
        return self.temperature

    def set_temperature(self, temp: float) -> None:
        self.temperature = max(float(temp), _GUMBEL_TEMP_MIN)

    def extra_repr(self) -> str:
        return (
            f"min_shifts={self.min_shifts}, max_shifts={self.max_shifts}, "
            f"temperature={self.temperature:.4f}, gumbel_decay={self.gumbel_decay}"
        )


# ---------------------------------------------------------------------------
# Relative Position Bias
# ---------------------------------------------------------------------------

class RelativePositionBias(nn.Module):
    def __init__(self, seq_len: int, num_heads: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        table_size = 2 * seq_len - 1
        self.bias_table = nn.Parameter(torch.zeros(table_size, num_heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)
        coords = torch.arange(seq_len)
        relative = (coords.unsqueeze(0) - coords.unsqueeze(1)) + (seq_len - 1)
        self.register_buffer("relative_index", relative)

    def forward(self) -> torch.Tensor:
        idx = self.relative_index.reshape(-1)
        return (
            self.bias_table[idx]
            .reshape(self.seq_len, self.seq_len, self.num_heads)
            .permute(2, 0, 1).unsqueeze(0)
        )


# ---------------------------------------------------------------------------
# Relative Coordinate Bias  (CORR-4: per-sample offsets)
# ---------------------------------------------------------------------------

class RelativeCoordinateBias(nn.Module):
    """Geometry-aware attention bias.  CORR-4: accepts per-sample offsets [B,N,2]."""

    def __init__(self, num_heads: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_heads),
        )
        nn.init.normal_(self.mlp[2].weight, std=_FINAL_LAYER_INIT_STD)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(
        self,
        coords: torch.Tensor,
        radius: float,
        offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        coords : [N, 2]
        offsets : optional [B, N, 2]  — CORR-4 per-sample
        Returns [B, H, N, N] if offsets given, else [1, H, N, N].
        """
        N = coords.shape[0]
        if offsets is not None:
            B = offsets.shape[0]
            coords_b = coords.unsqueeze(0).expand(B, -1, -1) + offsets  # [B,N,2]
            ci = coords_b.unsqueeze(2).expand(B, N, N, 2)
            cj = coords_b.unsqueeze(1).expand(B, N, N, 2)
            delta = (ci - cj) / (radius + _DIST_EPS)
            return self.mlp(delta).permute(0, 3, 1, 2)     # [B, H, N, N]
        else:
            ci = coords.unsqueeze(1).expand(N, N, 2)
            cj = coords.unsqueeze(0).expand(N, N, 2)
            delta = (ci - cj) / (radius + _DIST_EPS)
            return self.mlp(delta).permute(2, 0, 1).unsqueeze(0)   # [1, H, N, N]


# ---------------------------------------------------------------------------
# Deformable Offset MLP  (CORR-4: returns [B, N, 2])
# ---------------------------------------------------------------------------

class OffsetMLP(nn.Module):
    """Per-sample deformable offsets [B, N, 2].  CORR-4 (was batch-averaged)."""

    def __init__(self, d_model: int, hidden_dim: int = 32, max_offset: float = 1.0) -> None:
        super().__init__()
        self.max_offset = max_offset
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2),
        )
        nn.init.normal_(self.mlp[2].weight, std=_FINAL_LAYER_INIT_STD)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D] -> [B, N, 2]"""
        return self.max_offset * torch.tanh(self.mlp(x))


# ---------------------------------------------------------------------------
# LayerScale, AttentionPool1d, SwiGLUFFN
# ---------------------------------------------------------------------------

class LayerScale(nn.Module):
    def __init__(self, d_model: int, init_value: float = 1e-4) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((d_model,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class AttentionPool1d(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.proj_k = nn.Linear(d_model, d_model, bias=False)
        self.proj_v = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)
        attn = (q @ self.proj_k(x).transpose(-2, -1)) * self.scale
        return (attn.softmax(dim=-1) @ self.proj_v(x)).squeeze(1)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_in, d_hidden)
        self.up_proj = nn.Linear(d_in, d_hidden)
        self.down_proj = nn.Linear(d_hidden, d_out)
        nn.init.kaiming_normal_(self.gate_proj.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.up_proj.weight, nonlinearity="linear")
        nn.init.normal_(self.down_proj.weight, std=_FINAL_LAYER_INIT_STD)
        nn.init.zeros_(self.down_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# FlashMHA  (CORR-4: per-sample deformable offsets)
# ---------------------------------------------------------------------------

class FlashMHA(nn.Module):
    """MHA with Flash Attention, RelPosBias, CoordBias, optional deformable.
    CORR-4: OffsetMLP output [B,N,2] -> CoordBias [B,H,N,N] (no batch avg)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        seq_len: int,
        deformable: bool = False,
        coord_bias_hidden: int = 32,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.deformable = deformable
        self._last_attn_weights: Optional[torch.Tensor] = None

        self.rel_bias = RelativePositionBias(seq_len, num_heads)
        self.coord_bias = RelativeCoordinateBias(num_heads, coord_bias_hidden)
        self.offset_mlp: Optional[OffsetMLP] = (
            OffsetMLP(d_model, coord_bias_hidden) if deformable else None
        )

        if _HAS_FLASH_ATTN:
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model)
        else:
            self.mha = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True,
            )

    def forward(
        self,
        x: torch.Tensor,
        shift_coords: Optional[torch.Tensor] = None,
        radius: float = 1.0,
    ) -> torch.Tensor:
        B, S, D = x.shape
        pos_bias = self.rel_bias()   # [1, H, S, S]

        if shift_coords is not None:
            coords = shift_coords.to(dtype=x.dtype)
            if self.deformable and self.offset_mlp is not None:
                # CORR-4: per-sample [B, N, 2] — no batch averaging
                offsets = self.offset_mlp(x)
                coord_bias = self.coord_bias(coords, radius, offsets=offsets)  # [B,H,N,N]
            else:
                coord_bias = self.coord_bias(coords, radius)   # [1, H, N, N]
            attn_bias = pos_bias + coord_bias
        else:
            attn_bias = pos_bias

        if _HAS_FLASH_ATTN:
            qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            bias_exp = attn_bias.expand(B, -1, -1, -1)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias_exp,
                dropout_p=self.dropout if self.training else 0.0,
            ).transpose(1, 2).reshape(B, S, D)
            out = self.out_proj(out)
            scale = self.head_dim ** -0.5
            self._last_attn_weights = ((q @ k.transpose(-2,-1)) * scale + bias_exp).softmax(dim=-1)
            return out
        else:
            attn_mask = attn_bias.expand(B,-1,-1,-1).reshape(B*self.num_heads, S, S)
            out, attn_w = self.mha(x, x, x, attn_mask=attn_mask, need_weights=True, average_attn_weights=False)
            self._last_attn_weights = attn_w
            return out

    def get_attention_weights(self) -> torch.Tensor:
        if self._last_attn_weights is None:
            raise RuntimeError("No attention weights — call forward() first.")
        return self._last_attn_weights

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_heads={self.num_heads}, deformable={self.deformable}"


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        seq_len: int,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        deformable: bool = False,
        layer_scale_init: float = 1e-4,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = FlashMHA(d_model, num_heads, dropout, seq_len, deformable=deformable)
        self.ffn = SwiGLUFFN(d_model, ffn_multiplier * d_model, d_model)
        self.drop_attn = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate)
        self.ls1 = LayerScale(d_model, layer_scale_init)
        self.ls2 = LayerScale(d_model, layer_scale_init)

    def forward(
        self,
        x: torch.Tensor,
        shift_coords: Optional[torch.Tensor] = None,
        radius: float = 1.0,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.ls1(self.drop_attn(self.attn(self.norm1(x), shift_coords, radius))))
        x = x + self.drop_path(self.ls2(self.ffn(self.norm2(x))))
        return x

    def get_attention_weights(self) -> torch.Tensor:
        return self.attn.get_attention_weights()

    def extra_repr(self) -> str:
        return (
            f"d_model={self.attn.d_model}, num_heads={self.attn.num_heads}, "
            f"deformable={self.attn.deformable}, drop_path={self.drop_path.drop_prob}"
        )


# ---------------------------------------------------------------------------
# MoE Feature Experts  (CORR-5: returns router_entropy_loss)
# ---------------------------------------------------------------------------

class MoEFeatureExperts(nn.Module):
    """CORR-5: forward() returns (out, lb_loss, router_entropy_loss).
    router_entropy_loss = negative router entropy; minimise to prevent collapse.
    Scale by router_entropy_lambda (independent of moe_lambda).
    """

    EXPERT_NAMES: Tuple[str, ...] = ("texture", "homogeneous", "edge", "high_freq")

    def __init__(
        self,
        d_model: int,
        num_experts: int = _MOE_NUM_EXPERTS,
        topk: int = _MOE_TOPK,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        if topk > num_experts:
            raise ValueError(f"topk={topk} must be <= num_experts={num_experts}.")
        self.d_model = d_model
        self.num_experts = num_experts
        self.topk = topk
        self.router = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=_FINAL_LAYER_INIT_STD)
        d_ff = ffn_multiplier * d_model
        self.experts = nn.ModuleList([SwiGLUFFN(d_model, d_ff, d_model) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)

        topk_vals, topk_idx = probs.topk(self.topk, dim=1)
        gate = torch.zeros_like(probs)
        gate.scatter_(1, topk_idx, topk_vals)
        gate = gate / (gate.sum(dim=1, keepdim=True) + _LOG_EPS)

        expert_out = torch.stack([exp(x) for exp in self.experts], dim=1)
        out = (gate.unsqueeze(-1) * expert_out).sum(dim=1)

        # Switch-style load-balancing loss
        topk_onehot = torch.zeros_like(probs)
        topk_onehot.scatter_(1, topk_idx, 1.0)
        f = topk_onehot.mean(dim=0)
        p = probs.mean(dim=0)
        lb_loss = self.num_experts * (f * p).sum()

        # CORR-5: negative router entropy (minimise = maximise entropy = avoid collapse)
        mean_probs = probs.mean(dim=0)
        router_entropy_loss = (mean_probs * torch.log(mean_probs.clamp(min=_LOG_EPS))).sum()

        return out, lb_loss, router_entropy_loss

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_experts={self.num_experts}, topk={self.topk}"


# ---------------------------------------------------------------------------
# Radius Embedding
# ---------------------------------------------------------------------------

class RadiusEmbedding(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, d_out))
        nn.init.normal_(self.mlp[2].weight, std=_FINAL_LAYER_INIT_STD)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, radius: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if radius.ndim == 0:
            return self.mlp(radius.reshape(1,1).to(device=device, dtype=dtype)).squeeze(0)
        return self.mlp(radius.reshape(-1,1).to(device=device, dtype=dtype))


# ---------------------------------------------------------------------------
# ShiftLevel
# ---------------------------------------------------------------------------

class ShiftLevel(nn.Module):
    """Per-level geometry and token construction.  No per-level Transformer."""

    def __init__(
        self,
        num_shifts: int,
        channels: int,
        wavelet_channels: int,
        structure_channels: int,
        coordinate_embed_dim: int,
        num_heads: int,
        dropout: float,
        init_radius: float,
        pooling: str,
        eps: float,
        radius_descriptor_dim: int,
    ) -> None:
        super().__init__()
        pf = _pool_factor(pooling)
        token_dim: int = (
            pf * channels + pf * _CONFIDENCE_CHANNELS
            + pf * wavelet_channels + pf * structure_channels
            + coordinate_embed_dim
        )
        if token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim={token_dim} must be divisible by num_heads={num_heads}."
            )
        self.num_shifts = num_shifts
        self.channels = channels
        self.wavelet_channels = wavelet_channels
        self.structure_channels = structure_channels
        self.coordinate_embed_dim = coordinate_embed_dim
        self.num_heads = num_heads
        self.pooling = pooling
        self.eps = eps
        self.token_dim = token_dim
        self._pf = pf

        self.raw_shift_coords = nn.Parameter(torch.empty(num_shifts, 2, dtype=torch.float32))
        self._init_shift_coords(init_radius)

        self.learnable_radius = LearnableRadius(init_radius, radius_descriptor_dim)
        self.coordinate_embedding = nn.Sequential(
            nn.Linear(2, 32), nn.GELU(), nn.Linear(32, coordinate_embed_dim),
        )
        self.radius_embedding = RadiusEmbedding(coordinate_embed_dim)

        raw_dim = pf * (channels + _CONFIDENCE_CHANNELS + wavelet_channels + structure_channels) + 2 * coordinate_embed_dim
        self.input_proj = nn.Linear(raw_dim, token_dim)
        self.pool = _build_pool(pooling)
        self.conf_gate = nn.Linear(_CONFIDENCE_CHANNELS * pf, 1)
        nn.init.ones_(self.conf_gate.weight)
        nn.init.zeros_(self.conf_gate.bias)

    def _init_shift_coords(self, radius: float) -> None:
        with torch.no_grad():
            if self.num_shifts == 9:
                grid = [[-1.,-1.],[-1.,0.],[-1.,1.],[0.,-1.],[0.,0.],[0.,1.],[1.,-1.],[1.,0.],[1.,1.]]
                t = torch.tensor(grid, dtype=torch.float32)
                self.raw_shift_coords.data.copy_(torch.atanh((t / radius).clamp(-0.999, 0.999)))
            else:
                nn.init.uniform_(self.raw_shift_coords, -1.0, 1.0)

    def get_shift_coordinates(self, radius: Optional[torch.Tensor] = None) -> torch.Tensor:
        r = self.learnable_radius.base_radius() if radius is None else radius
        if isinstance(r, float):
            return r * torch.tanh(self.raw_shift_coords)
        return r.unsqueeze(1).unsqueeze(2) * torch.tanh(self.raw_shift_coords).unsqueeze(0)

    @staticmethod
    def apply_shift(x: torch.Tensor, shift_row: torch.Tensor, shift_col: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        lin_x = torch.linspace(-1.+1./W, 1.-1./W, W, device=device, dtype=dtype)
        lin_y = torch.linspace(-1.+1./H, 1.-1./H, H, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(lin_y, lin_x, indexing="ij")
        base = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B,-1,-1,-1)
        sx = base[..., 0] - shift_col * (2.0 / W)
        sy = base[..., 1] - shift_row * (2.0 / H)
        return F.grid_sample(x, torch.stack([sx, sy], dim=-1), mode="bilinear", padding_mode="reflection", align_corners=False)

    @staticmethod
    def inverse_shift(x, shift_row, shift_col):
        return ShiftLevel.apply_shift(x, -shift_row, -shift_col)

    def build_token_sequence(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
        structure_features: Sequence[torch.Tensor],
        level_token: torch.Tensor,
        radius_per_sample: torch.Tensor,
    ) -> torch.Tensor:
        coords = self.get_shift_coordinates()
        blk_dtype = next(self.input_proj.parameters()).dtype
        coords_c = coords.to(blk_dtype)
        lt = level_token.to(blk_dtype)
        mean_r = radius_per_sample.mean().detach()
        r_embed = self.radius_embedding(mean_r, coords.device, blk_dtype)
        tokens: List[torch.Tensor] = []
        for i, (x_i, sig_i, wav_i, st_i) in enumerate(zip(outputs, confidence_maps, wavelet_features, structure_features)):
            B = x_i.shape[0]
            z = _apply_pool(self.pool, x_i)
            c = _apply_pool(self.pool, sig_i)
            v = _apply_pool(self.pool, wav_i)
            s = _apply_pool(self.pool, st_i)
            g = self.coordinate_embedding(coords_c[i]).unsqueeze(0).expand(B, -1)
            r = r_embed.unsqueeze(0).expand(B, -1)
            raw = torch.cat([z, c, v, s, g, r], dim=1).to(blk_dtype)
            tok = self.input_proj(raw)
            tok = tok * torch.sigmoid(self.conf_gate(c))
            tok = tok + lt.unsqueeze(0)
            tokens.append(tok)
        return torch.stack(tokens, dim=1)

    def aggregate(self, outputs: Sequence[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        B = weights.shape[0]
        stacked = torch.stack(list(outputs), dim=0).permute(1, 0, 2, 3, 4)
        w = weights.view(B, self.num_shifts, 1, 1, 1)
        in_dtype = stacked.dtype
        if stacked.dtype != w.dtype:
            stacked = stacked.to(w.dtype)
        return (stacked * w).sum(dim=1).to(in_dtype)

    def radius_regularizer(self) -> torch.Tensor:
        c = self.get_shift_coordinates()
        return (c * c).sum()

    def repulsion_regularizer(self) -> torch.Tensor:
        c = self.get_shift_coordinates()
        diff = c.unsqueeze(0) - c.unsqueeze(1)
        dist = torch.sqrt((diff * diff).sum(-1) + _DIST_EPS)
        mask = 1.0 - torch.eye(self.num_shifts, device=c.device, dtype=c.dtype)
        return (torch.exp(-dist) * mask).sum()

    def mean_coordinate(self) -> torch.Tensor:
        return self.get_shift_coordinates().mean(dim=0)

    def average_shift_radius(self) -> float:
        with torch.no_grad():
            c = self.get_shift_coordinates()
            return float(torch.sqrt((c * c).sum(-1)).mean().item())

    def coordinate_variance(self) -> float:
        with torch.no_grad():
            return float(self.get_shift_coordinates().var(unbiased=False).item())

    def min_pairwise_distance(self) -> float:
        with torch.no_grad():
            c = self.get_shift_coordinates()
            diff = c.unsqueeze(0) - c.unsqueeze(1)
            dist = torch.sqrt((diff * diff).sum(-1) + _DIST_EPS)
            big = float(dist.max().item()) + 1.0
            dist = dist + torch.eye(self.num_shifts, device=c.device, dtype=c.dtype) * big
            return float(dist.min().item())

    def save_shift_statistics(self) -> Dict[str, float]:
        with torch.no_grad():
            c = self.get_shift_coordinates()
            radii = torch.sqrt((c * c).sum(-1))
        return {
            "avg_radius": float(radii.mean().item()),
            "max_radius": float(radii.max().item()),
            "coord_variance": self.coordinate_variance(),
            "min_pairwise_distance": self.min_pairwise_distance(),
            "mean_row": float(c[:, 0].mean().item()),
            "mean_col": float(c[:, 1].mean().item()),
            "base_radius": self.learnable_radius.base_radius(),
        }

    def freeze(self) -> None:
        for p in self.parameters(): p.requires_grad_(False)

    def unfreeze(self) -> None:
        for p in self.parameters(): p.requires_grad_(True)

    def freeze_coordinates(self) -> None:
        self.raw_shift_coords.requires_grad_(False)

    def unfreeze_coordinates(self) -> None:
        self.raw_shift_coords.requires_grad_(True)

    def extra_repr(self) -> str:
        return f"num_shifts={self.num_shifts}, token_dim={self.token_dim}, pooling={self.pooling!r}"


# ---------------------------------------------------------------------------
# UltimateCycleSpinning
# ---------------------------------------------------------------------------

class UltimateCycleSpinning(nn.Module):
    """Hierarchical cycle-spinning aggregator with post-freeze corrections CORR-1..8.

    New parameters vs A26i-q
    ------------------------
    router_entropy_lambda : float  (CORR-5, default 1e-3)
    gumbel_decay : float           (CORR-3, default 0.99995)

    Changed defaults
    ----------------
    feature_diversity_threshold : 0.6  (was 0.8, CORR-7)
    """

    def __init__(
        self,
        num_levels: int = 3,
        num_shifts: int = 9,
        channels: int = 1,
        wavelet_channels: int = 1,
        structure_channels: int = 13,  # FIX: was 12 — gave prime token_dim=31/level_token_dim=23, undividable by num_heads/cross_level_heads
        coordinate_embed_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 4,
        cross_level_heads: int = 2,
        cross_level_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 1.0,
        level_radii: Sequence[float] = (1.0, 3.0, 6.0),
        min_shifts: int = _MIN_SHIFTS_HARD,
        radius_lambda: float = 1e-4,
        repulsion_lambda: float = 1e-3,
        diversity_lambda: float = 1e-3,
        entropy_lambda: float = 1e-4,
        entropy_target: float = math.log(4),
        feature_diversity_lambda: float = 1e-3,
        feature_diversity_threshold: float = 0.6,      # CORR-7
        moe_lambda: float = 1e-3,
        router_entropy_lambda: float = 1e-3,           # CORR-5
        pooling: str = "avg",
        eps: float = 1e-8,
        drop_path_rate: float = 0.1,
        gradient_checkpointing: bool = False,
        use_frequency_pyramid: bool = True,
        use_moe: bool = True,
        use_deformable_attention: bool = False,
        use_sparse_routing: bool = False,
        deformable_every: int = 2,
        moe_ffn_multiplier: int = 4,
        gumbel_temperature: float = _GUMBEL_TEMP_INIT,
        gumbel_decay: float = _GUMBEL_DECAY_DEFAULT,   # CORR-3
    ) -> None:
        super().__init__()

        if num_levels <= 0:
            raise ValueError(f"num_levels must be > 0, got {num_levels}.")
        if use_frequency_pyramid and num_levels != 3:
            raise ValueError(f"use_frequency_pyramid=True requires num_levels=3.")
        if not (_MIN_SHIFTS_HARD <= num_shifts <= _MAX_SHIFTS_HARD):
            raise ValueError(f"num_shifts must be in [{_MIN_SHIFTS_HARD},{_MAX_SHIFTS_HARD}].")
        if not (_MIN_SHIFTS_HARD <= min_shifts <= num_shifts):
            raise ValueError(f"min_shifts must be in [{_MIN_SHIFTS_HARD},{num_shifts}].")
        level_radii_list = list(level_radii)
        if len(level_radii_list) != num_levels:
            raise ValueError(f"len(level_radii) must equal num_levels={num_levels}.")
        if any(r <= 0 for r in level_radii_list):
            raise ValueError("All level_radii must be > 0.")
        if pooling not in _VALID_POOLING_MODES:
            raise ValueError(f"pooling must be one of {sorted(_VALID_POOLING_MODES)}.")
        if deformable_every < 1:
            raise ValueError(f"deformable_every must be >= 1.")

        pf = _pool_factor(pooling)
        eff_wavelet_ch = wavelet_channels if use_frequency_pyramid else channels

        token_dim: int = (
            pf * channels + pf * _CONFIDENCE_CHANNELS
            + pf * eff_wavelet_ch + pf * structure_channels
            + coordinate_embed_dim
        )
        if token_dim % num_heads != 0:
            raise ValueError(f"token_dim={token_dim} must be divisible by num_heads={num_heads}.")

        level_token_dim: int = (
            pf * channels + pf * _CONFIDENCE_CHANNELS
            + pf * eff_wavelet_ch + pf * structure_channels
            + _LEVEL_STAT_DIM
        )
        if level_token_dim % cross_level_heads != 0:
            raise ValueError(f"level_token_dim={level_token_dim} not divisible by cross_level_heads={cross_level_heads}.")

        # Store hparams
        self.num_levels = num_levels
        self.num_shifts = num_shifts
        self.channels = channels
        self.wavelet_channels = wavelet_channels
        self.eff_wavelet_ch = eff_wavelet_ch
        self.structure_channels = structure_channels
        self.coordinate_embed_dim = coordinate_embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.cross_level_heads = cross_level_heads
        self.cross_level_layers = cross_level_layers
        self.dropout = dropout
        self.temperature = temperature
        self.level_radii = tuple(level_radii_list)
        self.min_shifts = min_shifts
        self.radius_lambda = radius_lambda
        self.repulsion_lambda = repulsion_lambda
        self.diversity_lambda = diversity_lambda
        self.entropy_lambda = entropy_lambda
        self.entropy_target = entropy_target
        self.feature_diversity_lambda = feature_diversity_lambda
        self.feature_diversity_threshold = feature_diversity_threshold
        self.moe_lambda = moe_lambda
        self.router_entropy_lambda = router_entropy_lambda
        self.pooling = pooling
        self.eps = eps
        self.drop_path_rate = drop_path_rate
        self.gradient_checkpointing = gradient_checkpointing
        self.use_frequency_pyramid = use_frequency_pyramid
        self.use_moe = use_moe
        self.use_deformable_attention = use_deformable_attention
        self.use_sparse_routing = use_sparse_routing
        self.deformable_every = deformable_every
        self.moe_ffn_multiplier = moe_ffn_multiplier
        self.gumbel_temperature = gumbel_temperature
        self.gumbel_decay = gumbel_decay
        self.token_dim = token_dim
        self.level_token_dim = level_token_dim
        self._pf = pf

        # Modules
        self.freq_pyramid: Optional[HaarPyramid] = (
            HaarPyramid(channels=channels) if use_frequency_pyramid else None
        )

        self.shift_levels = nn.ModuleList([
            ShiftLevel(
                num_shifts=num_shifts, channels=channels, wavelet_channels=eff_wavelet_ch,
                structure_channels=structure_channels, coordinate_embed_dim=coordinate_embed_dim,
                num_heads=num_heads, dropout=dropout, init_radius=level_radii_list[l],
                pooling=pooling, eps=eps, radius_descriptor_dim=_RADIUS_DESCRIPTOR_DIM,
            )
            for l in range(num_levels)
        ])

        self.level_tokens = nn.Parameter(torch.zeros(num_levels, token_dim))
        nn.init.trunc_normal_(self.level_tokens, std=0.02)

        # CORR-2: level ID embedding
        self.level_id_embedding = nn.Embedding(num_levels, token_dim)
        nn.init.trunc_normal_(self.level_id_embedding.weight, std=0.02)

        # A26k + CORR-3: shift count predictor with auto-annealing
        self.shift_count_predictor = ShiftCountPredictor(
            descriptor_dim=pf * channels,
            min_shifts=min_shifts, max_shifts=num_shifts,
            temperature=gumbel_temperature, gumbel_decay=gumbel_decay,
        )
        self._pool_for_shift_count = _build_pool(pooling)

        # Shared Transformer
        self.shared_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=token_dim, num_heads=num_heads, seq_len=num_shifts,
                ffn_multiplier=4, dropout=dropout, drop_path_rate=drop_path_rate,
                deformable=(use_deformable_attention and ((li % deformable_every) == 1)),
            )
            for li in range(num_layers)
        ])
        self.shared_attn_pool = AttentionPool1d(token_dim)

        self.moe: Optional[MoEFeatureExperts] = (
            MoEFeatureExperts(token_dim, _MOE_NUM_EXPERTS, _MOE_TOPK, moe_ffn_multiplier)
            if use_moe else None
        )
        self.shared_head = SwiGLUFFN(token_dim, 2 * token_dim, num_shifts)

        self.cross_pool = _build_pool(pooling)
        self.cross_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=level_token_dim, num_heads=cross_level_heads, seq_len=num_levels,
                ffn_multiplier=4, dropout=dropout, drop_path_rate=drop_path_rate, deformable=False,
            )
            for _ in range(cross_level_layers)
        ])
        self.cross_attn_pool = AttentionPool1d(level_token_dim)
        self.cross_head = SwiGLUFFN(level_token_dim, 2 * level_token_dim, num_levels)

    # ------------------------------------------------------------------
    # CORR-1: per-shift frequency decomposition
    # ------------------------------------------------------------------

    def _decompose_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompose one shift's tensor into (fine, medium, coarse)."""
        if self.use_frequency_pyramid and self.freq_pyramid is not None:
            ll, lh, hl, hh = self.freq_pyramid(x)
            return hh + hl + lh, x, ll
        return x, x, x

    def _build_per_shift_freq_seqs(
        self, outputs: Sequence[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """CORR-1: decompose every shift, return (fine_seq, medium_seq, coarse_seq)."""
        fine_seq, medium_seq, coarse_seq = [], [], []
        for out_i in outputs:
            f, m, c = self._decompose_input(out_i)
            fine_seq.append(f)
            medium_seq.append(m)
            coarse_seq.append(c)
        return fine_seq, medium_seq, coarse_seq

    # ------------------------------------------------------------------
    # CORR-8: 5-D radius descriptor
    # ------------------------------------------------------------------

    def _radius_descriptor(
        self,
        outputs: Sequence[torch.Tensor],
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """5-D descriptor: [timestep, entropy_proxy, ENL, sobel, coherence].  CORR-8."""
        ref = outputs[0].float()
        B, device, dtype = ref.shape[0], ref.device, outputs[0].dtype
        flat = ref.flatten(2)
        mu = flat.mean(dim=-1)
        std = flat.std(dim=-1).clamp(min=self.eps)
        enl = (mu ** 2 / (std ** 2 + self.eps)).mean(dim=1)
        sobel = _sobel_energy(ref)
        entropy_proxy = torch.log(std.mean(dim=1) + self.eps)
        coherence = _structure_coherence(ref)
        t_norm = (timestep.float().to(device) / 1000.0
                  if timestep is not None
                  else torch.zeros(B, device=device, dtype=torch.float32))
        return torch.stack([t_norm, entropy_proxy.to(dtype), enl.to(dtype),
                            sobel.to(dtype), coherence.to(dtype)], dim=1)

    # ------------------------------------------------------------------
    # Shift count mask
    # ------------------------------------------------------------------

    def _compute_shift_mask(self, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if not (self.use_sparse_routing or self.training):
            B = outputs[0].shape[0]
            return torch.ones(B, self.num_shifts, device=outputs[0].device, dtype=outputs[0].dtype)
        ref = outputs[0]
        pooled = _apply_pool(self._pool_for_shift_count, ref)
        return self.shift_count_predictor(pooled.float()).to(ref.dtype)

    # ------------------------------------------------------------------
    # CORR-2: shared transformer with level ID embedding
    # ------------------------------------------------------------------

    def _shared_transformer(
        self,
        tokens: torch.Tensor,
        shift_coords: torch.Tensor,
        radius: float,
        level_idx: int,
    ) -> torch.Tensor:
        """CORR-2: inject level_id_embedding before backbone."""
        lvl_id = torch.tensor(level_idx, device=tokens.device)
        tokens = tokens + self.level_id_embedding(lvl_id).unsqueeze(0).unsqueeze(0)

        for block in self.shared_blocks:
            if self.gradient_checkpointing and self.training:
                def make_fn(blk, sc, r):
                    def fn(t): return blk(t, sc, r)
                    return fn
                tokens = grad_ckpt.checkpoint(make_fn(block, shift_coords, radius), tokens, use_reentrant=False)
            else:
                tokens = block(tokens, shift_coords, radius)
        return tokens

    # ------------------------------------------------------------------
    # Per-level weight computation  (CORR-5: 3 return values)
    # ------------------------------------------------------------------

    def _get_level_weights(
        self,
        level_idx: int,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
        structure_features: Sequence[torch.Tensor],
        shift_mask: torch.Tensor,
        radius_per_sample: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (weights [B,N], moe_loss, router_entropy_loss)."""
        lv = self.shift_levels[level_idx]
        lt = self.level_tokens[level_idx]
        tokens = lv.build_token_sequence(
            outputs, confidence_maps, wavelet_features, structure_features, lt, radius_per_sample,
        )
        tokens = tokens * shift_mask.unsqueeze(-1)
        tokens = self._shared_transformer(
            tokens, lv.get_shift_coordinates(), lv.learnable_radius.base_radius(), level_idx,
        )
        pooled = self.shared_attn_pool(tokens)

        zero = torch.zeros((), device=pooled.device, dtype=pooled.dtype)
        moe_loss, router_entropy_loss = zero, zero
        if self.use_moe and self.moe is not None:
            pooled, moe_loss, router_entropy_loss = self.moe(pooled)

        logits = self.shared_head(pooled)
        logits = logits + (1.0 - shift_mask) * (-1e4)
        weights = F.softmax(logits / self.temperature, dim=1)
        return weights, moe_loss, router_entropy_loss

    # ------------------------------------------------------------------
    # Cross-level descriptor  (CORR-6: base_radius replaces w_max)
    # ------------------------------------------------------------------

    def _build_level_descriptor(
        self,
        level_idx: int,
        level_fused: torch.Tensor,
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
        structure_features: Sequence[torch.Tensor],
        level_weights: torch.Tensor,
    ) -> torch.Tensor:
        B = level_fused.shape[0]
        image_desc = _apply_pool(self.cross_pool, level_fused)
        conf_desc = _apply_pool(self.cross_pool, torch.stack(list(confidence_maps), dim=0).mean(0))
        wav_desc = _apply_pool(self.cross_pool, torch.stack(list(wavelet_features), dim=0).mean(0))
        st_desc = _apply_pool(self.cross_pool, torch.stack(list(structure_features), dim=0).mean(0))

        lv = self.shift_levels[level_idx]
        avg_radius = lv.average_shift_radius()
        coord_var = lv.coordinate_variance()
        base_radius = lv.learnable_radius.base_radius()   # CORR-6

        per_img_entropy = -(level_weights * torch.log(level_weights.clamp(min=self.eps))).sum(1)
        eff_n = torch.exp(per_img_entropy)
        fused_flat = level_fused.flatten(2)
        fused_mean = fused_flat.mean(dim=-1)
        fused_std = fused_flat.std(dim=-1).clamp(min=self.eps)
        enl = (fused_mean ** 2 / (fused_std ** 2 + self.eps)).mean(dim=1)
        sobel = _sobel_energy(level_fused)
        w_var = level_weights.var(dim=1, unbiased=False)

        dt, dv = image_desc.dtype, image_desc.device
        stats = torch.stack([
            torch.full((B,), avg_radius, dtype=dt, device=dv),
            torch.full((B,), coord_var, dtype=dt, device=dv),
            per_img_entropy.to(dt), eff_n.to(dt), enl.to(dt), sobel.to(dt), w_var.to(dt),
            torch.full((B,), base_radius, dtype=dt, device=dv),  # CORR-6: was w_max
        ], dim=1)
        return torch.cat([image_desc, conf_desc, wav_desc, st_desc, stats], dim=1)

    def _get_cross_level_weights(
        self,
        level_fused_list: List[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
        structure_features: Sequence[torch.Tensor],
        level_weights_list: List[torch.Tensor],
    ) -> torch.Tensor:
        descs = [
            self._build_level_descriptor(l, level_fused_list[l], confidence_maps,
                                         wavelet_features, structure_features, level_weights_list[l])
            for l in range(self.num_levels)
        ]
        tokens = torch.stack(descs, dim=1).to(next(self.cross_blocks[0].parameters()).dtype)
        for block in self.cross_blocks:
            if self.gradient_checkpointing and self.training:
                tokens = grad_ckpt.checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        pooled = self.cross_attn_pool(tokens)
        return F.softmax(self.cross_head(pooled) / self.temperature, dim=1)

    # ------------------------------------------------------------------
    # Uncertainty
    # ------------------------------------------------------------------

    def _compute_level_uncertainty(self, weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        entropy = -(weights * torch.log(weights.clamp(min=self.eps))).sum(1).mean()
        variance = weights.var(dim=1, unbiased=False).mean()
        return {"entropy": entropy, "variance": variance,
                "effective_shifts": torch.exp(entropy), "aleatoric": variance,
                "predictive_mean": weights.mean(dim=0)}

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def _validate_inputs(self, outputs, confidence_maps, wavelet_features, structure_features):
        if len(outputs) != self.num_shifts:
            raise ValueError(f"len(outputs)={len(outputs)} must equal num_shifts={self.num_shifts}.")
        ref = outputs[0]
        if ref.ndim != 4:
            raise ValueError(f"Each output must be [B,C,H,W]; got ndim={ref.ndim}.")
        if ref.shape[1] != self.channels:
            raise ValueError(f"Channel mismatch: expected {self.channels}, got {ref.shape[1]}.")
        for name, seq in [("confidence_maps", confidence_maps), ("wavelet_features", wavelet_features),
                          ("structure_features", structure_features)]:
            if len(seq) != self.num_shifts:
                raise ValueError(f"len({name})={len(seq)} must equal {self.num_shifts}.")
        return ref

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
        structure_features: Sequence[torch.Tensor],
        timestep: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        return_level_outputs: bool = False,
        return_uncertainty: bool = False,
    ) -> Union[torch.Tensor, Tuple]:
        """Aggregate cycle-shifted outputs.

        CORR-1: per-shift Haar decomposition feeds each level's token sequence
        with the appropriate frequency band for that shift.
        wavelet_features from the caller is still used in the cross-level descriptor.

        return_level_outputs=True now returns 6-tuple:
            (fused, alpha, level_outputs, level_weights_list, moe_losses, router_entropy_losses)
        """
        ref = self._validate_inputs(outputs, confidence_maps, wavelet_features, structure_features)
        B = ref.shape[0]

        # CORR-1: per-shift frequency sequences
        fine_seq, medium_seq, coarse_seq = self._build_per_shift_freq_seqs(outputs)
        level_freq_seqs = [fine_seq, medium_seq, coarse_seq]

        # CORR-8: 5-D radius descriptor
        radius_desc = self._radius_descriptor(outputs, timestep)
        shift_mask = self._compute_shift_mask(outputs)

        level_outputs: List[torch.Tensor] = []
        level_weights_list: List[torch.Tensor] = []
        uncertainties: List[Dict[str, torch.Tensor]] = []
        moe_losses: List[torch.Tensor] = []
        router_entropy_losses: List[torch.Tensor] = []

        for l in range(self.num_levels):
            lv = self.shift_levels[l]
            radius_per_sample = lv.learnable_radius(radius_desc)
            level_wav = level_freq_seqs[l]   # CORR-1: per-shift band for this level

            w_l, moe_l, re_l = self._get_level_weights(
                l, outputs, confidence_maps, level_wav, structure_features, shift_mask, radius_per_sample,
            )
            y_l = lv.aggregate(outputs, w_l)
            level_outputs.append(y_l)
            level_weights_list.append(w_l)
            moe_losses.append(moe_l)
            router_entropy_losses.append(re_l)
            if return_uncertainty:
                uncertainties.append(self._compute_level_uncertainty(w_l))

        alpha = self._get_cross_level_weights(
            level_outputs, confidence_maps, wavelet_features, structure_features, level_weights_list,
        )

        stacked = torch.stack(level_outputs, dim=0).permute(1, 0, 2, 3, 4)
        a_bc = alpha.view(B, self.num_levels, 1, 1, 1)
        in_dtype = stacked.dtype
        if stacked.dtype != a_bc.dtype:
            stacked = stacked.to(a_bc.dtype)
        fused = (stacked * a_bc).sum(dim=1).to(in_dtype)

        if return_level_outputs:
            base = (fused, alpha, level_outputs, level_weights_list, moe_losses, router_entropy_losses)
            return base + (uncertainties,) if return_uncertainty else base
        if return_weights:
            return (fused, alpha, uncertainties) if return_uncertainty else (fused, alpha)
        if return_uncertainty:
            return fused, uncertainties
        return fused

    # ------------------------------------------------------------------
    # Regularizers
    # ------------------------------------------------------------------

    def radius_regularizer(self) -> torch.Tensor:
        return self.radius_lambda * sum(lv.radius_regularizer() for lv in self.shift_levels)

    def repulsion_regularizer(self) -> torch.Tensor:
        return self.repulsion_lambda * sum(lv.repulsion_regularizer() for lv in self.shift_levels)

    def coordinate_regularizer(self) -> torch.Tensor:
        return self.radius_regularizer() + self.repulsion_regularizer()

    def cross_level_diversity_regularizer(self) -> torch.Tensor:
        """CORR-6: penalise similarity in coordinates AND base radius."""
        means = [lv.mean_coordinate() for lv in self.shift_levels]
        radii = [lv.learnable_radius.base_radius() for lv in self.shift_levels]
        total = torch.zeros((), dtype=means[0].dtype, device=means[0].device)
        for l in range(self.num_levels):
            for m in range(self.num_levels):
                if l == m:
                    continue
                coord_dist = torch.sqrt(((means[l] - means[m]) ** 2).sum() + _DIST_EPS)
                radius_diff = abs(radii[l] - radii[m])
                total = total + torch.exp(-coord_dist) + math.exp(-radius_diff)
        return self.diversity_lambda * total

    def entropy_regularizer(self, level_weights_list: List[torch.Tensor]) -> torch.Tensor:
        total = torch.zeros((), dtype=level_weights_list[0].dtype, device=level_weights_list[0].device)
        for w in level_weights_list:
            H = -(w * torch.log(w.clamp(min=self.eps))).sum(dim=1).mean()
            total = total + F.relu(torch.tensor(self.entropy_target, dtype=H.dtype, device=H.device) - H)
        return self.entropy_lambda * (total / self.num_levels)

    def feature_diversity_regularizer(self, level_fused_list: List[torch.Tensor]) -> torch.Tensor:
        """CORR-7: threshold 0.6."""
        flat = [y.flatten(1) for y in level_fused_list]
        total = torch.zeros((), dtype=flat[0].dtype, device=flat[0].device)
        count = 0
        for l in range(self.num_levels):
            for m in range(self.num_levels):
                if l == m: continue
                cos = F.cosine_similarity(flat[l], flat[m], dim=1).mean()
                total = total + F.relu(cos - self.feature_diversity_threshold)
                count += 1
        return self.feature_diversity_lambda * (total / max(count, 1))

    def total_regularizer(
        self,
        level_weights_list: Optional[List[torch.Tensor]] = None,
        level_fused_list: Optional[List[torch.Tensor]] = None,
        moe_losses: Optional[List[torch.Tensor]] = None,
        router_entropy_losses: Optional[List[torch.Tensor]] = None,  # CORR-5
    ) -> Dict[str, torch.Tensor]:
        """All regulariser terms with independent coefficients.

        New key: ``router_entropy`` (CORR-5).
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        zero = torch.zeros((), device=device, dtype=dtype)

        terms: Dict[str, torch.Tensor] = {
            "coord": self.coordinate_regularizer(),
            "diversity": self.cross_level_diversity_regularizer(),
            "entropy": self.entropy_regularizer(level_weights_list) if level_weights_list is not None else zero,
            "feature_div": self.feature_diversity_regularizer(level_fused_list) if level_fused_list is not None else zero,
            "moe": (self.moe_lambda * sum(moe_losses) / max(len(moe_losses), 1)
                    if moe_losses is not None else zero),
            "router_entropy": (self.router_entropy_lambda * sum(router_entropy_losses) / max(len(router_entropy_losses), 1)
                               if router_entropy_losses is not None else zero),
        }
        terms["total"] = sum(terms.values())
        return terms

    # ------------------------------------------------------------------
    # CORR-3: temperature management
    # ------------------------------------------------------------------

    def step_gumbel_temperature(self) -> float:
        """CORR-3: auto-decay. Call once per training step."""
        new_temp = self.shift_count_predictor.step_temperature()
        self.gumbel_temperature = new_temp
        return new_temp

    def set_gumbel_temperature(self, temp: float) -> None:
        self.gumbel_temperature = temp
        self.shift_count_predictor.set_temperature(temp)

    def set_temperature(self, temperature: float) -> None:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Freeze utilities
    # ------------------------------------------------------------------

    def freeze_level(self, level: int) -> None:
        self.shift_levels[level].freeze()

    def unfreeze_level(self, level: int) -> None:
        self.shift_levels[level].unfreeze()

    def freeze_shared_backbone(self) -> None:
        for mod in [self.shared_blocks, self.shared_attn_pool, self.shared_head]:
            for p in (mod.parameters() if hasattr(mod, 'parameters') else [mod]):
                if isinstance(p, torch.Tensor): p.requires_grad_(False)
        for p in self.shared_blocks.parameters(): p.requires_grad_(False)
        for p in self.shared_attn_pool.parameters(): p.requires_grad_(False)
        for p in self.shared_head.parameters(): p.requires_grad_(False)

    def unfreeze_shared_backbone(self) -> None:
        for p in self.shared_blocks.parameters(): p.requires_grad_(True)
        for p in self.shared_attn_pool.parameters(): p.requires_grad_(True)
        for p in self.shared_head.parameters(): p.requires_grad_(True)

    def freeze_all(self) -> None:
        for p in self.parameters(): p.requires_grad_(False)

    def unfreeze_all(self) -> None:
        for p in self.parameters(): p.requires_grad_(True)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def save_shift_statistics(self, level: int) -> Dict[str, float]:
        if not (0 <= level < self.num_levels):
            raise ValueError(f"level must be in [0, {self.num_levels}).")
        return self.shift_levels[level].save_shift_statistics()

    def save_statistics(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
        structure_features: Sequence[torch.Tensor],
    ) -> Dict[str, float]:
        with torch.no_grad():
            result = self.forward(outputs, confidence_maps, wavelet_features, structure_features, return_level_outputs=True)
        _, alpha, _, lw, _, _ = result
        entropy = -(alpha * torch.log(alpha.clamp(min=self.eps))).sum(1).mean()
        stats: Dict[str, float] = {
            "cross_entropy": float(entropy.item()),
            "cross_effective_num_levels": float(torch.exp(entropy).item()),
            "cross_max_alpha": float(alpha.max().item()),
            "gumbel_temperature": self.gumbel_temperature,
        }
        for l, (lv, w) in enumerate(zip(self.shift_levels, lw)):
            cs = lv.save_shift_statistics()
            we = -(w * torch.log(w.clamp(min=self.eps))).sum(1).mean()
            stats.update({
                f"level_{l}_avg_radius": cs["avg_radius"],
                f"level_{l}_base_radius": cs["base_radius"],
                f"level_{l}_coord_variance": cs["coord_variance"],
                f"level_{l}_weight_entropy": float(we.item()),
                f"level_{l}_effective_num_shifts": float(torch.exp(we).item()),
            })
        return stats

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_shift_coordinates(self, level: int) -> matplotlib.figure.Figure:
        if not (0 <= level < self.num_levels):
            raise ValueError(f"level must be in [0, {self.num_levels}).")
        lv = self.shift_levels[level]
        with torch.no_grad():
            coords = lv.get_shift_coordinates().cpu()
        rows, cols = coords[:, 0].tolist(), coords[:, 1].tolist()
        fig, ax = plt.subplots(figsize=(5, 5))
        color = _LEVEL_PLOT_COLORS[level % len(_LEVEL_PLOT_COLORS)]
        ax.scatter(cols, rows, s=80, zorder=3, color=color)
        ax.scatter([0], [0], s=120, marker="+", color="black", linewidths=2, zorder=4)
        for idx, (r, c) in enumerate(zip(rows, cols)):
            ax.annotate(str(idx), (c, r), textcoords="offset points", xytext=(5, 5), fontsize=8)
        lim = lv.learnable_radius.base_radius() * 1.15
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.grid(True)
        ax.set_xlabel("Column shift (Δc)"); ax.set_ylabel("Row shift (Δr)")
        ax.set_title(f"UltimateCycleSpinning Level {level} (N={lv.num_shifts}, r≈{lv.learnable_radius.base_radius():.2f})")
        fig.tight_layout()
        return fig

    def plot_attention_heads(self, level: int = 0, batch_idx: int = 0) -> matplotlib.figure.Figure:
        attn = self.shared_blocks[-1].get_attention_weights().detach().cpu()
        H = attn.shape[1]
        fig, axes = plt.subplots(1, H, figsize=(4*H, 4))
        if H == 1: axes = [axes]
        for h, ax in enumerate(axes):
            im = ax.imshow(attn[batch_idx, h].numpy(), cmap="viridis", vmin=0)
            ax.set_title(f"Head {h}"); ax.set_xlabel("Key shift"); ax.set_ylabel("Query shift")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Shared Backbone Attention (last layer), Level {level}")
        fig.tight_layout(); return fig

    def plot_cross_attention(self, batch_idx: int = 0) -> matplotlib.figure.Figure:
        attn = self.cross_blocks[-1].get_attention_weights().detach().cpu()
        H = attn.shape[1]
        fig, axes = plt.subplots(1, H, figsize=(4*H, 4))
        if H == 1: axes = [axes]
        labels = [f"L{l}" for l in range(self.num_levels)]
        for h, ax in enumerate(axes):
            im = ax.imshow(attn[batch_idx, h].numpy(), cmap="plasma", vmin=0)
            ax.set_xticks(range(self.num_levels)); ax.set_yticks(range(self.num_levels))
            ax.set_xticklabels(labels); ax.set_yticklabels(labels)
            ax.set_title(f"Cross-level Head {h}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("Cross-Level Attention"); fig.tight_layout(); return fig

    def plot_level_interactions(self, batch_idx: int = 0) -> matplotlib.figure.Figure:
        return self.plot_cross_attention(batch_idx)

    def plot_uncertainty_maps(self, uncertainties: List[Dict[str, torch.Tensor]]) -> matplotlib.figure.Figure:
        metrics = ["entropy", "variance", "effective_shifts"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
        for ax, metric in zip(axes, metrics):
            vals = [float(unc[metric].item()) for unc in uncertainties if metric in unc]
            ax.bar(range(len(vals)), vals, color=_LEVEL_PLOT_COLORS[:len(vals)])
            ax.set_xticks(range(len(vals))); ax.set_xticklabels([f"L{l}" for l in range(len(vals))])
            ax.set_title(metric); ax.set_xlabel("Level")
        fig.suptitle("Per-Level Uncertainty (A26m)"); fig.tight_layout(); return fig

    def plot_radius_schedule(self) -> matplotlib.figure.Figure:
        radii = [lv.learnable_radius.base_radius() for lv in self.shift_levels]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(range(self.num_levels), radii, color=_LEVEL_PLOT_COLORS[:self.num_levels])
        ax.set_xticks(range(self.num_levels)); ax.set_xticklabels([f"L{l}" for l in range(self.num_levels)])
        ax.set_ylabel("Base radius (softplus)"); ax.set_title("Learnable Radius Schedule (A26j)")
        fig.tight_layout(); return fig

    def plot_shift_count(self, outputs: Sequence[torch.Tensor]) -> matplotlib.figure.Figure:
        with torch.no_grad():
            mask = self._compute_shift_mask(outputs)
        effective_k = mask.sum(dim=1).mean().item()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(self.num_shifts), mask[0].cpu().numpy(), color="#1f77b4")
        ax.set_xlabel("Shift index"); ax.set_ylabel("Activity weight")
        ax.set_title(f"Dynamic Shift Count (A26k) — effective k ≈ {effective_k:.1f}")
        fig.tight_layout(); return fig

    def plot_expert_usage(
        self,
        outputs: Sequence[torch.Tensor],
        confidence_maps: Sequence[torch.Tensor],
        wavelet_features: Sequence[torch.Tensor],
        structure_features: Sequence[torch.Tensor],
    ) -> matplotlib.figure.Figure:
        if not self.use_moe or self.moe is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "MoE disabled", ha="center", va="center")
            return fig
        with torch.no_grad():
            fine_seq, medium_seq, coarse_seq = self._build_per_shift_freq_seqs(outputs)
            level_freq_seqs = [fine_seq, medium_seq, coarse_seq]
            pooled_list: List[torch.Tensor] = []
            for l in range(self.num_levels):
                lv = self.shift_levels[l]
                radius_desc = self._radius_descriptor(outputs)
                radius_ps = lv.learnable_radius(radius_desc)
                shift_mask = self._compute_shift_mask(outputs)
                tokens = lv.build_token_sequence(
                    outputs, confidence_maps, level_freq_seqs[l], structure_features,
                    self.level_tokens[l], radius_ps,
                )
                tokens = tokens * shift_mask.unsqueeze(-1)
                tokens = self._shared_transformer(tokens, lv.get_shift_coordinates(), lv.learnable_radius.base_radius(), l)
                pooled_list.append(self.shared_attn_pool(tokens))
            pooled_all = torch.stack(pooled_list, dim=0).mean(dim=0)
            probs = F.softmax(self.moe.router(pooled_all), dim=-1).mean(dim=0)
        expert_names = list(MoEFeatureExperts.EXPERT_NAMES)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(expert_names)), probs.cpu().numpy(), color=_LEVEL_PLOT_COLORS[:len(expert_names)])
        ax.set_xticks(range(len(expert_names))); ax.set_xticklabels(expert_names, rotation=15)
        ax.set_ylabel("Mean routing probability"); ax.set_title("MoE Expert Usage (A26q)")
        fig.tight_layout(); return fig

    def save_attention_heatmaps(self, outputs, confidence_maps, wavelet_features, structure_features, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.forward(outputs, confidence_maps, wavelet_features, structure_features)
        for l in range(self.num_levels):
            fig = self.plot_attention_heads(l)
            fig.savefig(os.path.join(save_dir, f"shared_attn_level{l}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
        fig = self.plot_cross_attention()
        fig.savefig(os.path.join(save_dir, "cross_level_attn.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def save_cross_attention_heatmaps(self, path: str, batch_idx: int = 0) -> None:
        fig = self.plot_cross_attention(batch_idx)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_all_levels(self) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(figsize=(6, 6))
        for l, lv in enumerate(self.shift_levels):
            with torch.no_grad():
                coords = lv.get_shift_coordinates().cpu()
            color = _LEVEL_PLOT_COLORS[l % len(_LEVEL_PLOT_COLORS)]
            ax.scatter(coords[:, 1].tolist(), coords[:, 0].tolist(), s=70, zorder=3, color=color,
                      label=f"Level {l} (r≈{lv.learnable_radius.base_radius():.2f})")
        ax.scatter([0], [0], s=140, marker="+", color="black", linewidths=2, zorder=4)
        lim = max(lv.learnable_radius.base_radius() for lv in self.shift_levels) * 1.15
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.grid(True)
        ax.set_xlabel("Δc"); ax.set_ylabel("Δr")
        ax.set_title(f"UltimateCycleSpinning — {self.num_levels} levels")
        ax.legend(); fig.tight_layout(); return fig

    def extra_repr(self) -> str:
        flags = ", ".join([
            f"freq_pyramid={self.use_frequency_pyramid}", f"moe={self.use_moe}",
            f"deformable={self.use_deformable_attention}", f"sparse_routing={self.use_sparse_routing}",
            f"deformable_every={self.deformable_every}", f"flash_attn={_HAS_FLASH_ATTN}",
            f"grad_ckpt={self.gradient_checkpointing}",
        ])
        return (
            f"num_levels={self.num_levels}, num_shifts={self.num_shifts}, "
            f"channels={self.channels}, wavelet_channels={self.eff_wavelet_ch}, "
            f"structure_channels={self.structure_channels}, "
            f"token_dim={self.token_dim}, level_token_dim={self.level_token_dim}, "
            f"num_heads={self.num_heads}, num_layers={self.num_layers}, "
            f"level_radii={self.level_radii}, "
            f"min_shifts={self.min_shifts}, max_shifts={self.num_shifts}, "
            f"pooling={self.pooling!r}, temperature={self.temperature}, "
            f"gumbel_temperature={self.gumbel_temperature:.4f}, "
            f"gumbel_decay={self.gumbel_decay}, "
            f"feature_diversity_threshold={self.feature_diversity_threshold}, "
            f"router_entropy_lambda={self.router_entropy_lambda}, "
            f"{flags}"
        )
