"""
structdiff/losses/eps_intercept_hook.py
========================================
A33 — EpsInterceptHook  (final, production version)

PURPOSE
-------
Captures (x_t, t, eps_hat) from the single UNet forward pass that
training_losses() already runs, so StructureConsistencyLoss can use the
SAME noise sample, the SAME x_t, and the SAME eps_hat as the diffusion
loss — with zero extra compute.

PROBLEM WITH THE PREVIOUS APPROACH
------------------------------------
The previous StructConsistencyHook called q_sample() and model() a
second time inside compute_struct_loss():

    Pass 1  (inside training_losses):   noise=ε₁,  x_t(ε₁),  eps(ε₁)
    Pass 2  (inside compute_struct_loss): noise=ε₂,  x_t(ε₂),  eps(ε₂)

This caused:
  • 2× UNet compute per step
  • Diffusion loss and structure loss trained on different noise samples
  • Inconsistent gradient signals → optimization variance

THIS FILE'S APPROACH
--------------------
    hook = EpsInterceptHook(self.ddp_model, learn_sigma=self.learn_sigma)

    # Pass hook instead of ddp_model — training_losses calls hook(x_t, t)
    # which runs the real UNet and stores x_t, t, eps_hat.
    losses = self.diffusion.training_losses(hook, micro, t, ...)

    # Now reconstruct x0_hat using stored tensors — no forward pass.
    x0_hat = reconstruct_x0(
        hook.last_x_t, hook.last_t, hook.last_eps_hat,
        self.diffusion.alphas_cumprod,
    )
    struct_loss = self.struct_loss_fn(x_hat=x0_hat, x_clean=micro.detach())

Result:
  • +0 UNet forward passes
  • ε₁ used by both diffusion loss and structure loss  (consistent)
  • eps_hat is a shared autograd node in both loss graphs

GRADIENT PATH
-------------
    struct_loss
        ↓
    x0_hat = (x_t - √(1−ᾱ)·eps_hat) / √ᾱ      ← pure arithmetic
        ↓
    eps_hat = hook._model(x_t, t, **cond)        ← UNet, has gradient
        ↓
    UNet parameters                               ✓

DDP COMPATIBILITY
-----------------
__getattr__ forwards every attribute lookup to the real DDP object, so
  ddp_model.no_sync(), .module, .device_ids, .parameters()
all work with zero changes to the calling code.
The existing no_sync() context in forward_backward is unaffected.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class EpsInterceptHook:
    """Transparent wrapper that captures (x_t, t, eps_hat) during forward.

    Parameters
    ----------
    model:
        self.ddp_model — the DDP-wrapped or bare UNet.
    learn_sigma:
        True when UNet outputs 2C channels [eps | log_variance].
        Hook slices channels [0:C] as eps_hat.
    """

    def __init__(self, model: nn.Module, learn_sigma: bool = True) -> None:
        self._model      = model
        self.learn_sigma = learn_sigma

        # Populated after __call__; None before.
        self.last_x_t:     Optional[torch.Tensor] = None
        self.last_t:       Optional[torch.Tensor] = None
        self.last_eps_hat: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Called by training_losses() as:  model(x_t, t, **model_kwargs)
    # ------------------------------------------------------------------

    def __call__(
        self,
        x_t: torch.Tensor,
        t:   torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Run the real UNet forward and capture its inputs/outputs.

        Returns the full model output unchanged so training_losses()
        computes its MSE / VLB loss exactly as if ddp_model were passed.
        """
        # ── Real forward pass — autograd graph intact ──────────────────
        out = self._model(x_t, t, **kwargs)

        # ── Capture inputs ─────────────────────────────────────────────
        # These are the same x_t and t that training_losses uses for the
        # diffusion loss — no noise mismatch possible.
        self.last_x_t = x_t
        self.last_t   = t

        # ── Slice eps from [eps | log_var] ─────────────────────────────
        if self.learn_sigma and out.shape[1] == 2 * x_t.shape[1]:
            eps_hat = out[:, : x_t.shape[1]]
        else:
            eps_hat = out

        # eps_hat is still connected to the autograd graph.
        # Gradients from struct_loss will flow through it to the UNet.
        self.last_eps_hat = eps_hat

        # ── Return full output — training_losses must see it unchanged ──
        return out

    # ------------------------------------------------------------------
    # Transparent attribute forwarding
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        # Only invoked when normal attribute lookup on self fails.
        # Forwards .no_sync(), .module, .device_ids, .parameters(), etc.
        return getattr(self._model, name)


# ---------------------------------------------------------------------------
# x0 reconstruction — pure arithmetic, no model call, fully differentiable
# ---------------------------------------------------------------------------

def reconstruct_x0(
    x_t:            torch.Tensor,
    t:              torch.Tensor,
    eps_hat:        torch.Tensor,
    alphas_cumprod: torch.Tensor,
    eps_stable:     float = 1e-6,
) -> torch.Tensor:
    """Reconstruct predicted x_0 from predicted noise eps_hat.

    Identical to guided-diffusion's internal _predict_xstart_from_eps().
    Reproduced here so gaussian_diffusion.py is never modified.

    DDPM forward process:
        x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps

    Inversion:
        x_0_hat = (x_t - sqrt(1 - abar_t) * eps_hat) / sqrt(abar_t)

    Parameters
    ----------
    x_t:
        Noisy image at timestep t, [B, C, H, W].
        Gradient does not need to flow through x_t.
    t:
        Integer timestep indices, [B].
    eps_hat:
        Predicted noise from UNet, [B, C, H, W].
        Must be in the autograd graph (captured by EpsInterceptHook).
    alphas_cumprod:
        ᾱ schedule, [T]. Read from diffusion.alphas_cumprod (not modified).
    eps_stable:
        Small divisor offset for numerical stability.

    Returns
    -------
    torch.Tensor
        x_0_hat [B, C, H, W].  Gradient: x_0_hat → eps_hat → UNet.
    """
    alphas_cumprod = torch.as_tensor(
        alphas_cumprod,
        device=x_t.device,
        dtype=x_t.dtype,
    )

    # Timesteps must be integer indices.
    t = t.to(torch.long)

    sqrt_ab   = alphas_cumprod[t].sqrt()          # [B]
    sqrt_1mab = (1.0 - alphas_cumprod[t]).sqrt()  # [B]

    # Broadcast [B] → [B, 1, 1, 1]
    def _b(v: torch.Tensor) -> torch.Tensor:
        for _ in range(x_t.ndim - 1):
            v = v.unsqueeze(-1)
        return v

    return (x_t - _b(sqrt_1mab) * eps_hat) / (_b(sqrt_ab) + eps_stable)
