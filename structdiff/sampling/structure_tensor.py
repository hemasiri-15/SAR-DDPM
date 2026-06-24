"""
sampling/structure_tensor.py
============================
A10 / A11 — Structure Tensor + Coherence Conditioning.

Computes per-pixel second-order image structure descriptors from the
current noisy sample xt and returns a StructureTensorMaps dataclass.

Changes from previous version
------------------------------
· cos2theta / sin2theta replace raw orientation as the canonical
  orientation encoding.  The raw angle θ has a π-ambiguity: θ and θ+π
  describe the same edge direction, making it unreliable as a conditioning
  signal.  cos(2θ) and sin(2θ) are π-periodic and unambiguous.

· orientation_confidence = coherence²  (∈ [0,1]).
  Squaring the coherence suppresses weakly-anisotropic regions more
  strongly than the linear weighting used by weighted_orientation.

· eigenvalue_entropy: Shannon entropy of the normalised eigenvalue
  pair (λ₁, λ₂), computed per pixel.  High entropy → isotropic,
  low entropy → strongly directional.

      p₁ = λ₁ / (λ₁ + λ₂ + ε)
      p₂ = λ₂ / (λ₁ + λ₂ + ε)
      H  = − p₁ log p₁ − p₂ log p₂

· scale_consistency: variance of λ₁ across integration scales,
  measuring how much the dominant gradient energy changes with scale.
  Useful for A15 adaptive scale weighting.

Architecture
------------
All convolution kernels (Sobel, Gaussian) are stored via register_buffer()
ensuring zero kernel allocations inside forward().

Orientation formula
-------------------
    θ = 0.5 · atan2(2·J₁₂, J₁₁ − J₂₂)

This gives the angle of the eigenvector for λ₁ (major axis).
Stored as (cos2θ, sin2θ) to avoid π-ambiguity.

Multi-scale integration
-----------------------
One structure tensor per integration sigma; results are fused by weighted
average.  Default sigmas: (1.5, 3.0, 6.0).  scale_consistency uses the
per-scale λ₁ values before fusion.

No numpy, no scipy, no skimage.
All computation is pure PyTorch and compile-safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class StructureTensorMaps:
    """
    All per-pixel structure tensor descriptors.

    Every field has shape [N,1,H,W] and dtype float32.

    Fields
    ------
    lambda1                — major eigenvalue   (edge / oriented strength)
    lambda2                — minor eigenvalue   (≥ 0 by construction)
    anisotropy             — (λ₁−λ₂)/(λ₁+λ₂+ε) ∈ [0,1]
    coherence              — alias for anisotropy (same tensor object)
    orientation            — dominant edge angle ∈ [−π/2, π/2]
                             (retained for backward compatibility)
    cos2theta              — cos(2θ), π-ambiguity-free orientation
    sin2theta              — sin(2θ), π-ambiguity-free orientation
    orientation_confidence — coherence² ∈ [0,1]
    lambda_sum             — λ₁ + λ₂  (total gradient energy)
    lambda_ratio           — λ₁ / (λ₂ + ε)
    weighted_orientation   — coherence · orientation
    eigenvalue_entropy     — −p₁ log p₁ − p₂ log p₂  ∈ [0, log 2]
    scale_consistency      — Var(λ₁ across integration scales)
    """
    lambda1:               torch.Tensor
    lambda2:               torch.Tensor
    anisotropy:            torch.Tensor
    coherence:             torch.Tensor   # alias: same tensor as anisotropy
    orientation:           torch.Tensor
    cos2theta:             torch.Tensor
    sin2theta:             torch.Tensor
    orientation_confidence:torch.Tensor
    lambda_sum:            torch.Tensor
    lambda_ratio:          torch.Tensor
    weighted_orientation:  torch.Tensor
    eigenvalue_entropy:    torch.Tensor
    scale_consistency:     torch.Tensor


# ---------------------------------------------------------------------------
# StructureTensorModule
# ---------------------------------------------------------------------------

class StructureTensorModule(nn.Module):
    """
    Differentiable, buffer-based structure tensor computation.

    Sobel kernels and all Gaussian kernels are allocated once in __init__
    and stored as named buffers.  forward() contains zero tensor allocations
    for kernels.

    Parameters
    ----------
    sigma_gradient : float
        Pre-smoothing sigma applied to the grayscale input before Sobel.
        Reduces noise sensitivity.  Set to 0.0 to skip pre-smoothing.
    integration_sigmas : Sequence[float]
        Integration window sigmas for outer-product averaging.
        One structure tensor is computed per sigma; the results are fused
        by weighted average (equal weights by default).
        Default: (1.5, 3.0, 6.0).
    fusion_weights : Optional[Sequence[float]]
        Per-sigma fusion weights.  None → equal weights.
    eps : float
        Numerical floor added to eigenvalue sums and denominators.
    """

    def __init__(
        self,
        sigma_gradient:     float = 1.0,
        integration_sigmas: Sequence[float] = (1.5, 3.0, 6.0),
        fusion_weights:     Optional[Sequence[float]] = None,
        eps:                float = 1e-8,
    ) -> None:
        super().__init__()
        self.sigma_gradient     = sigma_gradient
        self.integration_sigmas = list(integration_sigmas)
        self.eps                = eps

        # Fusion weights (normalised)
        n = len(integration_sigmas)
        if fusion_weights is None:
            fw = torch.ones(n) / n
        else:
            fw_t = torch.tensor(list(fusion_weights), dtype=torch.float32)
            fw   = fw_t / fw_t.sum()
        self.register_buffer("_fusion_weights", fw)   # [n_scales]

        # ── Sobel kernels  [1,1,3,3] ──────────────────────────────────
        kx = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]],
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1., -2., -1.],
             [ 0.,  0.,  0.],
             [ 1.,  2.,  1.]],
        ).view(1, 1, 3, 3)
        self.register_buffer("_kx", kx)
        self.register_buffer("_ky", ky)

        # ── Pre-smoothing Gaussian (if requested) ─────────────────────
        if sigma_gradient > 0.0:
            kh, kv = self._make_gaussian_1d(sigma_gradient)
            self.register_buffer("_pre_kh", kh)   # [1,1,1,2r+1]
            self.register_buffer("_pre_kv", kv)   # [1,1,2r+1,1]
        else:
            self._pre_kh = None  # type: ignore[assignment]
            self._pre_kv = None  # type: ignore[assignment]

        # ── Integration Gaussians — one (kh, kv) pair per sigma ───────
        for i, sig in enumerate(integration_sigmas):
            kh, kv = self._make_gaussian_1d(sig)
            self.register_buffer(f"_int_kh_{i}", kh)
            self.register_buffer(f"_int_kv_{i}", kv)

    # ------------------------------------------------------------------
    @staticmethod
    def _make_gaussian_1d(
        sigma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build a pair of 1-D separable Gaussian kernels.

        Returns
        -------
        kh : [1,1,1,2r+1]  horizontal kernel
        kv : [1,1,2r+1,1]  vertical kernel
        """
        radius = max(1, int(3.0 * sigma))
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        k = k / k.sum()
        kh = k.view(1, 1, 1, -1)
        kv = k.view(1, 1, -1, 1)
        return kh, kv

    # ------------------------------------------------------------------
    def _gaussian_smooth(
        self,
        x: torch.Tensor,
        kh: torch.Tensor,
        kv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Separable Gaussian smoothing of x: [N,C,H,W] → [N,C,H,W].

        Reflect padding prevents border artefacts on SAR chips.
        Groups = C so each channel is smoothed independently.
        """
        C  = x.shape[1]
        rh = kh.shape[-1] // 2
        rv = kv.shape[-2] // 2
        kh_ = kh.expand(C, 1, 1, -1)
        kv_ = kv.expand(C, 1, -1, 1)
        x   = F.conv2d(F.pad(x, (rh, rh, 0, 0), "reflect"), kh_, groups=C)
        x   = F.conv2d(F.pad(x, (0, 0, rv, rv), "reflect"), kv_, groups=C)
        return x

    # ------------------------------------------------------------------
    def _sobel(self, gray: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Ix and Iy via the registered 3×3 Sobel kernels.

        gray : [N,1,H,W]

        Returns (Ix, Iy) each [N,1,H,W].
        """
        Ix = F.conv2d(F.pad(gray, (1, 1, 1, 1), "reflect"), self._kx)
        Iy = F.conv2d(F.pad(gray, (1, 1, 1, 1), "reflect"), self._ky)
        return Ix, Iy

    # ------------------------------------------------------------------
    def _compute_at_scale(
        self,
        J11: torch.Tensor,
        J22: torch.Tensor,
        J12: torch.Tensor,
        kh: torch.Tensor,
        kv: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,  # lambda1
        torch.Tensor,  # lambda2
        torch.Tensor,  # coherence
        torch.Tensor,  # orientation
        torch.Tensor,  # lambda_ratio
        torch.Tensor,  # cos2theta
        torch.Tensor,  # sin2theta
    ]:
        """
        Integrate outer products with a Gaussian window and compute
        eigenvalues, coherence, orientation, and ambiguity-free encoding.

        Returns
        -------
        (lambda1, lambda2, coherence, orientation,
         lambda_ratio, cos2theta, sin2theta)
        """
        eps = self.eps

        # Integrate outer products
        iJ11 = self._gaussian_smooth(J11, kh, kv)
        iJ22 = self._gaussian_smooth(J22, kh, kv)
        iJ12 = self._gaussian_smooth(J12, kh, kv)

        # Eigenvalues (closed-form 2×2)
        half_trace = 0.5 * (iJ11 + iJ22)
        disc       = ((0.5 * (iJ11 - iJ22)) ** 2 + iJ12 ** 2).clamp(min=0.0).sqrt()

        lam1 = half_trace + disc
        lam2 = (half_trace - disc).clamp(min=0.0)

        # Coherence / anisotropy ∈ [0,1]
        coh = (lam1 - lam2) / (lam1 + lam2 + eps)

        # ── Orientation  ──────────────────────────────────────────────
        # Raw angle ∈ [−π/2, π/2]
        #   θ = 0.5 · atan2(2·J₁₂, J₁₁ − J₂₂)
        diff_diag = (iJ11 - iJ22).clamp(min=eps)
        theta = 0.5 * torch.atan2(2.0 * iJ12, diff_diag)

        # Ambiguity-free encoding via double-angle identity
        #   cos(2θ) = cos²θ − sin²θ  (directly via atan2 result)
        two_theta = 2.0 * theta
        cos2t = torch.cos(two_theta)
        sin2t = torch.sin(two_theta)

        # Eigenvalue ratio
        ratio = lam1 / (lam2 + eps)

        return lam1, lam2, coh, theta, ratio, cos2t, sin2t

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> StructureTensorMaps:
        """
        Compute multi-scale structure tensor descriptors.

        Parameters
        ----------
        x : torch.Tensor  [N,C,H,W]
            Current noisy sample.  Multi-channel inputs are averaged to
            grayscale before gradient computation.

        Returns
        -------
        StructureTensorMaps
            All fields [N,1,H,W] float32.
        """
        gray = x.mean(dim=1, keepdim=True).to(dtype=torch.float32)

        # Pre-smoothing (optional)
        if self._pre_kh is not None:
            gray = self._gaussian_smooth(gray, self._pre_kh, self._pre_kv)

        # Sobel gradients
        Ix, Iy = self._sobel(gray)

        # Outer products (shared across integration sigmas)
        J11 = Ix * Ix
        J22 = Iy * Iy
        J12 = Ix * Iy

        # ── Multi-scale fusion ────────────────────────────────────────
        fw = self._fusion_weights            # [n_scales]
        acc_lam1  = torch.zeros_like(J11)
        acc_lam2  = torch.zeros_like(J11)
        acc_coh   = torch.zeros_like(J11)
        acc_ori   = torch.zeros_like(J11)
        acc_rat   = torch.zeros_like(J11)
        acc_cos2t = torch.zeros_like(J11)
        acc_sin2t = torch.zeros_like(J11)

        # Per-scale λ₁ values for scale_consistency
        per_scale_lam1: List[torch.Tensor] = []

        n_scales = len(self.integration_sigmas)
        for i in range(n_scales):
            kh = getattr(self, f"_int_kh_{i}")
            kv = getattr(self, f"_int_kv_{i}")
            w  = fw[i]

            lam1, lam2, coh, ori, rat, cos2t, sin2t = self._compute_at_scale(
                J11, J22, J12, kh, kv
            )

            acc_lam1  = acc_lam1  + w * lam1
            acc_lam2  = acc_lam2  + w * lam2
            acc_coh   = acc_coh   + w * coh
            acc_ori   = acc_ori   + w * ori
            acc_rat   = acc_rat   + w * rat
            acc_cos2t = acc_cos2t + w * cos2t
            acc_sin2t = acc_sin2t + w * sin2t

            per_scale_lam1.append(lam1.detach())

        # ── Derived quantities ────────────────────────────────────────
        lambda_sum  = acc_lam1 + acc_lam2
        weighted_orientation = acc_coh * acc_ori

        # orientation_confidence = coherence²
        orientation_confidence = acc_coh ** 2

        # eigenvalue_entropy: −p₁ log p₁ − p₂ log p₂
        eps = self.eps
        denom = lambda_sum + eps
        p1 = acc_lam1 / denom
        p2 = acc_lam2 / denom
        # Clamp to avoid log(0); p₁ ∈ (0.5, 1] since λ₁ ≥ λ₂ by construction
        p1c = p1.clamp(min=eps, max=1.0 - eps)
        p2c = p2.clamp(min=eps, max=1.0 - eps)
        eigenvalue_entropy = -(p1c * torch.log(p1c) + p2c * torch.log(p2c))

        # scale_consistency = Var(λ₁ across scales)
        lam1_stack  = torch.stack(per_scale_lam1, dim=0)   # [S, N, 1, H, W]
        lam1_mean   = lam1_stack.mean(dim=0)
        scale_consistency = ((lam1_stack - lam1_mean) ** 2).mean(dim=0)

        return StructureTensorMaps(
            lambda1               = acc_lam1,
            lambda2               = acc_lam2,
            anisotropy            = acc_coh,
            coherence             = acc_coh,          # shared tensor
            orientation           = acc_ori,
            cos2theta             = acc_cos2t,
            sin2theta             = acc_sin2t,
            orientation_confidence= orientation_confidence,
            lambda_sum            = lambda_sum,
            lambda_ratio          = acc_rat,
            weighted_orientation  = weighted_orientation,
            eigenvalue_entropy    = eigenvalue_entropy,
            scale_consistency     = scale_consistency,
        )
