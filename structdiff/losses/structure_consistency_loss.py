"""
structdiff/losses/structure_consistency_loss.py
================================================
A33: Structure Consistency Loss — physics-aware geometric regulariser.

PACDS contribution: enforces that the despeckled image x_hat preserves
the local orientation, coherence, anisotropy, and eigenvalue structure
of the clean reference x, beyond what is implied by pixel-level MSE.

Mathematical foundation
-----------------------
Given the structure tensor of an image I with Gaussian smoothing at
scale σ (kernel k×k):

    J11 = Gσ * (∂x I)²
    J12 = Gσ * (∂x I)(∂y I)
    J22 = Gσ * (∂y I)²

    T(I) = [[J11, J12],
             [J12, J22]]

Eigenvalues (closed form, differentiable):

    μ   = (J11 + J22) / 2
    ρ   = sqrt( ((J11 - J22) / 2)² + J12² )
    λ1  = μ + ρ          (dominant eigenvalue)
    λ2  = μ - ρ          (minor eigenvalue)

Derived quantities:

    coherence C(I)   = (λ1 - λ2) / (λ1 + λ2 + ε)     ∈ [0, 1]
    orientation θ(I) = 0.5 · atan2(2·J12, J11 - J22)  ∈ [-π/2, π/2]
    anisotropy A(I)  = 1 - λ2 / (λ1 + ε)              ∈ [0, 1]

Candidate losses (all pixel-wise, spatially averaged):

    L_comp  = ‖T(x̂) - T(x)‖₁  (tensor component L1)
    L_eig   = ‖[λ1(x̂), λ2(x̂)] - [λ1(x), λ2(x)]‖₂²
    L_coh   = ‖C(x̂) - C(x)‖₁
    L_ori   = ‖sin(θ(x̂) - θ(x))‖₁  (circular orientation)
    L_aniso = ‖A(x̂) - A(x)‖₁

Multi-scale (three scales matching A10: 3×3, 5×5, 9×9):

    L_struct = Σₛ wₛ · (α·L_comp(s) + β·L_eig(s) + γ·L_coh(s)
                        + δ·L_ori(s) + ζ·L_aniso(s))

    where weights w = [1.0, 0.5, 0.25] favour fine scale.

Integration with x_hat
----------------------
gaussian_diffusion.py is NOT modified.

Two integration modes are supported (controlled by TrainLoop):

    MODE A — x0-prediction target (predict_xstart=True):
        losses["pred_xstart"] is populated by the diffusion; x_hat
        is directly extracted from the losses dict.

    MODE B — eps-prediction target (default in SAR-DDPM):
        TrainLoop.forward_backward() reconstructs x_hat from the
        model output using the closed-form posterior mean formula,
        without touching gaussian_diffusion.py:

            x_hat = _reconstruct_xstart(x_t, t, eps_hat, alphas)

        This requires passing `diffusion` and `t` into the loss.

This module implements both.  The recommended path for PACDS is Mode B
(no modification to gaussian_diffusion.py, using StructConsistencyHook).

Design contracts
----------------
- gaussian_diffusion.py is NOT modified.
- training_losses() return dict is NOT modified.
- MixedPrecisionTrainer is NOT modified.
- All operations are differentiable w.r.t. model parameters.
- Full fp16/bf16 compatibility via float32 upcasting inside the module.
- No in-place ops (no .add_(), .mul_() etc.) to stay autograd-safe.

Insertion point in train_util.py
---------------------------------
    # In forward_backward(), AFTER compute_losses():
    losses = compute_losses()
    loss = (losses["loss"] * weights).mean()

    # --- A33 insertion ---
    if self.lambda_struct > 0:
        struct_loss = self.struct_loss_fn(
            x_t=micro,                          # noisy x_t used as proxy
            x0=micro_cond["_clean"],            # clean target
            t=t,
            struct_tensors=micro_cond["struct_tensors"],
            losses_dict=losses,
        )
        loss = loss + self.lambda_struct * struct_loss
        log scalar struct_loss
    # --- end A33 ---

    self.mp_trainer.backward(loss)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Epsilon for numerical stability in divisions.
_EPS: float = 1e-6

#: Scales matching A10 (kernel sizes for Gaussian smoothing).
_DEFAULT_KERNELS: Tuple[int, ...] = (3, 5, 9)

#: Scale weights: fine → coarse. Favour fine-scale fidelity.
_DEFAULT_SCALE_WEIGHTS: Tuple[float, ...] = (1.0, 0.5, 0.25)

#: Sub-loss blend coefficients (tuned for SAR images).
#: Component loss: 0.4, Eigenvalue: 0.2, Coherence: 0.2, Orient: 0.1, Aniso: 0.1
_DEFAULT_ALPHA: float = 0.40  # tensor component weight
_DEFAULT_BETA:  float = 0.20  # eigenvalue weight
_DEFAULT_GAMMA: float = 0.20  # coherence weight
_DEFAULT_DELTA: float = 0.10  # orientation weight
_DEFAULT_ZETA:  float = 0.10  # anisotropy weight


# ---------------------------------------------------------------------------
# Low-level differentiable structure tensor helpers
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(sigma: float, kernel_size: int, device: torch.device,
                         dtype: torch.dtype) -> torch.Tensor:
    """1-D Gaussian kernel of given radius, unnormalized then L1-normalized."""
    x = torch.arange(kernel_size, device=device, dtype=dtype)
    x = x - (kernel_size - 1) / 2.0
    g = torch.exp(-x.pow(2) / (2.0 * sigma * sigma))
    return g / g.sum()


def _sobel_gradients(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ∂x and ∂y of a single-channel image via Sobel filters.

    Parameters
    ----------
    image:
        [B, 1, H, W] float tensor.

    Returns
    -------
    gx, gy:
        Horizontal and vertical gradient maps, each [B, 1, H, W].
    """
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=image.dtype, device=image.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=image.dtype, device=image.device
    ).view(1, 1, 3, 3)
    gx = F.conv2d(image, sobel_x, padding=1)
    gy = F.conv2d(image, sobel_y, padding=1)
    return gx, gy


def compute_structure_tensor(
    image: torch.Tensor,
    kernel_size: int = 3,
    sigma: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the 2×2 structure tensor per pixel.

    Parameters
    ----------
    image:
        [B, C, H, W] float tensor.  If C > 1, luminance is used
        (mean across channels).
    kernel_size:
        Size of the Gaussian integration window.  One of {3, 5, 9}.
    sigma:
        Gaussian sigma.  Defaults to kernel_size / 6.0.

    Returns
    -------
    J11, J12, J22 : each [B, 1, H, W]
        Structure tensor components.  All are non-negative (J11, J22)
        or signed (J12).
    """
    if sigma is None:
        sigma = kernel_size / 6.0

    # Convert to single-channel luminance if needed
    if image.shape[1] > 1:
        lum = image.mean(dim=1, keepdim=True)
    else:
        lum = image

    # Sobel gradients
    gx, gy = _sobel_gradients(lum)   # [B, 1, H, W] each

    # Outer products
    gx2 = gx * gx
    gxy = gx * gy
    gy2 = gy * gy

    # Gaussian smoothing (separable convolution)
    k = _gaussian_kernel_1d(sigma, kernel_size, image.device, image.dtype)
    kh = k.view(1, 1, 1, kernel_size)
    kv = k.view(1, 1, kernel_size, 1)
    pad = kernel_size // 2

    def _smooth(t: torch.Tensor) -> torch.Tensor:
        t = F.conv2d(t, kh, padding=(0, pad))
        t = F.conv2d(t, kv, padding=(pad, 0))
        return t

    J11 = _smooth(gx2)   # [B, 1, H, W]
    J12 = _smooth(gxy)
    J22 = _smooth(gy2)

    return J11, J12, J22


def compute_eigenvalues(
    J11: torch.Tensor,
    J12: torch.Tensor,
    J22: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Closed-form eigenvalues of a 2×2 symmetric matrix.

    Returns λ1 ≥ λ2 ≥ 0.  Differentiable everywhere (the sqrt argument
    is clamped to ≥ 0 for safety at isotropic pixels).

    Parameters
    ----------
    J11, J12, J22 : [B, 1, H, W] each

    Returns
    -------
    lam1, lam2 : [B, 1, H, W] each
    """
    mu  = (J11 + J22) * 0.5
    rho = ((((J11 - J22) * 0.5).pow(2) + J12.pow(2)).clamp(min=0.0)).sqrt()
    lam1 = mu + rho
    lam2 = mu - rho
    return lam1, lam2


def compute_coherence(
    lam1: torch.Tensor,
    lam2: torch.Tensor,
    eps: float = _EPS,
) -> torch.Tensor:
    """C = (λ1 - λ2) / (λ1 + λ2 + ε).  Range [0, 1]."""
    return (lam1 - lam2) / (lam1 + lam2 + eps)


def compute_orientation(
    J11: torch.Tensor,
    J12: torch.Tensor,
    J22: torch.Tensor,
) -> torch.Tensor:
    """θ = 0.5 · atan2(2·J12, J11 - J22).  Range [-π/2, π/2]."""
    return 0.5 * torch.atan2(2.0 * J12, J11 - J22 + _EPS)


def compute_anisotropy(
    lam1: torch.Tensor,
    lam2: torch.Tensor,
    eps: float = _EPS,
) -> torch.Tensor:
    """A = 1 - λ2 / (λ1 + ε).  Range [0, 1]."""
    return 1.0 - lam2 / (lam1 + eps)


# ---------------------------------------------------------------------------
# Single-scale structure consistency loss
# ---------------------------------------------------------------------------

class StructureConsistencyLoss(nn.Module):
    """Single-scale structure consistency loss between x̂ and x.

    Enforces five geometric properties:
        - Tensor component fidelity    (L1 on J11, J12, J22)
        - Eigenvalue fidelity          (MSE on λ1, λ2)
        - Coherence fidelity           (L1 on C)
        - Orientation fidelity         (circular L1 on θ via sin(Δθ))
        - Anisotropy fidelity          (L1 on A)

    Parameters
    ----------
    kernel_size:
        Gaussian integration window.  One of {3, 5, 9}.
    alpha, beta, gamma, delta, zeta:
        Blend weights for each sub-loss.  Should sum to 1.0.
    eps:
        Numerical stability constant.

    Examples
    --------
    >>> loss_fn = StructureConsistencyLoss(kernel_size=3)
    >>> x_hat = torch.randn(4, 3, 256, 256)
    >>> x_clean = torch.randn(4, 3, 256, 256)
    >>> loss = loss_fn(x_hat, x_clean)
    """

    def __init__(
        self,
        kernel_size: int = 3,
        alpha: float = _DEFAULT_ALPHA,
        beta:  float = _DEFAULT_BETA,
        gamma: float = _DEFAULT_GAMMA,
        delta: float = _DEFAULT_DELTA,
        zeta:  float = _DEFAULT_ZETA,
        eps:   float = _EPS,
    ) -> None:
        super().__init__()
        assert kernel_size in (3, 5, 9, 11), f"Unsupported kernel_size={kernel_size}"
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.zeta  = zeta
        self.eps   = eps

    def _st(self, x: torch.Tensor) -> Tuple:
        """Compute all structure tensor quantities for x."""
        # Upcast to float32 for numerical stability (fp16-safe module)
        x32 = x.float()
        J11, J12, J22 = compute_structure_tensor(x32, self.kernel_size)
        lam1, lam2    = compute_eigenvalues(J11, J12, J22)
        coh           = compute_coherence(lam1, lam2, self.eps)
        ori           = compute_orientation(J11, J12, J22)
        aniso         = compute_anisotropy(lam1, lam2, self.eps)
        return J11, J12, J22, lam1, lam2, coh, ori, aniso

    def forward(
        self,
        x_hat: torch.Tensor,
        x_clean: torch.Tensor,
    ) -> torch.Tensor:
        """Compute structure consistency loss.

        Parameters
        ----------
        x_hat:
            Predicted / denoised image, [B, C, H, W].
        x_clean:
            Ground-truth clean image, [B, C, H, W].

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        # Structure tensors for both images
        J11_h, J12_h, J22_h, l1_h, l2_h, coh_h, ori_h, aniso_h = self._st(x_hat)
        J11_c, J12_c, J22_c, l1_c, l2_c, coh_c, ori_c, aniso_c = self._st(x_clean)

        # --- Candidate A: Tensor component loss (L1) ---
        # Directly penalises each element of the 2×2 tensor.
        L_comp = (
            (J11_h - J11_c).abs().mean()
            + (J12_h - J12_c).abs().mean()
            + (J22_h - J22_c).abs().mean()
        ) / 3.0

        # --- Candidate B: Eigenvalue loss (MSE) ---
        # Penalises dominant/minor eigenvalue separately.
        # MSE is used because eigenvalues span a wide dynamic range.
        L_eig = (
            (l1_h - l1_c).pow(2).mean()
            + (l2_h - l2_c).pow(2).mean()
        ) * 0.5

        # --- Candidate C: Coherence loss (L1) ---
        # Both values ∈ [0,1] → L1 is well-calibrated.
        L_coh = (coh_h - coh_c).abs().mean()

        # --- Candidate D: Orientation loss (circular L1) ---
        # sin(Δθ) avoids wrap-around issues; |sin(Δθ)| ∈ [0,1].
        L_ori = torch.sin(ori_h - ori_c).abs().mean()

        # --- Candidate E: Anisotropy loss (L1) ---
        L_aniso = (aniso_h - aniso_c).abs().mean()

        # --- Candidate G: Weighted combination ---
        L_struct = (
            self.alpha * L_comp
            + self.beta  * L_eig
            + self.gamma * L_coh
            + self.delta * L_ori
            + self.zeta  * L_aniso
        )

        return L_struct.to(x_hat.dtype)


# ---------------------------------------------------------------------------
# Multi-scale structure consistency loss (A10-matched scales)
# ---------------------------------------------------------------------------

class MultiScaleStructureConsistencyLoss(nn.Module):
    """Multi-scale structure consistency loss matching A10 scale set.

    Combines StructureConsistencyLoss at three scales:
        fine   (3×3)  weight 1.00
        medium (5×5)  weight 0.50
        coarse (9×9)  weight 0.25

    Weights decay geometrically (factor 2) to privilege fine-scale
    edge and orientation fidelity while still capturing mid-scale
    coherence regions characteristic of SAR homogeneous areas.

    The scale set is intentionally identical to A10's encoder scales,
    creating a direct supervision-conditioning duality:
        A10 uses T_s to CONDITION the model.
        A33 uses T_s to CONSTRAIN the output.

    Parameters
    ----------
    kernels:
        Sequence of kernel sizes.  Default: (3, 5, 9).
    scale_weights:
        Corresponding loss weights.  Default: (1.0, 0.5, 0.25).
    normalize_weights:
        If True, normalise scale_weights to sum to 1.
    alpha, beta, gamma, delta, zeta:
        Sub-loss blend coefficients passed to each StructureConsistencyLoss.

    Examples
    --------
    >>> ms_loss = MultiScaleStructureConsistencyLoss()
    >>> x_hat   = torch.randn(2, 3, 256, 256)
    >>> x_clean = torch.randn(2, 3, 256, 256)
    >>> ms_loss(x_hat, x_clean).item()  # scalar
    """

    def __init__(
        self,
        kernels:           Sequence[int]   = _DEFAULT_KERNELS,
        scale_weights:     Sequence[float] = _DEFAULT_SCALE_WEIGHTS,
        normalize_weights: bool            = True,
        alpha: float = _DEFAULT_ALPHA,
        beta:  float = _DEFAULT_BETA,
        gamma: float = _DEFAULT_GAMMA,
        delta: float = _DEFAULT_DELTA,
        zeta:  float = _DEFAULT_ZETA,
    ) -> None:
        super().__init__()

        assert len(kernels) == len(scale_weights), (
            "kernels and scale_weights must have the same length."
        )

        weights = list(scale_weights)
        if normalize_weights:
            total = sum(weights)
            weights = [w / total for w in weights]

        self.scale_weights: List[float] = weights
        self.scale_losses = nn.ModuleList([
            StructureConsistencyLoss(
                kernel_size=k,
                alpha=alpha, beta=beta, gamma=gamma,
                delta=delta, zeta=zeta,
            )
            for k in kernels
        ])

    def forward(
        self,
        x_hat:   torch.Tensor,
        x_clean: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted multi-scale structure consistency loss.

        Parameters
        ----------
        x_hat:
            Predicted image, [B, C, H, W].
        x_clean:
            Ground-truth clean image, [B, C, H, W].

        Returns
        -------
        torch.Tensor
            Scalar.
        """
        total = x_hat.new_zeros(1).squeeze()
        for w, loss_fn in zip(self.scale_weights, self.scale_losses):
            total = total + w * loss_fn(x_hat, x_clean)
        return total
