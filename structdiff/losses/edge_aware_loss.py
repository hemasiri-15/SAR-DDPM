"""
structdiff/losses/edge_aware_loss.py
=====================================
A5: Edge-Aware Loss — explicit edge-structure preservation for SAR despeckling.

Mathematical Foundation
-----------------------
Standard diffusion losses (MSE on predicted noise ε̂) operate in a
pixel-wise, isotropic fashion.  They do not penalise the blurring of
high-frequency spatial structure — precisely the content that defines
edges, roads, building outlines, coastlines, and other SAR-relevant
targets.

We define the Edge-Aware Loss via Sobel gradient operators.

Let x_gt ∈ ℝ^{B×C×H×W} be the ground-truth clean image and
    x_pred ∈ ℝ^{B×C×H×W} be the predicted x_0 (reconstructed from ε̂).

Horizontal and vertical Sobel operators Kx, Ky ∈ ℝ^{3×3}:

    Kx = [[-1,  0,  1],          Ky = [[-1, -2, -1],
           [-2,  0,  2],                [ 0,  0,  0],
           [-1,  0,  1]]                [ 1,  2,  1]]

Gradient components:
    Gx(x) = Kx ⊛ x      (horizontal, ∂x)
    Gy(x) = Ky ⊛ x      (vertical,   ∂y)

Gradient magnitude (L2 norm of the gradient vector):
    |G(x)| = sqrt( Gx(x)² + Gy(x)² + ε )

where ε > 0 avoids the sqrt singularity at zero (important for
autograd stability).

Sub-loss terms:

    (1) Magnitude loss — penalises mismatch in edge strength:
        L_mag = E[ | |G(x_pred)| - |G(x_gt)| | ]          (L1)

    (2) Directional loss — penalises mismatch in horizontal and
        vertical gradient components independently:
        L_dir = E[ |Gx(x_pred) - Gx(x_gt)| ]
              + E[ |Gy(x_pred) - Gy(x_gt)| ]
        (averaged over the two terms)

    Combined:
        L_edge = α · L_mag + β · L_dir

    where α + β = 1 (α=0.6, β=0.4 default; tunable).

Combined training objective:
    L_total = L_diffusion
            + λ_struct · L_struct          (A33)
            + λ_edge   · L_edge            (A5)

Default λ_edge = 0.05 (lower than λ_struct=0.1 because Sobel
magnitudes are naturally larger than structure tensor components).

Why Sobel over alternatives?
-----------------------------
• Scharr:     Rotationally more accurate but larger coefficients
              amplify speckle noise response — bad for SAR.
• Prewitt:    Uniform weights, lower noise discrimination than Sobel.
• Laplacian:  Second-order derivative; highly sensitive to speckle
              noise.  A multiplicative noise process at σ² already
              produces large second derivatives; Laplacian would
              destabilise training.
• Canny:      Not differentiable (requires NMS, thresholding).
• Sobel:      3×3, first-order, diagonal weighting provides a
              balance of noise robustness and directional sensitivity.
              Widely accepted in image restoration literature.

Gradient flow
-------------
    L_edge
        ↓
    |G(x_pred)|  ← sqrt(Gx² + Gy² + ε)    (differentiable)
        ↓
    x_pred = x0_hat = (x_t - √(1-ᾱ)·ε̂) / √ᾱ    (pure arithmetic)
        ↓
    ε̂   = UNet(x_t, t, **cond)              ← gradient-bearing
        ↓
    UNet parameters                          ✓

Design contracts
----------------
- gaussian_diffusion.py is NOT modified.
- Reuses x0_hat already computed for A33 — zero extra UNet forward passes.
- Full fp16/bf16 compatibility via float32 upcasting inside the module.
- No in-place operations (autograd-safe).
- Sobel kernels registered as buffers (device-portable, not trained).
- Channel-agnostic: operates per-channel, averages across channels.
- AMP-safe: internal computation in float32, output cast to input dtype.

Integration point in train_util.py
------------------------------------
    # AFTER A33 struct_loss block, BEFORE mp_trainer.backward(loss):

    if self.lambda_edge > 0.0:
        edge_loss = self.edge_loss_fn(
            x_pred=x0_hat,               # in autograd graph
            x_gt=micro.float().detach(), # GT detached
        )
        loss = loss + self.lambda_edge * edge_loss
        logger.logkv_mean("edge_loss", edge_loss.item())
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Public constant — imported by train_util.py
# ---------------------------------------------------------------------------

#: Default λ_edge scaling factor for the combined loss.
#: Set to 0.0 in train_util.py to disable A5 entirely.
DEFAULT_LAMBDA_EDGE: float = 0.05


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

#: Epsilon inside sqrt( Gx² + Gy² + ε ) to prevent division-by-zero
#: and the zero-gradient singularity in autograd.
_GRAD_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# Sobel kernel definitions
# ---------------------------------------------------------------------------

def _make_sobel_kernels() -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the 3×3 Sobel kernels Kx (horizontal) and Ky (vertical).

    Kx detects ∂/∂x (left–right transitions).
    Ky detects ∂/∂y (top–bottom transitions).

    Both shaped [1, 1, 3, 3] for direct use with F.conv2d on a
    single-channel input.  Multi-channel inputs are handled by the
    calling code (channel loop / grouped convolution).

    Returns
    -------
    kx, ky : torch.Tensor
        Float32 tensors, shape [1, 1, 3, 3].
    """
    kx = torch.tensor(
        [[-1.0,  0.0,  1.0],
         [-2.0,  0.0,  2.0],
         [-1.0,  0.0,  1.0]],
        dtype=torch.float32,
    ).reshape(1, 1, 3, 3)

    ky = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [ 0.0,  0.0,  0.0],
         [ 1.0,  2.0,  1.0]],
        dtype=torch.float32,
    ).reshape(1, 1, 3, 3)

    return kx, ky


# ---------------------------------------------------------------------------
# Low-level gradient computation (functional, no module state)
# ---------------------------------------------------------------------------

def _sobel_gradients_multichannel(
    image: torch.Tensor,
    kx: torch.Tensor,
    ky: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Sobel gradients for a multi-channel image.

    Uses depthwise (grouped) convolution so every channel is filtered
    independently.  This is equivalent to applying the Sobel operator
    separately to each channel and concatenating, but is faster.

    Parameters
    ----------
    image : torch.Tensor
        [B, C, H, W] float32 tensor.
    kx, ky : torch.Tensor
        Sobel kernels [1, 1, 3, 3] float32, on the same device as image.

    Returns
    -------
    gx, gy : torch.Tensor
        Gradient maps, each [B, C, H, W] float32.
    """
    B, C, H, W = image.shape

    # Tile kernels for depthwise convolution: [C, 1, 3, 3]
    kx_c = kx.expand(C, 1, 3, 3)
    ky_c = ky.expand(C, 1, 3, 3)

    # Depthwise conv: groups=C means each input channel gets its own filter.
    # padding=1 preserves spatial resolution (same-padding for 3×3 kernel).
    gx = F.conv2d(image, kx_c, padding=1, groups=C)
    gy = F.conv2d(image, ky_c, padding=1, groups=C)

    return gx, gy


def _gradient_magnitude(
    gx: torch.Tensor,
    gy: torch.Tensor,
    eps: float = _GRAD_EPS,
) -> torch.Tensor:
    """Compute the L2 gradient magnitude |G| = sqrt(Gx² + Gy² + ε).

    The additive ε keeps the sqrt argument strictly positive, ensuring
    a well-defined (non-infinite) gradient through the sqrt at flat
    image regions.

    Parameters
    ----------
    gx, gy : torch.Tensor
        Gradient components, any shape.
    eps : float
        Numerical stability constant.

    Returns
    -------
    torch.Tensor
        Gradient magnitude, same shape as inputs.
    """
    return torch.sqrt(gx.pow(2) + gy.pow(2) + eps)


# ---------------------------------------------------------------------------
# EdgeAwareLoss module
# ---------------------------------------------------------------------------

class EdgeAwareLoss(nn.Module):
    """A5: Edge-Aware Loss for SAR image despeckling.

    Penalises mismatch in Sobel gradient structure between the predicted
    x_0 and the ground-truth clean image.

    Two complementary sub-losses are combined:

    1.  **Magnitude loss** (L_mag):
            L1 distance between per-pixel gradient magnitudes.
            Ensures that edge strength (how strong a transition is) is
            correctly reproduced.

    2.  **Directional loss** (L_dir):
            L1 distance between horizontal gradient maps (Gx) and
            between vertical gradient maps (Gy), averaged.
            Ensures that the spatial direction of edges is reproduced,
            not just their magnitude.

    Combined:
        L_edge = α · L_mag + β · L_dir

    Parameters
    ----------
    alpha : float
        Weight on the gradient-magnitude L1 term.  Default: 0.6.
    beta : float
        Weight on the directional (Gx/Gy) L1 term.  Default: 0.4.
    eps : float
        Numerical stability constant inside sqrt.  Default: 1e-6.

    Notes
    -----
    - Sobel kernels are registered as non-trainable buffers.
    - Input tensors are upcasted to float32 internally; output is cast
      back to the input dtype of x_pred for AMP compatibility.
    - No in-place operations are used, preserving the autograd graph
      through x_pred → eps_hat → UNet parameters.
    - The module is channel-agnostic: it works with grayscale (C=1),
      single-polarisation SAR (C=1), dual-pol (C=2), or RGB (C=3).

    Examples
    --------
    >>> loss_fn = EdgeAwareLoss(alpha=0.6, beta=0.4)
    >>> x_pred  = torch.randn(4, 1, 256, 256, requires_grad=True)
    >>> x_gt    = torch.randn(4, 1, 256, 256)
    >>> loss    = loss_fn(x_pred, x_gt)
    >>> loss.backward()
    >>> assert x_pred.grad is not None
    """

    def __init__(
        self,
        alpha: float = 0.6,
        beta:  float = 0.4,
        eps:   float = _GRAD_EPS,
    ) -> None:
        super().__init__()

        if not (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0):
            raise ValueError(
                f"alpha and beta must be in [0, 1]; got alpha={alpha}, beta={beta}"
            )
        if abs(alpha + beta - 1.0) > 1e-5:
            raise ValueError(
                f"alpha + beta should equal 1.0; got {alpha + beta:.6f}. "
                "Adjust values or set normalize=True."
            )

        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps

        # Register Sobel kernels as non-trainable buffers.
        # Buffers are automatically moved to the correct device when
        # .to(device) or .cuda() is called on the module, and are
        # saved/loaded with state_dict() (though they carry no gradients).
        kx, ky = _make_sobel_kernels()
        self.register_buffer("kx", kx)   # [1, 1, 3, 3]
        self.register_buffer("ky", ky)   # [1, 1, 3, 3]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gradients(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Gx, Gy, and |G| for x in float32.

        Parameters
        ----------
        x : torch.Tensor
            [B, C, H, W] float32 tensor.

        Returns
        -------
        gx, gy, mag : torch.Tensor
            Each [B, C, H, W] float32.
        """
        gx, gy = _sobel_gradients_multichannel(x, self.kx, self.ky)
        mag    = _gradient_magnitude(gx, gy, self.eps)
        return gx, gy, mag

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_pred: torch.Tensor,
        x_gt:   torch.Tensor,
    ) -> torch.Tensor:
        """Compute the edge-aware loss between x_pred and x_gt.

        Parameters
        ----------
        x_pred : torch.Tensor
            Predicted x_0 reconstructed from eps_hat, [B, C, H, W].
            Must be connected to the autograd graph (NOT detached).
            Typically: x0_hat from reconstruct_x0().
        x_gt : torch.Tensor
            Ground-truth clean image, [B, C, H, W].
            Should be detached (micro.float().detach()) so that gradients
            only flow through x_pred.

        Returns
        -------
        torch.Tensor
            Scalar edge loss.  Gradient: loss → x_pred → eps_hat → UNet.

        Raises
        ------
        ValueError
            If x_pred and x_gt have different shapes.
        """
        if x_pred.shape != x_gt.shape:
            raise ValueError(
                f"x_pred and x_gt must have the same shape; "
                f"got {x_pred.shape} vs {x_gt.shape}"
            )

        # Store dtype for output casting (AMP compatibility).
        out_dtype = x_pred.dtype

        # Upcast to float32 for numerical stability.
        # This mirrors the pattern in StructureConsistencyLoss.
        x_p = x_pred.float()
        x_g = x_gt.float()

        # Compute Sobel gradients for both images.
        gx_p, gy_p, mag_p = self._gradients(x_p)
        gx_g, gy_g, mag_g = self._gradients(x_g)

        # ── Sub-loss 1: Gradient magnitude L1 ─────────────────────────
        # Penalises mismatched edge *strength*.
        # L1 is preferred over L2 here because SAR edges can span a very
        # large dynamic range; L1 avoids the quadratic blow-up on
        # high-contrast targets (buildings, ships) while still giving
        # a meaningful signal on low-contrast coastlines.
        L_mag = (mag_p - mag_g).abs().mean()

        # ── Sub-loss 2: Directional L1 ────────────────────────────────
        # Penalises mismatched edge *direction/orientation* by comparing
        # raw Gx and Gy component maps.
        # This is complementary to L_mag: a blurred edge can have the
        # same gradient direction but smaller magnitude; a displaced edge
        # can have the same magnitude but different directional pattern.
        L_dir = (
            (gx_p - gx_g).abs().mean()
            + (gy_p - gy_g).abs().mean()
        ) * 0.5

        # ── Weighted combination ───────────────────────────────────────
        L_edge = self.alpha * L_mag + self.beta * L_dir

        # Cast back to the input dtype for AMP compatibility.
        return L_edge.to(out_dtype)


# ---------------------------------------------------------------------------
# Module-level convenience: expose gradient computation as a standalone fn
# ---------------------------------------------------------------------------

def compute_sobel_gradients(
    image: torch.Tensor,
    eps:   float = _GRAD_EPS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Functional interface: compute Sobel Gx, Gy, and |G| for any image.

    Kernels are created on-the-fly on the same device as `image`.
    Use :class:`EdgeAwareLoss` for training (kernels registered as buffers).
    This function is provided for evaluation scripts, the Edge Preservation
    Index metric, and ablation analysis.

    Parameters
    ----------
    image : torch.Tensor
        [B, C, H, W] or [C, H, W] float tensor.
    eps : float
        Sqrt stability constant.

    Returns
    -------
    gx, gy, magnitude : torch.Tensor
        Gradient components and magnitude.  Same shape as input.

    Examples
    --------
    >>> img = torch.randn(2, 1, 128, 128)
    >>> gx, gy, mag = compute_sobel_gradients(img)
    >>> assert gx.shape == img.shape
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    kx, ky = _make_sobel_kernels()
    kx = kx.to(image.device, dtype=torch.float32)
    ky = ky.to(image.device, dtype=torch.float32)

    x32 = image.float()
    gx, gy = _sobel_gradients_multichannel(x32, kx, ky)
    mag    = _gradient_magnitude(gx, gy, eps)

    if squeeze_back:
        gx, gy, mag = gx.squeeze(0), gy.squeeze(0), mag.squeeze(0)

    return gx, gy, mag


def edge_preservation_index(
    x_pred: torch.Tensor,
    x_gt:   torch.Tensor,
    eps:    float = _GRAD_EPS,
) -> float:
    """Compute the Edge Preservation Index (EPI) for evaluation.

    EPI measures how well the edge structure of x_pred matches x_gt.
    It is the Pearson correlation between the gradient magnitudes of
    the two images, averaged over batch and channel dimensions.

    EPI ∈ [−1, 1]; values close to 1 indicate excellent edge preservation.

    This metric is used in the ablation experiments (A4 vs A4+A5).

    Parameters
    ----------
    x_pred, x_gt : torch.Tensor
        [B, C, H, W] float tensors.  Detached; no gradient tracking needed.
    eps : float
        Stability constant.

    Returns
    -------
    float
        Mean EPI across the batch.
    """
    with torch.no_grad():
        _, _, mag_p = compute_sobel_gradients(x_pred, eps)
        _, _, mag_g = compute_sobel_gradients(x_gt,   eps)

        # Flatten spatial dims: [B, C, H*W]
        B, C = mag_p.shape[:2]
        mp = mag_p.reshape(B, C, -1).float()
        mg = mag_g.reshape(B, C, -1).float()

        # Pearson correlation per (batch item, channel)
        mp_mean = mp.mean(dim=-1, keepdim=True)
        mg_mean = mg.mean(dim=-1, keepdim=True)
        mp_c = mp - mp_mean
        mg_c = mg - mg_mean

        num   = (mp_c * mg_c).sum(dim=-1)
        denom = (
            mp_c.pow(2).sum(dim=-1).sqrt()
            * mg_c.pow(2).sum(dim=-1).sqrt()
            + eps
        )
        epi = (num / denom).mean().item()

    return epi
