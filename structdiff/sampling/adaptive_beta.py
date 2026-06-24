"""
sampling/adaptive_beta.py
=========================
A8 — Adaptive Spatial Beta Controller.

Produces a spatial beta_scale_map ∈ [min_beta, max_beta] per timestep.
The map drives spatially-variable stochasticity during DDIM reverse steps:

    β_scale_map(x,y) large  →  homogeneous region     →  more noise allowed
    β_scale_map(x,y) small  →  edge / oriented region →  preserve structure

Mathematical formulation
------------------------
    raw(x,y) = +w_enl        · enl_norm(x,y)
               +w_entropy    · entropy_norm(x,y)
               −w_edge       · edge_norm(x,y)
               −w_coherence  · coherence_norm(x,y)
               −w_lambda1    · lambda1_norm(x,y)   ← negative: edges → preserve
               −w_lambda2    · lambda2_norm(x,y)
               −w_anisotropy · anisotropy_norm(x,y)

    raw_smooth = Gaussian_σ(raw)        # optional; kernel stored as buffer

    beta_scale_map = sigmoid(raw_smooth / temperature)
                   × (max_beta − min_beta) + min_beta

Normalisation
-------------
Each input map is standardised with a fast mean±k·std scheme rather than
percentile (torch.quantile) to avoid the O(N log N) sort cost inside the
hot sampling loop.  The result is clipped to [0, 1].

Orientation map
---------------
Not yet included in the formula but wired through the state; AdaptiveBeta
accepts orientation_map as a future slot (weight = 0 until A26r).

Integration in SamplingController
----------------------------------
    ctrl     = AdaptiveBetaController(config)
    beta_map = ctrl.compute(state, device)    → [N,1,H,W]
    eta_eff  = ctrl.compute_dynamic_eta(beta_map, t, T)   → float ∈ [min_eta, max_eta]
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SamplingConfig
from .state  import SamplingState


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _std_normalize(
    x: torch.Tensor,
    k: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Fast robust normalisation via mean ± k·std clipping.

    Each sample in the batch is normalised independently.

    Steps
    -----
    1. Compute per-batch μ and σ over all spatial positions.
    2. Clip x to [μ − k·σ, μ + k·σ] to suppress outlier influence.
    3. Linearly rescale the clipped range to [0, 1].

    Parameters
    ----------
    x   : [N,1,H,W] float32
    k   : number of standard deviations defining the clip window
    eps : numerical floor to avoid division by zero in flat regions

    Returns
    -------
    [N,1,H,W] float32 ∈ [0, 1]

    Notes
    -----
    This replaces torch.quantile() (O(N log N) sort) with an O(N) mean/std
    computation, giving a material speed-up on large feature maps without
    meaningful accuracy loss for SAR statistics.
    """
    flat = x.reshape(x.shape[0], -1)           # [N, H*W]
    mu   = flat.mean(dim=1).view(-1, 1, 1, 1)
    sig  = flat.std(dim=1).view(-1, 1, 1, 1).clamp(min=eps)
    lo   = mu - k * sig
    hi   = mu + k * sig
    return ((x - lo) / (hi - lo + eps)).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Gaussian smoothing module (kernels registered as buffers)
# ---------------------------------------------------------------------------

class _GaussianSmoother(nn.Module):
    """
    Separable Gaussian smoothing.

    Kernels are stored via register_buffer() so they follow the module's
    device (correct under torch.compile, AMP, DDP) and are never
    re-allocated inside forward().

    Parameters
    ----------
    sigma : float
        Gaussian standard deviation.
    """

    def __init__(self, sigma: float = 1.5) -> None:
        super().__init__()
        radius = max(1, int(3.0 * sigma))
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        k = k / k.sum()
        self.radius = radius
        # [2r+1] — stored on whichever device the module is moved to
        self.register_buffer("kernel", k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply separable Gaussian smoothing.

        Parameters
        ----------
        x : [N,C,H,W]

        Returns
        -------
        [N,C,H,W]  — same shape, reflect-padded to avoid border artefacts.
        """
        k  = self.kernel                              # [2r+1]
        C  = x.shape[1]
        r  = self.radius
        kh = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
        kv = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
        x  = F.conv2d(F.pad(x, (r, r, 0, 0), "reflect"), kh, groups=C)
        x  = F.conv2d(F.pad(x, (0, 0, r, r), "reflect"), kv, groups=C)
        return x


# ---------------------------------------------------------------------------
# AdaptiveBetaController
# ---------------------------------------------------------------------------

class AdaptiveBetaController(nn.Module):
    """
    Computes a spatial beta_scale_map from image-statistics maps.

    Parameters
    ----------
    config : SamplingConfig
        Source of all weight and range hyperparameters.
    smooth_sigma : float
        Gaussian smoothing sigma applied to the raw logit map before
        sigmoid.  Set to 0.0 to disable smoothing entirely.

    Weights (from config)
    ---------------------
    enl_weight        — positive (high ENL = homogeneous = more noise OK)
    entropy_weight    — positive (high entropy = speckled = more noise OK)
    edge_weight       — negative (strong edge = preserve)
    coherence_weight  — negative (oriented structure = preserve)
    lambda1_weight    — negative (strong λ₁ = edge present = preserve)
    lambda2_weight    — negative (large λ₂ = structured noise = preserve)
    anisotropy_weight — negative (anisotropic = edge/line = preserve)

    The sign choices ensure that beta_scale_map is large in flat/speckled
    regions and small near edges and oriented structures.

    Usage
    -----
    ctrl     = AdaptiveBetaController(config)
    beta_map = ctrl.compute(state, device)            # [N,1,H,W]
    eta_eff  = ctrl.compute_dynamic_eta(beta_map, t, T)   # float
    """

    def __init__(
        self,
        config: SamplingConfig,
        smooth_sigma: float = 1.5,
    ) -> None:
        super().__init__()
        self.config       = config
        self.smooth_sigma = smooth_sigma

        # Smoother constructed once; kernel lives as a buffer.
        # Device placement is handled by .to() / DDP — never moved inside
        # compute() to avoid hidden transfers.
        if smooth_sigma > 0.0:
            self.smoother: Optional[_GaussianSmoother] = _GaussianSmoother(smooth_sigma)
        else:
            self.smoother = None

    # ------------------------------------------------------------------
    def compute(
        self,
        state: SamplingState,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute spatial beta_scale_map for the current SamplingState.

        At least one of the following state fields must be non-None:
            entropy_map, enl_map, edge_map, coherence_map,
            lambda1_map, lambda2_map, anisotropy_map.

        Parameters
        ----------
        state  : SamplingState
        device : torch.device

        Returns
        -------
        torch.Tensor
            [N,1,H,W] float32 ∈ [min_beta, max_beta].

        Raises
        ------
        ValueError
            If all feature maps in state are None.
        """
        cfg = self.config

        # Find reference shape from the first available map
        candidates = [
            state.enl_map, state.entropy_map, state.edge_map,
            state.coherence_map, state.lambda1_map,
            state.lambda2_map, state.anisotropy_map,
        ]
        ref = next((m for m in candidates if m is not None), None)
        if ref is None:
            raise ValueError(
                "AdaptiveBetaController.compute(): all feature maps are None. "
                "Ensure StatisticsExtractor has run before calling compute()."
            )

        N, _, H, W = ref.shape
        raw = torch.zeros(N, 1, H, W, dtype=torch.float32, device=device)

        def _add(
            tensor: Optional[torch.Tensor],
            weight: float,
            sign: float,
        ) -> None:
            """Normalise, scale, and accumulate one feature map into raw."""
            if tensor is None or weight == 0.0:
                return
            t = tensor.to(device=device, dtype=torch.float32)
            raw.add_(_std_normalize(t) * (sign * weight))

        # ── Positive contributions (more beta = more noise in flat regions)
        _add(state.enl_map,     cfg.enl_weight,     +1.0)
        _add(state.entropy_map, cfg.entropy_weight,  +1.0)

        # ── Negative contributions (less beta near structure)
        _add(state.edge_map,       cfg.edge_weight,       -1.0)
        _add(state.coherence_map,  cfg.coherence_weight,  -1.0)
        _add(state.lambda1_map,    cfg.lambda1_weight,    -1.0)  # strong λ₁ = edge
        _add(state.lambda2_map,    cfg.lambda2_weight,    -1.0)
        _add(state.anisotropy_map, cfg.anisotropy_weight, -1.0)

        # ── Optional spatial smoothing (smoother already on correct device
        #    because AdaptiveBetaController is an nn.Module moved by the
        #    caller; no .to() call here)
        if self.smoother is not None:
            raw = self.smoother(raw)

        # ── Temperature-scaled sigmoid → bounded output
        raw = raw / max(cfg.beta_temperature, 1e-6)
        scale = cfg.max_beta - cfg.min_beta
        beta_map = cfg.min_beta + scale * torch.sigmoid(raw)
        return beta_map.clamp(cfg.min_beta, cfg.max_beta)

    # ------------------------------------------------------------------
    def compute_dynamic_eta(
        self,
        beta_map: torch.Tensor,
        t: int,
        T: int,
    ) -> float:
        """
        A13 — Dynamic eta from spatial beta map and current timestep.

        Formula
        -------
            η_eff(t) = η_base × β_scale_mean × (t / T)^power

            clamped to [config.min_eta, config.max_eta]

        At t ≈ T:  η_eff ≈ η_base × β_mean   (maximum stochasticity)
        At t → 0:  η_eff → 0                  (deterministic DDIM)

        Parameters
        ----------
        beta_map : torch.Tensor  [N,1,H,W]
            Spatial beta_scale_map from compute().
        t : int
            Current timestep index (0 ≤ t ≤ T-1).
        T : int
            Total diffusion timesteps.

        Returns
        -------
        float  ∈ [min_eta, max_eta]
        """
        cfg       = self.config
        beta_mean = float(beta_map.mean().item())
        t_frac    = max(float(t) / float(T), 1e-8)
        eta_eff   = cfg.eta * beta_mean * (t_frac ** cfg.dynamic_eta_power)
        return float(max(cfg.min_eta, min(cfg.max_eta, eta_eff)))
