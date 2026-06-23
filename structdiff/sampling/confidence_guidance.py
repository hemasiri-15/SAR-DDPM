"""
sampling/confidence_guidance.py
================================
A9 — Confidence-Guided Reverse Sampling.

Produces a per-pixel guidance_map ∈ (0,1) that blends the model's x₀
prediction with the current noisy sample at each reverse step.

Changes from previous version
------------------------------
· EMA max replaces torch.quantile() for uncertainty normalisation.
  torch.quantile triggers a GPU sort (O(N log N)); EMA max is O(N) and
  naturally smoothed across timesteps:
      ema_max_t = α_max · ema_max_{t-1} + (1−α_max) · max(unc_t)
  The ema_max buffer is maintained in self._ema_unc_max.

· Spatial smoothing: after g = sigmoid(...), a 3×3 Gaussian is applied
  to suppress checkerboard artefacts caused by per-pixel guidance
  discontinuities.  Controlled by guidance_smooth_sigma (0.0 = disabled).

· Delta clamping: the correction term (pred_xstart − xt) is clamped to
  [−delta_clamp, +delta_clamp] before multiplying by λ·g.  This prevents
  overshoot when xt is very far from pred_xstart at early timesteps.
  Default: delta_clamp=1.0 (no practical effect for normalised images).

· Confidence entropy: H(confidence) = −∫ p·log(p)  is tracked alongside
  variance.  Approximated as the per-pixel binary entropy of the confidence
  map, then averaged.  Exposed via self.confidence_entropy().

· Confidence history buffer: a deque of the last ``history_len`` EMA-
  smoothed confidence maps is maintained.  self.temporal_variance() returns
  an estimate of Var(confidence) over the recent history.

Mathematical formulation (unchanged)
-------------------------------------
    combined(x,y) = clip(conf(x,y), clip_min, clip_max)
                    − uncertainty_temperature · unc_norm(x,y)

    guidance_map(x,y) = spatial_smooth(sigmoid(combined / conf_temp))

    guided_xt = pred_xstart + λ · g · clip(pred_xstart − xt, ±δ)

EMA smoothing (unchanged)
--------------------------
    conf_ema_t = α · conf_ema_{t-1} + (1−α) · conf_current
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SamplingConfig


class ConfidenceGuidance(nn.Module):
    """
    Pixel-wise confidence-guided denoising for A9.

    Parameters
    ----------
    config : SamplingConfig
        Reads:
            confidence_temperature  : float > 0
            confidence_ema_alpha    : float ∈ [0,1)
            confidence_clip_min     : float
            confidence_clip_max     : float

    Additional parameters
    ---------------------
    uncertainty_temperature : float
        Separate scale for the uncertainty subtraction term.
        Defaults to config.confidence_temperature for backward compat.
    guidance_lambda : float
        Residual blend strength λ.  1.0 = full correction; 0.0 = disabled.
    momentum_alpha : float
        EMA coefficient for guidance_map across timesteps (0 = no momentum).
    ema_max_alpha : float
        EMA coefficient for the uncertainty max tracker.
        0.0 = use instantaneous max (no smoothing).
    guidance_smooth_sigma : float
        Sigma of the 3×3 (or larger) Gaussian applied after sigmoid.
        0.0 disables spatial smoothing.
    delta_clamp : float
        Symmetric clamp on (pred_xstart − xt) before applying guidance.
        Prevents overshoot.  math.inf = no clamp.
    history_len : int
        Length of the confidence map history deque for temporal_variance().
        0 disables history tracking.

    State (maintained across calls)
    --------------------------------
    _ema_conf     : running EMA of the confidence map
    _ema_unc_max  : running EMA of the per-batch uncertainty maximum
    _prev_g_map   : previous guidance_map for momentum blending
    _conf_var_acc : running accumulator for Var(confidence) estimation
    _conf_var_n   : count of updates
    _conf_history : deque of recent EMA-smoothed confidence maps
    """

    def __init__(
        self,
        config: SamplingConfig,
        uncertainty_temperature:  Optional[float] = None,
        guidance_lambda:          float = 1.0,
        momentum_alpha:           float = 0.0,
        ema_max_alpha:            float = 0.9,
        guidance_smooth_sigma:    float = 0.5,
        delta_clamp:              float = math.inf,
        history_len:              int = 0,
    ) -> None:
        super().__init__()
        self.config               = config
        self.conf_temp            = config.confidence_temperature
        self.unc_temp             = (
            uncertainty_temperature
            if uncertainty_temperature is not None
            else config.confidence_temperature
        )
        self.ema_alpha            = config.confidence_ema_alpha
        self.clip_min             = config.confidence_clip_min
        self.clip_max             = config.confidence_clip_max
        self.guidance_lambda      = guidance_lambda
        self.momentum_alpha       = momentum_alpha
        self.ema_max_alpha        = ema_max_alpha
        self.guidance_smooth_sigma = guidance_smooth_sigma
        self.delta_clamp          = delta_clamp
        self.history_len          = history_len

        # ── Pre-build Gaussian smoothing kernel ────────────────────────
        if guidance_smooth_sigma > 0.0:
            radius = max(1, int(3.0 * guidance_smooth_sigma))
            x      = torch.arange(-radius, radius + 1, dtype=torch.float32)
            k1d    = torch.exp(-0.5 * (x / guidance_smooth_sigma) ** 2)
            k1d    = k1d / k1d.sum()
            self._smooth_kh = k1d.view(1, 1, 1, -1)  # [1,1,1,2r+1]
            self._smooth_kv = k1d.view(1, 1, -1, 1)  # [1,1,2r+1,1]
        else:
            self._smooth_kh = None  # type: ignore[assignment]
            self._smooth_kv = None  # type: ignore[assignment]

        # ── Running state ──────────────────────────────────────────────
        self._ema_conf:     Optional[torch.Tensor] = None
        self._ema_unc_max:  Optional[torch.Tensor] = None   # scalar tensor
        self._prev_g_map:   Optional[torch.Tensor] = None
        self._conf_var_acc: Optional[Tuple] = None
        self._conf_var_n:   int = 0
        self._conf_history: Deque[torch.Tensor] = deque(maxlen=max(history_len, 1))

    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        """
        Clear all temporal state.

        Must be called between independent sample_loop() runs to prevent
        EMA state from leaking across images or ablation variants.
        """
        self._ema_conf     = None
        self._ema_unc_max  = None
        self._prev_g_map   = None
        self._conf_var_acc = None
        self._conf_var_n   = 0
        self._conf_history.clear()

    # ------------------------------------------------------------------
    def _smooth_guidance(self, g: torch.Tensor) -> torch.Tensor:
        """
        Apply separable Gaussian smoothing to the guidance map.

        Parameters
        ----------
        g : [N,1,H,W]

        Returns
        -------
        [N,1,H,W] — spatially smoothed guidance map.
        """
        if self._smooth_kh is None:
            return g
        kh = self._smooth_kh.to(device=g.device, dtype=g.dtype)
        kv = self._smooth_kv.to(device=g.device, dtype=g.dtype)
        rh = kh.shape[-1] // 2
        rv = kv.shape[-2] // 2
        g  = F.conv2d(F.pad(g, (rh, rh, 0, 0), "reflect"), kh)
        g  = F.conv2d(F.pad(g, (0, 0, rv, rv), "reflect"), kv)
        return g

    # ------------------------------------------------------------------
    def _normalise_uncertainty(
        self,
        unc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalise uncertainty map using EMA max instead of torch.quantile.

        EMA max:
            ema_max_t = α · ema_max_{t-1} + (1−α) · max(unc_t)

        Parameters
        ----------
        unc : [N,1,H,W] ∈ [0,∞)

        Returns
        -------
        unc_norm : [N,1,H,W] ∈ [0,1]
        """
        batch_max = unc.reshape(unc.shape[0], -1).max(dim=1)[0].mean()  # scalar

        if self.ema_max_alpha > 0.0 and self._ema_unc_max is not None:
            ema_max = (
                self.ema_max_alpha * self._ema_unc_max.to(batch_max.device)
                + (1.0 - self.ema_max_alpha) * batch_max
            )
        else:
            ema_max = batch_max

        self._ema_unc_max = ema_max.detach()
        return (unc / ema_max.clamp(min=1e-8)).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    def compute_guidance_map(
        self,
        confidence_map:  torch.Tensor,
        uncertainty_map: Optional[torch.Tensor],
        device:          torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-pixel guidance strength with EMA smoothing, clipping,
        spatial smoothing, and momentum.

        Parameters
        ----------
        confidence_map : [N,1,H,W] ∈ [0,1]
        uncertainty_map : Optional[N,1,H,W] ∈ [0,∞)
        device : torch.device

        Returns
        -------
        guidance_map   : [N,1,H,W] ∈ (0,1)
        conf_for_state : [N,1,H,W]  EMA-smoothed confidence map.
        """
        conf = confidence_map.to(device=device, dtype=torch.float32)

        # ── Step 1: EMA smoothing ──────────────────────────────────────
        if self.ema_alpha > 0.0 and self._ema_conf is not None:
            ema = self._ema_conf.to(device=device)
            conf = self.ema_alpha * ema + (1.0 - self.ema_alpha) * conf
        self._ema_conf = conf.detach()

        # ── Step 2: Confidence clipping ───────────────────────────────
        conf_clipped = conf.clamp(self.clip_min, self.clip_max)

        # ── Step 3: Variance & entropy tracking ───────────────────────
        self._update_variance(conf_clipped)
        self._update_history(conf_clipped)

        # ── Step 4: Uncertainty term (EMA max normalisation) ──────────
        if uncertainty_map is not None:
            unc      = uncertainty_map.to(device=device, dtype=torch.float32)
            unc_norm = self._normalise_uncertainty(unc)
            combined = conf_clipped - self.unc_temp * unc_norm
        else:
            combined = conf_clipped

        # ── Step 5: Sigmoid with confidence temperature ───────────────
        g = torch.sigmoid(combined / max(self.conf_temp, 1e-6))

        # ── Step 6: Spatial smoothing (suppresses checkerboard) ───────
        g = self._smooth_guidance(g)

        # ── Step 7: Guidance momentum ─────────────────────────────────
        if self.momentum_alpha > 0.0 and self._prev_g_map is not None:
            prev = self._prev_g_map.to(device=device)
            g = self.momentum_alpha * prev + (1.0 - self.momentum_alpha) * g
        self._prev_g_map = g.detach()

        return g, conf.detach()

    # ------------------------------------------------------------------
    def apply_guidance(
        self,
        xt:           torch.Tensor,
        pred_xstart:  torch.Tensor,
        guidance_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply residual guidance blend with delta clamping.

        Formula
        -------
            delta   = clip(pred_xstart − xt, ±delta_clamp)
            guided  = pred_xstart + λ · g · delta

        Clamping prevents overshoot at early timesteps where xt may be far
        from pred_xstart.  With delta_clamp=inf the formula is identical to
        the previous version.

        Parameters
        ----------
        xt           : [N,C,H,W]
        pred_xstart  : [N,C,H,W]
        guidance_map : [N,1,H,W] ∈ (0,1)

        Returns
        -------
        torch.Tensor  [N,C,H,W]
        """
        g     = guidance_map.to(dtype=xt.dtype, device=xt.device)
        delta = pred_xstart - xt
        if math.isfinite(self.delta_clamp):
            delta = delta.clamp(-self.delta_clamp, self.delta_clamp)
        return pred_xstart + self.guidance_lambda * g * delta

    # ------------------------------------------------------------------
    def confidence_variance(self) -> Optional[float]:
        """
        Return running estimate of Var(confidence).

        Returns None if fewer than 2 updates have been recorded.
        Used by A14 for calibration diagnostics.
        """
        if self._conf_var_n < 2 or self._conf_var_acc is None:
            return None
        sum_sq, sum_, n = self._conf_var_acc
        mean    = sum_ / n
        mean_sq = sum_sq / n
        var     = (mean_sq - mean ** 2).clamp(min=0.0)
        return float(var.mean().item())

    # ------------------------------------------------------------------
    def confidence_entropy(self) -> Optional[float]:
        """
        Return the mean per-pixel binary entropy of the most recent
        EMA-smoothed confidence map.

        H(c) = −c·log(c) − (1−c)·log(1−c)   (binary entropy, nats)

        Returns None if no confidence map has been seen yet.
        Useful for A14 calibration diagnostics.
        """
        if self._ema_conf is None:
            return None
        c   = self._ema_conf.clamp(1e-6, 1.0 - 1e-6)
        H   = -(c * torch.log(c) + (1.0 - c) * torch.log(1.0 - c))
        return float(H.mean().item())

    # ------------------------------------------------------------------
    def temporal_variance(self) -> Optional[float]:
        """
        Return an estimate of confidence variance over recent history.

        Computed as the mean pixel-wise variance of the last
        ``history_len`` confidence maps stored in the deque.

        Returns None if fewer than 2 maps are in the history buffer.
        """
        if len(self._conf_history) < 2:
            return None
        stack = torch.stack(list(self._conf_history), dim=0)   # [K,N,1,H,W]
        var   = stack.var(dim=0, unbiased=True)
        return float(var.mean().item())

    # ------------------------------------------------------------------
    def _update_variance(self, conf: torch.Tensor) -> None:
        """Accumulate online variance statistics (sum-of-squares form)."""
        sq = (conf ** 2).detach().mean()
        mu = conf.detach().mean()
        if self._conf_var_acc is None:
            self._conf_var_acc = (sq, mu, torch.tensor(1.0))
        else:
            sum_sq, sum_, cnt = self._conf_var_acc
            self._conf_var_acc = (sum_sq + sq, sum_ + mu, cnt + 1.0)
        self._conf_var_n += 1

    # ------------------------------------------------------------------
    def _update_history(self, conf: torch.Tensor) -> None:
        """Append current confidence map to the history deque."""
        if self.history_len > 0:
            self._conf_history.append(conf.detach())
