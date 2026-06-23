"""
sampling/multiscale_maps.py
============================
A13 / A14 — Multi-Scale Feature Fusion.

Computes seven classes of local image statistics at multiple spatial scales
and fuses them into single summary maps via weighted average.

Changes from previous version
------------------------------
· Sobel and Laplacian kernels moved into __init__ / register_buffer()
  → zero kernel allocation inside forward(), compile-safe, DDP-safe.
· Real FFT energy via torch.fft.rfft2 + patch-averaged power spectrum
  replacing the Parseval proxy.  A12 freq_energy is now meaningful.
· cv_variance: Var(CV across scales) — texture complexity measure.
· skewness: E[(x-μ)³]/σ³ — speckle characterisation.
· kurtosis: E[(x-μ)⁴]/σ⁴ — tail weight, useful for A16.
· gradient_coherence: mean cosine similarity between neighbouring
  gradient vectors — edge descriptor complementary to Sobel magnitude.

Design philosophy
-----------------
· Five scales: 3, 5, 9, 15, 31 pixels.  Covers fine speckle (3px) through
  scene-level structure (31px) without redundancy.
· Weights are L1-normalised automatically.
· All computation is pure PyTorch and compile-safe.
· No numpy, no scipy, no skimage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class MultiScaleMaps:
    """
    Fused multi-scale feature maps.

    All fields are [N,1,H,W] float32 unless documented otherwise.

    Fields
    ------
    edge               — fused Sobel edge magnitude
    entropy            — fused MAD-based entropy proxy (not true Shannon entropy)
    enl                — fused Equivalent Number of Looks estimate
    cv                 — fused local coefficient of variation (σ/μ)
    texture            — fused Laplacian variance (fine edge density)
    freq_energy        — fused local FFT power spectrum energy (A12)
    cv_variance        — Var(CV across scales) per pixel [N,1,H,W]
    skewness           — fused local skewness E[(x-μ)³]/σ³
    kurtosis           — fused local kurtosis E[(x-μ)⁴]/σ⁴
    gradient_coherence — mean cosine similarity of neighbouring gradients
    """
    edge:               torch.Tensor
    entropy:            torch.Tensor
    enl:                torch.Tensor
    cv:                 torch.Tensor
    texture:            torch.Tensor
    freq_energy:        torch.Tensor
    cv_variance:        torch.Tensor
    skewness:           torch.Tensor
    kurtosis:           torch.Tensor
    gradient_coherence: torch.Tensor


# ---------------------------------------------------------------------------
# MultiScaleExtractor
# ---------------------------------------------------------------------------

_DEFAULT_SCALES  = (3, 5, 9, 15, 31)
_DEFAULT_WEIGHTS = (0.35, 0.25, 0.20, 0.12, 0.08)   # fine→coarse decay, sum=1.0


class MultiScaleExtractor(nn.Module):
    """
    Computes and fuses multi-scale image statistics.

    Kernels (Sobel, Laplacian, FFT window) are stored as buffers so they
    are allocated once and never reallocated inside forward().

    Parameters
    ----------
    scales : Sequence[int]
        Patch sizes.  Must be odd and ≥ 3.
    edge_weights, entropy_weights, enl_weights, cv_weights,
    texture_weights, freq_weights, skewness_weights, kurtosis_weights,
    grad_coh_weights : Optional[Sequence[float]]
        Per-scale unnormalised weights for each feature type.
        None → use the global default weight vector.
    fft_patch_size : int
        Patch size for the sliding-window FFT energy computation.
        Must be a power of 2 for efficiency.  Default: 32.
    """

    def __init__(
        self,
        scales:              Sequence[int]            = _DEFAULT_SCALES,
        edge_weights:        Optional[Sequence[float]] = None,
        entropy_weights:     Optional[Sequence[float]] = None,
        enl_weights:         Optional[Sequence[float]] = None,
        cv_weights:          Optional[Sequence[float]] = None,
        texture_weights:     Optional[Sequence[float]] = None,
        freq_weights:        Optional[Sequence[float]] = None,
        skewness_weights:    Optional[Sequence[float]] = None,
        kurtosis_weights:    Optional[Sequence[float]] = None,
        grad_coh_weights:    Optional[Sequence[float]] = None,
        fft_patch_size:      int = 32,
    ) -> None:
        super().__init__()
        self.scales        = list(scales)
        self.fft_patch_size = fft_patch_size

        def _w(override: Optional[Sequence[float]]) -> Tuple[float, ...]:
            return tuple(override) if override is not None else _DEFAULT_WEIGHTS

        self._edge_w     = _w(edge_weights)
        self._entropy_w  = _w(entropy_weights)
        self._enl_w      = _w(enl_weights)
        self._cv_w       = _w(cv_weights)
        self._texture_w  = _w(texture_weights)
        self._freq_w     = _w(freq_weights)
        self._skew_w     = _w(skewness_weights)
        self._kurt_w     = _w(kurtosis_weights)
        self._gcoh_w     = _w(grad_coh_weights)

        # ── Sobel kernels [1,1,3,3] ───────────────────────────────────
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

        # ── Laplacian kernel [1,1,3,3] ────────────────────────────────
        lap_k = torch.tensor(
            [[0., 1., 0.],
             [1., -4., 1.],
             [0., 1., 0.]],
        ).view(1, 1, 3, 3)
        self.register_buffer("_lap_k", lap_k)

        # ── Hann window for FFT [fft_patch_size, fft_patch_size] ──────
        hann_1d = torch.hann_window(fft_patch_size, periodic=False)
        hann_2d = hann_1d.unsqueeze(0) * hann_1d.unsqueeze(1)   # [P,P]
        self.register_buffer("_hann", hann_2d)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reflect_pool(x: torch.Tensor, k: int) -> torch.Tensor:
        """Average pool with reflect padding to avoid border bias."""
        p = k // 2
        return F.avg_pool2d(
            F.pad(x, (p, p, p, p), "reflect"),
            kernel_size=k, stride=1, padding=0,
        )

    def _sobel_grads(self, gray: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (Ix, Iy) using registered Sobel kernels."""
        Ix = F.conv2d(F.pad(gray, (1, 1, 1, 1), "reflect"), self._kx)
        Iy = F.conv2d(F.pad(gray, (1, 1, 1, 1), "reflect"), self._ky)
        return Ix, Iy

    # ------------------------------------------------------------------
    # Single-scale primitives
    # ------------------------------------------------------------------

    def _edge_scale(self, gray: torch.Tensor, k: int) -> torch.Tensor:
        """
        Sobel edge magnitude at effective scale k.

        For k=3, Sobel is applied directly.  For k>3, the input is
        average-pooled first to approximate the larger-scale derivative.
        Uses registered kernels — zero allocation.
        """
        inp = self._reflect_pool(gray, k) if k > 3 else gray
        Ix = F.conv2d(F.pad(inp, (1, 1, 1, 1), "reflect"), self._kx)
        Iy = F.conv2d(F.pad(inp, (1, 1, 1, 1), "reflect"), self._ky)
        return (Ix ** 2 + Iy ** 2).sqrt()

    def _entropy_scale(self, gray: torch.Tensor, k: int) -> torch.Tensor:
        """
        MAD-based local entropy proxy in a k×k patch.

        entropy_proxy ≈ log(MAD + ε)

        Using MAD rather than variance is more robust to SAR speckle
        outliers.  Note: this is an *entropy proxy*, not true Shannon entropy.
        """
        mu  = self._reflect_pool(gray, k)
        mad = self._reflect_pool((gray - mu).abs(), k).clamp(min=1e-8)
        return torch.log(mad)

    def _enl_scale(self, gray: torch.Tensor, k: int) -> torch.Tensor:
        """Local Equivalent Number of Looks: ENL = (μ/σ)² in a k×k patch."""
        g   = gray.clamp(min=1e-8)
        mu  = self._reflect_pool(g, k)
        mu2 = self._reflect_pool(g ** 2, k)
        var = (mu2 - mu ** 2).clamp(min=1e-8)
        return (mu ** 2) / var

    def _cv_scale(self, gray: torch.Tensor, k: int) -> torch.Tensor:
        """
        Local coefficient of variation: CV = σ/μ in a k×k patch.

        CV ≈ 1/√ENL for fully-developed speckle.
        """
        g   = gray.clamp(min=1e-8)
        mu  = self._reflect_pool(g, k).clamp(min=1e-8)
        mu2 = self._reflect_pool(g ** 2, k)
        var = (mu2 - mu ** 2).clamp(min=0.0)
        return var.sqrt() / mu

    def _texture_scale(self, gray: torch.Tensor, k: int) -> torch.Tensor:
        """
        Laplacian variance in a k×k patch — fine edge density.

        Uses registered Laplacian kernel — zero allocation.
        """
        lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), "reflect"), self._lap_k)
        mu  = self._reflect_pool(lap, k)
        mu2 = self._reflect_pool(lap ** 2, k)
        return (mu2 - mu ** 2).clamp(min=0.0)

    def _freq_energy_scale(self, gray: torch.Tensor, k: int) -> torch.Tensor:
        """
        Local FFT power spectrum energy at scale k.

        Implementation
        --------------
        Uses torch.fft.rfft2 on (k×k)-windowed patches extracted via
        unfold.  A Hann window is applied before FFT to reduce spectral
        leakage.  The power spectrum is averaged across frequency bins
        to produce a per-patch scalar, then scattered back to the
        spatial grid via fold.

        For large k (>= fft_patch_size), the image is first
        average-pooled to fft_patch_size to keep cost constant.

        Returns
        -------
        [N,1,H,W]  — local FFT energy map.
        """
        P   = self.fft_patch_size
        N, _, H, W = gray.shape

        # If requested scale is larger, pool down to P first
        if k > P:
            inp = self._reflect_pool(gray, k)
        else:
            inp = gray

        # Pad to make H,W divisible by stride=k (non-overlapping extraction)
        stride = max(1, k // 2)
        ph = (stride - H % stride) % stride
        pw = (stride - W % stride) % stride
        padded = F.pad(inp, (0, pw, 0, ph), "reflect")
        _, _, Hp, Wp = padded.shape

        # Unfold into patches [N, 1*k*k, num_patches]
        patch_k = min(k, P)
        patches = padded.unfold(2, patch_k, stride).unfold(3, patch_k, stride)
        # patches: [N, 1, n_h, n_w, patch_k, patch_k]
        n_h, n_w = patches.shape[2], patches.shape[3]
        patches = patches.contiguous().view(N * n_h * n_w, 1, patch_k, patch_k)

        # Apply Hann window (broadcast to patch size)
        if patch_k == P:
            win = self._hann.to(dtype=patches.dtype)
        else:
            win_1d = torch.hann_window(patch_k, periodic=False,
                                        dtype=patches.dtype, device=patches.device)
            win = win_1d.unsqueeze(0) * win_1d.unsqueeze(1)

        windowed = patches * win.unsqueeze(0).unsqueeze(0)

        # FFT and power spectrum
        spectrum = torch.fft.rfft2(windowed, norm="ortho")   # [M,1,k,k//2+1]
        power    = (spectrum.real ** 2 + spectrum.imag ** 2).mean(dim=(1, 2, 3))  # [M]

        # Scatter back to spatial map via nearest-neighbour upsampling
        energy_patches = power.view(N, 1, n_h, n_w)
        energy_map = F.interpolate(
            energy_patches.float(), size=(H, W), mode="nearest"
        )
        return energy_map

    def _skewness_scale(self, gray: torch.Tensor, k: int) -> torch.Tensor:
        """
        Local skewness E[(x-μ)³]/σ³ in a k×k patch.

        Useful for characterising asymmetric speckle distributions.
        Returns signed skewness ∈ (-∞, ∞).
        """
        mu     = self._reflect_pool(gray, k)
        diff   = gray - mu
        mu3    = self._reflect_pool(diff ** 3, k)
        mu2    = self._reflect_pool(diff ** 2, k).clamp(min=1e-8)
        sigma3 = mu2.sqrt() ** 3
        return mu3 / sigma3.clamp(min=1e-8)

    def _kurtosis_scale(self, gray: torch.Tensor, k: int) -> torch.Tensor:
        """
        Local excess kurtosis E[(x-μ)⁴]/σ⁴ - 3 in a k×k patch.

        Measures tail weight.  Useful for A16.  Returns excess kurtosis
        (Gaussian = 0, heavy-tailed > 0).
        """
        mu   = self._reflect_pool(gray, k)
        diff = gray - mu
        mu4  = self._reflect_pool(diff ** 4, k)
        mu2  = self._reflect_pool(diff ** 2, k).clamp(min=1e-8)
        return mu4 / (mu2 ** 2).clamp(min=1e-8) - 3.0

    def _gradient_coherence_scale(self, gray: torch.Tensor, k: int) -> torch.Tensor:
        """
        Mean cosine similarity between neighbouring gradient vectors in a
        k×k patch.

        Formula
        -------
        For each pixel (i,j) and its right/down neighbours (i,j+1), (i+1,j):
            cos_h = (Ix·Ix' + Iy·Iy') / (|G|·|G'| + ε)

        The horizontal and vertical coherence maps are averaged and then
        smoothed over the k×k window.

        Returns values ∈ [-1, 1].  High values → locally consistent edge
        direction, low values → isotropic / chaotic gradients.
        """
        Ix, Iy = self._sobel_grads(gray)

        # Shift gradients by one pixel
        Ix_r = torch.roll(Ix, shifts=-1, dims=3)   # right neighbour
        Iy_r = torch.roll(Iy, shifts=-1, dims=3)
        Ix_d = torch.roll(Ix, shifts=-1, dims=2)   # down neighbour
        Iy_d = torch.roll(Iy, shifts=-1, dims=2)

        mag   = (Ix ** 2  + Iy ** 2).sqrt()
        mag_r = (Ix_r ** 2 + Iy_r ** 2).sqrt()
        mag_d = (Ix_d ** 2 + Iy_d ** 2).sqrt()

        eps = 1e-8
        cos_h = (Ix * Ix_r + Iy * Iy_r) / (mag * mag_r + eps)
        cos_v = (Ix * Ix_d + Iy * Iy_d) / (mag * mag_d + eps)
        local_coh = 0.5 * (cos_h + cos_v)

        return self._reflect_pool(local_coh, k)

    # ------------------------------------------------------------------
    # Fusion helper
    # ------------------------------------------------------------------

    def _fuse(
        self,
        fn,
        gray: torch.Tensor,
        weights: Tuple[float, ...],
    ) -> torch.Tensor:
        """
        Compute fn(gray, k) at each scale and return the weighted average.

        Weights are L1-normalised internally.
        """
        w_sum = sum(weights)
        out: Optional[torch.Tensor] = None
        for k, w in zip(self.scales, weights):
            m = fn(gray, k) * (w / w_sum)
            out = m if out is None else out + m
        assert out is not None
        return out

    def _cv_variance(self, gray: torch.Tensor) -> torch.Tensor:
        """
        Per-pixel variance of CV across scales.

        Measures how much the local coefficient of variation changes as we
        zoom out — high values indicate scale-dependent texture complexity.

        Returns [N,1,H,W].
        """
        cv_maps = [self._cv_scale(gray, k) for k in self.scales]
        stack   = torch.stack(cv_maps, dim=0)   # [S, N, 1, H, W]
        mean    = stack.mean(dim=0)
        var     = ((stack - mean) ** 2).mean(dim=0)
        return var

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> MultiScaleMaps:
        """
        Compute all ten fused multi-scale maps.

        Parameters
        ----------
        x : torch.Tensor  [N,C,H,W]
            Current noisy sample.  Cast to float32 internally.

        Returns
        -------
        MultiScaleMaps  — all fields [N,1,H,W] float32.
        """
        gray = x.mean(dim=1, keepdim=True).to(dtype=torch.float32)

        return MultiScaleMaps(
            edge               = self._fuse(self._edge_scale,              gray, self._edge_w),
            entropy            = self._fuse(self._entropy_scale,           gray, self._entropy_w),
            enl                = self._fuse(self._enl_scale,               gray, self._enl_w),
            cv                 = self._fuse(self._cv_scale,                gray, self._cv_w),
            texture            = self._fuse(self._texture_scale,           gray, self._texture_w),
            freq_energy        = self._fuse(self._freq_energy_scale,       gray, self._freq_w),
            cv_variance        = self._cv_variance(gray),
            skewness           = self._fuse(self._skewness_scale,          gray, self._skew_w),
            kurtosis           = self._fuse(self._kurtosis_scale,          gray, self._kurt_w),
            gradient_coherence = self._fuse(self._gradient_coherence_scale,gray, self._gcoh_w),
        )
