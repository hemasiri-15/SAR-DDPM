"""
sampling/config.py
==================
Ultimate Sampling Framework — single source of truth for all hyperparameters.

SamplingMode uses string values (not auto()) so serialised configs remain
human-readable and round-trip cleanly through JSON / YAML without a custom
decoder.

Every field has a safe default so that SamplingConfig() (zero arguments)
produces a sensible deterministic DDIM run compatible with the original
SAR-DDPM checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Mode enum
# ---------------------------------------------------------------------------

class SamplingMode(Enum):
    """
    Operational mode of SamplingController.

    String values are used instead of auto() so that
    ``SamplingMode("adaptive")`` and ``mode.value == "adaptive"``
    work without a custom codec.

    FIXED      — A4:           plain DDIM with a fixed schedule and scalar eta.
    ADAPTIVE   — A4 + A8:     DDIM with spatial AdaptiveBeta + dynamic eta.
    CONFIDENCE — A4 + A9:     DDIM with ConfidenceGuidance applied per step.
    ULTIMATE   — All modules active + CycleSpinAggregator.
    """
    FIXED      = "fixed"
    ADAPTIVE   = "adaptive"
    CONFIDENCE = "confidence"
    ULTIMATE   = "ultimate"


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class SamplingConfig:
    """
    Immutable (by convention) configuration for the Ultimate Sampling Framework.

    Instantiate once and pass to SamplingController.  Do not mutate after
    construction; create a new instance per ablation variant.

    Schedule
    --------
    schedule_type : str
        Key into SCHEDULE_REGISTRY:
            "linear" | "quadratic" | "cubic" | "cosine" |
            "logarithmic" | "sigmoid" | "exponential" | "hybrid" | "custom".
        Preset aliases: "ddim10" | "ddim25" | "ddim50" |
                        "ddim100" | "ddim250" | "ddim500" | "ddim1000".
    num_steps : int
        Number of reverse DDIM steps (ignored for presets and "custom").
    eta : float
        Base DDIM stochasticity ∈ [0, 1].  0 → fully deterministic DDIM.

    Mode
    ----
    sampling_mode : SamplingMode
        Controls which sub-modules are active per step.

    Adaptive beta (A8)
    ------------------
    use_adaptive_beta : bool
    min_beta, max_beta : float
        Output range for beta_scale_map.
    beta_temperature : float
        Scales the raw logit fed into sigmoid(raw / temperature).
        Higher → softer spatial variation; lower → sharper.
    enl_weight : float
        Weight for ENL contribution (positive — high ENL = homogeneous).
    entropy_weight : float
        Weight for entropy proxy contribution (positive).
    edge_weight : float
        Weight for edge magnitude (negative sign in formula).
    coherence_weight : float
        Weight for structure-tensor coherence (negative sign).
    lambda1_weight : float
        Weight for major eigenvalue (negative — strong λ₁ = edge, preserve).
    lambda2_weight : float
        Weight for minor eigenvalue (negative — large λ₂ = isotropic noise).
    anisotropy_weight : float
        Weight for anisotropy index (negative — oriented = preserve).

    Dynamic eta (A13)
    -----------------
    use_dynamic_eta : bool
    dynamic_eta_power : float
        Exponent p in η(t) = η_base × β_scale_mean × (t/T)^p.
    min_eta : float
        Lower clamp for effective eta.  Allows a noise floor even at t→0.
    max_eta : float
        Upper clamp for effective eta.  Prevents runaway stochasticity.

    Confidence guidance (A9)
    ------------------------
    use_confidence_guidance : bool
    confidence_temperature : float
        Temperature for sigmoid((conf - temp·unc) / temp).
    confidence_ema_alpha : float
        EMA smoothing factor for the confidence map across timesteps.
        0.0 → no smoothing; values approaching 1.0 → very slow update.
    confidence_clip_min : float
        Lower clamp applied to raw confidence before sigmoid.
    confidence_clip_max : float
        Upper clamp applied to raw confidence before sigmoid.

    Structure tensor (A10/A11)
    --------------------------
    enable_structure_tensor : bool
    sigma_gradient : float
        Pre-smoothing sigma before Sobel (reduces noise sensitivity).
    sigma_integration : float
        Integration window sigma for outer-product averaging.

    Multi-scale maps (A13/A14)
    --------------------------
    enable_multiscale_maps : bool

    Cycle spinning (A26)
    --------------------
    use_cycle_spinning : bool
    cycle_width : int
        Spatial shift step in pixels.  0 disables spinning.
    cycle_strategy : str
        "uniform"     — equal weights (original behaviour).
        "confidence"  — reserved for A26r (confidence-weighted aggregation).

    Runtime & trajectory
    --------------------
    enable_runtime_profiler : bool
        Wrap each sample_loop call with RuntimeProfiler.
    enable_trajectory_logging : bool
        Activate TrajectoryLogger inside SamplingController.
    trajectory_timesteps : Tuple[int, ...]
        Timesteps at which xt snapshots are saved by TrajectoryLogger.

    Reproducibility
    ---------------
    seed : int
        Master RNG seed forwarded to torch.manual_seed before sampling.

    Precision
    ---------
    amp_enabled : bool
        Wrap forward passes in torch.autocast (FP16 on CUDA, BF16 if bf16).
    bf16_enabled : bool
        Use bfloat16 instead of float16 when amp_enabled=True.

    Device
    ------
    device : Optional[str]
        Target device string ("cuda", "cpu", "cuda:1", …).
        None → inferred from the model tensor at runtime.
    """

    # ── Schedule ──────────────────────────────────────────────────────────
    schedule_type: str  = "linear"
    num_steps:     int  = 100
    eta:           float = 0.0

    # ── Mode ──────────────────────────────────────────────────────────────
    sampling_mode: SamplingMode = SamplingMode.FIXED

    # ── Adaptive beta (A8) ────────────────────────────────────────────────
    use_adaptive_beta:  bool  = False
    min_beta:           float = 0.0
    max_beta:           float = 1.0
    beta_temperature:   float = 1.0
    enl_weight:         float = 1.0   # separate from edge_weight
    entropy_weight:     float = 0.5   # separate from edge_weight
    edge_weight:        float = 2.0
    coherence_weight:   float = 1.0
    lambda1_weight:     float = 0.5   # applied with negative sign (edge → preserve)
    lambda2_weight:     float = 0.5
    anisotropy_weight:  float = 0.5

    # ── Dynamic eta (A13) ─────────────────────────────────────────────────
    use_dynamic_eta:    bool  = False
    dynamic_eta_power:  float = 0.5
    min_eta:            float = 0.0   # floor clamp on effective eta
    max_eta:            float = 1.0   # ceiling clamp on effective eta

    # ── Confidence guidance (A9) ──────────────────────────────────────────
    use_confidence_guidance: bool  = False
    confidence_temperature:  float = 1.0
    confidence_ema_alpha:    float = 0.0    # 0 = no EMA smoothing
    confidence_clip_min:     float = -5.0   # pre-sigmoid clamp (logit space)
    confidence_clip_max:     float = +5.0

    # ── Structure tensor (A10/A11) ────────────────────────────────────────
    enable_structure_tensor: bool  = False
    sigma_gradient:          float = 1.0
    sigma_integration:       float = 3.0

    # ── Multi-scale maps (A13/A14) ────────────────────────────────────────
    enable_multiscale_maps: bool = False

    # ── Cycle spinning (A26) ──────────────────────────────────────────────
    use_cycle_spinning: bool  = False
    cycle_width:        int   = 0
    cycle_strategy:     str   = "uniform"   # "uniform" | "confidence"

    # ── Runtime & trajectory ──────────────────────────────────────────────
    enable_runtime_profiler:    bool              = False
    enable_trajectory_logging:  bool              = False
    trajectory_timesteps: Tuple[int, ...] = (999, 750, 500, 250, 100, 50, 25, 0)

    # ── Reproducibility ───────────────────────────────────────────────────
    seed: int = 42

    # ── Precision ─────────────────────────────────────────────────────────
    amp_enabled:  bool = False
    bf16_enabled: bool = False

    # ── Device ────────────────────────────────────────────────────────────
    device: Optional[str] = None    # None → infer from model at runtime

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        # eta
        if not (0.0 <= self.eta <= 1.0):
            raise ValueError(f"eta must be in [0, 1], got {self.eta}")

        # steps
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")

        # temperatures
        if self.beta_temperature <= 0.0:
            raise ValueError(
                f"beta_temperature must be > 0, got {self.beta_temperature}"
            )
        if self.confidence_temperature <= 0.0:
            raise ValueError(
                f"confidence_temperature must be > 0, "
                f"got {self.confidence_temperature}"
            )

        # EMA alpha
        if not (0.0 <= self.confidence_ema_alpha < 1.0):
            raise ValueError(
                f"confidence_ema_alpha must be in [0, 1), "
                f"got {self.confidence_ema_alpha}"
            )

        # eta clamps
        if not (0.0 <= self.min_eta <= self.max_eta <= 1.0):
            raise ValueError(
                f"Require 0 ≤ min_eta ≤ max_eta ≤ 1, "
                f"got min_eta={self.min_eta}, max_eta={self.max_eta}"
            )

        # beta range
        if not (0.0 <= self.min_beta <= self.max_beta <= 1.0):
            raise ValueError(
                f"Require 0 ≤ min_beta ≤ max_beta ≤ 1, "
                f"got min_beta={self.min_beta}, max_beta={self.max_beta}"
            )

        # cycle strategy
        if self.cycle_strategy not in ("uniform", "confidence"):
            raise ValueError(
                f"Unknown cycle_strategy '{self.cycle_strategy}'. "
                f"Valid: 'uniform', 'confidence'."
            )

        # mode coercion: accept bare strings for convenience
        if isinstance(self.sampling_mode, str):
            object.__setattr__(
                self, "sampling_mode", SamplingMode(self.sampling_mode)
            )
