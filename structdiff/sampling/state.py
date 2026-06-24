"""
sampling/state.py
=================
Immutable flow object passed between every stage of the reverse chain.

Design rules
------------
* Never mutated in-place — always call .replace(**kwargs) to produce a
  new instance.
* replace() deep-copies metadata so callers cannot share a mutable
  reference between successive states.
* trajectory is shallow-copied (list(self.trajectory)) on every replace()
  call, giving each state its own independent snapshot list.  This is
  slightly more expensive than the shared-reference approach but eliminates
  subtle aliasing bugs when states are branched (e.g. in ablation loops or
  cycle-spinning).
* dtype and device are derived from xt in __post_init__ and stored
  explicitly so downstream modules never need to interrogate tensors.
* pred_xstart and noise_pred are carried alongside xt so TrajectoryLogger
  can save them without an extra forward pass.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class SamplingState:
    """
    Carries the complete reverse-chain state across all USF modules.

    Core fields
    -----------
    xt : torch.Tensor  [N,C,H,W]
        Current noisy sample at the present timestep.
    timestep : int
        Current diffusion timestep index ∈ [0, T-1].
    eta : float
        Effective scalar eta for this step (may differ from config.eta
        when dynamic eta is active).
    schedule : List[int]
        Full strictly-descending timestep sequence for the current run.

    Denoising outputs (per step)
    ----------------------------
    pred_xstart : Optional[torch.Tensor]  [N,C,H,W]
        Model's x₀ prediction from the most recent DDIM step.
        Populated after diffusion.ddim_sample(); None before first step.
    noise_pred : Optional[torch.Tensor]  [N,C,H,W]
        Raw model output (epsilon or v-prediction) before conversion to
        x₀.  Populated by SamplingController if the diffusion wrapper
        exposes it; otherwise None.

    Statistics maps — all Optional[torch.Tensor]  [N,1,H,W]
    ---------------------------------------------------------
    entropy_map     — local entropy proxy (higher = more uncertain / speckled)
    enl_map         — local ENL estimate  (higher = more homogeneous)
    edge_map        — edge magnitude      (higher = stronger edge)
    beta_map        — spatial beta_scale  (output of AdaptiveBetaController)
    confidence_map  — model confidence   ∈ [0, 1]
    uncertainty_map — model uncertainty  ∈ [0, ∞)

    Structure tensor maps (A10/A11) — Optional[torch.Tensor]  [N,1,H,W]
    ---------------------------------------------------------------------
    lambda1_map     — major eigenvalue  (edge/anisotropy strength)
    lambda2_map     — minor eigenvalue
    coherence_map   — (λ₁-λ₂)/(λ₁+λ₂+ε) ∈ [0,1]
    anisotropy_map  — alias for coherence_map (A11 naming convention)
    orientation_map — dominant edge angle ∈ [-π/2, π/2]

    Trajectory
    ----------
    trajectory : List[Dict[str, Any]]
        Ordered list of snapshot dicts appended by TrajectoryLogger.
        Expected keys per snapshot:
            "t", "xt", "pred_xstart", "noise_pred",
            "eta", "beta_map", "confidence_map".
        Each replace() copies this list so states can diverge without
        aliasing (relevant when CycleSpinAggregator branches the chain).

    Housekeeping
    ------------
    dtype   : torch.dtype   — inferred from xt; set in __post_init__.
    device  : torch.device  — inferred from xt; set in __post_init__.
    metadata : Dict[str, Any]
        Arbitrary key-value store for extensions (e.g. ablation tags,
        run IDs, schedule hashes).  Deep-copied on every replace().
    """

    # ── Core ──────────────────────────────────────────────────────────────
    xt:        torch.Tensor
    timestep:  int
    eta:       float
    schedule:  List[int]

    # ── Denoising outputs ─────────────────────────────────────────────────
    pred_xstart: Optional[torch.Tensor] = None   # [N,C,H,W]
    noise_pred:  Optional[torch.Tensor] = None   # [N,C,H,W]

    # ── Single-scale statistics maps ──────────────────────────────────────
    entropy_map:     Optional[torch.Tensor] = None   # [N,1,H,W]
    enl_map:         Optional[torch.Tensor] = None   # [N,1,H,W]
    edge_map:        Optional[torch.Tensor] = None   # [N,1,H,W]
    beta_map:        Optional[torch.Tensor] = None   # [N,1,H,W]
    confidence_map:  Optional[torch.Tensor] = None   # [N,1,H,W]
    uncertainty_map: Optional[torch.Tensor] = None   # [N,1,H,W]

    # ── Structure tensor maps (A10/A11) ───────────────────────────────────
    lambda1_map:     Optional[torch.Tensor] = None   # [N,1,H,W]
    lambda2_map:     Optional[torch.Tensor] = None   # [N,1,H,W]
    coherence_map:   Optional[torch.Tensor] = None   # [N,1,H,W]
    anisotropy_map:  Optional[torch.Tensor] = None   # [N,1,H,W]
    orientation_map: Optional[torch.Tensor] = None   # [N,1,H,W]

    # ── Trajectory snapshots ──────────────────────────────────────────────
    trajectory: List[Dict[str, Any]] = field(default_factory=list)

    # ── Housekeeping ──────────────────────────────────────────────────────
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Derived — not constructor args; populated by __post_init__
    dtype:  torch.dtype  = field(init=False)
    device: torch.device = field(init=False)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype",  self.xt.dtype)
        object.__setattr__(self, "device", self.xt.device)

    # ------------------------------------------------------------------
    def replace(self, **kwargs) -> "SamplingState":
        """
        Return a new SamplingState with selected fields replaced.

        Copying policy
        --------------
        metadata  — deep-copied so callers cannot mutate a shared dict.
        trajectory — shallow-copied (new list, same inner dicts) so each
                     state owns its own list but snapshots are not cloned.
                     Inner dicts are treated as immutable once appended.
        All tensor fields — references are passed through (not cloned);
        tensors are immutable once placed in state by convention.
        """
        current: Dict[str, Any] = {
            "xt":             self.xt,
            "timestep":       self.timestep,
            "eta":            self.eta,
            "schedule":       self.schedule,
            "pred_xstart":    self.pred_xstart,
            "noise_pred":     self.noise_pred,
            "entropy_map":    self.entropy_map,
            "enl_map":        self.enl_map,
            "edge_map":       self.edge_map,
            "beta_map":       self.beta_map,
            "confidence_map": self.confidence_map,
            "uncertainty_map":self.uncertainty_map,
            "lambda1_map":    self.lambda1_map,
            "lambda2_map":    self.lambda2_map,
            "coherence_map":  self.coherence_map,
            "anisotropy_map": self.anisotropy_map,
            "orientation_map":self.orientation_map,
            "trajectory":     list(self.trajectory),        # shallow copy
            "metadata":       copy.deepcopy(self.metadata), # deep copy
        }
        current.update(kwargs)
        return SamplingState(**current)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """One-line human-readable description for logging."""
        active = [
            name for name in (
                "entropy_map", "enl_map", "edge_map", "beta_map",
                "confidence_map", "uncertainty_map",
                "lambda1_map", "lambda2_map", "coherence_map",
                "anisotropy_map", "orientation_map",
                "pred_xstart", "noise_pred",
            )
            if getattr(self, name) is not None
        ]
        return (
            f"SamplingState("
            f"t={self.timestep}, "
            f"eta={self.eta:.4f}, "
            f"shape={tuple(self.xt.shape)}, "
            f"dtype={self.dtype}, "
            f"device={self.device}, "
            f"active={active}, "
            f"frames={len(self.trajectory)})"
        )

