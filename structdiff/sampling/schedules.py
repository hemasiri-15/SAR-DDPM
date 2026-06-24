"""
sampling/schedules.py
=====================
A4 — DDIM Timestep Schedule Generators.

All schedule functions return a strictly descending list[int] of timestep
indices in [0, T-1] with the following guarantees:

    schedule[0]  == T - 1          (start at full noise)
    schedule[-1] == 0              (end at clean prediction)
    len(set(schedule)) == len(schedule)   (no duplicates)
    schedule[i] > schedule[i+1] for all i  (strict monotone decrease)

Supported schedule types
------------------------
    linear       — uniform spacing (default DDIM; Song et al. 2020)
    quadratic    — dense near low-t  (fine detail / SAR despeckle tail)
    cubic        — very dense near low-t
    cosine       — denser near both endpoints (Nichol & Dhariwal 2021)
    logarithmic  — dense near high-t (early coarse denoising)
    sigmoid      — middle-dense; smooth S-curve between extremes
    exponential  — dense near high-t with configurable decay rate α
    hybrid       — linear first half + cosine second half
    custom       — caller-supplied explicit list

Preset aliases (fixed num_steps, linear spacing)
-------------------------------------------------
    ddim10, ddim25, ddim50, ddim100, ddim250, ddim500, ddim1000

Public API
----------
    get_schedule(schedule_type, num_timesteps, num_steps, …) → List[int]
    schedule_hash(schedule)  → str   (SHA-256 hex, 16 chars)
    compression_ratio(schedule, num_timesteps)  → float
    schedule_info(schedule, num_timesteps)  → dict

Ablation compatibility
----------------------
A4, A8, A9, A10, A11, A13, A14, A21, A26 all consume this module's output
unchanged.  The registry architecture allows additional schedule types to be
injected at runtime (e.g. learned schedules from A21).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build(raw: torch.Tensor, T: int) -> List[int]:
    """
    Clamp, deduplicate, sort descending, and enforce endpoint guarantees.

    Parameters
    ----------
    raw : torch.Tensor
        Float tensor of candidate timestep values (need not be integers yet).
    T : int
        Total diffusion timesteps; valid index range is [0, T-1].

    Returns
    -------
    List[int]
        Strictly descending, unique, clamped, starting at T-1, ending at 0.
    """
    idx = raw.long().clamp(0, T - 1)
    idx = torch.unique(idx, sorted=True)   # ascending unique
    lst: List[int] = idx.flip(0).tolist()     # descending

    # Enforce start/end guarantees
    if not lst or lst[0] != T - 1:
        lst.insert(0, T - 1)
    if lst[-1] != 0:
        lst.append(0)

    # Re-deduplicate after insertions (rare edge case near small T)
    seen: set = set()
    out: List[int] = []
    for v in lst:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _validate(schedule: List[int], T: int) -> None:
    """Raise ValueError if schedule violates any contract."""
    if not schedule:
        raise ValueError("Schedule is empty.")
    if schedule[0] != T - 1:
        raise ValueError(
            f"schedule[0] must equal T-1={T - 1}, got {schedule[0]}"
        )
    if schedule[-1] != 0:
        raise ValueError(
            f"schedule[-1] must equal 0, got {schedule[-1]}"
        )
    for i in range(len(schedule) - 1):
        if schedule[i] <= schedule[i + 1]:
            raise ValueError(
                f"Schedule is not strictly decreasing at index {i}: "
                f"{schedule[i]} → {schedule[i + 1]}"
            )
    if len(schedule) != len(set(schedule)):
        raise ValueError("Schedule contains duplicate timestep indices.")


# ---------------------------------------------------------------------------
# Schedule constructors
# ---------------------------------------------------------------------------

def make_linear_schedule(num_timesteps: int, num_steps: int) -> List[int]:
    """
    Uniform spacing across [0, T-1].

        tᵢ = (T-1) · i / (N-1)   for i = 0 … N-1

    This is the default DDIM schedule from Song et al. 2020.
    """
    num_steps = min(num_steps, num_timesteps)
    raw = torch.linspace(0.0, float(num_timesteps - 1), num_steps)
    return _build(raw, num_timesteps)


def make_quadratic_schedule(num_timesteps: int, num_steps: int) -> List[int]:
    """
    Quadratic (x²) spacing — allocates more steps near x₀ (low t).

        tᵢ = (T-1) · (i / (N-1))²

    Rationale: in SAR despeckling, fine speckle structure emerges in the
    last denoising steps; quadratic spacing concentrates compute there.
    """
    num_steps = min(num_steps, num_timesteps)
    frac = torch.linspace(0.0, 1.0, num_steps)
    raw  = (frac ** 2) * (num_timesteps - 1)
    return _build(raw, num_timesteps)


def make_cubic_schedule(num_timesteps: int, num_steps: int) -> List[int]:
    """
    Cubic (x³) spacing — even denser near x₀ than quadratic.

        tᵢ = (T-1) · (i / (N-1))³
    """
    num_steps = min(num_steps, num_timesteps)
    frac = torch.linspace(0.0, 1.0, num_steps)
    raw  = (frac ** 3) * (num_timesteps - 1)
    return _build(raw, num_timesteps)


def make_cosine_schedule(num_timesteps: int, num_steps: int) -> List[int]:
    """
    Cosine spacing — denser near both endpoints of the chain.

        tᵢ = (T-1) · ½(1 - cos(π · i / (N-1)))

    Inspired by the improved-DDPM cosine beta schedule
    (Nichol & Dhariwal 2021).
    """
    num_steps = min(num_steps, num_timesteps)
    if num_steps == 1:
        return [num_timesteps - 1, 0]
    i    = torch.arange(num_steps, dtype=torch.float32)
    frac = 0.5 * (1.0 - torch.cos(math.pi * i / (num_steps - 1)))
    raw  = frac * (num_timesteps - 1)
    return _build(raw, num_timesteps)


def make_logarithmic_schedule(num_timesteps: int, num_steps: int) -> List[int]:
    """
    Logarithmic spacing — dense near high-t (early denoising stages).

        tᵢ = (T-1) · log(1 + i) / log(N)

    Useful when stochasticity is highest at large t and more refinement
    passes are desired during the initial coarse denoising phase.
    """
    num_steps = min(num_steps, num_timesteps)
    i    = torch.arange(num_steps, dtype=torch.float32)
    frac = torch.log1p(i) / math.log(num_steps + 1)
    raw  = frac * (num_timesteps - 1)
    return _build(raw, num_timesteps)


def make_sigmoid_schedule(
    num_timesteps: int,
    num_steps: int,
    k: float = 8.0,
) -> List[int]:
    """
    Sigmoid (S-curve) spacing — dense in the middle of the chain.

        tᵢ = (T-1) · [σ(k·(i/N - 0.5)) - σ(-k/2)] / [σ(k/2) - σ(-k/2)]

    Normalisation ensures the range covers exactly [0, T-1].
    Increasing k sharpens the transition; k=0 recovers linear spacing.

    Parameters
    ----------
    k : float
        Sigmoid sharpness.  Default 8.0 gives a pronounced S-curve.
    """
    num_steps = min(num_steps, num_timesteps)
    i    = torch.arange(num_steps, dtype=torch.float32)
    x    = k * (i / max(num_steps - 1, 1) - 0.5)
    sig  = torch.sigmoid(x)
    lo   = torch.sigmoid(torch.tensor(-k / 2.0))
    hi   = torch.sigmoid(torch.tensor(+k / 2.0))
    frac = (sig - lo) / (hi - lo + 1e-8)
    raw  = frac * (num_timesteps - 1)
    return _build(raw, num_timesteps)


def make_exponential_schedule(
    num_timesteps: int,
    num_steps: int,
    alpha: float = 3.0,
) -> List[int]:
    """
    Exponential spacing — dense near high-t with configurable decay.

        tᵢ = (T-1) · [exp(α · i/N) - 1] / [exp(α) - 1]

    Large alpha: most steps cluster near t=T (heavy early denoising).
    alpha → 0:   recovers linear spacing.

    Rationale for SAR: in high-noise regimes the largest structural
    information emerges at high t; allocating more steps there can improve
    coarse-scale reconstruction at the cost of fewer fine-detail passes.

    Parameters
    ----------
    alpha : float
        Exponential decay rate.  Default 3.0.  Must be > 0.
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha must be > 0 for exponential schedule, got {alpha}")
    num_steps = min(num_steps, num_timesteps)
    i    = torch.arange(num_steps, dtype=torch.float32)
    frac_raw = (torch.exp(alpha * i / max(num_steps - 1, 1)) - 1.0)
    frac_raw = frac_raw / (math.exp(alpha) - 1.0 + 1e-8)
    raw  = frac_raw * (num_timesteps - 1)
    return _build(raw, num_timesteps)


def make_hybrid_schedule(num_timesteps: int, num_steps: int) -> List[int]:
    """
    Hybrid linear + cosine — linear in the first half of the chain,
    cosine in the second half.

    Rationale for SAR: linear spacing at high t where the signal is
    dominated by large-scale noise; cosine (denser at endpoints) in the
    second half to refine both mid-frequency texture and fine speckle.

    The split point is t = T/2.  Each half receives ⌊N/2⌋ steps; the
    ceiling goes to the cosine half for fine-detail budget.
    """
    num_steps = min(num_steps, num_timesteps)
    half_T = num_timesteps // 2

    n_lin = num_steps // 2
    n_cos = num_steps - n_lin

    # Linear: upper half [T/2, T-1]
    if n_lin > 0:
        raw_lin = torch.linspace(float(half_T), float(num_timesteps - 1), n_lin)
    else:
        raw_lin = torch.tensor([], dtype=torch.float32)

    # Cosine: lower half [0, T/2]
    if n_cos > 0:
        i    = torch.arange(n_cos, dtype=torch.float32)
        frac = 0.5 * (1.0 - torch.cos(math.pi * i / max(n_cos - 1, 1)))
        raw_cos = frac * float(half_T)
    else:
        raw_cos = torch.tensor([], dtype=torch.float32)

    raw = torch.cat([raw_lin, raw_cos])
    return _build(raw, num_timesteps)


def make_custom_schedule(
    num_timesteps: int,
    timesteps: Sequence[int],
) -> List[int]:
    """
    User-supplied explicit timestep list.

    Validates that every value is in [0, T-1], deduplicates, sorts
    descending, and inserts T-1 / 0 if absent.

    Example
    -------
    >>> make_custom_schedule(1000,
    ...     [999, 900, 800, 600, 400, 200, 100, 50, 0])
    [999, 900, 800, 600, 400, 200, 100, 50, 0]
    """
    raw = torch.tensor(list(timesteps), dtype=torch.long)
    oob = ((raw < 0) | (raw >= num_timesteps)).nonzero(as_tuple=False)
    if oob.numel() > 0:
        bad = raw[oob.squeeze(-1)].tolist()
        raise ValueError(
            f"Custom schedule contains out-of-range timesteps "
            f"(T={num_timesteps}): {bad}"
        )
    return _build(raw.float(), num_timesteps)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCHEDULE_REGISTRY: Dict[str, Callable[[int, int], List[int]]] = {
    "linear":      make_linear_schedule,
    "quadratic":   make_quadratic_schedule,
    "cubic":       make_cubic_schedule,
    "cosine":      make_cosine_schedule,
    "logarithmic": make_logarithmic_schedule,
    "log":         make_logarithmic_schedule,      # alias
    "sigmoid":     make_sigmoid_schedule,
    "exponential": make_exponential_schedule,
    "exp":         make_exponential_schedule,      # alias
    "hybrid":      make_hybrid_schedule,
}

# Preset aliases → (schedule_type, num_steps)
_PRESETS: Dict[str, tuple] = {
    "ddim10":   ("linear", 10),
    "ddim25":   ("linear", 25),
    "ddim50":   ("linear", 50),
    "ddim100":  ("linear", 100),
    "ddim250":  ("linear", 250),
    "ddim500":  ("linear", 500),
    "ddim1000": ("linear", 1000),
}


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_schedule(
    schedule_type: str,
    num_timesteps: int,
    num_steps: int,
    custom_timesteps: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    Build and validate a timestep schedule.

    Parameters
    ----------
    schedule_type : str
        Key in SCHEDULE_REGISTRY, a preset alias, or "custom".
    num_timesteps : int
        Total diffusion timesteps T (e.g. 1000).
    num_steps : int
        Desired reverse-chain length.  Ignored for presets and "custom".
    custom_timesteps : Optional[Sequence[int]]
        Required when schedule_type == "custom".

    Returns
    -------
    List[int]
        Strictly descending, unique, with schedule[0]==T-1, schedule[-1]==0.

    Raises
    ------
    ValueError
        Unknown schedule_type, missing custom_timesteps, or out-of-range
        values in custom_timesteps.
    """
    # Preset aliases
    if schedule_type in _PRESETS:
        stype, nsteps = _PRESETS[schedule_type]
        sched = SCHEDULE_REGISTRY[stype](num_timesteps, nsteps)
        _validate(sched, num_timesteps)
        return sched

    # Custom
    if schedule_type == "custom":
        if custom_timesteps is None:
            raise ValueError(
                "schedule_type='custom' requires the custom_timesteps argument."
            )
        sched = make_custom_schedule(num_timesteps, custom_timesteps)
        _validate(sched, num_timesteps)
        return sched

    # Registry lookup
    fn = SCHEDULE_REGISTRY.get(schedule_type)
    if fn is None:
        raise ValueError(
            f"Unknown schedule_type '{schedule_type}'. "
            f"Valid: {sorted(SCHEDULE_REGISTRY)} "
            f"+ presets: {sorted(_PRESETS)} + 'custom'."
        )
    sched = fn(num_timesteps, num_steps)
    _validate(sched, num_timesteps)
    return sched


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def schedule_hash(schedule: List[int]) -> str:
    """SHA-256 hex digest (16 chars) of the schedule for reproducibility."""
    encoded = ",".join(str(t) for t in schedule).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def compression_ratio(schedule: List[int], num_timesteps: int) -> float:
    """
    Speed-up factor relative to full T-step chain.

        ratio = T / len(schedule)

    Higher values mean fewer steps (faster inference).
    """
    return num_timesteps / len(schedule)


def schedule_info(schedule: List[int], num_timesteps: int) -> dict:
    """
    Return a summary dict suitable for CSV logging and paper tables.

    Keys
    ----
    num_steps           : int
    t_max               : int      (should equal T-1)
    t_min               : int      (should equal 0)
    compression_ratio   : float
    gap_mean            : float    (mean inter-step gap)
    gap_std             : float
    gap_min             : int
    gap_max             : int
    density_q1          : int      steps in t ∈ [0,   T/4)
    density_q2          : int      steps in t ∈ [T/4, T/2)
    density_q3          : int      steps in t ∈ [T/2, 3T/4)
    density_q4          : int      steps in t ∈ [3T/4, T)
    hash                : str

    The four density_qN fields (quarter-bins) give finer granularity than
    the previous low/mid/high three-bin scheme and map directly to
    ablation tables in the paper.
    """
    ts   = torch.tensor(schedule, dtype=torch.float32)
    gaps = ts[:-1] - ts[1:]   # always positive due to strict decrease
    T    = float(num_timesteps)

    q1 = int(((ts >= 0)           & (ts < T * 0.25)).sum().item())
    q2 = int(((ts >= T * 0.25)    & (ts < T * 0.50)).sum().item())
    q3 = int(((ts >= T * 0.50)    & (ts < T * 0.75)).sum().item())
    q4 = int(((ts >= T * 0.75)    & (ts < T)).sum().item())

    return {
        "num_steps":          len(schedule),
        "t_max":              int(ts[0].item()),
        "t_min":              int(ts[-1].item()),
        "compression_ratio":  round(compression_ratio(schedule, num_timesteps), 3),
        "gap_mean":           round(float(gaps.mean().item()), 2),
        "gap_std":            round(float(gaps.std().item()),  2),
        "gap_min":            int(gaps.min().item()),
        "gap_max":            int(gaps.max().item()),
        "density_q1":         q1,
        "density_q2":         q2,
        "density_q3":         q3,
        "density_q4":         q4,
        "hash":               schedule_hash(schedule),
    }
