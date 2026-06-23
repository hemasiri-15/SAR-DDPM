"""
structdiff/inference/learnable_cycle_spinning.py
=================================================
A26a: LearnableCycleSpinning — learnable softmax aggregation of
cycle-shifted diffusion outputs for SAR despeckling.

Background
----------
Cycle spinning (Coifman & Donoho, 1995) is a translation-invariance
technique that applies a denoiser to *N* shifted copies of the input,
inverse-shifts each result, then averages them.  The original
SAR-DDPM implementation (see ``inference_sar.py`` and
``inference_sar_unet.py``) uses **equal weights**::

    pred_tensor += (1.0 / N) * sample   # inside the row/col loop

which is equivalent to::

    x̂ = (1/N) Σᵢ xᵢ,   i = 1 … N

This heuristic equal-weight average ignores the fact that some shift
positions may produce lower-quality estimates (edge artefacts, look
mismatch, azimuth spectral leakage) than others.

Learnable aggregation (A26a)
----------------------------
A26a replaces the fixed 1/N coefficients with **learned softmax
weights**.  Introduce trainable scalar logits a₁ … aₙ and compute::

    wᵢ = exp(aᵢ / τ) / Σⱼ exp(aⱼ / τ)

where τ > 0 is a temperature hyperparameter (default 1.0).  The
final prediction is::

    x̂ = Σᵢ wᵢ xᵢ

subject to wᵢ > 0 and Σᵢ wᵢ = 1.

Initialization guarantee
------------------------
When ``init_mode="uniform"`` (the default), all logits are
initialised to zero::

    shift_logits = [0, 0, …, 0]  →  softmax = [1/N, 1/N, …, 1/N]

This means **at step 0 A26a reproduces the original SAR-DDPM
cycle-spinning average exactly**, so any A12 / A13 checkpoint loads
cleanly and training begins from the known-good heuristic baseline.

Checkpoint compatibility
------------------------
``LearnableCycleSpinning`` is a new module with no analog in prior
stages (A1–A13).  It adds a single ``nn.Parameter`` (``shift_logits``,
shape ``[num_shifts]``).  When loading a pre-A26a checkpoint use::

    model.load_state_dict(checkpoint, strict=False)

Only the newly introduced parameter ``shift_logits`` will be missing.
The missing key will be silently ignored and the freshly initialised
(all-zeros) logits will be kept, preserving the original averaging
behaviour.

Future roadmap
--------------
This module is the base for the full A26 series:

* **A26b** — Adaptive Weight Prediction: replace scalar logits with a
  lightweight network that predicts per-shift weights conditioned on
  the input image.
* **A26c** — Spatial Attention Fusion: extend to per-pixel weights
  [N, B, 1, H, W] for spatially adaptive blending.
* **A26d** — Wavelet-Aware Fusion: weight derivation in the wavelet
  domain to exploit the LL/LH/HL/HH subband structure from A12.
* **A26e** — Confidence-Guided Fusion: use the diffusion model's
  predicted variance as a per-shift confidence signal.
* **A26f** — Transformer Fusion: cross-attention over the stack of N
  shifted outputs.
* **A26g** — Learnable Shift Coordinates: jointly learn the (row, col)
  shift grid rather than using a fixed uniform grid.
* **A26h** — Hierarchical Cycle Spinning: nested coarse + fine shift
  pyramids with independent learnable weight sets.
* **A26i** — Full Adaptive Cycle-Spinning Transformer: integrates
  A26b-h into a unified transformer-based aggregation module.
* **A26j** — Bayesian Cycle-Spinning: model shift weights as a
  Dirichlet distribution and estimate uncertainty over aggregation
  weights.  Enables principled confidence intervals over the fused
  prediction and is suitable for journal-level uncertainty
  quantification in SAR despeckling.
* **A26k** — Meta-Learned Cycle Spinning: learn shift aggregation
  policies across datasets, allowing zero-shot transfer to unseen SAR
  sensor configurations and domains.
* **A26l** — Reinforcement-Learned Shift Selection: use a policy
  network to decide adaptively which shifts to evaluate, reducing
  inference cost while preserving despeckling quality.
* **A26m** — Diffusion-Attention Aggregation: use timestep-dependent
  weights over shift outputs, exploiting the diffusion trajectory to
  modulate aggregation strength at each noise level.
* **A26n** — Dynamic Shift Count: learn how many shifts are actually
  necessary for a given image, replacing the fixed N with an
  adaptive per-image shift budget.

The interface of this module (``forward``, ``get_weights``,
``entropy``, ``reset_parameters``) is deliberately minimal so that
all future extensions can inherit or compose from it without breaking
changes.

References
----------
Coifman, R.R. & Donoho, D.L. (1995).  Translation-Invariant
De-Noising.  *Wavelets and Statistics*, Springer.

Notes
-----
* All computation is performed in PyTorch; no NumPy, no CPU transfer,
  no in-place operations, full autograd support.
* The module is device-agnostic: ``shift_logits`` moves with
  ``model.to(device)``.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Supported initialisation modes for ``shift_logits``.
_VALID_INIT_MODES: frozenset = frozenset({"uniform", "random", "manual"})

#: Standard deviation used when ``init_mode="random"``.
#: Set to 1e-3 so softmax weights start almost exactly uniform,
#: diverging from the SAR-DDPM heuristic baseline by only a tiny amount
#: and thereby maximising checkpoint compatibility.
_RANDOM_INIT_STD: float = 1e-3

#: Epsilon added inside the entropy logarithm for numerical stability.
#: Must satisfy _LOG_EPS << 1/N for any practical N.
_LOG_EPS: float = 1e-8


# ---------------------------------------------------------------------------
# LearnableCycleSpinning
# ---------------------------------------------------------------------------


class LearnableCycleSpinning(nn.Module):
    """Learnable softmax aggregation of cycle-shifted diffusion outputs.

    Replaces the fixed equal-weight averaging used in the original
    SAR-DDPM cycle-spinning implementation with a set of trainable
    scalar logits.  The *N* logits are passed through a temperature-
    scaled softmax to obtain a valid probability simplex, which is then
    used as a convex combination of the *N* inverse-shifted predictions.

    At initialisation with ``init_mode="uniform"`` (the default), all
    logits are zero and the module reproduces the original SAR-DDPM
    average exactly, guaranteeing backward compatibility with any
    pre-A26a checkpoint loaded with ``strict=False``.

    Parameters
    ----------
    num_shifts:
        Total number of cycle-spin shifts *N*.  Must be ≥ 1.
        Corresponds to the number of (row, col) shift pairs in the
        nested loop of the existing SAR-DDPM inference code.
    init_mode:
        Controls the initial state of ``shift_logits``.

        ``"uniform"`` (default)
            All logits set to zero → softmax weights = 1/N.
            Exactly reproduces the original heuristic average.
        ``"random"``
            Logits drawn from ``Normal(0, 1e-3)`` → weights near-
            uniform but with a non-trivial gradient at step 0.
        ``"manual"``
            Logits are set from ``manual_logits`` (which must be
            supplied).  The caller controls the exact starting point.

        Raises ``ValueError`` for any other string.
    temperature:
        Softmax temperature τ > 0.  Lower values sharpen the weight
        distribution (winner-takes-most); higher values flatten it.
        Default 1.0 (standard softmax).  Must be strictly positive.
    manual_logits:
        Optional tensor of shape ``[num_shifts]`` used only when
        ``init_mode="manual"``.  Copied into ``shift_logits`` during
        ``reset_parameters()``.  Ignored for all other modes.

    Attributes
    ----------
    num_shifts : int
        Number of cycle-spin shifts registered at construction.
    temperature : float
        Softmax temperature used in ``get_weights()``.
    init_mode : str
        Initialisation mode used for ``shift_logits``.
    manual_logits : torch.Tensor or None
        Stored reference for ``init_mode="manual"`` re-initialisation.
    shift_logits : nn.Parameter
        Trainable logit vector of shape ``[num_shifts]``.
        Registered as an ``nn.Parameter`` so it participates in
        gradient updates and is saved / restored by ``state_dict()``.

    Examples
    --------
    >>> import torch
    >>> from structdiff.inference.learnable_cycle_spinning import (
    ...     LearnableCycleSpinning,
    ... )
    >>> lcs = LearnableCycleSpinning(num_shifts=9)
    >>> lcs.num_shifts
    9
    >>> # Uniform init → weights all equal 1/9
    >>> w = lcs.get_weights()
    >>> w.shape
    torch.Size([9])
    >>> bool(torch.allclose(w, torch.full((9,), 1.0 / 9)))
    True

    >>> # forward with return_weights=False
    >>> outputs = [torch.randn(2, 3, 64, 64) for _ in range(9)]
    >>> fused = lcs(outputs)
    >>> fused.shape
    torch.Size([2, 3, 64, 64])

    >>> # forward with return_weights=True
    >>> fused, weights = lcs(outputs, return_weights=True)
    >>> fused.shape
    torch.Size([2, 3, 64, 64])
    >>> weights.shape
    torch.Size([9])
    """

    def __init__(
        self,
        num_shifts: int,
        init_mode: str = "uniform",
        temperature: float = 1.0,
        manual_logits: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        # ----------------------------------------------------------------
        # Input validation
        # ----------------------------------------------------------------
        if not isinstance(num_shifts, int) or num_shifts < 1:
            raise ValueError(
                f"num_shifts must be a positive integer, got {num_shifts!r}."
            )
        if init_mode not in _VALID_INIT_MODES:
            raise ValueError(
                f"init_mode must be one of {sorted(_VALID_INIT_MODES)}, "
                f"got {init_mode!r}."
            )
        if temperature <= 0.0:
            raise ValueError(
                f"temperature must be strictly positive, got {temperature}."
            )

        # ----------------------------------------------------------------
        # Attributes
        # ----------------------------------------------------------------
        self.num_shifts: int = num_shifts
        self.temperature: float = temperature
        self.init_mode: str = init_mode
        self.manual_logits: Optional[torch.Tensor] = manual_logits

        # ----------------------------------------------------------------
        # Learnable logits: shape [num_shifts].
        #
        # Registered as nn.Parameter so that:
        #   - They appear in model.parameters() and receive gradients.
        #   - They are included in state_dict() for checkpoint saving.
        #   - model.to(device) moves them to the correct device.
        #
        # Initialised below by reset_parameters() to keep the
        # initialisation logic in one place.
        # ----------------------------------------------------------------
        self.shift_logits: nn.Parameter = nn.Parameter(
            torch.zeros(num_shifts)
        )
        self.reset_parameters()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        """Restore ``shift_logits`` to the state specified by ``init_mode``.

        Supported modes
        ---------------
        ``"uniform"``
            Fill with zeros.  Produces softmax weights of exactly 1/N,
            reproducing the original SAR-DDPM equal-weight average.
        ``"random"``
            Sample from ``Normal(0, 1e-3)``.  Weights start near-
            uniform but break symmetry immediately, which can accelerate
            learning if the caller does not want to start from the
            heuristic baseline.
        ``"manual"``
            Copy ``self.manual_logits`` into ``shift_logits``, moving
            it to the correct device and dtype automatically.

        Raises
        ------
        ValueError
            If ``self.init_mode`` is not one of the supported strings.
        ValueError
            If ``init_mode="manual"`` and ``self.manual_logits`` is
            ``None``.
        ValueError
            If ``init_mode="manual"`` and ``self.manual_logits`` has a
            shape different from ``[self.num_shifts]``.

        Examples
        --------
        >>> lcs = LearnableCycleSpinning(4, init_mode="random")
        >>> lcs.reset_parameters()   # re-draw random init
        >>> w = lcs.get_weights()
        >>> bool(w.shape == torch.Size([4]))
        True
        """
        with torch.no_grad():
            if self.init_mode == "uniform":
                # zeros → softmax = [1/N, …, 1/N]
                self.shift_logits.fill_(0.0)

            elif self.init_mode == "random":
                nn.init.normal_(self.shift_logits, mean=0.0, std=_RANDOM_INIT_STD)

            elif self.init_mode == "manual":
                if self.manual_logits is None:
                    raise ValueError(
                        "manual_logits must be supplied when init_mode='manual'."
                    )
                if self.manual_logits.shape != self.shift_logits.shape:
                    raise ValueError(
                        f"manual_logits shape mismatch: expected "
                        f"{self.shift_logits.shape}, "
                        f"got {self.manual_logits.shape}."
                    )
                self.shift_logits.copy_(
                    self.manual_logits.to(
                        device=self.shift_logits.device,
                        dtype=self.shift_logits.dtype,
                    )
                )

            else:
                # Unreachable for a correctly initialised instance.
                raise ValueError(
                    f"Unsupported init_mode {self.init_mode!r}.  "
                    f"Must be one of {sorted(_VALID_INIT_MODES)}."
                )

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def get_weights(self) -> torch.Tensor:
        """Compute the softmax aggregation weights from the current logits.

        Returns
        -------
        torch.Tensor
            Shape ``[num_shifts]``, dtype ``float32``.
            All entries are strictly positive and sum to 1.0.
            Retains the autograd graph; never detached.

        Raises
        ------
        ValueError
            If ``self.temperature`` is not strictly positive.  Checked
            here in addition to ``__init__`` to guard against external
            mutation of the attribute.

        Notes
        -----
        The temperature τ controls the sharpness of the distribution::

            wᵢ = exp(aᵢ / τ) / Σⱼ exp(aⱼ / τ)

        * τ → 0⁺  :  winner-takes-all (argmax).
        * τ = 1.0  :  standard softmax.
        * τ → ∞    :  uniform distribution (1/N).

        Examples
        --------
        >>> lcs = LearnableCycleSpinning(4)
        >>> w = lcs.get_weights()
        >>> bool(torch.allclose(w.sum(), torch.tensor(1.0)))
        True
        >>> bool((w > 0).all())
        True
        """
        if self.temperature <= 0.0:
            raise ValueError(
                f"temperature must be strictly positive, got {self.temperature}."
            )
        return F.softmax(self.shift_logits / self.temperature, dim=0)

    # ------------------------------------------------------------------
    # Entropy
    # ------------------------------------------------------------------

    def entropy(self) -> torch.Tensor:
        """Compute the Shannon entropy of the current weight distribution.

        Returns the scalar tensor::

            H = -Σᵢ wᵢ · log(wᵢ)

        where wᵢ are the softmax weights from ``get_weights()``.

        Entropy is maximised (= log N) when all weights are equal (1/N)
        and is zero when the distribution is a delta (one weight = 1).
        It can be used as a regularisation term in the training loss to
        encourage or discourage peaked weight distributions depending on
        the sign of the regularisation coefficient.

        Returns
        -------
        torch.Tensor
            Scalar tensor (shape ``[]``), dtype ``float32``.
            Retains the autograd graph; can be added directly to a loss.

        Examples
        --------
        >>> import math, torch
        >>> lcs = LearnableCycleSpinning(8)
        >>> h = lcs.entropy()
        >>> bool(abs(h.item() - math.log(8)) < 1e-5)
        True
        """
        weights: torch.Tensor = self.get_weights()
        # Clamp inside log for numerical stability; _LOG_EPS << 1/N for
        # any practical N so this does not affect the gradient meaningfully.
        return -(weights * torch.log(weights.clamp(min=_LOG_EPS))).sum()

    # ------------------------------------------------------------------
    # Entropy regularizer
    # ------------------------------------------------------------------

    def entropy_regularizer(self, coefficient: float = 1.0) -> torch.Tensor:
        """Entropy regularization term for use directly in a training loss.

        Returns ``coefficient * H`` where H is the Shannon entropy of the
        current weight distribution.

        Parameters
        ----------
        coefficient:
            Scalar multiplier applied to the entropy.

            * Positive value  → maximise entropy → encourage uniform weights.
            * Negative value  → minimise entropy → encourage sparse weights.

            Default 1.0.

        Returns
        -------
        torch.Tensor
            Scalar tensor, retains the autograd graph.
            Can be added directly to a training loss::

                loss = diffusion_loss + lcs.entropy_regularizer(lambda_ent)

        Notes
        -----
        Useful for A26f (Transformer Fusion) and A26i (Full Adaptive
        Cycle-Spinning Transformer) where controlling weight sparsity
        is important for training stability.

        Examples
        --------
        >>> lcs = LearnableCycleSpinning(4)
        >>> reg = lcs.entropy_regularizer(coefficient=0.01)
        >>> reg.shape
        torch.Size([])
        """
        return coefficient * self.entropy()

    # ------------------------------------------------------------------
    # Effective number of shifts
    # ------------------------------------------------------------------

    def effective_num_shifts(self) -> torch.Tensor:
        """Compute the effective number of active shifts.

        Defined as::

            N_eff = exp(H)

        where H is the Shannon entropy of the current weight
        distribution (``self.entropy()``).

        Returns
        -------
        torch.Tensor
            Scalar tensor.

            * Uniform weights (all equal 1/N):  N_eff = num_shifts.
            * One dominant weight (≈ 1):         N_eff → 1.

        Notes
        -----
        N_eff is a standard information-theoretic measure of
        distribution peakedness, analogous to the perplexity of a
        language model.  It is useful for monitoring whether training
        is collapsing to a single shift position or maintaining a
        spread distribution.

        Retains the autograd graph; can be used as a loss term.

        Examples
        --------
        >>> import math, torch
        >>> lcs = LearnableCycleSpinning(8)
        >>> n_eff = lcs.effective_num_shifts()
        >>> bool(abs(n_eff.item() - 8.0) < 1e-4)
        True
        """
        return torch.exp(self.entropy())

    # ------------------------------------------------------------------
    # Weight variance
    # ------------------------------------------------------------------

    def weight_variance(self) -> torch.Tensor:
        """Compute the variance of the weight distribution.

        Returns
        -------
        torch.Tensor
            Scalar tensor (population variance, ``unbiased=False``).
            Retains the autograd graph.

        Notes
        -----
        High variance indicates that the learned weights are peaked
        (a few shifts dominate).  Low variance indicates a near-uniform
        distribution.  Useful for analysis and for constructing
        regularisation terms that penalise extreme peaking.

        Examples
        --------
        >>> lcs = LearnableCycleSpinning(4)
        >>> v = lcs.weight_variance()
        >>> v.shape
        torch.Size([])
        >>> bool(v.item() >= 0.0)
        True
        """
        weights: torch.Tensor = self.get_weights()
        return weights.var(unbiased=False)

    # ------------------------------------------------------------------
    # KL divergence to uniform
    # ------------------------------------------------------------------

    def kl_to_uniform(self) -> torch.Tensor:
        """Compute the KL divergence from the current distribution to uniform.

        Returns::

            KL(w ‖ u) = Σᵢ wᵢ · [log(wᵢ) - log(1/N)]
                       = Σᵢ wᵢ · log(N · wᵢ)

        where uᵢ = 1/N is the uniform distribution and wᵢ = get_weights()[i].

        Returns
        -------
        torch.Tensor
            Scalar tensor ≥ 0.  Zero iff ``w`` is exactly uniform.
            Retains the autograd graph; can be added to a training loss
            to penalise deviation from equal-weight averaging.

        Notes
        -----
        KL(w ‖ u) = log(N) - H(w), so minimising this term is equivalent
        to maximising entropy.  The explicit form is provided because it
        gives a physically interpretable magnitude: it is zero at the
        SAR-DDPM baseline and grows as weights become more peaked.

        This metric is particularly useful for ablation studies and
        publications comparing learnable vs. fixed aggregation.

        Examples
        --------
        >>> lcs = LearnableCycleSpinning(4)
        >>> kl = lcs.kl_to_uniform()
        >>> kl.shape
        torch.Size([])
        >>> bool(abs(kl.item()) < 1e-6)   # uniform init → KL = 0
        True
        """
        weights: torch.Tensor = self.get_weights()
        uniform: torch.Tensor = self.uniform_weights()
        return (
            weights
            * (
                torch.log(weights.clamp(min=_LOG_EPS))
                - torch.log(uniform)
            )
        ).sum()

    # ------------------------------------------------------------------
    # Gradient control
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Disable gradient updates for ``shift_logits``.

        After calling ``freeze()``, ``shift_logits`` will not receive
        gradients during backpropagation.  The module still participates
        in the forward pass and produces valid weights; only the
        parameter update is suppressed.

        Useful for ablation studies where the learnable aggregation
        should be held fixed at its current values (e.g. after
        convergence, or when evaluating the equal-weight baseline by
        freezing at uniform initialisation).

        See Also
        --------
        unfreeze : Re-enable gradient updates.

        Examples
        --------
        >>> lcs = LearnableCycleSpinning(4)
        >>> lcs.freeze()
        >>> lcs.shift_logits.requires_grad
        False
        """
        self.shift_logits.requires_grad_(False)

    def unfreeze(self) -> None:
        """Enable gradient updates for ``shift_logits``.

        Restores gradient computation after a prior call to ``freeze()``.
        Newly constructed modules have ``requires_grad=True`` by default;
        calling ``unfreeze()`` on them is a no-op.

        See Also
        --------
        freeze : Disable gradient updates.

        Examples
        --------
        >>> lcs = LearnableCycleSpinning(4)
        >>> lcs.freeze()
        >>> lcs.unfreeze()
        >>> lcs.shift_logits.requires_grad
        True
        """
        self.shift_logits.requires_grad_(True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _weights_no_grad(self) -> torch.Tensor:
        """Compute softmax weights without retaining the autograd graph.

        Used internally by logging methods (``weight_statistics``,
        ``summary``, ``save_statistics``) that need the weight values
        purely for inspection, not for gradient computation.  A single
        call here avoids redundant ``softmax`` evaluations across
        multiple logging accessors.

        Returns
        -------
        torch.Tensor
            Shape ``[num_shifts]``, detached from the autograd graph.
        """
        with torch.no_grad():
            return self.get_weights()

    def uniform_weights(self) -> torch.Tensor:
        """Return the uniform weight vector 1/N on the correct device and dtype.

        Creates a tensor of shape ``[num_shifts]`` where every entry equals
        ``1 / num_shifts``, placed on the same device and with the same
        dtype as ``shift_logits``.

        Centralising this construction avoids the repeated
        ``torch.full_like`` pattern that would otherwise appear in
        ``kl_to_uniform``, ``js_to_uniform``, and ``is_uniform``.

        Returns
        -------
        torch.Tensor
            Shape ``[num_shifts]``, all entries equal to ``1/num_shifts``.
            Not connected to the autograd graph.

        Examples
        --------
        >>> lcs = LearnableCycleSpinning(4)
        >>> u = lcs.uniform_weights()
        >>> u.shape
        torch.Size([4])
        >>> bool(torch.allclose(u, torch.full((4,), 0.25)))
        True
        """
        return torch.full(
            (self.num_shifts,),
            1.0 / self.num_shifts,
            device=self.shift_logits.device,
            dtype=self.shift_logits.dtype,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def weight_statistics(self) -> dict[str, float]:
        """Return useful statistics about the weight distribution for logging.

        Computes all statistics from a single ``get_weights()`` call
        to avoid redundant softmax evaluations.

        Returns
        -------
        dict
            A dictionary with the following ``str`` keys and ``float`` values:

            ``"entropy"``
                Shannon entropy H = -Σ wᵢ log(wᵢ).
            ``"effective_num_shifts"``
                N_eff = exp(H).
            ``"max_weight"``
                Maximum weight value.
            ``"min_weight"``
                Minimum weight value.
            ``"std_weight"``
                Standard deviation of the weight distribution
                (population std, ``unbiased=False``).

        Notes
        -----
        All values are detached scalars; this method is intended for
        logging and does not retain the autograd graph.

        Examples
        --------
        >>> import math
        >>> lcs = LearnableCycleSpinning(4)
        >>> stats = lcs.weight_statistics()
        >>> set(stats.keys()) == {
        ...     "entropy", "effective_num_shifts",
        ...     "max_weight", "min_weight", "std_weight"
        ... }
        True
        >>> bool(abs(stats["effective_num_shifts"] - 4.0) < 1e-4)
        True
        """
        weights: torch.Tensor = self._weights_no_grad()
        entropy_val: torch.Tensor = -(
            weights * torch.log(weights.clamp(min=_LOG_EPS))
        ).sum()

        return {
            "entropy": float(entropy_val.item()),
            "effective_num_shifts": float(torch.exp(entropy_val).item()),
            "max_weight": float(weights.max().item()),
            "min_weight": float(weights.min().item()),
            "std_weight": float(weights.std(unbiased=False).item()),
        }

    def summary(self) -> dict[str, float]:
        """Return a comprehensive diagnostic summary of the module state.

        Extends ``weight_statistics()`` with KL divergence and weight
        variance for a complete picture of the distribution.  All
        quantities are computed from a single ``get_weights()`` call.

        Returns
        -------
        dict
            A dictionary with the following ``str`` keys and ``float`` values:

            ``"entropy"``
                Shannon entropy H = -Σ wᵢ log(wᵢ).
            ``"effective_num_shifts"``
                N_eff = exp(H).
            ``"kl_to_uniform"``
                KL(w ‖ uniform) = log(N) - H.
            ``"max_weight"``
                Maximum weight value.
            ``"min_weight"``
                Minimum weight value.
            ``"weight_variance"``
                Population variance of the weight distribution.

        Notes
        -----
        All values are detached scalars; this method is intended for
        logging and does not retain the autograd graph.

        Examples
        --------
        >>> lcs = LearnableCycleSpinning(4)
        >>> s = lcs.summary()
        >>> set(s.keys()) == {
        ...     "entropy", "effective_num_shifts", "kl_to_uniform",
        ...     "max_weight", "min_weight", "weight_variance"
        ... }
        True
        >>> bool(abs(s["kl_to_uniform"]) < 1e-6)   # uniform init → KL = 0
        True
        """
        weights: torch.Tensor = self._weights_no_grad()
        entropy_val: torch.Tensor = -(
            weights * torch.log(weights.clamp(min=_LOG_EPS))
        ).sum()
        uniform: torch.Tensor = self.uniform_weights()
        kl_val: torch.Tensor = (
            weights
            * (
                torch.log(weights.clamp(min=_LOG_EPS))
                - torch.log(uniform)
            )
        ).sum()

        return {
            "entropy": float(entropy_val.item()),
            "effective_num_shifts": float(torch.exp(entropy_val).item()),
            "kl_to_uniform": float(kl_val.item()),
            "max_weight": float(weights.max().item()),
            "min_weight": float(weights.min().item()),
            "weight_variance": float(weights.var(unbiased=False).item()),
        }

    # ------------------------------------------------------------------
    # Index and range utilities
    # ------------------------------------------------------------------

    def max_weight_index(self) -> int:
        """Return the index of the shift position with the highest weight.

        Returns
        -------
        int
            Index ``i`` such that ``get_weights()[i]`` is maximal.

        Notes
        -----
        Useful for papers: identifies which cycle-spin shift the model
        has learned to trust most.  At uniform initialisation this
        returns 0 (argmax of a flat distribution picks the first entry).

        Examples
        --------
        >>> lcs = LearnableCycleSpinning(4)
        >>> lcs.max_weight_index()
        0
        """
        return int(self._weights_no_grad().argmax().item())

    def weight_range(self) -> tuple[float, float]:
        """Return ``(min_weight, max_weight)`` of the current distribution.

        Returns
        -------
        tuple of (float, float)
            ``(min_weight, max_weight)``.  Both values are in ``(0, 1]``
            and sum with the other weights to 1.

        Notes
        -----
        The spread ``max - min`` is a simple one-number summary of how
        peaked the distribution is; it is zero at uniform initialisation
        and grows as training converges on a preferred shift.

        Examples
        --------
        >>> lcs = LearnableCycleSpinning(4)
        >>> lo, hi = lcs.weight_range()
        >>> bool(abs(lo - 0.25) < 1e-6 and abs(hi - 0.25) < 1e-6)
        True
        """
        w: torch.Tensor = self._weights_no_grad()
        return float(w.min().item()), float(w.max().item())

    # ------------------------------------------------------------------
    # Divergence measures
    # ------------------------------------------------------------------

    def js_to_uniform(self) -> torch.Tensor:
        """Compute the Jensen-Shannon divergence between current weights and uniform.

        The Jensen-Shannon divergence (JSD) is a symmetric, bounded
        alternative to KL divergence:

            M  = (w + u) / 2
            JSD(w ‖ u) = [KL(w ‖ M) + KL(u ‖ M)] / 2

        where ``u`` is the uniform distribution (1/N) and ``M`` is the
        mixture.  JSD lies in ``[0, log(2)]`` regardless of the support,
        making it more suitable than KL divergence for plots and
        comparisons across different values of N.

        Returns
        -------
        torch.Tensor
            Scalar tensor in ``[0, log(2)]``.  Zero iff the weight
            distribution equals uniform (which it does at initialisation).
            Retains the autograd graph.

        Notes
        -----
        Because JSD is symmetric and bounded, it produces cleaner
        training curves than KL divergence and is preferred for
        publication figures comparing aggregation strategies.

        Examples
        --------
        >>> lcs = LearnableCycleSpinning(4)
        >>> jsd = lcs.js_to_uniform()
        >>> jsd.shape
        torch.Size([])
        >>> bool(abs(jsd.item()) < 1e-6)   # uniform init → JSD = 0
        True
        """
        weights: torch.Tensor = self.get_weights()
        uniform: torch.Tensor = self.uniform_weights()
        mixture: torch.Tensor = 0.5 * (weights + uniform)
        kl_w_m: torch.Tensor = (
            weights * (
                torch.log(weights.clamp(min=_LOG_EPS))
                - torch.log(mixture.clamp(min=_LOG_EPS))
            )
        ).sum()
        kl_u_m: torch.Tensor = (
            uniform * (
                torch.log(uniform.clamp(min=_LOG_EPS))
                - torch.log(mixture.clamp(min=_LOG_EPS))
            )
        ).sum()
        return 0.5 * (kl_w_m + kl_u_m)

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------

    def is_uniform(self, atol: float = 1e-6) -> bool:
        """Return ``True`` if the weight distribution is effectively uniform.

        Parameters
        ----------
        atol:
            Absolute tolerance passed to ``torch.allclose``.
            Default 1e-6 is appropriate for float32 precision.

        Returns
        -------
        bool
            ``True`` iff every weight is within ``atol`` of ``1/N``.

        Notes
        -----
        At ``init_mode="uniform"`` (all logits zero) this will return
        ``True`` exactly.  Useful as a sanity check at the start of
        training to confirm the checkpoint-compatible baseline is active.

        Examples
        --------
        >>> lcs = LearnableCycleSpinning(4)
        >>> lcs.is_uniform()
        True
        >>> import torch
        >>> lcs.shift_logits.data[0] = 5.0
        >>> lcs.is_uniform()
        False
        """
        return bool(
            torch.allclose(self._weights_no_grad(), self.uniform_weights(), atol=atol)
        )

    # ------------------------------------------------------------------
    # Checkpoint-friendly statistics snapshot
    # ------------------------------------------------------------------

    def save_statistics(self) -> dict[str, float]:
        """Return a detached statistics snapshot for checkpoint logging.

        Intended to be stored alongside a model checkpoint so that the
        state of the aggregation weights can be reconstructed from the
        log without reloading the model.

        Returns
        -------
        dict[str, float]
            A dictionary with the following keys:

            ``\"entropy\"``
                Shannon entropy H = -Σ wᵢ log(wᵢ).
            ``\"kl_to_uniform\"``
                KL(w ‖ uniform).
            ``\"weight_variance\"``
                Population variance of the weight distribution.
            ``\"max_weight\"``
                Maximum weight value.
            ``\"min_weight\"``
                Minimum weight value.
            ``\"max_weight_index\"``
                Index of the highest-weight shift position (as float
                for JSON/CSV serialisation compatibility).
            ``\"effective_num_shifts\"``
                N_eff = exp(H).

        Notes
        -----
        All values are detached scalars.  This method is intentionally
        separate from ``summary()`` so that logging code can call it
        without worrying about autograd side-effects.

        Examples
        --------
        >>> lcs = LearnableCycleSpinning(4)
        >>> s = lcs.save_statistics()
        >>> set(s.keys()) == {
        ...     "entropy", "kl_to_uniform", "weight_variance",
        ...     "max_weight", "min_weight", "max_weight_index",
        ...     "effective_num_shifts"
        ... }
        True
        """
        weights: torch.Tensor = self._weights_no_grad()
        entropy_val: float = float(
            -(weights * torch.log(weights.clamp(min=_LOG_EPS))).sum().item()
        )
        uniform: torch.Tensor = self.uniform_weights()
        kl_val: float = float(
            (
                weights
                * (
                    torch.log(weights.clamp(min=_LOG_EPS))
                    - torch.log(uniform)
                )
            ).sum().item()
        )
        return {
            "entropy": entropy_val,
            "kl_to_uniform": kl_val,
            "weight_variance": float(weights.var(unbiased=False).item()),
            "max_weight": float(weights.max().item()),
            "min_weight": float(weights.min().item()),
            "max_weight_index": float(weights.argmax().item()),
            "effective_num_shifts": float(math.exp(entropy_val)),
        }

    # ------------------------------------------------------------------
    # Temperature control
    # ------------------------------------------------------------------

    def set_temperature(self, temperature: float) -> None:
        """Update the softmax temperature in-place with validation.

        Parameters
        ----------
        temperature:
            New temperature value τ > 0.

            * Decrease τ to sharpen the weight distribution (encourage
              the model to specialise on fewer shift positions).
            * Increase τ to flatten the distribution (encourage uniform
              averaging; at τ → ∞ this recovers the SAR-DDPM baseline).

        Raises
        ------
        ValueError
            If ``temperature`` is not strictly positive.

        Notes
        -----
        Temperature annealing is a common technique for controlling
        learning dynamics in softmax-parametrised models.  This method
        will be used by A26b (Adaptive Weight Prediction) and A26f
        (Transformer Fusion) to schedule τ over training.

        Examples
        --------
        >>> lcs = LearnableCycleSpinning(4, temperature=1.0)
        >>> lcs.set_temperature(0.5)
        >>> lcs.temperature
        0.5
        >>> try:
        ...     lcs.set_temperature(-1.0)
        ... except ValueError as e:
        ...     print("caught:", e)
        caught: temperature must be strictly positive, got -1.0.
        """
        if temperature <= 0.0:
            raise ValueError(
                f"temperature must be strictly positive, got {temperature}."
            )
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Additional diagnostics and utilities
    # ------------------------------------------------------------------

    def min_weight_index(self) -> int:
        """Return the index of the shift position with the lowest weight.

        Returns
        -------
        int
            Index ``i`` such that ``get_weights()[i]`` is minimal.

        Notes
        -----
        Complements ``max_weight_index()``.  The pair
        ``(min_weight_index, max_weight_index)`` identifies which shift
        positions the model has learned to trust least and most,
        respectively — useful for ablation studies that remove or down-
        weight specific shift positions.

        Uses ``_weights_no_grad()`` because no gradient is needed for
        a diagnostic index query.

        Examples
        --------
        >>> lcs = LearnableCycleSpinning(4)
        >>> lcs.min_weight_index()   # uniform → argmin picks first entry
        0
        """
        return int(self._weights_no_grad().argmin().item())

    def entropy_per_shift(self) -> float:
        """Return the Shannon entropy normalised by the number of shifts.

        Defined as::

            H_per = H / N = (-Σᵢ wᵢ log wᵢ) / N

        where N = ``num_shifts``.

        Returns
        -------
        float
            Normalised entropy in ``[0, log(N) / N]``.

        Notes
        -----
        The raw entropy H = log(N) at uniform and therefore grows with N,
        making it difficult to compare models trained with different shift
        counts.  Dividing by N yields a per-shift contribution that is
        independent of N, enabling fair cross-configuration comparisons
        in paper tables.

        Examples
        --------
        >>> import math
        >>> lcs = LearnableCycleSpinning(8)
        >>> h_per = lcs.entropy_per_shift()
        >>> bool(abs(h_per - math.log(8) / 8) < 1e-5)
        True
        """
        return float(self.entropy().item()) / self.num_shifts

    def temperature_scaling_factor(self) -> float:
        """Return the reciprocal of the current temperature: 1 / τ.

        Returns
        -------
        float
            ``1.0 / self.temperature``.

        Notes
        -----
        The softmax computation is ``softmax(logits / τ)``, which is
        equivalent to ``softmax(logits * (1/τ))``.  Exposing the scaling
        factor explicitly makes it convenient for A26f (Transformer
        Fusion) and other future stages that pass this value to attention
        layers or learning-rate schedulers for temperature annealing.

        Examples
        --------
        >>> lcs = LearnableCycleSpinning(4, temperature=0.5)
        >>> lcs.temperature_scaling_factor()
        2.0
        """
        return 1.0 / self.temperature

    def is_frozen(self) -> bool:
        """Return ``True`` if ``shift_logits`` gradients are disabled.

        Returns
        -------
        bool
            ``True`` iff ``self.shift_logits.requires_grad`` is ``False``.

        Notes
        -----
        Useful in ablation scripts to assert module state before
        training begins::

            assert not lcs.is_frozen(), "shift_logits must be trainable"

        or to confirm that a baseline fixed-weight run is correctly
        frozen::

            lcs.freeze()
            assert lcs.is_frozen()

        See Also
        --------
        freeze : Disable gradient updates.
        unfreeze : Re-enable gradient updates.

        Examples
        --------
        >>> lcs = LearnableCycleSpinning(4)
        >>> lcs.is_frozen()
        False
        >>> lcs.freeze()
        >>> lcs.is_frozen()
        True
        """
        return not self.shift_logits.requires_grad

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        outputs: Sequence[torch.Tensor],
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Aggregate cycle-shifted diffusion outputs with learned weights.

        Parameters
        ----------
        outputs:
            A sequence of *N* tensors, one per cycle-spin shift.  Each
            tensor must have shape ``[B, C, H, W]``.

            All tensors must satisfy:

            * identical shape,
            * identical dtype,
            * identical device,
            * first dimension (batch size) must match across all tensors.

            Corresponds to the per-shift ``sample`` tensors produced
            inside the row/col loop in the existing SAR-DDPM inference
            code, *after* the inverse shift has been applied.
        return_weights:
            If ``False`` (default), return only the fused tensor.
            If ``True``, return ``(fused, weights)`` where ``weights``
            has shape ``[num_shifts]``.

        Returns
        -------
        torch.Tensor or tuple of (torch.Tensor, torch.Tensor)
            * ``return_weights=False``:  fused tensor, shape ``[B, C, H, W]``.
            * ``return_weights=True``:   ``(fused, weights)`` where
              ``fused`` has shape ``[B, C, H, W]`` and ``weights`` has
              shape ``[num_shifts]``.

        Raises
        ------
        ValueError
            If ``outputs`` is empty.
        ValueError
            If ``len(outputs) != self.num_shifts``.
        ValueError
            If any tensor is not 4-dimensional.
        ValueError
            If tensors have inconsistent shapes.
        ValueError
            If tensors have inconsistent dtypes.
        ValueError
            If tensors reside on inconsistent devices.

        Notes
        -----
        The aggregation is::

            wᵢ = softmax(shift_logits / temperature)[i]
            x̂  = Σᵢ wᵢ · xᵢ

        Weights are broadcast as ``[N, 1, 1, 1, 1]`` over the
        ``[N, B, C, H, W]`` stacked tensor, then summed along dim 0 to
        produce ``[B, C, H, W]``.  No in-place operations are used;
        the autograd graph is preserved throughout.

        When ``shift_logits`` are all zeros (the default uniform
        initialisation), wᵢ = 1/N for all *i*, and the output is
        identical to the original SAR-DDPM equal-weight average.

        Examples
        --------
        >>> import torch
        >>> from structdiff.inference.learnable_cycle_spinning import (
        ...     LearnableCycleSpinning,
        ... )
        >>> lcs = LearnableCycleSpinning(num_shifts=4)
        >>> outputs = [torch.ones(2, 1, 8, 8) * (i + 1.0) for i in range(4)]
        >>> fused = lcs(outputs)
        >>> fused.shape
        torch.Size([2, 1, 8, 8])
        >>> # Uniform weights → fused = mean([1,2,3,4]) = 2.5
        >>> bool(torch.allclose(fused, torch.full((2, 1, 8, 8), 2.5)))
        True

        >>> # return_weights=True
        >>> fused, w = lcs(outputs, return_weights=True)
        >>> w.shape
        torch.Size([4])
        >>> bool(torch.allclose(w.sum(), torch.tensor(1.0)))
        True
        """
        # ----------------------------------------------------------------
        # Validate outputs sequence
        # ----------------------------------------------------------------
        if len(outputs) == 0:
            raise ValueError(
                "outputs must be a non-empty sequence of tensors, got length 0."
            )
        if len(outputs) != self.num_shifts:
            raise ValueError(
                f"len(outputs) must equal num_shifts={self.num_shifts}, "
                f"got {len(outputs)}."
            )

        # Validate the reference tensor, then cross-validate all others.
        reference: torch.Tensor = outputs[0]

        if reference.ndim != 4:
            raise ValueError(
                f"Each output tensor must be 4-dimensional [B, C, H, W]; "
                f"outputs[0] has shape {reference.shape} (ndim={reference.ndim})."
            )

        ref_shape: torch.Size = reference.shape
        ref_dtype: torch.dtype = reference.dtype
        ref_device: torch.device = reference.device

        for idx, tensor in enumerate(outputs[1:], start=1):
            if tensor.ndim != 4:
                raise ValueError(
                    f"Each output tensor must be 4-dimensional [B, C, H, W]; "
                    f"outputs[{idx}] has shape {tensor.shape} "
                    f"(ndim={tensor.ndim})."
                )
            if tensor.shape != ref_shape:
                raise ValueError(
                    f"All output tensors must have the same shape; "
                    f"outputs[0].shape={ref_shape} but "
                    f"outputs[{idx}].shape={tensor.shape}."
                )
            if tensor.dtype != ref_dtype:
                raise ValueError(
                    f"All output tensors must have the same dtype; "
                    f"outputs[0].dtype={ref_dtype} but "
                    f"outputs[{idx}].dtype={tensor.dtype}."
                )
            if tensor.device != ref_device:
                raise ValueError(
                    f"All output tensors must reside on the same device; "
                    f"outputs[0].device={ref_device} but "
                    f"outputs[{idx}].device={tensor.device}."
                )

        # ----------------------------------------------------------------
        # Compute softmax weights
        #
        # Shape: [num_shifts]
        # ----------------------------------------------------------------
        weights: torch.Tensor = self.get_weights()  # [N]

        # ----------------------------------------------------------------
        # Stack outputs and aggregate
        #
        # Stack  : list of N × [B, C, H, W]  →  [N, B, C, H, W]
        # Weights: [N] → [N, 1, 1, 1, 1]   (broadcast over B, C, H, W)
        # Product: [N, B, C, H, W] * [N, 1, 1, 1, 1]  →  [N, B, C, H, W]
        # Sum    : dim=0  →  [B, C, H, W]
        # ----------------------------------------------------------------
        stacked: torch.Tensor = torch.stack(list(outputs), dim=0)  # [N, B, C, H, W]

        weights_broadcast: torch.Tensor = weights.view(
            self.num_shifts, 1, 1, 1, 1
        )  # [N, 1, 1, 1, 1]

        # Promote stacked dtype to match weights if necessary (e.g. fp16 inputs
        # with fp32 logits) so the multiplication is well-defined.
        # The original input dtype is saved and restored so that fp16 callers
        # receive fp16 output, preserving full compatibility with FP16 training.
        input_dtype: torch.dtype = stacked.dtype
        if stacked.dtype != weights_broadcast.dtype:
            stacked = stacked.to(weights_broadcast.dtype)

        fused: torch.Tensor = (stacked * weights_broadcast).sum(dim=0)  # [B, C, H, W]

        # Restore the caller's original dtype (e.g. fp16 → fp16).
        fused = fused.to(input_dtype)

        # ----------------------------------------------------------------
        # Return
        # ----------------------------------------------------------------
        if return_weights:
            return fused, weights
        return fused

    # ------------------------------------------------------------------
    # Module representation
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Return a concise parameter summary for ``print(module)``.

        Returns
        -------
        str
            Human-readable representation of the module's configuration,
            formatted to match the style of ``nn.Linear``, ``nn.Conv2d``,
            and the other encoders in this codebase.

        Examples
        --------
        >>> lcs = LearnableCycleSpinning(9, init_mode="uniform", temperature=0.5)
        >>> print(lcs)
        LearnableCycleSpinning(num_shifts=9, temperature=0.5, init_mode=uniform)
        """
        return (
            f"num_shifts={self.num_shifts}, "
            f"temperature={self.temperature}, "
            f"init_mode={self.init_mode}"
        )
