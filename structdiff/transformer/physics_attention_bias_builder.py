"""
PhysicsAttentionBiasBuilder: high-level orchestration module that converts
a raw structure tensor into the final physics attention bias consumed by
PhysicsAwareAttention.

Phase 3.5 of the physics-aware attention framework
----------------------------------------------------
    Phase 3.1 (done): ``PhysicsRelationBuilder`` -- interface/validation
                       only, no computation.
    Phase 3.2 (done):  ``OrientationExtractor`` -- raw structure tensor ->
                       orientation, coherence, anisotropy.
    Phase 3.3 (done):  ``PhysicsRelationMatrix`` -- pairwise orientation
                       relation matrix, [B, N, N].
    Phase 3.4 (done):  ``PhysicsBiasFusion`` -- fuses an arbitrary subset
                       of per-modality relation matrices into a single
                       additive attention bias, via learnable,
                       zero-initialized per-modality gates.
    Phase 3.5 (this file): orchestrate the three modules above into a
                       single callable that takes a raw structure tensor
                       (plus any already-computed non-orientation relation
                       matrices) and returns the final
                       ``physics_attention_bias`` ready to hand to
                       ``PhysicsAwareAttention``.

This module is standalone: it does not import from ``guided_diffusion``
or ``PhysicsAwareAttention``. Nothing in the repository imports it yet.
It performs no repository modifications -- it is a new file that only
composes ``OrientationExtractor``, ``PhysicsRelationMatrix``, and
``PhysicsBiasFusion``, all of which already work independently.

Why this module exists
-----------------------
Producing a physics attention bias from a raw structure tensor currently
requires three separate calls in a fixed order -- ``OrientationExtractor``
to get orientation/coherence, ``PhysicsRelationMatrix`` to turn those into
a pairwise relation matrix, and ``PhysicsBiasFusion`` to gate-fuse that
relation matrix (and any other available relation matrices) into the
final bias. Every call site that wants a physics bias -- today just a
smoke test, but eventually the UNet's transformer blocks -- would
otherwise need to know and correctly reproduce that three-step sequence,
including which intermediate values feed into which module and in what
order. ``PhysicsAttentionBiasBuilder`` exists purely to own that sequence
once, in one place, so every caller can instead make a single call.

Why orchestration is separated from feature extraction
--------------------------------------------------------
This module implements no physics of its own: no orientation formula, no
relation formula, no fusion formula. All of that logic already lives in,
and continues to live in, ``OrientationExtractor``,
``PhysicsRelationMatrix``, and ``PhysicsBiasFusion`` respectively, and
each of those modules remains independently testable, independently
correct, and independently owns its own validation of its own inputs.
Keeping the orchestration logic (the "what calls what, in what order,
with which outputs feeding which inputs") separate from the feature
logic (the "what is the actual formula") means a future change to, say,
the coherence formula only ever touches ``OrientationExtractor``, and a
future change to how modalities are weighted only ever touches
``PhysicsBiasFusion`` -- this file does not need to change for either,
since it never re-implements or duplicates either formula.

How this keeps the UNet simple
---------------------------------
Without this module, the UNet (or any other caller) would need to hold
references to three separate submodules and manually thread orientation,
coherence, and the orientation relation matrix between them, in the
correct order, on every forward pass. With this module, the UNet needs
only a single instance of ``PhysicsAttentionBiasBuilder`` and a single
call:

    physics_bias = self.physics_attention_bias_builder(
        struct_tensor,
        wavelet_relation,
        spectral_relation,
        confidence_relation,
    )

The internal pipeline (orientation extraction, relation matrix
construction, gated fusion) is entirely hidden behind that one call,
so the UNet's own code stays focused on how the resulting bias is used
inside attention, not on how it is produced.

How this improves maintainability
------------------------------------
Because ``PhysicsAttentionBiasBuilder`` only orchestrates -- it holds no
physics formulas and duplicates no validation logic from the modules it
wraps -- there is exactly one place in the repository that encodes the
*order* of the physics pipeline (structure tensor -> orientation/
coherence -> relation matrix -> fused bias), while the *content* of each
stage remains owned by that stage's own module. If a later phase adds a
new relation-matrix source (e.g. a wavelet-domain relation extractor),
that new module simply needs to be instantiated here and its output
passed into the existing ``wavelet_relation`` argument that
``PhysicsBiasFusion`` already accepts -- no change to
``OrientationExtractor``, ``PhysicsRelationMatrix``, or
``PhysicsBiasFusion`` is required, and no call site outside this module
needs to change either.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from structdiff.transformer.orientation_extractor import OrientationExtractor
from structdiff.transformer.physics_relation_matrix import (
    PhysicsRelationMatrix,
)
from structdiff.transformer.physics_bias_fusion import PhysicsBiasFusion


class PhysicsAttentionBiasBuilder(nn.Module):
    """Orchestrates the physics pipeline from a raw structure tensor to a
    fused physics attention bias.

    This module implements no new physics and duplicates no formulas from
    the modules it wraps. It instantiates ``OrientationExtractor``,
    ``PhysicsRelationMatrix``, and ``PhysicsBiasFusion`` internally, and
    its ``forward`` simply threads data through them in the fixed order:

        struct_tensor
            -> OrientationExtractor
            -> orientation, coherence, (anisotropy, unused downstream)
            -> PhysicsRelationMatrix
            -> orientation_relation
            -> PhysicsBiasFusion (with optional wavelet/spectral/
               confidence relation matrices)
            -> physics_attention_bias

    See the module docstring for the rationale behind this separation of
    orchestration from feature extraction.

    Input
    -----
    struct_tensor : torch.Tensor
        Raw structure-tensor components, shape ``[B, 3, H, W]``, with
        channels ordered ``(J11, J12, J22)`` -- see
        ``OrientationExtractor`` for the exact contract.
    wavelet_relation : torch.Tensor, optional
        Shape ``[B, N, N]`` where ``N = H * W``, matching the shape of
        the internally computed ``orientation_relation``. Passed straight
        through to ``PhysicsBiasFusion``.
    spectral_relation : torch.Tensor, optional
        Shape ``[B, N, N]``, matching ``orientation_relation``. Passed
        straight through to ``PhysicsBiasFusion``.
    confidence_relation : torch.Tensor, optional
        Shape ``[B, N, N]``, matching ``orientation_relation``. Passed
        straight through to ``PhysicsBiasFusion``.

    Output
    ------
    torch.Tensor
        ``physics_attention_bias``, shape ``[B, N, N]`` where
        ``N = H * W``, ready to be consumed by ``PhysicsAwareAttention``.

    Raises
    ------
    ValueError
        If ``struct_tensor`` does not have shape ``[B, 3, H, W]``. Shape
        errors involving ``orientation``/``coherence`` (raised by
        ``OrientationExtractor`` or ``PhysicsRelationMatrix``) or the
        optional relation matrices (raised by ``PhysicsBiasFusion``) are
        left to those modules' own validation, and are not duplicated
        here.
    """

    def __init__(self) -> None:
        super().__init__()
        self.orientation_extractor = OrientationExtractor()
        self.physics_relation_matrix = PhysicsRelationMatrix()
        self.physics_bias_fusion = PhysicsBiasFusion()

    def forward(
        self,
        struct_tensor: torch.Tensor,
        wavelet_relation: Optional[torch.Tensor] = None,
        spectral_relation: Optional[torch.Tensor] = None,
        confidence_relation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build the physics attention bias from a raw structure tensor.

        Parameters
        ----------
        struct_tensor : torch.Tensor
            Shape ``[B, 3, H, W]``; channels ``(J11, J12, J22)``.
        wavelet_relation : torch.Tensor, optional
            Shape ``[B, N, N]``, matching the internally computed
            ``orientation_relation``.
        spectral_relation : torch.Tensor, optional
            Shape ``[B, N, N]``, matching ``orientation_relation``.
        confidence_relation : torch.Tensor, optional
            Shape ``[B, N, N]``, matching ``orientation_relation``.

        Returns
        -------
        torch.Tensor
            ``physics_attention_bias``, shape ``[B, N, N]``.
        """
        self._validate_struct_tensor(struct_tensor)

        # 1. Raw structure tensor -> orientation, coherence, anisotropy.
        #    Anisotropy is not consumed further downstream in this
        #    pipeline; it is returned by OrientationExtractor for other
        #    potential consumers but is not needed to build the bias.
        descriptors = self.orientation_extractor(struct_tensor)
        orientation = descriptors["orientation"]
        coherence = descriptors["coherence"]

        # 2. orientation, coherence -> pairwise orientation relation matrix.
        orientation_relation = self.physics_relation_matrix(
            orientation, coherence
        )

        # 3. orientation_relation (+ any optional relation matrices) ->
        #    fused, gated physics attention bias.
        physics_attention_bias = self.physics_bias_fusion(
            orientation_relation,
            wavelet_relation=wavelet_relation,
            spectral_relation=spectral_relation,
            confidence_relation=confidence_relation,
        )

        return physics_attention_bias

    @staticmethod
    def _validate_struct_tensor(struct_tensor: torch.Tensor) -> None:
        """Validate the top-level entry-point shape only.

        This checks exactly the contract this module's own docstring
        promises for ``struct_tensor``. It does not re-check anything
        that ``OrientationExtractor``, ``PhysicsRelationMatrix``, or
        ``PhysicsBiasFusion`` already validate about their own inputs
        (e.g. the shapes of ``orientation``/``coherence``, or that the
        optional relation matrices match ``orientation_relation``'s
        shape) -- those checks are left entirely to those modules.
        """
        if struct_tensor.dim() != 4 or struct_tensor.shape[1] != 3:
            raise ValueError(
                f"Expected `struct_tensor` of shape [B, 3, H, W] "
                f"(channels: J11, J12, J22), got shape "
                f"{tuple(struct_tensor.shape)}."
            )


if __name__ == "__main__":
    struct_tensor = torch.randn(2, 3, 32, 32)

    builder = PhysicsAttentionBiasBuilder()
    physics_attention_bias = builder(struct_tensor)

    print(f"physics_attention_bias shape: {tuple(physics_attention_bias.shape)}")
    assert tuple(physics_attention_bias.shape) == (2, 1024, 1024)
    assert torch.all(physics_attention_bias == 0.0), (
        "Output must be exactly zero at init (all PhysicsBiasFusion gates "
        "start at 0)."
    )

    print("PhysicsAttentionBiasBuilder smoke test passed.")
