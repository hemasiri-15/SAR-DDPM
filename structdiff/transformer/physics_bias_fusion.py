"""
PhysicsBiasFusion: fuses one or more physics relation matrices into a
single attention bias.

Phase 3.4 of the physics-aware attention framework
----------------------------------------------------
    Phase 3.1 (done): ``PhysicsRelationBuilder`` -- interface/validation
                       only, no computation.
    Phase 3.2 (done):  ``OrientationExtractor`` -- raw structure tensor ->
                       orientation, coherence, anisotropy.
    Phase 3.3 (done):  ``PhysicsRelationMatrix`` -- pairwise orientation
                       relation matrix, [B, N, N].
    Phase 3.4 (this file): fuse an arbitrary subset of per-modality
                       relation matrices (orientation now; wavelet,
                       spectral, confidence in later phases) into a
                       single additive attention bias, via learnable,
                       zero-initialized per-modality gates.
    Phase 3.5 (future): wire the resulting bias into ``TransformerBlock``
                       via its existing ``attention_bias`` extension
                       hook.

This module is standalone: it does not import from ``guided_diffusion``,
``PhysicsRelationBuilder``, ``OrientationExtractor``,
``PhysicsRelationMatrix``, or ``TransformerBlock``. Nothing in the
repository imports it yet.

Why zero initialization preserves warm-start compatibility
--------------------------------------------------------------
Every gate (``alpha_orientation``, ``alpha_wavelet``, ``alpha_spectral``,
``alpha_confidence``) starts at exactly 0. Because the fusion is a pure
weighted sum with no other operation (no normalization, no softmax, no
bias term), a zero gate makes that modality's entire contribution to
``physics_attention_bias`` exactly zero, and a zero bias added into
attention leaves attention scores identical to a model that has no
physics conditioning at all. This means ``PhysicsBiasFusion`` can be
dropped into a ``TransformerBlock`` that was trained (or partially
trained) without it, and at the moment of insertion the model's forward
pass is byte-for-byte unchanged. Training can then continue from that
exact checkpoint, with the gates free to move away from zero only if the
data actually benefits from the corresponding physics relation. This is
the same zero-init warm-start philosophy already used elsewhere in this
repository (e.g. zero-initializing ``look_emb.embedding.weight`` so that
look-conditioning starts as a no-op) -- new capacity is introduced
without perturbing an existing, validated forward pass.

Why gated fusion is preferable to hard-coded summation
------------------------------------------------------------
A hard-coded sum (``orientation_relation + wavelet_relation + ...``)
would implicitly assume all physics modalities are equally reliable and
equally relevant to attention, which is very unlikely to be true in
practice -- e.g. orientation from a structure tensor may be far more
informative for SAR edge structure than a spectral relation is, or vice
versa depending on the scene. A hard-coded sum also has no way to
represent "this modality is currently unhelpful, turn it off" other than
literally removing that term from the input, which is a code change, not
a learned decision. A learnable per-modality scalar gate lets the model
discover, from data and gradient signal, how much (if at all) each
physics relation should influence attention, independently of the
others, while the zero initialization guarantees that this learned
weighting starts from the safe, backward-compatible "no effect" point
described above rather than from an arbitrary hard-coded contribution.

Why future physics modalities can be added without changing the API
-------------------------------------------------------------------------
``forward`` already accepts ``wavelet_relation``, ``spectral_relation``,
and ``confidence_relation`` as optional keyword arguments that default to
``None``, and the module already owns a zero-initialized gate for each,
even though only ``orientation_relation`` is actually produced by the
pipeline as of this phase. Any matrix left as ``None`` is simply skipped
in the summation and does not need to be shape-validated against the
others. This means Phase 3.4's public interface does not need to change
again when wavelet, spectral, or confidence relation matrices become
available in later phases -- the caller will simply start passing real
tensors into arguments that already exist, and the corresponding
already-existing gate (currently frozen at 0 by initialization, but free
to learn once that modality starts contributing meaningfully) will begin
to receive gradient. No new gates, no new arguments, no signature change.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PhysicsBiasFusion(nn.Module):
    """Fuses per-modality physics relation matrices into a single
    attention bias via learnable, zero-initialized scalar gates.

    Owns four learnable scalar gates -- ``alpha_orientation``,
    ``alpha_wavelet``, ``alpha_spectral``, ``alpha_confidence`` -- each an
    ``nn.Parameter`` initialized to exactly 0. The fused output is a pure
    weighted sum of whichever relation matrices are supplied; no
    normalization, softmax, clipping, thresholding, convolution,
    projection, or learned embedding is performed anywhere in this
    module. See the module docstring for why zero initialization and
    gated (rather than hard-coded) summation are used, and why this
    interface does not need to change as future physics modalities are
    added.

    Input
    -----
    orientation_relation : torch.Tensor
        Required. Shape ``[B, N, N]``.
    wavelet_relation : torch.Tensor, optional
        Shape ``[B, N, N]``, matching ``orientation_relation``. Not yet
        produced anywhere in the pipeline as of this phase; accepted now
        so the API will not need to change later.
    spectral_relation : torch.Tensor, optional
        Shape ``[B, N, N]``, matching ``orientation_relation``. Not yet
        produced anywhere in the pipeline as of this phase.
    confidence_relation : torch.Tensor, optional
        Shape ``[B, N, N]``, matching ``orientation_relation``. Not yet
        produced anywhere in the pipeline as of this phase.

    Output
    ------
    torch.Tensor
        ``physics_attention_bias``, shape ``[B, N, N]``. At
        initialization this is exactly zero for any input, since all four
        gates start at 0.

    Raises
    ------
    ValueError
        If ``orientation_relation`` is ``None``, or if any supplied
        relation matrix's shape does not match ``orientation_relation``'s
        shape.
    """

    def __init__(self) -> None:
        super().__init__()
        self.alpha_orientation = nn.Parameter(torch.zeros(()))
        self.alpha_wavelet = nn.Parameter(torch.zeros(()))
        self.alpha_spectral = nn.Parameter(torch.zeros(()))
        self.alpha_confidence = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        orientation_relation: torch.Tensor,
        wavelet_relation: Optional[torch.Tensor] = None,
        spectral_relation: Optional[torch.Tensor] = None,
        confidence_relation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse the supplied physics relation matrices into a bias.

        Parameters
        ----------
        orientation_relation : torch.Tensor
            Required. Shape ``[B, N, N]``.
        wavelet_relation : torch.Tensor, optional
            Shape ``[B, N, N]``, matching ``orientation_relation``.
        spectral_relation : torch.Tensor, optional
            Shape ``[B, N, N]``, matching ``orientation_relation``.
        confidence_relation : torch.Tensor, optional
            Shape ``[B, N, N]``, matching ``orientation_relation``.

        Returns
        -------
        torch.Tensor
            ``physics_attention_bias``, shape ``[B, N, N]``. See the class
            docstring for the exact definition and initialization
            behaviour.
        """
        self._validate(
            orientation_relation,
            wavelet_relation,
            spectral_relation,
            confidence_relation,
        )

        physics_attention_bias = self.alpha_orientation * orientation_relation

        if wavelet_relation is not None:
            physics_attention_bias = (
                physics_attention_bias + self.alpha_wavelet * wavelet_relation
            )
        if spectral_relation is not None:
            physics_attention_bias = (
                physics_attention_bias
                + self.alpha_spectral * spectral_relation
            )
        if confidence_relation is not None:
            physics_attention_bias = (
                physics_attention_bias
                + self.alpha_confidence * confidence_relation
            )

        return physics_attention_bias

    @staticmethod
    def _validate(
        orientation_relation: Optional[torch.Tensor],
        wavelet_relation: Optional[torch.Tensor],
        spectral_relation: Optional[torch.Tensor],
        confidence_relation: Optional[torch.Tensor],
    ) -> None:
        if orientation_relation is None:
            raise ValueError(
                "`orientation_relation` is required and cannot be None."
            )
        if orientation_relation.dim() != 3:
            raise ValueError(
                f"Expected `orientation_relation` of shape [B, N, N], got "
                f"{orientation_relation.dim()} dimensions with shape "
                f"{tuple(orientation_relation.shape)}."
            )

        reference_shape = tuple(orientation_relation.shape)

        named_optional = (
            ("wavelet_relation", wavelet_relation),
            ("spectral_relation", spectral_relation),
            ("confidence_relation", confidence_relation),
        )
        for name, tensor in named_optional:
            if tensor is None:
                continue
            if tuple(tensor.shape) != reference_shape:
                raise ValueError(
                    f"Expected `{name}` to have shape {reference_shape} "
                    f"(matching `orientation_relation`), got shape "
                    f"{tuple(tensor.shape)}."
                )


if __name__ == "__main__":
    orientation = torch.randn(2, 256, 256)

    fusion = PhysicsBiasFusion()
    output = fusion(orientation)

    print(f"output shape: {tuple(output.shape)}")
    print(f"alpha_orientation: {fusion.alpha_orientation.item()}")
    print(f"alpha_wavelet: {fusion.alpha_wavelet.item()}")
    print(f"alpha_spectral: {fusion.alpha_spectral.item()}")
    print(f"alpha_confidence: {fusion.alpha_confidence.item()}")

    assert tuple(output.shape) == (2, 256, 256)
    assert torch.all(output == 0.0), "Output must be exactly zero at init."

    print("PhysicsBiasFusion smoke test passed.")
