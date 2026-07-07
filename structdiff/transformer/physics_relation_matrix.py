"""
PhysicsRelationMatrix: pairwise physics relation matrix from orientation
and coherence maps.

Phase 3.3 of the physics-aware attention framework
----------------------------------------------------
    Phase 3.1 (done): ``PhysicsRelationBuilder`` -- interface/validation
                       only, no computation.
    Phase 3.2 (done):  ``OrientationExtractor`` -- raw structure tensor ->
                       orientation, coherence, anisotropy.
    Phase 3.3 (this file): pairwise comparison of orientation(i) vs
                       orientation(j), weighted by coherence, across all
                       spatial position pairs -> a physics relation matrix
                       of shape [B, N, N].
    Phase 3.4 (future): fuse this relation matrix with wavelet, spectral,
                       and confidence relation matrices into a single
                       zero-initialized, gated attention bias.
    Phase 3.5 (future): wire the bias into ``TransformerBlock`` via its
                       existing ``attention_bias`` hook.

This module is standalone: it does not import from ``guided_diffusion``,
``PhysicsRelationBuilder``, ``OrientationExtractor``, or
``TransformerBlock``. Nothing in the repository imports it yet.

Why cos(2*delta_theta) instead of cos(delta_theta)
----------------------------------------------------
Orientation, as produced by ``OrientationExtractor``, is the angle of an
eigenvector of a symmetric structure tensor -- i.e. it describes an
*undirected axis*, not a directed vector. A line at 0 degrees and a line
at 180 degrees are the physically identical axis, so orientation is only
meaningful modulo pi. A plain ``cos(delta_theta)`` does not respect this:
``cos(0) = 1`` but ``cos(pi) = -1``, which would incorrectly report two
pixels lying on the very same physical edge (one measured as 0 degrees,
the other as 180 degrees purely as an artifact of ``atan2``'s branch cut)
as having *opposite* orientation.

Doubling the angle before taking the cosine fixes this exactly:
``cos(2*(delta_theta + pi)) == cos(2*delta_theta + 2*pi) == cos(2*delta_theta)``.
So ``cos(2*delta_theta)`` is pi-periodic in ``delta_theta`` by
construction, which is exactly the periodicity the underlying physics
(an undirected axis) has. No explicit angle-wrapping step is needed to
get this right -- the doubled cosine already does the periodic
comparison for us:

    agreement(i, j) = cos(2*theta_i - 2*theta_j)
                     = cos(2*theta_i)*cos(2*theta_j) + sin(2*theta_i)*sin(2*theta_j)

which is computed below as an outer-product sum rather than by first
materializing an explicit [B, N, N] angle-difference tensor -- the result
is identical, but this form is a single pair of batched outer products
instead of an O(N^2) elementwise atan2/subtraction pass.

Why coherence weights the relation
------------------------------------
Orientation is only a physically meaningful quantity where the local
structure tensor actually has a well-defined dominant axis. Inside
homogeneous, speckle-dominated SAR regions, the structure tensor is
close to isotropic, coherence is near 0, and the orientation angle
recovered there is essentially noise (an arbitrary answer from
``atan2`` with no real underlying axis behind it). Multiplying the
angular agreement by ``sqrt(coherence_i * coherence_j)`` down-weights
exactly these unreliable comparisons: two pixels can only contribute a
strong relation value if *both* of them individually have a confidently
defined orientation. The square root (rather than the product itself)
keeps the weighting symmetric in scale with the two per-pixel coherence
values rather than compounding quadratically, so a single confidently
oriented pixel paired with a moderately confident one is not penalized
as heavily as two only-moderately-confident pixels would be.

Why this creates a physically meaningful pairwise relation matrix
---------------------------------------------------------------------
Together, ``relation(i, j) = cos(2*delta_theta) * sqrt(coherence_i *
coherence_j)`` answers the question "do pixels i and j plausibly lie on
the same underlying oriented physical structure (edge, road, field
boundary, coastline), and how confident are we in that answer?" -- both
in one signed scalar per pair, naturally bounded in [-1, 1]. Positive
values mean the two pixels are confidently aligned to the same axis
(candidates for representing the same physical linear feature, however
far apart they are in the image and however much speckle separates
them); negative values mean confidently orthogonal axes; values near 0
mean the comparison is either directionally ambiguous or one/both
pixels lack a reliable local orientation to begin with. This is exactly
the kind of long-range, content-addressable relation that a plain
convolutional or fixed-window attention mechanism cannot express, which
is why it is being built as a relation matrix to bias attention rather
than as another local feature map.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class PhysicsRelationMatrix(nn.Module):
    """Computes a pairwise physics relation matrix from orientation and
    coherence maps.

    This module performs pure tensor mathematics with no learnable
    parameters and no buffers -- no ``Conv``, no ``Linear``, nothing
    trainable. It is implemented as an ``nn.Module`` purely for
    compositional consistency with the rest of the physics-aware
    attention framework (``PhysicsRelationBuilder``,
    ``OrientationExtractor``), not because it has any state to manage.

    Input
    -----
    orientation : torch.Tensor
        Shape ``[B, 1, H, W]``, in radians. Meaningful modulo pi (as
        produced by ``OrientationExtractor``).
    coherence : torch.Tensor
        Shape ``[B, 1, H, W]``, expected in ``[0, 1]``.

    Output
    ------
    torch.Tensor
        ``physics_relation``, shape ``[B, N, N]`` where ``N = H * W``,
        with values naturally in ``[-1, 1]``:

        - positive: pixels i and j lie along the same orientation axis
        - negative: pixels i and j lie along orthogonal axes
        - magnitude: scaled by how confidently oriented both pixels are

    Raises
    ------
    ValueError
        If ``orientation`` or ``coherence`` does not have shape
        ``[B, 1, H, W]``, or if their shapes do not match each other.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, orientation: torch.Tensor, coherence: torch.Tensor
    ) -> torch.Tensor:
        """Compute the pairwise physics relation matrix.

        Parameters
        ----------
        orientation : torch.Tensor
            Shape ``[B, 1, H, W]``, in radians.
        coherence : torch.Tensor
            Shape ``[B, 1, H, W]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, N, N]`` where ``N = H * W``. See the class
            docstring for the exact definition and value range.
        """
        self._validate(orientation, coherence)

        b, _, h, w = orientation.shape
        n = h * w

        theta = orientation.reshape(b, n)  # [B, N]
        coh = coherence.reshape(b, n)  # [B, N]

        # --- Angular agreement, via doubled-angle cos/sin outer products ---
        # agreement(i, j) = cos(2*theta_i - 2*theta_j)
        #                 = cos(2*theta_i)*cos(2*theta_j)
        #                   + sin(2*theta_i)*sin(2*theta_j)
        # This is exactly pi-periodic in (theta_i - theta_j), which matches
        # the physical periodicity of an undirected structure-tensor axis --
        # see the module docstring for why doubling the angle is required
        # instead of a plain angular subtraction.
        cos2 = torch.cos(2.0 * theta)  # [B, N]
        sin2 = torch.sin(2.0 * theta)  # [B, N]

        agreement = torch.bmm(
            cos2.unsqueeze(-1), cos2.unsqueeze(-2)
        ) + torch.bmm(sin2.unsqueeze(-1), sin2.unsqueeze(-2))  # [B, N, N]

        # --- Coherence weighting ---
        # weight(i, j) = sqrt(coherence_i * coherence_j)
        # Clamp to >= 0 before the sqrt purely as numerical safety against
        # tiny negative floating-point noise if `coherence` arrives with
        # values fractionally below 0 at the boundary; per the input
        # contract, coherence is expected in [0, 1] already.
        coh_clamped = coh.clamp(min=0.0)
        coh_weight = torch.sqrt(
            torch.bmm(coh_clamped.unsqueeze(-1), coh_clamped.unsqueeze(-2))
        )  # [B, N, N]

        physics_relation = agreement * coh_weight  # [B, N, N]

        return physics_relation

    @staticmethod
    def _validate(orientation: torch.Tensor, coherence: torch.Tensor) -> None:
        def check_shape(name: str, tensor: torch.Tensor) -> None:
            if tensor.dim() != 4 or tensor.shape[1] != 1:
                raise ValueError(
                    f"Expected `{name}` of shape [B, 1, H, W], got shape "
                    f"{tuple(tensor.shape)}."
                )

        check_shape("orientation", orientation)
        check_shape("coherence", coherence)

        if orientation.shape != coherence.shape:
            raise ValueError(
                f"Expected `orientation` and `coherence` to have matching "
                f"shapes, got orientation {tuple(orientation.shape)} and "
                f"coherence {tuple(coherence.shape)}."
            )


if __name__ == "__main__":
    orientation = torch.rand(2, 1, 16, 16)
    coherence = torch.rand(2, 1, 16, 16)

    module = PhysicsRelationMatrix()
    output = module(orientation, coherence)

    print(f"output shape: {tuple(output.shape)}")
    assert tuple(output.shape) == (2, 256, 256)

    print("PhysicsRelationMatrix smoke test passed.")
