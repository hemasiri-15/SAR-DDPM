"""
OrientationExtractor: converts a raw 2x2 structure tensor field into
physically meaningful per-pixel descriptors (orientation, coherence,
anisotropy).

Phase 3.2 of the physics-aware attention framework
----------------------------------------------------
    Phase 3.1 (done): ``PhysicsRelationBuilder`` -- interface/validation
                       only, no computation.
    Phase 3.2 (this file): raw structure tensor -> orientation, coherence,
                       anisotropy. Pure tensor mathematics, no learning.
    Phase 3.3 (future): pairwise comparison of orientation(i) vs
                       orientation(j) across all spatial position pairs ->
                       a physics relation matrix.
    Phase 3.4 (future): fuse per-source relation matrices into a single
                       attention bias.
    Phase 3.5 (future): wire the bias into ``TransformerBlock`` via its
                       existing ``attention_bias`` hook.

This module is standalone: it does not import from
``guided_diffusion``, ``PhysicsRelationBuilder``, or
``TransformerBlock``, and nothing in the repository imports it yet.

Why these three descriptors, physically
-----------------------------------------
The 2-D structure tensor at a pixel is the symmetric matrix

    J = [[J11, J12],
         [J12, J22]]

formed from local image-gradient products (e.g. J11 = <Ix^2>, J22 = <Iy^2>,
J12 = <Ix*Iy>, spatially averaged). Its eigenvectors point along and across
the dominant local gradient direction, and its eigenvalues measure how much
gradient energy exists along each of those directions.

- Orientation (theta): the eigenvector directions are exactly
  ``0.5 * atan2(2*J12, J11 - J22)`` (mod pi, since a structure tensor
  describes an axis, not a directed vector). In SAR imagery, linear
  features -- roads, rivers, coastlines, field boundaries, building edges
  -- are exactly the kind of extended, oriented structure this angle
  captures, and orientation is what would let two pixels lying along the
  same physical edge be identified as "physically related" even though
  they may be far apart in the image and surrounded by speckle.
- Coherence: measures how strongly the local gradient is concentrated
  along a single direction versus spread out isotropically. Near a strong,
  clean edge, gradient energy is concentrated in one direction and
  coherence is high; inside homogeneous, speckle-dominated regions with no
  consistent gradient direction, coherence is low. This is exactly the
  distinction needed to tell a genuine SAR edge apart from speckle noise
  that happens to produce a locally large but directionless gradient.
- Anisotropy: derived from the true eigenvalues of J, ``(lambda1 -
  lambda2) / (lambda1 + lambda2 + eps)``, this measures the relative
  imbalance between the two eigenvalues -- i.e. how much more structure
  exists along one axis than its perpendicular. It quantifies local
  directional structure independent of overall gradient magnitude, which
  matters in SAR because absolute gradient magnitude is itself corrupted
  by multiplicative speckle, while the *ratio* of directional energy is
  comparatively more robust.

Note on coherence vs. anisotropy
----------------------------------
Given the exact formulas specified for this module, these two quantities
are mathematically identical: with ``trace = J11 + J22`` and
``disc = sqrt((J11 - J22)^2 + 4*J12^2)``, the true eigenvalues are
``lambda1, lambda2 = (trace +/- disc) / 2``, so
``lambda1 - lambda2 == disc`` and ``lambda1 + lambda2 == trace`` exactly.
That makes ``coherence == disc / (trace + eps)`` and
``anisotropy == (lambda1 - lambda2) / (lambda1 + lambda2 + eps)``
numerically the same quantity (up to floating-point rounding). This
implementation still computes the eigenvalues explicitly and derives
anisotropy from them (rather than reusing the coherence intermediate
value), exactly as specified ("compute the true eigenvalues, do not
approximate") -- this keeps the two code paths independently correct and
independently modifiable if a future phase changes one formula but not the
other, even though they currently coincide. This redundancy is a property
of the specified formulas, not an implementation choice, and is
intentionally not "fixed" here since this phase's job is to implement the
given design exactly.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

_EPS = 1e-8


class OrientationExtractor(nn.Module):
    """Extracts orientation, coherence, and anisotropy from a structure tensor.

    This module performs pure tensor mathematics with no learnable
    parameters, no buffers, and no training-time behaviour -- it computes
    the same deterministic, closed-form output regardless of ``.train()``/
    ``.eval()`` mode. It is implemented as an ``nn.Module`` purely so it can
    be composed naturally alongside the rest of the physics-aware attention
    framework (e.g. placed inside an ``nn.Sequential`` or called from
    another module's ``forward``), not because it has any state to manage.

    Parameters
    ----------
    eps : float, optional
        Small constant added to denominators to avoid division by zero in
        near-degenerate regions (e.g. constant, gradient-free patches
        where ``J11 == J22 == J12 == 0``). Default is ``1e-8``.

    Input
    -----
    struct_tensor : torch.Tensor
        Raw structure-tensor components, shape ``[B, 3, H, W]``, with
        channels ordered as:

        - channel 0: ``J11``
        - channel 1: ``J12``
        - channel 2: ``J22``

    Output
    ------
    dict of str to torch.Tensor
        A dictionary with exactly three keys, each mapping to a tensor of
        shape ``[B, 1, H, W]``:

        - ``"orientation"``: ``0.5 * atan2(2*J12, J11 - J22)``, in radians.
          Because a structure tensor describes an undirected axis (a line
          has no "forward" direction), this angle is only meaningful modulo
          pi; two orientations differing by pi describe the same physical
          axis. Downstream code comparing orientations (Phase 3.3) will
          need to account for this periodicity.
        - ``"coherence"``:
          ``sqrt((J11-J22)^2 + 4*J12^2) / (J11+J22+eps)``, unitless, in
          ``[0, 1]`` for a valid (positive semi-definite) structure tensor.
        - ``"anisotropy"``: ``(lambda1 - lambda2) / (lambda1 + lambda2 +
          eps)``, where ``lambda1, lambda2`` are the true eigenvalues of
          the 2x2 structure tensor (``lambda1 >= lambda2``), computed via
          the closed-form symmetric 2x2 eigenvalue solution. See the module
          docstring for why this is currently numerically identical to
          ``coherence`` given these specific formulas.

    Raises
    ------
    ValueError
        If ``struct_tensor`` does not have shape ``[B, 3, H, W]``.
    """

    def __init__(self, eps: float = _EPS) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, struct_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute orientation, coherence, and anisotropy maps.

        Parameters
        ----------
        struct_tensor : torch.Tensor
            Shape ``[B, 3, H, W]``; see the class docstring for channel
            ordering.

        Returns
        -------
        dict of str to torch.Tensor
            ``{"orientation": ..., "coherence": ..., "anisotropy": ...}``,
            each of shape ``[B, 1, H, W]``. See the class docstring for
            exact definitions.
        """
        if struct_tensor.dim() != 4 or struct_tensor.shape[1] != 3:
            raise ValueError(
                f"Expected `struct_tensor` of shape [B, 3, H, W] "
                f"(channels: J11, J12, J22), got shape "
                f"{tuple(struct_tensor.shape)}."
            )

        j11 = struct_tensor[:, 0:1, :, :]
        j12 = struct_tensor[:, 1:2, :, :]
        j22 = struct_tensor[:, 2:3, :, :]

        trace = j11 + j22
        diff = j11 - j22
        # Common discriminant term: appears both in the coherence formula
        # and in the closed-form eigenvalues of a symmetric 2x2 matrix.
        discriminant = torch.sqrt(diff ** 2 + 4.0 * j12 ** 2)

        # --- Orientation ---------------------------------------------------------
        orientation = 0.5 * torch.atan2(2.0 * j12, diff)

        # --- Coherence -------------------------------------------------------------
        coherence = discriminant / (trace + self.eps)

        # --- Anisotropy (via true eigenvalues, not the coherence shortcut) ----------
        half_trace = 0.5 * trace
        half_discriminant = 0.5 * discriminant
        lambda1 = half_trace + half_discriminant
        lambda2 = half_trace - half_discriminant
        anisotropy = (lambda1 - lambda2) / (lambda1 + lambda2 + self.eps)

        print("\n===== Orientation Extractor =====")
        print("orientation :", orientation.dtype)
        print("coherence   :", coherence.dtype)
        print("structure   :", struct_tensor.dtype)
        print("===============================\n")

        return {
            "orientation": orientation,
            "coherence": coherence,
            "anisotropy": anisotropy,
        }


if __name__ == "__main__":
    J = torch.randn(2, 3, 64, 64)

    extractor = OrientationExtractor()
    result = extractor(J)

    print(f"orientation shape: {tuple(result['orientation'].shape)}")
    print(f"coherence shape: {tuple(result['coherence'].shape)}")
    print(f"anisotropy shape: {tuple(result['anisotropy'].shape)}")

    print("OrientationExtractor smoke test passed.")
