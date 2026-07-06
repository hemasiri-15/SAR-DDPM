"""
PhysicsRelationBuilder: interface module for a future Physics-Aware Attention
framework.

Phase 3.1 (this revision)
--------------------------
This is the *first* phase of a planned multi-phase build:

    Phase 3.1 (this file): define the input/output interface only.
                            Validate shapes, package inputs into a dict.
                            No tensor computation of any kind.
    Phase 3.2 (future):     orientation/coherence extraction from the raw
                            structure tensor.
    Phase 3.3 (future):     pairwise physics-relation computation between
                            spatial positions (e.g. angular agreement,
                            frequency-content similarity).
    Phase 3.4 (future):     fuse per-source relations into a single additive
                            attention bias (with learnable, zero-initialized
                            gates per source, matching the zero-init
                            philosophy already used throughout
                            ``structdiff/transformer/transformer_block.py``).
    Phase 3.5 (future):     wire the resulting bias into
                            ``TransformerBlock`` (likely via its existing
                            ``attention_bias`` extension hook).

This module is intentionally standalone: it does not import from
``guided_diffusion``, ``structdiff.transformer.transformer_block``, or any
of the existing conditioning encoders (``StructTensorEncoder``,
``MultiScaleStructTensorEncoder``, ``TensorSpectralEncoder``,
``WaveletEncoder``). Nothing in the repository imports this module yet.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

# Expected channel count for each physics tensor, per the interface spec.
_STRUCT_TENSOR_CHANNELS = 3
_WAVELET_TENSOR_CHANNELS = 4
_SPECTRAL_TENSOR_CHANNELS = 12
_CONFIDENCE_MAP_CHANNELS = 1


class PhysicsRelationBuilder(nn.Module):
    """Validates and packages raw physics-conditioning tensors.

    This is Phase 1 of a planned physics-aware attention framework for
    ``TransformerBlock``. In this phase, ``PhysicsRelationBuilder`` performs
    **no tensor computation whatsoever** -- no convolutions, projections,
    pooling, eigendecomposition, or attention-bias construction. Its sole
    job is to define the *interface* that later phases will build on: it
    accepts the raw outputs of the repository's existing physics encoders,
    validates their shapes, and returns them unchanged in a dictionary
    keyed by source name.

    Later phases will replace the body of ``forward`` with real
    computation (orientation/coherence extraction, pairwise relation
    computation, bias fusion) *without changing this public API* -- the
    same incremental strategy used to build and integrate
    ``TransformerBlock`` itself (standalone module first, then integration,
    then feature phases).

    Inputs
    ------
    struct_tensor : torch.Tensor, optional
        Raw structure-tensor components, shape ``[B, 3, H, W]``. The 3
        channels are expected to be the independent components of the
        symmetric 2x2 structure tensor (e.g. ``J_xx``, ``J_xy``, ``J_yy``);
        this module does not interpret them in this phase, only validates
        the channel count.
    wavelet_tensor : torch.Tensor, optional
        Wavelet sub-band features, shape ``[B, 4, H/2, W/2]`` -- 4 sub-bands
        (e.g. LL/LH/HL/HH from a single-level 2-D DWT) at half the spatial
        resolution of the other physics tensors. If any full-resolution
        physics tensor (``struct_tensor``, ``spectral_tensor``, or
        ``confidence_map``) is *also* provided in the same call, this
        module additionally validates that the wavelet tensor's spatial
        dimensions are exactly half of that reference resolution (see
        "Cross-tensor resolution check" below). If ``wavelet_tensor`` is
        the *only* input provided, no such cross-check is possible and only
        its channel count is validated.
    spectral_tensor : torch.Tensor, optional
        Spectral/frequency-domain tensor features, shape
        ``[B, 12, H, W]``, at full spatial resolution.
    confidence_map : torch.Tensor, optional
        Per-pixel confidence map, shape ``[B, 1, H, W]``, at full spatial
        resolution.

    All four inputs are optional and independent -- any subset (including
    none) may be provided. Missing inputs are simply omitted from the
    returned dictionary rather than being filled with placeholders.

    Cross-tensor resolution check
    ------------------------------
    The spec describes ``wavelet_tensor`` as living at ``H/2, W/2`` -- half
    the resolution of the other three tensors, which all share the same
    full-resolution ``[H, W]``. Because every input is independently
    optional, there is not always a canonical ``(H, W)`` available to check
    against. This implementation resolves that as follows: if at least one
    full-resolution tensor (``struct_tensor``, ``spectral_tensor``, or
    ``confidence_map``) is present, its spatial shape is taken as the
    reference ``(H, W)``, and:

    * every other full-resolution tensor provided must match that
      reference ``(H, W)`` exactly, and
    * ``wavelet_tensor``, if provided, must have spatial shape
      ``(H // 2, W // 2)`` exactly (which additionally requires ``H`` and
      ``W`` to both be even).

    If ``wavelet_tensor`` is the only tensor provided, there is no
    reference resolution to check it against, so only its channel count
    (and batch size internal consistency, trivially) is validated.

    All provided tensors must additionally share the same batch size.

    Returns
    -------
    dict of str to torch.Tensor
        A dictionary containing only the keys for the inputs that were
        provided (as ``"struct_tensor"``, ``"wavelet_tensor"``,
        ``"spectral_tensor"``, ``"confidence_map"``), mapped to the
        corresponding input tensors, unmodified.

    Raises
    ------
    ValueError
        If any provided tensor does not have 4 dimensions, does not have
        the expected channel count for its role, has a batch size that
        disagrees with another provided tensor, or (per the cross-tensor
        resolution check above) has spatial dimensions inconsistent with
        the other provided tensors.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        struct_tensor: Optional[torch.Tensor] = None,
        wavelet_tensor: Optional[torch.Tensor] = None,
        spectral_tensor: Optional[torch.Tensor] = None,
        confidence_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Validate and package the provided physics tensors.

        See the class docstring for the full description of each argument,
        the validation rules, and the return value. This phase performs no
        computation -- only shape validation and dictionary packaging.
        """
        physics_features: Dict[str, torch.Tensor] = {}
        batch_size: Optional[int] = None
        reference_hw: Optional[Tuple[int, int]] = None

        def check_batch_size(name: str, tensor: torch.Tensor) -> None:
            nonlocal batch_size
            if batch_size is None:
                batch_size = tensor.shape[0]
            elif tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch: `{name}` has batch size "
                    f"{tensor.shape[0]}, but a previously validated input "
                    f"had batch size {batch_size}. All provided physics "
                    f"tensors must share the same batch size."
                )

        def validate_full_res(
            name: str, tensor: torch.Tensor, expected_channels: int
        ) -> None:
            nonlocal reference_hw
            if tensor.dim() != 4:
                raise ValueError(
                    f"Expected `{name}` to be a 4-D tensor "
                    f"[B, {expected_channels}, H, W], got {tensor.dim()} "
                    f"dimensions with shape {tuple(tensor.shape)}."
                )
            if tensor.shape[1] != expected_channels:
                raise ValueError(
                    f"Expected `{name}` to have {expected_channels} "
                    f"channel(s), got {tensor.shape[1]} in shape "
                    f"{tuple(tensor.shape)}."
                )
            check_batch_size(name, tensor)

            h, w = tensor.shape[2], tensor.shape[3]
            if reference_hw is None:
                reference_hw = (h, w)
            elif (h, w) != reference_hw:
                raise ValueError(
                    f"Spatial resolution mismatch: `{name}` has spatial "
                    f"shape ({h}, {w}), but a previously validated "
                    f"full-resolution input established the reference "
                    f"resolution {reference_hw}. All full-resolution "
                    f"physics tensors (struct_tensor, spectral_tensor, "
                    f"confidence_map) must share the same (H, W)."
                )

        # Validate full-resolution tensors first, so a reference (H, W) is
        # established (if any is present) before we validate the wavelet
        # tensor against it.
        if struct_tensor is not None:
            validate_full_res(
                "struct_tensor", struct_tensor, _STRUCT_TENSOR_CHANNELS
            )
            physics_features["struct_tensor"] = struct_tensor

        if spectral_tensor is not None:
            validate_full_res(
                "spectral_tensor", spectral_tensor, _SPECTRAL_TENSOR_CHANNELS
            )
            physics_features["spectral_tensor"] = spectral_tensor

        if confidence_map is not None:
            validate_full_res(
                "confidence_map", confidence_map, _CONFIDENCE_MAP_CHANNELS
            )
            physics_features["confidence_map"] = confidence_map

        if wavelet_tensor is not None:
            if wavelet_tensor.dim() != 4:
                raise ValueError(
                    f"Expected `wavelet_tensor` to be a 4-D tensor "
                    f"[B, {_WAVELET_TENSOR_CHANNELS}, H/2, W/2], got "
                    f"{wavelet_tensor.dim()} dimensions with shape "
                    f"{tuple(wavelet_tensor.shape)}."
                )
            if wavelet_tensor.shape[1] != _WAVELET_TENSOR_CHANNELS:
                raise ValueError(
                    f"Expected `wavelet_tensor` to have "
                    f"{_WAVELET_TENSOR_CHANNELS} channels, got "
                    f"{wavelet_tensor.shape[1]} in shape "
                    f"{tuple(wavelet_tensor.shape)}."
                )
            check_batch_size("wavelet_tensor", wavelet_tensor)

            wh, ww = wavelet_tensor.shape[2], wavelet_tensor.shape[3]
            if reference_hw is not None:
                ref_h, ref_w = reference_hw
                if ref_h % 2 != 0 or ref_w % 2 != 0:
                    raise ValueError(
                        f"Reference full resolution {reference_hw} "
                        f"(from struct_tensor/spectral_tensor/"
                        f"confidence_map) is not evenly divisible by 2, "
                        f"so no valid `wavelet_tensor` shape of "
                        f"(H/2, W/2) exists to compare against."
                    )
                expected_wavelet_hw = (ref_h // 2, ref_w // 2)
                if (wh, ww) != expected_wavelet_hw:
                    raise ValueError(
                        f"Expected `wavelet_tensor` spatial shape "
                        f"{expected_wavelet_hw} (half the reference "
                        f"resolution {reference_hw} established by another "
                        f"provided physics tensor), got ({wh}, {ww})."
                    )
            physics_features["wavelet_tensor"] = wavelet_tensor

        return physics_features


if __name__ == "__main__":
    struct = torch.randn(2, 3, 64, 64)
    wavelet = torch.randn(2, 4, 32, 32)
    spectral = torch.randn(2, 12, 64, 64)
    confidence = torch.randn(2, 1, 64, 64)

    builder = PhysicsRelationBuilder()
    physics_features = builder(
        struct_tensor=struct,
        wavelet_tensor=wavelet,
        spectral_tensor=spectral,
        confidence_map=confidence,
    )

    print(f"Returned keys: {list(physics_features.keys())}")

    print("PhysicsRelationBuilder smoke test passed.")
