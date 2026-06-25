"""
structdiff/conditioning/ms_struct_tensor_encoder.py
====================================================
A10: Multi-Scale Structure Tensor Encoder — conditioning module.

Maps three structure tensor maps (fine, medium, coarse) to a single
dense embedding vector [B, time_embed_dim] that is added to the
diffusion timestep embedding inside ``ConditionedSuperResModel.forward``:

    emb = time_emb + look_emb(look_num) + ms_struct_encoder(st1, st2, st3)

Architecture
------------
Shared ``StructTensorEncoder`` (A3, unchanged) + 3 learnable scale
embedding vectors:

    v1 = shared_enc(st1) + scale_emb[0]   # fine
    v2 = shared_enc(st2) + scale_emb[1]   # medium
    v3 = shared_enc(st3) + scale_emb[2]   # coarse
    output = v1 + v2 + v3                 # [B, time_embed_dim]

The shared encoder is the unmodified ``StructTensorEncoder`` from A3.
Scale embeddings are ``nn.Embedding(3, time_embed_dim)`` with near-zero
initialisation, so at warm-start the scale vectors contribute ~0 and
the encoder output per scale matches a plain A3 call.

Design contract
---------------
- ``StructTensorEncoder`` is NOT modified.
- ``UNetModel`` / ``ResBlock`` are NOT modified.
- ``GaussianDiffusion``, ``TrainLoop``, EMA are NOT modified.
- This module owns no training loop logic.
- Public API: ``forward(st1, st2, st3)`` returning [B, time_embed_dim].

Initialisation / warm-start
-----------------------------
``StructTensorEncoder.proj`` is already near-zero (std=0.02, A3 contract).
``scale_emb.weight`` is additionally initialised near-zero (std=0.02)
so that at A10 fine-tuning start from an A3 checkpoint the per-scale
bias is also near-zero.  The sum v1+v2+v3 is thus ~3× the single A3
output initially — dominated by the three ``shared_enc`` calls.

Checkpoint compatibility
------------------------
A3 checkpoint → A10 model:
    ``struct_encoder.*`` (A3) maps to ``ms_struct_encoder.shared_enc.*``
    (A10) via the migration utility (see migrate_a3_to_a10.py).
    ``ms_struct_encoder.scale_emb.*`` is absent from A3; it starts
    near-zero.  Use ``load_state_dict(strict=False)``.

Key prefix summary:
    A3:   struct_encoder.*
    A10:  ms_struct_encoder.shared_enc.*
          ms_struct_encoder.scale_emb.*
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path resolution: structdiff conditioning package
# ---------------------------------------------------------------------------
_COND_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _COND_DIR not in sys.path:
    sys.path.insert(0, _COND_DIR)

from conditioning.structure_tensor_encoder import StructTensorEncoder  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of scales (fine / medium / coarse).
_NUM_SCALES: int = 3

#: Near-zero std for scale embedding initialisation (matches LookEmbedding).
_SCALE_EMB_INIT_STD: float = 0.02


class MultiScaleStructTensorEncoder(nn.Module):
    """Encodes three structure tensor scales into a single conditioning vector.

    Each of the three input tensors (fine, medium, coarse) is processed
    by the *same* ``StructTensorEncoder`` instance, then biased by a
    per-scale learnable embedding, and the three results are summed.

    Parameters
    ----------
    time_embed_dim:
        Output dimensionality — must match ``UNetModel``'s ``time_embed``
        output (``model_channels * 4``).  For the default SAR-DDPM
        configuration (``--num_channels 192``) this is ``768``.

    Attributes
    ----------
    shared_enc : StructTensorEncoder
        The shared convolutional encoder (A3, unmodified).
        State dict prefix: ``ms_struct_encoder.shared_enc.*``
    scale_emb : nn.Embedding
        Shape ``[3, time_embed_dim]``.  Row i is the bias added to the
        shared encoder output for scale i (0=fine, 1=medium, 2=coarse).
        State dict prefix: ``ms_struct_encoder.scale_emb.*``

    Examples
    --------
    >>> enc = MultiScaleStructTensorEncoder(time_embed_dim=768)
    >>> st1 = torch.randn(4, 3, 256, 256)
    >>> st2 = torch.randn(4, 3, 256, 256)
    >>> st3 = torch.randn(4, 3, 256, 256)
    >>> v = enc(st1, st2, st3)
    >>> v.shape
    torch.Size([4, 768])
    """

    def __init__(self, time_embed_dim: int) -> None:
        super().__init__()

        if time_embed_dim <= 0:
            raise ValueError(
                f"time_embed_dim must be a positive integer, got {time_embed_dim}."
            )

        # ------------------------------------------------------------------
        # Shared convolutional encoder (A3, unmodified).
        # Its own proj layer is already near-zero (A3 contract).
        # ------------------------------------------------------------------
        self.shared_enc = StructTensorEncoder(time_embed_dim=time_embed_dim)

        # ------------------------------------------------------------------
        # Per-scale bias embeddings: 3 rows × time_embed_dim.
        # Indexed 0=fine, 1=medium, 2=coarse.
        # Near-zero init so scale bias contributes ~0 at warm-start.
        # ------------------------------------------------------------------
        self.scale_emb = nn.Embedding(
            num_embeddings=_NUM_SCALES,
            embedding_dim=time_embed_dim,
        )
        nn.init.normal_(self.scale_emb.weight, mean=0.0, std=_SCALE_EMB_INIT_STD)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def time_embed_dim(self) -> int:
        """Output embedding dimensionality."""
        return self.shared_enc.time_embed_dim

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        st1: torch.Tensor,
        st2: torch.Tensor,
        st3: torch.Tensor,
    ) -> torch.Tensor:
        """Encode three structure tensor maps into one conditioning vector.

        Parameters
        ----------
        st1:
            Fine-scale structure tensor, shape ``[B, 3, H, W]``, float32.
        st2:
            Medium-scale structure tensor, shape ``[B, 3, H, W]``, float32.
        st3:
            Coarse-scale structure tensor, shape ``[B, 3, H, W]``, float32.

        All three tensors must be on the same device as this module.
        Expected range [-1, 1] per channel (as produced by
        ``compute_structure_tensor_multiscale(..., normalise=True)``).

        Returns
        -------
        torch.Tensor
            Float tensor of shape ``[B, time_embed_dim]``.
            Add directly to the timestep + look embedding:

                emb = emb + ms_struct_encoder(st1, st2, st3)

        Raises
        ------
        ValueError
            If any input is not 4-D or does not have 3 channels.
        """
        # Validate all three inputs (StructTensorEncoder also validates,
        # but we give a clearer error message with the scale label here).
        for label, st in (("st1", st1), ("st2", st2), ("st3", st3)):
            if st.ndim != 4:
                raise ValueError(
                    f"{label} must be 4-D [B, 3, H, W], got shape {tuple(st.shape)}."
                )
            if st.shape[1] != 3:
                raise ValueError(
                    f"{label} must have 3 channels (J11, J12, J22), "
                    f"got {st.shape[1]}."
                )

        # Scale index tensors (scalar, expanded to [B] via expand).
        # Constructed on the same device as the inputs.
        B = st1.shape[0]
        device = st1.device

        idx0 = torch.zeros(B, dtype=torch.long, device=device)   # fine
        idx1 = torch.ones(B, dtype=torch.long, device=device)    # medium
        idx2 = torch.full((B,), 2, dtype=torch.long, device=device)  # coarse

        # Shared encoder: each call [B, 3, H, W] → [B, D]
        v1 = self.shared_enc(st1) + self.scale_emb(idx0)  # [B, D]
        v2 = self.shared_enc(st2) + self.scale_emb(idx1)  # [B, D]
        v3 = self.shared_enc(st3) + self.scale_emb(idx2)  # [B, D]

        # Sum across scales → [B, D]
        return v1 + v2 + v3
