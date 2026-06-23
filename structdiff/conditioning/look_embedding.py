"""
structdiff/conditioning/look_embedding.py
==========================================
A2: Look Embedding — conditioning module.

Maps a batch of SAR look counts to dense embedding vectors that are added
to the diffusion timestep embedding inside ``UNetModel.forward``.  The
combined vector is then consumed by every ``ResBlock`` in the network via
its ``emb_layers``, providing FiLM-style (scale + shift) conditioning when
``use_scale_shift_norm=True`` (the default SAR-DDPM configuration).

Design contract
---------------
- ``UNetModel`` and ``ResBlock`` are NOT modified by this module.
- ``SynthSARDataset`` / ``MultiLookDataset`` are NOT modified.
- This module is a pure ``nn.Module``; it owns no training loop logic.
- The public API is a single ``forward(look_num)`` call returning a tensor
  of shape ``[B, time_embed_dim]`` that callers add to ``emb``:

      emb = emb + look_embedding(look_num)

Look count → embedding index mapping
--------------------------------------
``nn.Embedding`` requires contiguous integer indices in ``[0, num_embeddings)``.
The supported look values ``{1, 2, 4, 8, 10}`` are not contiguous, so a fixed
lookup table maps each valid count to a dense index:

    look count  →  embedding index
    1           →  0
    2           →  1
    4           →  2
    8           →  3
    10          →  4

The mapping is registered as a non-trainable buffer so it is saved and
restored with the model ``state_dict`` and moves to the correct device
automatically.

Initialisation
--------------
Weights are initialised to zero (``nn.init.zeros_``).  At the start of
A2 fine-tuning from an A1 checkpoint the look embedding contributes
exactly zero to ``emb``, so A2 epoch 0 is mathematically identical to
the A1 model.  Symmetry between look slots is broken by the data
distribution — each row only receives gradients from samples with that
specific look count — so no random perturbation is needed.

Checkpoint compatibility
------------------------
This module adds exactly one new ``state_dict`` key:
    ``look_emb.embedding.weight``  shape ``[5, time_embed_dim]``

The ``look_to_idx`` buffer is registered with ``persistent=False`` and
does not appear in ``state_dict``.

A1 checkpoints do not contain this key, so ``load_state_dict`` must be
called with ``strict=False`` when loading A1 weights into an A2 model.
All other parameter shapes are unchanged.
"""

from __future__ import annotations

from typing import Final, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The ordered tuple of supported SAR look counts.
#: The position of each value in this tuple is its embedding index.
LOOK_VALUES: Final[Tuple[int, ...]] = (1, 2, 4, 8, 10)

#: Number of distinct look counts — equals ``nn.Embedding.num_embeddings``.
NUM_LOOKS: Final[int] = len(LOOK_VALUES)  # 5


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class LookEmbedding(nn.Module):
    """Learnable embedding that conditions the diffusion U-Net on look count.

    Each supported look count ``L ∈ {1, 2, 4, 8, 10}`` is mapped to a
    trainable vector of dimension ``time_embed_dim``.  The vector is added
    to the timestep embedding inside ``UNetModel.forward`` before it is
    distributed to every ``ResBlock``.

    Parameters
    ----------
    time_embed_dim:
        Dimensionality of the timestep embedding produced by
        ``UNetModel.time_embed``.  Must equal ``model_channels * 4`` as set
        in ``UNetModel.__init__``.  For the default SAR-DDPM configuration
        (``--num_channels 192``) this is ``768``.

    Attributes
    ----------
    embedding : nn.Embedding
        Shape ``[5, time_embed_dim]``.  Row ``i`` is the conditioning vector
        for the look count at position ``i`` in ``LOOK_VALUES``.
    look_to_idx : torch.Tensor
        Non-trainable buffer of shape ``[11]`` (indices ``0`` … ``10``).
        ``look_to_idx[L]`` gives the embedding row index for look count ``L``.
        Entries for unsupported look counts are set to ``-1``; accessing them
        raises an ``IndexError`` from ``nn.Embedding`` due to
        ``padding_idx``-style protection (see ``forward`` for the explicit
        guard).

    Examples
    --------
    Typical usage inside a modified ``UNetModel.forward``:

    >>> look_emb_module = LookEmbedding(time_embed_dim=768)
    >>> look_num = torch.tensor([1, 4, 10, 2])   # shape [B]
    >>> look_vec = look_emb_module(look_num)      # shape [B, 768]
    >>> emb = time_embed_output + look_vec
    """

    def __init__(self, time_embed_dim: int) -> None:
        super().__init__()

        if time_embed_dim <= 0:
            raise ValueError(
                f"time_embed_dim must be a positive integer, got {time_embed_dim}."
            )

        # ------------------------------------------------------------------
        # Trainable embedding table: 5 rows × time_embed_dim columns.
        # ------------------------------------------------------------------
        self.embedding: nn.Embedding = nn.Embedding(
            num_embeddings=NUM_LOOKS,
            embedding_dim=time_embed_dim,
        )
        # Initialise with small variance so that at the start of A2
        # fine-tuning the look signal is near-zero (warm-start from A1).
        # Zero init: A2 is mathematically identical to A1 at step 0.
        # Symmetry between look slots is broken by data, not by init —
        # each embedding row only receives gradients from samples with
        # that specific look count.
        nn.init.zeros_(self.embedding.weight)

        # ------------------------------------------------------------------
        # Non-trainable lookup buffer: look_count → embedding row index.
        # Size is max(LOOK_VALUES) + 1 = 11 so that any look count in
        # {1,2,4,8,10} can be used as a direct integer index.
        # Unsupported slots hold -1 as a sentinel for the runtime guard.
        # ------------------------------------------------------------------
        max_look: int = max(LOOK_VALUES)
        lut: torch.Tensor = torch.full(
            (max_look + 1,), fill_value=-1, dtype=torch.long
        )
        for idx, look_val in enumerate(LOOK_VALUES):
            lut[look_val] = idx

        # ``register_buffer`` ensures the LUT is:
        #   • saved and restored via ``state_dict`` (as a non-trainable key),
        #   • moved to the correct device by ``.to(device)`` / ``.cuda()``,
        #   • excluded from ``parameters()`` and gradient computation.
        # persistent=False: the LUT is deterministic from LOOK_VALUES and
        # carries no learned state, so it should not appear in state_dict().
        # The only new state_dict key introduced by A2 is look_emb.embedding.weight.
        self.register_buffer("look_to_idx", lut, persistent=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def time_embed_dim(self) -> int:
        """Embedding vector dimensionality (equals ``model_channels * 4``)."""
        return self.embedding.embedding_dim

    @property
    def num_looks(self) -> int:
        """Number of supported distinct look counts."""
        return self.embedding.num_embeddings

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, look_num: torch.Tensor) -> torch.Tensor:
        """Convert a batch of look counts to conditioning vectors.

        Parameters
        ----------
        look_num:
            Integer tensor of shape ``[B]`` containing SAR look counts.
            Every element must be one of ``{1, 2, 4, 8, 10}``.  Values
            are collected by ``MultiLookDataset.__getitem__`` and batched
            by the default ``collate_fn`` into ``torch.int64`` tensors.

        Returns
        -------
        torch.Tensor
            Float tensor of shape ``[B, time_embed_dim]`` on the same device
            as ``look_num``.  Add this directly to the timestep embedding:

                emb = emb + look_embedding(look_num)

        Raises
        ------
        ValueError
            If ``look_num`` is not 1-D or contains values outside the
            supported set ``{1, 2, 4, 8, 10}``.
        """
        if look_num.ndim != 1:
            raise ValueError(
                f"look_num must be a 1-D tensor of shape [B], "
                f"got shape {tuple(look_num.shape)}."
            )

        # Clamp to valid index range before LUT access to produce a clear
        # error rather than a CUDA illegal-memory-access crash.
        max_look: int = self.look_to_idx.shape[0] - 1  # 10
        out_of_range_mask: torch.Tensor = (look_num < 1) | (look_num > max_look)
        if out_of_range_mask.any():
            bad = look_num[out_of_range_mask].tolist()
            raise ValueError(
                f"look_num contains values outside [1, {max_look}]: {bad}. "
                f"Supported look counts are {list(LOOK_VALUES)}."
            )

        # Map raw look counts to contiguous embedding indices via the LUT.
        # look_to_idx lives on the same device as the module (moved by .to()).
        indices: torch.Tensor = self.look_to_idx[look_num]  # [B], dtype long

        # Guard against unsupported values that happen to be within [1, 10]
        # but are not in LOOK_VALUES (e.g., 3, 5, 6, 7, 9).
        unsupported_mask: torch.Tensor = indices == -1
        if unsupported_mask.any():
            bad = look_num[unsupported_mask].tolist()
            raise ValueError(
                f"look_num contains unsupported look counts: {bad}. "
                f"Supported look counts are {list(LOOK_VALUES)}."
            )

        # Embedding lookup: [B] → [B, time_embed_dim]
        return self.embedding(indices)
