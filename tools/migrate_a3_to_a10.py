"""
migrate_a3_to_a10.py
====================
Checkpoint migration utility: A3 → A10.

Renames all ``struct_encoder.*`` keys in an A3 checkpoint to
``ms_struct_encoder.shared_enc.*`` so they are recognised by the A10
``MultiScaleStructTensorEncoder.shared_enc`` attribute.

The new ``ms_struct_encoder.scale_emb.*`` parameters are NOT present in
the A3 checkpoint and are absent from the migrated file.  They will be
initialised to near-zero when the A10 model loads the migrated checkpoint
via ``load_state_dict(strict=False)``.

Key mapping
-----------
    A3 key                              A10 key
    ──────────────────────────────────  ────────────────────────────────────────────
    struct_encoder.conv1.weight         ms_struct_encoder.shared_enc.conv1.weight
    struct_encoder.norm1.weight         ms_struct_encoder.shared_enc.norm1.weight
    struct_encoder.norm1.bias           ms_struct_encoder.shared_enc.norm1.bias
    struct_encoder.conv2.weight         ms_struct_encoder.shared_enc.conv2.weight
    struct_encoder.norm2.weight         ms_struct_encoder.shared_enc.norm2.weight
    struct_encoder.norm2.bias           ms_struct_encoder.shared_enc.norm2.bias
    struct_encoder.conv3.weight         ms_struct_encoder.shared_enc.conv3.weight
    struct_encoder.norm3.weight         ms_struct_encoder.shared_enc.norm3.weight
    struct_encoder.norm3.bias           ms_struct_encoder.shared_enc.norm3.bias
    struct_encoder.proj.weight          ms_struct_encoder.shared_enc.proj.weight
    struct_encoder.proj.bias            ms_struct_encoder.shared_enc.proj.bias

All other keys are copied unchanged (UNet weights, look_embedding, etc.).

Usage
-----
    python migrate_a3_to_a10.py \
        --input  checkpoints/model_best_a3.pt \
        --output checkpoints/model_best_a10_init.pt

Then load in A10 training:
    model.load_state_dict(
        torch.load("checkpoints/model_best_a10_init.pt"), strict=False
    )
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Key prefix renaming
# ---------------------------------------------------------------------------

_A3_PREFIX: str = "struct_encoder."
_A10_PREFIX: str = "ms_struct_encoder.shared_enc."


def migrate_state_dict(state_dict: dict) -> dict:
    """Return a new state dict with A3 keys renamed to A10 keys.

    Parameters
    ----------
    state_dict:
        ``OrderedDict`` as returned by ``torch.load`` (or
        ``model.state_dict()``).

    Returns
    -------
    dict
        New dict with renamed keys.  All values are references to the
        original tensors (no copies).

    Raises
    ------
    ValueError
        If no ``struct_encoder.*`` keys are found (likely the wrong
        checkpoint was passed).
    """
    migrated: dict = {}
    renamed_count: int = 0

    for key, value in state_dict.items():
        if key.startswith(_A3_PREFIX):
            new_key = _A10_PREFIX + key[len(_A3_PREFIX):]
            migrated[new_key] = value
            renamed_count += 1
        else:
            migrated[key] = value

    if renamed_count == 0:
        raise ValueError(
            f"No keys starting with '{_A3_PREFIX}' found in the checkpoint. "
            "Are you sure this is an A3 checkpoint?\n"
            f"Keys present: {list(state_dict.keys())[:20]}"
        )

    print(f"Renamed {renamed_count} keys: '{_A3_PREFIX}*' → '{_A10_PREFIX}*'")
    return migrated


def migrate_checkpoint(input_path: str, output_path: str) -> None:
    """Load an A3 checkpoint, migrate keys, and save.

    Parameters
    ----------
    input_path:
        Path to the A3 ``.pt`` file.
    output_path:
        Destination path for the migrated ``.pt`` file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"ERROR: input checkpoint not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if output_path.exists():
        print(
            f"WARNING: output path already exists and will be overwritten: {output_path}",
            file=sys.stderr,
        )

    print(f"Loading A3 checkpoint: {input_path}")
    state_dict = torch.load(input_path, map_location="cpu")

    # Handle the common case where the .pt file is a plain state_dict
    # (as saved by TrainLoop.save) vs a dict with a 'model' key.
    if isinstance(state_dict, dict) and "model" in state_dict:
        # Wrapped format: migrate only the model sub-dict.
        state_dict["model"] = migrate_state_dict(state_dict["model"])
        migrated = state_dict
    else:
        migrated = migrate_state_dict(state_dict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, output_path)
    print(f"Saved migrated A10 checkpoint: {output_path}")

    # Sanity-print first few keys
    keys = list(migrated.keys()) if not isinstance(migrated.get("model"), dict) \
        else list(migrated["model"].keys())
    ms_keys = [k for k in keys if k.startswith("ms_struct_encoder")]
    print(f"A10 ms_struct_encoder keys ({len(ms_keys)}):")
    for k in ms_keys:
        print(f"  {k}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate an A3 SAR-DDPM checkpoint to A10 key layout."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the A3 checkpoint (.pt file).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Destination path for the A10-compatible checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    migrate_checkpoint(args.input, args.output)
