"""Standalone Transformer building blocks for structdiff.

Phase 1: reusable, UNet-agnostic Transformer block operating on
[B, C, H, W] feature maps. Not yet wired into guided_diffusion/unet.py.
"""

from .transformer_block import TransformerBlock

__all__ = ["TransformerBlock"]
