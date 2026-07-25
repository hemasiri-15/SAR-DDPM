"""sar_preprocessor: detect, read, normalize, patchify, and export SAR datasets.

Pipeline:

    detect dataset -> reader.read() -> normalize() -> patchify() -> export()

See `build_dataset.process_dataset` for the orchestrated pipeline and
`build_dataset.main` for the CLI entry point.
"""

from .scene import SARScene, make_metadata

__all__ = ["SARScene", "make_metadata"]
