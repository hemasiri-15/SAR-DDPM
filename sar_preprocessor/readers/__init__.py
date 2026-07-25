"""Reader modules for each supported SAR dataset type.

This package intentionally contains no orchestration logic. Each
submodule exposes:

    detect(path: Path) -> bool
    read(path: Path) -> SARScene

The list of readers and the order in which they are tried lives in
`sar_preprocessor.build_dataset`, not here.
"""

from . import airsar, dsifn, sentinel1, uavsar

__all__ = ["sentinel1", "dsifn", "uavsar", "airsar"]
