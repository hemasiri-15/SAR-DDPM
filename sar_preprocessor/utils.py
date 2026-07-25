"""Shared utility helpers: logging setup and small filesystem helpers.

Nothing in this module knows about SAR-specific concepts; it is kept
generic so both readers and the top-level pipeline can depend on it
without creating circular imports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.

    The root `sar_preprocessor` logger is configured exactly once (on the
    first call to this function) with a simple stream handler. Subsequent
    calls just return `logging.getLogger(name)`, so submodules can safely
    call this at import time.

    Args:
        name: Usually `__name__` of the calling module.

    Returns:
        A `logging.Logger` instance.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)


def walk_limited_depth(root: Path, max_depth: int) -> Iterator[Path]:
    """Yield all files and directories under `root` up to `max_depth`.

    Depth 0 is `root` itself's direct children. This avoids the cost (and
    surprise matches) of an unbounded `Path.rglob` on large directory
    trees, which matters for detection heuristics that only need to look
    a couple of levels down.

    Args:
        root: Directory to walk.
        max_depth: Maximum depth to descend, relative to `root`.

    Yields:
        Path objects for every file/directory encountered within the
        depth limit.
    """
    if not root.is_dir():
        return
    root_depth = len(root.parts)
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = list(current.iterdir())
        except (PermissionError, FileNotFoundError):
            continue
        for child in children:
            yield child
            child_depth = len(child.parts) - root_depth
            if child.is_dir() and child_depth < max_depth:
                stack.append(child)


def find_first(root: Path, pattern: str, max_depth: Optional[int] = None) -> Optional[Path]:
    """Return the first path under `root` matching a glob `pattern`.

    Args:
        root: Directory to search.
        pattern: Glob pattern, e.g. "*.SAFE" or "*.ann".
        max_depth: If given, restrict the search to this many levels
            below `root` using `walk_limited_depth`. If None, searches
            the full tree via `Path.rglob`.

    Returns:
        The first matching Path, or None if nothing matches.
    """
    if max_depth is None:
        return next(root.rglob(pattern), None)
    for candidate in walk_limited_depth(root, max_depth):
        if candidate.match(pattern):
            return candidate
    return None


def ensure_dir(path: Path) -> Path:
    """Create `path` (and parents) if it doesn't exist; return it.

    Args:
        path: Directory path to create.

    Returns:
        The same path, guaranteed to exist as a directory.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
