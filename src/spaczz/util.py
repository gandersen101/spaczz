"""Module for various utility functions."""
from __future__ import annotations

from collections import defaultdict
from functools import partial
from itertools import repeat, tee
from pathlib import Path
from typing import Any, Iterable, Union


def ensure_path(path: Union[str, Path]) -> Path:
    """Ensure string is converted to a Path.

    Args:
        path: Anything. If string, it's converted to Path.

    Returns:
        Path or original argument.
    """
    if isinstance(path, str):
        return Path(path)
    else:
        return path


def nest_defaultdict(default_factory: Any, depth: int = 1) -> defaultdict[Any, Any]:
    """Returns a nested `defaultdict`."""
    result = partial(defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(defaultdict, result)
    return result()


def n_wise(iterable: Iterable[Any], n: int) -> Iterable[Any]:
    """Iterates over an iterables in slices of length n by one step at a time."""
    iterables = tee(iterable, n)
    for i in range(len(iterables)):
        for _ in range(i):
            next(iterables[i], None)
    return zip(*iterables)


def read_from_disk(path: Union[str, Path], readers: Any, exclude: Any) -> Path:
    """Reads a pipeline component from disk."""
    path = ensure_path(path)
    for key, reader in readers.items():
        # Split to support file names like meta.json
        if key.split(".")[0] not in exclude:
            reader(path / key)
    return path


def write_to_disk(path: Union[str, Path], writers: Any, exclude: Any) -> Path:
    """Writes a pipeline component to disk."""
    path = ensure_path(path)
    if not path.exists():
        path.mkdir()
    for key, writer in writers.items():
        # Split to support file names like meta.json
        if key.split(".")[0] not in exclude:
            writer(path / key)
    return path
