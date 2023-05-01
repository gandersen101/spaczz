"""Module for various utility functions."""
from collections import defaultdict
from functools import partial
import itertools
from os import PathLike
from pathlib import Path
import typing as ty


def nest_defaultdict(
    default_factory: ty.Any, depth: int = 1, *args: ty.Any, **kwargs: ty.Any
) -> ty.DefaultDict[ty.Any, ty.Any]:
    """Nests defaultdicts where depth nesting is `defaultdict[default_factory]`."""
    result = partial(defaultdict, default_factory)
    for _ in itertools.repeat(None, depth):
        result = partial(defaultdict, result)
    return result(*args, **kwargs)


def ensure_path(path: ty.Union[str, PathLike]) -> Path:
    """Ensure str or Pathlike is converted to a Path.

    Args:
        path: str or PathLike, if it's not a Path, it's converted to Path.

    Returns:
        Path.
    """
    if not isinstance(path, Path):
        return Path(path)
    else:
        return path


def read_from_disk(
    path: ty.Union[str, PathLike], readers: ty.Any, exclude: ty.Any
) -> Path:
    """Reads a pipeline component from disk."""
    path = ensure_path(path)
    for key, reader in readers.items():
        # Split to support file names like meta.json
        if key.split(".")[0] not in exclude:
            reader(path / key)
    return path


def write_to_disk(
    path: ty.Union[str, PathLike], writers: ty.Any, exclude: ty.Any
) -> Path:
    """Writes a pipeline component to disk."""
    path = ensure_path(path)
    if not path.exists():
        path.mkdir()
    for key, writer in writers.items():
        # Split to support file names like meta.json
        if key.split(".")[0] not in exclude:
            writer(path / key)
    return path
