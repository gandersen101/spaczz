"""Module for various utility functions."""
from __future__ import annotations

from pathlib import Path
from typing import Any, DefaultDict, Union

from spacy.language import Vocab

from ._types import MatcherCallback


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


def read_from_disk(path: Union[str, Path], readers: Any, exclude: Any) -> Path:
    """Reads a pipeline component from disk."""
    path = ensure_path(path)
    for key, reader in readers.items():
        # Split to support file names like meta.json
        if key.split(".")[0] not in exclude:
            reader(path / key)
    return path


def unpickle_matcher(
    matcher: Any,
    vocab: Vocab,
    patterns: DefaultDict[str, DefaultDict[str, Any]],
    callbacks: dict[str, MatcherCallback],
    defaults: Any,
) -> Any:
    """Will return a matcher from pickle protocol."""
    matcher_instance = matcher(vocab, **defaults)
    for key, specs in patterns.items():
        callback = callbacks.get(key)
        if isinstance(specs, dict):
            matcher_instance.add(
                key, specs["patterns"], specs["kwargs"], on_match=callback
            )
        else:
            matcher_instance.add(key, specs, on_match=callback)
    return matcher_instance


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
