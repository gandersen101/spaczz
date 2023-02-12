"""Module for various utility functions."""
from collections import defaultdict
from functools import partial
import itertools
from os import PathLike
from pathlib import Path
import typing as ty

from spacy.tokens import Doc


def filter_overlapping_matches(
    matches: ty.Iterable[ty.Tuple[int, int, int, str]]
) -> ty.List[ty.Tuple[int, int, int, str]]:
    """Prevents multiple matches from overlapping.

    Expects matches to be pre-sorted by descending ratio
    then ascending start index.
    If more than one match includes the same tokens
    the first of these matches is kept.

    Args:
        matches: `Iterable` of match `Tuple`s (start index, end index, ratio).

    Returns:
        The filtered `List` of match `Tuple`s.

    Example:
        >>> from spaczz.util import filter_overlapping_matches
        >>> matches = [(1, 3, 80), (1, 2, 70)]
        >>> filter_overlapping_matches(matches)
        [(1, 3, 80)]
    """
    filtered_matches: ty.List[ty.Tuple[int, int, int, str]] = []
    for match in matches:
        if not set(range(match[0], match[1])).intersection(
            itertools.chain(*[set(range(n[0], n[1])) for n in filtered_matches])
        ):
            filtered_matches.append(match)
    return filtered_matches


def map_chars_to_tokens(doc: Doc) -> ty.Dict[int, int]:
    """Maps characters in a `Doc` object to tokens."""
    chars_to_tokens = {}
    for token in doc:
        for i in range(token.idx, token.idx + len(token.text)):
            chars_to_tokens[i] = token.i
    return chars_to_tokens


def nest_defaultdict(
    default_factory: ty.Any, depth: int = 1
) -> ty.DefaultDict[ty.Any, ty.Any]:
    """Nests defaultdicts where depth nesting is `defaultdict[default_factory]`."""
    result = partial(defaultdict, default_factory)
    for _ in itertools.repeat(None, depth):
        result = partial(defaultdict, result)
    return result()


def n_wise(iterable: ty.Iterable[ty.Any], n: int) -> ty.Iterable[ty.Any]:
    """Iterates over an iterable in slices of length n by one step at a time."""
    iterables = itertools.tee(iterable, n)
    for i in range(len(iterables)):
        for _ in range(i):
            next(iterables[i], None)
    return zip(*iterables)  # noqa B905


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
