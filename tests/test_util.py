"""Tests for utils."""
from pathlib import Path

from spaczz.util import ensure_path
from spaczz.util import nest_defaultdict


def test_nest_defaultdict() -> None:
    """It nests a defaultdict."""
    d = nest_defaultdict(list)
    d["a"]["b"].append(1)
    assert d["a"]["b"] == [1]


def test_nest_defaultdict2() -> None:
    """It nests a defaultdict."""
    d = nest_defaultdict(list, depth=2)
    d["a"]["b"]["c"].append(1)
    assert d["a"]["b"]["c"] == [1]


def test_ensure_path_w_not_path() -> None:
    """It converts a PathLike to a Path."""
    assert isinstance(ensure_path("test"), Path)


def test_ensure_path_w_path() -> None:
    """It keeps a Path as a Path."""
    assert isinstance(ensure_path(Path("test")), Path)
