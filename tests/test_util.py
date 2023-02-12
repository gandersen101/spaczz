"""Tests for utils."""
from pathlib import Path

from spacy.language import Language

from spaczz.util import ensure_path
from spaczz.util import filter_overlapping_matches
from spaczz.util import map_chars_to_tokens
from spaczz.util import nest_defaultdict


def test_filter_overlapping_matches() -> None:
    """It only returns the first match if more than one encompass the same tokens."""
    matches = [(1, 2, 80, "test"), (1, 3, 70, "test")]
    assert filter_overlapping_matches(matches) == [(1, 2, 80, "test")]


def test_map_chars_to_tokens(nlp: Language) -> None:
    """It creates map of character indices to token indices."""
    doc = nlp("Test sentence.")
    char_to_token_map = map_chars_to_tokens(doc)
    assert char_to_token_map[0] == 0
    assert char_to_token_map[5] == 1
    assert char_to_token_map[13] == 2
    assert len(char_to_token_map) == 13


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
