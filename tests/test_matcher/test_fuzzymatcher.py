"""Tests for the fuzzymatcher module."""
from typing import List, Tuple

import pytest
from spacy.language import Language
from spacy.tokens import Doc, Span

from spaczz.exceptions import KwargsWarning
from spaczz.matcher.fuzzymatcher import FuzzyMatcher


def add_name_ent(
    matcher: FuzzyMatcher, doc: Doc, i: int, matches: List[Tuple[str, int, int]]
) -> None:
    """Callback on match function for later testing. Adds "NAME" entities to doc."""
    match_id, start, end = matches[i]
    entity = Span(doc, start, end, label="NAME")
    doc.ents += (entity,)


@pytest.fixture
def test_doc(nlp: Language) -> Doc:
    """Doc for testing."""
    return nlp.make_doc(
        "The Heffer, 'mooooo, I'm a cow.' There's also a chken named Stephen."
    )


@pytest.fixture
def matcher_with_patterns(nlp: Language,) -> FuzzyMatcher:
    """Fuzzy matcher with patterns added."""
    animals = ["Heifer", "chicken"]
    sounds = ["mooo"]
    names = ["Steven"]
    fm = FuzzyMatcher(nlp.vocab)
    fm.add(
        "ANIMAL",
        [nlp.make_doc(animal) for animal in animals],
        kwargs=[{"ignore_case": False}, {}],
    )
    fm.add("SOUND", [nlp.make_doc(sound) for sound in sounds])
    fm.add("NAME", [nlp.make_doc(name) for name in names], on_match=add_name_ent)
    return fm


def test_adding_patterns(matcher_with_patterns: FuzzyMatcher) -> None:
    """It adds the "ANIMAL" label and some patterns with kwargs to the matcher."""
    assert matcher_with_patterns.patterns == [
        {
            "label": "ANIMAL",
            "pattern": "Heifer",
            "type": "fuzzy",
            "kwargs": {"ignore_case": False},
        },
        {"label": "ANIMAL", "pattern": "chicken", "type": "fuzzy"},
        {"label": "SOUND", "pattern": "mooo", "type": "fuzzy"},
        {"label": "NAME", "pattern": "Steven", "type": "fuzzy"},
    ]


def test_add_with_more_patterns_than_explicit_kwargs_warns(
    matcher_with_patterns: FuzzyMatcher, nlp: Language
) -> None:
    """It will warn when more patterns are added than explicit kwargs."""
    with pytest.warns(KwargsWarning):
        matcher_with_patterns.add(
            "TEST",
            [nlp.make_doc("Test1"), nlp.make_doc("Test2")],
            [{"ignore_case": False}],
        )


def test_add_with_more_explicit_kwargs_than_patterns_warns(
    matcher_with_patterns: FuzzyMatcher, nlp: Language
) -> None:
    """It will warn when more explicit kwargs are added than patterns."""
    with pytest.warns(KwargsWarning):
        matcher_with_patterns.add(
            "TEST",
            [nlp.make_doc("Test1")],
            [{"ignore_case": False}, {"ignore_case": False}],
        )


def test_add_without_doc_objects_raises_error(
    matcher_with_patterns: FuzzyMatcher,
) -> None:
    """Trying to add non Doc objects as patterns raises a TypeError."""
    with pytest.raises(TypeError):
        matcher_with_patterns.add("TEST", ["Test1"])


def test_add_where_kwargs_are_not_dicts_raises_error(
    matcher_with_patterns: FuzzyMatcher, nlp: Language
) -> None:
    """Trying to add non Dict objects as kwargs raises a TypeError."""
    with pytest.raises(TypeError):
        matcher_with_patterns.add("TEST", [nlp.make_doc("Test1")], ["ignore_case"])


def test_len_returns_count_of_labels_in_matcher(
    matcher_with_patterns: FuzzyMatcher,
) -> None:
    """It returns the sum of unique labels in the matcher."""
    assert len(matcher_with_patterns) == 3


def test_in_returns_bool_of_label_in_matcher(
    matcher_with_patterns: FuzzyMatcher,
) -> None:
    """It returns whether a label is in the matcher."""
    assert "ANIMAL" in matcher_with_patterns


def test_labels_returns_label_names(matcher_with_patterns: FuzzyMatcher) -> None:
    """It returns a tuple of all unique label names."""
    assert all(
        label in matcher_with_patterns.labels for label in ("SOUND", "ANIMAL", "NAME")
    )


def test_remove_label(matcher_with_patterns: FuzzyMatcher, nlp: Language) -> None:
    """It removes a label from the matcher."""
    matcher_with_patterns.add("TEST", [nlp.make_doc("test")])
    assert "TEST" in matcher_with_patterns
    matcher_with_patterns.remove("TEST")
    assert "TEST" not in matcher_with_patterns


def test_remove_label_raises_error_if_label_not_in_matcher(
    matcher_with_patterns: FuzzyMatcher,
) -> None:
    """It raises a ValueError if trying to remove a label not present."""
    with pytest.raises(ValueError):
        matcher_with_patterns.remove("TEST")


def test_matcher_returns_matches(
    matcher_with_patterns: FuzzyMatcher, test_doc: Doc
) -> None:
    """Calling the matcher on a Doc object returns matches."""
    assert matcher_with_patterns(test_doc) == [
        ("ANIMAL", 1, 2),
        ("SOUND", 4, 5),
        ("ANIMAL", 16, 17),
        ("NAME", 18, 19),
    ]


def test_matcher_returns_empty_list_if_no_matches(
    matcher_with_patterns: FuzzyMatcher, nlp: Language
) -> None:
    """Calling the matcher on a Doc object with no viable matches returns empty list."""
    temp_doc = nlp.make_doc("No matches here.")
    assert matcher_with_patterns(temp_doc) == []


def test_matcher_uses_on_match_callback(
    matcher_with_patterns: FuzzyMatcher, test_doc: Doc, nlp: Language
) -> None:
    """It utilizes callback on match functions passed when called on a Doc object."""
    matcher_with_patterns(test_doc)
    ent_text = [ent.text for ent in test_doc.ents]
    assert "Stephen" in ent_text


def test_matcher_pipe(nlp: Language) -> None:
    """It returns a stream of Doc objects."""
    doc_stream = (
        nlp.make_doc("test doc 1: Corvold"),
        nlp.make_doc("test doc 2: Prosh"),
    )
    fm = FuzzyMatcher(nlp.vocab)
    output = fm.pipe(doc_stream)
    assert list(output) == list(doc_stream)


def test_matcher_pipe_with_context(nlp: Language) -> None:
    """It returns a stream of Doc objects as tuples with context."""
    doc_stream = (
        (nlp.make_doc("test doc 1: Corvold"), "Jund"),
        (nlp.make_doc("test doc 2: Prosh"), "Jund"),
    )
    fm = FuzzyMatcher(nlp.vocab)
    output = fm.pipe(doc_stream, as_tuples=True)
    assert list(output) == list(doc_stream)


def test_matcher_pipe_with_matches(nlp: Language) -> None:
    """It returns a stream of Doc objects and matches as tuples."""
    doc_stream = (
        nlp.make_doc("test doc 1: Corvold"),
        nlp.make_doc("test doc 2: Prosh"),
    )
    fm = FuzzyMatcher(nlp.vocab)
    fm.add("DRAGON", [nlp.make_doc("Korvold"), nlp.make_doc("Prossh")])
    output = fm.pipe(doc_stream, return_matches=True)
    matches = [entry[1] for entry in output]
    assert matches == [[("DRAGON", 4, 5)], [("DRAGON", 4, 5)]]


def test_matcher_pipe_with_matches_and_context(nlp: Language) -> None:
    """It returns a stream of Doc objects and matches and context as tuples."""
    doc_stream = (
        (nlp.make_doc("test doc 1: Corvold"), "Jund"),
        (nlp.make_doc("test doc 2: Prosh"), "Jund"),
    )
    fm = FuzzyMatcher(nlp.vocab)
    fm.add("DRAGON", [nlp.make_doc("Korvold"), nlp.make_doc("Prossh")])
    output = fm.pipe(doc_stream, return_matches=True, as_tuples=True)
    matches = [(entry[0][1], entry[1]) for entry in output]
    assert matches == [([("DRAGON", 4, 5)], "Jund"), ([("DRAGON", 4, 5)], "Jund")]