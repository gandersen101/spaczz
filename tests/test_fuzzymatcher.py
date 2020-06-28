"""Tests for the fuzzymatcher module."""
from typing import List, Tuple

import pytest
import spacy
from spacy.tokens import Doc, Span

from spaczz.matcher.fuzzymatcher import FuzzyMatcher

# Global Variables
nlp = spacy.blank("en")
doc = nlp("The Heffer, 'mooooo, I'm a cow.' There's also a chken named Stephen.")
animals = ["Heifer", "chicken"]
fm = FuzzyMatcher(
    nlp.vocab
)  # This FuzzyMatcher instance will be updated through the testing.


def add_name_ent(
    matcher: FuzzyMatcher, doc: Doc, i: int, matches: List[Tuple[str, int, int]]
):
    """Callback on match function for later testing. Adds "NAME" entities to doc."""
    match_id, start, end = matches[i]
    entity = Span(doc, start, end, label="NAME")
    doc.ents += (entity,)


def test_add():
    """It adds the "SOUND" label and a pattern to the matcher."""
    fm.add("SOUND", [nlp.make_doc("mooo")])
    assert "SOUND" in fm


def test_add_with_kwargs():
    """It adds the "ANIMAL" label and some patterns with kwargs to the matcher."""
    fm.add(
        "ANIMAL",
        [nlp.make_doc(animal) for animal in animals],
        [{"ignore_case": False}, {}],
    )
    assert fm.patterns == [
        {"label": "SOUND", "pattern": "mooo", "type": "fuzzy"},
        {
            "label": "ANIMAL",
            "pattern": "Heifer",
            "type": "fuzzy",
            "kwargs": {"ignore_case": False},
        },
        {"label": "ANIMAL", "pattern": "chicken", "type": "fuzzy"},
    ]


def test_add_with_more_patterns_than_explicit_kwargs_warns():
    """It will warn when more patterns are added than explicit kwargs."""
    temp_fm = FuzzyMatcher(nlp.vocab)
    with pytest.warns(UserWarning):
        temp_fm.add(
            "TEST",
            [nlp.make_doc("Test1"), nlp.make_doc("Test2")],
            [{"ignore_case": False}],
        )


def test_add_with_more_explicit_kwargs_than_patterns_warns():
    """It will warn when more explicit kwargs are added than patterns."""
    temp_fm = FuzzyMatcher(nlp.vocab)
    with pytest.warns(UserWarning):
        temp_fm.add(
            "TEST",
            [nlp.make_doc("Test1")],
            [{"ignore_case": False}, {"ignore_case": False}],
        )


def test_add_without_doc_objects_raises_error():
    """Trying to add non Doc objects as patterns raises a TypeError."""
    temp_fm = FuzzyMatcher(nlp.vocab)
    with pytest.raises(TypeError):
        temp_fm.add("TEST", ["Test1"])


def test_add_where_kwargs_are_not_dicts_raises_error():
    """Trying to add non Dict objects as kwargs raises a TypeError."""
    temp_fm = FuzzyMatcher(nlp.vocab)
    with pytest.raises(TypeError):
        temp_fm.add("TEST", [nlp.make_doc("Test1")], ["ignore_case"])


def test_len_returns_count_of_labels_in_matcher():
    """It returns the correct length of labels."""
    assert len(fm) == 2


def test_in_returns_bool_of_label_in_matcher():
    """It returns whether a label is present."""
    assert "ANIMAL" in fm


def test_labels_returns_label_names():
    """It returns a tuple of all unique label names."""
    assert all(label in fm.labels for label in ("SOUND", "ANIMAL"))


def test_remove_label():
    """It removes a label from the matcher."""
    fm.add("TEST", [nlp.make_doc("test")])
    assert "TEST" in fm
    fm.remove("TEST")
    assert "TEST" not in fm


def test_remove_label_raises_error_if_label_not_in_matcher():
    """It raises a ValueError if trying to remove a label not present."""
    with pytest.raises(ValueError):
        fm.remove("TEST")


def test_matcher_returns_matches():
    """Calling the matcher on a Doc object returns matches."""
    assert fm(doc) == [("ANIMAL", 1, 2), ("SOUND", 4, 5), ("ANIMAL", 16, 17)]


def test_matcher_returns_empty_list_if_no_matches():
    """Calling the matcher on a Doc object with no viable matches returns empty list."""
    temp_doc = nlp("No matches here.")
    assert fm(temp_doc) == []


def test_matcher_uses_on_match_callback():
    """It utilizes callback on match functions passed when called on a Doc object."""
    fm.add("NAME", [nlp.make_doc("Steven")], on_match=add_name_ent)
    fm(doc)
    ent_text = [ent.text for ent in doc.ents]
    assert "Stephen" in ent_text


def test_matcher_pipe():
    """It returns a stream of Doc objects."""
    doc_stream = (
        nlp.make_doc("test doc 1: Corvold"),
        nlp.make_doc("test doc 2: Prosh"),
    )
    temp_fm = FuzzyMatcher(nlp.vocab)
    output = temp_fm.pipe(doc_stream)
    assert list(output) == list(doc_stream)


def test_matcher_pipe_with_context():
    """It returns a stream of Doc objects as tuples with context."""
    doc_stream = (
        (nlp.make_doc("test doc 1: Corvold"), "Jund"),
        (nlp.make_doc("test doc 2: Prosh"), "Jund"),
    )
    temp_fm = FuzzyMatcher(nlp.vocab)
    output = temp_fm.pipe(doc_stream, as_tuples=True)
    assert list(output) == list(doc_stream)


def test_matcher_pipe_with_matches():
    """It returns a stream of Doc objects and matches as tuples."""
    doc_stream = (
        nlp.make_doc("test doc 1: Corvold"),
        nlp.make_doc("test doc 2: Prosh"),
    )
    temp_fm = FuzzyMatcher(nlp.vocab)
    temp_fm.add("DRAGON", [nlp.make_doc("Korvold"), nlp.make_doc("Prossh")])
    output = temp_fm.pipe(doc_stream, return_matches=True)
    matches = [entry[1] for entry in output]
    assert matches == [[("DRAGON", 4, 5)], [("DRAGON", 4, 5)]]


def test_matcher_pipe_with_matches_and_context():
    """It returns a stream of Doc objects and matches as a tuple
    within a tuple that contains context.
    """
    doc_stream = (
        (nlp.make_doc("test doc 1: Corvold"), "Jund"),
        (nlp.make_doc("test doc 2: Prosh"), "Jund"),
    )
    temp_fm = FuzzyMatcher(nlp.vocab)
    temp_fm.add("DRAGON", [nlp.make_doc("Korvold"), nlp.make_doc("Prossh")])
    output = temp_fm.pipe(doc_stream, return_matches=True, as_tuples=True)
    matches = [(entry[0][1], entry[1]) for entry in output]
    assert matches == [([("DRAGON", 4, 5)], "Jund"), ([("DRAGON", 4, 5)], "Jund")]
