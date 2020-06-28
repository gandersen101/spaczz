"""Tests for the regexmatcher module."""
from typing import List, Tuple

import pytest
import spacy
from spacy.tokens import Doc, Span

from spaczz.matcher.regexmatcher import RegexMatcher

# Global Variables
nlp = spacy.blank("en")
doc = nlp.make_doc("I live at 555 Fake Street in Generic, TN 55555.")
rm = RegexMatcher(
    nlp.vocab
)  # This RegexMatcher instance will be updated through the testing.


def add_gpe_ent(
    matcher: RegexMatcher, doc: Doc, i: int, matches: List[Tuple[str, int, int]]
):
    """Callback on match function for later testing. Adds "GPE" entities to doc."""
    match_id, start, end = matches[i]
    entity = Span(doc, start, end, label="GPE")
    doc.ents += (entity,)


def test_add():
    """It adds the "GPE" label and a pattern to the matcher."""
    temp_rm = RegexMatcher(nlp.vocab)
    temp_rm.add("GPE", ["[U](nited|\\.?) ?[S](tates|\\.?)"])
    assert "GPE" in temp_rm


def test_add_with_kwargs():
    """It adds the "GPE" label and a pattern with some kwargs to the matcher."""
    temp_rm = RegexMatcher(nlp.vocab)
    temp_rm.add(
        "GPE", ["[U](nited|\\.?) ?[S](tates|\\.?)"], kwargs=[{"ignore_case": True}]
    )
    assert temp_rm.patterns == [
        {
            "label": "GPE",
            "pattern": "[U](nited|\\.?) ?[S](tates|\\.?)",
            "type": "regex",
            "kwargs": {"ignore_case": True},
        }
    ]


def test_add_with_more_patterns_than_explicit_kwargs_warns():
    """It will warn when more patterns are added than explicit kwargs."""
    temp_rm = RegexMatcher(nlp.vocab)
    with pytest.warns(UserWarning):
        temp_rm.add("TEST", ["Test1", "Test2"], [{"ignore_case": True}])


def test_add_with_more_explicit_kwargs_than_patterns_warns():
    """It will warn when more explicit kwargs are added than patterns."""
    temp_rm = RegexMatcher(nlp.vocab)
    with pytest.warns(UserWarning):
        temp_rm.add("TEST", ["Test1"], [{"ignore_case": True}, {"ignore_case": True}])


def test_add_without_string_pattern_raises_error():
    """Trying to add non strings as patterns raises a TypeError."""
    temp_rm = RegexMatcher(nlp.vocab)
    with pytest.raises(TypeError):
        temp_rm.add("TEST", [nlp.make_doc("Test1")])


def test_add_str_pattern_outside_list_raises_error():
    """Trying to add string as patterns, not iterable of strings, raises a TypeError."""
    temp_rm = RegexMatcher(nlp.vocab)
    with pytest.raises(TypeError):
        temp_rm.add("TEST", "Test1")


def test_add_where_kwargs_are_not_dicts_raises_error():
    """Trying to add non Dict objects as kwargs raises a TypeError."""
    temp_rm = RegexMatcher(nlp.vocab)
    with pytest.raises(TypeError):
        temp_rm.add("TEST", ["Test1"], ["ignore_case"])


def test_len_returns_count_of_labels_in_matcher():
    """It returns the correct length of labels."""
    temp_rm = RegexMatcher(nlp.vocab)
    temp_rm.add("GPE", ["[U](nited|\\.?) ?[S](tates|\\.?)"])
    assert len(temp_rm) == 1


def test_in_returns_bool_of_label_in_matcher():
    """It returns whether a label is present."""
    temp_rm = RegexMatcher(nlp.vocab)
    temp_rm.add("GPE", ["[U](nited|\\.?) ?[S](tates|\\.?)"])
    assert "GPE" in temp_rm


def test_labels_returns_label_names():
    """It returns a tuple of all unique label names."""
    temp_rm = RegexMatcher(nlp.vocab)
    temp_rm.add("GPE", ["[U](nited|\\.?) ?[S](tates|\\.?)"])
    assert temp_rm.labels == ("GPE",)


def test_remove_label():
    """It removes a label from the matcher."""
    rm.add("TEST", ["test"])
    assert "TEST" in rm
    rm.remove("TEST")
    assert "TEST" not in rm


def test_remove_label_raises_error_if_label_not_in_matcher():
    """It raises a ValueError if trying to remove a label not present."""
    with pytest.raises(ValueError):
        rm.remove("TEST")


def test_matcher_returns_matches():
    """Calling the matcher on a Doc object returns matches."""
    rm.add("STREET", ["street_addresses"], [{"predef": True}])
    rm.add("ZIP", ["zip_codes"], [{"predef": True}])
    rm.add("GPE", ["TN", "Generic"])
    assert rm(doc) == [("STREET", 3, 6), ("GPE", 7, 8), ("GPE", 9, 10), ("ZIP", 10, 11)]


def test_matcher_returns_empty_list_if_no_matches():
    """Calling the matcher on a Doc object with no viable matches returns empty list."""
    temp_doc = nlp("No matches here.")
    assert rm(temp_doc) == []


def test_matcher_uses_on_match_callback():
    """It utilizes callback on match functions passed when called on a Doc object."""
    temp_doc = nlp.make_doc("I live in the united states, or the US.")
    temp_rm = RegexMatcher(nlp.vocab)
    temp_rm.add(
        "GPE",
        ["[U](nited|\\.?) ?[S](tates|\\.?)"],
        kwargs=[{"ignore_case": True}],
        on_match=add_gpe_ent,
    )
    temp_rm(temp_doc)
    assert [ent.label_ for ent in temp_doc.ents] == ["GPE", "GPE"]


def test_patterns_with_and_without_kwargs():
    """It adds the "GPE" label and a pattern with some kwargs to the matcher."""
    temp_rm = RegexMatcher(nlp.vocab)
    temp_rm.add("GPE", ["[U](nited|\\.?) ?[S](tates|\\.?)"])
    temp_rm.add("ZIP", ["zip_codes"], [{"predef": True}])
    assert temp_rm.patterns == [
        {
            "label": "GPE",
            "pattern": "[U](nited|\\.?) ?[S](tates|\\.?)",
            "type": "regex",
        },
        {
            "label": "ZIP",
            "pattern": "zip_codes",
            "type": "regex",
            "kwargs": {"predef": True},
        },
    ]


def test_matcher_pipe():
    """It returns a stream of Doc objects."""
    doc_stream = (
        nlp.make_doc("test doc 1: United States"),
        nlp.make_doc("test doc 2: US"),
    )
    temp_rm = RegexMatcher(nlp.vocab)
    output = temp_rm.pipe(doc_stream)
    assert list(output) == list(doc_stream)


def test_matcher_pipe_with_context():
    """It returns a stream of Doc objects as tuples with context."""
    doc_stream = (
        (nlp.make_doc("test doc 1: United States"), "Country"),
        (nlp.make_doc("test doc 2: US"), "Country"),
    )
    temp_rm = RegexMatcher(nlp.vocab)
    output = temp_rm.pipe(doc_stream, as_tuples=True)
    assert list(output) == list(doc_stream)


def test_matcher_pipe_with_matches():
    """It returns a stream of Doc objects and matches as tuples."""
    doc_stream = (
        nlp.make_doc("test doc 1: United States"),
        nlp.make_doc("test doc 2: US"),
    )
    temp_rm = RegexMatcher(nlp.vocab)
    temp_rm.add("GPE", ["[Uu](nited|\\.?) ?[Ss](tates|\\.?)"])
    output = temp_rm.pipe(doc_stream, return_matches=True)
    matches = [entry[1] for entry in output]
    assert matches == [[("GPE", 4, 6)], [("GPE", 4, 5)]]


def test_matcher_pipe_with_matches_and_context():
    """It returns a stream of Doc objects and matches as a tuple
    within a tuple that contains context.
    """
    doc_stream = (
        (nlp.make_doc("test doc 1: United States"), "Country"),
        (nlp.make_doc("test doc 2: US"), "Country"),
    )
    temp_rm = RegexMatcher(nlp.vocab)
    temp_rm.add("GPE", ["[Uu](nited|\\.?) ?[Ss](tates|\\.?)"])
    output = temp_rm.pipe(doc_stream, return_matches=True, as_tuples=True)
    matches = [(entry[0][1], entry[1]) for entry in output]
    assert matches == [([("GPE", 4, 6)], "Country"), ([("GPE", 4, 5)], "Country")]
