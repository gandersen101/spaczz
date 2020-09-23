"""Tests for the regexmatcher module."""
from typing import List, Tuple

import pytest
from spacy.language import Language
from spacy.tokens import Doc, Span

from spaczz.exceptions import KwargsWarning
from spaczz.matcher.regexmatcher import RegexMatcher


def add_gpe_ent(
    matcher: RegexMatcher, doc: Doc, i: int, matches: List[Tuple[str, int, int]]
) -> None:
    """Callback on match function for later testing. Adds "GPE" entities to doc."""
    match_id, start, end, _fuzzy_counts = matches[i]
    entity = Span(doc, start, end, label="GPE")
    doc.ents += (entity,)


@pytest.fixture
def doc(nlp: Language) -> Doc:
    """Doc for testing."""
    return nlp.make_doc("I live at 555 Fake Street in Generic, TN 55555 in the usa.")


@pytest.fixture
def matcher(nlp: Language) -> RegexMatcher:
    """Regex matcher with patterns added."""
    matcher = RegexMatcher(nlp.vocab)
    matcher.add("GPE", ["(?i)[U](nited|\\.?) ?[S](tates|\\.?)"], on_match=add_gpe_ent)
    matcher.add("STREET", ["street_addresses"], kwargs=[{"predef": True}])
    matcher.add("ZIP", ["zip_codes"], kwargs=[{"predef": True}])
    return matcher


def test_adding_patterns(matcher: RegexMatcher) -> None:
    """Regex matcher with patterns added."""
    assert matcher.patterns == [
        {
            "label": "GPE",
            "pattern": "(?i)[U](nited|\\.?) ?[S](tates|\\.?)",
            "type": "regex",
        },
        {
            "label": "STREET",
            "pattern": "street_addresses",
            "type": "regex",
            "kwargs": {"predef": True},
        },
        {
            "label": "ZIP",
            "pattern": "zip_codes",
            "type": "regex",
            "kwargs": {"predef": True},
        },
    ]


def test_add_with_more_patterns_than_explicit_kwargs_warns(
    matcher: RegexMatcher,
) -> None:
    """It will warn when more patterns are added than explicit kwargs."""
    with pytest.warns(KwargsWarning):
        matcher.add("TEST", ["Test1", "Test2"], [{"ignore_case": True}])


def test_add_with_more_explicit_kwargs_than_patterns_warns(
    matcher: RegexMatcher,
) -> None:
    """It will warn when more explicit kwargs are added than patterns."""
    with pytest.warns(KwargsWarning):
        matcher.add("TEST", ["Test1"], [{"ignore_case": True}, {"ignore_case": True}])


def test_add_without_string_pattern_raises_error(
    matcher: RegexMatcher, nlp: Language
) -> None:
    """Trying to add non strings as patterns raises a TypeError."""
    with pytest.raises(TypeError):
        matcher.add("TEST", [nlp.make_doc("Test1")])


def test_add_str_pattern_outside_list_raises_error(matcher: RegexMatcher,) -> None:
    """Trying to add string as patterns, not iterable of strings, raises a TypeError."""
    with pytest.raises(TypeError):
        matcher.add("TEST", "Test1")


def test_add_where_kwargs_are_not_dicts_raises_error(matcher: RegexMatcher,) -> None:
    """Trying to add non Dict objects as kwargs raises a TypeError."""
    with pytest.raises(TypeError):
        matcher.add("TEST", ["Test1"], ["ignore_case"])


def test_len_returns_count_of_labels_in_matcher(matcher: RegexMatcher,) -> None:
    """It returns the correct length of labels."""
    assert len(matcher) == 3


def test_in_returns_bool_of_label_in_matcher(matcher: RegexMatcher,) -> None:
    """It returns whether a label is present."""
    assert "GPE" in matcher


def test_labels_returns_label_names(matcher: RegexMatcher) -> None:
    """It returns a tuple of all unique label names."""
    assert all(label in matcher.labels for label in ("GPE", "ZIP", "STREET"))


def test_remove_label(matcher: RegexMatcher) -> None:
    """It removes a label from the matcher."""
    matcher.add("TEST", ["test"])
    assert "TEST" in matcher
    matcher.remove("TEST")
    assert "TEST" not in matcher


def test_remove_label_raises_error_if_label_not_in_matcher(
    matcher: RegexMatcher,
) -> None:
    """It raises a ValueError if trying to remove a label not present."""
    with pytest.raises(ValueError):
        matcher.remove("TEST")


def test_matcher_returns_matches(matcher: RegexMatcher, doc: Doc) -> None:
    """Calling the matcher on a Doc object returns matches."""
    assert matcher(doc) == [
        ("STREET", 3, 6, (0, 0, 0)),
        ("ZIP", 10, 11, (0, 0, 0)),
        ("GPE", 13, 14, (0, 0, 0)),
    ]


def test_matcher_returns_empty_list_if_no_matches(
    matcher: RegexMatcher, nlp: Language
) -> None:
    """Calling the matcher on a Doc object with no viable matches returns empty list."""
    doc = nlp("No matches here.")
    assert matcher(doc) == []


def test_matcher_uses_on_match_callback(matcher: RegexMatcher, doc: Doc) -> None:
    """It utilizes callback on match functions passed when called on a Doc object."""
    matcher(doc)
    assert "usa" in [ent.text for ent in doc.ents]


def test_matcher_pipe(nlp: Language) -> None:
    """It returns a stream of Doc objects."""
    doc_stream = (
        nlp.make_doc("test doc 1: United States"),
        nlp.make_doc("test doc 2: US"),
    )
    matcher = RegexMatcher(nlp.vocab)
    output = matcher.pipe(doc_stream)
    assert list(output) == list(doc_stream)


def test_matcher_pipe_with_context(nlp: Language) -> None:
    """It returns a stream of Doc objects as tuples with context."""
    doc_stream = (
        (nlp.make_doc("test doc 1: United States"), "Country"),
        (nlp.make_doc("test doc 2: US"), "Country"),
    )
    matcher = RegexMatcher(nlp.vocab)
    output = matcher.pipe(doc_stream, as_tuples=True)
    assert list(output) == list(doc_stream)


def test_matcher_pipe_with_matches(nlp: Language) -> None:
    """It returns a stream of Doc objects and matches as tuples."""
    doc_stream = (
        nlp.make_doc("test doc 1: United States"),
        nlp.make_doc("test doc 2: US"),
    )
    matcher = RegexMatcher(nlp.vocab)
    matcher.add("GPE", ["[Uu](nited|\\.?) ?[Ss](tates|\\.?)"])
    output = matcher.pipe(doc_stream, return_matches=True)
    matches = [entry[1] for entry in output]
    assert matches == [[("GPE", 4, 6, (0, 0, 0))], [("GPE", 4, 5, (0, 0, 0))]]


def test_matcher_pipe_with_matches_and_context(nlp: Language) -> None:
    """It returns a stream of Doc objects, matches, and context as a tuple."""
    doc_stream = (
        (nlp.make_doc("test doc 1: United States"), "Country"),
        (nlp.make_doc("test doc 2: US"), "Country"),
    )
    matcher = RegexMatcher(nlp.vocab)
    matcher.add("GPE", ["[Uu](nited|\\.?) ?[Ss](tates|\\.?)"])
    output = matcher.pipe(doc_stream, return_matches=True, as_tuples=True)
    matches = [(entry[0][1], entry[1]) for entry in output]
    assert matches == [
        ([("GPE", 4, 6, (0, 0, 0))], "Country"),
        ([("GPE", 4, 5, (0, 0, 0))], "Country"),
    ]
