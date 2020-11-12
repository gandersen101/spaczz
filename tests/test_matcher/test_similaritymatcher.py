"""Tests for the similaritymatcher module."""
from typing import List, Tuple

import pytest
from spacy.language import Language
from spacy.tokens import Doc, Span

from spaczz.exceptions import KwargsWarning, MissingVectorsWarning
from spaczz.matcher.similaritymatcher import SimilarityMatcher


def add_ent(
    matcher: SimilarityMatcher, doc: Doc, i: int, matches: List[Tuple[str, int, int]]
) -> None:
    """Callback on match function. Adds "INSTRUMENT" entities to doc."""
    _match_id, start, end, _ratio = matches[i]
    entity = Span(doc, start, end, label="INSTRUMENT")
    doc.ents += (entity,)


@pytest.fixture
def doc(model: Language) -> Doc:
    """Doc for testing."""
    return model(
        """John Frusciante was the longtime guitarist for the band\n
        The Red Hot Chili Peppers."""
    )


@pytest.fixture
def matcher(model: Language) -> SimilarityMatcher:
    """Similarity matcher with patterns added."""
    instruments = ["guitar"]
    peppers = ["jalapenos"]
    matcher = SimilarityMatcher(model.vocab)
    matcher.add(
        "INSTRUMENT",
        [model(i) for i in instruments],
        kwargs=[{"min_r2": 70}],
        on_match=add_ent,
    )
    matcher.add("PEPPER", [model(pepper) for pepper in peppers])
    return matcher


def test_adding_patterns(matcher: SimilarityMatcher) -> None:
    """It adds the labels and patterns with kwargs to the matcher."""
    assert matcher.patterns == [
        {
            "label": "INSTRUMENT",
            "pattern": "guitar",
            "type": "similarity",
            "kwargs": {"min_r2": 70},
        },
        {"label": "PEPPER", "pattern": "jalapenos", "type": "similarity"},
    ]


def test_add_with_more_patterns_than_explicit_kwargs_warns(
    matcher: SimilarityMatcher, model: Language
) -> None:
    """It will warn when more patterns are added than explicit kwargs."""
    with pytest.warns(KwargsWarning):
        matcher.add(
            "TEST", [model("Test1"), model.make_doc("Test2")], [{"min_r1": 30}],
        )


def test_add_with_more_explicit_kwargs_than_patterns_warns(
    matcher: SimilarityMatcher, model: Language
) -> None:
    """It will warn when more explicit kwargs are added than patterns."""
    with pytest.warns(KwargsWarning):
        matcher.add(
            "TEST", [model("Test1")], [{"min_r1": 30}, {"min_r1": 30}],
        )


def test_add_without_doc_objects_raises_error(matcher: SimilarityMatcher,) -> None:
    """Trying to add non Doc objects as patterns raises a TypeError."""
    with pytest.raises(TypeError):
        matcher.add("TEST", ["Test1"])


def test_add_where_kwargs_are_not_dicts_raises_error(
    matcher: SimilarityMatcher, model: Language
) -> None:
    """Trying to add non Dict objects as kwargs raises a TypeError."""
    with pytest.raises(TypeError):
        matcher.add("TEST", [model("Test1")], ["min_r1"])


def test_len_returns_count_of_labels_in_matcher(matcher: SimilarityMatcher) -> None:
    """It returns the sum of unique labels in the matcher."""
    assert len(matcher) == 2


def test_in_returns_bool_of_label_in_matcher(matcher: SimilarityMatcher) -> None:
    """It returns whether a label is in the matcher."""
    assert "INSTRUMENT" in matcher


def test_labels_returns_label_names(matcher: SimilarityMatcher) -> None:
    """It returns a tuple of all unique label names."""
    assert all(label in matcher.labels for label in ("INSTRUMENT", "PEPPER"))


def test_remove_label(matcher: SimilarityMatcher, model: Language) -> None:
    """It removes a label from the matcher."""
    matcher.add("TEST", [model("test")])
    assert "TEST" in matcher
    matcher.remove("TEST")
    assert "TEST" not in matcher


def test_remove_label_raises_error_if_label_not_in_matcher(
    matcher: SimilarityMatcher,
) -> None:
    """It raises a ValueError if trying to remove a label not present."""
    with pytest.raises(ValueError):
        matcher.remove("TEST")


def test_matcher_returns_matches(matcher: SimilarityMatcher, doc: Doc) -> None:
    """Calling the matcher on a Doc object returns matches."""
    assert matcher(doc) == [("INSTRUMENT", 5, 6, 71), ("PEPPER", 14, 15, 100)]


def test_matcher_returns_empty_list_if_no_matches(
    matcher: SimilarityMatcher, model: Language,
) -> None:
    """Calling the matcher on a Doc object with no viable matches returns empty list."""
    temp_doc = model("No matches here.")
    assert matcher(temp_doc) == []


def test_matcher_uses_on_match_callback(matcher: SimilarityMatcher, doc: Doc,) -> None:
    """It utilizes callback on match functions passed when called on a Doc object."""
    matcher(doc)
    ent_text = [ent.text for ent in doc.ents]
    assert "guitarist" in ent_text


def test_matcher_w_vocab_wo_vectors_raises_warning(nlp: Language) -> None:
    """It raises a warning if matcher initialized with Vocab wo vectors."""
    with pytest.warns(MissingVectorsWarning):
        SimilarityMatcher(nlp.vocab)


def test_matcher_pipe(model: Language) -> None:
    """It returns a stream of Doc objects."""
    doc_stream = (
        model("test doc 1: grape"),
        model("test doc 2: kiwi"),
    )
    matcher = SimilarityMatcher(model.vocab)
    output = matcher.pipe(doc_stream)
    assert list(output) == list(doc_stream)


def test_matcher_pipe_with_context(model: Language) -> None:
    """It returns a stream of Doc objects as tuples with context."""
    doc_stream = (
        (model("test doc 1: grape"), "Juicy"),
        (model("test doc 2: kiwi"), "Juicy"),
    )
    matcher = SimilarityMatcher(model.vocab)
    output = matcher.pipe(doc_stream, as_tuples=True)
    assert list(output) == list(doc_stream)


def test_matcher_pipe_with_matches(model: Language) -> None:
    """It returns a stream of Doc objects and matches as tuples."""
    doc_stream = (model("test doc1: grape"), model("test doc2: kiwi"))
    matcher = SimilarityMatcher(model.vocab, min_r2=65)
    matcher.add("FRUIT", [model("fruit")])
    output = matcher.pipe(doc_stream, return_matches=True)
    matches = [entry[1] for entry in output]
    assert matches == [[("FRUIT", 3, 4, 72)], [("FRUIT", 3, 4, 68)]]


def test_matcher_pipe_with_matches_and_context(model: Language) -> None:
    """It returns a stream of Doc objects and matches and context as tuples."""
    doc_stream = (
        (model("test doc1: grape"), "Juicy"),
        (model("test doc2: kiwi"), "Juicy"),
    )
    matcher = SimilarityMatcher(model.vocab, min_r2=65)
    matcher.add("FRUIT", [model("fruit")])
    output = matcher.pipe(doc_stream, return_matches=True, as_tuples=True)
    matches = [(entry[0][1], entry[1]) for entry in output]
    assert matches == [
        ([("FRUIT", 3, 4, 72)], "Juicy"),
        ([("FRUIT", 3, 4, 68)], "Juicy"),
    ]
