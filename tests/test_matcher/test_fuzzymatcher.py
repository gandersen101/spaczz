"""Tests for the fuzzymatcher module."""
import pickle
import typing as ty

import pytest
from spacy.language import Language
from spacy.tokens import Doc
from spacy.tokens import Span

from spaczz.exceptions import KwargsWarning
from spaczz.matcher import FuzzyMatcher


def add_name_ent(
    matcher: FuzzyMatcher,
    doc: Doc,
    i: int,
    matches: ty.List[ty.Tuple[str, int, int, int, str]],
) -> None:
    """Callback on match function. Adds "NAME" entities to doc."""
    _match_id, start, end, _ratio, _pattern = matches[i]
    entity = Span(doc, start, end, label="NAME")
    doc.ents += (entity,)  # type: ignore


@pytest.fixture
def doc(nlp: Language) -> Doc:
    """Doc for testing."""
    return nlp.make_doc(
        "The Heffer, 'mooooo, I'm a cow.' There's also a chken named Stephen."
    )


@pytest.fixture
def matcher(nlp: Language) -> FuzzyMatcher:
    """Fuzzy matcher with patterns added."""
    animals = ["Heifer", "chicken"]
    sounds = ["mooo"]
    names = ["Steven"]
    matcher = FuzzyMatcher(nlp.vocab)
    matcher.add(
        "ANIMAL",
        [nlp.make_doc(animal) for animal in animals],
        kwargs=[{"ignore_case": False}, {}],
    )
    matcher.add("SOUND", [nlp.make_doc(sound) for sound in sounds])
    matcher.add("NAME", [nlp.make_doc(name) for name in names], on_match=add_name_ent)
    return matcher


def test_adding_patterns(matcher: FuzzyMatcher) -> None:
    """It adds the "ANIMAL" label and some patterns with kwargs to the matcher."""
    assert matcher.patterns == [
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
    matcher: FuzzyMatcher, nlp: Language
) -> None:
    """It will warn when more patterns are added than explicit kwargs."""
    with pytest.warns(KwargsWarning):
        matcher.add(
            "TEST",
            [nlp.make_doc("Test1"), nlp.make_doc("Test2")],
            [{"ignore_case": False}],
        )


def test_add_with_more_explicit_kwargs_than_patterns_warns(
    matcher: FuzzyMatcher, nlp: Language
) -> None:
    """It will warn when more explicit kwargs are added than patterns."""
    with pytest.warns(KwargsWarning):
        matcher.add(
            "TEST",
            [nlp.make_doc("Test1")],
            [{"ignore_case": False}, {"ignore_case": False}],
        )


def test_add_where_patterns_is_not_list_raises_error(matcher: FuzzyMatcher) -> None:
    """Trying to add non Doc objects as patterns raises a ValueError."""
    with pytest.raises(ValueError):
        matcher.add("TEST", "Test1")  # type: ignore


def test_add_where_patterns_are_not_doc_objects_raises_error(
    matcher: FuzzyMatcher,
) -> None:
    """Trying to add non Doc objects as patterns raises a ValueError."""
    with pytest.raises(ValueError):
        matcher.add("TEST", ["Test1"])  # type: ignore


def test_add_where_kwargs_are_not_dicts_raises_error(
    matcher: FuzzyMatcher, nlp: Language
) -> None:
    """Trying to add non Dict objects as kwargs raises a ValueError."""
    with pytest.raises(ValueError):
        matcher.add("TEST", [nlp.make_doc("Test1")], ["ignore_case"])  # type: ignore


def test_len_returns_count_of_labels_in_matcher(matcher: FuzzyMatcher) -> None:
    """It returns the sum of unique labels in the matcher."""
    assert len(matcher) == 3


def test_in_returns_bool_of_label_in_matcher(matcher: FuzzyMatcher) -> None:
    """It returns whether a label is in the matcher."""
    assert "ANIMAL" in matcher


def test_labels_returns_label_names(matcher: FuzzyMatcher) -> None:
    """It returns a tuple of all unique label names."""
    assert all(label in matcher.labels for label in ("SOUND", "ANIMAL", "NAME"))


def test_vocab_prop_returns_vocab(matcher: FuzzyMatcher, nlp: Language) -> None:
    """It returns the vocab it was initialized with."""
    assert matcher.vocab == nlp.vocab


def test_remove_label(matcher: FuzzyMatcher, nlp: Language) -> None:
    """It removes a label from the matcher."""
    matcher.add("TEST", [nlp.make_doc("test")])
    assert "TEST" in matcher
    matcher.remove("TEST")
    assert "TEST" not in matcher


def test_remove_label_raises_error_if_label_not_in_matcher(
    matcher: FuzzyMatcher,
) -> None:
    """It raises a ValueError if trying to remove a label not present."""
    with pytest.raises(ValueError):
        matcher.remove("TEST")


def test_matcher_returns_matches(matcher: FuzzyMatcher, doc: Doc) -> None:
    """Calling the matcher on a Doc object returns matches."""
    assert matcher(doc) == [
        ("ANIMAL", 1, 2, 83, "Heifer"),
        ("SOUND", 4, 5, 80, "mooo"),
        ("ANIMAL", 16, 17, 83, "chicken"),
        ("NAME", 18, 19, 77, "Steven"),
    ]


def test_matcher_returns_empty_list_if_no_matches(
    matcher: FuzzyMatcher, nlp: Language
) -> None:
    """Calling the matcher on a Doc object with no viable matches returns empty list."""
    temp_doc = nlp.make_doc("No matches here.")
    assert matcher(temp_doc) == []


def test_matcher_uses_on_match_callback(matcher: FuzzyMatcher, doc: Doc) -> None:
    """It utilizes callback on match functions passed when called on a Doc object."""
    matcher(doc)
    ent_text = [ent.text for ent in doc.ents]
    assert "Stephen" in ent_text


def test_pickling_matcher(matcher: FuzzyMatcher) -> None:
    """It pickles the matcher object."""
    bytestring = pickle.dumps(matcher)
    assert type(bytestring) == bytes


def test_unpickling_matcher(matcher: FuzzyMatcher, doc: Doc) -> None:
    """It unpickles the matcher object."""
    bytestring = pickle.dumps(matcher)
    matcher = pickle.loads(bytestring)
    assert matcher(doc) == [
        ("ANIMAL", 1, 2, 83, "Heifer"),
        ("SOUND", 4, 5, 80, "mooo"),
        ("ANIMAL", 16, 17, 83, "chicken"),
        ("NAME", 18, 19, 77, "Steven"),
    ]
