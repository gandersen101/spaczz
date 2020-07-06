"""Tests for the spaczzruler module."""
import os
import tempfile
from typing import Any, Dict, List

import pytest
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

from spaczz.exceptions import PatternTypeWarning
from spaczz.pipeline.spaczzruler import SpaczzRuler


@pytest.fixture
def doc(nlp: Language) -> Doc:
    """Doc for testing."""
    return nlp.make_doc(
        """Anderson, Grunt from 122 Fake Dr, Apt 55, Mawwah\n
            NJ in the USA was prescribed Zithroman."""
    )


@pytest.fixture
def patterns() -> List[Dict[str, Any]]:
    """Patterns for testing."""
    patterns = [
        {"label": "DRUG", "pattern": "Zithromax", "type": "fuzzy"},
        {"label": "GPE", "pattern": "Mahwah", "type": "fuzzy"},
        {
            "label": "NAME",
            "pattern": "Grant Andersen",
            "type": "fuzzy",
            "kwargs": {"fuzzy_func": "token_sort"},
        },
        {
            "label": "STREET",
            "pattern": "street_addresses",
            "type": "regex",
            "kwargs": {"predef": True},
        },
        {
            "label": "GPE",
            "pattern": r"(?i)[U](nited|\.?) ?[S](tates|\.?)",
            "type": "regex",
        },
    ]
    return patterns


def test_empty_default_ruler(nlp: Language) -> None:
    """It initialzes an empty ruler."""
    sr = SpaczzRuler(nlp)
    assert not sr.fuzzy_patterns
    assert not sr.regex_patterns


def test_ruler_with_changed_matcher_defaults(nlp: Language) -> None:
    """It intializes with changed defaults in the matchers."""
    sr = SpaczzRuler(nlp, spaczz_fuzzy_defaults={"ignore_case": False})
    assert sr.fuzzy_matcher.defaults == {"ignore_case": False}


def test_ruler_with_defaults_as_not_dict_raises_error(nlp: Language) -> None:
    """It raises a TypeError if defaults not dict."""
    with pytest.raises(TypeError):
        SpaczzRuler(nlp, spaczz_fuzzy_defaults="ignore_case")


def test_add_patterns(nlp: Language, patterns: List[Dict[str, Any]]) -> None:
    """It adds patterns correctly."""
    sr = SpaczzRuler(nlp, spaczz_patterns=patterns)
    assert len(sr) == 5


def test_add_patterns_raises_error_if_not_spaczz_pattern(nlp: Language,) -> None:
    """It raises a ValueError if patterns not correct format."""
    sr = SpaczzRuler(nlp)
    with pytest.raises(ValueError):
        sr.add_patterns([{"label": "GPE", "pattern": "Montana"}])


def test_add_patterns_raises_error_pattern_not_iter_of_dict(nlp: Language) -> None:
    """It raises a TypeError if pattern not iterable of dicts."""
    sr = SpaczzRuler(nlp)
    with pytest.raises(TypeError):
        sr.add_patterns({"label": "GPE", "pattern": "Montana"})


def test_add_patterns_warns_if_spaczz_type_unrecognized(nlp: Language,) -> None:
    """It raises a ValueError if patterns not correct format."""
    sr = SpaczzRuler(nlp)
    with pytest.warns(PatternTypeWarning):
        sr.add_patterns([{"label": "GPE", "pattern": "Montana", "type": "invalid"}])


def test_add_patterns_with_other_pipeline_components(
    patterns: List[Dict[str, Any]]
) -> None:
    """It disables other pipeline components when adding patterns."""
    nlp = spacy.blank("en")
    nlp.add_pipe(nlp.create_pipe("ner"))
    sr = SpaczzRuler(nlp)
    nlp.add_pipe(sr, first=True)
    nlp.get_pipe("spaczz_ruler").add_patterns(patterns)
    assert len(sr) == 5


def test_contains(nlp: Language, patterns: List[Dict[str, Any]]) -> None:
    """It returns True if label in ruler."""
    sr = SpaczzRuler(nlp, spaczz_patterns=patterns)
    assert "GPE" in sr


def test_labels(nlp: Language, patterns: List[Dict[str, Any]]) -> None:
    """It returns all unique labels."""
    sr = SpaczzRuler(nlp, spaczz_patterns=patterns)
    assert all([label in sr.labels for label in ["GPE", "STREET", "DRUG", "NAME"]])
    assert len(sr.labels) == 4


def test_patterns(nlp: Language, patterns: List[Dict[str, Any]]) -> None:
    """It returns list of all patterns."""
    sr = SpaczzRuler(nlp, spaczz_patterns=patterns)
    assert all([pattern in sr.patterns for pattern in patterns])


def test_calling_ruler(nlp: Language, patterns: List[Dict[str, Any]], doc: Doc) -> None:
    """It adds entities to doc."""
    sr = SpaczzRuler(nlp, spaczz_patterns=patterns)
    sr = sr(doc)
    assert len(doc.ents) == 5


def test_entities_that_would_overlap_keeps_longer_earlier_match(
    nlp: Language, patterns: List[Dict[str, Any]], doc: Doc
) -> None:
    """It matches the longest/earliest entities."""
    sr = SpaczzRuler(nlp, spaczz_patterns=patterns)
    sr.add_patterns([{"label": "TEST", "pattern": "Fake", "type": "fuzzy"}])
    doc = sr(doc)
    assert "FAKE" not in [ent.label_ for ent in doc.ents]


def test_calling_ruler_with_overwrite_ents(
    nlp: Language, patterns: List[Dict[str, Any]], doc: Doc
) -> None:
    """It overwrites existing entities."""
    sr = SpaczzRuler(nlp, spaczz_patterns=patterns, spaczz_overwrite_ents=True)
    doc.ents += (Span(doc, 2, 4, label="WRONG"),)
    doc = sr(doc)
    assert "WRONG" not in [ent.label_ for ent in doc.ents]


def test_calling_ruler_without_overwrite_will_keep_exisiting_ents(
    nlp: Language, patterns: List[Dict[str, Any]], doc: Doc
) -> None:
    """It keeps existing ents without overwrite_ents."""
    sr = SpaczzRuler(nlp, spaczz_patterns=patterns)
    doc.ents += (
        Span(doc, 2, 4, label="WRONG"),
        Span(doc, 15, 16, label="WRONG"),
    )
    doc = sr(doc)
    assert len([ent.label_ for ent in doc.ents if ent.label_ == "WRONG"]) == 2


def test_seeing_tokens_again(
    nlp: Language, patterns: List[Dict[str, Any]], doc: Doc
) -> None:
    """If ruler has already seen tokens, it ignores them."""
    sr = SpaczzRuler(nlp, spaczz_patterns=patterns)
    sr.add_patterns(
        [{"label": "ADDRESS", "pattern": "122 Fake St, Apt 54", "type": "fuzzy"}]
    )
    sr = sr(doc)
    assert "ADDRESS" in [ent.label_ for ent in doc.ents]


def test_spaczz_ruler_serialize_bytes(
    nlp: Language, patterns: List[Dict[str, Any]]
) -> None:
    """It serializes the ruler to bytes and reads from bytes correctly."""
    ruler = SpaczzRuler(nlp, spaczz_patterns=patterns)
    assert len(ruler) == len(patterns)
    assert len(ruler.labels) == 4
    ruler_bytes = ruler.to_bytes()
    new_ruler = SpaczzRuler(nlp)
    assert len(new_ruler) == 0
    assert len(new_ruler.labels) == 0
    new_ruler = new_ruler.from_bytes(ruler_bytes)
    assert len(new_ruler) == len(patterns)
    assert len(new_ruler.labels) == 4
    assert len(new_ruler.patterns) == len(ruler.patterns)
    for pattern in ruler.patterns:
        assert pattern in new_ruler.patterns
    assert sorted(new_ruler.labels) == sorted(ruler.labels)


def test_spaczz_ruler_to_from_disk(
    nlp: Language, patterns: List[Dict[str, Any]]
) -> None:
    """It writes the ruler to disk and reads it back correctly."""
    ruler = SpaczzRuler(nlp, spaczz_patterns=patterns, spaczz_overwrite_ents=True)
    assert len(ruler) == len(patterns)
    assert len(ruler.labels) == 4
    with tempfile.TemporaryDirectory() as tmpdir:
        ruler.to_disk(f"{tmpdir}/ruler")
        assert os.path.isdir(f"{tmpdir}/ruler")
        new_ruler = SpaczzRuler(nlp)
        new_ruler = new_ruler.from_disk(f"{tmpdir}/ruler")
    assert len(new_ruler) == len(patterns)
    assert len(new_ruler.labels) == 4
    assert len(new_ruler.patterns) == len(ruler.patterns)
    for pattern in ruler.patterns:
        assert pattern in new_ruler.patterns
    assert sorted(new_ruler.labels) == sorted(ruler.labels)
    assert new_ruler.overwrite is True


def test_spaczz_patterns_to_from_disk(
    nlp: Language, patterns: List[Dict[str, Any]]
) -> None:
    """It writes the patterns to disk and reads them back correctly."""
    ruler = SpaczzRuler(nlp, spaczz_patterns=patterns, spaczz_overwrite_ents=True)
    assert len(ruler) == len(patterns)
    assert len(ruler.labels) == 4
    with tempfile.TemporaryFile() as tmpfile:
        ruler.to_disk(f"{tmpfile}.jsonl")
        assert os.path.isfile(f"{tmpfile}.jsonl")
        new_ruler = SpaczzRuler(nlp)
        new_ruler = new_ruler.from_disk(f"{tmpfile}.jsonl")
    assert len(new_ruler) == len(patterns)
    assert len(new_ruler.labels) == 4
    assert len(new_ruler.patterns) == len(ruler.patterns)
    for pattern in ruler.patterns:
        assert pattern in new_ruler.patterns
    assert sorted(new_ruler.labels) == sorted(ruler.labels)
    assert new_ruler.overwrite is False
