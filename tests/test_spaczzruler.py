"""Tests for the spaczzruler module."""
import pytest
import spacy
from spacy.tokens import Span

from spaczz.pipeline.spaczzruler import SpaczzRuler


# Global Variables
nlp = spacy.blank("en")
text = "The patient from 122 Fake Dr, Mawwah, NJ in the USA was prescribed Zithroman."
sr = SpaczzRuler(nlp)
patterns = [
    {"label": "DRUG", "pattern": "Zithromax", "type": "fuzzy"},
    {"label": "GPE", "pattern": "Mahwah", "type": "fuzzy"},
    {
        "label": "GPE",
        "pattern": "NJ",
        "type": "fuzzy",
        "kwargs": {"ignore_case": False},
    },
    {
        "label": "STREET",
        "pattern": "street_addresses",
        "type": "regex",
        "kwargs": {"predef": True},
    },
    {"label": "GPE", "pattern": "[Uu](nited|\\.?) ?[Ss](tates|\\.?)", "type": "regex"},
]


def test_empty_default_ruler() -> None:
    """It initialzes an empty ruler."""
    temp_sr = SpaczzRuler(nlp)
    assert not temp_sr.fuzzy_patterns
    assert not temp_sr.regex_patterns


def test_ruler_with_changed_matcher_defaults() -> None:
    """It intializes with changed defaults in the matchers."""
    temp_sr = SpaczzRuler(nlp, spaczz_fuzzy_defaults={"ignore_case": False})
    assert temp_sr.fuzzy_matcher.defaults == {"ignore_case": False}


def test_ruler_with_defaults_as_not_dict_raises_error() -> None:
    """It raises a TypeError if defaults not dict."""
    with pytest.raises(TypeError):
        SpaczzRuler(nlp, spaczz_fuzzy_defaults="ignore_case")


def test_add_patterns() -> None:
    """It adds patterns correctly."""
    sr.add_patterns(patterns)
    assert len(sr.patterns) == 5


def test_add_patterns_raises_error_if_not_spaczz_pattern() -> None:
    """It raises a ValueError if patterns not correct format."""
    with pytest.raises(ValueError):
        sr.add_patterns([{"label": "GPE", "pattern": "Montana"}])


def test_add_patterns_raises_error_pattern_not_iter_of_dict() -> None:
    """IT raises a TypeError if pattern not iterable of dicts."""
    with pytest.raises(TypeError):
        sr.add_patterns({"label": "GPE", "pattern": "Montana"})


def test_add_patterns_with_other_pipeline_components() -> None:
    """It disables other pipeline components when adding patterns."""
    temp_nlp = spacy.blank("en")
    temp_nlp.add_pipe(nlp.create_pipe("ner"))
    temp_sr = SpaczzRuler(temp_nlp)
    temp_nlp.add_pipe(temp_sr, first=True)
    temp_nlp.get_pipe("spaczz_ruler").add_patterns(patterns)
    assert len(temp_sr.patterns) == 5


def test_ruler_initializes_with_patterns() -> None:
    """It add patterns when initialized with them."""
    temp_sr = SpaczzRuler(nlp, spaczz_patterns=patterns)
    assert len(temp_sr) == 5


def test_contains() -> None:
    """It returns True if label in ruler."""
    assert "GPE" in sr


def test_labels() -> None:
    """It returns all unique labels."""
    assert all([label in sr.labels for label in ["GPE", "STREET", "DRUG"]])
    assert len(sr.labels) == 3


def test_patterns() -> None:
    """It returns list of all patterns."""
    assert all([pattern in sr.patterns for pattern in patterns])


def test_calling_ruler() -> None:
    """It adds entities to doc."""
    doc = nlp.make_doc(text)
    doc = sr(doc)
    assert len(doc.ents) == 5


def test_entities_that_would_overlap_keeps_longer_earlier_match() -> None:
    """It matches the longest/earliest entities."""
    doc = nlp.make_doc(text)
    temp_sr = SpaczzRuler(nlp, spaczz_patterns=patterns)
    temp_sr.add_patterns([{"label": "TEST", "pattern": "FAKE", "type": "FUZZY"}])
    doc = sr(doc)
    assert "FAKE" not in [ent.label_ for ent in doc.ents]


def test_calling_ruler_with_overwrite_ents() -> None:
    """It overwrites existing entities."""
    temp_sr = SpaczzRuler(nlp, spaczz_overwrite_ents=True)
    temp_sr.add_patterns(patterns)
    doc = nlp.make_doc(text)
    doc.ents += (Span(doc, 2, 4, label="WRONG"),)
    doc = temp_sr(doc)
    assert "WRONG" not in [ent.label_ for ent in doc.ents]


def test_calling_ruler_without_overwrite_will_keep_exisiting_ents() -> None:
    """It keeps existing ents without overwrite_ents."""
    doc = nlp.make_doc(text)
    doc.ents += (Span(doc, 2, 4, label="WRONG"), Span(doc, 15, 16, label="WRONG"))
    doc = sr(doc)
    assert len([ent.label_ for ent in doc.ents if ent.label_ == "WRONG"]) == 2
