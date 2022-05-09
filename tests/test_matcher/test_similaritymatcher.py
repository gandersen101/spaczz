"""Tests for the similaritymatcher module."""
import pytest
from spacy.language import Language
from spacy.tokens import Doc

from spaczz.exceptions import MissingVectorsWarning
from spaczz.matcher.similaritymatcher import SimilarityMatcher


@pytest.fixture
def doc(model: Language) -> Doc:
    """Doc for testing."""
    return model(
        "John Frusciante was the longtime guitarist for the "
        "band The Red Hot Chili Peppers."
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


def test_matcher_returns_matches(matcher: SimilarityMatcher, doc: Doc) -> None:
    """Calling the matcher on a Doc object returns matches."""
    results = matcher(doc)
    assert results


def test_matcher_w_vocab_wo_vectors_raises_warning(nlp: Language) -> None:
    """It raises a warning if matcher initialized with Vocab wo vectors."""
    with pytest.warns(MissingVectorsWarning):
        SimilarityMatcher(nlp.vocab)
