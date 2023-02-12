"""Tests for similaritysearcher module."""
import pytest
from spacy.language import Language

from spaczz.exceptions import MissingVectorsWarning
from spaczz.search import SimilaritySearcher


@pytest.fixture
def searcher(model: Language) -> SimilaritySearcher:
    """It returns a similarity searcher."""
    return SimilaritySearcher(vocab=model.vocab)


def test_searcher_w_empty_vector_vocab(nlp: Language) -> None:
    """Raises a MissingVectorsWarning."""
    with pytest.warns(MissingVectorsWarning):
        SimilaritySearcher(vocab=nlp.vocab)


def test_compare_works_with_defaults(
    searcher: SimilaritySearcher, model: Language
) -> None:
    """Checks compare is working as intended."""
    assert searcher.compare(model("I like apples."), model("I like grapes.")) > 0


def test_compare_returns_0_w_no_vector(
    searcher: SimilaritySearcher, model: Language
) -> None:
    """Checks compare returns 0 when vector does not exist for span/token."""
    with pytest.warns(UserWarning):
        assert searcher.compare(model("spaczz"), model("python")) == 0
