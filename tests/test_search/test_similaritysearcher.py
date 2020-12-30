"""Tests for similaritysearcher module."""
import pytest
from spacy.language import Language

from spaczz.search import SimilaritySearcher


@pytest.fixture
def searcher(model: Language) -> SimilaritySearcher:
    """It returns a default similarity searcher."""
    return SimilaritySearcher(vocab=model.vocab)


def test_compare_works_with_defaults(
    searcher: SimilaritySearcher, model: Language
) -> None:
    """Checks compare is working as intended."""
    assert searcher.compare(model("I like apples."), model("I like grapes.")) == 94


def test_compare_returns_0_w_no_vector(
    searcher: SimilaritySearcher, model: Language
) -> None:
    """Checks compare returns 0 when vector does not exist for span/token."""
    assert searcher.compare(model("spaczz"), model("python")) == 0
