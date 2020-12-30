"""Tests for __phrasesearcher module."""
import pytest
from spacy.language import Language

from spaczz.search import _PhraseSearcher


@pytest.fixture
def searcher(nlp: Language) -> _PhraseSearcher:
    """It returns a phrase searcher."""
    return _PhraseSearcher(vocab=nlp.vocab)


def test_compare_works_with_caseless_matches(
    searcher: _PhraseSearcher, nlp: Language
) -> None:
    """Checks compare is working as intended - case insensitive."""
    assert searcher.compare(nlp("spaCy"), nlp("spacy")) == 100


def test_compare_works_with_cased_matches(
    searcher: _PhraseSearcher, nlp: Language
) -> None:
    """Checks compare is working as intended - case sensitive."""
    assert searcher.compare(nlp("spaCy"), nlp("spacy"), ignore_case=False) == 0
