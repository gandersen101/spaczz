"""Tests for phrasesearcher module."""
import pytest
from spacy.language import Language

from spaczz._search import PhraseSearcher


def test_phrasesearcher_is_abc(nlp: Language) -> None:
    """`PhraseSearcher` is an `ABC`."""
    with pytest.raises(TypeError):
        PhraseSearcher(vocab=nlp.vocab)  # type: ignore
