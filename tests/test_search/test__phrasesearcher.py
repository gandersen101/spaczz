"""Tests for __phrasesearcher module."""
import pytest
from spacy.language import Language

from spaczz.search import _PhraseSearcher


def test__phrasesearcher_is_abc(nlp: Language) -> None:
    """`_PhraseSearcher` is an `ABC`."""
    with pytest.raises(TypeError):
        _PhraseSearcher(vocab=nlp.vocab)  # type: ignore
