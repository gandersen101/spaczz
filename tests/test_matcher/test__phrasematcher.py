"""Tests for __phrasematcher module."""
import pytest
from spacy.language import Language

from spaczz.matcher._phrasematcher import PhraseMatcher


def test__phrasematcher_is_abc(nlp: Language) -> None:
    """`PhraseMatcher` is an `ABC`."""
    with pytest.raises(TypeError):
        PhraseMatcher(vocab=nlp.vocab)  # type: ignore
