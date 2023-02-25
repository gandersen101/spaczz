"""Tests for __phrasematcher module."""
import pytest
from spacy.language import Language

from spaczz.matcher import _PhraseMatcher


def test__phrasematcher_is_abc(nlp: Language) -> None:
    """`_PhraseMatcher` is an `ABC`."""
    with pytest.raises(TypeError):
        _PhraseMatcher(vocab=nlp.vocab)  # type: ignore
