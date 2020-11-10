"""Package-wide test fixures."""
import pytest
import spacy
from spacy.language import Language


@pytest.fixture
def nlp() -> Language:
    """Empty spacy English language pipeline."""
    return spacy.blank("en")


@pytest.fixture
def model() -> Language:
    """Medium English Core spaCy model."""
    return spacy.load("en_core_web_md")
