"""Package-wide test fixures."""
import pytest
import spacy
from spacy.language import Language


@pytest.fixture
def nlp() -> Language:
    """Empty spacy English language pipeline."""
    return spacy.blank("en")
