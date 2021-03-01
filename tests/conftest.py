"""Package-wide test fixures."""
from pathlib import Path

import pytest
import spacy
from spacy.language import Language


@pytest.fixture(scope="session")
def fixtures() -> Path:
    """Path to test fixtures."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def model() -> Language:
    """Medium English Core spaCy model."""
    return spacy.load("en_core_web_md")


@pytest.fixture(scope="session")
def nlp() -> Language:
    """Empty spaCy English language pipeline."""
    return spacy.blank("en")
