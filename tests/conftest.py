"""Package-wide test fixures."""
from pathlib import Path
import typing as ty

import pytest
import spacy
from spacy.language import Language


@pytest.fixture(scope="session")
def fixtures() -> ty.Generator[Path, None, None]:
    """Path to test fixtures."""
    yield Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def model() -> ty.Generator[Language, None, None]:
    """Medium English Core spaCy model."""
    yield spacy.load("en_core_web_md")


@pytest.fixture(scope="session")
def nlp() -> ty.Generator[Language, None, None]:
    """Empty spaCy English language pipeline."""
    yield spacy.blank("en")
