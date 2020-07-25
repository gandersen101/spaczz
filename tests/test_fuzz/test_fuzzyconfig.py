"""Tests for fuzzyconfig module."""
import pytest
from rapidfuzz import fuzz

from spaczz.exceptions import EmptyConfigError
from spaczz.fuzz.fuzzyconfig import FuzzyConfig


@pytest.fixture
def config() -> FuzzyConfig:
    """It returns a default fuzzy config."""
    return FuzzyConfig()


def test_config_has_matchers_andtrimmers(config: FuzzyConfig) -> None:
    """Matching/Trimming funcs exist in default config."""
    assert "simple" in config._fuzzy_funcs


def test_empty_fuzzy_config() -> None:
    """Initializes it with empty attributes."""
    empty_fc = FuzzyConfig(empty=True)
    assert empty_fc._fuzzy_funcs == {}


def test_get_fuzzy_alg_raises_error_if__fuzzy_funcs_empty() -> None:
    """It raises an EmptyConfigError if called on empty config."""
    empty_fc = FuzzyConfig(empty=True)
    with pytest.raises(EmptyConfigError):
        empty_fc.get_fuzzy_func("simple")


def test_get_fuzzy_alg_returns_alg(config: FuzzyConfig) -> None:
    """It returns the expected fuzzy matching function."""
    func = config.get_fuzzy_func("simple")
    assert func == fuzz.ratio


def test_get_fuzzy_alg_raises_error_with_unknown_name(config: FuzzyConfig) -> None:
    """It raises a ValueError if fuzzy_func does not match a predefined key name."""
    with pytest.raises(ValueError):
        config.get_fuzzy_func("unkown")
