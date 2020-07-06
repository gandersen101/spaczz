"""Tests for fuzzyconfig module."""
from fuzzywuzzy import fuzz
import pytest

from spaczz.exceptions import CaseConflictWarning, EmptyConfigError
from spaczz.fuzz.fuzzyconfig import FuzzyConfig


@pytest.fixture
def config() -> FuzzyConfig:
    """It returns a default fuzzy config."""
    return FuzzyConfig()


def test_empty_fuzzy_config() -> None:
    """Initializes it with empty attributes."""
    empty_fc = FuzzyConfig(empty=True)
    assert empty_fc._fuzzy_funcs == {}


def test_get_fuzzy_alg_raises_error_if__fuzzy_funcs_empty() -> None:
    """It raises an EmptyConfigError if called on empty config."""
    empty_fc = FuzzyConfig(empty=True)
    with pytest.raises(EmptyConfigError):
        empty_fc.get_fuzzy_func("simple")


def test__get_fuzzy_alg_returns_alg(config: FuzzyConfig) -> None:
    """It returns the expected fuzzy matching function."""
    func = config.get_fuzzy_func("simple")
    assert func == fuzz.ratio


def test__get_fuzzy_alg_raises_error_with_unknown_name(config: FuzzyConfig) -> None:
    """It raises a ValueError if fuzzy_func does not match a predefined key name."""
    with pytest.raises(ValueError):
        config.get_fuzzy_func("unkown")


def test__get_fuzzy_alg_warns_with_case_conflict(config: FuzzyConfig) -> None:
    """It warns-CaseConflictWarning if fuzzy func lower-cases but not ignore_case."""
    with pytest.warns(CaseConflictWarning):
        config.get_fuzzy_func("u_weighted", ignore_case=False)
