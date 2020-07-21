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
    assert "space" in config._span_trimmers


def test_empty_fuzzy_config() -> None:
    """Initializes it with empty attributes."""
    empty_fc = FuzzyConfig(empty=True)
    assert empty_fc._fuzzy_funcs == {}
    assert empty_fc._span_trimmers == {}


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


def test_get_end_trimmers(config: FuzzyConfig) -> None:
    """It returns a set of end trimming functions."""
    trimmers = config.get_trimmers("end", ["space"], end_trimmers=["punct"])
    assert trimmers == {config._span_trimmers["space"], config._span_trimmers["punct"]}


def test_get_trimmers_empty_end(config: FuzzyConfig) -> None:
    """It returns empty sets if no trimmers for end."""
    assert config.get_trimmers("end") == set()


def test_get_start_trimmers(config: FuzzyConfig) -> None:
    """It returns a set of start trimming functions."""
    trimmers = config.get_trimmers("start", ["space"], start_trimmers=["punct"])
    assert trimmers == {config._span_trimmers["space"], config._span_trimmers["punct"]}


def test_get_trimmers_empty_start(config: FuzzyConfig) -> None:
    """It returns empty sets if no trimmers for start."""
    assert config.get_trimmers("start") == set()


def test_get_trimmers_raises_error_with_unknown_name(config: FuzzyConfig) -> None:
    """It raises a ValueError if trimmer name does not exist in config."""
    with pytest.raises(ValueError):
        assert config.get_trimmers("start", ["unknown"])
