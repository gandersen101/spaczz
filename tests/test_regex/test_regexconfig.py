"""Tests for the regexconfig module."""
import pytest
import regex

from spaczz.regex._commonregex import _commonregex
from spaczz.regex.regexconfig import RegexConfig, RegexParseError


@pytest.fixture
def config() -> RegexConfig:
    """It returns a default regex config."""
    return RegexConfig(empty=False)


def test_empty_regex_config() -> None:
    """Initializes it with empty attributes."""
    empty_rc = RegexConfig(empty=True)
    assert empty_rc._predefs == {}


def test_get_predef_returns_existing_regex(config: RegexConfig) -> None:
    """It returns a predefined compiled regex pattern."""
    assert config.get_predef("times") == _commonregex["times"]


def test_get_predef_raises_error_with_undefined_regex(config: RegexConfig) -> None:
    """It raises a ValueError if predef is not actually predefined."""
    with pytest.raises(ValueError):
        config.get_predef("unknown")


def test_parse_regex_with_predef(config: RegexConfig) -> None:
    """It returns a predefined regex pattern."""
    assert config.parse_regex("phones", predef=True) == _commonregex["phones"]


def test_parse_regex_with_new_regex(config: RegexConfig) -> None:
    """It turns the string into a regex pattern."""
    assert config.parse_regex("(?i)Test",) == regex.compile("(?i)Test")


def test_invalid_regexfor_regex_compile_raises_error(config: RegexConfig) -> None:
    """Using an invalid type raises a RegexParseError."""
    with pytest.raises(RegexParseError):
        config.parse_regex("[")
