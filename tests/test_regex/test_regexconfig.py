"""Tests for the regexconfig module."""
import pytest
import regex

from spaczz.regex._commonregex import _commonregex
from spaczz.regex.regexconfig import RegexConfig, RegexParseError


@pytest.fixture
def default_rc() -> RegexConfig:
    """It returns a default regex config."""
    return RegexConfig(empty=False)


def test_empty_regex_config() -> None:
    """Initializes it with empty attributes."""
    empty_rc = RegexConfig(empty=True)
    assert empty_rc._predefs == {}


def test__get_predef_returns_existing_regex(default_rc: RegexConfig) -> None:
    """It returns a predefined compiled regex pattern."""
    assert default_rc._get_predef("times") == _commonregex["times"]


def test__get_predef_raises_error_with_undefined_regex(default_rc: RegexConfig) -> None:
    """It raises a ValueError if predef is not actually predefined."""
    with pytest.raises(ValueError):
        default_rc._get_predef("unknown")


def test_parse_regex_with_predef(default_rc: RegexConfig) -> None:
    """It returns a predefined regex pattern."""
    assert default_rc.parse_regex("phones", predef=True) == _commonregex["phones"]


def test_parse_regex_with_new_regex(default_rc: RegexConfig) -> None:
    """It turns the string into a regex pattern."""
    assert default_rc.parse_regex("(?i)Test",) == regex.compile("(?i)Test")


def test_incorrect_type_for_regex_compile_raises_error(default_rc: RegexConfig) -> None:
    """Using an invalid type raises a RegexParseError."""
    with pytest.raises(RegexParseError):
        default_rc.parse_regex(1)
