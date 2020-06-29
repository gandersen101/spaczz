"""Tests for the regexconfig module."""
import re

import pytest

from spaczz.regex._commonregex import _commonregex
from spaczz.regex.regexconfig import RegexConfig, RegexParseError

# Global Variables
rc = RegexConfig(empty=False)


def test_empty_regex_config() -> None:
    """Initializes it with empty attributes."""
    empty_rc = RegexConfig(empty=True)
    assert empty_rc._flags == {}
    assert empty_rc._predefs == {}


def test__get_predef_returns_existing_regex() -> None:
    """It returns a predefined compiled regex pattern."""
    assert rc._get_predef("times") == _commonregex["times"]


def test__get_predef_raises_error_with_undefined_regex() -> None:
    """It raises a ValueError if predef is not actually predefined."""
    with pytest.raises(ValueError):
        rc._get_predef("unknown") is None


def test__get_flags_returns_RegexFlag() -> None:
    """It returns RegexFlag objects based on parameters."""
    flags = rc._get_flags(ignore_case=True, use_ascii=True)
    assert all(flag in flags for flag in [re.IGNORECASE, re.ASCII])


def test__get_flags_ignores_false_kwargs() -> None:
    """It ignores flag parameters with arguments of False."""
    flags = rc._get_flags(ignore_case=True, use_ascii=True, verbose=False)
    assert all(flag in flags for flag in [re.IGNORECASE, re.ASCII])


def test__get_flags_raises_error_if_flag_undefined() -> None:
    """It raises a ValueError if the flag is not predefined."""
    with pytest.raises(ValueError):
        rc._get_flags(undefined=True)


def test__get_flags_raises_error_if_kwarg_non_boolean() -> None:
    """It raises a TypeErorr if the flag parameter argument is not boolean."""
    with pytest.raises(TypeError):
        rc._get_flags(ignore_case="True")


def test_parse_regex_with_predef() -> None:
    """It returns a predefined regex pattern."""
    assert rc.parse_regex("phones", predef=True) == _commonregex["phones"]


def test_parse_regex_with_new_regex() -> None:
    """It turns the string into a regex pattern."""
    assert rc.parse_regex(
        "Lord Windgrace", ignore_case=True, use_ascii=True
    ) == re.compile("Lord Windgrace", re.IGNORECASE | re.ASCII)


def test_using_precompiled_regex_raises_error_when_adding_flags() -> None:
    """Using a precompiled regex then adding flags raises a RegexParseError."""
    with pytest.raises(RegexParseError):
        rc.parse_regex(re.compile("Lord Windgrace"), ignore_case=True)


def test_incorrect_type_for_regex_compile_raises_error() -> None:
    """Using an invalid type raises a RegexParseError."""
    with pytest.raises(RegexParseError):
        rc.parse_regex(1)
