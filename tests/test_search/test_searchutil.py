"""Tests for search.searchutils module."""
import pytest
import regex as re

from spaczz.exceptions import RegexParseError
from spaczz.registry import get_re_pattern
from spaczz.search.searchutil import filter_overlapping_matches
from spaczz.search.searchutil import parse_regex


def test_filter_overlapping_matches() -> None:
    """It only returns the first match if more than one encompass the same tokens."""
    matches = [(1, 2, 80), (1, 3, 70)]
    assert filter_overlapping_matches(matches) == [(1, 2, 80)]


def test_parse_regex_with_predef() -> None:
    """It returns a predefined regex pattern."""
    assert parse_regex("phones", predef=True) == get_re_pattern("phones")


def test_parse_regex_with_new_regex() -> None:
    """It turns the string into a regex pattern."""
    assert parse_regex(r"(?i)Test") == re.compile(r"(?i)Test")


def test_parse_regex_w_invalid_regex_raises_error() -> None:
    """Using an invalid type raises a RegexParseError."""
    with pytest.raises(RegexParseError):
        parse_regex(r"[")
