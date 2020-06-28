"""Tests for fuzzyconfig module."""
from fuzzywuzzy import fuzz
import pytest

from spaczz.fuzz.fuzzyconfig import FuzzyConfig

# Global Variables
fc = FuzzyConfig(empty=False)


def test_empty_fuzzy_config():
    """Initializes it with empty attributes."""
    empty_fc = FuzzyConfig(empty=True)
    assert empty_fc._fuzzy_funcs == {}
    assert empty_fc._span_trimmers == {}


def test__get_fuzzy_alg_returns_alg():
    """It returns the expected fuzzy matching function."""
    alg = fc.get_fuzzy_func("simple", ignore_case=True)
    assert alg == fuzz.ratio


def test__get_fuzzy_alg_raises_error_with_unknown_name():
    """It raises a ValueError if fuzzy_func does not match a predefined key name."""
    with pytest.raises(ValueError):
        fc.get_fuzzy_func("unkown", ignore_case=True)


def test__get_fuzzy_alg_warns_with_case_conflict():
    """It provides a UserWarning if the fuzzy matching function
    lower cases but ignore_case is set to False.
    """
    with pytest.warns(UserWarning):
        fc.get_fuzzy_func("u_weighted", ignore_case=False)


def test__populate_trimmers_both_and_start():
    """It returns set of start trimmer functions
    and set of their key names as a tuple.
    """
    start_trimmers = fc.get_trimmers("start", ["space"], start_trimmers=["stop"])
    assert start_trimmers == (
        {fc._span_trimmers["space"], fc._span_trimmers["stop"]},
        {"space", "stop"},
    )


def test__populate_trimmers_both_and_end():
    """It returns set of end trimmer functions and set of their key names as a tuple."""
    end_trimmers = fc.get_trimmers("end", ["space"], end_trimmers=["punct"])
    assert end_trimmers == (
        {fc._span_trimmers["space"], fc._span_trimmers["punct"]},
        {"space", "punct"},
    )


def test__populate_trimmers_raises_error_with_invalid_trimmers():
    """It raises a ValueError if an unknown trimmer name is used."""
    with pytest.raises(ValueError):
        fc.get_trimmers("end", ["space", "unknown"])
