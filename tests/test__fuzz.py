"""Tests for _fuzz module."""
import pytest

from spaczz._fuzz import FuzzyFuncs


def test_fuzzyfuncs_raises_value_error_w_unkown_match_type() -> None:
    """The searcher with lower-cased text is working as intended."""
    with pytest.raises(ValueError):
        FuzzyFuncs(match_type="unknown")
