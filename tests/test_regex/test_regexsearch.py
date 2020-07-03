"""Tests for the regexsearcher module."""
import pytest
import regex
from spacy.language import Language

from spaczz.regex._commonregex import _commonregex
from spaczz.regex.regexconfig import RegexConfig
from spaczz.regex.regexsearcher import RegexSearcher


@pytest.fixture
def searcher() -> RegexSearcher:
    """It returns a default regex searcher."""
    return RegexSearcher()


def test_regexsearcher_config_contains_predefined_defaults(
    searcher: RegexSearcher,
) -> None:
    """It's config contains predefined defaults."""
    assert searcher._config._predefs == _commonregex


def test_regexsearcher_uses_passed_config() -> None:
    """It uses the config passed to it."""
    config = RegexConfig()
    config._predefs["test"] = regex.compile("test")
    searcher = RegexSearcher(config=config)
    assert "test" in searcher._config._predefs


def test_regexsearcher_raises_error_if_config_is_not_regexconfig() -> None:
    """It raises a TypeError if config is not recognized string or RegexConfig."""
    with pytest.raises(TypeError):
        RegexSearcher(config="Will cause error")


def test_regex_search_has_empty_config_if_empty_passed() -> None:
    """Its config is empty."""
    searcher = RegexSearcher(config="empty")
    assert searcher._config._predefs == {}


def test_multi_match(searcher: RegexSearcher, nlp: Language) -> None:
    """It produces matches."""
    doc = nlp("My phone number is (555) 555-5555, not (554) 554-5554.")
    matches = searcher.multi_match(doc, "phones", predef=True)
    assert matches == [(4, 10), (12, 18)]


def test_multi_match_will_expand_on_partial_match_if_partials(
    searcher: RegexSearcher, nlp: Language
) -> None:
    """It extends partial matches to span boundaries."""
    doc = nlp(
        "We want to identify 'USA' even though only first two letters will matched."
    )
    matches = searcher.multi_match(doc, "[Uu](nited|\\.?) ?[Ss](tates|\\.?)")
    assert matches == [(5, 6)]


def test_multi_match_will_not_expand_if_not_partials(
    searcher: RegexSearcher, nlp: Language
) -> None:
    """It will not extend partial matches to span boundaries if not partial."""
    doc = nlp(
        "We want to identify 'USA' even though only first two letters will matched."
    )
    matches = searcher.multi_match(
        doc, "[Uu](nited|\\.?) ?[Ss](tates|\\.?)", partial=False
    )
    assert matches == []


def test_multi_match_will_not_match_if_regex_starts_ends_with_space(
    searcher: RegexSearcher, nlp: Language
) -> None:
    """Regexes that match on spaces will not produce match."""
    doc = nlp(
        "We want to identify US but will fail because regex includes whitespaces."
    )
    matches = searcher.multi_match(doc, "\\s[Uu](nited|\\.?) ?[Ss](tates|\\.?)\\s")
    assert matches == []


def test_multi_match_raises_error_if_regex_str_not_str(
    searcher: RegexSearcher, nlp: Language
) -> None:
    """It raises a type error if regex_str is not a string."""
    doc = nlp("My phone number is (555) 555-5555.")
    with pytest.raises(TypeError):
        searcher.multi_match(doc, 1, predef=True)
