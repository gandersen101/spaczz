"""Tests for the regexsearcher module."""
from typing import Dict, Pattern

import pytest
import regex as re
from spacy.language import Language

from spaczz._commonregex import get_common_regex
from spaczz.exceptions import RegexParseError
from spaczz.search import RegexSearcher


@pytest.fixture
def searcher(nlp: Language) -> RegexSearcher:
    """It returns a default regex searcher."""
    return RegexSearcher(vocab=nlp.vocab)


@pytest.fixture(scope="module")
def predefs() -> Dict[str, Pattern]:
    """It returns predefined common regexes."""
    return get_common_regex()


def test_searcher_contains_predef_defaults(
    searcher: RegexSearcher, predefs: Dict[str, Pattern]
) -> None:
    """It's config contains predefined defaults."""
    assert searcher.predefs == predefs


def test_searcher_uses_passed_predefs(nlp: Language) -> None:
    """It uses the predefs passed to it."""
    predefs = {"test": re.compile("test")}
    searcher = RegexSearcher(vocab=nlp.vocab, predefs=predefs)
    assert "test" in searcher.predefs


def test_searcher_raises_error_if_predefs_is_not_valid(nlp: Language) -> None:
    """It raises a ValueError if predefs is not recognized string or valid dict."""
    with pytest.raises(ValueError):
        RegexSearcher(vocab=nlp.vocab, predefs="Will cause error")


def test_searcher_has_empty_predefs_if_empty_passed(nlp: Language) -> None:
    """Its predefs are empty."""
    searcher = RegexSearcher(vocab=nlp.vocab, predefs="empty")
    assert searcher.predefs == {}


def test_match(searcher: RegexSearcher, nlp: Language) -> None:
    """It produces matches."""
    doc = nlp("My phone number is (555) 555-5555, not (554) 554-5554.")
    matches = searcher.match(doc, "phones", predef=True)
    assert matches == [(4, 10, 100), (12, 18, 100)]


def test_match_will_expand_on_partial_match_if_partials(
    searcher: RegexSearcher, nlp: Language
) -> None:
    """It extends partial matches to span boundaries."""
    doc = nlp(
        "We want to identify 'USA' even though only first two letters will matched."
    )
    matches = searcher.match(doc, "[Uu](nited|\\.?) ?[Ss](tates|\\.?)")
    assert matches == [(5, 6, 100)]


def test_match_on_german_combination_words(
    searcher: RegexSearcher, nlp: Language
) -> None:
    """It extends partial matches to span boundaries."""
    doc = nlp(
        "We want to identify a geman word combination Aussagekraft or Kraftfahrzeug"
    )
    matches = searcher.match(doc, "(kraft|Kraft)")
    assert matches == [(8, 9, 100), (10, 11, 100)]


def test_match_will_not_expand_if_not_partials(
    searcher: RegexSearcher, nlp: Language
) -> None:
    """It will not extend partial matches to span boundaries if not partial."""
    doc = nlp(
        "We want to identify 'USA' even though only first two letters will matched."
    )
    matches = searcher.match(doc, "[Uu](nited|\\.?) ?[Ss](tates|\\.?)", partial=False)
    assert matches == []


def test_match_will_not_match_if_regex_starts_ends_with_space(
    searcher: RegexSearcher, nlp: Language
) -> None:
    """Regexes that match on spaces will not produce match."""
    doc = nlp(
        "We want to identify US but will fail because regex includes whitespaces."
    )
    matches = searcher.match(doc, "\\s[Uu](nited|\\.?) ?[Ss](tates|\\.?)\\s")
    assert matches == []


def test_match_raises_error_if_regex_str_not_str(
    searcher: RegexSearcher, nlp: Language
) -> None:
    """It raises a type error if regex_str is not a string."""
    doc = nlp("My phone number is (555) 555-5555.")
    with pytest.raises(TypeError):
        searcher.match(doc, 1, predef=True)  # type: ignore


def test_match_raises_error_if_doc_not_doc(
    searcher: RegexSearcher, nlp: Language
) -> None:
    """It raises a type error if regex_str is not a string."""
    doc = "not a doc"
    with pytest.raises(TypeError):
        searcher.match(doc, "test")  # type: ignore


def test_get_predef_returns_existing_regex(
    searcher: RegexSearcher, predefs: Dict[str, Pattern]
) -> None:
    """It returns a predefined compiled regex pattern."""
    assert searcher.get_predef("times") == predefs["times"]


def test_get_predef_raises_error_with_undefined_regex(searcher: RegexSearcher) -> None:
    """It raises a ValueError if predef is not actually predefined."""
    with pytest.raises(ValueError):
        searcher.get_predef("unknown")


def test_parse_regex_with_predef(
    searcher: RegexSearcher, predefs: Dict[str, Pattern]
) -> None:
    """It returns a predefined regex pattern."""
    assert searcher.parse_regex("phones", predef=True) == predefs["phones"]


def test_parse_regex_with_new_regex(searcher: RegexSearcher) -> None:
    """It turns the string into a regex pattern."""
    assert searcher.parse_regex(
        "(?i)Test",
    ) == re.compile("(?i)Test")


def test_invalid_regexfor_regex_compile_raises_error(searcher: RegexSearcher) -> None:
    """Using an invalid type raises a RegexParseError."""
    with pytest.raises(RegexParseError):
        searcher.parse_regex("[")
