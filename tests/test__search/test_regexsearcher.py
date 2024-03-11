"""Tests for the regexsearcher module."""
import pytest
from spacy.language import Language

from spaczz._search import RegexSearcher


@pytest.fixture
def searcher(nlp: Language) -> RegexSearcher:
    """It returns a default regex searcher."""
    return RegexSearcher(vocab=nlp.vocab)


def test_match(searcher: RegexSearcher, nlp: Language) -> None:
    """It produces matches."""
    doc = nlp("My phone number is (555) 555-5555, not (554) 554-5554.")
    matches = searcher.match(doc, "phones", predef=True)
    assert matches == [
        (4, 10, 100),
        (12, 18, 100),
    ]


def test_match_w_fuzzy_regex(searcher: RegexSearcher, nlp: Language) -> None:
    """It produces a fuzzy match."""
    doc = nlp("I live in the US.")
    re_pattern = r"(USA){d<=1}"
    matches = searcher.match(doc, re_pattern, ignore_case=False)
    assert matches == [(4, 5, 80)]


def test_match_w_fuzzy_regex2(searcher: RegexSearcher, nlp: Language) -> None:
    """It produces a fuzzy match."""
    doc = nlp("nic bole")
    re_pattern = r"(nicobolas){e<=5}"
    matches = searcher.match(doc, re_pattern, min_r=70)
    assert matches == [(0, 2, 71)]


def test_match_w_fuzzy_regex_w_min_r(searcher: RegexSearcher, nlp: Language) -> None:
    """It produces a fuzzy match."""
    doc = nlp("nic bole")
    re_pattern = r"(nicobolas){e<=5}"
    matches = searcher.match(doc, re_pattern, min_r=80)
    assert matches == []


def test_match_will_expand_on_partial_match_if_partials(
    searcher: RegexSearcher, nlp: Language
) -> None:
    """It extends partial matches to span boundaries."""
    doc = nlp(
        "We want to identify 'USA' even though only first two letters will matched."
    )
    matches = searcher.match(doc, r"[Uu](nited|\.?) ?[Ss](tates|\.?)")
    assert matches == [(5, 6, 100)]


def test_match_will_expand_on_partial_match_at_index_0(
    searcher: RegexSearcher, nlp: Language
) -> None:
    """It extends partial matches to span boundaries - index 0 bug is fixed."""
    doc = nlp("withh something")
    matches = searcher.match(doc, "with")
    assert matches == [(0, 1, 100)]


def test_match_on_german_combination_words(
    searcher: RegexSearcher, nlp: Language
) -> None:
    """It extends partial matches to span boundaries."""
    doc = nlp(
        "We want to identify a geman word combination Aussagekraft or Kraftfahrzeug"
    )
    matches = searcher.match(doc, r"(kraft|Kraft)")
    assert matches == [(8, 9, 100), (10, 11, 100)]


def test_match_will_not_expand_if_not_partials(
    searcher: RegexSearcher, nlp: Language
) -> None:
    """It will not extend partial matches to span boundaries if not partial."""
    doc = nlp(
        "We want to identify 'USA' even though only first two letters will matched."
    )
    matches = searcher.match(doc, r"[Uu](nited|\.?) ?[Ss](tates|\.?)", partial=False)
    assert matches == []


def test_match_will_not_match_if_regex_starts_ends_with_space(
    searcher: RegexSearcher, nlp: Language
) -> None:
    """Regexes that match on spaces will not produce match."""
    doc = nlp(
        "We want to identify US but will fail because regex includes whitespaces."
    )
    matches = searcher.match(doc, r"\s[Uu](nited|\.?) ?[Ss](tates|\.?)\s")
    assert matches == []


def test__map_chars_to_tokens(searcher: RegexSearcher, nlp: Language) -> None:
    """It creates map of character indices to token indices."""
    doc = nlp("Test sentence.")
    char_to_token_map = searcher._map_chars_to_tokens(doc)
    assert char_to_token_map[0] == 0
    assert char_to_token_map[5] == 1
    assert char_to_token_map[13] == 2
    assert len(char_to_token_map) == 13
