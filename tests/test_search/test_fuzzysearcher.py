"""Tests for fuzzysearcher module."""
from typing import Dict

import pytest
from spacy.language import Language
from spacy.tokens import Doc

from spaczz.exceptions import FlexWarning
from spaczz.search import FuzzySearcher


@pytest.fixture
def searcher(nlp: Language) -> FuzzySearcher:
    """It returns a fuzzy searcher."""
    return FuzzySearcher(vocab=nlp.vocab)


@pytest.fixture
def initial_matches() -> Dict[int, int]:
    """Example initial fuzzy matches."""
    return {1: 30, 4: 50, 5: 50, 8: 100, 9: 100}


@pytest.fixture
def scan_example(nlp: Language) -> Doc:
    """Example doc for scan_doc tests."""
    return nlp("Don't call me Sh1rley")


@pytest.fixture
def adjust_example(nlp: Language) -> Doc:
    """Example doc for adjust_left_right tests."""
    return nlp("There was a great basketball player named: Karem Abdul Jabar")


def test_compare_works_with_defaults(searcher: FuzzySearcher, nlp: Language) -> None:
    """Checks compare is working as intended."""
    assert searcher.compare(nlp("spaczz"), nlp("spacy")) == 73


def test_compare_without_ignore_case(searcher: FuzzySearcher, nlp: Language) -> None:
    """Checks ignore_case is working."""
    assert searcher.compare(nlp("SPACZZ"), nlp("spaczz"), ignore_case=False) == 0


def test_compare_raises_error_with_unknown_func_name(
    searcher: FuzzySearcher, nlp: Language
) -> None:
    """It raises a ValueError if fuzzy_func does not match a predefined key name."""
    with pytest.raises(ValueError):
        assert searcher.compare(nlp("spaczz"), nlp("spacy"), fuzzy_func="unknown")


def test__calc_flex_with_default(nlp: Language, searcher: FuzzySearcher) -> None:
    """It returns len(query) // 2 if set with "default"."""
    query = nlp("Test query")
    assert searcher._calc_flex(query, "default") == 1


def test__calc_flex_with_max(nlp: Language, searcher: FuzzySearcher) -> None:
    """It returns max(len(query) - 1, 0) if set with "max"."""
    query = nlp("Test query two")
    assert searcher._calc_flex(query, "max") == 2


def test__calc_flex_with_min(nlp: Language, searcher: FuzzySearcher) -> None:
    """It returns 0 if set with "min"."""
    query = nlp("Test query")
    assert searcher._calc_flex(query, "min") == 0


def test__calc_flex_passes_through_valid_value(
    nlp: Language, searcher: FuzzySearcher
) -> None:
    """It passes through a valid flex value (<= len(query))."""
    query = nlp("Test query")
    assert searcher._calc_flex(query, 0) == 0


def test__calc_flex_warns_if_flex_longer_than_query(
    nlp: Language, searcher: FuzzySearcher
) -> None:
    """It provides UserWarning if flex > len(query)."""
    query = nlp("Test query")
    with pytest.warns(FlexWarning):
        flex = searcher._calc_flex(query, 5)
        assert flex == 2


def test__calc_flex_warns_if_flex_less_than_0(
    nlp: Language, searcher: FuzzySearcher
) -> None:
    """It provides UserWarning if flex < 0."""
    query = nlp("Test query")
    with pytest.warns(FlexWarning):
        flex = searcher._calc_flex(query, -1)
        assert flex == 0


def test__calc_flex_raises_error_if_non_valid_value(
    nlp: Language, searcher: FuzzySearcher
) -> None:
    """It raises TypeError if flex is not an int or "default"."""
    query = nlp("Test query.")
    with pytest.raises(TypeError):
        searcher._calc_flex(query, None)  # type: ignore


def test__scan_returns_matches_over_min_r1(
    searcher: FuzzySearcher, nlp: Language, scan_example: Doc
) -> None:
    """It returns all spans of len(query) in doc if ratio >= min_r1."""
    query = nlp("Shirley")
    assert searcher._scan(
        scan_example, query, fuzzy_func="simple", min_r1=30, ignore_case=True
    ) == {4: 86}


def test__scan_returns_all_matches_with_no_min_r1(
    searcher: FuzzySearcher, nlp: Language, scan_example: Doc
) -> None:
    """It returns all spans of len(query) in doc if min_r1 = 0."""
    query = nlp("Shirley")
    assert searcher._scan(
        scan_example, query, fuzzy_func="simple", min_r1=0, ignore_case=True
    ) == {0: 0, 1: 0, 2: 18, 3: 22, 4: 86}


def test__scan_with_no_matches(
    searcher: FuzzySearcher, nlp: Language, scan_example: Doc
) -> None:
    """It returns None if no matches >= min_r1."""
    query = nlp("xenomorph")
    assert (
        searcher._scan(
            scan_example, query, fuzzy_func="simple", min_r1=30, ignore_case=True
        )
        is None
    )


def test__scan_returns_none_w_empty_query(
    searcher: FuzzySearcher, nlp: Language, scan_example: Doc
) -> None:
    """It returns None if passed an empty query string."""
    query = nlp("")
    assert (
        searcher._scan(
            scan_example, query, fuzzy_func="simple", min_r1=25, ignore_case=True
        )
        is None
    )


def test__optimize_finds_better_match(searcher: FuzzySearcher, nlp: Language) -> None:
    """It optimizes the initial match to find a better match."""
    doc = nlp("Patient was prescribed Zithromax tablets.")
    query = nlp("zithromax tablet")
    match_values = {0: 30, 2: 50, 3: 97, 4: 50}
    assert searcher._optimize(
        doc,
        query,
        match_values,
        pos=3,
        fuzzy_func="simple",
        min_r2=70,
        ignore_case=True,
        flex=1,
        thresh=100,
    ) == (3, 5, 97)


def test__optimize_finds_better_match2(
    searcher: FuzzySearcher, nlp: Language, adjust_example: Doc
) -> None:
    """It optimizes the initial match to find a better match."""
    query = nlp("Kareem Abdul-Jabbar")
    match_values = {0: 33, 1: 39, 2: 41, 3: 33, 5: 37, 6: 59, 7: 84}
    assert searcher._optimize(
        adjust_example,
        query,
        match_values,
        pos=7,
        fuzzy_func="simple",
        min_r2=70,
        ignore_case=True,
        flex=4,
        thresh=100,
    ) == (8, 11, 89)


def test__optimize_with_no_flex(searcher: FuzzySearcher, nlp: Language) -> None:
    """It returns the intial match when flex value = 0."""
    doc = nlp("Patient was prescribed Zithroma tablets.")
    query = nlp("zithromax")
    match_values = {3: 94}
    assert searcher._optimize(
        doc,
        query,
        match_values,
        pos=3,
        fuzzy_func="simple",
        min_r2=70,
        ignore_case=True,
        flex=0,
        thresh=100,
    ) == (3, 4, 94)


def test__optimize_where_bpl_would_equal_bpr(
    searcher: FuzzySearcher, nlp: Language
) -> None:
    """It returns the intial match when flex value = 0."""
    doc = nlp("trabalho, investimento e escolhas corajosas,")
    query = nlp("Courtillier Musqué")
    assert searcher.match(doc, query) == []


def test__filter_overlapping_matches_filters_correctly(
    searcher: FuzzySearcher,
) -> None:
    """It only returns the first match if more than one encompass the same tokens."""
    matches = [(1, 2, 80), (1, 3, 70)]
    assert searcher._filter_overlapping_matches(matches) == [(1, 2, 80)]


def test_match_finds_best_matches(searcher: FuzzySearcher, nlp: Language) -> None:
    """It returns all the fuzzy matches that meet threshold correctly sorted."""
    doc = nlp("chiken from Popeyes is better than chken from Chick-fil-A")
    query = nlp("chicken")
    assert searcher.match(doc, query, ignore_case=False) == [
        (0, 1, 92),
        (6, 7, 83),
    ]


def test_match_return_empty_list_when_no_matches_after_scan(
    searcher: FuzzySearcher, nlp: Language
) -> None:
    """It returns an empty list if no fuzzy matches meet min_r1 threshold."""
    doc = nlp("G-rant Anderson lives in TN.")
    query = nlp("xenomorph")
    assert searcher.match(doc, query) == []


def test_match_return_empty_list_when_no_matches_after_adjust(
    searcher: FuzzySearcher, nlp: Language
) -> None:
    """It returns an empty list if no fuzzy matches meet min_r2 threshold."""
    doc = nlp("G-rant Anderson lives in TN.")
    query = nlp("Garth, Anderdella")
    assert searcher.match(doc, query) == []


def test_match_raises_error_when_doc_not_doc_obj(
    searcher: FuzzySearcher, nlp: Language
) -> None:
    """It raises a TypeError if doc is not a Doc object."""
    doc = "G-rant Anderson lives in TN."
    query = nlp("xenomorph")
    with pytest.raises(TypeError):
        searcher.match(doc, query)


def test_match_raises_error_if_query_not_doc_obj(
    searcher: FuzzySearcher, nlp: Language
) -> None:
    """It raises a TypeError if query not a doc."""
    doc = nlp("This is a doc")
    query = "Not a doc"
    with pytest.raises(TypeError):
        searcher.match(doc, query)


def test_match_returns_empty_list_if_query_empty(
    searcher: FuzzySearcher, nlp: Language
) -> None:
    """Returns empty if query is empty string."""
    doc = nlp("This is a doc")
    query = nlp("")
    assert searcher.match(doc, query) == []


def test_match_returns_empty_list_if_doc_empty(
    searcher: FuzzySearcher, nlp: Language
) -> None:
    """Returns empty list if doc is empty string."""
    doc = nlp("")
    query = nlp("test")
    assert searcher.match(doc, query) == []


def test_match_returns_empty_list_if_doc_query_empty(
    searcher: FuzzySearcher, nlp: Language
) -> None:
    """Returns empty list if doc is empty string."""
    doc = nlp("")
    query = nlp("")
    assert searcher.match(doc, query) == []
