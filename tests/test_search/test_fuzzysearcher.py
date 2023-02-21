"""Tests for fuzzysearcher module."""
import typing as ty

import pytest
from spacy.language import Language
from spacy.tokens import Doc

from spaczz.exceptions import FlexWarning
from spaczz.exceptions import RatioWarning
from spaczz.search import FuzzySearcher


@pytest.fixture
def searcher(nlp: Language) -> FuzzySearcher:
    """It returns a fuzzy searcher."""
    return FuzzySearcher(vocab=nlp.vocab)


@pytest.fixture
def initial_matches() -> ty.Dict[int, int]:
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
    """It returns len(query) if set with "max"."""
    query = nlp("Test query two")
    assert searcher._calc_flex(query, "max") == 3


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
    """It raises ValueError if flex is not an int or "default"."""
    query = nlp("Test query.")
    with pytest.raises(ValueError):
        searcher._calc_flex(query, None)  # type: ignore


def test__set_ratios_passes_set_min_r1_r2_through(searcher: FuzzySearcher) -> None:
    """It passes set min_r1 and min_r2 values through."""
    assert searcher._set_ratios(75, 40, 80) == (40, 80)


def test__set_ratios_passes_set_min_r1_through(searcher: FuzzySearcher) -> None:
    """It passes set min_r1 value through while setting min_r2."""
    assert searcher._set_ratios(75, 40, None) == (40, 75)


def test__set_ratios_passes_set_min_r2_through(searcher: FuzzySearcher) -> None:
    """It passes set min_r2 value through while setting min_r1."""
    assert searcher._set_ratios(75, None, 80) == (50, 80)


def test__set_ratios_sets_min_r1_r2(searcher: FuzzySearcher) -> None:
    """It passes set min_r2 value through while setting min_r1."""
    assert searcher._set_ratios(75, None, None) == (50, 75)


def test__set_ratios_respects_0s(searcher: FuzzySearcher) -> None:
    """It passes set min_r1 and min_r2 values through, respecting 0s."""
    assert searcher._set_ratios(75, 0, 0) == (0, 0)


def test__check_ratios_passes_valid_values_w_flex(searcher: FuzzySearcher) -> None:
    """It passes through valid ratios with no changes."""
    assert searcher._check_ratios(50, 75, 100, 1) == (50, 75, 100)


def test__check_ratios_passes_valid_values_wo_flex(searcher: FuzzySearcher) -> None:
    """It passes through valid ratios changing `min_r1` to equal `min_r2`."""
    assert searcher._check_ratios(50, 75, 100, 0) == (75, 75, 100)


def test__check_ratios_ignores_issues_wo_flex(searcher: FuzzySearcher) -> None:
    """Changes `min_r1` to equal `min_r2` but ignores unnecessary `thresh`."""
    assert searcher._check_ratios(80, 75, 70, 0) == (75, 75, 70)


def test__check_ratios_warns_if_minr1_greater_min_r2(searcher: FuzzySearcher) -> None:
    """It raises a `RatioWarning`."""
    with pytest.warns(RatioWarning):
        ratios = searcher._check_ratios(80, 75, 100, 1)
    assert ratios == (75, 75, 100)


def test__check_ratios_warns_if_thresh_less_min_r2(searcher: FuzzySearcher) -> None:
    """It raises a `RatioWarning`."""
    with pytest.warns(RatioWarning):
        ratios = searcher._check_ratios(50, 75, 70, 1)
    assert ratios == (50, 75, 75)


def test__scan_returns_matches_over_min_r1(
    searcher: FuzzySearcher, nlp: Language, scan_example: Doc
) -> None:
    """It returns all spans of len(query) in doc if ratio >= min_r1."""
    query = nlp("Shirley")
    assert searcher._scan(
        scan_example, query, fuzzy_func="simple", min_r1=30, ignore_case=True
    ) == {4: 86}


def test__scan_returns_all_matches_gt0_with_no_min_r1(
    searcher: FuzzySearcher, nlp: Language, scan_example: Doc
) -> None:
    """It returns all spans of len(query) in doc if min_r1 = 0."""
    query = nlp("Shirley")
    assert searcher._scan(
        scan_example, query, fuzzy_func="simple", min_r1=0, ignore_case=True
    ) == {2: 18, 3: 22, 4: 86}


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


def test__optimize_finds_better_match_with_max_flex(
    searcher: FuzzySearcher, nlp: Language, adjust_example: Doc
) -> None:
    """It optimizes the initial match to find a better match when flex = max."""
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
    """It returns the initial match when flex value = 0."""
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
    """It returns the initial match when flex value = 0."""
    doc = nlp("trabalho, investimento e escolhas corajosas,")
    query = nlp("Courtillier Musqué")
    assert searcher.match(doc, query, flex="max") == []


def test_match_finds_best_matches(searcher: FuzzySearcher, nlp: Language) -> None:
    """It returns all the fuzzy matches that meet threshold correctly sorted."""
    doc = nlp("chiken from Popeyes is better than chken from Chick-fil-A")
    query = nlp("chicken")
    assert searcher.match(doc, query, ignore_case=False) == [
        (0, 1, 92, "chicken"),
        (6, 7, 83, "chicken"),
    ]


def test_match_finds_best_matches2(searcher: FuzzySearcher, nlp: Language) -> None:
    """It returns all the fuzzy matches that meet threshold correctly sorted."""
    doc = nlp("My favorite wine is white goldriesling.")
    query = nlp("gold riesling")
    assert searcher.match(doc, query) == [
        (5, 6, 96, "gold riesling"),
    ]


def test_match_finds_best_matches3(searcher: FuzzySearcher, nlp: Language) -> None:
    """It returns all the fuzzy matches that meet threshold correctly sorted."""
    doc = nlp("My favorite wine is white gold riesling.")
    query = nlp("goldriesling")
    assert searcher.match(doc, query, flex="max") == [
        (5, 7, 96, "goldriesling"),
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
