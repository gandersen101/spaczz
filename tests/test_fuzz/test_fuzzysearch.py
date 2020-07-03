"""Tests for fuzzymatcher module."""
from typing import Dict

import pytest
from spacy.language import Language
from spacy.tokens import Doc

from spaczz.exceptions import FlexWarning
from spaczz.fuzz.fuzzyconfig import FuzzyConfig
from spaczz.fuzz.fuzzysearch import FuzzySearch


@pytest.fixture
def default_fs() -> FuzzySearch:
    """It returns a default fuzzy searcher."""
    return FuzzySearch()


@pytest.fixture
def initial_matches() -> Dict[int, int]:
    """Example initial fuzzy matches."""
    return {1: 30, 4: 50, 5: 50, 8: 100, 9: 100}


@pytest.fixture
def scan_example(nlp: Language) -> Doc:
    """Example doc for scan_doc tests."""
    return nlp.make_doc("Don't call me Sh1rley")


@pytest.fixture
def adjust_example(nlp: Language) -> Doc:
    """Example doc for adjust_left_right tests."""
    return nlp.make_doc("There was a great basketball player named: Karem Abdul Jabar")


def test_fuzzysearch_config_contains_predefined_defaults(
    default_fs: FuzzySearch,
) -> None:
    """Its config contains predefined defaults."""
    assert default_fs._config._fuzzy_funcs


def test_fuzzysearch_uses_passed_config() -> None:
    """It uses the config passed to it."""
    fc = FuzzyConfig()
    fs = FuzzySearch(config=fc)
    assert fs._config._fuzzy_funcs


def test_fuzzysearch_raises_error_if_config_is_not_fuzzyconfig() -> None:
    """It raises a TypeError if config is not recognized string or FuzzyConfig."""
    with pytest.raises(TypeError):
        FuzzySearch(config="Will cause error")


def test_fuzzysearch_has_empty_config_if_empty_passed() -> None:
    """Its config is empty."""
    fs = FuzzySearch(config="empty")
    assert not fs._config._fuzzy_funcs


def test_match_works_with_defaults(default_fs: FuzzySearch) -> None:
    """Checks match is working as intended."""
    assert default_fs.match("spaczz", "spacy") == 73


def test_match_without_ignore_case(default_fs: FuzzySearch) -> None:
    """Checks ignore_case is working."""
    assert default_fs.match("SPACZZ", "spaczz", ignore_case=False) == 0


def test__calc_flex_with_default(nlp: Language, default_fs: FuzzySearch) -> None:
    """It returns len(query) if set with "default"."""
    query = nlp.make_doc("Test query.")
    assert default_fs._calc_flex(query, "default") == 3


def test__calc_flex_passes_through_valid_value(
    nlp: Language, default_fs: FuzzySearch
) -> None:
    """It passes through a valid flex value (<= len(query))."""
    query = nlp.make_doc("Test query.")
    assert default_fs._calc_flex(query, 1) == 1


def test__calc_flex_warns_if_flex_longer_than_query(
    nlp: Language, default_fs: FuzzySearch
) -> None:
    """It provides UserWarning if flex > len(query)."""
    query = nlp.make_doc("Test query.")
    with pytest.warns(FlexWarning):
        default_fs._calc_flex(query, 5)


def test__calc_flex_raises_error_if_non_valid_value(
    nlp: Language, default_fs: FuzzySearch
) -> None:
    """It raises TypeError if flex is not an int or "default"."""
    query = nlp("Test query.")
    with pytest.raises(TypeError):
        default_fs._calc_flex(query, None)


def test__index_max_returns_key_with_max_value(
    default_fs: FuzzySearch, initial_matches: Dict[int, int]
) -> None:
    """It returns the earliest key with the largest value."""
    assert default_fs._index_max(initial_matches) == 8


def test__indice_maxes_returns_n_keys_with_max_values(
    default_fs: FuzzySearch, initial_matches: Dict[int, int]
) -> None:
    """It returns the n keys correctly sorted."""
    assert default_fs._indice_maxes(initial_matches, 3) == [8, 9, 4]


def test__indice_maxes_returns_all_keys_if_n_is_0(
    default_fs: FuzzySearch, initial_matches: Dict[int, int]
) -> None:
    """It returns input unchanged if n is 0."""
    assert default_fs._indice_maxes(initial_matches, 0) == [1, 4, 5, 8, 9]


def test__scan_doc_returns_matches_over_min_r1(
    default_fs: FuzzySearch, nlp: Language, scan_example: Doc
) -> None:
    """It returns all spans of len(query) in doc if ratio >= min_r1."""
    query = nlp.make_doc("Shirley")
    assert default_fs._scan_doc(
        scan_example, query, fuzzy_func="simple", min_r1=30, ignore_case=True
    ) == {4: 86}


def test__scan_doc_returns_all_matches_with_no_min_r1(
    default_fs: FuzzySearch, nlp: Language, scan_example: Doc
) -> None:
    """It returns all spans of len(query) in doc if min_r1 = 0."""
    query = nlp.make_doc("Shirley")
    assert default_fs._scan_doc(
        scan_example, query, fuzzy_func="simple", min_r1=0, ignore_case=True
    ) == {0: 0, 1: 0, 2: 18, 3: 22, 4: 86}


def test__scan_doc_with_no_matches(
    default_fs: FuzzySearch, nlp: Language, scan_example: Doc
) -> None:
    """It returns None if no matches >= min_r1."""
    query = nlp.make_doc("xenomorph")
    assert (
        default_fs._scan_doc(
            scan_example, query, fuzzy_func="simple", min_r1=30, ignore_case=True
        )
        is None
    )


def test__adjust_left_right_positions_finds_better_match(
    default_fs: FuzzySearch, nlp: Language
) -> None:
    """It optimizes the initial match to find a better match."""
    doc = nlp.make_doc("Patient was prescribed Zithromax tablets.")
    query = nlp.make_doc("zithromax tablet")
    match_values = {0: 30, 2: 50, 3: 97, 4: 50}
    assert default_fs._adjust_left_right_positions(
        doc,
        query,
        match_values,
        pos=3,
        fuzzy_func="simple",
        min_r2=70,
        ignore_case=True,
        flex=2,
    ) == (3, 5, 97)


def test__adjust_left_right_positions_finds_better_match2(
    default_fs: FuzzySearch, nlp: Language, adjust_example: Doc
) -> None:
    """It optimizes the initial match to find a better match."""
    query = nlp.make_doc("Kareem Abdul-Jabbar")
    match_values = {0: 33, 1: 39, 2: 41, 3: 33, 5: 37, 6: 59, 7: 84}
    assert default_fs._adjust_left_right_positions(
        adjust_example,
        query,
        match_values,
        pos=7,
        fuzzy_func="simple",
        min_r2=70,
        ignore_case=True,
        flex=4,
    ) == (8, 11, 89)


def test__adjust_left_right_positions_with_no_flex(
    default_fs: FuzzySearch, nlp: Language
) -> None:
    """It returns the intial match when flex value = 0."""
    doc = nlp.make_doc("Patient was prescribed Zithroma tablets.")
    query = nlp.make_doc("zithromax")
    match_values = {3: 94}
    assert default_fs._adjust_left_right_positions(
        doc,
        query,
        match_values,
        pos=3,
        fuzzy_func="simple",
        min_r2=70,
        ignore_case=True,
        flex=0,
    ) == (3, 4, 94)


def test__filter_overlapping_matches_filters_correctly(default_fs: FuzzySearch) -> None:
    """It only returns the first match if more than one encompass the same tokens."""
    matches = [(1, 2, 80), (1, 3, 70)]
    assert default_fs._filter_overlapping_matches(matches) == [(1, 2, 80)]


def test_best_match_finds_best_match(default_fs: FuzzySearch, nlp: Language) -> None:
    """It finds the single best fuzzy match that meets threshold."""
    doc = nlp("G-rant Anderson lives in TN.")
    query = nlp("Grant Andersen")
    assert default_fs.best_match(doc, query) == (0, 4, 90)


def test_best_match_return_none_when_no_matches(
    default_fs: FuzzySearch, nlp: Language
) -> None:
    """It returns None if no fuzzy match meets threshold."""
    doc = nlp("G-rant Anderson lives in TN.")
    query = nlp("xenomorph")
    assert default_fs.best_match(doc, query) is None


def test_best_match_raises_error_when_doc_not_Doc(
    default_fs: FuzzySearch, nlp: Language
) -> None:
    """Raises a Type error if doc is not a Doc object."""
    doc = "G-rant Anderson lives in TN."
    query = nlp("xenomorph")
    with pytest.raises(TypeError):
        default_fs.best_match(doc, query)


def test_best_match_raises_error_when_query_not_Doc(
    default_fs: FuzzySearch, nlp: Language
) -> None:
    """Raises a Type error if query is not a Doc object."""
    doc = nlp("G-rant Anderson lives in TN.")
    query = "xenomorph"
    with pytest.raises(TypeError):
        default_fs.best_match(doc, query)


def test_multi_match_finds_best_matches(default_fs: FuzzySearch, nlp: Language) -> None:
    """It returns all the fuzzy matches that meet threshold correctly sorted."""
    doc = nlp("chiken from Popeyes is better than chken from Chick-fil-A")
    query = nlp("chicken")
    assert default_fs.multi_match(doc, query, ignore_case=False) == [
        (0, 1, 92),
        (6, 7, 83),
    ]


def test_multi_match_return_empty_list_when_no_matches_after_scan(
    default_fs: FuzzySearch, nlp: Language
) -> None:
    """It returns an empty list if no fuzzy matches meet min_r1 threshold."""
    doc = nlp("G-rant Anderson lives in TN.")
    query = nlp("xenomorph")
    assert default_fs.multi_match(doc, query) == []


def test_multi_match_return_empty_list_when_no_matches_after_adjust(
    default_fs: FuzzySearch, nlp: Language
) -> None:
    """It returns an empty list if no fuzzy matches meet min_r2 threshold."""
    doc = nlp("G-rant Anderson lives in TN.")
    query = nlp("Garth, Anderdella")
    assert default_fs.multi_match(doc, query) == []


def test_multi_match_with_n_less_than_actual_matches(
    default_fs: FuzzySearch, nlp: Language
) -> None:
    """It returns the n best fuzzy matches that meet threshold correctly sorted."""
    doc = nlp("cow, cow, cow, cow")
    query = nlp("cow")
    assert default_fs.multi_match(doc, query, n=2) == [(0, 1, 100), (2, 3, 100)]


def test_multi_match_raises_error_when_doc_not_Doc(
    default_fs: FuzzySearch, nlp: Language
) -> None:
    """It raises a TypeError if doc is not a Doc object."""
    doc = "G-rant Anderson lives in TN."
    query = nlp("xenomorph")
    with pytest.raises(TypeError):
        default_fs.multi_match(doc, query)


def test_multi_match_raises_error_when_query_not_Doc(
    default_fs: FuzzySearch, nlp: Language
) -> None:
    """It raises a TypeError if query is not a Doc object."""
    doc = nlp("G-rant Anderson lives in TN.")
    query = "xenomorph"
    with pytest.raises(TypeError):
        default_fs.multi_match(doc, query)
