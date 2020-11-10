"""Tests for fuzzymatcher module."""
from typing import Dict

import pytest
from spacy.language import Language
from spacy.tokens import Doc

from spaczz.exceptions import FlexWarning
from spaczz.similarity.similaritysearcher import SimilaritySearcher


@pytest.fixture
def searcher() -> SimilaritySearcher:
    """It returns a default fuzzy searcher."""
    return SimilaritySearcher()


@pytest.fixture
def initial_matches() -> Dict[int, int]:
    """Example initial fuzzy matches."""
    return {1: 30, 4: 50, 5: 50, 8: 100, 9: 100}


@pytest.fixture
def scan_example(model: Language) -> Doc:
    """Example doc for scan_doc tests."""
    return model("The move 'Airplane' has the famous quote: 'Don't call me Shirley.'")


@pytest.fixture
def adjust_example(model: Language) -> Doc:
    """Example doc for adjust_left_right tests."""
    return model("There was a great basketball player named: Kareem Abdul-Jabbar")


def test_compare_works_with_defaults(
    searcher: SimilaritySearcher, model: Language
) -> None:
    """Checks compare is working as intended."""
    assert searcher.compare(model("I like apples."), model("I like grapes.")) == 94


def test_compare_returns_0_w_no_vector(
    searcher: SimilaritySearcher, model: Language
) -> None:
    """Checks compare returns 0 when vector does not exist for span/token."""
    assert searcher.compare(model("spaczz"), model("python")) == 0


def test__calc_flex_with_default(nlp: Language, searcher: SimilaritySearcher) -> None:
    """It returns len(query) if set with "default"."""
    query = nlp.make_doc("Test query.")
    assert searcher._calc_flex(query, "default") == 3


def test__calc_flex_passes_through_valid_value(
    nlp: Language, searcher: SimilaritySearcher
) -> None:
    """It passes through a valid flex value (<= len(query))."""
    query = nlp.make_doc("Test query.")
    assert searcher._calc_flex(query, 1) == 1


def test__calc_flex_warns_if_flex_longer_than_query(
    nlp: Language, searcher: SimilaritySearcher
) -> None:
    """It provides UserWarning if flex > len(query)."""
    query = nlp.make_doc("Test query.")
    with pytest.warns(FlexWarning):
        searcher._calc_flex(query, 5)


def test__calc_flex_raises_error_if_non_valid_value(
    nlp: Language, searcher: SimilaritySearcher
) -> None:
    """It raises TypeError if flex is not an int or "default"."""
    query = nlp("Test query.")
    with pytest.raises(TypeError):
        searcher._calc_flex(query, None)


def test__indice_maxes_returns_n_keys_with_max_values(
    searcher: SimilaritySearcher, initial_matches: Dict[int, int]
) -> None:
    """It returns the n keys correctly sorted."""
    assert searcher._indice_maxes(initial_matches, 3) == [8, 9, 4]


def test__indice_maxes_returns_all_keys_if_n_is_0(
    searcher: SimilaritySearcher, initial_matches: Dict[int, int]
) -> None:
    """It returns input unchanged if n is 0."""
    assert searcher._indice_maxes(initial_matches, 0) == [1, 4, 5, 8, 9]


def test__scan_doc_returns_matches_over_min_r1(
    searcher: SimilaritySearcher, model: Language, scan_example: Doc
) -> None:
    """It returns all spans of len(query) in doc if ratio >= min_r1."""
    query = model("famous line")
    assert searcher._scan_doc(scan_example, query, min_r1=50) == {
        0: 51,
        5: 55,
        6: 82,
        7: 75,
    }


def test__scan_doc_returns_all_matches_with_no_min_r1(
    searcher: SimilaritySearcher, model: Language, scan_example: Doc
) -> None:
    """It returns all spans of len(query) in doc if min_r1 = 0."""
    query = model("famous line")
    assert searcher._scan_doc(scan_example, query, min_r1=0) == {
        0: 51,
        1: 39,
        2: 36,
        3: 36,
        4: 42,
        5: 55,
        6: 82,
        7: 75,
        8: 30,
        9: 22,
        10: 39,
        11: 38,
        12: 47,
        13: 43,
        14: 24,
        15: 22,
        16: 39,
    }


def test__scan_doc_with_no_matches(
    searcher: SimilaritySearcher, model: Language, scan_example: Doc
) -> None:
    """It returns None if no matches >= min_r1."""
    query = model("xenomorph")
    assert searcher._scan_doc(scan_example, query, min_r1=50) is None


def test__adjust_left_right_positions_finds_better_match(
    searcher: SimilaritySearcher, model: Language
) -> None:
    """It optimizes the initial match to find a better match."""
    doc = model("The Patient had ordered Zithromax tablets.")
    query = model("Patient was prescribed Zithromax tablets.")
    match_values = {0: 92, 1: 95}
    assert searcher._adjust_left_right_positions(
        doc, query, match_values, pos=1, min_r2=70, flex=6,
    ) == (1, 7, 95)


def test__adjust_left_right_positions_finds_better_match2(
    searcher: SimilaritySearcher, model: Language, adjust_example: Doc
) -> None:
    """It optimizes the initial match to find a better match."""
    query = model("amazing basketball player")
    match_values = {1: 57, 2: 85, 3: 97, 4: 88, 5: 68}
    assert searcher._adjust_left_right_positions(
        adjust_example, query, match_values, pos=2, min_r2=70, flex=3,
    ) == (3, 6, 97)


def test__adjust_left_right_positions_finds_better_match3(
    searcher: SimilaritySearcher, model: Language, adjust_example: Doc
) -> None:
    """It optimizes the initial match to find a better match."""
    query = model("amazing basketball player")
    match_values = {1: 57, 2: 85, 3: 97, 4: 88, 5: 68}
    assert searcher._adjust_left_right_positions(
        adjust_example, query, match_values, pos=4, min_r2=70, flex=3,
    ) == (3, 6, 97)


def test__adjust_left_right_positions_with_no_flex(
    searcher: SimilaritySearcher, model: Language, adjust_example: Doc
) -> None:
    """It returns the intial match when flex value = 0."""
    query = model("amazing basketball player")
    match_values = {1: 57, 2: 85, 3: 97, 4: 88, 5: 68}
    assert searcher._adjust_left_right_positions(
        adjust_example, query, match_values, pos=3, min_r2=70, flex=0,
    ) == (3, 6, 97)


def test__filter_overlapping_matches_filters_correctly(
    searcher: SimilaritySearcher,
) -> None:
    """It only returns the first match if more than one encompass the same tokens."""
    matches = [(1, 2, 80), (1, 3, 70)]
    assert searcher._filter_overlapping_matches(matches) == [(1, 2, 80)]


def test_match_finds_best_matches(
    searcher: SimilaritySearcher, model: Language
) -> None:
    """It returns all the matches that meet threshold correctly sorted."""
    doc = model("I would love a coke.")
    query = model("soda")
    assert searcher.match(doc, query, min_r2=65) == [
        (4, 5, 67),
    ]


def test_match_return_empty_list_when_no_matches_after_scan(
    searcher: SimilaritySearcher, model: Language
) -> None:
    """It returns an empty list if no matches meet min_r1 threshold."""
    doc = model("Grant Andersen lives in TN.")
    query = model("xenomorph")
    assert searcher.match(doc, query) == []


def test_match_return_empty_list_when_no_matches_after_adjust(
    searcher: SimilaritySearcher, model: Language
) -> None:
    """It returns an empty list if no matches meet min_r2 threshold."""
    doc = model("Grant Andersen lives in TN.")
    query = model("resides")
    assert searcher.match(doc, query) == []


def test_match_with_n_less_than_actual_matches(
    searcher: SimilaritySearcher, model: Language
) -> None:
    """It returns the n best matches that meet threshold correctly sorted."""
    doc = model("cow, cow, cow, cow")
    query = model("cow")
    assert searcher.match(doc, query, n=2) == [(0, 1, 100), (2, 3, 100)]


def test_match_raises_error_when_doc_not_Doc(
    searcher: SimilaritySearcher, model: Language
) -> None:
    """It raises a TypeError if doc is not a Doc object."""
    doc = "G-rant Anderson lives in TN."
    query = model("xenomorph")
    with pytest.raises(TypeError):
        searcher.match(doc, query)


def test_match_raises_error_if_query_not_Doc(
    searcher: SimilaritySearcher, model: Language
) -> None:
    """It raises a TypeError if query not a doc."""
    doc = model("This is a doc")
    query = "Not a doc"
    with pytest.raises(TypeError):
        searcher.match(doc, query)
