"""Module for search utilities."""
import itertools
import typing as ty

import regex as re

from ..customtypes import SearchResult
from ..exceptions import RegexParseError
from ..registry import get_re_pattern
from ..registry import get_re_weights


def filter_overlapping_matches(
    matches: ty.Iterable[SearchResult],
) -> ty.List[SearchResult]:
    """Prevents multiple matches from overlapping.

    Expects matches to be pre-sorted by descending ratio
    then ascending start index.
    If more than one match includes the same tokens
    the first of these matches is kept.

    Args:
        matches: Iterable of matches (start index, end index, ratio tuples).

    Returns:
        The filtered list of matches.

    Example:
        >>> from spaczz._search.searchutil import filter_overlapping_matches
        >>> matches = [(1, 3, 80), (1, 2, 70)]
        >>> filter_overlapping_matches(matches)
        [(1, 3, 80)]
    """
    filtered_matches: ty.List[SearchResult] = []
    for match in matches:
        if not set(range(match[0], match[1])).intersection(
            itertools.chain(*[set(range(n[0], n[1])) for n in filtered_matches])
        ):
            filtered_matches.append(match)
    return filtered_matches


def parse_regex(
    regex_str: str,
    predef: bool = False,
) -> ty.Pattern:
    """Parses a string into a regex pattern.

    Will treat `regex_str` as a key name for a predefined regex if `predef=True`.

    Args:
        regex_str: String to compile into a regex pattern.
        predef: Whether regex should be interpreted as a key to
            a predefined regex pattern or not. Default is `False`.

    Returns:
        A compiled regex pattern.

    Raises:
        RegexParseError: If regex compilation produces any errors.

    Example:
        >>> from spaczz._search.searchutil import parse_regex
        >>> pattern = parse_regex("Test")
        >>> isinstance(pattern, re.Pattern)
        True
    """
    if predef:
        return get_re_pattern(regex_str)
    try:
        return re.compile(
            regex_str,
        )
    except (re._regex_core.error, TypeError, ValueError) as e:
        raise RegexParseError(e)


def normalize_fuzzy_regex_counts(
    match: str, fuzzy_counts: ty.Tuple[int, int, int], fuzzy_weights: str
) -> int:
    """Normalizes fuzzy regex counts to a fuzzy ratio."""
    if fuzzy_counts == (0, 0, 0):
        return 100

    weights = get_re_weights(fuzzy_weights)
    s1_len = len(match) - fuzzy_counts[1] + fuzzy_counts[2]
    s2_len = len(match)
    fuzzy_counts_ = (
        a * b
        for a, b in zip(  # noqa: B905
            (fuzzy_counts[1], fuzzy_counts[2], fuzzy_counts[0]), weights
        )
    )

    if weights[2] <= weights[0] + weights[1]:
        dist_max = min(s1_len, s2_len) * weights[2]
    else:
        dist_max = s1_len * weights[1] + s2_len * weights[0]

    if s1_len > s2_len:
        dist_max += (s1_len - s2_len) * weights[1]
    elif s1_len < s2_len:
        dist_max += (s2_len - s1_len) * weights[0]

    r = 100 * sum(fuzzy_counts_) / dist_max
    return round(100 - r)
