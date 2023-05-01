"""Module for TokenSearcher: flexible token searching in spaCy `Doc` objects."""
import itertools
import typing as ty

import regex
from spacy.tokens import Doc
from spacy.tokens import Token
from spacy.vocab import Vocab

from .searchutil import normalize_fuzzy_regex_counts
from .searchutil import parse_regex
from ..registry import get_fuzzy_func


class TokenSearcher:
    """Class for token-based fuzzy match searching in spaCy `Doc` objects."""

    def __init__(self: "TokenSearcher", vocab: Vocab) -> None:
        """Initializes the searcher."""
        self.vocab = vocab

    def match(
        self: "TokenSearcher",
        doc: Doc,
        pattern: ty.List[ty.Dict[str, ty.Any]],
        min_r: int = 75,
    ) -> ty.List[ty.List[ty.Tuple[str, str, int]]]:
        """Finds potential token pattern matches in a `Doc` object.

        Make sure to use uppercase dictionary keys in patterns.

        Args:
            doc: `Doc` object to search for matches.
            pattern: Individual spaCy token pattern.
            min_r: Minimum match ratio required for fuzzy matching.
                Can be overwritten with pattern-level match settings.
                Default is `75`.

        Returns:
            A list of lists with each inner list representing a potential match.
            The inner lists will be populated with key, value, match ratio tuples for
            matched fuzzy token patterns. Non-fuzzy token patterns will have an empty
            string key and value, and a match ratio of `100`.

        Example:
            >>> import spacy
            >>> from spaczz._search import TokenSearcher
            >>> nlp = spacy.blank("en")
            >>> searcher = TokenSearcher(nlp)
            >>> doc = nlp("I was prescribed zithramax and advar")
            >>> pattern = [
                {"TEXT": {"FUZZY": "zithromax"}},
                {"POS": "CCONJ"},
                {"TEXT": {"FREGEX": "(advair){e<=1}"}}
                ]
            >>> searcher.match(doc, pattern)
            [[('TEXT', 'zithramax', 89), ('', '', 100), ('TEXT', 'advar', 91)]]
        """
        matches = []
        matches = [
            self._iter_pattern(seq, pattern, min_r=min_r)
            for seq in self._n_wise(doc, len(pattern))
        ]
        return [
            match
            for i, match in enumerate(matches)
            if match and match not in matches[:i]
        ]

    @staticmethod
    def fuzzy_compare(
        s1: str,
        s2: str,
        *,
        ignore_case: bool = True,
        min_r: int = 0,
        fuzzy_func: str = "simple",
    ) -> int:
        """Peforms fuzzy matching between two strings.

        Applies the given fuzzy matching algorithm (`fuzzy_func`)
        to two strings and returns the resulting fuzzy ratio.

        Args:
            s1: First string for comparison.
            s2: Second string for comparison.
            ignore_case: Whether to lower-case `s1` and `s2`
                before comparison or not. Default is `True`.
            min_r: Minimum ratio needed to match as a value between `0` and `100`.
                For ratio < min_r, `0` is returned instead.
                Default is `0`, which deactivates this behaviour.
            fuzzy_func: Key name of fuzzy matching function to use.

        Returns:
            The fuzzy ratio between `s1` and `s2`.

        Example:
            >>> import spacy
            >>> from spaczz._search import TokenSearcher
            >>> nlp = spacy.blank("en")
            >>> searcher = TokenSearcher(nlp.vocab)
            >>> searcher.fuzzy_compare("spaczz", "spacy")
            73
        """
        if ignore_case:
            s1 = s1.lower()
            s2 = s2.lower()

        return round(get_fuzzy_func(fuzzy_func)(s1, s2, score_cutoff=min_r))

    @staticmethod
    def regex_compare(
        text: str,
        pattern: str,
        *,
        ignore_case: bool = False,
        min_r: int = 0,
        fuzzy_weights: str = "indel",
        predef: bool = False,
    ) -> int:
        """Performs fuzzy-regex supporting regex matching between two strings.

        Args:
            text: The string to match against.
            pattern: The regex pattern string.
            ignore_case: Whether to lower-case text before matching.
                Default is `True`.
            min_r: Minimum match ratio required for fuzzy regex matching.
                Default is `75`.
            fuzzy_weights: Name of weighting method for regex insertion, deletion, and
                substituion counts.
            predef: Whether regex should be interpreted as a key to
                a predefined regex pattern or not. Default is `False`.

        Returns:
            The fuzzy ratio of the pattern to the text based on insertions,
            substitutions, and deletions.

        Example:
            >>> import spacy
            >>> from spaczz._search import TokenSearcher
            >>> nlp = spacy.blank("en")
            >>> searcher = TokenSearcher(nlp)
            >>> searcher.regex_compare("sequel", "(sql){i<=3}")
            67
        """
        pattern_ = parse_regex(pattern, predef=predef)
        if ignore_case:
            text = text.lower()
        match = regex.match(pattern_, text)
        if match:
            r = normalize_fuzzy_regex_counts(
                match.group(0),
                fuzzy_counts=getattr(match, "fuzzy_counts", (0, 0, 0)),
                fuzzy_weights=fuzzy_weights,
            )
            if r >= min_r:
                return r
        return 0

    def _iter_pattern(
        self: "TokenSearcher",
        seq: ty.Tuple[Token, ...],
        pattern: ty.List[ty.Dict[str, ty.Any]],
        min_r: int,
    ) -> ty.List[ty.Tuple[str, str, int]]:
        """Evaluates each token in a pattern against a doc token sequence."""
        seq_matches: ty.List[ty.Tuple[str, str, int]] = []
        for i, token in enumerate(pattern):
            pattern_dict, case, case_bool = self._parse_case(token)
            if isinstance(pattern_dict, dict):
                pattern_text, pattern_type = self._parse_type(pattern_dict)
                if pattern_text and pattern_type == "FUZZY":
                    r = self.fuzzy_compare(
                        seq[i].text,
                        pattern_text,
                        ignore_case=case_bool,
                        min_r=pattern_dict.get("MIN_R", min_r),
                        fuzzy_func=pattern_dict.get("FUZZY_FUNC", "simple"),
                    )
                    if r:
                        seq_matches.append((case, seq[i].text, r))
                    else:
                        return []
                elif pattern_text and pattern_type == "FREGEX":
                    r = self.regex_compare(
                        seq[i].text,
                        pattern_text,
                        predef=pattern_dict.get("PREDEF", False),
                        ignore_case=case_bool,
                        min_r=pattern_dict.get("MIN_R", min_r),
                        fuzzy_weights=pattern_dict.get("FUZZY_WEIGHTS", "indel"),
                    )
                    if r:
                        seq_matches.append((case, seq[i].text, r))
                    else:
                        return []
                else:
                    seq_matches.append(("", "", 100))
            else:
                seq_matches.append(("", "", 100))
        return seq_matches

    @staticmethod
    def _parse_case(
        token: ty.Dict[str, ty.Any]
    ) -> ty.Tuple[ty.Union[str, ty.Dict[ty.Any, ty.Any], None], str, bool]:
        """Parses the case of a token pattern."""
        text = token.get("TEXT")
        if text:
            return text, "TEXT", False
        return token.get("LOWER"), "LOWER", True

    @staticmethod
    def _parse_type(pattern_dict: ty.Dict[str, ty.Any]) -> ty.Tuple[str, str]:
        """Parses the type of a token pattern."""
        fuzzy_text = pattern_dict.get("FUZZY")
        regex_text = pattern_dict.get("FREGEX")
        if fuzzy_text:
            return fuzzy_text, "FUZZY"
        elif regex_text:
            return regex_text, "FREGEX"
        return "", ""

    @staticmethod
    def _n_wise(iterable: ty.Iterable[ty.Any], n: int) -> ty.Iterator[ty.Any]:
        """Iterates over an iterable in slices of length n by one step at a time."""
        iterables = itertools.tee(iterable, n)
        for i in range(len(iterables)):
            for _ in range(i):
                next(iterables[i], None)
        return zip(*iterables)  # noqa B905
