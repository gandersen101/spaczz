"""Module for TokenSearcher: flexible token searching in spaCy `Doc` objects."""
from __future__ import annotations

from typing import Any, Optional, Union

import regex
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab

from ..process import FuzzyFuncs
from ..util import n_wise


class TokenSearcher:
    """Class for flexbile token searching in spaCy `Doc` objects.

    Uses individual (and extended) spaCy token matching patterns to find
    match candidates. Candidates are used to generate new patterns to add
    to a spaCy `Matcher`.

    "FUZZY" and "FREGEX" are the two additional spaCy token pattern options.

    For example:
        {"TEXT": {"FREGEX": "(database){e<=1}"}},
        {"LOWER": {"FUZZY": "access", "MIN_R": 85, "FUZZY_FUNC": "quick_lev"}}

    Make sure to use uppercase dictionary keys in patterns.

    Attributes:
        vocab (Vocab): The shared vocabulary.
            Included for consistency and potential future-state.
        _fuzzy_funcs (FuzzyFuncs):
            Container class housing fuzzy matching functions.
            Functions are accessible via the classes `get()` method
            by their given key name. The following rapidfuzz matching
            functions with default settings are available:
            "simple" = `ratio`
            "quick" = `QRatio`
            "quick_lev" = `quick_lev_ratio`
    """

    def __init__(self: TokenSearcher, vocab: Vocab) -> None:
        """Initializes a token searcher.

        Args:
            vocab: A spaCy `Vocab` object.
                Purely for consistency between spaCy
                and spaczz matcher APIs for now.
                spaczz matchers are mostly pure-Python
                currently and do not share vocabulary
                with spaCy pipelines.
        """
        self.vocab = vocab
        self._fuzzy_funcs: FuzzyFuncs = FuzzyFuncs(match_type="token")

    def fuzzy_compare(
        self: TokenSearcher,
        a: str,
        b: str,
        ignore_case: bool = True,
        fuzzy_func: str = "simple",
    ) -> int:
        """Peforms fuzzy matching between two strings.

        Applies the given fuzzy matching algorithm (fuzzy_func)
        to two strings and returns the resulting fuzzy ratio.

        Args:
            a: First string for comparison.
            b: Second string for comparison.
            ignore_case: Whether to lower-case a and b
                before comparison or not. Default is `True`.
            fuzzy_func: Key name of fuzzy matching function to use.
                The following rapidfuzz matching functions with default
                settings are available:
                "simple" = `ratio`
                "quick" = `QRatio`
                "quick_lev" = `quick_lev_ratio`
                Default is `"simple"`.

        Returns:
            The fuzzy ratio between a and b.

        Example:
            >>> import spacy
            >>> from spaczz.search import TokenSearcher
            >>> nlp = spacy.blank("en")
            >>> searcher = TokenSearcher(nlp.vocab)
            >>> searcher.fuzzy_compare("spaczz", "spacy")
            73
        """
        if ignore_case:
            a = a.lower()
            b = b.lower()
        return round(self._fuzzy_funcs.get(fuzzy_func)(a, b))

    def match(
        self: TokenSearcher,
        doc: Doc,
        pattern: list[dict[str, Any]],
        min_r: int = 75,
        fuzzy_func: str = "simple",
    ) -> list[list[Optional[tuple[str, str]]]]:
        """Finds potential token pattern matches in a `Doc` object.

        Make sure to use uppercase dictionary keys in patterns.

        Args:
            doc: `Doc` object to search over.
            pattern: Individual spaCy token pattern.
            min_r: Minimum match ratio required for fuzzy matching.
                Can be overwritten with token pattern options.
                Default is `75`.
            fuzzy_func: Fuzzy matching function to use.
                Can be overwritten with token pattern options.
                Default is `simple`.

        Returns:
            A list of lists with each inner list representing a potential match.
            The inner lists will be populated with key, value tuples of token matches
            and `None` for placeholder tokens representing non-fuzzy tokens.

        Raises:
            TypeError: doc must be a `Doc` object.
            TypeError: pattern must be a `Sequence`.
            ValueError: pattern cannot have zero tokens.

        Example:
            >>> import spacy
            >>> from spaczz.search import TokenSearcher
            >>> nlp = spacy.blank("en")
            >>> searcher = TokenSearcher(nlp)
            >>> doc = nlp("I was prescribed zithramax and advar")
            >>> pattern = [
                {"TEXT": {"FUZZY": "zithromax"}},
                {"POS": "CCONJ"},
                {"TEXT": {"FREGEX": "(advair){e<=1}"}}
                ]
            >>> searcher.match(doc, pattern)
            [[('TEXT', 'zithramax'), None, ('TEXT', 'advar')]]
        """
        if not isinstance(doc, Doc):
            raise TypeError("doc must be a Doc object.")
        if not isinstance(pattern, list):
            raise TypeError(
                "pattern must be a list", "Make sure pattern is wrapped in a list.",
            )
        if len(pattern) == 0:
            raise ValueError("pattern cannot have zero tokens.")
        matches = []
        for seq in n_wise(doc, len(pattern)):
            seq_matches = self._iter_pattern(seq, pattern, min_r, fuzzy_func)
            if seq_matches:
                matches.append(seq_matches)
        if matches:
            filtered_matches = [
                i for n, i in enumerate(matches) if i not in matches[:n]
            ]
            return filtered_matches
        else:
            return matches

    @staticmethod
    def regex_compare(text: str, pattern: str, ignore_case: bool = False) -> bool:
        """Performs fuzzy-regex supporting regex matching between two strings.

        Args:
            text: The string to match against.
            pattern: The regex pattern string.
            ignore_case: Whether to lower-case text
                before comparison or not. Default is `False`.

        Returns:
            `True` if match, `False` if not.

        Example:
            >>> import spacy
            >>> from spaczz.search import TokenSearcher
            >>> nlp = spacy.blank("en")
            >>> searcher = TokenSearcher(nlp)
            >>> searcher.regex_compare("sequel", "(sql){i<=3}")
            True
        """
        if ignore_case:
            text = text.lower()
        if regex.match(pattern, text):
            return True
        else:
            return False

    def _iter_pattern(
        self: TokenSearcher,
        seq: tuple[Token, ...],
        pattern: list[dict[str, Any]],
        min_r: int,
        fuzzy_func: str,
    ) -> list[Optional[tuple[str, str]]]:
        """Evaluates each token in a pattern against a doc token sequence."""
        seq_matches: list[Optional[tuple[str, str]]] = []
        for i, token in enumerate(pattern):
            pattern_dict, case, case_bool = self._parse_case(token)
            if isinstance(pattern_dict, dict):
                pattern_text, pattern_type = self._parse_type(pattern_dict)
                if pattern_text and pattern_type == "FUZZY":
                    if self.fuzzy_compare(
                        seq[i].text,
                        pattern_text,
                        case_bool,
                        pattern_dict.get("FUZZY_FUNC", fuzzy_func),
                    ) >= pattern_dict.get("MIN_R", min_r):
                        seq_matches.append((case, seq[i].text))
                    else:
                        return []
                elif pattern_text and pattern_type == "FREGEX":
                    if self.regex_compare(seq[i].text, pattern_text, case_bool):
                        seq_matches.append((case, seq[i].text))
                    else:
                        return []
                else:
                    seq_matches.append(None)
            else:
                seq_matches.append(None)
        return seq_matches

    @staticmethod
    def _parse_case(token: dict[str, Any]) -> tuple[Union[str, dict, None], str, bool]:
        """Parses the case of a token pattern."""
        if token.get("TEXT"):
            return token.get("TEXT"), "TEXT", False
        else:
            return token.get("LOWER"), "LOWER", True

    @staticmethod
    def _parse_type(pattern_dict: dict[str, Any]) -> tuple[str, str]:
        """Parses the type of a token pattern."""
        fuzzy_text = pattern_dict.get("FUZZY")
        regex_text = pattern_dict.get("FREGEX")
        if isinstance(fuzzy_text, str):
            return fuzzy_text, "FUZZY"
        elif isinstance(regex_text, str):
            return regex_text, "FREGEX"
        else:
            return "", ""
