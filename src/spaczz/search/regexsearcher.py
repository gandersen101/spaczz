"""Module for RegexSearcher: multi-token regex matching in spaCy `Doc` objects."""
from __future__ import annotations

from typing import Any, Dict, List, Pattern, Tuple, Union

import regex as re
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab

from .._commonregex import get_common_regex
from ..exceptions import RegexParseError
from ..util import map_chars_to_tokens

WEIGHTS = (1, 1, 2)


class RegexSearcher:
    """Class for multi-token regex matching in spacy `Doc` objects.

    Regex matching is done on the character level and then
    mapped back to tokens.

    Attributes:
        vocab: The shared vocabulary.
            Included for consistency and potential future-state.
    """

    def __init__(
        self: RegexSearcher,
        vocab: Vocab,
        predefs: Union[str, Dict[str, Union[str, Pattern]]] = "default",
    ) -> None:
        """Initializes a regex searcher.

        Args:
            vocab: A spaCy Vocab.
                Purely for consistency between spaCy and spaczz matcher APIs for now.
                spaczz matchers are currently pure-Python and do not share vocabulary
                with spaCy pipelines.
            predefs: Predefined regex patterns.
                Uses the default predef patterns if "default", none if "empty",
                or a custom mapping of names to regex pattern strings.
                Default is `"default"`.

        Raises:
            ValueError: If predef patterns are not compiled,
                or cannot be compiled, to regex patterns.
                Or a unknown string value was provided.
        """
        self.vocab = vocab
        if isinstance(predefs, str):
            if predefs == "default":
                self._predefs = get_common_regex()
            elif predefs == "empty":
                self._predefs = {}
            else:
                raise ValueError(
                    "If `predefs` is a string value it must be one of "
                    "either `'default'` or `'empty'`."
                )
        else:
            try:
                predefs_ = {}
                for k, v in predefs.items():
                    if isinstance(v, str):
                        predefs_[k] = re.compile(v)
                    elif hasattr(v, "pattern"):
                        predefs_[k] = v
                    else:
                        raise ValueError(
                            "All predef regex patterns must either be compilable or "
                            "pre-compiled as regex patterns."
                            f"First problem pattern: {v}"
                        )
                self._predefs = predefs_
            except (re._regex_core.error, TypeError, ValueError) as e:
                raise ValueError(
                    "All predef regex patterns must either be compilable or "
                    "pre-compiled as regex patterns."
                    f"First problem pattern: {v}\n"
                    f"Due to: {e}"
                )

    @property
    def predefs(self: RegexSearcher) -> Dict[str, Pattern]:
        """Getter for predefined regex patterns."""
        return self._predefs.copy()

    def match(
        self: RegexSearcher,
        doc: Doc,
        query: str,
        partial: bool = True,
        predef: bool = False,
        min_r: int = 75,
    ) -> List[Tuple[int, int, int]]:
        """Returns regex matches in a `Doc` object.

        Matches on the character level and then maps matches back
        to tokens. If a character cannot be mapped back to a token it means
        the character is a space tokens are split on, which happens when regex
        matches produce leading or trailing whitespace. Confirm your regex pattern
        will not do this to avoid this issue.

        To utilize regex flags, use inline flags.

        Args:
            doc: Doc to search over.
            query: A string to be compiled to regex,
                or the key name of a predefined regex pattern.
            partial: Whether partial matches should be extended
                to existing span boundaries in doc or not, e.x.
                the regex only matches part of a token or span.
                Default is `True`.
            predef: Whether regex should be interpreted as a key to
                a predefined regex pattern or not. Default is `False`.
                The included regexes are:
                `"dates"`
                `"times"`
                `"phones"`
                `"phones_with_exts"`
                `"links"`
                `"emails"`
                `"ips"`
                `"ipv6s"`
                `"prices"`
                `"hex_colors"`
                `"credit_cards"`
                `"btc_addresses"`
                `"street_addresses"`
                `"zip_codes"`
                `"po_boxes"`
                `"ssn_number"`.
            min_r: Minimum match ratio required for fuzzy matching.
                Can be overwritten with regex pattern options.
                Default is `75`.

        Returns:
            A list of tuples of match start indices, end indices, and match ratios.

        Raises:
            TypeError: If `doc` is not a `Doc`
            TypeError: If `query` is not a `str`.

        Example:
            >>> import spacy
            >>> from spaczz.search import RegexSearcher
            >>> nlp = spacy.blank("en")
            >>> searcher = RegexSearcher(nlp.vocab)
            >>> doc = nlp("My phone number is (555) 555-5555.")
            >>> searcher.match(doc, "phones", predef=True)
            [(4, 10, 100)]
        """
        if not isinstance(doc, Doc):
            raise TypeError("doc must be a Doc object.")
        if isinstance(query, str):
            compiled_regex = self.parse_regex(query, predef)
        else:
            raise TypeError(f"query must be a str, not {type(query)}.")
        matches = []
        chars_to_tokens = map_chars_to_tokens(doc)
        for match in compiled_regex.finditer(doc.text):
            start, end = match.span()
            counts = getattr(match, "fuzzy_counts", (0, 0, 0))
            span = doc.char_span(start, end)
            if span:
                matches.append((span, counts))
            else:
                if partial:
                    start_token = chars_to_tokens.get(start)
                    end_token = chars_to_tokens.get(end - 1)
                    if start_token and end_token:
                        span = Span(doc, start_token, end_token + 1)
                        matches.append((span, counts))
        if matches:
            return [
                (
                    match[0].start,
                    match[0].end,
                    self._normalize_to_ratio(match[0].text, match[1]),
                )
                for match in matches
            ]
        else:
            return []

    def parse_regex(
        self: RegexSearcher,
        regex_str: str,
        predef: bool = False,
    ) -> Any:
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
            >>> import regex as re
            >>> import spacy
            >>> from spaczz.search import RegexSearcher
            >>> nlp = spacy.blank("en")
            >>> searcher = RegexSearcher(nlp.vocab)
            >>> pattern = searcher.parse_regex("Test")
            >>> isinstance(pattern, re.Pattern)
            True
        """
        if predef:
            compiled_regex = self.get_predef(regex_str)
        else:
            try:
                compiled_regex = re.compile(
                    regex_str,
                )
            except (re._regex_core.error, TypeError, ValueError) as e:
                raise RegexParseError(e)
        return compiled_regex

    def get_predef(self: RegexSearcher, predef: str) -> Any:
        """Returns a regex pattern from the predefined patterns available.

        Args:
            predef: The key name of a predefined regex pattern.

        Returns:
            A compiled regex pattern.

        Raises:
            ValueError: If the key does not exist in the predefined regex patterns.

        Example:
            >>> import regex as re
            >>> import spacy
            >>> from spaczz.search import RegexSearcher
            >>> nlp = spacy.blank("en")
            >>> searcher = RegexSearcher(nlp.vocab)
            >>> pattern = searcher.get_predef("phones")
            >>> isinstance(pattern, re.Pattern)
            True
        """
        predef_regex = self._predefs.get(predef)
        if predef_regex:
            return predef_regex
        else:
            raise ValueError(f"{predef} is not a regex pattern defined in `predefs`.")

    @staticmethod
    def _normalize_to_ratio(match: str, fuzzy_counts: Tuple[int, int, int]) -> int:
        """Normalizes fuzzy counts to a fuzzy ratio."""
        if fuzzy_counts == (0, 0, 0):
            return 100
        s1_len = len(match) + fuzzy_counts[0] - fuzzy_counts[1]
        s2_len = len(match)
        fuzzy_counts_ = tuple(a * b for a, b in zip(fuzzy_counts, WEIGHTS))
        if WEIGHTS[2] <= WEIGHTS[0] + WEIGHTS[1]:
            dist_max = min(s1_len, s2_len) * WEIGHTS[2]
        else:
            dist_max = s1_len * WEIGHTS[1] + s2_len * WEIGHTS[0]
        if s1_len > s2_len:
            dist_max += (s1_len - s2_len) * WEIGHTS[1]
        elif s1_len < s2_len:
            dist_max += (s2_len - s1_len) * WEIGHTS[0]
        r = 100 * sum(fuzzy_counts_) / dist_max
        return round(100 - r)
