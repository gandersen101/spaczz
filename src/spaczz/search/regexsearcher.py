"""Module for RegexSearcher: multi-token regex matching in spaCy `Doc` objects."""
import typing as ty

import regex as re
from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.vocab import Vocab

from ..exceptions import RegexParseError
from ..registry import re_patterns
from ..registry import re_weights
from ..util import filter_overlapping_matches
from ..util import map_chars_to_tokens


class RegexSearcher:
    """Class for multi-token regex matching in spacy `Doc` objects.

    Regex matching is done on the character level and then
    mapped back to tokens.

    Attributes:
        vocab: The shared vocabulary.
            Included for consistency and potential future-state.
    """

    def __init__(
        self: "RegexSearcher",
        vocab: Vocab,
    ) -> None:
        """Initializes a regex searcher.

        Args:
            vocab: A spaCy Vocab.
                Purely for consistency between spaCy and spaczz matcher APIs for now.
                spaczz matchers are currently pure-Python and do not share vocabulary
                with spaCy pipelines.
        """
        self.vocab = vocab

    def match(
        self: "RegexSearcher",
        doc: Doc,
        query: str,
        partial: bool = True,
        predef: bool = False,
        min_r: int = 75,
        fuzzy_weights: str = "indel",
    ) -> ty.List[ty.Tuple[int, int, int, str]]:
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
                Can be overwritten with regex pattern options. Fuzzy regex patterns
                allow more fined-grained control so by default no min_r is set.
                Ratio results are more for informational, and `SpaczzRuler` sorting
                purposes. Default is `0`.
            fuzzy_weights: Placeholder.

        Returns:
            A list of tuples of match start indices, end indices, and match ratios.

        Example:
            >>> import spacy
            >>> from spaczz.search import RegexSearcher
            >>> nlp = spacy.blank("en")
            >>> searcher = RegexSearcher(nlp.vocab)
            >>> doc = nlp("My phone number is (555) 555-5555.")
            >>> searcher.match(doc, "phones", predef=True)
            [(4, 10, 100)]
        """
        compiled_regex = self._parse_regex(query, predef=predef)
        char_to_token_map = map_chars_to_tokens(doc)

        regex_matches = (
            _spans_from_regex(
                doc, match=match, partial=partial, char_to_token_map=char_to_token_map
            )
            for match in compiled_regex.finditer(doc.text)
        )

        formatted_matches = (
            (
                regex_match[0].start,
                regex_match[0].end,
                self._normalize_to_ratio(
                    regex_match[0].text,
                    fuzzy_counts=regex_match[1],
                    fuzzy_weights=fuzzy_weights,
                ),
                compiled_regex.pattern,
            )
            for regex_match in regex_matches
            if regex_match
        )

        return filter_overlapping_matches(
            sorted(
                (
                    formatted_match
                    for formatted_match in formatted_matches
                    if formatted_match[2] >= min_r
                ),
                key=lambda x: (-x[2], x[0]),
            )
        )

    @staticmethod
    def _parse_regex(
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
            return re_patterns.get(regex_str)
        try:
            return re.compile(
                regex_str,
            )
        except (re._regex_core.error, TypeError, ValueError) as e:
            raise RegexParseError(e)

    @staticmethod
    def _normalize_to_ratio(
        match: str, fuzzy_counts: ty.Tuple[int, int, int], fuzzy_weights: str
    ) -> int:
        """Normalizes fuzzy counts to a fuzzy ratio."""
        if fuzzy_counts == (0, 0, 0):
            return 100

        weights = re_weights.get(fuzzy_weights)
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


def _spans_from_regex(
    doc: Doc, match: re.Match[str], partial: bool, char_to_token_map: ty.Dict[int, int]
) -> ty.Optional[ty.Tuple[Span, ty.Tuple[int, int, int]]]:
    start, end = match.span()
    counts = getattr(match, "fuzzy_counts", (0, 0, 0))
    span = doc.char_span(start, end)
    if span:
        return (span, counts)
    if partial:
        start_token = char_to_token_map.get(start)
        end_token = char_to_token_map.get(end - 1)
        if start_token and end_token:
            span = Span(doc, start_token, end_token + 1)
            return (span, counts)
    return None
