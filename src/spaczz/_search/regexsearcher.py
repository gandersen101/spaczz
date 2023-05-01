"""`RegexSearcher` searches for phrase-based regex matches in spaCy `Doc` objects."""
import typing as ty

try:
    from typing import Match
except ImportError:  # pragma: no cover
    from regex import Match  # type: ignore

from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.vocab import Vocab

from .searchutil import filter_overlapping_matches
from .searchutil import normalize_fuzzy_regex_counts
from .searchutil import parse_regex
from ..customtypes import SearchResult


class RegexSearcher:
    """Class for phrase-based regex match searching in spaCy `Doc` objects."""

    def __init__(
        self: "RegexSearcher",
        vocab: Vocab,
    ) -> None:
        """Initializes the searcher."""
        self.vocab = vocab

    def match(
        self: "RegexSearcher",
        doc: Doc,
        query: str,
        *,
        ignore_case: bool = True,
        min_r: int = 75,
        partial: bool = True,
        predef: bool = False,
        fuzzy_weights: str = "indel",
    ) -> ty.List[SearchResult]:
        """Performs regex matching on a `Doc` object.

        Matches on the character level and then maps matches back
        to tokens. If a character cannot be mapped back to a token it means
        the character is a space tokens are split on, which happens when regex
        matches produce leading or trailing whitespace. Confirm your regex pattern
        will not do this to avoid this issue.

        To utilize regex flags, use inline flags.

        Args:
            doc: `Doc` to search for matches.
            query: A string to be compiled to regex, or the key name of a predefined
                regex pattern.
            ignore_case: Whether to lower-case text before matching.
                Default is `True`.
            min_r: Minimum match ratio required for fuzzy regex matching.
                Default is `75`.
            fuzzy_weights: Name of weighting method for regex insertion, deletion, and
                substituion counts. Default is `"indel"`.
            partial: Whether partial matches should be extended
                to existing span boundaries in doc or not, e.x. the regex only matches
                part of a `Token` or `Span`. Default is `True`.
            predef: Whether regex should be interpreted as a key to
                a predefined regex pattern or not. Default is `False`.

        Returns:
            List of match tuples each containing a start index, end index,
            and match ratio.

        Example:
            >>> import spacy
            >>> from spaczz._search import RegexSearcher
            >>> nlp = spacy.blank("en")
            >>> searcher = RegexSearcher(nlp.vocab)
            >>> doc = nlp("My phone number is (555) 555-5555.")
            >>> searcher.match(doc, "phones", predef=True)
            [(4, 10, 100)]
        """
        compiled_regex = parse_regex(query, predef=predef)
        char_to_token_map = self._map_chars_to_tokens(doc)

        regex_matches = [
            self._spans_from_regex(
                doc, match=match, partial=partial, char_to_token_map=char_to_token_map
            )
            for match in compiled_regex.finditer(
                doc.text.lower() if ignore_case else doc.text
            )
        ]

        formatted_matches = [
            (
                regex_match[0].start,
                regex_match[0].end,
                normalize_fuzzy_regex_counts(
                    regex_match[0].text,
                    fuzzy_counts=regex_match[1],
                    fuzzy_weights=fuzzy_weights,
                ),
            )
            for regex_match in regex_matches
            if regex_match
        ]

        return filter_overlapping_matches(
            sorted(
                [
                    formatted_match
                    for formatted_match in formatted_matches
                    if formatted_match[2] >= min_r
                ],
                key=lambda x: (-x[2], x[0]),
            )
        )

    @staticmethod
    def _map_chars_to_tokens(doc: Doc) -> ty.Dict[int, int]:
        """Maps characters to tokens."""
        chars_to_tokens = {}
        for token in doc:
            for i in range(token.idx, token.idx + len(token.text)):
                chars_to_tokens[i] = token.i
        return chars_to_tokens

    @staticmethod
    def _spans_from_regex(
        doc: Doc,
        match: Match[str],
        partial: bool,
        char_to_token_map: ty.Dict[int, int],
    ) -> ty.Optional[ty.Tuple[Span, ty.Tuple[int, int, int]]]:
        """Creates `Span` objects from regex matches."""
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
