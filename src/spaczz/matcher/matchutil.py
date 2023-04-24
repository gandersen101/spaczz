"""Module for match utilities."""
from spacy.tokens import Doc
from spacy.tokens import Span

from ..customtypes import MatchResult
from ..customtypes import MatchType


def make_span_from_match(doc: Doc, match: MatchResult, match_type: MatchType) -> Span:
    """Create a `Span` and update custom attributes from matches."""
    span = Span(doc, match[1], match[2], label=match[0])
    for token in span:
        token._.spaczz_token = True
        token._.spaczz_type = match_type
        token._.spaczz_ratio = match[3]
        token._.spaczz_pattern = match[4]
