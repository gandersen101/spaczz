"""Custom spaczz types."""
import typing as ty

try:
    from typing import Literal
except ImportError:  # pragma: no cover
    from typing_extensions import Literal

from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.tokens import Token

DocLike = ty.Union[Doc, Span]
FlexType = ty.Union[int, Literal["default", "min", "max"]]
TextContainer = ty.Union[Doc, Span, Token]
SearchResult = ty.Tuple[int, int, int, str]
MatchResult = ty.Tuple[str, int, int, int, str]
MatchType = ty.Literal["fuzzy", "regex", "token", "similarity"]
