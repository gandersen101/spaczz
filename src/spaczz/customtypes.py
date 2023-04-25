"""Custom spaczz types."""
import typing as ty

try:
    from typing import Literal  # type: ignore
except ImportError:  # pragma: no cover
    from typing_extensions import Literal  # type: ignore

from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.tokens import Token

DocLike = ty.Union[Doc, Span]
FlexLiteral = Literal["default", "min", "max"]
FlexType = ty.Union[int, FlexLiteral]  # type: ignore
TextContainer = ty.Union[Doc, Span, Token]
SearchResult = ty.Tuple[int, int, int]
MatchResult = ty.Tuple[str, int, int, int, str]
MatchType = Literal["fuzzy", "regex", "token", "similarity", "_phrase"]
