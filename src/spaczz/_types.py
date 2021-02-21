"""Module containing custom types for spaczz."""
from __future__ import annotations

from collections import defaultdict
from functools import partial
from itertools import repeat
from typing import Any, Callable, DefaultDict, List, Optional, Tuple, Union

from spacy.tokens import Doc, Span


DocLike = Union[Doc, Span]
PhraseCallback = Optional[
    Callable[[Any, Doc, int, List[Tuple[str, int, int, int]]], None]
]
RegexCallback = Optional[
    Callable[[Any, Doc, int, List[Tuple[str, int, int, Tuple[int, int, int]]]], None]
]
TokenCallback = Optional[
    Callable[[Any, Doc, int, List[Tuple[str, int, int, None]]], None]
]
MatcherCallback = Union[PhraseCallback, RegexCallback, TokenCallback]


def nest_defaultdict(default_factory: Any, depth: int = 1) -> DefaultDict[Any, Any]:
    """Returns a nested `defaultdict`."""
    result = partial(defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(defaultdict, result)
    return result()
