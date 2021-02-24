"""Module for various doc/text processing classes/functions."""
from __future__ import annotations

from typing import Callable

from rapidfuzz import fuzz
from spacy.tokens import Doc


def map_chars_to_tokens(doc: Doc) -> dict[int, int]:
    """Maps characters in a `Doc` object to tokens."""
    chars_to_tokens = {}
    for token in doc:
        for i in range(token.idx, token.idx + len(token.text)):
            chars_to_tokens[i] = token.i
    return chars_to_tokens


class FuzzyFuncs:
    """Container class housing fuzzy matching functions.

    Functions are accessible via the classes `get()` method
    by their given key name. All rapidfuzz matching functions
    with default settings are available.

    Attributes:
        match_type (str): Whether the fuzzy matching functions
            should support multi-token strings ("phrase") or
            only single-token strings ("token").
        _fuzzy_funcs (dict[str, Callable[[str, str], int]]):
            The available fuzzy matching functions:
            "simple" = `ratio`
            "partial" = `partial_ratio`
            "token_set" = `token_set_ratio`
            "token_sort" = `token_sort_ratio`
            "partial_token_set" = `partial_token_set_ratio`
            "partial_token_sort" = `partial_token_sort_ratio`
            "quick" = `QRatio`
            "weighted" = `WRatio`
            "quick_lev" = `quick_lev_ratio`
            This is limited to "simple", "quick", and "quick_lev"
            if match_type = "token".
    """

    def __init__(self: FuzzyFuncs, match_type: str = "phrase") -> None:
        """Initializes a `FuzzyFuncs` container.

        Args:
            match_type: Whether the fuzzy matching functions
                support multi-token strings ("phrase") or
                only single-token strings ("token").

        Raises:
            ValueError: If match_type is not "phrase" or "token".
        """
        self.match_type = match_type
        if match_type == "phrase":
            self._fuzzy_funcs: dict[str, Callable[[str, str], int]] = {
                "simple": fuzz.ratio,
                "partial": fuzz.partial_ratio,
                "token_set": fuzz.token_set_ratio,
                "token_sort": fuzz.token_sort_ratio,
                "partial_token_set": fuzz.partial_token_set_ratio,
                "partial_token_sort": fuzz.partial_token_sort_ratio,
                "quick": fuzz.QRatio,
                "weighted": fuzz.WRatio,
                "token": fuzz.token_ratio,
                "partial_token": fuzz.partial_token_ratio,
            }
        elif match_type == "token":
            self._fuzzy_funcs = {
                "simple": fuzz.ratio,
                "quick": fuzz.QRatio,
            }
        else:
            raise ValueError("match_type must be either 'phrase' or 'token'.")

    def get(self: FuzzyFuncs, fuzzy_func: str) -> Callable[[str, str], float]:
        """Returns a fuzzy matching function based on it's key name.

        Args:
            fuzzy_func: Key name of the fuzzy matching function.

        Returns:
            A fuzzy matching function.

        Raises:
            ValueError: The fuzzy function was not a valid key name.

        Example:
            >>> import spacy
            >>> from spaczz.process import FuzzyFuncs
            >>> ff = FuzzyFuncs()
            >>> simple = ff.get("simple")
            >>> simple("hi", "hi")
            100.0
        """
        try:
            return self._fuzzy_funcs[fuzzy_func.lower()]
        except KeyError:
            raise ValueError(
                (
                    f"No fuzzy {self.match_type} matching function",
                    f"called: {fuzzy_func}.",
                    "Matching function must be in the following (case insensitive):",
                    f"{list(self._fuzzy_funcs.keys())}",
                )
            )
