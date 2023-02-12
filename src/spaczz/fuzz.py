"""Module for fuzzy matching functions."""
import typing as ty

import catalogue
from rapidfuzz import fuzz as rfuzz

fuzzy_funcs = catalogue.create("spaczz", "fuzz", entry_points=True)
fuzzy_funcs.register("simple", func=rfuzz.ratio)
fuzzy_funcs.register("partial", func=rfuzz.partial_ratio)
fuzzy_funcs.register("patial", func=rfuzz.partial_ratio_alignment)
fuzzy_funcs.register("token", func=rfuzz.token_ratio)
fuzzy_funcs.register("token_set", func=rfuzz.token_set_ratio)
fuzzy_funcs.register("token_sort", func=rfuzz.token_sort_ratio)
fuzzy_funcs.register("partial_token", func=rfuzz.partial_token_ratio)
fuzzy_funcs.register("partial_token_set", func=rfuzz.partial_token_set_ratio)
fuzzy_funcs.register("partial_token_sort", func=rfuzz.partial_token_sort_ratio)
fuzzy_funcs.register("weighted", func=rfuzz.WRatio)
fuzzy_funcs.register("quick", func=rfuzz.QRatio)


class FuzzyFuncs:
    """Container class housing fuzzy matching functions.

    Functions are accessible via the classes `` method
    by their given key name. All rapidfuzz matching functions
    with default settings are available.

    Attributes:
        match_type (str): Whether the fuzzy matching functions
            should support multi-token strings ("phrase") or
            only single-token strings ("token").
        fuzzy_funcs (dict[str, Callable[[str, str], int]]):
            The available fuzzy matching functions:
            "simple" = `ratio`
            "partial" = `partial_ratio`
            "token_set" = `token_set_ratio`
            "token_sort" = `token_sort_ratio`
            "token" = `token_ratio`
            "partial_token_set" = `partial_token_set_ratio`
            "partial_token_sort" = `partial_token_sort_ratio`
            "partial_token" = `partial_token_ratio`
            "weighted" = `WRatio`
            "quick" = `QRatio`
            This is limited to "simple" or "quick"
            if match_type = "token".
    """

    def __init__(self: "FuzzyFuncs", match_type: str = "phrase") -> None:
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
            self.fuzzy_funcs: ty.Dict[str, ty.Callable] = {
                "simple": rfuzz.ratio,
                "partial": rfuzz.partial_ratio,
                "token_set": rfuzz.token_set_ratio,
                "token_sort": rfuzz.token_sort_ratio,
                "token": rfuzz.token_ratio,
                "partial_token_set": rfuzz.partial_token_set_ratio,
                "partial_token_sort": rfuzz.partial_token_sort_ratio,
                "partial_token": rfuzz.partial_token_ratio,
                "weighted": rfuzz.WRatio,
                "quick": rfuzz.QRatio,
            }
        elif match_type == "token":
            self.fuzzy_funcs = {
                "simple": rfuzz.ratio,
                "quick": rfuzz.QRatio,
            }
        else:
            raise ValueError("match_type must be either 'phrase' or 'token'.")

    def get(self: "FuzzyFuncs", fuzzy_func: str) -> ty.Callable[..., float]:
        """Returns a fuzzy matching function based on it's key name.

        Args:
            fuzzy_func: Key name of the fuzzy matching function.

        Returns:
            A fuzzy matching function.

        Raises:
            ValueError: The fuzzy function was not a valid key name.

        Example:
            >>> import spacy
            >>> from spaczz.fuzz import FuzzyFuncs
            >>> ff = FuzzyFuncs()
            >>> simple = ff.get("simple")
            >>> simple("hi", "hi")
            100.0
        """
        try:
            return self.fuzzy_funcs[fuzzy_func.lower()]
        except KeyError:
            raise ValueError(
                (
                    f"No fuzzy {self.match_type} matching function",
                    f"called: {fuzzy_func}.",
                    "Matching function must be in the following (case insensitive):",
                    f"{list(self.fuzzy_funcs.keys())}",
                )
            )
