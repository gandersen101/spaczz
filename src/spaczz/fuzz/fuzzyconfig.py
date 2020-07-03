"""Module for FuzzyConfig class."""
from typing import Callable, Dict
import warnings

from fuzzywuzzy import fuzz

from ..exceptions import CaseConflictWarning, EmptyConfigError


class FuzzyConfig:
    """Class for housing predefined fuzzy matching functions.

    Currently includes all fuzzywuzzy matchers with default settings.
    Will eventually includes methods for adding/removing user functions.

    Attributes:
        _fuzzy_funcs (Dict[str, Callable[[str, str], int]]):
            Fuzzy matching functions accessible
            by their given key name. All fuzzywuzzy matchers
            with default settings are currently available.
    """

    def __init__(self, empty: bool = False) -> None:
        """Initilaizes the fuzzy config.

        Args:
            empty: Whether to initialize the instance without predefined
                fuzzy matchers. Will be more useful later once API is extended.
                Default is False.
        """
        if not empty:
            self._fuzzy_funcs: Dict[str, Callable[[str, str], int]] = {
                "simple": fuzz.ratio,
                "partial": fuzz.partial_ratio,
                "token_set": fuzz.token_set_ratio,
                "token_sort": fuzz.token_sort_ratio,
                "partial_token_set": fuzz.partial_token_set_ratio,
                "partial_token_sort": fuzz.partial_token_sort_ratio,
                "quick": fuzz.QRatio,
                "u_quick": fuzz.UQRatio,
                "weighted": fuzz.WRatio,
                "u_weighted": fuzz.UWRatio,
            }
        else:
            self._fuzzy_funcs = {}

    def get_fuzzy_func(
        self, fuzzy_func: str, ignore_case: bool = True
    ) -> Callable[[str, str], int]:
        """Returns a fuzzy matching function based on it's key name.

        Args:
            fuzzy_func: Key name of the fuzzy matching function.
            ignore_case: If fuzzy matching will be executed
                with case sensitivity or not.

        Returns:
            A fuzzy matching function.

        Raises:
            EmptyConfigError: If the config has no fuzzy matchers available.
            ValueError: fuzzy_func was not a valid key name.

        Warnings:
            CaseConflictWarning:
                If the fuzzy matching function will automatically
                lower-case the input but case_sensitive is set to True.

        Example:
            >>> from spaczz.fuzz import FuzzyConfig
            >>> fc = FuzzyConfig()
            >>> simple = fc.get_fuzzy_func("simple", False)
            >>> simple("hi", "hi")
            100
        """
        if not ignore_case and fuzzy_func in [
            "token_sort",
            "token_set",
            "partial_token_set",
            "partial_token_sort",
            "quick",
            "u_quick",
            "weighted",
            "u_weighted",
        ]:
            warnings.warn(
                f"""{fuzzy_func} algorithm lower cases input by default.\n
                    This overrides case_sensitive setting.""",
                CaseConflictWarning,
            )
        if not self._fuzzy_funcs:
            raise EmptyConfigError(
                (
                    "The config has no fuzzy matchers available to it.",
                    "Please add any matching functions you intend to use",
                    "or do not initialize the config as empty.",
                )
            )
        try:
            return self._fuzzy_funcs[fuzzy_func]
        except KeyError:
            raise ValueError(
                (
                    f"No fuzzy matching function called {fuzzy_func}.",
                    "Matcher must be in the following:",
                    f"{list(self._fuzzy_funcs.keys())}",
                )
            )
