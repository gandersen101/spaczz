"""Module for FuzzyConfig class."""
from typing import Callable, Dict, Iterable, Optional, Set

from rapidfuzz import fuzz
from spacy.tokens import Token

from ..exceptions import EmptyConfigError


class FuzzyConfig:
    """Class for housing predefined fuzzy matching and span trimming functions.

    Currently includes all rapidfuzz matchers with default settings.
    Will eventually includes methods for adding/removing user matchers/trimmers.

    Attributes:
        _fuzzy_funcs (Dict[str, Callable[[str, str], int]]):
            Fuzzy matching functions accessible
            by their given key name. All rapidfuzz matchers
            with default settings are currently available:
            "simple" = fuzz.ratio
            "partial" = fuzz.partial_ratio
            "token_set" = fuzz.token_set_ratio
            "token_sort" = fuzz.token_sort_ratio
            "partial_token_set" = fuzz.partial_token_set_ratio
            "partial_token_sort" = fuzz.partial_token_sort_ratio
            "quick" = fuzz.QRatio
            "weighted" = fuzz.WRatio
            "quick_lev" = fuzz.quick_lev_ratio
        _span_trimmers (Dict[str, Callable[[Token], bool]]):
            Span boundary trimming functions
            accessible by their given key name.
            These prevent start and/or end boundaries
            of fuzzy match spans from containing
            unwanted tokens like punctuation.
            Functions for punctuation, whitespace,
            and stop words are currently available as:
            "space", "punct", and "stop".
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
                "weighted": fuzz.WRatio,
                "quick_lev": fuzz.quick_lev_ratio,
            }
            self._span_trimmers: Dict[str, Callable[[Token], bool]] = {
                "space": lambda x: x.is_space,
                "punct": lambda x: x.is_punct,
                "stop": lambda x: x.is_stop,
            }
        else:
            self._fuzzy_funcs = {}
            self._span_trimmers = {}

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

        Example:
            >>> from spaczz.fuzz import FuzzyConfig
            >>> config = FuzzyConfig()
            >>> simple = config.get_fuzzy_func("simple", False)
            >>> simple("hi", "hi")
            100.0
        """
        if not self._fuzzy_funcs:
            raise EmptyConfigError(
                (
                    "The config has no fuzzy matchers available to it.",
                    "Please add any fuzzy matching functions you intend to use",
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

    def get_trimmers(
        self,
        side: str,
        trimmers: Optional[Iterable[str]] = None,
        start_trimmers: Optional[Iterable[str]] = None,
        end_trimmers: Optional[Iterable[str]] = None,
    ) -> Set[Callable[[Token], bool]]:
        """Gets trimmer rule functions by their key names.

        Returns start and end trimmers depending on side.
        Trimmers are direction agnostic - apply to both sides of match.

        Args:
            side: Whether to populate start or end.
            trimmers: Optional iterable of direction agnostic
                span trimmer key names.
            start_trimmers: Optional iterable of start index
                span trimmer key names.
            end_trimmers: Optional iterable of end index
                span trimmer key names.

        Returns:
            A set of functions and a set of their key names as a tuple.

        Raises:
            ValueError: One or more trimmer keys was not a valid trimmer rule.

        Example:
            >>> from spaczz.fuzz import FuzzyConfig
            >>> config = FuzzyConfig()
            >>> trimmers = config.get_trimmers("end", ["space"], end_trimmers=["punct"])
            >>> trimmers == {config._span_trimmers["space"],
                config._span_trimmers["punct"]}
            True

        """
        trimmer_funcs = set()
        trimmer_keys = set()
        if trimmers:
            for key in trimmers:
                trimmer_keys.add(key)
        if side == "start":
            if start_trimmers:
                for key in start_trimmers:
                    trimmer_keys.add(key)
        else:
            if end_trimmers:
                for key in end_trimmers:
                    trimmer_keys.add(key)
        if trimmer_keys:
            for key in trimmer_keys:
                try:
                    trimmer_funcs.add(self._span_trimmers[key])
                except KeyError:
                    raise ValueError(
                        (
                            f"No trimmer rule called {key}.",
                            "Trimmer rule must be in the following:",
                            f"{list(self._span_trimmers.keys())}",
                        )
                    )
        return trimmer_funcs
