"""Module for FuzzyConfig class."""
import operator
from typing import Callable, Iterable, Optional, Set, Tuple
import warnings

from fuzzywuzzy import fuzz
from spacy.tokens import Token


class FuzzyConfig:
    """Class for housing predefined fuzzy matching and span trimming functions.

    Will eventually includes methods for adding/removing user matchers/trimmers.

    Attributes:
        _fuzzy_funcs (Dict[str, Callable[[str, str], int]]):
            Fuzzy matching functions accessible
            by their given key name. All FuzzyWuzzy.fuzz
            functions with default settings are currently
            available.
        _span_trimmers (Dict[str, Callable[[Token], bool]]):
            Span boundary trimming functions
            accessible by their given key name.
            These prevent start and/or end boundaries
            of fuzzy match spans from containing
            unwanted tokens like punctuation.
            Functions for punctuations, spaces,
            and stop words are currently available.
    """

    def __init__(self, empty: bool = False) -> None:
        """Initilaizes the fuzzy config.

        Args:
            empty: Whether to initialize the instance without predefined
                fuzzy matchers and trimmers or not.
                Will be more useful later once API is extended.
                Default is False.
        """
        if not empty:
            self._fuzzy_funcs = {
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
            self._span_trimmers = {
                "space": lambda x: operator.truth(x.is_space),
                "punct": lambda x: operator.truth(x.is_punct),
                "stop": lambda x: operator.truth(x.is_stop),
            }
        else:
            self._fuzzy_funcs = {}
            self._span_trimmers = {}

    def get_fuzzy_func(
        self, fuzzy_func: str, ignore_case: bool
    ) -> Callable[[str, str], int]:
        """Returns a fuzzy matching function based on it's key name.

        Args:
            fuzzy_func: Key name of the fuzzy matching algorithm.
            ignore_case: If fuzzy matching will be executed
                with case sensitivity or not.

        Returns:
            A fuzzy matching function.

        Raises:
            ValueError: fuzzy_func was not a valid key name.

        Warnings:
            UserWarning:
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
                (
                    f"{fuzzy_func} algorithm lower cases input by default.",
                    "This overrides case_sensitive setting.",
                )
            )
        try:
            return self._fuzzy_funcs[fuzzy_func]
        except KeyError:
            raise ValueError(
                (
                    f"No fuzzy matching algorithm called {fuzzy_func}.",
                    "Algorithm must be in the following:",
                    f"{list(self._fuzzy_funcs.keys())}",
                )
            )

    def get_trimmers(
        self,
        side: str,
        trimmers: Optional[Iterable[str]] = None,
        start_trimmers: Optional[Iterable[str]] = None,
        end_trimmers: Optional[Iterable[str]] = None,
    ) -> Tuple[Set[Callable[[Token], bool]], Set[str]]:
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
            >>> fc = FuzzyConfig()
            >>> trimmers = fc.get_trimmers("end", ["space"], end_trimmers=["punct"])
            >>> trimmers == (
                {fc._span_trimmers["space"], fc._span_trimmers["punct"]},
                {"space", "punct"}
                )
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
        return trimmer_funcs, trimmer_keys
