"""Module for FuzzySearcher: fuzzy matching in spaCy `Doc` objects."""
from typing import Any, Callable, Dict, Union

from rapidfuzz import fuzz
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab

from . import _PhraseSearcher


class FuzzySearcher(_PhraseSearcher):
    """Class for fuzzy searching/matching in spacy `Doc` objects.

    Fuzzy searching is done on the token level.
    The class provides methods for finding the best fuzzy match
    span in a `Doc`, the n best fuzzy matched spans in a `Doc`,
    and fuzzy matching between any two given spaCy containers
    (`Doc`, `Span`, `Token`).

    Fuzzy matching is currently provided by rapidfuzz and the searcher
    contains provides access to all rapidfuzz matchers with default
    settings.

    Attributes:
        vocab (Vocab): The shared vocabulary.
            Included for consistency and potential future-state.
        _fuzzy_funcs (Dict[str, Callable[[str, str], int]]):
            Fuzzy matching functions accessible
            by their given key name. All rapidfuzz matchers
            with default settings are currently available:
            "simple" = ratio
            "partial" = partial_ratio
            "token_set" = token_set_ratio
            "token_sort" = token_sort_ratio
            "partial_token_set" = partial_token_set_ratio
            "partial_token_sort" = partial_token_sort_ratio
            "quick" = QRatio
            "weighted" = WRatio
            "quick_lev" = quick_lev_ratio
    """

    def __init__(self, vocab: Vocab) -> None:
        """Initializes a fuzzy searcher.

        Args:
            vocab: A spaCy `Vocab` object.
                Purely for consistency between spaCy
                and spaczz matcher APIs for now.
                spaczz matchers are mostly pure-Python
                currently and do not share vocabulary
                with spaCy pipelines.
        """
        super().__init__(vocab)
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

    def compare(
        self,
        a: Union[Doc, Span, Token],
        b: Union[Doc, Span, Token],
        ignore_case: bool = True,
        fuzzy_func: str = "simple",
        *args: Any,
        **kwargs: Any,
    ) -> int:
        """Peforms fuzzy matching between two strings.

        Applies the given fuzzy matching algorithm (fuzzy_func)
        to two string from spacy containers (`Doc`, `Span`, `Token`)
        and returns the resulting fuzzy ratio.

        Args:
            a: First container for comparison.
            b: Second container for comparison.
            ignore_case: Whether to lower-case a and b
                before comparison or not. Default is True.
            fuzzy_func: Key name of fuzzy matching function to use.
                All rapidfuzz matching functions with default settings
                are available:
                "simple" = fuzz.ratio
                "partial" = fuzz.partial_ratio
                "token_set" = fuzz.token_set_ratio
                "token_sort" = fuzz.token_sort_ratio
                "partial_token_set" = fuzz.partial_token_set_ratio
                "partial_token_sort" = fuzz.partial_token_sort_ratio
                "quick" = fuzz.QRatio
                "weighted" = fuzz.WRatio
                "quick_lev" = fuzz.quick_lev_ratio
                Default is "simple".
            *args: Overflow for child positional arguments.
            **kwargs: Overflow for child keyword arguments.

        Returns:
            The fuzzy ratio between a and b.

        Example:
            >>> import spacy
            >>> from spaczz.search import FuzzySearcher
            >>> nlp = spacy.blank("en")
            >>> searcher = FuzzySearcher()
            >>> searcher.compare(nlp("spaczz"), nlp("spacy"))
            73
        """
        if ignore_case:
            a_text = a.text.lower()
            b_text = b.text.lower()
        else:
            a_text = a.text
            b_text = b.text
        return round(self.get_fuzzy_func(fuzzy_func)(a_text, b_text))

    def get_fuzzy_func(self, fuzzy_func: str) -> Callable[[str, str], int]:
        """Returns a fuzzy matching function based on it's key name.

        Args:
            fuzzy_func: Key name of the fuzzy matching function.

        Returns:
            A fuzzy matching function.

        Raises:
            ValueError: fuzzy_func was not a valid key name.

        Example:
            >>> from spaczz.search import FuzzySearcher
            >>> searcher = FuzzySearcher()
            >>> simple = searcher.get_fuzzy_func("simple", False)
            >>> simple("hi", "hi")
            100.0
        """
        try:
            return self._fuzzy_funcs[fuzzy_func]
        except KeyError:
            raise ValueError(
                (
                    f"No fuzzy matching function called {fuzzy_func}.",
                    "Matching function must be in the following:",
                    f"{list(self._fuzzy_funcs.keys())}",
                )
            )
