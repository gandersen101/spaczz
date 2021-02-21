"""Module for FuzzySearcher: fuzzy matching in spaCy `Doc` objects."""
from __future__ import annotations

from typing import Any, Union

from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab

from . import _PhraseSearcher
from ..process import FuzzyFuncs


class FuzzySearcher(_PhraseSearcher):
    """Class for fuzzy searching/matching in spacy `Doc` objects.

    Fuzzy searching is done on the token level.
    The class provides methods for finding the best fuzzy match
    span in a `Doc`, the n best fuzzy matched spans in a `Doc`,
    and fuzzy matching between any two given spaCy containers
    (`Doc`, `Span`, `Token`).

    Fuzzy matching is currently provided by rapidfuzz and the searcher
    provides access to all rapidfuzz matchers with default settings.

    Attributes:
        vocab (Vocab): The shared vocabulary.
            Included for consistency and potential future-state.
        _fuzzy_funcs (FuzzyFuncs):
            Container class housing fuzzy matching functions.
            Functions are accessible via the classes `get()` method
            by their given key name. All rapidfuzz matching functions
            with default settings are available:
            "simple" = `ratio`
            "partial" = `partial_ratio`
            "token_set" = `token_set_ratio`
            "token_sort" = `token_sort_ratio`
            "partial_token_set" = `partial_token_set_ratio`
            "partial_token_sort" = `partial_token_sort_ratio`
            "quick" = `QRatio`
            "weighted" = `WRatio`
            "quick_lev" = `quick_lev_ratio`
    """

    def __init__(self: FuzzySearcher, vocab: Vocab) -> None:
        """Initializes a fuzzy searcher.

        Args:
            vocab: A spaCy `Vocab` object.
                Purely for consistency between spaCy
                and spaczz matcher APIs for now.
                spaczz matchers are mostly pure-Python
                currently and do not share vocabulary
                with spaCy pipelines.
        """
        super().__init__(vocab=vocab)
        self._fuzzy_funcs: FuzzyFuncs = FuzzyFuncs(match_type="phrase")

    def compare(
        self: FuzzySearcher,
        query: Union[Doc, Span, Token],
        reference: Union[Doc, Span, Token],
        ignore_case: bool = True,
        fuzzy_func: str = "simple",
        *args: Any,
        **kwargs: Any,
    ) -> int:
        """Peforms fuzzy matching between two spaCy container objects.

        Applies the given fuzzy matching algorithm (`fuzzy_func`)
        to two spacy containers (`Doc`, `Span`, `Token`)
        and returns the resulting fuzzy ratio.

        Args:
            query: First container for comparison.
            reference: Second container for comparison.
            ignore_case: Whether to lower-case `query` and `reference`
                before comparison or not. Default is `True`.
            fuzzy_func: Key name of fuzzy matching function to use.
                All rapidfuzz matching functions with default settings
                are available:
                "simple" = `ratio`
                "partial" = `partial_ratio`
                "token_set" = `token_set_ratio`
                "token_sort" = `token_sort_ratio`
                "partial_token_set" = `partial_token_set_ratio`
                "partial_token_sort" = `partial_token_sort_ratio`
                "quick" = `QRatio`
                "weighted" = `WRatio`
                "quick_lev" = `quick_lev_ratio`
                Default is `"simple"`.
            *args: Overflow for child positional arguments.
            **kwargs: Overflow for child keyword arguments.

        Returns:
            The fuzzy ratio between `query` and `reference` as an `int`.

        Example:
            >>> import spacy
            >>> from spaczz.search import FuzzySearcher
            >>> nlp = spacy.blank("en")
            >>> searcher = FuzzySearcher(nlp.vocab)
            >>> searcher.compare(nlp("spaczz"), nlp("spacy"))
            73
        """
        if ignore_case:
            query_text = query.text.lower()
            reference_text = reference.text.lower()
        else:
            query_text = query.text
            reference_text = reference.text
        return round(self._fuzzy_funcs.get(fuzzy_func)(query_text, reference_text))
