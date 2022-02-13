"""Module for FuzzySearcher: fuzzy matching in spaCy `Doc` objects."""
from __future__ import annotations

from typing import Any, Union

from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab

from . import _PhraseSearcher
from .._fuzz import FuzzyFuncs


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
        s1: Union[Doc, Span, Token],
        s2: Union[Doc, Span, Token],
        ignore_case: bool = True,
        score_cutoff: int = 0,
        fuzzy_func: str = "simple",
        *args: Any,
        **kwargs: Any,
    ) -> int:
        """Peforms fuzzy matching between two spaCy container objects.

        Applies the given fuzzy matching algorithm (`fuzzy_func`)
        to two spacy containers (`Doc`, `Span`, `Token`)
        and returns the resulting fuzzy ratio.

        Args:
            s1: First spaCy container for comparison.
            s2: Second spaCy container for comparison.
            ignore_case: Whether to lower-case `s1` and `s2`
                before comparison or not. Default is `True`.
            score_cutoff: Score threshold as a float between `0` and `100`.
                For ratio < score_cutoff, `0` is returned instead.
                Default is `0`, which deactivates this behaviour.
            fuzzy_func: Key name of fuzzy matching function to use.
                All rapidfuzz matching functions with default settings
                are available:
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
                Default is `"simple"`.
            *args: Overflow for child positional arguments.
            **kwargs: Overflow for child keyword arguments.

        Returns:
            The fuzzy ratio between `s1` and `s2` as an `int`.

        Example:
            >>> import spacy
            >>> from spaczz.search import FuzzySearcher
            >>> nlp = spacy.blank("en")
            >>> searcher = FuzzySearcher(nlp.vocab)
            >>> searcher.compare(nlp("spaczz"), nlp("spacy"))
            73
        """
        if ignore_case:
            s1_text = s1.text.lower()
            s2_text = s2.text.lower()
        else:
            s1_text = s1.text
            s2_text = s2.text
        return round(
            self._fuzzy_funcs.get(fuzzy_func)(
                s1_text, s2_text, score_cutoff=score_cutoff
            )
        )
