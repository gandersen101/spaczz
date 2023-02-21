"""Module for FuzzySearcher: fuzzy matching in spaCy `Doc` objects."""
import typing as ty

from spacy.vocab import Vocab

from ._phrasesearcher import _PhraseSearcher
from ..customtypes import TextContainer
from ..registry import fuzzy_funcs


class FuzzySearcher(_PhraseSearcher):
    """Class for fuzzy matching in spacy `Doc` objects.

    Fuzzy matching is done on the token level.
    The class provides methods for finding the best fuzzy match
    span in a `Doc`, the n best fuzzy matched spans in a `Doc`,
    and fuzzy matching between any two given `SpacyContainer`s
    (`Doc`, `Span`, `Token`).

    Fuzzy matching is currently provided by rapidfuzz and the searcher
    provides access to all rapidfuzz matchers with default settings.

    Attributes:
        vocab (Vocab): The shared vocabulary.
            Included for consistency and potential future-state.
    """

    def __init__(self: "FuzzySearcher", vocab: Vocab) -> None:
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

    def compare(
        self: "FuzzySearcher",
        s1: TextContainer,
        s2: TextContainer,
        ignore_case: bool = True,
        min_r: int = 0,
        fuzzy_func: str = "simple",
        **kwargs: ty.Any
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
            min_r: Minimum ratio needed to match as a value between `0` and `100`.
                For ratio < min_r, `0` is returned instead.
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
            **kwargs: Overflow for kwargs from parent class.

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
        return round(fuzzy_funcs.get(fuzzy_func)(s1_text, s2_text, score_cutoff=min_r))
