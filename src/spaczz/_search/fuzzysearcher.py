"""`FuzzySearcher` searches for phrase-based fuzzy matches in spaCy `Doc` objects."""
import typing as ty

from spacy.vocab import Vocab

from .phrasesearcher import PhraseSearcher
from ..customtypes import TextContainer
from ..registry.fuzzyfuncs import get_fuzzy_func


class FuzzySearcher(PhraseSearcher):
    """Class for phrase-based fuzzy match searching in spaCy `Doc` objects."""

    def __init__(self: "FuzzySearcher", vocab: Vocab) -> None:
        """Initializes the searcher."""
        super().__init__(vocab=vocab)

    def compare(
        self: "FuzzySearcher",
        s1: TextContainer,
        s2: TextContainer,
        *,
        ignore_case: bool = True,
        min_r: int = 0,
        fuzzy_func: str = "simple",
        **kwargs: ty.Any,
    ) -> int:
        """Peforms fuzzy matching between two spaCy container objects.

        spaCy containers are `Doc`, `Span`, or `Token` objects.

        Applies the given fuzzy matching algorithm (`fuzzy_func`)
        to two spaCy containers and returns the resulting fuzzy ratio.

        Args:
            s1: First spaCy container for comparison.
            s2: Second spaCy container for comparison.
            ignore_case: Whether to lower-case `s1` and `s2`
                before comparison or not. Default is `True`.
            min_r: Minimum ratio needed to match as a value between `0` and `100`.
                For ratio < `min_r`, `0` is returned instead.
                Default is `0`, which deactivates this behaviour.
            fuzzy_func: Key name of fuzzy matching function to use.
                Default is `"simple"`.
            **kwargs: Overflow for kwargs from parent class.

        Returns:
            The fuzzy ratio between `s1` and `s2`.

        Example:
            >>> import spacy
            >>> from spaczz._search import FuzzySearcher
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
        return round(get_fuzzy_func(fuzzy_func)(s1_text, s2_text, score_cutoff=min_r))
