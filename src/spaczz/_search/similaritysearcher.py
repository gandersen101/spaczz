"""`SimilaritySearcher` searches for phrase-based sim matches in spaCy `Doc` objects."""
import typing as ty
import warnings

from spacy.vocab import Vocab

from .phrasesearcher import PhraseSearcher
from ..customtypes import TextContainer
from ..exceptions import MissingVectorsWarning


class SimilaritySearcher(PhraseSearcher):
    """Class for phrase-based similarity match searching in spaCy `Doc` objects."""

    def __init__(self: "SimilaritySearcher", vocab: Vocab) -> None:
        """Initializes a the searcher."""
        super().__init__(vocab=vocab)
        if vocab.vectors.n_keys == 0:
            warnings.warn(
                """The spaCy `Vocab` object has no word vectors.
                Similarity results may not be useful.""",
                MissingVectorsWarning,
                stacklevel=2,
            )

    def compare(
        self: "SimilaritySearcher",
        s1: TextContainer,
        s2: TextContainer,
        **kwargs: ty.Any,
    ) -> int:
        """Peforms similarity matching between two spaCy container objects.

        spaCy containers are `Doc`, `Span`, or `Token` objects.

        Args:
            s1: First spaCy container for comparison.
            s2: Second spaCy container for comparison.
            **kwargs: Overflow for kwargs from parent class.

        Returns:
            The similarity score between `s1` and `s2`.

        Example:
            >>> import spacy
            >>> from spaczz._search import SimilaritySearcher
            >>> nlp = spacy.load("en_core_web_md")
            >>> searcher = SimilaritySearcher(nlp.vocab)
            >>> doc1 = nlp("I like apples.")
            >>> doc2 = nlp("I like grapes.")
            >>> searcher.compare(doc1, doc2) > 0
            True
        """
        return round(s1.similarity(s2) * 100)
