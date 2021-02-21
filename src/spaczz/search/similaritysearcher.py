"""Module for SimilaritySearcher: vector similarity matching in spaCy `Doc` objects."""
from __future__ import annotations

from typing import Any, Union
import warnings

from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab

from . import _PhraseSearcher
from ..exceptions import MissingVectorsWarning


class SimilaritySearcher(_PhraseSearcher):
    """Class for similarity matching in spacy `Doc` objects.

    Similarity matching is done on the token level.
    The class provides methods for finding the best similarity match
    span in a `Doc`, the n best similarity matched spans in a `Doc`,
    and similarity matching between any two given spaCy containers
    (`Doc`, `Span`, `Token`).

    Similarity matching uses spaCy word vectors if available,
    therefore spaCy vocabs without word vectors may not produce
    useful results. The spaCy medium and large English models provide
    work vectors that will work for this purpose.

    Searching in/with spaCy Docs that do not have vector values
    will always return a similarity score of 0.

    Warnings from spaCy about the above two scenarios are suppressed
    for convenience. However, spaczz will still warn about the former.

    Attributes:
        vocab (Vocab): The shared vocabulary.
            Included for consistency and potential future-state.
    """

    def __init__(self: SimilaritySearcher, vocab: Vocab) -> None:
        """Initializes a similarity searcher.

        Args:
            vocab: A spaCy `Vocab` object.
                Purely for consistency between spaCy
                and spaczz matcher APIs for now.
                spaczz matchers are mostly pure-Python
                currently and do not share vocabulary
                with spaCy pipelines.

        Warnings:
            MissingVectorsWarning:
                If vocab does not contain any word vectors.
        """
        super().__init__(vocab=vocab)
        if vocab.vectors.n_keys == 0:
            warnings.warn(
                """The spaCy Vocab object has no word vectors.
                Similarity results may not be useful.""",
                MissingVectorsWarning,
            )

    def compare(
        self: SimilaritySearcher,
        query: Union[Doc, Span, Token],
        reference: Union[Doc, Span, Token],
        *args: Any,
        **kwargs: Any
    ) -> int:
        """Peforms similarity matching between two spaCy container objects.

        spaCy containers are `Doc`, `Span`, or `Token` objects.

        Args:
            query: First container for comparison.
            reference: Second container for comparison.
            *args: Overflow for child positional arguments.
            **kwargs: Overflow for child keyword arguments.


        Returns:
            The similarity score between `query` and `reference` as an `int`.

        Example:
            >>> import spacy
            >>> from spaczz.search import SimilaritySearcher
            >>> nlp = spacy.load("en_core_web_md")
            >>> searcher = SimilaritySearcher(nlp.vocab)
            >>> doc1 = nlp("I like apples.")
            >>> doc2 = nlp("I like grapes.")
            >>> searcher.compare(doc1, doc2)
            94
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            if query.vector_norm and reference.vector_norm:
                return round(query.similarity(reference) * 100)
            else:
                return 0
