"""Module for SimilarityMatcher with an API semi-analogous to spaCy's PhraseMatcher."""
import typing as ty

from spacy.vocab import Vocab

from . import _PhraseMatcher
from ..customtypes import MatchType
from ..search import SimilaritySearcher


class SimilarityMatcher(_PhraseMatcher):
    """spaCy-like matcher for finding vector similarity matches in `Doc` objects.

    Similarity matches added patterns against the `Doc` object it is called on.
    Accepts labeled patterns in the form of `Doc` objects.

    Attributes:
        defaults: Keyword arguments to be used as default matching settings.
            See `SimilaritySearcher` documentation for details.
        name: Class attribute - the name of the matcher.
        type: The kind of matcher object.
        _callbacks:
            On match functions to modify `Doc` objects passed to the matcher.
            Can make use of the matches identified.
        _patterns:
            Patterns added to the matcher. Contains patterns
            and kwargs that should be passed to matching function
            for each labels added.
    """

    name = "similarity_matcher"

    def __init__(
        self: "SimilarityMatcher", vocab: Vocab, **defaults: ty.Dict[str, ty.Any]
    ) -> None:
        """Initializes the similarity matcher with the given defaults.

        Args:
            vocab: A spacy `Vocab` object.
                Purely for consistency between spaCy
                and spaczz matcher APIs for now.
                spaczz matchers are currently pure
                Python and do not share vocabulary
                with spacy pipelines.
            **defaults: Keyword arguments that will
                be used as default matching settings.
                These arguments will become the new defaults for matching.
                See `SimilaritySearcher` documentation for details.
        """
        super().__init__(vocab=vocab, **defaults)
        self.type: MatchType = "similarity"
        self._searcher = self._get_searcher(vocab)

    @staticmethod
    def _get_searcher(vocab: Vocab) -> SimilaritySearcher:
        """Placeholder."""
        return SimilaritySearcher(vocab)
