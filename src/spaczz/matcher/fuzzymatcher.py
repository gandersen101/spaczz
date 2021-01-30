"""Module for FuzzyMatcher with an API semi-analogous to spaCy's PhraseMatcher."""
from __future__ import annotations

from typing import Any

from spacy.vocab import Vocab

from . import _PhraseMatcher
from ..search.fuzzysearcher import FuzzySearcher


class FuzzyMatcher(_PhraseMatcher):
    """spaCy-like matcher for finding fuzzy matches in Doc objects.

    Fuzzy matches added patterns against the `Doc` object it is called on.
    Accepts labeled patterns in the form of `Doc` objects.

    Attributes:
        defaults: Keyword arguments to be used as default matching settings.
            See `FuzzySearcher` documentation for details.
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

    name = "fuzzy_matcher"

    def __init__(self: FuzzyMatcher, vocab: Vocab, **defaults: Any) -> None:
        """Initializes the fuzzy matcher with the given defaults.

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
                See `FuzzySearcher` documentation for details.
        """
        super().__init__(vocab=vocab, **defaults)
        self.type = "fuzzy"
        self._searcher = FuzzySearcher(vocab=vocab)
