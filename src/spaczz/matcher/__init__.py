"""Module for matchers."""
from ._phrasematcher import _PhraseMatcher
from .fuzzymatcher import FuzzyMatcher
from .regexmatcher import RegexMatcher
from .similaritymatcher import SimilarityMatcher
from .spaczzmatcher import SpaczzMatcher

__all__ = [
    "_PhraseMatcher",
    "FuzzyMatcher",
    "RegexMatcher",
    "SimilarityMatcher",
    "SpaczzMatcher",
]
