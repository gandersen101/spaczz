"""Module for matchers."""
from .fuzzymatcher import FuzzyMatcher
from .regexmatcher import RegexMatcher
from .similaritymatcher import SimilarityMatcher
from .tokenmatcher import TokenMatcher

__all__ = [
    "FuzzyMatcher",
    "RegexMatcher",
    "SimilarityMatcher",
    "TokenMatcher",
]
