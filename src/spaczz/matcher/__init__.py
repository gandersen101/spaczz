"""Module for matchers."""
from .fuzzymatcher import FuzzyMatcher
from .regexmatcher import RegexMatcher
from .similaritymatcher import SimilarityMatcher

__all__ = ["FuzzyMatcher", "RegexMatcher", "SimilarityMatcher"]
