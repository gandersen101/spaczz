"""Module for matchers."""
from ._phrasematcher import _PhraseMatcher
from .fuzzymatcher import FuzzyMatcher
from .regexmatcher import RegexMatcher
from .similaritymatcher import SimilarityMatcher

__all__ = ["_PhraseMatcher", "FuzzyMatcher", "RegexMatcher", "SimilarityMatcher"]
