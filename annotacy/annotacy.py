import re
import heapq
import itertools
import string
import operator
from operator import itemgetter, truth
from functools import partial
import spacy
from spacy.tokens import Span
from fuzzywuzzy import fuzz


def map_chars_to_tokens(doc):
    chars_to_tokens = {}
    for token in doc:
        for i in range(token.idx, token.idx + len(token.text)):
            chars_to_tokens[i] = token.i
    return chars_to_tokens


class RegexMatcher:
    name = "regex_matcher"

    def __init__(self, patterns):
        self.patterns = patterns

    def __call__(self, doc):
        chars_to_tokens = map_chars_to_tokens(doc)
        for label, pattern in self.patterns.items():
            for match in re.finditer(pattern, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end, label=label)
                if span:
                    doc = self.update_entities(doc, span)
                else:
                    start_token = chars_to_tokens.get(start)
                    end_token = chars_to_tokens.get(end)
                    if start_token and end_token:
                        span = Span(doc, start_token, end_token + 1, label=label)
                        doc = self.update_entities(doc, span)
        return doc

    @staticmethod
    def update_entities(doc, span):
        try:
            doc.ents += (span,)
        except ValueError:
            pass
        return doc


class FuzzySearch:
    def __init__(
        self,
        nlp,
        fuzzy_alg=fuzz.ratio,
        min_ratio=70,
        case_sensitive=False,
        left_ignores=("space", "punct", "stop"),
        right_ignores=("space", "punct", "stop"),
    ):
        self.nlp = nlp
        self.fuzzy_alg = fuzzy_alg
        self.min_ratio = min_ratio
        self.case_sensitive = case_sensitive
        self.left_ignores = left_ignores
        self.right_ignores = right_ignores
        self.left_ignore_rules = {
            "space": lambda x: truth(x.is_space),
            "punct": lambda x: truth(x.is_punct),
            "stop": lambda x: truth(x.is_stop),
        }
        self.right_ignore_rules = {
            "space": lambda x: truth(x.is_space),
            "punct": lambda x: truth(x.is_punct),
            "stop": lambda x: truth(x.is_stop),
        }

    def best_match(self, doc, query, step="default", flex="default", verbose=False):
        pass
