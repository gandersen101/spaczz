from collections import defaultdict
from spacy.vocab import Vocab
from ..process import map_chars_to_tokens


class RegexMatcher:
    name = "regex_matcher"

    def __init__(self, vocab: Vocab, **defaults):
        self.vocab = vocab
        self.regex_patterns = defaultdict(lambda: defaultdict(list))
        self.defaults = defaults
        self._callbacks = {}
